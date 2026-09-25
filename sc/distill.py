"""Block ① distill：80% 软水位 + P0 逐出 gate + push cap + tier-aware fsync + backpressure。

设计锁（gate v4 (A)）：
- 压缩前先 durable 化 P0（write-before-compress 硬 gate；源写失败 = 全量不压缩 + 零锚点发射）；
- P0 逐出 gate：digest 未 ready 的 P0 不可逐出（保留常驻）；structural P0（goal / ledger 尾 /
  用户指令）永不逐出；
- 80% 是软目标不是硬闸：逐出至水位即可，超物理窗口 → status=backpressure（硬 backpressure
  只在物理满/钉死段顶穿时发）；
- push cap：P0 常驻集 ≤ 15% usable（cap 压力下逐出非结构事实型 P0，goal 永不裁）；
- tier-aware fsync：P0 即写即 fsync+读回；P1/P2 lazy 批（断电丢一个批，有界）；
- 锚点只发射「已 durable」的逐出段（fsync+读回后）——宁可不压，不可断锚；
- gate⑤ 故障注入 marker：`SC_INJECT_FAULT` off-by-default（生产恒 inert），on → 注入双向可控；
- transient_fail（主 LLM 瞬态失败）：单事件 tiering/digest skip、P1/P2 停止逐出（over-store
  可接受）、结构地板保 user-msg P0、KPI #1 打 regime 标签、下事件自愈。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import scdata

from .aux_provider import marker_enabled
from .budgets import push_cap
from .schema import est_tokens
from .store import StoreError


class WindowSeg:
    """窗口内一段的呈现形态：resident（原文在窗口）| digest_only（仅 digest 摘录在窗口）。"""

    def __init__(self, segment) -> None:
        self.segment = segment
        self.form = "resident"

    @property
    def segment_id(self) -> str:
        return self.segment.segment_id

    def tokens(self) -> int:
        """窗口内 token 成本：resident = 原文；digest_only = digest 摘录（无 digest 走占位 marker）。"""
        if self.form == "resident":
            return est_tokens(self.segment.original_text)
        return est_tokens(self.segment.digest_text or "[no digest]")

    def excerpt(self) -> str:
        if self.form == "resident":
            return self.segment.original_text
        return self.segment.digest_text or "[no digest]"


class Window:
    def __init__(self, capacity_tokens: int, segs: list[WindowSeg]) -> None:
        self.capacity_tokens = int(capacity_tokens)
        self.segs = segs

    @property
    def used_tokens(self) -> int:
        return sum(ws.tokens() for ws in self.segs)


class Anchor:
    """锚点：指向单一事实源 jsonl 的段（content_hash 为 replay 索引键；segment_id 行级唯一）。

    发射契约：只在段 durable（fsync+读回）后发射——pending 字节物理上无指针。
    """

    def __init__(self, segment_id: str, content_hash: str, tier: str, digest_ref: str) -> None:
        self.segment_id = segment_id
        self.content_hash = content_hash
        self.tier = tier
        self.digest_ref = digest_ref


class DistillResult:
    def __init__(self, status: str, events: list[str], fault_state: dict | None,
                 anchor_manifest: list[Anchor], dropped: list, kept_p0_ids: list[str]) -> None:
        self.status = status            # "ok" | "backpressure" | "compress_failed"
        self.events = events
        self.fault_state = fault_state # marker on 时的注入点快照（fsynced_p0 / pending_p1p2_bytes）
        self.anchor_manifest = anchor_manifest
        self.dropped = dropped
        self.kept_p0_ids = kept_p0_ids


class Distiller:
    """纯蒸馏器 + 锚点发射器：不持有状态、不取数、不直接写 jsonl（记忆层管存）。"""

    def __init__(self, window: Window, store, aux, inject_fault_after: str | None = None) -> None:
        self.window = window
        self.store = store
        self.aux = aux
        self.inject_fault_after = inject_fault_after
        self.push_cap_tokens = push_cap(window)   # Push cap 读共享分母（一处定义）
        self.events: list[str] = []

    # ─────────────────────────── 主流程 ───────────────────────────
    def run(self) -> DistillResult:
        w, store, aux = self.window, self.store, self.aux
        cap = w.capacity_tokens
        water = int(0.8 * cap)
        result = DistillResult("ok", self.events, None, [], [], [])
        aux.transient = False   # 瞬态标志按事件重置：上事件的 transient 本事件自愈（单事件 skip 语义）

        # 1. write-before-compress 硬 gate：窗口 P0 必须 durable（P0 即写即 fsync + 读回）
        try:
            for ws in w.segs:
                seg = ws.segment
                if seg.tier == "P0" and seg.content_hash not in store.durable:
                    store.append(seg)
        except StoreError as e:
            self.events.append(str(e))   # "source_unavailable" 等
            result.status = "compress_failed"
            return result                 # 锚点不发射、窗口一个字节不动（fail-safe）

        # 2. 出 turn 即时 digest 生成（P0 非结构 pending）：aux 瞬态失败 → transient regime
        for ws in w.segs:
            seg = ws.segment
            if (seg.tier == "P0" and not seg.structural_p0
                    and seg.digest_ref == "pending" and not aux.in_turn):
                aux.ensure_digest(seg)

        # 3. transient_fail：单事件 skip，P1/P2 停止逐出（over-store 可接受），结构地板保 user-msg P0
        if aux.transient:
            self.events.append("transient_skip")
            store.kpi_log.append({"regime": "transient_fail"})
            result.status = "backpressure" if w.used_tokens > cap else "ok"
            result.kept_p0_ids = self._kept_p0_ids(w)
            return result

        store.kpi_log.append({"regime": "normal"})

        # 4. 80% 软水位：P2 → P1 → P0（digest ready 且非结构）逐出至水位
        self._evict_until(w, water, aux, result)

        # 5. push cap：P0 常驻集 ≤ 15%（goal / structural P0 永不裁）
        while self._p0_resident_tokens(w) > self.push_cap_tokens:
            cand = self._next_candidate(w, aux, tiers=("P0",))
            if cand is None:
                break
            self._evict(cand, result)

        # 6. gate⑤ 故障注入点（off-by-default：生产恒 inert；SC_INJECT_FAULT=1 → 真生效）
        if self.inject_fault_after and marker_enabled():
            self.events.append("fault:" + self.inject_fault_after)
            if self.inject_fault_after == "gate_boundary":
                store.flush()                       # gate 边界：P1/P2 批已 flush durable
            elif self.inject_fault_after == "mid_p1_batch" and store._pending:
                half = scdata.to_line(store._pending[0].to_dict())[:20]   # torn 尾模拟（无 fsync）
                with open(store.path, "a", encoding="utf-8") as f:
                    f.write(half)
            result.fault_state = {
                "fsynced_p0": store.fsynced_p0_count,
                "pending_p1p2_bytes": store.pending_p1p2_bytes,
            }
            result.anchor_manifest = self._emit_anchors(result, store)
            result.kept_p0_ids = self._kept_p0_ids(w)
            return result

        # 7. 正常收尾：P1/P2 lazy 批 flush（size trigger / turn 末），之后才发射锚点
        store.flush()
        result.anchor_manifest = self._emit_anchors(result, store)
        result.status = "backpressure" if w.used_tokens > cap else "ok"
        result.kept_p0_ids = self._kept_p0_ids(w)
        return result

    # ─────────────────────────── 逐出核心 ───────────────────────────
    def _next_candidate(self, w: Window, aux, tiers: tuple[str, ...]) -> WindowSeg | None:
        for tier in tiers:
            cands = sorted((ws for ws in w.segs
                            if ws.form == "resident" and ws.segment.tier == tier),
                           key=lambda ws: ws.segment.seq)
            for ws in cands:
                seg = ws.segment
                if seg.structural_p0:
                    continue                        # 结构 P0（goal / ledger 尾 / 用户指令）永不逐出
                if tier == "P0" and not seg.digest_ready:
                    if aux.in_turn:
                        continue                    # turn 内 digest 未 ready → P0 不可逐出（gate）
                    aux.ensure_digest(seg)          # 出 turn 即时生成
                    if not seg.digest_ready:
                        continue                    # 生成失败（transient）→ P0 不可逐出
                elif tier in ("P1", "P2") and seg.digest_ref == "pending" and not aux.in_turn:
                    aux.ensure_digest(seg)          # 出 turn 顺带生成 digest；失败 → digest_only 走 marker
                return ws
        return None

    def _evict_until(self, w: Window, water: int, aux, result: DistillResult) -> None:
        while w.used_tokens > water:
            cand = self._next_candidate(w, aux, tiers=("P2", "P1", "P0"))
            if cand is None:
                break                               # 无段可逐（结构 P0 钉死 / P0 digest 未 ready）
            self._evict(cand, result)

    def _evict(self, ws: WindowSeg, result: DistillResult) -> None:
        ws.form = "digest_only"
        result.dropped.append(ws.segment)

    def _p0_resident_tokens(self, w: Window) -> int:
        return sum(ws.tokens() for ws in w.segs
                   if ws.segment.tier == "P0" and ws.form == "resident")

    def _kept_p0_ids(self, w: Window) -> list[str]:
        return [ws.segment.segment_id for ws in w.segs
                if ws.segment.tier == "P0" and ws.form == "resident"]

    def _emit_anchors(self, result: DistillResult, store) -> list[Anchor]:
        """锚点只发射已 durable（fsync+读回）的逐出段——pending 字节物理上无指针。"""
        return [Anchor(seg.segment_id, seg.content_hash, seg.tier, seg.digest_ref)
                for seg in result.dropped if seg.content_hash in store.durable]
