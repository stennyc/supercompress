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

import logging
from dataclasses import dataclass, field

import scdata

from .aux_provider import marker_enabled
from .budgets import push_cap
from .schema import est_tokens
from .store import StoreError

logger = logging.getLogger(__name__)


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

    def __init__(self, window: Window, store, aux, inject_fault_after: str | None = None,
                 force_evict: bool = False, force_ratio: float = 0.75) -> None:
        self.window = window
        self.store = store
        self.aux = aux
        self.inject_fault_after = inject_fault_after
        self.force_evict = force_evict
        self.force_ratio = force_ratio    # R3 逃生门压力线（默认 0.75 = engine force_compact_ratio 同源）
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
        except (StoreError, OSError) as e:
            # R2：full-disk 时 P0 append 抛 OSError（EIO/ENOSPC）——store 写路径已归一为
            # StoreError("io_error: ...")，此处双 catch 防御测试直接 monkeypatch 注入裸
            # OSError。fail-safe 口径不变：源写失败 = 全量不压缩 + 零锚点发射，
            # 水位 WARNING 留痕（剩余空间 + 触发动作，供 ops 监控）
            self.events.append(str(e))
            result.status = "compress_failed"
            try:
                store.log_water_warn("p0-append-failed", e)
            except Exception:
                pass
            return result                 # 锚点不发射、窗口一个字节不动（fail-safe）

        # 2. 出 turn 即时 digest 生成（P0 非结构 pending）：aux 瞬态失败 → transient regime
        if not self.force_evict:
            for ws in w.segs:
                seg = ws.segment
                if (seg.tier == "P0" and not seg.structural_p0
                        and seg.digest_ref == "pending" and not aux.in_turn):
                    aux.ensure_digest(seg)

        # 3. transient_fail：单事件 skip，P1/P2 停止逐出（over-store 可接受），结构地板保 user-msg P0
        if aux.transient and not self.force_evict:
            self.events.append("transient_skip")
            store.kpi_log.append({"regime": "transient_fail"})
            over_cap = w.used_tokens > cap   # 逃生门逐出前的压力快照（status 口径不变）
            if self._p0_pending_escape_allowed(w):
                # R3 逃生门（pending 兜底）：端点瞬态 + 存在 pending 非结构 P0 + 窗口超
                # force_ratio 压力 → P0-only head80 占位摘要逐出（P1/P2 停止逐出语义保留、
                # 结构 P0 仍永不逐出）——宁可占位摘要落地，不让窗口无限 over-store。
                # status/KPI 保持保守 transient 口径（引擎侧 force_evict 兜底接更强逐出，
                # pending digest 经幂等三元组下事件再生）；双事件留痕（观测面 grep）。
                self.events.append("p0_pending_escape")
                self._evict_until(w, water, aux, result, tiers=("P0",))
            result.status = "backpressure" if over_cap else "ok"
            result.kept_p0_ids = self._kept_p0_ids(w)
            return result

        store.kpi_log.append({"regime": "normal"})

        # 4. 80% 软水位：P2 → P1 逐出至水位（P3 梯级：90% 水线不逐 P0/AC——P0 只走
        #    ① push cap（P0 常驻集 ≤15%）② 100% force_evict 兜底 两出口；本档 P0 豁免）
        self._evict_until(w, water, aux, result, tiers=("P2", "P1"))

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

        # 7. 正常收尾：P1/P2 lazy 批 flush（append 内 size trigger 累积 ≥cap 已即时触发；
        #    此处 = turn 末兜底 flush 点，之后才发射锚点），之后才发射锚点。
        # R2：full-disk 时 flush 抛 StoreError("io_error: ...") / OSError——distill 是异步面，
        # 本就允许丢一个 pending 批（断电丢批上界不变）；不上抛、不炸宿主：
        # 水位 WARNING 留痕 + pending 保留在内存（store.flush 失败语义：批未清、可重试），
        # 锚点发射经 store.durable 过滤天然不含 pending 段（宁可不压，不可断锚）。
        try:
            store.flush()
        except (StoreError, OSError) as e:
            self.events.append("flush_failed:" + str(e))
            try:
                store.log_water_warn("distill-flush-failed", e)
            except Exception:
                pass
            logger.warning("supercompress: distill final flush degraded (lossy, "
                           "pending kept for retry): %s", e)
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
                # force_evict 模式：跳过 digest 检查，直接逐出
                if self.force_evict:
                    return ws
                # 正常模式：检查 digest
                if tier == "P0" and not seg.digest_ready:
                    if aux.in_turn:
                        continue                    # turn 内 digest 未 ready → P0 不可逐出（gate）
                    aux.ensure_digest(seg)          # 出 turn 即时生成
                    if not seg.digest_ready:
                        # R3 pending 兜底（逃生门）：端点持续劣化 + digest 全 pending 且窗口
                        # 超 force_ratio 压力（默认 0.75）→ 允许 head80 占位摘要兜底逐出
                        # （digest_text 预填原文头部 80 字符——比 "[no digest]" marker 多保留
                        # P0 事实头部；token 成本 20 tok 封顶）——宁可占位摘要落地，不让窗口
                        # 无限 over-store。结构 P0 上方已 continue（逃生门不豁免结构地板）；
                        # 未劣化 / 未超压 / 正常路径（digest ready / turn 内）0 行为变化
                        # （回归钉 test_r3 / test_block1 transient）。
                        if self._p0_pending_escape_allowed(w):
                            if "p0_pending_escape" not in self.events:
                                self.events.append("p0_pending_escape")
                            if not seg.digest_text:
                                seg.digest_text = seg.original_text[:80]   # head80 占位摘要
                            return ws
                        continue                    # 生成失败（transient）且逃生门未开 → P0 不可逐出
                elif tier in ("P1", "P2") and seg.digest_ref == "pending" and not aux.in_turn:
                    aux.ensure_digest(seg)          # 出 turn 顺带生成 digest；失败 → digest_only 走 marker
                return ws
        return None

    def _evict_until(self, w: Window, water: int, aux, result: DistillResult,
                     tiers: tuple = ("P2", "P1", "P0")) -> None:
        while w.used_tokens > water:
            cand = self._next_candidate(w, aux, tiers=tiers)
            if cand is None:
                break                               # 无段可逐（结构 P0 钉死 / P0 digest 未 ready）
            self._evict(cand, result)

    def _evict(self, ws: WindowSeg, result: DistillResult) -> None:
        ws.form = "digest_only"
        result.dropped.append(ws.segment)

    def _p0_pending_escape_allowed(self, w: Window) -> bool:
        """R3 逃生门谓词（pending 兜底）：端点持续劣化 + digest 全 pending + 窗口超压。

        条件（全部满足才开逃生门，缺一不可）：
        ① 非 force_evict 模式（force_evict 自身即最强逐出口径，逃生门零冗余）；
        ② 端点持续劣化（aux.endpoint_degraded：真实端点调用连败 ≥ 3）——单一瞬态
           事件（连败 1~2）保持既有 transient_skip 单事件自愈口径 0 行为变化
           （回归钉 test_block1::test_transient_fail_single_event_self_heal）；
        ③ 窗口 used > capacity × force_ratio（默认 0.75 = engine force_compact_ratio
           同源口径；water=0.8×cap，故「超软水位」⊆「超逃生线」——正常逐出能解决的
           场景逃生门不介入，只在 (force_ratio×cap, water] 窄带 / 结构 P0 钉死顶穿时兜底）；
        ④ 窗口内存在 pending 且非结构的 P0 段（逃生门只兜 pending P0；结构地板
           goal/ledger/用户指令永不豁免——它们上方已 continue）。

        逃生门开后，逐出走 head80 占位摘要（digest_text 预填原文头部 80 字符——
        比 "[no digest]" marker 多保留一点 P0 事实头部，token 成本仍 20 tok 封顶；
        pending digest 经幂等三元组下事件再生）——宁可占位摘要落地，不让窗口
        无限 over-store（audit 报告#3：internal aux endpoint ~14-21% 超时 → P0 digest 全卡
        pending → 窗口慢性 over-store + P0 细节只留索引；本逃生门把「无限 over-store」
        封顶在占位摘要粒度）。
        """
        if self.force_evict:
            return False
        # ② 端点持续劣化门（R3 核心）：真实端点调用连败 ≥ 3 才放行；单一瞬态事件
        # 不触发（默认 aux 无该读面时 getattr 缺省 False = 逃生门关，fail-closed）。
        if not getattr(self.aux, "endpoint_degraded", False):
            return False
        if w.used_tokens <= int(w.capacity_tokens * self.force_ratio):
            return False
        return any(ws.segment.tier == "P0" and ws.segment.digest_ref == "pending"
                   and not ws.segment.structural_p0
                   for ws in w.segs)

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
