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
                 force_evict: bool = False, force_ratio: float = 0.75,
                 ledger_evict_cb=None) -> None:
        self.window = window
        self.store = store
        self.aux = aux
        self.inject_fault_after = inject_fault_after
        self.force_evict = force_evict
        self.force_ratio = force_ratio    # R3 逃生门压力线（默认 0.75 = engine force_compact_ratio 同源）
        # v6 §2.1：存根挂逐出事件闭包（engine 注入，形参 = 被逐出段 segment；
        # 缺省 None = 原逐出语义零变化——既有 Distiller 调用面/测试面不受影响）
        self.ledger_evict_cb = ledger_evict_cb
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

        # 4. S1 降档水位递进带（owner 拍板 2026-10-17 · 从 90% 往 55% 一档一档压）：
        #    每档"压到目标下就停、压不到 pass 给下一档"；估值 gate 分母=当前用量
        #    （每档进档前重算）；90/75/65 档确定性算（P0 逐 + 存根折省），55% 是 LLM
        #    出块面、不做执行前估值（兜底档，全 pass 不落地靠降档本身兜死、无死循环）。
        # 80-90% 区间 P0 面零变化（owner 红线③：P0 只在 90% 档逐，80-90% 区间 P0 行为
        # 与现状逐出链一致；P2/P1 软水线由 tier75 替代原 80% 软水线——水位线 80%→75%
        # 是 owner 定的降档带语义，非回归）。
        _cap = cap
        # 降档逐出面：90%（P0 超压折存根）→ 75%（P2/P1 常规逐）→ 65%（compact 兜底 pass-through）
        # → 55%（清垃圾 pass-through；LLM 出块面 S2 接、本切片 S1 只落 90/75 两确定性档 + gate）
        # 格式: (water_frac, gate_frac_of_used, tiers, label, p0_force, no_gate)
        # no_gate=True → 该档跳过估值 gate（90% 纯容量档，不走 gate-pass）
        _tiers_by_water = [
            (0.90, 0.00, ("P0",),       "tier90", True,  True),   # 90% 超压：纯容量 P0 逐出（no_gate）
            (0.75, 0.20, ("P2","P1"),   "tier75", False, False),  # 75%：软水线常规逐（替代 80% 软水线；P0 只在 90% 档）
        ]
        _executed = False
        for _wf, _gate, _tiers, _label, _p0_force, _no_gate in _tiers_by_water:
            _water_line = int(_wf * _cap)
            if w.used_tokens <= _water_line:
                break                       # 没超这条线 = 后面更低水线档也不用看（降档序：从高压往下压）
            # 估值 gate：分母=当前用量，进档前重算（no_gate 档跳过——纯容量面不走 gate）
            if not _no_gate:
                _est = self._reclaim_est(w, _tiers)
                if _est < _gate * w.used_tokens:
                    self.events.append(_label + ":gate-pass")     # 省不够 → pass 给下一档（更低压、更强逐出）
                    continue
            self.events.append(_label)
            if _p0_force:
                # 90% 超压：纯容量 P0 逐出（绕 digest 闸 + 放开 user 指令 structural P0）
                self._evict_until(w, _water_line, aux, result, tiers=_tiers,
                                   p0_force=True, ledger_evict_cb=self.ledger_evict_cb)
            else:
                self._evict_until(w, _water_line, aux, result, tiers=_tiers,
                                  ledger_evict_cb=self.ledger_evict_cb)
            _executed = True
            if w.used_tokens <= _water_line:
                break                       # 压到目标下 = 停（降档序核心：每档压够就停）

        # push cap 兜底（15%，既有 ③ 面保留）：若上面降档带没把 P0 常驻集压到 cap，
        # 逐 P0 常驻集 ≤15%（goal/structural 地板永不裁——_next_candidate 结构门保证）。
        if not _executed:
            while self._p0_resident_tokens(w) > self.push_cap_tokens:
                cand = self._next_candidate(w, aux, tiers=("P0",))
                if cand is None:
                    break
                self._evict(cand, result, self.ledger_evict_cb)

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
    def _next_candidate(self, w: Window, aux, tiers: tuple[str, ...],
                        p0_force: bool = False) -> WindowSeg | None:
        for tier in tiers:
            cands = sorted((ws for ws in w.segs
                            if ws.form == "resident" and ws.segment.tier == tier),
                           key=lambda ws: ws.segment.seq)
            for ws in cands:
                seg = ws.segment
                if seg.structural_p0:
                    # §4.5 90% 超压强制面（owner 拍板 2026-10-17）：放开 user 指令
                    # structural P0（kind="segment"，classify_user 标）；goal/ledger 尾 P0
                    # （kind=goal/ledger，结构地板）仍永不逐
                    if p0_force and tier == "P0" and seg.kind not in ("goal", "ledger"):
                        pass
                    else:
                        continue                        # 结构 P0（goal / ledger 尾 / 用户指令）永不逐出
                # force_evict 模式：跳过 digest 检查，直接逐出
                if self.force_evict:
                    return ws
                # 正常模式：检查 digest
                if tier == "P0" and not seg.digest_ready:
                    if p0_force:
                        # §4.5 90% 超压强制面：绕 digest 闸——digest 未 ready 也允许逐出
                        # （head80 占位摘要，同逃生门 L241 口径：digest_text 预填原文头 80 字符，
                        # pending digest 经幂等三元组下事件再生）——纯超压驱动，不依赖 aux 健康
                        if not seg.digest_text:
                            seg.digest_text = seg.original_text[:80]
                        return ws
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
                     tiers: tuple = ("P2", "P1", "P0"),
                     ledger_evict_cb=None, p0_force: bool = False) -> None:
        while w.used_tokens > water:
            cand = self._next_candidate(w, aux, tiers=tiers, p0_force=p0_force)
            if cand is None:
                break                               # 无段可逐（结构 P0 钉死 / P0 digest 未 ready）
            self._evict(cand, result, ledger_evict_cb)

    def _evict(self, ws: WindowSeg, result: DistillResult,
               ledger_evict_cb=None) -> None:
        """逐出 + v6 §2.1 存根挂逐出事件（唯一新增环节）。

        `ledger_evict_cb`（engine 注入的闭包，形参 = 被逐出段 segment）缺省
        None = 原逐出语义零变化（既有 `self._evict(cand, result)` 调用面不受影响）。
        逐出段原文已 durable（`store.durable` fsync+读回锚点，与 `_emit_anchors`
        同源过滤，owner 选 a 零新增沉盘）→ 闭包在活账本对应页折单行存根
        （`evict_page`，`evict_stub_line` 渲染，指向 durable 锚点 segment_id）；
        非账本段 / 非 durable 段 = 逐出照旧、不折存根（T1/T2 存根面只覆盖已在
        活账本里的被逐出段；召回段 source=recall 恒不进账本 → 恒不折存根，
        T4 防递归）。`evict_page` 全 fail-open（找不到 seq / IO 失败 → 静默）。"""
        ws.form = "digest_only"
        result.dropped.append(ws.segment)
        if ledger_evict_cb is None:
            return
        if ws.segment.content_hash in getattr(self.store, "durable", set()):
            ledger_evict_cb(ws.segment)

    # ───────────── S1 降档水位递进带（owner 拍板 2026-10-17 · 估值 gate 分母=当前用量）────
    def _reclaim_est(self, w: Window, tiers: tuple[str, ...]) -> int:
        """该档动作预估回收量（代码层确定性算、零 LLM）：Σ(resident 段 est_tokens(原文)
        − est_tokens(digest 摘录)) over 可逐 tiers 段。

        估值 gate 面（owner 拍"分母=当前用量 w.used_tokens"）：
          某档"预估省 < 阈值×used"→ 不执行该档、pass 给下一档（省 LLM/无效逐出开销）。
          阈值常量在 run() 降档分档里用（本方法只出分子、阈值判定在调用面）。
        - P0/P1/P2 段：est_tokens(original_text) − est_tokens(digest_text or "[no digest]")
          （digest ready → 省≈原文全量；pending → 省≈原文−head80 占位 ~20 tok）。
        - 结构 P0（goal/ledger 尾）不在可逐集（_next_candidate 永不逐）→ 不计回收（估 0）。
        纯函数、幂等、零写盘、零 LLM、零宿主——fail-open（估算异常 → 0，不炸宿主 turn）。"""
        try:
            total = 0
            for ws in w.segs:
                seg = ws.segment
                if seg.tier not in tiers or ws.form != "resident":
                    continue
                if seg.structural_p0 and seg.kind in ("goal", "ledger"):
                    continue                # 结构地板永不逐（估 0）
                _full = est_tokens(seg.original_text)
                _stub = est_tokens(seg.digest_text or "[no digest]")
                total += max(0, _full - _stub)
            return total
        except Exception:
            return 0

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
