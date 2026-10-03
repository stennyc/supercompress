"""P3 梯级水位（supercompress-precision-plan.md §1 水位表 + §2 P3）—— @orchestrator 落地。

四档全齐（本卡 DoD「四档齐全才算收口」）：
- 30% 零成本剔除档（注入端·非驱逐端）：fill_prune 由 P2 门禁 + 窗口压力水线两条件驱动，
  开时非结构 P1/P2 填充段锁 L0 指针（零 LLM、不落驱逐层，segment_id 可 O(1) 回查）。
- 55–80% 主 compact：engine.DISTILL_WATER_FRAC=0.55（触发挪动，压缩时真摘要）。
- 90% evict 非 P0：sc/distill step-4 tiers=("P2","P1")（P0/AC 豁免，P0 只走 push cap / 100% force）。
- 100% force evict P0：distill force_evict（最后兜底，跳过 digest gate）。

⚠️ 8 个 @coder worker 全死在「把 fill_prune 往 Distiller.run() 塞」= 误落驱逐层；
本档属注入池自剪（sc/inject.py choose_forms），不在 sc/distill.py。本测试面即钉死该边界。

跑法（repo 根）：
  python -m pytest tests/test_p3_tiers.py -v
"""
from __future__ import annotations

import tempfile

from helpers import mkseg, mk_window
from sc.distill import WindowSeg
from sc.inject import choose_forms, render_block, block_tokens
from sc.telemetry import filler_form_share


# ─────────────────────────── 30% 零成本剔除档：sc.inject.choose_forms(fill_prune) 层 ───────────────────────────
class TestFillPrune_InjectLayer:
    """fill_prune 是真实 mutation gate（True/False 结果不同）——8 worker 死点的正确落层。"""

    def _pool(self, store):
        goal = mkseg(store, "goal: ship v1", "P0", "user", structural=True)
        p1 = mkseg(store, "p1 filler " + "x" * 400, "P1")
        p2 = mkseg(store, "p2 filler " + "y" * 400, "P2")
        p0 = mkseg(store, "p0 fact " + "z" * 400, "P0")          # 非结构 P0 高信号
        segs = [WindowSeg(s) for s in (goal, p1, p2, p0)]
        return segs, goal, p1, p2, p0

    def test_fill_prune_true_locks_p1_p2_to_l0(self, store):
        segs, goal, p1, p2, p0 = self._pool(store)
        w = mk_window(100000, [goal, p1, p2, p0])
        forms = {ws.segment.segment_id: f for ws, f, _ in
                 choose_forms(segs, w, fill_prune=True)}
        assert forms[goal.segment_id] == "L3", "structural P0 豁免（活账本永不裁）"
        assert forms[p1.segment_id] == "L0", "P1 填充锁 L0 指针（零成本剔除）"
        assert forms[p2.segment_id] == "L0", "P2 填充锁 L0 指针（零成本剔除）"
        assert forms[p0.segment_id] in ("L3", "L2", "L1"), \
            f"非结构 P0 高信号不被当填充剪（实际 {forms[p0.segment_id]}）"

    def test_fill_prune_true_l0_pointer_carries_segment_id(self, store):
        segs, goal, p1, p2, p0 = self._pool(store)
        w = mk_window(100000, [goal, p1, p2, p0])
        block = render_block(choose_forms(segs, w, fill_prune=True))
        assert p1.segment_id in block, "L0 指针带 segment_id（O(1) 点查可回原文）"
        assert "on-demand" in block, "L0 指针文案"

    def test_fill_prune_false_is_default_and_pinned(self, store):
        """缺省 fill_prune=False = 旧行为逐字（钉死面零扰动——G4 确定性 + 回归护栏）。"""
        segs, goal, p1, p2, p0 = self._pool(store)
        w = mk_window(100000, [goal, p1, p2, p0])
        assert choose_forms(segs, w) == choose_forms(segs, w, fill_prune=False), \
            "缺省 == 显式 False（默认面逐字不变）"

    def test_fill_prune_deterministic_same_input_same_output(self, store):
        """G4：同 segs/window/fill_prune 两次全等（纯字面排序 + 纯函数成本）。"""
        segs, goal, p1, p2, p0 = self._pool(store)
        w = mk_window(100000, [goal, p1, p2, p0])
        r1 = choose_forms(segs, w, fill_prune=True)
        r2 = choose_forms(segs, w, fill_prune=True)
        assert r1 == r2, "fill_prune 确定性（同输入两次全等）"


# ─────────────────────────── 30% 两条件门禁：engine.compress 层（密度 ∧ 压力）───────────────────────────
class TestFillPrune_EngineGate:
    """P3 30% 档 = P2 密度门禁（fill_share>0.40）∧ 窗口压力水线（used/cap≥0.30）两条件。

    缺任一条 = fill_prune False = 旧行为逐字（低压不剪 → P1 全文保召回主力材料，
    钉 test_drift 空/旧 epoch 全文保留面不翻转的根因）。"""

    def _mk(self, cap):
        from engine import SupercompressContextEngine
        tmp = tempfile.mkdtemp()
        e = SupercompressContextEngine(context_length=cap, external_store_path=tmp)
        e.on_session_start("p3_gate")
        return e

    def test_density_open_plus_high_pressure_prunes_p1(self):
        """cap=10000，4×P1(~1005 tok)=4020 → 压力 0.402≥0.30 + 全 P1 池 fill_share 1.0>0.40 → 剪（P1→L0）。"""
        e = self._mk(10000)
        for i in range(4):
            mkseg(e.store, f"p1 filler {i} MARKFILL{i} " + "x" * 4000, "P1")
        e.store.flush()   # P1 pending → durable，才进 _window_fresh 回放窗口（否则空块=假绿）
        block = "\n".join(m["content"] for m in
                          e.compress([{"role": "user", "content": "继续"}])
                          if m["role"] == "system")
        assert "staged P0/P1 segments" in block, "前置：P1 真在注入池（非空块）"
        assert "MARKFILL" not in block, "高压(≥30%)+高密度：P1 填充被剪到 L0（全文不进块，仅指针）"

    def test_density_open_but_low_pressure_keeps_p1(self):
        """cap=100000，同样 4×P1 = 压力 0.040<0.30 → 压力条件关 → 不剪（P1 全文保召回主力材料）。"""
        e = self._mk(100000)
        for i in range(4):
            mkseg(e.store, f"p1 filler {i} MARKFILL{i} " + "x" * 4000, "P1")
        e.store.flush()
        block = "\n".join(m["content"] for m in
                          e.compress([{"role": "user", "content": "继续"}])
                          if m["role"] == "system")
        assert "MARKFILL" in block, "低压(<30%)：P1 全文保留（30% 档不激活，反失忆护栏）"

    def test_gate_is_single_source_from_p2_telemetry(self):
        """门禁单源在 telemetry：gate_re_add_30pct 逐字来自 filler_form_share（不硬编码 0.40 于壳）。"""
        goal = None
        tmp = tempfile.mkdtemp()
        from sc.store import JsonlStore
        store = JsonlStore(tmp + "/s.jsonl")
        p1s = [mkseg(store, "p1 f " + "a" * 400, "P1") for _ in range(3)]
        w = mk_window(100000, p1s)
        fs = filler_form_share(w, segs=p1s)
        assert fs["gate_re_add_30pct"] is True, "全 P1 池 fill_share>0.40 → 密度门禁开"
        # 对照：全 P0 高信号池 → fill_share 低 → 门禁关（撤 30 是默认）
        p0s = [mkseg(store, "p0 fact " + "b" * 400, "P0") for _ in range(3)]
        w2 = mk_window(100000, p0s)
        fs2 = filler_form_share(w2, segs=p0s)
        assert fs2["gate_re_add_30pct"] is False, "全 P0 高信号池 fill_share≤0.40 → 密度门禁关（撤 30 默认）"


# ─────────────────────────── 90% evict 非 P0 / P0 只在 100% 兜底：distill 架构面 ───────────────────────────
class Test90Tier_P0Exempt:
    """90% 水线不逐 P0/AC 的架构 pin（plan §1「P0/AC 豁免，P0 留 100% force_evict」）。

    注（诚实口径）：P0 于软水线豁免本身在常态路径是**意图声明**（push cap ≤15% 比软水位 80%
    更紧，任何「大到影响窗口」的 P0 反正走 push cap）；本组测试 pin 的是**可观测架构不变量**：
    structural P0/AC 永不逐出、非结构 P0 只在 push cap / 100% force 下离开 90% 档。"""

    def test_structural_p0_and_ac_never_evicted_at_high_water(self, store, aux):
        """AC（goal/ledger structural P0）在高压水位下永驻留（90% 档 + 100% force 都不逐）。"""
        from sc.distill import Distiller
        goal = mkseg(store, "user: ship it keep /srv/x " * 12, "P0", "user", structural=True)
        led = mkseg(store, "ledger:" * 45, "P0", "user", structural=True,
                    kind="ledger", goal="ship v1", phase_id="p1")
        filler = [mkseg(store, f"p1 ctx {i} " * 40, "P1") for i in range(3)]
        w = mk_window(100, [goal, led] + filler)
        form = {ws.segment_id: ws.form for ws in w.segs}
        assert w.used_tokens >= 0.8 * w.capacity_tokens, "前置：真到 80% 软水位（否则不逐=假绿）"
        Distiller(w, store, aux).run()                                   # 90% 档（正常流）
        form = {ws.segment_id: ws.form for ws in w.segs}
        assert form[goal.segment_id] == "resident", "goal(AC) 90% 档不逐"
        assert form[led.segment_id] == "resident", "ledger(AC) 90% 档不逐"
        # 100% force 也豁免 structural
        Distiller(w, store, aux, force_evict=True).run()
        form = {ws.segment_id: ws.form for ws in w.segs}
        assert form[goal.segment_id] == "resident" and form[led.segment_id] == "resident", \
            "structural P0/AC 连 100% force 都不逐（活账本永不裁，push cap 语义）"

    def test_non_struct_p0_not_evicted_at_soft_tier_only_pushcap(self, store, aux):
        """非结构 P0 < push cap(15%) → 90% 软水线不逐（P1 先逐满）；这是 P0「只在 push cap/100% 兜底」的可观测面。"""
        from sc.distill import Distiller
        p0 = mkseg(store, "p0 fact " * 4, "P0")              # ~12 tok < push cap(15% of 100=15)
        p1s = [mkseg(store, f"p1 f {i} " * 30, "P1") for i in range(3)]
        w = mk_window(100, [p0] + p1s)
        Distiller(w, store, aux).run()                         # 90% 档：P1 逐满至水位
        p0_form = next(ws.form for ws in w.segs if ws.segment_id == p0.segment_id)
        assert p0_form == "resident", \
            "非结构小 P0（<push cap）在 90% 软水线不逐（软水线 tiers=P2/P1，P0 豁免）"

    def test_force_100pct_does_demote_non_struct_p0_when_over_pushcap(self, store, aux):
        """对照：非结构 P0 大（>push cap）→ 100% force 兜底才逐（P0 出场的唯一正档）。"""
        from sc.distill import Distiller
        from helpers import complete_digest
        p0 = mkseg(store, "p0 big fact " * 200, "P0")         # ~1000 tok ≫ push cap(15)
        complete_digest(aux, p0)                               # digest ready
        w = mk_window(100, [p0])
        Distiller(w, store, aux, force_evict=True).run()
        p0_form = next(ws.form for ws in w.segs if ws.segment_id == p0.segment_id)
        assert p0_form == "digest_only", \
            "100% force 兜底：大 P0 被逐出（P0 唯一逐出正档，与 90% 档豁免对照成立）"
