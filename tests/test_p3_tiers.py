"""P3 梯级水位（supercompress-precision-plan.md §1 水位表 + §2 P3）—— @orchestrator 落地。

三档（30% 零成本剔除档已于 2026-10-04 撤销——用户拍板 B，活体 P1=0 → fill_share 恒 0.0，
该档永远 no-op；本测试面改钉「撤档后 P1/P2 不被强制锁 L0」负断言）：
- 55–80% 主 compact：engine.DISTILL_WATER_FRAC=0.55（触发挪动，压缩时真摘要）。
- 90% evict 非 P0：sc/distill step-4 tiers=("P2","P1")（P0/AC 豁免，P0 只走 push cap / 100% force）。
- 100% force evict P0：distill force_evict（最后兜底，跳过 digest gate）。

历史锚（防再犯）：P3 落地期 8 个 @coder worker 全死在「把 fill_prune 往 Distiller.run() 塞」
= 误落驱逐层；该档属注入池自剪（sc/inject.py choose_forms），不在 sc/distill.py——撤档后
distill.py 仍一字未动（驱逐端三档水位常量原样）。

跑法（repo 根）：
  python -m pytest tests/test_p3_tiers.py -v
"""
from __future__ import annotations

import tempfile

from helpers import mkseg, mk_window
from sc.distill import WindowSeg
from sc.inject import choose_forms, render_block, block_tokens
from sc.telemetry import filler_form_share


# ─────────────────────────── 30% 零成本剔除档已撤：sc.inject 层负断言面 ───────────────────────────
class TestFillPrune_InjectLayer:
    """P3 30% 档（fill_prune 锁 L0）已撤：choose_forms 不再接受 fill_prune，
    非结构 P1/P2 回到常规选档（L3/L2/L1/L0 按预算/高信号排），不被强制锁 L0。
    G4 确定性断言保留（撤档后分支更少，同输入同输出更纯）。"""

    def _pool(self, store):
        goal = mkseg(store, "goal: ship v1", "P0", "user", structural=True)
        p1 = mkseg(store, "p1 filler " + "x" * 400, "P1")
        p2 = mkseg(store, "p2 filler " + "y" * 400, "P2")
        p0 = mkseg(store, "p0 fact " + "z" * 400, "P0")          # 非结构 P0 高信号
        segs = [WindowSeg(s) for s in (goal, p1, p2, p0)]
        return segs, goal, p1, p2, p0

    def test_no_fill_prune_param_p1_p2_not_forced_to_l0(self, store):
        """撤档面：choose_forms 无 fill_prune 参数（signature 钉死）；P1/P2 常规选档，
        低压大池下 P1/P2 进 L3 全文（非被锁 L0）——撤档前 30% 档会把它们钉在 L0。"""
        import inspect
        assert "fill_prune" not in inspect.signature(choose_forms).parameters, \
            "fill_prune 参数已撤（30% 档 no-op 撤销）"
        segs, goal, p1, p2, p0 = self._pool(store)
        w = mk_window(100000, [goal, p1, p2, p0])
        forms = {ws.segment.segment_id: f for ws, f, _ in choose_forms(segs, w)}
        assert forms[goal.segment_id] == "L3", "structural P0 豁免（活账本永不裁）"
        assert forms[p1.segment_id] == "L3", "P1 回到常规选档（大池进 L3 全文，不被强制锁 L0）"
        assert forms[p2.segment_id] == "L3", "P2 回到常规选档（大池进 L3 全文，不被强制锁 L0）"
        assert forms[p0.segment_id] in ("L3", "L2", "L1"), \
            f"非结构 P0 高信号常规选档（实际 {forms[p0.segment_id]}）"

    def test_p1_p2_low_budget_still_degrades_gracefully(self, store):
        """撤档后 P1/P2 预算榨干时仍按常规路径逐档降级（L2/L1/L0），非 30% 档专属锁 L0。"""
        segs, goal, p1, p2, p0 = self._pool(store)
        w = mk_window(100000, [goal, p1, p2, p0])
        r = choose_forms(segs, w, index_budget=0, detail_budget=0)
        forms = {ws.segment.segment_id: f for ws, f, _ in r}
        assert forms[p1.segment_id] == "L0" and forms[p2.segment_id] == "L0", \
            "池归零时 P1/P2 常规降级到 L0 指针（预算语义，非 30% 档强制）"
        block = render_block(r)
        assert p1.segment_id in block and p2.segment_id in block, \
            "L0 指针带 segment_id（O(1) 点查可回原文）"

    def test_chosen_forms_default_and_pinned(self, store):
        """撤档后常规选档面逐字不变（钉死面零扰动——G4 确定性 + 回归护栏）。"""
        segs, goal, p1, p2, p0 = self._pool(store)
        w = mk_window(100000, [goal, p1, p2, p0])
        assert choose_forms(segs, w) == choose_forms(segs, w), "常规选档默认面逐字"

    def test_deterministic_same_input_same_output(self, store):
        """G4：同 segs/window/query 两次全等（纯字面排序 + 纯函数成本；撤档后分支更少更纯）。"""
        segs, goal, p1, p2, p0 = self._pool(store)
        w = mk_window(100000, [goal, p1, p2, p0])
        r1 = choose_forms(segs, w, query="ship v1")
        r2 = choose_forms(segs, w, query="ship v1")
        assert r1 == r2, "确定性（同输入两次全等）"


# ─────────────────────────── 30% 门禁已撤：engine.compress 层负断言面 ───────────────────────────
class TestFillPrune_EngineGate:
    """P3 30% 两条件门禁（fill_share 密度 ∧ 窗口压力水线）已撤：engine.compress 不再
    激活 fill_prune——高压下 P1 全文也保留（撤档 = 30% 档从未在活体生效，回归 fill_prune
    引入前逐字行为）。filler_form_share 仅作 telemetry 观测面保留（纯审计、G4）。"""

    def _mk(self, cap):
        from engine import SupercompressContextEngine
        tmp = tempfile.mkdtemp()
        e = SupercompressContextEngine(context_length=cap, external_store_path=tmp)
        e.on_session_start("p3_gate")
        return e

    def test_high_pressure_no_longer_prunes_p1(self):
        """撤档负断言：cap=10000，4×P1(~1005 tok)=4020 → 压力 0.402（撤档前 30% 档会剪 P1→L0）；
        撤档后高压也保留 P1 全文（inject 路径 = fill_prune 引入前逐字行为）。"""
        e = self._mk(10000)
        for i in range(4):
            mkseg(e.store, f"p1 filler {i} MARKFILL{i} " + "x" * 4000, "P1")
        e.store.flush()   # P1 pending → durable，才进 _window_fresh 回放窗口（否则空块=假绿）
        # S1 尾投递面（Plan C Phase 1）：staged 块并入末条 user 消息（不再独立 role=system）
        out = e.compress([{"role": "user", "content": "继续"}])
        block = "\n".join(str(m.get("content", "")) for m in out
                          if "[supercompress] staged" in str(m.get("content", "")))
        assert "staged P0/P1 segments" in block, "前置：P1 真在注入池（非空块）"
        assert "MARKFILL" in block, "撤档：高压不再剪 P1（30% 档已撤，全文保留）"

    def test_gate_is_audit_only_from_p2_telemetry(self):
        """gate_re_add_30pct 撤档后仅作 telemetry 记录面（无下游消费者）：
        全 P1 池 fill_share>0.40 → gate True（纯观测，不再驱动任何注入行为）。"""
        tmp = tempfile.mkdtemp()
        from sc.store import JsonlStore
        store = JsonlStore(tmp + "/s.jsonl")
        p1s = [mkseg(store, "p1 f " + "a" * 400, "P1") for _ in range(3)]
        w = mk_window(100000, p1s)
        fs = filler_form_share(w, segs=p1s)
        assert fs["gate_re_add_30pct"] is True, "全 P1 池 fill_share>0.40 → 密度门禁开（纯记录面）"
        # 对照：全 P0 高信号池 → fill_share 低 → 门禁关
        p0s = [mkseg(store, "p0 fact " + "b" * 400, "P0") for _ in range(3)]
        w2 = mk_window(100000, p0s)
        fs2 = filler_form_share(w2, segs=p0s)
        assert fs2["gate_re_add_30pct"] is False, "全 P0 高信号池 fill_share≤0.40 → 门禁关（纯记录面）"


# ─────────────────────────── 90% evict 非 P0 / P0 只在 100% 兜底：distill 架构面 ───────────────────────────
class Test90Tier_P0Exempt:
    """90% 水线不逐 P0/AC 的架构 pin（plan §1「P0/AC 豁免，P0 留 100% force_evict」）。

    注（诚实口径）：P0 于软水线豁免本身在常态路径是**意图声明**（push cap ≤15% 比软水位 80%
    更紧，任何「大到影响窗口」的 P0 反正走 push cap）；本组测试 pin 的是**可观测架构不变量**：
    structural P0/AC 永不逐出、非结构 P0 只在 push cap / 100% force 下离开 90% 档。"""

    def test_structural_p0_and_ac_never_evicted_at_high_water(self, store, aux):
        """旧锚更新（§4.5 90% 超压强制面）：cap=100 used≈427（427%）→ 90% 超压面开。

        - goal 变量（kind=segment，user 指令 structural P0）：90% 超压时放开可逐（owner 拍板 2026-10-17）
        - led（kind=ledger，结构地板）：永不逐
        - 100% force 仍豁免 kind=ledger（活账本永不裁）
        """
        from sc.distill import Distiller
        goal = mkseg(store, "user: ship it keep /srv/x " * 12, "P0", "user", structural=True)
        led = mkseg(store, "ledger:" * 45, "P0", "user", structural=True,
                    kind="ledger", goal="ship v1", phase_id="p1")
        filler = [mkseg(store, f"p1 ctx {i} " * 40, "P1") for i in range(3)]
        w = mk_window(100, [goal, led] + filler)
        form = {ws.segment_id: ws.form for ws in w.segs}
        assert w.used_tokens >= 0.8 * w.capacity_tokens, "前置：真到 80% 软水位（否则不逐=假绿）"
        d = Distiller(w, store, aux).run()                                   # 90% 档（正常流）
        form = {ws.segment_id: ws.form for ws in w.segs}
        assert "tier90" in d.events, "90% 超压强制面触发"
        assert form[goal.segment_id] == "digest_only", "user 指令 P0（kind=segment）90% 超压放开可逐"
        assert form[led.segment_id] == "resident", "ledger(AC 结构地板) 90% 超压仍不逐"
        # 100% force 也豁免 kind=ledger（活账本链尾永不裁）
        d2 = Distiller(w, store, aux, force_evict=True).run()
        form = {ws.segment_id: ws.form for ws in w.segs}
        assert form[led.segment_id] == "resident", "ledger 尾连 100% force 都不逐（活账本永不裁）"

    def test_non_struct_p0_not_evicted_at_soft_tier_only_pushcap(self, store, aux):
        """旧锚更新（§4.5 90% 超压强制面 owner 拍板 2026-10-17）：

        80-90% 区间（软水线 tier75）P0 豁免；但 90% 超压（tier90 p0_force）
        时非结构 P0 可逐。本锚 used≈167% cap → tier90 触发 → P0 被逐。
        """
        from sc.distill import Distiller
        p0 = mkseg(store, "p0 fact " * 4, "P0")              # ~12 tok < push cap(15% of 100=15)
        p1s = [mkseg(store, f"p1 f {i} " * 30, "P1") for i in range(3)]
        w = mk_window(100, [p0] + p1s)
        d = Distiller(w, store, aux).run()                     # used 167% → tier90 p0_force 逐 P0
        p0_form = next(ws.form for ws in w.segs if ws.segment_id == p0.segment_id)
        assert "tier90" in d.events, "90% 超压面触发（used>0.9cap）"
        assert p0_form == "digest_only", \
            "非结构小 P0 在 90% 超压时被 p0_force 逐出（§4.5 新行为）"

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
