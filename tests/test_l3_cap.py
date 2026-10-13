"""fix-3 t_5bcb33f2 · L3 单段 cap + 细节池可配置（P2 预算竞争根治）单测。

根因（验收报告 L3 裁定行，reviewer 钉正）：choose_forms 无单段 cap + P0 distill 豁免
+ 42 段 L3 抢 4K 池 → 单段可独吞细节池。本卡加单段 L3 cap（fail-safe 800 tok）+
细节池 detail_budget 可配置（默认 4096 不变，cap 未命中时零行为变更）。

cap 命中路径：单段 > cap 的 L3 候选 → 渲染 form = L2（有 digest）或 整段降级（不装 L3，
走 digest-L2 或 L0 指针，无 digest 兜底）；整段装入行为不再发生（无截断动作）。短段（≤ cap）
无扰动：form 与 cap 前逐同。
structural P0 豁免 cap（活账本永不裁，push cap 语义）。

红线：零 store/ledger/.db 写、零 gateway restart；改动限 sc/inject.py cap 逻辑 +
sc/__init__.py config 注入面；宿主 agent/ 零 diff；不推远端。

跑法（repo 根）：python -m pytest tests/test_l3_cap.py -v
"""
from __future__ import annotations

from helpers import mkseg, mk_window
from sc.inject import choose_forms, block_tokens, DEFAULT_L3_MAX_TOKENS
from sc.schema import est_tokens


def _overcap_seg(store, digest_text=None):
    """单段 > 800 tok（4009 char → est_tokens ~1003）的 P1 候选；可选带 digest。"""
    seg = mkseg(store, "big dump " + "b" * 4000, "P1")
    if digest_text is not None:
        seg.digest_text = digest_text
    return seg


class TestOvercapDrops:
    def test_overcap_with_digest_drops_to_l2(self, store):
        """单段 > cap 且带 digest → form = L2（不再整段 L3 装细节池）。"""
        seg = _overcap_seg(store, digest_text="short digest summary")
        assert est_tokens("big dump " + "b" * 4000) > DEFAULT_L3_MAX_TOKENS, "前置：本段确超 cap"
        w = mk_window(100000, [seg])
        by_id = {ws.segment.segment_id: ff for ws, ff, _ in choose_forms(w.segs, w)}
        assert by_id[seg.segment_id] == "L2", "超 cap + digest → L2（细节池免单段独吞）"

    def test_overcap_no_digest_falls_to_l1_or_l0(self, store):
        """单段 > cap 且无 digest → 截断 + L1/L0 指针兜底（绝不整段 L3）。"""
        seg = _overcap_seg(store)
        w = mk_window(100000, [seg])
        by_id = {ws.segment.segment_id: ff for ws, ff, _ in choose_forms(w.segs, w)}
        assert by_id[seg.segment_id] in ("L1", "L0"), "超 cap + 无 digest → L1/L0 指针兜底"
        assert by_id[seg.segment_id] != "L3", "整段装入行为不再发生"

    def test_short_segment_no_perturbation(self, store):
        """单段 ≤ cap（~23 tok）→ form 与 cap 前逐同（仍 L3，短段无扰动）。"""
        seg = mkseg(store, "small fact " + "a" * 80, "P0")
        assert est_tokens(seg.original_text) <= DEFAULT_L3_MAX_TOKENS, "前置：短段 ≤ cap"
        w = mk_window(100000, [seg])
        by_id = {ws.segment.segment_id: ff for ws, ff, _ in choose_forms(w.segs, w)}
        assert by_id[seg.segment_id] == "L3", "≤ cap 短段仍 L3（无扰动，回归不变）"

    def test_structural_p0_exempt_from_cap(self, store):
        """structural P0（活账本）超 cap 也永不裁 → 仍 L3（push cap 语义一致）。"""
        goal = mkseg(store, "goal: " + "g" * 4000, "P0", structural=True)
        assert est_tokens(goal.original_text) > DEFAULT_L3_MAX_TOKENS, "前置：structural 段超 cap"
        w = mk_window(100000, [goal])
        goal_form = [ff for ws, ff, _ in choose_forms(w.segs, w) if ws.segment is goal][0]
        assert goal_form == "L3", "structural P0 豁免 cap，永不降级"


class TestCapConfigSurface:
    def test_default_cap_is_800(self, store):
        """l3_max_tokens=None → cap = 800（fail-safe 保守默认）；放大则放行。"""
        seg = mkseg(store, "cap " + "d" * 3600, "P1")   # ~901 tok > 800
        assert est_tokens(seg.original_text) > 800
        seg.digest_text = "short digest"
        w = mk_window(100000, [seg])
        f_default = {ws.segment.segment_id: ff for ws, ff, _ in choose_forms(w.segs, w)}[seg.segment_id]
        f_wide = {ws.segment.segment_id: ff for ws, ff, _ in
                  choose_forms(w.segs, w, l3_max_tokens=2000)}[seg.segment_id]
        assert f_default == "L2", "默认 cap=800：901 tok 段拦下 → L2"
        assert f_wide == "L3", "cap 放大到 2000：901 tok 段放行 → L3"

    def test_tight_cap_forces_lower(self, store):
        """l3_max_tokens 收紧 → 更高档候选被拦（cap 是天花板，可配置）。"""
        seg = mkseg(store, "cap " + "d" * 3600, "P1")   # ~901 tok
        seg.digest_text = "short digest"
        w = mk_window(100000, [seg])
        f_tight = {ws.segment.segment_id: ff for ws, ff, _ in
                   choose_forms(w.segs, w, l3_max_tokens=500)}[seg.segment_id]
        f_loose = {ws.segment.segment_id: ff for ws, ff, _ in
                   choose_forms(w.segs, w, l3_max_tokens=1000)}[seg.segment_id]
        assert f_tight == "L2", "cap=500 < 901 → 拦下 L2"
        assert f_loose == "L3", "cap=1000 > 901 → 放行 L3"

    def test_detail_budget_still_overridable(self, store):
        """细节池 detail_budget 可配置（item 2，默认 4096）：cap 未命中时池行为不变。"""
        seg = mkseg(store, "mid " + "c" * 400, "P1")     # ~101 tok ≤ cap，装细节池
        w = mk_window(100000, [seg])
        # 池收紧到 100：101 tok 段装不下细节池（池语义，非 cap）→ 降 L1/L0
        f_tight_pool = {ws.segment.segment_id: ff for ws, ff, _ in
                        choose_forms(w.segs, w, detail_budget=100)}[seg.segment_id]
        f_default_pool = {ws.segment.segment_id: ff for ws, ff, _ in
                          choose_forms(w.segs, w)}[seg.segment_id]
        assert f_default_pool == "L3", "默认池 4096：101 tok 段 L3（cap 未命中，行为不变）"
        assert f_tight_pool in ("L1", "L0"), "池收紧到 100：装不下 → L1/L0（细节池可配置）"


class TestFortyTwoSegDistribution:
    def test_no_single_segment_monopolizes_pool(self, store):
        """根因场景 42 段 L3 抢 4K 池：cap 命中路径下无单段独吞细节池。

        cap-on（默认 800）：42 段全 > 800 → 无 L3，全降 L1/L0。
        cap-off（l3_max_tokens 放大）：前若干段进细节池 L3 → 证明 cap 才是拦截源。
        """
        segs = [mkseg(store, "seg%d " % i + "x" * 4000, "P1") for i in range(42)]
        w = mk_window(100000, segs)
        r_on = choose_forms(w.segs, w)                     # 默认 cap=800
        r_off = choose_forms(w.segs, w, l3_max_tokens=10_000_000)  # cap 实质关闭（对照）
        forms_on = {ff for _, ff, _ in r_on}
        forms_off = {ff for _, ff, _ in r_off}
        assert "L3" not in forms_on, "cap 命中：42 段全超 800 → 无单段 L3 独吞细节池"
        assert "L3" in forms_off, "cap 关闭对照：细节池可装下若干 L3（拦截源确为 cap）"
        assert block_tokens(r_on) <= 2048 + 4096 + 256, "cap-on 块有界（索引+细节帽）"
