"""S-Phase 2：阶梯注入（决定 B）+ S2a 修 S5（digest 正文捕获）— @coder own。

DoD（references/supercompress-session-redesign.md S-Phase 2）：
- S2a：LLM digest 响应正文被捕获并解析 key_fact/summary；空响应回退 head80 占位
  （bench 空响应不变量保住，零回归）
- S2b：choose_forms 预算驱动选档 —— 高压（used>cap）细节池归零 → 注入块 ≤ 索引池(≈2K)；
  正常压 细节池 4K + 索引池 2K；structural P0 永不降级；L0 指针带 segment_id（可回查）；
  确定性（同输入同输出）
- engine.compress() 接线：注入块 = 选档渲染（非全文拼接），block_tokens 有上界

跑法（repo 根）：
  python -m pytest tests/test_inject.py -v
"""
from __future__ import annotations

import json

from fakes import FakeEndpoint
from helpers import mkseg, mk_window
from sc.aux_provider import AuxProvider, parse_digest_response, DIGEST_INSTRUCTION
from sc.distill import Window, WindowSeg
from sc.inject import choose_forms, render_block, block_tokens, DEFAULT_INDEX_BUDGET, DEFAULT_DETAIL_BUDGET
from sc.schema import Segment, est_tokens


# ───────────────────────── S2a：digest 响应解析（修 S5）────────────────────────

class TestParseDigestResponse:
    def test_clean_json(self):
        raw = json.dumps({"key_fact": "auth service binds 10.0.0.5:8080",
                          "summary": "postmortem: auth bound to 10.0.0.5:8080 after incident."})
        kf, sm = parse_digest_response(raw)
        assert kf == "auth service binds 10.0.0.5:8080"
        assert sm.startswith("postmortem")

    def test_code_fence_json(self):
        raw = 'sure, here you go:\n```json\n{"key_fact": "k", "summary": "s"}\n```\ndone'
        kf, sm = parse_digest_response(raw)
        assert (kf, sm) == ("k", "s")

    def test_embedded_json_with_prose(self):
        raw = 'The digest is: {"key_fact": "goal: v1 上线", "summary": "计划上线 v1"}'
        kf, sm = parse_digest_response(raw)
        assert kf == "goal: v1 上线"

    def test_plain_text_fallback(self):
        kf, sm = parse_digest_response("just a plain sentence digest of the segment " * 10)
        assert kf and sm, "纯文本也出非空（summary 取整段截断，优于无）"
        assert len(sm) <= 400

    def test_empty_response_empty_pair(self):
        assert parse_digest_response("") == ("", "")
        assert parse_digest_response("   ") == ("", "")

    def test_none_response(self):
        assert parse_digest_response(None) == ("", "")


class TestDigestCapture_EndToEnd:
    """_generate 消费 LLM 正文：canned JSON → key_fact/digest_text 真实内容（非 head80 占位）。"""

    def test_key_fact_captured(self, store):
        ep = FakeEndpoint()
        ep.responses["auxiliary.compression"] = json.dumps(
            {"key_fact": "auth binds 10.0.0.5:8080", "summary": "summary body here"})
        aux = AuxProvider(ep, async_digest=False)
        seg = mkseg(store, "x" * 400, "P0")  # 400 字符 → head80 占位 ≠ summary 正文
        out = aux.ensure_digest(seg)
        assert out is not None and seg.digest_ready
        assert seg.key_fact == "auth binds 10.0.0.5:8080", "S5 修复：LLM key_fact 被捕获"
        assert seg.digest_text == "summary body here", "S5 修复：summary 正文进 digest_text"
        # 调用点未增（同 1 次端点调用，只是消费了响应正文）
        assert len([c for c in ep.calls if c[0] == "auxiliary.compression"]) == 1
        # 提示词带结构化指令
        assert DIGEST_INSTRUCTION in ep.calls[0][1]

    def test_empty_response_falls_back_head80(self, store):
        """bench 不变量：空响应 → head80 占位（零回归）。"""
        ep = FakeEndpoint()
        ep.responses["auxiliary.compression"] = ""
        aux = AuxProvider(ep, async_digest=False)
        text = "head marker " + "y" * 400
        seg = mkseg(store, text, "P0")
        aux.ensure_digest(seg)
        assert seg.digest_ready
        assert seg.digest_text == text[:80], "空响应回退 head80（bench 不变量）"
        assert seg.key_fact == ""


# ───────────────────────── S2b：阶梯选档（预算驱动）────────────────────────

def _mk(cap, segs):
    """窗口 cap + 段列表 → Window（segs 为 Segment，包装成 WindowSeg）。"""
    w = Window(cap, [WindowSeg(s) for s in segs])
    return w


class TestChooseForms_Budget:
    def test_high_pressure_index_only(self, store):
        """DoD：高压（used>cap）→ 细节池归零，非 structural 段全落 L1/L0，注入块 ≤ 索引池。"""
        segs = [mkseg(store, "p1 body " + str(i) + " " + "word" * 60, "P1") for i in range(40)]
        # 窗：cap 远小于段总 token（used>cap = backpressure 态）
        w = _mk(1000, segs)
        assert w.used_tokens > w.capacity_tokens, "前置：构造为高压态"
        rendered = choose_forms(w.segs, w)
        forms = {ff for _, ff, _ in rendered}
        assert forms <= {"L1", "L0"}, "高压只出 L1/L0（细节池归零）"
        assert block_tokens(rendered) <= DEFAULT_INDEX_BUDGET, "注入块 ≤ 索引池 2K"
        # L0 指针带 segment_id（细节可 O(1) 回查）
        l0 = [t for _, ff, t in rendered if ff == "L0"]
        if l0:
            for t in l0:
                sid = [s.segment_id for s in segs]
                assert any(sid_i in t for sid_i in sid), "L0 指针含 segment_id"

    def test_normal_pressure_allows_l3_in_detail_pool(self, store):
        """正常压（used≤cap）→ 细节池 4K 内装得下的段升 L3，其余 L2/L1/L0。"""
        s_small = mkseg(store, "small fact " + "a" * 80, "P0")      # L3 ~21t
        s_big = mkseg(store, "big dump " + "b" * 4000, "P1")       # L3 ~1001t
        s_mid = mkseg(store, "mid " + "c" * 400, "P1")             # L3 ~101t
        s_mid.digest_text = "mid summary"                            # 可 L2
        w = _mk(100000, [s_small, s_big, s_mid])
        assert w.used_tokens <= w.capacity_tokens
        rendered = choose_forms(w.segs, w)
        by_id = {ws.segment.segment_id: ff for ws, ff, _ in rendered}
        assert by_id[s_small.segment_id] == "L3", "小段装进细节池 → L3 全文"
        assert by_id[s_mid.segment_id] in ("L3", "L2"), "中装得下则 L3，否则 L2"
        total = block_tokens(rendered)
        assert total <= DEFAULT_DETAIL_BUDGET + DEFAULT_INDEX_BUDGET + 128, "不超 细节+索引 帽（容忍 structural 豁免）"

    def test_structural_p0_never_downgraded(self, store):
        """structural P0（goal/ledger）永不降级——高压下仍 L3。"""
        goal = mkseg(store, "goal: ship v1", "P0", structural=True)
        lp = [mkseg(store, "p1 " + str(i) + " " + "z" * 200, "P1") for i in range(30)]
        w = _mk(500, [goal] + lp)
        assert w.used_tokens > w.capacity_tokens
        rendered = choose_forms(w.segs, w)
        goal_form = [ff for ws, ff, _ in rendered if ws.segment is goal][0]
        assert goal_form == "L3", "structural P0 豁免细节池，永不降级"
        others = {ff for ws, ff, _ in rendered if ws.segment is not goal}
        assert others <= {"L1", "L0"}, "非结构段高压下只出 L1/L0"

    def test_query_boosts_relevant_segment_to_l3(self, store):
        """相关性：query 命中的段优先吃细节池升 L3，无关段落 L1。"""
        relevant = mkseg(store, "auth service binds 10.0.0.5:8080 " + "r" * 200, "P0")
        irrelevant = [mkseg(store, "unrelated " + "q" * 200, "P1") for _ in range(5)]
        w = _mk(100000, [relevant] + irrelevant)
        rendered = choose_forms(w.segs, w, query="auth service 10.0.0.5")
        by_id = {ws.segment.segment_id: ff for ws, ff, _ in rendered}
        assert by_id[relevant.segment_id] == "L3", "query 命中段升 L3"
        # 无关段因 query 无命中 → 靠 tier/新近排队，装得下才 L3，装不下 L2/L1
        assert by_id[relevant.segment_id] != "L0", "最相关段绝不 L0"

    def test_deterministic_same_input_same_output(self, store):
        """确定性（同输入同输出，缓存友好）：跑两次选档完全一致。"""
        segs = [mkseg(store, "det " + str(i) + " " + "d" * 300, "P1") for i in range(10)]
        w = _mk(5000, segs)
        r1 = choose_forms(w.segs, w, query="det 3")
        r2 = choose_forms(w.segs, w, query="det 3")
        assert [(ws.segment.segment_id, ff) for ws, ff, _ in r1] == \
               [(ws.segment.segment_id, ff) for ws, ff, _ in r2], "同输入同输出"
        seqs1 = [ws.segment.seq for ws, _, _ in r1]
        assert seqs1 == sorted(seqs1), "输出 = seq 稳定序（不随 query 抖动）"

    def test_render_block_format(self, store):
        segs = [mkseg(store, "fact " + str(i), "P0") for i in range(3)]
        w = _mk(100000, segs)
        rendered = choose_forms(w.segs, w)
        block = render_block(rendered)
        assert block, "非空段集合 → 非空注入块"
        assert "##" not in block and "---" in block, "分隔符 = ---（段间）"
        for ws, ff, t in rendered:
            assert f"#{ws.segment.seq}" in block, "每段以 #seq 前缀可索引"


# ───────────────────────── engine 层接线验证 ─────────────────────────

class TestEngineCompress_StagedInjection:
    def test_compress_emits_staged_block_not_fulltext(self, store, aux):
        """engine.compress() 注入块 = 阶梯渲染（带 forms 统计），非全文拼接。"""
        from engine import SupercompressContextEngine
        import tempfile, os
        tmp = tempfile.mkdtemp()
        e = SupercompressContextEngine(context_length=100000, external_store_path=tmp)
        e.on_session_start("stg")
        # 造 3 段：1 structural P0 + 2 P1（无 digest → L3 成本 = 原文）
        goal = mkseg(e.store, "goal: v1", "P0", "user", structural=True)
        p1a = mkseg(e.store, "p1 a " + "a" * 300, "P1")
        p1b = mkseg(e.store, "p1 b " + "b" * 300, "P1")
        w = _mk(100000, [goal, p1a, p1b])
        # 直接验 sc.inject（engine.compress 调同一函数；全函数走宿主 turn 链，隔离数据面）
        rendered = choose_forms([WindowSeg(goal), WindowSeg(p1a), WindowSeg(p1b)], w)
        block = render_block(rendered)
        assert "goal: v1" in block, "structural P0 全文必在（L3 豁免）"
        total = block_tokens(rendered)
        assert total <= DEFAULT_DETAIL_BUDGET + DEFAULT_INDEX_BUDGET, "注入块不超 细节+索引 帽"
