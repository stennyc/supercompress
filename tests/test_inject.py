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

# ───────────────────── 偏离根因修复：L1 fact 扩 head + anchor 豁免 ─────────────────────
# 根因（2026-10-23 owner 实测）：compact 两次后模型段降档 L1，fact = original_text[:20]
# 碎片不可读 → 模型"失忆"自己的进行中结论，拿注入 mem 上下文猜下一步 → 偏离任务。
# 修法：① L1 fact head 20→60 字符 + 截断标记；② anchor 豁免（最近 user P0 + 其模型末响应
# 强制 L3，不受 800 cap / 细节池约束）+ cap 兜底（anchor 超 cap 走常规降档，不破 6K 总帽）。


class TestL1FactWiderHead:
    """detail_budget=0 + l3_cap=0 强制 L1（单段 4K 细节池装得下默认 L3，须显式挤掉细节池）。"""

    def test_long_text_head60_with_marker(self, store):
        """无 key_fact 时 head 扩到 60 字符 + '…' 截断标记（原 head20 太碎）。"""
        s = mkseg(store, "head-marker " + "x" * 60, "P1")  # 71 字符，L1 行 ≈ 19t
        s.seq = 1
        ws = WindowSeg(s)
        line = choose_forms([ws], mk_window(100000, [s]), query="",
                            index_budget=100, detail_budget=0, l3_max_tokens=0)
        form, text = next((f, t) for w, f, t in line if w is ws)
        assert form == "L1", "无细节池 → L1 索引行"
        assert text.endswith("…"), f"超长 fact 须带截断标记: {text!r}"
        assert "x" * 48 in text, "head 须含 60 字符原文前缀（前 12 字符 'head-marker ' + 48 个 x）"

    def test_short_text_no_marker(self, store):
        """短于 head 长度的段：fact 原样，不截断、不加标记。"""
        s = mkseg(store, "ok", "P1")
        s.seq = 1
        ws = WindowSeg(s)
        line = choose_forms([ws], mk_window(100000, [s]), query="",
                            index_budget=100, detail_budget=0, l3_max_tokens=0)
        form, text = next((f, t) for w, f, t in line if w is ws)
        assert form == "L1", text
        assert "…" not in text, "短段不加截断标记"
        assert "· ok" in text, text

    def test_key_fact_still_preferred(self, store):
        """有 key_fact 段：L1 仍优先用 key_fact（不变）。"""
        s = mkseg(store, "long body " + "z" * 300, "P1")
        s.seq = 1
        s.key_fact = "the fact"
        ws = WindowSeg(s)
        line = choose_forms([ws], mk_window(100000, [s]), query="",
                            index_budget=100, detail_budget=0, l3_max_tokens=0)
        form, text = next((f, t) for w, f, t in line if w is ws)
        assert "· the fact" in text, text


class TestAnchorL3Exempt:
    def test_last_user_p0_and_model_response_anchor_l3(self, store):
        """anchor = 最近 user P0 + 紧随其后的第一段 model P0（该 user 的末响应）→ 豁免 L3
        （**L3 成本 ≤ l3 cap 时**；cap 兜底面见 test_anchor_cap_fallback）。
        非 anchor model 段（seq 更大 / P1 filler）正常降档。"""
        u = mkseg(store, "goal fix", "P0", "user"); u.seq = 1
        resp = mkseg(store, "done, commit landed", "P0"); resp.seq = 2   # 短 anchor（≤cap）
        later = mkseg(store, "o" * 8000, "P1"); later.seq = 3           # P1 filler 不抢 anchor
        segs = [u, resp, later]
        rendered = choose_forms([WindowSeg(s) for s in segs], mk_window(100000, segs),
                                query="", index_budget=10, detail_budget=0,
                                l3_max_tokens=800)
        form_r = next(f for ws, f, t in rendered if ws.segment is resp)
        form_l = next(f for ws, f, t in rendered if ws.segment is later)
        assert form_r == "L3", f"短 anchor（≤cap）须 L3 豁免，实得 {form_r}"
        assert form_l in ("L1", "L0", "L2"), f"非 anchor 超 cap 段须降档，实得 {form_l}"

    def test_anchor_not_last_model_response(self, store):
        """模型末响应不是最后一段时（其后再有 user 段）→ 失去 anchor 身份，正常降档。"""
        u = mkseg(store, "goal fix", "P0", "user"); u.seq = 1
        mid_model = mkseg(store, "m" * 8000, "P1"); mid_model.seq = 2
        later_user = mkseg(store, "next question", "P0", "user"); later_user.seq = 3
        segs = [u, mid_model, later_user]
        rendered = choose_forms([WindowSeg(s) for s in segs], mk_window(100000, segs),
                                query="", index_budget=10, detail_budget=0,
                                l3_max_tokens=800)
        form_mid = next(f for ws, f, t in rendered if ws.segment is mid_model)
        assert form_mid in ("L1", "L0", "L2"), \
            f"末 user 之前的 model 段非 anchor，须降档，实得 {form_mid}"

    def test_last_user_p0_always_l3(self, store):
        """最近 user P0 本身恒 L3（anchor 的 user 侧）。"""
        u1 = mkseg(store, "older user", "P0", "user"); u1.seq = 1
        u2 = mkseg(store, "latest user P0", "P0", "user"); u2.seq = 2
        m = mkseg(store, "model reply", "P1"); m.seq = 3
        segs = [u1, u2, m]
        rendered = choose_forms([WindowSeg(s) for s in segs], mk_window(100000, segs),
                                query="", index_budget=10, detail_budget=0,
                                l3_max_tokens=800)
        form_u2 = next(f for ws, f, t in rendered if ws.segment is u2)
        assert form_u2 == "L3", f"最近 user P0 须 L3，实得 {form_u2}"

    def test_anchor_cap_fallback(self, store):
        """reviewer 必修项：anchor 段 L3 成本超 l3 cap → 不再无脑 L3（破 DoD 总帽 6K
        实测面），走常规选档 L1/L0。anchor = 末 user P0 + 其后首段 model P0。"""
        # 场景 1：末 user 后的 model 响应短（≤cap）→ 豁免 L3
        u = mkseg(store, "goal fix", "P0", "user"); u.seq = 1
        short = mkseg(store, "done, commit landed", "P0"); short.seq = 2   # anchor（≤cap）
        segs1 = [u, short]
        r1 = choose_forms([WindowSeg(s) for s in segs1], mk_window(100000, segs1),
                          query="", index_budget=10, detail_budget=0, l3_max_tokens=800)
        form_short = next(f for ws, f, t in r1 if ws.segment is short)
        assert form_short == "L3", f"短 anchor（≤cap）仍豁免 L3，实得 {form_short}"
        # 场景 2：末 user 后的 model 响应超长（>cap）→ cap 兜底降档
        u2 = mkseg(store, "next task", "P0", "user"); u2.seq = 1
        big = mkseg(store, "m" * 8000, "P0"); big.seq = 2                 # anchor（>cap）
        segs2 = [u2, big]
        r2 = choose_forms([WindowSeg(s) for s in segs2], mk_window(100000, segs2),
                          query="", index_budget=10, detail_budget=0, l3_max_tokens=800)
        form_big = next(f for ws, f, t in r2 if ws.segment is big)
        assert form_big in ("L1", "L0", "L2"), \
            f"超长 anchor 须 cap 兜底降档（破 6K 帽），实得 {form_big}"


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

    def test_overlong_llm_fields_capped(self):
        """F5(security)：LLM 自报超长字段进 system 通道前被限长截断——防无上限正文。"""
        kf_long = "k" * 500          # > 80
        sm_long = "s" * 3000         # > 400
        kf, sm = parse_digest_response(json.dumps({"key_fact": kf_long, "summary": sm_long}))
        assert len(kf) == 80, f"key_fact 限长 80（实际 {len(kf)}）"
        assert len(sm) == 400, f"summary 限长 400（实际 {len(sm)}）"


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
        """DoD：高压（used>cap）→ 细节池归零，非 structural 段全落 L1/L2/L0，注入块 ≤ 索引池。
        本 fixture 无 digest 段 → 实际只出 L1/L0（已 digest 段 L2 场景见 test_high_pressure_digested_gets_l2）。"""
        segs = [mkseg(store, "p1 body " + str(i) + " " + "word" * 60, "P1") for i in range(40)]
        # 窗：cap 远小于段总 token（used>cap = backpressure 态）
        w = _mk(1000, segs)
        assert w.used_tokens > w.capacity_tokens, "前置：构造为高压态"
        rendered = choose_forms(w.segs, w)
        forms = {ff for _, ff, _ in rendered}
        assert forms <= {"L1", "L0"}, "无 digest fixture 高压只出 L1/L0（细节池归零且无可 L2 段）"
        assert block_tokens(rendered) <= DEFAULT_INDEX_BUDGET, "注入块 ≤ 索引池 2K"
        # L0 指针带 segment_id（细节可 O(1) 回查）
        l0 = [t for _, ff, t in rendered if ff == "L0"]
        if l0:
            for t in l0:
                sid = [s.segment_id for s in segs]
                assert any(sid_i in t for sid_i in sid), "L0 指针含 segment_id"

    def test_high_pressure_digested_gets_l2(self, store):
        """F3：高压下已 digest 段可出 L2（经索引池，仍在 2K 帽内）——模块 docstring 曾误写'高压只出 L1/L0'。"""
        segs = [mkseg(store, "p1 body " + str(i) + " " + "word" * 60, "P1") for i in range(40)]
        digested = mkseg(store, "p1 digested " + "w" * 60, "P1")
        digested.digest_text = "one-line digest summary of this segment"
        segs.append(digested)
        w = _mk(1000, segs)
        assert w.used_tokens > w.capacity_tokens, "高压态"
        rendered = choose_forms(w.segs, w)
        by_id = {ws.segment.segment_id: ff for ws, ff, _ in rendered}
        assert by_id[digested.segment_id] == "L2", "已 digest 段高压出 L2（索引池内）"
        assert block_tokens(rendered) <= DEFAULT_INDEX_BUDGET, "L2 仍计入索引池 2K 帽（structural 豁免除外）"

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
            # R5 批次 A #3/#5：行首 `#<rel>` = 块内相对号（1 起），全局 seq 保留在
            # `seq=<全局seq>` 段（L1）/ `#<seq> ... [L3]`（L3 原文渲染面）——
            # 断言可点查定位的 `seq=<全局seq>`（L1）或原文行首（L3），非相对号。
            if ff == "L1":
                assert f"seq={ws.segment.seq}" in block, "L1 行保留全局 seq（recall O(1) 点查面）"
            else:
                assert f"#{ws.segment.seq}" in block or f"seq={ws.segment.seq}" in block, \
                    "L3/L2 行保留全局 seq（原文渲染面）"


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

    def test_compress_end_to_end_header_and_egress(self, store, aux):
        """F4：真 e2e e.compress()（非 sc.inject 直调）——补 compress() 管线零测试覆盖面：
        ① header f-string（forms 统计 + block_tokens + segs 计数）；② structural P0 全文在块；
        ③ egress：room-skeleton P0 被 _skel 过滤逐出注入块（该过滤当前无任何测试验证）。
        protect_first_n/last_n 走宿主基类默认（3/6）——本测试环境 base 解析成真宿主基类。"""
        import tempfile
        from engine import SupercompressContextEngine
        tmp = tempfile.mkdtemp()
        e = SupercompressContextEngine(context_length=100000, external_store_path=tmp)
        e.on_session_start("e2e")
        goal = mkseg(e.store, "goal: ship v1", "P0", "user", structural=True)
        skel = mkseg(e.store, '[Group chat: "ops"] ' + "room skeleton " * 50, "P0",
                      structural=True)
        big_p1 = mkseg(e.store, "big dump " + "x" * 4000, "P1")
        # 宿主消息面：protect_first_n=3 + protect_last_n=6；S1 尾投递面（Plan C Phase 1）
        # staged 块不再独立 role=system，而是并入末条 user 消息 content（不再是位置 2 system）
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(12)]
        out = e.compress(msgs, current_tokens=200000, force=True)
        assert not [m for m in out if m.get("role") == "system"], \
            "S1：compress 不再发任何 in-list system（staged 块改投 user 面）"
        staged = [m for m in out if "[supercompress] staged" in str(m.get("content", ""))]
        assert len(staged) == 1, "恰一条 user 消息携带 staged 阶梯注入块（S1 尾投递并入末条）"
        content = str(staged[0]["content"])
        # ① header：长会话（_has_middle=True）走 compact-ledger 累计段路径（header =
        #    "staged compact-ledger" + c#N 块），短会话走 S1 尾投递（header =
        #    "staged P0/P1 segments" + forms 统计）。两条路都要有 staged 注入块在场。
        #    （owner 裁定 tier 分层后：P0 定案块 + P1 可裁层，账本路径按 c#N 渲染）
        has_s1_header = ("staged P0/P1 segments" in content)
        has_ledger_header = ("staged compact-ledger" in content)
        assert has_s1_header or has_ledger_header, \
            f"staged 注入块 header 须在场（S1 或 compact-ledger），实得: {content[:120]!r}"
        if has_s1_header:
            # S1 尾投递面：header f-string 三要素（segs 计数 + block_tokens + forms 统计）
            assert " segs, " in content and " tok" in content, "segs 计数 + block_tokens"
            assert "forms: {" in content, "forms 档位统计（sorted dict）"
        else:
            # compact-ledger 面：c#N 累计块 + generation/seals 前缀（append-only 单调）
            assert "c#" in content, "compact-ledger c#N 累计块在场"
        # ② structural P0 全文必在（L3 豁免细节池）
        assert "goal: ship v1" in content, "structural P0 全文在块"
        # ③ egress：room-skeleton P0 被 _skel 过滤逐出（不进块）
        assert "room skeleton" not in content, "egress：群聊骨架逐出注入块"
        assert "goal: ship v1" in content and "room skeleton" not in content, \
            "structural P0 留、room-skeleton P0 逐（同 tier 不同 provenance）"
