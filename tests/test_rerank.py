"""P3（S-Phase 4）gate LLM rerank 测试面（纯 sc 契约 + engine select_context 接线）。

gate 三选一（低分 / 显式搜索词 / 召回空）各 1 例 + 非触发 1 例；
缓存（同 query+window_version 两次 → 第 2 次 LLM 0 调用）；
fail-safe（LLM 不可用 / 解析失败 → 原样返回入参 hits，不降级不炸）；
结构化 JSON 解析（canned 排序重排 hits；多对象 / 纯文本 → fail-safe 原样）；
确定性（同输入两次 gate 判定全等 + rerank 结果全等）。

端点注入 = FakeEndpoint（测试替身住 tests/fakes.py，不进生产 import 面）；
LLM 面零硬编码模型名 / 零自定义 provider 块（用户硬约束）。
"""
import json

import pytest

from sc.schema import Segment
from sc.recall import Hit
from sc.rerank import (
    RerankGate, rerank_hits, rerank_cache_key, parse_rerank_response, RERANK_TASK,
)
from fakes import FakeEndpoint


def _mkseg(text, tier="P1", seq=0, prov="user"):
    # R3/R4 配套（2026-10-10）：默认 provenance 由 "model" 改 "user"——rerank/重排机制
    # 单测测的是 LLM 重排+缓存+fail-safe（对 user 真拍板段），不应被 R4 gate-drop-model
    # 行为短路（那会让这些机制测试恒返回空、失去覆盖）。R4 的 model-drop 行为由
    # TestRerankDropModelOnGate 专属类钉死（prov 显式 "model"）。prov 参数保留（可测 model 面）。
    return Segment(tier=tier, provenance=prov, original_text=text, seq=seq, ts=0.0)


def _hits(scores, n=3):
    return [Hit(_mkseg(f"segment-{i} " + "x" * 20, tier="P1", seq=i), score)
            for i, score in enumerate(scores)]


class _CannedEndpoint:
    """按序投喂 canned 响应（FakeEndpoint 的 list 版；calls 记录 (task, text)）。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.fail = False

    def call(self, task, text):
        self.calls.append((task, text))
        if self.fail:
            raise RuntimeError("transient_fail: endpoint rate-limited")
        return self.responses.pop(0) if self.responses else ""


# ───────────────────────── gate 三选一 + 非触发 ─────────────────────────

class TestGateTriggers:
    def test_gate_low_score_triggers(self):
        """① 低分：top-1 < 阈值（阈值可配；默认 0.0 即不触发——保守 gate）。"""
        g = RerankGate(low_score_threshold=0.5)
        assert g.should_fire(_hits([0.3, 0.1, 0.0]), "普通查询词") is True

    def test_gate_explicit_keyword_triggers(self):
        """② query 命中显式「搜一下/搜下/search」关键词 → 触发（正常高分召回也触发）。"""
        g = RerankGate()
        assert g.should_fire(_hits([5.0, 1.0, 0.5]), "数据库连接池 搜一下") is True
        assert g.should_fire(_hits([5.0, 1.0, 0.5]), "please SEARCH the config log") is True

    def test_gate_recall_empty_triggers(self):
        """③ 召回空（hits 全 0 分 / 无 hits）→ 触发。"""
        g = RerankGate()
        assert g.should_fire(_hits([0.0, 0.0, 0.0]), "任意查询") is True
        assert g.should_fire([], "任意查询") is True

    def test_gate_normal_high_score_not_trigger(self):
        """非触发：正常高分召回 + 无显式搜索词 + 默认保守阈值 → 不触发（零 LLM 调用面）。"""
        g = RerankGate()
        assert g.should_fire(_hits([5.0, 1.0, 0.5]), "数据库 连接池 超时") is False


# ───────────────────────── 缓存（命中 → 零 LLM 调用）────────────────────────

class TestRerankCache:
    def test_second_call_same_key_zero_llm(self):
        """同 query+window_version 两次 → 第 2 次 LLM 调用 0 次（FakeEndpoint.calls 长度不变）。"""
        ep = FakeEndpoint()
        hits = _hits([0.5, 0.4, 0.3])
        ids = [h.seg.segment_id for h in hits]
        ep.responses[RERANK_TASK] = json.dumps({"order": list(reversed(ids))})
        cache = {}
        r1 = rerank_hits(hits, "搜一下 部署", ep, cache=cache, window_version=7)
        n_calls = len(ep.calls)
        assert n_calls == 1, "gate 触发（显式搜索词）+ 缓存 miss → LLM 1 次"
        assert [h.seg.segment_id for h in r1] == ids[::-1], "canned 排序重排后命中"
        r2 = rerank_hits(hits, "搜一下 部署", ep, cache=cache, window_version=7)
        assert len(ep.calls) == n_calls, "同 key 第 2 次命中缓存 → 零 LLM 调用"
        assert [h.seg.segment_id for h in r2] == [h.seg.segment_id for h in r1], "缓存重放全等"
        # 缓存键纯函数：同输入全等（window_version 单调指纹进键）
        assert rerank_cache_key("搜一下 部署", 7) == rerank_cache_key("搜一下 部署", 7)
        assert rerank_cache_key("搜一下 部署", 7) != rerank_cache_key("搜一下 部署", 8)

    def test_gate_not_firing_zero_llm_even_with_cache(self):
        """非触发（正常高分 + 无搜索词 + 默认保守阈值）→ LLM 0 调用、不入缓存。"""
        ep = FakeEndpoint()
        cache = {}
        out = rerank_hits(_hits([5.0, 1.0, 0.5]), "数据库 连接池 超时", ep,
                          cache=cache, window_version=1)
        assert out is not None and len(ep.calls) == 0, "非触发 gate → 零 LLM"
        assert cache == {}, "非触发面不得写缓存"


# ───────────────────────── fail-safe（LLM 不可用 / 解析失败 → 原样返回）────────────────────────

class TestRerankFailSafe:
    def test_llm_unavailable_returns_hits_unchanged(self, store):
        """fail=True → 原样返回入参 hits（顺序不变、不 raise、不写缓存、留痕由 log 承接）。"""
        ep = FakeEndpoint()
        ep.fail = True
        cache = {}
        hits = _hits([5.0, 1.0, 0.5])
        out = rerank_hits(hits, "搜一下 部署", ep, cache=cache, window_version=3)
        assert [h.seg.segment_id for h in out] == [h.seg.segment_id for h in hits], "顺序不变"
        assert out is not None and all(isinstance(h, Hit) for h in out), "不炸：返回 Hit 列表"
        assert len(ep.calls) == 1, "gate 触发 + 缓存 miss 才发起 LLM（随后 fail）"
        assert cache == {}, "LLM 失败面不写缓存（下次 gate 再触发可重试）"

    def test_unparseable_multi_object_failsafe(self):
        """LLM 回多对象文本 → 解析失败 → 原样返回（不 raise）。"""
        ep = FakeEndpoint()
        ep.responses[RERANK_TASK] = (
            '{"order": ["b","a","c"]} 一些中文说明 {"note": "second"}')
        cache = {}
        hits = _hits([0.5, 0.4, 0.3])
        out = rerank_hits(hits, "搜一下", ep, cache=cache, window_version=1)
        assert [h.seg.segment_id for h in out] == [h.seg.segment_id for h in hits]
        assert cache == {}, "解析失败不写缓存"

    def test_plain_text_failsafe(self):
        """LLM 回纯文本（非 JSON）→ fail-safe 原样。"""
        ep = FakeEndpoint()
        ep.responses[RERANK_TASK] = "I cannot rank these candidates right now"
        out = rerank_hits(_hits([0.5, 0.4, 0.3]), "搜一下", ep,
                          cache={}, window_version=1)
        assert [h.seg.segment_id for h in out] == \
               [h.seg.segment_id for h in _hits([0.5, 0.4, 0.3])], "纯文本 → 原序（同段同分）"

    def test_recall_empty_no_llm_call(self):
        """hits=[]（召回空）→ 无候选可排：直接返回（不发起 LLM，避免无意义调用）。"""
        ep = FakeEndpoint()
        out = rerank_hits([], "搜一下", ep, cache={}, window_version=1)
        assert out == [] and len(ep.calls) == 0


# ───────────────────────── 结构化 JSON 解析（canned 排序 / 容错面）────────────────────────

class TestRerankJsonParse:
    def test_canned_order_reorders_hits(self):
        """canned 排序响应 → hits 按 LLM 序重排（top-k 严格全排列校验）。"""
        hits = _hits([0.1, 0.9, 0.5])
        a, b, c = hits
        out = parse_rerank_response(json.dumps({"order": [b.seg.segment_id,
                                                          c.seg.segment_id,
                                                          a.seg.segment_id]}),
                                   [a.seg.segment_id, b.seg.segment_id, c.seg.segment_id])
        assert out == [b.seg.segment_id, c.seg.segment_id, a.seg.segment_id]
        ep = FakeEndpoint()
        ep.responses[RERANK_TASK] = json.dumps({"order": [b.seg.segment_id,
                                                          c.seg.segment_id,
                                                          a.seg.segment_id]})
        r = rerank_hits(hits, "搜一下", ep, cache={}, window_version=1)
        assert [h.seg.segment_id for h in r] == [b.seg.segment_id, c.seg.segment_id, a.seg.segment_id]

    def test_missing_id_rejected(self):
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        assert parse_rerank_response(json.dumps({"order": [ids[0], ids[1]]}), ids) is None

    def test_duplicate_id_rejected(self):
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        assert parse_rerank_response(json.dumps({"order": [ids[0], ids[0], ids[2]]}), ids) is None

    def test_extra_id_rejected(self):
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        assert parse_rerank_response(json.dumps({"order": ids + ["zzz"]}), ids) is None

    def test_code_fence_tolerated(self):
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        out = parse_rerank_response(
            "```json\n" + json.dumps({"order": ids}) + "\n```", ids)
        assert out == ids

    def test_thinking_prose_prefix_rescued(self):
        """reasoning/thinking 散文前缀夹带 JSON（reasoning 内容被并进 message.content）
        → 括号平衡法提取首个完整 JSON 对象，仍能解析出正确 order（此前为 None）。"""
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        json_part = json.dumps({"order": ids})
        # 散文前缀里带自己的 " 和 { }，考验括号平衡法是否正确跳过字符串内的花括号
        prose = ('Let me think carefully. The user asks about {"note": "x}y"}, '
                 'so I reason through it step by step before answering:')
        out = parse_rerank_response(prose + " " + json_part, ids)
        assert out == ids

    def test_thinking_prose_prefix_no_truncation_still_rejects_bad_json(self):
        """散文前缀 + 合法 JSON 但缺 id（校验语义不变）→ 仍 None。"""
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        bad = json.dumps({"order": ids[:2]})   # 缺 1 个 id
        out = parse_rerank_response("Some prose first. " + bad, ids)
        assert out is None

    # ── 思考模型（reasoning 模型）实证失败面（2026-10-10 devops 日志 sc.rerank not parseable）──
    def test_index_order_str_accepted_maps_to_ids(self):
        """思考模型 9/10 翻车模式：候选序号 ["0","2","1"]（字符串数字）代替完整 id →
        按 prompt 候选顺序映射回 ids（与全排列校验同强度）。"""
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        out = parse_rerank_response('\n\n{"order": ["0", "2", "1"]}', ids)
        assert out == [ids[0], ids[2], ids[1]]

    def test_index_order_ints_accepted(self):
        """整数形式 [2, 0, 1] 同映射（LLM 偶尔吐 int 不吐字符串）。"""
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        out = parse_rerank_response('```json\n{"order": [2, 0, 1]}\n```', ids)
        assert out == [ids[2], ids[0], ids[1]]

    def test_index_order_out_of_range_rejected(self):
        """索引越界（"3" ∉ 0..k-1）→ 非 0..k-1 全排列 → 仍 None（校验不放宽）。"""
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        assert parse_rerank_response(json.dumps({"order": ["0", "1", "3"]}), ids) is None

    def test_index_order_duplicate_rejected(self):
        """索引重复（"0","0","2"）→ 非 0..k-1 全排列 → 仍 None。"""
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        assert parse_rerank_response(json.dumps({"order": ["0", "0", "2"]}), ids) is None

    def test_index_order_wrong_length_rejected(self):
        """索引个数 ≠ 候选数（2 个索引对 3 候选）→ 仍 None。"""
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        assert parse_rerank_response(json.dumps({"order": ["0", "1"]}), ids) is None

    def test_id_full_permutation_wins_over_index_face(self):
        """同一响应里若已有严格全排列（真 id），先命中 id 面——索引面不抢道。"""
        a, b, c = _hits([0.1, 0.2, 0.3])
        ids = [h.seg.segment_id for h in (a, b, c)]
        out = parse_rerank_response(
            "First I list indices [0, 2, 1], then the real answer: "
            + json.dumps({"order": [ids[1], ids[0], ids[2]]}), ids)
        assert out == [ids[1], ids[0], ids[2]], "id 全排列优先于索引面"


# ───────────────────────── 确定性 ─────────────────────────

class TestDeterminism:
    def test_gate_twice_identical(self):
        """同输入两次 gate 判定全等（纯函数）。"""
        g1, g2 = RerankGate(), RerankGate()
        for hits, q in ((_hits([0.0, 0.0, 0.0]), "任意"),
                        (_hits([5.0, 1.0]), "搜一下 x"),
                        (_hits([0.4, 0.2]), "普通查询")):
            assert g1.should_fire(hits, q) == g2.should_fire(hits, q)

    def test_rerank_twice_identical(self):
        """同输入 + 同 cache + 同 canned 响应两次全等（缓存重放面）。"""
        cache = {}
        def run():
            ep = FakeEndpoint()
            ep.responses[RERANK_TASK] = json.dumps(
                {"order": [h.seg.segment_id for h in _hits([1, 2, 3])[::-1]]})
            out = rerank_hits(_hits([1, 2, 3]), "搜一下 部署", ep,
                              cache=cache, window_version=5)
            return [h.seg.segment_id for h in out]
        assert run() == run()


# ───────────────────────── engine select_context 接线（sc 契约面）────────────────────────

class TestSelectContextRerankWiring:
    """engine 钩子（sc.recall → P2 soft_rank → P3 gate rerank → recalled）：
    端点注入 _aux.main_endpoint（宿主主模型共享模式），gate 默认（low_score_threshold=3.0，
    活体分数分布空档中位：top1 < 3.0 → LLM 重排兜弱/空/歧义面，≥ 3.0 → 零 LLM）。"""

    @pytest.fixture()
    def engine(self, tmp_path):
        from engine import SupercompressContextEngine
        e = SupercompressContextEngine(external_store_path=str(tmp_path), auto_recall=True)
        e._aux.main_endpoint = FakeEndpoint()
        return e

    def _seed(self, e):
        # P0/P1/P2 三段对本 query 均 >0 分（CJK bigram 重叠面；P2 走 网关限流 同词簇）：
        # 保证 LLM 重排对「recalled 顺序」可见（recalled 只含 P0/P1 >0 分；P2 恒被过滤）。
        segs = [e.store.append(_mkseg("网关限流阈值调至每秒五百请求并配置告警", tier="P0", seq=0)),
                e.store.append(_mkseg("部署发布版本四十二到生产环境保留回滚入口并输出工具结果",
                                      tier="P1", seq=1)),
                e.store.append(_mkseg("网关限流阈值调高到每秒八百并配置告警", tier="P2", seq=2))]
        e.store.flush()   # P1/P2 落盘 → _window_fresh() replay 可见（P0 即时 fsync 已在盘）
        return segs

    def _recalled_texts(self, out):
        """召回块 body（去 [supercompress recall] 标记）→ recalled 段序。

        S1 尾投递面（Plan C Phase 1）：召回块并入末条 user 消息（不再独立 role=system），
        取含标记的消息，去标记后按 \\n\\n 切即得段序。段序含 P0 垫底（P0 段被 P1 背景让位到块尾）。"""
        block = [m for m in out if "[supercompress recall]" in str(m.get("content", ""))][0]
        body = str(block["content"]).split("[supercompress recall]", 1)[1]
        return [t for t in body.strip().split("\n\n") if t]

    def test_explicit_search_triggers_llm_rerank_and_cache(self, engine):
        """显式搜索词 → gate 触发 + LLM 1 次重排（P1 段提到召回块首位，P0 段让位 = 重排可见）；
        第 2 次同窗口（store 不变）→ 缓存命中、LLM 0 次。"""
        p0, p1, _p2 = self._seed(engine)
        engine._aux.main_endpoint.responses[RERANK_TASK] = json.dumps(
            {"order": [p0.segment_id, p1.segment_id, _p2.segment_id]})
        window_version = engine.store.line_count   # 窗口指纹（O(1) 单调；缓存键位）
        q = {"role": "user", "content": "搜一下 网关限流 部署版本 工具输出"}
        out = engine.select_context(
            [{"role": "user", "content": "x"}], incoming_message=q)
        assert out is not None, "召回非空应产出替换请求"
        texts = self._recalled_texts(out)
        assert texts and texts[-1].startswith("网关限流阈值调至"), \
            "P0 垫底（Plan C Phase 1）：被召回 P0 段被 P1 背景让位到块尾（全请求最后字节=recency 最强位）"
        assert texts[0].startswith("部署发布版本"), \
            "P1 背景段在 P0 之前（块内排序 背景 P1/P2/Pn → P0 垫底）"
        assert engine._rerank_cache, "LLM 成功重排后缓存非空"
        assert len(engine._aux.main_endpoint.calls) == 1, "gate 触发 + 缓存 miss → LLM 恰 1 次"
        assert len(engine._rerank_cache) == 1
        key = list(engine._rerank_cache)[0]
        assert key[1] == window_version, "缓存键 window_version = 调用方传入的 store.line_count"
        # 第 2 次同窗口：缓存命中 → 零 LLM 调用、重放全等
        out2 = engine.select_context([{"role": "user", "content": "x"}], incoming_message=q)
        assert out2 == out, "缓存重放 → 输出全等"
        assert len(engine._aux.main_endpoint.calls) == 1, "同 key 第 2 次零 LLM 调用"

    def test_fail_safe_endpoint_unchanged_output(self, engine):
        """LLM fail（限流/超时）→ fail-safe：输出 = P1 管线原序（P1 段 10.59 分排首位；
        不炸、不降级、不写缓存）。与「rerank 命中」面（LLM 把 P0 提到首位）对照，
        证明 fail-safe 未破管线原序。"""
        self._seed(engine)
        engine._aux.main_endpoint.fail = True
        q = {"role": "user", "content": "搜一下 网关限流 部署版本 工具输出"}
        out = engine.select_context([{"role": "user", "content": "x"}], incoming_message=q)
        assert out is not None, "fail-safe 不炸：仍产出（P1 原序召回）"
        texts = self._recalled_texts(out)
        assert texts and texts[0].startswith("部署发布版本"), \
            "LLM 不可用 → P1 原序保持（P1 段 10.59 分仍首位，未被 LLM 重排）"
        assert engine._rerank_cache == {}, "失败面不写缓存（下次可重试）"
        assert len(engine._aux.main_endpoint.calls) == 1, "gate 触发 → LLM 恰好被试一次后 fail"

    def test_no_explicit_keyword_no_llm_call(self, engine):
        """默认保守 gate：无显式搜索词 + 正常高分召回（P0 段 >0 分）→ LLM 0 调用（常路径不放大）。"""
        self._seed(engine)
        engine.select_context(
            [{"role": "user", "content": "x"}],
            incoming_message={"role": "user", "content": "网关限流 阈值 调至五百请求"})
        assert len(engine._aux.main_endpoint.calls) == 0, "非触发面零 LLM（非常态调用）"
        assert engine._rerank_cache == {}

    def test_session_boundaries_clear_cache(self, engine, tmp_path):
        """会话边界（on_session_start / on_session_reset）对称清缓存（防跨 session 泄漏）。"""
        self._seed(engine)
        engine._rerank_cache["sentinel"] = ["x"]
        engine.on_session_reset()
        assert engine._rerank_cache == {}, "on_session_reset 必须清 rerank 缓存"
        engine.on_session_start("s_rerank")
        assert engine._rerank_cache == {}, "on_session_start 必须清 rerank 缓存"


# ───────────────────────── 3.0 默认阈值锁定（活体分数分布空档中位）────────────────────────

class TestRerankGate3Point0Default:
    """gate① 默认阈值 3.0（活体 463 段 store probe 2026-10-04：
    强匹配 14–16 / 中等 5–14 / 弱 2.08–2.13 / 不相关 0，空档 2.13–5.22 → 3.0 中位）。
    锁定 engine 默认 + gate① 在 (0,3.0) 触发 / ≥3.0 不触发 / 边界等号不触发。"""

    def test_engine_default_threshold_3_0(self, tmp_path):
        """engine 级默认 = 3.0（无 config.yaml 覆盖时）。"""
        from engine import SupercompressContextEngine
        e = SupercompressContextEngine(external_store_path=str(tmp_path))
        assert e.rerank_low_score_threshold == 3.0, \
            "engine 默认 rerank_low_score_threshold 应 = 3.0（活体分数分布空档中位）"

    def test_gate1_fires_for_weak_top1(self):
        """top1=2.5 < 3.0 → gate① 触发（弱/空/歧义召回兜底 LLM 重排）。"""
        gate = RerankGate(low_score_threshold=3.0)
        assert gate.should_fire(_hits([2.5, 1.0, 0.5]), "弱召回查询") is True, \
            "top1=2.5 < 3.0 → gate① 应触发"

    def test_gate1_not_fire_for_medium_top1(self):
        """top1=5.0 ≥ 3.0 → gate① 不触发（BM25 有把握、零 LLM 常态面）。"""
        gate = RerankGate(low_score_threshold=3.0)
        assert gate.should_fire(_hits([5.0, 3.5, 1.0]), "中等召回查询") is False, \
            "top1=5.0 ≥ 3.0 → gate① 不触发"

    def test_gate1_boundary_exactly_3_0(self):
        """top1=3.0（等号）→ 不触发（RerankGate 用严格 < 判定：≥ 不触发）。"""
        gate = RerankGate(low_score_threshold=3.0)
        assert gate.should_fire(_hits([3.0, 2.5, 1.0]), "边界") is False, \
            "top1=3.0 不 < 3.0 → gate① 不触发（等号归 ≥ 面）"


class TestCallLlmEndpointThinkingModel:
    """思考模型（enable_thinking 未关 / aux 通道未继承 custom_providers.extra_body）
    响应面：思维链走 message.reasoning_content，正文在 content。
    宿主 aux 端点（_CallLlmEndpoint）只取 content：
    - content 非空 → 正常返回（思维链留在 reasoning_content，下游 parse 不受影响）；
    - content 空 + reasoning 非空 → 返回 content（空串，下游 parse fail-safe 原样返回），
      并打 WARNING 诊断日志（2026-10-10 devops 日志 sc.rerank not parseable 的根因面）；
    绝不把 reasoning_content 并进 content（思考内容不是答案，不能冒充正文）。
    """

    def _endpoint(self, call_llm):
        from engine import _CallLlmEndpoint
        return _CallLlmEndpoint(call_llm, {"model": "m", "provider": "custom",
                                           "base_url": "", "api_key": "", "api_mode": ""})

    def test_namespace_response_uses_content_ignores_reasoning(self):
        class _Msg:
            content = '{"order": ["a", "b"]}'
            reasoning_content = "thinking..."
        class _Ch:
            message = _Msg()
        class _Resp:
            choices = [_Ch()]
        ep = self._endpoint(lambda **kw: _Resp())
        out = ep.call("auxiliary.supercompress_rerank", "prompt")
        assert out == '{"order": ["a", "b"]}', "namespace 面只取 content（reasoning 不并入）"

    def test_dict_response_empty_content_nonempty_reasoning_returns_empty(self):
        """思考模型 content 空 / reasoning 非空 → 返回空串（下游 fail-safe），不抛、不合并。"""
        resp = {"choices": [{"message": {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I think the order is seg_a, seg_b, seg_c...",
        }}]}
        ep = self._endpoint(lambda **kw: resp)
        out = ep.call("auxiliary.supercompress_rerank", "prompt")
        assert out == "", "content 空 → 返回空串（rerank 解析层 fail-safe 原样返回入参 hits）"

    def test_dict_response_content_present_returns_content(self):
        resp = {"choices": [{"message": {
            "role": "assistant",
            "content": '{"order": ["seg_a", "seg_b"]}',
            "reasoning_content": "thinking...",
        }}]}
        ep = self._endpoint(lambda **kw: resp)
        out = ep.call("auxiliary.supercompress_rerank", "prompt")
        assert out == '{"order": ["seg_a", "seg_b"]}'


# ─────────────────── R4（2026-10-10）：gate 触发时默认 drop model-provenance 段 ─────────────────
class TestRerankDropModelOnGate:
    """R4：rerank gate 触发（弱/空/显式搜索 = 不确定性高面）时，provenance="model" 的
    hit 段默认 drop（不进 top-k / 不写缓存 / 不进返回集）——agent 自述段不该在不确定
    召回面冒充用户级 P0 被 LLM 重排保留。user-provenance 段全保留。
    drop_model_on_gate=False → 回滚原 keep-all 现状。"""

    def test_gate_fired_drops_model_keeps_user(self):
        """gate 触发（显式搜索词）+ 混合 provenance → model 段被 drop、user 段保留。"""
        from sc.recall import Hit, Segment
        m1 = Hit(Segment(tier="P0", provenance="model", original_text="agent self report",
                         seq=0, ts=0.0), 5.0)
        u1 = Hit(Segment(tier="P0", provenance="user", original_text="user directive",
                         seq=1, ts=0.0), 4.0)
        ep = FakeEndpoint()
        out = rerank_hits([m1, u1], "搜一下 部署", ep, cache={}, window_version=1)
        assert [h.seg.segment_id for h in out] == [u1.seg.segment_id], \
            "gate 触发后 model 段 drop、user 段保留（只剩 user P0）"

    def test_gate_fired_all_model_returns_empty(self):
        """gate 触发 + 全 model 段 → 全 drop 后空召回（不发起 LLM、不炸）。"""
        from sc.recall import Hit, Segment
        hits = [Hit(Segment(tier="P1", provenance="model", original_text=f"agent {i}",
                            seq=i, ts=0.0), float(i + 1)) for i in range(3)]
        ep = FakeEndpoint()
        out = rerank_hits(hits, "搜一下", ep, cache={}, window_version=1)
        assert out == [], "全 model 段 gate 触发 → 全 drop → 空返回"
        assert len(ep.calls) == 0, "全 drop 后空召回 → 不发起 LLM 调用"

    def test_gate_not_fired_no_drop(self):
        """gate 不触发（非搜索词 + 高分）→ 不 drop（原样返回，R4 只在 gate 面生效）。"""
        from sc.recall import Hit, Segment
        m1 = Hit(Segment(tier="P1", provenance="model", original_text="agent x",
                         seq=0, ts=0.0), 9.0)
        m2 = Hit(Segment(tier="P1", provenance="model", original_text="agent y",
                         seq=1, ts=0.0), 8.0)
        ep = FakeEndpoint()
        # 非搜索词 + top1=9.0 ≥ 默认阈值 → gate 不触发 → 不 drop、零 LLM
        out = rerank_hits([m1, m2], "网关 限流 阈值", ep, cache={}, window_version=1)
        assert [h.seg.segment_id for h in out] == [h.seg.segment_id for h in [m1, m2]], \
            "gate 不触发 → R4 不生效（原样保留 model 段）"
        assert len(ep.calls) == 0

    def test_rollback_flag_restores_keep_all(self):
        """drop_model_on_gate=False → 回滚原 keep-all 现状（model 段不 drop）。"""
        from sc.recall import Hit, Segment
        m1 = Hit(Segment(tier="P0", provenance="model", original_text="agent self report",
                         seq=0, ts=0.0), 5.0)
        u1 = Hit(Segment(tier="P0", provenance="user", original_text="user directive",
                         seq=1, ts=0.0), 4.0)
        ep = FakeEndpoint()
        ep.responses[RERANK_TASK] = json.dumps(
            {"order": [u1.seg.segment_id, m1.seg.segment_id]})
        out = rerank_hits([m1, u1], "搜一下 部署", ep, cache={}, window_version=1,
                          drop_model_on_gate=False)
        assert {h.seg.segment_id for h in out} == \
               {m1.seg.segment_id, u1.seg.segment_id}, \
            "drop_model_on_gate=False → model 段保留（keep-all 现状）"
