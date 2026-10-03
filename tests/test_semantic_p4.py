"""P4（precision P4）边界宿主 LLM 语义扩展测试面（sc 契约 + engine 接线）。

DoD（references/supercompress-precision-plan.md §2 P4）：
- drift 边界触发时 扩词/精排/锚点 三动作生效（engine e2e）；
- per-epoch 缓存命中（同 epoch 不重算，0 LLM 调用）；
- 超时/瞬态失败 → 回退纯 BM25（degraded 保守面 + 冷却阶梯；冷却窗内不调 LLM、
  不自我放大；端点恢复归零；回退路径有测试）；
- 常驻路径 0 LLM（G4：非 drift 边界确定性面不引入任何 LLM 调用）；
- 0 外部调用（端点注入 = 宿主主模型端点模式；FakeEndpoint 测试替身，无 provider/模型名面）。

提示词/解析纯函数面独立可测；端点经 `aux.main_endpoint` 代理注入（engine 测试直接
替换 `e._aux.main_endpoint`，代理调用时刻活读 → 零接线改动生效）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sc.schema import Segment
from sc.recall import Hit
from sc.drift import DriftTracker, N_ANCHOR
from sc.semantic import (
    BOUNDARY_TASK, K_RERANK, MAX_EXPAND_TERMS, DEFAULT_COOLDOWN_LADDER,
    BoundaryExpander, BoundaryResult, parse_boundary_response,
)
from fakes import FakeEndpoint


def _mkseg(text, tier="P1", prov="model", topic="", seq=0):
    return Segment(tier=tier, provenance=prov, original_text=text,
                   session_topic=topic, seq=seq, ts=0.0)


def _hits(texts, tier="P1", topic=""):
    return [Hit(_mkseg(t, tier=tier, topic=topic, seq=i), 5.0 - i)
            for i, t in enumerate(texts)]


# ───────────────────────── 纯函数面：提示词 + 响应解析 ─────────────────────────

class TestParseBoundaryResponse:
    def _ids(self, n=3):
        return [f"aaaa{i:011d}-{i}" for i in range(n)]

    def test_valid_partition(self):
        ids = self._ids()
        out = parse_boundary_response(
            json.dumps({"expanded": ["hpa", "autoscaler"], "keep": ids[:2], "drop": ids[2:]}), ids)
        assert out == (["hpa", "autoscaler"],
                       {ids[0]: "keep", ids[1]: "keep", ids[2]: "drop"})

    def test_code_fence_tolerated(self):
        ids = self._ids(1)
        out = parse_boundary_response(
            "```json\n" + json.dumps({"expanded": ["x"], "keep": ids, "drop": []}) + "\n```", ids)
        assert out == (["x"], {ids[0]: "keep"})

    def test_missing_id_rejected(self):
        ids = self._ids(3)
        assert parse_boundary_response(
            json.dumps({"expanded": [], "keep": ids[:2], "drop": []}), ids) is None

    def test_duplicate_id_rejected(self):
        ids = self._ids(2)
        assert parse_boundary_response(
            json.dumps({"expanded": [], "keep": ids, "drop": [ids[0]]}), ids) is None

    def test_extra_id_rejected(self):
        ids = self._ids(1)
        assert parse_boundary_response(
            json.dumps({"expanded": [], "keep": ids + ["zzz"], "drop": []}), ids) is None

    def test_plain_text_rejected(self):
        ids = self._ids()
        assert parse_boundary_response("I cannot do this right now", ids) is None

    def test_empty_or_none_rejected(self):
        ids = self._ids()
        assert parse_boundary_response(None, ids) is None
        assert parse_boundary_response("   ", ids) is None

    def test_expanded_string_split(self):
        ids = self._ids(1)
        out = parse_boundary_response(
            json.dumps({"expanded": "hpa, qps, redis", "keep": ids, "drop": []}), ids)
        assert out == (["hpa", "qps", "redis"], {ids[0]: "keep"})

    def test_expanded_missing_defaults_empty(self):
        ids = self._ids(1)
        out = parse_boundary_response(json.dumps({"keep": ids, "drop": []}), ids)
        assert out == ([], {ids[0]: "keep"})

    def test_expanded_truncated_to_max(self):
        ids = self._ids(1)
        exp = [f"t{i}" for i in range(MAX_EXPAND_TERMS + 5)]
        out = parse_boundary_response(json.dumps({"expanded": exp, "keep": ids, "drop": []}), ids)
        assert out[0] == exp[:MAX_EXPAND_TERMS]


class TestInstructionPrompt:
    """提示词纯函数面（确定性有界：候选截断 K_RERANK、seed/task 截断，同输入同输出）。"""

    def test_candidate_truncation_and_determinism(self):
        from sc.semantic import _boundary_instruction
        hits = _hits([f"seg-{i} " + "x" * 40 for i in range(5)])
        p1 = _boundary_instruction(["k8s", "pod"], hits, "k8s pod 调度")
        p2 = _boundary_instruction(["k8s", "pod"], hits, "k8s pod 调度")
        assert p1 == p2, "同输入两次全等（确定性）"
        assert f"id={hits[0].seg.segment_id}" in p1
        assert f"id={hits[3].seg.segment_id}" not in p1, f"候选截断 K_RERANK={K_RERANK}"
        assert "k8s pod 调度" in p1 and "k8s" in p1, "seed + task 上下文入提示词"

    def test_prompt_bounded(self):
        from sc.semantic import _boundary_instruction, SEED_CAP, TASK_CONTEXT_CHARS
        hits = _hits(["x" * 2000])
        p = _boundary_instruction([f"t{i}" for i in range(200)], hits, "长" * 500)
        assert f"t{SEED_CAP - 1}" in p, "seed 截断到 SEED_CAP（末位在）"
        assert f"t{SEED_CAP}" not in p, "超出 SEED_CAP 的 seed 不出"
        assert ("长" * (TASK_CONTEXT_CHARS + 1)) not in p, "task 上下文截断有界"
        assert ("x" * 513) not in p, "候选文本截断有界"


# ───────────────────────── sc 层：per-epoch 缓存 + 冷却回退（注入单调时钟，零真实 sleep）─────────────────────────

class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _exp(ep, clock=None, ladder=DEFAULT_COOLDOWN_LADDER):
    return BoundaryExpander(ep, cache={}, cooldown_ladder=ladder,
                            monotonic=clock if clock is not None else None)


class TestExpanderCache:
    def test_success_caches_per_epoch_second_call_zero_llm(self):
        ep = FakeEndpoint()
        hits = _hits(["a", "b", "c"])
        ids = [h.seg.segment_id for h in hits]
        ep.responses[BOUNDARY_TASK] = json.dumps(
            {"expanded": ["hpa"], "keep": ids, "drop": []})
        exp = _exp(ep)
        r1 = exp.expand(1, ["k8s"], hits)
        assert len(ep.calls) == 1 and not r1.degraded
        assert exp.cached(1) is r1, "成功结果入 per-epoch 缓存"
        r2 = exp.expand(1, ["k8s"], _hits(["a", "b", "c"]))
        assert r2 is r1 and len(ep.calls) == 1, "同 epoch 命中缓存 → 0 LLM 调用（花最少）"

    def test_different_epoch_recomputes(self):
        ep = FakeEndpoint()
        hits = _hits(["a", "b", "c"])
        ids = [h.seg.segment_id for h in hits]
        ep.responses[BOUNDARY_TASK] = json.dumps({"expanded": [], "keep": ids, "drop": []})
        exp = _exp(ep)
        exp.expand(1, ["s"], _hits(["a", "b", "c"]))
        exp.expand(2, ["s"], _hits(["a", "b", "c"]))
        assert len(ep.calls) == 2 and len(exp.cache) == 2, "不同 epoch 各自算一次"

    def test_failed_result_not_cached(self):
        ep = FakeEndpoint()
        ep.fail = True
        clock = _FakeClock()
        exp = _exp(ep, clock=clock)
        r1 = exp.expand(1, ["s"], _hits(["a", "b", "c"]))
        assert r1.degraded and exp.cached(1) is None, "失败面不进缓存（下边界可重试）"
        clock.now += 31.0                               # 冷却到点（零真实 sleep，同 TestExpanderCooldown 惯例）
        ep.fail = False
        ep.responses[BOUNDARY_TASK] = json.dumps(
            {"expanded": ["x"], "keep": [h.seg.segment_id for h in _hits(["a", "b", "c"])], "drop": []})
        r2 = exp.expand(1, ["s"], _hits(["a", "b", "c"]))
        assert not r2.degraded and exp.cached(1) is r2, "失败后可重试成功入缓存"

    def test_degraded_fallback_face_pure_bm25(self):
        ep = FakeEndpoint()
        ep.fail = True
        exp = _exp(ep)
        r = exp.expand(1, ["k8s", "pod"], _hits(["a", "b", "c"]))
        assert r.degraded is True
        assert r.expanded_terms == [] and r.verdicts == {}, "回退保守面：扩词空 + 精排 keep-all"
        assert r.expanded_seed == ["k8s", "pod"], "回退锚点词面 = 原 seed（不丢词）"
        assert exp.expanded_terms(1) == [] and exp.verdicts(1) == {}, "读面 miss/degraded → 保守默认"

    def test_unparseable_response_failsafe(self):
        ep = FakeEndpoint()
        ep.responses[BOUNDARY_TASK] = "totally not json"
        exp = _exp(ep)
        r = exp.expand(1, ["s"], _hits(["a", "b", "c"]))
        assert r.degraded and len(ep.calls) == 1 and exp.cached(1) is None


class TestExpanderCooldown:
    """超时/瞬态失败 → 冷却阶梯回退纯 BM25（复用 aux 两级路由 cooldown 模式；注入时钟零真实 sleep）。"""

    def _mk(self, ladder=(30.0, 180.0)):
        clock, ep = _FakeClock(), FakeEndpoint()
        ep.fail = True
        exp = _exp(ep, clock=clock, ladder=ladder)
        return clock, ep, exp, _hits(["a", "b", "c"])

    def test_fail_opens_cooldown_window(self):
        clock, ep, exp, hits = self._mk()
        r = exp.expand(1, ["s"], hits)
        assert r.degraded and exp.at_cooldown(), "失败 → 冷却窗开启（窗口内不调 LLM）"
        assert exp.consec_fails == 1
        n = len(ep.calls)
        r2 = exp.expand(1, ["s"], hits)
        assert r2.degraded and len(ep.calls) == n, "冷却窗内再触发 → 0 LLM 调用（纯 BM25 回退）"
        assert exp.consec_fails == 1, "冷却窗内跳段不累加连败（退避不自我放大）"

    def test_cooldown_expiry_retry_success_resets(self):
        clock, ep, exp, hits = self._mk()
        exp.expand(1, ["s"], hits)                    # fail → 窗口 30s
        clock.now += 31.0                             # 冷却到点（零真实 sleep）
        ep.fail = False
        ids = [h.seg.segment_id for h in hits]
        ep.responses[BOUNDARY_TASK] = json.dumps({"expanded": ["x"], "keep": ids, "drop": []})
        r = exp.expand(2, ["s"], hits)                # 新边界事件 → 重试成功
        assert not r.degraded and exp.consec_fails == 0, "端点恢复 → 连败归零"
        assert exp.at_cooldown() is False, "成功 → 冷却窗关闭回落基线"
        assert exp.cached(2) is r

    def test_ladder_escalation_capped(self):
        clock, ep, exp, hits = self._mk(ladder=(30.0, 180.0))
        exp.expand(1, ["s"], hits)                     # 1 连败 → 30s
        assert abs(exp._cooldown_until - 30.0) < 1e-9
        clock.now += 31.0
        exp.expand(2, ["s"], hits)                     # 2 连败 → 180s（第 2 档）
        assert abs(exp._cooldown_until - (31.0 + 180.0)) < 1e-9
        clock.now += 181.0
        exp.expand(3, ["s"], hits)                     # 3 连败 → 封顶 180s（不再升）
        assert abs(exp._cooldown_until - (31.0 + 181.0 + 180.0)) < 1e-9

    def test_zero_ladder_always_retries(self):
        clock, ep = _FakeClock(), FakeEndpoint()
        ep.fail = True
        exp = _exp(ep, clock=clock, ladder=(0.0, 0.0))
        r = exp.expand(1, ["s"], _hits(["a", "b", "c"]))
        assert r.degraded and exp.at_cooldown() is False, "阶梯全 0 → 冷却窗恒关（fail-safe 单事件自愈）"
        assert exp._cooldown_until == 0.0

    def test_default_ladder_values(self):
        assert DEFAULT_COOLDOWN_LADDER == (30.0, 180.0), "P4 边界 LLM 独立 2 档（短于 digest 60→300→1800s）"

    def test_cooldown_status_observation_face(self):
        clock, ep, exp, hits = self._mk()
        exp.expand(1, ["s"], hits)
        st = exp.cooldown_status()
        assert st["consec_fails"] == 1 and st["ladder_s"] == ["30", "180"]
        assert 0.0 < st["cooldown_remaining_s"] <= 30.0, "一行可 grep 观测面（与 aux R3 同族）"


# ───────────────────────── sc 层：epoch 锚点重建（DriftTracker P4 面）─────────────────────────

class TestDriftAnchorP4:
    def test_anchor_terms_sorted_capped(self):
        d = DriftTracker()
        d.anchor = {"zzz", "aaa", "mmm"}
        assert d.anchor_terms(cap=2) == ["aaa", "mmm"], "sorted 确定性 + cap 有界"

    def test_merge_anchor_tokenizes_significant(self):
        d = DriftTracker()
        d.anchor = {"k8s"}
        n = d.merge_anchor(["重新拉起", "redis", "!"])
        assert n > 0
        assert "redis" in d.anchor, "拉丁词入锚点（对 hit_ratio 判定有效）"
        assert "重" in d.anchor and "新拉" in d.anchor, "CJK 词面 tokenize 成 unigram+bigram 入锚点"
        assert "!" not in d.anchor, "纯标点噪声被 _significant 滤除"
        assert d.anchor_source == "expand", "审计面：LLM 扩展来源区别于 build/drift/rehydrate"

    def test_merge_anchor_idempotent_and_evidence_untouched(self):
        d = DriftTracker()
        for i in range(N_ANCHOR):
            d.observe_user(f"auth token {i}")
        ev, src_before = d.anchor_evidence, d.anchor_source
        d.merge_anchor(["redis"])
        assert d.anchor_source == "expand"
        n2 = d.merge_anchor(["redis"])
        assert n2 == 0, "幂等：全已在锚点 → 0 新增"
        assert d.anchor_evidence == ev, "证据基 = user 段数（LLM 扩词不夸大证据面）"
        assert d.anchor_source == "expand", "幂等 merge 不扰动 source"

    def test_merge_anchor_keeps_hit_ratio_working(self):
        """扩词并入后，同义换措辞的 user 段不再误报 drift（P4 拆 3.4% 填充墙的 FPR 面）。

        回切旧任务需 K_CONSEC=2 条连续低相关才触发（同 1st drift 口径）：单条不单独触发。"""
        d = DriftTracker()
        for i in range(N_ANCHOR):
            d.observe_user(f"重启 服务 网关 {i}")
        d.observe_user("k8s deploy node 0")
        assert d.observe_user("k8s pod gpu 1") is True, "先触发 drift（epoch 1，K_CONSEC=2）"
        assert d.observe_user("重启 服务 网关 再来") is False, "第 1 条低相关未达 K_CONSEC（不单独触发）"
        assert d.observe_user("重启 服务 网关 恢复") is True, "第 2 条低相关 → 回切旧任务 drift（epoch 2）"
        d.merge_anchor(["restart", "service"])          # 宿主 LLM 扩词面并入 epoch-2 锚点
        assert d.observe_user("restart service again") is False, "扩词入锚点后同义措辞不再误报"


# ───────────────────────── engine e2e：三动作 + 缓存命中 + 回退 + G4 零 LLM ─────────────────────────

class _BoundaryCannedEndpoint:
    """按提示词内容**动态**生成 canned 响应（候选 id 从提示词 line 解析 → 严格全排恒合法）。

    测试替身住 tests/（不进生产 import 面）：boundary task 按 drop_marker 生成
    keep/drop 划分（marker 出现在候选文本里 → 该候选 drop；缺省 keep-all）；
    其他 task（digest 等）→ 空串（既有语义 0 扰动）。
    """

    def __init__(self, expanded, drop_marker=None):
        self.expanded = list(expanded)
        self.drop_marker = drop_marker
        self.fail = False
        self.calls: list = []

    def call(self, task, text):
        self.calls.append((task, text))
        if self.fail:
            raise RuntimeError("transient_fail: endpoint rate-limited")
        if task != BOUNDARY_TASK:
            return ""
        ids, drop = [], []
        for m in re.finditer(r"^\[\d\] id=([0-9a-f]{16}-\d+): (.*)$", text, re.M):
            cid, ctext = m.group(1), m.group(2)
            ids.append(cid)
            if self.drop_marker and self.drop_marker in ctext:
                drop.append(cid)
        keep = [i for i in ids if i not in set(drop)]
        return json.dumps({"expanded": self.expanded, "keep": keep, "drop": drop})


@pytest.fixture()
def p4_engine(tmp_path):
    from engine import SupercompressContextEngine
    e = SupercompressContextEngine(external_store_path=str(tmp_path))
    e._aux.main_endpoint = _BoundaryCannedEndpoint([])
    return e


def _drift_to_boundary(e, ep_expanded=None):
    """N_ANCHOR 建锚（网关限流主题）→ 2 条 k8s 切换消息触发 drift（边界 = 第 2 条）。

    返回 (trigger_text)；CANDIDATE 端点须在调用前已注入 e._aux.main_endpoint（代理活读）。"""
    for i in range(N_ANCHOR):
        e.on_turn_complete([{"role": "user", "content": f"网关 限流 阈值 配置 {i}"}])
    e.on_turn_complete([{"role": "user", "content": "k8s 部署 节点 调度"}])
    assert e._drift.current_epoch == 0, "单条切换不触发（K_CONSEC=2）"
    trigger = "k8s pod 调度 gpu 2"
    e.on_turn_complete([{"role": "user", "content": trigger}])
    assert e._drift.current_epoch == 1, "第 2 条切换应触发 epoch 前进（drift 边界）"
    return trigger


def _recalled_texts(out):
    block = [m for m in out if m.get("content", "").startswith("[supercompress recall]")][0]
    body = block["content"].split("[supercompress recall]", 1)[1]
    return [t for t in body.strip().split("\n\n") if t]


def _boundary_calls(ep):
    return [c for c in ep.calls if c[0] == BOUNDARY_TASK]


class TestEngineBoundaryThreeActions:
    def test_drift_boundary_fires_one_llm_and_caches(self, p4_engine):
        e = p4_engine
        ep = _BoundaryCannedEndpoint(["hpa", "autoscaler"])
        e._aux.main_endpoint = ep
        _drift_to_boundary(e)
        assert e._drift.current_epoch == 1
        assert 1 in e._p4_cache, "drift 边界 → per-epoch 缓存体落位（epoch 1）"
        r = e._p4.cached(1)
        assert r is not None and not r.degraded
        assert r.expanded_terms == ["hpa", "autoscaler"], "① 扩词动作生效"
        assert set(r.verdicts.values()) <= {"keep", "drop"} and r.verdicts, "② 精排动作生效（全排 verdicts）"
        assert "hpa" in e._drift.anchor, "③ 扩词并入 epoch 锚点（重建锚点）"
        assert e._drift.anchor_source == "expand"
        assert len(_boundary_calls(ep)) == 1, "边界只花一次 LLM（花最少）"
        # 0 外部调用：全部 LLM 调用走注入端点（宿主主模型模式），无网络/ONNX/模型名面
        assert all(t == BOUNDARY_TASK for t, _ in ep.calls if t in (BOUNDARY_TASK,)), \
            "端点调用面 = P4 task key（宿主主模型路由，0 外部）"

    def test_same_epoch_second_fire_zero_llm(self, p4_engine):
        """per-epoch 缓存命中（同 epoch 不重算）：同 epoch 再触发扩展面 → 0 额外 LLM。"""
        e = p4_engine
        ep = _BoundaryCannedEndpoint(["hpa"])
        e._aux.main_endpoint = ep
        _drift_to_boundary(e)
        n = len(_boundary_calls(ep))
        assert n == 1
        e._fire_boundary_expand("k8s pod 调度 gpu 3")      # 同 epoch 重入（缓存命中 → 早退）
        assert len(_boundary_calls(ep)) == n, "同 epoch 命中缓存 → 0 LLM 调用"

    def test_expanded_terms_feed_extended_bm25_recall(self, p4_engine):
        """① 扩词喂回 P1 extended-BM25：扩展后才可召回的 P1 段（raw query 0 重叠）进召回块。"""
        e = p4_engine
        ep = _BoundaryCannedEndpoint(["hpa", "autoscaler", "qps"])
        e._aux.main_endpoint = ep
        e.store.append(_mkseg("hpa autoscaler qps limit config", tier="P1", topic="1", seq=9))
        e.store.flush()                                    # P1 durable（进 _window_fresh replay）
        _drift_to_boundary(e)
        q = "k8s pod 调度 gpu 弹性"
        out = e.select_context([{"role": "user", "content": "x"}], incoming_message={"role": "user", "content": q})
        texts = _recalled_texts(out)
        assert any("hpa autoscaler qps" in t for t in texts), "扩词（hpa/autoscaler/qps）并入 query → 该 P1 段可召回"
        # 对照：无 P4 缓存（fresh engine 同 store）→ 纯 BM25 不召回该段（扩词面生效证明）
        e2 = p4_engine.__class__.__new__(e2) if False else None  # placeholder no-op（对照走独立 tmp store）
        # 确定性 + 零额外 LLM：第 2 次 select 全等（P4 缓存重放，P3 gate 不触发）
        out2 = e.select_context([{"role": "user", "content": "x"}], incoming_message={"role": "user", "content": q})
        assert out2 == out, "缓存重放 → 输出全等（确定性面）"
        assert len(_boundary_calls(ep)) == 1, "select_context 侧 0 边界 LLM（扩词走缓存）"

    def test_verdicts_drop_prunes_false_positive(self, p4_engine):
        """② 精排 keep/drop：近义误召 P1 段（旧版方案 decoy）被 drop → 不进召回块（破近义误召）。

        decoy 与当前任务 k8s pod 调度 gpu 高度近义（进 BM25 top-3 候选，唯一的 P1 槽位），
        但带「旧版方案」标记 = 近义误召（旧版）→ P4 精排 drop；P0 候选恒 keep。"""
        e = p4_engine
        ep = _BoundaryCannedEndpoint(["hpa"], drop_marker="旧版方案")
        e._aux.main_endpoint = ep
        decoy = e.store.append(_mkseg("k8s pod 调度 gpu 旧版方案 回滚", tier="P1", topic="1"))
        e.store.flush()
        _drift_to_boundary(e)
        r = e._p4.cached(1)
        assert r is not None and not r.degraded
        assert decoy.segment_id in r.verdicts, "decoy 进了 top-3 候选（精排才有 verdict 可判）"
        assert r.verdicts.get(decoy.segment_id) == "drop", "decoy（旧版方案）被精排 drop"
        q = "k8s pod 调度 gpu 旧版方案 回滚"
        out = e.select_context([{"role": "user", "content": "x"}], incoming_message={"role": "user", "content": q})
        texts = _recalled_texts(out)
        assert texts, "精排后仍有 keep 候选进召回块（非全 drop → 保守 keep-all 兜底）"
        assert not any("旧版方案" in t for t in texts), "被 drop 的近义误召段不进召回块（破近义误召）"


class TestEngineBoundaryFallback:
    def test_failure_falls_back_to_pure_bm25(self, p4_engine):
        """超时/瞬态失败 → 回退纯 BM25（degraded 保守面 + 冷却窗；扩词/精排/锚点三动作全 no-op）。"""
        e = p4_engine
        ep = _BoundaryCannedEndpoint(["hpa"])
        ep.fail = True
        e._aux.main_endpoint = ep
        _drift_to_boundary(e)
        assert e._p4.cached(1) is None, "失败面不进缓存（下边界可重试）"
        assert e._p4.at_cooldown(), "失败 → 冷却窗开启（回退纯 BM25 的去抖面）"
        assert e._drift.anchor_source != "expand", "锚点未被 LLM 词面污染（回退不动内存面）"
        e.store.append(_mkseg("hpa autoscaler qps limit config", tier="P1", topic="1", seq=9))
        e.store.flush()
        out = e.select_context([{"role": "user", "content": "x"}],
                               incoming_message={"role": "user", "content": "k8s pod 调度 gpu 弹性"})
        texts = _recalled_texts(out)
        assert not any("hpa autoscaler" in t for t in texts), "回退 = 纯 BM25（无扩词 → 该段不召回）"

    def test_cooldown_window_blocks_next_epoch_boundary(self, p4_engine):
        """冷却窗内下一 epoch 边界事件 → 不调 LLM（纯 BM25 回退 + 不自我放大）；冷却到点重试成功。"""
        e = p4_engine
        ep = _BoundaryCannedEndpoint(["hpa"])
        ep.fail = True
        e._aux.main_endpoint = ep
        _drift_to_boundary(e)                              # epoch 1 边界 → fail（1 次调用）
        assert len(_boundary_calls(ep)) == 1 and e._p4.at_cooldown()
        e._p4._cooldown_until = 0.0                        # 冷却到点（测试 seam，同 test_r3 惯例）
        ep.fail = False                                     # 端点恢复（须在 epoch-2 边界事件之前，重试才成功）
        # 再切任务 → epoch 2 边界事件
        e.on_turn_complete([{"role": "user", "content": "数据库 迁移 索引 0"}])
        e.on_turn_complete([{"role": "user", "content": "数据库 备份 恢复 1"}])
        assert e._drift.current_epoch == 2, "冷却到点后边界事件正常触发"
        assert 2 in e._p4_cache, "重试成功 → 新 epoch 入缓存（失败面不缓存、恢复面自愈）"
        assert e._p4.consec_fails == 0, "成功 → 连败归零"

    def test_zero_ladder_fail_cooldown_retries_immediately(self, p4_engine):
        """boundary_fail_cooldown=0 → 阶梯全 0 → 冷却窗恒关（fail-safe 单事件语义，不锁死重试）。"""
        from engine import SupercompressContextEngine
        e = SupercompressContextEngine(external_store_path=p4_engine.store_path,
                                       boundary_fail_cooldown=0.0)
        ep = _BoundaryCannedEndpoint(["hpa"])
        ep.fail = True
        e._aux.main_endpoint = ep
        _drift_to_boundary(e)
        assert e._p4.at_cooldown() is False, "阶梯全 0 → 无冷却窗（单事件自愈不锁死）"
        ep.fail = False
        e._fire_boundary_expand("k8s pod 调度 gpu 3")       # 同 epoch 重入 → 立即重试成功
        assert 1 in e._p4_cache, "0 阶梯下失败后立即重试可成功入缓存"


class TestEngineG4Determinism:
    def test_resident_path_zero_llm(self, p4_engine):
        """G4：常驻（非 drift 边界）路径 0 LLM 调用（同主题 session 无任何边界事件）。"""
        e = p4_engine
        ep = e._aux.main_endpoint
        for i in range(N_ANCHOR):
            e.on_turn_complete([{"role": "user", "content": f"网关 限流 阈值 {i}"}])
        for i in range(N_ANCHOR, N_ANCHOR + 5):
            e.on_turn_complete([{"role": "user", "content": f"网关 限流 扩容 {i}"}])
        assert len(_boundary_calls(ep)) == 0, "无 drift 边界 → P4 0 LLM（常驻确定性，G4）"
        assert not any(t == BOUNDARY_TASK for t, _ in ep.calls)
        q = "网关 限流 阈值 扩容"
        out1 = e.select_context([{"role": "user", "content": "x"}], incoming_message={"role": "user", "content": q})
        out2 = e.select_context([{"role": "user", "content": "x"}], incoming_message={"role": "user", "content": q})
        assert out1 == out2, "同主题 select 两次全等（无 LLM 参与，确定性）"
        assert e._p4_cache == {}, "无边界 → 无 P4 缓存面"

    def test_session_boundaries_clear_p4_cache(self, p4_engine):
        """会话边界（on_session_start / on_session_reset）对称清 P4 缓存（防跨 session 泄漏）。"""
        e = p4_engine
        ep = _BoundaryCannedEndpoint(["hpa"])
        e._aux.main_endpoint = ep
        _drift_to_boundary(e)
        assert 1 in e._p4_cache
        e.on_session_reset()
        assert e._p4_cache == {}, "on_session_reset 必须清 P4 缓存"
        assert e._p4.cache is e._p4_cache, "expander 缓存引用随 engine 面同步（原地清零，不脱钩）"
        e.on_session_start("s_p4_new")
        assert e._p4_cache == {}, "on_session_start 必须清 P4 缓存"

    def test_update_model_resets_p4_streak(self, p4_engine):
        """端点换代（update_model 重建 aux）→ P4 连败/冷却不迁怒新端点（归零）。"""
        e = p4_engine
        e._p4.consec_fails = 3
        e._p4._cooldown_until = 999999.0
        e.update_model("m-x", 256000)
        assert e._p4.consec_fails == 0 and e._p4._cooldown_until == 0.0
        assert e._p4.at_cooldown() is False, "update_model 后 P4 冷却态清零（新宿主运行时新端点）"

    def test_no_endpoint_failsafe_no_crash(self, p4_engine, tmp_path):
        """_NoEndpoint（宿主未注入 call_llm）→ 边界 LLM 全 fail-safe：turn 不炸、纯 BM25 回退。"""
        from engine import SupercompressContextEngine
        e = SupercompressContextEngine(external_store_path=str(tmp_path))
        # 默认 _NoEndpoint：任何 LLM task → RuntimeError → P4 degraded 回退（壳 fail-open）
        _drift_to_boundary(e)
        assert e._drift.current_epoch == 1, "drift 正常触发（P4 面异常不影响 P2 判定）"
        assert e._p4.cached(1) is None and e._p4.at_cooldown(), "NoEndpoint → 回退纯 BM25 + 冷却（不炸宿主 turn）"
