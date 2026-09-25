"""Block ④ aux provider（A 版：零新 provider）+ tagger（机器可解析 JSON）。"""
import json

from sc.schema import Segment
from sc.aux_provider import AuxProvider
from sc.tagger import Tagger
from sc.distill import Distiller
from helpers import mkseg, mk_window, TOPIC, MV


class TestBlock4_Aux:
    def test_tagging_scoring_uses_configured_main_endpoint(self, aux):
        ep = aux.main_endpoint
        tagger = Tagger(aux, TOPIC)
        tier, structural, regime, reason = tagger.classify("the module depends on A and B", "model")
        assert any(c[0] == "tagging" for c in ep.calls), "直调用户已配主模型端点（零新 provider）"
        assert tier in ("P0", "P1", "P2")
        assert aux.fallback_endpoint is ep, "(A) 失败域与主模型同源，无第二 provider"

    def test_digest_via_native_auxiliary_compression(self, aux, store):
        assert AuxProvider.COMPRESSION_TASK == "auxiliary.compression", "digest 映射原生 summarizer 任务"
        assert aux.fallback_chain[-1] == "main", "fallback_chain 链尾=main（原生兜底，0 行自包）"
        seg = mkseg(store, "context for digest block", "P1")
        d1 = aux.ensure_digest(seg)
        assert d1 is not None and seg.digest_ready
        assert any(c[0] == "auxiliary.compression" for c in aux.main_endpoint.calls)
        n = len(aux.main_endpoint.calls)
        d2 = aux.ensure_digest(seg)  # 幂等三元组：(content_hash, model_version, session_topic)
        assert d2["digest_ref_id"] == d1["digest_ref_id"]
        assert len(aux.main_endpoint.calls) == n, "同组不重算"

    def test_transient_fail_regime_tag_in_kpi1(self, store, aux):
        aux.main_endpoint.fail = True
        p0 = Segment(kind="segment", tier="P0", provenance="model",
                     original_text="llm p0 " * 10, session_topic=TOPIC, model_version=MV)
        p0s = [mkseg(store, "user: ship v1", "P0", "user", structural=True)]
        w = mk_window(100, p0s + [p0])
        d = Distiller(w, store, aux).run()
        assert "transient_skip" in d.events
        assert store.kpi_log[-1]["regime"] == "transient_fail", "KPI #1 带 regime 标签"
        assert w.segs[0].form == "resident", "结构地板仍保 user-msg P0"

    def test_single_slot_max_concurrency_1(self, aux, store):
        assert aux.max_concurrency == 1, "shipped default：单槽并发保护"
        seg = mkseg(store, "ctx for digest single slot", "P1")
        aux.start_turn()
        assert aux.ensure_digest(seg) is None, "turn 内调用排队（不与 turn 解码撞车）"
        comp = [c for c in aux.main_endpoint.calls if c[0] == "auxiliary.compression"]
        assert len(comp) == 0, "turn 未结束 → aux 调用未执行"
        aux.end_turn()
        comp = [c for c in aux.main_endpoint.calls if c[0] == "auxiliary.compression"]
        assert len(comp) == 1 and seg.digest_ready

    def test_tagger_output_machine_parseable_json(self, aux, store, monkeypatch):
        tagger = Tagger(aux, TOPIC)
        text = "plain statement about module X"
        ep = aux.main_endpoint
        ep.responses["tagging"] = json.dumps({"tier": "P2", "score": 3})
        tier, _, _, _ = tagger.classify(text, "model")
        assert tier == "P2"
        ep.responses["tagging"] = json.dumps({"tier": "P2", "score": 6})
        tier, _, _, _ = tagger.classify("another plain statement Y", "model")
        assert tier == "P1", "边界 ±1（6–8）→ P1 保守"
        ep.responses["tagging"] = "I think this is P2 with score 3"  # 自然语言 → 不可机器解析
        tier, structural, regime, reason = tagger.classify("one more statement Z", "model")
        assert tier == "P0" and structural, "不可解析 → 保守 P0，永不当 P2 丢"
