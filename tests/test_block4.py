"""Block ④ aux provider（A 版：零新 provider）+ tagger（机器可解析 JSON）。"""
import json

from sc.schema import Segment
from fakes import FakeEndpoint
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

    def test_fail_cooldown_skips_pending_segments(self, store):
        """去抖第 1 层：一次瞬态失败 → 冷却窗内 pending 段零端点调用（风暴封死）。"""
        import threading
        aux = AuxProvider(FakeEndpoint(), fail_cooldown=3600.0,
                          digest_sem=threading.BoundedSemaphore(2), async_digest=False)
        aux.main_endpoint.fail = True
        s1 = mkseg(store, "cool seg one", "P0")
        d1 = aux.ensure_digest(s1)
        assert d1 is None and aux.transient, "首次失败 → transient"
        n_calls = len(aux.main_endpoint.calls)
        assert n_calls == 1, "端点恰好被调一次（非每段重试）"
        s2 = mkseg(store, "cool seg two", "P0")
        s3 = mkseg(store, "cool seg three", "P0")
        assert aux.ensure_digest(s2) is None, "冷却窗内第 2 段跳过（零端点调用）"
        assert aux.ensure_digest(s3) is None, "冷却窗内第 3 段跳过（零端点调用）"
        assert len(aux.main_endpoint.calls) == n_calls, "冷却窗内端点调用数不变"

    def test_fail_cooldown_expires_resumes(self, store):
        """去抖可恢复：连败 2 次（阶梯 0.2s×1 档封顶）后手动拨回冷却窗，端点恢复 → 正常生成。"""
        import threading
        aux = AuxProvider(FakeEndpoint(), fail_cooldown=0.2,
                          cooldown_backoff=(0.2,),
                          digest_sem=threading.BoundedSemaphore(2), async_digest=False)
        aux.main_endpoint.fail = True
        s1 = mkseg(store, "pre fail", "P0")
        assert aux.ensure_digest(s1) is None
        s_x = mkseg(store, "fail two", "P0")
        aux._cooldown_until = 0.0      # 手动拨回时钟（测试确定性，零 sleep）
        assert aux.ensure_digest(s_x) is None
        assert aux._consec_fails == 2, "两次真实端点失败 = 2 连败"
        aux._cooldown_until = 0.0      # 冷却窗到点（模拟时间流逝，零 sleep）
        aux.main_endpoint.fail = False
        s2 = mkseg(store, "post cooldown", "P0")
        assert aux.ensure_digest(s2) is not None and s2.digest_ready, "冷却到点恢复生成"
        assert aux._consec_fails == 0, "成功复位连败计数"
        assert aux._cooldown_until == 0.0, "成功复位冷却窗"

    def test_digest_semaphore_caps_concurrency(self, store):
        """去抖第 2 层：digest_sem 在端点调用期间独占（慢端点下第二调用无法并发）。"""
        import threading
        in_call = threading.Event()
        release = threading.Event()
        sem = threading.BoundedSemaphore(1)

        class GateEndpoint(FakeEndpoint):
            def call(self, task, text):
                self.calls.append((task, text))
                in_call.set()
                release.wait(2.0)   # 模拟慢端点：持锁阻塞直至主线程确认独占
                return ""

        ep = GateEndpoint()
        aux = AuxProvider(ep, digest_sem=sem, async_digest=False)
        s = mkseg(store, "sem probe", "P0")
        done = []

        def worker():
            done.append(aux.ensure_digest(s))

        t = threading.Thread(target=worker)
        t.start()
        assert in_call.wait(2.0), "worker 应已进入端点调用"
        # 端点调用进行中 → 并发帽（max=1）被 digest 独占，第二调用拿不到许可
        assert sem.acquire(blocking=False) is False, "digest 调用期间并发帽被独占"
        release.set()               # 放行端点 → worker finally 归还自身许可
        t.join(2.0)
        assert done and done[0] is not None and s.digest_ready

    def test_async_digest_background_nonblocking(self, store):
        """异步：end_turn/ensure_digest 入队立即返回（调用线程零阻塞），后台生成 + flush 汇合。"""
        import time
        ep = FakeEndpoint()
        aux = AuxProvider(ep, fail_cooldown=0.0, async_digest=True)
        s1 = mkseg(store, "async batch one", "P0")
        s2 = mkseg(store, "async batch two", "P0")
        aux.start_turn()
        assert aux.ensure_digest(s1) is None, "turn 内入队 → None"
        assert aux.ensure_digest(s2) is None, "turn 内入队 → None"
        t0 = time.monotonic()
        aux.end_turn()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, f"end_turn 应立即返回（非阻塞），实测 {elapsed:.3f}s"
        # 未 flush 前可能尚未生成（后台竞态）—— flush 汇合后必 ready
        assert aux.flush(timeout=5.0)
        assert s1.digest_ready and s2.digest_ready, "flush 后批次段全 ready"
        assert aux._dropped == 0

    def test_async_single_shot_flush_ready(self, store):
        """出 turn 单发：async 入队立即返回 None，flush 后 ready；已缓存段同步返回结果。"""
        ep = FakeEndpoint()
        aux = AuxProvider(ep, fail_cooldown=0.0, async_digest=True)
        s = mkseg(store, "single shot", "P1")
        r = aux.ensure_digest(s)
        assert r is None, "async 单发入队 → 立即 None（非同步生成）"
        assert aux.flush(timeout=5.0)
        assert s.digest_ready
        r2 = aux.ensure_digest(s)
        assert r2 is not None and r2["digest_ref_id"] is not None, "已生成段同步返回缓存"
        assert len(aux.main_endpoint.calls) == 1, "幂等：同段只发一次端点调用"

    def test_async_end_to_end_distill(self, store):
        """Distiller + async aux：turn 内 P0 digest 排队 → end_turn 入队 → flush → 可逐出。"""
        ep = FakeEndpoint()
        aux = AuxProvider(ep, fail_cooldown=0.0, async_digest=True)
        p0s = [mkseg(store, "user: ship v1", "P0", "user", structural=True)]
        lp0 = mkseg(store, "llm p0 " * 40, "P0")
        p2 = mkseg(store, "dump " * 40, "P2")
        w = mk_window(100, p0s + [lp0, p2])
        # turn 内 P0 逐出需 digest ready：先走一遍 turn 末批次让它后台生成
        aux.start_turn()
        aux.ensure_digest(lp0)
        aux.end_turn()
        assert aux.flush(timeout=5.0)
        assert lp0.digest_ready
        d = Distiller(w, store, aux).run()
        assert d.status == "ok"
        evicted = [x.segment for x in w.segs if x.form == "digest_only"]
        assert evicted, "digest ready → 至少一段被逐出"

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
