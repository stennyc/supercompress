"""Block ① distill + P0 gate + tier-aware fsync + gate⑤ 钩子门控（①-slice）。

非平凡前置纪律（@reviewer）：每条 0-loss / 预算 / 断电测试都先断言「状态真到达」，防假绿。
①-slice GREEN bar：① 侧 7 条 distill/eviction/fail-safe + marker 双向（off-by-default）
+ 闭合非空前置 + 单一 scdata 权威实现委托 + 测试替身（FakeEndpoint）不进 prod import 面。
"""
import pytest

from sc.schema import Segment
from sc.store import JsonlStore
from sc.distill import Distiller
from helpers import mkseg, complete_digest, mk_window, TOPIC, MV


# ─────────────────────────── Block ① distill + P0 gate + tier-aware fsync ───────────────────────────
class TestBlock1_Distill:
    def test_p0_point_lookup_0loss(self, store, aux):
        p0s = [mkseg(store, f"decision {i}: keep path /srv/a{i}", "P0", "user", structural=True)
               for i in range(2)]
        p0_llm = mkseg(store, "llm p0 fact block " * 50, "P0")
        complete_digest(aux, p0_llm)
        p1s = [mkseg(store, f"ctx block {i} " * 60, "P1") for i in range(3)]
        p2s = [mkseg(store, f"dump block {i} " * 60, "P2") for i in range(2)]
        w = mk_window(1000, p0s + [p0_llm] + p1s + p2s)
        before = w.used_tokens
        # 非平凡前置：真到 80% 水位 + 真逐出过，否则 0-loss 是假绿
        assert before >= 0.8 * w.capacity_tokens, "窗口必须真到达 80% 水位"
        d = Distiller(w, store, aux).run()
        assert len(d.dropped) >= 1, "必须真逐出 P1/P2，0-loss 才有意义"
        assert w.used_tokens < before
        for p in p0s + [p0_llm]:
            got = store.get(p.segment_id)
            assert got is not None and got.original_text == p.original_text, "P0 点查 0-loss"

    def test_p0_eviction_blocked_when_digest_pending(self, store, aux):
        """旧锚更新（§4.5 90% 超压强制面 owner 拍板 2026-10-17）：

        cap=1000 单段 7000 tok（700% over）→ 90% 超压强制面开 →
        digest 未 ready 的 P0 走 head80 占位存根逐出（不再 backpressure）。
        旧锚"turn 内 digest 未 ready → 逐不出"被 90% 面的绕 digest 闸取代。
        """
        p0 = mkseg(store, "p0 pending fact " * 500, "P0")   # 7000 tok：90% 超压（cap 1000）
        aux.start_turn()
        assert aux.ensure_digest(p0) is None, "turn 内 digest 入队未 ready"
        w = mk_window(1000, [p0])
        d = Distiller(w, store, aux).run()
        assert "tier90" in d.events, "90% 超压强制面触发（纯超压驱动，aux 健康）"
        assert w.segs[0].form == "digest_only", "digest 未 ready 的 P0 在 90% 超压时走 head80 占位存根逐出"
        assert p0.digest_text and len(p0.digest_text) >= 80, "head80 占位摘要（原文头 80 字符）"
        aux.end_turn()  # digest 生成
        Distiller(w, store, aux).run()
        assert w.segs[0].form == "digest_only", "digest ready 后 P0 仍 digest_only（幂等）"

    def test_p0_evictable_only_when_digest_ready(self, store, aux):
        p0 = mkseg(store, "p0 ready fact " * 500, "P0")     # 7000 tok：真压穿水位
        complete_digest(aux, p0)
        assert p0.digest_ready
        w = mk_window(1000, [p0])
        d = Distiller(w, store, aux).run()
        assert w.segs[0].form == "digest_only"
        assert any(m.segment_id == p0.segment_id for m in d.anchor_manifest)

    def test_structural_p0_and_ledger_tail_never_evicted(self, store, aux):
        """旧锚更新（§4.5 90% 超压强制面）：cap=100 used≈335（335%）→ 90% 超压面开。

        - user 指令 P0（kind=segment，structural）：90% 超压时放开可逐（owner 拍板 2026-10-17）
        - ledger 尾 P0（kind=ledger，structural 结构地板）：永不逐
        - 非结构 P0/P1/P2：走 ③/⑤/90% 面正常逐
        status：90% 面可逐 user 指令 P0 + 非结构段 → 不再 backpressure（窗口压到 0.9cap 下）
        """
        u = mkseg(store, "user: ship it and keep /srv/x " * 12, "P0", "user", structural=True)
        led = mkseg(store, "ledger:" * 45, "P0", "user", structural=True, kind="ledger",
                    goal="ship v1", phase_id="p1")
        lp0 = mkseg(store, "llm p0 " * 30, "P0")
        complete_digest(aux, lp0)
        p1 = mkseg(store, "p1 ctx " * 30, "P1")
        p2 = mkseg(store, "p2 dump " * 30, "P2")
        w = mk_window(100, [u, led, lp0, p1, p2])
        d = Distiller(w, store, aux).run()
        forms = {s.segment_id: s.form for s in w.segs}
        assert "tier90" in d.events, "90% 超压强制面触发"
        assert forms[led.segment_id] == "resident", "ledger 尾（结构地板）永不逐"
        assert forms[u.segment_id] == "digest_only", "user 指令 P0 90% 超压放开可逐（head80 存根）"

    def test_p0_fsync_before_anchor_fail_safe(self, store, aux):
        p0 = Segment(kind="segment", tier="P0", provenance="model",
                     original_text="llm fact " * 100, session_topic=TOPIC,
                     model_version=MV, structural_p0=False)
        p2 = mkseg(store, "dump tail", "P2")
        w = mk_window(100, [p0, p2])
        store.arm_write_failure()
        d = Distiller(w, store, aux).run()
        assert d.status == "compress_failed"
        assert d.anchor_manifest == [], "源写失败 → 锚点不发射（fail-safe：宁可不压，不可断锚）"
        assert all(s.form == "resident" for s in w.segs), "窗口一个字节不动"
        assert store.fsynced_p0_count == 0

    def test_compression_failed_on_source_unavailable(self, store, aux):
        p0 = Segment(kind="segment", tier="P0", provenance="model",
                     original_text="llm fact " * 100, session_topic=TOPIC,
                     model_version=MV, structural_p0=False)
        w = mk_window(100, [p0])
        store.arm_write_failure()
        d = Distiller(w, store, aux).run()
        assert d.status == "compress_failed"
        assert "source_unavailable" in d.events
        assert store.line_count == 0, "源不可用 → 窗口原样、不产生任何行"

    def test_p1p2_lazy_batch_no_fsync(self, store):
        mkseg(store, "ctx " * 50, "P1")
        mkseg(store, "dump " * 50, "P2")
        assert store.fsynced_p0_count == 0, "P1/P2 不进 P0 fsync 路径"
        assert store.pending_count == 2 and store.pending_p1p2_bytes > 0
        mkseg(store, "decision: keep C:/proj/a", "P0", "user", structural=True)
        assert store.fsynced_p0_count == 1, "P0 即写即 fsync"
        assert store.pending_count == 2, "P1/P2 仍 lazy（不逐行 fsync）"
        store.flush()
        assert store.pending_count == 0

    def test_backpressure_only_on_physical_full(self, store, aux):
        """旧锚更新（§4.5 90% 超压强制面）：cap=100 u≈270（270% over）→ 90% 超压面开。

        user 指令 P0 走 head80 占位存根逐出 → 不再 backpressure（§4.5 的新行为）。
        """
        p2s = [mkseg(store, f"fill {i} " * 40, "P2") for i in range(3)]
        d1 = Distiller(mk_window(100, p2s), store, aux).run()
        assert d1.status == "ok", "80% 是软目标不是硬闸：可暂越、不 backpressure"
        u = mkseg(store, "user: keep /srv/x " * 60, "P0", "user", structural=True)
        d2 = Distiller(mk_window(100, [u]), store, aux).run()
        assert "tier90" in d2.events, "90% 超压强制面触发"
        assert d2.status == "ok", "90% 超压时 user 指令 P0 折存根逐出 → 窗口压下、不 backpressure"

    def test_transient_fail_single_event_self_heal(self, store, aux):
        """旧锚更新（§4.5 90% 超压强制面 + S1 降档带 owner 拍板 2026-10-17）：

        第一 run（transient）：单事件 skip，P1/P2 停止逐出（over-store 可接受），
        P0 不逐（transient regime 跳过 tiering）。
        第二 run（自愈）：transient 解除 → 正常 tiering 跑。used≈124% cap > 90%
        → tier90 p0_force 逐 P0（user 指令 P0 kind=segment 放开可逐 + 非结构 P0），
        used 压到 75% 下 → tier75 跳过（used 已低于 75% 线）。P2 段在本场景
        不逐出（tier90 已 overshoot 到 75% 下，降档带"压到位即停"语义）。
        """
        ep = aux.main_endpoint
        ep.fail = True
        p0s = [mkseg(store, "user: ship v1", "P0", "user", structural=True)]
        lp0 = mkseg(store, "llm p0 " * 40, "P0")
        p2 = mkseg(store, "dump " * 40, "P2")
        w = mk_window(100, p0s + [lp0, p2])
        d = Distiller(w, store, aux).run()
        assert "transient_skip" in d.events, "单事件 tiering 跳过"
        assert w.segs[2].form == "resident", "P1/P2 停止逐出（over-store 可接受）"
        assert d.status == "backpressure"
        assert store.kpi_log[-1]["regime"] == "transient_fail"
        ep.fail = False
        d2 = Distiller(w, store, aux).run()  # 下一事件自愈
        assert w.segs[1].form == "digest_only", "非结构 P0 自愈后 tier90 p0_force 逐出"
        assert w.segs[0].form == "digest_only", "user 指令 P0（kind=segment）§4.5 90% 超压放开可逐"
        assert w.segs[2].form == "resident", \
            "P2 段不逐出（tier90 已把 used 压到 75% 线下，降档带'压到位即停'→ tier75 跳过）"


# ─────────────────────────── gate ⑤ 钩子门控（marker 双向可控）───────────────────────────
class TestBlock5_Hook:
    def test_inject_fault_hook_noop_in_prod(self, store, aux, monkeypatch):
        monkeypatch.delenv("SC_INJECT_FAULT", raising=False)
        p0 = mkseg(store, "user: keep /srv/a", "P0", "user", structural=True)
        p1 = mkseg(store, "ctx " * 40, "P1")
        w = mk_window(100, [p0, p1])
        d = Distiller(w, store, aux, inject_fault_after="p1p2_pending").run()
        assert d.fault_state is None, "marker off → 参数被忽略（生产恒 inert）"
        assert not any(e.startswith("fault:") for e in d.events)
        assert store.dangling_anchors(d.anchor_manifest) == []

    def test_marker_on_bidirectional(self, store, aux, monkeypatch):
        monkeypatch.setenv("SC_INJECT_FAULT", "1")
        p0 = mkseg(store, "user: keep /srv/a", "P0", "user", structural=True)
        p1 = mkseg(store, "ctx " * 40, "P1")
        w = mk_window(100, [p0, p1])
        d = Distiller(w, store, aux, inject_fault_after="p1p2_pending").run()
        assert d.fault_state is not None, "marker on → 注入真生效（门控双向可控）"
        assert any(e.startswith("fault:") for e in d.events)

    def test_power_loss_mid_write_replay(self, store, aux, monkeypatch):
        monkeypatch.setenv("SC_INJECT_FAULT", "1")
        p0 = mkseg(store, "goal: ship v1", "P0", "user", structural=True)
        p1 = mkseg(store, "ctx " * 60, "P1")
        p2 = mkseg(store, "dump " * 60, "P2")
        w = mk_window(100, [p0, p1, p2])
        d = Distiller(w, store, aux, inject_fault_after="p1p2_pending").run()
        # 非平凡前置：混合态真建成（P0 已 fsync 且 P1/P2 真 pending），否则断电断言无意义
        assert d.fault_state["fsynced_p0"] > 0
        assert d.fault_state["pending_p1p2_bytes"] > 0
        s2 = JsonlStore(store.path)
        ids = {s.segment_id for s in s2.replay()}
        assert p0.segment_id in ids, "已 durable 的 P0 零丢失"
        lost = [s for s in (p1, p2) if s.segment_id not in ids]
        lost_bytes = sum(len(x.original_text) for x in lost)
        assert lost_bytes <= store.lazy_batch_cap + 1024, "断电丢失有界（一个 lazy 批）"
        assert s2.dangling_anchors(d.anchor_manifest) == [], "零悬空锚点"

    def test_power_loss_gate_boundary(self, store, aux, monkeypatch):
        monkeypatch.setenv("SC_INJECT_FAULT", "1")
        p0 = mkseg(store, "goal: ship v1", "P0", "user", structural=True)
        p1 = mkseg(store, "ctx " * 40, "P1")
        p2 = mkseg(store, "dump " * 40, "P2")
        w = mk_window(100, [p0, p1, p2])
        d = Distiller(w, store, aux, inject_fault_after="gate_boundary").run()
        s2 = JsonlStore(store.path)
        ids = {s.segment_id for s in s2.replay()}
        for x in (p0, p1, p2):
            assert x.segment_id in ids, "gate 边界：批已 flush durable → 断电无损失"
        assert s2.dangling_anchors(d.anchor_manifest) == []
