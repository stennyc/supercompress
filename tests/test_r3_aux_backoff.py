"""R3（报告#3）：aux 端点健康探测/降级 + P0 digest pending 兜底策略。

覆盖面（任务 t_d94cc07e @reviewer 单测钉死）：
1. 连败退避曲线：默认阶梯 = fail_cooldown × COOLDOWN_BACKOFF_MULTIPLIERS
   （60s 配置 → 60→300→1800s 封顶）；连败 N → 冷却窗 = ladder[min(N-1, len-1)]；
   阶梯可注入（零硬编码模型/provider）；fail_cooldown=0 → 阶梯全 0 → 冷却窗恒关。
2. 成功复位：端点 1 次成功 → 连败计数归零、冷却窗回落基线；再连败从第 1 档重启。
3. 逃生门（P0 pending 兜底）：端点持续劣化（consec ≥ 3）+ 窗口超 force_ratio 压力
   + 存在 pending 非结构 P0 → P0 head80 占位摘要逐出（事件钉 p0_pending_escape，
   观测面不静默）；未劣化 / 未超压 / 正常路径 0 行为变化（回归钉 test_block1/4）。

测试时钟策略：端点替身恒 fail/恒成功；连败推进用「手动把 _cooldown_until 拨回 0」
（冷却窗到点语义），不真 sleep——曲线/复位/逃生门全确定性。零新 endpoint/provider。
"""
import threading

from fakes import FakeEndpoint
from helpers import mkseg, mk_window
from sc.aux_provider import (
    AuxProvider,
    COOLDOWN_BACKOFF_MULTIPLIERS,
    effective_cooldown,
)
from sc.distill import Distiller


def _sem():
    return threading.BoundedSemaphore(2)


class TestCooldownBackoffCurve:
    """连败退避曲线：默认阶梯派生 + 曲线封顶 + 零 fail_cooldown 恒关 + 阶梯可注入。"""

    def test_default_ladder_derivation(self):
        """默认阶梯 = fail_cooldown × 乘数（60s 配置 → 60→300→1800s 契约曲线）。"""
        assert COOLDOWN_BACKOFF_MULTIPLIERS == (1.0, 5.0, 30.0), "退避乘数契约不得漂移"
        aux = AuxProvider(FakeEndpoint(), fail_cooldown=60.0,
                          digest_sem=_sem(), async_digest=False)
        assert aux.cooldown_backoff == (60.0, 300.0, 1800.0), "默认阶梯派生（×1/×5/×30）"

    def test_curve_capped(self):
        """曲线 = ladder[min(n-1, len-1)]：1→60, 2→300, 3→1800（封顶），此后恒 1800。"""
        aux = AuxProvider(FakeEndpoint(), fail_cooldown=60.0,
                          digest_sem=_sem(), async_digest=False)
        assert [effective_cooldown(aux, n) for n in (1, 2, 3, 4, 5)] == \
            [60.0, 300.0, 1800.0, 1800.0, 1800.0], "退避曲线（尾档封顶）不符"

    def test_zero_fail_cooldown_inert(self, store):
        """fail_cooldown=0 → 阶梯全 0 → 冷却窗恒关（sc 契约「transient 单事件自愈」0 行为变化）。"""
        ep = FakeEndpoint()
        ep.fail = True
        aux = AuxProvider(ep, fail_cooldown=0.0, digest_sem=_sem(), async_digest=False)
        assert aux.cooldown_backoff == (0.0, 0.0, 0.0), "零配置阶梯全 0"
        s = mkseg(store, "zero fc probe", "P0")
        assert aux.ensure_digest(s) is None, "端点失败 → 仍 transient"
        assert aux._consec_fails == 1, "连败计数照常统计（即使冷却窗关）"
        assert aux._cooldown_until == 0.0, "fail_cooldown=0 → 冷却窗恒关（sc 契约 0 变化）"
        assert effective_cooldown(aux, 1) == 0.0

    def test_custom_ladder_injectable(self):
        """阶梯可注入（零硬编码）：显式 (10, 50) → 1→10, 2→50, 3→50（尾档封顶）。"""
        aux = AuxProvider(FakeEndpoint(), fail_cooldown=60.0,
                         cooldown_backoff=(10.0, 50.0),
                         digest_sem=_sem(), async_digest=False)
        assert aux.cooldown_backoff == (10.0, 50.0)
        assert [effective_cooldown(aux, n) for n in (1, 2, 3)] == [10.0, 50.0, 50.0]

    def test_no_self_amplification(self, store):
        """冷却窗内跳段 ≠ 端点失败：计数只统计真实端点调用失败（退避不自我放大）。"""
        ep = FakeEndpoint()
        ep.fail = True
        aux = AuxProvider(ep, fail_cooldown=60.0, digest_sem=_sem(), async_digest=False)
        s1 = mkseg(store, "first fail", "P0")
        assert aux.ensure_digest(s1) is None and aux._consec_fails == 1
        s2 = mkseg(store, "cool seg two", "P0")
        s3 = mkseg(store, "cool seg three", "P0")
        assert aux.ensure_digest(s2) is None, "冷却窗内第 2 段跳过（零端点调用）"
        assert aux.ensure_digest(s3) is None, "冷却窗内第 3 段跳过（零端点调用）"
        assert aux._consec_fails == 1, "冷却窗内跳段不得累加连败计数"
        assert len(ep.calls) == 1, "端点恰好被调一次（非每段重试）"


class TestCooldownRecovery:
    """端点 1 次成功 → 连败计数归零、冷却窗回落基线；再连败从第 1 档（基线）重启。"""

    def _flapping(self):
        ep = FakeEndpoint()
        ep.fail = True
        aux = AuxProvider(ep, fail_cooldown=60.0, digest_sem=_sem(), async_digest=False)
        return ep, aux

    def test_success_resets_counter_and_cooldown(self, store):
        ep, aux = self._flapping()
        s1 = mkseg(store, "pre recover", "P0")
        assert aux.ensure_digest(s1) is None and aux._consec_fails == 1
        aux._cooldown_until = 0.0      # 冷却窗到点（零 sleep）
        ep.fail = False
        s2 = mkseg(store, "recovered", "P0")
        assert aux.ensure_digest(s2) is not None and s2.digest_ready, "成功段必须正常生成"
        assert aux._consec_fails == 0, "1 次成功 → 连败计数归零"
        assert aux._cooldown_until == 0.0, "成功 → 冷却窗关闭回落基线"

    def test_next_failure_restarts_from_base(self, store):
        """复位后再连败：从阶梯第 1 档（基线 fail_cooldown）重新开始，非上一轮高位。"""
        ep, aux = self._flapping()
        s1 = mkseg(store, "flap 1", "P0")
        assert aux.ensure_digest(s1) is None
        aux._cooldown_until = 0.0
        ep.fail = False
        s2 = mkseg(store, "flap 2", "P0")
        assert aux.ensure_digest(s2) is not None, "中间成功 → 复位"
        ep.fail = True
        s3 = mkseg(store, "flap 3", "P0")
        assert aux.ensure_digest(s3) is None and aux._consec_fails == 1, "复位后连败从第 1 档重启"
        assert aux._cooldown_until > 0.0, "再连败 → 冷却窗（基线档）重开"
        assert effective_cooldown(aux, aux._consec_fails) == 60.0, "重启从基线档（非上一轮高位）"


class TestP0PendingEscapeHatch:
    """P0 逐出 gate 逃生门（R3 pending 兜底）：端点持续劣化（consec ≥ 3）+ 窗口超
    force_ratio 压力 + 存在 pending 非结构 P0 → P0 head80 占位摘要逐出（事件钉
    p0_pending_escape，观测面不静默）。未劣化 / 未超压 / 正常路径 0 行为变化。"""

    def _degraded_aux(self, store):
        """fail_cooldown=0（冷却关、consec 照常）+ 3 连败 → endpoint_degraded=True。"""
        ep = FakeEndpoint()
        ep.fail = True
        aux = AuxProvider(ep, fail_cooldown=0.0, digest_sem=_sem(), async_digest=False)
        for i in range(3):
            probe = mkseg(store, "streak probe %d" % i, "P0")
            aux.ensure_digest(probe)          # 真实端点失败 → consec 推进到 3
        return ep, aux

    def _overpressure(self, store):
        """4 个 pending P0（各 320 字符=80 tok）+ 1 P2 填充（320 字符=80 tok）= used 400。
        cap 400 × force_ratio 0.75 = 300 → used(400) > 300 超压。"""
        p0s = [mkseg(store, "p0 " + "x" * 317, "P0") for _ in range(4)]  # 各 320 字符 = 80 tok
        p2 = mkseg(store, "n" * 320, "P2")                                # 320 字符 = 80 tok
        w = mk_window(400, p0s + [p2])
        assert w.used_tokens == 400, "前置 token 账（4×80 P0 + 80 P2 = 400）"
        assert w.used_tokens > int(w.capacity_tokens * 0.75), "前置：窗口超 force_ratio(0.75)"
        return w, p0s, p2

    def test_escape_fires_when_degraded_and_overpressure(self, store):
        """consec≥3 + 超压 + pending P0 → 逃生门开：P0 head80 占位摘要逐出（事件留痕）。"""
        ep, aux = self._degraded_aux(store)
        assert aux.endpoint_degraded, "3 连败 → 端点判持续劣化（consec ≥ 3）"
        w, p0s, p2 = self._overpressure(store)
        d = Distiller(w, store, aux, force_ratio=0.75).run()
        assert "p0_pending_escape" in d.events, "逃生门触发必须留事件观测面（非静默）"
        forms = {ws.segment_id: ws.form for ws in w.segs}
        evicted_p0 = [p for p in p0s if forms[p.segment_id] == "digest_only"]
        assert evicted_p0, "超压 + 劣化 + pending P0 → 至少一个 P0 占位摘要逐出"
        for p in evicted_p0:
            assert p.digest_text == p.original_text[:80], "逃生门逐出走 head80 占位摘要"
        assert w.used_tokens < int(0.8 * w.capacity_tokens), "逃生门逐出后窗口回落软水位(0.8)"

    def test_escape_blocked_below_pressure(self, store):
        """used ≤ cap×force_ratio → 逃生门关（即使端点劣化）：P0 pending 保持 resident。"""
        ep, aux = self._degraded_aux(store)
        assert aux.endpoint_degraded
        p0a = mkseg(store, "below " + "y" * 316, "P0")   # 320 字符 = 80 tok
        p2 = mkseg(store, "m" * 40, "P2")                 # 40 字符 = 10 tok
        w = mk_window(2000, [p0a, p2])
        assert w.used_tokens <= int(w.capacity_tokens * 0.75), "前置：窗口未超 force_ratio"
        d = Distiller(w, store, aux, force_ratio=0.75).run()
        assert w.segs[0].form == "resident", "未超压 → P0 pending 保持 resident（原 gate 不变）"
        assert "p0_pending_escape" not in d.events, "未超压 → 逃生门关"
        assert d.status == "ok"

    def test_single_transient_not_degraded_stays_resident(self, store):
        """单一瞬态事件（窗口仅 1 个 pending P0 → consec=1 < 3）→ 端点未判劣化 →
        逃生门关：P0 pending 保持 resident（回归钉 test_block1 transient 单事件自愈口径）。"""
        ep = FakeEndpoint()
        ep.fail = True
        aux = AuxProvider(ep, fail_cooldown=0.0, digest_sem=_sem(), async_digest=False)
        # 窗口恰好 1 个 pending P0 → 本事件 step2 端点失败 1 次 → consec=1（未达 3）
        p0 = mkseg(store, "llm p0 " * 200, "P0")          # 1400 字符 = 350 tok
        p2 = mkseg(store, "n" * 40, "P2")                  # 10 tok
        w = mk_window(400, [p0, p2])
        assert w.used_tokens > int(w.capacity_tokens * 0.75), "前置：窗口超 force_ratio"
        d = Distiller(w, store, aux, force_ratio=0.75).run()
        assert aux.consec_fails == 1, "单一瞬态事件 = 1 连败（未达劣化阈值 3）"
        assert not aux.endpoint_degraded, "consec=1 < 3 → 未判持续劣化"
        assert "p0_pending_escape" not in d.events, "单一瞬态（未劣化）→ 逃生门关（回归钉）"
        assert w.segs[0].form == "resident", "P0 pending 保持 resident（原 gate 不变）"

    def test_normal_path_no_escape(self, store):
        """回归钉：端点恒成功 → P0 拿到真 digest 正常逐出——无逃生门事件、0 行为变化。"""
        ep = FakeEndpoint()      # 恒成功
        aux = AuxProvider(ep, fail_cooldown=0.0, digest_sem=_sem(), async_digest=False)
        w, p0s, p2 = self._overpressure(store)
        d = Distiller(w, store, aux, force_ratio=0.75).run()
        assert not aux.endpoint_degraded, "端点恒成功 → 未劣化"
        assert "p0_pending_escape" not in d.events, "正常路径 0 逃生门行为"
        forms = {ws.segment_id: ws.form for ws in w.segs}
        assert any(forms[p.segment_id] == "digest_only" for p in p0s), \
            "digest ready P0 经正常路径逐出（水位逐出至 80%）"
