"""B 域宿主契约收口：token 哨兵 / compact 后 latch 生命周期对齐内置 ContextCompressor。

reviewer（独立子代理 sa-0-eb1ba3ae）逐码坐实的 4 条 shell 侧 bug（宿主 agent/ 零 diff，
只改 engine.py 契约面 + 本测试）：
1. **compact 后自动/idle compact 永久跳过**：宿主 _compress_context 提交后置位
   awaiting_real_usage_after_compression（conversation_compression.py:3496-3501），但壳
   update_from_response 从不清 → turn_preflight.py:273 / turn_context_compaction.py:163 的
   `if awaiting: return` skip 分支每轮命中，auto/idle compact 锁死（只 manual /compress 绕过）。
2. **/new（on_session_reset）不清该 latch**：旧 session 压缩后态泄漏进新 session 首 turn，
   同样锁死，只有进程重启/重建实例才解除。
3. **display source 口径反转**：壳缺 last_real_prompt_tokens 属性 → context_breakdown.
   context_display_source 的 getattr(...,None) 得 None → isinstance(None,(int,float)) 判否 →
   对壳恒返回 provider_usage，粗估态被伪装成 provider 真实读数（桌面 meter 永不出 `~`）。
4. **update_from_response({}) 冲掉 -1 哨兵**：provider 某响应无 usage 时宿主调 ({})（turn_
   usage.py:87-91），壳裸 int(get(...,0)) 把宿主刚写的 last_prompt_tokens=-1 覆盖成 0，
   破坏 #36718 哨兵语义（内置 ContextCompressor 经 rollback_interrupted_preflight_display_tokens
   对 -1 有保护，壳无）。

覆盖（新增断言镜像 tests/test_rotation_contract.py + tests/test_session_store.py 锚点口径）：
- 验收#1：awaiting=True + 一次 update_from_response({prompt_tokens:<threshold}) → 该 latch 被清
- 验收#2：on_session_reset 后 latch 归位 + 4 字段仍全 0（不回归 test_session_store 既有断言）
- 验收#3：last_prompt_tokens=-1 时 update_from_response({}) 不冲成 0（哨兵保留）
- 验收#4：last_real_prompt_tokens 真实 prompt>0 时写入 + context_breakdown.context_display_source
  对壳在粗估态能判 local_estimate（打 ~）而非恒 provider_usage
- 对称面：on_session_start（rebind/轮转）同样清 compact 后一次性 latch（与 P0.4 advisory 清零同口径）

跑法（repo 根）：
  python -m pytest tests/test_token_sentinel_contract.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engine import SupercompressContextEngine

# 宿主 context_breakdown 源目录仅在本机 hermes-agent 源树在场时可达（远端 117 CI 未必有）；
# 按 test_hermes_shell.py 同款优雅降级：可达则验 display source 判别器，不可达则该条 skipif。
try:  # noqa: S403
    from agent.context_breakdown import context_display_source as _host_display_source
except Exception:  # pragma: no cover - 宿主源树缺席
    _host_display_source = None


def _mkengine(tmp_path: Path, context_length: int = 256000) -> SupercompressContextEngine:
    """独立隔离的 engine 实例（external_store_path → tmp，不碰全局 active.jsonl）。

    context_length 默认 256000 → threshold_tokens = 0.8*256000 = 204800（sc.distill 软水位）；
    验收场景用 prompt_tokens < threshold（compact 成功后 prompt 回落）。"""
    return SupercompressContextEngine(context_length=context_length,
                                      external_store_path=str(tmp_path))


def _arm_compact_commit(e: SupercompressContextEngine) -> None:
    """复刻宿主 _compress_context 提交（conversation_compression.py:3496-3501）对壳实例写的
    两件 compact 后一次性态：last_prompt_tokens=-1 哨兵 + awaiting_real_usage latch。
    （宿主 set_usage_anchor(agent,None) 清的是 agent 侧锚点，非壳字段，本契约面不涉及。）"""
    e.last_prompt_tokens = -1
    e.awaiting_real_usage_after_compression = True


# ───────────────────────── 验收#1：update_from_response 清 compact 后 latch ─────────────────────────

def test_update_from_response_clears_compact_latch(tmp_path):
    """awaiting=True + 一次真实 response（prompt<threshold）→ 该 latch 被清（skip 分支不再命中）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_await")
    _arm_compact_commit(e)
    assert e.awaiting_real_usage_after_compression is True
    e.update_from_response({"prompt_tokens": 41000, "completion_tokens": 300})
    assert e.awaiting_real_usage_after_compression is False, \
        "真实 usage 到达必须清 compact 后一次性 latch（否则 auto/idle compact 永久 skip）"
    # token 桶照常写入（契约消费不破坏既有 token 账面）
    assert e.last_prompt_tokens == 41000
    assert e.last_total_tokens == 41300


def test_update_from_response_consumes_latch_even_without_prompt(tmp_path):
    """无 prompt_tokens 的一次 usage（如 provider 只回 completion）也消费 latch（防无限 latched）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_await2")
    _arm_compact_commit(e)
    e.update_from_response({"completion_tokens": 250})
    assert e.awaiting_real_usage_after_compression is False


# ───────────────────────── 验收#3：{} 不冲 -1 哨兵 ─────────────────────────

def test_empty_usage_preserves_minus_one_sentinel(tmp_path):
    """last_prompt_tokens=-1（compact 哨兵）+ update_from_response({})（no-usage）→ 仍 -1。

    此前裸 int(get('prompt_tokens',0)) 把 -1 冲成 0，破坏 #36718 哨兵（preflight 的
    last_prompt_tokens == -1 分支 + 宿主 meter 粗估判据都会失真）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_sentinel")
    _arm_compact_commit(e)
    e.update_from_response({})  # 宿主 turn_usage.py:87-91 no-usage 路径
    assert e.last_prompt_tokens == -1, "no-usage 不得冲掉 -1 哨兵（保留宿主 compact 后态）"
    # 全保留既有桶（含 -1），仅消费 latch：
    assert e.awaiting_real_usage_after_compression is False, "no-usage 也应消费 latch"
    # completion/total 桶随 no-usage 归 0（无真实 completion 可记），但 -1 prompt 哨兵独立保留：
    assert e.last_completion_tokens == 0
    # 下一次真实 response 才落 prompt 桶 + 记账 real：
    e.update_from_response({"prompt_tokens": 38000, "completion_tokens": 100})
    assert e.last_prompt_tokens == 38000


def test_empty_usage_keeps_zero_state_clean(tmp_path):
    """fresh engine（last_prompt_tokens=0，非哨兵）+ ({}) → 保持 0，哨兵无关。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_zero")
    e.update_from_response({})
    assert e.last_prompt_tokens == 0
    assert e.last_total_tokens == 0


# ───────────────────────── 验收#4：last_real_prompt_tokens 记账 + display source 判别 ─────────────────────────

def test_last_real_prompt_tokens_written_on_real_reading(tmp_path):
    """真实 prompt_tokens>0 → last_real_prompt_tokens 记账（属性存在且为 int，不再 getattr 得 None）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_real")
    assert e.last_real_prompt_tokens == 0, "fresh 默认 0（= 尚无 provider 真实读数）"
    e.update_from_response({"prompt_tokens": 41000, "completion_tokens": 300})
    assert e.last_real_prompt_tokens == 41000
    # no-usage ({}) 不得回退 real 账（保留最近一次 provider 真实读数）：
    e.update_from_response({})
    assert e.last_real_prompt_tokens == 41000


def test_display_source_flags_local_estimate_on_rough_state(tmp_path):
    """宿主 context_breakdown.context_display_source 对壳：粗估态（shown 落后于/超出于 real）
    判 local_estimate（打 ~），修复此前「恒 provider_usage」的口径反转。

    判别器口径（context_breakdown.py:100-102）：real 是数 且 shown>0 且 shown!=real → local_estimate。
    修前壳缺 last_real_prompt_tokens（getattr→None）→ 恒 provider_usage，粗估被伪装成真实读数。"""
    if _host_display_source is None:
        pytest.skip("宿主源树缺席（远端 CI 无 agent/ 目录）——本地 hermes-agent 源在场才验判别器")
    e = _mkengine(tmp_path)
    e.on_session_start("s_disp")
    # 粗估态：provider 真实读数 41000 已被壳记账，但当前展示值（last_prompt_tokens）是更高的粗估
    # （宿主 preflight rough estimate seed 场景，turn_context_compaction.py:271）→ 打 ~。
    e.last_real_prompt_tokens = 41000
    e.last_prompt_tokens = 45000
    assert _host_display_source(e) == "local_estimate", \
        "粗估落后/超前于 real 必须判 local_estimate（打 ~ 粗估标记），非 provider_usage"
    # 真实读数态：shown == real → provider_usage（粗估标记撤掉）：
    e.last_prompt_tokens = 41000
    assert _host_display_source(e) == "provider_usage"


def test_display_source_real_reading_lives_on_shell(tmp_path):
    """修前 bug 根面：壳必须暴露 last_real_prompt_tokens 数值属性（宿主 getattr 拿到 int 而非 None）。

    不依赖宿主模块——直接断言属性存在 + 类型 + 与 update_from_response 的记账联动。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_attr")
    real = getattr(e, "last_real_prompt_tokens", None)
    assert real is not None, "壳缺 last_real_prompt_tokens 属性 → 宿主 display 口径反转（bug#3）"
    assert isinstance(real, int)
    e.update_from_response({"prompt_tokens": 52000, "completion_tokens": 40})
    assert e.last_real_prompt_tokens == 52000, "真实 prompt>0 必须记账"


# ───────────────────────── 验收#2：on_session_reset 清 compact 后 latch（会话边界对齐）────────────────────────

def test_session_reset_clears_compact_latch(tmp_path):
    """/new 前 arm compact 后 latch → on_session_reset 必须清（防泄漏进新 session 首 turn）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_reset")
    _arm_compact_commit(e)
    assert e.awaiting_real_usage_after_compression is True
    e.on_session_reset()
    assert e.awaiting_real_usage_after_compression is False, \
        "compact 后一次性 latch 属旧 session，reset 必须清（否则新 session 仍被锁死）"
    assert e.last_real_prompt_tokens == 0, "real 读数账随 session 边界归 0（新 session 无 provider 读数）"
    assert e._verify_compaction_cleared_threshold is False


def test_session_reset_keeps_four_fields_zero(tmp_path):
    """不回归 test_session_store 既有断言：4 字段（prompt/completion/total + compression_count）
    仍全 0（与 latch 清零同函数、同 reset 集）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_reset2")
    e.last_prompt_tokens = 999
    e.last_completion_tokens = 77
    e.last_total_tokens = 1076
    e.compression_count = 3
    _arm_compact_commit(e)
    e.on_session_reset()
    assert e.last_prompt_tokens == 0
    assert e.last_completion_tokens == 0
    assert e.last_total_tokens == 0
    assert e.compression_count == 0


# ───────────────────────── 对称面：on_session_start（rebind/轮转）清 compact 后 latch ─────────────────────────

def test_session_start_clears_compact_latch_on_rebind(tmp_path):
    """compact 后 arm latch → on_session_start（轮转/rebind 到新 sid）必须清（与 P0.4 advisory
    会话边界清零同口径；live 幂等重入不清——本测走的是真正的新 sid rebind 路径）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_old")
    _arm_compact_commit(e)
    assert e.awaiting_real_usage_after_compression is True
    e.on_session_start("s_new")  # 真正的新 session rebind（_bound_session 变）
    assert e.awaiting_real_usage_after_compression is False, \
        "compact 后一次性 latch 不得跨 session rebind 泄漏"
    assert e.last_real_prompt_tokens == 0


def test_session_start_idempotent_reentry_keeps_live_state(tmp_path):
    """幂等重入（同实例同 sid）不清 live 态——契约清零只在真正边界发生（对称既有 on_session_start
    幂等早退语义：重入 = 活会话继续，不重置）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_idem")
    e.update_from_response({"prompt_tokens": 60000, "completion_tokens": 50})
    live_prompt = e.last_prompt_tokens
    live_real = e.last_real_prompt_tokens
    assert live_prompt == 60000 and live_real == 60000
    e.on_session_start("s_idem")  # 幂等重入（早退，不清）
    assert e.last_prompt_tokens == live_prompt, "幂等重入不得清 live 实态"
    assert e.last_real_prompt_tokens == live_real
