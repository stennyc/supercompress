"""cache-aware 决策面（t_d65365d8 · owner 指令「1 要」）：last_cache_hit_ratio 参与
should_compress / should_compress_preflight（压缩触发）+ select_context（召回 K）决策。

红线验收面（本测试封死）：
- fail-safe 零变化：cache_aware_enabled=False（默认）/ ratio None（无 usage / 旧宿主）/
  命中 < hit_min → 决策面恒等于现状（_effective_threshold / k=3），行为零变化。
- 高命中介入：ratio ≥ hit_min → 压缩触发线抬高 base×(1+delay_frac)（推迟压缩省 re-prime）；
  抬高线后基线本会触发但被拦住 → 打 `cache-aware delay compress (hit=NN%)` marker（活体证据）；
  抬高线后仍过线触发 → 打 `cache-aware keep trigger (hit=NN%)` marker。
- 召回 K：高命中 → select_context 收窄到 cache_aware_recall_k（少换 recall 内容）；
  fail-safe → 恒 3（现状硬编码 K，零变化）。
- 阈值/权重全走 config 嵌套块 compression.supercompress.cache_aware_compress.*
  （host kwarg 可覆盖）—— 不硬编码进决策路径。
- 决策面**只读** last_cache_hit_ratio 实例态（t_dc8e6b73 遥测面）—— 不改
  update_from_response / _record_cache_telemetry 存态逻辑。
- select_context 宿主接口签名零改（cache 不作参数；纯读实例态）。

跑法（repo 根或插件根）：
  python -m pytest plugins/context_engine/supercompress/tests/test_cache_aware_decision.py -v
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

import engine as _eng
from engine import (
    SupercompressContextEngine,
    DEFAULT_CACHE_AWARE_ENABLED,
    DEFAULT_CACHE_AWARE_HIT_MIN,
    DEFAULT_CACHE_AWARE_DELAY_FRAC,
    DEFAULT_CACHE_AWARE_RECALL_K,
)
from sc import est_tokens


# ── engine 构造面（hermetic 不读 live config.yaml 嵌套块 → 测试确定性）──
# _load_config() 在测试 env 会读 live config.yaml（可能已配嵌套块）→ monkeypatch 成显式块。
# host/测试 kwarg（cache_aware_* 扁平键）优先级 > 嵌套块 > 常量默认（__init__ 解析路径）。

def _mkengine(tmp_path: Path, monkeypatch, config_block: dict,
              context_length: int = 256000, **kwargs) -> SupercompressContextEngine:
    """独立隔离 engine（external_store_path → tmp）。config_block 直注
    compression.supercompress 子表（monkeypatch _load_config → hermetic）；
    kwarg 面（cache_aware_*）覆盖嵌套块同名键。threshold = context_length×0.55。"""
    monkeypatch.setattr(_eng, "_load_config", lambda: config_block)
    return SupercompressContextEngine(context_length=context_length,
                                       external_store_path=str(tmp_path),
                                       **kwargs)


# 测试用的嵌套块常量（走 owner 指定键名 compression.supercompress.cache_aware_compress.*）
_ON = {"cache_aware_compress": {"enabled": True}}
_OFF = {}  # 无嵌套块 → 全常量默认（enabled=False = fail-safe 关）


# ── 默认值面（fail-safe 红线：全默认 = 现状零变化）──────────────────────────────

def test_defaults_are_fail_safe_baseline():
    """常量默认值面：enabled=False（关）/ hit_min=0.5 / delay_frac=0.10 / recall_k=1。
    默认 enabled=False → 决策面恒走现状路径（零变化）—— fail-safe 红线钉死。"""
    assert DEFAULT_CACHE_AWARE_ENABLED is False, "默认必须关（fail-safe 现状零变化）"
    assert DEFAULT_CACHE_AWARE_HIT_MIN == 0.5
    assert DEFAULT_CACHE_AWARE_DELAY_FRAC == 0.10
    assert DEFAULT_CACHE_AWARE_RECALL_K == 1


def test_nested_block_parses_all_keys(tmp_path, monkeypatch):
    """宿主 config.yaml 嵌套块 cache_aware_compress.{enabled,hit_min,delay_frac,recall_k}
    经 __init__ 真实解析路径（monkeypatch _load_config）→ 逐键带默认规整进实例属性。"""
    block = {"cache_aware_compress": {"enabled": True, "hit_min": 0.7,
                                       "delay_frac": 0.25, "recall_k": 2}}
    e = _mkengine(tmp_path, monkeypatch, block)
    assert e.cache_aware_enabled is True
    assert e.cache_aware_hit_min == 0.7
    assert e.cache_aware_delay_frac == 0.25
    assert e.cache_aware_recall_k == 2


def test_nested_block_missing_key_falls_to_safe_default(tmp_path, monkeypatch):
    """嵌套块存在但缺键 → 该键落常量安全默认（enabled 缺 = False = fail-safe 关）。"""
    e = _mkengine(tmp_path, monkeypatch, {"cache_aware_compress": {"recall_k": 2}})
    assert e.cache_aware_enabled is False, "enabled 缺键必须落默认 False（fail-safe）"
    assert e.cache_aware_recall_k == 2
    assert e.cache_aware_hit_min == DEFAULT_CACHE_AWARE_HIT_MIN
    assert e.cache_aware_delay_frac == DEFAULT_CACHE_AWARE_DELAY_FRAC


def test_no_nested_block_defaults_to_off(tmp_path, monkeypatch):
    """嵌套块缺失（_OFF）→ 全常量默认：enabled=False（fail-safe 关），决策面恒现状。"""
    e = _mkengine(tmp_path, monkeypatch, _OFF)
    assert e.cache_aware_enabled is False
    assert e.cache_aware_hit_min == DEFAULT_CACHE_AWARE_HIT_MIN
    assert e.cache_aware_delay_frac == DEFAULT_CACHE_AWARE_DELAY_FRAC
    assert e.cache_aware_recall_k == DEFAULT_CACHE_AWARE_RECALL_K


def test_kwarg_overrides_nested_block(tmp_path, monkeypatch):
    """host/测试 kwarg（cache_aware_* 扁平键）优先于 config 嵌套块 → 测试确定性直注。"""
    e = _mkengine(tmp_path, monkeypatch, _ON,
                  cache_aware_hit_min=0.6, cache_aware_delay_frac=0.2,
                  cache_aware_recall_k=3)
    assert e.cache_aware_enabled is True      # 嵌套块 enabled（kwarg 未覆盖）
    assert e.cache_aware_hit_min == 0.6       # kwarg 覆盖
    assert e.cache_aware_delay_frac == 0.2
    assert e.cache_aware_recall_k == 3


# ── 压缩触发面 · fail-safe 零变化（关 / None / 低命中）──────────────────────────

def test_config_off_zero_behavior_change(tmp_path, monkeypatch, caplog):
    """cache-aware 关（默认）→ 触发线恒等于现状 _effective_threshold，零 marker（零变化可证）。"""
    e = _mkengine(tmp_path, monkeypatch, _OFF)
    e.on_session_start("s_off")
    base = e._effective_threshold()
    with caplog.at_level(logging.INFO, "engine"):
        assert e._cache_aware_trigger_threshold() == base, "关 → 线不抬高（现状 base）"
        assert e.should_compress(prompt_tokens=base + 1) is True   # 现状触发面
        assert e.should_compress(prompt_tokens=1) is False
    assert not [m for m in caplog.messages if "cache-aware" in m], \
        "config 关 → 零 cache-aware marker（现状路径零日志变化）"
    # 召回 K 亦 fail-safe = 现状 3
    assert e._cache_aware_recall_k() == 3


def test_ratio_none_zero_behavior_change(tmp_path, monkeypatch, caplog):
    """ratio None（无 usage / 旧宿主无 cache 键）+ enabled → 线不抬高（fail-safe 现状）、
    零 marker、召回 K=3（现状）。无 cache 读数 = 无决策介入 = 零变化。"""
    e = _mkengine(tmp_path, monkeypatch, _ON)
    e.on_session_start("s_none")
    assert e.last_cache_hit_ratio is None  # fresh engine 无 usage → None
    base = e._effective_threshold()
    with caplog.at_level(logging.INFO, "engine"):
        assert e._cache_aware_trigger_threshold() == base, "ratio None → 线不抬高"
        assert e.should_compress(prompt_tokens=base + 1) is True, "ratio None → 现状触发面"
        assert e.should_compress(prompt_tokens=1) is False
    assert not [m for m in caplog.messages if "cache-aware" in m], \
        "ratio None 不得打 marker（fail-safe 零变化）"
    assert e._cache_aware_recall_k() == 3


def test_low_hit_below_hit_min_no_lift(tmp_path, monkeypatch, caplog):
    """命中 0.3 < hit_min 0.5 → 线不抬高（维持现状触发）、零 marker、召回 K=3。"""
    e = _mkengine(tmp_path, monkeypatch, _ON)
    e.on_session_start("s_low")
    e.last_cache_hit_ratio = 0.3  # 模拟上 turn 低命中读数（直注实例态，纯读）
    base = e._effective_threshold()
    with caplog.at_level(logging.INFO, "engine"):
        assert e._cache_aware_trigger_threshold() == base, "低命中 < hit_min → 线不抬高"
        assert e.should_compress(prompt_tokens=base + 1) is True
    assert not [m for m in caplog.messages if "cache-aware" in m], "低命中不打 marker"
    assert e._cache_aware_recall_k() == 3


# ── 压缩触发面 · 高命中介入（抬高线 / 推迟 / 维持 / 活体 marker）────────────────

def test_high_hit_raises_trigger_threshold(tmp_path, monkeypatch):
    """高命中 0.9 ≥ hit_min 0.5 + delay_frac 0.10 → 线抬高 base×1.10（config 驱动）。"""
    e = _mkengine(tmp_path, monkeypatch, _ON)
    e.on_session_start("s_high")
    e.last_cache_hit_ratio = 0.9
    base = e._effective_threshold()
    assert e._cache_aware_trigger_threshold() == int(base * 1.10)
    # delay_frac 走 config 不硬编码：0.25 → base×1.25
    e2 = _mkengine(tmp_path, monkeypatch, _ON, cache_aware_delay_frac=0.25)
    e2.on_session_start("s_high2")
    e2.last_cache_hit_ratio = 0.9
    assert e2._cache_aware_trigger_threshold() == int(base * 1.25)


def test_high_hit_delays_baseline_trigger_logs_marker(tmp_path, monkeypatch, caplog):
    """高命中 + token 落在 (base, adjusted] → 基线本会触发但被抬高线推迟 →
    `cache-aware delay compress (hit=NN%)` marker（活体证据）+ should_compress=False。"""
    e = _mkengine(tmp_path, monkeypatch, _ON)
    e.on_session_start("s_delay")
    e.last_cache_hit_ratio = 0.9
    base = e._effective_threshold()
    adjusted = int(base * 1.10)
    mid = base + max(1, (adjusted - base) // 2)  # base < mid ≤ adjusted
    assert base < mid <= adjusted, "前置：mid 落在 (base, adjusted]（基线会触发、被推迟）"
    with caplog.at_level(logging.INFO, "engine"):
        assert e.should_compress(prompt_tokens=mid) is False, "抬高线后 mid 未过线 → 推迟"
    lines = [m for m in caplog.messages if "cache-aware delay compress" in m]
    assert len(lines) == 1, "推迟 = 活体 marker 恰一行"
    assert "hit=90%" in lines[0], "marker 文案含命中率 NN%（0.9 → 90%）"


def test_high_hit_keep_trigger_logs_marker(tmp_path, monkeypatch, caplog):
    """高命中 + token > adjusted（过抬高线仍触发）→ `cache-aware keep trigger (hit=NN%)`
    marker（维持触发的活体证据）+ should_compress=True。"""
    e = _mkengine(tmp_path, monkeypatch, _ON)
    e.on_session_start("s_keep")
    e.last_cache_hit_ratio = 0.85
    adjusted = int(e._effective_threshold() * 1.10)
    with caplog.at_level(logging.INFO, "engine"):
        assert e.should_compress(prompt_tokens=adjusted + 1) is True, "过抬高线仍触发（维持）"
    lines = [m for m in caplog.messages if "cache-aware keep trigger" in m]
    assert len(lines) == 1, "维持触发 = 活体 marker 恰一行"
    assert "hit=85%" in lines[0]


def test_no_spam_on_repeated_same_state(tmp_path, monkeypatch, caplog):
    """同态多次调用 → marker 只打一次（宿主同 turn 双调 / 同态多 turn 不刷屏）。"""
    e = _mkengine(tmp_path, monkeypatch, _ON)
    e.on_session_start("s_nospam")
    e.last_cache_hit_ratio = 0.9
    adjusted = int(e._effective_threshold() * 1.10)
    with caplog.at_level(logging.INFO, "engine"):
        for _ in range(5):
            e.should_compress(prompt_tokens=adjusted + 1)  # 同态（维持触发）×5
    lines = [m for m in caplog.messages if "cache-aware keep trigger" in m]
    assert len(lines) == 1, "同态重复调用 marker 只打一次（去抖哨兵）"


def test_transition_keep_to_delay_logs_each_flip(tmp_path, monkeypatch, caplog):
    """态 1（keep）→ 态 2（delay）翻转 → 各自打行（翻转才打、同态不刷）。"""
    e = _mkengine(tmp_path, monkeypatch, _ON)
    e.on_session_start("s_trans")
    e.last_cache_hit_ratio = 0.9
    base = e._effective_threshold()
    adjusted = int(base * 1.10)
    with caplog.at_level(logging.INFO, "engine"):
        e.should_compress(prompt_tokens=adjusted + 1)  # 态 1 keep
        e.should_compress(prompt_tokens=base + 1)      # 态 2 delay（token 跌进 (base,adjusted]）
    assert [m for m in caplog.messages if "cache-aware keep trigger" in m], "keep 行"
    assert [m for m in caplog.messages if "cache-aware delay compress" in m], "delay 行"


def test_preflight_uses_cache_aware_threshold(tmp_path, monkeypatch):
    """should_compress_preflight 同走 cache-aware 触发线（宿主 preflight 无 usage 面）。
    高命中 → preflight 粗估须 > 抬高线才触发（现状 preflight 只需 > base）。"""
    e = _mkengine(tmp_path, monkeypatch, _ON)
    e.on_session_start("s_preflight")
    e.last_cache_hit_ratio = 0.9
    base = e._effective_threshold()
    adjusted = int(base * 1.10)
    # 粗估 token 落在 (base, adjusted] → 现状 preflight 触发，cache-aware 抬高后推迟
    # est_tokens = ceil(len/4)：len = 4*(base+1) → 粗估 = base+1（落在区间）
    content_len = 4 * (base + 1)
    assert base < est_tokens("x" * content_len) <= adjusted, "前置：粗估 token 落 (base, adjusted]"
    msgs = [{"role": "user", "content": "x" * content_len}]
    assert e.should_compress_preflight(msgs) is False, "抬高线后 preflight 推迟"
    # fail-safe：ratio None → 线 = base → preflight 现状（粗估 > base 触发）
    e.last_cache_hit_ratio = None
    assert e.should_compress_preflight(msgs) is True, "ratio None → 现状 preflight 触发"
    # config 关 → 同现状
    e2 = _mkengine(tmp_path, monkeypatch, _OFF)
    e2.on_session_start("s_preflight2")
    e2.last_cache_hit_ratio = 0.9
    assert e2.should_compress_preflight(msgs) is True, "config 关 → 现状 preflight 触发"


# ── 召回 K 面（select_context 纯读实例态、宿主签名零改）──────────────────────────

def test_recall_k_high_hit_shrinks(tmp_path, monkeypatch):
    """高命中 + recall_k=1 → _cache_aware_recall_k()=1（收窄 top-K、少换 recall 内容）。"""
    e = _mkengine(tmp_path, monkeypatch, _ON, cache_aware_recall_k=1)
    e.on_session_start("s_k1")
    e.last_cache_hit_ratio = 0.9
    assert e._cache_aware_recall_k() == 1
    # recall_k 走 config：2 → K=2
    e2 = _mkengine(tmp_path, monkeypatch, _ON, cache_aware_recall_k=2)
    e2.on_session_start("s_k2")
    e2.last_cache_hit_ratio = 0.9
    assert e2._cache_aware_recall_k() == 2


def test_recall_k_fail_safe_stays_3(tmp_path, monkeypatch):
    """fail-safe（关 / None / 低命中）→ recall K 恒 3（= 现状硬编码 K，零变化）。"""
    e = _mkengine(tmp_path, monkeypatch, _ON, cache_aware_recall_k=1)
    e.on_session_start("s_kfail")
    assert e.last_cache_hit_ratio is None
    assert e._cache_aware_recall_k() == 3, "ratio None → K=3（现状）"
    e.last_cache_hit_ratio = 0.2  # 低命中
    assert e._cache_aware_recall_k() == 3, "低命中 → K=3（现状）"
    e2 = _mkengine(tmp_path, monkeypatch, _OFF, cache_aware_recall_k=1)
    e2.on_session_start("s_koff")
    e2.last_cache_hit_ratio = 0.9
    assert e2._cache_aware_recall_k() == 3, "config 关 → K=3（现状，零变化）"


def test_select_context_signature_unchanged():
    """宿主接口 select_context 签名零改：cache 不作参数、cache-aware 纯读实例态
    （红线：改签名 = 动 agent/context_engine.py = 破宿主红线；本卡纯读实例态）。"""
    sig = inspect.signature(SupercompressContextEngine.select_context)
    params = list(sig.parameters)
    assert "request_messages" in params
    assert "conversation_messages" in params
    assert "incoming_message" in params
    assert "budget_tokens" in params
    assert not any("cache" in p.lower() for p in params), \
        "不得向宿主 select_context 接口引入 cache 参数（cache-aware 走实例态纯读）"


# ── 决策面只读 last_cache_hit_ratio（不改存态逻辑）──────────────────────────────

def test_decision_face_is_read_only_on_cache_state(tmp_path, monkeypatch):
    """决策面（should_compress / _cache_aware_recall_k / _cache_aware_trigger_threshold）
    只读 last_cache_hit_ratio，不写实例态（无副作用）—— 调前后读数一致。"""
    e = _mkengine(tmp_path, monkeypatch, _ON)
    e.on_session_start("s_readonly")
    e.last_cache_hit_ratio = 0.7
    base = e._effective_threshold()
    before = e.last_cache_hit_ratio
    _ = e.should_compress(prompt_tokens=base + 1)
    _ = e._cache_aware_recall_k()
    _ = e._cache_aware_trigger_threshold()
    assert e.last_cache_hit_ratio is before, "决策面不得写 last_cache_hit_ratio（只读）"
