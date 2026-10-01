"""P0.4b 引擎侧兜底：rotation 信号契约面（pressure / rotation_reason /
should_carry_over + rotation_advisory 升 property）。

references/supercompress-revamp-plan.md「P0.4b 引擎侧兜底路线」节 DoD：
宿主 getattr(agent.context_compressor, "...") 即可读——引擎把 rotation 信号契约化，
宿主 0 改动、未来 TUI 卡只读一行。

覆盖（与 P0.4 既有 TestRotationAdvisory 互补，本文件只测新增 3 方法 + property 形状）：
- pressure() 除零安全（context_length <= 0 → 0.0）+ 正常比值
- rotation_reason() 置位文案（含 pressure / 连续 turn 数）& 未置位 "none"
- should_carry_over() 纯 advisory 谓词 + 零 side-effect（store / last_prompt_tokens /
  _adv_consec 调用前后全不变）
- rotation_advisory property：getter 返回 + setter 可写（对称 force_evict_mode）

跑法（repo 根）：
  python -m pytest tests/test_rotation_contract.py -v
"""
from __future__ import annotations

from pathlib import Path

from engine import (
    SupercompressContextEngine,
    ROTATION_PROMPT_RATIO,
)


def _mkengine(tmp_path: Path, context_length: int = 256000) -> SupercompressContextEngine:
    """独立隔离的 engine 实例（external_store_path → tmp，不碰全局 active.jsonl）。

    context_length 参数（P0.4b 测试面：小窗口 10000 造压力 / 0 验除零安全）；
    默认 256000 保持既有测试行为不变。"""
    return SupercompressContextEngine(context_length=context_length,
                                      external_store_path=str(tmp_path))


def _press(e: SupercompressContextEngine, prompt_tokens: int) -> None:
    """造一次 turn 末压力：usage 更新 + turn 末接线（advisory 检查在 turn 尾部执行）。"""
    e.update_from_response({"prompt_tokens": prompt_tokens})
    e.on_turn_complete([])


# ───────────────────────── pressure() ─────────────────────────

def test_pressure_returns_ratio(tmp_path):
    """pressure() = last_prompt_tokens / context_length（O(1) 纯读）。"""
    e = _mkengine(tmp_path, context_length=10000)
    e.on_session_start("s_p")
    e.update_from_response({"prompt_tokens": 8600})
    assert e.pressure() == 0.86


def test_pressure_zero_division_safe(tmp_path):
    """DoD 除零安全：context_length <= 0 → 返回 0.0（不抛 ZeroDivisionError，
    对称 _check_rotation_advisory 的 context_length<=0 直接 return 防除零）。"""
    e = _mkengine(tmp_path, context_length=0)
    e.last_prompt_tokens = 9000  # 即便 prompt 满格，context_length=0 也返回 0.0
    assert e.pressure() == 0.0
    # 负值同路径（<= 0 防御）
    e.context_length = -5
    assert e.pressure() == 0.0


def test_pressure_fresh_engine_is_zero(tmp_path):
    """fresh engine（last_prompt_tokens=0）pressure = 0.0，非 0 context 也安全。"""
    e = _mkengine(tmp_path, context_length=10000)
    e.on_session_start("s_p0")
    assert e.pressure() == 0.0


# ───────────────────────── rotation_reason() ─────────────────────────

def test_rotation_reason_unset_is_none(tmp_path):
    """未置位 → "none"（宿主读一行即知无 advisory）。"""
    e = _mkengine(tmp_path, context_length=10000)
    e.on_session_start("s_r")
    assert e.rotation_reason() == "none"
    # 单高压 turn（未连 2）仍不置位 → 仍 "none"
    _press(e, 9000)
    assert e.rotation_reason() == "none"


def test_rotation_reason_set_describes_pressure_and_debounce(tmp_path):
    """置位 → 原因串含 pressure（2 位小数）与连续 turn 数（防抖计数实态）。"""
    e = _mkengine(tmp_path, context_length=10000)
    e.on_session_start("s_r2")
    _press(e, 9000)   # turn1：consec=1，未置位
    _press(e, 8600)   # turn2：consec=2，置位（pressure=0.86）
    assert e.rotation_advisory is True
    reason = e.rotation_reason()
    assert reason.startswith("window pressure 0.86 for "), reason
    assert "2 consecutive turn(s)" in reason, reason
    # 再压 1 turn：consec 涨到 3（已置位幂等保持），原因串跟随实态
    _press(e, 9000)
    assert e._adv_consec == 3
    assert "3 consecutive turn(s)" in e.rotation_reason()


# ───────────────────────── should_carry_over() ─────────────────────────

def test_should_carry_over_requires_set_and_pressure(tmp_path):
    """纯 advisory 谓词 = rotation_advisory AND pressure() >= ROTATION_PROMPT_RATIO
    （复用置位线常量，零硬编码 0.85）。"""
    e = _mkengine(tmp_path, context_length=10000)
    e.on_session_start("s_c")
    # (a) 未置位 → False（即便当前 pressure 满格 0.90）
    _press(e, 9000)
    assert e.pressure() >= ROTATION_PROMPT_RATIO
    assert e.should_carry_over() is False, "未置位 advisory 不得报 carry"
    # (b) 连 2 turn 置位 + pressure≥0.85 → True
    _press(e, 8600)
    assert e.rotation_advisory is True
    assert e.should_carry_over() is True
    # (c) 置位后压力掉到中间带 [0.75, 0.85)：advisory 保持但 pressure<0.85 → False
    _press(e, 8000)
    assert e.rotation_advisory is True
    assert e.should_carry_over() is False, "pressure<0.85 时谓词必须 False（同源置位线）"
    # (d) 低压力清位 → 再次 False
    _press(e, 6000)
    assert e.rotation_advisory is False
    assert e.should_carry_over() is False


def test_should_carry_over_zero_side_effect(tmp_path):
    """DoD 核心：调用前后 store.line_count / last_prompt_tokens / _adv_consec /
    _rotation_advisory 全不变——纯读，宿主不调本方法即零行为（零注入、零 cache 破坏）。"""
    e = _mkengine(tmp_path, context_length=10000)
    e.on_session_start("s_c2")
    _press(e, 9000)
    _press(e, 8600)
    from sc.schema import Segment
    seg = Segment(kind="segment", tier="P0", provenance="user",
                  original_text="side-effect probe", session_topic="", model_version="m1")
    e.store.append(seg)
    lines_before = e.store.line_count
    p_before = e.last_prompt_tokens
    consec_before = e._adv_consec
    adv_before = e._rotation_advisory
    win_before = e._window
    # 反复调用（谓词纯函数，幂等 0 副作用）
    for _ in range(5):
        e.should_carry_over()
    assert e.store.line_count == lines_before, "store 行数不得变"
    assert e.last_prompt_tokens == p_before, "token 账不得变"
    assert e._adv_consec == consec_before, "防抖计数不得变"
    assert e._rotation_advisory is adv_before, "advisory 置位态不得变"
    assert e._window is win_before, "window 引用不得重建"


# ───────────────────────── rotation_advisory property（对称 force_evict_mode）──

def test_rotation_advisory_property_getter_setter(tmp_path):
    """property getter 返回 + setter 可写（形状对称 force_evict_mode engine.py:383-389）：
    宿主/测试 `e.rotation_advisory = True` 走 setter 落底层 _rotation_advisory；
    读 getter 即真值。"""
    e = _mkengine(tmp_path, context_length=10000)
    assert e.rotation_advisory is False, "fresh engine advisory 默认 False"
    e.rotation_advisory = True          # setter 路径
    assert e.rotation_advisory is True
    assert e._rotation_advisory is True, "setter 必须落底层私有存储"
    e.rotation_advisory = False
    assert e.rotation_advisory is False
    assert e._rotation_advisory is False
    # 形状对称 force_evict_mode（同一套 property+setter 范式）
    assert isinstance(type(e).rotation_advisory, property)
    assert isinstance(type(e).force_evict_mode, property)


def test_rotation_advisory_property_equivalence_with_plain_attr(tmp_path):
    """零回归面：升 property 后既有内部赋值点（_check_rotation_advisory 置/清位、
    会话边界清零）经 setter 走底层，行为与原 plain attribute 完全等价。"""
    e = _mkengine(tmp_path, context_length=10000)
    e.on_session_start("s_eq")
    _press(e, 9000)
    _press(e, 8600)
    assert e.rotation_advisory is True
    # 会话边界清零经 property setter（on_session_reset）
    e.on_session_reset()
    assert e.rotation_advisory is False and e._adv_consec == 0
    # 重新置位 + on_session_start 边界清零（轮转 rebind 不泄漏）
    e.on_session_start("s_eq2")
    _press(e, 9000)
    _press(e, 8600)
    assert e.rotation_advisory is True
    e.on_session_start("s_eq3")
    assert e.rotation_advisory is False and e._adv_consec == 0
