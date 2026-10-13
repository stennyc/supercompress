"""per-turn cache 遥测：读宿主 turn_usage.py:110-119 喂入 usage_dict 的 cache 桶
（cache_read_tokens / cache_write_tokens）→ 存实例态（last_cache_read_tokens /
last_cache_write_tokens / last_cache_hit_ratio）+ 打日志（marker `supercompress: turn cache hit`）。

把「注入是否破坏 prompt cache / 影响多大」从 estimate 升到 measurement。宿主 agent/ 零 diff、
不改 select_context 签名（宿主接口零变化），全程只改 engine.py 契约面 + 本测试。

验收面（对齐宿主 turn_usage.py 的 cache=X/Y (Z%) 口径）：
- 真实 usage 带 cache 键 → 存 read/write，命中率 = read/prompt（prompt>0）
- 缺 cache 键 / {}（宿主旧版 no-usage 路径）→ 容错 0、不炸、不冲 -1 哨兵（保留既有 cache 态）
- read>0 且 prompt>0 → 打 `supercompress: turn cache hit` 日志行（活体 agent.log 主路径可见）
- read==0（miss）→ 不打行（对齐宿主口径：miss = 无行，非 (0%) 噪声行）
- on_turn_complete 读宿主喂入 usage=agent._last_turn_usage → 存上 turn 命中率（纯读实例态，
  本 turn select_context/compress 可读；不改宿主接口签名——决策面另起卡）
- 会话边界（on_session_reset / on_session_start）清零 cache 态（防跨 session 泄漏）
- select_context 宿主接口签名零变化（cache 不作参数；纯读实例态）

跑法（repo 根）：
  python -m pytest tests/test_cache_telemetry.py -v
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict

import pytest

from engine import SupercompressContextEngine


def _mkengine(tmp_path: Path, context_length: int = 256000) -> SupercompressContextEngine:
    """独立隔离的 engine（external_store_path → tmp，不碰全局 active.jsonl）。
    context_length 256000 → threshold 0.55*256000；本面用 prompt>0 但 < threshold 的真实读数。"""
    return SupercompressContextEngine(context_length=context_length,
                                      external_store_path=str(tmp_path),
                                      auto_recall=True)


def _usage(**extra: Any) -> Dict[str, Any]:
    """宿主 turn_usage.py:110-119 usage_dict 形态（canonical + legacy 兼容键 + cache 桶）。"""
    base: Dict[str, Any] = {
        "prompt_tokens": 50000,
        "completion_tokens": 200,
        "total_tokens": 50200,
        "input_tokens": 50000,
        "output_tokens": 200,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    base.update(extra)
    return base


# ─────────────────────── 真实 usage 带 cache 键 → 存 read/write + 命中率 ───────────────────────

def test_update_from_response_stores_cache_buckets(tmp_path):
    e = _mkengine(tmp_path)
    e.on_session_start("s_cache")
    e.update_from_response(_usage(cache_read_tokens=48000, cache_write_tokens=1200))
    assert e.last_cache_read_tokens == 48000
    assert e.last_cache_write_tokens == 1200
    # 命中率 = read/prompt（prompt>0 时）
    assert e.last_cache_hit_ratio == pytest.approx(48000 / 50000)


def test_cache_hit_ratio_none_when_prompt_zero(tmp_path):
    """prompt 无真实读数（completion-only，prompt=0）→ 命中率分母无效 → None（不除零）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_ratio0")
    e.update_from_response({"completion_tokens": 250})  # prompt=0
    assert e.last_cache_hit_ratio is None


# ─────────────────────── 缺 cache 键 / {} no-usage → 容错 0、不炸、不冲哨兵 ───────────────────────

def test_missing_cache_keys_default_zero(tmp_path):
    """宿主旧版 usage_dict（无 cache 键）→ cache 桶容错 0、不炸；命中率 read=0/prompt>0 → 0.0。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_missing")
    e.update_from_response({"prompt_tokens": 30000, "completion_tokens": 100,
                            "total_tokens": 30100})
    assert e.last_cache_read_tokens == 0
    assert e.last_cache_write_tokens == 0
    assert e.last_cache_hit_ratio == 0.0


def test_no_usage_preserves_minus_one_sentinel_and_cache_state(tmp_path):
    """宿主 no-usage 路径（turn_usage.py:87-91 调 ({})）：prompt==0 且 completion==0 →
    不进真实读数块 → 保留既有 cache 态 + 不冲 -1 哨兵（镜像内置 rollback 对 -1 的保护）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_sentinel_cache")
    e.last_prompt_tokens = -1  # 宿主 _compress_context 提交 compact 后写的哨兵
    e.last_cache_read_tokens = 42  # 上一次真实读数的 cache 态
    e.update_from_response({})  # 宿主 no-usage 路径
    assert e.last_prompt_tokens == -1, "no-usage 不得冲掉 -1 哨兵"
    assert e.last_cache_read_tokens == 42, "no-usage 保留既有 cache 态（不误清）"


def test_no_usage_fresh_engine_cache_stays_zero(tmp_path):
    """fresh engine（非哨兵）+ {} no-usage → cache 态保持 __init__ 默认（0 / 0 / None）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_zero_cache")
    e.update_from_response({})
    assert e.last_cache_read_tokens == 0
    assert e.last_cache_write_tokens == 0
    assert e.last_cache_hit_ratio is None


# ─────────────────────── 日志面（活体 agent.log 主路径可见 per-turn cache_read） ───────────────────────

def test_cache_hit_log_emitted_on_read_hit(tmp_path, caplog):
    """read>0 且 prompt>0 → 打 `supercompress: turn cache hit` 行（marker 自含，走插件自己 logger）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_log")
    with caplog.at_level(logging.INFO, "engine"):
        e.update_from_response(_usage(cache_read_tokens=48000, cache_write_tokens=1200))
    lines = [m for m in caplog.messages if m.startswith("supercompress: turn cache hit")]
    assert len(lines) == 1, "一次真实命中读数应恰打一行 cache 遥测日志"
    line = lines[0]
    assert "read=48000" in line
    assert "prompt=50000" in line
    assert "96%" in line, "命中率 48000/50000 → 96%（对齐宿主 cache=X/Y (Z%) 口径）"
    assert "write=1200" in line


def test_cache_log_silent_on_miss(tmp_path, caplog):
    """read==0（miss）→ 不打行（对齐宿主 cache=X/Y 口径：miss = 无行，非 (0%) 噪声行）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_miss")
    with caplog.at_level(logging.INFO, "engine"):
        e.update_from_response(_usage(cache_read_tokens=0))
    assert not any(m.startswith("supercompress: turn cache hit") for m in caplog.messages), \
        "cache miss（read=0）不打行——与宿主 cache=X/Y 口径一致（miss = 无行）"


# ─────────────────────── on_turn_complete 存上 turn 命中率（item 3，纯读实例态） ───────────────────────

def test_on_turn_complete_stores_hit_ratio(tmp_path):
    """on_turn_complete 读宿主喂入的 usage=agent._last_turn_usage（含 cache）→ 存上 turn 命中率。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_ondone")
    e.on_turn_complete([{"role": "user", "content": "hi"}],
                       usage=_usage(cache_read_tokens=45000, prompt_tokens=50000))
    assert e.last_cache_hit_ratio == pytest.approx(45000 / 50000)


def test_on_turn_complete_no_usage_ratio_none(tmp_path):
    """usage=None（宿主旧版 / 缺 last_turn_usage）→ 命中率 None（不除零、不炸）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_ondone2")
    e.on_turn_complete([{"role": "user", "content": "hi"}], usage=None)
    assert e.last_cache_hit_ratio is None


def test_select_context_can_read_cache_state_without_signature_change(tmp_path):
    """本 turn select_context/compress 纯读实例态（cache 命中率可被读到；宿主接口签名零变化）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_read")
    e.on_turn_complete([{"role": "user", "content": "hi"}],
                       usage=_usage(cache_read_tokens=45000, prompt_tokens=50000))
    # select_context 走宿主既有契约调用面（无 cache 参数），可读到上 turn 命中率（纯读实例态）：
    e.select_context([{"role": "user", "content": "next turn"}],
                     conversation_messages=[{"role": "user", "content": "hi"}])
    assert e.last_cache_hit_ratio == pytest.approx(45000 / 50000), \
        "select_context 调用后可读上 turn cache 命中率（纯读实例态，宿主接口零变化）"


# ─────────────────────── 会话边界清零 cache 态（防跨 session 泄漏） ───────────────────────

def test_session_reset_zeros_cache_state(tmp_path):
    e = _mkengine(tmp_path)
    e.on_session_start("s_reset")
    e.update_from_response(_usage(cache_read_tokens=48000, cache_write_tokens=1200))
    assert e.last_cache_read_tokens == 48000
    e.on_session_reset()
    assert e.last_cache_read_tokens == 0
    assert e.last_cache_write_tokens == 0
    assert e.last_cache_hit_ratio is None


def test_session_start_zeros_cache_state(tmp_path):
    """rebind/轮转（on_session_start）不得让旧 session 的 cache 命中率泄漏进新 session 的读面。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_old")
    e.update_from_response(_usage(cache_read_tokens=48000, cache_write_tokens=1200))
    e.on_session_start("s_new")
    assert e.last_cache_read_tokens == 0
    assert e.last_cache_write_tokens == 0
    assert e.last_cache_hit_ratio is None


# ─────────────────────── select_context 宿主接口签名零变化（红线） ───────────────────────

def test_select_context_signature_unchanged(tmp_path):
    """宿主接口 select_context 签名零变化：cache 遥测不向宿主接口添加任何 cache 参数
    （红线：改 select_context 签名 = 动 agent/context_engine.py = 破宿主红线；本卡纯读实例态）。"""
    sig = inspect.signature(SupercompressContextEngine.select_context)
    params = list(sig.parameters)
    assert "request_messages" in params
    assert "conversation_messages" in params
    assert "incoming_message" in params
    assert "budget_tokens" in params
    assert not any("cache" in p.lower() for p in params), \
        "不得向宿主 select_context 接口引入 cache 参数"
