"""cache-stuck detection + head-freeze self-heal（2026-11-11 owner 指令「修」）。

根因（agent.log 实证，session 555e）：filemem 注入面逐轮跳变（67→11→109→168 条）
+ P3 cold-start replay 块（8348 chars）驻 system head 区（position=after_memory）
→ head 字节每轮变 → provider 前缀缓存只在静态前缀（≈8192 tokens）命中、
  后续 21 轮 read 锁死、命中率 25%→9%。对照 b59e/9e76（注入 9/2 条稳定）
  read 正常累积到 35840→41216（98%）。

本卡两层（全在 supercompress engine.py，宿主 agent/ 零 diff，不改宿主接口签名）：

**① 检测面（默认开，fail-open 纯观测，零行为变化）**
`_record_cache_telemetry` 里追踪"read 值连续 N 轮不变 且 prompt 在涨"
→ 打一条 WARNING `supercompress: cache_read stuck at <read> for N turns
  while prompt grew <prev>→<now>（provider 前缀缓存未累积，检查 head 区
  动态注入 / filemem 注入面稳定性）`。N 缺省 3（config 键
  `cache_stuck_detect` 整数，缺省 3；0/负 = 关）。只在真卡死打行，
  正常增长 / 首轮 / read 在变 → 零新增日志（与 T5 55% 分页面字节不变
  同口径：本面不改任何 prompt/账本字节，纯实例态 + 日志）。

**② 自愈面（默认关，config 键 `head_freeze_enabled` 显式 True 才启用）**
检测到卡死后置实例态 `_head_frozen`；`select_context` 顶部加一条独立分支：
_head_frozen 时把 request_messages 里 system 段头部（除最末一条 user
消息外的前缀区，含 filemem head 注入块）做"前缀指纹门控"——与上轮
`_head_fingerprint` 不一致的 system 段字节 → 折成单行存根指针
（`[head-stub] <n> section(s) frozen`，字节稳定）+ 原文移挂到末条
user 消息尾部（`_s1_tail_deliver` 同模式，尾部变更不打断前缀）。
auto_recall 关也走本面（不挂 auto_recall 开关——head 稳定化是独立
决策面，与 recall 归属错乱 A1 红线无关）。fail-open：指纹/移挂任一
异常 → 原样返回 request_messages（宿主契约：malformed 永不 reach
provider）。

红线（owner）：只改 sc/engine.py + 本测试；零宿主 agent/、零 .db、
不改 select_context / update_from_response 宿主签名（实例态全走 engine
内部）。

跑法（repo 根 / plugins/context_engine/supercompress）：
  python -m pytest tests/test_cache_stuck.py -v
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pytest

from engine import SupercompressContextEngine


def _mkengine(tmp_path: Path, **kw: Any) -> SupercompressContextEngine:
    base: Dict[str, Any] = dict(
        context_length=256000,
        external_store_path=str(tmp_path),
        auto_recall=False,
    )
    base.update(kw)
    return SupercompressContextEngine(**base)


def _usage(read: int, prompt: int) -> Dict[str, Any]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": 200,
        "total_tokens": prompt + 200,
        "cache_read_tokens": read,
        "cache_write_tokens": 0,
    }


def _feed_turn(e: SupercompressContextEngine, read: int, prompt: int) -> None:
    """宿主 turn 末喂入 usage（update_from_response 主路径，log_hit=True）。"""
    e.update_from_response(_usage(read, prompt))


# ─────────────────────────── ① 检测面：read 卡死 + prompt 增长 → WARNING ───────────────────────────


def test_cache_stuck_warns_after_n_rounds(tmp_path, caplog):
    """连续 3 轮 read 不变（8192）且 prompt 在涨 → 第 3 轮打 1 条 stuck WARNING。"""
    e = _mkengine(tmp_path)  # cache_stuck_detect 缺省 3
    e.on_session_start("s_stuck")
    with caplog.at_level(logging.WARNING, "engine"):
        _feed_turn(e, 8192, 32520)   # 轮 1：冷读首值（计数起点，不打）
        _feed_turn(e, 8192, 34398)   # 轮 2：read 不变 + prompt 涨 → 计数 2（不打）
        _feed_turn(e, 8192, 36832)   # 轮 3：计数 3 → 打 1 条
    stuck = [m for m in caplog.messages if "cache_read stuck" in m]
    assert len(stuck) == 1, f"第 3 轮应恰打 1 条 stuck WARNING（got {len(stuck)}）"
    assert "8192" in stuck[0]


def test_cache_stuck_no_warn_when_read_grows(tmp_path, caplog):
    """read 在增长（正常累积，对照 b59e）→ 永不打 stuck（零新增日志红线）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_grow")
    with caplog.at_level(logging.WARNING, "engine"):
        _feed_turn(e, 11264, 35123)
        _feed_turn(e, 35840, 36630)
        _feed_turn(e, 39680, 40050)
        _feed_turn(e, 41216, 42002)
    assert not [m for m in caplog.messages if "cache_read stuck" in m]


def test_cache_stuck_no_warn_when_prompt_shrinks(tmp_path, caplog):
    """read 不变但 prompt 不涨（compact 后回落）→ 不打（增长判据不满足）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_shrink")
    with caplog.at_level(logging.WARNING, "engine"):
        _feed_turn(e, 8192, 32520)
        _feed_turn(e, 8192, 32520)   # prompt 未涨
        _feed_turn(e, 8192, 30000)   # prompt 回落（compact 后）
    assert not [m for m in caplog.messages if "cache_read stuck" in m]


def test_cache_stuck_reset_after_recover(tmp_path, caplog):
    """卡 2 轮后 read 恢复增长 → 计数归零；再卡 3 轮才重新打（不累积旧计数）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_reset")
    with caplog.at_level(logging.WARNING, "engine"):
        _feed_turn(e, 8192, 32520)
        _feed_turn(e, 8192, 34000)        # 计数 2（未打）
        _feed_turn(e, 35000, 36000)       # read 变 → 计数归零
        _feed_turn(e, 35000, 37000)       # 重新计数 2
        _feed_turn(e, 35000, 38000)       # 计数 3 → 打 1
        _feed_turn(e, 40000, 39000)       # read 又变 → 归零
        _feed_turn(e, 40000, 40000)      # 计数 2（不打）
    stuck = [m for m in caplog.messages if "cache_read stuck" in m]
    assert len(stuck) == 1


def test_cache_stuck_disabled_via_config(tmp_path, caplog):
    """cache_stuck_detect=0 → 检测面整体关（零日志、零行为）。"""
    e = _mkengine(tmp_path, cache_stuck_detect=0)
    e.on_session_start("s_off")
    with caplog.at_level(logging.WARNING, "engine"):
        _feed_turn(e, 8192, 32520)
        _feed_turn(e, 8192, 34398)
        _feed_turn(e, 8192, 36832)
    assert not [m for m in caplog.messages if "cache_read stuck" in m]


def test_cache_stuck_custom_n(tmp_path, caplog):
    """cache_stuck_detect=5 → 第 5 轮才打（N 可配，缺省 3 的验证面）。"""
    e = _mkengine(tmp_path, cache_stuck_detect=5)
    e.on_session_start("s_n5")
    with caplog.at_level(logging.WARNING, "engine"):
        for i in range(4):
            _feed_turn(e, 8192, 32520 + 1000 * i)   # 计数 2..5 之前的 4 轮
        _feed_turn(e, 8192, 32520 + 5000)            # 第 5 轮 → 打 1
    assert len([m for m in caplog.messages if "cache_read stuck" in m]) == 1


# ─────────────────────────── 契约面：select_context 恢复原字节不变契约（head-freeze 已撤） ───────────────────────────


def test_select_context_still_noop_when_auto_recall_off(tmp_path, caplog):
    """2026-11-11 owner 拍板撤 head-freeze 后：auto_recall 关（默认）→ select_context
    恒 return None（宿主字节不变原契约），即使检测面已判卡死（纯观测不介入 request）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_sc_noop")
    with caplog.at_level(logging.WARNING, "engine"):
        _feed_turn(e, 8192, 32520)
        _feed_turn(e, 8192, 34398)
        _feed_turn(e, 8192, 36832)   # 卡死判定触发（WARNING 打 1 条，纯观测）
    assert [m for m in caplog.messages if "cache_read stuck" in m], "前置：检测面已判卡死"
    out = e.select_context(
        [{"role": "user", "content": "hi"}, {"role": "user", "content": "x" * 16}],
        incoming_message={"role": "user", "content": "x" * 16})
    assert out is None, "auto_recall 关 → select_context 字节不变（head-freeze 已撤）"


def test_no_head_freeze_attributes_exist(tmp_path):
    """撤 head-freeze 后：engine 不再有 head_freeze_enabled / _head_frozen /
    _head_fingerprint 实例态（防误用残留面，归属根因在 filemem prefetch 输出面另卡处理）。"""
    e = _mkengine(tmp_path)
    assert not hasattr(e, "head_freeze_enabled")
    assert not hasattr(e, "_head_frozen")
    assert not hasattr(e, "_head_fingerprint")
    assert not hasattr(e, "_head_freeze_apply")

