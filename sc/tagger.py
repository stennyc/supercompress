"""Block ④ tagger：P0 确定性结构地板（语言无关）+ LLM tiering（机器可解析 JSON）。

gate 锁语义：
- 结构地板（provenance=user）：短指令（len < 100）恒 P0 structural，不进 LLM 评分
  （transient_fail 时兜底仍保）；长 user-sourced 内容（group chat 转发 / compaction
  summary / 注入块，len ≥ 100）→ P0 non-structural（仍 P0 优先级，push cap 溢出可逐出），
  根治「注入内容打 user provenance → 100% structural → push cap 失效」；
- LLM tiering 走 aux 主模型端点（零新 provider），输出**机器可解析 JSON**
  （`{"tier":"P2","score":3}`），不吐自然语言再 regex；
- 边界 ±1（score 6–8）→ 升一级保守（P2 → P1）；
- 不可机器解析（自然语言 / 坏 JSON）→ 保守 P0 structural，永不当 P2 丢；
- transient_fail（端点异常）→ regime=transient_fail，落结构地板 P0，不误降级。
"""
from __future__ import annotations

import json

from .aux_provider import AuxProvider

TIERS = ("P0", "P1", "P2")
_TIERS_BY_TIER = {t: i for i, t in enumerate(TIERS)}
# user 短指令 structural 地板阈值（@reviewer 裁定 2026-09-29）：
# provenance=user 且 len < 100 → P0 structural；len ≥ 100 → P0 non-structural（可逐出）。
USER_STRUCTURAL_LEN = 100
# 长 user-sourced 内容的系统噪声标记（≥100 chars 时才判定，短指令恒保护）：
# 日志/技能提示词 → P2；compaction summary → P1（@researcher 分级表，@reviewer 锁定）。
_P2_MARKERS = ("Traceback (most recent call last)", "console.log(", "console.error(",
               "console.warn(", "UPDATE_SKILL", "update the skill library")
_P1_MARKERS = ("<compaction-summary>",)


def classify_user(text: str) -> tuple[str, bool]:
    """user-sourced 分级规则（单一实现源：Tagger.classify 与存量重打分 retag.py 共用）。

    - len < 100（短指令）→ ("P0", structural=True)：确定性地板，保护真实用户指令；
    - len ≥ 100 + 系统噪声标记（日志 / 技能提示词）→ ("P2", False)；
    - len ≥ 100 + compaction summary 标记 → ("P1", False)；
    - len ≥ 100 其余（group chat 转发 / 注入块）→ ("P0", False)：
      优先级不变（仍 P0，先于 P1/P2 保留），仅 push cap 溢出时可逐出。
    """
    if len(text) < USER_STRUCTURAL_LEN:
        return ("P0", True)
    if any(m in text for m in _P2_MARKERS):
        return ("P2", False)
    if any(m in text for m in _P1_MARKERS):
        return ("P1", False)
    return ("P0", False)


class Tagger:
    def __init__(self, aux: AuxProvider, session_topic: str) -> None:
        self.aux = aux
        self.session_topic = session_topic

    def classify(self, text: str, provenance: str) -> tuple[str, bool, str, str]:
        """返回 (tier, structural, regime, reason)。

        契约：user-sourced 短指令（< 100 chars）= P0 structural 地板（确定性，不过 aux）；
        user-sourced 长内容（≥ 100 chars，group chat 转发 / compaction summary 等注入块）
        = P0 non-structural（优先级不变，push cap 溢出时可逐出，注入类 0% structural）；
        model-sourced = aux 端点「tagging」任务 + 机器可解析 JSON。
        """
        if provenance == "user":
            tier, structural = classify_user(text)
            if structural:
                return (tier, True, "normal", "user-sourced short instruction structural floor (<100 chars, deterministic)")
            return (tier, False, "normal",
                    "user-sourced long content (group-chat/recall/injected block) -> %s non-structural, evictable under push cap" % tier)

        # LLM tiering + scoring（机器可解析 JSON）
        try:
            raw = self.aux.main_endpoint.call("tagging", text)
            obj = json.loads(raw)
        except Exception:
            return ("P0", True, "transient_fail", "tagging endpoint transient failure -> floor P0")

        if not isinstance(obj, dict):
            return ("P0", True, "normal", "non-JSON tagging output -> conservative P0 (never P2-drop)")
        tier = obj.get("tier", "P1")
        if tier not in _TIERS_BY_TIER:
            return ("P0", True, "normal", f"unknown tier {tier!r} -> conservative P0")
        try:
            score = int(obj.get("score", 5))
        except (TypeError, ValueError):
            score = 5
        # 边界 ±1（6–8）→ 升一级保守（P2 → P1；P1 → P0；P0 不动）
        if 6 <= score <= 8:
            i = _TIERS_BY_TIER[tier]
            tier = TIERS[max(0, i - 1)]
        return (tier, False, "normal", f"LLM tiering: tier={obj.get('tier')} score={score}")
