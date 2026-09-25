"""Block ④ tagger：P0 确定性结构地板（语言无关）+ LLM tiering（机器可解析 JSON）。

gate 锁语义：
- 结构地板（provenance=user）：恒 P0 structural，不进 LLM 评分（transient_fail 时兜底仍保）；
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


class Tagger:
    def __init__(self, aux: AuxProvider, session_topic: str) -> None:
        self.aux = aux
        self.session_topic = session_topic

    def classify(self, text: str, provenance: str) -> tuple[str, bool, str, str]:
        """返回 (tier, structural, regime, reason)。

        契约：user-sourced = P0 structural 地板（确定性，不过 aux）；
        model-sourced = aux 端点「tagging」任务 + 机器可解析 JSON。
        """
        if provenance == "user":
            return ("P0", True, "normal", "user-sourced structural floor (deterministic)")

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
