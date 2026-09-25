"""预算共享分母（gate ⑥ 定稿：定义只落装配 spec 一处，Pull / Push 共用）。

- usable_tokens = post-distill usable window tokens（蒸馏后可用窗口 token 总量）
- Pull 召回预算 = 20%（P1/P2 计入；P0 豁免但写 budget_breach）
- Push 常驻 cap = 15%（P0 常驻集上限；goal / structural P0 / 活账本链尾永不裁）
"""
from __future__ import annotations

PULL_BUDGET_FRAC = 0.20
PUSH_CAP_FRAC = 0.15


def usable_tokens(window) -> int:
    """共享分母唯一来源：post-distill usable tokens。"""
    return int(window.capacity_tokens)


def pull_budget(window) -> int:
    return int(PULL_BUDGET_FRAC * usable_tokens(window))


def push_cap(window) -> int:
    return int(PUSH_CAP_FRAC * usable_tokens(window))
