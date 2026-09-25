"""supercompress — 纯蒸馏器 + 锚点发射器（gate v4 (A) 定稿）。

模块边界：
- schema   Segment 信封 / NFC content_hash / token 估计
- store    单一事实源 jsonl（数据面委托 scdata 单一权威实现：replay/materialize/闭合校验；
           写路径 = tier-aware fsync：P0 即写即 fsync+读回，P1/P2 lazy 批）
- budgets  共享分母（post-distill usable tokens，Pull 20% / Push 15% 一处定义）
- aux      (A) 零新 provider：主模型端点直调 tagging/scoring + 原生 compression 任务 digest；
           gate⑤ 故障注入 marker off-by-default（测试替身住 tests/fakes.py，不进生产 import 面）
- distill  80% 软水位蒸馏 + P0 逐出 gate + push cap + tier-aware fsync + backpressure
- tagger   P0 确定性结构地板（语言无关）+ LLM tiering（机器可解析 JSON）   ← commit ④
- ledger   活账本双层（goal 钉死 + phase 可 supersede）+ 确定性触发 + DONE 三级证据 ← commit ③
- recall   点查 O(1) + BM25 top-k P0 boost + Pull 预算 enforce                        ← commit ②
"""

from .schema import Segment, nfc_hash, est_tokens
from .store import JsonlStore, StoreError
from .budgets import usable_tokens, pull_budget, push_cap
from .aux_provider import AuxProvider, marker_enabled
from .distill import Window, WindowSeg, Distiller, DistillResult, Anchor
from .live_endpoint import LiveEndpoint, resolve_live_endpoint

__all__ = [
    "Segment", "nfc_hash", "est_tokens",
    "JsonlStore", "StoreError",
    "usable_tokens", "pull_budget", "push_cap",
    "AuxProvider", "marker_enabled",
    "Window", "WindowSeg", "Distiller", "DistillResult", "Anchor",
    "LiveEndpoint", "resolve_live_endpoint",
]
