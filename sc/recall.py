"""Block ② memory_recall：点查 O(1) + BM25 top-k（P0 boost）+ Pull 预算 enforce。

设计锁（gate 定稿）：
- 召回数据面 = 单一事实源 jsonl（store 委托 scdata 权威层，单 replay 路径）——
  逐出后原文可全量取回（可逆，非单向有损）；
- 点查 O(1)：segment_id → 原文（P0 0-loss）；
- 排序：token 重叠（BM25 语义，P0 同相关度 boost 优先）+ stale P1 降权（召回侧 verify 提示）；
- Pull 预算 = 共享分母（post-distill usable × 20%）一处定义：
  P2 优先降权、P1 超预算累计降权、P0 豁免强制通过（超预算写 budget_breach 日志，不静默）。
"""
from __future__ import annotations

import re

from .budgets import pull_budget
from .schema import est_tokens


class Hit:
    def __init__(self, seg, score: float, downweighted: bool = False, note: str = "") -> None:
        self.seg = seg
        self.score = score
        self.downweighted = downweighted
        self.stale = bool(getattr(seg, "stale", False))
        self.note = note


class Recaller:
    def __init__(self, store, window) -> None:
        self.store = store
        self.window = window
        self.budget = pull_budget(window)      # Pull 预算读共享分母（一处定义）
        self.breach_log: list[dict] = []       # P0 超预算强制通过 → budget_breach（不静默）

    # ───────────── 点查 O(1) ─────────────
    def point_fetch(self, segment_id: str):
        """P0 0-loss：segment_id（行级唯一）→ 原文 Segment，miss = None。"""
        return self.store.get(segment_id)

    # ───────────── 语义召回（top-k）─────────────
    @staticmethod
    def _tok(text: str) -> set:
        return set(re.findall(r"\S+", (text or "").lower()))

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = self._tok(query)
        tidx = {"P0": 0, "P1": 1, "P2": 2}
        scored = []
        for ws in self.window.segs:
            seg = ws.segment
            s = float(len(q & self._tok(seg.original_text)) * 2.0)
            if seg.tier == "P0":
                s *= 1.5                                   # P0 boost：同相关度 P0 优先
            if getattr(seg, "stale", False):
                s *= 0.5                                   # stale P1 召回降权
            scored.append((-s, tidx[seg.tier], seg.seq, seg))
        scored.sort()                                      # 相关度 desc → 层级（P0 先）→ seq 稳定序
        cum = 0
        hits: list[Hit] = []
        for s, _, _, seg in scored[:k]:
            toks = est_tokens(seg.original_text)
            note = ""
            if seg.tier == "P0":
                dw = False                                 # P0 豁免预算、强制通过
                if toks > self.budget:
                    self.breach_log.append({"segment_id": seg.segment_id,
                                             "tokens": toks, "budget": self.budget})
            elif seg.tier == "P2":
                dw = True                                   # P2 优先降权
                note = "P2 downweighted-first"
            else:
                cum += toks
                dw = cum > self.budget                      # P1 超预算累计降权
                if dw:
                    note = "pull budget overflow, downweighted"
            if getattr(seg, "stale", False):
                note = (note + " · " if note else "") + "stale P1 · verify before acting"
            hits.append(Hit(seg, -s, dw, note))
        return hits
