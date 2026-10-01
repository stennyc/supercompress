#!/usr/bin/env python3
"""P1 DoD#1 对比 harness：旧词重叠基线 vs P1 BM25+CJK 层，同 corpus 量 recall@3。

同 commit 可复跑：语料/标注来自 bench/gold_recall_set.py（确定性），
两种 scorer 都是纯函数（无 wall-clock、无随机）→ 输出稳定。

判据（plan P1 DoD#1）：P1 recall@3 ≥ 旧基线 recall@3 + 30% 才收本阶段。
未达 → 调参（k1/b/boost）一轮后复测，仍不达则报用户决策（不静默放过）。
"""
from __future__ import annotations
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # supercompress/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import re
from sc.schema import Segment
from bench.gold_recall_set import build_segments, gold_pairs


# ── 旧基线 scorer（bfbe058 语义：\S+ 词重叠 × 2.0 + P0×1.5 + stale×0.5）──
def baseline_topk(query: str, segs: list[Segment], k: int = 3):
    q = set(re.findall(r"\S+", (query or "").lower()))
    tidx = {"P0": 0, "P1": 1, "P2": 2}

    def toks(t):
        return set(re.findall(r"\S+", (t or "").lower()))

    scored = []
    for seg in segs:
        s = float(len(q & toks(seg.original_text)) * 2.0)
        if seg.tier == "P0":
            s *= 1.5
        if getattr(seg, "stale", False):
            s *= 0.5
        scored.append((-s, tidx[seg.tier], seg.seq, seg))
    scored.sort()
    return [seg for _, _, _, seg in scored[:k]]


def recall_at_k(topk_segs, target: Segment) -> bool:
    return any(s.segment_id == target.segment_id for s in topk_segs)


def main() -> int:
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()[:12]
    except Exception:
        commit = "(no-git)"

    segs = build_segments()
    pairs = gold_pairs(segs)
    k = 3

    base_hits = sum(1 for q, t in pairs if recall_at_k(baseline_topk(q, segs, k), t))
    # P1 层 = 现盘上 sc.recall.Recaller
    from sc.recall import Recaller
    from sc.distill import Window, WindowSeg
    p1_hits = 0
    for q, t in pairs:
        w = Window(2000, [WindowSeg(s) for s in segs])
        rec = Recaller(None, w)          # store 仅 point_fetch 用；top-k 不需
        top = [h.seg for h in rec.search(q, k=k) if h.score > 0.0]
        p1_hits += 1 if recall_at_k(top, t) else 0

    base_r = base_hits / len(pairs)
    p1_r = p1_hits / len(pairs)
    delta = p1_r - base_r
    # +30% 是相对旧基线 recall@3 的相对提升：p1_r >= base_r * 1.30
    gate = p1_r >= base_r * 1.30 if base_r > 0 else p1_r > base_r

    print(f"commit={commit}  corpus={len(segs)}  gold={len(pairs)}  k={k}")
    print(f"  baseline(词重叠\\S+) recall@3 = {base_hits}/{len(pairs)} = {base_r:.3f}")
    print(f"  P1(BM25+CJK)      recall@3 = {p1_hits}/{len(pairs)} = {p1_r:.3f}")
    print(f"  Δ = {delta:+.3f}  相对提升 = {((p1_r-base_r)/base_r*100 if base_r else 0):+.1f}%")
    print(f"  GATE(p1 ≥ base×1.30): {gate}")

    # 逐条 CJK 无空格 case 展示（区分度证据：旧 `\S+` 把整串当一个 token → 命中≈0）
    print("  CJK 无空格 case（旧≈0，P1 主战场）:")
    for q, t in pairs:
        if any("\u4e00" <= c <= "\u9fff" for c in q) and not re.search(r"[a-z] ", q.lower()):
            b = "HIT" if recall_at_k(baseline_topk(q, segs, k), t) else "miss"
            w = Window(2000, [WindowSeg(s) for s in segs])
            p1_top = [h.seg for h in Recaller(None, w).search(q, k=k) if h.score > 0.0]
            p1 = "HIT" if recall_at_k(p1_top, t) else "miss"
            print(f"    {q:24s} base={b:4s} p1={p1}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
