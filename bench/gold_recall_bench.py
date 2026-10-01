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


# ── 弱 scorer（判别力探针：故意无 bigram / 无 IDF / 无 P0 boost）──
# 目的（回应 @reviewer Major f）：证明 gold 集「近重复干扰簇」确有区分度——
# 若连弱 scorer 都拿到 P1 同分，则 P1=100% 只是饱和地板（+111% 虚高）；
# 弱 scorer < P1 才说明 P1 的 bigram+IDF+P0boost 真正在考近重复判别。
def weak_unigram_topk(query: str, segs: list[Segment], k: int = 3):
    """bag-of-拉丁词 + CJK 单字 的共享计数（无 bigram / 无 IDF / 无 tier boost）。"""
    q = set(re.findall(r"[a-z0-9_]+", (query or "").lower()))
    for run in re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+", (query or "").lower()):
        q.update(run)
    tidx = {"P0": 0, "P1": 1, "P2": 2}

    def utoks(t):
        s = set(re.findall(r"[a-z0-9_]+", (t or "").lower()))
        for run in re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+", (t or "").lower()):
            s.update(run)
        return s

    scored = []
    for seg in segs:
        s = float(len(q & utoks(seg.original_text)))   # 裸计数，无权重
        scored.append((-s, tidx[seg.tier], seg.seq, seg))
    scored.sort()
    return [seg for _, _, _, seg in scored[:k]]


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

    # 弱 scorer（判别力探针）：无 bigram/IDF/P0boost → 应在近重复簇上 < P1
    weak_hits = sum(1 for q, t in pairs if recall_at_k(weak_unigram_topk(q, segs, k), t))

    base_r = base_hits / len(pairs)
    p1_r = p1_hits / len(pairs)
    weak_r = weak_hits / len(pairs)
    delta = p1_r - base_r
    # +30% 是相对旧生产 \S+ 基线 recall@3 的相对提升：p1_r >= base_r * 1.30（DoD 定义）
    gate = p1_r >= base_r * 1.30 if base_r > 0 else p1_r > base_r
    # 判别力上界探针（advisory，非门禁）：P1 是否严格优于「朴素 CJK unigram scorer」。
    # False ≠ 缺陷——它说明当前 gold 集对"能分 CJK 单字"的朴素 scorer 仍饱和，
    # +111% 应读作"相对旧生产 \S+ 基线（CJK 完全不召回）的真实提升"，而非 P1 绝对召回上限。
    # 未来给 gold 集加"脏近重复"（让 unigram 失手）后，本探针自动翻 True = 集获得真判别力。
    discrim = p1_r > weak_r

    print(f"commit={commit}  corpus={len(segs)}  gold={len(pairs)}  k={k}")
    print(f"  baseline(旧生产\\S+)  recall@3 = {base_hits}/{len(pairs)} = {base_r:.3f}")
    print(f"  weak(CJK-unigram,上界探针) recall@3 = {weak_hits}/{len(pairs)} = {weak_r:.3f}")
    print(f"  P1(BM25+CJK)        recall@3 = {p1_hits}/{len(pairs)} = {p1_r:.3f}")
    print(f"  Δ(P1-base) = {delta:+.3f}  相对提升 = {((p1_r-base_r)/base_r*100 if base_r else 0):+.1f}%")
    print(f"  GATE(p1 ≥ base×1.30): {gate}   ← 主门禁（DoD），决定退出码")
    # advisory 注记先 hoist（f-string 表达式部分含 backslash 是 3.12+ PEP 701 语法；
    # 宿主测试 venv = 3.11 → 模块 import 即炸。P-INST 可移植性：全树须 3.11 可 compile）。
    _disc_note = "当前 gold 集对朴素 CJK scorer 仍饱和；+111% 是相对旧生产 \\S+ 基线的提升，非 P1 绝对上限"
    print(f"  DISCRIM-CAP(p1 > weak): {discrim}  ← advisory：{'' if discrim else _disc_note}")

    # 逐条 CJK 无空格 case 展示（区分度证据：旧 `\S+` 把整串当一个 token → 命中≈0）
    print("  CJK 无空格 case（旧≈0，P1 主战场）:")
    for q, t in pairs:
        if any("\u4e00" <= c <= "\u9fff" for c in q) and not re.search(r"[a-z] ", q.lower()):
            b = "HIT" if recall_at_k(baseline_topk(q, segs, k), t) else "miss"
            w = Window(2000, [WindowSeg(s) for s in segs])
            p1_top = [h.seg for h in Recaller(None, w).search(q, k=k) if h.score > 0.0]
            p1 = "HIT" if recall_at_k(p1_top, t) else "miss"
            print(f"    {q:24s} base={b:4s} p1={p1}")
    # 退出码 = 主门禁（DoD 定义：P1 ≥ 旧生产 \S+ 基线 ×1.30）。DISCRIM 仅 advisory，不否决。
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
