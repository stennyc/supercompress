#!/usr/bin/env python3
"""P3（S-Phase 4）rerank 增量探针——**非饱和面**（plan T2 DoD 硬要求）。

背景（诚实披露前提）：
- P1 gold recall@3 已 21/21=1.000 **饱和**（本探针面 0 先复测 top-1 面，同样饱和）——
  重测饱和 gold 集只能证明「P3 在饱和面增量=0」，不是 P3 的价值面；
- P3 的价值面 = **非饱和探针**：近重复 top-1 歧义簇（P1 BM25 易同分/易混面）+ 中英混合语料。
  本探针定义 3 条非饱和 probe（每条 = query + 3 段近重复语料 + 唯一 target，
  P1 的 top-1 在该面上**可错可混**，P3 LLM rerank 才是判据）。

NO-FABRICATION GUARD（与 decision_layer_probe 同纪律，@researcher 收口）：
- P1 基线面（面 1）= 零 LLM 确定性，纯字面、commit 锚定可复跑，无条件出数；
- P3 LLM 增量面（面 2）= **必须走宿主主模型端点**（resolve_live_endpoint）；
  无 LIVE_ENDPOINT_* / OPENAI_* env → REFUSED 不产出增量数（对 FakeEndpoint 出增量数
  = 循环论证，同 75% 纪律）。

Run:
  python bench/rerank_probe.py             # 面 0 饱和披露 + 面 1 P1 基线 + 面 2 REFUSED（无 env）
  LIVE_ENDPOINT_URL=… LIVE_ENDPOINT_KEY=… LIVE_ENDPOINT_MODEL=<真模型名> \
      python bench/rerank_probe.py --run   # 面 2 活 LLM 增量真跑
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sc.distill import Window, WindowSeg
from sc.recall import Hit, Recaller
from sc.schema import Segment
from sc.rerank import RerankGate, rerank_hits
from bench.gold_recall_set import build_segments, gold_pairs


def _mkseg(text, tier="P1", seq=0):
    return Segment(tier=tier, provenance="model", original_text=text, seq=seq, ts=0.0)


# ── 非饱和探针集（近重复 top-1 歧义簇：P1 BM25 弱/易混面，P3 LLM rerank 价值面）──
# 每条 = (query, [3 段近重复语料], target_text)。语料同主题近重复、仅 1 处区分
# （maxsize/timeout、五百/八百、8080/8081）——散打 scorer 易同分选错篇。
_PROBES = [
    ("数据库连接池调大哪个参数",
     ["数据库连接池调大 maxsize 到两百并重启服务",
      "数据库连接池调大 timeout 到六十秒并告警",
      "数据库连接池调大 pool size 到三百并压测验证"],
     "数据库连接池调大 maxsize 到两百并重启服务"),
    ("网关 限流 每秒 五百",
     ["网关限流阈值调至每秒五百请求并配置告警",
      "网关限流阈值调至每秒八百请求并观察压测",
      "网关限流阈值调至每秒三百请求并灰度上线"],
     "网关限流阈值调至每秒五百请求并配置告警"),
    ("auth service rollback 到哪个端口",
     ["auth service rollback strategy on deploy failure, port 8080",
      "auth service rollback strategy on deploy failure, port 8081",
      "auth service restart strategy on deploy failure, port 8080"],
     "auth service rollback strategy on deploy failure, port 8080"),
]


def p1_top1(query: str, corpus: list) -> tuple:
    """P1 基线面（零 LLM、确定性）：Recaller top-1 + 分数。返回 (top1_text, scores)。"""
    segs = [_mkseg(t, seq=i) for i, t in enumerate(corpus)]
    w = Window(2000, [WindowSeg(s) for s in segs])
    hits = [h for h in Recaller(None, w).search(query, k=3) if h.score > 0.0]
    if not hits:
        return None, []
    return hits[0].seg.original_text, [(h.seg.original_text[:20], round(h.score, 3))
                                       for h in hits[:3]]


def main(argv) -> int:
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()[:12]
    except Exception:
        commit = "(no-git)"
    run = "--run" in argv

    # ── 面 0：饱和 gold 集复测（诚实披露 P3 在饱和面无增量）──
    segs = build_segments()
    pairs = gold_pairs(segs)
    w = Window(2000, [WindowSeg(s) for s in segs])
    top1_hits = sum(
        1 for q, t in pairs
        if (hh := [h for h in Recaller(None, w).search(q, k=3) if h.score > 0.0])
        and hh[0].seg.segment_id == t.segment_id)
    print(f"commit={commit}")
    print(f"[face-0 saturated-gold] P1 top-1 = {top1_hits}/{len(pairs)} "
          f"（饱和：P3 在此面增量预期=0，仅披露）")

    # ── 面 1：非饱和 probe（P1 基线，零 LLM 确定性）──
    print("[face-1 non-saturated probe] P1 top-1 基线（P3 价值面）:")
    p1_hits = 0
    for q, corpus, target in _PROBES:
        top1, scores = p1_top1(q, corpus)
        hit = top1 == target
        p1_hits += 1 if hit else 0
        print(f"  P1={'HIT ' if hit else 'MISS'} q={q!r}  top1={scores[0] if scores else None}")

    # ── 面 2：活 LLM rerank 增量（NO-FABRICATION：无活端点 → REFUSED）──
    ep = None
    if run:
        try:
            from sc.live_endpoint import resolve_live_endpoint
            ep = resolve_live_endpoint()
        except RuntimeError as e:
            print(f"\n[face-2 LLM-rerank-increment] REFUSED (no-fabrication guard): {e}")
            print("  P3 增量数 NOT emitted —— 对 FakeEndpoint 出增量 = 循环论证（同 75% 纪律）。")
            return 0
    if ep is None and not run:
        print("\n[face-2 LLM-rerank-increment] REFUSED: 无 LIVE_ENDPOINT_*/OPENAI_* env，"
              "活 LLM 面不出数（P1 基线面已出）。")
        print("  出数: LIVE_ENDPOINT_URL=… LIVE_ENDPOINT_KEY=… LIVE_ENDPOINT_MODEL=<真模型名> "
              "&& python bench/rerank_probe.py --run")
        return 0

    # --run：活 LLM 增量面（宿主主模型端点注入，零新 provider / 零模型名硬编码）
    if ep is None:                       # --run 且 resolve 失败已在上方 return；此路仅类型收窄
        print("\n[face-2 LLM-rerank-increment] REFUSED: 活端点未解析（同上方 75% 纪律，不出数）。")
        return 0
    cache: dict = {}
    p3_hits = 0
    for q, corpus, target in _PROBES:
        segs2 = [_mkseg(t, seq=i) for i, t in enumerate(corpus)]
        w2 = Window(2000, [WindowSeg(s) for s in segs2])
        hits = [Hit(h.seg, h.score) for h in Recaller(None, w2).search(q, k=3)
                if h.score > 0.0]
        # probe 面强制 gate 触发（阈值 inf）：非饱和面本来就是 P3 的价值面
        reranked = rerank_hits(hits, q, ep, cache=cache, window_version=1,
                               gate=RerankGate(low_score_threshold=float("inf")))
        hit = bool(reranked) and reranked[0].seg.original_text == target
        p3_hits += 1 if hit else 0
        print(f"  P3={'HIT ' if hit else 'MISS'} q={q!r}  "
              f"top1={reranked[0].seg.original_text[:20] if reranked else None}")
    print(f"\nRESULT commit={commit} model={ep.model} (model-specific，换档重跑)")
    print(f"  face-0 saturated-gold P1 top-1 = {top1_hits}/{len(pairs)} (P3 增量预期 0)")
    print(f"  face-1 P1 top-1 (non-saturated) = {p1_hits}/{len(_PROBES)}")
    print(f"  face-2 P3 rerank top-1 (non-saturated, LIVE) = {p3_hits}/{len(_PROBES)}  "
          f"← 增量面（可 0，如实披露）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
