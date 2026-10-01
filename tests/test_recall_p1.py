"""P1（S-Phase 3）BM25+CJK 召回层测试面。

覆盖：DoD#2 确定性（同 query 两次全等）、CJK unigram+bigram 分词、
P1.3 gold 门禁（P1 recall@3 ≥ 旧词重叠基线 ×1.30）。
宿主零改动、纯 sc/bench 面（依据 plan P1.x；数据面/点查/预算循环不变）。
"""
import re

from sc.schema import Segment
from sc.distill import Window, WindowSeg
from sc.recall import Recaller, tokenize
from helpers import mkseg, mk_window


def _seg(text, tier="P1", seq=0):
    return Segment(tier=tier, provenance="model", original_text=text, seq=seq, ts=0.0)


class TestP1_Tokenize:
    def test_cjk_unigram_and_bigram(self):
        """CJK 连续字串 → 单字 + 相邻双字（左到右、确定性）。"""
        toks = tokenize("端口占用")
        # unigram: 端 口 占 用 ; bigram: 端口 口占 占用
        assert "端口" in toks and "占用" in toks
        assert "端" in toks and "用" in toks
        # 确定性：两次同输出
        assert tokenize("端口占用") == tokenize("端口占用")

    def test_latin_lowercased(self):
        assert tokenize("Deploy Release 42") == ["deploy", "release", "42"]

    def test_mixed_cjk_and_latin(self):
        """中英混排：拉丁词 + CJK 各自动词。"""
        toks = tokenize("网关nginx限流")
        assert "nginx" in toks
        assert "网关" in toks and "限流" in toks

    def test_empty(self):
        assert tokenize("") == []


class TestP1_Determinism:
    def test_same_query_twice_identical(self, store, aux):
        """DoD#2：同 query 两次 search 结果全等（纯函数、无 wall-clock/随机）。"""
        p0 = mkseg(store, "auth service port 8080 bound to 192.168.55.55", "P0", "user", structural=True)
        p1 = mkseg(store, "部署发布版本四十二到生产环境", "P1")
        p2 = mkseg(store, "raw dump verbose unrelated", "P2")
        w = mk_window(2000, [p0, p1, p2])
        r = Recaller(store, w)
        a = [(h.seg.segment_id, round(h.score, 9)) for h in r.search("端口 8080 认证", k=3)]
        b = [(h.seg.segment_id, round(h.score, 9)) for h in Recaller(store, w).search("端口 8080 认证", k=3)]
        assert a == b, "确定性破坏：两次结果不一致"

    def test_decay_lambda_zero_invariant(self, store, aux):
        """decay_lambda 默认 0 → 不扰动既有排序（钉死面回归护）。"""
        p0 = mkseg(store, "deploy release 42 to srv", "P0", "user", structural=True)
        p2 = mkseg(store, "deploy deploy dump dump release release", "P2")
        w = mk_window(1000, [p0, p2])
        r = Recaller(store, w)
        hits = r.search("42 srv deploy release", k=2)
        assert hits[0].seg.segment_id == p0.segment_id, "P0 boost 排前（默认 λ=0 不变）"

    def test_decay_lambda_positive_demotes_oldest_and_newest_undimmed(self, store, aux):
        """Minor(e)：decay_lambda>0 → 同分下越早 seq 越多被降权；最新段（seq=n-1）因子 1.0 不降权。

        off-by-λ 钉死（sc/recall.py `1.0-λ(n-1-seq)`）：seq=n-1 → 因子 1.0，seq=0 → 1-λ(n-1)。
        三条同文 P1（base 全等）λ=0.5 时严格 [seq2 > seq1 > seq0]（无 tie 可依赖）；
        P0(seq0) 在 λ=0.5 被时序压过最新 P1(seq2) → P0 不再无条件排第一。
        确定性：两次全等（纯函数、无 wall-clock）。
        """
        from sc.recall import Recaller
        w = mk_window(1000, [
            mkseg(store, "same base text here", "P0", "user", structural=True),  # seq0
            mkseg(store, "same base text here", "P1"),                             # seq1
            mkseg(store, "same base text here", "P1"),                             # seq2（最新）
        ])
        r = Recaller(store, w)
        q = "same base text here"
        h_lam05 = r.search(q, k=3, decay_lambda=0.5)
        order = [h.seg.seq for h in h_lam05]
        assert order == [2, 1, 0], f"λ=0.5 同文应 [最新>中>最早]，实际 {order}"
        # P0(seq0) 被时序压过最新 P1(seq2)
        assert h_lam05[0].seg.tier == "P1" and h_lam05[0].seg.seq == 2, "最新 P1 应第一"
        # 严格 >（非 tie 兜底）：seq2 未降权 > seq0 降权最多
        assert h_lam05[0].score > h_lam05[2].score, "off-by-λ：最新段(因子1.0) > 最早段(因子1-λ(n-1))"
        # 确定性：同 query 两次全等
        b = [(h.seg.seq, round(h.score, 9)) for h in r.search(q, k=3, decay_lambda=0.5)]
        c = [(h.seg.seq, round(h.score, 9)) for h in Recaller(store, w).search(q, k=3, decay_lambda=0.5)]
        assert b == c, "decay_lambda>0 结果非确定"
        # 回退默认 λ=0：P0 boost 恢复排第一（钉死面，off-by-λ 不影响 λ=0 面）
        assert Recaller(store, w).search(q, k=3)[0].seg.tier == "P0", "λ=0 → P0 boost 仍排第一"


class TestP1_CjkRecall:
    def test_cjk_no_space_query_recalls_cjk_segment(self, store, aux):
        """P1 区分度核心：无空格 CJK query 命中 CJK 段（旧词重叠面命中≈0）。"""
        cjk = mkseg(store, "部署发布版本四十二到生产环境", "P0", "user", structural=True)
        eng = mkseg(store, "auth service port 8080 bound to 192.168.55.55", "P1")
        w = mk_window(2000, [cjk, eng])
        hits = Recaller(store, w).search("部署发布版本", k=3)
        assert hits and hits[0].seg.segment_id == cjk.segment_id, "CJK 无空格 query 未命中 CJK 段"
        assert hits[0].score > 0


class TestP1_GoldGate:
    def test_p1_recall_at3_beats_baseline(self, store, aux):
        """P1.3 DoD#1：P1 recall@3 ≥ 旧词重叠基线 ×1.30（gold 集，确定性复跑）。"""
        from bench.gold_recall_set import build_segments, gold_pairs
        from bench.gold_recall_bench import baseline_topk, recall_at_k

        segs = build_segments()
        pairs = gold_pairs(segs)
        k = 3
        base_hits = sum(1 for q, t in pairs if recall_at_k(baseline_topk(q, segs, k), t))

        p1_hits = 0
        for q, t in pairs:
            w = Window(2000, [WindowSeg(s) for s in segs])
            top = [h.seg for h in Recaller(None, w).search(q, k=k) if h.score > 0.0]
            p1_hits += 1 if recall_at_k(top, t) else 0

        base_r = base_hits / len(pairs)
        p1_r = p1_hits / len(pairs)
        gate = p1_r >= base_r * 1.30 if base_r > 0 else p1_r > base_r
        print(f"[gold] base={base_hits}/{len(pairs)} ({base_r:.3f}) p1={p1_hits}/{len(pairs)} ({p1_r:.3f})")
        assert gate, f"P1 门禁未达：p1={p1_r:.3f} < base×1.30 ({base_r * 1.30:.3f})"
