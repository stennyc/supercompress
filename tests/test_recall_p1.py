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


class TestP1_DecayNegatives:
    """招③ 时间衰减负例（G2/G4 扩展面：启用 decay 时按预期改变排序，钉死基座不破）。

    与 TestP1_Determinism 的 happy 路径互补：
    - ① λ 足够大 → 最早段被 max(0.0, 1-λ(n-1-seq)) 钉到 0（负乘子不会把无关段翻到零分段之上）；
    - ② 单调性：decay 是 seq 的纯函数 → 逐段 decayed ≤ undimmed，seq=n-1 因子 1.0 严格相等；
    - ③ 无 tie 不翻转：各段 base 相关度严格分开时，任何 decay（甚至满降 0）都不改变 top-k 名次。
    全部纯函数、无 wall-clock、无随机 → 两次全等（G4 确定性基座）。
    """

    def test_decay_excess_oldest_floored_to_zero_newest_undimmed(self, store, aux):
        """负例①：λ=0.5、n=3 → 逐段因子 [0.0, 0.5, 1.0]（seq 0/1/2，公式 1-λ(n-1-seq) 钉 0）。

        off-by-λ 钉死（sc/recall.py `max(0.0, 1.0-λ(n-1-seq))`）：
        - seq=0（最早）→ 1-0.5×2=0 → 乘子 0 → 最早段 score 钉 0、沉底；
        - seq=1（中）→ 0.5（非 0 非 1，验证逐段因子非二值、按 seq 线性）；
        - seq=2（最新）→ 1.0 不降权（score 恰等于其 λ=0 时 base）。
        P0 boost 面同时锁定：λ=0 时 P0 段（同文 tf）= 非 P0 段 base × P0_BOOST=1.5。
        负例：若 max(0.0,…) 缺失（允许负乘子），最早段 score <0 → 会沉到零分段之下、
        被无关段（score 0）翻到上面 → 本断言捕获该回归。
        """
        p0 = mkseg(store, "anchor probe", "P0", "user", structural=True)       # seq0
        p1 = mkseg(store, "anchor probe", "P1")                                # seq1
        p2 = mkseg(store, "anchor probe", "P2")                                # seq2（最新）
        w = mk_window(5000, [p0, p1, p2])
        lam0 = {h.seg.seq: h.score for h in Recaller(store, w).search("anchor probe", k=3, decay_lambda=0.0)}
        assert lam0[0] == Recaller(store, w).P0_BOOST * lam0[1], "P0 boost 面：λ=0 时 P0 = 非 P0 base × 1.5"
        assert lam0[1] > 0.0, "对照面：λ=0 时同文段 base 应 > 0"
        hits = Recaller(store, w).search("anchor probe", k=3, decay_lambda=0.5)
        assert [h.seg.seq for h in hits] == [2, 1, 0], f"λ=0.5 同文应 [最新>中>最早]，实际 {[h.seg.seq for h in hits]}"
        sc = {h.seg.seq: h.score for h in hits}
        assert sc[2] == lam0[2], "seq=n-1 因子 1.0：最新段不降权（score 应等于其 λ=0 base）"
        assert sc[1] == 0.5 * lam0[1], f"seq=1 因子 0.5（应 {0.5*lam0[1]!r}，实际 {sc[1]!r}）"
        assert sc[0] == 0.0, f"seq=0（最早）应钉 0，实际 {sc[0]!r}（max(0.0,…) 负例失败）"
        # 确定性负例：两次全等
        again = {h.seg.seq: round(h.score, 9) for h in Recaller(store, w).search("anchor probe", k=3, decay_lambda=0.5)}
        assert again == {2: round(sc[2], 9), 1: round(sc[1], 9), 0: round(sc[0], 9)}, "λ=0.5 钉 0 面非确定"

    def test_decay_monotone_per_segment_and_zero_length_undisturbed(self, store, aux):
        """负例②：decay 逐段单调 → 对任意 λ>0，score_λ(seq) ≤ score_0(seq)，seq=n-1 时严格相等；
        零长段（dlen=0 → base 0）不受 decay 扰动（0×因子=0，不会因 max(0.0,…) 变负）。

        纯函数性（seq 的函数）由 ② 锁定：同一 seg 在两个不同 λ 下因子随 λ 单调增（降权单调加深）。
        """
        p0 = mkseg(store, "alpha token", "P0", "user", structural=True)   # seq0
        p1 = mkseg(store, "alpha token", "P1")                            # seq1
        p2 = mkseg(store, "beta gamma delta", "P2")                       # seq2（不同文，与 P1 段无 tie）
        w = mk_window(5000, [p0, p1, p2])
        r = Recaller(store, w)
        undimmed = {h.seg.seq: h.score for h in r.search("alpha beta", k=3, decay_lambda=0.0)}
        # 逐 λ 扫描（0.3 / 0.6）：单调且 ≤ undimmed；seq=2（n-1）因子 1.0 → 严格相等
        prev = dict(undimmed)
        for lam in (0.3, 0.6):
            lam_scores = {h.seg.seq: h.score for h in r.search("alpha beta", k=3, decay_lambda=lam)}
            for seq, sc in lam_scores.items():
                assert sc <= prev[seq] + 1e-12, f"λ={lam} seq={seq} 未随 λ 单调降（{sc} > {prev[seq]}）"
            assert lam_scores[2] == prev[2], "seq=n-1 因子 1.0：最新段不降权（两次 λ 应全等）"
            # 零长段（空 text）base=0 → decay 不动（0 保持 0，非负乘子）
            prev = lam_scores
        # 零长段负例面（独立 window，避免与 alpha/beta 段共享 corpus 统计）：
        z0 = mkseg(store, "", "P0", "user", structural=True)
        z1 = mkseg(store, "", "P2")
        wz = mk_window(5000, [z0, z1])
        rz = Recaller(store, wz)
        for lam in (0.0, 0.5, 0.9):
            out = [round(h.score, 9) for h in rz.search("anything", k=2, decay_lambda=lam)]
            assert out == [0.0, 0.0], f"零长段 λ={lam} 应全 0，实际 {out}（负乘子或 dlen 面被扰动）"

    def test_decay_flips_topk_membership_as_expected(self, store, aux):
        """负例③（expected ranking change）：无 tie、相关度严格分开的三段同 tier 语料，
        decay 启用后 top-k 名次按「相关性 × seq 衰减」重排——这是 decay 的预期行为面
        （recency 优先），不是 bug；同时纯函数两次全等（G4）。

        设计：seq0 最相关（同词 tf=2，相关度最高）、seq2 次相关（tf=1）、seq1 无关 filler（base 0）。
        - λ=0（钉死面）：相关度序 [0, 2, 1] → top1 = 最相关段 seq0；
        - λ=1.0（满降）：seq0 因子 max(0, 1-1·2)=0 被钉 0 → 最相关段沉底；
          seq2 因子 1.0 保 base → top1 翻到 seq2（次相关但最新）。
          即「相关性最高但最旧」让位给「次相关但最新」= decay 的语义意图。
        - 确定性：λ=1.0 两次全等（纯函数、无 wall-clock）。
        """
        s0 = mkseg(store, "aaa bbb ccc ddd aaa bbb ccc ddd", "P1")     # seq0 最相关（tf=2）
        s1 = mkseg(store, "unrelated filler words here", "P1")          # seq1 无关（base 0）
        s2 = mkseg(store, "aaa bbb ccc ddd", "P1")                      # seq2 次相关（tf=1，最新）
        w = mk_window(5000, [s0, s1, s2])
        r = Recaller(store, w)
        q = "aaa bbb ccc ddd"
        order_lam0 = [h.seg.seq for h in r.search(q, k=3, decay_lambda=0.0)]
        assert order_lam0 == [0, 2, 1], f"λ=0 相关度序应 [0,2,1]（最相关最旧段在前），实际 {order_lam0}"
        order_lam1 = [h.seg.seq for h in r.search(q, k=3, decay_lambda=1.0)]
        assert order_lam1[0] == 2, f"decay 满降应把 top-k 头翻到次相关最新段（seq2），实际 {order_lam1}"
        # 最相关但最旧段被钉 0（满降因子 0）→ 沉到 base-0 的 filler 之上仅靠 seq tie-break（seq0<seq1）
        # 负例面：满降只按 seq 缩放、不产生负分（max(0.0,…) 护）
        s1s = {h.seg.seq: h.score for h in r.search(q, k=3, decay_lambda=1.0)}
        assert s1s[0] == 0.0 and s1s[1] == 0.0, f"满降 seq0/seq1 应钉 0，实际 {s1s}"
        # 确定性：两次全等
        a = [(h.seg.seq, round(h.score, 9)) for h in r.search(q, k=3, decay_lambda=1.0)]
        b = [(h.seg.seq, round(h.score, 9)) for h in Recaller(store, w).search(q, k=3, decay_lambda=1.0)]
        assert a == b, "λ=1.0 满降面非确定（两次结果不等）"


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
