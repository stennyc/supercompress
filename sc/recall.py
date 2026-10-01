"""Block ② memory_recall：点查 O(1) + BM25 top-k（P0 boost）+ Pull 预算 enforce。

设计锁（gate 定稿）：
- 召回数据面 = 单一事实源 jsonl（store 委托 scdata 权威层，单 replay 路径）——
  逐出后原文可全量取回（可逆，非单向有损）；
- 点查 O(1)：segment_id → 原文（P0 0-loss）；
- 排序（P1，2026-10-01 换层，数据面/点查/三元组零动，`search()`/`Hit`/`Recaller` 公共 API 不变）：
  **BM25（tf-idf，k1=1.2 / b=0.75）** × P0 boost × stale P1 降权。
  分词 = 拉丁词 + CJK unigram+bigram（纯标准库：无 jieba / 无网络 / 无 dict 加载），
  让中文/日韩 query 无需空格即可召回（现状 `\S+` 词重叠面把整段中文当一个 token，CJK query 命中≈0）。
  **确定性**：BM25 只读固定语料统计（N/df/avgdl，corpus = 当前 window），
  CJK bigram 按字面左到右生成、查询 term 按 sorted 序累加 → 同 query 两次结果全等；
  可选 seq 时效降权（`decay_lambda>0`）为纯函数、无 wall-clock，默认 0 不扰动既有钉死面。
- Pull 预算 = 共享分母（post-distill usable × 20%）一处定义：
  P2 优先降权、P1 超预算累计降权、P0 豁免强制通过（超预算写 budget_breach 日志，不静默）。

P1 依据：[决定 R；文档 S-Phase 3；plan P1.1/P1.2]。仅换 `search()` 的命中计分函数，
`Recaller.__init__/point_fetch/budget/breach_log`、`Hit` 面、预算下权循环逐字保留。
"""
from __future__ import annotations

import math
import re

from .budgets import pull_budget
from .schema import est_tokens

# ── 分词（纯标准库、确定性；CJK unigram 保词根 + bigram 保双字词，100+ 语言覆盖）────
# 拉丁词：小写 [a-z0-9_]+（标点/空格/斜杠为界）。
_LATIN_RE = re.compile(r"[a-z0-9_]+")
# CJK 表意文字（统一表意 + 扩展 A + 兼容区）：抓**连续**字串，再逐段出单字 + 相邻双字。
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+")


def tokenize(text: str) -> list[str]:
    """确定性多集分词：拉丁词（小写）+ CJK 单字 + CJK 相邻双字。

    左到右扫描，纯字面规则（无 jieba / 无网络 / 无 dict 加载）→ 同输入同输出。
    返回**多集**（保留重复，供 BM25 词频 tf）；调用方需要集合视图（query 去重 / idf 计 df）
    自行 `set()` / 计 freq。
    """
    s = (text or "").lower()
    toks: list[str] = _LATIN_RE.findall(s)
    for run in _CJK_RUN_RE.findall(s):
        for i in range(len(run)):
            toks.append(run[i])                 # unigram（保词根）
            if i + 1 < len(run):
                toks.append(run[i:i + 2])       # bigram（保双字词）
    return toks


class Hit:
    def __init__(self, seg, score: float, downweighted: bool = False, note: str = "") -> None:
        self.seg = seg
        self.score = score
        self.downweighted = downweighted
        self.stale = bool(getattr(seg, "stale", False))
        self.note = note


class Recaller:
    # P1 定稿参数（plan P1.2）：BM25 k1/b + 相关性增强因子。
    BM25_K1 = 1.2
    BM25_B = 0.75
    P0_BOOST = 1.5          # 同相关度 P0 优先（保 test_bm25_top3_p0_boost）
    STALE_DECAY = 0.5       # stale P1 召回降权（保 test_stale_p1_downweighted）

    def __init__(self, store, window) -> None:
        self.store = store
        self.window = window
        self.budget = pull_budget(window)      # Pull 预算读共享分母（一处定义）
        self.breach_log: list[dict] = []       # P0 超预算强制通过 → budget_breach（不静默）

    # ───────────── 点查 O(1) ─────────────
    def point_fetch(self, segment_id: str):
        """P0 0-loss：segment_id（行级唯一）→ 原文 Segment，miss = None。"""
        return self.store.get(segment_id)

    # ───────────── 语义召回（top-k，P1 BM25 层）─────────────
    def search(self, query: str, k: int = 3, decay_lambda: float = 0.0) -> list[Hit]:
        """BM25 top-k 排序 + Pull 预算 enforce（数据面/点查/预算循环不变，仅命中计分换层）。

        - 语料 = 当前 `self.window`（固定 corpus；N/df/avgdl 每 query 一次性算）；
        - 相关性 base = Σ_{t∈query} idf(t)·(tf·(k1+1))/(tf + k1·norm)，
          idf 非负变体 ln(1+(N-df+0.5)/(df+0.5))；query term 按 sorted 序累加 → 跨进程确定。
        - P0 boost ×P0_BOOST；stale P1 ×STALE_DECAY；可选 seq 时效降权（decay_lambda>0 时，
          同分下越早 seg.seq 越多被降，纯函数无 wall-clock；默认 0 不扰动钉死面）。
        - 排序键 (-base, tier(P0先), seq) 与预算下权循环**逐字保留** → 4 条钉死面不变。
        """
        q_set = set(tokenize(query))
        segs = [ws.segment for ws in self.window.segs]
        n = len(segs)

        # 语料统计（固定 corpus）：每段 tf 多集 + 长度；全局 df。
        doc_tfs: list[dict] = []
        doc_lens: list[int] = []
        df: dict[str, int] = {}
        for seg in segs:
            tf: dict[str, int] = {}
            for t in tokenize(seg.original_text):
                tf[t] = tf.get(t, 0) + 1
            doc_tfs.append(tf)
            doc_lens.append(sum(tf.values()))
            for t in tf:
                df[t] = df.get(t, 0) + 1
        avgdl = (sum(doc_lens) / n) if n else 0.0
        k1, b = self.BM25_K1, self.BM25_B

        def idf(t: str) -> float:
            d = df.get(t, 0)
            return math.log(1.0 + (n - d + 0.5) / (d + 0.5))   # 非负变体，df=n 时仍 >0

        tidx = {"P0": 0, "P1": 1, "P2": 2}
        scored = []
        for i, seg in enumerate(segs):
            tf = doc_tfs[i]
            dlen = doc_lens[i]
            if dlen == 0 or avgdl == 0.0:
                base = 0.0
            else:
                base = 0.0
                norm = 1.0 - b + b * (dlen / avgdl)            # >0（dlen,avgdl>0）
                for t in sorted(q_set):                        # sorted 序 → 跨进程确定累加
                    f = tf.get(t, 0)
                    if f == 0:
                        continue
                    base += idf(t) * (f * (k1 + 1)) / (f + k1 * norm)
            if seg.tier == "P0":
                base *= self.P0_BOOST
            if getattr(seg, "stale", False):
                base *= self.STALE_DECAY
            if decay_lambda > 0.0 and n > 0:
                base *= max(0.0, 1.0 - decay_lambda * (float(n) - float(seg.seq)))
            scored.append((-base, tidx[seg.tier], seg.seq, seg))
        scored.sort()                                      # 相关度 desc → 层级（P0 先）→ seq 稳定序

        # ── 预算下权（top-k 成员按相关性定；P1 降权按时序累计，与相关性排序解耦）──
        # 旧平直 scorer 下 P1 累计恰好落在相关性序（同分→seq），BM25 换层后近同文 P1 产生
        # base 微差、相关序与 seq 序可能分离 → 若仍按相关序累计会翻「最早 P1 留预算内」语义。
        # 故 P1 超预算判定固定按 seq（时序）累计：最早的 P1 先占 pull 预算，scorer 无关
        # （保 test_pull_budget_20pct_enforced 的「预算内 P1 保留 / 超预算 P1 降权 / P2 优先降权」）。
        topk_segs = [seg for _, _, _, seg in scored[:k]]
        p1_by_seq = sorted((s for s in topk_segs if s.tier == "P1"), key=lambda s: s.seq)
        p1_cum = 0
        p1_dw: dict = {}                                  # seg_id → downweighted
        for s in p1_by_seq:
            p1_cum += est_tokens(s.original_text)
            p1_dw[s.segment_id] = p1_cum > self.budget    # 时序累计超预算 → 降权

        hits: list[Hit] = []
        for negbase, _, _, seg in scored[:k]:
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
                dw = p1_dw[seg.segment_id]                  # P1 时序累计超预算降权
                if dw:
                    note = "pull budget overflow, downweighted"
            if getattr(seg, "stale", False):
                note = (note + " · " if note else "") + "stale P1 · verify before acting"
            hits.append(Hit(seg, -negbase, dw, note))
        return hits
