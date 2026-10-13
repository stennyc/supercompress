"""Block ② memory_recall：点查 O(1) + BM25 top-k（P0 boost）+ Pull 预算 enforce。

设计锁（gate 定稿）：
- 召回数据面 = 单一事实源 jsonl（store 委托 scdata 权威层，单 replay 路径）——
  逐出后原文可全量取回（可逆，非单向有损）；
- 点查 O(1)：segment_id → 原文（P0 0-loss）；
- 排序（P1，2026-10-01 换层，数据面/点查/三元组零动，`search()`/`Hit`/`Recaller` 公共 API 不变）：
  **BM25（tf-idf，k1=1.2 / b=0.75）** × P0 boost × stale P1 降权。
  分词 = 拉丁词 + CJK unigram+bigram（纯标准库：无 jieba / 无网络 / 无 dict 加载），
  让中文/日韩 query 无需空格即可召回（现状 \\S+ 词重叠面把整段中文当一个 token，CJK query 命中≈0）。
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
from .schema import Segment, est_tokens

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


# ── extended-BM25 精度基座（P1 · 纯标准库、确定性、零新模型）────────────────
# 在既有 BM25（非负 idf + sorted term 累加 + CJK 字面 bigram）之上加三招：
# ① 高信号 exact-match boost：三类标识符信号（纯数字 token / 路径 run / 下划线
#    标识符，AC/文件/函数名里的锚点）命中时按 term 类乘独立 boost 因子；
# ② 字段加权：P0 段 goal/open_tasks(AC) 字段 token 计 FIELD_WEIGHT 加权副本
#    （_field_tokens；P0 字段段权重 > 填充物段，按 slot/tier 确定性拉开）；
# ③ 时间衰减：Recaller(store, window, decay_lambda) 构造接线面——search() 缺省用
#    实例默认（engine 传入，按参数控制启用）；默认 0 不扰动钉死面，纯函数无 wall-clock。
# 全部为纯字面/纯函数规则（无 jieba/无网络/无 dict/无随机）→ 同 query 两次结果全等（G4）。

# 路径 run（确定性正则）：含 / 或 \ 的 ASCII 路径串（小写化；Windows C:/ 含冒号亦归）。
# 独立于 tokenize()（\S 词切分把 C:/srv 切成 c、srv）→ 路径面自成一词空间（df 独立计）。
_PATH_RUN_RE = re.compile(r"(?:[a-z0-9_.:\-]*[/\\][a-z0-9_.:\-]*)+")
# 纯数字 token：AC/配置里的量值（端口/版本/阈值/条数）——比散字段词更具体 → 叠乘更高 boost。
_NUMERIC_TOKEN_RE = re.compile(r"^[0-9]+$")


def path_terms(text: str) -> list[str]:
    """确定性路径 run 面（多集）：原文里的 / \\ 路径串（小写化）。同输入同输出。"""
    return _PATH_RUN_RE.findall((text or "").lower())


def is_numeric_token(t: str) -> bool:
    """纯数字 token（端口/版本/阈值/条数量值）→ exact-match 高信号。"""
    return bool(_NUMERIC_TOKEN_RE.match(t))


def _field_tokens(seg: "Segment") -> list[str]:
    """字段加权面（招②）：P0 段的 goal/open_tasks 字段 token（多集，供加权 tf 副本）。

    仅 P0 段取字段（P0 是权威结构面：user-sourced goal + AC 任务表）；P1/P2 填充物段
    字段 token 不参与加权（权重 1.0，靠 tier 自然下沉）→ 字段加权仍确定性、按 slot/tier。
    open_tasks 每项 = id + desc 两串拼接（AC 里的数字/标识符词在此被放大权重）。
    """
    if seg.tier != "P0":
        return []
    toks: list[str] = []
    goal = seg.goal
    if goal:
        toks.extend(tokenize(str(goal)))
    for t in (seg.open_tasks or []):
        if not isinstance(t, dict):
            continue
        toks.extend(tokenize(str(t.get("id", "")) + " " + str(t.get("desc", ""))))
    return toks


class Hit:
    def __init__(self, seg: "Segment", score: float, downweighted: bool = False, note: str = "") -> None:
        self.seg = seg
        self.score = score
        self.downweighted = downweighted
        self.stale = bool(getattr(seg, "stale", False))
        self.note = note


class Recaller:
    # P1 定稿参数（plan P1.2）：BM25 k1/b + 相关性增强因子。
    BM25_K1 = 1.2
    BM25_B = 0.75
    P0_BOOST = 1.5          # user P0 同相关度优先（保 test_bm25_top3_p0_boost）
    # R3（2026-10-10 · 缺口 D）：P0 boost 按 provenance 分叉——
    # user P0（用户拍板/指令）= P0_BOOST=1.5（不变）；
    # model P0（agent 自述，provenance="model"）= MODEL_P0_BOOST=1.0（不享用户级加权，
    # 压到"用户 P0 之下、普通 P1 之上"中间带：真有用的 agent 实证段强相关仍可召，
    # 但不再拿 1.5 加权冒充用户拍板）。分叉轴 = 谁说的（user vs model），零模型名依赖
    # （supercompress 早自适应任意 model，此处只按 provenance 段来源分叉，不碰 model_version）。
    MODEL_P0_BOOST = 1.0
    STALE_DECAY = 0.5      # stale P1 召回降权（保 test_stale_p1_downweighted）
    # extended-BM25 精度基座（P1 三招，纯字面系数、确定性）：
    EXACT_MATCH_BOOST = 2.0   # 招①：标识符型 term（路径/点分标识符）命中 → 该 term 贡献 ×2
    NUMERIC_EXACT_BOOST = 1.5 # 招①：纯数字 term 再叠乘 ×1.5（AC/配置里的量值最具体 → 总 ×3）
    FIELD_WEIGHT = 2.0        # 招②：P0 goal/AC(open_tasks) 字段 token 加权副本倍数（>1 放大权重）
    # 招③：实例默认 decay_lambda（engine 接线面）——search() 未显式给值时用本默认；
    # 默认 0.0 = 不启用（钉死面零扰动），engine 按参数控制启用面（>0 时同分下越早 seq 越多被降）。
    DEFAULT_DECAY_LAMBDA = 0.0

    def __init__(self, store, window, decay_lambda: float | None = None) -> None:
        self.store = store
        self.window = window
        # 招③：时间衰减实例默认（None → DEFAULT_DECAY_LAMBDA=0 钉死面；engine 传 >0 启用）
        self.decay_lambda = self.DEFAULT_DECAY_LAMBDA if decay_lambda is None else decay_lambda
        self.budget = pull_budget(window)      # Pull 预算读共享分母（一处定义）
        self.breach_log: list[dict] = []       # P0 超预算强制通过 → budget_breach（不静默）

    # ───────────── 点查 O(1) ─────────────
    def point_fetch(self, segment_id: str):
        """P0 0-loss：segment_id（行级唯一）→ 原文 Segment，miss = None。"""
        return self.store.get(segment_id)

    # ───────────── 语义召回（top-k，P1 BM25 层）─────────────
    def search(self, query: str, k: int = 3, decay_lambda: float | None = None) -> list[Hit]:
        """BM25 top-k 排序 + Pull 预算 enforce（数据面/点查/预算循环不变，仅命中计分换层）。

        - 语料 = 当前 `self.window`（固定 corpus；N/df/avgdl 每 query 一次性算）；
        - 相关性 base = Σ_{t∈query} boost(t)·idf(t)·(tf·(k1+1))/(tf + k1·norm)，
          idf 非负变体 ln(1+(N-df+0.5)/(df+0.5))；query term 按 sorted 序累加 → 跨进程确定。
        - extended-BM25 三招（纯字面系数、确定性，G4 钉死面不破）：
          ① exact-match boost：标识符型 term（路径 `is_path_token` / 纯数字
             `is_numeric_token`）贡献 ×EXACT_MATCH_BOOST（纯数字再叠 ×NUMERIC_EXACT_BOOST）；
          ② 字段加权：P0 段 goal/open_tasks(AC) 字段 token 计 FIELD_WEIGHT 加权副本进 tf
             （_field_tokens；P0 字段段权重 > 填充物段，按 slot/tier 确定性拉开）；
          ③ 时间衰减：decay_lambda 缺省 → 实例默认 self.decay_lambda（engine 接线面）；
             启用时同分下越早 seq 越多被降，纯函数无 wall-clock；默认 0 不扰动钉死面。
        - P0 boost ×P0_BOOST；stale P1 ×STALE_DECAY；排序键 (-base, tier(P0先), seq)
          与预算下权循环**逐字保留** → 4 条钉死面不变。
        """
        if decay_lambda is None:                    # 招③：缺省 → 实例默认（按参数控制启用面）
            decay_lambda = self.decay_lambda
        q_set = set(tokenize(query))
        q_path = set(path_terms(query))              # 招①：路径 run 面（独立词空间）
        segs = [ws.segment for ws in self.window.segs]
        n = len(segs)

        # 语料统计（固定 corpus）：每段 tf 多集 + 长度；全局 df。
        # 招② 字段加权：P0 段 goal/AC(open_tasks) 字段 token 计 FIELD_WEIGHT 加权副本
        # （_field_tokens 仅 P0 段非空、P1/P2 填充物段 0 副本 → 权重按 slot/tier 拉开）。
        # 招① 路径面：路径 run 自成 tf/df（独立词空间；tokenize 把 C:/srv 切成 c、srv，
        # 路径整串在此独立计 → 高信号锚点不被词切碎片稀释）。
        doc_tfs: list[dict] = []
        doc_lens: list[float] = []
        df: dict[str, int] = {}
        path_tfs: list[dict] = []
        df_path: dict[str, int] = {}
        fw = self.FIELD_WEIGHT
        for seg in segs:
            tf: dict = {}
            for t in tokenize(seg.original_text):
                tf[t] = tf.get(t, 0.0) + 1.0
            for t in _field_tokens(seg):            # 招②：字段 token 加权副本
                tf[t] = tf.get(t, 0.0) + fw
            pt: dict = {}
            for t in path_terms(seg.original_text):  # 招①：路径 run tf
                pt[t] = pt.get(t, 0.0) + 1.0
            path_tfs.append(pt)
            for t in pt:
                df_path[t] = df_path.get(t, 0) + 1
            doc_tfs.append(tf)
            doc_lens.append(sum(tf.values()))
            for t in tf:
                df[t] = df.get(t, 0) + 1
        avgdl = (sum(doc_lens) / n) if n else 0.0
        k1, b = self.BM25_K1, self.BM25_B

        def idf(t: str) -> float:
            d = df.get(t, 0)
            return math.log(1.0 + (n - d + 0.5) / (d + 0.5))   # 非负变体，df=n 时仍 >0

        def idf_path(t: str) -> float:
            d = df_path.get(t, 0)
            return math.log(1.0 + (n - d + 0.5) / (d + 0.5))   # 路径面独立 df，同非负变体

        tidx = {"P0": 0, "P1": 1, "P2": 2}
        em, nmb = self.EXACT_MATCH_BOOST, self.NUMERIC_EXACT_BOOST   # 招① 系数
        # 招① 高信号 exact-match boost：按 term 类（纯字面判定、确定性）——
        # 路径 run ×EXACT_MATCH_BOOST；下划线标识符（函数名/变量名）×EXACT_MATCH_BOOST；
        # 纯数字（AC/配置量值，最具体）×EXACT_MATCH_BOOST×NUMERIC_EXACT_BOOST（叠乘）。
        def term_boost(t: str) -> float:
            if is_numeric_token(t):
                return em * nmb
            if "_" in t and t != "_" and t != "_" * len(t):
                return em
            return 1.0

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
                    f = tf.get(t, 0.0)
                    if f == 0:
                        continue
                    base += term_boost(t) * idf(t) * (f * (k1 + 1)) / (f + k1 * norm)
                # 招① 路径 run 命中：路径面独立 tf/df，×EXACT_MATCH_BOOST
                pt = path_tfs[i]
                for t in sorted(q_path & pt.keys()):
                    f = pt[t]
                    base += em * idf_path(t) * (f * (k1 + 1)) / (f + k1 * norm)
            if seg.tier == "P0":
                # R3（2026-10-10 · 缺口 D）：P0 boost 按 provenance 分叉——
                # user P0（用户拍板/指令）= P0_BOOST=1.5（不变，保 test_bm25_top3_p0_boost）；
                # model P0（agent 自述，provenance="model"）= MODEL_P0_BOOST=1.0（不享用户级
                # 加权，避免 agent 自述段拿 1.5 冒充拍板排序到用户 P0 之上）。分叉轴 = 谁说的，
                # 零模型名依赖（supercompress 自适应任意 model，此处只读段 provenance 字段）。
                if getattr(seg, "provenance", "") == "model":
                    base *= self.MODEL_P0_BOOST
                else:
                    base *= self.P0_BOOST
            if getattr(seg, "stale", False):
                base *= self.STALE_DECAY
            if decay_lambda > 0.0 and n > 0:
                # seq=n-1（窗口最新段）→ 因子 1.0 不降权；seq=0（最早段）→ 1-λ(n-1) 降最多。
                # max(0.0,…) 钉在 0，永不产生负乘子把无关段重排到零分段之下。
                base *= max(0.0, 1.0 - decay_lambda * (float(n) - 1.0 - float(seg.seq)))
            scored.append((-base, tidx[seg.tier], seg.seq, seg))
        # 排序 key 法（P0 surgical）：显式 scalar key 让三要素（-base, tier, seq）全 tie 时
        # 按第 4 位 segment_id（content_hash[:16]+seq，跨 store 唯一行级 id）定全序，杜绝
        # 裸 Segment 比较。普通 @dataclass 无 order=True → '<' 不支 → 原裸 sort() 在 tie 面
        # 抛 TypeError → engine.py:1163-1164 select_context fail-open 返回 None（22.5h+
        # 慢性 no-op 根因；union-window 下主 store + model 旁路 seq 各自计数可碰撞同 seq）。
        # key 第 1-3 位与原 4-tuple 前 3 位逐一相同 → top-k 相对次序零变化；仅把裸 Segment
        # 比较换成合法全序（不碰 Segment 类定义 / P1 BM25 打分子 / _s1_tail_deliver 路径）。
        scored.sort(key=lambda t: (t[0], t[1], t[2], t[3].segment_id))

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
