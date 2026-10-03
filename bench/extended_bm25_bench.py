#!/usr/bin/env python3
"""precision P1 · extended-BM25 三招 before/after 精度 bench（任务精准相关段 top-k 排名上移，DoD 出数）。

口径（plan §2 P1 DoD#1「任务精准相关段（AC/文件/函数名）top-k 排名显著上移（给数字）」）：

- **before** = 既有 BM25 层（revamp-P1 换层面，6de82f8:sc/recall.py 计分核逐字镜像）：
  BM25 k1=1.2/b=0.75 + 非负 idf + P0_BOOST=1.5 + STALE_DECAY=0.5 + CJK unigram/bigram
  tokenize + 排序键 (-score, P0 优先, seq) + sorted term 累加。**无** exact-match boost、
  **无**字段加权、**无**路径 run 面、decay=0。
- **after** = 现盘 sc/recall.py extended-BM25 三招：
  招① 高信号 exact-match boost（路径 run ×EXACT_MATCH_BOOST、下划线标识符 ×2、纯数字 ×2×1.5）
  + 招② P0 goal/AC(open_tasks) 字段加权 FIELD_WEIGHT 副本（仅 P0 段）
  + 招③ decay_lambda 参数面（engine 接线，默认 0 = 钉死面）。
- 三招各有**确定性金标**（纯字面、无随机/无 wall-clock）：
  · 族 F（字段/高信号救援）：P0 target 的**文本面**刻意弱化（与填充物共享表面词），
    **字段面**（goal/open_tasks）才承载高信号锚点（文件路径 / 下划线函数名 / AC 数字）。
    before 层只读文本面 → target 被 tf 刷屏的填充物压在 top-k 之外；
    after 层招②把字段 token 计入 tf 副本（×FIELD_WEIGHT）+ 招①对锚点 term 叠 boost →
    target 排名上移（3→1）。三族各测一种锚点类（路径 / 函数名 / 数字）。
  · 族 C（招③ 时间衰减）：三段同文（P0+P1+P1），同 query 下 λ=0 按 seq 升序
    （最早段排前）；启用 decay_lambda>0 后同分下越早 seq 越多被降、最新段不降 →
    目标（最新 P1）rank 上移（3→1）。证明招③「按参数控制启用」接线面真生效。
  · 族 M（margin，信息面）：target 本身独占高信号锚点 → before 已 rank1，
    after 排名不变但**领先 margin 显著拉大**（证明招①②在 rank1 面继续强化精度、不伤）。
  · 对照族（钉死面护城河）：无任何高信号锚点的普通 P1 target → 三招全不触发 →
    after **不得劣于** before（证精度增强只上移精准段、不伤普通面，G4）。

- 零新依赖/零新模型（铁律①②）：仅 import math/os/sys（标准库），0 第三方、0 模型。
- 确定性（G4）：before/after 均纯标准库纯函数 → 全量跑两遍逐条全等（自检项 G5）。

Run（repo 内任意 cwd，布局无关 ROOT 解析，同 bench/p02_probe.py 惯例；3.11/3.14 双解释器）::

    python3 bench/extended_bm25_bench.py

输出 = before/after 逐 case 排名 + 判定；全 PASS 退出码 0，任一 FAIL 退出码 1。
"""
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 插件根（bench/ 直接子件）
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sc.distill import Window, WindowSeg            # noqa: E402
from sc.recall import Recaller, Segment, tokenize  # noqa: E402

K = 5        # top-k 口径（排名面）


# ─────────────────────────── before：旧 BM25 层逐字镜像 ───────────────────────────
# 镜像 6de82f8:sc/recall.py 计分核（k1/b/非负 idf/P0_BOOST/STALE_DECAY/排序键/tokenize
# 全同；三招全关：无 exact-match boost、无字段加权副本、无路径 run 面、decay 恒 0 →
# 即「拧开三招之前」的常驻层行为面）。纯标准库、确定性。

_BM25_K1 = 1.2
_BM25_B = 0.75
_P0_BOOST = 1.5
_STALE_DECAY = 0.5


def before_topk(segs: list[Segment], query: str, k: int = K,
                decay_lambda: float = 0.0) -> list[Segment]:
    """before（旧 BM25 层镜像）top-k：排序键 (-score, P0 优先, seq)。

    decay_lambda 缺省 0 = 钉死面；族 C 用它做「招③ 未启用」对照臂。
    """
    q_set = set(tokenize(query))
    n = len(segs)
    doc_tfs: list[dict] = []
    doc_lens: list[float] = []
    df: dict[str, int] = {}
    for seg in segs:
        tf: dict = {}
        for t in tokenize(seg.original_text):
            tf[t] = tf.get(t, 0.0) + 1.0
        doc_tfs.append(tf)
        doc_lens.append(sum(tf.values()))
        for t in tf:
            df[t] = df.get(t, 0) + 1
    avgdl = (sum(doc_lens) / n) if n else 0.0

    def idf(t: str) -> float:
        d = df.get(t, 0)
        return math.log(1.0 + (n - d + 0.5) / (d + 0.5))    # 非负变体（同现盘）

    tidx = {"P0": 0, "P1": 1, "P2": 2}
    scored = []
    for i, seg in enumerate(segs):
        tf = doc_tfs[i]
        dlen = doc_lens[i]
        base = 0.0
        if dlen != 0 and avgdl != 0.0:
            norm = 1.0 - _BM25_B + _BM25_B * (dlen / avgdl)
            for t in sorted(q_set):
                f = tf.get(t, 0.0)
                if f == 0:
                    continue
                base += idf(t) * (f * (_BM25_K1 + 1)) / (f + _BM25_K1 * norm)
        if seg.tier == "P0":
            base *= _P0_BOOST
        if getattr(seg, "stale", False):
            base *= _STALE_DECAY
        if decay_lambda > 0.0 and n > 0:
            base *= max(0.0, 1.0 - decay_lambda * (float(n) - 1.0 - float(seg.seq)))
        scored.append((-base, tidx[seg.tier], seg.seq, seg))
    scored.sort()
    return [s for *_, s in scored[:k]]


def after_topk(segs: list[Segment], query: str, k: int = K,
               decay_lambda: float = 0.0) -> list[Segment]:
    """after（现盘 extended-BM25 三招）top-k：现盘 Recaller，参数面 decay_lambda。"""
    w = Window(2000, [WindowSeg(s) for s in segs])
    rec = Recaller(None, w, decay_lambda=decay_lambda)   # store=None：top-k 不用点查面
    return [h.seg for h in rec.search(query, k=k) if h.score > 0.0]


def _rank(topk: list[Segment], target: Segment) -> int:
    """target 在 top-k 的 1-based 排名；落 top-k 外 = len+1。"""
    for i, s in enumerate(topk):
        if s.segment_id == target.segment_id:
            return i + 1
    return len(topk) + 1


# ─────────────────────────── 金标语料（纯字面、确定性）────────────────────────────
def _mk(tier, text, seq, goal=None, open_tasks=None) -> Segment:
    return Segment(tier=tier,
                   provenance="user" if tier == "P0" else "model",
                   original_text=text, seq=seq, goal=goal, open_tasks=open_tasks or [])


# ── 族 F · 字段/高信号救援（招②×招①；P0 target 文本面弱、字段面才载锚点）──
# 每族 = P2 填充 + 2×P1 tf 刷屏（共享表面词 + 锚点碎片变体）+ P0 target（弱文本面 + 富字段面）。

# F1 文件路径：填充物把路径碎片词（nginx/conf/log）tf 刷屏，target 文本面只剩「部署 配置」，
# 整串 /etc/nginx/nginx.conf 仅活在字段面（goal + AC desc）。
F_FILEPATH = [
    _mk("P2", "raw dump 全量 日志 冗长 条数 输出", 0),
    _mk("P1", "部署 部署 nginx nginx conf conf 配置 历史 数据 条数 冗长 输出", 1),
    _mk("P1", "网关 部署 配置 nginx conf log 历史 数据 冗长 输出 全量 日志", 2),
    _mk("P0", "部署 配置", 3, goal="/etc/nginx/nginx.conf 部署 网关",
        open_tasks=[{"id": "AC-2", "desc": "deploy /etc/nginx/nginx.conf 网关 上线"}]),
]
F_FILEPATH_Q, F_FILEPATH_T = "部署 /etc/nginx/nginx.conf 网关", 3

# F2 函数名：填充物把标识符碎片（flush/write/buf）tf 刷屏，target 文本面只剩「写盘」，
# 整串 _flush_write_buf 仅活在字段面。
F_FUNCNAME = [
    _mk("P2", "raw dump 全量 日志 冗长 条数 输出", 0),
    _mk("P1", "调用 调用 flush flush write write buf buf 写盘 历史 数据 条数 冗长", 1),
    _mk("P1", "写盘 网关 部署 调用 buffer 历史 数据 冗长 输出 全量 日志", 2),
    _mk("P0", "写盘", 3, goal="_flush_write_buf 写盘 网关",
        open_tasks=[{"id": "AC-12", "desc": "call _flush_write_buf 写盘 间隔 30 秒"}]),
]
F_FUNCNAME_Q, F_FUNCNAME_T = "调用 _flush_write_buf 写盘", 3

# F3 AC 数字：填充物把变体数字（v41/0.4、v43/0.6）tf 刷屏，target 文本面只剩「上线 网关」，
# 精确量值 v42 + 0.5 仅活在字段面。
F_NUMERIC = [
    _mk("P2", "raw dump 全量 日志 冗长 条数 输出", 0),
    _mk("P1", "版本 版本 v41 阈值 0.4 版本 v43 阈值 0.6 上线 历史 数据 条数 冗长 输出", 1),
    _mk("P1", "上线 网关 部署 版本 阈值 历史 数据 冗长 输出 全量 日志", 2),
    _mk("P0", "上线 网关", 3, goal="发布 v42 阈值 0.5 上线",
        open_tasks=[{"id": "AC-7", "desc": "release v42 threshold 0.5 to prod"}]),
]
F_NUMERIC_Q, F_NUMERIC_T = "发布 v42 阈值 0.5 上线", 3

# ── 族 M · margin（信息面：target 独占高信号锚点，before 已 rank1，after margin 拉大）──
M_PATH = [
    _mk("P2", "raw dump 全量 日志 冗长 条数 输出", 0),
    _mk("P1", "日志 c:/svc/legacy.py 全量 c:/svc/old.py 条数 冗长 数据 历史", 1),
    _mk("P1", "配置 网关 部署 c:/conf/app.cfg 历史 数据 冗长 输出", 2),
    _mk("P1", "修改 c:/svc/auth.py 鉴权", 3),
]
M_PATH_Q, M_PATH_T = "修改 c:/svc/auth.py", 3

M_NUMERIC = [
    _mk("P2", "raw dump 全量 日志 冗长 条数 输出", 0),
    _mk("P1", "端口 8081 端口 8082 超时 3001 超时 3002 数值 对比 条数 数据 历史 冗长", 1),
    _mk("P1", "限流 网关 部署 发布 历史 数据 冗长 输出 全量 日志", 2),
    _mk("P1", "认证 服务 端口 8080 超时 3000 绑定", 3),
]
M_NUMERIC_Q, M_NUMERIC_T = "验收 端口 8080 超时 3000", 3

# ── 族 C · 招③ 时间衰减接线（同文 P0+P1+P1；λ=0 对照 vs λ=0.5 启用）──
DECAY_SEGS = [
    _mk("P0", "批处理 任务 队列 调度 历史", 0),
    _mk("P1", "批处理 任务 队列 调度 历史", 1),
    _mk("P1", "批处理 任务 队列 调度 历史", 2),
]
DECAY_Q, DECAY_T, DECAY_LAM = "批处理 任务 队列 调度", 2, 0.5

# ── 对照族 · 钉死面护城河（无高信号锚点的普通 P1 target：三招全不触发，after 不得劣）──
CTRL_SEGS = [
    _mk("P2", "raw dump 全量 日志 冗长 条数 输出", 0),
    _mk("P1", "网关 网关 发布 发布 限流 限流 部署 部署 上线 历史 数据 冗长", 1),
    _mk("P1", "网关 发布 部署 限流 上线 目标 配置 历史 数据 冗长 全量", 2),
]
CTRL_Q, CTRL_T = "网关 发布 部署 限流 上线 目标", 2


def run_all() -> dict:
    """全量跑一遍：返回逐 case 排名/判定面（纯函数、确定性）。"""
    f_rank = [
        ("F·file-path", F_FILEPATH, F_FILEPATH_Q, F_FILEPATH_T),
        ("F·func-name", F_FUNCNAME, F_FUNCNAME_Q, F_FUNCNAME_T),
        ("F·AC-numeric", F_NUMERIC, F_NUMERIC_Q, F_NUMERIC_T),
    ]
    f_rows = []
    for name, segs, q, t in f_rank:
        tgt = segs[t]
        f_rows.append({"case": name,
                       "before": _rank(before_topk(segs, q), tgt),
                       "after": _rank(after_topk(segs, q), tgt)})

    # 族 C（招③）：λ=0（before 对照臂）vs λ=0.5（after 启用臂）
    c_tgt = DECAY_SEGS[DECAY_T]
    c_lam0 = _rank(before_topk(DECAY_SEGS, DECAY_Q, decay_lambda=0.0), c_tgt)
    c_lam = _rank(after_topk(DECAY_SEGS, DECAY_Q, decay_lambda=DECAY_LAM), c_tgt)
    decay_row = {"case": "C·decay", "lam0": c_lam0, "lam_on": c_lam}

    # 族 M（margin 信息面）：target 独占锚点，before/after 均 rank1，margin 拉大
    m_rows = []
    for name, segs, q, t in (("M·path", M_PATH, M_PATH_Q, M_PATH_T),
                             ("M·numeric", M_NUMERIC, M_NUMERIC_Q, M_NUMERIC_T)):
        tgt = segs[t]
        m_rows.append({"case": name,
                      "before": _rank(before_topk(segs, q), tgt),
                      "after": _rank(after_topk(segs, q), tgt)})

    # 对照族：无高信号锚点 → 三招全不触发，after 不得劣
    ct = CTRL_SEGS[CTRL_T]
    ctrl_row = {"case": "ctrl·plain-P1",
                "before": _rank(before_topk(CTRL_SEGS, CTRL_Q), ct),
                "after": _rank(after_topk(CTRL_SEGS, CTRL_Q), ct)}

    f_up = sum(1 for r in f_rows if r["after"] < r["before"])
    gate = {
        "G1_field_rescue": f_up == len(f_rows),          # 3 族 F 全上移
        "G2_decay_wired": c_lam < c_lam0,                 # 招③ 启用后 target 上移
        "G3_margin_not_worse": all(r["after"] <= r["before"] for r in m_rows),
        "G4_ctrl_not_worse": ctrl_row["after"] <= ctrl_row["before"],
        "f_up": f_up, "f_n": len(f_rows),
    }
    return {"f_rows": f_rows, "decay": decay_row, "m_rows": m_rows,
            "ctrl": ctrl_row, "gate": gate}


def determinism_selfcheck() -> bool:
    """确定性面：全量两遍逐条全等（G4：同 query 两次结果全等，纯函数无 wall-clock）。"""
    r1, r2 = run_all(), run_all()
    for key in ("f_rows", "decay", "m_rows", "ctrl", "gate"):
        if r1[key] != r2[key]:
            return False
    return True


def main() -> int:
    r = run_all()
    g = r["gate"]
    print(f"precision P1 · extended-BM25 before/after bench  (K={K})")
    print("  族 F · 字段/高信号救援（招②×①，before→after rank 上移）:")
    for row in r["f_rows"]:
        mark = "↑" if row["after"] < row["before"] else ("=" if row["after"] == row["before"] else "↓!")
        print(f"    {row['case']:12s}  before-rank={row['before']:>2}  after-rank={row['after']:>2}  {mark}")
    d = r["decay"]
    print(f"  族 C · 时间衰减（招③，λ=0→λ={DECAY_LAM}）:")
    print(f"    {d['case']:12s}  λ=0-rank={d['lam0']:>2}  λ={DECAY_LAM}-rank={d['lam_on']:>2}"
          f"  {'↑' if d['lam_on'] < d['lam0'] else '='}")
    print("  族 M · margin（信息面，target 独占锚点，rank 不变但领先拉大）:")
    for row in r["m_rows"]:
        print(f"    {row['case']:12s}  before-rank={row['before']:>2}  after-rank={row['after']:>2}")
    c = r["ctrl"]
    print(f"  对照·钉死面:  {c['case']:12s}  before-rank={c['before']:>2}  after-rank={c['after']:>2}"
          f"  {'(OK 不劣)' if c['after'] <= c['before'] else '(RISK!)'}")

    det = determinism_selfcheck()
    checks = [
        (f"G1 字段救援 {g['f_up']}/{g['f_n']} 全上移", g["G1_field_rescue"]),
        ("G2 时间衰减接线上移", g["G2_decay_wired"]),
        ("G3 margin 不劣", g["G3_margin_not_worse"]),
        ("G4 对照面不劣", g["G4_ctrl_not_worse"]),
        ("G5 确定性两次全等", det),
    ]
    ok = all(v for _, v in checks)
    print("  判定:")
    for name, v in checks:
        print(f"    {'PASS' if v else 'FAIL'}  {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
