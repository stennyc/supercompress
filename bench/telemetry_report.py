#!/usr/bin/env python3
"""precision P2 · 三遥测指标报告脚本（plan §2 P2 + G5 数据闸门——出数面，喂 P3 定触发点）。

本脚本 = 三指标的**出数层**（sc/telemetry.py 是内核，engine 是生产埋点；本脚本跑
bench 侧基线读数 + 读生产 JSONL sink 的跨会话攒数）：

指标① **填充占比 fill_share**（30% 零成本剪除档已于 2026-10-04 撤销——活体 P1=0 恒 0.0，纯审计面 G5）：
  基线 = gold 语料 + P1 填充流（`_filler_pool` 合成 6K 注入池，镜像 engine compress 面
  的 kept 池构成）→ ``filler_form_share``（sc/telemetry.py，choose_forms 同排序镜像）；
  两个压力臂：normal（细节池 4K 生效）vs high-pressure（细节池归零，同语义）。
  生产读数 = sink JSONL 里 metric=fill_share 记录（engine compress 观测行）。
指标② **语义 gap**（exact-match miss 里同义换措辞该召回的比例；ONNX 出局，gap 大的
  升级路径 = 扩大 P4 宿主 LLM 触发面，不装模型不走量）：
  词面 baseline = 金标 top-3 miss + 同义探针 miss（_GAP_PROBES 三族：EN query×CJK
  target 零重叠 / CJK 同义换措辞 / 数字锚点被刷屏）的 ``miss_score`` 均值/近失占比
  （确定性，0 外部）；可选 ``--llm-judge`` 走宿主 LIVE_ENDPOINT_* 端点判官（G3）。
指标③ **drift 误报 FPR**（标定 DRIFT_THRESH=0.30 / K_CONSEC=2，sc/drift.py:30-32）：
  合成任务切换流 + 人工 ground truth（哪几条是真任务切换、哪些是闲聊误报面）
  → ``DriftTracker(sink=...)`` 事件面 + ``DriftCounter`` 出 FPR/precision。
  生产读数 = sink JSONL 里 metric=drift_event 记录（engine drift _fire 留痕面）。

铁律（G2/G3）：纯本地观测 + 可选宿主 LLM 判官，0 外部、0 新模型、0 硬编码路径
（ROOT 从 __file__ 反推，同 extended_bm25_bench.py 惯例）；不碰宿主文件、不碰
DISK_WATER_HIGH_PCT。确定性（G4）：同一 commit 两次跑逐条全等（末项自检）。

Run（repo 内任意 cwd；双解释器 3.11/3.14）::

    python3 bench/telemetry_report.py                 # 词面 baseline（默认，0 外部）
    python3 bench/telemetry_report.py --sink <path>    # 附读生产 JSONL 攒数面
    python3 bench/telemetry_report.py --llm-judge     # 宿主 LIVE_ENDPOINT_* 判 gap（可选）

输出 = 报告（stdout + bench/telemetry_report_p2.md）；退出码 = 确定性自检 + 出数完整
（三指标全有数），全过 0，任一 1。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 插件根（bench/ 直接子件）
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sc.distill import Window, WindowSeg                 # noqa: E402
from sc.recall import Recaller                           # noqa: E402
from sc.schema import Segment                             # noqa: E402
from sc.telemetry import (DriftCounter, filler_form_share, miss_score, semantic_gap)  # noqa: E402
from sc.drift import DriftTracker, DRIFT_THRESH, K_CONSEC  # noqa: E402
from bench.gold_recall_set import build_segments, gold_pairs  # noqa: E402

# 与 gold_recall_bench 同窗口径（Recaller 的 search 不依赖 window 容量，容量只供
# fill_share 的高压判定读 used/capacity）：
_WINDOW_CAP = 2000

# 6K 池合成（镜像 engine compress 面 kept = P0/P1 实收集合）：gold 语料（含 P2 段）
# 加 N 条 P1 模型侧填充流（无高信号锚点，纯词面冗长）→ 低信号占比可测的基线池。
_FILLER_N = 20
_FILLER_TMPL = "model filler stream {i} verbose log lines here"

# 指标③ 合成任务切换流（人工 ground truth：true_switches = 流里**真任务切换**条数）。
# 口径：事件面 = DriftTracker(sink) 实际 fire 的 drift 事件；confirmed = ground truth
# 标注的真切换数。FPR = 1 - precision（误报事件 / 全部事件）——喂 DRIFT_THRESH/K_CONSEC 标定。
_DRIFT_STREAMS = [
    {
        "name": "A·真切换一条",   # 部署网关 → k8s 扩容（1 次真切换）+ 各主题内填充
        "msgs": [
            "部署 服务 网关 节点 alpha", "部署 服务 网关 节点 beta",
            "部署 服务 网关 发布 计划", "网关 部署 上线 流程 检查",
            "部署 网关 回滚 预案 演练",                                  # 建锚 N_ANCHOR=5
            "网关 部署 日志 输出 冗长", "部署 服务 网关 监控 指标",      # 同主题填充（不判 drift）
            "k8s pod gpu 调度 0", "k8s pod gpu 调度 1",                 # 真切换 → drift
            "k8s 节点 扩容 策略", "k8s 调度 日志 冗长 输出",            # 新任务内填充
            "k8s 节点 巡检 记录",
        ],
        "true_switches": 1,
    },
    {
        "name": "B·闲聊误报面",    # 同任务内两条闲聊（无显著任务切换，ground truth = 0）
        "msgs": [
            "部署 服务 网关 节点 alpha", "部署 服务 网关 节点 beta",
            "部署 服务 网关 发布 计划", "网关 部署 上线 流程 检查",
            "部署 网关 回滚 预案 演练",
            "今天 天气 不错 散步", "咖啡 买 了一杯",                   # 闲聊（误报面）
            "网关 部署 继续 节点 gamma", "部署 服务 网关 监控 指标",
            "k8s 调度 日志 冗长 输出",
        ],
        "true_switches": 0,
    },
]


def _mk(tier, text, seq, ts=0.0, structural=False, goal=None, open_tasks=None) -> Segment:
    return Segment(tier=tier,
                   provenance="user" if tier == "P0" else "model",
                   original_text=text, seq=seq, ts=float(ts),
                   structural_p0=structural, goal=goal, open_tasks=open_tasks or [])


# ─────────────────────────── 指标① 填充占比基线（6K 池合成）───────────────────────────

def filler_pool(capacity: int = _WINDOW_CAP, filler_n: int = _FILLER_N,
                filler_ts_base: float = 100.0) -> tuple[list[Segment], Window, list[Segment]]:
    """合成注入池 = gold 语料 + P1 填充流 → (裸段集合, Window, gold 段)。

    裸段集合供 filler_form_share(segs=...) 与 Recaller 索引（bench 侧直传口径）；
    Window 供高压臂（capacity 调小 → used>cap → 细节池归零，同 choose_forms 语义）。
    """
    gold = build_segments()
    fill = [_mk("P1", _FILLER_TMPL.format(i=i), seq=100 + i, ts=filler_ts_base + i)
            for i in range(filler_n)]
    pool = gold + fill
    w = Window(capacity, [WindowSeg(s) for s in pool])
    return pool, w, gold


def metric1_fill_share() -> dict:
    """指标① 出数（normal + high-pressure 双臂，before/after P3 触发面）：
    normal = 细节池 4K 生效（6K 池常规态）；high-pressure = 窗口高压（细节池归零，
    注入只剩 L2/L1/L0，填充占比口径同镜像）。"""
    pool, w_normal, _ = filler_pool(capacity=_WINDOW_CAP)
    _, w_press, _ = filler_pool(capacity=500)          # 500 ≪ used → backpressure 态
    normal = filler_form_share(w_normal, segs=pool)
    press = filler_form_share(w_press, segs=pool)
    return {
        "normal": normal,
        "high_pressure": press,
        "pool_segs": len(pool),
        "filler_segs": _FILLER_N,
        "note": ("gate_re_add_30pct=True 即 G5 审计证据面（>40% 才 re-add 30% 零成本剪除档；"
                 "该档已于 2026-10-04 撤销——活体 P1=0 → fill_share 恒 0.0，纯记录无下游消费者）；"
                 "normal vs high-pressure 两臂 = P3 梯级定案的压力对照"),
    }


# ─────────────────────────── 指标② 语义 gap 基线（gold miss + 词面/可选 LLM 判官）─────

def _gold_misses(pool: list[Segment], w: Window, k: int = 3) -> list[dict]:
    """gold 集 top-k miss 逐条（query 显著 token 落在 target 段内的 miss_score 词面面）：
    P1 extended-BM25 上线后 miss 集变稀 = P1 精度收益的直接读数；miss 里的词面近失占比
    = BM25 词面不够的真实 gap（ONNX 出局后的量化口径）。"""
    out = []
    for q, t in gold_pairs(pool):
        rec = Recaller(None, w)
        top = [h.seg for h in rec.search(q, k=k) if h.score > 0.0]
        if not any(s.segment_id == t.segment_id for s in top):
            out.append({"query": q, "target_text": t.original_text,
                       "miss_score": miss_score(q, t.original_text)})
    return out


# 指标② 语义 gap 探针面（同义换措辞 top-3 miss，by-construction 量化「BM25 词面不够」）：
# 三族各钉一种真实 gap 形态——EN query×CJK target（零词面重叠）/ CJK 同义换措辞 /
# 数字锚点被填充物刷屏（锚点词被 tf 灌水稀释）。每族独立小语料（同 bench/gold_recall_set
# 确定性金标口径，纯字面无随机）；探针若因 P1 精度升级开始命中 → 该探针自动退出 miss 面
# （诚实面：gap 集 = 实际 top-k miss，不捏造）。
_GAP_PROBES = [
    {
        "name": "G1·EN-query/CJK-target",   # 零词面重叠（miss_score≈0）= 纯语义 gap
        "corpus": [
            ("P2", "auth pipeline unstable test suite connection timeout verbose raw dump lines"),
            ("P2", "auth pipeline unstable test suite connection timeout dump stream 0 verbose log filler filler"),
            ("P2", "auth pipeline unstable test suite connection timeout dump stream 1 verbose log filler filler"),
            ("P2", "auth pipeline unstable test suite connection timeout dump stream 2 verbose log filler filler"),
        ],
        "query": "auth pipeline unstable test suite connection timeout",
        "target": "登录流程 不稳定 测试 不断 重试 连接 超时 了 网关",
    },
    {
        "name": "G2·CJK-syn/flood",         # 同义换措辞（词面部分重叠，miss_score 中档）
        "corpus": [
            ("P2", "数据库 连接池 连接数 打满 日志 刷屏 条数 0 冗长 输出 历史 数据"),
            ("P2", "数据库 连接池 连接数 打满 日志 刷屏 条数 1 冗长 输出 历史 数据"),
            ("P2", "数据库 连接池 连接数 打满 日志 刷屏 条数 2 冗长 输出 历史 数据"),
            ("P2", "数据库 连接池 连接数 打满 日志 刷屏 条数 3 冗长 输出 历史 数据"),
            ("P2", "数据库 连接池 连接数 打满 日志 刷屏 条数 4 冗长 输出 历史 数据"),
        ],
        "query": "数据库 连接池 连接数 打满 需要 调大 超时",
        "target": "pg 池 上限 不足 扩容 连接 timeout 参数 配置 网关",
    },
    {
        "name": "G3·num-anchor/flood",      # 数字锚点 8443 被多段 tf 刷屏（锚点不再独占）
        "corpus": [
            ("P2", "deploy gateway listen 8443 ssl cert stream verbose dump log filler"),
            ("P1", "gateway listen 8443 ssl cert rotate dump 冗长 输出 条数 0"),
            ("P1", "gateway listen 8443 ssl cert rotate dump 冗长 输出 条数 1"),
            ("P2", "gateway listen 8443 ssl cert renew stream verbose dump log filler 2"),
            ("P2", "gateway listen 8443 ssl cert renew stream verbose dump log filler 3"),
        ],
        "query": "deploy gateway listen 8443 ssl cert",
        "target": "网关 监听 8443 ssl 证书 部署 config 上线",
    },
]


def _gap_probe_misses() -> tuple[list[dict], list[dict]]:
    """探针面 → (miss 逐条, 逐探针判断面)：探针当前是 top-3 miss 才进 gap 集（诚实口径）。"""
    misses: list[dict] = []
    rows: list[dict] = []
    for p in _GAP_PROBES:
        # 探针 corpus：P1/P2 填充段（奇偶交替，与族设计文案面一致；纯字面确定性）
        corpus = [Segment(tier=("P1" if i % 2 else "P2"), provenance="model",
                          original_text=tx, seq=i, ts=float(i))
                  for i, (_t, tx) in enumerate(p["corpus"])]
        target_text = p["target"]
        pool = corpus + [Segment(tier="P0", provenance="model", original_text=target_text,
                                 seq=99, ts=99.0)]
        w = Window(_WINDOW_CAP, [WindowSeg(s) for s in pool])
        rec = Recaller(None, w)
        top = [h.seg for h in rec.search(p["query"], k=3) if h.score > 0.0]
        hit = any(s.original_text == target_text for s in top)
        ms = miss_score(p["query"], target_text)
        rows.append({"probe": p["name"], "top3_miss": not hit, "miss_score": ms})
        if not hit:
            misses.append({"query": p["query"], "target_text": target_text, "miss_score": ms})
    return misses, rows


def metric2_semantic_gap(endpoint=None) -> dict:
    """指标② 出数 = 金标 top-k miss + 同义探针 miss（两源合流，逐条带词面 miss_score）
    → semantic_gap（可选宿主 LLM 判官；端点不可用 → 词面 baseline，诚实 None）。"""
    pool, w, _ = filler_pool()
    misses = _gold_misses(pool, w)
    probe_misses, probe_rows = _gap_probe_misses()
    out = semantic_gap(misses + probe_misses, endpoint=endpoint)
    out["probe_rows"] = probe_rows
    out["gold_misses"] = len(misses)
    out["probe_misses"] = len(probe_misses)
    out["note"] = ("gap 集 = 金标 top-3 miss + 同义探针 top-3 miss（by-construction 词面不够面）；"
                   "语义 gap 大 → 升级路径 = 扩大 P4 宿主 LLM 扩展触发面（ONNX 出局，G3）")
    return out


# ─────────────────────────── 指标③ drift FPR 基线（合成流 + ground truth）─────────────

def metric3_drift_fpr() -> dict:
    counter = DriftCounter()
    rows = []
    for st in _DRIFT_STREAMS:
        sink: list[dict] = []
        d = DriftTracker(sink=sink)
        for text in st["msgs"]:
            d.observe_user(text)
        events = len(sink)
        counter.record(events=events, true_switches=st["true_switches"])
        rows.append({"stream": st["name"], "msgs": len(st["msgs"]),
                     "events": events, "true_switches": st["true_switches"],
                     "final_epoch": d.current_epoch,
                     "fired_at": [e["user_seq"] for e in sink]})
    report = counter.report()
    report["streams"] = rows
    report["note"] = ("events vs 人工确认 = FPR/precision 标定面（DRIFT_THRESH/K_CONSEC "
                      "见 thresholds）；stream A 真切换应 1 事件、stream B 闲聊流 = 误报面")
    return report


# ─────────────────────────── 生产 JSONL sink 读数（跨会话攒数面）──────────────────────

def read_sink(path: str) -> dict:
    """生产 telemetry.jsonl（engine 埋点跨会话 append-only）→ 三指标攒数汇总。

    缺省（未给 --sink / 文件不存在）→ 空面（本卡交付「埋点 + 首读基线」，1–2 周攒数
    由生产 gateway 进程自动落盘，P3 定案时再读本面）。读失败 fail-open（0 外部铁律，
    报告不因 sink 缺而炸）。"""
    summary = {"sink_path": path, "n_lines": 0}
    if not path or not os.path.exists(path):
        summary["note"] = "sink 不存在/未给 → 仅 bench 基线出数（首读基线交付口径）"
        return summary
    recs = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        summary["note"] = f"sink 读取失败（fail-open）: {e}"
        return summary
    summary["n_lines"] = len(recs)
    if not recs:
        return summary
    from sc.telemetry import TelemetryCollector
    col = TelemetryCollector()
    col.records.extend(recs)
    summary.update(col.summary())
    summary.setdefault("note", "生产 sink 攒数面（跨会话 append-only）")
    return summary


# ─────────────────────────── 报告装配 + 出数自检 ───────────────────────────

def build_report(arg_sink: str = "", use_llm: bool = False) -> dict:
    endpoint = None
    if use_llm:
        from sc.live_endpoint import resolve_live_endpoint
        try:
            endpoint = resolve_live_endpoint()   # 宿主 LIVE_ENDPOINT_* 现成解析（G3）
        except Exception:
            endpoint = None                     # 端点缺 → 词面 baseline（不误报 semantic 数）
    report = {
        "commit": _commit(),
        "drift_params": {"DRIFT_THRESH": DRIFT_THRESH, "K_CONSEC": K_CONSEC},
        "metric1_fill_share": metric1_fill_share(),
        "metric2_semantic_gap": metric2_semantic_gap(endpoint),
        "metric3_drift_fpr": metric3_drift_fpr(),
        "production_sink": read_sink(arg_sink),
    }
    # 出数完整自检：三指标全有数（② LLM 不可用时 lex baseline 面在，语义面诚实 None）
    m2 = report["metric2_semantic_gap"]
    report["complete"] = (
        "fill_share" in report["metric1_fill_share"]["normal"]
        and ("semantic_gap" in m2 and "lex_gap" in m2)
        and "drift_events" in report["metric3_drift_fpr"]
    )
    return report


def _commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()[:12]
    except Exception:
        return "(no-git)"


def determinism_selfcheck() -> bool:
    """G4 确定性：报告全量两遍逐条全等（纯本地观测 + 确定性合成流，无 wall-clock/随机）。"""
    r1, r2 = build_report(), build_report()
    r1.pop("commit", None)
    r2.pop("commit", None)
    return r1 == r2


def _sink_disp(sink: dict) -> str:
    """sink 显示面（3.11 兼容：P2 报告全树 3.11 可 compile——f-string 表达式内不用同字符嵌套引号）。"""
    path = sink.get("sink_path") or ""
    if path:
        return path
    return "(未给 --sink；HERMES_HOME 数据根 supercompress_store/telemetry.jsonl 待生产落盘)"


def render_md(r: dict, det: bool) -> str:
    m1 = r["metric1_fill_share"]
    m2 = r["metric2_semantic_gap"]
    m3 = r["metric3_drift_fpr"]
    sink = r["production_sink"]
    n = m1["normal"]
    p = m1["high_pressure"]
    lines = [
        "# precision P2 · 三遥测指标首读基线（数据闸门出数面）",
        "",
        f"commit={r['commit']}  drift 标定参数 DRIFT_THRESH={r['drift_params']['DRIFT_THRESH']} "
        f"/ K_CONSEC={r['drift_params']['K_CONSEC']}（sc/drift.py:30-32）",
        "",
        "## 指标① 填充占比 fill_share（30% 零成本剪除档已于 2026-10-04 撤销——纯审计面 G5）",
        f"- 合成 6K 池：{m1['pool_segs']} 段（gold {m1['pool_segs'] - m1['filler_segs']} + "
        f"P1 填充 {m1['filler_segs']}），双压力臂：",
        f"  - normal：pool={n['pool_tokens']}t  P0 高信号={n['high_signal_tokens']}t  "
        f"P1/P2 填充={n['filler_low_signal_tokens']}t  **fill_share={n['fill_share']:.3f}**  "
        f"gate_re_add_30pct={n['gate_re_add_30pct']}",
        f"  - high-pressure（细节池归零）：**fill_share={p['fill_share']:.3f}**  "
        f"gate_re_add_30pct={p['gate_re_add_30pct']}  detail_pool={p['detail_pool_tokens']}t",
        "## 指标② 语义 gap（exact-match miss 里同义换措辞该召回的比例；ONNX 出局）",
        f"- gap 集：n_misses={m2['n_misses']}（gold top-3 miss {m2.get('gold_misses', 0)} + "
        f"同义探针 miss {m2.get('probe_misses', 0)}；P1 extended-BM25 上线后 miss 集变稀 = 精度收益读数）",
    ]
    for pr in m2.get("probe_rows", []):
        mark = "MISS" if pr["top3_miss"] else "HIT(自动退出 gap 集)"
        lines.append(f"  - {pr['probe']}：{mark}  miss_score={pr['miss_score']:.3f}")
    lines += [
        f"- 词面 baseline：lex_gap={m2.get('lex_gap')!r}  "
        f"近失占比（≥{0.30}）={m2.get('near_share')!r}",
        f"- 判官面：judged_by={m2.get('judged_by')}  semantic_gap={m2.get('semantic_gap')!r}",
        "  （semantic_gap=None + judged_by=lexical_fallback = 端点不可用，只出词面基线，不误报）",
        "## 指标③ drift 误报 FPR（标定 DRIFT_THRESH=0.30 / K_CONSEC=2）",
        f"- 合成流 events={m3['drift_events']}  人工确认真切换={m3['true_switches_confirmed']}  "
        f"**FPR={m3['drift_fpr']:.3f}  precision={m3['drift_precision']:.3f}**",
    ]
    for row in m3.get("streams", []):
        lines.append(f"  - {row['stream']}：{row['msgs']} msgs → {row['events']} events "
                     f"(true={row['true_switches']}, fired_at user_seq={row['fired_at']}, "
                     f"final_epoch={row['final_epoch']})")
    lines += [
        "## 生产 JSONL sink 读数（跨会话攒数面，P3 定案时读取）",
        f"- sink={_sink_disp(sink)}",
        f"- n_lines={sink.get('n_lines')}  {sink.get('note', '')}",
    ]
    if sink.get("n_lines", 0):
        lines.append(f"- 攒数汇总：fill_share_mean={sink.get('fill_share_mean')}  "
                     f"max={sink.get('fill_share_max')}  "
                     f"gate_re_add_30pct={sink.get('gate_re_add_30pct')}  "
                     f"drift_events={sink.get('drift_events')}  fpr={sink.get('drift_fpr')}  "
                     f"lex_gap={sink.get('lex_gap')}  semantic_gap={sink.get('semantic_gap')}")
    lines += [
        "## 自检",
        f"- G4 确定性两遍全等：{'PASS' if det else 'FAIL'}",
        f"- 三指标出数完整：{'PASS' if r['complete'] else 'FAIL'}",
        "",
        f"> 报告生成：bench/telemetry_report.py @ {r['commit']}（纯本地 + 可选宿主 LLM 判官；"
        "0 外部 / 0 新模型 / 0 硬编码路径）",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="precision P2 三遥测指标报告（数据闸门出数面）")
    ap.add_argument("--sink", default="",
                    help="生产 telemetry.jsonl 路径（缺省 = 仅 bench 基线出数）")
    ap.add_argument("--llm-judge", action="store_true",
                    help="宿主 LIVE_ENDPOINT_* 端点判语义 gap（缺省词面 baseline，0 外部）")
    ap.add_argument("--no-write", action="store_true", help="不落盘报告（仅 stdout）")
    args = ap.parse_args()

    report = build_report(arg_sink=args.sink, use_llm=args.llm_judge)
    det = determinism_selfcheck()
    ok = report["complete"] and det
    print(render_md(report, det))

    md_path = os.path.join(ROOT, "bench", "telemetry_report_p2.md")
    if not args.no_write:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(render_md(report, det))
        print(f"报告已写：{md_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
