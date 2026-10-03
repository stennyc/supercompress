"""Block ⑧ telemetry（precision P2 · 三遥测指标数据闸门，plan §2 P2 + G5）。

三个可观测指标（喂 P3 定梯级触发点 / 30% re-add 证据 / drift 参数标定）：

1. **填充占比 fill_share**（``>40%`` 才 re-add 30% 零成本剪除档；G5：30% 撤销是默认，
   re-add 需证据）：注入 6K 池（细节池 4K L3/L2 + 索引池 2K L1/L0）里「低信号 P1/P2
   填充物」token 占比。``filler_form_share`` 镜像 ``sc.inject.choose_forms`` 的**同排序
   键 + 同池记账**（_rank_key 逐字语义：相关性 desc → tier(P0先) → 新近 desc；高压
   细节池归零同语义）——零新算法面，只把每段成本按 tier 标签汇总成观测数；确定性
   （G4）：同 window 同 query 两次全等。high-signal = P0（structural 活账本 + 非结构
   P0 goal/AC 权威面）；filler = P1/P2（模型侧推理/日志/填充段）。
2. **语义 gap**（exact-match miss 里「同义换措辞该召回」的比例，量化 BM25 词面不够的
   真实 gap；**ONNX 已出局**——gap 大则升级路径 = 扩大 P4 宿主 LLM 扩展触发面，
   不装模型不走量，G3）：词面近失 = ``miss_score``（query 显著 token 落在 target 段
   内比例，高 = 词面近失 / 该召回候选）；可选宿主 LLM 判官（``semantic_gap``，纯文进
   文出 JSON 裁定；端点不可用 → 回退词面 baseline，**不误报 semantic 数**）。
3. **drift 误报 FPR**（drift 事件 vs 人工确认真切换比例 → 标定 DRIFT_THRESH=0.30 /
   K_CONSEC=2，sc/drift.py:30-32）：``DriftCounter``（事件/确认数 → fpr/precision）+
   ``DriftTracker(sink=...)`` 事件留痕面（_fire 那一下才记，常驻路径零开销）；合成
   任务切换流 + ground truth 的 FPR 出数在 bench/telemetry_report.py（纯本地确定性
   模拟，复用 observe_user 判定核，无新分词/无随机/无 wall-clock）。

铁律（G2/G3）：纯本地观测 + 宿主 LLM（若需判 gap），0 外部 0 新模型 0 硬编码路径；
不碰宿主文件、不碰 DISK_WATER_HIGH_PCT（sc/store.py:27 是磁盘水位，非上下文梯级常量）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set

from .inject import (DEFAULT_DETAIL_BUDGET, DEFAULT_INDEX_BUDGET, _l0_ptr, _l1_line,
                     _l3_text, _rank_key)
from .recall import tokenize
from .schema import Segment, est_tokens


def _pool_view(ws: Any):
    """段集合 → 注入池视图（choose_forms 入口面逐字同形）：WindowSeg 原样透传，
    裸 Segment 包成 resident 视图（``excerpt()`` 优先 digest_only 语义，与
    ``distill.WindowSeg.excerpt`` 一致——有 digest 摘录时注入面就是摘录）。"""
    if hasattr(ws, "excerpt"):
        return ws
    return _BarePoolView(ws)


class _BarePoolView:
    """裸 Segment → 注入池视图适配（bench/report 侧直传段集合；零新算法面）。

    与 ``distill.WindowSeg`` 同形（.segment/.form/.excerpt()/.tokens()），
    digest_only 语义对齐：有 digest_text 时注入面 = digest 摘录（L3 成本同口径），
    无 digest = 原文。"""

    def __init__(self, segment: "Segment") -> None:
        self.segment = segment
        self.form = "digest_only" if getattr(segment, "digest_text", "") else "resident"

    @property
    def segment_id(self) -> str:
        return self.segment.segment_id

    def excerpt(self) -> str:
        seg = self.segment
        if self.form == "digest_only":
            return seg.digest_text or "[no digest]"
        return seg.original_text

    def tokens(self) -> int:
        return est_tokens(self.excerpt())

# ── P2 定稿参数（G5 数据闸门标定面）──
FILLER_RE_ADD_THRESHOLD = 0.40      # 填充占比 >40% 才 re-add 30% 零成本剪除档（plan §1/§3 G5）
GAP_LEXICAL_NEAR_MISS_FLOOR = 0.30  # 词面近失判据（miss_score ≥ 此值 = 该召回候选；口径自标定）
GAP_JUDGE_TASK = "supercompress.telemetry_gap_judge"   # 宿主 LLM 判官 task key（唯一 LLM 面变量，G3）


# ─────────────────────────── 指标① 填充占比（choose_forms 同排序镜像）───────────────────────────

def filler_form_share(window, query: str = "",
                      index_budget: Optional[int] = None,
                      detail_budget: Optional[int] = None,
                      segs: Optional[List[Any]] = None) -> Dict[str, Any]:
    """注入 6K 池的低信号填充物 token 占比（镜像 sc.inject.choose_forms 同排序 + 同池记账）。

    入参 = Window 样对象（.segs/.used_tokens/.capacity_tokens；``segs`` 即注入池段集合，
    可传 engine 侧 skel 过滤后的 kept 段窗口——used/capacity 随段集合自洽）；

    返回 {pool_tokens, high_signal_tokens, filler_low_signal_tokens, fill_share,
          gate_re_add_30pct, detail_pool_tokens}：
    - 排序键与 choose_forms 逐字一致，高压（used>cap）细节池归零同语义 → 选档结果与
      注入面完全一致；本函数只多一步「按 tier 标签把每段 form 成本汇总」= 纯观测面；
    - 成本口径 = 该段实际 form 的文本 token（L3=excerpt / L2=digest / L1=索引行 /
      L0=指针），与注入块真实占用同源（est_tokens 单源）；
    - high-signal = P0（structural + 非结构）；filler = P1/P2（模型侧推理/日志/填充段）；
    - fill_share = filler/pool（池空 → 0.0）；gate = fill_share > FILLER_RE_ADD_THRESHOLD；
    - detail_pool_tokens = 选档后剩余细节池（高压初始即 0；structural L3 豁免不扣）。
    确定性（G4）：纯字面排序 + 纯函数成本 → 同输入两次全等。

    ``segs`` 覆写（缺省 = window.segs）：engine 侧注入池 = skel 过滤后的 P0/P1 段
    （choose_forms 实收集合），与 window.segs（全量 replay）不同——覆写注入池段集合，
    而高压判定仍读 window.used/capacity（全量窗口压力，与 choose_forms 逐字同源）。
    """
    ib = DEFAULT_INDEX_BUDGET if index_budget is None else max(0, int(index_budget))
    db = DEFAULT_DETAIL_BUDGET if detail_budget is None else max(0, int(detail_budget))
    high_pressure = getattr(window, "used_tokens", 0) > getattr(window, "capacity_tokens", 0)
    detail_pool = 0 if high_pressure else db
    if segs is None:
        segs = list(getattr(window, "segs", []) or [])
    segs = [_pool_view(ws) for ws in segs]   # WindowSeg 原样 / 裸 Segment → 注入池视图（同一口径）

    query_terms: Set[str] = ({t for t in (query or "").lower().split() if len(t) >= 2}
                             if query else set())

    # 每段成本（与 choose_forms 同款：L3 excerpt / L2 digest / L1 索引行 / L0 指针）
    costs: Dict[str, Dict[str, Any]] = {}
    for ws in segs:
        seg = ws.segment
        has_digest = bool(getattr(seg, "digest_text", ""))
        costs[seg.segment_id] = {
            "l3": est_tokens(_l3_text(ws)),
            "l2": est_tokens(seg.digest_text) if has_digest else None,
            "l1": est_tokens(_l1_line(ws)),
            "l0": est_tokens(_l0_ptr(ws)),
            "digest": has_digest,
        }
    structural = {ws.segment.segment_id for ws in segs if ws.segment.structural_p0}

    form_of: Dict[str, str] = {}
    index_used = 0
    for ws in sorted(segs, key=lambda w: _rank_key(w, query_terms)):
        sid = ws.segment.segment_id
        if sid in structural:
            form_of[sid] = "L3"
            continue
        c = costs[sid]
        if c["l3"] <= detail_pool:
            form_of[sid] = "L3"
            detail_pool -= c["l3"]
        elif c["digest"] and c["l2"] is not None and index_used + c["l2"] <= ib:
            form_of[sid] = "L2"
            index_used += c["l2"]
        elif c["l1"] <= ib and index_used + c["l1"] <= ib:
            form_of[sid] = "L1"
            index_used += c["l1"]
        else:
            form_of[sid] = "L0"

    form_cost = {"L3": "l3", "L2": "l2", "L1": "l1", "L0": "l0"}
    high = filler = pool = 0
    for ws in segs:
        seg = ws.segment
        c = costs[seg.segment_id]
        form = form_of[seg.segment_id]
        cost = c[form_cost[form]]
        if form == "L2" and not c["digest"]:      # 防御面（L2 只在有 digest 时出现）
            cost = c["l0"]
        pool += cost
        if seg.tier == "P0":
            high += cost
        else:                                      # P1/P2 = 低信号填充物（tier 标签，确定性）
            filler += cost
    fill_share = (filler / pool) if pool > 0 else 0.0
    return {
        "pool_tokens": pool,
        "high_signal_tokens": high,
        "filler_low_signal_tokens": filler,
        "fill_share": fill_share,
        "gate_re_add_30pct": fill_share > FILLER_RE_ADD_THRESHOLD,
        "detail_pool_tokens": detail_pool,
    }


# ─────────────────────────── 指标② 语义 gap（词面 miss_score + 宿主 LLM 判官）──────────────────


def _significant(text: str) -> Set[str]:
    """显著 token（与 sc.drift._significant 同语义单源：CJK 表意单字 或 长度≥2 拉丁/数字）。"""
    out: Set[str] = set()
    for t in tokenize(text or ""):
        is_cjk = any("\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" for ch in t)
        if is_cjk or len(t) >= 2:
            out.add(t)
    return out


def miss_score(query: str, target_text: str) -> float:
    """词面近失分（0..1，确定性）：query 显著 token 落在 target 段内的比例。

    高 = 词面其实对得上、却因近重复/措辞差异排错（「该召回而没召回」的候选）→
    语义 gap 的词面 baseline。空 query / 无显著 token → 0。
    """
    q = _significant(query)
    if not q:
        return 0.0
    return len(q & _significant(target_text)) / len(q)


def gap_judge_prompt(query: str, target_text: str, score: float) -> str:
    """宿主 LLM 判官提示词（有界、纯函数）：同义换措辞该召回？文进文出 JSON。"""
    return ("Query: " + (query or "")[:200]
            + "\nTarget: " + (target_text or "")[:200]
            + "\nLexical near-miss: " + f"{float(score):.2f}"
            + "\nDoes the query refer to the same thing as the target, using different wording?"
            + ' Answer JSON exactly {"verdict": "should_recall"|"no"}')


def parse_judge_json(raw: Optional[str]) -> Optional[str]:
    """判官响应 → 裁定（"should_recall"/"no"）；坏 JSON/未知裁定/空 → None（fail-safe，不 raise）。"""
    s = (raw or "").strip()
    if not s:
        return None
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(s[i:j + 1])
            except Exception:
                return None
        else:
            return None
    if not isinstance(obj, dict):
        return None
    v = obj.get("verdict")
    return v if v in ("should_recall", "no") else None


def semantic_gap(misses: List[Dict[str, Any]], endpoint=None) -> Dict[str, Any]:
    """语义 gap 出数（0..1；无 miss 或 LLM 判官不可用 → None，不误报；诚实面单列词面 baseline）。

    misses = [{query, target_text, miss_score?}]（gold 集上 BM25 top-k miss 的逐条）；
    endpoint = 宿主 LLM 端点（.call(task, text) 文进文出，G3：0 外部 0 新模型）：
    - 可用 → semantic_gap = 判官「should_recall」占**有效裁定**比例（judged_by=host_llm）；
    - 端点 None / 调用异常 → 回退词面 baseline（judged_by=lexical_fallback，
      semantic_gap=None；lex_gap = mean(miss_score)，near_share = 近失占比）。
    坏响应（非 JSON/未知裁定）该条不判（不 raise、不炸宿主 turn）。
    """
    misses = [m for m in (misses or []) if m and m.get("query") and m.get("target_text")]
    out: Dict[str, Any] = {"n_misses": len(misses)}
    if not misses:
        out.update(semantic_gap=None, judged_by="none", lex_gap=None, near_share=None)
        return out

    scores: List[float] = []
    for m in misses:
        s = m.get("miss_score")
        scores.append(float(s) if isinstance(s, (int, float))
                    else miss_score(m["query"], m["target_text"]))
    near = sum(1 for s in scores if s >= GAP_LEXICAL_NEAR_MISS_FLOOR)
    lex = sum(scores) / len(scores)
    out["lex_gap"] = lex
    out["lexical_gap_baseline"] = lex
    out["near_share"] = near / len(misses)

    if endpoint is None:
        out.update(semantic_gap=None, judged_by="lexical_fallback")
        return out

    should = judged = 0
    failed = False
    for m, sc in zip(misses, scores):
        try:
            verdict = parse_judge_json(endpoint.call(
                GAP_JUDGE_TASK, gap_judge_prompt(m["query"], m["target_text"], sc)))
        except Exception:
            failed = True
            break
        if verdict is None:
            continue
        judged += 1
        should += 1 if verdict == "should_recall" else 0
    if failed or judged == 0:
        out.update(semantic_gap=None, judged_by="lexical_fallback")
    else:
        out.update(semantic_gap=should / judged, judged_by="host_llm")
    return out


# ─────────────────────────── 指标③ drift 误报 FPR ───────────────────────────


def drift_thresholds() -> Dict[str, float]:
    """drift 标定参数面（单源 = sc/drift.py:30-32；报告脚本引用，不抄数字）。"""
    from .drift import DRIFT_THRESH, K_CONSEC
    return {"DRIFT_THRESH": DRIFT_THRESH, "K_CONSEC": K_CONSEC}


class DriftCounter:
    """drift 事件 / 人工确认真切换 计数面（纯计数，确定性；确认数由 ground truth 填）。"""

    def __init__(self) -> None:
        self.drift_events: int = 0
        self.true_switches_confirmed: int = 0

    def record(self, events: int, true_switches: int) -> None:
        """一次观测面读数：本段 drift 事件数 + 其中人工确认真切换数（可多次累加）。"""
        self.drift_events += max(0, int(events))
        self.true_switches_confirmed += max(0, int(true_switches))

    def report(self) -> Dict[str, Any]:
        """FPR = 误报事件/全部事件（1 - precision）；标定参数面随报告出（drift.py 30-32）。"""
        ev = self.drift_events
        confirmed = min(self.true_switches_confirmed, ev)
        return {
            "drift_events": ev,
            "true_switches_confirmed": self.true_switches_confirmed,
            "drift_false_positives": ev - confirmed,
            "drift_fpr": ((ev - confirmed) / ev) if ev else 0.0,
            "drift_precision": (confirmed / ev) if ev else 0.0,
            "thresholds": drift_thresholds(),
        }


# ─────────────────────────── 会话内遥测 sink（engine 侧纯观测面）──────────────────────────


class TelemetryCollector:
    """会话内遥测记录（sink）：compress fill_share / drift 事件 / 离线 recall probe。

    纯 append 观测（零控制流、零宿主行为改动，G4）；会话边界 reset()（session 记忆
    scope，同 _drift.reset）。JSONL sink（跨会话 append-only 数据攒 1–2 周喂 P3）
    路径 = HERMES_HOME-aware 数据根（G2：0 硬编码路径），写失败 fail-open（遥测挂了
    不炸宿主 turn）。
    """

    def __init__(self, sink_path: Optional[str] = None) -> None:
        self.records: List[Dict[str, Any]] = []
        self.sink_path: Optional[str] = sink_path

    def record_compress(self, fill_share: float, pool_tokens: int, **extra: Any) -> None:
        """compress 侧观测：注入池填充分类（同排序镜像 choose_forms，见 filler_form_share）。"""
        fs = round(float(fill_share), 4)
        self.records.append({"metric": "fill_share", "fill_share": fs,
                            "pool_tokens": int(pool_tokens),
                            "gate_re_add_30pct": fs > FILLER_RE_ADD_THRESHOLD,
                            **extra})

    def record_drift(self, events: int, confirmed_switches: int = 0,
                     epoch: Optional[int] = None, fired: bool = False, **extra: Any) -> None:
        """drift 观测面：事件数增量（引擎侧 fired=True 记 1；报告侧可带确认数标定 FPR）。"""
        self.records.append({"metric": "drift_event", "drift_events": int(max(0, events)),
                            "confirmed_switches": int(max(0, confirmed_switches)),
                            "epoch": epoch, "fired": bool(fired), **extra})

    def record_recall_probe(self, gap: Dict[str, Any]) -> None:
        """离线 recall probe（bench 侧跑完写回）：语义 gap 出数（semantic/lex/near/判定面）。"""
        self.records.append({"metric": "recall_probe", **gap})

    def flush(self) -> int:
        """把会话内记录 append 进 sink_path（JSONL，跨会话攒数）→ 返回写入条数；清 records。

        sink_path None（测试/独立跑）→ 仅清会话内 records（防无界增长）；
        写失败 → fail-open（宿主契约：遥测永不炸 turn）。
        """
        n = 0
        if self.records and self.sink_path:
            try:
                os.makedirs(os.path.dirname(self.sink_path), exist_ok=True)
                with open(self.sink_path, "a", encoding="utf-8") as f:
                    for rec in self.records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n = len(self.records)
            except Exception:
                n = 0
        self.records = []
        return n

    def reset(self) -> None:
        """会话边界：清会话内记录（sink 文件 = 跨会话 append-only 观测面，不动）。"""
        self.records = []

    def summary(self) -> Dict[str, Any]:
        """会话内汇总（before/after 数字喂 P3 定触发点）：fill 均值/峰值 + G5 门
        + drift FPR 面（confirmed 缺省 0 = 全部待确认 → 只出事件数，fpr 由报告侧标定）
        + 最新 recall probe 的 gap 面。"""
        fill_recs = [r for r in self.records if r.get("metric") == "fill_share"]
        drifts = [r for r in self.records if r.get("metric") == "drift_event"]
        probes = [r for r in self.records if r.get("metric") == "recall_probe"]
        fills = [r["fill_share"] for r in fill_recs]
        ev = sum(r.get("drift_events", 0) for r in drifts)
        conf = sum(r.get("confirmed_switches", 0) for r in drifts)
        s: Dict[str, Any] = {
            "n_compress_events": len(fills),
            "fill_share_mean": (sum(fills) / len(fills)) if fills else None,
            "fill_share_max": max(fills) if fills else None,
            "gate_re_add_30pct": any(r.get("gate_re_add_30pct") for r in fill_recs),
            "drift_events": ev,
            "drift_confirmed": conf,
            "drift_fpr": ((ev - min(conf, ev)) / ev) if ev else 0.0,
            "semantic_gap": None,
            "lex_gap": None,
            "gap_judged_by": None,
            "gap_n_misses": None,
        }
        if probes:
            last = probes[-1]
            s["semantic_gap"] = last.get("semantic_gap")
            s["lex_gap"] = last.get("lex_gap")
            s["gap_judged_by"] = last.get("judged_by")
            s["gap_n_misses"] = last.get("n_misses")
        return s
