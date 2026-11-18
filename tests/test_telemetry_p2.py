"""precision P2 · 三遥测指标数据闸门（plan §2 P2 + G5）— 遥测内核 + 引擎接线 + 报告脚本。

三指标（全部纯本地观测 + 可选宿主 LLM 判 gap；0 外部 / 0 新模型 / 0 硬编码路径，G2/G3）：
1. **填充占比 fill_share**：注入 6K 池（细节池 4K + 索引池 2K）里低信号 P1 填充物
   token 占比 —— `>40%` 才 re-add 30% 零成本剪除档（G5：30% 撤销是默认，re-add 需证据）。
   观测面 = 与 choose_forms **同排序镜像**的填充分类（sc.telemetry filler_form_share，
   排序键逐字镜像 sc.inject._rank_key → 常驻路径确定性不破，G4）。
2. **语义 gap**：gold recall miss 里「BM25 词面不够、其实是同义换措辞该召回」的比例。
   词面近似 = miss_score（miss 时 query 显著 token 落在 target 段内的比例，高 = 近失）；
   可选宿主 LLM 判官（gap judge，纯文进文出、JSON 裁定）→ semantic_gap（0..1）；
   端点不可用 → 回退 None（词面 baseline 单列，不误报 semantic 数）。
3. **drift 误报 FPR**：drift 事件数 / 人工确认真切换数（ground truth 由报告脚本的
   合成任务切换流提供）—— 标定 DRIFT_THRESH=0.30 / K_CONSEC=2（sc/drift.py）。

DoD（plan P2）：三指标可观测 + bench/telemetry_report.py 出 before/after 数字喂 P3 定触发点；
本卡交付「埋点 + 首读基线」，不交付 2 周数据。引擎壳只加观测行（零算法体、零控制流改动）。
跑法（repo 根）：  python -m pytest tests/test_telemetry_p2.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

from fakes import FakeEndpoint
from helpers import mkseg, mk_window
from sc.drift import DriftTracker
from sc.distill import Window, WindowSeg
from sc.schema import Segment, est_tokens
from sc.recall import Recaller
from sc.telemetry import (
    DriftCounter,
    FILLER_RE_ADD_THRESHOLD,
    TelemetryCollector,
    filler_form_share,
    gap_judge_prompt,
    miss_score,
    parse_judge_json,
    semantic_gap,
)


# ─────────────────────────── 指标① 填充占比（choose_forms 同排序镜像）───────────────────────────

def _wscap(cap, segs):
    return Window(cap, [WindowSeg(s) for s in segs])


def test_filler_share_low_when_signal_heavy():
    """high-signal 主导（structural P0 + P0 goal/AC 字段）→ 低信号占比小。"""
    p0s = Segment(tier="P0", provenance="user", original_text="goal: ship v1 keep /srv/a1 online",
                  structural_p0=True, seq=0, ts=0.0, goal="ship v1",
                  open_tasks=[{"id": "ac1", "desc": "port 8080 bind"}])
    p0n = Segment(tier="P0", provenance="model", original_text="decision: rollback plan for auth "
                  "service on deploy failure at 02:00 window, notify ops team via runbook",
                  seq=1, ts=0.0)
    p1f = Segment(tier="P1", provenance="model", original_text="nginx listens on 443 ssl "
                  "termination host alpha verbose stdout dump", seq=2, ts=0.0)
    w = _wscap(6000, [p0s, p0n, p1f])
    share = filler_form_share(w)
    assert 0.0 <= share["fill_share"] <= 1.0
    assert share["pool_tokens"] > 0
    # 三段里只 1 段 P1 填充物 → 填充占比应显著低于 P0 总占比
    assert share["filler_low_signal_tokens"] < share["high_signal_tokens"]
    assert share["high_signal_tokens"] + share["filler_low_signal_tokens"] == share["pool_tokens"]


def test_filler_share_domination_by_p1_p2_bulk():
    """注入池被 P1 填充物刷屏（P0 只有 1 段、P1/P2 几十段）→ 占比 > 40% 才判 re-add。"""
    p0 = Segment(tier="P0", provenance="user", original_text="goal: fix auth", structural_p0=True,
                 seq=0, ts=0.0)
    bulk = [Segment(tier="P1", provenance="model", original_text=f"filler note stream {i} "
                          f"verbose log lines here", seq=1 + i, ts=float(i)) for i in range(40)]
    w = _wscap(6000, [p0] + bulk)
    share = filler_form_share(w)
    assert share["fill_share"] > FILLER_RE_ADD_THRESHOLD, "40 段 P1 刷屏应 >40% 触发 re-add 证据"
    assert share["gate_re_add_30pct"] is True


def test_filler_share_zero_when_only_high_signal():
    p0s = Segment(tier="P0", provenance="user", original_text="goal: ship", structural_p0=True,
                  seq=0, ts=0.0)
    w = _wscap(6000, [p0s])
    assert filler_form_share(w)["fill_share"] == 0.0
    assert filler_form_share(Window(6000, []))["pool_tokens"] == 0


def test_filler_share_high_pressure_zero_detail_pool():
    """高压（used>cap）细节池归零 → 只出 L1/L2/L0，share 仍同排序口径可算（不变量：≤1）。"""
    p1 = [Segment(tier="P1", provenance="model", original_text="x" * 200, seq=1, ts=0.0)
          for _ in range(30)]
    w = _wscap(500, p1)                      # 容量 500 ≪ used → backpressure 态
    assert w.used_tokens > w.capacity_tokens
    s = filler_form_share(w)
    assert s["fill_share"] == 1.0           # 全 P1 填充物 → 100%
    assert s["pool_tokens"] > 0
    assert s["detail_pool_tokens"] == 0, "高压细节池必须归零（镜像 choose_forms 高压语义）"


def test_filler_share_bare_segments_equivalent_to_windowsg():
    """bench/report 侧直传裸 Segment（segs=pool）与 engine 侧 WindowSeg 包装逐字同数
    （_pool_view 适配面：两种入口同一观测口径，G4 零行为分叉）。"""
    p0s = Segment(tier="P0", provenance="user", original_text="goal: ship v1", structural_p0=True,
                  seq=0, ts=0.0)
    p1 = [Segment(tier="P1", provenance="model", original_text=f"filler note stream {i} "
                      "verbose log", seq=1 + i, ts=float(i)) for i in range(6)]
    pool = [p0s] + p1
    w_seg = Window(500, [WindowSeg(s) for s in pool])
    assert filler_form_share(w_seg, segs=pool) == filler_form_share(w_seg,
                                                                   segs=[WindowSeg(s) for s in pool])


# ─────────────────────────── 指标② 语义 gap（miss 词面 + 宿主 LLM 判官）───────────────────────────

def test_miss_score_lexical_near_miss_high():
    """target 段含 query 多数显著 token 但 BM25 因近重复簇排错 = 词面近失 → 分数高。"""
    q = "auth service rollback on deploy failure"
    target = "rollback strategy for the auth service deploy window"
    # q 显著 {auth,service,rollback,on,deploy,failure}；target 含 {auth,service,rollback,deploy}
    # → 4/6 ≈ 0.667（词面近失但非全等 → 该召回候选）
    assert 0.5 < miss_score(q, target) < 1.0


def test_miss_score_unrelated_low():
    q = "数据库 连接池 超时 调大"
    target = "user: ship v1 and keep /srv/a1 online"
    assert miss_score(q, target) < 0.35


def test_miss_score_empty_query_zero():
    assert miss_score("", "anything at all") == 0.0


def test_gap_judge_parse_plain_json_and_fenced():
    assert parse_judge_json(json.dumps({"verdict": "should_recall", "rationale": "同义"})) == "should_recall"
    fenced = 'sure:\n```json\n{"verdict":"should_recall"}\n```\ndone'
    assert parse_judge_json(fenced) == "should_recall"
    assert parse_judge_json("") is None
    assert parse_judge_json("not json at all") is None
    assert parse_judge_json(json.dumps({"verdict": "weird"})) is None


def test_semantic_gap_llm_judge_available():
    """宿主 LLM 判官可用 → semantic_gap = 判官应召回比例（0..1，LLM 面）。"""
    class _Ep:
        def call(self, task, text):
            assert task == "supercompress.telemetry_gap_judge"
            return json.dumps({"verdict": "should_recall" if "同义" in text else "no"})
    misses = [
        {"query": "重启 服务 网关", "target_text": "重新拉起 gateway 服务并重启 8081 同义词近义表述",
         "miss_score": 0.9},
        {"query": "数据库 连接池", "target_text": "auth service 192.168.55.55 unrelated"},
    ]
    got = semantic_gap(misses, endpoint=_Ep())
    assert got is not None and got["judged_by"] == "host_llm"
    assert abs(got["semantic_gap"] - 0.5) < 1e-9      # 1/2 判官裁定应召回
    assert got["n_misses"] == 2


def test_semantic_gap_llm_unavailable_fallback_heuristic():
    """端点抛异常 → 回退词面 baseline（miss_score 均值高 = gap 大），不误报 semantic 数。"""
    class _BoomEp:
        def call(self, task, text):
            raise RuntimeError("endpoint unavailable")
    misses = [{"query": "重启 服务 网关", "target_text": "重新拉起 网关 服务 重启", "miss_score": 0.99}]
    got = semantic_gap(misses, endpoint=_BoomEp())
    assert got["judged_by"] == "lexical_fallback"
    assert got["semantic_gap"] is None, "无 LLM 不得产 semantic 数（诚实面）"
    assert got["lexical_gap_baseline"] is not None and got["lexical_gap_baseline"] > 0.3
    assert got["n_misses"] == 1


def test_semantic_gap_no_misses():
    got = semantic_gap([], endpoint=None)
    assert got["n_misses"] == 0 and got["semantic_gap"] is None


def test_gap_judge_prompt_bounded_and_pure():
    p = gap_judge_prompt("重启 服务", "重新拉起 gateway", 0.8)
    assert "重启" in p and "重新拉起" in p and "should_recall" in p
    assert len(p) <= 400, "判官提示词有界（短提示，不刷屏）"


# ─────────────────────────── 指标③ drift 误报 FPR ───────────────────────────


def test_drift_counter_fpr_ground_truth():
    c = DriftCounter()
    c.record(events=5, true_switches=3)      # 5 事件里 3 条人工确认真切换 → FPR = 2/5
    r = c.report()
    assert r["drift_events"] == 5
    assert r["true_switches_confirmed"] == 3
    assert abs(r["drift_fpr"] - 0.4) < 1e-9
    assert r["drift_precision"] == 0.6
    assert r["thresholds"] == {"DRIFT_THRESH": 0.30, "K_CONSEC": 2}, "标定参数面随报告面出"


def test_drift_counter_no_events_zero_fpr():
    c = DriftCounter()
    c.record(events=0, true_switches=0)
    assert c.report()["drift_fpr"] == 0.0


def test_drift_tracker_sink_records_fired_events():
    """DriftTracker 事件留痕面：sink 收到的 = _fire 事件（epoch 前进那一下）。"""
    sink = []
    d = DriftTracker(sink=sink)
    for i in range(5):                         # 建锚期 N_ANCHOR=5
        d.observe_user(f"部署 服务 网关 节点 {i}")
    d.observe_user("k8s pod gpu 调度 0")
    d.observe_user("k8s pod gpu 调度 1")       # 连续 K=2 低命中 → drift
    assert d.current_epoch == 1
    assert len(sink) == 1
    assert sink[0]["epoch"] == 1 and sink[0]["ratio"] < 0.30


# ─────────────────────────── 报告装配（喂 P3 定触发点）──────────────────────────


def test_collector_summary_shape_and_gate():
    col = TelemetryCollector()
    col.record_compress(fill_share=0.55, pool_tokens=5000)
    col.record_drift(events=4, confirmed_switches=2)
    col.record_recall_probe(gap={"semantic_gap": 0.25, "judged_by": "host_llm", "n_misses": 8})
    s = col.summary()
    for key in ("fill_share_mean", "fill_share_max", "gate_re_add_30pct",
                "drift_events", "drift_fpr", "semantic_gap", "lex_gap"):
        assert key in s, key
    assert s["fill_share_mean"] == 0.55 and s["fill_share_max"] == 0.55
    assert s["gate_re_add_30pct"] is True, "55%>40% → G5 re-add 证据面 True"
    assert s["drift_fpr"] == 0.5
    assert s["n_compress_events"] == 1


def test_collector_summary_empty():
    s = TelemetryCollector().summary()
    assert s["fill_share_mean"] is None and s["drift_fpr"] == 0.0
    assert s["n_compress_events"] == 0


# ─────────────────────────── 引擎接线（观测行，零控制流改动）──────────────────────────


def _engine(tmp_path):
    from engine import SupercompressContextEngine
    return SupercompressContextEngine(
        external_store_path=str(tmp_path), context_length=20000)


def test_engine_compress_records_telemetry_event(tmp_path):
    e = _engine(tmp_path)
    p0 = mkseg(e.store, "goal: ship v1", "P0")
    p1 = mkseg(e.store, "filler log line " * 20, "P1")
    e._window = mk_window(e.context_length, [p0, p1])
    msgs = [{"role": "user", "content": "x"}]
    out = e.compress(msgs, current_tokens=int(e.context_length * 0.6))
    assert out is not None
    assert len(e.telemetry.records) == 1
    rec = e.telemetry.records[0]
    assert rec["metric"] == "fill_share" and 0.0 <= rec["fill_share"] <= 1.0
    assert rec["pool_tokens"] > 0


def test_engine_drift_events_reach_telemetry(tmp_path):
    e = _engine(tmp_path)
    for i in range(5):
        e.on_turn_complete([{"role": "user", "content": f"部署 服务 网关 节点 {i}"}])
    for j in range(2):
        e.on_turn_complete([{"role": "user", "content": f"k8s pod gpu 调度 {j}"}])
    assert e._drift.current_epoch >= 1
    drift_recs = [r for r in e.telemetry.records if r["metric"] == "drift_event"]
    assert drift_recs and drift_recs[-1]["drift_events"] == 1


def test_engine_session_reset_clears_telemetry(tmp_path):
    e = _engine(tmp_path)
    p0 = mkseg(e.store, "goal: ship", "P0")
    e._window = mk_window(e.context_length, [p0])
    e.compress([{"role": "user", "content": "x"}], current_tokens=10000)
    assert len(e.telemetry.records) == 1
    e.on_session_reset()
    assert len(e.telemetry.records) == 0, "会话边界清 sink（session 记忆 scope，同 _drift.reset）"


def test_engine_reports_path_hermes_home_aware(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hhome"))
    e = _engine(tmp_path)
    p = e.telemetry_report_path()
    assert Path(p).name == "telemetry.jsonl" and Path(p).parent.name == "supercompress_store"  # OS-neutral（win32 分隔符面）
    assert str(p).startswith(str(tmp_path / "hhome"))
