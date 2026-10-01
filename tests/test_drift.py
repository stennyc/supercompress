"""P2（S-Phase 5）任务漂移发现（topic epoch）测试。

DoD（references/supercompress-session-redesign.md S-Phase 5 / plan P2）：
- P2.1 epoch 打标：user 段 session_topic = str(current_epoch)（整数串；非整数/"" 中性）
- P2.2 检测：头 N=5 建锚，后续连续 K=2 条命中 < 0.30 → drift，epoch 前进；
  检测延迟 ≤ 2 条消息（3 task 切换注入）；同主题 session 0 误报（FPR 面）
- P2.3 双档：soft（旧 epoch 非结构 P1/P2 召回 ×0.3 降权，P0/goal/ledger 不动）
  / hard（旧 epoch 非结构 P1/P2 → form=digest_only，可逆：jsonl 字节不动，回切即恢复）
- 冷重开 rehydrate：取盘上最大整数 epoch + 重建锚点（防 epoch 失配假阳性漂移）
- 整数护城河：session_topic 非整数串（"supercompress-dev" / ""）一律中性，不降权不逐出
  （tests/helpers.py 的 TOPIC + 生产 464 行恒 "" 都不参与 P2 → 133 钉死面安全）

关键实码事实（@reviewer 可复核，测试据此设计）：
- P1/P2 是 lazy 批：`store.replay()` / engine `_window_fresh()` 只见 **durable（已 flush）**
  段。生产 turn 末 `Distiller.run()` 步骤 7 `store.flush()`，故 P1 进窗口前必须先 flush。
  本文件 P1 e2e 测试据此显式 `store.flush()`，复刻生产语义。
- `digest_text` 是运行期字段（不落盘）：`_window_fresh()` replay 出的新 Segment 段 digest_text
  为空 → hard 逐出后该段 L3 = `"[no digest]"` 占位符（无 digest 时），不是摘要。故 hard e2e
  断言以「旧 epoch P1 全文出窗口、新 epoch P1 全文保留、原文在盘可逆」为准，不依赖 digest。
- `DriftTracker.soft_rank` 原地改 `Hit.score`（生产每次 `Recaller.search()` 返回**全新** Hit，
  无共享态）；确定性复跑须用**全新** Hit 列表。

零 LLM / 零网络 / 确定性（同 user-seg 序列两次全等）。
跑法（repo 根）：
  python -m pytest tests/test_drift.py -v
"""
from __future__ import annotations

from pathlib import Path

from engine import SupercompressContextEngine
from sc.drift import DriftTracker, hit_ratio, N_ANCHOR, K_CONSEC
from sc.recall import Hit
from sc.schema import Segment
from sc.distill import WindowSeg
from sc.store import JsonlStore


def _seg(text, tier="P1", prov="model", structural=False, topic="", kind="segment",
         goal=None, seq=0):
    return Segment(tier=tier, provenance=prov, original_text=text,
                   session_topic=topic, structural_p0=structural, kind=kind,
                   goal=goal, seq=seq)


# ─────────────────────────── sc 层：P2.1 建锚 + P2.2 检测 ───────────────────────────

def test_anchor_build_no_drift_in_build_phase():
    """P2.1：头 N=5 条同主题 user 段全进锚点（建锚期不判漂移）。"""
    d = DriftTracker()
    for i in range(N_ANCHOR):
        assert d.observe_user(f"auth service token {i}") is False
    assert d.current_epoch == 0
    assert d.anchor_user_count == N_ANCHOR
    assert d.drift_events == []


def test_same_topic_no_false_positive():
    """DoD FPR 面（20 session 实测，收口 @reviewer Minor）：20 个独立单任务 session，
    每个 5 建锚 + 15 同主题续判 → 每个 session drift_events 必须 0（FPR = 0/20 ≤ 1/20）。
    单任务 session 内部词根高度共享（k8s/集群/节点/部署/副本 轮转），hit_ratio 恒 ≥ 阈值。
    """
    total_drift = 0
    for s in range(20):
        d = DriftTracker()
        for i in range(N_ANCHOR):
            d.observe_user(f"部署 k8s 集群 节点 扩缩容 方案 {i}")
        for i in range(15):
            d.observe_user(f"k8s 集群 节点 部署 副本 扩缩容 方案 {i}")
        total_drift += len(d.drift_events)
        assert d.current_epoch == 0, f"session {s} 单任务不得 drift"
    assert total_drift == 0, f"20 session 实测 FPR 面：drift 事件总数 {total_drift} 必须 0（≤1/20）"


def test_two_task_switch_detection_delay_le_2():
    """DoD#1：切到第 2 个任务，第 2 条新任务消息即触发（检测延迟 = K_CONSEC = 2）。"""
    d = DriftTracker()
    for i in range(N_ANCHOR):
        d.observe_user(f"auth service token {i}")
    d.observe_user("k8s deploy cluster nodes")            # 切换后第 1 条（consec=1，未触发）
    assert d.current_epoch == 0, "单条新任务不触发（K_CONSEC=2）"
    assert d.observe_user("k8s pod scheduling gpu") is True, "第 2 条新任务应触发（延迟 2）"
    assert d.current_epoch == 1
    assert len(d.drift_events) == 1


def test_three_task_switch_all_detected():
    """DoD#1：3 task 依次切换 → 2 次 epoch 前进（t0→t1→t2），最终 current_epoch=2。"""
    d = DriftTracker()
    topics = [
        [f"auth service token {i}" for i in range(N_ANCHOR)],
        [f"k8s deploy node {i}" for i in range(N_ANCHOR + 5)],
        [f"数据库 迁移 索引 {i}" for i in range(N_ANCHOR + 5)],
    ]
    for tlist in topics:
        for t in tlist:
            d.observe_user(t)
    assert d.current_epoch == 2, f"3 task 应前进 2 次 epoch，实际 {d.current_epoch}"
    assert len(d.drift_events) == 2, "3 task → 2 次漂移事件"


def test_determinism_two_identical_sequences():
    """DoD#2：同 user-seg 序列两次全等（无 wall-clock / 随机 / dict 序）。"""
    seq = [f"auth service {i}" for i in range(N_ANCHOR)] + \
          [f"k8s deploy {i}" for i in range(10)] + \
          [f"数据库 迁移 {i}" for i in range(10)]

    def run() -> list:
        d = DriftTracker()
        states = []
        for t in seq:
            d.observe_user(t)
            states.append((d.current_epoch, len(d.drift_events)))
        return states

    assert run() == run(), "同序列两次 drift 状态必须全等（确定性）"


def test_hit_ratio_deterministic():
    """hit_ratio 纯函数：|q∩anchor|/|q|，空查询 0，确定性。"""
    anchor = {"auth", "service", "token"}
    assert hit_ratio("auth service login", anchor) > 0.5
    assert hit_ratio("k8s deploy pod", anchor) == 0.0
    assert hit_ratio("", anchor) == 0.0
    assert hit_ratio("auth", anchor) == 1.0


# ─────────────────────────── sc 层：P2.3 soft / hard + 整数护城河 ───────────────────────────

def test_is_old_epoch_integer_moat():
    """整数护城河：非整数 session_topic（helper TOPIC / 生产 ""）一律中性，不判旧。"""
    d = DriftTracker()
    d.current_epoch = 1
    assert d.is_old_epoch(_seg("x", topic="supercompress-dev")) is False  # 非整数 → 中性
    assert d.is_old_epoch(_seg("x", topic="")) is False                  # "" → 中性
    assert d.is_old_epoch(_seg("x", topic="0")) is True                  # 整数且 ≠ 1 → 旧
    assert d.is_old_epoch(_seg("x", topic="1")) is False                # 整数且 = 1 → 当前


def _fresh_hits():
    """构造 fresh Hit 列表（soft_rank 原地改分 → 每次需全新对象，确定性复跑同义）。"""
    p0 = _seg("goal", tier="P0", prov="user", structural=True, goal="g", seq=0)
    old_p1 = _seg("旧推理", tier="P1", topic="0", seq=1)
    old_p2 = _seg("旧观察", tier="P2", topic="0", seq=2)
    neu_p1 = _seg("中性推理", tier="P1", topic="", seq=3)             # 中性 topic
    ref = {"p0": p0, "old_p1": old_p1, "old_p2": old_p2, "neu_p1": neu_p1}
    return [Hit(old_p1, 10.0), Hit(old_p2, 10.0), Hit(neu_p1, 10.0), Hit(p0, 100.0)], ref


def test_soft_rank_downweights_old_epoch_p1_p2_only():
    """P2.3 soft：旧 epoch 非结构 P1/P2 ×0.3；P0/structural/中性 topic 不动。"""
    hits, ref = _fresh_hits()
    d = DriftTracker()
    d.current_epoch = 1
    out = d.soft_rank(hits)
    by_seg = {h.seg.segment_id: h.score for h in out}
    assert by_seg[ref["p0"].segment_id] == 100.0, "structural P0 不得降权"
    assert by_seg[ref["neu_p1"].segment_id] == 10.0, "中性 topic 段不得降权"
    assert abs(by_seg[ref["old_p1"].segment_id] - 10.0 * 0.30) < 1e-9, "旧 epoch P1 应 ×0.3"
    assert abs(by_seg[ref["old_p2"].segment_id] - 10.0 * 0.30) < 1e-9, "旧 epoch P2 应 ×0.3"
    # Nit 收口（copy-in）：soft_rank 不 mutate 入参 hits —— 入参 Hit.score 保持原值（可安全复用/重入）
    in_scores = {h.seg.segment_id: h.score for h in hits}
    assert abs(in_scores[ref["old_p1"].segment_id] - 10.0) < 1e-9, "入参旧 epoch P1 不得被原地改分"
    assert abs(in_scores[ref["neu_p1"].segment_id] - 10.0) < 1e-9, "入参中性 P1 不得被改分"
    # 确定性：全新 Hit + 全新 tracker 再跑一次 → 排序与分数全等
    d2 = DriftTracker()
    d2.current_epoch = 1
    assert [(h.seg.seq, round(h.score, 9)) for h in d.soft_rank(_fresh_hits()[0])] == \
           [(h.seg.seq, round(h.score, 9)) for h in d2.soft_rank(_fresh_hits()[0])]


def test_soft_rank_no_double_penalty_on_reuse():
    """Nit 收口反证：同一 hits 列表被两个不同 current_epoch 复用，各自独立计分（无累积降权）。

    旧实现原地改分会把复用同一 hits 的第二次调用再 ×0.3（双重降权）；copy-in 后每次从
    原始分出发 → 两次结果只反映各自 tracker 的 epoch，互不污染。
    """
    hits, ref = _fresh_hits()
    d_a = DriftTracker(); d_a.current_epoch = 1
    out_a = d_a.soft_rank(hits)
    by_a = {h.seg.segment_id: h.score for h in out_a}
    assert abs(by_a[ref["old_p1"].segment_id] - 10.0 * 0.30) < 1e-9
    # 同一 hits（未被 mutate）再喂给另一个 tracker（仍 epoch 1）→ 结果全等（非 0.3×0.3）
    d_b = DriftTracker(); d_b.current_epoch = 1
    out_b = d_b.soft_rank(hits)
    by_b = {h.seg.segment_id: h.score for h in out_b}
    assert by_b[ref["old_p1"].segment_id] == by_a[ref["old_p1"].segment_id], "复用不得二次降权"


def test_hard_evict_candidates_and_restore():
    """P2.3 hard：旧 epoch 非结构 P1/P2 是可逆候选；goal/ledger/structural P0/当前 epoch 豁免。"""
    d = DriftTracker()
    d.current_epoch = 1
    goal = _seg("顶层目标", tier="P0", prov="user", structural=True, goal="g", topic="0", seq=0)
    ledger = _seg("活账本", tier="P0", prov="user", structural=True, kind="ledger", topic="0", seq=1)
    old_p1 = _seg("旧推理", tier="P1", topic="0", seq=2)
    old_p2 = _seg("旧观察", tier="P2", topic="0", seq=3)
    neu_p1 = _seg("新推理", tier="P1", topic="1", seq=4)             # 当前 epoch
    all_segs = [goal, ledger, old_p1, old_p2, neu_p1]
    cands = {s.seq for s in d.hard_evict(all_segs)}
    assert cands == {2, 3}, f"只有旧 epoch 非结构 P1/P2（seq 2,3）应入候选，实际 {cands}"
    # 回切语义：restore_epoch("0") → current=0，则 "0" 段=当前（豁免）、"1" 段≠当前 → 变「旧」
    d.restore_epoch("0")
    cands_back = {s.seq for s in d.hard_evict(all_segs)}
    assert cands_back == {4}, f"回切 epoch0 后，'1' 段(seq4)变旧、'0' 段豁免，实际 {cands_back}"
    # 单调前进的常规路径不调 restore；字节从未逐出（jsonl 单一事实源，point_fetch 全量取回）


def test_hard_reversible_bytes_untouched(tmp_path):
    """DoD#2：hard 逐出只翻内存 form，jsonl 单一事实源字节不动（replay 仍全量取回原文）。"""
    store = JsonlStore(str(tmp_path / "d.jsonl"))
    old_p1 = _seg("旧推理内容", tier="P1", topic="0", seq=0)
    old_p2 = _seg("旧观察内容", tier="P2", topic="0", seq=1)
    store.append(old_p1)
    store.append(old_p2)
    store.flush()                                    # P1/P2 → durable（生产 turn 末 flush 语义）
    d = DriftTracker()
    d.current_epoch = 1
    cands = d.hard_evict(store.replay())
    assert len(cands) == 2, f"两条旧 epoch P1/P2 应为候选，实际 {len(cands)}"
    # 模拟 engine：翻 form=digest_only（内存），不动盘
    for ws in [WindowSeg(s) for s in cands]:
        ws.form = "digest_only"
    # 字节面：replay 仍取回原文（可逆）
    back = {s.segment_id: s.original_text for s in store.replay()}
    assert back[old_p1.segment_id] == "旧推理内容"
    assert back[old_p2.segment_id] == "旧观察内容"


# ─────────────────────────── sc 层：冷重开 rehydrate（防假阳性漂移）───────────────────────────

def test_rehydrate_no_false_positive_cold_restart():
    """网关冷重启：盘上已有 "1" 段，rehydrate 取 max epoch → 不把 "1" 段判旧。"""
    segs = [_seg("旧推理", tier="P1", topic="1", seq=0),
            _seg("旧观察", tier="P2", topic="1", seq=1),
            _seg("更早", tier="P1", topic="0", seq=2)]
    d = DriftTracker()
    d.rehydrate(segs)
    assert d.current_epoch == 1, "rehydrate 应取盘上最大整数 epoch（=1）"
    assert d.is_old_epoch(_seg("x", topic="1")) is False, "重启后 '1' 段不得判旧（假阳性护城河）"
    assert d.is_old_epoch(_seg("x", topic="0")) is True, "epoch 0 段（更早）仍判旧"


def test_rehydrate_all_neutral_equals_reset():
    """盘上全中性 topic（pre-P2 "" / 非整数）→ rehydrate 等价 reset（epoch 0、锚点空）。"""
    d = DriftTracker()
    d.rehydrate([_seg("a", topic=""), _seg("b", topic="supercompress-dev")])
    assert d.current_epoch == 0
    assert d.anchor == set()
    assert d.anchor_user_count == 0


def test_rehydrate_rebuilds_anchor_from_current_epoch():
    """rehydrate 用当前 epoch 头 N 段有显著 token 的重建锚点（seq 时序序）；同主题续判不触发。"""
    segs = [_seg("k8s deploy node 0", tier="P1", topic="1", seq=0),
            _seg("k8s pod scheduling", tier="P1", topic="1", seq=1),
            _seg("k8s gpu cluster", tier="P2", topic="1", seq=2)]
    d = DriftTracker()
    d.rehydrate(segs)
    assert d.current_epoch == 1
    assert "k8s" in d.anchor, "锚点应含当前 epoch 段的话题 token"
    assert d.observe_user("k8s deploy more nodes") is False, "锚点已对齐当前任务，不应触发"


def test_anchor_evidence_audit_face():
    """Nit 收口（审计面）：anchor_evidence/anchor_source 诚实反映锚点真实证据基，
    与 anchor_user_count（门控哨兵）分离 —— 不夸大（drift 重建锚点证据 = 最近 K 条，非 N_ANCHOR）。"""
    # build：头 N_ANCHOR 条同主题 → 证据 = N_ANCHOR，来源 build
    d = DriftTracker()
    for i in range(N_ANCHOR):
        d.observe_user(f"auth service token {i}")
    assert d.anchor_source == "build"
    assert d.anchor_evidence == N_ANCHOR, "build 期证据 = 已入锚 user 段数"

    # drift：触发后锚点由最近 K_CONSEC 条重建 → 证据 = K_CONSEC（< N_ANCHOR，哨兵 anchor_user_count=N_ANCHOR 不夸大）
    d.observe_user("k8s deploy node 0")
    d.observe_user("k8s pod scheduling gpu 1")      # 触发 drift（consec=K_CONSEC=2）
    assert d.current_epoch == 1
    assert d.anchor_source == "drift"
    assert d.anchor_user_count == N_ANCHOR, "门控哨兵跳过建锚期"
    assert d.anchor_evidence == K_CONSEC, "drift 锚点真实证据基 = 触发 K 条（≠ N_ANCHOR，诚实）"

    # rehydrate：3 段当前 epoch → 证据 = 3（可 < N_ANCHOR），来源 rehydrate
    d2 = DriftTracker()
    d2.rehydrate([_seg("k8s deploy 0", topic="1", seq=0),
                  _seg("k8s pod 1", topic="1", seq=1),
                  _seg("k8s gpu 2", topic="1", seq=2)])
    assert d2.anchor_source == "rehydrate"
    assert d2.anchor_evidence == 3, "rehydrate 锚点由盘上 3 段重建"

    # reset / fresh：无锚点 → source none, evidence 0
    d3 = DriftTracker()
    assert d3.anchor_source == "none"
    assert d3.anchor_evidence == 0


# ─────────────────────────── engine 级 e2e：epoch 打标 + soft/hard 接线 ───────────────────────────

def _mkengine(tmp_path: Path) -> SupercompressContextEngine:
    """独立隔离 engine（external_store_path → tmp_path；不碰全局 active.jsonl）。"""
    return SupercompressContextEngine(context_length=256000,
                                      external_store_path=str(tmp_path))


def test_engine_on_turn_complete_tags_user_epoch(tmp_path):
    """P2.1/P2.2 e2e：on_turn_complete 给 user 段打整数 epoch 标 + 切换触发漂移。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_e2e_tag")
    for i in range(N_ANCHOR):                          # 建锚期 5 条同主题 user（P0 即写 durable）
        e.on_turn_complete([{"role": "user", "content": f"auth service token {i}"}])
    assert e._drift.current_epoch == 0
    e.on_turn_complete([{"role": "user", "content": "k8s deploy node 0"}])
    assert e._drift.current_epoch == 0, "单条新任务不触发（K_CONSEC=2）"
    e.on_turn_complete([{"role": "user", "content": "k8s pod scheduling gpu 1"}])
    assert e._drift.current_epoch == 1, "2 条新任务应触发 epoch 前进"
    # user 段已打整数 epoch 标（P0 user 即写 durable，replay 可见）
    topics = [s.session_topic for s in e.store.replay() if s.provenance == "user"]
    assert topics and all(t.isdigit() for t in topics), f"user 段须整数 epoch 标，实际 {topics}"
    assert "1" in topics, "触发后新 user 段应打 epoch 1 标"


def test_engine_select_context_soft_downweights_old_epoch(tmp_path):
    """P2.3 soft e2e：select_context 走 soft_rank，当前 epoch P1 召回、旧 epoch P1 降权。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_e2e_soft")
    e._drift.current_epoch = 1
    old_p1 = _seg("OLD_EPOCH 推理 旧任务 上下文", tier="P1", topic="0", seq=0)
    new_p1 = _seg("NEW_EPOCH 当前 任务 部署 k8s 集群", tier="P1", topic="1", seq=1)
    e.store.append(old_p1)
    e.store.append(new_p1)
    e.store.flush()                                     # P1 → durable（进 _window_fresh）
    q = "当前 任务 部署 k8s 集群 NEW_EPOCH"
    out = e.select_context([{"role": "user", "content": q}],
                           conversation_messages=[{"role": "user", "content": q}])
    assert out is not None, "当前 epoch P1 应被召回注入"
    block = "\n".join(m["content"] for m in out if m["role"] == "system")
    assert "NEW_EPOCH 当前 任务" in block, "当前 epoch P1 应被召回注入"
    # 确定性：同一 select_context 两次 → 注入块全等（Recaller.search 每次返回 fresh Hit）
    out2 = e.select_context([{"role": "user", "content": q}],
                            conversation_messages=[{"role": "user", "content": q}])
    assert "\n".join(m["content"] for m in out2 if m["role"] == "system") == block


def test_engine_compress_hard_evicts_old_epoch(tmp_path):
    """P2.3 hard e2e：compress 把旧 epoch 非结构 P1 全文逐出窗口（无 digest → 占位符），可逆。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_e2e_hard")
    e._drift.current_epoch = 1
    goal = _seg("GOAL_PIN 顶层目标", tier="P0", prov="user", structural=True, goal="g",
                topic="0", seq=0)
    old_p1 = _seg("OLD_EPOCH 旧任务 推理内容 很长 中间推理 FULLTEXT_MARKER_OLD",
                  tier="P1", topic="0", seq=1)
    new_p1 = _seg("NEW_EPOCH 当前 任务 部署 FULLTEXT_MARKER_NEW", tier="P1", topic="1", seq=2)
    e.store.append(goal)
    e.store.append(old_p1)
    e.store.append(new_p1)
    e.store.flush()                                     # P1 → durable（进 _window_fresh）
    out = e.compress([{"role": "user", "content": "继续 当前 任务 部署"}])
    block = "\n".join(m["content"] for m in out if m["role"] == "system")
    # 旧 epoch P1 全文被 hard 逐出窗口（无 digest → 不渲染 FULLTEXT_MARKER_OLD）
    assert "FULLTEXT_MARKER_OLD" not in block, "旧 epoch P1 全文应被 hard 逐出（不在窗口）"
    # 当前 epoch P1 全文保留
    assert "FULLTEXT_MARKER_NEW" in block, "当前 epoch P1 应保留全文"
    # 可逆：jsonl 单一事实源原文仍在盘上
    back = {s.segment_id: s.original_text for s in e.store.replay()}
    assert "FULLTEXT_MARKER_OLD" in back[old_p1.segment_id], "hard 逐出字节不动（可逆）"


def test_engine_on_session_start_rehydrates(tmp_path):
    """冷重开 e2e：on_session_start 有盘内容 → rehydrate（取盘上最大 epoch，防假阳性漂移）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_rehy")
    e.store.append(_seg("旧推理", tier="P1", topic="1", seq=0))
    e.store.flush()                                     # 落盘（重启前写的事实）
    # 同 sid 冷重开（新实例绑同文件）
    e2 = _mkengine(tmp_path)
    e2.on_session_start("s_rehy")
    assert e2._drift.current_epoch == 1, "冷重开应 rehydrate 到盘上最大 epoch（=1）"
    assert e2._drift.is_old_epoch(_seg("x", topic="1")) is False, "重启前段不得判旧（假阳性护城河）"
