"""extended-BM25 招② 字段/slot 加权单元面（acceptance 面）。

覆盖 sc/recall.py 的 P0/goal/AC 字段加权（FIELD_WEIGHT 副本）对计分的确定性影响：
- _field_tokens：仅 P0 段取 goal/open_tasks(AC) 字段 token；P1/P2 填充物段 0 副本
  （按 slot/tier 确定性拉开权重）；
- 同词出现在高权重 slot（P0 goal/AC 字段）比出现在低权重 slot（P0/P1 原文 filler）
  贡献更多最终得分（同 tier 对比隔离出 FIELD_WEIGHT 效应，tier boost 不干扰）；
- 确定性：同 query 两次全等（含字段加权面）。
"""
from sc.recall import Recaller, _field_tokens
from helpers import mkseg, mk_window
from sc.schema import Segment

# P0 段 1：TERM 只出现在高权重 slot（goal + 两条 AC desc），原文纯填充物。
_FIELD_AC = [{"id": "AC1", "desc": "TERM"}, {"id": "AC2", "desc": "TERM"}]


def test_field_tokens_only_p0_tier():
    """_field_tokens 仅 P0 段非空（slot/tier 判定纯字面、确定性）。"""
    p1 = Segment(tier="P1", provenance="model", original_text="x",
                 goal="TERM", open_tasks=_FIELD_AC)
    assert _field_tokens(p1) == [], "P1 段字段 token 不得参与加权"
    p2 = Segment(tier="P2", provenance="model", original_text="x", goal="TERM")
    assert _field_tokens(p2) == []
    p0 = Segment(tier="P0", provenance="user", original_text="x",
                 goal="TERM", open_tasks=_FIELD_AC)
    toks = _field_tokens(p0)
    assert toks.count("term") == 3, "goal ×1 + 两条 AC desc ×1 + 原文 0 → term 3 副本"
    assert "ac1" in toks and "ac2" in toks
    assert _field_tokens(p0) == toks, "确定性：同段两次同输出"


def test_field_slot_beats_filler_slot_same_tier(store, aux):
    """acceptance：同 P0 tier，同词 TERM 在字段 slot（goal/AC）贡献 > 在 filler 原文 slot。

    两段均为 P0（P0_BOOST 同乘、排序键 tier 面同）→ 得分差纯由 FIELD_WEIGHT 加权副本
    拉开：字段段 tf(term)=3（×FIELD_WEIGHT 副本）> filler 段 tf(term)=1（原文单发）。
    """
    field_seg = mkseg(store, "filler filler filler filler",
                      "P0", "user", structural=True,
                      goal="TERM", open_tasks=_FIELD_AC)
    filler_seg = mkseg(store, "term filler filler filler", "P0", "user", structural=True)
    w = mk_window(2000, [field_seg, filler_seg])
    hits = Recaller(store, w).search("TERM", k=2)
    assert hits[0].seg.segment_id == field_seg.segment_id, \
        "同 tier 下字段 slot 段应排 filler slot 段之前（FIELD_WEIGHT 副本放大 tf）"
    assert hits[0].score > hits[1].score, "字段 slot 段得分应严格更高"


def test_p0_field_beats_p1_same_term(store, aux):
    """高权重 slot 组合面：P0 字段命中段（P0_BOOST × FIELD_WEIGHT）胜过同词 P1 填充物段。"""
    p0 = mkseg(store, "filler filler filler filler",
               "P0", "user", structural=True,
               goal="TERM", open_tasks=_FIELD_AC)
    p1 = mkseg(store, "term rest rest", "P1")
    w = mk_window(2000, [p0, p1])
    hits = Recaller(store, w).search("TERM", k=2)
    assert hits[0].seg.segment_id == p0.segment_id, "P0 字段段应排 P1 填充物段之前"
    assert hits[0].score > hits[1].score


def test_field_weight_determinism_two_runs(store, aux):
    """G4 确定性：字段加权面（CJK goal + AC）下同 query 两次全等。"""
    p0 = mkseg(store, "filler filler filler filler",
               "P0", "user", structural=True,
               goal="网关部署 nginx", open_tasks=[{"id": "AC", "desc": "网关 上线"}])
    p1 = mkseg(store, "网关 nginx 上线 部署 rest", "P1")
    w = mk_window(2000, [p0, p1])
    q = "网关 nginx"
    a = [(h.seg.segment_id, round(h.score, 9)) for h in Recaller(store, w).search(q, k=2)]
    b = [(h.seg.segment_id, round(h.score, 9)) for h in Recaller(store, w).search(q, k=2)]
    assert a == b, "字段加权面非确定（两次结果不等）"
    assert a[0][0] == p0.segment_id, "P0 字段段应排第一"
