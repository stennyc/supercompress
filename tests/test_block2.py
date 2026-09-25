"""Block ② memory_recall：点查 O(1) + BM25 top-k（P0 boost）+ Pull 预算 enforce + stale 降权。"""
from sc.store import JsonlStore
from sc.budgets import usable_tokens, pull_budget, push_cap
from sc.distill import Distiller
from sc.recall import Recaller
from sc.schema import est_tokens
from helpers import mkseg, mk_window


class TestBlock2_Recall:
    def test_point_lookup_o1_returns_original(self, store, aux):
        p0 = mkseg(store, "deploy release 42 to C:/srv", "P0", "user", structural=True)
        rec = Recaller(store, mk_window(1000, [p0]))
        got = rec.point_fetch(p0.segment_id)
        assert got is not None and got.original_text == p0.original_text
        assert rec.point_fetch("doesnotexist") is None

    def test_bm25_top3_p0_boost(self, store, aux):
        p0 = mkseg(store, "deploy release 42 to C:/srv", "P0", "user", structural=True)
        p2 = mkseg(store, "deploy deploy dump dump dump release release", "P2")
        rec = Recaller(store, mk_window(1000, [p0, p2]))
        hits = rec.search("42 C:/srv deploy release", k=2)
        assert len(hits) >= 1
        assert hits[0].seg.segment_id == p0.segment_id, "P0 boost：同相关度 P0 优先"

    def test_pull_budget_20pct_enforced(self, store, aux):
        p1a = mkseg(store, ("port 8080 used by nginx " * 30)[:700], "P1")
        p1b = mkseg(store, ("port 8080 conflict checked " * 30)[:700], "P1")
        p2a = mkseg(store, ("port 8080 stdout dump " * 30)[:700], "P2")
        w = mk_window(1000, [p1a, p1b, p2a])
        rec = Recaller(store, w)
        # 非平凡前置：P1/P2 候选总量真顶穿 20% 预算，降权分支才会 fire
        total = sum(est_tokens(s.original_text) for s in (p1a, p1b, p2a))
        assert total > rec.budget, "候选总量必须顶穿预算"
        hits = rec.search("port 8080", k=3)
        by_id = {h.seg.segment_id: h for h in hits}
        assert by_id[p1a.segment_id].downweighted is False, "预算内的 P1 保留"
        assert by_id[p1b.segment_id].downweighted is True, "超预算 P1 降权（P2 之后）"
        assert by_id[p2a.segment_id].downweighted is True, "P2 优先降权"

    def test_p0_recall_exempt_writes_breach_log(self, store, aux):
        p1 = mkseg(store, "port 8080 used by nginx ", "P1")
        p0 = mkseg(store, ("release 42 deploy C:/srv " * 20), "P0", "user", structural=True)
        w = mk_window(200, [p0, p1])  # 预算 = 40 tok
        rec = Recaller(store, w)
        hits = rec.search("release 42 deploy", k=2)
        p0_hit = {h.seg.segment_id: h for h in hits}[p0.segment_id]
        assert p0_hit.downweighted is False, "P0 豁免预算、强制通过"
        assert any(b["segment_id"] == p0.segment_id for b in rec.breach_log), "P0 超预算写 budget_breach"

    def test_shared_denominator_post_distill(self, store, aux):
        p0 = mkseg(store, "release 42 deploy", "P0", "user", structural=True)
        w = mk_window(1000, [p0])
        assert pull_budget(w) == int(0.20 * usable_tokens(w))
        assert push_cap(w) == int(0.15 * usable_tokens(w))
        rec = Recaller(store, w)
        assert rec.budget == pull_budget(w), "Pull 预算必须读共享分母"
        d = Distiller(w, store, aux)
        assert d.push_cap_tokens == push_cap(w), "Push cap 必须读同一分母（一处定义）"

    def test_no_digest_shows_marker_not_silent(self, store, aux):
        p1 = mkseg(store, "unique token zebra42 somewhere deep" * 20, "P1")
        aux.start_turn()  # digest 入队 → 逐出时未 ready
        w = mk_window(100, [p1])
        d = Distiller(w, store, aux).run()
        assert w.segs[0].form == "digest_only"
        assert "[no digest]" in w.segs[0].excerpt(), "无 digest 必须显式标记，不静默"
        assert w.segs[0].tokens() <= 16, "marker 走 head/tail 摘录，占位小"

    def test_stale_p1_downweighted(self, store, tmp_path, aux):
        s_old = JsonlStore(str(tmp_path / "old.jsonl"))
        p1_old = mkseg(s_old, "port 8080 nginx config on host alpha", "P1")
        s_old.flush()
        s2 = JsonlStore.materialize(str(tmp_path / "old.jsonl"),
                                    str(tmp_path / "active.jsonl"), reopen_from="arch-1")
        assert len(s2.by_hash) > 0, "非空集前置"
        p1_old2 = s2.get(p1_old.segment_id)
        assert p1_old2.reopened_from == "arch-1"
        p1_new = mkseg(s2, "port 8080 nginx config on host alpha", "P1")
        rec = Recaller(s2, mk_window(1000, [p1_old2, p1_new]))
        hits = rec.search("port 8080 nginx alpha", k=2)
        assert hits[0].seg.segment_id == p1_new.segment_id, "stale P1 召回降权"
        stale_hit = [h for h in hits if h.seg.segment_id == p1_old2.segment_id][0]
        assert stale_hit.stale is True and "verify before acting" in stale_hit.note
