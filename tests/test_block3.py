"""Block ③ 活账本：goal 钉死 + phase supersede + 确定性 on-delta checkpoint + DONE 三级证据。"""
from sc.budgets import push_cap
from sc.ledger import Ledger
from sc.distill import Distiller
from helpers import mkseg, complete_digest, mk_window, TOPIC


class TestBlock3_Ledger:
    def test_ledger_on_user_instruction(self, store):
        led = Ledger(store, TOPIC, goal="ship v1")
        led.on_user_instruction("build module X")
        assert led.phase_id == "p1" and led.state == "in_progress"
        row = store.get(led.last_segment.segment_id)
        assert row.kind == "ledger" and row.tier == "P0" and row.structural_p0
        led.add_task("t1", "write tests")
        led.on_user_instruction("build module Y")
        assert led.phase_id == "p2"
        closed = [r for r in led.history if r.phase_id == "p1" and r.state in ("done", "blocked")]
        assert closed and closed[0].state == "blocked", "有未完成 open task → 旧 phase 关为 blocked"

    def test_done_verified_requires_evidence(self, store):
        led = Ledger(store, TOPIC, goal="ship v1")
        led.on_user_instruction("task t1")
        led.add_task("t1", "write tests")
        assert led.on_done("t1", evidence="pytest exit 0") == "verified"
        t = led.open_tasks["t1"]
        assert t["verified"] is True and t["done_evidence"] == "pytest exit 0"
        assert "t1" not in led.resident_open_tasks(), "verified 可永久滚出"

    def test_done_declared_no_evidence_downgraded(self, store):
        led = Ledger(store, TOPIC, goal="ship v1")
        led.on_user_instruction("task t2")
        led.add_task("t2", "refactor")
        assert led.on_done("t2") == "declared"
        t = led.open_tasks["t2"]
        assert t["inferred"] is True and t["verified"] is False
        assert "t2" in led.resident_open_tasks(), "declared 保持可 re-open"

    def test_done_no_signal_not_logged(self, store):
        led = Ledger(store, TOPIC, goal="ship v1")
        led.on_user_instruction("task t3")
        n0 = store.line_count
        assert led.on_done("tX") == "missed"
        assert led.missed_count == 1
        assert store.line_count == n0, "无信号不落账（漏账计数进 bench）"

    def test_checkpoint_on_delta_skip_on_no_change(self, store):
        led = Ledger(store, TOPIC, "ship v1", checkpoint_every=5)
        led.on_user_instruction("start")
        for _ in range(4):
            assert led.turn_boundary() is False
        assert led.turn_boundary() is True  # 第 5 边界 + 有 delta → P0 写
        n, fs = store.line_count, store.fsynced_p0_count
        for _ in range(4):
            assert led.turn_boundary() is False
        assert led.turn_boundary() is False, "on-delta：状态未变 → 去重跳过"
        assert store.line_count == n and store.fsynced_p0_count == fs, "不写不 fsync"

    def test_checkpoint_on_delta_write_on_transition(self, store):
        led = Ledger(store, TOPIC, "ship v1", checkpoint_every=5)
        led.on_user_instruction("start")
        for _ in range(5):
            led.turn_boundary()
        n, fs = store.line_count, store.fsynced_p0_count
        led.add_task("t9", "new work")  # 真实状态迁移
        for _ in range(4):
            assert led.turn_boundary() is False
        assert led.turn_boundary() is True
        assert store.line_count == n + 1 and store.fsynced_p0_count == fs + 1

    def test_goal_never_trimmed_by_cap(self, store, aux):
        goal_row = mkseg(store, "goal: ship supercompress v1", "P0", "user", structural=True,
                         kind="ledger", goal="ship supercompress v1", phase_id="p1")
        lp0s = [mkseg(store, f"llm p0 fact {i} " * 12, "P0") for i in range(4)]
        for s in lp0s:
            complete_digest(aux, s)
        p2s = [mkseg(store, f"filler {i} " * 40, "P2") for i in range(4)]
        w = mk_window(1000, [goal_row] + lp0s + p2s)
        d = Distiller(w, store, aux).run()
        assert w.segs[0].form == "resident", "顶层 goal 永不裁（硬不变量）"
        p0_resident = sum(ws.tokens() for ws in w.segs
                          if ws.segment.tier == "P0" and ws.form == "resident")
        assert p0_resident <= push_cap(w), "cap 压力下降档事实型 P0，P0 常驻集回到 15% 内"

    def test_supersede_on_user_instruction(self, store):
        led = Ledger(store, TOPIC, goal="ship v1")
        led.on_user_instruction("phase 1 goal")
        led.add_task("t1", "work item")
        led.on_user_instruction("phase 2 goal")
        assert "t1" not in led.resident_open_tasks(), "旧 open task 逐出常驻集"
        p1_rows = [r for r in led.history if r.phase_id == "p1"]
        assert any(t.get("superseded") for r in p1_rows for t in r.open_tasks), "jsonl 留审计"

    def test_supersede_by_evidence_level(self, store):
        led = Ledger(store, TOPIC, goal="ship v1")
        led.on_user_instruction("tasks")
        led.add_task("tv", "verified task")
        led.on_done("tv", evidence="exit 0")
        led.add_task("td", "declared task")
        led.on_done("td")
        assert "tv" not in led.resident_open_tasks(), "verified done 永久滚出"
        assert "td" in led.resident_open_tasks(), "declared 保持可 re-open"
