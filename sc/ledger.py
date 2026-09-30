"""Block ③ 活账本：顶层 goal 钉死 + phase 可 supersede + 确定性触发 + DONE 三级证据。

设计锁（gate 定稿）：
- 活账本行 = P0 structural ledger 行（write-before-compress：即写即 fsync，永不逐出）；
- 顶层 goal 仅 user-sourced，永久 pin（push cap 压力下永不裁）；
- 新 user instruction → 旧 phase supersede：未完成 open task 关为 blocked（jsonl 留审计，
  不删行——superseded 标记进 open_tasks 快照）；
- 确定性触发：turn 边界每 N 个**且 on-delta**（状态签名未变 → 去重跳过，不写不 fsync）；
- DONE 三级证据：verified（带证据，永久滚出常驻集）/ declared（无证据，可 re-open）/
  missed（未知 task id，无信号不落账，漏账计数进 bench）。
"""
from __future__ import annotations

import json

from .schema import Segment


class Ledger:
    def __init__(self, store, session_topic: str, goal: str, checkpoint_every: int = 0) -> None:
        self.store = store
        self.session_topic = session_topic
        self.goal = goal                # 顶层目标：仅 user-sourced，永久 pin
        self.checkpoint_every = checkpoint_every
        self.phase_no = 0
        self.phase_id: str | None = None
        self.state: str | None = None
        self.last_segment: Segment | None = None
        self.history: list[Segment] = []        # 已关闭 phase 行（jsonl 审计链）
        self.open_tasks: dict[str, dict] = {}   # session 级 task 表（含 superseded 标记）
        self.missed_count = 0                   # 无信号不落账计数（进 bench 分母）
        self._boundary_count = 0
        self._last_sig = self._sig()

    # ───────────── phase 生命周期 ─────────────
    def on_user_instruction(self, text: str) -> Segment:
        """新 user instruction = 确定性 phase 迁移：旧 phase supersede 关闭 + 新 phase 开账。"""
        if self.phase_id is not None:
            self._close_phase()
        self.phase_no += 1
        self.phase_id = "p%d" % self.phase_no
        self.state = "in_progress"
        row = Segment(
            kind="ledger", tier="P0", provenance="user",
            original_text="ledger: " + text, structural_p0=True,
            goal=self.goal, phase_id=self.phase_id, state="in_progress",
            open_tasks=[dict(t) for t in self._phase_tasks()],
            session_topic=self.session_topic,
        )
        self.last_segment = self.store.append(row)   # P0 即写即 fsync
        return row

    def _close_phase(self) -> None:
        """supersede 语义：未完成 open task 的旧 phase 关为 blocked；completed-verified 才 done。
        jsonl 留审计：行不删，superseded 标记进快照。"""
        row = self.last_segment
        if row is None:
            return
        open_ts = [t for t in self._phase_tasks() if not (t["done"] and t["verified"])]
        for t in open_ts:
            t["superseded"] = True
        row.state = "blocked" if open_ts else "done"
        row.open_tasks = [dict(t) for t in self._phase_tasks()]
        self.history.append(row)

    def _phase_tasks(self) -> list[dict]:
        return [t for t in self.open_tasks.values() if t.get("phase") == self.phase_id]

    # ───────────── open tasks（三级证据）─────────────
    def add_task(self, task_id: str, desc: str) -> None:
        self.open_tasks[task_id] = {
            "id": task_id, "desc": desc, "phase": self.phase_id,
            "done": False, "verified": False, "inferred": False,
            "superseded": False, "done_evidence": None,
        }

    def on_done(self, task_id: str, evidence: str | None = None) -> str:
        t = self.open_tasks.get(task_id)
        if t is None:
            self.missed_count += 1                 # 无信号不落账（漏账计数进 bench）
            return "missed"
        t["done"] = True
        if evidence:
            t["verified"] = True
            t["done_evidence"] = evidence
            return "verified"
        t["inferred"] = True                       # declared：保持可 re-open
        t["verified"] = False
        return "declared"

    def resident_open_tasks(self) -> list[str]:
        """常驻集：superseded / verified-done 滚出；declared-done 保持（可 re-open）。"""
        return [tid for tid, t in self.open_tasks.items()
                if not t["superseded"] and not (t["done"] and t["verified"])]

    # ───────────── 确定性触发：on-delta checkpoint ─────────────
    def turn_boundary(self) -> bool:
        """每 N 边界 + 状态签名有 delta → P0 写 checkpoint；状态未变 → 去重跳过（不写不 fsync）。"""
        if not self.checkpoint_every:
            return False
        self._boundary_count += 1
        if self._boundary_count % self.checkpoint_every != 0:
            return False
        sig = self._sig()
        if sig == self._last_sig:
            return False
        self.store.append(Segment(
            kind="ledger", tier="P0", provenance="user",
            original_text="checkpoint: %s %s" % (self.phase_id, self.state),
            structural_p0=True, goal=self.goal, phase_id=self.phase_id,
            state=self.state, open_tasks=[dict(t) for t in self._phase_tasks()],
            session_topic=self.session_topic,
        ))
        self._last_sig = sig
        return True

    def _sig(self) -> tuple:
        tasks = sorted((k, tuple(sorted(v.items()))) for k, v in self.open_tasks.items())
        return (self.phase_id, self.state, self.goal, json.dumps(tasks, default=str))
