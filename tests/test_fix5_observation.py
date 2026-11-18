"""fix-5 t_d888d710 · 可观测面 audit 写面 + 零 mutation 硬验收（gate §3 双路径，纯 read-only 消费）。

gate §3 双路径（纯消费既有 idx.dropped / Segment.ts，零 mutation，不新造校验）：
- INFO 侧（t_62c8ad6f 已建，见 test_obs_replay_dropped.py）：observe_replay_drops / observe_ts_zero_ratio。
- audit 悬挂侧（本卡新增，sc/audit.py）：<sid>.audit.jsonl O_APPEND 单行 + 0600 + flush，
  写 OSError → WARNING + fail-soft 继续（零 raise），同 compact_ledger._persist 失败契约。

红线（卡体钉死）：
- 零 store/log/.db 写（只新增**独立** audit 侧车，主 store jsonl 字节不变）；宿主 agent/ 零 diff；
  不推远端；零 gateway restart。
- 本卡 git diff 只动 sc/inject.py（双路径统一入口 + INFO）+ 新增 sc/audit.py（audit 写面）
  + 本测试文件；scdata.py 生产逻辑零改动（消费既有 ReplayIndex.dropped，已单源实装）。

跑法（repo 根）：python -m pytest tests/test_fix5_observation.py -v
"""
from __future__ import annotations

import json
import logging
import os
import stat

import scdata
from helpers import mkseg
from sc.inject import observe_replay_window
from sc import audit

_OB = "sc.inject"
_AU = "sc.audit"


def _last_audit_record(jsonl_path: str) -> dict:
    """读 audit 侧车最后一行（O_APPEND 单行）。"""
    lines = open(audit.audit_path_for_jsonl(jsonl_path)).read().splitlines()
    assert lines, "audit 侧车非空"
    return json.loads(lines[-1])


class TestObservationNoMutation:
    """卡面硬验收：replay 后 by_hash / ts / seq / content_hash 全不变 + 主 store 字节零变。"""

    def test_observation_no_mutation(self, store, caplog):
        # 真实 store 布局（fixture API）：3 段默认 ts=0 + 一条内存态改 ts>0（零写盘），制造 mixed ratio
        segs = [mkseg(store, f"fact {i}") for i in range(3)]
        segs[0].ts = 1730000000.0
        store.flush()                       # P1 lazy 批落盘，物化主 store jsonl（观测前基线）
        jsonl_path = store.path
        # 快照：段级 (seq, ts, content_hash, segment_id) + 磁盘 store 字节 + by_hash 键集
        seg_snap = [(s.seq, s.ts, s.content_hash, s.segment_id) for s in segs]
        disk_before = open(jsonl_path, "rb").read()
        by_hash_before = set(store.by_hash.keys())
        idx = scdata.replay(jsonl_path)                       # 权威数据面 ReplayIndex（单源）
        with caplog.at_level(logging.INFO, _OB):
            rec = observe_replay_window(idx, segs, sid="s", jsonl_path=jsonl_path)
        # ── 零 mutation 断言 ──
        assert [(s.seq, s.ts, s.content_hash, s.segment_id) for s in segs] == seg_snap, \
            "replay/观测后段级 ts/seq/content_hash/segment_id 全不变"
        assert open(jsonl_path, "rb").read() == disk_before, "主 store jsonl 字节零变（audit 只写独立侧车）"
        assert set(store.by_hash.keys()) == by_hash_before, "by_hash 键集不变"
        # audit 侧车是**新增**独立文件（允许），record 字段齐 + 消费既有数据
        assert os.path.exists(audit.audit_path_for_jsonl(jsonl_path))
        assert rec["audit_written"] is True
        assert set(rec) >= {"ts", "sid", "dropped", "ts_zero_ratio"}
        assert rec["sid"] == "s"
        # 纯消费既有 Segment.ts：2/3 ts=0 → 0.6667；dropped 空（干净 store）→ {}
        assert abs(rec["ts_zero_ratio"] - (2 / 3)) < 1e-9, "ts_zero_ratio 只读消费 Segment.ts"
        assert rec["dropped"] == {}, "干净 store N=0 → reason 分布空 dict"


class TestAuditWritePrimitive:
    """写原语真值（security-engineer 裁定）：O_APPEND 单行 + 0600 + flush，非 os.replace。"""

    def test_append_single_line_0600(self, tmp_path):
        p = str(tmp_path / "s.audit.jsonl")
        assert audit.write_audit_record(p, {"ts": 1.0, "sid": "s", "dropped": {}, "ts_zero_ratio": 0.0}) is True
        if os.name == "posix":  # win32 stat mode bits 恒定 0666：0600 收紧面 POSIX-only
            assert stat.S_IMODE(os.stat(p).st_mode) == 0o600, "新文件 0600"
        lines = open(p).read().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["sid"] == "s"
        # 追加第二条（O_APPEND 同 inode，两条）
        audit.write_audit_record(p, {"ts": 2.0, "sid": "s", "dropped": {"dup_skip_idempotent": 1}, "ts_zero_ratio": 0.5})
        lines = open(p).read().splitlines()
        assert len(lines) == 2, "O_APPEND 追加（非整文件重写）"
        assert json.loads(lines[1])["dropped"] == {"dup_skip_idempotent": 1}
        # 非 os.replace：audit 写面无任何 os.replace( 调用（append-only 痕迹语义，整文件换 inode 会断档）
        assert "os.replace(" not in open(audit.__file__, encoding="utf-8").read()

    def test_fchmod_tightens_existing_loose(self, tmp_path):
        """已存在 0644 偏宽文件 → 追加后强制收紧 0600 + 原行保留（append 非替换）。"""
        p = str(tmp_path / "s.audit.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write('{"prev":1}\n')
        os.chmod(p, 0o644)
        audit.write_audit_record(p, {"ts": 1.0, "sid": "s", "dropped": {}, "ts_zero_ratio": 0.0})
        if os.name == "posix":  # win32 stat mode bits 恒定 0666：fchmod 收紧面 POSIX-only
            assert stat.S_IMODE(os.stat(p).st_mode) == 0o600, "fchmod 收紧到 0600"
        lines = open(p).read().splitlines()
        assert len(lines) == 2 and json.loads(lines[0]) == {"prev": 1}, "原行保留（append，非 os.replace 换 inode）"

    def test_oserror_fail_soft_zero_raise(self, tmp_path, caplog, monkeypatch):
        """写 OSError（ENOSPC/EIO）→ WARNING + fail-soft 返回 False，零 raise，主流程继续。"""
        import sc.audit as A
        monkeypatch.setattr(A.os, "open",
                            lambda file, flags, *a, **k: (_ for _ in ()).throw(
                                OSError(28, "No space left on device", file)))
        with caplog.at_level(logging.WARNING, _AU):
            ok = A.write_audit_record(str(tmp_path / "s.audit.jsonl"),
                                      {"ts": 1.0, "sid": "s", "dropped": {}, "ts_zero_ratio": 0.0})
        assert ok is False, "写失败 → 返回 False（fail-soft）"
        assert any("audit write failed" in r.getMessage() for r in caplog.records
                   if r.levelno == logging.WARNING), "打一行 WARNING（同 compact_ledger 契约）"
        assert not os.path.exists(str(tmp_path / "s.audit.jsonl")), "未落盘（零 raise，非炸宿主）"


class TestDualPathFailSoft:
    """双路径端到端：audit 写失败时 INFO 侧仍打、record 仍返回、零 raise（观测面挂了不影响主流程）。"""

    def test_observe_replay_window_fail_soft_end_to_end(self, store, caplog, monkeypatch):
        import sc.audit as A
        segs = [mkseg(store, f"fact {i}") for i in range(2)]
        monkeypatch.setattr(A.os, "open",
                            lambda file, flags, *a, **k: (_ for _ in ()).throw(
                                OSError(28, "No space left on device", file)))
        idx = scdata.replay(store.path)
        with caplog.at_level(logging.INFO, _OB):
            rec = observe_replay_window(idx, segs, sid="s", jsonl_path=store.path)
        assert rec["audit_written"] is False, "audit 写失败如实标记（record 仍返回）"
        # INFO 侧不依赖 audit 成败，仍打 dropped 行（观测面主流程零感知）
        assert any(r.getMessage().startswith("supercompress: replay dropped")
                   for r in caplog.records if r.levelno == logging.INFO)
        assert not os.path.exists(audit.audit_path_for_jsonl(store.path)), "audit 未落盘"


class TestDroppedReasonConsumption:
    """双路径 record 的 `dropped` 如实消费既有 ReplayIndex.dropped reason 分布（不新造 torn-detection）。"""

    def test_dropped_reason_distribution_land(self, tmp_path, caplog):
        p = str(tmp_path / "torn.jsonl")
        scdata.append_line(p, scdata.make_segment(0, "valid head fact"))
        bad = scdata.to_line(scdata.make_segment(2, "tail that is torn"))
        bad = bad.replace('"content_hash"', '"content_hash_x"', 1)   # 篡改 hash → content_hash_mismatch
        with open(p, "a", encoding="utf-8") as f:
            f.write(bad + "\n")
        idx = scdata.replay(p)
        assert any(r == "content_hash_mismatch" for _, r in idx.dropped), "前置：权威面确实丢进既有 dropped"
        with caplog.at_level(logging.INFO, _OB):
            rec = observe_replay_window(idx, [], sid="s", jsonl_path=p)   # 空窗口 → ts_zero_ratio 0.0
        assert rec["dropped"] == {"content_hash_mismatch": 1}, "reason 分布如实消费（非逐行刷屏）"
        assert rec["ts_zero_ratio"] == 0.0, "空窗口占比 0.0（无定义，caller 判空）"
        assert rec["audit_written"] is True
        assert _last_audit_record(p)["dropped"] == {"content_hash_mismatch": 1}
