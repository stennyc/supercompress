"""gate §3 t_62c8ad6f · 可观测面验收（纯 read-only 观测，零数据变更）。

红线（卡体钉死）：
- 本卡 git diff 只动 sc/inject.py（打 INFO 的调用侧观测函数 + 注入面）+ 本新增测试文件；
  scdata.py / model_store.py / store.py 生产逻辑零改动（消费既有机制，不新造校验）。
- 观测面 read-only：不迁移历史 ts=0 段、不碰 ledger append-only、不删行；
  零 store/ledger/.db 写；宿主 agent/ 零 diff；不推远端。

消费既有机制（不新写 torn-detection）：
- replay dropped 观测行 = 消费 scdata.replay 既有 ReplayIndex.dropped
  （scdata.py:89-93 单一实现产出的 (line_no, reason) 清单，reason 分布
  json_parse_fail / content_hash_mismatch / dup_skip_idempotent 已在盘）。
- ts_zero_ratio 观测行 = 当前窗口 ts==0 段占比，纯 INFO 不告警
  （阈值裁定面是 fix-2 修复卡，本卡只观测、不迁移、不重算）。

跑法（repo 根）：python -m pytest tests/test_obs_replay_dropped.py -v
"""
from __future__ import annotations

import logging

import scdata
from sc.inject import observe_replay_drops, observe_ts_zero_ratio
from helpers import mkseg, mk_window

_OBS = "sc.inject"


def _cap_info(caplog: logging.LogRecord, prefix: str) -> list[str]:
    """取 caplog 里本模块（sc.inject）发的前缀 INFO 消息文本（含未格式化 args 展开）。"""
    return [r.getMessage() for r in caplog.records
            if r.name == _OBS and r.levelno == logging.INFO and r.getMessage().startswith(prefix)]


def _write_torn_jsonl(tmp_path, extra_lines=()) -> str:
    """写一个含有效头 + torn 尾的 jsonl（replay 会因 hash 校验静默丢 torn 行及之后）。

    权威数据面 = scdata.append_line/to_line（scdata 单一实现，本卡零改动只消费其产物）。"""
    p = str(tmp_path / "torn.jsonl")
    scdata.append_line(p, scdata.make_segment(0, "valid head fact"))
    scdata.append_line(p, scdata.make_segment(1, "another valid fact"))
    for raw in extra_lines:
        with open(p, "a", encoding="utf-8") as f:
            f.write(raw + "\n")
    return p


class TestObserveReplayDrops:
    def test_zero_dropped_reports_n0(self, tmp_path, caplog):
        """干净 jsonl（零 torn）→ 一行 N=0 INFO（观测面恒定可 grep，不因无丢而静默）。"""
        p = _write_torn_jsonl(tmp_path)
        with caplog.at_level(logging.INFO, _OBS):
            observe_replay_drops(scdata.replay(p))
        rows = _cap_info(caplog, "supercompress: replay dropped")
        assert len(rows) == 1, "恰一行 INFO（不重复、不静默）"
        assert "replay dropped 0 line(s)" in rows[0], "N=0 也要报（可观测面）"

    def test_torn_tail_reports_count_and_reason(self, tmp_path, caplog):
        """torn 尾（hash mismatch）→ 消费既有 idx.dropped 报 N + reason 分布。

        不新写 torn-detection：直接读 scdata.replay 产物（单一实现），本卡只打 INFO。"""
        bad = scdata.to_line(scdata.make_segment(2, "tail that is torn"))
        # 篡改 hash 制造 content_hash_mismatch（replay 权威面会验出并静默丢该行 + 其后）
        import json as _json
        bad = bad.replace('"content_hash"', '"content_hash_x"', 1)
        p = _write_torn_jsonl(tmp_path, extra_lines=(bad,))
        idx = scdata.replay(p)
        # 前置：权威数据面确实把 torn 行丢进既有 dropped（消费的前提）
        assert any(r == "content_hash_mismatch" for _, r in idx.dropped)
        with caplog.at_level(logging.INFO, _OBS):
            observe_replay_drops(idx)
        rows = _cap_info(caplog, "supercompress: replay dropped")
        assert len(rows) == 1
        assert "replay dropped 1 line(s)" in rows[0]
        assert "content_hash_mismatch" in rows[0], "reason 分布如实上报（既有机制产物）"

    def test_reason_distribution_aggregates(self, tmp_path, caplog):
        """同一 reason 多行（dup_skip_idempotent）→ 分布聚合计数（reason dict 汇总，非逐行刷屏）。"""
        p2 = str(tmp_path / "dup.jsonl")
        scdata.append_line(p2, scdata.make_segment(0, "valid head fact"))
        # 追加一条与已入库段同 original_text 的重复行 → replay 幂等去重丢（既有 dup_skip_idempotent）
        scdata.append_line(p2, scdata.make_segment(9, "valid head fact"))
        idx = scdata.replay(p2)
        assert [r for _, r in idx.dropped] == ["dup_skip_idempotent"]
        with caplog.at_level(logging.INFO, _OBS):
            observe_replay_drops(idx)
        rows = _cap_info(caplog, "supercompress: replay dropped")
        assert "replay dropped 1 line(s)" in rows[0]
        assert "dup_skip_idempotent" in rows[0]

    def test_read_only_zero_write(self, tmp_path, caplog):
        """观测面 read-only 红线：打 INFO 前后磁盘 jsonl 字节零变（不迁历史、不删行、不重算）。"""
        p = str(tmp_path / "ro.jsonl")
        scdata.append_line(p, scdata.make_segment(0, "a"))
        scdata.append_line(p, scdata.make_segment(1, "b"))
        before = open(p, "rb").read()
        with caplog.at_level(logging.INFO, _OBS):
            observe_replay_drops(scdata.replay(p))
        after = open(p, "rb").read()
        assert before == after, "观测面零写（磁盘字节不变）"


class TestObserveTsZeroRatio:
    def _segs(self, store):
        return [mkseg(store, f"fact {i}", "P1") for i in range(4)]

    def test_all_ts_zero_ratio_1(self, store, caplog):
        """窗口段全 ts=0（既有 fixture 默认 ts=0.0）→ ratio=1.0000，纯 INFO 不告警。"""
        segs = self._segs(store)
        assert all(s.ts == 0 for s in segs), "前置：mkseg 默认 ts=0（历史段）"
        with caplog.at_level(logging.INFO, _OBS):
            observe_ts_zero_ratio(segs)
        rows = _cap_info(caplog, "supercompress: ts_zero_ratio")
        assert len(rows) == 1
        assert "ts_zero_ratio=1.0000" in rows[0]
        assert "(n=4/4" in rows[0], "count/len 如实（观测不裁定）"

    def test_mixed_ts_ratio(self, store, caplog):
        """4 段 1 段带 ts>0 → ratio=0.75（只报事实，不迁移/不重算历史 ts=0）。"""
        segs = self._segs(store)
        segs[0].ts = 1730000000.0   # 仅内存态改观测输入，零写盘
        with caplog.at_level(logging.INFO, _OBS):
            observe_ts_zero_ratio(segs)
        rows = _cap_info(caplog, "supercompress: ts_zero_ratio")
        assert "ts_zero_ratio=0.7500" in rows[0], "3/4 ts=0 → 0.75"
        assert "(n=3/4" in rows[0]

    def test_empty_window_no_op(self, caplog):
        """空窗口（无定义）→ 静默 no-op，不发占比行（不报 0/0）。"""
        with caplog.at_level(logging.INFO, _OBS):
            observe_ts_zero_ratio([])
        assert _cap_info(caplog, "supercompress: ts_zero_ratio") == []

    def test_uses_window_seg_surface(self, store, caplog):
        """窗口面 = Window.segs（WindowSeg 包装，ts 面同 Segment.ts）→ 观测读 WindowSeg 亦可。"""
        segs = self._segs(store)
        w = mk_window(100000, segs)
        with caplog.at_level(logging.INFO, _OBS):
            observe_ts_zero_ratio(w.segs)
        rows = _cap_info(caplog, "supercompress: ts_zero_ratio")
        assert "ts_zero_ratio=1.0000" in rows[0], "WindowSeg 透传 Segment.ts（零 mutation）"

    def test_observation_does_not_mutate_segments(self, store):
        """观测不迁移：打 ts_zero_ratio 前后各段 ts 值零变（只读，不动历史 ts=0 段）。"""
        segs = self._segs(store)
        snapshot = [s.ts for s in segs]
        observe_ts_zero_ratio(segs)
        assert [s.ts for s in segs] == snapshot, "观测零 mutation（不重算/不迁移）"
