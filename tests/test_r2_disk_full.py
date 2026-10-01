"""[R2] full-disk OSError 优雅降级 + 磁盘水位监控（报告#2 Top-2）。

背景：磁盘写满时（EIO/ENOSPC），store 写路径底层 OSError 曾直接逃逸到宿主
（单 turn 崩 / /new 崩；tui_gateway_crash.log 5 处 EIO unhandledRejection 同因面）。
R2 收口：
- store 写路径（append P0 / flush）OSError 归一为 StoreError("io_error: ...")——
  调用方 catch StoreError 一族即可，裸 OSError 不再逃逸；
- engine 壳（compress / on_session_reset / on_session_end）双 catch
  (StoreError, OSError)：full-disk 时优雅降级（log warning + 水位观测，跳过本轮
  持久化，不炸宿主），与 on_turn_complete 宽 catch 口径对齐；
- distill 收尾 flush（异步面本就允许丢一个 pending 批）失败不上抛，
  pending 留内存可重试；锚点发射经 durable 过滤天然不含 pending（宁可不压不可断锚）；
- 磁盘水位观测：写盘前 ≥90% 打一行 INFO、写盘失败打一行 WARNING
  （剩余空间 + 触发动作，供 ops 监控 grep；不做自动清理——#5 retention 是独立任务）。

单测面：mock OSError 注入（monkeypatch store.flush / 写块 open），断言
compress() 返回不抛 + reset 不抛 + 日志行存在。

跑法（repo 根）：
  PYTHONPATH=<venv site-packages> /usr/bin/python3 -m pytest tests/test_r2_disk_full.py -q
"""
import builtins
import logging
import os
import stat

import pytest

from engine import SupercompressContextEngine
from sc.store import JsonlStore, StoreError, DISK_WATER_HIGH_PCT
from sc.schema import Segment
from sc.distill import Distiller, Window, WindowSeg
from helpers import TOPIC, MV


def _mkseg(text, tier="P0", prov="user", structural=False):
    return Segment(kind="segment", tier=tier, provenance=prov,
                   original_text=text, session_topic=TOPIC, model_version=MV,
                   structural_p0=structural)


def _mkengine(tmp_path, context_length=256000):
    return SupercompressContextEngine(context_length=context_length,
                                      external_store_path=str(tmp_path))


def _append_user(engine: SupercompressContextEngine, text: str) -> Segment:
    return engine.store.append(_mkseg(text, tier="P0"))


def _append_p1(engine: SupercompressContextEngine, text: str) -> Segment:
    return engine.store.append(_mkseg(text, tier="P1", prov="model"))


def _capture_logs(names, levelno):
    """在指定 logger 上挂 capturing handler（caplog 无关的确定性抓取面；
    engine 壳 logger = "engine"，store 水位 logger = "sc.store"——双 logger 面）。"""
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    attached = []
    for name in names:
        lg = logging.getLogger(name)
        h = _H(level=levelno)
        lg.addHandler(h)
        attached.append((lg, h))
    return records, attached


def _detach(attached):
    for lg, h in attached:
        lg.removeHandler(h)


def _fake_write_open(store_path, err):
    """选择性 open 替身：写 store 目标路径（append 块）抛 OSError，其余 open 行为不变。

    覆盖 os.open(store_path, O_WRONLY|O_APPEND, 0o644) 写块 + builtins.open(store_path, "r")
    读回块（读回不 mock——OSError 先断在写块）。"""
    real_os_open = os.open
    real_open = builtins.open

    def fake_os_open(file, flags, *a, **k):
        if os.fspath(file) == store_path and any(
                f in flags for f in (stat.O_WRONLY, stat.O_RDWR, stat.O_APPEND)):
            raise err
        return real_os_open(file, flags, *a, **k)

    def fake_open(file, mode="r", *a, **k):
        if os.fspath(file) == store_path and "r" not in str(mode):
            raise err
        return real_open(file, mode, *a, **k)

    return fake_os_open, fake_open


# ─────────────────────────── store 写路径 OSError 归一 ───────────────────────────
class TestStore_IoErrorNormalization:
    def test_flush_oserror_normalized(self, store, monkeypatch):
        """底层 OSError（full-disk EIO/ENOSPC）→ StoreError("io_error: ...")，裸 OSError 不逃逸。"""
        store.append(_mkseg("ctx a", tier="P1"))
        store.append(_mkseg("ctx b", tier="P1"))
        assert store.pending_count == 2

        err = OSError(5, "Input/output error")
        fake_os_open, fake_open = _fake_write_open(store.path, err)
        monkeypatch.setattr(os, "open", fake_os_open)
        monkeypatch.setattr(builtins, "open", fake_open)
        try:
            with pytest.raises(StoreError) as excinfo:
                store.flush()
            assert "io_error" in str(excinfo.value), "归一消息须可 grep（io_error 前缀）"
            assert excinfo.value.__cause__ is not None and isinstance(excinfo.value.__cause__, OSError)
            assert store.pending_count == 2, "flush 失败 → pending 批留内存（可重试，断电丢批上界不变）"
        finally:
            monkeypatch.undo()
        # 解除注入后正常 flush 恢复（批可重试语义的实证）
        assert store.flush() == 2 and store.pending_count == 0

    def test_p0_append_oserror_normalized(self, store, monkeypatch):
        """P0 即写块 OSError → StoreError（readback 前就断掉），fsync 计数不动。"""
        seg = _mkseg("decision: keep /srv/x")
        err = OSError(28, "No space left on device")
        fake_os_open, fake_open = _fake_write_open(store.path, err)
        monkeypatch.setattr(os, "open", fake_os_open)
        monkeypatch.setattr(builtins, "open", fake_open)
        with pytest.raises(StoreError) as excinfo:
            store.append(seg)
        assert "io_error" in str(excinfo.value)
        assert store.fsynced_p0_count == 0
        assert store.by_hash == {}, "写失败 → 内存索引不得记录（fail-safe 回滚语义）"

    def test_existing_storeerror_paths_unchanged(self, store):
        """回归：arm_write_failure / readback 校验等既有 StoreError 语义不回归。"""
        store.arm_write_failure()
        with pytest.raises(StoreError):
            store.append(_mkseg("x"))
        with pytest.raises(StoreError):
            store.flush()
        store.disarm_write_failure()
        n = store.flush()
        assert n == 0 and store.pending_count == 0


# ─────────────────────────── 磁盘水位观测（纯观测、零清理）───────────────────────────
class TestDiskWaterLevelObservation:
    def test_log_water_check_silent_when_healthy(self, store, caplog):
        """健康盘（used < 90%）：写盘前例行观测静默——零日志污染。"""
        store.disk_water_level = lambda: (100 * 1024 ** 3, 0.20)
        with caplog.at_level(logging.INFO, "sc.store"):
            store.log_water_check("flush")
        assert "water-level" not in caplog.text, "健康态不得打水位行"

    def test_log_water_check_info_line_when_high(self, store, caplog):
        """used ≥ 90%：打一行 INFO（剩余空间 + 触发动作）——ops 监控 grep 面。"""
        store.disk_water_level = lambda: (2 * 1024 ** 3, 0.95)
        with caplog.at_level(logging.INFO, "sc.store"):
            store.log_water_check("p0-append")
        line = next(l for l in caplog.messages if "water-level" in l)
        assert "action=p0-append" in line and "free=" in line, "一行须含剩余空间 + 触发动作"
        assert _level_of(caplog, "water-level") == logging.INFO, "写盘前例行观测 = INFO"

    def test_log_water_warn_on_failure_always_logs(self, store, caplog):
        """写盘失败：恒打一行 WARNING（剩余空间 + 触发动作 + 错误），含 unknown 水位态。"""
        store.disk_water_level = lambda: (0, -1.0)   # statvfs 失败 → 水位未知
        exc = OSError(5, "Input/output error")
        with caplog.at_level(logging.WARNING, "sc.store"):
            store.log_water_warn("flush-failed", exc)
        line = next(l for l in caplog.messages if "water-level" in l)
        assert "action=flush-failed" in line and "unknown" in line and "Input/output error" in line

    def test_water_check_threshold_constant(self):
        """阈值面：≥ DISK_WATER_HIGH_PCT 才出 INFO（防满盘态日志风暴），纯观测零清理。"""
        assert DISK_WATER_HIGH_PCT == 0.90

    def test_append_triggers_pre_write_check(self, store, caplog):
        """P0 append 写盘前确实例行调水位观测（高水位时出 INFO 行——可观测性证明非死代码）。"""
        store.disk_water_level = lambda: (1 * 1024 ** 3, 0.95)
        with caplog.at_level(logging.INFO, "sc.store"):
            store.append(_mkseg("decision: keep /srv/y"))
        assert any("water-level" in l and "action=p0-append" in l for l in caplog.messages)
        assert store.fsynced_p0_count == 1, "水位观测不得改变写成功路径行为"


def _level_of(caplog, needle):
    """caplog 里含 needle 的记录的 levelno（取最低）。"""
    lv = [r.levelno for r in caplog.records if needle in r.getMessage()]
    return min(lv) if lv else -1


# ─────────────────────────── distill 收尾 flush 失败不上抛（异步面允许丢）──────────────
class TestDistill_FlushFailureLossy:
    def test_final_flush_failure_does_not_raise(self, store, aux):
        """R2 核心：distill 收尾 flush（步骤 7）失败 → 不上抛、事件留痕、pending 留内存。

        注入面 = 裸 OSError（EIO）——OSError 是 builtin（全模块树唯一类），不受
        test_hermes_shell importlib 双引擎面（第二份 sc.store 的 StoreError 异类）影响；
        且 full-disk 实证形态（tui_gateway EIO unhandledRejection）本就是 OSError 面。"""
        p0 = _mkseg("decision: keep path /srv/a")          # P0 → 即写即 fsync（durable）
        p1 = _mkseg("ctx block " * 60, tier="P1")          # P1 → lazy 批
        store.append(p1)                                    # 显式把 P1 送进 lazy 批
        assert store.pending_count == 1, "非平凡前置：P1 真在 lazy 批（flush 有货可写）"
        w = Window(1000, [WindowSeg(p0), WindowSeg(p1)])
        store.flush = lambda: (_ for _ in ()).throw(
            OSError(5, "Input/output error"))
        d = Distiller(w, store, aux).run()          # 不得抛（异步面丢批容忍）
        assert any(e.startswith("flush_failed:") for e in d.events), "flush 失败事件须留痕"
        assert d.anchor_manifest == [], "锚点只发射已 durable 段——pending 字节物理上无指针"
        assert store.pending_count == 1, "pending 批留内存可重试（失败不清批）"
        assert p0.segment_id in d.kept_p0_ids, "P0 常驻保留不受 flush 失败影响"

    def test_p0_gate_io_error_compress_failed_lossless(self, store, aux, monkeypatch):
        """gate① 双 catch 面：append 被直接注入裸 OSError（store 归一之外的防御层）
        → compress_failed 不上抛、窗口一个字节不动。"""
        p0 = _mkseg("llm fact " * 100, tier="P0", prov="model")   # 非结构、非 durable
        p2 = _mkseg("dump tail", tier="P2")
        store.append(p2)
        w = Window(100, [WindowSeg(p0), WindowSeg(p2)])
        monkeypatch.setattr(JsonlStore, "append",
                            lambda self, seg: (_ for _ in ()).throw(
                                OSError(28, "No space left on device")))
        d = Distiller(w, store, aux).run()
        assert d.status == "compress_failed"
        assert d.anchor_manifest == []
        assert all(s.form == "resident" for s in w.segs), "窗口一个字节不动（fail-safe）"


# ─────────────────────────── engine 壳面：compress / reset / end 全不炸宿主 ─────────────
class TestEngine_FullDiskGraceful:
    def test_compress_oserror_returns_messages_no_raise(self, tmp_path):
        """R2 验收①：OSError 注入（replay 读面裸 OSError）→ compress() 不抛，
        降级 WARNING + 水位行存在。"""
        e = _mkengine(tmp_path)
        e.on_session_start("s_disk")
        _append_user(e, "goal: ship v1")
        e.store.flush()

        # 注入 full-disk：_window_fresh 读路径抛裸 OSError（EIO 读面）
        def _raise_replay():
            raise OSError(5, "Input/output error")
        e.store.replay = _raise_replay

        captured, attached = _capture_logs(("engine", "sc.store"), logging.WARNING)
        try:
            msgs = [{"role": "user", "content": "继续"}]
            out = e.compress(msgs)          # 不得抛
            assert out == msgs, "fail-open：压缩失败 = 原样返回宿主消息面"
            assert e.compression_count == 1
            joined = "\n".join(r.getMessage() for r in captured)
            assert "compress degraded" in joined, "降级 WARNING 行必须存在"
            assert "water-level" in joined, "水位观测行必须存在（ops 监控 grep 面）"
        finally:
            _detach(attached)

    def test_compress_normalized_io_error_no_raise(self, tmp_path):
        """R2 验收②（主路径）：store 归一语义下（io_error → StoreError）compress 亦不炸，
        降级 WARNING + 水位观测行同出。"""
        e = _mkengine(tmp_path)
        e.on_session_start("s_disk2")
        _append_user(e, "goal: ship v2")
        e.store.flush()

        # 归一语义直注：replay 读面抛 io_error StoreError（store 写路径归一后的逃逸面）
        e.store.replay = lambda: (_ for _ in ()).throw(
            StoreError("io_error: [Errno 5] Input/output error"))
        captured, attached = _capture_logs(("engine", "sc.store"), logging.WARNING)
        try:
            out = e.compress([{"role": "user", "content": "继续"}])   # 不得抛
            assert out, "fail-open 返回非抛"
            joined = "\n".join(r.getMessage() for r in captured)
            assert "compress degraded" in joined
            assert "water-level" in joined, "归一异常面同样打水位观测行"
        finally:
            _detach(attached)

    def test_on_session_reset_oserror_no_raise(self, tmp_path):
        """R2 验收③：/new 交接面 flush 被注入 OSError → reset 不抛、重置流程走完、水位行存在。"""
        e = _mkengine(tmp_path)
        e.on_session_start("s_reset")
        _append_p1(e, "in-flight P1 推理")
        assert e.store.pending_count == 1

        e.store.flush = lambda: (_ for _ in ()).throw(OSError(5, "Input/output error"))
        captured, attached = _capture_logs(("engine",), logging.WARNING)
        water_calls = []
        e.store.log_water_warn = lambda action, exc: water_calls.append((action, str(exc)))

        try:
            e.last_prompt_tokens = 999
            e.compression_count = 3
            e.on_session_reset()          # 不得抛（full-disk 时 /new 不能崩）
            assert e.last_prompt_tokens == 0, "reset 实态字段重置集走完"
            assert e.compression_count == 0
            assert e._force_evict_mode is False
            assert e.store.pending_count == 1, "flush 失败 → pending 批留内存（未清批）"
            assert water_calls, "水位观测调用必须发生（log_water_warn 触发面）"
            assert water_calls[0][0] == "session-reset-flush-failed"
            assert any("session-reset flush failed" in r.getMessage() for r in captured), \
                "reset 降级 WARNING 行必须存在"
        finally:
            _detach(attached)

    def test_on_session_end_oserror_no_raise(self, tmp_path):
        """会话收尾面同口径：full-disk 时 on_session_end 不炸宿主。"""
        e = _mkengine(tmp_path)
        e.on_session_start("s_end")
        _append_p1(e, "tail P1")
        water_calls = []
        e.store.flush = lambda: (_ for _ in ()).throw(OSError(5, "Input/output error"))
        e.store.log_water_warn = lambda action, exc: water_calls.append(action)
        captured, attached = _capture_logs(("engine",), logging.WARNING)
        try:
            e.on_session_end("s_end", [])
            assert water_calls == ["session-end-flush-failed"]
            assert any("session-end flush failed" in r.getMessage() for r in captured)
        finally:
            _detach(attached)

    def test_water_level_reported_values(self, tmp_path, caplog):
        """水位值透传：观测行携带真实 statvfs 剩余空间（注入固定值免真实 FS 依赖）。"""
        e = _mkengine(tmp_path)
        e.on_session_start("s_wl")
        e.store.flush = lambda: (_ for _ in ()).throw(OSError(5, "Input/output error"))
        e.store.disk_water_level = lambda: (3 * 1024 ** 3, 0.97)
        with caplog.at_level(logging.WARNING, "sc.store"):
            e.on_session_end("s_wl", [])
        line = next((r for r in caplog.records if "water-level" in r.getMessage()), None)
        assert line is not None, "WARNING 水位行必须落日志"
        assert "free=3.0GiB" in line.getMessage() and "used=97.0%" in line.getMessage()


# ─────────────────────────── 真实 statvfs 盘满态直测（EIO 注入走 fd）──────────────
class TestStore_StatvfsRealDisk:
    def test_disk_water_level_shape(self, store):
        """真实 statvfs（不 monkeypatch）：返回 (int ≥0, float ≥0) 或 unknown (0, -1.0)。"""
        free_b, pct = store.disk_water_level()
        assert isinstance(free_b, int) and isinstance(pct, float)
        assert pct >= -1.0
        if pct >= 0:
            assert free_b >= 0

    def test_disk_water_level_oserror_returns_unknown(self, store, monkeypatch):
        """statvfs 自身失败（盘消失/只读挂载异常）→ (0, -1.0)，观测面绝不反向上抛。"""
        def _raise(*a, **k):
            raise OSError(5, "Input/output error")
        monkeypatch.setattr(os, "statvfs", _raise)
        assert store.disk_water_level() == (0, -1.0)
