"""P4 digest sidecar 水池 · 层级（sc/digest_pool.py 池算法）pytest 验收面（t_b5bf5201 补面）。

父卡 t_026a17f8 交付的 tests/test_digest_pool_hooks.py 封住引擎两钩子接线面（16 用例）；
本面封住池算法自身的层级语义——GC 原语（per-session 跟删 / 孤儿清扫）、保险丝（条目>段数
& >5MB 整池作废）、model_version 失配整池作废、原子写完整性、损坏/缺失池零崩回落，以及
spec 验收项「jsonl 源层字节 diff=0（删池 / 损坏池 均不伤事实源）」。

跑法（repo 根）：python -m pytest tests/test_digest_pool_layer.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

from sc import digest_pool as dp
from sc.store import JsonlStore


# ── 原子写完整性（tmp + os.replace；合法 JSON + 版本头 + LF-only + 无 tmp 残留）─────

class TestAtomicWriteIntegrity:
    def test_upsert_writes_valid_json_with_version_header(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "s1.digests.json"))
        p.upsert("h1", "mv1", "kf1", "sum1", ts=1.0)
        doc = json.loads(Path(p.path).read_text(encoding="utf-8"))
        assert doc["version"] == dp.POOL_VERSION
        assert {"content_hash": "h1", "model_version": "mv1",
                "key_fact": "kf1", "digest_text": "sum1", "ts": 1.0} in doc["entries"]

    def test_file_is_lf_only_no_crlf(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "lf.digests.json"))
        p.upsert("h", "mv", "k", "d")
        assert b"\r" not in Path(p.path).read_bytes()

    def test_consecutive_upserts_leaves_no_tmp_residue(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "resid.digests.json"))
        for i in range(3):
            p.upsert(f"h{i}", "mv", "k", "d")
        assert not list(tmp_path.glob(".digests-*.tmp"))

    def test_upsert_same_hash_overwrites_not_duplicates(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "dup.digests.json"))
        p.upsert("h", "mv1", "old", "old")
        p.upsert("h", "mv2", "new", "new")
        by = p.load()
        assert len(by) == 1 and by["h"]["model_version"] == "mv2" and by["h"]["digest_text"] == "new"

    def test_write_back_corrupt_pool_recovers(self, tmp_path: Path) -> None:
        """半写/损坏池 → load fail-open {} → upsert 原子重写为合法池（恢复面）。"""
        p = dp.DigestPool(str(tmp_path / "recov.digests.json"))
        Path(p.path).write_text("{{{ corrupt garbage", encoding="utf-8")
        p.upsert("h1", "mv", "k", "d")
        assert p.get("h1", "mv") is not None
        json.loads(Path(p.path).read_text(encoding="utf-8"))  # 重写后恒合法


# ── 损坏/缺失池 → 零崩回落 + jsonl 源层字节 diff=0（spec 验收项）──────────────────

class TestCorruptAndMissingPool:
    def test_jsonl_zero_diff_when_pool_deleted(self, tmp_path: Path) -> None:
        """删池后 jsonl 源层字节须零动（diff=0）。池与 jsonl 同目录（GC 对应关系面）。

        用 P0 段（即写即 fsync）让 jsonl 文件实际存在；P1 默认 lazy 批不落盘。"""
        store = JsonlStore(str(tmp_path / "s.jsonl"))
        from helpers import mkseg
        mkseg(store, "fact A", tier="P0", prov="user")
        assert Path(store.path).exists(), "P0 段须即写即 fsync 落 jsonl"
        before = Path(store.path).read_bytes()
        pool = dp.DigestPool(str(tmp_path / "s.digests.json"))
        pool.upsert("hA", "mv", "k", "d")
        assert pool.clear() is True
        assert Path(store.path).read_bytes() == before, "删池后 jsonl 源层字节须零动（diff=0）"

    def test_jsonl_zero_diff_when_pool_corrupt(self, tmp_path: Path) -> None:
        store = JsonlStore(str(tmp_path / "s.jsonl"))
        from helpers import mkseg
        mkseg(store, "fact B", tier="P0", prov="user")
        before = Path(store.path).read_bytes()
        pool = dp.DigestPool(str(tmp_path / "s.digests.json"))
        Path(pool.path).write_text("!!! not json", encoding="utf-8")
        assert pool.load() == {}
        assert Path(store.path).read_bytes() == before, "损坏池不影响 jsonl 源层（diff=0）"

    def test_missing_pool_load_returns_empty(self, tmp_path: Path) -> None:
        assert dp.DigestPool(str(tmp_path / "never.digests.json")).load() == {}

    def test_corrupt_pool_load_returns_empty(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "corrupt.digests.json"))
        Path(p.path).write_text("{not valid json ]", encoding="utf-8")
        assert p.load() == {}

    def test_bad_entries_silently_dropped_not_poisoned(self, tmp_path: Path) -> None:
        """坏条目静默丢（无 content_hash / 非 dict 元素），合法条目保留，不连累整池。"""
        p = dp.DigestPool(str(tmp_path / "mixed.digests.json"))
        Path(p.path).write_text(
            json.dumps({"version": 1, "entries": [
                {"content_hash": "good", "model_version": "mv", "key_fact": "k",
                 "digest_text": "d", "ts": 1.0},
                "not-a-dict",
                {"no_hash_key": True},
                {"content_hash": "", "model_version": "mv"},
            ]}) + "\n",
            encoding="utf-8")
        loaded = p.load()
        assert set(loaded) == {"good"}


# ── model_version 失配 → 整池作废 ─────────────────────────────────────────────────

class TestModelVersionInvalidation:
    def test_mismatch_flag_true_when_any_entry_differs(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "mm.digests.json"))
        p.upsert("h1", "m-old", "k", "d")
        assert p.pool_model_version_mismatch("m-new") is True

    def test_match_no_mismatch(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "mm2.digests.json"))
        p.upsert("h1", "m-same", "k", "d")
        assert p.pool_model_version_mismatch("m-same") is False

    def test_empty_pool_no_mismatch(self, tmp_path: Path) -> None:
        """空池 = 无可作废 → False（不误触整池作废）。"""
        assert dp.DigestPool(str(tmp_path / "mm3.digests.json")).pool_model_version_mismatch("m") is False

    def test_entry_missing_model_version_treated_mismatch(self, tmp_path: Path) -> None:
        """条目缺 model_version（无法溯源生成模型）→ 安全方向判失配 True。"""
        p = dp.DigestPool(str(tmp_path / "mm4.digests.json"))
        Path(p.path).write_text(
            json.dumps({"version": 1, "entries": [
                {"content_hash": "h", "key_fact": "k", "digest_text": "d", "ts": 1.0}]}) + "\n",
            encoding="utf-8")
        assert p.pool_model_version_mismatch("m-new") is True

    def test_guard_trips_and_deletes_pool_on_version_mismatch(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "mm5.digests.json"))
        p.upsert("h1", "m-old", "k", "d")
        assert p.guard(model_version="m-new") is True
        assert not Path(p.path).exists(), "版本失配 → 整池作废（删池文件，全量重建）"

    def test_guard_healthy_version_no_trip(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "mm6.digests.json"))
        p.upsert("h1", "m-x", "k", "d")
        assert p.guard(model_version="m-x") is False
        assert Path(p.path).exists(), "版本健康 → 池保留"


# ── GC 保险丝（条目>段数 / >5MB）触发整池作废 ─────────────────────────────────────

class TestGCBreaker:
    def test_entry_count_over_segments_trips(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "br1.digests.json"))
        for i in range(5):
            p.upsert(f"h{i}", "mv", "k", "d")
        reasons = p.check_safety(segment_count=2)   # 5 条目 > 2 段
        assert any("entry_count_over_segments" in r for r in reasons)

    def test_entry_count_equal_segments_no_trip(self, tmp_path: Path) -> None:
        """边界：条目数 == 段数 → 不 trip（> 才是违规，spec 铁律是严格大于）。"""
        p = dp.DigestPool(str(tmp_path / "br1b.digests.json"))
        p.upsert("h0", "mv", "k", "d")
        p.upsert("h1", "mv", "k", "d")
        assert p.check_safety(segment_count=2) == []

    def test_size_over_cap_trips(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "br2.digests.json"))
        Path(p.path).write_text(json.dumps({"version": 1, "entries": []})
                                + "x" * (dp.POOL_MAX_BYTES + 1), encoding="utf-8")
        reasons = p.check_safety()
        assert any("size_over_cap" in r for r in reasons)

    def test_healthy_pool_no_trip(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "br3.digests.json"))
        p.upsert("h1", "mv", "k", "d")
        p.upsert("h2", "mv", "k", "d")
        assert p.check_safety(segment_count=2) == []

    def test_corrupt_large_pool_size_still_trips(self, tmp_path: Path) -> None:
        """半写/损坏大文件同样须作废——size 判量原始字节（不依赖可解析）。

        注意：check_safety(segment_count=None) 才隔离 size 单判；给 segment_count 时
        损坏池 load() fail-open 回 {} → 条目数=0 不误判（条目项由 healthy 面封）。"""
        p = dp.DigestPool(str(tmp_path / "br4.digests.json"))
        Path(p.path).write_text("garbage" * (dp.POOL_MAX_BYTES + 1), encoding="utf-8")
        reasons = p.check_safety()
        assert any("size_over_cap" in r for r in reasons)

    def test_guard_cascade_version_and_size(self, tmp_path: Path) -> None:
        """guard 多因级联：版本失配 + 超限（同池两面同时命中）→ 一次 trip 删池。

        大 digest_text 条目撑起 >5MB（size 面）且条目由 m-old 生成（版本面）——
        两面都真命中（非损坏回落的空池面）。"""
        p = dp.DigestPool(str(tmp_path / "br5.digests.json"))
        p.upsert("h1", "m-old", "k", "x" * (dp.POOL_MAX_BYTES + 1))
        assert p.guard(model_version="m-new") is True
        assert not Path(p.path).exists()

    def test_invalidate_idempotent(self, tmp_path: Path) -> None:
        p = dp.DigestPool(str(tmp_path / "br6.digests.json"))
        p.upsert("h", "mv", "k", "d")
        assert p.invalidate() is True
        assert p.invalidate() is False  # 池已不在 → False（幂等，零崩）


# ── GC 原语（per-session 跟删 + 孤儿清扫）──────────────────────────────────────

class TestGCPrimitives:
    def test_gc_session_pool_deletes_live_pool_and_is_idempotent(self, tmp_path: Path) -> None:
        hh = str(tmp_path)
        pool_file = dp.pool_path_for_session("s1", hermes_home=hh)
        dp.DigestPool(pool_file).upsert("h", "mv", "k", "d")
        assert dp.gc_session_pool("s1", hermes_home=hh) is True
        assert not Path(pool_file).exists()
        assert dp.gc_session_pool("s1", hermes_home=hh) is False  # 幂等：不在 → False

    def test_gc_session_pool_jsonl_source_byte_diff_zero(self, tmp_path: Path) -> None:
        """per-session GC 只删池、绝不触碰 jsonl 源层（diff=0 验收）。

        池与源层 jsonl 同 sid 前缀（gc_orphan_pools 的对应关系面）；GC 池路径 =
        <hh>/supercompress_store/<sid>.digests.json，源层 jsonl 与其同住该目录。"""
        hh = str(tmp_path)
        store_dir = Path(dp.pool_path_for_session("s1", hermes_home=hh)).parent
        store_dir.mkdir(parents=True, exist_ok=True)
        jsonl = store_dir / "s1.jsonl"
        jsonl.write_text('{"a": 1}\n', encoding="utf-8")
        dp.gc_session_pool("s1", hermes_home=hh)
        assert jsonl.read_bytes() == b'{"a": 1}\n', "GC 只删池文件，jsonl 源层 diff=0"

    def test_gc_orphan_pools_removes_only_orphans(self, tmp_path: Path) -> None:
        hh = str(tmp_path)
        store_dir = Path(dp.pool_path_for_session("alive", hermes_home=hh)).parent
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "alive.jsonl").write_text("{}", encoding="utf-8")  # 会话源仍在
        dp.DigestPool(dp.pool_path_for_session("alive", hermes_home=hh)).upsert("h", "mv", "k", "d")
        dp.DigestPool(dp.pool_path_for_session("gone", hermes_home=hh)).upsert("h", "mv", "k", "d")
        removed = dp.gc_orphan_pools(hermes_home=hh)
        assert any(x.endswith("gone.digests.json") for x in removed)
        assert not any(x.endswith("alive.digests.json") for x in removed)
        assert Path(dp.pool_path_for_session("alive", hermes_home=hh)).exists()
        assert (store_dir / "alive.jsonl").read_bytes() == b"{}"  # jsonl 源零动

    def test_gc_orphan_pools_second_sweep_and_missing_dir(self, tmp_path: Path) -> None:
        hh = str(tmp_path)
        dp.DigestPool(dp.pool_path_for_session("x", hermes_home=hh)).upsert("h", "mv", "k", "d")
        dp.gc_orphan_pools(hermes_home=hh)
        assert dp.gc_orphan_pools(hermes_home=hh) == []          # 二轮无孤儿
        assert dp.gc_orphan_pools(hermes_home=str(tmp_path / "nope")) == []  # 目录缺失 → []
