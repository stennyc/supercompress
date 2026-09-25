"""Data-layer self-test for scdata (replay / materialize / anchor closure / idempotency).

Covers the data-side slice of gate ⑤: single replay path, byte-stable materialization,
zero-dangling-anchors via verify_anchor_closure, and two-level idempotent keys.

Hermetic：工件写进 per-run temp dir（不写 cwd、不留 t_*.jsonl 残骸）——
同目录二次跑不再见旧行，8/8 结果对 commit 可复现（@reviewer 非 stale 纪律）。
"""
import json
import os
import shutil
import sys
import tempfile
import unicodedata

sys.path.insert(0, ".")
from scdata import (ReplayIndex, append_line, digest_regen_key, make_segment,
                    materialize, replay, sha256_nfc, to_line, verify_anchor_closure)

_WORKDIR = tempfile.mkdtemp(prefix="scdata_ref_")


def path(name):
    return os.path.join(_WORKDIR, f"t_{name}.jsonl")


def seg(i, text, **kw):
    return make_segment(i, text, **kw)


def test_replay_drops_torn_tail_and_rebuilds():
    p = path("replay")
    for i in range(5):
        append_line(p, seg(i, f"user: 工作目录是 C:\\project\\{i}", tier="P0",
                           provenance="user", structural_p0=True))
    # torn tail: a P1 batch that never fsync'd — first 3 lines valid, 4th truncated
    s4 = json.dumps(seg(4, "P1 intermediate reasoning about module deps", tier="P1"),
                    ensure_ascii=False)
    with open(p, "a", encoding="utf-8") as f:
        f.write(s4[: len(s4) // 2])  # torn
    idx = replay(p)
    assert len(idx) == 5, len(idx)                 # valid head fully recovered
    assert len(idx.dropped) == 1 and idx.dropped[0][1] == "json_parse_fail"
    assert idx.lookup(sha256_nfc("user: 工作目录是 C:\\project\\0")) is not None
    print("OK torn-tail: 5 recovered, 1 dropped (json_parse_fail), O(1) point lookup hit")


def test_replay_idempotent_dup_skip_no_retiering():
    p = path("dup")
    base = seg(1, "重复的工具输出: import os", tier="P2")
    append_line(p, base)
    append_line(p, base)  # same content_hash -> skipped, index keeps FIRST occurrence
    idx = replay(p)
    assert len(idx) == 1
    assert len(idx.dropped) == 1 and idx.dropped[0][1] == "dup_skip_idempotent"
    assert idx.by_hash[base["content_hash"]]["seq"] == 1
    print("OK idempotent dedup: second identical line skipped (no re-tiering), first kept")


def test_replay_hash_mismatch_dropped():
    p = path("badhash")
    s = seg(1, "original")
    s["content_hash"] = "deadbeef"  # tampered stored hash
    with open(p, "w", encoding="utf-8") as f:
        f.write(to_line(s) + "\n")
    idx = replay(p)
    assert len(idx) == 0 and idx.dropped[0][1] == "content_hash_mismatch"
    print("OK tamper: stored-hash mismatch detected and line dropped")


def test_materialize_byte_stable_hash_unchanged():
    src, act = path("arch"), path("newactive")
    for i in range(3):
        append_line(src, seg(i, f"ledger goal: {unicodedata.normalize('NFC','目标 '+str(i))}",
                             kind="ledger", tier="P0", structural_p0=True))
    before = [sha256_nfc(l) for l in open(src, encoding="utf-8").read().splitlines() if l.strip()]
    idx = materialize(src, act)
    after = [sha256_nfc(l) for l in open(act, encoding="utf-8").read().splitlines() if l.strip()]
    assert before == after, "materialization drifted bytes -> dedup would break"
    # stronger: every active-file line's content_hash equals the archive line's
    arch_lines = [json.loads(l) for l in open(src, encoding="utf-8").read().splitlines() if l.strip()]
    act_lines = [json.loads(l) for l in open(act, encoding="utf-8").read().splitlines() if l.strip()]
    assert [a["content_hash"] for a in arch_lines] == [a["content_hash"] for a in act_lines]
    print(f"OK materialize: {len(after)} lines byte-stable, content_hash identical, single replay path")


def test_anchor_closure_zero_dangling_and_structural():
    p = path("closure")
    fsynced = [seg(i, f"P0 decision {i}: 改用 jsonl 不引 DB", tier="P0", structural_p0=True,
                    digest_ref="ready:abc") for i in range(3)]
    for s in fsynced:
        append_line(p, s)
    idx = replay(p)
    manifest = [{"segment_id": s["content_hash"]} for s in fsynced]
    assert verify_anchor_closure(manifest, idx) == []
    print(f"OK closure: {len(manifest)} fsync+readback anchors all resolve to hash-verified lines -> []")


def test_anchor_closure_detects_dangling():
    p = path("dangling")
    append_line(p, seg(1, "P0 唯一错误: EPERM 读 C:\\x\\y.txt", tier="P0", structural_p0=True))
    idx = replay(p)
    ghost = sha256_nfc("从未落盘的 P0 指令")  # anchor emitted for a segment that never landed
    got = verify_anchor_closure([ghost], idx)
    assert got == [ghost], got
    print("OK closure: ghost anchor (no hash-verified line) -> dangling list non-empty")


def test_anchor_closure_empty_inputs_semantics():
    p = path("emptyman")
    append_line(p, seg(1, "P0 指令: 工作目录 C:\\x", tier="P0", structural_p0=True))
    idx = replay(p)  # non-empty index
    assert verify_anchor_closure([], idx) == []  # empty manifest on live index: legal, no anchors
    try:
        verify_anchor_closure([sha256_nfc("x")], ReplayIndex())
        raise AssertionError("empty replay_index must raise, not return [] silently")
    except ValueError as e:
        assert "did not run" in str(e)
    print("OK empty-input: empty manifest (live index) -> [] ; empty replay_index -> ValueError (no silent green)")


def test_digest_regen_key_idempotent():
    s = seg(1, "工具输出: 测试 3/7 通过", tier="P1")
    k1 = digest_regen_key(s, "m-1", "topic-pin")
    k2 = digest_regen_key(seg(1, "工具输出: 测试 3/7 通过"), "m-1", "topic-pin")
    k3 = digest_regen_key(s, "m-2", "topic-pin")
    assert k1 == k2 != k3  # pure in (content, model_version, topic); version bump = intentional regen
    print("OK digest key: same triple idempotent; model_version bump regens; topic pinned at startup")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} DATA-LAYER TESTS PASSED")
