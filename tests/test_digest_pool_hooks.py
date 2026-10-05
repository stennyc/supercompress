"""P4 digest sidecar 水池 · engine 两钩子测试面（t_026a17f8 验收 + t_b5bf5201 用例）。

权威 spec = plan doc P4 节（sidecar 池：jsonl 源层零改动、池=可删派生缓存、
replay 查池命中免重蒸、digest 完成原子回写、保险丝/版本失配整池作废、损坏零崩回落现状）。

引擎薄壳纪律的可测锚点：钩子①（replay 查池 → 内存态 digest_ref=done + digest_text/key_fact
回填）+ 钩子②（成功 digest 生成 → upsert 池）。本面验证引擎接线（不是池算法单测——池算法
在 t_49148aa2 / t_567e3b29 的 digest_pool 层已封；此面 = 壳两钩子端到端行为）。

覆盖（spec 全条）：
- 池命中跳过蒸馏：replay 后命中段 digest_ready（done 语义）+ digest_text 回填（非 head-80）；
  distill 不对其 ensure_digest（免重蒸）。
- 池损坏回落现状：池文件 JSON 损坏 → 钩子① fail-open 全 pending、jsonl 字节 diff=0、零崩溃。
- model_version 失配作废：池条目模型 ≠ 当前模型 → guard 整池作废 → 全 pending 重蒸。
- GC 保险丝（条目>段数 / >5MB）触发：guard trip → 作废重建。
- 原子写完整性 + 池写失败 fail-open（full-disk → PoolWriteError 不炸宿主收尾面）。
- jsonl 源层字节零动不变式（删池 / 损坏池 均 diff=0）。

跑法（repo 根）：python -m pytest tests/test_digest_pool_hooks.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine import SupercompressContextEngine
from sc import digest_pool as sc_digest_pool
from sc.schema import Segment


# ── 引擎构造 + 段面（隔离 tmp，镜像 test_session_store._mkengine 口径）─────────────────

def _mkengine(tmp_path: Path, context_length: int = 256000) -> SupercompressContextEngine:
    """独立隔离 engine（external_store_path → tmp，不碰全局 active.jsonl）。

    池 = 与 session jsonl 同目录同 sid 前缀（sc_digest_pool.pool_path_for_jsonl），
    故本面池落 <tmp>/<sid>.digests.json（external_store_path 布局，零 HERMES_HOME 依赖）。"""
    return SupercompressContextEngine(context_length=context_length,
                                      external_store_path=str(tmp_path))


def _user_p0(e: SupercompressContextEngine, text: str) -> Segment:
    """落一条 user P0 段（digest 生成候选面；non-structural，可逐出可蒸馏）。"""
    seg = Segment(kind="segment", tier="P0", provenance="user", original_text=text,
                  session_topic="", model_version="")
    return e.store.append(seg)


def _pool_path(e: SupercompressContextEngine) -> Path:
    return Path(sc_digest_pool.pool_path_for_jsonl(e.store.path))


def _seed_pool(e: SupercompressContextEngine, segs, mv: str = "") -> None:
    """直接造池条目（绕开钩子②，隔离测钩子①）：upsert 每个段。"""
    pool = sc_digest_pool.DigestPool(str(_pool_path(e)))
    for s in segs:
        pool.upsert(content_hash=s.content_hash, model_version=mv,
                    key_fact=f"kf:{s.seq}", digest_text=f"sum:{s.original_text[:40]}")


def _jsonl_bytes(e: SupercompressContextEngine) -> bytes:
    return Path(e.store.path).read_bytes()


def _mkwindow(e):
    from sc.distill import Window, WindowSeg
    return Window(e.context_length, [WindowSeg(s) for s in e.store.replay()])


# ── 钩子① 池命中跳过蒸馏 ─────────────────────────────────────────────────────────────

def test_replay_hit_hydrates_done_and_backfills(tmp_path):
    """池命中 + 版本匹配 → 内存态 digest_ready（done）+ digest_text/key_fact 回填（非 head-80）。

    这是 P4 核心目标：重启后命中段免重蒸、注入 L2 用池里存的摘要（不降级 head-80）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_hit")
    s1 = _user_p0(e, "deploy the auth service to prod cluster v2")
    s2 = _user_p0(e, "rotate the API keys for the billing gateway")
    _seed_pool(e, [s1, s2])                       # 池 = 两段的 digest 产物（当前模型 mv=""）
    jsonl_before = _jsonl_bytes(e)

    # 冷重开：同 sid 新实例（模拟 gateway 重启）→ 重放触发钩子① 查池
    e2 = _mkengine(tmp_path)
    e2.on_session_start("s_hit")                  # 幂等绑定 + 冷重开重放
    segs = e2.store.replay()
    e2._digest_pool_replay_hook(segs)

    hit = {s.content_hash: s for s in segs}
    h1 = hit[s1.content_hash]
    assert h1.digest_ready is True, "命中段应 done（免重蒸）"
    assert h1.digest_ref.startswith("ready:"), "done 语义须落 ready: 前缀"
    assert h1.digest_text.startswith("sum:"), "回填池存摘要（池标记前缀）"
    assert h1.digest_text != s1.original_text[:80], "回填的是池摘要，非 head-80 占位"
    assert h1.key_fact == "kf:0", "回填池存 key_fact"
    assert hit[s2.content_hash].digest_ready is True
    # jsonl 源层字节零动（done 是纯内存态）
    assert _jsonl_bytes(e2) == jsonl_before, "钩子① 不得触碰 jsonl 源层字节"


def test_replay_miss_keeps_pending_flow(tmp_path):
    """池未命中 → 现有 pending 流程不变（digest_ref 保持 pending，等待正常蒸馏）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_miss")
    _user_p0(e, "cold seg never digested to pool")
    segs = e.store.replay()
    e._digest_pool_replay_hook(segs)
    assert all(s.digest_ref == "pending" for s in segs), "未命中段保持 pending 现状"


def test_hit_segment_skips_distill_ensure_digest(tmp_path, aux):
    """命中段（digest_ready）被 distill 门排除——不再 ensure_digest（免重蒸的实际效果）。"""
    from sc.distill import Distiller
    e = _mkengine(tmp_path)
    e.on_session_start("s_skip")
    s = _user_p0(e, "big P0 " * 30)
    _seed_pool(e, [s])
    # 冷重开 + 钩子① 命中 → store.by_hash 活体段 digest_ready（replay 重建的对象，勿用旧局部 s）
    segs = e.store.replay()
    e._digest_pool_replay_hook(segs)
    target = e.store.by_hash[s.content_hash]
    assert target.digest_ready is True
    # distill 逐出该段时，门应判定 digest_ready → 不调 aux.ensure_digest（用命中的活体段建窗口）
    from sc.distill import Window, WindowSeg
    w = Window(e.context_length, [WindowSeg(t) for t in segs])
    Distiller(w, e.store, aux).run()
    # 命中段免重蒸：端点零 compression 任务调用（FakeEndpoint.calls 无 aux）
    assert not any(t == "auxiliary.compression" for (t, _x) in aux.main_endpoint.calls), "命中段不应再蒸"


# ── 钩子② digest 完成回写池 ───────────────────────────────────────────────────────────

def test_write_back_upserts_new_digest(tmp_path, aux):
    """digest 成功生成（digest_text 填充 + digest_ready）→ 钩子② 原子 upsert 入池。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_wb")
    s = _user_p0(e, "make a digest of this turn")
    # 直接让 aux 生成该段 digest（sync 路径，端点给 canned response）
    aux.end_turn()
    out = aux.ensure_digest(s)
    assert out is not None and s.digest_ready
    e._digest_pool_write_hook()
    pool = sc_digest_pool.DigestPool(str(_pool_path(e)))
    entry = pool.get(s.content_hash)
    assert entry is not None, "成功 digest 应回写池"
    assert entry["digest_text"] == s.digest_text and entry["digest_text"] != ""


def test_write_back_skips_transient_no_digest(tmp_path, aux):
    """transient（端点失败 → digest_text 保持 ""）段不入池（池只兜成功产物）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_tr")
    s = _user_p0(e, "transient fail seg")
    aux.main_endpoint.fail = True                        # 端点 fail（transient → digest_text 保持 ""）
    aux.end_turn()
    aux.ensure_digest(s)
    assert not s.digest_ready and s.digest_text == ""
    e._digest_pool_write_hook()
    assert sc_digest_pool.DigestPool(str(_pool_path(e))).get(s.content_hash) is None


def test_write_back_idempotent_no_rewrite(tmp_path, aux):
    """已入池条目不重复写盘（池即事实源去重）：连调两次，第二次零 upsert。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_idem")
    s = _user_p0(e, "idem seg")
    aux.end_turn()
    assert aux.ensure_digest(s) is not None
    e._digest_pool_write_hook()
    p1 = _pool_path(e).read_text()
    e._digest_pool_write_hook()                       # 第二次：全在池 → 早退，池字节不变
    assert _pool_path(e).read_text() == p1, "已入池条目不得重复写盘"


def test_write_back_full_disk_fail_open(tmp_path, aux, monkeypatch):
    """池写 OSError（ENOSPC）→ PoolWriteError fail-open：钩子② 不炸宿主收尾面（宽 catch 吞掉）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_fd")
    s = _user_p0(e, "disk full seg")
    aux.end_turn()
    assert aux.ensure_digest(s) is not None
    real_upsert = sc_digest_pool.DigestPool.upsert

    def boom(self, *a, **k):
        raise sc_digest_pool.PoolWriteError("io_error: [Errno 28] No space left on device")

    monkeypatch.setattr(sc_digest_pool.DigestPool, "upsert", boom)
    try:
        e._digest_pool_write_hook()                    # 不抛（fail-open 吞 PoolWriteError）
    finally:
        monkeypatch.undo()
    # 池未更新（写失败），宿主面零炸（钩子正常返回）
    assert sc_digest_pool.DigestPool(str(_pool_path(e))).get(s.content_hash) is None


# ── 池损坏 → 回落现状（jsonl diff=0、零崩溃）────────────────────────────────────────

def test_corrupted_pool_falls_back_to_pending(tmp_path):
    """池文件 JSON 损坏 → 钩子① fail-open 全 pending、jsonl 字节 diff=0、零崩溃（DoD2）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_corrupt")
    s = _user_p0(e, "seg for corrupted pool test")
    _pool_path(e).write_text("{ this is not valid json ][[", encoding="utf-8")
    jsonl_before = _jsonl_bytes(e)
    segs = e.store.replay()
    e._digest_pool_replay_hook(segs)                    # 不抛
    assert all(sg.digest_ref == "pending" for sg in segs), "损坏池 → 全 pending 回落现状"
    assert _jsonl_bytes(e) == jsonl_before, "损坏池不得触碰 jsonl 源层（diff=0）"


def test_missing_pool_is_noop(tmp_path):
    """池从未写过（文件缺失）→ 合法态，全 pending 现状，零崩溃。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_missing")
    s = _user_p0(e, "no pool yet")
    assert not _pool_path(e).exists()
    segs = e.store.replay()
    e._digest_pool_replay_hook(segs)                    # 不抛
    assert all(sg.digest_ref == "pending" for sg in segs)


# ── model_version 失配 → 整池作废重蒸 ────────────────────────────────────────────────

def test_model_version_mismatch_invalidates_pool(tmp_path):
    """池条目模型 ≠ 当前模型 → guard 整池作废（重蒸）：钩子① 命中也回落 pending（DoD3）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_mvm")
    s = _user_p0(e, "old model digest")
    _seed_pool(e, [s], mv="m-old")                       # 池由旧模型 m-old 生成
    e.update_model("m-new", 256000)                      # 换当前模型 → m-new ≠ m-old
    segs = e.store.replay()
    e._digest_pool_replay_hook(segs)
    live = e.store.by_hash[s.content_hash]
    assert live.digest_ref == "pending", "版本失配 → 整池作废，回落 pending 重蒸"
    # 池文件被作废删除（invalidate）
    assert not _pool_path(e).exists(), "版本失配应整池作废（删除池文件）"


def test_model_version_match_keeps_pool(tmp_path):
    """池条目模型 = 当前模型 → 不触发版本作废，命中正常水合（对照面）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_mvmatch")
    s = _user_p0(e, "same model digest")
    _seed_pool(e, [s], mv="m-x")
    e.update_model("m-x", 256000)                        # 当前模型 = 池条目模型
    segs = e.store.replay()
    e._digest_pool_replay_hook(segs)
    live = e.store.by_hash[s.content_hash]
    assert live.digest_ready, "版本匹配 → 命中正常水合 done"


# ── GC 保险丝（条目>段数 / >5MB）触发整池作废 ────────────────────────────────────────

def test_guard_entry_count_over_segments_trips(tmp_path):
    """保险丝：池条目数 > 会话段数 → guard trip → 整池作废（全 pending 重建）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_gc_entry")
    s = _user_p0(e, "one real segment only")
    _seed_pool(e, [s])
    # 造「条目 > 段数」：再塞 3 条幽灵条目（无对应段）
    pool = sc_digest_pool.DigestPool(str(_pool_path(e)))
    for i in range(3):
        pool.upsert(content_hash=f"ghost{i}", model_version="",
                    key_fact="g", digest_text="g")
    segs = e.store.replay()                              # 只有 1 段
    e._digest_pool_replay_hook(segs)
    assert s.digest_ref == "pending", "条目>段数保险丝 → 整池作废，回落 pending"
    assert not _pool_path(e).exists()


def test_guard_size_over_cap_trips(tmp_path):
    """保险丝：池 >5MB → guard trip → 整池作废。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_gc_size")
    s = _user_p0(e, "size trip seg")
    _pool_path(e).write_text(
        json.dumps({"version": 1, "entries": []}) + "x" * (5 * 1024 * 1024 + 1),
        encoding="utf-8")
    segs = e.store.replay()
    e._digest_pool_replay_hook(segs)
    assert s.digest_ref == "pending", "池>5MB 保险丝 → 整池作废"
    assert not _pool_path(e).exists()


# ── 原子写完整性 ─────────────────────────────────────────────────────────────────────

def test_pool_write_is_atomic_and_valid(tmp_path, aux):
    """钩子② 回写后池文件 = 合法 JSON（tmp+rename 原子写不产半写）+ 版本头 + 无 CRLF。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_atomic")
    s = _user_p0(e, "atomic write seg")
    aux.end_turn()
    assert aux.ensure_digest(s) is not None
    e._digest_pool_write_hook()
    raw = _pool_path(e).read_bytes()
    doc = json.loads(raw.decode("utf-8"))               # 原子写 → 恒合法 JSON
    assert doc["version"] == sc_digest_pool.POOL_VERSION
    assert s.content_hash in {en["content_hash"] for en in doc["entries"]}
    assert b"\r" not in raw, "池文件须 LF-only（禁 CRLF）"


def test_missing_and_corrupt_pool_leave_jsonl_untouched(tmp_path):
    """删池 / 损坏池 后 jsonl 源层字节 diff=0（spec 验收项，池=派生层删了不伤事实源）。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_diff")
    _user_p0(e, "stable source layer seg")
    jsonl_before = _jsonl_bytes(e)
    _pool_path(e).write_text("garbage", encoding="utf-8")
    e._digest_pool_replay_hook(e.store.replay())
    assert _jsonl_bytes(e) == jsonl_before, "损坏池：jsonl diff=0"
    _pool_path(e).unlink(missing_ok=True)
    e._digest_pool_replay_hook(e.store.replay())
    assert _jsonl_bytes(e) == jsonl_before, "删池：jsonl diff=0"


# ── 端到端：生成→重启→命中免重蒸（P4 主目标闭环）─────────────────────────────────────

def test_generate_then_restart_reuse_end_to_end(tmp_path, aux):
    """完整闭环：会话内生成 digest → 会话末回写池 → 新进程冷重开 replay 命中 → 免重蒸。

    这是 P4 消除『gateway 重启后全量重蒸馏（~50 次 aux 调用）』的直接可测面。"""
    e = _mkengine(tmp_path)
    e.on_session_start("s_e2e")
    s = _user_p0(e, "e2e restart reuse seg")
    aux.end_turn()
    assert aux.ensure_digest(s) is not None
    e.on_session_end("s_e2e", [])                        # 会话末回写池（钩子②）
    assert sc_digest_pool.DigestPool(str(_pool_path(e))).get(s.content_hash) is not None

    # 模拟重启：新实例冷重开
    e2 = _mkengine(tmp_path)
    e2.on_session_start("s_e2e")
    segs = e2.store.replay()
    e2._digest_pool_replay_hook(segs)
    hit = {x.content_hash: x for x in segs}[s.content_hash]
    assert hit.digest_ready and hit.digest_text, "重启后命中段 done + digest_text 直用（免重蒸）"
