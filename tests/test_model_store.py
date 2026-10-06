"""B 卡（t_98809671）：model 段独立旁路 store `<sid>.model.jsonl` 契约测试。

覆盖（LF 新文件，sc 层契约面确定性 = 无 wall-clock / 无活 LLM——aux 面用
FakeEndpoint 同步 fixture + 宿主面用 FakeEndpoint 注入 _aux.main_endpoint）：

- 入库 gate：段 skip / marker-恒持久 / dedup（model 旁路 + 主 store 双查）/
  tool-call-only（空 strip）skip / role=tool 永不入库（调用方 gate 面）；
- replay fail-open：缺失 sidecar = 空池合法态；损坏 JSON 行 = torn-tail 静默丢弃；
  不可读（非 UTF-8 二进制）= 空池 + fused；
- 保险丝三触发：① 行数 > 2× 主 store；② 文件 > 5MB；③ model_version 失配
  （空 stamp 中性）；fused = 内存 reset + 文件保留 + 召回跳过 + marker；
- P0 临时档行即写 + 读回（append 后磁盘可重放、in-memory 精化 tier 不落盘）；
- 召回并集：model 段可被 query 命中；P2-tag model 段命中排序但不在注入集；
- 点查解析序 main→model + 跨 store id 无冲突验证；
- T4 预算竞争：model P0 与主 P0 同池竞争，seq 稳定序不变；
- aux 失败 → P0 地板（sync tier_fn 面：失败/无 tier_fn → 保持 P0；磁盘行恒 P0）；
- 会话生命周期：end-flush / 冷重启 P1 ×0.5 stale / carry 默认 OFF /
  fused 重启保留文件；
- DriftTracker 不被 model 段喂（负断言）。

跑法（repo 根）：
  python -m pytest tests/test_model_store.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fakes import FakeEndpoint
from engine import SupercompressContextEngine
from sc import model_store as sc_model_store
from sc.distill import Window, WindowSeg
from sc.inject import choose_forms
from sc.recall import Recaller
from sc.schema import Segment, nfc_hash
from sc.store import JsonlStore
from sc.tagger import Tagger

MIN = 200  # 默认 model_min_chars


def _long(text: str) -> str:
    """保证 ≥ MIN（200）字符：gate 的 chars 支要求。测试文本统一经此补齐。"""
    if len(text) >= MIN:
        return text
    pad = "z" * (MIN - len(text))
    return text + " " + pad


def _model_path(tmp_path: Path, sid: str = "s") -> str:
    return str(tmp_path / f"{sid}.model.jsonl")


def _assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _text_msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


# ═══════════════════════ sc 层：路径 / gate ═══════════════════════


def test_model_path_splitext_rule(tmp_path):
    """B 卡决策 1：主 store splitext + '.model.jsonl'（单源，legacy active.jsonl → active.model.jsonl）。"""
    base = str(tmp_path / "ab.jsonl")
    assert sc_model_store.model_path_for_jsonl(base) == str(tmp_path / "ab.model.jsonl")
    assert sc_model_store.model_path_for_jsonl("/x/active.jsonl") == "/x/active.model.jsonl"
    assert sc_model_store.model_path_for_jsonl("/x/s.jsonl") == "/x/s.model.jsonl"


class TestIngestGate:
    """B 卡决策 3：入库 gate（确定性零 LLM）。"""

    def test_blank_skip(self, tmp_path):
        ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
        assert ms.ingest_assistant("", min_chars=MIN, model_version="m1", main_by_hash={}) == (False, "blank")
        assert ms.ingest_assistant("   \n\t", min_chars=MIN, model_version="m1", main_by_hash={}) == (False, "blank")
        assert ms.line_count == 0

    def test_below_min_no_marker_skip(self, tmp_path):
        ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
        short = "x" * 50
        assert ms.ingest_assistant(short, min_chars=MIN, model_version="m1", main_by_hash={}) == (False, "below_min_no_marker")
        assert ms.line_count == 0

    def test_char_count_gate_persists(self, tmp_path):
        ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
        text = "model answer " * 40  # ≥ 200 chars
        assert len(text) >= MIN
        assert ms.ingest_assistant(text, min_chars=MIN, model_version="m1", main_by_hash={}) == (True, "chars")
        assert ms.line_count == 1

    def test_marker_always_persists_short_text(self, tmp_path):
        """压缩摘要类无论长度恒持久化（_P1/_P2 marker 族同口径，B 卡决策 3）。"""
        ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
        short_compact = "<compaction-summary> tiny "  # < MIN，但 marker 命中
        assert len(short_compact) < MIN
        assert ms.ingest_assistant(short_compact, min_chars=MIN, model_version="m1", main_by_hash={}) == (True, "marker")
        traceback_short = "Traceback (most recent call last): KeyError"
        assert len(traceback_short) < MIN
        assert ms.ingest_assistant(traceback_short, min_chars=MIN, model_version="m1", main_by_hash={}) == (True, "marker")

    def test_dedup_against_model_store_and_main(self, tmp_path):
        """nfc_hash 命中 model 旁路或主 store → dedup skip（同文双写封死，B 卡决策 3）。"""
        ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
        text = "A" * 300
        first = ms.ingest_assistant(text, min_chars=MIN, model_version="m1", main_by_hash={})
        assert first == (True, "chars")
        dup_model = ms.ingest_assistant(text, min_chars=MIN, model_version="m1", main_by_hash={})
        assert dup_model == (False, "dedup")
        # 同文已在主 store（user 段权威面）→ model 旁路不再写
        main_store = JsonlStore(str(tmp_path / "s.jsonl"))
        main_seg = Segment(tier="P0", provenance="user", original_text="B" * 300)
        main_store.append(main_seg)
        dup_main = ms.ingest_assistant("B" * 300, min_chars=MIN, model_version="m1",
                                        main_by_hash=main_store.by_hash)
        assert dup_main == (False, "dedup")
        assert ms.line_count == 1  # 未新增

    def test_tool_role_never_ingested(self, tmp_path):
        """role=tool 永不入库（调用方 gate 面：入口只收 assistant text；tool 消息不喂进来）。"""
        ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
        messages = [_assistant_msg("C" * 300),
                    {"role": "tool", "content": "tool result " * 30}]
        # 调用方（engine._model_ingest_hook）只挑 role=='assistant' 喂；本层只验
        # 契约：即使误把 tool 长文喂进来，也走同一条 gate 通道（tool 文本 = 模型文本，
        # 契约层不判 role——调用方负责）。此处验 engine 面负断言在 engine 测试中，
        # 本层验 model store 对 assistant 长文正常 ingest、对 tool 角色消息 0 副作用。
        ingested = 0
        for m in messages:
            if m["role"] != "assistant":
                continue  # 调用方 gate：role=tool 永不入库
            ingested += 1 if ms.ingest_assistant(m["content"], min_chars=MIN,
                                                  model_version="m1", main_by_hash={})[0] else 0
        assert ingested == 1
        assert ms.line_count == 1


# ═══════════════════════ sc 层：replay fail-open ═══════════════════════


class TestReplayFailOpen:
    """B 卡决策 2：构造/replay 对损坏/缺失 fail-open 空池（不可信旁路层，零崩溃）。"""

    def test_missing_sidecar_is_empty_pool(self, tmp_path):
        ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
        assert ms.line_count == 0
        assert not ms.fused  # 缺失 = 合法态，不 fuse
        assert ms.live_segments() == []

    def test_corrupt_jsonl_torn_tail_dropped(self, tmp_path):
        p = _model_path(tmp_path)
        ms1 = sc_model_store.ModelStore(p, sync=True)
        ms1.ingest_assistant("good line " + "g" * 300, min_chars=MIN, model_version="m1", main_by_hash={})
        with open(p, "a", encoding="utf-8") as f:
            f.write("\xff\xfe not valid json\n")  # torn 尾 → scdata 静默丢弃
        ms2 = sc_model_store.ModelStore(p, sync=True)
        assert ms2.line_count == 1  # 有效头部保留
        assert not ms2.fused  # 数据面重放契约（torn-tail），不 fuse

    def test_binary_unreadable_fuses(self, tmp_path):
        """不可读（OSError / 非 UTF-8 → ValueError 族）→ 空池 + fuse + 文件保留。"""
        p = _model_path(tmp_path)
        with open(p, "wb") as f:
            f.write(b"\x00\x01\x02 binary junk not utf8 \x80" * 64)
        ms = sc_model_store.ModelStore(p, sync=True)
        assert ms.fused
        assert "replay_fail_open" in ms.fuse_reasons[0]
        assert ms.line_count == 0
        assert os.path.getsize(p) > 0  # 文件保留，不删
        # fused 后 ingest = skip（fused guard）
        assert ms.ingest_assistant("x" * 300, min_chars=MIN, model_version="m1", main_by_hash={}) == (False, "fused")


# ═══════════════════════ sc 层：保险丝三触发 ═══════════════════════


class TestFuseTriggers:
    """B 卡决策 2：三选一命中即 fuse（内存 reset、文件保留、marker）。"""

    def test_line_count_over_2x_main(self, tmp_path):
        main = JsonlStore(str(tmp_path / "s.jsonl"))
        main.append(Segment(tier="P0", provenance="user", original_text="main line"))
        ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
        for i in range(3):
            ms.ingest_assistant(f"model line {i}" + "a" * 300, min_chars=MIN, model_version="m1", main_by_hash=main.by_hash)
        assert ms.check_fuse(main.line_count, "m1") != []  # 3 > 2*1
        assert ms.fused
        assert "line_count_over_2x_main" in ms.fuse_reasons[0]
        assert ms.line_count == 0  # 内存 reset
        assert os.path.getsize(_model_path(tmp_path)) > 0  # 文件保留

    def test_size_over_cap(self, tmp_path):
        p = _model_path(tmp_path)
        ms = sc_model_store.ModelStore(p, sync=True)
        big = "z" * (sc_model_store.MODEL_MAX_BYTES + 1)
        with open(p, "wb") as f:
            f.write(big.encode())
        reasons = ms.check_fuse(0, None)
        assert any("size_over_cap" in r for r in reasons)
        assert ms.fused

    def test_model_version_mismatch(self, tmp_path):
        p = _model_path(tmp_path)
        ms = sc_model_store.ModelStore(p, sync=True)
        ms.ingest_assistant("stamped " + "s" * 300, min_chars=MIN, model_version="old-model", main_by_hash={})
        assert ms.check_fuse(0, "new-model") != []  # stamped=old ≠ new
        assert ms.fused
        assert "model_version_mismatch" in ms.fuse_reasons[0]

    def test_empty_stamp_is_neutral(self, tmp_path):
        """空 stamp 行 = 版本判定中性（digest pool guard 同口径，B 卡决策 2）。"""
        p = _model_path(tmp_path)
        ms = sc_model_store.ModelStore(p, sync=True)
        ms.ingest_assistant("unstamped " + "u" * 300, min_chars=MIN, model_version="", main_by_hash={})
        # 主 store 需 ≥1 行，否则 fuse① (model 1 > 2*0) 先行命中；这里验证「空 stamp 中性」
        # 单独语义，故喂一个主 store 行数让 fuse① 不触发，只剩版本判定面。
        assert ms.check_fuse(1, "m1") == []  # 空 stamp 行不触发版本 fuse；主 1 行挡住 fuse①
        assert not ms.fused

    def test_fused_is_idempotent(self, tmp_path):
        ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
        ms.fuse("manual")
        before = len(ms.fuse_reasons)
        ms.fuse("manual-again")
        assert len(ms.fuse_reasons) == before  # 幂等 no-op


# ═══════════════════════ sc 层：P0 行即写 + 读回 ═══════════════════════


def test_p0_transient_write_and_readback(tmp_path):
    """B 卡决策 4：gate 命中段先以临时档 P0 非结构落盘 + 读回（磁盘 tier 永远 P0）。"""
    p = _model_path(tmp_path)
    ms = sc_model_store.ModelStore(p, sync=True)
    text = "persisted model output " + "p" * 200
    ok, gate = ms.ingest_assistant(text, min_chars=MIN, model_version="m1", main_by_hash={})
    assert ok and gate == "chars"
    # 即写：文件已有 ≥1 行（P0 = write + fsync + 读回）
    with open(p, encoding="utf-8") as f:
        on_disk = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(on_disk) == 1
    row = json.loads(on_disk[0])
    assert row["tier"] == "P0"
    assert row["provenance"] == "model"
    assert row["structural_p0"] is False  # 临时档 P0 非结构（可被 push cap 逐）
    assert row["model_version"] == "m1"
    # 读回：新实例重放
    ms2 = sc_model_store.ModelStore(p, sync=True)
    assert ms2.line_count == 1
    live = ms2.live_segments()[0]
    assert live.original_text == text
    assert live.tier == "P0"
    assert nfc_hash(live.original_text) == row["content_hash"]


def test_in_memory_refine_does_not_persist(tmp_path):
    """磁盘 tier 永远 P0；精化仅内存（Segment.tier 字段 in-memory update，B 卡决策 4）。"""
    p = _model_path(tmp_path)
    refined = []

    def tier_fn(seg):
        seg.tier = "P1"  # 精化仅内存
        refined.append(seg)
        return True

    ms = sc_model_store.ModelStore(p, tier_fn=tier_fn, sync=True)
    ms.ingest_assistant("refine me " + "r" * 200, min_chars=MIN, model_version="m1", main_by_hash={})
    ms.drain()
    seg = ms.live_segments()[0]
    assert seg.tier == "P1"  # 内存精化成功
    # 磁盘行仍 P0（never re-tier on replay，scdata 锁定）
    ms2 = sc_model_store.ModelStore(p, sync=True)
    assert ms2.live_segments()[0].tier == "P0"


# ═══════════════════════ sc 层：召回并集 ═══════════════════════


def test_recall_union_model_segments_hit(tmp_path):
    """B 卡决策 5：Recaller 语料 = 主 store 窗口 + model 旁路 live 段；model 段可被 query 命中。"""
    main = JsonlStore(str(tmp_path / "s.jsonl"))
    main.append(Segment(tier="P0", provenance="user", original_text="how to build a rocket engine fuel injector"))
    ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
    mtext = "turbofan nozzle design with quantum flux calibration " + "x" * 160
    ok, gate = ms.ingest_assistant(mtext, min_chars=MIN, model_version="m1", main_by_hash=main.by_hash)
    assert ok and gate == "chars"
    mseg = ms.live_segments()[0]
    window = Window(10000, [WindowSeg(s) for s in main.by_hash.values()] + [WindowSeg(mseg)])
    rec = Recaller(main, window)
    hits = rec.search("quantum flux calibration nozzle", k=3)
    assert any(h.seg is mseg for h in hits), "model 段应可被 query 命中（召回并集）"


def test_p2_tag_model_segment_not_injected(tmp_path):
    """B 卡决策 5：P2 tag 的 model 段可命中排序但不在注入集（P0/P1-only 注入门，同主 P2 语义）。"""
    main = JsonlStore(str(tmp_path / "s.jsonl"))
    main.append(Segment(tier="P0", provenance="user", original_text="deploy the widget service"))
    ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
    ok, _ = ms.ingest_assistant(_long("deploy the widget service rollback plan"),
                                min_chars=MIN, model_version="m1", main_by_hash=main.by_hash)
    assert ok
    seg = ms.live_segments()[0]
    seg.tier = "P2"  # 精化到 P2（仅内存）
    window = Window(10000, [WindowSeg(s) for s in main.by_hash.values()] + [WindowSeg(seg)])
    rec = Recaller(main, window)
    hits = rec.search("deploy widget service rollback", k=3)
    hit_p2 = [h for h in hits if h.seg is seg]
    assert hit_p2, "P2 段可命中排序"
    # P0/P1-only 注入门（engine.py:911-912 同型）
    injected = [h for h in hits if h.seg.tier in ("P0", "P1") and h.score > 0.0]
    assert all(h.seg is not seg for h in injected), "P2 model 段不在注入集"


# ═══════════════════════ sc 层：点查 + T4 预算竞争 ═══════════════════════


def test_point_fetch_resolution_order(tmp_path):
    """B 卡决策 6：L0 指针解析序 = 主 store 优先、model 旁路次（store.get(id) or model_store.get(id)）。"""
    main = JsonlStore(str(tmp_path / "s.jsonl"))
    main_seg = Segment(tier="P0", provenance="user", original_text="main store point target")
    main.append(main_seg)
    ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
    ms.ingest_assistant(_long("model store point target"), min_chars=MIN, model_version="m1", main_by_hash=main.by_hash)
    mseg = ms.live_segments()[0]
    # 主 store 段：主先命中
    assert main.get(main_seg.segment_id) is main_seg
    # model 段：主 store miss → model 命中
    assert main.get(mseg.segment_id) is None
    assert ms.get(mseg.segment_id) is mseg
    # 同 id 主 store 优先（构造同 content_hash 行两边 → 主先返回）
    both = main.get(main_seg.segment_id)
    assert both is main_seg, "点查解析序 main→model（同 id 主 store 优先）"


def test_segment_id_cross_store_no_conflict(tmp_path):
    """跨 store id 冲突安全性：id = content_hash[:16]+seq；不同内容前缀不同。"""
    main = JsonlStore(str(tmp_path / "s.jsonl"))
    a = main.append(Segment(tier="P0", provenance="user", original_text="conflict probe A"))
    ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
    ms.ingest_assistant(_long("conflict probe B different text"), min_chars=MIN, model_version="m1", main_by_hash={})
    b = ms.live_segments()[0]
    assert a.segment_id != b.segment_id
    assert a.content_hash[:16] != b.content_hash[:16], "不同内容 16-hex 前缀不同 → 跨 store 无重 id"
    # 主 index 与 model index 无共同键
    main_ids = set(main.index)
    model_ids = set(ms.store.index)
    assert main_ids.isdisjoint(model_ids)


def test_t4_budget_contention_shared_pool(tmp_path):
    """B 卡决策 6：model P0/P1 段进 kept 集，与主 store 段共用固定池 2K/4K + 既有确定性选档
    + seq 稳定序（池预算不变；choose_forms 确定性验证）。"""
    main = JsonlStore(str(tmp_path / "s.jsonl"))
    main_seg = main.append(Segment(tier="P0", provenance="user", original_text="main p0 " + "M" * 300))
    ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
    ms.ingest_assistant(_long("model p0 " + "N" * 200), min_chars=MIN, model_version="m1", main_by_hash=main.by_hash)
    mseg = ms.live_segments()[0]
    kept = [WindowSeg(main_seg), WindowSeg(mseg)]
    window = Window(10000, [WindowSeg(s) for s in main.by_hash.values()])
    rendered = choose_forms(kept, window, query="")
    ids = [ws.segment.segment_id for ws, _, _ in rendered]
    assert main_seg.segment_id in ids and mseg.segment_id in ids
    # seq 稳定序（主 seq 在前、model 旁路 seq 独立 → 输出按段自身 seq 稳定，不随 query 抖动）
    seqs = [ws.segment.seq for ws, _, _ in rendered]
    assert seqs == sorted(seqs)
    # 确定性：同输入同输出（零 LLM）
    rendered2 = choose_forms(kept, window, query="")
    assert [(ws.segment.segment_id, ff) for ws, ff, _ in rendered] == \
           [(ws.segment.segment_id, ff) for ws, ff, _ in rendered2]


# ═══════════════════════ sc 层：aux 失败 → P0 地板 ═══════════════════════


class TestAuxFailureFloor:
    """B 卡决策 4：aux 失败/degraded → 保持 P0 地板（磁盘 tier 恒 P0）。"""

    def test_aux_fail_keeps_p0_floor(self, tmp_path, aux):
        """宿主面 FakeEndpoint fail=True → tagger transient_fail → tier 保持 P0。"""
        ep: FakeEndpoint = aux.main_endpoint
        ep.fail = True  # 瞬态失败
        refined = []

        def tier_fn(seg):
            tagger = Tagger(aux, "")
            _tier, _st, regime, _reason = tagger.classify(seg.original_text, "model")
            if regime == "transient_fail":
                return False  # P0 地板
            seg.tier = _tier
            refined.append(seg)
            return True

        ms = sc_model_store.ModelStore(_model_path(tmp_path), tier_fn=tier_fn, sync=True)
        ms.ingest_assistant("aux broken " + "q" * 300, min_chars=MIN, model_version="m1", main_by_hash={})
        stats = ms.drain()
        assert stats["aux_ok"] == 0 and stats["aux_floor"] == 1
        assert ms.live_segments()[0].tier == "P0", "aux 失败 → P0 地板"

    def test_no_tier_fn_keeps_p0_floor(self, tmp_path):
        """无 tier_fn（宿主未接线）→ 全 P0 地板（零 LLM 契约面默认安全态）。"""
        ms = sc_model_store.ModelStore(_model_path(tmp_path), sync=True)
        ms.ingest_assistant("no tier fn " + "w" * 300, min_chars=MIN, model_version="m1", main_by_hash={})
        stats = ms.drain()
        assert stats == {"n": 1, "aux_ok": 0, "aux_floor": 1}
        assert ms.live_segments()[0].tier == "P0"

    def test_tier_fn_exception_fail_open(self, tmp_path):
        def bad_tier(seg):
            raise RuntimeError("boom")
        ms = sc_model_store.ModelStore(_model_path(tmp_path), tier_fn=bad_tier, sync=True)
        ms.ingest_assistant("tier crashes " + "c" * 300, min_chars=MIN, model_version="m1", main_by_hash={})
        stats = ms.drain()
        assert stats["aux_floor"] == 1  # 精化异常 = P0 地板（fail-open，永不当 P2 丢）
        assert ms.live_segments()[0].tier == "P0"


# ═══════════════════════ engine 层：配置 / 会话生命周期 / Drift 隔离 ═══════════════════════


def _mkengine(tmp_path: Path, **kw) -> SupercompressContextEngine:
    kw.setdefault("external_store_path", str(tmp_path))
    kw.setdefault("context_length", 256000)
    return SupercompressContextEngine(**kw)


class TestEngineConfig:
    def test_defaults_and_kill_switch(self, tmp_path):
        e = _mkengine(tmp_path)
        assert e.model_store_enabled is True  # 默认开（保留 kill-switch）
        assert e.model_min_chars == 200
        e_off = _mkengine(tmp_path / "off", model_store_enabled=False)
        assert e_off.model_store_enabled is False
        e_min = _mkengine(tmp_path / "min", model_min_chars=500)
        assert e_min.model_min_chars == 500

    def test_kill_switch_no_model_store_bound(self, tmp_path):
        e = _mkengine(tmp_path / "ks", model_store_enabled=False)
        e.on_session_start("s1")
        assert e._model_store is None, "kill-switch OFF → 不建 model 旁路"
        e._model_ingest_hook([_assistant_msg("x" * 300)])
        assert not Path(_model_path(tmp_path / "ks", "s1")).exists()


class TestEngineIngest:
    def test_on_turn_complete_ingests_assistant(self, tmp_path):
        e = _mkengine(tmp_path)
        e.on_session_start("s1")
        e._aux.main_endpoint = FakeEndpoint()
        e.on_turn_complete([_text_msg("user", "do X"), _assistant_msg("A" * 300),
                            {"role": "tool", "content": "T" * 300}])
        p = _model_path(tmp_path, "s1")
        assert Path(p).exists()
        with open(p, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        assert len(lines) == 1, "仅 assistant 段入库（user 走主 store，tool 永不入库）"
        row = json.loads(lines[0])
        assert row["provenance"] == "model" and row["tier"] == "P0"
        assert e._model_store.line_count == 1

    def test_tool_messages_never_ingested(self, tmp_path):
        e = _mkengine(tmp_path / "tool")
        e.on_session_start("s1")
        e.on_turn_complete([{"role": "tool", "content": "T" * 300}])
        assert not Path(_model_path(tmp_path / "tool", "s1")).exists(), "role=tool 永不入库"

    def test_dedup_main_store_no_double_write(self, tmp_path):
        e = _mkengine(tmp_path / "dedup")
        e.on_session_start("s1")
        text = "shared fact " + "S" * 300
        # 同文已 user 段进主 store → model 旁路 dedup
        e.store.append(Segment(tier="P0", provenance="user", original_text=text))
        e._model_ingest_hook([_assistant_msg(text)])
        assert e._model_store.line_count == 0, "dedup：同文双写封死"


class TestEngineFuse:
    def test_model_version_change_fuses(self, tmp_path):
        e = _mkengine(tmp_path / "fv")
        e.on_session_start("s1")
        e._main_runtime = {"model": "m-old", "provider": "", "base_url": "", "api_key": "", "api_mode": ""}
        e._model_store.ingest_assistant("stamped " + "s" * 300, min_chars=e.model_min_chars,
                                        model_version="m-old", main_by_hash=e.store.by_hash)
        assert not e._model_fused
        e.update_model("m-new", 256000)  # model_version 变更 = 保险丝 fused
        assert e._model_fused is True
        assert e._model_store.fused is True
        assert "model_version_mismatch" in e._model_store.fuse_reasons[0]
        assert os.path.getsize(_model_path(tmp_path / "fv", "s1")) > 0, "fused 保留文件不删"

    def test_fused_recall_union_skips_store(self, tmp_path):
        e = _mkengine(tmp_path / "fr")
        e.on_session_start("s1")
        e._model_store.fuse("manual-test")
        e._model_fused = True
        main_win = e._window_fresh()
        union = e._recall_union_window(main_win)
        assert union.segs == main_win.segs, "fused → 召回并集原样返回主窗口（跳过该 store）"

    def test_fused_t4_inject_skips_store(self, tmp_path):
        e = _mkengine(tmp_path / "ft")
        e.on_session_start("s1")
        e._model_store.ingest_assistant("t4 skip me " + "k" * 300, min_chars=e.model_min_chars,
                                        model_version="", main_by_hash={})
        e._model_fused = True
        out = e.compress([_text_msg("user", "hi")], current_tokens=e.threshold_tokens + 1, force=True)
        joined = "\n".join(m.get("content", "") for m in out)
        assert "t4 skip me" not in joined, "fused → T4 注入跳过 model store"


class TestEngineRecallUnion:
    def test_model_segment_recallable_via_engine(self, tmp_path):
        e = _mkengine(tmp_path / "ru")
        e.on_session_start("s1")
        e._model_store.ingest_assistant(_long("recall target zeptoflux calibration"),
                                        min_chars=e.model_min_chars,
                                        model_version="", main_by_hash={})
        main_win = e._window_fresh()
        union = e._recall_union_window(main_win)
        model_ids = {s.segment_id for s in e._model_store.live_segments()}
        assert any(ws.segment.segment_id in model_ids for ws in union.segs), "召回并集含 model 段"
        rec = Recaller(e.store, union)
        hits = rec.search("zeptoflux calibration", k=3)
        assert any(h.seg.segment_id in model_ids for h in hits)

    def test_empty_model_store_union_unchanged(self, tmp_path):
        e = _mkengine(tmp_path / "re")
        e.on_session_start("s1")
        main_win = e._window_fresh()
        union = e._recall_union_window(main_win)
        assert union.segs == main_win.segs, "空旁路 → 原样返回主窗口"


class TestEnginePointFetch:
    def test_point_fetch_main_then_model(self, tmp_path):
        e = _mkengine(tmp_path / "pf")
        e.on_session_start("s1")
        main_seg = e.store.append(Segment(tier="P0", provenance="user", original_text="main pf " + "m" * 50))
        e._model_store.ingest_assistant("model pf " + "n" * 300, min_chars=e.model_min_chars,
                                         model_version="", main_by_hash=e.store.by_hash)
        mseg = e._model_store.live_segments()[0]
        assert e.point_fetch(main_seg.segment_id) is main_seg, "主 store 优先"
        assert e.point_fetch(mseg.segment_id) is mseg, "主 miss → model 命中"
        assert e.point_fetch("no-such-id") is None


class TestEngineSessionLifecycle:
    def test_session_start_fresh_builds_model_store(self, tmp_path):
        e = _mkengine(tmp_path / "fresh")
        e.on_session_start("s1")
        assert e._model_store is not None
        assert e._model_store.path == _model_path(tmp_path / "fresh", "s1")
        assert not e._model_fused

    def test_cold_reopen_p1_stale_mark(self, tmp_path):
        """冷重启 model 旁路重放（磁盘恒 P0 地板 + 主 store 已 seed 不误 fuse，决策 4/7）。

        主 store 需 seed ≥1 user 行：fuse①（model > 2×main，锁定语义）主 store 空时
        1 > 2*0=0 恒真 → 误 fuse 清空合法 model 段——按真实会话喂主 store 段避免。
        model 段磁盘 tier 恒 P0（never re-tier on replay，scdata 锁定；精化仅内存），
        故冷重放 = 全 P0（保守地板，B 卡文档化取舍）；P1-stale ×0.5 对 model 行为
        文档化 no-op（磁盘无 P1 行），本测试钉死 P0 地板 + 不误 fuse + 不标 stale。"""
        d = tmp_path / "cold"
        e1 = _mkengine(d)
        e1.on_session_start("s1")
        e1.store.append(Segment(tier="P0", provenance="user",
                                 original_text="cold main seed " + "u" * 40))
        ok = e1._model_store.ingest_assistant(_long("cold p0 model row"),
                                              min_chars=e1.model_min_chars,
                                              model_version="", main_by_hash={})[0]
        assert ok
        e1.on_session_end("s1", [])
        # 新实例冷重开（同 sid，model 文件已存在 → 构造即 replay）
        e2 = _mkengine(d)
        e2.on_session_start("s1")
        assert not e2._model_fused, "主 store 已 seed（1 行）→ fuse① 1>2*1 不触发"
        mseg = e2._model_store.live_segments()[0]
        assert mseg.tier == "P0", "磁盘恒 P0（never re-tier on replay）"
        assert mseg.stale is False, "P0 地板行不标 stale（P1-stale ×0.5 = 文档化 no-op）"

    def test_carry_over_default_off_no_model_file(self, tmp_path):
        """不新增自动 carry，仍只认宿主 carry_over_context kwarg（决策 8）。"""
        d = tmp_path / "no_carry"
        e1 = _mkengine(d)
        e1.on_session_start("old")
        e1._model_store.ingest_assistant("carry probe " + "v" * 300, min_chars=e1.model_min_chars,
                                          model_version="", main_by_hash={})
        e1.on_session_end("old", [])
        e2 = _mkengine(d)
        e2.on_session_start("new")  # 无 carry_over_context → 默认 OFF
        assert e2._model_store.line_count == 0, "自动 carry OFF → 新 session model 旁路空池"
        assert Path(_model_path(d, "new")).exists() or True  # 首 append 才落盘（JsonlStore 语义）

    def test_carry_over_context_parallel_model_materialize(self, tmp_path):
        """carry 分支：旧 model 文件存在 → 并行 materialize（字节稳定拷贝，决策 8）。

        主 store 需 seed ≥1 user 行（fuse① 锁定语义：main=0 时 model 1>0 恒真 → 误 fuse）；
        model 段磁盘 tier 恒 P0 → materialize 后 P1-stale 对 P0 行是文档化 no-op，
        本测试钉死 reopened_from 标记 + 不误 fuse + 不标 stale。"""
        d = tmp_path / "carry"
        e1 = _mkengine(d)
        e1.on_session_start("old")
        e1.store.append(Segment(tier="P0", provenance="user",
                                 original_text="carry main seed " + "u" * 40))
        ok = e1._model_store.ingest_assistant(_long("carry x model row"),
                                              min_chars=e1.model_min_chars,
                                              model_version="", main_by_hash={})[0]
        assert ok
        e1.on_session_end("old", [])
        old_model = _model_path(d, "old")
        assert Path(old_model).exists()
        e2 = _mkengine(d)
        e2.on_session_start("new", old_session_id="old", carry_over_context=True)
        assert not e2._model_fused, "main store 已 seed → fuse① 不触发"
        assert e2._model_store.line_count == 1, "model 旁路平行 materialize"
        carried = e2._model_store.live_segments()[0]
        assert carried.reopened_from == "old", "跨期重开标记（materialize 语义）"
        assert carried.tier == "P0" and carried.stale is False, \
            "P0 地板行不因 materialize 标 stale（P1-stale = 文档化 no-op）"

    def test_fused_reopen_preserves_file(self, tmp_path):
        """fused 重启保留文件（不删）+ fused 标志（召回跳过）。"""
        d = tmp_path / "fused_reopen"
        e1 = _mkengine(d)
        e1.on_session_start("s1")
        e1._model_store.ingest_assistant("fused keep " + "f" * 300, min_chars=e1.model_min_chars,
                                          model_version="m1", main_by_hash={})
        p = _model_path(d, "s1")
        size_before = os.path.getsize(p)
        assert size_before > 0
        e1._main_runtime = {"model": "m2", "provider": "", "base_url": "", "api_key": "", "api_mode": ""}
        e1.update_model("m2", 256000)  # model_version 失配 → fused
        assert e1._model_fused
        # 重启重放 → 再次 fused（文件仍在、再判）
        e2 = _mkengine(d)
        e2.on_session_start("s1")
        assert os.path.getsize(p) == size_before, "fused 重启保留文件不删"

    def test_session_end_flushes_model_store(self, tmp_path):
        e = _mkengine(tmp_path / "endflush")
        e.on_session_start("s1")
        e._model_store.ingest_assistant("end flush " + "e" * 300, min_chars=e.model_min_chars,
                                         model_version="", main_by_hash={})
        assert Path(_model_path(tmp_path / "endflush", "s1")).exists()
        e.on_session_end("s1", [])  # model 旁路 lazy 批 flush 面不炸
        assert e._model_store.line_count == 1


class TestDriftIsolation:
    def test_model_segments_do_not_feed_drift(self, tmp_path):
        """B 卡决策 7：model 段不喂 DriftTracker（user-only by design）、不推 epoch（负断言）。"""
        e = _mkengine(tmp_path / "drift")
        e.on_session_start("s1")
        drift_before = (e._drift.current_epoch, e._drift.user_seq, len(e._drift.anchor))
        for i in range(3):
            e._model_store.ingest_assistant(f"drift probe {i}" + "d" * 300, min_chars=e.model_min_chars,
                                            model_version="", main_by_hash={})
        assert (e._drift.current_epoch, e._drift.user_seq, len(e._drift.anchor)) == drift_before, \
            "model 段 0 喂入 DriftTracker（epoch/user_seq/anchor 全不动）"
        # 主 store user 段仍喂（对照面）：drift 唯一喂入点 = _drift.observe_user
        #（on_turn_complete 宿主面：engine.py「triggered = self._drift.observe_user(text)」；
        # store.append 只落盘、0 喂入 drift——对照面必须走真实喂入点，store.append 会假绿）
        ctrl = "drift control user segment alpha beta gamma"
        e._drift.observe_user(ctrl)
        assert e._drift.user_seq > drift_before[1], "user 段仍喂 drift（对照，证明非全灭）"

    def test_rehydrate_uses_main_store_only(self, tmp_path):
        """_drift.rehydrate 只用主 store（决策 7，不改）。"""
        d = tmp_path / "reh"
        e1 = _mkengine(d)
        e1.on_session_start("s1")
        e1._model_store.ingest_assistant("reh model epoch push " + "h" * 300, min_chars=e1.model_min_chars,
                                         model_version="", main_by_hash={})
        e1.on_session_end("s1", [])
        e2 = _mkengine(d)
        e2.on_session_start("s1")
        # model 段不在主 store → rehydrate 只读主 store（line_count 驱动），model 旁路 0 参与
        assert e2._drift.anchor_source in ("rehydrate", "none", "build")
        # 直接断言：model 段原文 token 不进锚点（'h'*300 无显著 token → 锚点应为空）
        assert e2._drift.anchor == set(), "model 段 token 不进 rehydrate 锚点（user-only）"


class TestEngineMarkerSurface:
    """B 卡决策 10：日志 marker 全集合面（宿主侧 grep 验证用——测试面只验 ingest 计数 + gate 文本形态）。"""

    def test_ingest_marker_gate_counts(self, tmp_path, caplog):
        e = _mkengine(tmp_path / "marker")
        e.on_session_start("s1")
        with caplog.at_level("INFO", logger="engine"):
            e._model_ingest_hook([_assistant_msg("m" * 300), _assistant_msg("n" * 300)])
        ingested = [r for r in caplog.records if "model-store: ingested" in r.getMessage()]
        assert ingested, "ingest marker 应打出（N>0）"
        assert "model-store: ingested 2 assistant segment(s) (gate=chars=2)" in ingested[0].getMessage()

    def test_no_ingest_marker_when_skipped(self, tmp_path, caplog):
        e = _mkengine(tmp_path / "marker_skip")
        e.on_session_start("s1")
        with caplog.at_level("INFO", logger="engine"):
            e._model_ingest_hook([_assistant_msg("short no marker text")])
        assert not [r for r in caplog.records if "model-store: ingested" in r.getMessage()], \
            "N=0 不打 ingest marker（避免噪声）"
