"""预折数据面（切面 1 · PRD §4.1/§4.2）：6 槽 StubList schema + prefold.tmp 旁路文件。

owner 定稿（D1-D3/D7/D8/D10/D12，2026-10-28）——本测试面只钉数据面契约：
- 6 槽定形 + 顺序（D1）；
- 确定性 3 槽 vs LLM 3 槽 分叉（D2）；P0 段 LLM 3 槽恒 None = "确定性面"合法态（D3）；
- 槽值限长/项数兜底（LLM 自报字段防爆，同 parse_digest_response F5 口径）；
- 全空 fail-open（渲染回退纯编号锚点）；
- prefold.tmp 旁路：JSON 派生层 fail-open / 原子写 / 幂等 upsert / 清零（compact 后
  产物即失效，三处清零点同口径）；
- 旁路文件与 digest_pool 物理正交（独立文件名，不进 5MB guard / content_hash dedup 面
  —— 这里只钉"路径独立 + 独立文件"，guard/dedup 面属 digest_pool 既有契约零改动）。
"""
import json
import os

from sc.prefold import (
    StubList, StubListFile,
    STUB_LIST_SLOTS, DETERMINISTIC_SLOTS, LLM_SLOTS,
    SLOT_VALUE_MAX, SLOT_LIST_MAX_ITEMS,
    prefold_tmp_path_for_jsonl,
    prefold_evict_region,
    build_prefold_llm_prompt, parse_prefold_llm_response,
)


def test_slot_set_six_and_partition():
    assert len(STUB_LIST_SLOTS) == 6
    # D2：3 确定性 + 3 LLM 恰好覆盖 6 槽，互斥
    assert DETERMINISTIC_SLOTS | LLM_SLOTS == set(STUB_LIST_SLOTS)
    assert not (DETERMINISTIC_SLOTS & LLM_SLOTS)


def test_full_stub_roundtrip():
    sl = StubList(
        segment_id="abc",
        files_changed=["/a/b.py", "/c/d.py"],
        done="done thing",
        open=["task1", "task2"],
        user_decisions=["decision1"],
        why="root cause",
        next="next step",
    ).normalize()
    assert sl.is_empty() is False
    d = sl.to_dict()
    back = StubList.from_dict(d)
    assert back == sl


def test_p0_deterministic_llm_slots_none():
    # D3：P0 段 LLM 3 槽恒 None（"确定性面"合法态，非缺失错误）
    sl = StubList(segment_id="p0", files_changed=["/x"], p0_deterministic=True,
                  done="should-be-wiped", why="wipe", next="wipe")
    sl = sl.normalize()
    assert sl.done is None and sl.why is None and sl.next is None
    assert sl.files_changed == ["/x"]
    assert sl.is_empty() is False


def test_all_empty_fail_open_render():
    sl = StubList(segment_id="e").normalize()
    assert sl.is_empty() is True
    # 全空 → 渲染回退纯编号锚点（零膨胀）
    assert sl.render_line("#7 P0 [e] on-demand") == "#7 P0 [e] on-demand"


def test_slot_value_truncation():
    # 单串槽超 SLOT_VALUE_MAX → 截断 + '…'（LLM 自报字段限长兜底）
    long = "x" * (SLOT_VALUE_MAX + 50)
    sl = StubList(segment_id="t", done=long, why=long, next=long).normalize()
    assert sl.done and len(sl.done) == SLOT_VALUE_MAX + 1   # + '…'
    assert sl.done.endswith("…")


def test_slot_list_item_cap():
    # list 槽项数超 SLOT_LIST_MAX_ITEMS → 截断（防爆无界）
    sl = StubList(segment_id="c",
                  files_changed=[f"/p{i}" for i in range(SLOT_LIST_MAX_ITEMS + 5)]
                  ).normalize()
    assert len(sl.files_changed) == SLOT_LIST_MAX_ITEMS


def test_extract_files_changed_path_terms():
    # D2：files_changed 走 path_terms 路径词面（确定性，零 LLM）
    text = "edited /usr/lib/a.py and C:/srv/b.js, then /x/y/z.py again /usr/lib/a.py"
    got = StubList.extract_files_changed(text)
    assert got[0] == "/usr/lib/a.py"
    assert "/x/y/z.py" in got
    assert len(got) == 3                                   # 去重（/usr/lib/a.py 出现 2 次）


def test_render_line_shape():
    # D10/D12：锚点前置不变 + 6 槽尾追加（空槽省略；单串槽渲染时空格→下划线，
    # list 槽元素保留原字面直接 join）
    sl = StubList(segment_id="s", files_changed=["/a"], done="did x",
                  user_decisions=["pick b"]).normalize()
    line = sl.render_line("#3 P1 [s] on-demand")
    assert line.startswith("#3 P1 [s] on-demand · ")
    assert "files=[/a]" in line
    assert "done=did_x" in line
    assert "decisions=[pick b]" in line
    assert "open=" not in line and "why=" not in line and "next=" not in line   # 空槽省略


def test_prefold_path_independent_of_digest_pool():
    p = prefold_tmp_path_for_jsonl("/h/supercompress_store/abc.jsonl")
    assert p == "/h/supercompress_store/abc.prefold.tmp"
    assert ".digests.json" not in p                          # 与 durable 池物理正交


def test_stub_list_file_roundtrip(tmp_path):
    f = StubListFile(str(tmp_path / "abc.prefold.tmp"))
    assert f.load() == {}                                    # 缺失 → 空面（fail-open）
    sl = StubList(segment_id="m", files_changed=["/m"], why="w").normalize()
    f.upsert("m", sl)
    got = f.get("m")
    assert got == sl
    # 原子写落盘 + 文件内容可 JSON 直读
    assert json.loads(open(f.path).read())["m"]["files_changed"] == ["/m"]


def test_stub_list_file_corrupt_fallback(tmp_path):
    p = str(tmp_path / "x.prefold.tmp")
    open(p, "w").write("{not json")
    f = StubListFile(p)
    assert f.load() == {}                                     # 半写残留 → 当不存在（下事件重写）


def test_stub_list_file_upsert_many_counts_new(tmp_path):
    f = StubListFile(str(tmp_path / "y.prefold.tmp"))
    a = StubList(segment_id="a").normalize()
    b = StubList(segment_id="b").normalize()
    assert f.upsert_many({"a": a, "b": b}) == 2
    assert f.upsert_many({"a": a, "b": b}) == 0              # 幂等（已存 → 0 新）
    assert set(f.load()) == {"a", "b"}


def test_stub_list_file_clear(tmp_path):
    f = StubListFile(str(tmp_path / "z.prefold.tmp"))
    f.upsert("q", StubList(segment_id="q").normalize())
    assert f.clear() is True
    assert os.path.exists(f.path) is False
    assert f.clear() is False                                # 已无 → no-op 合法态


# ───────────── 触发面：蒸区划定（对齐 distill 选段语义）+ 逐段 LLM 预生成 prompt ─────────────
from sc.distill import Window, WindowSeg
from sc.schema import Segment
from sc.prefold import StubList as _SL


def _mkseg(seq, tier="P1", prov="model", text="x" * 100, digest_text="", structural=False,
           kind="segment"):
    s = Segment(kind=kind, tier=tier, provenance=prov, original_text=text,
                session_topic="t", model_version="m1",
                structural_p0=structural, goal=None, phase_id=None, state=None,
                open_tasks=[])
    s.seq = seq
    s.digest_ref = "ready:aa" if digest_text else "pending"
    s.digest_text = digest_text
    return s


def test_evict_region_below_water_is_empty():
    # 未超预折带（水线）→ 无蒸区（fail-open 零预折）
    # 短段实算 used ≈ 2×est("x"*100)=2×25=50 < 水线 500 → 空蒸区（实算 token 面驱动，无类属性 patch）
    w = Window(1000, [WindowSeg(_mkseg(1)), WindowSeg(_mkseg(2))])
    assert prefold_evict_region(w, 500) == []


def test_evict_region_picks_p2_first_p1_then_p0():
    # D4/D5：蒸区 = P2→P1→P0 逐 tier 选段（对齐 distill _evict_until 选段语义），
    # 返回区间按 seq 升序（跨 tier 合并，供 compact 精准替换按 seq 区间对齐，D9）。
    w = Window(1000, [
        WindowSeg(_mkseg(1, "P0", text="a" * 400, digest_text="a")),
        WindowSeg(_mkseg(2, "P1", text="b" * 400, digest_text="b")),
        WindowSeg(_mkseg(3, "P2", text="c" * 400, digest_text="c")),
    ])
    # 实算 used = 3×est("a"*400) = 3×100 = 300 > 水线 50 → 蒸区非空
    # tier 顺序 P2→P1→P0 逐段累加，P2(100)+P1(100) 蒸掉后 used=100 > 50 仍超 → P0 也进蒸区
    got = prefold_evict_region(w, 50)
    assert got == [1, 2, 3]                                   # 区间 seq 升序（P0(1)/P1(2)/P2(3)）


def test_evict_region_stops_at_water_line(monkeypatch):
    # 压到水线下停（对齐 _evict_until `while used > water: evict`）
    # 用大段（est 远大于 stub）使蒸掉 P2 后 used 跌破水线
    w = Window(100000, [
        WindowSeg(_mkseg(3, "P2", text="c" * 5000, digest_text="c")),
        WindowSeg(_mkseg(2, "P1", text="b" * 5000, digest_text="b")),
        WindowSeg(_mkseg(1, "P0", text="a" * 5000, digest_text="a")),
    ])
    # used = Σsegs.tokens() = 3×est("c"*5000)=3×1250=3750；水线 3000 → P2(3) 蒸掉
    # 后 used≈2500 < 3000 → 停。实算 token 面驱动，无类属性 patch（全仓 test 无隔离风险）。
    got = prefold_evict_region(w, 3000)
    assert got == [3]


def test_evict_region_excludes_structural_p0_floor():
    # structural P0（goal/ledger 尾，结构地板）永不进蒸区（对齐 distill 结构门）
    w = Window(1000, [
        WindowSeg(_mkseg(9, "P0", structural=True, kind="goal", text="g" * 400, digest_text="g")),
        WindowSeg(_mkseg(3, "P2", text="c" * 500, digest_text="c")),
    ])
    # 实算 used = est("g"*400)+est("c"*500) = 100+125 = 225 > 水线 50 → 进蒸区
    # P0 tier：seq 9 是 structural goal 地板 → 结构门跳；P2 tier：seq 3 进蒸区
    got = prefold_evict_region(w, 50)
    assert 9 not in got and 3 in got                          # goal 地板被跳、P2 进蒸区


def test_evict_region_user_p0_segment_included():
    # kind=segment 的 user 指令 P0（非结构地板）可进蒸区（对齐 90% p0_force 口径）
    w = Window(1000, [WindowSeg(_mkseg(5, "P0", prov="user", kind="segment",
                                        text="a" * 400, digest_text="a"))])
    # 实算 used = est("a"*400) = 100 > 水线 50 → 进蒸区
    assert prefold_evict_region(w, 50) == [5]


def test_prefold_llm_prompt_shape():
    # D6/task_407：逐段 prompt = 前段 key_fact + 段原文(≤4096) + 后段 key_fact（滑窗前1后1）
    seg = _mkseg(2, "P1", prov="model", text="did the thing")
    p = build_prefold_llm_prompt(seg, "prev fact", "next fact")
    assert p.startswith("[prefold stub-list]")
    assert "prev_segment_key_fact: prev fact" in p
    assert "did the thing" in p
    assert "next_segment_key_fact: next fact" in p
    assert "why/next/done" in p


def test_prefold_llm_prompt_omits_neighbor_when_missing():
    seg = _mkseg(1, "P1", text="only seg")
    p = build_prefold_llm_prompt(seg, "", "")
    assert "(无前段)" in p and "(无后段)" in p


def test_prefold_llm_prompt_truncates_long_text():
    seg = _mkseg(2, "P1", text="z" * 10000)
    p = build_prefold_llm_prompt(seg, "", "")
    assert "z" * 10000 not in p and "z" * 4096 in p


def test_parse_prefold_llm_response_json():
    r = parse_prefold_llm_response('{"why":"rc","next":"np","done":"dx"}')
    assert r == {"why": "rc", "next": "np", "done": "dx"}


def test_parse_prefold_llm_response_fenced():
    r = parse_prefold_llm_response('```json\n{"why":"a","next":"b","done":"c"}\n```')
    assert r["why"] == "a"


def test_parse_prefold_llm_response_garbage_empty():
    assert parse_prefold_llm_response("") == {}
    assert parse_prefold_llm_response("not json at all") == {}
    # 部分键缺失 → 缺键补 ""（壳侧 None 回退）
    r = parse_prefold_llm_response('{"why":"only"}')
    assert r == {"why": "only", "next": "", "done": ""}
