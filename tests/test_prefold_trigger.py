"""engine 预折触发面（切面 3/4）：0.45 水位查 + 蒸区划定 + 逐段 6 槽 LLM 预生成 + 邻段
key_fact 滑窗 + 白蒸回落清零 + 缺省 off no-op。

owner 定稿（D4/D5/D6/D8，2026-10-28）——本测试面钉 engine 侧接线契约：
- prefold_enabled 缺省 False → 触发/清零/替换三处 no-op（现状零变化，fail-safe 红线）；
- 水位 < 0.45 带 → 无蒸区 + 白蒸回落清零（D8 之一）；
- 水位 ≥ 0.45 带 → 划定蒸区（prefold_evict_region 对齐 distill）→ 逐段预蒸 6 槽 → 写
  prefold.tmp（P1/P2 LLM 3 槽 + 全段确定性 3 槽；P0 全确定性 LLM 槽 None）；
- 邻段 key_fact 滑窗前 1 后 1（task_407，首末段单邻 = 缺侧占位）；
- compact 提取替换（切面 5）+ compact 结算点清零（D8 之二）+ 会话边界清零（D8 之三）。
"""
import os

import pytest

import engine as em
from engine import SupercompressContextEngine
from fakes import FakeEndpoint
from sc.aux_provider import AuxProvider
from sc.prefold import (
    StubList, StubListFile, prefold_tmp_path_for_jsonl,
    build_prefold_llm_prompt, parse_prefold_llm_response, PREFOLD_LLM_TASK,
)
from sc.store import JsonlStore
from sc.distill import Window
from helpers import mkseg, complete_digest, TOPIC, MV


def _mk_engine(tmp_path, store_segs, enabled=True, water_frac=0.45, aux=None,
               context_length=100000):
    """构造 engine（绕过 __init__ 宿主注入面，只接预折相关实例态 + store + aux）。

    `_main_runtime` / `_aux` 是 _window_fresh → _digest_pool_replay_hook / Tagger 依赖面；
    aux 缺省 = FakeEndpoint 空响应（LLM 3 槽 → None 回退确定性面），测试面确定性。
    `context_length` 可调（预折带水线 = context_length × water_frac，测试面需与段体量匹配：
    短段实算 used_tokens 很小，水线须 < 实算 used 才出蒸区）。"""
    from fakes import FakeEndpoint
    e = SupercompressContextEngine.__new__(SupercompressContextEngine)
    e.context_length = context_length
    e.prefold_enabled = enabled
    e.prefold_water_frac = water_frac
    e.last_prompt_tokens = int(context_length * (water_frac + 0.05))   # 略超预折带（进带）
    e._main_runtime = None
    e._aux = aux if aux is not None else AuxProvider(
        FakeEndpoint(), fail_cooldown=0.0, async_digest=False)
    # store 由调用方按用例显式赋值（store_segs 传入仅表意，实际 e.store 调用方设）
    return e


class TestPrefoldDefaultOff:
    """prefold_enabled 缺省 False → 三处 no-op（现状零变化，fail-safe 红线）。"""

    def test_trigger_off_noop(self, tmp_path):
        store = JsonlStore(str(tmp_path / "s.jsonl"))
        seg = mkseg(store, "hello world content here")
        e = _mk_engine(tmp_path, [seg], enabled=False)
        e._prefold_trigger()                    # off → 立即 return，不触达 store/aux
        # 缺省 off 下无任何 prefold 文件产生
        assert not os.path.exists(prefold_tmp_path_for_jsonl(store.path))

    def test_replace_off_empty(self, tmp_path):
        store = JsonlStore(str(tmp_path / "s.jsonl"))
        p0 = mkseg(store, "content")
        e = _mk_engine(tmp_path, [p0], enabled=False)
        e.store = store
        assert e._prefold_replace_on_compact() == {}

    def test_clear_off_noop(self, tmp_path):
        store = JsonlStore(str(tmp_path / "s.jsonl"))
        p0 = mkseg(store, "content")
        e = _mk_engine(tmp_path, [p0], enabled=False)
        e.store = store
        e._prefold_clear("x")                    # off → no-op，零炸
        assert not os.path.exists(prefold_tmp_path_for_jsonl(store.path))


class TestPrefoldTrigger:
    """prefold_enabled=True 触发面。"""

    def test_below_band_no_region_and_clear_stale(self, tmp_path, monkeypatch):
        # 水位 < 0.45 带 → 无蒸区；旧产物白蒸回落清零（D8 之一）
        store = JsonlStore(str(tmp_path / "s.jsonl"))
        mkseg(store, "hello world content here")
        e = _mk_engine(tmp_path, [None], enabled=True)
        e.store = store
        e.last_prompt_tokens = int(100000 * 0.30)            # 低于 0.45 带
        # 预置一份旧产物（模拟上轮 0.45 带蒸的、本轮水位回落应作废）
        f = e._prefold_file()
        f.upsert("stale", StubList(segment_id="stale", done="x").normalize())
        e._prefold_trigger()
        # 回落 → 旧产物被清
        assert f.get("stale") is None
        assert not os.path.exists(f.path)

    def test_in_band_prefolds_region_and_writes_file(self, tmp_path):
        # 水位 ≥ 0.45 带 + 蒸区非空 → 逐段预蒸 6 槽 写 prefold.tmp
        # 用大段（est 远大于 stub）使蒸区非空（实算 token 面驱动，无类属性 patch 隔离风险）
        store = JsonlStore(str(tmp_path / "s.jsonl"))
        p0 = mkseg(store, "请修改 /a/b.py 完成功能 X " + "填充" * 500, tier="P0", prov="user")
        p1 = mkseg(store, "step one did the thing " + "数据" * 500, tier="P1", prov="model")
        p2 = mkseg(store, "step two filler " + "数据" * 500, tier="P2", prov="model")
        e = _mk_engine(tmp_path, [p0, p1, p2], enabled=True, water_frac=0.05,
                       context_length=5000,
                       aux=_fake_aux_with_response(
                           '{"why":"root cause X","next":"next step Y","done":"did Z"}'))
        e.store = store
        # 实算 used_tokens ≈ 765（3 段各 ~255t）；预折带水线 = 5000×0.05 = 250 < 765 → 蒸区非空
        # （last_prompt_tokens 由 _mk_engine 设为 5000×0.10=500 > 250，触发门通过；
        #  蒸区划定读 window 实算 used_tokens，非 last_prompt_tokens，故不覆盖）
        e._prefold_trigger()
        # prefold.tmp 已写（至少 P0 段确定性面成文；P1/P2 LLM 3 槽填充）
        f = e._prefold_file()
        written = f.load()
        assert p0.segment_id in written, "P0 段确定性面必须成文"
        # P0 段：LLM 3 槽恒 None（"确定性面"），files_changed 确定性抽 /a/b.py
        p0sl = written[p0.segment_id]
        assert p0sl.why is None and p0sl.next is None and p0sl.done is None
        assert "/a/b.py" in p0sl.files_changed
        # P1/P2 段：LLM 3 槽填充（canned 响应）
        if p1.segment_id in written:
            assert written[p1.segment_id].why == "root cause X"
            assert written[p1.segment_id].next == "next step Y"
            assert written[p1.segment_id].done == "did Z"

    def test_neighbor_key_fact_sliding_window(self, tmp_path, monkeypatch):
        # task_407：逐段 prompt 附带前段 + 后段 key_fact（滑窗前 1 后 1，首末单邻）
        store = JsonlStore(str(tmp_path / "s.jsonl"))
        a = mkseg(store, "seg a content", tier="P1", prov="model")
        b = mkseg(store, "seg b content", tier="P1", prov="model")
        c = mkseg(store, "seg c content", tier="P1", prov="model")
        a.key_fact = "KA"; b.key_fact = "KB"; c.key_fact = "KC"
        e = _mk_engine(tmp_path, [a, b, c], enabled=True)
        e.store = store
        # 邻段 key_fact 映射：b 取前 a(KA) + 后 c(KC)；首 a 缺前、末 c 缺后
        kf = e._prefold_neighbor_key_facts([a, b, c])
        assert kf[a.seq] == {"prev": "", "next": "KB"}
        assert kf[b.seq] == {"prev": "KA", "next": "KC"}
        assert kf[c.seq] == {"prev": "KB", "next": ""}
        # prompt 面：缺侧渲染占位
        p_first = build_prefold_llm_prompt(a, "", "KB")
        assert "(无前段)" in p_first
        p_last = build_prefold_llm_prompt(c, "KB", "")
        assert "(无后段)" in p_last


class TestPrefoldCompactReplace:
    """切面 5：compact 提取替换（纯数据，零 LLM）+ D8 之二 compact 结算点清零。"""

    def test_replace_extracts_prefolded_stubs(self, tmp_path):
        store = JsonlStore(str(tmp_path / "s.jsonl"))
        p0 = mkseg(store, "instruction to modify /x/y.py", tier="P0", prov="user")
        e = _mk_engine(tmp_path, [p0], enabled=True)
        e.store = store
        # 预置 prefold.tmp（模拟 0.45 带已蒸）
        f = e._prefold_file()
        f.upsert(p0.segment_id,
                 StubList(segment_id=p0.segment_id, files_changed=["/x/y.py"],
                          user_decisions=["settled_signal=1"]).normalize())
        # compact 提取替换（纯数据读 prefold.tmp，零 LLM）
        stubs = e._prefold_replace_on_compact()
        assert p0.segment_id in stubs
        assert stubs[p0.segment_id].files_changed == ["/x/y.py"]

    def test_replace_miss_returns_empty_dict(self, tmp_path):
        store = JsonlStore(str(tmp_path / "s.jsonl"))
        p0 = mkseg(store, "content", tier="P0", prov="user")
        e = _mk_engine(tmp_path, [p0], enabled=True)
        e.store = store
        # 未预折 → 提取面返回 {}（不炸，fail-open）
        assert e._prefold_replace_on_compact() == {}


# ───────────── 切面 6/7：L0 指针存根 list 行形 + auto_recall 存根 list 比对 ─────────────
from sc.inject import _l0_ptr
from sc.prefold import StubList as _SL2
from sc.distill import WindowSeg as _WS


class TestL0PointerStubList:
    """切面 6（D10/D12）：L0 指针行形 = 纯编号锚点，预折 6 槽 list 在场时追加尾。"""

    def test_l0_without_stub_is_pure_anchor(self):
        ws = _WS(_mkseg_test_seg())
        assert _l0_ptr(ws) == f"#{ws.segment.seq} {ws.segment.tier} [{ws.segment.segment_id}] on-demand"

    def test_l0_with_stub_appends_list_tail(self):
        ws = _WS(_mkseg_test_seg())
        stub = _SL2(segment_id=ws.segment.segment_id, files_changed=["/x.py"],
                    why="rc").normalize()
        line = _l0_ptr(ws, stub)
        assert line.startswith(f"#{ws.segment.seq} {ws.segment.tier} [{ws.segment.segment_id}] on-demand · ")
        assert "files=[/x.py]" in line and "why=rc" in line

    def test_l0_with_empty_stub_falls_back_pure_anchor(self):
        # 全空 StubList → is_empty → 回退纯编号锚点（零膨胀 fail-open）
        ws = _WS(_mkseg_test_seg())
        empty = _SL2(segment_id=ws.segment.segment_id).normalize()
        assert _l0_ptr(ws, empty) == f"#{ws.segment.seq} {ws.segment.tier} [{ws.segment.segment_id}] on-demand"


def _mkseg_test_seg():
    from sc.schema import Segment
    s = Segment(kind="segment", tier="P1", provenance="model",
                original_text="some content", session_topic="t",
                model_version="m1", structural_p0=False,
                goal=None, phase_id=None, state=None, open_tasks=[])
    s.seq = 7
    s.digest_ref = "pending"
    return s


class TestSessionBoundaryPrefoldClear:
    """D8 会话边界清零面（宿主 /new 走 on_session_reset，非 on_session_end）：
    /new 交接面必须清掉旧 session 的 prefold.tmp（此刻 store 仍绑旧 session，
    删的正是旧文件；宿主随后 on_session_start rebind 新路径）。"""

    def _full_engine(self, tmp_path):
        # 走真实 __init__（依赖面全）：aux 缺省 Async=False sync；store 外置 tmp。
        e = SupercompressContextEngine(context_length=100000,
                                      external_store_path=str(tmp_path))
        e.on_session_start("s_reset_pf")
        e.prefold_enabled = True                  # 开预折面（缺省 off no-op）
        return e

    def test_on_session_reset_clears_old_prefold(self, tmp_path):
        e = self._full_engine(tmp_path)
        f = e._prefold_file()
        f.upsert("old", StubList(segment_id="old", done="did thing").normalize())
        assert os.path.exists(f.path)
        e.on_session_reset()                      # 宿主 /new 交接面
        assert not os.path.exists(f.path), "reset 后旧 session prefold.tmp 必须被清"

    def test_on_session_end_clears_prefold(self, tmp_path):
        e = self._full_engine(tmp_path)
        f = e._prefold_file()
        f.upsert("old", StubList(segment_id="old", done="did thing").normalize())
        assert os.path.exists(f.path)
        e.on_session_end("s_reset_pf", [])
        assert not os.path.exists(f.path), "会话收尾面 prefold.tmp 必须被清"

    def test_reset_off_noop_zero_io(self, tmp_path):
        # prefold_enabled=False 缺省 → reset 整面 no-op（不触达 prefold 文件，零 IO）
        e = SupercompressContextEngine(context_length=100000,
                                       external_store_path=str(tmp_path))
        e.on_session_start("s_off")
        assert e.prefold_enabled is False
        e.on_session_reset()                      # off → 零 IO，不炸
        assert not os.path.exists(prefold_tmp_path_for_jsonl(e.store.path))


def _fake_aux_with_response(canned_json: str):
    """aux 端点投喂 canned LLM 响应（PREFOLD_LLM_TASK → 固定 JSON；零网络零崩溃）。"""
    ep = FakeEndpoint()
    ep.responses[PREFOLD_LLM_TASK] = canned_json
    return AuxProvider(ep, fail_cooldown=0.0, async_digest=False)
