"""v6 §2.1/§2.2 存根挂逐出事件 + 召回暂存先清（RED→GREEN 回归面）。

锁死（PRD T1-T4, T7, T10）：
T1  80% 逐出 P1/P2 → 账本活面该段折存根行（evict_stub_line 格式 + durable 锚点定位），原文退场
T2  90% force_evict P0 → 同样折存根（P0 走 push cap durable 锚点）
T3  召回段带 source=recall，compact 结算时被清（不进存根账、不留在活账本）
T4  防递归：同一信息「原始存根 + 召回来回 N 次」活账本里只 1 条存根
T7  settle/sink_p0/restore_p0 行为零变化（不碰 P0 任务定案面）
T10 存根与 notes 区正交：被逐出段折存根行（§2.1）与页头 notes 区（§2.4）互不干扰；
    召回段（§2.2）不进 notes 区
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from engine import SupercompressContextEngine
from sc.compact_ledger import CompactLedger, evict_stub_line
from sc.schema import Segment

from helpers import mk_window, complete_digest
from fakes import FakeEndpoint
from sc.aux_provider import AuxProvider


def _mk(tmp_path, context_length: int = 1000, **kw) -> SupercompressContextEngine:
    return SupercompressContextEngine(context_length=context_length,
                                      external_store_path=str(tmp_path), **kw)


def _seed_user(e, text, tier="P0", settled_signal=0, provenance="user"):
    e.store.append(Segment(kind="segment", tier=tier, provenance=provenance,
                           original_text=text, session_topic="", model_version="m1",
                           settled_signal=int(settled_signal)))
    e.store.flush()


def _mkmsgs_long() -> list:
    out = [{"role": "system", "content": "你是 ops 助手，负责网关部署与回滚，保留回滚入口。"}]
    for i in range(14):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
    return out


def _ledger_section(out) -> str:
    for m in out:
        c = str(m.get("content", ""))
        if "compact-ledger" in c:
            return c
    return ""


# ─────────────────────────── evict_stub_line 渲染面（T1/T2 格式锁）───────────────────────────

def test_evict_stub_line_format():
    """T1/T2：存根行格式 = `#seq=N <tier> evicted sha256=<16hex> (durable anchors: <sid>)`。"""
    line = evict_stub_line(3, "#1 P0 [L3]\n网关限流定案五百请求", "P0", "abcdef0123456789-0")
    assert line.startswith("#seq=3 P0 evicted sha256="), line
    assert "abcdef0123456789" in line, "sha256 前 16 hex 可审计（复用 _trimmed 同长口径）"
    assert "(durable anchors: abcdef0123456789-0)" in line, "定位面 = durable 锚点 segment_id"
    assert "full text in recall/store" not in line, "与 _trimmed_stub_line 定位面不同（durable 锚点非 recall 层）"


def test_evict_page_folds_stub_and_removes_full_text(tmp_path):
    """T1：被逐出段在活账本自己那页折存根行，原文退场（活渲染面不再重吐全文）。"""
    lg = CompactLedger(str(tmp_path / "a.compact.json"))
    e = lg.append("#1 P0 [L3]\n网关限流阈值定案五百请求", settled=False, kind="adhoc")
    assert e is not None
    full_before = lg.text()
    assert "网关限流阈值定案五百请求" in full_before
    folded = lg.evict_page(e["seq"], "P0", "abcd1234-0")
    assert folded is not None and folded["evicted"], "折存根 entry 带 evicted 存根字段"
    after = lg.text()
    assert "网关限流阈值定案五百请求" not in after, "原文退场（活渲染面只存根）"
    assert "evicted sha256=" in after, "存根行在场（evict_stub_line 渲染）"
    assert "(durable anchors: abcd1234-0)" in after
    # 幂等：再折一次 → None（零写盘）
    assert lg.evict_page(e["seq"], "P0", "abcd1234-0") is None
    # fail-open：找不到 seq → None
    assert lg.evict_page(99, "P0", "x") is None


def test_evict_page_stub_orthogonal_to_notes(tmp_path):
    """T10：存根行写在逐出页 entries 面，与页头 notes 区互不干扰（notes 只记 settled）。"""
    lg = CompactLedger(str(tmp_path / "b.compact.json"))
    lg.append("#1 P0 [L3]\n定案段A", settled=True, kind="adhoc")
    e2 = lg.append("#2 P1 [L3]\n推理段B")
    lg.evict_page(e2["seq"], "P0", "feedface-1")
    out = lg.text()
    _pg = re.search(r"── 页 c#2 ──\n---\n(.*?)(?:\n---\n── 页 c#|\Z)", out, re.S)
    assert _pg, f"页 c#2 页段在场：{out!r}"
    _body = _pg.group(1)
    # 页 c#2 notes 区 = 前页（c#1）settled 快照；存根在 c#2 正文面，二者同页正交
    _notes_m = re.match(r"notes c#1\=.*\n", _body)
    assert _notes_m and "c#1=settled" in _notes_m.group(0), \
        "notes 区只记 settled 状态（c#1 settled → 快照 settled）"
    assert "evicted" not in _notes_m.group(0), "notes 区无 evicted 词（存根不进 notes）"
    _stub_m = re.search(r"#seq=2 P0 evicted[^\n]*", _body)
    assert _stub_m, f"存根行在正文面：{_body!r}"
    assert "evicted" in _stub_m.group(0)


def test_evict_page_stub_does_not_count_against_cap(tmp_path):
    """存根块不占条数帽位（同 trimmed 口径）：折存根后全文块计数下降。"""
    lg = CompactLedger(str(tmp_path / "c.compact.json"), max_tokens=10_000, max_entries=2)
    lg.append("#1 P0 [L3]\nA" * 3)
    lg.append("#1 P0 [L3]\nB" * 3)
    # 超帽 seal 后重新积累：取任意在场页折存根 → 全文块计数 -1
    target = lg.entries[0]
    _before = lg._full_block_count()
    lg.evict_page(target["seq"], "P0", "anchor-x")
    assert target["trimmed"] is True, "evicted 块同 trimmed 口径（不占帽位）"
    assert lg._full_block_count() < _before, "折存根真腾帽位（语义 X 口径扩展）"


# ─────────────────────────── distill 逐出点接线（T1/T2）───────────────────────────

def _mk_store(tmp_path):
    from sc.store import JsonlStore
    return JsonlStore(str(tmp_path / "s_evict.jsonl"))


def _distill_evict_e2e(tmp_path, tier, force_evict=False, cap=3000):
    """造窗口 + store（durable）+ digest ready，Distiller 逐出 → 活账本存根在场。

    造段 token > 80% water（est_tokens = ceil(chars/4)；cap=1000 → water=800：
    "transient-segment" × 250 = 3250 chars ≈ 813 tok > 800 water，真超水线 →
    正常档（非 force）也会逐出 P2/P1（T1）；force_evict=True 走 P0 兜底逐出（T2）。"""
    from sc.distill import Distiller
    from sc.schema import Segment as Seg

    store = _mk_store(tmp_path)
    # provenance=model（P1 推理段可逐出；user provenance 长内容恒 P0 → 80% 档豁免）
    text = "transient-segment" * 250
    seg = Seg(kind="segment", tier=tier, provenance="model", original_text=text,
              session_topic="t", model_version="m1", digest_ref="pending")
    store.append(seg)
    store.flush()
    assert seg.content_hash in store.durable, "段已 durable（fsync+读回锚点在场）"
    # sc 契约口径（同 conftest aux fixture）：fail_cooldown=0 + 显式 sync 模式
    # （async_digest=False = 无 worker 线程、digest 立即生成 → 测试确定性）
    aux = AuxProvider(FakeEndpoint(), fail_cooldown=0.0, async_digest=False)
    aux.end_turn()
    complete_digest(aux, seg)
    window = mk_window(cap, [seg])
    ledger = CompactLedger(str(tmp_path / "s_evict.compact.json"))
    ledger.append(f"#1 {tier} [L3]\n{seg.original_text[:60]}")

    called = []
    def _cb(_seg):
        called.append(_seg.segment_id)
        ledger.evict_page(ledger.entries[0]["seq"], _seg.tier, _seg.segment_id)
    Distiller(window, store, aux, force_evict=force_evict,
              ledger_evict_cb=_cb).run()
    return ledger, called, window


def test_80pct_evict_p1_p2_folds_ledger_stub(tmp_path):
    """T1：80% 软水位逐出 P1（durable + digest ready）→ cb 触达 → 活账本折存根，原文退场。"""
    ledger, called, window = _distill_evict_e2e(tmp_path, "P1", cap=1000)
    evicted = [ws for ws in window.segs if ws.form == "digest_only"]
    assert evicted, "80% 水线触发逐出（P1 段翻 digest_only）"
    assert called, "durable 逐出段触达存根 cb（_evict 挂逐出事件面）"
    out = ledger.text()
    assert "evicted sha256=" in out, "活账本活面折存根行（_trimmed_stub_line 同形）"
    assert "(durable anchors:" in out, "存根定位面 = durable 锚点（owner 选 a）"
    assert "网关限流阈值调至每秒五百请求" not in out, "被逐出段原文退场（只存根）"


def test_90pct_force_evict_p0_folds_ledger_stub(tmp_path):
    """T2：90% force_evict P0 → 同样折存根（P0 走 push cap durable 锚点）。"""
    ledger, called, window = _distill_evict_e2e(tmp_path, "P0", force_evict=True, cap=1000)
    evicted = [ws for ws in window.segs if ws.form == "digest_only"]
    assert evicted, "force_evict 兜底逐出 P0"
    assert called, "durable P0 逐出段触达存根 cb"
    out = ledger.text()
    assert "evicted sha256=" in out, "P0 逐出存根在场（T2 与 T1 同形）"


def test_non_durable_segment_no_stub(tmp_path):
    """非 durable 段（pending 字节无锚点）逐出 → 不折存根（锚点未发射 = 存根不挂）。"""
    from sc.distill import Distiller

    store = _mk_store(tmp_path)
    # 瞬态段 token > 80% water（est_tokens = ceil(chars/4)，中文 char=char；
    # 1000 cap → water 800：400 × 5 字 = 2000 字 ≈ 500 tok… 用 ASCII 词面放大：
    # "transient-段" 混合不超估——直接 800 × 4 字 = 3200 字 ≈ 800 tok 水线平，
    # 取 3600 字 ≈ 900 tok > 800 water 保证超线；P1 lazy 批未 flush → 不在 durable）
    seg = Segment(kind="segment", tier="P1", provenance="model",
                  original_text="transient-segment" * 250, session_topic="t",
                  model_version="m1", digest_ref="pending")
    store.append(seg)            # P1 lazy：未 flush → 不在 store.durable
    aux = AuxProvider(FakeEndpoint(), fail_cooldown=0.0, async_digest=False)
    aux.end_turn()
    complete_digest(aux, seg)
    window = mk_window(1000, [seg])   # 段 token > 80% water → 逐出触发
    ledger = CompactLedger(str(tmp_path / "s_nd.compact.json"))
    ledger.append("#1 P1 [L3]\n瞬态段内容")
    called = []
    def _cb(_seg):
        called.append(_seg.segment_id)
        ledger.evict_page(ledger.entries[0]["seq"], "P1", _seg.segment_id)
    Distiller(window, store, aux, ledger_evict_cb=_cb).run()
    evicted = [ws for ws in window.segs if ws.form == "digest_only"]
    assert evicted, "80% 水线仍逐出该段（逐出语义零变化）"
    assert called == [], "非 durable 逐出段不触达存根 cb（锚点未发射）"
    assert "evicted" not in ledger.text(), "活账本无存根（原文照回喂）"


# ─────────────────────────── 召回暂存先清（T3/T4）───────────────────────────

def test_recall_staging_cleared_at_settle(tmp_path):
    """T3：召回段带 source=recall（instance 态 id 集），compact 结算时被清——
    不进存根账、不留在活账本（直接丢弃，owner 选 i）。"""
    e = _mk(tmp_path)
    e.on_session_start("s_recall")
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert _ledger_section(out1), "compact #1 账本在场"

    # 模拟上轮召回：store 段被召回注入（select_context 面），engine 记 id 集
    _seg0 = next(s for s in e.store.by_hash.values()
                 if s.original_text == "网关限流阈值调至每秒五百请求并配置告警通知")
    e._last_recall_seg_ids = {_seg0.segment_id}
    # compact #2：结算点先清上轮召回暂存段（staged 块含该段 → 对应账本页被丢）
    _seed_user(e, "回滚开关位置在网关配置第三节保留现场")
    out2 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    sec = _ledger_section(out2)
    assert sec, "compact #2 账本仍在场（清算后照常积累）"
    dropped_seqs = e._last_recall_page_seqs
    assert dropped_seqs, "结算点清掉了上轮召回暂存页（直接丢，不留占位）"
    for _sq in dropped_seqs:
        assert e._ledger.entry_by_seq(_sq) is None, "被丢页不进活账本（entry 已清）"
    assert "evicted" not in sec, "召回段不参与存根折叠（T3：不进存根账）"


def test_recall_recursive_stub_stays_single(tmp_path):
    """T4 防递归：同一信息「原始存根 + 召回来回 N 次」活账本里只 1 条存根。"""
    e = _mk(tmp_path)
    e.on_session_start("s_recur")
    _seed_user(e, "网关限流阈值定案五百请求配告警保留回滚入口")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert _ledger_section(out1)
    _orig_seg = next(s for s in e.store.by_hash.values()
                     if s.original_text == "网关限流阈值定案五百请求配告警保留回滚入口")
    for _i in range(3):
        e._last_recall_seg_ids = {_orig_seg.segment_id}
        out = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
        sec = _ledger_section(out)
        assert sec, f"第 {_i + 1} 轮账本在场"
        assert sec.count("evicted") <= 1, "召回态不进存根账：活账本至多 1 条存根（防递归）"
        assert e._last_recall_page_seqs or not e._last_recall_seg_ids, \
            "结算点清算面（有召回段 → 有被丢页；无 → 零丢）"


# ─────────────────────────── MED-1/MED-2 第二轮修复面 ───────────────────────────

def test_body_literal_seq_not_misfolded(tmp_path):
    """MED-2：P0 段 body 含字面 `seq=999`（代码片段/日志行）→ 逐出该段只折其真实
    seq 的存根页，**不误折 999 页**；行首段标记位锚定正则单源（_evict_seq_index）。"""
    e = _mk(tmp_path)
    e.on_session_start("s_med2")
    # body 里塞字面 seq=999（staged 段原文重渲染面会把它带进账本页 body）
    _seed_user(e, "def f():\n    x = 'seq=999'  # 日志片段字面\n    return x")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert _ledger_section(out1), "compact #1 账本在场"
    _lg = e._ledger
    _page = _lg.entries[0]
    assert "seq=999" in _page["text"], "body 字面 seq=999 已进账本页（误命中面在场）"
    # 段标记位锚定映射：真实全局 seq（行首 `#1 P0 prov=… seq=<N>`）映射到该页
    _real_seq = next(s.seq for s in e.store.replay()
                     if "seq=999" in s.original_text)
    _idx = e._evict_seq_index(_lg)
    assert str(_real_seq) in _idx, "真实行首段标记 seq 映射在场"
    assert _idx.get(str(_real_seq)) == _page["seq"], "真实 seq 映射到该页"
    assert _idx.get("999") is None or _idx["999"] != _page["seq"], \
        "body 字面 seq=999 未被误当段标记（999 非 store 全局段 seq → 不误折他页）"
    # 逐出该段（durable + digest ready 接线面）→ 只折真实 seq 的页
    _seg = next(s for s in e.store.replay() if "seq=999" in s.original_text)
    _anchor = _seg.segment_id
    _lg.evict_page(_idx[str(_real_seq)], _seg.tier, _anchor)
    _text = _lg.text()
    assert "evicted sha256=" in _text, "真实 seq 页已折存根"
    # 召回先清同面：body 字面 seq=999 不该让「999 在召回集」误丢本页
    _drop = e._recall_page_seqs(_lg, {"999"})
    assert _drop == [], "召回集仅含 body 字面 999 → 零误丢（非召回页不丢）"
    _drop2 = e._recall_page_seqs(_lg, {str(_real_seq)})
    assert _drop2 == [_page["seq"]], "真实 seq 在召回集 → 该页正常先清映射"


def test_l0_segment_no_evict_stub(tmp_path):
    """MED-1（owner 裁定 (b)）：L0 高压段行形 `#seq tier [segment_id] on-demand`
    （sc/inject._l0_ptr 缓存锚位，行首无 `seq=` 位）→ 逐出后账本活面**无**该段
    evicted 存根行——锁死「L0 不折存根」边界：高压最小足迹兜底本就不需要存根，
    原文 durable 锚点仍在 store 可审计（fail-open 零数据损失）。"""
    from sc.inject import _l0_ptr
    e = _mk(tmp_path)
    e.on_session_start("s_med1")
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知保留回滚入口")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert _ledger_section(out1), "compact #1 账本在场"
    _lg = e._ledger
    _seg = next(s for s in e.store.replay())
    # 构造 L0 行形（高压兜底档），验证其恒不命中段标记位锚定正则
    from types import SimpleNamespace
    class _WS:  # 最小 ws 桩（_l0_ptr 只读 segment.seq/tier/segment_id）
        segment = _seg
    _l0_line = _l0_ptr(_WS())
    assert "seq=" not in _l0_line, "L0 行形无 seq= 位（缓存锚位锁死，绝不改动）"
    _lg2 = CompactLedger(str(tmp_path / "s_med1_l0.compact.json"))
    _p = _lg2.append(_l0_line)
    _idx = e._evict_seq_index(_lg2)
    assert _idx.get(str(_seg.seq)) is None, \
        "L0 行形不映射段标记（_evict_seq_index 正则 miss = L0 不折存根边界）"
    # 即使强行调 cb 面：映射 miss → evict_page 不被触发（fail-open）
    _cb_pages = []
    def _probe(_s):
        _ps = e._evict_seq_index(_lg2).get(str(_s.seq))
        _cb_pages.append(_ps)
        if _ps is not None:
            _lg2.evict_page(_ps, _s.tier, _s.segment_id)
    _probe(_seg)
    assert _cb_pages == [None], "L0 段逐出映射 miss（cb 不折存根）"
    assert "evicted" not in _lg2.text(), "账本活面无该段 evicted 存根行（边界锁死）"


# ─────────────────────────── T7：P0 任务定案面零变化 ───────────────────────────

def test_settle_sink_restore_zero_change(tmp_path):
    """T7：settle/sink_p0/restore_p0 行为零变化（v6 改动面不碰 P0 任务定案面）。"""
    import scdata
    from sc.compact_ledger import _settled_stub_line
    lg = CompactLedger(str(tmp_path / "d.compact.json"))
    e1 = lg.append("#1 P0 [L3]\n定案原文", settled=True, kind="adhoc")
    assert e1["settled"] and e1["kind"] == "adhoc"
    out = lg.text()
    _sid = os.path.splitext(os.path.basename(lg.path))[0].replace(".compact", "")
    assert "c#1: " + _settled_stub_line(1, "#1 P0 [L3]\n定案原文", "adhoc", _sid) in out
    # settle 幂等面（既有行为）
    assert lg.settle(e1["seq"]) is None, "settle 幂等（已 settled → None）"
    # sink/restore 同事务面（既有行为）
    ok = scdata.sink_p0(str(tmp_path), e1["seq"], e1["text"], kind="adhoc", sid=_sid)
    assert ok
    kind, text = scdata.restore_p0(str(tmp_path), e1["seq"], sid=_sid)
    assert text == e1["text"] and kind == "adhoc"


# ─────────────────────────── engine 结算点先清（T3 壳面）───────────────────────────

def test_engine_compress_settle_drops_recall_pages(tmp_path):
    """engine compress() 结算点：上轮召回段 id 集 → 含该段的**旧账本页**被丢（T3 壳面）。

    纪律（§2.2 owner 选 i）：被丢页文本不再出现在**被丢页之后的新积累页**
    （store 段本身仍在 = 信息可审计；丢弃只指「旧账本页不重进活账本」。
    新 staged 块由 store 重建——若后续轮次 store 段仍在窗口，信息经新页再次
    入账是 store→窗口→staged 的正常流，非「召回暂存重进账本」；递归防护的
    可观测点 = 旧被丢页不再存在 + 每轮至多 1 条存根（T4）。"""
    e = _mk(tmp_path)
    e.on_session_start("s_shell")
    _seed_user(e, "网关限流阈值调至每秒五百请求并配置告警通知")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert _ledger_section(out1)
    # 模拟上轮召回：staged 块全文（= 账本页 text）对应段 id 集
    _lg = e._ledger
    _page_text = _lg.entries[0]["text"]
    # staged 块行首 `seq=<全局段序>` 标记 = engine._evict_seq_index 映射源
    _m = re.search(r"seq=(\d+)", _page_text)
    assert _m, "staged 块带 seq= 全局段序（精确映射源）"
    # 用 store 里真实段 id（与 staged 块同源段）造召回集
    _all_segs = e.store.replay()
    _seg0 = next((s for s in _all_segs if s.seq == int(_m.group(1))), None)
    assert _seg0 is not None, "store 段与 staged 块 seq 同源"
    _old_pages = {_e["seq"] for _e in _lg.entries}
    e._last_recall_seg_ids = {_seg0.segment_id}
    out2 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    # 结算点清算：含召回段的旧账本页被 discard（T3）
    assert e._last_recall_page_seqs, "被丢页 seq 登记（观测面）"
    assert all(_sq in _old_pages for _sq in e._last_recall_page_seqs), \
        "被丢的是既有页（新页不可能是上轮召回段）"
    _dropped = set(e._last_recall_page_seqs)
    for _sq in _dropped:
        assert e._ledger.entry_by_seq(_sq) is None, "被丢页不在活账本（entry 已清）"
    # 递归防护可观测点：活账本至多 1 条存根（召回态不进存根账，T4 壳面同源）
    assert _ledger_section(out2).count("evicted") <= 1, \
        "清算后活账本存根不递归堆叠"


# ─────────────────────────── LOW-4：同段多次逐出去重 ───────────────────────────

def test_same_segment_dedup_evict_stub(tmp_path):
    """LOW-4（owner 拍板）：同一 P1 段在 3 轮 80% 逐出（每轮 compact 重 staged →
    新账本页 → 逐出 cb 折新页存根）下，活账本活面该段 evicted 存根行恒 1 条
    （去重门控：同 segment_id 已存根 → 跳过，零写盘）；不同段 sha256 不同则
    各自折存根（去重不误并）。

    实现面：staged 块行首形 `#1 P1 prov=model seq=<全局seq> · <text>`（sc.inject.render_block
    同形），engine._evict_seq_index 按行首 `seq=<N>` 映射到账本页 seq。
    测试手动追加 3 轮 staged 块（同段 A 3 次 + 段 B 1 次），每轮都含真实
    store 全局 seq → _evict_seq_index 可映射 → evict_page 被调 → 去重生效。"""
    e = _mk(tmp_path)
    e.on_session_start("s_low4")
    # 造两个 P1 段（不同 content → 不同 segment_id；P1 lazy tier 可逐出）
    _seed_user(e, "LOW4-段A：网关限流阈值调至每秒五百请求", tier="P1")
    _seed_user(e, "LOW4-段B：回滚开关位置在网关配置第三节", tier="P1")
    out1 = e.compress(_mkmsgs_long(), current_tokens=4000, force=True)
    assert _ledger_section(out1), "compact #1 账本在场"
    _lg = e._ledger

    _seg_a = next(s for s in e.store.replay() if "LOW4-段A" in s.original_text)
    _seg_b = next(s for s in e.store.replay() if "LOW4-段B" in s.original_text)

    def _do_evict(lg, seg):
        """模拟 engine 侧 evict stub cb：行首 seq=<全局> 映射 + evict_page。"""
        _idx = e._evict_seq_index(lg)
        _page_seq = _idx.get(str(seg.seq))
        if _page_seq is not None:
            lg.evict_page(_page_seq, seg.tier, seg.segment_id)

    # 第 1 轮：A 折存根（staged 块行首含真实 seg.seq）
    _lg.append(f"#1 P1 prov=model seq={_seg_a.seq} · LOW4-段A-staged-轮1")
    _do_evict(_lg, _seg_a)
    assert "evicted" in _lg.text(), "第 1 轮：段 A 存根在场"

    # 第 2 轮：A 再折（同 segment_id → 去重，不新增）
    _lg.append(f"#1 P1 prov=model seq={_seg_a.seq} · LOW4-段A-staged-轮2")
    _do_evict(_lg, _seg_a)
    # 段 B 首次折存根（不同 segment_id → 正常新增）
    _lg.append(f"#1 P1 prov=model seq={_seg_b.seq} · LOW4-段B-staged-轮1")
    _do_evict(_lg, _seg_b)

    _out = _lg.text()
    _evicted_a_count = _out.count(f"(durable anchors: {_seg_a.segment_id})")
    _evicted_b_count = _out.count(f"(durable anchors: {_seg_b.segment_id})")
    assert _evicted_a_count == 1, (
        f"段 A evicted 存根应恒 1 条（去重），实际 {_evicted_a_count} 条：\n{_out!r}")
    assert _evicted_b_count == 1, (
        f"段 B evicted 存根应恒 1 条（不同 segment_id 不误并），实际 {_evicted_b_count} 条")

    # 第 3 轮：A 再折（去重，仍 1 条）
    _lg.append(f"#1 P1 prov=model seq={_seg_a.seq} · LOW4-段A-staged-轮3")
    _do_evict(_lg, _seg_a)
    _out3 = _lg.text()
    _evicted_a_count3 = _out3.count(f"(durable anchors: {_seg_a.segment_id})")
    assert _evicted_a_count3 == 1, (
        f"第 3 轮后段 A evicted 存根仍应恒 1 条，实际 {_evicted_a_count3} 条：\n{_out3!r}")
