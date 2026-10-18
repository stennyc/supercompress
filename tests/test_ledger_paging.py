"""v6.1 §2.4 账本页化 + 页头 notes 区快照冻结（RED→GREEN 回归面）。

锁死（PRD T5/T8/T9/T10 + 契约迁移）：
T5  55% 分页面字节不变（存根只 80/90 逐出点发生，55 分页器不动；
    页化只改渲染结构，分页/累计 append-only 行为零变化）
T8  页化 notes 区快照冻结：新页只 append 自己 entries + 前 N-1 页 settled 快照；
    无翻转时旧页字节恒不变（前缀缓存命中除最后一页外全部旧页）
T9  状态翻转归属：某页 settle 只写进其后新页 notes 区，不回改旧页
T10 存根与 notes 正交：被逐出段折存根行（§2.1 evict_page）与页头 notes 区（§2.4）
    互不干扰；召回段（§2.2 discard_entries）不进 notes 区
契约迁移（有意，非回归）：test_compact_ledger 旧「累计超集严格前缀」
（c2.startswith(c1)）已随页化迁移为「旧页字节冻结 + 新页只装自己 + notes 区跨页
一行索引」（见 test_ledger_paging_prefix_anchor_migration）。
"""
from __future__ import annotations

import re
from pathlib import Path

from sc.compact_ledger import CompactLedger, evict_stub_line

PAGE_RE = re.compile(r"── 页 c#\d+ ──")


def _pages(section: str) -> list[str]:
    """section 拆页（「── 页 c#N ──」页头到下一页头/段末；尾部页间分隔符裁掉）。"""
    _marks = list(PAGE_RE.finditer(section))
    _out = []
    for _i, _m in enumerate(_marks):
        _end = _marks[_i + 1].start() if _i + 1 < len(_marks) else len(section)
        _out.append(section[_m.start():_end].rstrip("\n---"))
    return _out


def _page_lines(page: str) -> list:
    """页段 splitlines：[页头, '', 'notes…', '', 正文行…]（MARK 行除外）。"""
    return page.splitlines()


def _page(section: str, tag: str) -> str:
    for _p in _pages(section):
        if _p.startswith(f"── 页 {tag} ──"):
            return _p
    return ""


def _notes_line(page: str) -> str:
    """页段内 notes 区行（'notes …' 或 'notes: -'；旧页占位面）。"""
    for _ln in _page_lines(page):
        if _ln.startswith("notes"):
            return _ln
    return ""


# ─────────────────────────── T8 快照冻结（核心）───────────────────────────

def test_new_page_appends_only_own_entries_and_prev_notes_snapshot(tmp_path):
    """T8：新页只 append 自己 entries + 前 N-1 页 settled 快照（写定后永不回改）。"""
    lg = CompactLedger(str(tmp_path / "p1.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    lg.append("#1 P0 [L3]\n段二原文")
    out = lg.text()
    assert len(_pages(out)) == 2, "两页"
    p2 = _page(out, "c#2")
    assert "段二原文" in p2, "页 c#2 装自己 entries"
    assert "段一原文" not in p2, "页 c#2 不重吐 c#1 全文（规则 1）"
    assert "notes c#1=open" in p2, "页 c#2 notes 区 = c#1 settled 快照（规则 2）"
    p1 = _page(out, "c#1")
    assert "notes: -" in p1, "首页 notes 区 = 空快照占位"
    # 写定后永不回改正文：再 append 一页，c#1/c#2 正文字节不回改；各自在「变旧页」
    # 那一刻 append 一行页脚（v6.1 §2.5，新页 c#3 此刻无页脚）
    lg.append("#1 P0 [L3]\n段三原文")
    out2 = lg.text()
    _p1b = _page(out2, "c#1")
    _p2b = _page(out2, "c#2")
    assert _p1b.startswith(p1), "无 settled 翻转 → 旧页 c#1 正文恒不变（规则 4）"
    assert _p2b.startswith(p2), "旧页 c#2 正文恒不变（规则 4）"
    assert _p1b.endswith("\n[页脚：本页及以上旧段为历史参考（不作为当前任务）；当前任务以最新 user 段为准]"), \
        "c#1 变旧页 → 追加 1 行页脚（§2.5）"
    assert _p2b.endswith("\n[页脚：本页及以上旧段为历史参考（不作为当前任务）；当前任务以最新 user 段为准]"), \
        "c#2 变旧页 → 追加 1 行页脚（§2.5）"
    p3 = _page(out2, "c#3")
    assert "notes c#1=open c#2=open" in p3, "新页 notes = 写入那一刻前 2 页逐页快照"
    assert "段三原文" in p3 and "段二原文" not in p3, "新页只装自己"
    assert not p3.endswith("[页脚：本页及以上旧段为历史参考（不作为当前任务）；当前任务以最新 user 段为准]"), \
        "当前最新页 c#3 无页脚（§2.5 只对旧页生效；当前页任务仍活）"


def test_notes_snapshot_frozen_no_flips_bytes_stable(tmp_path):
    """T8/T5 同族：55% 分页器不动（存根只 80/90 发生）——账本 append 序列页化渲染
    只增不回改；连续 append 无翻转时旧页字节恒不变（前缀缓存锚）。"""
    lg = CompactLedger(str(tmp_path / "p2.compact.json"))
    for _i in range(4):
        lg.append(f"#1 P0 [L3]\n块{_i}")
    _prev = lg.text()
    lg.append("#1 P0 [L3]\n块4")
    _now = lg.text()
    # 旧 4 页正文恒为 new 的前缀面（除新页 c#5 外正文不变；各自在变旧页那一刻
    # 追加 1 行页脚，v6.1 §2.5——c#1..c#3 在 _prev 时已是旧页、已带页脚）
    for _tag in ("c#1", "c#2", "c#3", "c#4"):
        assert _page(_now, _tag).startswith(_page(_prev, _tag)), \
            f"旧页 {_tag} 正文不变"
    assert _page(_now, "c#5") != "", "新页在场"
    assert "块4" in _page(_now, "c#5") and "块3" not in _page(_now, "c#5"), \
        "新页只装自己（55% 分页器语义零变化）"
    assert not _page(_now, "c#5").endswith("[页脚：本页及以上旧段为历史参考（不作为当前任务）；当前任务以最新 user 段为准]"), \
        "当前最新页 c#5 无页脚（§2.5 只对旧页生效）"


# ─────────────────────────── T9 状态翻转归属 ───────────────────────────

def test_settle_flip_only_in_subsequent_pages_notes(tmp_path):
    """T9：某页 later settle 只在它之后的新页 notes 区体现，绝不回改旧页。"""
    lg = CompactLedger(str(tmp_path / "p3.compact.json"))
    lg.append("#1 P0 [L3]\n段A")
    lg.append("#1 P0 [L3]\n段B")
    p1_before = _page(lg.text(), "c#1")
    # c#1 settle（状态翻转）：旧页 notes 区字节不回改（notes 写定即冻结）
    lg.settle(1)
    assert _notes_line(_page(lg.text(), "c#1")) == "notes: -", \
        "旧页 c#1 notes 区字节冻结（settle 不回改旧页）"
    # 翻转只写进其后新页 notes 区
    lg.append("#1 P0 [L3]\n段C")
    out3 = lg.text()
    p3 = _page(out3, "c#3")
    assert "notes c#1=settled c#2=open" in p3, \
        "翻转后的新页 notes = 当时快照（c#1 settled、c#2 open）"
    assert "settled" not in _notes_line(_page(out3, "c#1")), "旧页 c#1 notes 无翻转记录"
    assert "settled" not in _notes_line(_page(out3, "c#2")), "旧页 c#2 notes 无翻转记录"


def test_settle_page_body_folds_stub_but_notes_frozen(tmp_path):
    """T9 同源：settle 使某页正文折 settled 存根（既有渲染面），但其 notes 区
    行字节不回改（notes 只写写入那一刻快照；翻转记录只在新页 notes）。"""
    lg = CompactLedger(str(tmp_path / "p4.compact.json"))
    lg.append("#1 P0 [L3]\n定案段A")
    lg.append("#1 P0 [L3]\n段B")
    _notes_before = _notes_line(_page(lg.text(), "c#1"))
    lg.settle(1)
    _a = lg.text()
    assert "settled" in _page(_a, "c#1"), "c#1 正文折 settled 存根（既有渲染面）"
    assert _notes_line(_page(_a, "c#1")) == _notes_before, \
        "c#1 notes 区行字节冻结（settle 不回改旧页 notes）"
    assert _page(_a, "c#1").endswith("[页脚：本页及以上旧段为历史参考（不作为当前任务）；当前任务以最新 user 段为准]"), \
        "settle 后 c#1（旧页）正文尾折存根 → 追加 1 行页脚（§2.5，settled 页是旧任务）"


# ─────────────────────────── T10 存根与 notes 正交 ───────────────────────────

def test_evicted_stub_line_orthogonal_to_notes_region(tmp_path):
    """T10：被逐出段折存根行（§2.1 evict_page）写在逐出页 entries 面；
    页头 notes 区（§2.4）只记 settled 快照，二者互不干扰。"""
    lg = CompactLedger(str(tmp_path / "p5.compact.json"))
    lg.append("#1 P0 [L3]\n定案段A", settled=True, kind="adhoc")
    e2 = lg.append("#1 P1 [L3]\n推理段B")
    lg.evict_page(e2["seq"], "P1", "anchor-seg-1")
    out = lg.text()
    p2 = _page(out, "c#2")
    _lines = p2.splitlines()
    assert _lines[0] == "── 页 c#2 ──", "页头行"
    _notes = _notes_line(p2)
    assert "notes c#1=settled" in _notes, "notes 区 = settled 快照（跨页一行索引）"
    assert evict_stub_line(2, "#1 P1 [L3]\n推理段B", "P1", "anchor-seg-1") in p2, \
        "逐出存根行在 entries 面"
    assert "evicted" not in _notes, "存根不混进 notes 区行"
    assert not p2.endswith("[页脚：本页及以上旧段为历史参考（不作为当前任务）；当前任务以最新 user 段为准]"), \
        "当前最新页 c#2 无页脚（§2.5 只对旧页生效）"
    assert _page(out, "c#1").endswith("[页脚：本页及以上旧段为历史参考（不作为当前任务）；当前任务以最新 user 段为准]"), \
        "旧页 c#1（settled）带页脚（§2.5）"


def test_recall_discarded_pages_never_in_notes_region(tmp_path):
    """T10/T3：召回段（§2.2 discard_entries 直接丢）不进 notes 区——
    被丢页 seq 不出现在任何页的 notes 快照里（丢弃 = entries 删除，
    快照 = 写入时刻投影，天然不含已丢页）。"""
    lg = CompactLedger(str(tmp_path / "p6.compact.json"))
    e1 = lg.append("#1 P0 [L3]\n召回暂存段X")
    lg.discard_entries([e1["seq"]])
    lg.append("#1 P0 [L3]\n新段Y")
    out = lg.text()
    assert "召回暂存段X" not in out, "被丢页文本不在渲染面（直接丢）"
    _p = _page(out, "c#2")
    assert _p, "新页在场"
    assert "c#1" not in _p.split("\n---\n")[2], "新页 notes 区无被丢页 c#1（丢页不进快照）"


# ─────────────────────────── 旧盘兼容（fail-open）───────────────────────────

def test_old_sidecar_without_notes_renders_placeholder(tmp_path):
    """旧盘（无 notes 键）reload → notes 区渲染「notes: -」占位（不回填旧页 =
    不回改旧页字节，§2.4 铁律 3 精神），正文渲染面逐字节不变。"""
    _p = str(tmp_path / "old.compact.json")
    import json
    import os
    # 手工造旧格式账本（无 notes/evicted 键；version 须匹配 _FORMAT_VERSION）
    os.makedirs(os.path.dirname(_p), exist_ok=True)
    with open(_p, "w", encoding="utf-8") as _f:
        json.dump({"version": 1, "generation": 0, "seals": 0, "seq": 1,
                   "entries": [{"seq": 1, "text": "#1 P0 [L3]\n旧块原文",
                                 "tokens": 5, "settled": False, "kind": "adhoc",
                                 "trimmable": False}]}, _f)
    lg = CompactLedger(_p)
    out = lg.text()
    assert out, "旧盘 reload 渲染面非空"
    assert "notes: -" in out, "旧页无 notes 键 → 占位行（快照缺失不回填）"
    assert "#1 P0 [L3]\n旧块原文" in out, "旧页正文渲染面逐字节不变"
    # 新写页带 notes 键（新页快照面生效，旧页面不动）
    lg.append("#2 P0 [L3]\n新块")
    out2 = lg.text()
    assert _page(out2, "c#1") and "notes: -" in _page(out2, "c#1"), \
        "旧页 notes 仍占位（不回填）"
    assert "notes c#1=open" in _page(out2, "c#2"), "新页 notes 快照面生效"
    assert _page(out2, "c#1").endswith("[页脚：本页及以上旧段为历史参考（不作为当前任务）；当前任务以最新 user 段为准]"), \
        "旧页 c#1 变旧 → 追加页脚（§2.5，旧盘 fail-open 也生效）"
    assert not _page(out2, "c#2").endswith("[页脚：本页及以上旧段为历史参考（不作为当前任务）；当前任务以最新 user 段为准]"), \
        "当前最新页 c#2 无页脚（§2.5 只对旧页生效）"


# ─────────────────────────── T5 55% 分页面字节不变（存根只在 80/90 逐出点发生）───────────────────────────

def test_55pct_paging_pages_bytes_unchanged_by_stub_line(tmp_path):
    """T5：55% 分页器不动（存根只 80/90 逐出点发生）——无逐出事件时账本 append
    序列页化渲染字节恒稳定：除新页外全部旧页字节不变（前缀缓存命中除最后一页
    外全部旧页；55% 分页面零存根、零 notes 翻转）。"""
    lg = CompactLedger(str(tmp_path / "t5.compact.json"))
    for _i in range(3):
        lg.append(f"#1 P0 [L3]\n55分页块{_i}")
    _prev = lg.text()
    lg.append("#1 P0 [L3]\n55分页块3")
    _now = lg.text()
    # 无存根（80/90 未发生）：渲染面零 evicted 行
    assert "evicted" not in _now, "55% 分页面无逐出存根（存根只 80/90 逐出点）"
    # 无 settled 翻转：旧页正文恒不变（前缀缓存命中除最后一页外全部旧页；
    # 变旧页那一刻追加 1 行页脚，v6.1 §2.5）
    for _tag in ("c#1", "c#2", "c#3"):
        assert _page(_now, _tag).startswith(_page(_prev, _tag)), \
            f"旧页 {_tag} 正文不变（55% 分页面）"
    assert "55分页块3" in _page(_now, "c#4"), "新页只装自己（55% 分页器语义零变化）"
    assert not _page(_now, "c#4").endswith("[页脚：本页及以上旧段为历史参考（不作为当前任务）；当前任务以最新 user 段为准]"), \
        "当前最新页 c#4 无页脚（§2.5 只对旧页生效）"
    # 55% 分页面 notes 区全 open（无翻转）：前缀缓存唯一刷位 = 新页 notes + 尾滚动
    assert all("open" in _notes_line(_page(_now, _t))
               for _t in ("c#2", "c#3", "c#4")), "无翻转 → 各页 notes 全 open"


# ─────────────────────────── 契约迁移（旧累计超集 → 页化锚）───────────────────────────

def test_ledger_paging_prefix_anchor_migration(tmp_path):
    """契约迁移锁（有意，非回归）：旧契约「c2 是 c1 严格字节前缀」（累计超集
    重渲）已被页化替换为「除最后一页外全部旧页字节不变 + 新页只装自己 + notes
    区跨页一行索引」——本测试钉死新锚，防误回退累计超集渲染。"""
    lg = CompactLedger(str(tmp_path / "m.compact.json"))
    lg.append("#1 P0 [L3]\n段一")
    c1 = lg.text()
    lg.append("#1 P0 [L3]\n段二")
    c2 = lg.text()
    # 旧契约（累计超集：旧块全文在 c#2 块内重渲）已死：
    # 新契约（v6.1 锚）= 旧页**自身页面字节**（MARK + 该页段）是 c2 的严格前缀
    # （前缀缓存锚），但 c2 不是 c1 全段严格前缀（c#2 页带 notes 区，字节不同）
    assert _page(c1, "c#1") + "\n---" in c2, "旧页 c#1 段是 c2 前缀面（前缀缓存锚）"
    # 累计超集契约迁移锁：旧渲染面 c#2 块内重渲 c#1 全文（旧测试 c2.startswith(c1)
    # 依赖的超集形态）→ 页化后 c#1 正文只在自己的页里，c#2 页不重吐（新锚）
    assert "段一" not in _page(c2, "c#2"), "c#2 页不重吐 c#1 正文（累计超集已死，页化新锚）"
    # 新契约（v6.1 锚）：旧页正文冻结 + 新页只装自己（变旧页那一刻追加 1 行页脚，§2.5）
    assert _page(c2, "c#1").startswith(_page(c1, "c#1")), "旧页 c#1 正文恒不变（前缀缓存锚）"
    assert not _page(c2, "c#2").endswith("[页脚：本页及以上旧段为历史参考（不作为当前任务）；当前任务以最新 user 段为准]"), \
        "当前最新页 c#2 无页脚（§2.5 只对旧页生效）"
    assert "段二" in _page(c2, "c#2") and "段一" not in _page(c2, "c#2"), \
        "新页 c#2 只装自己 entries"
    assert "notes c#1=open" in _page(c2, "c#2"), "新页 notes 区跨页索引（不重吐 c#1 全文）"


def test_evicted_stub_in_page_entries_not_notes(tmp_path):
    """T10 补面：evicted 存根行与 notes 区同页渲染（entries 面 vs notes 面分离），
    且 evicted 块不占条数帽位（_full_block_count 口径含 evicted）。"""
    lg = CompactLedger(str(tmp_path / "n.compact.json"),
                       max_tokens=10_000, max_entries=2)
    e1 = lg.append("#1 P0 [L3]\n全文块一", trimmable=True)
    e2 = lg.append("#1 P1 [L3]\n全文块二", trimmable=True)
    _full = lg._full_block_count()
    lg.evict_page(e1["seq"], "P0", "a1")
    lg.evict_page(e2["seq"], "P1", "a2")
    assert lg._full_block_count() == 0, "evicted 块不占条数帽位（同 trimmed 口径）"
    out = lg.text()
    p1 = _page(out, "c#1")
    assert "evicted sha256=" in p1 and "notes" in p1, "存根行与 notes 区同页正交渲染"
    assert _full >= 2, "折存根前 2 个全文块"


# ─────────────────────────── LOW-3：text 预览面 notes 快照与 append 面同口径 ───────────────────────────

def test_text_preview_notes_snapshot_matches_append(tmp_path):
    """LOW-3（reviewer B）：text(new_text=...) 预览面的 notes 快照（setdefault 口径）
    与同参数 append 写面渲染逐字节一致——预览面 == append 面，锁死 setdefault 语义；
    settle 翻转后预览面 notes 随账本更新（不冻结），锁死'预览面 = 当时快照'语义。"""
    lg = CompactLedger(str(tmp_path / "low3.compact.json"))
    # 构造 c#1、c#2（均 settled=False）
    lg.append("#1 P0 [L3]\n段一原文", settled=False)
    lg.append("#1 P0 [L3]\n段二原文", settled=False)

    # 预览面：text(new_text=c#3 预览文本)
    _preview_text = "#1 P0 [L3]\n段三原文"
    _preview = lg.text(new_text=_preview_text)
    _prev_c3 = _page(_preview, "c#3")
    assert _prev_c3, "预览面 c#3 页在场"
    # 预览面 c#3 notes = 当时前 2 页 settled 快照（c#1=open c#2=open）
    _prev_notes = _notes_line(_prev_c3)
    assert "c#1=open" in _prev_notes, "预览面 c#3 notes = c#1 open（当时快照）"
    assert "c#2=open" in _prev_notes, "预览面 c#3 notes = c#2 open（当时快照）"

    # 实入账本：append 同参数 c#3 → text() 渲染 c#3 notes 区
    _actual = lg.append(_preview_text)
    assert _actual is not None, "append c#3 成功"
    _rendered = lg.text()
    _rendered_c3 = _page(_rendered, "c#3")
    _rendered_notes = _notes_line(_rendered_c3)
    # 预览面 c#3 notes == append 面 c#3 notes（逐字节，setdefault 口径锁死）
    assert _prev_notes == _rendered_notes, (
        f"预览面 notes 与 append 面 notes 逐字节不一致：\n"
        f"  预览面: {_prev_notes!r}\n  append面: {_rendered_notes!r}")

    # settle c#1 → 预览面/append 面都体现 c#1=settled（settled 投影随账本更新，预览不冻结）
    lg.settle(1)
    _preview2 = lg.text(new_text="#1 P0 [L3]\n段四原文")
    _prev_c4 = _page(_preview2, "c#4")
    _prev_notes2 = _notes_line(_prev_c4)
    assert "c#1=settled" in _prev_notes2, "settle c#1 后预览面 notes 体现 c#1=settled（随账本更新，不冻结）"
    assert "c#2=open" in _prev_notes2 and "c#3=open" in _prev_notes2, "其余页 open 状态保留"
    # append 面 c#4 notes 也体现 c#1=settled（同源口径）
    _actual2 = lg.append("#1 P0 [L3]\n段四原文")
    if _actual2 is not None:
        _rendered2 = lg.text()
        _rendered_c4 = _page(_rendered2, "c#4")
        _rendered_notes2 = _notes_line(_rendered_c4)
        assert "c#1=settled" in _rendered_notes2, "append 面 c#4 notes 也体现 c#1=settled"
