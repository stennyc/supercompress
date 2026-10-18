"""§2.5 页脚：旧页页尾 1 行固定「历史参考」指令（治漂移：模型读旧页时旧页自己标着参考）。

设计（owner 2026-10-16 拍 + 2026-10-18 修正最新页误伤）：
- 旧页 c#1..c#N-1 段尾（body 行之后、下一页头之前）append 1 行固定 PAGE_FOOTER；
- **最新页 c#N（当前任务正文页）不贴页脚**（2026-10-18 owner 拍）——原实现每页
  （含最新页）都贴，模型把最新页也当"历史参考"、把自己刚干的活降级，主观感觉
  "compact 把中间过程压掉了"；修正后只有"有旧页在场"的旧页带页脚，最新页不带；
- 文案固定（不引用页号 N）→ 各旧页字节逐字节相同 → 旧页生成后字节冻结（含页脚，
  铁律 4 不破：旧页 c#N 不回改）；最新页 c#N 本就是每次 compact 新增/变化的页，
  不在旧页冻结面，最新页去页脚零前缀缓存代价；
- 单页账本（只有 c#1 且是最新页）→ 零页脚（当前页不降级）；
- 推理面降级：模型读 c#1（漂移重灾区，最早那批 P0）时，c#1 自己的页脚就在几 token
  之内（同页尾）→ 指令贴身生效 → 旧 P0 段被当场标"历史参考、不作为任务"；
- token 面零扰动：不碰 ③ P0 push cap 15% / ④ R3 pending 兜底 / ⑤ cache-aware / settled 折存根。

前缀缓存面：旧页页脚是静态逐页 append → 旧页段（含页脚）是新一段渲染的严格前缀
（c2.startswith(c1) 不破）；每次 compact 的刷位仍是新页段（最新页无页脚，纯新字节）。

测试面（RED 4 + GREEN 修正 2）：
F1 旧页 c#1..c#N-1 含 PAGE_FOOTER 且是页段最后一行；最新页 c#N 不含 PAGE_FOOTER
F2 旧页字节冻结（含页脚）：append c#3 后 c#1/c#2 页段（含页脚）字节不变
F3 前缀缓存锚：c2.startswith(c1)（页脚静态 append-only，不破严格前缀）
F4 单页（只有 c#1、即最新页）→ 零页脚（当前页不降级）
G1 旧页全带页脚、最新页不带的分布钉死（3 页：c#1/c#2 有、c#3 无）
G2 最新页去页脚后，旧页 c#1/c#2（含页脚）仍是新渲染的严格前缀（前缀缓存面零扰动）
"""
from pathlib import Path
import re

from sc.compact_ledger import CompactLedger, PAGE_FOOTER

PAGE_RE = re.compile(r"── 页 c#\d+ ──")


def _pages(section: str) -> list[str]:
    _marks = list(PAGE_RE.finditer(section))
    _out = []
    for _i, _m in enumerate(_marks):
        _end = _marks[_i + 1].start() if _i + 1 < len(_marks) else len(section)
        _out.append(section[_m.start():_end].rstrip("\n---"))
    return _out


def _page(section: str, tag: str) -> str:
    for _p in _pages(section):
        if _p.startswith(f"── 页 {tag} ──"):
            return _p
    return ""


def _last_line(page: str) -> str:
    return page.splitlines()[-1] if page.splitlines() else ""


# ─────────────────────────── F1 每页页尾固定页脚 ───────────────────────────

def test_every_old_page_has_footer_latest_does_not(tmp_path):
    """F1/G1：旧页 c#1..c#N-1 含 PAGE_FOOTER 且是页段最后一行；最新页 c#N 不含。"""
    lg = CompactLedger(str(Path(tmp_path) / "f1.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    lg.append("#1 P0 [L3]\n段二原文")
    out = lg.text()
    pages = _pages(out)
    assert len(pages) == 2, "两页"
    assert _last_line(pages[0]) == PAGE_FOOTER, "旧页 c#1 最后一行 = PAGE_FOOTER"
    assert PAGE_FOOTER in pages[0], "旧页 c#1 含 PAGE_FOOTER"
    last = pages[1]
    assert PAGE_FOOTER not in last, "最新页 c#2 不含 PAGE_FOOTER（当前任务页不降级）"


def test_footer_text_is_fixed_not_paged(tmp_path):
    """页脚文案不引用页号（各旧页逐字节相同 → 旧页字节冻结前提）。"""
    assert "c#1" not in PAGE_FOOTER and "c#2" not in PAGE_FOOTER, "页脚固定文案不引用页号"
    assert not re.search(r"c#\d", PAGE_FOOTER), "页脚固定文案不含 c#N 页号"


def test_single_latest_page_no_footer(tmp_path):
    """F4：单页（只有 c#1、即最新页）→ 零页脚（当前页不降级）。"""
    lg = CompactLedger(str(Path(tmp_path) / "f4.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    out = lg.text()
    p1 = _page(out, "c#1")
    assert p1, "c#1 页在场"
    assert PAGE_FOOTER not in p1, "单页 = 最新页，不贴页脚（当前任务不降级）"


# ─────────────────────────── F2 旧页字节冻结（含页脚）───────────────────────────

def test_old_page_bytes_frozen_with_footer(tmp_path):
    """F2/G1：旧页 c#1/c#2（含页脚）字节冻结；c#3（最新页）不带页脚。"""
    lg = CompactLedger(str(Path(tmp_path) / "f2.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    lg.append("#1 P0 [L3]\n段二原文")
    out = lg.text()
    p1 = _page(out, "c#1")
    assert p1 and _last_line(p1) == PAGE_FOOTER, "旧页 c#1（含页脚）在场"
    lg.append("#1 P0 [L3]\n段三原文")
    out2 = lg.text()
    assert _page(out2, "c#1") == p1, "旧页 c#1（含页脚）字节恒不变（铁律 4）"
    p2 = _page(out2, "c#2")
    assert _last_line(p2) == PAGE_FOOTER, "此时 c#2 已变旧页 → 页脚补上（旧页面）"
    p3 = _page(out2, "c#3")
    assert PAGE_FOOTER not in p3, "最新页 c#3 不带页脚（当前任务页不降级）"


def test_three_page_footer_distribution(tmp_path):
    """G1：3 页账本 → c#1/c#2（旧页）带页脚、c#3（最新页）不带。"""
    lg = CompactLedger(str(Path(tmp_path) / "g1.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    lg.append("#1 P0 [L3]\n段二原文")
    lg.append("#1 P0 [L3]\n段三原文")
    out = lg.text()
    p1, p2, p3 = _page(out, "c#1"), _page(out, "c#2"), _page(out, "c#3")
    assert _last_line(p1) == PAGE_FOOTER, "旧页 c#1 带页脚"
    assert _last_line(p2) == PAGE_FOOTER, "旧页 c#2 带页脚"
    assert PAGE_FOOTER not in p3, "最新页 c#3 不带页脚"


# ─────────────────────────── F3 前缀缓存锚（严格前缀不破）───────────────────────────

def test_prefix_stable_with_footer(tmp_path):
    """G2：最新页去页脚后，旧页 c#1（含页脚）仍是新渲染的严格前缀（前缀缓存面零扰动）。"""
    lg = CompactLedger(str(Path(tmp_path) / "f3.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    lg.append("#1 P0 [L3]\n段二原文")
    c2 = lg.text()
    p1 = _page(c2, "c#1")
    lg.append("#1 P0 [L3]\n段三原文")
    c3 = lg.text()
    assert c3.startswith(c2) or c3.startswith(c2.rstrip("\n---")), \
        "旧页 c#1/c#2（含页脚）字节冻结 → c3 是 c2 的严格前缀（前缀缓存不破）"
    assert _page(c3, "c#1") == p1, "旧页 c#1（含页脚）跨两次 append 字节恒不变"


def test_footer_not_in_notes_area(tmp_path):
    """页脚（页尾）与 notes 区（页顶）正交：notes 行不被页脚面扰动。"""
    lg = CompactLedger(str(Path(tmp_path) / "fx.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    lg.append("#1 P0 [L3]\n段二原文")
    lg.append("#1 P0 [L3]\n段三原文")
    out = lg.text()
    p1, p2 = _page(out, "c#1"), _page(out, "c#2")
    _l1, _l2 = p1.splitlines(), p2.splitlines()
    assert "notes: -" in _l1, "c#1 notes 区占位不变（页脚面正交）"
    assert "notes c#1=open" in _l2, "c#2 notes 区跨页快照不变（页脚面正交）"
    # 页脚在旧页页尾（notes 区之后），不被 notes 面扰动位置（c#1/c#2 已变旧页）
    assert _l1.index(PAGE_FOOTER) > _l1.index("notes: -"), "旧页 c#1 页脚在 notes 区之后（页尾）"
    assert _l2.index(PAGE_FOOTER) > _l2.index("notes c#1=open"), "旧页 c#2 页脚在 notes 区之后（页尾）"
