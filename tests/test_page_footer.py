"""§2.5 页脚：每页页尾 1 行固定「历史参考」指令（治漂移：模型读旧页时旧页自己标着参考）。

设计（owner 2026-10-16 拍）：
- 每页 c#N 段尾（body 行之后、下一页头之前）append 1 行固定 PAGE_FOOTER；
- 文案固定（不引用页号 N）→ 各页字节逐字节相同 → 旧页生成后字节冻结（含页脚，
  铁律 4 不破：旧页 c#N 不回改，只有新页 c#N+1 是纯新字节 = 1 次可控刷位）；
- 推理面降级：模型读 c#1（漂移重灾区，最早那批 P0）时，c#1 自己的页脚就在几 token
  之内（同页尾）→ 指令贴身生效 → 旧 P0 段被当场标"历史参考、不作为任务"；
- token 面零扰动：不碰 ③ P0 push cap 15% / ④ 逃生门 / ⑤ cache-aware / settled 折存根。

前缀缓存面：页脚是静态逐页 append → 整条账本消息仍 append-only、旧页段（含页脚）
是新一段渲染的严格前缀（c2.startswith(c1) 不破）；每次 compact 的刷位仍是新页段。

测试面（RED 4）：
F1 每页 c#N 含 PAGE_FOOTER 且是页段最后一行
F2 旧页字节冻结（含页脚）：append c#3 后 c#1/c#2 页段（含页脚）字节不变
F3 前缀缓存锚：c2.startswith(c1)（页脚静态 append-only，不破严格前缀）
F4 单页 generation=0 时 c#1 页脚也含该行
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

def test_every_page_has_static_footer(tmp_path):
    lg = CompactLedger(str(Path(tmp_path) / "f1.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    lg.append("#1 P0 [L3]\n段二原文")
    out = lg.text()
    pages = _pages(out)
    assert len(pages) == 2, "两页"
    for _p in pages:
        assert _last_line(_p) == PAGE_FOOTER, "页脚是每页最后一行（固定文本）"
        assert PAGE_FOOTER in _p, "每页含 PAGE_FOOTER"


def test_footer_text_is_fixed_not_paged(tmp_path):
    """页脚文案不引用页号（各页逐字节相同 → 旧页字节冻结前提）。"""
    assert "c#1" not in PAGE_FOOTER and "c#2" not in PAGE_FOOTER, "页脚固定文案不引用页号"
    assert not re.search(r"c#\d", PAGE_FOOTER), "页脚固定文案不含 c#N 页号"


def test_single_page_has_footer(tmp_path):
    """F4：单页（generation=0、只有 c#1）时 c#1 页脚也在（历史参考自标）。"""
    lg = CompactLedger(str(Path(tmp_path) / "f4.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    out = lg.text()
    p1 = _page(out, "c#1")
    assert p1, "c#1 页在场"
    assert _last_line(p1) == PAGE_FOOTER, "单页 c#1 页脚含历史参考指令"


# ─────────────────────────── F2 旧页字节冻结（含页脚）───────────────────────────

def test_old_page_bytes_frozen_with_footer(tmp_path):
    lg = CompactLedger(str(Path(tmp_path) / "f2.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    lg.append("#1 P0 [L3]\n段二原文")
    out = lg.text()
    p1, p2 = _page(out, "c#1"), _page(out, "c#2")
    assert p1 and p2
    lg.append("#1 P0 [L3]\n段三原文")
    out2 = lg.text()
    assert _page(out2, "c#1") == p1, "旧页 c#1（含页脚）字节恒不变（铁律 4）"
    assert _page(out2, "c#2") == p2, "旧页 c#2（含页脚）字节恒不变（铁律 4）"


# ─────────────────────────── F3 前缀缓存锚（严格前缀不破）───────────────────────────

def test_prefix_stable_with_footer(tmp_path):
    lg = CompactLedger(str(Path(tmp_path) / "f3.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    c1 = lg.text()
    lg.append("#1 P0 [L3]\n段二原文")
    c2 = lg.text()
    assert c2.startswith(c1) or c2.startswith(c1.rstrip("\n---")), \
        "页脚静态 append-only → c#1 段（含页脚）是 c#2 渲染的严格前缀（前缀缓存不破）"


def test_footer_not_in_notes_area(tmp_path):
    """页脚（页尾）与 notes 区（页顶）正交：notes 行不被页脚面扰动。"""
    lg = CompactLedger(str(Path(tmp_path) / "fx.compact.json"))
    lg.append("#1 P0 [L3]\n段一原文")
    lg.append("#1 P0 [L3]\n段二原文")
    out = lg.text()
    p1, p2 = _page(out, "c#1"), _page(out, "c#2")
    _l1, _l2 = p1.splitlines(), p2.splitlines()
    assert "notes: -" in _l1, "c#1 notes 区占位不变（页脚面正交）"
    assert "notes c#1=open" in _l2, "c#2 notes 区跨页快照不变（页脚面正交）"
    # 页脚在页尾（notes 区之后），不被 notes 面扰动位置
    assert _l1.index(PAGE_FOOTER) > _l1.index("notes: -"), "页脚在 notes 区之后（页尾）"
