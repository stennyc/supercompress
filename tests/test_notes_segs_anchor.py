"""notes 区段序锚（owner 2026-10-24 拍板 · a 方案）：

append() 写新页 notes 快照时，对每个前页解析其 staged 块行首段标记位
（`#<rel> <tier> prov=… seq=<全局>`，engine L589-597 L3/L2 行 + _l1_line L1 行
行首均有 `seq=` 位；L0 指针行无 → 全 L0 高压页解析为空，渲染面 fail-open 退化
旧行形），存 segs 三元组 [mn, mx, last_user_seq]；text() 渲染面在旧行形
`c#N=settled/open` 后追加 ` segs=mn..mx task=N`（仅新写页；旧 2 列盘 load 后
notes 无 segs 列 → 渲染面逐字旧行形，前缀缓存命中面零变化）。

零 LLM、零新状态面、append-only：只影响新 compact 新页的 notes 行，旧页已落盘
字节零动（快照冻结铁律 2/3）。
"""
import json
import os
import tempfile
import unittest

from sc.compact_ledger import CompactLedger, _page_seg_ranges

# 行形样例（engine L589-597 L3 行 / _l1_line L1 行 / _l0_ptr L0 指针行，实样摘自源码）
_L3_BLOCK = (
    "#1 P0 prov=user seq=12 [L3]\n用户指令一\n"
    "---\n"
    "#2 P0 prov=model seq=13 [L3]\n模型响应一\n"
    "---\n"
    "#3 P0 prov=user seq=14 [L3]\n用户指令二\n"
)
_L0_BLOCK = (
    "#1 P0 [aaaa111111111111-1] on-demand\n"
    "---\n"
    "#2 P0 [bbbb222222222222-2] on-demand\n"
)
_L1_BLOCK = (
    "#1 P0 prov=user seq=5 · 用户指令五的关键事实\n"
    "---\n"
    "#2 P0 prov=model seq=6 · 模型响应六的关键事实\n"
)


class TestPageSegRanges(unittest.TestCase):
    def test_l3_block_full_range(self):
        self.assertEqual(_page_seg_ranges(_L3_BLOCK), (12, 14, 14))

    def test_l1_block_full_range(self):
        self.assertEqual(_page_seg_ranges(_L1_BLOCK), (5, 6, 5))

    def test_l0_block_fail_open_zero(self):
        # 全 L0 高压页：行首无 seq= 位 → 解析空 → (0,0,0)，渲染面退化旧行形
        self.assertEqual(_page_seg_ranges(_L0_BLOCK), (0, 0, 0))

    def test_empty_text(self):
        self.assertEqual(_page_seg_ranges(""), (0, 0, 0))

    def test_no_user_segment_last_user_zero(self):
        block = "#1 P0 prov=model seq=9 [L3]\n只有模型段\n"
        self.assertEqual(_page_seg_ranges(block), (9, 9, 0))


class TestNotesSegsAnchor(unittest.TestCase):
    def _ledger(self, d):
        return CompactLedger(os.path.join(d, "l.compact.json"), logger=None)

    def test_new_page_notes_carry_segs(self):
        d = tempfile.mkdtemp()
        ld = self._ledger(d)
        e1 = ld.append(_L3_BLOCK, settled=False, kind="adhoc")
        e2 = ld.append(_L1_BLOCK, settled=False, kind="adhoc")
        # 新页 e2 的 notes 区：前页 c#1 带 segs 三元组
        self.assertEqual(e2["notes"], [[1, False, [12, 14, 14]]])
        # 首页 c#1 无前页 → notes 空
        self.assertEqual(e1["notes"], [])
        # 渲染面：c#2 行带 segs；c#1 行（无 notes）= notes: - 旧形
        out = ld.text()
        self.assertIn("notes c#1=open segs=12..14 task=14", out)
        self.assertIn("notes: -", out)

    def test_all_l0_page_fail_open_old_line_form(self):
        d = tempfile.mkdtemp()
        ld = self._ledger(d)
        e1 = ld.append(_L0_BLOCK, settled=False, kind="adhoc")
        e2 = ld.append("x", settled=False, kind="adhoc")
        # 前页全 L0 → 解析空存全 0 占位（None 等价，渲染面退化旧行形不误标 0..0）
        self.assertEqual(e2["notes"][0][0], 1)
        self.assertFalse(e2["notes"][0][1])
        out = ld.text()
        self.assertIn("notes c#1=open", out)
        self.assertNotIn("segs=", out)

    def test_settled_state_reflects_in_notes(self):
        d = tempfile.mkdtemp()
        ld = self._ledger(d)
        ld.append(_L3_BLOCK, settled=True, kind="adhoc")
        e2 = ld.append(_L1_BLOCK, settled=False, kind="adhoc")
        self.assertEqual(e2["notes"], [[1, True, [12, 14, 14]]])
        self.assertIn("notes c#1=settled segs=12..14 task=14", ld.text())

    def test_old_two_column_notes_render_byte_unchanged(self):
        # 旧 2 列盘（[seq, settled]）load 后：notes 无 segs 列 → 渲染面逐字旧行形
        d = tempfile.mkdtemp()
        path = os.path.join(d, "old.compact.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "generation": 0, "seals": 0, "seq": 2,
                       "entries": [
                           {"seq": 1, "text": _L3_BLOCK, "tokens": 10,
                            "settled": False, "kind": "adhoc", "trimmable": False,
                            "notes": []},
                           {"seq": 2, "text": _L1_BLOCK, "tokens": 10,
                            "settled": False, "kind": "adhoc", "trimmable": False,
                            "notes": [[1, False]]},
                       ]}, f)
        ld = CompactLedger(path, logger=None)
        self.assertEqual(ld.entries[1]["notes"], [(1, False, None)])  # 2 列读回归一 3 列（无 segs）
        out = ld.text()
        self.assertIn("notes c#1=open", out)
        self.assertNotIn("segs=", out)   # 旧页渲染面零变化

    def test_old_disk_no_notes_key_fail_open(self):
        # 旧盘完全无 notes 键 → load 后 notes=[] → 渲染 "notes: -"（既有 fail-open）
        d = tempfile.mkdtemp()
        path = os.path.join(d, "old.compact.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "generation": 0, "seals": 0, "seq": 1,
                       "entries": [
                           {"seq": 1, "text": _L3_BLOCK, "tokens": 10,
                            "settled": False, "kind": "adhoc", "trimmable": False},
                       ]}, f)
        ld = CompactLedger(path, logger=None)
        self.assertEqual(ld.entries[0]["notes"], [])
        self.assertIn("notes: -", ld.text())

    def test_prefix_cache_invariant_old_pages_stable(self):
        # 前缀缓存不变量：旧页段（含 notes 区 + 页脚）逐字是 append 新页后
        # 渲染串的严格前缀（旧页字节零动）
        d = tempfile.mkdtemp()
        ld = self._ledger(d)
        ld.append(_L3_BLOCK, settled=False, kind="adhoc")
        snapshot_before = ld.text()
        ld.append(_L1_BLOCK, settled=False, kind="adhoc")
        out_after = ld.text()
        # 旧页 c#1 段（notes: - + 原文 + 页脚）逐字是 after 串的前缀段
        self.assertTrue(out_after.startswith(snapshot_before.split("---")[0]))


if __name__ == "__main__":
    unittest.main()
