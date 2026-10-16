"""staged 窗口账本 dedup（2026-10-16 · 治「未 settle P0 段每次 compact 重吐进新页」
+ 账本头自嵌段漏网）：

TDD 锚点（owner 裁定 1+2 修复面）：
- `_staged_ledger_dedup(rendered, ledger)`：窗口 P0/P1 kept 段里，全局 seq 已在活账本
  `_evict_seq_index` 内 → 剔（前页已记全文，新页只靠 notes 区记账）；
- 段文本 = 账本块头 / staged 头自嵌段（re-feed 污染）→ 剔；
- 未 settle 的**新**段（seq 不在账本内）→ 保留（不误杀）。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import _staged_ledger_dedup
from sc.compact_ledger import CompactLedger
from sc.distill import WindowSeg
from sc.schema import Segment


class _FakeEngine:
    def __init__(self, lg):
        self._lg = lg
        self._last_recall_page_seqs: list = []
        self._last_discarded_page_texts: dict = {}
        self._last_recall_seg_ids: set = set()
        self._recall_page_seqs_override: list = []
    def _evict_seq_index(self, ledger):
        # 复刻真引擎实现：行首段标记位整段行形
        _idx = {}
        try:
            _seq_re = re.compile(r"(?m)^#\d+ \w+ prov=\S+ seq=(\d+)")
            for _e in ledger.entries:
                for _m in _seq_re.finditer(_e.get("text", "")):
                    _idx[_m.group(1)] = _e["seq"]
        except Exception:
            _idx = {}
        return _idx
    def _recall_global_seqs(self):
        # 复刻真引擎 _recall_global_seqs：_last_recall_seg_ids → 全局 seq
        # （fake 用 override 直给，免 store.replay() 面）
        return set(self._recall_page_seqs_override)
    def _recall_page_seqs(self, ledger, global_seqs):
        # 复刻真引擎：全局 seq → 账本页 seq（_evict_seq_index 同映射）
        _drop = []
        for _g, _ps in self._evict_seq_index(ledger).items():
            if int(_g) in {int(x) for x in global_seqs}:
                _drop.append(_ps)
        return _drop


def _item(seq, tier, text, prov="user"):
    seg = Segment(tier=tier, provenance=prov, original_text=text, seq=seq,
                  session_topic="", model_version="m1")
    return (WindowSeg(seg), "L3", text)


class TestStagedLedgerDedup:
    def test_segs_already_in_ledger_are_dropped(self, tmp_path):
        """已在活账本内（_evict_seq_index 命中）的全局 seq 段 → 新 staged 块不再重吐。"""
        lg = CompactLedger(str(tmp_path / "s.compact.json"))
        # c#1 里记了 seq=5（P0 段，staged 块行形 `#1 P0 prov=user seq=5 [L3]`）
        lg.append("#1 P0 prov=user seq=5 [L3]\n已记段全文")
        # 窗口再次出现 seq=5（未 settle 全文保留在窗口 = 55% 闸正确行为）+ 新段 seq=6
        rendered = [
            _item(5, "P0", "已记段全文"),
            _item(6, "P0", "新段全文"),
        ]
        out = _staged_ledger_dedup(rendered, lg, _FakeEngine(lg))
        kept_seqs = [getattr(getattr(ws, "segment", None), "seq", None) for ws, _f, _t in out]
        assert 6 in kept_seqs, "新段 seq=6 保留（dedup 不误杀）"
        assert 5 not in kept_seqs, "已记账本 seq=5 重吐段被剔（新页不重吐 c#1 全文）"

    def test_ledger_header_self_embed_dropped(self, tmp_path):
        """re-feed 污染段（文本 = 账本块头 `[supercompress compact-ledger]…`）→ 不进块。"""
        lg = CompactLedger(str(tmp_path / "s2.compact.json"))
        lg.append("#1 P0 prov=user seq=1 [L3]\n正常段")
        polluted = _item(9, "P0", "[supercompress compact-ledger]\n── 页 c#1 ──\n...")
        fresh = _item(10, "P0", "正常新段")
        out = _staged_ledger_dedup([polluted, fresh], lg, _FakeEngine(lg))
        kept_seqs = [getattr(getattr(ws, "segment", None), "seq", None) for ws, _f, _t in out]
        assert 9 not in kept_seqs, "账本头自嵌段被剔（re-feed 污染不进 P0 块）"
        assert 10 in kept_seqs, "正常新段保留"

    def test_fresh_segs_all_kept_when_ledger_empty(self, tmp_path):
        """活账本空（首 compact）→ 所有段原样保留（dedup 零扰动首块积累）。"""
        lg = CompactLedger(str(tmp_path / "s3.compact.json"))
        rendered = [_item(1, "P0", "段一"), _item(2, "P0", "段二")]
        out = _staged_ledger_dedup(rendered, lg, _FakeEngine(lg))
        assert len(out) == 2, "空账本 dedup = 全保留"

    def test_dedup_survives_page_discard(self, tmp_path):
        """P0-1（T4 防递归闭环）：旧账本页被 recall 暂存清算 discard 之后，
        该页内的段重新 staged 时 dedup 必须剔除——否则旧信息每次召回后
        都重新灌进新页（活账本递归堆积的源头）。

        真引擎面：T3 清算点（engine.compress）discard 旧页并登记
        `_last_recall_page_seqs`，随后 `_staged_ledger_dedup` 的 dedup 面
        = 活账本 index ∪ 登记页（discard 当轮的过渡面：discard 后该轮
        staged 块仍可能重吐被丢页段，下轮起不再重吐）。"""
        lg = CompactLedger(str(tmp_path / "s4.compact.json"))
        lg.append("#1 P0 prov=user seq=5 [L3]\n召回后旧页段")
        # 模拟 T3 清算：快照被丢页 text + 旧账本页（entry seq=1）被 discard + 登记
        eng = _FakeEngine(lg)
        eng._last_discarded_page_texts = {1: "#1 P0 prov=user seq=5 [L3]\n召回后旧页段"}
        lg.discard_entries([1])
        eng._last_recall_page_seqs = [1]
        rendered = [_item(5, "P0", "召回后旧页段"), _item(6, "P0", "真新段")]
        out = _staged_ledger_dedup(rendered, lg, eng)
        kept_seqs = [getattr(getattr(ws, "segment", None), "seq", None)
                     for ws, _f, _t in out]
        assert 6 in kept_seqs, "真新段保留"
        assert 5 not in kept_seqs, "被丢页段该轮仍被 dedup 剔除（discard 过渡面不重吐）"
        # 下轮：登记面随 compact 消费清掉（compress 结算点
        # _last_recall_page_seqs=[] + _last_discarded_page_texts={}）
        eng._last_recall_page_seqs = []
        eng._last_discarded_page_texts = {}
        out2 = _staged_ledger_dedup(rendered, lg, eng)
        kept2 = [getattr(getattr(ws, "segment", None), "seq", None)
                 for ws, _f, _t in out2]
        assert 5 in kept2, "清算后登记面清掉 → 被丢页段若仍在窗口 = store→窗口→staged 正常流，合法重入账（T3 纪律）"
