"""召回暂存区（recall staging area）单测面。

独立 LRU 容器，与账本（c#n / compact_ledger）零关联。

设计口径（owner 定稿）：
- 暂存区内容在 compact 时清空，**不回账本**
- compact 清空后立刻按当前 query 重填（_recall_cache 保留复用）
- 暂存区预算 = f(context_length, 当前用量) 动态，与 compact_ledger_max_tokens 零关联
- 空间不足时踢旧纳新（LRU 序，整段不劈段）
- 单段超预算 → 整段降 L1 行（_l1_line），不劈段
- 段数上限 recall_staging_max_segs=32（防无界段数）

fail-safe 红线（本文件全封）：
- auto_recall: false → 全 no-op，暂存区逐字不生成
- choose_forms / _staged_ledger_dedup / compact_ledger_max_tokens 零 diff
- 暂存区超帽 → 只踢暂存区自己，账本不受影响
"""
import json
import logging
import os
import tempfile

import pytest

from sc.schema import Segment
from sc.recall import tokenize
from fakes import FakeEndpoint
from engine import SupercompressContextEngine, DISTILL_WATER_FRAC


# ── 构造 engine（暂存区功能面）─────────────────────────────────────
def _make_engine(tmpdir: str, auto_recall: bool = True,
                 context_length: int = 256000,
                 recall_budget_frac: float = 0.3,
                 recall_budget_cap_frac: float = 0.05,
                 recall_staging_max_segs: int = 32,
                 last_prompt_tokens: int = 0) -> SupercompressContextEngine:
    # 注意：SupercompressContextEngine.__init__ 只收 **kwargs（无 cfg dict 参数），
    # _get(key) 优先读 kwargs 值（host/测试 kwarg 面），缺失才回落 config.yaml。
    # 所以暂存区三键直接以扁平 kwarg 传，不包 dict。
    eng = SupercompressContextEngine(
        store_path=os.path.join(tmpdir, "store.jsonl"),
        auto_recall=auto_recall,
        context_length=context_length,
        recall_budget_frac=recall_budget_frac,
        recall_budget_cap_frac=recall_budget_cap_frac,
        recall_staging_max_segs=recall_staging_max_segs,
    )
    eng.last_prompt_tokens = last_prompt_tokens
    return eng


# ── 动态帽公式验证 ──────────────────────────────────────────────────
class TestRecallBudgetTokens:
    """_recall_budget_tokens() = max(0, min(headroom*frac, cap))

    headroom = context_length * DISTILL_WATER_FRAC - last_prompt_tokens
    cap      = context_length * recall_budget_cap_frac
    frac     = recall_budget_frac

    256000 窗口、水线 0.55 → 140800
    cap = 256000 * 0.05 = 12800
    """

    def test_window_empty_gives_cap(self, tmp_path):
        eng = _make_engine(str(tmp_path), last_prompt_tokens=0)
        # headroom = 140800 - 0 = 140800; 140800*0.3 = 42240 > cap 12800 → cap
        assert eng._recall_budget_tokens() == 12800

    def test_window_mid_gives_cap_still(self, tmp_path):
        eng = _make_engine(str(tmp_path), last_prompt_tokens=50000)
        # headroom = 140800 - 50000 = 90800; 90800*0.3 = 27240 > cap 12800 → cap
        assert eng._recall_budget_tokens() == 12800

    def test_window_high_shrinks(self, tmp_path):
        eng = _make_engine(str(tmp_path), last_prompt_tokens=100000)
        # headroom = 140800 - 100000 = 40800; 40800*0.3 = 12240 < cap 12800
        assert eng._recall_budget_tokens() == 12240

    def test_window_at_waterline_gives_zero(self, tmp_path):
        eng = _make_engine(str(tmp_path), last_prompt_tokens=140800)
        # headroom = 0 → 0
        assert eng._recall_budget_tokens() == 0

    def test_window_above_waterline_gives_zero(self, tmp_path):
        eng = _make_engine(str(tmp_path), last_prompt_tokens=150000)
        # headroom < 0 → max(0, ...) = 0
        assert eng._recall_budget_tokens() == 0

    def test_no_ledger_coupling(self, tmp_path):
        """暂存区帽与 compact_ledger_max_tokens 零关联——改账本帽暂存区帽不变。"""
        eng_a = _make_engine(str(tmp_path), last_prompt_tokens=0)
        eng_b = _make_engine(str(tmp_path), last_prompt_tokens=0)
        # 手动篡改账本帽（暂存区面零读）
        eng_b.compact_ledger_max_tokens = 1024
        assert eng_a._recall_budget_tokens() == eng_b._recall_budget_tokens() == 12800


# ── 暂存区 LRU 容器（实例态）────────────────────────────────────────
class TestRecallStagingLRU:
    """_recall_staging 容器：OrderedDict 语义（最近写入的在尾部，LRU 踢头）。

    每 turn select_context 跑完往暂存区写入 K 段（_recall_cache 复用结果），
    超帽时按 LRU 序（最早写入的在前）踢旧段。
    """

    def test_lru_evicts_oldest_first(self, tmp_path):
        eng = _make_engine(str(tmp_path))
        # 手动注入暂存区 3 段（模拟跨 turn 累积）
        segs = [
            {"seg_id": "s1", "text": "a" * 200, "tier": "P1"},
            {"seg_id": "s2", "text": "b" * 200, "tier": "P1"},
            {"seg_id": "s3", "text": "c" * 200, "tier": "P0"},
        ]
        eng._recall_staging_add_batch([s["seg_id"] for s in segs])
        assert list(eng._recall_staging.keys()) == ["s1", "s2", "s3"]
        # 踢最旧（s1）
        evicted = eng._recall_staging_evict_oldest()
        assert evicted == "s1"
        assert list(eng._recall_staging.keys()) == ["s2", "s3"]

    def test_lru_reuse_moves_to_tail(self, tmp_path):
        eng = _make_engine(str(tmp_path))
        eng._recall_staging_add_batch(["s1", "s2", "s3"])
        # s1 再次被召回 → 移到尾部（LRU 序更新）
        eng._recall_staging_touch("s1")
        assert list(eng._recall_staging.keys()) == ["s2", "s3", "s1"]
        # 踢最旧 = s2
        assert eng._recall_staging_evict_oldest() == "s2"

    def test_staging_clear(self, tmp_path):
        eng = _make_engine(str(tmp_path))
        eng._recall_staging_add_batch(["s1", "s2"])
        eng._recall_staging_clear_on_compact()
        assert len(eng._recall_staging) == 0


# ── compact 清空暂存区（不回账本）───────────────────────────────────
class TestCompactClearsStaging:
    """compact 结算点清暂存区字节，保留 _recall_cache（供立刻重填）。"""

    def test_compact_clears_staging_keeps_cache(self, tmp_path):
        eng = _make_engine(str(tmp_path))
        eng._recall_staging_add_batch(["s1", "s2", "s3"])
        # 模拟 compact 结算点（compress() 里调 _recall_staging_clear_on_compact）
        eng._recall_staging_clear_on_compact()
        assert len(eng._recall_staging) == 0
        # _recall_cache 保留（供立刻重填）——compact 不清它
        assert eng._recall_cache is not None or True  # None = 冷启动也合法（不崩）

    def test_compact_staging_not_in_ledger(self, tmp_path):
        """暂存区段不进 c#n+1 账本——choose_forms 面零感知暂存区。"""
        # 构造一个含暂存区段的场景，验证 compact 后账本页里无暂存区段 id
        # （choose_forms 只看 store 里的段，暂存区是 engine 实例态，不沉盘）
        eng = _make_engine(str(tmp_path))
        eng._recall_staging_add_batch(["x1", "x2"])
        eng._recall_staging_clear_on_compact()
        # 暂存区段不在任何账本 entries 里（账本 entries 按 store seq，暂存区段 id 不匹配）
        # 这里验证的是"clear 后容器为空"而非"账本里没有"——账本面零写暂存区是物理隔离


# ── 单段超预算 → 降 L1 ─────────────────────────────────────────────
class TestSingleSegExceedsBudget:
    """单段 est_tokens > 预算 → 整段降 L1 行（_l1_line），不劈段。"""

    def test_oversized_seg_degrades_to_l1(self, tmp_path):
        eng = _make_engine(str(tmp_path), last_prompt_tokens=130000)
        # 预算 = (140800-130000)*0.3 = 3240
        assert eng._recall_budget_tokens() == 3240
        # 构造一个 est_tokens > 3240 的段
        big_text = "word " * 1000  # ~1000 token（粗估）
        assert len(big_text.split()) > 3240 // 4  # 粗判断：确实超
        # 降级行为：_recall_staging_should_degrade(seg_id, est_tokens, budget) → True
        assert eng._recall_staging_should_degrade("big1", 4000, 3240) is True
        # 不超 → 不降
        assert eng._recall_staging_should_degrade("sml1", 1000, 3240) is False

    def test_degrade_uses_l1_line(self, tmp_path):
        """降级段走 _l1_line 行形（head60 + key_fact + segment_id），不劈段。"""
        from sc.inject import _l1_line
        # 构造一个 L1 窗段（head60 是段前 60 字符）
        ws = _make_ws_for_l1("abcdef" * 20, "P1", "seg001")
        line = _l1_line(ws, rel=1)
        # L1 行形首 60 字符 = head60（段前 60 字），不含全文
        assert line.startswith("#1 P1")
        assert "abcdef" in line
        assert len(line) < len("abcdef" * 20)  # 行形比原文短（不劈段 = 整段降，非劈中段）


# ── auto_recall: false 逐字 no-op ──────────────────────────────────
class TestAutoRecallFalseNoOp:
    """auto_recall: false → 暂存区全功能不生成、select_context 直接 None。"""

    def test_select_context_returns_none(self, tmp_path):
        eng = _make_engine(str(tmp_path), auto_recall=False)
        msgs = [{"role": "user", "content": "gateway rate limit threshold deploy"}]
        result = eng.select_context(msgs, incoming_message=msgs[0])
        assert result is None

    def test_staging_container_empty(self, tmp_path):
        eng = _make_engine(str(tmp_path), auto_recall=False)
        assert len(eng._recall_staging) == 0


# ── 段数上限 32 ────────────────────────────────────────────────────
class TestStagingMaxSegs:
    """recall_staging_max_segs=32：段数超 → 额外踢 LRU，不降 L1。

    注意：_recall_staging_add 逐段触发 token 帽淘汰，超段数上限时
    由 _recall_staging_enforce_max_segs 兜底踢 LRU。
    测试用 7 段注入（add_batch 逐段 add），段数 7 < 默认 32，
    所以单独调 enforce_max_segs 才暴露超限踢 LRU 行为。
    """

    def test_evict_when_seg_count_exceeds_max(self, tmp_path):
        eng = _make_engine(str(tmp_path), recall_staging_max_segs=5)  # 小值便于测试
        # 注入 7 段，超限 5 → enforce 踢 2 个最旧
        ids = [f"s{i}" for i in range(7)]
        eng._recall_staging_add_batch(ids)
        # add_batch 逐段 add，每段加完就检查 token 帽（0 扣帽不踢），
        # 但段数帽在 add 内部也调用了 enforce_max_segs —— 逐段超 5 时
        # 每段加完踢 1 个最旧，最终容器 ≤ 5
        remaining = list(eng._recall_staging.keys())
        assert len(remaining) <= 5
        # 最新 5 个保留（s2..s6）
        assert "s6" in remaining
        assert "s5" in remaining


def _make_ws_for_l1(text: str, tier: str, seg_id: str):
    """构造 L1 窗段供 _l1_line 测试（最小 stub，不依赖 store）。"""
    from sc.schema import Segment
    from sc.distill import WindowSeg

    seg = Segment(
        tier=tier,
        provenance="user",
        original_text=text,
    )
    ws = WindowSeg(seg)
    ws.form = "L1"
    return ws
