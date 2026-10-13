"""P0 · Segment 比较回归面（t_fd3c7177）：sc/recall.py 排序键 TypeError 红线。

背景：live 22.5h+ 慢性 no-op——`supercompress: select_context no-op (recall degraded):
'<' not supported between instances of 'Segment' and 'Segment'`（195 条跨 6 session）。
根因（前序只读卡定位，本卡复核确认）：
- sc/recall.py scored 4-tuple `(-base, tidx[tier], seq, seg)` 整体 `sort()`；
- Segment 为普通 @dataclass（sc/schema.py，无 order=True / __lt__）；
- 三主键（-base, tier, seq）全同的第 4 位 Segment 裸比较 → TypeError →
  engine.py:1163-1164 select_context fail-open 返回 None → recall 块从未发射。
- seq 非全局唯一：_recall_union_window（engine.py:824）主 store + model 旁路 live 段
  并集，两 store 各自 seq 计数 → 同 seq 碰撞（同文 stale 臂同 seq 亦同碰撞面）。
- 触发面（复核实测，CPython timsort）：仅当窗口段数多到 timsort 必须比较两个三主键
  全等的 Segment 才炸——**最小复现 = ≥3 段**（两段全等时 timsort 短路不触第 4 比较，
  是 no-crash guard 而非 red 复现；三段同 seq 同 base 时 timsort 被迫比第 4 位 → 炸）。

修复约束（二选一写死）：排序 key 法——sort 显式传 scalar key，三主键全 tie 时按
segment_id（content_hash[:16]+seq，跨 store 唯一）定全序（不改 Segment 类、
不动 P1 BM25 打分子、不碰 _s1_tail_deliver）；前 3 主键相对次序不变。

红→绿：修复前 `test_tie_broken_by_segment_id_deterministic`（三段同 seq 同 base）
必 TypeError（先跑贴红输出）；修复后确定性全序、两次 search 全等。
"""
from types import SimpleNamespace

from sc.distill import WindowSeg
from sc.recall import Recaller
from sc.schema import Segment


def _mk_union_window(segs):
    """union-window 形状：独立 seq 计数（主 store + model 旁路两 store 并集形态）。

    Window 只需 (capacity_tokens, segs) 面（pull_budget 取 capacity）→ 轻替身即可
    构造，无磁盘 jsonl；segs 为真实 Segment（seq 显式赋值，模拟
    _recall_union_window 跨 store seq 碰撞）。Recaller.search 不触 store。"""
    return SimpleNamespace(capacity_tokens=1000,
                           segs=[WindowSeg(s) for s in segs])


def _seg(text, tier, seq, prov="model"):
    """真实 Segment（默认字段完整，不依赖 JsonlStore.append 赋 seq）。"""
    return Segment(tier=tier, provenance=prov, original_text=text, seq=seq)


class TestRecall_SegmentKeyTotalOrder:
    def test_two_identical_segments_no_crash_guard(self):
        """guard（两段全等 = timsort 短路，修复前后均不炸）：union 窗口内两段
        同 tier + 同 seq + 同文。CPython timsort 对「两段三主键全等」不强制
        第 4 元素比较 → 修复前亦通过；本测封死「两段同 seq 必炸」的**误**断言，
        把 red 精确留给真正的 ≥3 段触发面。仍走 sc/recall.py 排序路径（scored.sort）。"""
        w = _mk_union_window([
            _seg("port 8080 auth", "P1", 3),
            _seg("port 8080 auth", "P1", 3),   # 跨 store 同 seq（union 碰撞面）
        ])
        r = Recaller(None, w)
        h = r.search("port 8080", k=2)
        assert len(h) == 2

    def test_tie_broken_by_segment_id_deterministic(self):
        """RED 复现（修复前必 TypeError）→ GREEN（修复后确定性全序）：三段同 seq、
        其中两段三主键（-base, tier, seq）全等 → timsort 被迫比第 4 元素 Segment 裸
        比较炸（'<' not supported between 'Segment' and 'Segment'）→ 修复后按
        segment_id 定全序；同 query 两次 search 全等（无 wall-clock / 无随机）。"""
        w = _mk_union_window([
            _seg("port 8080 auth", "P1", 3),
            _seg("port 8080 zeta", "P1", 3),
            # R3 配套（2026-10-10）：P0 段 provenance="user"——R3 把 model P0 boost 降到 1.0
            # （不享用户级加权），"P0 恒排前（×1.5）"断言只对 user P0 成立。P1 保持 model
            # （P1 无 boost 分叉，R3 零影响）。
            _seg("port 8080 auth extra", "P0", 3, prov="user"),
        ])
        r = Recaller(None, w)
        h1 = r.search("port 8080", k=3)
        h2 = Recaller(None, w).search("port 8080", k=3)
        assert [h.seg.segment_id for h in h1] == [h.seg.segment_id for h in h2], \
            "确定性破坏：两次结果不一致"
        assert len(h1) == 3
        # P0 boost ×1.5（user P0）+ 同 seq 衰减因子 1.0（seq=n-1 最新档）→ P0 段恒排前
        # （相对次序 = 原 3 主键语义，零变化；R3 只改 model P0 分叉，user P0 不变）
        assert h1[0].seg.tier == "P0"
        # 两个 P1 同 base 同 seq → 第 4 要素 segment_id 决次序（合法全序）
        p1_ids = [h.seg.segment_id for h in h1 if h.seg.tier == "P1"]
        assert p1_ids == sorted(p1_ids), "P1 tie 全序不成立"

    def test_same_tier_same_seq_different_text_relative_order(self):
        """GREEN 相关度主序断言：同 tier + 同 seq、不同文 → 命中独有 term 的段 base
        更高、恒排前（-base 主序不变）；两段 base 不同故修复前后都不触第 4 比较，
        本测封死「修 key 不改相对次序」硬约束 + 确定性两次全等。"""
        w = _mk_union_window([
            _seg("port 8080 zeta", "P1", 7),
            _seg("port 8080 alpha", "P1", 7),
        ])
        r = Recaller(None, w)
        h1 = r.search("port 8080 zeta", k=2, decay_lambda=0.5)
        h2 = Recaller(None, w).search("port 8080 zeta", k=2, decay_lambda=0.5)
        assert [h.seg.segment_id for h in h1] == [h.seg.segment_id for h in h2]
        # zeta 段命中 query 独有 term → base 更高 → 恒排前（相关度主序不变）
        assert "zeta" in h1[0].seg.original_text
        assert "alpha" in h1[1].seg.original_text
