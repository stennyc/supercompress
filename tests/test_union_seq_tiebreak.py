"""fix-1 · seq 同号展示序 tiebreak 回归面（t_c7002118，gate t_2d213ce9 验收通过）：
sc/inject.py 排序键 content_hash[:16] tiebreak 红线。

背景（recon-A 坐实，gate 验收 PASS）：主 store + model 旁路两个独立 `JsonlStore`
实例各自 `_next_seq` 从 0 起算（sc/store.py:49），`_recall_union_window`
（engine.py:897-911）直接拼接不重编号 → union 窗口内 main seq=N + model seq=N 同号
不同段（recon-A 实测 4 sid 对全撞，结构性非偶发）。`inject.py` 两处纯 seq 排序在
同号段上跨 replay/跨输入序非确定：
- 展示序：`sorted(segs, key=lambda w: w.segment.seq)`（inject.py:125）—
  CPython timsort 稳定排序对同 seq 段保留**输入序** → 渲染序随 union 拼接
  方向抖动（prompt-cache 前缀抖动面）；
- 选档序：`_rank_key` 第三维 `-ws.segment.seq`（inject.py:43-50）— 三主键
  全 tie 时同样依赖输入序 → 选档序与展示序可能反。

修复约束（@reviewer 钉死，全员签过，零 mutation）：
- 一行 tiebreak：seq 仍主键，`content_hash[:16]` 只在 seq 相等时决胜
  （同号必不同 hash → 序唯一且跨 replay 稳定，顺带稳住前缀缓存）；
- 不改 seq / 不改 segment_id（schema.py:80 `content_hash[:16]-seq`）/
  不碰 materialize / 不碰 replay / 不碰 compact-ledger；
- 同做：`_rank_key` 第四维同步加 content_hash[:16] tiebreak
  （否则选档序与展示序可能反；telemetry.filler_form_share 逐字镜像
  _rank_key → 同键同步，零语义漂移）。

红→绿：修复前 `test_display_order_input_order_invariant`（同 seq 对输入序反转
→ 输出序反转）、`test_rank_key_top3_relative_order_stable`（rank_key 同号 tie
未定全序）与 `test_two_replay_passes_deterministic`（replay 恢复序 = dict
插入序（磁盘行序），刻意置 hash 反序 → 纯 seq 稳定排序产出 hash 反序组）
必红；修复后 (seq, content_hash[:16]) 定全序，跨两次 replay 稳定，
dup_seq_count 展示面归零（排序确定——seq 值不变，同号段仍在渲染面，
只是同号时 tiebreak 定序，非 seq 本身去重）。
"""
import os
import tempfile

import scdata
from sc.distill import Window, WindowSeg
from sc.inject import choose_forms, _rank_key
from sc.schema import Segment
from sc.store import JsonlStore


def _seg(text, tier, seq, prov, prefix):
    """真实 Segment（默认字段完整）：content_hash 显式前缀 + seq 显式赋值
    （模拟双 0-based 计数器 union 形态，同 seq 不同段 → 不同 hash 前缀）。
    纯 WindowSeg 路径（不走 scdata.replay）可携带伪造 hash 前缀。"""
    return Segment(tier=tier, provenance=prov, original_text=text,
                   seq=seq, content_hash=prefix + "cccccccccccc")


def _union_window(main_segs, model_segs):
    """union-window 形状（engine._recall_union_window 同形：主 segs + model live 段
    直接拼接、零重编号）：同 seq 段同时落在两侧。"""
    segs = [WindowSeg(s) for s in main_segs] + [WindowSeg(s) for s in model_segs]
    return Window(100000, segs)


def _dup_seq_count(r1, r2):
    """dup_seq_count 展示面：两次渲染（两个 union 拼接方向 / 两次 replay）之间
    展示序不一致的同号 seq 组数。0 = 所有同号组展示序确定（tiebreak 全序）。
    语义 = 「序不定序组数」归零，非 seq 值去重：seq 值不变、同号段仍各自渲染，
    只是同号时 tiebreak 定序（同 t_fd3c7177 segment_id tiebreak 同形）。"""
    g1, g2 = {}, {}
    for ws, _, _ in r1:
        g1.setdefault(ws.segment.seq, []).append(ws.segment.segment_id)
    for ws, _, _ in r2:
        g2.setdefault(ws.segment.seq, []).append(ws.segment.segment_id)
    return sum(1 for seq, ids in g1.items()
               if len(ids) > 1 and ids != g2.get(seq))


def _mk_store(d: str, specs: list) -> JsonlStore:
    """构造独立 0-based 计数器的 JsonlStore（只写 tmp_path，零活体 store 触碰）：
    逐 spec (seq, tier, prov, text) 写 scdata.to_line 真 hash 行（seq 显式指定，
    双计数器 union 形态——recon 全撞结构）；store re-read → replay()
    （权威数据面：逐行验 hash + dict 插入序恢复，`_next_seq` 重算 max+1，
    seq 值原样保留零重编号）。specs 的**行序刻意控制**：replay 恢复序 =
    dict 插入序 = 磁盘行序 → 同号组内段序可独立于 hash 序（hash 反序场景，
    纯 seq 稳定排序的非确定性触发面）。"""
    path = os.path.join(d, "s.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for seq, tier, prov, text in specs:
            s = Segment(tier=tier, provenance=prov, original_text=text)
            s.seq = seq
            f.write(scdata.to_line(s.to_dict()) + "\n")
    return JsonlStore(path)


class TestUnion_SeqTiebreak_DisplayOrder:
    def test_display_order_input_order_invariant(self):
        """RED 复现（修复前必红）→ GREEN：同 seq 段（main+model 各 seq=3，
        不同 hash）渲染后序唯一且与 union 拼接方向（输入序）无关——两方向
        （模拟跨 replay 拼接抖动）渲染序全等，dup_seq_count 展示面归零。
        活体证据对（recon-A H1 seq=32 撞车样本同形：main user + model 旁路
        同号不同 hash 前缀 5a2162eb… / 851d05ca…）。"""
        m3a = _seg("我认为有可能是 cli 属于 one-shot 方式", "P0", 3, "user", "5a2162eb3968e83a")
        m3b = _seg("层 1 已经查清楚了。层 2 查询里", "P0", 3, "model", "851d05ca5d1b630e")
        # 两方向 union（模拟两次 replay 的拼接顺序抖动）
        w1 = _union_window([m3a], [m3b])
        w2 = _union_window([m3b], [m3a])
        r1 = choose_forms(list(w1.segs), w1)
        r2 = choose_forms(list(w2.segs), w2)
        ids1 = [ws.segment.segment_id for ws, _, _ in r1]
        ids2 = [ws.segment.segment_id for ws, _, _ in r2]
        assert ids1 == ids2, f"展示序随 union 拼接方向抖动：{ids1} != {ids2}"
        assert _dup_seq_count(r1, r2) == 0, "同号段渲染序不稳定（dup_seq_count 展示面未归零）"
        # 同号组按 hash 前缀升序定序（5a21… < 851d…，hex ASCII 序）
        assert ids1[0].startswith("5a2162eb3968e83a"), f"hash 前缀升序失钉：{ids1}"
        # seq 值零 mutation：同号段仍在渲染面（tiebreak 定序，非去重）
        assert sorted(ws.segment.seq for ws, _, _ in r1) == [3, 3]

    def test_display_order_full_total_order_stable(self):
        """GREEN 全序断言（修复前必红）：跨 5 段混合（2 组同 seq 对 + 1 唯一
        seq）→ 展示序 = (seq, content_hash[:16]) 升序逐位钉死（合法全序，同
        t_fd3c7177 segment_id tiebreak 同形）。修复前：同 seq=3 组内保留输入序
        （851d… 先于 5a21…）→ keys != sorted(keys) 必红。"""
        segs = [
            _seg("main seq0 text", "P1", 0, "user", "aaaaaaaaaaaaaaaa"),
            _seg("model seq3 text", "P0", 3, "model", "851d05ca5d1b630e"),
            _seg("main seq3 text", "P0", 3, "user", "5a2162eb3968e83a"),
            _seg("main seq0 dup-hash-diff text", "P1", 0, "user", "bbbbbbbbbbbbbbbb"),
            _seg("model seq1 text", "P1", 1, "model", "cccccccccccccccc"),
        ]
        w = _union_window(segs[:4], segs[4:])
        r = choose_forms(list(w.segs), w)
        keys = [(ws.segment.seq, ws.segment.content_hash[:16]) for ws, _, _ in r]
        assert keys == sorted(keys), f"展示序非法全序：{keys}"
        assert keys == [
            (0, "aaaaaaaaaaaaaaaa"),
            (0, "bbbbbbbbbbbbbbbb"),
            (1, "cccccccccccccccc"),
            (3, "5a2162eb3968e83a"),
            (3, "851d05ca5d1b630e"),
        ], "top-k 相对次序漂移（同号决胜位之外的次序变了）"


class TestUnion_SeqTiebreak_RankKey:
    def test_rank_key_top3_relative_order_stable(self):
        """RED 复现（修复前必红）→ GREEN：rank_key 三段（同 seq 同 tier tie +
        1 唯一 P1）选档序三主键全 tie 面 → 同号 tie 未定全序随输入序抖；
        修复后第四维 content_hash[:16] tiebreak 定全序，两方向输入选档序全等。
        前 3 主键（hits/tier/-seq）逐同 = top-k 相对次序不变（P0 组恒先于 P1，
        同号 P0 对内部 = hash 前缀升序 5a21… < 851d…）。"""
        a = _seg("port 8080 auth service", "P0", 3, "user", "5a2162eb3968e83a")
        b = _seg("layer 2 zeta query", "P0", 3, "model", "851d05ca5d1b630e")
        c = _seg("unique tail segment", "P1", 9, "user", "1111111111111111")
        ws_a, ws_b, ws_c = WindowSeg(a), WindowSeg(b), WindowSeg(c)
        order1 = [ws.segment.segment_id for ws in
                  sorted([ws_a, ws_b, ws_c], key=lambda w: _rank_key(w, set()))]
        order2 = [ws.segment.segment_id for ws in
                  sorted([ws_c, ws_b, ws_a], key=lambda w: _rank_key(w, set()))]
        assert order1 == order2, \
            f"选档序随输入序抖（同号 tie 未定全序，选档序与展示序可能反）：{order1} != {order2}"
        # 前 3 主键逐同钉死：P0 组（tier 主键）恒先于 P1（seq=9 虽更新但 tier 后）
        assert order1[-1] == c.segment_id, f"P1 相对次序漂移（主序变了）：{order1}"
        # 同号 P0 对内部 = hash 前缀升序（5a21… < 851d…）
        assert order1[:2] == [a.segment_id, b.segment_id], \
            f"同号决胜位未定全序：{order1}"
        # 第四位存在且 = hash 前缀（4-tuple 形状钉死）
        assert _rank_key(ws_a, set()) == (0, 0, -3, "5a2162eb3968e83a")


class TestUnion_SeqTiebreak_ReplayStable:
    def test_two_replay_passes_deterministic(self):
        """GREEN 跨 replay 稳定性 + RED 触发面（recon 双 0-based 计数器形态）：
        两个独立 JsonlStore（主 + model 旁路）各 2 段（seq 0/1 双空间全撞，
        recon 全撞结构同形）；**磁盘行序刻意置 hash 反序**（同 seq 组内高
        hash 先行）→ 纯 seq 稳定排序（修复前）恢复序 = 磁盘行序 = hash 反序
        组 → keys != sorted(keys) 必红；修复后 (seq, hash16) 全序定序，
        两次 replay（store re-read 重建，dict 插入序恢复）展示序逐位全等。
        seq 值零 mutation 断言：replay 前后 seq 原样（tiebreak 不动数据面）；
        _next_seq = max+1 重算（store.py:200 语义，非本 fix 面）。
        全部写 tmp_path（零活体 store 触碰）。"""
        d1, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
        # 真 hash 行（scdata.replay 逐行验 hash；同 seq 组内 hash 前缀序刻意
        # 反于磁盘行序 → 修复前纯 seq 稳定排序必然保留行序 = hash 反序组）
        main_a = "网关限流阈值调至每秒五百请求并配置告警"    # 组内高 hash 先行（磁盘行 1）
        main_b = "部署发布版本四十二到生产环境保留回滚入口"  # 组内低 hash 后行（磁盘行 2）
        model_a = "层 1 已经查清楚了 层 2 查询叙述面"         # 组内高 hash 先行
        model_b = "aux degraded 单事件自愈观测"               # 组内低 hash 后行
        s_main = _mk_store(d1, [(0, "P0", "user", main_a), (0, "P1", "user", main_b)])
        s_model = _mk_store(d2, [(0, "P0", "model", model_a), (0, "P1", "model", model_b)])

        def _replay_union():
            segs = s_main.replay() + s_model.replay()
            w = _union_window(segs[:2], segs[2:])
            return choose_forms(list(w.segs), w)

        r1 = _replay_union()
        r2 = _replay_union()
        ids1 = [ws.segment.segment_id for ws, _, _ in r1]
        ids2 = [ws.segment.segment_id for ws, _, _ in r2]
        assert ids1 == ids2, f"跨 replay 渲染序不稳：{ids1} != {ids2}"
        # 全序钉死：(seq, hash16) 升序 = 修复后唯一合法展示序（修复前 = 磁盘行序 hash 反序）
        keys = [(ws.segment.seq, ws.segment.content_hash[:16]) for ws, _, _ in r1]
        assert keys == sorted(keys), f"展示序非法全序（hash 反序未定序）：{keys}"
        # seq 零 mutation 断言：replay 前后 seq 值原样（tiebreak 不动数据面）
        assert sorted(s.seq for s in s_main.by_hash.values()) == [0, 0]
        assert sorted(s.seq for s in s_model.by_hash.values()) == [0, 0]
        # _next_seq 重算面 = max+1（store.py:200 语义，非本 fix 改动面）
        assert s_main._next_seq == 1 and s_model._next_seq == 1
        # dup_seq_count 展示面归零（两次 replay 间序一致 = 0）
        assert _dup_seq_count(r1, r2) == 0, "两次 replay 间展示序不一致（dup_seq 未归零）"
