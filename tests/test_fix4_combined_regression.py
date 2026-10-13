"""fix-4 · 联合回归集成验收层（tiebreak 稳定 / ts>0 / L3 cap，真实 fixture API，行为面断言）。

三张正卡（fix-1 t_ad82f732 / fix-2 t_a0f8a6d2 / fix-3 t_5bcb33f2）已各自闭环；本卡是
**集成验收层**：用真实 fixture API 联合回归三条行为面，防单卡回归漂移（防 mock 回原样）。

红线（test-only）：
- 零生产代码变更：本文件 + test_l3_cap.py docstring Nit 全部在 tests/ 内，sc/*.py 零 diff；
- 宿主 agent/ 零 diff；不推远端；零 gateway restart。
- fixture 全部走 tmp_path 承载（独立 JsonlStore / ModelStore 实例），零活体 store 触碰。

三条硬性验收行为面（@reviewer 承重判决，缺一条即 FAIL）：
1. **断言行为面，不是 mock 调用计数**：三个用例各自断言 choose_forms 实际选出的 form
   （L2/L3/L0 归属）+ live_segments() 里段真实的 ts 值（> 0 阈值 0，非 5%）。
   全文件零 mock / 零 patch / 零 call_count——只有真实 API 返回值的真实断言。
2. **tiebreak 路径前提（同号不同段）**：union window 里两段 **同 seq、内容不同
   （不同 content_hash）** 的 fixture 触发 inject.py:148 展示序 (seq, hash16) tiebreak +
   inject.py:54 _rank_key 第四维决胜。
   （卡身钉死 #2 原文写「同 content_hash 两段/不同 seq」，与本卡 测试结构 段「两段内容必须
    不同（不同 content_hash 才触发 tiebreak——同 hash 走 dedup 不进 tiebreak）」冲突；
    按代码 + 测试结构 段裁定：同 seq 才需要 tiebreak，同 hash 走 dedup。本卡按此落地。）
3. **L3 cap 验收窗口用真实 fixture 非裸 Segment**：choose_forms 入参 = WindowSeg 包装器
   （真实 live 段构造），L3 cap 命中路径用真实 fixture 段 est_tokens 估算（schema.py:30
   单源 ceil(len/4)），cap 命中 → 整段降级（不装 L3，走 digest-L2 或 L1/L0 指针），
   非手工拼的裸对象。

跑法（repo 根）：python -m pytest tests/test_fix4_combined_regression.py -v
"""
from __future__ import annotations

from sc.distill import Window, WindowSeg
from sc.inject import choose_forms, _rank_key, DEFAULT_L3_MAX_TOKENS
from sc.model_store import ModelStore
from sc.schema import Segment, est_tokens


def _model_store(tmp_path, name: str, text: str) -> ModelStore:
    """真实 ModelStore fixture（sync=True 无 worker，零活体触碰）：ingest 一段真内容
    （≥200 char 走 chars gate；普通文本 → T1 归一 fail-open 原文原样，est_tokens 一致）。"""
    ms = ModelStore(str(tmp_path / f"{name}.model.jsonl"), sync=True)
    ok, gate = ms.ingest_assistant(text, min_chars=200, model_version="m1", main_by_hash={})
    assert ok, f"ingest 失败 gate={gate}"
    return ms


def _union_window(main_segs, model_segs) -> Window:
    """union-window 形状（engine._recall_union_window 同形：主 live 段 + model live 段
    直接拼接、零重编号）：两侧独立 0-based 计数器 → 同 seq 段同时落在 union 窗口。"""
    segs = [WindowSeg(s) for s in main_segs] + [WindowSeg(s) for s in model_segs]
    return Window(100000, segs)


class TestTiebreakStable:
    """行为面 ①：union 双 0-based 计数器（recon H1 全撞结构）→ 展示序 + _rank_key 双
    排序键在同号段上定全序，跨 union 拼接方向 / 跨 replay 稳定（前缀缓存友好）。"""

    def test_display_and_rank_total_order_stable(self, tmp_path):
        ms_main = _model_store(tmp_path, "main", "main side fact " + "m" * 240)
        ms_model = _model_store(tmp_path, "model", "model side fact " + "n" * 240)
        main_segs, model_segs = ms_main.live_segments(), ms_model.live_segments()
        # 独立 0-based 计数器 union：两侧各 1 段 seq=0 全撞（recon 全撞结构同形）
        assert main_segs[0].seq == 0 and model_segs[0].seq == 0, "双 0-based 计数器 union 全撞"
        # 两段内容不同 → content_hash 不同（tiebreak 前提；同 hash 走 dedup 不进 tiebreak）
        assert main_segs[0].content_hash != model_segs[0].content_hash, "两段内容不同 → hash 不同"
        assert est_tokens(main_segs[0].original_text) <= DEFAULT_L3_MAX_TOKENS, "前置：短段 ≤ cap"
        assert est_tokens(model_segs[0].original_text) <= DEFAULT_L3_MAX_TOKENS, "前置：短段 ≤ cap"
        # 两方向 union（模拟跨 replay 拼接方向抖动）
        w1 = _union_window(main_segs, model_segs)
        w2 = _union_window(model_segs, main_segs)
        r1 = choose_forms(list(w1.segs), w1)
        r2 = choose_forms(list(w2.segs), w2)
        ids1 = [ws.segment.segment_id for ws, _, _ in r1]
        ids2 = [ws.segment.segment_id for ws, _, _ in r2]
        # 行为面 ①-tiebreak：展示序随拼接方向不变（同号组按 hash 前缀定全序）
        assert ids1 == ids2, f"展示序随 union 拼接方向抖（tiebreak 未定全序）：{ids1} != {ids2}"
        keys = [(ws.segment.seq, ws.segment.content_hash[:16]) for ws, _, _ in r1]
        assert keys == sorted(keys), f"展示序非法全序（(seq,hash16) 未定序）：{keys}"
        # _rank_key 第四维 = content_hash[:16]（4-tuple 形状钉死，fix-1）
        for ws, _, _ in r1:
            rk = _rank_key(ws, set())
            assert rk == (0, 0, -ws.segment.seq, ws.segment.content_hash[:16]), \
                f"_rank_key 第四维（hash16）失钉：{rk}"
        # 行为面 ①-form 归属：union 双段全在渲染面（零去重/零丢段），各自真实选档 = L3
        forms = {ws.segment.segment_id: ff for ws, ff, _ in r1}
        assert main_segs[0].segment_id in forms and model_segs[0].segment_id in forms, \
            "union 双段全在渲染面（零去重/零丢段）"
        assert forms[main_segs[0].segment_id] == "L3" and forms[model_segs[0].segment_id] == "L3", \
            f"短段（≤cap）选档归属漂移：{forms}"
        # 行为面 ①-ts：live_segments() 段真实 ts 值 > 0（阈值 0，fix-2 补齐面，非 5%）
        for seg in main_segs + model_segs:
            assert seg.ts > 0.0, f"live_segments 段 ts 未补（fix-2 回退）：ts={seg.ts}"


class TestTsPositive:
    """行为面 ②：ModelStore 两处构造面 ingest_assistant / supersede 盖 ts=time.time()；
    live_segments() 取段真实 ts 值全 > 0；零数据迁移（原段 ts=0 保留不回写）。"""

    def test_ingest_and_supersede_ts_nonzero_form_assigned(self, tmp_path):
        ms = ModelStore(str(tmp_path / "ts.model.jsonl"), sync=True)
        # 连续 ingest 三段：live_segments() 取段真实 ts 值全 > 0（阈值 0）
        for i in range(3):
            ok, _ = ms.ingest_assistant(f"ts case seg {i} " + "t" * 240, min_chars=200,
                                        model_version="m1", main_by_hash={})
            assert ok
        live = ms.live_segments()
        assert len(live) == 3
        for seg in live:
            assert seg.ts > 0.0, f"live_segments 段 ts 未补（fix-2 回退）：{seg.segment_id} ts={seg.ts}"
        # supersede 路径：原段（store.append 面 schema 默认 ts=0）零数据迁移保留 0；
        # 新段（supersede 构造面）盖 ts>0
        legacy = ms.store.append(Segment(tier="P0", provenance="model",
                                         original_text="legacy ts0 row " + "l" * 300,
                                         structural_p0=False))
        assert legacy.ts == 0.0, "存量段 = schema 默认 ts=0（pre-fix 形态 fixture）"
        new_seg = ms.supersede(legacy.segment_id, "cleaned seg " + "c" * 240, model_version="m1")
        assert new_seg is not None and new_seg.ts > 0.0, "supersede 新段 ts 非零"
        legacy_again = ms.get(legacy.segment_id)
        assert legacy_again is not None and legacy_again.ts == 0.0, "原段 ts=0 保留不回写（零数据迁移）"
        # 行为面 ②-form 归属：清洗后段（≤cap）进注入面选档 = L3（真实选档，非 mock）
        w = Window(100000, [WindowSeg(s) for s in ms.live_segments()])
        forms = {ws.segment.segment_id: ff for ws, ff, _ in choose_forms(w.segs, w)}
        assert new_seg.segment_id in forms, "supersede 新段进注入面"
        assert forms[new_seg.segment_id] == "L3", f"清洗后段 ≤cap 选档归属漂移：{forms}"
        # ts 行为面在注入窗口段上再断言一次（live_segments 源 → WindowSeg 包装）
        assert all(ws.segment.ts > 0.0 for ws in w.segs
                   if ws.segment.segment_id in forms and ws.segment is not legacy), \
            "注入窗口段 ts 未补"


class TestL3CapRealFixture:
    """行为面 ③：L3 单段 cap 命中路径，真实 fixture 窗口（WindowSeg 包装 live 段，非裸
    Segment），est_tokens 估算（schema.py:30 ceil(len/4)）：cap 命中 → 整段降级
    （不装 L3，走 digest-L2 / L1-L0 指针，**无截断动作**）；≤cap 短段无扰动仍 L3。"""

    def test_cap_hit_downgrades_not_truncates(self, tmp_path):
        ms = ModelStore(str(tmp_path / "cap.model.jsonl"), sync=True)
        # 真实 fixture 三段不同内容（est_tokens 单源估算）：
        short, _ = ms.ingest_assistant("cap short " + "c" * 240, min_chars=200,
                                       model_version="m1", main_by_hash={})          # ~63 tok ≤ cap
        bd, _ = ms.ingest_assistant("cap big digest " + "b" * 3800, min_chars=200,
                                    model_version="m1", main_by_hash={})             # ~955 tok > cap
        bn, _ = ms.ingest_assistant("cap big nodeg " + "d" * 3800, min_chars=200,
                                    model_version="m1", main_by_hash={})            # ~955 tok > cap
        assert short and bd and bn
        live = ms.live_segments()
        seg_short = [s for s in live if s.original_text.startswith("cap short")][0]
        seg_big_dig = [s for s in live if s.original_text.startswith("cap big digest")][0]
        seg_big_no = [s for s in live if s.original_text.startswith("cap big nodeg")][0]
        # 前置估算（est_tokens 单源，非手工 ceil(len/3)）：
        assert est_tokens(seg_short.original_text) <= DEFAULT_L3_MAX_TOKENS, "前置：短段 ≤ cap"
        assert est_tokens(seg_big_dig.original_text) > DEFAULT_L3_MAX_TOKENS, "前置：大段 > cap"
        assert est_tokens(seg_big_no.original_text) > DEFAULT_L3_MAX_TOKENS, "前置：大段 > cap"
        # 运行期 digest 字段（不落盘）：给「有 digest」大段挂摘要 → cap 命中走 L2
        seg_big_dig.digest_text = "short digest summary"
        # 真实 fixture 窗口：WindowSeg 包装 live 段（非裸 Segment 对象手工拼）
        w = Window(100000, [WindowSeg(s) for s in live])
        forms = {ws.segment.segment_id: ff for ws, ff, _ in choose_forms(w.segs, w)}
        # 行为面 ③-form 归属（cap 命中 → 整段降级，不装 L3）：
        assert forms[seg_short.segment_id] == "L3", "≤cap 短段无扰动仍 L3（回归不变）"
        assert forms[seg_big_dig.segment_id] == "L2", "超 cap + 有 digest → 整段降级 L2（不截断）"
        assert forms[seg_big_no.segment_id] in ("L1", "L0"), "超 cap + 无 digest → L1/L0 指针兜底（整段降级）"
        assert forms[seg_big_dig.segment_id] != "L3" and forms[seg_big_no.segment_id] != "L3", \
            "超 cap 段绝不整段装细节池（cap 命中根治单段独吞）"
        # ts 行为面：真实 fixture 段 ts 全 > 0（ModelStore ingest 盖 ts）
        assert all(s.ts > 0.0 for s in live), "真实 fixture 段 ts 未补（fix-2 回退）"
