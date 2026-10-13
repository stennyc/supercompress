"""combined regression · 跨 fix 联合行为钉死（tiebreak 稳定 / ts>0 / L3 cap）。

本卡 t_3604a7f6 只交付 tests/test_combined_fixes.py：三条跨 fix 的联合行为
面各钉一处——断言的是**跨 fix 交互**（单卡各测只覆盖单 fix，无法验证三条
fix 同时生效时的联合效果）：

1. tiebreak 稳定 × ts>0 联合（fix-1 + fix-2）：union 双 0-based 计数器全撞，
   展示序 + _rank_key 全序跨两次 replay 方向稳定，且同号双段 live ts 全 >0。
2. ts>0 × L3 cap 联合（fix-2 + fix-3）：同一注入窗口里短段(≤cap)→L3、
   有 digest 大段(>cap)→L2、无 digest 大段(>cap)→L1|L0，且全窗口 live 段 ts>0；
   存量 ts=0 段零迁移保留。
3. tiebreak × L3 cap 联合（fix-1 + fix-3）：混合窗口（同号双段 + 超 cap 段）
   渲染序仍按 (seq, hash16) 定全序，超 cap 段被 cap 降级、短段不受扰动。

真实 fixture API（reviewer 钉死，禁裸对象 / 禁 mock）：
- choose_forms 入参 = WindowSeg 包装器（非裸 Segment）；ts 断言走 live_segments()
  （ModelStore 无 get_latest()）；两段 fixture 内容必须不同（content_hash 不同）
  才触发 tiebreak 路径，同内容走 dedup 测不到。

红线（test-only）：零生产代码变更（本文件在 tests/ 内，sc/*.py 零 diff）；
宿主 agent/ 零 diff；不推远端；零 gateway restart；fixture 全走 tmp_path
（独立 ModelStore 实例），零活体 store 触碰；零 mock / 零 patch / 零 call_count。

跑法（repo 根）：python -m pytest tests/test_combined_fixes.py -v
"""
from __future__ import annotations

from sc.distill import Window, WindowSeg
from sc.inject import choose_forms, _rank_key, DEFAULT_L3_MAX_TOKENS
from sc.model_store import ModelStore
from sc.schema import Segment, est_tokens


def _mk_ms(tmp_path, name: str) -> ModelStore:
    """独立 ModelStore fixture（sync=True 无 worker，零活体触碰）。"""
    return ModelStore(str(tmp_path / f"{name}.model.jsonl"), sync=True)


def _ingest(ms: ModelStore, text: str, model_version: str = "m1") -> Segment:
    ok, gate = ms.ingest_assistant(text, min_chars=200,
                                    model_version=model_version, main_by_hash={})
    assert ok, f"ingest 失败 gate={gate}"
    return ms.live_segments()[-1]


def _union_window(main_segs, model_segs) -> Window:
    """union-window 形状（同 engine._recall_union_window：主 + model live 段直接
    拼接、零重编号）：两侧独立 0-based 计数器 → 同 seq 段同时落窗口。"""
    segs = [WindowSeg(s) for s in main_segs] + [WindowSeg(s) for s in model_segs]
    return Window(100000, segs)


class TestTiebreakTimesTs:
    """fix-1 × fix-2 联合：union 双 0-based 全撞 → 展示序/_rank_key 全序跨 replay
    方向稳定，且同号双段 live ts 全 >0。"""

    def test_total_order_stable_and_live_ts_positive(self, tmp_path):
        ms_main = _mk_ms(tmp_path, "tie_main")
        ms_model = _mk_ms(tmp_path, "tie_model")
        # 两侧各 ingest 一段、内容不同 → content_hash 不同（tiebreak 前提）。
        m_main = _ingest(ms_main, "tie main fact " + "m" * 240)
        m_model = _ingest(ms_model, "tie model fact " + "n" * 240)
        main_segs = [m_main]
        model_segs = [m_model]
        # 双 0-based 计数器 union：两侧各 1 段 seq=0 全撞（recon 全撞同形）。
        assert main_segs[0].seq == 0 and model_segs[0].seq == 0, "union 全撞（双 0-based）"
        assert main_segs[0].content_hash != model_segs[0].content_hash, "内容不同 → hash 不同"
        # 前置：两段均 ≤cap（非 L3 cap 命中路径）。
        assert est_tokens(m_main.original_text) <= DEFAULT_L3_MAX_TOKENS
        assert est_tokens(m_model.original_text) <= DEFAULT_L3_MAX_TOKENS
        # 两方向 union（模拟跨 replay 拼接方向抖动）→ 展示序 + _rank_key 全序不变。
        w1 = _union_window(main_segs, model_segs)
        w2 = _union_window(model_segs, main_segs)
        r1 = choose_forms(list(w1.segs), w1)
        r2 = choose_forms(list(w2.segs), w2)
        ids1 = [ws.segment.segment_id for ws, _, _ in r1]
        ids2 = [ws.segment.segment_id for ws, _, _ in r2]
        assert ids1 == ids2, f"展示序随 replay 方向抖（tiebreak 未定全序）：{ids1} != {ids2}"
        keys = [(ws.segment.seq, ws.segment.content_hash[:16]) for ws, _, _ in r1]
        assert keys == sorted(keys), f"展示序非 (seq,hash16) 全序：{keys}"
        # _rank_key 第四维 = content_hash[:16]（4-tuple 钉死，fix-1）。
        for ws, _, _ in r1:
            assert _rank_key(ws, set()) == (0, 0, -ws.segment.seq,
                                            ws.segment.content_hash[:16]), "rank_key 第四维失钉"
        # 联合 ts>0（fix-2）：同号双段 live ts 全 >0（阈值 0，非 5%）。
        for seg in main_segs + model_segs:
            assert seg.ts > 0.0, f"live 段 ts 未补（fix-2 回退）：{seg.segment_id} ts={seg.ts}"
        # 双段 form 归属（≤cap 短段 → L3）。
        forms = {ws.segment.segment_id: ff for ws, ff, _ in r1}
        assert forms[m_main.segment_id] == "L3" and forms[m_model.segment_id] == "L3", \
            f"短段（≤cap）选档漂移：{forms}"


class TestTsTimesCap:
    """fix-2 × fix-3 联合：同一注入窗口 ts>0 全量生效 + L3 cap 命中降级，
    且存量 ts=0 段零数据迁移保留。"""

    def test_ts_full_and_cap_downgrade_same_window(self, tmp_path):
        ms = _mk_ms(tmp_path, "ts_cap")
        _ingest(ms, "ts short seg " + "c" * 240)                      # ~63 tok ≤ cap
        big_dig = _ingest(ms, "ts big digest " + "b" * 3800)         # ~955 tok > cap
        _ingest(ms, "ts big nodeg " + "d" * 3800)                    # ~955 tok > cap
        # 存量 ts=0 段（store.append 面 schema 默认）+ supersede 新段 ts>0（fix-2）。
        legacy = ms.store.append(Segment(tier="P0", provenance="model",
                                         original_text="legacy ts0 " + "l" * 300,
                                         structural_p0=False))
        assert legacy.ts == 0.0, "存量段 = schema 默认 ts=0（pre-fix 形态）"
        new_seg = ms.supersede(legacy.segment_id, "superseded seg " + "s" * 240, model_version="m1")
        assert new_seg is not None, "supersede 新段非空"
        assert new_seg.ts > 0.0, "supersede 新段 ts 非零"
        assert ms.get(legacy.segment_id).ts == 0.0, "原段 ts=0 零迁移保留"
        # 真实 fixture 窗口：live 段 WindowSeg 包装（非裸 Segment）。
        live = ms.live_segments()
        seg_short = next(s for s in live if s.original_text.startswith("ts short"))
        seg_big_dig = next(s for s in live if s.original_text.startswith("ts big digest"))
        seg_big_no = next(s for s in live if s.original_text.startswith("ts big nodeg"))
        assert est_tokens(seg_short.original_text) <= DEFAULT_L3_MAX_TOKENS
        assert est_tokens(seg_big_dig.original_text) > DEFAULT_L3_MAX_TOKENS
        assert est_tokens(seg_big_no.original_text) > DEFAULT_L3_MAX_TOKENS
        seg_big_dig.digest_text = "short digest summary"             # 运行期 digest（不落盘）
        w = Window(100000, [WindowSeg(s) for s in live])
        forms = {ws.segment.segment_id: ff for ws, ff, _ in choose_forms(w.segs, w)}
        # 联合 form 归属（fix-3 cap 生效）：短段 L3 / 有 digest 大段 L2 / 无 digest 大段 L1|L0。
        assert forms[seg_short.segment_id] == "L3", f"≤cap 短段无扰动仍 L3：{forms}"
        assert forms[seg_big_dig.segment_id] == "L2", f"超 cap + digest → L2（不截断）：{forms}"
        assert forms[seg_big_no.segment_id] in ("L1", "L0"), f"超 cap + 无 digest → L1|L0：{forms}"
        assert forms[seg_big_dig.segment_id] != "L3" and forms[seg_big_no.segment_id] != "L3", \
            "超 cap 段绝不装细节池（cap 命中根治单段独吞）"
        # 联合 ts>0（fix-2）：live 段全 >0（supersede 新段也在其中且 >0）。
        assert all(s.ts > 0.0 for s in live), "live 段 ts 未补（fix-2 回退）"
        assert forms[new_seg.segment_id] == "L3", f"supersede 新段 ≤cap → L3：{forms}"


class TestTiebreakTimesCap:
    """fix-1 × fix-3 联合：混合窗口（union 全撞短双段 + 超 cap 段）→ 渲染序仍按
    (seq, hash16) 定全序，超 cap 段被 cap 降级、短段渲染不受扰动。"""

    def test_mixed_window_order_and_cap(self, tmp_path):
        ms = _mk_ms(tmp_path, "mix")
        _ingest(ms, "mix a " + "a" * 240)      # 短
        _ingest(ms, "mix b " + "b" * 240)      # 短
        big = _ingest(ms, "mix big " + "x" * 3800)   # 超 cap，无 digest
        live = ms.live_segments()
        w = Window(100000, [WindowSeg(s) for s in live])
        r = choose_forms(list(w.segs), w)
        forms = {ws.segment.segment_id: ff for ws, ff, _ in r}
        # 渲染序 = (seq, hash16) 全序（fix-1）。
        order_keys = [(ws.segment.seq, ws.segment.content_hash[:16]) for ws, _, _ in r]
        assert order_keys == sorted(order_keys), f"渲染序非全序：{order_keys}"
        # 同号混合（不同 seq 段）+ cap 联合：超 cap 大段降级、短段保持 L3（fix-3）。
        assert forms[big.segment_id] in ("L1", "L0"), f"超 cap 大段未降级：{forms}"
        shorts = [s for s in live if s.segment_id != big.segment_id]
        assert all(forms[s.segment_id] == "L3" for s in shorts), \
            f"短段被 cap 误伤：{forms}"
        assert forms[big.segment_id] != "L3", "超 cap 段不得装细节池"
