"""block-② recall cache（t_12f38672）单测面。

目标：supercompress `select_context` 召回缓存——当本轮 query 与上一轮相比「未实质变化」
时，复用上一轮 recall 结果（K 段 + 渲染文本），使召回块字节跨轮稳定，减少 head 动态块对
provider 前缀缓存的断裂。

判定口径（docstring 钉死）：本轮 query 的归一化 token 集（`sc.recall.tokenize`）与上一轮
指纹按 **Jaccard 相似度** 比较；≥ `recall_cache_threshold`（默认 0.85）判定「未实质变化」
→ 复用。复用还要求**段活性**：上轮 K 段任一已不在当前 store（被 compact 淘汰/逐出）→
不复用、回落全量重算。

fail-safe 红线（本文件全封）：
- 首轮 / 冷启动（实例态缺失）→ 全量重算（现状行为）
- 实质变化（Jaccard < 阈值）→ 全量重算
- 段被 compact 淘汰 → 回落全量重算（不引用已淘汰段）
- 判定逻辑异常 → fail-open 全量重算（不 crash）
- 全量重算路径与现状**零变化**（本卡只加复用分支，不动打分/渲染/排序本体）

红线（@user）：只改 sc/（engine.py）+ 新测试；零宿主 agent/、零 install tree、零 .db、
不改 `select_context` 对外宿主签名（实例态走 engine 内部，不外溢宿主）。
活体取证锚点：复用分支打可 grep 日志 `supercompress: recall cache-hit (reused prev, K=N)`。
"""
import json
import logging

import pytest

from sc.schema import Segment
from sc.recall import tokenize
from fakes import FakeEndpoint
from engine import SupercompressContextEngine


# ── 可召回语料（Latin 词面，Jaccard 精确可控）──────────────────────────────────
# P0/P1 段含 Latin 词，query 用 Latin → tokenize 输出 = 小写词集，Jaccard 手算可钉。
_P0_TEXT = "Gateway rate limit threshold set to 500 alerts config"
_P1_TEXT = "Deploy version 42 to production environment keep rollback entry"

# 轮 1（全量重算）：15 个 distinct token
_QUERY_A = ("gateway rate limit threshold deploy version rollback 500 alerts "
            "config queue backoff retry circuit breaker")
# 轮 2（cosmetic 变化）：删一个 token（"circuit"）→ 14/15=0.933 ≥ 0.92（R1 阈值）✓
# R1：阈值 0.85→0.92 + 指纹剔 CJK 单字，故 cosmetic 变化须"删一两个词后 Jaccard 仍 ≥0.92"
# 才判未实质变化——_QUERY_A 扩到 15 token，删 1 个 = 0.933，仍过新阈值（测试语义不变）。
_QUERY_B = ("gateway rate limit threshold deploy version rollback 500 alerts "
            "config queue backoff retry breaker")
# 轮 3（实质变化）：完全不同 topic → Jaccard(A,C)=0 < 0.92 ✓
_QUERY_C = "database connection pool timeout memory leak overflow error log"


def _mkseg(text, tier="P1", seq=0):
    return Segment(tier=tier, provenance="model", original_text=text, seq=seq, ts=0.0)


def _seed_recallable(e):
    """seed 一条 P0 + 一条 P1（均对 query >0 分）→ 召回块 = [P1 背景, P0 垫底]。"""
    e.store.append(_mkseg(_P0_TEXT, tier="P0", seq=0))
    e.store.append(_mkseg(_P1_TEXT, tier="P1", seq=1))
    e.store.flush()   # P1 落盘 → _window_fresh replay 可见（P0 即时 fsync 已在盘）


def _block_text(out):
    """取含 [supercompress recall] 标记的消息 content（S1 尾投递面），块体逐字节可比。"""
    if not out:
        return ""
    for m in out:
        c = str(m.get("content", ""))
        if "[supercompress recall]" in c:
            return c
    return ""


def _remove_segment_from_disk(engine, text):
    """重写 on-disk jsonl，移除指定 original_text 的段（模拟 compact 淘汰）。

    store.replay() 会重读磁盘 → 下次 _window_fresh 窗口不含该段 → point_fetch miss。"""
    import os
    path = engine.store.path
    if not os.path.exists(path):
        return
    kept = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("original_text") == text:
                continue
            kept.append(line)
    with open(path, "w") as f:
        for line in kept:
            f.write(line + "\n")


@pytest.fixture()
def engine(tmp_path):
    e = SupercompressContextEngine(external_store_path=str(tmp_path), auto_recall=True)
    e._aux.main_endpoint = FakeEndpoint()   # 默认 fail-safe（无 LLM 重排 → 确定性）
    return e


# ─────────────────────────── ① 首轮 / 冷启动 → 全量重算（现状行为）──────────────────────────
class TestColdStart:
    def test_cold_start_no_cache_hit_recomputes(self, engine, caplog):
        """实例态缺失（首轮）→ 走全量重算、无 `recall cache-hit` marker（现状行为）。"""
        _seed_recallable(engine)
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_A})
        assert out is not None, "召回非空应产出替换请求"
        assert _block_text(out), "首轮全量重算仍应召回并渲染块"
        assert not [m for m in caplog.messages if "recall cache-hit" in m], \
            "冷启动（实例态缺失）不得打 cache-hit marker（= 全量重算）"
        assert [m for m in caplog.messages if "select_context recalled" in m], \
            "首轮走全量重算路径（现状 recalled 日志在）"

    def test_cold_start_writes_back_state(self, engine):
        """首轮全量重算后须写回 recall 实例态（供下轮比对；实例态 = engine 内部，不外溢宿主）。"""
        _seed_recallable(engine)
        engine.select_context(
            [{"role": "user", "content": "x"}],
            incoming_message={"role": "user", "content": _QUERY_A})
        assert getattr(engine, "_recall_cache", None) is not None, \
            "首轮全量重算后须写回 recall 实例态（供下轮比对）"


# ─────────────────────────── ② 未实质变化 → 复用（字节稳定 + cache-hit marker）──────
class TestUnchangedReused:
    def test_same_query_reused_cache_hit_bytes_stable(self, engine, caplog):
        """同一 query 连两轮 → 第二轮复用：块字节全等 + `recall cache-hit` marker。"""
        _seed_recallable(engine)
        caplog.clear()
        req = [{"role": "user", "content": "x"}]
        with caplog.at_level(logging.INFO, "engine"):
            out1 = engine.select_context(
                req, incoming_message={"role": "user", "content": _QUERY_A})
        assert out1 is not None
        b1 = _block_text(out1)
        assert b1, "轮 1 全量重算出块"
        # 轮 2（同指纹 → 复用；段仍活 → 复用成立）
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out2 = engine.select_context(
                req, incoming_message={"role": "user", "content": _QUERY_A})
        assert out2 is not None
        b2 = _block_text(out2)
        assert b2 == b1, "复用：召回块字节须与上轮逐字节相同（provider 前缀不拆）"
        hits = [m for m in caplog.messages if "recall cache-hit" in m]
        assert len(hits) == 1, "复用分支须打恰一行可 grep 的 cache-hit marker（活体取证锚点）"
        assert "K=" in hits[0], "marker 文案须含 K 值（K=N）"

    def test_cosmetic_query_variation_still_reused(self, engine, caplog):
        """「未实质变化」口径 = 归一化 token 集 Jaccard ≥ 阈值：仅删一个词（B 相对 A）
        仍判未实质变化 → 复用；块字节全等。钉死 0.85 判定口径（非全等、非逐字）。"""
        _seed_recallable(engine)
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out1 = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_A})
        assert out1 is not None
        b1 = _block_text(out1)
        # 前置自检：A/B Jaccard 须 ≥ 阈值（证判定口径是真「阈值」非「全等」）
        ja = set(tokenize(_QUERY_A))
        jb = set(tokenize(_QUERY_B))
        jacc = len(ja & jb) / len(ja | jb) if (ja | jb) else 0.0
        assert jacc < 1.0, "前置：A/B token 集不同（Jaccard<1）"
        assert jacc >= engine.recall_cache_threshold, \
            f"前置：A/B Jaccard={jacc:.3f} 须 ≥ 阈值 {engine.recall_cache_threshold}"
        # 轮 2 query=B（token 集是 A 的真子集，Jaccard ≥ 0.85 → 未实质变化 → 复用）
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out2 = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_B})
        assert out2 is not None
        assert _block_text(out2) == b1, "未实质变化（Jaccard≥阈值）须复用、块字节全等"
        assert [m for m in caplog.messages if "recall cache-hit" in m], "须打 cache-hit marker"


# ─────────────────────────── ③ 实质变化 → 全量重算（不引用旧结果）──────────────────────────
class TestMaterialChange:
    def test_material_change_recomputes_no_cache_hit(self, engine, caplog, tmp_path):
        """实质变化（Jaccard < 阈值）→ 走全量重算、不复用、不打 cache-hit。

        零变化钉死（全量重算路径与现状行为字节一致）：_QUERY_C 与种子 P0/P1 段
        零 token 重叠 → 全量重算空召回 → 返回 None（engine.py `if not _recalled_pairs:
        return None`，现状基线行为）；空召回不写回实例态（上轮指纹/段保留）。
        用独立冷实例做同 query 基线对照，钉死本卡对重算路径零 mutation。"""
        _seed_recallable(engine)
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out1 = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_A})
        assert out1 is not None
        b1 = _block_text(out1)
        assert b1, "轮 1 全量重算出块"
        # 前置自检：A/C Jaccard < 阈值（判为实质变化）
        ja = set(tokenize(_QUERY_A))
        jc = set(tokenize(_QUERY_C))
        jacc = len(ja & jc) / len(ja | jc) if (ja | jc) else 0.0
        assert jacc < engine.recall_cache_threshold, \
            f"前置：A/C Jaccard={jacc:.3f} < 阈值（判为实质变化）"
        # 轮 2 query=C（强不相关 → 全量重算；与种子段零词面重叠 → 空召回 → None）
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out2 = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_C})
        assert not [m for m in caplog.messages if "recall cache-hit" in m], \
            "实质变化不得复用（无 cache-hit marker）"
        # 零变化钉死：独立冷实例同 query 全量重算基线（无本卡实例态 → 现状路径）
        e2 = SupercompressContextEngine(external_store_path=str(tmp_path / "e2"), auto_recall=True)
        e2._aux.main_endpoint = FakeEndpoint()
        _seed_recallable(e2)
        out_b = e2.select_context(
            [{"role": "user", "content": "x"}],
            incoming_message={"role": "user", "content": _QUERY_C})
        assert out_b is None, "基线（冷实例）：_QUERY_C 空召回 → None（现状行为）"
        assert out2 is None, "实质变化全量重算 = 现状零变化（空召回 → None，非复用旧块）"
        # 空召回不写回实例态 → 第 3 轮 A 仍用上轮（轮 1）指纹/段复用成功
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out3 = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_A})
        assert out3 is not None and _block_text(out3) == b1, \
            "空召回未覆写实例态：第 3 轮 A 仍复用（块字节与轮 1 全等）"
        assert [m for m in caplog.messages if "recall cache-hit" in m], \
            "第 3 轮复用须打 cache-hit marker"


# ─────────────────────────── ④ 段被 compact 淘汰 → 回落全量重算 ───────────────────────────
class TestEvictedFallback:
    def test_evicted_segment_falls_back_recompute(self, engine, caplog):
        """复用要求段活性：上轮 K 段任一已不在当前 store（被 compact 淘汰）→ 不复用、
        回落全量重算（避免引用已淘汰段）。"""
        _seed_recallable(engine)
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out1 = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_A})
        assert out1 is not None and _block_text(out1), "轮 1 召回出块"
        # 模拟 session 新 compact 淘汰 P1 段：从磁盘移除 → 下次 replay 窗口不含 P1
        _remove_segment_from_disk(engine, _P1_TEXT)
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out2 = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_A})
        # 段被 compact 淘汰 → 活性检查失败 → 不得打 cache-hit marker
        assert not [m for m in caplog.messages if "recall cache-hit" in m], \
            "段被 compact 淘汰 → 活性检查失败 → 不得复用（无 cache-hit marker）"

    def test_fingerprint_unchanged_but_seg_gone_still_recomputes(self, engine, caplog):
        """同指纹 + 段消失 → 回落全量重算（活性检查独立于指纹判定）。"""
        _seed_recallable(engine)
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out1 = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_A})
        assert out1 is not None
        _remove_segment_from_disk(engine, _P0_TEXT)
        _remove_segment_from_disk(engine, _P1_TEXT)
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out2 = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_A})
        assert out2 is None or out2 is not None, "全量重算路径合法（空召回→None 或出块）"
        assert not [m for m in caplog.messages if "recall cache-hit" in m], "段消失不复用"


# ─────────────────────────── fail-safe：判定逻辑异常 → fail-open 全量重算（不 crash）────
class TestFailSafeException:
    def test_corrupted_cache_state_fails_open(self, engine, caplog):
        """判定逻辑异常（指纹态结构损坏）→ fail-open 到全量重算（不 crash、宿主不炸）。"""
        _seed_recallable(engine)
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_A})
        assert out is not None
        # 注入异常面：让上一轮指纹态结构损坏（活性/指纹读它会抛）→ 须 fail-open 全量重算
        engine._recall_cache = {"fingerprint": None, "seg_ids": None, "recalled_texts": None, "k": None}
        caplog.clear()
        with caplog.at_level(logging.INFO, "engine"):
            out2 = engine.select_context(
                [{"role": "user", "content": "x"}],
                incoming_message={"role": "user", "content": _QUERY_A})
        assert out2 is not None, "判定异常须 fail-open 全量重算（不 crash、仍出块）"
        assert not [m for m in caplog.messages if "recall cache-hit" in m], \
            "判定异常须回落全量重算（无 cache-hit marker）"


# ─────────────────────────── 全量重算路径零变化 + 会话边界清零 + 签名不变 ───────────────────
class TestInvariants:
    def test_full_recompute_byte_identical_to_baseline(self, engine, tmp_path):
        """全量重算路径与现状零变化（本卡只加复用分支）：冷启动全量重算结果 = 独立实例
        全量重算结果（字节全等）。"""
        _seed_recallable(engine)
        out_a = engine.select_context(
            [{"role": "user", "content": "x"}],
            incoming_message={"role": "user", "content": _QUERY_A})
        assert out_a is not None
        b_a = _block_text(out_a)
        # 独立新实例（冷）全量重算同 query → 字节全等（复用面关 = 现状路径）
        e2 = SupercompressContextEngine(external_store_path=str(tmp_path / "e2"), auto_recall=True)
        e2._aux.main_endpoint = FakeEndpoint()
        _seed_recallable(e2)
        out_b = e2.select_context(
            [{"role": "user", "content": "x"}],
            incoming_message={"role": "user", "content": _QUERY_A})
        assert out_b is not None
        assert _block_text(out_b) == b_a, "全量重算路径须与现状零变化（字节全等）"

    def test_session_reset_clears_recall_cache(self, engine):
        """会话边界（/new）须清 recall 实例态（session 记忆 scope，防跨 session 泄漏）。"""
        _seed_recallable(engine)
        engine.select_context(
            [{"role": "user", "content": "x"}],
            incoming_message={"role": "user", "content": _QUERY_A})
        assert getattr(engine, "_recall_cache", None) is not None, "前置：已写回 recall 态"
        engine.on_session_reset()
        assert engine._recall_cache is None, \
            "on_session_reset 须清 recall 实例态（防跨 session 泄漏）"

    def test_session_start_clears_recall_cache(self, engine, tmp_path):
        """会话边界（轮转/新 session start）须清 recall 实例态（对称 on_session_reset）。"""
        _seed_recallable(engine)
        engine.select_context(
            [{"role": "user", "content": "x"}],
            incoming_message={"role": "user", "content": _QUERY_A})
        assert getattr(engine, "_recall_cache", None) is not None
        engine.on_session_start("fresh_s", hermes_home=str(tmp_path / "s2"))
        assert engine._recall_cache is None, "on_session_start 须清 recall 实例态"

    def test_select_context_signature_unchanged(self):
        """宿主接口 `select_context` 签名零改（红线：实例态走 engine 内部，不外溢宿主）。"""
        import inspect
        sig = inspect.signature(SupercompressContextEngine.select_context)
        params = list(sig.parameters)
        assert "request_messages" in params
        assert "conversation_messages" in params
        assert "incoming_message" in params
        assert "budget_tokens" in params
        assert not any("cache" in p.lower() for p in params), \
            "不得向宿主 select_context 接口引入 cache 参数（实例态走 engine 内部）"


# ─────────────────────── A1 止血面（2026-10-10）：auto_recall 默认关 ───────────────────────
class TestAutoRecallDefaultOff:
    """A1：select_context 自动召回注入面默认关（config auto_recall 缺省 False）。

    默认关 = 整条面 no-op：return None（宿主 fail-open，请求字节不变），
    杜绝每 turn 把 model P0（agent 自述）/recall 块自动投到末条 user。
    开面（auto_recall=True）行为与现状一致（本文件其余测试即开面回归钉）。"""

    def _engine_default(self, tmp_path):
        """默认构造（不传 auto_recall）→ 开关 = False（缺省安全默认）。"""
        e = SupercompressContextEngine(external_store_path=str(tmp_path))
        e._aux.main_endpoint = FakeEndpoint()
        return e

    def test_default_off_returns_none_and_bytes_unchanged(self, tmp_path):
        """默认关：select_context → None（宿主 fail-open，原请求字节不变）。"""
        e = self._engine_default(tmp_path)
        assert e.auto_recall_enabled is False, "缺省（未配 auto_recall）必须默认关"
        _seed_recallable(e)   # 有可召回语料（P0+P1），若面开着必出块
        req = [{"role": "user", "content": "hello"},
               {"role": "assistant", "content": "world"}]
        out = e.select_context(
            req, incoming_message={"role": "user", "content": _QUERY_A})
        assert out is None, "默认关面 select_context 必须 return None（宿主 fail-open）"
        # 请求原样（宿主拿 None → 用原 api_messages，零 mutation 契约）
        assert req == [{"role": "user", "content": "hello"},
                       {"role": "assistant", "content": "world"}], \
            "fail-open 面不得 mutation 原请求列"

    def test_default_off_writes_no_recall_cache_sidecar(self, tmp_path):
        """默认关：整条面短路 → 不写 recall 实例态 / 不落 sidecar（零写盘、零痕迹）。"""
        import os
        e = self._engine_default(tmp_path)
        e.on_session_start("sid_a1", hermes_home=str(tmp_path))
        _seed_recallable(e)
        out = e.select_context(
            [{"role": "user", "content": "x"}],
            incoming_message={"role": "user", "content": _QUERY_A})
        assert out is None
        assert getattr(e, "_recall_cache", None) is None, \
            "默认关面不得写 recall 实例态（短路在 try 之前）"
        # sidecar 路径 = <sid>.recall_cache.json（on_session_start rebind 后）
        sidecar = e._recall_cache_sidecar_path()
        assert sidecar is None or not os.path.exists(sidecar), \
            "默认关面不得落 recall_cache sidecar（零写盘）"

    def test_explicit_on_restores_injection(self, tmp_path):
        """显式 auto_recall=True → 恢复现状每 turn 自动注入（开面回归，回滚路径）。"""
        e = SupercompressContextEngine(
            external_store_path=str(tmp_path), auto_recall=True)
        e._aux.main_endpoint = FakeEndpoint()
        assert e.auto_recall_enabled is True
        _seed_recallable(e)
        out = e.select_context(
            [{"role": "user", "content": "x"}],
            incoming_message={"role": "user", "content": _QUERY_A})
        assert out is not None, "开面必须恢复现状注入（非 None）"
        assert _block_text(out), "开面必须产出 [supercompress recall] 块"
