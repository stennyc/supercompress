"""R4（2026-10-13）：写侧去独白面 + L1 渲染面 ts 刷新（owner 病灶 a/c 修复卡）。

对应 owner 诊断账本的两条设计缺失：

(a) #7 去独白：写侧（on_turn_complete 主 store 入库 + model 旁路 B 卡 gate +
    tagger.classify）全都不做内容噪声过滤——model 段开首的推理独白
    （"让我…"/"I need to…"/"Let me…"）被当正常 P0/P1 内容入库，L3 渲染时
    原样进 compact-ledger 占 60% 篇幅。修法（owner 拍板，零 LLM 确定性，
    对齐 filemem noise rows「直接丢弃」规矩但保守落地 = 降 P2 不丢）：

    - sc/tagger.py：新增 REASONING_MONOLOGUE_CUES（多语言开首独白词表）+
      谓词 is_reasoning_monologue(text)：text 开首（strip 后前 60 字符内）
      命中词表 → 判定纯推理独白段。
    - tagger.classify LLM 面：provenance=="model" 且判定独白 → 确定性降 P2
      （P2 = 不进 P0/P1 压缩注入面 = 不进账本 staged 块；段仍在主 store，
      P2 召回面/点查可达 = 不丢数据，对齐 filemem noise rows 的
      "stays in draft storage for audit" 精神）。判定面在 LLM tiering
      **之前**短路（确定性、零 LLM、零成本；LLM 面保留但独白段根本不会
      走 LLM = 白省一次 aux 调用）。
    - user 段免疫（与 room-skeleton / self-report 同模式）：独白词表对
      user 段一律不生效——user 说"让我试试"是正常指令，绝不判独白。

(c) #1 ts 不刷新：L1 索引行 `#N P0 ts=<int> · …` 的 ts 段复用段级 ts
    （sc/store.py 段落盘时打的 time.time()）；session 冷重开/回放时同批
    落盘的段 ts 近似 → 账本里 8 行共用同一 ts。修法（渲染面单点，
    owner 裁定口径）：sc/inject.py 的 L1 行渲染时，段的 ts 为 0
    或异常（回放段未刷新）时**不带 ts 段**（省略 `ts=…`，行形
    `#N P0 · <fact>`）——不新造时间戳（造了就是假数据），只让"ts
    不可信"段不再渲染出误导性的共用 ts。段 ts 正常（>0）的行字节不变。
"""
from __future__ import annotations

import pytest

from sc.tagger import Tagger, REASONING_MONOLOGUE_CUES, is_reasoning_monologue
from sc.schema import Segment
from sc.inject import _l1_line
from sc.distill import WindowSeg
from fakes import FakeEndpoint
from sc.aux_provider import AuxProvider


# ───────────────────────── 共 用 fixture ─────────────────────────
@pytest.fixture
def ep():
    """可控 LLM 端点：记录 calls；responses 可注入（缺省空 → tagger 走
    transient_fail 兜底 P0 structural，既有契约）。"""
    return FakeEndpoint()


@pytest.fixture
def aux(ep):
    return AuxProvider(ep, fail_cooldown=0.0, async_digest=False)


@pytest.fixture
def tagger(aux):
    return Tagger(aux, session_topic="t")


# ───────────────────────── (a) 独白词表 + 谓词 ─────────────────────────
class TestMonologueCuesTable:
    def test_covers_zh_and_en_openers(self):
        joined = " ".join(REASONING_MONOLOGUE_CUES)
        # 中：开首独白惯用形（账本实证词面，F4 后 14 条）
        for cue in ("让我", "我需要", "现在我", "我先", "我打算", "准备先", "让我核实"):
            assert cue in joined, f"{cue} 缺"
        # F4：宽词「查清楚」「查一下」已删（裸形 = 结论段开首，入表即误杀）——
        # 反向断言其不在表（宁漏勿错红线落地）
        for gone in ("查清楚", "查一下"):
            assert gone not in joined, f"{gone} 不应在表（F4 宽词红线）"
        # 英：开首独白惯用形（小写词表）
        joined_low = joined.lower()
        for cue in ("let me", "i need to", "now i need", "i will", "i'll check",
                    "now i", "first i", "i need to check"):
            assert cue in joined_low, f"{cue} 缺"
        # F4：「i found」「i discovered」已删（"I found that X" 是结论段开首，非独白）
        for gone in ("i found", "i discovered"):
            assert gone not in joined_low, f"{gone} 不应在表（F4 宽词红线）"

    def test_no_broad_words(self):
        # 宽词不入表（owner 红线：宁漏勿错，宽词误杀面）：单独"发现"
        # /"结果"/"完成"无开首限定不入表——词表只收"我 + 让/需/先/查 + 动作"
        # 开首独白结构。F4 补强：查清楚/查一下/i found/i discovered 同属
        # 结论词（"I found that X" / "查清楚 X 的根因" 是结论开首，非独白带前缀），
        # 一并断言其缺席。
        joined = " ".join(REASONING_MONOLOGUE_CUES).lower()
        assert "发现" not in joined
        assert "结果" not in joined
        assert "完成" not in joined
        assert "查清楚" not in joined
        assert "查一下" not in joined
        assert "i found" not in joined
        assert "i discovered" not in joined


class TestIsReasoningMonologue:
    def test_zh_openers(self):
        for text in (
            "让我先核实宿主到底提供了哪些 turn-boundary face，再判断。",
            "我需要找出当时 emo 的源文件是否真的被读取过。",
            "我先测试一下当前状态。",
        ):
            assert is_reasoning_monologue(text) is True, text

    def test_zh_conclusion_not_flagged(self):
        # F4：「查清楚了，结论是…」是**结论段开首**（查清某事 = 结论词，非独白带
        # 前缀），删「查清楚」cue 后不再命中 = 正确（误杀结论段才是真丢失）
        assert is_reasoning_monologue("查清楚了，结论是：footer 该推的地方确实推了。") is False

    def test_en_openers(self):
        for text in (
            "I need to check which tool actually read the emo files.",
            "Let me verify the order and run the suite.",
            "Now I need to confirm the final details and then TDD it.",
        ):
            assert is_reasoning_monologue(text) is True, text

    def test_en_conclusion_not_flagged(self):
        # F4：「I found the root cause — …」是**结论段开首**（"I found that X"
        # 发现某事 = 结论词，非独白带前缀），删「i found」cue 后不再命中 = 正确
        assert is_reasoning_monologue("I found the root cause — detect_root returns /root…") is False

    def test_non_monologue_not_flagged(self):
        for text in (
            "全量 133 passed，两个 commit 在本地 main，未推远端。",
            "## 改动总结：修了 3 处，验证通过。",
            "好，按你的决定钉死，先动手写测试。",
            "The gateway loads from /root/.hermes/plugins/…",
            "结果：全绿。",
        ):
            assert is_reasoning_monologue(text) is False, text

    def test_empty_and_blank(self):
        assert is_reasoning_monologue("") is False
        assert is_reasoning_monologue(None) is False

    def test_case_insensitive(self):
        assert is_reasoning_monologue("LET ME check the logs first") is True
        assert is_reasoning_monologue("Let me check the logs first") is True

    def test_prefix_window_only(self):
        # 开首 60 字符内才判（后段出现的"让我/我需/我先/查"不算开首独白）
        text = ("结论部分：这次修复全绿，133 passed，两个 commit 已验证。"
                "接下来看看剩余的边界 case，再动手。")
        assert is_reasoning_monologue(text) is False


# ───────────────────────── (a) tagger.classify 接线 ─────────────────────────
class TestTaggerMonologueGate:
    def test_model_monologue_deterministic_p2(self, tagger, ep):
        res = tagger.classify("让我先核实宿主到底提供了哪些 face，再判断能不能挪。", "model")
        tier, structural, regime, reason = res[:4]
        assert tier == "P2"
        assert structural is False
        assert regime == "normal"                 # 确定性判定，非 transient_fail
        assert "monologue" in reason.lower()
        # LLM 面短路：独白段根本不走 aux（确定性、零 LLM）
        assert ep.calls == []

    def test_model_non_monologue_goes_to_llm_path(self, tagger, ep):
        ep.responses["tagging"] = '{"tier":"P1","score":5,"settled_signal":0}'
        res = tagger.classify("全量 133 passed，两个 commit 在本地 main。", "model")
        assert res[:2] == ("P1", False)            # LLM 判定生效
        assert len(ep.calls) == 1                  # 非独白段走了 LLM 面（不短路）

    def test_user_short_monologue_is_structural_floor(self, tagger, ep):
        # user 短指令"让我试试"（<100 chars）→ 结构地板 P0 structural，独白词表不生效
        res = tagger.classify("让我试试", "user")
        assert res[:2] == ("P0", True)
        assert ep.calls == []                      # user 面恒不过 aux

    def test_user_long_monologue_still_not_p2(self, tagger, ep):
        # user 长内容（≥100 chars）开首"让我…"→ classify_user 既有规则
        # （P0 non-structural 可逐出），独白面不介入 user 段
        text = "让我把这个问题彻底查清楚，" + "细节" * 100
        res = tagger.classify(text, "user")
        tier, structural = res[:2]
        assert tier == "P0"                        # 长内容 = P0 可逐出（非 P2）
        assert structural is False


# ───────────────────────── (c) L1 行 ts 渲染 ─────────────────────────
def _seg(ts: float, key_fact: str = "fact-x") -> Segment:
    s = Segment(tier="P0", provenance="model", original_text="x" * 10)
    s.seq = 7
    s.ts = ts
    s.key_fact = key_fact
    return s


class TestL1LineTsRendering:
    def test_ts_zero_ratio_unchanged(self):
        # 既有观测面不受渲染面改动影响（只读段 ts，零 mutation）
        from sc.inject import _ts_zero_ratio
        segs = [_seg(0.0), _seg(0.0), _seg(1791822710.0)]
        assert _ts_zero_ratio(segs) == pytest.approx(2 / 3)
