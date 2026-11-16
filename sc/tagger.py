"""Block ④ tagger：P0 确定性结构地板（语言无关）+ LLM tiering（机器可解析 JSON）。

gate 锁语义：
- 结构地板（provenance=user）：短指令（len < 100）恒 P0 structural，不进 LLM 评分
  （transient_fail 时兜底仍保）；长 user-sourced 内容（group chat 转发 / compaction
  summary / 注入块，len ≥ 100）→ P0 non-structural（仍 P0 优先级，push cap 溢出可逐出），
  根治「注入内容打 user provenance → 100% structural → push cap 失效」；
- LLM tiering 走 aux 主模型端点（零新 provider），输出**机器可解析 JSON**
  （`{"tier":"P2","score":3}`），不吐自然语言再 regex；
- 边界 ±1（score 6–8）→ 升一级保守（P2 → P1）；
- 不可机器解析（自然语言 / 坏 JSON）→ 保守 P0 structural，永不当 P2 丢；
- transient_fail（端点异常）→ regime=transient_fail，落结构地板 P0，不误降级。
"""
from __future__ import annotations

import json
import re

from .aux_provider import AuxProvider

# Strip thinking/reasoning blocks that reasoning models embed in content.
_THINK_RE = re.compile(r"<thinking[^>]*>.*?</thinking>|<reasoning[^>]*>.*?</reasoning>",
                       re.DOTALL | re.IGNORECASE)
# Strip code-fence wrappers (````json ... ````) that some models add around JSON.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    """First fenced code block body, or the stripped text when unfenced."""
    text = _THINK_RE.sub("", text or "").strip()
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text


def _parse_llm_json(text: str):
    """Parse LLM response text as JSON, stripping <thinking>/<reasoning> blocks
    and optional code fences first."""
    return json.loads(_strip_code_fences(text))

TIERS = ("P0", "P1", "P2")
_TIERS_BY_TIER = {t: i for i, t in enumerate(TIERS)}
# user 短指令 structural 地板阈值（@reviewer 裁定 2026-09-29）：
# provenance=user 且 len < 100 → P0 structural；len ≥ 100 → P0 non-structural（可逐出）。
USER_STRUCTURAL_LEN = 100
# 长 user-sourced 内容的系统噪声标记（≥100 chars 时才判定，短指令恒保护）：
# 日志/技能提示词 → P2；compaction summary → P1（@researcher 分级表，@reviewer 锁定）。
_P2_MARKERS = ("Traceback (most recent call last)", "console.log(", "console.error(",
               "console.warn(", "UPDATE_SKILL", "update the skill library")
_P1_MARKERS = ("<compaction-summary>",)

# R4（2026-10-13 · owner 病灶 a 去独白面）：model 段开首推理独白词表。
# 判定式：text strip 后前 60 字符内命中任一 cue（大小写不敏感）→ 纯推理独白段。
# 词表只收「我 + 让/需/先/查 + 动作」开首独白结构（多语言各 1-2 个高频词），
# 不收宽词（"发现"/"结果"/"完成"无开首限定 = 结论词，入表即误杀结论段）——
# 宁漏勿错红线：漏 = 独白段照 P0/P1 原样入库（现状不变，零新风险）；
# 误删面只剩开首 60 字符词面命中这一个入口（纯字面、零 LLM、确定性，
# 同 is_agent_self_report R2-v2 模式；user 段免疫由 classify 调用面保证）。
REASONING_MONOLOGUE_CUES = (
    # 中：开首独白惯用形（账本实证词面：「让我…」「我需要…」「我先…」「我打算…」）
    # F4（owner 红线「不收宽词」）：删「查清楚」「查一下」——裸「查清楚 X 根因」直接开首
    # = 任务结论段开首（非独白带前缀），入表即误杀结论段；带前缀形「让我查清楚」由
    # 「让我」cue 覆盖，不单独收裸形。宁漏勿错（漏 = 独白照 P0/P1 入库零风险，
    # 误杀 = 结论段误降 P2 不进注入面 = 真丢失）。
    "让我", "我需要", "现在我", "我先", "我打算", "准备先",
    "让我核实",
    # 英：开首独白惯用形（lowercase table；判定时两侧小写）
    # F4：删「i found」「i discovered」——「I found that X」/「I discovered X」是**结论
    # 段开首**（发现/查清某事），非开首独白惯用形；「i'll check」「let me check」才是
    # 独白带前缀。同中文红线，结论词不收（误杀结论段 = 真丢失）。
    "let me", "i need to", "now i need", "i will", "i'll check",
    "now i", "first i", "i need to check",
)
_MONOLOGUE_WINDOW = 60     # 开首判定窗口（strip 后前 N 字符内命中才算开首独白）


def is_reasoning_monologue(text: str) -> bool:
    """R4 开首独白判定（确定性、零 LLM、纯 stdlib；大小写不敏感）。

    text strip 后前 60 字符内命中 REASONING_MONOLOGUE_CUES 任一 → True。
    空文本 → False。user 段免疫不在此函数（纯文本面），由调用面
    （Tagger.classify 只对 provenance=="model" 生效）保证。
    """
    t = str(text or "").strip().lower()
    if not t:
        return False
    head = t[:_MONOLOGUE_WINDOW]
    return any(c in head for c in REASONING_MONOLOGUE_CUES)


def classify_user(text: str) -> tuple[str, bool]:
    """user-sourced 分级规则（单一实现源：Tagger.classify 与存量重打分 retag.py 共用）。

    - len < 100（短指令）→ ("P0", structural=True)：确定性地板，保护真实用户指令；
    - len ≥ 100 + 系统噪声标记（日志 / 技能提示词）→ ("P2", False)；
    - len ≥ 100 + compaction summary 标记 → ("P1", False)；
    - len ≥ 100 其余（group chat 转发 / 注入块）→ ("P0", False)：
      优先级不变（仍 P0，先于 P1/P2 保留），仅 push cap 溢出时可逐出。
    """
    if len(text) < USER_STRUCTURAL_LEN:
        return ("P0", True)
    if any(m in text for m in _P2_MARKERS):
        return ("P2", False)
    if any(m in text for m in _P1_MARKERS):
        return ("P1", False)
    return ("P0", False)


class Tagger:
    def __init__(self, aux: AuxProvider, session_topic: str) -> None:
        self.aux = aux
        self.session_topic = session_topic

    def classify(self, text: str, provenance: str) -> tuple:
        """返回 (tier, structural, regime, reason[, settled_signal])——**长度可变**。

        契约：user-sourced 短指令（< 100 chars）= P0 structural 地板（确定性，不过 aux）；
        user-sourced 长内容（≥ 100 chars，group chat 转发 / compaction summary 等注入块）
        = P0 non-structural（优先级不变，push cap 溢出时可逐出，注入类 0% structural）；
        model-sourced = aux 端点「tagging」任务 + 机器可解析 JSON。

        4a 数字代号：定案信号码（settled_signal）由 LLM tagging 面顺手打（0/1/2），
        账本写侧纯读码不扫词——多语言干净，词表默认关。

        **返回值长度契约**：user 面 / LLM 面 fail-safe 路径（transient_fail、non-JSON、
        unknown tier）= 4 元组 (tier, structural, regime, reason)；LLM 面成功路径 =
        5 元组 (…, settled_signal)。消费方一律前 4 解包，第 5 位按
        `res[4] if len(res) > 4 else 0` 读（缺省 0 = 不定案全文回喂）。
        保持可变形是向后兼容选择：fail-safe 路径恒 4 元组（老消费方零改动），
        只有 LLM 成功面多吐一位码。
        """
        if provenance == "user":
            tier, structural = classify_user(text)
            if structural:
                return (tier, True, "normal", "user-sourced short instruction structural floor (<100 chars, deterministic)")
            return (tier, False, "normal",
                    "user-sourced long content (group-chat/recall/injected block) -> %s non-structural, evictable under push cap" % tier)

        # R4（2026-10-13 · owner 病灶 a 去独白面）：model 段开首推理独白（"让我…"/
        # "Let me…" 等词表命中）→ 确定性降 P2（不进 P0/P1 压缩注入面 = 不进
        # compact-ledger staged 块；段仍在 store，P2 召回/点查可达 = 不丢数据）。
        # 判定在 LLM tiering **之前**短路：零 LLM 成本，白省一次 aux 调用；
        # 宁漏勿错——未命中词表的段照 LLM 面原样分级（零行为变化）。
        # user 段免疫：独白面只对 provenance=="model" 生效（user 说"让我试试"
        # 是正常指令，走下方 user 分支，本面永不介入）。
        if provenance == "model" and is_reasoning_monologue(text):
            return ("P2", False, "normal",
                    "model-sourced reasoning monologue (open-line cue hit) -> P2 deterministic "
                    "(monologue nudge excluded from P0/P1 inject face; not dropped, still recallable)")

        # LLM tiering + scoring（机器可解析 JSON）
        try:
            raw = self.aux.main_endpoint.call("tagging", text)
            obj = _parse_llm_json(raw)
        except Exception:
            return ("P0", True, "transient_fail", "tagging endpoint transient failure -> floor P0")

        if not isinstance(obj, dict):
            return ("P0", True, "normal", "non-JSON tagging output -> conservative P0 (never P2-drop)")
        tier = obj.get("tier", "P1")
        if tier not in _TIERS_BY_TIER:
            return ("P0", True, "normal", f"unknown tier {tier!r} -> conservative P0")
        try:
            score = int(obj.get("score", 5))
        except (TypeError, ValueError):
            score = 5
        # 边界 ±1（6–8）→ 升一级保守（P2 → P1；P1 → P0；P0 不动）
        if 6 <= score <= 8:
            i = _TIERS_BY_TIER[tier]
            tier = TIERS[max(0, i - 1)]
        # 4a：读 LLM 打的 settled_signal 码（0/1/2，缺省 0 = 不定案全文回喂）
        try:
            settled_signal = int(obj.get("settled_signal", 0))
        except (TypeError, ValueError):
            settled_signal = 0
        if settled_signal not in (0, 1, 2):
            settled_signal = 0
        return (tier, False, "normal",
                f"LLM tiering: tier={obj.get('tier')} score={score} settled_signal={settled_signal}",
                settled_signal)
