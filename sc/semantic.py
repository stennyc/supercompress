"""Block ⑦ semantic（precision P4 · 边界宿主 LLM 语义扩展，「花最少」的语义通道）。

只在 **drift 边界那一下**花**一次**宿主 LLM（G3 铁律最硬）：
- 端点 = 宿主主模型端点（engine `aux.main_endpoint`：宿主 `update_model()` 注入
  `call_llm`+`main_runtime` 的 `_CallLlmEndpoint`，两级路由 aux 短超时 → 主模型兜底）——
  **不外部、不 ONNX、不硬编码模型名、不建自定义 provider/API 配置块**；
  唯一 LLM 面变量 = task key（`BOUNDARY_TASK`，与 P3 `RERANK_TASK` 同族，宿主按 task 路由）；
- 常驻（非 drift 边界）路径 **0 LLM 调用**（G4 确定性基座 100% 不破，非确定性只存在于此边界）；
- 边界一次 LLM 干**三件**（同一提示词、同一响应）：
  ① 查询扩展：当前任务关键词集 → 同义/别名/相关标识符（**词面扩词，不是向量**），
     扩完由 engine 喂回 P1 extended-BM25 的 query 面（Recaller.search 的 tokenize 词面通道）；
  ② top-N 精排：对 BM25 粗排的 top-N 候选判 keep/drop（破近义误召；fail-safe keep-all）；
  ③ 重建 epoch 锚点：扩展词面并入 `DriftTracker.anchor`（与 P0 合流——epoch 锚点 =
     任务关键词集，拆 3.4% 填充那堵墙；session_topic 整数 epoch 标 / jsonl 字节零动）。

克制（「花最少」硬 DoD，验收会查）：
- **per-epoch 缓存**：`cache: dict[int epoch -> BoundaryResult]`，同 epoch 不重算（命中 → 0 LLM）；
  会话边界由 engine 清零（防跨 session 泄漏）；失败/回退结果**不进缓存**（下边界可重试）；
- **超时/瞬态失败 → 回退纯 BM25**：复用 aux 两级路由的 cooldown 阶梯**模式**（本模块 2 档，
  默认 30s→180s 封顶，短于 digest 60→300→1800s——边界 LLM 更低频），冷却窗内**不调 LLM**，
  返回 degraded 保守面（扩词空、精排 keep-all）；端点恢复（一次成功）→ 连败归零、冷却窗关闭；
  冷却窗内跳段不累加连败（退避不自我放大，与 aux R3 同语义）；
- 提示词/解析全部**纯函数**（同输入同输出；候选按入参序截断 K_RERANK、verdicts 严格全排校验），
  解析失败 → None → 调用方 fail-safe（不 raise、不写缓存、不炸宿主 turn）。

端点注入（生产 = 宿主主模型端点代理，测试 = FakeEndpoint，住 tests/fakes.py，不进生产 import 面）。
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# 唯一 LLM 面变量 = task key（宿主主模型按 task 路由；零 provider / 模型名覆盖，G3；与 P3 RERANK_TASK 同族）
BOUNDARY_TASK = "auxiliary.supercompress_boundary"
# top-N 精排候选数（plan P4「BM25 粗排的 3 个候选」；超出截断，提示词有界）
K_RERANK = 3
# 单候选提示词文本截断（有界）
CANDIDATE_CHARS = 512
# 词面扩展上限（「花最少」：扩词集合有界）
MAX_EXPAND_TERMS = 24
# seed 词面上限（提示词有界）
SEED_CAP = 40
# 当前任务原文上下文截断（提示词有界）
TASK_CONTEXT_CHARS = 256
# 冷却阶梯（P4 边界 LLM 独立 2 档，短于 aux digest 的 60→300→1800s——边界 LLM 更低频；
# 全 0 → 冷却窗恒关，fail-safe「单事件自愈」语义不变）
DEFAULT_COOLDOWN_LADDER: tuple[float, ...] = (30.0, 180.0)

# Strip <thinking>...</thinking> / <reasoning>...</reasoning> blocks that
# reasoning models (e.g. agnes-2.5-flash) embed — those blocks may contain
# { or } characters that would break downstream greedy JSON extraction.
_THINK_RE = re.compile(
    r"<thinking[^>]*>.*?</thinking>|<reasoning[^>]*>.*?</reasoning>",
    re.DOTALL | re.IGNORECASE,
)


def _boundary_instruction(seed_terms: List[str], candidates: List, task_text: str) -> str:
    """P4 边界 LLM 提示词（确定性：纯字面，同输入同输出；宿主主模型文进文出，零向量）。

    一次提示词承载三动作：expanded（扩词，≤24 短词面）/ keep+drop（候选严格全排，
    keep-all on uncertainty）/ 当前任务上下文（触发 drift 的 user 段原文，截断有界）。
    """
    topk = list(candidates)[:K_RERANK]
    lines = [
        "You are a precision retrieval assistant at a task-switch (drift) boundary. "
        "Given the CURRENT task context, its seed terms, and the top coarse-recall candidate "
        "segments, do THREE things in ONE reply. Output strict JSON only (no code fence, no extra text):",
        '{"expanded": ["<=24 short synonyms/aliases/related identifiers of the seed terms"], '
        '"keep": ["<candidate id> that TRULY belong to the current task"], '
        '"drop": ["<candidate id> to discard (near-dup / off-task)"]}',
        "",
        "Rules: keep+drop must partition the candidate id set exactly (each id in exactly one). "
        "When unsure, prefer KEEP (never drop on uncertainty). expanded = short lexical terms ONLY "
        "(synonyms/aliases/related identifiers), no vectors, no full sentences.",
        "",
        "--- current task context ---",
        (task_text or "")[:TASK_CONTEXT_CHARS],
        "",
        "--- seed terms (current task keywords) ---",
        " ".join(seed_terms[:SEED_CAP]) if seed_terms else "(none)",
        "",
        "--- candidate segments ---",
    ]
    for i, c in enumerate(topk):
        lines.append(f"[{i}] id={c.seg.segment_id}: {(c.seg.original_text or '')[:CANDIDATE_CHARS]}")
    lines.append("--- reply with strict JSON only ---")
    return "\n".join(lines)


def parse_boundary_response(raw: "str | None", candidate_ids: List[str]):
    """LLM 响应 → (expanded_terms, verdicts)；失败 → None（调用方 fail-safe 回退纯 BM25）。

    容忍：裸 JSON 对象 / code-fence 包裹；keep+drop 必须 == 候选 id 全集（每个 id 恰好一次，
    缺/重/多 → None）。expanded 容忍 list[str] 或逗号/空白分隔 str；非 list/str → []；
    截断到 MAX_EXPAND_TERMS。verdicts 按入参 candidate_ids 顺序输出（确定性，无 dict 序）。
    """
    if raw is None:
        return None
    t = str(raw).strip()
    if not t:
        return None
    # Reasoning models embed <thinking>/<reasoning> blocks containing { } —
    # strip them first so the greedy \{.*\} below can't swallow them.
    t = _THINK_RE.sub("", t).strip()
    cands: list = [t]
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        cands.insert(0, m.group(1).strip())
    m = re.search(r"\{.*?\}", t, re.DOTALL)
    if m:
        cands.insert(1, m.group(0))
    for c in cands:
        try:
            d = json.loads(c)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        keep = d.get("keep")
        drop = d.get("drop")
        if not isinstance(keep, list) or not isinstance(drop, list):
            continue
        keep = [str(x) for x in keep]
        drop = [str(x) for x in drop]
        if sorted(keep + drop) != sorted(candidate_ids):
            continue
        keep_set = set(keep)
        verdicts = {cid: ("keep" if cid in keep_set else "drop") for cid in candidate_ids}
        exp = d.get("expanded")
        if exp is None:
            exp = []
        if isinstance(exp, str):
            exp = [s for s in re.split(r"[,\s]+", exp) if s]
        elif isinstance(exp, list):
            exp = [str(s).strip() for s in exp if str(s).strip()]
        else:
            exp = []
        return (exp[:MAX_EXPAND_TERMS], verdicts)
    return None


@dataclass
class BoundaryResult:
    """P4 边界 LLM 一次调用的三动作结果（per-epoch 缓存体）：

    - expanded_terms：扩词（同义/别名/相关标识符，词面；engine 喂回 extended-BM25 query 面）；
    - verdicts：候选 id → "keep"/"drop"（精排，破近义误召；缺 = 保守 keep-all）；
    - expanded_seed：seed + expanded（重建 epoch 锚点的词面，审计面）；
    - degraded：LLM 失败回退纯 BM25 标志（观测面；degraded 结果不进缓存）。
    """
    expanded_terms: list = field(default_factory=list)
    verdicts: dict = field(default_factory=dict)
    expanded_seed: list = field(default_factory=list)
    degraded: bool = False


class BoundaryExpander:
    """P4 边界 LLM 扩词/精排/锚点 + per-epoch 缓存 + 冷却回退（宿主主模型，0 外部调用）。

    - `endpoint`：宿主主模型端点（engine 注入代理；测试 = FakeEndpoint）；`.call(task, text)` → 文本；
    - `cache`：per-epoch 缓存 dict（int epoch → BoundaryResult；engine 会话边界清零）——
      同 epoch 命中 → 0 LLM 调用（「花最少」核心）；失败/回退结果不进缓存（下边界可重试）；
    - `cooldown_ladder`：冷却阶梯（默认 30s→180s 封顶 2 档；全 0 → 冷却窗恒关）；
      LLM 异常 → 连败 +1、冷却窗 = ladder[min(n-1, len-1)]，窗内 expand **不调 LLM**
      （degraded 纯 BM25 回退）；一次成功 → 连败归零、冷却窗关闭回落基线；
      冷却窗内跳段不累加连败（退避不自我放大，与 aux R3 同语义）；
    - `monotonic`：可注入单调时钟（测试模拟冷却推进，零真实 sleep）；默认 time.monotonic。
    """

    def __init__(self, endpoint, *, cache: Optional[dict] = None,
                 cooldown_ladder: tuple = DEFAULT_COOLDOWN_LADDER,
                 monotonic: Optional[Callable[[], float]] = None) -> None:
        self.endpoint = endpoint
        self.cache: dict = {} if cache is None else cache
        self.cooldown_ladder: tuple = tuple(cooldown_ladder)
        self._monotonic = monotonic if monotonic is not None else time.monotonic
        self.consec_fails: int = 0
        self._cooldown_until: float = 0.0

    # ── 冷却面（纯阶梯索引；端点恢复复位；测试可注入 monotonic 模拟冷却推进）──
    def _cooldown(self) -> float:
        if not self.cooldown_ladder:
            return 0.0
        return self.cooldown_ladder[min(max(self.consec_fails, 1) - 1, len(self.cooldown_ladder) - 1)]

    def at_cooldown(self) -> bool:
        """是否在冷却窗内（窗内 → 不调 LLM，纯 BM25 回退；阶梯全 0 / 冷却窗关闭 → False）。"""
        if not self.cooldown_ladder:
            return False
        return (self._cooldown_until - self._monotonic()) > 0.0

    def cooldown_status(self) -> dict:
        """观测面（一行可 grep，与 aux R3 cooldown_status 同族）：连败 / 冷却剩余 / 阶梯快照。"""
        remaining = max(0.0, self._cooldown_until - self._monotonic())
        return {
            "consec_fails": self.consec_fails,
            "cooldown_remaining_s": round(remaining, 3),
            "ladder_s": [f"{c:g}" for c in self.cooldown_ladder],
        }

    def _reset_success(self) -> None:
        """端点恢复（一次成功）→ 连败归零、冷却窗关闭回落基线（状态翻转打一行 INFO）。"""
        if self.consec_fails or self._cooldown_until:
            logger.info("sc-semantic: boundary endpoint recovered (fail streak %d -> 0, cooldown cleared)",
                        self.consec_fails)
        self.consec_fails = 0
        self._cooldown_until = 0.0

    def _fail_cooldown(self) -> None:
        """LLM 失败 → 连败 +1、冷却窗按阶梯拉长（回退纯 BM25；退避不自我放大由 at_cooldown 跳段承接）。"""
        self.consec_fails += 1
        cd = self._cooldown()
        if cd > 0:
            self._cooldown_until = self._monotonic() + cd
            logger.info("sc-semantic: boundary LLM fail streak=%d cooldown=%.0fs -> pure-BM25 fallback",
                        self.consec_fails, cd)

    # ── 三动作（一次 LLM）──
    def expand(self, epoch: int, seed_terms: List[str], candidates: List, task_text: str = "") -> BoundaryResult:
        """drift 边界那一下：per-epoch 缓存命中 / 冷却窗内 → 不重算（0 LLM）；否则一次 LLM 干三件。

        - 缓存命中（同 epoch）→ 直接返回缓存结果（0 LLM 调用，「花最少」）；
        - 冷却窗内（上次失败退避中）→ degraded 纯 BM25 保守面（不调 LLM、不写缓存）；
        - LLM 异常 / 响应不可解析 → degraded 保守面（扩词空、精排 keep-all、词面 = 原 seed），
          连败 +1 + 冷却窗（阶梯），**不进缓存**（下次边界事件可重试）；
        - 成功 → 缓存 per-epoch + 连败归零 + 冷却窗关闭。
        候选统一截断 K_RERANK（提示词有界）；verdicts 对该截断集做严格全排校验（缺一/重/多 → 全 fail-safe）。
        """
        if epoch in self.cache:
            return self.cache[epoch]                       # per-epoch 命中 → 0 LLM
        if self.at_cooldown():
            # 冷却窗内：不调 LLM（纯 BM25 保守回退；跳段不累加连败——退避不自我放大）
            return BoundaryResult(expanded_terms=[], verdicts={},
                                  expanded_seed=list(seed_terms), degraded=True)
        topk = list(candidates)[:K_RERANK]
        prompt = _boundary_instruction(seed_terms, topk, task_text)
        cand_ids = [c.seg.segment_id for c in topk]
        try:
            raw = self.endpoint.call(BOUNDARY_TASK, prompt)
        except Exception as e:
            self._fail_cooldown()
            logger.warning("sc-semantic: boundary LLM unavailable/timeout (%s) -> pure-BM25 fallback", e)
            return BoundaryResult(expanded_terms=[], verdicts={},
                                  expanded_seed=list(seed_terms), degraded=True)
        parsed = parse_boundary_response(raw, cand_ids)
        if parsed is None:
            self._fail_cooldown()
            logger.warning("sc-semantic: boundary LLM response not parseable -> pure-BM25 fallback (keep-all)")
            return BoundaryResult(expanded_terms=[], verdicts={},
                                  expanded_seed=list(seed_terms), degraded=True)
        exp, verdicts = parsed
        self._reset_success()
        res = BoundaryResult(expanded_terms=list(exp), verdicts=verdicts,
                             expanded_seed=list(seed_terms) + list(exp), degraded=False)
        self.cache[epoch] = res                            # per-epoch 缓存（成功面才进）
        return res

    # ── 引擎侧读面（select_context 消费；cache miss / degraded → 保守默认，0 LLM）──
    def cached(self, epoch: int) -> "BoundaryResult | None":
        """该 epoch 缓存体（无 → None；degraded 结果不入缓存，故非 None 即成功面）。"""
        return self.cache.get(epoch)

    def expanded_terms(self, epoch: int) -> List[str]:
        """该 epoch 的扩词（喂回 extended-BM25 query 面）；无缓存 / degraded → []（纯 BM25 原 query）。"""
        r = self.cache.get(epoch)
        if r is None or r.degraded:
            return []
        return list(r.expanded_terms)

    def verdicts(self, epoch: int) -> dict:
        """该 epoch 精排 keep/drop；无缓存 / degraded → {}（调用方 keep-all 保守面，不删候选）。"""
        r = self.cache.get(epoch)
        if r is None or r.degraded:
            return {}
        return dict(r.verdicts)
