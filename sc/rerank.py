"""Block ⑥ rerank：P3（S-Phase 4）LLM 两步 gate rerank——P1 BM25 之上的独立重排步。

fence（plan P3，用户硬约束贯穿）：
- P1 打分核（sc.recall.search）**零动**：rerank 是独立 step（同 P2 soft_rank 模式），
  只对入参 hits 做**纯重排**，不改 P1 打分 / 数据面 / 点查 / 预算循环；
- LLM 路径**零硬编码模型名 / 零自定义 provider 块**：端点注入
  （生产 = 宿主主模型端点，走 aux 的 main_endpoint.call(task, …) 模式；测试 = FakeEndpoint）；
  唯一 LLM 面变量 = task key（RERANK_TASK），宿主主模型按 task 路由，无 provider 覆盖；
- 门控三选一 + 缓存键是**硬 DoD**（select_context 每 provider 请求都跑——
  不 gate 会把 LLM 调用放大成关键路径，正犯「LLM 崩→风暴」病）：
  ① top-1 score < 低分阈值（话题弱相关）② query 命中显式「搜一下/搜下/search」关键词
  ③ 召回空（hits 全 0 分 / 无 hits）。默认保守：低分阈值 = 0.0 → ① 实际不触发，
  只在「召回空 / 显式搜索词」开，非常态 LLM 调用；
- 缓存键 = (sha256(query), window_version)，window_version = 调用方传入的
  store.line_count（窗口态指纹，单调、O(1)）；命中缓存 → 零 LLM 调用（纯重放排序）；
- fail-safe：LLM 不可用 / 超时 / 解析失败 → **原样返回入参 hits（不降级不炸）**，留痕 log；
  解析失败不写缓存（下次 gate 再触发时重试，不缓存失败面）；
- 确定性：gate 判定纯函数；缓存键纯函数；LLM 重排只在 gate 触发 + 缓存 miss 时发生。

gate 价值面（T2 DoD）：P1 gold recall@3 已 21/21=1.000 饱和——P3 的增量只可能落在
**非饱和探针**（dirty near-dup / CJK 中英混合脏语料，P1 弱面）。饱和集 P3 无增量
（已 1.0）：bench/rerank_probe.py 如实披露，不重测饱和集。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re

from .recall import Hit

logger = logging.getLogger(__name__)

# LLM task key（唯一 LLM 面变量：宿主主模型端点按 task 路由；零 provider / 模型名覆盖）
RERANK_TASK = "auxiliary.supercompress_rerank"
# step1：LLM 对 top-k（来自 P1 BM25）做相关性重排
K_RERANK = 3
# 候选文本截断（LLM 提示有界）
CANDIDATE_CHARS = 512

# gate② 显式搜索关键词（query 小写后子串匹配；CJK 关键词不受 lower 影响）
# R2-v2（2026-10-10 · 语言无关化）：搜索意图词多语化（中/英/日/韩/德/西/法/俄），
# 词表小且可追加——避免单语言（原仅"搜一下/搜下/search"）下其他语言的显式搜索
# 意图漏触发 gate② → LLM 重排面不补 → 弱相关面漏兜。子串匹配对 CJK 无大小写问题。
EXPLICIT_SEARCH_KEYWORDS = (
    "搜一下", "搜下", "搜索",                       # 中
    "search", "look up", "lookup",                  # 英
    "検索", "検索して",                             # 日
    "검색", "찾아",                                 # 韩
    "suche", "durchsuche", "durchsuchen",          # 德
    "buscar", "búsqueda",                            # 西
    "recherche",                                    # 法
    "поиск", "поиски",                              # 俄
)


def _query_digest(query: str) -> str:
    return hashlib.sha256((query or "").encode("utf-8")).hexdigest()


def rerank_cache_key(query: str, window_version: int) -> tuple:
    """缓存键 = (sha256(query), window_version)：纯函数（窗口指纹 O(1) 单调）。"""
    return (_query_digest(query or ""), int(window_version or 0))


class RerankGate:
    """三选一 gate（纯函数、确定性；同输入两次判定全等）。

    默认保守（low_score_threshold=0.0）：BM25 score ≥ 0 → ① 实际不触发，
    只在「召回空（③）/ 显式搜索词（②）」时开——非常态 LLM 调用（硬 DoD）。

    注：0.0 只是 sc 层独立构造的默认；生产 engine 构造时覆写为 3.0
    （engine.py:366 `float(_get("rerank_low_score_threshold", 3.0))`，tests/test_rerank.py:315
    `assert e.rerank_low_score_threshold == 3.0` 钉死）——勿把 sc 层默认读成生产 gate 阈值，
    正常搜索面 LLM 放大率由 3.0 决定。
    """

    def __init__(self, low_score_threshold: float = 0.0) -> None:
        self.low_score_threshold = float(low_score_threshold)

    def should_fire(self, hits: list, query: str) -> bool:
        # ③ 召回空（无 hits / 全 0 分）
        if not hits or all(h.score <= 0.0 for h in hits):
            return True
        # ① 话题弱相关：top-1 score < 低分阈值（默认 0.0 → 不触发）
        top1 = max(h.score for h in hits)
        if top1 < self.low_score_threshold:
            return True
        # ② query 命中显式「搜一下/搜下/search」关键词
        q = (query or "").lower()
        return any(k in q for k in EXPLICIT_SEARCH_KEYWORDS)


_RERANK_INSTRUCTION = (
    "You are a retrieval reranker. Reorder the candidate memory segments by relevance "
    "to the query (most relevant first). Use ONLY the given candidate ids, each exactly once. "
    "Reply with strict JSON only (no code fence, no extra text): "
    '{"order": ["<id>", ...]}\n'
    "--- query ---\n"
)


def _rerank_prompt(query: str, hits: list) -> str:
    """step1 提示：top-k 候选（截断 CANDIDATE_CHARS）+ 结构化 JSON 排序要求。确定性。"""
    lines = [_RERANK_INSTRUCTION, (query or "").strip(), "--- candidates ---"]
    for i, h in enumerate(hits[:K_RERANK]):
        seg = h.seg
        lines.append(
            f"[{i}] id={seg.segment_id} tier={seg.tier} score={h.score:.3f}: "
            f"{(seg.original_text or '')[:CANDIDATE_CHARS]}")
    return "\n".join(lines)


def _balanced_json_candidates(t: str) -> list:
    """括号平衡法：从 t 里依次提取所有「首个 { 起、配平 } 结束」的候选片段。

    正确处理字符串内的 { } 和转义引号，避免贪婪正则 \\{.*\\} 误抓散文里自带的
    片段。提取失败/截断（配不平）返回空 list —— 调用方 json.loads 自然走 fail-safe。
    """
    out = []
    i = 0
    n = len(t)
    while i < n:
        if t[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escaped = False
        j = i
        while j < n:
            ch = t[j]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(t[i:j + 1])
                        i = j + 1
                        break
            j += 1
        else:
            break  # 配不平（截断）：剩余部分不作为候选
    return out


def parse_rerank_response(text: str, valid_ids: list) -> "list[str] | None":
    """LLM 响应 → 重排 id 列表（top-k 的**严格全排列**）；失败 → None（fail-safe）。

    容忍裸 JSON 对象 {"order": […]} / 裸数组 / code fence 包裹；
    思考模型（reasoning 模型）实证失败面（2026-10-10 devops 日志 sc.rerank not parseable）：
    content 前缀带 \\n\\n + fence + 索引数字（"0","1","2"）代替完整 segment_id——
    提取层额外塞「整数索引面」：order 元素若是 0..k-1 的无重复整数/整数字符串，
    按 prompt 里候选的顺序映射回 valid_ids（与全排列校验同强度：不重不缺不多）。
    多对象 / 纯文本 / 缺 id / 重复 id / 多 id / 空响应 → None（调用方原样返回入参 hits）。
    """
    if not text or not str(text).strip():
        return None
    t = str(text).strip()
    candidates = [t]
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        candidates.insert(0, m.group(1).strip())
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        candidates.insert(1, m.group(0))
    # thinking/散文前缀混入（reasoning 被并进 message.content，可能自带 { } 和 " 字符串），
    # 贪婪正则 \{.*\} 会误抓散文里第一个 { 起的片段 → 额外用括号平衡法从首个 { 提取
    # 第一个「平衡」JSON 对象，作为独立候选（json.loads 失败则走原 fail-safe，语义不变）
    candidates.extend(_balanced_json_candidates(t))
    for c in candidates:
        try:
            d = json.loads(c)
        except Exception:
            continue
        if isinstance(d, dict):
            if "order" not in d:
                continue
            d = d["order"]
        if isinstance(d, list) and all(isinstance(x, str) for x in d):
            if sorted(d) == sorted(valid_ids):   # 严格全排列：不缺、不重、不多
                return d
        # 索引面：思考模型（实证 9/10 翻车）回 ["0","2","1"] 或 [2,0,1]（候选序号）
        if isinstance(d, list) and 0 < len(d) == len(valid_ids):
            idx = []
            ok = True
            for x in d:
                if isinstance(x, int) and not isinstance(x, bool):
                    idx.append(x)
                elif isinstance(x, str) and x.isdigit():
                    idx.append(int(x))
                else:
                    ok = False
                    break
            if ok and sorted(idx) == list(range(len(valid_ids))):
                return [valid_ids[i] for i in idx]
    return None


def rerank_hits(hits, query, llm_endpoint, *, cache, window_version,
                gate: "RerankGate | None" = None,
                drop_model_on_gate: bool = True) -> list:
    """P3 两步 rerank（gate 三选一 + 缓存键 + fail-safe；P1 打分核零动）。

    1. gate（纯函数）：不触发 → 原样返回（零 LLM 调用）；
    2. 缓存键 (sha256(query), window_version) 命中 → 零 LLM 调用，纯重放缓存排序；
    3. step1 LLM 对 top-k（P1 BM25）做相关性重排（结构化 JSON，端点注入：
       生产 = 宿主主模型端点 aux 路由模式，测试 = FakeEndpoint）；
       LLM 不可用 / 超时 / 解析失败 → step2 **原样返回入参 hits（不降级不炸）**，留痕 log。

    R4（2026-10-10 · 缺口 B）：gate 触发时（弱相关/空召回/显式搜索，= 不确定性高面）
    对 `provenance=="model"` 的 hit 段默认 drop（不进 top-k 候选、不写缓存、不进返回集），
    而非 keep-all——agent 自述段（P0 model）在不确定召回面里不该冒充用户级 P0 被 LLM
    重新排序保留。`drop_model_on_gate=False` 回滚为原 keep-all 现状（零变化 fail-safe）。
    确定性：drop 判定纯函数（只读 hit.seg.provenance 字段，零 LLM / 零 wall-clock）。

    确定性：gate 判定纯函数；缓存键纯函数；LLM 重排只在 gate 触发 + 缓存 miss 时发生。
    """
    gate = gate if gate is not None else RerankGate()
    if not gate.should_fire(hits, query):
        return hits
    if not hits:                                  # 召回空且无候选可排：无事可做（不发起 LLM）
        return hits
    # R4：gate 触发（不确定性高面）→ 默认 drop model-provenance 段（agent 自述，非用户拍板）。
    # user-provenance 段（真用户 P0/P1）全保留。drop_model_on_gate=False → 不 drop（原 keep-all）。
    if drop_model_on_gate:
        _dropped = [h for h in hits if getattr(h.seg, "provenance", "") == "model"]
        if _dropped:
            logger.info("supercompress: rerank gate fired → drop %d model-provenance "
                        "hit(s) (R4: agent self-report segments not kept on uncertain recall)",
                        len(_dropped))
        hits = [h for h in hits if getattr(h.seg, "provenance", "") != "model"]
        if not hits:                              # 全 drop 后空召回 → 无事可做（不发起 LLM）
            return hits

    key = rerank_cache_key(query, window_version)
    cached = cache.get(key) if cache is not None else None
    if cached is not None:                        # 命中缓存 → 零 LLM 调用（纯重放）
        by_id = {h.seg.segment_id: h for h in hits}
        cached_set = set(cached)
        out = [by_id[i] for i in cached if i in by_id]
        out += [h for h in hits if h.seg.segment_id not in cached_set]
        return out

    topk = hits[:K_RERANK]
    valid_ids = [h.seg.segment_id for h in topk]
    try:
        response = llm_endpoint.call(RERANK_TASK, _rerank_prompt(query, topk))
    except Exception as e:
        # step2 fail-safe：LLM 不可用 / 超时 → 原样返回入参（不降级不炸），留痕
        logger.warning("supercompress: rerank LLM unavailable/timeout (%s) "
                       "-> fail-safe, hits returned unchanged", e)
        return hits
    order = parse_rerank_response(response, valid_ids)
    if order is None:
        # step2 fail-safe：解析失败（多对象 / 纯文本 / 缺 id）→ 原样返回，留痕；不写缓存
        logger.warning("supercompress: rerank LLM response not parseable "
                       "-> fail-safe, hits returned unchanged")
        return hits

    by_id = {h.seg.segment_id: h for h in topk}
    rest = [h for h in hits if h.seg.segment_id not in set(valid_ids)]
    out = [by_id[i] for i in order] + rest
    if cache is not None:
        # 缓存成功重排（全量序 = top-k LLM 序 + 剩余原序）；失败面不缓存
        cache[key] = [i for i in order] + [h.seg.segment_id for h in rest]
    return out
