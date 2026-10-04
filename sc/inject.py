"""阶梯注入（S-Phase 2 / 决定 B）：索引 + 分档披露，预算驱动选档。

现状缺陷（S1）：engine.compress() 把**全部** P0/P1 的 original_text 拼进一条 system
消息——464 行 session 轻松几万 token 每 turn 重发，窗口单调膨胀。
本模块根治：注入 = L1 索引行（全量，每段 ~30t）+ 细节预算内 L3 全文 / L2 摘要，
超预算逐档降级（L3→L2→L1→L0 指针），**固定池、无 pull 耦合**：索引池 2K +
细节池 4K（总帽 6K，≤ DoD 8K）——与模型窗口解耦是决定 B 的核心：pull 耦合会让
注入块随窗口放大（256K 模型 → 51K 块），抵消窗口单调膨胀的修复。细节不丢：
原文在盘上 O(1) 点查（recall.get），L0 指针带 segment_id。

档位语义（段级，逐段选「装得下的最高档」）：
- L3 全文   = 原文（digest_only 段 = digest 摘录，与 distill.WindowSeg.excerpt 同义）；
  structural P0（goal/ledger 活账本）**永不降级**（push cap 语义：活账本永不裁）
- L2 摘要   = digest_text（LLM summary；无 digest 段不可用 L2 → 直落 L1）
- L1 索引行 = 「#seq tier ts · key_fact」一行（无 key_fact → head20）
- L0 指针   = 「#seq tier [segment_id] on-demand」（预算榨干时的最小足迹）

相关性排序（零 LLM / 零网络 / 确定性）：query 词面子串命中数 desc → tier（P0 先）
→ 新近 desc；S-Phase 3 把本排序升级为 BM25（数据面不动，仅换本函数）。
输出按 seq 稳定序 = prompt-cache 前缀友好（段顺序不随 query 抖动）。
"""
from __future__ import annotations

from .schema import est_tokens


def _l3_text(ws) -> str:
    """L3 全文：resident = 原文；digest_only = digest 摘录（与 distill.WindowSeg.excerpt 同义）。"""
    return ws.excerpt()


def _l1_line(ws) -> str:
    """L1 索引行（~20-40t）：seq + tier + 时间戳 + key_fact（无则 head20）。"""
    seg = ws.segment
    fact = getattr(seg, "key_fact", "") or seg.original_text[:20]
    return f"#{seg.seq} {seg.tier} ts={seg.ts:.0f} · {fact}"


def _l0_ptr(ws) -> str:
    return f"#{ws.segment.seq} {ws.segment.tier} [{ws.segment.segment_id}] on-demand"


def _rank_key(ws, query_terms: set[str]) -> tuple:
    """相关性排序键（确定性）：相关度 desc → tier（P0 先）→ 新近 desc（seq 大 = 新）。"""
    hits = 0
    if query_terms:
        low = ws.excerpt().lower()
        hits = sum(1 for t in query_terms if t in low)
    tier_order = {"P0": 0, "P1": 1, "P2": 2}.get(ws.segment.tier, 3)
    return (-hits, tier_order, -ws.segment.seq)


# 档位默认池（固定，无 pull 耦合——决定 B 核心，见模块 docstring；与模型窗口解耦）。
# index/detail 参数是扩展缝（默认 None → 常量）；compress() 现不传，保持默认。
DEFAULT_INDEX_BUDGET = 2048
DEFAULT_DETAIL_BUDGET = 4096

# P3 30% 零成本剔除档已于 2026-10-04 撤销（用户拍板 B）：活体 P1=0 → fill_share 恒 0.0，
# 该档永远 no-op。choose_forms 不再接受 fill_prune——P1/P2 回到常规选档（L3/L2/L1/L0 按预算/
# 高信号排），不再被强制锁 L0。telemetry 观测面（filler_form_share / gate_re_add_30pct）保留作审计。


def choose_forms(
    segs: list,
    window,
    query: str = "",
    index_budget: int | None = None,
    detail_budget: int | None = None,
) -> list[tuple]:
    """预算驱动选档（核心，确定性）→ [(ws, form, text), ...]，输出 = seq 稳定序。

    form ∈ {"L3","L2","L1","L0"}；预算 = 细节池（L3，默认 4K）+ 索引池（L1+L2 共池，默认 2K）。
    **高压语义**（window.used_tokens > capacity，即物理超窗/backpressure 状态）：细节池归零，
    只出 L1/L2/L0（L2 仍合法——已 digest 段经索引池出 L2，仍在 2K 帽内）→ 注入块 ≤ 索引池
    （≈2K，structural 豁免除外）。structural P0（goal/ledger）L3 豁免细节池
    （活账本永不裁，push cap 语义一致）；其余段按相关性（query 词面命中 → tier → 新近）
    逐段选「装得下的最高档」，超池逐档降级，绝不超预算。细节不丢：L0 指针带 segment_id，
    原文 O(1) 点查可回（recall.get）。

    确定性不变（同 segs/window/query → 同输出）。
    """
    ib = DEFAULT_INDEX_BUDGET if index_budget is None else max(0, int(index_budget))
    db = DEFAULT_DETAIL_BUDGET if detail_budget is None else max(0, int(detail_budget))
    high_pressure = getattr(window, "used_tokens", 0) > getattr(window, "capacity_tokens", 0)
    detail_pool = 0 if high_pressure else db

    query_terms = {t for t in query.lower().split() if len(t) >= 2} if query else set()

    # 每段成本（确定性 token 估计，schema.est_tokens 单源）
    costs: dict[str, dict] = {}
    for ws in segs:
        seg = ws.segment
        has_digest = bool(getattr(seg, "digest_text", ""))
        costs[seg.segment_id] = {
            "l3": est_tokens(_l3_text(ws)),
            "l2": est_tokens(seg.digest_text) if has_digest else None,
            "l1": est_tokens(_l1_line(ws)),
            "l0": est_tokens(_l0_ptr(ws)),
            "digest": has_digest,
        }
    structural = {ws.segment.segment_id for ws in segs if ws.segment.structural_p0}

    form_of: dict[str, tuple[str, str]] = {}
    index_used = 0
    # 相关性高者先选高档（structural P0 永不降级，直接 L3）
    for ws in sorted(segs, key=lambda w: _rank_key(w, query_terms)):
        sid = ws.segment.segment_id
        if sid in structural:
            form_of[sid] = ("L3", _l3_text(ws))
            continue
        c = costs[sid]
        if c["l3"] <= detail_pool:
            form_of[sid] = ("L3", _l3_text(ws))
            detail_pool -= c["l3"]
        elif c["digest"] and c["l2"] is not None and index_used + c["l2"] <= ib:
            form_of[sid] = ("L2", ws.segment.digest_text)
            index_used += c["l2"]
        elif c["l1"] <= ib and index_used + c["l1"] <= ib:
            form_of[sid] = ("L1", _l1_line(ws))
            index_used += c["l1"]
        else:
            form_of[sid] = ("L0", _l0_ptr(ws))

    out = []
    for ws in sorted(segs, key=lambda w: w.segment.seq):  # 输出 = seq 稳定序（缓存友好）
        form, text = form_of[ws.segment.segment_id]
        out.append((ws, form, text))
    return out


def render_block(rendered: list[tuple]) -> str:
    """选档结果 → 注入块文本（system 消息内容）。壳只调本函数，不持算法体。"""
    lines = []
    for ws, form, text in rendered:
        if form in ("L3", "L2"):
            lines.append(f"#{ws.segment.seq} {ws.segment.tier} [{form}]\n{text}")
        else:
            lines.append(text)
    return "\n---\n".join(lines)


def block_tokens(rendered: list[tuple]) -> int:
    """注入块总 token（验证/DoD 断言用：≤ 细节池 4K + 索引池 2K = 总帽 6K，固定池无 pull 耦合）。"""
    return sum(est_tokens(t) for _, _, t in rendered)
