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
- L1 索引行 = 「#seq tier ts · key_fact」一行（无 key_fact → head60 + '…' 截断标记；
  2026-10-23 偏离根因修复：原 head20 太碎，compact 两次后模型段降档失忆进行中结论）
- L0 指针   = 「#seq tier [segment_id] on-demand」（预算榨干时的最小足迹）

相关性排序（零 LLM / 零网络 / 确定性）：query 词面子串命中数 desc → tier（P0 先）
→ 新近 desc；S-Phase 3 把本排序升级为 BM25（数据面不动，仅换本函数）。
输出按 seq 稳定序 = prompt-cache 前缀友好（段顺序不随 query 抖动）。
"""
from __future__ import annotations

import logging
import time

from .schema import est_tokens
from . import audit

_obs_logger = logging.getLogger(__name__)


def _l3_text(ws) -> str:
    """L3 全文：resident = 原文；digest_only = digest 摘录（与 distill.WindowSeg.excerpt 同义）。"""
    return ws.excerpt()


# L1 fact head 长度（2026-10-23 偏离根因修复）：无 key_fact 段取 original_text 前 N 字符
# 作 L1 行 fact（原 N=20 太碎 → 模型降档后失忆进行中结论）。截断加 '…' 明示可点查。
_L1_FACT_HEAD = 60


def _l1_fact_head(seg) -> str:
    """无 key_fact 段 L1 fact：original_text 前 60 字符；超长加 '…' 截断标记。"""
    text = seg.original_text or ""
    head = text[:_L1_FACT_HEAD]
    return head + ("…" if len(text) > _L1_FACT_HEAD else "")


def _anchor_ids(segs: list) -> set:
    """任务锚点集合（2026-10-23 偏离根因修复）：最近 user P0 段 + 紧随其后的第一段 model P0。

    语义：compact 两次后，模型必须仍知道"当前任务是什么 + 我刚做到哪"——
    最近 user P0（任务指令）与其**紧邻其后的第一段 model P0**（该指令的结论级响应）
    组成 anchor 对，双双强制 L3 全文豁免（不受细节池 / l3 cap 约束，同 structural P0）。
    P1/P2 filler 不抢 anchor 身份（2026-10-23 test_p3_tiers 实证：P1 filler 段抢 anchor
    会把池归零降级面抢成 L3）。判定确定性（seq 顺序 + provenance + tier，零 LLM）；
    union 窗口双 seq 空间同号时按稳定序 tiebreak（与 _rank_key 同方向）。
    """
    anchors: set[str] = set()
    # 最近 user P0：tier==P0 且 provenance==user，seq 最大者
    user_p0s = [ws for ws in segs
                if ws.segment.tier == "P0" and ws.segment.provenance == "user"]
    if user_p0s:
        last_user = max(user_p0s, key=lambda w: (w.segment.seq, w.segment.content_hash[:16]))
        anchors.add(last_user.segment.segment_id)
        # 末响应：紧邻其后（seq 最小且 > last_user.seq）的第一段 **model P0** 段
        later = [ws for ws in segs
                 if ws.segment.provenance == "model" and ws.segment.tier == "P0"
                 and ws.segment.seq > last_user.segment.seq]
        if later:
            first_model = min(later, key=lambda w: (w.segment.seq, w.segment.content_hash[:16]))
            anchors.add(first_model.segment.segment_id)
    return anchors


def _l1_line(ws, rel: int = 1) -> str:
    """L1 索引行（~20-40t）：块内相对号 + tier + prov + 全局 seq + 时间戳 + key_fact（无则 head60 + '…' 截断标记）。

    R4（2026-10-13 · owner 病灶 c ts 不刷新）：段 ts 为 0/缺（回放段未刷新
    的不可信值，`_ts_zero_ratio` 观测面同口径）→ 省略 ts 段——不新造时间戳
    （造了就是假数据）。
    R5（owner 批次 A #3/#5）：行首 `#<rel>` = 块内相对号（render_block 逐段
    计数传入，账本面 c#N 块号之下不再出现段级全局号 = 双层编号去混淆）；
    `prov=<user|model>` 消 user/model 同号视觉重复（#3）；`seq=<全局seq>`
    保留可点查定位（recall O(1) 面不变）。

    诚实边界（reviewer P6）：R4 原 commit 措辞"正常 ts>0 的行字节逐字不变
    （前缀缓存契约零扰动）"失实——R5 #3/#5 把 L1 行形整体重构（加 prov=、
    #seq→#rel 相对号、+seq= 位），**ts>0 的行字节也随 R5 批次变了**；前缀缓存
    契约在 R5 批次被整体重构打破是 owner 已知事实（账本消息恒在缝位、字节单调
    稳定靠 text() 整段渲染保证，不靠单 L1 行逐字不变）。ts=0 省略 ts 段是 R4 的
    增量，叠加在 R5 新行形之上。"""
    seg = ws.segment
    # 2026-10-23 偏离根因修复：无 key_fact 段 L1 fact 取 head 前 60 字符（原 [:20]
    # 太碎——compact 两次后模型段降档 L1，"I'm checking whether…" 20 字符碎片让模型
    # 失忆自己的进行中结论，拿注入 mem 上下文猜下一步 → 偏离任务）。超长段加 '…'
    # 截断标记，明示"这行只是开头、全文在盘上可点查"，模型知道去 recall.get 捞原文。
    fact = getattr(seg, "key_fact", "") or _l1_fact_head(seg)
    ts = getattr(seg, "ts", 0) or 0
    ts_part = f" ts={ts:.0f}" if ts > 0 else ""
    prov = getattr(seg, "provenance", "") or "?"
    return (f"#{rel} {seg.tier} prov={prov} seq={seg.seq}{ts_part} · {fact}")


def _l0_ptr(ws) -> str:
    return f"#{ws.segment.seq} {ws.segment.tier} [{ws.segment.segment_id}] on-demand"


def _rank_key(ws, query_terms: set[str]) -> tuple:
    """相关性排序键（确定性）：相关度 desc → tier（P0 先）→ 新近 desc（seq 大 = 新）
    → 同号 tiebreak（union 窗口 main+model 双 seq 空间撞车，recon-A H1）：
    seq 仍主键，content_hash[:16] 只在 seq 相等时定全序（同号必不同 hash →
    序唯一 + 跨 replay 稳定 + 前缀缓存顺带稳；零 mutation：不动 seq/segment_id，
    仅排序键扩展第四位，fix-1 t_ad82f732 / gate t_2d213ce9 裁定表 #1）。"""
    hits = 0
    if query_terms:
        low = ws.excerpt().lower()
        hits = sum(1 for t in query_terms if t in low)
    tier_order = {"P0": 0, "P1": 1, "P2": 2}.get(ws.segment.tier, 3)
    return (-hits, tier_order, -ws.segment.seq, ws.segment.content_hash[:16])


# 档位默认池（固定，无 pull 耦合——决定 B 核心，见模块 docstring；与模型窗口解耦）。
# index/detail 参数是扩展缝（默认 None → 常量）；compress() 现不传，保持默认。
DEFAULT_INDEX_BUDGET = 2048
DEFAULT_DETAIL_BUDGET = 4096

# L3 单段 cap（fix-3 t_5bcb33f2 · P2 预算竞争根治）：单段 L3 全文装细节池的 token 上限。
# 根因（验收报告 L3 裁定行）：choose_forms 无单段 cap + P0 distill 豁免 + 42 段 L3 抢 4K 池
# → 单段可独吞细节池。cap = 保守 800 tok（保证单段最多占细节池 4K 的 ~1/5）。
# fail-safe：choose_forms(l3_max_tokens=None) → 800（卡内锁 800 起步，pytest 全绿后可微调）。
# 注入面 = sc/__init__.py 重导出；宿主 config 经参面覆盖，禁裸读 env（reviewer 裁定同源）。
DEFAULT_L3_MAX_TOKENS = 800

# P3 30% 零成本剔除档已于 2026-10-04 撤销（用户拍板 B）：活体 P1=0 → fill_share 恒 0.0，
# 该档永远 no-op。choose_forms 不再接受 fill_prune——P1/P2 回到常规选档（L3/L2/L1/L0 按预算/
# 高信号排），不再被强制锁 L0。telemetry 观测面（filler_form_share / gate_re_add_30pct）保留作审计。


def choose_forms(
    segs: list,
    window,
    query: str = "",
    index_budget: int | None = None,
    detail_budget: int | None = None,
    l3_max_tokens: int | None = None,
) -> list[tuple]:
    """预算驱动选档（核心，确定性）→ [(ws, form, text), ...]，输出 = seq 稳定序。

    form ∈ {"L3","L2","L1","L0"}；预算 = 细节池（L3，默认 4K）+ 索引池（L1+L2 共池，默认 2K）。
    **高压语义**（window.used_tokens > capacity，即物理超窗/backpressure 状态）：细节池归零，
    只出 L1/L2/L0（L2 仍合法——已 digest 段经索引池出 L2，仍在 2K 帽内）→ 注入块 ≤ 索引池
    （≈2K，structural 豁免除外）。structural P0（goal/ledger）L3 豁免细节池
    （活账本永不裁，push cap 语义一致）；其余段按相关性（query 词面命中 → tier → 新近）
    逐段选「装得下的最高档」，超池逐档降级，绝不超预算。细节不丢：L0 指针带 segment_id，
    原文 O(1) 点查可回（recall.get）。

    L3 单段 cap（fix-3）：非 structural 段单段 L3 全文 ≤ l3_max_tokens（None → 800 保守默认）；
    超 cap 段整段不再装细节池 → 降 L2（有 digest）/ L1/L0 指针（无 digest 兜底）。
    structural P0 豁免 cap（活账本永不裁，push cap 语义一致）。

    确定性不变（同 segs/window/query → 同输出）。
    """
    ib = DEFAULT_INDEX_BUDGET if index_budget is None else max(0, int(index_budget))
    db = DEFAULT_DETAIL_BUDGET if detail_budget is None else max(0, int(detail_budget))
    # L3 单段 cap（fix-3）：None → 800 保守默认；宿主 config 经参面覆盖，禁裸读 env。
    cap = DEFAULT_L3_MAX_TOKENS if l3_max_tokens is None else max(0, int(l3_max_tokens))
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
    # 任务锚点（2026-10-23 偏离根因修复）：最近 user P0 + 其模型末响应 强制 L3 豁免
    # （同 structural P0 语义：push cap 内永不降档——compact 后模型仍知"当前任务 +
    # 刚做到哪"）。豁免集随窗口逐次重算（确定性），不影响既有 structural 豁免面。
    # reviewer 必修项（2026-10-23）：anchor 段超 l3 cap → 不再无脑 L3（破 DoD 总帽 6K
    # 实测：anchor 8000 字符全文 block_tokens=2001 顶满）。cap 兜底：anchor 段 L3 成本
    # > cap → 走常规选档（L2/L1/L0），保"可定位+可读"不保"无限全文"。
    anchors = {a for a in _anchor_ids(segs) if costs[a]["l3"] <= cap}

    form_of: dict[str, tuple[str, str]] = {}
    index_used = 0
    # 相关性高者先选高档（structural P0 永不降级，直接 L3）
    for ws in sorted(segs, key=lambda w: _rank_key(w, query_terms)):
        sid = ws.segment.segment_id
        if sid in structural or sid in anchors:
            form_of[sid] = ("L3", _l3_text(ws))
            continue
        c = costs[sid]
        # L3 单段 cap（fix-3）：超 cap 段整段不再装细节池 → 落 L2（有 digest）/ L1/L0 指针（无 digest 兜底）
        if c["l3"] <= detail_pool and c["l3"] <= cap:
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
    # 展示序 = (seq, content_hash[:16]) 升序（fix-1 t_ad82f732 / gate t_2d213ce9 裁定表 #1：
    # union 窗口 main+model 双 seq 空间撞车（recon-A H1）→ 同号段按 hash 前缀 tiebreak
    # 定全序，渲染字节跨 replay / 跨输入序稳定（前缀缓存友好）；零 mutation：seq/segment_id
    # 值不变，仅排序键扩展）
    for ws in sorted(segs, key=lambda w: (w.segment.seq, w.segment.content_hash[:16])):
        form, text = form_of[ws.segment.segment_id]
        out.append((ws, form, text))
    return out


def render_block(rendered: list[tuple]) -> str:
    """选档结果 → 注入块文本（system 消息内容）。壳只调本函数，不持算法体。

    R5（owner 批次 A #3/#5）：行首 `#<rel>` = 本块内相对号（逐段 1..N 计数，
    与账本面 c#N 块号不再混淆）；L3/L2 全文行加 `prov=<user|model>` 消同号
    视觉重复；L1 行在 render_block 内经 `_l1_line(ws, rel)` 重算（choose_forms
    产出的 L1 text 仅供 token 预算估算，渲染面才定相对号/全局 seq 位）；段级
    全局 seq 移入 `seq=` 位（recall 点查面不变），零 LLM、零持久态。"""
    lines = []
    rel = 0
    for ws, form, text in rendered:
        rel += 1
        prov = getattr(ws.segment, "provenance", "") or "?"
        if form in ("L3", "L2"):
            lines.append(f"#{rel} {ws.segment.tier} prov={prov} seq={ws.segment.seq} [{form}]\n{text}")
        elif form == "L1":
            lines.append(_l1_line(ws, rel))
        else:
            lines.append(text)        # L0 指针行（_l0_ptr 原样，高压兜底最小面）
    return "\n---\n".join(lines)


def block_tokens(rendered: list[tuple]) -> int:
    """注入块总 token（验证/DoD 断言用：≤ 细节池 4K + 索引池 2K = 总帽 6K，固定池无 pull 耦合）。"""
    return sum(est_tokens(t) for _, _, t in rendered)


# ── gate §3 可观测面（t_62c8ad6f · 纯 read-only 观测，零数据变更，消费既有机制）────
# 红线：不迁移历史 ts=0 段、不新写 torn-detection（消费 scdata.replay 既有 idx.dropped，
# reason 分布 json_parse_fail / content_hash_mismatch / dup_skip_idempotent 已在盘）；
# 纯 INFO 不告警——ts 阈值裁定面是 fix-2 修复卡，本观测行只报事实。
# fix-5 t_d888d710 在此上加 audit 悬挂侧（<sid>.audit.jsonl O_APPEND+0600，fail-soft 零 raise）。


def _dropped_reasons(idx) -> dict:
    """消费既有 ReplayIndex.dropped（(line_no, reason) 清单）→ reason 分布 dict（按 reason 计数）。

    零 mutation：只读迭代，不改动 idx.dropped / idx.by_hash。"""
    reasons: dict[str, int] = {}
    for _, reason in idx.dropped:
        reasons[reason] = reasons.get(reason, 0) + 1
    return reasons


def _ts_zero_ratio(segs) -> float:
    """消费既有 Segment.ts（窗口 ts=0 段数 / 总段数）→ 占比 float。零 mutation。

    只读 getattr(seg, "ts")，零重算/零迁移；空窗口 → 0.0（无定义，caller 自行判空 no-op）。"""
    if not segs:
        return 0.0
    n_zero = sum(1 for s in segs if (getattr(s, "ts", 0) or 0) == 0)
    return n_zero / len(segs)


def observe_replay_drops(idx) -> None:
    """消费既有 ReplayIndex.dropped（scdata.py 单一实现产出的 torn 行清单），打一行 INFO。

    只读 idx.dropped（(line_no, reason) 元组，replay 权威数据面既有产物），
    零新增语义/完整性校验、零写路径触碰。零 dropped → 一行 N=0 INFO（观测面恒定可 grep）。
    """
    reasons = _dropped_reasons(idx)
    _obs_logger.info(
        "supercompress: replay dropped %d line(s) reason=%s", len(idx.dropped), reasons)


def observe_ts_zero_ratio(segs) -> None:
    """观测行：当前窗口 ts==0 段占比 count/len，纯 INFO 不告警。

    窗口 = 传入的 replay 段列表（engine._window_fresh 磁盘重放产物）；
    空窗口 no-op（无定义）。只读 Segment.ts，零 mutation / 零迁移 / 零重算。
    """
    if not segs:
        return
    n_zero = sum(1 for s in segs if (getattr(s, "ts", 0) or 0) == 0)
    _obs_logger.info(
        "supercompress: ts_zero_ratio=%.4f (n=%d/%d, observation-only, no alerting)",
        n_zero / len(segs), n_zero, len(segs))


def observe_replay_window(idx, segs, sid: str, jsonl_path: str) -> dict:
    """fix-5 双路径观测面统一入口（INFO + audit 悬挂），纯消费既有数据，零 mutation。

    一条 replay/注入事件观测一次 →
      ① INFO 侧：打 dropped（reason 分布）+ ts_zero_ratio 两行（见上面两函数，可 grep）；
      ② audit 悬挂侧：往 `<sid>.audit.jsonl`（audit.audit_path_for_jsonl(jsonl_path)）
         O_APPEND 单行 append 一条 record `{ts, sid, dropped, ts_zero_ratio}`
         （audit.write_audit_record：0600 + flush，写 OSError → WARNING fail-soft 零 raise，
         同 compact_ledger._persist 失败契约）。

    `dropped` = reason 分布 dict（消费既有 idx.dropped）；`ts_zero_ratio` = 窗口 ts=0 占比
    （消费既有 Segment.ts）；`ts` = time.time()（观测事件时间，非段 ts）。
    返回本事件 record dict（含 audit 落盘结果 bool `audit_written`）；即便 audit 写失败
    也照常打 INFO + 返回 record（fail-soft，主流程零感知）。零 store/log/.db 写、
    零 by_hash/ts/seq/content_hash 变更。
    """
    # ① INFO 侧（恒定可 grep，不依赖 audit 写成败）
    observe_replay_drops(idx)
    observe_ts_zero_ratio(segs)
    # ② 算观测 record（纯消费）
    record = {
        "ts": time.time(),
        "sid": sid,
        "dropped": _dropped_reasons(idx),
        "ts_zero_ratio": _ts_zero_ratio(segs),
    }
    # ③ audit 悬挂侧（O_APPEND 单行 + 0600 + flush；fail-soft 零 raise）
    record["audit_written"] = audit.write_audit_record(
        audit.audit_path_for_jsonl(jsonl_path), record, log=_obs_logger)
    return record
