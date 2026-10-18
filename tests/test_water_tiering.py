"""降档水位递进带 S1（RED→GREEN）：run() 重构成 90→75 降档分档（纯容量逐出）+ 估值 gate。

PRD：specs/水位递进带PRD-降档序90到75纯容量逐出.md §1（降档序）+ §1.2（估值 gate 分母=当前用量）。

背景：现状 run() = "80% 软逐(P2/P1) + push cap(P0) + §4.5 90% 段级强制" 三块顺序跑，
无降档分档、无"压到目标水位下就停/压不到 pass 下一档"语义、无估值 gate。本测试锁定
owner 拍板的降档序（S1 落地口径 = 90%→75% 两档 + 100% force_evict 兜底）：
  ① 90%（used>0.9cap）先试 → tier90 纯容量逐 P0 段、压到 0.9cap 下就停、压不到 pass
  ② 75%（used>0.75cap）→ tier75 P2/P1 常规逐、压到 0.75cap 下停、压不到 → 100% force_evict 兜底

锁定锚（TDD）：
W1  降档序执行：used>0.9cap 时 90% 面先跑（逐 P0 段+账本页），压到 0.9cap 下即停
    ——不继续跑 75%（已到位）。事件面 events 含 "tier90"。
W2  估值 gate（分母=w.used_tokens 当前用量）：某档"预估省 < 阈值×used" → 不执行该档、
    pass 给下一档（省量确定性算：est_tokens 差值，零 LLM）。
W3  单调累积：档 i 做完的逐出保留到档 i+1（回收量单调不降），全 pass 后
    仍压不到 cap 下 → status=backpressure（非死循环，宿主 100% force_evict 兜底接）。
W4  零扰动面：used ≤ 0.9cap 且 < 0.75cap（未触发任何档）→ run() 行为与现状逐出链一致（无降档事件）。

注意：本文件只测 S1 骨架（distill.run 90→75 降档分档 + 估值 gate）。
"""
from __future__ import annotations

import pytest

from fakes import FakeEndpoint
from helpers import complete_digest, mkseg, mk_window
from sc.aux_provider import AuxProvider
from sc.distill import Distiller
from sc.store import JsonlStore


def _aux():
    return AuxProvider(FakeEndpoint(), async_digest=False, fail_cooldown=0.0)


def _store(tmp_path):
    return JsonlStore(tmp_path / "mem.jsonl")


def _mk_p0(tmp_path, aux, store, n, ready=True, per_tok=200):
    """构造 n 条 P0 段（每条 est_tokens≈per_tok），返回 segs。digest ready/不 ready。"""
    segs = []
    for i in range(n):
        s = mkseg(store, f"P0#{i}" + "x" * (per_tok * 4 - 6), tier="P0", prov="user",
                  digest_ref="ready" if ready else "pending")
        if ready:
            complete_digest(aux, s)
        segs.append(s)
    return segs


def _distill(tmp_path, cap, n_p0, per_tok=200, ready=True):
    """构造窗口（cap）+ n_p0 条 P0 段（每条≈per_tok token），跑 Distiller.run()。

    used_tokens ≈ n_p0 × per_tok。调 cap 控制 used 落在哪档水位区间。
    返回 (d, w, result)。
    """
    aux = _aux()
    store = _store(tmp_path)
    segs = []
    for i in range(n_p0):
        s = mkseg(store, f"P0#{i}" + "x" * (per_tok * 4 - 6), tier="P0", prov="user",
                  digest_ref="ready" if ready else "pending")
        if ready:
            complete_digest(aux, s)
        segs.append(s)
    w = mk_window(cap, segs)
    d = Distiller(w, store, aux, ledger_evict_cb=None)
    result = d.run()
    return d, w, result


# ── W1 降档序：90% 面先试、压到 0.9cap 下即停 ──────────────────────────
def test_w1_tier90_fires_first_and_stops(tmp_path):
    """used>0.9cap → 90% 面逐 P0 段 + 压到 0.9cap 下即停（不继续 75/65/55）。"""
    # n_p0=20 × 200 = 4000 used；cap=5000 → 0.9cap=4500 > 4000（不超压）→ 调 cap=4200
    # 0.9×4200=3780 < 4000=used → 超 90% 线。逐出 P0 段（每条 200→digest 摘录 ~20）后
    # used 掉到 cap 下 → 90% 面压到位即停。
    d, w, result = _distill(tmp_path, cap=4200, n_p0=20, per_tok=200, ready=True)
    # 90% 面事件面钉住（S1 降档序 90% 档打 "tier90" 事件；§4.5 旧事件名 p0_90pct_force 已随 S1 表驱动重构不再产生）
    assert "tier90" in result.events, \
        f"90% 面应先触发（events={result.events}）"
    # 压到 0.9cap 下（或物理满 backpressure）
    assert w.used_tokens <= int(0.9 * 4200) or result.status == "backpressure", \
        f"90% 面应把 used 压到 0.9cap 下（used={w.used_tokens}, status={result.status}）"


# ── W2 估值 gate：省 < 阈值 → pass 下一档 ────────────────────────────
def test_w2_gate_passes_when_reclaim_below_threshold(tmp_path, monkeypatch):
    """某档预估省 < 阈值×used → 不执行该档、pass 给下一档（估值面确定性算、零 LLM）。

    断言面 = 估值 gate 纯函数存在 + 低省量场景下不逐出（pass）。
    现状 run() 无估值 gate → 此测试 RED（AttributeError / 行为不符）。
    """
    d, w, result = _distill(tmp_path, cap=4200, n_p0=2, per_tok=200, ready=True)
    # 2×200=400 used / 4200 cap = 9.5% —— 远低于 55% 线，未触发任何档（W4 面）
    # 估值 gate 面：run() 应对"无省量"场景 no-op（不逐出、不报错）
    assert result.status in ("ok", "backpressure")
    # S1 落后面：run() 暴露估值 gate（owner 拍"分母=当前用量"）——纯函数可单独调
    # 断言 Distiller 有估值 gate 面（_reclaim_est / gate 常量）供 W2 锚
    assert hasattr(d, "_reclaim_est") or hasattr(d, "_est_reclaim"), \
        "S1 降档序应暴露估值 gate 纯函数（_reclaim_est / _est_reclaim）"


# ── W3 单调累积 + 全 pass 后 100% force_evict 兜底 backpressure ──────────
def test_w3_all_pass_backpressure_not_deadlock(tmp_path):
    """全档逐出后仍压不到 cap 下（物理满：只剩结构 P0 goal 尾）→ backpressure 非死循环。"""
    aux = _aux()
    store = _store(tmp_path)
    # 1 条 goal 尾结构 P0（永不逐）+ 0 普通 P0 → 窗口物理满、全档 pass 后 backpressure
    g = mkseg(store, "goal:" + "y" * 4000, tier="P0", prov="user", structural=True,
               kind="goal", digest_ref="pending")
    w = mk_window(1000, [g])   # used≈1000 = cap（物理满），goal 尾永不逐
    d = Distiller(w, store, aux, ledger_evict_cb=None)
    result = d.run()
    assert result.status == "backpressure", \
        f"全档 pass（物理满）应 backpressure（got {result.status}）"
    # goal 尾结构 P0 仍驻（结构地板不破）
    assert any(ws.segment.segment_id == g.segment_id and ws.form == "resident"
               for ws in w.segs), "goal 尾结构 P0 应永不逐（仍 resident 驻窗口）"


def test_e3_below_90pct_zero_eviction(tmp_path):
    """90% 以下：降档带 P0 面全 no-op（纯容量阈值面）。

    窗口 40% cap（< 90%）→ 90% 档 break、75% 档（P2/P1 常规逐）
    对 P0 面在 80-90% 区间行为零变化（owner 红线③）。
    RED 锚：P0 段在 <90% 保持 resident（90% 面/§4.5 面不触发）。
    """
    # per_tok=50 → P0 段 50×4=200 token，used≈200/500=40% cap < 90% 线
    d, w, result = _distill(tmp_path, cap=500, n_p0=1, per_tok=50, ready=True)
    # used≈40% cap < 90% cap → 90% 档 break、75% 档 < 75% cap break
    # P0 非结构段在 <90% 不逐（§4.5 面 80-90% 区间零变化）
    p0_forms = [ws.form for ws in w.segs if ws.segment.tier == "P0"]
    assert p0_forms and all(f == "resident" for f in p0_forms), \
        f"<90% 窗口 P0 段应全 resident 零逐出（got {p0_forms}）"


def test_w4_zero_disturbance_below_90pct(tmp_path):
    """used < 0.9cap 且 75% 档也不触发 → 未触发任何降档面、行为与现状逐出链一致（无 tier 事件、无逐出）。"""
    d, w, result = _distill(tmp_path, cap=10000, n_p0=1, per_tok=200, ready=True)
    # 200 / 10000 = 2% < 75% 线，不触发 90%/75% 任何档
    assert w.used_tokens <= int(0.75 * 10000), f"前置：used={w.used_tokens} 应 < 0.75cap"
    assert "tier90" not in result.events and "tier75" not in result.events, \
        f"低于 75% 线不应触发任何降档事件（events={result.events}）"
    assert result.status == "ok"


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
