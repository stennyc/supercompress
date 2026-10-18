"""v6 §4.5 90% 超压强制 P0 折存根面（RED→GREEN 回归面）。

背景：现状 distill.run L158 注释钉死"90% 水线不逐 P0/AC——P0 只走 ① push cap ② 100%
force_evict 两出口；本档 P0 豁免"。4bb2e8 session 实证：窗口 501k/512k（98% cap）卡死，
P0 段被 ③ 三道门（structural P0 永不逐 + digest 闸 + R3 pending 兜底 aux 连败≥3）卡死，
90% 水线因"豁免 P0"设计不折 P0 存根 → 压缩结果 > 原始 → 宿主 refuse 闸否决。

owner 拍板（2026-10-17）：加 90% 强制面——窗口 > 0.9×cap 时强制折 P0 存根
（aux 健康也触发，纯超压驱动；绕 digest 闸走 head80 占位；放开 user 指令 structural
P0，保留 goal/ledger 尾结构地板；80%-90% 区间行为零变化）。

锁定（TDD）：
F1  90% 超压（w.used > 0.9*cap）+ 非结构 P0（digest ready）→ 被逐出折存根（evict_page）
F2  90% 超压 + digest 未 ready 的 P0 → 走 head80 占位存根（digest_text 预填原文头 80 字符）
F3  90% 超压 + user 指令 structural P0（<100 字）→ 允许逐（放开 structural 门）
F4  90% 超压 + goal/ledger 尾 structural P0 → 仍永不逐（结构地板不破）
F5  89% 区间（w.used < 0.9*cap）→ 90% 强制面不触发（保持现有 ③ 行为，P0 走 ③/⑤）
F6  R3 pending 兜底零扰动（aux.endpoint_degraded=False 时 90% 面照常开、R3 兜底事件 p0_pending_escape 不 fire）
"""
from __future__ import annotations

from sc.distill import Distiller
from sc.aux_provider import AuxProvider
from fakes import FakeEndpoint
from helpers import mkseg, mk_window, complete_digest

CAP = 265           # 窗口 capacity_tokens（token）
N_P0 = 12           # P0 段条数
# 实测：每段 est_tokens ≈ 20，12 段 ≈ 240 token（used_tokens）
# 90% 超压线：0.9×265=238 < 240 → 超压；89% 线：0.9×270=243 > 240 → 区间内


def _aux() -> AuxProvider:
    return AuxProvider(FakeEndpoint(), async_digest=False, fail_cooldown=0.0)


def _store(tmp_path):
    from sc.store import JsonlStore
    return JsonlStore(tmp_path / "mem.jsonl")


def _seed(tmp_path, structural_user=False, digest_ready=True, goal_tail=False,
          n_p0=N_P0):
    """构造 store 段（n_p0 条 P0 + goal 尾），返回 (aux, store, segs, goal_seq)。"""
    aux = _aux()
    store = _store(tmp_path)
    segs = []
    for i in range(n_p0):
        # 原文约 200 字符（中文 1 字符 ≈ 1 token）→ 每段 est_tokens ≈ 200
        text = ("用户指令" if structural_user else "普通内容") + f"#{i}" + "x" * 180
        s = mkseg(store, text, tier="P0", prov="user",
                  structural=structural_user,
                  digest_ref="ready" if digest_ready else "pending")
        segs.append(s)
    if digest_ready:
        for s in segs:
            complete_digest(aux, s)
    goal_seq = None
    if goal_tail:
        g = mkseg(store, "goal: 保留回滚入口" + "y" * 180, tier="P0", prov="user",
                  structural=True, kind="goal",
                  digest_ref="ready" if digest_ready else "pending")
        if digest_ready:
            complete_digest(aux, g)
        goal_seq = g.seq
        segs.append(g)
    return aux, store, segs, goal_seq


def _run(tmp_path, structural_user=False, digest_ready=True, goal_tail=False,
         cap=CAP, n_p0=N_P0):
    """构造窗口 + 跑 Distiller.run()，返回 (distiller, window, result, evicted_seqs, goal_seq)。

    cap 传参控制 90% 线（0.9×cap）。n_p0 × ~200 token 为窗口 used_tokens 实际值——
    测试前置断言 used vs 0.9×cap 关系，cap 取对应值即可（12×200≈2400 → cap=2500 时
    0.9×2500=2250 < 2400 = 90% 超压；cap=3000 时 0.9×3000=2700 > 2400 = 89% 区间）。
    """
    aux, store, segs, goal_seq = _seed(tmp_path, structural_user, digest_ready, goal_tail, n_p0)
    window = mk_window(cap, segs)
    d = Distiller(window, store, aux, ledger_evict_cb=None)
    result = d.run()
    evicted = [ws.segment.seq for ws in window.segs
               if ws.form == "digest_only" and ws.segment.tier == "P0"]
    return d, window, result, evicted, goal_seq


def test_f1_90pct_nonstructural_p0_evicted(tmp_path):
    """F1：90% 超压 + 非结构 P0（digest ready）→ 被逐出（evicted 非空）。"""
    d, w, result, evicted, goal_seq = _run(tmp_path, cap=CAP,
                                           structural_user=False, digest_ready=True)
    assert w.used_tokens > int(0.9 * w.capacity_tokens), \
        f"前置：used {w.used_tokens} 必须 > 0.9×cap {int(0.9*w.capacity_tokens)}"
    assert len(evicted) > 0, "90% 超压时非结构 P0 应被逐出（evicted 非空）"


def test_f2_90pct_digest_not_ready_head80(tmp_path):
    """F2：90% 超压 + digest 未 ready 的 P0 → 走 head80 占位存根（digest_text 预填 ≥80 字符）。"""
    d, w, result, evicted, goal_seq = _run(tmp_path, cap=CAP,
                                           structural_user=False, digest_ready=False)
    assert w.used_tokens > int(0.9 * w.capacity_tokens)
    assert len(evicted) > 0, "90% 超压时 digest 未 ready 的 P0 也应被逐（head80 占位）"
    for seq in evicted:
        seg = next(ws.segment for ws in w.segs if ws.segment.seq == seq)
        assert seg.digest_text and len(seg.digest_text) >= 80, \
            f"digest 未 ready 的 P0（seq={seq}）应预填 head80，实际={seg.digest_text!r}"


def test_f3_90pct_user_instruction_p0_allowed(tmp_path):
    """F3：90% 超压 + user 指令 structural P0（<100 字）→ 允许逐（放开 structural 门）。"""
    d, w, result, evicted, goal_seq = _run(tmp_path, cap=CAP,
                                            structural_user=True, digest_ready=True)
    assert w.used_tokens > int(0.9 * w.capacity_tokens)
    assert len(evicted) > 0, "90% 超压时 user 指令 structural P0 应被放开（evicted 非空）"


def test_f4_90pct_goal_tail_never_evicted(tmp_path):
    """F4：90% 超压 + goal/ledger 尾 structural P0 → 仍永不逐（结构地板不破）。"""
    d, w, result, evicted, goal_seq = _run(tmp_path, cap=CAP,
                                           structural_user=True, digest_ready=True, goal_tail=True)
    assert goal_seq is not None, "goal 尾段应被加入窗口"
    assert goal_seq not in evicted, \
        f"goal/ledger 尾 P0（seq={goal_seq}）应永不逐，实际被逐（evicted={evicted}）"


def test_f5_below_90pct_no_force(tmp_path):
    """F5：89% 区间（used < 0.9×cap）→ 90% 强制面不触发（events 无 90% 面标志）。"""
    # cap=270 → 0.9×270=243 > used≈240 → 89% 区间
    d, w, result, evicted, goal_seq = _run(tmp_path, cap=270,
                                           structural_user=False, digest_ready=True)
    assert w.used_tokens < int(0.9 * w.capacity_tokens), \
        f"前置：used {w.used_tokens} 必须 < 0.9×cap"
    assert "p0_90pct_force" not in result.events, \
        f"89% 区间 90% 强制面不应触发，events={result.events}"


def test_f6_r3_pending_escape_untouched(tmp_path):
    """F6：R3 pending 兜底零扰动——aux 健康（endpoint_degraded=False）时 90% 面照常开、R3 兜底不 fire。"""
    d, w, result, evicted, goal_seq = _run(tmp_path, cap=CAP,
                                           structural_user=False, digest_ready=True)
    assert w.used_tokens > int(0.9 * w.capacity_tokens)
    assert len(evicted) > 0, "90% 面应在 aux 健康时照常逐 P0（不依赖 endpoint_degraded）"
    assert "p0_pending_escape" not in result.events, \
        f"aux 健康时 R3 pending 兜底不应 fire，events={result.events}"
