"""t_9db0ec44 · 剩余间歇性 Segment 比较 no-op · 根因 = (B) 多副本/陈旧进程漂移（非 A 第二处裸比较点）。

判定与证据（本卡 recon，逐条钉死）：
  (A) 第二处裸 Segment 比较点 —— **排除**：/usr/local 活体 sc/ 全树每一处
      .sort()/sorted()/min()/max()/cmp_to_key/heapq 逐一复核，凡比较含 Segment 的集合
      （recall.search 的 scored 4-tuple、inject/telemetry 的 _rank_key、distill 的 candidate seq 序）
      全部已是 **scalar key**（第 4 位 = segment_id）。line 244 之外再无裸 Segment 比较位。
  (B) 多副本不一致 + 陈旧进程 —— **坐实**：
      * /usr/local/.../supercompress/sc/recall.py:244 = post-fix scalar key（活体源，已修）；
      * 38/63 installs env 冻结副本 recall.py:237 = pre-fix 裸 `scored.sort()`（与活体 byte-diff 不等）；
      * user-scope gateway（03:20 重启，晚于 02:21 磁盘落 fix）post-fix 后 **零** `recall degraded`；
      * orchestrator-scope tui_gateway（10-07 22:42 起，**早于** 02:21 落 fix）内存模块 = pre-fix
        → union-window ≥3 段三主键全 tie 触发 timsort 比第 4 位裸 Segment → TypeError →
        engine.py:1163-1164 fail-open 间歇 no-op（03:19-03:40，随窗口段组成翻转 = 间歇/非单调）。

本测试两面（同 red 触发面 = test_segment_key_order 的 ≥3 段全 tie）：
  * pytest 面（默认走活体 sc）→ GREEN：≥3 段三主键全 tie 时按 segment_id 定确定性全序（回归钉，
    封死「修复 key 不改相对次序」+ 确定性）。
  * __main__ 面（argv[1] = 某 sc/ 目录，可指 pre-fix 副本）→ 对该目录 recall.py 复跑同触发面：
    pre-fix 裸 sort 必抛 TypeError（RED，稳定复现 live 间歇 no-op）；post-fix 副本 → GREEN。

跑法（活体 venv 解释器；0 硬编码，路径由参数注入）::
    <venv>/python tests/test_residual_segment_copy.py            # 活体 sc → GREEN
    <venv>/python tests/test_residual_segment_copy.py \\
        /root/.hermes/installs/.../environments/<env>/workspace/plugins/context_engine/supercompress/sc
                                                                # pre-fix 副本 → RED（TypeError）

pytest 面::
    pytest tests/test_residual_segment_copy.py                  # 活体 GREEN
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_TRIGGER_SEGS = [  # ≥3 段：两个同文 P1 同 seq（三主键全等）+ 一个 P0 同 seq（test_segment_key_order 同款）
    ("port 8080 auth", "P1", 3),
    ("port 8080 zeta", "P1", 3),
    ("port 8080 auth extra", "P0", 3),
]


def _mk_union_window(WindowSeg, segs):
    """union-window 形状（独立 seq 计数，主 store + model 旁路并集；同 test_segment_key_order）。"""
    from types import SimpleNamespace
    return SimpleNamespace(capacity_tokens=1000, segs=[WindowSeg(s) for s in segs])


def _trigger(Recaller, Segment, WindowSeg):
    """≥3 段三主键全 tie 触发面（red/green 双面共用）。

    R3 配套（2026-10-10）：P0 段 provenance 由 "model" 改 "user"——R3 把 model P0 boost
    从 1.5 降到 1.0（不享用户级加权），P0 恒排前（断言 P0 boost ×1.5）语义只对 user P0
    成立。本测试钉的是 tie 全序确定性（P0 boost 保 user P0 恒排前），P0 段须 user 面。
    P1 段保持 model（P1 无 boost 分叉，R3 零影响，全序断言不变）。"""
    w = _mk_union_window(WindowSeg, [
        Segment(tier=t, provenance=("user" if t == "P0" else "model"),
                original_text=text, seq=seq)
        for text, t, seq in _TRIGGER_SEGS
    ])
    h1 = Recaller(None, w).search("port 8080", k=3)
    h2 = Recaller(None, w).search("port 8080", k=3)
    assert [h.seg.segment_id for h in h1] == [h.seg.segment_id for h in h2], "确定性破坏"
    assert len(h1) == 3
    assert h1[0].seg.tier == "P0"  # P0 boost ×1.5 → 恒排前（前 3 主键相对次序零变化）
    p1_ids = [h.seg.segment_id for h in h1 if h.seg.tier == "P1"]
    assert p1_ids == sorted(p1_ids), "P1 tie 全序不成立"
    return h1


def load_sc_classes(sc_dir):
    """从给定 sc/ 目录加载 (Recaller, Segment, WindowSeg)。

    sc_dir=None → 用当前可导的活体 `sc`。否则把该目录的 sc.* 子模块绑定为包 `sc`（pre/post 同布局，
    仅 recall.py 字节差异）后取类。__main__ pre-fix demo 用；pytest 面永不触 pre-fix 副本。
    """
    if sc_dir is None:
        from sc.recall import Recaller
        from sc.schema import Segment
        from sc.distill import WindowSeg
        return Recaller, Segment, WindowSeg

    sc_pkg = types.ModuleType("sc")
    sc_pkg.__path__ = [str(Path(sc_dir).resolve())]
    sys.modules["sc"] = sc_pkg
    # scdata 是插件根级模块（sc/*.py 顶部 `import scdata`）；pre/post 同布局
    root = Path(sc_dir).resolve().parent
    scdata_p = root / "scdata.py"
    if scdata_p.is_file() and "scdata" not in sys.modules:
        sd_spec = importlib.util.spec_from_file_location("scdata", str(scdata_p))
        if sd_spec and sd_spec.loader:
            sd = importlib.util.module_from_spec(sd_spec)
            sys.modules["scdata"] = sd
            sd_spec.loader.exec_module(sd)
    for sub in ("schema", "drift", "distill", "recall"):
        p = Path(sc_dir) / f"{sub}.py"
        spec = importlib.util.spec_from_file_location(f"sc.{sub}", str(p))
        if spec is None or spec.loader is None:
            raise SystemExit(f"sc/ copy missing {sub}.py: {p}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"sc.{sub}"] = mod
        setattr(sc_pkg, sub, mod)
        spec.loader.exec_module(mod)
    from sc.recall import Recaller
    from sc.schema import Segment
    from sc.distill import WindowSeg
    return Recaller, Segment, WindowSeg


# ── pytest 面（活体 sc）：≥3 段全 tie 触发面在活体 = GREEN 回归钉 ──────────────
def test_live_tree_green():
    """活体 sc（post-fix scalar key）：≥3 段三主键全 tie → 按 segment_id 确定性全序（不炸）。"""
    Recaller, Segment, WindowSeg = load_sc_classes(None)
    _trigger(Recaller, Segment, WindowSeg)


def test_two_segment_guard_no_crash():
    """guard（两段全等 timsort 短路）：两段同 tier+同 seq+同文，修复前后均不炸（封死误断言）。"""
    Recaller, Segment, WindowSeg = load_sc_classes(None)
    w = _mk_union_window(WindowSeg, [
        Segment(tier="P1", provenance="model", original_text="port 8080 auth", seq=3),
        Segment(tier="P1", provenance="model", original_text="port 8080 auth", seq=3),
    ])
    assert len(Recaller(None, w).search("port 8080", k=2)) == 2


# ── __main__ 面（可指 pre-fix 副本 → RED 复现）────────────────────────────────
def _main():
    sc_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    label = str(sc_dir) if sc_dir else "(live sc)"
    print(f"[residual-segment] loading sc.recall from: {label}")
    Recaller, Segment, WindowSeg = load_sc_classes(sc_dir)
    try:
        _trigger(Recaller, Segment, WindowSeg)
        print("[residual-segment] RESULT: GREEN — deterministic total order (scalar key / post-fix)")
        return 0
    except TypeError as e:
        import traceback
        print(f"[residual-segment] RESULT: RED — TypeError reproduced: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(_main())
