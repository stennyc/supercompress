"""gate⑤ preflight（ops 半边）— @devops own。

@coder ①-slice commit（71a3692）上开跑，②③④ 落仓后对最终 commit（1f7ee69）重跑。
断言面 @devops own，被测路径 = @coder 实现 + @data-engineer 权威数据面（scdata 单一实现）：

- 物化字节稳定（归档→活跃逐字节、归档不可变）+ 非空前置
- 单一 live 实现（verify_anchor_closure 全树恰好 1 个定义；store 只委托）
- 断电 3 注入点（mid_p1_batch / post_p0_fsync / gate_boundary，经
  distill(inject_fault_after=...) + SC_INJECT_FAULT marker，marker on 才真生效）
- 非平凡前置三条：混合态（fsynced_p0>0 且 pending>0）/ 闭合输入非空 / marker 双向
- 仓内 scdata 8/8 = tests/test_scdata_ref.py（同会话跑，本模块不重实现）

跑法（repo 根，用 repo 自带 .venv）：
  .venv\\Scripts\\python.exe -m pytest tests/ -v
KPI #3 决策层 75% 在 @researcher 车道、需活主模型端点，不在本 preflight 范围。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import scdata
from fakes import FakeEndpoint
from helpers import mk_window, mkseg
from sc.aux_provider import AuxProvider, marker_enabled
from sc.distill import Anchor, Distiller
from sc.store import JsonlStore

ROOT = Path(__file__).resolve().parents[1]


# ── 场景：P0×2（结构 + 非结构）+ P1 + P2，顶穿 80% 水位 → P0 被逐出 ──
# cap=500 → water=400；P0 合计 460 tokens > water，保证非结构 P0 被逐（evicted 真成立）。
def build_scenario(tmp_path, aux):
    store = JsonlStore(str(tmp_path / "s.jsonl"))
    segs = {
        "p0_goal": mkseg(store, "用户目标: supercompress 面向全球用户, 验收 = gate v4 全过" + "x" * 900,
                         tier="P0", prov="user", structural=True),
        "p0_decision": mkseg(store, "决策: jsonl 单一事实源, 不引 DB, tier-aware fsync" + "y" * 900,
                             tier="P0", prov="model"),
        "p1": mkseg(store, "分析: 模块 A 依赖 B, 改 X 必须同时改 Y, 测试 3/7 过" + "z" * 560,
                    tier="P1"),
        "p2": mkseg(store, "stdout 全量: " + "o" * 380, tier="P2"),
    }
    window = mk_window(500, list(segs.values()))
    return store, window, segs


def _distill(store, window, aux, inject):
    return Distiller(window, store, aux, inject_fault_after=inject).run()


# ── 断电 3 注入点（marker on = 真生效）──

def test_power_loss_mid_p1_batch(tmp_path, aux, monkeypatch):
    monkeypatch.setenv("SC_INJECT_FAULT", "1")
    store, window, segs = build_scenario(tmp_path, aux)
    res = _distill(store, window, aux, "mid_p1_batch")
    fs = res.fault_state
    assert fs is not None and "fault:mid_p1_batch" in res.events
    # 前置①：混合态真建成（P0 已 fsync + P1/P2 仍 pending）
    assert fs["fsynced_p0"] > 0, "混合态未建成: P0 未过 fsync gate"
    assert fs["pending_p1p2_bytes"] > 0, "混合态未建成: P1/P2 未处 pending"
    # 前置②：evicted 态真成立（非平凡前置，防假绿）
    evicted = [ws for ws in window.segs
               if ws.segment.segment_id == segs["p0_decision"].segment_id]
    assert evicted and evicted[0].form == "digest_only", "P0 未真正逐出 → 臂无效"
    # 「断电」= 此刻进程死，从磁盘重建（单一 replay 路径）
    idx = scdata.replay(store.path)
    # 前置③：闭合输入非空
    assert len(idx) > 0, "replay 索引为空 → 闭合检查没跑"
    # P0（含结构 goal）100% 在
    for k in ("p0_goal", "p0_decision"):
        assert segs[k].content_hash in idx.by_hash, f"P0 {k} 断电丢失"
    # P1/P2 缺失有界：缺失量 = 一个 lazy 批 ≤ cap（size trigger 字节上界）
    assert fs["pending_p1p2_bytes"] <= store.lazy_batch_cap
    assert segs["p1"].content_hash not in idx.by_hash
    assert segs["p2"].content_hash not in idx.by_hash
    # torn 尾被 hash/JSON 验出并静默丢
    assert "json_parse_fail" in [r for _, r in idx.dropped]
    # 零悬空锚点：前置 — manifest 非空（evicted P0 已 durable → 有锚点）
    assert len(res.anchor_manifest) > 0, "manifest 为空 → 闭合断言 vacuous"
    assert store.dangling_anchors(res.anchor_manifest) == []


def test_power_loss_post_p0_fsync(tmp_path, aux, monkeypatch):
    """kill 落在「P0 gate 已过、P1/P2 lazy 批未 flush、无 torn 写」。"""
    monkeypatch.setenv("SC_INJECT_FAULT", "1")
    store, window, segs = build_scenario(tmp_path, aux)
    res = _distill(store, window, aux, "post_p0_fsync")
    fs = res.fault_state
    assert fs is not None and fs["pending_p1p2_bytes"] > 0
    idx = scdata.replay(store.path)
    assert len(idx) > 0
    for k in ("p0_goal", "p0_decision"):
        assert segs[k].content_hash in idx.by_hash
    # 无 torn 写：文件干净（P1/P2 压根没落盘，不是坏行）
    assert "json_parse_fail" not in [r for _, r in idx.dropped]
    assert fs["pending_p1p2_bytes"] <= store.lazy_batch_cap
    assert len(res.anchor_manifest) > 0
    assert store.dangling_anchors(res.anchor_manifest) == []


def test_power_loss_gate_boundary(tmp_path, aux, monkeypatch):
    """gate 边界：锚点已发射 + P1/P2 批已 flush durable → 断电零缺失。"""
    monkeypatch.setenv("SC_INJECT_FAULT", "1")
    store, window, segs = build_scenario(tmp_path, aux)
    res = _distill(store, window, aux, "gate_boundary")
    fs = res.fault_state
    assert fs is not None
    # 该注入点语义 = flush 已完成（边界两侧验证：P1/P2 durable）
    assert fs["pending_p1p2_bytes"] == 0
    idx = scdata.replay(store.path)
    assert len(idx) > 0
    for k in ("p0_goal", "p0_decision", "p1", "p2"):
        assert segs[k].content_hash in idx.by_hash, f"{k} 在 gate 边界后丢失"
    assert len(res.anchor_manifest) > 0
    assert store.dangling_anchors(res.anchor_manifest) == []


# ── marker 双向（生产态恒 inert + on 态真生效）──

def test_inject_fault_hook_noop_in_prod(tmp_path, aux, monkeypatch):
    monkeypatch.delenv("SC_INJECT_FAULT", raising=False)
    store, window, segs = build_scenario(tmp_path, aux)
    assert not marker_enabled()
    res = _distill(store, window, aux, "mid_p1_batch")   # 参数传了，但 marker off → 被忽略
    assert not [e for e in res.events if e.startswith("fault:")], "prod 态钩子泄漏"
    assert res.fault_state is None
    idx = scdata.replay(store.path)
    assert "json_parse_fail" not in [r for _, r in idx.dropped], "prod 态写出了 torn 片段"
    # prod 正常收尾：step-7 flush → P1/P2 durable，零丢失
    assert segs["p1"].content_hash in idx.by_hash and segs["p2"].content_hash in idx.by_hash
    # 同进程翻回 on → 真生效（双向可控证明）
    monkeypatch.setenv("SC_INJECT_FAULT", "1")
    store2, window2, _ = build_scenario(tmp_path, aux)
    res2 = _distill(store2, window2, aux, "mid_p1_batch")
    assert "fault:mid_p1_batch" in res2.events and res2.fault_state is not None


# ── 物化字节稳定 + 归档不可变（跨期重开的部署层规则）──

def test_materialization_byte_stable(tmp_path):
    arch = str(tmp_path / "arch.jsonl")
    store = JsonlStore(arch)
    mkseg(store, "goal: v1 上线, 关掉重开记忆还在" + "a" * 400, tier="P0", prov="user", structural=True)
    mkseg(store, "P1: 中间推理" + "b" * 200, tier="P1")
    store.flush()
    before = open(arch, "rb").read()
    lines_before = before.decode("utf-8").splitlines()
    assert len(lines_before) == 2, "前置: 物化输入非空（空文件 → vacuous pass；写了 2 段 = 2 行）"
    active = str(tmp_path / "active.jsonl")
    JsonlStore.materialize(arch, active)          # 单 replay 路径（scdata.materialize 权威）
    lines_after = open(active, "rb").read().decode("utf-8").splitlines()
    assert lines_after == lines_before, "物化后行字节漂移 → content_hash 去重失效、双写风险"
    assert open(arch, "rb").read() == before, "归档字节被改（必须只读）"
    idx = scdata.replay(active)
    assert len(idx) == 2
    assert not [r for _, r in idx.dropped], "重开重放出现坏行"


# ── 单一 live 实现（gate⑤ 4 号前置，机械自检不靠人工）──

def test_single_live_verify_anchor_closure(tmp_path):
    # 「恰好 1 个」限定在**生产 live 路径**（repo 根 + sc/ + scdata.py）；
    # tests/ 引用/调用该实现合法，危险的 divergence 源是第二套*生产*定义。
    prod_roots = [ROOT / "scdata.py"] + list((ROOT / "sc").glob("*.py"))
    defs = [str(p.relative_to(ROOT)) for p in prod_roots
            if p.exists() and "def verify_anchor_closure" in p.read_text(encoding="utf-8")]
    assert defs == ["scdata.py"], f"生产 live 实现必须恰好 1 个, found: {defs}"
    store_src = (ROOT / "sc" / "store.py").read_text(encoding="utf-8")
    assert "scdata.verify_anchor_closure" in store_src, "store 必须委托 scdata 权威实现"
    # 行为一致性：幽灵锚点在两个入口同判（单一不变量、两套入口、一个实现）
    store = JsonlStore(str(tmp_path / "s.jsonl"))
    mkseg(store, "goal " + "x" * 40, tier="P0", prov="user", structural=True)
    phantom = "phantom-hash-not-on-disk"
    assert scdata.verify_anchor_closure([phantom], scdata.replay(store.path)) == [phantom]
    assert store.dangling_anchors([Anchor("ph", phantom, "P0", "none")]) == [phantom]


def test_commit_anchor_recorded():
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and len(out.stdout.strip()) == 40, "repo HEAD 缺失"
    print(f"[preflight] gate⑤ ops 半跑于 commit {out.stdout.strip()}")
