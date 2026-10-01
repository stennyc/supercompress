#!/usr/bin/env python3
"""P-0.2 决定性探针（scratch/probe_p02.py 转正式仓内件，P-INST 可移植性口径①：零硬编码路径）：
合成 HERMES_HOME 下跑真 engine，断言生产日志 DoD 签名。复跑即再证。

P-0.2 DoD（plan）：
- D1: `session <sid> start (fresh), store=.../supercompress_store/<sid>.jsonl`（HERMES_HOME 解析链）
- D2: compress 块 `forms: {...}` 统计（S-Phase 2 阶梯注入真跑）
- D3: P2 hard 逐出（旧 epoch P1 全文出窗口、当前 epoch 保留、字节可逆）
- D4: egress room-skeleton 过滤真生效
另：P2 FPR 静默面（同主题 0 事件）。

Run（repo 内任意 cwd，布局无关；3.11/3.14 双解释器均可——纯 engine 面）:
  python bench/p02_probe.py
输出 = 9 项断言结果（P-0.2 证据面）；全 PASS 退出码 0，任一 FAIL 退出码 1。
"""
import logging
import os
import shutil
import sys
import tempfile

# 布局无关解析（bench/ 直接子件 = 插件根；同 bench/rerank_probe.py 惯例）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 合成 HERMES_HOME（解析链验证：env 在 → 数据面落 $HERMES_HOME/supercompress_store/）
tmp_home = tempfile.mkdtemp(prefix="p02_probe_")
os.environ["HERMES_HOME"] = tmp_home

import engine as _eng_mod
from engine import SupercompressContextEngine
from sc.schema import Segment

CAPTURED = []


class Cap(logging.Handler):
    def emit(self, rec):
        CAPTURED.append(self.format(rec))


# 直接抓模块自身 logger 对象（engine.py 顶层 import 时 __name__="engine"）——挂 handler 到
# logger 对象本身，绕开名字解析（探针与生产包路径名不一致 → 旧版捕获 0 行的根因）。
log = _eng_mod.logger
cap = Cap()
cap.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(cap)
log.setLevel(logging.INFO)

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


# ── D1：fresh session 签名 + 解析链（HERMES_HOME → <home>/supercompress_store/<sid>.jsonl）──
e = SupercompressContextEngine(context_length=256000)   # 无 external_store_path → env 解析
expected = os.path.join(tmp_home, "supercompress_store", "probe_p02_s1.jsonl")
e.on_session_start("probe_p02_s1", hermes_home=tmp_home)
check("D1 store path 解析链", e.store.path == expected, f"({e.store.path})")
check("D1 fresh 签名", any("start (fresh)" in l and "probe_p02_s1" in l for l in CAPTURED))

# ── D3：P2 hard 逐出（engine._drift.current_epoch=1；旧 epoch P1 全文出窗口）──
goal = Segment(tier="P0", provenance="user", original_text="GOAL_PIN 顶层目标",
               structural_p0=True, goal="g", session_topic="0", seq=0)
old_p1 = Segment(tier="P1", provenance="model",
                 original_text="OLD_EPOCH 旧任务 推理内容 很长 FULLTEXT_MARKER_OLD",
                 session_topic="0", seq=1)
new_p1 = Segment(tier="P1", provenance="model",
                 original_text="NEW_EPOCH 当前 任务 部署 FULLTEXT_MARKER_NEW",
                 session_topic="1", seq=2)
e.store.append(goal)
e.store.append(old_p1)
e.store.append(new_p1)
e.store.flush()
e._drift.current_epoch = 1
out = e.compress([{"role": "user", "content": "继续 当前 任务 部署"}], force=True)
block = "\n".join(m["content"] for m in out if m["role"] == "system")
check("D2 forms 统计块", "forms:" in block and "staged P0/P1 segments" in block,
      f"({block.splitlines()[1] if len(block.splitlines()) > 1 else block[:60]})")
check("D3 hard 旧 epoch 全文逐出", "FULLTEXT_MARKER_OLD" not in block)
check("D3 hard 当前 epoch 保留", "FULLTEXT_MARKER_NEW" in block)
back = {s.segment_id: s.original_text for s in e.store.replay()}
check("D3 可逆（字节在盘）", "FULLTEXT_MARKER_OLD" in back[old_p1.segment_id])

# ── D4：egress room-skeleton 过滤（compress 面）──
skel = Segment(tier="P0", provenance="user",
               original_text='[Group chat: "room"] You are @devops, one participant in a group chat…',
               structural_p0=True, session_topic="0", seq=3)
e.store.append(skel)
e.store.flush()
out2 = e.compress([{"role": "user", "content": "继续 当前 任务 部署"}], force=True)
block2 = "\n".join(m["content"] for m in out2 if m["role"] == "system")
check("D4 egress 骨架被裁", "You are @devops" not in block2)
check("D4 egress 留痕", any("egress cut" in l for l in CAPTURED))

# ── P2 静默确认：本探针 user 内容同主题（无任务切换）→ drift_events 恒空（FPR 面）──
e2 = SupercompressContextEngine(context_length=256000)
e2.on_session_start("probe_p02_s2", hermes_home=tmp_home)
for i in range(5):
    e2.on_turn_complete([{"role": "user", "content": f"部署 k8s 集群 节点 扩缩容 {i}"}])
for i in range(5):
    e2.on_turn_complete([{"role": "user", "content": f"k8s 集群 节点 部署 副本 {i}"}])
check("P2 FPR 静默（同主题 0 事件）", e2._drift.drift_events == [] and e2._drift.current_epoch == 0)

shutil.rmtree(tmp_home, ignore_errors=True)
print()
if FAILS:
    print(f"P-0.2 探针 FAIL: {FAILS}")
    sys.exit(1)
print("P-0.2 探针全 PASS（D1/D2/D3/D4 + P2 FPR 静默）")
