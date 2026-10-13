"""C.1a · compact-ledger forms 字段观测面扩展 —— 行为验收。

窄缝门控判据：相邻 c# 的 form 集相同但 text 变 → 统计发生率（7 天观测反推）。
本层验收四件事（全部读活体 CompactLedger，零依赖 engine 壳）：
  1. forms 非空 → 新条目落 forms（排序 (form,count) 列表）
  2. forms=None/空 → 条目不含 forms 键（零体积膨胀）
  3. 旧侧车（无 forms 键）读回 → entry.get("forms") 容忍缺失，零 KeyError
  4. 零 mutation：加 forms 不改既有 seq/text/tokens/generation/seals 值
"""
from __future__ import annotations

import json
import os

from sc.compact_ledger import CompactLedger


def test_forms_nonempty_persisted_sorted(tmp_path):
    ld = CompactLedger(os.path.join(tmp_path, "x.compact.json"), logger=None)
    e = ld.append("[supercompress] staged A", forms={"L3": 2, "L1": 1, "L0": 3})
    assert e is not None
    # forms 落盘 = 排序后的 (form,count) 列表（确定性读面）
    assert e["forms"] == [("L0", 3), ("L1", 1), ("L3", 2)]
    # 既有键原样
    assert e["seq"] == 1 and "text" in e and "tokens" in e


def test_forms_none_or_empty_omits_key(tmp_path):
    ld = CompactLedger(os.path.join(tmp_path, "y.compact.json"), logger=None)
    e1 = ld.append("block-1", forms=None)
    e2 = ld.append("block-2", forms={})
    # forms=None / 空 dict → 条目**不含** forms 键（零体积膨胀，安全约束）
    assert e1 is not None and "forms" not in e1
    assert e2 is not None and "forms" not in e2


def test_read_old_sidecar_without_forms_key(tmp_path):
    """旧侧车（无 forms 字段）读回不抛异常 + get('forms') 容忍缺失（→None）。"""
    path = os.path.join(tmp_path, "old.compact.json")
    # 手造一个"旧版"侧车：entries 无 forms 键
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "generation": 3, "seals": 1, "seq": 2,
                   "entries": [{"seq": 1, "text": "old", "tokens": 5},
                               {"seq": 2, "text": "old2", "tokens": 7}]}, f)
    ld = CompactLedger(path, logger=None)
    assert len(ld.entries) == 2
    # 读面容忍缺失：旧条目 get("forms") → None（不 KeyError）
    assert ld.entries[0].get("forms") is None
    # 续写仍能 append forms（新旧混存）
    e3 = ld.append("new-block", forms={"L2": 4})
    assert e3 is not None and e3["forms"] == [("L2", 4)]
    assert ld.entries[0].get("forms") is None  # 旧条仍缺，未被回填


def test_zero_mutation_of_existing_entry_values(tmp_path):
    """加 forms 不改动既有 seq/text/tokens/generation/seals 的值（零 mutation）。"""
    path = os.path.join(tmp_path, "z.compact.json")
    ld = CompactLedger(path, logger=None)
    e1 = ld.append("first", forms={"L1": 1})
    assert e1 is not None
    snap = {"seq": e1["seq"], "text": e1["text"], "tokens": e1["tokens"],
            "generation": ld.generation, "seals": ld.seals}
    # 再 append 一块（触发新 c#2）→ c#1 的既有值必须原样（append-only）
    e2 = ld.append("second", forms={"L3": 1})
    assert e2 is not None and e2["seq"] == e1["seq"] + 1
    # 重载侧车读回 c#1：seq/text/tokens/generation/seals 与快照一致（字节稳定）
    ld2 = CompactLedger(path, logger=None)
    c1 = next(e for e in ld2.entries if e["seq"] == snap["seq"])
    assert c1["seq"] == snap["seq"] and c1["text"] == snap["text"] and c1["tokens"] == snap["tokens"]
    assert ld2.generation == snap["generation"] and ld2.seals == snap["seals"]
