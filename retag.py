"""D 步：存量 active.jsonl 按新 user 分级规则重打分（一次性迁移，plugin 内实现，零核心触碰）。

背景（@reviewer 裁定 2026-09-29）：A1 只改新写入判定；存量行仍是历史 tag
（100% structural），不改 tag 则 push cap 依旧 0 可逐出、compress 照样膨胀被拒。
本脚本对存量行按 sc.tagger.classify_user（单一实现源，与 A1 同源）重打
tier/structural_p0，使 A1 上线即生效。

安全契约（@data-engineer 数据面纪律）：
- 只改 tier / structural_p0 两个字段；original_text / content_hash / seq /
  其余行字节字段全部原样保留（replay 逐行验 content_hash，改原文 = 哈希失配
  = 行被 torn-tail 规则整体丢弃，绝对禁止）；
- 只重打 provenance=="user" 行（scdata 契约：replay NEVER re-runs tiering 指
  model LLM 评分；user 行 = 确定性规则，可安全重算）；
- 幂等：规则纯函数（原文 → (tier, structural)），重放不改变行字节（第二次跑
  零改动）；
- 原子写：先写 .tmp 再 rename；写前自动备份 .bak_<ts>；
- 行数守恒：输出行数 == 输入行数（只改字段，不增删行）。

CLI: python3 retag.py [active.jsonl 路径]   （缺省 ~/.hermes/supercompress_store/active.jsonl）
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sc.tagger import classify_user  # 单一实现源（与 A1 共用，不另造规则）

DEFAULT_PATH = os.path.expanduser("~/.hermes/supercompress_store/active.jsonl")


def retag_file(path: str, backup: bool = True) -> dict:
    """重打分单文件。返回报告 dict（lines / user_rows / changed / before / after）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln for ln in f.read().splitlines()]

    before = {"structural": 0, "non_structural": 0, "P0": 0, "P1": 0, "P2": 0}
    after = dict(before)
    out_lines: list[str] = []
    changed = 0
    user_rows = 0

    for raw in raw_lines:
        s = raw.strip()
        if not s:
            continue
        d = json.loads(s)
        for k in ("structural", "non_structural"):
            before[k] += 1 if (k == "structural") == bool(d.get("structural_p0")) else 0
        for k in ("P0", "P1", "P2"):
            before[k] += d.get("tier") == k
        if d.get("provenance") == "user":
            user_rows += 1
            tier, structural = classify_user(str(d.get("original_text", "")))
            if (d.get("tier"), bool(d.get("structural_p0"))) != (tier, structural):
                changed += 1
            d["tier"] = tier
            d["structural_p0"] = structural
        for k in ("structural", "non_structural"):
            after[k] += 1 if (k == "structural") == bool(d.get("structural_p0")) else 0
        for k in ("P0", "P1", "P2"):
            after[k] += d.get("tier") == k
        out_lines.append(json.dumps(d, ensure_ascii=False))

    n_lines = len(out_lines)
    if backup:
        bak = f"{path}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
        with open(bak, "w", encoding="utf-8") as f:
            f.write("\n".join(raw_lines) + "\n")

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # 原子：rename 同卷原子替换

    return {
        "lines": n_lines,
        "user_rows": user_rows,
        "changed": changed,
        "before": before,
        "after": after,
        "structural_pct_before": round(100.0 * before["structural"] / n_lines) if n_lines else 0,
        "structural_pct_after": round(100.0 * after["structural"] / n_lines) if n_lines else 0,
        "evictable_p0_before": before["P0"] - before["structural"],
        "evictable_p0_after": after["P0"] - after["structural"],
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    rep = retag_file(target)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"已写入 {target}（自动备份 .bak_*. jsonl 行数守恒 {rep['lines']}）")
