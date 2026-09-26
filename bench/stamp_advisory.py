"""③ stamp advisory — 非 pytest 单点查的独立模块（@devops nit 拆分，无 fakes/sc 依赖）。

背景：`check_stamp_advisory` 原先活在 `tests/test_preflight_ops.py`（该文件顶层为 gate⑤ 而
`from fakes import FakeEndpoint`），脱离 pytest 直接 import 会 ModuleNotFoundError。
本模块只含 ③ 的三态分类 + 仓内 stamp 读取，纯 stdlib，两条路径都可用：

- pytest 内：`from bench.stamp_advisory import _stamp_state, check_stamp_advisory`
  （root conftest 已把 repo root 放进 sys.path）
- 非 pytest 单点查（repo root 下）：
    .venv\\Scripts\\python.exe -c "from bench.stamp_advisory import check_stamp_advisory; check_stamp_advisory()"

三态（与 640399b 起 ③ 的契约 / f627935 文档同形）：
  PASS            双锚（head + model）全匹配
  WARN            head 或 model 任一失配（75% 数锚在旧 head/旧 model，需 --run 重推）
  WARN-no-record  文件缺失或坏 JSON（绝不当静默通过——vacuous pass 同款病）
纯 advisory：只打印、永不 raise、永不 block build。
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# repo root = 本模块的上上级（bench/stamp_advisory.py → repo root）
ROOT = Path(__file__).resolve().parent.parent
STAMP_PATH = ROOT / "bench" / ".last_75_stamp"


def _stamp_state(stamp_text, *, current_head, current_model):
    """纯分类：给定 stamp 原始文本 + 当前 head/model，返回三态之一。"""
    if stamp_text is None:
        return "WARN-no-record"
    try:
        rec = json.loads(stamp_text)
        assert set(("head", "model")) <= set(rec), "stamp 缺 head/model 双锚字段"
    except (ValueError, AssertionError):
        return "WARN-no-record"
    if rec["head"] == current_head and rec["model"] == current_model:
        return "PASS"
    return "WARN"


def check_stamp_advisory(stamp_path=None, *, live_env=True):
    """读仓内 stamp，取当前 head/model，打印三态 + 失配时打重跑命令。只读、永不 raise。"""
    p = Path(stamp_path) if stamp_path else STAMP_PATH
    text = p.read_text(encoding="utf-8") if p.exists() else None
    if live_env:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        model = os.environ.get("LIVE_ENDPOINT_MODEL") or os.environ.get("OPENAI_MODEL", "")
    else:
        head, model = "", ""
    state = _stamp_state(text, current_head=head, current_model=model)
    if state == "PASS":
        print(f"[preflight-③] PASS stamp 双锚匹配 head={head[:7]} model={model!r} → 75% 数可用")
    elif state == "WARN-no-record":
        print(f"[preflight-③] WARN 无记录: {p.name} 缺失/不可解析, 引用 75% 前先跑探针")
    else:  # WARN（head 或 model 任一失配）
        print(f"[preflight-③] WARN stamp 双锚失配 (head/model 至少一个 ≠ 当前), 75% 数可能过期。重跑: "
              f"LIVE_ENDPOINT_MODEL={model or '<新模型名>'} && .venv/Scripts/python.exe bench/decision_layer_probe.py --run 8")
    return state
