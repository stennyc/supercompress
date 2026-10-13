"""观测 audit 悬挂侧（fix-5 t_d888d710 · gate §3 双路径观测面，纯 read-only 消费既有数据）。

audit 痕迹 = append-only jsonl 侧车，与 compact.json / digests.json 同目录、同 sid 前缀
（GC 联动按同目录同前缀删，语义自洽）。本模块**只追加、不读、不重算、不迁移**——
消费 `inject.observe_replay_window` 算好的 record dict，落一行。

写原语（security-engineer 裁定）：
- **O_APPEND 单行 + 0600 权限 + flush**，**非 `os.replace`**。audit 是 append-only 痕迹文件，
  O_APPEND 单行写入即可；`os.replace`（换 inode）会破坏 append-only 语义——并发读 / 后续
  append 期望的是同一 inode 的轨迹，换 inode 后痕迹断档。
失败语义（同 `compact_ledger.py:141-148` 契约）：写 OSError（ENOSPC/EIO/permission/
is-a-directory）→ 只打 `WARNING` + **fail-soft 继续**（观测面挂了不影响主流程/宿主），
**零 raise**。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_AUDIT_PERM = 0o600


def audit_path_for_jsonl(jsonl_path: str) -> str:
    """audit 侧车路径 = 主 store 路径 splitext + '.audit.jsonl'（单源，零硬编码）。

    `<sid>.jsonl` → `<sid>.audit.jsonl`；legacy `active.jsonl` → `active.audit.jsonl`。
    与 compact.json / digests.json 同目录、同 sid 前缀——GC 联动语义自洽。
    """
    return os.path.splitext(jsonl_path)[0] + ".audit.jsonl"


def write_audit_record(path: str, record: dict,
                       log: Optional[logging.Logger] = None) -> bool:
    """append 一行 audit record（O_APPEND 单行 + 0600 + flush；**非 os.replace**）。

    - O_WRONLY|O_APPEND|O_CREAT，初始权限 0o600；
    - `fchmod(fd, 0o600)` 强制收紧：即使文件已存在且权限偏宽（他人可读）也改回 0600；
    - `flush` + `fsync` 落盘；
    - 成功 → True；写 OSError（含 is-a-directory / EIO / ENOSPC）→ `WARNING` +
      fail-soft 返回 False（**零 raise**，观测面故障绝不炸主流程/宿主，同
      compact_ledger._persist 失败契约）。
    """
    lg = log if log is not None else logger
    try:
        data = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _AUDIT_PERM)
        with os.fdopen(fd, "ab", closefd=True) as f:
            f.write(data)                 # O_APPEND：内核保证定位到 EOF，单行原子追加
            f.flush()
            try:
                os.fchmod(f.fileno(), _AUDIT_PERM)   # 收紧到 0600（best-effort）
            except (AttributeError, OSError):
                pass                       # 平台不支持 fchmod / 只读 → 静默（best-effort，不阻断落盘）
            os.fsync(f.fileno())
        return True
    except OSError as e:
        lg.warning(
            "supercompress: audit write failed (%s) -> observation-only skip, fail-soft zero-raise", e)
        return False
