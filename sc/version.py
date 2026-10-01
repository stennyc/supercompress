"""版本戳 + 脱钩自检（报告#1 #1 代码侧）：一眼判断"生产跑的是哪一代"。

背景（审计轮报告#1 Top-1）：生产 gateway 跑的是 installs 里 env 的**代冻结快照**
（install/env 重建 = 冻结一份 supercompress 副本，与 repo 天然脱钩），repo 上的新
commit 没有任何进程在跑。版本戳把"查错树"的坑（查 repo pycache 却漏了活进程 map
的是 env 副本）彻底堵死：

- **build_id（稳定可复现）**：目录含可用 `.git` → `git rev-parse --short HEAD`
  （working tree 有未提交改动含 untracked → 带 `-dirty` 后缀，"脏"本身就是代际
  信息）；无 `.git`（env 冻结副本）/ git 命令不可用 → 内容 hash（本目录全部普通
  文件 路径+字节 → sha256 前 12 位，`c-` 前缀，确定性可复现：同一份字节在任何
  位置算出同一 build_id，跳过 .git/__pycache__ 等产物目录——pyc 随解释器变、
  .git 随仓库变，都不该进代际号）。
- **脱钩自检（只警告不阻断）**：``repo_root()`` 可推导 repo 源插件目录时（① env
  override ``SC_REPO_ROOT`` 显式指认；② 本插件目录自身是 git checkout → 即 repo；
  ③ 上溯「含 .git 且 hermes-agent 仓根层级对」的祖先），对比 **内容 hash**
  （git 版号与内容 hash 跨不可比，内容 hash 可比）：不一致打一行
  ``version desync: running=<build> repo=<build>`` WARNING。宿主 0 改动原则下只加
  日志、绝不 raise / 不阻断启动；repo 不可推导时静默跳过。

本模块零 LLM、零宿主 import、路径 0 硬编码（全走 ``sc/_bootstrap`` 同源语义：
__file__ 派生 + env override）。engine.py 在 shell v2 initialized / on_session_start
日志行带上 ``build=<id>``；运维工具 ``sc/sync_to_envs.py`` 复用 build_id 逐目录对
代际。
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

_PLUGIN_DIR = Path(__file__).resolve().parent.parent  # …/context_engine/supercompress
# 产物/噪声目录：字节不进内容 hash（.git 内容因仓库而变、pyc 随解释器变，都会让
# "同一份交付代码"在不同位置算出不同代际号）
_EXCLUDED_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"})
# 内容 hash 缓存：{abs_root: (stat 签名, content_id)}。engine 每 session start 构造一次
# build_id/desync，生产里反复重 hash 全插件（~1MB 级）没必要——签名（各文件 mtime+size）
# 不变即命中缓存；任一文件动过 → 签名变 → 重算。键含真实路径，不跨副本串味。
_CONTENT_ID_CACHE: dict = {}


def _stat_signature(root: Path) -> Optional[tuple]:
    """目录 stat 签名（路径 + 每文件 (mtime, size) 排序元组）；读不全 → None（不缓存）。"""
    parts = []
    try:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if not p.is_file() or any(part in _EXCLUDED_DIRS for part in rel.parts):
                continue
            st = p.stat()
            parts.append((rel.as_posix(), st.st_mtime, st.st_size))
    except OSError:
        return None
    return tuple(parts)


def content_id(root: Path, _use_cache: bool = True) -> str:
    """目录内容 hash（确定性可复现）：全部普通文件按相对路径排序，sha256 逐文件喂
    (相对路径 NUL 字节 内容)，取前 12 位 hex，``c-`` 前缀。同一份文件集在任何目录
    位置得到同一 content_id——这是跨「git repo / env 无 git 副本」唯一可比对的代际量。

    优化：以 (各文件 mtime,size) 签名为键缓存结果，签名不变即直接返回（engine 热路径
    反复调用不必重 hash）；签名变 → 重算。纯本地、零网络、零 LLM。"""
    key = str(root)
    sig = _stat_signature(root)
    if sig is not None and _use_cache:
        hit = _CONTENT_ID_CACHE.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or any(part in _EXCLUDED_DIRS for part in rel.parts):
            continue
        h.update(rel.as_posix().encode("utf-8") + b"\0")
        with open(p, "rb") as f:
            h.update(f.read())
    cid = "c-" + h.hexdigest()[:12]
    if sig is not None and _use_cache:
        _CONTENT_ID_CACHE[key] = (sig, cid)
    return cid


def build_id(root: Optional[Path] = None) -> str:
    """某 supercompress 目录的稳定 build_id（缺省 = 本插件目录）。

    有 ``.git`` 且 rev-parse 可用 → git 短 sha（dirty 加后缀）；否则（env 冻结副本 /
    git 不可用/失败）→ 内容 hash。任何情况下纯本地、零网络、零 LLM、绝不 raise。
    """
    root = Path(root) if root is not None else _PLUGIN_DIR
    if (root / ".git").exists():
        try:
            r = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                capture_output=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                bid = r.stdout.decode().strip()
                d = subprocess.run(
                    ["git", "-C", str(root), "status", "--porcelain"],
                    capture_output=True, timeout=10)
                if d.returncode == 0 and d.stdout.strip():
                    bid += "-dirty"
                return bid
        except Exception:
            pass  # git 沙箱/只读失败 → 内容 hash 降级（可读，不崩）
    return content_id(root)


def repo_root(override: Optional[str] = None) -> Optional[str]:
    """repo 源插件目录（脱钩自检对比面）：

    1. 显式 override / env ``SC_REPO_ROOT``（env 副本上溯找不到 .git，只能外部指认
       repo 的 supercompress 目录；须含 ``sc/_bootstrap.py``，否则视为无效 → None）；
    2. 本插件目录自身即 git checkout（含 .git）→ 自身就是 repo 源（自比，恒一致）；
    3. 上溯祖先：最近一个「含 .git 且
       ``<dir>/plugins/context_engine/supercompress/sc/_bootstrap.py`` 存在」的
       ``<dir>/plugins/context_engine/supercompress``（hermes-agent 仓根布局；与
       _bootstrap.require_install_root 同源层级语义）。

    找不到 → None（自检静默跳过，绝不 raise）。
    """
    o = (override or os.environ.get("SC_REPO_ROOT", "")).strip()
    if o:
        d = Path(o).expanduser()
        return str(d) if (d / "sc" / "_bootstrap.py").is_file() else None
    if (_PLUGIN_DIR / ".git").exists():
        return str(_PLUGIN_DIR)
    for d in _PLUGIN_DIR.parents:
        if (d / ".git").exists() and (
                d / "plugins" / "context_engine" / "supercompress" / "sc" / "_bootstrap.py").is_file():
            return str(d / "plugins" / "context_engine" / "supercompress")
    return None


def repo_build_id(override: Optional[str] = None) -> Optional[str]:
    """repo 源插件目录的 build_id；repo_root 不可推导 → None。"""
    rr = repo_root(override)
    return build_id(Path(rr)) if rr else None


def desync_warning(running_dir: Optional[Path] = None, log: Optional[logging.Logger] = None,
                   repo_override: Optional[str] = None) -> Optional[str]:
    """脱钩自检核心（纯函数面，可单测）：

    - repo 不可推导（无 .git、无 SC_REPO_ROOT）→ None（静默跳过，不阻断）；
    - running vs repo **内容 hash** 一致 → None；
    - 不一致 → 一行 ``version desync: running=<build> repo=<build>``（build 用各自
      build_id 便于人读：repo 侧通常是 git 短 sha，env 侧是 c- 内容 hash）；log 非
      None 时落 WARNING（只警告不阻断；日志面自身失败吞掉，不影响启动）。返回文案。
    """
    running_dir = Path(running_dir) if running_dir is not None else _PLUGIN_DIR
    rr = repo_root(repo_override)
    if rr is None:
        return None
    repo_dir = Path(rr)
    if content_id(running_dir) == content_id(repo_dir):
        return None
    msg = f"version desync: running={build_id(running_dir)} repo={build_id(repo_dir)}"
    if log is not None:
        try:
            log.warning("tcm-supercompress: %s", msg)
        except Exception:
            pass
    return msg
