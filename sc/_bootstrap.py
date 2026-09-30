"""安装可移植路径的单一事实源（不写死任何安装路径 / 用户 home / profile 名）。

设计目标：supercompress 要能安装到任意环境（不同 hermes root、不同 $HOME、
不同 profile 数据根）。本模块只从两处派生路径，零硬编码：
- ``INSTALL_ROOT``：从本文件 ``__file__`` 位置反推 hermes-agent 仓根
  （``sc/`` 上溯 4 级 = 仓根），换任何安装目录都成立；
- ``HERMES_HOME``：读进程 env（gateway / cron / systemd 子进程按 profile 各带
  ``HERMES_HOME``，profile 数据根 = 该值；缺省回退 ``~/.hermes``）。

所有 CLI 脚本（sc/retag.py、sc/patch_*.py、sc/check_supercompress.py、根 retag.py）
与 engine.py 生产数据面共用本模块解析，避免各写各的绝对路径。
"""
from __future__ import annotations

import os
from pathlib import Path

# hermes-agent 仓根 = 本文件（…/plugins/context_engine/supercompress/sc/_bootstrap.py）
# 上溯 4 级：sc → supercompress → context_engine → plugins → <仓根>。
# 换任何安装位置（/usr/local、/opt、venv、源码 checkout）都自动成立。
INSTALL_ROOT: str = str(Path(__file__).resolve().parents[4])


def hermes_home() -> str:
    """数据根（HERMES_HOME）：进程 env 优先（per-profile），缺省 ``~/.hermes``。

    与宿主 ``hermes_constants.get_hermes_home()`` 同源语义（override → env → 平台默认）；
    本模块不 import 宿主（保持插件独立可运行），仅约定同一个 env 名 ``HERMES_HOME``。
    """
    hh = os.environ.get("HERMES_HOME", "").strip()
    return hh if hh else os.path.expanduser("~/.hermes")


def config_path() -> str:
    """当前 profile 的 config.yaml（数据根下，不写死某 profile 名）。"""
    return os.path.join(hermes_home(), "config.yaml")


def store_dir() -> str:
    """supercompress 数据目录（数据根下）：per-session <sid>.jsonl + legacy active.jsonl 同住。"""
    return os.path.join(hermes_home(), "supercompress_store")


def active_store_path() -> str:
    """legacy 单一 active.jsonl（数据根下）。"""
    return os.path.join(store_dir(), "active.jsonl")


def setup_script_path() -> str:
    """aux 端点 setup 脚本落点（数据根下，随 profile 各存各的，不写死 profile 名）。"""
    return os.path.join(hermes_home(), "setup_supercompress_aux.sh")


def install_root() -> str:
    """仓根（= INSTALL_ROOT，别名供脚本侧统一命名导入）。"""
    return INSTALL_ROOT


def require_install_root() -> str:
    """校验仓根层级（让深度假设失败得更可读，@reviewer f3a3df3 Minor②）。

    本插件须置于 ``…/plugins/context_engine/supercompress/sc/`` 层级——
    ``INSTALL_ROOT`` 由 ``__file__`` 上溯 4 级反推，层级不符时仓根解析到错处。
    正确层级下 ``INSTALL_ROOT/plugins/context_engine/supercompress/sc/_bootstrap.py``
    必存在（= 本文件）；层级被搬动则缺失。缺失即 raise 带说明的
    ``RuntimeError``（而非等到 ``import plugins…`` 抛裸 ``ImportError``）。

    返回校验通过的仓根路径。生产正确层级下恒不 raise。
    """
    root = Path(INSTALL_ROOT)
    rel = "plugins/context_engine/supercompress/sc/_bootstrap.py"
    if not (root / rel).is_file():
        raise RuntimeError(
            f"supercompress 安装层级不符：{root} 下缺少 {rel}。"
            "本插件须置于 hermes-agent 仓的 plugins/context_engine/supercompress/ 内"
            "（…/plugins/context_engine/supercompress/sc/）；若已复制到其他位置，"
            "请调整目录结构后再运行 CLI 脚本。"
        )
    return str(root)
