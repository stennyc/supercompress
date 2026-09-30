"""Supercompress Context Engine — Hermes 宿主适配壳（thin adapter, v2 重构版）。

loader 契约（远端 117 `plugins/plugin_loader.py` 实码核清，@coder 直查 2026-09-27）：
- `load_plugin_module` exec 本 `__init__.py`（qualified 名 `plugins.context_engine.supercompress`，
  submodule_search_locations = 本目录 → `.engine` 相对 import 可解）；
- `instance_from_module` 优先调 `mod.register(collector)`，collector（_EngineCollector）
  `register_context_engine(engine)` 收实例；register 缺失/异常 → 回退扫描模块属性找
  ContextEngine 子类无参构造；
- `probe_availability` = 实例非 None 且 `.is_available()`（若有）为真；
- 内置引擎目录启发式 `_is_context_engine_dir` 对 bundled 目录跳过（`plugins/context_engine/<name>`
  天然命中），本 __init__ 仍保留 `register_context_engine` 字样供 user-ns 发现链。

壳零业务逻辑：`SupercompressContextEngine` 全委托 `sc` 组件（distill/recall/budgets），
实现在 engine.py。旧 `get_engine()` / `is_available()` 顶层入口保留（bench 探针 / 直引兼容）。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register(ctx: Any) -> None:
    """Hermes context-engine loader 入口：构造并登记引擎实例。

    ctx 须有 `register_context_engine(engine)`（plugins/context_engine/__init__.py 的
    _EngineCollector）。实例化走 kwargs 默认（宿主 config `compression.supercompress` 段
    由 engine.py::_load_config 读取）；构造失败 → 本函数 re-raise → loader 捕获 →
    warn + None → 宿主回落内置 compressor（本壳的静默失败模式，静态验收门 #1 封死）。
    """
    from .engine import SupercompressContextEngine
    ctx.register_context_engine(SupercompressContextEngine())


def get_engine(**kwargs: Any):
    """旧顶层导入面兼容（宿主 loader 走 register；bench / 直引走这里）。"""
    from .engine import get_engine as _get_engine
    return _get_engine(**kwargs)


def is_available() -> bool:
    """顶层可用性（旧契约保留；loader probe 走实例 .is_available()，两者一致 = True）。"""
    from .engine import SupercompressContextEngine
    return bool(SupercompressContextEngine)
