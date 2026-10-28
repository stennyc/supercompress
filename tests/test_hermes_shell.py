"""壳静态验收门（@reviewer 门 #1 本地复现 + 壳薄度 grep 判据 @reviewer 门 #2）。

加载链 = 远端 `plugins/plugin_loader.py::instance_from_module` 同款语义（实码核清）：
mod.register(collector) 优先 → collector.engine 须为宿主 ContextEngine 子类；
probe_availability = 实例非 None 且 .is_available() 为真。
本测试三条判据：
1. 壳在「纯 loader 语境」（插件目录 NOT 预先挂 sys.path）能经 engine.py 的 path 引导
   把 sc/scdata 导入成功——封死 sc/*.py 裸 `import scdata` 断链 → 静默回落内置 compressor。
2. register(ctx) 产出的实例非 None、is_available() 为真，且类型 = probe 包自己的
   engine.SupercompressContextEngine（同一 module 对象身份；跨 module 同名类 isinstance 无意义）。
3. 宿主 ContextEngine 可达时（本机 desktop hermes-agent 源目录），probe 引擎类须
   issubclass 真宿主基类——即远端 loader `instance_from_module` 的 `issubclass(attr, base_cls)`
   判据本地复现。远端 117 跑本测试时 hermes_root=None（AppData 路径不存在），此支 no-op，
   真宿主基类检查由 Hermes 进程自身 loader 在启动时裁决。
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo 根 = 交付面插件目录


def _hermes_agent_root():
    """宿主 agent.context_engine 源目录（本机 desktop = AppData/Local/hermes/hermes-agent）；
    找不到 → None。"""
    cand = os.path.join(os.path.expanduser("~"), "AppData", "Local", "hermes", "hermes-agent")
    if os.path.isfile(os.path.join(cand, "agent", "context_engine.py")):
        return cand
    return None


class _Collector:
    """复刻 plugins/context_engine/__init__.py::_EngineCollector（只收 context engine）。"""

    def __init__(self):
        self.engine = None

    def register_context_engine(self, engine):
        self.engine = engine

    def register_command(self, *a, **k):
        pass


def _load_plugin_module(plugin_dir, module_name):
    """复刻 plugin_loader.load_plugin_module 最小必需语义：
    exec 目录 `__init__.py` 为 module_name（submodule_search_locations = 目录，使 `.engine` 可解析）。"""
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(plugin_dir, "__init__.py"),
        submodule_search_locations=[plugin_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _probe_mod_name():
    return "supercompress_probe_pkg"


def test_sys_path_bootstrap_survives_loader_context(tmp_path):
    """scdata 裸 import 命门：插件目录不在 sys.path 时，engine 模块仍能 import sc/scdata。"""
    for m in list(sys.modules):
        if m in ("scdata",) or m.startswith("sc."):
            del sys.modules[m]
    saved_syspath = list(sys.path)
    sys.path = [p for p in saved_syspath if os.path.abspath(p) != os.path.abspath(_ROOT)]
    try:
        assert os.path.abspath(_ROOT) not in [os.path.abspath(p) for p in sys.path]
        spec = importlib.util.spec_from_file_location(
            "probe_scdata_probe", os.path.join(_ROOT, "engine.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # 触发 engine.py 顶部 path 引导 + import sc
        assert "sc" in sys.modules, "engine import sc failed under loader context"
        assert "scdata" in sys.modules, "engine/scdata bootstrap failed under loader context"
    finally:
        sys.path = saved_syspath
        for m in list(sys.modules):
            if m in ("probe_scdata_probe", "scdata") or m.startswith("sc."):
                del sys.modules[m]


def test_register_produces_active_context_engine(tmp_path):
    """instance_from_module 语义：register(collector) → 实例非 None、probe active、类型=probe 引擎类。"""
    saved = list(sys.path)
    sys.path.insert(0, _ROOT)
    hermes_root = _hermes_agent_root()
    if hermes_root and hermes_root not in saved:
        sys.path.insert(0, hermes_root)
    mod_name = _probe_mod_name()
    # P1 根因修复：del 前保存原始 engine module 身份（module_A = conftest 首次 import 的对象）。
    # test_recall_cache.py 在 COLLECTION 阶段 `from engine import SupercompressContextEngine`
    # 绑定了 module_A；若本 test 把 sys.modules["engine"] 删掉后 fresh import → module_B，
    # conftest autouse fixture 在后续 EXECUTION 阶段 patch 的是 module_B，
    # 而 class_A.__init__ 的 _load_config 仍从 module_A.__dict__ 解析 → 宿主 config 泄漏。
    _orig_engine = sys.modules.get("engine")
    for m in list(sys.modules):
        if m == mod_name or m.startswith(mod_name + ".") or m == "engine":
            del sys.modules[m]
    import importlib
    if _orig_engine is None:
        # 从未 import 过 engine（异常场景）→ 走 importlib 兜底
        sys.modules["engine"] = importlib.import_module("engine")
    try:
        mod = _load_plugin_module(_ROOT, mod_name)
        assert hasattr(mod, "register"), "shell __init__.py must expose register(ctx)"
        ctx = _Collector()
        mod.register(ctx)
        assert ctx.engine is not None, \
            "register() must populate the engine (else loader warns + None → silent fallback)"
        assert ctx.engine.is_available() is True, \
            "probe_availability requires active instance"
        # 类型身份 = probe 包自己的 engine 类（跨 module 同名类无 isinstance 意义）
        probe_cls = importlib.import_module(f"{mod_name}.engine").SupercompressContextEngine
        assert type(ctx.engine) is probe_cls, \
            "register() must return the probe-loaded engine class identity"
        if hermes_root:
            from agent.context_engine import ContextEngine
            assert issubclass(probe_cls, ContextEngine), \
                "engine class must subclass the real host ContextEngine (loader gate #1)"
    finally:
        sys.path = saved
        for m in list(sys.modules):
            if m == mod_name or m.startswith(mod_name + ".") or m == "engine":
                del sys.modules[m]
        # 恢复原始 engine module 身份（module_A = conftest/test_recall_cache 绑定的对象）。
        # 不恢复 module_A 而是 fresh import → module_B → conftest patch 落在 module_A 上，
        # module_B 的 _load_config 未 patch → 宿主 config 泄漏 → TestAutoRecallDefaultOff 失败。
        import importlib
        if _orig_engine is not None:
            sys.modules["engine"] = _orig_engine
        else:
            sys.modules["engine"] = importlib.import_module("engine")


def test_shell_thin_invariant_no_algorithm_in_shell():
    """壳薄度门禁（@reviewer grep 判据）：壳文件内 distill/recall 只有调用无算法体。"""
    with open(os.path.join(_ROOT, "engine.py"), encoding="utf-8") as f:
        body = f.read()
    for forbidden in ("def _evict(", "def _next_candidate", "def _tok", "re.findall",
                      "P0 boost", "content_hash_mismatch", "nfc_hash(self"):
        assert forbidden not in body, f"shell grew algorithm logic: {forbidden!r}"
    assert "Distiller(" in body and "Recaller(" in body, "shell must delegate to sc components"


def test_host_live_compression_contract():
    """宿主实调位回归门（tui_gateway/session_compression.py:169）：
    engine._coerce_threshold_tokens_cap 必须存在且规整语义 = 内置 ContextCompressor 同款
    （正 int 保留 / None、非数、<=0 → None）；缺它 = 宿主每 session 打
    "Could not apply live compression config" WARNING（2026-09-27 17:22:20 实锤）。"""
    import sys
    import importlib
    saved = list(sys.path)
    sys.path.insert(0, _ROOT)
    # 保存原 engine module 对象身份（conftest 的 monkeypatch 打在它身上）：
    # del sys.modules["engine"] + 重新 import engine 会拿到一个新 exec 的 module 对象，
    # 上面 conftest patch 的 _load_config 不在新对象上 → 后续测试（test_recall_cache
    # 的 TestAutoRecallDefaultOff）读宿主 config 泄漏 → "默认关"断言失败。
    # finally 恢复 sys.modules["engine"] = 原对象身份，保证 conftest patch 持续有效。
    _orig_engine = sys.modules.get("engine")
    for m in list(sys.modules):
        if m == "engine" or m.startswith("sc.") or m in ("sc", "scdata"):
            del sys.modules[m]
    try:
        import engine as eng
        # 实例面可访问（宿主是拿活实例调的）
        inst_probe = eng.SupercompressContextEngine
        f = inst_probe._coerce_threshold_tokens_cap
        assert f(4096) == 4096
        assert f("1234") == 1234          # 宿主 config.yaml 值可能是 str
        assert f(None) is None
        assert f(0) is None
        assert f(-5) is None
        assert f("abc") is None
        # threshold_tokens_cap 属性存在（宿主写 self.threshold_tokens_cap = ...）
        import inspect
        src = inspect.getsource(inst_probe.__init__)
        assert "threshold_tokens_cap" in src, "threshold_tokens_cap attr must be initialized"
    finally:
        sys.path = saved
        # 恢复原 engine module 身份（_orig_engine = del 前的 sys.modules["engine"] 对象）：
        # 让后续 import engine 拿到 conftest patch 过的那个对象，TestAutoRecallDefaultOff
        # 的 "缺省默认关" 断言才不被宿主 config 泄漏击穿。
        if _orig_engine is not None:
            sys.modules["engine"] = _orig_engine
        else:
            sys.modules["engine"] = importlib.import_module("engine")
