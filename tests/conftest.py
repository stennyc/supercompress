"""共享 fixtures（pytest 自动加载 tests/conftest.py；tests/ 目录随之进 sys.path）。"""
import pytest

from helpers import mkseg, complete_digest, mk_window, TOPIC, MV  # noqa: F401  (供测试文件复用命名空间)
from fakes import FakeEndpoint
from sc.aux_provider import AuxProvider
from sc.store import JsonlStore


@pytest.fixture(autouse=True)
def _hermetic_hermes_home(monkeypatch):
    """测试 hermetic：清 HERMES_HOME env + patch engine._load_config 返回空 dict，
    保证 engine 默认路径回退（~/.hermes）与宿主 config 读取解耦——
    宿主 venv 里 hermes_cli 可导入时，_load_config 会读 HERMES_HOME/config.yaml，
    把宿主 profile 的 auto_recall:true 等键泄漏进测试面（TestAutoRecallDefaultOff 2 条失败根因）。
    patch 引擎模块的 _load_config 恒返回 {}（最 hermetic，不依赖 env 删净程度）；
    env 驱动路径解析有专属测试（test_session_store.TestStoreRoot_EnvAware）单独封住。"""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    # 宿主 config 泄漏面：engine._load_config() → hermes_cli.load_config() → HERMES_HOME/config.yaml。
    # patch 为恒空 dict（零宿主 config），保证「缺省 = 代码默认」语义（auto_recall 缺省 False 等）。
    import engine as _engine_mod
    monkeypatch.setattr(_engine_mod, "_load_config", lambda: {}, raising=True)


@pytest.fixture
def store(tmp_path):
    return JsonlStore(str(tmp_path / "s.jsonl"))


@pytest.fixture
def aux():
    # sc 层契约测试：关冷却窗（fail_cooldown=0）保持「transient 单事件、下事件自愈」纯净；
    # 引擎层去抖策略（生产默认 60s 冷却）由 engine.py 注入，不进 sc 契约面。
    # 显式 sync 模式（async_digest=False）= 无 worker 线程、digest 立即生成 → 测试确定性；
    # 生产 engine.py 走默认 async（worker 后台生成，turn 线程零阻塞）。
    return AuxProvider(FakeEndpoint(), fail_cooldown=0.0, async_digest=False)
