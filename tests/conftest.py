"""共享 fixtures（pytest 自动加载 tests/conftest.py；tests/ 目录随之进 sys.path）。"""
import pytest

from helpers import mkseg, complete_digest, mk_window, TOPIC, MV  # noqa: F401  (供测试文件复用命名空间)
from fakes import FakeEndpoint
from sc.aux_provider import AuxProvider
from sc.store import JsonlStore


@pytest.fixture
def store(tmp_path):
    return JsonlStore(str(tmp_path / "s.jsonl"))


@pytest.fixture
def aux():
    return AuxProvider(FakeEndpoint())
