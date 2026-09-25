"""活端点 resolve 路径机械测试（@user 判定：活端点 = Hermes 现路由，零新配置）。

不出 75%（那是 @researcher 决策层探针 + 活端点出数）——这里只锁 resolve 语义：
显式顺序、model 标识随行、端点缺失显式失败（不静默落回 FakeEndpoint）。
"""
from sc.live_endpoint import LiveEndpoint, resolve_live_endpoint


def _clear(monkeypatch):
    for var in ("LIVE_ENDPOINT_URL", "LIVE_ENDPOINT_KEY", "LIVE_ENDPOINT_MODEL",
                "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)


def test_resolve_live_endpoint_env_precedence(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LIVE_ENDPOINT_URL", "http://10.1.1.116:8080/v1")
    monkeypatch.setenv("LIVE_ENDPOINT_KEY", "k1")
    monkeypatch.setenv("LIVE_ENDPOINT_MODEL", "agnes-3.0-flash")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.default")
    monkeypatch.setenv("OPENAI_API_KEY", "k2")
    monkeypatch.setenv("OPENAI_MODEL", "other-model")
    ep = resolve_live_endpoint()
    assert isinstance(ep, LiveEndpoint)
    assert ep.base_url == "http://10.1.1.116:8080/v1", "LIVE_ENDPOINT_* 优先"
    assert ep.api_key == "k1" and ep.model == "agnes-3.0-flash"


def test_resolve_falls_back_to_openai_compat(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.default")
    monkeypatch.setenv("OPENAI_API_KEY", "k2")
    monkeypatch.setenv("OPENAI_MODEL", "agnes-3.0-flash")
    ep = resolve_live_endpoint()
    assert ep.base_url == "http://openai.default" and ep.api_key == "k2"
    assert ep.model == "agnes-3.0-flash"


def test_resolve_model_identifier_recorded(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LIVE_ENDPOINT_URL", "http://x/v1")
    monkeypatch.setenv("LIVE_ENDPOINT_KEY", "k")
    monkeypatch.setenv("LIVE_ENDPOINT_MODEL", "agnes-3.0-flash")
    ep = resolve_live_endpoint()
    assert ep.model == "agnes-3.0-flash", "确切 model 标识随行（换档 = 重推 75%）"


def test_resolve_missing_model_refuses_no_placeholder(monkeypatch):
    """@reviewer 口径②：model 缺省落到占位 'main' = 非标识 → 拒出数（同端点缺失纪律）。"""
    _clear(monkeypatch)
    monkeypatch.setenv("LIVE_ENDPOINT_URL", "http://x/v1")
    monkeypatch.setenv("LIVE_ENDPOINT_KEY", "k")
    try:
        resolve_live_endpoint()
        raise AssertionError("缺真实 model 名必须显式失败，不落占位 'main'")
    except RuntimeError as e:
        assert "LIVE_ENDPOINT_MODEL" in str(e)


def test_resolve_missing_endpoint_raises_no_silent_fallback(monkeypatch):
    _clear(monkeypatch)
    try:
        resolve_live_endpoint()
        raise AssertionError("端点缺失必须显式失败，不静默落回 FakeEndpoint")
    except RuntimeError as e:
        assert "活主模型端点" in str(e)
