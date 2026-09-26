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


def _http_server(handler):
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_call_refuses_redirect_not_silently_followed():
    """@user GO + @reviewer 隔离的坑：`http://` 端点被 307 到 https 时，urllib 跟随重定向
    （跨 scheme/host 丢 Bearer key）→ 静默 401。LiveEndpoint 遇**任何**重定向直接
    RuntimeError 显式拒出——「静默丢 key」变「显式报错」，本地 307 服务器真跑复现。"""
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/v1/chat/completions":
                self.send_response(307)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_address[1]}/target")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                body = b'{"choices":[{"message":{"content":"X"}}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv, port = _http_server(H)
    try:
        ep = LiveEndpoint(f"http://127.0.0.1:{port}/v1", "k", "agnes-3.0-flash")
        try:
            ep.call("x", "y")
            raise AssertionError("重定向必须显式拒出，不静默跟随（防静默丢 key）")
        except RuntimeError as e:
            assert "重定向" in str(e), "报错须点明是重定向"
            assert "/target" in str(e), "报错须带重定向目的地，便于直接改 canonical URL"
    finally:
        srv.shutdown()


def test_call_plain_200_still_works():
    """重定向守卫不得误伤正常路径：无重定向的 200 直返照常出内容。"""
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = b'{"choices":[{"message":{"content":"ok-content"}}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv, port = _http_server(H)
    try:
        ep = LiveEndpoint(f"http://127.0.0.1:{port}/v1", "k", "agnes-3.0-flash")
        assert ep.call("x", "y") == "ok-content"
    finally:
        srv.shutdown()
