"""测试替身（@reviewer 方向锁：测试脚手架不进生产 import 面，只住 tests/）。

FakeEndpoint：伪造 LLM 端点。
- `fail=True` → call 抛 RuntimeError（模拟主模型瞬态失败：限流/超时 → transient_fail regime）；
- `responses[task]` → 按任务投喂 canned response（tagging 机器可解析 JSON 测试）；
- `calls` 记录全部 (task, text) 调用（断言「零新 provider：只调主端点」用）。
"""


class FakeEndpoint:
    def __init__(self) -> None:
        self.fail: bool = False
        self.calls: list[tuple[str, str]] = []
        self.responses: dict[str, str] = {}

    def call(self, task: str, text: str) -> str:
        self.calls.append((task, text))
        if self.fail:
            raise RuntimeError("transient_fail: endpoint rate-limited")
        return self.responses.get(task, "")
