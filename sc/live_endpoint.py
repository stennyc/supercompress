"""活端点：直接调用 Hermes 已配的模型路由（@user 点破——不另配任何新服务）。

@user 的判定：bench 探针 / aux digest 需要的「活主模型端点」= Hermes 现在就在用的
主模型路由本身（本机 = Sub2API OpenAI-compat 网关），零新增服务、零新增 provider。
(A) 语义「main == digest 生成者」天然满足——digest/tagging/bench 探针全部注入这同一个
`main_endpoint`，不需要任何配对动作。

`LiveEndpoint` 与测试替身 `FakeEndpoint` 同形（`call(task, text) -> str` + `fail` 语义由
网络异常承接 → `AuxProvider._generate` 的 except → transient_fail regime，既有单事件
skip + 下事件自愈链原样复用，零新故障面）。

解析顺序（显式、可复现，不猜）：
1. `LIVE_ENDPOINT_URL` + `LIVE_ENDPOINT_KEY` + `LIVE_ENDPOINT_MODEL`（必填真实模型名，
   占位 "main" 拒出数——@reviewer 口径②）——部署侧把 Hermes 现路由填进来即可
   （本机 = Sub2API 的 /v1 + x-api-key + agnes 模型名）；
2. OpenAI 兼容默认：`OPENAI_BASE_URL` + `OPENAI_API_KEY` + `OPENAI_MODEL`（同上必填）；
3. 都没有 → RuntimeError（显式失败：活端点缺失 = 决策层 75% 不可出数，不静默落回
   FakeEndpoint——对替身出 75% 是循环论证，@researcher 纪律）。

model 标识随端点对象携带（@data-engineer 数据层要求：75% 结果记确切 model 名，
它是 digest 幂等再生三元组的语义位，换 main-model 档位 = 数必须重推）。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class LiveEndpoint:
    """OpenAI-compat 主模型端点（同形 FakeEndpoint：`call(task, text) -> str`）。

    网络异常向上抛（限流/超时/鉴权失败）——调用侧 `AuxProvider` 统一接成 transient_fail。
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model          # 确切 model 标识（75% 结果随数记录，换档 = 重推）

    def call(self, task: str, text: str) -> str:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": f"[{task}] {text}"}],
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 重定向（3xx）显式拒出，不静默跟随：跨 scheme/host 的重定向丢 Bearer key
            # （`http://` 被 307 到 `https://` = 落地 401 的经典坑，@reviewer 隔离、@devops curl 复现）。
            # 报错带目的地，让人直接改成 canonical https URL，而非对着 401 瞎猜。
            if 300 <= e.code < 400:
                loc = e.headers.get("Location", "?")
                raise RuntimeError(
                    f"LiveEndpoint: 端点重定向（HTTP {e.code} -> {loc}）——不跟随重定向"
                    f"（跨 scheme/host 会静默丢 Bearer key → 401）。"
                    f"把 base_url 直接指向 canonical URL（如 https://…/v1）。") from e
            raise  # 4xx/5xx 原样向上抛（限流/鉴权/超时 → AuxProvider transient_fail 自愈链）
        return data["choices"][0]["message"]["content"]


def resolve_live_endpoint() -> LiveEndpoint:
    """按显式顺序解析 Hermes 现路由（不猜）；端点或 model 标识缺失 → RuntimeError
    （决策层 75% 前置未满足，不得静默落回 FakeEndpoint / 占位 model 名）。

    model 标识纪律（@reviewer 口径②）：`LIVE_ENDPOINT_MODEL`/`OPENAI_MODEL` 必须是真实
    模型名（如 agnes-3.0-flash）——它是 digest 幂等再生三元组第 3 位 + 75% 数归属的机械锚；
    缺省落到字面量 "main" = 非标识、换档不可查 → 视为未解析，拒出数（同端点缺失那条纪律）。
    """
    url = os.environ.get("LIVE_ENDPOINT_URL")
    key = os.environ.get("LIVE_ENDPOINT_KEY")
    if url and key:
        model = os.environ.get("LIVE_ENDPOINT_MODEL")
        if not model:
            raise RuntimeError(
                "resolve_live_endpoint: LIVE_ENDPOINT_MODEL 未设真实模型名——"
                "决策层 75% 是 model-specific，缺确切 model 标识 = 数无法归属；"
                "显式失败，不落占位 'main'")
        return LiveEndpoint(url, key, model)
    url = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")
    if url and key:
        model = os.environ.get("OPENAI_MODEL")
        if not model:
            raise RuntimeError(
                "resolve_live_endpoint: OPENAI_MODEL 未设真实模型名——"
                "同上，缺 model 标识拒出数")
        return LiveEndpoint(url, key, model)
    raise RuntimeError(
        "resolve_live_endpoint: 无活主模型端点（LIVE_ENDPOINT_URL/KEY 或 OPENAI_BASE_URL/API_KEY 均未设）"
        "——决策层 75% 不可出数；显式失败，不回落测试替身")
