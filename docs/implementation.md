# supercompress — 实现层（机制模块 + 端点面，@coder own）

> 本段只写**实查过的事实**（@ `ee06945`，Windows 开发机布局——本机路径为历史锚不随发行；代码面路径零硬编码，解析链见 `sc/_bootstrap.py`），非设计期话术。
> 装配注：本节供 `docs/features.md` 组装直接收录；数据层归 `docs/ops-deployment.md` 上游（@data-engineer 报过的 jsonl 骨干段），两节不混。

## 四个机制模块

| 模块 | 功能（一句话） | 关键不变量 |
|---|---|---|
| ① `sc/distill.py` 蒸馏器 | 纯无状态 distiller：tiering（P0/P1/P2）→ P0 write-before-compress（即写即 fsync + 读回过了才发射锚点）→ P1/P2 lazy 批（size trigger：pending ≥ 256 KB 即 flush）→ 锚点 + JSONL 行 | P0 护栏 100% durable；断电只丢一个 ≤256 KB 批（可重跑 P1/P2）；digest pending 时 P0 **不可驱逐**；未 fsync 的 segment 永不发锚点 → 崩溃后死指针恒为 0 |
| ② `sc/recall.py` 检索 | 点查 O(1) + 纯 Python token-overlap top-k（BM25 语义，仅标准库 `re`，无 `rank_bm25` import，P0 同相关度 boost）+ Pull 预算 20% enforce | 无 embedding、无向量库；semantic-gap 纯遥测（>15% 才触发可选 v2 语义臂，默认永不自动开） |
| ③ `sc/ledger.py` 账本 | 新 user 指令 supersede 旧 guardrail/task；被替代项标记后降 digest-only | 账本尾部结构 P0；digest 未就绪前 P0 不驱逐；跨期重开走同一条 replay 路径 |
| ④ `sc/tagger.py` 标记 | P0 结构标记 + 外部引用跨期 stale 标记（行动前 verify） | tag 决定 tier 门控，不进数据面 |

## 辅助面

- **`sc/aux_provider.py`**（Option A）：aux ≡ 主模型端点（Hermes 原生 `auxiliary.compression`，零新增服务、provider-agnostic）；网络异常 → 单事件 transient skip + 下一事件自愈。测试替身 `FakeEndpoint` 在 `tests/fakes.py`，prod import 面零脚手架。
- **`sc/live_endpoint.py` 端点面**：把 Hermes 现路由（团队 OpenAI 兼容网关 `/v1` + 网关 key）resolve 成 bench 可注入的 `LiveEndpoint`；解析链 `LIVE_ENDPOINT_* → OPENAI_* → RuntimeError`，**占位模型名拒出**（不落 `"main"`、不静默回 FakeEndpoint——循环论证纪律写死在 resolve 里）。75% 数的双锚归属（HEAD + 真模型名）机械可查，换档 = 一行 `--run` 重推。

## ③ 换档守卫（决策层 75%/80% 的防过期半边）

- 契约 + 三态语义落 `bench/decision_layer_probe.py` 头注（`f627935`）：`bench/.last_75_stamp` 双锚 JSON `{"head","model","n","pass"}`；preflight 断言（`640399b`）：双锚匹配 → PASS / head **或** model 任一失配 → WARN 拒引用 / 文件缺失 → WARN 无记录（非 vacuous pass）；全程 advisory 不 block build。
- 正确动作 = 看到 WARN 跑一次 `LIVE_ENDPOINT_MODEL=<新模型名> && .venv/Scripts/python.exe bench/decision_layer_probe.py --run 8`，stamp 自然刷新——**绝不手改 stamp head 消 WARN**（那是整条 commit-锚链要杀的假归属）。

## 故障注入面（prod 纪律）

- `distill(inject_fault_after=…)` 钩子在 `SC_INJECT_FAULT` env 未设时**恒 inert**（不用 `__debug__` 门控）；双向 marker 测试锁死 off/inert + on/生效 两态。

## 测试面

- 全量 `pytest tests/` **60/60 @ `ee06945`**（含 ① 侧 7 条子集、marker 双向、闭合非空前置、单一实现自检、③ 的 4 条断言 + `LiveEndpoint` 3xx 拒重定向 2 条本地服务器测试）；`FakeEndpoint` 不出现在任何 prod import 路径。
