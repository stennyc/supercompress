# supercompress

> **上下文蒸馏器 + session 级记忆骨干** — 压缩管窗口密度、记忆管持久，两层不混。
> **Context distiller + session-level memory backbone** — compression manages window density; memory manages persistence. The two layers never mix.

它兑现两个原始诉求 / It delivers two original guarantees:

- **防跑偏** — 目标 + 活账本每阶段落账、常驻窗内（锚点永不裁）。*Goal + live ledger, anchored and always resident; anchors are never evicted.*
- **防失忆** — 关掉 / 崩了 / 断电重开，session 记忆还在。*Close / crash / power-loss, then reopen: the session memory is still there.*

零向量库 · 零外部服务 · 无常驻进程。运行时**纯 Python 标准库**，无第三方依赖。
Zero vector DB · zero external services · no daemon. **Pure Python stdlib at runtime, no third-party dependencies.**

---

## 功能特性 / Features

| 能力 / Capability | 行为 / Behavior |
|---|---|
| **P0 写前即持久 / write-before-compress durability** | 目标、决策、护栏等 P0 段**即写即 fsync + 读回**，过了才发射锚点；源写失败 = 全量不压缩、零锚点。P0 is fsync'd + read back on write, *before* compression; the anchor is only emitted once durable. A failed source write aborts compression with zero anchors. |
| **断电上界有界 / bounded power-loss window** | 断电最多丢一个 ≤256 KB 的 P1/P2 pending 批（size trigger 已实装：pending ≥ `lazy_batch_cap` 即 flush，批上界恒 ≤ cap；默认可配）；已 durable 的 P0 不丢。Power loss loses at most one P1/P2 pending batch ≤ **256 KB** (size trigger implemented: pending ≥ `lazy_batch_cap` flushes immediately, so batch is always ≤ cap; configurable); durable P0 is never lost. |
| **可逆压缩 / reversible compression** | 逐出后原文可全量取回（jsonl 单一事实源），非单向有损。Evicted original text is fully retrievable from the single jsonl source of truth — compression is reversible, not lossy. |
| **语义召回 / recall** | 点查 O(1)（`segment_id`→原文）+ BM25 语义 top-k（P0 boost、stale P1 降权）+ Pull 预算 enforce。无 embedding、无向量库。O(1) point-fetch + BM25-semantic top-k (P0 boost, stale-P1 downweight) + pull-budget enforcement. No embeddings, no vector store. |
| **活账本 / live ledger** | 顶层 goal 永久钉死；新 user 指令 supersede 旧 phase（未完成 task 关为 `blocked`，jsonl 留审计不删行）；DONE 三级证据（verified / declared / missed）。Top-level goal is permanently pinned; a new user instruction supersedes the old phase (open tasks close as `blocked`, kept in the jsonl audit trail, never deleted). DONE has three evidence levels. |
| **跨期重开 / reopen across periods** | 归档 → 新活跃文件 = **字节稳定拷贝** + 单 replay 路径；外部引用跨期标 `stale`，行动前 verify。*「恢复你的计划，不恢复你的世界」*。Archive → new active file is a byte-stable copy on a single replay path; cross-period external refs are marked `stale` and verified before acting. *We recover your plan, not your world.* |

---

## 架构 / Architecture

五个模块，职责单一 / Five single-responsibility modules:

| 模块 / Module | 一句话 / One-liner |
|---|---|
| ① `sc/distill.py` 蒸馏器 / distiller | 纯无状态 tiering → P0 写前持久 → P1/P2 lazy 批（size trigger：pending ≥ 256KB 即 flush）→ 锚点 + jsonl 行。Stateless tiering → P0 write-before-compress → P1/P2 lazy batches (size trigger: flush when pending ≥ 256KB) → anchors + jsonl rows. |
| ② `sc/recall.py` 检索 / recall | 点查 O(1) + 纯 BM25 top-k（P0 boost）+ Pull 预算 20% enforce。O(1) point-fetch + pure BM25 top-k + pull budget. |
| ③ `sc/ledger.py` 活账本 / ledger | 双层（goal 永不裁 / 子目标可 supersede）+ 确定性触发 + DONE 三级证据。Two-layer ledger + deterministic triggers + three-level DONE evidence. |
| ④ `sc/tagger.py` 标记 / tagger | P0 确定性结构地板（语言无关）+ LLM tiering（机器可解析 JSON）。Deterministic language-agnostic P0 floor + LLM tiering via machine-parseable JSON. |
| 数据层 `scdata.py` jsonl 骨干 / data layer | 单一事实源、append-only、`content_hash=sha256(NFC)` 幂等去重、torn-tail 单 replay、跨期同一条重放。Single source of truth, append-only, idempotent `sha256(NFC)` dedup, single torn-tail-safe replay path. |

辅助面 / supporting surface: `sc/aux_provider.py`（aux ≡ 主模型端点，零新增 provider、provider-agnostic；测试替身住 `tests/fakes.py`，不进生产 import 面）和 `sc/live_endpoint.py`（把 Hermes 现路由 resolve 成 bench 可注入端点，占位模型名拒出）。
`sc/aux_provider.py` (aux ≡ the main-model endpoint, zero new providers, provider-agnostic; the test double lives in `tests/fakes.py`, kept out of the production import surface) and `sc/live_endpoint.py` (resolves the active routing into a bench-injectable endpoint; refuses placeholder model names).

详细设计依据见 / full design rationale in [`docs/`](docs/): [`product.md`](docs/product.md) · [`features.md`](docs/features.md) · [`implementation.md`](docs/implementation.md) · [`ops-deployment.md`](docs/ops-deployment.md) · [`design-rationale.md`](docs/design-rationale.md).

---

## 安装 / Installation

supercompress 是一个**纯源码树**（无打包、无 `pip install .`）——把仓库根目录加进 `PYTHONPATH`（或就在仓库根目录运行），`import sc` / `import scdata` 即可。运行时**零第三方依赖**；开发/测试需 `pytest`。

supercompress is a **plain source tree** (no packaging, no `pip install .`) — put the repo root on `PYTHONPATH` (or just run from the repo root) and `import sc` / `import scdata` works out of the box. **Zero third-party runtime deps**; dev/test needs `pytest`.

**Requirements:** Python 3.9+ (tested on 3.12).

### 一键脚本 / one-command scripts

| 平台 / Platform | 命令 / Command |
|---|---|
| macOS / Linux / WSL / MSYS | `bash install.sh` |
| Windows PowerShell | `powershell -ExecutionPolicy Bypass -File install.ps1` |

两个脚本都做四件事 / both scripts do four things: 创建 `.venv` → 装 `pytest` → 跑全量测试套件验证 → 自动配齐 Hermes `config.yaml`（定位 `~/.hermes` 或 `$HERMES_HOME`，缺失写全键默认块、已有不覆盖仅报缺键）。
create `.venv` → install `pytest` → run the full test suite → auto-populate Hermes `config.yaml` (locates `~/.hermes` or `$HERMES_HOME`; writes the full default block when missing, never overwrites an existing file — reports missing keys only).

> 给目标 agent 的完整安装/配置提示词（5 步 + 红线）见 [`docs/AGENT-INSTALL.md`](docs/AGENT-INSTALL.md) — full agent-facing install/config prompt: see [`docs/AGENT-INSTALL.md`](docs/AGENT-INSTALL.md).

### 手动 / manual

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # macOS / Linux / WSL
# .venv\Scripts\pip install -r requirements.txt    # Windows
```

---

## 快速上手 / Quick start

```python
import scdata
from sc import Segment, JsonlStore, Window, WindowSeg, Distiller
from sc.aux_provider import AuxProvider
from sc.recall import Recaller
from sc.ledger import Ledger
from tests.fakes import FakeEndpoint   # 测试替身 — 生产请注入真实端点
                                       # test double — inject a real endpoint in prod

# 1) 数据层：单一 append-only jsonl 事实源
#    data layer: one append-only jsonl source of truth
store = JsonlStore("memory/session.jsonl")

# 2) P0 段：即写即 fsync + 读回（write-before-compress 硬 gate）
goal = store.append(Segment(tier="P0", provenance="user",
    original_text="goal: ship the migration by Friday"))

# 3) 活账本：goal 钉死 + phase 可 supersede + DONE 三级证据
ledger = Ledger(store, session_topic="migration", goal=goal.original_text,
                checkpoint_every=4)
ledger.on_user_instruction("first, drain the old DB")
ledger.add_task("drain", "drain the old DB")
ledger.on_done("drain", evidence="drain complete, 0 pending txns")  # -> "verified"

# 4) 蒸馏器：80% 软水位 + P0 逐出 gate + push cap + 锚点发射
aux = AuxProvider(FakeEndpoint())
window = Window(capacity_tokens=1000,
                segs=[WindowSeg(s) for s in store.by_hash.values()])
result = Distiller(window, store, aux).run()
print(result.status)          # "ok" | "backpressure" | "compress_failed"

# 5) 召回：点查 O(1) + BM25 top-k（P0 boost）
rec = Recaller(store, window)
for hit in rec.search("auth service port 8080", k=3):
    print(hit.seg.segment_id, hit.score, hit.stale, hit.note)

# 6) 跨期重开：归档 → 新活跃文件（字节稳定 + 单 replay 路径）
store2 = JsonlStore.materialize("memory/session.jsonl",
                                "memory/next.jsonl", reopen_from="2026-09-27")
print(len(scdata.replay("memory/next.jsonl")), "segments replayed")
```

> 生产端点 / production endpoint：把 `AuxProvider` 的 `main_endpoint` 换成
> 你的 Hermes 现路由（OpenAI 兼容网关）。`resolve_live_endpoint()` 读
> `LIVE_ENDPOINT_URL/KEY/MODEL` → `OPENAI_*` → `RuntimeError`（显式失败，
> 不静默落回替身）。In prod, swap `main_endpoint` for your active routing.
> `resolve_live_endpoint()` reads `LIVE_ENDPOINT_URL/KEY/MODEL` → `OPENAI_*` →
> `RuntimeError` (explicit failure, never a silent fallback to a fake).

---

## 运行测试 / Running tests

```bash
export PYTHONPATH="$(pwd)"                 # 让 sc / scdata 可达
.venv/bin/python -m pytest                 # 全量套件 / full suite
.venv/bin/python -m pytest tests/test_block1.py   # 单模块 / single module
```

基准探针（需真实端点，可选）/ benchmark probes (need a live endpoint, optional):

```bash
LIVE_ENDPOINT_MODEL=<model> .venv/bin/python bench/recall_bench.py
LIVE_ENDPOINT_MODEL=<model> .venv/bin/python bench/decision_layer_probe.py --run 8
```

---

## 目录结构 / Layout

```
sc/                 # 产品树 — 运行时纯标准库 / product tree, pure stdlib at runtime
  schema.py         #   Segment 信封 + NFC content_hash + token 估计
  store.py          #   单一事实源 jsonl（tier-aware fsync；数据面委托 scdata）
  budgets.py        #   共享分母（Pull 20% / Push 15% 一处定义）
  distill.py        #   ① 蒸馏器
  recall.py         #   ② 检索
  ledger.py         #   ③ 活账本
  tagger.py         #   ④ 标记
  aux_provider.py   #   aux ≡ 主模型端点（零新 provider）
  live_endpoint.py  #   bench 可注入端点 resolve
scdata.py           # 权威数据层 — 行字节 / replay / materialize / 闭合校验 单点产出
bench/              # 基准探针 + 换档守卫（advisory，不 block build）
tests/              # 全量测试 + 测试替身（fakes.py，不进生产 import 面）
docs/               # 设计依据（产品/实现/部署/rationale，中文）
install.sh          # POSIX 安装脚本 / installer
install.ps1         # Windows 安装脚本 / installer
requirements.txt    # 开发/测试依赖（仅 pytest）/ dev deps only
```
