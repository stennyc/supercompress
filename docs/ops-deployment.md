# supercompress — 部署 / 运行面（ops 段，@devops own）

> 本段只写**实查过的事实**（@ `76810d3`，`C:\Users\user\supercompress`），设计期话术与成品区分开标注。

## 1. 部署足迹（实查口径）

- **一个 jsonl + 内存 `content_hash→line_offset` 索引 + `rank_bm25` + `numpy`**。
  依赖实装于 `.venv`：`rank_bm25-0.2.2`（其 requires = `numpy` + dev-extra `pytest`）、`numpy-2.5.3`（pip wheel，免编译）。
- **无常驻进程、无后台**（实查）：prod 树（`sc/ scdata.py bench/`）grep `inotify|watch|daemon|threading|asyncio|subprocess.Popen` = 0 命中。所有组件都是被调用的纯函数/对象，没有 watch hook、没有 daemon。
- **零向量库、零外部服务**：无 qdrant/embedding/faiss 依赖；检索 = 纯 BM25 词法（+ P0 boost）。

## 2. 路径与归档（部署参数，非 prod 常量）

- prod 代码**不硬编码 jsonl 落点**：`JsonlStore(path)` / `materialize(archive, active, reopen_from)` 全由调用方传路径。
- 部署约定（本机）：活跃文件放 scratch，session 结束**原子 rename** 到 `~/.hermes/sessions/<id>.jsonl`。
  约束：scratch 与归档目录**必须同卷**（本机都在 C:，NTFS 同卷 rename 原子；跨卷退化为 copy+delete、有非原子窗口）。
- 归档文件 = 只读审计，**不进召回索引**（索引只从活跃文件构建，防 session-scoped 语义被历史文件污染）。
- Retention：默认 **keep 不删**；LRU/自动清理是 opt-in，默认未启用。

## 3. IO 策略（tier-aware fsync）

| 层 | 写盘行为 |
|---|---|
| P0（含账本 + 顶层 goal） | **即写即 fsync + 读回校验**，过了才发射锚点（write-before-compress 硬 gate） |
| P1/P2 | lazy 批：≤256 KB 或 ~2 s 触发 flush，**不逐行 fsync**（append + 批量，无随机 IO 惩罚） |

未 fsync 的 segment **永不发锚点** → 崩溃后死指针数量恒为零（`verify_anchor_closure` 单一实现，`scdata.py`）。
Torn-tail 重放免费：每行自带 `content_hash`，坏行/截断行静默丢弃，从有效头部重建索引。

## 4. 崩溃边界（已知边界文档，已 test-mapped）

- **进程级崩溃（重启/OOM，日常最常见）≈ 0 内容丢失**：OS page cache 存活，replay 基本全救回；工作窗口态清零 = session 重启（短期记忆定位的已知边界，非故障）。
- **断电级 = 丢一个 P1/P2 lazy 批**（上界 256 KB + in-flight）；已 durable 的 P0（goal/决策/护栏）100% 不丢。gate⑤ 断电三注入点（`mid_p1_batch` / `post_p0_fsync` / `gate_boundary`）在 `tests/test_preflight_ops.py` 真跑闭环，`SC_INJECT_FAULT` marker off 态恒 inert（prod 泄漏自检）。
- **降级触发源**（disk 面）：`ENOSPC / fsync 失败 / 文件锁` → `degraded` 状态 + backpressure；P0/P1 abort、仅 P2 过置信度门走 `lossy_one_way`（显式标记，非静默有损）。

## 5. ops 验证入口（ship 时机械跑）

| 动作 | 命令 / 位置 | 证什么 |
|---|---|---|
| 全量套件 | `.venv\Scripts\python.exe -m pytest tests/` | 58/58 @ HEAD（42 实现 + 8 数据 ref + 8 ops/preflight，含 ③ 四条） |
| gate⑤ 断电半边 | `tests/test_preflight_ops.py` power-loss 三注入 | 断电上界、零悬空锚点、marker 双向 |
| ③ stamp advisory | 同文件 `check_stamp_advisory`（随 pytest 出三态） | 75% 数双锚（HEAD + model）失配即 WARN，不 block build |
| 网关健康 | `curl /v1/models` 200 才放行决策层出数 | Sub2API 不可达 → 探针 REFUSED、不产数、不写 stamp |
| 换模型档位 | `LIVE_ENDPOINT_MODEL=<新模型名> && .venv\Scripts\python.exe bench/decision_layer_probe.py --run 30` | 一行重推，stamp 自动刷新（**别手改 stamp head 消 WARN**） |

## 6. 未落项（如实标注，不进 ship 口径）

- 归档/retention **路径与策略是部署约定，不是 prod 代码**——超压缩仓库本身不替你建 `~/.hermes/sessions/`，接入方（Hermes）落地时按 §2 约束实现。
- 「甩掉 numpy、自写纯 Python BM25」为可选 v2（footprint 再减一项 native wheel），默认不建（无消费者不建）。
