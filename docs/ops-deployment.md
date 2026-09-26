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
| 全量套件 | `.venv\Scripts\python.exe -m pytest tests/` | 60/60 @ HEAD（34 实现 + 8 数据 ref + 11 ops/preflight + 7 端点面含 3xx 拒重定向 2，含 ③ 四条） |
| gate⑤ 断电半边 | `tests/test_preflight_ops.py` power-loss 三注入 | 断电上界、零悬空锚点、marker 双向 |
| ③ stamp advisory | 同文件 `check_stamp_advisory`（随 pytest 出三态） | 75% 数双锚（HEAD + model）失配即 WARN，不 block build |
| 网关健康 | 团队真路由 `LIVE_ENDPOINT_URL`（**必须 https canonical**——`http://` 会被网关 307 到 https、urllib 跨 scheme 丢 Bearer key → 401；现 = `https://api.example-host.com/v1`，随 Hermes provider 走，**勿写死某台实例 IP**）`/v1/models` 回 200 才放行决策层出数 | 网关不可达/陈旧路由/误写 http:// → 探针 REFUSED、不产数、不写 stamp；换路由 = 换 `LIVE_ENDPOINT_*` 三件套（scheme 一并换对），非代码项 |
| 换模型档位 | `LIVE_ENDPOINT_MODEL=<新模型名> && .venv\Scripts\python.exe bench/decision_layer_probe.py --run 30` | 一行重推，stamp 自动刷新（**别手改 stamp head 消 WARN**） |

**端点面边界注（防误读）：** `sc/live_endpoint.py` 的 `resolve_live_endpoint()` 虽物理住在生产树 `sc/` 里、自己读 `LIVE_ENDPOINT_*`/`OPENAI_*`，但全仓 caller 只有 `bench/decision_layer_probe.py` + 其测试——**没有任何生产调用路径触达它**。prod 端点 = 调用方（Hermes）注入的构造参数（`AuxProvider(main_endpoint)`），零配置跟随默认模型。正确说法不是「`sc/` 里无端点 env 读取」（有），而是「生产路径从不读」——`LIVE_ENDPOINT_URL` 仅活在 bench 出数那一下 shell env，跑完即弃。

**术语致敬注（防误读，file-memory 互补判定 ①②）：**
- `sc/store.py:1` docstring「filememory 式」= 设计措辞致敬（jsonl 单一事实源存储借用 file-memory 的 row/replay/materialize 习惯），**非活依赖**——全仓零共享 schema / 零 import / 零配置耦合，`filememory`（无连字符）源面恰此一处 + 2 个 `.pyc` 二进制副产（`sc/__pycache__/store.cpython-311/312.pyc`）。supercompress（会话内 P0/P1/P2 保真，热层）与 file-memory 插件（跨会话 L0/L1/L2 沉淀，冷层）为「热层丢给冷层」互补两层，不重叠、不合并。
- `docs/supercompress-mechanism.html:43-44` grep 到的 2 条 `L0` 是 SVG marker 坐标（`M0,0 L0,6 L8,3`），**非 tier（L0/L1/L2）引用**；代码面 `grep \bL0\b|\bL1\b|\bL2\b` 在 `sc/`+`scdata.py` 零命中，supercompress 无 L 级概念。

**TCM 退役删除门（切片③，@devops own，物理删 commit 的验收锚）—【已执行 @ user「清」，2026-09-26】：** file-memory 插件把 TCM 晋升管线（`TCMPromotionEngine`）退役、让位给 Gold 知识层。执行序：门① pause（`hermes -p coder cron pause 92ae706bd44e`，20:43:26）→ @coder 落 5 模式删 commit（代码半，self-verifying transform）→ 门② remove（`hermes -p coder cron remove 92ae706bd44e`，pre: jobs=['92ae706bd44e'] / post: jobs=[]，`hermes cron list`="No scheduled jobs"）→ @reviewer 双 receipt（5 模式全 0 + cron 条目 gone）齐签 = 全链收口。各子门状态（验收锚，复跑用）：
- **cron job `92ae706bd44e`**（`profiles/coder/cron/jobs.json`，no_agent、240m、`file-memory-consolidation` → wrapper `filememory_consolidation.py` → `run.py --incremental`）：活写消费者，写 `pending_upgrades.jsonl`。删除 commit 前**先 `hermes -p coder cron pause 92ae706bd44e`**（用 hermes CLI 改活 scheduler，**不手改 live `jobs.json`**——有 active `.jobs.lock` + ticker，手改竞态）；flag 周期证零调用后，物理删 commit 再 `hermes -p coder cron remove 92ae706bd44e` 整条带走。
- **代码 5 模式 grep 门（删完全 0 命中，含 CamelCase）：** `TCMPromotionEngine`（`__init__.py:364` 类 + call form）｜`promotion_engine`（`:1239` 热循环 + `_tool_check_promotion` 内 + `:942` 构造 `self.promotion_engine = TCMPromotionEngine()`）｜`check_promotion`（工具 def + dispatch + `_tool_check_promotion`）｜`promote_to_tcm`（`L2` config knob，随引擎同 commit 删、不留 inert config）｜`user_requested_promotion`（规则条件 + review_drafts promote 标记）。**行号 build 半后已上移，本门是字符串匹配非行匹配，删除后 `grep -cE "5 模式" = 0` 即验收，行号仅作定位锚。**
- **Gold 层隔离（删 commit 天然安全区，@coder 坐实 OVERLAP=0，@devops 复核）：** 5 模式全仓 grep 的命中行与 Gold 符号（`upsert_knowledge`/`_load_knowledge_pool`/`_memory_id_set`/`_quarantine_raw_pending` + `:2049/:2103` tier-boost）逐行不相交——删 commit 跑 grep 时**不会误伤 build 半新代码**；`run.py:42,75,187` 三条 `provider.promotion_engine.evaluate`（`--full`/incremental/兜底腿）一并带。@reviewer 核签"5 模式全 0"时，Gold 层 6 方法 + 2 处 boost 引用是放行区，不跟 TCM 面共行。
- **`consolidation/run.py:21` 的 `from __init__ import … TCMPromotionEngine` import + `:90` 实例化**（M6 巡检入口，cron 独立调用腿；`--full` 是读全 4 类 memory 逐条 evaluate 的独立宿主，删除腿一并带）。
- **删除注记跟 commit 走（回滚锚）：** 插件 `plugins/file-memory/` 与 `profiles/` **都不是 git repo**——"删除即落盘、无回滚锚"。唯一可回滚仓是本 supercompress 仓，故 TCM 删除的根因（`pg_` 前缀 pending key 从未对齐 memory id 形制、`upgraded_from` 才是真 provenance 终点）+ 上面 5 模式 grep 门 + cron job 移除，全部写在本注记里，跟物理删 commit 同仓同 commit，删错时靠这条找锚。
- **build 半（Gold 知识层）canonical 验收 recipe（@devops 双跑裁决，@coder 的 Hermes-venv 写跑不通）：**
  ```
  cd C:/Users/user/.hermes/plugins/file-memory
  PYTHONPATH="C:/Users/user/AppData/Local/hermes/hermes-agent;C:/Users/user/.hermes/plugins/file-memory" \
    /c/Users/user/supercompress/.venv/Scripts/python.exe -m pytest tests/test_gold_knowledge_red.py -v
  ```
  （双路 PYTHONPATH = hermes-agent 源目录供 `from agent.memory_provider import …` + 插件目录；解释器用真正带 pytest 的那个。全插件套件现 **23/23**（build 半 26 → 删 TCM 半随行去掉 3 条 M6 测试，ref-count 4 + Gold 4 + 存量零回归）。）
- **ops 前向脆弱性（登记不建）：** 上面 proof 的解释器 = 兄弟工程 `supercompress/.venv`，file-memory 插件自身无 venv；该 `.venv` 一旦被盘清删掉，flag 周期"零调用"证明就无解释器。触发条件 = 切片③ 证明卡在这缺解释器时，才起专用 env / 装 pytest 进插件自身 env，现不预建（无消费者不建）。

## 6. 未落项（如实标注，不进 ship 口径）

- 归档/retention **路径与策略是部署约定，不是 prod 代码**——超压缩仓库本身不替你建 `~/.hermes/sessions/`，接入方（Hermes）落地时按 §2 约束实现。
- 「甩掉 numpy、自写纯 Python BM25」为可选 v2（footprint 再减一项 native wheel），默认不建（无消费者不建）。
