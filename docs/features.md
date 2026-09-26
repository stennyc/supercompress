# supercompress — 功能总览（features.md 装配页）

> 可 ship 交付物的单页功能说明。`C:\Users\user\supercompress`。
> 本装配页引用 5 个源段（全部 tracked，装配后 worktree 应 clean）：
>
> | 源段 | 视角 | owner |
> |---|---|---|
> | [`docs/product.md`](product.md) | 产品：定位 / KPI / 用户文案 / 口径注脚 | @product-manager |
> | [`docs/implementation.md`](implementation.md) | 实现层：4 机制 block + 端点面 + ③ 守卫 + 故障注入 | @coder |
> | [`docs/ops-deployment.md`](ops-deployment.md) | 部署 / 运行面：足迹 / 路径 / IO / 崩溃边界 / ops 入口 | @devops |
> | [`docs/design-rationale.md`](design-rationale.md) | 设计依据：WHY + 证据工件 + 实测数口径边界 | @researcher |
>
> 所有实测数字锚定 commit（非工作树），全量 **58/58 绿**，全链 commit-锚定、零 working-tree 漂移。

## 定位（一句话）

纯**蒸馏器 + 锚点发射器** + 一个 **session 级 jsonl 持久骨干**——不是内容仓库。
压缩管窗口密度、记忆管持久，两层不混。兑现两个诉求：**防跑偏**（目标钉窗内 + 活账本落账）
＋ **防失忆**（关掉/崩了/断电重开，session 记忆还在）。

## 五个模块

| 模块 | 一句话 | 详见 |
|---|---|---|
| ① `sc/distill.py` 蒸馏器 | tiering → P0 write-before-compress（即写即 fsync+读回过了才发锚点）→ P1/P2 lazy 批 | implementation.md |
| ② `sc/recall.py` 检索 | 点查 O(1) + 纯 BM25（P0 boost）+ Pull 预算 20%；**无 embedding、无向量库** | implementation.md |
| ③ `sc/ledger.py` 账本 | 新 user 指令 supersede 旧 guardrail/task；双层（goal 永不裁 / 子目标可 supersede） | implementation.md |
| ④ `sc/tagger.py` 标记 | P0 结构地板 + 外部引用跨期 stale 标 | implementation.md |
| 数据层 jsonl 骨干 | 单一事实源、append-only、`content_hash=sha256(NFC)` 幂等去重、torn-tail 单 replay、跨期重开同一条重放 | ops-deployment.md §3-4 |

## 关键承诺（用户可读）

- **关掉 agent / 主模型崩 / 断电 → session 记忆还在**（进程级 ≈0，重开回填活账本 + P0）
- **断电** = 最多丢一批 P1/P2 可重跑输出（上界 256 KB）；durable 的 P0 目标/决策/护栏不丢
- **跨期重开 = 恢复你的计划，不恢复你的世界**（外部引用跨期标 stale、行动前 verify）
- **防跑偏** = 目标恒可见 ＋ 每 turn 告诉你偏没偏（遥测）——**不承诺「系统替你纠正」**

## KPI v1（实测锚）

| # | KPI | 阈值 | 现状 |
|---|---|---|---|
| 1 | P0 0-loss + tagging-recall | 硬不变量 + ≥95% 软 | 已证 @ evicted |
| 2 | 信号密度 | 滚动窗 + ≥90% 下限 | 已 test-map |
| 3 | 决策层自一致性 recall @ evicted | ≥75% | **同模型三发抽样：87%（26/30）@ `119374a`、93%（28/30）@ `ba1ae50`、80%（24/30）@ `640399b`**，全 ≥75% gate PASS（单发参考点；对外写 **~80–93% band**，见下方注脚 1） |
| 4 | 召回预算（Pull 20%/Push 15%） | breach 频率 + 单峰 | 已证 |
| 5 | 持久性（断电上界） | 256 KB | 8/8 + power-loss 三注入 |

## 口径注脚（对外 claim 必带）

1. **决策层数 = 参考 main-model 配置 + 单题型合成 fixture 的抽样点**——证「该查模式下模型知道查 + BM25 捞得到」，
   不是全题型空间的任何单点数；喂真实 session 才量实际用法（下一步研究项）。
   **抽样 band 纪律（@researcher 观察）**：同模型三发独立抽样 80%（@ `640399b` 旧路由时代）/
   93%（@ `ba1ae50`）/ 87%（@ `119374a`，key 放行后 https canonical），波动 13pp → 单发数非稳定属性、是参考点，
   对外 claim 写 **~80–93% band**、不写任何单点；稳定数 = 研究 follow-up（多次抽样中位数 / n≥50），非 v1 门。
2. **数 model-scoped**：只对 `main == digest 生成者`（现档 = `LIVE_ENDPOINT_*` 路由 + agnes-3.0-flash，路由随 Hermes provider 走、**不锚实例 IP**；当前实例 = `api.example-host.com/v1`）有效，
   换档 = 一行 `--run` 机械重推（stamp 双锚 + ③ advisory 兜住）；**禁止手改 stamp head 消 WARN**。

## footprint（精确口径）

零向量库 · 零外部服务 · 无常驻进程；依赖 = `rank_bm25 + numpy`（pip wheel、免编译）。
**非「纯 Python 零依赖」。**

## 设计依据（WHY，非拍脑袋）

每个关键决定带文献背书（MemGPT / Generative Agents 分层记忆、Self-RAG 决策层瓶颈、
Lost-in-the-Middle U 形注意力、BM25-vs-embedding 规模论证）——详见
[`design-rationale.md`](design-rationale.md)；证据工件全 commit 锚定、防编数守卫落码。

## ship 入口（机械五条）

见 [`ops-deployment.md`](ops-deployment.md) §5：全量 `pytest tests/` 58/58 @ HEAD / 断电半边 /
③ stamp advisory 随套件出三态 / 网关 `/v1/models` 200 才放行决策层出数 / 换档一行 `--run` 重推。
