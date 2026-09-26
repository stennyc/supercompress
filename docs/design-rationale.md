# 设计依据（研究 lane：WHY + 证据工件 + 口径注脚）

> 装配页 `docs/features.md` 的源段之一。实现面见 `docs/implementation.md`，
> 部署面见 `docs/ops-deployment.md`，KPI/用户文案见产品节。本文只讲
> 「每个关键决定为什么成立」+「哪些数是实测的、口径边界在哪」。
> 全部锚定 @ `bb1f41f`（可 ship 码态，中位数 follow-up 已出）；文献为设计期输入，非运行依赖。

## 1. 关键决定与依据（不是拍脑袋）

| 决定 | 依据 | 文献 / 先例 |
|---|---|---|
| 两层分界：蒸馏管窗口密度、记忆管持久 | 混合 = 职责混淆（存取/召回归属记忆层，压缩只管 distill） | MemGPT / Generative Agents 的分层记忆先例 |
| 检索 = 纯 BM25 + P0 boost，不上 embedding/qdrant | session 级语料 = MB 量级、高信号 P0 事实库——此规模词法检索够用；embedding 引入 2 个新依赖 + 一套向量基建，可测召回收益近零（为博客上 K8s 型过度设计） | BM25-vs-embedding 对比（小规模高信号语料词法持平或更优） |
| 决策层探针是一等公民，不只报 recall@k | 瓶颈在「模型知不知道该查」，不在「检索准不准」；只测检索层会得到「检索很准但线上模型不召回」的假结论 | Self-RAG：recalling 的决策质量是 RAG 效果主因 |
| P0 反填回窗口（digest 进高注意力槽） | 长上下文 U 形注意力：窗口中段信息利用率最低，关键事实放中段 = 静默丢失 | Lost in the Middle |
| semantic-gap 遥测 >15% 才触发 v2 语义臂，默认关 | 防「静默漂去上 embedding」：先量出真实 gap，有数再升级 | — |
| P0 0-loss 为硬不变量、tagging-recall 为软指标（95%） | 两种故障 owner 不同（tagging bug vs compression bug），混在一个 95% 里会放过真 P0 丢失 | KPI #2 拆分（质量门判定） |

## 2. 证据工件（全部 commit 锚定、可复现、防编数）

- `bench/decision_layer_probe.py` — 决策层探针，三守卫机械落码：
  缺活端点 / 缺真模型名（占位 `"main"` 拒出）/ 网关不可达 → **REFUSED，绝不产出数**；
  隐藏不变量（`TARGET_FACT` 漏进 digest-head 视图 → arm INVALID 拒数）；
  成功 `--run` 尾部自动写 `bench/.last_75_stamp`（双锚 head+model，与数同 commit/链），
  未产出数的路径永不写 stamp。
- `bench/recall_bench.py` — 检索层 recall@3 evicted 臂。
- `bench/.last_75_stamp` — 现值 `{"head":"9dead01","model":"agnes-3.0-flash","n":30,"pass":24}`（`bb1f41f` 中位数 follow-up 的**最后一发 RUN3** 自动写入，随数同 commit）。**注意：磁盘 stamp = 最后一发（RUN3 的 80%，pass:24），不是中位数 93%**——中位数只在 commit message + 本 §3，防后人拿 stamp 当现行数误判。历史单发 93%@`ba1ae50` / 87%@`119374a` / 80%@`640399b` 以 commit 锚留档，非删改。

## 3. 实测数与口径边界（对外 claim 必须带定语）

| 数 | 锚 | 口径边界（诚实注脚） |
|---|---|---|
| **稳定数 = 3 发中位数 93%**（93/93/80，各 n=30）｜参考带中心，**非 CI 硬 claim** | `bb1f41f` @ `9dead01` / agnes-3.0-flash（https canonical） | 真扛审的稳定数仍是**单发 n≥50**（v1 follow-up、非门）；中位数 93% 只是把 3×n=30 的抽样带中心指出来，对外写「~80–93% band + 中位数 93%」，**不写「可复现的 93%」**。 |
| 决策层 n=30 = 87%（26/30 ≥ 75% gate，带 margin）｜band 点 | `119374a` / agnes-3.0-flash（key 放行后 https canonical） | **单一题型合成 fixture**：30 样本同问一个 IP:port 事实。它证「该查模式下模型知道查 + BM25 捞得到」，**不是全题型空间的 87%**。 |
| 决策层 n=30 = 93%（28/30）｜band 点 | `ba1ae50` / agnes-3.0-flash | 同模型多次抽样点之一（80/93/87/93），非「被别的发推翻」——同数多次抽样、非稳定属性。 |
| 决策层 n=30 = 80%（24/30）｜band 点 | `640399b` / agnes-3.0-flash | 留档：Sub2API 旧路由（10.1.1.116:8080）时代的数。 |
| 检索层 recall@3 / 断电半边 | 各 commit 锚 | 确定性量（非活模型随机量），重跑数不变。 |

**model-scoped 注脚**：所有活模型出的数只对 `main == digest 生成者` 那档有效
（现档 = `LIVE_ENDPOINT_*` 路由 + agnes-3.0-flash，**不锚实例 IP**——当前实例仅作说明，
换路由 = 换三件套）。换档 = 一行 `--run` 机械重推，stamp 双锚 + ③ advisory
让过期数「引用时可见」而非静默失效。**禁止手改 stamp head 消 WARN**（= 假归属）。

## 4. 下一步研究项（不进 v1、不 block ship）

1. 多题型 / 真实 session 语料重测决策层（80% → 全题型空间的数）；
2. semantic-gap 遥测数据攒到 >15% 时评估 v2 语义臂（届时才引 embedding，且默认关）；
3. N≥30 已出、3 发中位数 follow-up 已出（`bb1f41f`，93/93/80 → 中位数 93%）；若换 main-model 档位，按 stamp 双锚重推而非外推。
