# 产品节（定位 + KPI + 用户文案 + 口径注脚，@product-manager own）

> 装配页 `docs/features.md` 的源段之一。实现面见 `docs/implementation.md`，
> 部署面见 `docs/ops-deployment.md`，设计依据见 `docs/design-rationale.md`。
> 本文只讲「它是什么、承诺什么、KPI 怎么量、对外话术红线」。最后数据点锚 `9dead01`（3 发 n=30，中位数 93%，@ `bb1f41f` 落仓；全量 60/60 绿）。

## 1. 一句话定位（产品层）

supercompress = **上下文蒸馏器 + session 级记忆骨干**，不是内容仓库。
压缩管窗口密度、记忆管持久，两层不混。它兑现两个原始诉求：

- **防跑偏**：目标 + 活账本每阶段落账、常驻窗内（锚点永不裁）；
- **防失忆**：关掉 / 崩了 / 断电重开，session 记忆还在。

## 2. 能力 → 用户承诺（定稿文案，可直接进首屏/文档）

| 能力 | 对用户说的 |
|---|---|
| 关掉 agent / 主模型崩 / 断电 | **session 记忆还在**（进程级 ≈0 丢失，重开即回填活账本 + P0） |
| 断电 | 最多丢一批 P1/P2 可重跑输出（上界 **256 KB**，默认可配）；已 durable 的 P0 目标/决策/护栏不丢（毫秒级 fsync 瞬间除外） |
| 跨期重开 | **恢复你的计划，不恢复你的世界**（外部引用跨期标 stale、行动前 verify） |
| 防跑偏 | 目标恒可见（锚点永不裁）＋ 每 turn 告诉你偏没偏（遥测）——**不承诺「系统替你纠正」** |

**措辞红线**：不写「80% 必压」（80% 是软目标，硬 backpressure 只在物理窗口满触发）；
80% 数字对外必须带定语（见 §4 注脚 1）。

## 3. KPI v1（单 regime + transient_fail，(A) 终形态）

| # | KPI | 阈值 | 现状（实测锚） |
|---|---|---|---|
| 1 | P0 0-loss + P0 tagging-recall | 硬不变量 + ≥95%（软） | 已证 @ evicted 态 |
| 2 | 信号密度（事实数/1K + 锚点命中率） | 滚动窗 + 绝对下限 ≥90% | 已 test-map |
| 3 | 决策层自一致性 recall @ evicted | ≥75% | **稳定数 = 3 发中位数 93%（93/93/80，各 n=30 @ `9dead01`，@ `bb1f41f`，参考带中心非 CI 硬 claim）；全量 band 留档 80%@`640399b` / 93%@`ba1ae50` / 87%@`119374a` / 93%/93%/80%@`9dead01`**，全 ≥75% gate PASS（对外写 band + 中位数，不写单点，见 §4 注脚 1） |
| 4 | 召回预算（Pull 20% / Push 15%，P0 豁免记 breach） | breach 频率 + 单峰 | 已证（recall@3 顶回 evicted P0 + breach 日志正确） |
| 5 | 持久性（进程≈0 / 断电≤256KB+in-flight） | 上界 256 KB（默认可配）+ ms fsync 窗口 | 8/8 @ HEAD + power-loss 三注入点 |

## 4. 两条对外注脚（签字，@researcher 口径我认）

1. **决策层数 = 「参考 main-model 配置 + 单题型合成 fixture」的抽样点**——它证的是
   「该查模式下模型知道查 + BM25 捞得到」，**不是全题型空间的任何单点数**。
   喂真实长 dev session 才量实际用法（下一步研究项，见 `design-rationale.md` §4）。
   对外 claim 必须带「参考配置 + 单题型」定语。
   **抽样 band 纪律（@researcher 观察）**：同模型多轮独立抽样 80%（@ `640399b`，旧路由时代）/
   93%（@ `ba1ae50`）/ 87%（@ `119374a`）/ 93%,93%,80%（3 发 @ `9dead01`）波动 13pp，
   实证**单发数不是稳定属性、是参考点**——**稳定数 = 3 发中位数 93%（参考带中心，非 CI 硬 claim）**，
   对外 claim 框成 **~80–93% band + 中位数 93%**，不写「可复现的单点 93%」；
   真扛审稳定数 = 单发 n≥50（v1 follow-up 研究项，非 v1 门）。
2. **数 model-scoped**：只对 `main == digest 生成者`（现档 = `LIVE_ENDPOINT_*` 路由 + agnes-3.0-flash，
   路由随 Hermes provider 走、**不锚实例 IP**；当前实例 = `api.example-host.com/v1`，换路由 = 换 `LIVE_ENDPOINT_*` 三件套）有效。
   换档 = 一行 `--run` 机械重推（stamp 双锚 + ③ advisory 兜住，过期数引用时可见、不静默失效）。

## 5. footprint（产品可说的版本）

零向量库 · 零外部服务 · 无常驻进程；依赖 = `rank_bm25 + numpy`（pip wheel、免编译，
精确口径见 `ops-deployment.md` §1）——**不是「纯 Python 零依赖」**，这句别用。

## 6. open 项（均不 block ship）

- **N≥30 已出 + 3 发中位数 follow-up 已出**（@researcher 收掉，`bb1f41f`）→ 唯一数值口径注脚 = §4 两条。
- **@devops 的 nit 已收口**：`_stamp_state` 拆独立模块 `bench/stamp_advisory.py`
  （`@devops` lane，commit `ee128a2`，纯 stdlib 无 fakes/sc 依赖，全树该函数唯一 live 定义；
  非 pytest 单点查路径已通）。非阻塞，已落地。

> 注：③ advisory 现 `stamp.head=9dead01 < HEAD`（`bb1f41f` 之上的 docs-only commit）→ 下次 preflight 读
> **WARN 属 by-design**。⚠️ **读 stamp 别误判现行数**：磁盘 stamp 记的是**最后一发成功 `--run`**（RUN3 = 80%，pass:24），
> **不是 3 发中位数 93%**（中位数只在 `bb1f41f` commit message + 本文 §3/§4）——stamp = 最后一发落点、非稳定数，
> 防误读那条纪律照旧：别手改 stamp head，换档/换路由一行 `--run` 自然刷新。
