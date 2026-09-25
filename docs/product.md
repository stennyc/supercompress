# 产品节（定位 + KPI + 用户文案 + 口径注脚，@product-manager own）

> 装配页 `docs/features.md` 的源段之一。实现面见 `docs/implementation.md`，
> 部署面见 `docs/ops-deployment.md`，设计依据见 `docs/design-rationale.md`。
> 本文只讲「它是什么、承诺什么、KPI 怎么量、对外话术红线」。锚定 @ `0dde91b`。

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
| 3 | 决策层自一致性 recall @ evicted | ≥75% | **n=30 = 80% PASS（带 margin）@ `640399b` / agnes-3.0-flash** |
| 4 | 召回预算（Pull 20% / Push 15%，P0 豁免记 breach） | breach 频率 + 单峰 | 已证（recall@3 顶回 evicted P0 + breach 日志正确） |
| 5 | 持久性（进程≈0 / 断电≤256KB+in-flight） | 上界 256 KB（默认可配）+ ms fsync 窗口 | 8/8 @ HEAD + power-loss 三注入点 |

## 4. 两条对外注脚（签字，@researcher 口径我认）

1. **80% 是「参考 main-model 配置 + 单题型合成 fixture」的数**——它证的是
   「该查模式下模型知道查 + BM25 捞得到」，**不是全题型空间的 80%**。
   喂真实长 dev session 才量实际用法（下一步研究项，见 `design-rationale.md` §4）。
   对外 claim 必须带「参考配置 + 单题型」定语。
2. **数 model-scoped**：只对 `main == digest 生成者`（现 Sub2API agnes-3.0-flash 这档）有效。
   换档 = 一行 `--run` 机械重推（stamp 双锚 + ③ advisory 兜住，过期数引用时可见、不静默失效）。

## 5. footprint（产品可说的版本）

零向量库 · 零外部服务 · 无常驻进程；依赖 = `rank_bm25 + numpy`（pip wheel、免编译，
精确口径见 `ops-deployment.md` §1）——**不是「纯 Python 零依赖」**，这句别用。

## 6. 两个 open（都不 block ship）

- **N≥30 已出**（@researcher 收掉）→ 唯一数值口径注脚 = §4 两条。
- **@devops 的 nit**：`_stamp_state` 拆独立模块供非 pytest 单点查 stamp（~5 行，
  @user 要再点头才做，非 ship 阻塞）。
