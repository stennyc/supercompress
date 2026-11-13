# AGENT-INSTALL — 给目标 agent 的安装/配置提示词（v2.0.1+ · cefb752）

> 用法：把下方「任务提示词」整段发给目标机器上的 agent（或人），按其 5 步执行 + 红线收口。
> 所有路径/行为以宿主实码为准（`plugins/context_engine/__init__.py` 发现链 + `install.sh` 第 5 步）。

## 任务提示词（复制即用）

```
## 任务：为本机 Hermes 安装并配置 supercompress 上下文引擎（stennyc/supercompress）

背景：supercompress 是我方自研 compact provider（Hermes context engine 插件，GitHub: stennyc/supercompress，
public，默认分支 master，v2.0.1+）。它不是 pip 包，是源码树插件——宿主按目录发现 + config.yaml 点名加载。

### 第 1 步：定位 Hermes 宿主仓
  - 宿主仓 = 含 hermes_cli/ 与 plugins/ 的 git 目录。常见位置 /usr/local/lib/hermes-agent，
    用 `git rev-parse --show-toplevel` 在 hermes_cli 所在目录确认。
  - 记 <HERMES_LIB> = 宿主仓根。

### 第 2 步：clone 到宿主 bundled 位置（全局共享，一份代码服务所有 profile）
  cd <HERMES_LIB>/plugins/context_engine/
  git clone https://github.com/stennyc/supercompress.git
  cd supercompress
  已存在则：git fetch origin && git checkout master && git pull（勿 force）

  ⚠ 发现链（宿主实码）：context engine 先扫 bundled <HERMES_LIB>/plugins/context_engine/<name>/，
     再扫 user $HERMES_HOME/plugins/<name>/，bundled 优先。本插件 __init__.py 含
     register_context_engine 字样，过 _is_context_engine_dir 启发式，两条路都能发现；
     选 bundled 与现网部署一致。

### 第 3 步：运行安装脚本（自动配齐 config.yaml）
  bash install.sh    # Windows: powershell -ExecutionPolicy Bypass -File install.ps1
  脚本 4 步：探测 Python 3.9+ → 建 .venv → 装 pytest → 跑全量测试（验证代码面）
  第 5 步（cefb752+ 起）自动定位 Hermes config.yaml 并配齐 26 键默认值：
    - 定位序：$HERMES_HOME（已设=profile 数据根锚定，缺失也写这里）
              > ~/.hermes/profiles/$HERMES_ACTIVE_PROFILE/config.yaml
              > ~/.hermes/config.yaml（bundled 形态）
    - config 缺失 → 写 config.default.yaml 完整块
    - config 已有 → 不覆盖，仅逐键报告缺失项（0 写入，安全）

### 第 4 步：调参（按需，非必须）
  所有键缺失 = 走 engine 内置默认（config.default.yaml 列出值即默认本身）。
  最常调的三个：
    compression.supercompress.context_length  ← 按你的模型窗口（默认 256000）
    compression.supercompress.compact_ratio   ← 压缩触发水位（默认 0.5）
    compression.supercompress.auto_recall     ← 保持 false（A1 止血态，勿开）
  完整键表：见 supercompress/config.default.yaml（单一事实源，带中文注释）。

### 第 5 步：验证（盘上真值，不信"装好了"自报）
  1) 代码面：.venv/bin/python -c "from sc import Segment, JsonlStore; print('OK')"
     （或 PYTHONPATH=<clone 目录> 直跑）
  2) 测试面：.venv/bin/python -m pytest -q（全绿）
  3) 配置面：cat 第 3 步定位到的 config.yaml，确认 context: engine: supercompress 在场
  4) 生效面：重启 gateway 后跑一段对话到触发 compact，
     grep ~/.hermes/logs/agent.log 'supercompress' 看 marker（turn cache hit / select_context 等），
     并查 <HERMES_HOME>/supercompress_store/ 是否生成 <sid>.jsonl + <sid>.compact.json

### 红线
  - 宿主 agent/ 目录零 diff（改动只在 supercompress 壳内）
  - 不 commit/push（本地安装面，签核另走 stennyc/supercompress 线）
  - 已有 config.yaml 永不覆盖（脚本行为已封死，勿手改）
```

## 维护注

- 新增配置键 → 改 `config.default.yaml`（单一事实源），install 脚本自动跟上，本文档「第 4 步」不用动（键表指向 config.default.yaml 而非内联复制）。
- 宿主发现链若改（bundled/user 优先级、`_is_context_engine_dir` 启发式）→ 同步本文「第 2 步 ⚠」段。
- tag 注意：`v2.0.1` 指 `d8fecf1`（不含 install 第 5 步）；含自动配置面的是 `cefb752`+，clone 默认走 master 即可。
