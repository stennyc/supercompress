# AGENT-INSTALL — 给目标 agent 的安装/配置提示词（v2.0.1+ · cefb752 · 发现根勘误）

> 用法：把下方「任务提示词」整段发给目标机器上的 agent（或人），按其 5 步执行 + 红线收口。
> 所有路径/行为以宿主实码为准（`plugins/context_engine/__init__.py` 发现链 + `install.sh` 第 5/6 步）。
> 勘误（2026-11-14）：发现根只有两条"直接子目录"（bundled / user），嵌套
> `plugins/context_engine/<name>` 是死布局——loader 绝不 descend，引擎不会出现在
> dashboard 下拉菜单且静默回退内置 compressor；install 第 6 步起自动注册正确布局。

## 任务提示词（复制即用）

```
## 任务：为本机 Hermes 安装并配置 supercompress 上下文引擎（stennyc/supercompress）

背景：supercompress 是我方自研 compact provider（Hermes context engine 插件，GitHub: stennyc/supercompress，
public，默认分支 master，v2.0.1+）。它不是 pip 包，是源码树插件——宿主按目录发现 + config.yaml 点名加载。

### 第 1 步：定位 Hermes 宿主仓
  - 宿主仓 = 含 hermes_cli/ 与 plugins/ 的 git 目录。常见位置 /usr/local/lib/hermes-agent，
    用 `git rev-parse --show-toplevel` 在 hermes_cli 所在目录确认。
  - 记 <HERMES_LIB> = 宿主仓根。

### 第 2 步：clone 到宿主（两种合法发现根，择一；自动更新宿主优先 user 根）
  cd <HERMES_LIB>/plugins/context_engine/          # 根 A（bundled）
  git clone https://github.com/stennyc/supercompress.git
  cd supercompress
  已存在则：git fetch origin && git checkout master && git pull（勿 force）
  或 user 根形态（根 B，自动更新宿主首选）：
    把 repo 放到（或 clone 到）$HERMES_HOME/plugins/supercompress/（直接子目录），
    多 profile：每个 profile 的 home 各放一份直接子 link（install 第 6 步自动注册）。

  ⚠ 发现链（宿主实码 plugins/context_engine/__init__.py::_iter_engine_dirs +
     plugin_loader.iter_plugin_dirs）——只有两条合法根，且都是"直接子目录"语义：
       根 A（bundled）= <HERMES_LIB>/plugins/context_engine/<name>/
       根 B（user）   = $HERMES_HOME/plugins/<name>/
       bundled 碰撞时优先。本插件 __init__.py 含 register_context_engine 字样，
       过 _is_context_engine_dir 启发式，两条路都能发现。
     死布局（绝不放这里）：$HERMES_HOME/plugins/context_engine/<name>/ —— 多套一层
     context_engine/ 包裹。iter_plugin_dirs 只列直接子目录、绝不 descend，嵌套层
     对发现链完全不可见 → dashboard plugins 界面下拉菜单无该选项 + 运行时静默
     回退内置 compressor（无任何报错；2026-11-14 Windows host 事故根因）。
     风险面：根 A 是宿主 git 仓里的 untracked 目录时，`hermes update` 的 autostash
     会把它清掉（残留空壳 → "loaded but no engine instance found" 或整段消失）
     → 会自动更新的宿主（桌面宿主 / 定期 hermes update 的 CLI 宿主）优先用根 B。
     选 bundled 与现网部署一致的前提是该宿主不自动更新；否则 user 根。

### 第 3 步：运行安装脚本（自动配齐 config.yaml + 注册可发现布局）
  bash install.sh          # Windows: powershell -ExecutionPolicy Bypass -File install.ps1
  多 profile / multiplex gateway：bash install.sh --all-profiles
                            （Windows: -AllProfiles；逐个配齐 profiles/*/config.yaml，已有不覆盖）
  脚本 4 步：探测 Python 3.9+ → 建 .venv → 装 pytest → 跑全量测试（验证代码面）
  第 5 步（cefb752+ 起）自动定位 Hermes config.yaml 并配齐 26 键默认值：
    - 定位序：$HERMES_HOME（已设=profile 数据根锚定，缺失也写这里）
              > ~/.hermes/profiles/$HERMES_ACTIVE_PROFILE/config.yaml
              > ~/.hermes/config.yaml（bundled 形态）
    - config 缺失 → 写 config.default.yaml 完整块
    - config 已有 → 不覆盖，仅逐键报告缺失项（0 写入，安全）
  第 6 步（发现根勘误起）自动注册可发现布局：对每个目标 home（--all-profiles 时 =
    数据根 + 每个 profiles/<p>）保证 直接子目录 plugins/supercompress 可发现
    （symlink → junction → 复制交付面，逐级降级；已存在绝不覆盖），并侦测嵌套死布局
    plugins/context_engine/（WARN 提示，可安全删除）——杜绝"引擎不在下拉菜单"事故。
    平台细则：POSIX 用 ln -s 后必须 reparse 验证（test -L）；Windows 首选 junction
    （目录 reparse，无需特权；SymbolicLink 需 Developer Mode/管理员）。
    陷阱（2026-11-14 实测）：MSYS/GNU 的 ln -s 在 Windows 上可能 rc=0 却静默降级为
    "整目录复制"（连 .git 一起复制 = 代码分叉，升级不跟随）——脚本已做 reparse
    验证（test -L / fsutil reparsepoint query）并清掉假目录；自建链接后务必验证
    （PowerShell: (Get-Item <path>).LinkType ∈ {Junction, SymbolicLink}）。

### 多 profile / multiplex mode 面（必读）
  - 代码面（bundled 形态）：一份 clone 服务所有 profile（宿主发现链全局共享），clone 只做一次。
  - 代码面（user 根形态，自动更新宿主）：每个 profile 的 home（profiles/<p>/）各需一份
    直接子目录 plugins/supercompress（symlink 到同一份代码根即可，单一代码源）。
    第 6 步 --all-profiles 自动注册全部 profile；嵌套 plugins/context_engine/<name> 是
    死布局（见第 2 步 ⚠），见到即 WARN，可安全删除。
  - 配置面：multiplex gateway 每个 profile turn 把 HERMES_HOME 切到该 profile 目录，
    load_config() 读"当前 profile 自己的 config.yaml"（宿主实码 hermes_constants ContextVar 链）。
    各 profile 要走 supercompress，各自的 config.yaml 都要有 context.engine +
    compression.supercompress → 一次 --all-profiles 配齐全部。
  - 数据面：supercompress_store/ 随 HERMES_HOME per-profile 落盘，天然隔离，零额外配置。
  - 生效面：改完各 profile config.yaml 后，重启 gateway（或新开 session）才加载新引擎；
    已存在的旧 session 不回溯切换。

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
  4) 发现面（离线，不重启）：宿主仓根入 sys.path + HERMES_HOME=<目标 home>，
     python -c 'from plugins.context_engine import discover_context_engines; print(discover_context_engines())'
     期望出现 ('supercompress', '<desc>', True)。
     注意依赖面：宿主若经 launcher 注入 venv generation（如桌面宿主的 pm-runtime），
     探针须把该 venv 的 site-packages 一并挂上 sys.path——宿主 hermes_yaml 硬依赖
     ruamel.yaml，裸解释器缺它会误报 engine unavailable（与发现根无关的另一条坑）。
  5) 生效面：重启 gateway 后跑一段对话到触发 compact，
     grep ~/.hermes/logs/agent.log 'supercompress' 看 marker（turn cache hit / select_context 等），
     并查 <HERMES_HOME>/supercompress_store/ 是否生成 <sid>.jsonl + <sid>.compact.json
     （多 profile：每个 profile 查各自目录；旧 session 不回溯，验证要开新 session）
     dashboard 面：plugins 界面 context engine 下拉菜单应有 supercompress 选项
     （hub 现扫目录 + 5s memo TTL，目录就位后刷新即现，无需重启）

### 红线
  - 宿主 agent/ 目录零 diff（改动只在 supercompress 壳内）
  - 不 commit/push（本地安装面，签核另走 stennyc/supercompress 线）
  - 已有 config.yaml 永不覆盖（脚本行为已封死，勿手改）
```

## 维护注

- 新增配置键 → 改 `config.default.yaml`（单一事实源），install 脚本自动跟上，本文档「第 4 步」不用动（键表指向 config.default.yaml 而非内联复制）。
- 宿主发现链若改（bundled/user 优先级、`_is_context_engine_dir` 启发式）→ 同步本文「第 2 步 ⚠」段。
- tag 注意：`v2.0.1` 指 `d8fecf1`（不含 install 第 5 步）；含自动配置面的是 `cefb752`+，clone 默认走 master 即可。
