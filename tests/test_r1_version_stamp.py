"""R1（报告#1 #1）：版本戳 + 脱钩自检 + sync_to_envs 工具（纯插件 + 运维工具，不碰宿主）。

覆盖面（任务 t_945d67a6 单测钉死）：
1. build_id 稳定可复现：同字节不同位置 → 同 build_id（内容 hash 层）；git 目录 →
   短 sha（dirty 带后缀）；产物目录（__pycache__/.git）不参与内容 hash。
2. 脱钩自检触发/不触发：repo 可推导且内容不一致 → 一行
   ``version desync: running=<..> repo=<..>`` WARNING（只警告不阻断，引擎可正常构造）；
   内容一致 → 不触发；repo 不可推导 → 静默跳过（0 WARNING，不崩）。
3. sync dry-run vs apply 幂等性：dry-run 不写盘（目标字节不变、计数正确）；
   apply 真写且两次运行结果一致（第二次 0 改动）；不删目标里非源文件（插件外文件保留）。

跑法（repo 根，宿主/installs 各 venv 解释器）：
  <python with pytest> -m pytest tests/test_r1_version_stamp.py -q
"""
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List

from engine import SupercompressContextEngine
from sc import version
from sc.version import build_id, content_id, desync_warning, repo_root

_ROOT = Path(__file__).resolve().parent.parent


def _mkplugin_files(root: Path) -> None:
    """造一个最小 supercompress 布局（只含 sync 会走到的普通文件层级），供无 git 内容 hash 测试。
    必须含 sc/_bootstrap.py —— repo_root 的 override 指认以此为「有效 repo 源」判据。"""
    (root / "sc").mkdir(parents=True, exist_ok=True)
    (root / "sc" / "a.py").write_text("A = 1\n", encoding="utf-8")
    (root / "sc" / "_bootstrap.py").write_text("# bootstrap\n", encoding="utf-8")
    (root / "engine.py").write_text("X = 2\n", encoding="utf-8")
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "note.md").write_text("# n\n", encoding="utf-8")


def _capture_logs(names, levelno):
    records: List[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    attached = []
    for name in names:
        lg = logging.getLogger(name)
        h = _H(level=levelno)
        lg.addHandler(h)
        attached.append((lg, h))
    return records, attached


def _detach(attached):
    for lg, h in attached:
        lg.removeHandler(h)


# ───────────────────────── build_id 稳定可复现 ─────────────────────────


def test_content_hash_reproducible_across_locations(tmp_path):
    """同字节放在两个不同目录（模拟 repo 副本 vs env 冻结副本）→ 同内容 hash（稳定可复现）。"""
    a, b = tmp_path / "repo_copy", tmp_path / "env_copy"
    _mkplugin_files(a)
    shutil.copytree(a, b)
    assert content_id(a) == content_id(b)
    assert content_id(a).startswith("c-")


def test_content_hash_excludes_artifacts(tmp_path):
    """产物目录（__pycache__/.git/.pytest_cache）不进内容 hash——同代码不同 pyc 产物 = 同 build_id。"""
    a, b = tmp_path / "src", tmp_path / "dst"
    _mkplugin_files(a)
    shutil.copytree(a, b)
    # 两侧撒完全不同的产物字节
    (a / "sc" / "__pycache__").mkdir()
    (a / "sc" / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"\x00different-artifacts-1")
    (b / "sc" / "__pycache__").mkdir()
    (b / "sc" / "__pycache__" / "a.cpython-314.pyc").write_bytes(b"\x00different-artifacts-2")
    (a / ".git").mkdir()
    (a / ".git" / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    (b / ".git").mkdir()
    (b / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    assert content_id(a) == content_id(b)


def test_git_dir_build_id_is_short_sha_dirty_suffixed(tmp_path, monkeypatch):
    """含 .git 且 git 可用 → build_id = 短 sha；working tree 脏（untracked）→ 带 -dirty 后缀。"""
    g = tmp_path / "gitrepo"
    (g / "sc").mkdir(parents=True)
    (g / "sc" / "a.py").write_text("A = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=g, check=True)
    subprocess.run(["git", "-C", str(g), "add", "."], check=True)
    subprocess.run(["git", "-C", str(g), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "-m", "base"], check=True)
    monkeypatch.delenv("SC_REPO_ROOT", raising=False)
    bid_clean = build_id(g)
    assert "-" not in bid_clean and not bid_clean.startswith("c-")  # 短 sha，非内容 hash
    # 脏工作区 → -dirty 后缀
    (g / "sc" / "new_untracked.py").write_text("Z = 9\n", encoding="utf-8")
    assert build_id(g) == bid_clean + "-dirty"


def test_frozen_copy_build_id_is_content_hash(tmp_path, monkeypatch):
    """env 冻结副本（无 .git）→ build_id 恒为 c-内容 hash（不依赖 git 是否装在 env 里）。"""
    c = tmp_path / "frozen"
    _mkplugin_files(c)
    monkeypatch.delenv("SC_REPO_ROOT", raising=False)
    assert build_id(c).startswith("c-")


# ───────────────────────── 脱钩自检 触发/不触发 ─────────────────────────


def test_desync_triggered_on_content_mismatch(tmp_path, monkeypatch):
    """repo 可推导（SC_REPO_ROOT 指认）且运行副本内容 ≠ repo 内容 → 一行 desync WARNING。
    引擎侧脱钩自检的 running 面 = sc.version._PLUGIN_DIR（构造期 build_id 同源），
    monkeypatch 到 tmp running 副本，使全测试不依赖真仓状态。"""
    repo, running = tmp_path / "repo", tmp_path / "running"
    _mkplugin_files(repo)
    shutil.copytree(repo, running)
    (running / "engine.py").write_text("X = 3  # env 冻结的旧代\n", encoding="utf-8")
    monkeypatch.setenv("SC_REPO_ROOT", str(repo))
    monkeypatch.setattr(version, "_PLUGIN_DIR", running)
    records, attached = _capture_logs(["engine"], logging.WARNING)
    try:
        eng = SupercompressContextEngine(context_length=256000,
                                        external_store_path=str(tmp_path))
    finally:
        _detach(attached)
    desync = [r for r in records if "version desync" in r.getMessage()]
    assert len(desync) == 1
    msg = desync[0].getMessage()
    assert "version desync: running=" in msg and " repo=" in msg
    assert build_id(running) in msg  # running 面 = tmp running 副本的 build_id
    assert build_id(repo) in msg
    assert eng.build_id == build_id(running)  # 引擎 build 字段 = running 副本 build_id（稳定可复现）


def test_desync_not_triggered_on_content_match(tmp_path, monkeypatch):
    """运行副本与 repo 内容一致（= 刚 sync 完）→ 不触发（0 条 desync WARNING，引擎正常构造）。"""
    repo = tmp_path / "repo"
    _mkplugin_files(repo)
    monkeypatch.setenv("SC_REPO_ROOT", str(repo))
    # 让 engine 自比：running 面（_PLUGIN_DIR）= repo 指认目录本身 → 内容恒一致
    monkeypatch.setattr(version, "_PLUGIN_DIR", repo)
    records, attached = _capture_logs(["engine"], logging.WARNING)
    try:
        eng = SupercompressContextEngine(context_length=256000,
                                        external_store_path=str(tmp_path))
    finally:
        _detach(attached)
    assert eng.build_id == build_id(repo)
    assert not [r for r in records if "version desync" in r.getMessage()]


def test_desync_silent_when_repo_not_derivable(tmp_path, monkeypatch):
    """repo 不可推导（无 .git、无 SC_REPO_ROOT、无仓根祖先）→ 静默跳过：0 WARNING、不崩、
    build_id 照常给出（fail-open 观测面，宿主 0 改动原则下不能崩车）。"""
    # 从插件树临时摘走 .git：repo_root 上溯在 tmp 布局里天然找不到 → 用 override 指无效目录
    monkeypatch.setenv("SC_REPO_ROOT", str(tmp_path / "does_not_exist"))
    records, attached = _capture_logs(["engine"], logging.WARNING)
    try:
        eng = SupercompressContextEngine(context_length=256000,
                                        external_store_path=str(tmp_path))
    finally:
        _detach(attached)
    assert eng.build_id != "unknown"
    assert not [r for r in records if "version desync" in r.getMessage()]
    # 纯函数面同语义
    assert desync_warning(repo_override=str(tmp_path / "does_not_exist")) is None


def test_repo_root_walks_to_agent_repo_ancestor(tmp_path, monkeypatch):
    """仓根祖先推导：agent_root（含 .git）下 repo 源插件目录 + 其下的 worktree 运行副本
    → 自副本上溯可指认 repo 源（与 _bootstrap.require_install_root 同源层级语义）。"""
    monkeypatch.delenv("SC_REPO_ROOT", raising=False)
    agent_root = tmp_path / "agent"
    repo_plugin = agent_root / "plugins" / "context_engine" / "supercompress"
    (repo_plugin / "sc").mkdir(parents=True)
    (repo_plugin / "sc" / "_bootstrap.py").write_text("# bootstrap\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=agent_root, check=True)

    # "运行副本" = 同 agent_root 树内更深的 worktree 布局（无自身 .git），上溯须指认 repo 源
    worktree_copy = (agent_root / ".worktrees" / "wt1" / "plugins" / "context_engine"
                     / "supercompress")
    shutil.copytree(repo_plugin, worktree_copy)
    monkeypatch.setattr(version, "_PLUGIN_DIR", worktree_copy)
    assert repo_root() == str(repo_plugin)


def test_desync_pure_function_surface(tmp_path):
    """desync_warning 纯函数面（不构造引擎也能测触发/不触发，幂等无副作用）。"""
    repo, running = tmp_path / "r", tmp_path / "run"
    _mkplugin_files(repo)
    shutil.copytree(repo, running)
    _pure_surface_asserts(repo, running)


def _pure_surface_asserts(repo: Path, running: Path) -> None:
    # 一致 → None
    assert desync_warning(running_dir=running, repo_override=str(repo)) is None
    # 内容不同 → 文案面，且只警告不 raise
    (running / "engine.py").write_text("X = 4\n", encoding="utf-8")
    msg = desync_warning(running_dir=running, repo_override=str(repo))
    assert msg and msg.startswith("version desync: running=") and " repo=" in msg


# ───────────────────────── sync dry-run vs apply 幂等性 ─────────────────────────


def _mk_installs(tmp_path: Path, src: Path, drift: bool) -> Path:
    """造 installs/<install-id>/environments/<env-id>/workspace/plugins/.../supercompress 目标
    布局（真实 install 布局的两层层级；发现面兼容直挂）。"""
    installs = tmp_path / "installs"
    dst = (installs / "inst1" / "environments" / "env1" / "workspace"
           / "plugins" / "context_engine" / "supercompress")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*version._EXCLUDED_DIRS, "*.pyc"))
    # 目标里塞一个非源文件（插件外文件 / 旧代文件）：sync 不得删它
    (dst / "stray_not_in_src.txt").write_text("keep me\n", encoding="utf-8")
    if drift:
        # 旧文件覆写场景：目标里一份旧字节
        (dst / "engine.py").write_text("OLD=1\n", encoding="utf-8")
        # 新增文件带场景：目标里缺一份源有的文件（删掉再跑）
        missing = dst / "docs" / "note.md"
        if missing.is_file():
            missing.unlink()
    return installs


def test_sync_dry_run_writes_nothing(tmp_path, monkeypatch):
    """dry-run（默认）：目标字节 0 改动（缺文件不补、旧文件不覆），但正确报出将做的改动计数。"""
    src = tmp_path / "src"
    _mkplugin_files(src)
    installs = _mk_installs(tmp_path, src, drift=True)
    monkeypatch.setenv("SC_INSTALLS_ROOT", str(installs))
    import sc.sync_to_envs as syncmod
    monkeypatch.setattr(syncmod, "_PLUGIN_ROOT", str(src))
    monkeypatch.setattr(syncmod, "require_install_root", lambda: str(tmp_path))
    state_before = {str(p): p.read_bytes() for p in installs.rglob("*") if p.is_file()}
    rc = syncmod.main(["--installs-root", str(installs)])  # 无 --apply = dry-run
    state_after = {str(p): p.read_bytes() for p in installs.rglob("*") if p.is_file()}
    assert state_before == state_after  # dry-run 不写盘（幂等前提）
    assert (installs / "inst1" / "environments" / "env1" / "workspace" / "plugins"
            / "context_engine" / "supercompress" / "stray_not_in_src.txt").is_file()
    assert rc == 1  # dry-run 检测到 drift → 非 0 退出（门禁可 grep）


def test_sync_apply_is_idempotent_two_runs(tmp_path, monkeypatch):
    """apply 真写：缺文件补 + 旧文件覆 + 新增文件带；跑两次结果一致（第二次 0 改动 = 幂等），
    且插件外 stray 文件保留（不删非本插件文件）。"""
    src = tmp_path / "src"
    _mkplugin_files(src)
    installs = _mk_installs(tmp_path, src, drift=True)
    monkeypatch.setenv("SC_INSTALLS_ROOT", str(installs))
    import sc.sync_to_envs as syncmod
    monkeypatch.setattr(syncmod, "_PLUGIN_ROOT", str(src))
    monkeypatch.setattr(syncmod, "require_install_root", lambda: str(tmp_path))

    rc1 = syncmod.main(["--apply", "--installs-root", str(installs)])
    dst = (installs / "inst1" / "environments" / "env1" / "workspace"
           / "plugins" / "context_engine" / "supercompress")
    assert rc1 == 0
    # 代际对齐（单向判定：源文件在目标里字节全等；目标 stray 文件不影响——sync 不删它们）
    assert syncmod.aligned(src, dst)
    # 缺文件已补、旧文件已覆
    assert (dst / "docs" / "note.md").is_file()
    assert dst.joinpath("engine.py").read_bytes() == (src / "engine.py").read_bytes()
    # 插件外文件保留（不删）
    assert dst.joinpath("stray_not_in_src.txt").read_text() == "keep me\n"

    # 第二跑（幂等）：0 改动、0 覆写，结果一致
    snapshot = {str(p.relative_to(dst)): p.read_bytes() for p in dst.rglob("*") if p.is_file()}
    rc2 = syncmod.main(["--apply", "--installs-root", str(installs)])
    snapshot2 = {str(p.relative_to(dst)): p.read_bytes() for p in dst.rglob("*") if p.is_file()}
    assert rc2 == 0
    assert snapshot == snapshot2  # 跑两次结果一致


def test_sync_refuses_target_outside_installs_root(tmp_path, monkeypatch, capsys):
    """越界保护：--target 指向 installs 根之外 → 拒绝（工具写面永远限定在 installs 根之下）。"""
    src = tmp_path / "src"
    _mkplugin_files(src)
    installs = _mk_installs(tmp_path, src, drift=False)
    outside = tmp_path / "outside" / "plugins" / "context_engine" / "supercompress"
    outside.mkdir(parents=True)
    monkeypatch.setenv("SC_INSTALLS_ROOT", str(installs))
    import sc.sync_to_envs as syncmod
    monkeypatch.setattr(syncmod, "_PLUGIN_ROOT", str(src))
    monkeypatch.setattr(syncmod, "require_install_root", lambda: str(tmp_path))
    rc = syncmod.main(["--apply", "--installs-root", str(installs), "--target", str(outside)])
    out = capsys.readouterr()
    assert rc == 0  # 越界 target 被拒后，正常 env 目标已对齐 → 整体 0
    assert "拒绝越界" in out.err
