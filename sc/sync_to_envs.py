#!/usr/bin/env python3
"""sync_to_envs.py — 把 repo 的 supercompress 同步到 installs 各 env 的插件副本（报告#1 #1 运维侧）。

背景：生产 gateway 跑的是 installs 里 env 的**代冻结快照**（install/env 重建时冻结一份
supercompress 副本到 <env>/workspace/plugins/context_engine/supercompress/），repo 上新
commit 没有任何进程在跑、也无自动机制搬上车。本工具收口「搬上车 + 对上代际」：

- 源 = 本插件目录（repo 侧，**只读**：只从源读字节，绝不写源）；
- 目标 = HERMES 安装根（``<hermes_home>/installs``，路径全走 sc/_bootstrap，0 硬编码；
  env override ``SC_INSTALLS_ROOT`` 可显式指认）下所有 env 的
  ``workspace/plugins/context_engine/supercompress/``（缺目录的 env 自动跳过，
  「缺文件补、旧文件覆、新增文件带」= 逐文件 sync，**不删**目标里非本插件/插件外文件，
  也绝不动 installs 根之外的任何东西）；
- 逐目标文件 diff（大小 + 字节全等才跳过）→ dry-run 打印将做的改动（默认，安全）；
  ``--apply`` 才真写（源文件原子替换：先写 ``<name>.scsync-tmp`` 再 rename，目标只读时
  自然失败报错，不产生半截文件）；
- 同步后逐目录对 build_id（sc/version.build_id，内容 hash 确定性可复现）：报
  「已同步 N 个 env，build=<>」，任一对不齐即非 0 退出码（供 cron 门禁 grep）。

跑法（repo 根，路径 0 硬编码——源/目标全由 __file__ + _bootstrap 派生）::

    python3 sc/sync_to_envs.py            # dry-run（默认）：只列将改什么
    python3 sc/sync_to_envs.py --apply    # 真写
    SC_INSTALLS_ROOT=<hermes root>/installs python3 sc/sync_to_envs.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# 安装可移植路径引导（零硬编码，与 sc/retag.py / sc/check_supercompress.py 同款）：
# 从 __file__ 反推插件根（sc.* 可导），仓根走 sc/_bootstrap（层级校验）。
_PLUGIN_ROOT = str(Path(__file__).resolve().parents[1])   # supercompress/
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from sc._bootstrap import require_install_root, hermes_home  # noqa: E402
from sc import version  # noqa: E402

_EXCLUDED_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"})
_EXCLUDED_SUFFIXES = (".pyc",)
_TMP_SUFFIX = ".scsync-tmp"


def _walk_source_files(src: Path) -> List[Path]:
    """源插件全部交付文件（相对路径）：跳过产物目录/字节码后缀。"""
    out: List[Path] = []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in _EXCLUDED_DIRS for part in rel.parts):
            continue
        if not p.is_file() or p.suffix in _EXCLUDED_SUFFIXES:
            continue
        out.append(p)
    return out


def _plugin_targets(installs_root: Path, extra: List[str]) -> List[Path]:
    """目标 = installs 下各 env 的 supercompress 插件目录 + 显式 --target。

    install 布局两层：<installs_root>/<install-id>/environments/<env-id>/workspace/
    plugins/context_engine/supercompress；也容忍 <installs_root>/<env>/workspace/... 直挂
    （缺目录的 env 不在列表——无插件 = 无目标，不算失败）。只读扫描 installs 下，
    写面永远限定在目标插件目录内；显式 --target 逐条校验「必须在 installs_root 之下」
    ——绝不碰 installs 根之外的东西。
    """
    targets: List[Path] = []
    if installs_root.is_dir():
        for install in sorted(p for p in installs_root.iterdir() if p.is_dir()):
            envs = install / "environments"
            env_dirs = sorted(p for p in envs.iterdir() if p.is_dir()) if envs.is_dir() else []
            # 双布局：标准 <install-id>/environments/<env-id> ；直挂 <install-id> 自身当 env
            bases = env_dirs if env_dirs else [install]
            for e in bases:
                t = e / "workspace" / "plugins" / "context_engine" / "supercompress"
                if t.is_dir():
                    targets.append(t)
    try:
        root_resolved = installs_root.resolve()
    except OSError:
        root_resolved = None
    for raw in extra:
        t = Path(raw).expanduser()
        if root_resolved is not None:
            try:
                if not t.resolve().is_relative_to(root_resolved):
                    print(f"[skip] target 不在 installs 根之下（拒绝越界）: {raw}", file=sys.stderr)
                    continue
            except OSError:
                print(f"[skip] target 解析失败: {raw}", file=sys.stderr)
                continue
        targets.append(t)
    # 去重（同一路径多次给出）保序
    seen = set()
    uniq: List[Path] = []
    for t in targets:
        key = str(t)
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


def diff_file(src_file: Path, dst_file: Path) -> bool:
    """需覆写 = 目标缺失，或字节不全等（先比大小短路，再逐块比对——幂等：内容同即不动）。"""
    if not dst_file.is_file():
        return True
    try:
        if dst_file.stat().st_size != src_file.stat().st_size:
            return True
    except OSError:
        return True
    with open(src_file, "rb") as a, open(dst_file, "rb") as b:
        while True:
            ba, bb = a.read(1024 * 1024), b.read(1024 * 1024)
            if ba != bb:
                return True
            if not ba:
                return False


def aligned(src: Path, target: Path) -> bool:
    """代际对齐判定（单向 源→env）：源里每个交付文件在目标里都有字节全等副本。

    目标里多出的非源文件（env 副本里的 stray/本插件外文件）不影响对齐——sync 不删
    它们（「缺文件补、旧文件覆、新增文件带；不删非本插件文件」），全等即对齐。
    """
    for src_file in _walk_source_files(src):
        if diff_file(src_file, target / src_file.relative_to(src)):
            return False
    return True


def sync_target(src: Path, target: Path, apply: bool) -> Tuple[int, int, int]:
    """一个目标目录的 sync：返回 (新增/补, 覆写, 未动) 文件数。dry-run 只计不写。"""
    added = overwritten = unchanged = 0
    for src_file in _walk_source_files(src):
        rel = src_file.relative_to(src)
        dst_file = target / rel
        if not diff_file(src_file, dst_file):
            unchanged += 1
            continue
        if dst_file.is_file():
            overwritten += 1
        else:
            added += 1
        if apply:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst_file.with_name(dst_file.name + _TMP_SUFFIX)
            shutil.copyfile(src_file, tmp)
            os.replace(tmp, dst_file)  # 原子替换：目标只读/权限异常在此如实报错，无半截文件
    return added, overwritten, unchanged


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="repo supercompress → installs 各 env 插件副本同步")
    ap.add_argument("--apply", action="store_true",
                    help="真写（默认 dry-run 只打印将改什么）")
    ap.add_argument("--target", action="append", default=[],
                    help="显式目标插件目录（须在 installs 根之下，可多次给出）")
    ap.add_argument("--installs-root", default="",
                    help="installs 根（缺省 = <hermes_home>/installs，env override SC_INSTALLS_ROOT）")
    args = ap.parse_args(argv)

    # 层级校验（fail-open）：正常安装层级恒通过；被搬走/测试布局下只警告不挡工具跑
    # （同步工具本身不依赖仓根，_PLUGIN_ROOT 由 __file__ 自推导即可）
    try:
        require_install_root()
    except Exception as e:
        print(f"[warn] 安装层级校验未通过（继续运行，源 = 本插件目录）: {e}", file=sys.stderr)
    installs_root = Path(
        args.installs_root
        or os.environ.get("SC_INSTALLS_ROOT", "").strip()
        or os.path.join(hermes_home(), "installs")
    )

    src = Path(_PLUGIN_ROOT)
    targets = _plugin_targets(installs_root, args.target)
    if not targets:
        print(f"no plugin copies found under {installs_root}（无 env 带 supercompress 插件，无事可做）")
        return 0

    # 代际对齐用 aligned()（单向 源→env：源文件在目标里字节全等；目标 stray 文件不影响）；
    # 汇报面 build_id 用源目录 build_id（git 仓 → 短 sha，便于人读"搬的是哪一代"）。
    src_bid = version.build_id(src)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode}: src={src} build={src_bid} installs_root={installs_root} targets={len(targets)}")

    all_ok = True
    in_sync = 0
    for t in targets:
        added, overwritten, unchanged = sync_target(src, t, apply=args.apply)
        ok = aligned(src, t)
        in_sync += 1 if ok else 0
        all_ok &= ok
        t_bid = version.build_id(t)
        note = "ok" if ok else f"MISMATCH(build={t_bid} src={src_bid})"
        print(f"  [{note}] {t} files: +{added} added / {overwritten} overwritten / {unchanged} unchanged"
              + ("" if args.apply else " (dry-run, 未写)"))
    print(f"已同步 {in_sync}/{len(targets)} 个 env，build={src_bid}"
          + ("" if args.apply else "（dry-run：--apply 才真写）"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
