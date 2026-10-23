#!/usr/bin/env python3
"""
retag.py - 存量条目重打分脚本

用途: 在 A1 修复后，对 active.jsonl 中已有的非结构 P0 段生成 digest
执行时机: A1 修改 tagger.py 后立即运行一次

用法:
    python3 retag.py [--dry-run]
    
    --dry-run: 仅打印不会修改
"""
import sys
import json
from pathlib import Path

# 安装可移植路径引导（零硬编码）：从 __file__ 反推插件根（sc.* 可导）与仓根
# （plugins.context_engine.* 可导），换任何安装目录/任何用户都成立。
import os
_PLUGIN_ROOT = str(Path(__file__).resolve().parents[1])   # supercompress/
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from sc._bootstrap import active_store_path, require_install_root
_INSTALL_ROOT = require_install_root()  # 层级校验：深度假设失败时报可读说明（reviewer Minor②）
if _INSTALL_ROOT not in sys.path:
    sys.path.insert(0, _INSTALL_ROOT)

from plugins.context_engine.supercompress.engine import SupercompressContextEngine
from sc.store import JsonlStore
from sc.schema import Segment
from scdata import to_line as scdata_to_line
from sc.tagger import is_reasoning_monologue

def retag_segments(store_path: str, dry_run: bool = False, max_segments: int = 0):
    """对存量条目重打分并生成 digest"""
    
    # 加载 engine
    e = SupercompressContextEngine()
    store = JsonlStore(store_path)
    
    # 读取所有段
    segs = store.replay()
    print(f"读取到 {len(segs)} 个段")
    
    # 分类统计
    pending_digest = []
    ready_digest = []
    no_digest_attr = []
    
    for seg in segs:
        digest_ref = getattr(seg, 'digest_ref', None)
        if digest_ref is None:
            no_digest_attr.append(seg)
        elif digest_ref == 'pending':
            pending_digest.append(seg)
        else:
            ready_digest.append(seg)
    
    print(f"\n分类:")
    print(f"  已有 digest: {len(ready_digest)}")
    print(f"  Pending digest: {len(pending_digest)}")
    print(f"  无 digest 属性: {len(no_digest_attr)}")
    
    # 只为非结构 P0 生成 digest
    non_struct_p0_pending = [
        seg for seg in pending_digest
        if seg.tier == 'P0' and not seg.structural_p0
    ]
    
    print(f"\n需要生成 digest 的非结构 P0: {len(non_struct_p0_pending)}")
    
    if max_segments > 0:
        non_struct_p0_pending = non_struct_p0_pending[:max_segments]
        print(f"处理前 {max_segments} 条...")
    
    if not non_struct_p0_pending:
        print("无需处理")
        return
    
    # 尝试生成 digest（sync 立即生成 / async 入队后台生成）
    for i, seg in enumerate(non_struct_p0_pending):
        # 重置 transient 状态
        e._aux.transient = False

        # 尝试生成 digest（async 下返回 None=已入队，非失败）
        e._aux.ensure_digest(seg)
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(non_struct_p0_pending)}...")

    # 异步 worker 汇合（sync 模式 no-op 立即返回）；之后按 digest_ready 统一判定
    e._aux.flush(timeout=120.0)

    success_count = 0
    fail_count = 0
    for seg in non_struct_p0_pending:
        if seg.digest_ready:
            success_count += 1
        else:
            fail_count += 1
            if fail_count <= 10:
                print(f"  ✗ {seg.segment_id[:8]}: not ready (transient={e._aux.transient})")

    print(f"\n结果:")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    
    if fail_count > 0 and not dry_run:
        print("\n⚠️  digest 生成失败（端点瞬态/超时）")
        print("   生产路径零配置（宿主 update_model 注入 agent 运行时）；")
        print("   离线工具需 shell 导出 LIVE_ENDPOINT_URL/KEY/MODEL（sc/live_endpoint 解析链）")
        print("   或检查 agent LLM API 端点健康度")

def retag_monologue(store_path: str, dry_run: bool = True) -> list:
    """R2（2026-10-23 data-engineer 复盘项 #3）：`i'm ` 独白词批量修复面。

    tagger REASONING_MONOLOGUE_CUES 加 "i'm " 后，历史 store 里已有的 model P0
    非结构段（开首 "I'm checking/noticing…" 独白）仍按旧 tagger 留在 P0 注入面，
    新段才降 P2 → 同 session 内新旧 tier 断层。本函数对全量段重跑
    `is_reasoning_monologue`，命中者降 P2 并落盘（tier 可变，content_hash 只验
    original_text → 原地重写合法；replay 不重跑 tiering → 批量重打是唯一一致面）。

    面：只翻转「P0 + 非 structural + provenance=model + 独白命中」段；
    user 段 / structural P0 / 结论句（i'm + 结论词）一律不动。幂等（已 P2 段
    不再翻转）。dry_run=True（默认）只报不写，返回 [segment_id, ...]。
    """
    store = JsonlStore(store_path)
    segs = store.replay()
    flipped = []
    for seg in segs:
        if seg.tier != "P0" or seg.structural_p0 or seg.provenance != "model":
            continue
        if not is_reasoning_monologue(seg.original_text or ""):
            continue
        flipped.append(seg.segment_id)
        if not dry_run:
            seg.tier = "P2"
    if not dry_run and flipped:
        _rewrite_store(store, segs)
    print(f"独白重打标: 命中 {len(flipped)} 段"
          + ("（dry-run，未写盘）" if dry_run else "（已降 P2 落盘）"))
    return flipped


def _rewrite_store(store, segs):
    """tier 变更批量落盘：整文件按 replay 行序重写（原子：临时文件 + rename）。

    行字节 = scdata.to_line(seg.to_dict())（与 append 面同源），content_hash 不变
    （只验 original_text）→ 重写后 replay 验证面零扰动。"""
    tmp_path = store.path + ".retag.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for seg in segs:
            f.write(scdata_to_line(seg.to_dict()) + "\n")
    os.replace(tmp_path, store.path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Retag supercompress segments')
    parser.add_argument('--max', type=int, default=0, help='Max segments to process (0=all)')
    args = parser.parse_args()
    
    store_path = active_store_path()
    retag_segments(store_path, dry_run=False, max_segments=args.max)
