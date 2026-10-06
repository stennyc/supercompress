#!/bin/bash
# Supercompress Retag 快速脚本
# 只处理前 N 条非结构 P0 以验证功能

set -e

echo "=== Supercompress Retag (Batch Mode) ==="
echo ""

# 插件相对路径：脚本自身就在 supercompress/tests/ 下，PLUGIN_DIR 为其父目录
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# API key 从插件目录 .env 读（宿主插件仓自带），避免硬编码 profile 路径
API_KEY_FILE="${PLUGIN_DIR}/.env"
if [ -f "$API_KEY_FILE" ]; then
  API_KEY=$(grep HERMES_CUSTOM_10_1_1_116_8080_API_KEY "$API_KEY_FILE" | cut -d= -f2)
fi
export LIVE_ENDPOINT_URL="${LIVE_ENDPOINT_URL:-https://api.riway30.com/v1}"
export LIVE_ENDPOINT_MODEL="${LIVE_ENDPOINT_MODEL:-agnes-2.5-flash}"
export LIVE_ENDPOINT_KEY="${LIVE_ENDPOINT_KEY:-$API_KEY}"

# 只处理前 50 条作为测试
MAX_SEGMENTS=${1:-50}

cd "$PLUGIN_DIR"
PYTHONPATH=".:$PYTHONPATH" /usr/local/lib/hermes-agent/venv/bin/python3 << PYEOF
import sys
import os
from plugins.context_engine.supercompress.engine import SupercompressContextEngine

e = SupercompressContextEngine()
segs = e.store.replay()

# 过滤非结构 P0
non_struct_p0 = [s for s in segs if s.tier == 'P0' and not s.structural_p0]
print(f"Found {len(non_struct_p0)} non-structural P0 segments")
print(f"Processing first {min({MAX_SEGMENTS}, len(non_struct_p0))} segments...")
print()

success = 0
fail = 0
for i, seg in enumerate(non_struct_p0[:int({MAX_SEGMENTS})]):
    e._aux.transient = False
    result = e._aux.ensure_digest(seg)
    if result:
        success += 1
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{min({MAX_SEGMENTS}, len(non_struct_p0))}...")
    else:
        fail += 1
        if fail <= 5:
            print(f"  ✗ Segment {seg.segment_id[:8]}: failed")

print()
print(f"Results: {success} success, {fail} failed")

# 检查状态
segs_after = e.store.replay()
pending = sum(1 for s in segs_after if getattr(s, 'digest_ref', '') == 'pending')
ready = sum(1 for s in segs_after if getattr(s, 'digest_ref', '').startswith('ready:'))
print(f"Digest status: {ready} ready, {pending} pending")

if ready > 0:
    print("\n✓ Digest generation working! Run full retag:")
    print(f"  PYTHONPATH='.:$PYTHONPATH' /usr/local/lib/hermes-agent/venv/bin/python3 {PLUGIN_DIR}/sc/retag.py")
PYEOF
