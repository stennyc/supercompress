#!/usr/bin/env python3
"""
临时 patch：为当前测试会话设置 aux 端点环境变量

用法（路径自动推导，无安装目录硬编码）：
    source <HERMES_HOME>/setup_supercompress_aux.sh
    python3 <...>/plugins/context_engine/supercompress/sc/patch_aux_endpoint.py
"""
import os
import sys
from pathlib import Path

# 安装可移植路径引导（零硬编码）：从 __file__ 反推插件根（sc.* 可导）。
_PLUGIN_ROOT = str(Path(__file__).resolve().parents[1])   # supercompress/
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

# 检查环境变量
print("=== 检查环境变量 ===")
for var in ['LIVE_ENDPOINT_URL', 'LIVE_ENDPOINT_KEY', 'LIVE_ENDPOINT_MODEL']:
    val = os.environ.get(var, '')
    if val:
        masked = val[:8] + '***' + val[-4:] if len(val) > 12 else '***'
        print(f"✓ {var}={masked}")
    else:
        print(f"✗ {var}=NOT_SET")

# 尝试解析端点
print("\n=== 解析端点 ===")
try:
    from sc.live_endpoint import resolve_live_endpoint
    ep = resolve_live_endpoint()
    print(f"✓ 端点已解析: {ep.base_url}")
    print(f"  模型: {ep.model}")
    
    # 测试调用
    print("\n=== 测试调用 ===")
    result = ep.call("auxiliary.compression", "test content for digest generation")
    print(f"✓ 调用成功: {result[:50]}...")
    
except Exception as e:
    print(f"✗ 解析失败: {e}")
    print("\n需要执行以下步骤：")
    print("  1. 导出 LIVE_ENDPOINT_URL / LIVE_ENDPOINT_KEY / LIVE_ENDPOINT_MODEL")
    print("     （指向你的 agent LLM 端点，参考 sc/live_endpoint.py 解析链）")
