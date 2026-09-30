#!/usr/bin/env python3
"""
Supercompress 健康检查与修复助手

步骤：
1. 检查当前配置状态
2. 注入 aux 端点环境变量（复用 Hermes 当前模型）
3. 运行 retag 重新生成 digest

路径零硬编码：config/store 走 HERMES_HOME（sc/_bootstrap），安装根从 __file__ 推导。
"""
import os
import sys
import subprocess
from pathlib import Path

# 安装可移植路径引导（零硬编码）：从 __file__ 反推插件根（sc.* / engine 可导）。
_PLUGIN_ROOT = str(Path(__file__).resolve().parents[1])   # supercompress/
_INSTALL_ROOT = str(Path(__file__).resolve().parents[4])  # hermes-agent 仓根
for _p in (_PLUGIN_ROOT, _INSTALL_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sc._bootstrap import config_path, active_store_path

# Hermes 配置文件路径（HERMES_HOME-aware，不写死某 profile / 某 home）
CONFIG_PATH = Path(config_path())
STORE_PATH = active_store_path()

def check_config():
    """检查配置状态"""
    print("=== 配置检查 ===")
    
    with open(CONFIG_PATH) as f:
        content = f.read()
    
    if 'use_aux_endpoint: true' in content:
        print("✓ use_aux_endpoint: true (已配置)")
        return True
    else:
        print("✗ use_aux_endpoint 未配置")
        return False

def get_hermes_model_info():
    """从 config.yaml 提取模型信息"""
    with open(CONFIG_PATH) as f:
        content = f.read()
    
    import re
    # 提取 base_url
    url_match = re.search(r'base_url:\s*(https?://\S+)', content)
    base_url = url_match.group(1) if url_match else ''
    
    # 提取 model
    model_match = re.search(r'model:\s*(\S+)', content)
    model = model_match.group(1) if model_match else ''
    
    # 提取 key_env
    key_env_match = re.search(r'key_env:\s*(\S+)', content)
    key_env = key_env_match.group(1) if key_env_match else ''
    
    return {
        'base_url': base_url,
        'model': model,
        'key_env': key_env
    }

def set_env_vars():
    """设置 aux 端点环境变量"""
    print("\n=== 设置环境变量 ===")
    
    info = get_hermes_model_info()
    print(f"模型信息: {info}")
    
    # 从 config.yaml 解析
    base_url = info['base_url']
    model = info['model']
    key_env = info['key_env']
    
    # 获取 API key
    api_key = os.environ.get(key_env, '')
    if not api_key:
        print(f"✗ 环境变量 {key_env} 未设置")
        return False
    
    # 设置环境变量
    os.environ['LIVE_ENDPOINT_URL'] = base_url
    os.environ['LIVE_ENDPOINT_KEY'] = api_key
    os.environ['LIVE_ENDPOINT_MODEL'] = model
    
    print(f"✓ LIVE_ENDPOINT_URL={base_url}")
    print(f"✓ LIVE_ENDPOINT_MODEL={model}")
    print(f"✓ LIVE_ENDPOINT_KEY={api_key[:8]}***{api_key[-4:]}")
    
    return True

def test_endpoint():
    """测试端点连接"""
    print("\n=== 测试端点 ===")
    
    try:
        from sc.live_endpoint import resolve_live_endpoint
        ep = resolve_live_endpoint()
        print(f"✓ 端点已解析: {ep.base_url}")
        print(f"  模型: {ep.model}")
        
        # 测试调用
        result = ep.call("test", "Hello, this is a test message for digest generation.")
        print(f"✓ 调用成功: {result[:50]}...")
        return True
    except Exception as e:
        print(f"✗ 端点测试失败: {e}")
        return False

def run_retag():
    """运行 retag 脚本"""
    print("\n=== 运行 Retag ===")
    
    try:
        from plugins.context_engine.supercompress.engine import SupercompressContextEngine
        from sc.store import JsonlStore
        
        e = SupercompressContextEngine()
        store = JsonlStore(STORE_PATH)
        segs = store.replay()
        
        print(f"读取到 {len(segs)} 个段")
        
        # 统计非结构 P0
        non_struct_p0 = [s for s in segs if s.tier == 'P0' and not s.structural_p0]
        print(f"非结构 P0: {len(non_struct_p0)}")
        
        # 生成 digest（sync 立即 / async 入队）；汇合后按 digest_ready 统一判定
        for seg in non_struct_p0[:10]:  # 只测试前10条
            e._aux.transient = False
            e._aux.ensure_digest(seg)
        e._aux.flush(timeout=60.0)

        success = 0
        fail = 0
        for seg in non_struct_p0[:10]:
            if seg.digest_ready:
                success += 1
            else:
                fail += 1
        
        print(f"测试生成: 成功 {success}, 失败 {fail}")
        
        if fail == 0:
            print("✓ Digest 生成正常，可以运行完整 retag")
        else:
            print("✗ Digest 生成失败，需要检查端点配置")
            
    except Exception as e:
        print(f"✗ Retag 测试失败: {e}")

def main():
    print("=" * 60)
    print("Supercompress 健康检查与修复")
    print("=" * 60)
    
    # 步骤1: 检查配置
    config_ok = check_config()
    
    # 步骤2: 设置环境变量
    if not set_env_vars():
        print("\n❌ 无法设置环境变量，请手动设置（替换为你的 agent LLM 端点）:")
        print("  export LIVE_ENDPOINT_URL=<your-agent-llm-endpoint>/v1")
        print("  export LIVE_ENDPOINT_KEY=<your-agent-api-key>")
        print("  export LIVE_ENDPOINT_MODEL=<your-agent-model>")
        return 1
    
    # 步骤3: 测试端点
    if not test_endpoint():
        print("\n❌ 端点测试失败")
        return 1
    
    # 步骤4: 运行 retag
    run_retag()
    
    print("\n" + "=" * 60)
    print("修复完成！")
    print("=" * 60)
    print("\n下一步：")
    print("1. 重启 Hermes 服务以加载新配置")
    print("2. 或者手动设置环境变量后运行 retag.py")
    print("3. 验证压缩是否正常工作")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
