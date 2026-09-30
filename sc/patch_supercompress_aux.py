#!/usr/bin/env python3
"""
patch_supercompress_aux.py - 为 supercompress 配置 aux 端点

用途: 在 config.yaml 中添加 use_aux_endpoint: true
      并确保环境变量指向 Hermes 当前使用的模型

用法:
    python3 patch_supercompress_aux.py
"""
import os
import re
from pathlib import Path

CONFIG_PATH = Path('/root/.hermes/profiles/orchestrator/config.yaml')

def get_hermes_model_config():
    """从 config.yaml 读取当前模型配置"""
    with open(CONFIG_PATH) as f:
        content = f.read()
    
    # 提取 model 配置
    model_match = re.search(r'model:\s*\n((?:  .+\n)*)', content)
    if not model_match:
        return None
    
    model_cfg = {}
    for line in model_match.group(1).split('\n'):
        line = line.strip()
        if ':' in line:
            key, val = line.split(':', 1)
            model_cfg[key.strip()] = val.strip().strip('"\'')
    
    # 提取 custom_providers
    providers_match = re.search(r'custom_providers:\s*\n((?:  -\s*\n(?:    .+\n)*)*)', content)
    if providers_match:
        providers_text = providers_match.group(1)
        # 找到当前 provider
        provider_name = model_cfg.get('provider', '')
        if provider_name:
            prov_match = re.search(r'-\s*\n\s*name:\s*(\S+)\s*\n(?:.*\n)*?(?:.*\n)*?(?:.*\n)*?(?:.*\n)*?(?=\n\s*-\s*|\Z)', 
                                   providers_text, re.MULTILINE)
            if prov_match:
                # 查找对应的 base_url 和 key_env
                for match in re.finditer(r'-\s*\n\s*name:\s*(\S+)\s*\n((?:    .+\n)*)', providers_text):
                    name = match.group(1)
                    provider_cfg = {}
                    for line in match.group(2).split('\n'):
                        line = line.strip()
                        if ':' in line:
                            k, v = line.split(':', 1)
                            provider_cfg[k.strip()] = v.strip().strip('"\'')
                    if name == provider_name:
                        return {
                            'base_url': provider_cfg.get('base_url', ''),
                            'model': provider_cfg.get('model', model_cfg.get('default', '')),
                            'key_env': provider_cfg.get('key_env', model_cfg.get('api_key', '').replace('${', '').replace('}', ''))
                        }
    
    return model_cfg

def main():
    print("=== Supercompress Aux Endpoint Setup ===\n")
    
    # 获取当前模型配置
    model_cfg = get_hermes_model_config()
    if not model_cfg:
        print("ERROR: 无法读取模型配置")
        return
    
    print(f"当前 Hermes 模型配置:")
    print(f"  base_url: {model_cfg.get('base_url')}")
    print(f"  model: {model_cfg.get('model')}")
    print(f"  key_env: {model_cfg.get('key_env', 'N/A')}")
    
    # 获取 API key
    key_env = model_cfg.get('key_env', '')
    api_key = os.environ.get(key_env, '') if key_env else ''
    
    if not api_key:
        print(f"\nWARNING: 环境变量 {key_env} 未设置或为空")
        print("请先设置正确的 API key")
        return
    
    print(f"\nAPI key: {api_key[:8]}***{api_key[-4:]} ({len(api_key)} chars)")
    
    # 读取当前 config
    with open(CONFIG_PATH) as f:
        content = f.read()
    
    # 检查是否已有 use_aux_endpoint
    if 'use_aux_endpoint' in content:
        print("\n✓ use_aux_endpoint 已存在")
        if 'use_aux_endpoint: true' in content:
            print("✓ use_aux_endpoint 已为 true")
            return
    else:
        # 添加 use_aux_endpoint
        content = content.replace(
            'compression:\n  supercompress:',
            'compression:\n  supercompress:\n    use_aux_endpoint: true'
        )
        print("\n✓ 已添加 use_aux_endpoint: true")
    
    # 写回 config
    with open(CONFIG_PATH, 'w') as f:
        f.write(content)
    
    # 输出环境变量设置命令
    print("\n=== 需要设置的环境变量 ===")
    print(f'export LIVE_ENDPOINT_URL="{model_cfg.get("base_url")}"')
    print(f'export LIVE_ENDPOINT_KEY="{api_key}"')
    print(f'export LIVE_ENDPOINT_MODEL="{model_cfg.get("model")}"')
    
    print("\n=== 验证命令 ===")
    print("source /root/.hermes/profiles/orchestrator/setup_supercompress_aux.sh")
    print("python3 /usr/local/lib/hermes-agent/plugins/context_engine/supercompress/sc/retag.py")
    
    # 创建环境变量设置脚本
    setup_script = Path('/root/.hermes/profiles/orchestrator/setup_supercompress_aux.sh')
    with open(setup_script, 'w') as f:
        f.write(f'''#!/bin/bash
# Supercompress aux endpoint setup
# 使用 Hermes 当前模型配置
export LIVE_ENDPOINT_URL="{model_cfg.get('base_url')}"
export LIVE_ENDPOINT_KEY="${{{{key_env}}}}"
export LIVE_ENDPOINT_MODEL="{model_cfg.get('model')}"
''')
    setup_script.chmod(0o755)
    
    print(f"\n✓ 已创建环境变量脚本: {setup_script}")

if __name__ == '__main__':
    main()
