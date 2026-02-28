#!/usr/bin/env python3
"""
SkillRL 环境测试脚本
检查所有关键依赖和模块是否正确安装
"""

import sys
from typing import List, Tuple

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """测试模块导入"""
    try:
        if package_name:
            mod = __import__(package_name)
            version = getattr(mod, '__version__', '未知版本')
        else:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', '未知版本')
        return True, f"✓ {module_name} (版本: {version})"
    except ImportError as e:
        return False, f"✗ {module_name} - 未安装: {str(e)}"
    except Exception as e:
        return False, f"✗ {module_name} - 错误: {str(e)}"

def test_cuda():
    """测试CUDA可用性"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            return True, f"✓ CUDA可用 (版本: {cuda_version}, 设备数: {device_count})"
        else:
            return False, "✗ CUDA不可用"
    except Exception as e:
        return False, f"✗ CUDA测试失败: {str(e)}"

def main():
    print("=" * 60)
    print("SkillRL 环境测试报告")
    print("=" * 60)
    print()
    
    results = []
    
    # 基础环境
    print("【基础环境】")
    print(f"Python版本: {sys.version.split()[0]}")
    print()
    
    # 核心深度学习库
    print("【核心深度学习库】")
    results.append(test_import("torch", "torch"))
    results.append(test_cuda())
    results.append(test_import("transformers", "transformers"))
    results.append(test_import("numpy", "numpy"))
    results.append(test_import("accelerate", "accelerate"))
    print()
    
    # vLLM和Flash Attention
    print("【推理加速库】")
    results.append(test_import("vllm", "vllm"))
    results.append(test_import("flash_attn", "flash_attn"))
    print()
    
    # 分布式和训练框架
    print("【分布式和训练框架】")
    results.append(test_import("ray", "ray"))
    results.append(test_import("hydra", "hydra"))
    results.append(test_import("verl", "verl"))
    print()
    
    # 数据处理
    print("【数据处理库】")
    results.append(test_import("datasets", "datasets"))
    results.append(test_import("pandas", "pandas"))
    results.append(test_import("peft", "peft"))
    print()
    
    # 可选依赖（用于检索内存）
    print("【可选依赖（检索内存）】")
    results.append(test_import("sentence_transformers", "sentence_transformers"))
    try:
        import faiss
        results.append((True, "✓ faiss-cpu (已安装)"))
    except:
        results.append((False, "✗ faiss-cpu - 未安装"))
    print()
    
    # 环境特定库
    print("【环境特定库】")
    results.append(test_import("alfworld", "alfworld"))
    results.append(test_import("gymnasium", "gymnasium"))
    results.append(test_import("stable_baselines3", "stable_baselines3"))
    print()
    
    # 其他工具
    print("【其他工具】")
    results.append(test_import("wandb", "wandb"))
    print()
    
    # 测试verl.trainer导入
    print("【verl.trainer模块】")
    try:
        from verl.trainer import main_ppo
        results.append((True, "✓ verl.trainer.main_ppo 导入成功"))
    except Exception as e:
        results.append((False, f"✗ verl.trainer.main_ppo - 导入失败: {str(e)}"))
    print()
    
    # 打印所有结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for success, message in results:
        print(message)
    
    # 统计
    passed = sum(1 for s, _ in results if s)
    total = len(results)
    print()
    print(f"通过: {passed}/{total} ({passed*100//total}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！环境配置正确。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试未通过，请检查缺失的依赖。")
        return 1

if __name__ == "__main__":
    sys.exit(main())

