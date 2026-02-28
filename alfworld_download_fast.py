#!/usr/bin/env python3
"""
ALFWorld 加速下载脚本
支持多种加速方式：
1. 使用 GitHub 镜像（ghproxy.com）
2. 使用多线程下载（aria2c）
3. 使用代理
"""

import os
import sys
import argparse
import subprocess
import requests
from pathlib import Path
from tqdm import tqdm

# GitHub 镜像列表（按优先级排序）
MIRRORS = [
    "",  # 原始 GitHub（最后备选）
    "ghproxy.com/https://github.com",  # 国内常用镜像
    "mirror.ghproxy.com/https://github.com",
]

# 下载 URL
FILES = {
    "json_2.1.1_json.zip": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip",
    "json_2.1.1_pddl.zip": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip",
    "json_2.1.2_tw-pddl.zip": "https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip",
    "mrcnn_alfred_objects_sep13_004.pth": "https://github.com/alfworld/alfworld/releases/download/0.2.2/mrcnn_alfred_objects_sep13_004.pth",
}


def get_mirror_url(original_url, mirror=""):
    """将原始 URL 转换为镜像 URL"""
    if not mirror:
        return original_url
    # 替换 github.com 为镜像
    return original_url.replace("https://github.com", f"https://{mirror}")


def download_with_aria2(url, output_path):
    """使用 aria2c 多线程下载"""
    try:
        cmd = [
            "aria2c",
            "-x", "16",  # 16 个连接
            "-s", "16",  # 16 个分片
            "-k", "1M",  # 最小分片大小
            "--dir", str(output_path.parent),
            "--out", output_path.name,
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except Exception as e:
        print(f"aria2c 下载失败: {e}")
    return False


def download_with_requests(url, output_path, proxy=None):
    """使用 requests 下载（支持代理）"""
    try:
        session = requests.Session()
        if proxy:
            session.proxies = {"http": proxy, "https": proxy}
        
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"requests 下载失败: {e}")
        return False


def download_file(url, output_path, force=False, mirror="", proxy=None, use_aria2=True):
    """下载文件（自动选择最佳方式）"""
    output_path = Path(output_path)
    
    if output_path.exists() and not force:
        print(f"文件已存在: {output_path} (使用 --force 强制重新下载)")
        return True
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 尝试使用镜像
    if mirror:
        url = get_mirror_url(url, mirror)
        print(f"使用镜像: {mirror}")
    
    print(f"正在下载: {output_path.name}")
    print(f"URL: {url}")
    
    # 优先使用 aria2c（多线程，速度快）
    if use_aria2 and download_with_aria2(url, output_path):
        print(f"✅ 下载完成: {output_path}")
        return True
    
    # 回退到 requests（支持代理）
    if download_with_requests(url, output_path, proxy):
        print(f"✅ 下载完成: {output_path}")
        return True
    
    print(f"❌ 下载失败: {output_path.name}")
    return False


def main():
    parser = argparse.ArgumentParser(description="ALFWorld 加速下载工具")
    parser.add_argument("--data-dir", default=os.getenv("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")),
                        help="数据保存目录")
    parser.add_argument("--mirror", choices=["ghproxy", "auto", "none"], default="auto",
                        help="GitHub 镜像选择: ghproxy (使用 ghproxy.com), auto (自动选择), none (不使用镜像)")
    parser.add_argument("--proxy", help="HTTP/HTTPS 代理 (例如: http://127.0.0.1:7890)")
    parser.add_argument("--force", action="store_true", help="强制重新下载已存在的文件")
    parser.add_argument("--no-aria2", action="store_true", help="不使用 aria2c（即使已安装）")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    detectors_dir = data_dir / "detectors"
    detectors_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择镜像
    mirror = ""
    if args.mirror == "ghproxy":
        mirror = "ghproxy.com/https://github.com"
    elif args.mirror == "auto":
        # 自动选择：先尝试 ghproxy
        mirror = "ghproxy.com/https://github.com"
        print("使用自动镜像选择（ghproxy.com）")
    
    print(f"数据目录: {data_dir}")
    print(f"镜像: {mirror if mirror else '无（使用原始 GitHub）'}")
    if args.proxy:
        print(f"代理: {args.proxy}")
    print("-" * 60)
    
    # 下载所有文件
    success_count = 0
    for filename, url in FILES.items():
        if filename.endswith(".pth"):
            output_path = detectors_dir / filename
        else:
            output_path = data_dir / filename
        
        if download_file(url, output_path, args.force, mirror, args.proxy, not args.no_aria2):
            success_count += 1
    
    print("-" * 60)
    print(f"下载完成: {success_count}/{len(FILES)} 个文件")
    
    if success_count == len(FILES):
        print("\n✅ 所有文件下载完成！")
        print(f"现在可以运行: alfworld-download -f")
        print("（alfworld-download 会检测到文件已存在，直接解压）")
    else:
        print(f"\n⚠️  有 {len(FILES) - success_count} 个文件下载失败")
        print("建议：")
        print("1. 检查网络连接")
        print("2. 尝试使用代理: --proxy http://your-proxy:port")
        print("3. 尝试不同的镜像: --mirror ghproxy")
        sys.exit(1)


if __name__ == "__main__":
    main()

