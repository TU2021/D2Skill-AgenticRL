# Flash-Attn 安装指南

## 需要安装的版本
根据 README.md 第 91 行，需要安装：
```
flash-attn==2.7.4.post1
```

## 当前环境信息
- Python: 3.10
- PyTorch: 2.8.0+cu128（或 2.6+cu12，以 `python -c "import torch; print(torch.__version__)"` 为准）
- CUDA: 12.x
- 平台: x86_64 (Linux)

**若使用 WebShop 等 GRPO 脚本且报 `ModuleNotFoundError: No module named 'flash_attn'`，需在对应 conda 环境（如 `skill-webshop`）中安装本包。**

## 安装方案

### 方案 1: 使用预编译 wheel（推荐，但可能不兼容）

**注意：** 官方只提供了 torch 2.6 的预编译 wheel，你的环境是 torch 2.8.0，可能不兼容。但可以尝试：

#### 选项 A: 使用 torch 2.6 的 wheel（Python 3.10, cxx11abi=FALSE）
```bash
conda activate skillrl
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

#### 选项 B: 使用 torch 2.6 的 wheel（Python 3.10, cxx11abi=TRUE）
```bash
conda activate skillrl
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

**如果预编译 wheel 不兼容，会报错，需要改用方案 2。**

#### 选项 C: PyTorch 2.8 + CUDA 12.8 + Python 3.10（社区预编译）

适用于 **torch 2.8.0+cu128、Python 3.10、Linux x86_64**。来自 [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) Release v0.7.16：

```bash
# 直接下载（约 100+ MB，请确保下载完整）
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4+cu128torch2.8-cp310-cp310-linux_x86_64.whl

conda activate skill-webshop
pip install --no-cache-dir flash_attn-2.7.4+cu128torch2.8-cp310-cp310-linux_x86_64.whl
```

**直链：**  
https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4+cu128torch2.8-cp310-cp310-linux_x86_64.whl  

下载后可用 `ls -l` 检查文件大小（正常应为几十 MB 以上），避免不完整导致 pip 报 invalid wheel。

### 方案 2: 从源码编译（需要先安装 ninja 加速）

**重要：** 必须先安装 `ninja`，否则编译会非常慢（可能 2 小时以上）！

```bash
conda activate skillrl

# 1. 安装 ninja（加速编译）
pip install ninja

# 2. 设置编译线程数（根据你的 CPU 核心数调整，建议 32-64）
export MAX_JOBS=64

# 3. 从源码编译安装
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

**或者使用详细输出模式（可以看到编译进度）：**
```bash
pip install -v flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

### 方案 3: 降级 PyTorch 到 2.6.0（如果方案 1 和 2 都不行）

如果 torch 2.8.0 确实不兼容，可以考虑降级到 2.6.0：

```bash
conda activate skillrl
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 然后使用预编译 wheel
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 验证安装

安装完成后，验证是否成功：

```bash
conda activate skillrl
python -c "import flash_attn; print('✅ flash-attn 安装成功'); print(f'版本: {flash_attn.__version__}')"
```

## 常见问题

### 1. 编译卡住
- 确保安装了 `ninja`
- 检查是否有足够的磁盘空间
- 检查 CUDA 工具链是否完整

### 2. 版本不兼容
- 如果预编译 wheel 不兼容，必须从源码编译
- 或者降级 PyTorch 到 2.6.0

### 3. 编译时间过长
- 确保安装了 `ninja`（可以加速 10-100 倍）
- 设置 `MAX_JOBS` 环境变量使用多线程编译

## 推荐步骤

1. **先尝试方案 1（预编译 wheel）** - 最快
2. **如果失败，使用方案 2（从源码编译 + ninja）** - 需要 10-30 分钟
3. **如果还是不行，考虑方案 3（降级 PyTorch）**






