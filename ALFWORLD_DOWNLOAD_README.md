# ALFWorld 加速下载指南

## 问题
`alfworld-download -f` 下载速度很慢，因为从 GitHub releases 下载大文件。

## 解决方案

### 方案 1: 使用加速下载脚本（推荐）

已创建加速下载脚本 `alfworld_download_fast.py`，支持多种加速方式：

#### 基本使用（自动选择镜像）
```bash
conda activate skillrl
python alfworld_download_fast.py
```

#### 使用 GitHub 镜像加速
```bash
python alfworld_download_fast.py --mirror ghproxy
```

#### 使用代理加速
```bash
python alfworld_download_fast.py --proxy http://127.0.0.1:7890
```

#### 组合使用（镜像 + 代理）
```bash
python alfworld_download_fast.py --mirror ghproxy --proxy http://127.0.0.1:7890
```

#### 安装 aria2c 实现多线程下载（可选，速度更快）
```bash
# Ubuntu/Debian
sudo apt-get install aria2

# CentOS/RHEL
sudo yum install aria2

# 或使用 conda
conda install -c conda-forge aria2
```

安装后，脚本会自动使用 aria2c 进行多线程下载。

### 方案 2: 手动设置代理环境变量

如果使用代理，可以设置环境变量：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
alfworld-download -f
```

### 方案 3: 使用 GitHub 镜像网站手动下载

1. 访问 GitHub 镜像网站（如 https://ghproxy.com/）
2. 在镜像网站中打开以下链接：
   - https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip
   - https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip
   - https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip
   - https://github.com/alfworld/alfworld/releases/download/0.2.2/mrcnn_alfred_objects_sep13_004.pth

3. 下载后放到 `~/.cache/alfworld/` 目录（或 `$ALFWORLD_DATA` 指定的目录）
4. `.pth` 文件放到 `~/.cache/alfworld/detectors/` 目录
5. 运行 `alfworld-download -f` 进行解压和设置

### 方案 4: 使用 wget/curl 配合代理下载

```bash
# 设置代理（如果需要）
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 下载文件
mkdir -p ~/.cache/alfworld/detectors
cd ~/.cache/alfworld

wget https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip
wget https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip
wget https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip
wget -P detectors/ https://github.com/alfworld/alfworld/releases/download/0.2.2/mrcnn_alfred_objects_sep13_004.pth

# 然后运行解压
alfworld-download -f
```

## 下载完成后

下载完成后，运行：
```bash
alfworld-download -f
```

这会自动检测已下载的文件，进行解压和设置。

## 验证安装

```bash
python -c "import alfworld; print('ALFWorld 安装成功！')"
```

