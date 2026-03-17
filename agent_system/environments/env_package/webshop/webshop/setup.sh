#!/bin/bash

# Displays information on how to use script
helpFunction()
{
  echo "Usage: $0 [-d small|all]"
  echo -e "\t-d small|all - Specify whether to download entire dataset (all) or just 1000 (small)"
  exit 1 # Exit script after printing help
}

# Get values of command line flags
while getopts d: flag
do
  case "${flag}" in
    d) data=${OPTARG};;
  esac
done

if [ -z "$data" ]; then
  echo "[ERROR]: Missing -d flag"
  helpFunction
fi

# Install Python Dependencies
pip install -r requirements.txt;

conda install mkl
conda install -c conda-forge faiss-cpu

# Install Environment Dependencies via `conda`
# conda install -c pytorch faiss-cpu;
conda install -c conda-forge openjdk=11;

# Download dataset into `data` folder via `gdown` command
mkdir -p data;
cd data;
if [ "$data" == "small" ]; then
  gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib; # items_shuffle_1000 - product scraped info
  gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; # items_ins_v2_1000 - product attributes
elif [ "$data" == "all" ]; then
  gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib; # items_shuffle_1000 - product scraped info
  gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; # items_ins_v2_1000 - product attributes
  gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB; # items_shuffle
  gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; # items_ins_v2
else
  echo "[ERROR]: argument for `-d` flag not recognized"
  helpFunction
fi
gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O # items_human_ins
cd ..

# Download spaCy large NLP model
# 使用 conda 安装（推荐，支持镜像源，网络更稳定）
echo "Installing spaCy models via conda (supports mirror sources)..."
conda install -c conda-forge spacy-model-en_core_web_sm -y
conda install -c conda-forge spacy-model-en_core_web_lg -y || echo "Warning: en_core_web_lg installation failed (optional, code mainly uses en_core_web_sm)"

# 如果 conda 安装失败，回退到 pip 方式（需要手动配置镜像或使用代理）
if ! python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    echo "Conda installation failed, trying pip with direct download..."
    # 直接从 GitHub releases 下载 wheel 文件安装
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl || \
    python -m spacy download en_core_web_sm || echo "Error: Failed to install en_core_web_sm"
fi

# 验证必需模型安装
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✓ en_core_web_sm installed successfully')" || echo "✗ en_core_web_sm installation failed - please install manually"

# Build search engine index
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh
cd ..

# Create logging folder + samples of log data
# get_human_trajs () {
#   PYCMD=$(cat <<EOF
# import gdown
# url="https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
# gdown.download_folder(url, quiet=True, remaining_ok=True)
# EOF
#   )
#   python -c "$PYCMD"
# }
# mkdir -p user_session_logs/
# cd user_session_logs/
# echo "Downloading 50 example human trajectories..."
# get_human_trajs
# echo "Downloading example trajectories complete"
# cd ..