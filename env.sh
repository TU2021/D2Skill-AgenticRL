#!/bin/bash
# verl environment configuration file

# Ray Dashboard configuration (optional)
# Uncomment and set if you want to use Ray dashboard
# RAY_DASHBOARD_CONFIG="trainer.ray_dashboard_host=0.0.0.0 trainer.ray_dashboard_port=8265"

# Set RAY_DASHBOARD_CONFIG to empty if not needed
RAY_DASHBOARD_CONFIG=""

# SwanLab configuration
export SWANLAB_MODE="cloud"
export SWANLAB_API_KEY="17hxTIcoUR4LyfcvSDsa5"
export SWANLAB_LOG_DIR="swanlog"
# 禁用交互式提示（非交互模式）
export SWANLAB_DISABLE_PROMPT=1
# 或者使用环境变量禁用交互
export SWANLAB_NON_INTERACTIVE=1

# CUDA configuration (automatically set by conda cuda-toolkit)
# If CUDA_HOME is not set, deep_gemm will try to find it via nvcc
# Uncomment and set manually if needed:
# export CUDA_HOME=/data/home/zdhs0006/.conda/envs/verl_0_7

# FAISS GPU library path (required for faiss-gpu to work)
# Set LD_LIBRARY_PATH to use conda environment's libstdc++ instead of system's
# Only add if not already set to avoid conflicts with PyTorch CUDA libraries
# if [ -n "$CONDA_PREFIX" ] && [ -z "$FAISS_LD_LIBRARY_PATH_SET" ]; then
#     export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib:$LD_LIBRARY_PATH
#     export FAISS_LD_LIBRARY_PATH_SET=1
# fi

# Threading configuration for Ray distributed training
# 限制多线程库的线程数，避免CPU线程竞争和资源浪费
# 这对于Ray分布式环境和sglang agent_loop非常重要，可以避免worker初始化时的线程死锁
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_INTRAOP_THREADS=1
export TORCH_INTEROP_THREADS=1

# Ray worker startup timeout configuration
# 增加 Ray worker 启动超时时间，解决 agent_loop worker 初始化时的注册超时问题
# 默认是 60 秒，对于需要加载模型和 tokenizer 的 agent_loop worker 可能不够
export RAY_worker_register_timeout_seconds=300

# Multiprocessing start method (optional, sglang will force spawn anyway)
# export RAY_START_METHOD=spawn

# Other environment variables can be added here
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export NCCL_DEBUG=INFO
