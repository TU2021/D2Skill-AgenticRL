#!/usr/bin/env bash
# Evaluate OpenAI-compatible API model on test env (alfworld/webshop) and log to swanlab.
#
# Usage:
#   ./train_sft/run_eval_api_test_env.sh [alfworld|webshop]
#
# 环境变量（例如 source env.sh）：
#   OPENAI_BASE_URL   OPENAI_API_KEY   OPENAI_MODEL
# 可选：TOKENIZER_MODEL_PATH  API_MAX_WORKERS  OPENAI_TEMPERATURE  OPENAI_CHAT_EXTRA_JSON
#
# --- 带 Skill JSON 评估（可选，与 run_collect_sft_data_api.sh 对齐）---
#   EVAL_WITH_SKILLS=1  启用 skills-only memory：从 JSON 加载 task_skills/step_skills，并通过 embedding 检索服务注入 prompt。
#   需先启动检索服务，例如：
#     CUDA_VISIBLE_DEVICES=0 python examples/grpo_trainer/skill_retrieval_server.py \
#       --skills_json_path <你的 skills.json> \
#       --embedding_model_path "$EMBEDDING_MODEL_PATH" --num_gpus 1 --port 8003
#   SKILL_RETRIEVAL_SERVICE_URL  默认 http://127.0.0.1:8003/retrieve_batch
#   SKILLS_JSON_PATH             初始技能库 JSON；未设置时按环境给默认路径（见下方），请按需覆盖为你的 checkpoint
#   LOAD_INITIAL_SKILLS          默认 true
#   管理/检索超参：MANAGEMENT_RETRIEVAL_TOP_2K 等（与 run_collect_sft_data_api.sh 一致）
#
# 示例（AlfWorld + 训练导出的 skills）：
#   EVAL_WITH_SKILLS=1 SKILLS_JSON_PATH=/path/to/updated_skills_train_stepXX.json \
#     ./train_sft/run_eval_api_test_env.sh alfworld
#
# 兼容：若曾使用 USE_SKILLS_ONLY_MEMORY=True，与 EVAL_WITH_SKILLS=1 同样会走完整 skill 配置（需检索服务 + JSON）。

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "${PROJECT_DIR}/env.sh" 2>/dev/null || true

ENV_ARG="${1:-${ENV_NAME:-alfworld}}"
ENV_NAME_LOWER="$(echo "$ENV_ARG" | tr '[:upper:]' '[:lower:]')"
if [[ "$ENV_NAME_LOWER" == "webshop" ]]; then
  ENV_NAME="Webshop"
elif [[ "$ENV_NAME_LOWER" == "alfworld" ]] || [[ "$ENV_NAME_LOWER" == "alfworld/alfredtwenv" ]]; then
  ENV_NAME="alfworld/AlfredTWEnv"
else
  ENV_NAME="$ENV_ARG"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is empty. Please export OPENAI_API_KEY first."
  exit 1
fi

if [[ "${RAY_USER_LOCAL_CLUSTER:-1}" == "1" ]]; then
  export RAY_TMPDIR="${RAY_TMPDIR:-$HOME/ray_tmp_${USER:-user}}"
  mkdir -p "$RAY_TMPDIR"
  export RAY_ADDRESS=local
fi

val_data_size="${val_data_size:-128}"
NUM_EVAL_BATCHES="${NUM_EVAL_BATCHES:-1}"
MAX_STEPS="${MAX_STEPS:-}"
API_MAX_WORKERS="${API_MAX_WORKERS:-16}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"

python3 -m examples.data_preprocess.prepare \
  --mode 'text' \
  --train_data_size 2 \
  --val_data_size "$val_data_size"

DATA_DIR="${DATA_DIR:-$HOME/data/verl-agent}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$DATA_DIR/text/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$DATA_DIR/text/test.parquet}"

OVERRIDES=(
  "data.train_files=$TRAIN_PARQUET"
  "data.val_files=$VAL_PARQUET"
  "data.train_batch_size=2"
  "data.val_batch_size=$val_data_size"
  "data.max_prompt_length=4096"
  "data.max_response_length=1024"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.return_raw_chat=True"
  "env.env_name=$ENV_NAME"
  "env.seed=${SEED:-0}"
  "env.rollout.n=1"
  "env.history_length=${HISTORY_LENGTH:-2}"
  "env.resources_per_worker.num_cpus=${NUM_CPUS_PER_ENV_WORKER:-0.1}"
)

if [[ -n "$MAX_STEPS" ]]; then
  OVERRIDES+=("env.max_steps=$MAX_STEPS")
elif [[ "$ENV_NAME_LOWER" == "alfworld" ]]; then
  OVERRIDES+=("env.max_steps=50")
else
  OVERRIDES+=("env.max_steps=15")
fi

if [[ "$ENV_NAME_LOWER" == "webshop" ]]; then
  OVERRIDES+=(
    "env.webshop.use_small=${WEBSHOP_USE_SMALL:-True}"
    "env.webshop.human_goals=${WEBSHOP_HUMAN_GOALS:-False}"
  )
fi

if [[ "$ENV_NAME_LOWER" == "alfworld" ]]; then
  OVERRIDES+=(
    "env.alfworld.eval_dataset=${ALFWORLD_EVAL_DATASET:-eval_in_distribution}"
  )
fi

# ---------- 与 run_collect_sft_data_api.sh 一致的 skills-only 评估（默认关闭）----------
# 注意：SKILLS_JSON_PATH 必须是含 task_skills / step_skills 的技能库（如 updated_skills_train_step*.json），
# 不可用 failed_trajectories_*.json（轨迹列表，载入会报错或无法检索）。
if [[ "${EVAL_WITH_SKILLS:-1}" == "1" ]] || [[ "${USE_SKILLS_ONLY_MEMORY:-True}" == "True" ]] || [[ "${USE_SKILLS_ONLY_MEMORY:-True}" == "true" ]]; then
  SKILL_RETRIEVAL_SERVICE_URL="${SKILL_RETRIEVAL_SERVICE_URL:-http://127.0.0.1:8003/retrieve_batch}"
  if [[ -z "${SKILLS_JSON_PATH:-}" ]]; then
    if [[ "$ENV_NAME_LOWER" == "alfworld" ]]; then
      SKILLS_JSON_PATH="/data/home/zdhs0006/SkillRL/checkpoints/verl_agent_alfworld/grpo_qwen3-4b_skills_train_260321-task_step-mangement-eviction-top3-max300-o3/updated_skills_train_step161.json"
    elif [[ "$ENV_NAME_LOWER" == "webshop" ]]; then
      SKILLS_JSON_PATH="${PROJECT_DIR}/checkpoints/verl_agent_webshop/grpo_qwen2.5-7b_webshop_skills_management-task_step-eviction-max300-o3/updated_skills_train_step160.json"
    else
      echo "[ERROR] Skill eval enabled but SKILLS_JSON_PATH is unset and no default for env '${ENV_NAME_LOWER}'."
      exit 1
    fi
  fi
  LOAD_INITIAL_SKILLS="${LOAD_INITIAL_SKILLS:-true}"
  TOP_K_TASK="${TOP_K_TASK:-1}"
  TOP_K_STEP="${TOP_K_STEP:-1}"

  MANAGEMENT_RETRIEVAL_TOP_2K="${MANAGEMENT_RETRIEVAL_TOP_2K:-10}"
  MANAGEMENT_RETRIEVAL_ALPHA="${MANAGEMENT_RETRIEVAL_ALPHA:-0.1}"
  MANAGEMENT_RETRIEVAL_UCB_C="${MANAGEMENT_RETRIEVAL_UCB_C:-0.05}"
  MANAGEMENT_UTILITY_EMA_BETA="${MANAGEMENT_UTILITY_EMA_BETA:-0.5}"
  MANAGEMENT_UTILITY_EMA_BETA_TASK="${MANAGEMENT_UTILITY_EMA_BETA_TASK:-0.5}"
  MANAGEMENT_UTILITY_EMA_BETA_STEP="${MANAGEMENT_UTILITY_EMA_BETA_STEP:-0.5}"

  OVERRIDES+=(
    "+env.use_skills_only_memory=True"
    "+env.skills_only_memory.skills_json_path=${SKILLS_JSON_PATH}"
    "+env.skills_only_memory.retrieval_mode=embedding"
    "+env.skills_only_memory.skill_retrieval_service_url=${SKILL_RETRIEVAL_SERVICE_URL}"
    "+env.skills_only_memory.skill_text_for_retrieval=when_to_apply"
    "+env.skills_only_memory.load_initial_skills=${LOAD_INITIAL_SKILLS}"
    "+env.skills_only_memory.similarity_threshold=0.7"
    "+env.skills_only_memory.top_k_task=${TOP_K_TASK}"
    "+env.skills_only_memory.top_k_step=${TOP_K_STEP}"
    "+env.skills_only_memory.skill_gen_mode=task_step"
    "+env.skills_only_memory.max_concurrent=10"
    "+env.skills_only_memory.per_step_retrieval=True"
    "+env.skills_only_memory.retrieval_obs=True"
    "+env.skills_only_memory.enable_dynamic_update=False"
    "+env.skills_only_memory.record_retrieved_skills=True"
    "+env.skills_only_memory.enable_dynamic_management=True"
    "+env.skills_only_memory.management.baseline_ab_split=false"
    "+env.skills_only_memory.management.retrieval_top_2k=${MANAGEMENT_RETRIEVAL_TOP_2K}"
    "+env.skills_only_memory.management.retrieval_alpha=${MANAGEMENT_RETRIEVAL_ALPHA}"
    "+env.skills_only_memory.management.retrieval_ucb_c=${MANAGEMENT_RETRIEVAL_UCB_C}"
    "+env.skills_only_memory.management.utility_ema_beta=${MANAGEMENT_UTILITY_EMA_BETA}"
    "+env.skills_only_memory.management.utility_ema_beta_task=${MANAGEMENT_UTILITY_EMA_BETA_TASK}"
    "+env.skills_only_memory.management.utility_ema_beta_step=${MANAGEMENT_UTILITY_EMA_BETA_STEP}"
  )
else
  OVERRIDES+=("+env.use_skills_only_memory=False")
fi

export API_MAX_WORKERS
export TOKENIZER_MODEL_PATH
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-verl_agent_${ENV_NAME_LOWER}_api_eval}"
export SWANLAB_EXPERIMENT="${SWANLAB_EXPERIMENT:-api_eval_${ENV_NAME_LOWER}_$(date +%Y%m%d_%H%M%S)}"

python3 -m train_sft.eval_api_test_env \
  --config_path "${PROJECT_DIR}/verl/trainer/config" \
  --config_name ppo_trainer \
  --num_eval_batches "$NUM_EVAL_BATCHES" \
  --override "${OVERRIDES[@]}"

