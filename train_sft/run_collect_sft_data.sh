#!/usr/bin/env bash
# Collect SFT data: teacher model (e.g. Qwen3-30B) interacts with WebShop / AlfWorld (or other agentic envs)
# with WEBSHOP_TEMPLATE_WITH_MEMORY / ALFWORLD_TEMPLATE_WITH_MEMORY and deployment URL retrieval + initial skills.
# Output: parquet with "messages" for MultiTurnSFTDataset (Qwen3-4B SFT init).
#
# Usage:
#   ./train_sft/run_collect_sft_data.sh [webshop|alfworld]   # env name
#   Or set ENV_NAME=webshop|alfworld and run ./train_sft/run_collect_sft_data.sh
#
# Prerequisites:
#   - Skill retrieval service running (embedding server) if using retrieval:
#     e.g. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/grpo_trainer/skill_retrieval_server.py \
#          --skills_json_path memory_data/webshop/claude_style_skills.json \
#          --embedding_model_path $EMBEDDING_MODEL_PATH --num_gpus 8 --port 8002
#   - Data placeholder: run prepare (text mode) so train.parquet exists (val is minimal placeholder only).

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "${PROJECT_DIR}/env.sh" 2>/dev/null || true

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-600}"

# ---------- Env selection (webshop | alfworld | or full name e.g. alfworld/AlfredTWEnv) ----------
ENV_ARG="${1:-${ENV_NAME:-webshop}}"
ENV_NAME_LOWER="$(echo "$ENV_ARG" | tr '[:upper:]' '[:lower:]')"
if [[ "$ENV_NAME_LOWER" == "webshop" ]]; then
  ENV_NAME="Webshop"
elif [[ "$ENV_NAME_LOWER" == "alfworld" ]] || [[ "$ENV_NAME_LOWER" == "alfworld/alfredtwenv" ]]; then
  ENV_NAME="alfworld/AlfredTWEnv"
else
  ENV_NAME="$ENV_ARG"
fi

# ---------- Teacher model (large model used to generate SFT data) ----------
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/data/group/project3/project3_cluster3_data/hf_models/Qwen3-30B-A3B-Instruct-2507}"

# ---------- Retrieval: deployment URL + initial skills JSON ----------
SKILL_RETRIEVAL_SERVICE_URL="${SKILL_RETRIEVAL_SERVICE_URL:-http://127.0.0.1:8004/retrieve_batch}"
if [[ "$ENV_NAME_LOWER" == "webshop" ]]; then
  SKILLS_JSON_PATH="${SKILLS_JSON_PATH:-/data/home/zdhs0006/SkillRL/checkpoints/verl_agent_webshop/grpo_qwen3_4b_webshop_skills_train-continue80-260312-summarize_success-zero-st0.6-retrieval_obs/updated_skills_train_step60.json}"
elif [[ "$ENV_NAME_LOWER" == "alfworld" ]]; then
  SKILLS_JSON_PATH="${SKILLS_JSON_PATH:-memory_data/alfworld/claude_style_skills.json}"
else
  SKILLS_JSON_PATH="${SKILLS_JSON_PATH:-}"
fi

# ---------- Output and train data size (no val: SFT collect is train-only; val is minimal placeholder for API) ----------
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_DIR}/outputs/sft_data_${ENV_NAME_LOWER}_$(date +%Y%m%d_%H%M%S).parquet}"
train_data_size="${train_data_size:-32}"
group_size="${group_size:-1}"

# ---------- Collect options: rounds and success-only (set in script or override with env) ----------
NUM_EPOCHS="${NUM_EPOCHS:-20}"           # 跑 10 轮，每轮 train_data_size 条轨迹
FILTER_SUCCESS_ONLY="${FILTER_SUCCESS_ONLY:-1}"   # 只保留成功轨迹

# ---------- Prepare placeholder data: train only; val parquet minimal (1 row) for loader API ----------
python3 -m examples.data_preprocess.prepare \
  --mode 'text' \
  --train_data_size "$train_data_size" \
  --val_data_size 1

DATA_DIR="${DATA_DIR:-$HOME/data/verl-agent}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$DATA_DIR/text/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$DATA_DIR/text/test.parquet}"

# ---------- Build overrides: env, model, retrieval (WITH_MEMORY prompt), collect options ----------
OVERRIDES=(
  "algorithm.adv_estimator=grpo"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "data.train_files=$TRAIN_PARQUET"
  "data.val_files=$VAL_PARQUET"
  "data.train_batch_size=$train_data_size"
  "data.val_batch_size=1"
  "data.max_prompt_length=4096"
  "data.max_response_length=1024"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.return_raw_chat=True"
  "actor_rollout_ref.model.path=$TEACHER_MODEL_PATH"
  "actor_rollout_ref.actor.ppo_mini_batch_size=128"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8"
  "env.env_name=${ENV_NAME}"
  "env.seed=0"
  "env.max_steps=15"
  "env.rollout.n=$group_size"
  "env.history_length=2"
  "env.resources_per_worker.num_cpus=0.1"
  "+env.use_skills_only_memory=True"
  "+env.skills_only_memory.skills_json_path=$SKILLS_JSON_PATH"
  "+env.skills_only_memory.retrieval_mode=embedding"
  "+env.skills_only_memory.skill_retrieval_service_url=$SKILL_RETRIEVAL_SERVICE_URL"
  "+env.skills_only_memory.skill_text_for_retrieval=when_to_apply"
  "+env.skills_only_memory.load_initial_skills=True"
  "+env.skills_only_memory.similarity_threshold=0.7"
  "+env.skills_only_memory.top_k=5"
  "+env.skills_only_memory.per_step_retrieval=True"
  "+env.skills_only_memory.retrieval_obs=True"
  "trainer.n_gpus_per_node=8"
  "trainer.nnodes=1"
  "ray_init.num_cpus=80"
)

# WebShop-specific
if [[ "$ENV_NAME_LOWER" == "webshop" ]]; then
  OVERRIDES+=(
    "env.webshop.use_small=True"
    "env.webshop.human_goals=False"
  )
fi

# AlfWorld-specific
if [[ "$ENV_NAME_LOWER" == "alfworld" ]]; then
  OVERRIDES+=(
    "env.alfworld.eval_dataset=eval_in_distribution"
  )
  OVERRIDES+=("env.max_steps=50")
fi

# Pass num_epochs, filter_success_only (defaults set above); optionally MAX_TRAJECTORIES
EXTRA_ARGS=(--num_epochs "$NUM_EPOCHS")
[[ "${FILTER_SUCCESS_ONLY}" == "1" ]] && EXTRA_ARGS+=(--filter_success_only)
[[ -n "${MAX_TRAJECTORIES:-}" ]] && EXTRA_ARGS+=(--max_trajectories "$MAX_TRAJECTORIES")

CONFIG_DIR="${PROJECT_DIR}/verl/trainer/config"
python3 -m train_sft.collect_sft_data \
  --config_path "$CONFIG_DIR" \
  --config_name ppo_trainer \
  --output_path "$OUTPUT_PATH" \
  "${EXTRA_ARGS[@]}" \
  --override "${OVERRIDES[@]}"

echo "Done. SFT data written to: $OUTPUT_PATH"
