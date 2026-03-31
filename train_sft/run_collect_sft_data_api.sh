#!/usr/bin/env bash
# SFT 数据收集：仅通过 OpenAI 兼容 API 调用外部模型（无本地 LLM）。
# WebShop 环境内部仍会用 Ray 起多进程 env（与脚本是否用 GPU 无关）；AlfWorld 不走该路径。
#
# 环境变量（例如 source env.sh）：
#   OPENAI_BASE_URL   OPENAI_API_KEY   OPENAI_MODEL
# 可选：TOKENIZER_MODEL_PATH  OPENAI_MAX_TOKENS  OPENAI_TEMPERATURE  OPENAI_CHAT_EXTRA_JSON
#
# --- Skill 检索（可选）---
#   COLLECT_WITH_SKILLS=1  启用与训练一致的 skills-only memory（task_skills + step_skills，embedding + 远程检索服务）。
#   需先启动检索服务，例如（与 examples/grpo_trainer/run_alfworld_skill_train_management.sh 注释一致）：
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/grpo_trainer/skill_retrieval_server.py \
#       --skills_json_path memory_data/alfworld/claude_style_skills.json \
#       --embedding_model_path $EMBEDDING_MODEL_PATH --num_gpus 8 --port 8003
#   SKILL_RETRIEVAL_SERVICE_URL  默认 http://127.0.0.1:8003/retrieve_batch
#   SKILLS_JSON_PATH               初始化经验库 JSON（task_skills/step_skills）；默认按环境选 alfworld/webshop
#   LOAD_INITIAL_SKILLS            默认 true（从 JSON 加载初始库）
#
#   检索与管理参数默认对齐 run_alfworld_skill_train_management.sh（SimUtil-UCB 等），可用环境变量覆盖：
#   MANAGEMENT_RETRIEVAL_TOP_2K  MANAGEMENT_RETRIEVAL_ALPHA  MANAGEMENT_RETRIEVAL_UCB_C
#   以及 MANAGEMENT_* 其它变量（见该脚本）。数据收集时默认关闭「动态更新技能」与 A/B 半批无 skill。
#
# 用法：
#   ./train_sft/run_collect_sft_data_api.sh [webshop|alfworld]

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "${PROJECT_DIR}/env.sh" 2>/dev/null || true

ENV_ARG="${1:-${ENV_NAME:-webshop}}"
ENV_NAME_LOWER="$(echo "$ENV_ARG" | tr '[:upper:]' '[:lower:]')"
if [[ "$ENV_NAME_LOWER" == "webshop" ]]; then
  ENV_NAME="Webshop"
elif [[ "$ENV_NAME_LOWER" == "alfworld" ]] || [[ "$ENV_NAME_LOWER" == "alfworld/alfredtwenv" ]]; then
  ENV_NAME="alfworld/AlfredTWEnv"
else
  ENV_NAME="$ENV_ARG"
fi

# ---------- 与他人 Ray 共存（同机多账号、无法 ray stop 别人时）----------
# 若遇 GCS 连接失败，先：export RAY_USER_LOCAL_CLUSTER=1 再跑本脚本（见 run_alfworld_skill_train_management.sh 注释）
if [[ "${RAY_USER_LOCAL_CLUSTER:-1}" == "1" ]]; then
  export RAY_TMPDIR="${RAY_TMPDIR:-$HOME/ray_tmp_${USER:-user}}"
  mkdir -p "$RAY_TMPDIR"
  export RAY_ADDRESS=local
fi

OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_DIR}/outputs/sft_data_api_${ENV_NAME_LOWER}_$(date +%Y%m%d_%H%M%S).parquet}"
train_data_size="${train_data_size:-16}"
group_size="${group_size:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
FILTER_SUCCESS_ONLY="${FILTER_SUCCESS_ONLY:-0}"

# 与 run_alfworld_skill_train_management.sh 对齐的管理默认值（可用环境变量覆盖）
MANAGEMENT_RETRIEVAL_TOP_2K="${MANAGEMENT_RETRIEVAL_TOP_2K:-10}"
MANAGEMENT_RETRIEVAL_ALPHA="${MANAGEMENT_RETRIEVAL_ALPHA:-0.1}"
MANAGEMENT_RETRIEVAL_UCB_C="${MANAGEMENT_RETRIEVAL_UCB_C:-0.05}"
MANAGEMENT_UTILITY_EMA_BETA="${MANAGEMENT_UTILITY_EMA_BETA:-0.5}"
MANAGEMENT_UTILITY_EMA_BETA_TASK="${MANAGEMENT_UTILITY_EMA_BETA_TASK:-0.5}"
MANAGEMENT_UTILITY_EMA_BETA_STEP="${MANAGEMENT_UTILITY_EMA_BETA_STEP:-0.5}"

python3 -m examples.data_preprocess.prepare \
  --mode 'text' \
  --train_data_size "$train_data_size" \
  --val_data_size 1

DATA_DIR="${DATA_DIR:-$HOME/data/verl-agent}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$DATA_DIR/text/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$DATA_DIR/text/test.parquet}"

CONFIG_DIR="${PROJECT_DIR}/verl/trainer/config"
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
  "actor_rollout_ref.model.path=${TOKENIZER_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
  "env.env_name=${ENV_NAME}"
  "env.seed=1"
  "env.max_steps=15"
  "env.rollout.n=$group_size"
  "env.history_length=2"
  "env.resources_per_worker.num_cpus=0.1"
  "trainer.n_gpus_per_node=1"
  "trainer.nnodes=1"
  "ray_init.num_cpus=16"
)

if [[ "$ENV_NAME_LOWER" == "webshop" ]]; then
  OVERRIDES+=(
    "env.webshop.use_small=True"
    "env.webshop.human_goals=False"
  )
fi

if [[ "$ENV_NAME_LOWER" == "alfworld" ]]; then
  OVERRIDES+=(
    "env.alfworld.eval_dataset=eval_in_distribution"
    "env.max_steps=50"
  )
fi

if [[ "${COLLECT_WITH_SKILLS:-1}" == "1" ]]; then
  SKILL_RETRIEVAL_SERVICE_URL="${SKILL_RETRIEVAL_SERVICE_URL:-http://127.0.0.1:8003/retrieve_batch}"
  # SKILLS_JSON_PATH="${SKILLS_JSON_PATH:-/data/home/zdhs0006/SkillRL/checkpoints/verl_agent_alfworld/grpo_qwen3-4b_skills_train_260321-task_step-mangement-eviction-top3-max300-o3/updated_skills_train_step160.json}"
  SKILLS_JSON_PATH="${SKILLS_JSON_PATH:-/data/home/zdhs0006/SkillRL/checkpoints/verl_agent_webshop/grpo_qwen2.5-7b_webshop_skills_management-task_step-eviction-max300-o3/updated_skills_train_step160.json}"
  LOAD_INITIAL_SKILLS="${LOAD_INITIAL_SKILLS:-true}"
  TOP_K_TASK="${TOP_K_TASK:-3}"
  TOP_K_STEP="${TOP_K_STEP:-3}"

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

EXTRA_ARGS=(--num_epochs "$NUM_EPOCHS")
[[ "${FILTER_SUCCESS_ONLY}" == "1" ]] && EXTRA_ARGS+=(--filter_success_only)
[[ -n "${MAX_TRAJECTORIES:-}" ]] && EXTRA_ARGS+=(--max_trajectories "$MAX_TRAJECTORIES")

python3 -m train_sft.collect_sft_data_api \
  --config_path "$CONFIG_DIR" \
  --config_name ppo_trainer \
  --output_path "$OUTPUT_PATH" \
  "${EXTRA_ARGS[@]}" \
  --override "${OVERRIDES[@]}"

echo "Done. SFT data (API) written to: $OUTPUT_PATH"
