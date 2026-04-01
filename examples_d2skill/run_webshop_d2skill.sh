#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_webshop_d2skill.sh [ENGINE=vllm] [Hydra overrides...]

Example:
  ./run_webshop_d2skill.sh vllm +env.skills_only_memory.management.eviction_enabled=true
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ENGINE="${1:-vllm}"
shift 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/env.sh"

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_worker_register_timeout_seconds=600

# Run Ray in per-user local mode when you cannot stop others' clusters
export RAY_USER_LOCAL_CLUSTER="${RAY_USER_LOCAL_CLUSTER:-1}"
if [[ "${RAY_USER_LOCAL_CLUSTER}" == "1" ]]; then
  export RAY_TMPDIR="${RAY_TMPDIR:-$HOME/ray_tmp_${USER:-user}}"
  mkdir -p "$RAY_TMPDIR"
  export RAY_ADDRESS=local
fi

# skill retrieval service (separate process). Default port aligns with comments below.
SKILL_RETRIEVAL_SERVICE_URL="http://127.0.0.1:8003/retrieve_batch"

num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=128
group_size=8

ACTOR_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=1.0 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=None \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.skill_retrieval_service_url="$SKILL_RETRIEVAL_SERVICE_URL" \
    +env.skills_only_memory.skill_text_for_retrieval=when_to_apply \
    +env.skills_only_memory.load_initial_skills=False \
    +env.skills_only_memory.similarity_threshold=0.7 \
    +env.skills_only_memory.top_k_task=1 \
    +env.skills_only_memory.top_k_step=1 \
    +env.skills_only_memory.skill_gen_mode=task_step \
    +env.skills_only_memory.max_concurrent=10 \
    +env.skills_only_memory.enable_dynamic_update=True \
    +env.skills_only_memory.update_save_traj=True \
    +env.skills_only_memory.update_source=train \
    +env.skills_only_memory.skill_update_group_success_rate_threshold=0.5 \
    +env.skills_only_memory.max_trajectories_for_skill_update=10 \
    +env.skills_only_memory.record_retrieved_skills=True \
    +env.skills_only_memory.enable_dynamic_management=True \
    +env.skills_only_memory.management.baseline_ab_split=true \
    +env.skills_only_memory.management.baseline_ab_ratio=0.5 \
    +env.skills_only_memory.management.utility_ema_beta=0.5 \
    +env.skills_only_memory.management.utility_ema_beta_task=0.5 \
    +env.skills_only_memory.management.utility_ema_beta_step=0.5 \
    +env.skills_only_memory.management.retrieval_top_2k=10 \
    +env.skills_only_memory.management.retrieval_alpha=0.1 \
    +env.skills_only_memory.management.retrieval_ucb_c=0.05 \
    +env.skills_only_memory.management.intrinsic_reward_enabled=true \
    +env.skills_only_memory.management.intrinsic_reward_coefficient=1 \
    +env.skills_only_memory.management.credit_use_baseline=true \
    +env.skills_only_memory.management.eviction_enabled=true \
    +env.skills_only_memory.management.eviction_max_task_skills=300 \
    +env.skills_only_memory.management.eviction_max_step_skills=300 \
    +env.skills_only_memory.management.eviction_protect_recent_steps=10 \
    +env.skills_only_memory.management.eviction_score_c=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='grpo_qwen2.5-7b_skills_d2skill' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.log_val_generations=10 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.val_before_train=True \
    trainer.ray_wait_register_center_timeout=3600 \
    ray_init.num_cpus=80 \
    "$@" \
    2>&1 | tee run_webshop_skills_management.log
