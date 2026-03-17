set -x

ENGINE=${1:-vllm}

PROJECT_DIR="$(pwd)"
source ${0%/*}/../../env.sh

train_data_size=256
val_data_size=512
group_size=4

# TRAIN_DATA="/data/home/zdhs0006/data/searchR1_processed_direct/train.parquet"
# VAL_DATA="/data/home/zdhs0006/data/searchR1_processed_direct/test.parquet"

TRAIN_DATA="/data/home/zdhs0006/data_verl_agent/search/test_512.parquet"
VAL_DATA="/data/home/zdhs0006/data_verl_agent/search/test_512.parquet"

# export WANDB_API_KEY=""
ACTOR_MODEL_PATH="/data/home/zdhs0006/SkillRL/models/Search-7B-SFT"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=5000 \
    data.max_response_length=700 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url='http://127.0.0.1:8005/retrieve' \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/search/claude_style_skills_search.json \
    +env.skills_only_memory.top_k=6 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name='grpo_qwen2.5_7b_instruct' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.validation_data_dir="${PROJECT_DIR}/swanlog/validation_generations" \
    trainer.log_val_generations=50 \
    trainer.save_freq=50 \
    trainer.test_freq=200 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    ray_init.num_cpus=96 \
    trainer.ray_wait_register_center_timeout=600 \
    2>&1 | tee run_search_skills.log
