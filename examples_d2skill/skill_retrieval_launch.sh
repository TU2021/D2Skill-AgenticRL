#!/usr/bin/env bash
# Launch skill retrieval server (embedding mode), 8-GPU by default.
# Run:  bash examples/grpo_trainer/skill_retrieval_launch.sh
# Then in training config set: env.skills_only_memory.skill_retrieval_service_url=http://127.0.0.1:8003/retrieve_batch

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 8 cards by default; use CUDA_VISIBLE_DEVICES to limit (e.g. 0,1,2,3 for 4 cards)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
SKILLS_JSON=None
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/group/project3/project3_cluster3_data/hf_models/Qwen3-Embedding-0.6B}"
# EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/group/project3/project3_cluster3_data/hf_models/Qwen3-Embedding-4B}"
PORT="${PORT:-8003}"

# Set NO_LOAD_INITIAL_SKILLS=1 to start with empty skill bank (skills loaded later via /reload_skills)
LOAD_INITIAL_SKILLS="${NO_LOAD_INITIAL_SKILLS:-0}"

ARGS=(
  --device cuda
  --port "$PORT"
  --num_gpus "$NUM_GPUS"
  --embedding_model_path "$EMBEDDING_MODEL"
)
if [[ "$LOAD_INITIAL_SKILLS" == "0" ]]; then
  ARGS+=(--no_load_initial_skills)
else
  ARGS+=(--skills_json_path "$SKILLS_JSON")
fi

python examples_d2skill/skill_retrieval_server.py "${ARGS[@]}" "$@"