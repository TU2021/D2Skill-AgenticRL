# SFT 数据收集 (train_sft)

用**教师大模型**（如 Qwen3-30B-A3B）与 **WebShop / AlfWorld** 等 agent 环境交互，生成多轮对话轨迹，并整理成 SFT 用的 `messages` 格式，用于小模型（如 Qwen3-4B）的初始化 SFT。

## 特性

- **Prompt 与现有训练一致**：WebShop 使用 `WEBSHOP_TEMPLATE_WITH_MEMORY`，AlfWorld 使用 `ALFWORLD_TEMPLATE_WITH_MEMORY`（见 `agent_system/environments/prompts/`）。
- **检索**：支持通过**部署 URL** 做 skill 检索，并指定**初始化检索库**（`skills_json_path`）。
- **环境**：支持 WebShop、AlfWorld 等所有本仓库中 `make_envs` 支持的 agentic 环境，逻辑统一。

## 依赖

1. **占位数据**：与 GRPO 相同，需先跑 `examples.data_preprocess.prepare`（text 模式），生成 `train.parquet` / `test.parquet`（仅用于 batch 条数，内容由环境提供）。
2. **检索服务**（若启用 `use_skills_only_memory`）：先启动 skill retrieval 服务，例如：
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/grpo_trainer/skill_retrieval_server.py \
     --skills_json_path memory_data/webshop/claude_style_skills.json \
     --embedding_model_path $EMBEDDING_MODEL_PATH --num_gpus 8 --port 8002
   ```

## 启动数据收集

在项目根目录执行：

```bash
# WebShop（默认）
./train_sft/run_collect_sft_data.sh

# 或显式指定环境
./train_sft/run_collect_sft_data.sh webshop
./train_sft/run_collect_sft_data.sh alfworld
```

### 环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `TEACHER_MODEL_PATH` | 教师模型路径（用于生成轨迹） | `Qwen3-30B-A3B-Instruct-2507` |
| `SKILL_RETRIEVAL_SERVICE_URL` | 检索服务 URL | `http://127.0.0.1:8002/retrieve_batch` |
| `SKILLS_JSON_PATH` | 初始化 skill 库 JSON | webshop: `memory_data/webshop/claude_style_skills.json`，alfworld: `memory_data/alfworld/claude_style_skills.json` |
| `OUTPUT_PATH` | 输出 parquet 路径 | `outputs/sft_data_<env>_<timestamp>.parquet` |
| `FILTER_SUCCESS_ONLY` | 仅保留成功轨迹 | `0`（设为 `1` 启用） |
| `MAX_TRAJECTORIES` | 最多收集轨迹数 | 不限制 |
| `NUM_EPOCHS` | 收集轮数（每轮 = 一整遍 dataloader，即 train_data_size 条 env 各跑一局） | `1` |
| `train_data_size` | 每轮 batch 大小（占位数据条数） | `32` |

示例：**跑 10 轮，只保留成功轨迹**（共最多 10×32 条轨迹，只写入 success 的）：

```bash
NUM_EPOCHS=10 FILTER_SUCCESS_ONLY=1 ./train_sft/run_collect_sft_data.sh webshop
```

示例：只保留成功轨迹、最多 5000 条：

```bash
FILTER_SUCCESS_ONLY=1 MAX_TRAJECTORIES=5000 ./train_sft/run_collect_sft_data.sh webshop
```

## 输出格式

生成的 parquet 包含列：

- `messages`: `List[{"role":"user"|"assistant","content":"..."}]`，与 `verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset` 的 `messages_key="messages"` 一致。
- `reward`, `success`, `traj_uid`, `episode_length`（可选，便于过滤或分析）。

后续可用该 parquet 做 Qwen3-4B 等多轮 SFT 训练。

## 直接调用 Python

```bash
python3 -m train_sft.collect_sft_data \
  --output_path /path/to/sft.parquet \
  --override \
    env.env_name=Webshop \
    actor_rollout_ref.model.path=/path/to/Qwen3-30B-A3B-Instruct-2507 \
    "+env.use_skills_only_memory=True" \
    "+env.skills_only_memory.skill_retrieval_service_url=http://127.0.0.1:8002/retrieve_batch" \
    "+env.skills_only_memory.skills_json_path=memory_data/webshop/claude_style_skills.json" \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet
```

## 外部 API（无本地推理）

不加载教师模型，仅用 **OpenAI 兼容 HTTP API**（`OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`），与 `env.sh` 用法一致。

**默认（无 skill）**：`COLLECT_WITH_SKILLS` 未设置或为 `0`，等价 `+env.use_skills_only_memory=False`。

**带 skill 检索（task_skills + step_skills，embedding + 远程服务）**：与 `examples/grpo_trainer/run_alfworld_skill_train_management.sh` 同类配置，需先启动 `skill_retrieval_server.py`，再例如：

```bash
export COLLECT_WITH_SKILLS=1
export SKILLS_JSON_PATH=memory_data/alfworld/claude_style_skills.json   # 初始化经验库
export SKILL_RETRIEVAL_SERVICE_URL=http://127.0.0.1:8003/retrieve_batch
./train_sft/run_collect_sft_data_api.sh alfworld
```

检索与管理超参（`retrieval_top_2k` / `retrieval_alpha` / `retrieval_ucb_c` 等）可用 `MANAGEMENT_*` 环境变量覆盖，默认值与上述 management 训练脚本对齐；数据收集侧默认 `enable_dynamic_update=False`（不写回技能库）、`management.baseline_ab_split=false`（整批均注入检索到的 skill）。

```bash
source env.sh
./train_sft/run_collect_sft_data_api.sh webshop
# 或 alfworld
```

占位数据仍用 `examples.data_preprocess.prepare`；**需本机可拉取** `TOKENIZER_MODEL_PATH`（默认 `Qwen/Qwen2.5-0.5B-Instruct`）以构建 dataloader。实现见 `train_sft/collect_sft_data_api.py`。

## API 测试集评估（AlfWorld / WebShop）

如果你要测试**开源模型（通过 OpenAI 兼容 API 暴露）**在 test env 的性能，并把总指标与子任务指标写入 SwanLab，可用：

```bash
source env.sh

# AlfWorld test env 评估
./train_sft/run_eval_api_test_env.sh alfworld

# WebShop test env 评估
./train_sft/run_eval_api_test_env.sh webshop
```

对应实现：

- 启动脚本：`train_sft/run_eval_api_test_env.sh`
- 核心逻辑：`train_sft/eval_api_test_env.py`

指标说明（与 `run_alfworld_origin.sh` 的验证逻辑一致地复用 `val_envs.success_evaluator`）：

- `val/success_rate`：总体成功率
- AlfWorld 还会自动包含各子任务成功率（例如 `val/pick_and_place_success_rate` 等）
- WebShop 还会包含 `val/webshop_task_score (not success_rate)` 的均值
- 另含 `val/api_eval/episode_reward_mean`、`val/api_eval/episode_length_mean` 等统计

常用环境变量：

- 必填：`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`
- SwanLab：`SWANLAB_API_KEY`、`SWANLAB_PROJECT`、`SWANLAB_EXPERIMENT`
- 评估规模：`NUM_EVAL_BATCHES`、`val_data_size`
- 环境控制：`ALFWORLD_EVAL_DATASET`、`WEBSHOP_USE_SMALL`
