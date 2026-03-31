# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Collect SFT data by running a teacher model (e.g. Qwen3-30B) against agentic envs
# (WebShop, AlfWorld, etc.) with optional skill retrieval (deployment URL + initial skills).
# Output: parquet with "messages" column for MultiTurnSFTDataset.

import os
import sys

# Run from project root so that verl and agent_system are importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import pandas as pd
import numpy as np
import ray
from omegaconf import OmegaConf
from pprint import pprint

from verl import DataProto
from verl.utils.fs import copy_to_local


def _parse_args():
    parser = argparse.ArgumentParser(description="Collect SFT data from teacher model + env (WebShop/AlfWorld/...)")
    parser.add_argument("--config_path", default=None, help="Path to config dir (default: verl/trainer/config)")
    parser.add_argument("--config_name", default="ppo_trainer", help="Base config name")
    parser.add_argument("--output_path", required=True, help="Output parquet path for SFT data")
    parser.add_argument("--filter_success_only", action="store_true", help="Only keep successful trajectories")
    parser.add_argument("--max_trajectories", type=int, default=None, help="Max trajectories to collect (default: all)")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of rounds to run (each round = one full pass over dataloader, batch_size envs)")
    parser.add_argument("--override", nargs="*", default=[], help="Hydra overrides, e.g. env.env_name=Webshop ...")
    return parser.parse_args()


def run_collect(config):
    """Run data collection in the main process (no Ray remote) so we can use the same env/trainer setup as PPO."""
    from agent_system.environments import make_envs
    from agent_system.multi_turn_rollout import TrajectoryCollector
    from agent_system.multi_turn_rollout.utils import to_list_of_dict
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils import hf_processor, hf_tokenizer
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.ppo.ray_trainer import (
        RayPPOTrainer,
        ResourcePoolManager,
        Role,
    )
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    collect_cfg = config.get("collect") or {}
    output_path = collect_cfg.get("output_path") or config.get("collect_output_path")
    filter_success_only = collect_cfg.get("filter_success_only", False)
    max_trajectories = collect_cfg.get("max_trajectories")
    num_epochs = collect_cfg.get("num_epochs", 1)

    if not output_path:
        raise ValueError("Provide collect.output_path or --output_path")

    pprint(OmegaConf.to_container(config, resolve=True))

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    envs, val_envs = make_envs(config)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

    train_dataset = create_rl_dataset(
        config.data.train_files, config.data, tokenizer, processor
    )
    val_dataset = create_rl_dataset(
        config.data.val_files, config.data, tokenizer, processor
    )
    train_sampler = create_rl_sampler(config.data, train_dataset)
    batch_size = config.data.get("gen_batch_size", config.data.train_batch_size)
    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=config.data.get("dataloader_num_workers", 0),
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )

    # Only ActorRollout (no Critic, no Ref) for data collection
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        },
        mapping={Role.ActorRollout: global_pool_id},
    )

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        device_name=config.trainer.device,
        traj_collector=traj_collector,
        envs=envs,
        val_envs=val_envs,
    )
    trainer.init_workers()

    from train_sft.trajectory_to_sft import trajectory_to_messages  # noqa: E402

    all_rows = []
    num_collected = 0

    def _write_checkpoint(msg: str) -> None:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(all_rows).to_parquet(output_path, index=False)
        print(f"[SFT collect] Saved {len(all_rows)} trajectories to {output_path} ({msg})")

    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    # Build same pop list as ray_trainer so gen_batch has everything needed for preprocess_batch
    def get_non_tensor_pop_keys(batch):
        non_tensor = ["raw_prompt_ids", "data_source"]
        if "multi_modal_data" in getattr(batch, "non_tensor_batch", {}):
            non_tensor.append("multi_modal_data")
        if "raw_prompt" in getattr(batch, "non_tensor_batch", {}):
            non_tensor.append("raw_prompt")
        if "tools_kwargs" in getattr(batch, "non_tensor_batch", {}):
            non_tensor.append("tools_kwargs")
        if "env_kwargs" in getattr(batch, "non_tensor_batch", {}):
            non_tensor.append("env_kwargs")
        return non_tensor

    for epoch in range(num_epochs):
        for batch_dict in train_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            non_tensor_batch_keys_to_pop = get_non_tensor_pop_keys(batch)
            gen_batch = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            gen_batch.meta_info = getattr(batch, "meta_info", {})

            total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, _ = (
                trainer.traj_collector.vanilla_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=trainer.actor_rollout_wg,
                    envs=trainer.envs,
                )
            )

            success_rate = success.get("success_rate")
            if success_rate is None and success:
                success_rate = np.array([False] * len(total_batch_list))
            elif success_rate is None:
                success_rate = np.array([False] * len(total_batch_list))

            for i in range(len(total_batch_list)):
                if max_trajectories is not None and num_collected >= max_trajectories:
                    break
                sval = success_rate[i]
                won = bool(sval.item() if hasattr(sval, "item") else sval)
                if filter_success_only and not won:
                    continue
                messages = trajectory_to_messages(total_batch_list[i], tokenizer, include_inactive_steps=False)
                if not messages:
                    continue
                all_rows.append({
                    "messages": messages,
                    "reward": float(episode_rewards[i]),
                    "success": won,
                    "traj_uid": str(traj_uid[i]) if hasattr(traj_uid[i], "item") else str(traj_uid[i]),
                    "episode_length": int(episode_lengths[i]),
                })
                num_collected += 1

            if max_trajectories is not None and num_collected >= max_trajectories:
                break
        _write_checkpoint(f"after epoch {epoch + 1}/{num_epochs}")
        if max_trajectories is not None and num_collected >= max_trajectories:
            break

    return output_path


def main():
    args = _parse_args()
    config_dir = args.config_path or os.path.join(_PROJECT_ROOT, "verl", "trainer", "config")
    # Do not add collect.* to overrides: base config has struct mode, so new keys are rejected by Hydra.
    overrides = list(args.override)

    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=args.config_name, overrides=overrides)

    OmegaConf.resolve(config)
    # Add collect section after compose (base config has no "collect", and struct would block overrides).
    with open_dict(config):
        config.collect = {
            "output_path": args.output_path,
            "filter_success_only": args.filter_success_only,
            "max_trajectories": args.max_trajectories,
            "num_epochs": args.num_epochs,
        }

    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
        runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init") or {}
        re = dict(runtime_env)
        re.update(ray_init_kwargs.get("runtime_env", {}))
        ray.init(
            runtime_env=re,
            num_cpus=ray_init_kwargs.get("num_cpus"),
        )

    run_collect(config)


if __name__ == "__main__":
    main()
