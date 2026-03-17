# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import random
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch
try:
    from agent_system.skills_only_config import is_dynamic_management_enabled
except ImportError:
    def is_dynamic_management_enabled(_):
        return False

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}
        val_retrieved_list = []  # collect retrieved_memories per validation batch for logging
        val_per_step_retrieved_list = []  # per-step retrieved skills per batch (when per_step_retrieval is True)

        # Lists to collect samples for the table
        # We'll collect full dialogue histories per trajectory instead of per step
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        # 每轮 val：对每个失败任务用 refined_trajectory/query_texts 收集，最后统一生成 skill
        all_failed_trajectories = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs (initial prompts for each trajectory)
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            print('validation generation end')
            if hasattr(self.val_envs, "retrieved_memories") and self.val_envs.retrieved_memories is not None:
                val_retrieved_list.append(deepcopy(self.val_envs.retrieved_memories))
                val_per_step_retrieved_list.append(
                    test_output_gen_batch.non_tensor_batch.get("per_step_retrieved_by_traj")
                )
            # Remove trajectory-level key so val_reward_fn / data[i] do not index it by step -> IndexError
            if hasattr(test_output_gen_batch, "non_tensor_batch"):
                test_output_gen_batch.non_tensor_batch.pop("per_step_retrieved_by_traj", None)
            del test_batch
            test_batch = test_output_gen_batch
            
            # Extract full multi-turn dialogue history by grouping by traj_uid
            traj_uids = test_output_gen_batch.non_tensor_batch.get('traj_uid', [])
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            
            # Try to extract observations from input_ids if available
            # For environment interactions, we want to show: Observation -> Action
            input_ids_list = test_output_gen_batch.batch.get("input_ids", None)
            observation_texts = []
            if input_ids_list is not None:
                # Decode input_ids to get observations (these contain the environment state)
                for ids in input_ids_list:
                    obs_text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    observation_texts.append(obs_text)
            else:
                # Fallback: use empty observations
                observation_texts = [""] * len(output_texts)
            
            # Group outputs and observations by trajectory to form complete dialogue history
            traj_to_turns = {}  # {traj_uid: [(obs, action), ...]}
            for i, (uid, action) in enumerate(zip(traj_uids, output_texts)):
                if uid not in traj_to_turns:
                    traj_to_turns[uid] = []
                obs = observation_texts[i] if i < len(observation_texts) else ""
                traj_to_turns[uid].append((obs, action))
            
            # Build full dialogue history for each unique trajectory
            # Format: "Initial: <prompt>\n\nTurn 1:\n  Observation: <obs1>\n  Action: <action1>\nTurn 2: ..."
            unique_traj_uids = list(set(traj_uids))
            # uid -> trajectory index (0-based order of first occurrence); traj_uid can be int or str (e.g. UUID)
            uid_to_traj_idx = {}
            for u in traj_uids:
                if u not in uid_to_traj_idx:
                    uid_to_traj_idx[u] = len(uid_to_traj_idx)
            
            # Map initial prompts to trajectories (by first-occurrence order so each trajectory gets its real initial prompt)
            for idx, uid in enumerate(unique_traj_uids):
                traj_idx = uid_to_traj_idx.get(uid, 0)
                if input_texts and 0 <= traj_idx < len(input_texts):
                    initial_prompt = input_texts[traj_idx]
                else:
                    initial_prompt = input_texts[0] if input_texts else "N/A"

                # Build full dialogue with observations and actions
                if uid in traj_to_turns:
                    dialogue_parts = [f"Initial Prompt: {initial_prompt}\n"]
                    for turn_idx, (obs, action) in enumerate(traj_to_turns[uid]):
                        if obs.strip():  # If observation exists, show Observation -> Action
                            dialogue_parts.append(f"Turn {turn_idx + 1}:")
                            dialogue_parts.append(f"  Observation: {obs[:500]}...")  # Truncate long observations
                            dialogue_parts.append(f"  Action: {action}")
                        else:  # Fallback: just show action
                            dialogue_parts.append(f"Turn {turn_idx + 1}: {action}")
                    full_dialogue = "\n".join(dialogue_parts)
                else:
                    full_dialogue = f"Initial Prompt: {initial_prompt}\n(No responses)"
                
                sample_inputs.append(initial_prompt)  # Keep initial prompt for reference
                sample_outputs.append(full_dialogue)  # Full dialogue history with observations

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            # reward_tensor: (num_steps,) or (num_steps, response_len); need one score per step for zip with traj_uids
            if reward_tensor.dim() == 1:
                scores = reward_tensor.cpu().tolist()
            else:
                scores = reward_tensor.sum(-1).cpu().tolist()
            
            # Map scores to unique trajectories (use last step's score per traj_uid = episode reward)
            unique_traj_uids = list(set(traj_uids))
            traj_to_score = {}
            for uid, score in zip(traj_uids, scores):
                traj_to_score[uid] = score  # overwrite so we keep the last step's reward (final episode score)
            
            # Add scores for each unique trajectory
            for uid in unique_traj_uids:
                sample_scores.append(traj_to_score.get(uid, 0.0))

            # 本 batch 内用 batch 收集失败轨迹（带 refined_trajectory / query_texts），供本轮 val 统一生成 skill
            n_steps = len(traj_uids)
            input_per_step = []
            for i in range(n_steps):
                uid_i = traj_uids[i]
                traj_idx = uid_to_traj_idx.get(uid_i, 0)
                if input_texts and 0 <= traj_idx < len(input_texts):
                    input_per_step.append(input_texts[traj_idx])
                else:
                    input_per_step.append(input_texts[0] if input_texts else "N/A")
            scores_per_step = reward_tensor.cpu().tolist() if reward_tensor.dim() == 1 else reward_tensor.sum(-1).cpu().tolist()
            failed_this_batch = self._collect_failed_trajectories(
                input_per_step, output_texts, scores_per_step, batch=test_batch
            )
            all_failed_trajectories.extend(failed_this_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        if val_retrieved_list:
            self._record_retrieved_skills(
                step=self.global_steps,
                phase="validation",
                memories_list=val_retrieved_list,
                per_step_retrievals_list=val_per_step_retrieved_list if val_per_step_retrieved_list else None,
            )

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        # === Skill Bank 动态更新：每轮 val 对每个失败任务分别总结并生成 skill（用 refined_trajectory/query_texts），最后拼在一起
        if self.config.env.get('skills_only_memory', {}).get('enable_dynamic_update', False):
            update_source = self.config.env.get('skills_only_memory', {}).get('update_source', 'validation')
            if update_source in ['validation', 'all'] and all_failed_trajectories:
                self._update_skills_from_validation(
                    sample_inputs=sample_inputs,
                    sample_outputs=sample_outputs,
                    sample_scores=sample_scores,
                    success_rate=success_rate,
                    failed_trajectories=all_failed_trajectories,
                )

        return metric_dict

    def _update_skills_from_validation(
        self,
        sample_inputs: list,
        sample_outputs: list,
        sample_scores: list,
        success_rate: dict,
        failed_trajectories: list = None,
    ):
        """
        根据 validation 结果更新 skill bank。

        若传入 failed_trajectories（每轮 val 已按 batch 用 refined_trajectory/query_texts 收集），
        则直接使用，对每个失败任务分别总结并生成 skill 后拼在一起；
        否则沿用旧逻辑：仅当某任务类型成功率低于阈值时再收集失败轨迹并更新。
        """
        update_config = self.config.env.skills_only_memory

        if failed_trajectories is None:
            threshold = update_config.get('update_threshold', 0.5)
            needs_update = False
            low_success_tasks = []
            for task_key, rate in (success_rate or {}).items():
                if not task_key or task_key == 'success_rate':
                    continue
                if rate < threshold:
                    needs_update = True
                    task_type = task_key.replace('_success_rate', '')
                    if task_type:
                        low_success_tasks.append(task_type)
            if not needs_update:
                print(f"[SkillUpdate] All task success rates above {threshold}, skipping update")
                return
            print(f"[SkillUpdate] Low success tasks: {low_success_tasks}, triggering skill update...")
            failed_trajectories = self._collect_failed_trajectories(
                sample_inputs, sample_outputs, sample_scores
            )
        else:
            print(f"[SkillUpdate] Using {len(failed_trajectories)} pre-collected failed trajectories (refined_trajectory/query_texts) for this val round")

        if not failed_trajectories:
            print("[SkillUpdate] No failed trajectories found")
            return

        # 初始化 SkillUpdater (lazy init, 使用 Azure OpenAI o3)
        if not hasattr(self, 'skill_updater'):
            from agent_system.memory.skill_updater import SkillUpdater
            self.skill_updater = SkillUpdater(
                max_new_skills_per_update=update_config.get('max_new_skills', 3),
                skill_gen_mode=update_config.get('skill_gen_mode', 'direct'),
                summarizer_model=update_config.get('summarizer_model'),
                summarizer_max_concurrent=update_config.get('summarizer_max_concurrent', 4),
                retrieval_obs=update_config.get('retrieval_obs', False),
            )

        # 获取当前 skills
        retrieval_memory = self.val_envs.retrieval_memory
        if retrieval_memory is None:
            print("[SkillUpdate] No retrieval_memory found in val_envs")
            return

        # 保存失败轨迹到磁盘（如果配置启用）
        save_traj = update_config.get('update_save_traj', False)
        save_dir = self.config.trainer.get('default_local_dir', './outputs')
        if save_traj:
            failed_traj_path = os.path.join(save_dir, f'failed_trajectories_step{self.global_steps}.json')
            try:
                import json
                os.makedirs(save_dir, exist_ok=True)
                
                # 初始化 SkillUpdater（如果还没有）以便使用其格式化方法（与上面参数一致）
                if not hasattr(self, 'skill_updater'):
                    from agent_system.memory.skill_updater import SkillUpdater
                    self.skill_updater = SkillUpdater(
                        max_new_skills_per_update=update_config.get('max_new_skills', 3),
                        skill_gen_mode=update_config.get('skill_gen_mode', 'direct'),
                        summarizer_model=update_config.get('summarizer_model'),
                        summarizer_max_concurrent=update_config.get('summarizer_max_concurrent', 4),
                        retrieval_obs=update_config.get('retrieval_obs', False),
                    )

                # 格式化轨迹，使其与传给 LLM 的内容一致
                formatted_trajectories = []
                for traj in failed_trajectories:
                    # 只保存 task / task_type / refined_trajectory 等，不再存 original_trajectory、llm_prompt_section
                    formatted_traj = {
                        'task': traj['task'],
                        'task_type': traj['task_type'],
                        'full_dialogue': traj.get('full_dialogue', False),
                    }
                    if 'refined_trajectory' in traj:
                        formatted_traj['refined_trajectory'] = traj['refined_trajectory']  # AlfWorld 精炼格式
                    formatted_trajectories.append(formatted_traj)
                
                with open(failed_traj_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_trajectories, f, indent=2, ensure_ascii=False)
                print(f"[SkillUpdate] Saved {len(formatted_trajectories)} failed trajectories to {failed_traj_path}")
            except Exception as e:
                print(f"[SkillUpdate] Warning: Failed to save trajectories: {e}")
                import traceback
                traceback.print_exc()

        # 分析失败并生成新 skills
        print(f"[SkillUpdate] Analyzing {len(failed_trajectories)} failed trajectories with o3...")
        new_skills, llm_metadata = self.skill_updater.analyze_failures(
            failed_trajectories=failed_trajectories,
            current_skills=retrieval_memory.skills,
            return_metadata=True,
        )

        # Save complete LLM call information (prompt, response, metadata)
        if save_traj:
            try:
                llm_call_path = os.path.join(save_dir, f'llm_call_step{self.global_steps}.json')
                llm_call_data = {
                    'step': self.global_steps,
                    'update_source': 'validation',
                    'llm_metadata': llm_metadata,
                    'failed_trajectories_analyzed': len(failed_trajectories),
                    'new_skills_generated': len(new_skills),
                }
                with open(llm_call_path, 'w', encoding='utf-8') as f:
                    json.dump(llm_call_data, f, indent=2, ensure_ascii=False)
                print(f"[SkillUpdate] Saved complete LLM call info to {llm_call_path}")
                # 把输入给 summarizer 的 query 写回 failed_trajectories JSON（无则用 skill_updater 现算）
                summarizer_queries = llm_metadata.get('summarizer_queries', [])
                if not summarizer_queries and hasattr(self, 'skill_updater') and failed_trajectories:
                    mode = getattr(self.skill_updater, 'skill_gen_mode', None)
                    if mode == 'summarize':
                        summarizer_queries = [self.skill_updater._build_summarizer_prompt_for_trajectory(t) for t in failed_trajectories]
                    elif mode == 'summarize_success':
                        summarizer_queries = [self.skill_updater._build_summarizer_prompt_for_group(t) for t in failed_trajectories]
                if summarizer_queries:
                    failed_traj_path = os.path.join(save_dir, f'failed_trajectories_step{self.global_steps}.json')
                    formatted_with_queries = []
                    for i, traj in enumerate(failed_trajectories):
                        ft = {'task': traj['task'], 'task_type': traj['task_type'], 'full_dialogue': traj.get('full_dialogue', False)}
                        if 'refined_trajectory' in traj:
                            ft['refined_trajectory'] = traj['refined_trajectory']
                        for key in ('group_uid', 'group_success_rate', 'group_size', 'group_failed_count'):
                            if key in traj:
                                ft[key] = traj[key]
                        ft['summarizer_query'] = summarizer_queries[i] if i < len(summarizer_queries) else None
                        formatted_with_queries.append(ft)
                    with open(failed_traj_path, 'w', encoding='utf-8') as f:
                        json.dump(formatted_with_queries, f, indent=2, ensure_ascii=False)
                    print(f"[SkillUpdate] Appended summarizer_query to {failed_traj_path}")
            except Exception as e:
                print(f"[SkillUpdate] Warning: Failed to save LLM call info: {e}")

        if new_skills:
            # 添加到 skill bank
            added = retrieval_memory.add_skills(new_skills, category='general')
            print(f"[SkillUpdate] Added {added} new skills to val_envs")

            # 保存更新后的 skills
            save_path = os.path.join(save_dir, f'updated_skills_step{self.global_steps}.json')
            retrieval_memory.save_skills(save_path)
            self._sync_skills_to_retrieval_server(retrieval_memory)

            # 同步到训练环境
            if hasattr(self, 'envs') and hasattr(self.envs, 'retrieval_memory') and self.envs.retrieval_memory:
                self.envs.retrieval_memory.add_skills(new_skills, category='general')
                print(f"[SkillUpdate] Synced {len(new_skills)} new skills to training envs")
        else:
            print("[SkillUpdate] No new skills generated")

    def _sync_skills_to_retrieval_server(self, retrieval_memory) -> None:
        """If using skill_retrieval_service_url, push current skills to the server so it sees dynamic updates."""
        som_cfg = self.config.env.get("skills_only_memory", {})
        url = som_cfg.get("skill_retrieval_service_url")
        if not url or not getattr(retrieval_memory, "skills", None):
            return
        urls = [url] if isinstance(url, str) else list(url)
        if not urls:
            return
        base = str(urls[0]).strip().rstrip("/")
        if "/retrieve_batch" in base:
            base = base.split("/retrieve_batch")[0].rstrip("/")
        reload_url = f"{base}/reload_skills"
        try:
            import requests
            r = requests.post(reload_url, json={"skills": retrieval_memory.skills}, timeout=30)
            r.raise_for_status()
            total = r.json().get("total_skills", "?")
            print(f"[SkillUpdate] Synced skills to retrieval server ({total} skills)")
        except Exception as e:
            print(f"[SkillUpdate] Warning: Failed to sync skills to server: {e}")

    def _record_retrieved_skills(
        self,
        step: int,
        phase: str,
        memories_list: list,
        per_step_retrievals_list: list | None = None,
    ) -> None:
        """
        Record which skills were retrieved at this step (train or validation).
        Writes one JSON file per step under default_local_dir.
        memories_list: list of per-batch retrieved_memories; each element is a list of
          per-sample dicts as returned by retrieval_memory.retrieve().
        per_step_retrievals_list: optional list of length len(memories_list); each element
          is a list of per-sample lists of {"step": int, "query_text": str, "general_skills": [...], "task_specific_skills": [...]}
          with each skill as {"title", "input_to_retrieval", "similarity"} (only when per_step_retrieval is True).
        """
        som_cfg = self.config.env.get("skills_only_memory", {})
        if not som_cfg.get("record_retrieved_skills", True):
            return
        if not memories_list or not any(memories_list):
            return
        save_dir = self.config.trainer.get("default_local_dir", "./outputs")
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            return
        records = []
        for batch_idx, memories in enumerate(memories_list):
            if not memories:
                continue
            per_step_for_batch = (per_step_retrievals_list[batch_idx] if per_step_retrievals_list and batch_idx < len(per_step_retrievals_list) else None)
            samples = []
            skill_text_mode = som_cfg.get("skill_text_for_retrieval", "full")
            def skill_row(s):
                if skill_text_mode == "when_to_apply":
                    input_to_retrieval = (s.get("when_to_apply") or "").strip()
                elif skill_text_mode == "principle":
                    input_to_retrieval = (s.get("principle") or "").strip()
                else:
                    parts = [s.get("title", ""), s.get("principle", ""), s.get("when_to_apply", "")]
                    input_to_retrieval = ". ".join(p for p in parts if p and str(p).strip()).strip(". ")
                return {"title": s.get("title", ""), "input_to_retrieval": input_to_retrieval, "similarity": s.get("similarity")}

            for i, mem in enumerate(memories):
                task_type = mem.get("task_type", "")
                sample = {
                    "sample_idx": i,
                    "query_text": mem.get("query_text", ""),
                    "task_type": task_type,
                    "general_skills": [skill_row(s) for s in mem.get("general_skills", [])],
                    "task_specific_skills": [skill_row(s) for s in mem.get("task_specific_skills", [])],
                }
                if per_step_for_batch is not None and i < len(per_step_for_batch):
                    sample["per_step_skills"] = per_step_for_batch[i]
                samples.append(sample)
            records.append({"batch_idx": batch_idx, "samples": samples})
        out = {
            "step": step,
            "phase": phase,
            "num_batches": len(records),
            "per_step_retrieval": som_cfg.get("per_step_retrieval", False),
            "retrievals": records,
        }
        filename = f"retrieved_skills_{phase}_step{step}.json"
        path = os.path.join(save_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"[RetrievedSkills] Recorded {len(records)} batch(es) to {path}")
        except Exception as e:
            print(f"[RetrievedSkills] Failed to write {path}: {e}")

    def _collect_failed_trajectories(
        self,
        inputs: list,
        outputs: list,
        scores: list,
        batch=None,
    ) -> list:
        """
        收集失败的 trajectories 用于分析
        
        如果提供了 batch 对象，会通过 traj_uid 重建完整的轨迹历史（包括 observations），
        与验证时的逻辑保持一致。
        
        否则，假设 outputs 已经包含完整的对话历史。
        """
        failed = []
        
        # If batch is provided, reconstruct full trajectory history like validation does
        if batch is not None and 'traj_uid' in batch.non_tensor_batch:
            traj_uids = batch.non_tensor_batch['traj_uid']
            
            # Debug: print batch structure
            print(f"[CollectFailedTraj] Batch keys: {list(batch.batch.keys())}")
            print(f"[CollectFailedTraj] Non-tensor keys: {list(batch.non_tensor_batch.keys())}")
            print(f"[CollectFailedTraj] Batch size: {len(outputs)}, Unique traj_uids: {len(set(traj_uids))}")
            
            # Extract observations from input_ids
            # input_ids contains the full input sequence for each step
            # For training, each step's input_ids contains: initial_prompt + all previous turns + current observation
            observation_texts = []
            if 'input_ids' in batch.batch:
                input_ids_list = batch.batch['input_ids']
                print(f"[CollectFailedTraj] Found input_ids, shape: {input_ids_list.shape if hasattr(input_ids_list, 'shape') else len(input_ids_list)}")
                
                # Decode each input_ids to get the full text
                # The observation is embedded in the input_ids, but we need to extract it
                # For now, we'll use the full decoded text as observation context
                # (The actual observation extraction is complex and depends on the exact format)
                for idx, full_input_ids in enumerate(input_ids_list):
                    # Decode the full input (contains prompt + observation history)
                    full_text = self.tokenizer.decode(full_input_ids, skip_special_tokens=True)
                    
                    # For training, input_ids at each step contains the accumulated history
                    # We can use the full text as observation context, or extract just the latest observation
                    # For simplicity, we'll use the full text minus the prompt and response
                    prompt_text = inputs[idx] if idx < len(inputs) else ""
                    response_text = outputs[idx] if idx < len(outputs) else ""
                    
                    # Try to extract observation by removing known parts
                    obs_text = full_text
                    if prompt_text and prompt_text in obs_text:
                        # Remove prompt from the beginning
                        obs_text = obs_text[len(prompt_text):].strip()
                    if response_text and response_text in obs_text:
                        # Remove response from the end
                        obs_text = obs_text[:-len(response_text)].strip()
                    
                    # If observation is still very long, it might contain the full history
                    # In that case, we keep it as is (it will be handled by _format_trajectory)
                    observation_texts.append(obs_text)
            else:
                print(f"[CollectFailedTraj] Warning: No input_ids in batch, using empty observations")
                observation_texts = [""] * len(outputs)
            
            # Group by traj_uid to reconstruct full trajectories (include query_text per step for retrieval_obs)
            query_texts_batch = batch.non_tensor_batch.get('query_text') if batch is not None else None
            traj_dict = {}
            for idx, (inp, out, score, traj_uid) in enumerate(zip(inputs, outputs, scores, traj_uids)):
                if traj_uid not in traj_dict:
                    traj_dict[traj_uid] = {
                        'inputs': [],
                        'outputs': [],
                        'observations': [],
                        'query_texts': [],
                        'scores': [],
                        'indices': []
                    }
                traj_dict[traj_uid]['inputs'].append(inp)
                traj_dict[traj_uid]['outputs'].append(out)
                traj_dict[traj_uid]['observations'].append(observation_texts[idx] if idx < len(observation_texts) else "")
                if query_texts_batch is not None and idx < len(query_texts_batch):
                    qt = query_texts_batch[idx]
                    traj_dict[traj_uid]['query_texts'].append(qt if isinstance(qt, str) else str(qt))
                else:
                    traj_dict[traj_uid]['query_texts'].append("")
                traj_dict[traj_uid]['scores'].append(score)
                traj_dict[traj_uid]['indices'].append(idx)
            
            print(f"[CollectFailedTraj] Grouped into {len(traj_dict)} unique trajectories")
            # for tid, tdata in traj_dict.items():
            #     print(f"[CollectFailedTraj] Traj {tid}: {len(tdata['outputs'])} steps")
            
            # Decide which failed trajectories to keep: by-group (one per low-success-rate group) or all
            traj_uids_to_collect = []
            chosen_traj_info = {}  # traj_uid -> {group_uid, group_success_rate, group_size, group_failed_count}
            has_uid = 'uid' in batch.non_tensor_batch
            print(f"[CollectFailedTraj] batch has 'uid': {has_uid}")
            if has_uid:
                uids = batch.non_tensor_batch['uid']
                try:
                    sample_uids = list(np.unique(uids))[:3] if hasattr(np, 'unique') else list(set(uids.ravel().tolist()))[:3]
                except Exception:
                    sample_uids = list(uids)[:3] if hasattr(uids, '__len__') else []
                print(f"[CollectFailedTraj] sample group uids (first 3): {sample_uids}")
                group_threshold = self.config.env.get('skills_only_memory', {}).get(
                    'skill_update_group_success_rate_threshold', 0.5
                )
                # traj_uid -> uid (from first row of each traj)
                traj_to_uid = {}
                for idx, tid in enumerate(traj_uids):
                    if tid not in traj_to_uid:
                        traj_to_uid[tid] = uids[idx] if idx < len(uids) else None
                # uid -> list of traj_uid
                uid_to_trajs = defaultdict(list)
                for tid, uid in traj_to_uid.items():
                    if uid is not None:
                        uid_to_trajs[uid].append(tid)
                num_groups = len(uid_to_trajs)
                group_sizes = [len(tids) for tids in uid_to_trajs.values()]
                print(f"[CollectFailedTraj] Grouping: {num_groups} groups (uids), trajectories per group: min={min(group_sizes)}, max={max(group_sizes)} (expected group_size from env.rollout.n)")
                # For each group with success rate <= threshold, pick one failed traj at random
                for uid, tids in uid_to_trajs.items():
                    success_count = sum(
                        1 for t in tids
                        if not any(s <= 0 for s in traj_dict[t]['scores'])
                    )
                    rate = success_count / len(tids) if tids else 0.0
                    if rate <= group_threshold:
                        failed_in_group = [
                            t for t in tids
                            if any(s <= 0 for s in traj_dict[t]['scores'])
                        ]
                        if failed_in_group:
                            chosen = random.choice(failed_in_group)
                            traj_uids_to_collect.append(chosen)
                            chosen_traj_info[chosen] = {
                                'group_uid': str(uid),
                                'group_success_rate': round(rate, 4),
                                'group_size': len(tids),
                                'group_failed_count': len(failed_in_group),
                            }
                            print(f"[CollectFailedTraj] Selected traj {chosen} from group uid={uid}, "
                                  f"group_success_rate={rate:.4f}, group_size={len(tids)}, failed_in_group={len(failed_in_group)}")
                print(f"[CollectFailedTraj] Group-based: {len(traj_uids_to_collect)} groups with success_rate<={group_threshold}, one traj per group")
            else:
                traj_uids_to_collect = [
                    tid for tid, tdata in traj_dict.items()
                    if any(s <= 0 for s in tdata['scores'])
                ]
            
            def build_failed_item(traj_uid, traj_data):
                initial_prompt = traj_data['inputs'][0]
                task_type = self._detect_task_type_from_input(initial_prompt)
                dialogue_parts = [f"Initial Prompt: {initial_prompt}"]
                for turn_idx, (obs, action) in enumerate(zip(traj_data['observations'], traj_data['outputs'])):
                    if obs.strip():
                        dialogue_parts.append(f"\nTurn {turn_idx + 1}:")  # 1-based (Turn 1, 2, ...) so summarizer ERROR_TURN and query_texts[idx] align
                        dialogue_parts.append(f"  Observation: {obs}")
                        dialogue_parts.append(f"  Action: {action}")
                    else:
                        dialogue_parts.append(f"\nTurn {turn_idx + 1}: {action}")
                full_dialogue = "\n".join(dialogue_parts)
                failed_item = {
                    'task': initial_prompt,
                    'trajectory': [{'action': full_dialogue, 'observation': ''}],
                    'task_type': task_type,
                    'full_dialogue': True,
                    'query_texts': traj_data.get('query_texts', []),
                }
                try:
                    from agent_system.memory.trajectory_refinement import build_refined_trajectory
                    from agent_system.memory.task_extraction import extract_short_task_for_retrieval
                    task_short = extract_short_task_for_retrieval(initial_prompt)
                    obs_list = None
                    if 'anchor_obs' in batch.non_tensor_batch:
                        anchor_obs = batch.non_tensor_batch['anchor_obs']
                        obs_list = [anchor_obs[i] if i < len(anchor_obs) else '' for i in traj_data['indices']]
                    if obs_list is None or len(obs_list) != len(traj_data['outputs']):
                        # Fallback: build obs_list from query_texts (format "task\n\nCurrent observation: obs")
                        qts = traj_data.get('query_texts', [])
                        if qts and len(qts) == len(traj_data['outputs']):
                            sep = "\n\nCurrent observation: "
                            obs_list = []
                            for qt in qts:
                                if sep in (qt or ""):
                                    obs_list.append((qt or "").split(sep, 1)[1].strip())
                                else:
                                    obs_list.append((qt or "").strip())
                            if not task_short and qts:
                                raw = (qts[0] or "").split(sep, 1)[0].strip() if sep in (qts[0] or "") else ""
                                task_short = extract_short_task_for_retrieval(raw) if raw else extract_short_task_for_retrieval(initial_prompt)
                        else:
                            obs_list = None
                    if obs_list is not None and len(obs_list) == len(traj_data['outputs']):
                        refined = build_refined_trajectory(task_short or extract_short_task_for_retrieval(initial_prompt), obs_list, traj_data['outputs'])
                        failed_item['refined_trajectory'] = refined
                except Exception as e:
                    print(f"[CollectFailedTraj] refined_trajectory failed: {e}")
                return failed_item

            for traj_uid in traj_uids_to_collect:
                traj_data = traj_dict[traj_uid]
                failed_item = build_failed_item(traj_uid, traj_data)
                if traj_uid in chosen_traj_info:
                    failed_item.update(chosen_traj_info[traj_uid])
                    # For summarize_success: attach one success trajectory from same group if any
                    group_uid = failed_item.get('group_uid')
                    tids = uid_to_trajs.get(group_uid, [])
                    success_in_group = [
                        t for t in tids
                        if not any(s <= 0 for s in traj_dict[t]['scores'])
                    ]
                    if success_in_group:
                        chosen_success = random.choice(success_in_group)
                        failed_item['success_trajectory'] = build_failed_item(chosen_success, traj_dict[chosen_success])
                    else:
                        failed_item['success_trajectory'] = None
                failed.append(failed_item)
        else:
            # Original logic: process each sample independently (for validation or old format)
            for inp, out, score in zip(inputs, outputs, scores):
                if score <= 0:  # 失败的 trajectory
                    # 尝试解析 task type
                    task_type = self._detect_task_type_from_input(inp)
                    
                    # 检查 output 是否包含完整对话历史（包含 "Turn" 关键字）
                    if "Turn" in out and ("Observation:" in out or "Action:" in out):
                        # 这是完整的对话历史格式，直接保存（不截断，保留完整信息）
                        failed.append({
                            'task': inp,  # 不截断，保留完整初始 prompt
                            'trajectory': [{'action': out, 'observation': ''}],  # 完整对话历史保存在 action 字段
                            'task_type': task_type,
                            'full_dialogue': True,  # 标记这是完整对话格式
                        })
                    else:
                        # 这是旧的单 action 格式，保持原有逻辑但增加截断长度
                        failed.append({
                            'task': inp[:2000] if len(inp) > 2000 else inp,  # 增加到 2000 字符
                            'trajectory': [{'action': out[:2000] if len(out) > 2000 else out, 'observation': ''}],
                            'task_type': task_type,
                            'full_dialogue': False,
                        })
        max_traj = self.config.env.get('skills_only_memory', {}).get('max_trajectories_for_skill_update', 10)
        capped = failed[:max_traj]
        if len(failed) > max_traj:
            print(f"[CollectFailedTraj] Capped to {max_traj} trajectories (had {len(failed)}); config: max_trajectories_for_skill_update")
        return capped

    def _detect_task_type_from_input(self, inp: str) -> str:
        """从输入中检测任务类型"""
        inp_lower = inp.lower()
        if 'clean' in inp_lower:
            return 'clean'
        elif 'heat' in inp_lower:
            return 'heat'
        elif 'cool' in inp_lower:
            return 'cool'
        elif 'look at' in inp_lower and ('lamp' in inp_lower or 'light' in inp_lower):
            return 'look_at_obj_in_light'
        elif 'examine' in inp_lower:
            return 'examine'
        else:
            return 'pick_and_place'

    def _update_skills_from_training_data(self, current_step_failures: list):
        """
        Update the skill bank using failed trajectories from the current step only.
        Called every training step; uses only the failures just collected this step.
        """
        if not current_step_failures:
            print("[SkillUpdate-Train] No failed trajectories from current step, skipping update")
            return

        trajectories_to_analyze = current_step_failures
        update_config = self.config.env.skills_only_memory

        # lazy-init SkillUpdater (backed by Azure OpenAI o3)
        if not hasattr(self, 'skill_updater'):
            from agent_system.memory.skill_updater import SkillUpdater
            self.skill_updater = SkillUpdater(
                max_new_skills_per_update=update_config.get('max_new_skills', 3),
                skill_gen_mode=update_config.get('skill_gen_mode', 'direct'),
                summarizer_model=update_config.get('summarizer_model'),
                summarizer_max_concurrent=update_config.get('summarizer_max_concurrent', 4),
                retrieval_obs=update_config.get('retrieval_obs', False),
            )

        # use the training envs' retrieval_memory directly (not via val_envs)
        retrieval_memory = None
        if hasattr(self, 'envs') and hasattr(self.envs, 'retrieval_memory'):
            retrieval_memory = self.envs.retrieval_memory
        if retrieval_memory is None:
            print("[SkillUpdate-Train] No retrieval_memory found in training envs")
            return

        # 保存失败轨迹到磁盘（如果配置启用）
        save_traj = update_config.get('update_save_traj', False)
        save_dir = self.config.trainer.get('default_local_dir', './outputs')
        if save_traj:
            failed_traj_path = os.path.join(save_dir, f'failed_trajectories_train_step{self.global_steps}.json')
            try:
                import json
                os.makedirs(save_dir, exist_ok=True)
                print(f"[SkillUpdate-Train] Using {len(trajectories_to_analyze)} trajectories for save/LLM (current step)")
                
                # 初始化 SkillUpdater（如果还没有）以便使用其格式化方法
                if not hasattr(self, 'skill_updater'):
                    from agent_system.memory.skill_updater import SkillUpdater
                    self.skill_updater = SkillUpdater(
                        max_new_skills_per_update=update_config.get('max_new_skills', 3),
                        skill_gen_mode=update_config.get('skill_gen_mode', 'direct'),
                        summarizer_model=update_config.get('summarizer_model'),
                        summarizer_max_concurrent=update_config.get('summarizer_max_concurrent', 4),
                        retrieval_obs=update_config.get('retrieval_obs', False),
                    )
                
                # 格式化轨迹，使其与传给 LLM 的内容一致
                formatted_trajectories = []
                for traj in trajectories_to_analyze:
                    # 只保存 task / task_type / refined_trajectory 等，不再存 original_trajectory、llm_prompt_section
                    formatted_traj = {
                        'task': traj['task'],
                        'task_type': traj['task_type'],
                        'full_dialogue': traj.get('full_dialogue', False),
                    }
                    if 'refined_trajectory' in traj:
                        formatted_traj['refined_trajectory'] = traj['refined_trajectory']  # AlfWorld 精炼格式
                    # 调试：保存按 group 收集时的正确率等信息
                    for key in ('group_uid', 'group_success_rate', 'group_size', 'group_failed_count'):
                        if key in traj:
                            formatted_traj[key] = traj[key]
                    formatted_trajectories.append(formatted_traj)
                
                with open(failed_traj_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_trajectories, f, indent=2, ensure_ascii=False)
                print(f"[SkillUpdate-Train] Saved {len(formatted_trajectories)} failed trajectories to {failed_traj_path}")
            except Exception as e:
                print(f"[SkillUpdate-Train] Warning: Failed to save trajectories: {e}")
                import traceback
                traceback.print_exc()

        print(f"[SkillUpdate-Train] Analyzing {len(trajectories_to_analyze)} trajectories (current step) with o3...")
        if trajectories_to_analyze and any('group_success_rate' in t for t in trajectories_to_analyze):
            for i, t in enumerate(trajectories_to_analyze):
                if 'group_uid' in t:
                    print(f"  traj[{i}] group_uid={t.get('group_uid')}, group_success_rate={t.get('group_success_rate')}, group_size={t.get('group_size')}, group_failed_count={t.get('group_failed_count')}")
        new_skills, llm_metadata = self.skill_updater.analyze_failures(
            failed_trajectories=trajectories_to_analyze,
            current_skills=retrieval_memory.skills,
            return_metadata=True,
        )

        # Save complete LLM call information (prompt, response, metadata)
        if save_traj:
            try:
                llm_call_path = os.path.join(save_dir, f'llm_call_train_step{self.global_steps}.json')
                llm_call_data = {
                    'step': self.global_steps,
                    'update_source': 'train',
                    'llm_metadata': llm_metadata,
                    'failed_trajectories_analyzed': len(trajectories_to_analyze),
                    'new_skills_generated': len(new_skills),
                }
                with open(llm_call_path, 'w', encoding='utf-8') as f:
                    json.dump(llm_call_data, f, indent=2, ensure_ascii=False)
                print(f"[SkillUpdate-Train] Saved complete LLM call info to {llm_call_path}")
                # 把输入给 summarizer 的 query 写回 failed_trajectories JSON（无则用 skill_updater 现算，保证 AlfWorld/WebShop 都落盘）
                summarizer_queries = llm_metadata.get('summarizer_queries', [])
                if not summarizer_queries and hasattr(self, 'skill_updater') and trajectories_to_analyze:
                    mode = getattr(self.skill_updater, 'skill_gen_mode', None)
                    if mode == 'summarize':
                        summarizer_queries = [self.skill_updater._build_summarizer_prompt_for_trajectory(t) for t in trajectories_to_analyze]
                    elif mode == 'summarize_success':
                        summarizer_queries = [self.skill_updater._build_summarizer_prompt_for_group(t) for t in trajectories_to_analyze]
                if summarizer_queries:
                    failed_traj_path = os.path.join(save_dir, f'failed_trajectories_train_step{self.global_steps}.json')
                    formatted_with_queries = []
                    for i, traj in enumerate(trajectories_to_analyze):
                        ft = {'task': traj['task'], 'task_type': traj['task_type'], 'full_dialogue': traj.get('full_dialogue', False)}
                        if 'refined_trajectory' in traj:
                            ft['refined_trajectory'] = traj['refined_trajectory']
                        for key in ('group_uid', 'group_success_rate', 'group_size', 'group_failed_count'):
                            if key in traj:
                                ft[key] = traj[key]
                        ft['summarizer_query'] = summarizer_queries[i] if i < len(summarizer_queries) else None
                        formatted_with_queries.append(ft)
                    with open(failed_traj_path, 'w', encoding='utf-8') as f:
                        json.dump(formatted_with_queries, f, indent=2, ensure_ascii=False)
                    print(f"[SkillUpdate-Train] Appended summarizer_query to {failed_traj_path}")
            except Exception as e:
                print(f"[SkillUpdate-Train] Warning: Failed to save LLM call info: {e}")

        if new_skills:
            added = retrieval_memory.add_skills(new_skills, category='general')
            print(f"[SkillUpdate-Train] Added {added} new skills to training envs")

            # sync to val envs (read-only direction: train → val, never the reverse)
            if hasattr(self, 'val_envs') and hasattr(self.val_envs, 'retrieval_memory') \
                    and self.val_envs.retrieval_memory:
                self.val_envs.retrieval_memory.add_skills(new_skills, category='general')
                print(f"[SkillUpdate-Train] Synced {len(new_skills)} new skills to val_envs")

            # save snapshot to disk
            save_path = os.path.join(save_dir, f'updated_skills_train_step{self.global_steps}.json')
            retrieval_memory.save_skills(save_path)
            self._sync_skills_to_retrieval_server(retrieval_memory)

        else:
            print("[SkillUpdate-Train] No new skills generated")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # load saved skills from checkpoint dir so resume continues with the same skill bank
        self._load_resume_skills(global_step_folder)

    def _load_resume_skills(self, global_step_folder: str):
        """If env uses SkillsOnlyMemory, load the latest updated_skills_train_step*.json from the checkpoint dir."""
        import re
        retrieval_memory = getattr(self.envs, "retrieval_memory", None)
        if retrieval_memory is None or not hasattr(retrieval_memory, "load_skills"):
            return
        # Use normpath so trailing slash does not make dirname point to global_step_* instead of exp dir
        exp_dir = os.path.dirname(os.path.normpath(global_step_folder))
        if not os.path.isdir(exp_dir):
            return
        pattern = re.compile(r"updated_skills_train_step(\d+)\.json")
        best_path, best_step = None, -1
        try:
            for f in os.listdir(exp_dir):
                m = pattern.match(f)
                if m:
                    step = int(m.group(1))
                    if step <= self.global_steps and step > best_step:
                        path = os.path.join(exp_dir, f)
                        if os.path.isfile(path):
                            best_path, best_step = path, step
        except OSError as e:
            print(f"[Resume] Could not list skills in {exp_dir}: {e}")
            return
        if best_path is None:
            return
        if retrieval_memory.load_skills(best_path):
            if hasattr(self, "val_envs") and getattr(self.val_envs, "retrieval_memory", None) and hasattr(self.val_envs.retrieval_memory, "load_skills"):
                self.val_envs.retrieval_memory.load_skills(best_path)
            self._sync_skills_to_retrieval_server(retrieval_memory)
            print(f"[Resume] Loaded skills from {best_path} (step {best_step}) and synced to retrieval server")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                        # record retrieved skills for this train step
                        if (hasattr(self.envs, "retrieved_memories") and self.envs.retrieved_memories is not None):
                            per_step_list = None
                            if hasattr(gen_batch_output, "non_tensor_batch") and gen_batch_output.non_tensor_batch.get("per_step_retrieved_by_traj") is not None:
                                per_step_list = [gen_batch_output.non_tensor_batch["per_step_retrieved_by_traj"]]
                            self._record_retrieved_skills(
                                step=self.global_steps,
                                phase="train",
                                memories_list=[self.envs.retrieved_memories],
                                per_step_retrievals_list=per_step_list,
                            )
                        # per_step_retrieved_by_traj is trajectory-level (len=num_trajectories); batch is step-level
                        # (len=total steps). Remove it so adjust_batch/select_idxs do not index it by step index -> IndexError.
                        if hasattr(gen_batch_output, "non_tensor_batch"):
                            gen_batch_output.non_tensor_batch.pop("per_step_retrieved_by_traj", None)
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output

                    # Dynamic management: log per-task utility (task_utility = mean(success|with_skills) - mean(success|without_skills))
                    if is_dynamic_management_enabled(self.config):
                        task_key = batch.non_tensor_batch.get("task_key")
                        with_skills = batch.non_tensor_batch.get("with_skills_mask")
                        succ = batch.non_tensor_batch.get("success")
                        if task_key is not None and with_skills is not None and succ is not None:
                            task_key = np.asarray(task_key)
                            with_skills = np.asarray(with_skills)
                            succ = np.asarray(succ)
                            by_task = defaultdict(lambda: {"with": [], "without": []})
                            for t, w, s in zip(task_key, with_skills, succ):
                                t = str(t)
                                if w:
                                    by_task[t]["with"].append(float(s))
                                else:
                                    by_task[t]["without"].append(float(s))
                            task_utilities = []
                            for v in by_task.values():
                                if v["with"] and v["without"]:
                                    task_utilities.append(np.mean(v["with"]) - np.mean(v["without"]))
                            if task_utilities:
                                metrics["dynamic_mgmt/mean_abs_task_utility"] = float(np.mean(np.abs(task_utilities)))

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    
                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # === dynamic skill update from training data ===
                    if self.config.env.get('skills_only_memory', {}).get('enable_dynamic_update', False):
                        update_source = self.config.env.get('skills_only_memory', {}).get('update_source', 'validation')
                        if update_source in ['train', 'all']:
                            # decode current training batch (independent of rollout_data_dir)
                            # Debug: check batch structure before decoding
                            print(f"[TrainSkillUpdate] Batch keys: {list(batch.batch.keys())}")
                            print(f"[TrainSkillUpdate] Batch size: {len(batch.batch.get('prompts', []))}")
                            if 'traj_uid' in batch.non_tensor_batch:
                                unique_traj_uids = len(set(batch.non_tensor_batch['traj_uid']))
                                total_steps = len(batch.non_tensor_batch['traj_uid'])
                                print(f"[TrainSkillUpdate] Total steps: {total_steps}, Unique traj_uids: {unique_traj_uids}, Avg steps per traj: {total_steps / max(unique_traj_uids, 1):.2f}")
                            
                            _train_inputs = self.tokenizer.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True
                            )
                            _train_outputs = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True
                            )
                            _train_scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

                            # collect failed trajectories (one per group with low success rate)
                            new_failures = self._collect_failed_trajectories(
                                _train_inputs, _train_outputs, _train_scores, batch=batch
                            )
                            # every step: update skill bank using only this step's failures
                            self._update_skills_from_training_data(current_step_failures=new_failures)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
