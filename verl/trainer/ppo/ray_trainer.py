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
import re
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
from agent_system.memory.skill_updater import SkillUpdater, skill_updater_kwargs_from_config

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

def _has_think_block(text: str) -> bool:
    """Return True if text contains a non-empty <think>...</think> or <thinking>...</thinking> block."""
    if not text:
        return False
    # <think>...</think> (content may include newlines)
    think_match = re.search(r'<think>\s*(.*?)\s*</think>', text, re.DOTALL | re.IGNORECASE)
    if think_match and think_match.group(1).strip():
        return True
    # <thinking>...</thinking>
    thinking_match = re.search(r'<thinking>\s*(.*?)\s*</thinking>', text, re.DOTALL | re.IGNORECASE)
    if thinking_match and thinking_match.group(1).strip():
        return True
    return False


def apply_invalid_action_penalty(
    data: DataProto,
    invalid_action_penalty_coef=float,
    tokenizer=None,
    use_think_penalty: bool = True,
):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        # Step valid only when BOTH think and action are valid; otherwise penalize (same pattern as action-only)
        step_valid = float(action_valids) if action_valids.ndim == 0 else float(action_valids.item())
        if use_think_penalty and tokenizer is not None:
            response_ids = data_item.batch['responses']
            response_length = int(valid_response_length.item())
            response_ids_trim = response_ids[:response_length]
            if hasattr(response_ids_trim, 'cpu'):
                response_ids_trim = response_ids_trim.cpu().tolist()
            response_text = tokenizer.decode(response_ids_trim, skip_special_tokens=False)
            if not _has_think_block(response_text):
                step_valid = 0.0
        step_invalids = torch.tensor(1.0 - step_valid, dtype=torch.float32, device=prompt_ids.device)
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * step_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * step_invalids

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
        # train / val 共用同一 retrieval_memory（含 eviction 后），避免只改 envs 而 val 仍用旧库。
        som_cfg = self.config.env.get("skills_only_memory") or {}
        if self.config.env.get("use_skills_only_memory"):
            erm = getattr(self.envs, "retrieval_memory", None)
            venv = getattr(self, "val_envs", None)
            if erm is not None and venv is not None and getattr(venv, "retrieval_memory", None) is not erm:
                venv.retrieval_memory = erm
            if som_cfg.get("skill_retrieval_service_url") and erm is not None:
                self._sync_skills_to_retrieval_server(erm)

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
            # Remove trajectory-level keys so val_reward_fn / data[i] do not index by row -> IndexError
            if hasattr(test_output_gen_batch, "non_tensor_batch"):
                test_output_gen_batch.non_tensor_batch.pop("per_step_retrieved_by_traj", None)
                test_output_gen_batch.non_tensor_batch.pop("per_step_retrieved_for_record", None)
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
                input_per_step, output_texts, scores_per_step, batch=test_batch,
                with_skills_mask=test_batch.non_tensor_batch.get("with_skills_mask"),
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

        # === Skill Bank 动态更新（验证侧）：写入 val + train 的 retrieval_memory。
        # 若 update_source 为 train 或 all，训练循环里每步还会 _update_skills_from_training_data 再写 train；
        # 两套入口会对相似失败各跑一次 LLM，易产出内容相同的 skill；add_skills 已按内容去重，避免 task_090 / task_100 双份。
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

        # Skill bank eviction：与 test_freq 对齐（仅在本轮跑了 validation 时执行）；单一真源 = envs.retrieval_memory
        self._run_skill_eviction_after_validation()

        return metric_dict

    def _run_skill_eviction_after_validation(self) -> None:
        """
        在每次 validation 结束后（与 test_freq 一致）按 management 配置删减 skill 池。
        仅修改 self.envs.retrieval_memory；val_envs 已在 _validate 开头与其对齐为同一实例。
        """
        som = self.config.env.get("skills_only_memory") or {}
        mgr = som.get("management") or {}
        ev_on = mgr.get("eviction_enabled")
        if isinstance(ev_on, str):
            ev_on = ev_on.lower() in ("1", "true", "yes")
        if not ev_on:
            return
        rm = getattr(self.envs, "retrieval_memory", None)
        if rm is None or not hasattr(rm, "evict_excess_skills"):
            return
        max_t = mgr.get("eviction_max_task_skills")
        max_s = mgr.get("eviction_max_step_skills")
        if max_t is not None:
            try:
                max_t = int(max_t)
                if max_t <= 0:
                    max_t = None
            except (TypeError, ValueError):
                max_t = None
        if max_s is not None:
            try:
                max_s = int(max_s)
                if max_s <= 0:
                    max_s = None
            except (TypeError, ValueError):
                max_s = None
        if max_t is None and max_s is None:
            return
        try:
            protect = int(mgr.get("eviction_protect_recent_steps") or 0)
        except (TypeError, ValueError):
            protect = 0
        try:
            score_c = float(mgr.get("eviction_score_c") if mgr.get("eviction_score_c") is not None else 1.0)
            if score_c != score_c:  # nan
                score_c = 1.0
        except (TypeError, ValueError):
            score_c = 1.0
        try:
            gstep = int(self.global_steps)
        except (TypeError, ValueError):
            gstep = 0
        try:
            result = rm.evict_excess_skills(
                current_step=gstep,
                max_task_skills=max_t,
                max_step_skills=max_s,
                protect_recent_steps=protect,
                score_c=score_c,
            )
        except Exception as e:
            print(f"[SkillEviction] Failed: {e}")
            return
        removed = result.get("removed") or []
        warnings = result.get("warnings") or []
        if not removed and not warnings:
            return
        save_dir = self.config.trainer.get("default_local_dir", "./outputs")
        try:
            os.makedirs(save_dir, exist_ok=True)
            audit_path = os.path.join(save_dir, f"evicted_skills_step{self.global_steps}.json")
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[SkillEviction] Wrote audit to {audit_path}")
        except Exception as e:
            print(f"[SkillEviction] Audit write failed: {e}")
        if removed:
            try:
                bank_path = os.path.join(save_dir, f"skills_after_eviction_step{self.global_steps}.json")
                rm.save_skills(bank_path)
            except Exception as e:
                print(f"[SkillEviction] save_skills failed: {e}")
            self._sync_skills_to_retrieval_server(rm)
        elif warnings:
            print(f"[SkillEviction] No removals (see warnings in audit); skipping save_skills / server sync")

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
            # Fallback: no batch -> no with_skills_mask; baseline_ab_split filter not applied here
            failed_trajectories = self._collect_failed_trajectories(
                sample_inputs, sample_outputs, sample_scores
            )
        else:
            print(f"[SkillUpdate] Using {len(failed_trajectories)} pre-collected failed trajectories (refined_trajectory/query_texts) for this val round")

        if not failed_trajectories:
            print("[SkillUpdate] No failed trajectories found")
            return

        # 初始化 SkillUpdater（外部 API / Azure / 本地 OpenAI 兼容服务见 env.skills_only_memory.skill_llm_*）
        if not hasattr(self, 'skill_updater'):
            self.skill_updater = SkillUpdater(**skill_updater_kwargs_from_config(update_config))

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
                    self.skill_updater = SkillUpdater(**skill_updater_kwargs_from_config(update_config))

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
        print(
            f"[SkillUpdate] Analyzing {len(failed_trajectories)} failed trajectories "
            f"(model={self.skill_updater.model}, backend={self.skill_updater.api_type})..."
        )
        new_skills, llm_metadata = self.skill_updater.analyze_failures(
            failed_trajectories=failed_trajectories,
            current_skills=retrieval_memory.skills,
            return_metadata=True,
        )

        # Save LLM call information whenever we ran skill update (so we can verify input/output even if save_traj=False)
        save_dir = self.config.trainer.get('default_local_dir', './outputs')
        try:
            os.makedirs(save_dir, exist_ok=True)
            llm_call_path = os.path.join(save_dir, f'llm_call_step{self.global_steps}.json')
            prompts_sent_to_llm = llm_metadata.get('summarizer_queries', [])
            llm_call_data = {
                'step': self.global_steps,
                'update_source': 'validation',
                'prompts_sent_to_llm': prompts_sent_to_llm,
                'raw_responses': llm_metadata.get('raw_responses', []),
                'llm_metadata': llm_metadata,
                'failed_trajectories_analyzed': len(failed_trajectories),
                'new_skills_generated': len(new_skills),
            }
            with open(llm_call_path, 'w', encoding='utf-8') as f:
                json.dump(llm_call_data, f, indent=2, ensure_ascii=False)
            print(f"[SkillUpdate] Saved LLM call info (prompts + raw_responses) to {llm_call_path}")
        except Exception as e:
            print(f"[SkillUpdate] Warning: Failed to save LLM call info: {e}")
        if save_traj:
            try:
                # 把输入给 summarizer 的 query 与 error_turn 写回 failed_trajectories JSON
                summarizer_queries = llm_metadata.get('summarizer_queries', [])
                error_turns = llm_metadata.get('error_turns', [])
                if summarizer_queries or error_turns:
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
                        # ERROR_TURN: N 供存记忆使用（与 expert_wrong_step_penalty 无关）
                        ft['error_turn'] = error_turns[i] if i < len(error_turns) else None
                        formatted_with_queries.append(ft)
                    with open(failed_traj_path, 'w', encoding='utf-8') as f:
                        json.dump(formatted_with_queries, f, indent=2, ensure_ascii=False)
                    print(f"[SkillUpdate] Appended summarizer_query and error_turn to {failed_traj_path}")
            except Exception as e:
                print(f"[SkillUpdate] Warning: Failed to save LLM call info: {e}")

        mode = llm_metadata.get('mode', '')
        is_task_step_mode = mode in ('task_only', 'step_only', 'task_step')
        task_skills = llm_metadata.get('task_skills', []) if is_task_step_mode else []
        if new_skills and mode in ('step_only', 'task_step'):
            _cs = int(self.global_steps)
            added = retrieval_memory.add_skills(new_skills, category='step', created_at_step=_cs)
            print(f"[SkillUpdate] Added {added} step_skills to val_envs")
            if hasattr(self, 'envs') and hasattr(self.envs, 'retrieval_memory') and self.envs.retrieval_memory:
                self.envs.retrieval_memory.add_skills(new_skills, category='step', created_at_step=_cs)
        elif new_skills:
            print("[SkillUpdate] Ignoring new_skills (only task/step categories supported)")
        else:
            print("[SkillUpdate] No new step_skills generated")
        if task_skills and mode in ('task_only', 'task_step'):
            _cs = int(self.global_steps)
            added_task = retrieval_memory.add_skills(task_skills, category='task', created_at_step=_cs)
            print(f"[SkillUpdate] Added {added_task} task_skills to val_envs")
            if hasattr(self, 'envs') and self.envs.retrieval_memory:
                self.envs.retrieval_memory.add_skills(task_skills, category='task', created_at_step=_cs)
        if new_skills or task_skills:
            save_path = os.path.join(save_dir, f'updated_skills_step{self.global_steps}.json')
            retrieval_memory.save_skills(save_path)
            self._sync_skills_to_retrieval_server(retrieval_memory)

    def _sync_skills_to_retrieval_server(self, retrieval_memory) -> None:
        """If using skill_retrieval_service_url, push current skills to the server so it sees dynamic updates."""
        som_cfg = self.config.env.get("skills_only_memory", {})
        url = som_cfg.get("skill_retrieval_service_url")
        skills = getattr(retrieval_memory, "skills", None)
        if not url or not skills:
            return
        # Do not push empty skills so we don't overwrite a server that was started with --skills_json_path
        n = len(skills.get("task_skills", [])) + len(skills.get("step_skills", []))
        if n == 0:
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
            r = requests.post(reload_url, json={"skills": skills}, timeout=30)
            r.raise_for_status()
            total = r.json().get("total_skills", "?")
            print(f"[SkillUpdate] Synced skills to retrieval server ({total} skills)")
        except Exception as e:
            print(f"[SkillUpdate] Warning: Failed to sync skills to server: {e}")

    def _apply_expert_wrong_step_from_llm(self, batch: DataProto) -> None:
        """
        Label expert_wrong_step by: after trajectory ends, use failed trajectories + one summarizer
        call (with ERROR_TURN in prompt) to get which step went wrong, then write back to
        batch.non_tensor_batch['expert_wrong_step']. Only requires expert_wrong_step_penalty > 0.
        Works regardless of retrieval_obs or use_skills_only_memory: we always use retrieval_obs=True
        for this call so the summarizer returns ERROR_TURN. Summarizer config from
        env.skills_only_memory, fallback to reward_model.expert_wrong_step_summarizer_model when
        skills_only_memory is disabled.
        """
        penalty = self.config.reward_model.get("expert_wrong_step_penalty", 0.0)
        if penalty <= 0:
            return
        som_cfg = self.config.env.get("skills_only_memory", {}) or {}
        inp = batch.batch["input_ids"] if "input_ids" in batch.batch else batch.batch.get("prompts")
        if inp is None or "traj_uid" not in batch.non_tensor_batch:
            return
        n = inp.shape[0] if hasattr(inp, "shape") else len(inp)
        if n == 0:
            return
        traj_uid_arr = batch.non_tensor_batch["traj_uid"]
        # Decode batch for _collect_failed_trajectories (same as later train path)
        try:
            _inputs = [
                self.tokenizer.decode(inp[i], skip_special_tokens=True)
                for i in range(n)
            ]
            _outputs = [
                self.tokenizer.decode(batch.batch["responses"][i], skip_special_tokens=True)
                for i in range(n)
            ]
        except Exception as e:
            print(f"[ExpertWrongStep] Decode failed: {e}")
            return
        scores_arr = batch.non_tensor_batch.get("episode_rewards")
        if scores_arr is None:
            return
        failed_list = self._collect_failed_trajectories(_inputs, _outputs, scores_arr, batch=batch)
        if not failed_list:
            batch.non_tensor_batch["expert_wrong_step"] = np.zeros(n, dtype=bool)
            return
        # Summarizer config: prefer skills_only_memory, fallback to reward_model when use_skills_only_memory=False
        rm_cfg = self.config.reward_model or {}
        summarizer_model = som_cfg.get("summarizer_model") or rm_cfg.get("expert_wrong_step_summarizer_model") or rm_cfg.get("summarizer_model")
        if not summarizer_model:
            print("[ExpertWrongStep] No summarizer_model (skills_only_memory or reward_model.expert_wrong_step_summarizer_model); skipping LLM labeling")
            batch.non_tensor_batch["expert_wrong_step"] = np.zeros(n, dtype=bool)
            return
        # Use a dedicated SkillUpdater with retrieval_obs=True so summarizer returns ERROR_TURN (do not overwrite self.skill_updater)
        summarizer_for_penalty = SkillUpdater(
            **skill_updater_kwargs_from_config(som_cfg),
            summarizer_model=summarizer_model,
        )
        error_turns = summarizer_for_penalty.get_error_turns_for_failed_trajectories(failed_list)
        traj_uid_to_error_turn = {}
        for item, et in zip(failed_list, error_turns):
            tid = item.get("traj_uid")
            if tid is not None and et is not None and et >= 1:
                traj_uid_to_error_turn[tid] = et
        # step_1based[i] = 1-based step index of row i within its trajectory (same as "Turn N" in dialogue)
        step_1based = np.zeros(n, dtype=np.int32)
        traj_counter = {}
        for i in range(n):
            uid = traj_uid_arr[i]
            traj_counter[uid] = traj_counter.get(uid, 0) + 1
            step_1based[i] = traj_counter[uid]
        # Match: error_turn from LLM is 1-based; step_1based is 1-based -> direct equality
        expert_wrong_step = np.array([
            traj_uid_to_error_turn.get(traj_uid_arr[i], -1) == step_1based[i]
            for i in range(n)
        ], dtype=bool)
        batch.non_tensor_batch["expert_wrong_step"] = expert_wrong_step
        num_marked = int(expert_wrong_step.sum())
        if num_marked > 0:
            print(f"[ExpertWrongStep] Labeled {num_marked} step(s) as wrong from {len(failed_list)} failed trajs (LLM ERROR_TURN)")
        elif failed_list:
            # 兜底：有失败轨迹但 LLM 未返回有效 ERROR_TURN（或返回 0/unclear）时，不施加 step 惩罚
            print(f"[ExpertWrongStep] Fallback: {len(failed_list)} failed trajs but no valid ERROR_TURN from summarizer -> no step penalty applied")

    @staticmethod
    def _json_safe_for_retrieved_skills(obj):
        """Recursively convert numpy / non-JSON types so json.dump never fails mid-stream."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {str(k): RayPPOTrainer._json_safe_for_retrieved_skills(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [RayPPOTrainer._json_safe_for_retrieved_skills(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return RayPPOTrainer._json_safe_for_retrieved_skills(obj.tolist())
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (str, int, float, bool)):
            return obj
        return str(obj)

    @staticmethod
    def _align_per_step_to_envs(per_step_data, traj_index_arr, n_envs: int):
        """
        per_step_retrieved_by_traj may be row-expanded (len >> n_envs). Align to one entry per parallel env.
        Each entry must be the full list of step snapshots for that trajectory (so per_step_skills can store all steps).
        """
        if per_step_data is None or n_envs <= 0:
            return None
        arr = np.asarray(per_step_data, dtype=object)
        # 若曾被存成 (n_traj, n_steps) 的二维 object（各轨迹等长时 np.array 会如此），先压回每轨迹一条 list
        if arr.ndim == 2 and arr.size > 0:
            _rows, _cols = arr.shape[0], arr.shape[1]
            _flat = np.empty(_rows, dtype=object)
            for _ri in range(_rows):
                _flat[_ri] = [arr[_ri, _cj] for _cj in range(_cols)]
            arr = _flat
        else:
            arr = arr.ravel()
        if len(arr) == n_envs:
            # trajectory-level: each arr[i] is already the full list of steps for traj i
            return [arr[i] if isinstance(arr[i], list) else [arr[i]] for i in range(n_envs)]
        if traj_index_arr is None:
            return [arr[i] if i < len(arr) else [] for i in range(n_envs)]
        ti = np.asarray(traj_index_arr).ravel().astype(np.int64)
        if len(ti) != len(arr):
            return [arr[i] if i < len(arr) else [] for i in range(n_envs)]
        n_traj = int(np.max(ti)) + 1 if ti.size else 0
        first_row = []
        for t in range(n_traj):
            idxs = np.where(ti == t)[0]
            if len(idxs):
                first_row.append(int(idxs[0]))
            else:
                first_row.append(0)
        # One entry per trajectory: take the full list from the first row of each traj (all rows of same traj share that list)
        if n_envs == n_traj:
            out = []
            for t in range(n_traj):
                if first_row[t] < len(arr):
                    val = arr[first_row[t]]
                    out.append(val if isinstance(val, list) else [val])
                else:
                    out.append([])
            return out
        out = []
        for i in range(n_envs):
            if i < len(first_row) and first_row[i] < len(arr):
                val = arr[first_row[i]]
                out.append(val if isinstance(val, list) else [val])
            elif i < len(arr):
                val = arr[i]
                out.append(val if isinstance(val, list) else [val])
            else:
                out.append([])
        return out

    def _record_retrieved_skills(
        self,
        step: int,
        phase: str,
        memories_list: list,
        per_step_retrievals_list: list | None = None,
        traj_index_for_record: list | None = None,
    ) -> None:
        """
        Record which skills were retrieved at this step (train or validation).
        Writes one JSON file per step under default_local_dir.
        Only task_skills and step_skills are recorded (no general_skills / task_specific_skills / task_type).
        memories_list: list of per-batch retrieved_memories; each element is a list of
          per-sample dicts with task_skills, step_skills, query_text.
        per_step_retrievals_list: optional; per-env per_step_skills (aligned to memories length).
        traj_index_for_record: optional; row-level traj_index for aligning row-expanded per_step to envs.
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
            raw_per_step = (
                per_step_retrievals_list[batch_idx]
                if per_step_retrievals_list and batch_idx < len(per_step_retrievals_list)
                else None
            )
            ti_batch = (
                traj_index_for_record[batch_idx]
                if traj_index_for_record and batch_idx < len(traj_index_for_record)
                else None
            )
            per_step_for_batch = self._align_per_step_to_envs(raw_per_step, ti_batch, len(memories))
            samples = []

            def task_step_skill_row(s):
                inp = (s.get("retrieval_obs") or "").strip() or ". ".join(
                    p for p in [s.get("title", ""), s.get("principle", ""), s.get("when_to_apply", "")] if p and str(p).strip()
                ).strip(". ")
                row = {"title": s.get("title", ""), "input_to_retrieval": inp, "similarity": s.get("similarity")}
                if "utility" in s:
                    row["utility"] = s.get("utility")
                if "ucb" in s:
                    row["ucb"] = s.get("ucb")
                if "retrieval_score" in s:
                    row["retrieval_score"] = s.get("retrieval_score")
                return row

            for i, mem in enumerate(memories):
                sample = {
                    "sample_idx": i,
                    "query_text": mem.get("query_text", ""),
                    "task_skills": [task_step_skill_row(s) for s in mem.get("task_skills", [])],
                    "step_skills": [task_step_skill_row(s) for s in mem.get("step_skills", [])],
                }
                if per_step_for_batch is not None and i < len(per_step_for_batch):
                    val = per_step_for_batch[i]
                    # per_step_skills must be a list of all steps for this trajectory (array in JSON)
                    if not isinstance(val, list):
                        val = [val] if val is not None else []
                    sample["per_step_skills"] = self._json_safe_for_retrieved_skills(val)
                samples.append(sample)
            records.append({"batch_idx": batch_idx, "samples": samples})
        out = {
            "step": step,
            "phase": phase,
            "num_batches": len(records),
            "per_step_retrieval": (som_cfg.get("skill_gen_mode") or "task_step") in ("step_only", "task_step"),
            "retrievals": records,
        }
        out = self._json_safe_for_retrieved_skills(out)
        filename = f"retrieved_skills_{phase}_step{step}.json"
        path = os.path.join(save_dir, filename)
        import tempfile

        try:
            fd, tmp_path = tempfile.mkstemp(dir=save_dir, suffix=".json.tmp", text=True)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            print(f"[RetrievedSkills] Recorded {len(records)} batch(es) to {path}")
        except Exception as e:
            print(f"[RetrievedSkills] Failed to write {path}: {e}")

    def _update_skill_utilities_from_rollout(self, batch) -> None:
        """
        Update skill utilities (EMA) from this rollout. Per-task:
        - task_utility_g = (加经验一半平均成功率) - (不加入记忆一半平均成功率); can be negative.
          Used to update **task_skills** retrieved in task g (once per task).
        - step_skill credit_i = success_i - success_baseline_g (trajectory i in task g); can be negative.
          Reflects step-level advantage over the task's baseline; used to update **step_skills**.
        Only runs when env.skills_only_memory.enable_dynamic_management is True.
        In task_only mode we have no per_step_retrieved_by_traj; build a synthetic one from envs.retrieved_memories
        so task_skills still get utility/retrieval_count/last_retrieval_step updated.
        """
        som_cfg = self.config.env.get("skills_only_memory", {}) or {}
        if not som_cfg.get("enable_dynamic_management", False):
            return
        retrieval_memory = getattr(self.envs, "retrieval_memory", None)
        if retrieval_memory is None or not hasattr(retrieval_memory, "update_utilities_for_trajectory"):
            return
        mgr = som_cfg.get("management", {}) or {}
        nt = getattr(batch, "non_tensor_batch", None) or {}
        traj_index_arr = nt.get("traj_index")
        with_skills_mask = nt.get("with_skills_mask")
        success_per_traj = nt.get("success_per_traj")
        per_step_raw = nt.get("per_step_retrieved_by_traj")
        # When row-level per_step is missing: try trajectory-level per_step_retrieved_for_record, or synthetic from envs.retrieved_memories (task_only / step_only)
        if per_step_raw is None and traj_index_arr is not None:
            ti = np.asarray(traj_index_arr).ravel().astype(np.int64)
            n_rows = len(ti)
            # 1) Try per_step_retrieved_for_record (trajectory-level): expand to row-level so utility/UCB can be updated
            per_step_for_record = nt.get("per_step_retrieved_for_record")
            if per_step_for_record is not None and len(per_step_for_record) > 0:
                n_traj_record = len(per_step_for_record)
                per_step_raw = np.array(
                    [per_step_for_record[int(ti[r])] if int(ti[r]) < n_traj_record else [] for r in range(n_rows)],
                    dtype=object,
                )
            # 2) task_only / step_only: rollout does not collect per_step; build synthetic from envs.retrieved_memories
            if per_step_raw is None and getattr(self.envs, "retrieved_memories", None) is not None:
                mode = (som_cfg.get("skill_gen_mode") or "task_step").strip().lower()
                mems = self.envs.retrieved_memories
                n_traj = len(mems)
                if mode == "task_only":
                    per_step_traj = [
                        [{"task_skills": list(mems[i].get("task_skills", [])), "step_skills": []}]
                        for i in range(n_traj)
                    ]
                elif mode == "step_only":
                    per_step_traj = [
                        [{"task_skills": [], "step_skills": list(mems[i].get("step_skills", []))}]
                        for i in range(n_traj)
                    ]
                else:
                    per_step_traj = None
                if per_step_traj is not None:
                    per_step_raw = np.array(
                        [per_step_traj[int(ti[r])] if int(ti[r]) < n_traj else [] for r in range(n_rows)],
                        dtype=object,
                    )
        if success_per_traj is None or per_step_raw is None or traj_index_arr is None:
            return
        traj_index_arr = np.asarray(traj_index_arr).ravel().astype(np.int64)
        n_traj = int(np.max(traj_index_arr)) + 1
        if n_traj == 0:
            return
        # Collapse row-level to trajectory-level (first row per trajectory)
        first_row = np.array([np.where(traj_index_arr == g)[0][0] for g in range(n_traj)], dtype=np.int64)
        success_per_traj = np.asarray(success_per_traj).ravel()[first_row]
        if with_skills_mask is not None:
            with_skills_mask = np.asarray(with_skills_mask).ravel()[first_row]
        else:
            with_skills_mask = np.ones(n_traj, dtype=bool)
        per_step = [per_step_raw[int(first_row[g])] for g in range(n_traj)]
        default_beta = float(mgr.get("utility_ema_beta", 0.1))
        beta_task = float(mgr.get("utility_ema_beta_task", default_beta))
        beta_step = float(mgr.get("utility_ema_beta_step", default_beta))
        if beta_task <= 0 and beta_step <= 0:
            return
        use_baseline_for_credit = mgr.get("credit_use_baseline", True)
        rollout_n = int(getattr(self.config.env.rollout, "n", 0) or 0) or 1
        num_tasks = (n_traj + rollout_n - 1) // rollout_n
        total_updated = 0
        for g in range(num_tasks):
            start, end = g * rollout_n, min((g + 1) * rollout_n, n_traj)
            group_idx = np.arange(start, end)
            baseline_idx = group_idx[~with_skills_mask[start:end]]
            with_skills_idx = group_idx[with_skills_mask[start:end]]
            success_baseline_g = float(np.mean(success_per_traj[baseline_idx])) if len(baseline_idx) > 0 and use_baseline_for_credit else 0.0
            task_utility_g = (float(np.mean(success_per_traj[with_skills_idx])) - success_baseline_g) if len(with_skills_idx) > 0 else 0.0
            task_skill_ids_g = set()
            for i in with_skills_idx:
                for step_snap in (per_step[i] if i < len(per_step) else []):
                    for s in step_snap.get("task_skills", []):
                        sid = s.get("skill_id")
                        if sid:
                            task_skill_ids_g.add(sid)
            if task_skill_ids_g and beta_task > 0:
                n = retrieval_memory.update_utilities_for_trajectory(
                    list(task_skill_ids_g), task_utility_g, self.global_steps, beta_task
                )
                total_updated += n
            for i in with_skills_idx:
                # step_skill credit = success_i - success_baseline_所在任务，体现相对本任务 baseline 的优势（可为负）
                credit_i = float(success_per_traj[i]) - success_baseline_g if use_baseline_for_credit else float(success_per_traj[i])
                step_skill_ids_i = set()
                for step_snap in (per_step[i] if i < len(per_step) else []):
                    for s in step_snap.get("step_skills", []):
                        sid = s.get("skill_id")
                        if sid:
                            step_skill_ids_i.add(sid)
                if step_skill_ids_i and beta_step > 0:
                    n = retrieval_memory.update_utilities_for_trajectory(
                        list(step_skill_ids_i), credit_i, self.global_steps, beta_step
                    )
                    total_updated += n
        if total_updated > 0:
            print(f"[SkillUtility] Updated {total_updated} skill utility entries (per-task task_utility + per-trajectory step_utility)")

    def _apply_intrinsic_reward(self, batch) -> None:
        """
        Add intrinsic_reward_i = (success_i - success_baseline_i) * coeff to the last valid token
        of each trajectory. success_baseline_i is **per-task**: mean(success of baseline trajectories
        in the same task as trajectory i). Uses traj_index (stable across balance_batch) to map rows.
        Only runs when enable_dynamic_management is True; params from .management.
        """
        som_cfg = self.config.env.get("skills_only_memory", {}) or {}
        if not som_cfg.get("enable_dynamic_management", False):
            return
        mgr = som_cfg.get("management", {}) or {}
        if not mgr.get("intrinsic_reward_enabled", False):
            return
        nt = getattr(batch, "non_tensor_batch", None) or {}
        success_per_traj = nt.get("success_per_traj")
        with_skills_mask = nt.get("with_skills_mask")
        traj_index_arr = nt.get("traj_index")
        if success_per_traj is None or traj_index_arr is None:
            return
        success_per_traj = np.asarray(success_per_traj).ravel()
        traj_index_arr = np.asarray(traj_index_arr).ravel().astype(np.int64)
        n_traj = int(np.max(traj_index_arr)) + 1
        if n_traj == 0:
            return
        # Collapse row-level to trajectory-level (first row per trajectory)
        first_row = np.array([np.where(traj_index_arr == g)[0][0] for g in range(n_traj)], dtype=np.int64)
        success_per_traj = success_per_traj[first_row]
        if with_skills_mask is not None:
            with_skills_mask = np.asarray(with_skills_mask).ravel()[first_row]
        else:
            with_skills_mask = np.ones(n_traj, dtype=bool)
        rollout_n = int(getattr(self.config.env.rollout, "n", 0) or 0) or 1
        num_tasks = (n_traj + rollout_n - 1) // rollout_n
        success_baseline_per_traj = np.zeros(n_traj, dtype=np.float64)
        for g in range(num_tasks):
            start, end = g * rollout_n, min((g + 1) * rollout_n, n_traj)
            group_idx = np.arange(start, end)
            baseline_idx = group_idx[~with_skills_mask[start:end]]
            baseline_g = float(np.mean(success_per_traj[baseline_idx])) if len(baseline_idx) > 0 else 0.0
            success_baseline_per_traj[start:end] = baseline_g
        coeff = float(mgr.get("intrinsic_reward_coefficient", 0.1))
        intrinsic_per_traj = (success_per_traj - success_baseline_per_traj) * coeff
        last_row_per_traj = {}
        for i in range(len(traj_index_arr)):
            t = int(traj_index_arr[i])
            last_row_per_traj[t] = i
        scores = batch.batch["token_level_scores"]
        response_mask = batch.batch.get("response_mask")
        if response_mask is None:
            response_mask = batch.batch.get("attention_mask")
        if response_mask is None:
            return
        num_rows = scores.shape[0]
        resp_len = scores.shape[1]
        if response_mask.shape[0] != num_rows or response_mask.shape[1] < resp_len:
            return
        for i in range(num_rows):
            if i >= len(traj_index_arr):
                continue
            traj_idx = int(traj_index_arr[i])
            if i != last_row_per_traj.get(traj_idx, -1):
                continue
            if traj_idx < 0 or traj_idx >= len(intrinsic_per_traj):
                continue
            last_valid = None
            for j in range(resp_len - 1, -1, -1):
                if j < response_mask.shape[1] and response_mask[i, j].item() != 0:
                    last_valid = j
                    break
            if last_valid is not None:
                scores[i, last_valid] = scores[i, last_valid] + float(intrinsic_per_traj[traj_idx])

    def _collect_failed_trajectories(
        self,
        inputs: list,
        outputs: list,
        scores: list,
        batch=None,
        with_skills_mask=None,
    ) -> list:
        """
        收集失败的 trajectories 用于分析。

        若提供 batch，则按 traj_uid 重建完整轨迹历史（含 observations），与验证逻辑一致。
        若 enable_dynamic_management 且 baseline_ab_split 且传入 with_skills_mask，则只保留
        「加入经验」的失败轨迹（with_skills_mask=True）用于反思/获取 skills；成功轨迹不受限。
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
            # Per-traj with_skills (加入经验): when baseline_ab_split, only use failed trajs with with_skills=True for reflection/skill update
            traj_uid_to_with_skills = {}
            if with_skills_mask is not None:
                wsm = np.asarray(with_skills_mask).ravel()
                for tid, tdata in traj_dict.items():
                    idx0 = tdata["indices"][0]
                    traj_uid_to_with_skills[tid] = bool(wsm[idx0]) if idx0 < len(wsm) else True
            
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
            # When enable_dynamic_management and baseline_ab_split: only use failed trajs that are 加入经验 (with_skills=True) for reflection/skill update
            _mgr = (self.config.env.get("skills_only_memory") or {}).get("management") or {}
            _enable_mgmt = (self.config.env.get("skills_only_memory") or {}).get("enable_dynamic_management", False)
            _baseline_ab = _enable_mgmt and _mgr.get("baseline_ab_split", False)
            if _baseline_ab and traj_uid_to_with_skills:
                n_before = len(traj_uids_to_collect)
                traj_uids_to_collect = [tid for tid in traj_uids_to_collect if traj_uid_to_with_skills.get(tid, True)]
                if n_before != len(traj_uids_to_collect):
                    print(f"[CollectFailedTraj] baseline_ab_split: only experience-group failures for reflection/skills, kept {len(traj_uids_to_collect)} of {n_before}")
            
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
                failed_item["traj_uid"] = traj_uid  # for expert_wrong_step: match (traj_uid, error_turn) back to batch
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

    def _set_expert_wrong_step_on_batch_from_error_turns(
        self, batch: DataProto, failed_list: list, error_turns: list
    ) -> None:
        """
        Set batch.non_tensor_batch['expert_wrong_step'] from (failed_list, error_turns).
        failed_list items must have 'traj_uid'; error_turns is 1-based per failed traj, same order.
        Indexing convention: both ERROR_TURN (from summarizer) and step_1based (batch row's step
        within trajectory) are 1-based, matching "Turn 1", "Turn 2", ... in the dialogue sent to LLM.
        """
        if not failed_list or not error_turns or "traj_uid" not in batch.non_tensor_batch:
            return
        inp = batch.batch["input_ids"] if "input_ids" in batch.batch else batch.batch.get("prompts")
        if inp is None:
            return
        n = inp.shape[0] if hasattr(inp, "shape") else len(inp)
        traj_uid_arr = batch.non_tensor_batch["traj_uid"]
        traj_uid_to_error_turn = {}
        for item, et in zip(failed_list, error_turns):
            tid = item.get("traj_uid")
            if tid is not None and et is not None and et >= 1:
                traj_uid_to_error_turn[tid] = et
        # step_1based[i] = 1-based step of row i within its trajectory (matches "Turn N" / ERROR_TURN: N)
        step_1based = np.zeros(n, dtype=np.int32)
        traj_counter = {}
        for i in range(n):
            uid = traj_uid_arr[i]
            traj_counter[uid] = traj_counter.get(uid, 0) + 1
            step_1based[i] = traj_counter[uid]
        expert_wrong_step = np.array([
            traj_uid_to_error_turn.get(traj_uid_arr[i], -1) == step_1based[i]
            for i in range(n)
        ], dtype=bool)
        batch.non_tensor_batch["expert_wrong_step"] = expert_wrong_step

    def _update_skills_from_training_data(self, current_step_failures: list, batch: DataProto = None):
        """
        Update the skill bank using failed trajectories from the current step only.
        Called every training step; uses only the failures just collected this step.
        If batch is provided and analyze_failures returns error_turns (retrieval_obs), and
        expert_wrong_step_penalty > 0, writes expert_wrong_step to batch so the penalty path
        does not need to call the LLM again.
        Returns True if batch.non_tensor_batch['expert_wrong_step'] was set by this call.
        """
        expert_wrong_step_set = False
        if not current_step_failures:
            print("[SkillUpdate-Train] No failed trajectories from current step, skipping update")
            return expert_wrong_step_set

        trajectories_to_analyze = current_step_failures
        update_config = self.config.env.skills_only_memory

        # lazy-init SkillUpdater（外部 API / 本地 skill 服务见 skill_updater_kwargs_from_config）
        if not hasattr(self, 'skill_updater'):
            self.skill_updater = SkillUpdater(**skill_updater_kwargs_from_config(update_config))

        # use the training envs' retrieval_memory directly (not via val_envs)
        retrieval_memory = None
        if hasattr(self, 'envs') and hasattr(self.envs, 'retrieval_memory'):
            retrieval_memory = self.envs.retrieval_memory
        if retrieval_memory is None:
            print("[SkillUpdate-Train] No retrieval_memory found in training envs")
            return expert_wrong_step_set

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
                    self.skill_updater = SkillUpdater(**skill_updater_kwargs_from_config(update_config))
                
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

        print(
            f"[SkillUpdate-Train] Analyzing {len(trajectories_to_analyze)} trajectories (current step) "
            f"(model={self.skill_updater.model}, backend={self.skill_updater.api_type})..."
        )
        if trajectories_to_analyze and any('group_success_rate' in t for t in trajectories_to_analyze):
            for i, t in enumerate(trajectories_to_analyze):
                if 'group_uid' in t:
                    print(f"  traj[{i}] group_uid={t.get('group_uid')}, group_success_rate={t.get('group_success_rate')}, group_size={t.get('group_size')}, group_failed_count={t.get('group_failed_count')}")
        new_skills, llm_metadata = self.skill_updater.analyze_failures(
            failed_trajectories=trajectories_to_analyze,
            current_skills=retrieval_memory.skills,
            return_metadata=True,
        )

        # 用 reflection 得到的 error_turns 写回 batch，供本步 penalty 使用
        if batch is not None and self.config.reward_model.get("expert_wrong_step_penalty", 0.0) > 0:
            error_turns = llm_metadata.get("error_turns", [])
            if error_turns and len(error_turns) == len(trajectories_to_analyze):
                self._set_expert_wrong_step_on_batch_from_error_turns(
                    batch, trajectories_to_analyze, error_turns
                )
                expert_wrong_step_set = True
                num_marked = int(batch.non_tensor_batch["expert_wrong_step"].sum())
                if num_marked > 0:
                    print(f"[SkillUpdate-Train] Set expert_wrong_step for {num_marked} step(s) from summarizer ERROR_TURN")

        # Save complete LLM call information (prompt, response, metadata)
        # Save LLM call info whenever we ran skill update (so we can verify input/output even if save_traj=False)
        save_dir = self.config.trainer.get('default_local_dir', './outputs')
        try:
            os.makedirs(save_dir, exist_ok=True)
            llm_call_path = os.path.join(save_dir, f'llm_call_train_step{self.global_steps}.json')
            prompts_sent_to_llm = llm_metadata.get('summarizer_queries', [])
            llm_call_data = {
                'step': self.global_steps,
                'update_source': 'train',
                'prompts_sent_to_llm': prompts_sent_to_llm,
                'raw_responses': llm_metadata.get('raw_responses', []),
                'llm_metadata': llm_metadata,
                'failed_trajectories_analyzed': len(trajectories_to_analyze),
                'new_skills_generated': len(new_skills),
            }
            with open(llm_call_path, 'w', encoding='utf-8') as f:
                json.dump(llm_call_data, f, indent=2, ensure_ascii=False)
            print(f"[SkillUpdate-Train] Saved LLM call info (prompts + raw_responses) to {llm_call_path}")
        except Exception as e:
            print(f"[SkillUpdate-Train] Warning: Failed to save LLM call info: {e}")
        if save_traj:
            try:
                # 把输入给 summarizer 的 query 与 ERROR_TURN 写回 failed_trajectories JSON（存记忆需要）
                summarizer_queries = llm_metadata.get('summarizer_queries', [])
                error_turns = llm_metadata.get('error_turns', [])
                if summarizer_queries or error_turns:
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
                        ft['error_turn'] = error_turns[i] if i < len(error_turns) else None
                        formatted_with_queries.append(ft)
                    with open(failed_traj_path, 'w', encoding='utf-8') as f:
                        json.dump(formatted_with_queries, f, indent=2, ensure_ascii=False)
                    print(f"[SkillUpdate-Train] Appended summarizer_query and error_turn to {failed_traj_path}")
            except Exception as e:
                print(f"[SkillUpdate-Train] Warning: Failed to save LLM call info: {e}")

        mode = llm_metadata.get('mode', '')
        is_task_step_mode = mode in ('task_only', 'step_only', 'task_step')
        task_skills = llm_metadata.get('task_skills', []) if is_task_step_mode else []
        if new_skills and mode in ('step_only', 'task_step'):
            _cs = int(self.global_steps)
            added = retrieval_memory.add_skills(new_skills, category='step', created_at_step=_cs)
            print(f"[SkillUpdate-Train] Added {added} step_skills to training envs")
            if hasattr(self, 'val_envs') and hasattr(self.val_envs, 'retrieval_memory') and self.val_envs.retrieval_memory:
                self.val_envs.retrieval_memory.add_skills(new_skills, category='step', created_at_step=_cs)
        elif new_skills:
            print("[SkillUpdate-Train] Ignoring new_skills (only task/step categories supported)")
        else:
            print("[SkillUpdate-Train] No new skills generated")
        if task_skills and mode in ('task_only', 'task_step'):
            _cs = int(self.global_steps)
            added_task = retrieval_memory.add_skills(task_skills, category='task', created_at_step=_cs)
            print(f"[SkillUpdate-Train] Added {added_task} task_skills to training envs")
            if hasattr(self, 'val_envs') and self.val_envs.retrieval_memory:
                self.val_envs.retrieval_memory.add_skills(task_skills, category='task', created_at_step=_cs)
        if new_skills or task_skills:
            save_path = os.path.join(save_dir, f'updated_skills_train_step{self.global_steps}.json')
            retrieval_memory.save_skills(save_path)
            self._sync_skills_to_retrieval_server(retrieval_memory)
        return expert_wrong_step_set

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

        # Sync current skills to retrieval server so validation (and first train) can retrieve;
        # otherwise remote server may have no skills before first train update.
        if getattr(self.envs, "retrieval_memory", None) and self.config.env.get("skills_only_memory", {}).get("skill_retrieval_service_url"):
            self._sync_skills_to_retrieval_server(self.envs.retrieval_memory)

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
                            nt = getattr(gen_batch_output, "non_tensor_batch", None) or {}
                            # Prefer trajectory-level key for recording (works with/without enable_dynamic_management)
                            per_step_list = None
                            ti_record = None
                            # 优先用 by_traj + traj_index：与 batch 行数一致，按轨迹取首行即可得到整条轨迹的全 step 列表
                            if nt.get("per_step_retrieved_by_traj") is not None and nt.get("traj_index") is not None:
                                per_step_list = [nt["per_step_retrieved_by_traj"]]
                                ti_record = [nt["traj_index"]]
                            elif nt.get("per_step_retrieved_for_record") is not None:
                                per_step_list = [nt["per_step_retrieved_for_record"]]
                            elif nt.get("per_step_retrieved_by_traj") is not None:
                                per_step_list = [nt["per_step_retrieved_by_traj"]]
                                if nt.get("traj_index") is not None:
                                    ti_record = [nt["traj_index"]]
                            self._record_retrieved_skills(
                                step=self.global_steps,
                                phase="train",
                                memories_list=[self.envs.retrieved_memories],
                                per_step_retrievals_list=per_step_list,
                                traj_index_for_record=ti_record,
                            )
                            # Dynamic memory: utility EMA update (only when enable_dynamic_management)
                            if self.config.env.get("skills_only_memory", {}).get("enable_dynamic_management", False):
                                self._update_skill_utilities_from_rollout(gen_batch_output)
                            # Remove per-step keys before adjust_batch so row-level indexing never sees trajectory-level length.
                            if hasattr(gen_batch_output, "non_tensor_batch"):
                                gen_batch_output.non_tensor_batch.pop("per_step_retrieved_by_traj", None)
                                gen_batch_output.non_tensor_batch.pop("per_step_retrieved_for_record", None)
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

                    # Skill 更新与存记忆提前：用 episode_rewards 收集失败轨迹，一次 summarizer 得到 ERROR_TURN 并写回 expert_wrong_step，避免步骤 2 再调一次 LLM
                    expert_wrong_step_set_by_skill_update = False
                    if self.config.env.get('skills_only_memory', {}).get('enable_dynamic_update', False):
                        update_source = self.config.env.get('skills_only_memory', {}).get('update_source', 'validation')
                        if update_source in ['train', 'all']:
                            _inp = batch.batch["prompts"] if "prompts" in batch.batch else batch.batch["input_ids"]
                            _train_inputs = self.tokenizer.batch_decode(_inp, skip_special_tokens=True)
                            _train_outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            ep_rew = batch.non_tensor_batch.get("episode_rewards")
                            _train_scores = np.asarray(ep_rew).ravel().tolist() if ep_rew is not None else [0.0] * len(_train_inputs)
                            new_failures = self._collect_failed_trajectories(
                                _train_inputs, _train_outputs, _train_scores, batch=batch,
                                with_skills_mask=batch.non_tensor_batch.get("with_skills_mask"),
                            )
                            expert_wrong_step_set_by_skill_update = self._update_skills_from_training_data(
                                current_step_failures=new_failures, batch=batch
                            )
                    # expert_wrong_step_penalty > 0 时，若未在 skill 更新中写回，则必须调一次 summarizer 获得 ERROR_TURN（与 retrieval_obs / use_skills_only_memory 无关）
                    if (
                        self.config.reward_model.get("expert_wrong_step_penalty", 0.0) > 0
                        and not expert_wrong_step_set_by_skill_update
                    ):
                        self._apply_expert_wrong_step_from_llm(batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        # Apply expert wrong-step penalty to per-step rewards (GiGPO step-level signal).
                        expert_penalty = self.config.reward_model.get("expert_wrong_step_penalty", 0.0)
                        if expert_penalty > 0 and "expert_wrong_step" in batch.non_tensor_batch:
                            r = np.asarray(batch.non_tensor_batch["rewards"], dtype=np.float32)
                            mask = np.asarray(batch.non_tensor_batch["expert_wrong_step"], dtype=bool)
                            n_penalized = int(mask.sum())
                            if n_penalized > 0:
                                idx_penalized = np.where(mask)[0]
                                for idx in idx_penalized[:5]:  # debug: first 5
                                    rb, ra = float(r[idx]), float(r[idx]) - expert_penalty
                                    print(f"[ExpertWrongStep-Penalty] GiGPO step_idx={int(idx)} reward_before={rb:.4f} reward_after={ra:.4f} penalty={expert_penalty:.4f}")
                                if n_penalized > 5:
                                    print(f"[ExpertWrongStep-Penalty] GiGPO total {n_penalized} steps penalized (shown first 5)")
                            r[mask] -= expert_penalty
                            batch.non_tensor_batch["rewards"] = r
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

                        # Intrinsic reward: success_i - success_baseline, add to last token of each trajectory
                        self._apply_intrinsic_reward(batch)

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                tokenizer=self.tokenizer,
                                use_think_penalty=self.config.actor_rollout_ref.actor.get('use_think_penalty', True),
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
                            _prompts = batch.batch["prompts"] if "prompts" in batch.batch else batch.batch["input_ids"]
                            inputs = self.tokenizer.batch_decode(_prompts, skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

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
