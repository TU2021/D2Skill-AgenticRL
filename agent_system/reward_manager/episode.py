# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from verl import DataProto
import torch
import numpy as np

class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, normalize_by_length=False, expert_wrong_step_penalty=0.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length
        # When > 0: subtract this from score for the sampled step that is marked as expert_wrong_step (GRPO/GAE).
        # Only the specific sample (that step) is penalized, not all samples at that step index.
        # Scale with success reward: e.g. if success reward=10, use 1.0~2.0 (10–20%) or 2~5 for stronger signal. Set to 0 to disable.
        self.expert_wrong_step_penalty = float(expert_wrong_step_penalty)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        penalty_debug_printed = 0
        n_penalized_this_batch = 0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            # ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            multi_modal_inputs = data_item.non_tensor_batch.get('multi_modal_inputs', None)
            if multi_modal_inputs is not None:
                pixel_values = multi_modal_inputs['pixel_values']
                image_grid_thw = multi_modal_inputs['image_grid_thw']


            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']

            if self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards
            # Apply expert wrong-step penalty only for this sampled step (not for other steps).
            if self.expert_wrong_step_penalty != 0:
                wrong_step = bool(data_item.non_tensor_batch.get('expert_wrong_step', False))
                if wrong_step:
                    n_penalized_this_batch += 1
                    score_before = float(score)
                    score = score - self.expert_wrong_step_penalty
                    if penalty_debug_printed < 5:
                        print(f"[ExpertWrongStep-Penalty] GRPO/GAE sample_idx={i} score_before={score_before:.4f} score_after={score:.4f} penalty={self.expert_wrong_step_penalty:.4f}")
                        penalty_debug_printed += 1
            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][score]", score)

        if n_penalized_this_batch > 5:
            print(f"[ExpertWrongStep-Penalty] GRPO/GAE batch: {n_penalized_this_batch} samples penalized (shown first 5)")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor
