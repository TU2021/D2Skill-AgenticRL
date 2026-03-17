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

# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

# WEBSHOP_TEMPLATE_WITH_MEMORY = """
# You are an expert autonomous agent operating in the WebShop e‑commerce environment.
# Your task is to: {task_description}.

# ## Retrieved Relevant Experience

# {retrieved_memories}

# ## Current Progress

# Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
# You are now at step {current_step} and your current observation is: {current_observation}.
# Your admissible actions of the current situation are:
# [
# {available_actions}
# ].

# Now it's your turn to take one action for the current step.
# You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
# Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
# """

WEBSHOP_TEMPLATE_WITH_MEMORY = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.

Your goal is to complete the following task:
{task_description}


====================
## Current Progress
====================

You have already taken {step_count} step(s).

Recent interaction history (observation → action):
{action_history}

Current step: {current_step}

Current observation:
{current_observation}

Admissible actions at this step:
[{available_actions}]


====================
## Relevant Experience
====================

Below are past experiences retrieved from memory that may be relevant.

You should review these experiences before deciding your next action.

When reasoning, you may:

- Compare the current situation with past experiences
- Reuse successful actions if the situations are similar
- Avoid actions that previously led to failure
- Use past experience to guide your next step when helpful

If a retrieved experience seems relevant, consider it during your reasoning.
If it is not relevant, you may ignore it.

Retrieved experiences:
{retrieved_memories}


====================
## Instructions
====================

For the current step, you should follow this process:

1. Analyze the current observation
2. Review the retrieved experiences
3. Think about whether any past experience applies
4. Reason step-by-step
5. Choose the best admissible action

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""