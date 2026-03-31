# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Convert rollout trajectory batch to SFT messages format for MultiTurnSFTDataset.

from typing import List, Dict, Any
import numpy as np


def trajectory_to_messages(
    trajectory: List[Dict[str, Any]],
    tokenizer,
    include_inactive_steps: bool = False,
) -> List[Dict[str, str]]:
    """
    Convert one trajectory (list of per-step dicts from rollout) to a list of
    messages [{"role":"user","content":...}, {"role":"assistant","content":...}, ...].

    Each step in trajectory should have:
      - raw_prompt: list of messages, e.g. [{"role":"user","content": obs_text}]
      - responses: token ids (tensor or list) for model output
      - active_masks: whether this step was taken (optional; if False we skip unless include_inactive_steps)
    """
    messages = []
    for step in trajectory:
        if not include_inactive_steps and step.get("active_masks") is False:
            continue
        raw_prompt = step.get("raw_prompt")
        if not raw_prompt or not isinstance(raw_prompt, (list, np.ndarray)):
            continue
        # raw_prompt can be list of dicts; take first user content
        first_msg = raw_prompt[0] if hasattr(raw_prompt, "__getitem__") else raw_prompt
        if isinstance(first_msg, dict):
            user_content = first_msg.get("content", "")
        else:
            user_content = str(first_msg)
        responses = step.get("responses")
        if responses is None:
            continue
        if hasattr(responses, "cpu"):
            responses = responses.cpu().numpy()
        if isinstance(responses, np.ndarray) and responses.ndim > 0:
            response_ids = responses.ravel()
        else:
            response_ids = responses
        assistant_content = tokenizer.decode(
            response_ids, skip_special_tokens=True
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
    return messages
