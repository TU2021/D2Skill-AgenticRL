# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
#
# Moved here from env_package/alfworld so that importing it does not trigger
# alfworld/__init__.py -> envs.py -> gymnasium (avoids "No module named 'gymnasium'" when
# building refined_trajectory for WebShop or other envs that do not use gymnasium).

"""
Failed-trajectory refinement (shared by AlfWorld / WebShop / related envs).

This module converts verbose raw trajectories (including system/task text,
retrieved-experience blocks, and full per-step inputs) into a compact dialogue
format: task description + per-step environment observation + per-step agent action.
Task text is extracted by the unified extract_short_task_for_retrieval helper
(e.g. from "Your task is to: ...").
"""

from typing import List, Dict, Any


def is_alfworld_env(env_name: str) -> bool:
    """Check whether the current environment is AlfWorld-like."""
    if env_name is None:
        return False
    return "alfworld" in env_name.lower()


def extract_task_short(first_input: str) -> str:
    """
    Extract a short task description from the first-step full input.
    Delegates to the shared extractor (AlfWorld/WebShop both use "Your task is to:").
    """
    from agent_system.memory.task_extraction import extract_short_task_for_retrieval
    if not first_input or not isinstance(first_input, str):
        return ""
    return extract_short_task_for_retrieval(first_input)


def build_refined_trajectory(
    task_short: str,
    observations: List[str],
    actions: List[str],
) -> Dict[str, Any]:
    """
    Build a refined trajectory that keeps only task + per-step (observation, action).

    Turn indexing convention: turns is a 0-based list, where turns[0] is step 1.
    This aligns with summarizer output such as "Turn 1" / ERROR_TURN: 1, so callers
    can map using error_turn_1based - 1.

    Args:
        task_short: Short task description from extract_task_short.
        observations: Per-step environment observations; should match actions length.
            Usually from batch anchor_obs.
        actions: Per-step agent outputs (including think/action).

    Returns:
        {"task": str, "turns": [{"observation": str, "action": str}, ...]}  (turns 0-based)
    """
    turns = []
    n = min(len(observations), len(actions))
    for i in range(n):
        obs = observations[i]
        if obs is None:
            obs = ""
        elif not isinstance(obs, str):
            obs = str(obs)
        turns.append({
            "observation": obs.strip(),
            "action": (actions[i] or "").strip() if i < len(actions) else "",
        })
    return {
        "task": (task_short or "").strip(),
        "turns": turns,
    }
