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
失败轨迹精炼（AlfWorld / WebShop 等通用）。

将冗长的原始轨迹（含 system、task、## Retrieved Relevant Experience、每步完整输入）
提炼为「仅任务描述 + 每步环境 obs + 每步 agent action」的对话式格式。
任务描述由统一的 extract_short_task_for_retrieval 提取（"Your task is to: " 等）。
"""

from typing import List, Dict, Any


def is_alfworld_env(env_name: str) -> bool:
    """判断当前是否为 AlfWorld 类环境（用于决定是否调用本精炼逻辑）。"""
    if env_name is None:
        return False
    return "alfworld" in env_name.lower()


def extract_task_short(first_input: str) -> str:
    """
    从首步完整输入中提取简短任务描述。委托给统一提取函数（AlfWorld/WebShop 均用 "Your task is to:"）。
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
    构建精炼轨迹：仅保留 task + 每步 (observation, action)。

    Turn 索引约定：turns 为 0-based 列表，turns[0] = 第 1 步 (obs, action)，
    与 summarizer 的 "Turn 1" / ERROR_TURN: 1 对应；使用时用 error_turn_1based - 1 下标即可。

    Args:
        task_short: 简短任务描述（由 extract_task_short 得到）。
        observations: 每步的环境观测，应与 actions 同长；通常来自 batch 的 anchor_obs。
        actions: 每步智能体输出（含 think/action）。

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
