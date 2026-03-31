"""
Unified short-task extraction for retrieval consistency across envs (AlfWorld, WebShop, etc.).
Used by: env_manager (extract_task at inference), ray_trainer (build refined_trajectory),
skill_updater (retrieval_obs when no refined_trajectory).
Both AlfWorld and WebShop prompts use "Your task is to: ..."; raw WebShop env obs may use [SEP] Instruction:.
"""
from typing import Optional


def extract_short_task_for_retrieval(text: str, _env_name: Optional[str] = None) -> str:
    """
    Extract short task string from full prompt/observation for retrieval consistency.
    Tries in order: "Your goal is to complete the following task:", "Your task is to:",
    then [SEP] Instruction: (raw WebShop env obs), else truncate. env_name is ignored.
    """
    if not text or not text.strip():
        return (text or "").strip()
    s = text.strip()

    # WebShop WITH_MEMORY template: "Your goal is to complete the following task:\n<task>\n\n\n===================="
    goal_marker = "Your goal is to complete the following task:"
    idx_goal = s.find(goal_marker)
    if idx_goal != -1:
        start = idx_goal + len(goal_marker)
        end = s.find("\n\n====================", start)
        if end == -1:
            end = s.find("\n\n##", start)
        if end == -1:
            end = s.find("\n\n\n", start)
        if end == -1:
            end = s.find("\n\n", start)
        if end == -1:
            return s[start:].strip()
        return s[start:end].strip()

    # "Your task is to: ..." until "\n\n##" or first "\n\n" (AlfWorld & WebShop NO_HIS/TEMPLATE)
    marker = "Your task is to:"
    idx = s.find(marker)
    if idx != -1:
        start = idx + len(marker)
        end = s.find("\n\n##", start)
        if end == -1:
            end = s.find("\n\n", start)
        if end == -1:
            return s[start:].strip()
        return s[start:end].strip()

    # Fallback: raw WebShop env obs " [SEP] Instruction: [SEP] <instruction> ..."
    if " [SEP] " in s and "Instruction:" in s:
        parts = s.split(" [SEP] ")
        for i, p in enumerate(parts):
            if p.strip() == "Instruction:" and i + 1 < len(parts):
                return parts[i + 1].strip()

    return s[:500]
