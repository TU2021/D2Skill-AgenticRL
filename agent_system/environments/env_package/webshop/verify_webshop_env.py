#!/usr/bin/env python3
"""
Validate whether the skill-webshop environment is configured correctly and runnable.

Run from SkillRL project root:
  PYTHONPATH=. python agent_system/environments/env_package/webshop/verify_webshop_env.py

Or from agent_system/environments:
  PYTHONPATH=../.. python env_package/webshop/verify_webshop_env.py
"""
from __future__ import annotations

import os
import sys

# Add webshop subpackage to sys.path so web_agent_site can be imported.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WEBSHOP_ROOT = os.path.join(_SCRIPT_DIR, "webshop")
if _WEBSHOP_ROOT not in sys.path:
    sys.path.insert(0, _WEBSHOP_ROOT)


def main() -> None:
    use_small = True  # Keep consistent with current default config.
    if use_small:
        file_path = os.path.join(_SCRIPT_DIR, "webshop", "data", "items_shuffle_1000.json")
        attr_path = os.path.join(_SCRIPT_DIR, "webshop", "data", "items_ins_v2_1000.json")
    else:
        file_path = os.path.join(_SCRIPT_DIR, "webshop", "data", "items_shuffle.json")
        attr_path = os.path.join(_SCRIPT_DIR, "webshop", "data", "items_ins_v2.json")

    for name, path in [("items", file_path), ("attrs", attr_path)]:
        if not os.path.isfile(path):
            print(f"[FAIL] Data file not found: {path}")
            sys.exit(1)
    print("[OK] Data files found")

    import gym
    from web_agent_site.envs import WebAgentTextEnv

    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": True,
        "file_path": file_path,
        "attr_path": attr_path,
        "seed": 42,
    }
    print("Creating WebAgentTextEnv (this loads products and retrieval indexes; please wait)...")
    env = gym.make("WebAgentTextEnv-v0", **env_kwargs)
    print("[OK] Environment created successfully")

    print("Running reset(session=0)...")
    obs, info = env.reset(session=0)
    assert obs is not None, "reset should return an observation"
    print("[OK] reset succeeded")
    print("  Observation length:", len(obs) if isinstance(obs, str) else "N/A")

    actions = env.get_available_actions()
    print("  Example available actions:", list(actions)[:5] if hasattr(actions, "__iter__") else actions)

    # If search is available, run one search step; otherwise use the first available action.
    step_ok = False
    if isinstance(actions, dict) and actions.get("has_search_bar"):
        action = "search[shoes]"
    elif isinstance(actions, (list, tuple)) and actions:
        action = actions[0] if isinstance(actions[0], str) else "search[shoes]"
    else:
        action = "search[shoes]"
    print(f"Running one step: {action}")
    obs, reward, done, info = env.step(action)
    step_ok = True
    print(f"[OK] step succeeded (reward={reward}, done={done})")

    env.close()
    print("\n========== Validation passed: skill-webshop environment is ready ==========")


if __name__ == "__main__":
    main()
