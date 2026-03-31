#!/usr/bin/env python3
"""
验证 skill-webshop 环境是否配置正确、能否正常使用。

在 SkillRL 项目根目录下执行：
  PYTHONPATH=. python agent_system/environments/env_package/webshop/verify_webshop_env.py

或在 agent_system/environments 目录下执行：
  PYTHONPATH=../.. python env_package/webshop/verify_webshop_env.py
"""
from __future__ import annotations

import os
import sys

# 把 webshop 子包加入 path，以便 import web_agent_site
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WEBSHOP_ROOT = os.path.join(_SCRIPT_DIR, "webshop")
if _WEBSHOP_ROOT not in sys.path:
    sys.path.insert(0, _WEBSHOP_ROOT)


def main() -> None:
    use_small = True  # 与当前默认配置一致
    if use_small:
        file_path = os.path.join(_SCRIPT_DIR, "webshop", "data", "items_shuffle_1000.json")
        attr_path = os.path.join(_SCRIPT_DIR, "webshop", "data", "items_ins_v2_1000.json")
    else:
        file_path = os.path.join(_SCRIPT_DIR, "webshop", "data", "items_shuffle.json")
        attr_path = os.path.join(_SCRIPT_DIR, "webshop", "data", "items_ins_v2.json")

    for name, path in [("items", file_path), ("attrs", attr_path)]:
        if not os.path.isfile(path):
            print(f"[FAIL] 数据文件不存在: {path}")
            sys.exit(1)
    print("[OK] 数据文件存在")

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
    print("正在创建 WebAgentTextEnv（会加载商品与检索索引，请稍候）...")
    env = gym.make("WebAgentTextEnv-v0", **env_kwargs)
    print("[OK] 环境创建成功")

    print("执行 reset(session=0)...")
    obs, info = env.reset(session=0)
    assert obs is not None, "reset 应返回观测"
    print("[OK] reset 成功")
    print("  观测长度:", len(obs) if isinstance(obs, str) else "N/A")

    actions = env.get_available_actions()
    print("  当前可用动作示例:", list(actions)[:5] if hasattr(actions, "__iter__") else actions)

    # 若有 search，执行一步 search；否则用第一个可用动作
    step_ok = False
    if isinstance(actions, dict) and actions.get("has_search_bar"):
        action = "search[shoes]"
    elif isinstance(actions, (list, tuple)) and actions:
        action = actions[0] if isinstance(actions[0], str) else "search[shoes]"
    else:
        action = "search[shoes]"
    print(f"执行一步: {action}")
    obs, reward, done, info = env.step(action)
    step_ok = True
    print(f"[OK] step 成功 (reward={reward}, done={done})")

    env.close()
    print("\n========== 验证通过：skill-webshop 环境可正常使用 ==========")


if __name__ == "__main__":
    main()
