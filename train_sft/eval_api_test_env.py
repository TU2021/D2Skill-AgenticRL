#!/usr/bin/env python3
"""Evaluate OpenAI-compatible API model on test envs (AlfWorld/WebShop)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate API model on test env and log metrics to SwanLab.")
    p.add_argument("--config_path", default=None, help="Hydra config dir (default: verl/trainer/config)")
    p.add_argument("--config_name", default="ppo_trainer", help="Base config name")
    p.add_argument("--override", nargs="*", default=[], help="Hydra overrides")
    p.add_argument("--num_eval_batches", type=int, default=1, help="How many val batches to evaluate")
    return p.parse_args()


def _obs_text_at(obs: Dict[str, Any], i: int) -> str:
    t = obs["text"]
    if t is None:
        return ""
    if isinstance(t, (list, tuple)):
        return str(t[i])
    return str(np.asarray(t).reshape(-1)[i])


def _build_openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for API evaluation.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _chat_completion(client, model: str, user_content: str, extra: Optional[Dict[str, Any]] = None) -> str:
    extra = extra or {}
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.4")),
    }
    mt = os.environ.get("OPENAI_MAX_TOKENS")
    kwargs["max_tokens"] = int(mt) if mt else 1024
    kwargs.update(extra)
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def _parse_extra_chat_kwargs() -> Dict[str, Any]:
    raw = os.environ.get("OPENAI_CHAT_EXTRA_JSON", "").strip()
    if not raw:
        return {}
    return json.loads(raw)


def _run_api_rollout(
    envs,
    config,
    client,
    model: str,
    env_kwargs: Any,
    chat_extra: Dict[str, Any],
    max_workers: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    obs, infos = envs.reset(kwargs=env_kwargs)
    batch_size = len(infos)
    if obs.get("text") is None:
        raise RuntimeError("Env returned no text observations; API evaluation supports text agent envs only.")

    is_done = np.zeros(batch_size, dtype=bool)
    episode_rewards = np.zeros(batch_size, dtype=np.float32)
    episode_lengths = np.zeros(batch_size, dtype=np.float32)
    total_batch_list: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
    total_infos: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]

    max_steps = int(config.env.max_steps)
    workers = max(1, min(max_workers, batch_size))

    for _step in range(max_steps):
        active_masks = np.logical_not(is_done)
        if not active_masks.any():
            break

        def call_one(idx: int) -> Tuple[int, str]:
            user_text = _obs_text_at(obs, idx)
            action_text = _chat_completion(client, model, user_text, chat_extra)
            return idx, action_text

        indices = [i for i in range(batch_size) if active_masks[i]]
        text_actions = [""] * batch_size
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(call_one, i): i for i in indices}
            for fut in as_completed(futs):
                idx, reply = fut.result()
                text_actions[idx] = reply

        next_obs, rewards, dones, step_infos = envs.step(text_actions)
        rewards_f = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones_b = np.asarray(dones, dtype=bool).reshape(-1)
        episode_rewards[active_masks] += rewards_f[active_masks]
        episode_lengths[active_masks] += 1.0

        for i in range(batch_size):
            total_batch_list[i].append({"active_masks": bool(active_masks[i])})
            total_infos[i].append(step_infos[i])

        is_done = np.logical_or(is_done, dones_b)
        obs = next_obs

    success = envs.success_evaluator(
        total_infos=total_infos,
        total_batch_list=total_batch_list,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
    )
    return episode_rewards, episode_lengths, success


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()) if arr.size > 0 else 0.0, float(arr.std()) if arr.size > 0 else 0.0


def run_eval_api(config, num_eval_batches: int) -> Dict[str, float]:
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl import DataProto
    from verl.trainer.main_ppo import create_rl_dataset
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.fs import copy_to_local
    from agent_system.environments import make_envs

    tokenizer_path = os.environ.get("TOKENIZER_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    local_tok = copy_to_local(tokenizer_path, use_shm=False)
    trust_remote_code = config.data.get("trust_remote_code", True)
    tokenizer = hf_tokenizer(local_tok, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_tok, trust_remote_code=trust_remote_code, use_fast=True)

    envs, val_envs = make_envs(config)
    _ = envs  # keep training env created by current factory design

    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    val_loader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=config.data.val_batch_size,
        num_workers=config.data.get("dataloader_num_workers", 0),
        drop_last=True,
        collate_fn=collate_fn,
    )

    client = _build_openai_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    chat_extra = _parse_extra_chat_kwargs()
    max_workers = int(os.environ.get("API_MAX_WORKERS", "16"))

    ep_rewards_all: List[float] = []
    ep_lengths_all: List[float] = []
    success_lists: Dict[str, List[float]] = defaultdict(list)

    started = time.time()
    for batch_idx, batch_dict in enumerate(val_loader):
        if batch_idx >= num_eval_batches:
            break
        batch = DataProto.from_single_dict(batch_dict)
        env_kwargs = None
        if batch.non_tensor_batch is not None:
            env_kwargs = batch.non_tensor_batch.get("env_kwargs")

        ep_rewards, ep_lengths, success = _run_api_rollout(
            envs=val_envs,
            config=config,
            client=client,
            model=model,
            env_kwargs=env_kwargs,
            chat_extra=chat_extra,
            max_workers=max_workers,
        )
        ep_rewards_all.extend(ep_rewards.tolist())
        ep_lengths_all.extend(ep_lengths.tolist())
        for key, arr in success.items():
            success_lists[key].extend(np.asarray(arr, dtype=np.float32).reshape(-1).tolist())

    elapsed = time.time() - started
    reward_mean, reward_std = _mean_std(ep_rewards_all)
    length_mean, length_std = _mean_std(ep_lengths_all)

    metrics: Dict[str, float] = {
        "val/api_eval/episode_count": float(len(ep_rewards_all)),
        "val/api_eval/episode_reward_mean": reward_mean,
        "val/api_eval/episode_reward_std": reward_std,
        "val/api_eval/episode_length_mean": length_mean,
        "val/api_eval/episode_length_std": length_std,
        "val/api_eval/elapsed_sec": float(elapsed),
    }
    for key, vals in success_lists.items():
        m, _ = _mean_std(vals)
        metrics[f"val/{key}"] = m

    try:
        envs.close()
    except Exception:
        pass
    try:
        val_envs.close()
    except Exception:
        pass

    return metrics


def _log_to_swanlab(metrics: Dict[str, float]) -> None:
    project = os.environ.get("SWANLAB_PROJECT", "verl_agent_api_eval")
    exp = os.environ.get("SWANLAB_EXPERIMENT", f"api_eval_{time.strftime('%Y%m%d_%H%M%S')}")
    mode = os.environ.get("SWANLAB_MODE", "cloud")
    api_key = os.environ.get("SWANLAB_API_KEY", "")

    try:
        import swanlab
    except Exception as e:
        print(f"[WARN] swanlab import failed, skip logging: {e}")
        return

    if api_key:
        try:
            swanlab.login(api_key)
        except Exception as e:
            print(f"[WARN] swanlab login failed, continue without explicit login: {e}")

    run = swanlab.init(
        project=project,
        experiment_name=exp,
        mode=mode,
        config={"openai_model": os.environ.get("OPENAI_MODEL", ""), "base_url": os.environ.get("OPENAI_BASE_URL", "")},
    )
    swanlab.log(metrics, step=0)
    run.finish()


def main() -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from pprint import pprint

    args = _parse_args()
    config_dir = args.config_path or os.path.join(_PROJECT_ROOT, "verl", "trainer", "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=args.config_name, overrides=list(args.override))
    OmegaConf.resolve(config)
    pprint(OmegaConf.to_container(config, resolve=True))

    metrics = run_eval_api(config, num_eval_batches=args.num_eval_batches)
    print("\n=== API Eval Metrics ===")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]:.6f}")
    _log_to_swanlab(metrics)


if __name__ == "__main__":
    main()
