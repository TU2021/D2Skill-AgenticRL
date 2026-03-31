# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Collect SFT data via OpenAI-compatible HTTP API (no local LLM / no Ray).
# Hydra env: set +env.use_skills_only_memory=False (plain prompts) or True with
# +env.skills_only_memory.* (task/step retrieval via skill_retrieval_server), same as GRPO.

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
from pprint import pprint

from omegaconf import OmegaConf, open_dict


def _parse_args():
    p = argparse.ArgumentParser(
        description="Collect SFT parquet via external OpenAI-compatible API (no local actor model)."
    )
    p.add_argument("--config_path", default=None, help="Hydra config dir (default: verl/trainer/config)")
    p.add_argument("--config_name", default="ppo_trainer", help="Base config name")
    p.add_argument("--output_path", required=True, help="Output parquet path")
    p.add_argument("--filter_success_only", action="store_true", help="Only keep successful trajectories")
    p.add_argument("--max_trajectories", type=int, default=None, help="Max trajectories to write")
    p.add_argument("--num_epochs", type=int, default=1, help="Epochs over train dataloader")
    p.add_argument("--override", nargs="*", default=[], help="Hydra overrides")
    return p.parse_args()


def _obs_text_at(obs: Dict[str, Any], i: int) -> str:
    t = obs["text"]
    if t is None:
        return ""
    if isinstance(t, (list, tuple)):
        return str(t[i])
    # numpy str array or ndarray
    return str(np.asarray(t).reshape(-1)[i])


def _build_openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for API-based SFT collection. "
            "Set it in the environment (e.g. source env.sh)."
        )
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _chat_completion(
    client,
    model: str,
    user_content: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Single-turn: full observation string already includes task + history (env template)."""
    extra = extra or {}
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.4")),
    }
    mt = os.environ.get("OPENAI_MAX_TOKENS")
    if mt:
        kwargs["max_tokens"] = int(mt)
    else:
        kwargs["max_tokens"] = 1024
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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, List[List[Tuple[str, str]]]]:
    """One parallel batch until done or max_steps."""
    obs, infos = envs.reset(kwargs=env_kwargs)
    batch_size = len(infos)
    if obs.get("text") is None:
        raise RuntimeError("Env returned no text observations; API collector supports text agent envs only.")

    is_done = np.zeros(batch_size, dtype=bool)
    episode_rewards = np.zeros(batch_size, dtype=np.float32)
    episode_lengths = np.zeros(batch_size, dtype=np.float32)
    traj_uid = np.array([str(__import__("uuid").uuid4()) for _ in range(batch_size)], dtype=object)

    total_batch_list: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
    total_infos: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
    # Per-env: list of (user_str, assistant_str) for SFT
    dialogues: List[List[Tuple[str, str]]] = [[] for _ in range(batch_size)]

    max_steps = int(config.env.max_steps)
    workers = max(1, min(max_workers, batch_size))

    for _step in range(max_steps):
        active_masks = np.logical_not(is_done)
        if not active_masks.any():
            break

        def call_one(idx: int) -> Tuple[int, str]:
            u = _obs_text_at(obs, idx)
            text = _chat_completion(client, model, u, chat_extra)
            return idx, text

        indices = [i for i in range(batch_size) if active_masks[i]]
        text_actions: List[str] = [""] * batch_size
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(call_one, i): i for i in indices}
            for fut in as_completed(futs):
                idx, reply = fut.result()
                text_actions[idx] = reply
                u = _obs_text_at(obs, idx)
                dialogues[idx].append((u, reply))

        next_obs, rewards, dones, step_infos = envs.step(text_actions)
        if len(rewards.shape) == 2:
            rewards = rewards.squeeze(1)
        if len(dones.shape) == 2:
            dones = dones.squeeze(1)

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
    return episode_rewards, episode_lengths, success, traj_uid, dialogues


def _dialogues_to_messages(dialogue: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for u, a in dialogue:
        out.append({"role": "user", "content": u})
        out.append({"role": "assistant", "content": a})
    return out


def run_collect_api(config):
    from verl import DataProto
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils import hf_processor, hf_tokenizer
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.utils.fs import copy_to_local

    from agent_system.environments import make_envs

    # Webshop uses Ray in WebshopMultiProcessEnv; Ray connect() requires sys.argv[0] not None.
    try:
        from agent_system.environments.env_package.webshop.envs import _ensure_sys_argv_for_ray

        _ensure_sys_argv_for_ray()
    except Exception:
        pass

    collect_cfg = config.get("collect") or {}
    output_path = collect_cfg.get("output_path")
    filter_success_only = collect_cfg.get("filter_success_only", False)
    max_trajectories = collect_cfg.get("max_trajectories")
    num_epochs = collect_cfg.get("num_epochs", 1)

    tokenizer_path = os.environ.get(
        "TOKENIZER_MODEL_PATH",
        "Qwen/Qwen2.5-0.5B-Instruct",
    )
    local_tok = copy_to_local(tokenizer_path, use_shm=False)
    trust_remote_code = config.data.get("trust_remote_code", True)
    tokenizer = hf_tokenizer(local_tok, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_tok, trust_remote_code=trust_remote_code, use_fast=True)

    envs, val_envs = make_envs(config)

    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    train_sampler = create_rl_sampler(config.data, train_dataset)
    batch_size = config.data.get("gen_batch_size", config.data.train_batch_size)
    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=config.data.get("dataloader_num_workers", 0),
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )

    client = _build_openai_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    chat_extra = _parse_extra_chat_kwargs()
    max_workers = int(os.environ.get("API_MAX_WORKERS", "16"))

    all_rows: List[Dict[str, Any]] = []
    num_collected = 0

    def _write_checkpoint(msg: str) -> None:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(all_rows).to_parquet(output_path, index=False)
        print(f"[SFT collect API] Saved {len(all_rows)} trajectories to {output_path} ({msg})")

    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]

    def get_non_tensor_pop_keys(batch):
        non_tensor = ["raw_prompt_ids", "data_source"]
        if "multi_modal_data" in getattr(batch, "non_tensor_batch", {}):
            non_tensor.append("multi_modal_data")
        if "raw_prompt" in getattr(batch, "non_tensor_batch", {}):
            non_tensor.append("raw_prompt")
        if "tools_kwargs" in getattr(batch, "non_tensor_batch", {}):
            non_tensor.append("tools_kwargs")
        if "env_kwargs" in getattr(batch, "non_tensor_batch", {}):
            non_tensor.append("env_kwargs")
        return non_tensor

    for epoch in range(num_epochs):
        for batch_dict in train_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            non_tensor_batch_keys_to_pop = get_non_tensor_pop_keys(batch)
            gen_batch = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            gen_batch.meta_info = getattr(batch, "meta_info", {})
            env_kwargs = None
            if gen_batch.non_tensor_batch is not None:
                env_kwargs = gen_batch.non_tensor_batch.pop("env_kwargs", None)

            ep_rew, ep_len, success, traj_uid, dialogues = _run_api_rollout(
                envs,
                config,
                client,
                model,
                env_kwargs,
                chat_extra,
                max_workers,
            )

            sr = success.get("success_rate")
            if sr is None:
                won = np.zeros(len(dialogues), dtype=np.float32)
            else:
                won = np.asarray(sr, dtype=np.float32).reshape(-1)

            bs = len(dialogues)
            for i in range(bs):
                if max_trajectories is not None and num_collected >= max_trajectories:
                    break
                w = float(won[i]) > 0.5 if i < len(won) else False
                if filter_success_only and not w:
                    continue
                messages = _dialogues_to_messages(dialogues[i])
                if not messages:
                    continue
                all_rows.append(
                    {
                        "messages": messages,
                        "reward": float(ep_rew[i]),
                        "success": w,
                        "traj_uid": str(traj_uid[i]),
                        "episode_length": int(ep_len[i]),
                    }
                )
                num_collected += 1

            if max_trajectories is not None and num_collected >= max_trajectories:
                break
        _write_checkpoint(f"after epoch {epoch + 1}/{num_epochs}")
        if max_trajectories is not None and num_collected >= max_trajectories:
            break

    # Close train + val env managers (val Webshop stack would otherwise GC __del__ after Ray shutdown).
    try:
        envs.close()
    except Exception:
        pass
    try:
        if val_envs is not None:
            val_envs.close()
    except Exception:
        pass

    return output_path


def main():
    args = _parse_args()
    config_dir = args.config_path or os.path.join(_PROJECT_ROOT, "verl", "trainer", "config")

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=args.config_name, overrides=list(args.override))

    OmegaConf.resolve(config)
    with open_dict(config):
        config.collect = {
            "output_path": args.output_path,
            "filter_success_only": args.filter_success_only,
            "max_trajectories": args.max_trajectories,
            "num_epochs": args.num_epochs,
        }

    pprint(OmegaConf.to_container(config, resolve=True))
    run_collect_api(config)


if __name__ == "__main__":
    main()
