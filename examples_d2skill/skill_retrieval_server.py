#!/usr/bin/env python3
"""
Skill retrieval HTTP server (embedding mode).

Single URL, multi-GPU in one process (recommended):
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/grpo_trainer/skill_retrieval_server.py \\
    --skills_json_path memory_data/alfworld/claude_style_skills_alfworld.json \\
    --num_gpus 8 --port 8002

Then set one URL in trainer config:
  env.skills_only_memory.skill_retrieval_service_url=http://127.0.0.1:8002/retrieve_batch

The server splits each batch (e.g. 128 queries) into 8 chunks and encodes them in parallel
on 8 GPUs, so one URL achieves multi-GPU parallelism without configuring multiple URLs.

Single GPU: omit --num_gpus or use --num_gpus 1.
"""

import argparse
import sys
from pathlib import Path

# Repo root
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

app = FastAPI()

# Global memory instance (loaded at startup)
_skill_memory = None


class ReloadSkillsRequest(BaseModel):
    """Reload skills from file path or inline. For dynamic skill update during training."""
    path: Optional[str] = None   # absolute path to JSON file (server must have read access)
    skills: Optional[Dict[str, Any]] = None  # full skills dict (works across machines)


class RetrieveBatchRequest(BaseModel):
    queries: List[str]
    top_k: int = 6
    pool: Optional[str] = None  # "task_skills" | "step_skills" (required for retrieval)


@app.post("/retrieve_batch")
def retrieve_batch_endpoint(request: RetrieveBatchRequest):
    """
    Batch skill retrieval. pool must be "task_skills" or "step_skills".
    Returns list of { task_skills/step_skills: [...], query_text } per query.
    """
    global _skill_memory
    n_queries = len(request.queries) if request.queries else 0
    print(f"[SkillRetrievalServer] /retrieve_batch pool={request.pool} queries={n_queries} top_k={request.top_k}", flush=True)
    if _skill_memory is None:
        raise HTTPException(status_code=500, detail="Skill memory not initialized")
    if request.pool not in ("task_skills", "step_skills"):
        raise HTTPException(status_code=400, detail="pool must be 'task_skills' or 'step_skills'")
    if request.pool == "task_skills":
        result = _skill_memory.retrieve_task_skills_batch(request.queries, top_k=request.top_k)
        return {"result": result}
    result = _skill_memory.retrieve_step_skills_batch(request.queries, top_k=request.top_k)
    return {"result": result}


@app.post("/reload_skills")
def reload_skills_endpoint(request: ReloadSkillsRequest):
    """
    Reload skill bank from file or inline dict so that dynamic updates from the trainer
    are visible to the server. Call this after the trainer saves updated skills.
    Embedding cache is not cleared: on next retrieve, only new skills are encoded
    and existing embeddings are reused (incremental update in SkillsOnlyMemory).
    """
    global _skill_memory
    src = "inline" if request.skills is not None else ("file:" + (request.path or ""))
    print(f"[SkillRetrievalServer] /reload_skills source={src}", flush=True)
    if _skill_memory is None:
        raise HTTPException(status_code=500, detail="Skill memory not initialized")
    def _count_skills(sk: Dict[str, Any]) -> int:
        return len(sk.get("task_skills", [])) + len(sk.get("step_skills", []))

    if request.skills is not None:
        if hasattr(_skill_memory, "replace_skills_keep_cache_incremental"):
            _skill_memory.replace_skills_keep_cache_incremental(request.skills)
        else:
            _skill_memory.skills = {
                "task_skills": request.skills.get("task_skills") or [],
                "step_skills": request.skills.get("step_skills") or [],
            }
            if hasattr(_skill_memory, "_task_skill_embeddings_cache"):
                _skill_memory._task_skill_embeddings_cache = None
            if hasattr(_skill_memory, "_step_skill_embeddings_cache"):
                _skill_memory._step_skill_embeddings_cache = None
        print(f"[SkillRetrievalServer] /reload_skills done total_skills={_count_skills(_skill_memory.skills)}", flush=True)
        return {"status": "ok", "source": "inline", "total_skills": _count_skills(_skill_memory.skills)}
    if request.path:
        path = Path(request.path)
        if not path.is_absolute():
            path = _repo_root / path
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {path}")
        with open(path, "r") as f:
            import json
            _skill_memory.skills = json.load(f)
        if hasattr(_skill_memory, "_task_skill_embeddings_cache"):
            _skill_memory._task_skill_embeddings_cache = None
        if hasattr(_skill_memory, "_step_skill_embeddings_cache"):
            _skill_memory._step_skill_embeddings_cache = None
        print(f"[SkillRetrievalServer] /reload_skills done path={path} total_skills={_count_skills(_skill_memory.skills)}", flush=True)
        return {"status": "ok", "source": "file", "path": str(path), "total_skills": _count_skills(_skill_memory.skills)}
    raise HTTPException(status_code=400, detail="Provide either 'path' or 'skills'")


def main():
    global _skill_memory
    parser = argparse.ArgumentParser(description="Skill retrieval server (embedding mode).")
    parser.add_argument(
        "--skills_json_path",
        type=str,
        default=None,
        help="Path to Claude-style skills JSON. Optional when --no_load_initial_skills.",
    )
    parser.add_argument(
        "--no_load_initial_skills",
        action="store_true",
        help="Start with empty skill bank (do not load from skills_json_path). Skills can be loaded via /reload_skills.",
    )
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer model for embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the embedding model when num_gpus=1 (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use in this process. When > 1, batch is split and encoded in parallel (one URL, multi-GPU).",
    )
    parser.add_argument("--port", type=int, default=8002, help="Port for the HTTP server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    args = parser.parse_args()
    load_initial = not getattr(args, "no_load_initial_skills", False)
    if load_initial and not args.skills_json_path:
        parser.error("--skills_json_path is required unless --no_load_initial_skills is set.")

    from agent_system.memory import SkillsOnlyMemory

    _skill_memory = SkillsOnlyMemory(
        skills_json_path=args.skills_json_path if load_initial else None,
        retrieval_mode="embedding",
        embedding_model_path=args.embedding_model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        skill_text_for_retrieval="full",  # overridden per request by client when sent
        load_initial_skills=load_initial,
    )
    # When starting with empty skill bank, warm up the embedding model so the first
    # retrieve after /reload_skills does not block on model load (avoids client timeout).
    if not load_initial:
        print("[SkillRetrievalServer] Warming up embedding model (empty skill bank)...")
        _skill_memory._get_embedding_model()
        print("[SkillRetrievalServer] Embedding model ready.")
    print(f"[SkillRetrievalServer] Listening on {args.host}:{args.port} (num_gpus={args.num_gpus})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
