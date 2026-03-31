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

"""
Lightweight skills-only memory system.

This is a simplified version of RetrievalMemory that only uses Claude-style skills
without the overhead of loading and indexing trajectory memories.

Supports two retrieval modes:
  - "template": keyword-based task type detection + return all task-specific skills
    (original behaviour, zero latency, no GPU needed)
  - "embedding": encode the task description with Qwen3-Embedding-0.6B and rank
    both general and task-specific skills by cosine similarity, so only the
    top-k most relevant ones are injected into the prompt
"""

import json
import math
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from .base import BaseMemory

# Defaults for dynamic memory management fields (missing => 0)
DEFAULT_UTILITY = 0.0
DEFAULT_RETRIEVAL_COUNT = 0
DEFAULT_LAST_RETRIEVAL_STEP = 0
DEFAULT_CREATED_AT_STEP = 0


class SkillsOnlyMemory(BaseMemory):
    """
    Lightweight memory system that only uses Claude-style skills.

    Retrieval mode is controlled by the ``retrieval_mode`` constructor argument:

    * ``"template"`` (default) – keyword matching selects the task category;
      *all* task-specific skills for that category are returned, and the first
      ``top_k`` general skills are returned in document order.  No embedding
      model is needed.

    * ``"embedding"`` – the task description is encoded with a
      SentenceTransformer model (Qwen3-Embedding-0.6B by default).  Both
      general skills and task-specific skills (searched across **all**
      categories) are ranked by cosine similarity and the top-k are returned.
      Skill embeddings are pre-computed once and cached in memory.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        skills_json_path: Optional[str] = None,
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
        task_specific_top_k: Optional[int] = None,
        device: Optional[str] = None,
        skill_retrieval_service_url: Optional[Union[str, List[str]]] = None,
        num_gpus: int = 1,
        skill_text_for_retrieval: str = "full",
        load_initial_skills: bool = True,
        similarity_threshold: Optional[float] = None,
        skill_retrieval_timeout: int = 60,
        retrieval_top_2k: Optional[int] = None,
        retrieval_alpha: Optional[float] = None,
        retrieval_ucb_c: float = 0.5,
        eviction_enabled: bool = False,
    ):
        """
        Args:
            skills_json_path:     Path to Claude-style skills JSON file. Can be None when load_initial_skills=False.
            skill_retrieval_timeout: Timeout in seconds for remote retrieval HTTP requests (default 60).
            retrieval_mode:       ``"template"`` or ``"embedding"``.
            skill_text_for_retrieval: Which skill fields to use as document input.
                                  ``"full"`` = title + principle + when_to_apply (default);
                                  ``"when_to_apply"`` = only when_to_apply;
                                  ``"principle"`` = only principle.
            load_initial_skills:  If False, do not load from skills_json_path; start with empty skill bank.
            similarity_threshold: If set (embedding mode), only return skills with similarity >= this value.
            embedding_model_path: Local path (or HF model ID) for the
                                  SentenceTransformer embedding model.  Only
                                  used when ``retrieval_mode="embedding"`` and
                                  ``skill_retrieval_service_url`` is not set.
                                  Defaults to ``"Qwen/Qwen3-Embedding-0.6B"``.
            task_specific_top_k:  Maximum number of task-specific skills to
                                  return.  ``None`` means *return all* in
                                  template mode and use ``top_k`` (general
                                  skills count) in embedding mode.
            device:               Device for the embedding model when running
                                  in-process (e.g. ``"cuda:0"``, ``"cpu"``).
                                  Only used when ``retrieval_mode="embedding"``
                                  and ``skill_retrieval_service_url`` is None.
                                  Defaults to SentenceTransformer default (often cuda:0).
            skill_retrieval_service_url: If set, retrieval is done via HTTP
                                  to this URL or list of URLs (batch endpoint).
                                  When a list (e.g. 8 URLs for 8 GPUs), queries
                                  are split across URLs and requested in parallel
                                  (e.g. 128 queries -> 8×16 to 8 servers).
                                  No local embedding model is loaded.
            num_gpus:             When > 1 and not using remote URL, load this many
                                  embedding models on cuda:0..num_gpus-1 and
                                  encode query batches in parallel (one URL server
                                  can thus use multiple GPUs).
            retrieval_top_2k:     For SimUtil-UCB: take this many by similarity first (default 2*top_k).
            retrieval_alpha:     SimUtil-UCB: score = alpha*sim + (1-alpha)*(utility+UCB). None => similarity only.
            retrieval_ucb_c:      SimUtil-UCB exploration coefficient c.
            eviction_enabled:   When True, trainer may call ``evict_excess_skills`` at validation steps.
        """
        if retrieval_mode not in ("template", "embedding"):
            raise ValueError(
                f"retrieval_mode must be 'template' or 'embedding', got '{retrieval_mode}'"
            )

        if load_initial_skills:
            if not skills_json_path or not os.path.exists(skills_json_path):
                raise FileNotFoundError(f"Skills file not found: {skills_json_path}")
            with open(skills_json_path, 'r') as f:
                loaded = json.load(f)
            self.skills = {
                "task_skills": loaded.get("task_skills") or [],
                "step_skills": loaded.get("step_skills") or [],
            }
        else:
            self.skills = {"task_skills": [], "step_skills": []}
        self.skills.setdefault("task_skills", [])
        self.skills.setdefault("step_skills", [])

        self.retrieval_mode = retrieval_mode
        self.embedding_model_path = embedding_model_path or "Qwen/Qwen3-Embedding-0.6B"
        self.task_specific_top_k = task_specific_top_k
        self.device = device
        self._num_gpus = max(1, int(num_gpus)) if not getattr(num_gpus, "__iter__", None) else 1
        # Normalize to list of URLs for single or multi-server parallel retrieval
        raw_url = skill_retrieval_service_url
        if raw_url is None:
            self._retrieval_service_urls = None
        elif isinstance(raw_url, str):
            u = raw_url.strip()
            self._retrieval_service_urls = [u] if u else None
        else:
            # list, tuple, or OmegaConf ListConfig
            self._retrieval_service_urls = [str(u).strip() for u in raw_url if (u or "").strip()]
            if not self._retrieval_service_urls:
                self._retrieval_service_urls = None

        if skill_text_for_retrieval not in ("full", "when_to_apply", "principle"):
            raise ValueError(
                f"skill_text_for_retrieval must be 'full', 'when_to_apply', or 'principle', got '{skill_text_for_retrieval}'"
            )
        self._skill_text_for_retrieval = skill_text_for_retrieval
        self.load_initial_skills = load_initial_skills
        self.similarity_threshold = similarity_threshold
        self._retrieval_timeout = max(60, int(skill_retrieval_timeout))
        self._retrieval_top_2k = retrieval_top_2k
        self._retrieval_alpha = retrieval_alpha
        self._retrieval_ucb_c = float(retrieval_ucb_c)
        self._eviction_enabled = bool(eviction_enabled)

        # Lazy-initialised embedding state (only used in embedding mode, in-process)
        self._embedding_model = None
        self._embedding_models: Optional[List[Any]] = None  # when _num_gpus > 1
        # Caches for task_skills and step_skills (keyed by retrieval_obs)
        self._task_skill_embeddings_cache: Optional[Dict] = None
        self._step_skill_embeddings_cache: Optional[Dict] = None

        self._normalize_all_skill_meta()
        n_tsk = len(self.skills.get('task_skills', []))
        n_stp = len(self.skills.get('step_skills', []))
        print(
            f"[SkillsOnlyMemory] Loaded skills: {n_tsk} task_skills, {n_stp} step_skills  "
            f"| retrieval_mode={retrieval_mode}"
            + (f" | remote={len(self._retrieval_service_urls)} server(s)" if self._retrieval_service_urls else "")
            + (f" | num_gpus={self._num_gpus}" if self._num_gpus > 1 else "")
        )

    def _normalize_skill_meta(self, skill: Dict[str, Any]) -> None:
        """Ensure dynamic memory fields exist; missing => 0. Modifies skill in place."""
        if skill.get("utility") is None:
            skill["utility"] = DEFAULT_UTILITY
        if skill.get("retrieval_count") is None:
            skill["retrieval_count"] = DEFAULT_RETRIEVAL_COUNT
        if skill.get("last_retrieval_step") is None:
            skill["last_retrieval_step"] = DEFAULT_LAST_RETRIEVAL_STEP
        if skill.get("created_at_step") is None:
            skill["created_at_step"] = DEFAULT_CREATED_AT_STEP

    def _normalize_all_skill_meta(self) -> None:
        """Normalize utility/retrieval_count/last_retrieval_step/created_at_step for all skills in pool."""
        for s in self.skills.get("task_skills", []):
            self._normalize_skill_meta(s)
        for s in self.skills.get("step_skills", []):
            self._normalize_skill_meta(s)

    # ------------------------------------------------------------------ #
    # Embedding helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_embedding_model(self):
        """Lazy-load the SentenceTransformer model(s). When _num_gpus > 1, load one per GPU and return the first."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embedding retrieval. "
                "Install with: pip install sentence-transformers"
            )
        # Prefer GPU when available; fall back to CPU
        target_device = self.device
        if not target_device:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(target_device).startswith("cuda") and not torch.cuda.is_available():
            target_device = "cpu"
            print(f"[SkillsOnlyMemory] CUDA not available, using CPU for embedding model.")

        if self._num_gpus > 1 and torch.cuda.is_available():
            if self._embedding_models is None:
                n = min(self._num_gpus, torch.cuda.device_count())
                print(f"[SkillsOnlyMemory] Loading {n} embedding models on cuda:0..{n-1}")
                self._embedding_models = [
                    SentenceTransformer(self.embedding_model_path, device=f"cuda:{i}")
                    for i in range(n)
                ]
                self._embedding_model = self._embedding_models[0]
                print(f"[SkillsOnlyMemory] {n} embedding models ready.")
            return self._embedding_models[0]
        if self._embedding_model is None:
            print(f"[SkillsOnlyMemory] Loading embedding model: {self.embedding_model_path} on {target_device}")
            self._embedding_model = SentenceTransformer(self.embedding_model_path, device=target_device)
            print(f"[SkillsOnlyMemory] Embedding model ready on {target_device}.")
        return self._embedding_model

    def _skill_to_text(self, skill: Dict[str, Any], mode: Optional[str] = None) -> str:
        """Build the text used as document input for retrieval. mode overrides self._skill_text_for_retrieval.
        If skill has non-empty 'retrieval_obs' (task+obs at error turn), use that for embedding (server/trainer auto-adapt)."""
        ro = (skill.get("retrieval_obs") or "").strip()
        if ro:
            return ro
        use = (mode or self._skill_text_for_retrieval)
        if use == "retrieval_obs":
            use = self._skill_text_for_retrieval
        if use == "when_to_apply":
            return (skill.get("when_to_apply") or "").strip()
        if use == "principle":
            return (skill.get("principle") or "").strip()
        parts = []
        for field in ("title", "principle", "when_to_apply"):
            val = (skill.get(field) or "").strip()
            if val:
                parts.append(val)
        return ". ".join(parts)

    def _skill_item_key(self, skill: Dict, index: int) -> str:
        """Stable key for cache reuse: skill_id if present, else index-based."""
        return (skill.get("skill_id") or f"_{index}")

    def replace_skills_keep_cache_incremental(self, new_skills: Dict[str, Any]) -> None:
        """
        Replace skill bank but keep embedding cache incremental when possible.
        When the new list is a prefix-preserving superset of the cached list (same skill_ids
        in order for the prefix), only encode the new tail and merge. Otherwise clear cache.
        Used by the retrieval server on reload_skills to avoid re-encoding the full set every sync.
        """
        import numpy as np
        new_skills = {
            "task_skills": new_skills.get("task_skills") or [],
            "step_skills": new_skills.get("step_skills") or [],
        }
        for pool, attr in [("task_skills", "_task_skill_embeddings_cache"), ("step_skills", "_step_skill_embeddings_cache")]:
            new_items = new_skills.get(pool, [])
            existing = getattr(self, attr, None)
            if not new_items:
                setattr(self, attr, None)
                continue
            cached_items = (existing.get("items", []) if existing else []) or []
            if not existing or len(cached_items) == 0 or len(new_items) < len(cached_items):
                setattr(self, attr, None)
                continue
            ids_cached = tuple(s.get("skill_id", f"_{i}") for i, s in enumerate(cached_items))
            ids_new_prefix = tuple(s.get("skill_id", f"_{i}") for i, s in enumerate(new_items[: len(cached_items)]))
            if ids_cached != ids_new_prefix:
                setattr(self, attr, None)
                continue
            tail = new_items[len(cached_items) :]
            if not tail:
                # Same list (or same length), just update items reference
                setattr(self, attr, {"items": list(new_items), "embeddings": existing["embeddings"]})
                continue
            # Encode only the tail and concatenate
            texts = [(s.get("retrieval_obs") or "").strip() or self._skill_to_text(s) for s in tail]
            model = self._get_embedding_model()
            tail_embs = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if hasattr(tail_embs, "shape") and len(tail_embs.shape) == 1:
                tail_embs = np.expand_dims(tail_embs, axis=0)
            old_embs = existing["embeddings"]
            merged = np.concatenate([old_embs, tail_embs], axis=0)
            setattr(self, attr, {"items": list(new_items), "embeddings": merged})
        self.skills = new_skills

    def _compute_pool_embeddings(self, pool: str) -> Optional[Dict]:
        """Compute and cache embeddings for task_skills or step_skills using retrieval_obs. pool is 'task_skills' or 'step_skills'.
        Reuses existing cache when skill set is unchanged (same length and same skill_ids order); cache is cleared on add_skills / load_skills / remove_skill."""
        import numpy as np
        items = self.skills.get(pool, [])
        attr = "_task_skill_embeddings_cache" if pool == "task_skills" else "_step_skill_embeddings_cache"
        existing = getattr(self, attr, None)
        if not items:
            if existing is None:
                setattr(self, attr, {"items": [], "embeddings": np.array([]).reshape(0, 0)})
            return getattr(self, attr)
        cached_items = existing.get("items", []) if existing else []
        if existing is not None and len(cached_items) == len(items):
            # Reuse only if skill_id sequence matches (avoid wrong reuse when same length but different content)
            ids_now = tuple(s.get("skill_id", f"_{i}") for i, s in enumerate(items))
            ids_cached = tuple(s.get("skill_id", f"_{i}") for i, s in enumerate(cached_items))
            if ids_now == ids_cached:
                return existing
        texts = [(s.get("retrieval_obs") or "").strip() or self._skill_to_text(s) for s in items]
        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if hasattr(embeddings, "shape") and len(embeddings.shape) == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        cache = {"items": list(items), "embeddings": embeddings}
        if pool == "task_skills":
            self._task_skill_embeddings_cache = cache
        else:
            self._step_skill_embeddings_cache = cache
        return cache

    def _apply_simutil_ucb(
        self,
        pool_name: str,
        indices_2k: List[int],
        sims_1d: "np.ndarray",
        top_k: int,
    ) -> List[int]:
        """
        From top-2k indices by similarity, compute score = alpha*sim + (1-alpha)*(utility+UCB)
        and return top_k indices by score. Uses pool's utility/retrieval_count.
        """
        import numpy as np
        items = self.skills.get(pool_name, [])
        alpha = self._retrieval_alpha
        c = self._retrieval_ucb_c
        if alpha is None or not indices_2k:
            return list(indices_2k[:top_k])
        N = sum(int(s.get("retrieval_count", 0)) for s in items)
        # When N=0 no skill has been retrieved yet; use log(2) so UCB is positive for exploration
        log_N = math.log(max(2, 1 + N)) if N >= 0 else math.log(2)
        scored: List[Tuple[float, int]] = []
        for idx in indices_2k:
            sim = float(sims_1d[idx])
            sim_norm = (sim + 1.0) / 2.0 if sim >= -1 else 0.0  # map [-1,1] -> [0,1]
            skill = items[idx] if 0 <= idx < len(items) else {}
            self._normalize_skill_meta(skill)
            u = float(skill.get("utility", 0))
            n = int(skill.get("retrieval_count", 0))
            denom = 1 + n
            exploration_bonus = c * (math.sqrt(log_N / denom) if log_N > 0 and denom > 0 else 0.0)
            score = alpha * sim_norm + (1.0 - alpha) * (u + exploration_bonus)
            scored.append((score, idx))
        scored.sort(key=lambda x: -x[0])
        return [idx for _, idx in scored[:top_k]]

    def _get_skill_ranking_meta(self, pool_name: str, idx: int, sim_val: float) -> Dict[str, Any]:
        """
        Return utility, UCB, and retrieval_score for a skill (for logging in retrieved_skills JSON).
        When retrieval_alpha is set: score = alpha*sim_norm + (1-alpha)*(utility+UCB).
        When not set: no UCB, retrieval_score = sim_norm (or similarity for display).
        """
        items = self.skills.get(pool_name, [])
        skill = items[idx] if 0 <= idx < len(items) else {}
        self._normalize_skill_meta(skill)
        u = float(skill.get("utility", 0))
        sim_norm = (float(sim_val) + 1.0) / 2.0 if float(sim_val) >= -1 else 0.0
        alpha = self._retrieval_alpha
        c = self._retrieval_ucb_c
        if alpha is None:
            return {"utility": u, "ucb": 0.0, "retrieval_score": sim_norm}
        n = int(skill.get("retrieval_count", 0))
        N = sum(int(s.get("retrieval_count", 0)) for s in items)
        # Standard UCB: use log(1+N). When N=0, log_N=0 and UCB naturally stays 0
        # without extra boosting or hard-forcing behavior.
        log_N = math.log(1 + N) if N > 0 else 0.0
        denom = 1 + n
        exploration_bonus = c * (math.sqrt(log_N / denom) if log_N > 0 and denom > 0 else 0.0)
        score = alpha * sim_norm + (1.0 - alpha) * (u + exploration_bonus)
        return {"utility": u, "ucb": exploration_bonus, "retrieval_score": score}

    def _enrich_remote_result_with_ranking_meta(
        self, result_list: List[Dict[str, Any]], pool_key: str
    ) -> List[Dict[str, Any]]:
        """
        After remote retrieval, attach utility/ucb/retrieval_score using **client's** pool
        so N = sum(retrieval_count) is from the client (updated by update_utilities_for_trajectory).
        Server-side N is often 0 until reload_skills, which would make recorded UCB always 0.
        """
        pool = "task_skills" if pool_key == "task_skills" else "step_skills"
        items = self.skills.get(pool, [])
        skill_id_to_idx = {s.get("skill_id"): i for i, s in enumerate(items) if s.get("skill_id")}
        for item in result_list:
            skills = item.get(pool_key, [])
            for sk in skills:
                sid = sk.get("skill_id")
                sim = sk.get("similarity")
                if sim is None:
                    continue
                sim_val = float(sim)
                if sid is not None and sid in skill_id_to_idx:
                    idx = skill_id_to_idx[sid]
                    meta = self._get_skill_ranking_meta(pool, idx, sim_val)
                else:
                    # Skill not in client pool (e.g. server has newer skills); still use client's N for UCB
                    meta = self._get_skill_ranking_meta_unknown_skill(pool, sim_val)
                sk.update(meta)
        return result_list

    def _get_skill_ranking_meta_unknown_skill(self, pool_name: str, sim_val: float) -> Dict[str, Any]:
        """UCB/score for a skill not in client pool: use client's N, n=0 (so UCB = c*sqrt(log_N))."""
        items = self.skills.get(pool_name, [])
        u = 0.0
        sim_norm = (float(sim_val) + 1.0) / 2.0 if float(sim_val) >= -1 else 0.0
        alpha = self._retrieval_alpha
        c = self._retrieval_ucb_c
        if alpha is None:
            return {"utility": u, "ucb": 0.0, "retrieval_score": sim_norm}
        n = 0
        N = sum(int(s.get("retrieval_count", 0)) for s in items)
        log_N = math.log(1 + N) if N > 0 else 0.0
        denom = 1 + n
        exploration_bonus = c * (math.sqrt(log_N / denom) if log_N > 0 and denom > 0 else 0.0)
        score = alpha * sim_norm + (1.0 - alpha) * (u + exploration_bonus)
        return {"utility": u, "ucb": exploration_bonus, "retrieval_score": score}

    def retrieve_task_skills_batch(
        self,
        task_descriptions: List[str],
        top_k: int = 6,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k task_skills for each task description (by retrieval_obs). Returns list of { task_skills: [...], query_text: task }."""
        pool = "task_skills"
        items = self.skills.get(pool, [])
        if not items:
            return [{"task_skills": [], "query_text": t} for t in task_descriptions]
        if self.retrieval_mode == "embedding" and self._retrieval_service_urls:
            timeout = getattr(self, "_retrieval_timeout", 60)
            return self._remote_retrieve_batch(
                task_descriptions, top_k=top_k, timeout=timeout, pool="task_skills"
            )
        if self.retrieval_mode == "embedding":
            import numpy as np
            cache = self._compute_pool_embeddings(pool)
            if cache["embeddings"].size == 0:
                return [{"task_skills": [], "query_text": t} for t in task_descriptions]
            model = self._get_embedding_model()
            query_embs = model.encode(
                task_descriptions,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if hasattr(query_embs, "shape") and len(query_embs.shape) == 1:
                query_embs = np.expand_dims(query_embs, axis=0)
            sims = cache["embeddings"] @ query_embs.T
            top_2k = self._retrieval_top_2k if self._retrieval_top_2k is not None else max(2 * top_k, top_k + 1)
            top_2k = min(top_2k, len(cache["items"]))
            out = []
            for j in range(len(task_descriptions)):
                s = np.asarray(sims[:, j]).ravel()
                idx_2k = np.lexsort((-np.arange(len(s)), -s))[:top_2k].tolist()
                if self._retrieval_alpha is not None:
                    idx_final = self._apply_simutil_ucb(pool, idx_2k, s, top_k)
                else:
                    idx_final = idx_2k[:top_k]
                skills = []
                for i in idx_final:
                    sk = dict(cache["items"][int(i)])
                    sk["similarity"] = float(s[int(i)])
                    sk.update(self._get_skill_ranking_meta(pool, int(i), float(s[int(i)])))
                    if self.similarity_threshold is None or sk["similarity"] >= self.similarity_threshold:
                        skills.append(sk)
                out.append({"task_skills": skills, "query_text": task_descriptions[j]})
            return out
        # Template: return first top_k
        return [{"task_skills": [dict(s) for s in items[:top_k]], "query_text": t} for t in task_descriptions]

    def retrieve_step_skills_batch(
        self,
        query_texts: List[str],
        top_k: int = 6,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k step_skills for each query (task + Current observation: obs). Returns list of { step_skills: [...], query_text: query }."""
        pool = "step_skills"
        items = self.skills.get(pool, [])
        if not items:
            return [{"step_skills": [], "query_text": q} for q in query_texts]
        if self.retrieval_mode == "embedding" and self._retrieval_service_urls:
            timeout = getattr(self, "_retrieval_timeout", 60)
            return self._remote_retrieve_batch(
                query_texts, top_k=top_k, timeout=timeout, pool="step_skills"
            )
        if self.retrieval_mode == "embedding":
            import numpy as np
            cache = self._compute_pool_embeddings(pool)
            if cache["embeddings"].size == 0:
                return [{"step_skills": [], "query_text": q} for q in query_texts]
            model = self._get_embedding_model()
            query_embs = model.encode(
                query_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if hasattr(query_embs, "shape") and len(query_embs.shape) == 1:
                query_embs = np.expand_dims(query_embs, axis=0)
            sims = cache["embeddings"] @ query_embs.T
            top_2k = self._retrieval_top_2k if self._retrieval_top_2k is not None else max(2 * top_k, top_k + 1)
            top_2k = min(top_2k, len(cache["items"]))
            out = []
            for j in range(len(query_texts)):
                s = np.asarray(sims[:, j]).ravel()
                idx_2k = np.lexsort((-np.arange(len(s)), -s))[:top_2k].tolist()
                if self._retrieval_alpha is not None:
                    idx_final = self._apply_simutil_ucb(pool, idx_2k, s, top_k)
                else:
                    idx_final = idx_2k[:top_k]
                skills = []
                for i in idx_final:
                    sk = dict(cache["items"][int(i)])
                    sk["similarity"] = float(s[int(i)])
                    sk.update(self._get_skill_ranking_meta(pool, int(i), float(s[int(i)])))
                    if self.similarity_threshold is None or sk["similarity"] >= self.similarity_threshold:
                        skills.append(sk)
                out.append({"step_skills": skills, "query_text": query_texts[j]})
            return out
        return [{"step_skills": [dict(s) for s in items[:top_k]], "query_text": q} for q in query_texts]

    def _remote_retrieve_batch(
        self,
        task_descriptions: List[str],
        top_k: int = 6,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Call the skill retrieval HTTP service(s) for batch retrieval.
        When multiple URLs are configured, queries are split across URLs and
        requested in parallel (e.g. 128 queries, 8 servers -> 8×16 concurrent).
        """
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        timeout = timeout if timeout is not None else self._retrieval_timeout
        urls = self._retrieval_service_urls
        n = len(urls)

        def _normalize_url(u: str) -> str:
            u = u.rstrip("/")
            return u if "/retrieve_batch" in u else f"{u}/retrieve_batch"

        def _request_one(url: str, queries: List[str], pool: Optional[str] = None) -> List[Dict[str, Any]]:
            payload = {
                "queries": queries,
                "top_k": top_k,
                "task_specific_top_k": kwargs.get("task_specific_top_k") or self.task_specific_top_k,
                "skill_text_for_retrieval": kwargs.get("skill_text_for_retrieval") or self._skill_text_for_retrieval,
            }
            if pool is not None:
                payload["pool"] = pool
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if "result" in data:
                return data["result"]
            return data

        if n == 0:
            return []

        def _apply_threshold_to_pool_result(result_list: List[Dict[str, Any]], pool_key: str) -> List[Dict[str, Any]]:
            """After remote retrieval, filter by similarity_threshold on client (server does not apply it)."""
            if self.similarity_threshold is None or not result_list:
                return result_list
            out_filtered = []
            for item in result_list:
                skills = item.get(pool_key, [])
                filtered = [s for s in skills if s.get("similarity") is not None and s["similarity"] >= self.similarity_threshold]
                out_filtered.append({**item, pool_key: filtered})
            return out_filtered

        raw = _request_one(_normalize_url(urls[0]), task_descriptions, kwargs.get("pool")) if n == 1 else None
        if n == 1:
            pool = kwargs.get("pool")
            pool_key = "task_skills" if pool == "task_skills" else "step_skills"
            if raw is not None:
                raw = self._enrich_remote_result_with_ranking_meta(raw, pool_key)
            if pool == "task_skills" and raw is not None:
                return _apply_threshold_to_pool_result(raw, "task_skills")
            if pool == "step_skills" and raw is not None:
                return _apply_threshold_to_pool_result(raw, "step_skills")
            return raw if raw is not None else []

        # Split queries across URLs (e.g. 128 -> 8 chunks of 16)
        size = len(task_descriptions)
        chunk_size = (size + n - 1) // n
        chunks = [
            task_descriptions[i : i + chunk_size]
            for i in range(0, size, chunk_size)
        ]
        # Pad to n chunks so we have one chunk per URL
        while len(chunks) < n:
            chunks.append([])
        chunks = chunks[:n]

        results_by_idx: List[Optional[List[Dict[str, Any]]]] = [None] * n
        with ThreadPoolExecutor(max_workers=n) as executor:
            pool = kwargs.get("pool")
            futures = {
                executor.submit(_request_one, _normalize_url(urls[i]), chunks[i], pool): i
                for i in range(n)
                if chunks[i]
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results_by_idx[idx] = fut.result()
        # Concatenate in order to preserve query order
        out = []
        for i in range(n):
            if results_by_idx[i] is not None:
                out.extend(results_by_idx[i])
        pool = kwargs.get("pool")
        pool_key = "task_skills" if pool == "task_skills" else "step_skills"
        if out:
            out = self._enrich_remote_result_with_ranking_meta(out, pool_key)
        if pool == "task_skills" and out:
            return _apply_threshold_to_pool_result(out, "task_skills")
        if pool == "step_skills" and out:
            return _apply_threshold_to_pool_result(out, "step_skills")
        return out

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        """
        Format retrieved skills (task_skills + step_skills) into a string for prompt injection.
        """
        sections = []
        task_skills = retrieved_memories.get('task_skills', [])
        if task_skills:
            lines = ["### Task-level experience (for this kind of task)"]
            for skill in task_skills:
                title = skill.get('title', '')
                principle = skill.get('principle', '')
                # when = skill.get('when_to_apply', '')
                lines.append(f"- **{title}**: {principle}")
                # if when:
                #     lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))
        step_skills = retrieved_memories.get('step_skills', [])
        if step_skills:
            lines = ["### Step-level experience (relevant to current situation)"]
            for skill in step_skills:
                title = skill.get('title', '')
                principle = skill.get('principle', '')
                # when = skill.get('when_to_apply', '')
                lines.append(f"- **{title}**: {principle}")
                # if when:
                #     lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))
        return "\n\n".join(sections) if sections else "No relevant skills found for this task."

    # ------------------------------------------------------------------ #
    # BaseMemory interface (not used in skills-only memory)               #
    # ------------------------------------------------------------------ #

    def reset(self, batch_size: int):
        pass

    def store(self, record: Dict[str, List[Any]]):
        pass

    def fetch(self, step: int):
        pass

    def __len__(self):
        return len(self.skills.get('task_skills', [])) + len(self.skills.get('step_skills', []))

    def __getitem__(self, idx: int):
        return self.skills

    # ------------------------------------------------------------------ #
    # Dynamic update methods                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_skill_text(s: Any) -> str:
        if s is None:
            return ""
        t = str(s).strip().lower()
        return re.sub(r"\s+", " ", t)

    def _skill_content_fingerprint(self, skill: Dict[str, Any]) -> str:
        """
        Content key for deduplication: same title/principle/when_to_apply/retrieval_obs
        => same skill (LLM often re-emits identical skills with new IDs across updates).
        """
        parts = [
            self._normalize_skill_text(skill.get("retrieval_obs")),
            self._normalize_skill_text(skill.get("title")),
            self._normalize_skill_text(skill.get("principle")),
            self._normalize_skill_text(skill.get("when_to_apply")),
        ]
        return "\x00".join(parts)

    def _pool_content_fingerprints(self, pool_name: str) -> set:
        seen = set()
        for s in self.skills.get(pool_name, []):
            fp = self._skill_content_fingerprint(s)
            if fp:
                seen.add(fp)
        return seen

    def _next_task_skill_id(self) -> str:
        """Return next task_xxx id (task_001, task_002, ...)."""
        pool = self.skills.get("task_skills", [])
        max_n = 0
        for s in pool:
            sid = (s.get("skill_id") or "")
            if sid.startswith("task_"):
                try:
                    max_n = max(max_n, int(sid[5:].lstrip("0") or "0"))
                except ValueError:
                    pass
        return f"task_{max_n + 1:03d}"

    def _next_step_skill_id(self) -> str:
        """Return next step_xxx id (step_001, step_002, ...)."""
        pool = self.skills.get("step_skills", [])
        max_n = 0
        for s in pool:
            sid = (s.get("skill_id") or "")
            if sid.startswith("step_"):
                try:
                    max_n = max(max_n, int(sid[5:].lstrip("0") or "0"))
                except ValueError:
                    pass
        return f"step_{max_n + 1:03d}"

    def add_skills(
        self,
        new_skills: List[Dict],
        category: str = 'task',
        created_at_step: Optional[int] = None,
        dedupe_by_content: bool = True,
    ) -> int:
        """
        Add new skills to the bank. category must be 'task' or 'step'.

        Args:
            new_skills: List of skill dicts (title, principle, when_to_apply; retrieval_obs set by caller).
            category:   'task' -> task_skills; 'step' -> step_skills.
            created_at_step: Optional global step when adding (for created_at_step field); default 0.
            dedupe_by_content: If True, skip skills whose normalized content matches an existing
                or already-added skill in the same pool (avoids task_090 vs task_100 duplicates).

        Returns:
            Number of skills actually added (duplicates skipped).
        """
        if category not in ("task", "step"):
            raise ValueError(f"category must be 'task' or 'step', got {category!r}")
        step = created_at_step if created_at_step is not None else DEFAULT_CREATED_AT_STEP
        added = 0
        skipped_content = 0
        existing_ids = self._get_all_skill_ids()
        pool_name = "task_skills" if category == "task" else "step_skills"
        content_keys = self._pool_content_fingerprints(pool_name) if dedupe_by_content else set()
        for skill in new_skills:
            skill = dict(skill)
            skill.setdefault("utility", DEFAULT_UTILITY)
            skill.setdefault("retrieval_count", DEFAULT_RETRIEVAL_COUNT)
            skill.setdefault("last_retrieval_step", DEFAULT_LAST_RETRIEVAL_STEP)
            # When created_at_step is provided by the caller, always overwrite it
            # with the current training step to avoid stale 0 values from LLM/templates.
            if created_at_step is not None:
                skill["created_at_step"] = int(created_at_step)
            else:
                skill.setdefault("created_at_step", step)
            fp = self._skill_content_fingerprint(skill) if dedupe_by_content else ""
            if dedupe_by_content and fp in content_keys:
                skipped_content += 1
                continue
            if category == "task":
                skill["skill_id"] = skill.get("skill_id") or self._next_task_skill_id()
                if skill["skill_id"] in existing_ids:
                    continue
                self.skills.setdefault("task_skills", []).append(skill)
                existing_ids.add(skill["skill_id"])
                if dedupe_by_content:
                    content_keys.add(fp)
                self._task_skill_embeddings_cache = None
            else:
                skill["skill_id"] = skill.get("skill_id") or self._next_step_skill_id()
                if skill["skill_id"] in existing_ids:
                    continue
                self.skills.setdefault("step_skills", []).append(skill)
                existing_ids.add(skill["skill_id"])
                if dedupe_by_content:
                    content_keys.add(fp)
                self._step_skill_embeddings_cache = None
            added += 1
        if skipped_content:
            print(f"[SkillsOnlyMemory] Skipped {skipped_content} skill(s) (duplicate content in {pool_name})")
        return added

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill by ID and invalidate the embedding cache."""
        removed = False
        for pool_name in ("task_skills", "step_skills"):
            pool = self.skills.get(pool_name, [])
            self.skills[pool_name] = [s for s in pool if s.get("skill_id") != skill_id]
            if len(self.skills[pool_name]) < len(pool):
                removed = True
                if pool_name == "task_skills":
                    self._task_skill_embeddings_cache = None
                else:
                    self._step_skill_embeddings_cache = None
        if removed:
            print(f"[SkillsOnlyMemory] Removed skill: {skill_id}")
        return removed

    def evict_excess_skills(
        self,
        current_step: int,
        max_task_skills: Optional[int] = None,
        max_step_skills: Optional[int] = None,
        protect_recent_steps: int = 0,
        score_c: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Shrink task_skills / step_skills pools when over max size.

        Deletion order: among skills **not** created in the last ``protect_recent_steps``
        (i.e. ``created_at_step <= current_step - protect_recent_steps``), sort by
        ``utility + score_c * ucb_bonus`` **ascending** (smallest first), where
        ``ucb_bonus`` uses the same formula as retrieval: ``c * sqrt(log(1+N)/(1+n))``
        per pool with ``N = sum(retrieval_count)``, ``n = skill's retrieval_count``.

        Returns:
            Dict with removed skill records and counts for logging / JSON audit.
        """
        out: Dict[str, Any] = {
            "current_step": int(current_step),
            "removed": [],
            "task_skills_before": 0,
            "task_skills_after": 0,
            "step_skills_before": 0,
            "step_skills_after": 0,
            "warnings": [],
        }
        c_ucb = float(self._retrieval_ucb_c)
        protect_recent_steps = max(0, int(protect_recent_steps))
        try:
            score_c = float(score_c)
            if math.isnan(score_c) or math.isinf(score_c):
                score_c = 1.0
        except (TypeError, ValueError):
            score_c = 1.0
        cutoff_created = int(current_step) - protect_recent_steps

        def _evict_pool(pool_name: str, max_size: Optional[int]) -> None:
            pool = self.skills.get(pool_name, [])
            key_before = f"{pool_name.replace('_skills', '')}_skills_before"
            key_after = f"{pool_name.replace('_skills', '')}_skills_after"
            if max_size is None or max_size < 0:
                n = len(pool)
                out[key_before] = out[key_after] = n
                return
            out[key_before] = len(pool)
            if len(pool) <= max_size:
                out[key_after] = len(pool)
                return
            excess = len(pool) - max_size
            N = sum(int(s.get("retrieval_count", 0)) for s in pool)
            log_N = math.log(1 + N) if N > 0 else 0.0
            # Always delete by index. Deleting by skill_id can over-delete when
            # duplicate skill_id values exist in the pool.
            candidates: List[Tuple[float, int, Dict[str, Any]]] = []
            for idx, skill in enumerate(pool):
                if not isinstance(skill, dict):
                    continue
                try:
                    self._normalize_skill_meta(skill)
                    created = int(skill.get("created_at_step", 0) or 0)
                    if created > cutoff_created:
                        continue
                    u = float(skill.get("utility", 0))
                    if math.isnan(u) or math.isinf(u):
                        u = 0.0
                    n = int(skill.get("retrieval_count", 0))
                    denom = 1 + n
                    ucb_bonus = c_ucb * (math.sqrt(log_N / denom) if log_N > 0 and denom > 0 else 0.0)
                    sort_key = u + score_c * ucb_bonus
                    if math.isnan(sort_key) or math.isinf(sort_key):
                        sort_key = u
                    candidates.append((sort_key, idx, skill))
                except (TypeError, ValueError):
                    continue
            candidates.sort(key=lambda x: x[0])
            picked = candidates[:excess]
            remove_idx = {c[1] for c in picked}
            if len(picked) < excess:
                out["warnings"].append(
                    f"{pool_name}: need_remove={excess} but only {len(picked)} deletable "
                    f"(protected by recent_steps={protect_recent_steps}); pool may stay above max."
                )
            if not picked:
                out[key_after] = len(pool)
                return

            self.skills[pool_name] = [s for i, s in enumerate(pool) if i not in remove_idx]
            if pool_name == "task_skills":
                self._task_skill_embeddings_cache = None
            else:
                self._step_skill_embeddings_cache = None
            for sort_key, pidx, skill in picked:
                try:
                    sk = float(sort_key)
                    if math.isnan(sk) or math.isinf(sk):
                        sk = 0.0
                except (TypeError, ValueError):
                    sk = 0.0
                try:
                    uu = float(skill.get("utility", 0) or 0)
                    if math.isnan(uu) or math.isinf(uu):
                        uu = 0.0
                except (TypeError, ValueError):
                    uu = 0.0
                try:
                    rc = int(skill.get("retrieval_count", 0) or 0)
                except (TypeError, ValueError):
                    rc = 0
                try:
                    cas = int(skill.get("created_at_step", 0) or 0)
                except (TypeError, ValueError):
                    cas = 0
                out["removed"].append(
                    {
                        "pool": pool_name,
                        "pool_index": int(pidx),
                        "skill_id": skill.get("skill_id") or f"(no_id_index={pidx})",
                        "title": (str(skill.get("title") or ""))[:500],
                        "utility": uu,
                        "retrieval_count": rc,
                        "created_at_step": cas,
                        "eviction_sort_key": sk,
                    }
                )
            out[key_after] = len(self.skills[pool_name])
            print(
                f"[SkillsOnlyMemory] Evicted {len(picked)} from {pool_name} "
                f"({out[key_before]} -> {out[key_after]}, max={max_size})"
            )

        _evict_pool("task_skills", max_task_skills)
        _evict_pool("step_skills", max_step_skills)
        return out

    def update_utilities_for_trajectory(
        self,
        skill_ids: List[str],
        credit: float,
        global_step: int,
        beta: float,
    ) -> int:
        """
        Update utility (EMA), retrieval_count, and last_retrieval_step for each skill
        that was retrieved in a trajectory. Used after rollout: credit = success (or success - baseline).

        new_utility = (1 - beta) * current_utility + beta * credit;
        retrieval_count += 1; last_retrieval_step = global_step.

        Args:
            skill_ids: Unique skill_id list (task + step skills retrieved in this trajectory).
            credit: Single credit for this trajectory (e.g. success 0/1 or success - success_baseline).
            global_step: Current training step.
            beta: EMA step size (e.g. 0.05--0.3).

        Returns:
            Number of skills updated.
        """
        if not skill_ids or beta <= 0:
            return 0
        seen: set = set()
        updated = 0
        for pool_name in ("task_skills", "step_skills"):
            for skill in self.skills.get(pool_name, []):
                sid = skill.get("skill_id")
                if sid not in skill_ids or sid in seen:
                    continue
                seen.add(sid)
                self._normalize_skill_meta(skill)
                u = float(skill.get("utility", 0))
                skill["utility"] = (1.0 - beta) * u + beta * credit
                skill["retrieval_count"] = int(skill.get("retrieval_count", 0)) + 1
                skill["last_retrieval_step"] = global_step
                updated += 1
        return updated

    def save_skills(self, path: str):
        """Persist the current skill bank to a JSON file."""
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.skills, f, indent=2)
        print(f"[SkillsOnlyMemory] Saved {len(self)} skills to {path}")

    def load_skills(self, path: str) -> bool:
        """
        Load skill bank from a JSON file (e.g. updated_skills_train_stepN.json).
        Use when resuming training to restore the skill state saved at checkpoint.

        Args:
            path: Path to the skills JSON file.

        Returns:
            True if loaded successfully, False if file not found or load failed.
        """
        if not path or not os.path.isfile(path):
            return False
        try:
            with open(path, 'r') as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                return False
            self.skills = {
                'task_skills': loaded.get('task_skills') or [],
                'step_skills': loaded.get('step_skills') or [],
            }
            self._normalize_all_skill_meta()
            self._task_skill_embeddings_cache = None
            self._step_skill_embeddings_cache = None
            print(f"[SkillsOnlyMemory] Loaded {len(self)} skills from {path}")
            return True
        except Exception as e:
            print(f"[SkillsOnlyMemory] Failed to load skills from {path}: {e}")
            return False

    def _get_all_skill_ids(self) -> set:
        ids = set()
        for s in self.skills.get('task_skills', []):
            if s.get('skill_id'):
                ids.add(s['skill_id'])
        for s in self.skills.get('step_skills', []):
            if s.get('skill_id'):
                ids.add(s['skill_id'])
        return ids

    def get_skill_count(self) -> Dict[str, int]:
        return {
            'task_skills': len(self.skills.get('task_skills', [])),
            'step_skills': len(self.skills.get('step_skills', [])),
            'total': len(self),
        }
