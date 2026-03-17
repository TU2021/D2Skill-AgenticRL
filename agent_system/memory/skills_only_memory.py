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
import os
from typing import Dict, Any, List, Optional, Union
from .base import BaseMemory


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
    ):
        """
        Args:
            skills_json_path:     Path to Claude-style skills JSON file. Can be None when load_initial_skills=False.
            retrieval_mode:       ``"template"`` or ``"embedding"``.
            skill_text_for_retrieval: Which skill fields to use as document input.
                                  ``"full"`` = title + principle + when_to_apply (default);
                                  ``"when_to_apply"`` = only when_to_apply.
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
        """
        if retrieval_mode not in ("template", "embedding"):
            raise ValueError(
                f"retrieval_mode must be 'template' or 'embedding', got '{retrieval_mode}'"
            )

        if load_initial_skills:
            if not skills_json_path or not os.path.exists(skills_json_path):
                raise FileNotFoundError(f"Skills file not found: {skills_json_path}")
            with open(skills_json_path, 'r') as f:
                self.skills = json.load(f)
        else:
            self.skills = {"general_skills": [], "task_specific_skills": {}, "common_mistakes": []}

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

        if skill_text_for_retrieval not in ("full", "when_to_apply"):
            raise ValueError(
                f"skill_text_for_retrieval must be 'full' or 'when_to_apply', got '{skill_text_for_retrieval}'"
            )
        self._skill_text_for_retrieval = skill_text_for_retrieval
        self.load_initial_skills = load_initial_skills
        self.similarity_threshold = similarity_threshold

        # Lazy-initialised embedding state (only used in embedding mode, in-process)
        self._embedding_model = None
        self._embedding_models: Optional[List[Any]] = None  # when _num_gpus > 1
        self._skill_embeddings_cache: Optional[Dict] = None

        n_general = len(self.skills.get('general_skills', []))
        n_task = sum(len(v) for v in self.skills.get('task_specific_skills', {}).values())
        n_mistakes = len(self.skills.get('common_mistakes', []))
        print(
            f"[SkillsOnlyMemory] Loaded skills: {n_general} general, "
            f"{n_task} task-specific, {n_mistakes} mistakes  "
            f"| retrieval_mode={retrieval_mode}"
            + (f" | remote={len(self._retrieval_service_urls)} server(s)" if self._retrieval_service_urls else "")
            + (f" | num_gpus={self._num_gpus}" if self._num_gpus > 1 else "")
        )

        # In embedding mode without remote URL, pre-compute skill embeddings eagerly.
        if retrieval_mode == "embedding" and not self._retrieval_service_urls:
            self._compute_skill_embeddings()

    # ------------------------------------------------------------------ #
    # Task-type detection (template mode)                                  #
    # ------------------------------------------------------------------ #

    def _detect_task_type(self, task_description: str) -> str:
        """
        Infer the task category from ``task_description`` using keyword rules.

        Auto-detects whether the loaded skills belong to ALFWorld or WebShop
        by inspecting the task-specific skill keys.
        """
        task_specific = self.skills.get('task_specific_skills', {})
        goal = task_description.lower()

        # ---- ALFWorld categories ----------------------------------------
        if 'pick_and_place' in task_specific or 'clean' in task_specific:
            if 'look at' in goal and 'under' in goal:
                return 'look_at_obj_in_light'
            elif 'clean' in goal:
                return 'clean'
            elif 'heat' in goal:
                return 'heat'
            elif 'cool' in goal:
                return 'cool'
            elif 'examine' in goal or 'find' in goal:
                return 'examine'
            else:
                return 'pick_and_place'

        # ---- WebShop categories -----------------------------------------
        elif 'apparel' in task_specific or 'electronics' in task_specific:
            if any(kw in goal for kw in [
                'shirt', 'dress', 'jacket', 'pant', 'coat', 'sweater',
                'blouse', 'clothing', 'clothes', 't-shirt',
            ]):
                return 'apparel'
            elif any(kw in goal for kw in [
                'shoe', 'boot', 'sneaker', 'sandal', 'heel', 'slipper',
                'footwear',
            ]):
                return 'footwear'
            elif any(kw in goal for kw in [
                'laptop', 'phone', 'computer', 'tablet', 'charger',
                'cable', 'headphone', 'speaker', 'camera', 'electronic',
            ]):
                return 'electronics'
            elif any(kw in goal for kw in [
                'necklace', 'ring', 'bracelet', 'earring', 'watch',
                'jewelry', 'bag', 'purse', 'wallet',
            ]):
                return 'accessories'
            elif any(kw in goal for kw in [
                'furniture', 'lamp', 'curtain', 'pillow', 'bedding',
                'decor', 'candle', 'vase', 'rug',
            ]):
                return 'home_decor'
            elif any(kw in goal for kw in [
                'cream', 'lotion', 'shampoo', 'conditioner', 'moisturizer',
                'serum', 'makeup', 'beauty', 'vitamin', 'supplement',
            ]):
                return 'beauty_health'
            else:
                return 'other'

        # ---- Fallback: first key in task_specific_skills, or 'unknown' --
        else:
            return next(iter(task_specific), 'unknown')

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
        """Build the text used as document input for retrieval. mode overrides self._skill_text_for_retrieval."""
        use = (mode or self._skill_text_for_retrieval)
        if use == "when_to_apply":
            return (skill.get("when_to_apply") or "").strip()
        parts = []
        for field in ("title", "principle", "when_to_apply"):
            val = (skill.get(field) or "").strip()
            if val:
                parts.append(val)
        return ". ".join(parts)

    def _apply_similarity_threshold(
        self,
        general_skills: List[Dict[str, Any]],
        task_specific_skills: List[Dict[str, Any]],
    ) -> tuple:
        """If similarity_threshold is set (embedding mode), keep only skills with similarity >= threshold."""
        if self.similarity_threshold is None:
            return general_skills, task_specific_skills
        th = self.similarity_threshold
        # Only filter when we have numeric similarities (embedding mode); template mode has all None → leave as-is
        gs = general_skills
        ts = task_specific_skills
        if any(s.get("similarity") is not None for s in general_skills + task_specific_skills):
            gs = [s for s in general_skills if s.get("similarity") is not None and s["similarity"] >= th]
            ts = [s for s in task_specific_skills if s.get("similarity") is not None and s["similarity"] >= th]
        return gs, ts

    def _compute_skill_embeddings(self, mode: Optional[str] = None) -> Dict:
        """
        Pre-compute and cache normalised embeddings for every skill.
        mode: "full" | "when_to_apply"; defaults to self._skill_text_for_retrieval. Cache is per mode.
        """
        use_mode = mode or self._skill_text_for_retrieval
        if self._skill_embeddings_cache is None:
            self._skill_embeddings_cache = {}
        if use_mode in self._skill_embeddings_cache:
            return self._skill_embeddings_cache[use_mode]

        import numpy as np

        general_items = [
            ('general', None, s)
            for s in self.skills.get('general_skills', [])
        ]
        task_items = [
            ('task_specific', task_type, s)
            for task_type, skills in self.skills.get('task_specific_skills', {}).items()
            for s in skills
        ]
        all_items = general_items + task_items
        texts = [self._skill_to_text(item[2], use_mode) for item in all_items]

        if not texts:
            # Empty skill bank: avoid model.encode([]) which may return (0,0) and break matmul later
            embeddings = np.array([]).reshape(0, 0)
        else:
            model = self._get_embedding_model()
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        self._skill_embeddings_cache[use_mode] = {
            'items': all_items,
            'embeddings': embeddings,
            'n_general': len(general_items),
        }
        print(
            f"[SkillsOnlyMemory] Cached embeddings for {len(all_items)} skills (mode={use_mode}) "
            f"({len(general_items)} general + {len(task_items)} task-specific)"
        )
        return self._skill_embeddings_cache[use_mode]

    def _embedding_retrieve(
        self,
        task_description: str,
        top_k_general: int,
        top_k_task_specific: int,
    ):
        """
        Retrieve the most relevant general and task-specific skills using
        cosine similarity between the task description and all cached skill
        embeddings.

        Args:
            task_description:   Free-form task goal string.
            top_k_general:      Number of general skills to return.
            top_k_task_specific: Number of task-specific skills to return
                                 (searched across **all** categories).

        Returns:
            Tuple of (general_skills, task_specific_skills).
        """
        import numpy as np

        cache = self._compute_skill_embeddings()
        n_skills = cache['embeddings'].shape[0]
        if n_skills == 0:
            return [], [], [], []
        model = self._get_embedding_model()

        query_emb = model.encode(
            [task_description],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]  # shape: (dim,)

        sims = cache['embeddings'] @ query_emb  # cosine similarity, shape: (n,)

        n_general = cache['n_general']
        general_sims = sims[:n_general]
        task_sims = sims[n_general:]

        # Top-k general skills
        general_idx = np.argsort(general_sims)[::-1][:top_k_general]
        general_skills = [cache['items'][int(i)][2] for i in general_idx]
        general_scores = [float(general_sims[i]) for i in general_idx]

        # Top-k task-specific skills (cross-category search)
        task_idx = np.argsort(task_sims)[::-1][:top_k_task_specific]
        task_skills = [cache['items'][n_general + int(i)][2] for i in task_idx]
        task_scores = [float(task_sims[i]) for i in task_idx]

        return general_skills, task_skills, general_scores, task_scores

    def _embedding_retrieve_batch(
        self,
        task_descriptions: List[str],
        top_k_general: int,
        top_k_task_specific: int,
        mode: Optional[str] = None,
    ) -> List[tuple]:
        """
        Batch version of _embedding_retrieve: encode all queries once (or in
        parallel across num_gpus when > 1), then compute top-k for each.
        mode: override for skill document (full / when_to_apply); defaults to self._skill_text_for_retrieval.
        """
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not task_descriptions:
            return []

        cache = self._compute_skill_embeddings(mode)
        n_skills = cache['embeddings'].shape[0]
        if n_skills == 0:
            return [([], [], [], []) for _ in task_descriptions]

        models = self._embedding_models if self._num_gpus > 1 and self._embedding_models else [self._get_embedding_model()]
        n_models = len(models)

        if n_models <= 1:
            model = models[0]
            query_embs = model.encode(
                task_descriptions,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        else:
            # Split queries across GPUs and encode in parallel
            def _encode_chunk(model, chunk):
                if not chunk:
                    return None
                return model.encode(
                    chunk,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

            size = len(task_descriptions)
            chunk_size = (size + n_models - 1) // n_models
            chunks = [
                task_descriptions[i : i + chunk_size]
                for i in range(0, size, chunk_size)
            ]
            while len(chunks) < n_models:
                chunks.append([])
            chunks = chunks[:n_models]
            parts = []
            with ThreadPoolExecutor(max_workers=n_models) as ex:
                futs = {ex.submit(_encode_chunk, models[i], chunks[i]): i for i in range(n_models)}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    part = fut.result()
                    if part is not None and getattr(part, "shape", None) and len(part.shape) > 0 and part.shape[0] > 0:
                        parts.append((idx, part))
            parts.sort(key=lambda x: x[0])
            query_embs = np.vstack([p[1] for p in parts])
        if hasattr(query_embs, 'shape') and len(query_embs.shape) == 1:
            query_embs = np.expand_dims(query_embs, axis=0)
        n_queries = len(task_descriptions)
        n_general = cache['n_general']
        n_skills = cache['embeddings'].shape[0]

        sims_all = cache['embeddings'] @ query_embs.T
        if hasattr(sims_all, 'shape') and len(sims_all.shape) == 1:
            sims_all = np.expand_dims(sims_all, axis=1)

        results = []
        for j in range(n_queries):
            sims = sims_all[:, j]
            general_sims = sims[:n_general]
            task_sims = sims[n_general:]
            general_idx = np.argsort(general_sims)[::-1][:top_k_general]
            task_idx = np.argsort(task_sims)[::-1][:top_k_task_specific]
            general_skills = [cache['items'][int(i)][2] for i in general_idx]
            task_skills = [cache['items'][n_general + int(i)][2] for i in task_idx]
            general_scores = [float(general_sims[i]) for i in general_idx]
            task_scores = [float(task_sims[i]) for i in task_idx]
            results.append((general_skills, task_skills, general_scores, task_scores))
        return results

    def _remote_retrieve_batch(
        self,
        task_descriptions: List[str],
        top_k: int = 6,
        timeout: int = 60,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Call the skill retrieval HTTP service(s) for batch retrieval.
        When multiple URLs are configured, queries are split across URLs and
        requested in parallel (e.g. 128 queries, 8 servers -> 8×16 concurrent).
        """
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        urls = self._retrieval_service_urls
        n = len(urls)

        def _normalize_url(u: str) -> str:
            u = u.rstrip("/")
            return u if "/retrieve_batch" in u else f"{u}/retrieve_batch"

        def _request_one(url: str, queries: List[str]) -> List[Dict[str, Any]]:
            payload = {
                "queries": queries,
                "top_k": top_k,
                "task_specific_top_k": kwargs.get("task_specific_top_k") or self.task_specific_top_k,
                "skill_text_for_retrieval": kwargs.get("skill_text_for_retrieval") or self._skill_text_for_retrieval,
            }
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if "result" in data:
                return data["result"]
            return data

        if n == 0:
            return []
        if n == 1:
            return _request_one(_normalize_url(urls[0]), task_descriptions)

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
            futures = {
                executor.submit(_request_one, _normalize_url(urls[i]), chunks[i]): i
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
        return out

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        task_description: str,
        top_k: int = 6,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Retrieve skills for a given task description.

        Args:
            task_description: Current task goal string.
            top_k:            Number of *general* skills to include.
                              In embedding mode this also serves as the
                              default for task-specific skills when
                              ``task_specific_top_k`` is not set.

        Returns:
            Dictionary with keys:
              - ``general_skills``       – list of skill dicts
              - ``task_specific_skills`` – list of skill dicts
              - ``mistakes_to_avoid``    – list of common-mistake dicts
              - ``task_type``            – detected task type string
              - ``task_specific_examples`` – always ``[]`` (reserved)
              - ``retrieval_mode``       – which mode was used
        """
        common_mistakes = self.skills.get('common_mistakes', [])[:5]

        # ----------------------------------------------------------------
        # Embedding mode via remote service (no local model)
        # ----------------------------------------------------------------
        if self.retrieval_mode == "embedding" and self._retrieval_service_urls:
            batch = self._remote_retrieve_batch([task_description], top_k=top_k, **kwargs)
            if batch:
                out = batch[0]
                out.setdefault("mistakes_to_avoid", common_mistakes)
                out.setdefault("query_text", task_description)
                return out
            return {
                'query_text': task_description,
                'general_skills': [],
                'task_specific_skills': [],
                'mistakes_to_avoid': common_mistakes,
                'task_type': self._detect_task_type(task_description),
                'task_specific_examples': [],
                'retrieval_mode': 'embedding',
            }

        # ----------------------------------------------------------------
        # Embedding mode: semantic ranking of all skills (in-process)
        # ----------------------------------------------------------------
        if self.retrieval_mode == "embedding":
            ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
            res = self._embedding_retrieve(
                task_description=task_description,
                top_k_general=top_k,
                top_k_task_specific=ts_top_k,
            )
            general_skills, task_skills = res[0], res[1]
            general_scores = res[2] if len(res) >= 4 else [None] * len(general_skills)
            task_scores = res[3] if len(res) >= 4 else [None] * len(task_skills)
            gs = [{**dict(s), "similarity": general_scores[j]} for j, s in enumerate(general_skills)]
            ts = [{**dict(s), "similarity": task_scores[j]} for j, s in enumerate(task_skills)]
            gs, ts = self._apply_similarity_threshold(gs, ts)
            task_type = self._detect_task_type(task_description)
            return {
                'query_text': task_description,
                'general_skills': gs,
                'task_specific_skills': ts,
                'mistakes_to_avoid': common_mistakes,
                'task_type': task_type,
                'task_specific_examples': [],
                'retrieval_mode': 'embedding',
            }

        # ----------------------------------------------------------------
        # Template mode: keyword detection + return (sub)set of category skills
        # ----------------------------------------------------------------
        task_type = self._detect_task_type(task_description)
        general_skills = self.skills.get('general_skills', [])[:top_k]
        all_task_skills = self.skills.get('task_specific_skills', {}).get(task_type, [])

        if self.task_specific_top_k is not None:
            task_skills = all_task_skills[:self.task_specific_top_k]
        else:
            task_skills = all_task_skills
        gs = [{**dict(s), "similarity": None} for s in general_skills]
        ts = [{**dict(s), "similarity": None} for s in task_skills]
        gs, ts = self._apply_similarity_threshold(gs, ts)
        return {
            'query_text': task_description,
            'general_skills': gs,
            'task_specific_skills': ts,
            'mistakes_to_avoid': common_mistakes,
            'task_type': task_type,
            'task_specific_examples': [],
            'retrieval_mode': 'template',
        }

    def retrieve_batch(
        self,
        task_descriptions: List[str],
        top_k: int = 6,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve skills for multiple task descriptions in one call (batch encode
        in embedding mode). Returns one dict per query in the same order.

        Args:
            task_descriptions: List of task goal strings.
            top_k: Number of general skills per query (same as retrieve).
            **kwargs: Passed through to retrieve() in template mode; ignored in embedding mode.

        Returns:
            List of dicts with same structure as :meth:`retrieve`.
        """
        if not task_descriptions:
            return []

        common_mistakes = self.skills.get('common_mistakes', [])[:5]

        if self.retrieval_mode == "embedding" and self._retrieval_service_urls:
            kwargs.setdefault("skill_text_for_retrieval", self._skill_text_for_retrieval)
            batch = self._remote_retrieve_batch(task_descriptions, top_k=top_k, **kwargs)
            for i, d in enumerate(batch):
                d.setdefault("mistakes_to_avoid", common_mistakes)
                d.setdefault("query_text", task_descriptions[i] if i < len(task_descriptions) else "")
                gs, ts = self._apply_similarity_threshold(
                    d.get("general_skills", []), d.get("task_specific_skills", [])
                )
                d["general_skills"] = gs
                d["task_specific_skills"] = ts
            return batch

        if self.retrieval_mode == "embedding":
            ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
            mode = kwargs.get("skill_text_for_retrieval") or self._skill_text_for_retrieval
            batch_results = self._embedding_retrieve_batch(
                task_descriptions=task_descriptions,
                top_k_general=top_k,
                top_k_task_specific=ts_top_k,
                mode=mode,
            )
            out = []
            for i, res in enumerate(batch_results):
                gs, ts = res[0], res[1]
                g_scores = res[2] if len(res) >= 4 else [None] * len(gs)
                t_scores = res[3] if len(res) >= 4 else [None] * len(ts)
                gs_with_sim = [{**dict(s), "similarity": g_scores[j]} for j, s in enumerate(gs)]
                ts_with_sim = [{**dict(s), "similarity": t_scores[j]} for j, s in enumerate(ts)]
                gs_with_sim, ts_with_sim = self._apply_similarity_threshold(gs_with_sim, ts_with_sim)
                out.append({
                    'query_text': task_descriptions[i],
                    'general_skills': gs_with_sim,
                    'task_specific_skills': ts_with_sim,
                    'mistakes_to_avoid': common_mistakes,
                    'task_type': self._detect_task_type(task_descriptions[i]),
                    'task_specific_examples': [],
                    'retrieval_mode': 'embedding',
                })
            return out
        return [self.retrieve(q, top_k=top_k, **kwargs) for q in task_descriptions]

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        """
        Format retrieved skills into a string suitable for prompt injection.

        Args:
            retrieved_memories: Dict returned by :meth:`retrieve`.

        Returns:
            Formatted multi-section string to insert into the agent prompt.
        """
        sections = []
        task_type = retrieved_memories.get('task_type', 'unknown')
        mode = retrieved_memories.get('retrieval_mode', 'template')

        # General skills
        general_skills = retrieved_memories.get('general_skills', [])
        if general_skills:
            lines = ["### General Principles"]
            for skill in general_skills:
                title = skill.get('title', '')
                principle = skill.get('principle', '')
                lines.append(f"- **{title}**: {principle}")
            sections.append("\n".join(lines))

        # Task-specific skills
        task_skills = retrieved_memories.get('task_specific_skills', [])
        if task_skills:
            if mode == "embedding":
                section_title = "### Task-Relevant Skills"
            else:
                task_name = task_type.replace('_', ' ').title()
                section_title = f"### {task_name} Skills"
            lines = [section_title]
            for skill in task_skills:
                title = skill.get('title', '')
                principle = skill.get('principle', '')
                when = skill.get('when_to_apply', '')
                lines.append(f"- **{title}**: {principle}")
                if when:
                    lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))

        # Common mistakes
        mistakes = retrieved_memories.get('mistakes_to_avoid', [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for mistake in mistakes:
                desc = mistake.get('description', '')
                fix = mistake.get('how_to_avoid', '')
                if desc:
                    lines.append(f"- **Don't**: {desc}")
                    if fix:
                        lines.append(f"  **Instead**: {fix}")
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
        return (
            len(self.skills.get('general_skills', [])) +
            sum(len(v) for v in self.skills.get('task_specific_skills', {}).values()) +
            len(self.skills.get('common_mistakes', []))
        )

    def __getitem__(self, idx: int):
        return self.skills

    # ------------------------------------------------------------------ #
    # Dynamic update methods                                               #
    # ------------------------------------------------------------------ #

    def add_skills(self, new_skills: List[Dict], category: str = 'general') -> int:
        """
        Add new skills to the bank and invalidate the embedding cache.

        Args:
            new_skills: List of skill dicts to add.
            category:   ``'general'`` or a task-type key (e.g. ``'clean'``).

        Returns:
            Number of skills actually added (duplicates are skipped).
        """
        added = 0
        existing_ids = self._get_all_skill_ids()

        for skill in new_skills:
            skill_id = skill.get('skill_id')
            if skill_id in existing_ids:
                print(f"[SkillsOnlyMemory] Skipping duplicate skill: {skill_id}")
                continue

            if category == 'general':
                self.skills.setdefault('general_skills', []).append(skill)
            else:
                self.skills.setdefault('task_specific_skills', {}).setdefault(category, []).append(skill)
            added += 1
            print(f"[SkillsOnlyMemory] Added skill: {skill_id} - {skill.get('title', 'N/A')}")

        if added > 0:
            # Invalidate embedding cache so it is recomputed on next retrieve
            self._skill_embeddings_cache = {}

        return added

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill by ID and invalidate the embedding cache."""
        removed = False

        original_len = len(self.skills.get('general_skills', []))
        self.skills['general_skills'] = [
            s for s in self.skills.get('general_skills', [])
            if s.get('skill_id') != skill_id
        ]
        if len(self.skills.get('general_skills', [])) < original_len:
            removed = True

        for task_type in self.skills.get('task_specific_skills', {}):
            original_len = len(self.skills['task_specific_skills'][task_type])
            self.skills['task_specific_skills'][task_type] = [
                s for s in self.skills['task_specific_skills'][task_type]
                if s.get('skill_id') != skill_id
            ]
            if len(self.skills['task_specific_skills'][task_type]) < original_len:
                removed = True

        if removed:
            self._skill_embeddings_cache = {}
            print(f"[SkillsOnlyMemory] Removed skill: {skill_id}")
        return removed

    def save_skills(self, path: str):
        """Persist the current skill bank to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.skills, f, indent=2)
        print(f"[SkillsOnlyMemory] Saved {len(self)} skills to {path}")

    def _get_all_skill_ids(self) -> set:
        ids = set()
        for s in self.skills.get('general_skills', []):
            if s.get('skill_id'):
                ids.add(s['skill_id'])
        for task_skills in self.skills.get('task_specific_skills', {}).values():
            for s in task_skills:
                if s.get('skill_id'):
                    ids.add(s['skill_id'])
        return ids

    def get_skill_count(self) -> Dict[str, int]:
        return {
            'general': len(self.skills.get('general_skills', [])),
            'task_specific': sum(len(v) for v in self.skills.get('task_specific_skills', {}).values()),
            'common_mistakes': len(self.skills.get('common_mistakes', [])),
            'total': len(self),
        }
