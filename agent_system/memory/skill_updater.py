"""
LLM-based skill updater that generates new skills from failed trajectories.
Supports Azure OpenAI, standard OpenAI-compatible APIs, and a **local** OpenAI-compatible
HTTP server (e.g. vLLM ``openai.api_server``) for self-hosted reflection.

Priority (first match wins):
    1) ``skill_llm_reflection_url`` (Hydra ``env.skills_only_memory.skill_llm_reflection_url``) or
       env ``SKILL_LLM_REFLECTION_URL`` → **batch** HTTP ``POST .../reflect_batch`` (multi-GPU server,
       one request per skill round; see ``examples/grpo_trainer/skill_reflection_server.py``).
    2) ``skill_llm_base_url`` / ``SKILL_LLM_BASE_URL`` → OpenAI-compatible client (e.g. vLLM ``/v1``).
    3) Azure OpenAI (``AZURE_OPENAI_API_KEY`` + ``AZURE_OPENAI_ENDPOINT``).
    4) Standard API: ``OPENAI_API_KEY`` + ``OPENAI_BASE_URL`` / ``OPENAI_MODEL``.

Local env (optional; overridden by Hydra when set there):
    SKILL_LLM_REFLECTION_URL – e.g. http://127.0.0.1:8126 (batch server; no ``/v1``)
    SKILL_LLM_BASE_URL       – e.g. http://127.0.0.1:8125/v1 (OpenAI-compatible)
    SKILL_LLM_MODEL          – served model name for OpenAI-compatible servers
    SKILL_LLM_API_KEY        – defaults to ``EMPTY`` for local OpenAI servers
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from openai import AzureOpenAI, OpenAI


def _normalize_openai_base_url(url: str) -> str:
    """Ensure base URL ends with /v1 for the OpenAI Python client."""
    u = url.strip().rstrip("/")
    if u.endswith("/v1"):
        return u
    return f"{u}/v1"


def skill_updater_kwargs_from_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build ``SkillUpdater`` kwargs from ``env.skills_only_memory`` (or similar dict)."""
    if not config:
        config = {}
    out: Dict[str, Any] = {
        "skill_gen_mode": config.get("skill_gen_mode", "task_step"),
        "max_concurrent": config.get("max_concurrent", 4),
        "max_completion_tokens": int(config.get("skill_llm_max_completion_tokens", 2048)),
        "skill_llm_base_url": config.get("skill_llm_base_url"),
        "skill_llm_model": config.get("skill_llm_model"),
        "skill_llm_api_key": config.get("skill_llm_api_key"),
        "skill_llm_reflection_url": config.get("skill_llm_reflection_url"),
        "skill_llm_reflection_timeout": int(config.get("skill_llm_reflection_timeout", 3600)),
    }
    return out


def _norm_skill_field(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _skill_fingerprint(skill: Dict) -> str:
    return "\x00".join(
        [
            _norm_skill_field(skill.get("retrieval_obs")),
            _norm_skill_field(skill.get("title")),
            _norm_skill_field(skill.get("principle")),
            _norm_skill_field(skill.get("when_to_apply")),
        ]
    )


def _dedupe_skills_by_content(skills: List[Dict]) -> List[Dict]:
    """One LLM round can return the same template for many trajectories; keep first only."""
    seen = set()
    out = []
    for s in skills:
        fp = _skill_fingerprint(s)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(s)
    return out


class SkillUpdater:
    def __init__(
        self,
        max_completion_tokens: int = 2048,
        model: Optional[str] = None,
        skill_gen_mode: str = "task_step",
        max_concurrent: int = 4,
        skill_llm_base_url: Optional[str] = None,
        skill_llm_model: Optional[str] = None,
        skill_llm_api_key: Optional[str] = None,
        skill_llm_reflection_url: Optional[str] = None,
        skill_llm_reflection_timeout: int = 3600,
        summarizer_model: Optional[str] = None,
    ):
        # Read credentials from environment variables — never hardcode secrets.
        reflection_url = (skill_llm_reflection_url or os.environ.get("SKILL_LLM_REFLECTION_URL") or "").strip()
        skill_url = (skill_llm_base_url or os.environ.get("SKILL_LLM_BASE_URL") or "").strip()
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        openai_model = model or os.environ.get("OPENAI_MODEL", "o3")

        # 1) Batch reflection server (HF multi-GPU): single POST /reflect_batch per round
        if reflection_url:
            self.skill_llm_reflection_url = reflection_url.rstrip("/")
            self.client = None
            self.skill_batch_timeout = int(
                os.environ.get("SKILL_LLM_REFLECTION_TIMEOUT", str(int(skill_llm_reflection_timeout)))
            )
            self.model = (
                summarizer_model
                or model
                or skill_llm_model
                or os.environ.get("SKILL_LLM_MODEL")
                or "skill-reflection-batch"
            )
            self.api_type = "skill_batch"
            print(
                f"[SkillUpdater] skill_llm_reflection_url={self.skill_llm_reflection_url} "
                f"(batch multi-GPU server, timeout={self.skill_batch_timeout}s)"
            )
        # 2) Local / self-hosted OpenAI-compatible server (vLLM, etc.)
        elif skill_url:
            api_key = (skill_llm_api_key or os.environ.get("SKILL_LLM_API_KEY") or "EMPTY").strip()
            base_url = _normalize_openai_base_url(skill_url)
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = (
                summarizer_model
                or model
                or skill_llm_model
                or os.environ.get("SKILL_LLM_MODEL")
                or openai_model
            )
            self.api_type = "skill_local"
            self.skill_llm_reflection_url = None
            self.skill_batch_timeout = int(skill_llm_reflection_timeout or 3600)
            print(
                f"[SkillUpdater] skill_llm_base_url={base_url} model={self.model} "
                f"(skill_local / OpenAI-compatible)"
            )
        # 3) Azure OpenAI
        elif azure_api_key and azure_endpoint:
            self.client = AzureOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
            self.model = summarizer_model or model or "o3"
            self.api_type = "azure"
            self.skill_llm_reflection_url = None
            self.skill_batch_timeout = int(skill_llm_reflection_timeout or 3600)
            print(f"[SkillUpdater] Using Azure OpenAI model={self.model}")
        # 4) Standard OpenAI-compatible (cloud)
        elif openai_api_key:
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_base_url,
            )
            self.model = summarizer_model or model or openai_model
            self.api_type = "openai"
            self.skill_llm_reflection_url = None
            self.skill_batch_timeout = int(skill_llm_reflection_timeout or 3600)
            print(f"[SkillUpdater] Using OpenAI-compatible API model={self.model}")
        else:
            raise EnvironmentError(
                "SkillUpdater requires one of:\n"
                "  - skill_llm_reflection_url or SKILL_LLM_REFLECTION_URL (batch /reflect_batch server), or\n"
                "  - skill_llm_base_url or SKILL_LLM_BASE_URL (local OpenAI-compatible server), or\n"
                "  - AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (Azure OpenAI), or\n"
                "  - OPENAI_API_KEY (standard OpenAI-compatible API)"
            )

        self.max_completion_tokens = max_completion_tokens
        self.update_history = []
        self.skill_gen_mode = (skill_gen_mode or "task_step").lower()
        if self.skill_gen_mode not in ("task_only", "step_only", "task_step"):
            self.skill_gen_mode = "task_step"
        self.max_concurrent = max(1, int(max_concurrent))
        self.retrieval_obs = True  # always use retrieval_obs for task/step skills
        print(f"[SkillUpdater] skill_gen_mode={self.skill_gen_mode} (retrieval: task_only=task_skills only, step_only=step_skills only, task_step=both)")

    def _reflect_batch_endpoint(self) -> str:
        assert self.skill_llm_reflection_url
        base = self.skill_llm_reflection_url.rstrip("/")
        if base.endswith("/reflect_batch"):
            return base
        return f"{base}/reflect_batch"

    def _reflect_batch_http(self, prompts: List[str]) -> List[str]:
        import requests

        url = self._reflect_batch_endpoint()
        resp = requests.post(
            url,
            json={"prompts": prompts, "max_completion_tokens": self.max_completion_tokens},
            timeout=self.skill_batch_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        responses = data.get("responses") or []
        if len(responses) < len(prompts):
            responses = list(responses) + [""] * (len(prompts) - len(responses))
        elif len(responses) > len(prompts):
            responses = responses[: len(prompts)]
        return responses

    def analyze_failures(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
        return_metadata: bool = False,
    ) -> List[Dict]:
        """
        Analyse failed trajectories and generate new skills to address the gaps.

        Args:
            failed_trajectories: List of dicts with keys:
                ``task``, ``trajectory``, ``task_type``; for summarize_success when
                grouped by uid, each item may also have ``success_trajectory`` (same
                shape, optional) and ``group_uid``, ``group_success_rate``, etc.
            current_skills: The current skill bank dict (with keys
                ``task_skills``, ``step_skills``)
            return_metadata: If True, returns tuple (skills, metadata) where metadata
                contains full prompt, raw response, and call info.

        Returns:
            If return_metadata=False: List of new skill dicts ready to be passed to
            ``SkillsOnlyMemory.add_skills()``.
            If return_metadata=True: Tuple of (skills_list, metadata_dict).
        """
        if not failed_trajectories:
            return [] if not return_metadata else ([], {})

        if self.skill_gen_mode in ("task_only", "step_only", "task_step"):
            # One LLM call per (failed + optional success): first_error_step, step_skill, task_skill; filter by mode
            to_process = failed_trajectories
            results_by_idx: Dict[int, Tuple[Optional[Dict], Optional[Dict], Optional[int], Optional[str]]] = {}
            if self.api_type == "skill_batch":
                prompts = [self._build_reflect_per_trajectory_prompt(t) for t in to_process]
                print(
                    f"[SkillUpdater] {self.skill_gen_mode}: {len(to_process)} trajectories -> "
                    f"single batch POST {self._reflect_batch_endpoint()} (multi-GPU server splits work)..."
                )
                try:
                    raw_responses = self._reflect_batch_http(prompts)
                except Exception as e:
                    print(f"[SkillUpdater] batch reflect failed: {e}")
                    raw_responses = [""] * len(to_process)
                for i, traj in enumerate(to_process):
                    raw = raw_responses[i] if i < len(raw_responses) else ""
                    task_query = self._short_task_for_summarizer(traj)
                    error_turn, step_skill, task_skill = self._parse_reflect_response(raw)
                    if task_skill and task_query:
                        task_skill = dict(task_skill)
                        task_skill["retrieval_obs"] = task_query
                    results_by_idx[i] = (step_skill, task_skill, error_turn, raw or None)
            else:
                max_workers = min(self.max_concurrent, len(to_process))
                print(f"[SkillUpdater] {self.skill_gen_mode}: {len(to_process)} parallel reflection calls (max_workers={max_workers})...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._reflect_one_trajectory, traj, i): i
                        for i, traj in enumerate(to_process)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            step_skill, task_skill, error_turn, raw = future.result()
                            results_by_idx[idx] = (step_skill, task_skill, error_turn, raw or None)
                        except Exception as e:
                            print(f"[SkillUpdater] {self.skill_gen_mode} idx={idx} failed: {e}")
                            results_by_idx[idx] = (None, None, None, None)
            step_skills = []
            task_skills = []
            for i, traj in enumerate(to_process):
                step_skill, task_skill, error_turn, _ = results_by_idx.get(i, (None, None, None, None))
                if self.skill_gen_mode in ("step_only", "task_step") and step_skill is not None:
                    step_skill = dict(step_skill)
                    step_skill["retrieval_obs"] = self._get_retrieval_obs_from_traj(traj, error_turn) or ""
                    step_skills.append(step_skill)
                if self.skill_gen_mode in ("task_only", "task_step") and isinstance(task_skill, dict) and (task_skill.get("title") or task_skill.get("principle")):
                    task_skills.append(task_skill)
            step_skills = _dedupe_skills_by_content(step_skills)
            task_skills = _dedupe_skills_by_content(task_skills)
            error_turns_list = [results_by_idx.get(i, (None, None, None, None))[2] for i in range(len(to_process))]
            raw_responses_list = [results_by_idx.get(i, (None, None, None, None))[3] for i in range(len(to_process))]
            self.update_history.append({
                'num_failures_analyzed': len(failed_trajectories),
                'num_step_skills': len(step_skills),
                'num_task_skills': len(task_skills),
            })
            if return_metadata:
                reflect_prompts = [self._build_reflect_per_trajectory_prompt(t) for t in to_process]
                metadata = {
                    'llm_model': self.model,
                    'num_failed_trajectories': len(failed_trajectories),
                    'step_skills': step_skills,
                    'task_skills': task_skills,
                    'error_turns': error_turns_list,
                    'summarizer_queries': reflect_prompts,
                    'raw_responses': raw_responses_list,
                    'mode': self.skill_gen_mode,
                }
                return step_skills, metadata
            return step_skills

        # Unreachable: skill_gen_mode is always one of task_only, step_only, task_step
        return [] if not return_metadata else ([], {})

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _strip_trailing_task_from_observation(self, text: str) -> str:
        """
        Remove trailing 'Your task is to: ...' from observation so retrieval_obs
        does not duplicate the task. AlfWorld/TextWorld often append this to the obs.
        """
        if not text or not text.strip():
            return text
        # Strip one or more trailing lines that look like "Your task is to: ..."
        while True:
            stripped = re.sub(r"\n\s*Your task is to:\s*.*$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            if stripped == text:
                break
            text = stripped
        return text.strip()

    def _extract_current_observation(self, obs: str) -> str:
        """
        Extract only the current step's observation from a string that may be the full
        prompt or multi-turn history. Removes "Your admissible actions of the current
        situation are: [...]" and keeps only the content after "Current observation: ";
        also strips trailing "Your task is to: ..." to avoid duplicating the task in retrieval_obs.
        """
        if not obs or len(obs.strip()) == 0:
            return obs
        result = None
        # 1) Take only what is after "Current observation: " (drops task + admissible actions prefix)
        m = re.search(
            r"Current observation:\s*(.*)$",
            obs,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            result = m.group(1).strip().strip("'\"").strip()
        if result is None:
            m = re.search(
                r"(?:your )?current observation is:\s*(.*?)(?=Your admissible actions|\n\s*Your admissible|$)",
                obs,
                re.DOTALL | re.IGNORECASE,
            )
            if m:
                result = m.group(1).strip().strip("'\"").strip()
        if result is None and ("Your admissible actions" in obs or "admissible actions of the current situation" in obs):
            idx = re.search(r"Your admissible actions", obs, re.IGNORECASE)
            if idx:
                before = obs[: idx.start()].strip()
                step_obs = re.search(
                    r"You are now at step \d+ and your current observation is:\s*(.*)",
                    before,
                    re.DOTALL | re.IGNORECASE,
                )
                if step_obs:
                    result = step_obs.group(1).strip().rstrip("'\"")
                elif "current observation is:" in before:
                    start = before.lower().rfind("current observation is:")
                    result = before[start:].split(":", 1)[-1].strip().strip("'\"").strip()
                else:
                    co = re.search(r"Current observation:\s*(.*)$", before, re.DOTALL | re.IGNORECASE)
                    if co:
                        result = co.group(1).strip().strip("'\"").strip()
        if result is None and "Turn " in obs and "Observation:" in obs:
            parts = re.split(r"Turn\s+\d+\s*:", obs, flags=re.IGNORECASE)
            if parts:
                last_part = parts[-1].strip()
                obs_match = re.match(r"(?:Observation:\s*)?(.*?)(?=Action:|$)", last_part, re.DOTALL | re.IGNORECASE)
                if obs_match:
                    result = obs_match.group(1).strip()
        if result is None:
            result = obs
        return self._strip_trailing_task_from_observation(result)

    def _extract_short_task_from_prompt(self, prompt: str) -> str:
        """
        Extract short task from full prompt when refined_trajectory is not available.
        Uses unified extract_short_task_for_retrieval (Your task is to: first, then [SEP] Instruction:).
        """
        from agent_system.memory.task_extraction import extract_short_task_for_retrieval
        return extract_short_task_for_retrieval(prompt or "")

    def _get_retrieval_obs_from_traj(self, traj: Dict, error_turn_1based: Optional[int]) -> str:
        """
        Build task + current observation string for retrieval_obs, in the same format
        as retrieval query_text: "task\\n\\nCurrent observation: <obs>".
        error_turn_1based: 1-based turn index where the agent went wrong; 0 or None = use task only.
        refined_trajectory["turns"] is 0-based (turns[0] = Turn 1), so we use turns[error_turn_1based - 1].
        When traj has refined_trajectory (e.g. AlfWorld), use ref["task"] (short task) so retrieval_obs
        matches the query format at inference; traj["task"] may be the full prompt and must not be used.
        When refined_trajectory is absent (e.g. WebShop), try _extract_short_task_from_prompt so
        retrieval_obs still matches inference (task + current obs).
        """
        ref = traj.get("refined_trajectory")
        raw_task = (ref.get("task") if ref is not None else None) or traj.get("task") or ""
        task = self._extract_short_task_from_prompt(raw_task) or (raw_task or "").strip()
        if not error_turn_1based or error_turn_1based < 1:
            return task
        obs = ""
        if ref is not None:
            turns = ref.get("turns", [])  # 0-based: turns[0] = Turn 1
            if 1 <= error_turn_1based <= len(turns):
                obs = (turns[error_turn_1based - 1].get("observation") or "").strip()
        else:
            # Non-refined: trajectory is [{"action": full_dialogue, "observation": ""}]; full_dialogue uses "Turn 1", "Turn 2", ... (1-based, same as build_failed_item)
            steps = traj.get("trajectory") or []
            if steps and isinstance(steps[0].get("action"), str):
                full_dialogue = steps[0].get("action", "")
                pattern = rf"Turn\s+{re.escape(str(error_turn_1based))}\s*:\s*(?:Observation:\s*)?(.*?)(?=Action:|Turn\s+\d+|$)"
                match = re.search(pattern, full_dialogue, re.DOTALL | re.IGNORECASE)
                if match:
                    obs = match.group(1).strip()
        # Normalize to "task + Current observation: obs" and avoid storing full prompt
        obs = self._extract_current_observation(obs) if obs else ""
        if not obs:
            return task
        return f"{task}\n\nCurrent observation: {obs}"

    def _build_reflect_per_trajectory_prompt(self, group_item: Dict) -> str:
        """Build prompt for one (failed + optional success) trajectory: output first_error_step, step-level skill, task-level summary."""
        task_text = self._short_task_for_summarizer(group_item)
        task_type = group_item.get("task_type", "")
        failed_text = self._format_trajectory_full(group_item)
        success_traj = group_item.get("success_trajectory")
        if success_traj:
            success_text = self._format_trajectory_full(success_traj)
            intro = (
                "You are given one failed trajectory and one successful trajectory for the same task. "
                "Analyze them and produce exactly three outputs.\n\n"
            )
            body = (
                f"Task: {task_text}\nTask Type: {task_type}\n\n"
                "Failed trajectory:\n"
                f"{failed_text}\n\n"
                "Successful trajectory (reference):\n"
                f"{success_text}\n\n"
            )
        else:
            intro = (
                "You are given one failed trajectory. Analyze it and produce exactly three outputs.\n\n"
            )
            body = (
                f"Task: {task_text}\nTask Type: {task_type}\n\n"
                "Failed trajectory:\n"
                f"{failed_text}\n\n"
            )
        return intro + body + """Output the following in order (use the exact section headers):

1) FIRST_ERROR_STEP: N
   Where N is the 1-based turn number in the *failed* trajectory where the agent first went wrong (e.g. Turn 1, Turn 2). Use 0 if unclear.

2) STEP_REFLECTION (one step-level experience for that step only):
   Output a JSON object with exactly: "title", "principle", "when_to_apply".
   This should be a concise experience for what to do at that specific step/situation (e.g. "At this step you should ...").

Example: {"title": "Check object location first", "principle": "Before picking, verify the object is in the expected receptacle.", "when_to_apply": "When the observation mentions an object but its location is unclear"}

3) TASK_REFLECTION (one task-level skill for the whole task):
   Output a JSON object with exactly: "title", "principle", "when_to_apply".
   This summarizes for the *whole task*: what to avoid and how to succeed in this kind of task (will be retrieved by task description).

Example: {"title": "Plan object location before acting", "principle": "For this task type, first identify where the object is, then plan the sequence of actions.", "when_to_apply": "When the task involves finding or moving a specific object"}

Output format (use these exact labels):
FIRST_ERROR_STEP: N

STEP_REFLECTION:
<single JSON object>

TASK_REFLECTION:
<single JSON object>"""

    def _parse_reflect_response(self, raw: str) -> Tuple[Optional[int], Optional[Dict], Optional[Dict]]:
        """Parse LLM output into (error_turn_1based, step_skill_dict, task_reflection_dict with task_query and summary)."""
        error_turn = None
        step_skill = None
        task_summary = None
        lines = raw.split("\n")
        # FIRST_ERROR_STEP
        for line in lines:
            line = line.strip()
            if line.upper().startswith("FIRST_ERROR_STEP:"):
                try:
                    n = int(line.split(":", 1)[1].strip().split()[0])
                    error_turn = max(0, n)
                except (ValueError, IndexError):
                    pass
                break
        # STEP_REFLECTION: single JSON object
        in_step = False
        step_buf = []
        for line in lines:
            if line.strip().upper().startswith("STEP_REFLECTION:"):
                in_step = True
                rest = line.split(":", 1)[-1].strip()
                if rest and rest.startswith("{"):
                    step_buf.append(rest)
                continue
            if in_step:
                if line.strip().upper().startswith("TASK_REFLECTION:"):
                    break
                step_buf.append(line)
        if step_buf:
            step_str = "\n".join(step_buf)
            try:
                json_start = step_str.find("{")
                json_end = step_str.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    obj = json.loads(step_str[json_start:json_end])
                    if isinstance(obj, dict) and obj.get("title") and obj.get("principle"):
                        step_skill = obj
            except json.JSONDecodeError:
                pass
        # TASK_REFLECTION: single JSON object (task-level skill)
        task_skill = None
        in_task = False
        task_buf = []
        for line in lines:
            if line.strip().upper().startswith("TASK_REFLECTION:"):
                in_task = True
                rest = line.split(":", 1)[-1].strip()
                if rest:
                    task_buf.append(rest)
                continue
            if in_task:
                task_buf.append(line.strip())
        if task_buf:
            task_str = "\n".join(task_buf)
            try:
                json_start = task_str.find("{")
                json_end = task_str.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    obj = json.loads(task_str[json_start:json_end])
                    if isinstance(obj, dict) and obj.get("title") and obj.get("principle"):
                        task_skill = obj
            except json.JSONDecodeError:
                pass
        return (error_turn, step_skill, task_skill)

    def _reflect_one_trajectory(
        self,
        group_item: Dict,
        idx: int,
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[int]]:
        """
        One LLM call for one (failed + optional success) trajectory.
        Returns (step_skill_dict, task_skill_dict, error_turn_1based).
        task_skill_dict has title, principle, when_to_apply (retrieval_obs set by caller = task_query).
        """
        prompt = self._build_reflect_per_trajectory_prompt(group_item)
        task_query = self._short_task_for_summarizer(group_item)
        try:
            if self.client is None:
                raise RuntimeError(
                    "SkillUpdater: OpenAI client not set (use skill_llm_reflection_url + batch server, not _reflect_one_trajectory)"
                )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_completion_tokens,
            )
            raw = (response.choices[0].message.content or "").strip()
            error_turn, step_skill, task_skill = self._parse_reflect_response(raw)
            if task_skill and task_query:
                task_skill = dict(task_skill)
                task_skill["retrieval_obs"] = task_query
            return (step_skill, task_skill, error_turn, raw)
        except Exception as e:
            print(f"[SkillUpdater] _reflect_one_trajectory idx={idx} failed: {e}")
            return (None, None, None, None)

    def _format_trajectory_full(self, traj: Dict) -> str:
        """Format full trajectory (no truncation) for summarizer. Used in summarize and summarize_success."""
        if "refined_trajectory" in traj:
            ref = traj["refined_trajectory"]
            turns = ref.get("turns", [])
            return self._format_refined_turns(turns)
        return self._format_trajectory(traj.get("trajectory", []))

    def _short_task_for_summarizer(self, traj: Dict) -> str:
        """Return short task only (for summarizer Task: line). Never the full concatenated prompt.
        Always run extraction so that even when refined_trajectory['task'] was stored as the full
        prompt (e.g. WebShop WITH_MEMORY), we get only the short task line."""
        raw = (traj.get("refined_trajectory") or {}).get("task") or traj.get("task", "") or ""
        return self._extract_short_task_from_prompt(raw)

    def get_error_turns_for_failed_trajectories(
        self, failed_trajectories: List[Dict]
    ) -> List[Optional[int]]:
        """
        Call reflect LLM on each failed trajectory to get FIRST_ERROR_STEP (1-based).
        Returns list of Optional[int] in same order as failed_trajectories.
        """
        if not failed_trajectories:
            return []
        if self.api_type == "skill_batch":
            prompts = [self._build_reflect_per_trajectory_prompt(t) for t in failed_trajectories]
            try:
                raw_responses = self._reflect_batch_http(prompts)
            except Exception as e:
                print(f"[SkillUpdater] get_error_turns batch failed: {e}")
                return [None] * len(failed_trajectories)
            if len(raw_responses) < len(failed_trajectories):
                raw_responses = list(raw_responses) + [""] * (len(failed_trajectories) - len(raw_responses))
            out: List[Optional[int]] = []
            for i in range(len(failed_trajectories)):
                raw = raw_responses[i] if i < len(raw_responses) else ""
                error_turn, _, _ = self._parse_reflect_response(raw or "")
                out.append(error_turn if error_turn is not None and error_turn >= 1 else None)
            return out

        max_workers = min(self.max_concurrent, len(failed_trajectories))
        order: List[int] = []
        results: Dict[int, Tuple[Optional[Dict], Optional[Dict], Optional[int]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._reflect_one_trajectory, traj, i): i
                for i, traj in enumerate(failed_trajectories)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    _, _, error_turn, _ = future.result()
                    results[idx] = (None, None, error_turn)
                    order.append(idx)
                except Exception as e:
                    print(f"[SkillUpdater] get_error_turns reflect task failed for idx {idx}: {e}")
                    results[idx] = (None, None, None)
                    order.append(idx)
        order.sort()
        out: List[Optional[int]] = [None] * len(failed_trajectories)
        for i in order:
            r = results.get(i)
            if r is not None:
                _, _, error_turn = r
                out[i] = error_turn if (error_turn is not None and error_turn >= 1) else None
        return out

    def _extract_action_tag_content(self, action_str: str) -> Optional[str]:
        """
        Extract only the content inside <action>...</action> from model output.
        Model often outputs <think>...</think><action>go to diningtable 1</action>; we keep only the action part
        to avoid long <think> blocks in summarizer input. If multiple <action> blocks, use the last one.
        Returns the content (possibly empty string) when tags are found; returns None when no
        <action>...</action> is found, so callers can treat that case as "no valid action" and use "".
        """
        if not action_str or not action_str.strip():
            return None
        matches = re.findall(r"<action>\s*(.*?)\s*</action>", action_str, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        return None

    def _format_refined_turns(self, turns: List[Dict], max_obs_len: int = 800) -> str:
        """
        Format refined trajectory turns (list of {observation, action}) for LLM prompt.
        turns is 0-based; we display Turn 1, Turn 2, ... so summarizer's ERROR_TURN: N (1-based) matches.
        Action is shortened to only the content inside <action>...</action> to avoid <think> bloat.
        """
        lines = []
        for idx, step in enumerate(turns, start=1):  # idx 1-based for display, step = turns[idx-1]
            obs = step.get('observation', '') or ''
            action = step.get('action', '') or ''
            if len(obs) > max_obs_len:
                obs = obs[:max_obs_len] + "..."
            extracted = self._extract_action_tag_content(action)
            action_short = extracted if extracted is not None else ""
            lines.append(f"Turn {idx}:")
            lines.append(f"  Observation: {obs}")
            lines.append(f"  Action: {action_short}")
        return '\n'.join(lines) if lines else "  (no steps)"

    def _format_trajectory(self, steps: List[Dict]) -> str:
        """
        Format trajectory steps for LLM prompt.
        
        Handles two formats:
        1. Full dialogue format: action contains "Initial Prompt: ... Turn 1: ..." (complete history)
        2. Old format: separate action and observation fields
        """
        lines = []
        for step in steps:
            action = step.get('action', 'unknown')
            obs = step.get('observation', '')
            
            # Check if this is the new full dialogue format
            if "Turn" in action and ("Observation:" in action or "Action:" in action):
                # This is complete dialogue history, use it directly (may be long but contains all info)
                # For very long dialogues, we could truncate, but for skill generation, full context is better
                # Limit to last 3000 chars to avoid exceeding token limits, but keep recent turns
                if len(action) > 3000:
                    # Keep the beginning (Initial Prompt) and the end (recent turns)
                    truncated = action[:500] + "\n... (truncated middle) ...\n" + action[-2500:]
                    lines.append(f"  Complete Dialogue History:\n{truncated}")
                else:
                    lines.append(f"  Complete Dialogue History:\n{action}")
            else:
                # Old format: separate action and observation
                obs_truncated = obs[:200] if obs else ""
                lines.append(f"  Action: {action}\n  Observation: {obs_truncated}")
        return '\n'.join(lines)

    def get_update_summary(self) -> Dict:
        if not self.update_history:
            return {'total_updates': 0, 'total_skills_generated': 0}
        total = 0
        for h in self.update_history:
            total += h.get('num_skills_generated', 0) + h.get('num_step_skills', 0) + h.get('num_task_skills', 0)
        return {
            'total_updates': len(self.update_history),
            'total_skills_generated': total,
            'all_skill_ids': [],
        }
