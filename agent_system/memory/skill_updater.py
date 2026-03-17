"""
LLM-based skill updater that generates new skills from failed trajectories.
Supports both Azure OpenAI and standard OpenAI-compatible APIs.

Required environment variables (choose one):
    Option 1 - Azure OpenAI:
        AZURE_OPENAI_API_KEY      – Azure OpenAI API key
        AZURE_OPENAI_ENDPOINT     – Azure OpenAI endpoint URL
        AZURE_OPENAI_API_VERSION  – API version (default: 2025-01-01-preview)
    
    Option 2 - Standard OpenAI-compatible API:
        OPENAI_API_KEY            – OpenAI API key
        OPENAI_BASE_URL           – OpenAI-compatible API base URL (optional, defaults to https://api.openai.com/v1)
        OPENAI_MODEL              – Model name (optional, defaults to "o3")

Summarizer (when skill_gen_mode=summarize):
    SUMMARIZER_OPENAI_MODEL      – Model for summarizing each trajectory (optional)
    SUMMARIZER_OPENAI_BASE_URL   – Base URL for summarizer (optional, defaults to OPENAI_BASE_URL)
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Union
from openai import AzureOpenAI, OpenAI


class SkillUpdater:
    def __init__(
        self,
        max_new_skills_per_update: int = 3,
        max_completion_tokens: int = 2048,
        model: Optional[str] = None,
        skill_gen_mode: str = "direct",
        summarizer_model: Optional[str] = None,
        summarizer_max_concurrent: int = 4,
        retrieval_obs: bool = False,
    ):
        # Read credentials from environment variables — never hardcode secrets.
        # Try Azure OpenAI first
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        
        # Try standard OpenAI-compatible API
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        openai_model = model or os.environ.get("OPENAI_MODEL", "o3")

        # Determine which API to use
        if azure_api_key and azure_endpoint:
            # Use Azure OpenAI
            self.client = AzureOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
            self.model = model or "o3"
            self.api_type = "azure"
        elif openai_api_key:
            # Use standard OpenAI-compatible API
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_base_url,
            )
            self.model = openai_model
            self.api_type = "openai"
        else:
            raise EnvironmentError(
                "SkillUpdater requires either:\n"
                "  - AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (for Azure OpenAI), or\n"
                "  - OPENAI_API_KEY (for standard OpenAI-compatible API)"
            )

        self.max_completion_tokens = max_completion_tokens
        self.max_new_skills_per_update = max_new_skills_per_update
        self.update_history = []
        self.skill_gen_mode = (skill_gen_mode or "direct").lower()
        self.summarizer_model = summarizer_model or os.environ.get("SUMMARIZER_OPENAI_MODEL") or self.model
        self.summarizer_max_concurrent = max(1, int(summarizer_max_concurrent))
        if self.skill_gen_mode == "summarize":
            print(f"[SkillUpdater] skill_gen_mode=summarize: summarizer_model={self.summarizer_model} (from config/env SUMMARIZER_OPENAI_MODEL or main model)")
        if self.skill_gen_mode == "summarize_success":
            print(f"[SkillUpdater] skill_gen_mode=summarize_success: one summary per group (failure + optional success), then one skill per group")
        self.retrieval_obs = bool(retrieval_obs)
        if self.retrieval_obs:
            print("[SkillUpdater] retrieval_obs=True: summarize will output error turn, skills will store task+obs at that turn for retrieval.")

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
                ``general_skills``, ``task_specific_skills``, etc.)
            return_metadata: If True, returns tuple (skills, metadata) where metadata
                contains full prompt, raw response, and call info.

        Returns:
            If return_metadata=False: List of new skill dicts ready to be passed to
            ``SkillsOnlyMemory.add_skills()``.
            If return_metadata=True: Tuple of (skills_list, metadata_dict).
        """
        if not failed_trajectories:
            return [] if not return_metadata else ([], {})

        next_dyn_idx = self._next_dyn_index(current_skills)
        summaries = []  # used in summarize mode; also saved in metadata for debugging

        retrieval_obs_list: Optional[List[str]] = None  # when retrieval_obs and summarize, one per summary

        if self.skill_gen_mode == "summarize":
            # Mode B: summarize each trajectory with summarizer LLM (parallel), then generate skills from summaries
            print(f"[SkillUpdater] Calling summarizer (model={self.summarizer_model}) for {len(failed_trajectories)} trajectories...")
            summaries_result = self._summarize_trajectories_parallel(failed_trajectories)
            if self.retrieval_obs and summaries_result and isinstance(summaries_result[0], tuple):
                summaries = [s[0] for s in summaries_result]
                retrieval_obs_list = [s[1] for s in summaries_result]
            else:
                summaries = summaries_result if (not summaries_result or isinstance(summaries_result[0], str)) else [s[0] for s in summaries_result]
            if not summaries:
                print("[SkillUpdater] No summaries produced, falling back to direct mode for this call")
                prompt = self._build_analysis_prompt(
                    failed_trajectories, current_skills, next_dyn_idx
                )
            else:
                print(f"[SkillUpdater] Summarize mode: got {len(summaries)} summaries, passing to main LLM")
                prompt = self._build_prompt_from_summaries(
                    summaries, current_skills, next_dyn_idx, strict_count=self.retrieval_obs
                )
        elif self.skill_gen_mode == "summarize_success":
            # Mode C: per-group summary (failure + optional success together), then one skill per group, unified update
            print(f"[SkillUpdater] summarize_success: summarizing {len(failed_trajectories)} groups (failure + optional success each)...")
            summaries_result = self._summarize_groups_failure_and_success_parallel(failed_trajectories)
            if self.retrieval_obs and summaries_result and isinstance(summaries_result[0], tuple):
                summaries = [s[0] for s in summaries_result]
                retrieval_obs_list = [s[1] for s in summaries_result]
            else:
                summaries = summaries_result if (not summaries_result or isinstance(summaries_result[0], str)) else [s[0] for s in summaries_result]
            if not summaries:
                print("[SkillUpdater] No group summaries produced, skipping update")
                return [] if not return_metadata else ([], {})
            print(f"[SkillUpdater] summarize_success: got {len(summaries)} group summaries, generating one skill per group")
            prompt = self._build_prompt_from_group_summaries(
                summaries, current_skills, next_dyn_idx, strict_count=self.retrieval_obs
            )
        else:
            # Mode A (direct): current behavior
            prompt = self._build_analysis_prompt(
                failed_trajectories, current_skills, next_dyn_idx
            )

        try:
            import time
            call_start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_completion_tokens,
            )
            
            call_end_time = time.time()
            raw_response_text = response.choices[0].message.content
            
            raw_skills = self._parse_skills_response(raw_response_text)

            # Reassign dyn_ IDs on our side to guarantee no collisions,
            # regardless of what the LLM returned.
            reassigned = self._reassign_dyn_ids(raw_skills, next_dyn_idx)
            # In default/summarize mode we ask for ~len(summaries) skills, so keep all returned (no fixed cap)
            final_skills = reassigned

            # When retrieval_obs was used, attach task+obs string to each skill for retrieval
            if retrieval_obs_list is not None:
                for i, skill in enumerate(final_skills):
                    skill["retrieval_obs"] = retrieval_obs_list[i] if i < len(retrieval_obs_list) else ""

            self.update_history.append({
                'num_failures_analyzed': len(failed_trajectories),
                'num_skills_generated': len(final_skills),
                'skill_ids': [s.get('skill_id') for s in final_skills],
            })

            if return_metadata:
                metadata = {
                    'llm_model': self.model,
                    'llm_api_type': self.api_type,
                    'llm_prompt': prompt,
                    'llm_raw_response': raw_response_text,
                    'llm_parsed_skills': raw_skills,  # Before ID reassignment
                    'llm_call_duration_seconds': call_end_time - call_start_time,
                    'llm_max_completion_tokens': self.max_completion_tokens,
                    'num_failed_trajectories': len(failed_trajectories),
                    'next_dyn_idx': next_dyn_idx,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(call_start_time)),
                }
                if self.skill_gen_mode == "summarize" and summaries:
                    metadata['summarizer_summaries'] = summaries  # 保存每条轨迹的 summary，便于调试
                if self.skill_gen_mode == "summarize_success" and summaries:
                    metadata['group_summaries'] = summaries
                return final_skills, metadata
            else:
                return final_skills

        except Exception as e:
            print(f"[SkillUpdater] Error calling o3: {e}")
            import traceback
            traceback.print_exc()
            if return_metadata:
                return [], {'error': str(e), 'traceback': traceback.format_exc()}
            return []

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _extract_current_observation(self, obs: str) -> str:
        """
        Extract only the current step's observation from a string that may be the full
        prompt or multi-turn history. Matches the format used by retrieval query_text
        (task + "Current observation: " + obs). Returns obs shortened to just the
        "current" observation when possible.
        """
        if not obs or len(obs.strip()) == 0:
            return obs
        # Retrieval-style or prompt fragment: ".... Current observation: <obs>" (colon, no "is")
        # Handles when anchor_obs / stored obs contains "Your admissible actions...\n\nCurrent observation: X"
        m = re.search(
            r"Current observation:\s*(.*)$",
            obs,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().strip("'\"").strip()
        # Webshop-style full prompt: "You are now at step N and your current observation is: '...'"
        # Capture from "current observation is:" until "Your admissible actions" (observation can contain quotes/newlines)
        m = re.search(
            r"(?:your )?current observation is:\s*(.*?)(?=Your admissible actions|\n\s*Your admissible|$)",
            obs,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().strip("'\"").strip()
        # Fallback: full prompt contains "Your admissible actions" – take only what precedes it
        if "Your admissible actions" in obs or "admissible actions of the current situation" in obs:
            idx = re.search(r"Your admissible actions", obs, re.IGNORECASE)
            if idx:
                before = obs[: idx.start()].strip()
                step_obs = re.search(
                    r"You are now at step \d+ and your current observation is:\s*(.*)",
                    before,
                    re.DOTALL | re.IGNORECASE,
                )
                if step_obs:
                    return step_obs.group(1).strip().rstrip("'\"")
                if "current observation is:" in before:
                    start = before.lower().rfind("current observation is:")
                    return before[start:].split(":", 1)[-1].strip().strip("'\"").strip()
                # "Current observation: X" in the part before "Your admissible actions"
                co = re.search(r"Current observation:\s*(.*)$", before, re.DOTALL | re.IGNORECASE)
                if co:
                    return co.group(1).strip().strip("'\"").strip()
        # Multiple "Observation:" in Turn format – keep only the last (current) one for this turn
        if "Turn " in obs and "Observation:" in obs:
            parts = re.split(r"Turn\s+\d+\s*:", obs, flags=re.IGNORECASE)
            if parts:
                last_part = parts[-1].strip()
                obs_match = re.match(r"(?:Observation:\s*)?(.*?)(?=Action:|$)", last_part, re.DOTALL | re.IGNORECASE)
                if obs_match:
                    return obs_match.group(1).strip()
        return obs

    def _extract_short_task_from_prompt(self, prompt: str) -> str:
        """
        Extract a short task string from a full prompt so retrieval_obs matches inference query format.
        Used when refined_trajectory is not available (e.g. WebShop). AlfWorld should use
        refined_trajectory["task"] instead.
        """
        if not prompt or not prompt.strip():
            return (prompt or "").strip()
        s = prompt.strip()
        # WebShop: " ... [SEP] Instruction: <task> [SEP] ..." -> task is the segment after "Instruction:"
        if " [SEP] " in s and "Instruction:" in s:
            parts = s.split(" [SEP] ")
            for i, p in enumerate(parts):
                if p.strip() == "Instruction:" and i + 1 < len(parts):
                    return parts[i + 1].strip()
        # AlfWorld-style (fallback when no refined_trajectory): "Your task is to: ..." until \n\n## or \n\n
        if "Your task is to:" in s:
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
        return s

    def _get_retrieval_obs_from_traj(self, traj: Dict, error_turn_1based: Optional[int]) -> str:
        """
        Build task + current observation string for retrieval_obs, in the same format
        as retrieval query_text: "task\\n\\nCurrent observation: <obs>".
        error_turn_1based: 1-based turn index where the agent went wrong; 0 or None = use task only.
        When traj has refined_trajectory (e.g. AlfWorld), use ref["task"] (short task) so retrieval_obs
        matches the query format at inference; traj["task"] may be the full prompt and must not be used.
        When refined_trajectory is absent (e.g. WebShop), try _extract_short_task_from_prompt so
        retrieval_obs still matches inference (task + current obs).
        """
        ref = traj.get("refined_trajectory")
        if ref is not None:
            task = (ref.get("task") or traj.get("task") or "").strip()
        else:
            raw_task = (traj.get("task") or "").strip()
            task = self._extract_short_task_from_prompt(raw_task) or raw_task
        if not error_turn_1based or error_turn_1based < 1:
            return task
        obs = ""
        if ref is not None:
            turns = ref.get("turns", [])
            if 1 <= error_turn_1based <= len(turns):
                obs = (turns[error_turn_1based - 1].get("observation") or "").strip()
        else:
            # Non-refined: trajectory is [{"action": full_dialogue, "observation": ""}]; parse Turn N: Observation: ...
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

    def _next_dyn_index(self, current_skills: Dict) -> int:
        """
        Scan the current skill bank for existing ``dyn_NNN`` IDs and return
        the next unused integer index (1-based).
        """
        max_idx = 0
        pattern = re.compile(r'^dyn_(\d+)$')

        for skill in current_skills.get('general_skills', []):
            m = pattern.match(skill.get('skill_id', ''))
            if m:
                max_idx = max(max_idx, int(m.group(1)))

        for skills in current_skills.get('task_specific_skills', {}).values():
            for skill in skills:
                m = pattern.match(skill.get('skill_id', ''))
                if m:
                    max_idx = max(max_idx, int(m.group(1)))

        return max_idx + 1

    def _reassign_dyn_ids(self, skills: List[Dict], start_idx: int) -> List[Dict]:
        """
        Replace whatever skill_id values the LLM returned with guaranteed-unique
        ``dyn_NNN`` IDs starting from ``start_idx``.
        """
        reassigned = []
        for i, skill in enumerate(skills):
            updated = dict(skill)
            updated['skill_id'] = f"dyn_{start_idx + i:03d}"
            reassigned.append(updated)
        return reassigned

    def _build_analysis_prompt(
        self,
        failed_trajectories: List[Dict],
        current_skills: Dict,
        next_dyn_idx: int,
    ) -> str:
        # Direct mode: per trajectory use only the last 10 steps to avoid token explosion.
        last_n_steps = 10
        max_examples = 10  # cap number of trajectories to avoid prompt overflow
        trajectories_to_use = failed_trajectories[:max_examples]
        n = len(trajectories_to_use)

        failure_examples = []
        for i, traj in enumerate(trajectories_to_use):
            if 'refined_trajectory' in traj:
                ref = traj['refined_trajectory']
                task_text = ref.get('task', traj['task'])
                turns = ref.get('turns', [])
                steps = turns[-last_n_steps:] if len(turns) > last_n_steps else turns
                trajectory_text = self._format_refined_turns(steps)
            else:
                task_text = traj['task']
                steps = traj['trajectory']
                steps = steps[-last_n_steps:] if len(steps) > last_n_steps else steps
                trajectory_text = self._format_trajectory(steps)
            failure_examples.append(
                f"\nExample {i + 1}:\n"
                f"Task: {task_text}\n"
                f"Task Type: {traj['task_type']}\n"
                f"Trajectory (last {last_n_steps} steps):\n"
                f"{trajectory_text}\n"
            )

        # Dynamic count: ask for ~n skills (align with summarize: one per failure), capped by max_new_skills_per_update
        num_skills = min(n, self.max_new_skills_per_update) if self.max_new_skills_per_update else n
        num_skills = max(1, num_skills)

        return f"""Analyze these failed agent trajectories and suggest NEW skills to add to the skill bank.

FAILED TRAJECTORIES:
{''.join(failure_examples)}

Generate about {num_skills} NEW actionable skills that would help avoid these failures (roughly one per example above; consolidate or split as appropriate).
Each skill must have: title (3-5 words), principle (1-2 sentences), when_to_apply. (skill_id is assigned automatically in code, omit it or use any placeholder.)

Return a JSON array of about that many skills, no other text.
Example format:
[{{"title": "Verify Object Location First", "principle": "Before attempting to pick up an object, always verify its current location by examining the environment.", "when_to_apply": "When the task requires moving an object but its location is uncertain"}}]
"""

    def _format_trajectory_full(self, traj: Dict) -> str:
        """Format full trajectory (no truncation) for summarizer. Used in summarize and summarize_success."""
        if "refined_trajectory" in traj:
            ref = traj["refined_trajectory"]
            turns = ref.get("turns", [])
            return self._format_refined_turns(turns)
        return self._format_trajectory(traj.get("trajectory", []))

    def _summarize_trajectory(self, traj: Dict) -> Union[str, Tuple[str, Optional[int]]]:
        """
        Send a single failed trajectory (full episode) to the summarizer model; return summary + optional error turn.
        When retrieval_obs is True, prompt asks for ERROR_TURN: N and returns (summary, N); else returns summary only.
        """
        task_text = traj.get("task", "")
        trajectory_text = self._format_trajectory_full(traj)
        if self.retrieval_obs:
            prompt = (
                "Summarize this failed agent trajectory and identify where the agent went wrong.\n\n"
                f"Task: {task_text}\nTask Type: {traj.get('task_type', '')}\n\n"
                f"Full trajectory:\n{trajectory_text}\n\n"
                "Provide: (1) a short summary of what the agent did, (2) where it likely failed and why. "
                "At the very end, add exactly one line: ERROR_TURN: N (where N is the 1-based turn number where the agent went wrong, or 0 if unclear)."
            )
        else:
            prompt = (
                "Summarize this failed agent trajectory and identify where the agent went wrong.\n\n"
                f"Task: {task_text}\nTask Type: {traj.get('task_type', '')}\n\n"
                f"Full trajectory:\n{trajectory_text}\n\n"
                "Provide: (1) a short summary of what the agent did, (2) where it likely failed and why."
            )
        try:
            response = self.client.chat.completions.create(
                model=self.summarizer_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024,
            )
            raw = (response.choices[0].message.content or "").strip()
            if not self.retrieval_obs:
                return raw
            error_turn = None
            for line in raw.split("\n")[::-1]:
                line = line.strip()
                if line.upper().startswith("ERROR_TURN:"):
                    try:
                        n = int(line.split(":", 1)[1].strip().split()[0])
                        error_turn = max(0, n)
                    except (ValueError, IndexError):
                        pass
                    break
            summary = raw
            if error_turn is not None and "ERROR_TURN:" in raw:
                summary = raw.rsplit("ERROR_TURN:", 1)[0].strip()
            return (summary, error_turn)
        except Exception as e:
            print(f"[SkillUpdater] Summarizer call failed: {e}")
            return "" if not self.retrieval_obs else ("", None)

    def _summarize_trajectories_parallel(
        self, failed_trajectories: List[Dict]
    ) -> Union[List[str], List[Tuple[str, str]]]:
        """
        Run summarizer on each trajectory in parallel.
        When retrieval_obs is False: return list of summary strings (skip empty).
        When retrieval_obs is True: return list of (summary, retrieval_obs_str) in same order as trajectories.
        """
        max_workers = min(self.summarizer_max_concurrent, len(failed_trajectories))
        order: List[int] = []
        results: Dict[int, Union[str, Tuple[str, Optional[int]]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._summarize_trajectory, traj): i
                for i, traj in enumerate(failed_trajectories)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    s = future.result()
                    if s:
                        results[idx] = s
                        order.append(idx)
                except Exception as e:
                    print(f"[SkillUpdater] Summarizer task failed: {e}")
        order.sort()
        if not self.retrieval_obs:
            return [results[i] for i in order if isinstance(results[i], str)]
        out: List[Tuple[str, str]] = []
        for i in order:
            r = results[i]
            if isinstance(r, tuple):
                summary, error_turn = r
                traj = failed_trajectories[i]
                if traj.get("query_texts") and error_turn and 1 <= error_turn <= len(traj["query_texts"]):
                    retrieval_obs_str = (traj["query_texts"][error_turn - 1] or "").strip()
                else:
                    retrieval_obs_str = ""
                if not retrieval_obs_str:
                    retrieval_obs_str = self._get_retrieval_obs_from_traj(traj, error_turn)
                out.append((summary, retrieval_obs_str))
        return out

    def _summarize_group_failure_and_success(self, group_item: Dict) -> Union[str, Tuple[str, Optional[int]]]:
        """
        Summarize one group: failed trajectory + optional success trajectory together.
        When retrieval_obs is True, also ask for ERROR_TURN and return (summary, error_turn).
        """
        task_text = group_item.get("task", "")
        task_type = group_item.get("task_type", "")
        failed_text = self._format_trajectory_full(group_item)
        success_traj = group_item.get("success_trajectory")
        if self.retrieval_obs:
            extra = " At the very end, add exactly one line: ERROR_TURN: N (1-based turn number in the *failed* trajectory where the agent went wrong, or 0 if unclear)."
        else:
            extra = ""
        if success_traj:
            success_text = self._format_trajectory_full(success_traj)
            prompt = (
                "Same task, two trajectories: one failed and one succeeded.\n\n"
                f"Task: {task_text}\nTask Type: {task_type}\n\n"
                "Failed trajectory:\n"
                f"{failed_text}\n\n"
                "Successful trajectory:\n"
                f"{success_text}\n\n"
                "Summarize in one short paragraph: (1) what went wrong in the failure, (2) what the success did right. "
                "Focus on the contrast so we can derive one actionable skill." + extra
            )
        else:
            prompt = (
                "Summarize this failed agent trajectory in one short paragraph: what went wrong and where.\n\n"
                f"Task: {task_text}\nTask Type: {task_type}\n\n"
                "Failed trajectory:\n"
                f"{failed_text}\n\n"
                "Provide a concise summary so we can derive one actionable skill." + extra
            )
        try:
            response = self.client.chat.completions.create(
                model=self.summarizer_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=512,
            )
            raw = (response.choices[0].message.content or "").strip()
            if not self.retrieval_obs:
                return raw
            error_turn = None
            for line in raw.split("\n")[::-1]:
                line = line.strip()
                if line.upper().startswith("ERROR_TURN:"):
                    try:
                        n = int(line.split(":", 1)[1].strip().split()[0])
                        error_turn = max(0, n)
                    except (ValueError, IndexError):
                        pass
                    break
            summary = raw
            if error_turn is not None and "ERROR_TURN:" in raw:
                summary = raw.rsplit("ERROR_TURN:", 1)[0].strip()
            return (summary, error_turn)
        except Exception as e:
            print(f"[SkillUpdater] Group summarizer call failed: {e}")
            return "" if not self.retrieval_obs else ("", None)

    def _summarize_groups_failure_and_success_parallel(
        self, group_items: List[Dict]
    ) -> Union[List[str], List[Tuple[str, str]]]:
        """Run per-group summarizer in parallel. When retrieval_obs, return list of (summary, retrieval_obs_str) in order."""
        max_workers = min(self.summarizer_max_concurrent, len(group_items))
        order: List[int] = []
        results: Dict[int, Union[str, Tuple[str, Optional[int]]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._summarize_group_failure_and_success, item): i
                for i, item in enumerate(group_items)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    s = future.result()
                    if s:
                        results[idx] = s
                        order.append(idx)
                except Exception as e:
                    print(f"[SkillUpdater] Group summarizer task failed: {e}")
        order.sort()
        if not self.retrieval_obs:
            return [results[i] for i in order if isinstance(results[i], str)]
        out: List[Tuple[str, str]] = []
        for i in order:
            r = results[i]
            if isinstance(r, tuple):
                summary, error_turn = r
                traj = group_items[i]
                if traj.get("query_texts") and error_turn and 1 <= error_turn <= len(traj["query_texts"]):
                    retrieval_obs_str = (traj["query_texts"][error_turn - 1] or "").strip()
                else:
                    retrieval_obs_str = ""
                if not retrieval_obs_str:
                    retrieval_obs_str = self._get_retrieval_obs_from_traj(traj, error_turn)
                out.append((summary, retrieval_obs_str))
        return out

    def _build_prompt_from_group_summaries(
        self,
        group_summaries: List[str],
        current_skills: Dict,
        next_dyn_idx: int,
        strict_count: bool = False,
    ) -> str:
        """Build prompt for main LLM. When strict_count=True, require exactly N skills (one per group summary)."""
        n = len(group_summaries)
        parts = [
            "The following are group summaries (each group: failure vs success contrast, or failure only). "
            "For each summary, output exactly one new actionable skill.\n"
        ]
        for i, s in enumerate(group_summaries, start=1):
            parts.append(f"\n--- Group summary {i} ---\n{s}")
        count_instruction = (
            f"Generate exactly {n} NEW skills (one per group summary above)."
            if strict_count
            else f"Generate about {n} NEW skills (roughly one per group summary above)."
        )
        parts.append(
            f"""

{count_instruction} Each skill should address the failure/success contrast or the failure pattern.
Each skill must have: title (3-5 words), principle (1-2 sentences), when_to_apply. (skill_id is assigned automatically in code, omit it or use any placeholder.)

Return a JSON array of exactly {n} skills, no other text.
Example format:
[{{"title": "...", "principle": "...", "when_to_apply": "..."}}, ...]
"""
        )
        return "".join(parts)

    def _build_prompt_from_summaries(
        self,
        summaries: List[str],
        current_skills: Dict,
        next_dyn_idx: int,
        strict_count: bool = False,
    ) -> str:
        """Build the skill-generation prompt from summarizer outputs. When strict_count=True, require exactly N skills (one per summary)."""
        n = len(summaries)
        parts = [
            "The following are summaries of failed agent trajectories and where they went wrong:\n"
        ]
        for i, s in enumerate(summaries, start=1):
            parts.append(f"\n--- Failure summary {i} ---\n{s}")
        count_instruction = (
            f"Generate exactly {n} NEW actionable skills (one per failure summary above)."
            if strict_count
            else f"Generate about {n} NEW actionable skills (roughly one per failure summary above; consolidate or split as appropriate)."
        )
        parts.append(
            f"""

{count_instruction}
Each skill must have: title (3-5 words), principle (1-2 sentences), when_to_apply. (skill_id is assigned automatically in code, omit it or use any placeholder.)

Return a JSON array of """ + ("exactly " + str(n) if strict_count else "about that many") + """ skills, no other text.
Example format:
[{{"title": "Verify Object Location First", "principle": "Before attempting to pick up an object, always verify its current location.", "when_to_apply": "When the task requires moving an object but its location is uncertain"}}]
"""
        )
        return "".join(parts)

    def _format_refined_turns(self, turns: List[Dict], max_obs_len: int = 800) -> str:
        """
        Format refined trajectory turns (list of {observation, action}) for LLM prompt.
        Used when trajectory has been refined (e.g. AlfWorld anchor_obs); outputs clear Turn N: Obs/Action.
        """
        lines = []
        for idx, step in enumerate(turns, start=1):
            obs = step.get('observation', '') or ''
            action = step.get('action', '') or ''
            if len(obs) > max_obs_len:
                obs = obs[:max_obs_len] + "..."
            lines.append(f"Turn {idx}:")
            lines.append(f"  Observation: {obs}")
            lines.append(f"  Action: {action}")
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

    def _parse_skills_response(self, response: str) -> List[Dict]:
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                skills = json.loads(response[json_start:json_end])
                return [
                    s for s in skills
                    if isinstance(s, dict) and all(k in s for k in ['title', 'principle'])
                ]
        except json.JSONDecodeError as e:
            print(f"[SkillUpdater] JSON parse error: {e}")
        return []

    def get_update_summary(self) -> Dict:
        if not self.update_history:
            return {'total_updates': 0, 'total_skills_generated': 0}
        return {
            'total_updates': len(self.update_history),
            'total_skills_generated': sum(h['num_skills_generated'] for h in self.update_history),
            'all_skill_ids': [sid for h in self.update_history for sid in h['skill_ids']],
        }
