"""LLM-backed PatchPlanner with Search/Replace diff compilation and self-correction loop."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .incident_mapper import RepairContext
from .patch_executor import PatchExecutionResult, execute_patch_plan
from .patch_planner import (
    _DEFAULT_TEST_PREFIXES,
    _strip_excerpt_line_prefixes,
    PatchPlan,
    PatchPlanValidation,
    validate_patch_plan,
)
from .production_guard import is_emergency_stop_active, redact_secrets
from .repair_runs import RepairLesson, load_past_lessons


SUPPORTED_PROVIDERS = ("auto", "gemini", "groq", "openrouter", "openai", "anthropic")

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "openai/gpt-4o-mini",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
}

_PROVIDER_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY", "GROQ"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPENROUTER KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


def load_env_file(env_path: str | Path | None = None) -> dict[str, str]:
    """Load key-value pairs from a .env file (supports both KEY=VAL and legacy KEY: VAL)."""
    candidates: list[Path] = []
    if env_path is not None:
        candidates.append(Path(env_path))
    else:
        candidates.append(Path.cwd() / ".env")
        candidates.append(Path(__file__).resolve().parents[1] / ".env")

    loaded: dict[str, str] = {}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            content = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and value:
                loaded[key] = value
        break

    if env_path is not None:
        return loaded

    merged = dict(loaded)
    for k, v in os.environ.items():
        if v:
            merged[k] = v
    return merged


def resolve_provider_and_key(
    provider: str = "auto",
    env: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve an explicit or 'auto' provider name and its API key from env/.env."""
    env_map = env if env is not None else load_env_file()
    normalized = provider.strip().lower() or "auto"
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unsupported LLM provider {provider!r}; choose from {', '.join(SUPPORTED_PROVIDERS)}"
        )

    def _find_key(prov: str) -> str:
        for alias in _PROVIDER_KEY_ALIASES.get(prov, ()):
            val = env_map.get(alias, "").strip()
            if val:
                return val
        return ""

    if normalized == "auto":
        for candidate in ("gemini", "groq", "openrouter", "openai", "anthropic"):
            key = _find_key(candidate)
            if key:
                return candidate, key
        raise ValueError(
            "no LLM API key found in environment or .env (checked GEMINI_API_KEY, "
            "GROQ_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)"
        )

    key = _find_key(normalized)
    if not key:
        aliases = ", ".join(_PROVIDER_KEY_ALIASES.get(normalized, ()))
        raise ValueError(f"missing API key for provider {normalized!r} (expected {aliases} in .env or environment)")
    return normalized, key


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    """Extract and parse a JSON object from an LLM text response."""
    text = raw_response.strip()
    fence_match = _JSON_BLOCK_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response did not contain valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def build_repair_prompt(
    context: RepairContext,
    *,
    allowed_test_prefixes: tuple[tuple[str, ...], ...] = _DEFAULT_TEST_PREFIXES,
    previous_feedback: list[str] | None = None,
    past_lessons: list[RepairLesson] | None = None,
    challenge: str | None = None,
) -> tuple[str, str]:
    """Construct the system and user prompts from a deterministic RepairContext."""
    allowed_runners = [" ".join(prefix) for prefix in allowed_test_prefixes]
    eligible_evidence = [item for item in context.evidence if item.patch_eligible]

    system_prompt = (
        "You are ASRE (Autonomous Software Reliability Engineer). "
        "Your job is to diagnose a software incident from its deterministic RepairContext "
        "and return a minimal, safe JSON patch plan.\n\n"
        "STRICT RULES:\n"
        "1. Respond ONLY with a single valid JSON object (no markdown commentary outside JSON).\n"
        "2. You may ONLY modify files listed under PATCH-ELIGIBLE FILES.\n"
        "3. Prefer returning 'edits' with exact verbatim 'search' and 'replace' blocks taken from "
        "the RAW SOURCE WINDOWS below (do NOT include line numbers in 'search' or 'replace').\n"
        "4. Every command in 'tests_to_run' MUST start with 'python -m unittest' (preferred) or one of: "
        f"{', '.join(repr(r) for r in allowed_runners)} and MUST NOT contain shell operators (; & | < > ` $).\n"
        "5. Required JSON schema:\n"
        "{\n"
        '  "confirmed_evidence": ["Verified fact with file:line citation from context"],\n'
        '  "suspected_cause": "Root-cause hypothesis explaining why the confirmed evidence occurred",\n'
        '  "diagnosis": "Concise root-cause analysis and fix explanation",\n'
        '  "contributing_factors": ["Possible contributing factor such as recent commit churn or missing guard"],\n'
        '  "recommended_actions": ["Apply minimal code fix", "Run regression test suite"],\n'
        '  "deliberately_unchanged": ["Explain what callers, signatures, or tests were intentionally left unchanged and why"],\n'
        '  "risk_assessment": "Brief summary of change impact and blast radius",\n'
        '  "alternatives": ["Alternative approach considered and why the chosen fix is safer"],\n'
        '  "files_to_modify": ["relative/path.py"],\n'
        '  "edits": [\n'
        "    {\n"
        '      "path": "relative/path.py",\n'
        '      "search": "exact existing lines to replace",\n'
        '      "replace": "updated lines"\n'
        "    }\n"
        "  ],\n"
        '  "tests_to_run": ["python -m unittest tests/test_file.py"],\n'
        '  "confidence": 0.85\n'
        "}"
    )

    sections: list[str] = [
        "=== INCIDENT ===",
        f"ID: {context.incident.incident_id}",
        f"Exception Type: {context.incident.exception_type or 'unknown'}",
        f"Message: {context.incident.message}",
    ]
    if context.incident.stack_trace:
        sections.append(f"Stack Trace:\n{context.incident.stack_trace}")
    if context.incident.metadata:
        sections.append(f"Metadata: {json.dumps(context.incident.metadata)}")

    sections.append("\n=== PATCH-ELIGIBLE FILES ===")
    if not eligible_evidence:
        sections.append("(No patch-eligible files found in context)")
    for item in eligible_evidence:
        sections.append(
            f"\nFile: {item.path} (evidence_tier={item.evidence_tier.upper()}, "
            f"score={item.score:.1f}, reason={item.patch_eligibility_reason}, "
            f"risk={item.risk_label} {item.risk_score}/90, "
            f"direct_dependents={item.direct_dependents}, transitive_dependents={item.transitive_dependents}, "
            f"recent_git_changes_90d={item.recent_changes}, last_modified={item.last_modified or 'unknown'})"
        )
        if item.valid_frame_lines:
            sections.append(f"Confirmed Crash/Frame lines: {item.valid_frame_lines}")
        if item.calls:
            sections.append(f"Call relationships (invoked symbols): {', '.join(item.calls[:15])}")
        if item.api_endpoints:
            sections.append(f"API endpoints: {', '.join(item.api_endpoints)}")
        if item.db_interactions:
            sections.append(f"Database interactions: {', '.join(item.db_interactions)}")
        if item.service_calls:
            sections.append(f"External service dependencies: {', '.join(item.service_calls)}")
        if item.config_keys:
            sections.append(f"Configuration / env keys read: {', '.join(item.config_keys)}")
        if item.related_commits:
            commits_str = "; ".join(
                f"[{c['sha']}] {c['message']} ({c['author']}, {c['date']})"
                for c in item.related_commits
            )
            sections.append(f"Related git commits: {commits_str}")
        if item.affected_components:
            sections.append(f"Affected downstream components: {', '.join(item.affected_components)}")
        if item.contributing_factors:
            sections.append(f"Possible contributing factors: {'; '.join(item.contributing_factors)}")
        for excerpt in item.excerpts:
            raw_code = _strip_excerpt_line_prefixes(excerpt.content)
            sections.append(
                f"--- Excerpt {excerpt.excerpt_id} (lines {excerpt.start_line}-{excerpt.end_line}) ---\n"
                f"NUMBERED VIEW:\n{excerpt.content}\n\n"
                f"RAW SOURCE WINDOW (copy from here for 'search'):\n{raw_code}"
            )

    if past_lessons:
        pos_lessons = [l for l in past_lessons if getattr(l, "polarity", "positive") == "positive"]
        neg_lessons = [l for l in past_lessons if getattr(l, "polarity", "positive") == "negative"]
        if pos_lessons:
            sections.append("\n=== HISTORICAL LESSONS FROM PAST REPAIRS (TIER 2 & TIER 3 PLAYBOOKS) ===")
            for idx, lesson in enumerate(pos_lessons, start=1):
                op_tag = f" [operator={lesson.structural_operator}]" if getattr(lesson, "structural_operator", "") else ""
                sections.append(
                    f"{idx}. Past Incident {lesson.incident_id} ({lesson.status}){op_tag} on {', '.join(lesson.files_modified)}:\n"
                    f"   Error: {lesson.exception_type}: {lesson.message}\n"
                    f"   Verified Diagnosis: {lesson.diagnosis}"
                )
                if getattr(lesson, "unified_diff", ""):
                    sections.append(f"   Verified Patch Diff:\n{lesson.unified_diff[:400]}")
        if neg_lessons:
            sections.append("\n=== NEGATIVE TRAJECTORY GUARDRAILS (DO NOT REPEAT THESE FAILED / ROLLED-BACK FIXES) ===")
            for idx, lesson in enumerate(neg_lessons, start=1):
                warn_text = getattr(lesson, "anti_pattern_warning", "") or lesson.diagnosis
                sections.append(
                    f"{idx}. Anti-Pattern from {lesson.incident_id} ({lesson.status}) on {', '.join(lesson.files_modified)}:\n"
                    f"   {warn_text}"
                )

    if challenge and challenge.strip():
        sections.append(
            "\n=== ENGINEER CHALLENGE / FURTHER INVESTIGATION REQUEST ===\n"
            "The engineer challenged the conclusion or requested deeper investigation:\n"
            f'"{challenge.strip()}"\n'
            "You MUST directly evaluate and address this challenge in your root-cause hypothesis, "
            "diagnosis, contributing_factors, and patch plan."
        )

    if previous_feedback:
        sections.append("\n=== PREVIOUS ATTEMPT FEEDBACK (FIX THESE ERRORS) ===")
        for idx, feedback in enumerate(previous_feedback, start=1):
            sections.append(f"Attempt {idx} failure:\n{feedback}")

    user_prompt = redact_secrets("\n".join(sections))
    return system_prompt, user_prompt


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int = 60,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ASRE/1.0",
            **headers,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"LLM API HTTP {exc.code}: {err_body[:500]}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LLM API request failed: {exc}") from exc


def _call_provider_api(
    provider: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 60,
) -> str:
    env_map = env or {}
    if provider in {"groq", "openrouter", "openai"}:
        if provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
        elif provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
        else:
            url = env_map.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        response = _http_post_json(url, payload, headers, timeout=timeout)
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected {provider} response structure: {response}") from exc

    if provider == "gemini":
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        response = _http_post_json(url, payload, {}, timeout=timeout)
        try:
            return str(response["candidates"][0]["content"]["parts"][0]["text"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected gemini response structure: {response}") from exc

    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model,
            "max_tokens": 4096,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        response = _http_post_json(url, payload, headers, timeout=timeout)
        try:
            return str(response["content"][0]["text"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected anthropic response structure: {response}") from exc

    raise ValueError(f"unsupported provider: {provider}")


@dataclass
class RepairAttemptRecord:
    attempt: int
    plan: Optional[PatchPlan] = None
    validation: Optional[PatchPlanValidation] = None
    execution: Optional[PatchExecutionResult] = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "plan": self.plan.to_dict() if self.plan else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "execution": self.execution.to_dict() if self.execution else None,
            "error": self.error,
        }


@dataclass
class RepairLoopResult:
    provider: str
    model: str
    plan: Optional[PatchPlan]
    validation: Optional[PatchPlanValidation]
    execution: Optional[PatchExecutionResult]
    attempts: list[RepairAttemptRecord] = field(default_factory=list)
    lessons_used: list[RepairLesson] = field(default_factory=list)
    challenge: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "provider": self.provider,
            "model": self.model,
            "attempts_used": len(self.attempts),
            "lessons_used": [lesson.to_dict() for lesson in self.lessons_used],
            "plan": self.plan.to_dict() if self.plan else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "execution": self.execution.to_dict() if self.execution else None,
            "attempts": [a.to_dict() for a in self.attempts],
        }
        if self.challenge:
            result["challenge"] = self.challenge
        return result


class LLMPatchPlanner:
    """Provider-neutral LLM implementation of PatchPlanner."""

    def __init__(
        self,
        *,
        provider: str = "auto",
        model: str | None = None,
        env_path: str | Path | None = None,
        env: dict[str, str] | None = None,
        transport: Optional[Callable[[str, str], str]] = None,
        timeout: int = 60,
    ):
        self._env = env if env is not None else load_env_file(env_path)
        self._transport = transport
        self._timeout = timeout
        self._auto_mode = provider.strip().lower() in {"", "auto"}
        self._explicit_model = model
        if transport is not None and self._auto_mode:
            self.provider = "custom"
            self.api_key = ""
        else:
            self.provider, self.api_key = resolve_provider_and_key(provider, self._env)
        self.model = model or DEFAULT_MODELS.get(self.provider, "custom-model")

    def _candidate_providers(self) -> list[tuple[str, str, str]]:
        if not self._auto_mode:
            return [(self.provider, self.api_key, self.model)]
        candidates: list[tuple[str, str, str]] = []
        for prov in ("gemini", "groq", "openrouter", "openai", "anthropic"):
            for alias in _PROVIDER_KEY_ALIASES.get(prov, ()):
                val = self._env.get(alias, "").strip()
                if val:
                    candidates.append((prov, val, self._explicit_model or DEFAULT_MODELS[prov]))
                    break
        return candidates or [(self.provider, self.api_key, self.model)]

    def plan(
        self,
        context: RepairContext,
        *,
        previous_feedback: list[str] | None = None,
        past_lessons: list[RepairLesson] | None = None,
        challenge: str | None = None,
    ) -> PatchPlan:
        stop_active, stop_reason = is_emergency_stop_active(context.repository_path, self._env)
        if stop_active:
            raise RuntimeError(f"cannot generate patch plan: {stop_reason}")
        eligible = [item for item in context.evidence if item.patch_eligible]
        if not eligible:
            raise ValueError("cannot generate a patch plan: RepairContext has no patch-eligible files")

        system_prompt, user_prompt = build_repair_prompt(
            context,
            previous_feedback=previous_feedback,
            past_lessons=past_lessons,
            challenge=challenge,
        )
        if self._transport is not None:
            raw_text = self._transport(system_prompt, user_prompt)
            payload = _extract_json_object(raw_text)
            return PatchPlan.from_dict(payload, repo_root=context.repository_path)

        last_err: Exception | None = None
        for prov, key, mdl in self._candidate_providers():
            try:
                raw_text = _call_provider_api(
                    prov,
                    key,
                    mdl,
                    system_prompt,
                    user_prompt,
                    env=self._env,
                    timeout=self._timeout,
                )
                self.provider = prov
                self.api_key = key
                self.model = mdl
                payload = _extract_json_object(raw_text)
                return PatchPlan.from_dict(payload, repo_root=context.repository_path)
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(str(last_err) if last_err else "all configured LLM providers failed")


def plan_with_repair_loop(
    planner: LLMPatchPlanner,
    context: RepairContext,
    *,
    verify: bool = False,
    max_attempts: int = 2,
    repair_run_dir: str | None = None,
    challenge: str | None = None,
) -> RepairLoopResult:
    """
    Generate, validate, and optionally verify a PatchPlan with automatic error feedback.
    If attempt N fails schema/diff compilation, safety validation, or worktree test execution,
    the exact errors are fed back into attempt N+1.
    """
    attempts_limit = max(1, max_attempts)
    past_lessons = load_past_lessons(repair_run_dir, context) if repair_run_dir else []
    feedback_history: list[str] = []
    records: list[RepairAttemptRecord] = []
    last_plan: Optional[PatchPlan] = None
    last_validation: Optional[PatchPlanValidation] = None
    last_execution: Optional[PatchExecutionResult] = None

    for attempt_num in range(1, attempts_limit + 1):
        try:
            plan = planner.plan(
                context,
                previous_feedback=feedback_history or None,
                past_lessons=past_lessons or None,
                challenge=challenge,
            )
        except Exception as exc:
            err_msg = f"plan generation/compilation failed: {exc}"
            feedback_history.append(err_msg)
            records.append(RepairAttemptRecord(attempt=attempt_num, error=err_msg))
            continue

        last_plan = plan
        validation = validate_patch_plan(plan, context)
        last_validation = validation
        if not validation.valid:
            err_msg = "patch plan validation failed: " + "; ".join(validation.errors)
            feedback_history.append(err_msg)
            records.append(
                RepairAttemptRecord(
                    attempt=attempt_num,
                    plan=plan,
                    validation=validation,
                    error=err_msg,
                )
            )
            continue

        if not verify:
            records.append(
                RepairAttemptRecord(
                    attempt=attempt_num,
                    plan=plan,
                    validation=validation,
                )
            )
            break

        execution = execute_patch_plan(plan, context)
        last_execution = execution
        records.append(
            RepairAttemptRecord(
                attempt=attempt_num,
                plan=plan,
                validation=validation,
                execution=execution,
                error=execution.error,
            )
        )
        if execution.status == "verified":
            break

        cmd_failures = [
            f"command {' '.join(cmd.command)} exited {cmd.returncode}:\n{cmd.stderr or cmd.stdout}"
            for cmd in execution.commands
            if cmd.returncode != 0
        ]
        err_msg = f"worktree verification {execution.status} ({execution.error}):\n" + "\n".join(cmd_failures)
        feedback_history.append(err_msg)

    return RepairLoopResult(
        provider=planner.provider,
        model=planner.model,
        plan=last_plan,
        validation=last_validation,
        execution=last_execution,
        attempts=records,
        lessons_used=past_lessons,
        challenge=(challenge or "").strip(),
    )
