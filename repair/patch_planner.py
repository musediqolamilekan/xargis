"""Provider-neutral patch-plan contracts and deterministic safety validation."""

from __future__ import annotations

import difflib
import json
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from .incident_mapper import RepairContext


_DIFF_HEADER = re.compile(r"^(---|\+\+\+) (?:a/|b/)?(?P<path>[^\t\n]+)(?:\t.*)?$", re.MULTILINE)
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)
_COMMAND_FORBIDDEN = re.compile(r"[;&|<>`$\r\n]")
_EXCERPT_PREFIX_RE = re.compile(r"^[> ]?\s*\d+:\s?(.*)$")
_SECURITY_DIFF_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("arbitrary code execution via eval()/exec()", re.compile(r"\b(?:eval|exec)\s*\(")),
    ("shell injection risk via shell=True or os.system()", re.compile(r"(?:\bsubprocess\.[A-Za-z_]+\s*\([^)]*shell\s*=\s*True|\bos\.system\s*\()")),
    ("unsafe deserialization via pickle.load/loads()", re.compile(r"\bpickle\.loads?\s*\(")),
    ("TLS certificate verification disabled (verify=False)", re.compile(r"\bverify\s*=\s*False\b")),
    (
        "hardcoded secret or API key literal",
        re.compile(
            r"(?:api_key|secret|password|token)\s*=\s*['\"](?:sk-[A-Za-z0-9_-]{12,}|gsk_[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9_-]{12,}|AKIA[0-9A-Z]{16})['\"]",
            re.IGNORECASE,
        ),
    ),
)
_DEFAULT_TEST_PREFIXES = (
    ("python", "-m", "unittest"),
    ("python3", "-m", "unittest"),
    ("venv/bin/python", "-m", "unittest"),
    ("pytest",),
    ("venv/bin/pytest",),
)


class PatchPlanner(Protocol):
    """An LLM or human adapter that produces a structured patch plan."""

    def plan(self, context: RepairContext) -> "PatchPlan":
        """Return a patch plan constrained to the supplied repair context."""


@dataclass(frozen=True)
class PatchEdit:
    """A deterministic search-and-replace block within a repository file."""
    path: str
    search: str
    replace: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PatchEdit":
        if not isinstance(payload, dict):
            raise ValueError("each edit must be a JSON object")
        path = payload.get("path") or payload.get("file")
        search = payload.get("search")
        replace = payload.get("replace")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("edit requires a non-empty 'path'")
        if not isinstance(search, str) or not search:
            raise ValueError(f"edit for {path!r} requires a non-empty 'search' block")
        if not isinstance(replace, str):
            raise ValueError(f"edit for {path!r} requires a string 'replace' block")
        return cls(path.strip(), search, replace)

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "search": self.search, "replace": self.replace}


def _strip_excerpt_line_prefixes(block: str) -> str:
    """Strip '>   12: ' or '    12: ' prefixes if an LLM copied them from a RepairContext excerpt."""
    lines = block.splitlines()
    if not lines:
        return block
    matches = [_EXCERPT_PREFIX_RE.match(line) for line in lines]
    if all(m is not None for m in matches):
        stripped = "\n".join(m.group(1) for m in matches if m is not None)
        if block.endswith("\n"):
            stripped += "\n"
        return stripped
    return block


def compile_edits_to_unified_diff(
    edits: Iterable[PatchEdit],
    repo_root: str,
) -> tuple[tuple[str, ...], str]:
    """Deterministically compile Search/Replace blocks against repo_root into a valid unified diff."""
    edit_list = list(edits)
    if not edit_list:
        raise ValueError("patch plan 'edits' must contain at least one edit")

    root = Path(repo_root).resolve()
    edits_by_path: dict[str, list[PatchEdit]] = {}
    ordered_paths: list[str] = []
    for edit in edit_list:
        norm = _normalise_repo_path(edit.path)
        if norm is None:
            raise ValueError(f"unsafe edit file path: {edit.path!r}")
        if norm not in edits_by_path:
            edits_by_path[norm] = []
            ordered_paths.append(norm)
        edits_by_path[norm].append(PatchEdit(norm, edit.search, edit.replace))

    diff_chunks: list[str] = []
    for rel_path in ordered_paths:
        target_file = (root / rel_path).resolve()
        if not str(target_file).startswith(str(root) + "/") or not target_file.is_file():
            raise ValueError(f"edit target file does not exist in repository: {rel_path}")
        original_raw = target_file.read_text(encoding="utf-8")
        original_normalized = original_raw.replace("\r\n", "\n")
        modified = original_normalized

        for edit in edits_by_path[rel_path]:
            search_norm = edit.search.replace("\r\n", "\n")
            replace_norm = edit.replace.replace("\r\n", "\n")
            if search_norm not in modified:
                stripped_search = _strip_excerpt_line_prefixes(search_norm)
                stripped_replace = _strip_excerpt_line_prefixes(replace_norm)
                if stripped_search and stripped_search in modified:
                    search_norm = stripped_search
                    replace_norm = stripped_replace
                else:
                    raise ValueError(
                        f"search block not found in {rel_path}: {edit.search.splitlines()[0] if edit.search.splitlines() else edit.search!r}"
                    )
            if search_norm == replace_norm:
                raise ValueError(f"edit in {rel_path} has identical search and replace blocks")
            modified = modified.replace(search_norm, replace_norm, 1)

        orig_lines = original_normalized.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)
        if orig_lines and not orig_lines[-1].endswith("\n"):
            orig_lines[-1] += "\n"
        if mod_lines and not mod_lines[-1].endswith("\n"):
            mod_lines[-1] += "\n"

        file_diff = "".join(
            difflib.unified_diff(
                orig_lines,
                mod_lines,
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )
        if not file_diff.strip():
            raise ValueError(f"edits for {rel_path} produced an empty diff")
        diff_chunks.append(file_diff if file_diff.endswith("\n") else file_diff + "\n")

    return tuple(ordered_paths), "".join(diff_chunks)


@dataclass(frozen=True)
class PatchPlan:
    diagnosis: str
    files_to_modify: tuple[str, ...]
    unified_diff: str
    tests_to_run: tuple[str, ...]
    confidence: float
    alternatives: tuple[str, ...] = ()
    risk_assessment: str = ""
    confirmed_evidence: tuple[str, ...] = ()
    suspected_cause: str = ""
    contributing_factors: tuple[str, ...] = ()
    recommended_actions: tuple[str, ...] = ()
    deliberately_unchanged: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        repo_root: str | None = None,
    ) -> "PatchPlan":
        if not isinstance(payload, dict):
            raise ValueError("patch plan must be a JSON object")
        diagnosis = payload.get("diagnosis")
        files = payload.get("files_to_modify")
        diff = payload.get("unified_diff")
        raw_edits = payload.get("edits")
        tests = payload.get("tests_to_run")
        confidence = payload.get("confidence")
        raw_alternatives = payload.get("alternatives") or ()
        risk_assessment = str(payload.get("risk_assessment") or "").strip()
        raw_confirmed = payload.get("confirmed_evidence") or ()
        suspected_cause = str(payload.get("suspected_cause") or "").strip()
        raw_factors = payload.get("contributing_factors") or ()
        raw_actions = payload.get("recommended_actions") or ()
        raw_unchanged = payload.get("deliberately_unchanged") or payload.get("untouched_reasons") or ()
        if not isinstance(diagnosis, str) or not diagnosis.strip():
            raise ValueError("patch plan requires a non-empty 'diagnosis'")

        if (not isinstance(diff, str) or not diff.strip()) and isinstance(raw_edits, list) and repo_root:
            parsed_edits = [PatchEdit.from_dict(item) for item in raw_edits]
            compiled_files, diff = compile_edits_to_unified_diff(parsed_edits, repo_root)
            if files is None:
                files = list(compiled_files)

        if not isinstance(files, list) or not all(isinstance(path, str) for path in files):
            raise ValueError("patch plan 'files_to_modify' must be a list of strings")
        if not isinstance(diff, str) or not diff.strip():
            raise ValueError("patch plan requires a non-empty 'unified_diff'")
        if not isinstance(tests, list) or not all(isinstance(command, str) for command in tests):
            raise ValueError("patch plan 'tests_to_run' must be a list of strings")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError("patch plan 'confidence' must be a number")
        alternatives: tuple[str, ...] = ()
        if isinstance(raw_alternatives, (list, tuple)):
            alternatives = tuple(str(a).strip() for a in raw_alternatives if str(a).strip())
        confirmed_evidence: tuple[str, ...] = ()
        if isinstance(raw_confirmed, (list, tuple)):
            confirmed_evidence = tuple(str(c).strip() for c in raw_confirmed if str(c).strip())
        contributing_factors: tuple[str, ...] = ()
        if isinstance(raw_factors, (list, tuple)):
            contributing_factors = tuple(str(f).strip() for f in raw_factors if str(f).strip())
        recommended_actions: tuple[str, ...] = ()
        if isinstance(raw_actions, (list, tuple)):
            recommended_actions = tuple(str(r).strip() for r in raw_actions if str(r).strip())
        deliberately_unchanged: tuple[str, ...] = ()
        if isinstance(raw_unchanged, (list, tuple)):
            deliberately_unchanged = tuple(str(u).strip() for u in raw_unchanged if str(u).strip())
        return cls(
            diagnosis.strip(),
            tuple(files),
            diff,
            tuple(tests),
            float(confidence),
            alternatives,
            risk_assessment,
            confirmed_evidence,
            suspected_cause,
            contributing_factors,
            recommended_actions,
            deliberately_unchanged,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "diagnosis": self.diagnosis,
            "files_to_modify": list(self.files_to_modify),
            "unified_diff": self.unified_diff,
            "tests_to_run": list(self.tests_to_run),
            "confidence": self.confidence,
        }
        if self.confirmed_evidence:
            result["confirmed_evidence"] = list(self.confirmed_evidence)
        if self.suspected_cause:
            result["suspected_cause"] = self.suspected_cause
        if self.contributing_factors:
            result["contributing_factors"] = list(self.contributing_factors)
        if self.recommended_actions:
            result["recommended_actions"] = list(self.recommended_actions)
        if self.deliberately_unchanged:
            result["deliberately_unchanged"] = list(self.deliberately_unchanged)
        if self.alternatives:
            result["alternatives"] = list(self.alternatives)
        if self.risk_assessment:
            result["risk_assessment"] = self.risk_assessment
        return result


@dataclass(frozen=True)
class PatchPlanValidation:
    valid: bool
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": list(self.errors)}


def load_patch_plan(path: str) -> PatchPlan:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load patch plan JSON: {exc}") from exc
    return PatchPlan.from_dict(payload)


def _normalise_repo_path(path: str) -> str | None:
    if not path or "\\" in path:
        return None
    normalized = posixpath.normpath(path)
    if normalized in {".", ".."} or normalized.startswith("../") or normalized.startswith("/"):
        return None
    return normalized


def _diff_paths(unified_diff: str) -> tuple[set[str], list[str]]:
    errors: list[str] = []
    if "GIT binary patch" in unified_diff or "Binary files " in unified_diff:
        errors.append("binary patches are not supported")
    headers = list(_DIFF_HEADER.finditer(unified_diff))
    if len(headers) < 2 or len(headers) % 2:
        errors.append("unified diff must contain paired --- and +++ file headers")
        return set(), errors
    if not _HUNK_HEADER.search(unified_diff):
        errors.append("unified diff must contain at least one @@ hunk header")
    paths: set[str] = set()
    for before, after in zip(headers[::2], headers[1::2]):
        if before.group(1) != "---" or after.group(1) != "+++":
            errors.append("unified diff file headers must be ordered as --- then +++")
            continue
        before_path = _normalise_repo_path(before.group("path"))
        after_path = _normalise_repo_path(after.group("path"))
        if before_path is None or after_path is None:
            errors.append("unified diff contains an unsafe file path")
            continue
        if before_path != after_path:
            errors.append("renames, additions, and deletions are not supported in patch plans")
            continue
        paths.add(before_path)
    return paths, errors


def _is_allowed_test_command(command: str, prefixes: tuple[tuple[str, ...], ...]) -> bool:
    if not command.strip() or _COMMAND_FORBIDDEN.search(command):
        return False
    try:
        tokens = tuple(shlex.split(command))
    except ValueError:
        return False
    return any(tokens[:len(prefix)] == prefix for prefix in prefixes)


def scan_diff_security(unified_diff: str) -> list[str]:
    """Scan added lines (+) in a unified diff for dangerous code patterns or hardcoded secrets."""
    violations: list[str] = []
    for raw_line in unified_diff.splitlines():
        if not raw_line.startswith("+") or raw_line.startswith("+++"):
            continue
        added_code = raw_line[1:]
        for label, pattern in _SECURITY_DIFF_PATTERNS:
            if pattern.search(added_code):
                msg = f"security check failed in diff: {label}"
                if msg not in violations:
                    violations.append(msg)
    return violations


def validate_patch_plan(
    plan: PatchPlan,
    context: RepairContext,
    *,
    allowed_test_prefixes: tuple[tuple[str, ...], ...] = _DEFAULT_TEST_PREFIXES,
) -> PatchPlanValidation:
    errors: list[str] = []
    evidence_by_path = {item.path: item for item in context.evidence}
    allowed_paths = set(evidence_by_path)
    planned_paths: set[str] = set()
    for path in plan.files_to_modify:
        normalized = _normalise_repo_path(path)
        if normalized is None:
            errors.append(f"unsafe planned file path: {path!r}")
        else:
            planned_paths.add(normalized)
    if not planned_paths:
        errors.append("patch plan must modify at least one file")
    if len(planned_paths) != len(plan.files_to_modify):
        errors.append("patch plan file paths must be unique")
    for path in sorted(planned_paths - allowed_paths):
        errors.append(f"planned file is outside repair context: {path}")
    for path in sorted(planned_paths & allowed_paths):
        item = evidence_by_path[path]
        if not item.patch_eligible:
            errors.append(
                f"planned file is not patch-eligible ({item.patch_eligibility_reason}): {path}"
            )

    diff_paths, diff_errors = _diff_paths(plan.unified_diff)
    errors.extend(diff_errors)
    if diff_paths != planned_paths:
        errors.append("files_to_modify must exactly match unified diff file paths")
    errors.extend(scan_diff_security(plan.unified_diff))

    if not 0.0 <= plan.confidence <= 1.0:
        errors.append("confidence must be between 0 and 1")
    has_confirmed_evidence = bool(plan.confirmed_evidence) or any(
        item.evidence_tier == "confirmed" for item in context.evidence
    )
    if plan.confidence > 0.5 and not has_confirmed_evidence:
        errors.append(
            "confident explanation (> 0.5) requires confirmed supporting evidence in repair context"
        )
    if not plan.tests_to_run:
        errors.append("patch plan must include at least one verification command")
    for command in plan.tests_to_run:
        if not _is_allowed_test_command(command, allowed_test_prefixes):
            errors.append(f"verification command is not allowlisted: {command!r}")
    return PatchPlanValidation(not errors, tuple(errors))
