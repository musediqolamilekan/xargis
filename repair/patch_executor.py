"""Isolated application and verification of validated patch plans."""

from __future__ import annotations

import os
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .incident_mapper import RepairContext
from .patch_planner import PatchPlan, PatchPlanValidation, validate_patch_plan
from .production_guard import ApprovalPolicyDecision, is_emergency_stop_active


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_dict(self) -> dict:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
        }


@dataclass(frozen=True)
class PatchExecutionResult:
    status: str
    validation: PatchPlanValidation
    revision: str = ""
    commands: tuple[CommandResult, ...] = ()
    worktree_removed: bool = False
    error: str = ""
    reproduced_before_patch: bool = False
    incident_replay_passed: bool = False
    pre_patch_commands: tuple[CommandResult, ...] = ()
    impact_summary: dict[str, Any] = field(default_factory=dict)
    simulation_report: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "validation": self.validation.to_dict(),
            "revision": self.revision,
            "commands": [result.to_dict() for result in self.commands],
            "worktree_removed": self.worktree_removed,
            "error": self.error,
            "reproduced_before_patch": self.reproduced_before_patch,
            "incident_replay_passed": self.incident_replay_passed,
            "pre_patch_commands": [result.to_dict() for result in self.pre_patch_commands],
            "impact_summary": dict(self.impact_summary),
        }
        if self.simulation_report is not None:
            result["simulation_report"] = dict(self.simulation_report)
        return result


@dataclass(frozen=True)
class PatchApplicationResult:
    status: str
    approved: bool
    applied_files: tuple[str, ...] = ()
    commands: tuple[CommandResult, ...] = ()
    error: str = ""
    branch_name: str = ""
    commit_sha: str = ""
    pr_url: str = ""
    pr_title: str = ""
    pr_body: str = ""
    health_check: Optional[CommandResult] = None
    rolled_back: bool = False
    policy_decision: Optional[dict[str, Any]] = None
    watchdog_probes: tuple[CommandResult, ...] = ()
    escalation_alert: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "approved": self.approved,
            "applied_files": list(self.applied_files),
            "commands": [result.to_dict() for result in self.commands],
            "error": self.error,
            "rolled_back": self.rolled_back,
        }
        if self.branch_name:
            result["branch_name"] = self.branch_name
        if self.commit_sha:
            result["commit_sha"] = self.commit_sha
        if self.pr_url:
            result["pr_url"] = self.pr_url
        if self.pr_title:
            result["pr_title"] = self.pr_title
        if self.pr_body:
            result["pr_body"] = self.pr_body
        if self.health_check is not None:
            result["health_check"] = self.health_check.to_dict()
        if self.watchdog_probes:
            result["watchdog_probes"] = [p.to_dict() for p in self.watchdog_probes]
        if self.escalation_alert:
            result["escalation_alert"] = self.escalation_alert
        if self.policy_decision is not None:
            result["policy_decision"] = self.policy_decision
        return result


def _truncate(output: str, limit: int = 20000) -> str:
    if len(output) <= limit:
        return output
    return output[:limit] + "\n[output truncated]"


def _run(
    command: list[str],
    *,
    cwd: Optional[str] = None,
    input_text: Optional[str] = None,
    timeout: int = 120,
    extra_env: Optional[dict[str, str]] = None,
) -> CommandResult:
    resolved_command = list(command)
    if resolved_command and resolved_command[0] in {"python", "python3", "venv/bin/python"}:
        resolved_command[0] = sys.executable
    env = None
    if extra_env:
        env = {**os.environ, **extra_env}
    try:
        completed = subprocess.run(
            resolved_command,
            cwd=cwd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        return CommandResult(
            tuple(command),
            completed.returncode,
            _truncate(completed.stdout),
            _truncate(completed.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        out_str = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err_str = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(
            tuple(command),
            124,
            _truncate(out_str),
            _truncate(err_str),
            True,
        )
    except OSError as exc:
        return CommandResult(tuple(command), 127, "", str(exc))


def execute_patch_plan(
    plan: PatchPlan,
    context: RepairContext,
    *,
    command_timeout: int = 120,
) -> PatchExecutionResult:
    stop_active, stop_reason = is_emergency_stop_active(context.repository_path)
    validation = validate_patch_plan(plan, context)
    if stop_active:
        return PatchExecutionResult("rejected", validation, error=f"emergency stop active: {stop_reason}")
    if not validation.valid:
        return PatchExecutionResult("rejected", validation, error="patch plan validation failed")

    repository = Path(context.repository_path).resolve()
    root_result = _run(["git", "-C", str(repository), "rev-parse", "--show-toplevel"])
    if root_result.returncode != 0:
        return PatchExecutionResult("error", validation, commands=(root_result,), error="repository is not a Git worktree")
    repository = Path(root_result.stdout.strip())
    requested_revision = context.incident.commit_sha or "HEAD"
    revision_result = _run(["git", "-C", str(repository), "rev-parse", "--verify", f"{requested_revision}^{{commit}}"])
    if revision_result.returncode != 0:
        return PatchExecutionResult("error", validation, commands=(root_result, revision_result), error="incident revision is unavailable")
    revision = revision_result.stdout.strip()

    commands: list[CommandResult] = [root_result, revision_result]
    pre_patch_commands: list[CommandResult] = []
    reproduced_before_patch = False
    incident_replay_passed = False
    ev_by_path = {item.path: item for item in context.evidence}
    downstream_consumers: list[str] = []
    for path in plan.files_to_modify:
        ev = ev_by_path.get(path)
        if ev is not None:
            for comp in ev.affected_components:
                if comp not in downstream_consumers and comp not in plan.files_to_modify:
                    downstream_consumers.append(comp)
    for item in context.evidence:
        if item.path not in plan.files_to_modify and "test" not in item.path.lower():
            if item.path not in downstream_consumers:
                downstream_consumers.append(item.path)

    modified_impacts = [
        {
            "path": path,
            "risk_score": ev_by_path[path].risk_score if path in ev_by_path else 0,
            "risk_label": ev_by_path[path].risk_label if path in ev_by_path else "Low",
            "direct_dependents": ev_by_path[path].direct_dependents if path in ev_by_path else 0,
            "transitive_dependents": ev_by_path[path].transitive_dependents if path in ev_by_path else 0,
            "affected_components": list(ev_by_path[path].affected_components) if path in ev_by_path else [],
        }
        for path in plan.files_to_modify
    ]
    impact_summary: dict[str, Any] = {
        "files_modified": len(plan.files_to_modify),
        "max_risk_score": max((item["risk_score"] for item in modified_impacts), default=0),
        "total_direct_dependents": sum(item["direct_dependents"] for item in modified_impacts),
        "total_transitive_dependents": sum(item["transitive_dependents"] for item in modified_impacts),
        "downstream_consumers": downstream_consumers,
        "adjacent_test_files": [],
        "adjacent_regressions": 0,
        "per_file": modified_impacts,
    }
    temporary_root = Path(tempfile.mkdtemp(prefix="asre-patch-"))
    worktree = temporary_root / "worktree"
    worktree_added = False
    worktree_removed = False
    forward_check_ok = False
    t4_start = time.perf_counter()
    status = "error"
    error = ""
    try:
        add_result = _run(["git", "-C", str(repository), "worktree", "add", "--detach", str(worktree), revision])
        commands.append(add_result)
        if add_result.returncode != 0:
            error = "failed to create isolated worktree"
        else:
            worktree_added = True
            src_dir = worktree / "src"
            py_path = f"{src_dir}:{worktree}" if src_dir.is_dir() else str(worktree)
            test_env = {"PYTHONPATH": py_path, "PYTHONDONTWRITEBYTECODE": "1"}

            effective_test_commands = _build_blast_radius_test_commands(plan, context, worktree)
            adjacent_test_files = [
                cmd.split()[-1]
                for cmd in effective_test_commands
                if cmd not in plan.tests_to_run
            ]
            impact_summary["adjacent_test_files"] = adjacent_test_files

            for test_command in effective_test_commands:
                pre_res = _run(
                    shlex.split(test_command),
                    cwd=str(worktree),
                    timeout=command_timeout,
                    extra_env=test_env,
                )
                pre_patch_commands.append(pre_res)
            reproduced_before_patch = any(r.returncode != 0 for r in pre_patch_commands)

            check_result = _run(
                ["git", "apply", "--check", "--whitespace=error", "-"],
                cwd=str(worktree),
                input_text=plan.unified_diff,
            )
            commands.append(check_result)
            forward_check_ok = check_result.returncode == 0
            if not forward_check_ok:
                status = "failed"
                error = "patch does not apply cleanly"
            else:
                apply_result = _run(
                    ["git", "apply", "--whitespace=error", "-"],
                    cwd=str(worktree),
                    input_text=plan.unified_diff,
                )
                commands.append(apply_result)
                if apply_result.returncode != 0:
                    status = "failed"
                    error = "patch application failed"
                else:
                    for test_command in effective_test_commands:
                        test_result = _run(
                            shlex.split(test_command),
                            cwd=str(worktree),
                            timeout=command_timeout,
                            extra_env=test_env,
                        )
                        commands.append(test_result)
                    test_results = commands[5:]
                    primary_count = len(plan.tests_to_run)
                    adjacent_results = test_results[primary_count:]
                    impact_summary["adjacent_regressions"] = sum(
                        1 for r in adjacent_results if r.returncode != 0 or not _did_execute_real_tests([r])
                    )
                    if not test_results:
                        status = "failed"
                        error = "no verification test commands were executed"
                    elif not _did_execute_real_tests(test_results):
                        status = "failed"
                        error = "verification test suite ran 0 tests"
                    elif not all(result.returncode == 0 for result in test_results):
                        status = "failed"
                        error = "one or more verification commands failed"
                    else:
                        status = "verified"
                        error = ""
                        incident_replay_passed = reproduced_before_patch
    finally:
        if worktree_added:
            remove_result = _run(["git", "-C", str(repository), "worktree", "remove", "--force", str(worktree)])
            commands.append(remove_result)
            worktree_removed = remove_result.returncode == 0
        shutil.rmtree(temporary_root, ignore_errors=True)
    t4_elapsed_ms = (time.perf_counter() - t4_start) * 1000.0

    from .repair_simulator import run_progressive_repair_simulation

    sim_report = run_progressive_repair_simulation(
        plan,
        context,
        forward_apply_clean=forward_check_ok,
        pre_patch_reproduced=reproduced_before_patch,
        post_patch_passed=(status == "verified"),
        adjacent_consumers_tested=len(impact_summary.get("adjacent_test_files") or ()),
        adjacent_regressions=int(impact_summary.get("adjacent_regressions") or 0),
        worktree_cleaned=worktree_removed,
        tier4_elapsed_ms=t4_elapsed_ms,
    )
    impact_summary["r_action"] = round(sim_report.action_risk.r_action, 4)
    impact_summary["execution_band"] = sim_report.decision_matrix.execution_band
    impact_summary["counterfactual_summary"] = sim_report.tier2_graph_capacity.counterfactual_summary

    return PatchExecutionResult(
        status,
        validation,
        revision,
        tuple(commands),
        worktree_removed,
        error,
        reproduced_before_patch=reproduced_before_patch,
        incident_replay_passed=incident_replay_passed,
        pre_patch_commands=tuple(pre_patch_commands),
        impact_summary=impact_summary,
        simulation_report=sim_report.to_dict(),
    )


_HEALTH_FORBIDDEN_RE = re.compile(r"[;&|<>`$\r\n]")
_ZERO_TESTS_RE = re.compile(r"\b(?:Ran 0 tests|no tests ran|collected 0 items)\b", re.IGNORECASE)


def _build_blast_radius_test_commands(
    plan: PatchPlan,
    context: RepairContext,
    worktree: Path,
) -> list[str]:
    """
    Combine the plan's primary verification commands with any adjacent unittest
    files mapped in context.evidence that cover the blast radius and are not
    already covered by a suite-wide `discover` command.
    """
    commands = list(plan.tests_to_run)
    if any("discover" in cmd for cmd in commands):
        return commands

    for item in context.evidence:
        rel_path = item.path
        base = os.path.basename(rel_path).lower()
        if not (rel_path.startswith("tests/") or base.startswith("test_")) or not rel_path.endswith(".py"):
            continue
        if any(rel_path in cmd for cmd in commands):
            continue
        candidate_file = worktree / rel_path
        if not candidate_file.is_file():
            continue
        try:
            content = candidate_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "unittest" in content and "def test_" in content:
            commands.append(f"python -m unittest {rel_path}")
    return commands


def _did_execute_real_tests(test_results: list[CommandResult]) -> bool:
    if not test_results:
        return False
    for res in test_results:
        combined = f"{res.stdout}\n{res.stderr}"
        if _ZERO_TESTS_RE.search(combined):
            return False
    return True


def apply_verified_patch(
    plan: PatchPlan,
    context: RepairContext,
    execution: Optional[PatchExecutionResult],
    *,
    policy_decision: Optional[ApprovalPolicyDecision] = None,
    health_check_command: Optional[str] = None,
    watchdog_probes: int = 1,
    command_timeout: int = 120,
) -> PatchApplicationResult:
    """
    Apply a patch to the live repository ONLY after:
      1. Emergency stop is clear
      2. Isolated worktree verification succeeded
      3. Production guard policy (RBAC, environment, risk threshold, sensitive domain) permits it
      4. If `health_check_command` is supplied, runs the Automatic Rollback Watchdog across
         `watchdog_probes` consecutive probes and IMMEDIATELY REVERTS the patch via
         `git apply --reverse` and triggers an on-call human escalation alert if any probe fails.
    """
    policy_dict = policy_decision.to_dict() if policy_decision is not None else None
    stop_active, stop_reason = is_emergency_stop_active(context.repository_path)
    if stop_active:
        return PatchApplicationResult(
            status="rejected",
            approved=False,
            error=f"refusing to apply patch: {stop_reason}",
            policy_decision=policy_dict,
        )
    if execution is None or execution.status != "verified":
        return PatchApplicationResult(
            status="rejected",
            approved=True,
            error="refusing to apply patch: patch has not passed isolated worktree verification",
            policy_decision=policy_dict,
        )
    if policy_decision is not None and not policy_decision.allowed_to_apply:
        return PatchApplicationResult(
            status="rejected",
            approved=False,
            error=f"production guard policy blocked application: {policy_decision.reason}",
            policy_decision=policy_dict,
        )
    validation = validate_patch_plan(plan, context)
    if not validation.valid:
        return PatchApplicationResult(
            status="rejected",
            approved=True,
            error="patch plan validation failed",
            policy_decision=policy_dict,
        )

    repository = Path(context.repository_path).resolve()
    check_result = _run(
        ["git", "apply", "--check", "--whitespace=error", "-"],
        cwd=str(repository),
        input_text=plan.unified_diff,
    )
    if check_result.returncode != 0:
        return PatchApplicationResult(
            status="failed",
            approved=True,
            commands=(check_result,),
            error="patch does not apply cleanly to live repository working tree",
            policy_decision=policy_dict,
        )

    apply_result = _run(
        ["git", "apply", "--whitespace=error", "-"],
        cwd=str(repository),
        input_text=plan.unified_diff,
    )
    if apply_result.returncode != 0:
        return PatchApplicationResult(
            status="failed",
            approved=True,
            commands=(check_result, apply_result),
            error="failed to apply patch to live repository working tree",
            policy_decision=policy_dict,
        )

    commands_run: list[CommandResult] = [check_result, apply_result]
    if health_check_command and health_check_command.strip():
        if _HEALTH_FORBIDDEN_RE.search(health_check_command):
            rollback_res = _run(
                ["git", "apply", "--reverse", "--whitespace=error", "-"],
                cwd=str(repository),
                input_text=plan.unified_diff,
            )
            commands_run.append(rollback_res)
            return PatchApplicationResult(
                status="rolled_back",
                approved=True,
                applied_files=plan.files_to_modify,
                commands=tuple(commands_run),
                error="unsafe health check command; patch automatically rolled back",
                rolled_back=True,
                policy_decision=policy_dict,
            )

        src_dir = repository / "src"
        py_path = f"{src_dir}:{repository}" if src_dir.is_dir() else str(repository)
        probe_count = max(1, watchdog_probes)
        probe_results: list[CommandResult] = []
        last_hc: Optional[CommandResult] = None

        for probe_idx in range(1, probe_count + 1):
            hc_result = _run(
                shlex.split(health_check_command),
                cwd=str(repository),
                timeout=command_timeout,
                extra_env={"PYTHONPATH": py_path, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            commands_run.append(hc_result)
            probe_results.append(hc_result)
            last_hc = hc_result
            if hc_result.returncode != 0:
                rollback_res = _run(
                    ["git", "apply", "--reverse", "--whitespace=error", "-"],
                    cwd=str(repository),
                    input_text=plan.unified_diff,
                )
                commands_run.append(rollback_res)
                escalation = (
                    f"ON-CALL ESCALATION: Post-remediation Rollback Watchdog failed on probe "
                    f"{probe_idx}/{probe_count} (exit {hc_result.returncode}); patch automatically reverted "
                    f"on {', '.join(plan.files_to_modify)} for incident '{context.incident.incident_id}'."
                )
                return PatchApplicationResult(
                    status="rolled_back",
                    approved=True,
                    applied_files=plan.files_to_modify,
                    commands=tuple(commands_run),
                    error="deployment health check failed; patch automatically rolled back",
                    health_check=hc_result,
                    rolled_back=rollback_res.returncode == 0,
                    policy_decision=policy_dict,
                    watchdog_probes=tuple(probe_results),
                    escalation_alert=escalation,
                )
        return PatchApplicationResult(
            status="applied",
            approved=True,
            applied_files=plan.files_to_modify,
            commands=tuple(commands_run),
            health_check=last_hc,
            policy_decision=policy_dict,
            watchdog_probes=tuple(probe_results),
        )

    return PatchApplicationResult(
        status="applied",
        approved=True,
        applied_files=plan.files_to_modify,
        commands=tuple(commands_run),
        policy_decision=policy_dict,
    )


def rollback_applied_patch(
    repo_root: str,
    diff_or_run_path: str,
) -> PatchApplicationResult:
    """Roll back a previously applied patch using its unified diff or `.asre/runs/<id>` directory."""
    repository = Path(repo_root).resolve()
    candidate_path = Path(diff_or_run_path)
    if candidate_path.is_dir():
        diff_file = candidate_path / "patch.diff"
        if not diff_file.is_file():
            return PatchApplicationResult(
                status="failed",
                approved=True,
                error=f"no patch.diff found in run directory {diff_or_run_path}",
            )
        unified_diff = diff_file.read_text(encoding="utf-8")
    elif candidate_path.is_file():
        unified_diff = candidate_path.read_text(encoding="utf-8")
    else:
        unified_diff = diff_or_run_path

    check_rev = _run(
        ["git", "apply", "--reverse", "--check", "--whitespace=error", "-"],
        cwd=str(repository),
        input_text=unified_diff,
    )
    if check_rev.returncode != 0:
        return PatchApplicationResult(
            status="failed",
            approved=True,
            commands=(check_rev,),
            error="reverse patch does not apply cleanly to working tree",
        )

    apply_rev = _run(
        ["git", "apply", "--reverse", "--whitespace=error", "-"],
        cwd=str(repository),
        input_text=unified_diff,
    )
    if apply_rev.returncode != 0:
        return PatchApplicationResult(
            status="failed",
            approved=True,
            commands=(check_rev, apply_rev),
            error="failed to apply reverse patch during rollback",
        )

    return PatchApplicationResult(
        status="rolled_back",
        approved=True,
        commands=(check_rev, apply_rev),
        rolled_back=True,
    )


def explain_untouched_evidence(
    plan: PatchPlan,
    context: RepairContext,
) -> list[str]:
    """
    Return explicit reasons why ASRE deliberately chose NOT to change certain
    inspected files, callers, or contracts.
    """
    reasons: list[str] = list(plan.deliberately_unchanged)
    modified_set = set(plan.files_to_modify)

    for item in context.evidence:
        if item.path in modified_set:
            continue
        lower_path = item.path.lower()
        if "test" in lower_path:
            reason = (
                f"{item.path} — Deliberately left test file unchanged to preserve regression "
                "assertions and avoid weakening the test contract"
            )
        elif not item.patch_eligible:
            reason = (
                f"{item.path} — Deliberately excluded ({item.patch_eligibility_reason}; "
                f"evidence_tier={item.evidence_tier})"
            )
        elif item.evidence_tier == "corroborated":
            reason = (
                f"{item.path} — Deliberately left dependency-graph neighbor unchanged; "
                "fixing the root cause directly avoids masking upstream/downstream callers"
            )
        else:
            reason = (
                f"{item.path} — Inspected in stack trace (score={item.score:.1f}) "
                "but left unchanged in favor of a minimal root-cause fix"
            )
        if reason not in reasons:
            reasons.append(reason)

    if not reasons:
        reasons.append(
            f"Preserved public function signatures and existing behavior outside {', '.join(plan.files_to_modify)}"
        )
    return reasons


def build_pull_request_markdown(
    plan: PatchPlan,
    context: RepairContext,
    execution: Optional[PatchExecutionResult] = None,
) -> tuple[str, str]:
    """Build a structured Pull Request (title, markdown_body) for a verified repair."""
    inc = context.incident
    short_diag = plan.diagnosis.splitlines()[0][:72]
    title = f"fix({inc.incident_id}): {short_diag}"

    untouched = explain_untouched_evidence(plan, context)
    confirmed = list(plan.confirmed_evidence)
    if not confirmed:
        for item in context.evidence:
            if item.evidence_tier == "confirmed" and item.valid_frame_lines:
                confirmed.append(
                    f"`{item.path}` at line(s) {', '.join(str(l) for l in item.valid_frame_lines)}"
                )

    lines: list[str] = [
        f"## Summary — Incident `{inc.incident_id}`",
        f"- **Exception:** `{inc.exception_type or 'Error'}`",
        f"- **Error Message:** {inc.message}",
        f"- **Confidence:** `{plan.confidence * 100:.0f}%`",
        "",
        "## Root-Cause Diagnosis",
        plan.diagnosis,
        "",
    ]
    if confirmed:
        lines.append("## Confirmed Supporting Evidence")
        for fact in confirmed:
            lines.append(f"- {fact}")
        lines.append("")

    lines.append("## Proposed Changes")
    for path in plan.files_to_modify:
        lines.append(f"- `{path}`")
    lines.append("")

    lines.append("## What ASRE Deliberately Chose Not to Change")
    for item_reason in untouched:
        lines.append(f"- {item_reason}")
    lines.append("")

    if plan.alternatives:
        lines.append("## Alternatives Considered")
        for alt in plan.alternatives:
            lines.append(f"- {alt}")
        lines.append("")

    lines.append("## Verification & Impact")
    if execution is not None:
        lines.append(f"- **Verification Status:** `{execution.status}`")
        lines.append(
            f"- **Original Incident Replayed Before Patch:** `{'Yes' if execution.reproduced_before_patch else 'No'}`"
        )
        lines.append(
            f"- **Original Incident Replayed Against Patch (No Longer Reproduces):** `{'Yes' if execution.incident_replay_passed else 'No'}`"
        )
        if execution.impact_summary:
            imp = execution.impact_summary
            lines.append(
                f"- **Blast Radius:** max risk `{imp.get('max_risk_score', 0)}/90`, "
                f"`{imp.get('total_direct_dependents', 0)}` direct / "
                f"`{imp.get('total_transitive_dependents', 0)}` transitive dependents, "
                f"`{imp.get('adjacent_regressions', 0)}` adjacent regressions"
            )
            consumers = imp.get("downstream_consumers") or []
            if consumers:
                lines.append(f"- **Downstream Consumers Verified:** `{', '.join(consumers[:6])}`")
    for test_cmd in plan.tests_to_run:
        lines.append(f"- **Test Command:** `{test_cmd}`")

    return title, "\n".join(lines) + "\n"


def create_repair_pull_request(
    plan: PatchPlan,
    context: RepairContext,
    execution: Optional[PatchExecutionResult],
    *,
    branch_prefix: str = "asre/fix",
) -> PatchApplicationResult:
    """
    Create a dedicated Git branch (`asre/fix-<incident_id>`), apply and commit the
    verified patch with a structured RCA commit body, and optionally create a GitHub PR.
    """
    if execution is None or execution.status != "verified":
        return PatchApplicationResult(
            status="rejected",
            approved=True,
            error="refusing to create PR: patch has not passed isolated worktree verification",
        )
    validation = validate_patch_plan(plan, context)
    if not validation.valid:
        return PatchApplicationResult(
            status="rejected",
            approved=True,
            error="patch plan validation failed",
        )

    repository = Path(context.repository_path).resolve()
    clean_id = re.sub(r"[^A-Za-z0-9._-]+", "-", context.incident.incident_id).strip("-") or "incident"
    branch_name = f"{branch_prefix}-{clean_id}"
    pr_title, pr_body = build_pull_request_markdown(plan, context, execution)

    commands: list[CommandResult] = []
    check_result = _run(
        ["git", "apply", "--check", "--whitespace=error", "-"],
        cwd=str(repository),
        input_text=plan.unified_diff,
    )
    commands.append(check_result)
    if check_result.returncode != 0:
        return PatchApplicationResult(
            status="failed",
            approved=True,
            commands=tuple(commands),
            error="patch does not apply cleanly to repository working tree",
        )

    checkout_res = _run(["git", "checkout", "-B", branch_name], cwd=str(repository))
    commands.append(checkout_res)
    if checkout_res.returncode != 0:
        return PatchApplicationResult(
            status="failed",
            approved=True,
            commands=tuple(commands),
            error=f"failed to create repair branch {branch_name}",
        )

    apply_res = _run(
        ["git", "apply", "--whitespace=error", "-"],
        cwd=str(repository),
        input_text=plan.unified_diff,
    )
    commands.append(apply_res)
    if apply_res.returncode != 0:
        return PatchApplicationResult(
            status="failed",
            approved=True,
            commands=tuple(commands),
            branch_name=branch_name,
            error="failed to apply patch on repair branch",
        )

    add_res = _run(["git", "add", "--", *plan.files_to_modify], cwd=str(repository))
    commands.append(add_res)
    commit_res = _run(
        [
            "git",
            "-c",
            "user.name=ASRE Autonomous Repair",
            "-c",
            "user.email=asre@autonomous.local",
            "commit",
            "-m",
            pr_title,
            "-m",
            pr_body,
        ],
        cwd=str(repository),
    )
    commands.append(commit_res)
    if commit_res.returncode != 0:
        return PatchApplicationResult(
            status="failed",
            approved=True,
            commands=tuple(commands),
            branch_name=branch_name,
            error="failed to commit verified patch on repair branch",
        )

    sha_res = _run(["git", "rev-parse", "HEAD"], cwd=str(repository))
    commands.append(sha_res)
    commit_sha = sha_res.stdout.strip() if sha_res.returncode == 0 else ""

    pr_url = ""
    if shutil.which("gh"):
        gh_res = _run(
            ["gh", "pr", "create", "--title", pr_title, "--body", pr_body],
            cwd=str(repository),
            timeout=30,
        )
        commands.append(gh_res)
        if gh_res.returncode == 0:
            pr_url = gh_res.stdout.strip().splitlines()[-1] if gh_res.stdout.strip() else ""

    return PatchApplicationResult(
        status="pr_created" if pr_url else "pr_branch_ready",
        approved=True,
        applied_files=plan.files_to_modify,
        commands=tuple(commands),
        branch_name=branch_name,
        commit_sha=commit_sha,
        pr_url=pr_url,
        pr_title=pr_title,
        pr_body=pr_body,
    )
