"""Persistent, auditable artifacts for incident repair attempts."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from .incident_mapper import RepairContext
from .patch_executor import (
    PatchApplicationResult,
    PatchExecutionResult,
    build_pull_request_markdown,
)
from .patch_planner import PatchPlan, PatchPlanValidation
from .production_guard import ApprovalPolicyDecision, append_audit_log, redact_secrets


_RUN_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class RepairRun:
    path: str
    status: str
    incident_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "status": self.status, "incident_id": self.incident_id}


@dataclass(frozen=True)
class RepairLesson:
    incident_id: str
    status: str
    exception_type: str
    message: str
    diagnosis: str
    files_modified: tuple[str, ...] = ()
    tests_to_run: tuple[str, ...] = ()
    unified_diff: str = ""
    polarity: str = "positive"  # "positive" (verified playbook) or "negative" (failed/rolled_back/regressed anti-pattern)
    structural_operator: str = ""
    anti_pattern_warning: str = ""
    confidence_score: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "incident_id": self.incident_id,
            "status": self.status,
            "exception_type": self.exception_type,
            "message": self.message,
            "diagnosis": self.diagnosis,
            "files_modified": list(self.files_modified),
            "tests_to_run": list(self.tests_to_run),
            "polarity": self.polarity,
            "structural_operator": self.structural_operator,
            "confidence_score": round(self.confidence_score, 3),
        }
        if self.unified_diff:
            result["unified_diff"] = self.unified_diff
        if self.anti_pattern_warning:
            result["anti_pattern_warning"] = self.anti_pattern_warning
        return result


def _run_name(incident_id: str) -> str:
    normalized = _RUN_NAME.sub("-", incident_id).strip(".-")
    return normalized[:100] or "incident"


def _next_run_path(root: Path, incident_id: str) -> Path:
    base = _run_name(incident_id)
    candidate = root / base
    sequence = 1
    while candidate.exists():
        candidate = root / f"{base}-{sequence}"
        sequence += 1
    return candidate


def _status(
    validation: Optional[PatchPlanValidation],
    execution: Optional[PatchExecutionResult],
    application: Optional[PatchApplicationResult] = None,
) -> str:
    if application is not None and application.status in {
        "applied",
        "pr_created",
        "pr_branch_ready",
        "rolled_back",
    }:
        return application.status
    if execution is not None:
        return execution.status
    if validation is not None and not validation.valid:
        return "rejected"
    return "needs-review"


def _write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(redact_secrets(text), encoding="utf-8")


def save_repair_run(
    root_directory: str,
    context: RepairContext,
    *,
    plan: Optional[PatchPlan] = None,
    validation: Optional[PatchPlanValidation] = None,
    execution: Optional[PatchExecutionResult] = None,
    application: Optional[PatchApplicationResult] = None,
    policy_decision: Optional[ApprovalPolicyDecision] = None,
    dead_end_actions: Sequence[str] = (),
    kb_trust_status: Optional[str] = None,
    promoted_by: str = "",
) -> RepairRun:
    from .knowledge_base import distill_post_incident_record

    root = Path(root_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = _next_run_path(root, context.incident.incident_id)
    temporary_directory = Path(tempfile.mkdtemp(prefix=".asre-run-", dir=root))
    status = _status(validation, execution, application)
    effective_promoted_by = promoted_by or (
        policy_decision.operator
        if policy_decision and policy_decision.allowed_to_apply and not policy_decision.auto_apply_permitted
        else ""
    )
    if kb_trust_status is None:
        if (
            application is not None
            and application.approved
            and application.status in {"applied", "pr_created", "pr_branch_ready"}
        ):
            effective_kb_trust = "verified"
        else:
            effective_kb_trust = "unverified"
    else:
        effective_kb_trust = kb_trust_status

    try:
        _write_json(temporary_directory / "incident.json", context.incident.to_dict())
        _write_json(temporary_directory / "repair-context.json", context.to_dict())
        verification = execution.to_dict() if execution is not None else {
            "status": status,
            "validation": validation.to_dict() if validation is not None else None,
        }
        if application is not None:
            verification["application"] = application.to_dict()
        _write_json(temporary_directory / "verification.json", verification)
        if plan is not None:
            _write_json(temporary_directory / "patch-plan.json", plan.to_dict())
            (temporary_directory / "patch.diff").write_text(plan.unified_diff, encoding="utf-8")
            pr_title, pr_body = build_pull_request_markdown(plan, context, execution)
            (temporary_directory / "pull-request.md").write_text(
                f"# {pr_title}\n\n{pr_body}",
                encoding="utf-8",
            )
        now_iso = datetime.now(timezone.utc).isoformat()
        run_meta: dict[str, Any] = {
            "incident_id": context.incident.incident_id,
            "status": status,
            "kb_trust_status": effective_kb_trust,
            "created_at": now_iso,
        }
        if effective_promoted_by:
            run_meta["promoted_by"] = effective_promoted_by
        if dead_end_actions:
            run_meta["dead_end_actions"] = list(dead_end_actions)
        if execution is not None:
            run_meta["trajectory"] = {
                "captured_at": now_iso,
                "reproduced_before_patch": execution.reproduced_before_patch,
                "incident_replay_passed": execution.incident_replay_passed,
                "executed_commands": [" ".join(cmd.command) for cmd in execution.commands],
            }
        if application is not None:
            run_meta["human_approved"] = application.approved
            run_meta["application_status"] = application.status
            run_meta["rolled_back"] = application.rolled_back
        if policy_decision is not None:
            run_meta["policy_decision"] = policy_decision.to_dict()
        if plan is not None:
            run_meta["lesson"] = {
                "exception_type": context.incident.exception_type,
                "message": context.incident.message,
                "diagnosis": plan.diagnosis,
                "files_modified": list(plan.files_to_modify),
                "tests_to_run": list(plan.tests_to_run),
            }
            distilled = distill_post_incident_record(
                context.incident.incident_id,
                context.incident.exception_type,
                context.incident.message,
                plan.files_to_modify,
                plan.diagnosis,
                plan.unified_diff,
                plan.tests_to_run,
                commit_sha=context.incident.commit_sha or (execution.revision if execution else ""),
                kb_trust_status=effective_kb_trust,
                promoted_by=effective_promoted_by,
                dead_end_actions=dead_end_actions,
                impact_summary=execution.impact_summary if execution else None,
                repo_root=context.repository_path,
            )
            run_meta["distilled_record"] = distilled.to_dict()
            distilled_dir = root / ".distilled"
            try:
                distilled_dir.mkdir(parents=True, exist_ok=True)
                _write_json(distilled_dir / f"{destination.name}.json", distilled.to_dict())
                (distilled_dir / f"{destination.name}.md").write_text(
                    redact_secrets(distilled.reflection_markdown),
                    encoding="utf-8",
                )
            except OSError:
                pass
        _write_json(temporary_directory / "run.json", run_meta)
        os.replace(temporary_directory, destination)
        append_audit_log(
            root,
            incident_id=context.incident.incident_id,
            action="save_repair_run",
            status=status,
            operator=policy_decision.operator if policy_decision else "cli-operator",
            role=policy_decision.role if policy_decision else "sre",
            environment=policy_decision.environment if policy_decision else "development",
            policy_decision=policy_decision,
            details={"run_path": str(destination)},
        )
        # Dual-write mirror to local `.asre/supabase_mirror` (and cloud when explicitly synced)
        try:
            from storage.supabase_store import SupabaseStorageEngine

            db_engine = SupabaseStorageEngine(
                env={},
                queue_dir=root.parent if root.name == "runs" else root / ".db_cache",
            )
            db_engine.sync_repair_run(
                destination.name,
                context,
                status=status,
                kb_trust_status=effective_kb_trust,
                promoted_by=effective_promoted_by,
                plan=plan,
                execution=execution,
                application=application,
                dead_end_actions=dead_end_actions,
                distilled_record=run_meta.get("distilled_record"),
            )
        except Exception:
            pass
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return RepairRun(str(destination), status, context.incident.incident_id)


def load_past_lessons(
    root_directory: str,
    context: RepairContext,
    *,
    limit: int = 3,
    include_negative: bool = True,
) -> list[RepairLesson]:
    """
    Stage 8 (Learn — 3-Tier Episodic & Procedural Memory Retrieval):
    Retrieve historical lessons from past repair runs using Hybrid Graph-RAG
    (BM25 + Subword/AST Cosine + Structural File Overlap RRF) with post-repair
    recurrence decay. Returns both verified positive repair playbooks and
    negative trajectory guardrails (failed, rolled-back, or regressed patches).
    """
    from .knowledge_base import load_tier2_episodic_cases, retrieve_tier2_episodic_cases

    root = Path(root_directory).resolve()
    if not root.is_dir():
        return []

    context_paths = [item.path for item in context.evidence]
    query_text = f"{context.incident.exception_type} {context.incident.message}"
    cases = load_tier2_episodic_cases(str(root), context.repository_path)
    if not cases:
        return []

    pos_cases, neg_cases = retrieve_tier2_episodic_cases(
        query_text,
        context_paths,
        cases,
        top_k_positive=max(1, limit),
        top_k_negative=2 if include_negative else 0,
    )

    lessons: list[RepairLesson] = []
    for c in pos_cases:
        if not c.diagnosis and not c.unified_diff:
            continue
        lessons.append(
            RepairLesson(
                incident_id=c.incident_id,
                status=c.status,
                exception_type=c.exception_type,
                message=c.error_message,
                diagnosis=c.diagnosis,
                files_modified=c.affected_files,
                tests_to_run=c.tests_to_run,
                unified_diff=c.unified_diff,
                polarity="positive",
                structural_operator=c.structural_operator,
                anti_pattern_warning="",
                confidence_score=min(0.98, 0.70 + 0.28 * c.similarity_score),
            )
        )

    if include_negative:
        for c in neg_cases:
            # Only surface negative trajectories that directly overlap a file in context or have high similarity
            if not (set(c.affected_files) & set(context_paths)) and c.similarity_score < 0.35:
                continue
            lessons.append(
                RepairLesson(
                    incident_id=c.incident_id,
                    status=c.status,
                    exception_type=c.exception_type,
                    message=c.error_message,
                    diagnosis=c.diagnosis or c.outcome_summary,
                    files_modified=c.affected_files,
                    tests_to_run=c.tests_to_run,
                    unified_diff=c.unified_diff,
                    polarity="negative",
                    structural_operator=c.structural_operator,
                    anti_pattern_warning=c.anti_pattern_warning,
                    confidence_score=c.similarity_score,
                )
            )

    return lessons[: max(1, limit + (2 if include_negative else 0))]
