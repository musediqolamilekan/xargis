"""
Enterprise Production Protection & Policy Engine (Section E).

Provides:
  1. Human approval policies & risk-based approval thresholds
     (e.g., auto-fix low-risk config/code in development, require explicit approval
      for database migrations, authentication changes, high-risk files, or production)
  2. Role-Based Access Control (RBAC: viewer, developer, sre, admin)
  3. Environment restrictions (development, staging, production)
  4. Secret protection (deterministic redaction of secrets from prompts, logs & artifacts)
  5. Emergency stop controls (global & per-repo kill switch)
  6. Append-only enterprise audit logging (.asre/runs/audit.jsonl)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .incident_mapper import RepairContext
from .patch_planner import PatchPlan


SUPPORTED_ROLES = ("viewer", "developer", "sre", "admin")
SUPPORTED_ENVIRONMENTS = ("development", "staging", "production")

_ROLE_RANK: dict[str, int] = {
    "viewer": 0,
    "developer": 1,
    "sre": 2,
    "admin": 3,
}

_SECRET_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI / OpenRouter / Groq / GitHub / AWS / Google / Supabase tokens
    re.compile(r"\b(?:sk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|AQ\.Ab8[A-Za-z0-9_-]{16,}|sb_(?:publishable|secret)_[A-Za-z0-9_-]{16,})\b"),
    # Bearer tokens
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{16,}", re.IGNORECASE),
    # Key/value password or secret assignments
    re.compile(r"((?:password|passwd|secret|api_key|apikey|access_token|private_key|supabase_key)\s*[:=]\s*['\"]?)([^'\"\s,;]{8,})(['\"]?)", re.IGNORECASE),
    # Database connection URLs with credentials (postgres://user:pass@host)
    re.compile(r"((?:postgres|postgresql|mysql|mongodb|redis)://[^:\s/]+:)([^@\s]+)(@)"),
)

_AUTH_SECURITY_PATH_RE = re.compile(
    r"(?:^|/|_|-)(?:auth|authentication|login|oauth|jwt|token|session|permission|rbac|acl|security|crypto|password|secret|sso)(?:$|/|_|-|\.)",
    re.IGNORECASE,
)
_DB_MIGRATION_PATH_RE = re.compile(
    r"(?:^|/|_|-)(?:migration|migrations|alembic|flyway|liquibase|schema|prisma)(?:$|/|_|-|\.)|\.sql$",
    re.IGNORECASE,
)
_DB_DDL_DIFF_RE = re.compile(
    r"\b(?:ALTER\s+TABLE|DROP\s+TABLE|CREATE\s+TABLE|TRUNCATE\s+TABLE|op\.(?:add_column|drop_column|create_table|drop_table|alter_column)|migrations\.Migration)\b",
    re.IGNORECASE,
)


def redact_secrets(text: str) -> str:
    """Deterministically redact API keys, Bearer tokens, passwords, and DB credentials."""
    if not text:
        return text
    redacted = text
    redacted = _SECRET_REDACTION_PATTERNS[0].sub("[REDACTED_SECRET]", redacted)
    redacted = _SECRET_REDACTION_PATTERNS[1].sub(r"\1[REDACTED_SECRET]", redacted)
    redacted = _SECRET_REDACTION_PATTERNS[2].sub(r"\1[REDACTED_SECRET]\3", redacted)
    redacted = _SECRET_REDACTION_PATTERNS[3].sub(r"\1[REDACTED_SECRET]\3", redacted)
    return redacted


def _emergency_stop_file(repo_root: str | Path | None = None) -> Path:
    base = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    return base / ".asre" / "EMERGENCY_STOP"


def is_emergency_stop_active(
    repo_root: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Check whether the global or repository emergency stop kill switch is engaged."""
    env_map = env if env is not None else os.environ
    env_flag = (env_map.get("ASRE_EMERGENCY_STOP") or "").strip().lower()
    if env_flag in {"1", "true", "yes", "on", "active"}:
        return True, "ASRE_EMERGENCY_STOP environment kill switch is active"

    for root_candidate in (repo_root, Path.cwd()):
        if root_candidate is None:
            continue
        stop_file = _emergency_stop_file(root_candidate)
        if stop_file.is_file():
            try:
                data = json.loads(stop_file.read_text(encoding="utf-8"))
                reason = str(data.get("reason") or "Emergency stop lockfile present")
                actor = str(data.get("actor") or "operator")
                return True, f"Emergency stop active ({stop_file}): {reason} [by {actor}]"
            except (OSError, json.JSONDecodeError):
                return True, f"Emergency stop lockfile active at {stop_file}"

    return False, ""


def set_emergency_stop(
    repo_root: str | Path,
    active: bool,
    *,
    reason: str = "Manual emergency stop triggered",
    actor: str = "operator",
) -> dict[str, Any]:
    """Engage or clear the repository emergency stop lockfile."""
    stop_file = _emergency_stop_file(repo_root)
    if active:
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active": True,
            "reason": reason,
            "actor": actor,
            "activated_at": datetime.now(timezone.utc).isoformat(),
        }
        stop_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return {"active": True, "path": str(stop_file), **payload}

    if stop_file.is_file():
        stop_file.unlink()
    return {"active": False, "path": str(stop_file), "cleared_by": actor}


def detect_sensitive_domains(
    plan: PatchPlan,
    context: RepairContext,
) -> tuple[str, ...]:
    """
    Identify whether a patch touches high-sensitivity domains that must NEVER
    be auto-applied without explicit approval:
      - "authentication_or_security"
      - "database_migration_or_schema"
    """
    domains: list[str] = []
    ev_by_path = {item.path: item for item in context.evidence}

    for path in plan.files_to_modify:
        if _AUTH_SECURITY_PATH_RE.search(path):
            if "authentication_or_security" not in domains:
                domains.append("authentication_or_security")
        if _DB_MIGRATION_PATH_RE.search(path):
            if "database_migration_or_schema" not in domains:
                domains.append("database_migration_or_schema")

        ev = ev_by_path.get(path)
        if ev is not None:
            for sym in ev.symbols:
                if _AUTH_SECURITY_PATH_RE.search(sym) and "authentication_or_security" not in domains:
                    domains.append("authentication_or_security")
            for db_op in ev.db_interactions:
                if _DB_DDL_DIFF_RE.search(db_op) and "database_migration_or_schema" not in domains:
                    domains.append("database_migration_or_schema")

    if _DB_DDL_DIFF_RE.search(plan.unified_diff):
        if "database_migration_or_schema" not in domains:
            domains.append("database_migration_or_schema")

    return tuple(domains)


@dataclass(frozen=True)
class ApprovalPolicyDecision:
    environment: str
    operator: str
    role: str
    max_risk_score: int
    risk_tier: str
    sensitive_domains: tuple[str, ...]
    auto_apply_permitted: bool
    requires_explicit_approval: bool
    required_minimum_role: str
    role_authorized: bool
    explicit_human_approval: bool
    approver: str
    allowed_to_apply: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "operator": self.operator,
            "role": self.role,
            "max_risk_score": self.max_risk_score,
            "risk_tier": self.risk_tier,
            "sensitive_domains": list(self.sensitive_domains),
            "auto_apply_permitted": self.auto_apply_permitted,
            "requires_explicit_approval": self.requires_explicit_approval,
            "required_minimum_role": self.required_minimum_role,
            "role_authorized": self.role_authorized,
            "explicit_human_approval": self.explicit_human_approval,
            "approver": self.approver,
            "allowed_to_apply": self.allowed_to_apply,
            "reason": self.reason,
        }


def _normalize_environment(env_name: str) -> str:
    lowered = env_name.strip().lower()
    if lowered in {"prod", "production"}:
        return "production"
    if lowered in {"stage", "staging", "preprod"}:
        return "staging"
    return "development"


def evaluate_approval_policy(
    plan: PatchPlan,
    context: RepairContext,
    *,
    environment: str | None = None,
    role: str = "sre",
    operator: str = "cli-operator",
    explicit_human_approval: bool = False,
    approver: str = "",
) -> ApprovalPolicyDecision:
    """
    Evaluate enterprise production protection rules:
      - Auto-apply without explicit human sign-off is ONLY permitted for
        Low-risk (< 20) changes in `development` that do NOT touch
        authentication/security or database migrations/schema.
      - Production environment, High/Critical risk (>= 40), authentication,
        or database migrations require `sre` or `admin` role + explicit human approval.
      - Medium risk (20-39) or `staging` requires `developer`+ role + explicit human approval.
      - `viewer` role is never allowed to apply patches.
    """
    raw_env = environment
    if not raw_env:
        inc_env = context.incident.environment
        if isinstance(inc_env, dict):
            raw_env = str(inc_env.get("env") or inc_env.get("environment") or "development")
        else:
            raw_env = "development"
    norm_env = _normalize_environment(raw_env)

    norm_role = role.strip().lower()
    if norm_role not in _ROLE_RANK:
        raise ValueError(f"unsupported RBAC role {role!r}; choose from {', '.join(SUPPORTED_ROLES)}")

    ev_by_path = {item.path: item for item in context.evidence}
    risk_scores = [ev_by_path[p].risk_score for p in plan.files_to_modify if p in ev_by_path]
    max_risk = max(risk_scores, default=0)
    if max_risk >= 65:
        risk_tier = "Critical"
    elif max_risk >= 40:
        risk_tier = "High"
    elif max_risk >= 20:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    sensitive_domains = detect_sensitive_domains(plan, context)

    # Determine required minimum RBAC role
    if norm_env == "production" or sensitive_domains or max_risk >= 40:
        required_min_role = "sre"
    else:
        required_min_role = "developer"

    role_authorized = _ROLE_RANK[norm_role] >= _ROLE_RANK[required_min_role]

    # Auto-apply threshold: ONLY Low-risk (< 20) in development with zero sensitive domains
    auto_apply_permitted = (
        norm_env == "development"
        and max_risk < 20
        and len(sensitive_domains) == 0
    )
    requires_explicit = not auto_apply_permitted
    has_explicit_approval = bool(explicit_human_approval or approver.strip())
    effective_approver = approver.strip() or (operator if has_explicit_approval else "")

    if not role_authorized:
        allowed = False
        reason = (
            f"RBAC policy denied: role '{norm_role}' is below required minimum role '{required_min_role}' "
            f"(env={norm_env}, risk={risk_tier} {max_risk}/90, sensitive_domains={list(sensitive_domains)})"
        )
    elif requires_explicit and not has_explicit_approval:
        allowed = False
        reasons_list: list[str] = []
        if sensitive_domains:
            reasons_list.append(f"touches protected domain(s) {', '.join(sensitive_domains)}")
        if norm_env != "development":
            reasons_list.append(f"target environment is '{norm_env}'")
        if max_risk >= 20:
            reasons_list.append(f"repair risk score {max_risk}/90 ({risk_tier}) exceeds auto-approve threshold (<20)")
        reason = "Explicit human approval required: " + "; ".join(reasons_list)
    else:
        allowed = True
        if auto_apply_permitted and not has_explicit_approval:
            reason = (
                f"Auto-approval permitted: low-risk ({max_risk}/90) change in 'development' "
                "with no sensitive auth/migration domains"
            )
        else:
            reason = (
                f"Explicitly approved by '{effective_approver}' with authorized role '{norm_role}' "
                f"(env={norm_env}, risk={risk_tier} {max_risk}/90)"
            )

    return ApprovalPolicyDecision(
        environment=norm_env,
        operator=operator,
        role=norm_role,
        max_risk_score=max_risk,
        risk_tier=risk_tier,
        sensitive_domains=sensitive_domains,
        auto_apply_permitted=auto_apply_permitted,
        requires_explicit_approval=requires_explicit,
        required_minimum_role=required_min_role,
        role_authorized=role_authorized,
        explicit_human_approval=has_explicit_approval,
        approver=effective_approver,
        allowed_to_apply=allowed,
        reason=reason,
    )


def append_audit_log(
    root_directory: str | Path,
    *,
    incident_id: str,
    action: str,
    status: str,
    operator: str = "cli-operator",
    role: str = "sre",
    environment: str = "development",
    policy_decision: Optional[ApprovalPolicyDecision] = None,
    details: Optional[dict[str, Any]] = None,
) -> Path:
    """Append an immutable JSONL audit record to <root_directory>/audit.jsonl."""
    root = Path(root_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    audit_file = root / "audit.jsonl"
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "incident_id": incident_id,
        "operator": operator,
        "role": role,
        "environment": environment,
        "action": action,
        "status": status,
        "policy": policy_decision.to_dict() if policy_decision is not None else None,
        "details": details or {},
    }
    with open(audit_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return audit_file
