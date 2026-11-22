"""
3-Tier Cognitive Repair Memory and Knowledge Base Engine for ASRE.

Structures ASRE's institutional knowledge into three distinct memory tiers to prevent
hallucinated or destructive actions and compound repair intelligence over time:

  Tier 1 — Semantic Knowledge Base (Static & Slow-Changing Docs, CODEOWNERS, IaC & Runbooks):
    • Ingests architecture docs, runbooks (README, ARCHITECTURE, RUNBOOK, docs/*.md),
      service ownership (CODEOWNERS), and infrastructure manifests (Dockerfile, compose, Terraform, pyproject).
    • Uses Parent-Document Hierarchical Chunking (child chunks indexed for precision,
      mapped back to full parent sections so safety preconditions in Step 1 are never
      separated from execution commands in Step 3).
    • Retrieves via Hybrid Search (BM25 IDF + Subword/Identifier Dense Vectors via RRF)
      followed by a Token-Interaction Cross-Encoder Reranker.

  Tier 2 — Episodic Incident Memory (Case-Based Reasoning + Graph-RAG + Negative Trajectories):
    • Stores structured 4-stage incident cases:
      (Triggering Alerts + Causal Root Cause + Git Commit) -> (Attempted Actions / Diffs) -> (Outcome & MTTR).
    • Explicitly tracks both Positive Trajectories (verified/applied fixes that survived without
      post-repair recurrence) and Negative Trajectories (failed test runs, rejected plans,
      post-deploy health-check rollbacks, and fixes that later regressed) so ASRE never repeats
      failed hypotheses.

  Tier 3 — Procedural Repair Memory (Executable Skills & Playbooks + Thompson Sampling Bandit):
    • Maintains a registry of parameterized remediation workflows and AST code-repair playbooks
      extracted from verified diffs.
    • Ranks skills and playbooks using a Beta-Bernoulli Multi-Armed Bandit (Thompson Sampling
      & Bayesian posterior confidence alpha / (alpha + beta) weighted by recovery speed).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from .cross_incident_intelligence import (
    DrainTemplateMiner,
    _bm25_scores,
    _build_dense_semantic_vector,
    _cosine_similarity,
    _decompose_identifier,
    _tokenize,
)
from .incident_mapper import RepairContext
from .production_guard import (
    _AUTH_SECURITY_PATH_RE,
    _DB_DDL_DIFF_RE,
    _DB_MIGRATION_PATH_RE,
    append_audit_log,
    is_emergency_stop_active,
    redact_secrets,
)


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")
_PRECONDITION_RE = re.compile(
    r"\b(?:precondition|prerequisite|warning|caution|before|safety|backup|verify|check|must|never|do not)\b",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data Models for the 3-Tier Memory Architecture
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CodeOwnerRule:
    pattern: str
    owners: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"pattern": self.pattern, "owners": list(self.owners)}


@dataclass(frozen=True)
class SemanticParentDocument:
    """Tier 1 Parent Document preserving full section context & safety preconditions."""
    parent_id: str
    source_path: str
    doc_kind: str  # runbook, architecture_doc, codeowners, iac_manifest, package_manifest
    section_title: str
    full_content: str
    has_safety_preconditions: bool
    precondition_summary: str
    child_chunks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "source_path": self.source_path,
            "doc_kind": self.doc_kind,
            "section_title": self.section_title,
            "full_content": self.full_content,
            "has_safety_preconditions": self.has_safety_preconditions,
            "precondition_summary": self.precondition_summary,
            "child_chunk_count": len(self.child_chunks),
        }


@dataclass(frozen=True)
class SemanticRetrievalHit:
    """Tier 1 Reranked Parent-Document Retrieval Result."""
    parent_doc: SemanticParentDocument
    matched_child_excerpt: str
    rrf_score: float
    cross_encoder_score: float
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_doc": self.parent_doc.to_dict(),
            "matched_child_excerpt": self.matched_child_excerpt,
            "rrf_score": round(self.rrf_score, 4),
            "cross_encoder_score": round(self.cross_encoder_score, 4),
            "final_score": round(self.final_score, 4),
        }


@dataclass(frozen=True)
class EpisodicIncidentCase:
    """
    Tier 2 Case-Based Reasoning (CBR) Node:
    (Trigger + Root Cause + Commit) -> (Attempted Actions / Diff) -> (Outcome & Survival).
    """
    case_id: str
    incident_id: str
    trajectory_polarity: str  # "positive" (verified & survived) or "negative" (failed, rolled_back, regressed)
    status: str  # verified, applied, failed, rejected, rolled_back, regressed_after_repair
    exception_type: str
    error_message: str
    drain_template: str
    affected_files: tuple[str, ...]
    commit_sha: str
    diagnosis: str
    unified_diff: str
    structural_operator: str
    tests_to_run: tuple[str, ...]
    outcome_summary: str
    anti_pattern_warning: str
    recovery_time_seconds: float
    post_repair_survived: bool
    similarity_score: float = 0.0
    kb_trust_status: str = "unverified"  # "unverified", "verified", or "archived_drifted"
    promoted_by: str = ""
    drift_status: str = "clean"  # "clean" or "drifted"
    drift_reason: str = ""
    dead_end_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "incident_id": self.incident_id,
            "trajectory_polarity": self.trajectory_polarity,
            "status": self.status,
            "kb_trust_status": self.kb_trust_status,
            "promoted_by": self.promoted_by,
            "drift_status": self.drift_status,
            "drift_reason": self.drift_reason,
            "exception_type": self.exception_type,
            "error_message": self.error_message,
            "drain_template": self.drain_template,
            "affected_files": list(self.affected_files),
            "commit_sha": self.commit_sha,
            "diagnosis": self.diagnosis,
            "unified_diff": self.unified_diff,
            "structural_operator": self.structural_operator,
            "tests_to_run": list(self.tests_to_run),
            "dead_end_actions": list(self.dead_end_actions),
            "outcome_summary": self.outcome_summary,
            "anti_pattern_warning": self.anti_pattern_warning,
            "recovery_time_seconds": round(self.recovery_time_seconds, 2),
            "post_repair_survived": self.post_repair_survived,
            "similarity_score": round(self.similarity_score, 4),
        }


@dataclass(frozen=True)
class ProceduralRepairSkill:
    """
    Tier 3 Executable Skill / Parameterized Playbook ranked via Beta-Bernoulli
    Thompson Sampling & Bayesian Posterior Confidence.
    """
    skill_id: str
    name: str
    skill_kind: str  # "operational_workflow" or "ast_code_playbook"
    structural_operator: str
    description: str
    parameters: tuple[str, ...]
    preconditions: tuple[str, ...]
    applicable_exceptions: tuple[str, ...]
    target_file_patterns: tuple[str, ...]
    sample_diff_or_command: str
    alpha: float  # Beta distribution alpha (1 + successes)
    beta_param: float  # Beta distribution beta (1 + failures + 2*rollbacks + 2*recurrences)
    posterior_confidence: float
    thompson_bandit_score: float
    mean_recovery_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "skill_kind": self.skill_kind,
            "structural_operator": self.structural_operator,
            "description": self.description,
            "parameters": list(self.parameters),
            "preconditions": list(self.preconditions),
            "applicable_exceptions": list(self.applicable_exceptions),
            "target_file_patterns": list(self.target_file_patterns),
            "sample_diff_or_command": self.sample_diff_or_command,
            "alpha": round(self.alpha, 2),
            "beta_param": round(self.beta_param, 2),
            "posterior_confidence": round(self.posterior_confidence, 4),
            "thompson_bandit_score": round(self.thompson_bandit_score, 4),
            "mean_recovery_seconds": round(self.mean_recovery_seconds, 2),
        }


@dataclass(frozen=True)
class DistilledRepairRecord:
    """
    Post-Incident Reflection Record produced by the Closed-Loop Knowledge Distillation Pipeline:
      1. Root Cause Signature
      2. Winning Fix (exact command / diff / PR)
      3. Dead-End Actions to Avoid
      4. Preconditions & Blast Radius
    """
    record_id: str
    incident_id: str
    kb_trust_status: str  # "unverified", "verified", or "archived_drifted"
    promoted_by: str
    root_cause_signature: dict[str, Any]
    winning_fix: dict[str, Any]
    dead_end_actions_to_avoid: tuple[str, ...]
    preconditions_and_blast_radius: dict[str, Any]
    drift_status: str = "clean"
    drift_reason: str = ""
    reflection_markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "incident_id": self.incident_id,
            "kb_trust_status": self.kb_trust_status,
            "promoted_by": self.promoted_by,
            "drift_status": self.drift_status,
            "drift_reason": self.drift_reason,
            "root_cause_signature": dict(self.root_cause_signature),
            "winning_fix": dict(self.winning_fix),
            "dead_end_actions_to_avoid": list(self.dead_end_actions_to_avoid),
            "preconditions_and_blast_radius": dict(self.preconditions_and_blast_radius),
            "reflection_markdown": self.reflection_markdown,
        }


@dataclass(frozen=True)
class PreExecutionGuardrailResult:
    """
    3-Check Safety Guardrail verdict before autonomous execution of a stored repair:
      1. Precondition & Zero-Drift Verification
      2. Risk, Blast-Radius & KB Trust Gate Policy
      3. Automatic Rollback Watchdog Readiness
    """
    autonomous_execution_permitted: bool
    preconditions_verified: bool
    drift_free: bool
    trust_gate_verified: bool
    risk_policy_passed: bool
    watchdog_configured: bool
    requires_human_approval: bool
    max_risk_score: int
    sensitive_domains: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "autonomous_execution_permitted": self.autonomous_execution_permitted,
            "preconditions_verified": self.preconditions_verified,
            "drift_free": self.drift_free,
            "trust_gate_verified": self.trust_gate_verified,
            "risk_policy_passed": self.risk_policy_passed,
            "watchdog_configured": self.watchdog_configured,
            "requires_human_approval": self.requires_human_approval,
            "max_risk_score": self.max_risk_score,
            "sensitive_domains": list(self.sensitive_domains),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class KnowledgeBaseReport:
    query: str
    total_semantic_parent_docs: int
    total_episodic_cases: int
    positive_cases_count: int
    negative_cases_count: int
    total_procedural_skills: int
    codeowners_rules: tuple[CodeOwnerRule, ...]
    resolved_owners: dict[str, tuple[str, ...]]
    tier1_semantic_hits: tuple[SemanticRetrievalHit, ...]
    tier2_positive_cases: tuple[EpisodicIncidentCase, ...]
    tier2_negative_cases: tuple[EpisodicIncidentCase, ...]
    tier3_procedural_skills: tuple[ProceduralRepairSkill, ...]
    persisted_index_path: str = ""
    verified_trust_count: int = 0
    unverified_trust_count: int = 0
    drifted_cases_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "summary": {
                "total_semantic_parent_docs": self.total_semantic_parent_docs,
                "total_episodic_cases": self.total_episodic_cases,
                "positive_cases_count": self.positive_cases_count,
                "negative_cases_count": self.negative_cases_count,
                "verified_trust_count": self.verified_trust_count,
                "unverified_trust_count": self.unverified_trust_count,
                "drifted_cases_count": self.drifted_cases_count,
                "total_procedural_skills": self.total_procedural_skills,
                "persisted_index_path": self.persisted_index_path,
            },
            "codeowners_rules": [r.to_dict() for r in self.codeowners_rules],
            "resolved_owners": {k: list(v) for k, v in self.resolved_owners.items()},
            "tier1_semantic_hits": [h.to_dict() for h in self.tier1_semantic_hits],
            "tier2_positive_cases": [c.to_dict() for c in self.tier2_positive_cases],
            "tier2_negative_cases": [c.to_dict() for c in self.tier2_negative_cases],
            "tier3_procedural_skills": [s.to_dict() for s in self.tier3_procedural_skills],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1 — Semantic Knowledge Base: Parent-Document Chunking + Cross-Encoder Reranker
# ──────────────────────────────────────────────────────────────────────────────

def parse_codeowners(repo_root: str) -> tuple[CodeOwnerRule, ...]:
    """Parse CODEOWNERS rules from root, .github/CODEOWNERS, or docs/CODEOWNERS."""
    root = Path(repo_root)
    candidates = (
        root / "CODEOWNERS",
        root / ".github" / "CODEOWNERS",
        root / "docs" / "CODEOWNERS",
        root / "OWNERS",
    )
    rules: list[CodeOwnerRule] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                rules.append(CodeOwnerRule(pattern=parts[0], owners=tuple(parts[1:])))
        break
    return tuple(rules)


def resolve_owners_for_files(
    file_paths: Iterable[str],
    rules: Sequence[CodeOwnerRule],
) -> dict[str, tuple[str, ...]]:
    """Resolve matching CODEOWNERS for each file path."""
    resolved: dict[str, tuple[str, ...]] = {}
    if not rules:
        return resolved
    for f_path in file_paths:
        matched_owners: tuple[str, ...] = ()
        for rule in rules:
            pat = rule.pattern.lstrip("/")
            if pat == "*" or f_path.startswith(pat.rstrip("*")) or f_path.endswith(pat.lstrip("*")):
                matched_owners = rule.owners
        if matched_owners:
            resolved[f_path] = matched_owners
    return resolved


def _extract_precondition_summary(text: str) -> tuple[bool, str]:
    pre_lines: list[str] = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if stripped and _PRECONDITION_RE.search(stripped):
            pre_lines.append(stripped.lstrip("#*-0123456789. "))
    if not pre_lines:
        return False, ""
    return True, "; ".join(pre_lines[:2])[:220]


def _chunk_markdown_into_parent_docs(
    rel_path: str,
    doc_kind: str,
    content: str,
) -> list[SemanticParentDocument]:
    """
    Parent-Document Hierarchical Chunker:
      - Groups top-level sections (`#` / `##`) into full Parent Documents so safety
        preconditions (Step 1) and remediation commands (Step 3) stay together.
      - Splits each Parent Document into finer Child Chunks (`###` subsections or
        paragraph blocks) for high-precision dense + sparse retrieval.
    """
    lines = content.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = Path(rel_path).name
    current_lines: list[str] = []

    for line in lines:
        m = _HEADING_RE.match(line.strip())
        if m and len(m.group(1)) <= 2:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = m.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    parent_docs: list[SemanticParentDocument] = []
    for idx, (sec_title, sec_lines) in enumerate(sections, start=1):
        full_text = "\n".join(sec_lines).strip()
        if len(full_text) < 20:
            continue
        full_text_bounded = full_text[:3000]
        # Build child chunks by splitting on `###` or double newlines
        raw_children = re.split(r"\n(?=#{3,4}\s)|\n\s*\n", full_text_bounded)
        child_chunks = tuple(c.strip() for c in raw_children if len(c.strip()) >= 15)[:12]
        if not child_chunks:
            child_chunks = (full_text_bounded,)

        has_pre, pre_sum = _extract_precondition_summary(full_text_bounded)
        pid = f"doc:{rel_path}#sec-{idx}"
        parent_docs.append(
            SemanticParentDocument(
                parent_id=pid,
                source_path=rel_path,
                doc_kind=doc_kind,
                section_title=sec_title,
                full_content=full_text_bounded,
                has_safety_preconditions=has_pre,
                precondition_summary=pre_sum,
                child_chunks=child_chunks,
            )
        )
    return parent_docs


def ingest_tier1_semantic_documents(repo_root: str) -> list[SemanticParentDocument]:
    """Scan repository for runbooks, architecture docs, CODEOWNERS, and IaC/package manifests."""
    root = Path(repo_root)
    if not root.is_dir():
        return []

    candidate_files: list[tuple[str, str]] = []
    # 1. Top-level documentation, runbooks, CODEOWNERS, and manifests
    for entry in sorted(root.iterdir()):
        if entry.name.startswith(".") and entry.name not in {".github"}:
            continue
        if entry.is_file():
            lower_name = entry.name.lower()
            if any(k in lower_name for k in ("runbook", "playbook", "postmortem", "incident")):
                candidate_files.append((entry.name, "runbook"))
            elif lower_name.startswith(("readme", "architecture", "contributing", "security")):
                candidate_files.append((entry.name, "architecture_doc"))
            elif lower_name in {"codeowners", "owners"}:
                candidate_files.append((entry.name, "codeowners"))
            elif lower_name.startswith(("dockerfile", "docker-compose")) or lower_name.endswith(".tf"):
                candidate_files.append((entry.name, "iac_manifest"))
            elif lower_name in {"pyproject.toml", "package.json", "requirements.txt"}:
                candidate_files.append((entry.name, "package_manifest"))

    # 2. Subdirectories: docs/, runbooks/, .asre/runbooks/, .github/
    for sub_name, default_kind in (
        ("runbooks", "runbook"),
        (".asre/runbooks", "runbook"),
        ("docs", "architecture_doc"),
        (".github", "codeowners"),
    ):
        sub_dir = root / sub_name
        if not sub_dir.is_dir():
            continue
        for child in sorted(sub_dir.rglob("*")):
            if not child.is_file() or len(candidate_files) >= 35:
                continue
            rel = str(child.relative_to(root)).replace("\\", "/")
            lname = child.name.lower()
            if lname in {"codeowners", "owners"}:
                candidate_files.append((rel, "codeowners"))
            elif lname.endswith((".md", ".rst", ".txt")):
                kind = "runbook" if any(k in rel.lower() for k in ("runbook", "playbook", "incident")) else default_kind
                candidate_files.append((rel, kind))
            elif lname.endswith((".tf", ".yaml", ".yml")) and "workflow" not in rel.lower():
                candidate_files.append((rel, "iac_manifest"))

    parent_docs: list[SemanticParentDocument] = []
    seen_paths: set[str] = set()
    for rel_path, doc_kind in candidate_files:
        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)
        try:
            raw_text = (root / rel_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not raw_text.strip():
            continue
        parent_docs.extend(_chunk_markdown_into_parent_docs(rel_path, doc_kind, raw_text))

    return parent_docs


def _cross_encoder_interaction_rerank(
    query: str,
    parent_doc: SemanticParentDocument,
    matched_child: str,
) -> float:
    """
    Stage 2 Pairwise Cross-Encoder Reranker:
    Scores fine-grained token interaction between `query` and `(parent_doc, matched_child)`:
      1. Exact query term coverage in child chunk and parent section title
      2. Ordered query bigram / phrase adjacency inside the document
      3. Safety-precondition preservation bonus for actionable runbooks
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.5
    q_set = set(q_tokens)
    child_tokens = _tokenize(matched_child)
    parent_tokens = _tokenize(f"{parent_doc.section_title} {parent_doc.full_content}")
    title_tokens = set(_tokenize(f"{parent_doc.source_path} {parent_doc.section_title}"))

    child_set = set(child_tokens)
    parent_set = set(parent_tokens)

    # 1. Coverage score
    child_cov = len(q_set & child_set) / max(len(q_set), 1)
    parent_cov = len(q_set & parent_set) / max(len(q_set), 1)
    title_cov = len(q_set & title_tokens) / max(len(q_set), 1)

    # 2. Ordered bigram interaction score
    q_bigrams = {(q_tokens[i], q_tokens[i + 1]) for i in range(len(q_tokens) - 1)}
    doc_bigrams = {(parent_tokens[i], parent_tokens[i + 1]) for i in range(len(parent_tokens) - 1)}
    bigram_overlap = len(q_bigrams & doc_bigrams) / max(len(q_bigrams), 1) if q_bigrams else 0.0

    # 3. Runbook / precondition priority bonus
    kind_bonus = 0.15 if parent_doc.doc_kind in {"runbook", "codeowners"} else 0.05
    pre_bonus = 0.10 if parent_doc.has_safety_preconditions else 0.0

    score = (
        0.35 * child_cov
        + 0.25 * parent_cov
        + 0.15 * title_cov
        + 0.15 * bigram_overlap
        + kind_bonus
        + pre_bonus
    )
    return min(1.0, score)


def retrieve_tier1_semantic_knowledge(
    query: str,
    parent_docs: Sequence[SemanticParentDocument],
    *,
    top_k: int = 4,
    rrf_k: int = 60,
) -> tuple[SemanticRetrievalHit, ...]:
    """
    Retrieve Tier 1 Semantic Knowledge using:
      1. Child-Chunk Indexing with Hybrid BM25 + Subword Dense Cosine via RRF
      2. Parent-Document Context Expansion (preventing missing safety preconditions)
      3. Cross-Encoder Pairwise Interaction Reranking
    """
    if not parent_docs:
        return ()

    # Flatten child chunks with pointers to their parent documents
    child_entries: list[tuple[SemanticParentDocument, str]] = []
    for pdoc in parent_docs:
        for chunk in pdoc.child_chunks:
            child_entries.append((pdoc, chunk))
    if not child_entries:
        return ()

    q_tokens = _tokenize(query)
    q_vec = _build_dense_semantic_vector(query, (), ())

    doc_token_lists = [
        _tokenize(f"{pdoc.source_path} {pdoc.section_title} {chunk}")
        for pdoc, chunk in child_entries
    ]
    bm25_raw = _bm25_scores(q_tokens, doc_token_lists)
    sem_raw = [
        _cosine_similarity(
            q_vec,
            _build_dense_semantic_vector(f"{pdoc.section_title} {chunk}", (pdoc.source_path,), ()),
        )
        for pdoc, chunk in child_entries
    ]

    n_chunks = len(child_entries)
    bm25_order = sorted(range(n_chunks), key=lambda i: (-bm25_raw[i], i))
    sem_order = sorted(range(n_chunks), key=lambda i: (-sem_raw[i], i))
    bm25_ranks = {idx: r for r, idx in enumerate(bm25_order, start=1)}
    sem_ranks = {idx: r for r, idx in enumerate(sem_order, start=1)}

    # Deduplicate to best child chunk per Parent Document
    best_by_parent: dict[str, tuple[SemanticParentDocument, str, float]] = {}
    for idx, (pdoc, chunk) in enumerate(child_entries):
        rrf = 1.0 / (rrf_k + bm25_ranks[idx]) + 1.0 / (rrf_k + sem_ranks[idx])
        prev = best_by_parent.get(pdoc.parent_id)
        if prev is None or rrf > prev[2]:
            best_by_parent[pdoc.parent_id] = (pdoc, chunk, rrf)

    hits: list[SemanticRetrievalHit] = []
    max_rrf = 2.0 / (rrf_k + 1.0)
    for pdoc, best_chunk, rrf_val in best_by_parent.values():
        ce_score = _cross_encoder_interaction_rerank(query, pdoc, best_chunk)
        norm_rrf = rrf_val / max_rrf
        final_score = 0.45 * norm_rrf + 0.55 * ce_score
        hits.append(
            SemanticRetrievalHit(
                parent_doc=pdoc,
                matched_child_excerpt=best_chunk[:240],
                rrf_score=rrf_val,
                cross_encoder_score=ce_score,
                final_score=final_score,
            )
        )

    hits.sort(key=lambda h: (-h.final_score, -h.cross_encoder_score, h.parent_doc.parent_id))
    return tuple(hits[:top_k])


# ──────────────────────────────────────────────────────────────────────────────
# Tier 2 — Episodic Incident Memory: Case-Based Reasoning (Positive & Negative)
# ──────────────────────────────────────────────────────────────────────────────

def classify_structural_fix_operator(unified_diff: str) -> str:
    """
    Domain-neutral AST/diff structural operator classifier that categorizes a unified
    diff by the syntactic transformation it performs on `-` and `+` lines.
    """
    if not unified_diff or not unified_diff.strip():
        return "unclassified_change"
    removed: list[str] = []
    added: list[str] = []
    for ln in unified_diff.splitlines():
        if ln.startswith("-") and not ln.startswith("---"):
            removed.append(ln[1:].strip())
        elif ln.startswith("+") and not ln.startswith("+++"):
            added.append(ln[1:].strip())

    rem_text = "\n".join(removed)
    add_text = "\n".join(added)

    if ("try:" in add_text or "except " in add_text) and "except " not in rem_text:
        return "exception_handling_guard"
    if re.search(r"\bif\s+.*(?:\bis\s+None\b|\bis\s+not\s+None\b|\bnot\s+)", add_text) and len(added) > len(removed):
        return "precondition_or_null_guard"
    if re.search(r"\[[^\]]*:[^\]]*\]|\brange\s*\(|\blen\s*\(|\bmin\s*\(|\bmax\s*\(", add_text):
        return "collection_boundary_or_slice_fix"
    if re.search(r"\b(?:timeout|retry|max_retries|pool_size|limit)\s*=", add_text):
        return "timeout_or_capacity_parameter_tuning"
    if re.search(r"\breturn\b", add_text) and re.search(r"\breturn\b", rem_text):
        return "return_expression_or_contract_fix"
    return "statement_logic_refinement"


def detect_repair_record_drift(
    repo_root: str,
    affected_files: Sequence[str],
    unified_diff: str,
) -> tuple[str, str]:
    """
    Decay & Drift Pruning:
    Check whether a stored repair record has drifted against the current repository
    source tree (e.g., target file deleted/moved, or neither the pre-patch `-` lines
    nor post-patch `+` lines from the unified diff exist in the file anymore).
    """
    if not repo_root or not affected_files:
        return "clean", ""
    root = Path(repo_root)
    if not root.is_dir():
        return "clean", ""

    for rel_path in affected_files:
        full_p = root / rel_path
        if not full_p.is_file():
            return "drifted", f"Target file '{rel_path}' no longer exists in repository"

    if not unified_diff or not unified_diff.strip():
        return "clean", ""

    # Extract meaningful code lines (>3 chars) from the diff for the primary file
    snippet_candidates: list[str] = []
    for raw_ln in unified_diff.splitlines():
        if raw_ln.startswith(("+++", "---", "@@")):
            continue
        if raw_ln.startswith(("+", "-")) and len(raw_ln[1:].strip()) >= 6:
            snippet_candidates.append(raw_ln[1:].strip())

    if snippet_candidates:
        primary_file = root / affected_files[0]
        try:
            current_source = primary_file.read_text(encoding="utf-8", errors="ignore")
            if not any(s in current_source for s in snippet_candidates):
                return (
                    "drifted",
                    f"Architectural/code drift in '{affected_files[0]}': stored diff anchor lines no longer match current source",
                )
        except OSError:
            pass

    return "clean", ""


def distill_post_incident_record(
    incident_id: str,
    exception_type: str,
    message: str,
    affected_files: Sequence[str],
    diagnosis: str,
    unified_diff: str,
    tests_to_run: Sequence[str],
    *,
    commit_sha: str = "",
    mttr_seconds: float = 2.0,
    kb_trust_status: str = "unverified",
    promoted_by: str = "",
    dead_end_actions: Sequence[str] = (),
    impact_summary: Optional[dict[str, Any]] = None,
    repo_root: str = "",
    llm_reflection_agent: Optional[Callable[[str], str]] = None,
) -> DistilledRepairRecord:
    """
    Post-Incident Reflection Distiller (Deterministic + Optional LLM Reflection Critic):
    Distills an incident repair trajectory into a structured Repair Record with:
      1. Root Cause Signature
      2. Winning Fix (exact command / diff)
      3. Dead-End Actions to Avoid
      4. Preconditions & Blast Radius
    """
    miner = DrainTemplateMiner()
    drain_tpl = miner.extract_template(f"{exception_type}: {message}" if exception_type else message).template
    operator = classify_structural_fix_operator(unified_diff)
    drift_status, drift_reason = detect_repair_record_drift(repo_root, affected_files, unified_diff)
    effective_trust = "archived_drifted" if drift_status == "drifted" else kb_trust_status

    imp = impact_summary or {}
    max_risk = int(imp.get("max_risk_score") or 0)
    direct_deps = int(imp.get("total_direct_dependents") or 0)
    trans_deps = int(imp.get("total_transitive_dependents") or 0)

    root_sig = {
        "exception_type": exception_type or "Error",
        "drain_template": drain_tpl,
        "target_files": list(affected_files),
        "structural_operator": operator,
        "commit_sha": commit_sha,
    }
    winning_fix = {
        "diagnosis": diagnosis,
        "structural_operator": operator,
        "unified_diff": unified_diff,
        "tests_to_run": list(tests_to_run),
        "mttr_seconds": round(mttr_seconds, 2),
    }
    preconds_and_blast = {
        "max_risk_score": max_risk,
        "direct_dependents": direct_deps,
        "transitive_dependents": trans_deps,
        "downstream_consumers": list(imp.get("downstream_consumers") or []),
        "required_preconditions": [
            "Emergency stop switch must be clear",
            f"Target files ({', '.join(affected_files) or 'repo'}) must have zero architectural drift",
            "Pre-patch failure reproduction and post-patch test suite must execute >0 tests",
        ],
    }
    dead_ends = tuple(dict.fromkeys(d for d in dead_end_actions if d.strip()))

    md_lines = [
        f"# Post-Incident Distilled Repair Record — `{incident_id}`",
        f"- **KB Trust Status:** `{effective_trust}`" + (f" (promoted by `{promoted_by}`)" if promoted_by else ""),
        f"- **Architectural Drift Status:** `{drift_status}`" + (f" ({drift_reason})" if drift_reason else ""),
        "",
        "## 1. Root Cause Signature",
        f"- **Exception:** `{root_sig['exception_type']}`",
        f"- **Drain Log Template:** `{drain_tpl}`",
        f"- **Target File(s):** `{', '.join(affected_files) or 'N/A'}`",
        f"- **Structural Fix Operator:** `{operator}`",
        "",
        "## 2. Winning Fix",
        f"- **Diagnosis:** {diagnosis}",
        f"- **Verification Suite:** `{', '.join(tests_to_run) or 'N/A'}`",
        f"- **Recovery Time (MTTR):** `{mttr_seconds:.1f}s`",
        "```diff",
        unified_diff.strip(),
        "```",
        "",
        "## 3. Dead-End Actions to Avoid",
    ]
    if dead_ends:
        for de in dead_ends:
            md_lines.append(f"- ⚠️ {de}")
    else:
        md_lines.append("- None recorded on this run (resolved on first verified trajectory).")

    md_lines.extend([
        "",
        "## 4. Preconditions & Blast Radius",
        f"- **Blast Radius:** max risk `{max_risk}/90`, `{direct_deps}` direct / `{trans_deps}` transitive dependents",
    ])
    for pre in preconds_and_blast["required_preconditions"]:
        md_lines.append(f"- **Precondition:** {pre}")

    reflection_md = "\n".join(md_lines) + "\n"
    if llm_reflection_agent is not None:
        try:
            llm_note = llm_reflection_agent(reflection_md).strip()
            if llm_note:
                reflection_md += f"\n## 5. LLM Reflection Critic Summary\n{llm_note}\n"
        except Exception:
            pass

    return DistilledRepairRecord(
        record_id=f"rec:{incident_id}",
        incident_id=incident_id,
        kb_trust_status=effective_trust,
        promoted_by=promoted_by,
        root_cause_signature=root_sig,
        winning_fix=winning_fix,
        dead_end_actions_to_avoid=dead_ends,
        preconditions_and_blast_radius=preconds_and_blast,
        drift_status=drift_status,
        drift_reason=drift_reason,
        reflection_markdown=reflection_md,
    )


def load_tier2_episodic_cases(
    repair_run_dir: Optional[str],
    repo_root: str,
) -> list[EpisodicIncidentCase]:
    """
    Load all historical incident cases from `.asre/runs/*` into Case-Based Reasoning
    nodes, detecting post-repair recurrences and labeling both Positive Trajectories
    and Negative Trajectories (failed tests, rejected plans, rollbacks, or regressions).
    """
    candidate_roots: list[Path] = []
    if repair_run_dir:
        p = Path(repair_run_dir).resolve()
        if p.is_dir():
            candidate_roots.append(p)
    default_root = Path(repo_root).resolve() / ".asre" / "runs"
    if default_root.is_dir() and default_root not in candidate_roots:
        candidate_roots.append(default_root)

    miner = DrainTemplateMiner()
    raw_entries: list[dict[str, Any]] = []

    for root in candidate_roots:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            run_file = entry / "run.json"
            inc_file = entry / "incident.json"
            plan_file = entry / "patch-plan.json"
            diff_file = entry / "patch.diff"
            ver_file = entry / "verification.json"
            ctx_file = entry / "repair-context.json"
            if not run_file.is_file():
                continue
            try:
                run_data = json.loads(run_file.read_text(encoding="utf-8"))
                inc_data = json.loads(inc_file.read_text(encoding="utf-8")) if inc_file.is_file() else {}
                plan_data = json.loads(plan_file.read_text(encoding="utf-8")) if plan_file.is_file() else {}
                ver_data = json.loads(ver_file.read_text(encoding="utf-8")) if ver_file.is_file() else {}
                ctx_data = json.loads(ctx_file.read_text(encoding="utf-8")) if ctx_file.is_file() else {}
                diff_text = diff_file.read_text(encoding="utf-8") if diff_file.is_file() else str(plan_data.get("unified_diff") or "")
            except (OSError, json.JSONDecodeError):
                continue

            lesson = run_data.get("lesson") or {}
            inc_id = str(run_data.get("incident_id") or inc_data.get("incident_id") or entry.name)
            status = str(run_data.get("status") or ver_data.get("status") or "needs-review")
            rolled_back = bool(run_data.get("rolled_back") or (ver_data.get("application") or {}).get("rolled_back"))
            if rolled_back:
                status = "rolled_back"

            exc_type = str(inc_data.get("exception_type") or lesson.get("exception_type") or "Error").strip()
            message = str(inc_data.get("message") or lesson.get("message") or "").strip()
            diagnosis = str(plan_data.get("diagnosis") or lesson.get("diagnosis") or "").strip()
            files_mod = list(plan_data.get("files_to_modify") or lesson.get("files_modified") or [])
            if not files_mod and isinstance(ctx_data.get("evidence"), list):
                for ev in ctx_data["evidence"]:
                    if isinstance(ev, dict) and ev.get("patch_eligible") and ev.get("path"):
                        files_mod.append(str(ev["path"]))
                        break

            tests_to_run = tuple(str(t) for t in (plan_data.get("tests_to_run") or lesson.get("tests_to_run") or ()))
            ver_error = str(ver_data.get("error") or (ver_data.get("application") or {}).get("error") or "")
            val_errors = ((ver_data.get("validation") or {}).get("errors")) or []
            if not ver_error and val_errors:
                ver_error = "; ".join(str(e) for e in val_errors)

            raw_entries.append({
                "folder": entry.name,
                "incident_id": inc_id,
                "status": status,
                "created_at": str(run_data.get("created_at") or ""),
                "exception_type": exc_type,
                "message": message,
                "diagnosis": diagnosis,
                "files": tuple(files_mod),
                "commit_sha": str(inc_data.get("commit_sha") or ver_data.get("revision") or ""),
                "unified_diff": diff_text,
                "tests_to_run": tests_to_run,
                "error": ver_error,
                "mttr_seconds": float(run_data.get("mttr_seconds") or 2.0),
                "kb_trust_status": str(
                    run_data.get("kb_trust_status")
                    or ("verified" if run_data.get("human_approved") and status in {"verified", "applied", "pr_created", "pr_branch_ready"} else "unverified")
                ),
                "promoted_by": str(run_data.get("promoted_by") or ""),
                "dead_end_actions": tuple(str(d) for d in (run_data.get("dead_end_actions") or ())),
            })

    # Hydrate any cloud-synced `asre_repair_runs` records from `.asre/supabase_mirror` not already on local disk
    seen_folders = {r["folder"] for r in raw_entries} | {r["incident_id"] for r in raw_entries}
    for root in candidate_roots:
        if str(root).startswith("/tmp"):
            continue
        mirror_runs = root.parent / "supabase_mirror" / "asre_repair_runs.jsonl"
        if not mirror_runs.is_file():
            continue
        try:
            for ln in mirror_runs.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                row = json.loads(ln)
                inc_id = str(row.get("incident_id") or "")
                run_id_tail = str(row.get("run_id") or "").split(":", 1)[-1]
                if not inc_id or inc_id in seen_folders or run_id_tail in seen_folders:
                    continue
                seen_folders.add(inc_id)
                dist = row.get("distilled_record_json") if isinstance(row.get("distilled_record_json"), dict) else {}
                root_sig = dist.get("root_cause_signature") if isinstance(dist.get("root_cause_signature"), dict) else {}
                raw_entries.append({
                    "folder": run_id_tail or inc_id,
                    "incident_id": inc_id,
                    "status": str(row.get("status") or "needs-review"),
                    "created_at": str(row.get("created_at") or ""),
                    "exception_type": str(root_sig.get("exception_type") or "Error"),
                    "message": str(row.get("suspected_cause") or row.get("diagnosis") or ""),
                    "diagnosis": str(row.get("diagnosis") or ""),
                    "files": tuple(str(f) for f in (row.get("files_modified_json") or ())),
                    "commit_sha": str(root_sig.get("commit_sha") or ""),
                    "unified_diff": str(row.get("unified_diff") or ""),
                    "tests_to_run": tuple(str(t) for t in (row.get("tests_to_run_json") or ())),
                    "error": "",
                    "mttr_seconds": float(row.get("mttr_seconds") or 2.0),
                    "kb_trust_status": str(row.get("kb_trust_status") or "unverified"),
                    "promoted_by": str(row.get("promoted_by") or ""),
                    "dead_end_actions": tuple(str(d) for d in (row.get("dead_end_actions_json") or ())),
                })
        except (OSError, json.JSONDecodeError):
            pass

    raw_entries.sort(key=lambda r: (r["created_at"], r["folder"]))

    # Detect whether a verified/applied repair was followed by a recurrence of the same exception on the same file
    regressed_folders: set[str] = set()
    prior_verified_by_file_exc: dict[tuple[str, str], str] = {}
    for item in raw_entries:
        exc_key = item["exception_type"].lower()
        for f_path in item["files"]:
            pair = (f_path, exc_key)
            if pair in prior_verified_by_file_exc:
                regressed_folders.add(prior_verified_by_file_exc[pair])
            if item["status"] in {"verified", "applied"}:
                prior_verified_by_file_exc[pair] = item["folder"]

    cases: list[EpisodicIncidentCase] = []
    for item in raw_entries:
        folder = item["folder"]
        status = item["status"]
        survived = folder not in regressed_folders
        drift_status, drift_reason = detect_repair_record_drift(
            repo_root,
            item["files"],
            item["unified_diff"],
        )
        kb_trust = item["kb_trust_status"]
        if drift_status == "drifted":
            kb_trust = "archived_drifted"

        if status in {"verified", "applied"} and not survived:
            effective_status = "regressed_after_repair"
            polarity = "negative"
            kb_trust = "unverified"
        elif status in {"verified", "applied"}:
            effective_status = status
            polarity = "positive"
        elif status in {"failed", "rejected", "rolled_back"}:
            effective_status = status
            polarity = "negative"
            kb_trust = "unverified"
        else:
            # Skip unverified "needs-review" drafts without execution verdict unless they have validation errors
            if not item["error"]:
                continue
            effective_status = "rejected"
            polarity = "negative"
            kb_trust = "unverified"

        operator = classify_structural_fix_operator(item["unified_diff"])
        tpl = miner.extract_template(f"{item['exception_type']}: {item['message']}").template
        files_str = ", ".join(item["files"]) or "repository"

        if polarity == "positive":
            outcome_summary = (
                f"Verified fix ({operator}) on {files_str} resolved {item['exception_type']} "
                f"in {item['mttr_seconds']:.1f}s with 0 subsequent recurrences."
            )
            anti_pattern = ""
        else:
            if effective_status == "rolled_back":
                reason = item["error"] or "post-deployment health check failed"
                anti_pattern = (
                    f"NEGATIVE TRAJECTORY (ROLLED BACK): Patch on {files_str} ({item['diagnosis'] or operator}) "
                    f"failed post-apply health check ({reason}) and triggered automatic rollback. Do NOT repeat this change."
                )
            elif effective_status == "regressed_after_repair":
                anti_pattern = (
                    f"NEGATIVE TRAJECTORY (INCOMPLETE FIX / REGRESSED): Prior patch on {files_str} "
                    f"({item['diagnosis'] or operator}) passed initial tests but {item['exception_type']} recurred later. "
                    "Address the deeper root cause rather than reusing that partial fix."
                )
            else:
                reason = item["error"] or "failed isolated worktree verification tests"
                anti_pattern = (
                    f"NEGATIVE TRAJECTORY ({effective_status.upper()}): Attempted fix on {files_str} "
                    f"({item['diagnosis'] or operator}) failed with '{reason}'. Avoid this approach."
                )
            outcome_summary = anti_pattern

        cases.append(
            EpisodicIncidentCase(
                case_id=f"case:{folder}",
                incident_id=item["incident_id"],
                trajectory_polarity=polarity,
                status=effective_status,
                exception_type=item["exception_type"],
                error_message=item["message"],
                drain_template=tpl,
                affected_files=item["files"],
                commit_sha=item["commit_sha"],
                diagnosis=item["diagnosis"],
                unified_diff=item["unified_diff"],
                structural_operator=operator,
                tests_to_run=item["tests_to_run"],
                outcome_summary=outcome_summary,
                anti_pattern_warning=anti_pattern,
                recovery_time_seconds=item["mttr_seconds"],
                post_repair_survived=survived and polarity == "positive",
                kb_trust_status=kb_trust,
                promoted_by=item["promoted_by"],
                drift_status=drift_status,
                drift_reason=drift_reason,
                dead_end_actions=item["dead_end_actions"],
            )
        )

    return cases


def retrieve_tier2_episodic_cases(
    query_text: str,
    query_files: Iterable[str],
    cases: Sequence[EpisodicIncidentCase],
    *,
    top_k_positive: int = 3,
    top_k_negative: int = 3,
    rrf_k: int = 60,
) -> tuple[tuple[EpisodicIncidentCase, ...], tuple[EpisodicIncidentCase, ...]]:
    """
    Retrieve both Positive and Negative Case-Based Reasoning trajectories via
    Hybrid Graph-RAG (BM25 + Dense Subword/Identifier Cosine + File/Structural Overlap RRF).
    """
    if not cases:
        return (), ()

    q_files_set = set(query_files)
    q_tokens = _tokenize(f"{query_text} {' '.join(q_files_set)}")
    q_vec = _build_dense_semantic_vector(query_text, q_files_set, ())

    doc_tokens = [
        _tokenize(f"{c.exception_type} {c.error_message} {c.diagnosis} {c.structural_operator} {' '.join(c.affected_files)}")
        for c in cases
    ]
    bm25_raw = _bm25_scores(q_tokens, doc_tokens)
    sem_raw: list[float] = []
    graph_raw: list[float] = []

    for c in cases:
        c_vec = _build_dense_semantic_vector(
            f"{c.exception_type} {c.error_message} {c.diagnosis}",
            c.affected_files,
            (c.structural_operator,),
        )
        sem_raw.append(_cosine_similarity(q_vec, c_vec))
        c_files = set(c.affected_files)
        if q_files_set and c_files:
            inter = len(q_files_set & c_files)
            union = len(q_files_set | c_files)
            graph_raw.append(inter / max(union, 1))
        else:
            graph_raw.append(0.0)

    n = len(cases)
    bm25_ranks = {idx: r for r, idx in enumerate(sorted(range(n), key=lambda i: (-bm25_raw[i], i)), start=1)}
    sem_ranks = {idx: r for r, idx in enumerate(sorted(range(n), key=lambda i: (-sem_raw[i], i)), start=1)}
    graph_ranks = {idx: r for r, idx in enumerate(sorted(range(n), key=lambda i: (-graph_raw[i], i)), start=1)}

    scored_pos: list[EpisodicIncidentCase] = []
    scored_neg: list[EpisodicIncidentCase] = []

    for idx, c in enumerate(cases):
        if bm25_raw[idx] <= 0.0 and sem_raw[idx] <= 0.05 and graph_raw[idx] <= 0.0:
            continue
        rrf = (
            1.0 / (rrf_k + bm25_ranks[idx])
            + 1.0 / (rrf_k + sem_ranks[idx])
            + 1.0 / (rrf_k + graph_ranks[idx])
        )
        direct_sim = 0.45 * sem_raw[idx] + 0.35 * graph_raw[idx] + 0.20 * min(1.0, bm25_raw[idx] / 8.0)
        combined = 0.5 * (rrf / (3.0 / (rrf_k + 1.0))) + 0.5 * direct_sim

        enriched = EpisodicIncidentCase(
            case_id=c.case_id,
            incident_id=c.incident_id,
            trajectory_polarity=c.trajectory_polarity,
            status=c.status,
            exception_type=c.exception_type,
            error_message=c.error_message,
            drain_template=c.drain_template,
            affected_files=c.affected_files,
            commit_sha=c.commit_sha,
            diagnosis=c.diagnosis,
            unified_diff=c.unified_diff,
            structural_operator=c.structural_operator,
            tests_to_run=c.tests_to_run,
            outcome_summary=c.outcome_summary,
            anti_pattern_warning=c.anti_pattern_warning,
            recovery_time_seconds=c.recovery_time_seconds,
            post_repair_survived=c.post_repair_survived,
            similarity_score=combined,
            kb_trust_status=c.kb_trust_status,
            promoted_by=c.promoted_by,
            drift_status=c.drift_status,
            drift_reason=c.drift_reason,
            dead_end_actions=c.dead_end_actions,
        )
        if c.trajectory_polarity == "positive":
            scored_pos.append(enriched)
        else:
            scored_neg.append(enriched)

    scored_pos.sort(key=lambda x: (-x.similarity_score, x.incident_id))
    scored_neg.sort(key=lambda x: (-x.similarity_score, x.incident_id))
    return tuple(scored_pos[:top_k_positive]), tuple(scored_neg[:top_k_negative])


# ──────────────────────────────────────────────────────────────────────────────
# Tier 3 — Procedural Repair Memory: Playbook Registry + Thompson Sampling Bandit
# ──────────────────────────────────────────────────────────────────────────────

def _thompson_bandit_score(
    alpha: float,
    beta_param: float,
    mean_recovery_seconds: float,
    seed_key: str,
) -> tuple[float, float]:
    """
    Compute Bayesian Posterior Confidence E[Beta(alpha, beta)] = alpha / (alpha + beta)
    and a reproducible Thompson Sampling draw theta ~ Beta(alpha, beta) weighted by
    recovery speed factor 1 / (1 + 0.02 * mttr).
    """
    posterior_mean = alpha / max(alpha + beta_param, 1e-9)
    seed_int = int(hashlib.sha1(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed_int)
    sampled_theta = rng.betavariate(max(0.1, alpha), max(0.1, beta_param))
    speed_factor = 1.0 / (1.0 + 0.02 * max(0.0, mean_recovery_seconds))
    bandit_score = 0.65 * sampled_theta + 0.25 * posterior_mean + 0.10 * speed_factor
    return posterior_mean, min(1.0, bandit_score)


def build_tier3_procedural_skills(
    cases: Sequence[EpisodicIncidentCase],
    query_text: str = "",
) -> tuple[ProceduralRepairSkill, ...]:
    """
    Build the Tier 3 Procedural Skill & Playbook Registry by combining:
      1. Built-in parameterized operational remediation skills
      2. Learned AST code-repair playbooks aggregated by `structural_operator` across
         historical episodic cases, scored via Beta-Bernoulli Thompson Sampling.
    """
    skills: list[ProceduralRepairSkill] = []

    # 1. Learned AST Code-Repair Playbooks from Episodic History
    by_operator: dict[str, list[EpisodicIncidentCase]] = defaultdict(list)
    for c in cases:
        if c.unified_diff or c.diagnosis:
            by_operator[c.structural_operator].append(c)

    operator_descriptions: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
        "collection_boundary_or_slice_fix": (
            "Synthesize slice stride, range boundary, or collection window fix for iteration/batching defects",
            ("target_file", "symbol_name", "slice_or_range_expr", "verification_test"),
            ("Verify collection length > 0 and preserve return container type", "Run unit & adjacent blast-radius tests in isolated worktree"),
        ),
        "precondition_or_null_guard": (
            "Insert defensive precondition or None/empty validation guard before dereferencing or indexing",
            ("target_file", "symbol_name", "guard_condition", "fallback_or_raise"),
            ("Confirm upstream callers expect fallback value or typed exception", "Replay original incident traceback in detached git worktree"),
        ),
        "exception_handling_guard": (
            "Wrap fragile I/O or parsing call in scoped try/except handler with deterministic fallback",
            ("target_file", "symbol_name", "caught_exception_type", "recovery_action"),
            ("Never catch bare Exception without logging or re-raising domain error", "Verify no silent data corruption in downstream consumers"),
        ),
        "timeout_or_capacity_parameter_tuning": (
            "Update connection pool, retry backoff, or request timeout parameter to prevent resource starvation",
            ("target_file", "config_or_kwarg_name", "updated_bound", "health_check_cmd"),
            ("Check downstream service capacity limits before increasing concurrency", "Run post-apply --health-check with automatic rollback"),
        ),
        "return_expression_or_contract_fix": (
            "Repair function return expression or output contract to match caller expectations",
            ("target_file", "symbol_name", "expected_return_contract", "verification_test"),
            ("Inspect all direct callers in DependencyGraph to ensure backward compatibility", "Run adjacent consumer regression tests"),
        ),
        "statement_logic_refinement": (
            "Apply minimal localized AST statement fix at confirmed stack-frame crash line",
            ("target_file", "target_line", "search_block", "replace_block"),
            ("Confirm target file is patch_eligible with confirmed stack frame evidence", "Validate diff security scan before worktree execution"),
        ),
    }

    for op_name, op_cases in by_operator.items():
        successes = sum(
            1
            for c in op_cases
            if c.trajectory_polarity == "positive" and c.post_repair_survived and c.drift_status == "clean"
        )
        failures = sum(1 for c in op_cases if c.status in {"failed", "rejected"})
        rollbacks = sum(1 for c in op_cases if c.status == "rolled_back")
        recurrences = sum(1 for c in op_cases if c.status == "regressed_after_repair")
        drifted_cnt = sum(1 for c in op_cases if c.drift_status == "drifted")

        # Beta-Bernoulli update: alpha = 1 + successes; beta = 1 + failures + 2*rollbacks + 2*recurrences + drifted
        alpha = 1.0 + float(successes)
        beta_param = 1.0 + float(failures) + 2.0 * float(rollbacks) + 2.0 * float(recurrences) + float(drifted_cnt)
        mttrs = [c.recovery_time_seconds for c in op_cases if c.trajectory_polarity == "positive"]
        mean_mttr = sum(mttrs) / len(mttrs) if mttrs else 2.5

        desc, params, preconds = operator_descriptions.get(
            op_name,
            (
                f"Apply validated '{op_name}' code transformation in isolated worktree",
                ("target_file", "search_block", "replace_block", "verification_test"),
                ("Verify confirmed evidence tier before patching", "Replay original incident failure pre- and post-patch"),
            ),
        )
        pos_samples = [c for c in op_cases if c.trajectory_polarity == "positive" and c.unified_diff]
        sample_diff = (pos_samples[0].unified_diff if pos_samples else op_cases[0].unified_diff)[:320]
        excs = tuple(dict.fromkeys(c.exception_type for c in op_cases if c.exception_type))[:5]
        files = tuple(dict.fromkeys(f for c in op_cases for f in c.affected_files))[:5]

        post_conf, bandit = _thompson_bandit_score(alpha, beta_param, mean_mttr, f"playbook:{op_name}:{query_text[:24]}")
        skills.append(
            ProceduralRepairSkill(
                skill_id=f"playbook:{op_name}",
                name=f"AST Playbook — {op_name}",
                skill_kind="ast_code_playbook",
                structural_operator=op_name,
                description=desc,
                parameters=params,
                preconditions=preconds,
                applicable_exceptions=excs,
                target_file_patterns=files,
                sample_diff_or_command=sample_diff,
                alpha=alpha,
                beta_param=beta_param,
                posterior_confidence=post_conf,
                thompson_bandit_score=bandit,
                mean_recovery_seconds=mean_mttr,
            )
        )

    # 2. Built-in Parameterized Operational Remediation Skills
    builtin_specs = (
        (
            "skill:isolate_and_replay_worktree",
            "isolate_and_replay_worktree(repo, revision, unified_diff, test_cmd)",
            "worktree_verification",
            "Create detached git worktree at incident commit, reproduce pre-patch failure, apply diff, and run regression + blast-radius tests",
            ("repo_path", "commit_sha", "unified_diff", "tests_to_run"),
            ("Emergency stop switch must be clear", "Unified diff must pass static security & path-containment gates"),
            ("All",),
            ("*",),
            "git worktree add --detach <tmp> <sha> && python -m unittest <tests>",
            5.0,
            1.0,
            1.8,
        ),
        (
            "skill:rollback_deployment_patch",
            "rollback_applied_patch(repo, incident_id, outcome_or_diff_path)",
            "automatic_or_manual_rollback",
            "Reverse a live patch using `git apply --reverse` when post-deployment health checks fail or on operator command",
            ("repo_path", "rollback_target", "operator"),
            ("Verify reverse diff applies cleanly with `git apply --reverse --check`", "Record rollback event in immutable `.asre/audit.jsonl`"),
            ("HealthCheckFailure", "RegressionError"),
            ("*",),
            "python main.py --local <repo> --rollback <run_dir_or_diff>",
            4.0,
            1.0,
            0.9,
        ),
        (
            "skill:create_guarded_repair_pr",
            "create_repair_pull_request(plan, context, execution, branch_prefix='asre/fix')",
            "pull_request_escalation",
            "Commit verified patch to dedicated branch `asre/fix-<id>` with 9-part RCA, untouched-file justification, and blast-radius report",
            ("repo_path", "incident_id", "branch_name", "pr_body_markdown"),
            ("Patch execution status MUST be 'verified' with >0 tests executed", "Redact all secrets from PR body before committing"),
            ("All",),
            ("*",),
            "python main.py --local <repo> --incident <json> --generate-patch-plan --verify-patch-plan --create-pr",
            4.0,
            1.0,
            2.2,
        ),
    )

    for sid, sname, sop, sdesc, sparams, spre, sexc, sfiles, scmd, salpha, sbeta, smttr in builtin_specs:
        p_conf, b_score = _thompson_bandit_score(salpha, sbeta, smttr, f"{sid}:{query_text[:24]}")
        skills.append(
            ProceduralRepairSkill(
                skill_id=sid,
                name=sname,
                skill_kind="operational_workflow",
                structural_operator=sop,
                description=sdesc,
                parameters=sparams,
                preconditions=spre,
                applicable_exceptions=sexc,
                target_file_patterns=sfiles,
                sample_diff_or_command=scmd,
                alpha=salpha,
                beta_param=sbeta,
                posterior_confidence=p_conf,
                thompson_bandit_score=b_score,
                mean_recovery_seconds=smttr,
            )
        )

    skills.sort(
        key=lambda s: (
            1 if s.skill_kind == "ast_code_playbook" else 0,
            s.thompson_bandit_score,
            s.posterior_confidence,
        ),
        reverse=True,
    )
    return tuple(skills)


# ──────────────────────────────────────────────────────────────────────────────
# Unified 3-Tier Knowledge Base Orchestrator & Persistence
# ──────────────────────────────────────────────────────────────────────────────

def build_and_query_knowledge_base(
    repo_root: str,
    *,
    query: str = "",
    target_files: Iterable[str] = (),
    repair_run_dir: Optional[str] = ".asre/runs",
    persist_index: bool = True,
) -> KnowledgeBaseReport:
    """
    Build, index, and query all three tiers of the ASRE Repair Memory & Knowledge Base:
      • Tier 1: Semantic Docs, CODEOWNERS, Runbooks & IaC (Parent-Document Chunking + Cross-Encoder Reranker)
      • Tier 2: Episodic Incident Cases (Case-Based Reasoning with Positive & Negative Trajectories)
      • Tier 3: Procedural Repair Skills & AST Playbooks (Beta-Bernoulli Thompson Sampling Bandit)
    """
    target_file_list = list(target_files)
    effective_query = query.strip() or " ".join(target_file_list) or "incident repair runbook architecture"

    # Tier 1: Semantic KB & CODEOWNERS
    codeowners_rules = parse_codeowners(repo_root)
    parent_docs = ingest_tier1_semantic_documents(repo_root)
    tier1_hits = retrieve_tier1_semantic_knowledge(effective_query, parent_docs, top_k=4)

    # Tier 2: Episodic CBR Memory (Positive & Negative Trajectories)
    all_cases = load_tier2_episodic_cases(repair_run_dir, repo_root)
    pos_cases, neg_cases = retrieve_tier2_episodic_cases(
        effective_query,
        target_file_list,
        all_cases,
        top_k_positive=3,
        top_k_negative=3,
    )
    if not pos_cases and all_cases:
        pos_cases = tuple(c for c in all_cases if c.trajectory_polarity == "positive")[:3]
    if not neg_cases and all_cases:
        neg_cases = tuple(c for c in all_cases if c.trajectory_polarity == "negative")[:3]

    all_referenced_files = list(
        dict.fromkeys([*target_file_list, *(f for c in all_cases for f in c.affected_files)])
    )
    resolved_owners = resolve_owners_for_files(all_referenced_files, codeowners_rules)

    # Tier 3: Procedural Skills & Playbooks ranked by Thompson Sampling
    tier3_skills = build_tier3_procedural_skills(all_cases, query_text=effective_query)

    persisted_path = ""
    if persist_index:
        kb_dir = Path(repair_run_dir).resolve().parent if repair_run_dir else (Path(repo_root).resolve() / ".asre")
        try:
            kb_dir.mkdir(parents=True, exist_ok=True)
            kb_file = kb_dir / "knowledge_base.json"
            persisted_path = str(kb_file)
        except OSError:
            persisted_path = ""

    report = KnowledgeBaseReport(
        query=effective_query,
        total_semantic_parent_docs=len(parent_docs),
        total_episodic_cases=len(all_cases),
        positive_cases_count=sum(1 for c in all_cases if c.trajectory_polarity == "positive"),
        negative_cases_count=sum(1 for c in all_cases if c.trajectory_polarity == "negative"),
        total_procedural_skills=len(tier3_skills),
        codeowners_rules=codeowners_rules,
        resolved_owners=resolved_owners,
        tier1_semantic_hits=tier1_hits,
        tier2_positive_cases=pos_cases,
        tier2_negative_cases=neg_cases,
        tier3_procedural_skills=tier3_skills,
        persisted_index_path=persisted_path,
        verified_trust_count=sum(1 for c in all_cases if c.kb_trust_status == "verified"),
        unverified_trust_count=sum(1 for c in all_cases if c.kb_trust_status == "unverified"),
        drifted_cases_count=sum(1 for c in all_cases if c.drift_status == "drifted"),
    )

    if persisted_path:
        try:
            Path(persisted_path).write_text(
                redact_secrets(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"),
                encoding="utf-8",
            )
        except OSError:
            pass

    return report


def promote_knowledge_base_record(
    repair_run_dir: str,
    incident_id: str,
    *,
    approver: str,
) -> dict[str, Any]:
    """
    Human-in-the-Loop Trust Promotion Gate:
    Promote a stored repair run from `kb_trust_status: "unverified"` to `"verified"`
    after an engineer approves the postmortem PR or repair review, unlocking it for
    future autonomous execution.
    """
    if not approver or not approver.strip():
        raise ValueError("human approver identifier is required to promote a Knowledge Base repair record")

    root = Path(repair_run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"repair run directory not found: {repair_run_dir!r}")

    matched_run_file: Optional[Path] = None
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        run_file = entry / "run.json"
        if not run_file.is_file():
            continue
        try:
            data = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if entry.name == incident_id or str(data.get("incident_id")) == incident_id:
            matched_run_file = run_file
            break

    if matched_run_file is None:
        raise ValueError(f"no repair run matching {incident_id!r} found in {repair_run_dir!r}")

    run_data = json.loads(matched_run_file.read_text(encoding="utf-8"))
    run_data["kb_trust_status"] = "verified"
    run_data["human_approved"] = True
    run_data["promoted_by"] = approver.strip()
    run_data["promoted_at"] = datetime.now(timezone.utc).isoformat()
    if isinstance(run_data.get("distilled_record"), dict):
        run_data["distilled_record"]["kb_trust_status"] = "verified"
        run_data["distilled_record"]["promoted_by"] = approver.strip()

    matched_run_file.write_text(
        redact_secrets(json.dumps(run_data, indent=2, sort_keys=True) + "\n"),
        encoding="utf-8",
    )
    append_audit_log(
        root,
        incident_id=incident_id,
        action="promote_knowledge_base_record",
        status="verified",
        operator=approver.strip(),
        role="sre",
        environment="production",
        details={"run_file": str(matched_run_file), "kb_trust_status": "verified"},
    )
    return {
        "incident_id": incident_id,
        "kb_trust_status": "verified",
        "promoted_by": approver.strip(),
        "run_file": str(matched_run_file),
    }


def verify_stored_repair_guardrails(
    repo_root: str,
    case: EpisodicIncidentCase,
    *,
    max_risk_score: int = 0,
    auto_risk_threshold: int = 20,
    explicit_human_approval: bool = False,
    watchdog_configured: bool = True,
) -> PreExecutionGuardrailResult:
    """
    4. Safety Guardrails Before Autonomous Execution of a Stored Repair:
      • Check 1 (Precondition & Drift Verification): Emergency stop clear, target files exist,
        zero architectural/code drift, and positive trajectory polarity.
      • Check 2 (Risk, Blast-Radius & KB Trust Gate Policy): Must be `kb_trust_status == "verified"`,
        low blast radius (`max_risk_score < auto_risk_threshold`), and touch 0 sensitive auth/DB domains
        unless `explicit_human_approval` is provided.
      • Check 3 (Automatic Rollback Watchdog): Confirms post-remediation watchdog is configured.
    """
    reasons: list[str] = []
    stop_active, stop_reason = is_emergency_stop_active(repo_root)
    drift_status, drift_reason = detect_repair_record_drift(
        repo_root,
        case.affected_files,
        case.unified_diff,
    )
    drift_free = (drift_status == "clean") and (case.drift_status == "clean")
    preconditions_ok = (not stop_active) and drift_free and (case.trajectory_polarity == "positive")

    if stop_active:
        reasons.append(f"Blocked by emergency stop: {stop_reason}")
    if not drift_free:
        reasons.append(f"Blocked by architectural/code drift: {drift_reason or case.drift_reason}")
    if case.trajectory_polarity != "positive":
        reasons.append(f"Blocked: stored record '{case.incident_id}' is a negative trajectory ({case.status})")

    trust_verified = case.kb_trust_status == "verified"
    if not trust_verified:
        reasons.append(
            f"Trust gate requires promotion: record '{case.incident_id}' has kb_trust_status='{case.kb_trust_status}'"
        )

    sens_list: list[str] = []
    for f_path in case.affected_files:
        if _AUTH_SECURITY_PATH_RE.search(f_path) and "authentication_or_security" not in sens_list:
            sens_list.append("authentication_or_security")
        if _DB_MIGRATION_PATH_RE.search(f_path) and "database_migration_or_schema" not in sens_list:
            sens_list.append("database_migration_or_schema")
    if case.unified_diff and _DB_DDL_DIFF_RE.search(case.unified_diff):
        if "database_migration_or_schema" not in sens_list:
            sens_list.append("database_migration_or_schema")
    sensitive = tuple(sens_list)
    low_blast_radius = (max_risk_score < auto_risk_threshold) and (len(sensitive) == 0)
    risk_policy_passed = low_blast_radius or explicit_human_approval

    if sensitive and not explicit_human_approval:
        reasons.append(
            f"Requires 1-click human approval: touches stateful/sensitive domain(s) {', '.join(sensitive)}"
        )
    if max_risk_score >= auto_risk_threshold and not explicit_human_approval:
        reasons.append(
            f"Requires 1-click human approval: blast-radius risk {max_risk_score}/90 >= auto threshold {auto_risk_threshold}"
        )
    if not watchdog_configured:
        reasons.append("Automatic rollback watchdog must be configured before execution")

    requires_human = (not trust_verified) or (not low_blast_radius)
    autonomous_ok = (
        preconditions_ok
        and (trust_verified or explicit_human_approval)
        and risk_policy_passed
        and watchdog_configured
    )
    if autonomous_ok and not reasons:
        reasons.append(
            f"All 3 safety guardrails passed for stored repair '{case.incident_id}' "
            f"(preconditions verified, kb_trust_status=verified, risk={max_risk_score}/90, rollback watchdog active)"
        )

    return PreExecutionGuardrailResult(
        autonomous_execution_permitted=autonomous_ok,
        preconditions_verified=preconditions_ok,
        drift_free=drift_free,
        trust_gate_verified=trust_verified,
        risk_policy_passed=risk_policy_passed,
        watchdog_configured=watchdog_configured,
        requires_human_approval=requires_human and not explicit_human_approval,
        max_risk_score=max_risk_score,
        sensitive_domains=sensitive,
        reasons=tuple(reasons),
    )

