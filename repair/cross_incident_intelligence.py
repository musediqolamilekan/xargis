"""
Cross-Incident Intelligence Engine — Domain-Neutral 3-Layer Hybrid Graph-RAG + Temporal Causal Pipeline.

Connects incidents of ANY error type, language, or application domain that appear unrelated
by discovering their shared underlying cause through pure graph topology, AST code vocabulary,
temporal lead-lag causality, and optional LLM synthesis—with zero hardcoded domain assumptions:

  Layer A — Semantic Memory & AST-Grounded Vector Space (Hybrid Graph-RAG & Reciprocal Rank Fusion):
    • Decomposes arbitrary CamelCase / snake_case identifiers, exception names, stack symbols,
      and AST-extracted code vocabulary (functions, calls, I/O surfaces, config keys) into
      corpus-adaptive IDF + character N-gram (3/4-gram) subword vectors.
    • Fuses BM25 sparse keyword search, subword/identifier cosine similarity, and AST
      Dependency-Cone structural overlap via Reciprocal Rank Fusion (RRF).

  Layer B — Topology & Blast Radius (Personalized PageRank + Topological Sink / LCA Discovery):
    • Constructs a multi-layer topology graph over incidents, source files, AST import edges,
      and AST-discovered shared resources (functions/symbols, DB calls, service calls, config keys).
    • Runs Personalized PageRank (PPR) + Multi-Incident Reverse-BFS Reachability + Topological
      Sink Scoring (in-reach vs. out-reach in the dependency DAG) to isolate the Lowest Common
      Ancestor (LCA) root-cause module for any repository structure.

  Layer C — Temporal & Causal Correlation (Drain Template Mining + Lead-Lag Causality + Change-Points):
    • Extracts deterministic log templates using a Drain3-style fixed-depth token parse tree.
    • Orders incidents into a directed causal cascade (Root Trigger → Downstream Symptoms)
      using temporal onset lag + directed dependency paths + topological sink depth.
    • Correlates cluster onset with recent Git commit change-points on the root-cause module
      and optionally refines the natural-language hypothesis via the configured LLM provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from intelligence.dependency_graph import DependencyGraph
from intelligence.file_profile import FileProfile
from intelligence.git_analyzer import get_recent_commits_for_files
from .incident_intelligence import (
    load_telemetry_events,
    normalize_telemetry_payload,
)
from .incident_mapper import (
    Incident,
    _match_repository_path_detailed,
    extract_stack_frames,
)


# ──────────────────────────────────────────────────────────────────────────────
# Layer C.1 — Drain3-Style Log Template Miner (Format-Neutral)
# ──────────────────────────────────────────────────────────────────────────────

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_IP_PORT_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b")
_DURATION_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ns|us|µs|ms|s|sec|seconds|m|min|h)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])")
_QUOTED_PARAM_RE = re.compile(r"'[^']{1,64}'|\"[^\"]{1,64}\"")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{1,63}|\d+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "was", "were", "are",
    "has", "have", "had", "not", "but", "into", "while", "when", "during",
    "file", "line", "most", "recent", "call", "last", "traceback",
})


@dataclass(frozen=True)
class DrainLogTemplate:
    template_id: str
    template: str
    parameter_count: int
    sample_parameters: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "template": self.template,
            "parameter_count": self.parameter_count,
            "sample_parameters": list(self.sample_parameters),
        }


class DrainTemplateMiner:
    """
    Fixed-depth Drain3-style log template parser that groups raw messages by token length
    and prefix structure, masking dynamic variables into `<*>` wildcards regardless of
    programming language or log format.
    """

    def __init__(self, similarity_threshold: float = 0.65) -> None:
        self.similarity_threshold = similarity_threshold
        self._clusters_by_len: dict[int, list[list[str]]] = defaultdict(list)

    @staticmethod
    def _preprocess(message: str) -> tuple[list[str], list[str]]:
        params: list[str] = []

        def _capture(match: re.Match[str]) -> str:
            params.append(match.group(0))
            return "<*>"

        text = message.strip()
        text = _UUID_RE.sub(_capture, text)
        text = _HEX_RE.sub(_capture, text)
        text = _IP_PORT_RE.sub(_capture, text)
        text = _DURATION_RE.sub(_capture, text)
        text = _QUOTED_PARAM_RE.sub(_capture, text)
        text = _NUMBER_RE.sub(_capture, text)
        tokens = text.split()
        return tokens, params

    def extract_template(self, message: str) -> DrainLogTemplate:
        tokens, params = self._preprocess(message)
        if not tokens:
            return DrainLogTemplate("tpl-empty", "<empty>", 0, ())

        length = len(tokens)
        candidates = self._clusters_by_len[length]
        best_cluster: Optional[list[str]] = None
        best_sim = -1.0

        for cluster_tokens in candidates:
            if (
                cluster_tokens[0] != "<*>"
                and tokens[0] != "<*>"
                and cluster_tokens[0].lower() != tokens[0].lower()
            ):
                continue
            matches = sum(
                1
                for t1, t2 in zip(cluster_tokens, tokens)
                if t1.lower() == t2.lower() or t1 == "<*>" or t2 == "<*>"
            )
            sim = matches / max(length, 1)
            if sim >= self.similarity_threshold and sim > best_sim:
                best_sim = sim
                best_cluster = cluster_tokens

        if best_cluster is None:
            new_cluster = list(tokens)
            candidates.append(new_cluster)
            template_tokens = new_cluster
        else:
            for idx, (c_tok, m_tok) in enumerate(zip(best_cluster, tokens)):
                if c_tok.lower() != m_tok.lower():
                    if c_tok != "<*>":
                        params.append(m_tok)
                    best_cluster[idx] = "<*>"
            template_tokens = best_cluster

        template_str = " ".join(template_tokens)
        tpl_id = "tpl-" + hashlib.sha1(template_str.lower().encode("utf-8")).hexdigest()[:8]
        return DrainLogTemplate(
            template_id=tpl_id,
            template=template_str,
            parameter_count=template_tokens.count("<*>"),
            sample_parameters=tuple(params[:5]),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Data Models for Cross-Incident Intelligence
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NormalizedIncidentNode:
    incident_id: str
    title: str
    service: str
    exception_type: str
    message: str
    drain_template: DrainLogTemplate
    timestamp: str
    epoch_seconds: Optional[float]
    trace_id: str
    mapped_files: tuple[str, ...]
    dependency_cone: dict[str, int]
    shared_resources: tuple[str, ...]
    concepts: tuple[str, ...]
    raw_incident: Incident
    span_id: str = ""
    parent_span_id: str = ""
    outbound_service_calls: tuple[str, ...] = ()
    inbound_api_endpoints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "service": self.service,
            "exception_type": self.exception_type,
            "message": self.message,
            "drain_template": self.drain_template.to_dict(),
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "mapped_files": list(self.mapped_files),
            "dependency_cone": dict(self.dependency_cone),
            "shared_resources": list(self.shared_resources),
            "outbound_service_calls": list(self.outbound_service_calls),
            "inbound_api_endpoints": list(self.inbound_api_endpoints),
            "concepts": list(self.concepts),
        }


@dataclass(frozen=True)
class IncidentGraphNode:
    node_id: str
    kind: str  # incident, code_module, shared_resource, commit_changepoint
    label: str
    ppr_score: float = 0.0
    reaching_incidents: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "ppr_score": round(self.ppr_score, 4),
            "reaching_incidents": self.reaching_incidents,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IncidentGraphEdge:
    source: str
    target: str
    relation: str  # maps_to_file, depends_on, shares_dependency, shares_infrastructure, temporal_causal_cascade, shared_trace, triggered_by_commit
    weight: float
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": round(self.weight, 3),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class IncidentRelationshipGraph:
    nodes: tuple[IncidentGraphNode, ...]
    edges: tuple[IncidentGraphEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


@dataclass(frozen=True)
class TemporalCausalLink:
    cause_incident_id: str
    effect_incident_id: str
    lag_seconds: float
    causal_confidence: float
    mechanism: str
    cause_service: str = ""
    effect_service: str = ""
    granger_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause_incident_id": self.cause_incident_id,
            "effect_incident_id": self.effect_incident_id,
            "cause_service": self.cause_service,
            "effect_service": self.effect_service,
            "lag_seconds": round(self.lag_seconds, 2),
            "causal_confidence": round(self.causal_confidence, 2),
            "granger_score": round(self.granger_score, 3),
            "mechanism": self.mechanism,
        }


@dataclass(frozen=True)
class ServiceCausalAttribution:
    """
    Counterfactual Do-Calculus Root-Cause Attribution for a service/incident node
    in the multi-hop cross-service causal DAG.
    """
    service: str
    incident_id: str
    counterfactual_resolution_pct: float
    downstream_incidents_caused: tuple[str, ...]
    downstream_services_affected: tuple[str, ...]
    causal_propagation_path: tuple[str, ...]
    granger_lag_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "incident_id": self.incident_id,
            "counterfactual_resolution_pct": round(self.counterfactual_resolution_pct, 1),
            "downstream_incidents_caused": list(self.downstream_incidents_caused),
            "downstream_services_affected": list(self.downstream_services_affected),
            "causal_propagation_path": list(self.causal_propagation_path),
            "granger_lag_score": round(self.granger_lag_score, 3),
        }


@dataclass(frozen=True)
class HistoricalIncidentComparison:
    historical_incident_id: str
    historical_status: str
    rrf_score: float
    similarity_pct: float
    bm25_rank: int
    semantic_rank: int
    graph_rank: int
    shared_files: tuple[str, ...]
    shared_concepts: tuple[str, ...]
    historical_diagnosis: str
    reusable_lesson: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "historical_incident_id": self.historical_incident_id,
            "historical_status": self.historical_status,
            "rrf_score": round(self.rrf_score, 4),
            "similarity_pct": round(self.similarity_pct, 1),
            "bm25_rank": self.bm25_rank,
            "semantic_rank": self.semantic_rank,
            "graph_rank": self.graph_rank,
            "shared_files": list(self.shared_files),
            "shared_concepts": list(self.shared_concepts),
            "historical_diagnosis": self.historical_diagnosis,
            "reusable_lesson": self.reusable_lesson,
        }


@dataclass(frozen=True)
class CorrelatedIncidentGroup:
    correlation_id: str
    common_root_cause_file: Optional[str]
    common_root_cause_resource: Optional[str]
    root_trigger_incident_id: str
    root_cause_hypothesis: str
    ppr_score: float
    confidence: float
    incidents: tuple[NormalizedIncidentNode, ...]
    shared_dependencies: tuple[str, ...]
    shared_resources: tuple[str, ...]
    temporal_causal_chain: tuple[TemporalCausalLink, ...]
    changepoint_commits: tuple[dict[str, str], ...]
    correlation_evidence: tuple[str, ...]
    merged_incident: Incident
    service_causal_attributions: tuple[ServiceCausalAttribution, ...] = ()
    service_propagation_paths: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "common_root_cause_file": self.common_root_cause_file,
            "common_root_cause_resource": self.common_root_cause_resource,
            "root_trigger_incident_id": self.root_trigger_incident_id,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "ppr_score": round(self.ppr_score, 4),
            "confidence": round(self.confidence, 2),
            "incidents": [inc.to_dict() for inc in self.incidents],
            "shared_dependencies": list(self.shared_dependencies),
            "shared_resources": list(self.shared_resources),
            "temporal_causal_chain": [link.to_dict() for link in self.temporal_causal_chain],
            "service_causal_attributions": [a.to_dict() for a in self.service_causal_attributions],
            "service_propagation_paths": [list(p) for p in self.service_propagation_paths],
            "changepoint_commits": list(self.changepoint_commits),
            "correlation_evidence": list(self.correlation_evidence),
            "merged_incident": self.merged_incident.to_dict(),
        }


@dataclass(frozen=True)
class CrossIncidentIntelligenceReport:
    total_incidents_analyzed: int
    correlated_root_causes: int
    uncorrelated_incidents: int
    correlated_groups: tuple[CorrelatedIncidentGroup, ...]
    relationship_graph: IncidentRelationshipGraph
    historical_comparisons: tuple[HistoricalIncidentComparison, ...]
    recurring_patterns: tuple[str, ...]

    @property
    def primary_merged_incident(self) -> Optional[Incident]:
        if self.correlated_groups:
            return self.correlated_groups[0].merged_incident
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_incidents_analyzed": self.total_incidents_analyzed,
                "correlated_root_causes": self.correlated_root_causes,
                "uncorrelated_incidents": self.uncorrelated_incidents,
            },
            "correlated_groups": [g.to_dict() for g in self.correlated_groups],
            "relationship_graph": self.relationship_graph.to_dict(),
            "historical_comparisons": [h.to_dict() for h in self.historical_comparisons],
            "recurring_patterns": list(self.recurring_patterns),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Layer A — Domain-Neutral Identifier Decomposition & AST-Grounded Vector Space
# ──────────────────────────────────────────────────────────────────────────────

def _decompose_identifier(raw: str) -> list[str]:
    """
    Decompose any CamelCase, PascalCase, snake_case, kebab-case, or dotted identifier
    into normalized morphological root tokens without domain assumptions.
    Example: 'ConnectionPoolExhaustedError' -> ['connection', 'pool', 'exhausted', 'error']
    """
    expanded = _CAMEL_BOUNDARY_RE.sub(" ", raw)
    expanded = re.sub(r"[/_.:-]+", " ", expanded)
    parts: list[str] = []
    for tok in _TOKEN_RE.findall(expanded.lower()):
        if len(tok) >= 2 and not tok.isdigit() and tok not in _STOPWORDS:
            parts.append(tok)
    return parts


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_tok in _TOKEN_RE.findall(text):
        lower = raw_tok.lower()
        if len(lower) >= 2 and lower not in _STOPWORDS:
            tokens.append(lower)
        for sub in _decompose_identifier(raw_tok):
            if sub != lower and sub not in _STOPWORDS:
                tokens.append(sub)
    return tokens


def _light_stem(token: str) -> str:
    """Language-neutral morphological stemmer for technical terms."""
    for suffix in ("ing", "tion", "ions", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _extract_structural_concepts(
    text: str,
    files: Iterable[str],
    resources: Iterable[str],
    profiles: dict[str, FileProfile],
) -> tuple[str, ...]:
    """
    Derive structural failure concepts dynamically from the incident's decomposed
    exception/message terms, stack-frame file stems, and AST-discovered resource tags.
    Zero hardcoded domain lists.
    """
    concepts: list[str] = []
    for res in resources:
        if res not in concepts:
            concepts.append(res)

    for path in files:
        stem = Path(path).stem.lower()
        if stem and stem != "__init__":
            tag = f"module:{stem}"
            if tag not in concepts:
                concepts.append(tag)
        prof = profiles.get(path)
        if prof is not None:
            for sym in (getattr(prof, "functions", []) or [])[:8]:
                sym_name = getattr(sym, "name", str(sym))
                for part in _decompose_identifier(sym_name):
                    if len(part) >= 4:
                        st = f"term:{_light_stem(part)}"
                        if st not in concepts:
                            concepts.append(st)

    for tok in _decompose_identifier(text):
        if len(tok) >= 4:
            st = f"term:{_light_stem(tok)}"
            if st not in concepts:
                concepts.append(st)

    return tuple(concepts[:24])


def _build_dense_semantic_vector(
    text: str,
    files: Iterable[str],
    resources: Iterable[str],
    profiles: Optional[dict[str, FileProfile]] = None,
) -> dict[str, float]:
    """
    Build a domain-neutral sparse-dense hybrid vector combining:
      1. Decomposed morphological root stems from the incident text & exception
      2. AST code-level vocabulary (symbols, called functions, config keys, I/O calls)
         from the incident's dependency cone so upstream callers and downstream
         callees share code-vocabulary dimensions even when error messages differ
      3. Character 3-gram & 4-gram subword embeddings for morphological invariance
    """
    vec: dict[str, float] = defaultdict(float)
    file_list = list(files)
    res_list = list(resources)

    # 1. Incident text & decomposed identifiers
    text_tokens = _tokenize(text)
    for tok in text_tokens:
        stem = _light_stem(tok)
        vec[f"stem:{stem}"] += 2.0

    # 2. AST code-level vocabulary from mapped files and dependency cone
    code_tokens: list[str] = []
    for path in file_list:
        code_tokens.extend(_decompose_identifier(path))
        if profiles and path in profiles:
            prof = profiles[path]
            for call_name in (getattr(prof, "calls", []) or [])[:25]:
                code_tokens.extend(_decompose_identifier(str(call_name)))
            for fn in (getattr(prof, "functions", []) or [])[:20]:
                fn_name = getattr(fn, "name", str(fn))
                code_tokens.extend(_decompose_identifier(fn_name))
            for cfg in (getattr(prof, "config_keys", []) or [])[:15]:
                code_tokens.extend(_decompose_identifier(str(cfg)))

    for r in res_list:
        vec[f"res:{r.lower()}"] += 3.0
        code_tokens.extend(_decompose_identifier(r))

    for tok in code_tokens:
        stem = _light_stem(tok)
        vec[f"ast:{stem}"] += 1.5
        vec[f"stem:{stem}"] += 1.0

    # 3. Character 3-grams and 4-grams over unique stems
    unique_stems = sorted({ _light_stem(t) for t in (*text_tokens, *code_tokens) if len(t) >= 3 })
    for tok in unique_stems:
        padded = f"^{tok}$"
        for n in (3, 4):
            for i in range(max(0, len(padded) - n + 1)):
                ngram = padded[i : i + n]
                vec[f"ng{n}:{ngram}"] += 0.25

    return dict(vec)


def _cosine_similarity(v1: dict[str, float], v2: dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    norm1 = math.sqrt(sum(val * val for val in v1.values()))
    norm2 = math.sqrt(sum(val * val for val in v2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _bm25_scores(
    query_tokens: list[str],
    documents: list[list[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not documents or not query_tokens:
        return [0.0 for _ in documents]
    n_docs = len(documents)
    avgdl = sum(len(doc) for doc in documents) / max(n_docs, 1)
    df: Counter[str] = Counter()
    for doc in documents:
        for tok in set(doc):
            df[tok] += 1

    scores: list[float] = []
    q_unique = list(dict.fromkeys(query_tokens))
    for doc in documents:
        tf = Counter(doc)
        doc_len = len(doc)
        score = 0.0
        for q in q_unique:
            if tf[q] == 0:
                continue
            idf = math.log(1.0 + (n_docs - df[q] + 0.5) / (df[q] + 0.5))
            denom = tf[q] + k1 * (1.0 - b + b * (doc_len / max(avgdl, 1.0)))
            score += idf * ((tf[q] * (k1 + 1.0)) / max(denom, 1e-9))
        scores.append(score)
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Layer B — Topology Mapping & Personalized PageRank (PPR)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_timestamp(raw_ts: str) -> Optional[float]:
    if not raw_ts or not raw_ts.strip():
        return None
    cleaned = raw_ts.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        try:
            return float(cleaned)
        except ValueError:
            return None


def _reachable_dependency_cone(
    graph: DependencyGraph,
    start_files: Iterable[str],
    max_hops: int = 3,
) -> dict[str, int]:
    cone: dict[str, int] = {}
    for start in start_files:
        if start not in cone:
            cone[start] = 0
        frontier = [start]
        for hop in range(1, max_hops + 1):
            next_frontier: list[str] = []
            for node in frontier:
                for dep in graph.dependencies(node):
                    if dep not in cone or hop < cone[dep]:
                        cone[dep] = hop
                        next_frontier.append(dep)
            frontier = next_frontier
    return cone


def _extract_shared_resources(
    cone_files: dict[str, int],
    profiles: dict[str, FileProfile],
    stack_trace: str,
) -> tuple[str, ...]:
    """
    Extract shared resources & architectural surfaces purely from the codebase's
    AST FileProfiles (db_interactions, service_calls, config_keys, api_endpoints,
    and cross-module invoked symbols) and stack frame functions.
    """
    resources: list[str] = []

    # 1. Collect symbols invoked by upstream files in the cone or present in stack frames
    invoked_calls: set[str] = set()
    for frame in extract_stack_frames(stack_trace):
        if frame.function and not frame.function.startswith("<"):
            invoked_calls.add(frame.function.split(".")[-1])
    for path in cone_files:
        prof = profiles.get(path)
        if prof is not None:
            for call_name in getattr(prof, "calls", []) or []:
                invoked_calls.add(str(call_name).split(".")[-1])

    # 2. Extract AST-discovered resources ordered by downstream hop depth (deepest dependencies first)
    ordered_files = sorted(cone_files.keys(), key=lambda p: (-cone_files[p], p))
    for path in ordered_files:
        prof = profiles.get(path)
        if prof is None:
            continue
        for db in getattr(prof, "db_interactions", []) or []:
            tag = f"db:{db.lower()}"
            if tag not in resources:
                resources.append(tag)
        for svc in getattr(prof, "service_calls", []) or []:
            tag = f"service:{svc.lower()}"
            if tag not in resources:
                resources.append(tag)
        for cfg in getattr(prof, "config_keys", []) or []:
            tag = f"config:{cfg}"
            if tag not in resources:
                resources.append(tag)
        for ep in getattr(prof, "api_endpoints", []) or []:
            tag = f"endpoint:{ep}"
            if tag not in resources:
                resources.append(tag)

        # Shared downstream symbols defined in this file and invoked in the call chain
        for fn in getattr(prof, "functions", []) or []:
            fn_name = getattr(fn, "name", str(fn))
            if fn_name in invoked_calls and not fn_name.startswith("_"):
                tag = f"symbol:{path}:{fn_name}"
                if tag not in resources:
                    resources.append(tag)

    return tuple(resources)


def compute_personalized_pagerank(
    adjacency: dict[str, dict[str, float]],
    seed_weights: dict[str, float],
    *,
    alpha: float = 0.85,
    iterations: int = 25,
) -> dict[str, float]:
    """
    Compute Personalized PageRank (PPR) over a weighted directed graph given a
    non-negative seed teleportation distribution.
    """
    all_nodes: set[str] = set(seed_weights)
    for src, targets in adjacency.items():
        all_nodes.add(src)
        all_nodes.update(targets)
    if not all_nodes:
        return {}

    total_seed = sum(max(0.0, w) for w in seed_weights.values())
    if total_seed <= 0.0:
        teleport = {n: 1.0 / len(all_nodes) for n in all_nodes}
    else:
        teleport = {n: max(0.0, seed_weights.get(n, 0.0)) / total_seed for n in all_nodes}

    scores = dict(teleport)
    out_sums = {
        src: sum(max(0.0, w) for w in targets.values())
        for src, targets in adjacency.items()
    }

    for _ in range(iterations):
        next_scores = {n: (1.0 - alpha) * teleport.get(n, 0.0) for n in all_nodes}
        dangling_mass = 0.0
        for src in all_nodes:
            src_score = scores.get(src, 0.0)
            total_out = out_sums.get(src, 0.0)
            if total_out <= 0.0:
                dangling_mass += alpha * src_score
            else:
                for dst, weight in adjacency.get(src, {}).items():
                    if weight > 0.0:
                        next_scores[dst] += alpha * src_score * (weight / total_out)
        if dangling_mass > 0.0:
            for n in all_nodes:
                next_scores[n] += dangling_mass * teleport.get(n, 0.0)
        scores = next_scores

    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Incident Loading & Normalization Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_incidents_from_inputs(paths: Sequence[str]) -> list[Incident]:
    """
    Load multiple Incidents from any combination of:
      - Standard ASRE incident JSON files
      - JSON arrays of incident objects
      - Multi-provider alert/telemetry payloads or raw log files
    """
    incidents: list[Incident] = []
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"incident file not found: {path!r}")
        raw_text = file_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            raise ValueError(f"incident file is empty: {path!r}")

        parsed_json: Any = None
        try:
            parsed_json = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed_json = None

        if isinstance(parsed_json, dict):
            if isinstance(parsed_json.get("incidents"), list):
                for item in parsed_json["incidents"]:
                    if isinstance(item, dict):
                        if "message" in item or "error" in item:
                            incidents.append(Incident.from_dict(item))
                        else:
                            for ev in normalize_telemetry_payload(item):
                                incidents.append(ev.to_incident())
            elif ("message" in parsed_json or "error" in parsed_json) and "events" not in parsed_json and "alerts" not in parsed_json:
                incidents.append(Incident.from_dict(parsed_json))
            else:
                for ev in normalize_telemetry_payload(parsed_json):
                    incidents.append(ev.to_incident())
        elif isinstance(parsed_json, list):
            for item in parsed_json:
                if isinstance(item, dict) and ("message" in item or "error" in item) and "source" not in item:
                    incidents.append(Incident.from_dict(item))
                else:
                    for ev in normalize_telemetry_payload(item):
                        incidents.append(ev.to_incident())
        else:
            for ev in load_telemetry_events(path):
                incidents.append(ev.to_incident())

    return incidents


def _normalize_incident_node(
    incident: Incident,
    graph: DependencyGraph,
    profiles: dict[str, FileProfile],
    repo_root: str,
    miner: DrainTemplateMiner,
) -> NormalizedIncidentNode:
    available_paths = set(profiles)
    mapped_files: list[str] = []

    for frame in extract_stack_frames(incident.stack_trace):
        res = _match_repository_path_detailed(
            frame.path,
            repo_root,
            available_paths,
            frame.function,
        )
        if res.mapped_path and res.mapped_path not in mapped_files:
            mapped_files.append(res.mapped_path)

    file_hint = str(incident.metadata.get("file_hint") or incident.metadata.get("file") or "")
    if file_hint:
        res = _match_repository_path_detailed(file_hint, repo_root, available_paths, "")
        if res.mapped_path and res.mapped_path not in mapped_files:
            mapped_files.append(res.mapped_path)

    if not mapped_files:
        combined_text = f"{incident.message}\n{incident.stack_trace}"
        for path in sorted(available_paths):
            base = Path(path).name
            stem = Path(path).stem
            if path in combined_text or base in combined_text or (len(stem) >= 4 and re.search(rf"\b{re.escape(stem)}\b", combined_text, re.IGNORECASE)):
                mapped_files.append(path)

    cone = _reachable_dependency_cone(graph, mapped_files, max_hops=3)
    combined_for_sem = f"{incident.exception_type} {incident.message} {incident.stack_trace}"
    resources = _extract_shared_resources(cone, profiles, incident.stack_trace)
    concepts = _extract_structural_concepts(combined_for_sem, cone.keys(), resources, profiles)
    drain_tpl = miner.extract_template(f"{incident.exception_type}: {incident.message}" if incident.exception_type else incident.message)

    service = str(
        incident.environment.get("service")
        or incident.metadata.get("service")
        or (Path(mapped_files[0]).parts[0] if mapped_files else "application")
    )
    timestamp = str(
        incident.metadata.get("timestamp")
        or incident.environment.get("timestamp")
        or ""
    )
    trace_id = str(
        incident.metadata.get("trace_id")
        or incident.environment.get("trace_id")
        or ""
    )
    span_id = str(
        incident.metadata.get("span_id")
        or incident.environment.get("span_id")
        or ""
    )
    parent_span_id = str(
        incident.metadata.get("parent_span_id")
        or incident.environment.get("parent_span_id")
        or ""
    )
    outbound_calls: list[str] = []
    inbound_eps: list[str] = []
    for f_path in mapped_files:
        prof = profiles.get(f_path)
        if prof is not None:
            for sc in getattr(prof, "service_calls", []) or []:
                if sc not in outbound_calls:
                    outbound_calls.append(str(sc))
            for ep in getattr(prof, "api_endpoints", []) or []:
                if ep not in inbound_eps:
                    inbound_eps.append(str(ep))
        # Also scan file text for explicit HTTP/gRPC route or service URLs if readable
        full_p = Path(repo_root) / f_path
        if full_p.is_file():
            try:
                src_txt = full_p.read_text(encoding="utf-8", errors="ignore")
                for url_m in re.findall(r"(?:https?|grpc)://[A-Za-z0-9_.:/-]+|/[A-Za-z0-9_/-]{3,64}", src_txt):
                    if url_m not in outbound_calls:
                        outbound_calls.append(url_m)
            except OSError:
                pass
    for meta_call in incident.metadata.get("service_calls") or ():
        if str(meta_call) not in outbound_calls:
            outbound_calls.append(str(meta_call))
    for meta_ep in incident.metadata.get("api_endpoints") or ():
        if str(meta_ep) not in inbound_eps:
            inbound_eps.append(str(meta_ep))

    title = f"{incident.exception_type}: {incident.message[:80]}" if incident.exception_type else incident.message[:80]

    return NormalizedIncidentNode(
        incident_id=incident.incident_id,
        title=title,
        service=service,
        exception_type=incident.exception_type or "Error",
        message=incident.message,
        drain_template=drain_tpl,
        timestamp=timestamp,
        epoch_seconds=_parse_timestamp(timestamp),
        trace_id=trace_id,
        mapped_files=tuple(mapped_files),
        dependency_cone=cone,
        shared_resources=resources,
        concepts=concepts,
        raw_incident=incident,
        span_id=span_id,
        parent_span_id=parent_span_id,
        outbound_service_calls=tuple(outbound_calls[:20]),
        inbound_api_endpoints=tuple(inbound_eps[:20]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Layer C.2 — Graph-Theoretic Topological Sink & Temporal Causal Discovery
# ──────────────────────────────────────────────────────────────────────────────

def _topological_origin_score(
    node: NormalizedIncidentNode,
    members: Sequence[NormalizedIncidentNode],
    root_cause_file: Optional[str],
) -> float:
    """
    Domain-neutral topological score measuring how foundational an incident is
    within a correlated cluster:
      1. Direct stack-frame hit on the cluster's Lowest Common Ancestor (`root_cause_file`)
      2. Number of other incidents in the cluster whose dependency cones depend on
         this incident's mapped files (downstream sink in-degree)
      3. Inverse of how many other cluster incidents' mapped files this incident
         depends on (fewer external cluster dependencies = deeper foundation)
    """
    score = 0.0
    node_mapped = set(node.mapped_files)
    if root_cause_file and root_cause_file in node_mapped:
        score += 5.0

    # How many other incidents depend on `node`'s mapped files?
    dependents_count = sum(
        1
        for other in members
        if other.incident_id != node.incident_id and bool(node_mapped & set(other.dependency_cone))
    )
    score += dependents_count * 3.0

    # How many other incidents' direct files does `node` depend on at hop > 0?
    upstream_deps_count = sum(
        1
        for other in members
        if other.incident_id != node.incident_id
        and any(node.dependency_cone.get(f, 0) > 0 for f in other.mapped_files)
    )
    score -= upstream_deps_count * 1.5
    return score


def _detect_direct_call_or_dependency(
    caller: NormalizedIncidentNode,
    callee: NormalizedIncidentNode,
    root_cause_file: Optional[str],
) -> tuple[bool, float, list[str]]:
    """
    Stage 1 (Multi-Modal Service Topology): Determine if `caller` directly invokes
    or depends on `callee` at 1 hop, which implies faults in `callee` propagate
    causally upward to `caller` (`callee --causes--> caller`).
    Checks:
      1. OpenTelemetry span tree (`callee.parent_span_id == caller.span_id`)
      2. Cross-service HTTP/gRPC endpoint or service hostname call (`caller.outbound_service_calls` -> `callee.inbound_api_endpoints` / `callee.service`)
      3. Direct 1-hop AST module import (`callee.mapped_files` at hop 1 in `caller.dependency_cone`)
    """
    if caller.incident_id == callee.incident_id:
        return False, 0.0, []

    reasons: list[str] = []
    boost = 0.0
    is_direct = False

    # 1. OpenTelemetry Parent-Child Span Call (caller span spawned callee span)
    if caller.span_id and callee.parent_span_id and caller.span_id == callee.parent_span_id:
        is_direct = True
        boost += 0.30
        reasons.append(
            f"OTel span causality: '{caller.service}' (span {caller.span_id}) called '{callee.service}' (child span {callee.span_id})"
        )

    # 2. Cross-Service HTTP / gRPC / RPC Endpoint & Service Name Matching
    if caller.outbound_service_calls:
        callee_svc_norm = callee.service.lower().strip()
        for out_call in caller.outbound_service_calls:
            out_lower = out_call.lower()
            # Match against callee's inbound API endpoints
            matched_ep = next(
                (ep for ep in callee.inbound_api_endpoints if ep and len(ep) >= 3 and ep.lower() in out_lower),
                None,
            )
            if matched_ep:
                is_direct = True
                boost += 0.25
                reasons.append(
                    f"Cross-service RPC/API call: '{caller.service}' invokes endpoint '{matched_ep}' on '{callee.service}'"
                )
                break
            if (
                callee_svc_norm
                and callee_svc_norm != caller.service.lower()
                and len(callee_svc_norm) >= 3
                and callee_svc_norm in out_lower
            ):
                is_direct = True
                boost += 0.24
                reasons.append(
                    f"Cross-service RPC call: '{caller.service}' calls '{out_call}' targeting '{callee.service}'"
                )
                break

    # 3. Direct 1-hop AST dependency in the import graph
    callee_files = set(callee.mapped_files)
    direct_hop1 = [
        f for f in callee_files
        if caller.dependency_cone.get(f, -1) == 1
        or (
            caller.dependency_cone.get(f, -1) == 0
            and f == root_cause_file
            and set(caller.mapped_files) != callee_files
        )
    ]
    if direct_hop1:
        is_direct = True
        boost += 0.20
        reasons.append(
            f"'{caller.service}' depends on '{direct_hop1[0]}' (1-hop downstream dependency)"
        )

    return is_direct, boost, reasons


def _compute_granger_lag_score(
    cause: NormalizedIncidentNode,
    effect: NormalizedIncidentNode,
    temporal_window_seconds: float = 900.0,
) -> tuple[float, float]:
    """
    Stage 2 (Granger-Style Lagged Temporal Cross-Correlation):
    Compute temporal onset lag (delta_t = t_effect - t_cause) and an exponential
    impulse-response decay score G(cause -> effect) in [0.0, 1.0].
    """
    if cause.epoch_seconds is None or effect.epoch_seconds is None:
        return 0.0, 0.50
    lag = effect.epoch_seconds - cause.epoch_seconds
    if lag < 0.0:
        # Allow minor clock skew only when explicit OTel child->parent span exists
        if effect.span_id and cause.parent_span_id == effect.span_id and abs(lag) <= 5.0:
            return 0.0, 0.85
        return lag, 0.0
    if lag > temporal_window_seconds:
        return lag, 0.05
    decay = math.exp(-lag / max(60.0, temporal_window_seconds * 0.25))
    score = min(1.0, 0.55 + 0.45 * decay)
    return lag, score


def _compute_temporal_causal_chain(
    members: Sequence[NormalizedIncidentNode],
    root_cause_file: Optional[str],
    temporal_window_seconds: float = 900.0,
) -> tuple[
    str,
    tuple[TemporalCausalLink, ...],
    tuple[ServiceCausalAttribution, ...],
    tuple[tuple[str, ...], ...],
]:
    """
    4-Stage Cross-Service Causal Discovery & Counterfactual Attribution:
      1. Multi-modal service topology (OTel span parent-child + RPC endpoints + AST imports)
      2. Granger-style lagged temporal correlation
      3. PC-Algorithm skeleton orientation & transitive reduction (d-separation pruning
         to reconstruct multi-hop cascades like Service D -> Service C -> Service B -> Service A)
      4. Counterfactual Do-Calculus root-cause attribution ranking
    """
    if not members:
        return "", (), (), ()
    if len(members) == 1:
        m0 = members[0]
        attr = ServiceCausalAttribution(
            service=m0.service,
            incident_id=m0.incident_id,
            counterfactual_resolution_pct=100.0,
            downstream_incidents_caused=(),
            downstream_services_affected=(m0.service,),
            causal_propagation_path=(f"{m0.service} ({m0.incident_id})",),
            granger_lag_score=1.0,
        )
        return m0.incident_id, (), (attr,), ((f"{m0.service} ({m0.incident_id})",),)

    has_timestamps = all(m.epoch_seconds is not None for m in members)
    min_epoch = min((m.epoch_seconds for m in members if m.epoch_seconds is not None), default=0.0)

    # Compute node priority (earlier onset + deeper topological sink + OTel leaf depth)
    span_children_count: Counter[str] = Counter()
    for m in members:
        if m.parent_span_id:
            for parent_cand in members:
                if parent_cand.span_id and parent_cand.span_id == m.parent_span_id:
                    span_children_count[parent_cand.incident_id] += 1

    def _trigger_priority(m: NormalizedIncidentNode) -> tuple[float, float, str]:
        topo = _topological_origin_score(m, members, root_cause_file)
        # In OTel span trees, leaf spans (0 children, has parent_span_id) are deeper callees
        otel_depth_bonus = 3.0 if (m.parent_span_id and span_children_count[m.incident_id] == 0) else 0.0
        if m.epoch_seconds is not None:
            lag_from_first = max(0.0, m.epoch_seconds - min_epoch)
            time_bonus = max(0.0, 10.0 - (lag_from_first / max(temporal_window_seconds, 1.0)) * 10.0)
        else:
            time_bonus = 0.0
        return (topo + otel_depth_bonus + time_bonus, -float(m.epoch_seconds or 0.0), m.incident_id)

    ordered = sorted(members, key=_trigger_priority, reverse=True)
    rank_index = {m.incident_id: idx for idx, m in enumerate(ordered)}
    node_by_id = {m.incident_id: m for m in ordered}

    # Step 1: Discover all direct 1-hop service/module causal edges (u --causes--> v)
    # where `v` directly invokes/imports `u` (via OTel span child, RPC endpoint, or 1-hop AST import)
    direct_edges: dict[tuple[str, str], TemporalCausalLink] = {}
    has_incoming_direct: set[str] = set()
    has_outgoing_direct: set[str] = set()

    for u in ordered:
        for v in ordered:
            if u.incident_id == v.incident_id:
                continue
            is_direct, direct_boost, mechanisms = _detect_direct_call_or_dependency(
                caller=v,
                callee=u,
                root_cause_file=root_cause_file,
            )
            if not is_direct:
                continue
            lag, granger = _compute_granger_lag_score(u, v, temporal_window_seconds)
            if has_timestamps and lag < 0.0 and granger <= 0.0:
                continue
            # Prevent 2-cycles if two modules mutually reference each other; orient by onset time / priority
            if (v.incident_id, u.incident_id) in direct_edges:
                continue

            shared_res = set(u.shared_resources) & set(v.shared_resources)
            if shared_res:
                mechanisms.append(f"shared contention on {', '.join(sorted(shared_res)[:4])}")
            if has_timestamps and lag >= 0.0:
                mechanisms.append(f"onset lag +{lag:.1f}s after '{u.incident_id}'")

            conf = min(
                0.99,
                0.65
                + direct_boost
                + (0.08 if shared_res else 0.0)
                + (0.10 * granger if has_timestamps else 0.05),
            )
            direct_edges[(u.incident_id, v.incident_id)] = TemporalCausalLink(
                cause_incident_id=u.incident_id,
                effect_incident_id=v.incident_id,
                lag_seconds=max(0.0, lag),
                causal_confidence=conf,
                mechanism="; ".join(mechanisms),
                cause_service=u.service,
                effect_service=v.service,
                granger_score=granger,
            )
            has_incoming_direct.add(v.incident_id)
            has_outgoing_direct.add(u.incident_id)

    # Identify root_trigger: prefer a node with outgoing direct causal edges and 0 incoming direct causal edges
    root_candidates = [
        m for m in ordered
        if m.incident_id not in has_incoming_direct and m.incident_id in has_outgoing_direct
    ]
    root_trigger = root_candidates[0] if root_candidates else ordered[0]

    # Step 2: PC-Style Transitive Reduction over direct_edges
    # Prune (u -> v) if there is an intermediate w such that (u -> w) and (w -> v) are both in direct_edges
    pruned_keys: set[tuple[str, str]] = set()
    for u_id, v_id in list(direct_edges.keys()):
        for w in ordered:
            w_id = w.incident_id
            if w_id not in {u_id, v_id} and (u_id, w_id) in direct_edges and (w_id, v_id) in direct_edges:
                pruned_keys.add((u_id, v_id))
                break

    final_links: list[TemporalCausalLink] = [
        link for key, link in direct_edges.items() if key not in pruned_keys
    ]

    # Step 3: For any member node v != root_trigger that is not yet reached by the direct causal DAG,
    # connect root_trigger -> v via shared downstream dependency / resource / temporal lag
    reached_in_dag: set[str] = {root_trigger.incident_id}
    dag_adj: dict[str, list[str]] = defaultdict(list)
    for link in final_links:
        dag_adj[link.cause_incident_id].append(link.effect_incident_id)
    bfs_q = [root_trigger.incident_id]
    while bfs_q:
        curr = bfs_q.pop(0)
        for nxt in dag_adj.get(curr, []):
            if nxt not in reached_in_dag:
                reached_in_dag.add(nxt)
                bfs_q.append(nxt)

    for v in ordered:
        if v.incident_id in reached_in_dag:
            continue
        lag, granger = _compute_granger_lag_score(root_trigger, v, temporal_window_seconds)
        mechanisms: list[str] = []
        conf = 0.65

        shared_downstreams = set(v.dependency_cone) & (
            set(root_trigger.mapped_files) | ({root_cause_file} if root_cause_file else set())
        )
        if shared_downstreams:
            dep_file = sorted(shared_downstreams)[0]
            hop = v.dependency_cone.get(dep_file, 1)
            mechanisms.append(f"'{v.service}' depends on '{dep_file}' ({hop}-hop downstream dependency)")
            conf += 0.18

        shared_res = set(root_trigger.shared_resources) & set(v.shared_resources)
        if shared_res:
            mechanisms.append(f"shared contention on {', '.join(sorted(shared_res)[:4])}")
            conf += 0.12

        if has_timestamps and lag >= 0.0:
            mechanisms.append(f"onset lag +{lag:.1f}s after '{root_trigger.incident_id}'")
            conf += 0.07

        if not mechanisms:
            mechanisms.append(f"shared root-cause module '{root_cause_file or 'common dependency'}'")

        final_links.append(
            TemporalCausalLink(
                cause_incident_id=root_trigger.incident_id,
                effect_incident_id=v.incident_id,
                lag_seconds=max(0.0, lag),
                causal_confidence=min(0.99, conf),
                mechanism="; ".join(mechanisms),
                cause_service=root_trigger.service,
                effect_service=v.service,
                granger_score=granger,
            )
        )

    final_links.sort(
        key=lambda l: (
            node_by_id[l.cause_incident_id].epoch_seconds or 0.0,
            l.lag_seconds,
            rank_index.get(l.cause_incident_id, 0),
            l.effect_incident_id,
        )
    )

    # Stage 4: Counterfactual Root-Cause Attribution & Multi-Hop Service Propagation Paths
    adj: dict[str, list[TemporalCausalLink]] = defaultdict(list)
    in_degree: Counter[str] = Counter()
    for link in final_links:
        adj[link.cause_incident_id].append(link)
        in_degree[link.effect_incident_id] += 1

    total_nodes = len(ordered)
    attributions: list[ServiceCausalAttribution] = []
    all_paths: list[tuple[str, ...]] = []

    def _longest_path_from(start_id: str, visited: set[str]) -> tuple[str, ...]:
        node_obj = node_by_id[start_id]
        label = f"{node_obj.service} ({start_id})"
        best_sub: tuple[str, ...] = ()
        for edge in adj.get(start_id, []):
            nxt = edge.effect_incident_id
            if nxt not in visited:
                sub = _longest_path_from(nxt, visited | {nxt})
                if len(sub) > len(best_sub):
                    best_sub = sub
        return (label, *best_sub)

    for m in ordered:
        # BFS to find all downstream incidents causally reachable from `m`
        reached: list[str] = []
        queue = [m.incident_id]
        seen = {m.incident_id}
        granger_vals: list[float] = []
        while queue:
            curr = queue.pop(0)
            for edge in adj.get(curr, []):
                granger_vals.append(edge.granger_score)
                if edge.effect_incident_id not in seen:
                    seen.add(edge.effect_incident_id)
                    reached.append(edge.effect_incident_id)
                    queue.append(edge.effect_incident_id)

        # Counterfactual resolution %: fixing `m` prevents `m` + all causally downstream incidents
        resolved_count = 1 + len(reached)
        cf_pct = (resolved_count / max(total_nodes, 1)) * 100.0
        downstream_svcs = tuple(
            dict.fromkeys([m.service, *(node_by_id[r].service for r in reached)])
        )
        prop_path = _longest_path_from(m.incident_id, {m.incident_id})
        if len(prop_path) > 1 and prop_path not in all_paths:
            all_paths.append(prop_path)

        avg_granger = sum(granger_vals) / len(granger_vals) if granger_vals else 1.0
        attributions.append(
            ServiceCausalAttribution(
                service=m.service,
                incident_id=m.incident_id,
                counterfactual_resolution_pct=cf_pct,
                downstream_incidents_caused=tuple(reached),
                downstream_services_affected=downstream_svcs,
                causal_propagation_path=prop_path,
                granger_lag_score=avg_granger,
            )
        )

    attributions.sort(
        key=lambda a: (-a.counterfactual_resolution_pct, rank_index.get(a.incident_id, 0))
    )
    return root_trigger.incident_id, tuple(final_links), tuple(attributions), tuple(all_paths)


# ──────────────────────────────────────────────────────────────────────────────
# Layer A.2 — Historical Hybrid Graph-RAG with Reciprocal Rank Fusion (RRF)
# ──────────────────────────────────────────────────────────────────────────────

def _retrieve_historical_comparisons_rrf(
    active_nodes: Sequence[NormalizedIncidentNode],
    shared_files: Sequence[str],
    shared_resources: Sequence[str],
    profiles: dict[str, FileProfile],
    repair_run_dir: Optional[str],
    repo_root: str,
    *,
    top_k: int = 5,
    rrf_k: int = 60,
) -> tuple[HistoricalIncidentComparison, ...]:
    """
    Query past incidents in `.asre/runs` using domain-neutral Hybrid Graph-RAG:
      Ranker 1: BM25 Sparse Keyword Search
      Ranker 2: AST-Grounded Subword & Morphological Stem Cosine Similarity
      Ranker 3: AST Dependency-Cone & File Structural Jaccard Similarity
    Fused via Reciprocal Rank Fusion: RRF(d) = sum(1 / (rrf_k + rank_m(d))).
    """
    candidates_dirs: list[Path] = []
    if repair_run_dir:
        p = Path(repair_run_dir)
        if p.is_dir():
            candidates_dirs.append(p)
    default_runs = Path(repo_root) / ".asre" / "runs"
    if default_runs.is_dir() and default_runs not in candidates_dirs:
        candidates_dirs.append(default_runs)

    active_ids = {n.incident_id for n in active_nodes}
    historical_records: list[dict[str, Any]] = []
    seen_hist_ids: set[str] = set()

    for run_root in candidates_dirs:
        for child in sorted(run_root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            run_file = child / "run.json"
            inc_file = child / "incident.json"
            ctx_file = child / "repair-context.json"
            if not run_file.is_file():
                continue
            try:
                run_data = json.loads(run_file.read_text(encoding="utf-8"))
                inc_data = json.loads(inc_file.read_text(encoding="utf-8")) if inc_file.is_file() else {}
                ctx_data = json.loads(ctx_file.read_text(encoding="utf-8")) if ctx_file.is_file() else {}
            except (OSError, json.JSONDecodeError):
                continue

            hist_id = str(run_data.get("incident_id") or inc_data.get("incident_id") or child.name)
            if hist_id in active_ids or hist_id in seen_hist_ids:
                continue
            seen_hist_ids.add(hist_id)

            lesson = run_data.get("lesson") or {}
            exc_type = str(lesson.get("exception_type") or inc_data.get("exception_type") or "Error")
            message = str(lesson.get("message") or inc_data.get("message") or "")
            diagnosis = str(lesson.get("diagnosis") or "")
            files_mod = list(lesson.get("files_modified") or [])
            evidence_files = [
                str(ev.get("path"))
                for ev in (ctx_data.get("evidence") or [])
                if isinstance(ev, dict) and ev.get("path")
            ]
            all_hist_files = list(dict.fromkeys(files_mod + evidence_files))
            historical_records.append(
                {
                    "incident_id": hist_id,
                    "status": str(run_data.get("status") or "unknown"),
                    "exception_type": exc_type,
                    "message": message,
                    "diagnosis": diagnosis,
                    "files": all_hist_files,
                    "tests": list(lesson.get("tests_to_run") or []),
                }
            )

    if not historical_records:
        return ()

    query_text = " ".join(f"{n.exception_type} {n.message}" for n in active_nodes)
    query_files = (
        set(shared_files)
        | {f for n in active_nodes for f in n.mapped_files}
        | {f for n in active_nodes for f in n.dependency_cone}
    )
    query_tokens = _tokenize(f"{query_text} {' '.join(query_files)}")
    query_vec = _build_dense_semantic_vector(query_text, query_files, shared_resources, profiles)
    query_concepts = set(_extract_structural_concepts(query_text, query_files, shared_resources, profiles))

    doc_tokens_list = [
        _tokenize(f"{rec['exception_type']} {rec['message']} {rec['diagnosis']} {' '.join(rec['files'])}")
        for rec in historical_records
    ]
    bm25_raw = _bm25_scores(query_tokens, doc_tokens_list)

    sem_raw: list[float] = []
    graph_raw: list[float] = []

    for rec in historical_records:
        rec_text = f"{rec['exception_type']} {rec['message']} {rec['diagnosis']}"
        rec_resources = _extract_shared_resources({f: 0 for f in rec["files"]}, profiles, "")
        rec_vec = _build_dense_semantic_vector(rec_text, rec["files"], rec_resources, profiles)
        sem_raw.append(_cosine_similarity(query_vec, rec_vec))

        rec_files = set(rec["files"])
        if not query_files or not rec_files:
            graph_raw.append(0.0)
        else:
            inter = len(query_files & rec_files)
            union = len(query_files | rec_files)
            graph_raw.append(inter / max(union, 1))

    n_recs = len(historical_records)
    bm25_order = sorted(range(n_recs), key=lambda i: (-bm25_raw[i], i))
    sem_order = sorted(range(n_recs), key=lambda i: (-sem_raw[i], i))
    graph_order = sorted(range(n_recs), key=lambda i: (-graph_raw[i], i))

    bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_order, start=1)}
    sem_ranks = {idx: rank for rank, idx in enumerate(sem_order, start=1)}
    graph_ranks = {idx: rank for rank, idx in enumerate(graph_order, start=1)}

    max_possible_rrf = 3.0 / (rrf_k + 1.0)
    comparisons: list[HistoricalIncidentComparison] = []

    for idx, rec in enumerate(historical_records):
        if bm25_raw[idx] <= 0.0 and sem_raw[idx] <= 0.05 and graph_raw[idx] <= 0.0:
            continue
        rrf_score = (
            1.0 / (rrf_k + bm25_ranks[idx])
            + 1.0 / (rrf_k + sem_ranks[idx])
            + 1.0 / (rrf_k + graph_ranks[idx])
        )
        direct_sim = 0.45 * sem_raw[idx] + 0.35 * graph_raw[idx] + 0.20 * min(1.0, bm25_raw[idx] / 10.0)
        similarity_pct = min(99.0, max(10.0, (0.5 * (rrf_score / max_possible_rrf) + 0.5 * direct_sim) * 100.0))

        rec_resources = _extract_shared_resources({f: 0 for f in rec["files"]}, profiles, "")
        rec_concepts = set(
            _extract_structural_concepts(
                f"{rec['exception_type']} {rec['message']} {rec['diagnosis']}",
                rec["files"],
                rec_resources,
                profiles,
            )
        )
        shared_files_match = tuple(sorted(query_files & set(rec["files"])))
        shared_concepts_match = tuple(sorted(query_concepts & rec_concepts))
        lesson_str = (
            f"Prior {rec['status']} fix modified {', '.join(rec['files'][:3]) or 'repository'}"
            + (f" and verified with {', '.join(rec['tests'][:2])}" if rec["tests"] else "")
        )

        comparisons.append(
            HistoricalIncidentComparison(
                historical_incident_id=rec["incident_id"],
                historical_status=rec["status"],
                rrf_score=rrf_score,
                similarity_pct=similarity_pct,
                bm25_rank=bm25_ranks[idx],
                semantic_rank=sem_ranks[idx],
                graph_rank=graph_ranks[idx],
                shared_files=shared_files_match,
                shared_concepts=shared_concepts_match,
                historical_diagnosis=rec["diagnosis"] or rec["message"],
                reusable_lesson=lesson_str,
            )
        )

    comparisons.sort(key=lambda c: (-c.rrf_score, -c.similarity_pct, c.historical_incident_id))
    return tuple(comparisons[:top_k])


# ──────────────────────────────────────────────────────────────────────────────
# Optional LLM Hypothesis Synthesizer (Provider-Neutral)
# ──────────────────────────────────────────────────────────────────────────────

def _synthesize_hypothesis(
    member_nodes: Sequence[NormalizedIncidentNode],
    common_root_file: Optional[str],
    common_root_resource: Optional[str],
    root_trigger_id: str,
    best_file_ppr: float,
    causal_links: Sequence[TemporalCausalLink],
    llm_synthesizer: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Generate a domain-neutral root-cause hypothesis grounded in the topological LCA,
    shared AST resources, and temporal lead-lag evidence. If `llm_synthesizer` is
    provided, uses the LLM to refine the structured evidence into a concise explanation.
    """
    if len(member_nodes) == 1:
        m0 = member_nodes[0]
        return f"Single isolated incident '{m0.incident_id}' in '{common_root_file or m0.service}': {m0.message}"

    services_list = sorted({m.service for m in member_nodes})
    inc_titles = [f"{m.incident_id} ({m.service}: {m.exception_type})" for m in member_nodes]
    cause_target = common_root_file or common_root_resource or "shared downstream dependency"
    res_detail = f" and shared resource '{common_root_resource}'" if common_root_resource else ""
    cascade_detail = (
        f"triggered first by '{root_trigger_id}' and cascading upstream into {', '.join(services_list)}"
        if causal_links
        else f"affecting {', '.join(services_list)}"
    )
    deterministic_hypothesis = (
        f"All {len(member_nodes)} incidents ({', '.join(inc_titles)}) converge on a common underlying "
        f"root cause in '{cause_target}'{res_detail} (PPR={best_file_ppr:.3f}), {cascade_detail}."
    )

    if llm_synthesizer is not None:
        prompt = (
            "Synthesize a single concise root-cause hypothesis sentence explaining how these "
            f"{len(member_nodes)} correlated incidents share a common underlying cause:\n"
            f"- Common Root Module: {common_root_file}\n"
            f"- Shared AST Resource: {common_root_resource}\n"
            f"- Root Trigger Incident: {root_trigger_id}\n"
            f"- Incidents: {'; '.join(f'{m.incident_id} [{m.service}] {m.exception_type}: {m.message}' for m in member_nodes)}\n"
        )
        try:
            llm_text = llm_synthesizer(prompt).strip()
            if llm_text:
                return llm_text
        except Exception:
            pass

    return deterministic_hypothesis


# ──────────────────────────────────────────────────────────────────────────────
# Main Orchestrator — 3-Layer Cross-Incident Intelligence Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def analyze_cross_incident_intelligence(
    incidents: Sequence[Incident],
    graph: DependencyGraph,
    profiles: dict[str, FileProfile],
    repo_root: str,
    *,
    repair_run_dir: Optional[str] = None,
    temporal_window_seconds: float = 900.0,
    llm_synthesizer: Optional[Callable[[str], str]] = None,
) -> CrossIncidentIntelligenceReport:
    """
    Execute the domain-neutral 3-Layer Hybrid Graph-RAG + Temporal Causal
    Cross-Incident Intelligence pipeline across multiple incidents:
      1. Ingest & Drain3 template normalization + AST dependency cone & resource mapping
      2. Multi-signal correlation (Dependency Graph LCA + Shared AST Resources +
         AST-Grounded Subword Cosine Similarity + Distributed Trace + Sliding Time Window)
      3. Personalized PageRank (PPR) + Topological Sink root-cause localization &
         IncidentRelationshipGraph construction
      4. Temporal lead-lag causal cascade ordering & git commit change-point attribution
      5. Historical Hybrid Graph-RAG comparison via Reciprocal Rank Fusion (RRF)
    """
    miner = DrainTemplateMiner()
    nodes = [
        _normalize_incident_node(inc, graph, profiles, repo_root, miner)
        for inc in incidents
    ]
    if not nodes:
        return CrossIncidentIntelligenceReport(
            total_incidents_analyzed=0,
            correlated_root_causes=0,
            uncorrelated_incidents=0,
            correlated_groups=(),
            relationship_graph=IncidentRelationshipGraph((), ()),
            historical_comparisons=(),
            recurring_patterns=(),
        )

    dense_vectors = {
        n.incident_id: _build_dense_semantic_vector(
            f"{n.exception_type} {n.message}",
            n.dependency_cone.keys(),
            n.shared_resources,
            profiles,
        )
        for n in nodes
    }

    # ── Build Multi-Layer Topology Graph & Pairwise Correlation ────
    parent: dict[str, str] = {n.incident_id: n.incident_id for n in nodes}
    pair_evidence: dict[tuple[str, str], list[str]] = defaultdict(list)

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str, reason: str) -> None:
        ra, rb = find(a), find(b)
        key = (min(a, b), max(a, b))
        if reason not in pair_evidence[key]:
            pair_evidence[key].append(reason)
        if ra != rb:
            parent[rb] = ra

    graph_edges: list[IncidentGraphEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def _add_edge(src: str, dst: str, rel: str, weight: float, evidence: str) -> None:
        key = (src, dst, rel)
        if key not in seen_edges:
            seen_edges.add(key)
            graph_edges.append(IncidentGraphEdge(src, dst, rel, weight, evidence))

    ppr_adjacency: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ppr_seeds: dict[str, float] = defaultdict(float)

    for n in nodes:
        inc_node_id = f"inc:{n.incident_id}"
        ppr_seeds[inc_node_id] = 1.0
        for f_path in n.mapped_files:
            file_node_id = f"file:{f_path}"
            ppr_adjacency[inc_node_id][file_node_id] += 3.0
            ppr_seeds[file_node_id] += 2.0
            _add_edge(
                inc_node_id,
                file_node_id,
                "maps_to_file",
                3.0,
                f"Incident '{n.incident_id}' stack trace/hint maps to '{f_path}'",
            )

        for dep_path, hop in n.dependency_cone.items():
            if hop > 0:
                dep_node_id = f"file:{dep_path}"
                w = 2.0 / hop
                ppr_adjacency[inc_node_id][dep_node_id] += w
                for src_f in n.mapped_files:
                    if dep_path in graph.dependencies(src_f):
                        ppr_adjacency[f"file:{src_f}"][dep_node_id] += 2.5
                        _add_edge(
                            f"file:{src_f}",
                            dep_node_id,
                            "depends_on",
                            2.5,
                            f"'{src_f}' imports downstream dependency '{dep_path}'",
                        )

        for res in n.shared_resources:
            ppr_adjacency[inc_node_id][res] += 2.2
            _add_edge(
                inc_node_id,
                res,
                "uses_infrastructure",
                2.2,
                f"Incident '{n.incident_id}' interacts with AST resource '{res}'",
            )
            # Connect the resource node to the files in the dependency cone that define or invoke it
            for dep_path, hop in n.dependency_cone.items():
                ppr_adjacency[res][f"file:{dep_path}"] += 2.0 / (1.0 + hop)

    # Evaluate pairwise cross-incident relationships across 5 domain-neutral signals
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i + 1 :]:
            # Signal 1: Shared downstream dependency in the AST Dependency Graph (Layer B LCA)
            common_deps = set(n1.dependency_cone) & set(n2.dependency_cone)
            if common_deps:
                best_dep = max(
                    common_deps,
                    key=lambda p: (
                        1 if p in set(n1.mapped_files) | set(n2.mapped_files) else 0,
                        profiles[p].risk_score if p in profiles else 0,
                        -(n1.dependency_cone[p] + n2.dependency_cone[p]),
                    ),
                )
                union(
                    n1.incident_id,
                    n2.incident_id,
                    f"Shared dependency convergence: '{n1.incident_id}' ({n1.service}) and '{n2.incident_id}' ({n2.service}) both depend on '{best_dep}'",
                )
                _add_edge(
                    f"inc:{n1.incident_id}",
                    f"inc:{n2.incident_id}",
                    "shares_dependency",
                    2.8,
                    f"Both incidents converge on downstream module '{best_dep}'",
                )

            # Signal 2: Shared AST-discovered resource (shared symbol, DB interaction, service call, config key)
            common_res = set(n1.shared_resources) & set(n2.shared_resources)
            if common_res:
                best_res = sorted(common_res)[0]
                union(
                    n1.incident_id,
                    n2.incident_id,
                    f"Shared AST resource contention on '{best_res}' across '{n1.incident_id}' and '{n2.incident_id}'",
                )
                _add_edge(
                    f"inc:{n1.incident_id}",
                    f"inc:{n2.incident_id}",
                    "shares_infrastructure",
                    2.5,
                    f"Shared AST resource '{best_res}'",
                )

            # Signal 3: Distributed Trace & OpenTelemetry Span-Tree correlation
            if (n1.trace_id and n1.trace_id == n2.trace_id) or (
                n1.span_id and n1.span_id == n2.parent_span_id
            ) or (
                n2.span_id and n2.span_id == n1.parent_span_id
            ):
                trace_label = n1.trace_id or n2.trace_id or f"span:{n1.span_id or n2.span_id}"
                union(
                    n1.incident_id,
                    n2.incident_id,
                    f"Shared distributed trace/span tree '{trace_label}' linking '{n1.service}' and '{n2.service}'",
                )
                _add_edge(
                    f"inc:{n1.incident_id}",
                    f"inc:{n2.incident_id}",
                    "shared_trace",
                    3.0,
                    f"Shared distributed trace/span '{trace_label}'",
                )

            # Signal 3b: Cross-Service HTTP / gRPC Endpoint or Service Call Topology
            n1_calls_n2, _, r_12 = _detect_direct_call_or_dependency(n1, n2, None)
            n2_calls_n1, _, r_21 = _detect_direct_call_or_dependency(n2, n1, None)
            if n1_calls_n2 or n2_calls_n1:
                call_reason = (r_12 or r_21)[0]
                union(n1.incident_id, n2.incident_id, call_reason)

            # Signal 4: Drain3 Log Template equivalence (Recurring failure pattern)
            if n1.drain_template.template_id == n2.drain_template.template_id:
                union(
                    n1.incident_id,
                    n2.incident_id,
                    f"Identical Drain log template [{n1.drain_template.template_id}]: '{n1.drain_template.template}'",
                )

            # Signal 5: AST-Grounded Subword & Stem Cosine Similarity within Temporal Sliding Window
            within_window = True
            if n1.epoch_seconds is not None and n2.epoch_seconds is not None:
                within_window = abs(n1.epoch_seconds - n2.epoch_seconds) <= temporal_window_seconds
            sem_sim = _cosine_similarity(dense_vectors[n1.incident_id], dense_vectors[n2.incident_id])
            if within_window and sem_sim >= 0.35:
                union(
                    n1.incident_id,
                    n2.incident_id,
                    f"Temporal-semantic correlation (AST & subword cosine similarity {sem_sim * 100:.0f}%) within sliding window",
                )

    # ── Run Personalized PageRank (PPR) on the Multi-Layer Graph ───
    ppr_scores = compute_personalized_pagerank(ppr_adjacency, ppr_seeds)

    clusters_by_root: dict[str, list[NormalizedIncidentNode]] = defaultdict(list)
    for n in nodes:
        clusters_by_root[find(n.incident_id)].append(n)

    all_candidate_files = sorted({f for n in nodes for f in n.dependency_cone})
    commits_by_file = (
        get_recent_commits_for_files(repo_root, all_candidate_files, max_per_file=2)
        if all_candidate_files
        else {}
    )

    correlated_groups: list[CorrelatedIncidentGroup] = []
    for idx, (_, member_nodes) in enumerate(
        sorted(clusters_by_root.items(), key=lambda kv: (-len(kv[1]), kv[0])),
        start=1,
    ):
        member_ids = {m.incident_id for m in member_nodes}
        evidence_list: list[str] = []
        for (a, b), evs in pair_evidence.items():
            if a in member_ids and b in member_ids:
                for e in evs:
                    if e not in evidence_list:
                        evidence_list.append(e)

        # Rank candidate common root-cause files purely by graph topology:
        #   1. Multi-incident reachability (how many incidents in this cluster reach `f`)
        #   2. Topological sink depth (how many cluster incidents reach `f` at hop > 0 minus how many cluster files `f` depends on)
        #   3. Personalized PageRank score `ppr_scores["file:" + f]`
        #   4. File architectural risk score
        file_reach_count: Counter[str] = Counter()
        downstream_inbound_count: Counter[str] = Counter()
        for m in member_nodes:
            for dep_path, hop in m.dependency_cone.items():
                file_reach_count[dep_path] += 1
                if hop > 0:
                    downstream_inbound_count[dep_path] += 1

        common_root_file: Optional[str] = None
        best_file_ppr = 0.0
        if file_reach_count:
            cluster_files = set(file_reach_count)
            common_root_file = max(
                file_reach_count,
                key=lambda f: (
                    file_reach_count[f],
                    downstream_inbound_count[f] - len(set(graph.dependencies(f)) & cluster_files),
                    ppr_scores.get(f"file:{f}", 0.0),
                    profiles[f].risk_score if f in profiles else 0,
                ),
            )
            best_file_ppr = ppr_scores.get(f"file:{common_root_file}", 0.0)

        # Identify shared AST resources across the cluster (preferring resources defined/used in common_root_file)
        res_counts: Counter[str] = Counter()
        for m in member_nodes:
            for r in m.shared_resources:
                res_counts[r] += 1
        shared_res = tuple(
            r for r, cnt in res_counts.most_common()
            if cnt > 1 or len(member_nodes) == 1
        )
        common_root_resource: Optional[str] = None
        if shared_res:
            # Prefer a shared symbol or resource directly associated with common_root_file
            if common_root_file:
                file_matched_res = [r for r in shared_res if common_root_file in r]
                common_root_resource = file_matched_res[0] if file_matched_res else shared_res[0]
            else:
                common_root_resource = shared_res[0]

        shared_deps = tuple(
            f for f, cnt in file_reach_count.most_common()
            if cnt > 1 or f == common_root_file
        )

        # Layer C: 4-Stage Cross-Service Causal Discovery & Counterfactual Attribution
        root_trigger_id, causal_links, service_attributions, propagation_paths = _compute_temporal_causal_chain(
            member_nodes,
            common_root_file,
            temporal_window_seconds=temporal_window_seconds,
        )
        for link in causal_links:
            _add_edge(
                f"inc:{link.cause_incident_id}",
                f"inc:{link.effect_incident_id}",
                "temporal_causal_cascade",
                link.causal_confidence * 3.0,
                f"Lag +{link.lag_seconds:.1f}s (Granger={link.granger_score:.2f}): {link.mechanism}",
            )

        # Change-Point Commits on common root-cause file
        changepoints: list[dict[str, str]] = []
        if common_root_file and common_root_file in commits_by_file:
            for commit in commits_by_file[common_root_file]:
                changepoints.append({"file": common_root_file, **commit})
                commit_node_id = f"commit:{commit['sha']}"
                _add_edge(
                    commit_node_id,
                    f"file:{common_root_file}",
                    "triggered_by_commit",
                    2.5,
                    f"Recent commit [{commit['sha']}] '{commit['message']}' modified '{common_root_file}'",
                )

        hypothesis = _synthesize_hypothesis(
            member_nodes,
            common_root_file,
            common_root_resource,
            root_trigger_id,
            best_file_ppr,
            causal_links,
            llm_synthesizer=llm_synthesizer,
        )

        if not evidence_list and common_root_file:
            evidence_list.append(f"Localized to '{common_root_file}' (PPR={best_file_ppr:.3f})")

        trigger_node = next((m for m in member_nodes if m.incident_id == root_trigger_id), member_nodes[0])
        other_nodes = [m for m in member_nodes if m.incident_id != trigger_node.incident_id]
        ordered_for_merge = [trigger_node, *other_nodes]
        merged_stacks = [m.raw_incident.stack_trace for m in ordered_for_merge if m.raw_incident.stack_trace]
        merged_incident = Incident(
            incident_id=trigger_node.incident_id,
            message=f"{trigger_node.message} [Correlated root cause of {len(member_nodes)} incidents: {', '.join(m.incident_id for m in ordered_for_merge)}]",
            stack_trace="\n".join(merged_stacks),
            exception_type=trigger_node.exception_type,
            runtime=trigger_node.raw_incident.runtime,
            commit_sha=trigger_node.raw_incident.commit_sha,
            environment=trigger_node.raw_incident.environment,
            metadata={
                **trigger_node.raw_incident.metadata,
                "cross_incident_correlation_id": f"corr-{idx}",
                "correlated_incident_ids": [m.incident_id for m in ordered_for_merge],
                "common_root_cause_file": common_root_file,
                "common_root_cause_resource": common_root_resource,
                "root_cause_hypothesis": hypothesis,
            },
        )

        confidence = min(
            0.98,
            0.70
            + (0.10 if common_root_file and file_reach_count[common_root_file] == len(member_nodes) else 0.0)
            + (0.08 if common_root_resource else 0.0)
            + (0.07 if causal_links else 0.0),
        )

        correlated_groups.append(
            CorrelatedIncidentGroup(
                correlation_id=f"corr-{idx}",
                common_root_cause_file=common_root_file,
                common_root_cause_resource=common_root_resource,
                root_trigger_incident_id=root_trigger_id,
                root_cause_hypothesis=hypothesis,
                ppr_score=best_file_ppr,
                confidence=confidence,
                incidents=tuple(ordered_for_merge),
                shared_dependencies=shared_deps,
                shared_resources=shared_res,
                temporal_causal_chain=causal_links,
                changepoint_commits=tuple(changepoints),
                correlation_evidence=tuple(evidence_list),
                merged_incident=merged_incident,
                service_causal_attributions=service_attributions,
                service_propagation_paths=propagation_paths,
            )
        )

    # ── Construct Final IncidentRelationshipGraph Nodes ────────────
    graph_nodes: list[IncidentGraphNode] = []
    for n in nodes:
        nid = f"inc:{n.incident_id}"
        graph_nodes.append(
            IncidentGraphNode(
                node_id=nid,
                kind="incident",
                label=f"{n.incident_id} ({n.service}: {n.exception_type})",
                ppr_score=ppr_scores.get(nid, 0.0),
                reaching_incidents=1,
                metadata={"service": n.service, "template_id": n.drain_template.template_id},
            )
        )

    global_file_reach: Counter[str] = Counter()
    for n in nodes:
        for f in n.dependency_cone:
            global_file_reach[f] += 1

    for f_path, reach_cnt in global_file_reach.most_common():
        fid = f"file:{f_path}"
        graph_nodes.append(
            IncidentGraphNode(
                node_id=fid,
                kind="code_module",
                label=f_path,
                ppr_score=ppr_scores.get(fid, 0.0),
                reaching_incidents=reach_cnt,
                metadata={"risk_score": profiles[f_path].risk_score if f_path in profiles else 0},
            )
        )

    global_res_reach: Counter[str] = Counter()
    for n in nodes:
        for r in n.shared_resources:
            global_res_reach[r] += 1

    for res_id, reach_cnt in global_res_reach.most_common():
        graph_nodes.append(
            IncidentGraphNode(
                node_id=res_id,
                kind="shared_resource",
                label=res_id,
                ppr_score=ppr_scores.get(res_id, 0.0),
                reaching_incidents=reach_cnt,
            )
        )

    for grp in correlated_groups:
        for cp in grp.changepoint_commits:
            cid = f"commit:{cp['sha']}"
            if not any(gn.node_id == cid for gn in graph_nodes):
                graph_nodes.append(
                    IncidentGraphNode(
                        node_id=cid,
                        kind="commit_changepoint",
                        label=f"[{cp['sha']}] {cp['message']}",
                        ppr_score=0.0,
                        reaching_incidents=len(grp.incidents),
                        metadata=dict(cp),
                    )
                )

    # ── Recurring Pattern Detection (Drain Templates & Structural Terms) ─
    recurring_patterns: list[str] = []
    tpl_counts = Counter(n.drain_template.template for n in nodes)
    for tpl, cnt in tpl_counts.most_common():
        if cnt > 1:
            recurring_patterns.append(f"Recurring Drain log template ({cnt}×): '{tpl}'")
    concept_counts = Counter(c for n in nodes for c in n.concepts)
    for concept, cnt in concept_counts.most_common(6):
        if cnt > 1:
            recurring_patterns.append(f"Shared structural signal ({cnt}× incidents): '{concept}'")

    # ── Layer A: Historical Hybrid Graph-RAG Comparisons (RRF) ─────
    all_shared_files = [f for g in correlated_groups for f in g.shared_dependencies]
    all_shared_res = [r for g in correlated_groups for r in g.shared_resources]
    historical_comparisons = _retrieve_historical_comparisons_rrf(
        nodes,
        all_shared_files,
        all_shared_res,
        profiles,
        repair_run_dir,
        repo_root,
    )

    multi_incident_groups = sum(1 for g in correlated_groups if len(g.incidents) > 1)
    single_incident_groups = sum(1 for g in correlated_groups if len(g.incidents) == 1)

    return CrossIncidentIntelligenceReport(
        total_incidents_analyzed=len(nodes),
        correlated_root_causes=len(correlated_groups),
        uncorrelated_incidents=single_incident_groups if multi_incident_groups > 0 else 0,
        correlated_groups=tuple(correlated_groups),
        relationship_graph=IncidentRelationshipGraph(
            nodes=tuple(graph_nodes),
            edges=tuple(graph_edges),
        ),
        historical_comparisons=historical_comparisons,
        recurring_patterns=tuple(recurring_patterns),
    )
