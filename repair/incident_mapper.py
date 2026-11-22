"""Deterministic conversion of production incidents into repair context."""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence
from urllib.parse import unquote, urlparse

from intelligence.dependency_graph import DependencyGraph
from intelligence.file_profile import FileProfile
from intelligence.git_analyzer import get_recent_commits_for_files
from intelligence.models import EXTENSION_TO_LANGUAGE
from intelligence.query_engine import query_codebase


_NON_TRACEBACK_LANGUAGES = frozenset({"json", "yaml", "toml", "html", "css"})
_TRACEBACK_EXTENSIONS = tuple(
    sorted(
        {
            ext.lstrip(".")
            for ext, language in EXTENSION_TO_LANGUAGE.items()
            if language not in _NON_TRACEBACK_LANGUAGES
        }
        | {"mjs", "cjs", "kts", "hxx"},
        key=len,
        reverse=True,
    )
)
_EXT_GROUP = "|".join(re.escape(ext) for ext in _TRACEBACK_EXTENSIONS)
_PATH_PREFIX = (
    r"(?:file://[^\r\n:\"'`]+|"
    r"(?:[A-Za-z]:[\\/]|/|\.{1,2}[\\/]|[^\s\\/:`\"']+[\\/])[^\r\n:\"'`]*|"
    r"[^\s\\/:`\"']+)"
)
_GO_SYMBOL = r"[A-Za-z_](?:[\w./()*]|\[[^\r\n\]]*\])*"

_PYTHON_FRAME = re.compile(
    r'^\s*File "(?P<path>[^"\r\n]+)",\s*line (?P<line>\d+)(?:,\s*in (?P<function>[^\r\n,]+?))?\s*$'
)
_AT_LINE = re.compile(r"^\s*at\s+(?P<body>.+?)\s*$")
_LOCATION_SUFFIX = re.compile(
    rf"^(?P<path>{_PATH_PREFIX}\.(?:{_EXT_GROUP})):(?P<line>\d+)(?::(?P<column>\d+))?$"
)
_GO_CALL_LINE = re.compile(
    rf"^(?P<function>{_GO_SYMBOL})\([^()\r\n]*\)\s*$"
)
_GO_CREATED_BY_LINE = re.compile(
    rf"^created by (?P<function>{_GO_SYMBOL})(?:\s+in goroutine \d+)?\s*$"
)
_GO_LOCATION_LINE = re.compile(
    rf"^\s+(?P<path>{_PATH_PREFIX}\.go):(?P<line>\d+)(?:\s+\+0x[0-9a-fA-F]+)?\s*$"
)
_RUBY_FRAME = re.compile(
    rf"^\s*(?P<from_kw>from\s+)?(?P<path>{_PATH_PREFIX}\.rb):(?P<line>\d+):in\s+[`'\x22](?P<function>[^`'\x22\r\n]+)[`'\x22]\s*$"
)
_PHP_FRAME = re.compile(
    rf"^\s*#(?P<frame_num>\d+)\s+(?P<path>{_PATH_PREFIX}\.php)\((?P<line>\d+)\):\s*(?P<function>[^\r\n(]+?)(?:\(.*\))?\s*$"
)
_RUST_BACKTRACE_FUNC = re.compile(
    r"^\s*\d+:\s+(?:0x[0-9a-fA-F]+\s+-\s+)?(?P<function>(?:<[A-Za-z_]|[A-Za-z_])[\w:$<>.,\{\}\[\]&* ]*?)\s*$"
)
_RUST_PANIC_FRAME = re.compile(
    rf"^\s*thread\s+'[^']+'\s+panicked\s+at\s+(?:'[^']*',\s*)?(?P<path>{_PATH_PREFIX}\.rs):(?P<line>\d+)(?::(?P<column>\d+))?:?\s*$"
)

_TRACEBACK_MOST_RECENT_HEADER = re.compile(
    r"^\s*Traceback \(most recent call last\):\s*$",
    re.IGNORECASE,
)
_PYTHON_CHAIN_HEADER = re.compile(
    r"^\s*(?:During handling of the above exception, another exception occurred:|"
    r"The above exception was the direct cause of the following exception:)\s*$"
)
_CAUSED_BY_HEADER = re.compile(r"^\s*(?:Caused by|Suppressed):\s*.+$")
_CSHARP_INNER_HEADER = re.compile(r"^\s*--->\s*.+$")
_GO_GOROUTINE_HEADER = re.compile(r"^\s*goroutine\s+\d+\s+\[[^\]]+\]:\s*$")


FrameOrder = Literal["outer_to_inner", "inner_to_outer", "unknown"]
OrderConfidence = Literal["high", "medium", "unknown"]
FrameKind = Literal["call", "origin", "cause_header"]
MappingStatus = Literal["mapped", "unmapped"]
ValidationStatus = Literal[
    "ok",
    "line_non_positive",
    "line_out_of_range",
    "file_missing",
    "file_unreadable",
    "unmapped",
]
FunctionMatchStatus = Literal[
    "exact",
    "container_match",
    "mismatch",
    "anonymous_or_module",
    "unverified",
    "unknown",
]
FileStatus = Literal["ok", "file_missing", "file_unreadable"]
LineEndingStyle = Literal["lf", "crlf", "cr", "mixed", "none"]
SymbolVerificationStatus = Literal["verified", "unavailable", "not_indexed"]
ExcerptRole = Literal["frame", "signature"]
EvidenceTier = Literal["confirmed", "corroborated", "suspected"]

_SOURCE_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")
_ANONYMOUS_OR_MODULE_NAMES = frozenset({
    "<module>",
    "<lambda>",
    "<anonymous>",
    "anonymous",
    "{{closure}}",
    "<listcomp>",
    "<dictcomp>",
    "<setcomp>",
    "<genexpr>",
})


@dataclass(frozen=True)
class IncidentFrame:
    path: str
    line: int
    function: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "function": self.function}


@dataclass(frozen=True)
class TraceSegment:
    segment_id: int
    header: str
    language: str | None
    frame_order: FrameOrder
    order_confidence: OrderConfidence
    order_reason: str
    frame_indices: list[int] = field(default_factory=list)
    innermost_call_frame_index: int | None = None
    innermost_mapped_frame_index: int | None = None
    innermost_valid_frame_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "header": self.header,
            "language": self.language,
            "frame_order": self.frame_order,
            "order_confidence": self.order_confidence,
            "order_reason": self.order_reason,
            "frame_indices": list(self.frame_indices),
            "innermost_call_frame_index": self.innermost_call_frame_index,
            "innermost_mapped_frame_index": self.innermost_mapped_frame_index,
            "innermost_valid_frame_index": self.innermost_valid_frame_index,
        }


@dataclass(frozen=True)
class FrameMapping:
    status: MappingStatus
    mapped_path: str | None
    match_strategy: str
    candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mapped_path": self.mapped_path,
            "match_strategy": self.match_strategy,
            "candidate_count": self.candidate_count,
        }


@dataclass(frozen=True)
class FrameValidation:
    status: ValidationStatus = "unmapped"
    file_exists: bool | None = None
    total_lines: int | None = None
    line_in_range: bool | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "file_exists": self.file_exists,
            "total_lines": self.total_lines,
            "line_in_range": self.line_in_range,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SourceSymbolInfo:
    name: str
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class FrameSymbolMatch:
    status: FunctionMatchStatus = "unknown"
    reported_function: str | None = None
    normalized_reported_function: str | None = None
    enclosing_symbol: SourceSymbolInfo | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reported_function": self.reported_function,
            "normalized_reported_function": self.normalized_reported_function,
            "enclosing_symbol": (
                self.enclosing_symbol.to_dict() if self.enclosing_symbol is not None else None
            ),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RepairFrame:
    frame_index: int
    segment_id: int
    segment_position: int
    raw_text: str
    reported_path: str
    reported_line: int
    reported_column: int | None
    reported_function: str | None
    language: str | None
    frame_kind: FrameKind
    frame_order: FrameOrder
    is_innermost_call: bool | None
    is_innermost_mapped: bool | None
    is_innermost_valid: bool | None
    call_depth_from_innermost: int | None
    mapping: FrameMapping
    validation: FrameValidation = field(default_factory=FrameValidation)
    symbol_match: FrameSymbolMatch = field(default_factory=FrameSymbolMatch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "segment_id": self.segment_id,
            "segment_position": self.segment_position,
            "raw_text": self.raw_text,
            "reported_path": self.reported_path,
            "reported_line": self.reported_line,
            "reported_column": self.reported_column,
            "reported_function": self.reported_function,
            "language": self.language,
            "frame_kind": self.frame_kind,
            "frame_order": self.frame_order,
            "is_innermost_call": self.is_innermost_call,
            "is_innermost_mapped": self.is_innermost_mapped,
            "is_innermost_valid": self.is_innermost_valid,
            "call_depth_from_innermost": self.call_depth_from_innermost,
            "mapping": self.mapping.to_dict(),
            "validation": self.validation.to_dict(),
            "symbol_match": self.symbol_match.to_dict(),
        }


@dataclass(frozen=True)
class ParsedStackTrace:
    frames: list[RepairFrame] = field(default_factory=list)
    trace_segments: list[TraceSegment] = field(default_factory=list)

    @property
    def segments(self) -> list[TraceSegment]:
        return self.trace_segments

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": [frame.to_dict() for frame in self.frames],
            "trace_segments": [segment.to_dict() for segment in self.trace_segments],
        }


@dataclass(frozen=True)
class Incident:
    incident_id: str
    message: str
    stack_trace: str = ""
    exception_type: str = ""
    runtime: str = ""
    commit_sha: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Incident":
        if not isinstance(payload, dict):
            raise ValueError("incident payload must be a JSON object")
        message = payload.get("message") or payload.get("error") or ""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("incident payload requires a non-empty 'message' or 'error'")
        stack_trace = payload.get("stack_trace") or payload.get("stack") or ""
        if isinstance(stack_trace, list):
            stack_trace = "\n".join(str(line) for line in stack_trace)
        if not isinstance(stack_trace, str):
            raise ValueError("incident 'stack_trace' must be a string or list of strings")
        return cls(
            incident_id=str(payload.get("incident_id") or payload.get("id") or "incident"),
            message=message.strip(),
            stack_trace=stack_trace,
            exception_type=str(payload.get("exception_type") or payload.get("type") or ""),
            runtime=str(payload.get("runtime") or ""),
            commit_sha=str(payload.get("commit_sha") or payload.get("commit") or ""),
            environment=dict(payload.get("environment") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "exception_type": self.exception_type,
            "runtime": self.runtime,
            "commit_sha": self.commit_sha,
            "environment": self.environment,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CodeExcerpt:
    excerpt_id: str
    role: ExcerptRole
    start_line: int
    end_line: int
    target_lines: list[int] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    enclosing_symbol: str | None = None
    effective_radius: int | None = None
    clipped_top: bool = False
    clipped_bottom: bool = False
    context_trimmed: bool = False
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "excerpt_id": self.excerpt_id,
            "role": self.role,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "target_lines": list(self.target_lines),
            "frame_indices": list(self.frame_indices),
            "enclosing_symbol": self.enclosing_symbol,
            "effective_radius": self.effective_radius,
            "clipped_top": self.clipped_top,
            "clipped_bottom": self.clipped_bottom,
            "context_trimmed": self.context_trimmed,
            "content": self.content,
        }


@dataclass(frozen=True)
class OmittedTargetLine:
    path: str
    line: int
    frame_index: int
    frame_indices: list[int] = field(default_factory=list)
    excerpt_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "frame_index": self.frame_index,
            "frame_indices": list(self.frame_indices),
            "excerpt_id": self.excerpt_id,
        }


@dataclass(frozen=True)
class ExcerptCharTruncation:
    truncated: bool = False
    limit_chars: int = 12000
    actual_chars: int = 0
    context_trimmed_excerpt_ids: list[str] = field(default_factory=list)
    omitted_excerpt_ids: list[str] = field(default_factory=list)
    target_lines_omitted: bool = False
    omitted_target_lines: list[OmittedTargetLine] = field(default_factory=list)

    @property
    def omitted_excerpts(self) -> int:
        return len(self.omitted_excerpt_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "truncated": self.truncated,
            "limit_chars": self.limit_chars,
            "actual_chars": self.actual_chars,
            "context_trimmed_excerpt_ids": list(self.context_trimmed_excerpt_ids),
            "omitted_excerpt_ids": list(self.omitted_excerpt_ids),
            "omitted_excerpts": self.omitted_excerpts,
            "target_lines_omitted": self.target_lines_omitted,
            "omitted_target_lines": [item.to_dict() for item in self.omitted_target_lines],
        }


@dataclass(frozen=True)
class DimensionTruncation:
    truncated: bool = False
    limit: int = 0
    total_candidates: int = 0
    retained: int = 0
    omitted: int = 0
    omitted_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "truncated": self.truncated,
            "limit": self.limit,
            "total_candidates": self.total_candidates,
            "retained": self.retained,
            "omitted": self.omitted,
            "omitted_paths": list(self.omitted_paths),
        }


@dataclass(frozen=True)
class ContextTruncation:
    excerpt_chars: ExcerptCharTruncation = field(default_factory=ExcerptCharTruncation)
    frame_files: DimensionTruncation | None = None
    non_frame_evidence: DimensionTruncation | None = None

    @property
    def truncated(self) -> bool:
        return self.excerpt_chars.truncated

    @property
    def limit_chars(self) -> int:
        return self.excerpt_chars.limit_chars

    @property
    def actual_chars(self) -> int:
        return self.excerpt_chars.actual_chars

    @property
    def context_trimmed_excerpt_ids(self) -> list[str]:
        return self.excerpt_chars.context_trimmed_excerpt_ids

    @property
    def omitted_excerpt_ids(self) -> list[str]:
        return self.excerpt_chars.omitted_excerpt_ids

    @property
    def omitted_excerpts(self) -> int:
        return self.excerpt_chars.omitted_excerpts

    @property
    def target_lines_omitted(self) -> bool:
        return self.excerpt_chars.target_lines_omitted

    @property
    def omitted_target_lines(self) -> list[OmittedTargetLine]:
        return self.excerpt_chars.omitted_target_lines

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "excerpt_chars": self.excerpt_chars.to_dict(),
        }
        if self.frame_files is not None:
            result["frame_files"] = self.frame_files.to_dict()
        if self.non_frame_evidence is not None:
            result["non_frame_evidence"] = self.non_frame_evidence.to_dict()
        return result


@dataclass
class RepairEvidence:
    path: str
    score: float
    reasons: list[str] = field(default_factory=list)
    frame_lines: list[int] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    excerpt: str = ""
    frame_indices: list[int] = field(default_factory=list)
    valid_frame_lines: list[int] = field(default_factory=list)
    invalid_frame_lines: list[int] = field(default_factory=list)
    excerpts: list[CodeExcerpt] = field(default_factory=list)
    primary_excerpt_id: str | None = None
    omitted_excerpt_ids: list[str] = field(default_factory=list)
    file_status: FileStatus | None = None
    total_lines: int | None = None
    line_ending_style: LineEndingStyle = "none"
    symbol_verification: SymbolVerificationStatus = "unavailable"
    patch_eligible: bool = False
    patch_eligibility_reason: str = "ineligible_unverified"
    evidence_tier: EvidenceTier = "suspected"
    risk_score: int = 0
    risk_label: str = "Low"
    direct_dependents: int = 0
    transitive_dependents: int = 0
    recent_changes: int = 0
    last_modified: str = ""
    calls: list[str] = field(default_factory=list)
    api_endpoints: list[str] = field(default_factory=list)
    db_interactions: list[str] = field(default_factory=list)
    service_calls: list[str] = field(default_factory=list)
    config_keys: list[str] = field(default_factory=list)
    related_commits: list[dict[str, str]] = field(default_factory=list)
    affected_components: list[str] = field(default_factory=list)
    contributing_factors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.evidence_tier == "suspected" and (self.valid_frame_lines or self.patch_eligible):
            if self.patch_eligibility_reason == "eligible_direct_graph_neighbor":
                self.evidence_tier = "corroborated"
            else:
                self.evidence_tier = "confirmed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "frame_lines": self.frame_lines,
            "symbols": self.symbols,
            "excerpt": self.excerpt,
            "frame_indices": list(self.frame_indices),
            "valid_frame_lines": list(self.valid_frame_lines),
            "invalid_frame_lines": list(self.invalid_frame_lines),
            "excerpts": [item.to_dict() for item in self.excerpts],
            "primary_excerpt_id": self.primary_excerpt_id,
            "omitted_excerpt_ids": list(self.omitted_excerpt_ids),
            "file_status": self.file_status,
            "total_lines": self.total_lines,
            "line_ending_style": self.line_ending_style,
            "symbol_verification": self.symbol_verification,
            "patch_eligible": self.patch_eligible,
            "patch_eligibility_reason": self.patch_eligibility_reason,
            "evidence_tier": self.evidence_tier,
            "risk_score": self.risk_score,
            "risk_label": self.risk_label,
            "direct_dependents": self.direct_dependents,
            "transitive_dependents": self.transitive_dependents,
            "recent_changes": self.recent_changes,
            "last_modified": self.last_modified,
            "calls": list(self.calls),
            "api_endpoints": list(self.api_endpoints),
            "db_interactions": list(self.db_interactions),
            "service_calls": list(self.service_calls),
            "config_keys": list(self.config_keys),
            "related_commits": list(self.related_commits),
            "affected_components": list(self.affected_components),
            "contributing_factors": list(self.contributing_factors),
        }


@dataclass
class RepairContext:
    incident: Incident
    repository_path: str
    evidence: list[RepairEvidence] = field(default_factory=list)
    unmapped_frames: list[IncidentFrame] = field(default_factory=list)
    frames: list[RepairFrame] = field(default_factory=list)
    trace_segments: list[TraceSegment] = field(default_factory=list)
    truncation: ContextTruncation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident": self.incident.to_dict(),
            "repository_path": self.repository_path,
            "evidence": [item.to_dict() for item in self.evidence],
            "unmapped_frames": [frame.to_dict() for frame in self.unmapped_frames],
            "frames": [frame.to_dict() for frame in self.frames],
            "trace_segments": [segment.to_dict() for segment in self.trace_segments],
            "truncation": self.truncation.to_dict() if self.truncation is not None else None,
        }


def load_incident(path: str) -> Incident:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load incident JSON: {exc}") from exc
    return Incident.from_dict(payload)


def _clean_frame_path(raw_path: str) -> str:
    path = raw_path.strip()
    if path.startswith("file://"):
        parsed = urlparse(path)
        url_path = unquote(parsed.path or "")
        if re.match(r"^/[A-Za-z]:[\\/]", url_path):
            path = url_path[1:]
        else:
            path = url_path
    return path.strip()


def _parse_optional_column(raw_column: str | None) -> int | None:
    if not raw_column:
        return None
    col = int(raw_column)
    return col if col > 0 else None


def _infer_at_frame_language(reported_path: str, used_rust_backtrace: bool = False) -> str:
    if used_rust_backtrace or reported_path.lower().endswith(".rs"):
        return "rust"
    _, ext = posixpath.splitext(reported_path.replace("\\", "/"))
    ext_lower = ext.lower()
    if ext_lower in {".java", ".kt", ".kts", ".scala"}:
        return "jvm"
    if ext_lower in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        return "node"
    if ext_lower == ".cs":
        return "csharp"
    if ext_lower in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}:
        return "native"
    if ext_lower == ".py":
        return "python"
    if ext_lower == ".rb":
        return "ruby"
    if ext_lower == ".php":
        return "php"
    if ext_lower == ".go":
        return "go"
    return EXTENSION_TO_LANGUAGE.get(ext_lower, "generic")


@dataclass(frozen=True)
class _ParsedRawFrame:
    raw_text: str
    reported_path: str
    reported_line: int
    reported_column: int | None
    reported_function: str | None
    language: str
    frame_kind: FrameKind
    syntax_family: str


def _build_raw_frame(
    raw_text: str,
    raw_path: str,
    raw_line: str,
    raw_function: str = "",
    raw_column: str | None = None,
    *,
    language: str,
    frame_kind: FrameKind = "call",
    syntax_family: str,
) -> _ParsedRawFrame | None:
    path = _clean_frame_path(raw_path)
    line_number = int(raw_line)
    if not path or line_number <= 0:
        return None
    fn = raw_function.strip()
    return _ParsedRawFrame(
        raw_text=raw_text,
        reported_path=path,
        reported_line=line_number,
        reported_column=_parse_optional_column(raw_column),
        reported_function=fn if fn else None,
        language=language,
        frame_kind=frame_kind,
        syntax_family=syntax_family,
    )


def _parse_at_raw_frame(
    raw_line: str,
    rust_default_function: str = "",
    rust_raw_line: str = "",
) -> _ParsedRawFrame | None:
    at_match = _AT_LINE.match(raw_line)
    if not at_match:
        return None
    body = at_match.group("body")

    if body.endswith(")"):
        location_text = ""
        function_text = ""
        if " (" in body:
            split_index = body.find(" (")
            function_text = body[:split_index].strip()
            location_text = body[split_index + 2 : -1].strip()
        elif "(" in body:
            split_index = body.find("(")
            candidate_fn = body[:split_index].strip()
            if candidate_fn and not any(ch.isspace() for ch in candidate_fn):
                function_text = candidate_fn
                location_text = body[split_index + 1 : -1].strip()
        if location_text:
            loc_match = _LOCATION_SUFFIX.match(location_text)
            if loc_match:
                cleaned_fn = re.sub(r"^(?:(?:async|new)\s+)+", "", function_text).strip()
                raw_path = loc_match.group("path")
                is_rs = raw_path.lower().endswith(".rs")
                used_rust_bt = bool(is_rs and not cleaned_fn and rust_default_function)
                fallback_fn = rust_default_function if is_rs else ""
                combined_raw = f"{rust_raw_line}\n{raw_line}" if used_rust_bt and rust_raw_line else raw_line
                lang = _infer_at_frame_language(raw_path, used_rust_backtrace=used_rust_bt)
                return _build_raw_frame(
                    combined_raw,
                    raw_path,
                    loc_match.group("line"),
                    cleaned_fn or fallback_fn,
                    loc_match.group("column"),
                    language=lang,
                    frame_kind="call",
                    syntax_family="rust" if is_rs else "at",
                )

    loc_match = _LOCATION_SUFFIX.match(body)
    if loc_match:
        raw_path = loc_match.group("path")
        is_rs = raw_path.lower().endswith(".rs")
        used_rust_bt = bool(is_rs and rust_default_function)
        fallback_fn = rust_default_function if is_rs else ""
        combined_raw = f"{rust_raw_line}\n{raw_line}" if used_rust_bt and rust_raw_line else raw_line
        lang = _infer_at_frame_language(raw_path, used_rust_backtrace=used_rust_bt)
        return _build_raw_frame(
            combined_raw,
            raw_path,
            loc_match.group("line"),
            fallback_fn,
            loc_match.group("column"),
            language=lang,
            frame_kind="call",
            syntax_family="rust" if is_rs else "at",
        )
    return None


@dataclass
class _SegmentBuilder:
    header: str = ""
    has_most_recent_header: bool = False
    has_go_header: bool = False
    has_go_func_pair: bool = False
    raw_frames: list[_ParsedRawFrame] = field(default_factory=list)


_DOCUMENTED_AT_LANGUAGES = frozenset({"jvm", "node", "csharp", "native"})


def _resolve_segment_ordering(
    seg: _SegmentBuilder,
) -> tuple[str | None, FrameOrder, OrderConfidence, str]:
    languages = {f.language for f in seg.raw_frames if f.language}
    segment_language = next(iter(languages)) if len(languages) == 1 else None
    syntax_families = {f.syntax_family for f in seg.raw_frames}

    # Check intrinsic direction per frame in this segment
    intrinsic_directions: set[FrameOrder] = set()
    for frame in seg.raw_frames:
        if frame.syntax_family == "python":
            intrinsic_directions.add("outer_to_inner")
        elif frame.syntax_family == "ruby":
            intrinsic_directions.add(
                "outer_to_inner" if seg.has_most_recent_header else "inner_to_outer"
            )
        elif frame.syntax_family in {"go", "rust", "php"}:
            intrinsic_directions.add("inner_to_outer")
        elif frame.syntax_family == "at" and frame.language in _DOCUMENTED_AT_LANGUAGES:
            intrinsic_directions.add("inner_to_outer")
        else:
            intrinsic_directions.add("unknown")

    if len(intrinsic_directions) > 1:
        return (segment_language, "unknown", "unknown", "mixed_frame_formats")

    direction = next(iter(intrinsic_directions))
    if direction == "unknown":
        return (segment_language, "unknown", "unknown", "unknown_at_frame_order")

    if direction == "outer_to_inner":
        if seg.has_most_recent_header:
            reason = (
                "ruby_traceback_header"
                if segment_language == "ruby"
                else "python_traceback_header"
            )
            return (segment_language, "outer_to_inner", "high", reason)
        return (segment_language, "outer_to_inner", "medium", "python_file_frames")

    # direction == "inner_to_outer"
    if seg.has_most_recent_header:
        return (segment_language, "unknown", "unknown", "contradictory_header_and_frames")

    if len(syntax_families) > 1:
        return (segment_language, "unknown", "unknown", "mixed_frame_formats")

    family = next(iter(syntax_families))
    if family == "go":
        confidence: OrderConfidence = (
            "high" if (seg.has_go_header or seg.has_go_func_pair) else "medium"
        )
        return ("go", "inner_to_outer", confidence, "go_goroutine_stack")
    if family == "rust":
        return ("rust", "inner_to_outer", "high", "rust_backtrace")
    if family == "php":
        return ("php", "inner_to_outer", "high", "php_numbered_frames")
    if family == "ruby":
        return ("ruby", "inner_to_outer", "medium", "ruby_from_frames")

    # family == "at" — only documented single-language at-chains receive inner_to_outer
    if segment_language == "jvm":
        return ("jvm", "inner_to_outer", "high", "jvm_at_chain")
    if segment_language == "node":
        return ("node", "inner_to_outer", "high", "node_at_chain")
    if segment_language == "csharp":
        return ("csharp", "inner_to_outer", "high", "csharp_at_chain")
    if segment_language == "native":
        return ("native", "inner_to_outer", "high", "native_at_chain")
    return (segment_language, "unknown", "unknown", "unknown_at_frame_order")


def extract_stack_frames_detailed(stack_trace: str) -> ParsedStackTrace:
    segment_builders: list[_SegmentBuilder] = []
    current_segment = _SegmentBuilder()

    pending_go_function = ""
    pending_go_raw_line = ""
    pending_go_kind: FrameKind = "call"
    pending_rust_function = ""
    pending_rust_raw_line = ""

    def _clear_pending() -> None:
        nonlocal pending_go_function, pending_go_raw_line, pending_go_kind
        nonlocal pending_rust_function, pending_rust_raw_line
        pending_go_function = ""
        pending_go_raw_line = ""
        pending_go_kind = "call"
        pending_rust_function = ""
        pending_rust_raw_line = ""

    def _start_or_update_segment(
        header_text: str,
        *,
        is_most_recent: bool = False,
        is_go_header: bool = False,
    ) -> None:
        nonlocal current_segment
        if current_segment.raw_frames:
            segment_builders.append(current_segment)
            current_segment = _SegmentBuilder(
                header=header_text,
                has_most_recent_header=is_most_recent,
                has_go_header=is_go_header,
            )
        else:
            if is_most_recent:
                current_segment.has_most_recent_header = True
                current_segment.header = header_text
            elif is_go_header:
                current_segment.has_go_header = True
                current_segment.header = header_text
            elif not current_segment.header:
                current_segment.header = header_text

    for raw_line in stack_trace.splitlines():
        if _TRACEBACK_MOST_RECENT_HEADER.match(raw_line):
            _clear_pending()
            _start_or_update_segment(raw_line.strip(), is_most_recent=True)
            continue

        if _GO_GOROUTINE_HEADER.match(raw_line):
            _clear_pending()
            _start_or_update_segment(raw_line.strip(), is_go_header=True)
            continue

        if (
            _PYTHON_CHAIN_HEADER.match(raw_line)
            or _CAUSED_BY_HEADER.match(raw_line)
            or _CSHARP_INNER_HEADER.match(raw_line)
        ):
            _clear_pending()
            _start_or_update_segment(raw_line.strip())
            continue

        go_call_match = _GO_CALL_LINE.match(raw_line)
        if go_call_match:
            pending_go_function = go_call_match.group("function").strip()
            pending_go_raw_line = raw_line
            pending_go_kind = "call"
            pending_rust_function = ""
            pending_rust_raw_line = ""
            continue

        go_created_match = _GO_CREATED_BY_LINE.match(raw_line)
        if go_created_match:
            pending_go_function = go_created_match.group("function").strip()
            pending_go_raw_line = raw_line
            pending_go_kind = "origin"
            pending_rust_function = ""
            pending_rust_raw_line = ""
            continue

        rust_func_match = _RUST_BACKTRACE_FUNC.match(raw_line)
        if rust_func_match:
            pending_rust_function = rust_func_match.group("function").strip()
            pending_rust_raw_line = raw_line
            pending_go_function = ""
            pending_go_raw_line = ""
            pending_go_kind = "call"
            continue

        go_loc_match = _GO_LOCATION_LINE.match(raw_line)
        if go_loc_match:
            had_go_fn = bool(pending_go_function)
            combined_raw = (
                f"{pending_go_raw_line}\n{raw_line}" if pending_go_raw_line else raw_line
            )
            frame = _build_raw_frame(
                combined_raw,
                go_loc_match.group("path"),
                go_loc_match.group("line"),
                pending_go_function,
                language="go",
                frame_kind=pending_go_kind if had_go_fn else "call",
                syntax_family="go",
            )
            _clear_pending()
            if frame is not None:
                if had_go_fn:
                    current_segment.has_go_func_pair = True
                current_segment.raw_frames.append(frame)
            continue

        at_frame = _parse_at_raw_frame(
            raw_line,
            rust_default_function=pending_rust_function,
            rust_raw_line=pending_rust_raw_line,
        )
        if at_frame is not None:
            _clear_pending()
            current_segment.raw_frames.append(at_frame)
            continue

        _clear_pending()
        for pattern, lang, family in (
            (_PYTHON_FRAME, "python", "python"),
            (_RUBY_FRAME, "ruby", "ruby"),
            (_PHP_FRAME, "php", "php"),
            (_RUST_PANIC_FRAME, "rust", "rust"),
        ):
            match = pattern.match(raw_line)
            if match:
                group_dict = match.groupdict()
                frame = _build_raw_frame(
                    raw_line,
                    match.group("path"),
                    match.group("line"),
                    group_dict.get("function") or "",
                    group_dict.get("column"),
                    language=lang,
                    frame_kind="call",
                    syntax_family=family,
                )
                if frame is not None:
                    current_segment.raw_frames.append(frame)
                break

    if current_segment.raw_frames:
        segment_builders.append(current_segment)

    all_frames: list[RepairFrame] = []
    all_segments: list[TraceSegment] = []
    unmapped_default = FrameMapping(
        status="unmapped",
        mapped_path=None,
        match_strategy="unmapped_not_evaluated",
        candidate_count=0,
    )

    global_frame_index = 0
    for seg_id, seg_builder in enumerate(segment_builders):
        seg_lang, frame_order, order_confidence, order_reason = _resolve_segment_ordering(
            seg_builder
        )
        seg_frame_indices = list(
            range(global_frame_index, global_frame_index + len(seg_builder.raw_frames))
        )
        call_positions = [
            pos
            for pos, rf in enumerate(seg_builder.raw_frames)
            if rf.frame_kind == "call"
        ]

        innermost_call_frame_index: int | None = None
        call_depth_by_pos: dict[int, int] = {}
        if frame_order == "outer_to_inner" and call_positions:
            innermost_pos = call_positions[-1]
            innermost_call_frame_index = global_frame_index + innermost_pos
            total_calls = len(call_positions)
            for call_rank, pos in enumerate(call_positions):
                call_depth_by_pos[pos] = (total_calls - 1) - call_rank
        elif frame_order == "inner_to_outer" and call_positions:
            innermost_pos = call_positions[0]
            innermost_call_frame_index = global_frame_index + innermost_pos
            for call_rank, pos in enumerate(call_positions):
                call_depth_by_pos[pos] = call_rank

        for seg_pos, rf in enumerate(seg_builder.raw_frames):
            f_idx = global_frame_index + seg_pos
            if frame_order == "unknown" or rf.frame_kind != "call":
                is_innermost_call: bool | None = None
                call_depth: int | None = None
            else:
                is_innermost_call = f_idx == innermost_call_frame_index
                call_depth = call_depth_by_pos.get(seg_pos)

            all_frames.append(
                RepairFrame(
                    frame_index=f_idx,
                    segment_id=seg_id,
                    segment_position=seg_pos,
                    raw_text=rf.raw_text,
                    reported_path=rf.reported_path,
                    reported_line=rf.reported_line,
                    reported_column=rf.reported_column,
                    reported_function=rf.reported_function,
                    language=rf.language,
                    frame_kind=rf.frame_kind,
                    frame_order=frame_order,
                    is_innermost_call=is_innermost_call,
                    is_innermost_mapped=None,
                    is_innermost_valid=None,
                    call_depth_from_innermost=call_depth,
                    mapping=unmapped_default,
                    validation=FrameValidation(),
                    symbol_match=FrameSymbolMatch(
                        status="unknown",
                        reported_function=rf.reported_function,
                    ),
                )
            )

        all_segments.append(
            TraceSegment(
                segment_id=seg_id,
                header=seg_builder.header,
                language=seg_lang,
                frame_order=frame_order,
                order_confidence=order_confidence,
                order_reason=order_reason,
                frame_indices=seg_frame_indices,
                innermost_call_frame_index=innermost_call_frame_index,
                innermost_mapped_frame_index=None,
                innermost_valid_frame_index=None,
            )
        )
        global_frame_index += len(seg_builder.raw_frames)

    return ParsedStackTrace(frames=all_frames, trace_segments=all_segments)


def extract_stack_frames(stack_trace: str) -> list[IncidentFrame]:
    parsed = extract_stack_frames_detailed(stack_trace)
    return [
        IncidentFrame(
            path=frame.reported_path,
            line=frame.reported_line,
            function=frame.reported_function or "",
        )
        for frame in parsed.frames
    ]


_WINDOWS_ROOT_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|\\)")
_JVM_EXTENSIONS = frozenset({".java", ".kt", ".kts", ".scala"})
_JVM_PREFIX_TOKEN = r"[A-Za-z_][A-Za-z0-9_.$@-]*"
_JVM_FUNCTION_RE = re.compile(
    rf"^(?:(?P<prefix>{_JVM_PREFIX_TOKEN}(?:/(?:{_JVM_PREFIX_TOKEN})?)?)/)?(?P<symbol>[A-Za-z_$][A-Za-z0-9_$.$<>-]*)$"
)
_JVM_IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_JVM_METHOD_RE = re.compile(r"^(?:[A-Za-z_$][A-Za-z0-9_$-]*|<init>|<clinit>)$")
_EXTERNAL_ENV_SEGMENTS = frozenset({
    ".venv",
    "venv",
    "site-packages",
    "dist-packages",
    "node_modules",
    "__pypackages__",
    ".tox",
    ".nox",
    ".eggs",
})
_EXTERNAL_CACHE_SUBPATHS = (
    ".cargo/registry/",
    ".cargo/git/",
    "pkg/mod/",
)


def _normalize_relative_posix(raw_path: str) -> str | None:
    path = raw_path.strip()
    if not path:
        return None
    if os.path.isabs(path) or _WINDOWS_ROOT_RE.match(path):
        return None
    posix_path = path.replace("\\", "/")
    while posix_path.startswith("./"):
        posix_path = posix_path[2:]
    normalized = posixpath.normpath(posix_path)
    if normalized in ("", ".", "..") or normalized.startswith("../") or normalized.startswith("/"):
        return None
    return normalized


def _is_physically_within_repo(repo_root: str, target_path: str) -> bool:
    if not repo_root:
        return True
    try:
        real_root = os.path.realpath(repo_root)
        real_target = os.path.realpath(target_path)
        return (
            real_target != real_root
            and os.path.commonpath([real_target, real_root]) == real_root
        )
    except ValueError:
        return False


def _has_external_env_segment(posix_rel_path: str) -> bool:
    lowered = posix_rel_path.lower()
    if any(lowered.startswith(prefix) or f"/{prefix}" in lowered for prefix in _EXTERNAL_CACHE_SUBPATHS):
        return True
    return any(part in _EXTERNAL_ENV_SEGMENTS for part in lowered.split("/"))


def _jvm_qualified_relpath(filename: str, frame_function: str) -> str | None:
    if "/" in filename or not frame_function:
        return None
    file_stem, ext = posixpath.splitext(filename)
    ext_lower = ext.lower()
    if not file_stem or ext_lower not in _JVM_EXTENSIONS:
        return None
    fn_match = _JVM_FUNCTION_RE.match(frame_function.strip())
    if not fn_match:
        return None
    prefix = fn_match.group("prefix")
    if prefix and ".." in prefix:
        return None
    symbol = fn_match.group("symbol")
    parts = symbol.split(".")
    if len(parts) < 3 or any(not part for part in parts):
        return None
    package_parts = parts[:-2]
    class_part = parts[-2]
    method_part = parts[-1]
    if not all(_JVM_IDENT_RE.match(pkg) for pkg in package_parts):
        return None
    if not _JVM_IDENT_RE.match(class_part) or not _JVM_METHOD_RE.match(method_part):
        return None

    allowed_stems = {file_stem, f"{file_stem}Kt"} if ext_lower in {".kt", ".kts"} else {file_stem}
    outer_name = class_part.split("$", 1)[0]
    if outer_name not in allowed_stems:
        return None

    return "/".join((*package_parts, filename))


def _match_repository_path_detailed(
    frame_path: str,
    repo_root: str,
    available_paths: set[str],
    frame_function: str | None = None,
) -> FrameMapping:
    cleaned = _clean_frame_path(frame_path)
    if not cleaned:
        return FrameMapping(
            status="unmapped",
            mapped_path=None,
            match_strategy="unmapped_not_found",
            candidate_count=0,
        )

    available_by_posix: dict[str, list[str]] = {}
    for raw_available in available_paths:
        norm_available = _normalize_relative_posix(raw_available)
        if norm_available is not None:
            available_by_posix.setdefault(norm_available, []).append(raw_available)

    def _resolve_unique_candidate(posix_key: str, strategy: str) -> FrameMapping:
        originals = available_by_posix.get(posix_key)
        if not originals:
            return FrameMapping(
                status="unmapped",
                mapped_path=None,
                match_strategy="unmapped_not_found",
                candidate_count=0,
            )
        if len(originals) != 1:
            return FrameMapping(
                status="unmapped",
                mapped_path=None,
                match_strategy="unmapped_ambiguous",
                candidate_count=len(originals),
            )
        if repo_root:
            candidate_fs_path = os.path.join(repo_root, *posix_key.split("/"))
            if not _is_physically_within_repo(repo_root, candidate_fs_path):
                return FrameMapping(
                    status="unmapped",
                    mapped_path=None,
                    match_strategy="unmapped_escape",
                    candidate_count=0,
                )
        return FrameMapping(
            status="mapped",
            mapped_path=originals[0],
            match_strategy=strategy,
            candidate_count=1,
        )

    is_windows_rooted = bool(_WINDOWS_ROOT_RE.match(cleaned))
    if os.path.isabs(cleaned) or is_windows_rooted:
        if is_windows_rooted and os.name != "nt":
            return FrameMapping(
                status="unmapped",
                mapped_path=None,
                match_strategy="unmapped_external",
                candidate_count=0,
            )
        if not repo_root or not _is_physically_within_repo(repo_root, cleaned):
            return FrameMapping(
                status="unmapped",
                mapped_path=None,
                match_strategy="unmapped_external",
                candidate_count=0,
            )
        real_root = os.path.realpath(repo_root)
        try:
            abs_cleaned = os.path.abspath(cleaned)
            parent_real = os.path.realpath(os.path.dirname(abs_cleaned))
            lexical_in_real_parent = os.path.join(parent_real, os.path.basename(abs_cleaned))
            if (
                lexical_in_real_parent != real_root
                and os.path.commonpath([lexical_in_real_parent, real_root]) == real_root
            ):
                parent_rel = _normalize_relative_posix(
                    os.path.relpath(lexical_in_real_parent, real_root)
                )
                if parent_rel is not None and parent_rel in available_by_posix:
                    return _resolve_unique_candidate(parent_rel, "resolved_in_repo")
        except ValueError:
            return FrameMapping(
                status="unmapped",
                mapped_path=None,
                match_strategy="unmapped_not_found",
                candidate_count=0,
            )
        try:
            real_frame = os.path.realpath(cleaned)
            real_rel = _normalize_relative_posix(os.path.relpath(real_frame, real_root))
        except ValueError:
            return FrameMapping(
                status="unmapped",
                mapped_path=None,
                match_strategy="unmapped_not_found",
                candidate_count=0,
            )
        if real_rel is not None and real_rel in available_by_posix:
            return _resolve_unique_candidate(real_rel, "resolved_in_repo")
        return FrameMapping(
            status="unmapped",
            mapped_path=None,
            match_strategy="unmapped_not_found",
            candidate_count=0,
        )

    normalized = _normalize_relative_posix(cleaned)
    if normalized is None:
        return FrameMapping(
            status="unmapped",
            mapped_path=None,
            match_strategy="unmapped_external",
            candidate_count=0,
        )

    if normalized in available_by_posix:
        return _resolve_unique_candidate(normalized, "exact_relpath")

    if repo_root:
        repo_names = (
            os.path.basename(os.path.normpath(os.path.abspath(repo_root))),
            os.path.basename(os.path.normpath(os.path.realpath(repo_root))),
        )
        for repo_name in dict.fromkeys(repo_names):
            if repo_name and normalized.startswith(repo_name + "/"):
                without_repo_prefix = normalized[len(repo_name) + 1 :]
                if without_repo_prefix in available_by_posix:
                    return _resolve_unique_candidate(without_repo_prefix, "repo_prefix_strip")

    if _has_external_env_segment(normalized):
        return FrameMapping(
            status="unmapped",
            mapped_path=None,
            match_strategy="unmapped_external",
            candidate_count=0,
        )

    if "/" in normalized:
        candidate = normalized
        strategy = "normalized_suffix"
    elif frame_function:
        candidate = _jvm_qualified_relpath(normalized, frame_function)
        if candidate is None:
            return FrameMapping(
                status="unmapped",
                mapped_path=None,
                match_strategy="unmapped_not_found",
                candidate_count=0,
            )
        strategy = "jvm_qualified"
    else:
        return FrameMapping(
            status="unmapped",
            mapped_path=None,
            match_strategy="unmapped_not_found",
            candidate_count=0,
        )

    matched_keys = [
        posix_key
        for posix_key in available_by_posix
        if posix_key == candidate or posix_key.endswith("/" + candidate)
    ]
    if not matched_keys:
        return FrameMapping(
            status="unmapped",
            mapped_path=None,
            match_strategy="unmapped_not_found",
            candidate_count=0,
        )
    if len(matched_keys) > 1:
        return FrameMapping(
            status="unmapped",
            mapped_path=None,
            match_strategy="unmapped_ambiguous",
            candidate_count=len(matched_keys),
        )
    return _resolve_unique_candidate(matched_keys[0], strategy)


def _match_repository_path(
    frame_path: str,
    repo_root: str,
    available_paths: set[str],
    frame_function: str | None = None,
) -> str | None:
    return _match_repository_path_detailed(
        frame_path,
        repo_root,
        available_paths,
        frame_function,
    ).mapped_path


def _graph_distances(graph: DependencyGraph, origins: set[str], maximum_distance: int) -> dict[str, int]:
    distances: dict[str, int] = {path: 0 for path in origins}
    queue = deque(origins)
    while queue:
        path = queue.popleft()
        distance = distances[path]
        if distance >= maximum_distance:
            continue
        neighbors = graph.dependencies(path) + graph.dependents(path)
        for neighbor in neighbors:
            if neighbor not in distances:
                distances[neighbor] = distance + 1
                queue.append(neighbor)
    return distances


@dataclass(frozen=True)
class _InspectedSourceFile:
    file_status: FileStatus
    file_exists: bool
    lines: tuple[str, ...] | None
    total_lines: int | None
    line_ending_style: LineEndingStyle
    detail: str | None


def _split_source_lines(text: str) -> tuple[list[str], LineEndingStyle]:
    """Split source file text strictly on \\r\\n, \\r, or \\n (never str.splitlines)."""
    if not text:
        return [], "none"
    endings = set(_SOURCE_LINE_BREAK_RE.findall(text))
    if not endings:
        style: LineEndingStyle = "none"
    elif endings == {"\n"}:
        style = "lf"
    elif endings == {"\r\n"}:
        style = "crlf"
    elif endings == {"\r"}:
        style = "cr"
    else:
        style = "mixed"
    parts = _SOURCE_LINE_BREAK_RE.split(text)
    if text.endswith(("\r\n", "\r", "\n")):
        parts = parts[:-1]
    return parts, style


def _inspect_repository_file(repo_root: str, rel_path: str) -> _InspectedSourceFile:
    norm_rel = _normalize_relative_posix(rel_path)
    if norm_rel is None:
        return _InspectedSourceFile(
            file_status="file_unreadable",
            file_exists=False,
            lines=None,
            total_lines=None,
            line_ending_style="none",
            detail="invalid relative repository path",
        )
    if not repo_root:
        return _InspectedSourceFile(
            file_status="file_missing",
            file_exists=False,
            lines=None,
            total_lines=None,
            line_ending_style="none",
            detail="repository root is not configured",
        )

    full_path = os.path.join(repo_root, *norm_rel.split("/"))
    if not _is_physically_within_repo(repo_root, full_path):
        return _InspectedSourceFile(
            file_status="file_unreadable",
            file_exists=False,
            lines=None,
            total_lines=None,
            line_ending_style="none",
            detail="path escapes repository root",
        )

    if not os.path.exists(full_path):
        return _InspectedSourceFile(
            file_status="file_missing",
            file_exists=False,
            lines=None,
            total_lines=None,
            line_ending_style="none",
            detail="mapped file does not exist on disk",
        )

    if not os.path.isfile(full_path):
        return _InspectedSourceFile(
            file_status="file_unreadable",
            file_exists=False,
            lines=None,
            total_lines=None,
            line_ending_style="none",
            detail="mapped path is not a regular file",
        )

    try:
        with open(full_path, "rb") as handle:
            raw_bytes = handle.read()
    except OSError as exc:
        return _InspectedSourceFile(
            file_status="file_unreadable",
            file_exists=True,
            lines=None,
            total_lines=None,
            line_ending_style="none",
            detail=f"mapped file could not be read: {exc}",
        )

    if b"\x00" in raw_bytes:
        return _InspectedSourceFile(
            file_status="file_unreadable",
            file_exists=True,
            lines=None,
            total_lines=None,
            line_ending_style="none",
            detail="mapped file appears to be binary",
        )

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return _InspectedSourceFile(
            file_status="file_unreadable",
            file_exists=True,
            lines=None,
            total_lines=None,
            line_ending_style="none",
            detail="mapped file is not valid UTF-8 text",
        )

    lines, line_ending_style = _split_source_lines(text)
    return _InspectedSourceFile(
        file_status="ok",
        file_exists=True,
        lines=tuple(lines),
        total_lines=len(lines),
        line_ending_style=line_ending_style,
        detail=None,
    )


def _validate_mapped_frame(
    reported_line: int,
    inspected: _InspectedSourceFile,
) -> FrameValidation:
    if inspected.file_status == "file_missing":
        return FrameValidation(
            status="file_missing",
            file_exists=False,
            total_lines=None,
            line_in_range=None,
            detail=inspected.detail,
        )
    if inspected.file_status == "file_unreadable":
        return FrameValidation(
            status="file_unreadable",
            file_exists=inspected.file_exists,
            total_lines=None,
            line_in_range=None,
            detail=inspected.detail,
        )

    total_lines = inspected.total_lines or 0
    if reported_line <= 0:
        return FrameValidation(
            status="line_non_positive",
            file_exists=True,
            total_lines=total_lines,
            line_in_range=False,
            detail=f"reported line {reported_line} must be >= 1",
        )
    if reported_line > total_lines:
        return FrameValidation(
            status="line_out_of_range",
            file_exists=True,
            total_lines=total_lines,
            line_in_range=False,
            detail=f"reported line {reported_line} exceeds file line count {total_lines}",
        )
    return FrameValidation(
        status="ok",
        file_exists=True,
        total_lines=total_lines,
        line_in_range=True,
        detail=None,
    )


_V8_ALIAS_RE = re.compile(r"\s+\[as\s+(?P<alias>[^\]]+)\]\s*$")


def _normalize_reported_function(reported_function: str | None) -> str | None:
    if reported_function is None:
        return None
    cleaned = reported_function.strip()
    if not cleaned:
        return None
    cleaned = _V8_ALIAS_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"^(?:(?:async|new)\s+)+", "", cleaned).strip()
    if "//" in cleaned:
        cleaned = cleaned.split("//", 1)[1].strip()
    if cleaned.endswith("::{{closure}}"):
        return "{{closure}}"
    if cleaned in _ANONYMOUS_OR_MODULE_NAMES:
        return cleaned
    if cleaned.startswith("<") and ">::" in cleaned:
        cleaned = cleaned.split(">::", 1)[1].strip()
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"\([^()]*\)$", "", cleaned).strip()
    if not cleaned:
        return None
    tokens = [tok for tok in re.split(r"::|->|\\|\.|/", cleaned) if tok]
    if not tokens:
        return cleaned
    return tokens[-1].strip() or None


def _coerce_symbol_list(raw_entry: Any) -> list[SourceSymbolInfo] | None:
    if raw_entry is None:
        return None
    if isinstance(raw_entry, dict) and "symbols" in raw_entry:
        raw_items = raw_entry["symbols"]
    elif hasattr(raw_entry, "symbols"):
        raw_items = raw_entry.symbols
    elif isinstance(raw_entry, (list, tuple, set, frozenset)):
        raw_items = raw_entry
    else:
        return None

    if raw_items is None:
        return None

    coerced: list[SourceSymbolInfo] = []
    for item in raw_items:
        if isinstance(item, SourceSymbolInfo):
            coerced.append(item)
        elif isinstance(item, dict):
            name = str(item.get("name") or "")
            qualified_name = str(item.get("qualified_name") or name)
            kind = str(item.get("kind") or "function")
            start_line = int(item["start_line"])
            end_line = int(item["end_line"])
            signature = item.get("signature")
            coerced.append(
                SourceSymbolInfo(
                    name=name,
                    qualified_name=qualified_name,
                    kind=kind,
                    start_line=start_line,
                    end_line=end_line,
                    signature=str(signature) if signature is not None else None,
                )
            )
        else:
            name = str(getattr(item, "name", "") or "")
            qualified_name = str(getattr(item, "qualified_name", "") or name)
            kind = str(getattr(item, "kind", "function") or "function")
            start_line = int(getattr(item, "start_line"))
            end_line = int(getattr(item, "end_line"))
            signature = getattr(item, "signature", None)
            coerced.append(
                SourceSymbolInfo(
                    name=name,
                    qualified_name=qualified_name,
                    kind=kind,
                    start_line=start_line,
                    end_line=end_line,
                    signature=str(signature) if signature is not None else None,
                )
            )
    return coerced


def _symbol_matches_function(
    symbol: SourceSymbolInfo,
    raw_function: str,
    normalized_function: str,
) -> bool:
    candidate_names = {normalized_function, raw_function.strip()}
    alias_match = _V8_ALIAS_RE.search(raw_function)
    if alias_match:
        alias_name = alias_match.group("alias").strip()
        if alias_name:
            candidate_names.add(alias_name)

    sym_norm = _normalize_reported_function(symbol.qualified_name) or symbol.name
    for candidate in candidate_names:
        if (
            candidate == symbol.name
            or candidate == symbol.qualified_name
            or candidate == sym_norm
            or symbol.qualified_name.endswith(f".{candidate}")
            or symbol.qualified_name.endswith(f"::{candidate}")
        ):
            return True
    return False


def _verify_frame_symbol(
    raw_frame: RepairFrame,
    mapping: FrameMapping,
    validation: FrameValidation,
    symbol_index: dict[str, Any] | None,
) -> FrameSymbolMatch:
    norm_fn = _normalize_reported_function(raw_frame.reported_function)
    if (
        mapping.status != "mapped"
        or mapping.mapped_path is None
        or validation.status != "ok"
        or symbol_index is None
        or mapping.mapped_path not in symbol_index
    ):
        return FrameSymbolMatch(
            status="unknown",
            reported_function=raw_frame.reported_function,
            normalized_reported_function=norm_fn,
            enclosing_symbol=None,
            detail=None,
        )

    file_symbols = _coerce_symbol_list(symbol_index[mapping.mapped_path])
    if file_symbols is None:
        return FrameSymbolMatch(
            status="unknown",
            reported_function=raw_frame.reported_function,
            normalized_reported_function=norm_fn,
            enclosing_symbol=None,
            detail=None,
        )

    spanning = [
        sym
        for sym in file_symbols
        if sym.start_line <= raw_frame.reported_line <= sym.end_line
    ]
    spanning.sort(
        key=lambda sym: (
            sym.end_line - sym.start_line,
            -sym.start_line,
            sym.qualified_name,
            sym.name,
        )
    )
    enclosing = spanning[0] if spanning else None
    raw_fn = raw_frame.reported_function

    if raw_fn is None or norm_fn is None:
        return FrameSymbolMatch(
            status="unverified",
            reported_function=raw_fn,
            normalized_reported_function=None,
            enclosing_symbol=enclosing,
            detail="frame does not report a function name",
        )

    if (
        norm_fn in _ANONYMOUS_OR_MODULE_NAMES
        or "$anonfun" in norm_fn
        or "$lambda" in norm_fn
    ):
        if enclosing is not None and enclosing.name == norm_fn:
            return FrameSymbolMatch(
                status="exact",
                reported_function=raw_fn,
                normalized_reported_function=norm_fn,
                enclosing_symbol=enclosing,
                detail=None,
            )
        if (
            norm_fn == "<module>"
            and enclosing is not None
            and enclosing.kind in {"function", "method"}
        ):
            return FrameSymbolMatch(
                status="mismatch",
                reported_function=raw_fn,
                normalized_reported_function=norm_fn,
                enclosing_symbol=enclosing,
                detail=(
                    f"reported <module> inside {enclosing.kind} "
                    f"{enclosing.qualified_name!r} at line {raw_frame.reported_line}"
                ),
            )
        if norm_fn == "{{closure}}" and enclosing is not None and "::{{closure}}" in raw_fn:
            outer_raw = raw_fn[: -len("::{{closure}}")]
            outer_norm = _normalize_reported_function(outer_raw)
            if outer_norm and not any(
                _symbol_matches_function(sym, outer_raw, outer_norm) for sym in spanning
            ):
                return FrameSymbolMatch(
                    status="mismatch",
                    reported_function=raw_fn,
                    normalized_reported_function=norm_fn,
                    enclosing_symbol=enclosing,
                    detail=(
                        f"closure parent {outer_norm!r} does not match symbol "
                        f"{enclosing.qualified_name!r} at line {raw_frame.reported_line}"
                    ),
                )
        return FrameSymbolMatch(
            status="anonymous_or_module",
            reported_function=raw_fn,
            normalized_reported_function=norm_fn,
            enclosing_symbol=enclosing,
            detail=None,
        )

    if enclosing is not None:
        if _symbol_matches_function(enclosing, raw_fn, norm_fn):
            return FrameSymbolMatch(
                status="exact",
                reported_function=raw_fn,
                normalized_reported_function=norm_fn,
                enclosing_symbol=enclosing,
                detail=None,
            )
        for outer_sym in spanning[1:]:
            if _symbol_matches_function(outer_sym, raw_fn, norm_fn):
                return FrameSymbolMatch(
                    status="container_match",
                    reported_function=raw_fn,
                    normalized_reported_function=norm_fn,
                    enclosing_symbol=enclosing,
                    detail=(
                        f"reported function {norm_fn!r} matches outer container "
                        f"{outer_sym.qualified_name!r} enclosing {enclosing.qualified_name!r}"
                    ),
                )
        return FrameSymbolMatch(
            status="mismatch",
            reported_function=raw_fn,
            normalized_reported_function=norm_fn,
            enclosing_symbol=enclosing,
            detail=(
                f"reported function {norm_fn!r} does not match symbol "
                f"{enclosing.qualified_name!r} at line {raw_frame.reported_line}"
            ),
        )

    if not file_symbols:
        return FrameSymbolMatch(
            status="unverified",
            reported_function=raw_fn,
            normalized_reported_function=norm_fn,
            enclosing_symbol=None,
            detail="no symbols indexed for file",
        )

    return FrameSymbolMatch(
        status="mismatch",
        reported_function=raw_fn,
        normalized_reported_function=norm_fn,
        enclosing_symbol=None,
        detail=(
            f"no symbol spans line {raw_frame.reported_line} "
            f"for reported function {norm_fn!r}"
        ),
    )


def _read_excerpt(repo_root: str, path: str, line: int, radius: int = 2) -> str:
    inspected = _inspect_repository_file(repo_root, path)
    if inspected.file_status != "ok" or not inspected.lines:
        return ""
    lines = inspected.lines
    if line < 1 or line > len(lines):
        return ""
    start = max(line - radius, 1)
    end = min(line + radius, len(lines))
    return "\n".join(f"{number}: {lines[number - 1]}" for number in range(start, end + 1))


def _frame_priority_key(
    frame: RepairFrame,
    file_rank: int = 0,
) -> tuple[int, int, int, int, int]:
    innermost_rank = 0 if frame.is_innermost_valid is True else 1
    kind_rank = 0 if frame.frame_kind == "call" else 1
    if frame.call_depth_from_innermost is not None:
        depth_rank = frame.call_depth_from_innermost
    elif frame.frame_order == "outer_to_inner":
        depth_rank = -frame.segment_position
    elif frame.frame_order == "inner_to_outer":
        depth_rank = frame.segment_position
    else:
        depth_rank = frame.frame_index
    return (innermost_rank, kind_rank, depth_rank, file_rank, frame.frame_index)


def _format_excerpt_content(
    lines: Sequence[str],
    start_line: int,
    end_line: int,
    target_lines: list[int],
) -> str:
    target_set = set(target_lines)
    width = max(4, len(str(end_line)))
    rendered: list[str] = []
    for number in range(start_line, end_line + 1):
        prefix = "> " if number in target_set else "  "
        rendered.append(f"{prefix}{number:>{width}}: {lines[number - 1]}")
    return "\n".join(rendered)


def _select_enclosing_symbol_for_window(
    frame_indices: list[int],
    context_frames: list[RepairFrame],
    file_rank: int,
) -> str | None:
    if not frame_indices:
        return None
    ordered_indices = sorted(
        frame_indices,
        key=lambda idx: _frame_priority_key(context_frames[idx], file_rank),
    )
    for idx in ordered_indices:
        enclosing = context_frames[idx].symbol_match.enclosing_symbol
        if enclosing is not None and enclosing.qualified_name:
            return enclosing.qualified_name
    return None


def _build_file_excerpts(
    path: str,
    lines: Sequence[str],
    valid_frame_lines: list[int],
    path_frame_indices: list[int],
    context_frames: list[RepairFrame],
    line_radii: dict[int, int],
    initial_radius: int,
    file_rank: int,
) -> list[CodeExcerpt]:
    total_lines = len(lines)
    if total_lines <= 0 or not valid_frame_lines:
        return []

    valid_indices = [
        idx
        for idx in path_frame_indices
        if context_frames[idx].validation.status == "ok"
    ]
    if not valid_indices:
        return []

    raw_windows: list[dict[str, Any]] = []
    for target_line in sorted(set(valid_frame_lines)):
        if target_line not in line_radii:
            continue
        if target_line < 1 or target_line > total_lines:
            continue
        radius = max(0, line_radii[target_line])
        start_line = max(1, target_line - radius)
        end_line = min(total_lines, target_line + radius)
        line_frame_indices = [
            idx
            for idx in valid_indices
            if context_frames[idx].reported_line == target_line
        ]
        raw_windows.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "target_lines": [target_line],
                "frame_indices": list(line_frame_indices),
                "radii": [radius],
            }
        )

    if not raw_windows:
        return []

    raw_windows.sort(key=lambda item: (item["start_line"], item["end_line"]))
    merged_windows: list[dict[str, Any]] = [raw_windows[0]]
    for candidate in raw_windows[1:]:
        current = merged_windows[-1]
        if candidate["start_line"] <= current["end_line"] + 1:
            current["start_line"] = min(current["start_line"], candidate["start_line"])
            current["end_line"] = max(current["end_line"], candidate["end_line"])
            current["target_lines"] = sorted(
                set(current["target_lines"]) | set(candidate["target_lines"])
            )
            current["frame_indices"] = sorted(
                set(current["frame_indices"]) | set(candidate["frame_indices"])
            )
            current["radii"].extend(candidate["radii"])
        else:
            merged_windows.append(candidate)

    merged_windows.sort(key=lambda item: (item["start_line"], item["end_line"]))
    excerpts: list[CodeExcerpt] = []
    for ordinal, win in enumerate(merged_windows, start=1):
        start_line = int(win["start_line"])
        end_line = int(win["end_line"])
        target_lines = list(win["target_lines"])
        frame_indices = list(win["frame_indices"])
        radii = list(win["radii"])
        effective_radius = min(radii) if radii else 0
        context_trimmed = any(r < initial_radius for r in radii)
        enclosing_symbol = _select_enclosing_symbol_for_window(
            frame_indices,
            context_frames,
            file_rank,
        )
        content = _format_excerpt_content(lines, start_line, end_line, target_lines)
        excerpts.append(
            CodeExcerpt(
                excerpt_id=f"{path}#{ordinal}",
                role="frame",
                start_line=start_line,
                end_line=end_line,
                target_lines=target_lines,
                frame_indices=frame_indices,
                enclosing_symbol=enclosing_symbol,
                effective_radius=effective_radius,
                clipped_top=start_line > 1,
                clipped_bottom=end_line < total_lines,
                context_trimmed=context_trimmed,
                content=content,
            )
        )
    return excerpts


def _select_primary_excerpt(
    excerpts: list[CodeExcerpt],
    context_frames: list[RepairFrame],
    file_rank: int,
) -> CodeExcerpt | None:
    if not excerpts:
        return None

    def _excerpt_priority(excerpt: CodeExcerpt) -> tuple[tuple[int, int, int, int, int], int]:
        if excerpt.frame_indices:
            best_frame_key = min(
                _frame_priority_key(context_frames[idx], file_rank)
                for idx in excerpt.frame_indices
            )
        else:
            best_frame_key = (1, 1, excerpt.start_line, file_rank, excerpt.start_line)
        return (best_frame_key, excerpt.start_line)

    return min(excerpts, key=_excerpt_priority)


def _fit_excerpts_to_budget(
    ranked_paths: list[str],
    inspected_by_path: dict[str, _InspectedSourceFile],
    valid_lines_by_path: dict[str, list[int]],
    frame_indices_by_path: dict[str, list[int]],
    context_frames: list[RepairFrame],
    *,
    excerpt_radius: int,
    max_excerpt_chars: int,
) -> tuple[
    dict[str, list[CodeExcerpt]],
    dict[str, list[str]],
    ExcerptCharTruncation,
]:
    initial_radius = max(0, excerpt_radius)
    limit_chars = max(0, max_excerpt_chars)
    file_rank_by_path = {path: rank_idx for rank_idx, path in enumerate(ranked_paths)}

    line_radii_by_path: dict[str, dict[int, int]] = {}
    valid_line_frames_by_path: dict[str, dict[int, list[int]]] = {}
    excerpts_by_path: dict[str, list[CodeExcerpt]] = {}
    omitted_ids_by_path: dict[str, list[str]] = {path: [] for path in ranked_paths}

    for path in ranked_paths:
        inspected = inspected_by_path[path]
        valid_lines = valid_lines_by_path.get(path, [])
        path_indices = frame_indices_by_path.get(path, [])
        if (
            inspected.file_status != "ok"
            or not inspected.lines
            or (inspected.total_lines or 0) <= 0
            or not valid_lines
        ):
            line_radii_by_path[path] = {}
            valid_line_frames_by_path[path] = {}
            excerpts_by_path[path] = []
            continue

        valid_indices = [
            idx
            for idx in path_indices
            if context_frames[idx].validation.status == "ok"
        ]
        line_to_frames: dict[int, list[int]] = {}
        for line_no in valid_lines:
            line_to_frames[line_no] = [
                idx
                for idx in valid_indices
                if context_frames[idx].reported_line == line_no
            ]
        valid_line_frames_by_path[path] = line_to_frames
        line_radii_by_path[path] = {line_no: initial_radius for line_no in valid_lines}
        excerpts_by_path[path] = _build_file_excerpts(
            path,
            inspected.lines,
            valid_lines,
            path_indices,
            context_frames,
            line_radii_by_path[path],
            initial_radius,
            file_rank_by_path[path],
        )

    def _target_line_priority(path: str, line_no: int) -> tuple[tuple[int, int, int, int, int], int, int]:
        file_rank = file_rank_by_path[path]
        f_indices = valid_line_frames_by_path.get(path, {}).get(line_no, [])
        if f_indices:
            best_key = min(
                _frame_priority_key(context_frames[idx], file_rank)
                for idx in f_indices
            )
        else:
            best_key = (1, 1, line_no, file_rank, line_no)
        return (best_key, file_rank, line_no)

    def _total_chars() -> int:
        return sum(
            len(ex.content)
            for path in ranked_paths
            for ex in excerpts_by_path.get(path, [])
        )

    # Phase A: Trim surrounding context radius down toward 0 (lowest-priority targets first)
    # before ever removing any valid target line.
    if _total_chars() > limit_chars and initial_radius > 0:
        all_targets: list[tuple[str, int]] = [
            (path, line_no)
            for path in ranked_paths
            for line_no in line_radii_by_path.get(path, {})
        ]
        # Reverse priority order: trim lowest-priority target lines first, innermost valid last.
        targets_low_to_high = sorted(
            all_targets,
            key=lambda item: _target_line_priority(item[0], item[1]),
            reverse=True,
        )
        for target_radius in range(initial_radius - 1, -1, -1):
            for path, line_no in targets_low_to_high:
                if line_radii_by_path[path][line_no] > target_radius:
                    line_radii_by_path[path][line_no] = target_radius
                    inspected = inspected_by_path[path]
                    excerpts_by_path[path] = _build_file_excerpts(
                        path,
                        inspected.lines or [],
                        valid_lines_by_path[path],
                        frame_indices_by_path[path],
                        context_frames,
                        line_radii_by_path[path],
                        initial_radius,
                        file_rank_by_path[path],
                    )
                    if _total_chars() <= limit_chars:
                        break
            if _total_chars() <= limit_chars:
                break

    omitted_target_records: list[OmittedTargetLine] = []

    # Phase B: Only if 0-radius target lines still exceed limit_chars, retain highest-priority
    # 0-radius windows/target lines that fit and record omitted windows & target lines.
    if _total_chars() > limit_chars:
        zero_radius_candidates: list[tuple[str, CodeExcerpt]] = [
            (path, ex)
            for path in ranked_paths
            for ex in excerpts_by_path.get(path, [])
        ]

        def _window_priority(item: tuple[str, CodeExcerpt]) -> tuple[tuple[int, int, int, int, int], int, int]:
            path, ex = item
            file_rank = file_rank_by_path[path]
            if ex.frame_indices:
                best_key = min(
                    _frame_priority_key(context_frames[idx], file_rank)
                    for idx in ex.frame_indices
                )
            else:
                best_key = (1, 1, ex.start_line, file_rank, ex.start_line)
            return (best_key, file_rank, ex.start_line)

        candidates_high_to_low = sorted(zero_radius_candidates, key=_window_priority)
        retained_by_path: dict[str, list[CodeExcerpt]] = {path: [] for path in ranked_paths}
        used_chars = 0

        for path, ex in candidates_high_to_low:
            cost = len(ex.content)
            if used_chars + cost <= limit_chars:
                retained_by_path[path].append(ex)
                used_chars += cost
                continue

            inspected = inspected_by_path[path]
            lines = inspected.lines or []
            total_lines = len(lines)
            file_rank = file_rank_by_path[path]

            if len(ex.target_lines) > 1 and total_lines > 0:
                sorted_targets = sorted(
                    ex.target_lines,
                    key=lambda line_no: _target_line_priority(path, line_no),
                )
                kept_lines: list[int] = []
                kept_content = ""
                for line_no in sorted_targets:
                    trial_lines = sorted(kept_lines + [line_no])
                    if trial_lines[-1] - trial_lines[0] + 1 != len(trial_lines):
                        continue
                    trial_content = _format_excerpt_content(
                        lines,
                        trial_lines[0],
                        trial_lines[-1],
                        trial_lines,
                    )
                    if used_chars + len(trial_content) <= limit_chars:
                        kept_lines = trial_lines
                        kept_content = trial_content

                if kept_lines:
                    kept_set = set(kept_lines)
                    kept_frame_indices = [
                        idx
                        for idx in ex.frame_indices
                        if context_frames[idx].reported_line in kept_set
                    ]
                    enclosing_symbol = _select_enclosing_symbol_for_window(
                        kept_frame_indices,
                        context_frames,
                        file_rank,
                    )
                    trimmed_ex = CodeExcerpt(
                        excerpt_id=ex.excerpt_id,
                        role="frame",
                        start_line=kept_lines[0],
                        end_line=kept_lines[-1],
                        target_lines=kept_lines,
                        frame_indices=kept_frame_indices,
                        enclosing_symbol=enclosing_symbol,
                        effective_radius=0,
                        clipped_top=kept_lines[0] > 1,
                        clipped_bottom=kept_lines[-1] < total_lines,
                        context_trimmed=True,
                        content=kept_content,
                    )
                    retained_by_path[path].append(trimmed_ex)
                    used_chars += len(kept_content)
                    for dropped_line in sorted(set(ex.target_lines) - kept_set):
                        f_indices = valid_line_frames_by_path.get(path, {}).get(dropped_line, [])
                        primary_f_idx = f_indices[0] if f_indices else -1
                        omitted_target_records.append(
                            OmittedTargetLine(
                                path=path,
                                line=dropped_line,
                                frame_index=primary_f_idx,
                                frame_indices=list(f_indices),
                                excerpt_id=ex.excerpt_id,
                            )
                        )
                    continue

            omitted_ids_by_path[path].append(ex.excerpt_id)
            for dropped_line in ex.target_lines:
                f_indices = valid_line_frames_by_path.get(path, {}).get(dropped_line, [])
                primary_f_idx = f_indices[0] if f_indices else -1
                omitted_target_records.append(
                    OmittedTargetLine(
                        path=path,
                        line=dropped_line,
                        frame_index=primary_f_idx,
                        frame_indices=list(f_indices),
                        excerpt_id=ex.excerpt_id,
                    )
                )

        for path in ranked_paths:
            retained_by_path[path].sort(
                key=lambda item: (item.start_line, item.end_line, item.excerpt_id)
            )
            omitted_ids_by_path[path].sort(
                key=lambda eid: int(eid.rsplit("#", 1)[1]) if "#" in eid else 0
            )
        excerpts_by_path = retained_by_path
        omitted_target_records.sort(
            key=lambda item: (
                file_rank_by_path.get(item.path, 0),
                item.line,
                item.frame_index,
            )
        )

    actual_chars = _total_chars()
    context_trimmed_excerpt_ids = [
        ex.excerpt_id
        for path in ranked_paths
        for ex in excerpts_by_path.get(path, [])
        if ex.context_trimmed
    ]
    all_omitted_excerpt_ids = [
        eid
        for path in ranked_paths
        for eid in omitted_ids_by_path.get(path, [])
    ]
    target_lines_omitted = len(omitted_target_records) > 0
    truncated = bool(
        context_trimmed_excerpt_ids
        or all_omitted_excerpt_ids
        or target_lines_omitted
    )

    excerpt_truncation = ExcerptCharTruncation(
        truncated=truncated,
        limit_chars=limit_chars,
        actual_chars=actual_chars,
        context_trimmed_excerpt_ids=context_trimmed_excerpt_ids,
        omitted_excerpt_ids=all_omitted_excerpt_ids,
        target_lines_omitted=target_lines_omitted,
        omitted_target_lines=omitted_target_records,
    )
    return excerpts_by_path, omitted_ids_by_path, excerpt_truncation


def build_repair_context(
    incident: Incident,
    graph: DependencyGraph,
    profiles: dict[str, FileProfile],
    repo_root: str,
    *,
    top_n: int = 10,
    maximum_graph_distance: int = 2,
    symbol_index: dict[str, Any] | None = None,
    excerpt_radius: int = 8,
    max_excerpt_chars: int = 12000,
) -> RepairContext:
    available_paths = set(profiles)
    parsed_trace = extract_stack_frames_detailed(incident.stack_trace)

    mapped: dict[str, list[int]] = {}
    valid_mapped: dict[str, list[int]] = {}
    invalid_mapped: dict[str, list[int]] = {}
    mapped_frame_indices: dict[str, list[int]] = {}
    unmapped: list[IncidentFrame] = []
    frame_mappings: list[FrameMapping] = []
    frame_validations: list[FrameValidation] = []
    frame_symbol_matches: list[FrameSymbolMatch] = []
    inspected_files: dict[str, _InspectedSourceFile] = {}

    def _get_inspected_file(rel_path: str) -> _InspectedSourceFile:
        if rel_path not in inspected_files:
            inspected_files[rel_path] = _inspect_repository_file(repo_root, rel_path)
        return inspected_files[rel_path]

    for raw_frame in parsed_trace.frames:
        mapping = _match_repository_path_detailed(
            raw_frame.reported_path,
            repo_root,
            available_paths,
            raw_frame.reported_function or "",
        )
        frame_mappings.append(mapping)
        if mapping.mapped_path is None:
            unmapped.append(
                IncidentFrame(
                    path=raw_frame.reported_path,
                    line=raw_frame.reported_line,
                    function=raw_frame.reported_function or "",
                )
            )
            validation = FrameValidation(status="unmapped")
            symbol_match = FrameSymbolMatch(
                status="unknown",
                reported_function=raw_frame.reported_function,
                normalized_reported_function=_normalize_reported_function(
                    raw_frame.reported_function
                ),
            )
        else:
            mapped_path = mapping.mapped_path
            inspected = _get_inspected_file(mapped_path)
            validation = _validate_mapped_frame(raw_frame.reported_line, inspected)
            symbol_match = _verify_frame_symbol(
                raw_frame,
                mapping,
                validation,
                symbol_index,
            )
            mapped.setdefault(mapped_path, []).append(raw_frame.reported_line)
            mapped_frame_indices.setdefault(mapped_path, []).append(raw_frame.frame_index)
            if validation.status == "ok":
                valid_mapped.setdefault(mapped_path, []).append(raw_frame.reported_line)
            else:
                invalid_mapped.setdefault(mapped_path, []).append(raw_frame.reported_line)

        frame_validations.append(validation)
        frame_symbol_matches.append(symbol_match)

    is_innermost_mapped_by_idx: dict[int, bool | None] = {}
    is_innermost_valid_by_idx: dict[int, bool | None] = {}
    context_segments: list[TraceSegment] = []
    for seg in parsed_trace.trace_segments:
        if seg.frame_order == "unknown":
            innermost_mapped_idx: int | None = None
            innermost_valid_idx: int | None = None
            for idx in seg.frame_indices:
                is_innermost_mapped_by_idx[idx] = None
                is_innermost_valid_by_idx[idx] = None
        else:
            mapped_call_indices = [
                idx
                for idx in seg.frame_indices
                if parsed_trace.frames[idx].frame_kind == "call"
                and frame_mappings[idx].status == "mapped"
            ]
            if mapped_call_indices:
                innermost_mapped_idx = (
                    mapped_call_indices[-1]
                    if seg.frame_order == "outer_to_inner"
                    else mapped_call_indices[0]
                )
                for idx in seg.frame_indices:
                    is_innermost_mapped_by_idx[idx] = idx == innermost_mapped_idx
            else:
                innermost_mapped_idx = None
                for idx in seg.frame_indices:
                    is_innermost_mapped_by_idx[idx] = False

            valid_call_indices = [
                idx
                for idx in mapped_call_indices
                if frame_validations[idx].status == "ok"
            ]
            if valid_call_indices:
                innermost_valid_idx = (
                    valid_call_indices[-1]
                    if seg.frame_order == "outer_to_inner"
                    else valid_call_indices[0]
                )
                for idx in seg.frame_indices:
                    is_innermost_valid_by_idx[idx] = idx == innermost_valid_idx
            else:
                innermost_valid_idx = None
                for idx in seg.frame_indices:
                    is_innermost_valid_by_idx[idx] = False

        context_segments.append(
            replace(
                seg,
                innermost_mapped_frame_index=innermost_mapped_idx,
                innermost_valid_frame_index=innermost_valid_idx,
            )
        )

    context_frames: list[RepairFrame] = [
        replace(
            raw_frame,
            is_innermost_mapped=is_innermost_mapped_by_idx.get(raw_frame.frame_index),
            is_innermost_valid=is_innermost_valid_by_idx.get(raw_frame.frame_index),
            mapping=frame_mappings[raw_frame.frame_index],
            validation=frame_validations[raw_frame.frame_index],
            symbol_match=frame_symbol_matches[raw_frame.frame_index],
        )
        for raw_frame in parsed_trace.frames
    ]

    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for path, lines in mapped.items():
        valid_lines_for_path = sorted(set(valid_mapped.get(path, [])))
        invalid_lines_for_path = sorted(set(invalid_mapped.get(path, [])))
        path_frames = [
            context_frames[idx]
            for idx in mapped_frame_indices.get(path, [])
            if 0 <= idx < len(context_frames)
        ]
        valid_path_frames = [
            frame for frame in path_frames if frame.validation.status == "ok"
        ]
        if valid_lines_for_path:
            base_score = 100.0
            if any(frame.is_innermost_valid is True for frame in valid_path_frames):
                base_score += 20.0
            valid_depths = [
                frame.call_depth_from_innermost
                for frame in valid_path_frames
                if frame.call_depth_from_innermost is not None
                and frame.call_depth_from_innermost >= 0
            ]
            if valid_depths:
                base_score += 15.0 / (1.0 + min(valid_depths))
            scores[path] = base_score
        else:
            scores[path] = 40.0
        path_reasons = [f"stack frame at line {line}" for line in valid_lines_for_path]
        path_reasons.extend(
            f"invalid stack frame at line {line}" for line in invalid_lines_for_path
        )
        reasons[path] = path_reasons

    for path, distance in _graph_distances(graph, set(mapped), maximum_graph_distance).items():
        if distance == 0:
            continue
        scores[path] = scores.get(path, 0.0) + 30.0 / distance
        reasons.setdefault(path, []).append(f"dependency graph distance {distance} from stack frame")

    query = " ".join(part for part in (incident.exception_type, incident.message) if part)
    if query:
        for match in query_codebase(profiles, query, top_n=top_n):
            path = match.profile.file
            scores[path] = scores.get(path, 0.0) + match.final_score / 5.0
            reasons.setdefault(path, []).extend(match.match_reasons)

    valid_frame_origins = {path for path, lines in valid_mapped.items() if lines}
    valid_graph_distances = _graph_distances(
        graph,
        valid_frame_origins,
        maximum_graph_distance,
    )

    ranked = sorted(scores, key=lambda path: (-scores[path], path))[:top_n]
    inspected_by_ranked: dict[str, _InspectedSourceFile] = {}
    valid_lines_by_ranked: dict[str, list[int]] = {}
    invalid_lines_by_ranked: dict[str, list[int]] = {}
    frame_indices_by_ranked: dict[str, list[int]] = {}
    for path in ranked:
        inspected_by_ranked[path] = _get_inspected_file(path)
        valid_lines_by_ranked[path] = sorted(set(valid_mapped.get(path, [])))
        invalid_lines_by_ranked[path] = sorted(set(invalid_mapped.get(path, [])))
        frame_indices_by_ranked[path] = list(mapped_frame_indices.get(path, []))

    excerpts_by_path, omitted_ids_by_path, excerpt_truncation = _fit_excerpts_to_budget(
        ranked,
        inspected_by_ranked,
        valid_lines_by_ranked,
        frame_indices_by_ranked,
        context_frames,
        excerpt_radius=excerpt_radius,
        max_excerpt_chars=max_excerpt_chars,
    )
    recent_commits_by_path = get_recent_commits_for_files(repo_root, ranked)

    evidence: list[RepairEvidence] = []
    for file_rank, path in enumerate(ranked):
        profile = profiles[path]
        inspected = inspected_by_ranked[path]
        valid_frame_lines = valid_lines_by_ranked[path]
        invalid_frame_lines = invalid_lines_by_ranked[path]
        frame_lines = list(valid_frame_lines)
        symbols = profile.functions + profile.classes + profile.interfaces + profile.structs

        path_frame_indices = frame_indices_by_ranked[path]
        if symbol_index is None or inspected.file_status != "ok":
            symbol_verification: SymbolVerificationStatus = "unavailable"
        elif path not in symbol_index or _coerce_symbol_list(symbol_index[path]) is None:
            symbol_verification = "not_indexed"
        elif path_frame_indices and all(
            frame_validations[idx].status == "ok"
            and frame_symbol_matches[idx].status
            in {"exact", "container_match", "anonymous_or_module"}
            for idx in path_frame_indices
        ):
            symbol_verification = "verified"
        else:
            symbol_verification = "unavailable"

        if inspected.file_status != "ok":
            patch_eligible = False
            patch_eligibility_reason = "ineligible_file_not_readable"
        elif valid_frame_lines and (inspected.total_lines or 0) > 0:
            patch_eligible = True
            patch_eligibility_reason = "eligible_valid_stack_frame"
        elif invalid_frame_lines:
            patch_eligible = False
            patch_eligibility_reason = "ineligible_no_valid_frames"
        elif (inspected.total_lines or 0) <= 0:
            patch_eligible = False
            patch_eligibility_reason = "ineligible_empty_file"
        elif valid_graph_distances.get(path, 0) > 0:
            patch_eligible = True
            patch_eligibility_reason = "eligible_direct_graph_neighbor"
        else:
            patch_eligible = False
            patch_eligibility_reason = "ineligible_query_match_only"

        if inspected.file_status == "ok" and valid_frame_lines and (inspected.total_lines or 0) > 0:
            evidence_tier: EvidenceTier = "confirmed"
        elif inspected.file_status == "ok" and valid_graph_distances.get(path, 0) > 0 and not invalid_frame_lines:
            evidence_tier = "corroborated"
        else:
            evidence_tier = "suspected"

        file_excerpts = excerpts_by_path.get(path, [])
        primary_excerpt = _select_primary_excerpt(
            file_excerpts,
            context_frames,
            file_rank,
        )
        primary_excerpt_id = primary_excerpt.excerpt_id if primary_excerpt is not None else None
        legacy_excerpt = primary_excerpt.content if primary_excerpt is not None else ""
        omitted_excerpt_ids = list(omitted_ids_by_path.get(path, []))

        file_commits = recent_commits_by_path.get(path, [])
        imported_by_list = list(getattr(profile, "imported_by", []) or [])
        trans_list = list(getattr(profile, "transitive_consumers", []) or [])
        affected_components = list(dict.fromkeys(imported_by_list[:8] + trans_list[:8]))[:10]

        file_recent_changes = getattr(profile, "recent_changes", 0) or 0
        file_direct_deps = getattr(profile, "direct_impact", 0) or 0
        file_trans_deps = getattr(profile, "transitive_impact", 0) or 0
        file_db = list(getattr(profile, "db_interactions", []) or [])
        file_svc = list(getattr(profile, "service_calls", []) or [])
        file_cfg = list(getattr(profile, "config_keys", []) or [])

        factors: list[str] = []
        if file_commits:
            latest = file_commits[0]
            factors.append(
                f"Recent commit activity on {path}: [{latest['sha']}] {latest['message']} ({latest['author']}, {latest['date']})"
            )
        elif file_recent_changes > 0:
            factors.append(f"Recent code churn: {file_recent_changes} commit(s) in the last 90 days")
        if file_trans_deps >= 5 or file_direct_deps >= 3:
            factors.append(
                f"High downstream blast radius: {file_direct_deps} direct and {file_trans_deps} transitive dependent module(s)"
            )
        if file_db:
            factors.append(f"Database interaction surface: {', '.join(file_db[:3])}")
        if file_svc:
            factors.append(f"External service dependency: {', '.join(file_svc[:3])}")
        if file_cfg:
            factors.append(f"Configuration/env data-flow dependency: {', '.join(file_cfg[:4])}")

        evidence.append(
            RepairEvidence(
                path=path,
                score=scores[path],
                reasons=list(dict.fromkeys(reasons.get(path, []))),
                frame_lines=frame_lines,
                symbols=symbols[:25],
                excerpt=legacy_excerpt,
                frame_indices=path_frame_indices,
                valid_frame_lines=valid_frame_lines,
                invalid_frame_lines=invalid_frame_lines,
                excerpts=file_excerpts,
                primary_excerpt_id=primary_excerpt_id,
                omitted_excerpt_ids=omitted_excerpt_ids,
                file_status=inspected.file_status,
                total_lines=inspected.total_lines,
                line_ending_style=inspected.line_ending_style,
                symbol_verification=symbol_verification,
                patch_eligible=patch_eligible,
                patch_eligibility_reason=patch_eligibility_reason,
                evidence_tier=evidence_tier,
                risk_score=getattr(profile, "risk_score", 0) or 0,
                risk_label=getattr(profile, "risk_label", "Low") or "Low",
                direct_dependents=file_direct_deps,
                transitive_dependents=file_trans_deps,
                recent_changes=file_recent_changes,
                last_modified=getattr(profile, "last_modified", "") or "",
                calls=list(getattr(profile, "calls", []) or [])[:25],
                api_endpoints=list(getattr(profile, "api_endpoints", []) or []),
                db_interactions=file_db,
                service_calls=file_svc,
                config_keys=file_cfg,
                related_commits=file_commits,
                affected_components=affected_components,
                contributing_factors=factors,
            )
        )
    return RepairContext(
        incident=incident,
        repository_path=repo_root,
        evidence=evidence,
        unmapped_frames=unmapped,
        frames=context_frames,
        trace_segments=context_segments,
        truncation=ContextTruncation(excerpt_chars=excerpt_truncation),
    )


_EXCEPTION_LINE_RE = re.compile(
    r"^\s*(?:(?P<exc>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Panic|Fault|Failure|Interrupt|Exit|Iteration|Violation|Timeout|Abort|Crash)):\s*(?P<msg>.+)|"
    r"(?P<panic_kw>panic|fatal error|fatal|FAIL):\s*(?P<panic_msg>.+)|"
    r"(?P<generic_exc>(?!File\b|at\b|http\b|https\b|Caused\b|Traceback\b)[A-Z][A-Za-z0-9_.]*):\s*(?P<generic_msg>.+))$"
)
_DETECT_FORBIDDEN_RE = re.compile(r"[;&|<>`$\r\n]")


def detect_incident_from_command(
    repo_root: str,
    command: str,
    *,
    incident_id: str = "detected-incident",
    timeout: int = 120,
) -> Incident:
    """
    Stage 1 (Detect): Execute a command or test runner against repo_root,
    detect failures/tracebacks, and construct a normalized Incident.
    """
    if not command or not command.strip() or _DETECT_FORBIDDEN_RE.search(command):
        raise ValueError(f"unsafe or empty detection command: {command!r}")
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"malformed detection command: {exc}") from exc
    if not tokens:
        raise ValueError("detection command must contain at least one token")

    exec_tokens = list(tokens)
    if exec_tokens[0] in {"python", "python3", "venv/bin/python"}:
        exec_tokens[0] = sys.executable

    root_path = Path(repo_root).resolve()
    src_dir = root_path / "src"
    py_path = f"{src_dir}:{root_path}" if src_dir.is_dir() else str(root_path)
    env = {**os.environ, "PYTHONPATH": py_path}

    try:
        completed = subprocess.run(
            exec_tokens,
            cwd=str(root_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        combined = "\n".join(part for part in (out, err) if part).strip()
        return Incident(
            incident_id=incident_id,
            message=f"Command timed out after {timeout}s: {command}",
            stack_trace=combined,
            exception_type="TimeoutError",
            metadata={"detected_by_command": command, "verification_test": command, "timed_out": True},
        )
    except OSError as exc:
        raise ValueError(f"failed to execute detection command {command!r}: {exc}") from exc

    combined_output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part and part.strip()
    )
    parsed_frames = extract_stack_frames(combined_output)
    if completed.returncode == 0 and not parsed_frames:
        raise ValueError(f"command succeeded with exit code 0; no failure detected: {command!r}")

    exc_type = ""
    exc_message = ""
    for line in reversed(combined_output.splitlines()):
        match = _EXCEPTION_LINE_RE.match(line.strip())
        if match:
            exc_type = (
                match.group("exc")
                or match.group("panic_kw")
                or match.group("generic_exc")
                or ""
            ).strip()
            exc_message = (
                match.group("msg")
                or match.group("panic_msg")
                or match.group("generic_msg")
                or ""
            ).strip()
            break

    if not exc_message:
        non_empty = [ln.strip() for ln in combined_output.splitlines() if ln.strip()]
        exc_message = non_empty[-1] if non_empty else f"Command failed with exit code {completed.returncode}"
    if not exc_type:
        exc_type = "CommandFailure"

    return Incident(
        incident_id=incident_id,
        message=exc_message,
        stack_trace=combined_output,
        exception_type=exc_type,
        metadata={
            "detected_by_command": command,
            "verification_test": command,
            "returncode": completed.returncode,
        },
    )
