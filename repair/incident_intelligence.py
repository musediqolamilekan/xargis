"""
Incident Intelligence & Alert Correlation Engine (Section A).

Connects to and normalizes telemetry from:
  - Error Monitoring:   Sentry, Datadog, New Relic
  - Observability:      OpenTelemetry (OTLP spans/logs), Prometheus / Grafana Alertmanager
  - Log Analysis:       Structured JSONL logs and raw multi-line application log files
  - Distributed Traces: Correlates events sharing trace_id / span_id across services

Then performs:
  1. Incident Grouping (fingerprinting & deduplicating 500+ raw alerts into symptom groups)
  2. Anomaly Detection (identifying error storms, cross-service cascades, and regressions)
  3. Alert Correlation (using the Code Dependency Graph + Distributed Trace IDs to link
     multiple symptom alerts to a single shared underlying root-cause file)
  4. Incident Prioritisation (ranking root-cause clusters by business + technical impact)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from intelligence.dependency_graph import DependencyGraph
from intelligence.file_profile import FileProfile
from .incident_mapper import (
    Incident,
    _EXT_GROUP,
    _match_repository_path_detailed,
    extract_stack_frames,
)


_VOLATILE_TOKEN_RE = re.compile(
    r"\b(?:0x[0-9a-fA-F]+|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|\d{2,})\b"
)
_LOG_HEADER_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\s*"
    r"(?:\[(?P<level1>ERROR|CRITICAL|FATAL|WARN|WARNING|EXCEPTION)\]|(?P<level2>ERROR|CRITICAL|FATAL|WARN|WARNING|EXCEPTION))\s*"
    r"(?:\[(?P<service>[^\]]+)\])?\s*:?\s*(?P<msg>.*)$",
    re.IGNORECASE,
)
_TRACE_KV_RE = re.compile(r"\b(?:trace_id|traceId|request_id|req_id)=([A-Za-z0-9_-]+)")
_SERVICE_KV_RE = re.compile(r"\bservice=([A-Za-z0-9_.-]+)")
_EXCEPTION_TAIL_RE = re.compile(
    r"^\s*(?:(?P<exc>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Panic|Fault|Failure|Interrupt|Exit|Iteration|Violation|Timeout|Abort|Crash)):\s*(?P<msg>.+)|"
    r"(?P<generic_exc>(?!File\b|at\b|http\b|https\b|Caused\b|Traceback\b)[A-Z][A-Za-z0-9_.]*):\s*(?P<generic_msg>.+))$"
)
_FILE_HINT_RE = re.compile(rf"([A-Za-z0-9_./-]+\.(?:{_EXT_GROUP}))")


@dataclass(frozen=True)
class TelemetryEvent:
    """Normalized representation of an alert, error event, OTel span, or log record."""
    event_id: str
    source: str  # sentry, datadog, newrelic, opentelemetry, prometheus, grafana, logs, asre
    exception_type: str
    message: str
    stack_trace: str = ""
    service: str = "default"
    environment: str = "production"
    severity: str = "error"  # critical, fatal, error, warning
    timestamp: str = ""
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    commit_sha: str = ""
    count: int = 1
    file_hint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_incident(self) -> Incident:
        return Incident(
            incident_id=self.event_id,
            message=self.message,
            stack_trace=self.stack_trace,
            exception_type=self.exception_type,
            runtime=self.source,
            commit_sha=self.commit_sha,
            environment={"env": self.environment, "service": self.service, "severity": self.severity},
            metadata={
                **self.metadata,
                "source": self.source,
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "parent_span_id": self.parent_span_id,
                "timestamp": self.timestamp,
                "count": self.count,
                "file_hint": self.file_hint,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "exception_type": self.exception_type,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "service": self.service,
            "environment": self.environment,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "commit_sha": self.commit_sha,
            "count": self.count,
            "file_hint": self.file_hint,
            "metadata": self.metadata,
        }


@dataclass
class IncidentGroup:
    """Deduplicated group of related error events sharing a fingerprint."""
    group_id: str
    fingerprint: str
    title: str
    exception_type: str
    message_template: str
    event_count: int
    sources: list[str]
    services: list[str]
    environments: list[str]
    severities: list[str]
    trace_ids: list[str]
    mapped_files: list[str]
    primary_file: str | None
    representative_incident: Incident

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "fingerprint": self.fingerprint,
            "title": self.title,
            "exception_type": self.exception_type,
            "message_template": self.message_template,
            "event_count": self.event_count,
            "sources": self.sources,
            "services": self.services,
            "environments": self.environments,
            "severities": self.severities,
            "trace_ids": self.trace_ids,
            "mapped_files": self.mapped_files,
            "primary_file": self.primary_file,
            "representative_incident": self.representative_incident.to_dict(),
        }


@dataclass
class RootCauseCluster:
    """
    Correlated cluster of one or more IncidentGroups that share an underlying root cause
    via the Code Dependency Graph, shared root file, or Distributed Trace chain.
    """
    cluster_id: str
    root_cause_file: str | None
    priority_score: float
    priority_label: str  # P0-Critical, P1-High, P2-Medium, P3-Low
    total_events: int
    group_count: int
    correlation_reasons: list[str]
    anomalies: list[str]
    sources: list[str]
    services: list[str]
    trace_ids: list[str]
    affected_files: list[str]
    technical_impact_score: float
    business_impact_score: float
    anomaly_score: float
    groups: list[IncidentGroup]
    primary_incident: Incident

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "root_cause_file": self.root_cause_file,
            "priority_score": round(self.priority_score, 1),
            "priority_label": self.priority_label,
            "total_events": self.total_events,
            "group_count": self.group_count,
            "correlation_reasons": self.correlation_reasons,
            "anomalies": self.anomalies,
            "sources": self.sources,
            "services": self.services,
            "trace_ids": self.trace_ids,
            "affected_files": self.affected_files,
            "technical_impact_score": round(self.technical_impact_score, 1),
            "business_impact_score": round(self.business_impact_score, 1),
            "anomaly_score": round(self.anomaly_score, 1),
            "groups": [g.to_dict() for g in self.groups],
            "primary_incident": self.primary_incident.to_dict(),
        }


@dataclass
class IncidentIntelligenceReport:
    total_raw_events: int
    deduplicated_groups: int
    root_cause_clusters: int
    noise_reduction_pct: float
    sources_detected: list[str]
    services_affected: list[str]
    distributed_traces_correlated: int
    clusters: list[RootCauseCluster]

    @property
    def top_incident(self) -> Optional[Incident]:
        return self.clusters[0].primary_incident if self.clusters else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_raw_events": self.total_raw_events,
                "deduplicated_groups": self.deduplicated_groups,
                "root_cause_clusters": self.root_cause_clusters,
                "noise_reduction_pct": round(self.noise_reduction_pct, 2),
                "sources_detected": self.sources_detected,
                "services_affected": self.services_affected,
                "distributed_traces_correlated": self.distributed_traces_correlated,
            },
            "clusters": [c.to_dict() for c in self.clusters],
        }


# ──────────────────────────────────────────────────────────────────────────────
# 1. Multi-Provider Telemetry & Log Normalizers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_sentry_event(payload: dict[str, Any], index: int = 1) -> TelemetryEvent:
    event_id = str(payload.get("event_id") or payload.get("id") or f"sentry-{index}")
    exc_values = ((payload.get("exception") or {}).get("values")) or []
    exc_type = ""
    exc_msg = ""
    frame_lines: list[str] = []
    file_hint = ""

    if isinstance(exc_values, list) and exc_values:
        primary_exc = exc_values[-1] if isinstance(exc_values[-1], dict) else {}
        exc_type = str(primary_exc.get("type") or "")
        exc_msg = str(primary_exc.get("value") or "")
        frames = ((primary_exc.get("stacktrace") or {}).get("frames")) or []
        if isinstance(frames, list):
            for fr in frames:
                if not isinstance(fr, dict):
                    continue
                fn_path = str(fr.get("filename") or fr.get("abs_path") or "")
                lineno = int(fr.get("lineno") or 1)
                func = str(fr.get("function") or "<module>")
                if fn_path:
                    file_hint = fn_path
                    frame_lines.append(f'  File "{fn_path}", line {lineno}, in {func}')

    if not exc_msg:
        exc_msg = str(payload.get("message") or payload.get("title") or "Sentry error event")
    if not exc_type:
        exc_type = str(payload.get("type") or "Error")

    stack_trace = str(payload.get("stack_trace") or payload.get("stack") or "")
    if not stack_trace and frame_lines:
        stack_trace = "Traceback (most recent call last):\n" + "\n".join(frame_lines) + f"\n{exc_type}: {exc_msg}"

    tags = payload.get("tags") or {}
    if isinstance(tags, list):
        tags = {k: str(v) for k, v in tags if isinstance(k, str)}
    contexts = payload.get("contexts") or {}
    trace_ctx = (contexts.get("trace") or {}) if isinstance(contexts, dict) else {}
    trace_id = str(trace_ctx.get("trace_id") or payload.get("trace_id") or tags.get("trace_id") or "")
    span_id = str(trace_ctx.get("span_id") or payload.get("span_id") or "")
    service = str(tags.get("service") or payload.get("server_name") or payload.get("project") or "default")
    environment = str(payload.get("environment") or tags.get("environment") or "production")
    severity = str(payload.get("level") or "error").lower()
    count = max(1, int(payload.get("count") or 1))
    commit_sha = str(payload.get("release") or payload.get("commit_sha") or "")

    return TelemetryEvent(
        event_id=event_id,
        source="sentry",
        exception_type=exc_type,
        message=exc_msg,
        stack_trace=stack_trace,
        service=service,
        environment=environment,
        severity=severity,
        timestamp=str(payload.get("timestamp") or ""),
        trace_id=trace_id,
        span_id=span_id,
        commit_sha=commit_sha,
        count=count,
        file_hint=file_hint,
    )


def _normalize_datadog_event(payload: dict[str, Any], index: int = 1) -> TelemetryEvent:
    attrs = payload.get("attributes") if isinstance(payload.get("attributes"), dict) else payload
    err = attrs.get("error") if isinstance(attrs.get("error"), dict) else (
        payload.get("error") if isinstance(payload.get("error"), dict) else {}
    )
    exc_type = str(err.get("kind") or err.get("type") or attrs.get("error.kind") or payload.get("type") or "DatadogError")
    msg = str(err.get("message") or attrs.get("message") or payload.get("message") or "Datadog error event")
    stack = str(err.get("stack") or attrs.get("error.stack") or payload.get("stack_trace") or payload.get("stack") or "")
    service = str(attrs.get("service") or payload.get("service") or "default")
    env = str(attrs.get("env") or payload.get("env") or payload.get("environment") or "production")
    severity = str(attrs.get("status") or payload.get("status") or "error").lower()
    trace_id = str(attrs.get("dd.trace_id") or attrs.get("trace_id") or payload.get("trace_id") or "")
    span_id = str(attrs.get("dd.span_id") or attrs.get("span_id") or payload.get("span_id") or "")
    count = max(1, int(payload.get("count") or attrs.get("count") or 1))
    file_hint = str(err.get("file") or attrs.get("file") or "")

    return TelemetryEvent(
        event_id=str(payload.get("id") or f"datadog-{index}"),
        source="datadog",
        exception_type=exc_type,
        message=msg,
        stack_trace=stack,
        service=service,
        environment=env,
        severity=severity,
        timestamp=str(payload.get("timestamp") or attrs.get("timestamp") or ""),
        trace_id=trace_id,
        span_id=span_id,
        count=count,
        file_hint=file_hint,
    )


def _normalize_newrelic_event(payload: dict[str, Any], index: int = 1) -> TelemetryEvent:
    exc_type = str(
        payload.get("error.class")
        or payload.get("errorClass")
        or payload.get("exception_type")
        or "NewRelicError"
    )
    msg = str(
        payload.get("error.message")
        or payload.get("errorMessage")
        or payload.get("message")
        or "New Relic error event"
    )
    raw_stack = payload.get("stackTrace") or payload.get("stack_trace") or ""
    if isinstance(raw_stack, list):
        raw_stack = "\n".join(str(x) for x in raw_stack)
    service = str(payload.get("appName") or payload.get("service_name") or payload.get("service") or "default")
    env = str(payload.get("environment") or "production")
    trace_id = str(payload.get("traceId") or payload.get("trace_id") or "")
    span_id = str(payload.get("spanId") or payload.get("span_id") or "")
    count = max(1, int(payload.get("count") or 1))

    return TelemetryEvent(
        event_id=str(payload.get("id") or f"newrelic-{index}"),
        source="newrelic",
        exception_type=exc_type,
        message=msg,
        stack_trace=str(raw_stack),
        service=service,
        environment=env,
        severity=str(payload.get("severity") or "error").lower(),
        timestamp=str(payload.get("timestamp") or ""),
        trace_id=trace_id,
        span_id=span_id,
        count=count,
    )


def _normalize_otel_event(payload: dict[str, Any], index: int = 1) -> TelemetryEvent:
    attrs = payload.get("attributes") if isinstance(payload.get("attributes"), dict) else {}
    res_attrs = ((payload.get("resource") or {}).get("attributes")) if isinstance(payload.get("resource"), dict) else {}
    if not isinstance(res_attrs, dict):
        res_attrs = {}

    exc_type = str(
        attrs.get("exception.type")
        or payload.get("exception.type")
        or payload.get("exception_type")
        or "OpenTelemetryException"
    )
    msg = str(
        attrs.get("exception.message")
        or payload.get("exception.message")
        or payload.get("body")
        or payload.get("message")
        or "OpenTelemetry span error"
    )
    stack = str(
        attrs.get("exception.stacktrace")
        or payload.get("exception.stacktrace")
        or payload.get("stack_trace")
        or ""
    )
    service = str(
        res_attrs.get("service.name")
        or attrs.get("service.name")
        or payload.get("service")
        or "default"
    )
    env = str(
        res_attrs.get("deployment.environment")
        or attrs.get("deployment.environment")
        or payload.get("environment")
        or "production"
    )
    trace_id = str(payload.get("trace_id") or payload.get("traceId") or attrs.get("trace_id") or "")
    span_id = str(payload.get("span_id") or payload.get("spanId") or "")
    parent_span_id = str(payload.get("parent_span_id") or payload.get("parentSpanId") or "")
    count = max(1, int(payload.get("count") or 1))
    file_hint = str(attrs.get("code.filepath") or payload.get("file") or "")

    return TelemetryEvent(
        event_id=str(payload.get("id") or span_id or f"otel-{index}"),
        source="opentelemetry",
        exception_type=exc_type,
        message=msg,
        stack_trace=stack,
        service=service,
        environment=env,
        severity=str(payload.get("severity_text") or payload.get("severity") or "error").lower(),
        timestamp=str(payload.get("timestamp") or ""),
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        count=count,
        file_hint=file_hint,
    )


def _normalize_prometheus_alert(alert: dict[str, Any], index: int = 1, source: str = "prometheus") -> TelemetryEvent:
    labels = alert.get("labels") if isinstance(alert.get("labels"), dict) else {}
    annotations = alert.get("annotations") if isinstance(alert.get("annotations"), dict) else {}
    alertname = str(labels.get("alertname") or alert.get("alertname") or "MetricAlert")
    summary = str(
        annotations.get("summary")
        or annotations.get("description")
        or alert.get("message")
        or f"{alertname} firing"
    )
    service = str(labels.get("service") or labels.get("job") or alert.get("service") or "default")
    env = str(labels.get("env") or labels.get("environment") or "production")
    severity = str(labels.get("severity") or "warning").lower()
    file_hint = str(labels.get("file") or labels.get("module") or annotations.get("file") or "")
    line_hint = int(labels.get("line") or annotations.get("line") or 1)
    trace_id = str(labels.get("trace_id") or annotations.get("trace_id") or alert.get("trace_id") or "")
    stack = str(annotations.get("stack_trace") or alert.get("stack_trace") or "")
    if not stack and file_hint:
        stack = f'  File "{file_hint}", line {line_hint}, in { labels.get("function") or "<alert>" }'
    count = max(1, int(alert.get("count") or 1))

    return TelemetryEvent(
        event_id=str(alert.get("fingerprint") or alert.get("id") or f"{source}-{index}"),
        source=source,
        exception_type=alertname,
        message=summary,
        stack_trace=stack,
        service=service,
        environment=env,
        severity=severity,
        timestamp=str(alert.get("startsAt") or alert.get("timestamp") or ""),
        trace_id=trace_id,
        count=count,
        file_hint=file_hint,
    )


def normalize_telemetry_payload(payload: Any) -> list[TelemetryEvent]:
    """
    Auto-detect and normalize telemetry payloads from Sentry, Datadog, New Relic,
    OpenTelemetry, Prometheus/Grafana Alertmanager, or standard ASRE incident objects.
    """
    if isinstance(payload, list):
        events: list[TelemetryEvent] = []
        for item in payload:
            events.extend(normalize_telemetry_payload(item))
        return events

    if not isinstance(payload, dict):
        raise ValueError("telemetry payload must be a JSON object or array of objects")

    # Wrapper with {"events": [...]} or {"incidents": [...]}
    if isinstance(payload.get("events"), list):
        return normalize_telemetry_payload(payload["events"])
    if isinstance(payload.get("incidents"), list):
        return normalize_telemetry_payload(payload["incidents"])

    # Prometheus / Grafana Alertmanager webhook payload: {"alerts": [...]}
    if isinstance(payload.get("alerts"), list):
        src = "grafana" if "grafana" in str(payload.get("receiver", "")).lower() or payload.get("orgId") else "prometheus"
        return [
            _normalize_prometheus_alert(alert, idx, source=src)
            for idx, alert in enumerate(payload["alerts"], start=1)
            if isinstance(alert, dict)
        ]

    explicit_source = str(payload.get("source") or payload.get("ddsource") or "").lower().strip()

    if explicit_source == "sentry" or "exception" in payload and isinstance((payload.get("exception") or {}).get("values"), list):
        return [_normalize_sentry_event(payload)]

    if explicit_source == "datadog" or "ddsource" in payload or (
        isinstance(payload.get("attributes"), dict) and "error" in payload["attributes"]
    ):
        return [_normalize_datadog_event(payload)]

    if explicit_source in {"newrelic", "new_relic"} or "error.class" in payload or "errorClass" in payload:
        return [_normalize_newrelic_event(payload)]

    if (
        explicit_source in {"opentelemetry", "otel"}
        or "traceId" in payload
        or "spanId" in payload
        or (isinstance(payload.get("attributes"), dict) and any(k.startswith("exception.") for k in payload["attributes"]))
    ):
        return [_normalize_otel_event(payload)]

    if explicit_source in {"prometheus", "grafana"} or ("labels" in payload and "annotations" in payload):
        src = explicit_source if explicit_source in {"prometheus", "grafana"} else "prometheus"
        return [_normalize_prometheus_alert(payload, source=src)]

    # Standard ASRE Incident or generic alert object
    msg = str(payload.get("message") or payload.get("error") or payload.get("summary") or "").strip()
    if not msg:
        raise ValueError("telemetry event requires a non-empty message/error field")
    stack = payload.get("stack_trace") or payload.get("stack") or ""
    if isinstance(stack, list):
        stack = "\n".join(str(line) for line in stack)
    exc_type = str(payload.get("exception_type") or payload.get("type") or "Error")
    env_field = payload.get("environment")
    if isinstance(env_field, dict):
        env_name = str(env_field.get("env") or "production")
        service = str(payload.get("service") or env_field.get("service") or "default")
    else:
        env_name = str(env_field or "production")
        service = str(payload.get("service") or "default")

    return [
        TelemetryEvent(
            event_id=str(payload.get("incident_id") or payload.get("id") or "event-1"),
            source=explicit_source or "asre",
            exception_type=exc_type,
            message=msg,
            stack_trace=str(stack),
            service=service,
            environment=env_name,
            severity=str(payload.get("severity") or "error").lower(),
            timestamp=str(payload.get("timestamp") or ""),
            trace_id=str(payload.get("trace_id") or ""),
            span_id=str(payload.get("span_id") or ""),
            commit_sha=str(payload.get("commit_sha") or payload.get("commit") or ""),
            count=max(1, int(payload.get("count") or 1)),
            file_hint=str(payload.get("file") or payload.get("file_hint") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )
    ]


def parse_application_logs(raw_log_text: str) -> list[TelemetryEvent]:
    """
    Parse raw application log text (including multi-line tracebacks, timestamps,
    trace_id=..., and service=... tags) into normalized TelemetryEvents.
    """
    events: list[TelemetryEvent] = []
    lines = raw_log_text.splitlines()
    idx = 0
    event_seq = 1

    while idx < len(lines):
        line = lines[idx]
        header_match = _LOG_HEADER_RE.match(line.strip())
        is_traceback_start = line.strip().startswith("Traceback (most recent call last):")
        if not header_match and not is_traceback_start:
            idx += 1
            continue

        timestamp = header_match.group("ts") or "" if header_match else ""
        level = (header_match.group("level1") or header_match.group("level2") or "ERROR").lower() if header_match else "error"
        service = (header_match.group("service") or "") if header_match else ""
        msg_body = (header_match.group("msg") or "").strip() if header_match else line.strip()

        trace_match = _TRACE_KV_RE.search(line)
        trace_id = trace_match.group(1) if trace_match else ""
        if not service:
            svc_match = _SERVICE_KV_RE.search(line)
            service = svc_match.group(1) if svc_match else "default"

        block_lines: list[str] = [line]
        idx += 1
        while idx < len(lines):
            next_line = lines[idx]
            if _LOG_HEADER_RE.match(next_line.strip()):
                break
            if next_line.strip() == "" and len(block_lines) > 1 and not lines[min(idx + 1, len(lines) - 1)].startswith("  "):
                idx += 1
                break
            block_lines.append(next_line)
            if not trace_id:
                tm = _TRACE_KV_RE.search(next_line)
                if tm:
                    trace_id = tm.group(1)
            idx += 1

        block_text = "\n".join(block_lines)
        exc_type = "LogError"
        exc_msg = msg_body
        for b_line in reversed(block_lines):
            tail_m = _EXCEPTION_TAIL_RE.match(b_line.strip())
            if tail_m:
                exc_type = (tail_m.group("exc") or tail_m.group("generic_exc") or "LogError").strip()
                exc_msg = (tail_m.group("msg") or tail_m.group("generic_msg") or "").strip()
                break

        file_m = _FILE_HINT_RE.search(block_text)
        file_hint = file_m.group(1) if file_m else ""

        events.append(
            TelemetryEvent(
                event_id=f"log-{event_seq}",
                source="logs",
                exception_type=exc_type,
                message=exc_msg or "Application log error",
                stack_trace=block_text,
                service=service,
                environment="production",
                severity="critical" if level in {"critical", "fatal"} else level,
                timestamp=timestamp,
                trace_id=trace_id,
                count=1,
                file_hint=file_hint,
            )
        )
        event_seq += 1

    return events


def load_telemetry_events(path: str) -> list[TelemetryEvent]:
    """Load telemetry events from a JSON array/object, JSONL stream, or raw .log file."""
    file_path = Path(path)
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"failed to read telemetry file {path!r}: {exc}") from exc

    stripped = raw_text.strip()
    if not stripped:
        raise ValueError(f"telemetry file {path!r} is empty")

    # 1. Try standard JSON object or array
    try:
        payload = json.loads(stripped)
        return normalize_telemetry_payload(payload)
    except json.JSONDecodeError:
        pass

    # 2. Try JSONL (one JSON object per line)
    jsonl_events: list[TelemetryEvent] = []
    non_empty_lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if non_empty_lines and all(ln.startswith("{") and ln.endswith("}") for ln in non_empty_lines):
        try:
            for ln in non_empty_lines:
                jsonl_events.extend(normalize_telemetry_payload(json.loads(ln)))
            if jsonl_events:
                return jsonl_events
        except json.JSONDecodeError:
            pass

    # 3. Parse as raw multi-line application log file
    log_events = parse_application_logs(raw_text)
    if not log_events:
        raise ValueError(f"could not parse any telemetry events or error logs from {path!r}")
    return log_events


# ──────────────────────────────────────────────────────────────────────────────
# 2. Incident Grouping, Anomaly Detection, Graph Correlation & Prioritisation
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_message_template(message: str) -> str:
    cleaned = _VOLATILE_TOKEN_RE.sub("<N>", message.strip())
    return " ".join(cleaned.split())[:160]


def _map_event_files(
    event: TelemetryEvent,
    repo_root: str,
    available_paths: set[str],
) -> list[str]:
    mapped: list[str] = []
    for frame in extract_stack_frames(event.stack_trace):
        res = _match_repository_path_detailed(
            frame.path,
            repo_root,
            available_paths,
            frame.function,
        )
        if res.mapped_path and res.mapped_path not in mapped:
            mapped.append(res.mapped_path)

    if not mapped and event.file_hint:
        res = _match_repository_path_detailed(event.file_hint, repo_root, available_paths, "")
        if res.mapped_path:
            mapped.append(res.mapped_path)

    if not mapped:
        for candidate in _FILE_HINT_RE.findall(f"{event.message}\n{event.stack_trace}"):
            res = _match_repository_path_detailed(candidate, repo_root, available_paths, "")
            if res.mapped_path and res.mapped_path not in mapped:
                mapped.append(res.mapped_path)

    return mapped


def group_events_into_incidents(
    events: Iterable[TelemetryEvent],
    repo_root: str,
    profiles: dict[str, FileProfile],
) -> list[IncidentGroup]:
    """Deduplicate raw telemetry events into fingerprinted IncidentGroups."""
    available_paths = set(profiles)
    buckets: dict[str, list[tuple[TelemetryEvent, list[str]]]] = defaultdict(list)

    for event in events:
        mapped_files = _map_event_files(event, repo_root, available_paths)
        primary_file = mapped_files[-1] if mapped_files else (event.file_hint or "unmapped")
        msg_template = _normalize_message_template(event.message)
        fp_raw = f"{primary_file}|{event.exception_type.lower()}|{msg_template.lower()}"
        fingerprint = hashlib.sha1(fp_raw.encode("utf-8")).hexdigest()[:12]
        buckets[fingerprint].append((event, mapped_files))

    groups: list[IncidentGroup] = []
    for idx, (fp, items) in enumerate(
        sorted(buckets.items(), key=lambda kv: (-sum(ev.count for ev, _ in kv[1]), kv[0])),
        start=1,
    ):
        first_event, _ = items[0]
        best_event = max(items, key=lambda pair: (len(pair[1]), len(pair[0].stack_trace), pair[0].count))[0]
        total_count = sum(ev.count for ev, _ in items)
        sources = sorted({ev.source for ev, _ in items})
        services = sorted({ev.service for ev, _ in items if ev.service})
        environments = sorted({ev.environment for ev, _ in items if ev.environment})
        severities = sorted({ev.severity for ev, _ in items if ev.severity})
        trace_ids = sorted({ev.trace_id for ev, _ in items if ev.trace_id})
        all_mapped: list[str] = []
        for _, m_files in items:
            for path in m_files:
                if path not in all_mapped:
                    all_mapped.append(path)
        primary_file = all_mapped[-1] if all_mapped else None
        msg_template = _normalize_message_template(first_event.message)
        title = f"{first_event.exception_type}: {msg_template[:80]}"

        groups.append(
            IncidentGroup(
                group_id=f"group-{idx}",
                fingerprint=fp,
                title=title,
                exception_type=first_event.exception_type,
                message_template=msg_template,
                event_count=total_count,
                sources=sources,
                services=services,
                environments=environments,
                severities=severities,
                trace_ids=trace_ids,
                mapped_files=all_mapped,
                primary_file=primary_file,
                representative_incident=best_event.to_incident(),
            )
        )
    return groups


def _reachable_dependencies(graph: DependencyGraph, path: str, max_hops: int = 2) -> dict[str, int]:
    """Return {target_file: hop_distance} for files that `path` depends on within max_hops."""
    distances: dict[str, int] = {path: 0}
    frontier = [path]
    for hop in range(1, max_hops + 1):
        next_frontier: list[str] = []
        for node in frontier:
            for dep in graph.dependencies(node):
                if dep not in distances:
                    distances[dep] = hop
                    next_frontier.append(dep)
        frontier = next_frontier
    return distances


def correlate_and_prioritize_incidents(
    events: Iterable[TelemetryEvent],
    graph: DependencyGraph,
    profiles: dict[str, FileProfile],
    repo_root: str,
) -> IncidentIntelligenceReport:
    """
    Full Section A pipeline:
      1. Deduplicate raw events into IncidentGroups
      2. Correlate IncidentGroups that share a distributed trace_id or converge on a
         shared downstream dependency in the Code Dependency Graph
      3. Detect anomalies (error storms, cross-service cascades, critical-path faults)
      4. Score & rank RootCauseClusters by combined technical + business + anomaly impact
    """
    event_list = list(events)
    total_raw_events = sum(ev.count for ev in event_list)
    groups = group_events_into_incidents(event_list, repo_root, profiles)
    if not groups:
        return IncidentIntelligenceReport(
            total_raw_events=0,
            deduplicated_groups=0,
            root_cause_clusters=0,
            noise_reduction_pct=0.0,
            sources_detected=[],
            services_affected=[],
            distributed_traces_correlated=0,
            clusters=[],
        )

    # Precompute forward dependency cone (up to 2 hops) for each group's mapped files
    group_dep_cones: dict[str, dict[str, int]] = {}
    for g in groups:
        cone: dict[str, int] = {}
        for f_path in g.mapped_files:
            for target, dist in _reachable_dependencies(graph, f_path, max_hops=2).items():
                if target not in cone or dist < cone[target]:
                    cone[target] = dist
        group_dep_cones[g.group_id] = cone

    # Union-Find across groups for correlation
    parent: dict[str, str] = {g.group_id: g.group_id for g in groups}
    pair_reasons: dict[tuple[str, str], list[str]] = defaultdict(list)

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str, reason: str) -> None:
        ra, rb = find(a), find(b)
        key = (min(a, b), max(a, b))
        if reason not in pair_reasons[key]:
            pair_reasons[key].append(reason)
        if ra != rb:
            parent[rb] = ra

    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1 :]:
            # Signal 1: Shared distributed trace_id
            shared_traces = set(g1.trace_ids) & set(g2.trace_ids)
            if shared_traces:
                sample_trace = sorted(shared_traces)[0]
                union(
                    g1.group_id,
                    g2.group_id,
                    f"Shared distributed trace_id '{sample_trace}' across {', '.join(sorted(set(g1.services) | set(g2.services)))}",
                )

            # Signal 2: Direct shared mapped file
            shared_files = set(g1.mapped_files) & set(g2.mapped_files)
            if shared_files:
                sample_file = sorted(shared_files)[0]
                union(
                    g1.group_id,
                    g2.group_id,
                    f"Multiple distinct error signatures originate in '{sample_file}'",
                )
                continue

            # Signal 3: Code Dependency Graph convergence (symptom callers depend on shared downstream file)
            cone1 = group_dep_cones[g1.group_id]
            cone2 = group_dep_cones[g2.group_id]
            common_deps = set(cone1) & set(cone2)
            if common_deps:
                best_dep = max(
                    common_deps,
                    key=lambda p: (
                        1 if p in set(g1.mapped_files) | set(g2.mapped_files) else 0,
                        profiles[p].risk_score if p in profiles else 0,
                        -(cone1[p] + cone2[p]),
                    ),
                )
                union(
                    g1.group_id,
                    g2.group_id,
                    f"Dependency graph convergence: callers {', '.join(sorted(set(g1.mapped_files) | set(g2.mapped_files)))} depend on '{best_dep}'",
                )

    clustered_groups: dict[str, list[IncidentGroup]] = defaultdict(list)
    for g in groups:
        clustered_groups[find(g.group_id)].append(g)

    raw_clusters: list[RootCauseCluster] = []
    for root_gid, member_groups in clustered_groups.items():
        member_ids = {g.group_id for g in member_groups}
        reasons: list[str] = []
        for (a, b), r_list in pair_reasons.items():
            if a in member_ids and b in member_ids:
                for r in r_list:
                    if r not in reasons:
                        reasons.append(r)

        total_events = sum(g.event_count for g in member_groups)
        sources = sorted({s for g in member_groups for s in g.sources})
        services = sorted({s for g in member_groups for s in g.services})
        trace_ids = sorted({t for g in member_groups for t in g.trace_ids})
        affected_files: list[str] = []
        for g in member_groups:
            for f in g.mapped_files:
                if f not in affected_files:
                    affected_files.append(f)

        # Identify the shared downstream root-cause file in the cluster
        candidate_counts: Counter[str] = Counter()
        direct_frame_files = {f for g in member_groups for f in g.mapped_files}
        for g in member_groups:
            for dep_path, dist in group_dep_cones[g.group_id].items():
                # Weight by how many symptom groups reach this dependency + whether it has direct stack frames
                weight = 3 if dist == 0 else (2 if dist == 1 else 1)
                candidate_counts[dep_path] += weight * g.event_count

        root_cause_file: str | None = None
        if candidate_counts:
            # Prefer files that are reached by the most groups in the cluster, breaking ties by direct frame hit & risk_score
            groups_reaching: Counter[str] = Counter()
            for g in member_groups:
                for dep_path in group_dep_cones[g.group_id]:
                    groups_reaching[dep_path] += 1
            root_cause_file = max(
                candidate_counts,
                key=lambda p: (
                    groups_reaching[p],
                    1 if p in direct_frame_files else 0,
                    profiles[p].risk_score if p in profiles else 0,
                    candidate_counts[p],
                ),
            )

        if not reasons:
            if len(member_groups) == 1 and member_groups[0].event_count > 1:
                reasons.append(
                    f"Deduplicated {member_groups[0].event_count} identical alerts into 1 group"
                )
            elif root_cause_file:
                reasons.append(f"Mapped directly to '{root_cause_file}'")

        # ── Anomaly Detection ──────────────────────────────────────
        anomalies: list[str] = []
        event_share = total_events / max(total_raw_events, 1)
        if total_events >= 10 or (total_raw_events >= 5 and event_share >= 0.5):
            anomalies.append(
                f"Error storm anomaly: {total_events} alerts ({event_share * 100:.0f}% of total volume)"
            )
        if len(services) > 1:
            anomalies.append(
                f"Cross-service cascade across {len(services)} services ({', '.join(services)})"
            )
        if len(sources) > 1:
            anomalies.append(
                f"Multi-system alert correlation ({', '.join(sources)})"
            )
        root_profile = profiles.get(root_cause_file) if root_cause_file else None
        if root_profile and root_profile.risk_score >= 65:
            anomalies.append(
                f"Critical architectural hotspot affected ({root_cause_file}, risk {root_profile.risk_score}/90)"
            )

        # ── Business & Technical Impact Prioritisation (0–100) ─────
        # 1. Technical impact (max 40): file risk score (0-90) + transitive consumers
        tech_score = 0.0
        if root_profile is not None:
            tech_score = min(40.0, (root_profile.risk_score / 90.0) * 32.0 + min(8.0, root_profile.transitive_impact * 0.5))
        elif affected_files:
            max_file_risk = max((profiles[f].risk_score for f in affected_files if f in profiles), default=0)
            tech_score = min(40.0, (max_file_risk / 90.0) * 32.0)

        # 2. Business impact (max 35): event volume + environment + severity
        envs = {e.lower() for g in member_groups for e in g.environments}
        sevs = {s.lower() for g in member_groups for s in g.severities}
        env_pts = 12.0 if envs & {"production", "prod"} else 6.0
        sev_pts = 10.0 if sevs & {"critical", "fatal"} else (7.0 if "error" in sevs else 3.0)
        vol_pts = min(13.0, float(total_events) ** 0.5 * 1.5)
        biz_score = min(35.0, env_pts + sev_pts + vol_pts)

        # 3. Anomaly & cascade score (max 25)
        anom_score = min(
            25.0,
            (8.0 if len(member_groups) > 1 else 0.0)
            + (6.0 if len(services) > 1 else 0.0)
            + (5.0 if len(sources) > 1 else 0.0)
            + (6.0 if total_events >= 10 else 0.0),
        )

        priority_score = min(100.0, tech_score + biz_score + anom_score)
        if priority_score >= 70.0:
            priority_label = "P0-Critical"
        elif priority_score >= 50.0:
            priority_label = "P1-High"
        elif priority_score >= 30.0:
            priority_label = "P2-Medium"
        else:
            priority_label = "P3-Low"

        # Choose the best representative incident for repair (prefer group directly hitting root_cause_file)
        ordered_groups = sorted(
            member_groups,
            key=lambda g: (
                1 if root_cause_file and root_cause_file in g.mapped_files else 0,
                g.event_count,
                len(g.representative_incident.stack_trace),
            ),
            reverse=True,
        )
        best_inc = ordered_groups[0].representative_incident
        # Enrich primary incident with merged stack traces from correlated groups so build_repair_context sees all frames
        merged_stacks = [
            g.representative_incident.stack_trace
            for g in ordered_groups
            if g.representative_incident.stack_trace
        ]
        primary_incident = Incident(
            incident_id=best_inc.incident_id,
            message=best_inc.message,
            stack_trace="\n".join(merged_stacks),
            exception_type=best_inc.exception_type,
            runtime=best_inc.runtime,
            commit_sha=best_inc.commit_sha,
            environment=best_inc.environment,
            metadata={
                **best_inc.metadata,
                "correlated_events": total_events,
                "correlated_groups": len(member_groups),
                "root_cause_file": root_cause_file,
                "priority_label": priority_label,
            },
        )

        raw_clusters.append(
            RootCauseCluster(
                cluster_id="",
                root_cause_file=root_cause_file,
                priority_score=priority_score,
                priority_label=priority_label,
                total_events=total_events,
                group_count=len(member_groups),
                correlation_reasons=reasons,
                anomalies=anomalies,
                sources=sources,
                services=services,
                trace_ids=trace_ids,
                affected_files=affected_files,
                technical_impact_score=tech_score,
                business_impact_score=biz_score,
                anomaly_score=anom_score,
                groups=ordered_groups,
                primary_incident=primary_incident,
            )
        )

    raw_clusters.sort(key=lambda c: (-c.priority_score, -c.total_events, c.root_cause_file or ""))
    final_clusters = [
        RootCauseCluster(
            cluster_id=f"cluster-{idx}",
            root_cause_file=c.root_cause_file,
            priority_score=c.priority_score,
            priority_label=c.priority_label,
            total_events=c.total_events,
            group_count=c.group_count,
            correlation_reasons=c.correlation_reasons,
            anomalies=c.anomalies,
            sources=c.sources,
            services=c.services,
            trace_ids=c.trace_ids,
            affected_files=c.affected_files,
            technical_impact_score=c.technical_impact_score,
            business_impact_score=c.business_impact_score,
            anomaly_score=c.anomaly_score,
            groups=c.groups,
            primary_incident=c.primary_incident,
        )
        for idx, c in enumerate(raw_clusters, start=1)
    ]

    noise_reduction = (
        ((total_raw_events - len(final_clusters)) / total_raw_events) * 100.0
        if total_raw_events > len(final_clusters)
        else 0.0
    )
    all_sources = sorted({s for c in final_clusters for s in c.sources})
    all_services = sorted({s for c in final_clusters for s in c.services})
    all_traces = {t for c in final_clusters for t in c.trace_ids}

    return IncidentIntelligenceReport(
        total_raw_events=total_raw_events,
        deduplicated_groups=len(groups),
        root_cause_clusters=len(final_clusters),
        noise_reduction_pct=noise_reduction,
        sources_detected=all_sources,
        services_affected=all_services,
        distributed_traces_correlated=len(all_traces),
        clusters=final_clusters,
    )
