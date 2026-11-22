"""
Supabase Database & ML/LLM Training Data Storage Engine for ASRE.

Provides dual-write persistence (Supabase Cloud Postgres/PostgREST + Local `.asre/` store)
with automatic secret redaction, offline queue resilience (`.asre/pending_sync.jsonl`),
automatic schema migration (`storage/supabase_schema.sql`), and ML/LLM training dataset
exporters (SFT JSONL, DPO Preference Pairs JSONL, and Tabular JIT Change-Risk CSV/JSONL).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from repair.cross_incident_intelligence import DrainTemplateMiner
from repair.knowledge_base import (
    classify_structural_fix_operator,
    load_tier2_episodic_cases,
)
from repair.llm_planner import load_env_file
from repair.production_guard import redact_secrets


SUPABASE_TABLES: tuple[str, ...] = (
    "asre_repositories",
    "asre_incidents",
    "asre_causal_graphs",
    "asre_repair_runs",
    "asre_knowledge_base",
    "asre_change_risk_records",
    "asre_slo_snapshots",
    "asre_audit_logs",
    "asre_integrations",
    "asre_tenant_jobs",
    "asre_organizations",
    "asre_users",
)

TABLE_PRIMARY_KEYS: dict[str, str] = {
    "asre_repositories": "repo_id",
    "asre_incidents": "incident_id",
    "asre_causal_graphs": "correlation_id",
    "asre_repair_runs": "run_id",
    "asre_knowledge_base": "entry_id",
    "asre_change_risk_records": "check_id",
    "asre_slo_snapshots": "snapshot_id",
    "asre_audit_logs": "audit_id",
    "asre_integrations": "integration_id",
    "asre_tenant_jobs": "job_id",
    "asre_organizations": "org_id",
    "asre_users": "user_id",
}


@dataclass(frozen=True)
class SupabaseSyncResult:
    table: str
    records_attempted: int
    synced_to_cloud: bool
    queued_locally: bool
    status_code: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "records_attempted": self.records_attempted,
            "synced_to_cloud": self.synced_to_cloud,
            "queued_locally": self.queued_locally,
            "status_code": self.status_code,
            "error": self.error,
        }


@dataclass(frozen=True)
class DatabaseStatusReport:
    configured: bool
    supabase_url: str
    has_api_key: bool
    has_db_url: bool
    rest_reachable: bool
    schema_ready: bool
    pending_offline_records: int
    table_counts: dict[str, int] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "supabase_url": self.supabase_url,
            "has_api_key": self.has_api_key,
            "has_db_url": self.has_db_url,
            "rest_reachable": self.rest_reachable,
            "schema_ready": self.schema_ready,
            "pending_offline_records": self.pending_offline_records,
            "table_counts": dict(self.table_counts),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class TrainingExportReport:
    output_dir: str
    sft_records_count: int
    dpo_pairs_count: int
    change_risk_rows_count: int
    causal_graph_records_count: int
    files_written: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "sft_records_count": self.sft_records_count,
            "dpo_pairs_count": self.dpo_pairs_count,
            "change_risk_rows_count": self.change_risk_rows_count,
            "causal_graph_records_count": self.causal_graph_records_count,
            "files_written": list(self.files_written),
        }


def _normalize_repo_id(repo_path_or_name: str) -> str:
    if not repo_path_or_name:
        return "default-repo"
    p = Path(repo_path_or_name).resolve()
    return p.name.lower().strip() or "default-repo"


def _scrub_payload(payload: Any) -> Any:
    """Pass any dictionary or list through deterministic secret redaction."""
    raw_json = json.dumps(payload, default=str, sort_keys=True)
    scrubbed_json = redact_secrets(raw_json)
    return json.loads(scrubbed_json)


def _encode_db_url_password(db_url: str) -> str:
    """Ensure special characters like '*' in postgresql://user:pass@host/db are URL-encoded for drivers."""
    if not db_url or "://" not in db_url or "@" not in db_url:
        return db_url
    scheme, rest = db_url.split("://", 1)
    userinfo, hostpart = rest.rsplit("@", 1)
    if ":" not in userinfo:
        return db_url
    user, pwd = userinfo.split(":", 1)
    # Only encode if raw unencoded special chars exist
    if "*" in pwd or "#" in pwd or " " in pwd:
        pwd_encoded = urllib.parse.quote(pwd, safe="")
        return f"{scheme}://{user}:{pwd_encoded}@{hostpart}"
    return db_url


def apply_supabase_schema_migration(
    env: Optional[dict[str, str]] = None,
    schema_path: Optional[str | Path] = None,
) -> tuple[bool, str]:
    """
    Apply `storage/supabase_schema.sql` directly against `SUPABASE_DB_URL`
    using psycopg2/psycopg (if available) or `psql` CLI.
    """
    env_map = env if env is not None else load_env_file()
    raw_db_url = (env_map.get("SUPABASE_DB_URL") or env_map.get("DATABASE_URL") or "").strip()
    if not raw_db_url:
        return False, "SUPABASE_DB_URL is not configured in .env"

    sql_file = Path(schema_path) if schema_path else (Path(__file__).resolve().parent / "supabase_schema.sql")
    if not sql_file.is_file():
        return False, f"Schema SQL file not found at {sql_file}"
    sql_text = sql_file.read_text(encoding="utf-8")
    safe_db_url = _encode_db_url_password(raw_db_url)

    # 1. Try psycopg2 or psycopg if installed
    for mod_name in ("psycopg2", "psycopg"):
        try:
            pg_mod = __import__(mod_name)
            conn = pg_mod.connect(safe_db_url, connect_timeout=15)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(sql_text)
            conn.close()
            return True, f"Applied Supabase schema migration via {mod_name}"
        except ImportError:
            continue
        except Exception as exc:
            err_msg = redact_secrets(str(exc))
            # Continue to try psql fallback if driver failed
            last_driver_err = f"{mod_name} migration error: {err_msg}"
            break
    else:
        last_driver_err = ""

    # 2. Try `psql` CLI if available
    try:
        completed = subprocess.run(
            ["psql", safe_db_url, "-v", "ON_ERROR_STOP=1", "-f", str(sql_file)],
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
        if completed.returncode == 0:
            return True, "Applied Supabase schema migration via psql CLI"
        return False, redact_secrets(f"psql exited {completed.returncode}: {completed.stderr.strip()}")
    except FileNotFoundError:
        return (
            False,
            last_driver_err or "Neither psycopg2/psycopg nor psql CLI is installed; run storage/supabase_schema.sql in Supabase SQL Editor",
        )
    except Exception as exc:
        return False, redact_secrets(str(exc))


class SupabaseStorageEngine:
    """
    Provider-safe Supabase PostgREST + Local Dual-Write Storage Engine.
    """

    def __init__(
        self,
        *,
        env: Optional[dict[str, str]] = None,
        queue_dir: str | Path = ".asre",
        transport: Optional[Callable[[str, str, dict[str, str], Optional[ bytes ]], tuple[int, str]]] = None,
        timeout: int = 12,
    ) -> None:
        self._env = env if env is not None else load_env_file()
        self.supabase_url = (self._env.get("SUPABASE_URL") or "").strip().rstrip("/")
        self.supabase_key = (
            self._env.get("SUPABASE_SERVICE_ROLE_KEY")
            or self._env.get("SUPABASE_KEY")
            or self._env.get("SUPABASE_ANON_KEY")
            or ""
        ).strip()
        self.db_url = (self._env.get("SUPABASE_DB_URL") or self._env.get("DATABASE_URL") or "").strip()
        self.queue_dir = Path(queue_dir).resolve()
        self.pending_queue_file = self.queue_dir / "pending_sync.jsonl"
        self.local_tables_dir = self.queue_dir / "supabase_mirror"
        self._transport = transport
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    def _headers(self, *, prefer: str = "resolution=merge-duplicates,return=minimal") -> dict[str, str]:
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": prefer,
            "User-Agent": "ASRE-SupabaseStore/1.0",
        }

    def _http_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body_bytes: Optional[bytes] = None,
    ) -> tuple[int, str]:
        if self._transport is not None:
            return self._transport(method, url, headers, body_bytes)

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                return resp.status, text
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return exc.code, redact_secrets(err_body[:500])
        except Exception as exc:
            return 0, redact_secrets(str(exc))

    def _mirror_locally(self, table: str, rows: list[dict[str, Any]]) -> None:
        """Maintain a local JSONL mirror of every table under `.asre/supabase_mirror/<table>.jsonl`."""
        try:
            self.local_tables_dir.mkdir(parents=True, exist_ok=True)
            mirror_file = self.local_tables_dir / f"{table}.jsonl"
            pk = TABLE_PRIMARY_KEYS.get(table, "id")
            existing: dict[str, dict[str, Any]] = {}
            if mirror_file.is_file():
                for ln in mirror_file.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    try:
                        obj = json.loads(ln)
                        key_val = str(obj.get(pk) or "")
                        if key_val:
                            existing[key_val] = obj
                    except json.JSONDecodeError:
                        continue
            for r in rows:
                k = str(r.get(pk) or hashlib.sha1(json.dumps(r, sort_keys=True).encode("utf-8")).hexdigest()[:12])
                existing[k] = r
            lines = [json.dumps(v, sort_keys=True) for v in existing.values()]
            mirror_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        except OSError:
            pass

    def _queue_pending(self, table: str, rows: list[dict[str, Any]]) -> None:
        try:
            self.queue_dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "table": table,
                "rows": rows,
            }
            with open(self.pending_queue_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError:
            pass

    def pending_records_count(self) -> int:
        if not self.pending_queue_file.is_file():
            return 0
        try:
            return sum(1 for ln in self.pending_queue_file.read_text(encoding="utf-8").splitlines() if ln.strip())
        except OSError:
            return 0

    def upsert(
        self,
        table: str,
        records: dict[str, Any] | Sequence[dict[str, Any]],
        *,
        queue_on_failure: bool = True,
    ) -> SupabaseSyncResult:
        """
        Scrub secrets, mirror locally to `.asre/supabase_mirror/<table>.jsonl`,
        and upsert to Supabase PostgREST `/rest/v1/<table>?on_conflict=<pk>`.
        """
        raw_list = [records] if isinstance(records, dict) else list(records)
        if not raw_list:
            return SupabaseSyncResult(table=table, records_attempted=0, synced_to_cloud=True, queued_locally=False)

        scrubbed_rows: list[dict[str, Any]] = [_scrub_payload(r) for r in raw_list]
        self._mirror_locally(table, scrubbed_rows)

        if not self.is_configured:
            if queue_on_failure:
                self._queue_pending(table, scrubbed_rows)
            return SupabaseSyncResult(
                table=table,
                records_attempted=len(scrubbed_rows),
                synced_to_cloud=False,
                queued_locally=queue_on_failure,
                error="Supabase URL or API key not configured",
            )

        pk = TABLE_PRIMARY_KEYS.get(table, "")
        query_suffix = f"?on_conflict={pk}" if pk else ""
        url = f"{self.supabase_url}/rest/v1/{table}{query_suffix}"
        body = json.dumps(scrubbed_rows).encode("utf-8")
        status_code, resp_text = self._http_request("POST", url, self._headers(), body)

        if 200 <= status_code < 300:
            return SupabaseSyncResult(
                table=table,
                records_attempted=len(scrubbed_rows),
                synced_to_cloud=True,
                queued_locally=False,
                status_code=status_code,
            )

        if queue_on_failure:
            self._queue_pending(table, scrubbed_rows)
        return SupabaseSyncResult(
            table=table,
            records_attempted=len(scrubbed_rows),
            synced_to_cloud=False,
            queued_locally=queue_on_failure,
            status_code=status_code,
            error=resp_text[:300] or f"HTTP {status_code}",
        )

    def select(
        self,
        table: str,
        *,
        filters: Optional[dict[str, str]] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query records from Supabase PostgREST, merged with local mirror."""
        pk = TABLE_PRIMARY_KEYS.get(table, "id")
        merged: dict[str, dict[str, Any]] = {}

        # 1. Load local mirror first
        mirror_file = self.local_tables_dir / f"{table}.jsonl"
        if mirror_file.is_file():
            try:
                for ln in mirror_file.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    obj = json.loads(ln)
                    if filters and not all(str(obj.get(k)) == v for k, v in filters.items()):
                        continue
                    k_val = str(obj.get(pk) or len(merged))
                    merged[k_val] = obj
            except Exception:
                pass

        # 2. Overlay cloud records when configured and reachable
        if self.is_configured:
            params = [f"limit={max(1, limit)}"]
            if filters:
                for k, v in filters.items():
                    params.append(f"{urllib.parse.quote(k)}=eq.{urllib.parse.quote(v)}")
            url = f"{self.supabase_url}/rest/v1/{table}?" + "&".join(params)
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Accept": "application/json",
                "User-Agent": "ASRE-SupabaseStore/1.0",
            }
            status_code, resp_text = self._http_request("GET", url, headers, None)
            if 200 <= status_code < 300 and resp_text.strip():
                try:
                    parsed = json.loads(resp_text)
                    if isinstance(parsed, list):
                        for obj in parsed:
                            if isinstance(obj, dict):
                                k_val = str(obj.get(pk) or len(merged))
                                merged[k_val] = obj
                except json.JSONDecodeError:
                    pass

        return list(merged.values())[:limit]

    def flush_pending_sync(self) -> dict[str, Any]:
        """Flush queued records in `.asre/pending_sync.jsonl` to Supabase."""
        if not self.pending_queue_file.is_file():
            return {"flushed_batches": 0, "remaining_batches": 0, "synced_to_cloud": True}

        lines = [ln for ln in self.pending_queue_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        remaining: list[str] = []
        flushed = 0

        for ln in lines:
            try:
                item = json.loads(ln)
                table = str(item.get("table") or "")
                rows = list(item.get("rows") or [])
            except json.JSONDecodeError:
                continue
            if not table or not rows:
                continue
            res = self.upsert(table, rows, queue_on_failure=False)
            if res.synced_to_cloud:
                flushed += 1
            else:
                remaining.append(ln)

        if remaining:
            self.pending_queue_file.write_text("\n".join(remaining) + "\n", encoding="utf-8")
        else:
            try:
                self.pending_queue_file.unlink()
            except OSError:
                pass

        return {
            "flushed_batches": flushed,
            "remaining_batches": len(remaining),
            "synced_to_cloud": len(remaining) == 0,
        }

    def check_status(self) -> DatabaseStatusReport:
        if not self.is_configured:
            return DatabaseStatusReport(
                configured=False,
                supabase_url=self.supabase_url,
                has_api_key=bool(self.supabase_key),
                has_db_url=bool(self.db_url),
                rest_reachable=False,
                schema_ready=False,
                pending_offline_records=self.pending_records_count(),
                detail="SUPABASE_URL or SUPABASE_KEY is missing in .env",
            )

        url = f"{self.supabase_url}/rest/v1/asre_repositories?select=repo_id&limit=1"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Accept": "application/json",
            "User-Agent": "ASRE-SupabaseStore/1.0",
        }
        code, body = self._http_request("GET", url, headers, None)
        rest_ok = code in {200, 206, 404, 400} or (200 <= code < 500)
        schema_ok = 200 <= code < 300

        counts: dict[str, int] = {}
        for tbl in SUPABASE_TABLES:
            mirror_f = self.local_tables_dir / f"{tbl}.jsonl"
            if mirror_f.is_file():
                try:
                    counts[tbl] = sum(1 for ln in mirror_f.read_text(encoding="utf-8").splitlines() if ln.strip())
                except OSError:
                    counts[tbl] = 0
            else:
                counts[tbl] = 0

        if schema_ok:
            detail = "Connected to Supabase PostgREST and ASRE schema tables are active."
        elif rest_ok:
            detail = f"Supabase PostgREST reachable (HTTP {code}), but schema tables need migration (`--init-db` or storage/supabase_schema.sql): {body[:160]}"
        else:
            detail = f"Supabase PostgREST unreachable (HTTP {code}): {body[:160]}"

        return DatabaseStatusReport(
            configured=True,
            supabase_url=self.supabase_url,
            has_api_key=bool(self.supabase_key),
            has_db_url=bool(self.db_url),
            rest_reachable=rest_ok,
            schema_ready=schema_ok,
            pending_offline_records=self.pending_records_count(),
            table_counts=counts,
            detail=detail,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # High-Level ASRE Lifecycle Sync Methods
    # ──────────────────────────────────────────────────────────────────────────

    def sync_repository_scan(
        self,
        repo_path: str,
        intel: Any,
        graph: Optional[Any] = None,
        profiles: Optional[dict[str, Any]] = None,
    ) -> SupabaseSyncResult:
        repo_id = _normalize_repo_id(repo_path)
        gstats = graph.stats() if graph is not None and hasattr(graph, "stats") else {}
        hotspots: list[dict[str, Any]] = []
        if profiles:
            top_profs = sorted(profiles.values(), key=lambda p: getattr(p, "risk_score", 0), reverse=True)[:10]
            hotspots = [
                {
                    "path": getattr(p, "path", ""),
                    "risk_score": getattr(p, "risk_score", 0),
                    "risk_label": getattr(p, "risk_label", "Low"),
                    "transitive_impact": getattr(p, "transitive_impact", 0),
                }
                for p in top_profs
            ]
        git_meta = getattr(intel, "git", None)
        head_sha = getattr(git_meta, "last_commit_hash", "") if git_meta else ""
        row = {
            "repo_id": repo_id,
            "repo_name": getattr(intel, "repo_name", repo_id),
            "repo_url": getattr(intel, "repo_url", ""),
            "head_commit_sha": head_sha or "",
            "total_files": int(getattr(intel, "total_files", 0)),
            "total_loc": int(getattr(intel, "total_lines", 0)),
            "dependency_edges_count": int(gstats.get("total_edges", 0)),
            "languages_json": dict(getattr(intel, "languages", {}) or {}),
            "critical_hotspots_json": hotspots,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        return self.upsert("asre_repositories", row)

    def sync_incident(
        self,
        incident: Any,
        repo_path: str,
        *,
        mapped_files: Sequence[str] = (),
    ) -> SupabaseSyncResult:
        repo_id = _normalize_repo_id(repo_path)
        miner = DrainTemplateMiner()
        exc = getattr(incident, "exception_type", "") or "Error"
        msg = getattr(incident, "message", "")
        tpl = miner.extract_template(f"{exc}: {msg}")
        env_dict = getattr(incident, "environment", {}) or {}
        meta_dict = getattr(incident, "metadata", {}) or {}

        row = {
            "incident_id": str(getattr(incident, "incident_id", "incident")),
            "repo_id": repo_id,
            "source": str(meta_dict.get("source") or getattr(incident, "runtime", "") or "asre"),
            "service": str(env_dict.get("service") or meta_dict.get("service") or "default"),
            "environment": str(env_dict.get("env") or "production"),
            "severity": str(env_dict.get("severity") or "error"),
            "exception_type": exc,
            "message": msg,
            "drain_template_id": tpl.template_id,
            "drain_template": tpl.template,
            "stack_trace": str(getattr(incident, "stack_trace", "") or "")[:15000],
            "trace_id": str(meta_dict.get("trace_id") or ""),
            "span_id": str(meta_dict.get("span_id") or ""),
            "parent_span_id": str(meta_dict.get("parent_span_id") or ""),
            "commit_sha": str(getattr(incident, "commit_sha", "") or ""),
            "mapped_files_json": list(mapped_files),
            "metadata_json": dict(meta_dict),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return self.upsert("asre_incidents", row)

    def sync_cross_incident_report(
        self,
        report: Any,
        repo_path: str,
    ) -> list[SupabaseSyncResult]:
        repo_id = _normalize_repo_id(repo_path)
        results: list[SupabaseSyncResult] = []
        inc_rows: list[dict[str, Any]] = []
        causal_rows: list[dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for grp in getattr(report, "correlated_groups", ()):
            for node in grp.incidents:
                inc_rows.append({
                    "incident_id": node.incident_id,
                    "repo_id": repo_id,
                    "source": "cross_incident",
                    "service": node.service,
                    "environment": "production",
                    "severity": "error",
                    "exception_type": node.exception_type,
                    "message": node.message,
                    "drain_template_id": node.drain_template.template_id,
                    "drain_template": node.drain_template.template,
                    "stack_trace": node.raw_incident.stack_trace[:15000],
                    "trace_id": node.trace_id,
                    "span_id": node.span_id,
                    "parent_span_id": node.parent_span_id,
                    "commit_sha": node.raw_incident.commit_sha,
                    "mapped_files_json": list(node.mapped_files),
                    "metadata_json": {"correlation_id": grp.correlation_id},
                    "created_at": now_iso,
                })
            causal_rows.append({
                "correlation_id": f"{repo_id}:{grp.correlation_id}",
                "repo_id": repo_id,
                "root_trigger_incident_id": grp.root_trigger_incident_id,
                "common_root_cause_file": grp.common_root_cause_file or "",
                "common_root_cause_resource": grp.common_root_cause_resource or "",
                "ppr_score": float(grp.ppr_score),
                "confidence": float(grp.confidence),
                "root_cause_hypothesis": grp.root_cause_hypothesis,
                "correlated_incident_ids_json": [n.incident_id for n in grp.incidents],
                "temporal_causal_chain_json": [l.to_dict() for l in grp.temporal_causal_chain],
                "counterfactual_attributions_json": [a.to_dict() for a in grp.service_causal_attributions],
                "relationship_graph_json": report.relationship_graph.to_dict(),
                "created_at": now_iso,
            })

        if inc_rows:
            results.append(self.upsert("asre_incidents", inc_rows))
        if causal_rows:
            results.append(self.upsert("asre_causal_graphs", causal_rows))
        return results

    def sync_repair_run(
        self,
        run_id: str,
        context: Any,
        *,
        status: str,
        kb_trust_status: str = "unverified",
        promoted_by: str = "",
        plan: Optional[Any] = None,
        execution: Optional[Any] = None,
        application: Optional[Any] = None,
        dead_end_actions: Sequence[str] = (),
        distilled_record: Optional[dict[str, Any]] = None,
        llm_provider: str = "auto",
        llm_model: str = "",
    ) -> list[SupabaseSyncResult]:
        repo_id = _normalize_repo_id(context.repository_path)
        mapped_files = [ev.path for ev in getattr(context, "evidence", ()) if getattr(ev, "patch_eligible", False)]
        results: list[SupabaseSyncResult] = [
            self.sync_incident(context.incident, context.repository_path, mapped_files=mapped_files)
        ]

        diff_text = getattr(plan, "unified_diff", "") if plan else ""
        operator = classify_structural_fix_operator(diff_text)
        rolled_back = bool(getattr(application, "rolled_back", False)) if application else False
        polarity = (
            "positive"
            if status in {"verified", "applied", "pr_created", "pr_branch_ready"} and not rolled_back
            else "negative"
        )
        sim_dict = (getattr(execution, "simulation_report", None) or {}) if execution else {}
        r_action_val = float((sim_dict.get("action_risk") or {}).get("r_action", 0.5))
        exec_band = str((sim_dict.get("decision_matrix") or {}).get("execution_band", ""))
        adj_reg = int((getattr(execution, "impact_summary", {}) or {}).get("adjacent_regressions", 0)) if execution else 0

        row = {
            "run_id": f"{repo_id}:{run_id}",
            "incident_id": str(context.incident.incident_id),
            "repo_id": repo_id,
            "status": status,
            "kb_trust_status": kb_trust_status,
            "promoted_by": promoted_by,
            "trajectory_polarity": polarity,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "diagnosis": getattr(plan, "diagnosis", "") if plan else "",
            "suspected_cause": getattr(plan, "suspected_cause", "") if plan else "",
            "confirmed_evidence_json": list(getattr(plan, "confirmed_evidence", ()) if plan else ()),
            "files_modified_json": list(getattr(plan, "files_to_modify", ()) if plan else mapped_files),
            "unified_diff": diff_text,
            "structural_operator": operator,
            "deliberately_unchanged_json": list(getattr(plan, "deliberately_unchanged", ()) if plan else ()),
            "tests_to_run_json": list(getattr(plan, "tests_to_run", ()) if plan else ()),
            "reproduced_before_patch": bool(getattr(execution, "reproduced_before_patch", False)) if execution else False,
            "incident_replay_passed": bool(getattr(execution, "incident_replay_passed", False)) if execution else False,
            "adjacent_regressions": adj_reg,
            "r_action_score": r_action_val,
            "execution_band": exec_band,
            "simulation_report_json": sim_dict,
            "dead_end_actions_json": list(dead_end_actions),
            "distilled_record_json": distilled_record or {},
            "mttr_seconds": 2.0,
            "post_repair_survived": polarity == "positive",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        results.append(self.upsert("asre_repair_runs", row))
        return results

    def sync_knowledge_base(
        self,
        kb_report: Any,
        repo_path: str,
    ) -> SupabaseSyncResult:
        repo_id = _normalize_repo_id(repo_path)
        now_iso = datetime.now(timezone.utc).isoformat()
        rows: list[dict[str, Any]] = []

        for hit in getattr(kb_report, "tier1_semantic_hits", ()):
            pdoc = hit.parent_doc
            rows.append({
                "entry_id": f"{repo_id}:{pdoc.parent_id}",
                "repo_id": repo_id,
                "tier": "tier1_semantic_doc",
                "title": f"{pdoc.source_path} :: {pdoc.section_title}",
                "structural_operator": pdoc.doc_kind,
                "trajectory_polarity": "positive",
                "kb_trust_status": "verified",
                "drift_status": "clean",
                "alpha": 1.0,
                "beta_param": 1.0,
                "posterior_confidence": float(hit.final_score),
                "thompson_bandit_score": float(hit.cross_encoder_score),
                "payload_json": hit.to_dict(),
                "updated_at": now_iso,
            })

        for case in (*getattr(kb_report, "tier2_positive_cases", ()), *getattr(kb_report, "tier2_negative_cases", ())):
            rows.append({
                "entry_id": f"{repo_id}:{case.case_id}",
                "repo_id": repo_id,
                "tier": "tier2_episodic_case",
                "title": f"{case.incident_id} ({case.exception_type})",
                "structural_operator": case.structural_operator,
                "trajectory_polarity": case.trajectory_polarity,
                "kb_trust_status": case.kb_trust_status,
                "drift_status": case.drift_status,
                "alpha": 2.0 if case.trajectory_polarity == "positive" else 1.0,
                "beta_param": 1.0 if case.trajectory_polarity == "positive" else 3.0,
                "posterior_confidence": float(case.similarity_score),
                "thompson_bandit_score": float(case.similarity_score),
                "payload_json": case.to_dict(),
                "updated_at": now_iso,
            })

        for skill in getattr(kb_report, "tier3_procedural_skills", ()):
            rows.append({
                "entry_id": f"{repo_id}:{skill.skill_id}",
                "repo_id": repo_id,
                "tier": "tier3_procedural_skill",
                "title": skill.name,
                "structural_operator": skill.structural_operator,
                "trajectory_polarity": "positive",
                "kb_trust_status": "verified",
                "drift_status": "clean",
                "alpha": float(skill.alpha),
                "beta_param": float(skill.beta_param),
                "posterior_confidence": float(skill.posterior_confidence),
                "thompson_bandit_score": float(skill.thompson_bandit_score),
                "payload_json": skill.to_dict(),
                "updated_at": now_iso,
            })

        return self.upsert("asre_knowledge_base", rows)

    def sync_change_risk_report(
        self,
        risk_report: Any,
        repo_path: str,
        *,
        incident_caused_label: Optional[int] = None,
    ) -> SupabaseSyncResult:
        repo_id = _normalize_repo_id(repo_path)
        tm = getattr(risk_report, "tabular_metrics", None)
        gm = getattr(risk_report, "graph_metrics", None)
        si = getattr(risk_report, "semantic_inspection", None)
        gov = getattr(risk_report, "governance_decision", None)

        stable_sha = str(getattr(risk_report, "stable_commit", ""))
        target_sha = str(getattr(risk_report, "target_commit", ""))
        check_id = f"{repo_id}:{stable_sha[:8]}..{target_sha[:8]}"
        past_inc_total = sum(getattr(cf, "past_incidents", 0) for cf in getattr(risk_report, "changed_files", ()))

        row = {
            "check_id": check_id,
            "repo_id": repo_id,
            "stable_commit": stable_sha,
            "target_commit": target_sha,
            "lines_added": int(getattr(tm, "lines_added", 0)) if tm else 0,
            "lines_deleted": int(getattr(tm, "lines_deleted", 0)) if tm else 0,
            "files_touched": int(getattr(tm, "files_touched", 0)) if tm else len(getattr(risk_report, "changed_files", ())),
            "subsystems_touched": int(getattr(tm, "subsystems_touched", 0)) if tm else 1,
            "change_entropy": float(getattr(tm, "change_entropy", 0.0)) if tm else 0.0,
            "author_name": str(getattr(tm, "author_name", "")) if tm else "",
            "author_familiarity_pct": float(getattr(tm, "author_familiarity_pct", 100.0)) if tm else 100.0,
            "recent_bugfix_commits": int(getattr(tm, "recent_bugfix_commits", 0)) if tm else 0,
            "is_off_hours_deploy": bool(getattr(tm, "is_off_hours_or_friday_deploy", False)) if tm else False,
            "max_reverse_pagerank": float(getattr(gm, "max_reverse_pagerank", 0.0)) if gm else 0.0,
            "max_betweenness_centrality": float(getattr(gm, "max_betweenness_centrality", 0.0)) if gm else 0.0,
            "total_direct_dependents": int(getattr(gm, "total_direct_dependents", 0)) if gm else 0,
            "total_transitive_dependents": int(getattr(gm, "total_transitive_dependents", 0)) if gm else 0,
            "modified_symbols_count": len(getattr(si, "modified_symbols", ())) if si else 0,
            "config_keys_modified_count": len(getattr(si, "config_keys_modified", ())) if si else 0,
            "ast_hazards_count": len(getattr(si, "ast_semantic_hazards", ())) if si else 0,
            "prior_incidents_count": past_inc_total,
            "calibrated_probability": float(getattr(risk_report, "calibrated_probability", 0.0)),
            "deployment_risk_level": str(getattr(risk_report, "deployment_risk_level", "LOW")),
            "autonomous_action": str(getattr(gov, "autonomous_action", "")) if gov else "",
            "shap_attributions_json": [s.to_dict() for s in getattr(risk_report, "shap_attributions", ())],
            "incident_caused_label": incident_caused_label if incident_caused_label is not None else (1 if past_inc_total > 0 else 0),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return self.upsert("asre_change_risk_records", row)

    def sync_slo_report(
        self,
        slo_report: Any,
        repo_path: str,
    ) -> SupabaseSyncResult:
        repo_id = _normalize_repo_id(repo_path)
        fc = slo_report.forecast
        gov = slo_report.governance_policy
        alerts = slo_report.mwmbr_alerts
        now_iso = datetime.now(timezone.utc).isoformat()
        snap_hash = hashlib.sha1(f"{repo_id}:{fc.consumed_budget_pct:.2f}:{fc.ewma_burn_rate:.2f}".encode("utf-8")).hexdigest()[:10]

        row = {
            "snapshot_id": f"{repo_id}:slo-{snap_hash}",
            "repo_id": repo_id,
            "slo_target_pct": float(fc.slo_target_pct),
            "observed_availability_pct": float(fc.observed_availability_pct),
            "remaining_budget_pct": float(fc.remaining_budget_pct),
            "remaining_budget_minutes": float(fc.remaining_budget_minutes),
            "ewma_burn_rate": float(fc.ewma_burn_rate),
            "time_to_exhaustion_hours": float(fc.time_to_exhaustion_hours) if fc.time_to_exhaustion_hours is not None else None,
            "fast_burn_firing": bool(alerts[0].is_firing) if len(alerts) > 0 else False,
            "high_burn_firing": bool(alerts[1].is_firing) if len(alerts) > 1 else False,
            "slow_burn_firing": bool(alerts[2].is_firing) if len(alerts) > 2 else False,
            "governance_state": gov.governance_state,
            "feature_freeze_active": bool(gov.feature_freeze_active),
            "mwmbr_alerts_json": [a.to_dict() for a in alerts],
            "code_attributions_json": [c.to_dict() for c in slo_report.code_attributions],
            "created_at": now_iso,
        }
        return self.upsert("asre_slo_snapshots", row)

    def sync_audit_log_entry(
        self,
        entry: dict[str, Any],
        repo_path: str = "",
    ) -> SupabaseSyncResult:
        repo_id = _normalize_repo_id(repo_path)
        ts = str(entry.get("timestamp") or datetime.now(timezone.utc).isoformat())
        inc_id = str(entry.get("incident_id") or "incident")
        act = str(entry.get("action") or "audit")
        aid = hashlib.sha1(f"{repo_id}:{ts}:{inc_id}:{act}".encode("utf-8")).hexdigest()[:16]

        row = {
            "audit_id": f"aud-{aid}",
            "repo_id": repo_id,
            "timestamp": ts,
            "incident_id": inc_id,
            "action": act,
            "status": str(entry.get("status") or "unknown"),
            "operator": str(entry.get("operator") or "cli-operator"),
            "role": str(entry.get("role") or "sre"),
            "environment": str(entry.get("environment") or "development"),
            "policy_decision_json": dict(entry.get("policy_decision") or {}),
            "details_json": dict(entry.get("details") or {}),
        }
        return self.upsert("asre_audit_logs", row)

    def sync_historical_runs_and_audit_logs(
        self,
        repo_path: str,
        repair_run_dir: Optional[str] = ".asre/runs",
    ) -> dict[str, Any]:
        """Backfill existing local `.asre/runs/*` repair trajectories and `audit.jsonl` into Supabase."""
        repo_id = _normalize_repo_id(repo_path)
        cases = load_tier2_episodic_cases(repair_run_dir, repo_path)
        run_rows: list[dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for c in cases:
            run_rows.append({
                "run_id": f"{repo_id}:{c.case_id}",
                "incident_id": c.incident_id,
                "repo_id": repo_id,
                "status": c.status,
                "kb_trust_status": c.kb_trust_status,
                "promoted_by": c.promoted_by,
                "trajectory_polarity": c.trajectory_polarity,
                "llm_provider": "auto",
                "llm_model": "",
                "diagnosis": c.diagnosis,
                "suspected_cause": c.diagnosis,
                "confirmed_evidence_json": list(c.affected_files),
                "files_modified_json": list(c.affected_files),
                "unified_diff": c.unified_diff,
                "structural_operator": c.structural_operator,
                "deliberately_unchanged_json": [],
                "tests_to_run_json": list(c.tests_to_run),
                "reproduced_before_patch": c.trajectory_polarity == "positive",
                "incident_replay_passed": c.trajectory_polarity == "positive",
                "adjacent_regressions": 0,
                "r_action_score": 0.18 if c.trajectory_polarity == "positive" else 0.78,
                "execution_band": "FULLY_AUTONOMOUS_EXECUTION" if c.trajectory_polarity == "positive" else "HUMAN_IN_THE_LOOP_APPROVAL",
                "simulation_report_json": {},
                "dead_end_actions_json": list(c.dead_end_actions),
                "distilled_record_json": {"outcome_summary": c.outcome_summary},
                "mttr_seconds": float(c.recovery_time_seconds),
                "post_repair_survived": c.post_repair_survived,
                "created_at": now_iso,
            })

        audit_rows: list[dict[str, Any]] = []
        audit_candidates = []
        if repair_run_dir:
            audit_candidates.append(Path(repair_run_dir).resolve() / "audit.jsonl")
        audit_candidates.append(Path(repo_path).resolve() / ".asre" / "runs" / "audit.jsonl")

        seen_aids: set[str] = set()
        for af in audit_candidates:
            if not af.is_file():
                continue
            try:
                for ln in af.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    entry = json.loads(ln)
                    ts = str(entry.get("timestamp") or now_iso)
                    inc_id = str(entry.get("incident_id") or "incident")
                    act = str(entry.get("action") or "audit")
                    aid = "aud-" + hashlib.sha1(f"{repo_id}:{ts}:{inc_id}:{act}".encode("utf-8")).hexdigest()[:16]
                    if aid in seen_aids:
                        continue
                    seen_aids.add(aid)
                    audit_rows.append({
                        "audit_id": aid,
                        "repo_id": repo_id,
                        "timestamp": ts,
                        "incident_id": inc_id,
                        "action": act,
                        "status": str(entry.get("status") or "unknown"),
                        "operator": str(entry.get("operator") or "cli-operator"),
                        "role": str(entry.get("role") or "sre"),
                        "environment": str(entry.get("environment") or "development"),
                        "policy_decision_json": dict(entry.get("policy_decision") or {}),
                        "details_json": dict(entry.get("details") or {}),
                    })
            except Exception:
                continue

        runs_res = self.upsert("asre_repair_runs", run_rows) if run_rows else SupabaseSyncResult("asre_repair_runs", 0, True, False)
        aud_res = self.upsert("asre_audit_logs", audit_rows) if audit_rows else SupabaseSyncResult("asre_audit_logs", 0, True, False)
        return {
            "repair_runs": runs_res.to_dict(),
            "audit_logs": aud_res.to_dict(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # ML & LLM Training Dataset Exporter (SFT, DPO, Tabular LightGBM/XGBoost)
    # ──────────────────────────────────────────────────────────────────────────

    def export_training_datasets(
        self,
        output_dir: str | Path,
        *,
        repo_root: str = ".",
        repair_run_dir: Optional[str] = ".asre/runs",
    ) -> TrainingExportReport:
        """
        Export stored Supabase + local episodic & change-risk records into ready-to-train
        datasets for future LLM & tabular ML model training:
          1. `sft_repairs.jsonl`: Supervised Fine-Tuning (ChatML messages format)
          2. `dpo_preference_pairs.jsonl`: Direct Preference Optimization (prompt, chosen, rejected)
          3. `jit_change_risk_training.csv` & `.jsonl`: Labeled tabular feature matrix for LightGBM/XGBoost
          4. `causal_graph_trajectories.jsonl`: Multi-hop cross-service causal training trajectories
        """
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        # Combine cloud/mirror repair runs with any local episodic cases
        cases = load_tier2_episodic_cases(repair_run_dir, repo_root)
        cloud_runs = self.select("asre_repair_runs", limit=500)
        seen_case_ids = {c.incident_id + ":" + c.status for c in cases}

        pos_records: list[dict[str, Any]] = []
        neg_records: list[dict[str, Any]] = []

        for c in cases:
            rec = {
                "incident_id": c.incident_id,
                "exception_type": c.exception_type,
                "message": c.error_message,
                "drain_template": c.drain_template,
                "files": list(c.affected_files),
                "diagnosis": c.diagnosis,
                "unified_diff": c.unified_diff,
                "structural_operator": c.structural_operator,
                "tests_to_run": list(c.tests_to_run),
                "status": c.status,
                "dead_end_actions": list(c.dead_end_actions),
                "anti_pattern_warning": c.anti_pattern_warning,
            }
            if c.trajectory_polarity == "positive":
                pos_records.append(rec)
            else:
                neg_records.append(rec)

        for cr in cloud_runs:
            key = f"{cr.get('incident_id')}:{cr.get('status')}"
            if key in seen_case_ids:
                continue
            seen_case_ids.add(key)
            rec = {
                "incident_id": str(cr.get("incident_id") or ""),
                "exception_type": "Error",
                "message": str(cr.get("diagnosis") or ""),
                "drain_template": "",
                "files": list(cr.get("files_modified_json") or []),
                "diagnosis": str(cr.get("diagnosis") or ""),
                "unified_diff": str(cr.get("unified_diff") or ""),
                "structural_operator": str(cr.get("structural_operator") or ""),
                "tests_to_run": list(cr.get("tests_to_run_json") or []),
                "status": str(cr.get("status") or ""),
                "dead_end_actions": list(cr.get("dead_end_actions_json") or []),
                "anti_pattern_warning": f"Failed/rolled-back status: {cr.get('status')}",
            }
            if cr.get("trajectory_polarity") == "positive":
                pos_records.append(rec)
            else:
                neg_records.append(rec)

        # 1. Write SFT Dataset (`sft_repairs.jsonl`)
        sft_file = out_path / "sft_repairs.jsonl"
        sft_lines: list[str] = []
        for pos in pos_records:
            if not pos["unified_diff"] and not pos["diagnosis"]:
                continue
            user_prompt = (
                f"Incident ID: {pos['incident_id']}\n"
                f"Exception: {pos['exception_type']}: {pos['message']}\n"
                f"Target Files: {', '.join(pos['files'])}\n"
            )
            assistant_completion = json.dumps({
                "diagnosis": pos["diagnosis"],
                "structural_operator": pos["structural_operator"],
                "files_to_modify": pos["files"],
                "unified_diff": pos["unified_diff"],
                "tests_to_run": pos["tests_to_run"],
            })
            sft_entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are ASRE, an evidence-based Autonomous Software Reliability Engineer.",
                    },
                    {"role": "user", "content": redact_secrets(user_prompt)},
                    {"role": "assistant", "content": redact_secrets(assistant_completion)},
                ]
            }
            sft_lines.append(json.dumps(sft_entry, sort_keys=True))
        sft_file.write_text("\n".join(sft_lines) + ("\n" if sft_lines else ""), encoding="utf-8")

        # 2. Write DPO Preference Pairs (`dpo_preference_pairs.jsonl`)
        dpo_file = out_path / "dpo_preference_pairs.jsonl"
        dpo_lines: list[str] = []
        for pos in pos_records:
            prompt_str = (
                f"Diagnose and repair incident '{pos['incident_id']}' ({pos['exception_type']}: {pos['message']}) "
                f"in {', '.join(pos['files'])}."
            )
            chosen_str = json.dumps({
                "diagnosis": pos["diagnosis"],
                "unified_diff": pos["unified_diff"],
                "tests_to_run": pos["tests_to_run"],
            })
            # Match against negative records on same file or dead-end actions
            matching_negs = [
                n for n in neg_records
                if set(n["files"]) & set(pos["files"]) or n["incident_id"] == pos["incident_id"]
            ]
            for neg in matching_negs:
                rejected_str = json.dumps({
                    "diagnosis": neg["diagnosis"] or neg["anti_pattern_warning"],
                    "unified_diff": neg["unified_diff"],
                    "status": neg["status"],
                })
                dpo_lines.append(
                    json.dumps(
                        {
                            "prompt": redact_secrets(prompt_str),
                            "chosen": redact_secrets(chosen_str),
                            "rejected": redact_secrets(rejected_str),
                            "rejection_reason": redact_secrets(neg["anti_pattern_warning"]),
                        },
                        sort_keys=True,
                    )
                )
            for de in pos["dead_end_actions"]:
                dpo_lines.append(
                    json.dumps(
                        {
                            "prompt": redact_secrets(prompt_str),
                            "chosen": redact_secrets(chosen_str),
                            "rejected": redact_secrets(de),
                            "rejection_reason": "Failed intermediate repair loop trajectory",
                        },
                        sort_keys=True,
                    )
                )
        dpo_file.write_text("\n".join(dpo_lines) + ("\n" if dpo_lines else ""), encoding="utf-8")

        # 3. Write Tabular JIT Change-Risk Dataset (`jit_change_risk_training.jsonl` & `.csv`)
        risk_rows = self.select("asre_change_risk_records", limit=1000)
        risk_jsonl_file = out_path / "jit_change_risk_training.jsonl"
        risk_csv_file = out_path / "jit_change_risk_training.csv"
        feature_cols = [
            "check_id",
            "repo_id",
            "lines_added",
            "lines_deleted",
            "files_touched",
            "subsystems_touched",
            "change_entropy",
            "author_familiarity_pct",
            "recent_bugfix_commits",
            "is_off_hours_deploy",
            "max_reverse_pagerank",
            "max_betweenness_centrality",
            "total_direct_dependents",
            "total_transitive_dependents",
            "modified_symbols_count",
            "config_keys_modified_count",
            "ast_hazards_count",
            "prior_incidents_count",
            "calibrated_probability",
            "incident_caused_label",
        ]
        risk_jsonl_lines = [json.dumps({k: r.get(k) for k in feature_cols}, sort_keys=True) for r in risk_rows]
        risk_jsonl_file.write_text("\n".join(risk_jsonl_lines) + ("\n" if risk_jsonl_lines else ""), encoding="utf-8")

        with open(risk_csv_file, "w", newline="", encoding="utf-8") as csv_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=feature_cols)
            writer.writeheader()
            for r in risk_rows:
                writer.writerow({k: r.get(k, "") for k in feature_cols})

        # 4. Write Cross-Incident Causal Trajectories (`causal_graph_trajectories.jsonl`)
        causal_rows = self.select("asre_causal_graphs", limit=500)
        causal_file = out_path / "causal_graph_trajectories.jsonl"
        causal_lines = [json.dumps(r, sort_keys=True) for r in causal_rows]
        causal_file.write_text("\n".join(causal_lines) + ("\n" if causal_lines else ""), encoding="utf-8")

        return TrainingExportReport(
            output_dir=str(out_path),
            sft_records_count=len(sft_lines),
            dpo_pairs_count=len(dpo_lines),
            change_risk_rows_count=len(risk_rows),
            causal_graph_records_count=len(causal_rows),
            files_written=(
                str(sft_file),
                str(dpo_file),
                str(risk_jsonl_file),
                str(risk_csv_file),
                str(causal_file),
            ),
        )
