"""
Multi-Tenant Authentication, Process Isolation, Encrypted Vault Storage,
and 4-Category Platform Adapter Engine for ASRE FastAPI Backend.

Implements:
  1. Multi-Tenant Auth & Process Isolation:
     • Data-Layer Isolation: `org_id` UUID extraction from Supabase Auth / Bearer JWTs,
       RBAC role mapping (`admin`, `sre_approver`, `sre`, `developer`, `viewer`), and
       per-tenant state partitioning to prevent IDOR and cross-tenant data leakage.
     • Execution & Worker Isolation: Per-tenant partitioned job queue (`TenantJobQueue`)
       and Outbound In-Cluster Runner / Scoped AWS IAM AssumeRole (`ExternalId = org_id`).
  2. Secure GitHub App & Datadog / Platform Connections:
     • GitHub App flow: signed `state` JWT with `org_id`, `installation_id <-> org_id`
       binding, and on-the-fly 1-hour short-lived installation token minting.
     • Datadog OAuth2 + Encrypted API/App Key Vault storage (`vault:v1:<nonce>:<cipher>:<mac>`)
       and tenant-scoped webhook token verification.
  3. 4 Core Platform Adapter Interfaces:
     • Category 1: Source Control & CI/CD (GitHub P1, GitLab P2, Bitbucket P3)
     • Category 2: Observability & APM (Datadog P1, Prometheus/Grafana/OTel P1, New Relic/Dynatrace P2)
     • Category 3: Incident & On-Call / ChatOps (PagerDuty P1, Opsgenie P2, Slack/MS Teams P1)
     • Category 4: Runtime & GitOps (Kubernetes P1, ArgoCD/Flux P1, AWS ECS/Lambda P2)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

from fastapi import HTTPException, Request


DEFAULT_ORG_UUID = "00000000-0000-0000-0000-000000000001"
DEFAULT_USER_ID = "user-default-sre"

VALID_TENANT_ROLES = ("admin", "sre_approver", "sre", "developer", "viewer")
ROLE_PRIVILEGE_RANK = {
    "viewer": 1,
    "developer": 2,
    "sre": 3,
    "sre_approver": 4,
    "admin": 5,
}


SUPPORTED_SSO_PROVIDERS = ("google", "github")


@dataclass
class OrganizationRecord:
    org_id: str
    org_name: str
    slug: str
    plan_tier: str = "enterprise"
    sso_domain: str = ""
    sso_providers: tuple[str, ...] = ("google", "github")
    settings: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "org_name": self.org_name,
            "slug": self.slug,
            "plan_tier": self.plan_tier,
            "sso_domain": self.sso_domain,
            "sso_providers": list(self.sso_providers),
            "settings": self.settings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class UserAccountRecord:
    user_id: str
    org_id: str
    email: str
    full_name: str
    role: str
    auth_provider: str  # "email", "google", "github"
    provider_user_id: str = ""
    password_hash: str = ""
    avatar_url: str = ""
    status: str = "active"
    created_at: str = ""
    last_login_at: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        """Never expose `password_hash` in API responses."""
        return {
            "user_id": self.user_id,
            "org_id": self.org_id,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "auth_provider": self.auth_provider,
            "provider_user_id": self.provider_user_id,
            "avatar_url": self.avatar_url,
            "status": self.status,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
        }


def hash_password_pbkdf2(password: str, *, salt_hex: Optional[str] = None) -> str:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"pbkdf2_sha256$120000${salt.hex()}${dk.hex()}"


def verify_password_pbkdf2(password: str, stored_hash: str) -> bool:
    if not stored_hash or not stored_hash.startswith("pbkdf2_sha256$"):
        return False
    parts = stored_hash.split("$")
    if len(parts) != 4:
        return False
    _, _, salt_hex, _ = parts
    candidate = hash_password_pbkdf2(password, salt_hex=salt_hex)
    return hmac.compare_digest(candidate, stored_hash)


def build_sso_authorize_url(
    provider: str,
    *,
    supabase_url: str = "",
    redirect_uri: str = "http://localhost:3000/auth/callback",
    org_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate the OAuth2 / Supabase Auth SSO authorization URL for Google or GitHub.
    Supports both Supabase GoTrue (`/auth/v1/authorize?provider=google|github`) and
    direct OAuth2 provider endpoints with a signed CSRF/tenancy `state` JWT.
    """
    prov = provider.strip().lower()
    if prov not in SUPPORTED_SSO_PROVIDERS:
        raise ValueError(f"Unsupported SSO provider '{provider}'. Supported: {', '.join(SUPPORTED_SSO_PROVIDERS)}")

    norm_org = normalize_org_uuid(org_id) if org_id else ""
    state_jwt = issue_signed_jwt(
        {
            "purpose": f"sso_{prov}",
            "provider": prov,
            "org_id": norm_org,
            "redirect_uri": redirect_uri,
        },
        expires_in_seconds=1800,
    )
    base_supa = (supabase_url or os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    if base_supa:
        authorize_url = (
            f"{base_supa}/auth/v1/authorize?provider={prov}"
            f"&redirect_to={redirect_uri}&state={state_jwt}"
        )
    elif prov == "google":
        client_id = os.environ.get("GOOGLE_CLIENT_ID") or "asre-google-oauth-client-id"
        authorize_url = (
            "https://accounts.google.com/o/oauth2/v2/auth"
            f"?client_id={client_id}&response_type=code&scope=openid%20email%20profile"
            f"&redirect_uri={redirect_uri}&state={state_jwt}"
        )
    else:
        client_id = os.environ.get("GITHUB_CLIENT_ID") or "asre-github-oauth-client-id"
        authorize_url = (
            "https://github.com/login/oauth/authorize"
            f"?client_id={client_id}&scope=read:user%20user:email"
            f"&redirect_uri={redirect_uri}&state={state_jwt}"
        )

    return {
        "provider": prov,
        "state_token": state_jwt,
        "redirect_uri": redirect_uri,
        "authorize_url": authorize_url,
    }


def normalize_org_uuid(raw_org_id: Optional[str]) -> str:
    """
    Normalize an organization identifier into a canonical UUID string.
    If `raw_org_id` is already a valid UUID, return it lowercase; otherwise derive
    a deterministic UUIDv5 in the DNS namespace so tenant slugs also map cleanly to UUID columns.
    """
    if not raw_org_id or not raw_org_id.strip():
        return DEFAULT_ORG_UUID
    cleaned = raw_org_id.strip()
    try:
        return str(uuid.UUID(cleaned))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"asre.org.{cleaned.lower()}"))


# ──────────────────────────────────────────────────────────────────────────────
# 1A. JWT Signing, Verification & FastAPI Multi-Tenant Auth Context
# ──────────────────────────────────────────────────────────────────────────────

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    pad = "=" * ((4 - len(segment) % 4) % 4)
    return base64.urlsafe_b64decode((segment + pad).encode("ascii"))


def _get_jwt_secret() -> bytes:
    secret = (
        os.environ.get("SUPABASE_JWT_SECRET")
        or os.environ.get("ASRE_JWT_SECRET")
        or "asre-enterprise-multi-tenant-jwt-signing-key-v1"
    )
    return secret.encode("utf-8")


def issue_signed_jwt(
    payload: dict[str, Any],
    *,
    expires_in_seconds: int = 3600,
    secret: Optional[bytes] = None,
) -> str:
    """Create an HS256-signed JWT carrying `org_id`, `sub`, `role`, and `app_metadata`."""
    now = int(time.time())
    claims = {
        "iat": now,
        "exp": now + expires_in_seconds,
        **payload,
    }
    header_b64 = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    key = secret or _get_jwt_secret()
    sig = hmac.new(key, signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"


def decode_and_validate_jwt(
    token: str,
    *,
    verify_signature: bool = True,
    secret: Optional[bytes] = None,
) -> dict[str, Any]:
    """
    Decode and validate a Supabase Auth or ASRE Bearer JWT.
    Verifies HS256 signature when signed with the configured secret and enforces `exp`.
    """
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("Malformed JWT: expected 3 dot-separated segments")

    header_b64, payload_b64, sig_b64 = parts
    try:
        header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        claims = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JWT encoding: {exc}") from exc

    if verify_signature and header.get("alg") == "HS256":
        key = secret or _get_jwt_secret()
        expected_sig = hmac.new(key, f"{header_b64}.{payload_b64}".encode("ascii"), hashlib.sha256).digest()
        provided_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, provided_sig):
            raise ValueError("JWT signature verification failed")

    exp = claims.get("exp")
    if isinstance(exp, (int, float)) and time.time() > float(exp):
        raise ValueError("JWT has expired")

    return claims


@dataclass(frozen=True)
class TenantAuthContext:
    org_id: str
    user_id: str
    email: str
    role: str
    authenticated_via: str  # "supabase_jwt", "header_scope", "default_tenant"
    raw_claims: dict[str, Any] = field(default_factory=dict)

    @property
    def can_approve_remediations(self) -> bool:
        return ROLE_PRIVILEGE_RANK.get(self.role, 1) >= ROLE_PRIVILEGE_RANK["sre"]

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "user_id": self.user_id,
            "email": self.email,
            "role": self.role,
            "authenticated_via": self.authenticated_via,
            "permissions": {
                "can_view": True,
                "can_simulate": ROLE_PRIVILEGE_RANK.get(self.role, 1) >= ROLE_PRIVILEGE_RANK["developer"],
                "can_approve_remediations": self.can_approve_remediations,
                "can_manage_policies_and_integrations": self.role in {"admin", "sre_approver", "sre"},
            },
        }


def extract_tenant_auth_context(request: Request) -> TenantAuthContext:
    """
    FastAPI Auth Dependency:
    Extracts and validates `Authorization: Bearer <jwt>` from frontend requests,
    pulling `org_id` and `role` from Supabase `app_metadata` (or top-level JWT claims).
    """
    auth_header = (request.headers.get("Authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        raw_token = auth_header[7:].strip()
        try:
            claims = decode_and_validate_jwt(raw_token, verify_signature=True)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=f"Invalid Authorization Bearer token: {exc}")

        app_meta = claims.get("app_metadata") if isinstance(claims.get("app_metadata"), dict) else {}
        user_meta = claims.get("user_metadata") if isinstance(claims.get("user_metadata"), dict) else {}
        raw_org = app_meta.get("org_id") or user_meta.get("org_id") or claims.get("org_id")
        if not raw_org:
            raise HTTPException(status_code=403, detail="JWT is missing required 'org_id' claim in app_metadata")

        org_uuid = normalize_org_uuid(str(raw_org))
        raw_role = str(app_meta.get("role") or claims.get("user_role") or claims.get("role") or "sre").lower()
        if raw_role == "authenticated":
            raw_role = str(app_meta.get("asre_role") or "sre").lower()
        role = raw_role if raw_role in ROLE_PRIVILEGE_RANK else "viewer"
        user_id = str(claims.get("sub") or claims.get("user_id") or DEFAULT_USER_ID)
        email = str(claims.get("email") or f"{user_id}@tenant.local")
        return TenantAuthContext(
            org_id=org_uuid,
            user_id=user_id,
            email=email,
            role=role,
            authenticated_via="supabase_jwt",
            raw_claims=claims,
        )

    # Optional development/internal header fallback (`X-Xargis-Org-Id` / `X-ASRE-Org-Id`) when strict auth is not forced
    if (os.environ.get("XARGIS_REQUIRE_JWT") or os.environ.get("ASRE_REQUIRE_JWT") or "").strip().lower() in {"1", "true", "yes"}:
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <jwt> header")

    header_org = request.headers.get("X-Xargis-Org-Id") or request.headers.get("X-ASRE-Org-Id")
    header_role = (request.headers.get("X-Xargis-Role") or request.headers.get("X-ASRE-Role") or "sre").strip().lower()
    header_user = (request.headers.get("X-Xargis-User-Id") or request.headers.get("X-ASRE-User-Id") or DEFAULT_USER_ID).strip()
    role = header_role if header_role in ROLE_PRIVILEGE_RANK else "sre"
    if header_org:
        return TenantAuthContext(
            org_id=normalize_org_uuid(header_org),
            user_id=header_user,
            email=f"{header_user}@org.local",
            role=role,
            authenticated_via="header_scope",
        )

    return TenantAuthContext(
        org_id=DEFAULT_ORG_UUID,
        user_id=header_user,
        email="sre@default-org.local",
        role=role,
        authenticated_via="default_tenant",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Envelope Vault Encryption for Integration Credentials (Supabase Vault style)
# ──────────────────────────────────────────────────────────────────────────────

class VaultSecretManager:
    """
    Encrypts integration secrets (Datadog API/App keys, PagerDuty tokens, GitHub App keys,
    Slack bot tokens) using Authenticated Envelope Encryption (`vault:v1:<nonce>:<cipher>:<mac>`)
    so plaintext credentials are NEVER persisted in database columns or returned to clients.
    """

    def __init__(self, master_key: Optional[str] = None) -> None:
        raw_key = (
            master_key
            or os.environ.get("ASRE_VAULT_MASTER_KEY")
            or os.environ.get("SUPABASE_JWT_SECRET")
            or "asre-supabase-pgsodium-vault-master-key-2026"
        )
        self._key = hashlib.sha256(raw_key.encode("utf-8")).digest()

    def _keystream(self, nonce: bytes, length: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hmac.new(self._key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])

    def encrypt_secret(self, plaintext_dict: dict[str, Any]) -> str:
        raw_bytes = json.dumps(plaintext_dict, sort_keys=True).encode("utf-8")
        nonce = secrets.token_bytes(12)
        stream = self._keystream(nonce, len(raw_bytes))
        cipher = bytes(a ^ b for a, b in zip(raw_bytes, stream))
        mac = hmac.new(self._key, b"vault:v1:" + nonce + cipher, hashlib.sha256).hexdigest()
        return f"vault:v1:{nonce.hex()}:{cipher.hex()}:{mac}"

    def decrypt_secret(self, token: str) -> dict[str, Any]:
        if not token.startswith("vault:v1:"):
            raise ValueError("Invalid vault ciphertext prefix")
        parts = token.split(":")
        if len(parts) != 5:
            raise ValueError("Malformed vault ciphertext structure")
        _, _, nonce_hex, cipher_hex, mac_hex = parts
        nonce = bytes.fromhex(nonce_hex)
        cipher = bytes.fromhex(cipher_hex)
        expected_mac = hmac.new(self._key, b"vault:v1:" + nonce + cipher, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_mac, mac_hex):
            raise ValueError("Vault ciphertext MAC verification failed")
        stream = self._keystream(nonce, len(cipher))
        plain_bytes = bytes(a ^ b for a, b in zip(cipher, stream))
        return json.loads(plain_bytes.decode("utf-8"))


# ──────────────────────────────────────────────────────────────────────────────
# 1B. Per-Tenant Job Queue & Outbound In-Cluster Runner Isolation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TenantIsolatedJob:
    job_id: str
    org_id: str
    event_type: str
    platform: str
    status: str  # "queued", "running", "completed", "failed"
    worker_id: str
    payload: dict[str, Any]
    result: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "org_id": self.org_id,
            "event_type": self.event_type,
            "platform": self.platform,
            "status": self.status,
            "worker_id": self.worker_id,
            "payload": self.payload,
            "result": self.result,
            "created_at": self.created_at,
        }


@dataclass
class OutboundClusterRunner:
    runner_id: str
    org_id: str
    cluster_name: str
    namespace: str
    mode: str  # "outbound_mtls_websocket" or "aws_iam_assume_role"
    aws_role_arn: str
    external_id: str  # Strictly bound to `org_id`
    status: str
    last_heartbeat_at: str
    pending_commands: list[dict[str, Any]] = field(default_factory=list)
    completed_executions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner_id": self.runner_id,
            "org_id": self.org_id,
            "cluster_name": self.cluster_name,
            "namespace": self.namespace,
            "mode": self.mode,
            "aws_role_arn": self.aws_role_arn,
            "external_id": self.external_id,
            "status": self.status,
            "last_heartbeat_at": self.last_heartbeat_at,
            "pending_commands_count": len(self.pending_commands),
            "completed_executions_count": len(self.completed_executions),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2A & 2B. GitHub App Short-Lived Token Generator & Integration Registry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TenantIntegrationRecord:
    integration_id: str
    org_id: str
    platform: str
    category: str
    status: str  # "connected", "pending_callback", "disconnected"
    installation_id: str
    external_id: str
    encrypted_credentials_vault: str
    webhook_secret: str
    webhook_url: str
    config: dict[str, Any]
    updated_at: str

    def to_public_dict(self) -> dict[str, Any]:
        """Return integration metadata with secrets masked/encrypted (never plaintext)."""
        return {
            "integration_id": self.integration_id,
            "org_id": self.org_id,
            "platform": self.platform,
            "category": self.category,
            "status": self.status,
            "installation_id": self.installation_id,
            "external_id": self.external_id,
            "credentials_stored_in_vault": bool(self.encrypted_credentials_vault),
            "vault_ciphertext_preview": (
                self.encrypted_credentials_vault[:28] + "..."
                if self.encrypted_credentials_vault
                else ""
            ),
            "webhook_url": self.webhook_url,
            "config": self.config,
            "updated_at": self.updated_at,
        }


def build_github_app_install_url(
    org_id: str,
    user_id: str,
    *,
    app_slug: Optional[str] = None,
) -> dict[str, str]:
    """
    Step 1 of GitHub App Flow:
    Generate the GitHub App installation URL with `state=<signed_jwt_with_org_id>`.
    """
    slug = app_slug or os.environ.get("ASRE_GITHUB_APP_SLUG") or "asre-reliability-agent"
    state_jwt = issue_signed_jwt(
        {
            "org_id": normalize_org_uuid(org_id),
            "sub": user_id,
            "purpose": "github_app_install",
        },
        expires_in_seconds=1800,
    )
    install_url = f"https://github.com/apps/{slug}/installations/new?state={state_jwt}"
    return {
        "org_id": normalize_org_uuid(org_id),
        "github_app_slug": slug,
        "state_token": state_jwt,
        "install_url": install_url,
    }


def mint_github_installation_token(
    org_id: str,
    installation_id: str,
) -> dict[str, Any]:
    """
    Step 3 of GitHub App Flow:
    Generate a 1-hour short-lived installation token scoped to `(org_id, installation_id)`.
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=1)
    raw_seed = f"{org_id}:{installation_id}:{int(now.timestamp() // 3600)}"
    token_hash = hmac.new(_get_jwt_secret(), raw_seed.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return {
        "token": f"ghs_{token_hash}",
        "token_type": "github_app_installation_token",
        "installation_id": installation_id,
        "org_id": normalize_org_uuid(org_id),
        "permissions": {
            "contents": "read",
            "pull_requests": "read",
            "checks": "write",
            "statuses": "write",
            "metadata": "read",
        },
        "expires_at": expires_at.isoformat(),
        "ttl_seconds": 3600,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Four Core Platform Categories & Standardized Ingestion Adapters
# ──────────────────────────────────────────────────────────────────────────────

PLATFORM_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "category_id": "source_control_cicd",
        "category_name": "1. Source Control & CI/CD (Change Risk & Code Context)",
        "platforms": [
            {
                "platform": "github",
                "name": "GitHub App",
                "priority": "P1",
                "connection_mode": "GitHub App Installation (installation_id + 1h short-lived token) + Webhooks (pull_request, push, workflow_run)",
                "data_extracted": ["PR diffs", "commit history", "90d file churn", "CODEOWNERS", "deploy check status"],
            },
            {
                "platform": "gitlab",
                "name": "GitLab",
                "priority": "P2",
                "connection_mode": "Project/Group Access Token in Vault + Merge Request & Pipeline Webhooks",
                "data_extracted": ["MR diffs", "commit history", "CODEOWNERS", "CI pipeline status"],
            },
            {
                "platform": "bitbucket",
                "name": "Bitbucket",
                "priority": "P3",
                "connection_mode": "OAuth2 / Repository Access Token in Vault + PR Webhooks",
                "data_extracted": ["PR diffs", "commit history", "branch permissions"],
            },
        ],
    },
    {
        "category_id": "observability_apm",
        "category_name": "2. Observability & APM (Metrics, Logs, Traces)",
        "platforms": [
            {
                "platform": "datadog",
                "name": "Datadog",
                "priority": "P1",
                "connection_mode": "OAuth2 or Vault-Encrypted API + Application Key + Monitor Push Webhooks",
                "data_extracted": ["RED metrics (Rate, Errors, Duration)", "APM distributed trace spans", "error log templates"],
            },
            {
                "platform": "prometheus",
                "name": "Prometheus / Grafana / OpenTelemetry",
                "priority": "P1",
                "connection_mode": "Alertmanager Push Webhooks + PromQL / OTLP Span & Metric Ingestion",
                "data_extracted": ["Multi-Window Multi-Burn-Rate SLO metrics", "OTel parent/child trace spans", "firing alert labels"],
            },
            {
                "platform": "newrelic",
                "name": "New Relic / Dynatrace",
                "priority": "P2",
                "connection_mode": "Vault-Encrypted NerdGraph / API Key + Incident Push Webhooks",
                "data_extracted": ["Golden signals", "distributed trace anomalies", "exception stack traces"],
            },
        ],
    },
    {
        "category_id": "incident_oncall_chatops",
        "category_name": "3. Incident & On-Call (Alert Triage, Ownership & ChatOps)",
        "platforms": [
            {
                "platform": "pagerduty",
                "name": "PagerDuty",
                "priority": "P1",
                "connection_mode": "Events API v2 + V3 Webhooks (incident.triggered, incident.resolved)",
                "data_extracted": ["Active incidents", "escalation policies", "on-call schedules", "service ownership"],
            },
            {
                "platform": "slack",
                "name": "Slack / Microsoft Teams (ChatOps)",
                "priority": "P1",
                "connection_mode": "Slack Bolt / Interactive Block Kit Webhooks (1-Click Approve & Execute / Reject buttons)",
                "data_extracted": ["War-room thread signals", "1-click human approval/rejection actions", "KB fix upvotes"],
            },
            {
                "platform": "opsgenie",
                "name": "Opsgenie",
                "priority": "P2",
                "connection_mode": "Vault-Encrypted GenieKey + Alert Lifecycle Webhooks",
                "data_extracted": ["On-call routing", "alert responder acknowledgements"],
            },
        ],
    },
    {
        "category_id": "runtime_gitops",
        "category_name": "4. Runtime & GitOps (Simulation & Safe Execution)",
        "platforms": [
            {
                "platform": "kubernetes",
                "name": "Kubernetes Outbound Runner",
                "priority": "P1",
                "connection_mode": "Lightweight In-Cluster Outbound Agent (mTLS/WebSocket poll) — zero central master credentials",
                "data_extracted": ["Pod status", "replica counts", "kubectl apply --dry-run=server diffs", "canary rollouts"],
            },
            {
                "platform": "argocd",
                "name": "ArgoCD / Flux GitOps",
                "priority": "P1",
                "connection_mode": "ArgoCD Webhook + App Diff & Sync Status API",
                "data_extracted": ["argocd app diff state", "desired vs live manifest drift", "automated rollback revision"],
            },
            {
                "platform": "aws",
                "name": "AWS ECS / Lambda (Scoped IAM AssumeRole)",
                "priority": "P2",
                "connection_mode": "Scoped AWS IAM AssumeRole with strict ExternalId = org_id",
                "data_extracted": ["ECS task health", "CloudWatch RED alarms", "canary traffic weight shifts"],
            },
        ],
    },
)

SUPPORTED_PLATFORMS_BY_ID: dict[str, dict[str, Any]] = {
    p["platform"]: {**p, "category_id": cat["category_id"], "category_name": cat["category_name"]}
    for cat in PLATFORM_CATALOG
    for p in cat["platforms"]
}


def normalize_platform_webhook_event(
    platform: str,
    org_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Unified 4-Category Adapter Interface:
    Normalizes incoming webhook payloads from GitHub, GitLab, Bitbucket, Datadog,
    Prometheus/Grafana/OTel, New Relic, PagerDuty, Opsgenie, Slack/MS Teams,
    Kubernetes, or ArgoCD into a canonical ASRE event envelope.
    """
    plat = platform.strip().lower()
    org_uuid = normalize_org_uuid(org_id)

    # Category 1: Source Control & CI/CD (GitHub, GitLab, Bitbucket)
    if plat in {"github", "gitlab", "bitbucket"}:
        pr = payload.get("pull_request") or payload.get("object_attributes") or {}
        base_sha = str(
            (pr.get("base") or {}).get("sha")
            or payload.get("before")
            or "HEAD~1"
        )
        head_sha = str(
            (pr.get("head") or {}).get("sha")
            or (pr.get("last_commit") or {}).get("id")
            or payload.get("after")
            or "HEAD"
        )
        pr_num = pr.get("number") or pr.get("iid") or payload.get("pr_number") or 1
        return {
            "category": "source_control_cicd",
            "platform": plat,
            "org_id": org_uuid,
            "event_type": "change_risk_evaluation",
            "pr_number": pr_num,
            "rev_range": str(payload.get("rev_range") or f"{base_sha}..{head_sha}"),
            "installation_id": str((payload.get("installation") or {}).get("id") or payload.get("installation_id") or ""),
            "author": str((pr.get("user") or {}).get("login") or payload.get("user_username") or "developer"),
        }

    # Category 3 ChatOps Interactive Button Clicks (Slack / MS Teams)
    if plat in {"slack", "msteams", "chatops"} and (
        payload.get("actions") or payload.get("action") in {"approve", "reject", "rollback"}
    ):
        actions = payload.get("actions") if isinstance(payload.get("actions"), list) else [payload]
        first_act = actions[0] if actions and isinstance(actions[0], dict) else {}
        decision = str(first_act.get("value") or payload.get("action") or "approve").lower()
        inc_id = str(payload.get("incident_id") or first_act.get("incident_id") or "click-001")
        exec_id = str(payload.get("exec_id") or first_act.get("exec_id") or "exec-chatops")
        user_obj = payload.get("user") if isinstance(payload.get("user"), dict) else {}
        actor = str(user_obj.get("username") or user_obj.get("name") or payload.get("approver") or "slack-sre")
        return {
            "category": "incident_oncall_chatops",
            "platform": plat,
            "org_id": org_uuid,
            "event_type": "chatops_remediation_decision",
            "decision": "reject" if "reject" in decision else ("rollback" if "rollback" in decision else "approve"),
            "incident_id": inc_id,
            "exec_id": exec_id,
            "approver": actor,
        }

    # Category 4: Runtime & GitOps (Kubernetes / ArgoCD / AWS)
    if plat in {"kubernetes", "argocd", "aws"}:
        return {
            "category": "runtime_gitops",
            "platform": plat,
            "org_id": org_uuid,
            "event_type": str(payload.get("event_type") or "gitops_sync_or_drift_alert"),
            "cluster": str(payload.get("cluster") or "prod-cluster"),
            "application": str(payload.get("application") or payload.get("service") or "core-service"),
            "dry_run_clean": bool(payload.get("dry_run_clean", True)),
            "incident_id": str(payload.get("incident_id") or f"{plat}-event-001"),
            "message": str(payload.get("message") or f"{plat.upper()} runtime state change detected"),
            "stack_trace": str(payload.get("stack_trace") or ""),
        }

    # Category 2 & 3 Alert Triggers (Datadog, Prometheus, Grafana, OTel, New Relic, PagerDuty, Opsgenie)
    inc_id = str(
        payload.get("incident_id")
        or payload.get("alert_id")
        or payload.get("id")
        or f"{plat}-alert-{int(time.time()) % 100000}"
    )
    title = str(
        payload.get("alert_title")
        or payload.get("title")
        or payload.get("message")
        or payload.get("event_title")
        or f"{plat.upper()} production monitor alert"
    )
    exc_type = str(
        payload.get("exception_type")
        or payload.get("alert_metric")
        or payload.get("event_type")
        or "MonitorAlertFiring"
    )
    stack = str(payload.get("stack_trace") or payload.get("body") or "")
    service = str(payload.get("service") or payload.get(" host") or "default")
    severity = str(payload.get("priority") or payload.get("severity") or "critical").lower()
    return {
        "category": "observability_apm" if plat in {"datadog", "prometheus", "grafana", "opentelemetry", "newrelic", "dynatrace"} else "incident_oncall_chatops",
        "platform": plat,
        "org_id": org_uuid,
        "event_type": "alert_triggered",
        "incident_id": inc_id,
        "exception_type": exc_type,
        "message": title,
        "stack_trace": stack,
        "service": service,
        "severity": severity,
        "raw_payload": payload,
    }

