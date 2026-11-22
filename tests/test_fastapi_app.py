import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import ASREWorkspaceState, create_fastapi_app


class TestFastAPIBackend(unittest.TestCase):
    def test_all_war_room_changes_services_memory_policies_analytics_and_webhooks_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "payment_platform"
            repo.mkdir()
            (repo / "requirements.txt").write_text("click>=8.1\nfastapi>=0.110\n", encoding="utf-8")
            (repo / ".github").mkdir()
            (repo / ".github" / "CODEOWNERS").write_text(
                "src/payments/* @payments-core-team @alice-sre\n"
                "src/auth/* @security-team\n",
                encoding="utf-8",
            )
            (repo / "src" / "payments").mkdir(parents=True)
            (repo / "src" / "auth").mkdir(parents=True)
            (repo / "src" / "payments" / "ledger.py").write_text(
                "def charge_card(amount: int) -> int:\n    if amount < 0:\n        raise ValueError('negative amount')\n    return amount\n",
                encoding="utf-8",
            )
            (repo / "src" / "auth" / "jwt_guard.py").write_text(
                "def check_jwt(token: str) -> bool:\n    return bool(token)\n",
                encoding="utf-8",
            )

            # Initialize git repository with 2 commits so pre-deployment change risk works on HEAD~1..HEAD
            subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "sre@example.com"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "SRE Bot"], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "initial commit"], check=True)

            (repo / "src" / "payments" / "ledger.py").write_text(
                "def charge_card(amount: int) -> int:\n    if amount <= 0:\n        raise ValueError('non-positive amount')\n    return amount\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fix zero amount check in charge_card"], check=True)

            state = ASREWorkspaceState(
                str(repo),
                repair_run_dir=str(repo / ".asre" / "runs"),
                private_mode=True,
            )
            client = TestClient(create_fastapi_app(state))

            # System health
            r_health = client.get("/api/v1/health")
            self.assertEqual(200, r_health.status_code)
            self.assertEqual("ok", r_health.json()["status"])

            # F. Ingestion Webhooks -> POST /api/v1/webhooks/alerts
            r_alert = client.post(
                "/api/v1/webhooks/alerts",
                json={
                    "incident_id": "inc-pay-500",
                    "exception_type": "ValueError",
                    "message": "non-positive amount in charge_card",
                    "stack_trace": '  File "src/payments/ledger.py", line 3, in charge_card\n    raise ValueError("non-positive amount")\nValueError: non-positive amount',
                },
            )
            self.assertEqual(200, r_alert.status_code)
            self.assertTrue(r_alert.json()["webhook_processed"])
            inc_id = r_alert.json()["incident_id"]

            # A. War-Room & Active Incident Endpoints
            r_list_inc = client.get("/api/v1/incidents")
            self.assertEqual(200, r_list_inc.status_code)
            self.assertGreaterEqual(r_list_inc.json()["total"], 1)

            r_war = client.get(f"/api/v1/incidents/{inc_id}/war-room")
            self.assertEqual(200, r_war.status_code)
            war_data = r_war.json()
            self.assertEqual("@payments-core-team", war_data["routed_owner_team"])
            self.assertIn("causal_root_cause_graph", war_data)
            self.assertIn("matched_historical_memory", war_data)
            self.assertEqual(5, len(war_data["agent_reasoning_steps"]))

            r_sim = client.post(
                f"/api/v1/incidents/{inc_id}/simulate",
                json={
                    "diagnosis": "Clamp negative payment amounts to zero",
                    "files_to_modify": ["src/payments/ledger.py"],
                    "unified_diff": "--- a/src/payments/ledger.py\n+++ b/src/payments/ledger.py\n@@ -1,4 +1,4 @@\n def charge_card(amount: int) -> int:\n-    if amount <= 0:\n+    if amount < 0:\n         raise ValueError('non-positive amount')\n     return amount\n",
                    "tests_to_run": ["python -m unittest"],
                },
            )
            self.assertEqual(200, r_sim.status_code)
            exec_id = r_sim.json()["exec_id"]
            self.assertIn("action_risk", r_sim.json()["simulation_report"])

            r_approve = client.post(
                f"/api/v1/incidents/{inc_id}/remediations/{exec_id}/approve",
                json={"decision": "approve", "approver": "alice-sre", "role": "sre", "environment": "production"},
            )
            self.assertEqual(200, r_approve.status_code)
            self.assertEqual("approved_and_verified", r_approve.json()["status"])
            self.assertTrue(r_approve.json()["watchdog"]["active"])

            r_rb = client.post(
                f"/api/v1/incidents/{inc_id}/remediations/{exec_id}/rollback",
                json={"operator": "alice-sre", "reason": "Latency spike detected in canary"},
            )
            self.assertEqual(200, r_rb.status_code)
            self.assertEqual("rolled_back", r_rb.json()["status"])

            # B. Predictive Change Risk & GitHub PR Endpoints
            r_gh = client.post(
                "/api/v1/webhooks/github",
                json={"pr_number": 42, "rev_range": "HEAD~1..HEAD"},
            )
            self.assertEqual(200, r_gh.status_code)
            self.assertEqual("pr-42", r_gh.json()["deployment_id"])

            r_deps = client.get("/api/v1/changes/deployments")
            self.assertEqual(200, r_deps.status_code)
            self.assertGreaterEqual(r_deps.json()["total"], 1)

            r_dep_detail = client.get("/api/v1/changes/deployments/pr-42")
            self.assertEqual(200, r_dep_detail.status_code)
            self.assertIn("shap_attributions", r_dep_detail.json())

            r_override = client.post(
                "/api/v1/changes/deployments/pr-42/override",
                json={"approver": "vp-eng@example.com", "role": "sre", "reason": "Emergency hotfix approved"},
            )
            self.assertEqual(200, r_override.status_code)
            self.assertTrue(r_override.json()["overridden"])

            # C. Service Topology, SLOs & Ownership
            r_topo = client.get("/api/v1/services/topology")
            self.assertEqual(200, r_topo.status_code)
            self.assertGreaterEqual(r_topo.json()["total_nodes"], 2)
            self.assertIn("slo_summary", r_topo.json())

            r_svc_sc = client.get("/api/v1/services/src/payments/scorecard")
            self.assertEqual(200, r_svc_sc.status_code)
            self.assertEqual("@payments-core-team", r_svc_sc.json()["owner_team"])
            self.assertIn("dora_and_slo_metrics", r_svc_sc.json())

            # D. Repair Memory & Governance Policies
            r_mem = client.get("/api/v1/memory/repairs?query=ValueError")
            self.assertEqual(200, r_mem.status_code)
            self.assertIn("tier2_episodic_repairs", r_mem.json())

            r_verify = client.patch(
                f"/api/v1/memory/repairs/{inc_id}/verify",
                json={"approver": "alice-sre", "status": "verified"},
            )
            self.assertEqual(200, r_verify.status_code)
            self.assertEqual("verified", r_verify.json()["kb_trust_status"])

            r_pol_get = client.get("/api/v1/policies")
            self.assertEqual(200, r_pol_get.status_code)
            self.assertIn("custom_repair_policy", r_pol_get.json())

            r_pol_put = client.put(
                "/api/v1/policies",
                json={"max_files_per_patch": 4, "read_only_shadow_mode": True, "operator": "cto"},
            )
            self.assertEqual(200, r_pol_put.status_code)
            self.assertEqual(4, r_pol_put.json()["custom_repair_policy"]["max_files_per_patch"])
            self.assertTrue(r_pol_put.json()["read_only_shadow_mode"])

            # E. Executive Analytics & Cost ROI
            r_analytics = client.get("/api/v1/analytics/overview")
            self.assertEqual(200, r_analytics.status_code)
            self.assertIn("headline_roi", r_analytics.json())
            self.assertGreater(r_analytics.json()["headline_roi"]["estimated_incident_cost_saved_usd"], 0.0)

            r_audit = client.get("/api/v1/analytics/audit-logs")
            self.assertEqual(200, r_audit.status_code)
            self.assertGreaterEqual(r_audit.json()["total"], 3)

            # G. Multi-Tenant Auth, Cross-Tenant IDOR Isolation, GitHub App, Datadog Vault & 4-Category Platforms
            tok_a_resp = client.post(
                "/api/v1/auth/token",
                json={"org_id": "11111111-1111-1111-1111-111111111111", "user_id": "alice-org-a", "role": "admin"},
            )
            self.assertEqual(200, tok_a_resp.status_code)
            jwt_a = tok_a_resp.json()["access_token"]
            org_a = tok_a_resp.json()["org_id"]

            tok_b_resp = client.post(
                "/api/v1/auth/token",
                json={"org_id": "22222222-2222-2222-2222-222222222222", "user_id": "bob-org-b", "role": "sre"},
            )
            self.assertEqual(200, tok_b_resp.status_code)
            jwt_b = tok_b_resp.json()["access_token"]

            headers_a = {"Authorization": f"Bearer {jwt_a}"}
            headers_b = {"Authorization": f"Bearer {jwt_b}"}

            # Verify /api/v1/auth/me and 4-category platform catalog
            me_a = client.get("/api/v1/auth/me", headers=headers_a)
            self.assertEqual(200, me_a.status_code)
            self.assertEqual(org_a, me_a.json()["org_id"])

            cat_a = client.get("/api/v1/platforms/catalog", headers=headers_a)
            self.assertEqual(200, cat_a.status_code)
            self.assertEqual(4, cat_a.json()["total_categories"])

            # GitHub App Flow: install-url -> callback -> 1h short-lived token
            gh_url_resp = client.get("/api/v1/integrations/github/install-url", headers=headers_a)
            self.assertEqual(200, gh_url_resp.status_code)
            state_jwt = gh_url_resp.json()["state_token"]

            gh_cb_resp = client.post(
                "/api/v1/integrations/github/callback",
                json={"installation_id": "inst-998877", "state": state_jwt},
            )
            self.assertEqual(200, gh_cb_resp.status_code)
            self.assertTrue(gh_cb_resp.json()["connected"])
            self.assertTrue(gh_cb_resp.json()["ephemeral_installation_token"]["token"].startswith("ghs_"))

            # Datadog Connection Flow: encrypted Vault storage + tenant-scoped webhook
            dd_conn = client.post(
                "/api/v1/integrations/datadog/connect",
                headers=headers_a,
                json={"api_key": "dd_api_super_secret_key", "app_key": "dd_app_super_secret_key", "site": "datadoghq.eu"},
            )
            self.assertEqual(200, dd_conn.status_code)
            wh_token = dd_conn.json()["webhook_token"]
            # Plaintext key is NEVER returned in public integration response
            self.assertNotIn("dd_api_super_secret_key", json.dumps(dd_conn.json()))
            self.assertTrue(dd_conn.json()["integration"]["vault_ciphertext_preview"].startswith("vault:v1:"))

            # Trigger Datadog alert webhook for Org A
            dd_wh = client.post(
                f"/api/v1/webhooks/datadog/{org_a}?token={wh_token}",
                json={
                    "incident_id": "dd-org-a-only-777",
                    "alert_title": "High p99 latency on ledger.py",
                    "exception_type": "LatencyAnomaly",
                },
            )
            self.assertEqual(200, dd_wh.status_code)
            self.assertEqual(org_a, dd_wh.json()["org_id"])

            # Org A can access its war-room incident and isolated job queue
            war_a = client.get("/api/v1/incidents/dd-org-a-only-777/war-room", headers=headers_a)
            self.assertEqual(200, war_a.status_code)
            jobs_a = client.get("/api/v1/jobs", headers=headers_a)
            self.assertEqual(1, jobs_a.json()["total"])

            # IDOR Prevention: Org B cannot access Org A's incident or see Org A's jobs/integrations!
            war_b_idor = client.get("/api/v1/incidents/dd-org-a-only-777/war-room", headers=headers_b)
            self.assertEqual(404, war_b_idor.status_code)
            jobs_b = client.get("/api/v1/jobs", headers=headers_b)
            self.assertEqual(0, jobs_b.json()["total"])
            int_b = client.get("/api/v1/integrations", headers=headers_b)
            self.assertEqual(0, int_b.json()["total"])

            # Outbound In-Cluster Runner Registration enforces ExternalId == org_id
            run_reg = client.post(
                "/api/v1/runners/register",
                headers=headers_a,
                json={"cluster_name": "prod-eu-eks", "namespace": "asre-agent", "external_id": org_a},
            )
            self.assertEqual(200, run_reg.status_code)
            self.assertEqual("online_ready", run_reg.json()["status"])

            # H. Full Auth Lifecycle (Register, Login, Refresh, Logout), Google/GitHub SSO & User/Org CRUD
            reg_resp = client.post(
                "/api/v1/auth/register",
                json={
                    "email": "cto@fintech-corp.io",
                    "password": "StrongPassword!2026",
                    "full_name": "Ada Lovelace",
                    "org_name": "Fintech Corp",
                    "role": "admin",
                },
            )
            self.assertEqual(201, reg_resp.status_code)
            reg_data = reg_resp.json()
            self.assertIn("access_token", reg_data)
            self.assertIn("refresh_token", reg_data)
            self.assertNotIn("password_hash", json.dumps(reg_data))
            fintech_headers = {"Authorization": f"Bearer {reg_data['access_token']}"}

            # Login with registered credentials
            login_resp = client.post(
                "/api/v1/auth/login",
                json={"email": "cto@fintech-corp.io", "password": "StrongPassword!2026"},
            )
            self.assertEqual(200, login_resp.status_code)

            # Refresh token exchange
            ref_resp = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": login_resp.json()["refresh_token"]},
            )
            self.assertEqual(200, ref_resp.status_code)

            # Google & GitHub SSO authorize & callback
            sso_provs = client.get("/api/v1/auth/sso/providers")
            self.assertEqual(200, sso_provs.status_code)
            self.assertEqual(2, len(sso_provs.json()["providers"]))

            for prov in ("google", "github"):
                sso_auth = client.get(f"/api/v1/auth/sso/{prov}/authorize")
                self.assertEqual(200, sso_auth.status_code)
                sso_cb = client.post(
                    f"/api/v1/auth/sso/{prov}/callback",
                    json={
                        "code": f"{prov}-oauth-code-123",
                        "state": sso_auth.json()["state_token"],
                        "email": f"sre.{prov}@fintech-corp.io",
                        "full_name": f"{prov.capitalize()} SRE",
                    },
                )
                self.assertEqual(200, sso_cb.status_code)
                self.assertTrue(sso_cb.json()["sso_authenticated"])

            # User Profile Update & Team Member CRUD (/api/v1/users)
            upd_me = client.put(
                "/api/v1/auth/me",
                headers=fintech_headers,
                json={"full_name": "Ada Lovelace (CTO)"},
            )
            self.assertEqual(200, upd_me.status_code)
            self.assertEqual("Ada Lovelace (CTO)", upd_me.json()["user"]["full_name"])

            invite_resp = client.post(
                "/api/v1/users",
                headers=fintech_headers,
                json={"email": "oncall@fintech-corp.io", "full_name": "OnCall Engineer", "role": "sre_approver"},
            )
            self.assertEqual(201, invite_resp.status_code)
            invited_uid = invite_resp.json()["user"]["user_id"]

            list_users = client.get("/api/v1/users", headers=fintech_headers)
            self.assertEqual(200, list_users.status_code)
            self.assertGreaterEqual(list_users.json()["total"], 2)

            patch_user = client.patch(
                f"/api/v1/users/{invited_uid}",
                headers=fintech_headers,
                json={"role": "admin"},
            )
            self.assertEqual(200, patch_user.status_code)
            self.assertEqual("admin", patch_user.json()["user"]["role"])

            del_user = client.delete(f"/api/v1/users/{invited_uid}", headers=fintech_headers)
            self.assertEqual(200, del_user.status_code)
            self.assertTrue(del_user.json()["deleted"])

            # Organization CRUD (/api/v1/organizations/current)
            org_upd = client.put(
                "/api/v1/organizations/current",
                headers=fintech_headers,
                json={"org_name": "Fintech Corp Global", "sso_domain": "fintech-corp.io"},
            )
            self.assertEqual(200, org_upd.status_code)
            self.assertEqual("Fintech Corp Global", org_upd.json()["organization"]["org_name"])

            # Frontend Incident Creation & Challenge Counter-Hypothesis (no CLI needed)
            create_inc = client.post(
                "/api/v1/incidents",
                headers=fintech_headers,
                json={
                    "incident_id": "ui-created-inc-1",
                    "exception_type": "ValueError",
                    "message": "charge_card rejected negative value",
                    "stack_trace": '  File "src/payments/ledger.py", line 3, in charge_card\nValueError',
                },
            )
            self.assertEqual(201, create_inc.status_code)

            challenge_plan = client.post(
                "/api/v1/incidents/ui-created-inc-1/plan",
                headers=fintech_headers,
                json={"challenge": "Check if zero amount should be permitted for free trials"},
            )
            self.assertEqual(200, challenge_plan.status_code)
            self.assertTrue(len(challenge_plan.json()["patch_plan"]["diagnosis"]) > 10)

            del_inc = client.delete("/api/v1/incidents/ui-created-inc-1", headers=fintech_headers)
            self.assertEqual(200, del_inc.status_code)

            del_int = client.delete("/api/v1/integrations/datadog", headers=headers_a)
            self.assertEqual(200, del_int.status_code)

            logout_resp = client.post("/api/v1/auth/logout", headers=fintech_headers)
            self.assertEqual(200, logout_resp.status_code)


if __name__ == "__main__":
    unittest.main()

