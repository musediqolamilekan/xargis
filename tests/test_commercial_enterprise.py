import json
import tempfile
import unittest
from pathlib import Path

from intelligence.commercial_enterprise import (
    build_commercial_enterprise_report,
    dispatch_repair_api_request,
    evaluate_custom_repair_policy,
    load_custom_repair_policy,
)
from intelligence.dependency_graph import build_dependency_graph
from intelligence.file_profile import build_file_profiles
from intelligence.scanner import scan_repository


class TestCommercialEnterprise(unittest.TestCase):
    def test_all_10_commercial_capabilities_and_repair_api_dispatcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = Path(tmp) / "billing_service"
            repo_b = Path(tmp) / "checkout_service"
            repo_a.mkdir()
            repo_b.mkdir()

            (repo_a / "requirements.txt").write_text("click>=8.1\npydantic>=2.0\n", encoding="utf-8")
            (repo_b / "requirements.txt").write_text("click>=8.1\nhttpx>=0.27\n", encoding="utf-8")

            (repo_a / ".github").mkdir()
            (repo_a / ".github" / "CODEOWNERS").write_text(
                "src/billing/* @fintech-billing-team @alice-sre\n"
                "src/auth/* @security-team\n",
                encoding="utf-8",
            )
            (repo_a / "src" / "billing").mkdir(parents=True)
            (repo_a / "src" / "auth").mkdir(parents=True)
            (repo_a / "src" / "billing" / "invoice.py").write_text(
                "def calculate_total(items):\n    return sum(items)\n",
                encoding="utf-8",
            )
            (repo_a / "src" / "auth" / "token_guard.py").write_text(
                "def verify_token(tok):\n    return bool(tok)\n",
                encoding="utf-8",
            )
            (repo_b / "app.py").write_text(
                "def checkout():\n    return 'ok'\n",
                encoding="utf-8",
            )

            # Write a custom declarative repair policy (.asre/policy.json)
            policy_dir = repo_a / ".asre"
            policy_dir.mkdir()
            (policy_dir / "policy.json").write_text(
                json.dumps(
                    {
                        "protected_path_globs": ["**/auth/**"],
                        "max_files_per_patch": 2,
                        "max_changed_lines": 50,
                        "environment_max_auto_risk": {"development": 40, "staging": 25, "production": 10},
                        "required_approver_teams_by_glob": {"**/auth/**": "@security-team"},
                    }
                ),
                encoding="utf-8",
            )

            intel_a = scan_repository(str(repo_a), verbose=False)
            graph_a = build_dependency_graph(intel_a)
            profiles_a = build_file_profiles(graph_a, intel_a, {}, {})

            report = build_commercial_enterprise_report(
                str(repo_a),
                intel_a,
                graph_a,
                profiles_a,
                additional_repos=[str(repo_b)],
                repair_run_dir=str(policy_dir / "runs"),
                private_mode=True,
            )

            # 1. Multi-repository intelligence detects both repos and shared 'click' dependency
            self.assertEqual(2, report.multi_repo_federation.repositories_analyzed)
            self.assertIn("click", report.multi_repo_federation.shared_dependencies)
            self.assertGreaterEqual(len(report.multi_repo_federation.cross_repo_edges), 1)

            # 2. Service ownership mapping parses CODEOWNERS
            self.assertTrue(report.service_ownership.has_codeowners_file)
            owners_by_sub = {s.subsystem: s.team_handle for s in report.service_ownership.subsystems}
            self.assertEqual("@fintech-billing-team", owners_by_sub.get("src/billing"))

            # 3 & 4 & 5. Incident cost ROI, repair effectiveness & reliability scorecard
            self.assertGreater(report.incident_cost_estimation.net_savings_usd, 0.0)
            self.assertGreater(report.incident_cost_estimation.mttr_speedup_factor, 10.0)
            self.assertGreaterEqual(report.reliability_scorecard.overall_score, 50.0)
            self.assertIn(report.reliability_scorecard.letter_grade, {"A+", "A", "B+", "B", "C"})

            # 6. Custom repair policy blocks modification of protected auth path and requires @security-team
            policy = load_custom_repair_policy(str(repo_a))
            verdict_blocked = evaluate_custom_repair_policy(
                policy,
                ["src/auth/token_guard.py"],
                environment="development",
                max_file_risk_score=5,
            )
            self.assertFalse(verdict_blocked.allowed_for_autonomous_modification)
            self.assertIn("@security-team", verdict_blocked.requires_team_approvals)

            verdict_allowed = evaluate_custom_repair_policy(
                policy,
                ["src/billing/invoice.py"],
                environment="development",
                max_file_risk_score=10,
            )
            self.assertTrue(verdict_allowed.allowed_for_autonomous_modification)

            # 8 & 9. Private deployment mode and multi-environment matrix
            self.assertTrue(report.private_deployment.private_mode_active)
            self.assertTrue(report.private_deployment.zero_external_egress_enforced)
            self.assertEqual(3, len(report.multi_environment_matrix))

            # 10. Repair REST API dispatcher endpoints
            code_health, body_health = dispatch_repair_api_request(
                "GET",
                "/api/v1/health",
                None,
                repo_path=str(repo_a),
                intel=intel_a,
                graph=graph_a,
                profiles=profiles_a,
                private_mode=True,
            )
            self.assertEqual(200, code_health)
            self.assertEqual("ok", body_health["status"])

            code_diag, body_diag = dispatch_repair_api_request(
                "POST",
                "/api/v1/incidents/diagnose",
                {
                    "incident_id": "inc-api-99",
                    "exception_type": "TypeError",
                    "message": "calculate_total failed in invoice.py",
                    "stack_trace": '  File "src/billing/invoice.py", line 2, in calculate_total\n    return sum(items)\nTypeError: unsupported operand',
                },
                repo_path=str(repo_a),
                intel=intel_a,
                graph=graph_a,
                profiles=profiles_a,
            )
            self.assertEqual(200, code_diag)
            self.assertEqual("@fintech-billing-team", body_diag["routed_owner_team"])


if __name__ == "__main__":
    unittest.main()

