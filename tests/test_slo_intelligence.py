import subprocess
import tempfile
import unittest
from pathlib import Path

from intelligence.dependency_graph import build_dependency_graph
from intelligence.file_profile import build_file_profiles
from intelligence.scanner import scan_repository
from intelligence.slo_intelligence import (
    build_slo_intelligence_report,
    evaluate_error_budget_deployment_gate,
)
from repair.incident_intelligence import TelemetryEvent


class TestSLOIntelligence(unittest.TestCase):
    def _create_repo(self) -> tempfile.TemporaryDirectory:
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        (root / "payments").mkdir()
        (root / "orders").mkdir()

        (root / "payments" / "processor.py").write_text(
            "def charge_card(amount):\n"
            "    return {'charged': amount}\n",
            encoding="utf-8",
        )
        (root / "orders" / "checkout.py").write_text(
            "from payments.processor import charge_card\n\n"
            "def complete_checkout(total):\n"
            "    return charge_card(total)\n",
            encoding="utf-8",
        )
        for cmd in (
            ["git", "init"],
            ["git", "config", "user.email", "asre@example.invalid"],
            ["git", "config", "user.name", "ASRE Test"],
            ["git", "add", "."],
            ["git", "commit", "-m", "initial commit on payments and orders"],
        ):
            subprocess.run(cmd, cwd=str(root), check=True, capture_output=True, text=True)
        return temp_dir

    def test_mwmbr_fast_burn_code_attribution_tte_forecast_and_feature_freeze_gate(self):
        temp_dir = self._create_repo()
        self.addCleanup(temp_dir.cleanup)
        repo = temp_dir.name

        intel = scan_repository(repo, verbose=False)
        graph = build_dependency_graph(intel)
        profiles = build_file_profiles(graph, intel, {}, {})

        # 99.9% SLO on 100,000 requests over 30 days => total error budget = 100 failed requests (43.2 minutes)
        # Hourly error budget = 100 / 720 = 0.1389 requests/hr => 14.4x fast burn = 2.0 failures in 1h
        events = [
            TelemetryEvent(
                event_id="ev-1",
                source="sentry",
                exception_type="PaymentGatewayTimeout",
                message="charge_card timed out after 5000ms",
                stack_trace='  File "payments/processor.py", line 2, in charge_card\n    return {"charged": amount}',
                service="payment-service",
                severity="critical",
                count=75,
            ),
            TelemetryEvent(
                event_id="ev-2",
                source="datadog",
                exception_type="CheckoutError",
                message="complete_checkout failed",
                stack_trace='  File "orders/checkout.py", line 4, in complete_checkout\n    return charge_card(total)',
                service="order-service",
                severity="error",
                count=15,
            ),
        ]

        report = build_slo_intelligence_report(
            repo,
            graph,
            profiles,
            slo_target_pct=99.9,
            window_days=30,
            total_requests=100_000,
            telemetry_events=events,
            repair_run_dir=str(Path(repo) / ".asre" / "empty_runs"),
            recent_1h_failures=12,  # 12 / 0.1389 = 86.4x burn rate (>= 14.4x Fast Burn!)
            recent_6h_failures=40,
            recent_24h_failures=90,
        )

        # Pillar 1: Fast Burn (14.4x) and High Burn (6.0x) are firing
        fast_alert = report.mwmbr_alerts[0]
        self.assertTrue(fast_alert.is_firing)
        self.assertGreaterEqual(fast_alert.long_window_burn_rate, 14.4)

        # Pillar 2: AST Code, Symbol & Commit Error-Budget Attribution
        self.assertEqual(2, len(report.code_attributions))
        top_attr = report.code_attributions[0]
        self.assertEqual("payments/processor.py", top_attr.path)
        self.assertIn("charge_card", top_attr.attributed_symbols)
        self.assertTrue(top_attr.latest_commit_sha)
        self.assertAlmostEqual(83.3, top_attr.budget_burn_share_pct, places=1)
        self.assertGreater(top_attr.budget_minutes_consumed, 30.0)

        # Pillar 3: EWMA TTE Forecast (90% of 100-failure budget consumed => 10% remaining)
        self.assertAlmostEqual(90.0, report.forecast.consumed_budget_pct, places=1)
        self.assertAlmostEqual(10.0, report.forecast.remaining_budget_pct, places=1)
        self.assertIsNotNone(report.forecast.time_to_exhaustion_hours)
        self.assertLess(report.forecast.time_to_exhaustion_hours, 24.0)

        # Pillar 4: Closed-Loop Error Budget Governance engages Feature Freeze
        self.assertEqual("ERROR_BUDGET_EXHAUSTED_FEATURE_FREEZE", report.governance_policy.governance_state)
        self.assertTrue(report.governance_policy.feature_freeze_active)

        # Routine feature deployment is BLOCKED during Feature Freeze
        feature_verdict = evaluate_error_budget_deployment_gate(
            report,
            commit_message="feat: add promotional banner to checkout page",
            is_verified_asre_repair=False,
        )
        self.assertFalse(feature_verdict.allowed_to_deploy)
        self.assertIn("BLOCKED BY ERROR BUDGET FEATURE FREEZE", feature_verdict.reason)

        # Verified ASRE reliability repair that reduces error rate is PERMITTED during Feature Freeze
        repair_verdict = evaluate_error_budget_deployment_gate(
            report,
            commit_message="fix(payments): add timeout guard in charge_card",
            is_verified_asre_repair=True,
            expected_error_rate_delta_pct=-95.0,
        )
        self.assertTrue(repair_verdict.allowed_to_deploy)
        self.assertTrue(repair_verdict.is_reliability_or_bugfix_change)


if __name__ == "__main__":
    unittest.main()

