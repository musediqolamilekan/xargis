import unittest

from repair.incident_mapper import Incident, RepairContext, RepairEvidence
from repair.patch_planner import PatchPlan
from repair.repair_simulator import (
    run_progressive_repair_simulation,
    run_tier1_policy_invariants,
    run_tier2_graph_capacity_simulation,
    run_tier3_dry_run_inspection,
)


class TestRepairSimulator(unittest.TestCase):
    def _stateless_low_risk_context(self) -> RepairContext:
        return RepairContext(
            incident=Incident("inc-sim-low", "off-by-one slice in helper", exception_type="IndexError"),
            repository_path="/tmp/nonexistent-sim-repo",
            evidence=[
                RepairEvidence(
                    path="utils/formatter.py",
                    score=110.0,
                    valid_frame_lines=[10],
                    frame_lines=[10],
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                    evidence_tier="confirmed",
                    risk_score=8,
                    risk_label="Low",
                    direct_dependents=1,
                    transitive_dependents=1,
                )
            ],
        )

    def _stateful_high_risk_context(self) -> RepairContext:
        return RepairContext(
            incident=Incident("inc-sim-high", "auth migration deadlock", exception_type="OperationalError"),
            repository_path="/tmp/nonexistent-sim-repo",
            evidence=[
                RepairEvidence(
                    path="auth/migrations/001_schema.py",
                    score=140.0,
                    valid_frame_lines=[25],
                    frame_lines=[25],
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                    evidence_tier="confirmed",
                    risk_score=85,
                    risk_label="Critical",
                    direct_dependents=12,
                    transitive_dependents=64,
                    db_interactions=["conn.execute"],
                )
            ],
        )

    def test_low_risk_stateless_patch_routes_to_fully_autonomous_execution(self):
        ctx = self._stateless_low_risk_context()
        plan = PatchPlan(
            "Fix slice bound in formatter",
            ("utils/formatter.py",),
            "--- a/utils/formatter.py\n+++ b/utils/formatter.py\n@@ -10 +10 @@\n-    return items[:n-1]\n+    return items[:n]\n",
            ("python -m unittest tests/test_formatter.py",),
            0.95,
        )
        report = run_progressive_repair_simulation(
            plan,
            ctx,
            forward_apply_clean=True,
            pre_patch_reproduced=True,
            post_patch_passed=True,
        )
        self.assertTrue(report.all_tiers_passed)
        self.assertTrue(report.tier1_policy_invariants.passed)
        self.assertTrue(report.tier2_graph_capacity.passed)
        self.assertLess(
            report.tier2_graph_capacity.utilization_rho_after,
            report.tier2_graph_capacity.utilization_rho_before,
        )
        self.assertLess(report.tier2_graph_capacity.counterfactual_latency_delta_pct, 0.0)
        self.assertTrue(report.tier3_dry_run_state_diff.reverse_rollback_clean)
        self.assertLess(report.action_risk.r_action, 0.25)
        self.assertEqual("FULLY_AUTONOMOUS_EXECUTION", report.decision_matrix.execution_band)
        self.assertTrue(report.decision_matrix.can_execute_autonomously)

    def test_medium_risk_patch_routes_to_progressive_canary_execution(self):
        ctx = RepairContext(
            incident=Incident("inc-sim-med", "service timeout in order handler", exception_type="TimeoutError"),
            repository_path="/tmp/nonexistent-sim-repo",
            evidence=[
                RepairEvidence(
                    path="orders/handler.py",
                    score=120.0,
                    valid_frame_lines=[14],
                    frame_lines=[14],
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                    evidence_tier="confirmed",
                    risk_score=48,
                    risk_label="High",
                    direct_dependents=5,
                    transitive_dependents=22,
                    db_interactions=["cursor.execute"],
                )
            ],
        )
        plan = PatchPlan(
            "Update query timeout in order handler",
            ("orders/handler.py",),
            "--- a/orders/handler.py\n+++ b/orders/handler.py\n@@ -14 +14 @@\n-    cursor.execute(q)\n+    cursor.execute(q, timeout=5)\n",
            ("python -m unittest tests/test_orders.py",),
            0.85,
        )
        report = run_progressive_repair_simulation(
            plan,
            ctx,
            forward_apply_clean=True,
            pre_patch_reproduced=True,
            post_patch_passed=True,
        )
        self.assertGreaterEqual(report.action_risk.r_action, 0.25)
        self.assertLessEqual(report.action_risk.r_action, 0.65)
        self.assertEqual("PROGRESSIVE_CANARY_EXECUTION", report.decision_matrix.execution_band)
        self.assertEqual(5, report.decision_matrix.canary_traffic_pct)

    def test_high_risk_ddl_or_policy_violation_routes_to_human_in_the_loop_approval(self):
        ctx = self._stateful_high_risk_context()
        plan = PatchPlan(
            "Alter auth schema table",
            ("auth/migrations/001_schema.py",),
            "--- a/auth/migrations/001_schema.py\n+++ b/auth/migrations/001_schema.py\n@@ -25 +25 @@\n-    pass\n+    conn.execute('ALTER TABLE users ADD COLUMN role TEXT')\n",
            ("python -m unittest tests/test_auth.py",),
            0.70,
        )
        report = run_progressive_repair_simulation(
            plan,
            ctx,
            forward_apply_clean=True,
            pre_patch_reproduced=False,
            post_patch_passed=False,
            active_connections_pct=85.0,
            is_peak_traffic_window=True,
        )
        self.assertFalse(report.tier1_policy_invariants.passed)
        self.assertFalse(report.tier3_dry_run_state_diff.reverse_rollback_clean)
        self.assertGreater(report.action_risk.r_action, 0.65)
        self.assertEqual("HUMAN_IN_THE_LOOP_APPROVAL", report.decision_matrix.execution_band)
        self.assertFalse(report.decision_matrix.can_execute_autonomously)
        self.assertIn("Interactive Repair Approval Card", report.decision_matrix.interactive_approval_card_markdown)


if __name__ == "__main__":
    unittest.main()

