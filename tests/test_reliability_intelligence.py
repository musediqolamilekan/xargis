import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from intelligence.dependency_graph import build_dependency_graph
from intelligence.file_profile import build_file_profiles
from intelligence.reliability_intelligence import (
    analyze_pre_deployment_risk,
    build_reliability_intelligence_report,
)
from intelligence.scanner import scan_repository
from repair.incident_mapper import Incident, RepairContext, RepairEvidence
from repair.patch_planner import PatchPlan
from repair.repair_runs import save_repair_run


class TestReliabilityIntelligence(unittest.TestCase):
    def _create_repo_with_history(self) -> tempfile.TemporaryDirectory:
        directory = tempfile.TemporaryDirectory()
        repo = directory.name
        os.makedirs(os.path.join(repo, "core"))
        os.makedirs(os.path.join(repo, "api"))
        os.makedirs(os.path.join(repo, "auth"))

        # core/engine.py: fragile hub imported by api/routes.py and auth/guard.py, with FIXME/BUG debt and external service call
        with open(os.path.join(repo, "core", "engine.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "import requests\n\n"
                "# FIXME: handle upstream timeout cleanly\n"
                "# BUG: retries can duplicate items\n"
                "def process_batch(items):\n"
                "    resp = requests.post('https://api.internal/batch')\n"
                "    return [x for x in items]\n"
            )
        with open(os.path.join(repo, "api", "routes.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "from core.engine import process_batch\n\n"
                "def handle_request(data):\n"
                "    return process_batch(data)\n"
            )
        with open(os.path.join(repo, "auth", "guard.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "from core.engine import process_batch\n\n"
                "def verify_token(token):\n"
                "    return bool(token)\n"
            )

        for cmd in (
            ["git", "init"],
            ["git", "config", "user.email", "asre@example.invalid"],
            ["git", "config", "user.name", "ASRE Test"],
            ["git", "add", "."],
            ["git", "commit", "-m", "initial commit"],
        ):
            subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True)

        # Second commit modifying core/engine.py and auth/guard.py for pre-deploy risk check
        with open(os.path.join(repo, "core", "engine.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "import requests\n\n"
                "# FIXME: handle upstream timeout cleanly\n"
                "# BUG: retries can duplicate items\n"
                "def process_batch(items):\n"
                "    resp = requests.post('https://api.internal/batch', timeout=5)\n"
                "    return [x * 2 for x in items]\n"
            )
        with open(os.path.join(repo, "auth", "guard.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "from core.engine import process_batch\n\n"
                "def verify_token(token):\n"
                "    return token == 'valid'\n"
            )
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", "update engine and auth guard"], cwd=repo, check=True, capture_output=True, text=True)
        return directory

    def test_reliability_report_identifies_recurrences_failing_components_fragile_deps_debt_and_pre_deploy_risk(self):
        directory = self._create_repo_with_history()
        self.addCleanup(directory.cleanup)
        repo = directory.name

        intel = scan_repository(repo, verbose=False)
        graph = build_dependency_graph(intel)
        profiles = build_file_profiles(
            graph,
            intel,
            {"core/engine.py": 5, "auth/guard.py": 2},
            {"core/engine.py": "2026-11-21"},
        )

        with tempfile.TemporaryDirectory() as runs_dir:
            # Run 1: Verified repair for TypeError on core/engine.py
            ctx1 = RepairContext(
                incident=Incident("inc-101", "batch type error", exception_type="TypeError"),
                repository_path=repo,
                evidence=[
                    RepairEvidence(
                        path="core/engine.py",
                        score=130.0,
                        patch_eligible=True,
                        patch_eligibility_reason="eligible_valid_stack_frame",
                    )
                ],
            )
            plan1 = PatchPlan(
                "Fix batch iterator type error",
                ("core/engine.py",),
                "--- a/core/engine.py\n+++ b/core/engine.py\n@@ -1 +1 @@\n-old\n+new\n",
                ("python -m unittest",),
                0.9,
            )
            save_repair_run(runs_dir, ctx1, plan=plan1)
            # Mark run 1 as "verified" in run.json to simulate a completed repair
            run1_json = Path(runs_dir) / "inc-101" / "run.json"
            data1 = run1_json.read_text(encoding="utf-8").replace('"needs-review"', '"verified"')
            run1_json.write_text(data1, encoding="utf-8")

            # Run 2: Post-repair recurrence of TypeError on core/engine.py!
            ctx2 = RepairContext(
                incident=Incident("inc-102", "batch type error recurred on empty list", exception_type="TypeError"),
                repository_path=repo,
                evidence=[
                    RepairEvidence(
                        path="core/engine.py",
                        score=130.0,
                        patch_eligible=True,
                        patch_eligibility_reason="eligible_valid_stack_frame",
                    )
                ],
            )
            save_repair_run(runs_dir, ctx2, plan=plan1)

            report = build_reliability_intelligence_report(
                repo,
                graph,
                profiles,
                repair_run_dir=runs_dir,
            )

            self.assertEqual(2, report.total_historical_runs)
            # 1. Frequently failing components
            self.assertTrue(report.failing_components)
            self.assertEqual("core/engine.py", report.failing_components[0].path)
            self.assertEqual(2, report.failing_components[0].incident_count)

            # 2. Recurring exceptions
            self.assertTrue(report.recurring_exceptions)
            self.assertEqual("TypeError", report.recurring_exceptions[0].exception_type)
            self.assertEqual(2, report.recurring_exceptions[0].occurrences)

            # 3. Post-repair recurrence tracking
            self.assertEqual(1, len(report.post_repair_recurrences))
            rec = report.post_repair_recurrences[0]
            self.assertEqual("core/engine.py", rec.path)
            self.assertEqual("inc-101", rec.prior_repair_incident_id)
            self.assertEqual("inc-102", rec.recurring_incident_id)

            # 4. Fragile dependencies
            self.assertTrue(report.fragile_dependencies)
            self.assertEqual("core/engine.py", report.fragile_dependencies[0].path)

            # 5. Technical debt hotspots (FIXME + BUG markers)
            self.assertTrue(report.technical_debt_hotspots)
            self.assertEqual("core/engine.py", report.technical_debt_hotspots[0].path)
            self.assertEqual(2, report.technical_debt_hotspots[0].bug_fixme_count)

            # 6. Preventive recommendations
            self.assertTrue(report.recommendations)
            self.assertEqual("P0", report.recommendations[0].priority)
            self.assertEqual("post_repair_recurrence", report.recommendations[0].category)

            # 7. Pre-deployment risky change detection (HEAD~1..HEAD)
            pre_deploy = analyze_pre_deployment_risk(
                repo,
                graph,
                profiles,
                rev_range="HEAD~1..HEAD",
                repair_run_dir=runs_dir,
            )
            self.assertIn(pre_deploy.deployment_risk_level, {"HIGH", "CRITICAL"})
            changed_paths = {f.path for f in pre_deploy.changed_files}
            self.assertIn("core/engine.py", changed_paths)
            self.assertIn("auth/guard.py", changed_paths)
            engine_risk = next(f for f in pre_deploy.changed_files if f.path == "core/engine.py")
            self.assertIn("process_batch", engine_risk.touched_symbols)
            self.assertEqual(2, engine_risk.past_incidents)
            self.assertGreater(engine_risk.reverse_pagerank, 0.0)
            auth_risk = next(f for f in pre_deploy.changed_files if f.path == "auth/guard.py")
            self.assertIn("authentication_or_security", auth_risk.sensitive_domains)

            # Verify 3-Signal Architecture: Tabular JIT, Graph Centrality, Semantic AST+Memory, SHAP & Governance
            self.assertIsNotNone(pre_deploy.tabular_metrics)
            self.assertGreater(pre_deploy.tabular_metrics.lines_added, 0)
            self.assertGreaterEqual(pre_deploy.tabular_metrics.change_entropy, 0.0)
            self.assertGreaterEqual(pre_deploy.tabular_metrics.historical_pr_auc, 0.5)

            self.assertIsNotNone(pre_deploy.graph_metrics)
            self.assertGreater(pre_deploy.graph_metrics.max_reverse_pagerank, 0.0)
            self.assertIn("core/engine.py", pre_deploy.graph_metrics.degraded_dependencies)

            self.assertIsNotNone(pre_deploy.semantic_inspection)
            self.assertTrue(any("inc-101" in link for link in pre_deploy.semantic_inspection.historical_incident_links))

            # SHAP additive attributions sum + base_probability == calibrated_probability
            self.assertTrue(pre_deploy.shap_attributions)
            self.assertIsNotNone(pre_deploy.governance_decision)
            gov = pre_deploy.governance_decision
            self.assertGreater(gov.calibrated_probability, 0.70)
            self.assertEqual("HIGH", gov.risk_tier)
            self.assertEqual("BLOCK_AUTO_DEPLOY_REQUIRE_SENIOR_SRE_SIGN_OFF", gov.autonomous_action)
            self.assertIn("SHAP Feature Attributions", gov.pr_comment_markdown)
            shap_sum = gov.base_probability + sum(a.shap_delta_pct for a in pre_deploy.shap_attributions) / 100.0
            self.assertAlmostEqual(gov.calibrated_probability, shap_sum, places=4)


if __name__ == "__main__":
    unittest.main()

