import os
import subprocess
import tempfile
import unittest

from repair.incident_mapper import Incident, RepairContext, RepairEvidence
from repair.patch_executor import (
    apply_verified_patch,
    create_repair_pull_request,
    execute_patch_plan,
    explain_untouched_evidence,
)
from repair.patch_planner import PatchPlan


class TestPatchExecutor(unittest.TestCase):
    def _create_repository(self):
        directory = tempfile.TemporaryDirectory()
        repo = directory.name
        os.makedirs(os.path.join(repo, "app"))
        os.makedirs(os.path.join(repo, "tests"))
        with open(os.path.join(repo, "app", "service.py"), "w", encoding="utf-8") as handle:
            handle.write("VALUE = 1\n")
        with open(os.path.join(repo, "tests", "test_service.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "import unittest\nfrom pathlib import Path\n\n"
                "class ServiceTest(unittest.TestCase):\n"
                "    def test_value(self):\n"
                "        self.assertIn('VALUE = 2', Path('app/service.py').read_text())\n"
            )
        for command in (
            ["git", "init"],
            ["git", "config", "user.email", "asre@example.invalid"],
            ["git", "config", "user.name", "ASRE Test"],
            ["git", "add", "."],
            ["git", "commit", "-m", "initial"],
        ):
            subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)
        return directory

    def _context(self, repository):
        return RepairContext(
            incident=Incident("inc-1", "service failure"),
            repository_path=repository,
            evidence=[
                RepairEvidence(
                    path="app/service.py",
                    score=100,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                )
            ],
        )

    def test_verified_patch_runs_in_removed_worktree_without_mutating_source(self):
        directory = self._create_repository()
        self.addCleanup(directory.cleanup)
        plan = PatchPlan(
            "Update service value",
            ("app/service.py",),
            "--- a/app/service.py\n+++ b/app/service.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
            ("python -m unittest discover -s tests",),
            0.9,
        )

        result = execute_patch_plan(plan, self._context(directory.name))

        self.assertEqual("verified", result.status, result.to_dict())
        self.assertTrue(result.worktree_removed)
        with open(os.path.join(directory.name, "app", "service.py"), encoding="utf-8") as handle:
            self.assertEqual("VALUE = 1\n", handle.read())

    def test_invalid_plan_is_rejected_before_git_commands(self):
        directory = self._create_repository()
        self.addCleanup(directory.cleanup)
        plan = PatchPlan(
            "Unsafe target",
            ("outside.py",),
            "--- a/outside.py\n+++ b/outside.py\n@@ -1 +1 @@\n-old\n+new\n",
            ("python -m unittest",),
            0.5,
        )

        result = execute_patch_plan(plan, self._context(directory.name))

        self.assertEqual("rejected", result.status)
        self.assertEqual((), result.commands)

    def test_apply_verified_patch_mutates_live_repo_only_when_verified(self):
        directory = self._create_repository()
        self.addCleanup(directory.cleanup)
        context = self._context(directory.name)
        plan = PatchPlan(
            "Update service value",
            ("app/service.py",),
            "--- a/app/service.py\n+++ b/app/service.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
            ("python -m unittest discover -s tests",),
            0.9,
        )

        # Cannot apply before worktree verification
        unverified_apply = apply_verified_patch(plan, context, None)
        self.assertEqual("rejected", unverified_apply.status)

        execution = execute_patch_plan(plan, context)
        self.assertEqual("verified", execution.status)

        # Now apply after verification
        applied = apply_verified_patch(plan, context, execution)
        self.assertEqual("applied", applied.status)
        self.assertEqual(("app/service.py",), applied.applied_files)
        with open(os.path.join(directory.name, "app", "service.py"), encoding="utf-8") as handle:
            self.assertEqual("VALUE = 2\n", handle.read())

    def test_create_repair_pull_request_and_explain_untouched_evidence(self):
        directory = self._create_repository()
        self.addCleanup(directory.cleanup)
        context = RepairContext(
            incident=Incident("inc-pr-42", "service failure"),
            repository_path=directory.name,
            evidence=[
                RepairEvidence(
                    path="app/service.py",
                    score=135.0,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                ),
                RepairEvidence(
                    path="tests/test_service.py",
                    score=100.0,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                ),
                RepairEvidence(
                    path="app/caller.py",
                    score=30.0,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_direct_graph_neighbor",
                    evidence_tier="corroborated",
                ),
            ],
        )
        plan = PatchPlan(
            "Update service value to 2",
            ("app/service.py",),
            "--- a/app/service.py\n+++ b/app/service.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
            ("python -m unittest discover -s tests",),
            0.92,
            deliberately_unchanged=("Preserved VALUE constant name for backward compatibility",),
        )

        untouched = explain_untouched_evidence(plan, context)
        self.assertTrue(any("Preserved VALUE constant name" in r for r in untouched))
        self.assertTrue(any("tests/test_service.py — Deliberately left test file unchanged" in r for r in untouched))
        self.assertTrue(any("app/caller.py — Deliberately left dependency-graph neighbor unchanged" in r for r in untouched))

        execution = execute_patch_plan(plan, context)
        self.assertEqual("verified", execution.status)

        pr_result = create_repair_pull_request(plan, context, execution)
        self.assertIn(pr_result.status, {"pr_branch_ready", "pr_created"})
        self.assertEqual("asre/fix-inc-pr-42", pr_result.branch_name)
        self.assertTrue(pr_result.commit_sha)
        self.assertIn("What ASRE Deliberately Chose Not to Change", pr_result.pr_body)

    def test_patch_is_not_verified_when_zero_tests_actually_run(self):
        directory = self._create_repository()
        self.addCleanup(directory.cleanup)
        # Overwrite test file so it contains 0 test methods (unittest exits 0 with "Ran 0 tests")
        with open(os.path.join(directory.name, "tests", "test_service.py"), "w", encoding="utf-8") as handle:
            handle.write("import unittest\n")
        subprocess.run(["git", "add", "."], cwd=directory.name, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", "empty tests"], cwd=directory.name, check=True, capture_output=True, text=True)

        plan = PatchPlan(
            "Update service value",
            ("app/service.py",),
            "--- a/app/service.py\n+++ b/app/service.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
            ("python -m unittest discover -s tests",),
            0.9,
        )
        result = execute_patch_plan(plan, self._context(directory.name))
        self.assertEqual("failed", result.status)
        self.assertIn("ran 0 tests", result.error)

    def test_blast_radius_analysis_and_original_incident_replay_with_adjacent_tests(self):
        directory = self._create_repository()
        self.addCleanup(directory.cleanup)
        # Add an adjacent downstream consumer module and its regression test
        with open(os.path.join(directory.name, "app", "caller.py"), "w", encoding="utf-8") as handle:
            handle.write("from app.service import VALUE\n\ndef compute_double():\n    return VALUE * 2\n")
        with open(os.path.join(directory.name, "tests", "test_caller.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "import unittest\nfrom app.caller import compute_double\n\n"
                "class TestCaller(unittest.TestCase):\n"
                "    def test_compute_double_stays_positive(self):\n"
                "        self.assertGreaterEqual(compute_double(), 2)\n"
            )
        subprocess.run(["git", "add", "."], cwd=directory.name, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", "add caller and adjacent test"], cwd=directory.name, check=True, capture_output=True, text=True)

        context = RepairContext(
            incident=Incident(incident_id="inc-blast-1", message="VALUE was 1 instead of 2"),
            repository_path=directory.name,
            evidence=[
                RepairEvidence(
                    path="app/service.py",
                    score=14.0,
                    reasons=["stack trace match"],
                    frame_lines=[1],
                    symbols=["VALUE"],
                    excerpt="VALUE = 1\n",
                    patch_eligible=True,
                    evidence_tier="confirmed",
                    risk_score=35,
                    risk_label="Medium",
                    direct_dependents=1,
                    transitive_dependents=2,
                    affected_components=["app/caller.py"],
                ),
                RepairEvidence(
                    path="app/caller.py",
                    score=6.0,
                    reasons=["direct dependent of app/service.py"],
                    frame_lines=[],
                    symbols=["compute_double"],
                    excerpt="def compute_double():\n    return VALUE * 2\n",
                    patch_eligible=True,
                    evidence_tier="corroborated",
                ),
                RepairEvidence(
                    path="tests/test_caller.py",
                    score=4.0,
                    reasons=["adjacent regression test for app/caller.py"],
                    frame_lines=[],
                    symbols=["TestCaller"],
                    excerpt="",
                    patch_eligible=False,
                    evidence_tier="corroborated",
                ),
            ],
        )
        # Plan only lists the primary incident test; executor automatically runs adjacent tests/test_caller.py too
        plan = PatchPlan(
            "Fix VALUE to 2",
            ("app/service.py",),
            "--- a/app/service.py\n+++ b/app/service.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
            ("python -m unittest tests/test_service.py",),
            0.95,
        )
        result = execute_patch_plan(plan, context)
        self.assertEqual("verified", result.status)
        self.assertTrue(result.reproduced_before_patch)
        self.assertTrue(result.incident_replay_passed)
        self.assertEqual(["app/caller.py"], result.impact_summary["downstream_consumers"])
        self.assertEqual(["tests/test_caller.py"], result.impact_summary["adjacent_test_files"])
        self.assertEqual(0, result.impact_summary["adjacent_regressions"])


if __name__ == "__main__":
    unittest.main()
