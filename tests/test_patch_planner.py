import unittest

from repair.incident_mapper import Incident, RepairContext, RepairEvidence
from repair.patch_planner import PatchPlan, validate_patch_plan


class TestPatchPlanner(unittest.TestCase):
    def setUp(self):
        self.context = RepairContext(
            incident=Incident("inc-1", "broken"),
            repository_path="/repo",
            evidence=[
                RepairEvidence(
                    path="app/service.py",
                    score=100,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                )
            ],
        )
        self.diff = "--- a/app/service.py\n+++ b/app/service.py\n@@ -1 +1 @@\n-old\n+new\n"

    def test_valid_plan_is_accepted(self):
        plan = PatchPlan("Fix service failure", ("app/service.py",), self.diff, ("python -m unittest tests.test_service",), 0.9)
        validation = validate_patch_plan(plan, self.context)
        self.assertTrue(validation.valid, validation.errors)

    def test_plan_rejects_ineligible_evidence_file(self):
        context = RepairContext(
            incident=Incident("inc-1", "broken"),
            repository_path="/repo",
            evidence=[
                RepairEvidence(
                    path="app/service.py",
                    score=20,
                    patch_eligible=False,
                    patch_eligibility_reason="ineligible_query_match_only",
                )
            ],
        )
        plan = PatchPlan("Fix query-only match", ("app/service.py",), self.diff, ("python -m unittest tests.test_service",), 0.9)
        validation = validate_patch_plan(plan, context)
        self.assertFalse(validation.valid)
        self.assertTrue(any("not patch-eligible (ineligible_query_match_only)" in error for error in validation.errors))

    def test_plan_rejects_overconfident_explanation_without_confirmed_evidence(self):
        # Corroborated neighbor is patch-eligible, but has no confirmed stack frame evidence
        context = RepairContext(
            incident=Incident("inc-1", "broken"),
            repository_path="/repo",
            evidence=[
                RepairEvidence(
                    path="app/service.py",
                    score=30,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_direct_graph_neighbor",
                    evidence_tier="corroborated",
                )
            ],
        )
        overconfident_plan = PatchPlan(
            "Guessing neighbor caused crash",
            ("app/service.py",),
            self.diff,
            ("python -m unittest tests.test_service",),
            0.85,
        )
        validation = validate_patch_plan(overconfident_plan, context)
        self.assertFalse(validation.valid)
        self.assertTrue(any("confirmed supporting evidence" in error for error in validation.errors))

        # Cautious confidence (<= 0.5) on corroborated-only evidence is accepted
        cautious_plan = PatchPlan(
            "Suspected neighbor hypothesis",
            ("app/service.py",),
            self.diff,
            ("python -m unittest tests.test_service",),
            0.45,
            suspected_cause="Downstream helper may return None",
        )
        cautious_validation = validate_patch_plan(cautious_plan, context)
        self.assertTrue(cautious_validation.valid, cautious_validation.errors)

    def test_plan_rejects_diff_introducing_security_vulnerabilities(self):
        unsafe_diff = (
            "--- a/app/service.py\n"
            "+++ b/app/service.py\n"
            "@@ -1 +1,2 @@\n"
            "-old\n"
            "+result = eval(user_input)\n"
            "+subprocess.run(cmd, shell=True)\n"
        )
        plan = PatchPlan(
            "Unsafe fix using eval",
            ("app/service.py",),
            unsafe_diff,
            ("python -m unittest tests.test_service",),
            0.9,
        )
        validation = validate_patch_plan(plan, self.context)
        self.assertFalse(validation.valid)
        self.assertTrue(any("security check failed in diff: arbitrary code execution" in e for e in validation.errors))
        self.assertTrue(any("security check failed in diff: shell injection risk" in e for e in validation.errors))

    def test_plan_rejects_files_outside_context_and_unsafe_commands(self):
        plan = PatchPlan(
            "Change unrelated file",
            ("app/other.py",),
            "--- a/app/other.py\n+++ b/app/other.py\n@@ -1 +1 @@\n-old\n+new\n",
            ("python -m unittest; rm -rf /",),
            0.8,
        )
        validation = validate_patch_plan(plan, self.context)
        self.assertFalse(validation.valid)
        self.assertTrue(any("outside repair context" in error for error in validation.errors))
        self.assertTrue(any("not allowlisted" in error for error in validation.errors))

    def test_plan_rejects_path_traversal_and_diff_mismatch(self):
        plan = PatchPlan("Escape", ("../outside.py",), self.diff, ("pytest tests",), 1.2)
        validation = validate_patch_plan(plan, self.context)
        self.assertFalse(validation.valid)
        self.assertTrue(any("unsafe planned file path" in error for error in validation.errors))
        self.assertTrue(any("exactly match" in error for error in validation.errors))
        self.assertTrue(any("confidence" in error for error in validation.errors))


if __name__ == "__main__":
    unittest.main()
