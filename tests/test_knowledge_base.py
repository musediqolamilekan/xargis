import json
import tempfile
import unittest
from pathlib import Path

from repair.incident_mapper import Incident, RepairContext, RepairEvidence
from repair.knowledge_base import (
    build_and_query_knowledge_base,
    classify_structural_fix_operator,
    detect_repair_record_drift,
    distill_post_incident_record,
    load_tier2_episodic_cases,
    promote_knowledge_base_record,
    verify_stored_repair_guardrails,
)
from repair.llm_planner import build_repair_prompt
from repair.patch_executor import (
    PatchApplicationResult,
    apply_verified_patch,
    execute_patch_plan,
)
from repair.patch_planner import PatchPlan
from repair.repair_runs import load_past_lessons, save_repair_run


class TestKnowledgeBase(unittest.TestCase):
    def _create_repo_with_docs_codeowners_and_runs(self) -> tuple[tempfile.TemporaryDirectory, str]:
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        (root / "docs").mkdir()
        (root / ".github").mkdir()
        (root / "core").mkdir()

        # Tier 1: Runbook with explicit Step 1 safety preconditions & Step 3 remediation
        (root / "docs" / "RUNBOOK_BATCH.md").write_text(
            "# Batch Processing Operations\n\n"
            "## Batch Iterator Failure Recovery\n"
            "### Step 1: Safety Preconditions\n"
            "WARNING: Before modifying core/batch.py, verify that active stream consumers are drained "
            "and never mutate tuple items in-place.\n\n"
            "### Step 2: Diagnostic Check\n"
            "Run `python -m unittest tests/test_batch.py` in an isolated worktree.\n\n"
            "### Step 3: Remediation\n"
            "Ensure slice stride `xs[i:i + n]` is used instead of repeating `[x] * n`.\n",
            encoding="utf-8",
        )

        # Tier 1: CODEOWNERS
        (root / ".github" / "CODEOWNERS").write_text(
            "# Core platform owners\n"
            "core/* @sre-core-team @platform-lead\n",
            encoding="utf-8",
        )

        (root / "core" / "batch.py").write_text(
            "def split_batches(xs, n):\n    return [tuple(xs[i:i+n]) for i in range(0, len(xs), n)]\n",
            encoding="utf-8",
        )

        runs_dir = str(root / ".asre" / "runs")

        # Case 1 (Positive Trajectory): Verified slice stride fix that survived
        ctx_pos = RepairContext(
            incident=Incident("inc-pos-001", "split_batches returned repeated elements", exception_type="AssertionError"),
            repository_path=str(root),
            evidence=[
                RepairEvidence(
                    path="core/batch.py",
                    score=120.0,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                )
            ],
        )
        plan_pos = PatchPlan(
            "Use slice stride range(0, len(xs), n) in split_batches",
            ("core/batch.py",),
            "--- a/core/batch.py\n+++ b/core/batch.py\n@@ -1,2 +1,2 @@\n def split_batches(xs, n):\n-    return [tuple([x]*n) for x in xs]\n+    return [tuple(xs[i:i+n]) for i in range(0, len(xs), n)]\n",
            ("python -m unittest",),
            0.95,
        )
        run_pos = save_repair_run(runs_dir, ctx_pos, plan=plan_pos)
        run_pos_json = Path(run_pos.path) / "run.json"
        run_pos_json.write_text(
            run_pos_json.read_text(encoding="utf-8").replace('"needs-review"', '"verified"'),
            encoding="utf-8",
        )

        # Case 2 (Negative Trajectory — Rolled Back after health check failure!)
        ctx_neg = RepairContext(
            incident=Incident("inc-neg-002", "split_batches timeout on empty stride", exception_type="TimeoutError"),
            repository_path=str(root),
            evidence=[
                RepairEvidence(
                    path="core/batch.py",
                    score=115.0,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                )
            ],
        )
        plan_neg = PatchPlan(
            "Remove stride validation check in split_batches",
            ("core/batch.py",),
            "--- a/core/batch.py\n+++ b/core/batch.py\n@@ -1,2 +1,2 @@\n def split_batches(xs, n):\n-    if n <= 0: raise ValueError()\n+    pass\n",
            ("python -m unittest",),
            0.70,
        )
        app_rolled_back = PatchApplicationResult(
            status="rolled_back",
            approved=True,
            applied_files=("core/batch.py",),
            error="post-apply health check exited 1",
            rolled_back=True,
        )
        save_repair_run(runs_dir, ctx_neg, plan=plan_neg, application=app_rolled_back)

        return temp_dir, runs_dir

    def test_3_tier_knowledge_base_parent_doc_reranking_negative_cbr_and_thompson_sampling(self):
        temp_dir, runs_dir = self._create_repo_with_docs_codeowners_and_runs()
        self.addCleanup(temp_dir.cleanup)

        report = build_and_query_knowledge_base(
            temp_dir.name,
            query="Batch Iterator Failure slice stride core/batch.py",
            target_files=("core/batch.py",),
            repair_run_dir=runs_dir,
        )

        # Tier 1: Parent-Document chunking preserves Step 1 safety preconditions alongside Step 3
        self.assertGreaterEqual(report.total_semantic_parent_docs, 1)
        self.assertTrue(report.tier1_semantic_hits)
        top_hit = report.tier1_semantic_hits[0]
        self.assertEqual("docs/RUNBOOK_BATCH.md", top_hit.parent_doc.source_path)
        self.assertTrue(top_hit.parent_doc.has_safety_preconditions)
        self.assertIn("Step 1: Safety Preconditions", top_hit.parent_doc.full_content)
        self.assertIn("Step 3: Remediation", top_hit.parent_doc.full_content)

        # CODEOWNERS resolution
        self.assertIn("core/batch.py", report.resolved_owners)
        self.assertIn("@sre-core-team", report.resolved_owners["core/batch.py"])

        # Tier 2: Episodic CBR stores both Positive (verified) and Negative (rolled_back) trajectories
        self.assertEqual(1, report.positive_cases_count)
        self.assertEqual(1, report.negative_cases_count)
        self.assertEqual("inc-pos-001", report.tier2_positive_cases[0].incident_id)
        self.assertEqual("collection_boundary_or_slice_fix", report.tier2_positive_cases[0].structural_operator)
        self.assertEqual("inc-neg-002", report.tier2_negative_cases[0].incident_id)
        self.assertIn("ROLLED BACK", report.tier2_negative_cases[0].anti_pattern_warning)

        # Tier 3: Procedural Playbooks ranked via Thompson Sampling Bandit (verified playbook > rolled-back playbook)
        self.assertGreaterEqual(report.total_procedural_skills, 4)
        playbooks = [s for s in report.tier3_procedural_skills if s.skill_kind == "ast_code_playbook"]
        self.assertTrue(playbooks)
        self.assertEqual("playbook:collection_boundary_or_slice_fix", playbooks[0].skill_id)
        self.assertGreater(playbooks[0].posterior_confidence, 0.60)

        # Verify LLM prompt receives both Positive Playbooks (with diff) and Negative Trajectory Guardrails
        query_ctx = RepairContext(
            incident=Incident("inc-new-003", "split_batches slice stride bug", exception_type="AssertionError"),
            repository_path=temp_dir.name,
            evidence=[
                RepairEvidence(
                    path="core/batch.py",
                    score=125.0,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                )
            ],
        )
        lessons = load_past_lessons(runs_dir, query_ctx, include_negative=True)
        polarities = {l.polarity for l in lessons}
        self.assertIn("positive", polarities)
        self.assertIn("negative", polarities)

        _, user_prompt = build_repair_prompt(query_ctx, past_lessons=lessons)
        self.assertIn("TIER 2 & TIER 3 PLAYBOOKS", user_prompt)
        self.assertIn("Verified Patch Diff:", user_prompt)
        self.assertIn("NEGATIVE TRAJECTORY GUARDRAILS", user_prompt)
        self.assertIn("ROLLED BACK", user_prompt)

    def test_classify_structural_fix_operator(self):
        diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n+if val is None:\n+    return []\n"
        self.assertEqual("precondition_or_null_guard", classify_structural_fix_operator(diff))

    def test_closed_loop_distillation_trust_promotion_drift_pruning_and_safety_guardrails(self):
        temp_dir, runs_dir = self._create_repo_with_docs_codeowners_and_runs()
        self.addCleanup(temp_dir.cleanup)
        repo = temp_dir.name

        # 1. Newly saved repair run starts as kb_trust_status='unverified' and includes 4-part DistilledRepairRecord
        cases_initial = load_tier2_episodic_cases(runs_dir, repo)
        pos_case = next(c for c in cases_initial if c.incident_id == "inc-pos-001")
        self.assertEqual("unverified", pos_case.kb_trust_status)
        self.assertEqual("clean", pos_case.drift_status)

        distilled = distill_post_incident_record(
            "inc-pos-001",
            "AssertionError",
            "split_batches returned repeated elements",
            ("core/batch.py",),
            pos_case.diagnosis,
            pos_case.unified_diff,
            pos_case.tests_to_run,
            dead_end_actions=("Attempt 1 mutated input list in-place and failed test_immutability",),
            repo_root=repo,
        )
        self.assertEqual("unverified", distilled.kb_trust_status)
        self.assertIn("## 1. Root Cause Signature", distilled.reflection_markdown)
        self.assertIn("## 2. Winning Fix", distilled.reflection_markdown)
        self.assertIn("## 3. Dead-End Actions to Avoid", distilled.reflection_markdown)
        self.assertIn("Attempt 1 mutated input list", distilled.reflection_markdown)
        self.assertIn("## 4. Preconditions & Blast Radius", distilled.reflection_markdown)

        # 2. Pre-execution guardrail blocks autonomous execution while kb_trust_status == 'unverified'
        guard_before = verify_stored_repair_guardrails(repo, pos_case, max_risk_score=10)
        self.assertFalse(guard_before.autonomous_execution_permitted)
        self.assertFalse(guard_before.trust_gate_verified)
        self.assertTrue(guard_before.requires_human_approval)

        # 3. Human-in-the-Loop Promotion Gate promotes inc-pos-001 to kb_trust_status='verified'
        promo = promote_knowledge_base_record(runs_dir, "inc-pos-001", approver="senior-sre-alice")
        self.assertEqual("verified", promo["kb_trust_status"])

        cases_after = load_tier2_episodic_cases(runs_dir, repo)
        promoted_case = next(c for c in cases_after if c.incident_id == "inc-pos-001")
        self.assertEqual("verified", promoted_case.kb_trust_status)
        self.assertEqual("senior-sre-alice", promoted_case.promoted_by)

        # 4. Now all 3 pre-execution safety guardrails pass for low-risk reversible repair
        guard_after = verify_stored_repair_guardrails(repo, promoted_case, max_risk_score=10)
        self.assertTrue(guard_after.autonomous_execution_permitted)
        self.assertTrue(guard_after.preconditions_verified)
        self.assertTrue(guard_after.trust_gate_verified)
        self.assertTrue(guard_after.risk_policy_passed)

        # High-blast-radius or sensitive-domain repair still requires 1-click human approval even when promoted
        guard_high_risk = verify_stored_repair_guardrails(repo, promoted_case, max_risk_score=65)
        self.assertFalse(guard_high_risk.autonomous_execution_permitted)
        self.assertTrue(guard_high_risk.requires_human_approval)

        # 5. Decay & Architectural Drift Pruning: refactor core/batch.py completely so the stored diff drifts
        (Path(repo) / "core" / "batch.py").write_text(
            "def completely_refactored_pipeline(stream):\n    return list(stream)\n",
            encoding="utf-8",
        )
        drift_status, drift_reason = detect_repair_record_drift(
            repo,
            promoted_case.affected_files,
            promoted_case.unified_diff,
        )
        self.assertEqual("drifted", drift_status)
        self.assertIn("drift", drift_reason.lower())

        guard_drifted = verify_stored_repair_guardrails(repo, promoted_case, max_risk_score=10)
        self.assertFalse(guard_drifted.autonomous_execution_permitted)
        self.assertFalse(guard_drifted.drift_free)


if __name__ == "__main__":
    unittest.main()
