import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from repair.incident_mapper import Incident, RepairContext, RepairEvidence
from repair.patch_executor import (
    apply_verified_patch,
    execute_patch_plan,
    rollback_applied_patch,
)
from repair.patch_planner import PatchPlan
from repair.production_guard import (
    evaluate_approval_policy,
    is_emergency_stop_active,
    redact_secrets,
    set_emergency_stop,
)
from repair.repair_runs import save_repair_run


class TestProductionGuard(unittest.TestCase):
    def _create_repo(self) -> tempfile.TemporaryDirectory:
        directory = tempfile.TemporaryDirectory()
        repo = directory.name
        os.makedirs(os.path.join(repo, "config"))
        os.makedirs(os.path.join(repo, "auth"))
        os.makedirs(os.path.join(repo, "migrations"))
        os.makedirs(os.path.join(repo, "tests"))
        with open(os.path.join(repo, "config", "settings.py"), "w", encoding="utf-8") as handle:
            handle.write("TIMEOUT = 5\n")
        with open(os.path.join(repo, "auth", "login.py"), "w", encoding="utf-8") as handle:
            handle.write("ALLOW_ANONYMOUS = False\n")
        with open(os.path.join(repo, "migrations", "001_init.sql"), "w", encoding="utf-8") as handle:
            handle.write("CREATE TABLE users (id INT);\n")
        with open(os.path.join(repo, "tests", "test_settings.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "import unittest\nfrom pathlib import Path\n\n"
                "class SettingsTest(unittest.TestCase):\n"
                "    def test_timeout(self):\n"
                "        self.assertIn('TIMEOUT = 10', Path('config/settings.py').read_text())\n"
            )
        for cmd in (
            ["git", "init"],
            ["git", "config", "user.email", "asre@example.invalid"],
            ["git", "config", "user.name", "ASRE Test"],
            ["git", "add", "."],
            ["git", "commit", "-m", "initial"],
        ):
            subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True)
        return directory

    def test_low_risk_config_in_dev_auto_approves_while_auth_migration_and_prod_require_explicit_approval(self):
        # 1. Low-risk config change in development -> auto_apply_permitted=True for developer
        dev_ctx = RepairContext(
            incident=Incident("inc-dev-cfg", "timeout too low"),
            repository_path="/repo",
            evidence=[
                RepairEvidence(
                    path="config/settings.py",
                    score=120.0,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                    risk_score=8,
                    risk_label="Low",
                )
            ],
        )
        cfg_plan = PatchPlan(
            "Increase timeout in dev config",
            ("config/settings.py",),
            "--- a/config/settings.py\n+++ b/config/settings.py\n@@ -1 +1 @@\n-TIMEOUT = 5\n+TIMEOUT = 10\n",
            ("python -m unittest discover -s tests",),
            0.9,
        )
        dev_decision = evaluate_approval_policy(
            cfg_plan,
            dev_ctx,
            environment="development",
            role="developer",
            operator="alice",
            explicit_human_approval=False,
        )
        self.assertTrue(dev_decision.auto_apply_permitted)
        self.assertFalse(dev_decision.requires_explicit_approval)
        self.assertTrue(dev_decision.allowed_to_apply)

        # 2. Same low-risk config change in PRODUCTION -> requires explicit approval + sre/admin role
        prod_decision = evaluate_approval_policy(
            cfg_plan,
            dev_ctx,
            environment="production",
            role="developer",
            operator="alice",
            explicit_human_approval=False,
        )
        self.assertFalse(prod_decision.auto_apply_permitted)
        self.assertTrue(prod_decision.requires_explicit_approval)
        self.assertFalse(prod_decision.role_authorized)
        self.assertFalse(prod_decision.allowed_to_apply)

        # 3. Authentication change EVEN in development with low risk_score -> requires explicit approval + sre/admin role
        auth_ctx = RepairContext(
            incident=Incident("inc-auth", "login error"),
            repository_path="/repo",
            evidence=[
                RepairEvidence(
                    path="auth/login.py",
                    score=120.0,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                    risk_score=10,
                    risk_label="Low",
                )
            ],
        )
        auth_plan = PatchPlan(
            "Modify auth login",
            ("auth/login.py",),
            "--- a/auth/login.py\n+++ b/auth/login.py\n@@ -1 +1 @@\n-ALLOW_ANONYMOUS = False\n+ALLOW_ANONYMOUS = True\n",
            ("python -m unittest discover -s tests",),
            0.9,
        )
        auth_unapproved = evaluate_approval_policy(
            auth_plan,
            auth_ctx,
            environment="development",
            role="sre",
            operator="bob",
            explicit_human_approval=False,
        )
        self.assertIn("authentication_or_security", auth_unapproved.sensitive_domains)
        self.assertFalse(auth_unapproved.auto_apply_permitted)
        self.assertFalse(auth_unapproved.allowed_to_apply)

        auth_approved = evaluate_approval_policy(
            auth_plan,
            auth_ctx,
            environment="development",
            role="sre",
            operator="bob",
            explicit_human_approval=True,
            approver="bob-sre-lead",
        )
        self.assertTrue(auth_approved.allowed_to_apply)

        # 4. Database migration change -> detected as database_migration_or_schema and requires explicit approval
        mig_ctx = RepairContext(
            incident=Incident("inc-mig", "migration failure"),
            repository_path="/repo",
            evidence=[
                RepairEvidence(
                    path="migrations/001_init.sql",
                    score=120.0,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                    risk_score=5,
                    risk_label="Low",
                )
            ],
        )
        mig_plan = PatchPlan(
            "Alter users table",
            ("migrations/001_init.sql",),
            "--- a/migrations/001_init.sql\n+++ b/migrations/001_init.sql\n@@ -1 +1,2 @@\n CREATE TABLE users (id INT);\n+ALTER TABLE users ADD COLUMN email TEXT;\n",
            ("python -m unittest discover -s tests",),
            0.9,
        )
        mig_decision = evaluate_approval_policy(
            mig_plan,
            mig_ctx,
            environment="development",
            role="sre",
            explicit_human_approval=False,
        )
        self.assertIn("database_migration_or_schema", mig_decision.sensitive_domains)
        self.assertFalse(mig_decision.auto_apply_permitted)
        self.assertFalse(mig_decision.allowed_to_apply)

    def test_deployment_health_check_failure_triggers_automatic_rollback_and_manual_rollback_works(self):
        directory = self._create_repo()
        self.addCleanup(directory.cleanup)
        repo = directory.name

        context = RepairContext(
            incident=Incident("inc-hc", "timeout issue"),
            repository_path=repo,
            evidence=[
                RepairEvidence(
                    path="config/settings.py",
                    score=130.0,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                    risk_score=8,
                    risk_label="Low",
                )
            ],
        )
        plan = PatchPlan(
            "Update TIMEOUT to 10",
            ("config/settings.py",),
            "--- a/config/settings.py\n+++ b/config/settings.py\n@@ -1 +1 @@\n-TIMEOUT = 5\n+TIMEOUT = 10\n",
            ("python -m unittest discover -s tests",),
            0.92,
        )
        execution = execute_patch_plan(plan, context)
        self.assertEqual("verified", execution.status)

        # 1. Failing post-apply deployment health check -> automatically rolls back the live file to TIMEOUT = 5
        failed_hc = apply_verified_patch(
            plan,
            context,
            execution,
            health_check_command="python -m unittest tests/test_missing_health_check.py",
        )
        self.assertEqual("rolled_back", failed_hc.status)
        self.assertTrue(failed_hc.rolled_back)
        self.assertEqual("TIMEOUT = 5\n", (Path(repo) / "config" / "settings.py").read_text(encoding="utf-8"))

        # 2. Passing post-apply deployment health check -> applies cleanly, then manual rollback restores it
        passed_hc = apply_verified_patch(
            plan,
            context,
            execution,
            health_check_command="python -m unittest discover -s tests",
        )
        self.assertEqual("applied", passed_hc.status)
        self.assertFalse(passed_hc.rolled_back)
        self.assertEqual("TIMEOUT = 10\n", (Path(repo) / "config" / "settings.py").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as runs_dir:
            run = save_repair_run(runs_dir, context, plan=plan, execution=execution, application=passed_hc)
            self.assertTrue((Path(runs_dir) / "audit.jsonl").is_file())
            rb = rollback_applied_patch(repo, run.path)
            self.assertEqual("rolled_back", rb.status)
            self.assertTrue(rb.rolled_back)
            self.assertEqual("TIMEOUT = 5\n", (Path(repo) / "config" / "settings.py").read_text(encoding="utf-8"))

    def test_emergency_stop_blocks_execution_and_secret_redaction_scrubs_tokens(self):
        directory = self._create_repo()
        self.addCleanup(directory.cleanup)
        repo = directory.name

        set_emergency_stop(repo, True, reason="Database maintenance window", actor="sre-oncall")
        active, reason = is_emergency_stop_active(repo)
        self.assertTrue(active)
        self.assertIn("Database maintenance window", reason)

        context = RepairContext(
            incident=Incident("inc-stop", "broken"),
            repository_path=repo,
            evidence=[
                RepairEvidence(
                    path="config/settings.py",
                    score=120.0,
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                )
            ],
        )
        plan = PatchPlan(
            "Update TIMEOUT",
            ("config/settings.py",),
            "--- a/config/settings.py\n+++ b/config/settings.py\n@@ -1 +1 @@\n-TIMEOUT = 5\n+TIMEOUT = 10\n",
            ("python -m unittest discover -s tests",),
            0.9,
        )
        blocked_exec = execute_patch_plan(plan, context)
        self.assertEqual("rejected", blocked_exec.status)
        self.assertIn("emergency stop active", blocked_exec.error)

        set_emergency_stop(repo, False, actor="sre-oncall")
        self.assertFalse(is_emergency_stop_active(repo)[0])

        raw_secret_text = (
            "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 "
            "api_key='sk-or-v1-355407900eac67c1705540d503ccf139' "
            "postgres://admin:SuperSecretPass123@db.internal:5432/prod"
        )
        scrubbed = redact_secrets(raw_secret_text)
        self.assertNotIn("sk-or-v1-355407900eac67c1705540d503ccf139", scrubbed)
        self.assertNotIn("SuperSecretPass123", scrubbed)
        self.assertNotIn("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", scrubbed)
        self.assertIn("[REDACTED_SECRET]", scrubbed)


if __name__ == "__main__":
    unittest.main()
