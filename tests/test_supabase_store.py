import json
import tempfile
import unittest
from pathlib import Path

from repair.incident_mapper import Incident, RepairContext, RepairEvidence
from repair.patch_executor import PatchApplicationResult
from repair.patch_planner import PatchPlan
from repair.repair_runs import save_repair_run
from storage.supabase_store import SupabaseStorageEngine, _encode_db_url_password


class TestSupabaseStore(unittest.TestCase):
    def test_encode_db_url_password_encodes_asterisk_safely(self):
        raw = "postgresql://postgres:myPass*45@db.demo-project-ref.supabase.co:5432/postgres"
        encoded = _encode_db_url_password(raw)
        self.assertIn("myPass%2A45@", encoded)

    def test_dual_write_secret_redaction_offline_queue_flush_and_training_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_dir = root / ".asre"
            runs_dir = queue_dir / "runs"

            captured_requests: list[tuple[str, str, str]] = []
            simulate_online = [False]

            def fake_transport(method: str, url: str, headers: dict, body: bytes | None) -> tuple[int, str]:
                payload_text = body.decode("utf-8") if body else ""
                captured_requests.append((method, url, payload_text))
                if not simulate_online[0]:
                    return 503, "Service temporarily unavailable"
                return 201, "[]"

            store = SupabaseStorageEngine(
                env={
                    "SUPABASE_URL": "https://demo-project-ref.supabase.co",
                    "SUPABASE_KEY": "sb_publishable_testkey1234567890",
                },
                queue_dir=queue_dir,
                transport=fake_transport,
            )

            # 1. Upsert while offline: secrets are scrubbed, row is mirrored locally and queued in pending_sync.jsonl
            res_offline = store.upsert(
                "asre_incidents",
                {
                    "incident_id": "inc-db-1",
                    "repo_id": "click",
                    "message": "Failed with token sk-or-v1-999999999999999999999999 and sb_publishable_abcdefghijklmnop",
                },
            )
            self.assertFalse(res_offline.synced_to_cloud)
            self.assertTrue(res_offline.queued_locally)
            self.assertEqual(1, store.pending_records_count())

            # Verify secret redaction in transmitted/queued payload
            self.assertIn("[REDACTED_SECRET]", captured_requests[0][2])
            self.assertNotIn("sk-or-v1-999999999999999999999999", captured_requests[0][2])
            self.assertNotIn("sb_publishable_abcdefghijklmnop", captured_requests[0][2])

            # 2. Switch online and flush pending queue
            simulate_online[0] = True
            flushed = store.flush_pending_sync()
            self.assertEqual(1, flushed["flushed_batches"])
            self.assertEqual(0, flushed["remaining_batches"])
            self.assertEqual(0, store.pending_records_count())

            # 3. Create a positive and negative repair run and export SFT, DPO, and Tabular Change-Risk datasets
            (root / "app").mkdir()
            (root / "app" / "math_util.py").write_text("def add_one(x):\n    return x + 1\n", encoding="utf-8")

            ctx_pos = RepairContext(
                incident=Incident("inc-train-pos", "add_one returned wrong result", exception_type="AssertionError"),
                repository_path=str(root),
                evidence=[
                    RepairEvidence(
                        path="app/math_util.py",
                        score=110.0,
                        patch_eligible=True,
                        patch_eligibility_reason="eligible_valid_stack_frame",
                    )
                ],
            )
            plan_pos = PatchPlan(
                "Fix addition offset in add_one",
                ("app/math_util.py",),
                "--- a/app/math_util.py\n+++ b/app/math_util.py\n@@ -1,2 +1,2 @@\n def add_one(x):\n-    return x - 1\n+    return x + 1\n",
                ("python -m unittest",),
                0.95,
            )
            run_pos = save_repair_run(
                str(runs_dir),
                ctx_pos,
                plan=plan_pos,
                dead_end_actions=("Attempt 1 used x * 2 and failed unit test",),
                kb_trust_status="verified",
                promoted_by="sre-lead",
            )
            run_pos_json = Path(run_pos.path) / "run.json"
            run_pos_json.write_text(
                run_pos_json.read_text(encoding="utf-8").replace('"needs-review"', '"verified"'),
                encoding="utf-8",
            )

            # Negative rolled-back run on the same file with a different exception
            ctx_neg = RepairContext(
                incident=Incident("inc-train-neg", "add_one division by zero", exception_type="ZeroDivisionError"),
                repository_path=str(root),
                evidence=[
                    RepairEvidence(
                        path="app/math_util.py",
                        score=110.0,
                        patch_eligible=True,
                        patch_eligibility_reason="eligible_valid_stack_frame",
                    )
                ],
            )
            plan_neg = PatchPlan(
                "Broken division in add_one",
                ("app/math_util.py",),
                "--- a/app/math_util.py\n+++ b/app/math_util.py\n@@ -1,2 +1,2 @@\n def add_one(x):\n-    return x - 1\n+    return x / 0\n",
                ("python -m unittest",),
                0.40,
            )
            app_rb = PatchApplicationResult(
                status="rolled_back",
                approved=True,
                applied_files=("app/math_util.py",),
                error="health check failed",
                rolled_back=True,
            )
            save_repair_run(str(runs_dir), ctx_neg, plan=plan_neg, application=app_rb)

            # Record a change-risk row in the store
            store.upsert(
                "asre_change_risk_records",
                {
                    "check_id": "test:aaa..bbb",
                    "repo_id": "test",
                    "stable_commit": "aaa",
                    "target_commit": "bbb",
                    "lines_added": 12,
                    "lines_deleted": 3,
                    "files_touched": 2,
                    "subsystems_touched": 1,
                    "change_entropy": 0.65,
                    "author_familiarity_pct": 40.0,
                    "recent_bugfix_commits": 2,
                    "is_off_hours_deploy": True,
                    "max_reverse_pagerank": 0.08,
                    "max_betweenness_centrality": 0.02,
                    "total_direct_dependents": 5,
                    "total_transitive_dependents": 20,
                    "modified_symbols_count": 2,
                    "config_keys_modified_count": 0,
                    "ast_hazards_count": 1,
                    "prior_incidents_count": 2,
                    "calibrated_probability": 0.88,
                    "incident_caused_label": 1,
                },
            )

            export_dir = root / "training_out"
            export_rep = store.export_training_datasets(
                export_dir,
                repo_root=str(root),
                repair_run_dir=str(runs_dir),
            )
            self.assertGreaterEqual(export_rep.sft_records_count, 1)
            self.assertGreaterEqual(export_rep.dpo_pairs_count, 2)
            self.assertEqual(1, export_rep.change_risk_rows_count)
            self.assertTrue((export_dir / "sft_repairs.jsonl").is_file())
            self.assertTrue((export_dir / "dpo_preference_pairs.jsonl").is_file())
            self.assertTrue((export_dir / "jit_change_risk_training.csv").is_file())


if __name__ == "__main__":
    unittest.main()
