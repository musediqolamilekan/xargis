import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from repair.incident_mapper import (
    CodeExcerpt,
    Incident,
    RepairContext,
    RepairEvidence,
    detect_incident_from_command,
)
from repair.llm_planner import (
    LLMPatchPlanner,
    build_repair_prompt,
    load_env_file,
    plan_with_repair_loop,
    resolve_provider_and_key,
)
from repair.patch_planner import PatchEdit, compile_edits_to_unified_diff, validate_patch_plan
from repair.repair_runs import load_past_lessons, save_repair_run


class TestLLMPlanner(unittest.TestCase):
    def _create_git_repo(self) -> tempfile.TemporaryDirectory:
        directory = tempfile.TemporaryDirectory()
        repo = directory.name
        os.makedirs(os.path.join(repo, "app"))
        os.makedirs(os.path.join(repo, "tests"))
        with open(os.path.join(repo, "app", "service.py"), "w", encoding="utf-8") as handle:
            handle.write("def compute(x):\n    return x - 1\n")
        with open(os.path.join(repo, "tests", "test_service.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "import unittest\n"
                "from app.service import compute\n\n"
                "class ServiceTest(unittest.TestCase):\n"
                "    def test_compute(self):\n"
                "        self.assertEqual(11, compute(10))\n"
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

    def _build_context(self, repo_path: str) -> RepairContext:
        return RepairContext(
            incident=Incident(
                incident_id="inc-llm-1",
                message="AssertionError: 11 != 9",
                stack_trace='  File "app/service.py", line 2, in compute',
                exception_type="AssertionError",
            ),
            repository_path=repo_path,
            evidence=[
                RepairEvidence(
                    path="app/service.py",
                    score=135.0,
                    valid_frame_lines=[2],
                    frame_lines=[2],
                    excerpts=[
                        CodeExcerpt(
                            excerpt_id="app/service.py#1",
                            role="frame",
                            start_line=1,
                            end_line=2,
                            target_lines=[2],
                            content="     1: def compute(x):\n>    2:     return x - 1",
                        )
                    ],
                    primary_excerpt_id="app/service.py#1",
                    excerpt="     1: def compute(x):\n>    2:     return x - 1",
                    patch_eligible=True,
                    patch_eligibility_reason="eligible_valid_stack_frame",
                ),
                RepairEvidence(
                    path="app/readonly.py",
                    score=12.0,
                    patch_eligible=False,
                    patch_eligibility_reason="ineligible_query_match_only",
                ),
            ],
        )

    def test_load_env_file_and_resolve_provider_auto_and_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "OPENROUTER_API_KEY=sk-or-test\n"
                "GEMINI_API_KEY=gemini-test\n"
                "GROQ_API_KEY=gsk-test\n",
                encoding="utf-8",
            )
            loaded = load_env_file(env_file)
            self.assertEqual("sk-or-test", loaded["OPENROUTER_API_KEY"])
            self.assertEqual("gemini-test", loaded["GEMINI_API_KEY"])
            self.assertEqual("gsk-test", loaded["GROQ_API_KEY"])

            prov, key = resolve_provider_and_key("auto", env=loaded)
            self.assertEqual("gemini", prov)
            self.assertEqual("gemini-test", key)

            g_prov, g_key = resolve_provider_and_key("groq", env=loaded)
            self.assertEqual("groq", g_prov)
            self.assertEqual("gsk-test", g_key)

    def test_compile_edits_to_unified_diff_handles_raw_and_numbered_excerpt_blocks(self):
        directory = self._create_git_repo()
        self.addCleanup(directory.cleanup)

        files, diff = compile_edits_to_unified_diff(
            [
                PatchEdit(
                    path="app/service.py",
                    search=">    2:     return x - 1",
                    replace=">    2:     return x + 1",
                )
            ],
            directory.name,
        )
        self.assertEqual(("app/service.py",), files)
        self.assertIn("--- a/app/service.py", diff)
        self.assertIn("+++ b/app/service.py", diff)
        self.assertIn("-    return x - 1", diff)
        self.assertIn("+    return x + 1", diff)

    def test_build_repair_prompt_includes_only_patch_eligible_files(self):
        directory = self._create_git_repo()
        self.addCleanup(directory.cleanup)
        context = self._build_context(directory.name)

        system_prompt, user_prompt = build_repair_prompt(context)
        self.assertIn("PATCH-ELIGIBLE FILES", system_prompt)
        self.assertIn("app/service.py", user_prompt)
        self.assertNotIn("app/readonly.py", user_prompt)

    def test_llm_planner_and_self_correction_loop_verifies_in_git_worktree(self):
        directory = self._create_git_repo()
        self.addCleanup(directory.cleanup)
        context = self._build_context(directory.name)

        responses = [
            # Attempt 1: LLM proposes a wrong fix (x * 2 instead of x + 1), which will fail unit tests
            json.dumps({
                "diagnosis": "First attempt with wrong math",
                "files_to_modify": ["app/service.py"],
                "edits": [
                    {
                        "path": "app/service.py",
                        "search": "def compute(x):\n    return x - 1",
                        "replace": "def compute(x):\n    return x * 2",
                    }
                ],
                "tests_to_run": ["python -m unittest discover -s tests"],
                "confidence": 0.6,
            }),
            # Attempt 2: After receiving test failure stderr, LLM self-corrects to x + 1
            "```json\n"
            + json.dumps({
                "diagnosis": "Add 1 instead of subtracting 1",
                "files_to_modify": ["app/service.py"],
                "edits": [
                    {
                        "path": "app/service.py",
                        "search": "def compute(x):\n    return x - 1",
                        "replace": "def compute(x):\n    return x + 1",
                    }
                ],
                "tests_to_run": ["python -m unittest discover -s tests"],
                "confidence": 0.95,
            })
            + "\n```",
        ]
        seen_prompts: list[str] = []

        def fake_transport(system_prompt: str, user_prompt: str) -> str:
            seen_prompts.append(user_prompt)
            return responses[len(seen_prompts) - 1]

        planner = LLMPatchPlanner(transport=fake_transport)
        result = plan_with_repair_loop(planner, context, verify=True, max_attempts=2)

        self.assertEqual(2, len(result.attempts))
        self.assertEqual("failed", result.attempts[0].execution.status)
        self.assertEqual("verified", result.attempts[1].execution.status)
        self.assertTrue(result.attempts[1].execution.reproduced_before_patch)
        self.assertIsNotNone(result.plan)
        self.assertTrue(validate_patch_plan(result.plan, context).valid)
        self.assertEqual("verified", result.execution.status)
        self.assertIn("PREVIOUS ATTEMPT FEEDBACK", seen_prompts[1])
        self.assertIn("AssertionError: 11 != 20", seen_prompts[1])

    def test_detect_recommend_and_learn_full_lifecycle(self):
        directory = self._create_git_repo()
        self.addCleanup(directory.cleanup)

        # Stage 1 (Detect): actively detect the failing test command
        detected = detect_incident_from_command(
            directory.name,
            "python -m unittest discover -s tests",
            incident_id="inc-detected-1",
        )
        self.assertEqual("AssertionError", detected.exception_type)
        self.assertIn("11 != 9", detected.message)

        context = self._build_context(directory.name)
        with tempfile.TemporaryDirectory() as runs_dir:
            def fake_transport(system_prompt: str, user_prompt: str) -> str:
                return json.dumps({
                    "diagnosis": "Use addition instead of subtraction in compute()",
                    "risk_assessment": "Low blast radius; only compute() in app/service.py is modified.",
                    "alternatives": ["Use operator.add(x, 1) instead of x + 1"],
                    "files_to_modify": ["app/service.py"],
                    "edits": [
                        {
                            "path": "app/service.py",
                            "search": "def compute(x):\n    return x - 1",
                            "replace": "def compute(x):\n    return x + 1",
                        }
                    ],
                    "tests_to_run": ["python -m unittest discover -s tests"],
                    "confidence": 0.96,
                })

            planner = LLMPatchPlanner(transport=fake_transport)
            first_loop = plan_with_repair_loop(
                planner,
                context,
                verify=True,
                max_attempts=1,
                repair_run_dir=runs_dir,
            )
            self.assertEqual("verified", first_loop.execution.status)
            self.assertEqual(("Use operator.add(x, 1) instead of x + 1",), first_loop.plan.alternatives)
            self.assertIn("Low blast radius", first_loop.plan.risk_assessment)

            # Save run to .asre/runs
            save_repair_run(
                runs_dir,
                context,
                plan=first_loop.plan,
                validation=first_loop.validation,
                execution=first_loop.execution,
            )

            # Stage 8 (Learn): subsequent investigation retrieves historical lesson
            lessons = load_past_lessons(runs_dir, context)
            self.assertEqual(1, len(lessons))
            self.assertEqual("inc-llm-1", lessons[0].incident_id)
            self.assertIn("addition instead of subtraction", lessons[0].diagnosis)

            second_prompts: list[str] = []

            def second_transport(system_prompt: str, user_prompt: str) -> str:
                second_prompts.append(user_prompt)
                return fake_transport(system_prompt, user_prompt)

            second_planner = LLMPatchPlanner(transport=second_transport)
            second_loop = plan_with_repair_loop(
                second_planner,
                context,
                verify=False,
                max_attempts=1,
                repair_run_dir=runs_dir,
                challenge="Verify whether negative inputs or zero offsets could also trigger this",
            )
            self.assertEqual(1, len(second_loop.lessons_used))
            self.assertEqual(
                "Verify whether negative inputs or zero offsets could also trigger this",
                second_loop.challenge,
            )
            self.assertIn("HISTORICAL LESSONS FROM PAST REPAIRS", second_prompts[0])
            self.assertIn("ENGINEER CHALLENGE / FURTHER INVESTIGATION REQUEST", second_prompts[0])
            self.assertIn("negative inputs or zero offsets", second_prompts[0])
            self.assertIn("addition instead of subtraction", second_prompts[0])


if __name__ == "__main__":
    unittest.main()

