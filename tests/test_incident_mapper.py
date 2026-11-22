import json
import os
import tempfile
import unittest

from intelligence.dependency_graph import build_dependency_graph
from intelligence.file_profile import build_file_profiles
from intelligence.models import FileAnalysis, RepoIntelligence
from repair.incident_mapper import (
    CodeExcerpt,
    ContextTruncation,
    ExcerptCharTruncation,
    FrameSymbolMatch,
    FrameValidation,
    Incident,
    IncidentFrame,
    OmittedTargetLine,
    ParsedStackTrace,
    RepairEvidence,
    RepairFrame,
    SourceSymbolInfo,
    TraceSegment,
    _inspect_repository_file,
    _match_repository_path,
    _split_source_lines,
    _validate_mapped_frame,
    build_repair_context,
    extract_stack_frames,
    extract_stack_frames_detailed,
    load_incident,
)


class TestIncidentMapper(unittest.TestCase):
    def _graph_and_profiles(self, repo_root):
        intel = RepoIntelligence(
            repo_path=repo_root,
            repo_url="",
            file_analyses=[
                FileAnalysis("app/main.py", "python", 12, 1, functions=["run"], imports=["from app.service import execute"]),
                FileAnalysis("app/service.py", "python", 8, 1, functions=["execute"]),
                FileAnalysis("app/other.py", "python", 3, 1, functions=["other"]),
            ],
        )
        graph = build_dependency_graph(intel)
        return graph, build_file_profiles(graph, intel)

    def test_extract_stack_frames_supports_python_and_javascript_formats(self):
        frames = extract_stack_frames('File "app/main.py", line 7, in run\n    at execute (app/service.js:12:3)')
        self.assertEqual(
            [
                IncidentFrame("app/main.py", 7, "run"),
                IncidentFrame("app/service.js", 12, "execute"),
            ],
            frames,
        )
        self.assertEqual(
            {"path": "app/main.py", "line": 7, "function": "run"},
            frames[0].to_dict(),
        )

    def test_extract_stack_frames_preserves_original_traceback_order(self):
        stack_trace = "\n".join([
            "    at bootstrap (app/entry.js:3:1)",
            '  File "app/main.py", line 21, in main',
            "    at execute (app/service.js:12:3)",
            '  File "app/other.py", line 9, in finish',
        ])
        frames = extract_stack_frames(stack_trace)
        self.assertEqual(
            [
                IncidentFrame("app/entry.js", 3, "bootstrap"),
                IncidentFrame("app/main.py", 21, "main"),
                IncidentFrame("app/service.js", 12, "execute"),
                IncidentFrame("app/other.py", 9, "finish"),
            ],
            frames,
        )

    def test_extract_stack_frames_handles_multiple_frames_from_same_file(self):
        stack_trace = "\n".join([
            "Traceback (most recent call last):",
            '  File "app/main.py", line 30, in main',
            "    run()",
            '  File "app/main.py", line 14, in run',
            "    items = list(map(lambda x: step(x), data))",
            '  File "app/main.py", line 14, in <lambda>',
            "    items = list(map(lambda x: step(x), data))",
            '  File "app/main.py", line 8, in step',
            "    return step(value - 1)",
            '  File "app/main.py", line 8, in step',
            "    raise RuntimeError('boom')",
            "RuntimeError: boom",
        ])
        frames = extract_stack_frames(stack_trace)
        self.assertEqual(
            [
                IncidentFrame("app/main.py", 30, "main"),
                IncidentFrame("app/main.py", 14, "run"),
                IncidentFrame("app/main.py", 14, "<lambda>"),
                IncidentFrame("app/main.py", 8, "step"),
                IncidentFrame("app/main.py", 8, "step"),
            ],
            frames,
        )

    def test_extract_stack_frames_ignores_ordinary_error_messages_and_code_snippets(self):
        stack_trace = "\n".join([
            "Traceback (most recent call last):",
            '  File "app/main.py", line 7, in run',
            '    raise RuntimeError("failed while reading app/other.py:99")',
            "RuntimeError: failed while reading app/other.py:99",
            "ValueError: invalid configuration in app/service.py:15:2",
            "Error: could not load app/client.js:42:8 during startup",
            'RuntimeError: File "app/fake.py", line 12, in not_a_frame',
        ])
        frames = extract_stack_frames(stack_trace)
        self.assertEqual([IncidentFrame("app/main.py", 7, "run")], frames)

    def test_extract_stack_frames_handles_malformed_and_incomplete_tracebacks(self):
        self.assertEqual([], extract_stack_frames(""))
        self.assertEqual([], extract_stack_frames("Traceback (most recent call last):\nRuntimeError: boom"))

        incomplete_and_malformed = "\n".join([
            "Traceback (most recent call last):",
            '  File "app/main.py", line 10, in main',
            "    execute()",
            '  File "app/service.py", line 18',
            '  File "app/broken.py", line 0, in invalid_zero_line',
            '  File "app/broken.py", line -4, in invalid_negative_line',
            '  File "app/broken.py", line abc, in invalid_non_numeric',
            '  File "   ", line 12, in empty_path',
            '  File "app/truncated.py", line 15, in',
            '  File "app/truncated.py", line ',
            '  File "app/truncated.py"',
            "    at app/anon.js:24:5",
            "    at bad_js_frame (app/anon.js:0:1)",
            "    at incomplete_js_frame (app/anon.js:",
        ])
        frames = extract_stack_frames(incomplete_and_malformed)
        self.assertEqual(
            [
                IncidentFrame("app/main.py", 10, "main"),
                IncidentFrame("app/service.py", 18, ""),
                IncidentFrame("app/anon.js", 24, ""),
            ],
            frames,
        )

    def test_extract_stack_frames_supports_v8_aliases_async_and_multi_token_names(self):
        stack_trace = "\n".join([
            "Error: failure",
            "    at Object.execute [as run] (app/service.js:12:3)",
            "    at Object.get value [as value] (app/model.ts:28:9)",
            "    at new async Service (app/factory.mjs:44:1)",
            "    at async handleRequest (app/controller.cjs:19:7)",
        ])
        self.assertEqual(
            [
                IncidentFrame("app/service.js", 12, "Object.execute [as run]"),
                IncidentFrame("app/model.ts", 28, "Object.get value [as value]"),
                IncidentFrame("app/factory.mjs", 44, "Service"),
                IncidentFrame("app/controller.cjs", 19, "handleRequest"),
            ],
            extract_stack_frames(stack_trace),
        )

    def test_extract_stack_frames_supports_java_and_kotlin_frames_without_space_before_paren(self):
        stack_trace = "\n".join([
            "java.lang.IllegalStateException: broken",
            "    at com.example.service.Runner.execute(Runner.java:12)",
            "    at com.example.service.KotlinRunner.run(KotlinRunner.kt:34)",
        ])
        self.assertEqual(
            [
                IncidentFrame("Runner.java", 12, "com.example.service.Runner.execute"),
                IncidentFrame("KotlinRunner.kt", 34, "com.example.service.KotlinRunner.run"),
            ],
            extract_stack_frames(stack_trace),
        )

    def test_extract_stack_frames_supports_paths_with_spaces_parentheses_and_file_urls(self):
        stack_trace = "\n".join([
            "    at Page (app/(dashboard)/page.tsx:12:3)",
            "    at execute (/Users/dev/My App/src/service.ts:25:7)",
            "    at load (file:///app/src/loader.mjs:18:2)",
            "    at file:///Users/dev/My%20App/src/entry.js:9:1",
        ])
        self.assertEqual(
            [
                IncidentFrame("app/(dashboard)/page.tsx", 12, "Page"),
                IncidentFrame("/Users/dev/My App/src/service.ts", 25, "execute"),
                IncidentFrame("/app/src/loader.mjs", 18, "load"),
                IncidentFrame("/Users/dev/My App/src/entry.js", 9, ""),
            ],
            extract_stack_frames(stack_trace),
        )

    def test_extract_stack_frames_supports_go_ruby_php_rust_and_other_mapped_languages(self):
        stack_trace = "\n".join([
            "goroutine 1 [running]:",
            "github.com/acme/project/internal/pkg.(*Runner).Execute(0x1, 0x2)",
            "\t/workspace/internal/pkg/service.go:42 +0x85",
            "created by main.main in goroutine 1",
            "\t/workspace/main.go:19 +0x1c",
            "    from app/workers/job.rb:15:in `perform'",
            "    from app/workers/runner.rb:8:in 'run'",
            "#0 /var/www/app/Http/Controller.php(51): App\\Http\\Controller->index()",
            "thread 'main' panicked at src/main.rs:27:5:",
            "   0: crate::engine::process",
            "             at src/engine.rs:88:13",
            "    at App.Program.Main (src/Program.cs:16:4)",
            "    at NativeRunner (src/native/worker.cpp:64:2)",
            "    at ScalaService.run(ScalaService.scala:31)",
            "    at SwiftHandler.handle (Sources/App/Handler.swift:22:1)",
            "    at lua_entry (scripts/handler.lua:11:1)",
        ])
        self.assertEqual(
            [
                IncidentFrame("/workspace/internal/pkg/service.go", 42, "github.com/acme/project/internal/pkg.(*Runner).Execute"),
                IncidentFrame("/workspace/main.go", 19, "main.main"),
                IncidentFrame("app/workers/job.rb", 15, "perform"),
                IncidentFrame("app/workers/runner.rb", 8, "run"),
                IncidentFrame("/var/www/app/Http/Controller.php", 51, "App\\Http\\Controller->index"),
                IncidentFrame("src/main.rs", 27, ""),
                IncidentFrame("src/engine.rs", 88, "crate::engine::process"),
                IncidentFrame("src/Program.cs", 16, "App.Program.Main"),
                IncidentFrame("src/native/worker.cpp", 64, "NativeRunner"),
                IncidentFrame("ScalaService.scala", 31, "ScalaService.run"),
                IncidentFrame("Sources/App/Handler.swift", 22, "SwiftHandler.handle"),
                IncidentFrame("scripts/handler.lua", 11, "lua_entry"),
            ],
            extract_stack_frames(stack_trace),
        )

    def test_extract_stack_frames_prevents_cross_language_pending_function_leakage(self):
        stack_trace = "\n".join([
            '  File "app/main.py", line 10, in main',
            "    execute()",
            "    at app/anon.js:24:5",
            "   0: crate::engine::process",
            "    at app/other_anon.js:31:2",
            "   1: crate::engine::orphan",
            "\t/workspace/main.go:19 +0x1c",
            "main.goFunc(0x1)",
            "             at src/engine.rs:88:13",
        ])
        self.assertEqual(
            [
                IncidentFrame("app/main.py", 10, "main"),
                IncidentFrame("app/anon.js", 24, ""),
                IncidentFrame("app/other_anon.js", 31, ""),
                IncidentFrame("/workspace/main.go", 19, ""),
                IncidentFrame("src/engine.rs", 88, ""),
            ],
            extract_stack_frames(stack_trace),
        )

    def test_extract_stack_frames_supports_go_generics_and_legacy_created_by(self):
        stack_trace = "\n".join([
            "goroutine 18 [running]:",
            "main.Run[int](0x1)",
            "\t/app/main.go:10 +0x10",
            "pkg.(*Cache[string, int]).Get(0x2)",
            "\t/app/cache.go:44 +0x28",
            "created by main.spawn",
            "\t/app/main.go:20 +0x12",
        ])
        self.assertEqual(
            [
                IncidentFrame("/app/main.go", 10, "main.Run[int]"),
                IncidentFrame("/app/cache.go", 44, "pkg.(*Cache[string, int]).Get"),
                IncidentFrame("/app/main.go", 20, "main.spawn"),
            ],
            extract_stack_frames(stack_trace),
        )

    def test_extract_stack_frames_supports_rust_trait_impl_and_closure_symbols(self):
        stack_trace = "\n".join([
            "   0: <app::Service as app::Runner>::run::{{closure}}",
            "             at src/lib.rs:42:9",
            "   1: 0x55555555 - <core::future::GenFuture<T> as core::future::Future>::poll",
            "             at src/future.rs:15:3",
        ])
        self.assertEqual(
            [
                IncidentFrame("src/lib.rs", 42, "<app::Service as app::Runner>::run::{{closure}}"),
                IncidentFrame("src/future.rs", 15, "<core::future::GenFuture<T> as core::future::Future>::poll"),
            ],
            extract_stack_frames(stack_trace),
        )

    def test_extract_stack_frames_ignores_non_runtime_data_and_markup_extensions(self):
        stack_trace = "\n".join([
            "    at config.json:10:2",
            "    at settings.yaml:5:1",
            "    at rules.yml:8:3",
            "    at pyproject.toml:14:1",
            "    at index.html:22:4",
            "    at styles.css:9:1",
            '  File "app/main.py", line 7, in run',
        ])
        self.assertEqual(
            [IncidentFrame("app/main.py", 7, "run")],
            extract_stack_frames(stack_trace),
        )

    def test_context_prioritizes_stack_frames_and_graph_neighbors(self):
        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"))
            with open(os.path.join(repo_root, "app", "main.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(["value = 1"] * 12))
            graph, profiles = self._graph_and_profiles(repo_root)
            incident = Incident("inc-1", "execute failed", 'File "app/main.py", line 7, in run', "RuntimeError")

            context = build_repair_context(incident, graph, profiles, repo_root)

            self.assertEqual("app/main.py", context.evidence[0].path)
            self.assertIn("app/service.py", [item.path for item in context.evidence])
            self.assertIn("7: value = 1", context.evidence[0].excerpt)
            self.assertEqual([], context.unmapped_frames)

    def test_load_incident_validates_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "incident.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"id": "inc-2", "error": "broken", "stack": ["line one"]}, handle)
            incident = load_incident(path)
            self.assertEqual("inc-2", incident.incident_id)
            self.assertEqual("broken", incident.message)
            self.assertEqual("line one", incident.stack_trace)

    def test_match_repository_path_resolves_valid_exact_and_source_root_paths(self):
        with tempfile.TemporaryDirectory() as parent_dir:
            repo_root = os.path.join(parent_dir, "click")
            os.makedirs(repo_root)
            available = {
                "main.py",
                "app/main.py",
                ".github/scripts/ci.py",
                "src/click/core.py",
                "pkg/server/handler.go",
                "vendor/internal/helper.py",
                "services/api/pkg/auth/token.go",
            }
            self.assertEqual("main.py", _match_repository_path("main.py", repo_root, available))
            self.assertEqual("app/main.py", _match_repository_path("app/main.py", repo_root, available))
            self.assertEqual("app/main.py", _match_repository_path("./app/main.py", repo_root, available))
            self.assertEqual("app/main.py", _match_repository_path(r"app\main.py", repo_root, available))
            self.assertEqual(
                ".github/scripts/ci.py",
                _match_repository_path(".github/scripts/ci.py", repo_root, available),
            )
            self.assertEqual(
                "app/main.py",
                _match_repository_path("click/app/main.py", repo_root, available),
            )
            self.assertEqual(
                "src/click/core.py",
                _match_repository_path("click/core.py", repo_root, available),
            )
            self.assertEqual(
                "pkg/server/handler.go",
                _match_repository_path("pkg/server/handler.go", repo_root, available),
            )
            self.assertEqual(
                "vendor/internal/helper.py",
                _match_repository_path("vendor/internal/helper.py", repo_root, available),
            )
            self.assertEqual(
                "services/api/pkg/auth/token.go",
                _match_repository_path("pkg/auth/token.go", repo_root, available),
            )

    def test_match_repository_path_enforces_physical_symlink_containment(self):
        with tempfile.TemporaryDirectory() as sandbox:
            repo_root = os.path.join(sandbox, "repo")
            outside_dir = os.path.join(sandbox, "outside")
            os.makedirs(os.path.join(repo_root, "app"))
            os.makedirs(outside_dir)

            real_file = os.path.join(repo_root, "app", "main.py")
            with open(real_file, "w", encoding="utf-8") as handle:
                handle.write("x = 1\n")

            internal_link = os.path.join(repo_root, "app", "internal_link.py")
            os.symlink(real_file, internal_link)

            outside_file = os.path.join(outside_dir, "secret.py")
            with open(outside_file, "w", encoding="utf-8") as handle:
                handle.write("secret = 1\n")

            escaping_link = os.path.join(repo_root, "app", "escape.py")
            os.symlink(outside_file, escaping_link)

            broken_escaping_link = os.path.join(repo_root, "app", "broken_escape.py")
            os.symlink(os.path.join(outside_dir, "nonexistent.py"), broken_escaping_link)

            symlink_repo_root = os.path.join(sandbox, "repo_alias")
            os.symlink(repo_root, symlink_repo_root)

            available = {
                "app/main.py",
                "app/internal_link.py",
                "app/escape.py",
                "app/broken_escape.py",
            }

            self.assertEqual(
                "app/main.py",
                _match_repository_path(real_file, repo_root, available),
            )
            self.assertEqual(
                "app/main.py",
                _match_repository_path(f"file://{real_file}", repo_root, available),
            )
            self.assertEqual(
                "app/main.py",
                _match_repository_path(
                    os.path.join(symlink_repo_root, "app", "main.py"),
                    repo_root,
                    available,
                ),
            )
            self.assertEqual(
                "app/main.py",
                _match_repository_path(real_file, symlink_repo_root, {"app/main.py"}),
            )
            self.assertEqual(
                "app/internal_link.py",
                _match_repository_path(internal_link, repo_root, {"app/internal_link.py"}),
            )
            # Preserves link identity when both link and target are in available_paths,
            # including when repo_root or frame_path uses the symlinked repo alias
            self.assertEqual(
                "app/internal_link.py",
                _match_repository_path(internal_link, repo_root, available),
            )
            self.assertEqual(
                "app/internal_link.py",
                _match_repository_path(internal_link, symlink_repo_root, available),
            )
            self.assertEqual(
                "app/internal_link.py",
                _match_repository_path(
                    os.path.join(symlink_repo_root, "app", "internal_link.py"),
                    repo_root,
                    available,
                ),
            )
            # Repo-name prefix works for both lexical alias name and physical repo name
            self.assertEqual(
                "app/main.py",
                _match_repository_path("repo_alias/app/main.py", symlink_repo_root, available),
            )
            self.assertEqual(
                "app/main.py",
                _match_repository_path("repo/app/main.py", symlink_repo_root, available),
            )

            # Symlinks escaping repo_root must be rejected whether passed as absolute or relative
            self.assertIsNone(_match_repository_path(escaping_link, repo_root, available))
            self.assertIsNone(_match_repository_path(escaping_link, symlink_repo_root, available))
            self.assertIsNone(_match_repository_path("app/escape.py", repo_root, available))
            self.assertIsNone(_match_repository_path("app/escape.py", symlink_repo_root, available))
            self.assertIsNone(_match_repository_path(broken_escaping_link, repo_root, available))
            self.assertIsNone(_match_repository_path("app/broken_escape.py", repo_root, available))

    def test_match_repository_path_rejects_external_and_unrelated_paths(self):
        with tempfile.TemporaryDirectory() as repo_root:
            available = {"main.py", "app/main.py", "app/service.py", "index.js", "src/click/core.py"}

            # Relative external dependency and unrelated longer paths
            self.assertIsNone(
                _match_repository_path("site-packages/app/main.py", repo_root, available)
            )
            self.assertIsNone(
                _match_repository_path(".venv/lib/python3.12/site-packages/app/main.py", repo_root, available)
            )
            self.assertIsNone(
                _match_repository_path("node_modules/pkg/index.js", repo_root, available)
            )
            self.assertIsNone(
                _match_repository_path("vendor/acme/app/service.py", repo_root, available)
            )
            self.assertIsNone(
                _match_repository_path("pkg/mod/github.com/acme/lib@v1.0.0/app/main.py", repo_root, available)
            )
            self.assertIsNone(
                _match_repository_path("other_service/app/main.py", repo_root, available)
            )
            self.assertIsNone(
                _match_repository_path("external/main.py", repo_root, available)
            )
            self.assertIsNone(
                _match_repository_path("../app/main.py", repo_root, available)
            )

            # In-repo virtualenv or untracked absolute paths
            self.assertIsNone(
                _match_repository_path(
                    os.path.join(repo_root, ".venv", "lib", "python3.12", "site-packages", "app", "main.py"),
                    repo_root,
                    available,
                )
            )
            self.assertIsNone(
                _match_repository_path(
                    os.path.join(repo_root, "untracked", "main.py"),
                    repo_root,
                    available,
                )
            )

            # External POSIX and Windows absolute paths
            self.assertIsNone(
                _match_repository_path("/usr/lib/python3.12/site-packages/app/main.py", repo_root, available)
            )
            self.assertIsNone(
                _match_repository_path("/opt/other_repo/app/main.py", repo_root, available)
            )
            self.assertIsNone(
                _match_repository_path(r"C:\other_repo\app\main.py", repo_root, available)
            )
            self.assertIsNone(
                _match_repository_path("file:///C:/other_repo/app/main.py", repo_root, available)
            )
            self.assertIsNone(
                _match_repository_path(r"\Windows\System32\main.py", repo_root, available)
            )

    def test_match_repository_path_rejects_duplicate_filenames_ambiguous_suffixes_and_collisions(self):
        with tempfile.TemporaryDirectory() as repo_root:
            # Bare filename not at repository root
            self.assertIsNone(
                _match_repository_path("service.py", repo_root, {"app/service.py"})
            )
            self.assertIsNone(
                _match_repository_path("service.py", repo_root, {"app/service.py", "billing/service.py"})
            )

            # Ambiguous multi-segment suffix match
            self.assertIsNone(
                _match_repository_path(
                    "core/handler.py",
                    repo_root,
                    {"service_a/core/handler.py", "service_b/core/handler.py"},
                )
            )

            # Colliding normalized paths in available_paths must return None
            self.assertIsNone(
                _match_repository_path(
                    "app/main.py",
                    repo_root,
                    {"app/main.py", "./app/main.py"},
                )
            )
            self.assertIsNone(
                _match_repository_path(
                    "app/main.py",
                    repo_root,
                    {"app/main.py", r"app\main.py"},
                )
            )

    def test_match_repository_path_validates_jvm_package_qualified_frames(self):
        with tempfile.TemporaryDirectory() as repo_root:
            available = {
                "Main.java",
                "src/main/java/com/acme/Service.java",
                "src/main/java/com/acme/auth/Service.java",
                "src/main/java/com/acme/billing/Service.java",
                "src/main/kotlin/com/acme/auth.kt",
                "src/main/kotlin/com/acme/auth/Service.kt",
            }

            # Disambiguates duplicate Service.java via package prefix in frame_function
            self.assertEqual(
                "src/main/java/com/acme/auth/Service.java",
                _match_repository_path(
                    "Service.java",
                    repo_root,
                    available,
                    "com.acme.auth.Service.login",
                ),
            )
            self.assertEqual(
                "src/main/java/com/acme/billing/Service.java",
                _match_repository_path(
                    "Service.java",
                    repo_root,
                    available,
                    "com.acme.billing.Service.charge",
                ),
            )

            # Nested/anonymous classes ($-separated JVM binary class names)
            self.assertEqual(
                "src/main/java/com/acme/auth/Service.java",
                _match_repository_path(
                    "Service.java",
                    repo_root,
                    available,
                    "com.acme.auth.Service$Inner$1.run",
                ),
            )

            # Kotlin-generated class and synthetic method names
            self.assertEqual(
                "src/main/kotlin/com/acme/auth/Service.kt",
                _match_repository_path(
                    "Service.kt",
                    repo_root,
                    available,
                    "com.acme.auth.ServiceKt.login",
                ),
            )
            self.assertEqual(
                "src/main/kotlin/com/acme/auth/Service.kt",
                _match_repository_path(
                    "Service.kt",
                    repo_root,
                    available,
                    "com.acme.auth.ServiceKt$login$1.invokeSuspend",
                ),
            )
            self.assertEqual(
                "src/main/kotlin/com/acme/auth/Service.kt",
                _match_repository_path(
                    "Service.kt",
                    repo_root,
                    available,
                    "com.acme.auth.Service.login$default",
                ),
            )

            # Java 9+ module and classloader prefixes
            self.assertEqual(
                "src/main/java/com/acme/auth/Service.java",
                _match_repository_path(
                    "Service.java",
                    repo_root,
                    available,
                    "app//com.acme.auth.Service.login",
                ),
            )
            self.assertEqual(
                "src/main/java/com/acme/auth/Service.java",
                _match_repository_path(
                    "Service.java",
                    repo_root,
                    available,
                    "com.acme.auth@1.0/com.acme.auth.Service.login",
                ),
            )
            self.assertEqual(
                "src/main/java/com/acme/auth/Service.java",
                _match_repository_path(
                    "Service.java",
                    repo_root,
                    available,
                    "loader/com.acme.auth@1.0/com.acme.auth.Service.login",
                ),
            )

            # Default-package class matches root Main.java, but never nested com/acme/Main.java
            self.assertEqual(
                "Main.java",
                _match_repository_path("Main.java", repo_root, available, "Main.main"),
            )
            self.assertIsNone(
                _match_repository_path(
                    "Main.java",
                    repo_root,
                    {"src/main/java/com/acme/Main.java"},
                    "Main.main",
                )
            )

            # External JVM packages, mismatched class stems, and malformed function names must return None
            self.assertIsNone(
                _match_repository_path(
                    "Service.java",
                    repo_root,
                    available,
                    "org.springframework.web.Service.login",
                )
            )
            self.assertIsNone(
                _match_repository_path(
                    "Service.java",
                    repo_root,
                    available,
                    "com.acme.auth.UnrelatedClass.login",
                )
            )
            # Package/class-name collisions must not mistake a package component for the outer class
            self.assertIsNone(
                _match_repository_path(
                    "Service.java",
                    repo_root,
                    available,
                    "com.acme.Service.UnrelatedClass.login",
                )
            )
            self.assertIsNone(
                _match_repository_path(
                    "Service.java",
                    repo_root,
                    available,
                    "com.acme.auth.Service.Inner.run",
                )
            )
            self.assertIsNone(
                _match_repository_path(
                    "auth.kt",
                    repo_root,
                    available,
                    "com.acme.auth.UnrelatedClassKt.login",
                )
            )
            for malformed_fn in (
                "com..acme.auth.Service.login",
                "com.acme.123bad.Service.login",
                "//com.acme.auth.Service.login",
                "..//com.acme.auth.Service.login",
                "---@/com.acme.auth.Service.login",
                "app..bad/com.acme.auth.Service.login",
                "a/b/c/com.acme.auth.Service.login",
                "com.acme.auth.Service.login with spaces",
                "../com/acme/auth/Service.login",
                "login",
                "",
            ):
                self.assertIsNone(
                    _match_repository_path("Service.java", repo_root, available, malformed_fn),
                    msg=f"Expected None for malformed function {malformed_fn!r}",
                )

    def test_build_repair_context_leaves_external_suffix_collisions_unmapped(self):
        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"))
            with open(os.path.join(repo_root, "app", "main.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(["value = 1"] * 12))
            graph, profiles = self._graph_and_profiles(repo_root)
            stack_trace = "\n".join([
                '  File ".venv/lib/python3.12/site-packages/app/service.py", line 4, in external_call',
                '  File "app/main.py", line 7, in run',
            ])
            incident = Incident("inc-ext", "execute failed", stack_trace, "RuntimeError")

            context = build_repair_context(incident, graph, profiles, repo_root)

            self.assertEqual("app/main.py", context.evidence[0].path)
            service_evidence = next(item for item in context.evidence if item.path == "app/service.py")
            self.assertEqual([], service_evidence.frame_lines)
            self.assertEqual(
                [IncidentFrame(".venv/lib/python3.12/site-packages/app/service.py", 4, "external_call")],
                context.unmapped_frames,
            )

    def test_build_repair_context_maps_jvm_frames_and_rejects_external_or_mismatched_classes(self):
        with tempfile.TemporaryDirectory() as repo_root:
            acme_java = "src/main/java/com/acme/Service.java"
            auth_java = "src/main/java/com/acme/auth/Service.java"
            billing_java = "src/main/java/com/acme/billing/Service.java"
            auth_kt = "src/main/kotlin/com/acme/auth/Service.kt"
            for rel_path in (acme_java, auth_java, billing_java, auth_kt):
                full_path = os.path.join(repo_root, *rel_path.split("/"))
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as handle:
                    handle.write("\n".join(["// line"] * 25))

            intel = RepoIntelligence(
                repo_path=repo_root,
                repo_url="",
                file_analyses=[
                    FileAnalysis(acme_java, "java", 25, 1, classes=["Service"], functions=["rootCall"]),
                    FileAnalysis(auth_java, "java", 25, 1, classes=["Service"], functions=["login"]),
                    FileAnalysis(billing_java, "java", 25, 1, classes=["Service"], functions=["charge"]),
                    FileAnalysis(auth_kt, "kotlin", 25, 1, functions=["login"]),
                ],
            )
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel)

            stack_trace = "\n".join([
                "    at app//com.acme.auth.Service$Inner.login(Service.java:14)",
                "    at com.acme.auth.ServiceKt$login$1.invokeSuspend(Service.kt:9)",
                "    at org.springframework.web.Service.dispatch(Service.java:20)",
                "    at com.acme.auth.UnrelatedClass.helper(Service.java:18)",
                "    at com.acme.Service.UnrelatedClass.login(Service.java:22)",
                "    at ..//com.acme.billing.Service.charge(Service.java:11)",
            ])
            incident = Incident("inc-jvm", "auth failed", stack_trace, "IllegalStateException")

            context = build_repair_context(incident, graph, profiles, repo_root)

            evidence_by_path = {item.path: item for item in context.evidence}
            self.assertEqual([14], evidence_by_path[auth_java].frame_lines)
            self.assertEqual([9], evidence_by_path[auth_kt].frame_lines)
            self.assertNotIn(acme_java, evidence_by_path)
            self.assertNotIn(billing_java, evidence_by_path)
            self.assertEqual(
                [
                    IncidentFrame("Service.java", 20, "org.springframework.web.Service.dispatch"),
                    IncidentFrame("Service.java", 18, "com.acme.auth.UnrelatedClass.helper"),
                    IncidentFrame("Service.java", 22, "com.acme.Service.UnrelatedClass.login"),
                    IncidentFrame("Service.java", 11, "..//com.acme.billing.Service.charge"),
                ],
                context.unmapped_frames,
            )

    def test_build_repair_context_rejects_escaping_symlinks_and_supports_symlinked_repo_root(self):
        with tempfile.TemporaryDirectory() as sandbox:
            repo_root = os.path.join(sandbox, "repo")
            outside_dir = os.path.join(sandbox, "outside")
            os.makedirs(os.path.join(repo_root, "app"))
            os.makedirs(outside_dir)

            real_main = os.path.join(repo_root, "app", "main.py")
            with open(real_main, "w", encoding="utf-8") as handle:
                handle.write("\n".join(["value = 1"] * 12))

            outside_secret = os.path.join(outside_dir, "secret.py")
            with open(outside_secret, "w", encoding="utf-8") as handle:
                handle.write("secret = 42\n")

            escaping_service = os.path.join(repo_root, "app", "service.py")
            os.symlink(outside_secret, escaping_service)

            symlink_repo_root = os.path.join(sandbox, "repo_alias")
            os.symlink(repo_root, symlink_repo_root)

            graph, profiles = self._graph_and_profiles(symlink_repo_root)
            stack_trace = "\n".join([
                f'  File "{escaping_service}", line 1, in leaked',
                '  File "app/service.py", line 1, in leaked_rel',
                '  File "repo/app/main.py", line 7, in run',
            ])
            incident = Incident("inc-sym", "execute failed", stack_trace, "RuntimeError")

            context = build_repair_context(incident, graph, profiles, symlink_repo_root)

            evidence_by_path = {item.path: item for item in context.evidence}
            self.assertEqual([7], evidence_by_path["app/main.py"].frame_lines)
            self.assertEqual([], evidence_by_path["app/service.py"].frame_lines)
            self.assertEqual(
                [
                    IncidentFrame(escaping_service, 1, "leaked"),
                    IncidentFrame("app/service.py", 1, "leaked_rel"),
                ],
                context.unmapped_frames,
            )

    def test_build_repair_context_rejects_duplicate_filenames_ambiguous_suffixes_and_profile_collisions(self):
        with tempfile.TemporaryDirectory() as repo_root:
            for rel_path in (
                "app/main.py",
                "app/service.py",
                "billing/service.py",
                "svc_a/core/handler.py",
                "svc_b/core/handler.py",
                "app/collide.py",
            ):
                full_path = os.path.join(repo_root, *rel_path.split("/"))
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as handle:
                    handle.write("\n".join(["value = 1"] * 12))

            intel = RepoIntelligence(
                repo_path=repo_root,
                repo_url="",
                file_analyses=[
                    FileAnalysis("app/main.py", "python", 12, 1, functions=["run"]),
                    FileAnalysis("app/service.py", "python", 12, 1, functions=["execute"]),
                    FileAnalysis("billing/service.py", "python", 12, 1, functions=["execute"]),
                    FileAnalysis("svc_a/core/handler.py", "python", 12, 1, functions=["handle"]),
                    FileAnalysis("svc_b/core/handler.py", "python", 12, 1, functions=["handle"]),
                    FileAnalysis("app/collide.py", "python", 12, 1, functions=["clash"]),
                ],
            )
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel)
            profiles["./app/collide.py"] = profiles["app/collide.py"]

            stack_trace = "\n".join([
                '  File "service.py", line 3, in execute',
                '  File "core/handler.py", line 5, in handle',
                '  File "app/collide.py", line 6, in clash',
                '  File "app/main.py", line 7, in run',
            ])
            incident = Incident("inc-ambig", "failure", stack_trace, "RuntimeError")

            context = build_repair_context(incident, graph, profiles, repo_root)

            self.assertEqual([ "app/main.py" ], [item.path for item in context.evidence if item.frame_lines])
            self.assertEqual([7], context.evidence[0].frame_lines)
            self.assertEqual(
                [
                    IncidentFrame("service.py", 3, "execute"),
                    IncidentFrame("core/handler.py", 5, "handle"),
                    IncidentFrame("app/collide.py", 6, "clash"),
                ],
                context.unmapped_frames,
            )

    def test_extract_stack_frames_detailed_preserves_duplicates_functions_columns_and_python_order(self):
        stack_trace = "\n".join([
            "Traceback (most recent call last):",
            '  File "app/main.py", line 30, in main',
            "    run()",
            '  File "app/main.py", line 14, in run',
            '  File "app/main.py", line 14, in <lambda>',
            '  File "app/main.py", line 8, in step',
            '  File "app/main.py", line 8, in step',
            '  File "app/service.py", line 18',
            "RuntimeError: boom",
        ])
        parsed = extract_stack_frames_detailed(stack_trace)

        self.assertIsInstance(parsed, ParsedStackTrace)
        self.assertEqual(1, len(parsed.trace_segments))
        self.assertEqual(parsed.trace_segments, parsed.segments)

        seg = parsed.trace_segments[0]
        self.assertEqual(0, seg.segment_id)
        self.assertEqual("Traceback (most recent call last):", seg.header)
        self.assertEqual("python", seg.language)
        self.assertEqual("outer_to_inner", seg.frame_order)
        self.assertEqual("high", seg.order_confidence)
        self.assertEqual("python_traceback_header", seg.order_reason)
        self.assertEqual([0, 1, 2, 3, 4, 5], seg.frame_indices)
        self.assertEqual(5, seg.innermost_call_frame_index)
        self.assertIsNone(seg.innermost_mapped_frame_index)
        self.assertIsNone(seg.innermost_valid_frame_index)

        self.assertEqual(6, len(parsed.frames))
        self.assertEqual(
            [
                (0, 0, "app/main.py", 30, "main", False, 5),
                (1, 1, "app/main.py", 14, "run", False, 4),
                (2, 2, "app/main.py", 14, "<lambda>", False, 3),
                (3, 3, "app/main.py", 8, "step", False, 2),
                (4, 4, "app/main.py", 8, "step", False, 1),
                (5, 5, "app/service.py", 18, None, True, 0),
            ],
            [
                (
                    f.frame_index,
                    f.segment_position,
                    f.reported_path,
                    f.reported_line,
                    f.reported_function,
                    f.is_innermost_call,
                    f.call_depth_from_innermost,
                )
                for f in parsed.frames
            ],
        )
        for frame in parsed.frames:
            self.assertIsInstance(frame, RepairFrame)
            self.assertEqual("call", frame.frame_kind)
            self.assertEqual("outer_to_inner", frame.frame_order)
            self.assertIsNone(frame.reported_column)
            self.assertIsNone(frame.is_innermost_mapped)
            self.assertIsNone(frame.is_innermost_valid)

    def test_extract_stack_frames_detailed_handles_inner_to_outer_and_origin_frames(self):
        node_trace = "\n".join([
            "Error: boom",
            "    at execute (app/service.ts:12:7)",
            "    at main (app/entry.js:4:2)",
        ])
        node_parsed = extract_stack_frames_detailed(node_trace)
        self.assertEqual(1, len(node_parsed.trace_segments))
        node_seg = node_parsed.trace_segments[0]
        self.assertEqual("node", node_seg.language)
        self.assertEqual("inner_to_outer", node_seg.frame_order)
        self.assertEqual("high", node_seg.order_confidence)
        self.assertEqual(0, node_seg.innermost_call_frame_index)
        self.assertEqual(7, node_parsed.frames[0].reported_column)
        self.assertTrue(node_parsed.frames[0].is_innermost_call)
        self.assertEqual(0, node_parsed.frames[0].call_depth_from_innermost)
        self.assertFalse(node_parsed.frames[1].is_innermost_call)
        self.assertEqual(1, node_parsed.frames[1].call_depth_from_innermost)

        go_trace = "\n".join([
            "goroutine 1 [running]:",
            "main.work(0x1)",
            "\t/workspace/work.go:42 +0x85",
            "main.run()",
            "\t/workspace/main.go:18 +0x20",
            "created by main.main in goroutine 1",
            "\t/workspace/main.go:10 +0x1c",
        ])
        go_parsed = extract_stack_frames_detailed(go_trace)
        self.assertEqual(1, len(go_parsed.trace_segments))
        go_seg = go_parsed.trace_segments[0]
        self.assertEqual("go", go_seg.language)
        self.assertEqual("inner_to_outer", go_seg.frame_order)
        self.assertEqual("high", go_seg.order_confidence)
        self.assertEqual(0, go_seg.innermost_call_frame_index)
        self.assertEqual(["call", "call", "origin"], [f.frame_kind for f in go_parsed.frames])
        self.assertEqual([True, False, None], [f.is_innermost_call for f in go_parsed.frames])
        self.assertEqual([0, 1, None], [f.call_depth_from_innermost for f in go_parsed.frames])
        self.assertIn("main.work(0x1)\n\t/workspace/work.go:42 +0x85", go_parsed.frames[0].raw_text)

    def test_extract_stack_frames_detailed_segments_chained_tracebacks_and_never_guesses_ambiguous_order(self):
        chained_python = "\n".join([
            "Traceback (most recent call last):",
            '  File "app/db.py", line 10, in query',
            '  File "app/driver.py", line 55, in send',
            "ConnectionError: socket closed",
            "",
            "The above exception was the direct cause of the following exception:",
            "",
            "Traceback (most recent call last):",
            '  File "app/main.py", line 20, in run',
            '  File "app/service.py", line 14, in execute',
            "RuntimeError: query failed",
        ])
        parsed_chained = extract_stack_frames_detailed(chained_python)
        self.assertEqual(2, len(parsed_chained.trace_segments))
        self.assertEqual([0, 1], parsed_chained.trace_segments[0].frame_indices)
        self.assertEqual(1, parsed_chained.trace_segments[0].innermost_call_frame_index)
        self.assertEqual([2, 3], parsed_chained.trace_segments[1].frame_indices)
        self.assertEqual(3, parsed_chained.trace_segments[1].innermost_call_frame_index)
        self.assertEqual([0, 1, 0, 1], [f.segment_position for f in parsed_chained.frames])
        self.assertEqual([False, True, False, True], [f.is_innermost_call for f in parsed_chained.frames])

        ambiguous_mixed = "\n".join([
            "    at bootstrap (app/entry.js:3:1)",
            '  File "app/main.py", line 21, in main',
        ])
        parsed_ambiguous = extract_stack_frames_detailed(ambiguous_mixed)
        self.assertEqual(1, len(parsed_ambiguous.trace_segments))
        amb_seg = parsed_ambiguous.trace_segments[0]
        self.assertEqual("unknown", amb_seg.frame_order)
        self.assertEqual("unknown", amb_seg.order_confidence)
        self.assertIsNone(amb_seg.innermost_call_frame_index)
        for frame in parsed_ambiguous.frames:
            self.assertEqual("unknown", frame.frame_order)
            self.assertIsNone(frame.is_innermost_call)
            self.assertIsNone(frame.is_innermost_mapped)
            self.assertIsNone(frame.is_innermost_valid)
            self.assertIsNone(frame.call_depth_from_innermost)

    def test_build_repair_context_populates_frames_segments_frame_indices_and_keeps_frame_lines_sorted_unique(self):
        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"))
            with open(os.path.join(repo_root, "app", "main.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"line_{i} = {i}" for i in range(1, 35)]))
            graph, profiles = self._graph_and_profiles(repo_root)

            stack_trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/main.py", line 30, in main',
                '  File "app/main.py", line 14, in run',
                '  File "app/main.py", line 14, in <lambda>',
                '  File "app/main.py", line 8, in step',
                '  File "app/main.py", line 30, in main_retry',
                '  File ".venv/lib/python3.12/site-packages/ext/lib.py", line 99, in crash',
                "RuntimeError: execute failed",
            ])
            incident = Incident("inc-p1", "execute failed", stack_trace, "RuntimeError")

            context = build_repair_context(incident, graph, profiles, repo_root)

            # RepairEvidence.frame_lines must remain sorted and unique for backward compatibility
            main_evidence = next(item for item in context.evidence if item.path == "app/main.py")
            service_evidence = next(item for item in context.evidence if item.path == "app/service.py")
            self.assertEqual([8, 14, 30], main_evidence.frame_lines)
            self.assertEqual([0, 1, 2, 3, 4], main_evidence.frame_indices)
            self.assertEqual([], service_evidence.frame_lines)
            self.assertEqual([], service_evidence.frame_indices)

            # RepairContext.frames preserves exact traceback appearance order, duplicates, functions, and lines
            self.assertEqual(6, len(context.frames))
            self.assertEqual(
                [
                    ("app/main.py", 30, "main", "mapped", "app/main.py", False, False),
                    ("app/main.py", 14, "run", "mapped", "app/main.py", False, False),
                    ("app/main.py", 14, "<lambda>", "mapped", "app/main.py", False, False),
                    ("app/main.py", 8, "step", "mapped", "app/main.py", False, False),
                    ("app/main.py", 30, "main_retry", "mapped", "app/main.py", False, True),
                    (".venv/lib/python3.12/site-packages/ext/lib.py", 99, "crash", "unmapped", None, True, False),
                ],
                [
                    (
                        f.reported_path,
                        f.reported_line,
                        f.reported_function,
                        f.mapping.status,
                        f.mapping.mapped_path,
                        f.is_innermost_call,
                        f.is_innermost_mapped,
                    )
                    for f in context.frames
                ],
            )

            # TraceSegment distinguishes innermost_call_frame_index (5) from innermost_mapped_frame_index (4)
            self.assertEqual(1, len(context.trace_segments))
            seg = context.trace_segments[0]
            self.assertEqual(5, seg.innermost_call_frame_index)
            self.assertEqual(4, seg.innermost_mapped_frame_index)
            self.assertEqual(4, seg.innermost_valid_frame_index)

            # Serialization includes frames, trace_segments, and frame_indices
            serialized = context.to_dict()
            self.assertEqual(6, len(serialized["frames"]))
            self.assertEqual(1, len(serialized["trace_segments"]))
            self.assertEqual([0, 1, 2, 3, 4], serialized["evidence"][0]["frame_indices"])
            self.assertEqual([8, 14, 30], serialized["evidence"][0]["frame_lines"])

    def test_resolve_segment_ordering_fails_closed_on_unknown_or_generic_at_frames(self):
        for label, trace_text in (
            (
                "generic_lua_at_frames",
                "\n".join([
                    "    at lua_entry (scripts/handler.lua:11:1)",
                    "    at lua_main (scripts/main.lua:4:1)",
                ]),
            ),
            (
                "generic_swift_at_frames",
                "\n".join([
                    "    at SwiftHandler.handle (Sources/App/Handler.swift:22:1)",
                    "    at SwiftApp.run (Sources/App/App.swift:9:1)",
                ]),
            ),
            (
                "non_standard_python_at_frames",
                "\n".join([
                    "    at run (app/main.py:10:1)",
                    "    at execute (app/service.py:20:1)",
                ]),
            ),
            (
                "mixed_node_and_jvm_at_frames",
                "\n".join([
                    "    at execute (app/service.js:12:3)",
                    "    at com.acme.Runner.run(Runner.java:24)",
                ]),
            ),
            (
                "contradictory_python_header_with_node_at_frames",
                "\n".join([
                    "Traceback (most recent call last):",
                    "    at execute (app/service.js:12:3)",
                ]),
            ),
        ):
            with self.subTest(case=label):
                parsed = extract_stack_frames_detailed(trace_text)
                self.assertEqual(1, len(parsed.trace_segments))
                seg = parsed.trace_segments[0]
                self.assertEqual("unknown", seg.frame_order)
                self.assertEqual("unknown", seg.order_confidence)
                self.assertIsNone(seg.innermost_call_frame_index)
                self.assertIsNone(seg.innermost_mapped_frame_index)
                self.assertIsNone(seg.innermost_valid_frame_index)
                for frame in parsed.frames:
                    self.assertEqual("unknown", frame.frame_order)
                    self.assertIsNone(frame.is_innermost_call)
                    self.assertIsNone(frame.is_innermost_mapped)
                    self.assertIsNone(frame.is_innermost_valid)
                    self.assertIsNone(frame.call_depth_from_innermost)

    def test_extract_stack_frames_backward_compatibility_regression_matrix(self):
        cases = [
            (
                "python_standard_and_duplicates_and_missing_fn",
                "\n".join([
                    "Traceback (most recent call last):",
                    '  File "app/main.py", line 30, in main',
                    "    run()",
                    '  File "app/main.py", line 14, in run',
                    '  File "app/main.py", line 14, in <lambda>',
                    '  File "app/main.py", line 8, in step',
                    '  File "app/main.py", line 8, in step',
                    '  File "app/service.py", line 18',
                    "RuntimeError: boom",
                ]),
                [
                    IncidentFrame("app/main.py", 30, "main"),
                    IncidentFrame("app/main.py", 14, "run"),
                    IncidentFrame("app/main.py", 14, "<lambda>"),
                    IncidentFrame("app/main.py", 8, "step"),
                    IncidentFrame("app/main.py", 8, "step"),
                    IncidentFrame("app/service.py", 18, ""),
                ],
            ),
            (
                "nodejs_v8_named_anonymous_and_file_url",
                "\n".join([
                    "Error: failure",
                    "    at Object.execute [as run] (app/service.js:12:3)",
                    "    at async handleRequest (app/controller.cjs:19:7)",
                    "    at app/anon.js:24:5",
                    "    at file:///Users/dev/My%20App/src/entry.js:9:1",
                ]),
                [
                    IncidentFrame("app/service.js", 12, "Object.execute [as run]"),
                    IncidentFrame("app/controller.cjs", 19, "handleRequest"),
                    IncidentFrame("app/anon.js", 24, ""),
                    IncidentFrame("/Users/dev/My App/src/entry.js", 9, ""),
                ],
            ),
            (
                "java_kotlin_scala_jvm",
                "\n".join([
                    "java.lang.IllegalStateException: broken",
                    "    at app//com.example.service.Runner$Inner.execute(Runner.java:12)",
                    "    at com.example.service.KotlinRunner.run(KotlinRunner.kt:34)",
                    "    at ScalaService.run(ScalaService.scala:31)",
                ]),
                [
                    IncidentFrame("Runner.java", 12, "app//com.example.service.Runner$Inner.execute"),
                    IncidentFrame("KotlinRunner.kt", 34, "com.example.service.KotlinRunner.run"),
                    IncidentFrame("ScalaService.scala", 31, "ScalaService.run"),
                ],
            ),
            (
                "go_calls_generics_created_by_and_orphan_location",
                "\n".join([
                    "goroutine 18 [running]:",
                    "main.Run[int](0x1)",
                    "\t/app/main.go:10 +0x10",
                    "pkg.(*Cache[string, int]).Get(0x2)",
                    "\t/app/cache.go:44 +0x28",
                    "created by main.spawn in goroutine 1",
                    "\t/app/main.go:20 +0x12",
                    "created by main.legacySpawn",
                    "\t/app/main.go:25",
                    "\t/app/orphan.go:9 +0x04",
                ]),
                [
                    IncidentFrame("/app/main.go", 10, "main.Run[int]"),
                    IncidentFrame("/app/cache.go", 44, "pkg.(*Cache[string, int]).Get"),
                    IncidentFrame("/app/main.go", 20, "main.spawn"),
                    IncidentFrame("/app/main.go", 25, "main.legacySpawn"),
                    IncidentFrame("/app/orphan.go", 9, ""),
                ],
            ),
            (
                "rust_panic_and_numbered_backtrace",
                "\n".join([
                    "thread 'main' panicked at 'boom', src/main.rs:27:5",
                    "   0: <app::Service as app::Runner>::run::{{closure}}",
                    "             at src/lib.rs:42:9",
                    "   1: 0x55555555 - crate::engine::process",
                    "             at src/engine.rs:88:13",
                    "             at src/orphan.rs:12:1",
                ]),
                [
                    IncidentFrame("src/main.rs", 27, ""),
                    IncidentFrame("src/lib.rs", 42, "<app::Service as app::Runner>::run::{{closure}}"),
                    IncidentFrame("src/engine.rs", 88, "crate::engine::process"),
                    IncidentFrame("src/orphan.rs", 12, ""),
                ],
            ),
            (
                "ruby_and_php_frames",
                "\n".join([
                    "    from app/workers/job.rb:15:in `perform'",
                    "app/workers/runner.rb:8:in 'run'",
                    '    from app/workers/mailer.rb:21:in "deliver"',
                    "#0 /var/www/app/Http/Controller.php(51): App\\Http\\Controller->index()",
                    "#1 /var/www/public/index.php(14): Kernel->handle(Object(Request))",
                ]),
                [
                    IncidentFrame("app/workers/job.rb", 15, "perform"),
                    IncidentFrame("app/workers/runner.rb", 8, "run"),
                    IncidentFrame("app/workers/mailer.rb", 21, "deliver"),
                    IncidentFrame("/var/www/app/Http/Controller.php", 51, "App\\Http\\Controller->index"),
                    IncidentFrame("/var/www/public/index.php", 14, "Kernel->handle"),
                ],
            ),
            (
                "malformed_and_mixed_interleaved_trace",
                "\n".join([
                    "Traceback (most recent call last):",
                    '  File "app/broken.py", line 0, in zero_line',
                    '  File "app/broken.py", line -3, in neg_line',
                    '  File "   ", line 10, in blank_path',
                    "    at bad_js (app/broken.js:0:2)",
                    "   0: crate::engine::dangling",
                    '  File "app/main.py", line 11, in main',
                    "main.danglingGo(0x1)",
                    "    at app/anon.js:19:4",
                ]),
                [
                    IncidentFrame("app/main.py", 11, "main"),
                    IncidentFrame("app/anon.js", 19, ""),
                ],
            ),
        ]

        for label, stack_trace, expected in cases:
            with self.subTest(case=label):
                actual = extract_stack_frames(stack_trace)
                self.assertEqual(expected, actual)
                self.assertEqual(
                    [d.to_dict() for d in expected],
                    [a.to_dict() for a in actual],
                )

    def test_f10_path_mapping_contract_preserved_in_detailed_frames_and_context(self):
        with tempfile.TemporaryDirectory() as sandbox:
            repo_root = os.path.join(sandbox, "repo")
            outside_dir = os.path.join(sandbox, "outside")
            os.makedirs(os.path.join(repo_root, "app"))
            os.makedirs(os.path.join(repo_root, "billing"))
            os.makedirs(os.path.join(repo_root, "svc_a", "core"))
            os.makedirs(os.path.join(repo_root, "svc_b", "core"))
            os.makedirs(os.path.join(repo_root, "src", "main", "java", "com", "acme", "auth"))
            os.makedirs(os.path.join(repo_root, "src", "main", "java", "com", "acme", "billing"))
            os.makedirs(outside_dir)

            in_repo_files = (
                "app/main.py",
                "app/service.py",
                "billing/service.py",
                "svc_a/core/handler.py",
                "svc_b/core/handler.py",
                "src/main/java/com/acme/auth/Service.java",
                "src/main/java/com/acme/billing/Service.java",
            )
            for rel_path in in_repo_files:
                full_path = os.path.join(repo_root, *rel_path.split("/"))
                with open(full_path, "w", encoding="utf-8") as handle:
                    handle.write("\n".join(["x = 1"] * 20))

            outside_file = os.path.join(outside_dir, "secret.py")
            with open(outside_file, "w", encoding="utf-8") as handle:
                handle.write("secret = 1\n")

            escaping_link = os.path.join(repo_root, "app", "escape.py")
            os.symlink(outside_file, escaping_link)

            intel = RepoIntelligence(
                repo_path=repo_root,
                repo_url="",
                file_analyses=[
                    FileAnalysis("app/main.py", "python", 20, 1, functions=["run"]),
                    FileAnalysis("app/service.py", "python", 20, 1, functions=["execute"]),
                    FileAnalysis("billing/service.py", "python", 20, 1, functions=["execute"]),
                    FileAnalysis("svc_a/core/handler.py", "python", 20, 1, functions=["handle"]),
                    FileAnalysis("svc_b/core/handler.py", "python", 20, 1, functions=["handle"]),
                    FileAnalysis("app/escape.py", "python", 1, 1, functions=["leak"]),
                    FileAnalysis("src/main/java/com/acme/auth/Service.java", "java", 20, 1, classes=["Service"], functions=["login"]),
                    FileAnalysis("src/main/java/com/acme/billing/Service.java", "java", 20, 1, classes=["Service"], functions=["charge"]),
                ],
            )
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel)

            abs_main = os.path.join(repo_root, "app", "main.py")
            stack_trace = "\n".join([
                f'  File "{abs_main}", line 5, in run',
                '  File "app\\main.py", line 9, in run',
                '  File "../app/main.py", line 3, in traversal',
                f'  File "{escaping_link}", line 1, in escaped_abs',
                '  File "app/escape.py", line 1, in escaped_rel',
                '  File "service.py", line 4, in dup_basename',
                '  File "core/handler.py", line 6, in ambig_suffix',
                '  File "C:\\other_repo\\app\\main.py", line 8, in win_abs',
                "    at com.acme.auth.Service.login(Service.java:12)",
                "    at org.external.Service.login(Service.java:15)",
            ])
            incident = Incident("inc-f10", "security regression check", stack_trace, "RuntimeError")

            context = build_repair_context(incident, graph, profiles, repo_root)

            evidence_by_path = {item.path: item for item in context.evidence}
            self.assertEqual([5, 9], evidence_by_path["app/main.py"].frame_lines)
            self.assertEqual([0, 1], evidence_by_path["app/main.py"].frame_indices)
            self.assertEqual(
                [12],
                evidence_by_path["src/main/java/com/acme/auth/Service.java"].frame_lines,
            )
            self.assertEqual(
                [8],
                evidence_by_path["src/main/java/com/acme/auth/Service.java"].frame_indices,
            )

            self.assertEqual(
                [
                    ("mapped", "app/main.py", "resolved_in_repo"),
                    ("mapped", "app/main.py", "exact_relpath"),
                    ("unmapped", None, "unmapped_external"),
                    ("unmapped", None, "unmapped_external"),
                    ("unmapped", None, "unmapped_escape"),
                    ("unmapped", None, "unmapped_not_found"),
                    ("unmapped", None, "unmapped_ambiguous"),
                    ("unmapped", None, "unmapped_external"),
                    ("mapped", "src/main/java/com/acme/auth/Service.java", "jvm_qualified"),
                    ("unmapped", None, "unmapped_not_found"),
                ],
                [
                    (f.mapping.status, f.mapping.mapped_path, f.mapping.match_strategy)
                    for f in context.frames
                ],
            )
            self.assertEqual(
                [
                    IncidentFrame("../app/main.py", 3, "traversal"),
                    IncidentFrame(escaping_link, 1, "escaped_abs"),
                    IncidentFrame("app/escape.py", 1, "escaped_rel"),
                    IncidentFrame("service.py", 4, "dup_basename"),
                    IncidentFrame("core/handler.py", 6, "ambig_suffix"),
                    IncidentFrame(r"C:\other_repo\app\main.py", 8, "win_abs"),
                    IncidentFrame("Service.java", 15, "org.external.Service.login"),
                ],
                context.unmapped_frames,
            )

    def test_p2_valid_and_invalid_frame_lines_multiple_and_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"))
            with open(os.path.join(repo_root, "app", "main.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"line_{i} = {i}" for i in range(1, 21)]) + "\n")
            with open(os.path.join(repo_root, "app", "service.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"svc_{i} = {i}" for i in range(1, 11)]) + "\n")

            graph, profiles = self._graph_and_profiles(repo_root)
            stack_trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/main.py", line 1, in first_line_boundary',
                '  File "app/main.py", line 18, in main',
                '  File "app/main.py", line 10, in run',
                '  File "app/main.py", line 10, in <lambda>',
                '  File "app/main.py", line 4, in step',
                '  File "app/main.py", line 20, in last_line_boundary',
                '  File "app/main.py", line 18, in main_retry',
                '  File "app/main.py", line 21, in one_past_eof',
                '  File "app/main.py", line 25, in stale_line',
                '  File "app/main.py", line 25, in stale_dup',
                '  File "app/main.py", line 99, in out_of_range_99',
                '  File "app/main.py", line 10000, in far_past_eof',
                '  File ".venv/lib/python3.12/site-packages/ext/lib.py", line 77, in crash',
                "RuntimeError: execute failed",
            ])
            incident = Incident("inc-p2-lines", "execute failed", stack_trace, "RuntimeError")

            context = build_repair_context(incident, graph, profiles, repo_root)
            evidence_by_path = {item.path: item for item in context.evidence}
            main_ev = evidence_by_path["app/main.py"]

            # RepairEvidence.frame_lines contains ONLY valid frame lines, sorted and unique
            self.assertEqual([1, 4, 10, 18, 20], main_ev.frame_lines)
            self.assertEqual([1, 4, 10, 18, 20], main_ev.valid_frame_lines)
            self.assertEqual([21, 25, 99, 10000], main_ev.invalid_frame_lines)
            self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], main_ev.frame_indices)
            self.assertEqual("ok", main_ev.file_status)
            self.assertEqual(20, main_ev.total_lines)
            self.assertEqual("lf", main_ev.line_ending_style)
            self.assertTrue(main_ev.patch_eligible)
            self.assertEqual("eligible_valid_stack_frame", main_ev.patch_eligibility_reason)
            self.assertIn("1: line_1 = 1", main_ev.excerpt)

            # Segment distinguishes innermost_call (12), innermost_mapped (11), and innermost_valid (6)
            self.assertEqual(1, len(context.trace_segments))
            seg = context.trace_segments[0]
            self.assertEqual(12, seg.innermost_call_frame_index)
            self.assertEqual(11, seg.innermost_mapped_frame_index)
            self.assertEqual(6, seg.innermost_valid_frame_index)

            self.assertEqual(
                [
                    (0, 1, "ok", True, 20, True, False, False, False),
                    (1, 18, "ok", True, 20, True, False, False, False),
                    (2, 10, "ok", True, 20, True, False, False, False),
                    (3, 10, "ok", True, 20, True, False, False, False),
                    (4, 4, "ok", True, 20, True, False, False, False),
                    (5, 20, "ok", True, 20, True, False, False, False),
                    (6, 18, "ok", True, 20, True, False, False, True),
                    (7, 21, "line_out_of_range", True, 20, False, False, False, False),
                    (8, 25, "line_out_of_range", True, 20, False, False, False, False),
                    (9, 25, "line_out_of_range", True, 20, False, False, False, False),
                    (10, 99, "line_out_of_range", True, 20, False, False, False, False),
                    (11, 10000, "line_out_of_range", True, 20, False, False, True, False),
                    (12, 77, "unmapped", None, None, None, True, False, False),
                ],
                [
                    (
                        f.frame_index,
                        f.reported_line,
                        f.validation.status,
                        f.validation.file_exists,
                        f.validation.total_lines,
                        f.validation.line_in_range,
                        f.is_innermost_call,
                        f.is_innermost_mapped,
                        f.is_innermost_valid,
                    )
                    for f in context.frames
                ],
            )

            # Explicit boundary matrix on the 20-line file for lines 1, 20, 21, 0, -1, -10, 99, and 10000
            inspected_main = _inspect_repository_file(repo_root, "app/main.py")
            boundary_expectations = [
                (
                    1,
                    FrameValidation(
                        status="ok",
                        file_exists=True,
                        total_lines=20,
                        line_in_range=True,
                        detail=None,
                    ),
                ),
                (
                    20,
                    FrameValidation(
                        status="ok",
                        file_exists=True,
                        total_lines=20,
                        line_in_range=True,
                        detail=None,
                    ),
                ),
                (
                    21,
                    FrameValidation(
                        status="line_out_of_range",
                        file_exists=True,
                        total_lines=20,
                        line_in_range=False,
                        detail="reported line 21 exceeds file line count 20",
                    ),
                ),
                (
                    0,
                    FrameValidation(
                        status="line_non_positive",
                        file_exists=True,
                        total_lines=20,
                        line_in_range=False,
                        detail="reported line 0 must be >= 1",
                    ),
                ),
                (
                    -1,
                    FrameValidation(
                        status="line_non_positive",
                        file_exists=True,
                        total_lines=20,
                        line_in_range=False,
                        detail="reported line -1 must be >= 1",
                    ),
                ),
                (
                    -10,
                    FrameValidation(
                        status="line_non_positive",
                        file_exists=True,
                        total_lines=20,
                        line_in_range=False,
                        detail="reported line -10 must be >= 1",
                    ),
                ),
                (
                    99,
                    FrameValidation(
                        status="line_out_of_range",
                        file_exists=True,
                        total_lines=20,
                        line_in_range=False,
                        detail="reported line 99 exceeds file line count 20",
                    ),
                ),
                (
                    10000,
                    FrameValidation(
                        status="line_out_of_range",
                        file_exists=True,
                        total_lines=20,
                        line_in_range=False,
                        detail="reported line 10000 exceeds file line count 20",
                    ),
                ),
            ]
            for line_num, expected_validation in boundary_expectations:
                with self.subTest(line=line_num):
                    self.assertEqual(
                        expected_validation,
                        _validate_mapped_frame(line_num, inspected_main),
                    )

    def test_p2_missing_unreadable_binary_non_utf8_and_empty_files(self):
        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"))
            # app/missing.py is intentionally NOT created on disk
            with open(os.path.join(repo_root, "app", "empty.py"), "wb") as handle:
                handle.write(b"")
            with open(os.path.join(repo_root, "app", "binary.py"), "wb") as handle:
                handle.write(b"x = 1\x00\ny = 2\n")
            with open(os.path.join(repo_root, "app", "bad_utf8.py"), "wb") as handle:
                handle.write(b"\xff\xfe\xfd\n")
            os.makedirs(os.path.join(repo_root, "app", "dir_as_file.py"))

            intel = RepoIntelligence(
                repo_path=repo_root,
                repo_url="",
                file_analyses=[
                    FileAnalysis("app/missing.py", "python", 10, 1, functions=["missing_fn"]),
                    FileAnalysis("app/empty.py", "python", 0, 1, functions=["empty_fn"]),
                    FileAnalysis("app/binary.py", "python", 2, 1, functions=["bin_fn"]),
                    FileAnalysis("app/bad_utf8.py", "python", 1, 1, functions=["utf8_fn"]),
                    FileAnalysis("app/dir_as_file.py", "python", 1, 1, functions=["dir_fn"]),
                ],
            )
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel)

            stack_trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/missing.py", line 5, in missing_fn',
                '  File "app/empty.py", line 1, in empty_fn',
                '  File "app/binary.py", line 1, in bin_fn',
                '  File "app/bad_utf8.py", line 1, in utf8_fn',
                '  File "app/dir_as_file.py", line 1, in dir_fn',
                "RuntimeError: broken files",
            ])
            incident = Incident("inc-p2-files", "broken files", stack_trace, "RuntimeError")

            context = build_repair_context(incident, graph, profiles, repo_root)
            evidence_by_path = {item.path: item for item in context.evidence}

            self.assertEqual(
                [
                    ("app/missing.py", "file_missing", False, None, None),
                    ("app/empty.py", "line_out_of_range", True, 0, False),
                    ("app/binary.py", "file_unreadable", True, None, None),
                    ("app/bad_utf8.py", "file_unreadable", True, None, None),
                    ("app/dir_as_file.py", "file_unreadable", False, None, None),
                ],
                [
                    (
                        f.mapping.mapped_path,
                        f.validation.status,
                        f.validation.file_exists,
                        f.validation.total_lines,
                        f.validation.line_in_range,
                    )
                    for f in context.frames
                ],
            )

            self.assertIsNone(context.trace_segments[0].innermost_valid_frame_index)
            self.assertTrue(all(f.is_innermost_valid is False for f in context.frames))

            missing_ev = evidence_by_path["app/missing.py"]
            self.assertEqual("file_missing", missing_ev.file_status)
            self.assertIsNone(missing_ev.total_lines)
            self.assertEqual([], missing_ev.frame_lines)
            self.assertEqual([], missing_ev.valid_frame_lines)
            self.assertEqual([5], missing_ev.invalid_frame_lines)
            self.assertEqual("", missing_ev.excerpt)
            self.assertFalse(missing_ev.patch_eligible)
            self.assertEqual("ineligible_file_not_readable", missing_ev.patch_eligibility_reason)

            empty_ev = evidence_by_path["app/empty.py"]
            self.assertEqual("ok", empty_ev.file_status)
            self.assertEqual(0, empty_ev.total_lines)
            self.assertEqual("none", empty_ev.line_ending_style)
            self.assertEqual([], empty_ev.frame_lines)
            self.assertEqual([], empty_ev.valid_frame_lines)
            self.assertEqual([1], empty_ev.invalid_frame_lines)
            self.assertEqual("", empty_ev.excerpt)
            self.assertFalse(empty_ev.patch_eligible)
            self.assertEqual("ineligible_no_valid_frames", empty_ev.patch_eligibility_reason)

            for unreadable_path in ("app/binary.py", "app/bad_utf8.py", "app/dir_as_file.py"):
                ev = evidence_by_path[unreadable_path]
                self.assertEqual("file_unreadable", ev.file_status)
                self.assertIsNone(ev.total_lines)
                self.assertEqual([], ev.frame_lines)
                self.assertEqual([], ev.valid_frame_lines)
                self.assertEqual([1], ev.invalid_frame_lines)
                self.assertEqual("", ev.excerpt)
                self.assertFalse(ev.patch_eligible)
                self.assertEqual("ineligible_file_not_readable", ev.patch_eligibility_reason)

            # Empty file reached as a graph neighbor of a valid frame file must NEVER become patch-eligible
            with open(os.path.join(repo_root, "app", "valid_origin.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"v_{i} = {i}" for i in range(1, 11)]) + "\n")
            with open(os.path.join(repo_root, "app", "empty_neighbor.py"), "wb") as handle:
                handle.write(b"")

            intel_neighbor = RepoIntelligence(
                repo_path=repo_root,
                repo_url="",
                file_analyses=[
                    FileAnalysis(
                        "app/valid_origin.py",
                        "python",
                        10,
                        1,
                        functions=["run"],
                        imports=["from app.empty_neighbor import helper"],
                    ),
                    FileAnalysis("app/empty_neighbor.py", "python", 0, 1, functions=["helper"]),
                ],
            )
            graph_neighbor = build_dependency_graph(intel_neighbor)
            profiles_neighbor = build_file_profiles(graph_neighbor, intel_neighbor)
            neighbor_incident = Incident(
                "inc-empty-neighbor",
                "helper failed",
                '  File "app/valid_origin.py", line 5, in run',
                "RuntimeError",
            )
            neighbor_ctx = build_repair_context(
                neighbor_incident,
                graph_neighbor,
                profiles_neighbor,
                repo_root,
            )
            neighbor_ev = {item.path: item for item in neighbor_ctx.evidence}
            self.assertTrue(neighbor_ev["app/valid_origin.py"].patch_eligible)
            self.assertEqual("ok", neighbor_ev["app/empty_neighbor.py"].file_status)
            self.assertEqual(0, neighbor_ev["app/empty_neighbor.py"].total_lines)
            self.assertFalse(neighbor_ev["app/empty_neighbor.py"].patch_eligible)
            self.assertEqual(
                "ineligible_empty_file",
                neighbor_ev["app/empty_neighbor.py"].patch_eligibility_reason,
            )

    def test_p2_strict_source_line_splitting_and_line_ending_styles(self):
        self.assertEqual(([], "none"), _split_source_lines(""))
        self.assertEqual((["alpha"], "none"), _split_source_lines("alpha"))
        self.assertEqual((["alpha", "beta"], "lf"), _split_source_lines("alpha\nbeta\n"))
        self.assertEqual((["alpha", "beta"], "crlf"), _split_source_lines("alpha\r\nbeta\r\n"))
        self.assertEqual((["alpha", "beta"], "cr"), _split_source_lines("alpha\rbeta\r"))
        self.assertEqual((["alpha", "beta", "gamma"], "mixed"), _split_source_lines("alpha\r\nbeta\ngamma\r"))

        # F03-09: \x0b, \x0c, \x85, \u2028, \u2029 must NOT split lines
        non_break_text = "line1_a\x0cline1_b\x0bline1_c\nline2_a\x85line2_b\u2028line2_c\u2029line2_d\nline3\n"
        lines, style = _split_source_lines(non_break_text)
        self.assertEqual("lf", style)
        self.assertEqual(3, len(lines))

        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"))
            with open(os.path.join(repo_root, "app", "main.py"), "wb") as handle:
                handle.write(non_break_text.encode("utf-8"))
            with open(os.path.join(repo_root, "app", "service.py"), "wb") as handle:
                handle.write(b"svc_1 = 1\r\nsvc_2 = 2\r\n")

            graph, profiles = self._graph_and_profiles(repo_root)
            stack_trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/main.py", line 3, in run',
                '  File "app/main.py", line 4, in phantom_line_after_form_feed',
                '  File "app/service.py", line 2, in execute',
                "RuntimeError: execute failed",
            ])
            incident = Incident("inc-p2-split", "execute failed", stack_trace, "RuntimeError")
            context = build_repair_context(incident, graph, profiles, repo_root)
            evidence_by_path = {item.path: item for item in context.evidence}

            self.assertEqual(3, evidence_by_path["app/main.py"].total_lines)
            self.assertEqual([3], evidence_by_path["app/main.py"].frame_lines)
            self.assertEqual([3], evidence_by_path["app/main.py"].valid_frame_lines)
            self.assertEqual([4], evidence_by_path["app/main.py"].invalid_frame_lines)
            self.assertEqual("lf", evidence_by_path["app/main.py"].line_ending_style)
            self.assertEqual("crlf", evidence_by_path["app/service.py"].line_ending_style)
            self.assertEqual(2, evidence_by_path["app/service.py"].total_lines)

    def test_p2_symbol_index_verification_match_mismatch_container_anonymous_and_unknown(self):
        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"))
            with open(os.path.join(repo_root, "app", "main.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"line_{i} = {i}" for i in range(1, 31)]) + "\n")
            with open(os.path.join(repo_root, "app", "service.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"svc_{i} = {i}" for i in range(1, 21)]) + "\n")
            with open(os.path.join(repo_root, "app", "other.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"oth_{i} = {i}" for i in range(1, 11)]) + "\n")

            graph, profiles = self._graph_and_profiles(repo_root)
            stack_trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/main.py", line 2, in <module>',
                '  File "app/main.py", line 12, in execute',
                '  File "app/main.py", line 13, in Service',
                '  File "app/main.py", line 14, in <lambda>',
                '  File "app/main.py", line 15, in wrong_fn',
                '  File "app/main.py", line 16',
                '  File "app/main.py", line 99, in execute',
                '  File "app/service.py", line 5, in execute',
                '  File "app/other.py", line 3, in other',
                '  File "external/pkg.py", line 4, in ext_fn',
                "RuntimeError: execute failed",
            ])
            incident = Incident("inc-p2-sym", "execute failed", stack_trace, "RuntimeError")

            # 1. When symbol_index is None, all frames remain "unknown" and evidence is "unavailable"
            ctx_no_index = build_repair_context(incident, graph, profiles, repo_root)
            for frame in ctx_no_index.frames:
                self.assertEqual("unknown", frame.symbol_match.status)
                self.assertIsNone(frame.symbol_match.enclosing_symbol)
            for ev in ctx_no_index.evidence:
                self.assertEqual("unavailable", ev.symbol_verification)

            # 2. When symbol_index is provided for app/main.py (and None for app/other.py, omitted for app/service.py)
            symbol_index = {
                "app/main.py": [
                    SourceSymbolInfo(
                        name="Service",
                        qualified_name="Service",
                        kind="class",
                        start_line=8,
                        end_line=25,
                        signature="class Service:",
                    ),
                    SourceSymbolInfo(
                        name="execute",
                        qualified_name="Service.execute",
                        kind="method",
                        start_line=10,
                        end_line=20,
                        signature="def execute(self):",
                    ),
                ],
                "app/other.py": None,
            }
            ctx_with_index = build_repair_context(
                incident,
                graph,
                profiles,
                repo_root,
                symbol_index=symbol_index,
            )
            ev_by_path = {item.path: item for item in ctx_with_index.evidence}
            # Because app/main.py has mixed match and mismatch/unverified/invalid frames, file-level symbol_verification must NOT be "verified"
            self.assertEqual("unavailable", ev_by_path["app/main.py"].symbol_verification)
            self.assertEqual("not_indexed", ev_by_path["app/service.py"].symbol_verification)
            self.assertEqual("not_indexed", ev_by_path["app/other.py"].symbol_verification)

            self.assertEqual(
                [
                    (0, "anonymous_or_module", "<module>", None),
                    (1, "exact", "execute", "Service.execute"),
                    (2, "container_match", "Service", "Service.execute"),
                    (3, "anonymous_or_module", "<lambda>", "Service.execute"),
                    (4, "mismatch", "wrong_fn", "Service.execute"),
                    (5, "unverified", None, "Service.execute"),
                    (6, "unknown", "execute", None),
                    (7, "unknown", "execute", None),
                    (8, "unknown", "other", None),
                    (9, "unknown", "ext_fn", None),
                ],
                [
                    (
                        f.frame_index,
                        f.symbol_match.status,
                        f.symbol_match.normalized_reported_function,
                        (
                            f.symbol_match.enclosing_symbol.qualified_name
                            if f.symbol_match.enclosing_symbol is not None
                            else None
                        ),
                    )
                    for f in ctx_with_index.frames
                ],
            )
            self.assertIsNotNone(ctx_with_index.frames[4].symbol_match.detail)

            # 3. Only when a file has mapped frames, ALL mapped frames are valid ("ok"), and ALL have accepted symbol statuses
            # ("exact", "container_match", or "anonymous_or_module") does file-level symbol_verification become "verified"
            all_verified_trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/main.py", line 2, in <module>',
                '  File "app/main.py", line 12, in execute',
                '  File "app/main.py", line 13, in Service',
                '  File "app/main.py", line 14, in <lambda>',
                "RuntimeError: execute failed",
            ])
            symbol_index_with_non_frame_service = {
                **symbol_index,
                "app/service.py": [
                    SourceSymbolInfo(
                        name="execute",
                        qualified_name="execute",
                        kind="function",
                        start_line=1,
                        end_line=10,
                        signature="def execute():",
                    )
                ],
            }
            ctx_all_verified = build_repair_context(
                Incident("inc-p2-sym-ok", "execute failed", all_verified_trace, "RuntimeError"),
                graph,
                profiles,
                repo_root,
                symbol_index=symbol_index_with_non_frame_service,
            )
            ev_all_verified = {item.path: item for item in ctx_all_verified.evidence}
            self.assertEqual("verified", ev_all_verified["app/main.py"].symbol_verification)
            # app/service.py is in symbol_index, but has no mapped frames in this trace, so it must not be marked "verified"
            self.assertEqual("unavailable", ev_all_verified["app/service.py"].symbol_verification)

            # 4. A single mismatch or out-of-range frame alongside an exact match prevents file-level "verified"
            for bad_case, bad_line in (
                ("mixed_exact_and_mismatch", '  File "app/main.py", line 15, in wrong_fn'),
                ("mixed_exact_and_unverified", '  File "app/main.py", line 16'),
                ("mixed_exact_and_out_of_range", '  File "app/main.py", line 99, in execute'),
            ):
                with self.subTest(case=bad_case):
                    mixed_trace = "\n".join([
                        "Traceback (most recent call last):",
                        '  File "app/main.py", line 12, in execute',
                        bad_line,
                        "RuntimeError: execute failed",
                    ])
                    ctx_mixed = build_repair_context(
                        Incident("inc-p2-mixed", "execute failed", mixed_trace, "RuntimeError"),
                        graph,
                        profiles,
                        repo_root,
                        symbol_index=symbol_index,
                    )
                    ev_mixed = {item.path: item for item in ctx_mixed.evidence}
                    self.assertEqual("unavailable", ev_mixed["app/main.py"].symbol_verification)

    def test_p2_frame_only_evidence_with_no_valid_frame_lines_is_not_patch_eligible(self):
        # Default RepairEvidence fails closed
        default_ev = RepairEvidence(path="app/main.py", score=100.0)
        self.assertFalse(default_ev.patch_eligible)
        self.assertEqual("ineligible_unverified", default_ev.patch_eligibility_reason)
        self.assertIsNone(default_ev.file_status)

        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"))
            with open(os.path.join(repo_root, "app", "main.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"line_{i} = {i}" for i in range(1, 11)]) + "\n")
            with open(os.path.join(repo_root, "app", "service.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"svc_{i} = {i}" for i in range(1, 9)]) + "\n")
            with open(os.path.join(repo_root, "app", "repo.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"repo_{i} = {i}" for i in range(1, 7)]) + "\n")
            with open(os.path.join(repo_root, "app", "other.py"), "w", encoding="utf-8") as handle:
                handle.write("other = 1\n")

            intel = RepoIntelligence(
                repo_path=repo_root,
                repo_url="",
                file_analyses=[
                    FileAnalysis(
                        "app/main.py",
                        "python",
                        10,
                        1,
                        functions=["run"],
                        imports=["from app.service import execute"],
                    ),
                    FileAnalysis(
                        "app/service.py",
                        "python",
                        8,
                        1,
                        functions=["execute"],
                        imports=["from app.repo import fetch"],
                    ),
                    FileAnalysis("app/repo.py", "python", 6, 1, functions=["fetch"]),
                    FileAnalysis("app/other.py", "python", 1, 1, functions=["other"]),
                ],
            )
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel)

            # Case 1: Frame-only evidence for app/main.py has NO valid frame lines (lines 50 and 80 > 10).
            # Neither app/main.py, direct neighbor app/service.py (distance 1), nor multi-hop neighbor app/repo.py (distance 2)
            # may become patch-eligible.
            invalid_trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/main.py", line 50, in run',
                '  File "app/main.py", line 80, in crash',
                "RuntimeError: other failure",
            ])
            invalid_incident = Incident("inc-invalid-only", "other failure", invalid_trace, "RuntimeError")
            invalid_ctx = build_repair_context(invalid_incident, graph, profiles, repo_root)
            invalid_ev_by_path = {item.path: item for item in invalid_ctx.evidence}

            main_invalid_ev = invalid_ev_by_path["app/main.py"]
            self.assertEqual("ok", main_invalid_ev.file_status)
            self.assertEqual(10, main_invalid_ev.total_lines)
            self.assertEqual([], main_invalid_ev.frame_lines)
            self.assertEqual([], main_invalid_ev.valid_frame_lines)
            self.assertEqual([50, 80], main_invalid_ev.invalid_frame_lines)
            self.assertEqual([0, 1], main_invalid_ev.frame_indices)
            self.assertEqual("", main_invalid_ev.excerpt)
            self.assertFalse(main_invalid_ev.patch_eligible)
            self.assertEqual("ineligible_no_valid_frames", main_invalid_ev.patch_eligibility_reason)

            # Direct neighbor (distance 1) and multi-hop neighbor (distance 2) of invalid-only frame file are both ineligible
            self.assertFalse(invalid_ev_by_path["app/service.py"].patch_eligible)
            self.assertFalse(invalid_ev_by_path["app/repo.py"].patch_eligible)

            # Case 2: Valid frame line on app/main.py makes app/main.py and readable direct/multi-hop neighbors eligible,
            # while query-only app/other.py remains ineligible
            valid_trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/main.py", line 7, in run',
                "RuntimeError: other failure",
            ])
            valid_incident = Incident("inc-valid", "other failure", valid_trace, "RuntimeError")
            valid_ctx = build_repair_context(valid_incident, graph, profiles, repo_root)
            valid_ev_by_path = {item.path: item for item in valid_ctx.evidence}

            self.assertTrue(valid_ev_by_path["app/main.py"].patch_eligible)
            self.assertEqual(
                "eligible_valid_stack_frame",
                valid_ev_by_path["app/main.py"].patch_eligibility_reason,
            )
            self.assertTrue(valid_ev_by_path["app/service.py"].patch_eligible)
            self.assertEqual(
                "eligible_direct_graph_neighbor",
                valid_ev_by_path["app/service.py"].patch_eligibility_reason,
            )
            self.assertTrue(valid_ev_by_path["app/repo.py"].patch_eligible)
            self.assertEqual(
                "eligible_direct_graph_neighbor",
                valid_ev_by_path["app/repo.py"].patch_eligibility_reason,
            )
            self.assertFalse(valid_ev_by_path["app/other.py"].patch_eligible)
            self.assertEqual(
                "ineligible_query_match_only",
                valid_ev_by_path["app/other.py"].patch_eligibility_reason,
            )

            # Case 3: A file with mapped frames but NO valid frame lines (app/main.py at line 50) that is also a
            # direct or multi-hop graph neighbor of a valid frame file (app/repo.py at line 3) must STILL remain ineligible!
            neighbor_of_valid_trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/main.py", line 50, in stale_caller',
                '  File "app/service.py", line 40, in stale_middle',
                '  File "app/repo.py", line 3, in fetch',
                "RuntimeError: fetch failed",
            ])
            neighbor_of_valid_ctx = build_repair_context(
                Incident("inc-invalid-neighbor", "fetch failed", neighbor_of_valid_trace, "RuntimeError"),
                graph,
                profiles,
                repo_root,
            )
            neighbor_of_valid_ev = {item.path: item for item in neighbor_of_valid_ctx.evidence}
            # app/repo.py has a valid frame at line 3 -> eligible
            self.assertTrue(neighbor_of_valid_ev["app/repo.py"].patch_eligible)
            self.assertEqual(
                "eligible_valid_stack_frame",
                neighbor_of_valid_ev["app/repo.py"].patch_eligibility_reason,
            )
            # app/service.py (distance 1 from app/repo.py) and app/main.py (distance 2 from app/repo.py)
            # both had mapped frames with 0 valid frame lines -> BOTH must be rejected with "ineligible_no_valid_frames"
            self.assertFalse(neighbor_of_valid_ev["app/service.py"].patch_eligible)
            self.assertEqual(
                "ineligible_no_valid_frames",
                neighbor_of_valid_ev["app/service.py"].patch_eligibility_reason,
            )
            self.assertFalse(neighbor_of_valid_ev["app/main.py"].patch_eligible)
            self.assertEqual(
                "ineligible_no_valid_frames",
                neighbor_of_valid_ev["app/main.py"].patch_eligibility_reason,
            )

    def test_p3_multiple_frames_overlapping_and_adjacent_window_merging(self):
        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"), exist_ok=True)
            with open(os.path.join(repo_root, "app", "service.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"svc_line_{i} = {i}" for i in range(1, 31)]) + "\n")

            intel = RepoIntelligence(
                repo_path=repo_root,
                repo_url="",
                file_analyses=[
                    FileAnalysis("app/service.py", "python", 30, 1, functions=["first", "second", "third"]),
                ],
            )
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel)

            # With excerpt_radius=2:
            # - frame 0 at line 4 -> window [2, 6]
            # - frame 1 at line 6 -> window [4, 8] (overlaps [2, 6])
            # - frame 2 at line 11 -> window [9, 13] (adjacent to [2, 8] since 9 == 8 + 1)
            # - frame 3 at line 4 (duplicate valid frame at line 4)
            # - frame 4 at line 99 (invalid out-of-range frame; must be ignored by excerpt generator)
            # All valid windows [2, 6], [4, 8], [9, 13] must merge into a single window [2, 13]
            # preserving target_lines=[4, 6, 11] and frame_indices=[0, 1, 2, 3].
            trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/service.py", line 4, in first',
                '  File "app/service.py", line 6, in second',
                '  File "app/service.py", line 11, in third',
                '  File "app/service.py", line 4, in first',
                '  File "app/service.py", line 99, in stale_frame',
                "ValueError: merge test",
            ])
            symbol_index = {
                "app/service.py": [
                    SourceSymbolInfo("first", "first", "function", 1, 5),
                    SourceSymbolInfo("second", "second", "function", 6, 8),
                    SourceSymbolInfo("third", "Service.third", "function", 9, 15),
                ]
            }
            context = build_repair_context(
                Incident("inc-p3-merge", "merge test", trace, "ValueError"),
                graph,
                profiles,
                repo_root,
                symbol_index=symbol_index,
                excerpt_radius=2,
            )

            self.assertEqual(1, len(context.evidence))
            ev = context.evidence[0]
            self.assertEqual("app/service.py", ev.path)
            self.assertEqual([4, 6, 11], ev.frame_lines)
            self.assertEqual([4, 6, 11], ev.valid_frame_lines)
            self.assertEqual([99], ev.invalid_frame_lines)
            self.assertEqual(1, len(ev.excerpts))

            merged = ev.excerpts[0]
            self.assertEqual("app/service.py#1", merged.excerpt_id)
            self.assertEqual("frame", merged.role)
            self.assertEqual(2, merged.start_line)
            self.assertEqual(13, merged.end_line)
            self.assertEqual([4, 6, 11], merged.target_lines)
            self.assertEqual([0, 1, 2, 3], merged.frame_indices)
            self.assertEqual(2, merged.effective_radius)
            self.assertTrue(merged.clipped_top)
            self.assertTrue(merged.clipped_bottom)
            self.assertFalse(merged.context_trimmed)
            self.assertEqual("first", merged.enclosing_symbol)
            self.assertEqual("app/service.py#1", ev.primary_excerpt_id)
            self.assertEqual(merged.content, ev.excerpt)
            self.assertEqual([], ev.omitted_excerpt_ids)

            rendered_lines = merged.content.splitlines()
            self.assertEqual(12, len(rendered_lines))
            for offset, line_no in enumerate(range(2, 14)):
                line_text = rendered_lines[offset]
                if line_no in {4, 6, 11}:
                    self.assertTrue(
                        line_text.startswith(f"> {line_no:>4}: svc_line_{line_no} = {line_no}"),
                        f"expected target marker on line {line_no}, got {line_text!r}",
                    )
                else:
                    self.assertTrue(
                        line_text.startswith(f"  {line_no:>4}: svc_line_{line_no} = {line_no}"),
                        f"expected context marker on line {line_no}, got {line_text!r}",
                    )

            self.assertIsNotNone(context.truncation)
            self.assertFalse(context.truncation.truncated)
            self.assertEqual(12000, context.truncation.limit_chars)
            self.assertEqual(len(merged.content), context.truncation.actual_chars)
            self.assertEqual([], context.truncation.context_trimmed_excerpt_ids)
            self.assertEqual([], context.truncation.omitted_excerpt_ids)
            self.assertFalse(context.truncation.target_lines_omitted)
            self.assertEqual([], context.truncation.omitted_target_lines)

    def test_p3_distant_frames_multi_window_ids_and_primary_excerpt_selection(self):
        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"), exist_ok=True)
            with open(os.path.join(repo_root, "app", "worker.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"py_line_{i} = {i}" for i in range(1, 61)]) + "\n")
            with open(os.path.join(repo_root, "app", "worker.js"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"const js_line_{i} = {i};" for i in range(1, 61)]) + "\n")
            with open(os.path.join(repo_root, "app", "script.sh"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"echo sh_line_{i}" for i in range(1, 61)]) + "\n")

            intel = RepoIntelligence(
                repo_path=repo_root,
                repo_url="",
                file_analyses=[
                    FileAnalysis("app/worker.py", "python", 60, 1, functions=["entry", "middle", "leaf"]),
                    FileAnalysis("app/worker.js", "javascript", 60, 1, functions=["leafJs", "entryJs"]),
                    FileAnalysis("app/script.sh", "shell", 60, 1, functions=["stepA", "stepB"]),
                ],
            )
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel)

            # 1. Python (outer_to_inner): frames at lines 5, 30, 55 with radius=3
            # Disjoint windows: #1 [2, 8], #2 [27, 33], #3 [52, 58].
            # Innermost valid frame is the last one (line 55 -> app/worker.py#3).
            py_trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/worker.py", line 5, in entry',
                '  File "app/worker.py", line 30, in middle',
                '  File "app/worker.py", line 55, in leaf',
                "RuntimeError: leaf failed",
            ])
            py_ctx = build_repair_context(
                Incident("inc-p3-py", "leaf failed", py_trace, "RuntimeError"),
                graph,
                profiles,
                repo_root,
                excerpt_radius=3,
            )
            py_ev = {item.path: item for item in py_ctx.evidence}["app/worker.py"]
            self.assertEqual(
                ["app/worker.py#1", "app/worker.py#2", "app/worker.py#3"],
                [ex.excerpt_id for ex in py_ev.excerpts],
            )
            self.assertEqual(
                [(2, 8, [5], [0]), (27, 33, [30], [1]), (52, 58, [55], [2])],
                [
                    (ex.start_line, ex.end_line, ex.target_lines, ex.frame_indices)
                    for ex in py_ev.excerpts
                ],
            )
            self.assertEqual("app/worker.py#3", py_ev.primary_excerpt_id)
            self.assertEqual(py_ev.excerpts[2].content, py_ev.excerpt)
            self.assertIn(">   55: py_line_55 = 55", py_ev.excerpt)

            # 2. Node.js (inner_to_outer): frame 0 at line 12 (innermost), frame 1 at line 48 (outer caller)
            # Disjoint windows: #1 [9, 15] (line 12), #2 [45, 51] (line 48).
            # Innermost valid frame is frame 0 (line 12 -> app/worker.js#1).
            js_trace = "\n".join([
                "TypeError: js boom",
                "    at leafJs (app/worker.js:12:5)",
                "    at entryJs (app/worker.js:48:9)",
            ])
            js_ctx = build_repair_context(
                Incident("inc-p3-js", "js boom", js_trace, "TypeError"),
                graph,
                profiles,
                repo_root,
                excerpt_radius=3,
            )
            js_ev = {item.path: item for item in js_ctx.evidence}["app/worker.js"]
            self.assertEqual(
                ["app/worker.js#1", "app/worker.js#2"],
                [ex.excerpt_id for ex in js_ev.excerpts],
            )
            self.assertEqual("app/worker.js#1", js_ev.primary_excerpt_id)
            self.assertEqual(js_ev.excerpts[0].content, js_ev.excerpt)
            self.assertIn(">   12: const js_line_12 = 12;", js_ev.excerpt)

            # 3. Unknown segment ordering (shell `at` frames): frame 0 at line 45, frame 1 at line 10.
            # Windows are sorted by ascending start_line: #1 [7, 13] (line 10, frame 1), #2 [42, 48] (line 45, frame 0).
            # When segment order is "unknown", primary excerpt is the window containing the FIRST valid frame
            # in original traceback appearance order (frame 0 at line 45 -> app/script.sh#2).
            sh_trace = "\n".join([
                "    at stepA (app/script.sh:45:1)",
                "    at stepB (app/script.sh:10:1)",
            ])
            sh_ctx = build_repair_context(
                Incident("inc-p3-sh", "shell error", sh_trace, "Error"),
                graph,
                profiles,
                repo_root,
                excerpt_radius=3,
            )
            sh_ev = {item.path: item for item in sh_ctx.evidence}["app/script.sh"]
            self.assertEqual("unknown", sh_ctx.trace_segments[0].frame_order)
            self.assertEqual(
                ["app/script.sh#1", "app/script.sh#2"],
                [ex.excerpt_id for ex in sh_ev.excerpts],
            )
            self.assertEqual("app/script.sh#2", sh_ev.primary_excerpt_id)
            self.assertEqual(sh_ev.excerpts[1].content, sh_ev.excerpt)
            self.assertIn(">   45: echo sh_line_45", sh_ev.excerpt)

    def test_p3_character_budget_trims_surrounding_context_before_removing_target_lines(self):
        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"), exist_ok=True)
            with open(os.path.join(repo_root, "app", "caller.py"), "w", encoding="utf-8") as handle:
                handle.write(
                    "\n".join([f"caller_statement_{i:02d} = '{'x' * 20}'" for i in range(1, 51)]) + "\n"
                )
            with open(os.path.join(repo_root, "app", "callee.py"), "w", encoding="utf-8") as handle:
                handle.write(
                    "\n".join([f"callee_statement_{i:02d} = '{'y' * 20}'" for i in range(1, 51)]) + "\n"
                )

            intel = RepoIntelligence(
                repo_path=repo_root,
                repo_url="",
                file_analyses=[
                    FileAnalysis("app/caller.py", "python", 50, 1, functions=["outer", "mid"]),
                    FileAnalysis("app/callee.py", "python", 50, 1, functions=["inner"]),
                ],
            )
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel)

            trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/caller.py", line 10, in outer',
                '  File "app/caller.py", line 40, in mid',
                '  File "app/callee.py", line 25, in inner',
                "RuntimeError: budget test",
            ])
            incident = Incident("inc-p3-budget-trim", "budget test", trace, "RuntimeError")

            # Unconstrained context at excerpt_radius=8
            full_ctx = build_repair_context(
                incident,
                graph,
                profiles,
                repo_root,
                excerpt_radius=8,
                max_excerpt_chars=12000,
            )
            full_chars = full_ctx.truncation.actual_chars
            self.assertFalse(full_ctx.truncation.truncated)

            # Set a budget smaller than full_chars, but large enough to hold all 3 target lines with reduced context
            tight_budget = 600
            self.assertLess(tight_budget, full_chars)

            trimmed_ctx = build_repair_context(
                incident,
                graph,
                profiles,
                repo_root,
                excerpt_radius=8,
                max_excerpt_chars=tight_budget,
            )
            self.assertIsNotNone(trimmed_ctx.truncation)
            self.assertTrue(trimmed_ctx.truncation.truncated)
            self.assertEqual(tight_budget, trimmed_ctx.truncation.limit_chars)
            self.assertLessEqual(trimmed_ctx.truncation.actual_chars, tight_budget)
            self.assertGreater(trimmed_ctx.truncation.actual_chars, 0)
            self.assertTrue(len(trimmed_ctx.truncation.context_trimmed_excerpt_ids) >= 1)
            self.assertEqual([], trimmed_ctx.truncation.omitted_excerpt_ids)
            self.assertEqual(0, trimmed_ctx.truncation.omitted_excerpts)
            self.assertFalse(trimmed_ctx.truncation.target_lines_omitted)
            self.assertEqual([], trimmed_ctx.truncation.omitted_target_lines)

            ev_by_path = {item.path: item for item in trimmed_ctx.evidence}
            caller_ev = ev_by_path["app/caller.py"]
            callee_ev = ev_by_path["app/callee.py"]

            # All 3 windows and all 3 target lines (10, 40 in caller.py; 25 in callee.py) are preserved!
            self.assertEqual(2, len(caller_ev.excerpts))
            self.assertEqual([10], caller_ev.excerpts[0].target_lines)
            self.assertEqual([40], caller_ev.excerpts[1].target_lines)
            self.assertIn(">   10: caller_statement_10", caller_ev.excerpts[0].content)
            self.assertIn(">   40: caller_statement_40", caller_ev.excerpts[1].content)

            self.assertEqual(1, len(callee_ev.excerpts))
            self.assertEqual([25], callee_ev.excerpts[0].target_lines)
            self.assertIn(">   25: callee_statement_25", callee_ev.excerpts[0].content)

            # Innermost valid frame (callee.py line 25) must retain >= effective_radius of outer frames
            outer_radius_0 = caller_ev.excerpts[0].effective_radius
            outer_radius_1 = caller_ev.excerpts[1].effective_radius
            inner_radius = callee_ev.excerpts[0].effective_radius
            self.assertGreaterEqual(inner_radius, outer_radius_1)
            self.assertGreaterEqual(outer_radius_1, outer_radius_0)

    def test_p3_extreme_budget_omits_low_priority_windows_and_records_omitted_targets(self):
        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"), exist_ok=True)
            with open(os.path.join(repo_root, "app", "service.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"line_{i:02d} = {i}" for i in range(1, 51)]) + "\n")

            intel = RepoIntelligence(
                repo_path=repo_root,
                repo_url="",
                file_analyses=[
                    FileAnalysis("app/service.py", "python", 50, 1, functions=["f1", "f2", "f3"]),
                ],
            )
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel)

            # Three distant frames in app/service.py: line 5 (outer), line 25 (mid), line 45 (innermost)
            trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/service.py", line 5, in f1',
                '  File "app/service.py", line 25, in f2',
                '  File "app/service.py", line 45, in f3',
                "RuntimeError: extreme budget",
            ])
            incident = Incident("inc-p3-extreme", "extreme budget", trace, "RuntimeError")

            # At r=0, each single-line excerpt ">   45: line_45 = 45" is 20 chars.
            # Set max_excerpt_chars=42 so exactly 2 target lines fit (innermost line 45 and mid line 25),
            # while the outermost window (app/service.py#1 at line 5) must be omitted.
            ctx_two = build_repair_context(
                incident,
                graph,
                profiles,
                repo_root,
                excerpt_radius=8,
                max_excerpt_chars=42,
            )
            ev_two = ctx_two.evidence[0]
            self.assertEqual(
                ["app/service.py#2", "app/service.py#3"],
                [ex.excerpt_id for ex in ev_two.excerpts],
            )
            self.assertEqual([[25], [45]], [ex.target_lines for ex in ev_two.excerpts])
            self.assertEqual([0, 0], [ex.effective_radius for ex in ev_two.excerpts])
            self.assertTrue(all(ex.context_trimmed for ex in ev_two.excerpts))
            self.assertEqual("app/service.py#3", ev_two.primary_excerpt_id)
            self.assertIn(">   45: line_45 = 45", ev_two.excerpt)
            self.assertEqual(["app/service.py#1"], ev_two.omitted_excerpt_ids)

            self.assertTrue(ctx_two.truncation.truncated)
            self.assertEqual(42, ctx_two.truncation.limit_chars)
            self.assertEqual(40, ctx_two.truncation.actual_chars)
            self.assertLessEqual(ctx_two.truncation.actual_chars, ctx_two.truncation.limit_chars)
            self.assertEqual(
                ["app/service.py#2", "app/service.py#3"],
                ctx_two.truncation.context_trimmed_excerpt_ids,
            )
            self.assertEqual(["app/service.py#1"], ctx_two.truncation.omitted_excerpt_ids)
            self.assertEqual(1, ctx_two.truncation.omitted_excerpts)
            self.assertTrue(ctx_two.truncation.target_lines_omitted)
            self.assertEqual(1, len(ctx_two.truncation.omitted_target_lines))
            omitted_item = ctx_two.truncation.omitted_target_lines[0]
            self.assertEqual("app/service.py", omitted_item.path)
            self.assertEqual(5, omitted_item.line)
            self.assertEqual(0, omitted_item.frame_index)
            self.assertEqual([0], omitted_item.frame_indices)
            self.assertEqual("app/service.py#1", omitted_item.excerpt_id)

            # Adjacent target lines (line 10 and line 11) under a 1-line budget (25 chars):
            # At r=0 they merge into [10, 11], which is 41 chars > 25 chars.
            # Phase B must split/trim the merged r=0 window to preserve the innermost target line (line 11)
            # and record line 10 in omitted_target_lines!
            adj_trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/service.py", line 10, in f1',
                '  File "app/service.py", line 11, in f2',
                "RuntimeError: adjacent extreme budget",
            ])
            ctx_adj = build_repair_context(
                Incident("inc-p3-adj", "adjacent extreme budget", adj_trace, "RuntimeError"),
                graph,
                profiles,
                repo_root,
                excerpt_radius=4,
                max_excerpt_chars=25,
            )
            ev_adj = ctx_adj.evidence[0]
            self.assertEqual(1, len(ev_adj.excerpts))
            self.assertEqual([11], ev_adj.excerpts[0].target_lines)
            self.assertEqual(11, ev_adj.excerpts[0].start_line)
            self.assertEqual(11, ev_adj.excerpts[0].end_line)
            self.assertIn(">   11: line_11 = 11", ev_adj.excerpt)
            self.assertTrue(ctx_adj.truncation.target_lines_omitted)
            self.assertEqual([10], [item.line for item in ctx_adj.truncation.omitted_target_lines])
            self.assertLessEqual(ctx_adj.truncation.actual_chars, 25)

    def test_p3_never_fabricates_excerpts_for_invalid_missing_unreadable_empty_or_non_frame_files(self):
        with tempfile.TemporaryDirectory() as repo_root:
            os.makedirs(os.path.join(repo_root, "app"), exist_ok=True)
            with open(os.path.join(repo_root, "app", "valid.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"valid_line_{i} = {i}" for i in range(1, 11)]) + "\n")
            with open(os.path.join(repo_root, "app", "stale.py"), "w", encoding="utf-8") as handle:
                handle.write("\n".join([f"stale_line_{i} = {i}" for i in range(1, 6)]) + "\n")
            with open(os.path.join(repo_root, "app", "empty.py"), "wb") as handle:
                handle.write(b"")
            with open(os.path.join(repo_root, "app", "binary.py"), "wb") as handle:
                handle.write(b"x = 1\x00y = 2\n")
            with open(os.path.join(repo_root, "app", "neighbor.py"), "w", encoding="utf-8") as handle:
                handle.write("def helper():\n    return 42\n")

            intel = RepoIntelligence(
                repo_path=repo_root,
                repo_url="",
                file_analyses=[
                    FileAnalysis(
                        "app/valid.py",
                        "python",
                        10,
                        1,
                        functions=["run"],
                        imports=["from app.neighbor import helper"],
                    ),
                    FileAnalysis("app/stale.py", "python", 5, 1, functions=["old"]),
                    FileAnalysis("app/missing.py", "python", 5, 1, functions=["gone"]),
                    FileAnalysis("app/empty.py", "python", 0, 1, functions=[]),
                    FileAnalysis("app/binary.py", "python", 2, 1, functions=["bin_fn"]),
                    FileAnalysis("app/neighbor.py", "python", 2, 1, functions=["helper"]),
                ],
            )
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel)

            trace = "\n".join([
                "Traceback (most recent call last):",
                '  File "app/missing.py", line 2, in gone',
                '  File "app/empty.py", line 1, in empty_fn',
                '  File "app/binary.py", line 1, in bin_fn',
                '  File "app/stale.py", line 50, in old',
                '  File "app/valid.py", line 4, in run',
                "RuntimeError: non-fabrication check",
            ])
            context = build_repair_context(
                Incident("inc-p3-no-fab", "non-fabrication check", trace, "RuntimeError"),
                graph,
                profiles,
                repo_root,
            )
            ev_by_path = {item.path: item for item in context.evidence}

            # Valid file produces exactly 1 real excerpt
            valid_ev = ev_by_path["app/valid.py"]
            self.assertEqual(1, len(valid_ev.excerpts))
            self.assertEqual("app/valid.py#1", valid_ev.primary_excerpt_id)
            self.assertIn(">    4: valid_line_4 = 4", valid_ev.excerpt)

            # Missing, empty, binary/unreadable, invalid-only frame, and non-frame neighbor files
            # must never have fabricated excerpts.
            for non_excerpt_path in (
                "app/missing.py",
                "app/empty.py",
                "app/binary.py",
                "app/stale.py",
                "app/neighbor.py",
            ):
                ev = ev_by_path[non_excerpt_path]
                self.assertEqual([], ev.excerpts, f"unexpected excerpts for {non_excerpt_path}")
                self.assertIsNone(ev.primary_excerpt_id, f"unexpected primary_excerpt_id for {non_excerpt_path}")
                self.assertEqual([], ev.omitted_excerpt_ids, f"unexpected omitted_excerpt_ids for {non_excerpt_path}")
                self.assertEqual("", ev.excerpt, f"unexpected excerpt string for {non_excerpt_path}")

            # Valid crash frame outranks invalid-only frame files
            self.assertGreater(ev_by_path["app/valid.py"].score, ev_by_path["app/stale.py"].score)


if __name__ == "__main__":
    unittest.main()
