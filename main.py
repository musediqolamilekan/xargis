#!/usr/bin/env python3
"""
asre — Repository Intelligence Layer
Usage:
    python main.py <github_url_or_owner/repo> [options]
"""

import argparse
import json
import sys
import os

from intelligence.cloner import clone_repository, parse_github_url
from intelligence.scanner import scan_repository, to_dict
from intelligence.dependency_graph import build_dependency_graph
from intelligence.file_profile import build_file_profiles, get_profile
from intelligence.query_engine import query_codebase
from intelligence.git_analyzer import get_file_change_counts, get_file_last_modified
from intelligence.reliability_intelligence import (
    analyze_pre_deployment_risk,
    build_reliability_intelligence_report,
)
from intelligence.slo_intelligence import build_slo_intelligence_report
from intelligence.investor_verification import run_investor_verification_suite
from intelligence.commercial_enterprise import (
    build_commercial_enterprise_report,
    run_repair_api_server,
)
from repair import (
    LLMPatchPlanner,
    SUPPORTED_ENVIRONMENTS,
    SUPPORTED_PROVIDERS,
    SUPPORTED_ROLES,
    analyze_cross_incident_intelligence,
    apply_verified_patch,
    build_and_query_knowledge_base,
    build_repair_context,
    correlate_and_prioritize_incidents,
    create_repair_pull_request,
    detect_incident_from_command,
    evaluate_approval_policy,
    execute_patch_plan,
    is_emergency_stop_active,
    load_incident,
    load_incidents_from_inputs,
    load_patch_plan,
    load_telemetry_events,
    plan_with_repair_loop,
    promote_knowledge_base_record,
    rollback_applied_patch,
    save_repair_run,
    set_emergency_stop,
    validate_patch_plan,
)
from reporting.reporter import print_report
from reporting.graph_reporter import (
    print_graph_overview,
    print_impact_report,
    print_file_relationships,
)
from reporting.profile_reporter import (
    print_who_uses,
    print_file_profile,
    print_query_results,
)
from reporting.repair_reporter import (
    print_commercial_enterprise_report,
    print_cross_incident_intelligence_report,
    print_database_status_report,
    print_incident_intelligence_report,
    print_investor_verification_report,
    print_knowledge_base_report,
    print_pre_deployment_risk_report,
    print_reliability_intelligence_report,
    print_repair_report,
    print_slo_intelligence_report,
    print_training_export_report,
    prompt_and_print_application,
)
from storage import SupabaseStorageEngine, apply_supabase_schema_migration


def _needs_graph(args) -> bool:
    return any([
        args.graph,
        args.impact,
        args.relationships,
        args.list_critical,
        args.who_uses,
        args.profile,
        args.query,
        args.all_profiles,
        args.incident,
        args.detect_command,
        args.correlate_alerts,
        args.cross_incident,
        args.patch_plan,
        args.generate_patch_plan,
        args.verify_patch_plan,
        args.repair_run_dir,
        args.reliability_report,
        args.pre_deploy_check,
        args.slo_report,
        args.sync_db,
        args.commercial_report,
        args.multi_repo,
        args.serve_api,
    ])


def _fuzzy_find(nodes: list[str], query: str) -> list[str]:
    """Returns nodes where query is a substring of the path."""
    return [n for n in nodes if query.lower() in n.lower()]


def main():
    parser = argparse.ArgumentParser(
        prog="xargis",
        description="🔍 Xargis — Autonomous Software Reliability & Causal Repair Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py pallets/click
  python main.py django/django --graph
  python main.py django/django --who-uses models.py
  python main.py django/django --profile core.py
  python main.py django/django --query "authentication error"
  python main.py django/django --list-critical
  python main.py --local ./my-project --all-profiles
        """
    )

    # ── Positional ─────────────────────────────────────────────────
    parser.add_argument(
        "repo",
        nargs="?",
        help="GitHub URL or owner/repo (e.g. 'django/django')",
    )
    parser.add_argument(
        "--local",
        metavar="PATH",
        help="Scan a local directory instead of cloning",
    )

    # ── Dependency graph ───────────────────────────────────────────
    graph_group = parser.add_argument_group("Dependency Graph")
    graph_group.add_argument(
        "--graph",
        action="store_true",
        help="Show CODE RELATIONSHIP GRAPH overview",
    )
    graph_group.add_argument(
        "--impact",
        metavar="FILE",
        help="Impact analysis: what breaks if FILE breaks?",
    )
    graph_group.add_argument(
        "--relationships",
        metavar="FILE",
        help="Show direct import relationships for FILE",
    )
    graph_group.add_argument(
        "--list-critical",
        action="store_true",
        help="Ranked list of all files by impact score",
    )

    # ── Reverse dependency / profiles ──────────────────────────────
    profile_group = parser.add_argument_group("Reverse Dependency & File Profiles")
    profile_group.add_argument(
        "--who-uses",
        metavar="FILE",
        help="Reverse dependency tree: who imports FILE?",
    )
    profile_group.add_argument(
        "--profile",
        metavar="FILE",
        help="Full intelligence profile for FILE (imports, symbols, risk, JSON)",
    )
    profile_group.add_argument(
        "--all-profiles",
        action="store_true",
        help="Export full profiles for every file as JSON (use with --output)",
    )

    # ── Query engine ───────────────────────────────────────────────
    query_group = parser.add_argument_group("Query Engine")
    query_group.add_argument(
        "--query",
        metavar="TEXT",
        help="'What part of the codebase matters for this problem?' — keyword or error message",
    )

    incident_group = parser.add_argument_group("Incident Context")
    incident_group.add_argument(
        "--incident",
        metavar="JSON",
        help="Load a normalized production incident JSON payload",
    )
    incident_group.add_argument(
        "--detect-command",
        metavar="CMD",
        help="Stage 1 (Detect): Run a command/test against the repo and auto-capture any failure as an Incident",
    )
    incident_group.add_argument(
        "--correlate-alerts",
        metavar="FILE",
        help="Ingest, deduplicate, correlate, and prioritize alerts/logs from Sentry, Datadog, New Relic, OTel, Prometheus, or log files",
    )
    incident_group.add_argument(
        "--cross-incident",
        nargs="+",
        metavar="FILE",
        help="Run 3-Layer Hybrid Graph-RAG + Temporal Causal Cross-Incident Intelligence across multiple incidents to find a shared root cause",
    )
    incident_group.add_argument(
        "--explain-context",
        action="store_true",
        help="Print deterministic repository evidence for --incident as JSON",
    )
    incident_group.add_argument(
        "--patch-plan",
        metavar="JSON",
        help="Validate a structured patch plan against the incident repair context",
    )
    incident_group.add_argument(
        "--generate-patch-plan",
        action="store_true",
        help="Generate a validated patch plan from the incident repair context using an LLM",
    )
    incident_group.add_argument(
        "--llm-provider",
        choices=SUPPORTED_PROVIDERS,
        default="auto",
        help="LLM provider for --generate-patch-plan (default: auto, reads API keys from .env)",
    )
    incident_group.add_argument(
        "--llm-model",
        metavar="MODEL",
        help="Override the default model name for --generate-patch-plan",
    )
    incident_group.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        metavar="N",
        help="Max generation + verification attempts for --generate-patch-plan (default: 2)",
    )
    incident_group.add_argument(
        "--challenge",
        metavar="TEXT",
        help="Challenge ASRE's conclusion or provide a counter-hypothesis for deeper investigation",
    )
    incident_group.add_argument(
        "--verify-patch-plan",
        action="store_true",
        help="Apply a validated plan in a temporary Git worktree and run its allowlisted tests",
    )
    incident_group.add_argument(
        "--apply",
        action="store_true",
        help="Apply a worktree-verified patch to the live repository (skips interactive [y/N] prompt)",
    )
    incident_group.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a dedicated repair branch, commit the verified patch with an RCA body, and prepare/open a PR",
    )
    incident_group.add_argument(
        "--repair-run-dir",
        metavar="DIR",
        help="Write auditable incident, context, plan, and verification artifacts under DIR",
    )

    # ── Enterprise Production Protection ───────────────────────────
    guard_group = parser.add_argument_group("Production Protection")
    guard_group.add_argument(
        "--environment",
        choices=SUPPORTED_ENVIRONMENTS,
        default=None,
        help="Target deployment environment policy (development, staging, production)",
    )
    guard_group.add_argument(
        "--role",
        choices=SUPPORTED_ROLES,
        default="sre",
        help="Operator RBAC role (viewer, developer, sre, admin; default: sre)",
    )
    guard_group.add_argument(
        "--operator",
        default=os.environ.get("USER", "cli-operator"),
        metavar="NAME",
        help="Operator identifier recorded in audit logs",
    )
    guard_group.add_argument(
        "--approver",
        default="",
        metavar="NAME",
        help="Explicit human approver sign-off for production, high-risk, auth, or DB migration changes",
    )
    guard_group.add_argument(
        "--health-check",
        metavar="CMD",
        help="Post-apply deployment health check command; triggers automatic rollback on failure",
    )
    guard_group.add_argument(
        "--rollback",
        metavar="PATH",
        help="Roll back a previously applied repair run directory or patch.diff file",
    )
    guard_group.add_argument(
        "--emergency-stop",
        choices=("enable", "disable", "status"),
        help="Engage, clear, or check the repository emergency stop kill switch",
    )

    # ── Reliability Intelligence & Prevention (Section F) ──────────
    rel_group = parser.add_argument_group("Reliability Intelligence & Prevention")
    rel_group.add_argument(
        "--reliability-report",
        action="store_true",
        help="Identify frequently failing components, recurring exceptions, post-repair recurrences, fragile dependencies, technical debt, and preventive actions",
    )
    rel_group.add_argument(
        "--pre-deploy-check",
        nargs="?",
        const="HEAD~1..HEAD",
        metavar="BASE..TARGET",
        help="Detect risky changes before deployment by analyzing modified AST symbols, blast radius, and incident history (default: HEAD~1..HEAD)",
    )
    rel_group.add_argument(
        "--investor-report",
        action="store_true",
        help="Execute live verification benchmarks across all 10 investor due-diligence criteria (multi-repo, MTTD/MTTR, safety matrix, ROI, moat)",
    )
    rel_group.add_argument(
        "--knowledge-base",
        nargs="?",
        const="incident repair runbook",
        metavar="QUERY",
        help="Build and query the 3-Tier Repair Memory & Knowledge Base (Semantic Docs/CODEOWNERS, Episodic CBR Positive/Negative Cases, Procedural Bandit Playbooks)",
    )
    rel_group.add_argument(
        "--promote-kb",
        metavar="INCIDENT_ID",
        help="Human-in-the-Loop Trust Promotion Gate: promote a stored repair record from 'unverified' to 'verified' (requires --approver)",
    )
    rel_group.add_argument(
        "--slo-report",
        action="store_true",
        help="Execute 4-Pillar SLO & Error Budget Intelligence (MWMBR 14.4x/6x/1x burn alerts, AST code attribution, TTE forecast, release freeze governance)",
    )
    rel_group.add_argument(
        "--slo-target",
        type=float,
        default=99.9,
        metavar="PCT",
        help="Target SLO availability percentage for --slo-report (default: 99.9)",
    )
    rel_group.add_argument(
        "--slo-telemetry",
        metavar="FILE",
        help="Optional alert/telemetry JSON or log file to include in --slo-report burn-rate & code attribution",
    )

    # ── Database & ML Training Data (Supabase) ─────────────────────
    db_group = parser.add_argument_group("Supabase Database & ML Training Store")
    db_group.add_argument(
        "--init-db",
        action="store_true",
        help="Apply storage/supabase_schema.sql migration (8 tables + 2 ML training views) to Supabase Postgres",
    )
    db_group.add_argument(
        "--db-status",
        action="store_true",
        help="Check Supabase PostgREST connectivity, schema status, offline queue, and local mirror table counts",
    )
    db_group.add_argument(
        "--sync-db",
        action="store_true",
        help="Sync repository intelligence, repair runs, knowledge base, and flush pending offline queue to Supabase",
    )
    db_group.add_argument(
        "--export-training-data",
        metavar="DIR",
        help="Export LLM SFT, DPO preference pairs, Tabular JIT change-risk CSV/JSONL, and Causal Graph datasets to DIR",
    )

    # ── Commercial Enterprise & Product Capabilities ───────────────
    comm_group = parser.add_argument_group("Commercial Enterprise & Product Capabilities")
    comm_group.add_argument(
        "--commercial-report",
        action="store_true",
        help="Run all 10 Commercial Differentiation Capabilities (Multi-Repo, Ownership, Cost ROI, Effectiveness, Scorecard, Custom Policies, Team KB, Private Mode, Multi-Env, Repair API)",
    )
    comm_group.add_argument(
        "--multi-repo",
        nargs="+",
        default=(),
        metavar="PATH",
        help="Additional repository paths to include in Multi-Repository Federated Intelligence",
    )
    comm_group.add_argument(
        "--policy-file",
        metavar="JSON",
        help="Custom declarative repair policy file (default: .asre/policy.json)",
    )
    comm_group.add_argument(
        "--private-mode",
        action="store_true",
        help="Enforce strict air-gapped / zero-external-egress private deployment mode with SHA-256 compliance attestation",
    )
    comm_group.add_argument(
        "--serve-api",
        action="store_true",
        help="Start the HTTP REST & Webhook Repair API server (/api/v1/health, /api/v1/scorecard, /api/v1/incidents/diagnose, /api/v1/pre-deploy/check)",
    )
    comm_group.add_argument(
        "--api-port",
        type=int,
        default=8080,
        metavar="PORT",
        help="Port for --serve-api (default: 8080)",
    )

    # ── Output ─────────────────────────────────────────────────────
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--json",
        action="store_true",
        help="Output full intelligence scan as JSON",
    )
    out_group.add_argument(
        "--output",
        metavar="FILE",
        help="Save JSON report to a file",
    )
    out_group.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip git pull if repo is already cached",
    )

    args = parser.parse_args()

    incident_sources = sum(bool(x) for x in (args.incident, args.detect_command, args.correlate_alerts, args.cross_incident))
    if incident_sources > 1:
        parser.error("--incident, --detect-command, --correlate-alerts, and --cross-incident are mutually exclusive")
    if (
        args.explain_context
        or args.patch_plan
        or args.generate_patch_plan
        or args.verify_patch_plan
        or args.repair_run_dir
    ) and incident_sources == 0:
        parser.error("incident context options require --incident, --detect-command, --correlate-alerts, or --cross-incident")
    if args.patch_plan and args.generate_patch_plan:
        parser.error("--patch-plan and --generate-patch-plan are mutually exclusive")
    if args.verify_patch_plan and not (args.patch_plan or args.generate_patch_plan):
        parser.error("--verify-patch-plan requires --patch-plan or --generate-patch-plan")
    if (args.apply or args.create_pr) and not args.verify_patch_plan:
        parser.error("--apply and --create-pr require --verify-patch-plan")
    if args.role == "viewer" and (args.generate_patch_plan or args.verify_patch_plan or args.apply or args.create_pr):
        parser.error("RBAC role 'viewer' is read-only and cannot generate, verify, or apply patches")

    if args.serve_api and not (args.repo or args.local):
        from api.app import run_fastapi_server

        run_fastapi_server(
            None,
            port=args.api_port,
            repair_run_dir=args.repair_run_dir or ".asre/runs",
            policy_file=args.policy_file,
            private_mode=args.private_mode,
            additional_repos=args.multi_repo,
        )
        return

    if not args.repo and not args.local:
        if args.init_db or args.db_status or args.sync_db or args.export_training_data:
            args.local = "."
        else:
            parser.print_help()
            sys.exit(1)

    repo_path = None
    repo_url = ""

    # ── Resolve repo path ─────────────────────────────────────────
    if args.local:
        repo_path = os.path.abspath(args.local)
        if not os.path.isdir(repo_path):
            print(f"Error: '{repo_path}' is not a valid directory.", file=sys.stderr)
            sys.exit(1)
        print(f"\n📂 Scanning local repository: {repo_path}")
    else:
        repo_input = args.repo
        owner, repo_name = parse_github_url(repo_input)
        if not owner:
            print(f"Error: Could not parse repository '{repo_input}'.", file=sys.stderr)
            sys.exit(1)

        repo_url = f"https://github.com/{owner}/{repo_name}"
        print(f"\n🌐 Target: {repo_url}")
        repo_path = clone_repository(repo_url, no_pull=args.no_pull)
        print(f"✅ Repository ready at: {repo_path}\n")

    if args.emergency_stop:
        if args.emergency_stop == "status":
            active, reason = is_emergency_stop_active(repo_path)
            print(json.dumps({"emergency_stop": {"active": active, "reason": reason}}, indent=2))
        else:
            state = set_emergency_stop(
                repo_path,
                args.emergency_stop == "enable",
                actor=args.operator,
            )
            print(json.dumps({"emergency_stop": state}, indent=2))
        return

    db_store = SupabaseStorageEngine()
    if args.init_db:
        ok, msg = apply_supabase_schema_migration()
        print(json.dumps({"supabase_schema_migration": {"success": ok, "message": msg}}, indent=2))
        if not (args.db_status or args.sync_db or args.export_training_data):
            return

    if args.db_status:
        status_rep = db_store.check_status()
        if args.json:
            print(json.dumps({"database_status": status_rep.to_dict()}, indent=2))
        else:
            print_database_status_report(status_rep)
        if not (args.sync_db or args.export_training_data):
            return

    if args.export_training_data and not args.sync_db:
        export_rep = db_store.export_training_datasets(
            args.export_training_data,
            repo_root=repo_path,
            repair_run_dir=args.repair_run_dir or ".asre/runs",
        )
        if args.json:
            print(json.dumps({"training_export": export_rep.to_dict()}, indent=2))
        else:
            print_training_export_report(export_rep)
        return

    if args.rollback:
        rb_result = rollback_applied_patch(repo_path, args.rollback)
        db_store.sync_historical_runs_and_audit_logs(
            repo_path,
            repair_run_dir=args.repair_run_dir or ".asre/runs",
        )
        print(json.dumps({"rollback": rb_result.to_dict()}, indent=2))
        return

    if args.investor_report:
        inv_report = run_investor_verification_suite(
            repo_path,
            repair_run_dir=args.repair_run_dir or ".asre/runs",
        )
        if args.json:
            print(json.dumps({"investor_verification": inv_report.to_dict()}, indent=2))
        else:
            print_investor_verification_report(inv_report)
        return

    if args.promote_kb:
        try:
            promoted = promote_knowledge_base_record(
                args.repair_run_dir or ".asre/runs",
                args.promote_kb,
                approver=args.approver or args.operator,
            )
            kb_rep = build_and_query_knowledge_base(
                repo_path,
                repair_run_dir=args.repair_run_dir or ".asre/runs",
            )
            db_store.sync_knowledge_base(kb_rep, repo_path)
            db_store.sync_historical_runs_and_audit_logs(
                repo_path,
                repair_run_dir=args.repair_run_dir or ".asre/runs",
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps({"knowledge_base_promotion": promoted}, indent=2))
        return

    if args.knowledge_base:
        kb_report = build_and_query_knowledge_base(
            repo_path,
            query=args.knowledge_base,
            repair_run_dir=args.repair_run_dir or ".asre/runs",
        )
        db_store.sync_knowledge_base(kb_report, repo_path)
        if args.json:
            print(json.dumps({"knowledge_base": kb_report.to_dict()}, indent=2))
        else:
            print_knowledge_base_report(kb_report)
        return

    # ── Run intelligence scan ─────────────────────────────────────
    print("🧠 Running intelligence scan...\n")
    intel = scan_repository(repo_path, repo_url=repo_url, verbose=True)
    ecosystem = intel.ecosystem_context
    if ecosystem and ecosystem.detected_frameworks:
        print(f"   Detected frameworks: {', '.join(ecosystem.detected_frameworks)}")

    # ── Build dependency graph + profiles (if any graph flag set) ──
    graph = None
    profiles = None

    if _needs_graph(args):
        print("\n🕸️  Building dependency graph...")
        print("   → Applying ecosystem module aliases (PSR-4 / package imports / path maps)")
        graph = build_dependency_graph(intel)
        gstats = graph.stats()
        print(f"   {gstats['total_nodes']} nodes, {gstats['total_edges']} edges resolved")

        print("📋 Building file profiles & risk scores...")
        print("   → Collecting git file change history (last 90 days)...")
        file_changes  = get_file_change_counts(repo_path, days=90)
        file_modified = get_file_last_modified(repo_path)
        print(f"   → {len(file_changes)} files with recent git activity")
        profiles = build_file_profiles(graph, intel, file_changes, file_modified)
        print(f"   {len(profiles)} profiles computed\n")

    db_store.sync_repository_scan(repo_path, intel, graph, profiles)

    # ── Standard output (no flags = full repo report) ─────────────
    if args.output or args.json:
        result = to_dict(intel)

        if args.all_profiles and profiles:
            result["file_profiles"] = {
                path: p.to_dict() for path, p in profiles.items()
            }

        json_str = json.dumps(result, indent=2, default=str)

        if args.output:
            with open(args.output, "w") as f:
                f.write(json_str)
            print(f"\n💾 JSON report saved to: {args.output}")

        if args.json and not args.output:
            print(json_str)
        else:
            print_report(intel)
    else:
        print_report(intel)

    if graph is None or profiles is None:
        return

    all_nodes = graph.all_nodes()

    # ── Graph overview ────────────────────────────────────────────
    if args.graph:
        print_graph_overview(graph)

    # ── Impact analysis ───────────────────────────────────────────
    if args.impact:
        matches = _fuzzy_find(all_nodes, args.impact)
        if not matches:
            print(f"\n⚠️  No file matching '{args.impact}' found. Try a partial path.")
        else:
            for m in matches[:3]:
                print_impact_report(graph.get_impact_report(m))

    # ── Relationships ─────────────────────────────────────────────
    if args.relationships:
        matches = _fuzzy_find(all_nodes, args.relationships)
        if not matches:
            print(f"\n⚠️  No file matching '{args.relationships}' found.")
        else:
            for m in matches[:5]:
                print_file_relationships(graph, m)

    # ── List critical ─────────────────────────────────────────────
    if args.list_critical:
        from intelligence.risk_scorer import risk_icon
        # Use 0-100 risk scores from profiles
        ranked = sorted(profiles.values(), key=lambda p: p.risk_score, reverse=True)
        ranked = [p for p in ranked if p.risk_score > 0]
        if not ranked:
            print("\n  No files with risk scores > 0 found.")
        else:
            print(f"\n{'─'*65}")
            print(f"  {'File':<42} {'Score':>5}  {'Risk':<10}  Consumers")
            print(f"{'─'*65}")
            for p in ranked[:50]:
                icon = risk_icon(p.risk_score)
                name = p.basename[:40]
                print(f"  {name:<42} {p.risk_score:>5}  {icon} {p.risk_label:<8}  {p.transitive_impact}")
            print(f"{'─'*65}\n")

    # ── Who uses (reverse dependency) ────────────────────────────
    if args.who_uses:
        matched = get_profile(profiles, args.who_uses)
        if not matched:
            print(f"\n⚠️  No file matching '{args.who_uses}' found.")
        else:
            for p in matched[:3]:
                print_who_uses(p)

    # ── Full file profile ─────────────────────────────────────────
    if args.profile:
        matched = get_profile(profiles, args.profile)
        if not matched:
            print(f"\n⚠️  No file matching '{args.profile}' found.")
        else:
            for p in matched[:2]:
                print_file_profile(p)

    # ── All profiles as JSON ──────────────────────────────────────
    if args.all_profiles and not args.output:
        all_data = {path: p.to_dict() for path, p in profiles.items()}
        print(json.dumps(all_data, indent=2, default=str))

    # ── Query engine ──────────────────────────────────────────────
    if args.query:
        results = query_codebase(profiles, args.query)
        print_query_results(results, args.query)

    repair_context = None
    patch_plan = None
    validation = None
    execution = None
    application = None
    policy_decision = None
    loop_result = None
    repair_run = None
    if (
        args.correlate_alerts
        or args.cross_incident
        or args.detect_command
        or args.explain_context
        or args.patch_plan
        or args.generate_patch_plan
        or args.verify_patch_plan
        or args.repair_run_dir
    ):
        try:
            if args.cross_incident:
                multi_incidents = load_incidents_from_inputs(args.cross_incident)
                cross_report = analyze_cross_incident_intelligence(
                    multi_incidents,
                    graph,
                    profiles,
                    repo_path,
                    repair_run_dir=args.repair_run_dir,
                )
                db_store.sync_cross_incident_report(cross_report, repo_path)
                if args.json:
                    print(json.dumps({"cross_incident_intelligence": cross_report.to_dict()}, indent=2, default=str))
                else:
                    print_cross_incident_intelligence_report(cross_report)
                if cross_report.primary_merged_incident is None:
                    return
                incident = cross_report.primary_merged_incident
            elif args.correlate_alerts:
                raw_events = load_telemetry_events(args.correlate_alerts)
                intel_report = correlate_and_prioritize_incidents(
                    raw_events,
                    graph,
                    profiles,
                    repo_path,
                )
                if args.json:
                    print(json.dumps({"incident_intelligence": intel_report.to_dict()}, indent=2, default=str))
                else:
                    print_incident_intelligence_report(intel_report)
                if intel_report.top_incident is None:
                    return
                incident = intel_report.top_incident
            elif args.detect_command:
                incident = detect_incident_from_command(repo_path, args.detect_command)
            else:
                incident = load_incident(args.incident)
        except ValueError as exc:
            parser.error(str(exc))
        repair_context = build_repair_context(incident, graph, profiles, repo_path)
        db_store.sync_incident(
            incident,
            repo_path,
            mapped_files=[ev.path for ev in repair_context.evidence if ev.patch_eligible],
        )

    if args.explain_context and repair_context is not None:
        print(json.dumps(repair_context.to_dict(), indent=2, default=str))

    if args.generate_patch_plan and repair_context is not None:
        try:
            planner = LLMPatchPlanner(
                provider=args.llm_provider,
                model=args.llm_model,
            )
            loop_result = plan_with_repair_loop(
                planner,
                repair_context,
                verify=args.verify_patch_plan,
                max_attempts=args.max_attempts,
                repair_run_dir=args.repair_run_dir,
                challenge=args.challenge,
            )
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        patch_plan = loop_result.plan
        validation = loop_result.validation
        execution = loop_result.execution
        if args.json:
            print(json.dumps({"llm_repair": loop_result.to_dict()}, indent=2, default=str))

    if args.patch_plan and repair_context is not None:
        try:
            patch_plan = load_patch_plan(args.patch_plan)
        except ValueError as exc:
            parser.error(str(exc))
        validation = validate_patch_plan(patch_plan, repair_context)
        print(json.dumps({
            "patch_plan": patch_plan.to_dict(),
            "validation": validation.to_dict(),
        }, indent=2, default=str))
        if args.verify_patch_plan:
            execution = execute_patch_plan(patch_plan, repair_context)
            print(json.dumps({"execution": execution.to_dict()}, indent=2, default=str))

    if repair_context is not None and patch_plan is not None:
        policy_decision = evaluate_approval_policy(
            patch_plan,
            repair_context,
            environment=args.environment,
            role=args.role,
            operator=args.operator,
            explicit_human_approval=bool(args.approver.strip()),
            approver=args.approver,
        )

    if args.generate_patch_plan and repair_context is not None and not args.json:
        print_repair_report(
            repair_context,
            loop_result=loop_result,
            plan=patch_plan,
            validation=validation,
            execution=execution,
            policy_decision=policy_decision,
        )

    if (
        repair_context is not None
        and patch_plan is not None
        and execution is not None
        and execution.status == "verified"
    ):
        if args.json:
            if args.create_pr:
                application = create_repair_pull_request(patch_plan, repair_context, execution)
                print(json.dumps({"application": application.to_dict()}, indent=2, default=str))
            elif args.apply:
                application = apply_verified_patch(
                    patch_plan,
                    repair_context,
                    execution,
                    policy_decision=policy_decision,
                    health_check_command=args.health_check,
                )
                print(json.dumps({"application": application.to_dict()}, indent=2, default=str))
        elif args.generate_patch_plan or args.apply or args.create_pr:
            def _handle_challenge(challenge_text: str):
                nonlocal loop_result, patch_plan, validation, execution, policy_decision
                planner = LLMPatchPlanner(
                    provider=args.llm_provider,
                    model=args.llm_model,
                )
                loop_result = plan_with_repair_loop(
                    planner,
                    repair_context,
                    verify=args.verify_patch_plan,
                    max_attempts=args.max_attempts,
                    repair_run_dir=args.repair_run_dir,
                    challenge=challenge_text,
                )
                patch_plan = loop_result.plan
                validation = loop_result.validation
                execution = loop_result.execution
                if patch_plan is not None:
                    policy_decision = evaluate_approval_policy(
                        patch_plan,
                        repair_context,
                        environment=args.environment,
                        role=args.role,
                        operator=args.operator,
                        explicit_human_approval=bool(args.approver.strip()),
                        approver=args.approver,
                    )
                print_repair_report(
                    repair_context,
                    loop_result=loop_result,
                    plan=patch_plan,
                    validation=validation,
                    execution=execution,
                    policy_decision=policy_decision,
                )
                if patch_plan is not None and execution is not None and execution.status == "verified":
                    return prompt_and_print_application(
                        repair_context,
                        patch_plan,
                        execution,
                        auto_apply=False,
                        create_pr=False,
                        interactive=sys.stdin.isatty(),
                        policy_decision=policy_decision,
                        health_check_command=args.health_check,
                        apply_fn=apply_verified_patch,
                        pr_fn=create_repair_pull_request,
                    )
                return None

            application = prompt_and_print_application(
                repair_context,
                patch_plan,
                execution,
                auto_apply=args.apply,
                create_pr=args.create_pr,
                interactive=sys.stdin.isatty(),
                policy_decision=policy_decision,
                health_check_command=args.health_check,
                apply_fn=apply_verified_patch,
                pr_fn=create_repair_pull_request,
                challenge_fn=_handle_challenge if args.generate_patch_plan else None,
            )

    if args.repair_run_dir and repair_context is not None:
        try:
            repair_run = save_repair_run(
                args.repair_run_dir,
                repair_context,
                plan=patch_plan,
                validation=validation,
                execution=execution,
                application=application,
                policy_decision=policy_decision,
            )
        except OSError as exc:
            parser.error(f"failed to write repair run: {exc}")
        db_store.sync_repair_run(
            os.path.basename(repair_run.path),
            repair_context,
            status=repair_run.status,
            kb_trust_status="verified" if (application and application.approved) else "unverified",
            promoted_by=args.approver or "",
            plan=patch_plan,
            execution=execution,
            application=application,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model or "",
        )
        db_store.sync_historical_runs_and_audit_logs(
            repo_path,
            repair_run_dir=args.repair_run_dir,
        )
        if not args.generate_patch_plan or args.json:
            print(json.dumps({"repair_run": repair_run.to_dict()}, indent=2, default=str))
        else:
            print(f"  💾 Audit artifacts saved to: {repair_run.path} (status: {repair_run.status})\n")

    if args.reliability_report:
        rel_report = build_reliability_intelligence_report(
            repo_path,
            graph,
            profiles,
            repair_run_dir=args.repair_run_dir or ".asre/runs",
        )
        if args.json:
            print(json.dumps({"reliability_intelligence": rel_report.to_dict()}, indent=2, default=str))
        else:
            print_reliability_intelligence_report(rel_report)

    if args.pre_deploy_check:
        try:
            deploy_report = analyze_pre_deployment_risk(
                repo_path,
                graph,
                profiles,
                rev_range=args.pre_deploy_check,
                repair_run_dir=args.repair_run_dir or ".asre/runs",
            )
            db_store.sync_change_risk_report(deploy_report, repo_path)
        except ValueError as exc:
            parser.error(str(exc))
        if args.json:
            print(json.dumps({"pre_deployment_risk": deploy_report.to_dict()}, indent=2, default=str))
        else:
            print_pre_deployment_risk_report(deploy_report)

    if args.slo_report:
        try:
            slo_rep = build_slo_intelligence_report(
                repo_path,
                graph,
                profiles,
                slo_target_pct=args.slo_target,
                telemetry_path=args.slo_telemetry,
                repair_run_dir=args.repair_run_dir or ".asre/runs",
            )
            db_store.sync_slo_report(slo_rep, repo_path)
        except ValueError as exc:
            parser.error(str(exc))
        if args.json:
            print(json.dumps({"slo_intelligence": slo_rep.to_dict()}, indent=2, default=str))
        else:
            print_slo_intelligence_report(slo_rep)

    if args.commercial_report or args.multi_repo:
        comm_rep = build_commercial_enterprise_report(
            repo_path,
            intel,
            graph,
            profiles,
            additional_repos=args.multi_repo,
            repair_run_dir=args.repair_run_dir or ".asre/runs",
            policy_file=args.policy_file,
            private_mode=args.private_mode,
            supabase_configured=db_store.is_configured,
        )
        if args.json:
            print(json.dumps({"commercial_enterprise": comm_rep.to_dict()}, indent=2, default=str))
        else:
            print_commercial_enterprise_report(comm_rep)

    if args.serve_api:
        run_repair_api_server(
            repo_path,
            intel,
            graph,
            profiles,
            port=args.api_port,
            repair_run_dir=args.repair_run_dir or ".asre/runs",
            policy_file=args.policy_file,
            private_mode=args.private_mode,
        )

    if args.sync_db:
        repo_sync = db_store.sync_repository_scan(repo_path, intel, graph, profiles)
        kb_rep = build_and_query_knowledge_base(
            repo_path,
            repair_run_dir=args.repair_run_dir or ".asre/runs",
        )
        kb_sync = db_store.sync_knowledge_base(kb_rep, repo_path)
        hist_sync = db_store.sync_historical_runs_and_audit_logs(
            repo_path,
            repair_run_dir=args.repair_run_dir or ".asre/runs",
        )
        flushed = db_store.flush_pending_sync()
        print(
            json.dumps(
                {
                    "supabase_sync": {
                        "repository": repo_sync.to_dict(),
                        "knowledge_base": kb_sync.to_dict(),
                        "repair_runs": hist_sync["repair_runs"],
                        "audit_logs": hist_sync["audit_logs"],
                        "pending_queue_flush": flushed,
                    }
                },
                indent=2,
            )
        )
        if args.export_training_data:
            export_rep = db_store.export_training_datasets(
                args.export_training_data,
                repo_root=repo_path,
                repair_run_dir=args.repair_run_dir or ".asre/runs",
            )
            if args.json:
                print(json.dumps({"training_export": export_rep.to_dict()}, indent=2))
            else:
                print_training_export_report(export_rep)


if __name__ == "__main__":
    main()
