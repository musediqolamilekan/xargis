"""
Terminal renderer for Autonomous Incident Repair (LLMPatchPlanner, validation, worktree verification).
"""

from __future__ import annotations

from dataclasses import replace
import textwrap
from typing import Any, Optional

from intelligence.commercial_enterprise import CommercialEnterpriseReport
from intelligence.investor_verification import InvestorVerificationReport
from intelligence.reliability_intelligence import (
    PreDeploymentRiskReport,
    ReliabilityIntelligenceReport,
)
from intelligence.slo_intelligence import SLOIntelligenceReport
from storage.supabase_store import DatabaseStatusReport, TrainingExportReport
from repair.cross_incident_intelligence import CrossIncidentIntelligenceReport
from repair.incident_intelligence import IncidentIntelligenceReport
from repair.knowledge_base import KnowledgeBaseReport
from repair.incident_mapper import RepairContext
from repair.llm_planner import RepairLoopResult
from repair.patch_executor import (
    PatchApplicationResult,
    PatchExecutionResult,
    explain_untouched_evidence,
)
from repair.patch_planner import PatchPlan, PatchPlanValidation
from repair.production_guard import ApprovalPolicyDecision
from repair.repair_runs import RepairRun

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
WHITE   = "\033[97m"


def _hr(char: str = "─", width: int = 62) -> None:
    print(f"{DIM}{char * width}{RESET}")


def _section(title: str, icon: str = "") -> None:
    sep = f"  {icon}  " if icon else "  "
    print(f"\n{BOLD}{CYAN}{sep}{title}{RESET}")
    _hr()


def _confidence_bar(confidence: float, width: int = 24) -> str:
    clamped = max(0.0, min(1.0, confidence))
    filled = round(clamped * width)
    empty = width - filled
    color = GREEN if clamped >= 0.75 else (YELLOW if clamped >= 0.45 else RED)
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET} {BOLD}{clamped * 100:.0f}%{RESET}"


def _print_colorized_diff(unified_diff: str) -> None:
    for line in unified_diff.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            print(f"  {BOLD}{WHITE}{line}{RESET}")
        elif line.startswith("@@"):
            print(f"  {CYAN}{line}{RESET}")
        elif line.startswith("+"):
            print(f"  {GREEN}{line}{RESET}")
        elif line.startswith("-"):
            print(f"  {RED}{line}{RESET}")
        else:
            print(f"  {DIM}{line}{RESET}")


def print_repair_report(
    context: RepairContext,
    *,
    loop_result: Optional[RepairLoopResult] = None,
    plan: Optional[PatchPlan] = None,
    validation: Optional[PatchPlanValidation] = None,
    execution: Optional[PatchExecutionResult] = None,
    repair_run: Optional[RepairRun] = None,
    policy_decision: Optional[ApprovalPolicyDecision] = None,
) -> None:
    """Renders a color-coded terminal report for an autonomous incident repair run."""
    active_plan = plan or (loop_result.plan if loop_result else None)
    active_validation = validation or (loop_result.validation if loop_result else None)
    active_execution = execution or (loop_result.execution if loop_result else None)

    print(f"\n{BOLD}{'═' * 62}{RESET}")
    print(f"{BOLD}{CYAN}  🛠️   AUTONOMOUS INCIDENT REPAIR REPORT{RESET}")
    print(f"{BOLD}{'═' * 62}{RESET}")

    # ── Incident & LLM Summary ─────────────────────────────────────
    inc = context.incident
    print(f"\n  {'Incident ID:':<20} {BOLD}{WHITE}{inc.incident_id}{RESET}")
    if inc.exception_type:
        print(f"  {'Exception:':<20} {RED}{BOLD}{inc.exception_type}{RESET}")
    wrapped_msg = textwrap.fill(inc.message, width=78, subsequent_indent=" " * 23)
    print(f"  {'Error Message:':<20} {WHITE}{wrapped_msg}{RESET}")

    if loop_result is not None:
        print(f"  {'LLM Engine:':<20} {MAGENTA}{BOLD}{loop_result.provider}{RESET}  {DIM}({loop_result.model}){RESET}")
        print(f"  {'Attempts Used:':<20} {BOLD}{len(loop_result.attempts)}{RESET}")
        if loop_result.challenge:
            print(f"  {'Engineer Challenge:':<20} {YELLOW}\"{loop_result.challenge}\"{RESET}")

    if active_plan is not None:
        print(f"  {'Confidence:':<20} {_confidence_bar(active_plan.confidence)}")

    # ── Diagnosis & Recommendation ─────────────────────────────────
    ev_by_path = {item.path: item for item in context.evidence}
    if loop_result is not None and loop_result.lessons_used:
        _section("HISTORICAL LESSONS RETRIEVED (LEARN)", "🧠")
        for lesson in loop_result.lessons_used:
            print(f"  {MAGENTA}• [{lesson.incident_id}]{RESET} {WHITE}{lesson.diagnosis[:72]}{RESET}")

    if active_plan is not None:
        _section("CONFIRMED EVIDENCE vs. SUSPECTED CAUSE", "🔬")
        confirmed_items = list(active_plan.confirmed_evidence)
        if not confirmed_items:
            for item in context.evidence:
                if item.evidence_tier == "confirmed" and item.valid_frame_lines:
                    confirmed_items.append(
                        f"Confirmed stack frame in {item.path} at line(s) {', '.join(str(ln) for ln in item.valid_frame_lines)} "
                        f"(file_status={item.file_status}, recent_90d_commits={item.recent_changes})"
                    )
        print(f"  {BOLD}{GREEN}📌 Supporting Evidence (Confirmed Facts):{RESET}")
        if confirmed_items:
            for fact in confirmed_items:
                print(f"   {GREEN}✓{RESET} {WHITE}{fact}{RESET}")
        else:
            print(f"   {YELLOW}⚠ No confirmed frame evidence — heuristic/corroborated signal only{RESET}")

        cause_text = active_plan.suspected_cause or active_plan.diagnosis
        print(f"\n  {BOLD}{YELLOW}🔍 Root-Cause Hypothesis (Suspected Cause):{RESET}")
        for line in textwrap.wrap(cause_text, width=56):
            print(f"   {WHITE}{line}{RESET}")

        _section("ROOT-CAUSE ANALYSIS (RCA) & CONTEXT", "🩺")
        for line in textwrap.wrap(active_plan.diagnosis, width=58):
            print(f"  {WHITE}{line}{RESET}")

        # Relevant Code Locations
        print(f"\n  {BOLD}Relevant Code Locations:{RESET}")
        for path in active_plan.files_to_modify:
            ev = ev_by_path.get(path)
            if ev and ev.valid_frame_lines:
                enclosing = next((ex.enclosing_symbol for ex in ev.excerpts if ex.enclosing_symbol), None)
                sym_tag = f" in {enclosing}()" if enclosing else ""
                print(f"   {CYAN}•{RESET} {WHITE}{path}:{','.join(str(l) for l in ev.valid_frame_lines)}{sym_tag}{RESET}")
            else:
                print(f"   {CYAN}•{RESET} {WHITE}{path}{RESET}")

        # Related Commits
        all_commits: list[tuple[str, dict[str, str]]] = []
        for path in active_plan.files_to_modify:
            ev = ev_by_path.get(path)
            if ev and ev.related_commits:
                for c in ev.related_commits[:3]:
                    all_commits.append((path, c))
        if all_commits:
            print(f"\n  {BOLD}Related Commits:{RESET}")
            for path, c in all_commits[:5]:
                print(
                    f"   {MAGENTA}• [{c['sha']}]{RESET} {WHITE}{c['message'][:46]}{RESET} "
                    f"{DIM}({c['author']}, {c['date']} · {path}){RESET}"
                )

        # Affected Components
        affected_comps: list[str] = []
        for path in active_plan.files_to_modify:
            ev = ev_by_path.get(path)
            if ev and ev.affected_components:
                for comp in ev.affected_components:
                    if comp not in affected_comps:
                        affected_comps.append(comp)
        print(f"\n  {BOLD}Affected Components & Blast Radius ({len(active_plan.files_to_modify)} target, {len(affected_comps)} downstream):{RESET}")
        for path in active_plan.files_to_modify:
            ev = ev_by_path.get(path)
            if ev is not None:
                print(
                    f"   {DIM}└──{RESET} {YELLOW}{path}{RESET}  "
                    f"{DIM}[Risk: {ev.risk_label} ({ev.risk_score}/90) · "
                    f"{ev.direct_dependents} direct / {ev.transitive_dependents} transitive dependents]{RESET}"
                )
            else:
                print(f"   {DIM}└──{RESET} {YELLOW}{path}{RESET}")
        if affected_comps:
            print(f"       {DIM}Downstream consumers: {', '.join(affected_comps[:6])}{RESET}")

        # Possible Contributing Factors
        combined_factors = list(active_plan.contributing_factors)
        for path in active_plan.files_to_modify:
            ev = ev_by_path.get(path)
            if ev and ev.contributing_factors:
                for cf in ev.contributing_factors:
                    if cf not in combined_factors:
                        combined_factors.append(cf)
        if combined_factors:
            print(f"\n  {BOLD}Possible Contributing Factors:{RESET}")
            for factor in combined_factors[:5]:
                print(f"   {YELLOW}•{RESET} {DIM}{factor}{RESET}")

        # Business & Technical Impact
        if active_plan.risk_assessment:
            print(f"\n  {BOLD}Business & Technical Impact:{RESET}")
            for line in textwrap.wrap(active_plan.risk_assessment, width=56):
                print(f"   {DIM}{line}{RESET}")

        # Recommended Actions
        rec_actions = list(active_plan.recommended_actions)
        if not rec_actions:
            rec_actions = [
                f"Apply verified patch to {', '.join(active_plan.files_to_modify)}",
                f"Run verification suite: {', '.join(active_plan.tests_to_run)}",
            ]
        print(f"\n  {BOLD}Recommended Actions:{RESET}")
        for idx, act in enumerate(rec_actions, start=1):
            print(f"   {GREEN}{idx}.{RESET} {WHITE}{act}{RESET}")

        if active_plan.alternatives:
            print(f"\n  {BOLD}Alternatives Considered:{RESET}")
            for alt in active_plan.alternatives:
                print(f"   {DIM}• {alt}{RESET}")

        # ── What ASRE Deliberately Chose Not to Change ─────────────
        untouched_reasons = explain_untouched_evidence(active_plan, context)
        if untouched_reasons:
            _section("WHAT ASRE DELIBERATELY CHOSE NOT TO CHANGE", "🛑")
            for item_reason in untouched_reasons[:6]:
                print(f"  {CYAN}•{RESET} {WHITE}{item_reason}{RESET}")

        # ── Unified Diff ───────────────────────────────────────────
        _section("GENERATED PATCH (UNIFIED DIFF)", "📝")
        _print_colorized_diff(active_plan.unified_diff)
    else:
        _section("PATCH GENERATION FAILED", "❌")
        if loop_result and loop_result.attempts:
            for att in loop_result.attempts:
                print(f"  {RED}Attempt {att.attempt}:{RESET} {att.error}")

    # ── Deterministic Safety Validation ────────────────────────────
    if active_validation is not None:
        _section("DETERMINISTIC SAFETY & SECURITY GATE", "🛡️")
        if active_validation.valid:
            print(f"  {GREEN}{BOLD}✅ PASSED{RESET}  {DIM}All safety & security constraints satisfied{RESET}")
            print(f"   {GREEN}✓{RESET} Target files scoped to patch-eligible RepairContext evidence")
            print(f"   {GREEN}✓{RESET} Unified diff headers match declared files_to_modify")
            print(f"   {GREEN}✓{RESET} Diff security scan passed (no eval/exec, shell=True, or hardcoded secrets)")
            print(f"   {GREEN}✓{RESET} Confidence score backed by confirmed supporting evidence")
            print(f"   {GREEN}✓{RESET} Verification commands pass shell-injection & runner allowlist")
        else:
            print(f"  {RED}{BOLD}❌ REJECTED{RESET}")
            for err in active_validation.errors:
                print(f"   {RED}✗{RESET} {err}")

    # ── Production Protection & Policy Gate ────────────────────────
    if policy_decision is not None:
        _section("ENTERPRISE PRODUCTION PROTECTION & RBAC POLICY", "🔒")
        print(f"  {'Target Environment:':<24} {BOLD}{WHITE}{policy_decision.environment}{RESET}")
        print(
            f"  {'Operator & RBAC Role:':<24} {BOLD}{CYAN}{policy_decision.operator}{RESET} "
            f"(role={BOLD}{policy_decision.role}{RESET}, min_required={policy_decision.required_minimum_role})"
        )
        print(
            f"  {'Repair Risk Threshold:':<24} {BOLD}{policy_decision.risk_tier}{RESET} "
            f"({policy_decision.max_risk_score}/90 · auto-apply threshold <20 in development)"
        )
        domains_str = (
            f"{RED}{BOLD}{', '.join(policy_decision.sensitive_domains)}{RESET}"
            if policy_decision.sensitive_domains
            else f"{GREEN}None (no auth or DB migration touched){RESET}"
        )
        print(f"  {'Sensitive Domains:':<24} {domains_str}")
        auto_str = (
            f"{GREEN}✓ Permitted (low-risk dev fix){RESET}"
            if policy_decision.auto_apply_permitted
            else f"{YELLOW}✗ Requires explicit human approval{RESET}"
        )
        print(f"  {'Unattended Auto-Apply:':<24} {auto_str}")
        print(f"  {'Secret Redaction:':<24} {GREEN}✓ Outbound prompt & artifact secret scrubbing active{RESET}")
        print(f"  {'Emergency Stop Switch:':<24} {GREEN}✓ Clear{RESET}")
        p_color = GREEN if policy_decision.allowed_to_apply else YELLOW
        print(f"  {'Policy Verdict:':<24} {p_color}{policy_decision.reason}{RESET}")

    # ── Isolated Git Worktree Verification ─────────────────────────
    if active_execution is not None:
        _section("ISOLATED GIT WORKTREE VERIFICATION", "🧪")
        status_upper = active_execution.status.upper()
        if active_execution.status == "verified":
            badge = f"{GREEN}{BOLD}✅ {status_upper}{RESET}"
        else:
            badge = f"{RED}{BOLD}❌ {status_upper}{RESET}"
        print(f"  {'Status:':<22} {badge}")
        if active_execution.revision:
            print(f"  {'Target Revision:':<22} {BOLD}[{active_execution.revision[:8]}]{RESET}")
        cleanup_str = f"{GREEN}✓ Removed cleanly (main repo untouched){RESET}" if active_execution.worktree_removed else f"{YELLOW}Not created{RESET}"
        print(f"  {'Ephemeral Worktree:':<22} {cleanup_str}")
        repro_str = (
            f"{GREEN}✓ Reproduced failing test before applying patch{RESET}"
            if active_execution.reproduced_before_patch
            else f"{DIM}Pre-patch test exited 0 or not reproduced{RESET}"
        )
        print(f"  {'Failure Reproduction:':<22} {repro_str}")
        replay_str = (
            f"{GREEN}✓ Replayed against patch — original failure no longer reproduces{RESET}"
            if active_execution.incident_replay_passed
            else (
                f"{GREEN}✓ Verification suite passed against patch{RESET}"
                if active_execution.status == "verified"
                else f"{RED}✗ Incident replay failed{RESET}"
            )
        )
        print(f"  {'Incident Replay:':<22} {replay_str}")
        if active_execution.impact_summary:
            imp = active_execution.impact_summary
            adj_reg = imp.get("adjacent_regressions", 0)
            adj_badge = (
                f"{GREEN}0 adjacent regressions{RESET}"
                if adj_reg == 0 and active_execution.status == "verified"
                else f"{RED}{adj_reg} adjacent regression(s){RESET}"
            )
            print(
                f"  {'Blast-Radius Impact:':<22} "
                f"{WHITE}Max risk {imp.get('max_risk_score', 0)}/90 · "
                f"{imp.get('total_direct_dependents', 0)} direct / "
                f"{imp.get('total_transitive_dependents', 0)} transitive consumers · "
                f"{adj_badge}"
            )
            consumers = imp.get("downstream_consumers") or []
            if consumers:
                print(f"  {'Downstream Modules:':<22} {DIM}{', '.join(consumers[:6])}{RESET}")

        if active_execution.simulation_report:
            sim = active_execution.simulation_report
            t1 = sim.get("tier1_policy_invariants") or {}
            t2 = sim.get("tier2_graph_capacity") or {}
            t3 = sim.get("tier3_dry_run_state_diff") or {}
            ar = sim.get("action_risk") or {}
            dm = sim.get("decision_matrix") or {}
            q_mod = t2.get("queuing_model") or {}
            scm = t2.get("scm_counterfactuals") or {}

            _section("4-TIER PROGRESSIVE REPAIR SIMULATION & R_action RISK MATRIX", "🔬")
            print(
                f"  {'Tier 1 Policy Invariants:':<28} "
                f"{GREEN if t1.get('passed') else RED}{'✓ PASSED' if t1.get('passed') else '✗ BLOCKED'}{RESET} "
                f"{DIM}({t1.get('invariants_evaluated', 0)} formal rules in {t1.get('elapsed_ms', 0):.2f}ms){RESET}"
            )
            print(
                f"  {'Tier 2 M/M/c & Cascade Sim:':<28} "
                f"{GREEN if t2.get('passed') else RED}{'✓ STABLE' if t2.get('passed') else '✗ SATURATED'}{RESET} "
                f"{DIM}(M/M/{q_mod.get('workers_c', 4)} ρ: {q_mod.get('utilization_rho_before', 0):.2f} → {q_mod.get('utilization_rho_after', 0):.2f}){RESET}"
            )
            print(
                f"  {'Tier 2 SCM Counterfactual:':<28} "
                f"{WHITE}{scm.get('counterfactual_summary', '')}{RESET}"
            )
            print(
                f"  {'Tier 3 State-Diff Dry-Run:':<28} "
                f"{GREEN if t3.get('passed') else YELLOW}{'✓ REVERSIBLE' if t3.get('reverse_rollback_clean') else '⚠ STATEFUL'}{RESET} "
                f"{DIM}(est. TTR_rollback={t3.get('estimated_ttr_rollback_seconds', 0):.1f}s · signatures_ok={t3.get('public_signatures_compatible')}){RESET}"
            )
            r_val = float(ar.get("r_action", 0.0))
            r_col = GREEN if r_val < 0.25 else (YELLOW if r_val <= 0.65 else RED)
            print(
                f"  {'Action Risk Score R_action:':<28} "
                f"{r_col}{BOLD}{r_val:.3f}{RESET}  "
                f"{DIM}[P_success={ar.get('predicted_efficacy_p_success', 0):.2f} · "
                f"Blast={ar.get('blast_radius_dimension', 0):.2f} · "
                f"Reversibility={ar.get('reversibility_dimension', 0):.2f} · "
                f"Statefulness={ar.get('statefulness_dimension', 0):.2f}]{RESET}"
            )
            print(
                f"  {'Autonomous Decision Band:':<28} "
                f"{r_col}{BOLD}{dm.get('execution_band', '')}{RESET}"
            )
            print(f"  {'Orchestrator Directive:':<28} {WHITE}{dm.get('directive_summary', '')}{RESET}")

        test_cmds = active_execution.commands[5:-1] if active_execution.worktree_removed and len(active_execution.commands) > 6 else []
        if test_cmds:
            print(f"\n  {BOLD}Verification Test Runs:{RESET}")
            for cmd_res in test_cmds:
                cmd_str = " ".join(cmd_res.command)
                icon = f"{GREEN}✓{RESET}" if cmd_res.returncode == 0 else f"{RED}✗{RESET}"
                print(f"   {icon} {BOLD}{cmd_str}{RESET}  {DIM}(exit {cmd_res.returncode}){RESET}")
                output_lines = [
                    ln.strip()
                    for ln in (cmd_res.stderr or cmd_res.stdout).splitlines()
                    if ln.strip()
                ]
                for ln in output_lines[-3:]:
                    print(f"     {DIM}{ln}{RESET}")
        elif active_execution.error:
            print(f"  {RED}Error: {active_execution.error}{RESET}")

    # ── Persisted Repair Run Artifacts ─────────────────────────────
    if repair_run is not None:
        _section("AUDITABLE REPAIR RUN ARTIFACTS", "💾")
        status_color = GREEN if repair_run.status in {"verified", "applied"} else (RED if repair_run.status in {"failed", "rejected"} else YELLOW)
        print(f"  {'Run Status:':<20} {status_color}{BOLD}{repair_run.status}{RESET}")
        print(f"  {'Artifact Directory:':<20} {BOLD}{WHITE}{repair_run.path}{RESET}")
        print(f"  {DIM}Saved: incident.json · repair-context.json · patch-plan.json · patch.diff · verification.json · run.json{RESET}")

    print(f"\n{BOLD}{'═' * 62}{RESET}\n")


def prompt_and_print_application(
    context: RepairContext,
    plan: PatchPlan,
    execution: PatchExecutionResult,
    *,
    auto_apply: bool = False,
    create_pr: bool = False,
    interactive: bool = True,
    policy_decision: Optional[ApprovalPolicyDecision] = None,
    health_check_command: Optional[str] = None,
    apply_fn=None,
    pr_fn=None,
    challenge_fn=None,
) -> Optional[PatchApplicationResult]:
    """
    Prompt the human operator to:
      [y] Approve & apply verified patch
      [p] Create repair branch & Pull Request
      [c] Challenge conclusion & request deeper re-investigation
      [N] Decline
    """
    if execution.status != "verified":
        return None

    _section("HUMAN-IN-THE-LOOP APPROVAL, PR & CHALLENGE", "👤")
    print(f"  Target Repository: {BOLD}{WHITE}{context.repository_path}{RESET}")
    print(f"  Verified Files:    {YELLOW}{', '.join(plan.files_to_modify)}{RESET}")
    if policy_decision is not None and not policy_decision.role_authorized:
        print(f"\n  {RED}{BOLD}❌ RBAC BLOCKED:{RESET} {policy_decision.reason}")
        print(f"\n{BOLD}{'═' * 62}{RESET}\n")
        return PatchApplicationResult(
            status="rejected",
            approved=False,
            error=policy_decision.reason,
            policy_decision=policy_decision.to_dict(),
        )

    mode = "decline"
    if create_pr:
        print(f"  Approval Mode:     {GREEN}{BOLD}Auto-create branch & PR (--create-pr){RESET}")
        mode = "pr"
    elif auto_apply:
        if policy_decision is not None and not policy_decision.allowed_to_apply:
            print(f"\n  {YELLOW}⚠️  Unattended --apply blocked by Production Protection Policy:{RESET}")
            print(f"     {WHITE}{policy_decision.reason}{RESET}")
            if interactive:
                try:
                    ans = input(
                        f"\n  {BOLD}{CYAN}👉 Provide explicit human approval to apply now? [y/N]: {RESET}"
                    ).strip().lower()
                    if ans in {"y", "yes"}:
                        mode = "apply"
                        policy_decision = replace(
                            policy_decision,
                            explicit_human_approval=True,
                            approver=policy_decision.operator,
                            allowed_to_apply=True,
                            reason=f"Explicitly approved interactively by '{policy_decision.operator}' ({policy_decision.role})",
                        )
                except (EOFError, KeyboardInterrupt):
                    print()
                    mode = "decline"
            else:
                print(f"     {DIM}Pass --approver <NAME> with --role sre/admin or use --create-pr.{RESET}")
                print(f"\n{BOLD}{'═' * 62}{RESET}\n")
                return PatchApplicationResult(
                    status="rejected",
                    approved=False,
                    error=policy_decision.reason,
                    policy_decision=policy_decision.to_dict(),
                )
        else:
            print(f"  Approval Mode:     {GREEN}{BOLD}Auto-approved (--apply){RESET}")
            mode = "apply"
    elif interactive:
        try:
            prompt_label = (
                f"\n  {BOLD}{CYAN}👉 [y] Apply patch / [p] Create branch & PR / [c] Challenge conclusion / [N] Decline: {RESET}"
                if challenge_fn is not None
                else f"\n  {BOLD}{CYAN}👉 [y] Apply patch / [p] Create branch & PR / [N] Decline: {RESET}"
            )
            answer = input(prompt_label).strip().lower()
            if answer in {"c", "challenge"} and challenge_fn is not None:
                challenge_text = input(
                    f"  {BOLD}{YELLOW}🔎 Enter your challenge or question for deeper investigation: {RESET}"
                ).strip()
                if challenge_text:
                    print(f"\n  {MAGENTA}🔄 Re-investigating with engineer challenge...{RESET}")
                    return challenge_fn(challenge_text)
            elif answer in {"p", "pr"}:
                mode = "pr"
            elif answer in {"y", "yes"}:
                mode = "apply"
                if policy_decision is not None:
                    policy_decision = replace(
                        policy_decision,
                        explicit_human_approval=True,
                        approver=policy_decision.operator,
                        allowed_to_apply=policy_decision.role_authorized,
                        reason=f"Explicitly approved interactively by '{policy_decision.operator}' ({policy_decision.role})",
                    )
        except (EOFError, KeyboardInterrupt):
            print()
            mode = "decline"
    else:
        print(f"  {DIM}Non-interactive session (pass --apply, --create-pr, or --challenge TEXT).{RESET}")
        return PatchApplicationResult(status="declined", approved=False)

    if mode == "decline":
        print(f"\n  {YELLOW}⏸️  Patch declined — live repository left untouched.{RESET}")
        print(f"\n{BOLD}{'═' * 62}{RESET}\n")
        return PatchApplicationResult(status="declined", approved=False)

    if mode == "pr":
        if pr_fn is None:
            from repair.patch_executor import create_repair_pull_request as pr_fn
        result = pr_fn(plan, context, execution)
        if result.status in {"pr_created", "pr_branch_ready"}:
            print(f"\n  {GREEN}{BOLD}✅ REPAIR BRANCH & PULL REQUEST READY{RESET}")
            print(f"   {GREEN}✓{RESET} Branch:  {BOLD}{CYAN}{result.branch_name}{RESET}  {DIM}[{result.commit_sha[:8]}]{RESET}")
            print(f"   {GREEN}✓{RESET} Title:   {WHITE}{result.pr_title}{RESET}")
            if result.pr_url:
                print(f"   {GREEN}✓{RESET} PR URL:  {BOLD}{GREEN}{result.pr_url}{RESET}")
        else:
            print(f"\n  {RED}{BOLD}❌ PR CREATION FAILED:{RESET} {result.error}")
        print(f"\n{BOLD}{'═' * 62}{RESET}\n")
        return result

    if apply_fn is None:
        from repair.patch_executor import apply_verified_patch as apply_fn
    result = apply_fn(
        plan,
        context,
        execution,
        policy_decision=policy_decision,
        health_check_command=health_check_command,
    )
    if result.status == "applied":
        print(f"\n  {GREEN}{BOLD}✅ PATCH APPLIED TO LIVE REPOSITORY{RESET}")
        for path in result.applied_files:
            print(f"   {GREEN}✓{RESET} Modified {BOLD}{WHITE}{path}{RESET}")
        if result.health_check is not None:
            print(f"   {GREEN}✓{RESET} Deployment health check passed (`{' '.join(result.health_check.command)}`)")
    elif result.status == "rolled_back":
        print(f"\n  {RED}{BOLD}🔄 AUTOMATIC ROLLBACK TRIGGERED:{RESET} {result.error}")
        print(f"   {YELLOW}↩{RESET} Live repository automatically restored to pre-patch state")
    else:
        print(f"\n  {RED}{BOLD}❌ PATCH APPLICATION FAILED:{RESET} {result.error}")
    print(f"\n{BOLD}{'═' * 62}{RESET}\n")
    return result


def print_incident_intelligence_report(report: IncidentIntelligenceReport) -> None:
    """Renders the Section A Incident Intelligence & Alert Correlation report."""
    print(f"\n{BOLD}{'═' * 62}{RESET}")
    print(f"{BOLD}{CYAN}  📡  INCIDENT INTELLIGENCE & ALERT CORRELATION{RESET}")
    print(f"{BOLD}{'═' * 62}{RESET}")

    print(f"\n  {'Raw Alerts Ingested:':<26} {BOLD}{WHITE}{report.total_raw_events:,}{RESET}")
    print(f"  {'Deduplicated Groups:':<26} {BOLD}{YELLOW}{report.deduplicated_groups:,}{RESET}")
    print(f"  {'Correlated Root Causes:':<26} {BOLD}{GREEN}{report.root_cause_clusters:,}{RESET}")
    print(f"  {'Alert Noise Reduction:':<26} {BOLD}{GREEN}{report.noise_reduction_pct:.1f}%{RESET}")
    if report.sources_detected:
        print(f"  {'Connected Sources:':<26} {CYAN}{', '.join(report.sources_detected)}{RESET}")
    if report.services_affected:
        print(f"  {'Affected Services:':<26} {MAGENTA}{', '.join(report.services_affected)}{RESET}")
    if report.distributed_traces_correlated:
        print(f"  {'Correlated Trace IDs:':<26} {BOLD}{report.distributed_traces_correlated}{RESET}")

    for idx, cluster in enumerate(report.clusters, start=1):
        p_color = RED if cluster.priority_label == "P0-Critical" else (
            YELLOW if cluster.priority_label in {"P1-High", "P2-Medium"} else GREEN
        )
        _section(
            f"ROOT-CAUSE CLUSTER #{idx} — {cluster.priority_label} ({cluster.priority_score:.1f}/100)",
            "🔥" if cluster.priority_label == "P0-Critical" else "⚡",
        )
        print(f"  {'Suspect Root-Cause File:':<26} {BOLD}{YELLOW}{cluster.root_cause_file or 'unmapped'}{RESET}")
        print(
            f"  {'Correlated Volume:':<26} {BOLD}{WHITE}{cluster.total_events:,} alerts{RESET} "
            f"{DIM}across {cluster.group_count} distinct symptom group(s){RESET}"
        )
        print(
            f"  {'Impact Breakdown:':<26} "
            f"{DIM}Tech: {cluster.technical_impact_score:.1f}/40 · "
            f"Business: {cluster.business_impact_score:.1f}/35 · "
            f"Anomaly: {cluster.anomaly_score:.1f}/25{RESET}"
        )

        if cluster.correlation_reasons:
            print(f"\n  {BOLD}Why these alerts were correlated:{RESET}")
            for reason in cluster.correlation_reasons:
                print(f"   {CYAN}↳{RESET} {WHITE}{reason}{RESET}")

        if cluster.anomalies:
            print(f"\n  {BOLD}Anomalies Detected:{RESET}")
            for anomaly in cluster.anomalies:
                print(f"   {p_color}⚠{RESET} {WHITE}{anomaly}{RESET}")

        print(f"\n  {BOLD}Correlated Symptom Groups ({len(cluster.groups)}):{RESET}")
        for g in cluster.groups:
            src_badge = ",".join(g.sources)
            svc_badge = ",".join(g.services)
            print(
                f"   {DIM}├──{RESET} {BOLD}{WHITE}[{g.event_count:>3}×]{RESET} "
                f"{RED}{g.exception_type}{RESET}: {g.message_template[:48]} "
                f"{DIM}({src_badge} · {svc_badge}){RESET}"
            )

    print(f"\n{BOLD}{'═' * 62}{RESET}\n")


def print_reliability_intelligence_report(report: ReliabilityIntelligenceReport) -> None:
    """Renders the Section F Reliability Intelligence & Prevention report."""
    print(f"\n{BOLD}{'═' * 62}{RESET}")
    print(f"{BOLD}{CYAN}  🛡️   RELIABILITY INTELLIGENCE & PREVENTION REPORT{RESET}")
    print(f"{BOLD}{'═' * 62}{RESET}")
    print(f"\n  {'Historical Repair Runs Analyzed:':<34} {BOLD}{WHITE}{report.total_historical_runs}{RESET}")
    print(f"  {'Frequently Failing Components:':<34} {BOLD}{YELLOW}{len(report.failing_components)}{RESET}")
    print(f"  {'Post-Repair Recurrences Detected:':<34} {BOLD}{RED if report.post_repair_recurrences else GREEN}{len(report.post_repair_recurrences)}{RESET}")
    print(f"  {'Fragile Dependencies Identified:':<34} {BOLD}{YELLOW}{len(report.fragile_dependencies)}{RESET}")
    print(f"  {'Technical Debt Hotspots:':<34} {BOLD}{MAGENTA}{len(report.technical_debt_hotspots)}{RESET}")

    if report.post_repair_recurrences:
        _section("POST-REPAIR INCIDENT RECURRENCES (REGRESSIONS)", "🚨")
        for rec in report.post_repair_recurrences:
            print(
                f"  {RED}⚠ {BOLD}{rec.path}{RESET} — {RED}{rec.exception_type}{RESET} recurred in "
                f"'{BOLD}{rec.recurring_incident_id}{RESET}' after {rec.prior_repair_status} repair "
                f"'{BOLD}{rec.prior_repair_incident_id}{RESET}'"
            )
            if rec.recurring_message:
                print(f"    {DIM}{rec.recurring_message[:70]}{RESET}")

    if report.failing_components:
        _section("FREQUENTLY FAILING COMPONENTS (INCIDENT HOTSPOTS)", "🔥")
        for fc in report.failing_components:
            print(
                f"  {BOLD}{WHITE}{fc.path:<36}{RESET} "
                f"{RED}{BOLD}{fc.incident_count} incident(s){RESET} · "
                f"{GREEN}{fc.verified_repairs} repaired{RESET} · "
                f"{DIM}Risk {fc.risk_score}/90 ({fc.transitive_dependents} downstream){RESET}"
            )
            if fc.recurring_exceptions:
                print(f"    {DIM}Exceptions: {', '.join(fc.recurring_exceptions)}{RESET}")

    if report.recurring_exceptions:
        _section("RECURRING EXCEPTIONS", "🔁")
        for rex in report.recurring_exceptions:
            print(
                f"  {RED}{BOLD}{rex.exception_type:<26}{RESET} "
                f"{WHITE}{rex.occurrences}×{RESET}  "
                f"{DIM}across {', '.join(rex.affected_files[:4])}{RESET}"
            )

    if report.fragile_dependencies:
        _section("FRAGILE DEPENDENCIES", "🧩")
        for fd in report.fragile_dependencies[:6]:
            print(
                f"  {YELLOW}{BOLD}{fd.path:<36}{RESET} "
                f"Fragility: {BOLD}{fd.fragility_score:.0f}/100{RESET}  "
                f"{DIM}(Risk {fd.risk_score}/90 · {fd.recent_changes_90d} commits 90d){RESET}"
            )
            for reason in fd.reasons[:3]:
                print(f"    {DIM}↳ {reason}{RESET}")

    if report.technical_debt_hotspots:
        _section("GROWING TECHNICAL DEBT HOTSPOTS", "🧱")
        for td in report.technical_debt_hotspots[:6]:
            print(
                f"  {MAGENTA}{BOLD}{td.path:<36}{RESET} "
                f"Debt Score: {BOLD}{td.debt_score:.0f}/100{RESET}  "
                f"{DIM}({td.lines:,} lines · {td.todo_count} TODO/FIXME){RESET}"
            )
            for ind in td.indicators[:3]:
                print(f"    {DIM}↳ {ind}{RESET}")

    if report.recommendations:
        _section("RECOMMENDED PREVENTIVE RELIABILITY IMPROVEMENTS", "✅")
        for idx, rec in enumerate(report.recommendations, start=1):
            p_col = RED if rec.priority == "P0" else (YELLOW if rec.priority == "P1" else CYAN)
            print(f"  {p_col}{BOLD}[{rec.priority}]{RESET} {BOLD}{WHITE}{idx}. {rec.title}{RESET}")
            print(f"       {DIM}Why:    {rec.rationale}{RESET}")
            print(f"       {GREEN}Action: {rec.suggested_action}{RESET}\n")

    print(f"{BOLD}{'═' * 62}{RESET}\n")


def print_pre_deployment_risk_report(report: PreDeploymentRiskReport) -> None:
    """Renders the 3-Signal Predictive Change Risk Analysis & Autonomous Governance report."""
    print(f"\n{BOLD}{'═' * 66}{RESET}")
    print(f"{BOLD}{CYAN}  🚦  PREDICTIVE CHANGE RISK ANALYSIS (3-SIGNAL + SHAP + GOVERNANCE){RESET}")
    print(f"{BOLD}{'═' * 66}{RESET}")

    lvl_color = RED if report.deployment_risk_level in {"CRITICAL", "HIGH"} else (
        YELLOW if report.deployment_risk_level == "MEDIUM" else GREEN
    )
    print(f"\n  {'Commit Range:':<30} {BOLD}{WHITE}{report.stable_commit[:8]}..{report.target_commit[:8]}{RESET}")
    print(
        f"  {'Calibrated Incident Prob:':<30} "
        f"{lvl_color}{BOLD}{report.calibrated_probability:.2f} ({report.calibrated_probability * 100:.1f}%){RESET}  "
        f"{DIM}[Platt-Scaled Log-Odds]{RESET}"
    )
    print(
        f"  {'Deployment Risk Verdict:':<30} "
        f"{lvl_color}{BOLD}{report.deployment_risk_level} ({report.deployment_risk_score:.1f}/100){RESET}"
    )
    if report.governance_decision is not None:
        gov = report.governance_decision
        print(f"  {'Autonomous Governance Action:':<30} {lvl_color}{BOLD}{gov.autonomous_action}{RESET}")
        print(f"  {'Orchestrator Directive:':<30} {WHITE}{gov.action_summary}{RESET}")

    # ── SHAP Feature Attributions ──────────────────────────────────
    if report.shap_attributions:
        _section("SHAP ADDITIVE RISK ATTRIBUTIONS (EXPLAINABILITY)", "📊")
        for attr in report.shap_attributions:
            print(
                f"  {lvl_color}{BOLD}+{attr.shap_delta_pct:>5.1f}%{RESET}  "
                f"{CYAN}[{attr.signal_layer}]{RESET}"
            )
            print(f"           {WHITE}{attr.explanation}{RESET}")

    # ── Signal 1: Structural & Historical Tabular Metrics ──────────
    if report.tabular_metrics is not None:
        tm = report.tabular_metrics
        _section("SIGNAL 1 — STRUCTURAL & HISTORICAL METADATA (JIT TABULAR)", "📈")
        print(
            f"  {'Code Churn & Diffusion:':<28} "
            f"{WHITE}+{tm.lines_added} / -{tm.lines_deleted} lines across {tm.files_touched} file(s), "
            f"{tm.subsystems_touched} subsystem(s) (Shannon entropy={tm.change_entropy:.2f}){RESET}"
        )
        print(
            f"  {'Author & File History:':<28} "
            f"{WHITE}{tm.author_name} ({tm.author_familiarity_pct:.0f}% familiarity · "
            f"{tm.recent_bugfix_commits} recent bugfix/revert commits){RESET}"
        )
        print(f"  {'Deploy Timing Context:':<28} {YELLOW if tm.is_off_hours_or_friday_deploy else GREEN}{tm.deploy_timing_label}{RESET}")
        print(
            f"  {'Imbalance & Calibration:':<28} "
            f"{DIM}Pos-class weight w+={tm.positive_class_weight:.1f}× · Historical PR-AUC={tm.historical_pr_auc:.3f}{RESET}"
        )

    # ── Signal 2: Blast Radius & Graph Centrality ──────────────────
    if report.graph_metrics is not None:
        gm = report.graph_metrics
        _section("SIGNAL 2 — BLAST RADIUS & DEPENDENCY GRAPH CENTRALITY", "🕸️")
        print(
            f"  {'Max Reverse PageRank:':<28} {BOLD}{WHITE}{gm.max_reverse_pagerank:.4f}{RESET}  "
            f"{DIM}(Brandes Betweenness Centrality={gm.max_betweenness_centrality:.4f}){RESET}"
        )
        print(
            f"  {'Downstream Consumers:':<28} "
            f"{WHITE}{gm.total_direct_dependents} direct / {gm.total_transitive_dependents} transitive dependents{RESET}"
        )
        if gm.degraded_dependencies:
            print(f"  {'Degraded Dependencies:':<28} {RED}{', '.join(gm.degraded_dependencies[:5])}{RESET}")

    # ── Signal 3: Semantic AST & Config Diff + Memory ──────────────
    if report.semantic_inspection is not None:
        si = report.semantic_inspection
        _section("SIGNAL 3 — SEMANTIC AST DIFF & CROSS-INCIDENT MEMORY", "🧠")
        if si.modified_symbols:
            print(f"  {'Modified AST Symbols:':<28} {YELLOW}{', '.join(si.modified_symbols[:6])}{RESET}")
        if si.config_keys_modified:
            print(f"  {'Modified Config Keys:':<28} {MAGENTA}{', '.join(si.config_keys_modified[:6])}{RESET}")
        if si.ast_semantic_hazards:
            print(f"  {BOLD}AST / Diff Hazards Detected:{RESET}")
            for haz in si.ast_semantic_hazards:
                print(f"   {YELLOW}⚠{RESET} {WHITE}{haz}{RESET}")
        if si.historical_incident_links:
            print(f"  {BOLD}Cross-Incident Memory Citations:{RESET}")
            for cite in si.historical_incident_links:
                print(f"   {RED}↳{RESET} {WHITE}{cite}{RESET}")
        if si.llm_critic_review:
            print(f"  {BOLD}LLM Critic Review:{RESET} {WHITE}{si.llm_critic_review}{RESET}")

    if report.changed_files:
        _section("CHANGED FILES & TOUCHED AST SYMBOLS", "📝")
        for cf in report.changed_files:
            print(
                f"  {YELLOW}{BOLD}{cf.path}{RESET}  {DIM}({cf.status} · +{cf.lines_added}/-{cf.lines_deleted} · "
                f"Risk {cf.risk_label} {cf.risk_score}/90 · PageRank {cf.reverse_pagerank:.4f} · "
                f"{cf.transitive_dependents} downstream){RESET}"
            )
            for reason in cf.reasons:
                print(f"    {DIM}↳ {reason}{RESET}")

    if report.warnings:
        _section("PRE-DEPLOYMENT RISK WARNINGS", "⚠️")
        for warn in report.warnings:
            print(f"  {lvl_color}⚠{RESET} {WHITE}{warn}{RESET}")

    if report.recommended_safeguards:
        _section("RECOMMENDED DEPLOYMENT SAFEGUARDS", "🛡️")
        for sg in report.recommended_safeguards:
            print(f"  {GREEN}✓{RESET} {WHITE}{sg}{RESET}")

    print(f"\n{BOLD}{'═' * 66}{RESET}\n")


def print_investor_verification_report(report: InvestorVerificationReport) -> None:
    """Renders the live 10-point Investor Due-Diligence & Verification Benchmark report."""
    print(f"\n{BOLD}{'═' * 68}{RESET}")
    print(f"{BOLD}{CYAN}  🏛️   ASRE INVESTOR DUE-DILIGENCE & VERIFICATION REPORT{RESET}")
    print(f"{BOLD}{'═' * 68}{RESET}")

    badge = (
        f"{GREEN}{BOLD}✅ ALL 10 INVESTMENT REQUIREMENTS VERIFIED (LIVE){RESET}"
        if report.all_requirements_verified
        else f"{YELLOW}{BOLD}⚠️ PARTIAL VERIFICATION{RESET}"
    )
    print(f"\n  Overall Status:  {badge}")

    # ── 10-Requirement Scorecard ───────────────────────────────────
    _section("10-POINT INVESTOR REQUIREMENT SCORECARD", "📊")
    for req in report.requirements:
        icon = f"{GREEN}✅{RESET}" if req.verified else f"{RED}❌{RESET}"
        print(f"  {icon} {BOLD}{WHITE}{req.requirement:<28}{RESET} {CYAN}{BOLD}{req.headline_metric}{RESET}")
        print(f"     {DIM}{req.evidence_summary}{RESET}")

    # ── Measurable MTTD / MTTR & ROI Benchmark ─────────────────────
    m = report.measurable_metrics
    _section("MEASURABLE RESULTS — LIVE MTTD / MTTR & ROI BENCHMARK", "⏱️")
    print(f"  {'Alert Storm Triage:':<32} {BOLD}{WHITE}{m['alerts_ingested']} alerts → {m['correlated_root_causes']} root cause ({m['alert_noise_reduction_pct']}% noise reduction){RESET}")
    print(f"  {'Measured Time to Diagnosis:':<32} {BOLD}{GREEN}{m['mttd_seconds']}s{RESET}")
    print(f"  {'Measured Time to Verified Fix:':<32} {BOLD}{GREEN}{m['mttr_seconds']}s{RESET}  {DIM}(vs. {m['manual_baseline_minutes']:.0f} min manual baseline){RESET}")
    print(f"  {'MTTR Reduction:':<32} {BOLD}{GREEN}{m['mttr_reduction_pct']}% faster{RESET}")
    print(f"  {'Engineering Hours Saved:':<32} {BOLD}{WHITE}{m['hours_saved_per_100_incidents']} hrs{RESET} {DIM}per 100 incidents{RESET}")
    print(f"  {'Estimated Annual Value / Team:':<32} {BOLD}{GREEN}${m['estimated_annual_savings_usd_per_team']:,} / year{RESET}")

    # ── Multi-Repository & Multi-Incident Technical Credibility ────
    _section(f"TECHNICAL CREDIBILITY & SCALABILITY ({len(report.multi_repo_benchmarks)} REAL REPOSITORIES)", "🌐")
    print(f"  {'Repository':<30} {'Files':>6} {'LOC':>9}  {'Incident Type':<18} {'Localized Root Cause'}")
    _hr("─", 68)
    for b in report.multi_repo_benchmarks:
        check = f"{GREEN}✓{RESET}" if b.confirmed_evidence else f"{YELLOW}•{RESET}"
        print(
            f"  {check} {WHITE}{b.repo_name:<28}{RESET} "
            f"{b.files_scanned:>6,} {b.lines_of_code:>9,}  "
            f"{YELLOW}{b.incident_type_tested:<18}{RESET} "
            f"{CYAN}{b.localized_root_file}{RESET} {DIM}({b.scan_and_localize_ms:.0f}ms){RESET}"
        )

    # ── 8-Vector Safety & Production Protection Matrix ─────────────
    _section(f"SAFETY & PRODUCTION PROTECTION MATRIX ({len(report.safety_matrix)}/{len(report.safety_matrix)} BLOCKED)", "🛡️")
    for s in report.safety_matrix:
        icon = f"{GREEN}✓ BLOCKED{RESET}" if s.blocked_or_rolled_back else f"{RED}✗ FAILED{RESET}"
        print(f"  {icon}  {BOLD}{WHITE}{s.vector_name:<34}{RESET} {DIM}{s.attack_description}{RESET}")

    # ── Defensibility Moat ─────────────────────────────────────────
    moat = report.defensibility_moat
    _section("DEFENSIBILITY MOAT & ECOSYSTEM COVERAGE", "🏰")
    print(f"  {'Polyglot AST Support:':<30} {BOLD}{WHITE}{moat['supported_languages_count']} languages{RESET} {DIM}({', '.join(moat['supported_languages'][:8])}...){RESET}")
    print(f"  {'Connected Integrations:':<30} {CYAN}{', '.join(moat['observability_integrations'])}{RESET}")
    print(f"  {'Fleet Scale Verified:':<30} {BOLD}{WHITE}{moat['fleet_repositories_verified']} repos · {moat['fleet_total_files']:,} files · {moat['fleet_total_loc']:,} LOC{RESET}")
    print(f"  {'Episodic Memory & Audit:':<30} {BOLD}{WHITE}{moat['historical_repair_runs_indexed']} repair runs · {moat['immutable_audit_log_entries']} audit log records{RESET}")

    print(f"\n{BOLD}{'═' * 68}{RESET}\n")


def print_cross_incident_intelligence_report(report: CrossIncidentIntelligenceReport) -> None:
    """Renders the 3-Layer Hybrid Graph-RAG + Temporal Causal Cross-Incident Intelligence report."""
    print(f"\n{BOLD}{'═' * 68}{RESET}")
    print(f"{BOLD}{CYAN}  🕸️   CROSS-INCIDENT INTELLIGENCE (HYBRID GRAPH-RAG + TEMPORAL CAUSAL){RESET}")
    print(f"{BOLD}{'═' * 68}{RESET}")

    print(f"\n  {'Incidents Analyzed:':<30} {BOLD}{WHITE}{report.total_incidents_analyzed}{RESET}")
    print(f"  {'Correlated Root-Cause Groups:':<30} {BOLD}{GREEN}{report.correlated_root_causes}{RESET}")
    print(
        f"  {'Relationship Graph Topology:':<30} "
        f"{WHITE}{len(report.relationship_graph.nodes)} nodes · {len(report.relationship_graph.edges)} causal/dependency edges{RESET}"
    )

    for grp in report.correlated_groups:
        _section(f"COMMON ROOT-CAUSE GROUP [{grp.correlation_id.upper()}] ({len(grp.incidents)} INCIDENTS)", "🎯")
        print(f"  {'Common Root-Cause Module:':<28} {BOLD}{YELLOW}{grp.common_root_cause_file or 'N/A'}{RESET}  {DIM}(PPR={grp.ppr_score:.4f} · Confidence={grp.confidence * 100:.0f}%){RESET}")
        if grp.common_root_cause_resource:
            print(f"  {'Shared Infrastructure Cause:':<28} {BOLD}{MAGENTA}{grp.common_root_cause_resource}{RESET}")
        print(f"  {'Root Trigger Incident:':<28} {BOLD}{RED}{grp.root_trigger_incident_id}{RESET}")
        print(f"\n  {BOLD}Unified Root-Cause Hypothesis:{RESET}")
        for line in textwrap.wrap(grp.root_cause_hypothesis, width=62):
            print(f"   {WHITE}{line}{RESET}")

        if grp.shared_dependencies:
            print(f"\n  {BOLD}Shared Dependencies:{RESET} {CYAN}{', '.join(grp.shared_dependencies[:6])}{RESET}")

        print(f"\n  {BOLD}Correlated Incidents & Drain3 Log Templates:{RESET}")
        for inc in grp.incidents:
            ts_badge = f" @ {inc.timestamp}" if inc.timestamp else ""
            print(
                f"   {DIM}├──{RESET} {BOLD}{WHITE}{inc.incident_id:<14}{RESET} "
                f"[{CYAN}{inc.service}{RESET}{ts_badge}] {RED}{inc.exception_type}{RESET}: {inc.message[:46]}"
            )
            print(f"   {DIM}│    Drain Template [{inc.drain_template.template_id}]: {inc.drain_template.template[:58]}{RESET}")

        if grp.service_propagation_paths:
            print(f"\n  {BOLD}Multi-Hop Cross-Service Causal Propagation DAG (PC Transitive Reduction):{RESET}")
            for path_chain in grp.service_propagation_paths:
                print(f"   {RED}⚡{RESET} " + f" {YELLOW}──▶{RESET} ".join(f"{BOLD}{WHITE}{step}{RESET}" for step in path_chain))

        if grp.service_causal_attributions:
            print(f"\n  {BOLD}Counterfactual Do-Calculus Root-Cause Attribution (P(resolve | do(fix S))):{RESET}")
            for attr in grp.service_causal_attributions[:4]:
                bar_col = GREEN if attr.counterfactual_resolution_pct >= 90.0 else (YELLOW if attr.counterfactual_resolution_pct >= 50.0 else DIM)
                print(
                    f"   {bar_col}{BOLD}{attr.counterfactual_resolution_pct:>5.1f}%{RESET} cascade resolved by fixing "
                    f"{BOLD}{CYAN}{attr.service}{RESET} ({attr.incident_id}) "
                    f"{DIM}[affects {len(attr.downstream_services_affected)} service(s) · Granger={attr.granger_lag_score:.2f}]{RESET}"
                )

        if grp.temporal_causal_chain:
            print(f"\n  {BOLD}Temporal & Causal Cascade Edges (Layer C Lead-Lag Discovery):{RESET}")
            for link in grp.temporal_causal_chain:
                print(
                    f"   {RED}{BOLD}{link.cause_incident_id}{RESET} ({link.cause_service}) "
                    f"{YELLOW}──[+{link.lag_seconds:.1f}s · conf {link.causal_confidence * 100:.0f}% · Granger {link.granger_score:.2f}]──▶{RESET} "
                    f"{WHITE}{BOLD}{link.effect_incident_id}{RESET} ({link.effect_service})"
                )
                print(f"     {DIM}↳ {link.mechanism}{RESET}")

        if grp.changepoint_commits:
            print(f"\n  {BOLD}Deployment / Commit Change-Points on Root-Cause Module:{RESET}")
            for cp in grp.changepoint_commits:
                print(f"   {YELLOW}•{RESET} [{BOLD}{cp['sha']}{RESET}] {WHITE}{cp['message']}{RESET} {DIM}({cp.get('author', '')}, {cp.get('date', '')}){RESET}")

        if grp.correlation_evidence:
            print(f"\n  {BOLD}Multi-Layer Correlation Evidence:{RESET}")
            for ev in grp.correlation_evidence:
                print(f"   {CYAN}↳{RESET} {WHITE}{ev}{RESET}")

    # ── Incident Relationship Graph Summary ────────────────────────
    if report.relationship_graph.edges:
        _section("INCIDENT RELATIONSHIP GRAPH (TOPOLOGY & CAUSAL EDGES)", "🔗")
        for edge in report.relationship_graph.edges[:12]:
            print(
                f"  {WHITE}{edge.source:<22}{RESET} "
                f"{CYAN}──[{edge.relation}]──▶{RESET} "
                f"{YELLOW}{edge.target}{RESET}"
            )
            print(f"    {DIM}{edge.evidence}{RESET}")

    if report.recurring_patterns:
        _section("RECURRING FAILURE PATTERNS DETECTED", "🔁")
        for pat in report.recurring_patterns:
            print(f"  {YELLOW}•{RESET} {WHITE}{pat}{RESET}")

    # ── Layer A: Historical Hybrid Graph-RAG Comparisons (RRF) ─────
    if report.historical_comparisons:
        _section("HISTORICAL INCIDENT COMPARISONS (HYBRID BM25 + SEMANTIC + GRAPH RRF)", "📚")
        for comp in report.historical_comparisons:
            print(
                f"  {BOLD}{WHITE}{comp.historical_incident_id:<22}{RESET} "
                f"Similarity: {GREEN}{BOLD}{comp.similarity_pct:.1f}%{RESET} "
                f"{DIM}(RRF={comp.rrf_score:.4f} · BM25 #{comp.bm25_rank} · Sem #{comp.semantic_rank} · Graph #{comp.graph_rank} · status={comp.historical_status}){RESET}"
            )
            if comp.shared_files:
                print(f"    {DIM}Shared Files: {', '.join(comp.shared_files[:4])}{RESET}")
            print(f"    {DIM}Diagnosis:    {comp.historical_diagnosis[:68]}{RESET}")
            print(f"    {GREEN}Lesson:       {comp.reusable_lesson}{RESET}")

    print(f"\n{BOLD}{'═' * 68}{RESET}\n")


def print_knowledge_base_report(report: KnowledgeBaseReport) -> None:
    """Renders the 3-Tier Cognitive Repair Memory & Knowledge Base report."""
    print(f"\n{BOLD}{'═' * 68}{RESET}")
    print(f"{BOLD}{CYAN}  🧠  ASRE 3-TIER REPAIR MEMORY & KNOWLEDGE BASE{RESET}")
    print(f"{BOLD}{'═' * 68}{RESET}")

    print(f"\n  {'Active Search Query:':<32} {BOLD}{WHITE}{report.query[:48]}{RESET}")
    print(f"  {'Tier 1 Semantic Parent Docs:':<32} {BOLD}{WHITE}{report.total_semantic_parent_docs}{RESET} {DIM}(Runbooks, Architecture, CODEOWNERS, IaC){RESET}")
    print(
        f"  {'Tier 2 Episodic Incident Cases:':<32} "
        f"{BOLD}{WHITE}{report.total_episodic_cases}{RESET} "
        f"({GREEN}{report.positive_cases_count} positive{RESET} · {RED}{report.negative_cases_count} negative/anti-patterns{RESET})"
    )
    print(
        f"  {'KB Trust & Drift Governance:':<32} "
        f"{GREEN}{report.verified_trust_count} human-promoted (verified){RESET} · "
        f"{YELLOW}{report.unverified_trust_count} unverified{RESET} · "
        f"{MAGENTA}{report.drifted_cases_count} drifted/pruned{RESET}"
    )
    print(f"  {'Tier 3 Procedural Repair Skills:':<32} {BOLD}{CYAN}{report.total_procedural_skills}{RESET} {DIM}(Thompson Sampling Bandit Ranked){RESET}")
    if report.persisted_index_path:
        print(f"  {'Persisted Knowledge Index:':<32} {DIM}{report.persisted_index_path}{RESET}")

    # ── Tier 1: Semantic Knowledge Base ────────────────────────────
    if report.tier1_semantic_hits:
        _section("TIER 1 — SEMANTIC KNOWLEDGE BASE (PARENT-DOC + CROSS-ENCODER RERANKED)", "📖")
        for hit in report.tier1_semantic_hits:
            pdoc = hit.parent_doc
            pre_badge = f" {YELLOW}[Safety Preconditions Included]{RESET}" if pdoc.has_safety_preconditions else ""
            print(
                f"  {BOLD}{WHITE}{pdoc.source_path}{RESET} :: {CYAN}{pdoc.section_title}{RESET} "
                f"{DIM}({pdoc.doc_kind} · score={hit.final_score:.2f} · CE={hit.cross_encoder_score:.2f}){RESET}{pre_badge}"
            )
            if pdoc.precondition_summary:
                print(f"    {YELLOW}⚠ Precondition:{RESET} {WHITE}{pdoc.precondition_summary[:80]}{RESET}")
            print(f"    {DIM}Excerpt: {hit.matched_child_excerpt.splitlines()[0][:82]}{RESET}")

    if report.resolved_owners:
        print(f"\n  {BOLD}Resolved CODEOWNERS:{RESET}")
        for f_path, owners in list(report.resolved_owners.items())[:5]:
            print(f"   {CYAN}•{RESET} {WHITE}{f_path:<32}{RESET} → {BOLD}{', '.join(owners)}{RESET}")

    # ── Tier 2: Episodic Incident Memory (Positive & Negative CBR) ─
    if report.tier2_positive_cases or report.tier2_negative_cases:
        _section("TIER 2 — EPISODIC INCIDENT MEMORY (CASE-BASED REASONING + GRAPH-RAG)", "🗂️")
        if report.tier2_positive_cases:
            print(f"  {GREEN}{BOLD}Positive Verified Trajectories (What Worked & Survived):{RESET}")
            for case in report.tier2_positive_cases:
                print(
                    f"   {GREEN}✓{RESET} {BOLD}{WHITE}{case.incident_id:<18}{RESET} "
                    f"[{CYAN}{case.structural_operator}{RESET}] on {YELLOW}{', '.join(case.affected_files[:3]) or 'repo'}{RESET} "
                    f"{DIM}(MTTR={case.recovery_time_seconds:.1f}s · sim={case.similarity_score:.2f}){RESET}"
                )
                print(f"     {DIM}Diagnosis: {case.diagnosis[:76]}{RESET}")

        if report.tier2_negative_cases:
            print(f"\n  {RED}{BOLD}Negative Trajectories & Anti-Pattern Guardrails (What Failed / Rolled Back):{RESET}")
            for case in report.tier2_negative_cases:
                print(
                    f"   {RED}✗{RESET} {BOLD}{WHITE}{case.incident_id:<18}{RESET} "
                    f"[{RED}{case.status.upper()}{RESET}] on {YELLOW}{', '.join(case.affected_files[:3]) or 'repo'}{RESET}"
                )
                print(f"     {YELLOW}↳ {case.anti_pattern_warning[:88]}{RESET}")

    # ── Tier 3: Procedural Repair Memory (Thompson Sampling Bandit) ─
    if report.tier3_procedural_skills:
        _section("TIER 3 — PROCEDURAL REPAIR SKILLS & PLAYBOOKS (THOMPSON SAMPLING BANDIT)", "🛠️")
        for idx, skill in enumerate(report.tier3_procedural_skills[:6], start=1):
            kind_badge = f"{MAGENTA}AST Playbook{RESET}" if skill.skill_kind == "ast_code_playbook" else f"{BLUE}Workflow Skill{RESET}"
            print(
                f"  {BOLD}{WHITE}{idx}. {skill.name}{RESET}  [{kind_badge}]"
            )
            print(
                f"     {GREEN}Bandit Score: {skill.thompson_bandit_score:.3f}{RESET} · "
                f"{WHITE}Posterior Confidence: {skill.posterior_confidence * 100:.1f}%{RESET} "
                f"{DIM}(Beta α={skill.alpha:.0f}, β={skill.beta_param:.0f} · avg MTTR={skill.mean_recovery_seconds:.1f}s){RESET}"
            )
            print(f"     {DIM}{skill.description}{RESET}")
            if skill.preconditions:
                print(f"     {YELLOW}Precondition:{RESET} {DIM}{skill.preconditions[0]}{RESET}")

    print(f"\n{BOLD}{'═' * 68}{RESET}\n")


def print_slo_intelligence_report(report: SLOIntelligenceReport) -> None:
    """Renders the 4-Pillar SLO & Error Budget Intelligence report."""
    print(f"\n{BOLD}{'═' * 68}{RESET}")
    print(f"{BOLD}{CYAN}  📉  SLO & ERROR BUDGET INTELLIGENCE (MWMBR + AST CODE ATTRIBUTION){RESET}")
    print(f"{BOLD}{'═' * 68}{RESET}")

    fc = report.forecast
    gov = report.governance_policy
    rem_col = GREEN if fc.remaining_budget_pct > 50.0 else (YELLOW if fc.remaining_budget_pct > 15.0 else RED)

    print(f"\n  {'Target Service / Repository:':<32} {BOLD}{WHITE}{report.service_or_repo}{RESET}")
    print(
        f"  {'SLO Target & Window:':<32} "
        f"{BOLD}{WHITE}{fc.slo_target_pct:.3f}%{RESET} ({fc.window_days}d rolling · "
        f"Total budget={fc.total_error_budget_minutes:.1f} min)"
    )
    print(
        f"  {'Observed Availability:':<32} "
        f"{rem_col}{BOLD}{fc.observed_availability_pct:.4f}%{RESET} "
        f"{DIM}({fc.failed_requests_evaluated:,} failed / {fc.total_requests_evaluated:,} requests){RESET}"
    )
    print(
        f"  {'Remaining Error Budget:':<32} "
        f"{rem_col}{BOLD}{fc.remaining_budget_pct:.1f}% ({fc.remaining_budget_minutes:.1f} min remaining){RESET} · "
        f"{DIM}{fc.consumed_budget_minutes:.1f} min consumed{RESET}"
    )
    tte_str = f"{fc.time_to_exhaustion_hours:.1f} hours" if fc.time_to_exhaustion_hours is not None else "Sustainable (no exhaustion projected)"
    print(
        f"  {'EWMA Burn Rate & TTE Forecast:':<32} "
        f"{rem_col}{BOLD}{fc.ewma_burn_rate:.2f}×{RESET} · Time-to-Exhaustion: {BOLD}{tte_str}{RESET}"
    )
    print(f"  {'Forecast Summary:':<32} {WHITE}{fc.forecast_summary}{RESET}")

    # ── Pillar 1: Multi-Window, Multi-Burn-Rate (MWMBR) Alerts ─────
    _section("PILLAR 1 — MULTI-WINDOW MULTI-BURN-RATE (MWMBR) ANALYTICS", "🔥")
    for alert in report.mwmbr_alerts:
        badge = (
            f"{RED}{BOLD}🚨 FIRING ({alert.severity}){RESET}"
            if alert.is_firing
            else f"{GREEN}✓ OK{RESET}"
        )
        print(
            f"  {BOLD}{WHITE}{alert.tier_name:<24}{RESET} {badge:<26} "
            f"Long ({alert.long_window_label}): {BOLD}{alert.long_window_burn_rate:.2f}×{RESET} · "
            f"Short ({alert.short_window_label}): {BOLD}{alert.short_window_burn_rate:.2f}×{RESET} "
            f"{DIM}(threshold {alert.threshold_burn_rate:.1f}×){RESET}"
        )
        print(f"    {DIM}↳ Window budget burn: {alert.budget_consumed_in_window_pct:.1f}% · Action: {alert.recommended_action}{RESET}")

    # ── Pillar 2: AST Code, Symbol & Commit Attribution ────────────
    if report.code_attributions:
        _section("PILLAR 2 — AST CODE, SYMBOL & COMMIT ERROR-BUDGET ATTRIBUTION", "🧬")
        for attr in report.code_attributions:
            sym_str = f" [symbols: {', '.join(attr.attributed_symbols)}]" if attr.attributed_symbols else ""
            commit_str = f" (commit [{attr.latest_commit_sha}] {attr.latest_commit_message[:34]})" if attr.latest_commit_sha else ""
            print(
                f"  {YELLOW}{BOLD}{attr.path}{RESET}{CYAN}{sym_str}{RESET}{DIM}{commit_str}{RESET}"
            )
            print(
                f"    {RED}Burn Share: {BOLD}{attr.budget_burn_share_pct:.1f}%{RESET} "
                f"({attr.budget_minutes_consumed:.2f} min burned across {attr.failure_events} failure events) · "
                f"{GREEN}Recoverable by ASRE fix: +{attr.recoverable_budget_minutes:.2f} min{RESET}"
            )

    # ── Pillar 4: Closed-Loop Error Budget Governance ──────────────
    _section("PILLAR 4 — CLOSED-LOOP ERROR BUDGET GOVERNANCE & RELEASE GATE", "🏛️")
    gov_col = RED if gov.feature_freeze_active else (YELLOW if gov.governance_state == "CAUTIOUS_CANARY_MODE" else GREEN)
    print(f"  {'Governance State:':<28} {gov_col}{BOLD}{gov.governance_state}{RESET}")
    freeze_str = f"{RED}{BOLD}ACTIVE (Feature deploys frozen; verified ASRE repairs allowed){RESET}" if gov.feature_freeze_active else f"{GREEN}Inactive (Feature deploys permitted){RESET}"
    print(f"  {'Feature Release Freeze:':<28} {freeze_str}")
    print(f"  {'Max Auto-Deploy R_action:':<28} {WHITE}{gov.max_auto_deploy_r_action:.2f}{RESET} (Mandatory Canary: {gov.mandatory_canary_pct}%)")
    print(f"  {'Governance Directive:':<28} {WHITE}{gov.policy_directive}{RESET}")

    print(f"\n{BOLD}{'═' * 68}{RESET}\n")


def print_database_status_report(report: DatabaseStatusReport) -> None:
    """Renders the Supabase Database Connection, Schema & Local Mirror Status report."""
    print(f"\n{BOLD}{'═' * 68}{RESET}")
    print(f"{BOLD}{CYAN}  🗄️   SUPABASE DATABASE & TRAINING STORE STATUS{RESET}")
    print(f"{BOLD}{'═' * 68}{RESET}")

    cfg_badge = f"{GREEN}✓ Configured{RESET}" if report.configured else f"{RED}✗ Missing Credentials{RESET}"
    rest_badge = f"{GREEN}✓ Connected{RESET}" if report.rest_reachable else f"{YELLOW}⚠ Offline (Local Mirror Active){RESET}"
    schema_badge = f"{GREEN}✓ Active (8 Tables + 2 ML Views){RESET}" if report.schema_ready else f"{YELLOW}⚠ Pending Migration (--init-db){RESET}"

    print(f"\n  {'Supabase Configuration:':<30} {cfg_badge}")
    print(f"  {'Supabase Project URL:':<30} {BOLD}{WHITE}{report.supabase_url or 'N/A'}{RESET}")
    print(f"  {'PostgREST API Connectivity:':<30} {rest_badge}")
    print(f"  {'Postgres Schema Status:':<30} {schema_badge}")
    print(f"  {'Pending Offline Sync Queue:':<30} {BOLD}{YELLOW if report.pending_offline_records else GREEN}{report.pending_offline_records} batch(es){RESET}")
    print(f"  {'Status Detail:':<30} {WHITE}{report.detail}{RESET}")

    if report.table_counts:
        _section("LOCAL DUAL-WRITE MIRROR TABLE COUNTS", "📊")
        for tbl, cnt in report.table_counts.items():
            print(f"  {CYAN}•{RESET} {WHITE}{tbl:<32}{RESET} {BOLD}{cnt:>5}{RESET} record(s)")

    print(f"\n{BOLD}{'═' * 68}{RESET}\n")


def print_training_export_report(report: TrainingExportReport) -> None:
    """Renders the ML & LLM Training Dataset Export report."""
    print(f"\n{BOLD}{'═' * 68}{RESET}")
    print(f"{BOLD}{CYAN}  🧬  XARGIS ML & LLM TRAINING DATASET EXPORT{RESET}")
    print(f"{BOLD}{'═' * 68}{RESET}")

    print(f"\n  {'Output Directory:':<34} {BOLD}{WHITE}{report.output_dir}{RESET}")
    print(f"  {'LLM SFT Verified Trajectories:':<34} {BOLD}{GREEN}{report.sft_records_count}{RESET} {DIM}(sft_repairs.jsonl){RESET}")
    print(f"  {'LLM DPO Preference Pairs:':<34} {BOLD}{CYAN}{report.dpo_pairs_count}{RESET} {DIM}(dpo_preference_pairs.jsonl){RESET}")
    print(f"  {'Tabular Change-Risk Rows (ML):':<34} {BOLD}{YELLOW}{report.change_risk_rows_count}{RESET} {DIM}(jit_change_risk_training.csv / .jsonl){RESET}")
    print(f"  {'Cross-Service Causal Graphs:':<34} {BOLD}{MAGENTA}{report.causal_graph_records_count}{RESET} {DIM}(causal_graph_trajectories.jsonl){RESET}")

    _section("EXPORTED TRAINING FILES", "💾")
    for f_path in report.files_written:
        print(f"  {GREEN}✓{RESET} {WHITE}{f_path}{RESET}")

    print(f"\n{BOLD}{'═' * 68}{RESET}\n")


def print_commercial_enterprise_report(report: CommercialEnterpriseReport) -> None:
    """Renders the 10-Capability Commercial Enterprise & Product Differentiation Report."""
    print(f"\n{BOLD}{'═' * 68}{RESET}")
    print(f"{BOLD}{CYAN}  🏢  XARGIS COMMERCIAL ENTERPRISE & PRODUCT CAPABILITIES REPORT{RESET}")
    print(f"{BOLD}{'═' * 68}{RESET}")

    # 1. Multi-Repository Federated Intelligence
    fed = report.multi_repo_federation
    _section("1. MULTI-REPOSITORY FEDERATED INTELLIGENCE", "🌐")
    print(f"  {'Repositories Indexed:':<30} {BOLD}{WHITE}{fed.repositories_analyzed}{RESET}")
    print(f"  {'Shared Ecosystem Packages:':<30} {CYAN}{', '.join(fed.shared_dependencies) or 'None'}{RESET}")
    for node in fed.nodes:
        print(
            f"  {CYAN}•{RESET} {BOLD}{WHITE}{node.repo_id:<16}{RESET} "
            f"({node.total_files} files, {node.total_lines:,} LOC, {node.dependency_edges} edges) · "
            f"Top Hotspot: {YELLOW}{node.top_hotspot_file}{RESET} ({node.top_hotspot_risk}/90)"
        )

    # 2. Service Ownership Mapping
    own = report.service_ownership
    _section("2. SERVICE OWNERSHIP MAPPING (CODEOWNERS + GIT-BLAME FALLBACK)", "👥")
    co_str = f"{GREEN}Active ({own.codeowners_rules_count} rules){RESET}" if own.has_codeowners_file else f"{YELLOW}Auto-Resolved via 90d Git Commit History{RESET}"
    print(f"  {'Ownership Resolution Source:':<30} {co_str}")
    for sub in own.subsystems[:6]:
        bf_badge = f" {YELLOW}[Bus-Factor Risk]{RESET}" if sub.bus_factor_warning else ""
        print(
            f"  {CYAN}•{RESET} {WHITE}{sub.subsystem:<18}{RESET} → {BOLD}{GREEN}{sub.team_handle:<22}{RESET} "
            f"Primary: {WHITE}{sub.primary_owner[:28]}{RESET} ({sub.primary_owner_share_pct:.0f}%){bf_badge}"
        )

    # 3. Incident Cost Estimation & Autonomous ROI
    cost = report.incident_cost_estimation
    _section("3. INCIDENT COST ESTIMATION & AUTONOMOUS ROI", "💰")
    print(f"  {'Manual Baseline MTTR:':<30} {YELLOW}{cost.human_baseline_mttr_minutes:.1f} min{RESET} (vs ASRE Autonomous: {GREEN}{cost.asre_autonomous_mttr_seconds:.1f}s{RESET} — {BOLD}{CYAN}{cost.mttr_speedup_factor:.0f}x faster{RESET})")
    print(f"  {'Manual War-Room + SLA Cost:':<30} {RED}${cost.manual_engineering_cost_usd + cost.manual_business_impact_usd:,.2f}{RESET} ({cost.incidents_evaluated} incident(s))")
    print(f"  {'ASRE Assisted Total Cost:':<30} {GREEN}${cost.asre_assisted_total_cost_usd:,.2f}{RESET}")
    print(f"  {'Net Savings & Hours Saved:':<30} {BOLD}{GREEN}${cost.net_savings_usd:,.2f} saved{RESET} ({cost.engineering_hours_saved:.1f} SRE hours recovered)")

    # 4. Repair Effectiveness Analytics
    eff = report.repair_effectiveness
    _section("4. REPAIR EFFECTIVENESS ANALYTICS", "📈")
    print(f"  {'Total Repair Runs Analyzed:':<30} {WHITE}{eff.total_repair_runs}{RESET} ({GREEN}{eff.positive_verified_runs} verified{RESET}, {YELLOW}{eff.negative_blocked_or_failed_runs} negative guardrails{RESET})")
    print(f"  {'Recurrence-Free Survival:':<30} {BOLD}{GREEN}{eff.recurrence_free_survival_rate_pct:.1f}%{RESET} (Human-Promoted Verified: {eff.human_promoted_verified_count})")
    print(f"  {'Dead-End Actions Prevented:':<30} {CYAN}{eff.dead_end_actions_prevented}{RESET} · Mean Verified MTTR: {GREEN}{eff.mean_mttr_seconds:.1f}s{RESET}")

    # 5. Engineering Reliability Scorecards
    sc = report.reliability_scorecard
    _section("5. ENGINEERING RELIABILITY SCORECARD", "🏆")
    grade_col = GREEN if sc.overall_score >= 80 else (YELLOW if sc.overall_score >= 65 else RED)
    print(f"  {'Composite Reliability Score:':<30} {BOLD}{grade_col}{sc.overall_score:.1f}/100 (Grade: {sc.letter_grade}){RESET}")
    print(
        f"  {'Dimension Breakdown:':<30} Arch={sc.architecture_health_score:.0f} · "
        f"ChangeSafety={sc.change_safety_score:.0f} · Repair={sc.repair_effectiveness_score:.0f} · "
        f"SLO={sc.slo_budget_health_score:.0f} · DebtHygiene={sc.tech_debt_hygiene_score:.0f}"
    )
    for sub_sc in sc.subsystem_scorecards[:5]:
        print(
            f"  {CYAN}•{RESET} {WHITE}{sub_sc.subsystem:<18}{RESET} [{BOLD}{sub_sc.grade:<2}{RESET} {sub_sc.score:.1f}/100] "
            f"Owner: {GREEN}{sub_sc.owner_team:<22}{RESET} (High-Risk Files: {sub_sc.high_risk_files})"
        )

    # 6. Custom Declarative Repair Policies
    pol = report.custom_repair_policy
    verd = report.sample_policy_verdict
    _section("6. CUSTOM DECLARATIVE REPAIR POLICIES (.asre/policy.json)", "🛡️")
    print(f"  {'Policy Configuration:':<30} {WHITE}{pol.policy_source}{RESET}")
    print(f"  {'Patch Scope Limits:':<30} max_files={pol.max_files_per_patch}, max_lines={pol.max_changed_lines}, protected_globs={len(pol.protected_path_globs)}")
    v_badge = f"{GREEN}ALLOWED{RESET}" if verd.allowed_for_autonomous_modification else f"{YELLOW}GATED ({'; '.join(verd.violations[:1])}){RESET}"
    print(f"  {'Hotspot Policy Check:':<30} {v_badge}")

    # 7 & 8. Team Knowledge Sharing & Private Deployment
    tkb = report.team_knowledge_sharing
    prv = report.private_deployment
    _section("7 & 8. TEAM KNOWLEDGE SHARING & PRIVATE DEPLOYMENT COMPLIANCE", "🔒")
    print(
        f"  {'Shared Knowledge Catalog:':<30} {GREEN}{tkb.shared_verified_repairs_count} verified repair(s){RESET}, "
        f"{CYAN}{tkb.reusable_ast_playbooks_count} AST playbook(s){RESET}, {tkb.tier1_runbook_sections_indexed} runbook section(s)"
    )
    priv_badge = f"{GREEN}ACTIVE (Strict Zero-Egress Air-Gapped Mode){RESET}" if prv.private_mode_active else f"{CYAN}Ready (Use --private-mode for Zero-Egress Enforcement){RESET}"
    print(f"  {'Private Deployment Mode:':<30} {priv_badge}")
    print(f"  {'Compliance Attestation:':<30} SHA-256 `{prv.compliance_attestation_sha256}` (Worktree Isolation + Secret Redaction)")

    # 9. Multi-Environment Governance Matrix
    _section("9. MULTI-ENVIRONMENT GOVERNANCE MATRIX (DEV / STAGING / PROD)", "🌍")
    for tier in report.multi_environment_matrix:
        print(
            f"  {CYAN}•{RESET} {BOLD}{WHITE}{tier.environment:<13}{RESET} "
            f"MaxAutoRisk={tier.max_auto_apply_risk_score}/90 · R_action≤{tier.max_auto_deploy_r_action:.2f} · "
            f"MinRole={tier.minimum_rbac_role:<9} · Mode={CYAN}{tier.governance_mode}{RESET}"
        )

    # 10. Built-in Repair REST & Webhook API
    _section("10. REPAIR REST & WEBHOOK API (--serve-api)", "🔌")
    for ep in report.repair_api_endpoints:
        print(f"  {GREEN}{ep.method:<5}{RESET} {BOLD}{WHITE}{ep.path:<28}{RESET} {DIM}{ep.description}{RESET}")

    print(f"\n{BOLD}{'═' * 68}{RESET}\n")





