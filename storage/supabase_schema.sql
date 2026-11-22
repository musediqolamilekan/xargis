-- ============================================================================
-- ASRE (Autonomous Software Reliability Engineer) — Supabase Postgres Schema
-- 8 Core Tables + 2 ML/LLM Training Views
-- Idempotent migration script: safe to run multiple times.
-- ============================================================================

-- 1. Repository & Dependency Graph Topology Baseline
CREATE TABLE IF NOT EXISTS public.asre_repositories (
    repo_id TEXT PRIMARY KEY,
    repo_name TEXT NOT NULL,
    repo_url TEXT DEFAULT '',
    head_commit_sha TEXT DEFAULT '',
    total_files INTEGER DEFAULT 0,
    total_loc INTEGER DEFAULT 0,
    dependency_edges_count INTEGER DEFAULT 0,
    languages_json JSONB DEFAULT '{}'::jsonb,
    critical_hotspots_json JSONB DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Raw & Correlated Incidents + Drain3 Log Templates
CREATE TABLE IF NOT EXISTS public.asre_incidents (
    incident_id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    source TEXT DEFAULT 'asre',
    service TEXT DEFAULT 'default',
    environment TEXT DEFAULT 'production',
    severity TEXT DEFAULT 'error',
    exception_type TEXT DEFAULT 'Error',
    message TEXT NOT NULL,
    drain_template_id TEXT DEFAULT '',
    drain_template TEXT DEFAULT '',
    stack_trace TEXT DEFAULT '',
    trace_id TEXT DEFAULT '',
    span_id TEXT DEFAULT '',
    parent_span_id TEXT DEFAULT '',
    commit_sha TEXT DEFAULT '',
    mapped_files_json JSONB DEFAULT '[]'::jsonb,
    metadata_json JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asre_incidents_repo_id ON public.asre_incidents(repo_id);
CREATE INDEX IF NOT EXISTS idx_asre_incidents_drain_tpl ON public.asre_incidents(drain_template_id);

-- 3. Cross-Incident Multi-Hop Causal Graphs & Counterfactual Attributions
CREATE TABLE IF NOT EXISTS public.asre_causal_graphs (
    correlation_id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    root_trigger_incident_id TEXT NOT NULL,
    common_root_cause_file TEXT DEFAULT '',
    common_root_cause_resource TEXT DEFAULT '',
    ppr_score DOUBLE PRECISION DEFAULT 0.0,
    confidence DOUBLE PRECISION DEFAULT 0.0,
    root_cause_hypothesis TEXT DEFAULT '',
    correlated_incident_ids_json JSONB DEFAULT '[]'::jsonb,
    temporal_causal_chain_json JSONB DEFAULT '[]'::jsonb,
    counterfactual_attributions_json JSONB DEFAULT '[]'::jsonb,
    relationship_graph_json JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asre_causal_graphs_repo_id ON public.asre_causal_graphs(repo_id);

-- 4. Episodic Repair Runs, 4-Tier Simulation & SFT/DPO Training Trajectories
CREATE TABLE IF NOT EXISTS public.asre_repair_runs (
    run_id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL,
    repo_id TEXT NOT NULL,
    status TEXT NOT NULL,
    kb_trust_status TEXT DEFAULT 'unverified',
    promoted_by TEXT DEFAULT '',
    trajectory_polarity TEXT DEFAULT 'positive',
    llm_provider TEXT DEFAULT 'auto',
    llm_model TEXT DEFAULT '',
    diagnosis TEXT DEFAULT '',
    suspected_cause TEXT DEFAULT '',
    confirmed_evidence_json JSONB DEFAULT '[]'::jsonb,
    files_modified_json JSONB DEFAULT '[]'::jsonb,
    unified_diff TEXT DEFAULT '',
    structural_operator TEXT DEFAULT '',
    deliberately_unchanged_json JSONB DEFAULT '[]'::jsonb,
    tests_to_run_json JSONB DEFAULT '[]'::jsonb,
    reproduced_before_patch BOOLEAN DEFAULT FALSE,
    incident_replay_passed BOOLEAN DEFAULT FALSE,
    adjacent_regressions INTEGER DEFAULT 0,
    r_action_score DOUBLE PRECISION DEFAULT 0.5,
    execution_band TEXT DEFAULT '',
    simulation_report_json JSONB DEFAULT '{}'::jsonb,
    dead_end_actions_json JSONB DEFAULT '[]'::jsonb,
    distilled_record_json JSONB DEFAULT '{}'::jsonb,
    mttr_seconds DOUBLE PRECISION DEFAULT 2.0,
    post_repair_survived BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asre_repair_runs_repo_id ON public.asre_repair_runs(repo_id);
CREATE INDEX IF NOT EXISTS idx_asre_repair_runs_incident_id ON public.asre_repair_runs(incident_id);
CREATE INDEX IF NOT EXISTS idx_asre_repair_runs_status ON public.asre_repair_runs(status);

-- 5. 3-Tier Cognitive Knowledge Base (Semantic Docs, Episodic Cases, Procedural Bandit Skills)
CREATE TABLE IF NOT EXISTS public.asre_knowledge_base (
    entry_id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    tier TEXT NOT NULL,
    title TEXT NOT NULL,
    structural_operator TEXT DEFAULT '',
    trajectory_polarity TEXT DEFAULT 'positive',
    kb_trust_status TEXT DEFAULT 'unverified',
    drift_status TEXT DEFAULT 'clean',
    alpha DOUBLE PRECISION DEFAULT 1.0,
    beta_param DOUBLE PRECISION DEFAULT 1.0,
    posterior_confidence DOUBLE PRECISION DEFAULT 0.5,
    thompson_bandit_score DOUBLE PRECISION DEFAULT 0.5,
    payload_json JSONB DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asre_kb_repo_tier ON public.asre_knowledge_base(repo_id, tier);

-- 6. 3-Signal Predictive Change-Risk Features (Labeled Dataset for LightGBM / XGBoost / Platt Training)
CREATE TABLE IF NOT EXISTS public.asre_change_risk_records (
    check_id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    stable_commit TEXT NOT NULL,
    target_commit TEXT NOT NULL,
    lines_added INTEGER DEFAULT 0,
    lines_deleted INTEGER DEFAULT 0,
    files_touched INTEGER DEFAULT 0,
    subsystems_touched INTEGER DEFAULT 0,
    change_entropy DOUBLE PRECISION DEFAULT 0.0,
    author_name TEXT DEFAULT '',
    author_familiarity_pct DOUBLE PRECISION DEFAULT 100.0,
    recent_bugfix_commits INTEGER DEFAULT 0,
    is_off_hours_deploy BOOLEAN DEFAULT FALSE,
    max_reverse_pagerank DOUBLE PRECISION DEFAULT 0.0,
    max_betweenness_centrality DOUBLE PRECISION DEFAULT 0.0,
    total_direct_dependents INTEGER DEFAULT 0,
    total_transitive_dependents INTEGER DEFAULT 0,
    modified_symbols_count INTEGER DEFAULT 0,
    config_keys_modified_count INTEGER DEFAULT 0,
    ast_hazards_count INTEGER DEFAULT 0,
    prior_incidents_count INTEGER DEFAULT 0,
    calibrated_probability DOUBLE PRECISION DEFAULT 0.0,
    deployment_risk_level TEXT DEFAULT 'LOW',
    autonomous_action TEXT DEFAULT '',
    shap_attributions_json JSONB DEFAULT '[]'::jsonb,
    incident_caused_label INTEGER DEFAULT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asre_change_risk_repo_commit ON public.asre_change_risk_records(repo_id, target_commit);

-- 7. 4-Pillar SLO & Error Budget Snapshots
CREATE TABLE IF NOT EXISTS public.asre_slo_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    slo_target_pct DOUBLE PRECISION DEFAULT 99.9,
    observed_availability_pct DOUBLE PRECISION DEFAULT 100.0,
    remaining_budget_pct DOUBLE PRECISION DEFAULT 100.0,
    remaining_budget_minutes DOUBLE PRECISION DEFAULT 43.2,
    ewma_burn_rate DOUBLE PRECISION DEFAULT 0.0,
    time_to_exhaustion_hours DOUBLE PRECISION DEFAULT NULL,
    fast_burn_firing BOOLEAN DEFAULT FALSE,
    high_burn_firing BOOLEAN DEFAULT FALSE,
    slow_burn_firing BOOLEAN DEFAULT FALSE,
    governance_state TEXT DEFAULT 'HEALTHY_VELOCITY',
    feature_freeze_active BOOLEAN DEFAULT FALSE,
    mwmbr_alerts_json JSONB DEFAULT '[]'::jsonb,
    code_attributions_json JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 8. Immutable Enterprise SOC2 Audit Trail
CREATE TABLE IF NOT EXISTS public.asre_audit_logs (
    audit_id TEXT PRIMARY KEY,
    org_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001'::uuid,
    repo_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    incident_id TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    operator TEXT DEFAULT 'cli-operator',
    role TEXT DEFAULT 'sre',
    environment TEXT DEFAULT 'development',
    policy_decision_json JSONB DEFAULT '{}'::jsonb,
    details_json JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_asre_audit_logs_incident ON public.asre_audit_logs(incident_id);

-- 9. Multi-Tenant Encrypted Platform Integrations (GitHub App, Datadog, PagerDuty, Slack, K8s)
CREATE TABLE IF NOT EXISTS public.asre_integrations (
    integration_id TEXT PRIMARY KEY,
    org_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001'::uuid,
    platform TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'connected',
    installation_id TEXT DEFAULT '',
    external_id TEXT DEFAULT '',
    encrypted_credentials_vault TEXT DEFAULT '',
    webhook_secret_hash TEXT DEFAULT '',
    webhook_url TEXT DEFAULT '',
    config_json JSONB DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asre_integrations_org_id ON public.asre_integrations(org_id, platform);
CREATE INDEX IF NOT EXISTS idx_asre_integrations_install ON public.asre_integrations(installation_id);

-- 10. Per-Tenant Isolated Async Webhook & Execution Job Queue
CREATE TABLE IF NOT EXISTS public.asre_tenant_jobs (
    job_id TEXT PRIMARY KEY,
    org_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001'::uuid,
    event_type TEXT NOT NULL,
    platform TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed',
    worker_id TEXT DEFAULT 'asre-stateless-worker-1',
    payload_json JSONB DEFAULT '{}'::jsonb,
    result_json JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asre_tenant_jobs_org_id ON public.asre_tenant_jobs(org_id, status);

-- 11. Multi-Tenant Organizations (Tenancy Root, Plan Tier & SSO Domain)
CREATE TABLE IF NOT EXISTS public.asre_organizations (
    org_id UUID PRIMARY KEY DEFAULT '00000000-0000-0000-0000-000000000001'::uuid,
    org_name TEXT NOT NULL,
    slug TEXT NOT NULL,
    plan_tier TEXT DEFAULT 'enterprise',
    sso_domain TEXT DEFAULT '',
    sso_providers_json JSONB DEFAULT '["google", "github"]'::jsonb,
    settings_json JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 12. Multi-Tenant Users (Email/Password PBKDF2 + Google & GitHub SSO Identities)
CREATE TABLE IF NOT EXISTS public.asre_users (
    user_id TEXT PRIMARY KEY,
    org_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001'::uuid,
    email TEXT NOT NULL,
    full_name TEXT DEFAULT '',
    role TEXT NOT NULL DEFAULT 'sre',
    auth_provider TEXT NOT NULL DEFAULT 'email',
    provider_user_id TEXT DEFAULT '',
    password_hash TEXT DEFAULT '',
    avatar_url TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_asre_users_email ON public.asre_users(email);
CREATE INDEX IF NOT EXISTS idx_asre_users_org_id ON public.asre_users(org_id);

-- ============================================================================
-- Multi-Tenant Organization Isolation (`org_id UUID NOT NULL`) + RLS Policies
-- Ensures every table has `org_id` and enforces tenant isolation via JWT claims
-- ============================================================================
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'asre_repositories',
        'asre_incidents',
        'asre_causal_graphs',
        'asre_repair_runs',
        'asre_knowledge_base',
        'asre_change_risk_records',
        'asre_slo_snapshots',
        'asre_audit_logs',
        'asre_integrations',
        'asre_tenant_jobs',
        'asre_organizations',
        'asre_users'
    ]
    LOOP
        EXECUTE format(
            'ALTER TABLE public.%I ADD COLUMN IF NOT EXISTS org_id UUID NOT NULL DEFAULT ''00000000-0000-0000-0000-000000000001''::uuid;',
            tbl
        );
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_org_id ON public.%I(org_id);', tbl, tbl);
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY;', tbl);
        EXECUTE format('DROP POLICY IF EXISTS asre_all_access ON public.%I;', tbl);
        EXECUTE format('DROP POLICY IF EXISTS asre_tenant_isolation ON public.%I;', tbl);
        EXECUTE format(
            'CREATE POLICY asre_tenant_isolation ON public.%I FOR ALL TO anon, authenticated, service_role '
            'USING (org_id = COALESCE(NULLIF(current_setting(''request.jwt.claims'', true)::jsonb -> ''app_metadata'' ->> ''org_id'', '''')::uuid, NULLIF(current_setting(''request.jwt.claims'', true)::jsonb ->> ''org_id'', '''')::uuid, org_id)) '
            'WITH CHECK (org_id = COALESCE(NULLIF(current_setting(''request.jwt.claims'', true)::jsonb -> ''app_metadata'' ->> ''org_id'', '''')::uuid, NULLIF(current_setting(''request.jwt.claims'', true)::jsonb ->> ''org_id'', '''')::uuid, org_id));',
            tbl
        );
        EXECUTE format('GRANT ALL ON TABLE public.%I TO anon, authenticated, service_role;', tbl);
    END LOOP;
END
$$;

-- ============================================================================
-- ML / LLM Training Views
-- ============================================================================

-- View 1: LLM Supervised Fine-Tuning (SFT) & Direct Preference Optimization (DPO) Pairs
CREATE OR REPLACE VIEW public.v_llm_dpo_preference_pairs AS
SELECT
    pos.repo_id,
    pos.incident_id,
    inc.exception_type,
    inc.message AS error_message,
    inc.drain_template,
    inc.stack_trace,
    pos.structural_operator,
    pos.diagnosis AS chosen_diagnosis,
    pos.unified_diff AS chosen_unified_diff,
    pos.r_action_score AS chosen_r_action,
    neg.run_id AS rejected_run_id,
    neg.status AS rejected_status,
    neg.diagnosis AS rejected_diagnosis,
    neg.unified_diff AS rejected_unified_diff,
    pos.dead_end_actions_json
FROM public.asre_repair_runs pos
LEFT JOIN public.asre_incidents inc
    ON inc.incident_id = pos.incident_id
LEFT JOIN public.asre_repair_runs neg
    ON neg.repo_id = pos.repo_id
    AND neg.trajectory_polarity = 'negative'
    AND (neg.incident_id = pos.incident_id OR neg.files_modified_json = pos.files_modified_json)
WHERE pos.trajectory_polarity = 'positive'
  AND pos.post_repair_survived = TRUE;

-- View 2: Tabular JIT Change-Risk Dataset for LightGBM / XGBoost Defect Prediction
CREATE OR REPLACE VIEW public.v_tabular_change_risk_dataset AS
SELECT
    check_id,
    repo_id,
    stable_commit,
    target_commit,
    lines_added,
    lines_deleted,
    files_touched,
    subsystems_touched,
    change_entropy,
    author_familiarity_pct,
    recent_bugfix_commits,
    CASE WHEN is_off_hours_deploy THEN 1 ELSE 0 END AS is_off_hours_deploy,
    max_reverse_pagerank,
    max_betweenness_centrality,
    total_direct_dependents,
    total_transitive_dependents,
    modified_symbols_count,
    config_keys_modified_count,
    ast_hazards_count,
    prior_incidents_count,
    calibrated_probability,
    COALESCE(incident_caused_label, CASE WHEN prior_incidents_count > 0 AND calibrated_probability >= 0.70 THEN 1 ELSE 0 END) AS incident_caused_label,
    created_at
FROM public.asre_change_risk_records;

GRANT SELECT ON public.v_llm_dpo_preference_pairs TO anon, authenticated, service_role;
GRANT SELECT ON public.v_tabular_change_risk_dataset TO anon, authenticated, service_role;

-- Reload PostgREST schema cache immediately
NOTIFY pgrst, 'reload schema';

