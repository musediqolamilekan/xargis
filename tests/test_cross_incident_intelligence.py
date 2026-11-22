import json
import os
import tempfile
import unittest
from pathlib import Path

from intelligence.dependency_graph import build_dependency_graph
from intelligence.file_profile import build_file_profiles
from intelligence.scanner import scan_repository
from repair.cross_incident_intelligence import (
    DrainTemplateMiner,
    analyze_cross_incident_intelligence,
    compute_personalized_pagerank,
    load_incidents_from_inputs,
)
from repair.incident_mapper import Incident


class TestCrossIncidentIntelligence(unittest.TestCase):
    def _create_microservices_repo(self) -> tempfile.TemporaryDirectory:
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        (root / "db").mkdir()
        (root / "orders").mkdir()
        (root / "payments").mkdir()

        # Shared downstream database connection pool module
        (root / "db" / "pool.py").write_text(
            "import os\nimport sqlite3\n\n"
            "MAX_POOL_CONNECTIONS = int(os.environ.get('DB_POOL_SIZE', '5'))\n\n"
            "def acquire_connection():\n"
            "    conn = sqlite3.connect(':memory:')\n"
            "    conn.execute('SELECT 1')\n"
            "    return conn\n",
            encoding="utf-8",
        )
        # Order creation service imports db.pool
        (root / "orders" / "service.py").write_text(
            "from db.pool import acquire_connection\n\n"
            "def create_order(order_id: int) -> dict:\n"
            "    conn = acquire_connection()\n"
            "    conn.execute('INSERT INTO orders VALUES (?)', (order_id,))\n"
            "    return {'order_id': order_id}\n",
            encoding="utf-8",
        )
        # Payment service imports db.pool
        (root / "payments" / "gateway.py").write_text(
            "from db.pool import acquire_connection\n\n"
            "def process_payment(payment_id: str) -> bool:\n"
            "    conn = acquire_connection()\n"
            "    conn.execute('SELECT status FROM payments')\n"
            "    return True\n",
            encoding="utf-8",
        )
        return temp_dir

    def test_drain_template_miner_masks_dynamic_parameters(self):
        miner = DrainTemplateMiner()
        t1 = miner.extract_template("Payment request 550e8400-e29b-41d4-a716-446655440000 timed out after 5000ms on 10.0.0.12:5432")
        t2 = miner.extract_template("Payment request 123e4567-e89b-12d3-a456-426614174000 timed out after 3200ms on 10.0.0.19:5432")
        self.assertEqual(t1.template_id, t2.template_id)
        self.assertIn("<*>", t1.template)
        self.assertGreaterEqual(t1.parameter_count, 3)

    def test_correlates_payment_timeout_order_failure_and_db_pool_into_one_root_cause(self):
        temp_dir = self._create_microservices_repo()
        self.addCleanup(temp_dir.cleanup)

        intel = scan_repository(temp_dir.name, verbose=False)
        graph = build_dependency_graph(intel)
        profiles = build_file_profiles(graph, intel, {}, {})

        # Create a historical repair run to verify Layer A Hybrid Graph-RAG RRF comparison
        hist_run_dir = Path(temp_dir.name) / ".asre" / "runs" / "hist-db-pool-001"
        hist_run_dir.mkdir(parents=True)
        (hist_run_dir / "run.json").write_text(
            json.dumps(
                {
                    "incident_id": "hist-db-pool-001",
                    "status": "verified",
                    "lesson": {
                        "exception_type": "OperationalError",
                        "message": "Database connection pool exhausted: too many open SQL connections",
                        "diagnosis": "Connection leak in db/pool.py exhausted MAX_POOL_CONNECTIONS under load",
                        "files_modified": ["db/pool.py"],
                        "tests_to_run": ["python -m unittest discover -s tests"],
                    },
                }
            ),
            encoding="utf-8",
        )

        # Incident A: Payment requests timing out (t = +8s)
        incident_a = Incident(
            incident_id="incident-A-payment-timeout",
            exception_type="TimeoutError",
            message="Payment request timed out after 5000ms waiting for transaction lock",
            stack_trace='  File "payments/gateway.py", line 5, in process_payment\n    conn = acquire_connection()',
            environment={"service": "payment-service", "env": "production"},
            metadata={"timestamp": "2026-11-21T12:00:08Z"},
        )
        # Incident B: Order creation failing (t = +3s)
        incident_b = Incident(
            incident_id="incident-B-order-failure",
            exception_type="RuntimeError",
            message="Order creation failed: unable to persist order 99812",
            stack_trace='  File "orders/service.py", line 5, in create_order\n    conn = acquire_connection()',
            environment={"service": "order-service", "env": "production"},
            metadata={"timestamp": "2026-11-21T12:00:03Z"},
        )
        # Incident C: Database connection errors (t = 0s, root trigger)
        incident_c = Incident(
            incident_id="incident-C-db-connection-error",
            exception_type="OperationalError",
            message="Database connection pool exhausted: max_connections=5 reached",
            stack_trace='  File "db/pool.py", line 7, in acquire_connection\n    conn = sqlite3.connect(":memory:")',
            environment={"service": "database-pool", "env": "production"},
            metadata={"timestamp": "2026-11-21T12:00:00Z"},
        )

        report = analyze_cross_incident_intelligence(
            [incident_a, incident_b, incident_c],
            graph,
            profiles,
            temp_dir.name,
        )

        self.assertEqual(3, report.total_incidents_analyzed)
        self.assertEqual(1, report.correlated_root_causes)
        group = report.correlated_groups[0]

        # Layer B: Common root cause isolated to db/pool.py & AST-discovered shared symbol/resource via PPR + Reverse BFS
        self.assertEqual("db/pool.py", group.common_root_cause_file)
        self.assertIn("db/pool.py:acquire_connection", group.common_root_cause_resource or "")
        self.assertGreater(group.ppr_score, 0.0)
        self.assertIn("db/pool.py", group.root_cause_hypothesis)

        # Layer C: Temporal lead-lag causal ordering identifies Incident C (t=0s) as Root Trigger
        self.assertEqual("incident-C-db-connection-error", group.root_trigger_incident_id)
        self.assertEqual(2, len(group.temporal_causal_chain))
        effects = {link.effect_incident_id: link for link in group.temporal_causal_chain}
        self.assertIn("incident-B-order-failure", effects)
        self.assertIn("incident-A-payment-timeout", effects)
        self.assertEqual(3.0, effects["incident-B-order-failure"].lag_seconds)
        self.assertEqual(8.0, effects["incident-A-payment-timeout"].lag_seconds)

        # Incident Relationship Graph contains incident, code_module, and shared_resource nodes
        node_kinds = {n.kind for n in report.relationship_graph.nodes}
        self.assertIn("incident", node_kinds)
        self.assertIn("code_module", node_kinds)
        self.assertIn("shared_resource", node_kinds)
        edge_relations = {e.relation for e in report.relationship_graph.edges}
        self.assertIn("shares_dependency", edge_relations)
        self.assertIn("temporal_causal_cascade", edge_relations)

        # Layer A: Historical Hybrid Graph-RAG RRF retrieves hist-db-pool-001
        self.assertEqual(1, len(report.historical_comparisons))
        hist = report.historical_comparisons[0]
        self.assertEqual("hist-db-pool-001", hist.historical_incident_id)
        self.assertIn("db/pool.py", hist.shared_files)
        self.assertGreater(hist.similarity_pct, 50.0)

    def test_domain_neutral_correlation_on_custom_non_standard_errors_and_llm_synthesizer(self):
        """
        Verify that completely novel, non-DB, non-HTTP error classes (e.g., an audio DSP
        pipeline with custom exception names like 'StemExportAborted', 'SpectrogramNaNDetected',
        and 'FFTWindowBufferUnderflow') correlate purely through AST dependency topology,
        shared symbol invocation, subword morphology, and optional LLM hypothesis synthesis.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dsp").mkdir()
            (root / "exporter").mkdir()
            (root / "analyzer").mkdir()

            (root / "dsp" / "fft_kernel.py").write_text(
                "def compute_stft_window(samples):\n"
                "    return [x * 0.5 for x in samples]\n",
                encoding="utf-8",
            )
            (root / "exporter" / "stem_writer.py").write_text(
                "from dsp.fft_kernel import compute_stft_window\n\n"
                "def bounce_audio_stem(track):\n"
                "    return compute_stft_window(track)\n",
                encoding="utf-8",
            )
            (root / "analyzer" / "spectrum.py").write_text(
                "from dsp.fft_kernel import compute_stft_window\n\n"
                "def render_spectrogram(clip):\n"
                "    return compute_stft_window(clip)\n",
                encoding="utf-8",
            )

            intel = scan_repository(tmp, verbose=False)
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel, {}, {})

            inc_export = Incident(
                incident_id="inc-stem-export",
                exception_type="StemExportAborted",
                message="Audio bounce terminated due to corrupted frame buffer",
                stack_trace='  File "exporter/stem_writer.py", line 4, in bounce_audio_stem\n    return compute_stft_window(track)',
                environment={"service": "stem-exporter"},
                metadata={"timestamp": "2026-11-21T15:00:05Z"},
            )
            inc_spectrum = Incident(
                incident_id="inc-spectrogram-nan",
                exception_type="SpectrogramNaNDetected",
                message="Non-finite energy bin encountered during frequency analysis",
                stack_trace='  File "analyzer/spectrum.py", line 4, in render_spectrogram\n    return compute_stft_window(clip)',
                environment={"service": "spectrum-analyzer"},
                metadata={"timestamp": "2026-11-21T15:00:02Z"},
            )
            inc_fft = Incident(
                incident_id="inc-fft-underflow",
                exception_type="FFTWindowBufferUnderflow",
                message="Window frame length 0 in STFT kernel",
                stack_trace='  File "dsp/fft_kernel.py", line 2, in compute_stft_window\n    return [x * 0.5 for x in samples]',
                environment={"service": "dsp-kernel"},
                metadata={"timestamp": "2026-11-21T15:00:00Z"},
            )

            report = analyze_cross_incident_intelligence(
                [inc_export, inc_spectrum, inc_fft],
                graph,
                profiles,
                tmp,
                llm_synthesizer=lambda prompt: "LLM-verified: Zero-length STFT window in dsp/fft_kernel.py caused NaN spectrogram bins and aborted stem export.",
            )

            self.assertEqual(1, report.correlated_root_causes)
            grp = report.correlated_groups[0]
            self.assertEqual("dsp/fft_kernel.py", grp.common_root_cause_file)
            self.assertEqual("symbol:dsp/fft_kernel.py:compute_stft_window", grp.common_root_cause_resource)
            self.assertEqual("inc-fft-underflow", grp.root_trigger_incident_id)
            self.assertIn("LLM-verified: Zero-length STFT window", grp.root_cause_hypothesis)

    def test_multi_hop_cross_service_causal_dag_otel_spans_rpc_and_counterfactual_attribution(self):
        """
        Verify a 4-tier multi-hop microservice cascade:
          api-gateway (span S1, t=+6s)
             --calls--> order-service (span S2, parent=S1, t=+4s)
                --calls HTTP /v1/inventory/reserve--> inventory-service (span S3, parent=S2, t=+1.5s)
                   --imports--> ledger/storage.py (span S4, parent=S3, t=0s)
        Fault propagates in reverse:
          ledger-storage (inc-4) -> inventory-service (inc-3) -> order-service (inc-2) -> api-gateway (inc-1)
        And PC transitive reduction prunes shortcut edges (e.g., inc-4 -> inc-1) so the exact
        3-edge multi-hop chain is reconstructed, with 100% counterfactual resolution attributed to ledger-storage.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ledger").mkdir()
            (root / "inventory").mkdir()
            (root / "orders").mkdir()
            (root / "gateway").mkdir()

            (root / "ledger" / "storage.py").write_text(
                "def commit_shard_entry(entry):\n"
                "    raise RuntimeError('shard write quorum lost')\n",
                encoding="utf-8",
            )
            (root / "inventory" / "stock.py").write_text(
                "from ledger.storage import commit_shard_entry\n\n"
                "def reserve_stock(sku):\n"
                "    return commit_shard_entry(sku)\n",
                encoding="utf-8",
            )
            (root / "orders" / "workflow.py").write_text(
                "import requests\n\n"
                "def submit_order(payload):\n"
                "    return requests.post('http://inventory-service/v1/inventory/reserve', json=payload)\n",
                encoding="utf-8",
            )
            (root / "gateway" / "router.py").write_text(
                "import requests\n\n"
                "def route_checkout(req):\n"
                "    return requests.post('http://order-service/v1/orders/create', json=req)\n",
                encoding="utf-8",
            )

            intel = scan_repository(tmp, verbose=False)
            graph = build_dependency_graph(intel)
            profiles = build_file_profiles(graph, intel, {}, {})

            inc_gateway = Incident(
                incident_id="inc-1-gateway",
                exception_type="GatewayTimeout",
                message="Upstream order-service returned 504 on /v1/orders/create",
                stack_trace='  File "gateway/router.py", line 4, in route_checkout\n    return requests.post("http://order-service/v1/orders/create", json=req)',
                environment={"service": "api-gateway"},
                metadata={
                    "timestamp": "2026-11-21T18:00:06Z",
                    "trace_id": "trace-multi-hop-77",
                    "span_id": "span-1",
                    "parent_span_id": "",
                },
            )
            inc_orders = Incident(
                incident_id="inc-2-orders",
                exception_type="ServiceUnavailable",
                message="Failed to reserve inventory via /v1/inventory/reserve",
                stack_trace='  File "orders/workflow.py", line 4, in submit_order\n    return requests.post("http://inventory-service/v1/inventory/reserve", json=payload)',
                environment={"service": "order-service"},
                metadata={
                    "timestamp": "2026-11-21T18:00:04Z",
                    "trace_id": "trace-multi-hop-77",
                    "span_id": "span-2",
                    "parent_span_id": "span-1",
                    "api_endpoints": ["/v1/orders/create"],
                },
            )
            inc_inventory = Incident(
                incident_id="inc-3-inventory",
                exception_type="ReservationFailure",
                message="Inventory reservation aborted when committing ledger shard",
                stack_trace='  File "inventory/stock.py", line 4, in reserve_stock\n    return commit_shard_entry(sku)',
                environment={"service": "inventory-service"},
                metadata={
                    "timestamp": "2026-11-21T18:00:01.500Z",
                    "trace_id": "trace-multi-hop-77",
                    "span_id": "span-3",
                    "parent_span_id": "span-2",
                    "api_endpoints": ["/v1/inventory/reserve"],
                },
            )
            inc_ledger = Incident(
                incident_id="inc-4-ledger",
                exception_type="QuorumLost",
                message="Shard write quorum lost on storage node",
                stack_trace='  File "ledger/storage.py", line 2, in commit_shard_entry\n    raise RuntimeError("shard write quorum lost")',
                environment={"service": "ledger-storage"},
                metadata={
                    "timestamp": "2026-11-21T18:00:00Z",
                    "trace_id": "trace-multi-hop-77",
                    "span_id": "span-4",
                    "parent_span_id": "span-3",
                },
            )

            report = analyze_cross_incident_intelligence(
                [inc_gateway, inc_orders, inc_inventory, inc_ledger],
                graph,
                profiles,
                tmp,
            )

            self.assertEqual(1, report.correlated_root_causes)
            grp = report.correlated_groups[0]
            self.assertEqual("inc-4-ledger", grp.root_trigger_incident_id)

            # Verify PC transitive reduction reconstructed the exact 3-hop chain:
            # inc-4-ledger -> inc-3-inventory -> inc-2-orders -> inc-1-gateway
            edge_pairs = [(l.cause_incident_id, l.effect_incident_id) for l in grp.temporal_causal_chain]
            self.assertEqual(
                [
                    ("inc-4-ledger", "inc-3-inventory"),
                    ("inc-3-inventory", "inc-2-orders"),
                    ("inc-2-orders", "inc-1-gateway"),
                ],
                edge_pairs,
            )

            # Verify Counterfactual Do-Calculus Root-Cause Attribution
            self.assertEqual(4, len(grp.service_causal_attributions))
            top_attr = grp.service_causal_attributions[0]
            self.assertEqual("ledger-storage", top_attr.service)
            self.assertEqual("inc-4-ledger", top_attr.incident_id)
            self.assertEqual(100.0, top_attr.counterfactual_resolution_pct)
            self.assertEqual(4, len(top_attr.causal_propagation_path))

    def test_load_incidents_from_inputs_and_personalized_pagerank(self):
        ppr = compute_personalized_pagerank(
            {"inc:A": {"file:db/pool.py": 2.0}, "inc:B": {"file:db/pool.py": 2.0}},
            {"inc:A": 1.0, "inc:B": 1.0},
        )
        self.assertGreater(ppr["file:db/pool.py"], ppr["inc:A"])

        example_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples",
            "click-001",
            "cross-incidents.json",
        )
        loaded = load_incidents_from_inputs([example_path])
        self.assertEqual(3, len(loaded))


if __name__ == "__main__":
    unittest.main()

