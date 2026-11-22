import json
import tempfile
import unittest
from pathlib import Path

from intelligence.dependency_graph import build_dependency_graph
from intelligence.file_profile import build_file_profiles
from intelligence.models import FileAnalysis, RepoIntelligence
from repair.incident_intelligence import (
    correlate_and_prioritize_incidents,
    load_telemetry_events,
    normalize_telemetry_payload,
    parse_application_logs,
)


class TestIncidentIntelligence(unittest.TestCase):
    def _build_fixture_graph_and_profiles(self, repo_root: str):
        # checkout.py and subscription.py both depend on gateway.py
        intel = RepoIntelligence(
            repo_path=repo_root,
            repo_url="",
            file_analyses=[
                FileAnalysis(
                    path="api/checkout.py",
                    language="python",
                    lines=40,
                    size_bytes=800,
                    functions=["process_checkout"],
                    imports=["from services.gateway import charge_card"],
                ),
                FileAnalysis(
                    path="api/subscription.py",
                    language="python",
                    lines=35,
                    size_bytes=700,
                    functions=["renew_subscription"],
                    imports=["from services.gateway import charge_card"],
                ),
                FileAnalysis(
                    path="services/gateway.py",
                    language="python",
                    lines=60,
                    size_bytes=1200,
                    functions=["charge_card"],
                    imports=[],
                ),
                FileAnalysis(
                    path="utils/health.py",
                    language="python",
                    lines=15,
                    size_bytes=200,
                    functions=["ping"],
                    imports=[],
                ),
            ],
        )
        graph = build_dependency_graph(intel)
        profiles = build_file_profiles(
            graph,
            intel,
            {"services/gateway.py": 8, "api/checkout.py": 2},
            {"services/gateway.py": "2026-11-20"},
        )
        return graph, profiles

    def test_500_multi_provider_alerts_correlate_into_single_root_cause_cluster(self):
        with tempfile.TemporaryDirectory() as repo_root:
            root = Path(repo_root)
            for rel in ("api/checkout.py", "api/subscription.py", "services/gateway.py", "utils/health.py"):
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("\n".join(f"line_{i} = {i}" for i in range(1, 30)) + "\n", encoding="utf-8")

            graph, profiles = self._build_fixture_graph_and_profiles(repo_root)

            # Simulate 500 alerts across Sentry, Datadog, OpenTelemetry, New Relic, and Prometheus
            # all caused by a fault in services/gateway.py:
            payload = [
                # 1. Sentry: 250 checkout errors with trace_id=trace-abc-999
                {
                    "source": "sentry",
                    "event_id": "sentry-checkout-1",
                    "count": 250,
                    "environment": "production",
                    "level": "fatal",
                    "tags": {"service": "checkout-api", "trace_id": "trace-abc-999"},
                    "exception": {
                        "values": [
                            {
                                "type": "GatewayTimeoutError",
                                "value": "charge_card timed out for order 98123",
                                "stacktrace": {
                                    "frames": [
                                        {"filename": "api/checkout.py", "lineno": 14, "function": "process_checkout"}
                                    ]
                                },
                            }
                        ]
                    },
                },
                # 2. Datadog: 150 subscription renewal errors (converges on services/gateway.py via DependencyGraph!)
                {
                    "ddsource": "datadog",
                    "id": "dd-sub-1",
                    "count": 150,
                    "attributes": {
                        "service": "billing-worker",
                        "env": "production",
                        "status": "error",
                        "trace_id": "trace-abc-999",
                        "error": {
                            "kind": "PaymentDeclinedException",
                            "message": "Renewal failed for user 4412",
                            "stack": '  File "api/subscription.py", line 19, in renew_subscription',
                        },
                    },
                },
                # 3. OpenTelemetry span directly in services/gateway.py
                {
                    "source": "opentelemetry",
                    "traceId": "trace-abc-999",
                    "spanId": "span-gw-1",
                    "count": 60,
                    "resource": {"attributes": {"service.name": "payment-gateway", "deployment.environment": "production"}},
                    "attributes": {
                        "exception.type": "ConnectionResetError",
                        "exception.message": "Upstream TLS socket closed on port 443",
                        "exception.stacktrace": '  File "services/gateway.py", line 22, in charge_card',
                    },
                },
                # 4. Prometheus Alertmanager latency spike on services/gateway.py
                {
                    "alerts": [
                        {
                            "fingerprint": "prom-gw-latency",
                            "count": 39,
                            "labels": {
                                "alertname": "HighGatewayErrorRate",
                                "service": "payment-gateway",
                                "severity": "critical",
                                "file": "services/gateway.py",
                                "line": "22",
                            },
                            "annotations": {
                                "summary": "Gateway error rate > 45% over 5m",
                            },
                        }
                    ]
                },
                # 5. Unrelated low-priority New Relic warning in utils/health.py (1 event)
                {
                    "error.class": "HealthCheckWarning",
                    "error.message": "Slow ping response 210ms",
                    "appName": "health-monitor",
                    "environment": "staging",
                    "severity": "warning",
                    "count": 1,
                    "stackTrace": ['  File "utils/health.py", line 5, in ping'],
                },
            ]

            events = normalize_telemetry_payload(payload)
            report = correlate_and_prioritize_incidents(events, graph, profiles, repo_root)

            # 500 raw alerts (250 + 150 + 60 + 39 + 1) -> 5 deduplicated symptom groups -> 2 root-cause clusters!
            self.assertEqual(500, report.total_raw_events)
            self.assertEqual(5, report.deduplicated_groups)
            self.assertEqual(2, report.root_cause_clusters)
            self.assertGreater(report.noise_reduction_pct, 99.0)

            top_cluster = report.clusters[0]
            self.assertEqual("services/gateway.py", top_cluster.root_cause_file)
            self.assertEqual(499, top_cluster.total_events)
            self.assertEqual(4, top_cluster.group_count)
            self.assertEqual("P0-Critical", top_cluster.priority_label)
            self.assertEqual(["datadog", "opentelemetry", "prometheus", "sentry"], top_cluster.sources)
            self.assertTrue(any("Dependency graph convergence" in r for r in top_cluster.correlation_reasons))
            self.assertTrue(any("trace-abc-999" in r for r in top_cluster.correlation_reasons))
            self.assertTrue(any("Error storm anomaly" in a for a in top_cluster.anomalies))

            low_cluster = report.clusters[1]
            self.assertEqual("utils/health.py", low_cluster.root_cause_file)
            self.assertEqual(1, low_cluster.total_events)
            self.assertLess(low_cluster.priority_score, top_cluster.priority_score)

    def test_parse_application_logs_extracts_tracebacks_and_trace_ids(self):
        raw_logs = "\n".join([
            "2026-11-21T10:15:01Z [ERROR] [checkout-service] trace_id=req-7781 Failed to execute batch",
            "Traceback (most recent call last):",
            '  File "src/click/core.py", line 146, in batch',
            "    return list(zip(*repeat(iterable, batch_size), strict=False))",
            "TypeError: repeat() received raw iterable",
            "",
            "2026-11-21T10:15:02Z [CRITICAL] [cli-worker] trace_id=req-7781 Unhandled command crash",
            '  File "src/click/decorators.py", line 42, in wrapper',
            "RuntimeError: downstream batch failure",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "app.log"
            log_file.write_text(raw_logs, encoding="utf-8")
            events = load_telemetry_events(str(log_file))

        self.assertEqual(2, len(events))
        self.assertEqual("logs", events[0].source)
        self.assertEqual("TypeError", events[0].exception_type)
        self.assertEqual("req-7781", events[0].trace_id)
        self.assertEqual("checkout-service", events[0].service)
        self.assertEqual("RuntimeError", events[1].exception_type)
        self.assertEqual("req-7781", events[1].trace_id)


if __name__ == "__main__":
    unittest.main()

