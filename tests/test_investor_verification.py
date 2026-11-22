import unittest
from pathlib import Path

from intelligence.investor_verification import run_investor_verification_suite

ROOT = Path(__file__).resolve().parents[1]


class TestInvestorVerification(unittest.TestCase):
    def test_investor_verification_suite_proves_all_10_criteria(self):
        report = run_investor_verification_suite(str(ROOT))
        self.assertTrue(report.all_requirements_verified)
        self.assertEqual(10, len(report.requirements))
        self.assertTrue(all(r.verified for r in report.requirements))
        self.assertEqual(8, len(report.safety_matrix))
        self.assertTrue(all(s.blocked_or_rolled_back for s in report.safety_matrix))
        self.assertGreaterEqual(len(report.multi_repo_benchmarks), 2)
        self.assertGreater(report.measurable_metrics["mttr_reduction_pct"], 99.0)


if __name__ == "__main__":
    unittest.main()

