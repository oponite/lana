"""Check benchmark accounting, not a machine-specific speed threshold."""
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("metrics", ROOT / "scripts/version_metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class BenchmarkAccountingTests(unittest.TestCase):
    def test_missing_hardware_counts_are_not_estimated(self):
        scores = metrics.scores(dict(units=20, elapsed_ns=100, allocations=4, retained_bytes=16))
        self.assertIsNone(scores["units_per_cycle"])
        self.assertIsNone(scores["units_per_bit"])
        self.assertEqual(scores["units_per_allocation"], 5)
        self.assertEqual(scores["units_per_retained_byte"], 1.25)
        self.assertEqual(scores["units_per_second"], 200_000_000)

    def test_invalid_denominators_are_unavailable(self):
        for value in (0, -1, float("inf"), float("nan"), None):
            with self.subTest(value=value):
                scores = metrics.scores(dict(units=1, elapsed_ns=1, cycles=value, memory_bits=value))
                self.assertIsNone(scores["units_per_cycle"])
                self.assertIsNone(scores["units_per_bit"])

    def test_incomplete_trials_are_rejected(self):
        for units, elapsed in ((0, 1), (-1, 1), (1.5, 1), (1, 0), (1, -1)):
            with self.assertRaises(ValueError):
                metrics.scores(dict(units=units, elapsed_ns=elapsed))

    def test_corpus_obeys_state_invariants(self):
        for index in range(256):
            for p, re, im in metrics.inputs(index):
                self.assertTrue(0 <= p <= 1)
                self.assertLessEqual(math.hypot(re, im), 1)
                if p in (0, 1):
                    self.assertEqual((re, im), (0, 0))


if __name__ == "__main__":
    unittest.main()
