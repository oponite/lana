"""Checks that differential testing cannot silently accept missing evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from source_contract import check_failure

spec = importlib.util.spec_from_file_location(
    "differential", Path(__file__).parent / "conformance/differential/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ConformanceRunnerTests(unittest.TestCase):
    def test_source_failure_requires_exact_stage_exit_and_no_stdout(self):
        diagnostic = "fixture.lana:1:1-1:2: error[type/LANA_ERR_TYPE]: expected failure\n"
        good = subprocess.CompletedProcess([], 1, "", diagnostic)
        check_failure(good, "LANA_ERR_TYPE", "expected failure")
        for code, stdout, stderr in (
            (0, "", diagnostic), (-11, "", diagnostic), (2, "", diagnostic),
            (1, "partial", diagnostic), (1, "", diagnostic.replace("type/", "parse/")),
            (1, "", diagnostic + diagnostic), (1, "", "expected failure\n"),
        ):
            with self.assertRaises(AssertionError):
                check_failure(subprocess.CompletedProcess([], code, stdout, stderr),
                              "LANA_ERR_TYPE", "expected failure")

    def test_missing_or_duplicate_statistics_fail(self):
        for stderr in (b"", b"LANAVM_STATS {}\nLANAVM_STATS {}\n"):
            with self.assertRaises(AssertionError):
                runner.statistics(stderr)

    def test_only_allocation_and_timing_statistics_are_excluded(self):
        record = dict(instructions=2, state_transitions=0, opcodes={"HALT": 1},
                      allocations=1, allocated_bytes=24, elapsed_ns=42)
        self.assertEqual(runner.statistics(b"LANAVM_STATS " + json.dumps(record).encode()),
                         dict(instructions=2, state_transitions=0, opcodes={"HALT": 1}))

    def test_memory_normalization_preserves_limit_and_rejects_overrun(self):
        self.assertNotEqual(runner.diagnostic(b"  resource: memory limit 100, observed 1 bytes"),
                            runner.diagnostic(b"  resource: memory limit 200, observed 1 bytes"))
        with self.assertRaises(AssertionError):
            runner.diagnostic(b"  resource: memory limit 100, observed 101 bytes")
        for text in (b"wrong error", b"  resource: instructions limit 100, observed 101 instructions"):
            self.assertEqual(runner.diagnostic(text), text)

    def test_matching_implementations_still_need_correct_outcome(self):
        expected = dict(exit=1, stdout="", stderr="expected error\n")
        for result in (
            subprocess.CompletedProcess([], 0, b"", b"expected error\n"),
            subprocess.CompletedProcess([], 1, b"partial output", b"expected error\n"),
            subprocess.CompletedProcess([], 1, b"", b"different error\n"),
        ):
            with self.assertRaises(AssertionError):
                runner.assert_contract(result, expected)


if __name__ == "__main__":
    unittest.main()
