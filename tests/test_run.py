"""The test runner must reject missing, skipped, duplicated, or stale evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runner", ROOT / "tests/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestRunnerTests(unittest.TestCase):
    def test_candidate_hash_tracks_source_but_not_external_symlink_contents(self):
        with tempfile.TemporaryDirectory(prefix="lana-fingerprint-") as directory:
            directory = Path(directory)
            root = directory / "repo"
            root.mkdir()
            external = directory / "external"
            external.write_text("first")
            (root / "link").symlink_to(external)
            (root / "source").write_bytes(b"one\0two")
            with patch.object(runner, "ROOT", root), patch.object(runner, "git", return_value=b"link\0source\0"):
                before = runner.tree_fingerprint()
                external.write_text("changed outside candidate")
                self.assertEqual(runner.tree_fingerprint(), before)
                (root / "source").write_bytes(b"one\0three")
                self.assertNotEqual(runner.tree_fingerprint(), before)
                before = runner.tree_fingerprint()
                (root / "source").unlink()
                self.assertNotEqual(runner.tree_fingerprint(), before)

    def test_rust_counts_cannot_hide_ignored_or_missing_tests(self):
        self.assertEqual(runner.cargo_counts("test result: ok. 3 passed; 0 failed; 0 ignored;"),
                         {"passed": 3, "failed": 0, "ignored": 0})
        for output in ("", "test result: ok. 0 passed; 0 failed; 0 ignored;",
                       "test result: ok. 3 passed; 0 failed; 1 ignored;"):
            with self.assertRaises(RuntimeError):
                runner.cargo_counts(output)

    def test_missing_empty_and_duplicate_junit_cases_fail(self):
        for xml in ('<testsuite/>', '<testsuite><testcase name="a"/></testsuite>',
                    '<testsuite><testcase name="a"/><testcase name="a"/></testsuite>'):
            with tempfile.TemporaryDirectory(prefix="lana-runner-test-") as directory:
                path = Path(directory) / "results.xml"
                path.write_text(xml)
                with self.assertRaises(RuntimeError):
                    runner.junit_results(path, {"a", "b"})

    def test_skips_and_failures_are_not_counted_as_passes(self):
        with tempfile.TemporaryDirectory(prefix="lana-runner-test-") as directory:
            path = Path(directory) / "results.xml"
            path.write_text('<testsuite><testcase name="a"/><testcase name="b"><skipped/></testcase>'
                            '<testcase name="c"><failure/></testcase></testsuite>')
            self.assertEqual(runner.junit_results(path, {"a", "b", "c"}),
                             {"a": "passed", "b": "skipped", "c": "failed"})

    def test_registry_rejects_dangling_gates_tests_and_files(self):
        manifest = json.loads((ROOT / "tests/claims.json").read_text())
        inventory = {name for claim in manifest["claims"] for name in claim.get("ctest", [])}
        runner.validate_registry(manifest, inventory)
        for field, invalid in (("requires", ["does-not-exist"]), ("ctest", ["does-not-exist"]),
                               ("authority", ["does-not-exist.md"]), ("evidence", ["../outside.py"])):
            changed = copy.deepcopy(manifest)
            changed["claims"][0][field] = invalid
            with self.assertRaises(RuntimeError):
                runner.validate_registry(changed, inventory)

    def test_repository_registration_is_complete(self):
        runner.check_registration()


if __name__ == "__main__":
    unittest.main()
