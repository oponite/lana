"""Execute source fixtures that need arguments or an isolated filesystem."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LANA = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else ROOT / "build/lana"


def run(name, directory, *arguments):
    result = subprocess.run(
        [str(LANA), "run", str(ROOT / "tests/regression" / name), "--", *map(str, arguments)],
        cwd=directory, text=True, capture_output=True, timeout=30,
        env={**os.environ, "LANA_STDLIB_DIR": str(ROOT / "stdlib")})
    assert result.returncode == 0, (name, result.stderr)
    assert result.stderr == "", (name, result.stderr)
    return result.stdout


class SourceInputTests(unittest.TestCase):
    def test_durable_pipeline(self):
        with tempfile.TemporaryDirectory(prefix="lana-durable-pipeline-") as directory:
            self.assertEqual(run(ROOT / "tests/conformance/durable/durable_pipeline.lana", directory),
                             "DURABLE_PIPELINE_PASS\n")
            self.assertTrue((Path(directory) / "pipeline.db/manifest").is_file())

    def test_library_workflows(self):
        with tempfile.TemporaryDirectory(prefix="lana-library-") as directory:
            for name, marker in (
                ("workflow_library_pass.lana", "WORKFLOW_LIBRARY_PASS\n"),
                ("replay_library_pass.lana", "REPLAY_LIBRARY_PASS\n"),
                ("reference_apps_pass.lana", "REFERENCE_APPS_PASS\n"),
                ("policy_library_pass.lana", "POLICY_LIBRARY_PASS\n"),
                ("host_hash_pass.lana", ""),
            ):
                with self.subTest(name=name):
                    self.assertEqual(run(name, directory), marker)
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_filesystem_boundary(self):
        with tempfile.TemporaryDirectory(prefix="lana-filesystem-") as directory:
            root = Path(directory)
            (root / "existing-directory").mkdir()
            self.assertEqual(run("host_filesystem_pass.lana", directory, directory), "HOST_FILESYSTEM_PASS\n")
            files = {str(path.relative_to(root)): path.read_text()
                     for path in root.rglob("*") if path.is_file()}
            self.assertEqual(files, {"lana-host-boundary-test/atomic.txt": "atomic"})

    def test_lexer_tokens_and_source_positions(self):
        with tempfile.TemporaryDirectory(prefix="lana-lexer-") as directory:
            source = Path(directory) / "input.lana"
            source.write_text('let word = "if"; // comment\nword;\n')
            actual = json.loads(run("lexer_driver.lana", directory, source))
            self.assertEqual(actual, [
                ["identifier", "let", 1, 1], ["identifier", "word", 1, 5],
                ["symbol", "=", 1, 10], ["string", "if", 1, 12],
                ["symbol", ";", 1, 16], ["identifier", "word", 2, 1],
                ["symbol", ";", 2, 5], ["eof", "<eof>", 3, 1],
            ])

    def test_parser_preserves_string_map_keys(self):
        with tempfile.TemporaryDirectory(prefix="lana-parser-") as directory:
            source = Path(directory) / "input.lana"
            source.write_text('let word = "if";\nprint({"key": word});\n')
            actual = json.loads(run("parser_driver.lana", directory, source))
            self.assertEqual(actual, [0, [], [
                [2, "word", None, [20, "string", "if", 1, 12], 1, 5],
                [8, [28, [["key", [21, "word", 2, 15]]], 2, 7], 2, 1],
            ], [], []])


if __name__ == "__main__":
    unittest.main()
