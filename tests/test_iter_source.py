"""Check iterator/for-loop type diagnostics and that failed compilation publishes no bytecode."""

import pathlib
import subprocess
import sys
import tempfile


CASES = [
    ('let xs = array_new(0);\nfor x in 42 { print(x); }', 'for iterable must be an array or generator'),
    ('let xs = array_new(0);\nlet kept = filter(42, xs);', 'filter requires a function name as its first argument'),
    ('let xs = array_new(0);\nlet total = reduce(42, xs, 0);', 'reduce requires a function name as its first argument'),
]

with tempfile.TemporaryDirectory(prefix="lana-iter-source-") as directory:
    root = pathlib.Path(directory)
    for index, (expression, diagnostic) in enumerate(CASES):
        source = root / f"case-{index}.lana"
        output = root / f"case-{index}.labc"
        source.write_text(expression)
        result = subprocess.run(
            [sys.argv[1], "compile", str(source), "-o", str(output)],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode != 0, expression
        assert diagnostic in result.stderr, (expression, result.stderr)
        assert not output.exists(), expression

print("ITER_SOURCE_ERRORS_PASS")
