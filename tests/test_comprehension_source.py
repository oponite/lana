"""Check comprehension type diagnostics and that failed compilation publishes no bytecode."""

import pathlib
import subprocess
import sys
import tempfile


CASES = [
    ('let xs = [x for x in 42];', 'comprehension iterable must be an array or generator'),
    ('let xs = {x for x in 42};', 'comprehension iterable must be an array or generator'),
    ('let xs = {k: v for k, v in 42};', 'comprehension iterable must be an array or generator'),
]

with tempfile.TemporaryDirectory(prefix="lana-comprehension-source-") as directory:
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

print("COMPREHENSION_SOURCE_ERRORS_PASS")
