"""Check yield/next type diagnostics and that failed compilation publishes no bytecode."""

import pathlib
import subprocess
import sys
import tempfile


CASES = [
    ('yield 1;', 'yield outside a generator function'),
    ('let it = next(42);', 'next requires a Generator'),
]

with tempfile.TemporaryDirectory(prefix="lana-generator-source-") as directory:
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

print("GENERATOR_SOURCE_ERRORS_PASS")
