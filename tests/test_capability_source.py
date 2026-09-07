"""Check grant/revoke type diagnostics and that failed compilation publishes no bytecode."""

import pathlib
import subprocess
import sys
import tempfile


CASES = [
    ('grant(42, "use");', 'grant requires an admin capability'),
    ('grant(cap, "read");', 'grant permission must be use or admin'),
    ('let p = "use"; grant(cap, p);', 'grant permission must be a string literal'),
    ('revoke(42);', 'revoke requires a shared capability'),
]

with tempfile.TemporaryDirectory(prefix="lana-capability-source-") as directory:
    root = pathlib.Path(directory)
    for index, (expression, diagnostic) in enumerate(CASES):
        source = root / f"case-{index}.lana"
        output = root / f"case-{index}.labc"
        source.write_text('let cap = capability("gpu");\n' + expression)
        result = subprocess.run(
            [sys.argv[1], "compile", str(source), "-o", str(output)],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode != 0, expression
        assert diagnostic in result.stderr, (expression, result.stderr)
        assert not output.exists(), expression

print("CAPABILITY_SOURCE_ERRORS_PASS")
