"""Guarded branches reject effects and escaping unresolved values before execution."""
import pathlib
import subprocess
import sys
import tempfile

CASES = [
    ('if (guard) { print("LEAK"); }', "cannot perform effects"),
    ('let xs = [0]; if (guard) { xs[0] = 1; }', "cannot perform effects"),
    ('if (guard) { write_text("leak.txt", "LEAK"); }', "cannot perform effects"),
    ('let x = 0; if (guard) { x = 1; } else { x = 2; } print(x);', "requires resolve"),
    ('let x = true; if (guard) { x = false; } while (x) { x = false; }', "requires resolve"),
]

with tempfile.TemporaryDirectory(prefix="lana-guarded-source-") as directory:
    root = pathlib.Path(directory)
    for index, (body, diagnostic) in enumerate(CASES):
        source = root / f"case-{index}.lana"
        output = root / f"case-{index}.labc"
        source.write_text('let guard = possibility([true, false]);\n' + body)
        result = subprocess.run(
            [sys.argv[1], "compile", str(source), "-o", str(output)],
            cwd=root, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode != 0, body
        assert diagnostic in result.stderr, (body, result.stderr)
        assert not output.exists(), body
        assert not (root / "leak.txt").exists(), body
        assert result.stdout == "", result.stdout
print("GUARDED_SOURCE_ERRORS_PASS")
