"""Check immutable-set diagnostics: argument-count errors at compile time and
STATE rejection at run time."""

import pathlib
import subprocess
import sys
import tempfile


COMPILE_CASES = [
    ('let s = set_new(); set_add(s);', 'set_add has invalid argument count'),
    ('let s = set_new(); set_contains(s);', 'set_contains has invalid argument count'),
    ('let s = set_new(); set_union(s);', 'set_union has invalid argument count'),
    ('let s = set_new(); set_intersect(s);', 'set_intersect has invalid argument count'),
    ('let s = set_new(); set_difference(s);', 'set_difference has invalid argument count'),
]

with tempfile.TemporaryDirectory(prefix="lana-set-source-") as directory:
    root = pathlib.Path(directory)
    for index, (expression, diagnostic) in enumerate(COMPILE_CASES):
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

    # STATE is not a set member: rejected at run time with LANA_ERR_TYPE.
    state_source = root / "state.lana"
    state_source.write_text(
        'state q = probability(0.5);\n'
        'let s = set_new();\n'
        's = set_add(s, q);\n'
    )
    result = subprocess.run(
        [sys.argv[1], "run", str(state_source)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0, "STATE insertion should fail at run time"
    assert "LANA_ERR_TYPE" in result.stderr, result.stderr

print("SET_SOURCE_ERRORS_PASS")
