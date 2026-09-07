"""Check reduction diagnostics and that failed compilation publishes no bytecode."""

import pathlib
import subprocess
import sys
import tempfile


CASES = [
    ('sum(t, axes: 0);', 'expected axis:'),
    ('sum(axis: 0, t);', 'second and final argument'),
    ('sum(t, axis: 0, axis: 1);', 'final argument'),
    ('sum(t, axis: 0, 1);', 'final argument'),
    ('sum(t, 0, axis: 1);', 'second and final argument'),
    ('sum(t, "axis": 0);', 'expected axis:'),
    ('shape(t, axis: 0);', 'named arguments are not supported for this call'),
    ('sum(t,', 'expected argument or closing parenthesis'),
    ('sum(t,);', 'expected argument after comma'),
    ('sum();', 'expects a tensor and optional axis'),
    ('sum(t, 0, 1);', 'expects a tensor and optional axis'),
    # Indexing and slicing boundaries: arrays take exactly one position,
    # empty brackets are rejected, and assignment takes no slices.
    ('let a = [1, 2]; a[0:1];', 'arrays cannot be sliced'),
    ('let a = [1, 2]; a[0, 1];', 'array indexing accepts one position'),
    ('t[];', 'expected index position'),
    ('t[0:2, 1] = tensor([1]);', 'slices cannot be assigned'),
    ('t[0, 1] = 5.0;', 'assignment accepts a single index position'),
]

with tempfile.TemporaryDirectory(prefix="lana-tensor-source-") as directory:
    root = pathlib.Path(directory)
    for index, (expression, diagnostic) in enumerate(CASES):
        source = root / f"case-{index}.lana"
        output = root / f"case-{index}.labc"
        source.write_text('let t = tensor([[1, 2], [3, 4]]);\n' + expression)
        result = subprocess.run(
            [sys.argv[1], "compile", str(source), "-o", str(output)],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode != 0, expression
        assert diagnostic in result.stderr, (expression, result.stderr)
        assert not output.exists(), expression

print("TENSOR_SOURCE_ERRORS_PASS")
