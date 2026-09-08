"""Independent state/matrix oracles and executable README examples (stdlib only)."""
import argparse
import cmath
import json
import math
from pathlib import Path
import random
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--lana", type=Path, default=ROOT / "build/lana")
parser.add_argument("--rust", type=Path)
options, remaining = parser.parse_known_args()
LANA = options.lana.resolve()
VMS = [LANA.with_name("lanavm")]
if options.rust:
    VMS.append(options.rust.resolve())


def number(value):
    # The source lexer has decimal literals, not Python's exponent notation.
    return format(value, ".17f").rstrip("0").rstrip(".") or "0"


def run(command, directory):
    result = subprocess.run(list(map(str, command)), cwd=directory,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (command, result.stderr)
    assert result.stderr == "", (command, result.stderr)
    return result.stdout


def execute(source):
    with tempfile.TemporaryDirectory(prefix="lana-oracle-") as directory:
        source_path = Path(directory) / "oracle.lana"
        bytecode = Path(directory) / "oracle.labc"
        source_path.write_text(source)
        assert run([LANA, "compile", source_path, "-o", bytecode], directory) == ""
        for vm in VMS:
            yield vm.name, run([vm, "run", bytecode], directory)
        assert {p.name for p in Path(directory).iterdir()} == {"oracle.lana", "oracle.labc"}


def probabilities(matrix):
    # Born probabilities for the ordered bases in semantics.md section 2.4.
    vectors = ((0, 1), (1 / math.sqrt(2), -1 / math.sqrt(2)),
               (1 / math.sqrt(2), -1j / math.sqrt(2)))
    return [sum(complex(v[i]).conjugate() * matrix[i][j] * v[j]
                for i in range(2) for j in range(2)).real for v in vectors]


class OracleTests(unittest.TestCase):
    def assert_numbers(self, actual, expected, context):
        self.assertEqual(len(actual), len(expected), context)
        for index, (left, right) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isfinite(left), (context, index, left))
            self.assertTrue(math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12),
                            (context, index, left, right))

    def test_seeded_state_measurement_transforms_and_append(self):
        rng = random.Random(0x1A4A)
        cases = [(p, complex(re, im)) for p in (0, 1, 0.5, 0.25, 0.75)
                 for re, im in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))]
        cases += [(rng.random(), cmath.rect(math.sqrt(rng.random()), rng.uniform(-math.pi, math.pi)))
                  for _ in range(103)]
        source = '''fn probe(p, re, im) {
    state s = state(p: p, d_re: re, d_im: im);
    let values = [measure s as probability, measure s in x as probability,
                  measure s in y as probability, s.d_re, s.d_im];
    transform s with invert();
    array_push(values, measure s as probability);
    array_push(values, measure s in x as probability);
    array_push(values, measure s in y as probability);
    transform s with invert();
    array_push(values, s.p); array_push(values, s.d_re); array_push(values, s.d_im);
    transform s with neutralize();
    array_push(values, measure s in x as probability);
    array_push(values, measure s in y as probability);
    state b = state(p: 0.3, d: 0.2);
    let joined = append(s, b);
    array_push(values, measure joined as probability);
    print(json_stringify(values));
}
'''
        expected = []
        for p, d in cases:
            source += f"probe({number(p)}, {number(d.real)}, {number(d.imag)});\n"
            d = d if 0 < p < 1 else 0j
            c = d * math.sqrt(p * (1 - p))
            rho = [[1 - p, c], [c.conjugate(), p]]
            flipped = [[rho[1 - i][1 - j] for j in range(2)] for i in range(2)]
            expected.append(probabilities(rho) + [d.real, d.imag] + probabilities(flipped)
                            + [p, d.real, d.imag, 0.5, 0.5, 1 - (1 - p) * 0.7])
        for vm, output in execute(source):
            rows = [json.loads(line) for line in output.splitlines()]
            self.assertEqual(len(rows), len(cases), vm)
            for index, (row, oracle) in enumerate(zip(rows, expected)):
                self.assert_numbers(row, oracle, (vm, index, cases[index]))

    def test_seeded_rectangular_matmul_and_axis_reductions(self):
        rng = random.Random(741)
        source, expected = [], []
        for index in range(24):
            m, k, n = (rng.randint(1, 5) for _ in range(3))
            a = [[rng.randint(-8, 8) / 4 for _ in range(k)] for _ in range(m)]
            b = [[rng.randint(-8, 8) / 4 for _ in range(n)] for _ in range(k)]
            product = [[sum(a[i][x] * b[x][j] for x in range(k)) for j in range(n)] for i in range(m)]
            flat = [item for row in product for item in row]
            oracle = flat + [sum(row) for row in product] + [sum(product[i][j] for i in range(m)) for j in range(n)]
            expected.append(oracle)
            dtype = ("f64", "f32", "f16")[index % 3]
            source.append(f'''fn probe_{index}() {{
    let a = tensor({json.dumps(a)}, dtype: "{dtype}");
    let b = tensor({json.dumps(b)}, dtype: "{dtype}");
    let c = matmul(a, b);
    let values = []; let i = 0;
    while (i < {m}) {{ let j = 0;
        while (j < {n}) {{ array_push(values, c[i, j]); j = j + 1; }}
        i = i + 1;
    }}
    let rows = sum(c, axis: 1); let columns = sum(c, axis: 0); i = 0;
    while (i < {m}) {{ array_push(values, rows[i]); i = i + 1; }}
    i = 0;
    while (i < {n}) {{ array_push(values, columns[i]); i = i + 1; }}
    print(json_stringify(values));
}}
probe_{index}();''')
        for vm, output in execute("\n".join(source)):
            rows = [json.loads(line) for line in output.splitlines()]
            self.assertEqual(len(rows), len(expected), vm)
            for index, (row, oracle) in enumerate(zip(rows, expected)):
                self.assert_numbers(row, oracle, (vm, index))

    def test_readme_programs(self):
        snippets = re.findall(r"^```lana\n(.*?)^```", (ROOT / "README.md").read_text(), re.M | re.S)
        # A new example must provide an expected result, not silently go untested.
        expected = [0.05, 5, 0.75, 0.76]
        self.assertEqual(len(snippets), len(expected))
        for index, (snippet, value) in enumerate(zip(snippets, expected)):
            for vm, output in execute(snippet):
                self.assert_numbers([float(output)], [value], (vm, "README", index))


if __name__ == "__main__":
    unittest.main(argv=[__file__, *remaining])
