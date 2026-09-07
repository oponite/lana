"""Cross-language differential test: Lana (quantum) vs Python/Java (classical).

Runs the same five decision experiments in three implementations and asserts
three properties:

1. parity    — the two classical models (Python, Java) agree with each other.
2. recovery  — Lana's classical projection matches the classical model exactly
               (when the disposition is ignored, quantum reduces to classical).
3. divergence— Lana's quantum result differs from the classical result only
               where the disposition `d` carries information — the exact spots
               Python/Java cannot express.

The comparison is model-vs-model: Python and Java stand in for the classical
probability model (one scalar `p`); Lana stands in for the density-operator
model (`p` + `d`).
"""

import json
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "tests", "quantum_vs_classical")
LANA = os.environ.get("LANA_BIN", os.path.join(ROOT, "build", "lana"))

# Homebrew openjdk is not on PATH by default, and /usr/bin/javac is a stub
# that fails without a JDK. Prefer the real JDK when present.
_JDK = "/opt/homebrew/opt/openjdk@21/bin"
JAVAC = os.path.join(_JDK, "javac") if os.path.exists(os.path.join(_JDK, "javac")) else shutil.which("javac")
JAVA = os.path.join(_JDK, "java") if os.path.exists(os.path.join(_JDK, "java")) else shutil.which("java")

EXPERIMENTS = ["distinguish", "invert", "distance", "neutralize", "agreement"]


def run(cmd, cwd=None):
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def close(a, b, tol=1e-9):
    """Recursive numeric-tolerant equality for parsed JSON."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"key mismatch: {set(a)} vs {set(b)}"
        for k in a:
            close(a[k], b[k], tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        assert math.isclose(a, b, abs_tol=tol), f"{a!r} != {b!r}"
    else:
        assert a == b, f"{a!r} != {b!r}"


def test_quantum_vs_classical():
    # Compile the Java baseline once.
    subprocess.run(
        [JAVAC, os.path.join(DIR, "Classical.java")], check=True, capture_output=True
    )

    py = json.loads(run([sys.executable, os.path.join(DIR, "classical.py")]))
    java = json.loads(run([JAVA, "-cp", DIR, "Classical"]))
    lana = json.loads(run([LANA, "run", os.path.join(DIR, "quantum.lana")]))

    # 1. parity: the two classical implementations agree.
    close(py, java)

    # 2. recovery: Lana's classical projection matches the classical model.
    for exp in EXPERIMENTS:
        close(lana[exp]["classical"], py[exp]["classical"])

    # 3. divergence: Lana's quantum result differs exactly where d matters.

    # distinguish — classical conflates opposite phases, quantum separates them.
    assert py["distinguish"]["classical"]["plus"] == py["distinguish"]["classical"]["minus"]
    assert not math.isclose(
        lana["distinguish"]["quantum"]["x_plus"],
        lana["distinguish"]["quantum"]["x_minus"],
        abs_tol=1e-9,
    )

    # invert — classical sees only p -> 1-p; quantum sees the phase flip.
    assert math.isclose(lana["invert"]["quantum"]["y_before"], 0.8666, abs_tol=1e-3)
    assert math.isclose(lana["invert"]["quantum"]["y_after"], 0.1334, abs_tol=1e-3)

    # distance — classical metric is degenerate; quantum trace distance is not.
    assert py["distance"]["classical"] == 0.0
    assert math.isclose(lana["distance"]["quantum"], 0.8, abs_tol=1e-9)

    # neutralize — destroys the phase signal (classical probability is lossy).
    assert math.isclose(lana["neutralize"]["quantum"]["x_before"], 0.1, abs_tol=1e-9)
    assert math.isclose(lana["neutralize"]["quantum"]["x_after"], 0.5, abs_tol=1e-9)

    # agreement — classical fused p is identical; quantum disposition spread differs.
    assert math.isclose(
        py["agreement"]["classical"]["agree"],
        py["agreement"]["classical"]["oppose"],
        abs_tol=1e-9,
    )
    assert math.isclose(
        lana["agreement"]["quantum"]["input_distance_agree"], 0.0, abs_tol=1e-9
    )
    assert lana["agreement"]["quantum"]["input_distance_oppose"] > 0.5
    assert math.isclose(
        lana["agreement"]["quantum"]["aligned_sample_d_re"], 0.6, abs_tol=1e-9
    )
    assert not math.isclose(
        lana["agreement"]["quantum"]["opposed_sample_d_re"], 0.6, abs_tol=1e-9
    )
