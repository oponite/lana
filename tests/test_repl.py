"""REPL session tests (LIP-020).

Covers session persistence, state interaction, multiline input, error
recovery (SYNTAX-10), :save round-trip, and determinism. The REPL is a thin
loop over the existing compiler and VM, so both the Rust and C CLIs must
behave identically.
"""

import pathlib
import subprocess
import sys
import tempfile


def repl(cli, lines):
    """Run the REPL with the given input lines; return (stdout, stderr, code)."""
    return subprocess.run(
        [cli, "repl"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )


def run(cli, source):
    """Run a .lana source file; return (stdout, stderr, code)."""
    return subprocess.run(
        [cli, "run", str(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )


with tempfile.TemporaryDirectory(prefix="lana-repl-") as directory:
    root = pathlib.Path(directory)
    cli = sys.argv[1]

    # 1. Session persistence: a binding defined in one input is usable in the
    #    next.
    result = repl(cli, ["let x = 5;", "x + 1;", ":quit"])
    assert result.returncode == 0, result.stderr
    assert "6" in result.stdout, result.stdout

    # 2. State interaction: construct a STATE, then use it in a later input.
    result = repl(cli, [
        "state s = state(p: 0.5, d: 0.0);",
        "s.p;",
        ":quit",
    ])
    assert result.returncode == 0, result.stderr
    assert "0.5" in result.stdout, result.stdout

    # 3. Multiline input: a line ending in an open bracket continues to the
    #    next line.
    result = repl(cli, ["let total = (1 +", "2);", "total;", ":quit"])
    assert result.returncode == 0, result.stderr
    assert "3" in result.stdout, result.stdout

    # 4. Error recovery: a syntax error reports the SYNTAX-10 recovery message
    #    and the session continues.
    result = repl(cli, ["let x = 5;", "let y = ;", "x + 1;", ":quit"])
    assert result.returncode == 0, result.stderr
    assert "expected expression" in result.stderr, result.stderr
    assert "6" in result.stdout, result.stdout

    # 5. :save round-trip: the saved file re-executes to the same state.
    saved = root / "session.lana"
    result = repl(cli, [
        "let a = 10;",
        "let b = 20;",
        f":save {saved}",
        ":quit",
    ])
    assert result.returncode == 0, result.stderr
    assert saved.exists(), "saved file missing"
    text = saved.read_text()
    assert "let a = 10;" in text, text
    assert "let b = 20;" in text, text
    # Re-running the saved file reproduces the session's bindings.
    check = root / "check.lana"
    check.write_text(text + "print(a + b);\n")
    result = run(cli, str(check))
    assert result.returncode == 0, result.stderr
    assert "30" in result.stdout, result.stdout

    # 6. Determinism: the same input sequence and seed produce the same output.
    lines = ["let x = 3;", "let y = 4;", "x * y;", ":quit"]
    first = repl(cli, lines)
    second = repl(cli, lines)
    assert first.stdout == second.stdout, "non-deterministic stdout"
    assert first.stderr == second.stderr, "non-deterministic stderr"

print("REPL_SESSION_PASS")
