"""The tap release guard must never downgrade or change a published version."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from release_tap import prepare, render_formula  # noqa: E402


with tempfile.TemporaryDirectory() as directory:
    candidate = Path(directory) / "candidate.rb"
    current = Path(directory) / "lana.rb"
    digest = "a" * 64
    candidate.write_text(render_formula("4.1.0", digest))
    current.write_text(render_formula("4.0.0", "b" * 64))
    assert prepare(candidate, current, "4.1.0", digest) == "updated"
    assert prepare(candidate, current, "4.1.0", digest) == "already-matching"
    candidate.write_text(render_formula("4.1.0", digest).replace("end\n", '  system "unexpected Ruby"\nend\n', 1))
    try:
        prepare(candidate, current, "4.1.0", digest)
        raise AssertionError("modified release formula accepted")
    except ValueError:
        pass
    candidate.write_text(render_formula("4.1.0", digest))
    current.write_text(render_formula("4.1.0", "c" * 64))
    try:
        prepare(candidate, current, "4.1.0", digest)
        raise AssertionError("same-version replacement accepted")
    except ValueError:
        pass
    current.write_text(render_formula("4.2.0", digest))
    try:
        prepare(candidate, current, "4.1.0", digest)
        raise AssertionError("downgrade accepted")
    except ValueError:
        pass

print("RELEASE_TAP_PASS")
