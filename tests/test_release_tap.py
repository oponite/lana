"""Release admission and concurrent tap publication preserve every version."""
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from release_tap import prepare, publish, render_formula


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("unsafe admission accepted")


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    current = root / "lana.rb"
    candidate = root / "candidate.rb"
    archived = root / "lana@4.1.0.rb"
    digest = "a" * 64
    candidate.write_text(render_formula("4.1.0", digest))
    assert prepare(candidate, current, "4.1.0", digest) == "updated"
    assert "class LanaAT410 < Formula" in archived.read_text()
    assert 'keg_only "exact releases must not replace the default lana command"' in archived.read_text()
    assert prepare(candidate, current, "4.1.0", digest) == "already-matching"
    original = {p: p.read_bytes() for p in (current, archived)}
    candidate.write_text(render_formula("4.1.0", digest).replace("end\n", '  system "unexpected Ruby"\nend\n', 1))
    rejected(lambda: prepare(candidate, current, "4.1.0", digest))
    assert all(p.read_bytes() == content for p, content in original.items())
    candidate.write_text(render_formula("4.1.0", digest))
    archived.write_text(render_formula("4.1.0", "b" * 64))
    current.write_text(render_formula("4.0.0", "b" * 64))
    before = current.read_bytes()
    rejected(lambda: prepare(candidate, current, "4.1.0", digest))
    assert current.read_bytes() == before
    archived.unlink()
    current.write_text(render_formula("4.1.0", "c" * 64))
    rejected(lambda: prepare(candidate, current, "4.1.0", digest))
    assert not archived.exists()
    current.write_text(render_formula("4.2.0", digest))
    before = current.read_bytes()
    assert prepare(candidate, current, "4.1.0", digest) == "updated"
    assert current.read_bytes() == before and archived.exists()
    for version, checksum in [("1.1.1", digest), ("../bad", digest), ("4.1.0", "bad"), ("4.1.0-beta", digest)]:
        rejected(lambda: render_formula(version, checksum))

    def git(path, *args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()

    remote = root / "remote.git"
    remote.mkdir()
    git(remote, "init", "--bare", "--initial-branch=main")
    seed = root / "seed"
    git(root, "clone", str(remote), str(seed))
    git(seed, "config", "user.name", "test")
    git(seed, "config", "user.email", "test@example.invalid")
    (seed / "Formula").mkdir()
    (seed / "Formula/lana.rb").write_text(render_formula("4.0.0", digest))
    git(seed, "add", ".")
    git(seed, "commit", "-m", "initial")
    git(seed, "push", "origin", "main")
    clones = [root / "older", root / "newer"]
    for clone in clones:
        git(root, "clone", str(remote), str(clone))
        git(clone, "config", "user.name", "test")
        git(clone, "config", "user.email", "test@example.invalid")
    newer = root / "newer.rb"
    newer.write_text(render_formula("4.2.0", digest))
    original_run = subprocess.run
    raced = False

    def race(command, *args, **kwargs):
        global raced
        if not raced and command[:4] == ["git", "-C", str(clones[0]), "push"]:
            raced = True
            assert publish(newer, clones[1] / "Formula/lana.rb", "4.2.0", digest) == "published"
        return original_run(command, *args, **kwargs)

    # Include staged admission, as the release workflow audits before publishing.
    prepare(candidate, clones[0] / "Formula/lana.rb", "4.1.0", digest)
    git(clones[0], "add", "Formula/lana.rb", "Formula/lana@4.1.0.rb")
    with patch("release_tap.subprocess.run", side_effect=race):
        assert publish(candidate, clones[0] / "Formula/lana.rb", "4.1.0", digest) == "published"
    assert raced
    assert "v4.2.0/" in git(remote, "show", "main:Formula/lana.rb")
    for version in ["4.1.0", "4.2.0"]:
        assert f"v{version}/" in git(remote, "show", f"main:Formula/lana@{version}.rb")
    git(clones[0], "fetch", "origin", "main")
    assert publish(candidate, clones[0] / "Formula/lana.rb", "4.1.0", digest) == "already-matching"
    (clones[0] / "unrelated.txt").write_text("keep")
    rejected(lambda: publish(candidate, clones[0] / "Formula/lana.rb", "4.1.0", digest))
    assert (clones[0] / "unrelated.txt").read_text() == "keep"

print("RELEASE_TAP_PASS")
