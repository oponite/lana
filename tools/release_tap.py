#!/usr/bin/env python3
"""Generate, admit, and publish verified latest and versioned Homebrew formulas."""
import argparse
from pathlib import Path
import re
import subprocess


URL = re.compile(r'url "https://github\.com/oponite/lana/releases/download/v(\d+\.\d+\.\d+)/lana-\1-source\.tar\.gz"')
SHA = re.compile(r'sha256 "([0-9a-f]{64})"')


def render_formula(version, digest):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid release version or source checksum")
    if int(version.split(".")[0]) < 2:
        raise ValueError("the tap supports public Lana releases from 2.0.0 onward")
    return '''class Lana < Formula
  desc "Verified register-based language with explicit uncertainty"
  homepage "https://github.com/oponite/lana"
  url "https://github.com/oponite/lana/releases/download/v@VERSION@/lana-@VERSION@-source.tar.gz"
  sha256 "@SHA256@"
  license "Apache-2.0"

  depends_on "python@3.14" => :build
  depends_on "rust" => :build

  def install
    system "python3", "tools/build.py", "install", "--prefix", prefix
  end

  test do
    assert_match "Lana #{version} (LABC v2,", shell_output("#{bin}/lana version")
    system bin/"lana", "new", "hello-lana"
    Dir.chdir("hello-lana") do
      system bin/"lana", "build"
      system bin/"lana", "run"
    end
  end
end
'''.replace("@VERSION@", version).replace("@SHA256@", digest)


def identity(path):
    text = path.read_text()
    version = URL.search(text)
    digest = SHA.search(text)
    if not version or not digest:
        raise ValueError(f"invalid Lana formula: {path}")
    return tuple(map(int, version.group(1).split('.'))), digest.group(1)


def prepare(candidate, current, version, digest):
    expected = render_formula(version, digest).encode()
    if candidate.read_bytes() != expected:
        raise ValueError("release formula differs from the generated formula")
    archived = current.with_name(f"lana@{version}.rb")
    versioned = expected.replace(b"class Lana < Formula", f"class LanaAT{version.replace('.', '')} < Formula".encode()).replace(
        b'  license "Apache-2.0"\n', b'  license "Apache-2.0"\n\n  keg_only :versioned_formula\n'
    )
    if archived.exists() and archived.read_bytes() != versioned:
        raise ValueError("same-version archived formula differs from the published formula")
    wanted = tuple(map(int, version.split('.')))
    update_latest = True
    if current.exists():
        installed, _ = identity(current)
        if installed == wanted:
            if current.read_bytes() != expected:
                raise ValueError("same-version tap formula differs from the published formula")
        update_latest = installed < wanted
    changed = not archived.exists() or update_latest
    if not archived.exists():
        archived.write_bytes(versioned)
    if update_latest:
        current.write_bytes(expected)
    return "updated" if changed else "already-matching"


def publish(candidate, current, version, digest):
    """Publish from a disposable tap clone; retry only changes to remote main."""
    tap = current.parent.parent

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(tap), *arguments], text=True).rstrip("\n")

    paths = ["Formula/lana.rb", f"Formula/lana@{version}.rb"]
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
        raise ValueError("publication requires a fresh tap clone at origin/main")
    status = git("status", "--porcelain", "--untracked-files=all")
    if any(line[3:] not in paths for line in status.splitlines()):
        raise ValueError("tap clone contains unrelated changes")
    for attempt in range(3):
        prepare(candidate, current, version, digest)
        git("add", "--", *paths)
        if not git("diff", "--cached", "--name-only"):
            return "already-matching"
        base = git("rev-parse", "origin/main")
        git("commit", "-m", f"lana {version}")
        result = subprocess.run(["git", "-C", str(tap), "push", "origin", "HEAD:main"])
        if result.returncode == 0:
            return "published"
        git("fetch", "origin", "main")
        if git("rev-parse", "origin/main") == base or attempt == 2:
            raise RuntimeError("tap push failed; rerun publication from a fresh clone")
        # Only this function's commits and admitted files exist in the disposable clone.
        git("reset", "--hard", "origin/main")
    raise AssertionError("unreachable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("output", type=Path)
    generate.add_argument("version")
    generate.add_argument("digest")
    for name in ("admit", "publish"):
        command = commands.add_parser(name)
        command.add_argument("candidate", type=Path)
        command.add_argument("current", type=Path)
        command.add_argument("version")
        command.add_argument("digest")
    args = parser.parse_args()
    if args.command == "generate":
        args.output.write_text(render_formula(args.version, args.digest))
    elif args.command == "admit":
        print(prepare(args.candidate, args.current, args.version, args.digest))
    else:
        print(publish(args.candidate, args.current, args.version, args.digest))
