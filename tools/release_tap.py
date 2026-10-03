#!/usr/bin/env python3
"""Generate and admit the exact latest-version Homebrew formula."""
import argparse
from pathlib import Path
import re


URL = re.compile(r'url "https://github\.com/oponite/lana/releases/download/v(\d+\.\d+\.\d+)/lana-\1-source\.tar\.gz"')
SHA = re.compile(r'sha256 "([0-9a-f]{64})"')


def render_formula(version, digest):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid release version or source checksum")
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
    wanted = tuple(map(int, version.split('.')))
    if current.exists():
        installed, _ = identity(current)
        if installed > wanted:
            raise ValueError("tap already has a newer Lana version")
        if installed == wanted:
            if current.read_bytes() != expected:
                raise ValueError("same-version tap formula differs from the published formula")
            return "already-matching"
    current.write_bytes(expected)
    return "updated"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("output", type=Path)
    generate.add_argument("version")
    generate.add_argument("digest")
    admit = commands.add_parser("admit")
    admit.add_argument("candidate", type=Path)
    admit.add_argument("current", type=Path)
    admit.add_argument("version")
    admit.add_argument("digest")
    args = parser.parse_args()
    if args.command == "generate":
        args.output.write_text(render_formula(args.version, args.digest))
    else:
        print(prepare(args.candidate, args.current, args.version, args.digest))
