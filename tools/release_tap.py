#!/usr/bin/env python3
"""Admit the exact published formula to the latest-version tap."""
import argparse
from pathlib import Path
import re


URL = re.compile(r'url "https://github\.com/oponite/lana/releases/download/v(\d+\.\d+\.\d+)/lana-\1-source\.tar\.gz"')
SHA = re.compile(r'sha256 "([0-9a-f]{64})"')


def identity(path):
    text = path.read_text()
    version = URL.search(text)
    digest = SHA.search(text)
    if not version or not digest:
        raise ValueError(f"invalid Lana formula: {path}")
    return tuple(map(int, version.group(1).split('.'))), digest.group(1)


def prepare(candidate, current, version, digest):
    wanted = tuple(map(int, version.split('.')))
    if identity(candidate) != (wanted, digest):
        raise ValueError("release formula URL or source checksum does not match published assets")
    if current.exists():
        installed, _ = identity(current)
        if installed > wanted:
            raise ValueError("tap already has a newer Lana version")
        if installed == wanted:
            if current.read_bytes() != candidate.read_bytes():
                raise ValueError("same-version tap formula differs from the published formula")
            return "already-matching"
    current.write_bytes(candidate.read_bytes())
    return "updated"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("version")
    parser.add_argument("digest")
    args = parser.parse_args()
    print(prepare(args.candidate, args.current, args.version, args.digest))
