"""Seeded loader/verifier mutations; preserve mismatches for reproduction."""
import argparse
import hashlib
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def outcome(binary, path):
    result = subprocess.run([str(binary), "verify", str(path)], capture_output=True, timeout=5)
    assert result.returncode in (0, 1), (binary, result.returncode, result.stderr)
    if result.returncode == 0:
        assert result.stderr == b"", result.stderr
        assert result.stdout.endswith(b"\nverified\n"), result.stdout
        return "LANA_OK", result.stdout
    assert result.stdout == b"", (binary, result.stdout)
    errors = re.findall(rb"error\[(?:[a-z_-]+/)?(LANA_ERR_[A-Z_]+)\]", result.stderr)
    assert len(errors) == 1, (binary, result.stderr)
    return errors[0].decode(), b""


def mutations(valid, count, seed):
    yield "valid", valid, True
    yield "bad-magic", b"FAIL" + valid[4:], False
    yield "trailing", valid + b"\x00", False
    for length in range(len(valid)):
        yield f"truncated-{length}", valid[:length], False
    rng = random.Random(seed)
    for index in range(count):
        value = bytearray(valid)
        for _ in range(rng.randint(1, 4)):
            offset = rng.randrange(len(value))
            value[offset] ^= 1 << rng.randrange(8)
        yield f"mutated-{index}", bytes(value), None
        yield f"random-{index}", rng.randbytes(rng.randrange(256)), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c11", type=Path, default=os.environ.get("C11", ROOT / "build/lanavm"))
    parser.add_argument("--rust", type=Path, default=os.environ.get("RUST", ROOT / "target/debug/lana-cli"))
    parser.add_argument("--cases", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "build/bytecode-failures")
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    binaries = [args.c11.resolve(), args.rust.resolve()]
    count = 0
    with tempfile.TemporaryDirectory(prefix="lana-bytecode-") as directory:
        directory = Path(directory)
        assembly = directory / "seed.lasm"
        assembly.write_text("LOAD_CONST R0 42\nPRINT R0\nHALT\n")
        path = directory / "case.labc"
        subprocess.run([str(binaries[0]), "asm", str(assembly), "-o", str(path)],
                       capture_output=True, check=True, timeout=5)
        valid = path.read_bytes()
        for name, data, expected in mutations(valid, args.cases, args.seed):
            path.write_bytes(data)
            try:
                left, right = [outcome(binary, path) for binary in binaries]
                assert left == right, (name, left, right)
                if expected is not None:
                    assert (left[0] == "LANA_OK") == expected, (name, left, expected)
            except (AssertionError, OSError, subprocess.TimeoutExpired):
                args.artifacts.mkdir(parents=True, exist_ok=True)
                saved = args.artifacts / f"{name}-{hashlib.sha256(data).hexdigest()[:12]}.labc"
                saved.write_bytes(data)
                print(f"Failing input: {saved}", flush=True)
                raise
            count += 1
    print(f"{count}/{count} bytecode cases passed (seed {args.seed})")


if __name__ == "__main__":
    main()
