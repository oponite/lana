"""Assemble and run every .lasm fixture; compare exit, stdout, stderr with <name>.expect.json.

The C reference VM that these fixtures were once diffed against is retired, so the
expectations record the Rust VM's reviewed output. Refresh them with --update.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def run_fixture(lana: str, fixture: Path, work: Path) -> dict:
    chunk = work / (fixture.stem + ".labc")
    assembled = subprocess.run([lana, "asm", str(fixture), "-o", str(chunk)],
                               capture_output=True, text=True, timeout=30)
    assert assembled.returncode == 0, (fixture.name, assembled.stderr)
    args_file = fixture.with_suffix(".args")
    extra = args_file.read_text().split() if args_file.exists() else []
    result = subprocess.run([lana, "run-bytecode", str(chunk), *extra],
                            cwd=work, capture_output=True, text=True, timeout=30)
    return {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main() -> None:
    lana = str(Path(sys.argv[1]).resolve())
    update = "--update" in sys.argv[2:]
    fixtures = sorted(ROOT.glob("*/*.lasm"))
    assert fixtures, "no fixtures found"
    failures = []
    for fixture in fixtures:
        with tempfile.TemporaryDirectory(prefix="lana-differential-") as directory:
            actual = run_fixture(lana, fixture, Path(directory))
        expect_file = fixture.with_suffix(".expect.json")
        if update:
            expect_file.write_text(json.dumps(actual, indent=2) + "\n")
        elif not expect_file.exists() or json.loads(expect_file.read_text()) != actual:
            failures.append((fixture.relative_to(ROOT), actual))
    for name, actual in failures:
        print(f"FAIL {name}: {json.dumps(actual)}", file=sys.stderr)
    if failures:
        sys.exit(1)
    print(f"DIFFERENTIAL_PASS {len(fixtures)}")


if __name__ == "__main__":
    main()
