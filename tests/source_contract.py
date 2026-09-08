"""Assert a source failure, including its diagnostic and lack of output/effects."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STAGES = {"TYPE": "type", "PARSE": "parse", "ASSERTION": "assertion"}


def check_failure(result, code, expected):
    assert result.returncode == 1, (result.returncode, result.stderr)
    assert result.stdout == "", f"partial output: {result.stdout!r}"
    stage = STAGES.get(code.removeprefix("LANA_ERR_"), "validation")
    # A substring alone could match a secondary error, or the wrong stage.
    assert len(result.stderr.splitlines()) == 1, result.stderr
    assert re.fullmatch(
        rf".+:[0-9]+:[0-9]+-[0-9]+:[0-9]+: error\[{stage}/{code}\]: .+\n",
        result.stderr), result.stderr
    assert re.search(expected, result.stderr), result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lana", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mode", choices=("compile", "run"), default="compile")
    parser.add_argument("--code", default="LANA_ERR_TYPE")
    parser.add_argument("--expect", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lana-source-contract-") as directory:
        command = [str(args.lana.resolve()), args.mode, str(args.source.resolve())]
        if args.mode == "compile":
            command += ["-o", str(Path(directory) / "program.labc")]
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                                timeout=30, env={**os.environ, "LANA_STDLIB_DIR": str(ROOT / "stdlib")})
        check_failure(result, args.code, args.expect)
        assert not list(Path(directory).iterdir()), "failure left output files"


if __name__ == "__main__":
    main()
