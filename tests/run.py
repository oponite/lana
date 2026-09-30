#!/usr/bin/env python3
"""Run the complete source/CLI regression and bootstrap suite without CMake."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build import build, DEFAULT
import workflows


def check_result(case, result):
    if result.returncode < 0:
        raise AssertionError(f"process terminated by signal {-result.returncode}")
    if (result.returncode != 0) != case.get("expect_failure", False):
        raise AssertionError(f"unexpected exit {result.returncode}")
    for pattern in case.get("contains", []):
        if re.search(pattern, result.stdout) is None:
            raise AssertionError(f"missing output pattern {pattern!r}")
    if "source_span" in case:
        if not re.search(re.escape(case["source_span"]) + r':\d+:\d+-\d+:\d+', result.stdout):
            raise AssertionError("missing full source span")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=DEFAULT)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--filter", default="", help="regular expression over preserved test names")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    cases = json.loads((ROOT / "tests/cases.json").read_text())
    if len({case['name'] for case in cases}) != len(cases):
        parser.error("duplicate test names")
    cases = [case for case in cases if re.search(args.filter, case['name'])]
    if not cases:
        parser.error("no tests matched")
    if args.list:
        print('\n'.join(case['name'] for case in cases))
        return 0
    prefix = args.prefix.resolve()
    if not args.no_build:
        build(prefix)
    cli, compiler = prefix / 'bin/lana', prefix / 'bin/lana-compiler.labc'
    if not cli.is_file() or not compiler.is_file():
        parser.error(f"missing built CLI/compiler in {prefix}/bin")
    failed = []
    start = time.monotonic()
    for case in cases:
        output = ''
        with tempfile.TemporaryDirectory(prefix='lana-test-') as directory:
            work = Path(directory)
            substitutions = dict(root=str(ROOT), cli=str(cli), compiler=str(compiler),
                                 work=str(work), bin=str(cli.parent), python=sys.executable)
            def expand(value):
                for key, replacement in substitutions.items():
                    value = value.replace('{' + key + '}', replacement)
                return value
            env = dict(os.environ, LANA_COMPILER_LABC=str(compiler), LANA_STDLIB_DIR=str(ROOT / 'stdlib'),
                       LANA_EXECUTABLE=str(cli), LANA_CLI=str(cli), PYTHONUNBUFFERED='1')
            env.update({k:expand(v) for k,v in case.get('env', {}).items()})
            try:
                if 'workflow' in case:
                    workflows.run(case['workflow'], cli, compiler, work, env)
                else:
                    result = workflows.run_process([expand(arg) for arg in case['command']],
                        cwd=expand(case.get('cwd', '{work}')), env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=case.get('timeout', 180))
                    output = result.stdout
                    expanded = dict(case)
                    if 'source_span' in case: expanded['source_span'] = expand(case['source_span'])
                    check_result(expanded, result)
                print(f"PASS {case['name']}", flush=True)
            except (OSError, ValueError, AssertionError, subprocess.TimeoutExpired) as error:
                failed.append(case['name'])
                print(f"FAIL {case['name']}: {error}\n{output}", flush=True)
    print(f"{len(cases) - len(failed)}/{len(cases)} passed in {time.monotonic() - start:.2f}s", flush=True)
    return int(bool(failed))


if __name__ == '__main__':
    sys.exit(main())
