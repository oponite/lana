"""Paired Release gate from docs/dev/RELEASE_CHECKLIST.md (30 calls, discard 5)."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import re
from pathlib import Path
import select
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def summarize(samples, warmups=5):
    if not 0 <= warmups < len(samples) or any(not 0 < value < float('inf') for value in samples):
        raise ValueError('need finite positive timings and at least one warm sample')
    return {'first_ms': samples[0] * 1000, 'warm_median_ms': statistics.median(samples[warmups:]) * 1000,
            'samples_seconds': samples, 'warmups': warmups}


def paired(old, new, runs=30):
    samples = [[], []]
    for index in range(runs):
        for side in ((0, 1) if index % 2 == 0 else (1, 0)):
            samples[side].append((old, new)[side]())
    before, after = map(summarize, samples)
    ratio = after['warm_median_ms'] / before['warm_median_ms']
    return {'baseline': before, 'candidate': after, 'ratio': ratio, 'pass': ratio <= 1.05}


def validate_versions(versions, expected):
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", expected):
        raise ValueError("candidate VERSION is not a release version")
    baselines = ('3.0.2', '4.0.0', '4.1.0')
    if not any(versions[0].startswith(f'Lana {version} ') for version in baselines):
        raise ValueError(f'unsupported baseline: {versions[0]}')
    if not versions[1].startswith(f'Lana {expected} '):
        raise ValueError(f'candidate executable does not match VERSION {expected}: {versions[1]}')
    return versions[0].startswith('Lana 3.0.2 ')


def timed(command, env, cwd):
    start = time.perf_counter()
    subprocess.run(command, env=env, cwd=cwd, capture_output=True, check=True, timeout=30)
    return time.perf_counter() - start


def worker():
    config = json.loads(sys.stdin.readline())
    sys.path.insert(0, str(Path(config['root']) / 'integrations/python/src'))
    from lana_integrations.lana import Lana
    options = {'executable': config['cli']}
    if config.get('library'):
        options['library'] = config['library']
    lana = Lana(**options)
    print(json.dumps({'backend': lana.backend}), flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            start = time.perf_counter()
            result = getattr(lana, request['operation'])(request['path'], request['input'])
            elapsed = time.perf_counter() - start
            assert result.ok and result.value == request['input'], repr(result)
            print(json.dumps({'seconds': elapsed, 'backend': result.backend}), flush=True)
    finally:
        if hasattr(lana, 'close'):
            lana.close()


def receive(process):
    ready, _, _ = select.select([process.stdout], [], [], 40)
    if not ready:
        raise RuntimeError('benchmark worker timed out')
    line = process.stdout.readline()
    if not line:
        raise RuntimeError('benchmark worker exited before returning a result')
    return json.loads(line)


@contextmanager
def python_call(config, operation, source, env):
    with subprocess.Popen([sys.executable, __file__, '--worker'], env=env, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True) as process:
        try:
            process.stdin.write(json.dumps(config) + '\n'); process.stdin.flush()
            expected = 'native' if config.get('library') else 'worker'
            assert receive(process)['backend'] == expected, 'benchmark backend silently changed'

            def call():
                process.stdin.write(json.dumps({'operation': operation, 'path': str(source),
                                                'input': {'x': 17, 'text': 'paired'}}) + '\n')
                process.stdin.flush()
                response = receive(process)
                assert response['backend'] == expected, response
                return response['seconds']

            yield call
            process.stdin.close()
            process.wait(timeout=10)
            assert process.returncode == 0
        finally:
            if process.poll() is None:
                process.kill(); process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--baseline-cli', type=Path, required=True)
    parser.add_argument('--baseline-library', type=Path, help='required for the 3.0.2 native bridge')
    parser.add_argument('--candidate-root', type=Path, default=ROOT)
    parser.add_argument('--candidate-cli', type=Path)
    parser.add_argument('--compiler', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old_root, new_root = args.baseline_root.resolve(), args.candidate_root.resolve()
    old_cli = args.baseline_cli.resolve()
    new_cli = (args.candidate_cli or new_root / 'target/lana/bin/lana').resolve()
    compiler = (args.compiler or new_cli.parent / 'lana-compiler.labc').resolve()
    env = dict(os.environ, LANA_COMPILER_LABC=str(compiler), LANA_STDLIB_DIR=str(new_root / 'stdlib'))
    versions = [subprocess.check_output([cli, 'version'], text=True).strip() for cli in (old_cli, new_cli)]
    try:
        baseline_3 = validate_versions(versions, (new_root / 'VERSION').read_text().strip())
    except ValueError as error:
        parser.error(str(error))
    if baseline_3 and args.baseline_library is None:
        parser.error('--baseline-library is required for Lana 3.0.2')
    old_config = {'root': str(old_root), 'cli': str(old_cli)}
    if baseline_3:
        old_config['library'] = str(args.baseline_library.resolve())
    configs = [old_config, {'root': str(new_root), 'cli': str(new_cli)}]
    old_source = 'echo_bridge_c11.lana' if baseline_3 else 'echo_bridge.lana'
    sources = [old_root / 'integrations/lana' / old_source, new_root / 'integrations/lana/echo_bridge.lana']
    with tempfile.TemporaryDirectory(prefix='lana-paired-benchmark-') as directory:
        work = Path(directory)
        bytecodes = [work / 'old.labc', work / 'new.labc']
        for cli, source, bytecode in zip((old_cli, new_cli), sources, bytecodes):
            timed([cli, 'compile', source, '-o', bytecode], env, work)
        workloads = {}
        for name, operation, paths in [('python_bytecode', 'run_labc', bytecodes), ('python_source', 'run', sources)]:
            with python_call(configs[0], operation, paths[0], env) as old, python_call(configs[1], operation, paths[1], env) as new:
                workloads[name] = paired(old, new)
        fixture_source = new_root / 'tests/regression/isa_ops_pass.lana'
        fixture = work / 'isa.labc'
        timed([new_cli, 'compile', fixture_source, '-o', fixture], env, work)
        outputs = [subprocess.check_output([cli, 'run', fixture], env=env, cwd=work, timeout=30)
                   for cli in (old_cli, new_cli)]
        if outputs[0] != outputs[1]:
            raise RuntimeError('precompiled CLI workloads returned different output')
        fixture_sha256 = hashlib.sha256(fixture.read_bytes()).hexdigest()
        workloads['precompiled_cli'] = paired(lambda: timed([old_cli, 'run', fixture], env, work),
                                             lambda: timed([new_cli, 'run', fixture], env, work))
        source = sources[1]
        workloads['rust_compile'] = paired(lambda: timed([old_cli, 'compile', source, '-o', work / 'old-output.labc'], env, work),
                                           lambda: timed([new_cli, 'compile', source, '-o', work / 'new-output.labc'], env, work))
    report = {'schema': 2, 'fixture_sha256': fixture_sha256, 'versions': versions, 'runs': 30, 'warmups': 5,
              'compiler_sha256': hashlib.sha256(compiler.read_bytes()).hexdigest(),
              'binary_sha256': [hashlib.sha256(cli.read_bytes()).hexdigest() for cli in (old_cli, new_cli)],
               'boundaries': {'python_bytecode': 'Python API call: worker or native bridge, bytecode load, VM execution; excludes Python process startup',
                             'python_source': 'Python API call including source compilation; excludes Python process startup',
                             'precompiled_cli': 'CLI process startup, bytecode load and VM execution; excludes compilation',
                             'rust_compile': 'CLI process startup, source compilation and output publication'},
              'workloads': workloads, 'pass': all(item['pass'] for item in workloads.values())}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: {'ratio': value['ratio'], 'pass': value['pass']} for name, value in workloads.items()}, indent=2))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        worker()
    else:
        sys.exit(main())
