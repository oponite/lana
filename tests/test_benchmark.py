"""Check the current paired gate's warmup exclusion, pairing, and failure threshold."""
import importlib.util
from pathlib import Path
from contextlib import contextmanager
import tempfile
import json
import sys
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('lana_benchmark', Path(__file__).resolve().parents[1] / 'tools/benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_paired_method():
    calls = []

    def timing(side):
        def call():
            calls.append(side)
            return 1.0 if side == 'old' else 1.051
        return call

    report = benchmark.paired(timing('old'), timing('new'))
    assert calls[:4] == ['old', 'new', 'new', 'old']
    assert len(calls) == 60 and not report['pass']
    assert report['baseline']['warmups'] == 5
    result = benchmark.summarize([100.0] * 5 + [0.001] * 25)
    assert result['first_ms'] == 100000 and result['warm_median_ms'] == 1
    assert benchmark.paired(lambda: 1, lambda: 1.05)['pass']
    for samples in ([], [0], [float('nan')] * 30, [float('inf')] * 30):
        try:
            benchmark.summarize(samples)
        except ValueError:
            pass
        else:
            raise AssertionError(samples)


def test_versions_and_workload_boundaries():
    for version in ('4.1.1', '4.2.0'):
        assert not benchmark.validate_versions(['Lana 4.0.0 (test)', f'Lana {version} (test)'], version)
    for old, new, expected in [('4.0.0', '4.1.0', '4.1.1'), ('9.0.0', '4.1.1', '4.1.1')]:
        try:
            benchmark.validate_versions([f'Lana {old} (test)', f'Lana {new} (test)'], expected)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid version accepted')
    commands = []
    @contextmanager
    def calls(*args):
        yield lambda: 1.0
    def timed(command, env, cwd):
        commands.append(command)
        if command[1] == 'compile': Path(command[-1]).write_bytes(b'bytecode')
        return 1.0
    def output(command, **options):
        if command[1] == 'version': return 'Lana ' + ('4.0.0' if command[0].name == 'old' else '4.1.1') + ' (test)'
        assert Path(command[-1]).suffix == '.labc', command
        return b'ISA_OPS_PASS'
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'VERSION').write_text('4.1.1')
        for name in ('old', 'new', 'compiler'): (root / name).write_bytes(b'fixture')
        arguments = ['benchmark', '--baseline-root', str(root), '--baseline-cli', str(root / 'old'),
                     '--candidate-root', str(root), '--candidate-cli', str(root / 'new'),
                     '--compiler', str(root / 'compiler'), '--output', str(root / 'result.json')]
        with patch.object(sys, 'argv', arguments), patch.object(benchmark, 'timed', timed), \
             patch.object(benchmark, 'python_call', calls), patch.object(benchmark.subprocess, 'check_output', output):
            assert benchmark.main() == 0
        report = json.loads((root / 'result.json').read_text())
        assert report['schema'] == 2 and 'precompiled_cli' in report['boundaries']
        assert 'rust_vm' not in report['workloads']
        runs = [command for command in commands if command[1] == 'run']
        assert len(runs) == 60 and all(Path(command[-1]).suffix == '.labc' for command in runs)
        first_run = commands.index(runs[0])
        assert any(command[1] == 'compile' and command[2].name == 'isa_ops_pass.lana' for command in commands[:first_run])


if __name__ == '__main__':
    test_paired_method()
    test_versions_and_workload_boundaries()
    print('BENCHMARK_METHOD_PASS')
