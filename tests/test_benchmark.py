"""Check the current paired gate's warmup exclusion, pairing, and failure threshold."""
import importlib.util
from pathlib import Path

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


if __name__ == '__main__':
    test_paired_method()
    print('BENCHMARK_METHOD_PASS')
