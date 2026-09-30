"""Publication preflight and idempotency; GitHub mutations are mocked."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
from unittest.mock import patch

repo = Path(__file__).resolve().parents[1]
lana = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('package_release', repo / 'tools/package_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
env = dict(GITHUB_REPOSITORY='owner/demo', GITHUB_REF_NAME='lana-v1.0.0',
           GITHUB_SHA='a' * 40, LANA_REF_PROTECTED='true',
           LANA_COMPILER_LABC=os.environ.get('LANA_COMPILER_LABC', str(lana.parent / 'lana-compiler.labc')),
           LANA_STDLIB_DIR=str(repo / 'stdlib'))

with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, env):
    root = Path(temporary)
    source, output = root / 'source', root / 'output'
    source.mkdir()
    (source / 'src').mkdir()
    (source / 'tests').mkdir()
    (source / 'lana.toml').write_text('schema = 1\nname = "demo"\nversion = "1.0.0"\nentry = "src/main.lana"\n')
    (source / 'src/main.lana').write_text('fn value() { return 7; }\n')
    (source / 'tests/test.lana').write_text('import "../src/main.lana" as demo;\nassert(demo.value() == 7, "value");\n')
    prepared = release.prepare(lana, source, output)
    asset = prepared['asset']
    assert hashlib.sha256((output / asset).read_bytes()).hexdigest() == prepared['sha256']
    (source / 'tests/test.lana').write_text('assert(false, "failed test");\n')
    try:
        release.prepare(lana, source, root / 'failed')
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError('failed package tests must prevent qualification')
    calls = []
    def absent(path):
        if '/commits/' in path:
            return {'sha': env['GITHUB_SHA']}
        raise urllib.error.HTTPError(path, 404, 'missing', {}, None)
    with patch.object(release, 'api', absent), patch.object(release, 'run', lambda *args, **kw: calls.append(args)):
        assert release.publish(output)['status'] == 'published'
        assert len(calls) == 1 and calls[0][:3] == ('gh', 'release', 'create')
        assert '--verify-tag' in calls[0] and '--clobber' not in calls[0]
    def exists(path):
        if '/commits/' in path:
            return {'sha': env['GITHUB_SHA']}
        return {'draft': False, 'prerelease': False, 'assets': [{'name': asset}, {'name': 'SHA256SUMS'}]}
    def download(*args):
        assert args[:3] == ('gh', 'release', 'download')
        destination = Path(args[args.index('--dir') + 1])
        for name in (asset, 'SHA256SUMS'):
            (destination / name).write_bytes((output / name).read_bytes())
    with patch.object(release, 'api', exists), patch.object(release, 'run', download):
        assert release.publish(output)['status'] == 'unchanged'
    def changed(*args):
        download(*args)
        (Path(args[args.index('--dir') + 1]) / asset).write_bytes(b'changed')
    for api, runner in ((exists, changed), (lambda _: {'sha': 'b' * 40}, download)):
        with patch.object(release, 'api', api), patch.object(release, 'run', runner):
            try:
                release.publish(output)
            except ValueError:
                pass
            else:
                raise AssertionError('changed asset/tag must fail')
    with patch.dict(os.environ, LANA_REF_PROTECTED='false'):
        try:
            release.publish(output)
        except ValueError:
            pass
        else:
            raise AssertionError('unprotected tag must fail')
print('PACKAGE_RELEASE_PASS')
