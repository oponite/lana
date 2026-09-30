"""Source-package pack/add/import checks; --network-fixture needs the test-origin build."""
import gzip
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading

lana = Path(sys.argv[1]).resolve()
repo = Path(__file__).resolve().parents[1]
network = '--network-fixture' in sys.argv
faults = '--fault-injection' in sys.argv
env = dict(os.environ, LANA_COMPILER_LABC=os.environ.get('LANA_COMPILER_LABC', str(lana.parent / 'lana-compiler.labc')), LANA_STDLIB_DIR=str(repo / 'stdlib'))

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    app = root / 'app'
    app.mkdir()
    (app / 'src').mkdir()
    (app / 'lana.toml').write_text('schema = 1\nname = "app"\nversion = "0.1.0"\nentry = "src/main.lana"\n[dependencies]\n')
    (app / 'src/main.lana').write_text('print(7);\n')

    def call(*args, ok=True, stage=None, cwd=app):
        child_env = dict(env)
        if stage:
            child_env['LANA_TEST_ATOMIC_STAGE'] = stage
        result = subprocess.run([lana, *map(str, args)], cwd=cwd, env=child_env,
                                capture_output=True, text=True, timeout=120)
        assert (result.returncode == 0) == ok, (args, result.stdout, result.stderr)
        return result

    def make(name, version='1.0.0', dependencies=None, source='fn value() { return 7; }\n'):
        directory = root / f'{name}-{version}'
        directory.mkdir(exist_ok=True)
        (directory / 'src').mkdir(exist_ok=True)
        (directory / 'tests').mkdir(exist_ok=True)
        manifest = f'schema = 1\nname = "{name}"\nversion = "{version}"\nentry = "src/main.lana"\n'
        if dependencies:
            manifest += '[hosted_dependencies]\n' + ''.join(f'{alias} = "{dep}"\n' for alias, dep in dependencies.items())
        (directory / 'lana.toml').write_text(manifest)
        (directory / 'src/main.lana').write_text(source)
        (directory / 'tests/main.lana').write_text('print(1);\n')
        destination = root / f'{name}-{version}-lana.tar.gz'
        report = json.loads(call('package', 'pack', directory, '-o', destination).stdout)
        assert report['sha256'] == hashlib.sha256(destination.read_bytes()).hexdigest()
        return directory, destination

    directory, packed = make('demo')
    original = packed.read_bytes()
    os.utime(directory / 'src/main.lana', (12345, 12345))
    call('package', 'pack', directory, '-o', packed)
    assert packed.read_bytes() == original
    assert original[3:8] == b'\0' * 5 and original[9] == 255
    with tarfile.open(fileobj=io.BytesIO(original), mode='r:gz') as archive:
        names = archive.getnames()
        assert names == sorted(names)
        for member in archive.getmembers():
            assert member.uid == member.gid == member.mtime == 0
            assert member.mode == (0o755 if member.isdir() else 0o644)
            assert member.uname == member.gname == ''
    raw = gzip.decompress(original)
    assert raw.endswith(b'\0' * 1024)
    manifest = (directory / 'lana.toml').read_text()
    for suffix in ('[dependencies]\nlocal = "../app"\n', 'build = "execute-me"\n'):
        (directory / 'lana.toml').write_text(manifest + suffix)
        call('package', 'pack', directory, '-o', packed, ok=False)
        assert packed.read_bytes() == original
    (directory / 'lana.toml').write_text(manifest)
    if hasattr(os, 'symlink'):
        (directory / 'src/link').symlink_to(app / 'src/main.lana')
        call('package', 'pack', directory, '-o', packed, ok=False)
        (directory / 'src/link').unlink()
    call('package', 'pack', directory, '-o', directory / 'src/output.gz', ok=False)
    for i in range(1000):
        (directory / 'src' / f'limit-{i}.lana').write_text('')
    assert 'LANA_ERR_LIMIT' in call('package', 'pack', directory, '-o', packed, ok=False).stderr
    for path in (directory / 'src').glob('limit-*.lana'):
        path.unlink()
    assert packed.read_bytes() == original
    for key in ('Owner/demo@1.0.0', 'owner/demo@1.0', 'owner/demo@1.0.0-beta', '../demo@1.0.0'):
        call('package', 'add', key, ok=False)

    if network:
        routes = {}
        partial = set()
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                body = routes.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Length', str(len(body) + (10 if self.path in partial else 0)))
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env['LANA_TEST_PACKAGE_ORIGIN'] = f'http://127.0.0.1:{server.server_port}'
        def release(name, version='1.0.0', dependencies=None, source='fn value() { return 7; }\n'):
            _, path = make(name, version, dependencies, source)
            key = f'/owner/{name}/releases/download/lana-v{version}/'
            asset = f'{name}-{version}-lana.tar.gz'
            data = path.read_bytes()
            routes[key + asset] = data
            routes[key + 'SHA256SUMS'] = f'{hashlib.sha256(data).hexdigest()}  {asset}\n'.encode()
            return key, asset
        def replace(key, asset, data):
            routes[key + asset] = data
            routes[key + 'SHA256SUMS'] = f'{hashlib.sha256(data).hexdigest()}  {asset}\n'.encode()
        try:
            local, _ = make('local')
            with (app / 'lana.toml').open('a') as manifest:
                manifest.write('local = "../local-1.0.0"\n')
            release('leaf')
            key, asset = release('demo', dependencies={'leaf': 'owner/leaf@1.0.0'},
                source='import "pkg/owner/leaf/src/main.lana" as leaf;\nfn value() { return leaf.value(); }\n')
            # Migration preserves ordinary local-path build behavior.
            call('build')
            assert (app / 'lana.lock').read_text().startswith('schema = 1\n')
            added = json.loads(call('package', 'add', 'owner/demo@1.0.0').stdout)
            assert added['packages'] == 2
            lock_path = app / 'lana.lock'
            saved = lock_path.read_bytes()
            lock = json.loads(saved)
            assert saved == (json.dumps(lock, sort_keys=True, separators=(',', ':')) + '\n').encode()
            assert [p['identity'] for p in lock['packages']] == ['owner/demo', 'owner/leaf']
            assert lock['direct'] == ['owner/demo@1.0.0']
            assert lock['packages'][0]['dependencies'] == ['owner/leaf@1.0.0']
            stamp = lock_path.stat().st_mtime_ns
            assert not json.loads(call('package', 'add', 'owner/demo@1.0.0').stdout)['changed']
            assert lock_path.stat().st_mtime_ns == stamp
            (app / 'src/main.lana').write_text('import "pkg/owner/demo/src/main.lana" as demo;\nprint(demo.value());\n')
            assert call('run', 'src/main.lana').stdout.strip() == '7'
            call('build')
            assert lock_path.read_bytes() == saved
            prior_builds = set((app / '.lana/cache').glob('*.labc'))
            (local / 'src/main.lana').write_text('fn value() { return 8; }\n')
            call('build')
            assert set((app / '.lana/cache').glob('*.labc')) != prior_builds
            assert lock_path.read_bytes() == saved
            # Cached builds recheck both archive and extracted bytes.
            demo = lock['packages'][0]
            cache = app / '.lana/packages' / demo['sha256']
            source = cache / 'demo-1.0.0/src/main.lana'
            original_source = source.read_bytes()
            source.write_text('fn value() { return 999; }\n')
            call('build', ok=False)
            source.write_bytes(original_source)
            cached_archive = cache / 'archive.tar.gz'
            archive_bytes = cached_archive.read_bytes()
            cached_archive.write_bytes(b'bad')
            call('build', ok=False)
            cached_archive.write_bytes(archive_bytes)
            call('build')
            for target in ('pkg/owner/demo/src/../../escape.lana', 'pkg/owner/unknown/src/main.lana'):
                (app / 'src/main.lana').write_text(f'import "{target}" as demo;\nprint(demo.value());\n')
                call('compile', 'src/main.lana', '-o', str(root / 'compiled.labc'), ok=False)
            (app / 'src/main.lana').write_text('import "pkg/owner/demo/src/main.lana" as demo;\nprint(demo.value());\n')
            # Every failed add leaves the prior lock and build usable.
            def rejected(package):
                call('package', 'add', package, ok=False)
                assert lock_path.read_bytes() == saved
                call('build')
            old_asset, old_sum = routes[key + asset], routes[key + 'SHA256SUMS']
            replace(key, asset, b'replaced release')
            rejected('owner/demo@1.0.0')
            routes[key + asset], routes[key + 'SHA256SUMS'] = old_asset, old_sum
            bad_key, bad_asset = release('bad')
            routes[bad_key + bad_asset] = b'bad digest'
            rejected('owner/bad@1.0.0')
            partial_key, partial_asset = release('partial')
            partial.add(partial_key + partial_asset)
            rejected('owner/partial@1.0.0')
            release('cyclea', dependencies={'b': 'owner/cycleb@1.0.0'})
            release('cycleb', dependencies={'a': 'owner/cyclea@1.0.0'})
            rejected('owner/cyclea@1.0.0')
            release('conflict', dependencies={'leaf': 'owner/leaf@2.0.0'})
            rejected('owner/conflict@1.0.0')
            release('missing', dependencies={'gone': 'owner/gone@1.0.0'})
            rejected('owner/missing@1.0.0')
            unsafe_key, unsafe_asset = release('unsafe')
            for member_name, kind in (('../escape', tarfile.REGTYPE), ('unsafe-1.0.0/src/link', tarfile.SYMTYPE)):
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode='w', format=tarfile.USTAR_FORMAT) as archive:
                    info = tarfile.TarInfo(member_name)
                    info.type = kind
                    if kind == tarfile.SYMTYPE:
                        info.linkname = '/tmp/escape'
                    archive.addfile(info)
                replace(unsafe_key, unsafe_asset, gzip.compress(buffer.getvalue(), mtime=0))
                rejected('owner/unsafe@1.0.0')
            wrong_key, wrong_asset = release('wrong')
            replace(wrong_key, wrong_asset, old_asset)
            rejected('owner/wrong@1.0.0')
            leaf_digest = lock['packages'][1]['sha256']
            release('escape', dependencies={'leaf': 'owner/leaf@1.0.0'}, source=
                f'import "../../../{leaf_digest}/leaf-1.0.0/src/main.lana" as leaf;\nfn value() {{ return leaf.value(); }}\n')
            call('package', 'add', 'owner/escape@1.0.0')
            escape_digest = next(p['sha256'] for p in json.loads(lock_path.read_bytes())['packages'] if p['identity'] == 'owner/escape')
            (app / 'src/main.lana').write_text('import "pkg/owner/escape/src/main.lana" as escape;\nprint(escape.value());\n')
            call('compile', 'src/main.lana', '-o', str(root / 'compiled.labc'), ok=False)
            lock_path.write_bytes(saved)
            (app / 'src/main.lana').write_text(f'import "../.lana/packages/{escape_digest}/escape-1.0.0/src/main.lana" as escape;\nprint(escape.value());\n')
            assert 'LANA_ERR_UNSUPPORTED_VALUE' in call('compile', 'src/main.lana', '-o', str(root / 'compiled.labc'), ok=False).stderr
            # Top-level source is validated but never executed during add.
            release('effect', source='write_text("should-not-exist", "bad");\nfn value() { return 1; }\n')
            call('package', 'add', 'owner/effect@1.0.0')
            assert not (app / 'should-not-exist').exists()
            (app / 'src/main.lana').write_text('import "pkg/owner/effect/src/main.lana" as effect;\nprint(effect.value());\n')
            call('compile', 'src/main.lana', '-o', str(root / 'compiled.labc'), ok=False)
            (app / 'src/main.lana').write_text('import "pkg/owner/demo/src/main.lana" as demo;\nprint(demo.value());\n')
            lock_path.write_bytes(saved)
            if faults:
                # Promote an already cached transitive dependency: only lock publication writes.
                for stage in ('before_file_sync', 'before_rename', 'after_rename'):
                    lock_path.write_bytes(saved)
                    error = call('package', 'add', 'owner/leaf@1.0.0', ok=False, stage=stage).stderr
                    assert 'LANA_ERR_IO' in error
                    assert ('uncertain' in error) == (stage == 'after_rename')
                    assert (lock_path.read_bytes() != saved) == (stage == 'after_rename')
                    call('build')
                lock_path.write_bytes(saved)
            # Offline builds use the verified cache only.
            server.shutdown()
            thread.join()
            call('build')
            assert call('run', 'src/main.lana').stdout.strip() == '7'
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
print('PACKAGES_PASS')
