"""Prepare or publish a protected-tag source package. Publication is explicit."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request


def run(*args, cwd=None):
    return subprocess.run(list(map(str, args)), cwd=cwd, check=True, capture_output=True, text=True).stdout


def identity():
    repository = os.environ['GITHUB_REPOSITORY']
    tag = os.environ['GITHUB_REF_NAME']
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*', repository) or not re.fullmatch(
            r'lana-v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)', tag):
        raise ValueError('invalid package repository or tag')
    if os.environ.get('LANA_REF_PROTECTED') != 'true':
        raise ValueError('publication requires a protected tag')
    name, version = repository.split('/')[1], tag.removeprefix('lana-v')
    return repository, tag, name, version, f'{name}-{version}-lana.tar.gz'


def prepare(lana, directory, output):
    _, _, name, version, asset = identity()
    lana = Path(lana).resolve()
    directory, output = Path(directory).resolve(), Path(output).resolve()
    output.mkdir(exist_ok=False)
    archive = output / asset
    report = json.loads(run(lana, 'package', 'pack', directory, '-o', archive))
    if (report['name'], report['version'], report['asset']) != (name, version, asset):
        raise ValueError('manifest does not match repository and protected tag')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != report['sha256']:
        raise ValueError('packed checksum mismatch')
    (output / 'SHA256SUMS').write_text(f'{digest}  {asset}\n')
    with tempfile.TemporaryDirectory(prefix='lana-package-release-') as temporary:
        with tarfile.open(archive, 'r:gz') as packed:
            packed.extractall(temporary, filter='data')
        clean = Path(temporary) / f'{name}-{version}'
        manifest = tomllib.loads((clean / 'lana.toml').read_text())
        for dependency in sorted(set(manifest.get('hosted_dependencies', {}).values())):
            run(lana, 'package', 'add', dependency, cwd=clean)
        run(lana, 'build', cwd=clean)
        if (clean / 'tests').exists():
            run(lana, 'test', cwd=clean)
    return dict(asset=asset, sha256=digest)


def api(path):
    request = urllib.request.Request('https://api.github.com/' + path,
        headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
                 'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def publish(directory):
    repository, tag, _, _, asset = identity()
    directory = Path(directory).resolve()
    if {path.name for path in directory.iterdir()} != {asset, 'SHA256SUMS'}:
        raise ValueError('release must contain exactly the archive and SHA256SUMS')
    digest = hashlib.sha256((directory / asset).read_bytes()).hexdigest()
    sums = (directory / 'SHA256SUMS').read_bytes()
    if sums != f'{digest}  {asset}\n'.encode():
        raise ValueError('release checksum mismatch')
    if api(f'repos/{repository}/commits/{tag}')['sha'] != os.environ['GITHUB_SHA']:
        raise ValueError('tag moved after qualification')
    try:
        release = api(f'repos/{repository}/releases/tags/{tag}')
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    else:
        if release['draft'] or release['prerelease'] or {item['name'] for item in release['assets']} != {asset, 'SHA256SUMS'}:
            raise ValueError('existing release differs; refusing overwrite')
        with tempfile.TemporaryDirectory(prefix='lana-release-compare-') as temporary:
            run('gh', 'release', 'download', tag, '--repo', repository, '--dir', temporary)
            previous = Path(temporary)
            if (previous / 'SHA256SUMS').read_bytes() != sums or hashlib.sha256((previous / asset).read_bytes()).hexdigest() != digest:
                raise ValueError('existing release assets differ; refusing overwrite')
        return {'status': 'unchanged', 'tag': tag}
    run('gh', 'release', 'create', tag, directory / asset, directory / 'SHA256SUMS',
        '--repo', repository, '--verify-tag', '--title', tag, '--notes', 'Lana source package.')
    return {'status': 'published', 'tag': tag}


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == 'prepare':
        print(json.dumps(prepare(*sys.argv[2:])))
    elif len(sys.argv) == 3 and sys.argv[1] == 'publish':
        print(json.dumps(publish(sys.argv[2])))
    else:
        sys.exit('usage: package_release.py prepare LANA DIRECTORY OUTPUT | publish OUTPUT')
