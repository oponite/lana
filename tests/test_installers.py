"""Exercise the release scripts with real bundles and a fixture download transport."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build import EXECUTABLE, install_files, package_prefix, render_installers

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('cli', type=Path)
parser.add_argument('compiler', type=Path)
parser.add_argument('--archive', type=Path)
args = parser.parse_args()
version = (ROOT / 'VERSION').read_text().strip()
windows = os.name == 'nt'
architecture = 'arm64' if platform.machine().lower() in ('arm64', 'aarch64') else 'x64'
system = platform.system()
release_platform = f'linux-{architecture}' if system == 'Linux' else 'macos-universal'
asset = f'lana-{version}-windows-{architecture}.zip' if windows else f'lana-{version}-{release_platform}.tar.gz'
base = f'https://github.com/oponite/lana/releases/download/v{version}'


with tempfile.TemporaryDirectory(prefix='lana-installer-tests-') as directory:
    work = Path(directory)
    assets = work / 'assets'
    scripts = render_installers(work / 'scripts')
    assets.mkdir()
    if args.archive:
        shutil.copyfile(args.archive, assets / asset)
    else:
        prefix = install_files(args.cli.resolve(), args.compiler.resolve(), work / 'prefix')
        package_prefix(prefix, assets / asset)
    good = (assets / asset).read_bytes()
    script = (scripts / ('install.ps1' if windows else 'install.sh')).read_text()
    assert '@LANA_VERSION@' not in script
    mock_bin = work / 'mock-bin'
    mock_bin.mkdir()
    if not windows:
        for name, body in {
            'uname': 'import os, sys\nprint(os.environ["LANA_TEST_SYSTEM"] if sys.argv[1] == "-s" else os.environ["LANA_TEST_MACHINE"])\n',
            'curl': 'import os, pathlib, shutil, sys\na=sys.argv[1:]\nurl=a[-3]\n'
                    'assert url.startswith(os.environ["LANA_TEST_BASE"] + "/"), url\n'
                    'shutil.copyfile(pathlib.Path(os.environ["LANA_TEST_ASSETS"]) / url.rsplit("/",1)[1], a[-1])\n',
        }.items():
            path = mock_bin / name
            path.write_text(f'#!{sys.executable}\n' + body)
            path.chmod(0o755)
    else:
        transport = r'''
function Invoke-WebRequest {
    param([switch]$UseBasicParsing, $Uri, $OutFile, $TimeoutSec)
    if (-not $Uri.StartsWith($env:LANA_TEST_BASE + '/')) { throw "Unexpected URL: $Uri" }
    Copy-Item -LiteralPath (Join-Path $env:LANA_TEST_ASSETS ($Uri.Split('/')[-1])) -Destination $OutFile
}
$registry = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment')
$existed = @($registry.GetValueNames() | Where-Object { $_ -ieq 'Path' }).Count -ne 0
$original = $registry.GetValue('Path', $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
$kind = if ($existed) { $registry.GetValueKind('Path') } else { [Microsoft.Win32.RegistryValueKind]::String }
$held = $null
try {
    $registry.SetValue('Path', "%LANA_TEST_TOKEN%\bin;$original", [Microsoft.Win32.RegistryValueKind]::ExpandString)
    if ($env:LANA_TEST_LOCK_TARGET) { $held = [IO.File]::Open($env:LANA_TEST_LOCK_TARGET, 'Open', 'Read', 'None') }
    & ([ScriptBlock]::Create([IO.File]::ReadAllText($env:LANA_TEST_SCRIPT)))
    $raw = $registry.GetValue('Path', $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    if (-not $raw.Contains('%LANA_TEST_TOKEN%\bin;') -or
        $registry.GetValueKind('Path') -ne [Microsoft.Win32.RegistryValueKind]::ExpandString) { throw 'User PATH references/type were changed' }
    if ($env:LANA_TEST_REPEAT) {
        $before = [Environment]::GetEnvironmentVariable('Path', 'User')
        & ([ScriptBlock]::Create([IO.File]::ReadAllText($env:LANA_TEST_SCRIPT)))
        if ([Environment]::GetEnvironmentVariable('Path', 'User') -cne $before) { throw 'Reinstall changed user PATH' }
    }
    [IO.File]::WriteAllText($env:LANA_TEST_REPORT, ([Environment]::GetEnvironmentVariable('Path', 'User')))
} finally {
    if ($held) { $held.Dispose() }
    if ($existed) { $registry.SetValue('Path', $original, $kind) } else { $registry.DeleteValue('Path', $false) }
    $registry.Dispose()
    if ('Oponite.LanaInstaller.Notification' -as [type]) {
        $notification = [UIntPtr]::Zero
        [Oponite.LanaInstaller.Notification]::SendMessageTimeout([IntPtr]0xffff, 0x1a, [UIntPtr]::Zero,
            'Environment', 2, 5000, [ref]$notification) | Out-Null
    }
}
'''

    def manifest(data=good):
        (assets / asset).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        (assets / 'SHA256SUMS').write_text(f'{digest}  {asset}\n' + '0' * 64 + '  another-platform.zip\n')

    def destination(home):
        return home / ('Programs/Lana' if windows else '.local/share/lana')

    def run(home, success=True, shell='/bin/zsh', engine=None, extra_env=None):
        home.mkdir(exist_ok=True)
        env = dict(os.environ, HOME=str(home), LOCALAPPDATA=str(home), SHELL=shell,
                   PATH=str(mock_bin) + os.pathsep + os.environ['PATH'],
                   LANA_TEST_ASSETS=str(assets), LANA_TEST_BASE=base,
                   LANA_TEST_SYSTEM=system, LANA_TEST_MACHINE=platform.machine(),
                   LANA_TEST_SCRIPT=str(scripts / 'install.ps1'), LANA_TEST_REPORT=str(work / 'path-report'))
        # A working checkout's compiler paths must not mask missing bundled files.
        env['LANA_COMPILER_LABC'] = str(work / 'does-not-exist')
        env['LANA_STDLIB_DIR'] = str(work / 'does-not-exist')
        env.update(extra_env or {})
        if windows:
            result = subprocess.run([engine or 'powershell.exe', '-NoProfile', '-Command', transport],
                                    env=env, cwd=work, text=True, capture_output=True, timeout=180)
        else:
            # Feed stdin exactly as curl | sh does.
            result = subprocess.run(['sh'], input=script, env=env, cwd=work, text=True, capture_output=True, timeout=180)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return result

    def tree(path):
        return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob('*') if p.is_file()}

    def changed_archive(remove=None, extra=None, replacement=None, link=False):
        output = io.BytesIO()
        if windows:
            with zipfile.ZipFile(io.BytesIO(good)) as source, zipfile.ZipFile(output, 'w') as target:
                for info in source.infolist():
                    if info.filename != remove:
                        target.writestr(info, replacement if info.filename == 'version.txt' and replacement else source.read(info))
                if extra:
                    target.writestr(extra, b'escape')
                if link:
                    info = zipfile.ZipInfo('bin/linked')
                    info.external_attr = 0o120777 << 16
                    target.writestr(info, '../../escaped')
        else:
            with tarfile.open(fileobj=io.BytesIO(good)) as source, tarfile.open(fileobj=output, mode='w:gz') as target:
                for info in source:
                    if info.name != remove:
                        if info.name == 'version.txt' and replacement:
                            info.size = len(replacement)
                            target.addfile(info, io.BytesIO(replacement))
                        else:
                            target.addfile(info, source.extractfile(info) if info.isfile() else None)
                if extra:
                    info = tarfile.TarInfo(extra); info.size = 6
                    target.addfile(info, io.BytesIO(b'escape'))
                if link:
                    info = tarfile.TarInfo('bin/linked'); info.type = tarfile.SYMTYPE
                    info.linkname = '../../escaped'
                    target.addfile(info)
        return output.getvalue()

    manifest()
    home = work / 'home spaces é'
    home.mkdir()
    if not windows:
        profile = home / '.zshrc'
        profile.write_text('# existing user settings\n')
        profile.chmod(0o600)
    run(home, extra_env={'LANA_TEST_REPEAT': '1'} if windows else None)
    installed = destination(home)
    baseline = tree(installed)
    assert (installed / '.lana-install').read_text().strip() == 'lana-installer-v1'
    if windows:
        assert str(installed / 'bin').lower() in (work / 'path-report').read_text(encoding='utf-8').lower()
    else:
        assert profile.stat().st_mode & 0o777 == 0o600
        assert profile.read_text().startswith('# existing user settings\n')
        content = profile.read_bytes()
        result = subprocess.run(['zsh', '-ic', 'command -v lana'], env=dict(os.environ, HOME=str(home)), text=True,
                                capture_output=True, timeout=30) if shutil.which('zsh') else None
        if result:
            assert result.returncode == 0 and str(installed / 'bin/lana') in result.stdout, result
    (installed / 'share/lana/stdlib/obsolete.lana').write_text('old module')
    (installed / 'version.txt').write_text('0.0.1\n')
    run(home)
    assert tree(installed) == baseline, 'upgrade retained stale files or changed the bundle'
    run(home)
    if not windows:
        assert profile.read_bytes() == content, 'reinstall duplicated shell configuration'
        bash_home = work / 'bash-home'
        run(bash_home, shell='/bin/bash')
        for name in ('.bashrc', '.bash_profile'):
            configured = subprocess.run(['bash', '-c', f'. "$HOME/{name}"; command -v lana'],
                                        env=dict(os.environ, HOME=str(bash_home)), text=True, capture_output=True, check=True)
            assert str(destination(bash_home) / 'bin/lana') in configured.stdout
        # Fail after promotion; both the installed bundle and earlier profile edits roll back.
        (bash_home / '.bashrc').unlink()
        (bash_home / '.bashrc').mkdir()
        (bash_home / '.bash_profile').write_text('# restore this user configuration\n')
        old_profile = (bash_home / '.bash_profile').read_bytes()
        (destination(bash_home) / 'keep-me').write_text('previous installation')
        before = tree(destination(bash_home))
        run(bash_home, success=False, shell='/bin/bash')
        assert tree(destination(bash_home)) == before
        assert (bash_home / '.bash_profile').read_bytes() == old_profile
        # A truncated script must never start installation.
        truncated_home = work / 'truncated'
        subprocess.run(['sh'], input=script[:script.index('main "$@"')], text=True,
                       env=dict(os.environ, HOME=str(truncated_home)), capture_output=True, check=True)
        assert not truncated_home.exists()
        linked_home = work / 'linked-profile'
        linked_home.mkdir()
        target = linked_home / 'dotfiles-zshrc'
        target.write_text('# managed by the user\n')
        (linked_home / '.zshrc').symlink_to(target.name)
        run(linked_home)
        assert (linked_home / '.zshrc').is_symlink()
        assert target.read_text().startswith('# managed by the user\n')

    for data in (b'corrupt download', changed_archive(remove='bin/lana-compiler.labc'),
                 changed_archive(remove='share/lana/stdlib/core.lana'), changed_archive(extra='../escaped'),
                 changed_archive(remove='bin/' + EXECUTABLE), changed_archive(remove='share/doc/lana/LICENSE'),
                 changed_archive(replacement=b'0.0.0\n'), changed_archive(link=True)):
        manifest(data)
        run(home, success=False)
        assert tree(installed) == baseline
        assert not (installed.parent / 'escaped').exists()
    manifest()
    (assets / asset).write_bytes(b'checksum mismatch')
    result = run(home, success=False)
    assert 'checksum' in (result.stdout + result.stderr).lower()
    assert tree(installed) == baseline
    manifest()
    (assets / 'SHA256SUMS').write_text('0' * 64 + '  different-name\n')
    run(home, success=False)
    manifest()
    with (assets / 'SHA256SUMS').open('a') as stream:
        stream.write(f'not-a-digest  {asset}\n')
    run(home, success=False)
    manifest()
    (assets / 'SHA256SUMS').unlink()
    run(home, success=False)
    manifest()
    unrelated = work / 'unrelated'
    dest = destination(unrelated)
    dest.mkdir(parents=True)
    (dest / 'personal-file').write_text('keep')
    run(unrelated, success=False)
    assert tree(dest) == {'personal-file': b'keep'}
    lock = installed.parent / '.lana-install.lock'
    lock.write_text('busy') if windows else lock.mkdir()
    run(home, success=False)
    lock.unlink() if windows else lock.rmdir()
    assert tree(installed) == baseline
    if windows:
        result = run(home, success=False, extra_env={'LANA_TEST_LOCK_TARGET': str(installed / 'bin/lana.exe')})
        assert 'Close programs' in result.stderr, result
        assert tree(installed) == baseline
        for engine in ('powershell.exe', 'pwsh.exe'):
            assert shutil.which(engine), f'required Windows shell missing: {engine}'
            run(home, engine=engine, extra_env={'LANA_TEST_REPEAT': '1'})
        assert not list(installed.parent.glob('.lana-install-*'))
    else:
        for test_system, machine, bundle in (
            ('Darwin', 'arm64', 'macos-universal'), ('Darwin', 'x86_64', 'macos-universal'),
            ('Linux', 'x86_64', 'linux-x64'), ('Linux', 'aarch64', 'linux-arm64'),
        ):
            name = f'lana-{version}-{bundle}.tar.gz'
            (assets / name).write_bytes(good)
            (assets / 'SHA256SUMS').write_text(f'{hashlib.sha256(good).hexdigest()}  {name}\n')
            run(home, extra_env={'LANA_TEST_SYSTEM': test_system, 'LANA_TEST_MACHINE': machine})
        manifest()
        run(home, success=False, extra_env={'LANA_TEST_MACHINE': 'riscv64'})
        run(home, success=False, extra_env={'LANA_TEST_SYSTEM': 'FreeBSD'})
        assert not list(installed.parent.glob('.lana-install.*'))
print('INSTALLERS_PASS')
