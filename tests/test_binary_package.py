"""An installed archive works independently; missing assets preserve old output."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build import install_files, EXECUTABLE

cli = Path(sys.argv[1]).resolve()
compiler = Path(sys.argv[2]).resolve()
with tempfile.TemporaryDirectory(prefix="lana-binary-package-") as directory:
    work = Path(directory)
    prefix = install_files(cli, compiler, work / "prefix")
    stale = prefix / "share/lana/stdlib/obsolete.lana"
    stale.write_text("obsolete", encoding="utf-8")
    install_files(cli, compiler, prefix)
    assert not stale.exists(), "reinstall retained a removed standard-library module"
    output = work / "lana.tar.gz"
    def package(destination, check=True):
        return subprocess.run([sys.executable, ROOT / "tools/build.py", "package", "--from", prefix, "--output", destination],
                              text=True, capture_output=True, check=check)
    result = package(output)
    report = json.loads(result.stdout)
    assert report["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    old = output.read_bytes()
    again = work / "again.tar.gz"
    package(again)
    assert again.read_bytes() == old, "same installed prefix produced different archive bytes"
    with tarfile.open(output) as archive:
        archive.extractall(work / "unpacked", filter="data")
    installed = work / "unpacked"
    assert (installed / "share/doc/lana/LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    check = work / "check.lana"
    check.write_text('import "std/core" as core; print(resolve(core.distribution([[42, 1]])));\n')
    env = dict(os.environ)
    for key in ["LANA_COMPILER_LABC", "LANA_STDLIB_DIR"]:
        env.pop(key, None)
    result = subprocess.run([installed / "bin" / EXECUTABLE, "run", check], cwd=work, env=env, text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "42", result.stdout
    for relative in ["bin/lana-compiler.labc", "share/lana/stdlib/core.lana", "share/doc/lana/LICENSE"]:
        asset = prefix / relative
        saved = asset.read_bytes()
        asset.unlink()
        failed = package(output, check=False)
        assert failed.returncode != 0 and "missing installation asset" in failed.stderr, failed
        assert output.read_bytes() == old
        asset.write_bytes(saved)
    zipped = work / "lana.zip"
    package(zipped)
    zip_bytes = zipped.read_bytes()
    package(zipped)
    assert zipped.read_bytes() == zip_bytes, "ZIP output is not reproducible"
    with zipfile.ZipFile(zipped) as archive:
        assert f"bin/{EXECUTABLE}" in archive.namelist()
        archive.extractall(work / "unzipped")
    result = subprocess.run([work / "unzipped/bin" / EXECUTABLE, "version"], text=True, capture_output=True, check=True) if os.name == "nt" else None
    if result:
        assert (ROOT / "VERSION").read_text().strip() in result.stdout
    assert not list(work.glob(".lana-package-*"))
print("BINARY_PACKAGE_PASS")
