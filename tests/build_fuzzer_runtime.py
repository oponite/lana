#!/usr/bin/env python3
"""Build the pinned libFuzzer with deterministic RSS-thread shutdown on macOS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

VERSION = "22.1.8"
SHA256 = "922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888"
ARCHIVE = f"llvm-project-{VERSION}.src.tar.xz"
URL = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{VERSION}/{ARCHIVE}"
SOURCE = f"llvm-project-{VERSION}.src/compiler-rt/lib/fuzzer"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = Path(args.cc).resolve()
    version = subprocess.check_output([compiler, "--version"], text=True)
    if f"clang version {VERSION}" not in version:
        raise SystemExit(f"The qualified macOS fuzz runtime requires Clang {VERSION}")
    archive = output / ARCHIVE
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SHA256:
        raise SystemExit(f"LLVM source checksum mismatch: {digest}")
    # Extract only flat libFuzzer source files, never paths or links from the archive.
    source = output / "source"
    source.mkdir(exist_ok=True)
    with tarfile.open(archive) as package:
        for member in package:
            path = Path(member.name)
            if member.isfile() and str(path.parent) == SOURCE and path.suffix in (".cpp", ".h", ".def"):
                (source / path.name).write_bytes(package.extractfile(member).read())
    patch = Path(__file__).with_name("libfuzzer-rss-shutdown.patch")
    subprocess.run(["patch", "--batch", "-p1", "-i", patch.resolve()], cwd=source, check=True)
    objects = []

    def compile_file(path):
        target = output / (path.stem + ".o")
        subprocess.run([compiler, "-g", "-O2", "-fno-omit-frame-pointer", "-std=c++17",
                        "-c", path, "-o", target], check=True)
        return target

    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 8)) as pool:
        objects = list(pool.map(compile_file, sorted(source.glob("*.cpp"))))
    library = output / "libFuzzer.a"
    subprocess.run([compiler.with_name("llvm-ar"), "rcs", library, *objects], check=True)
    probe = output / "runtime-probe"
    subprocess.run([compiler, "-fsanitize=address,fuzzer-no-link",
                    Path(__file__).with_name("fuzzer_runtime_probe.c").resolve(),
                    library, "-lc++", "-o", probe], check=True)
    controls = {}
    for mode, marker in (("noop-1", None), ("noop-2", None), ("noop-3", None),
                         ("leak", "LeakSanitizer"), ("crash", "deadly signal"),
                         ("timeout", "timeout"), ("rss", "out-of-memory")):
        env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=1"}
        env.pop("LANA_FUZZ_PROBE", None)
        if marker:
            env["LANA_FUZZ_PROBE"] = mode
        flags = ["-runs=2", "-timeout=1"]
        if mode == "rss":
            flags = ["-runs=2", "-timeout=10", "-rss_limit_mb=128"]
        log = output / f"control-{mode}.log"
        with log.open("w") as stream:
            result = subprocess.run([probe, *flags], cwd=output, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=15)
        controls[mode] = result.returncode
        if (marker is None and result.returncode != 0) or (
                marker is not None and (result.returncode == 0 or marker not in log.read_text())):
            raise SystemExit(f"libFuzzer control failed: {mode}; see {log}")
    (output / "provenance.json").write_text(json.dumps({
        "source_url": URL, "source_sha256": SHA256, "compiler": str(compiler),
        "compiler_version": version, "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "controls": controls,
    }, indent=2) + "\n")
    print(library)


if __name__ == "__main__":
    main()
