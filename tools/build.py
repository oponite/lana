#!/usr/bin/env python3
"""Cargo build, explicit installation, and universal macOS distribution."""
import argparse
import gzip
import hashlib
import tarfile
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "target/lana"


def cargo_build(target=None, profile="release"):
    command = ["cargo", "build", "--locked", "-p", "lana-cli", "--message-format=json"]
    if profile == "release":
        command.append("--release")
    if target:
        command.extend(["--target", target])
    env = dict(os.environ)
    if target:
        # Rustup owns the target libraries installed by the release prerequisites.
        command[0] = subprocess.check_output(["rustup", "which", "cargo"], cwd=ROOT, text=True).strip()
        env["RUSTC"] = subprocess.check_output(["rustup", "which", "rustc"], cwd=ROOT, text=True).strip()
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, text=True)
    executable = compiler = None
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-message":
            print(message["message"].get("rendered", message["message"]["message"]), file=sys.stderr, end="")
        if message.get("reason") == "compiler-artifact" and message.get("target", {}).get("name") == "lana":
            executable = Path(message["executable"])
        if message.get("reason") == "build-script-executed" and "lana-cli" in message.get("package_id", ""):
            compiler = Path(message["out_dir"]) / "lana-compiler.labc"
    if result.returncode:
        raise RuntimeError("Cargo build failed")
    if executable is None or compiler is None or not compiler.is_file():
        raise RuntimeError("Cargo did not report the CLI and compiler artifact")
    return executable, compiler


def install_files(executable, compiler, prefix, stdlib=ROOT / "stdlib", license_file=ROOT / "LICENSE"):
    prefix = Path(prefix).resolve()
    for source in [executable, compiler, license_file, *(Path(stdlib) / path.name for path in (ROOT / "stdlib").glob("*.lana"))]:
        if not Path(source).is_file():
            raise RuntimeError(f"missing installation asset: {source}")
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    # Replace complete files, preserving existing installations if copying fails.
    for source, name in [(executable, "lana"), (compiler, "lana-compiler.labc")]:
        with tempfile.NamedTemporaryFile(dir=prefix / "bin", delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            shutil.copy2(source, staged)
            staged.replace(prefix / "bin" / name)
        finally:
            staged.unlink(missing_ok=True)
    stdlib_dir = prefix / "share/lana/stdlib"
    stdlib_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=stdlib_dir.parent) as staged_dir:
        staged = Path(staged_dir) / "stdlib"
        shutil.copytree(stdlib, staged)
        backup = Path(staged_dir) / "previous"
        if stdlib_dir.exists():
            stdlib_dir.replace(backup)
        try:
            staged.replace(stdlib_dir)
        except OSError:
            if backup.exists():
                backup.replace(stdlib_dir)
            raise
    (prefix / "share/doc/lana").mkdir(parents=True, exist_ok=True)
    shutil.copy2(license_file, prefix / "share/doc/lana/LICENSE")
    return prefix


def build(output=DEFAULT, profile="release"):
    return install_files(*cargo_build(profile=profile), output)


def universal(output):
    if sys.platform != "darwin":
        raise RuntimeError("universal builds require macOS and both Apple Rust targets")
    arm, compiler = cargo_build("aarch64-apple-darwin")
    intel, other_compiler = cargo_build("x86_64-apple-darwin")
    if compiler.read_bytes() != other_compiler.read_bytes():
        raise RuntimeError("architecture compiler artifacts differ")
    with tempfile.TemporaryDirectory(prefix="lana-universal-") as work:
        binary = Path(work) / "lana"
        subprocess.run(["lipo", "-create", arm, intel, "-output", binary], check=True)
        subprocess.run(["lipo", binary, "-verify_arch", "arm64", "x86_64"], check=True)
        prefix = install_files(binary, compiler, output)
    # Require both slices to actually run, not just appear in a fat header.
    version = (ROOT / "VERSION").read_text().strip()
    with tempfile.TemporaryDirectory(prefix="lana-universal-check-") as work:
        example = Path(work) / "belief.lana"
        shutil.copy2(ROOT / "examples/basic-programs/belief.lana", example)
        env = dict(os.environ)
        for key in ["LANA_COMPILER_LABC", "LANA_STDLIB_DIR"]:
            env.pop(key, None)
        for arch in ["arm64", "x86_64"]:
            command = ["arch", "-" + arch, str(prefix / "bin/lana")]
            result = subprocess.run([*command, "version"], cwd=work, env=env, text=True, capture_output=True, check=True)
            if f"Lana {version} (LABC v2," not in result.stdout:
                raise RuntimeError(f"wrong {arch} version: {result.stdout}")
            subprocess.run([*command, "run", str(example)], cwd=work, env=env, check=True)
    return prefix


def package_prefix(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lana-package-", dir=output.parent) as directory:
        work = Path(directory)
        prefix = install_files(source / "bin/lana", source / "bin/lana-compiler.labc", work / "prefix",
                               source / "share/lana/stdlib", source / "share/doc/lana/LICENSE")
        version = (ROOT / "VERSION").read_text().strip()
        (prefix / "version.txt").write_text(version + "\n")
        # Qualify the staged assets without checkout configuration before replacing output.
        env = dict(os.environ)
        for key in ["LANA_COMPILER_LABC", "LANA_STDLIB_DIR"]:
            env.pop(key, None)
        result = subprocess.run([prefix / "bin/lana", "version"], cwd=work, env=env,
                                capture_output=True, text=True, check=True, timeout=60)
        if f"Lana {version} (LABC v2," not in result.stdout:
            raise RuntimeError("installed prefix version does not match VERSION")
        check = work / "check.lana"
        check.write_text('import "std/core" as core; assert(resolve(core.distribution([[7, 1]])) == 7, "packaged stdlib");\n')
        subprocess.run([prefix / "bin/lana", "run", check], cwd=work, env=env,
                       capture_output=True, text=True, check=True, timeout=60)
        staged = work / "archive.tar.gz"
        def normalize(info):
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            return info

        with staged.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(prefix.iterdir()):
                    archive.add(path, arcname=path.name, filter=normalize)
        with staged.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            os.fsync(stream.fileno())
        os.replace(staged, output)
        return {"archive": str(output), "sha256": digest, "version": version}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    native = commands.add_parser("build")
    native.add_argument("--output", type=Path, default=DEFAULT)
    native.add_argument("--profile", choices=["debug", "release"], default="release")
    install = commands.add_parser("install")
    install.add_argument("--prefix", type=Path, required=True)
    install.add_argument("--from", dest="source", type=Path, help="install an already built prefix")
    fat = commands.add_parser("universal")
    fat.add_argument("--output", type=Path, default=ROOT / "target/universal")
    package = commands.add_parser("package", help="validate and atomically archive an installed prefix")
    package.add_argument("--from", dest="source", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "package":
        print(json.dumps(package_prefix(args.source, args.output), sort_keys=True))
        return
    if args.command == "build":
        prefix = build(args.output, args.profile)
    elif args.command == "universal":
        prefix = universal(args.output)
    elif args.source:
        prefix = install_files(args.source / "bin/lana", args.source / "bin/lana-compiler.labc", args.prefix, args.source / "share/lana/stdlib", args.source / "share/doc/lana/LICENSE")
    else:
        prefix = install_files(*cargo_build(), args.prefix)
    print(prefix)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
