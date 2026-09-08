#!/usr/bin/env python3
"""Build and test one candidate tree, with logs and a machine-readable report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
GATES = {"native", "rust", "rust-no-tls", "bootstrap-rust", "differential", "net-live", "store-process", "oracles-rust", "bytecode", "asan", "tsan",
         "fuzz", "hardware", "universal", "integrations", "python-integrations"}


def git(*arguments):
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments])


def tree_fingerprint():
    digest = hashlib.sha256()
    for name in sorted(set(git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")) - {b""}):
        path = ROOT / os.fsdecode(name)
        digest.update(name + b"\0")
        mode = path.lstat().st_mode if path.exists() or path.is_symlink() else 0
        digest.update(mode.to_bytes(4, "little"))
        if path.is_symlink():
            data = os.fsencode(os.readlink(path))
        else:
            data = path.read_bytes() if path.is_file() else b"<deleted>"
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def junit_results(path, expected):
    cases = ET.parse(path).getroot().findall(".//testcase")
    names = [case.attrib["name"] for case in cases]
    if not names or len(names) != len(set(names)) or set(names) != set(expected):
        raise RuntimeError(f"JUnit inventory differs from CTest: {set(expected) - set(names)} missing")
    statuses = {case.attrib["name"]: "skipped" if case.find("skipped") is not None else
                "failed" if case.find("failure") is not None or case.find("error") is not None else
                "passed" for case in cases}
    return statuses


def cargo_counts(output):
    results = re.findall(r"test result: ok\. ([0-9]+) passed; ([0-9]+) failed; ([0-9]+) ignored;", output)
    counts = dict(zip(("passed", "failed", "ignored"), map(sum, zip(*(map(int, row) for row in results)))))
    if not counts.get("passed") or counts["failed"] or counts["ignored"]:
        raise RuntimeError(f"Rust test evidence is incomplete: {counts}")
    return counts


def validate_registry(manifest, inventory):
    ids = set()
    for claim in manifest["claims"]:
        if claim["id"] in ids or not claim["requires"] or not set(claim["requires"]) <= GATES:
            raise RuntimeError(f"invalid claim gate: {claim['id']}")
        ids.add(claim["id"])
        for name in claim.get("ctest", []):
            if name not in inventory:
                raise RuntimeError(f"claim {claim['id']} names an absent test: {name}")
        for name in claim["authority"] + claim["evidence"]:
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise RuntimeError(f"claim {claim['id']} names an absent file: {name}")


def check_registration():
    registrations = [ROOT / "CMakeLists.txt", Path(__file__), *ROOT.glob("cmake/*.cmake"),
                     *ROOT.glob(".github/workflows/*.yml")]
    text = "\n".join(path.read_text() for path in registrations)
    scripts = list((ROOT / "tests").glob("test_*.py"))
    missing = [str(path.relative_to(ROOT)) for path in scripts if path.name not in text]
    fixture_text = text + "\n" + "\n".join(path.read_text() for path in scripts)
    sources = list((ROOT / "tests/regression").rglob("*.lana"))
    referenced = {path.resolve() for path in sources if path.name in fixture_text}
    pending = list(referenced)
    while pending:
        source = pending.pop()
        for name in re.findall(r'^import\s+"([^"\n]+)"', source.read_text(), re.M):
            child = (source.parent / name).resolve()
            if child.is_relative_to(ROOT) and child.is_file() and child not in referenced:
                referenced.add(child)
                pending.append(child)
    missing += [str(path.relative_to(ROOT)) for path in sources if path.resolve() not in referenced]
    for contract in (ROOT / "tests/conformance").rglob("*.expect.json"):
        stem = contract.name.removesuffix(".expect.json")
        if not any(contract.with_name(stem + suffix).is_file() for suffix in (".lasm", ".lana")):
            missing.append(str(contract.relative_to(ROOT)))
    if missing:
        raise RuntimeError("unregistered tests or orphan contracts: " + ", ".join(sorted(missing)))


def stop_process(process):
    # Each command owns a process group. Stop compiler/server children as well.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("quick", "full", "asan", "tsan", "integrations", "nightly", "hardware", "release"), nargs="?", default="quick")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--cc", help="C compiler for the native build")
    parser.add_argument("--fuzz-cc", help="Clang with the libFuzzer runtime")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Python assertions must be enabled; remove -O and PYTHONOPTIMIZE")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.profile in ("hardware", "release") and sys.platform != "darwin":
        parser.error("hardware/release requires macOS for Metal and both installed architecture slices")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.report_dir or ROOT / "build/test-evidence" / f"{args.profile}-{stamp}").resolve()
    if output.is_relative_to(ROOT) and subprocess.run(
        ["git", "check-ignore", "-q", str(output)], cwd=ROOT).returncode != 0:
        parser.error("--report-dir must be outside the checkout or ignored by Git")
    output.mkdir(parents=True, exist_ok=False)
    build = args.build_dir.resolve()
    report = dict(schema=1, profile=args.profile, started_at=stamp, commit=git("rev-parse", "HEAD").decode().strip(),
                  tree_sha256=tree_fingerprint(), platform=platform.platform(), python=sys.version,
                  status="running", steps=[])
    started = time.monotonic()
    manifest = json.loads((ROOT / "tests/claims.json").read_text())

    def command(name, arguments, timeout=900, env=None, cwd=ROOT):
        arguments = list(map(str, arguments))
        log = output / f"{len(report['steps']):02d}-{name}.log"
        step = dict(name=name, command=arguments, cwd=str(cwd), log=str(log), status="running")
        report["steps"].append(step)
        print(f"RUN  {name}", flush=True)
        begin = time.monotonic()
        try:
            with log.open("wb") as stream:
                process = subprocess.Popen(arguments, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                           env={**os.environ, **(env or {})}, start_new_session=True)
                try:
                    step["exit"] = process.wait(timeout=timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    stop_process(process)
                    raise
            if step["exit"] != 0:
                raise RuntimeError(f"{name} exited {step['exit']}: {log}\n{log.read_text(errors='replace')[-3000:]}")
            step["status"] = "passed"
        except BaseException:
            step["status"] = "failed"
            raise
        finally:
            step["seconds"] = round(time.monotonic() - begin, 3)
        print(f"PASS {name} ({step['seconds']}s)", flush=True)
        return step

    def configure(directory, *options):
        command("configure-" + directory.name, ["cmake", "-S", ROOT, "-B", directory,
                "-DCMAKE_BUILD_TYPE=Debug", "-DBUILD_TESTING=ON", "-DLANA_ENABLE_SANITIZERS=OFF", "-DLANA_ENABLE_TSAN=OFF",
                "-DLANA_BUILD_FUZZERS=OFF", "-DLANA_BUILD_INTEGRATIONS=OFF", *options])
        command("build-" + directory.name, ["cmake", "--build", directory, "--parallel", args.jobs])

    def ctest(name, directory, *selection):
        discovery = command("inventory-" + name, ["ctest", "--test-dir", directory, "--show-only=json-v1", *selection])
        inventory = json.loads(Path(discovery["log"]).read_text())["tests"]
        expected = {test["name"] for test in inventory}
        junit = output / f"{name}.xml"
        step = command(name, ["ctest", "--test-dir", directory, "--output-on-failure", "--no-tests=error",
                              "--timeout", "300", "--output-junit", junit, *selection], timeout=1800)
        try:
            step["tests"] = junit_results(junit, expected)
            incomplete = [name for name, status in step["tests"].items() if status != "passed"]
            if incomplete:
                raise RuntimeError(f"{name}: tests did not pass: {', '.join(incomplete)}")
        except BaseException:
            step["status"] = "failed"
            raise
        return expected

    try:
        check_registration()
        command("tools", ["cmake", "--version"])
        command("rust-toolchain", ["cargo", "--version"])
        options = ["-DCMAKE_C_COMPILER=" + args.cc] if args.cc else []
        if args.profile in ("asan", "tsan"):
            options.append("-D" + ("LANA_ENABLE_SANITIZERS" if args.profile == "asan" else "LANA_ENABLE_TSAN") + "=ON")
        if args.profile == "integrations":
            options += ["-DCMAKE_BUILD_TYPE=Release", "-DLANA_BUILD_INTEGRATIONS=ON"]
        configure(build, *options)
        all_tests = command("inventory-all", ["ctest", "--test-dir", build, "--show-only=json-v1"])
        validate_registry(manifest, {t["name"] for t in json.loads(Path(all_tests["log"]).read_text())["tests"]})
        if args.profile == "hardware":
            ctest("hardware", build, "-L", "hardware")
        elif args.profile in ("asan", "tsan", "integrations"):
            ctest(args.profile, build)
        else:
            ctest("native", build, *(["-LE", "optional|hardware"] if args.profile == "quick" else []))
            rust_step = command("rust", ["cargo", "test", "--workspace", "--locked"],
                                env={"LANA_COMPILER_LABC": str(build / "lana-compiler.labc"), "CARGO_TERM_COLOR": "never"})
            try:
                rust_step["counts"] = cargo_counts(Path(rust_step["log"]).read_text())
            except RuntimeError:
                rust_step["status"] = "failed"
                raise
            no_tls = command("rust-no-tls", ["cargo", "test", "--locked", "-p", "lana-vm",
                "--no-default-features", "--lib", "https_without_tls_never_falls_back_to_plaintext"])
            try:
                no_tls["counts"] = cargo_counts(Path(no_tls["log"]).read_text())
            except RuntimeError:
                no_tls["status"] = "failed"
                raise
            command("build-rust-cli", ["cargo", "build", "--locked", "-p", "lana-cli"])
            command("build-store-probe", ["cargo", "build", "--locked", "-p", "lana-runtime", "--example", "lana_store_probe"])
            command("store-process", [sys.executable, ROOT / "tests/test_store_process.py",
                "--c11", build / "lana_store_tests", "--rust", ROOT / "target/debug/examples/lana_store_probe"])
            rust = ROOT / "target/debug/lana-cli"
            if args.profile != "quick":
                command("bootstrap-rust", ["cmake", "-DLANA_VM=" + str(rust),
                    "-DLANA_COMPILER=" + str(build / "lana-compiler.labc"),
                    "-DLANA_BUNDLE=" + str(build / "compiler-bootstrap.lana"),
                    "-DLANA_REFERENCE=" + str(ROOT / "compiler/bootstrap/compiler.lasm"),
                    "-DLANA_OUTPUT=" + str(output / "rust-bootstrap.lasm"),
                    "-P", ROOT / "cmake/VerifyNativeBootstrap.cmake"], timeout=300)
            version = (ROOT / "VERSION").read_text().strip()
            for name, binary in (("c11", build / "lana"), ("rust", rust)):
                step = command("version-" + name, [binary, "version"])
                if not Path(step["log"]).read_text().startswith(f"Lana {version} (LABC v2,"):
                    step["status"] = "failed"
                    raise RuntimeError(f"{name} reports the wrong version")
            command("differential", [sys.executable, ROOT / "tests/conformance/differential/run.py",
                "core", "hostcalls", "tasks", "durable", "ffi", "net", "--c11", build / "lanavm",
                "--rust", rust, "--compiler", build / "lana-compiler.labc"])
            command("oracles-rust", [sys.executable, ROOT / "tests/test_oracles.py", "--lana", build / "lana", "--rust", rust])
            command("net-live", [sys.executable, ROOT / "tests/test_net_live.py",
                "--lana", build / "lana", "--c11", build / "lanavm", "--rust", rust])
            command("bytecode", [sys.executable, ROOT / "tests/test_bytecode.py", "--c11", build / "lanavm",
                "--rust", rust, "--cases", 256 if args.profile == "quick" else 1024,
                "--artifacts", output / "bytecode-failures"])
        if args.profile in ("nightly", "release"):
            for name, option in (("asan", "LANA_ENABLE_SANITIZERS"), ("tsan", "LANA_ENABLE_TSAN")):
                directory = build.with_name(build.name + "-" + name)
                configure(directory, "-D" + option + "=ON")
                ctest(name, directory)
            compiler = args.fuzz_cc or ("/opt/homebrew/opt/llvm/bin/clang" if sys.platform == "darwin" else "clang")
            compiler = shutil.which(compiler)
            if not compiler or not Path(compiler).is_file():
                raise RuntimeError("fuzzing requires Clang with libFuzzer; set --fuzz-cc")
            fuzz = build.with_name(build.name + "-fuzz")
            command("configure-fuzz", ["cmake", "-S", ROOT, "-B", fuzz, "-DCMAKE_BUILD_TYPE=Debug",
                "-DCMAKE_C_COMPILER=" + compiler, "-DLANA_ENABLE_SANITIZERS=ON", "-DLANA_ENABLE_TSAN=OFF", "-DLANA_BUILD_FUZZERS=ON"])
            command("build-fuzz", ["cmake", "--build", fuzz, "--target", "lana_bytecode_fuzz", "--parallel", args.jobs])
            corpus = output / "fuzz-corpus"
            corpus.mkdir()
            command("fuzz-seed", [build / "lanavm", "asm", ROOT / "tests/conformance/fuzz/state.lasm", "-o", corpus / "state.labc"])
            command("fuzz", [fuzz / "lana_bytecode_fuzz", "-seed=42", "-max_total_time=600", "-timeout=5",
                             "-artifact_prefix=" + str(corpus) + "/", corpus], timeout=660)
        if args.profile in ("hardware", "release"):
            universal = build.with_name(build.name + "-universal")
            configure(universal, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64")
            prefix = output / "install"
            command("install-universal", ["cmake", "--install", universal, "--prefix", prefix])
            command("universal", [sys.executable, ROOT / "tests/verify_install.py", "--prefix", prefix,
                                  "--architecture", "arm64", "--architecture", "x86_64"])
        if args.profile in ("release", "integrations"):
            integrations = build
            if args.profile == "release":
                integrations = build.with_name(build.name + "-integrations")
                configure(integrations, "-DCMAKE_BUILD_TYPE=Release", "-DLANA_BUILD_INTEGRATIONS=ON")
                ctest("integrations", integrations)
            venv = output / "python-venv"
            command("python-venv", [sys.executable, "-m", "venv", venv])
            command("python-dependencies", [venv / "bin/python", "-m", "pip", "install", "-e", str(ROOT / "integrations/python") + "[test]"])
            command("python-integrations", [venv / "bin/python", "-m", "pytest", "-q", ROOT / "integrations/python/tests"],
                    env={"LANA_BIN": str(integrations / "lana")})
        command("whitespace", ["git", "diff", "--check"])
        report["status"] = "passed"
    except (RuntimeError, OSError, ValueError, ET.ParseError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = str(error) or "interrupted"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report["tree_sha256_after"] = tree_fingerprint()
        if report["tree_sha256_after"] != report["tree_sha256"]:
            report["status"] = "failed"
            report["error"] = "candidate tree changed during the run"
        passed = {step["name"] for step in report["steps"] if step["status"] == "passed"}
        passed_tests = {name for step in report["steps"] for name, status in step.get("tests", {}).items()
                        if step["status"] == "passed" and status == "passed"}
        report["claims"] = {claim["id"]: "checked" if set(claim["requires"]) <= passed
                            and set(claim.get("ctest", [])) <= passed_tests else "not_checked"
                            for claim in manifest["claims"]}
        if report["tree_sha256_after"] != report["tree_sha256"]:
            report["claims"] = {claim["id"]: "invalidated" for claim in manifest["claims"]}
        report["seconds"] = round(time.monotonic() - started, 3)
        path = output / "report.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{report['status'].upper()}: {path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
