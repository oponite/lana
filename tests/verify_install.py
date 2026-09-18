"""Check an installed prefix without compiler or stdlib paths from the checkout."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--architecture", action="append", default=[])
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    version = (ROOT / "VERSION").read_text().strip()
    for name in ("bin/lana", "bin/lanavm", "bin/lana-compiler.labc", "lib/liblanaruntime.a"):
        assert (prefix / name).is_file(), f"missing installed file: {name}"
    env = {key: value for key, value in os.environ.items() if not key.startswith("LANA_")}
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    with tempfile.TemporaryDirectory(prefix="lana-clean-install-") as directory:
        directory = Path(directory)
        source = directory / "belief.lana"
        shutil.copyfile(ROOT / "examples/belief.lana", source)
        for binary in ("lana", "lanavm"):
            if args.architecture:
                subprocess.run(["lipo", str(prefix / "bin" / binary), "-verify_arch", *args.architecture],
                               check=True, capture_output=True, timeout=30)
            for architecture in args.architecture or [None]:
                command = (["arch", "-" + architecture] if architecture else []) + [str(prefix / "bin" / binary)]
                result = subprocess.run([*command, "version"], cwd=directory, env=env,
                                        capture_output=True, text=True, timeout=30)
                assert result.returncode == 0 and result.stderr == "", result
                labc = "v1-v5" if binary == "lana" else "v1-v4"
                assert result.stdout.startswith(f"Lana {version} (LABC {labc},") and len(result.stdout.splitlines()) == 1, result.stdout
                if binary == "lana":
                    result = subprocess.run([*command, "run", str(source)], cwd=directory, env=env,
                                            capture_output=True, text=True, timeout=30)
                    assert result.returncode == 0 and result.stderr == "", result
                    assert result.stdout == "0.05\n", result.stdout
        core_source = directory / "core-v5.lana"
        core_bytecode = directory / "core-v5.labc"
        core_source.write_text(
            'import "std/core" as info;\n'
            'let weighted = info.distribution([["confirmed", 1.0]]);\n'
            'assert(sample_value(sample(weighted)) == "confirmed", "Core distribution");\n',
            encoding="utf-8")
        command = [str(prefix / "bin" / "lana")]
        result = subprocess.run([*command, "compile", str(core_source), "-o", str(core_bytecode)],
                                cwd=directory, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 and result.stderr == "", result
        assert core_bytecode.read_bytes()[4:8] == (5).to_bytes(4, "little"), "Core source did not emit LABC v5"
        result = subprocess.run([str(prefix / "bin" / "lanavm"), "run", str(core_bytecode)],
                                cwd=directory, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert "unsupported LABC v5: lanavm supports v1-v4" in result.stderr, result.stderr
    print("CLEAN_INSTALL_PASS")


if __name__ == "__main__":
    main()
