"""Check an installed prefix without compiler or stdlib paths from the checkout."""
import argparse
import ctypes
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--architecture", action="append", default=[])
    parser.add_argument("--libdir", default="lib")
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    version = (ROOT / "VERSION").read_text().strip()
    for name in ("bin/lana", "bin/lanavm", "bin/lana-compiler.labc", f"{args.libdir}/liblanaruntime.a"):
        assert (prefix / name).is_file(), f"missing installed file: {name}"
    library_name = "liblana_ffi.dylib" if sys.platform == "darwin" else "liblana_ffi.so"
    library = prefix / args.libdir / library_name
    assert library.is_file(), f"missing installed file: {library}"
    if sys.platform == "darwin":
        identities = subprocess.check_output(["otool", "-D", str(library)], text=True)
        assert all(line == f"@rpath/{library_name}" for line in identities.splitlines() if not line.endswith(":")), identities
    if args.architecture:
        subprocess.run(["lipo", str(library), "-verify_arch", *args.architecture], check=True, timeout=30)
    # A host Python cannot load a library built solely for another architecture.
    if not args.architecture or platform.machine() in args.architecture:
        ffi = ctypes.CDLL(str(library))
        ffi.lana_bridge_version.restype = ctypes.c_char_p
        assert ffi.lana_bridge_version().decode() == version
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
                assert result.stdout.startswith(f"Lana {version} (LABC v2,") and len(result.stdout.splitlines()) == 1, result.stdout
                if binary == "lana":
                    result = subprocess.run([*command, "run", str(source)], cwd=directory, env=env,
                                            capture_output=True, text=True, timeout=30)
                    assert result.returncode == 0 and result.stderr == "", result
                    assert result.stdout == "0.05\n", result.stdout
    print("CLEAN_INSTALL_PASS")


if __name__ == "__main__":
    main()
