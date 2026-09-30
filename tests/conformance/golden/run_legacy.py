"""Published v1-v2 bytecode checks after retirement of the C reference."""

from pathlib import Path
import subprocess
import sys
import tempfile


def run(lana: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([lana, *args], text=True, capture_output=True, check=False)


def main() -> None:
    lana = sys.argv[1]
    root = Path(__file__).parent
    for name, expected in [
        ("v1", "7\n"),
        ("v2", "state(p=0.6, d_re=0.0591751709536, d_im=0)\n"),
    ]:
        chunk = root / f"{name}.labc"
        assert run(lana, "verify", str(chunk)).returncode == 0
        result = run(lana, "run-bytecode", str(chunk))
        assert result.returncode == 0 and result.stdout == expected, result
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rebuilt.labc"
            assert run(lana, "asm", str(root / f"{name}.lasm"), "-o", str(output)).returncode == 0
            assert output.read_bytes() == chunk.read_bytes(), name
    original = (root / "v2.labc").read_bytes()
    with tempfile.TemporaryDirectory() as temporary:
        for name, bytes_ in [
            ("bad-magic", b"NOPE" + original[4:]),
            ("truncated", original[:-1]),
            ("trailing", original + b"\x00"),
        ]:
            path = Path(temporary) / f"{name}.labc"
            path.write_bytes(bytes_)
            assert run(lana, "verify", str(path)).returncode != 0, name
    print("LABC_V1_V2_PASS")


if __name__ == "__main__":
    main()
