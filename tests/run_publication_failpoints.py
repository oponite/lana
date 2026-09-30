"""Build fault injection in a disposable target directory and check the normal CLI."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
cli, compiler = (Path(value).resolve() for value in sys.argv[1:3])
original = hashlib.sha256(cli.read_bytes()).digest()
with tempfile.TemporaryDirectory(prefix="lana-failpoints-") as directory:
    work = Path(directory)
    target = work / "target"
    subprocess.run(["cargo", "build", "--locked", "-p", "lana-cli", "--features", "publication-fault-injection",
                    "--target-dir", target], cwd=ROOT, check=True, timeout=480)
    subprocess.run([sys.executable, ROOT / "tests/test_publication_failpoints.py", target / "debug/lana", compiler],
                   cwd=ROOT, check=True, timeout=120)
    source = work / "normal.lana"
    source.write_text("print(1);\n")
    subprocess.run([cli, "compile", source, "-o", work / "normal.labc"], check=True, timeout=30,
                   env=dict(os.environ, LANA_COMPILER_LABC=str(compiler), LANA_TEST_ATOMIC_STAGE="before_rename"))
    assert hashlib.sha256(cli.read_bytes()).digest() == original
print("ISOLATED_FAILPOINTS_PASS")
