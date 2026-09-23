"""Failed compilation and replacement preserve existing output, across processes."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

lana = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    source, output = work / 'main.lana', work / 'main.labc'
    def compile(ok):
        result = subprocess.run([lana, 'compile', str(source), '-o', str(output)],
                                capture_output=True, text=True, timeout=120)
        assert (result.returncode == 0) == ok, result.stdout + result.stderr
    source.write_text('print(42);\n')
    compile(True)
    before = output.read_bytes()
    source.write_text('let broken = ;\n')
    compile(False)
    assert output.read_bytes() == before
    output.unlink()
    compile(False)
    assert not output.exists()
    output.mkdir()
    source.write_text('print(42);\n')
    compile(False)
    assert output.is_dir()
    assert sorted(p.name for p in work.iterdir()) == ['main.labc', 'main.lana']
print('COMPILER_OUTPUT_PASS')
