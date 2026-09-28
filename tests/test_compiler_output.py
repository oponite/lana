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
        return result
    source.write_text('print(42);\n')
    compile(True)
    before = output.read_bytes()
    source.write_text('let broken = ;\n')
    failed = compile(False)
    assert 'LANA_ERR_PARSE' in failed.stderr, failed.stderr
    assert output.read_bytes() == before
    verified = subprocess.run([lana, 'verify', str(output)],
                              capture_output=True, text=True, timeout=120)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    output.unlink()
    failed = compile(False)
    assert 'LANA_ERR_PARSE' in failed.stderr, failed.stderr
    assert not output.exists()
    output.mkdir()
    source.write_text('print(42);\n')
    failed = compile(False)
    assert 'LANA_ERR_IO' in failed.stderr, failed.stderr
    assert 'durability: uncertain' not in failed.stderr, failed.stderr
    assert output.is_dir()
    assert sorted(p.name for p in work.iterdir()) == ['main.labc', 'main.lana']
    output.rmdir()
    loop_source, loop_compiler = work / 'loop.lasm', work / 'loop.labc'
    loop_source.write_text('loop:\nJUMP loop\n')
    assembled = subprocess.run([lana, 'asm', str(loop_source), '-o', str(loop_compiler)],
                               capture_output=True, text=True, timeout=120)
    assert assembled.returncode == 0, assembled.stdout + assembled.stderr
    for present in (False, True):
        if present:
            output.write_bytes(before)
        else:
            output.unlink(missing_ok=True)
        limited = subprocess.run([lana, 'compile', str(source), '-o', str(output)],
                                 env=dict(os.environ, LANA_COMPILER_LABC=str(loop_compiler)),
                                 capture_output=True, text=True, timeout=30)
        assert limited.returncode != 0 and 'LANA_ERR_LIMIT' in limited.stderr, limited.stderr
        assert 'instructions limit 50000000' in limited.stderr, limited.stderr
        if present:
            assert output.read_bytes() == before
        else:
            assert not output.exists()
        if present:
            verified = subprocess.run([lana, 'verify', str(output)], capture_output=True, text=True, timeout=120)
            assert verified.returncode == 0, verified.stdout + verified.stderr
    memory_source, memory_compiler = work / 'memory.lasm', work / 'memory.labc'
    memory_source.write_text('LOAD_CONST R0 1000000\n' +
                             ''.join(f'HOST_CALL array_new R0 1 R{index}\n' for index in range(1, 8)) +
                             'HALT\n')
    assembled = subprocess.run([lana, 'asm', str(memory_source), '-o', str(memory_compiler)],
                               capture_output=True, text=True, timeout=120)
    assert assembled.returncode == 0, assembled.stdout + assembled.stderr
    for present in (False, True):
        if present:
            output.write_bytes(before)
        else:
            output.unlink(missing_ok=True)
        limited = subprocess.run([lana, 'compile', str(source), '-o', str(output)],
                                 env=dict(os.environ, LANA_COMPILER_LABC=str(memory_compiler)),
                                 capture_output=True, text=True, timeout=30)
        assert limited.returncode != 0 and 'LANA_ERR_OOM' in limited.stderr, limited.stderr
        assert 'memory limit 268435456' in limited.stderr, limited.stderr
        if present:
            assert output.read_bytes() == before
            verified = subprocess.run([lana, 'verify', str(output)], capture_output=True, text=True, timeout=120)
            assert verified.returncode == 0, verified.stdout + verified.stderr
        else:
            assert not output.exists()
    cancel_source, cancel_compiler = work / 'cancel.lasm', work / 'cancel.labc'
    cancel_source.write_text('.function main 0 8\nFORK worker R1 0 R0\nCANCEL R0\nJOIN R0 R1\nHALT\n'
                             '.function worker 0 4\nloop:\nJUMP loop\n')
    assembled = subprocess.run([lana, 'asm', str(cancel_source), '-o', str(cancel_compiler)],
                               capture_output=True, text=True, timeout=120)
    assert assembled.returncode == 0, assembled.stdout + assembled.stderr
    for present in (False, True):
        if present:
            output.write_bytes(before)
        else:
            output.unlink(missing_ok=True)
        cancelled = subprocess.run([lana, 'compile', str(source), '-o', str(output)],
                                   env=dict(os.environ, LANA_COMPILER_LABC=str(cancel_compiler)),
                                   capture_output=True, text=True, timeout=30)
        assert cancelled.returncode != 0 and 'LANA_ERR_CANCELLED' in cancelled.stderr, cancelled.stderr
        assert 'cancellation: lineage' in cancelled.stderr, cancelled.stderr
        if present:
            assert output.read_bytes() == before
            verified = subprocess.run([lana, 'verify', str(output)], capture_output=True, text=True, timeout=120)
            assert verified.returncode == 0, verified.stdout + verified.stderr
        else:
            assert not output.exists()
    malformed_compiler = work / 'malformed.labc'
    malformed_compiler.write_bytes(b'LABCbad')
    for present in (False, True):
        if present:
            output.write_bytes(before)
        else:
            output.unlink(missing_ok=True)
        rejected = subprocess.run([lana, 'compile', str(source), '-o', str(output)],
                                  env=dict(os.environ, LANA_COMPILER_LABC=str(malformed_compiler)),
                                  capture_output=True, text=True, timeout=30)
        assert rejected.returncode != 0 and 'LANA_ERR_FORMAT' in rejected.stderr, rejected.stderr
        if present:
            assert output.read_bytes() == before
        else:
            assert not output.exists()

    project = work / 'project'
    created = subprocess.run([lana, 'new', str(project)], capture_output=True, text=True, timeout=120)
    assert created.returncode == 0, created.stdout + created.stderr
    def build(ok):
        result = subprocess.run([lana, 'build'], cwd=project,
                                capture_output=True, text=True, timeout=120)
        assert (result.returncode == 0) == ok, result.stdout + result.stderr
    build(True)
    project_output = next((project / 'build').glob('*.labc'))
    lock = project / 'lana.lock'
    old_output, old_lock = project_output.read_bytes(), lock.read_bytes()
    cache = next((project / '.lana' / 'cache').glob('*.labc'))
    cache.write_bytes(b'broken cache')
    build(False)
    assert project_output.read_bytes() == old_output
    assert lock.read_bytes() == old_lock
    cache.unlink()
    build(True)
    project_output.unlink()
    project_output.mkdir()
    build(False)
    assert project_output.is_dir()
    assert lock.read_bytes() == old_lock
print('COMPILER_OUTPUT_PASS')
