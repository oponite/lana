"""Opt-in CLI build: publication failures leave complete old or new artifacts."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile

lana = str(Path(sys.argv[1]).resolve())
compiler = str(Path(sys.argv[2]).resolve())

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / 'program.lana'
    output = root / 'program.labc'
    brain = root / 'model.lbrn'
    env = dict(os.environ, LANA_COMPILER_LABC=compiler)

    def run(*args, stage=None):
        process_env = dict(env)
        if stage:
            process_env['LANA_TEST_ATOMIC_STAGE'] = stage
        result = subprocess.run([lana, *map(str, args)], env=process_env,
                                capture_output=True, text=True, timeout=120)
        return result

    source.write_text('print(1);\n')
    assert run('compile', source, '-o', output).returncode == 0
    old_compiler = output.read_bytes()
    source.write_text('print(2);\n')
    assert run('compile', source, '-o', output).returncode == 0
    new_compiler = output.read_bytes()
    assert old_compiler != new_compiler
    assert run('brain', 'new', brain, 3, 2, 4, 1).returncode == 0
    old_brain = brain.read_bytes()
    assert run('brain', 'new', brain, 3, 2, 4, 2).returncode == 0
    new_brain = brain.read_bytes()
    assert old_brain != new_brain

    for present in (False, True):
        for stage in ('before_file_sync', 'before_rename', 'after_rename'):
            for path, old, new, command, inspect in (
                (output, old_compiler, new_compiler,
                 ('compile', source, '-o', output), ('verify', output)),
                (brain, old_brain, new_brain,
                 ('brain', 'new', brain, 3, 2, 4, 2), ('brain', 'inspect', brain)),
            ):
                path.unlink(missing_ok=True)
                if present:
                    path.write_bytes(old)
                failed = run(*command, stage=stage)
                assert failed.returncode != 0, (command, stage, failed.stdout, failed.stderr)
                assert 'LANA_ERR_IO' in failed.stderr, (command, stage, failed.stderr)
                uncertain = stage == 'after_rename'
                assert ('uncertain' in failed.stderr) == uncertain, (command, stage, failed.stderr)
                if uncertain:
                    assert str(path) in failed.stderr, failed.stderr
                expected = new if uncertain else old if present else None
                assert (path.read_bytes() if path.exists() else None) == expected, (command, stage)
                if expected is not None:
                    checked = run(*inspect)
                    assert checked.returncode == 0, (inspect, checked.stdout, checked.stderr)
                assert not list(root.glob(path.name + '.lana-*.tmp'))

    train = root / 'train.jsonl'
    valid = root / 'valid.jsonl'
    train.write_text('{"tokens":[0,1],"target":2}\n{"tokens":[1],"target":2}\n')
    valid.write_text('{"tokens":[0],"target":1}\n')
    architecture = root / 'architecture.json'
    architecture.write_text('{"hidden":[{"width":3,"activation":"relu"}]}')
    for format_name in ('LBRN1', 'LBRN2'):
        path = root / f'fit-{format_name}.lbrn'
        if format_name == 'LBRN1':
            assert run('brain', 'new', path, 3, 2, 3, 7).returncode == 0
        else:
            assert run('brain', 'new', path, 3, 2, '--architecture', architecture, '--seed', 7).returncode == 0
        old = path.read_bytes()
        command = ('brain', 'fit', path, train, valid, '--learning-rate', 0.1,
                   '--max-epochs', 2, '--patience', 1)
        assert run(*command).returncode == 0
        new = path.read_bytes()
        assert new != old
        new_steps = json.loads(run('brain', 'inspect', path).stdout.splitlines()[-1])['training_steps']
        for stage in ('before_file_sync', 'before_rename', 'after_rename'):
            path.write_bytes(old)
            failed = run(*command, stage=stage)
            assert failed.returncode != 0 and not failed.stdout, (format_name, stage, failed.stdout, failed.stderr)
            error = json.loads(failed.stderr.splitlines()[-1])
            assert error['error'] == 'LANA_ERR_IO', (format_name, stage, error)
            uncertain = stage == 'after_rename'
            assert ('durability' in error) == uncertain, (format_name, stage, error)
            if uncertain:
                assert error['durability'] == 'uncertain' and error['path'] == str(path), error
            assert path.read_bytes() == (new if uncertain else old), (format_name, stage)
            inspected = run('brain', 'inspect', path)
            assert inspected.returncode == 0, inspected.stderr
            assert json.loads(inspected.stdout.splitlines()[-1])['training_steps'] == (new_steps if uncertain else 0)
            assert not list(root.glob(path.name + '.lana-*.tmp'))

print('PUBLICATION_FAILPOINTS_PASS')
