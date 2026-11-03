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
    for present in (False, True):
        for stage in ('before_file_sync', 'before_rename', 'after_rename'):
            path, old, new = output, old_compiler, new_compiler
            command, inspect = ('compile', source, '-o', output), ('verify', output)
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

    for operation in ("run", "run_labc", "start_live", "python"):
        store = root / ("store-" + operation)
        source.write_text('store_open(' + json.dumps(str(store)) + '); store_put("key", 42); store_commit();\n')
        assert run('compile', source, '-o', output).returncode == 0
        failed_env = dict(env, LANA_TEST_STORE_STAGE="commit_manifest")
        if operation == "python":
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/python/src"))
            from lana_integrations import Lana
            # Set only the child's environment; restore it before recovery.
            previous = os.environ.get("LANA_TEST_STORE_STAGE")
            os.environ["LANA_TEST_STORE_STAGE"] = "commit_manifest"
            try:
                with Lana(lana) as host:
                    result = host.run(source, {})
                assert not result.ok
                error = result.error
            finally:
                if previous is None: os.environ.pop("LANA_TEST_STORE_STAGE", None)
                else: os.environ["LANA_TEST_STORE_STAGE"] = previous
        else:
            request = {"schema":1,"op":operation,"path":str(output if operation == "run_labc" else source),"input":{}}
            failed = subprocess.run([lana, "bridge-worker"], input=json.dumps(request) + "\n",
                env=failed_env, capture_output=True, text=True, timeout=30)
            assert failed.returncode == 0, failed.stderr
            error = json.loads(failed.stdout)["error"]
        assert error["code"] == "LANA_ERR_IO", error
        assert error["durability"] == "uncertain" and error["path"] == str(store), error
        source.write_text('store_open(' + json.dumps(str(store)) + '); print(store_current_revision()); print(store_get("key"));\n')
        recovered = run('run', source)
        assert recovered.returncode == 0 and recovered.stdout == "1\n42\n", recovered.stderr

print('PUBLICATION_FAILPOINTS_PASS')
