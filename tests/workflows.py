"""Multi-command acceptance checks formerly driven by CMake scripts."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build import install_files


def run_process(command, *, timeout, **kwargs):
    # A timed-out shell fixture must not leave its HTTP server or VM children running.
    input_text = kwargs.pop("input", None)
    with subprocess.Popen(command, stdin=subprocess.PIPE if input_text is not None else None,
                          start_new_session=True, **kwargs) as process:
        try:
            output, error = process.communicate(input_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, output, error)


def bundle_compiler(destination):
    modules = ["lexer", "syntax", "parser", "resolver", "ir", "emitter", "main"]
    texts = []
    for module in modules:
        text = (ROOT / "compiler" / (module + ".lana")).read_text()
        text = re.sub(r'import[ \t]+"[^"]+"[ \t]+as[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t]*;[ \t]*\r?\n', '', text)
        for alias in modules[:-1]:
            text = re.sub(r'(^|[^A-Za-z0-9_])' + alias + r'\.', r'\1', text)
        texts.append(text + "\n")
    destination.write_text(''.join(texts))


def run(name, cli, compiler, work, env):
    def call(*args, cwd=work, input=None, clean=False):
        child_env = dict(env)
        if clean:
            for key in ["LANA_COMPILER_LABC", "LANA_STDLIB_DIR"]:
                child_env.pop(key, None)
        result = run_process(list(map(str, args)), cwd=cwd, env=child_env, input=input,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        if result.returncode:
            raise AssertionError(f"{args}: exit {result.returncode}\n{result.stdout}")
        return result.stdout

    def compile_to(source, output):
        call(cli, "run", compiler, "--memory-limit-mib", "256", "--instruction-limit", "50000000", "--", source, output)

    if name == "bootstrap":
        bundle = work / "compiler-bootstrap.lana"
        bundle_compiler(bundle)
        reference = (ROOT / "compiler/bootstrap/compiler.lasm").read_bytes()
        for suffix in ["first", "repeat"]:
            output = work / (suffix + ".lasm")
            compile_to(bundle, output)
            assert output.read_bytes() == reference, "compiler bootstrap is stale or nondeterministic"
    elif name == "probability_identity":
        outputs = []
        for kind in ["prob", "state"]:
            output = work / (kind + ".lasm")
            compile_to(ROOT / f"tests/regression/probability_identity_{kind}.lana", output)
            outputs.append(output.read_bytes())
        assert outputs[0] == outputs[1], "probability and state constructors differ"
    elif name == "lsp_protocol":
        messages = [{"jsonrpc":"2.0", "id":1, "method":"initialize", "params":{}},
                    {"jsonrpc":"2.0", "method":"initialized", "params":{}},
                    {"jsonrpc":"2.0", "method":"$/setTrace", "params":{"value":"off"}},
                    {"jsonrpc":"2.0", "id":2, "method":"shutdown", "params":None},
                    {"jsonrpc":"2.0", "method":"exit"}]
        body = ""
        for message in messages:
            text = json.dumps(message, separators=(',', ':'))
            body += f"Content-Length: {len(text)}\r\n\r\n{text}"
        response = call(cli, "lsp", input=body)
        assert "lana-lsp" in response and "renameProvider" in response, response
        assert '"id":null' not in response, "server responded to a notification"
    elif name == "project":
        source = work / "compile-only.lana"
        source.write_text('print("COMPILE_ONLY_SENTINEL"); assert(false, "must not execute");\n')
        before = {path.relative_to(work): path.read_bytes() for path in work.rglob("*") if path.is_file()}
        for arguments in [(cli, "check"), (cli, "check", source)]:
            rejected = run_process(list(map(str, arguments)), cwd=work, env=env,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            assert rejected.returncode == 2, rejected.stderr
            assert not re.search(r"\bcheck\b", rejected.stderr), rejected.stderr
            after = {path.relative_to(work): path.read_bytes() for path in work.rglob("*") if path.is_file()}
            assert after == before, "removed command changed artifacts"
        compiled = work / "compile-only.labc"
        output = call(cli, "compile", source, "-o", compiled)
        assert compiled.is_file() and "COMPILE_ONLY_SENTINEL" not in output, output
        project = work / "project-workflow"
        call(cli, "new", project)
        call(cli, "new", work / "project-workflow-dep")
        with (project / "lana.toml").open('a') as file:
            file.write('dep = "../project-workflow-dep"\n')
        assert (project / "src/belief.lana").is_file()
        assert 'import "./belief.lana" as belief' in (project / "src/main.lana").read_text()
        assert 'import "../src/belief.lana" as belief' in (project / "tests/main_test.lana").read_text()
        for command in ["build", "test", "run", "doc", "fmt"]:
            call(cli, command, cwd=project)
        before = (project / "src/main.lana").read_bytes()
        call(cli, "fmt", "--check", cwd=project)
        assert (project / "src/main.lana").read_bytes() == before, "formatter changed checked source"
        assert (project / "docs/api.md").is_file() and (project / ".lana/cache").is_dir()
        assert "dependency.dep" in (project / "lana.lock").read_text()
        source = project / "src/main.lana"
        original = source.read_bytes()
        source.write_text("let broken = ;\n")
        rejected = run_process([str(cli), "build"], cwd=project, env=env,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        assert rejected.returncode != 0 and "error[parse/LANA_ERR_PARSE]" in rejected.stderr, rejected.stderr
        source.write_bytes(original)
        call(cli, "build", cwd=project)
    elif name == "debugger":
        response = call(cli, "debug", ROOT / "tests/regression/m10_inspector_pass.lana", "--break", "4", input="s\nc\n")
        assert "BREAK line=4" in response and response.count("BREAK line=") == 2 and "frames=1" in response, response
    elif name == "install":
        prefix = install_files(cli, compiler, work / "install-prefix")
        installed = prefix / "bin/lana"
        child_env = dict(env, PATH=str(prefix / 'bin') + os.pathsep + env.get('PATH', ''))
        for key in ["LANA_COMPILER_LABC", "LANA_STDLIB_DIR"]:
            child_env.pop(key, None)
        result = subprocess.run(['bash', str(ROOT / 'scripts/verify-install.sh')], cwd=work,
                                env=child_env, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        # Imports must use the installed standard library, without the checkout as cwd.
        example = work / "installed.lana"
        example.write_text('import "std/core" as core; print(resolve(core.distribution([[1, 1]])));\n')
        call(installed, "run", example, clean=True)
        assert (ROOT / 'VERSION').read_text().strip() in call(installed, 'version', clean=True)
    else:
        raise ValueError(f"unknown workflow {name}")
