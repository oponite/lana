from pathlib import Path
import subprocess

import pytest

from lana_tooling.compiler import CompileError, compile_source


ROOT = Path(__file__).parents[1]


def test_new_syntax_emits_native_state_bytecode_assembly() -> None:
    assembly = compile_source((ROOT / "examples" / "belief.lana").read_text())

    assert "STATE_BUILD" in assembly
    assert "APPLY" in assembly
    assert "MEASURE R0 probability" in assembly


def test_general_scalar_control_flow_compiles() -> None:
    assembly = compile_source(
        """
        let total = 0;
        let step = 1;
        while (total < 3) {
            total = total + step;
        }
        print(total);
        """
    )

    assert "COMPARE" in assembly
    assert "JUMP_IF_FALSE" in assembly
    assert "BINARY" in assembly


def test_state_rejects_matrix_style_invalid_probability() -> None:
    with pytest.raises(CompileError, match="p must be"):
        compile_source("state bad = state { p: 1.1, d: 0.2 };")


def test_end_to_end_new_source_executes_in_c_vm(tmp_path: Path) -> None:
    vm = ROOT / "build" / "ssvm"
    if not vm.exists():
        pytest.skip("native VM has not been built")
    assembly = tmp_path / "belief.ssa"
    bytecode = tmp_path / "belief.ssb"
    assembly.write_text(compile_source((ROOT / "examples" / "belief.lana").read_text()))
    subprocess.run([vm, "asm", assembly, "-o", bytecode], check=True)
    result = subprocess.run([vm, "run", bytecode], check=True, text=True, capture_output=True)

    assert result.stdout == "0.76\n"


def test_arrays_remain_ordinary_values_in_c_vm(tmp_path: Path) -> None:
    vm = ROOT / "build" / "ssvm"
    if not vm.exists():
        pytest.skip("native VM has not been built")
    assembly = tmp_path / "array.ssa"
    bytecode = tmp_path / "array.ssb"
    assembly.write_text(compile_source("let values = [1, 2, 3]; print(values[1]);"))
    subprocess.run([vm, "asm", assembly, "-o", bytecode], check=True)
    result = subprocess.run([vm, "run", bytecode], check=True, text=True, capture_output=True)

    assert result.stdout == "2\n"


def test_general_function_and_loop_execute_in_c_vm(tmp_path: Path) -> None:
    vm = ROOT / "build" / "ssvm"
    if not vm.exists():
        pytest.skip("native VM has not been built")
    assembly = tmp_path / "general.ssa"
    bytecode = tmp_path / "general.ssb"
    assembly.write_text(compile_source((ROOT / "examples" / "general.lana").read_text()))
    subprocess.run([vm, "asm", assembly, "-o", bytecode], check=True)
    result = subprocess.run([vm, "run", bytecode], check=True, text=True, capture_output=True)

    assert result.stdout == "5\n"


def test_multi_state_apply_executes_in_c_vm(tmp_path: Path) -> None:
    vm = ROOT / "build" / "ssvm"
    if not vm.exists():
        pytest.skip("native VM has not been built")
    assembly = tmp_path / "multiple.ssa"
    bytecode = tmp_path / "multiple.ssb"
    assembly.write_text(compile_source((ROOT / "examples" / "multiple_states.lana").read_text()))
    subprocess.run([vm, "asm", assembly, "-o", bytecode], check=True)
    result = subprocess.run([vm, "run", bytecode], check=True, text=True, capture_output=True)

    assert result.stdout == "0.5\n"


def test_indexes_and_history_compile_to_native_operations() -> None:
    assembly = compile_source(
        'state belief = state { p: 0.5, d: 0.2, timestamp: 1, source: "sensor", weight: 0.8, confidence: 0.9 };\n'
        "history belief latest 4;\n"
        "state evidence = state { p: 0.9, d: 0.4 };\n"
        "apply evidence -> belief;\n"
        "let delta = change(belief);\n"
        "let confidence = belief.confidence;\n"
        "print(delta);"
    )

    assert "SET_INDEX R0 timestamp" in assembly
    assert "SET_INDEX R0 source" in assembly
    assert "HISTORY R0 latest" in assembly
    assert "CHANGE R0" in assembly
    assert "GET_INDEX R0 confidence" in assembly


def test_malformed_bytecode_is_rejected_without_crashing(tmp_path: Path) -> None:
    vm = ROOT / "build" / "ssvm"
    if not vm.exists():
        pytest.skip("native VM has not been built")
    malformed = tmp_path / "malformed.ssb"
    malformed.write_bytes(b"SSBC\x00\x01")
    result = subprocess.run([vm, "verify", malformed], check=False, text=True, capture_output=True)

    assert result.returncode == 1
    assert "SS_ERR_FORMAT" in result.stderr


def run_lana_source(tmp_path: Path, source: str, *program_args: str) -> subprocess.CompletedProcess[str]:
    vm = ROOT / "build" / "ssvm"
    assembly = tmp_path / "program.ssa"
    bytecode = tmp_path / "program.ssb"
    assembly.write_text(compile_source(source))
    subprocess.run([vm, "asm", assembly, "-o", bytecode], check=True)
    return subprocess.run(
        [vm, "run", bytecode, "--", *program_args], check=False, text=True, capture_output=True
    )


def test_runtime_state_and_strings_with_whitespace(tmp_path: Path) -> None:
    result = run_lana_source(
        tmp_path,
        'let p = 0.8; let d = 0.4; state belief = state(p: p, d: d, source: "sensor one"); print(belief.source); print(measure belief);',
    )

    assert result.returncode == 0
    assert result.stdout == "sensor one\n0.8\n"


def test_host_arguments_and_text_io(tmp_path: Path) -> None:
    output = tmp_path / "output with spaces.txt"
    result = run_lana_source(
        tmp_path,
        f'let values = args(); let ignored = write_text("{output}", "hello world"); let text = read_text("{output}"); print(values); print(text);',
        "alpha",
        "beta",
    )

    assert result.returncode == 0
    assert result.stdout == "[alpha, beta]\nhello world\n"
    assert output.read_text() == "hello world"


def test_fork_snapshot_isolation_and_ordered_join_all(tmp_path: Path) -> None:
    result = run_lana_source(
        tmp_path,
        """
        fn update(belief, evidence) {
            apply evidence -> belief;
            return belief;
        }
        state belief = state(p: 0.5, d: 0.2);
        state positive = state(p: 0.9, d: 0.5);
        state negative = state(p: 0.1, d: 0.5);
        taskgroup {
            let first = fork update(belief, positive);
            let second = fork update(belief, negative);
            let results = join_all([first, second]);
            let first_result = results[0];
            let second_result = results[1];
            print(measure first_result);
            print(measure second_result);
            print(measure belief);
        }
        """,
    )

    assert result.returncode == 0
    assert result.stdout == "0.7\n0.3\n0.5\n"


def test_cancelled_unjoined_task_is_cleaned_up_by_group(tmp_path: Path) -> None:
    result = run_lana_source(
        tmp_path,
        """
        fn spin() {
            let value = 0;
            while (true) { value = value + 1; }
            return value;
        }
        taskgroup {
            let work = fork spin();
            cancel(work);
        }
        print("cancelled safely");
        """,
    )

    assert result.returncode == 0
    assert result.stdout == "cancelled safely\n"


def test_join_timeout_reports_runtime_error(tmp_path: Path) -> None:
    result = run_lana_source(
        tmp_path,
        """
        fn spin() {
            let value = 0;
            while (true) { value = value + 1; }
            return value;
        }
        let work = fork spin();
        let result = join_timeout(work, 0.001);
        """,
    )

    assert result.returncode == 1
    assert "SS_ERR_TIMEOUT" in result.stderr
