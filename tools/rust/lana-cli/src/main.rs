//! Command-line driver for the Rust Lana runtime (phase 2 of the Rust runtime
//! boundary).
//!
//! `lana run <file.labc> [--seed N] [--stats]` loads and verifies a chunk,
//! runs it on the Rust VM, and reports the result. Output matches `tools/c/cli.c`
//! `load_command` so differential spot-checks can compare the two byte-for-byte.
//!
//! The full command surface mirrors `tools/c/cli.c` `main()`: `version`, `new`,
//! `lsp`, `fmt`, `doc`, `build`, `test`, `compile`, `check`, `asm`, `debug`,
//! `run`, `run-bytecode`, `dis`, and `verify`. Commands that need the
//! self-hosted compiler locate `lana-compiler.labc` and run it on the Rust VM.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use lana_bytecode::{Chunk, LanaError, LanaErrorInfo, OpCode, Value};
use lana_vm::Vm;

mod repl;

#[cfg(test)]
mod review_tests {
    use super::*;
    use std::ffi::{CStr, CString};
    use lana_ffi::{lana_bridge_free, lana_bridge_run_labc};

    #[test]
    fn memory_limit_boundaries() {
        let maximum = usize::MAX / (1024 * 1024);
        let path = temp_path("lana-memory-test");
        write_chunk(&lana_bytecode::assemble("HALT\n").unwrap(), path.to_str().unwrap()).unwrap();
        for value in [1, maximum - 1, maximum] {
            assert_eq!(parse_memory_limit(&value.to_string()), Some(value * 1024 * 1024));
            let args = vec![path.to_string_lossy().into_owned(), "--memory-limit-mib".into(), value.to_string()];
            assert_eq!(run_command(&args), ExitCode::SUCCESS);
            assert_eq!(disassemble_command(&args), Ok(()));
        }
        for value in ["0".to_string(), "-1".into(), "invalid".into(),
                      (maximum + 1).to_string(), usize::MAX.to_string(), "18446744073709551615".into()] {
            assert_eq!(parse_memory_limit(&value), None);
            // An invalid flag must fail before trying to open the bytecode.
            let args = vec!["missing.labc".into(), "--memory-limit-mib".into(), value];
            assert_eq!(run_command(&args), ExitCode::from(2));
            assert_eq!(disassemble_command(&args), Err(2));
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn bridge_repeated_success_and_failure() {
        let labc = temp_path("lana-bridge-test");
        let response = temp_path("lana-response-test");
        let store = temp_path("lana-bridge-store");
        let labc_c = CString::new(labc.to_str().unwrap()).unwrap();
        let response_c = CString::new(response.to_str().unwrap()).unwrap();
        let hex = |text: &str| text.as_bytes().iter().map(|b| format!("{b:02x}")).collect::<String>();
        let message = "é 中文 😀 \"quoted\" \\ slash\n\t\u{0001}";
        let programs = [
            (format!(".function main 0 8\nLOAD_STRING R0 {}\nHOST_CALL store_open R0 1 R2\nLOAD_STRING R0 6b6579\nLOAD_CONST R1 42\nHOST_CALL store_put R0 2 R2\nHOST_CALL store_commit R0 0 R2\nHOST_CALL store_get R0 1 R2\nLOAD_CONST R3 42\nCOMPARE R2 == R3 R4\nLOAD_STRING R5 73746f7265\nHOST_CALL assert R4 2 R6\nLOAD_CONST R0 0\nLOAD_STRING R1 7b7d\nHOST_CALL adapter_load R0 2 R2\nHOST_CALL adapter_fetch R1 1 R2\nLOAD_STRING R0 {}\nLOAD_STRING R1 7b7d\nHOST_CALL write_text R0 2 R2\nRETURN R2\n", hex(store.to_str().unwrap()), hex(response.to_str().unwrap())), 0, "\"ok\":true".into()),
            (format!(".function main 0 4\nLOAD_STRING R0 {}\nLOAD_STRING R1 7b7d\nHOST_CALL write_text R0 2 R2\nRETURN R2\n", hex(response.to_str().unwrap())), 0, "\"ok\":true".to_string()),
            (format!(".function main 0 4\nLOAD_CONST R0 false\nLOAD_STRING R1 {}\nHOST_CALL assert R0 2 R2\nRETURN R2\n", hex(message)), LanaError::Assertion as i32,
             format!("\"message\":{}", lana_runtime::data::json_stringify(&lana_vm::value::Value::string(message.into())).unwrap())),
            (format!(".function main 0 4\nLOAD_STRING R0 {}\nLOAD_STRING R1 696e76616c6964\nHOST_CALL write_text R0 2 R2\nRETURN R2\n", hex(response.to_str().unwrap())), -4, "LANA_RESPONSE_INVALID".into()),
            ("HALT\n".into(), -3, "LANA_RESPONSE_MISSING".into()),
        ];
        for (source, status, expected) in programs {
            let mut chunk = lana_bytecode::assemble(&source).unwrap();
            // A retained constant makes the original leak substantial and detectable.
            chunk.constants.push(Value::String("x".repeat(1024 * 1024)));
            write_chunk(&chunk, labc.to_str().unwrap()).unwrap();
            for _ in 0..16 {
                let mut envelope = std::ptr::null_mut();
                let result = unsafe {
                    lana_bridge_run_labc(labc_c.as_ptr(), c"request.json".as_ptr(), response_c.as_ptr(), std::ptr::null(), &mut envelope)
                };
                assert!(!envelope.is_null());
                let text = unsafe { CStr::from_ptr(envelope).to_str().unwrap().to_string() };
                unsafe { lana_bridge_free(envelope); }
                assert_eq!(result, status, "{text}");
                assert!(text.contains(&expected), "{text}");
                assert!(lana_runtime::data::json_parse(&text).is_ok());
            }
        }
        std::fs::remove_file(labc).unwrap();
        let _ = std::fs::remove_file(response);
        std::fs::remove_dir_all(store).unwrap();
    }
}

fn lana_version() -> &'static str {
    include_str!("../../../../VERSION").trim()
}

/// Full usage text, mirroring `usage()` in `tools/c/cli.c` (with `lanavm` folded
/// into the single `lana` binary).
fn usage(program: &str) {
    eprintln!(
        "usage:\n  {program} compile program.lana -o program.labc\n  {program} new directory\n  {program} lsp\n  {program} debug program.lana\n  {program} repl\n  {program} build|run|test|check|fmt|doc\n  {program} check program.lana\n  {program} asm program.lasm -o program.labc\n  {program} run program.labc [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--memory-limit-mib N] [--instruction-limit N]\n  {program} run-bytecode program.labc [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--memory-limit-mib N] [--instruction-limit N]\n  {program} dis program.labc\n  {program} verify program.labc\n  {program} inspect program.lana [--format json|dot]"
    );
}

/// Run-specific usage, kept byte-identical to the original `run` command.
fn run_usage(program: &str) {
    eprintln!(
        "usage: {program} run <file.labc> [--seed N] [--workers N] [--max-tasks N] [--memory-limit-mib N] [--instruction-limit N] [--stats]"
    );
}

fn report_error(error: &lana_vm::VmError) {
    let path = if error.function.is_empty() { "<bytecode>" } else { &error.function };
    // Matches `tools/c/cli.c` `report_error`: VM runtime errors carry a source
    // span of (line, 1)-(line, 1) and an error kind, e.g.
    // `<bytecode>:7:1-7:1: error[validation/LANA_ERR_HISTORY]: ...`.
    eprintln!(
        "{}:{}:1-{}:1: error[{}/{}]: {} (operation {}) (instruction {}, opcode {})",
        path,
        error.line,
        error.line,
        error.code.kind_name(),
        error.code.name(),
        error.message,
        error.operation,
        error.ip,
        OpCode::try_from(error.opcode).map(|op| op.name()).unwrap_or("UNKNOWN"),
    );
    if error.resolution_reason != lana_vm::LANA_RESOLUTION_REASON_NONE {
        // `lana_error_set_resolution` always marks the count as present, so
        // the C11 CLI prints `, remaining alternatives: N` even for 0.
        eprintln!(
            "  resolution: {}, remaining alternatives: {}",
            lana_vm::resolution_reason_name(error.resolution_reason),
            error.remaining_alternatives,
        );
    }
    if let Some((support, detail)) = &error.exact_support {
        eprintln!(
            "  exact support: {} ({})",
            lana_vm::exact_support_name(*support),
            detail,
        );
    }
    if let Some((lineage, reason)) = &error.cancellation {
        eprintln!("  cancellation: lineage {lineage} ({reason})");
    }
    if let Some((resource, limit, observed, unit)) = &error.resource_limit {
        eprintln!(
            "  resource: {} limit {limit}, observed {observed} {unit}",
            lana_vm::resource_kind_name(*resource),
        );
    }
}

/// A failure from one of the compiler-delegating commands.
enum CliError {
    /// Failed to read/load a bytecode file (compiler or otherwise).
    Load { path: String, info: LanaErrorInfo },
    /// The compiler (or a program) failed at runtime on the VM.
    Run(lana_vm::VmError),
    /// The compiler emitted assembly that failed to assemble.
    Assemble { path: String, info: LanaErrorInfo },
    /// Failed to write a chunk to disk.
    Write { path: String, info: LanaErrorInfo },
    /// A project-level failure (missing manifest, bad plan, I/O).
    Project,
}

fn report_cli_error(error: &CliError) {
    match error {
        CliError::Load { path, info } => {
            eprintln!("{path}:{}: error[{}]: {}", info.line, info.code.name(), info.message);
        }
        CliError::Run(vm_error) => report_error(vm_error),
        CliError::Assemble { path, info } => {
            eprintln!(
                "{path}:{}: error[{}]: {} (instruction {}, opcode {})",
                info.line,
                info.code.name(),
                info.message,
                info.ip,
                OpCode::try_from(info.opcode).map(|op| op.name()).unwrap_or("UNKNOWN"),
            );
        }
        CliError::Write { path, info } => {
            eprintln!("{path}: error[{}]: {}", info.code.name(), info.message);
        }
        CliError::Project => {
            eprintln!("project build failed");
        }
    }
}

/// A unique temporary path, standing in for the C CLI's `mkstemp` calls.
fn temp_path(prefix: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
}

/// Resolve the self-hosted compiler bytecode, mirroring `find_compiler` in
/// `tools/c/cli.c`: `LANA_COMPILER_LABC`, then `lana-compiler.labc` in the CWD,
/// then next to the executable, then each `PATH` entry.
fn find_compiler() -> Option<PathBuf> {
    if let Ok(configured) = std::env::var("LANA_COMPILER_LABC") {
        let path = PathBuf::from(&configured);
        if path.exists() {
            return Some(path);
        }
    }
    let cwd = PathBuf::from("lana-compiler.labc");
    if cwd.exists() {
        return Some(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("lana-compiler.labc");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    if let Ok(path) = std::env::var("PATH") {
        for entry in path.split(':') {
            let dir = if entry.is_empty() { Path::new(".") } else { Path::new(entry) };
            let candidate = dir.join("lana-compiler.labc");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Point the compiler at the installed stdlib, mirroring `set_stdlib_dir` in
/// `tools/c/compiler_service.c`: `<prefix>/bin/lana-compiler.labc` →
/// `<prefix>/share/lana/stdlib`, set only when that directory exists and never
/// overriding a user-supplied value.
fn set_stdlib_dir(compiler: &Path) {
    if std::env::var("LANA_STDLIB_DIR").is_ok() {
        return;
    }
    if let Some(dir) = compiler
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("share/lana/stdlib"))
    {
        if dir.is_dir() {
            std::env::set_var("LANA_STDLIB_DIR", dir);
        }
    }
}

/// Run the compiler bytecode on the Rust VM with the given program args,
/// mirroring `run_compiler_program` in `tools/c/cli.c`.
fn run_compiler_program(compiler: &Path, args: &[String]) -> Result<(), CliError> {
    let path_str = compiler.to_string_lossy().into_owned();
    let bytes = std::fs::read(compiler).map_err(|_| CliError::Load {
        path: path_str.clone(),
        info: LanaErrorInfo::new(LanaError::Io, 0, 0, 0, "cannot read compiler bytecode"),
    })?;
    let chunk = lana_bytecode::loader::load(&bytes)
        .map_err(|info| CliError::Load { path: path_str, info })?;
    let mut vm = Vm::new(&chunk);
    set_stdlib_dir(compiler);
    vm.set_program_args(args);
    let result = vm.run();
    if result != LanaError::Ok {
        return Err(CliError::Run(vm.error().clone()));
    }
    Ok(())
}

/// Serialize a chunk to the LABC v2 on-disk format, mirroring
/// `lana_chunk_write_file` in `vm/c/bytecode.c` (the inverse of
/// `lana_bytecode::loader::load`).
fn write_chunk(chunk: &Chunk, path: &str) -> Result<(), LanaErrorInfo> {
    let mut out = Vec::new();
    out.extend_from_slice(b"LABC");
    out.extend_from_slice(&chunk.version.to_le_bytes());
    out.extend_from_slice(&(chunk.constants.len() as u32).to_le_bytes());
    out.extend_from_slice(&(chunk.functions.len() as u32).to_le_bytes());
    out.extend_from_slice(&(chunk.code.len() as u32).to_le_bytes());
    out.extend_from_slice(&chunk.entry.to_le_bytes());
    for constant in &chunk.constants {
        out.push(constant.value_type() as u8);
        match constant {
            Value::Null => {}
            Value::Number(number) => out.extend_from_slice(&number.to_bits().to_le_bytes()),
            Value::Bool(boolean) => out.push(if *boolean { 1 } else { 0 }),
            Value::String(string) => {
                out.extend_from_slice(&(string.len() as u32).to_le_bytes());
                out.extend_from_slice(string.as_bytes());
            }
        }
    }
    for function in &chunk.functions {
        out.extend_from_slice(&(function.name.len() as u32).to_le_bytes());
        out.extend_from_slice(function.name.as_bytes());
        out.extend_from_slice(&function.entry.to_le_bytes());
        out.extend_from_slice(&function.register_count.to_le_bytes());
        out.extend_from_slice(&function.arity.to_le_bytes());
    }
    for instruction in &chunk.code {
        out.push(instruction.opcode as u8);
        out.extend_from_slice(&instruction.a.to_le_bytes());
        out.extend_from_slice(&instruction.b.to_le_bytes());
        out.extend_from_slice(&instruction.c.to_le_bytes());
        out.extend_from_slice(&instruction.imm.to_le_bytes());
        out.extend_from_slice(&instruction.line.to_le_bytes());
    }
    std::fs::write(path, out)
        .map_err(|_| LanaErrorInfo::new(LanaError::Io, 0, 0, 0, "cannot write output file"))
}

/// Compile a `.lana` source to a `.labc` chunk, mirroring
/// `compile_source_file` in `tools/c/cli.c`: run the compiler to emit assembly,
/// assemble it, and write the chunk.
fn compile_source_file(compiler: &Path, source_path: &str, output_path: &str) -> Result<(), CliError> {
    let asm_path = temp_path("lana-assembly");
    let asm_str = asm_path.to_string_lossy().into_owned();
    let program_args = vec![source_path.to_string(), asm_str.clone()];
    if let Err(error) = run_compiler_program(compiler, &program_args) {
        let _ = std::fs::remove_file(&asm_path);
        return Err(error);
    }
    let asm_text = match std::fs::read_to_string(&asm_path) {
        Ok(text) => text,
        Err(_) => {
            let _ = std::fs::remove_file(&asm_path);
            return Err(CliError::Project);
        }
    };
    let _ = std::fs::remove_file(&asm_path);
    let chunk = lana_bytecode::assemble(&asm_text)
        .map_err(|info| CliError::Assemble { path: source_path.to_string(), info })?;
    write_chunk(&chunk, output_path)
        .map_err(|info| CliError::Write { path: output_path.to_string(), info })
}

/// Compile an in-memory `.lana` source string to a chunk, mirroring
/// `compile_source_file` but returning the chunk instead of writing it to
/// disk. Used by the REPL (LIP-020) to compile each input.
fn compile_source_to_chunk(compiler: &Path, source_text: &str) -> Result<Chunk, CliError> {
    let source_path = temp_path("lana-repl-source");
    let source_str = source_path.to_string_lossy().into_owned();
    if std::fs::write(&source_path, source_text).is_err() {
        return Err(CliError::Load {
            path: source_str,
            info: LanaErrorInfo::new(LanaError::Io, 0, 0, 0, "cannot write source"),
        });
    }
    let asm_path = temp_path("lana-repl-assembly");
    let asm_str = asm_path.to_string_lossy().into_owned();
    let program_args = vec![source_str.clone(), asm_str.clone()];
    let result = run_compiler_program(compiler, &program_args);
    let _ = std::fs::remove_file(&source_path);
    if let Err(error) = result {
        let _ = std::fs::remove_file(&asm_path);
        return Err(error);
    }
    let asm_text = match std::fs::read_to_string(&asm_path) {
        Ok(text) => text,
        Err(_) => {
            let _ = std::fs::remove_file(&asm_path);
            return Err(CliError::Project);
        }
    };
    let _ = std::fs::remove_file(&asm_path);
    lana_bytecode::assemble(&asm_text)
        .map_err(|info| CliError::Assemble { path: source_str, info })
}

fn parse_memory_limit(text: &str) -> Option<usize> {
    text.parse::<usize>().ok().filter(|&mib| mib > 0)?.checked_mul(1024 * 1024)
}

fn run_command(args: &[String]) -> ExitCode {
    let mut seed: u64 = 0x4c414e41;
    let mut workers: Option<usize> = None;
    let mut max_tasks: Option<usize> = None;
    let mut memory_limit: Option<usize> = None;
    let mut instruction_limit: Option<u64> = None;
    let mut stats = false;
    let mut trace = false;
    let mut debug = false;
    let mut break_line = None;
    let mut path: Option<&str> = None;
    let mut program_args: Vec<String> = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--" => {
                program_args = args[index + 1..].to_vec();
                break;
            }
            "--seed" => {
                if index + 1 >= args.len() {
                    eprintln!("invalid seed");
                    return ExitCode::from(2);
                }
                seed = match args[index + 1].parse() {
                    Ok(value) => value,
                    Err(_) => {
                        eprintln!("invalid seed");
                        return ExitCode::from(2);
                    }
                };
                index += 2;
            }
            "--workers" | "--max-tasks" => {
                if index + 1 >= args.len() {
                    eprintln!("invalid scheduler limit");
                    return ExitCode::from(2);
                }
                let parsed: usize = match args[index + 1].parse() {
                    Ok(value) if value > 0 => value,
                    _ => {
                        eprintln!("invalid scheduler limit");
                        return ExitCode::from(2);
                    }
                };
                if args[index] == "--workers" {
                    workers = Some(parsed);
                } else {
                    max_tasks = Some(parsed);
                }
                index += 2;
            }
            "--memory-limit-mib" => {
                if index + 1 >= args.len() {
                    eprintln!("invalid memory limit");
                    return ExitCode::from(2);
                }
                let parsed = match parse_memory_limit(&args[index + 1]) {
                    Some(value) => value,
                    _ => {
                        eprintln!("invalid memory limit");
                        return ExitCode::from(2);
                    }
                };
                memory_limit = Some(parsed);
                index += 2;
            }
            "--instruction-limit" => {
                if index + 1 >= args.len() {
                    eprintln!("invalid instruction limit");
                    return ExitCode::from(2);
                }
                let parsed: u64 = match args[index + 1].parse() {
                    Ok(value) if value > 0 => value,
                    _ => {
                        eprintln!("invalid instruction limit");
                        return ExitCode::from(2);
                    }
                };
                instruction_limit = Some(parsed);
                index += 2;
            }
            "--trace" => {
                trace = true;
                index += 1;
            }
            "--debug" => {
                debug = true;
                index += 1;
            }
            "--break" => {
                break_line = args.get(index + 1)
                    .and_then(|text| text.parse::<u32>().ok()).filter(|&line| line > 0);
                if break_line.is_none() {
                    eprintln!("invalid breakpoint line");
                    return ExitCode::from(2);
                }
                debug = true;
                index += 2;
            }
            "--stats" => {
                stats = true;
                index += 1;
            }
            value if value.starts_with('-') => {
                run_usage("lana");
                return ExitCode::from(2);
            }
            value => {
                if path.is_some() {
                    run_usage("lana");
                    return ExitCode::from(2);
                }
                path = Some(value);
                index += 1;
            }
        }
    }
    let Some(path) = path else {
        run_usage("lana");
        return ExitCode::from(2);
    };
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("{path}: error[LANA_ERR_IO]: {error}");
            return ExitCode::from(1);
        }
    };
    let chunk = match lana_bytecode::loader::load(&bytes) {
        Ok(chunk) => chunk,
        Err(info) => {
            eprintln!(
                "{path}:{}: error[{}]: {}",
                info.line,
                info.code.name(),
                info.message,
            );
            return ExitCode::from(1);
        }
    };
    let mut vm = lana_vm::Vm::new(&chunk);
    if debug || trace {
        let mut step = debug && break_line.is_none();
        vm.set_instruction_hook(move |chunk, ip, function, frames, task_id| {
            use std::io::Write;
            let line = chunk.code[ip].line;
            if task_id == 0 && (step || break_line == Some(line)) {
                println!("BREAK line={line} instruction={ip} function={function} frames={frames}");
                print!("debug [s]tep [c]ontinue [q]uit> ");
                if std::io::stdout().flush().is_err() { return false; }
                let mut command = String::new();
                match std::io::stdin().read_line(&mut command) {
                    Ok(0) | Err(_) => return false,
                    _ if command.starts_with('q') => return false,
                    _ => {}
                }
                step = command.starts_with('s');
                if !step { break_line = None; }
            }
            if trace {
                if task_id != 0 { print!("[task {task_id}] "); }
                print!("{}", lana_bytecode::disassembler::disassemble_instruction(chunk, ip));
            }
            true
        });
    }
    vm.seed(seed);
    if let Some(workers) = workers {
        if vm.set_worker_count(workers) != LanaError::Ok {
            return ExitCode::from(1);
        }
    }
    if let Some(max_tasks) = max_tasks {
        if vm.set_task_limit(max_tasks) != LanaError::Ok {
            return ExitCode::from(1);
        }
    }
    if let Some(bytes) = memory_limit {
        vm.set_memory_limit(bytes);
    }
    if let Some(limit) = instruction_limit {
        vm.set_instruction_limit(limit);
    }
    vm.set_program_args(&program_args);
    let mut store_host = lana_runtime::host_calls::StoreHost::with_heap(vm.heap());
    vm.set_host_call_extension(Box::new(move |host_id, args, out| {
        store_host.dispatch(host_id, args, out)
    }));
    let result = vm.run();
    if stats {
        let mut opcodes = String::new();
        for (opcode, count) in vm.opcode_counts().iter().enumerate() {
            if opcode != 0 {
                opcodes.push(',');
            }
            let name = OpCode::try_from(opcode as u8).map(|op| op.name()).unwrap_or("UNKNOWN");
            opcodes.push_str(&format!("\"{name}\":{count}"));
        }
        eprintln!(
            "LANAVM_STATS {{\"instructions\":{},\"state_transitions\":{},\"allocations\":{},\"allocated_bytes\":{},\"elapsed_ns\":0,\"opcodes\":{{{}}}}}",
            vm.instruction_count(),
            vm.state_transition_count(),
            vm.allocation_count(),
            vm.allocated_bytes(),
            opcodes,
        );
    }
    if result != LanaError::Ok {
        report_error(vm.error());
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Load a `.labc` and disassemble it to stdout, mirroring `load_command` with
/// `execute == false`. Returns `Ok(())` on success or `Err(exit_code)`.
fn disassemble_command(args: &[String]) -> Result<(), u8> {
    let mut path: Option<&str> = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--" => break,
            "--trace" | "--debug" | "--stats" => {
                index += 1;
            }
            "--break" => {
                if index + 1 >= args.len() {
                    eprintln!("invalid breakpoint line");
                    return Err(2);
                }
                let parsed: u32 = match args[index + 1].parse() {
                    Ok(value) if value > 0 => value,
                    _ => {
                        eprintln!("invalid breakpoint line");
                        return Err(2);
                    }
                };
                let _ = parsed;
                index += 2;
            }
            "--seed" => {
                if index + 1 >= args.len() {
                    eprintln!("invalid seed");
                    return Err(2);
                }
                if args[index + 1].parse::<u64>().is_err() {
                    eprintln!("invalid seed");
                    return Err(2);
                }
                index += 2;
            }
            "--memory-limit-mib" => {
                if index + 1 >= args.len() {
                    eprintln!("invalid memory limit");
                    return Err(2);
                }
                let parsed = match parse_memory_limit(&args[index + 1]) {
                    Some(value) => value,
                    _ => {
                        eprintln!("invalid memory limit");
                        return Err(2);
                    }
                };
                let _ = parsed;
                index += 2;
            }
            "--instruction-limit" => {
                if index + 1 >= args.len() {
                    eprintln!("invalid instruction limit");
                    return Err(2);
                }
                let parsed: u64 = match args[index + 1].parse() {
                    Ok(value) if value > 0 => value,
                    _ => {
                        eprintln!("invalid instruction limit");
                        return Err(2);
                    }
                };
                let _ = parsed;
                index += 2;
            }
            "--workers" | "--max-tasks" => {
                if index + 1 >= args.len() {
                    eprintln!("invalid scheduler limit");
                    return Err(2);
                }
                let parsed: usize = match args[index + 1].parse() {
                    Ok(value) if value > 0 => value,
                    _ => {
                        eprintln!("invalid scheduler limit");
                        return Err(2);
                    }
                };
                let _ = parsed;
                index += 2;
            }
            value if value.starts_with('-') => {
                usage("lana");
                return Err(2);
            }
            value => {
                if path.is_some() {
                    usage("lana");
                    return Err(2);
                }
                path = Some(value);
                index += 1;
            }
        }
    }
    let Some(path) = path else {
        usage("lana");
        return Err(2);
    };
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("{path}: error[LANA_ERR_IO]: {error}");
            return Err(1);
        }
    };
    let chunk = match lana_bytecode::loader::load(&bytes) {
        Ok(chunk) => chunk,
        Err(info) => {
            eprintln!(
                "{path}:{}: error[{}]: {}",
                info.line,
                info.code.name(),
                info.message,
            );
            return Err(1);
        }
    };
    print!("{}", lana_bytecode::disassembler::disassemble(&chunk));
    Ok(())
}

/// Assemble a `.lasm` text file to a `.labc` chunk, mirroring
/// `assemble_command` in `tools/c/cli.c`.
fn assemble_command(args: &[String]) -> ExitCode {
    if args.len() != 3 || args[1] != "-o" {
        usage("lana");
        return ExitCode::from(2);
    }
    let input = &args[0];
    let output = &args[2];
    let text = match std::fs::read_to_string(input) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("{input}: error[LANA_ERR_IO]: {error}");
            return ExitCode::from(1);
        }
    };
    let chunk = match lana_bytecode::assemble(&text) {
        Ok(chunk) => chunk,
        Err(info) => {
            report_cli_error(&CliError::Assemble { path: input.clone(), info });
            return ExitCode::from(1);
        }
    };
    if let Err(info) = write_chunk(&chunk, output) {
        report_cli_error(&CliError::Write { path: output.clone(), info });
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Run a compiler project tool (`--project-fmt` / `--project-doc`), mirroring
/// `run_project_tool` in `tools/c/cli.c`.
fn run_project_tool(mode: &str, argument: &str) -> ExitCode {
    let Some(compiler) = find_compiler() else {
        eprintln!("native Lana compiler bytecode not found");
        return ExitCode::from(1);
    };
    let program_args = vec![mode.to_string(), argument.to_string()];
    match run_compiler_program(&compiler, &program_args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            report_cli_error(&error);
            ExitCode::from(1)
        }
    }
}

struct Project {
    name: String,
    #[allow(dead_code)] // parsed and validated, but not used by the build path
    version: String,
    entry: String,
}

/// Extract `key = "value"` from a TOML-ish manifest, mirroring `quoted_value`
/// in `tools/c/project.c`.
fn quoted_value(text: &str, key: &str) -> Option<String> {
    let prefix = format!("{key} = \"");
    let start = text.find(&prefix)? + prefix.len();
    let end = text[start..].find('"')? + start;
    Some(text[start..end].to_string())
}

/// Load a project manifest, mirroring `lana_project_load` in `tools/c/project.c`.
fn project_load(directory: &str) -> Option<Project> {
    let manifest_path = Path::new(directory).join("lana.toml");
    let text = match std::fs::read_to_string(&manifest_path) {
        Ok(text) => text,
        Err(_) => {
            eprintln!("lana.toml not found");
            return None;
        }
    };
    if !text.contains("schema = 1") {
        eprintln!("invalid lana.toml schema");
        return None;
    }
    let name = match quoted_value(&text, "name") {
        Some(value) => value,
        None => {
            eprintln!("invalid lana.toml schema");
            return None;
        }
    };
    let version = match quoted_value(&text, "version") {
        Some(value) => value,
        None => {
            eprintln!("invalid lana.toml schema");
            return None;
        }
    };
    let entry = match quoted_value(&text, "entry") {
        Some(value) => value,
        None => {
            eprintln!("invalid lana.toml schema");
            return None;
        }
    };
    Some(Project { name, version, entry })
}

/// Everything after the first line of a plan file (the dependency lock lines).
fn extract_dependencies(plan: &str) -> String {
    match plan.find('\n') {
        Some(position) => plan[position + 1..].to_string(),
        None => String::new(),
    }
}

/// Finish a project build: compile the entry source into the content-addressed
/// cache, copy it to `build/<name>.labc`, and write `lana.lock`. Mirrors
/// `project_finish_build` in `tools/c/project.c`.
fn project_finish_build(
    directory: &str,
    project: &Project,
    hash: u64,
    locked: &str,
    compiler: &Path,
    output: &mut String,
) -> Result<(), ()> {
    let source_path = Path::new(directory).join(&project.entry);
    let cache_dir = Path::new(directory).join(".lana").join("cache");
    let build_dir = Path::new(directory).join("build");
    if std::fs::create_dir_all(&cache_dir).is_err() || std::fs::create_dir_all(&build_dir).is_err() {
        return Err(());
    }
    let cache_path = cache_dir.join(format!("{hash:016x}.labc"));
    let output_path = build_dir.join(format!("{}.labc", project.name));
    if !cache_path.exists() {
        if let Err(error) = compile_source_file(
            compiler,
            &source_path.to_string_lossy(),
            &cache_path.to_string_lossy(),
        ) {
            report_cli_error(&error);
            return Err(());
        }
    }
    if std::fs::copy(&cache_path, &output_path).is_err() {
        return Err(());
    }
    let lock_path = Path::new(directory).join("lana.lock");
    let lock = format!(
        "schema = 1\nproject = \"{}\"\ncontent = \"{hash:016x}\"\n{locked}",
        project.name
    );
    if std::fs::write(&lock_path, lock).is_err() {
        return Err(());
    }
    *output = output_path.to_string_lossy().into_owned();
    println!("built {}", output_path.display());
    Ok(())
}

/// Build a project using the compiler's `--project-plan` output, mirroring
/// `lana_project_build_with_plan` in `tools/c/project.c`.
fn project_build_with_plan(directory: &str, compiler: &Path, output: &mut String) -> Result<(), ()> {
    let project = project_load(directory).ok_or(())?;
    let plan_path = temp_path("lana-project-plan");
    let plan_str = plan_path.to_string_lossy().into_owned();
    let program_args = vec![
        "--project-plan".to_string(),
        directory.to_string(),
        plan_str.clone(),
    ];
    if let Err(error) = run_compiler_program(compiler, &program_args) {
        report_cli_error(&error);
        let _ = std::fs::remove_file(&plan_path);
        return Err(());
    }
    let plan_text = match std::fs::read_to_string(&plan_path) {
        Ok(text) => text,
        Err(_) => {
            let _ = std::fs::remove_file(&plan_path);
            return Err(());
        }
    };
    let _ = std::fs::remove_file(&plan_path);
    let content = match quoted_value(&plan_text, "content") {
        Some(value) if value.len() == 16 => value,
        _ => return Err(()),
    };
    let hash = match u64::from_str_radix(&content, 16) {
        Ok(hash) => hash,
        Err(_) => return Err(()),
    };
    let locked = extract_dependencies(&plan_text);
    project_finish_build(directory, &project, hash, &locked, compiler, output)
}

/// Run a project's tests, mirroring `lana_project_test` in `tools/c/project.c`.
fn project_test(directory: &str) -> u8 {
    let cmake_path = Path::new(directory).join("CMakeLists.txt");
    if cmake_path.exists() {
        let status = std::process::Command::new("ctest")
            .args(["--test-dir", "build", "--output-on-failure"])
            .current_dir(directory)
            .status();
        return match status {
            Ok(status) if status.success() => 0,
            Ok(status) => status.code().unwrap_or(1) as u8,
            Err(_) => 1,
        };
    }
    let tests_dir = Path::new(directory).join("tests");
    let Ok(entries) = std::fs::read_dir(&tests_dir) else {
        return 1;
    };
    let executable = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("lana"));
    let mut count = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("lana") {
            continue;
        }
        let status = std::process::Command::new(&executable)
            .arg("run")
            .arg(&path)
            .status();
        match status {
            Ok(status) if status.success() => count += 1,
            _ => return 1,
        }
    }
    println!("{count} project tests passed");
    0
}

/// Compile a source and run it, mirroring the `debug` command in `tools/c/cli.c`.
fn debug_command(args: &[String]) -> ExitCode {
    // args[0] == "debug", args[1] == source, optional args[2] == "--break".
    let valid = (args.len() == 2 || args.len() == 4)
        && args[1].ends_with(".lana")
        && (args.len() != 4 || (args[2] == "--break"
            && args[3].parse::<u32>().is_ok_and(|line| line > 0)));
    if !valid {
        usage("lana");
        return ExitCode::from(2);
    }
    let Some(compiler) = find_compiler() else {
        eprintln!("native Lana compiler bytecode not found");
        return ExitCode::from(1);
    };
    let bytecode_path = temp_path("lana-debug");
    let bytecode_str = bytecode_path.to_string_lossy().into_owned();
    if let Err(error) = compile_source_file(&compiler, &args[1], &bytecode_str) {
        let _ = std::fs::remove_file(&bytecode_path);
        report_cli_error(&error);
        return ExitCode::from(1);
    }
    let mut run_args = vec![bytecode_str, "--debug".into()];
    run_args.extend_from_slice(&args[2..]);
    let code = run_command(&run_args);
    let _ = std::fs::remove_file(&bytecode_path);
    code
}

/// Serialize a program's returned state distribution as JSON or Graphviz DOT,
/// mirroring `inspect_command` in `tools/c/cli.c` (LIP-002).
fn inspect_command(args: &[String]) -> ExitCode {
    // args[0] == "inspect", args[1] == path, optional `--format json|dot`.
    if args.len() < 2 {
        usage("lana");
        return ExitCode::from(2);
    }
    let mut format = lana_vm::InspectFormat::Json;
    let mut path: Option<&str> = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--format" => {
                if index + 1 >= args.len() {
                    eprintln!("invalid inspect format");
                    return ExitCode::from(2);
                }
                format = match args[index + 1].as_str() {
                    "json" => lana_vm::InspectFormat::Json,
                    "dot" => lana_vm::InspectFormat::Dot,
                    _ => {
                        eprintln!("invalid inspect format");
                        return ExitCode::from(2);
                    }
                };
                index += 2;
            }
            value if value.starts_with('-') => {
                usage("lana");
                return ExitCode::from(2);
            }
            value => {
                if path.is_some() {
                    usage("lana");
                    return ExitCode::from(2);
                }
                path = Some(value);
                index += 1;
            }
        }
    }
    let Some(path) = path else {
        usage("lana");
        return ExitCode::from(2);
    };
    // A `.lana` source is compiled to a temporary chunk first, mirroring the
    // C CLI's `main()` dispatch.
    let bytecode_path;
    let effective_path;
    if path.ends_with(".lana") {
        let Some(compiler) = find_compiler() else {
            eprintln!("native Lana compiler bytecode not found");
            return ExitCode::from(1);
        };
        bytecode_path = temp_path("lana-inspect");
        let bytecode_str = bytecode_path.to_string_lossy().into_owned();
        if let Err(error) = compile_source_file(&compiler, path, &bytecode_str) {
            let _ = std::fs::remove_file(&bytecode_path);
            report_cli_error(&error);
            return ExitCode::from(1);
        }
        effective_path = bytecode_str;
    } else {
        effective_path = path.to_string();
    }
    let bytes = match std::fs::read(&effective_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("{effective_path}: error[LANA_ERR_IO]: {error}");
            return ExitCode::from(1);
        }
    };
    let chunk = match lana_bytecode::loader::load(&bytes) {
        Ok(chunk) => chunk,
        Err(info) => {
            eprintln!(
                "{effective_path}:{}: error[{}]: {}",
                info.line,
                info.code.name(),
                info.message,
            );
            return ExitCode::from(1);
        }
    };
    let mut vm = lana_vm::Vm::new(&chunk);
    if vm.run() != LanaError::Ok {
        report_error(vm.error());
        return ExitCode::from(1);
    }
    let result_value = vm.result();
    if !matches!(result_value.kind, lana_vm::ValueKind::StateDist(_)) {
        eprintln!(
            "inspect: program did not return a state_dist (got {})",
            result_value.type_name()
        );
        return ExitCode::from(1);
    }
    let dist = match &result_value.kind {
        lana_vm::ValueKind::StateDist(dist) => dist.clone(),
        _ => unreachable!(),
    };
    match lana_vm::inspect(&dist, format) {
        Ok(out) => {
            println!("{out}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("inspect: {}", error.name());
            ExitCode::from(1)
        }
    }
}

/// Language Server Protocol frontend, mirroring `tools/c/lsp.c`. Speaks JSON-RPC
/// over stdin/stdout with `Content-Length` framing and drives the self-hosted
/// compiler (`lana-compiler.labc`) on the Rust VM for diagnostics and symbols.
mod lsp {
    use std::io::{BufRead, Write};
    use std::path::{Path, PathBuf};
    use std::process::ExitCode;

    /// Minimal JSON DOM, mirroring `JsonValue` in `tools/c/json.c`.
    #[derive(Debug)]
    enum JValue {
        Null,
        #[allow(dead_code)]
        Bool(bool),
        Number(f64),
        String(String),
        Array(Vec<JValue>),
        Object(Vec<(String, JValue)>),
    }

    fn jget<'a>(value: &'a JValue, key: &str) -> Option<&'a JValue> {
        match value {
            JValue::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn jstring<'a>(value: Option<&'a JValue>) -> Option<&'a str> {
        match value {
            Some(JValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    fn jnumber(value: Option<&JValue>) -> f64 {
        match value {
            Some(JValue::Number(n)) => *n,
            _ => 0.0,
        }
    }

    fn hex_value(c: u8) -> i32 {
        match c {
            b'0'..=b'9' => (c - b'0') as i32,
            b'a'..=b'f' => (c - b'a' + 10) as i32,
            b'A'..=b'F' => (c - b'A' + 10) as i32,
            _ => -1,
        }
    }

    fn push_codepoint(out: &mut Vec<u8>, codepoint: u32) {
        if codepoint < 0x80 {
            out.push(codepoint as u8);
        } else if codepoint < 0x800 {
            out.push(0xC0 | (codepoint >> 6) as u8);
            out.push(0x80 | (codepoint & 0x3F) as u8);
        } else if codepoint < 0x10000 {
            out.push(0xE0 | (codepoint >> 12) as u8);
            out.push(0x80 | ((codepoint >> 6) & 0x3F) as u8);
            out.push(0x80 | (codepoint & 0x3F) as u8);
        } else {
            out.push(0xF0 | (codepoint >> 18) as u8);
            out.push(0x80 | ((codepoint >> 12) & 0x3F) as u8);
            out.push(0x80 | ((codepoint >> 6) & 0x3F) as u8);
            out.push(0x80 | (codepoint & 0x3F) as u8);
        }
    }

    struct JParser<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> JParser<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            Self { bytes, pos: 0 }
        }

        fn skip_ws(&mut self) {
            while self.pos < self.bytes.len() {
                match self.bytes[self.pos] {
                    b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                    _ => break,
                }
            }
        }

        fn parse_value(&mut self) -> Option<JValue> {
            self.skip_ws();
            let c = *self.bytes.get(self.pos)?;
            match c {
                b'{' => self.parse_object(),
                b'[' => self.parse_array(),
                b'"' => self.parse_string().map(JValue::String),
                b't' => self.parse_literal("true").map(|_| JValue::Bool(true)),
                b'f' => self.parse_literal("false").map(|_| JValue::Bool(false)),
                b'n' => self.parse_literal("null").map(|_| JValue::Null),
                b'-' | b'0'..=b'9' => self.parse_number().map(JValue::Number),
                _ => None,
            }
        }

        fn parse_literal(&mut self, literal: &str) -> Option<()> {
            let lit = literal.as_bytes();
            if self.pos + lit.len() > self.bytes.len() {
                return None;
            }
            if &self.bytes[self.pos..self.pos + lit.len()] == lit {
                self.pos += lit.len();
                Some(())
            } else {
                None
            }
        }

        fn parse_number(&mut self) -> Option<f64> {
            let start = self.pos;
            if self.bytes.get(self.pos) == Some(&b'-') {
                self.pos += 1;
            }
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            if self.pos < self.bytes.len() && self.bytes[self.pos] == b'.' {
                self.pos += 1;
                while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
            }
            if self.pos < self.bytes.len()
                && (self.bytes[self.pos] == b'e' || self.bytes[self.pos] == b'E')
            {
                self.pos += 1;
                if self.pos < self.bytes.len()
                    && (self.bytes[self.pos] == b'+' || self.bytes[self.pos] == b'-')
                {
                    self.pos += 1;
                }
                while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
            }
            let text = std::str::from_utf8(&self.bytes[start..self.pos]).ok()?;
            text.parse::<f64>().ok()
        }

        fn parse_array(&mut self) -> Option<JValue> {
            self.pos += 1; // '['
            let mut items = Vec::new();
            self.skip_ws();
            if self.bytes.get(self.pos) == Some(&b']') {
                self.pos += 1;
                return Some(JValue::Array(items));
            }
            loop {
                items.push(self.parse_value()?);
                self.skip_ws();
                match self.bytes.get(self.pos) {
                    Some(b',') => {
                        self.pos += 1;
                        self.skip_ws();
                    }
                    Some(b']') => {
                        self.pos += 1;
                        return Some(JValue::Array(items));
                    }
                    _ => return None,
                }
            }
        }

        fn parse_object(&mut self) -> Option<JValue> {
            self.pos += 1; // '{'
            let mut entries: Vec<(String, JValue)> = Vec::new();
            self.skip_ws();
            if self.bytes.get(self.pos) == Some(&b'}') {
                self.pos += 1;
                return Some(JValue::Object(entries));
            }
            loop {
                self.skip_ws();
                let key = self.parse_string()?;
                self.skip_ws();
                if self.bytes.get(self.pos) != Some(&b':') {
                    return None;
                }
                self.pos += 1;
                self.skip_ws();
                let value = self.parse_value()?;
                entries.push((key, value));
                self.skip_ws();
                match self.bytes.get(self.pos) {
                    Some(b',') => {
                        self.pos += 1;
                    }
                    Some(b'}') => {
                        self.pos += 1;
                        return Some(JValue::Object(entries));
                    }
                    _ => return None,
                }
            }
        }

        fn parse_string(&mut self) -> Option<String> {
            if self.bytes.get(self.pos) != Some(&b'"') {
                return None;
            }
            self.pos += 1;
            let mut out: Vec<u8> = Vec::new();
            while self.pos < self.bytes.len() {
                match self.bytes[self.pos] {
                    b'"' => {
                        self.pos += 1;
                        return String::from_utf8(out).ok();
                    }
                    b'\\' => {
                        self.pos += 1;
                        let esc = *self.bytes.get(self.pos)?;
                        self.pos += 1;
                        match esc {
                            b'"' => out.push(b'"'),
                            b'\\' => out.push(b'\\'),
                            b'/' => out.push(b'/'),
                            b'b' => out.push(0x08),
                            b'f' => out.push(0x0C),
                            b'n' => out.push(b'\n'),
                            b'r' => out.push(b'\r'),
                            b't' => out.push(b'\t'),
                            b'u' => {
                                let cp = self.parse_hex4()?;
                                if (0xD800..0xDC00).contains(&cp) {
                                    if self.bytes.get(self.pos) == Some(&b'\\')
                                        && self.bytes.get(self.pos + 1) == Some(&b'u')
                                    {
                                        self.pos += 2;
                                        let low = self.parse_hex4()?;
                                        if (0xDC00..0xE000).contains(&low) {
                                            let combined = 0x10000
                                                + ((cp - 0xD800) << 10)
                                                + (low - 0xDC00);
                                            push_codepoint(&mut out, combined);
                                        } else {
                                            return None;
                                        }
                                    } else {
                                        return None;
                                    }
                                } else {
                                    push_codepoint(&mut out, cp);
                                }
                            }
                            _ => return None,
                        }
                    }
                    other => {
                        self.pos += 1;
                        out.push(other);
                    }
                }
            }
            None
        }

        fn parse_hex4(&mut self) -> Option<u32> {
            if self.pos + 4 > self.bytes.len() {
                return None;
            }
            let mut value: u32 = 0;
            for _ in 0..4 {
                let d = hex_value(self.bytes[self.pos]) as u32;
                if d > 15 {
                    return None;
                }
                value = value * 16 + d;
                self.pos += 1;
            }
            Some(value)
        }
    }

    fn parse_json(text: &str) -> Option<JValue> {
        let mut parser = JParser::new(text.as_bytes());
        let value = parser.parse_value()?;
        parser.skip_ws();
        if parser.pos != parser.bytes.len() {
            return None;
        }
        Some(value)
    }

    fn send_message(json: &str) {
        print!("Content-Length: {}\r\n\r\n{}", json.len(), json);
        let _ = std::io::stdout().flush();
    }

    fn send_result(id: &str, result: &str) {
        let message = format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{result}}}");
        send_message(&message);
    }

    fn format_id(id: Option<&JValue>) -> String {
        match id {
            Some(JValue::Number(n)) => {
                if n.fract() == 0.0 {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
            Some(JValue::String(s)) => format!("\"{s}\""),
            _ => "null".to_string(),
        }
    }

    fn uri_to_path(uri: &str) -> String {
        let mut read = uri.as_bytes();
        if read.starts_with(b"file://") {
            read = &read[7..];
        }
        let mut out: Vec<u8> = Vec::new();
        let mut i = 0;
        while i < read.len() {
            if read[i] == b'%' && i + 2 < read.len() {
                let hi = hex_value(read[i + 1]);
                let lo = hex_value(read[i + 2]);
                if hi >= 0 && lo >= 0 {
                    out.push((hi * 16 + lo) as u8);
                    i += 3;
                    continue;
                }
            }
            out.push(read[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn document_uri(params: &JValue) -> Option<&str> {
        let text_doc = jget(params, "textDocument")?;
        jstring(jget(text_doc, "uri"))
    }

    fn document_text(params: &JValue) -> Option<&str> {
        if let Some(text_doc) = jget(params, "textDocument") {
            if let Some(text) = jstring(jget(text_doc, "text")) {
                return Some(text);
            }
        }
        if let Some(JValue::Array(changes)) = jget(params, "contentChanges") {
            if let Some(change) = changes.first() {
                return jstring(jget(change, "text"));
            }
        }
        None
    }

    fn document_position(params: &JValue) -> (i64, i64) {
        let position = jget(params, "position");
        let line = jnumber(position.and_then(|p| jget(p, "line"))) as i64;
        let character = jnumber(position.and_then(|p| jget(p, "character"))) as i64;
        (line, character)
    }

    fn publish_diagnostics(uri: &str, diagnostics: &str) {
        let message = format!(
            "{{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/publishDiagnostics\",\"params\":{{\"uri\":\"{uri}\",\"diagnostics\":{diagnostics}}}}}"
        );
        send_message(&message);
    }

    fn diagnostic_message(message: &str) -> &str {
        let Some(column_pos) = message.find("column ") else {
            return message;
        };
        let rest = &message[column_pos..];
        let Some(colon) = rest.find(':') else {
            return message;
        };
        let after = &rest[colon + 1..];
        if after.starts_with(' ') {
            &after[1..]
        } else {
            after
        }
    }

    fn json_escape_message(text: &str) -> String {
        let mut out = String::new();
        for c in text.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                }
                c => out.push(c),
            }
        }
        out
    }

    fn compiler_error_position(message: &str) -> (u32, u32) {
        let Some(pos) = message.find(" at line ") else {
            return (1, 1);
        };
        let rest = &message[pos + " at line ".len()..];
        let mut parts = rest.splitn(2, " column ");
        let line = parts.next().and_then(|s| s.trim().parse::<u32>().ok());
        let column = parts.next().and_then(|s| {
            s.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<u32>()
                .ok()
        });
        (line.unwrap_or(1), column.unwrap_or(1))
    }

    fn compiler_diagnostic_json(error: &lana_vm::VmError) -> String {
        let (one_line, one_column) = compiler_error_position(&error.message);
        let line = one_line.saturating_sub(1);
        let column = one_column.saturating_sub(1);
        let message = diagnostic_message(&error.message);
        let escaped = json_escape_message(message);
        format!(
            "{{\"range\":{{\"start\":{{\"line\":{line},\"character\":{column}}},\"end\":{{\"line\":{line},\"character\":{}}}}},\"severity\":1,\"source\":\"lana\",\"message\":\"{escaped}\"}}",
            column + 1
        )
    }

    fn unique_suffix() -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("{}-{nanos}", std::process::id())
    }

    fn lsp_write_temp_source(source_text: &str, source_path: &str) -> Option<PathBuf> {
        let path = Path::new(source_path);
        let candidate = match path.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => {
                dir.join(format!(".lana-lsp-{}", unique_suffix()))
            }
            _ => std::env::temp_dir().join(format!("lana-lsp-{}", unique_suffix())),
        };
        std::fs::write(&candidate, source_text).ok()?;
        Some(candidate)
    }

    fn compiler_check(
        compiler: &Path,
        source_text: &str,
        source_path: &str,
    ) -> Option<lana_vm::VmError> {
        let source_temp = lsp_write_temp_source(source_text, source_path)?;
        let asm_temp = super::temp_path("lana-lsp-asm");
        let args = vec![
            source_temp.to_string_lossy().into_owned(),
            asm_temp.to_string_lossy().into_owned(),
        ];
        let result = super::run_compiler_program(compiler, &args);
        let _ = std::fs::remove_file(&source_temp);
        let _ = std::fs::remove_file(&asm_temp);
        match result {
            Err(super::CliError::Run(vm_error)) => Some(vm_error),
            _ => None,
        }
    }

    fn compiler_symbols(
        compiler: &Path,
        source_text: &str,
        source_path: &str,
    ) -> Option<String> {
        let source_temp = lsp_write_temp_source(source_text, source_path)?;
        let output_temp = super::temp_path("lana-lsp-sym");
        let args = vec![
            "--symbols".to_string(),
            source_temp.to_string_lossy().into_owned(),
            output_temp.to_string_lossy().into_owned(),
        ];
        let result = super::run_compiler_program(compiler, &args);
        let _ = std::fs::remove_file(&source_temp);
        if result.is_err() {
            let _ = std::fs::remove_file(&output_temp);
            return None;
        }
        let text = std::fs::read_to_string(&output_temp).ok();
        let _ = std::fs::remove_file(&output_temp);
        text
    }

    fn check_and_publish(compiler: Option<&Path>, uri: Option<&str>, text: Option<&str>) {
        match (compiler, uri, text) {
            (Some(compiler), Some(uri), Some(text)) => {
                let path = uri_to_path(uri);
                match compiler_check(compiler, text, &path) {
                    None => publish_diagnostics(uri, "[]"),
                    Some(error) => {
                        let diagnostic = compiler_diagnostic_json(&error);
                        let array = format!("[{diagnostic}]");
                        publish_diagnostics(uri, &array);
                    }
                }
            }
            _ => publish_diagnostics(uri.unwrap_or(""), "[]"),
        }
    }

    struct LspDocuments {
        entries: Vec<(String, Option<String>)>,
    }

    impl LspDocuments {
        fn new() -> Self {
            Self { entries: Vec::new() }
        }

        fn set(&mut self, uri: &str, text: Option<&str>) {
            for entry in &mut self.entries {
                if entry.0 == uri {
                    entry.1 = text.map(|t| t.to_string());
                    return;
                }
            }
            self.entries.push((uri.to_string(), text.map(|t| t.to_string())));
        }

        fn get(&self, uri: Option<&str>) -> Option<&str> {
            let uri = uri?;
            self.entries
                .iter()
                .find(|entry| entry.0 == uri)
                .and_then(|entry| entry.1.as_deref())
        }

        fn remove(&mut self, uri: Option<&str>) {
            let Some(uri) = uri else { return };
            if let Some(pos) = self.entries.iter().position(|entry| entry.0 == uri) {
                self.entries.swap_remove(pos);
            }
        }
    }

    fn symbol_at<'a>(symbols: &'a JValue, line: i64, character: i64) -> Option<&'a str> {
        let one_line = line + 1;
        let one_column = character + 1;
        if let Some(JValue::Array(references)) = jget(symbols, "references") {
            for reference in references {
                let Some(name) = jstring(jget(reference, "name")) else {
                    continue;
                };
                let ref_line = jnumber(jget(reference, "line")) as i64;
                let ref_column = jnumber(jget(reference, "column")) as i64;
                let length = name.len() as i64;
                if ref_line == one_line
                    && ref_column <= one_column
                    && one_column < ref_column + length
                {
                    return Some(name);
                }
            }
        }
        if let Some(JValue::Array(definitions)) = jget(symbols, "definitions") {
            for definition in definitions {
                let Some(name) = jstring(jget(definition, "name")) else {
                    continue;
                };
                let def_line = jnumber(jget(definition, "line")) as i64;
                let def_column = jnumber(jget(definition, "column")) as i64;
                let length = name.len() as i64;
                if def_line == one_line
                    && def_column <= one_column
                    && one_column < def_column + length
                {
                    return Some(name);
                }
            }
        }
        None
    }

    fn find_definition<'a>(symbols: &'a JValue, name: &str) -> Option<&'a JValue> {
        let definitions = jget(symbols, "definitions")?;
        if let JValue::Array(definitions) = definitions {
            for definition in definitions {
                if jstring(jget(definition, "name")) == Some(name) {
                    return Some(definition);
                }
            }
        }
        None
    }

    fn append_location(uri: &str, entry: &JValue) -> String {
        let name = jstring(jget(entry, "name"));
        let line = jnumber(jget(entry, "line")) as i64 - 1;
        let column = jnumber(jget(entry, "column")) as i64 - 1;
        let length = name.map(|n| n.len() as i64).unwrap_or(1);
        format!(
            "{{\"uri\":\"{uri}\",\"range\":{{\"start\":{{\"line\":{line},\"character\":{column}}},\"end\":{{\"line\":{line},\"character\":{}}}}}}}",
            column + length
        )
    }

    fn format_references(uri: &str, symbols: &JValue, name: &str) -> String {
        let mut out = String::from("[");
        let mut first = true;
        if let Some(JValue::Array(references)) = jget(symbols, "references") {
            for reference in references {
                if jstring(jget(reference, "name")) != Some(name) {
                    continue;
                }
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(&append_location(uri, reference));
            }
        }
        out.push(']');
        out
    }

    fn format_completion(symbols: &JValue) -> String {
        let mut out = String::from("{\"isIncomplete\":false,\"items\":[");
        let mut first = true;
        if let Some(JValue::Array(definitions)) = jget(symbols, "definitions") {
            for definition in definitions {
                let Some(name) = jstring(jget(definition, "name")) else {
                    continue;
                };
                let kind = jstring(jget(definition, "kind"));
                let ty = jstring(jget(definition, "type"));
                let item_kind = if kind == Some("function") { 3 } else { 6 };
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(&format!(
                    "{{\"label\":\"{name}\",\"kind\":{item_kind},\"detail\":\"{}\"}}",
                    ty.unwrap_or("unknown")
                ));
            }
        }
        out.push_str("]}");
        out
    }

    fn append_edit(entry: &JValue, new_name: &str) -> String {
        let name = jstring(jget(entry, "name"));
        let line = jnumber(jget(entry, "line")) as i64 - 1;
        let column = jnumber(jget(entry, "column")) as i64 - 1;
        let length = name.map(|n| n.len() as i64).unwrap_or(1);
        format!(
            "{{\"range\":{{\"start\":{{\"line\":{line},\"character\":{column}}},\"end\":{{\"line\":{line},\"character\":{}}}}},\"newText\":\"{new_name}\"}}",
            column + length
        )
    }

    fn format_rename(uri: &str, symbols: &JValue, name: &str, new_name: &str) -> String {
        let mut out = format!("{{\"changes\":{{\"{uri}\":[");
        let mut first = true;
        if let Some(definition) = find_definition(symbols, name) {
            out.push_str(&append_edit(definition, new_name));
            first = false;
        }
        if let Some(JValue::Array(references)) = jget(symbols, "references") {
            for reference in references {
                if jstring(jget(reference, "name")) != Some(name) {
                    continue;
                }
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(&append_edit(reference, new_name));
            }
        }
        out.push_str("]}}");
        out
    }

    fn run_symbols(
        compiler: Option<&Path>,
        uri: Option<&str>,
        text: Option<&str>,
    ) -> Option<JValue> {
        let compiler = compiler?;
        let uri = uri?;
        let text = text?;
        let path = uri_to_path(uri);
        let json_text = compiler_symbols(compiler, text, &path)?;
        parse_json(&json_text)
    }

    fn handle_hover(id: &str, compiler: Option<&Path>, documents: &LspDocuments, params: &JValue) {
        let uri = document_uri(params);
        let text = documents.get(uri);
        let (line, character) = document_position(params);
        let Some(symbols) = run_symbols(compiler, uri, text) else {
            send_result(id, "null");
            return;
        };
        let Some(name) = symbol_at(&symbols, line, character) else {
            send_result(id, "null");
            return;
        };
        let definition = find_definition(&symbols, name);
        let ty = definition.and_then(|d| jstring(jget(d, "type")));
        let kind = definition.and_then(|d| jstring(jget(d, "kind")));
        let result = format!(
            "{{\"contents\":{{\"kind\":\"markdown\",\"value\":\"`{name}`: {} ({})\"}}}}",
            ty.unwrap_or("unknown"),
            kind.unwrap_or("variable"),
        );
        send_result(id, &result);
    }

    fn handle_definition(
        id: &str,
        compiler: Option<&Path>,
        documents: &LspDocuments,
        params: &JValue,
    ) {
        let uri = document_uri(params);
        let text = documents.get(uri);
        let (line, character) = document_position(params);
        let Some(symbols) = run_symbols(compiler, uri, text) else {
            send_result(id, "[]");
            return;
        };
        let Some(name) = symbol_at(&symbols, line, character) else {
            send_result(id, "[]");
            return;
        };
        let Some(definition) = find_definition(&symbols, name) else {
            send_result(id, "[]");
            return;
        };
        let location = append_location(uri.unwrap_or(""), definition);
        let result = format!("[{location}]");
        send_result(id, &result);
    }

    fn handle_references(
        id: &str,
        compiler: Option<&Path>,
        documents: &LspDocuments,
        params: &JValue,
    ) {
        let uri = document_uri(params);
        let text = documents.get(uri);
        let (line, character) = document_position(params);
        let Some(symbols) = run_symbols(compiler, uri, text) else {
            send_result(id, "[]");
            return;
        };
        let Some(name) = symbol_at(&symbols, line, character) else {
            send_result(id, "[]");
            return;
        };
        let result = format_references(uri.unwrap_or(""), &symbols, name);
        send_result(id, &result);
    }

    fn handle_completion(
        id: &str,
        compiler: Option<&Path>,
        documents: &LspDocuments,
        params: &JValue,
    ) {
        let uri = document_uri(params);
        let text = documents.get(uri);
        let Some(symbols) = run_symbols(compiler, uri, text) else {
            send_result(id, "{\"isIncomplete\":false,\"items\":[]}");
            return;
        };
        let result = format_completion(&symbols);
        send_result(id, &result);
    }

    fn handle_rename(
        id: &str,
        compiler: Option<&Path>,
        documents: &LspDocuments,
        params: &JValue,
    ) {
        let uri = document_uri(params);
        let text = documents.get(uri);
        let new_name = jstring(jget(params, "newName"));
        let (line, character) = document_position(params);
        let symbols = run_symbols(compiler, uri, text);
        let (Some(symbols), Some(new_name)) = (symbols, new_name) else {
            send_result(id, "{\"changes\":{}}");
            return;
        };
        let Some(name) = symbol_at(&symbols, line, character) else {
            send_result(id, "{\"changes\":{}}");
            return;
        };
        let result = format_rename(uri.unwrap_or(""), &symbols, name, new_name);
        send_result(id, &result);
    }

    fn read_message(reader: &mut impl BufRead) -> Option<String> {
        let mut length: usize = 0;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).ok()? == 0 {
                return None;
            }
            if let Some(value) = line.strip_prefix("Content-Length:") {
                if let Ok(parsed) = value.trim().parse::<usize>() {
                    length = parsed;
                }
                continue;
            }
            if line == "\r\n" || line == "\n" {
                break;
            }
        }
        if length == 0 {
            return None;
        }
        let mut body = vec![0u8; length];
        reader.read_exact(&mut body).ok()?;
        Some(String::from_utf8_lossy(&body).into_owned())
    }

    pub fn run() -> ExitCode {
        let compiler = super::find_compiler();
        let compiler_ref = compiler.as_deref();
        let mut documents = LspDocuments::new();
        let mut shutdown_requested = false;
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        loop {
            let Some(body) = read_message(&mut reader) else {
                return ExitCode::SUCCESS;
            };
            let Some(request) = parse_json(&body) else {
                continue;
            };
            let Some(method) = jstring(jget(&request, "method")).map(|s| s.to_string()) else {
                continue;
            };
            let id = format_id(jget(&request, "id"));
            let params = jget(&request, "params");
            match method.as_str() {
                "exit" => {
                    return if shutdown_requested {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    };
                }
                "textDocument/didOpen" | "textDocument/didChange" => {
                    let uri = params.and_then(document_uri);
                    let text = params.and_then(document_text);
                    if let Some(uri) = uri {
                        documents.set(uri, text);
                    }
                    check_and_publish(compiler_ref, uri, text);
                }
                "textDocument/didClose" => {
                    let uri = params.and_then(document_uri);
                    documents.remove(uri);
                    publish_diagnostics(uri.unwrap_or(""), "[]");
                }
                "initialize" => {
                    send_result(&id, &format!("{{\"serverInfo\":{{\"name\":\"lana-lsp\",\"version\":\"{}\"}},\"capabilities\":{{\"textDocumentSync\":1,\"hoverProvider\":true,\"completionProvider\":{{}},\"definitionProvider\":true,\"referencesProvider\":true,\"renameProvider\":{{\"prepareProvider\":true}}}}}}", super::lana_version()));
                }
                "shutdown" => {
                    shutdown_requested = true;
                    send_result(&id, "null");
                }
                "textDocument/hover" => {
                    handle_hover(&id, compiler_ref, &documents, params.unwrap_or(&JValue::Null));
                }
                "textDocument/completion" => {
                    handle_completion(
                        &id,
                        compiler_ref,
                        &documents,
                        params.unwrap_or(&JValue::Null),
                    );
                }
                "textDocument/definition" => {
                    handle_definition(
                        &id,
                        compiler_ref,
                        &documents,
                        params.unwrap_or(&JValue::Null),
                    );
                }
                "textDocument/references" => {
                    handle_references(
                        &id,
                        compiler_ref,
                        &documents,
                        params.unwrap_or(&JValue::Null),
                    );
                }
                "textDocument/rename" => {
                    handle_rename(&id, compiler_ref, &documents, params.unwrap_or(&JValue::Null));
                }
                _ => {
                    send_result(&id, "[]");
                }
            }
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        // Bare `lana` (no subcommand) launches the REPL (LIP-020).
        let Some(compiler) = find_compiler() else {
            eprintln!("native Lana compiler bytecode not found");
            return ExitCode::from(1);
        };
        return repl::run_repl(&compiler);
    }
    match args[1].as_str() {
        "repl" => {
            if args.len() != 2 {
                usage("lana");
                return ExitCode::from(2);
            }
            let Some(compiler) = find_compiler() else {
                eprintln!("native Lana compiler bytecode not found");
                return ExitCode::from(1);
            };
            repl::run_repl(&compiler)
        }
        "version" => {
            println!("Lana {} (LABC v2, Rust VM, native compiler)", lana_version());
            ExitCode::SUCCESS
        }
        "new" => {
            if args.len() != 3 {
                usage("lana");
                return ExitCode::from(2);
            }
            let Some(compiler) = find_compiler() else {
                eprintln!("native Lana compiler bytecode not found");
                return ExitCode::from(1);
            };
            let program_args = vec!["--project-new".to_string(), args[2].clone()];
            match run_compiler_program(&compiler, &program_args) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    report_cli_error(&error);
                    ExitCode::from(1)
                }
            }
        }
        "lsp" => {
            if args.len() != 2 {
                usage("lana");
                return ExitCode::from(2);
            }
            lsp::run()
        }
        "fmt" | "doc" => {
            let is_fmt = args[1] == "fmt";
            if is_fmt {
                if args.len() != 2 && !(args.len() == 3 && args[2] == "--check") {
                    usage("lana");
                    return ExitCode::from(2);
                }
            } else if args.len() != 2 {
                usage("lana");
                return ExitCode::from(2);
            }
            let mode = if is_fmt { "--project-fmt" } else { "--project-doc" };
            let argument = if is_fmt {
                if args.len() == 3 { "check" } else { "write" }
            } else {
                "."
            };
            run_project_tool(mode, argument)
        }
        "build" => {
            if args.len() > 3 {
                usage("lana");
                return ExitCode::from(2);
            }
            let Some(compiler) = find_compiler() else {
                eprintln!("native Lana compiler bytecode not found");
                return ExitCode::from(1);
            };
            let mut output = String::new();
            match project_build_with_plan(args.get(2).map_or(".", String::as_str), &compiler, &mut output) {
                Ok(()) => ExitCode::SUCCESS,
                Err(()) => ExitCode::from(1),
            }
        }
        "test" => {
            if args.len() > 3 {
                usage("lana");
                return ExitCode::from(2);
            }
            ExitCode::from(project_test(args.get(2).map_or(".", String::as_str)))
        }
        "compile" => {
            if args.len() != 5 || args[3] != "-o" {
                usage("lana");
                return ExitCode::from(2);
            }
            let Some(compiler) = find_compiler() else {
                eprintln!("native Lana compiler bytecode not found");
                return ExitCode::from(1);
            };
            match compile_source_file(&compiler, &args[2], &args[4]) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    report_cli_error(&error);
                    ExitCode::from(1)
                }
            }
        }
        "check" => {
            let Some(compiler) = find_compiler() else {
                eprintln!("native Lana compiler bytecode not found");
                return ExitCode::from(1);
            };
            if args.len() == 2 || (args.len() == 3 && Path::new(&args[2]).is_dir()) {
                let mut output = String::new();
                match project_build_with_plan(args.get(2).map_or(".", String::as_str), &compiler, &mut output) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(()) => ExitCode::from(1),
                }
            } else if args.len() == 3 {
                let bytecode_path = temp_path("lana-check");
                let bytecode_str = bytecode_path.to_string_lossy().into_owned();
                let result = compile_source_file(&compiler, &args[2], &bytecode_str);
                let _ = std::fs::remove_file(&bytecode_path);
                match result {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(error) => {
                        report_cli_error(&error);
                        ExitCode::from(1)
                    }
                }
            } else {
                usage("lana");
                ExitCode::from(2)
            }
        }
        "asm" => assemble_command(&args[2..]),
        "debug" => debug_command(&args[1..]),
        "run" => {
            if args.len() == 2 || Path::new(&args[2]).is_dir() {
                let Some(compiler) = find_compiler() else {
                    eprintln!("native Lana compiler bytecode not found");
                    return ExitCode::from(1);
                };
                let mut output = String::new();
                if project_build_with_plan(args.get(2).map_or(".", String::as_str), &compiler, &mut output).is_err() {
                    return ExitCode::from(1);
                }
                let mut run_args = vec![output];
                if args.len() > 3 { run_args.extend_from_slice(&args[3..]); }
                run_command(&run_args)
            } else if args.len() >= 3 && args[2].ends_with(".lana") {
                let Some(compiler) = find_compiler() else {
                    eprintln!("native Lana compiler bytecode not found");
                    return ExitCode::from(1);
                };
                let bytecode_path = temp_path("lana-program");
                let bytecode_str = bytecode_path.to_string_lossy().into_owned();
                if let Err(error) = compile_source_file(&compiler, &args[2], &bytecode_str) {
                    let _ = std::fs::remove_file(&bytecode_path);
                    report_cli_error(&error);
                    return ExitCode::from(1);
                }
                let mut run_args = vec![bytecode_str];
                run_args.extend_from_slice(&args[3..]);
                let code = run_command(&run_args);
                let _ = std::fs::remove_file(&bytecode_path);
                code
            } else {
                run_command(&args[2..])
            }
        }
        "run-bytecode" => run_command(&args[2..]),
        "dis" => match disassemble_command(&args[2..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(code) => ExitCode::from(code),
        },
        "verify" => match disassemble_command(&args[2..]) {
            Ok(()) => {
                println!("verified");
                ExitCode::SUCCESS
            }
            Err(code) => ExitCode::from(code),
        },
        "inspect" => inspect_command(&args[1..]),
        _ => {
            usage("lana");
            ExitCode::from(2)
        }
    }
}
