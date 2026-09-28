//! Command-line driver for the Rust Lana runtime.
//!
//! `lana run <file.labc> [--seed N] [--stats]` loads and verifies a chunk,
//! runs it on the Rust VM, and reports the result.
//!
//! The full command surface includes `version`, `new`,
//! `lsp`, `fmt`, `doc`, `build`, `test`, `compile`, `asm`, `debug`,
//! `run`, `run-bytecode`, `dis`, and `verify`. Commands that need the
//! self-hosted compiler locate `lana-compiler.labc` and run it on the Rust VM.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::io::{Read, Write};
#[cfg(unix)]
use std::sync::atomic::{AtomicBool, Ordering};

use lana_bytecode::{Chunk, LanaError, LanaErrorInfo, OpCode};
use lana_runtime::brain::{Activation, Brain};
use lana_runtime::execution::ExecutionConfig;
use lana_vm::{Vm, ValueKind as RuntimeValueKind};
use serde::Deserialize;

mod bridge_worker;
mod packages;
mod lsp;

const LANA_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(unix)]
static BRAIN_FIT_CANCELLED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn brain_fit_cancel_signal(_: libc::c_int) {
    BRAIN_FIT_CANCELLED.store(true, Ordering::Relaxed);
}

fn brain_fit_check_cancelled() -> Result<(), FitError> {
    #[cfg(unix)]
    if BRAIN_FIT_CANCELLED.load(Ordering::Relaxed) { return Err("LANA_ERR_CANCELLED".into()); }
    Ok(())
}

fn brain_fit_install_cancel_handler() -> Result<(), FitError> {
    #[cfg(unix)] {
        BRAIN_FIT_CANCELLED.store(false, Ordering::Relaxed);
        unsafe {
            let handler = brain_fit_cancel_signal as *const () as libc::sighandler_t;
            if libc::signal(libc::SIGINT, handler) == libc::SIG_ERR
                || libc::signal(libc::SIGTERM, handler) == libc::SIG_ERR {
                return Err("LANA_ERR_IO".into());
            }
        }
    }
    Ok(())
}

/// Full usage text for the single `lana` binary.
fn usage(program: &str) {
    eprintln!(
        "usage:\n  {program} compile program.lana -o program.labc\n  {program} new directory\n  {program} package pack DIRECTORY -o ARCHIVE | package add owner/repo@X.Y.Z\n  {program} brain new|train|fit|evaluate|save|load|inspect|remember|recall|memory|alias|forecast|index|compress|chat\n  {program} lsp\n  {program} debug program.lana\n  {program} build|run|test|fmt|doc\n  {program} asm program.lasm -o program.labc\n  {program} run program.labc [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--instruction-limit N]\n  {program} run-bytecode program.labc [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--instruction-limit N]\n  {program} dis program.labc\n  {program} verify program.labc\n  {program} inspect program.lana [--format json|dot]\n  {program} execution-config init --metadata PATH --key PATH --capability-id ID --origin HTTPS_ORIGIN --credential-key-id ID [--ca-file PATH]"
    );
}

/// Run-specific usage, kept byte-identical to the original `run` command.
fn run_usage(program: &str) {
    eprintln!(
        "usage: {program} run <file.labc> [--seed N] [--workers N] [--max-tasks N] [--memory-limit-mib N] [--instruction-limit N] [--stats]"
    );
}

fn execution_config_command(args: &[String]) -> ExitCode {
    if args.first().map(String::as_str) != Some("init") { usage("lana"); return ExitCode::from(2); }
    let mut metadata = None;
    let mut key = None;
    let mut capability_id = None;
    let mut origin = None;
    let mut credential_key_id = None;
    let mut ca_file = None;
    let mut index = 1;
    while index < args.len() {
        let Some(value) = args.get(index + 1) else { usage("lana"); return ExitCode::from(2); };
        let slot = match args[index].as_str() {
            "--metadata" => &mut metadata,
            "--key" => &mut key,
            "--capability-id" => &mut capability_id,
            "--origin" => &mut origin,
            "--credential-key-id" => &mut credential_key_id,
            "--ca-file" => &mut ca_file,
            _ => { usage("lana"); return ExitCode::from(2); }
        };
        if slot.replace(value.as_str()).is_some() { usage("lana"); return ExitCode::from(2); }
        index += 2;
    }
    let (Some(metadata), Some(key), Some(capability_id), Some(origin), Some(credential_key_id)) =
        (metadata, key, capability_id, origin, credential_key_id) else { usage("lana"); return ExitCode::from(2); };
    match ExecutionConfig::write(Path::new(metadata), Path::new(key), capability_id, origin, credential_key_id, ca_file) {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => { eprintln!("execution-config: initialization failed"); ExitCode::from(1) }
    }
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
    if let Some(durability) = &error.durability {
        eprintln!("  durability: {durability}");
    }
    if let Some(path) = &error.path {
        eprintln!("  path: {path}");
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
    /// A compiler failure, attributed to the source file rather than compiler bytecode.
    Compile { path: String, error: lana_vm::VmError },
    /// The compiler emitted assembly that failed to assemble.
    Assemble { path: String, info: LanaErrorInfo },
    /// Failed to write a chunk to disk.
    Write { path: String, info: LanaErrorInfo, durability_uncertain: bool },
    /// A project-level failure (missing manifest, bad plan, I/O).
    Project,
}

fn report_cli_error(error: &CliError) {
    match error {
        CliError::Load { path, info } => {
            eprintln!("{path}:{}: error[{}]: {}", info.line, info.code.name(), info.message);
        }
        CliError::Run(vm_error) => report_error(vm_error),
        CliError::Compile { path, error } => {
            let mut line = 1u32;
            let mut column = 1u32;
            if let Some(rest) = error.message.strip_prefix("parse error at line ").or_else(|| error.message.strip_prefix("type error at line ")) {
                if let Some((line_text, column_text)) = rest.split_once(" column ") {
                    line = line_text.parse().unwrap_or(1);
                    column = column_text.split(':').next().unwrap_or("1").parse().unwrap_or(1);
                }
            }
            let kind = if error.message.starts_with("parse error") { "parse/LANA_ERR_PARSE" }
                else if error.message.starts_with("type error") { "validation/LANA_ERR_TYPE" }
                else if error.code == LanaError::Assertion { "assertion/LANA_ERR_ASSERTION" }
                else { return report_error(error); };
            eprintln!("{path}:{line}:{column}-{line}:{column}: error[{kind}]: {}", error.message);
        }
        CliError::Assemble { path, info } => {
            eprintln!(
                "{path}:{}:1-{}:1: error[{}]: {} (instruction {}, opcode {})",
                info.line, info.line,
                info.code.name(),
                info.message,
                info.ip,
                OpCode::try_from(info.opcode).map(|op| op.name()).unwrap_or("UNKNOWN"),
            );
        }
        CliError::Write { path, info, durability_uncertain } => {
            eprintln!("{path}: error[{}]: {}", info.code.name(), info.message);
            if *durability_uncertain {
                eprintln!("  durability: uncertain");
                eprintln!("  path: {path}");
            }
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
    let built = PathBuf::from(env!("LANA_BUILT_COMPILER"));
    // A plain Cargo development binary may use its own build-script output.
    // Installed binaries must find their packaged compiler instead.
    let development = std::env::current_exe().ok().is_some_and(|exe|
        exe.parent() == built.ancestors().nth(4));
    (development && built.is_file()).then_some(built)
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
    vm.set_program_args(args);
    let input = if args.first().is_some_and(|arg| arg == "--symbols") { args.get(3).or_else(|| args.get(1)) } else if args.first().is_some_and(|arg| arg.starts_with("--")) { args.get(1) } else { args.first() };
    if let Some(input) = input {
        let source = Path::new(input);
        let anchor = if source.exists() { Some(source) } else { source.parent().filter(|parent| parent.exists()) };
        if let Some(anchor) = anchor {
        let paths = packages::compiler_paths(anchor).map_err(|message| CliError::Load {
            path:input.clone(), info:LanaErrorInfo::new(LanaError::Schema, 0, 0, 0, &message),
        })?;
        vm.set_package_paths(paths);
        }
    }
    let result = vm.run();
    if result != LanaError::Ok {
        return Err(CliError::Run(vm.error().clone()));
    }
    Ok(())
}

/// Serialize a chunk to the LABC v2 on-disk format, mirroring
/// `lana_chunk_write_file` in `vm/c/bytecode.c` (the inverse of
/// `lana_bytecode::loader::load`).
fn write_chunk(chunk: &Chunk, path: &str) -> Result<(), (LanaErrorInfo, bool)> {
    let out = lana_bytecode::encoder::encode(chunk);
    lana_runtime::atomic_file::write(Path::new(path), &out)
        .map_err(|error| (LanaErrorInfo::new(LanaError::Io, 0, 0, 0, &format!("cannot write output file: {error}")),
            lana_runtime::atomic_file::durability_uncertain(&error)))
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
        return Err(match error {
            CliError::Run(error) => CliError::Compile { path: source_path.to_string(), error },
            other => other,
        });
    }
    let asm_text = match std::fs::read_to_string(&asm_path) {
        Ok(text) => text,
        Err(_) => {
            let _ = std::fs::remove_file(&asm_path);
            return Err(CliError::Project);
        }
    };
    let _ = std::fs::remove_file(&asm_path);
    let mut chunk = lana_bytecode::assemble(&asm_text)
        .map_err(|info| CliError::Assemble { path: source_path.to_string(), info })?;
    for (ip, pair) in chunk.code.windows(2).enumerate() {
        if pair[0].opcode == OpCode::PossibilityBuild
            && pair[1].opcode == OpCode::InfoSample
            && pair[0].b == pair[1].a
        {
            return Err(CliError::Assemble {
                path: source_path.to_string(),
                info: LanaErrorInfo::new(LanaError::UnsupportedOperation, ip + 1, pair[1].opcode as u8, pair[1].line,
                    "possibility has no weights; use distribution(...) before sample"),
            });
        }
    }
    if chunk.code.iter().any(|ins| ins.opcode == OpCode::InfoSample) {
        chunk.version = lana_bytecode::opcode::LABC_VERSION_5;
    }
    write_chunk(&chunk, output_path)
        .map_err(|(info, durability_uncertain)| CliError::Write { path: output_path.to_string(), info, durability_uncertain })
}

fn run_command(args: &[String]) -> ExitCode {
    let mut seed: u64 = 0x4c414e41;
    let mut workers: Option<usize> = None;
    let mut max_tasks: Option<usize> = None;
    let mut memory_limit: Option<usize> = None;
    let mut instruction_limit: Option<u64> = None;
    let mut stats = false;
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
                let parsed: usize = match args[index + 1].parse() {
                    Ok(value) if value > 0 => value,
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
    if let Some(mib) = memory_limit {
        if let Err(error) = vm.set_memory_limit(mib * 1024 * 1024) {
            eprintln!("run: {}", error.name());
            return ExitCode::from(1);
        }
    }
    if let Some(limit) = instruction_limit {
        vm.set_instruction_limit(limit);
    }
    vm.set_program_args(&program_args);
    let mut store_host = lana_runtime::host_calls::StoreHost::new();
    store_host.set_chunk_bytes(bytes);
    vm.set_host_call_extension(Box::new(move |vm, host_id, args, out| {
        store_host.dispatch(vm, host_id, args, out)
    }));
    let result = vm.run();
    if result != LanaError::Ok {
        report_error(vm.error());
        return ExitCode::from(1);
    }
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
                let parsed: usize = match args[index + 1].parse() {
                    Ok(value) if value > 0 => value,
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
    if let Err((info, durability_uncertain)) = write_chunk(&chunk, output) {
        report_cli_error(&CliError::Write { path: output.clone(), info, durability_uncertain });
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
    hosted: bool,
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
    let bytes = std::fs::read(&cache_path).map_err(|_| ())?;
    lana_bytecode::loader::load(&bytes).map_err(|_| ())?;
    if lana_runtime::atomic_file::write(&output_path, &bytes).is_err() {
        return Err(());
    }
    let lock_path = Path::new(directory).join("lana.lock");
    let lock = format!(
        "schema = 1\nproject = \"{}\"\ncontent = \"{hash:016x}\"\n{locked}",
        project.name
    );
    if !hosted &&
        lana_runtime::atomic_file::write(&lock_path, lock.as_bytes()).is_err() {
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
    let (hash, hosted) = packages::build_hash(Path::new(directory), hash).map_err(|error| eprintln!("{error}"))?;
    project_finish_build(directory, &project, hash, &locked, hosted, compiler, output)
}

/// Run a project's tests, mirroring `lana_project_test` in `tools/c/project.c`.
fn project_test(directory: &str) -> u8 {
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

/// Compile a source and interactively step it, mirroring `tools/c/cli.c`.
fn debug_command(args: &[String]) -> ExitCode {
    // args[0] == "debug", args[1] == source, optional args[2] == "--break".
    let valid = (args.len() == 2 || args.len() == 4)
        && args[1].ends_with(".lana")
        && (args.len() != 4 || args[2] == "--break");
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
    let breakpoint = if args.len() == 4 { args[3].parse::<u32>().ok() } else { None };
    let code = match std::fs::read(&bytecode_path).ok().and_then(|bytes| lana_bytecode::loader::load(&bytes).ok()) {
        Some(chunk) => {
            let mut vm = Vm::new(&chunk);
            if let Some(line) = breakpoint {
                vm.set_breakpoint_line(line);
                let result = vm.run();
                if result != LanaError::Ok { report_error(vm.error()); return ExitCode::from(1); }
            }
            loop {
                let Some((instruction, line, function, frames)) = vm.debug_location() else { break ExitCode::SUCCESS; };
                println!("BREAK line={line} instruction={} function={function} frames={frames}", instruction + 1);
                print!("debug [s]tep [c]ontinue [q]uit> ");
                if std::io::stdout().flush().is_err() { break ExitCode::from(1); }
                let mut command = String::new();
                if std::io::stdin().read_line(&mut command).is_err() || command.starts_with('q') { break ExitCode::from(1); }
                let result = if command.starts_with('s') { vm.debug_step() } else { vm.debug_continue() };
                if result != LanaError::Ok { report_error(vm.error()); break ExitCode::from(1); }
                if command.starts_with('c') { break ExitCode::SUCCESS; }
            }
        }
        None => ExitCode::from(1),
    };
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
    let result_value = match vm.result() {
        Ok(value) => value,
        Err(error) => { eprintln!("inspect: {}", error.name()); return ExitCode::from(1); }
    };
    if result_value.value_type() != lana_bytecode::ValueType::StateDist {
        eprintln!(
            "inspect: program did not return a state_dist (got {})",
            result_value.type_name()
        );
        return ExitCode::from(1);
    }
    match result_value.inspect_state_dist(format) {
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

fn lsp_map(value: &lana_vm::Value) -> Option<std::sync::Arc<std::sync::Mutex<lana_vm::value::Map>>> {
    if let RuntimeValueKind::Map(map) = &value.kind { Some(map.clone()) } else { None }
}

fn lsp_member(value: &lana_vm::Value, key: &str) -> Option<lana_vm::Value> {
    lsp_map(value)?.lock().ok()?.get(key).cloned()
}

fn lsp_string(value: &lana_vm::Value, key: &str) -> Option<String> {
    match lsp_member(value, key)?.kind { RuntimeValueKind::String(text) => Some(text.to_string()), _ => None }
}

fn lsp_number(value: &lana_vm::Value, key: &str) -> Option<usize> {
    match lsp_member(value, key)?.kind { RuntimeValueKind::Number(number) if number >= 0.0 => Some(number as usize), _ => None }
}

fn json_quote(text: &str) -> String {
    let mut quoted = String::from("\"");
    for character in text.chars() {
        match character { '\"' => quoted.push_str("\\\""), '\\' => quoted.push_str("\\\\"), '\n' => quoted.push_str("\\n"), '\r' => quoted.push_str("\\r"), '\t' => quoted.push_str("\\t"), c if c.is_control() => quoted.push_str(&format!("\\u{:04x}", c as u32)), c => quoted.push(c) }
    }
    quoted.push('\"');
    quoted
}

fn bridge_response(output: std::process::Output) -> Result<lana_vm::Value, String> {
    if !output.status.success() { return Err(String::from_utf8_lossy(&output.stderr).into_owned()); }
    let text = std::str::from_utf8(&output.stdout).map_err(|_| "bridge returned invalid UTF-8".to_owned())?;
    let value = lana_runtime::json_parse(text).map_err(|_| "bridge returned malformed JSON".to_owned())?;
    if lsp_string(&value, "status").as_deref() != Some("ok") { return Err("bridge did not return ok status".to_owned()); }
    Ok(value)
}

fn bridge_tokens(bridge: &str, tokenizer: &str, text: &str) -> Result<Vec<usize>, String> {
    let output = std::process::Command::new(bridge).args(["tokenize", tokenizer, text]).output().map_err(|_| "cannot start LANA_HF bridge".to_owned())?;
    let value = bridge_response(output)?;
    let Some(lana_vm::Value { kind: lana_vm::value::ValueKind::Array(ids), .. }) = lsp_member(&value, "token_ids") else { return Err("bridge returned no token IDs".to_owned()); };
    let ids = ids.lock().unwrap();
    ids.items().iter().map(|value| match value.kind {
        lana_vm::value::ValueKind::Number(id) if id.is_finite() && id >= 0.0 && id.fract() == 0.0 && id < usize::MAX as f64 && id <= 9_007_199_254_740_991.0 => Ok(id as usize),
        _ => Err("bridge returned invalid token ID".to_owned()),
    }).collect()
}

fn bridge_text(bridge: &str, tokenizer: &str, token: usize) -> Result<String, String> {
    let output = std::process::Command::new(bridge).args(["detokenize", tokenizer, &format!("[{token}]")]).output().map_err(|_| "cannot start LANA_HF bridge".to_owned())?;
    lsp_string(&bridge_response(output)?, "text").ok_or_else(|| "bridge returned no text".to_owned())
}

fn bridge_package(args: &[&str]) -> Result<(), String> {
    let bridge = std::env::var("LANA_HF").map_err(|_| "set LANA_HF to the installed local bridge".to_owned())?;
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let output = std::process::Command::new(bridge).args(args).env("LANA_CLI", executable)
        .output().map_err(|_| "cannot start LANA_HF bridge".to_owned())?;
    bridge_response(output).map(|_| ())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FitExample {
    tokens: Vec<usize>,
    target: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BrainArchitecture {
    hidden: Vec<ArchitectureLayer>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchitectureLayer {
    width: usize,
    activation: ArchitectureActivation,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ArchitectureActivation { Relu, Gelu }

fn fit_examples(path: &Path, vocabulary: usize) -> Result<Vec<FitExample>, String> {
    let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    if file.metadata().map_err(|error| error.to_string())?.len() > 64 * 1024 * 1024 {
        return Err("fit input exceeds 64 MiB".into());
    }
    let mut bytes = Vec::new();
    Read::by_ref(&mut file).take(64 * 1024 * 1024 + 1).read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > 64 * 1024 * 1024 { return Err("fit input exceeds 64 MiB".into()); }
    let input = std::str::from_utf8(&bytes).map_err(|error| error.to_string())?;
    let mut examples = Vec::new();
    for line in input.lines() {
        #[cfg(unix)]
        if BRAIN_FIT_CANCELLED.load(Ordering::Relaxed) { return Err("LANA_ERR_CANCELLED".into()); }
        if line.trim().is_empty() { return Err("blank fit record".into()); }
        let example: FitExample = serde_json::from_str(&line).map_err(|error| error.to_string())?;
        if !(1..=4096).contains(&example.tokens.len()) || example.target >= vocabulary ||
            example.tokens.iter().any(|&token| token >= vocabulary) {
            return Err("fit token or target out of range".into());
        }
        examples.push(example);
        if examples.len() > 100_000 { return Err("too many fit records".into()); }
    }
    if examples.is_empty() { return Err("empty fit input".into()); }
    Ok(examples)
}

struct FitError {
    message: String,
    durability_uncertain: bool,
    path: Option<String>,
}

impl From<String> for FitError {
    fn from(message: String) -> Self { Self { message, durability_uncertain: false, path: None } }
}

impl From<&str> for FitError {
    fn from(message: &str) -> Self { message.to_owned().into() }
}

fn brain_save(brain: &Brain, path: &Path) -> Result<(), FitError> {
    brain.save_with_status(path).map_err(|error| FitError {
        message: error.code.name().to_owned(),
        durability_uncertain: error.durability_uncertain,
        path: Some(path.to_string_lossy().into_owned()),
    })
}

fn brain_failure(error: FitError) -> ExitCode {
    let mut report = serde_json::json!({"status":"error","error":error.message});
    if error.durability_uncertain {
        report["durability"] = "uncertain".into();
        report["path"] = error.path.into();
    }
    eprintln!("brain: {}\n{report}", error.message);
    ExitCode::from(1)
}

fn brain_fit(args: &[String]) -> Result<String, FitError> {
    if args.len() < 10 || (args.len() - 4) % 2 != 0 {
        return Err("usage: brain fit FILE TRAIN.jsonl VALID.jsonl --learning-rate RATE --max-epochs N --patience N [--warmup-steps N] [--weight-decay RATE]".into());
    }
    let mut options = HashMap::new();
    for pair in args[4..].chunks_exact(2) {
        if !matches!(pair[0].as_str(), "--learning-rate" | "--max-epochs" | "--patience" | "--warmup-steps" | "--weight-decay") ||
            options.insert(pair[0].as_str(), pair[1].as_str()).is_some() {
            return Err("invalid or repeated fit option".into());
        }
    }
    let parse_f32 = |key: &str, default: Option<f32>| -> Result<f32, String> {
        let value = match options.get(key) {
            Some(value) => value.parse::<f32>().map_err(|_| format!("invalid {key}"))?,
            None => default.ok_or_else(|| format!("missing {key}"))?,
        };
        if !value.is_finite() { return Err(format!("invalid {key}")); }
        Ok(value)
    };
    let parse_u64 = |key: &str, default: Option<u64>| -> Result<u64, String> {
        match options.get(key) {
            Some(value) => value.parse().map_err(|_| format!("invalid {key}")),
            None => default.ok_or_else(|| format!("missing {key}")),
        }
    };
    let learning_rate = parse_f32("--learning-rate", None)?;
    let weight_decay = parse_f32("--weight-decay", Some(0.0))?;
    let max_epochs = parse_u64("--max-epochs", None)?;
    let patience = parse_u64("--patience", None)?;
    let warmup_steps = parse_u64("--warmup-steps", Some(0))?;
    if !(0.0 < learning_rate && learning_rate <= 1.0) ||
        !(0.0..=1.0).contains(&weight_decay) ||
        !(1..=1000).contains(&max_epochs) || patience == 0 || patience > max_epochs {
        return Err("fit option out of range".into());
    }
    brain_fit_install_cancel_handler()?;
    let path = Path::new(&args[1]);
    let mut brain = Brain::load(path).map_err(|error| error.name().to_owned())?;
    let train = fit_examples(Path::new(&args[2]), brain.vocabulary)?;
    let valid = fit_examples(Path::new(&args[3]), brain.vocabulary)?;
    brain_fit_check_cancelled()?;
    if let (Ok(train_path), Ok(valid_path)) =
        (std::fs::canonicalize(&args[2]), std::fs::canonicalize(&args[3])) {
        if train_path == valid_path { return Err("training and validation files must differ".into()); }
    }
    #[cfg(unix)] {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(train_file), Ok(valid_file)) =
            (std::fs::metadata(&args[2]), std::fs::metadata(&args[3])) {
            if train_file.dev() == valid_file.dev() && train_file.ino() == valid_file.ino() {
                return Err("training and validation files must differ".into());
            }
        }
    }
    let total_steps = max_epochs.checked_mul(train.len() as u64).ok_or("fit step limit")?;
    if warmup_steps > total_steps { return Err("warmup exceeds total steps".into()); }
    let mut best = None;
    let mut best_loss = f32::INFINITY;
    let mut best_epoch = 0;
    let mut unimproved = 0;
    let mut trace = Vec::new();
    let mut stop_reason = "max_epochs";
    for epoch in 0..max_epochs {
        let mut train_loss = 0.0_f64;
        let mut first_rate = 0.0;
        let mut last_rate = 0.0;
        for (index, example) in train.iter().enumerate() {
            brain_fit_check_cancelled()?;
            let step = epoch * train.len() as u64 + index as u64;
            let rate = Brain::scheduled_learning_rate(learning_rate, step, warmup_steps, total_steps)
                .map_err(|error| error.name().to_owned())?;
            if rate * weight_decay > 1.0 { return Err("fit decay exceeds one".into()); }
            if index == 0 { first_rate = rate; }
            last_rate = rate;
            train_loss += brain.train_next_token_with_weight_decay(&example.tokens, example.target, rate, weight_decay)
                .map_err(|error| error.name().to_owned())? as f64;
        }
        let mut valid_loss = 0.0_f64;
        for example in &valid {
            brain_fit_check_cancelled()?;
            valid_loss += brain.next_token_loss(&example.tokens, example.target)
                .map_err(|error| error.name().to_owned())? as f64;
        }
        let train_mean = (train_loss / train.len() as f64) as f32;
        let valid_mean = (valid_loss / valid.len() as f64) as f32;
        if !train_mean.is_finite() || !valid_mean.is_finite() { return Err("non-finite fit loss".into()); }
        trace.push(serde_json::json!({"epoch": epoch + 1, "train_mean_loss": train_mean,
            "validation_mean_loss": valid_mean, "first_step_rate": first_rate, "last_step_rate": last_rate}));
        if valid_mean < best_loss {
            best_loss = valid_mean;
            best_epoch = epoch + 1;
            best = Some(brain.clone());
            unimproved = 0;
        } else {
            unimproved += 1;
            if unimproved >= patience { stop_reason = "patience"; break; }
        }
    }
    let best = best.ok_or("no finite fit checkpoint")?;
    brain_fit_check_cancelled()?;
    brain_save(&best, path)?;
    Ok(serde_json::json!({"schema_version": 1, "seed": best.seed.to_string(),
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "max_epochs": max_epochs, "patience": patience, "warmup_steps": warmup_steps,
        "epochs": trace, "best_epoch": best_epoch, "stop_reason": stop_reason,
        "saved_brain_format": if best.layers.is_empty() { "LBRN1" } else { "LBRN2" },
        "saved_brain_version": best.version.to_string()}).to_string())
}

fn brain_memory_command(args: &[String]) -> Result<String, FitError> {
    use lana_runtime::brain_memory::{Evidence, Memory};
    use lana_runtime::information_codec::{canonical, Tagged};
    let operation = args.first().map(String::as_str).unwrap_or("");
    if !matches!((operation, args.len()), ("inspect", 3) | ("add", 5) | ("observe", 5)) {
        return Err("usage: brain memory add FILE ROOT_ID SOURCE VALUE.json | observe FILE ROOT_ID OBS_ID EVIDENCE.json | inspect FILE ROOT_ID".into());
    }
    let path = Path::new(&args[1]);
    let mut brain = Brain::load(path).map_err(|error| error.name().to_string())?;
    let mut memory = Memory::load(&brain.typed_memory_json).map_err(|error| error.name().to_string())?;
    if operation == "inspect" {
        return memory.inspect(&args[2]).map(|report| report.to_string()).map_err(|error| error.name().to_string().into());
    }
    let mut bytes = Vec::new();
    std::fs::File::open(&args[4]).and_then(|file| file.take(16 * 1024 * 1024 + 1).read_to_end(&mut bytes))
        .map_err(|error| error.to_string())?;
    if bytes.len() > 16 * 1024 * 1024 { return Err("LANA_ERR_LIMIT".into()); }
    let changed = if operation == "add" {
        let information: Tagged = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        memory.add(&args[2], &args[3], information)
    } else {
        let evidence: Evidence = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        memory.observe(&args[2], &args[3], evidence)
    }.map_err(|error| error.name().to_string())?;
    if changed {
        brain.typed_memory_json = canonical(&memory).map_err(|error| error.name().to_string())?;
        brain.version = brain.version.checked_add(1).ok_or("LANA_ERR_LIMIT")?;
        brain.save_with_status(path).map_err(|error| FitError { message: error.code.name().to_string(),
            durability_uncertain: error.durability_uncertain, path: Some(path.to_string_lossy().into_owned()) })?;
    }
    Ok(serde_json::json!({"status":"ok","changed":changed,"memory_revision":memory.memory_revision}).to_string())
}

fn brain_forecast_command(args: &[String]) -> Result<String, FitError> {
    use lana_runtime::brain_memory::{ForecastInput, Memory};
    use lana_runtime::information_codec::canonical;
    if !matches!((args.first().map(String::as_str), args.len()), (Some("add"), 6) | (Some("score"), 5)) {
        return Err("usage: brain forecast add FILE ID TARGET HORIZON FORECAST.json | score FILE ID OBSERVED_LABEL OBSERVED_AT".into());
    }
    let path = Path::new(&args[1]);
    let mut brain = Brain::load(path).map_err(|error| error.name().to_owned())?;
    let mut memory = Memory::load(&brain.typed_memory_json).map_err(|error| error.name().to_owned())?;
    let (changed, score) = if args[0] == "add" {
        let horizon = args[4].parse::<u64>().map_err(|_| "invalid horizon")?;
        let mut bytes = Vec::new();
        std::fs::File::open(&args[5]).and_then(|file| file.take(16 * 1024 * 1024 + 1).read_to_end(&mut bytes))
            .map_err(|error| error.to_string())?;
        if bytes.len() > 16 * 1024 * 1024 { return Err("LANA_ERR_LIMIT".into()); }
        let input: ForecastInput = serde_json::from_slice(&bytes).map_err(|_| "LANA_ERR_SCHEMA")?;
        (memory.forecast_add(&args[2], &args[3], horizon, input).map_err(|error| error.name().to_owned())?, None)
    } else {
        let observed_at = args[4].parse::<u64>().map_err(|_| "invalid observed_at")?;
        let (changed, score) = memory.forecast_score(&args[2], &args[3], observed_at)
            .map_err(|error| error.name().to_owned())?;
        (changed, Some(score))
    };
    if changed {
        brain.typed_memory_json = canonical(&memory).map_err(|error| error.name().to_owned())?;
        brain.version = brain.version.checked_add(1).ok_or("LANA_ERR_LIMIT")?;
        brain_save(&brain, path)?;
    }
    Ok(serde_json::json!({"status":"ok","changed":changed,"memory_revision":memory.memory_revision,
        "forecast_id":args[2],"score":score}).to_string())
}

fn brain_grounded_command(args: &[String]) -> Result<String, FitError> {
    use lana_runtime::brain_memory::Memory;
    use lana_runtime::information_codec::canonical;
    let path = Path::new(&args[1]);
    let mut brain = Brain::load(path).map_err(|error| error.name().to_string())?;
    let mut memory = Memory::load(&brain.typed_memory_json).map_err(|error| error.name().to_string())?;
    let (changed, report) = if args[0] == "chat" {
        let report = memory.grounded(&brain, &args[3]).map_err(|error| error.name().to_string())?;
        let exact = report["resolution"] == "exact";
        if exact {
            brain.memory.push(format!("user:{}", args[3]));
            let answer = report["answer"].as_str().map(ToString::to_string).unwrap_or_else(|| report["answer"].to_string());
            brain.memory.push(format!("assistant:{answer}"));
        }
        (exact, report)
    } else {
        let kind = if args[0] == "alias" { "fact" } else { "root" };
        if (kind == "fact" && brain.recall_fact(&args[2]).is_none()) ||
            (kind == "root" && !memory.roots.iter().any(|root| root.id == args[2])) {
            return Err("LANA_ERR_UNSUPPORTED_VALUE".into());
        }
        let changed = memory.alias(kind, &args[2], &args[3]).map_err(|error| error.name().to_string())?;
        if changed { brain.typed_memory_json = canonical(&memory).map_err(|error| error.name().to_string())?; }
        (changed, serde_json::json!({"status":"ok","changed":changed,"memory_revision":memory.memory_revision}))
    };
    if changed {
        brain.version = brain.version.checked_add(1).ok_or("LANA_ERR_LIMIT")?;
        brain.save_with_status(path).map_err(|error| FitError { message: error.code.name().to_string(),
            durability_uncertain: error.durability_uncertain, path: Some(path.to_string_lossy().into_owned()) })?;
    }
    Ok(report.to_string())
}

fn brain_tokenizer_bytes(path: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path).map_err(|error| error.to_string())?
        .take(16 * 1024 * 1024 + 1).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
    if bytes.len() > 16 * 1024 * 1024 { return Err("LANA_ERR_LIMIT".into()); }
    Ok(bytes)
}

fn brain_index_command(args: &[String]) -> Result<serde_json::Value, String> {
    use lana_runtime::brain_index::Index;
    use lana_runtime::brain_memory::Memory;
    if !matches!(args.len(), 3 | 6) || (args.len() == 6 && args[3] != "--calibrate") {
        return Err("usage: lana brain index FILE TOKENIZER [--calibrate DEVELOPMENT.jsonl HOLDOUT.jsonl]".into());
    }
    let path = Path::new(&args[1]);
    let brain = Brain::load(path).map_err(|error| error.name().to_owned())?;
    let memory = Memory::load(&brain.typed_memory_json).map_err(|error| error.name().to_owned())?;
    let bridge = std::env::var("LANA_HF").map_err(|_| "set LANA_HF to the installed local bridge")?;
    let tokenizer_bytes = brain_tokenizer_bytes(&args[2])?;
    bridge_tokens(&bridge, &args[2], "").map_err(|_| "LANA_ERR_SCHEMA")?;
    let tokenizer: serde_json::Value = serde_json::from_slice(&tokenizer_bytes).map_err(|_| "LANA_ERR_SCHEMA")?;
    let unknown = tokenizer["model"]["unk_token"].as_str().unwrap_or("[UNK]");
    let unknown_id = tokenizer["model"]["vocab"][unknown].as_u64().ok_or("LANA_ERR_SCHEMA")? as usize;
    // ponytail: one strict tokenizer bridge call per record; batch if large-corpus indexing is too slow.
    let mut index = Index::build(&brain, &memory, &tokenizer_bytes, unknown_id,
        |text| bridge_tokens(&bridge, &args[2], text).map_err(|_| LanaError::Schema))
        .map_err(|error| error.name().to_owned())?;
    let calibration = if args.len() == 6 {
        let read = |path: &str| -> Result<Vec<u8>, String> {
            let mut bytes = Vec::new();
            std::fs::File::open(path).map_err(|error| error.to_string())?
                .take(16 * 1024 * 1024 + 1).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
            if bytes.len() > 16 * 1024 * 1024 { return Err("LANA_ERR_LIMIT".into()); }
            Ok(bytes)
        };
        let development = read(&args[4])?;
        let heldout = read(&args[5])?;
        Some(index.calibrate(&brain, &memory, &development, &heldout, unknown_id,
            |text| bridge_tokens(&bridge, &args[2], text).map_err(|_| LanaError::Schema))
            .map_err(|error| error.name().to_owned())?)
    } else { None };
    let mut index_path = path.as_os_str().to_os_string();
    index_path.push(".index.json");
    let index_path = PathBuf::from(index_path);
    let bytes = index.bytes().map_err(|error| error.name().to_owned())?;
    lana_runtime::atomic_file::write(&index_path, &bytes).map_err(|error| {
        if lana_runtime::atomic_file::durability_uncertain(&error) {
            format!("LANA_ERR_IO: durability uncertain; inspect {}", index_path.display())
        } else { format!("LANA_ERR_IO: {error}") }
    })?;
    Ok(serde_json::json!({"status":"ok","index":index_path,
        "records":index.records.len(),"corpus_sha256":index.corpus_sha256,
        "calibration":calibration}))
}

fn brain_compress_command(args: &[String]) -> Result<serde_json::Value, String> {
    use lana_runtime::{brain_index::Index, brain_memory::Memory, brain_selector::Selector};
    if args.len() != 7 || args[1] != "fit" {
        return Err("usage: lana brain compress fit FILE TOKENIZER TRAIN.jsonl VALID.jsonl MODEL".into());
    }
    let brain = Brain::load(Path::new(&args[2])).map_err(|error| error.name().to_owned())?;
    let memory = Memory::load(&brain.typed_memory_json).map_err(|error| error.name().to_owned())?;
    let tokenizer_bytes = brain_tokenizer_bytes(&args[3])?;
    let bridge = std::env::var("LANA_HF").map_err(|_| "set LANA_HF to the installed local bridge")?;
    bridge_tokens(&bridge, &args[3], "").map_err(|_| "LANA_ERR_SCHEMA")?;
    let tokenizer: serde_json::Value = serde_json::from_slice(&tokenizer_bytes).map_err(|_| "LANA_ERR_SCHEMA")?;
    let unknown = tokenizer["model"]["unk_token"].as_str().unwrap_or("[UNK]");
    let unknown_id = tokenizer["model"]["vocab"][unknown].as_u64().ok_or("LANA_ERR_SCHEMA")? as usize;
    let index_path = format!("{}.index.json", args[2]);
    let index = Index::load(Path::new(&index_path), &brain, &memory, &tokenizer_bytes)
        .map_err(|error| error.name().to_owned())?;
    let read = |path: &str| -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        std::fs::File::open(path).map_err(|error| error.to_string())?.take(64 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes).map_err(|error| error.to_string())?;
        if bytes.len() > 64 * 1024 * 1024 { return Err("LANA_ERR_LIMIT".into()); }
        Ok(bytes)
    };
    let (selector, mut report) = Selector::fit(&brain, &memory, &index, &tokenizer_bytes,
        &read(&args[4])?, &read(&args[5])?, unknown_id,
        |text| bridge_tokens(&bridge, &args[3], text).map_err(|_| LanaError::Schema))
        .map_err(|error| error.name().to_owned())?;
    let output = if selector.active { PathBuf::from(&args[6]) } else { PathBuf::from(format!("{}.inactive.json", args[6])) };
    let resolved_output = if output.exists() { std::fs::canonicalize(&output) } else {
        let parent = output.parent().filter(|path| !path.as_os_str().is_empty()).unwrap_or(Path::new("."));
        std::fs::canonicalize(parent).map(|parent| parent.join(output.file_name().unwrap_or_default()))
    }.map_err(|error| error.to_string())?;
    for input in [&args[2], &args[3], &args[4], &args[5], &index_path] {
        if std::fs::canonicalize(input).map_err(|error| error.to_string())? == resolved_output {
            return Err("LANA_ERR_INVALID_PARAMETERS: selector output overlaps an input".into());
        }
    }
    report["artifact"] = serde_json::json!(output);
    let bytes = if selector.active { selector.bytes() } else {
        lana_runtime::information_codec::canonical(&serde_json::json!({"selector":selector,"report":report}))
    }.map_err(|error| error.name().to_owned())?;
    if bytes.len() > 64 * 1024 * 1024 { return Err("LANA_ERR_LIMIT".into()); }
    lana_runtime::atomic_file::write(&output, &bytes).map_err(|error| {
        if lana_runtime::atomic_file::durability_uncertain(&error) {
            format!("LANA_ERR_IO: durability uncertain; inspect {}", output.display())
        } else { format!("LANA_ERR_IO: {error}") }
    })?;
    Ok(report)
}

fn brain_semantic_command(args: &[String]) -> Result<serde_json::Value, String> {
    use lana_runtime::brain_index::Index;
    use lana_runtime::brain_memory::Memory;
    let path = Path::new(&args[1]);
    let brain = Brain::load(path).map_err(|error| error.name().to_owned())?;
    let memory = Memory::load(&brain.typed_memory_json).map_err(|error| error.name().to_owned())?;
    let bridge = std::env::var("LANA_HF").map_err(|_| "set LANA_HF to the installed local bridge")?;
    let tokenizer_bytes = brain_tokenizer_bytes(&args[2])?;
    let tokenizer: serde_json::Value = serde_json::from_slice(&tokenizer_bytes).map_err(|_| "LANA_ERR_SCHEMA")?;
    let unknown = tokenizer["model"]["unk_token"].as_str().unwrap_or("[UNK]");
    let unknown_id = tokenizer["model"]["vocab"][unknown].as_u64().ok_or("LANA_ERR_SCHEMA")? as usize;
    let mut index_path = path.as_os_str().to_os_string();
    index_path.push(".index.json");
    let index = Index::load(Path::new(&index_path), &brain, &memory, &tokenizer_bytes)
        .map_err(|error| error.name().to_owned())?;
    if args.len() == 7 {
        let selector = lana_runtime::brain_selector::Selector::load(Path::new(&args[6]), &index)
            .map_err(|error| error.name().to_owned())?;
        return selector.query(&brain, &memory, &index, &tokenizer_bytes, &args[3], unknown_id,
            |text| bridge_tokens(&bridge, &args[2], text).map_err(|_| LanaError::Schema))
            .map_err(|error| error.name().to_owned());
    }
    let ids = bridge_tokens(&bridge, &args[2], &args[3]).map_err(|_| "LANA_ERR_SCHEMA")?;
    index.query(&brain, &memory, &tokenizer_bytes, &ids, unknown_id)
        .map_err(|error| error.name().to_owned())
}

fn brain_command(args: &[String]) -> ExitCode {
    let fail = |message: &str| { eprintln!("brain: {message}\n{{\"status\":\"error\",\"error\":{}}}", json_quote(message)); ExitCode::from(1) };
    if args.is_empty() { return fail("usage: lana brain new|train|fit|evaluate|save|load|inspect|remember|recall|memory|alias|alias-root|forecast|index|compress|chat"); }
    match args[0].as_str() {
        "index" => match brain_index_command(args) {
            Ok(report) => { println!("{report}"); ExitCode::SUCCESS }
            Err(error) => fail(&error),
        },
        "compress" => match brain_compress_command(args) {
            Ok(report) => { println!("{report}"); ExitCode::SUCCESS }
            Err(error) => fail(&error),
        },
        "chat" if (args.len() == 5 || (args.len() == 7 && args[5] == "--selector")) && args[4] == "--semantic" => match brain_semantic_command(args) {
            Ok(report) => { println!("{report}"); ExitCode::SUCCESS }
            Err(error) => fail(&error),
        },
        "alias" | "alias-root" | "chat" if (args[0] != "chat" && args.len() == 4) ||
            (args[0] == "chat" && args.len() == 5 && args[4] == "--grounded") => match brain_grounded_command(args) {
            Ok(report) => { println!("{report}"); ExitCode::SUCCESS }
            Err(error) if error.durability_uncertain => {
                eprintln!("{}", serde_json::json!({"status":"error","error":error.message,"durability":"uncertain","path":error.path}));
                ExitCode::from(1)
            }
            Err(error) => fail(&error.message),
        },
        "memory" => match brain_memory_command(&args[1..]) {
            Ok(report) => { println!("{report}"); ExitCode::SUCCESS }
            Err(error) if error.durability_uncertain => {
                eprintln!("{}", serde_json::json!({"status":"error","error":error.message,"durability":"uncertain","path":error.path}));
                ExitCode::from(1)
            }
            Err(error) => fail(&error.message),
        },
        "forecast" => match brain_forecast_command(&args[1..]) {
            Ok(report) => { println!("{report}"); ExitCode::SUCCESS }
            Err(error) => brain_failure(error),
        },
        "new" if (args.len() == 6 || args.len() == 8) && args[4] == "--architecture" => {
            let (Ok(vocabulary), Ok(embedding_width)) = (args[2].parse(), args[3].parse()) else {
                return fail("vocabulary and embedding width must be positive integers");
            };
            let seed = if args.len() == 8 {
                if args[6] != "--seed" { return fail("expected --seed"); }
                match args[7].parse() { Ok(seed) => seed, Err(_) => return fail("seed must be an unsigned integer") }
            } else { 0x4c414e41 };
            let architecture = std::fs::File::open(&args[5])
                .map_err(|error| error.to_string())
                .and_then(|file| {
                    let mut bytes = Vec::new();
                    file.take(64 * 1024 + 1).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
                    if bytes.len() > 64 * 1024 { return Err("architecture file too large".into()); }
                    serde_json::from_slice::<BrainArchitecture>(&bytes).map_err(|error| error.to_string())
                });
            let Ok(architecture) = architecture else { return fail("invalid architecture JSON"); };
            let layers: Vec<_> = architecture.hidden.into_iter().map(|layer| (layer.width, match layer.activation {
                ArchitectureActivation::Relu => Activation::Relu,
                ArchitectureActivation::Gelu => Activation::Gelu,
            })).collect();
            match Brain::new_layers(vocabulary, embedding_width, &layers, seed)
                .map_err(|error| error.name().to_owned().into())
                .and_then(|brain| brain_save(&brain, Path::new(&args[1]))) {
                Ok(()) => { println!("brain created: {}\n{{\"status\":\"created\"}}", args[1]); ExitCode::SUCCESS }
                Err(error) => brain_failure(error),
            }
        }
        "fit" => match brain_fit(args) {
            Ok(report) => { println!("{report}"); ExitCode::SUCCESS }
            Err(error) if error.durability_uncertain => {
                eprintln!("brain: {}\n{}", error.message, serde_json::json!({
                    "status": "error", "error": error.message, "durability": "uncertain",
                    "path": error.path
                }));
                ExitCode::from(1)
            }
            Err(error) => fail(&error.message),
        },
        "new" if args.len() == 5 || args.len() == 6 => {
            let parse = |index: usize| args[index].parse::<usize>().map_err(|_| ());
            let Ok(vocabulary) = parse(2) else { return fail("vocabulary must be a positive integer"); };
            let Ok(embedding_width) = parse(3) else { return fail("embedding width must be a positive integer"); };
            let Ok(hidden_width) = parse(4) else { return fail("hidden width must be a positive integer"); };
            let seed = if args.len() == 6 { match args[5].parse() { Ok(seed) => seed, Err(_) => return fail("seed must be an integer") } } else { 0x4c414e41 };
            match Brain::new(vocabulary, embedding_width, hidden_width, seed)
                .map_err(|error| error.name().to_owned().into())
                .and_then(|brain| brain_save(&brain, Path::new(&args[1]))) {
                Ok(()) => { println!("brain created: {}\n{{\"status\":\"created\"}}", args[1]); ExitCode::SUCCESS }
                Err(error) => brain_failure(error),
            }
        }
        "train" if args.len() >= 5 => {
            let target = match args[2].parse() { Ok(value) => value, Err(_) => return fail("target must be a token ID") };
            let learning_rate = match args[3].parse() { Ok(value) => value, Err(_) => return fail("learning rate must be a number") };
            let tokens: Result<Vec<usize>, _> = args[4..].iter().map(|token| token.parse()).collect();
            let Ok(tokens) = tokens else { return fail("tokens must be token IDs"); };
            let path = Path::new(&args[1]);
            match Brain::load(path).map_err(|error| FitError::from(error.name())).and_then(|brain| {
                let trained = brain.trained_next_token(&tokens, target, learning_rate).map_err(|error| FitError::from(error.name()))?;
                brain_save(&trained.brain, path)?;
                Ok(trained)
            }) {
                Ok(trained) => {
                    println!("brain trained: loss={}\n{}", trained.loss, serde_json::json!({
                        "status": "trained", "loss": trained.loss, "version": trained.brain.version,
                        "changed_groups": trained.changed_groups
                    }));
                    ExitCode::SUCCESS
                }
                Err(error) => brain_failure(error),
            }
        }
        "evaluate" if args.len() >= 3 => {
            let tokens: Result<Vec<usize>, _> = args[2..].iter().map(|token| token.parse()).collect();
            let Ok(tokens) = tokens else { return fail("tokens must be token IDs"); };
            match Brain::load(Path::new(&args[1])).and_then(|brain| brain.logits(&tokens)) {
                Ok(logits) => { println!("brain evaluated\n{{\"status\":\"evaluated\",\"logits\":{:?}}}", logits); ExitCode::SUCCESS }
                Err(error) => fail(error.name()),
            }
        }
        "save" if args.len() == 4 => match bridge_package(&["package", &args[1], &args[2], &args[3]]) {
            Ok(()) => { println!("brain package saved: {}\n{{\"status\":\"saved\"}}", args[2]); ExitCode::SUCCESS }
            Err(error) => fail(&error),
        },
        "save" if args.len() == 3 => match Brain::load(Path::new(&args[1]))
            .map_err(|error| FitError::from(error.name()))
            .and_then(|brain| brain_save(&brain, Path::new(&args[2]))) {
            Ok(()) => { println!("brain saved: {}\n{{\"status\":\"saved\"}}", args[2]); ExitCode::SUCCESS }
            Err(error) => brain_failure(error),
        },
        "load" if args.len() == 3 => match bridge_package(&["unpackage", &args[1], &args[2]]) {
            Ok(()) => { println!("brain package loaded: {}\n{{\"status\":\"loaded\"}}", args[2]); ExitCode::SUCCESS }
            Err(error) => fail(&error),
        },
        "load" | "inspect" if args.len() == 2 => match Brain::load(Path::new(&args[1])) {
            Ok(brain) => {
                let memory = lana_runtime::brain_memory::Memory::load(&brain.typed_memory_json).expect("validated Brain memory");
                println!("brain version {}\n{}", brain.version, serde_json::json!({
                    "status":"loaded", "version":brain.version, "vocabulary":brain.vocabulary,
                    "embedding_width":brain.embedding_width,"hidden_width":brain.hidden_width,
                    "replay_steps":brain.replay_steps,"training_steps":brain.training_history.len(),
                    "memory_revision":(brain.memory.len() - brain.fact_revision()) / 2,"fact_revision":brain.fact_revision(),
                    "typed_memory_revision":memory.memory_revision,"parameter_sha256":brain.parameter_sha256()
                }));
                ExitCode::SUCCESS
            }
            Err(error) => fail(error.name()),
        },
        "remember" if args.len() == 4 => {
            let path = Path::new(&args[1]);
            match Brain::load(path).map_err(|error| FitError::from(error.name())).and_then(|mut brain| {
                let changed = brain.remember_fact(&args[2], &args[3]).map_err(|error| FitError::from(error.name()))?;
                if changed { brain_save(&brain, path)?; }
                Ok((changed, brain.fact_revision()))
            }) {
                Ok((changed, revision)) => { println!("brain fact saved\n{{\"status\":\"ok\",\"changed\":{changed},\"fact_revision\":{revision}}}"); ExitCode::SUCCESS }
                Err(error) => brain_failure(error),
            }
        }
        "recall" if args.len() == 3 => match Brain::load(Path::new(&args[1])) {
            Ok(brain) => match brain.recall_fact(&args[2]) {
                Some(value) => { println!("brain fact recalled\n{{\"status\":\"ok\",\"value\":{},\"resolution\":\"exact\",\"assumptions\":[],\"unsupported\":false}}", json_quote(value)); ExitCode::SUCCESS }
                None => { println!("{{\"status\":\"unsupported\",\"unsupported\":true}}"); ExitCode::from(1) }
            },
            Err(error) => fail(error.name()),
        },
        "chat" if args.len() == 6 && args[4] == "--fact" => {
            let path = Path::new(&args[1]);
            match Brain::load(path).map_err(|error| FitError::from(error.name())).and_then(|mut brain| {
                let response = brain.recall_fact(&args[5]).ok_or(LanaError::UnsupportedOperation)
                    .map_err(|error| FitError::from(error.name()))?.to_owned();
                let version = brain.version.checked_add(1).ok_or(LanaError::Limit)
                    .map_err(|error| FitError::from(error.name()))?;
                brain.memory.push(format!("user:{}", args[3]));
                brain.memory.push(format!("assistant:{response}"));
                brain.version = version;
                brain_save(&brain, path)?;
                Ok((response, brain.version, (brain.memory.len() - brain.fact_revision()) / 2))
            }) {
                Ok((response, version, revision)) => { println!("brain chat: {response}\n{{\"status\":\"ok\",\"response\":{},\"version\":{version},\"memory_revision\":{revision},\"resolution\":\"exact\",\"assumptions\":[],\"unsupported\":false}}", json_quote(&response)); ExitCode::SUCCESS }
                Err(error) if error.message == LanaError::UnsupportedOperation.name() => { println!("{{\"status\":\"unsupported\",\"unsupported\":true}}"); ExitCode::from(1) }
                Err(error) => brain_failure(error),
            }
        }
        "chat" if args.len() == 4 => {
            let Ok(bridge) = std::env::var("LANA_HF") else { return fail("set LANA_HF to the installed local bridge"); };
            let path = Path::new(&args[1]);
            match Brain::load(path).map_err(|error| FitError::from(error.name())).and_then(|mut brain| {
                let mut context = Vec::new();
                let turns: Vec<_> = brain.memory.iter().rev().filter(|entry| !entry.starts_with("\0fact\0")).take(16).collect();
                for turn in turns.into_iter().rev() {
                    context.extend(bridge_tokens(&bridge, &args[2], turn).map_err(|_| FitError::from(LanaError::UnsupportedOperation.name()))?);
                }
                let tokens = bridge_tokens(&bridge, &args[2], &args[3]).map_err(|_| FitError::from(LanaError::UnsupportedOperation.name()))?;
                if tokens.is_empty() { return Err(FitError::from(LanaError::InvalidParameters.name())); }
                context.extend(tokens);
                if context.len() > 4096 { context.drain(..context.len() - 4096); }
                let logits = brain.logits(&context).map_err(|error| FitError::from(error.name()))?;
                let next = logits.iter().enumerate().max_by(|left, right| left.1.total_cmp(right.1)).map(|(index, _)| index)
                    .ok_or_else(|| FitError::from(LanaError::InvalidState.name()))?;
                let response = bridge_text(&bridge, &args[2], next).map_err(|_| FitError::from(LanaError::UnsupportedOperation.name()))?;
                brain.memory.push(format!("user:{}", args[3]));
                brain.memory.push(format!("assistant:{response}"));
                brain.version = brain.version.checked_add(1).ok_or_else(|| FitError::from(LanaError::Limit.name()))?;
                brain_save(&brain, path)?;
                Ok((response, brain.version, (brain.memory.len() - brain.fact_revision()) / 2))
            }) {
                Ok((response, version, revision)) => { println!("brain chat: {response}\n{{\"status\":\"ok\",\"response\":{},\"version\":{version},\"memory_revision\":{revision}}}", json_quote(&response)); ExitCode::SUCCESS }
                Err(error) => brain_failure(error),
            }
        }
        "chat" => fail("usage: lana brain chat brain.lbrn tokenizer.json text [--grounded | --semantic [--selector MODEL] | --fact KEY]"),
        _ => fail("usage: lana brain new|train|fit|evaluate|save|load|inspect|remember|recall|memory|alias|alias-root|forecast|index|compress|chat"),
    }
}

fn new_project(directory: &str) -> ExitCode {
    let root = Path::new(directory);
    if root.exists()
        || std::fs::create_dir_all(root.join("src")).is_err()
        || std::fs::create_dir_all(root.join("tests")).is_err()
    {
        eprintln!("new: cannot create {directory}");
        return ExitCode::from(1);
    }
    let files = [
        ("lana.toml", "schema = 1\nname = \"hello-lana\"\nversion = \"0.1.0\"\nentry = \"src/main.lana\"\n\n[dependencies]\n"),
        ("src/belief.lana", "fn label() { return \"confirmed\"; }\n"),
        ("src/main.lana", "import \"./belief.lana\" as belief;\n\nprint(belief.label());\n"),
        ("tests/main_test.lana", "import \"../src/belief.lana\" as belief;\n\nassert(belief.label() == \"confirmed\", \"belief label\");\n"),
    ];
    for (path, contents) in files {
        if std::fs::write(root.join(path), contents).is_err() {
            eprintln!("new: cannot write {directory}/{path}");
            return ExitCode::from(1);
        }
    }
    println!("created {directory}");
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    if std::env::var_os("LANA_STDLIB_DIR").is_none() {
        if let Ok(executable) = std::env::current_exe() {
            if let Some(bin) = executable.parent() {
                let installed = bin.join("../share/lana/stdlib");
                if installed.is_dir() {
                    std::env::set_var("LANA_STDLIB_DIR", installed);
                }
            }
        }
    }
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage("lana");
        return ExitCode::from(2);
    }
    match args[1].as_str() {
        "package" => match packages::command(&args[2..]) {
            Ok(report) => { println!("{report}"); ExitCode::SUCCESS }
            Err(error) => { eprintln!("package: {error}"); ExitCode::from(1) }
        },
        "brain" => brain_command(&args[2..]),
        "bridge-worker" if args.len() == 2 => bridge_worker::serve(),
        "execution-config" => execution_config_command(&args[2..]),
        "version" => {
            println!("Lana {LANA_VERSION} (LABC v2, Rust VM, native compiler)");
            ExitCode::SUCCESS
        }
        "new" => {
            if args.len() != 3 {
                usage("lana");
                return ExitCode::from(2);
            }
            new_project(&args[2])
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
            if args.len() != 2 {
                usage("lana");
                return ExitCode::from(2);
            }
            let Some(compiler) = find_compiler() else {
                eprintln!("native Lana compiler bytecode not found");
                return ExitCode::from(1);
            };
            let mut output = String::new();
            match project_build_with_plan(".", &compiler, &mut output) {
                Ok(()) => ExitCode::SUCCESS,
                Err(()) => ExitCode::from(1),
            }
        }
        "test" => {
            if args.len() != 2 {
                usage("lana");
                return ExitCode::from(2);
            }
            ExitCode::from(project_test("."))
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
        "asm" => assemble_command(&args[2..]),
        "debug" => debug_command(&args[1..]),
        "run" => {
            if args.len() == 2 {
                let Some(compiler) = find_compiler() else {
                    eprintln!("native Lana compiler bytecode not found");
                    return ExitCode::from(1);
                };
                let mut output = String::new();
                if project_build_with_plan(".", &compiler, &mut output).is_err() {
                    return ExitCode::from(1);
                }
                run_command(&[output])
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
