//! Command-line driver for the Rust Lana runtime.
//!
//! `lana run <file.labc> [--seed N] [--stats]` loads and verifies a chunk,
//! runs it on the Rust VM, and reports the result.
//!
//! The full command surface includes `version`, `new`,
//! `lsp`, `fmt`, `doc`, `build`, `test`, `compile`, `asm`, `debug`,
//! `run`, `run-bytecode`, `dis`, and `verify`. Commands that need the
//! self-hosted compiler locate `lana-compiler.labc` and run it on the Rust VM.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::io::Write;

use lana_bytecode::{Chunk, LanaError, LanaErrorInfo, OpCode};
use lana_runtime::execution::ExecutionConfig;
use lana_vm::Vm;

mod bridge_worker;
mod packages;
mod lsp;

const LANA_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Full usage text for the single `lana` binary.
fn usage(program: &str) {
    eprintln!(
        "usage:\n  {program} compile program.lana -o program.labc\n  {program} new directory\n  {program} package pack DIRECTORY -o ARCHIVE | package add owner/repo@X.Y.Z\n  {program} lsp\n  {program} debug program.lana\n  {program} build|run|test|fmt|doc\n  {program} asm program.lasm -o program.labc\n  {program} run program.labc [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--instruction-limit N]\n  {program} run-bytecode program.labc [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--instruction-limit N]\n  {program} dis program.labc\n  {program} verify program.labc\n  {program} inspect program.lana [--format json|dot]\n  {program} execution-config init --metadata PATH --key PATH --capability-id ID --origin HTTPS_ORIGIN --credential-key-id ID [--ca-file PATH]"
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

fn json_quote(text: &str) -> String {
    let mut quoted = String::from("\"");
    for character in text.chars() {
        match character { '\"' => quoted.push_str("\\\""), '\\' => quoted.push_str("\\\\"), '\n' => quoted.push_str("\\n"), '\r' => quoted.push_str("\\r"), '\t' => quoted.push_str("\\t"), c if c.is_control() => quoted.push_str(&format!("\\u{:04x}", c as u32)), c => quoted.push(c) }
    }
    quoted.push('\"');
    quoted
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
