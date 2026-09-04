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

const LANA_VERSION: &str = "1.2.0";

/// Full usage text, mirroring `usage()` in `tools/c/cli.c` (with `lanavm` folded
/// into the single `lana` binary).
fn usage(program: &str) {
    eprintln!(
        "usage:\n  {program} compile program.lana -o program.labc\n  {program} new directory\n  {program} lsp\n  {program} debug program.lana\n  {program} build|run|test|check|fmt|doc\n  {program} check program.lana\n  {program} asm program.lasm -o program.labc\n  {program} run program.labc [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--memory-limit-mib N] [--instruction-limit N]\n  {program} run-bytecode program.labc [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--memory-limit-mib N] [--instruction-limit N]\n  {program} dis program.labc\n  {program} verify program.labc\n  {program} inspect program.lana [--format json|dot]"
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
        vm.set_memory_limit(mib * 1024 * 1024);
    }
    if let Some(limit) = instruction_limit {
        vm.set_instruction_limit(limit);
    }
    vm.set_program_args(&program_args);
    let mut store_host = lana_runtime::host_calls::StoreHost::new();
    vm.set_host_call_extension(Box::new(move |host_id, args, out| {
        store_host.dispatch(host_id, args, out)
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
/// The Rust VM has no interactive debugger, so breakpoints are ignored.
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
    eprintln!("debug: interactive stepping is not supported by the Rust VM; running without breakpoints");
    let code = run_command(&[bytecode_str]);
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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage("lana");
        return ExitCode::from(2);
    }
    match args[1].as_str() {
        "version" => {
            println!("Lana {LANA_VERSION} (LABC v2, Rust VM, native compiler)");
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
            eprintln!("lsp: not yet implemented in the Rust CLI");
            ExitCode::from(1)
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
        "check" => {
            let Some(compiler) = find_compiler() else {
                eprintln!("native Lana compiler bytecode not found");
                return ExitCode::from(1);
            };
            if args.len() == 2 {
                let mut output = String::new();
                match project_build_with_plan(".", &compiler, &mut output) {
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
