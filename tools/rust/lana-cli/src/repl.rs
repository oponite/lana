//! Interactive REPL (LIP-020).
//!
//! A thin loop over the existing compiler and VM: each completed input is
//! compiled and executed against the accumulated session source, so top-level
//! bindings persist across inputs. The session is deterministic given the same
//! input sequence and seed (the RNG stream is per-session).
//!
//! Commands (`:help`, `:quit`/`:exit`, `:load`, `:save`, `:clear`) are prefixed
//! with `:` and are not Lana syntax.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::process::ExitCode;

use lana_bytecode::{Chunk, LanaError};
use lana_vm::{Value, Vm};

use crate::{CliError, compile_source_to_chunk, report_cli_error};

/// The per-session RNG seed, matching the default used by `run`.
const REPL_SEED: u64 = 0x4c414e41;

/// Statement keywords that begin a non-expression input. A line starting with
/// one of these is compiled as a statement (no result is displayed).
const STATEMENT_KEYWORDS: &[&str] = &[
    "let", "fn", "if", "while", "for", "return", "print", "import", "from",
    "match", "type", "struct", "async", "generator", "yield", "break",
    "continue", "assert", "use", "const", "var", "shared", "capability",
    "effect", "claim", "task", "spawn", "await", "export",
];

/// Run the interactive REPL. `compiler` is the self-hosted compiler bytecode.
pub fn run_repl(compiler: &Path) -> ExitCode {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut session_source = String::new();
    let mut buffer = String::new();
    loop {
        let prompt = if buffer.is_empty() { "lana> " } else { "...> " };
        print!("{prompt}");
        let _ = io::stdout().flush();
        let line = match lines.next() {
            Some(Ok(line)) => line,
            _ => break, // EOF ends the session.
        };
        if buffer.is_empty() {
            let trimmed = line.trim();
            if trimmed.starts_with(':') {
                if handle_command(compiler, trimmed, &mut session_source) {
                    break; // :quit / :exit
                }
                continue;
            }
        }
        buffer.push_str(&line);
        buffer.push('\n');
        if !is_complete(&buffer) {
            continue; // Multiline: wait for the closing bracket.
        }
        process_input(compiler, &mut session_source, &buffer);
        buffer.clear();
    }
    ExitCode::SUCCESS
}

/// True when the accumulated input is a complete statement (balanced brackets,
/// no unterminated string literal).
fn is_complete(buffer: &str) -> bool {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for c in buffer.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
    }
    depth <= 0 && !in_string
}

/// Handle a `:`-prefixed command. Returns `true` when the session should end.
fn handle_command(compiler: &Path, command: &str, session_source: &mut String) -> bool {
    let (name, argument) = match command.find(' ') {
        Some(index) => (&command[..index], command[index + 1..].trim()),
        None => (command, ""),
    };
    match name {
        ":help" => {
            println!("commands:");
            println!("  :help              list commands");
            println!("  :quit, :exit       end the session");
            println!("  :load <file>       compile and run a file in the session");
            println!("  :save <file>       export the session's bindings as a .lana file");
            println!("  :clear             drop all bindings");
            false
        }
        ":quit" | ":exit" => true,
        ":clear" => {
            session_source.clear();
            false
        }
        ":save" => {
            if argument.is_empty() {
                eprintln!("usage: :save <file>");
            } else if let Err(error) = std::fs::write(argument, session_source.as_bytes()) {
                eprintln!(":save: {error}");
            } else {
                println!("saved {argument}");
            }
            false
        }
        ":load" => {
            if argument.is_empty() {
                eprintln!("usage: :load <file>");
                return false;
            }
            let contents = match std::fs::read_to_string(argument) {
                Ok(contents) => contents,
                Err(error) => {
                    eprintln!(":load: {error}");
                    return false;
                }
            };
            // Append the file's contents to the session and recompile+rerun.
            let candidate = format!("{session_source}{contents}");
            match compile_and_run(compiler, &candidate) {
                Ok(_) => {
                    *session_source = candidate;
                    println!("loaded {argument}");
                }
                Err(ReplError::Compile(error)) => report_compile_error(&error),
                Err(ReplError::Runtime(error)) => report_error(&error),
            }
            false
        }
        _ => {
            eprintln!("unknown command: {name} (try :help)");
            false
        }
    }
}

/// A failure while compiling or running a REPL input.
enum ReplError {
    Compile(CliError),
    Runtime(lana_vm::VmError),
}

/// Compile and run a source text, returning the program's result value.
fn compile_and_run(compiler: &Path, source: &str) -> Result<Value, ReplError> {
    let chunk = compile_source_to_chunk(compiler, source).map_err(ReplError::Compile)?;
    run_chunk(&chunk).map_err(ReplError::Runtime)
}

/// Run a chunk on a fresh VM and return the result value.
fn run_chunk(chunk: &Chunk) -> Result<Value, lana_vm::VmError> {
    let mut vm = Vm::new(chunk);
    vm.seed(REPL_SEED);
    let mut store_host = lana_runtime::host_calls::StoreHost::with_heap(vm.heap());
    vm.set_host_call_extension(Box::new(move |host_id, args, out| {
        store_host.dispatch(host_id, args, out)
    }));
    let result = vm.run();
    if result != LanaError::Ok {
        return Err(vm.error().clone());
    }
    Ok(vm.result().clone())
}

/// Compile and execute one complete input, committing it to the session on
/// success. A bare expression's value is printed; a statement prints nothing.
fn process_input(compiler: &Path, session_source: &mut String, buffer: &str) {
    if buffer.trim().is_empty() {
        return;
    }
    let trimmed = buffer.trim_end();
    let trimmed = trimmed.strip_suffix(';').unwrap_or(trimmed).trim_end();
    let is_expression = !trimmed.is_empty() && !starts_with_keyword(trimmed);

    if is_expression {
        // Probe form: `return (<line>);` captures the expression's value so the
        // REPL can display it. If the line is not a bare expression this fails
        // to compile and we fall back to the plain statement form below.
        let probe = format!("{session_source}{buffer}return ({trimmed});\n");
        match compile_and_run(compiler, &probe) {
            Ok(result) => {
                *session_source = format!("{session_source}{buffer}");
                println!("{}", result.print());
                return;
            }
            Err(ReplError::Compile(_)) => {
                // Not a bare expression; fall through to the plain form.
            }
            Err(ReplError::Runtime(error)) => {
                report_error(&error);
                return;
            }
        }
    }

    let plain = format!("{session_source}{buffer}");
    match compile_and_run(compiler, &plain) {
        Ok(_) => {
            *session_source = plain;
        }
        Err(ReplError::Compile(error)) => report_compile_error(&error),
        Err(ReplError::Runtime(error)) => report_error(&error),
    }
}

/// True when the trimmed line begins with a statement keyword.
fn starts_with_keyword(line: &str) -> bool {
    for keyword in STATEMENT_KEYWORDS {
        if line == *keyword || line.starts_with(&format!("{keyword} ")) {
            return true;
        }
    }
    false
}

/// Report a compile error with the compiler's recovery message (what is
/// missing and where), per SYNTAX-10.
fn report_compile_error(error: &CliError) {
    match error {
        CliError::Run(vm_error) => {
            eprintln!("{}", vm_error.message);
        }
        _ => report_cli_error(error),
    }
}

/// Report a runtime error, mirroring the `run` command's error output.
fn report_error(error: &lana_vm::VmError) {
    crate::report_error(error);
}
