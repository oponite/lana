//! Persistent JSON bridge for one-shot runs and process-local live programs.

use std::io::{BufRead, Read, Write};
use std::path::Path;
use std::process::ExitCode;

use lana_bytecode::LanaError;
use lana_runtime::live::{LiveFailure, LiveHost};

use serde::Deserialize;

use crate::{compile_source_file, find_compiler, json_quote, temp_path, CliError, LANA_VERSION};

const MAX_MESSAGE: u64 = 64 * 1024 * 1024;

#[derive(Default, Deserialize)]
struct Controls {
    seed: Option<u64>,
    memory_limit_mib: Option<usize>,
    instruction_limit: Option<u64>,
    workers: Option<usize>,
    max_tasks: Option<usize>,
}

impl Controls {
    fn parse(request: &serde_json::Value) -> Result<Self, String> {
        let controls = Self::deserialize(request).map_err(|error| error.to_string())?;
        for name in ["seed", "memory_limit_mib", "instruction_limit", "workers", "max_tasks"] {
            if let Some(value) = request.get(name) {
                if value.as_u64().filter(|&n| n > 0).is_none() {
                    return Err(format!("{name} must be a positive integer"));
                }
            }
        }
        if controls.memory_limit_mib.is_some_and(|mib| mib.checked_mul(1024 * 1024).is_none()) {
            return Err("memory_limit_mib overflows the native byte limit".into());
        }
        Ok(controls)
    }

    fn apply(&self, vm: &mut lana_vm::Vm) -> Result<(), LanaError> {
        if let Some(seed) = self.seed { vm.seed(seed); }
        if let Some(workers) = self.workers {
            let status = vm.set_worker_count(workers);
            if status != LanaError::Ok { return Err(status); }
        }
        if let Some(tasks) = self.max_tasks {
            let status = vm.set_task_limit(tasks);
            if status != LanaError::Ok { return Err(status); }
        }
        if let Some(mib) = self.memory_limit_mib {
            vm.set_memory_limit(mib.checked_mul(1024 * 1024).ok_or(LanaError::Limit)?)?;
        }
        if let Some(limit) = self.instruction_limit { vm.set_instruction_limit(limit); }
        Ok(())
    }
}

fn failure(phase: &str, code: &str, message: &str, stdout: &str) -> String {
    format!(
        "{{\"schema\":1,\"ok\":false,\"phase\":{},\"error\":{{\"code\":{},\"message\":{}}},\"stdout\":{},\"stderr\":\"\",\"execution\":{{\"engine\":\"rust-worker\",\"lana_version\":{}}}}}",
        json_quote(phase), json_quote(code), json_quote(message), json_quote(stdout), json_quote(LANA_VERSION)
    )
}

fn success(result: &str, stdout: &str) -> String {
    format!(
        "{{\"schema\":1,\"ok\":true,\"result\":{result},\"stdout\":{},\"stderr\":\"\",\"execution\":{{\"engine\":\"rust-worker\",\"lana_version\":{}}}}}",
        json_quote(stdout), json_quote(LANA_VERSION)
    )
}

struct Scratch(Vec<std::path::PathBuf>);

impl Drop for Scratch {
    fn drop(&mut self) {
        for path in &self.0 {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn live_failure(error: LiveFailure) -> String {
    failure("live", error.code.name(), &error.message, "")
}

fn live_request(request: &serde_json::Value, operation: &str, controls: &Controls, host: &mut LiveHost) -> String {
    let mut stdout = String::new();
    let required = |field: &str| request.get(field).and_then(serde_json::Value::as_str)
        .ok_or_else(|| LiveFailure { code: LanaError::Schema, message: format!("missing {field}") });
    let result = (|| -> Result<serde_json::Value, LiveFailure> {
        match operation {
            "start_live" | "start_live_labc" => {
                let path = Path::new(required("path")?);
                if !path.is_file() { return Err(LiveFailure { code: LanaError::Io, message: "program file not found".into() }); }
                let path = path.canonicalize().map_err(|error| LiveFailure { code: LanaError::Io, message: error.to_string() })?;
                let bytecode = temp_path("lana-live-bytecode");
                let _scratch = Scratch(vec![bytecode.clone()]);
                let effective = if operation == "start_live" {
                    let compiler = find_compiler().ok_or_else(|| LiveFailure { code: LanaError::Io, message: "compiler bytecode not found".into() })?;
                    compile_source_file(&compiler, &path.to_string_lossy(), &bytecode.to_string_lossy())
                        .map_err(|error| match error {
                            CliError::Run(error) | CliError::Compile { error, .. } => LiveFailure { code: error.code, message: error.message },
                            CliError::Load { info, .. } | CliError::Assemble { info, .. } | CliError::Write { info, .. } => LiveFailure { code: info.code, message: info.message },
                            CliError::Project => LiveFailure { code: LanaError::Io, message: "project build failed".into() },
                        })?;
                    &bytecode
                } else { &path };
                let bytes = std::fs::read(effective).map_err(|error| LiveFailure { code: LanaError::Io, message: error.to_string() })?;
                let original_dir = std::env::current_dir().map_err(|error| LiveFailure { code: LanaError::Io, message: error.to_string() })?;
                if operation == "start_live" {
                    std::env::set_current_dir(path.parent().unwrap_or(Path::new(".")))
                        .map_err(|error| LiveFailure { code: LanaError::Io, message: error.to_string() })?;
                }
                let result = host.start_live_labc_with(&bytes, |vm| controls.apply(vm));
                let _ = std::env::set_current_dir(original_dir);
                let handle = result?;
                stdout = host.output(&handle)?;
                Ok(serde_json::json!({"handle":handle,"state":"QUIESCENT","names":host.names(&handle)?}))
            }
            "observe_live" => host.observe_live(required("handle")?, required("name")?,
                request.get("evidence").ok_or_else(|| LiveFailure { code: LanaError::Schema, message: "missing evidence".into() })?.clone()),
            "inspect_live" => host.inspect_live(required("handle")?, required("name")?),
            "pause_live" => host.pause_live(required("handle")?),
            "resume_live" => host.resume_live(required("handle")?),
            "delete_live" => host.delete_live(required("handle")?),
            _ => unreachable!(),
        }
    })();
    match result {
        Ok(value) => success(&value.to_string(), &stdout),
        Err(error) => live_failure(error),
    }
}

fn run_request(line: &str, host: &mut LiveHost) -> String {
    let request: serde_json::Value = match serde_json::from_str(line) {
        Ok(request) => request,
        Err(_) => return failure("protocol", "LANA_ERR_PARSE", "invalid request JSON", ""),
    };
    if request.get("schema").and_then(|value| value.as_u64()) != Some(1) {
        return failure("protocol", "LANA_ERR_SCHEMA", "expected schema 1", "");
    }
    let controls = match Controls::parse(&request) {
        Ok(controls) => controls,
        Err(message) => return failure("protocol", "LANA_ERR_SCHEMA", &message, ""),
    };
    let Some(operation) = request.get("op").and_then(|value| value.as_str()) else {
        return failure("protocol", "LANA_ERR_SCHEMA", "missing operation", "");
    };
    if matches!(operation, "start_live" | "start_live_labc" | "observe_live" | "inspect_live"
        | "pause_live" | "resume_live" | "delete_live") {
        return live_request(&request, operation, &controls, host);
    }
    if operation != "run" && operation != "run_labc" {
        return failure("protocol", "LANA_ERR_UNSUPPORTED_OPERATION", "unknown operation", "");
    }
    let Some(path) = request.get("path").and_then(|value| value.as_str()) else {
        return failure("protocol", "LANA_ERR_SCHEMA", "missing path", "");
    };
    let Some(input) = request.get("input") else {
        return failure("protocol", "LANA_ERR_SCHEMA", "missing input", "");
    };
    let input = match lana_runtime::json_parse(&input.to_string()).and_then(|value| lana_runtime::json_stringify(&value)) {
        Ok(input) => input,
        Err(error) => return failure("protocol", error.name(), "unsupported input JSON", ""),
    };
    let source = Path::new(path);
    if !source.is_file() {
        return failure("load", "LANA_ERR_IO", "program file not found", "");
    }
    let path = match source.canonicalize() {
        Ok(path) => path,
        Err(_) => return failure("load", "LANA_ERR_IO", "cannot resolve program path", ""),
    };
    let compiler = if operation == "run" { find_compiler() } else { None };
    let request_path = temp_path("lana-worker-request");
    let response_path = temp_path("lana-worker-response");
    let bytecode_path = temp_path("lana-worker-bytecode");
    let _scratch = Scratch(vec![request_path.clone(), response_path.clone(), bytecode_path.clone()]);
    if std::fs::write(&request_path, input).is_err() {
        return failure("protocol", "LANA_ERR_IO", "cannot write request", "");
    }
    let original_dir = match std::env::current_dir() {
        Ok(dir) => dir,
        Err(_) => return failure("run", "LANA_ERR_IO", "cannot read working directory", ""),
    };
    if operation == "run" && std::env::set_current_dir(path.parent().unwrap_or(Path::new("."))).is_err() {
        return failure("run", "LANA_ERR_IO", "cannot enter program directory", "");
    }
    let result = run_program(operation, &path, compiler.as_deref(), &bytecode_path, &request_path, &response_path, &controls);
    let _ = std::env::set_current_dir(original_dir);
    result
}

fn run_program(operation: &str, path: &Path, compiler: Option<&Path>, bytecode: &Path, request: &Path, response: &Path, controls: &Controls) -> String {
    let effective = if operation == "run" {
        let Some(compiler) = compiler else {
            return failure("compile", "LANA_ERR_IO", "compiler bytecode not found", "");
        };
        if let Err(error) = compile_source_file(compiler, &path.to_string_lossy(), &bytecode.to_string_lossy()) {
            let (code, message) = match error {
                CliError::Run(error) | CliError::Compile { error, .. } => (error.code.name(), error.message),
                CliError::Load { info, .. } | CliError::Assemble { info, .. } | CliError::Write { info, .. } => (info.code.name(), info.message),
                CliError::Project => ("LANA_ERR_IO", "project build failed".to_string()),
            };
            return failure("compile", code, &message, "");
        }
        bytecode
    } else {
        path
    };
    let bytes = match std::fs::read(effective) {
        Ok(bytes) => bytes,
        Err(_) => return failure("load", "LANA_ERR_IO", "cannot read bytecode", ""),
    };
    let chunk = match lana_bytecode::loader::load(&bytes) {
        Ok(chunk) => chunk,
        Err(info) => return failure("load", info.code.name(), &info.message, ""),
    };
    let mut vm = lana_vm::Vm::new(&chunk);
    if let Err(error) = controls.apply(&mut vm) {
        return failure("run", error.name(), "cannot apply VM controls", "");
    }
    vm.capture_output();
    vm.set_program_args(&[request.to_string_lossy().into_owned(), response.to_string_lossy().into_owned()]);
    let mut host = lana_runtime::host_calls::StoreHost::new();
    host.set_chunk_bytes(bytes);
    vm.set_host_call_extension(Box::new(move |vm, id, args, out| host.dispatch(vm, id, args, out)));
    let status = vm.run();
    let stdout = vm.output().unwrap_or_default();
    if status != LanaError::Ok {
        return failure("run", status.name(), &vm.error().message, &stdout);
    }
    let size = match std::fs::metadata(response) {
        Ok(meta) => meta.len(),
        Err(_) => return failure("protocol", "LANA_RESPONSE_MISSING", "program did not write a response", &stdout),
    };
    if size > MAX_MESSAGE {
        return failure("protocol", "LANA_ERR_LIMIT", "response exceeds 64 MiB", &stdout);
    }
    let body = match std::fs::read_to_string(response) {
        Ok(body) => body,
        Err(_) => return failure("protocol", "LANA_ERR_IO", "cannot read response", &stdout),
    };
    let result = match lana_runtime::json_parse(&body).and_then(|value| lana_runtime::json_stringify(&value)) {
        Ok(result) => result,
        Err(_) => return failure("protocol", "LANA_RESPONSE_INVALID", "program wrote invalid response JSON", &stdout),
    };
    success(&result, &stdout)
}

pub fn serve() -> ExitCode {
    let mut host = LiveHost::new();
    let stdin = std::io::stdin();
    let mut input = std::io::BufReader::new(stdin.lock());
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    loop {
        let mut line = Vec::new();
        let read = match (&mut input).take(MAX_MESSAGE + 1).read_until(b'\n', &mut line) {
            Ok(read) => read,
            Err(_) => return ExitCode::from(1),
        };
        if read == 0 {
            return ExitCode::SUCCESS;
        }
        if read as u64 > MAX_MESSAGE || !line.ends_with(b"\n") {
            return ExitCode::from(1);
        }
        let response = match std::str::from_utf8(&line) {
            Ok(line) => run_request(line, &mut host),
            Err(_) => failure("protocol", "LANA_ERR_PARSE", "request is not UTF-8", ""),
        };
        if writeln!(output, "{response}").is_err() || output.flush().is_err() {
            return ExitCode::from(1);
        }
    }
}

fn take_word<'a>(input: &mut &'a str) -> Option<&'a str> {
    *input = input.trim_start();
    if input.is_empty() { return None; }
    let end = input.find(char::is_whitespace).unwrap_or(input.len());
    let (word, rest) = input.split_at(end);
    *input = rest;
    Some(word)
}

/// Foreground line session. Responses use the worker envelope so every command
/// has the same error code and result shape as the Python host.
pub fn live(path: &str) -> ExitCode {
    let mut host = LiveHost::new();
    let operation = if path.ends_with(".labc") { "start_live_labc" } else { "start_live" };
    let request = serde_json::json!({"schema":1,"op":operation,"path":path});
    let response = run_request(&request.to_string(), &mut host);
    println!("{response}");
    if serde_json::from_str::<serde_json::Value>(&response).ok()
        .and_then(|value| value.get("ok").and_then(serde_json::Value::as_bool)) != Some(true) {
        return ExitCode::from(1);
    }
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        match input.read_line(&mut line) { Ok(0) => break, Err(_) => return ExitCode::from(1), Ok(_) => {} }
        let mut rest = line.trim();
        let Some(command) = take_word(&mut rest) else { continue; };
        if command == "quit" { break; }
        let request = match command {
            "load" => {
                let path = rest.trim();
                if path.is_empty() { None } else {
                    Some(serde_json::json!({"schema":1,"op":if path.ends_with(".labc") { "start_live_labc" } else { "start_live" },"path":path}))
                }
            }
            "observe" => {
                let handle = take_word(&mut rest);
                let name = take_word(&mut rest);
                match (handle, name, serde_json::from_str::<serde_json::Value>(rest.trim())) {
                    (Some(handle), Some(name), Ok(evidence)) =>
                        Some(serde_json::json!({"schema":1,"op":"observe_live","handle":handle,"name":name,"evidence":evidence})),
                    _ => None,
                }
            }
            "inspect" => {
                let handle = take_word(&mut rest);
                let name = take_word(&mut rest);
                match (handle, name) { (Some(handle), Some(name)) if rest.trim().is_empty() =>
                    Some(serde_json::json!({"schema":1,"op":"inspect_live","handle":handle,"name":name})), _ => None }
            }
            "pause" | "resume" | "delete" => {
                let handle = take_word(&mut rest);
                match handle { Some(handle) if rest.trim().is_empty() =>
                    Some(serde_json::json!({"schema":1,"op":format!("{command}_live"),"handle":handle})), _ => None }
            }
            _ => None,
        };
        let response = request.map(|request| run_request(&request.to_string(), &mut host))
            .unwrap_or_else(|| failure("protocol", "LANA_ERR_SCHEMA", "invalid live command", ""));
        println!("{response}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controls_preserve_integer_precision_and_reject_invalid_limits() {
        let request = serde_json::json!({"seed": u64::MAX, "instruction_limit": u64::MAX});
        let controls = Controls::parse(&request).unwrap();
        assert_eq!(controls.seed, Some(u64::MAX));
        assert_eq!(controls.instruction_limit, Some(u64::MAX));
        assert!(Controls::parse(&serde_json::json!({})).is_ok());
        for name in ["seed", "instruction_limit", "workers", "max_tasks", "memory_limit_mib"] {
            for value in [serde_json::Value::Null, serde_json::json!(true), serde_json::json!(0),
                          serde_json::json!(-1), serde_json::json!(1.5), serde_json::json!("1")] {
                let mut request = serde_json::json!({});
                request[name] = value;
                assert!(Controls::parse(&request).is_err());
            }
        }
        assert!(Controls::parse(&serde_json::json!({"memory_limit_mib": usize::MAX})).is_err());
    }
}
