//! Persistent JSON bridge. Each request runs in a fresh VM.

use std::io::{BufRead, Read, Write};
use std::path::Path;
use std::process::ExitCode;

use lana_bytecode::LanaError;

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

fn run_request(line: &str) -> String {
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
            Ok(line) => run_request(line),
            Err(_) => failure("protocol", "LANA_ERR_PARSE", "request is not UTF-8", ""),
        };
        if writeln!(output, "{response}").is_err() || output.flush().is_err() {
            return ExitCode::from(1);
        }
    }
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
