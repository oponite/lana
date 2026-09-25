//! Persistent JSON bridge. Each request runs in a fresh VM.

use std::io::{BufRead, Read, Write};
use std::path::Path;
use std::process::ExitCode;

use lana_bytecode::LanaError;

use crate::{compile_source_file, find_compiler, json_quote, lsp_member, lsp_number, lsp_string, temp_path, CliError, LANA_VERSION};

const MAX_MESSAGE: u64 = 64 * 1024 * 1024;

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
    let request = match lana_runtime::json_parse(line) {
        Ok(request) => request,
        Err(_) => return failure("protocol", "LANA_ERR_PARSE", "invalid request JSON", ""),
    };
    if lsp_number(&request, "schema") != Some(1) {
        return failure("protocol", "LANA_ERR_SCHEMA", "expected schema 1", "");
    }
    let Some(operation) = lsp_string(&request, "op") else {
        return failure("protocol", "LANA_ERR_SCHEMA", "missing operation", "");
    };
    if operation != "run" && operation != "run_labc" {
        return failure("protocol", "LANA_ERR_UNSUPPORTED_OPERATION", "unknown operation", "");
    }
    let Some(path) = lsp_string(&request, "path") else {
        return failure("protocol", "LANA_ERR_SCHEMA", "missing path", "");
    };
    let Some(input) = lsp_member(&request, "input") else {
        return failure("protocol", "LANA_ERR_SCHEMA", "missing input", "");
    };
    let input = match lana_runtime::json_stringify(&input) {
        Ok(input) => input,
        Err(error) => return failure("protocol", error.name(), "unsupported input JSON", ""),
    };
    let source = Path::new(&path);
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
    let result = run_program(&operation, &path, compiler.as_deref(), &bytecode_path, &request_path, &response_path);
    let _ = std::env::set_current_dir(original_dir);
    result
}

fn run_program(operation: &str, path: &Path, compiler: Option<&Path>, bytecode: &Path, request: &Path, response: &Path) -> String {
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
    vm.capture_output();
    vm.set_program_args(&[request.to_string_lossy().into_owned(), response.to_string_lossy().into_owned()]);
    let mut host = lana_runtime::host_calls::StoreHost::new();
    vm.set_host_call_extension(Box::new(move |id, args, out| host.dispatch(id, args, out)));
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
