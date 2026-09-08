//! WebAssembly bindings for the Lana runtime (LIP-002 / LIP-025).
//!
//! Exposes `check`, `run`, and `run_bytecode` over `wasm-bindgen`, compiling
//! Lana source with the embedded self-hosted compiler (`lana-compiler.labc`)
//! and running it on the Rust VM. The compiler's file-backed host calls
//! (`read_text`, `write_text`, `path_exists`) resolve against the VM's
//! in-memory filesystem, so no host filesystem is required.
//!
//! Both entry points return a JSON string so the boundary stays a single
//! `String` (no `serde`/`JsValue` marshalling):
//!
//!   check(source) -> {"ok":true} | {"ok":false,"error":{"line":N,"message":"..."}}
//!   run(source, input, capabilities) -> {"ok":true,"result":"..."} | {"ok":false,"error":{...}}
//!   run_bytecode(labc, input, capabilities) -> same as run
//!
//! `input` is passed to the program as its single argument (available via
//! `args()`); an empty string passes no argument.
//!
//! # Host-call policy (LIP-025 §3)
//!
//! Filesystem and networking host calls are unavailable by default in WASM
//! (there is no ambient filesystem or network). A program that uses a gated
//! host call — the filesystem calls (`read_text`, `write_text`,
//! `directory_list`, `directory_create`, `path_exists`, `write_text_atomic`)
//! or the networking calls (`http_get`, `http_post`, `socket_connect`,
//! `socket_send`, `socket_recv`, `socket_close`) — fails with
//! `LANA_ERR_UNSUPPORTED_OPERATION` unless the host explicitly wires that
//! capability. The `capabilities` argument is a JSON object naming which gated
//! host calls are wired (value `true`) plus optional `seed`,
//! `instruction_limit`, and `memory_limit` (MiB) keys:
//!
//!   {"read_text":true,"write_text":true,"seed":42,"instruction_limit":1000}
//!
//! Gating is enforced by scanning the compiled chunk for `HOST_CALL`
//! instructions before execution, so the VM itself is unchanged and native
//! behavior is byte-identical.

use std::collections::HashSet;

use wasm_bindgen::prelude::*;

use lana_bytecode::{Chunk, LanaError, OpCode};
use lana_vm::vm::{
    LANA_HOST_DIRECTORY_CREATE, LANA_HOST_DIRECTORY_LIST, LANA_HOST_HTTP_GET,
    LANA_HOST_HTTP_POST, LANA_HOST_PATH_EXISTS, LANA_HOST_READ_TEXT,
    LANA_HOST_SOCKET_CLOSE, LANA_HOST_SOCKET_CONNECT, LANA_HOST_SOCKET_RECV,
    LANA_HOST_SOCKET_SEND, LANA_HOST_WRITE_TEXT, LANA_HOST_WRITE_TEXT_ATOMIC,
};
use lana_vm::Vm;

/// The self-hosted compiler, copied into `OUT_DIR` by `build.rs`.
const COMPILER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/lana-compiler.labc"));

/// Virtual paths the compiler reads/writes within the in-memory filesystem.
const SOURCE_PATH: &str = "/src/main.lana";
const ASM_PATH: &str = "/src/main.lasm";

/// Host calls gated behind explicit capabilities in WASM (LIP-025 §3). These
/// are the filesystem and networking host calls; there is no ambient FS or
/// network in a WASM embedding, so a program that names one without wiring it
/// fails with `LANA_ERR_UNSUPPORTED_OPERATION`.
const GATED_HOST_CALLS: &[(u32, &str)] = &[
    (LANA_HOST_READ_TEXT, "read_text"),
    (LANA_HOST_WRITE_TEXT, "write_text"),
    (LANA_HOST_DIRECTORY_LIST, "directory_list"),
    (LANA_HOST_DIRECTORY_CREATE, "directory_create"),
    (LANA_HOST_PATH_EXISTS, "path_exists"),
    (LANA_HOST_WRITE_TEXT_ATOMIC, "write_text_atomic"),
    (LANA_HOST_HTTP_GET, "http_get"),
    (LANA_HOST_HTTP_POST, "http_post"),
    (LANA_HOST_SOCKET_CONNECT, "socket_connect"),
    (LANA_HOST_SOCKET_SEND, "socket_send"),
    (LANA_HOST_SOCKET_RECV, "socket_recv"),
    (LANA_HOST_SOCKET_CLOSE, "socket_close"),
];

/// Parsed capabilities: which gated host calls are wired, plus optional
/// determinism/resource controls.
struct Capabilities {
    enabled: HashSet<u32>,
    seed: Option<u64>,
    instruction_limit: Option<u64>,
    memory_limit_mib: Option<u64>,
}

/// Map a gated host-call name to its ID.
fn gated_host_call_id(name: &str) -> Option<u32> {
    GATED_HOST_CALLS.iter().find(|(_, n)| *n == name).map(|(id, _)| *id)
}

/// Map a gated host-call ID to its name.
fn gated_host_call_name(id: u32) -> &'static str {
    GATED_HOST_CALLS
        .iter()
        .find(|(gated_id, _)| *gated_id == id)
        .map(|(_, name)| *name)
        .unwrap_or("unknown")
}

/// Parse the `capabilities` JSON object. Accepts `{"name":true,...}` for gated
/// host calls plus `seed`, `instruction_limit`, and `memory_limit` numbers.
/// An empty string parses to no capabilities (all gated calls rejected).
fn parse_capabilities(json: &str) -> Capabilities {
    let mut caps = Capabilities {
        enabled: HashSet::new(),
        seed: None,
        instruction_limit: None,
        memory_limit_mib: None,
    };
    let bytes = json.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end] != b'"' {
                end += 1;
            }
            let key = &json[start..end];
            // Advance past the colon and whitespace to the value.
            let mut j = end + 1;
            while j < bytes.len() && (bytes[j] == b':' || bytes[j].is_ascii_whitespace()) {
                j += 1;
            }
            match key {
                "seed" | "instruction_limit" | "memory_limit" => {
                    let num_start = j;
                    let mut num_end = num_start;
                    while num_end < bytes.len()
                        && (bytes[num_end].is_ascii_digit() || bytes[num_end] == b'.')
                    {
                        num_end += 1;
                    }
                    let value: f64 = json[num_start..num_end].parse().unwrap_or(0.0);
                    match key {
                        "seed" => caps.seed = Some(value as u64),
                        "instruction_limit" => caps.instruction_limit = Some(value as u64),
                        _ => caps.memory_limit_mib = Some(value as u64),
                    }
                }
                _ => {
                    // A gated host call is enabled only when its value is `true`.
                    if j < bytes.len() && bytes[j] == b't' {
                        if let Some(id) = gated_host_call_id(key) {
                            caps.enabled.insert(id);
                        }
                    }
                }
            }
            i = end + 1;
        } else {
            i += 1;
        }
    }
    caps
}

/// Reject a chunk that uses a gated host call the host has not wired.
fn check_host_call_capabilities(chunk: &Chunk, enabled: &HashSet<u32>) -> Result<(), String> {
    for instruction in &chunk.code {
        if instruction.opcode == OpCode::HostCall {
            let id = instruction.b;
            if gated_host_call_name(id) != "unknown" && !enabled.contains(&id) {
                return Err(format!(
                    "{{\"line\":{},\"message\":\"LANA_ERR_UNSUPPORTED_OPERATION: host call '{}' is not available in this embedding\"}}",
                    instruction.line,
                    gated_host_call_name(id)
                ));
            }
        }
    }
    Ok(())
}

/// Escape a string for inclusion in a JSON string literal.
fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if (character as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => out.push(character),
        }
    }
    out
}

/// Format a VM error as a JSON `error` object.
fn error_json(vm: &Vm) -> String {
    let error = vm.error();
    format!(
        "{{\"line\":{},\"message\":\"{}\"}}",
        error.line,
        json_escape(&error.message),
    )
}

/// Compile Lana source to a chunk, running the embedded compiler against the
/// VM's in-memory filesystem. Returns the assembled chunk or a JSON error.
fn compile_source(source: &str) -> Result<Chunk, String> {
    let compiler_chunk = lana_bytecode::loader::load(COMPILER)
        .map_err(|info| format!("{{\"line\":0,\"message\":\"cannot load compiler: {}\"}}", json_escape(&info.message)))?;
    let mut vm = Vm::new(&compiler_chunk);
    vm.set_virtual_file(SOURCE_PATH, source.to_string());
    vm.set_program_args(&[SOURCE_PATH.to_string(), ASM_PATH.to_string()]);
    if vm.run() != LanaError::Ok {
        return Err(error_json(&vm));
    }
    let assembly = vm
        .take_virtual_file(ASM_PATH)
        .ok_or_else(|| "{\"line\":0,\"message\":\"compiler did not emit assembly\"}".to_string())?;
    lana_bytecode::assemble(&assembly)
        .map_err(|info| format!("{{\"line\":{},\"message\":\"{}\"}}", info.line, json_escape(&info.message)))
}

/// Compile-check a Lana source program, returning a JSON status string.
#[wasm_bindgen]
pub fn check(source: &str) -> String {
    match compile_source(source) {
        Ok(_) => "{\"ok\":true}".to_string(),
        Err(error) => format!("{{\"ok\":false,\"error\":{error}}}"),
    }
}

/// Run a compiled chunk with the given input and capabilities.
fn run_chunk(chunk: Chunk, input: &str, capabilities: &str) -> String {
    let caps = parse_capabilities(capabilities);
    if let Err(error) = check_host_call_capabilities(&chunk, &caps.enabled) {
        return format!("{{\"ok\":false,\"error\":{error}}}");
    }
    let mut vm = Vm::new(&chunk);
    if let Some(seed) = caps.seed {
        vm.seed(seed);
    }
    if let Some(limit) = caps.instruction_limit {
        vm.set_instruction_limit(limit);
    }
    if let Some(mib) = caps.memory_limit_mib {
        vm.set_memory_limit(mib as usize * 1024 * 1024);
    }
    if !input.is_empty() {
        vm.set_program_args(&[input.to_string()]);
    }
    let mut store_host = lana_runtime::host_calls::StoreHost::with_heap(vm.heap());
    vm.set_host_call_extension(Box::new(move |host_id, args, out| {
        store_host.dispatch(host_id, args, out)
    }));
    if vm.run() != LanaError::Ok {
        return format!("{{\"ok\":false,\"error\":{}}}", error_json(&vm));
    }
    let result = vm.result().print();
    format!("{{\"ok\":true,\"result\":\"{}\"}}", json_escape(&result))
}

/// Compile and run a Lana source program, returning a JSON result string.
/// `input` is passed to the program as its single argument (empty = none).
/// `capabilities` is a JSON object naming wired host calls and optional
/// `seed`/`instruction_limit`/`memory_limit` (see module docs).
#[wasm_bindgen]
pub fn run(source: &str, input: &str, capabilities: &str) -> String {
    let chunk = match compile_source(source) {
        Ok(chunk) => chunk,
        Err(error) => return format!("{{\"ok\":false,\"error\":{error}}}"),
    };
    run_chunk(chunk, input, capabilities)
}

/// Run a precompiled LABC bytecode blob, returning a JSON result string.
/// `input` and `capabilities` behave as in `run`.
#[wasm_bindgen]
pub fn run_bytecode(labc: &[u8], input: &str, capabilities: &str) -> String {
    let chunk = match lana_bytecode::loader::load(labc) {
        Ok(chunk) => chunk,
        Err(info) => {
            return format!(
                "{{\"ok\":false,\"error\":{{\"line\":{},\"message\":\"{}\"}}}}",
                info.line,
                json_escape(&info.message)
            )
        }
    };
    run_chunk(chunk, input, capabilities)
}
