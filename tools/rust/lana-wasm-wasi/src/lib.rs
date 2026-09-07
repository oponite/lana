//! WASI command entry for the Lana runtime (LIP-025).
//!
//! This crate is a thin command-line wrapper over the same compile / gate /
//! run path the browser `lana-wasm` crate exposes, so a program compiles to the
//! same LABC v2 and runs byte-identically on a `wasm32-wasip1` target running
//! under a WASI runtime (e.g. `wasmtime`). Unlike `lana-wasm`, it does not
//! depend on `wasm-bindgen`, so it builds for WASI as well as any host.
//!
//! `run(source, input, capabilities) -> String` mirrors `lana-wasm::run`
//! exactly: it compiles `source` with the embedded self-hosted compiler,
//! rejects any gated filesystem or network host call the host has not wired
//! (`LANA_ERR_UNSUPPORTED_OPERATION`), and returns the same JSON contract:
//!
//!   {"ok":true,"result":"..."} | {"ok":false,"error":{"line":N,"message":"..."}}
//!
//! Resource limits (256 MiB, 50,000,000 instructions) are the Rust VM's
//! defaults and are enforced in the run loop, byte-identical to native and to
//! the browser target.

use std::collections::HashSet;

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

/// Host calls gated behind explicit capabilities (LIP-025 §3): the filesystem
/// and networking host calls. Mirrors `lana-wasm` so both targets agree.
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

/// Map a gated host-call ID to its name, or "unknown" when not gated.
fn gated_host_call_name(id: u32) -> &'static str {
    GATED_HOST_CALLS
        .iter()
        .find(|(gated_id, _)| *gated_id == id)
        .map(|(_, name)| *name)
        .unwrap_or("unknown")
}

/// Parse the `capabilities` JSON object (`{"name":true,...}` plus optional
/// `seed`, `instruction_limit`, `memory_limit`). Empty string = none.
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
/// VM's in-memory filesystem. Mirrors `lana-wasm::compile_source`.
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
    if vm.run() != LanaError::Ok {
        return format!("{{\"ok\":false,\"error\":{}}}", error_json(&vm));
    }
    let result = vm.result().print();
    format!("{{\"ok\":true,\"result\":\"{}\"}}", json_escape(&result))
}

/// Compile and run a Lana source program, returning a JSON result string.
/// Mirrors `lana-wasm::run`.
pub fn run(source: &str, input: &str, capabilities: &str) -> String {
    let chunk = match compile_source(source) {
        Ok(chunk) => chunk,
        Err(error) => return format!("{{\"ok\":false,\"error\":{error}}}"),
    };
    run_chunk(chunk, input, capabilities)
}

/// Run a precompiled LABC blob given as a hex string, returning the same JSON
/// contract as `run`. Lets the WASI conformance feed a hand-assembled
/// `HOST_CALL http_get` chunk through the identical gating path, so LIP-025 §3
/// network gating is provable on the WASI target even though the compiler at
/// this fork point cannot yet emit network host calls from source.
pub fn run_bytecode_hex(hex: &str, capabilities: &str) -> String {
    let bytes = match decode_hex(hex) {
        Some(bytes) => bytes,
        None => return "{\"ok\":false,\"error\":{\"line\":0,\"message\":\"invalid hex bytecode\"}}".to_string(),
    };
    let chunk = match lana_bytecode::loader::load(&bytes) {
        Ok(chunk) => chunk,
        Err(info) => {
            return format!(
                "{{\"ok\":false,\"error\":{{\"line\":{},\"message\":\"{}\"}}}}",
                info.line,
                json_escape(&info.message)
            );
        }
    };
    run_chunk(chunk, "", capabilities)
}

/// Decode a hex string (whitespace ignored) into bytes.
fn decode_hex(hex: &str) -> Option<Vec<u8>> {
    let clean: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
    if clean.len() % 2 != 0 {
        return None;
    }
    (0..clean.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&clean[i..i + 2], 16).ok())
        .collect()
}
