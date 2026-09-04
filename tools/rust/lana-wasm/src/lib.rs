//! WebAssembly bindings for the Lana runtime (LIP-002 / plan item 1.5).
//!
//! Exposes `check` and `run` over `wasm-bindgen`, compiling Lana source with
//! the embedded self-hosted compiler (`lana-compiler.labc`) and running it on
//! the Rust VM. The compiler's file-backed host calls (`read_text`,
//! `write_text`, `path_exists`) resolve against the VM's in-memory filesystem,
//! so no host filesystem is required.
//!
//! Both entry points return a JSON string so the boundary stays a single
//! `String` (no `serde`/`JsValue` marshalling):
//!
//!   check(source) -> {"ok":true} | {"ok":false,"error":{"line":N,"message":"..."}}
//!   run(source, input) -> {"ok":true,"result":"..."} | {"ok":false,"error":{...}}
//!
//! `input` is passed to the program as its single argument (available via
//! `args()`); an empty string passes no argument.

use wasm_bindgen::prelude::*;

use lana_bytecode::{Chunk, LanaError};
use lana_vm::Vm;

/// The self-hosted compiler, copied into `OUT_DIR` by `build.rs`.
const COMPILER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/lana-compiler.labc"));

/// Virtual paths the compiler reads/writes within the in-memory filesystem.
const SOURCE_PATH: &str = "/src/main.lana";
const ASM_PATH: &str = "/src/main.lasm";

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

/// Compile and run a Lana source program, returning a JSON result string.
/// `input` is passed to the program as its single argument (empty = none).
#[wasm_bindgen]
pub fn run(source: &str, input: &str) -> String {
    let chunk = match compile_source(source) {
        Ok(chunk) => chunk,
        Err(error) => return format!("{{\"ok\":false,\"error\":{error}}}"),
    };
    let mut vm = Vm::new(&chunk);
    if !input.is_empty() {
        vm.set_program_args(&[input.to_string()]);
    }
    let mut store_host = lana_runtime::host_calls::StoreHost::new();
    vm.set_host_call_extension(Box::new(move |host_id, args, out| {
        store_host.dispatch(host_id, args, out)
    }));
    if vm.run() != LanaError::Ok {
        return format!("{{\"ok\":false,\"error\":{}}}", error_json(&vm));
    }
    let result = vm.result().print();
    format!("{{\"ok\":true,\"result\":\"{}\"}}", json_escape(&result))
}
