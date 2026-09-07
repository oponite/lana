//! Native-vs-WASM conformance for the `lana-wasm` entry points.
//!
//! `check`/`run`/`run_bytecode` are `#[wasm_bindgen]` functions, but they
//! remain ordinary Rust functions when the crate is built as an `rlib`, so this
//! test calls them directly and pins the exact JSON they must produce. The same
//! expectations are asserted against the compiled wasm module by
//! `tests/conformance.mjs`, so a change that diverges on either target fails
//! one of the two suites.

use lana_wasm::{check, run, run_bytecode};

#[test]
fn check_accepts_valid_source() {
    let source = "state a = state(p: 0.5, d: 0.0);\nlet p = measure a as probability;\nprint(p);\n";
    assert_eq!(check(source), "{\"ok\":true}");
}

#[test]
fn check_rejects_invalid_source() {
    let source = "let x = ;\n";
    let result = check(source);
    assert!(result.starts_with("{\"ok\":false,\"error\":{\"line\":"));
    assert!(result.contains("\"message\":\"parse error at line 1 column 9: expected expression, got symbol ;\""));
}

#[test]
fn run_returns_scalar() {
    assert_eq!(run("return 42;\n", "", ""), "{\"ok\":true,\"result\":\"42\"}");
}

#[test]
fn run_returns_state_dist() {
    let source = "state a = state(p: 0.2, d: 0.0);\nstate b = state(p: 0.3, d: 0.0);\nlet c = append(a, b);\nreturn c;\n";
    assert_eq!(run(source, "", ""), "{\"ok\":true,\"result\":\"state_dist\"}");
}

#[test]
fn run_passes_input_as_single_argument() {
    let source = "let a = args();\nreturn a[0];\n";
    assert_eq!(run(source, "hello", ""), "{\"ok\":true,\"result\":\"hello\"}");
}

#[test]
fn run_escapes_string_result() {
    let source = "return \"a\\\"b\\n\";\n";
    assert_eq!(run(source, "", ""), "{\"ok\":true,\"result\":\"a\\\"b\\n\"}");
}

/// Serialize a minimal LABC blob: `LOAD_CONST R0 42; RETURN R0; HALT`.
fn labc_return_42() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LABC");
    bytes.extend_from_slice(&2u32.to_le_bytes()); // version
    bytes.extend_from_slice(&1u32.to_le_bytes()); // constants
    bytes.extend_from_slice(&0u32.to_le_bytes()); // functions
    bytes.extend_from_slice(&3u32.to_le_bytes()); // instructions
    bytes.extend_from_slice(&0u32.to_le_bytes()); // entry
    bytes.push(1); // ValueType::Number
    bytes.extend_from_slice(&42.0f64.to_bits().to_le_bytes());
    // LOAD_CONST R0 42 (opcode 1, a=0, imm=0)
    bytes.push(1);
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes()); // line
    // RETURN R0 (opcode 29, a=0)
    bytes.push(29);
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes()); // line
    // HALT (opcode 31)
    bytes.push(31);
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&3u32.to_le_bytes()); // line
    bytes
}

/// Serialize a minimal LABC blob containing a single HOST_CALL with the given
/// host id, followed by HALT. The gating scan in `run_chunk` runs over the
/// compiled chunk before execution, so a call to a gated host (for example
/// `http_get`, host id 160) is rejected without requiring the instruction to
/// be fully valid against register semantics.
fn labc_host_call(host_id: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LABC");
    bytes.extend_from_slice(&2u32.to_le_bytes()); // version
    bytes.extend_from_slice(&1u32.to_le_bytes()); // constants
    bytes.extend_from_slice(&0u32.to_le_bytes()); // functions
    bytes.extend_from_slice(&1u32.to_le_bytes()); // instructions
    bytes.extend_from_slice(&0u32.to_le_bytes()); // entry
    bytes.push(1); // ValueType::Number
    bytes.extend_from_slice(&0.0f64.to_bits().to_le_bytes());
    // HOST_CALL (opcode 39): a=8 (dest), b=host_id, c=0, imm=0, line=1
    bytes.push(39);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.extend_from_slice(&host_id.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes()); // line
    bytes
}

#[test]
fn run_bytecode_returns_scalar() {
    let labc = labc_return_42();
    assert_eq!(run_bytecode(&labc, "", ""), "{\"ok\":true,\"result\":\"42\"}");
}

#[test]
fn run_gates_http_get_when_not_wired() {
    // LIP-025 §3: networking host calls are unavailable by default.
    let result = run_bytecode(&labc_host_call(160), "", "");
    assert!(
        result.starts_with("{\"ok\":false,\"error\":{\"line\":"),
        "expected gated error, got {result}"
    );
    assert!(result.contains("LANA_ERR_UNSUPPORTED_OPERATION"));
    assert!(result.contains("host call 'http_get' is not available"));
}

#[test]
fn run_lifts_gate_on_wired_http_get() {
    // When the capability is wired, the gate is lifted and the call proceeds to
    // the host (here it fails at the network boundary, not with the gate error).
    let result = run_bytecode(&labc_host_call(160), "", "{\"http_get\":true}");
    assert!(
        !result.contains("LANA_ERR_UNSUPPORTED_OPERATION"),
        "gate should be lifted when wired, got {result}"
    );
}

#[test]
fn run_gates_socket_connect_when_not_wired() {
    // socket_connect (host id 162) is gated by default like http_get.
    let result = run_bytecode(&labc_host_call(162), "", "");
    assert!(result.contains("LANA_ERR_UNSUPPORTED_OPERATION"));
    assert!(result.contains("host call 'socket_connect' is not available"));
}

#[test]
fn run_bytecode_rejects_invalid_labc() {
    assert!(run_bytecode(&[0xff, 0xff, 0xff], "", "").starts_with("{\"ok\":false,\"error\":"));
}

#[test]
fn run_gates_directory_list_when_not_wired() {
    let source = "let e = directory_list(\"/tmp\");\nreturn e;\n";
    let result = run(source, "", "");
    assert!(result.starts_with("{\"ok\":false,\"error\":{\"line\":"));
    assert!(result.contains("\"message\":\"LANA_ERR_UNSUPPORTED_OPERATION: host call 'directory_list' is not available in this embedding\""));
}

#[test]
fn run_gates_read_text_when_not_wired() {
    let source = "let t = read_text(\"/tmp/x\");\nreturn t;\n";
    let result = run(source, "", "");
    assert!(result.starts_with("{\"ok\":false,\"error\":{\"line\":"));
    assert!(result.contains("\"message\":\"LANA_ERR_UNSUPPORTED_OPERATION: host call 'read_text' is not available in this embedding\""));
}

#[test]
fn run_wires_directory_list() {
    // When the capability is wired, the gate is lifted and the call behaves as
    // native: directory_list("/tmp") succeeds on the host.
    let source = "let e = directory_list(\"/tmp\");\nreturn e;\n";
    let result = run(source, "", "{\"directory_list\":true}");
    assert!(result.starts_with("{\"ok\":true"), "expected ok, got {result}");
}

#[test]
fn run_wires_path_exists() {
    let source = "return path_exists(\"/tmp\");\n";
    assert_eq!(run(source, "", "{\"path_exists\":true}"), "{\"ok\":true,\"result\":\"true\"}");
}

#[test]
fn run_seeded_is_deterministic() {
    // Same source + seed must produce the same result every time.
    let source = "state a = state(p: 0.5, d: 0.0);\nstate b = state(p: 0.3, d: 0.0);\nlet c = append(a, b);\nlet s = sample(c);\nreturn s;\n";
    let first = run(source, "", "{\"seed\":42}");
    let second = run(source, "", "{\"seed\":42}");
    assert_eq!(first, second);
    assert!(first.starts_with("{\"ok\":true"));
}

#[test]
fn run_enforces_instruction_limit() {
    // A program that exceeds the instruction budget fails, as it does natively.
    let source = "let i = 0;\nwhile (i < 1000000) { i = i + 1; }\nreturn i;\n";
    let result = run(source, "", "{\"instruction_limit\":100}");
    assert!(result.starts_with("{\"ok\":false,\"error\":"));
}
