//! Native-vs-WASM conformance for the `lana-wasm` entry points.
//!
//! `check`/`run` are `#[wasm_bindgen]` functions, but they remain ordinary
//! Rust functions when the crate is built as an `rlib`, so this test calls
//! them directly and pins the exact JSON they must produce. The same
//! expectations are asserted against the compiled wasm module by
//! `tests/conformance.mjs`, so a change that diverges on either target fails
//! one of the two suites.

use lana_wasm::{check, run};

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
    assert_eq!(run("return 42;\n", ""), "{\"ok\":true,\"result\":\"42\"}");
}

#[test]
fn run_returns_state_dist() {
    let source = "state a = state(p: 0.2, d: 0.0);\nstate b = state(p: 0.3, d: 0.0);\nlet c = append(a, b);\nreturn c;\n";
    assert_eq!(run(source, ""), "{\"ok\":true,\"result\":\"state_dist\"}");
}

#[test]
fn run_passes_input_as_single_argument() {
    let source = "let a = args();\nreturn a[0];\n";
    assert_eq!(run(source, "hello"), "{\"ok\":true,\"result\":\"hello\"}");
}

#[test]
fn run_escapes_string_result() {
    let source = "return \"a\\\"b\\n\";\n";
    assert_eq!(run(source, ""), "{\"ok\":true,\"result\":\"a\\\"b\\n\"}");
}
