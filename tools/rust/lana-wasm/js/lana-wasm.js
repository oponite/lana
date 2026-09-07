// lana-wasm — thin JS wrapper over the generated wasm-bindgen bindings
// (LIP-025). Exposes `check`, `run`, `runBytecode`, `measure`, and `sample`.
//
// The wrapper is a thin layer over the WASM exports; it does not reimplement the
// language. It converts the JSON-string boundary of the Rust entry points into
// plain JS objects and accepts a `capabilities` map naming which gated host
// calls are wired (plus optional `seed`/`instruction_limit`/`memory_limit`).
//
//   const { run, runBytecode, measure, sample } = await import("lana-wasm");
//   const result = run(source);            // compile + run
//   const out = runBytecode(labcBytes);    // run precompiled LABC
//
// Filesystem host calls (`read_text`, `write_text`, `directory_list`,
// `directory_create`, `path_exists`, `write_text_atomic`) are unavailable by
// default and fail with `LANA_ERR_UNSUPPORTED_OPERATION` unless wired:
//
//   run(source, "", { read_text: true, write_text: true });

import { check as _check, run as _run, run_bytecode as _runBytecode } from './lana_wasm.js';

function capabilitiesJson(capabilities) {
    if (!capabilities) return '{}';
    return JSON.stringify(capabilities);
}

/** Compile-check a Lana source program. Returns `{ok: true}` or `{ok:false, error}`. */
export function check(source) {
    return JSON.parse(_check(source));
}

/** Compile and run a Lana source program. Returns `{ok:true, result}` or `{ok:false, error}`. */
export function run(source, input = '', capabilities = {}) {
    return JSON.parse(_run(source, input, capabilitiesJson(capabilities)));
}

/** Run a precompiled LABC bytecode blob. Returns `{ok:true, result}` or `{ok:false, error}`. */
export function runBytecode(labcBytes, input = '', capabilities = {}) {
    return JSON.parse(_runBytecode(labcBytes, input, capabilitiesJson(capabilities)));
}

/** Compile and run a program that measures a state. Alias of `run`. */
export function measure(source, input = '', capabilities = {}) {
    return run(source, input, capabilities);
}

/** Compile and run a program that samples a state. Alias of `run`. */
export function sample(source, input = '', capabilities = {}) {
    return run(source, input, capabilities);
}
