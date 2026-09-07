// Native-vs-WASM conformance for the compiled `lana-wasm` module.
//
// Mirrors `tests/conformance.rs`: the same expectations are asserted against
// the actual wasm32 build, so a divergence between the native and wasm targets
// fails one of the two suites. Run via `tests/run-wasm-conformance.sh`, which
// builds the wasm crate, generates the nodejs bindings, and invokes this file.

import { strict as assert } from 'node:assert';

const modulePath = process.env.LANA_WASM_JS;
if (!modulePath) {
    throw new Error('LANA_WASM_JS must point at the generated lana_wasm.js bindings');
}

const { check, run, run_bytecode } = await import(modulePath);

// Serialize a minimal LABC blob: LOAD_CONST R0 42; RETURN R0; HALT.
function labcReturn42() {
    const bytes = [];
    const pushU32 = (n) => {
        const b = new Uint8Array(4);
        new DataView(b.buffer).setUint32(0, n, true);
        bytes.push(...b);
    };
    bytes.push(0x4c, 0x41, 0x42, 0x43); // "LABC"
    pushU32(2); // version
    pushU32(1); // constants
    pushU32(0); // functions
    pushU32(3); // instructions
    pushU32(0); // entry
    bytes.push(1); // ValueType::Number
    const bits = new Uint8Array(8);
    new DataView(bits.buffer).setFloat64(0, 42.0, true);
    bytes.push(...bits);
    // LOAD_CONST R0 42 (opcode 1)
    bytes.push(1);
    pushU32(0); pushU32(0); pushU32(0); pushU32(0); pushU32(1);
    // RETURN R0 (opcode 29)
    bytes.push(29);
    pushU32(0); pushU32(0); pushU32(0); pushU32(0); pushU32(2);
    // HALT (opcode 31)
    bytes.push(31);
    pushU32(0); pushU32(0); pushU32(0); pushU32(0); pushU32(3);
    return new Uint8Array(bytes);
}

assert.equal(
    check('state a = state(p: 0.5, d: 0.0);\nlet p = measure a as probability;\nprint(p);\n'),
    '{"ok":true}',
);

const invalid = check('let x = ;\n');
assert.ok(invalid.startsWith('{"ok":false,"error":{"line":'));
assert.ok(invalid.includes('"message":"parse error at line 1 column 9: expected expression, got symbol ;"'));

assert.equal(run('return 42;\n', '', ''), '{"ok":true,"result":"42"}');

assert.equal(
    run('state a = state(p: 0.2, d: 0.0);\nstate b = state(p: 0.3, d: 0.0);\nlet c = append(a, b);\nreturn c;\n', '', ''),
    '{"ok":true,"result":"state_dist"}',
);

assert.equal(run('let a = args();\nreturn a[0];\n', 'hello', ''), '{"ok":true,"result":"hello"}');

const escapedSource = 'return "a\\"b\\n";\n';
const escapedExpected = '{"ok":true,"result":"a\\"b\\n"}';
assert.equal(run(escapedSource, '', ''), escapedExpected);

// run_bytecode executes a precompiled LABC blob.
assert.equal(run_bytecode(labcReturn42(), '', ''), '{"ok":true,"result":"42"}');
assert.ok(run_bytecode(new Uint8Array([0xff, 0xff, 0xff]), '', '').startsWith('{"ok":false,"error":'));

// A LABC blob holding a single HOST_CALL with the given host id. The gating
// scan in run_chunk covers compiled chunks, so a call to a gated network host
// is rejected before execution.
function labcHostCall(hostId) {
    const bytes = [];
    const pushU32 = (n) => {
        const b = new Uint8Array(4);
        new DataView(b.buffer).setUint32(0, n, true);
        bytes.push(...b);
    };
    bytes.push(0x4c, 0x41, 0x42, 0x43); // "LABC"
    pushU32(2); // version
    pushU32(1); // constants
    pushU32(0); // functions
    pushU32(1); // instructions
    pushU32(0); // entry
    bytes.push(1); // ValueType::Number
    const zero = new Uint8Array(8);
    bytes.push(...zero);
    // HOST_CALL (opcode 39): a=8 (dest), b=hostId, c=0, imm=0, line=1
    bytes.push(39);
    pushU32(8);
    pushU32(hostId);
    pushU32(0);
    pushU32(0);
    pushU32(1);
    return new Uint8Array(bytes);
}

// LIP-025 §3: networking host calls are gated by default; wiring lifts the gate.
const netGated = run_bytecode(labcHostCall(160), '', '');
assert.ok(netGated.startsWith('{"ok":false,"error":{"line":'));
assert.ok(netGated.includes('LANA_ERR_UNSUPPORTED_OPERATION'));
assert.ok(netGated.includes("host call 'http_get' is not available"));

const socketGated = run_bytecode(labcHostCall(162), '', '');
assert.ok(socketGated.includes('LANA_ERR_UNSUPPORTED_OPERATION'));
assert.ok(socketGated.includes("host call 'socket_connect' is not available"));

const netWired = run_bytecode(labcHostCall(160), '', '{"http_get":true}');
assert.ok(!netWired.includes('LANA_ERR_UNSUPPORTED_OPERATION'));

// Host-call gating: a gated FS call fails with LANA_ERR_UNSUPPORTED_OPERATION
// when not wired, byte-identical to the native VM.
const gated = run('let e = directory_list("/tmp");\nreturn e;\n', '', '');
assert.ok(gated.startsWith('{"ok":false,"error":{"line":'));
assert.ok(gated.includes('"message":"LANA_ERR_UNSUPPORTED_OPERATION: host call \'directory_list\' is not available in this embedding"'));

const gatedRead = run('let t = read_text("/tmp/x");\nreturn t;\n', '', '');
assert.ok(gatedRead.includes('"message":"LANA_ERR_UNSUPPORTED_OPERATION: host call \'read_text\' is not available in this embedding"'));

// Determinism: same source + seed produces the same result.
const seededSource = 'state a = state(p: 0.5, d: 0.0);\nstate b = state(p: 0.3, d: 0.0);\nlet c = append(a, b);\nlet s = sample(c);\nreturn s;\n';
const first = run(seededSource, '', '{"seed":42}');
const second = run(seededSource, '', '{"seed":42}');
assert.equal(first, second);
assert.ok(first.startsWith('{"ok":true'));

// Resource-limit enforcement: exceeding the instruction budget fails.
const limited = run('let i = 0;\nwhile (i < 1000000) { i = i + 1; }\nreturn i;\n', '', '{"instruction_limit":100}');
assert.ok(limited.startsWith('{"ok":false,"error":'));

console.log('wasm conformance: 14 assertions passed');
