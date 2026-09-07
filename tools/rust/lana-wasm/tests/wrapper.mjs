// Tests the `lana-wasm` JS wrapper surface (LIP-025): `run`, `runBytecode`,
// `measure`, `sample`, and the `capabilities` map. The wrapper is copied next to
// the generated bindings by `run-wasm-conformance.sh`; this file imports it
// from `LANA_WASM_WRAPPER_JS`.

import { strict as assert } from 'node:assert';

const wrapperPath = process.env.LANA_WASM_WRAPPER_JS;
if (!wrapperPath) {
    throw new Error('LANA_WASM_WRAPPER_JS must point at the lana-wasm.js wrapper');
}

const { check, run, runBytecode, measure, sample } = await import(wrapperPath);

// run returns a plain object.
assert.deepEqual(run('return 42;\n'), { ok: true, result: '42' });

// input is passed as the single argument.
assert.deepEqual(run('let a = args();\nreturn a[0];\n', 'hello'), { ok: true, result: 'hello' });

// capabilities map: a gated FS call fails with LANA_ERR_UNSUPPORTED_OPERATION
// when not wired.
const gated = run('let e = directory_list("/tmp");\nreturn e;\n');
assert.equal(gated.ok, false);
assert.ok(gated.error.message.includes('LANA_ERR_UNSUPPORTED_OPERATION'));

// runBytecode executes a precompiled LABC blob.
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
assert.deepEqual(runBytecode(labcReturn42()), { ok: true, result: '42' });

// measure and sample are aliases of run.
assert.deepEqual(measure('return 7;\n'), { ok: true, result: '7' });
assert.deepEqual(sample('return 9;\n'), { ok: true, result: '9' });

// check returns a plain object.
assert.deepEqual(check('return 1;\n'), { ok: true });

console.log('wasm wrapper: 8 assertions passed');
