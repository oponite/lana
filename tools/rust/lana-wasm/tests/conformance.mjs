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

const { check, run } = await import(modulePath);

assert.equal(
    check('state a = state(p: 0.5, d: 0.0);\nlet p = measure a as probability;\nprint(p);\n'),
    '{"ok":true}',
);

const invalid = check('let x = ;\n');
assert.ok(invalid.startsWith('{"ok":false,"error":{"line":'));
assert.ok(invalid.includes('"message":"parse error at line 1 column 9: expected expression, got symbol ;"'));

assert.equal(run('return 42;\n', ''), '{"ok":true,"result":"42"}');

assert.equal(
    run('state a = state(p: 0.2, d: 0.0);\nstate b = state(p: 0.3, d: 0.0);\nlet c = append(a, b);\nreturn c;\n', ''),
    '{"ok":true,"result":"state_dist"}',
);

assert.equal(run('let a = args();\nreturn a[0];\n', 'hello'), '{"ok":true,"result":"hello"}');

const escapedSource = 'return "a\\"b\\n";\n';
const escapedExpected = '{"ok":true,"result":"a\\"b\\n"}';
assert.equal(run(escapedSource, ''), escapedExpected);

console.log('wasm conformance: 6 assertions passed');
