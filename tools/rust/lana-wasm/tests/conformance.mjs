// Native-vs-WASM conformance for the compiled `lana-wasm` module.
//
// Mirrors `tests/conformance.rs`: the same expectations are asserted against
// the actual wasm32 build, so a divergence between the native and wasm targets
// fails one of the two suites. Run via `tests/run-wasm-conformance.sh`, which
// builds the wasm crate, generates the nodejs bindings, and invokes this file.

import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';

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

assert.deepEqual(JSON.parse(run('import "std/core" as core; return type_of(core.identity_kernel([0, 1]));', '')),
    {ok: true, result: 'kernel'});
assert.deepEqual(JSON.parse(run('fn worker() { return 7; } let task = fork worker(); return join(task);', '')),
    {ok: true, result: '7'});
for (const module of ['core', 'decision', 'execution', 'evaluation', 'future_messages', 'rules', 'trees']) {
    assert.deepEqual(JSON.parse(check(`import "std/${module}" as lib; return 1;`)), {ok: true}, module);
}
assert.equal(JSON.parse(check('import "std/missing" as lib; return 1;')).ok, false);
for (const name of ['core_identity_kernel', 'rules_learn_pass', 'trees_fit_pass', 'evaluation_walk_forward_pass']) {
    const source = readFileSync(new URL(`../../../../tests/regression/${name}.lana`, import.meta.url), 'utf8');
    const result = JSON.parse(run(source, ''));
    assert.equal(result.ok, true, `${name}: ${JSON.stringify(result)}`);
}
for (let i = 0; i < 32; i++) {
    assert.deepEqual(JSON.parse(run('let a = [null]; a[0] = a; return 42;', '')),
        {ok: true, result: '42'});
}
for (const expression of ['now()', 'sleep(1)', 'read_text("/outside")',
    'http_get("http://localhost/", map_new(), 1)', 'execution_capability()',
    'store_open("/outside")', 'socket_connect("127.0.0.1", 80, 1)']) {
    const result = JSON.parse(run(`return ${expression};`, ''));
    assert.equal(result.ok, false, expression);
    assert.match(result.error.message, /unsupported/i, `${expression}: ${JSON.stringify(result)}`);
}
const oom = JSON.parse(run('return zeros([1000000000]);', ''));
assert.equal(oom.ok, false);
assert.match(oom.error.message, /memory|oom/i);
const limit = JSON.parse(run('while (true) {}', ''));
assert.equal(limit.ok, false);
assert.match(limit.error.message, /limit|instruction/i);
assert.equal(run('return 42;', ''), '{"ok":true,"result":"42"}');
console.log('wasm conformance: source, stdlib, tasks, repeated cycles, host errors and limits passed');
