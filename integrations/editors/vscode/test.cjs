const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const grammar = require('./syntaxes/lana.tmLanguage.json');
const patterns = grammar.repository;
assert.equal(new RegExp(patterns.numbers.patterns[0].match).exec('let x = 12.5;')[0], '12.5');
assert.equal(new RegExp(patterns.strings.patterns[0].patterns[0].match).exec('a\\nb')[0], '\\n');
for (const word of ['class', 'interface', 'mutable', 'Self']) {
  assert(new RegExp(patterns.keywords.patterns[1].match).test(word));
}
assert(!new RegExp(patterns.keywords.patterns[1].match).test('classification'));
assert.equal(new RegExp(patterns.functions.patterns[0].match).exec('fn   hello()')[3], 'hello');
assert(grammar.patterns.findIndex(x => x.include === '#functions') <
       grammar.patterns.findIndex(x => x.include === '#keywords'));

async function checkActivation(version, expected) {
  let started = false;
  let rejected = false;
  const exports = {};
  const modules = {
    'node:child_process': { spawnSync: () => ({ status: 0, stdout: version }) },
    vscode: { workspace: { getConfiguration: () => ({ get: () => '/test/lana' }) },
              window: { showErrorMessage: async () => { rejected = true; } } },
    'vscode-languageclient/node': { LanguageClient: class {
      constructor(id, name, server, options) {
        assert.equal(server.command, '/test/lana');
        assert.deepEqual(Array.from(server.args), ['lsp']);
        assert.equal(options.documentSelector[0].language, 'lana');
      }
      async start() { started = true; }
      async stop() {}
    } },
  };
  vm.runInNewContext(fs.readFileSync(`${__dirname}/out/extension.js`, 'utf8'), {
    exports, require: name => { assert(name in modules, name); return modules[name]; },
  });
  await exports.activate({ subscriptions: [] });
  assert.equal(started, expected, version);
  assert.equal(rejected, !expected, version);
  await exports.deactivate();
}
(async () => {
  await checkActivation('Lana 4.1.0 (LABC v2, Rust)', true);
  await checkActivation('Lana 3.0.2 (LABC v2, Rust)', true);
  await checkActivation('Lana 2.1.0 (LABC v2, Rust)', false);
  await checkActivation('unrelated executable', false);
  console.log('PASS: grammar patterns and language server activation');
})().catch(error => { console.error(error); process.exitCode = 1; });
