// Run by VS Code's extension test host, not a mocked vscode module.
const assert = require('node:assert/strict');
const path = require('node:path');
const vscode = require('vscode');
exports.run = async () => {
  const root = vscode.workspace.workspaceFolders[0].uri.fsPath;
  const main = vscode.Uri.file(path.join(root, 'main.lana'));
  const dependency = vscode.Uri.file(path.join(root, 'math.lana'));
  const document = await vscode.workspace.openTextDocument(main);
  await vscode.window.showTextDocument(document);
  const extension = vscode.extensions.getExtension('lana-language.lana-language-support');
  assert(extension);
  await extension.activate();
  const position = new vscode.Position(1, 19);
  const definitions = await vscode.commands.executeCommand('vscode.executeDefinitionProvider', main, position);
  assert.equal(definitions[0].uri.fsPath, dependency.fsPath);
  const edits = await vscode.commands.executeCommand('vscode.executeDocumentRenameProvider', main, position, 'double_value');
  assert.equal(edits.get(main).length, 1);
  assert.equal(edits.get(dependency).length, 1);
  require('node:fs').writeFileSync(process.env.LANA_EDITOR_RESULT, 'VSCODE_LIVE_PASS');
  console.log('VSCODE_LIVE_PASS');
};
