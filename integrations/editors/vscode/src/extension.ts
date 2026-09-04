import { spawnSync } from "node:child_process";
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

function compatibleVersion(command: string): string | undefined {
  const completed = spawnSync(command, ["version"], {
    encoding: "utf8",
    timeout: 5000,
  });
  if (completed.error || completed.status !== 0) {
    return undefined;
  }
  const output = completed.stdout.trim();
  return /^Lana 1\.\d+\.\d+ \(LABC v2,/.test(output) ? output : undefined;
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const configuration = vscode.workspace.getConfiguration("lana");
  const command = configuration.get<string>("server.path", "lana");
  if (!compatibleVersion(command)) {
    const action = await vscode.window.showErrorMessage(
      `Lana Language Support requires Lana 1.x with LABC v2. Could not use: ${command}`,
      "Open Settings",
    );
    if (action === "Open Settings") {
      await vscode.commands.executeCommand(
        "workbench.action.openSettings",
        "lana.server.path",
      );
    }
    return;
  }

  const workspace = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  const serverOptions: ServerOptions = {
    command,
    args: ["lsp"],
    options: workspace ? { cwd: workspace } : undefined,
  };
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "lana" }],
    synchronize: { configurationSection: "lana" },
    markdown: { isTrusted: false },
  };
  client = new LanguageClient(
    "lanaLsp",
    "Lana Language Server",
    serverOptions,
    clientOptions,
  );
  context.subscriptions.push({ dispose: () => void client?.stop() });
  await client.start();
}

export async function deactivate(): Promise<void> {
  await client?.stop();
  client = undefined;
}
