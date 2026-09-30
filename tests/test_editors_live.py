"""Launch real editor clients in disposable workspaces against the built server."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import tempfile
import sys
import plistlib
from workflows import run_process

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("editor", choices=["neovim", "vscode"])
    parser.add_argument("cli", type=Path)
    args = parser.parse_args()
    env = dict(os.environ, LANA_CLI=str(args.cli.resolve()))
    if args.editor == "neovim":
        subprocess.run(["nvim", "--headless", "-u", "NONE", "-l", str(ROOT / "integrations/editors/neovim/test-live.lua")],
                       cwd=ROOT, env=env, timeout=60, check=True)
        return
    with tempfile.TemporaryDirectory(prefix="lana-vscode-") as work:
        work = Path(work).resolve()
        project = work / "project"
        (project / ".vscode").mkdir(parents=True)
        (project / ".vscode/settings.json").write_text(json.dumps({"lana.server.path": str(args.cli.resolve())}))
        (project / "main.lana").write_text('import "./math.lana" as math;\nlet answer = math.twice(2);\n')
        (project / "math.lana").write_text('fn twice(value) { return value * 2; }\n')
        extension = ROOT / "integrations/editors/vscode"
        executable = os.environ.get("VSCODE", "/usr/share/code/code")
        if sys.platform == "darwin" and "VSCODE" not in os.environ:
            contents = Path("/Applications/Visual Studio Code.app/Contents")
            bundle = plistlib.loads((contents / "Info.plist").read_bytes())
            executable = str(contents / "MacOS" / bundle["CFBundleExecutable"])
        env.pop("ELECTRON_RUN_AS_NODE", None)
        env.pop("VSCODE_IPC_HOOK_CLI", None)
        marker = work / "result"
        env["LANA_EDITOR_RESULT"] = str(marker)
        result = run_process([executable, "--user-data-dir", str(work / "user"),
                        "--extensions-dir", str(work / "extensions"), "--disable-extensions", "--disable-gpu", "--wait",
                        "--skip-welcome", "--skip-release-notes", "--disable-workspace-trust",
                        "--extensionDevelopmentPath=" + str(extension),
                        "--extensionTestsPath=" + str(extension / "test-live.cjs"), str(project)],
                       env=env, timeout=90, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == 0, result.stdout
        assert marker.is_file() and marker.read_text() == "VSCODE_LIVE_PASS", result.stdout
        print(marker.read_text())


if __name__ == "__main__":
    main()
