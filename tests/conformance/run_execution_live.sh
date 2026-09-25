#!/usr/bin/env bash
set -euo pipefail

root="${LANA_SOURCE_DIR:?LANA_SOURCE_DIR must name the source tree}"
build="${LANA_BUILD_DIR:?LANA_BUILD_DIR must name the build tree}"
lana="${LANA:?LANA must name the Rust CLI}"
work="$(mktemp -d /tmp/lana-execution.XXXXXX)"
trap 'kill "${server:-}" 2>/dev/null || true; rm -rf "$work"' EXIT

openssl req -x509 -newkey rsa:2048 -keyout "$work/key.pem" -out "$work/cert.pem" \
    -sha256 -days 1 -nodes -subj '/CN=127.0.0.1' \
    -addext 'subjectAltName=IP:127.0.0.1' >/dev/null 2>&1

ROOT="$root" WORK="$work" python3 - <<'PY' &
import http.server
import os
import ssl
import time

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(size).decode("utf-8")
        with open(os.path.join(os.environ["WORK"], "requests.log"), "a", encoding="utf-8") as out:
            out.write(self.path + " " + body + "\n")
        if self.path == "/timeout":
            time.sleep(31)
            return
        self.send_response(302 if self.path == "/redirect" else 204 if self.path == "/ok" else 503)
        if self.path == "/redirect":
            self.send_header("Location", "/ok")
        self.end_headers()
    def log_message(self, *_):
        pass

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(os.path.join(os.environ["WORK"], "port"), "w", encoding="utf-8") as out:
    out.write(str(server.server_port))
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(os.path.join(os.environ["WORK"], "cert.pem"), os.path.join(os.environ["WORK"], "key.pem"))
server.socket = context.wrap_socket(server.socket, server_side=True)
server.serve_forever()
PY
server=$!

for _ in $(seq 1 50); do [ -f "$work/port" ] && break; sleep 0.1; done
port="$(cat "$work/port")"
"$lana" execution-config init --metadata "$work/metadata.lxe" --key "$work/key" \
    --capability-id execution-live --origin "https://127.0.0.1:$port" \
    --credential-key-id live --ca-file "$work/cert.pem"
export LANA_EXECUTION_METADATA="$work/metadata.lxe"
export LANA_EXECUTION_KEY="$work/key"
export LANA_EXECUTION_CREDENTIAL_live="Bearer test-token"
export LANA_COMPILER_LABC="$build/lana-compiler.labc"
export LANA_STDLIB_DIR="$root/stdlib"
cd "$work"
"$lana" run "$root/tests/regression/execution_live_success_pass.lana"
"$lana" run "$root/tests/regression/execution_live_failure_pass.lana"
test "$(wc -l < "$work/requests.log" | tr -d ' ')" = 2
grep -Fx '/ok {"event":"approved"}' "$work/requests.log"
grep -Fx '/fail {"event":"rejected"}' "$work/requests.log"

LANA="$lana" ROOT="$root" WORK="$work" python3 - <<'PYTEST'
import os
import pathlib
import subprocess

work = pathlib.Path(os.environ["WORK"])
base = (pathlib.Path(os.environ["ROOT"]) / "tests/regression/execution_live_success_pass.lana").read_text()
curl_home = work / "curl-home"
curl_home.mkdir()
(curl_home / ".curlrc").write_text('insecure\nlocation\nretry = 3\n')
env = dict(os.environ, CURL_HOME=str(curl_home), TMPDIR=str(work))

def run(name, source, error=None, overrides=None):
    path = work / (name + ".lana")
    path.write_text(source.replace("execution-live-success-store", name + "-store"))
    result = subprocess.run([os.environ["LANA"], "run", str(path)], env=dict(env, **(overrides or {})), capture_output=True, text=True, timeout=45)
    output = result.stdout + result.stderr
    assert "test-token" not in output, output
    assert (result.returncode == 0) if error is None else (result.returncode != 0 and error in output), output
    assert not list(work.glob("lana-curl-*")), "credential temporary file leaked"

run("redirect", base.replace('"/ok"', '"/redirect"').replace('== "Succeeded"', '== "Failed"'))
run("glob", base.replace('"/ok"', '"/{ok,fail}"').replace('== "Succeeded"', '== "Failed"'))
run("mismatch", base.replace('let receipt = execution.execute', 'plan = execution.plan_webhook("/different", {event: "approved"});\nlet receipt = execution.execute'), "LANA_ERR_CAPABILITY")
run("timeout", base.replace('"/ok"', '"/timeout"').replace('== "Succeeded"', '== "Unknown"'))
run("timeout", base.replace('"/ok"', '"/timeout"').replace('== "Succeeded"', '== "Unknown"'), "LANA_ERR_CONFLICT")
run("spawn-failure", base.replace('== "Succeeded"', '== "Unknown"'), overrides={"PATH": str(work / "no-tools")})
# A user curlrc containing insecure must not bypass server certificate checks.
subprocess.run([os.environ["LANA"], "execution-config", "init", "--metadata", str(work / "untrusted.lxe"),
                "--key", str(work / "untrusted.key"), "--capability-id", "untrusted",
                "--origin", "https://127.0.0.1:" + (work / "port").read_text(),
                "--credential-key-id", "live"], check=True, capture_output=True)
run("untrusted", base.replace('== "Succeeded"', '== "Unknown"'), overrides={
    "LANA_EXECUTION_METADATA": str(work / "untrusted.lxe"), "LANA_EXECUTION_KEY": str(work / "untrusted.key")})
requests = (work / "requests.log").read_text().splitlines()
assert len(requests) == 5, requests
assert any(line.startswith('/{ok,fail} ') for line in requests), requests
for path in work.rglob("*"):
    if path.is_file() and path.name not in ("key", "key.pem", "metadata.lxe"):
        assert b"Bearer test-token" not in path.read_bytes(), path
print("EXECUTION_BOUNDARIES_PASS")
PYTEST
