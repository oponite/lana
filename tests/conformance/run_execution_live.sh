#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
lana="${LANA:?LANA must name the Rust CLI}"
work="$(mktemp -d /tmp/lana-execution.XXXXXX)"
trap 'kill "${server:-}" 2>/dev/null || true; rm -rf "$work" "$root/build/execution-live-success-store" "$root/build/execution-live-failure-store"' EXIT

openssl req -x509 -newkey rsa:2048 -keyout "$work/key.pem" -out "$work/cert.pem" \
    -sha256 -days 1 -nodes -subj '/CN=127.0.0.1' \
    -addext 'subjectAltName=IP:127.0.0.1' >/dev/null 2>&1

ROOT="$root" WORK="$work" python3 - <<'PY' &
import http.server
import os
import ssl

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(size).decode("utf-8")
        with open(os.path.join(os.environ["WORK"], "requests.log"), "a", encoding="utf-8") as out:
            out.write(self.path + " " + body + "\n")
        self.send_response(204 if self.path == "/ok" else 503)
        self.end_headers()
    def log_message(self, *_):
        pass

server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
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
export LANA_COMPILER_LABC="$root/build/lana-compiler.labc"
export LANA_STDLIB_DIR="$root/stdlib"
"$lana" run "$root/tests/regression/execution_live_success_pass.lana"
"$lana" run "$root/tests/regression/execution_live_failure_pass.lana"
test "$(wc -l < "$work/requests.log" | tr -d ' ')" = 2
grep -Fx '/ok {"event":"approved"}' "$work/requests.log"
grep -Fx '/fail {"event":"rejected"}' "$work/requests.log"
