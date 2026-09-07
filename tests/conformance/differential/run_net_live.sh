#!/usr/bin/env bash
# LIP-019 live-network conformance (C VM, real sockets).
#
# Unlike the deterministic differential fixtures (run_net.sh), these exercise
# real loopback HTTP/HTTPS: http_get over the .lana std/http surface with
# provenance Information rooting, TLS certificate rejection by default, and the
# verify:false opt-out. Non-deterministic, so C-only (Rust parity is unit-
# tested in lana-vm).
#
#   LANA="$REPO_ROOT/build-gate/lana" \
#   LANAVM="$REPO_ROOT/build-gate/lanavm" \
#   ./tests/conformance/differential/run_net_live.sh
#
# Self-skips (exits 0) if openssl or python3 is unavailable, so it degrades
# gracefully in minimal build environments.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STDLIB="$REPO_ROOT/stdlib"
FIXTURES="$REPO_ROOT/tests/conformance/differential/net_live"
WORK="$(mktemp -d)"
HTTP_PORT=18100
HTTPS_PORT=18101
trap 'rm -rf "$WORK"' EXIT

LANA="${LANA:-$REPO_ROOT/build-gate/lana}"
LANAVM="${LANAVM:-$REPO_ROOT/build-gate/lanavm}"

if [[ ! -x "$LANA" ]] || [[ ! -x "$LANAVM" ]]; then
    echo "SKIP run_net_live: lana compiler/VM not found" >&2
    exit 0
fi
if ! command -v openssl >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP run_net_live: openssl or python3 missing" >&2
    exit 0
fi

# Self-signed certificate for localhost/127.0.0.1.
openssl req -x509 -newkey rsa:2048 -keyout "$WORK/k.pem" -out "$WORK/c.pem" \
    -days 2 -nodes -subj "/CN=localhost" \
    -addext "subjectAltName=IP:127.0.0.1" 2>/dev/null

cat > "$WORK/srv.py" <<'PY'
import socket, threading, sys, ssl
which = sys.argv[1]
port = int(sys.argv[2])
cert = sys.argv[3]
key = sys.argv[4]
ready = sys.argv[5]
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
if which == "https":
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
srv.bind(("127.0.0.1", port))
srv.listen(16)
open(ready, "w").close()
def h(c):
    try:
        c.recv(8192)
        c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 14\r\n\r\nhello from lana")
    except OSError:
        pass
    finally:
        c.close()
while True:
    c, _ = srv.accept()
    if which == "https":
        try:
            c = ctx.wrap_socket(c, server_side=True)
        except (ssl.SSLError, OSError):
            c.close()
            continue
    threading.Thread(target=h, args=(c,), daemon=True).start()
PY

pids=()
start_server() { # $1=kind $2=port
    python3 "$WORK/srv.py" "$1" "$2" "$WORK/c.pem" "$WORK/k.pem" "$WORK/ready.$2" &
    pids+=($!)
}
start_server http $HTTP_PORT
start_server https $HTTPS_PORT
for p in $HTTP_PORT $HTTPS_PORT; do
    for _ in $(seq 1 50); do
        [[ -e "$WORK/ready.$p" ]] && break
        sleep 0.1
    done
done

failures=0
for src in "$FIXTURES"/*.lana; do
    name="$(basename "$src" .lana)"
    if ! LANA_STDLIB_DIR="$STDLIB" "$LANA" compile "$src" -o "$WORK/$name.labc" \
            >"$WORK/$name.compile.log" 2>&1; then
        echo "FAIL $name: compile failed"; cat "$WORK/$name.compile.log"; failures=$((failures + 1)); continue
    fi
    if LANA_STDLIB_DIR="$STDLIB" "$LANAVM" run "$WORK/$name.labc" >"$WORK/$name.run.log" 2>&1; then
        echo "ok   $name"
    else
        echo "FAIL $name: run failed"; cat "$WORK/$name.run.log"; failures=$((failures + 1))
    fi
done

for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done

total="$(ls "$FIXTURES"/*.lana 2>/dev/null | wc -l | tr -d ' ')"
echo
echo "$((total - failures))/$total live fixtures passed"
[[ $failures -eq 0 ]]
