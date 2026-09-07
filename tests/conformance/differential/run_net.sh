#!/usr/bin/env bash
# Differential conformance spot-checks for the LIP-019 networking host calls.
#
# The fixtures exercise the deterministic paths: capability denial, argument
# type rejection, and timeout-as-Result. The timeout fixture connects to a
# local TCP server (started here) that accepts and holds the connection, so
# both VMs observe a read timeout and return the same `{"error": "timeout"}`
# Result. Real external network calls are C-only (non-deterministic) and are
# covered by the C unit test instead.
#
#   cargo build -p lana-cli
#   ./tests/conformance/differential/run_net.sh
#
# The C11 binary is expected at build/lanavm relative to the repo root.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
C11="${C11:-$REPO_ROOT/build/lanavm}"
RUST="${RUST:-$REPO_ROOT/target/debug/lana-cli}"
FIXTURES="$REPO_ROOT/tests/conformance/differential/net"
WORK="$(mktemp -d)"
PORT=18080
SERVER_PID=""
trap 'rm -rf "$WORK"; [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null' EXIT

if [[ ! -x "$C11" ]]; then
    echo "C11 lanavm not found at $C11 (build it first)" >&2
    exit 1
fi
if [[ ! -x "$RUST" ]]; then
    echo "Rust lana-cli not found at $RUST (cargo build -p lana-cli first)" >&2
    exit 1
fi

# Start a local TCP server that accepts and holds connections (never responds),
# so a client with a short timeout observes a read timeout deterministically.
# It writes a READY marker once bound so the runner never races the bind.
READY="$WORK/server.ready"
python3 - "$PORT" "$READY" <<'PY' &
import socket, sys, threading
port = int(sys.argv[1])
ready = sys.argv[2]
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port))
srv.listen(16)
open(ready, "w").close()
def hold(c):
    try:
        while True:
            if not c.recv(4096):
                break
    except OSError:
        pass
    finally:
        c.close()
while True:
    c, _ = srv.accept()
    threading.Thread(target=hold, args=(c,), daemon=True).start()
PY
SERVER_PID=$!
# Wait until the server has bound its listening socket.
for _ in $(seq 1 50); do
    [[ -e "$READY" ]] && break
    sleep 0.1
done

failures=0
count=0
for fixture in "$FIXTURES"/*.lasm; do
    name="$(basename "$fixture" .lasm)"
    count=$((count + 1))

    if ! "$C11" asm "$fixture" -o "$WORK/$name.labc" >"$WORK/$name.asm.out" 2>&1; then
        echo "FAIL $name: assembly failed"
        cat "$WORK/$name.asm.out"
        failures=$((failures + 1))
        continue
    fi

    "$C11" run "$WORK/$name.labc" >"$WORK/$name.c11.out" 2>"$WORK/$name.c11.err"
    c11_exit=$?
    "$RUST" run "$WORK/$name.labc" >"$WORK/$name.rust.out" 2>"$WORK/$name.rust.err"
    rust_exit=$?

    ok=1
    if [[ $c11_exit -ne $rust_exit ]]; then
        echo "FAIL $name: exit $c11_exit (C11) != $rust_exit (Rust)"
        ok=0
    fi
    if ! cmp -s "$WORK/$name.c11.out" "$WORK/$name.rust.out"; then
        echo "FAIL $name: stdout differs"
        diff "$WORK/$name.c11.out" "$WORK/$name.rust.out" | head -20
        ok=0
    fi
    if ! cmp -s "$WORK/$name.c11.err" "$WORK/$name.rust.err"; then
        echo "FAIL $name: stderr differs"
        diff "$WORK/$name.c11.err" "$WORK/$name.rust.err" | head -20
        ok=0
    fi

    if [[ $ok -eq 1 ]]; then
        echo "ok   $name"
    else
        failures=$((failures + 1))
    fi
done

echo
echo "$((count - failures))/$count fixtures match"
[[ $failures -eq 0 ]]
