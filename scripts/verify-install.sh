#!/usr/bin/env bash

# Lana Installation Verification Script
# Checks for the presence and basic functionality of the Lana toolchain.

echo "--- Lana Installation Verification ---"

# 1. Check lana binary
if ! command -v lana &> /dev/null; then
    echo "FAILED: 'lana' binary not found in PATH"
    exit 1
fi
LANA_BIN=$(command -v lana)
echo "PASSED: 'lana' binary found ($LANA_BIN)"

# 2. Version reporting
VERSION=$(lana version 2>&1)
if [[ -z "$VERSION" ]]; then
    echo "FAILED: 'lana version' returned no output"
    exit 1
fi
echo "PASSED: 'lana version' works ($VERSION)"

# 3. Source execution check
# Create a trivial program in a temp dir and run it
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
cat << 'S_EOF' > "${TMP_DIR}/verify.lana"
let value = 1;
print(value);
S_EOF

if ! lana run "${TMP_DIR}/verify.lana" | grep -q "1"; then
    echo "FAILED: 'lana run' did not produce expected output"
    exit 1
fi
echo "PASSED: 'lana run' verified"

# 4. Compiler artifact check
# The compiler bytecode must sit next to the lana binary.
COMPILER_PATH="$(dirname "$LANA_BIN")/lana-compiler.labc"
if [[ ! -f "$COMPILER_PATH" ]]; then
    echo "FAILED: 'lana-compiler.labc' not found next to lana binary"
    exit 1
fi
echo "PASSED: Compiler artifact found"

echo "--- ALL CHECKS PASSED ---"
exit 0
