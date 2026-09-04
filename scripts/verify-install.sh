#!/bin/bash

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

# 2. Check lanavm binary
if ! command -v lanavm &> /dev/null; then
    echo "FAILED: 'lanavm' binary not found in PATH"
    exit 1
fi
echo "PASSED: 'lanavm' binary found"

# 3. Version reporting
VERSION=$(lana version 2>&1)
if [[ -z "$VERSION" ]]; then
    echo "FAILED: 'lana version' returned no output"
    exit 1
fi
echo "PASSED: 'lana version' works ($VERSION)"

# 4. lanavm version reporting
VM_VERSION=$(lanavm version 2>&1)
if [[ -z "$VM_VERSION" ]]; then
    echo "FAILED: 'lanavm version' returned no output"
    exit 1
fi
echo "PASSED: 'lanavm version' works ($VM_VERSION)"

# 5. Source execution check
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

# 6. Compiler artifact check
# The compiler bytecode must sit next to the lana binary.
COMPILER_PATH="$(dirname "$LANA_BIN")/lana-compiler.labc"
if [[ ! -f "$COMPILER_PATH" ]]; then
    echo "FAILED: 'lana-compiler.labc' not found next to lana binary"
    exit 1
fi
echo "PASSED: Compiler artifact found"

# 7. Runtime library check
# liblanaruntime.a must sit in the lib directory one level up from bin.
LIB_PATH="$(dirname "$(dirname "$LANA_BIN")")/lib/liblanaruntime.a"
if [[ ! -f "$LIB_PATH" ]]; then
    echo "FAILED: 'liblanaruntime.a' not found in lib directory"
    exit 1
fi
echo "PASSED: Runtime library found"

echo "--- ALL CHECKS PASSED ---"
exit 0
