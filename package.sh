#!/bin/zsh
set -e

# Lana Distribution Packager
# Bundles runtime, VM, and compiler into a distributable tarball.
#
# Usage: zsh package.sh [BUILD_DIR]
#   BUILD_DIR defaults to build-release; a Release build is required.

BUILD_DIR="${1:-build-release}"
VERSION=$(cat VERSION)
DIST_DIR="dist/lana-${VERSION}"
TARBALL="dist/lana-${VERSION}-macos.tar.gz"

if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "FAILED: build directory '${BUILD_DIR}' not found (run a Release build first)" >&2
    exit 1
fi

mkdir -p "${DIST_DIR}/bin"
mkdir -p "${DIST_DIR}/lib"

echo "Packaging Lana ${VERSION} from ${BUILD_DIR}..."

# 1. Copy Binaries
for artifact in lana lanavm; do
    if [[ ! -f "${BUILD_DIR}/${artifact}" ]]; then
        echo "FAILED: '${BUILD_DIR}/${artifact}' not found" >&2
        exit 1
    fi
    cp "${BUILD_DIR}/${artifact}" "${DIST_DIR}/bin/"
done

# 2. Copy Runtime Library
if [[ ! -f "${BUILD_DIR}/liblanaruntime.a" ]]; then
    echo "FAILED: '${BUILD_DIR}/liblanaruntime.a' not found" >&2
    exit 1
fi
cp "${BUILD_DIR}/liblanaruntime.a" "${DIST_DIR}/lib/"

# 3. Copy Compiler Artifact (compiled bytecode, not the bootstrap source)
if [[ ! -f "${BUILD_DIR}/lana-compiler.labc" ]]; then
    echo "FAILED: '${BUILD_DIR}/lana-compiler.labc' not found" >&2
    exit 1
fi
cp "${BUILD_DIR}/lana-compiler.labc" "${DIST_DIR}/bin/"

# 4. Copy Distribution Metadata
echo "${VERSION}" > "${DIST_DIR}/version.txt"

# Create tarball
rm -f "${TARBALL}"
tar -czf "${TARBALL}" -C dist "lana-${VERSION}"

echo "Package created: ${TARBALL}"
