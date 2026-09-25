set(LANA_COMPILER_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/compiler-bootstrap.lana")
set(LANA_NATIVE_COMPILER "${CMAKE_CURRENT_BINARY_DIR}/lana-compiler.labc")
set(LANA_RUST_CLI "${CMAKE_CURRENT_BINARY_DIR}/lana-rust")

find_program(LANA_CARGO_EXECUTABLE cargo HINTS "$ENV{HOME}/.cargo/bin" REQUIRED)
find_program(LANA_RUSTC_EXECUTABLE rustc HINTS "$ENV{HOME}/.cargo/bin" REQUIRED)

file(GLOB_RECURSE LANA_RUST_INPUTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/vm/rust/*.rs"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/rust/*.rs"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/*.rs"
    "${CMAKE_CURRENT_SOURCE_DIR}/vm/rust/Cargo.toml"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/rust/Cargo.toml"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/Cargo.toml")

if(APPLE AND "arm64" IN_LIST CMAKE_OSX_ARCHITECTURES AND "x86_64" IN_LIST CMAKE_OSX_ARCHITECTURES)
    find_program(LANA_LIPO_EXECUTABLE lipo REQUIRED)
    add_custom_command(
        OUTPUT "${LANA_RUST_CLI}"
        COMMAND "${CMAKE_COMMAND}" -E env
                "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-target"
                "RUSTC=${LANA_RUSTC_EXECUTABLE}"
                "${LANA_CARGO_EXECUTABLE}" build --locked --manifest-path
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/lana-cli/Cargo.toml" --release --target aarch64-apple-darwin
        COMMAND "${CMAKE_COMMAND}" -E env
                "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-target"
                "RUSTC=${LANA_RUSTC_EXECUTABLE}"
                "${LANA_CARGO_EXECUTABLE}" build --locked --manifest-path
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/lana-cli/Cargo.toml" --release --target x86_64-apple-darwin
        COMMAND "${LANA_LIPO_EXECUTABLE}" -create
                "${CMAKE_CURRENT_BINARY_DIR}/cargo-target/aarch64-apple-darwin/release/lana"
                "${CMAKE_CURRENT_BINARY_DIR}/cargo-target/x86_64-apple-darwin/release/lana"
                -output "${LANA_RUST_CLI}"
        DEPENDS Cargo.toml Cargo.lock ${LANA_RUST_INPUTS}
        VERBATIM)
else()
    add_custom_command(
        OUTPUT "${LANA_RUST_CLI}"
        COMMAND "${CMAKE_COMMAND}" -E env
                "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-target"
                "RUSTC=${LANA_RUSTC_EXECUTABLE}"
                "${LANA_CARGO_EXECUTABLE}" build --locked --manifest-path
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/lana-cli/Cargo.toml" --release
        COMMAND "${CMAKE_COMMAND}" -E copy
                "${CMAKE_CURRENT_BINARY_DIR}/cargo-target/release/lana" "${LANA_RUST_CLI}"
        DEPENDS Cargo.toml Cargo.lock ${LANA_RUST_INPUTS}
        VERBATIM)
endif()
add_custom_target(lana_rust_cli ALL DEPENDS "${LANA_RUST_CLI}")

add_custom_command(
    OUTPUT "${LANA_COMPILER_BUNDLE}"
    COMMAND "${CMAKE_COMMAND}"
            -DLANA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DLANA_BUNDLE_OUTPUT=${LANA_COMPILER_BUNDLE}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BundleCompiler.cmake"
    DEPENDS
        compiler/lexer.lana compiler/syntax.lana compiler/parser.lana
        compiler/resolver.lana compiler/ir.lana compiler/emitter.lana
        compiler/main.lana cmake/BundleCompiler.cmake
    VERBATIM)
add_custom_target(lana_compiler_bundle ALL DEPENDS "${LANA_COMPILER_BUNDLE}")

add_custom_command(
    OUTPUT "${LANA_NATIVE_COMPILER}"
    COMMAND "${LANA_RUST_CLI}" asm
            "${CMAKE_CURRENT_SOURCE_DIR}/compiler/bootstrap/compiler.lasm"
            -o "${LANA_NATIVE_COMPILER}"
    DEPENDS lana_rust_cli compiler/bootstrap/compiler.lasm
    VERBATIM)
add_custom_target(lana_native_compiler ALL DEPENDS "${LANA_NATIVE_COMPILER}")
add_custom_target(lana ALL
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${LANA_RUST_CLI}" "${CMAKE_CURRENT_BINARY_DIR}/lana"
    DEPENDS lana_rust_cli lana_native_compiler
    VERBATIM)

install(PROGRAMS "${LANA_RUST_CLI}" DESTINATION bin RENAME lana)
install(FILES "${LANA_NATIVE_COMPILER}" DESTINATION bin)
install(DIRECTORY stdlib/ DESTINATION share/lana/stdlib)
