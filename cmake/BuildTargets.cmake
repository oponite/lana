set(LANA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/vm/include
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/include)

set(LANA_RUNTIME_SOURCES
    vm/c/gc.c
    vm/c/error.c
    runtime/c/sha256.c
    vm/c/state.c
    vm/c/value.c
    runtime/c/data.c
    vm/c/bytecode.c
    vm/c/vm.c
    runtime/c/shared.c
    tools/c/project.c
    tools/c/lsp.c
    tools/c/json.c
    tools/c/compiler_service.c
    vm/c/assembler.c
    runtime/c/codec.c
    runtime/c/store.c
    runtime/c/state_codec.c
    runtime/c/policy.c
    runtime/c/ledger.c
    runtime/c/claims.c
    runtime/c/vendor/tweetnacl.c
    runtime/c/vendor/tweetnacl_random.c
    runtime/c/effects.c
    runtime/c/adapters.c
)

set_source_files_properties(runtime/c/vendor/tweetnacl.c PROPERTIES
    COMPILE_OPTIONS "-Wno-sign-compare;-Wno-unterminated-string-initialization")

add_library(lanaruntime STATIC ${LANA_RUNTIME_SOURCES})
target_include_directories(lanaruntime PUBLIC ${LANA_INCLUDE_DIRS})
target_link_libraries(lanaruntime PUBLIC Threads::Threads)
target_compile_options(lanaruntime PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(NOT APPLE)
    target_compile_definitions(lanaruntime PRIVATE _POSIX_C_SOURCE=200809L)
endif()
# Adapter facade locates dlopen plugins in the build directory.
target_compile_definitions(lanaruntime PRIVATE
    LANA_ADAPTER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    LANA_ADAPTER_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}")

if(LANA_ENABLE_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(lanaruntime PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(lanaruntime PUBLIC -fsanitize=address,undefined)
endif()
if(LANA_ENABLE_TSAN AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(lanaruntime PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(lanaruntime PUBLIC -fsanitize=thread)
endif()

add_executable(lanavm tools/c/cli.c)
target_link_libraries(lanavm PRIVATE lanaruntime m)
target_compile_options(lanavm PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_library(lanaruntime_release STATIC ${LANA_RUNTIME_SOURCES})
target_include_directories(lanaruntime_release PUBLIC ${LANA_INCLUDE_DIRS})
target_compile_options(lanaruntime_release PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-format-truncation -O3 -DNDEBUG)
if(NOT APPLE)
    target_compile_definitions(lanaruntime_release PRIVATE _POSIX_C_SOURCE=200809L)
endif()
target_compile_definitions(lanaruntime_release PRIVATE
    LANA_ADAPTER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    LANA_ADAPTER_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_executable(lanavm_release tools/c/cli.c)
target_link_libraries(lanavm_release PRIVATE lanaruntime_release m)
target_compile_options(lanavm_release PRIVATE -Wall -Wextra -Wpedantic -Werror -O3 -DNDEBUG)

set(LANA_COMPILER_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/compiler-bootstrap.lana")
set(LANA_NATIVE_COMPILER "${CMAKE_CURRENT_BINARY_DIR}/lana-compiler.labc")
set(LANA_RUST_CLI "${CMAKE_CURRENT_BINARY_DIR}/lana-rust")

find_program(LANA_CARGO_EXECUTABLE cargo
    HINTS "$ENV{HOME}/.cargo/bin"
    REQUIRED)
find_program(LANA_RUSTC_EXECUTABLE rustc
    HINTS "$ENV{HOME}/.cargo/bin"
    REQUIRED)

if(APPLE AND "arm64" IN_LIST CMAKE_OSX_ARCHITECTURES AND "x86_64" IN_LIST CMAKE_OSX_ARCHITECTURES)
    find_program(LANA_LIPO_EXECUTABLE lipo REQUIRED)
    add_custom_command(
        OUTPUT "${LANA_RUST_CLI}"
        COMMAND "${CMAKE_COMMAND}" -E env
                "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-target"
                "RUSTC=${LANA_RUSTC_EXECUTABLE}"
                "${LANA_CARGO_EXECUTABLE}" build --manifest-path
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/lana-cli/Cargo.toml" --release --target aarch64-apple-darwin
        COMMAND "${CMAKE_COMMAND}" -E env
                "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-target"
                "RUSTC=${LANA_RUSTC_EXECUTABLE}"
                "${LANA_CARGO_EXECUTABLE}" build --manifest-path
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/lana-cli/Cargo.toml" --release --target x86_64-apple-darwin
        COMMAND "${LANA_LIPO_EXECUTABLE}" -create
                "${CMAKE_CURRENT_BINARY_DIR}/cargo-target/aarch64-apple-darwin/release/lana"
                "${CMAKE_CURRENT_BINARY_DIR}/cargo-target/x86_64-apple-darwin/release/lana"
                -output "${LANA_RUST_CLI}"
        DEPENDS
            Cargo.toml Cargo.lock
            vm/rust/lana-bytecode/src/lib.rs vm/rust/lana-bytecode/src/opcode.rs
            vm/rust/lana-vm/src/lib.rs vm/rust/lana-vm/src/vm.rs
            runtime/rust/lana-runtime/src/lib.rs
            tools/rust/lana-cli/src/main.rs
        VERBATIM
    )
else()
    add_custom_command(
        OUTPUT "${LANA_RUST_CLI}"
        COMMAND "${CMAKE_COMMAND}" -E env
                "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo-target"
                "RUSTC=${LANA_RUSTC_EXECUTABLE}"
                "${LANA_CARGO_EXECUTABLE}" build --manifest-path
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/lana-cli/Cargo.toml" --release
        COMMAND "${CMAKE_COMMAND}" -E copy
                "${CMAKE_CURRENT_BINARY_DIR}/cargo-target/release/lana" "${LANA_RUST_CLI}"
        DEPENDS
            Cargo.toml Cargo.lock
            vm/rust/lana-bytecode/src/lib.rs vm/rust/lana-bytecode/src/opcode.rs
            vm/rust/lana-vm/src/lib.rs vm/rust/lana-vm/src/vm.rs
            runtime/rust/lana-runtime/src/lib.rs
            tools/rust/lana-cli/src/main.rs
        VERBATIM
    )
endif()
add_custom_target(lana_rust_cli ALL DEPENDS "${LANA_RUST_CLI}")

add_executable(lana tools/c/cli.c)
target_link_libraries(lana PRIVATE lanaruntime m)
target_compile_options(lana PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(lana PRIVATE LANA_VERSION="${LANA_VERSION}")
target_compile_definitions(lanavm PRIVATE LANA_VERSION="${LANA_VERSION}")
target_compile_definitions(lanavm_release PRIVATE LANA_VERSION="${LANA_VERSION}")

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
    VERBATIM
)
add_custom_target(lana_compiler_bundle ALL DEPENDS "${LANA_COMPILER_BUNDLE}")

add_custom_command(
    OUTPUT "${LANA_NATIVE_COMPILER}"
    COMMAND "${LANA_RUST_CLI}" asm
            "${CMAKE_CURRENT_SOURCE_DIR}/compiler/bootstrap/compiler.lasm"
            -o "${LANA_NATIVE_COMPILER}"
    DEPENDS lana_rust_cli compiler/bootstrap/compiler.lasm
    VERBATIM
)
add_custom_target(lana_native_compiler ALL DEPENDS "${LANA_NATIVE_COMPILER}")
add_dependencies(lana lana_native_compiler)

install(TARGETS lanaruntime ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS lanavm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(PROGRAMS "${LANA_RUST_CLI}" DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME lana)
install(FILES "${LANA_NATIVE_COMPILER}" DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY stdlib/ DESTINATION ${CMAKE_INSTALL_DATADIR}/lana/stdlib)
# Public headers keep the `lana/` install prefix; the three layer include dirs
# are flattened into a single `${INCLUDEDIR}/lana` namespace.
install(DIRECTORY vm/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lana)
install(DIRECTORY runtime/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lana)
install(DIRECTORY tools/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lana)
