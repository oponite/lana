execute_process(COMMAND "${CARGO}" build --locked --manifest-path
    "${ROOT}/runtime/rust/lana-ffi/Cargo.toml" --target-dir "${BUILD}/cargo-target"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Rust FFI build failed")
endif()
execute_process(COMMAND "${CC}" -std=c11 -Wall -Wextra -Wpedantic -Werror -UNDEBUG
    "-I${ROOT}/runtime/include" "-I${ROOT}/vm/include"
    "${ROOT}/tests/unit/test_rust_records.c" "-L${BUILD}/cargo-target/debug"
    -llana_ffi "-Wl,-rpath,${BUILD}/cargo-target/debug" -o "${BUILD}/rust-record-consumer"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Rust FFI C consumer compilation failed")
endif()
execute_process(COMMAND "${BUILD}/rust-record-consumer" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Rust FFI C consumer failed")
endif()
