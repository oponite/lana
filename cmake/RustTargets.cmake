# Rust CLI build (phase 3 of the Rust runtime boundary).
#
# The C targets in BuildTargets.cmake remain buildable for the differential
# conformance tests, but the installed `lana` / `lanavm` executables now ship
# from the Rust CLI. Build the CLI and FFI library in release mode and install
# the single binary under both names, then keep `lana-compiler.labc` (installed
# by BuildTargets.cmake) so the Rust CLI can locate the self-hosted compiler at
# runtime via `find_compiler`.

find_program(LANA_CARGO cargo REQUIRED)

set(LANA_RUST_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/rust-target")
set(LANA_RUST_BINARY "${CMAKE_CURRENT_BINARY_DIR}/lana-rust${CMAKE_EXECUTABLE_SUFFIX}")
set(LANA_RUST_FFI_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}lana_ffi${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(LANA_RUST_FFI "${CMAKE_CURRENT_BINARY_DIR}/${LANA_RUST_FFI_NAME}")

# Every Rust source under the three layer trees, plus the workspace manifest
# and lock file, is an input to the cargo build. `CONFIGURE_DEPENDS` picks up
# files added after the initial configure.
file(GLOB_RECURSE LANA_RUST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/vm/rust/*.rs"
    "${CMAKE_CURRENT_SOURCE_DIR}/vm/rust/*.toml"
    "${CMAKE_CURRENT_SOURCE_DIR}/vm/rust/*.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/rust/*.rs"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/rust/*.toml"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/*.rs"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/*.toml"
)

set(LANA_RUST_TARGETS host)
set(LANA_RUST_ENV "CARGO_TARGET_DIR=${LANA_RUST_TARGET_DIR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(LANA_RUST_TARGETS)
    foreach(architecture IN LISTS CMAKE_OSX_ARCHITECTURES)
        if(architecture STREQUAL "arm64")
            list(APPEND LANA_RUST_TARGETS aarch64-apple-darwin)
        elseif(architecture STREQUAL "x86_64")
            list(APPEND LANA_RUST_TARGETS x86_64-apple-darwin)
        else()
            message(FATAL_ERROR "Unsupported Rust macOS architecture: ${architecture}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES LANA_RUST_TARGETS)
endif()
if(APPLE AND CMAKE_OSX_DEPLOYMENT_TARGET)
    list(APPEND LANA_RUST_ENV "MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()

set(LANA_RUST_COMMANDS)
set(LANA_RUST_BINARIES)
set(LANA_RUST_LIBRARIES)
foreach(target IN LISTS LANA_RUST_TARGETS)
    set(target_args)
    set(artifact_dir "${LANA_RUST_TARGET_DIR}/release")
    if(NOT target STREQUAL "host")
        set(target_args --target "${target}")
        set(artifact_dir "${LANA_RUST_TARGET_DIR}/${target}/release")
    endif()
    if(APPLE)
        list(APPEND LANA_RUST_COMMANDS
            COMMAND "${CMAKE_COMMAND}" -E env ${LANA_RUST_ENV}
                "${LANA_CARGO}" build --locked --release -p lana-cli ${target_args}
            COMMAND "${CMAKE_COMMAND}" -E env ${LANA_RUST_ENV}
                "${LANA_CARGO}" rustc --locked --release -p lana-ffi --lib ${target_args}
                -- -C "link-arg=-Wl,-install_name,@rpath/${LANA_RUST_FFI_NAME}")
    else()
        list(APPEND LANA_RUST_COMMANDS
            COMMAND "${CMAKE_COMMAND}" -E env ${LANA_RUST_ENV}
                "${LANA_CARGO}" build --locked --release -p lana-cli -p lana-ffi ${target_args})
    endif()
    list(APPEND LANA_RUST_BINARIES "${artifact_dir}/lana-cli${CMAKE_EXECUTABLE_SUFFIX}")
    list(APPEND LANA_RUST_LIBRARIES "${artifact_dir}/${LANA_RUST_FFI_NAME}")
endforeach()

list(LENGTH LANA_RUST_TARGETS target_count)
if(target_count GREATER 1)
    find_program(LANA_LIPO lipo REQUIRED)
    list(APPEND LANA_RUST_COMMANDS
        COMMAND "${LANA_LIPO}" -create ${LANA_RUST_BINARIES} -output "${LANA_RUST_BINARY}"
        COMMAND "${LANA_LIPO}" -create ${LANA_RUST_LIBRARIES} -output "${LANA_RUST_FFI}")
else()
    list(APPEND LANA_RUST_COMMANDS
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${LANA_RUST_BINARIES} "${LANA_RUST_BINARY}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${LANA_RUST_LIBRARIES} "${LANA_RUST_FFI}")
endif()

add_custom_command(
    OUTPUT "${LANA_RUST_BINARY}" "${LANA_RUST_FFI}"
    ${LANA_RUST_COMMANDS}
    DEPENDS
        "${CMAKE_CURRENT_LIST_FILE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
        "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock"
        "${CMAKE_CURRENT_SOURCE_DIR}/VERSION"
        ${LANA_RUST_SOURCES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM
)

add_custom_target(lana_rust ALL DEPENDS "${LANA_RUST_BINARY}" "${LANA_RUST_FFI}")

install(PROGRAMS "${LANA_RUST_BINARY}" DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME lana)
install(PROGRAMS "${LANA_RUST_BINARY}" DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME lanavm)
if(WIN32)
    install(FILES "${LANA_RUST_FFI}" DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
    install(FILES "${LANA_RUST_FFI}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
