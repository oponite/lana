cmake_minimum_required(VERSION 3.20)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" LANA_VERSION
     REGEX "^[0-9]+\\.[0-9]+\\.[0-9]+$")
if(NOT LANA_VERSION)
    message(FATAL_ERROR "Could not read Lana version from VERSION")
endif()
project(lana_vm VERSION ${LANA_VERSION} LANGUAGES C ASM)

include(GNUInstallDirs)
find_package(Threads REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(SS_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(SS_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if(SS_ENABLE_SANITIZERS AND SS_ENABLE_TSAN)
    message(FATAL_ERROR "SS_ENABLE_SANITIZERS and SS_ENABLE_TSAN are mutually exclusive")
endif()

set(SS_RUNTIME_SOURCES
    csrc/error.c
    csrc/state.c
    csrc/value.c
    csrc/data.c
    csrc/bytecode.c
    csrc/vm.c
    csrc/assembler.c
)

list(LENGTH CMAKE_OSX_ARCHITECTURES SS_OSX_ARCH_COUNT)
if(APPLE AND SS_OSX_ARCH_COUNT GREATER 1)
    list(APPEND SS_RUNTIME_SOURCES csrc/asm_probe_fallback.c)
    message(STATUS "Universal build; using the portable C probe")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    list(APPEND SS_RUNTIME_SOURCES asm/arm64/apply_probe.S)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    list(APPEND SS_RUNTIME_SOURCES asm/x86_64/apply_probe.S)
else()
    list(APPEND SS_RUNTIME_SOURCES csrc/asm_probe_fallback.c)
    message(STATUS "No assembly probe for ${CMAKE_SYSTEM_PROCESSOR}; using the portable C probe")
endif()

add_library(ssruntime STATIC ${SS_RUNTIME_SOURCES})
target_include_directories(ssruntime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ssruntime PUBLIC Threads::Threads)
target_compile_options(ssruntime PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic;-Werror>)

if(SS_ENABLE_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(ssruntime PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(ssruntime PUBLIC -fsanitize=address,undefined)
endif()
if(SS_ENABLE_TSAN AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(ssruntime PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(ssruntime PUBLIC -fsanitize=thread)
endif()

add_executable(ssvm csrc/cli.c)
target_link_libraries(ssvm PRIVATE ssruntime m)
target_compile_options(ssvm PRIVATE -Wall -Wextra -Wpedantic -Werror)

set(LANA_NATIVE_COMPILER "${CMAKE_CURRENT_BINARY_DIR}/lana-compiler.ssb")
target_compile_definitions(ssvm PRIVATE
    LANA_VERSION="${LANA_VERSION}"
    SS_COMPILER_BYTECODE_PATH="${LANA_NATIVE_COMPILER}")

add_executable(lana csrc/cli.c)
target_link_libraries(lana PRIVATE ssruntime m)
target_compile_options(lana PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(lana PRIVATE
    LANA_VERSION="${LANA_VERSION}"
    SS_COMPILER_BYTECODE_PATH="${LANA_NATIVE_COMPILER}")

enable_testing()
add_executable(ss_runtime_tests tests/c/test_runtime.c)
target_link_libraries(ss_runtime_tests PRIVATE ssruntime m)
target_compile_options(ss_runtime_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(ss_runtime_tests PRIVATE SS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME ss_runtime_tests COMMAND ss_runtime_tests)

set(LANA_COMPILER_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/compiler-bootstrap.lana")
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
    COMMAND $<TARGET_FILE:ssvm> asm
            "${CMAKE_CURRENT_SOURCE_DIR}/compiler/bootstrap/compiler.ssa"
            -o "${LANA_NATIVE_COMPILER}"
    DEPENDS ssvm compiler/bootstrap/compiler.ssa
    VERBATIM
)
add_custom_target(lana_native_compiler ALL DEPENDS "${LANA_NATIVE_COMPILER}")
add_dependencies(lana lana_native_compiler)

add_test(NAME native_compile_v4 COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/examples/belief.lana")
add_test(NAME native_compile_information_v5 COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/information.lana")
add_test(NAME native_run_source COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/examples/general.lana")
add_test(NAME native_imports COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/import_main.lana")
add_test(NAME native_import_cycle_rejected COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/import_cycle_a.lana")
set_tests_properties(native_import_cycle_rejected PROPERTIES WILL_FAIL TRUE)
add_test(
    NAME native_compiler_bootstrap
    COMMAND "${CMAKE_COMMAND}"
        -DLANA_VM=$<TARGET_FILE:ssvm>
        -DLANA_COMPILER=${LANA_NATIVE_COMPILER}
        -DLANA_BUNDLE=${LANA_COMPILER_BUNDLE}
        -DLANA_REFERENCE=${CMAKE_CURRENT_SOURCE_DIR}/compiler/bootstrap/compiler.ssa
        -DLANA_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/compiler-selfcheck.ssa
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyNativeBootstrap.cmake"
)

install(TARGETS ssvm lana RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES "${LANA_NATIVE_COMPILER}" DESTINATION ${CMAKE_INSTALL_BINDIR})
