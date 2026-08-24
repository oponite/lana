cmake_minimum_required(VERSION 3.20)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" LANA_VERSION
     REGEX "^[0-9]+\\.[0-9]+\\.[0-9]+$")
if(NOT LANA_VERSION)
    message(FATAL_ERROR "Could not read Lana version from VERSION")
endif()
project(lana_vm VERSION ${LANA_VERSION} LANGUAGES C)

include(GNUInstallDirs)
find_package(Threads REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(LANA_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(LANA_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(LANA_BUILD_FUZZERS "Build libFuzzer targets" OFF)
option(LANA_BUILD_INTEGRATIONS "Build optional native integration facade" OFF)
if(LANA_ENABLE_SANITIZERS AND LANA_ENABLE_TSAN)
    message(FATAL_ERROR "LANA_ENABLE_SANITIZERS and LANA_ENABLE_TSAN are mutually exclusive")
endif()

set(LANA_RUNTIME_SOURCES
    csrc/gc.c
    csrc/error.c
    csrc/state.c
    csrc/value.c
    csrc/data.c
    csrc/bytecode.c
    csrc/vm.c
    csrc/shared.c
    csrc/project.c
    csrc/lsp.c
    csrc/assembler.c
)

add_library(lanaruntime STATIC ${LANA_RUNTIME_SOURCES})
target_include_directories(lanaruntime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lanaruntime PUBLIC Threads::Threads)
target_compile_options(lanaruntime PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic;-Werror>)

if(LANA_BUILD_INTEGRATIONS)
    set_target_properties(lanaruntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(lana_bridge SHARED integrations/native/lana_bridge.c)
    target_include_directories(lana_bridge PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/integrations/native/include)
    target_link_libraries(lana_bridge PRIVATE lanaruntime m)
    target_compile_options(lana_bridge PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(lana_bridge PRIVATE LANA_VERSION="${LANA_VERSION}")
    set_target_properties(lana_bridge PROPERTIES VERSION ${LANA_VERSION} SOVERSION 1)
endif()

if(LANA_ENABLE_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(lanaruntime PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(lanaruntime PUBLIC -fsanitize=address,undefined)
endif()
if(LANA_ENABLE_TSAN AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(lanaruntime PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(lanaruntime PUBLIC -fsanitize=thread)
endif()

add_executable(lanavm csrc/cli.c)
target_link_libraries(lanavm PRIVATE lanaruntime m)
target_compile_options(lanavm PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Release build of lanavm for fast bootstrap (optimized runtime + VM)
add_library(lanaruntime_release STATIC ${LANA_RUNTIME_SOURCES})
target_include_directories(lanaruntime_release PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(lanaruntime_release PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-format-truncation -O3 -DNDEBUG)

add_executable(lanavm_release csrc/cli.c)
target_link_libraries(lanavm_release PRIVATE lanaruntime_release m)
target_compile_options(lanavm_release PRIVATE -Wall -Wextra -Wpedantic -Werror -O3 -DNDEBUG)

set(LANA_NATIVE_COMPILER "${CMAKE_CURRENT_BINARY_DIR}/lana-compiler.labc")
target_compile_definitions(lanavm PRIVATE
    LANA_VERSION="${LANA_VERSION}")
target_compile_definitions(lanavm_release PRIVATE
    LANA_VERSION="${LANA_VERSION}")

add_executable(lana csrc/cli.c)
target_link_libraries(lana PRIVATE lanaruntime m)
target_compile_options(lana PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(lana PRIVATE
    LANA_VERSION="${LANA_VERSION}")

enable_testing()
add_executable(lana_runtime_tests tests/c/test_runtime.c)
target_link_libraries(lana_runtime_tests PRIVATE lanaruntime m)
target_compile_options(lana_runtime_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(lana_runtime_tests PRIVATE LANA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME lana_runtime_tests COMMAND lana_runtime_tests)

add_executable(lana_gc_tests tests/c/test_gc.c)
target_link_libraries(lana_gc_tests PRIVATE lanaruntime m)
target_compile_options(lana_gc_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME lana_gc_tests COMMAND lana_gc_tests)

add_executable(lana_error_tests tests/c/test_errors.c)
target_link_libraries(lana_error_tests PRIVATE lanaruntime m)
target_compile_options(lana_error_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME lana_error_tests COMMAND lana_error_tests)

add_executable(lana_public_api_smoke tests/c/test_public_api.c)
target_link_libraries(lana_public_api_smoke PRIVATE lanaruntime m)
target_compile_options(lana_public_api_smoke PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME lana_public_api_smoke COMMAND lana_public_api_smoke)

add_executable(lana_shared_tests tests/c/test_shared.c)
target_link_libraries(lana_shared_tests PRIVATE lanaruntime m)
target_compile_options(lana_shared_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME lana_shared_tests COMMAND lana_shared_tests)

if(LANA_BUILD_FUZZERS)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "LANA_BUILD_FUZZERS requires Clang")
    endif()
    if(NOT LANA_ENABLE_SANITIZERS)
        message(FATAL_ERROR "LANA_BUILD_FUZZERS requires LANA_ENABLE_SANITIZERS")
    endif()
    add_library(lanaruntime_fuzz STATIC ${LANA_RUNTIME_SOURCES})
    target_include_directories(lanaruntime_fuzz PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(lanaruntime_fuzz PUBLIC Threads::Threads)
    target_compile_options(lanaruntime_fuzz PRIVATE
        -Wall -Wextra -Wpedantic -Werror
        -fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer)
    add_executable(lana_bytecode_fuzz tests/c/fuzz_bytecode.c)
    target_link_libraries(lana_bytecode_fuzz PRIVATE lanaruntime_fuzz m)
    target_compile_options(lana_bytecode_fuzz PRIVATE
        -Wall -Wextra -Wpedantic -Werror -fsanitize=fuzzer,address,undefined
        -fno-omit-frame-pointer)
    target_link_options(lana_bytecode_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()

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
    COMMAND $<TARGET_FILE:lanavm> asm
            "${CMAKE_CURRENT_SOURCE_DIR}/compiler/bootstrap/compiler.lasm"
            -o "${LANA_NATIVE_COMPILER}"
    DEPENDS lanavm compiler/bootstrap/compiler.lasm
    VERBATIM
)
add_custom_target(lana_native_compiler ALL DEPENDS "${LANA_NATIVE_COMPILER}")
add_dependencies(lana lana_native_compiler)

if(LANA_BUILD_INTEGRATIONS)
    add_executable(lana_bridge_tests integrations/native/test_bridge.c)
    target_link_libraries(lana_bridge_tests PRIVATE lana_bridge)
    target_compile_options(lana_bridge_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME lana_native_bridge
        COMMAND "${CMAKE_COMMAND}"
            -DLANAVM=$<TARGET_FILE:lanavm>
            -DCOMPILER=${LANA_NATIVE_COMPILER}
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/integrations/lana/echo_bridge.lana
            -DTEST=$<TARGET_FILE:lana_bridge_tests>
            -DROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestNativeBridge.cmake")
endif()

add_test(NAME native_compile_density COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/examples/belief.lana")
add_test(NAME native_compile_information COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/information.lana")
add_test(NAME native_provenance COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/provenance.lana")
add_test(NAME native_m4_types_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m4_types_pass.lana")
add_test(NAME native_m4_sample_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m4_sample_pass.lana")
add_test(NAME native_m4_sample_metadata_run COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m4_sample_pass.lana")
set_tests_properties(native_m4_sample_metadata_run PROPERTIES PASS_REGULAR_EXPRESSION "host:random;random")
add_test(NAME native_m4_claim_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m4_claim_pass.lana")
add_test(NAME native_m4_information_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m4_information_pass.lana")
add_test(NAME native_m4_effect_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m4_effect_pass.lana")
add_test(NAME native_m4_result_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m4_result_pass.lana")
add_test(NAME native_m6_reactive_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m6_reactive_pass.lana")
set_tests_properties(native_m6_reactive_pass PROPERTIES PASS_REGULAR_EXPRESSION "M6_REACTIVE_PASS")
add_test(NAME native_m6_effect_once_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m6_effect_once_pass.lana")
set_tests_properties(native_m6_effect_once_pass PROPERTIES PASS_REGULAR_EXPRESSION "1")
add_test(NAME native_m7_shared_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m7_shared_pass.lana")
set_tests_properties(native_m7_shared_pass PROPERTIES PASS_REGULAR_EXPRESSION "M7_SHARED_PASS")
add_test(NAME native_m10_inspector_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m10_inspector_pass.lana")
set_tests_properties(native_m10_inspector_pass PROPERTIES PASS_REGULAR_EXPRESSION "M10_INSPECTOR_PASS")
add_test(NAME native_m6_unrelated_rejected COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m6_unrelated_rejected.lana")
set_tests_properties(native_m6_unrelated_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_m6_uncertain_effect_rejected COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m6_uncertain_effect_rejected.lana")
set_tests_properties(native_m6_uncertain_effect_rejected PROPERTIES WILL_FAIL TRUE)

function(add_native_compile_failure name source expected)
    add_test(
        NAME ${name}
        COMMAND "${CMAKE_COMMAND}"
            -DLANA=$<TARGET_FILE:lana>
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/${source}
            -DEXPECT=${expected}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExpectCompileFailure.cmake"
    )
endfunction()

add_native_compile_failure(native_m4_parse_span tests/native/m4_invalid_source.lana "error\\[parse/LANA_ERR_PARSE\\]")
add_native_compile_failure(native_m4_sample_unwrap tests/native/m4_sample_requires_unwrap.lana "requires explicit sample_value")
add_native_compile_failure(native_m4_claim_match tests/native/m4_claim_requires_match.lana "requires explicit claim_value")
add_native_compile_failure(native_m4_information_effect tests/native/m4_information_effect_rejected.lana "unresolved Information branches cannot perform effects")
add_native_compile_failure(native_m4_planned_effect tests/native/m4_planned_effect_requires_execution.lana "cannot consume an unexecuted planned effect")
add_native_compile_failure(native_m4_result_match tests/native/m4_result_requires_match.lana "requires explicit result_value")
add_native_compile_failure(native_m4_capability_literal tests/native/m4_capability_requires_literal.lana "capability name must be a string literal")
add_native_compile_failure(native_m7_read_capability tests/native/m7_read_requires_capability.lana "shared read requires a read shared capability")
add_native_compile_failure(native_m9_discarded_information tests/native/m9_discarded_information.lana "discarded unresolved Information value")
add_test(NAME native_run_source COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/examples/general.lana")
add_test(NAME native_external_prediction_data COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/examples/external_prediction_data.lana")
add_test(NAME native_imports COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/import_main.lana")
add_test(NAME native_import_cycle_rejected COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/import_cycle_a.lana")
set_tests_properties(native_import_cycle_rejected PROPERTIES WILL_FAIL TRUE)
add_test(
    NAME native_compiler_bootstrap
    COMMAND "${CMAKE_COMMAND}"
        -DLANA_VM=$<TARGET_FILE:lanavm_release>
        -DLANA_COMPILER=${LANA_NATIVE_COMPILER}
        -DLANA_BUNDLE=${LANA_COMPILER_BUNDLE}
        -DLANA_REFERENCE=${CMAKE_CURRENT_SOURCE_DIR}/compiler/bootstrap/compiler.lasm
        -DLANA_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/compiler-selfcheck.lasm
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyNativeBootstrap.cmake"
)
add_test(NAME lana_lsp_protocol
    COMMAND "${CMAKE_COMMAND}" -DLANA=$<TARGET_FILE:lana>
        -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/lsp-test-output.txt
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestLsp.cmake")
add_test(NAME lana_project_workflow
    COMMAND "${CMAKE_COMMAND}" -DLANA=$<TARGET_FILE:lana>
        -DROOT=${CMAKE_CURRENT_BINARY_DIR}/project-workflow
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestProjectWorkflow.cmake")
add_test(NAME lana_source_debugger
    COMMAND "${CMAKE_COMMAND}" -DLANA=$<TARGET_FILE:lana>
        -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/tests/native/m10_inspector_pass.lana
        -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/debugger-test-output.txt
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestDebugger.cmake")

install(TARGETS lanaruntime ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS lanavm lana RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES "${LANA_NATIVE_COMPILER}" DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/lana DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(LANA_BUILD_INTEGRATIONS)
    install(TARGETS lana_bridge LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES integrations/native/include/lana/bridge.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lana)
endif()
