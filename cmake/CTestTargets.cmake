enable_testing()

function(add_lana_c_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE lanaruntime m)
    # Tests rely on assert() for their checks; keep it active in Release builds.
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

add_lana_c_test(lana_runtime_tests tests/c/test_runtime.c)
target_compile_definitions(lana_runtime_tests PRIVATE LANA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_lana_c_test(lana_gc_tests tests/c/test_gc.c)
add_lana_c_test(lana_error_tests tests/c/test_errors.c)
add_lana_c_test(lana_public_api_smoke tests/c/test_public_api.c)
add_lana_c_test(lana_store_tests tests/c/test_store.c)
add_lana_c_test(lana_codec_tests tests/c/test_codec.c)
add_lana_c_test(lana_state_codec_tests tests/c/test_state_codec.c)
add_lana_c_test(lana_claims_tests tests/c/test_claims.c)
add_lana_c_test(lana_policy_ledger_tests tests/c/test_policy_ledger.c)
add_lana_c_test(lana_effects_tests tests/c/test_effects.c)
add_lana_c_test(lana_ledger_coverage_tests tests/c/test_ledger_coverage.c)
add_lana_c_test(lana_shared_tests tests/c/test_shared.c)

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

add_native_compile_failure(native_import_cycle_rejected_a tests/native/import_cycle_a.lana "LANA_ERR_ASSERTION")
add_native_compile_failure(native_import_cycle_rejected_b tests/native/import_cycle_b.lana "LANA_ERR_ASSERTION")
add_test(NAME native_import_math COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/import_math.lana")
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

add_test(NAME native_compiler_bootstrap
    COMMAND "${CMAKE_COMMAND}"
        -DLANA_VM=$<TARGET_FILE:lanavm_release>
        -DLANA_COMPILER=${LANA_NATIVE_COMPILER}
        -DLANA_BUNDLE=${LANA_COMPILER_BUNDLE}
        -DLANA_REFERENCE=${CMAKE_CURRENT_SOURCE_DIR}/compiler/bootstrap/compiler.lasm
        -DLANA_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/compiler-selfcheck.lasm
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyNativeBootstrap.cmake")
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
if(CMAKE_OSX_ARCHITECTURES)
    set(LANA_INSTALL_EXPECTED_ARCH -DEXPECTED_ARCH=${CMAKE_OSX_ARCHITECTURES})
endif()
add_test(NAME lana_local_install
    COMMAND "${CMAKE_COMMAND}"
        -DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -DROOT=${CMAKE_CURRENT_BINARY_DIR}/local-install
        -DVERIFY_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify-install.sh
        ${LANA_INSTALL_EXPECTED_ARCH}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestLocalInstall.cmake")
