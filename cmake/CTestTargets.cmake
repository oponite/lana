enable_testing()

function(add_lana_c_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE lanaruntime m)
    # Tests rely on assert() for their checks; keep it active in Release builds.
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

add_lana_c_test(lana_runtime_tests tests/unit/test_runtime.c)
target_compile_definitions(lana_runtime_tests PRIVATE LANA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_lana_c_test(lana_mix_tests tests/unit/test_mix.c)
add_lana_c_test(lana_operations2_tests tests/unit/test_operations2.c)
add_lana_c_test(lana_map_tests tests/unit/test_map.c)
add_lana_c_test(lana_support_tests tests/unit/test_support.c)
add_lana_c_test(lana_expect_tests tests/unit/test_expect.c)
add_lana_c_test(lana_inspect_tests tests/unit/test_inspect.c)
add_lana_c_test(lana_validate_tests tests/unit/test_validate.c)
add_lana_c_test(lana_revision_tests tests/unit/test_revision.c)
add_lana_c_test(lana_gc_tests tests/unit/test_gc.c)
add_lana_c_test(lana_error_tests tests/unit/test_errors.c)
add_lana_c_test(lana_public_api_smoke tests/unit/test_public_api.c)
add_lana_c_test(lana_store_tests tests/unit/test_store.c)
add_lana_c_test(lana_codec_tests tests/unit/test_codec.c)
add_lana_c_test(lana_state_codec_tests tests/unit/test_state_codec.c)
add_lana_c_test(lana_claims_tests tests/unit/test_claims.c)
add_lana_c_test(lana_policy_ledger_tests tests/unit/test_policy_ledger.c)
add_lana_c_test(lana_effects_tests tests/unit/test_effects.c)
add_lana_c_test(lana_ledger_coverage_tests tests/unit/test_ledger_coverage.c)
add_lana_c_test(lana_shared_tests tests/unit/test_shared.c)

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
    target_include_directories(lanaruntime_fuzz PUBLIC ${LANA_INCLUDE_DIRS})
    target_link_libraries(lanaruntime_fuzz PUBLIC Threads::Threads)
    target_compile_definitions(lanaruntime_fuzz PRIVATE
        LANA_ADAPTER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
        LANA_ADAPTER_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}")
    target_compile_options(lanaruntime_fuzz PRIVATE
        -Wall -Wextra -Wpedantic -Werror
        -fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer)
    add_executable(lana_bytecode_fuzz tests/unit/fuzz_bytecode.c)
    target_link_libraries(lana_bytecode_fuzz PRIVATE lanaruntime_fuzz m)
    target_compile_options(lana_bytecode_fuzz PRIVATE
        -Wall -Wextra -Wpedantic -Werror -fsanitize=fuzzer,address,undefined
        -fno-omit-frame-pointer)
    target_link_options(lana_bytecode_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()

add_native_compile_failure(native_import_cycle_rejected_a tests/regression/import_cycle_a.lana "LANA_ERR_ASSERTION")
add_native_compile_failure(native_import_cycle_rejected_b tests/regression/import_cycle_b.lana "LANA_ERR_ASSERTION")
add_test(NAME native_import_math COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/import_math.lana")
add_test(NAME native_compile_density COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/examples/belief.lana")
add_test(NAME native_compile_information COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/information.lana")
add_test(NAME native_provenance COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/provenance.lana")
add_test(NAME native_m4_types_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_types_pass.lana")
add_test(NAME native_m4_sample_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_sample_pass.lana")
add_test(NAME native_m4_sample_metadata_run COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_sample_pass.lana")
set_tests_properties(native_m4_sample_metadata_run PROPERTIES PASS_REGULAR_EXPRESSION "host:random;random")
add_test(NAME native_m4_claim_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_claim_pass.lana")
add_test(NAME native_m4_information_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_information_pass.lana")
add_test(NAME native_m4_effect_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_effect_pass.lana")
add_test(NAME native_m4_result_pass COMMAND lana check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_result_pass.lana")
add_test(NAME native_m6_reactive_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m6_reactive_pass.lana")
set_tests_properties(native_m6_reactive_pass PROPERTIES PASS_REGULAR_EXPRESSION "M6_REACTIVE_PASS")
add_test(NAME native_m6_effect_once_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m6_effect_once_pass.lana")
set_tests_properties(native_m6_effect_once_pass PROPERTIES PASS_REGULAR_EXPRESSION "1")
add_test(NAME native_m7_shared_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m7_shared_pass.lana")
set_tests_properties(native_m7_shared_pass PROPERTIES PASS_REGULAR_EXPRESSION "M7_SHARED_PASS")
add_test(NAME native_m10_inspector_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m10_inspector_pass.lana")
set_tests_properties(native_m10_inspector_pass PROPERTIES PASS_REGULAR_EXPRESSION "M10_INSPECTOR_PASS")
add_test(NAME native_m6_unrelated_rejected COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m6_unrelated_rejected.lana")
set_tests_properties(native_m6_unrelated_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_m6_uncertain_effect_rejected COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m6_uncertain_effect_rejected.lana")
set_tests_properties(native_m6_uncertain_effect_rejected PROPERTIES WILL_FAIL TRUE)
add_native_compile_failure(native_m4_parse_span tests/regression/m4_invalid_source.lana "error\\[parse/LANA_ERR_PARSE\\]")
add_native_compile_failure(native_m4_sample_unwrap tests/regression/m4_sample_requires_unwrap.lana "requires explicit sample_value")
add_native_compile_failure(native_m4_claim_match tests/regression/m4_claim_requires_match.lana "requires explicit claim_value")
add_native_compile_failure(native_m4_information_effect tests/regression/m4_information_effect_rejected.lana "unresolved Information branches cannot perform effects")
add_native_compile_failure(native_m4_planned_effect tests/regression/m4_planned_effect_requires_execution.lana "cannot consume an unexecuted planned effect")
add_native_compile_failure(native_m4_result_match tests/regression/m4_result_requires_match.lana "requires explicit result_value")
add_native_compile_failure(native_m4_capability_literal tests/regression/m4_capability_requires_literal.lana "capability name must be a string literal")
add_native_compile_failure(native_m7_read_capability tests/regression/m7_read_requires_capability.lana "shared read requires a read shared capability")
add_native_compile_failure(native_m9_discarded_information tests/regression/m9_discarded_information.lana "discarded unresolved Information value")
add_test(NAME native_mix_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/mix_pass.lana")
add_test(NAME native_isa_ops_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/isa_ops_pass.lana")
set_tests_properties(native_isa_ops_pass PROPERTIES PASS_REGULAR_EXPRESSION "ISA_OPS_PASS")
add_test(NAME native_operations2_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/operations2_pass.lana")
set_tests_properties(native_operations2_pass PROPERTIES PASS_REGULAR_EXPRESSION "OPERATIONS2_PASS")
add_test(NAME native_adt_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/adt_pass.lana")
set_tests_properties(native_adt_pass PROPERTIES PASS_REGULAR_EXPRESSION "ADT_PASS")
add_native_compile_failure(native_adt_nonexhaustive tests/regression/adt_nonexhaustive.lana "match is not exhaustive")
add_native_compile_failure(native_adt_badfields tests/regression/adt_badfields.lana "expects 2 fields")
add_native_compile_failure(native_adt_badvariant tests/regression/adt_badvariant.lana "unknown variant")
add_test(NAME native_probability_constructor COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/probability_constructor.lana")
set_tests_properties(native_probability_constructor PROPERTIES PASS_REGULAR_EXPRESSION "PROBABILITY_CONSTRUCTOR_PASS")
add_test(NAME native_probability_identity
    COMMAND "${CMAKE_COMMAND}"
        -DLANA_VM=$<TARGET_FILE:lanavm>
        -DLANA_COMPILER=${LANA_NATIVE_COMPILER}
        -DLANA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DLANA_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/probability-identity
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestProbabilityConstructor.cmake")
add_test(NAME native_run_source COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/examples/general.lana")
add_test(NAME native_datasets_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/datasets_pass.lana")
set_tests_properties(native_datasets_pass PROPERTIES PASS_REGULAR_EXPRESSION "DATASETS_PASS")
add_test(NAME native_bootstrap_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/bootstrap_pass.lana")
set_tests_properties(native_bootstrap_pass PROPERTIES PASS_REGULAR_EXPRESSION "BOOTSTRAP_PASS")
add_test(NAME native_surprisal_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/surprisal_pass.lana")
set_tests_properties(native_surprisal_pass PROPERTIES PASS_REGULAR_EXPRESSION "SURPRISAL_PASS")
add_test(NAME native_surprisal_negative_rejected COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/surprisal_negative_rejected.lana")
set_tests_properties(native_surprisal_negative_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_inspect_json COMMAND lana inspect "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/inspect_state_dist.lana")
set_tests_properties(native_inspect_json PROPERTIES PASS_REGULAR_EXPRESSION "\"node_count\":3.*\"transform_count\":1")
add_test(NAME native_inspect_dot COMMAND lana inspect "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/inspect_state_dist.lana" --format dot)
set_tests_properties(native_inspect_dot PROPERTIES PASS_REGULAR_EXPRESSION "digraph state_dist")
add_test(NAME native_external_prediction_data COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/examples/external_prediction_data.lana")
add_test(NAME native_imports COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/import_main.lana")

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
find_program(LANA_PYTHON3 NAMES python3)
if(LANA_PYTHON3)
    add_test(NAME lana_lsp_roundtrip
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lsp.py"
            $<TARGET_FILE:lana>)
    set_tests_properties(lana_lsp_roundtrip PROPERTIES
        PASS_REGULAR_EXPRESSION "LSP_ROUNDTRIP_PASS")
endif()
add_test(NAME lana_project_workflow
    COMMAND "${CMAKE_COMMAND}" -DLANA=$<TARGET_FILE:lana>
        -DROOT=${CMAKE_CURRENT_BINARY_DIR}/project-workflow
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestProjectWorkflow.cmake")
add_test(NAME lana_source_debugger
    COMMAND "${CMAKE_COMMAND}" -DLANA=$<TARGET_FILE:lana>
        -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m10_inspector_pass.lana
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
