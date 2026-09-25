enable_testing()

function(add_native_compile_failure name source expected)
    add_test(
        NAME ${name}
        COMMAND "${CMAKE_COMMAND}"
            -DLANA=${LANA_RUST_CLI}
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/${source}
            -DEXPECT=${expected}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExpectCompileFailure.cmake"
    )
endfunction()

add_native_compile_failure(native_import_cycle_rejected_a tests/regression/import_cycle_a.lana "LANA_ERR_ASSERTION")
add_native_compile_failure(native_import_cycle_rejected_b tests/regression/import_cycle_b.lana "LANA_ERR_ASSERTION")
add_test(NAME native_import_math COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/import_math.lana")
add_test(NAME native_compile_density COMMAND "${LANA_RUST_CLI}" check "${CMAKE_CURRENT_SOURCE_DIR}/examples/belief.lana")
add_test(NAME native_compile_information COMMAND "${LANA_RUST_CLI}" check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/information.lana")
add_test(NAME native_provenance COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/provenance.lana")
add_test(NAME native_m4_types_pass COMMAND "${LANA_RUST_CLI}" check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_types_pass.lana")
add_test(NAME native_m4_sample_pass COMMAND "${LANA_RUST_CLI}" check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_sample_pass.lana")
add_test(NAME native_m4_sample_metadata_run COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_sample_pass.lana")
set_tests_properties(native_m4_sample_metadata_run PROPERTIES PASS_REGULAR_EXPRESSION "host:random;random")
add_test(NAME native_m4_claim_pass COMMAND "${LANA_RUST_CLI}" check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_claim_pass.lana")
add_test(NAME native_m4_claim_record COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_claim_pass.lana")
add_test(NAME native_m4_information_pass COMMAND "${LANA_RUST_CLI}" check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_information_pass.lana")
add_test(NAME native_m4_effect_pass COMMAND "${LANA_RUST_CLI}" check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_effect_pass.lana")
add_test(NAME native_m4_result_pass COMMAND "${LANA_RUST_CLI}" check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_result_pass.lana")
add_test(NAME native_m6_reactive_pass COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m6_reactive_pass.lana")
set_tests_properties(native_m6_reactive_pass PROPERTIES PASS_REGULAR_EXPRESSION "M6_REACTIVE_PASS")
add_test(NAME native_m6_effect_once_pass COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m6_effect_once_pass.lana")
set_tests_properties(native_m6_effect_once_pass PROPERTIES PASS_REGULAR_EXPRESSION "1")
add_test(NAME native_m7_shared_pass COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m7_shared_pass.lana")
set_tests_properties(native_m7_shared_pass PROPERTIES PASS_REGULAR_EXPRESSION "M7_SHARED_PASS")
add_test(NAME native_m10_inspector_pass COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m10_inspector_pass.lana")
set_tests_properties(native_m10_inspector_pass PROPERTIES PASS_REGULAR_EXPRESSION "M10_INSPECTOR_PASS")
add_test(NAME native_m6_unrelated_rejected COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m6_unrelated_rejected.lana")
set_tests_properties(native_m6_unrelated_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_m6_uncertain_effect_rejected COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m6_uncertain_effect_rejected.lana")
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
add_test(NAME native_mix_pass COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/mix_pass.lana")
add_test(NAME native_isa_ops_pass COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/isa_ops_pass.lana")
set_tests_properties(native_isa_ops_pass PROPERTIES PASS_REGULAR_EXPRESSION "ISA_OPS_PASS")
add_test(NAME native_operations2_pass COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/operations2_pass.lana")
set_tests_properties(native_operations2_pass PROPERTIES PASS_REGULAR_EXPRESSION "OPERATIONS2_PASS")
add_test(NAME native_adt_pass COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/adt_pass.lana")
set_tests_properties(native_adt_pass PROPERTIES PASS_REGULAR_EXPRESSION "ADT_PASS")
add_native_compile_failure(native_adt_nonexhaustive tests/regression/adt_nonexhaustive.lana "match is not exhaustive")
add_native_compile_failure(native_adt_badfields tests/regression/adt_badfields.lana "expects 2 fields")
add_native_compile_failure(native_adt_badvariant tests/regression/adt_badvariant.lana "unknown variant")
add_test(NAME native_probability_constructor COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/probability_constructor.lana")
set_tests_properties(native_probability_constructor PROPERTIES PASS_REGULAR_EXPRESSION "PROBABILITY_CONSTRUCTOR_PASS")
add_test(NAME native_probability_identity
    COMMAND "${CMAKE_COMMAND}"
        -DLANA_VM=${LANA_RUST_CLI}
        -DLANA_COMPILER=${LANA_NATIVE_COMPILER}
        -DLANA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DLANA_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/probability-identity
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestProbabilityConstructor.cmake")
add_test(NAME native_run_source COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/examples/general.lana")
add_test(NAME native_datasets_pass COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/datasets_pass.lana")
add_test(NAME native_dataset_definite COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/dataset_definite_pass.lana")
foreach(case uncertain_filter uncertain_key)
    add_test(NAME native_dataset_${case}_rejected
        COMMAND "${CMAKE_COMMAND}" -DLANA=${LANA_RUST_CLI}
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/dataset_${case}_rejected.lana
            -DEXPECT=LANA_ERR_UNRESOLVED_VALUE
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExpectRuntimeFailure.cmake")
endforeach()
set_tests_properties(native_datasets_pass PROPERTIES PASS_REGULAR_EXPRESSION "DATASETS_PASS")
add_test(NAME native_bootstrap_pass COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/bootstrap_pass.lana")
set_tests_properties(native_bootstrap_pass PROPERTIES PASS_REGULAR_EXPRESSION "BOOTSTRAP_PASS")
add_test(NAME native_surprisal_pass COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/surprisal_pass.lana")
set_tests_properties(native_surprisal_pass PROPERTIES PASS_REGULAR_EXPRESSION "SURPRISAL_PASS")
add_test(NAME native_surprisal_negative_rejected COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/surprisal_negative_rejected.lana")
set_tests_properties(native_surprisal_negative_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_inspect_json COMMAND "${LANA_RUST_CLI}" inspect "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/inspect_state_dist.lana")
set_tests_properties(native_inspect_json PROPERTIES PASS_REGULAR_EXPRESSION "\"node_count\":3.*\"transform_count\":1")
add_test(NAME native_inspect_dot COMMAND "${LANA_RUST_CLI}" inspect "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/inspect_state_dist.lana" --format dot)
set_tests_properties(native_inspect_dot PROPERTIES PASS_REGULAR_EXPRESSION "digraph state_dist")
add_test(NAME native_external_prediction_data COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/examples/external_prediction_data.lana")
add_test(NAME native_imports COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/import_main.lana")
add_test(NAME native_core_import COMMAND "${LANA_RUST_CLI}" check "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/core_import_only.lana")
add_test(NAME native_core_distribution COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/core_distribution.lana")
add_test(NAME native_core_refinement_map COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/core_refinement_map.lana")
add_test(NAME native_core_refinement_finite COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/core_refinement_finite.lana")
add_native_compile_failure(native_core_possibility_sample_rejected tests/regression/core_possibility_sample_rejected.lana "possibility has no weights")
add_test(NAME native_core_possibility_sample_indirect_rejected
    COMMAND "${CMAKE_COMMAND}" -DLANA=${LANA_RUST_CLI}
        -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/core_possibility_sample_indirect_rejected.lana
        -DEXPECT=LANA_ERR_UNSUPPORTED_OPERATION
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExpectRuntimeFailure.cmake")
add_test(NAME native_execution_plan COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/execution_plan_pass.lana")
add_test(NAME native_execution_plan_absolute_url_rejected COMMAND "${LANA_RUST_CLI}" run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/execution_plan_absolute_url_rejected.lana")
set_tests_properties(native_execution_plan_absolute_url_rejected PROPERTIES WILL_FAIL TRUE)
set_tests_properties(
    native_core_import
    native_core_distribution
    native_core_refinement_finite
    native_execution_plan
    native_execution_plan_absolute_url_rejected
    PROPERTIES ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")

add_test(NAME native_compiler_bootstrap
    COMMAND "${CMAKE_COMMAND}"
        -DLANA_VM=${LANA_RUST_CLI}
        -DLANA_COMPILER=${LANA_NATIVE_COMPILER}
        -DLANA_BUNDLE=${LANA_COMPILER_BUNDLE}
        -DLANA_REFERENCE=${CMAKE_CURRENT_SOURCE_DIR}/compiler/bootstrap/compiler.lasm
        -DLANA_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/compiler-selfcheck.lasm
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyNativeBootstrap.cmake")
add_test(NAME native_future_messages
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/conformance/durable/run_future_messages.sh"
        "${LANA_RUST_CLI}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_tests_properties(native_future_messages PROPERTIES PASS_REGULAR_EXPRESSION "FUTURE_MESSAGES_PASS")
add_test(NAME lana_lsp_protocol
    COMMAND "${CMAKE_COMMAND}" -DLANA=${LANA_RUST_CLI}
        -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/lsp-test-output.txt
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestLsp.cmake")
find_program(LANA_PYTHON3 NAMES python3)
if(LANA_PYTHON3)
    add_test(NAME lana_legacy_bytecode
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/conformance/golden/run_legacy.py"
            ${LANA_RUST_CLI})
    add_test(NAME lana_compiler_output COMMAND ${LANA_PYTHON3}
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_compiler_output.py" ${LANA_RUST_CLI})
    add_test(NAME lana_hf_bridge
        COMMAND ${LANA_PYTHON3} -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/tools/lana-hf/tests" -q)
    add_test(NAME lana_brain_workflow
        COMMAND ${LANA_PYTHON3} "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_brain_workflow.py" ${LANA_RUST_CLI})
    add_test(NAME lana_lsp_roundtrip
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lsp.py"
            ${LANA_RUST_CLI})
    set_tests_properties(lana_lsp_roundtrip PROPERTIES
        PASS_REGULAR_EXPRESSION "LSP_ROUNDTRIP_PASS")
endif()
add_test(NAME lana_project_workflow
    COMMAND "${CMAKE_COMMAND}" -DLANA=${LANA_RUST_CLI}
        -DROOT=${CMAKE_CURRENT_BINARY_DIR}/project-workflow
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestProjectWorkflow.cmake")
add_test(NAME lana_source_debugger
    COMMAND "${CMAKE_COMMAND}" -DLANA=${LANA_RUST_CLI}
        -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m10_inspector_pass.lana
        -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/debugger-test-output.txt
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestDebugger.cmake")
if(UNIX AND LANA_PYTHON3)
    add_test(NAME lana_execution_live
        COMMAND "${CMAKE_COMMAND}" -E env
            "LANA=${LANA_RUST_CLI}"
            "LANA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "LANA_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/conformance/run_execution_live.sh")
    set_tests_properties(lana_execution_live PROPERTIES
        PASS_REGULAR_EXPRESSION "EXECUTION_SUCCESS_PASS;EXECUTION_FAILURE_PASS")
endif()
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

foreach(case recommendation review)
    add_test(NAME native_decision_${case} COMMAND "${LANA_RUST_CLI}" run
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/decision_${case}_pass.lana")
    set_tests_properties(native_decision_${case} PROPERTIES
        ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
endforeach()
