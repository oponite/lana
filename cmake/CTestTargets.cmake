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
add_lana_c_test(lana_tensor_tests tests/unit/test_tensor.c)
add_lana_c_test(lana_linalg_tests tests/unit/test_linalg.c)
add_lana_c_test(lana_state_tensor_tests tests/unit/test_state_tensor.c)
add_lana_c_test(lana_training_tests tests/unit/test_training.c)
add_lana_c_test(lana_inference_tests tests/unit/test_inference.c)
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

# LIP-018 two-way FFI: a test shared library (a deterministic function and a
# deliberately-faulting one) plus the unit test that loads and calls it.
add_library(lana_ffi_test_lib SHARED tests/unit/ffi_test_lib.c)
target_compile_options(lana_ffi_test_lib PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(lana_ffi_test_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_lana_c_test(lana_ffi_tests tests/unit/test_ffi.c)
target_compile_definitions(lana_ffi_tests PRIVATE
    LANA_FFI_TEST_LIB="$<TARGET_FILE:lana_ffi_test_lib>")
# LIP-019 networking: real-network paths (C-only, non-deterministic).
add_lana_c_test(lana_net_tests tests/unit/test_net.c)
# LIP-019 live networking over the .lana std/http surface: http_get with
# provenance rooting plus TLS verify-on/off against a loopback server. Skips
# itself if openssl/python3 are unavailable.
add_test(
    NAME net_live_conformance
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/conformance/differential/run_net_live.sh")
set_tests_properties(net_live_conformance PROPERTIES
    ENVIRONMENT "LANA=$<TARGET_FILE:lana>;LANAVM=$<TARGET_FILE:lanavm>")

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
    target_link_libraries(lanaruntime_fuzz PUBLIC Threads::Threads PkgConfig::FFI OpenSSL::SSL ${LANA_BLAS_LIBS} ${LANA_METAL_LIBS})
    target_compile_definitions(lanaruntime_fuzz PRIVATE
        LANA_ADAPTER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
        LANA_ADAPTER_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}"
        ${LANA_BLAS_DEFINES})
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
add_test(NAME native_m12_capability_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m12_capability_pass.lana")
set_tests_properties(native_m12_capability_pass PROPERTIES PASS_REGULAR_EXPRESSION "M12_CAPABILITY_PASS")
add_test(NAME native_m12_capability_revoked COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m12_capability_revoked.lana")
set_tests_properties(native_m12_capability_revoked PROPERTIES WILL_FAIL TRUE)
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
add_test(NAME native_tensor_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_pass.lana")
set_tests_properties(native_tensor_pass PROPERTIES PASS_REGULAR_EXPRESSION "TENSOR_PASS")
add_test(NAME native_tensor_dtype_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_dtype_pass.lana")
set_tests_properties(native_tensor_dtype_pass PROPERTIES PASS_REGULAR_EXPRESSION "TENSOR_DTYPE_PASS")
add_test(NAME native_tensor_matmul_dtype_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_matmul_dtype_pass.lana")
set_tests_properties(native_tensor_matmul_dtype_pass PROPERTIES PASS_REGULAR_EXPRESSION "TENSOR_MATMUL_DTYPE_PASS")
add_test(NAME native_tensor_matmul_fp32_accum_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_matmul_fp32_accum_pass.lana")
set_tests_properties(native_tensor_matmul_fp32_accum_pass PROPERTIES PASS_REGULAR_EXPRESSION "TENSOR_MATMUL_FP32_ACCUM_PASS")
add_test(NAME native_tensor_reduce_dtype_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_reduce_dtype_pass.lana")
set_tests_properties(native_tensor_reduce_dtype_pass PROPERTIES PASS_REGULAR_EXPRESSION "TENSOR_REDUCE_DTYPE_PASS")
add_test(NAME native_tensor_f16_overflow_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_f16_overflow_pass.lana")
set_tests_properties(native_tensor_f16_overflow_pass PROPERTIES PASS_REGULAR_EXPRESSION "TENSOR_F16_OVERFLOW_PASS")
add_test(NAME native_tensor_dtype_unknown_rejected COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_dtype_unknown_rejected.lana")
set_tests_properties(native_tensor_dtype_unknown_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_tensor_dtype_non_tensor_rejected COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_dtype_non_tensor_rejected.lana")
set_tests_properties(native_tensor_dtype_non_tensor_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_tensor_dtype_named_arg_rejected COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_dtype_named_arg_rejected.lana")
set_tests_properties(native_tensor_dtype_named_arg_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_tensor_boundaries_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_boundaries_pass.lana")
set_tests_properties(native_tensor_boundaries_pass PROPERTIES PASS_REGULAR_EXPRESSION "TENSOR_BOUNDARIES_PASS" TIMEOUT 15)
add_test(NAME native_tensor_index_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_index_pass.lana")
set_tests_properties(native_tensor_index_pass PROPERTIES PASS_REGULAR_EXPRESSION "TENSOR_INDEX_PASS" TIMEOUT 15)
add_test(NAME native_tensor_matmul_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_matmul_pass.lana")
set_tests_properties(native_tensor_matmul_pass PROPERTIES PASS_REGULAR_EXPRESSION "TENSOR_MATMUL_PASS" TIMEOUT 15)
add_test(NAME native_lip011_grad_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/lip011_grad_pass.lana")
set_tests_properties(native_lip011_grad_pass PROPERTIES PASS_REGULAR_EXPRESSION "LIP011_GRAD_PASS" TIMEOUT 15)
add_test(NAME native_lip007_state_tensor_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/lip007_state_tensor_pass.lana")
set_tests_properties(native_lip007_state_tensor_pass PROPERTIES PASS_REGULAR_EXPRESSION "LIP007_STATE_TENSOR_PASS" TIMEOUT 15)
add_native_compile_failure(native_lip011_grad_impure tests/regression/lip011_grad_impure.lana "requires a pure function")
add_test(NAME native_lip006_train_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/lip006_train_pass.lana")
set_tests_properties(native_lip006_train_pass PROPERTIES PASS_REGULAR_EXPRESSION "LIP006_TRAIN_PASS" TIMEOUT 15)
add_test(NAME native_lip006_train_capability COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/lip006_train_capability.lana")
set_tests_properties(native_lip006_train_capability PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_lip014_resume_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/lip014_resume_pass.lana")
set_tests_properties(native_lip014_resume_pass PROPERTIES PASS_REGULAR_EXPRESSION "LIP014_RESUME_PASS" TIMEOUT 15)
add_test(NAME native_lip009_infer_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/lip009_infer_pass.lana")
set_tests_properties(native_lip009_infer_pass PROPERTIES PASS_REGULAR_EXPRESSION "LIP009_INFER_PASS" TIMEOUT 15)
add_test(NAME native_lip009_infer_capability COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/lip009_infer_capability.lana")
set_tests_properties(native_lip009_infer_capability PROPERTIES WILL_FAIL TRUE)
add_native_compile_failure(native_lip006_grad_nondiff tests/regression/lip006_grad_nondiff.lana "requires a differentiable function")
add_test(NAME native_lip013_model_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/lip013_model_pass.lana")
set_tests_properties(native_lip013_model_pass PROPERTIES PASS_REGULAR_EXPRESSION "LIP013_MODEL_PASS" TIMEOUT 15)
add_test(NAME native_lip013_dynamic_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/lip013_dynamic_pass.lana")
set_tests_properties(native_lip013_dynamic_pass PROPERTIES PASS_REGULAR_EXPRESSION "LIP013_DYNAMIC_PASS" TIMEOUT 15)
add_native_compile_failure(native_lip013_connection_mismatch tests/regression/lip013_connection_mismatch.lana "connection shape mismatch")
add_native_compile_failure(native_lip013_dense_weights tests/regression/lip013_dense_weights.lana "Dense weights shape")
add_native_compile_failure(native_lip013_nonexhaustive tests/regression/lip013_nonexhaustive.lana "match is not exhaustive")
add_test(NAME native_lip008_uncertainty_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/lip008_uncertainty_pass.lana")
set_tests_properties(native_lip008_uncertainty_pass PROPERTIES PASS_REGULAR_EXPRESSION "LIP008_UNCERTAINTY_PASS" TIMEOUT 15)
add_native_compile_failure(native_lip008_uncertain_as_tensor tests/regression/lip008_uncertain_as_tensor.lana "requires a certain Tensor")
add_native_compile_failure(native_lip008_uncertainty_on_certain tests/regression/lip008_uncertainty_on_certain.lana "uncertainty on a certain Tensor")
add_native_compile_failure(native_lip008_max_uncertain tests/regression/lip008_max_uncertain.lana "requires a certain Tensor")
add_test(NAME native_m4_gpu_matmul_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m4_gpu_matmul_pass.lana")
set_tests_properties(native_m4_gpu_matmul_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "GPU_MATMUL_PASS" TIMEOUT 15
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_ml_tensor_shapes_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/ml_tensor_shapes_pass.lana")
set_tests_properties(native_ml_tensor_shapes_pass PROPERTIES PASS_REGULAR_EXPRESSION "ML_TENSOR_SHAPES_PASS")
add_test(NAME native_ml_tensor_math_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/ml_tensor_math_pass.lana")
set_tests_properties(native_ml_tensor_math_pass PROPERTIES PASS_REGULAR_EXPRESSION "ML_TENSOR_MATH_PASS")
add_test(NAME native_ml_tensor_device_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/ml_tensor_device_pass.lana")
set_tests_properties(native_ml_tensor_device_pass PROPERTIES PASS_REGULAR_EXPRESSION "ML_TENSOR_DEVICE_PASS")
add_test(NAME native_ml_metal_fit_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/ml_metal_fit_pass.lana")
set_tests_properties(native_ml_metal_fit_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "ML_METAL_FIT_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_ml_metal_revoked COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/ml_metal_revoked.lana")
set_tests_properties(native_ml_metal_revoked PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_ml_metal_mixed_device COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/ml_metal_mixed_device.lana")
set_tests_properties(native_ml_metal_mixed_device PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_tensor_ragged_rejected COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/tensor_ragged_rejected.lana")
set_tests_properties(native_tensor_ragged_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_inspect_json COMMAND lana inspect "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/inspect_state_dist.lana")
set_tests_properties(native_inspect_json PROPERTIES PASS_REGULAR_EXPRESSION "\"node_count\":3.*\"transform_count\":1")
add_test(NAME native_inspect_dot COMMAND lana inspect "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/inspect_state_dist.lana" --format dot)
set_tests_properties(native_inspect_dot PROPERTIES PASS_REGULAR_EXPRESSION "digraph state_dist")
add_test(NAME native_external_prediction_data COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/examples/external_prediction_data.lana")
add_test(NAME native_imports COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/import_main.lana")
add_test(NAME native_m22_generator_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m22_generator_pass.lana")
set_tests_properties(native_m22_generator_pass PROPERTIES PASS_REGULAR_EXPRESSION "M22_GENERATOR_PASS")
add_test(NAME native_m22_generator_exhausted COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m22_generator_exhausted.lana")
set_tests_properties(native_m22_generator_exhausted PROPERTIES PASS_REGULAR_EXPRESSION "M22_GENERATOR_EXHAUSTED")
add_test(NAME native_async_simple_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/async_simple_pass.lana")
set_tests_properties(native_async_simple_pass PROPERTIES PASS_REGULAR_EXPRESSION "ASYNC_SIMPLE_PASS")
add_test(NAME native_async_concurrent_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/async_concurrent_pass.lana")
set_tests_properties(native_async_concurrent_pass PROPERTIES PASS_REGULAR_EXPRESSION "ASYNC_CONCURRENT_PASS")
add_test(NAME native_async_determinism_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/async_determinism_pass.lana")
set_tests_properties(native_async_determinism_pass PROPERTIES PASS_REGULAR_EXPRESSION "ASYNC_DETERMINISM_PASS")
add_test(NAME native_async_future_all_race_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/async_future_all_race_pass.lana")
set_tests_properties(native_async_future_all_race_pass PROPERTIES PASS_REGULAR_EXPRESSION "ASYNC_FUTURE_ALL_RACE_PASS")
add_test(NAME native_async_sleep_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/async_sleep_pass.lana")
set_tests_properties(native_async_sleep_pass PROPERTIES PASS_REGULAR_EXPRESSION "ASYNC_SLEEP_PASS")
add_test(NAME native_m22_set_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m22_set_pass.lana")
set_tests_properties(native_m22_set_pass PROPERTIES PASS_REGULAR_EXPRESSION "M22_SET_PASS")
add_test(NAME native_m22_set_state_rejected COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m22_set_state_rejected.lana")
set_tests_properties(native_m22_set_state_rejected PROPERTIES WILL_FAIL TRUE)
add_test(NAME native_m22_iter_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m22_iter_pass.lana")
set_tests_properties(native_m22_iter_pass PROPERTIES PASS_REGULAR_EXPRESSION "M22_ITER_PASS")
add_test(NAME native_m22_map_filter_reduce_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m22_map_filter_reduce_pass.lana")
set_tests_properties(native_m22_map_filter_reduce_pass PROPERTIES PASS_REGULAR_EXPRESSION "M22_MAP_FILTER_REDUCE_PASS")
add_test(NAME native_m22_comprehension_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m22_comprehension_pass.lana")
set_tests_properties(native_m22_comprehension_pass PROPERTIES PASS_REGULAR_EXPRESSION "M22_COMPREHENSION_PASS")
add_test(NAME native_m23_csv_toml_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m23_csv_toml_pass.lana")
set_tests_properties(native_m23_csv_toml_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "M23_CSV_TOML_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_m23_json_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m23_json_pass.lana")
set_tests_properties(native_m23_json_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "M23_JSON_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_m21_format_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m21_format_pass.lana")
set_tests_properties(native_m21_format_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "M21_FORMAT_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_m21_unicode_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m21_unicode_pass.lana")
set_tests_properties(native_m21_unicode_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "M21_UNICODE_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_m21_regex_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/m21_regex_pass.lana")
set_tests_properties(native_m21_regex_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "M21_REGEX_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_std_import_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/std_import_pass.lana")
set_tests_properties(native_std_import_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "STD_IMPORT_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_std_modules_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/std_modules_pass.lana")
set_tests_properties(native_std_modules_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "STD_MODULES_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_decision_voi_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/decision_voi_pass.lana")
set_tests_properties(native_decision_voi_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "DECISION_VOI_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_decision_voi_invalid COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/decision_voi_invalid.lana")
set_tests_properties(native_decision_voi_invalid PROPERTIES
    WILL_FAIL TRUE
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_ml_regression_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/ml_regression_pass.lana")
set_tests_properties(native_ml_regression_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "ML_REGRESSION_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_ml_invalid_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/ml_invalid_pass.lana")
set_tests_properties(native_ml_invalid_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "ML_INVALID_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_ml_families_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/ml_families_pass.lana")
set_tests_properties(native_ml_families_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "ML_FAMILIES_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_ml_dataset_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/ml_dataset_pass.lana")
set_tests_properties(native_ml_dataset_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "ML_DATASET_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")
add_test(NAME native_ml_kalman_multivariate_pass COMMAND lana run "${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/ml_kalman_multivariate_pass.lana")
set_tests_properties(native_ml_kalman_multivariate_pass PROPERTIES
    PASS_REGULAR_EXPRESSION "ML_KALMAN_MULTIVARIATE_PASS"
    ENVIRONMENT "LANA_STDLIB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/stdlib")

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
    add_test(NAME lana_tensor_source_errors
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor_source.py"
            $<TARGET_FILE:lana>)
    set_tests_properties(lana_tensor_source_errors PROPERTIES
        PASS_REGULAR_EXPRESSION "TENSOR_SOURCE_ERRORS_PASS" TIMEOUT 30)
    add_test(NAME lana_capability_source_errors
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_capability_source.py"
            $<TARGET_FILE:lana>)
    set_tests_properties(lana_capability_source_errors PROPERTIES
        PASS_REGULAR_EXPRESSION "CAPABILITY_SOURCE_ERRORS_PASS" TIMEOUT 30)
    add_test(NAME lana_generator_source_errors
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_generator_source.py"
            $<TARGET_FILE:lana>)
    set_tests_properties(lana_generator_source_errors PROPERTIES
        PASS_REGULAR_EXPRESSION "GENERATOR_SOURCE_ERRORS_PASS" TIMEOUT 30)
    add_test(NAME lana_set_source_errors
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_set_source.py"
            $<TARGET_FILE:lana>)
    set_tests_properties(lana_set_source_errors PROPERTIES
        PASS_REGULAR_EXPRESSION "SET_SOURCE_ERRORS_PASS" TIMEOUT 30)
    add_test(NAME lana_iter_source_errors
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_iter_source.py"
            $<TARGET_FILE:lana>)
    set_tests_properties(lana_iter_source_errors PROPERTIES
        PASS_REGULAR_EXPRESSION "ITER_SOURCE_ERRORS_PASS" TIMEOUT 30)
    add_test(NAME lana_async_source_errors
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_async_source.py"
            $<TARGET_FILE:lana>)
    set_tests_properties(lana_async_source_errors PROPERTIES
        PASS_REGULAR_EXPRESSION "ASYNC_SOURCE_ERRORS_PASS" TIMEOUT 30)
    add_test(NAME lana_comprehension_source_errors
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_comprehension_source.py"
            $<TARGET_FILE:lana>)
    set_tests_properties(lana_comprehension_source_errors PROPERTIES
        PASS_REGULAR_EXPRESSION "COMPREHENSION_SOURCE_ERRORS_PASS" TIMEOUT 30)
    add_test(NAME lana_lsp_roundtrip
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lsp.py"
            $<TARGET_FILE:lana>)
    set_tests_properties(lana_lsp_roundtrip PROPERTIES
        PASS_REGULAR_EXPRESSION "LSP_ROUNDTRIP_PASS")
    add_test(NAME lana_repl_session
        COMMAND ${LANA_PYTHON3}
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_repl.py"
            $<TARGET_FILE:lana>)
    set_tests_properties(lana_repl_session PROPERTIES
        PASS_REGULAR_EXPRESSION "REPL_SESSION_PASS" TIMEOUT 30)
endif()
# WASM conformance (LIP-025): the node and WASI runners assert native-vs-WASM
# byte-identical results and host-call gating. Both are soft gates where a
# missing wasm toolchain (wasm-bindgen CLI / wasmtime) makes the script exit 0
# with a SKIP notice, so they are safe in a non-Rust CI environment.
add_test(NAME lana_wasm_node_conformance
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/lana-wasm/tests/run-wasm-conformance.sh")
set_tests_properties(lana_wasm_node_conformance PROPERTIES TIMEOUT 300)
add_test(NAME lana_wasm_wasi_conformance
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/rust/lana-wasm/tests/run-wasi-conformance.sh")
set_tests_properties(lana_wasm_wasi_conformance PROPERTIES TIMEOUT 300)
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
