if(NOT DEFINED LANA_VM OR NOT DEFINED LANA_COMPILER OR NOT DEFINED LANA_SOURCE_DIR OR NOT DEFINED LANA_OUTPUT)
    message(FATAL_ERROR "probability constructor test paths are required")
endif()

# LIP-001 bytecode identity: probability(p) must lower to the same assembly as
# state(p: p, d: 0.0).
execute_process(
    COMMAND "${LANA_VM}" run "${LANA_COMPILER}" --memory-limit-mib 256
            -- "${LANA_SOURCE_DIR}/tests/regression/probability_identity_prob.lana"
            "${LANA_OUTPUT}.prob.lasm"
    RESULT_VARIABLE _prob_result
)
if(NOT _prob_result EQUAL 0)
    message(FATAL_ERROR "probability(p) compile failed: ${_prob_result}")
endif()

execute_process(
    COMMAND "${LANA_VM}" run "${LANA_COMPILER}" --memory-limit-mib 256
            -- "${LANA_SOURCE_DIR}/tests/regression/probability_identity_state.lana"
            "${LANA_OUTPUT}.state.lasm"
    RESULT_VARIABLE _state_result
)
if(NOT _state_result EQUAL 0)
    message(FATAL_ERROR "state(p:, d:) compile failed: ${_state_result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${LANA_OUTPUT}.prob.lasm" "${LANA_OUTPUT}.state.lasm"
    RESULT_VARIABLE _compare_result
)
file(REMOVE "${LANA_OUTPUT}.prob.lasm" "${LANA_OUTPUT}.state.lasm")
if(NOT _compare_result EQUAL 0)
    message(FATAL_ERROR "probability(p) bytecode differs from state(p: p, d: 0.0)")
endif()
