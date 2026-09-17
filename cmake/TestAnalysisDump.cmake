# Golden-output check for the compiler's `--dump-analysis` mode. Runs the
# native compiler bytecode directly and diffs its dataflow state (per-block
# IN/OUT constant-propagation and reachability) text against a checked-in
# fixture.
if(NOT DEFINED LANA_VM OR NOT DEFINED LANA_COMPILER OR NOT DEFINED LANA_INPUT
   OR NOT DEFINED LANA_REFERENCE OR NOT DEFINED LANA_OUTPUT)
    message(FATAL_ERROR "analysis dump test paths are required")
endif()

execute_process(
    COMMAND "${LANA_VM}" run "${LANA_COMPILER}"
            --memory-limit-mib 256 --instruction-limit 100000000
            -- --dump-analysis "${LANA_INPUT}" "${LANA_OUTPUT}"
    RESULT_VARIABLE _analysis_dump_result
)
if(NOT _analysis_dump_result EQUAL 0)
    message(FATAL_ERROR "analysis dump failed: ${_analysis_dump_result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${LANA_REFERENCE}" "${LANA_OUTPUT}"
    RESULT_VARIABLE _analysis_compare_result
)
if(NOT _analysis_compare_result EQUAL 0)
    message(FATAL_ERROR "analysis dump output is stale")
endif()
