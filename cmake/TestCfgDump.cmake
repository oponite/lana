# Golden-output check for the compiler's `--dump-cfg` analysis mode. Runs the
# native compiler bytecode directly (the CLI exposes no dump passthrough) and
# diffs its block/edge text against a checked-in fixture.
if(NOT DEFINED LANA_VM OR NOT DEFINED LANA_COMPILER OR NOT DEFINED LANA_INPUT
   OR NOT DEFINED LANA_REFERENCE OR NOT DEFINED LANA_OUTPUT)
    message(FATAL_ERROR "cfg dump test paths are required")
endif()

execute_process(
    COMMAND "${LANA_VM}" run "${LANA_COMPILER}"
            --memory-limit-mib 256 --instruction-limit 100000000
            -- --dump-cfg "${LANA_INPUT}" "${LANA_OUTPUT}"
    RESULT_VARIABLE _cfg_dump_result
)
if(NOT _cfg_dump_result EQUAL 0)
    message(FATAL_ERROR "cfg dump failed: ${_cfg_dump_result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${LANA_REFERENCE}" "${LANA_OUTPUT}"
    RESULT_VARIABLE _cfg_compare_result
)
if(NOT _cfg_compare_result EQUAL 0)
    message(FATAL_ERROR "cfg dump output is stale")
endif()
