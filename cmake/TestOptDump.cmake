# Golden-output check for the compiler's `--dump-optimizations` mode. Runs the
# native compiler bytecode directly and diffs its structured optimization
# records (constant folds + branch simplifications) against a checked-in
# fixture.
if(NOT DEFINED LANA_VM OR NOT DEFINED LANA_COMPILER OR NOT DEFINED LANA_INPUT
   OR NOT DEFINED LANA_REFERENCE OR NOT DEFINED LANA_OUTPUT)
    message(FATAL_ERROR "opt dump test paths are required")
endif()

execute_process(
    COMMAND "${LANA_VM}" run "${LANA_COMPILER}"
            --memory-limit-mib 256 --instruction-limit 100000000
            -- --dump-optimizations "${LANA_INPUT}" "${LANA_OUTPUT}"
    RESULT_VARIABLE _opt_dump_result
)
if(NOT _opt_dump_result EQUAL 0)
    message(FATAL_ERROR "opt dump failed: ${_opt_dump_result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${LANA_REFERENCE}" "${LANA_OUTPUT}"
    RESULT_VARIABLE _opt_compare_result
)
if(NOT _opt_compare_result EQUAL 0)
    message(FATAL_ERROR "opt dump output is stale")
endif()
