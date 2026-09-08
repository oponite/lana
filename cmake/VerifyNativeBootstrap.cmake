if(NOT DEFINED LANA_VM OR NOT DEFINED LANA_COMPILER OR NOT DEFINED LANA_BUNDLE OR NOT DEFINED LANA_REFERENCE OR NOT DEFINED LANA_OUTPUT)
    message(FATAL_ERROR "native bootstrap verification paths are required")
endif()

# Use Release VM for bootstrap speed; LANA_VM can be overridden to lanavm_release
execute_process(
    COMMAND "${LANA_VM}" run "${LANA_COMPILER}" --memory-limit-mib 256 --instruction-limit 50000000 -- "${LANA_BUNDLE}" "${LANA_OUTPUT}"
    RESULT_VARIABLE _lana_compile_result
)
if(NOT _lana_compile_result EQUAL 0)
    message(FATAL_ERROR "native compiler bootstrap pass failed: ${_lana_compile_result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${LANA_REFERENCE}" "${LANA_OUTPUT}"
    RESULT_VARIABLE _lana_compare_result
)
if(NOT _lana_compare_result EQUAL 0)
    message(FATAL_ERROR "native compiler bootstrap output is stale")
endif()

set(_lana_repeat_output "${LANA_OUTPUT}.repeat")
execute_process(
    COMMAND "${LANA_VM}" run "${LANA_COMPILER}" --memory-limit-mib 256 --instruction-limit 50000000 -- "${LANA_BUNDLE}" "${_lana_repeat_output}"
    RESULT_VARIABLE _lana_repeat_result
)
if(NOT _lana_repeat_result EQUAL 0)
    message(FATAL_ERROR "native compiler bootstrap repeat pass failed: ${_lana_repeat_result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${LANA_OUTPUT}" "${_lana_repeat_output}"
    RESULT_VARIABLE _lana_repeat_compare_result
)
file(REMOVE "${_lana_repeat_output}")
if(NOT _lana_repeat_compare_result EQUAL 0)
    message(FATAL_ERROR "native compiler bootstrap output is not deterministic")
endif()
