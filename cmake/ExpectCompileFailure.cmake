if(NOT DEFINED LANA OR NOT DEFINED SOURCE OR NOT DEFINED EXPECT)
    message(FATAL_ERROR "LANA, SOURCE, and EXPECT are required")
endif()

execute_process(
    COMMAND "${LANA}" check "${SOURCE}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)
set(output "${stdout}${stderr}")

if(result EQUAL 0)
    message(FATAL_ERROR "expected compilation to fail for ${SOURCE}")
endif()
if(NOT output MATCHES "${EXPECT}")
    message(FATAL_ERROR "missing diagnostic '${EXPECT}' in:\n${output}")
endif()
string(REGEX MATCH "${SOURCE}:[0-9]+:[0-9]+-[0-9]+:[0-9]+" source_span "${output}")
if(source_span STREQUAL "")
    message(FATAL_ERROR "missing full source span in:\n${output}")
endif()
