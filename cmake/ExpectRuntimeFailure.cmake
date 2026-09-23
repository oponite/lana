execute_process(COMMAND "${LANA}" run "${SOURCE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "${EXPECT}")
    message(FATAL_ERROR "Expected ${EXPECT}; got ${result}: ${output}${error}")
endif()
