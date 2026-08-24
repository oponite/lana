set(input "${OUTPUT}.input")
file(WRITE "${input}" "s\nc\n")
execute_process(COMMAND "${LANA}" debug "${SOURCE}" INPUT_FILE "${input}"
                OUTPUT_FILE "${OUTPUT}" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "source debugger failed: ${result}")
endif()
file(READ "${OUTPUT}" transcript)
if(NOT transcript MATCHES "BREAK line=1" OR
   NOT transcript MATCHES "instruction=1" OR
   NOT transcript MATCHES "frames=1")
    message(FATAL_ERROR "source debugger did not step deterministically")
endif()
