if(NOT DEFINED LANAVM OR NOT DEFINED COMPILER OR NOT DEFINED SOURCE OR
   NOT DEFINED TEST OR NOT DEFINED ROOT)
    message(FATAL_ERROR "native bridge test paths are required")
endif()

set(assembly "${ROOT}/bridge-test.lasm")
set(bytecode "${ROOT}/bridge-test.labc")
set(request "${ROOT}/bridge-request.json")
set(response "${ROOT}/bridge-response.json")

execute_process(
    COMMAND "${LANAVM}" run "${COMPILER}" -- "${SOURCE}" "${assembly}"
    RESULT_VARIABLE compile_result
)
if(NOT compile_result EQUAL 0)
    message(FATAL_ERROR "bridge source compilation failed")
endif()
execute_process(
    COMMAND "${LANAVM}" asm "${assembly}" -o "${bytecode}"
    RESULT_VARIABLE assemble_result
)
if(NOT assemble_result EQUAL 0)
    message(FATAL_ERROR "bridge assembly failed")
endif()
execute_process(
    COMMAND "${TEST}" "${bytecode}" "${request}" "${response}"
    RESULT_VARIABLE test_result
)
file(REMOVE "${assembly}" "${bytecode}" "${request}" "${response}")
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "native bridge test failed")
endif()
