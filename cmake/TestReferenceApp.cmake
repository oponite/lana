if(NOT DEFINED LANAVM OR NOT DEFINED COMPILER OR NOT DEFINED SOURCE OR
   NOT DEFINED TEST OR NOT DEFINED ROOT OR NOT DEFINED FIXTURE_DIR OR
   NOT DEFINED APP_NAME)
    message(FATAL_ERROR "reference app test paths are required")
endif()

set(assembly "${ROOT}/${APP_NAME}-test.lasm")
set(bytecode "${ROOT}/${APP_NAME}-test.labc")
set(request "${ROOT}/${APP_NAME}-request.json")
set(request_refuse "${ROOT}/${APP_NAME}-request-refuse.json")
set(response "${ROOT}/${APP_NAME}-response.json")
set(store "${ROOT}/${APP_NAME}-store")
set(store_refuse "${ROOT}/${APP_NAME}-store-refuse")
set(store_determinism "${ROOT}/${APP_NAME}-store-determinism")
set(policy "${FIXTURE_DIR}/policy.json")
set(envelope_a "${ROOT}/${APP_NAME}-envelope-a.json")
set(envelope_b "${ROOT}/${APP_NAME}-envelope-b.json")

# Start from a clean slate: a stale store from a failed run would conflict
# with the deterministic decision id.
file(REMOVE_RECURSE "${store}" "${store_refuse}" "${store_determinism}")

execute_process(
    COMMAND "${LANAVM}" run "${COMPILER}" -- "${SOURCE}" "${assembly}"
    RESULT_VARIABLE compile_result
)
if(NOT compile_result EQUAL 0)
    message(FATAL_ERROR "${APP_NAME} source compilation failed")
endif()
execute_process(
    COMMAND "${LANAVM}" asm "${assembly}" -o "${bytecode}"
    RESULT_VARIABLE assemble_result
)
if(NOT assemble_result EQUAL 0)
    message(FATAL_ERROR "${APP_NAME} assembly failed")
endif()

# Generate request files with the absolute fixture directory substituted.
file(READ "${FIXTURE_DIR}/request.json" request_template)
string(REPLACE "__FIXTURE_DIR__" "${FIXTURE_DIR}" request_text "${request_template}")
file(WRITE "${request}" "${request_text}")
file(READ "${FIXTURE_DIR}/request_refuse.json" request_refuse_template)
string(REPLACE "__FIXTURE_DIR__" "${FIXTURE_DIR}" request_refuse_text "${request_refuse_template}")
file(WRITE "${request_refuse}" "${request_refuse_text}")

# Authorize path.
execute_process(
    COMMAND "${TEST}" "${bytecode}" "${request}" "${response}" "${store}"
                "${policy}" "authorize" "${envelope_a}"
    RESULT_VARIABLE authorize_result
)
if(NOT authorize_result EQUAL 0)
    message(FATAL_ERROR "${APP_NAME} authorize pipeline test failed")
endif()

# Refuse path.
execute_process(
    COMMAND "${TEST}" "${bytecode}" "${request_refuse}" "${response}" "${store_refuse}"
                "${policy}" "refuse"
    RESULT_VARIABLE refuse_result
)
if(NOT refuse_result EQUAL 0)
    message(FATAL_ERROR "${APP_NAME} refuse pipeline test failed")
endif()

# Determinism: the same request and seed must produce an identical envelope.
# A fresh store keeps the input revision at zero so the envelope is comparable.
execute_process(
    COMMAND "${TEST}" "${bytecode}" "${request}" "${response}" "${store_determinism}"
                "${policy}" "authorize" "${envelope_b}"
    RESULT_VARIABLE determinism_result
)
if(NOT determinism_result EQUAL 0)
    message(FATAL_ERROR "${APP_NAME} determinism run failed")
endif()
file(READ "${envelope_a}" envelope_a_text)
file(READ "${envelope_b}" envelope_b_text)
if(NOT envelope_a_text STREQUAL envelope_b_text)
    message(FATAL_ERROR "${APP_NAME} pipeline is not deterministic")
endif()

file(REMOVE "${assembly}" "${bytecode}" "${request}" "${request_refuse}"
           "${response}" "${envelope_a}" "${envelope_b}")
file(REMOVE_RECURSE "${store}" "${store_refuse}" "${store_determinism}")
