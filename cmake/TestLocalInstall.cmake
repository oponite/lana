if(NOT DEFINED BUILD_DIR OR NOT DEFINED ROOT OR NOT DEFINED VERIFY_SCRIPT OR NOT DEFINED PYTHON)
    message(FATAL_ERROR "local install test paths are required")
endif()

string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(prefix "${ROOT}/install-${suffix}")

# Install the build tree into a temporary prefix.
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${prefix}"
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "cmake --install failed")
endif()

# Verify exact outputs from an isolated cwd, without compiler path overrides.
set(architecture_args)
if(DEFINED EXPECTED_ARCH)
    string(REPLACE "," ";" architectures "${EXPECTED_ARCH}")
    foreach(architecture IN LISTS architectures)
        list(APPEND architecture_args --architecture "${architecture}")
    endforeach()
endif()
execute_process(
    COMMAND "${PYTHON}" -E "${VERIFY_SCRIPT}" --prefix "${prefix}" ${architecture_args}
    RESULT_VARIABLE verify_result
)
if(NOT verify_result EQUAL 0)
    message(FATAL_ERROR "clean install verification failed; prefix retained at ${prefix}")
endif()

file(REMOVE_RECURSE "${prefix}")
