if(NOT DEFINED BUILD_DIR OR NOT DEFINED ROOT OR NOT DEFINED VERIFY_SCRIPT)
    message(FATAL_ERROR "local install test paths are required")
endif()

set(prefix "${ROOT}/install-prefix")
file(REMOVE_RECURSE "${prefix}")

# Install the build tree into a temporary prefix.
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${prefix}"
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "cmake --install failed")
endif()

# Run the installation verification script with the temp prefix on PATH.
set(env_path "${prefix}/bin:$ENV{PATH}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "PATH=${env_path}" "${VERIFY_SCRIPT}"
    RESULT_VARIABLE verify_result
)
if(NOT verify_result EQUAL 0)
    message(FATAL_ERROR "verify-install.sh failed against the installed tree")
endif()

# When building for a specific architecture, confirm the installed binaries
# actually carry it (a universal build must not silently ship a single slice).
if(DEFINED EXPECTED_ARCH)
    foreach(binary lana lanavm)
        execute_process(
            COMMAND lipo -archs "${prefix}/bin/${binary}"
            RESULT_VARIABLE lipo_result
            OUTPUT_VARIABLE lipo_archs
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT lipo_result EQUAL 0)
            message(FATAL_ERROR "lipo -archs failed for ${binary}")
        endif()
        if(NOT lipo_archs STREQUAL "${EXPECTED_ARCH}")
            message(FATAL_ERROR "${binary} has archs '${lipo_archs}', expected '${EXPECTED_ARCH}'")
        endif()
    endforeach()
endif()

file(REMOVE_RECURSE "${prefix}")
