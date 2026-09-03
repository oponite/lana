# Adapter plugins, HTTP service, and adapter tests. Included from
# IntegrationTargets.cmake under LANA_BUILD_INTEGRATIONS.

find_package(SQLite3)
if(SQLite3_FOUND)
    add_library(lana_adapter_sqlite SHARED csrc/adapters/sqlite_adapter.c)
    target_include_directories(lana_adapter_sqlite PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/adapters)
    target_link_libraries(lana_adapter_sqlite PRIVATE lanaruntime SQLite3::SQLite3 m)
    target_compile_options(lana_adapter_sqlite PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(lana_adapter_sqlite PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set(LANA_ADAPTER_SQLITE_AVAILABLE TRUE)
else()
    set(LANA_ADAPTER_SQLITE_AVAILABLE FALSE)
endif()

add_library(lana_adapter_http_json SHARED csrc/adapters/http_json_adapter.c)
target_include_directories(lana_adapter_http_json PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/adapters)
target_link_libraries(lana_adapter_http_json PRIVATE lanaruntime m)
target_compile_options(lana_adapter_http_json PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(lana_adapter_http_json PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(LANA_ADAPTER_HTTP_AVAILABLE TRUE)

add_executable(lana_http_service csrc/http_service.c)
target_compile_options(lana_http_service PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Shared runtime for the ctypes Python bindings.
add_library(lanaruntime_shared SHARED ${LANA_RUNTIME_SOURCES})
target_include_directories(lanaruntime_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lanaruntime_shared PUBLIC Threads::Threads m)
target_compile_options(lanaruntime_shared PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(lanaruntime_shared PRIVATE
    LANA_ADAPTER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    LANA_ADAPTER_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_executable(lana_adapter_tests tests/c/test_adapters.c)
target_link_libraries(lana_adapter_tests PRIVATE lanaruntime m)
target_compile_options(lana_adapter_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
if(LANA_ADAPTER_SQLITE_AVAILABLE)
    target_compile_definitions(lana_adapter_tests PRIVATE LANA_ADAPTER_SQLITE_AVAILABLE)
    target_link_libraries(lana_adapter_tests PRIVATE SQLite3::SQLite3)
endif()
if(LANA_ADAPTER_HTTP_AVAILABLE)
    target_compile_definitions(lana_adapter_tests PRIVATE LANA_ADAPTER_HTTP_AVAILABLE)
endif()
add_test(NAME lana_adapter_tests COMMAND lana_adapter_tests)

add_executable(lana_http_adapter_tests tests/c/test_http_adapter.c)
target_link_libraries(lana_http_adapter_tests PRIVATE lanaruntime m)
target_compile_options(lana_http_adapter_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
if(LANA_ADAPTER_HTTP_AVAILABLE)
    target_compile_definitions(lana_http_adapter_tests PRIVATE
        LANA_ADAPTER_HTTP_AVAILABLE
        LANA_HTTP_SERVICE="$<TARGET_FILE:lana_http_service>")
endif()
add_test(NAME lana_http_adapter_tests COMMAND lana_http_adapter_tests)

find_program(LANA_PYTEST NAMES pytest)
if(LANA_PYTEST)
    add_test(NAME lana_python_api_tests
        COMMAND ${LANA_PYTEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/integrations/python/tests/test_api.py
            ${CMAKE_CURRENT_SOURCE_DIR}/integrations/python/tests/test_http_client.py
            -q)
    set_tests_properties(lana_python_api_tests PROPERTIES
        ENVIRONMENT
            "LANA_RUNTIME_LIBRARY=$<TARGET_FILE:lanaruntime_shared>;LANA_HTTP_SERVICE=$<TARGET_FILE:lana_http_service>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/integrations/python/src")
endif()
