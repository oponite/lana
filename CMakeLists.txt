cmake_minimum_required(VERSION 3.20)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" LANA_VERSION
     REGEX "^[0-9]+\\.[0-9]+\\.[0-9]+$")
if(NOT LANA_VERSION)
    message(FATAL_ERROR "Could not read Lana version from VERSION")
endif()
project(lana_vm VERSION ${LANA_VERSION} LANGUAGES C)

include(GNUInstallDirs)
find_package(Threads REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Expose POSIX functions (strdup, dlopen) under strict C11 on glibc.
add_compile_definitions(_DEFAULT_SOURCE)

option(LANA_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(LANA_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(LANA_BUILD_FUZZERS "Build libFuzzer targets" OFF)
option(LANA_BUILD_INTEGRATIONS "Build optional native integration facade" OFF)
if(LANA_ENABLE_SANITIZERS AND LANA_ENABLE_TSAN)
    message(FATAL_ERROR "LANA_ENABLE_SANITIZERS and LANA_ENABLE_TSAN are mutually exclusive")
endif()

include(cmake/BuildTargets.cmake)
include(cmake/CTestTargets.cmake)
include(cmake/IntegrationTargets.cmake)
