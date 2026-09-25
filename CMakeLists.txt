cmake_minimum_required(VERSION 3.20)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" LANA_VERSION
     REGEX "^[0-9]+\\.[0-9]+\\.[0-9]+$")
if(NOT LANA_VERSION)
    message(FATAL_ERROR "Could not read Lana version from VERSION")
endif()
project(lana_vm VERSION ${LANA_VERSION} LANGUAGES NONE)

include(cmake/BuildTargets.cmake)
include(cmake/CTestTargets.cmake)
