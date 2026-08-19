cmake_minimum_required(VERSION 3.20)
project(lana_vm VERSION 0.1.0 LANGUAGES C ASM)

include(GNUInstallDirs)
find_package(Threads REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(SS_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(SS_RUNTIME_SOURCES
    csrc/error.c
    csrc/state.c
    csrc/value.c
    csrc/bytecode.c
    csrc/vm.c
    csrc/assembler.c
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    list(APPEND SS_RUNTIME_SOURCES asm/arm64/apply_probe.S)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    list(APPEND SS_RUNTIME_SOURCES asm/x86_64/apply_probe.S)
else()
    list(APPEND SS_RUNTIME_SOURCES csrc/asm_probe_fallback.c)
    message(STATUS "No assembly probe for ${CMAKE_SYSTEM_PROCESSOR}; using the portable C probe")
endif()

add_library(ssruntime STATIC ${SS_RUNTIME_SOURCES})
target_include_directories(ssruntime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ssruntime PUBLIC Threads::Threads)
target_compile_options(ssruntime PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic;-Werror>)

if(SS_ENABLE_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(ssruntime PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(ssruntime PUBLIC -fsanitize=address,undefined)
endif()

add_executable(ssvm csrc/cli.c)
target_link_libraries(ssvm PRIVATE ssruntime m)
target_compile_options(ssvm PRIVATE -Wall -Wextra -Wpedantic -Werror)

enable_testing()
add_executable(ss_runtime_tests tests/c/test_runtime.c)
target_link_libraries(ss_runtime_tests PRIVATE ssruntime m)
target_compile_options(ss_runtime_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(ss_runtime_tests PRIVATE SS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME ss_runtime_tests COMMAND ss_runtime_tests)

install(TARGETS ssvm RUNTIME DESTINATION lana_tooling/bin)
