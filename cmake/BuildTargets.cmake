set(LANA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/vm/include
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/include)

set(LANA_RUNTIME_SOURCES
    vm/c/gc.c
    vm/c/error.c
    runtime/c/sha256.c
    vm/c/state.c
    vm/c/value.c
    runtime/c/data.c
    vm/c/bytecode.c
    vm/c/vm.c
    runtime/c/shared.c
    tools/c/project.c
    tools/c/lsp.c
    tools/c/json.c
    tools/c/compiler_service.c
    vm/c/assembler.c
    runtime/c/codec.c
    runtime/c/store.c
    runtime/c/state_codec.c
    runtime/c/policy.c
    runtime/c/ledger.c
    runtime/c/claims.c
    runtime/c/vendor/tweetnacl.c
    runtime/c/vendor/tweetnacl_random.c
    runtime/c/effects.c
    runtime/c/adapters.c
)

set_source_files_properties(runtime/c/vendor/tweetnacl.c PROPERTIES
    COMPILE_OPTIONS "-Wno-sign-compare;-Wno-unterminated-string-initialization")

add_library(lanaruntime STATIC ${LANA_RUNTIME_SOURCES})
target_include_directories(lanaruntime PUBLIC ${LANA_INCLUDE_DIRS})
target_link_libraries(lanaruntime PUBLIC Threads::Threads)
target_compile_options(lanaruntime PRIVATE -Wall -Wextra -Wpedantic -Werror)
# Adapter facade locates dlopen plugins in the build directory.
target_compile_definitions(lanaruntime PRIVATE
    LANA_ADAPTER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    LANA_ADAPTER_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}")

if(LANA_ENABLE_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(lanaruntime PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(lanaruntime PUBLIC -fsanitize=address,undefined)
endif()
if(LANA_ENABLE_TSAN AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(lanaruntime PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(lanaruntime PUBLIC -fsanitize=thread)
endif()

add_executable(lanavm tools/c/cli.c)
target_link_libraries(lanavm PRIVATE lanaruntime m)
target_compile_options(lanavm PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_library(lanaruntime_release STATIC ${LANA_RUNTIME_SOURCES})
target_include_directories(lanaruntime_release PUBLIC ${LANA_INCLUDE_DIRS})
target_compile_options(lanaruntime_release PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-format-truncation -O3 -DNDEBUG)
target_compile_definitions(lanaruntime_release PRIVATE
    LANA_ADAPTER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    LANA_ADAPTER_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_executable(lanavm_release tools/c/cli.c)
target_link_libraries(lanavm_release PRIVATE lanaruntime_release m)
target_compile_options(lanavm_release PRIVATE -Wall -Wextra -Wpedantic -Werror -O3 -DNDEBUG)

set(LANA_COMPILER_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/compiler-bootstrap.lana")
set(LANA_NATIVE_COMPILER "${CMAKE_CURRENT_BINARY_DIR}/lana-compiler.labc")

add_executable(lana tools/c/cli.c)
target_link_libraries(lana PRIVATE lanaruntime m)
target_compile_options(lana PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(lana PRIVATE LANA_VERSION="${LANA_VERSION}")
target_compile_definitions(lanavm PRIVATE LANA_VERSION="${LANA_VERSION}")
target_compile_definitions(lanavm_release PRIVATE LANA_VERSION="${LANA_VERSION}")

add_custom_command(
    OUTPUT "${LANA_COMPILER_BUNDLE}"
    COMMAND "${CMAKE_COMMAND}"
            -DLANA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DLANA_BUNDLE_OUTPUT=${LANA_COMPILER_BUNDLE}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BundleCompiler.cmake"
    DEPENDS
        compiler/lexer.lana compiler/syntax.lana compiler/parser.lana
        compiler/resolver.lana compiler/ir.lana compiler/emitter.lana
        compiler/main.lana cmake/BundleCompiler.cmake
    VERBATIM
)
add_custom_target(lana_compiler_bundle ALL DEPENDS "${LANA_COMPILER_BUNDLE}")

add_custom_command(
    OUTPUT "${LANA_NATIVE_COMPILER}"
    COMMAND $<TARGET_FILE:lanavm> asm
            "${CMAKE_CURRENT_SOURCE_DIR}/compiler/bootstrap/compiler.lasm"
            -o "${LANA_NATIVE_COMPILER}"
    DEPENDS lanavm compiler/bootstrap/compiler.lasm
    VERBATIM
)
add_custom_target(lana_native_compiler ALL DEPENDS "${LANA_NATIVE_COMPILER}")
add_dependencies(lana lana_native_compiler)

install(TARGETS lanaruntime ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS lanavm lana RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES "${LANA_NATIVE_COMPILER}" DESTINATION ${CMAKE_INSTALL_BINDIR})
# Public headers keep the `lana/` install prefix; the three layer include dirs
# are flattened into a single `${INCLUDEDIR}/lana` namespace.
install(DIRECTORY vm/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lana)
install(DIRECTORY runtime/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lana)
install(DIRECTORY tools/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lana)
