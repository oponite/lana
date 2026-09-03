set(LANA_RUNTIME_SOURCES
    csrc/gc.c
    csrc/error.c
    csrc/sha256.c
    csrc/state.c
    csrc/value.c
    csrc/data.c
    csrc/bytecode.c
    csrc/vm.c
    csrc/shared.c
    csrc/project.c
    csrc/lsp.c
    csrc/assembler.c
    csrc/codec.c
    csrc/store.c
    csrc/state_codec.c
    csrc/policy.c
    csrc/ledger.c
    csrc/claims.c
    csrc/vendor/tweetnacl.c
    csrc/vendor/tweetnacl_random.c
    csrc/effects.c
    csrc/adapters.c
)

set_source_files_properties(csrc/vendor/tweetnacl.c PROPERTIES
    COMPILE_OPTIONS "-Wno-sign-compare;-Wno-unterminated-string-initialization")

add_library(lanaruntime STATIC ${LANA_RUNTIME_SOURCES})
target_include_directories(lanaruntime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
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

add_executable(lanavm csrc/cli.c)
target_link_libraries(lanavm PRIVATE lanaruntime m)
target_compile_options(lanavm PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_library(lanaruntime_release STATIC ${LANA_RUNTIME_SOURCES})
target_include_directories(lanaruntime_release PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(lanaruntime_release PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-format-truncation -O3 -DNDEBUG)
target_compile_definitions(lanaruntime_release PRIVATE
    LANA_ADAPTER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    LANA_ADAPTER_SUFFIX="${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_executable(lanavm_release csrc/cli.c)
target_link_libraries(lanavm_release PRIVATE lanaruntime_release m)
target_compile_options(lanavm_release PRIVATE -Wall -Wextra -Wpedantic -Werror -O3 -DNDEBUG)

set(LANA_COMPILER_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/compiler-bootstrap.lana")
set(LANA_NATIVE_COMPILER "${CMAKE_CURRENT_BINARY_DIR}/lana-compiler.labc")

add_executable(lana csrc/cli.c)
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
install(DIRECTORY include/lana DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
