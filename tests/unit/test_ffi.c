/* LIP-018 two-way FFI unit test (C VM).
 *
 * Exercises the deterministic FFI paths that need a real shared library:
 * ffi_declare signature parsing, ffi_load + ffi_call success against the
 * bundled test library, capability denial, and crash containment (a callee
 * that faults is reported as {"error": external} and the VM survives).
 *
 * The differential fixtures cover the no-library paths (signature parsing,
 * capability denial, argument type rejection); this test covers the dlopen
 * path, which is C-only because the Rust libloading error strings differ.
 */

#include "assembler.h"
#include "error.h"
#include "vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef LANA_FFI_TEST_LIB
#error "LANA_FFI_TEST_LIB must point at the built ffi_test_lib shared library"
#endif

#define CHECK(condition) do { \
    if (!(condition)) { \
        (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                      __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

/* Assemble `source` into `chunk`. */
static int assemble(const char *source, LanaChunk *chunk) {
    char path[] = "/tmp/lana-ffi-XXXXXX.lasm";
    int fd;
    LanaErrorInfo error = {0};
    LanaError result;
    fd = mkstemps(path, 5);
    CHECK(fd >= 0);
    CHECK(write(fd, source, strlen(source)) == (ssize_t)strlen(source));
    CHECK(close(fd) == 0);
    lana_chunk_init(chunk);
    result = lana_assemble_file(path, chunk, &error);
    (void)unlink(path);
    CHECK(result == LANA_OK);
    return 0;
}

/* Run `chunk`, capturing stdout into `out` (caller frees). Returns the VM
 * error code. */
static LanaError run_capture(LanaChunk *chunk, char **out) {
    char out_path[] = "/tmp/lana-ffi-out-XXXXXX";
    int saved_fd;
    int fd;
    FILE *capture;
    LanaVM vm;
    LanaError result;
    long size;
    fd = mkstemp(out_path);
    CHECK(fd >= 0);
    saved_fd = dup(STDOUT_FILENO);
    CHECK(saved_fd >= 0);
    /* Flush any buffered banner output before redirecting stdout. */
    fflush(stdout);
    CHECK(dup2(fd, STDOUT_FILENO) >= 0);
    lana_vm_init(&vm, chunk);
    result = lana_vm_run(&vm);
    lana_vm_free(&vm);
    fflush(stdout);
    CHECK(dup2(saved_fd, STDOUT_FILENO) >= 0);
    close(saved_fd);
    capture = fdopen(fd, "r");
    CHECK(capture != NULL);
    CHECK(fseek(capture, 0, SEEK_END) == 0);
    size = ftell(capture);
    CHECK(size >= 0);
    *out = malloc((size_t)size + 1u);
    CHECK(*out != NULL);
    rewind(capture);
    CHECK(fread(*out, 1u, (size_t)size, capture) == (size_t)size);
    (*out)[size] = '\0';
    fclose(capture);
    (void)unlink(out_path);
    return result;
}

static int test_declare(void) {
    LanaChunk chunk;
    char *out = NULL;
    LanaError result;
    printf("Testing ffi_declare signature parsing...\n");
    CHECK(assemble(
        ".function main 0 16\n"
        "LOAD_STRING R0 646f75626c652061646428646f75626c652c20646f75626c6529\n"
        "HOST_CALL ffi_declare R0 1 R1\n"
        "PRINT R1\n"
        "RETURN R0\n",
        &chunk) == 0);
    result = run_capture(&chunk, &out);
    CHECK(result == LANA_OK);
    if (out == NULL || strcmp(out, "0\n") != 0) {
        (void)fprintf(stderr, "declare output was: [%s]\n", out ? out : "(null)");
        return 1;
    }
    free(out);
    return 0;
}

static int test_call_success(void) {
    LanaChunk chunk;
    char *out = NULL;
    LanaError result;
    char source[1024];
    printf("Testing ffi_load + ffi_call success...\n");
    (void)snprintf(source, sizeof(source),
        ".function main 0 16\n"
        "LOAD_CONST R0 ffi\n"
        "HOST_CALL shared_information R0 1 R1\n"
        "LOAD_CONST R2 use\n"
        "HOST_CALL grant R1 2 R3\n"
        "LOAD_STRING R4 646f75626c652061646428646f75626c652c20646f75626c6529\n"
        "HOST_CALL ffi_declare R4 1 R5\n"
        "LOAD_CONST R6 %s\n"
        "HOST_CALL ffi_load R6 1 R7\n"
        "LOAD_CONST R8 2\n"
        "LOAD_CONST R9 3\n"
        "ARRAY_NEW R10 R8 2\n"
        "LOAD_CONST R11 1\n"
        "ARRAY_SET R10 R11 R9\n"
        "LOAD_CONST R0 0\n"
        "LOAD_CONST R1 0\n"
        "MOVE R2 R10\n"
        "HOST_CALL ffi_call R0 3 R12\n"
        "PRINT R12\n"
        "RETURN R0\n",
        LANA_FFI_TEST_LIB);
    CHECK(assemble(source, &chunk) == 0);
    result = run_capture(&chunk, &out);
    CHECK(result == LANA_OK);
    CHECK(out != NULL && strstr(out, "\"ok\": 5") != NULL);
    free(out);
    return 0;
}

static int test_capability_denial(void) {
    LanaChunk chunk;
    char *out = NULL;
    LanaError result;
    printf("Testing ffi_call capability denial...\n");
    CHECK(assemble(
        ".function main 0 16\n"
        "LOAD_CONST R0 0\n"
        "LOAD_CONST R1 0\n"
        "LOAD_CONST R2 2\n"
        "LOAD_CONST R3 3\n"
        "ARRAY_NEW R4 R2 2\n"
        "LOAD_CONST R5 1\n"
        "ARRAY_SET R4 R5 R3\n"
        "MOVE R2 R4\n"
        "HOST_CALL ffi_call R0 3 R6\n"
        "PRINT R6\n"
        "RETURN R0\n",
        &chunk) == 0);
    result = run_capture(&chunk, &out);
    CHECK(result == LANA_ERR_EXTERNAL);
    free(out);
    return 0;
}

static int test_crash_containment(void) {
    LanaChunk chunk;
    char *out = NULL;
    LanaError result;
    char source[1024];
    printf("Testing crash containment (faulting callee)...\n");
    (void)snprintf(source, sizeof(source),
        ".function main 0 16\n"
        "LOAD_CONST R0 ffi\n"
        "HOST_CALL shared_information R0 1 R1\n"
        "LOAD_CONST R2 use\n"
        "HOST_CALL grant R1 2 R3\n"
        "LOAD_STRING R4 766f696420637261736828766f696429\n"
        "HOST_CALL ffi_declare R4 1 R5\n"
        "LOAD_CONST R6 %s\n"
        "HOST_CALL ffi_load R6 1 R7\n"
        "LOAD_CONST R8 0\n"
        "ARRAY_NEW R9 R8 0\n"
        "LOAD_CONST R0 0\n"
        "LOAD_CONST R1 0\n"
        "MOVE R2 R9\n"
        "HOST_CALL ffi_call R0 3 R10\n"
        "PRINT R10\n"
        "RETURN R0\n",
        LANA_FFI_TEST_LIB);
    CHECK(assemble(source, &chunk) == 0);
    result = run_capture(&chunk, &out);
    /* The fault is contained: the VM reports {"error": external} and the
     * process survives. */
    CHECK(result == LANA_OK);
    CHECK(out != NULL && strstr(out, "\"error\": external") != NULL);
    free(out);
    return 0;
}

int main(void) {
    if (test_declare() != 0) return 1;
    if (test_call_success() != 0) return 1;
    if (test_capability_denial() != 0) return 1;
    if (test_crash_containment() != 0) return 1;
    printf("All FFI tests passed.\n");
    return 0;
}
