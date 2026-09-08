#define _POSIX_C_SOURCE 200809L

#include "bytecode.h"
#include "vm.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

static bool can_execute(const LanaChunk *chunk) {
    for (size_t i = 0u; i < chunk->code_count; ++i) {
        switch ((OpCode)chunk->code[i].opcode) {
            case OP_NOP: case OP_LOAD_CONST: case OP_MOVE:
            case OP_STATE_NEW: case OP_STATE_BUILD: case OP_TRANSFORM:
            case OP_MEASURE: case OP_APPEND: case OP_SAMPLE_STATE_DIST:
            case OP_MEASURE_BASIS: case OP_GET_FIELD:
            case OP_BINARY: case OP_UNARY: case OP_COMPARE:
            case OP_JUMP: case OP_JUMP_IF_TRUE: case OP_JUMP_IF_FALSE:
            case OP_ARRAY_NEW: case OP_ARRAY_GET: case OP_ARRAY_SET:
            case OP_CALL: case OP_RETURN: case OP_HALT:
                break;
            /* No host I/O, output, worker threads, or unbounded inner-loop estimators. */
            default: return false;
        }
    }
    return true;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    char path[] = "/tmp/lana-labc-fuzz-XXXXXX";
    LanaChunk chunk;
    LanaErrorInfo error = {0};
    size_t written = 0u;
    int fd = mkstemp(path);
    if (fd < 0) return 0;
    while (written < size) {
        ssize_t count = write(fd, data + written, size - written);
        if (count <= 0) {
            if (count < 0 && errno == EINTR) continue;
            (void)close(fd);
            (void)unlink(path);
            return 0;
        }
        written += (size_t)count;
    }
    if (close(fd) != 0) {
        (void)unlink(path);
        return 0;
    }
    if (lana_chunk_read_file(&chunk, path, &error) == LANA_OK) {
        if (lana_chunk_verify(&chunk, &error) != LANA_OK) abort();
        if (can_execute(&chunk)) {
            LanaVM vm;
            lana_vm_init(&vm, &chunk);
            lana_vm_set_memory_limit(&vm, 1024u * 1024u);
            vm.instruction_limit = 1000u;
            lana_vm_seed(&vm, 42u);
            (void)lana_vm_run(&vm);
            if (vm.allocated_bytes > vm.memory_limit) abort();
            lana_vm_free(&vm);
        }
        lana_chunk_free(&chunk);
    }
    (void)unlink(path);
    return 0;
}

#ifdef LANA_FUZZ_SELF_TEST
#include <assert.h>
int main(void) {
    LanaInstruction code[] = {{OP_LOAD_CONST, 0, 0, 0, 0, 1}, {OP_HALT, 0, 0, 0, 0, 2}};
    LanaChunk chunk;
    lana_chunk_init(&chunk);
    chunk.code = code; chunk.code_count = 2u;
    assert(can_execute(&chunk));
    const OpCode forbidden[] = {OP_HOST_CALL, OP_PRINT, OP_FORK, OP_ASYNC, OP_ESTIMATE_MEASURE_PROBABILITY};
    for (size_t i = 0u; i < sizeof(forbidden) / sizeof(forbidden[0]); ++i) {
        code[1].opcode = (uint8_t)forbidden[i];
        assert(!can_execute(&chunk));
    }
    assert(LLVMFuzzerTestOneInput((const uint8_t *)"not bytecode", 12u) == 0);
    return 0;
}
#endif
