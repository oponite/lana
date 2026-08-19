#include "ss/assembler.h"
#include "ss/vm.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(FILE *out) {
    (void)fprintf(out,
        "usage:\n"
        "  ssvm asm program.ssa -o program.ssb\n"
        "  ssvm run program.ssb [--trace] [--stats] [--seed N]\n"
        "  ssvm run-bytecode program.ssb [--trace] [--stats] [--seed N]\n"
        "  ssvm dis program.ssb\n"
        "  ssvm verify program.ssb\n");
}

static int report_error(const SSErrorInfo *error) {
    (void)fprintf(stderr, "%s at instruction %zu, opcode %s, source line %u: %s\n",
                  ss_error_name(error->code), error->ip, ss_opcode_name(error->opcode),
                  error->line, error->message);
    return 1;
}

static int assemble_command(int argc, char **argv) {
    SSChunk chunk; SSErrorInfo error = {0}; SSError result;
    if (argc != 5 || strcmp(argv[3], "-o") != 0) { usage(stderr); return 2; }
    result = ss_assemble_file(argv[2], &chunk, &error);
    if (result == SS_OK) result = ss_chunk_write_file(&chunk, argv[4], &error);
    if (result != SS_OK) return report_error(&error);
    ss_chunk_free(&chunk); return 0;
}

static int load_command(int argc, char **argv, bool execute) {
    SSChunk chunk; SSErrorInfo error = {0}; SSError result;
    bool trace = false; bool stats = false; uint64_t seed = UINT64_C(0x4c414e41); int index;
    int program_argc = 0; const char **program_argv = NULL;
    if (argc < 3) { usage(stderr); return 2; }
    for (index = 3; index < argc; ++index) {
        if (strcmp(argv[index], "--") == 0) {
            program_argc = argc - index - 1;
            program_argv = (const char **)&argv[index + 1];
            break;
        }
        if (strcmp(argv[index], "--trace") == 0) trace = true;
        else if (strcmp(argv[index], "--stats") == 0) stats = true;
        else if (strcmp(argv[index], "--seed") == 0 && index + 1 < argc) {
            char *end; errno = 0; seed = strtoull(argv[++index], &end, 10);
            if (errno != 0 || *end != '\0') { (void)fprintf(stderr, "invalid seed\n"); return 2; }
        } else { usage(stderr); return 2; }
    }
    result = ss_chunk_read_file(&chunk, argv[2], &error);
    if (result != SS_OK) return report_error(&error);
    if (!execute) ss_disassemble(&chunk, stdout);
    else {
        struct timespec started, finished;
        uint64_t elapsed_ns;
        VM vm; ss_vm_init(&vm, &chunk); vm.trace = trace; ss_vm_seed(&vm, seed);
        ss_vm_set_program_args(&vm, program_argc, program_argv);
        (void)timespec_get(&started, TIME_UTC);
        result = ss_vm_run(&vm);
        (void)timespec_get(&finished, TIME_UTC);
        elapsed_ns = (uint64_t)(finished.tv_sec - started.tv_sec) * UINT64_C(1000000000) +
                     (uint64_t)(finished.tv_nsec - started.tv_nsec);
        if (result != SS_OK) { error = vm.error; ss_vm_free(&vm); ss_chunk_free(&chunk); return report_error(&error); }
        if (stats) {
            size_t opcode;
            (void)fprintf(stderr,
                          "SSVM_STATS {\"instructions\":%llu,\"state_transitions\":%llu,"
                          "\"allocations\":%llu,\"allocated_bytes\":%zu,\"elapsed_ns\":%llu,\"opcodes\":{",
                          (unsigned long long)vm.instruction_count,
                          (unsigned long long)vm.state_transition_count,
                          (unsigned long long)vm.allocation_count, vm.allocated_bytes,
                          (unsigned long long)elapsed_ns);
            for (opcode = 0; opcode < OP_COUNT; ++opcode) {
                if (opcode != 0u) (void)fputc(',', stderr);
                (void)fprintf(stderr, "\"%s\":%llu", ss_opcode_name((uint8_t)opcode),
                              (unsigned long long)vm.opcode_counts[opcode]);
            }
            (void)fprintf(stderr, "}}\n");
        }
        ss_vm_free(&vm);
    }
    ss_chunk_free(&chunk); return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(stderr); return 2; }
    if (strcmp(argv[1], "asm") == 0) return assemble_command(argc, argv);
    if (strcmp(argv[1], "run") == 0 || strcmp(argv[1], "run-bytecode") == 0) return load_command(argc, argv, true);
    if (strcmp(argv[1], "dis") == 0) return load_command(argc, argv, false);
    if (strcmp(argv[1], "verify") == 0) {
        int result = load_command(argc, argv, false);
        if (result == 0) (void)printf("verified\n");
        return result;
    }
    usage(stderr); return 2;
}
