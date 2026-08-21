#define _POSIX_C_SOURCE 200809L

#include "ss/assembler.h"
#include "ss/vm.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef SS_COMPILER_BYTECODE_PATH
#define SS_COMPILER_BYTECODE_PATH "lana-compiler.ssb"
#endif

static int report_error(const SSErrorInfo *error);

static void usage(FILE *out) {
    (void)fprintf(out,
        "usage:\n"
        "  lana compile program.lana -o program.ssb\n"
        "  lana check program.lana\n"
        "  ssvm asm program.ssa -o program.ssb\n"
        "  ssvm run program.ssb [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--memory-limit-mib N] [--instruction-limit N]\n"
        "  ssvm run-bytecode program.ssb [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--memory-limit-mib N] [--instruction-limit N]\n"
        "  ssvm dis program.ssb\n"
        "  ssvm verify program.ssb\n");
}

static bool has_suffix(const char *text, const char *suffix) {
    size_t text_length = strlen(text), suffix_length = strlen(suffix);
    return text_length >= suffix_length && strcmp(text + text_length - suffix_length, suffix) == 0;
}

static bool compiler_candidate(const char *directory, char *out, size_t out_size) {
    int written = snprintf(out, out_size, "%s/%s", directory, "lana-compiler.ssb");
    return written > 0 && (size_t)written < out_size && access(out, R_OK) == 0;
}

static bool find_compiler(const char *argv0, char *out, size_t out_size) {
    const char *configured = getenv("LANA_COMPILER_SSBC");
    const char *path;
    char executable[4096], paths[8192], *cursor, *next;
    if (configured != NULL && snprintf(out, out_size, "%s", configured) > 0 && access(out, R_OK) == 0) return true;
    if (access(SS_COMPILER_BYTECODE_PATH, R_OK) == 0) {
        return snprintf(out, out_size, "%s", SS_COMPILER_BYTECODE_PATH) > 0;
    }
    if (strchr(argv0, '/') != NULL) {
        char *separator;
        if (snprintf(executable, sizeof(executable), "%s", argv0) <= 0) return false;
        separator = strrchr(executable, '/');
        if (separator != NULL) { *separator = '\0'; if (compiler_candidate(executable, out, out_size)) return true; }
    }
    path = getenv("PATH");
    if (path == NULL || snprintf(paths, sizeof(paths), "%s", path) <= 0) return false;
    cursor = paths;
    while (cursor != NULL) {
        next = strchr(cursor, ':'); if (next != NULL) *next++ = '\0';
        if (compiler_candidate(*cursor == '\0' ? "." : cursor, out, out_size)) return true;
        cursor = next;
    }
    return false;
}

static int compile_source_file(const char *compiler_path, const char *source_path, const char *output_path) {
    SSChunk compiler_chunk, output_chunk; SSErrorInfo error = {0}; SSError result;
    char assembly_path[] = "/tmp/lana-assembly-XXXXXX"; int descriptor;
    const char *arguments[2] = {source_path, assembly_path}; VM vm;
    descriptor = mkstemp(assembly_path);
    if (descriptor < 0) { (void)fprintf(stderr, "cannot create compiler temporary file\n"); return 1; }
    (void)close(descriptor);
    result = ss_chunk_read_file(&compiler_chunk, compiler_path, &error);
    if (result != SS_OK) { (void)unlink(assembly_path); return report_error(&error); }
    ss_vm_init(&vm, &compiler_chunk); vm.memory_limit = 256u * 1024u * 1024u; vm.instruction_limit = UINT64_C(50000000);
    ss_vm_set_program_args(&vm, 2, arguments); result = ss_vm_run(&vm);
    if (result != SS_OK) {
        error = vm.error; ss_vm_free(&vm); ss_chunk_free(&compiler_chunk); (void)unlink(assembly_path); return report_error(&error);
    }
    ss_vm_free(&vm); ss_chunk_free(&compiler_chunk);
    result = ss_assemble_file(assembly_path, &output_chunk, &error); (void)unlink(assembly_path);
    if (result == SS_OK) result = ss_chunk_write_file(&output_chunk, output_path, &error);
    if (result != SS_OK) return report_error(&error);
    ss_chunk_free(&output_chunk); return 0;
}

static int report_error(const SSErrorInfo *error) {
    (void)fprintf(stderr, "%s:%u: error[%s]: %s (instruction %zu, opcode %s)\n",
                  error->function[0] == '\0' ? "<bytecode>" : error->function,
                  error->line, ss_error_name(error->code), error->message,
                  error->ip, ss_opcode_name(error->opcode));
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
    bool trace = false; bool stats = false; uint64_t seed = UINT64_C(0x4c414e41), instruction_limit = 0u; int index;
    size_t workers = 0u, max_tasks = 0u, memory_limit = 0u;
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
        } else if (strcmp(argv[index], "--memory-limit-mib") == 0 && index + 1 < argc) {
            char *end; unsigned long long parsed;
            errno = 0; parsed = strtoull(argv[++index], &end, 10);
            if (errno != 0 || *end != '\0' || parsed == 0u || parsed > SIZE_MAX / (1024u * 1024u)) {
                (void)fprintf(stderr, "invalid memory limit\n"); return 2;
            }
            memory_limit = (size_t)parsed * 1024u * 1024u;
        } else if (strcmp(argv[index], "--instruction-limit") == 0 && index + 1 < argc) {
            char *end; errno = 0; instruction_limit = strtoull(argv[++index], &end, 10);
            if (errno != 0 || *end != '\0' || instruction_limit == 0u) { (void)fprintf(stderr, "invalid instruction limit\n"); return 2; }
        } else if ((strcmp(argv[index], "--workers") == 0 || strcmp(argv[index], "--max-tasks") == 0) && index + 1 < argc) {
            char *end; unsigned long parsed; bool is_workers = strcmp(argv[index], "--workers") == 0;
            errno = 0; parsed = strtoul(argv[++index], &end, 10);
            if (errno != 0 || *end != '\0' || parsed == 0u) { (void)fprintf(stderr, "invalid scheduler limit\n"); return 2; }
            if (is_workers) workers = (size_t)parsed; else max_tasks = (size_t)parsed;
        } else { usage(stderr); return 2; }
    }
    result = ss_chunk_read_file(&chunk, argv[2], &error);
    if (result != SS_OK) return report_error(&error);
    if (!execute) ss_disassemble(&chunk, stdout);
    else {
        struct timespec started, finished;
        uint64_t elapsed_ns;
        VM vm; ss_vm_init(&vm, &chunk); vm.trace = trace; ss_vm_seed(&vm, seed);
        if (memory_limit > 0u) vm.memory_limit = memory_limit;
        if (instruction_limit > 0u) vm.instruction_limit = instruction_limit;
        if ((workers > 0u && ss_vm_set_worker_count(&vm, workers) != SS_OK) ||
            (max_tasks > 0u && ss_vm_set_task_limit(&vm, max_tasks) != SS_OK)) {
            ss_vm_free(&vm); ss_chunk_free(&chunk); return 1;
        }
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
    char compiler_path[4096];
    if (argc < 2) { usage(stderr); return 2; }
    if (strcmp(argv[1], "version") == 0) { (void)printf("Lana %s (SSBC v5, C VM, native compiler)\n", LANA_VERSION); return 0; }
    if (strcmp(argv[1], "compile") == 0) {
        if (argc != 5 || strcmp(argv[3], "-o") != 0) { usage(stderr); return 2; }
        if (!find_compiler(argv[0], compiler_path, sizeof(compiler_path))) { (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1; }
        return compile_source_file(compiler_path, argv[2], argv[4]);
    }
    if (strcmp(argv[1], "check") == 0) {
        char bytecode_path[] = "/tmp/lana-check-XXXXXX"; int descriptor, result;
        if (argc != 3) { usage(stderr); return 2; }
        if (!find_compiler(argv[0], compiler_path, sizeof(compiler_path))) { (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1; }
        descriptor = mkstemp(bytecode_path); if (descriptor < 0) return 1; (void)close(descriptor);
        result = compile_source_file(compiler_path, argv[2], bytecode_path); (void)unlink(bytecode_path); return result;
    }
    if (strcmp(argv[1], "asm") == 0) return assemble_command(argc, argv);
    if (strcmp(argv[1], "run") == 0 && argc >= 3 && has_suffix(argv[2], ".lana")) {
        char bytecode_path[] = "/tmp/lana-program-XXXXXX"; char *source = argv[2]; int descriptor, result;
        if (!find_compiler(argv[0], compiler_path, sizeof(compiler_path))) { (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1; }
        descriptor = mkstemp(bytecode_path); if (descriptor < 0) return 1; (void)close(descriptor);
        result = compile_source_file(compiler_path, source, bytecode_path);
        if (result == 0) { argv[2] = bytecode_path; result = load_command(argc, argv, true); argv[2] = source; }
        (void)unlink(bytecode_path); return result;
    }
    if (strcmp(argv[1], "run") == 0 || strcmp(argv[1], "run-bytecode") == 0) return load_command(argc, argv, true);
    if (strcmp(argv[1], "dis") == 0) return load_command(argc, argv, false);
    if (strcmp(argv[1], "verify") == 0) {
        int result = load_command(argc, argv, false);
        if (result == 0) (void)printf("verified\n");
        return result;
    }
    usage(stderr); return 2;
}
