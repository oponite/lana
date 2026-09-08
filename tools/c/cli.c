#define _POSIX_C_SOURCE 200809L

#include "assembler.h"
#include "vm.h"
#include "project.h"
#include "lsp.h"
#include "compiler_service.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef LANA_COMPILER_BYTECODE_PATH
#define LANA_COMPILER_BYTECODE_PATH "lana-compiler.labc"
#endif

static int report_error(const LanaErrorInfo *error);

static bool debugger_hook(LanaVM *vm, size_t instruction, uint32_t source_line,
                          void *context) {
    char command[64];
    LanaFrame *frame = &vm->frames[vm->frame_count - 1u];
    const char *function = frame->function < vm->chunk->function_count
        ? vm->chunk->functions[frame->function].name : "<entry>";
    (void)context;
    (void)printf("BREAK line=%u instruction=%zu function=%s frames=%zu\n",
                 source_line, instruction, function, vm->frame_count);
    (void)printf("debug [s]tep [c]ontinue [q]uit> ");
    (void)fflush(stdout);
    if (fgets(command, sizeof(command), stdin) == NULL || command[0] == 'q')
        return false;
    if (command[0] == 's') vm->debug_step = true;
    else vm->debug_break_line = 0u;
    return true;
}

static void usage(FILE *out) {
    (void)fprintf(out,
        "usage:\n"
        "  lana compile program.lana -o program.labc\n"
        "  lana new directory\n"
        "  lana lsp\n"
        "  lana debug program.lana\n"
        "  lana inspect program.lana [--format json|dot]\n"
        "  lana repl\n"
        "  lana build|run|test|check|fmt|doc\n"
        "  lana check program.lana\n"
        "  lanavm asm program.lasm -o program.labc\n"
        "  lanavm run program.labc [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--memory-limit-mib N] [--instruction-limit N]\n"
        "  lanavm run-bytecode program.labc [--trace] [--stats] [--seed N] [--workers N] [--max-tasks N] [--memory-limit-mib N] [--instruction-limit N]\n"
        "  lanavm dis program.labc\n"
        "  lanavm verify program.labc\n");
}

static bool has_suffix(const char *text, const char *suffix) {
    size_t text_length = strlen(text), suffix_length = strlen(suffix);
    return text_length >= suffix_length && strcmp(text + text_length - suffix_length, suffix) == 0;
}

static int run_project_tool(const char *argv0, const char *mode,
                            const char *argument) {
    char compiler_path[4096];
    const char *arguments[2] = {mode, argument};
    LanaErrorInfo error = {0};
    if (!lana_compiler_find(argv0, compiler_path, sizeof(compiler_path))) {
        (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1;
    }
    if (lana_compiler_run(compiler_path, 2u, arguments, &error) != 0)
        return report_error(&error);
    return 0;
}

static int compile_source_file(const char *compiler_path, const char *source_path, const char *output_path) {
    LanaChunk output_chunk; LanaErrorInfo error = {0}; LanaError result;
    char assembly_path[] = "/tmp/lana-assembly-XXXXXX"; int descriptor;
    const char *arguments[2] = {source_path, assembly_path};
    descriptor = mkstemp(assembly_path);
    if (descriptor < 0) { (void)fprintf(stderr, "cannot create compiler temporary file\n"); return 1; }
    (void)close(descriptor);
    if (lana_compiler_run(compiler_path, 2u, arguments, &error) != 0) {
        lana_compiler_error_context(&error, source_path);
        (void)unlink(assembly_path); return report_error(&error);
    }
    result = lana_assemble_file(assembly_path, &output_chunk, &error); (void)unlink(assembly_path);
    if (result == LANA_OK) result = lana_chunk_write_file(&output_chunk, output_path, &error);
    if (result != LANA_OK) return report_error(&error);
    lana_chunk_free(&output_chunk); return 0;
}

static int project_compile(const char *source, const char *output,
                           void *context) {
    return compile_source_file(context, source, output);
}

static int project_plan(const char *directory, const char *output,
                        void *context) {
    const char *arguments[3] = {"--project-plan", directory, output};
    LanaErrorInfo error = {0};
    if (lana_compiler_run(context, 3u, arguments, &error) != 0)
        return report_error(&error);
    return 0;
}

static int report_error(const LanaErrorInfo *error) {
    size_t index;
    const char *path = error->source.path[0] == '\0'
                           ? (error->function[0] == '\0' ? "<bytecode>" : error->function)
                           : error->source.path;
    uint32_t start_line = error->source.start_line == 0u ? error->line : error->source.start_line;
    (void)fprintf(stderr, "%s:%u:%u-%u:%u: error[%s/%s]: %s",
                  path, start_line, error->source.start_column,
                  error->source.end_line, error->source.end_column,
                  lana_error_kind_name(error->kind), lana_error_name(error->code),
                  error->message);
    if (error->operation[0] != '\0')
        (void)fprintf(stderr, " (operation %s)", error->operation);
    (void)fprintf(stderr, " (instruction %zu, opcode %s)\n",
                  error->ip, lana_opcode_name(error->opcode));
    for (index = 0u; index < error->cause_count; ++index) {
        const LanaErrorCause *cause = &error->causes[index];
        (void)fprintf(stderr, "  caused by [%s/%s] %s: %s\n",
                      lana_error_kind_name(cause->kind), lana_error_name(cause->code),
                      cause->operation[0] == '\0' ? "unknown" : cause->operation,
                      cause->message);
    }
    if (error->cause_chain_truncated) (void)fprintf(stderr, "  caused by: additional causes omitted\n");
    if (error->resolution_reason != LANA_RESOLUTION_REASON_NONE) {
        (void)fprintf(stderr, "  resolution: %s",
                      lana_resolution_reason_name(error->resolution_reason));
        if (error->has_remaining_alternatives)
            (void)fprintf(stderr, ", remaining alternatives: %zu", error->remaining_alternatives);
        (void)fputc('\n', stderr);
    }
    if (error->exact_support != LANA_EXACT_SUPPORT_UNKNOWN)
        (void)fprintf(stderr, "  exact support: %s (%s)\n",
                      lana_exact_support_name(error->exact_support),
                      error->exact_support_detail);
    if (error->cancellation.present)
        (void)fprintf(stderr, "  cancellation: lineage %llu (%s)\n",
                      (unsigned long long)error->cancellation.task_lineage,
                      error->cancellation.reason);
    if (error->resource_limit.present)
        (void)fprintf(stderr, "  resource: %s limit %llu, observed %llu %s\n",
                      lana_resource_kind_name(error->resource_limit.resource),
                      (unsigned long long)error->resource_limit.limit,
                      (unsigned long long)error->resource_limit.observed,
                      error->resource_limit.unit);
    return 1;
}

static int assemble_command(int argc, char **argv) {
    LanaChunk chunk; LanaErrorInfo error = {0}; LanaError result;
    if (argc != 5 || strcmp(argv[3], "-o") != 0) { usage(stderr); return 2; }
    result = lana_assemble_file(argv[2], &chunk, &error);
    if (result == LANA_OK) result = lana_chunk_write_file(&chunk, argv[4], &error);
    if (result != LANA_OK) return report_error(&error);
    lana_chunk_free(&chunk); return 0;
}

static int load_command(int argc, char **argv, bool execute) {
    LanaChunk chunk; LanaErrorInfo error = {0}; LanaError result;
    bool trace = false; bool stats = false; bool debug = false;
    uint32_t break_line = 0u;
    uint64_t seed = UINT64_C(0x4c414e41), instruction_limit = 0u; int index;
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
        else if (strcmp(argv[index], "--debug") == 0) debug = true;
        else if (strcmp(argv[index], "--break") == 0 && index + 1 < argc) {
            char *end; unsigned long parsed;
            errno = 0; parsed = strtoul(argv[++index], &end, 10);
            if (errno != 0 || *end != '\0' || parsed == 0u || parsed > UINT32_MAX) {
                (void)fprintf(stderr, "invalid breakpoint line\n"); return 2;
            }
            break_line = (uint32_t)parsed; debug = true;
        }
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
    result = lana_chunk_read_file(&chunk, argv[2], &error);
    if (result != LANA_OK) return report_error(&error);
    if (!execute) lana_disassemble(&chunk, stdout);
    else {
        struct timespec started, finished;
        uint64_t elapsed_ns;
        LanaVM vm; lana_vm_init(&vm, &chunk); vm.trace = trace; vm.profile_opcodes = stats; lana_vm_seed(&vm, seed);
        if (debug) {
            vm.debug_hook = debugger_hook;
            vm.debug_break_line = break_line;
            vm.debug_step = break_line == 0u;
        }
        if (memory_limit > 0u) lana_vm_set_memory_limit(&vm, memory_limit);
        if (instruction_limit > 0u) vm.instruction_limit = instruction_limit;
        if ((workers > 0u && lana_vm_set_worker_count(&vm, workers) != LANA_OK) ||
            (max_tasks > 0u && lana_vm_set_task_limit(&vm, max_tasks) != LANA_OK)) {
            lana_vm_free(&vm); lana_chunk_free(&chunk); return 1;
        }
        lana_vm_set_program_args(&vm, program_argc, program_argv);
        (void)timespec_get(&started, TIME_UTC);
        result = lana_vm_run(&vm);
        (void)timespec_get(&finished, TIME_UTC);
        elapsed_ns = (uint64_t)(finished.tv_sec - started.tv_sec) * UINT64_C(1000000000) +
                     (uint64_t)(finished.tv_nsec - started.tv_nsec);
        if (result != LANA_OK) { error = vm.error; lana_vm_free(&vm); lana_chunk_free(&chunk); return report_error(&error); }
        if (stats) {
            size_t opcode;
            (void)fprintf(stderr,
                          "LANAVM_STATS {\"instructions\":%llu,\"state_transitions\":%llu,"
                          "\"allocations\":%llu,\"allocated_bytes\":%zu,\"elapsed_ns\":%llu,\"opcodes\":{",
                          (unsigned long long)vm.instruction_count,
                          (unsigned long long)vm.state_transition_count,
                          (unsigned long long)vm.allocation_count, vm.allocated_bytes,
                          (unsigned long long)elapsed_ns);
            for (opcode = 0; opcode < OP_COUNT; ++opcode) {
                if (opcode != 0u) (void)fputc(',', stderr);
                (void)fprintf(stderr, "\"%s\":%llu", lana_opcode_name((uint8_t)opcode),
                              (unsigned long long)vm.opcode_counts[opcode]);
            }
            (void)fprintf(stderr, "}}\n");
        }
        lana_vm_free(&vm);
    }
    lana_chunk_free(&chunk); return 0;
}

static int inspect_command(int argc, char **argv) {
    LanaChunk chunk; LanaErrorInfo error = {0}; LanaError result;
    LanaInspectFormat format = LANA_INSPECT_JSON;
    const char *path; int index;
    if (argc < 3) { usage(stderr); return 2; }
    path = argv[2];
    for (index = 3; index < argc; ++index) {
        if (strcmp(argv[index], "--format") == 0 && index + 1 < argc) {
            if (strcmp(argv[++index], "json") == 0) format = LANA_INSPECT_JSON;
            else if (strcmp(argv[index], "dot") == 0) format = LANA_INSPECT_DOT;
            else { (void)fprintf(stderr, "invalid inspect format\n"); return 2; }
        } else { usage(stderr); return 2; }
    }
    result = lana_chunk_read_file(&chunk, path, &error);
    if (result != LANA_OK) return report_error(&error);
    {
        LanaVM vm; lana_vm_init(&vm, &chunk);
        result = lana_vm_run(&vm);
        if (result != LANA_OK) { error = vm.error; lana_vm_free(&vm); lana_chunk_free(&chunk); return report_error(&error); }
        if (vm.result.type != VAL_STATE_DIST) {
            (void)fprintf(stderr, "inspect: program did not return a state_dist (got %s)\n",
                          lana_value_type_name(vm.result.type));
            lana_vm_free(&vm); lana_chunk_free(&chunk); return 1;
        }
        {
            char *out = NULL;
            result = lana_vm_state_dist_inspect(vm.result.as.state_dist, format, &out);
            if (result != LANA_OK) {
                (void)fprintf(stderr, "inspect: %s\n", lana_error_name(result));
                lana_vm_free(&vm); lana_chunk_free(&chunk); return 1;
            }
            (void)printf("%s\n", out);
            free(out);
        }
        lana_vm_free(&vm);
    }
    lana_chunk_free(&chunk); return 0;
}

/* ------------------------------------------------------------------ */
/* Interactive REPL (LIP-020). A thin loop over the compiler and VM:   */
/* each completed input is compiled and executed against the           */
/* accumulated session source, so top-level bindings persist.          */
/* ------------------------------------------------------------------ */

#define REPL_SEED UINT64_C(0x4c414e41)

/* Append `src` to the dynamic string `dst` (realloc'd), updating `len`. */
static char *repl_str_append(char *dst, size_t *len, const char *src) {
    size_t src_len = strlen(src);
    char *new_dst = realloc(dst, *len + src_len + 1u);
    if (new_dst == NULL) return dst;
    memcpy(new_dst + *len, src, src_len);
    *len += src_len;
    new_dst[*len] = '\0';
    return new_dst;
}

/* True when the accumulated input is a complete statement (balanced
 * brackets, no unterminated string literal). */
static bool repl_is_complete(const char *buffer) {
    int depth = 0; bool in_string = false; bool escaped = false;
    const char *p;
    for (p = buffer; *p != '\0'; ++p) {
        if (in_string) {
            if (escaped) escaped = false;
            else if (*p == '\\') escaped = true;
            else if (*p == '"') in_string = false;
            continue;
        }
        if (*p == '"') in_string = true;
        else if (*p == '(' || *p == '[' || *p == '{') ++depth;
        else if (*p == ')' || *p == ']' || *p == '}') --depth;
    }
    return depth <= 0 && !in_string;
}

/* True when the first `len` bytes of `line` begin a statement keyword. */
static bool repl_starts_with_keyword(const char *line, size_t len) {
    static const char *const keywords[] = {
        "let", "fn", "if", "while", "for", "return", "print", "import",
        "from", "match", "type", "struct", "async", "generator", "yield",
        "break", "continue", "assert", "use", "const", "var", "shared",
        "capability", "effect", "claim", "task", "spawn", "await", "export",
        NULL
    };
    size_t index;
    for (index = 0u; keywords[index] != NULL; ++index) {
        size_t klen = strlen(keywords[index]);
        if (len >= klen && strncmp(line, keywords[index], klen) == 0 &&
            (len == klen || line[klen] == ' ' || line[klen] == '\t'))
            return true;
    }
    return false;
}

/* Compile an in-memory source string to a chunk. On failure prints the
 * compiler's recovery message (SYNTAX-10) unless `quiet` is set, and returns
 * nonzero. `quiet` is used for the display probe, whose compile failure is
 * expected for non-expression inputs. */
static int repl_compile_source(const char *compiler_path, const char *source_text,
                               LanaChunk *chunk, bool quiet) {
    LanaErrorInfo error = {0}; LanaError result;
    char source_path[] = "/tmp/lana-repl-source-XXXXXX";
    char assembly_path[] = "/tmp/lana-repl-assembly-XXXXXX";
    int descriptor;
    const char *arguments[2] = {source_path, assembly_path};
    descriptor = mkstemp(source_path);
    if (descriptor < 0) return 1;
    size_t source_length = strlen(source_text);
    if (write(descriptor, source_text, source_length) != (ssize_t)source_length) {
        (void)close(descriptor);
        (void)unlink(source_path);
        return 1;
    }
    (void)close(descriptor);
    descriptor = mkstemp(assembly_path);
    if (descriptor < 0) { (void)unlink(source_path); return 1; }
    (void)close(descriptor);
    if (lana_compiler_run(compiler_path, 2u, arguments, &error) != 0) {
        lana_compiler_error_context(&error, source_path);
        (void)unlink(source_path); (void)unlink(assembly_path);
        if (!quiet) (void)fprintf(stderr, "%s\n", error.message);
        return 1;
    }
    (void)unlink(source_path);
    result = lana_assemble_file(assembly_path, chunk, &error);
    (void)unlink(assembly_path);
    if (result != LANA_OK) {
        if (!quiet) (void)fprintf(stderr, "%s\n", error.message);
        return 1;
    }
    return 0;
}

/* Run a chunk on a fresh VM. On success prints the result value when
 * `print_result` is set. Returns 0 on success, nonzero on runtime error. */
static int repl_run_chunk(const LanaChunk *chunk, uint64_t seed, bool print_result) {
    LanaVM vm; LanaErrorInfo error = {0}; LanaError run_result;
    lana_vm_init(&vm, chunk);
    lana_vm_seed(&vm, seed);
    run_result = lana_vm_run(&vm);
    if (run_result != LANA_OK) {
        error = vm.error;
        lana_vm_free(&vm);
        return report_error(&error);
    }
    if (print_result) { lana_value_print(&vm.result); (void)printf("\n"); }
    lana_vm_free(&vm);
    return 0;
}

/* Handle a `:`-prefixed command. Sets `*quit` when the session should end. */
static void repl_handle_command(const char *compiler_path, const char *command,
                                char **session_source, size_t *session_len,
                                bool *quit) {
    char name[64]; const char *argument = "";
    const char *space; size_t name_len, cmd_len;
    char *cmd_copy = strdup(command);
    if (cmd_copy == NULL) return;
    cmd_len = strlen(cmd_copy);
    while (cmd_len > 0u && (cmd_copy[cmd_len - 1u] == '\n' ||
                            cmd_copy[cmd_len - 1u] == '\r' ||
                            cmd_copy[cmd_len - 1u] == ' ' ||
                            cmd_copy[cmd_len - 1u] == '\t'))
        cmd_copy[--cmd_len] = '\0';
    space = strchr(cmd_copy, ' ');
    if (space != NULL) {
        name_len = (size_t)(space - cmd_copy);
        argument = space + 1;
        while (*argument == ' ') ++argument;
    } else {
        name_len = cmd_len;
    }
    if (name_len >= sizeof(name)) name_len = sizeof(name) - 1u;
    memcpy(name, cmd_copy, name_len); name[name_len] = '\0';

    if (strcmp(name, ":help") == 0) {
        (void)printf("commands:\n");
        (void)printf("  :help              list commands\n");
        (void)printf("  :quit, :exit       end the session\n");
        (void)printf("  :load <file>       compile and run a file in the session\n");
        (void)printf("  :save <file>       export the session's bindings as a .lana file\n");
        (void)printf("  :clear             drop all bindings\n");
    } else if (strcmp(name, ":quit") == 0 || strcmp(name, ":exit") == 0) {
        *quit = true;
    } else if (strcmp(name, ":clear") == 0) {
        *session_len = 0u;
        if (*session_source != NULL) (*session_source)[0] = '\0';
    } else if (strcmp(name, ":save") == 0) {
        if (*argument == '\0') {
            (void)fprintf(stderr, "usage: :save <file>\n");
        } else {
            FILE *file = fopen(argument, "w");
            if (file == NULL) {
                (void)fprintf(stderr, ":save: cannot open %s\n", argument);
            } else {
                if (*session_source != NULL)
                    (void)fwrite(*session_source, 1, *session_len, file);
                (void)fclose(file);
                (void)printf("saved %s\n", argument);
            }
        }
    } else if (strcmp(name, ":load") == 0) {
        if (*argument == '\0') {
            (void)fprintf(stderr, "usage: :load <file>\n");
        } else {
            FILE *file = fopen(argument, "r");
            if (file == NULL) {
                (void)fprintf(stderr, ":load: cannot open %s\n", argument);
            } else {
                char contents[65536]; size_t n;
                n = fread(contents, 1, sizeof(contents) - 1u, file);
                (void)fclose(file);
                contents[n] = '\0';
                {
                    char *candidate = malloc(*session_len + n + 1u);
                    if (candidate != NULL) {
                        if (*session_source != NULL)
                            memcpy(candidate, *session_source, *session_len);
                        memcpy(candidate + *session_len, contents, n);
                        candidate[*session_len + n] = '\0';
                        {
                            LanaChunk chunk;
                            if (repl_compile_source(compiler_path, candidate, &chunk, false) == 0) {
                                if (repl_run_chunk(&chunk, REPL_SEED, false) == 0) {
                                    free(*session_source);
                                    *session_source = candidate;
                                    *session_len += n;
                                    (void)printf("loaded %s\n", argument);
                                } else {
                                    free(candidate);
                                }
                                lana_chunk_free(&chunk);
                            } else {
                                free(candidate);
                            }
                        }
                    }
                }
            }
        }
    } else {
        (void)fprintf(stderr, "unknown command: %s (try :help)\n", name);
    }
    free(cmd_copy);
}

/* Compile and execute one complete input, committing it to the session on
 * success. A bare expression's value is printed; a statement prints nothing. */
static void repl_process_input(const char *compiler_path, char **session_source,
                               size_t *session_len, const char *buffer) {
    const char *trimmed = buffer;
    size_t tlen, blen = strlen(buffer);
    char *candidate;
    while (*trimmed == ' ' || *trimmed == '\t' || *trimmed == '\n') ++trimmed;
    if (*trimmed == '\0') return;
    tlen = strlen(trimmed);
    while (tlen > 0u && (trimmed[tlen - 1u] == ' ' || trimmed[tlen - 1u] == '\t' ||
                         trimmed[tlen - 1u] == '\n')) --tlen;
    if (tlen > 0u && trimmed[tlen - 1u] == ';') --tlen;
    while (tlen > 0u && (trimmed[tlen - 1u] == ' ' || trimmed[tlen - 1u] == '\t')) --tlen;

    candidate = malloc(*session_len + blen + 1u);
    if (candidate == NULL) return;
    if (*session_source != NULL) memcpy(candidate, *session_source, *session_len);
    memcpy(candidate + *session_len, buffer, blen);
    candidate[*session_len + blen] = '\0';

    if (tlen > 0u && !repl_starts_with_keyword(trimmed, tlen)) {
        /* Probe form: `return (<line>);` captures the expression's value. If
         * the line is not a bare expression this fails to compile and we fall
         * back to the plain statement form below. */
        size_t probe_len = *session_len + blen + tlen + 16u;
        char *probe = malloc(probe_len);
        if (probe != NULL) {
            size_t off = 0u;
            if (*session_source != NULL) {
                memcpy(probe, *session_source, *session_len);
                off = *session_len;
            }
            memcpy(probe + off, buffer, blen); off += blen;
            memcpy(probe + off, "return (", 8u); off += 8u;
            memcpy(probe + off, trimmed, tlen); off += tlen;
            memcpy(probe + off, ");\n", 3u); off += 3u;
            probe[off] = '\0';
            {
                LanaChunk chunk;
                if (repl_compile_source(compiler_path, probe, &chunk, true) == 0) {
                    if (repl_run_chunk(&chunk, REPL_SEED, true) == 0) {
                        free(*session_source);
                        *session_source = candidate;
                        *session_len += blen;
                    } else {
                        free(candidate);
                    }
                    lana_chunk_free(&chunk);
                    free(probe);
                    return;
                }
            }
            free(probe);
        }
    }

    {
        LanaChunk chunk;
        if (repl_compile_source(compiler_path, candidate, &chunk, false) == 0) {
            if (repl_run_chunk(&chunk, REPL_SEED, false) == 0) {
                free(*session_source);
                *session_source = candidate;
                *session_len += blen;
            } else {
                free(candidate);
            }
            lana_chunk_free(&chunk);
        } else {
            free(candidate);
        }
    }
}

/* Run the interactive REPL. */
static int repl_command(const char *compiler_path) {
    char *session_source = NULL; size_t session_len = 0u;
    char *buffer = NULL; size_t buffer_len = 0u;
    char line[4096];
    bool quit = false;
    while (!quit) {
        const char *prompt = buffer_len == 0u ? "lana> " : "...> ";
        (void)printf("%s", prompt); (void)fflush(stdout);
        if (fgets(line, sizeof(line), stdin) == NULL) break; /* EOF */
        if (buffer_len == 0u) {
            const char *trimmed = line;
            while (*trimmed == ' ' || *trimmed == '\t') ++trimmed;
            if (*trimmed == ':') {
                repl_handle_command(compiler_path, trimmed, &session_source,
                                    &session_len, &quit);
                continue;
            }
        }
        buffer = repl_str_append(buffer, &buffer_len, line);
        if (!repl_is_complete(buffer)) continue; /* multiline */
        repl_process_input(compiler_path, &session_source, &session_len, buffer);
        buffer_len = 0u;
    }
    free(session_source);
    free(buffer);
    return 0;
}

int main(int argc, char **argv) {
    char compiler_path[4096];
    if (argc < 2) {
        /* Bare `lana` (no subcommand) launches the REPL (LIP-020). */
        if (!lana_compiler_find(argv[0], compiler_path, sizeof(compiler_path))) {
            (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1;
        }
        return repl_command(compiler_path);
    }
    if (strcmp(argv[1], "repl") == 0) {
        if (argc != 2) { usage(stderr); return 2; }
        if (!lana_compiler_find(argv[0], compiler_path, sizeof(compiler_path))) {
            (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1;
        }
        return repl_command(compiler_path);
    }
    if (strcmp(argv[1], "version") == 0) { (void)printf("Lana %s (LABC v2, C VM, native compiler)\n", LANA_VERSION); return 0; }
    if (strcmp(argv[1], "new") == 0) {
        if (argc != 3) { usage(stderr); return 2; }
        if (!lana_compiler_find(argv[0], compiler_path, sizeof(compiler_path))) {
            (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1;
        }
        {
            const char *arguments[2] = {"--project-new", argv[2]};
            LanaErrorInfo error = {0};
            if (lana_compiler_run(compiler_path, 2u, arguments, &error) != 0)
                return report_error(&error);
            return 0;
        }
    }
    if (strcmp(argv[1], "lsp") == 0) {
        if (argc != 2) { usage(stderr); return 2; }
        return lana_lsp_run(argv[0]);
    }
    if (strcmp(argv[1], "fmt") == 0 || strcmp(argv[1], "doc") == 0 ||
        strcmp(argv[1], "build") == 0 || strcmp(argv[1], "test") == 0) {
        if ((argc != 2 && !(argc == 3 && strcmp(argv[1], "fmt") == 0 &&
                           strcmp(argv[2], "--check") == 0))) {
            usage(stderr); return 2;
        }
        if (strcmp(argv[1], "fmt") == 0)
            return run_project_tool(argv[0], "--project-fmt", argc == 3 ? "check" : "write");
        if (strcmp(argv[1], "doc") == 0)
            return run_project_tool(argv[0], "--project-doc", ".");
        if (strcmp(argv[1], "test") == 0) return lana_project_test(".", argv[0]);
        if (!lana_compiler_find(argv[0], compiler_path, sizeof(compiler_path))) {
            (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1;
        }
        {
            char output[4096];
            return lana_project_build_with_plan(".", project_plan,
                                                project_compile, compiler_path,
                                                output, sizeof(output));
        }
    }
    if (strcmp(argv[1], "compile") == 0) {
        if (argc != 5 || strcmp(argv[3], "-o") != 0) { usage(stderr); return 2; }
        if (!lana_compiler_find(argv[0], compiler_path, sizeof(compiler_path))) { (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1; }
        return compile_source_file(compiler_path, argv[2], argv[4]);
    }
    if (strcmp(argv[1], "check") == 0) {
        char bytecode_path[] = "/tmp/lana-check-XXXXXX"; int descriptor, result;
        if (!lana_compiler_find(argv[0], compiler_path, sizeof(compiler_path))) { (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1; }
        if (argc == 2) return lana_project_build_with_plan(
            ".", project_plan, project_compile, compiler_path,
            (char[4096]){0}, 4096u);
        if (argc != 3) { usage(stderr); return 2; }
        descriptor = mkstemp(bytecode_path); if (descriptor < 0) return 1; (void)close(descriptor);
        result = compile_source_file(compiler_path, argv[2], bytecode_path); (void)unlink(bytecode_path); return result;
    }
    if (strcmp(argv[1], "asm") == 0) return assemble_command(argc, argv);
    if (strcmp(argv[1], "debug") == 0) {
        char bytecode_path[] = "/tmp/lana-debug-XXXXXX";
        char *debug_argv[6] = {argv[0], "run-bytecode", bytecode_path,
                               "--debug", NULL, NULL};
        int descriptor; int result; int debug_argc = 4;
        if ((argc != 3 && argc != 5) || !has_suffix(argv[2], ".lana") ||
            (argc == 5 && strcmp(argv[3], "--break") != 0)) { usage(stderr); return 2; }
        if (argc == 5) {
            debug_argv[4] = "--break"; debug_argv[5] = argv[4]; debug_argc = 6;
        }
        if (!lana_compiler_find(argv[0], compiler_path, sizeof(compiler_path))) { (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1; }
        descriptor = mkstemp(bytecode_path); if (descriptor < 0) return 1;
        (void)close(descriptor);
        result = compile_source_file(compiler_path, argv[2], bytecode_path);
        if (result == 0) result = load_command(debug_argc, debug_argv, true);
        (void)unlink(bytecode_path); return result;
    }
    if (strcmp(argv[1], "run") == 0 && argc == 2) {
        char output[4096]; char *run_argv[3] = {argv[0], "run-bytecode", output};
        if (!lana_compiler_find(argv[0], compiler_path, sizeof(compiler_path))) { (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1; }
        if (lana_project_build_with_plan(".", project_plan, project_compile,
                                         compiler_path, output,
                                         sizeof(output)) != 0) return 1;
        return load_command(3, run_argv, true);
    }
    if (strcmp(argv[1], "run") == 0 && argc >= 3 && has_suffix(argv[2], ".lana")) {
        char bytecode_path[] = "/tmp/lana-program-XXXXXX"; char *source = argv[2]; int descriptor, result;
        if (!lana_compiler_find(argv[0], compiler_path, sizeof(compiler_path))) { (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1; }
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
    if (strcmp(argv[1], "inspect") == 0) {
        if (argc >= 3 && has_suffix(argv[2], ".lana")) {
            char bytecode_path[] = "/tmp/lana-inspect-XXXXXX"; char *source = argv[2]; int descriptor, result;
            if (!lana_compiler_find(argv[0], compiler_path, sizeof(compiler_path))) { (void)fprintf(stderr, "native Lana compiler bytecode not found\n"); return 1; }
            descriptor = mkstemp(bytecode_path); if (descriptor < 0) return 1; (void)close(descriptor);
            result = compile_source_file(compiler_path, source, bytecode_path);
            if (result == 0) { argv[2] = bytecode_path; result = inspect_command(argc, argv); argv[2] = source; }
            (void)unlink(bytecode_path); return result;
        }
        return inspect_command(argc, argv);
    }
    usage(stderr); return 2;
}
