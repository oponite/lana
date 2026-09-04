#define _POSIX_C_SOURCE 200809L

#include "compiler_service.h"
#include "vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef LANA_COMPILER_BYTECODE_PATH
#define LANA_COMPILER_BYTECODE_PATH "lana-compiler.labc"
#endif

static bool compiler_candidate(const char *directory, char *out, size_t out_size) {
    int written = snprintf(out, out_size, "%s/%s", directory, "lana-compiler.labc");
    return written > 0 && (size_t)written < out_size && access(out, R_OK) == 0;
}

bool lana_compiler_find(const char *argv0, char *out, size_t out_size) {
    const char *configured = getenv("LANA_COMPILER_LABC");
    const char *path;
    char executable[4096], paths[8192], *cursor, *next;
    if (configured != NULL && snprintf(out, out_size, "%s", configured) > 0 && access(out, R_OK) == 0) return true;
    if (access(LANA_COMPILER_BYTECODE_PATH, R_OK) == 0) {
        return snprintf(out, out_size, "%s", LANA_COMPILER_BYTECODE_PATH) > 0;
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

int lana_compiler_run(const char *compiler_path, size_t argument_count,
                      const char **arguments, LanaErrorInfo *error) {
    LanaChunk compiler_chunk;
    LanaVM vm;
    LanaError result;
    *error = (LanaErrorInfo){0};
    result = lana_chunk_read_file(&compiler_chunk, compiler_path, error);
    if (result != LANA_OK) return 1;
    lana_vm_init(&vm, &compiler_chunk);
    vm.memory_limit = 256u * 1024u * 1024u;
    vm.instruction_limit = UINT64_C(50000000);
    lana_vm_set_program_args(&vm, argument_count, arguments);
    result = lana_vm_run(&vm);
    if (result != LANA_OK) *error = vm.error;
    lana_vm_free(&vm);
    lana_chunk_free(&compiler_chunk);
    return result == LANA_OK ? 0 : 1;
}

void lana_compiler_error_context(LanaErrorInfo *error, const char *source_path) {
    const char *location;
    uint32_t line = 1u, column = 1u;
    if (error == NULL) return;
    location = strstr(error->message, " at line ");
    if (location != NULL)
        (void)sscanf(location, " at line %u column %u", &line, &column);
    if (strstr(error->message, "parse error") != NULL) {
        error->code = LANA_ERR_PARSE;
        error->kind = lana_error_kind_from_code(error->code);
    } else if (strstr(error->message, "type error") != NULL) {
        error->code = LANA_ERR_TYPE;
        error->kind = lana_error_kind_from_code(error->code);
    }
    lana_error_set_source_span(error, source_path, line, column, line, column + 1u);
    lana_error_set_operation(error, "compile");
}

static int write_temp_source(const char *source_text, const char *source_path,
                             char *out, size_t out_size) {
    const char *slash = strrchr(source_path, '/');
    size_t dir_len = slash == NULL ? 0u : (size_t)(slash - source_path);
    size_t written;
    int descriptor;
    if (dir_len == 0u) {
        if (out_size < 32u) return -1;
        (void)snprintf(out, out_size, "/tmp/lana-lsp-XXXXXX");
    } else {
        if (dir_len + 17u >= out_size) return -1;
        memcpy(out, source_path, dir_len);
        memcpy(out + dir_len, "/.lana-lsp-XXXXXX", 17u);
        out[dir_len + 17u] = '\0';
    }
    descriptor = mkstemp(out);
    if (descriptor < 0) return -1;
    written = strlen(source_text);
    if (write(descriptor, source_text, written) != (ssize_t)written) {
        (void)close(descriptor); (void)unlink(out); return -1;
    }
    (void)close(descriptor);
    return 0;
}

int lana_compiler_check(const char *compiler_path, const char *source_text,
                        const char *source_path, LanaErrorInfo *error_out) {
    char source_temp[4096];
    char assembly_temp[] = "/tmp/lana-lsp-asm-XXXXXX";
    const char *arguments[2];
    int assembly_descriptor;
    int result;
    if (write_temp_source(source_text, source_path, source_temp, sizeof(source_temp)) != 0)
        return 1;
    assembly_descriptor = mkstemp(assembly_temp);
    if (assembly_descriptor < 0) { (void)unlink(source_temp); return 1; }
    (void)close(assembly_descriptor);
    arguments[0] = source_temp;
    arguments[1] = assembly_temp;
    result = lana_compiler_run(compiler_path, 2u, arguments, error_out);
    if (result != 0) lana_compiler_error_context(error_out, source_path);
    (void)unlink(source_temp);
    (void)unlink(assembly_temp);
    return result;
}

static char *read_file_text(const char *path) {
    FILE *file = fopen(path, "rb");
    long size;
    char *text;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
        (size = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) (void)fclose(file);
        return NULL;
    }
    text = malloc((size_t)size + 1u);
    if (text == NULL || fread(text, 1u, (size_t)size, file) != (size_t)size) {
        free(text); (void)fclose(file); return NULL;
    }
    text[size] = '\0';
    (void)fclose(file);
    return text;
}

int lana_compiler_symbols(const char *compiler_path, const char *source_text,
                          const char *source_path, char **json_out) {
    char source_temp[4096];
    char output_temp[] = "/tmp/lana-lsp-sym-XXXXXX";
    const char *arguments[3];
    int output_descriptor;
    int result;
    LanaErrorInfo error;
    if (json_out == NULL) return 1;
    *json_out = NULL;
    if (write_temp_source(source_text, source_path, source_temp, sizeof(source_temp)) != 0)
        return 1;
    output_descriptor = mkstemp(output_temp);
    if (output_descriptor < 0) { (void)unlink(source_temp); return 1; }
    (void)close(output_descriptor);
    arguments[0] = "--symbols";
    arguments[1] = source_temp;
    arguments[2] = output_temp;
    result = lana_compiler_run(compiler_path, 3u, arguments, &error);
    if (result == 0) *json_out = read_file_text(output_temp);
    (void)unlink(source_temp);
    (void)unlink(output_temp);
    return result;
}

static char *json_escape(const char *text, size_t length) {
    size_t capacity = length * 2u + 3u;
    size_t used = 0u;
    size_t index;
    char *out = malloc(capacity);
    if (out == NULL) return NULL;
    for (index = 0u; index < length; ++index) {
        char c = text[index];
        const char *replacement = NULL;
        switch (c) {
            case '"': replacement = "\\\""; break;
            case '\\': replacement = "\\\\"; break;
            case '\n': replacement = "\\n"; break;
            case '\r': replacement = "\\r"; break;
            case '\t': replacement = "\\t"; break;
            default: break;
        }
        if (replacement != NULL) {
            size_t repl_len = strlen(replacement);
            if (used + repl_len + 1u > capacity) {
                capacity = capacity * 2u + repl_len;
                char *grown = realloc(out, capacity);
                if (grown == NULL) { free(out); return NULL; }
                out = grown;
            }
            memcpy(out + used, replacement, repl_len);
            used += repl_len;
        } else if ((unsigned char)c < 0x20u) {
            char buffer[8];
            int written = snprintf(buffer, sizeof(buffer), "\\u%04x", (unsigned int)(unsigned char)c);
            if (used + (size_t)written + 1u > capacity) {
                capacity = capacity * 2u + (size_t)written;
                char *grown = realloc(out, capacity);
                if (grown == NULL) { free(out); return NULL; }
                out = grown;
            }
            memcpy(out + used, buffer, (size_t)written);
            used += (size_t)written;
        } else {
            if (used + 2u > capacity) {
                capacity = capacity * 2u;
                char *grown = realloc(out, capacity);
                if (grown == NULL) { free(out); return NULL; }
                out = grown;
            }
            out[used++] = c;
        }
    }
    out[used] = '\0';
    return out;
}

static const char *diagnostic_message(const char *message) {
    const char *column = strstr(message, "column ");
    const char *colon;
    if (column == NULL) return message;
    colon = strchr(column, ':');
    if (colon == NULL) return message;
    return colon[1] == ' ' ? colon + 2 : colon + 1;
}

char *lana_compiler_diagnostic_json(const LanaErrorInfo *error) {
    const char *message;
    char *escaped;
    char *json;
    size_t needed;
    uint32_t line, character;
    if (error == NULL) return NULL;
    message = diagnostic_message(error->message);
    escaped = json_escape(message, strlen(message));
    if (escaped == NULL) return NULL;
    line = error->source.start_line == 0u ? 0u : error->source.start_line - 1u;
    character = error->source.start_column == 0u ? 0u : error->source.start_column - 1u;
    needed = strlen(escaped) + 160u;
    json = malloc(needed);
    if (json == NULL) { free(escaped); return NULL; }
    (void)snprintf(json, needed,
        "{\"range\":{\"start\":{\"line\":%u,\"character\":%u},\"end\":{\"line\":%u,\"character\":%u}},"
        "\"severity\":1,\"source\":\"lana\",\"message\":\"%s\"}",
        line, character, line, character + 1u, escaped);
    free(escaped);
    return json;
}
