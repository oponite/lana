#define _POSIX_C_SOURCE 200809L

#include "lsp.h"

#include "compiler_service.h"
#include "json.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void send_message(const char *json) {
    (void)printf("Content-Length: %zu\r\n\r\n%s", strlen(json), json);
    (void)fflush(stdout);
}

static void send_result(const char *id, const char *result) {
    char *message;
    size_t needed = strlen(id) + strlen(result) + 64u;
    message = malloc(needed);
    if (message == NULL) return;
    (void)snprintf(message, needed,
                   "{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":%s}",
                   id, result);
    send_message(message);
    free(message);
}

static void format_id(const JsonValue *id, char *out, size_t size) {
    if (id == NULL) { (void)snprintf(out, size, "null"); return; }
    if (id->type == JSON_NUMBER) {
        double number = id->as.number;
        if (number == (double)(long long)number)
            (void)snprintf(out, size, "%lld", (long long)number);
        else
            (void)snprintf(out, size, "%.17g", number);
    } else if (id->type == JSON_STRING) {
        (void)snprintf(out, size, "\"%s\"", id->as.string);
    } else {
        (void)snprintf(out, size, "null");
    }
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void uri_to_path(const char *uri, char *out, size_t size) {
    const char *read = uri;
    char *write = out;
    size_t remaining = size;
    if (strncmp(read, "file://", 7u) == 0) read += 7u;
    while (*read != '\0' && remaining > 1u) {
        if (read[0] == '%' && hex_value(read[1]) >= 0 && hex_value(read[2]) >= 0) {
            *write++ = (char)(hex_value(read[1]) * 16 + hex_value(read[2]));
            read += 3;
        } else {
            *write++ = *read++;
        }
        --remaining;
    }
    *write = '\0';
}

static const char *document_uri(const JsonValue *params) {
    JsonValue *text_doc = json_get(params, "textDocument");
    return json_string(json_get(text_doc, "uri"));
}

static const char *document_text(const JsonValue *params) {
    JsonValue *text_doc = json_get(params, "textDocument");
    const char *text = json_string(json_get(text_doc, "text"));
    if (text != NULL) return text;
    {
        JsonValue *changes = json_get(params, "contentChanges");
        if (changes != NULL && changes->type == JSON_ARRAY && changes->as.array.count > 0u) {
            JsonValue *change = changes->as.array.items[0];
            return json_string(json_get(change, "text"));
        }
    }
    return NULL;
}

static void publish_diagnostics(const char *uri, const char *diagnostics) {
    char *message;
    size_t needed = strlen(uri) + strlen(diagnostics) + 128u;
    message = malloc(needed);
    if (message == NULL) return;
    (void)snprintf(message, needed,
        "{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/publishDiagnostics\","
        "\"params\":{\"uri\":\"%s\",\"diagnostics\":%s}}",
        uri, diagnostics);
    send_message(message);
    free(message);
}

static void check_and_publish(const char *compiler_path, bool compiler_found,
                              const char *uri, const char *text) {
    char path[4096];
    LanaErrorInfo error;
    char *diagnostic;
    if (!compiler_found || uri == NULL || text == NULL) {
        publish_diagnostics(uri == NULL ? "" : uri, "[]");
        return;
    }
    uri_to_path(uri, path, sizeof(path));
    if (lana_compiler_check(compiler_path, text, path, &error) == 0) {
        publish_diagnostics(uri, "[]");
        return;
    }
    diagnostic = lana_compiler_diagnostic_json(&error);
    if (diagnostic == NULL) { publish_diagnostics(uri, "[]"); return; }
    {
        char *array;
        size_t needed = strlen(diagnostic) + 4u;
        array = malloc(needed);
        if (array == NULL) { free(diagnostic); return; }
        (void)snprintf(array, needed, "[%s]", diagnostic);
        publish_diagnostics(uri, array);
        free(array);
    }
    free(diagnostic);
}

/* Document cache: the LSP sends text only on didOpen/didChange, so query
 * requests (hover/definition/...) must read the text back from here. */
typedef struct {
    char *uri;
    char *text;
} LspDocument;

static LspDocument *documents = NULL;
static size_t document_count = 0u;
static size_t document_capacity = 0u;

static void document_set(const char *uri, const char *text) {
    size_t index;
    if (uri == NULL) return;
    for (index = 0u; index < document_count; ++index) {
        if (strcmp(documents[index].uri, uri) == 0) {
            free(documents[index].text);
            documents[index].text = text == NULL ? NULL : strdup(text);
            return;
        }
    }
    if (document_count == document_capacity) {
        size_t capacity = document_capacity == 0u ? 8u : document_capacity * 2u;
        LspDocument *grown = realloc(documents, capacity * sizeof(*grown));
        if (grown == NULL) return;
        documents = grown;
        document_capacity = capacity;
    }
    documents[document_count].uri = strdup(uri);
    documents[document_count].text = text == NULL ? NULL : strdup(text);
    ++document_count;
}

static const char *document_get(const char *uri) {
    size_t index;
    if (uri == NULL) return NULL;
    for (index = 0u; index < document_count; ++index) {
        if (strcmp(documents[index].uri, uri) == 0) return documents[index].text;
    }
    return NULL;
}

static void document_remove(const char *uri) {
    size_t index;
    if (uri == NULL) return;
    for (index = 0u; index < document_count; ++index) {
        if (strcmp(documents[index].uri, uri) == 0) {
            free(documents[index].uri);
            free(documents[index].text);
            documents[index] = documents[document_count - 1u];
            --document_count;
            return;
        }
    }
}

/* Growable string buffer for building JSON responses. */
typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} StrBuf;

static void sb_init(StrBuf *sb) { sb->data = NULL; sb->length = 0u; sb->capacity = 0u; }
static void sb_free(StrBuf *sb) { free(sb->data); sb->data = NULL; sb->length = 0u; sb->capacity = 0u; }

static int sb_append(StrBuf *sb, const char *text) {
    size_t length = strlen(text);
    if (sb->length + length + 1u > sb->capacity) {
        size_t capacity = sb->capacity == 0u ? 256u : sb->capacity * 2u;
        char *grown;
        while (capacity < sb->length + length + 1u) capacity *= 2u;
        grown = realloc(sb->data, capacity);
        if (grown == NULL) return 0;
        sb->data = grown;
        sb->capacity = capacity;
    }
    memcpy(sb->data + sb->length, text, length);
    sb->length += length;
    sb->data[sb->length] = '\0';
    return 1;
}

static int sb_appendf(StrBuf *sb, const char *format, ...) {
    char buffer[512];
    va_list args;
    int written;
    va_start(args, format);
    written = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    if (written < 0) return 0;
    return sb_append(sb, buffer);
}

/* Run the compiler's --symbols mode and parse the JSON symbol table. */
static JsonValue *run_symbols(const char *compiler_path, bool compiler_found,
                              const char *uri, const char *text) {
    char path[4096];
    char *json_text;
    JsonValue *symbols;
    if (!compiler_found || uri == NULL || text == NULL) return NULL;
    uri_to_path(uri, path, sizeof(path));
    if (lana_compiler_symbols(compiler_path, text, path, &json_text) != 0) return NULL;
    if (json_text == NULL) return NULL;
    symbols = json_parse(json_text);
    free(json_text);
    return symbols;
}

static void document_position(const JsonValue *params, int *line, int *character) {
    JsonValue *position = json_get(params, "position");
    *line = (int)json_number(json_get(position, "line"));
    *character = (int)json_number(json_get(position, "character"));
}

/* Find the symbol name at a 0-based LSP position. Returns a pointer into the
 * symbol table (valid while `symbols` is alive), or NULL. */
static const char *symbol_at(const JsonValue *symbols, int line, int character) {
    int one_line = line + 1;
    int one_column = character + 1;
    JsonValue *references = json_get(symbols, "references");
    JsonValue *definitions = json_get(symbols, "definitions");
    size_t index;
    if (references != NULL && references->type == JSON_ARRAY) {
        for (index = 0u; index < references->as.array.count; ++index) {
            JsonValue *ref = references->as.array.items[index];
            const char *name = json_string(json_get(ref, "name"));
            int ref_line = (int)json_number(json_get(ref, "line"));
            int ref_column = (int)json_number(json_get(ref, "column"));
            if (name == NULL) continue;
            if (ref_line == one_line && ref_column <= one_column &&
                one_column < ref_column + (int)strlen(name)) return name;
        }
    }
    if (definitions != NULL && definitions->type == JSON_ARRAY) {
        for (index = 0u; index < definitions->as.array.count; ++index) {
            JsonValue *def = definitions->as.array.items[index];
            const char *name = json_string(json_get(def, "name"));
            int def_line = (int)json_number(json_get(def, "line"));
            int def_column = (int)json_number(json_get(def, "column"));
            if (name == NULL) continue;
            if (def_line == one_line && def_column <= one_column &&
                one_column < def_column + (int)strlen(name)) return name;
        }
    }
    return NULL;
}

static JsonValue *find_definition(const JsonValue *symbols, const char *name) {
    JsonValue *definitions = json_get(symbols, "definitions");
    size_t index;
    if (definitions == NULL || definitions->type != JSON_ARRAY) return NULL;
    for (index = 0u; index < definitions->as.array.count; ++index) {
        JsonValue *def = definitions->as.array.items[index];
        const char *def_name = json_string(json_get(def, "name"));
        if (def_name != NULL && strcmp(def_name, name) == 0) return def;
    }
    return NULL;
}

static int append_location(StrBuf *sb, const char *uri, const JsonValue *entry) {
    const char *name = json_string(json_get(entry, "name"));
    int line = (int)json_number(json_get(entry, "line")) - 1;
    int column = (int)json_number(json_get(entry, "column")) - 1;
    int length = name == NULL ? 1 : (int)strlen(name);
    return sb_appendf(sb,
        "{\"uri\":\"%s\",\"range\":{\"start\":{\"line\":%d,\"character\":%d},"
        "\"end\":{\"line\":%d,\"character\":%d}}}",
        uri, line, column, line, column + length);
}

static char *format_references(const char *uri, const JsonValue *symbols, const char *name) {
    JsonValue *references = json_get(symbols, "references");
    StrBuf sb;
    size_t index;
    sb_init(&sb);
    if (!sb_append(&sb, "[")) { sb_free(&sb); return NULL; }
    if (references != NULL && references->type == JSON_ARRAY) {
        for (index = 0u; index < references->as.array.count; ++index) {
            JsonValue *ref = references->as.array.items[index];
            const char *ref_name = json_string(json_get(ref, "name"));
            if (ref_name == NULL || strcmp(ref_name, name) != 0) continue;
            if (sb.length > 1u && !sb_append(&sb, ",")) { sb_free(&sb); return NULL; }
            if (!append_location(&sb, uri, ref)) { sb_free(&sb); return NULL; }
        }
    }
    if (!sb_append(&sb, "]")) { sb_free(&sb); return NULL; }
    return sb.data;
}

static char *format_completion(const JsonValue *symbols) {
    JsonValue *definitions = json_get(symbols, "definitions");
    StrBuf sb;
    size_t index;
    int first = 1;
    sb_init(&sb);
    if (!sb_append(&sb, "{\"isIncomplete\":false,\"items\":[")) { sb_free(&sb); return NULL; }
    if (definitions != NULL && definitions->type == JSON_ARRAY) {
        for (index = 0u; index < definitions->as.array.count; ++index) {
            JsonValue *def = definitions->as.array.items[index];
            const char *name = json_string(json_get(def, "name"));
            const char *kind = json_string(json_get(def, "kind"));
            const char *type = json_string(json_get(def, "type"));
            int item_kind = 6; /* variable */
            if (name == NULL) continue;
            if (kind != NULL && strcmp(kind, "function") == 0) item_kind = 3;
            if (kind != NULL && strcmp(kind, "parameter") == 0) item_kind = 6;
            if (!first && !sb_append(&sb, ",")) { sb_free(&sb); return NULL; }
            first = 0;
            if (!sb_appendf(&sb, "{\"label\":\"%s\",\"kind\":%d,\"detail\":\"%s\"}",
                            name, item_kind, type == NULL ? "unknown" : type)) {
                sb_free(&sb); return NULL;
            }
        }
    }
    if (!sb_append(&sb, "]}")) { sb_free(&sb); return NULL; }
    return sb.data;
}

static int append_edit(StrBuf *sb, const JsonValue *entry, const char *new_name) {
    const char *name = json_string(json_get(entry, "name"));
    int line = (int)json_number(json_get(entry, "line")) - 1;
    int column = (int)json_number(json_get(entry, "column")) - 1;
    int length = name == NULL ? 1 : (int)strlen(name);
    return sb_appendf(sb,
        "{\"range\":{\"start\":{\"line\":%d,\"character\":%d},"
        "\"end\":{\"line\":%d,\"character\":%d}},\"newText\":\"%s\"}",
        line, column, line, column + length, new_name);
}

static char *format_rename(const char *uri, const JsonValue *symbols,
                           const char *name, const char *new_name) {
    JsonValue *references = json_get(symbols, "references");
    JsonValue *definition = find_definition(symbols, name);
    StrBuf sb;
    size_t index;
    sb_init(&sb);
    if (!sb_appendf(&sb, "{\"changes\":{\"%s\":[", uri)) { sb_free(&sb); return NULL; }
    if (definition != NULL) {
        if (!append_edit(&sb, definition, new_name)) { sb_free(&sb); return NULL; }
    }
    if (references != NULL && references->type == JSON_ARRAY) {
        for (index = 0u; index < references->as.array.count; ++index) {
            JsonValue *ref = references->as.array.items[index];
            const char *ref_name = json_string(json_get(ref, "name"));
            if (ref_name == NULL || strcmp(ref_name, name) != 0) continue;
            if (sb.length > 1u && !sb_append(&sb, ",")) { sb_free(&sb); return NULL; }
            if (!append_edit(&sb, ref, new_name)) { sb_free(&sb); return NULL; }
        }
    }
    if (!sb_appendf(&sb, "]}}")) { sb_free(&sb); return NULL; }
    return sb.data;
}

static void handle_hover(const char *id, const char *compiler_path, bool compiler_found,
                         const JsonValue *params) {
    const char *uri = document_uri(params);
    const char *text = document_get(uri);
    int line, character;
    JsonValue *symbols;
    const char *name;
    document_position(params, &line, &character);
    symbols = run_symbols(compiler_path, compiler_found, uri, text);
    if (symbols == NULL) { send_result(id, "null"); return; }
    name = symbol_at(symbols, line, character);
    if (name == NULL) { json_free(symbols); send_result(id, "null"); return; }
    {
        JsonValue *def = find_definition(symbols, name);
        const char *type = def == NULL ? NULL : json_string(json_get(def, "type"));
        const char *kind = def == NULL ? NULL : json_string(json_get(def, "kind"));
        char *result;
        size_t needed = strlen(name) + (type == NULL ? 7u : strlen(type)) +
                        (kind == NULL ? 8u : strlen(kind)) + 128u;
        result = malloc(needed);
        if (result == NULL) { json_free(symbols); send_result(id, "null"); return; }
        (void)snprintf(result, needed,
            "{\"contents\":{\"kind\":\"markdown\",\"value\":\"`%s`: %s (%s)\"}}",
            name, type == NULL ? "unknown" : type, kind == NULL ? "variable" : kind);
        send_result(id, result);
        free(result);
    }
    json_free(symbols);
}

static void handle_definition(const char *id, const char *compiler_path, bool compiler_found,
                              const JsonValue *params) {
    const char *uri = document_uri(params);
    const char *text = document_get(uri);
    int line, character;
    JsonValue *symbols;
    const char *name;
    JsonValue *def;
    char *result;
    document_position(params, &line, &character);
    symbols = run_symbols(compiler_path, compiler_found, uri, text);
    if (symbols == NULL) { send_result(id, "[]"); return; }
    name = symbol_at(symbols, line, character);
    if (name == NULL) { json_free(symbols); send_result(id, "[]"); return; }
    def = find_definition(symbols, name);
    if (def == NULL) { json_free(symbols); send_result(id, "[]"); return; }
    {
        StrBuf sb;
        sb_init(&sb);
        if (!sb_append(&sb, "[") || !append_location(&sb, uri, def) || !sb_append(&sb, "]")) {
            sb_free(&sb); json_free(symbols); send_result(id, "[]"); return;
        }
        result = sb.data;
    }
    send_result(id, result);
    free(result);
    json_free(symbols);
}

static void handle_references(const char *id, const char *compiler_path, bool compiler_found,
                              const JsonValue *params) {
    const char *uri = document_uri(params);
    const char *text = document_get(uri);
    int line, character;
    JsonValue *symbols;
    const char *name;
    char *result;
    document_position(params, &line, &character);
    symbols = run_symbols(compiler_path, compiler_found, uri, text);
    if (symbols == NULL) { send_result(id, "[]"); return; }
    name = symbol_at(symbols, line, character);
    if (name == NULL) { json_free(symbols); send_result(id, "[]"); return; }
    result = format_references(uri, symbols, name);
    if (result == NULL) { json_free(symbols); send_result(id, "[]"); return; }
    send_result(id, result);
    free(result);
    json_free(symbols);
}

static void handle_completion(const char *id, const char *compiler_path, bool compiler_found,
                              const JsonValue *params) {
    const char *uri = document_uri(params);
    const char *text = document_get(uri);
    JsonValue *symbols;
    char *result;
    symbols = run_symbols(compiler_path, compiler_found, uri, text);
    if (symbols == NULL) { send_result(id, "{\"isIncomplete\":false,\"items\":[]}"); return; }
    result = format_completion(symbols);
    if (result == NULL) { json_free(symbols); send_result(id, "{\"isIncomplete\":false,\"items\":[]}"); return; }
    send_result(id, result);
    free(result);
    json_free(symbols);
}

static void handle_rename(const char *id, const char *compiler_path, bool compiler_found,
                          const JsonValue *params) {
    const char *uri = document_uri(params);
    const char *text = document_get(uri);
    const char *new_name = json_string(json_get(params, "newName"));
    int line, character;
    JsonValue *symbols;
    const char *name;
    char *result;
    document_position(params, &line, &character);
    symbols = run_symbols(compiler_path, compiler_found, uri, text);
    if (symbols == NULL || new_name == NULL) { json_free(symbols); send_result(id, "{\"changes\":{}}"); return; }
    name = symbol_at(symbols, line, character);
    if (name == NULL) { json_free(symbols); send_result(id, "{\"changes\":{}}"); return; }
    result = format_rename(uri, symbols, name, new_name);
    if (result == NULL) { json_free(symbols); send_result(id, "{\"changes\":{}}"); return; }
    send_result(id, result);
    free(result);
    json_free(symbols);
}

int lana_lsp_run(const char *executable) {
    char header[256];
    char compiler_path[4096];
    bool compiler_found;
    int shutdown_requested = 0;
    compiler_found = lana_compiler_find(executable, compiler_path, sizeof(compiler_path));
    for (;;) {
        size_t length = 0u;
        char *body;
        JsonValue *request;
        JsonValue *method_value;
        JsonValue *id_value;
        JsonValue *params;
        const char *method;
        char id[64];
        while (fgets(header, sizeof(header), stdin) != NULL) {
            if (sscanf(header, "Content-Length: %zu", &length) == 1) continue;
            if (strcmp(header, "\r\n") == 0 || strcmp(header, "\n") == 0) break;
        }
        if (length == 0u || feof(stdin)) break;
        body = malloc(length + 1u);
        if (body == NULL || fread(body, 1u, length, stdin) != length) {
            free(body); return 1;
        }
        body[length] = '\0';
        request = json_parse(body);
        if (request == NULL) { free(body); continue; }
        method_value = json_get(request, "method");
        id_value = json_get(request, "id");
        params = json_get(request, "params");
        method = json_string(method_value);
        format_id(id_value, id, sizeof(id));
        if (method == NULL) {
            json_free(request); free(body); continue;
        }
        if (strcmp(method, "exit") == 0) {
            json_free(request); free(body);
            return shutdown_requested ? 0 : 1;
        }
        if (strcmp(method, "textDocument/didOpen") == 0 ||
            strcmp(method, "textDocument/didChange") == 0) {
            const char *uri = document_uri(params);
            const char *text = document_text(params);
            document_set(uri, text);
            check_and_publish(compiler_path, compiler_found, uri, text);
        } else if (strcmp(method, "textDocument/didClose") == 0) {
            const char *uri = document_uri(params);
            document_remove(uri);
            publish_diagnostics(uri == NULL ? "" : uri, "[]");
        } else if (strcmp(method, "initialize") == 0) {
            send_result(id, "{\"serverInfo\":{\"name\":\"lana-lsp\",\"version\":\"2.0\"},\"capabilities\":{\"textDocumentSync\":1,\"hoverProvider\":true,\"completionProvider\":{},\"definitionProvider\":true,\"referencesProvider\":true,\"renameProvider\":{\"prepareProvider\":true}}}");
        } else if (strcmp(method, "shutdown") == 0) {
            shutdown_requested = 1;
            send_result(id, "null");
        } else if (strcmp(method, "textDocument/hover") == 0) {
            handle_hover(id, compiler_path, compiler_found, params);
        } else if (strcmp(method, "textDocument/completion") == 0) {
            handle_completion(id, compiler_path, compiler_found, params);
        } else if (strcmp(method, "textDocument/definition") == 0) {
            handle_definition(id, compiler_path, compiler_found, params);
        } else if (strcmp(method, "textDocument/references") == 0) {
            handle_references(id, compiler_path, compiler_found, params);
        } else if (strcmp(method, "textDocument/rename") == 0) {
            handle_rename(id, compiler_path, compiler_found, params);
        } else {
            send_result(id, "[]");
        }
        json_free(request);
        free(body);
    }
    return 0;
}
