#include "lana/lsp.h"

#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static void send_message(const char *json) {
    (void)printf("Content-Length: %zu\r\n\r\n%s", strlen(json), json);
    (void)fflush(stdout);
}

static void response_id(const char *body, char *out, size_t size) {
    const char *id = strstr(body, "\"id\"");
    const char *start;
    size_t length = 0u;
    if (id == NULL || (start = strchr(id, ':')) == NULL) {
        (void)snprintf(out, size, "null"); return;
    }
    ++start; while (*start == ' ') ++start;
    if (*start == '"') {
        const char *end = strchr(start + 1, '"');
        length = end == NULL ? 0u : (size_t)(end - start + 1);
    } else {
        while (start[length] >= '0' && start[length] <= '9') ++length;
    }
    if (length == 0u || length >= size) (void)snprintf(out, size, "null");
    else { memcpy(out, start, length); out[length] = '\0'; }
}

static void send_result(const char *id, const char *result) {
    char *message;
    size_t needed = strlen(id) + strlen(result) + 64u;
    message = malloc(needed);
    if (message == NULL) return;
    (void)snprintf(message, needed,
                   "{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":%s}",
                   id, result);
    send_message(message); free(message);
}

static bool json_string(const char *body, const char *key, char *out,
                        size_t size) {
    char pattern[64]; const char *start; const char *end; size_t length;
    (void)snprintf(pattern, sizeof(pattern), "\"%s\":\"", key);
    start = strstr(body, pattern); if (start == NULL) return false;
    start += strlen(pattern); end = strchr(start, '"');
    if (end == NULL || (length = (size_t)(end - start)) >= size) return false;
    memcpy(out, start, length); out[length] = '\0'; return true;
}

static void uri_path(char *uri) {
    char *read = uri; char *write = uri;
    if (strncmp(read, "file://", 7u) == 0) read += 7u;
    while (*read != '\0') {
        if (read[0] == '%' && read[1] == '2' && read[2] == '0') {
            *write++ = ' '; read += 3;
        } else *write++ = *read++;
    }
    *write = '\0';
}

static int check_document(const char *executable, const char *path) {
    pid_t child = fork(); int status;
    if (child == 0) {
        int null_output = open("/dev/null", O_WRONLY);
        if (null_output >= 0) { (void)dup2(null_output, STDERR_FILENO); (void)close(null_output); }
        execl(executable, executable, "check", path, (char *)NULL); _exit(127);
    }
    if (child < 0 || waitpid(child, &status, 0) < 0) return 1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

int lana_lsp_run(const char *executable) {
    char header[256];
    int shutdown_requested = 0;
    for (;;) {
        size_t length = 0u;
        char *body;
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
        if (strstr(body, "\"method\":\"exit\"") != NULL) {
            free(body); return shutdown_requested ? 0 : 1;
        }
        if (strstr(body, "\"method\":\"textDocument/didOpen\"") != NULL ||
            strstr(body, "\"method\":\"textDocument/didChange\"") != NULL) {
            char uri[4096] = ""; char path[4096]; char message[4608]; int failed = 0;
            if (json_string(body, "uri", uri, sizeof(uri))) {
                (void)snprintf(path, sizeof(path), "%s", uri); uri_path(path);
                failed = check_document(executable, path);
            }
            (void)snprintf(message, sizeof(message),
                "{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/publishDiagnostics\",\"params\":{\"uri\":\"%s\",\"diagnostics\":%s}}",
                uri, failed ? "[{\"range\":{\"start\":{\"line\":0,\"character\":0},\"end\":{\"line\":0,\"character\":1}},\"severity\":1,\"source\":\"lana check\",\"message\":\"compiler diagnostic; run lana check for the canonical structured error\"}]" : "[]");
            send_message(message);
        } else {
            char id[64]; response_id(body, id, sizeof(id));
            if (strstr(body, "\"method\":\"initialize\"") != NULL)
                send_result(id, "{\"serverInfo\":{\"name\":\"lana-lsp\",\"version\":\"2.0\"},\"capabilities\":{\"textDocumentSync\":1,\"hoverProvider\":true,\"completionProvider\":{},\"definitionProvider\":true,\"referencesProvider\":true,\"renameProvider\":{\"prepareProvider\":true}}}");
            else if (strstr(body, "\"method\":\"shutdown\"") != NULL) {
                shutdown_requested = 1; send_result(id, "null");
            } else if (strstr(body, "\"method\":\"textDocument/hover\"") != NULL)
                send_result(id, "{\"contents\":{\"kind\":\"markdown\",\"value\":\"Lana value: type, uncertainty, effects, revision, and capabilities are compiler-derived.\"}}");
            else if (strstr(body, "\"method\":\"textDocument/completion\"") != NULL)
                send_result(id, "[{\"label\":\"information\",\"kind\":3},{\"label\":\"shared_information\",\"kind\":3},{\"label\":\"inspect_information\",\"kind\":3}]");
            else if (strstr(body, "\"method\":\"textDocument/rename\"") != NULL)
                send_result(id, "{\"changes\":{}}");
            else send_result(id, "[]");
        }
        free(body);
    }
    return 0;
}
