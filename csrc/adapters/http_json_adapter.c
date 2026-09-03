#include "lana/adapters.h"
#include "lana/data.h"
#include "lana/vm.h"
#include "adapter_plugin.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

typedef struct {
    int port;
} HttpAdapter;

LanaError lana_adapter_plugin_load(const LanaAdapterOptions *options, void **out_ctx) {
    HttpAdapter *adapter;
    if (options == NULL || out_ctx == NULL) return LANA_ERR_INVALID_STATE;
    adapter = calloc(1u, sizeof(*adapter));
    if (adapter == NULL) return LANA_ERR_OOM;
    adapter->port = options->config == NULL ? 8080 : atoi(options->config);
    *out_ctx = adapter;
    return LANA_OK;
}

static LanaError map_error_code(const char *code) {
    if (strcmp(code, "LANA_ERR_NOT_FOUND") == 0) return LANA_ERR_NOT_FOUND;
    if (strcmp(code, "LANA_ERR_LIMIT") == 0) return LANA_ERR_LIMIT;
    if (strcmp(code, "LANA_ERR_PARSE") == 0) return LANA_ERR_PARSE;
    if (strcmp(code, "LANA_ERR_UNSUPPORTED_OPERATION") == 0) return LANA_ERR_UNSUPPORTED_OPERATION;
    return LANA_ERR_IO;
}

LanaError lana_adapter_plugin_fetch(void *ctx, LanaVM *vm, const char *query,
                                    Value *out_value) {
    HttpAdapter *adapter = (HttpAdapter *)ctx;
    int sock;
    struct sockaddr_in server;
    char request[1024];
    char response[65536];
    size_t total = 0u;
    char *body;
    Value parsed;
    LanaError error;
    if (adapter == NULL || vm == NULL || query == NULL || out_value == NULL)
        return LANA_ERR_INVALID_STATE;
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return LANA_ERR_IO;
    memset(&server, 0, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons((uint16_t)adapter->port);
    if (inet_pton(AF_INET, "127.0.0.1", &server.sin_addr) != 1) { close(sock); return LANA_ERR_IO; }
    if (connect(sock, (struct sockaddr *)&server, sizeof(server)) != 0) { close(sock); return LANA_ERR_IO; }
    (void)snprintf(request, sizeof(request),
                   "GET /%s HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n", query);
    if (write(sock, request, strlen(request)) < 0) { close(sock); return LANA_ERR_IO; }
    while (total < sizeof(response) - 1u) {
        ssize_t n = read(sock, response + total, sizeof(response) - 1u - total);
        if (n <= 0) break;
        total += (size_t)n;
    }
    close(sock);
    if (total == 0u) return LANA_ERR_IO;
    response[total] = '\0';
    body = strstr(response, "\r\n\r\n");
    if (body == NULL) return LANA_ERR_PARSE;
    body += 4;
    error = lana_json_parse(vm, body, &parsed);
    if (error != LANA_OK) return error;
    if (parsed.type != VAL_MAP) return LANA_ERR_CORRUPTION;
    {
        Value ok_value, evidence_value;
        if (lana_map_get(parsed.as.map, "ok", &ok_value) != LANA_OK || ok_value.type != VAL_BOOL)
            return LANA_ERR_CORRUPTION;
        if (!ok_value.as.boolean) {
            Value error_value;
            if (lana_map_get(parsed.as.map, "error", &error_value) != LANA_OK || error_value.type != VAL_MAP)
                return LANA_ERR_CORRUPTION;
            if (lana_map_get(error_value.as.map, "code", &error_value) != LANA_OK || error_value.type != VAL_STRING)
                return LANA_ERR_CORRUPTION;
            return map_error_code(error_value.as.string);
        }
        if (lana_map_get(parsed.as.map, "evidence", &evidence_value) != LANA_OK)
            return LANA_ERR_CORRUPTION;
        *out_value = evidence_value;
    }
    return LANA_OK;
}

void lana_adapter_plugin_close(void *ctx) { free(ctx); }
