/* lana_http_service: minimal evidence HTTP server for the HTTP_JSON adapter.
 *
 * Binds 127.0.0.1 only. GET /evidence/<id> serves <data-dir>/<id>.json with
 * an X-Lana-Schema: 1 header and a {"schema":1,"ok":true,"evidence":...} body.
 * Errors are structured: {"schema":1,"ok":false,"error":{"code":...}}.
 */

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define LANA_HTTP_MAX_FILE 1048576u

static const char *status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        default: return "Internal Server Error";
    }
}

static void write_all(int client, const char *data, size_t length) {
    while (length != 0u) {
        ssize_t written = write(client, data, length);
        if (written > 0) {
            data += written;
            length -= (size_t)written;
        } else if (written < 0 && errno != EINTR) {
            return;
        }
    }
}

static void send_response(int client, int status, const char *body, size_t length) {
    char header[512];
    int header_length = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "X-Lana-Schema: 1\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text(status), length);
    if (header_length > 0 && (size_t)header_length < sizeof(header))
        write_all(client, header, (size_t)header_length);
    if (length != 0u) write_all(client, body, length);
}

static void send_json_error(int client, int status, const char *code, const char *message) {
    char body[1024];
    int length = snprintf(body, sizeof(body),
        "{\"schema\":1,\"ok\":false,\"error\":{\"code\":\"%s\",\"message\":\"%s\"}}",
        code, message);
    send_response(client, status, body, (size_t)length);
}

static bool header_end_seen(const char *buffer, size_t length) {
    size_t index;
    if (length < 4u) return false;
    for (index = 0u; index + 3u < length; ++index)
        if (buffer[index] == '\r' && buffer[index + 1u] == '\n' &&
            buffer[index + 2u] == '\r' && buffer[index + 3u] == '\n')
            return true;
    return false;
}

static void handle_client(int client, const char *data_dir, size_t limit) {
    char *request;
    size_t received = 0u;
    bool over_limit = false;
    char method[16], path[1024];
    const char *id;
    char file_path[1024];
    char *content;
    long file_length;
    FILE *file;
    char *body;
    size_t body_length;

    request = malloc(limit + 1u);
    if (request == NULL) { send_json_error(client, 500, "LANA_ERR_OOM", "out of memory"); return; }
    while (received < limit) {
        ssize_t n = read(client, request + received, limit - received);
        if (n <= 0) break;
        received += (size_t)n;
        if (header_end_seen(request, received)) break;
    }
    if (received >= limit && !header_end_seen(request, received)) over_limit = true;
    request[received] = '\0';
    if (over_limit) { send_json_error(client, 413, "LANA_ERR_LIMIT", "request too large"); free(request); return; }

    if (sscanf(request, "%15s %1023s", method, path) != 2) {
        send_json_error(client, 400, "LANA_ERR_PARSE", "malformed request");
        free(request); return;
    }
    if (strcmp(method, "GET") != 0) {
        send_json_error(client, 405, "LANA_ERR_UNSUPPORTED_OPERATION", "method not allowed");
        free(request); return;
    }
    if (strncmp(path, "/evidence/", 10u) != 0) {
        send_json_error(client, 404, "LANA_ERR_NOT_FOUND", "unknown path");
        free(request); return;
    }
    id = path + 10u;
    if (id[0] == '\0' || strchr(id, '/') != NULL || strchr(id, '.') != NULL) {
        send_json_error(client, 400, "LANA_ERR_PARSE", "invalid evidence id");
        free(request); return;
    }
    (void)snprintf(file_path, sizeof(file_path), "%s/%s.json", data_dir, id);
    file = fopen(file_path, "rb");
    if (file == NULL) {
        send_json_error(client, 404, "LANA_ERR_NOT_FOUND", "evidence not found");
        free(request); return;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (file_length = ftell(file)) < 0 ||
        (size_t)file_length > LANA_HTTP_MAX_FILE || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        send_json_error(client, 500, "LANA_ERR_IO", "evidence unreadable");
        free(request); return;
    }
    content = malloc((size_t)file_length + 1u);
    if (content == NULL) {
        fclose(file);
        send_json_error(client, 500, "LANA_ERR_OOM", "out of memory");
        free(request); return;
    }
    if (fread(content, 1u, (size_t)file_length, file) != (size_t)file_length) {
        fclose(file); free(content);
        send_json_error(client, 500, "LANA_ERR_IO", "evidence unreadable");
        free(request); return;
    }
    fclose(file);
    content[file_length] = '\0';

    /* Body: {"schema":1,"ok":true,"evidence":<content>} */
    body_length = (size_t)file_length + 40u;
    body = malloc(body_length);
    if (body == NULL) {
        free(content);
        send_json_error(client, 500, "LANA_ERR_OOM", "out of memory");
        free(request); return;
    }
    body_length = (size_t)snprintf(body, body_length,
        "{\"schema\":1,\"ok\":true,\"evidence\":%s}", content);
    send_response(client, 200, body, body_length);
    free(body); free(content); free(request);
}

int main(int argc, char **argv) {
    int port = 8080;
    size_t limit = 4096u;
    const char *data_dir = ".";
    int index;
    int server, opt = 1;
    struct sockaddr_in addr;
    socklen_t addr_length;

    for (index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--port") == 0 && index + 1 < argc) {
            port = atoi(argv[++index]);
        } else if (strcmp(argv[index], "--data") == 0 && index + 1 < argc) {
            data_dir = argv[++index];
        } else if (strcmp(argv[index], "--limit") == 0 && index + 1 < argc) {
            long parsed = atol(argv[++index]);
            if (parsed > 0) limit = (size_t)parsed;
        } else {
            fprintf(stderr, "usage: %s [--port N] [--data DIR] [--limit N]\n", argv[0]);
            return 2;
        }
    }

    (void)signal(SIGPIPE, SIG_IGN);
    server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) { perror("socket"); return 1; }
    (void)setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(server, (struct sockaddr *)&addr, sizeof(addr)) != 0) { perror("bind"); return 1; }
    if (port == 0) {
        addr_length = sizeof(addr);
        if (getsockname(server, (struct sockaddr *)&addr, &addr_length) == 0) {
            printf("%u\n", (unsigned)ntohs(addr.sin_port));
            fflush(stdout);
        }
    }
    if (listen(server, 16) != 0) { perror("listen"); return 1; }
    for (;;) {
        int client = accept(server, NULL, NULL);
        if (client < 0) continue;
        handle_client(client, data_dir, limit);
        close(client);
    }
}
