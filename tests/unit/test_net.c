/* LIP-019 networking unit test (C VM).
 *
 * Exercises the real-network paths that are C-only (non-deterministic, so not
 * differential fixtures): http_get/http_post against a local HTTP server,
 * socket_connect/send/recv/close against a local echo server, a read timeout
 * against a holding server, and capability denial.
 *
 * The differential fixtures (tests/conformance/differential/net) cover the
 * deterministic paths (capability denial, type rejection, timeout-as-Result)
 * byte-identically across the C11 and Rust VMs.
 */

#include "assembler.h"
#include "error.h"
#include "vm.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                      __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

/* A minimal HTTP server: responds to GET with a fixed body, to POST by echoing
 * the request body, and to a path of /hold by holding the connection open. */
typedef struct {
    int port;
    int hold;
    volatile int *ready;
} ServerConfig;

static void *http_server(void *arg) {
    ServerConfig *cfg = (ServerConfig *)arg;
    int listen_fd, client_fd;
    struct sockaddr_in addr;
    socklen_t len = sizeof(addr);
    char buf[8192];
    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) return NULL;
    {
        int one = 1;
        (void)setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    }
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)cfg->port);
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) return NULL;
    if (listen(listen_fd, 8) < 0) return NULL;
    if (cfg->ready != NULL) *cfg->ready = 1;
    for (;;) {
        client_fd = accept(listen_fd, (struct sockaddr *)&addr, &len);
        if (client_fd < 0) continue;
        if (cfg->hold) {
            /* Hold the connection open without responding. */
            while (recv(client_fd, buf, sizeof(buf), 0) > 0) { }
            close(client_fd);
            continue;
        }
        {
            ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
            if (n > 0) {
                buf[n] = '\0';
                if (strncmp(buf, "POST ", 5) == 0) {
                    const char *body = strstr(buf, "\r\n\r\n");
                    const char *resp = body ? body + 4 : "";
                    char out[8192];
                    (void)snprintf(out, sizeof(out),
                        "HTTP/1.1 200 OK\r\nContent-Length: %zu\r\n\r\n%s",
                        strlen(resp), resp);
                    (void)send(client_fd, out, strlen(out), 0);
                } else {
                    const char *resp = "hello from lana";
                    char out[8192];
                    (void)snprintf(out, sizeof(out),
                        "HTTP/1.1 200 OK\r\nContent-Length: %zu\r\n\r\n%s",
                        strlen(resp), resp);
                    (void)send(client_fd, out, strlen(out), 0);
                }
            }
            close(client_fd);
        }
    }
    return NULL;
}

/* Assemble `source` into `chunk`. */
static int assemble(const char *source, LanaChunk *chunk) {
    char path[] = "/tmp/lana-net-XXXXXX.lasm";
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
    char out_path[] = "/tmp/lana-net-out-XXXXXX";
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

static int test_http_get(void) {
    LanaChunk chunk;
    char *out = NULL;
    LanaError result;
    char source[1024];
    printf("Testing http_get against local server...\n");
    (void)snprintf(source, sizeof(source),
        ".function main 0 16\n"
        "LOAD_CONST R0 net\n"
        "HOST_CALL shared_information R0 1 R1\n"
        "LOAD_CONST R2 use\n"
        "HOST_CALL grant R1 2 R3\n"
        "LOAD_STRING R4 687474703a2f2f3132372e302e302e313a31383039302f\n"
        "LOAD_CONST R5 k\n"
        "LOAD_CONST R6 0\n"
        "HOST_CALL map_new R5 2 R7\n"
        "MOVE R5 R7\n"
        "LOAD_CONST R6 5000\n"
        "HOST_CALL http_get R4 3 R8\n"
        "PRINT R8\n"
        "RETURN R0\n");
    CHECK(assemble(source, &chunk) == 0);
    result = run_capture(&chunk, &out);
    CHECK(result == LANA_OK);
    CHECK(out != NULL && strstr(out, "\"status\": 200") != NULL);
    CHECK(out != NULL && strstr(out, "hello from lana") != NULL);
    free(out);
    return 0;
}

static int test_http_post(void) {
    LanaChunk chunk;
    char *out = NULL;
    LanaError result;
    char source[1024];
    printf("Testing http_post against local server...\n");
    (void)snprintf(source, sizeof(source),
        ".function main 0 16\n"
        "LOAD_CONST R0 net\n"
        "HOST_CALL shared_information R0 1 R1\n"
        "LOAD_CONST R2 use\n"
        "HOST_CALL grant R1 2 R3\n"
        "LOAD_STRING R4 687474703a2f2f3132372e302e302e313a31383039302f\n"
        "LOAD_STRING R5 70696e67\n"
        "LOAD_CONST R6 k\n"
        "LOAD_CONST R7 0\n"
        "HOST_CALL map_new R6 2 R8\n"
        "MOVE R6 R8\n"
        "LOAD_CONST R7 5000\n"
        "HOST_CALL http_post R4 4 R9\n"
        "PRINT R9\n"
        "RETURN R0\n");
    CHECK(assemble(source, &chunk) == 0);
    result = run_capture(&chunk, &out);
    CHECK(result == LANA_OK);
    CHECK(out != NULL && strstr(out, "\"status\": 200") != NULL);
    CHECK(out != NULL && strstr(out, "ping") != NULL);
    free(out);
    return 0;
}

static int test_socket_echo(void) {
    LanaChunk chunk;
    char *out = NULL;
    LanaError result;
    printf("Testing socket_connect/send/recv/close...\n");
    CHECK(assemble(
        ".function main 0 16\n"
        "LOAD_CONST R0 net\n"
        "HOST_CALL shared_information R0 1 R1\n"
        "LOAD_CONST R2 use\n"
        "HOST_CALL grant R1 2 R3\n"
        "LOAD_STRING R4 3132372e302e302e31\n"
        "LOAD_CONST R5 18091\n"
        "HOST_CALL socket_connect R4 2 R6\n"
        "PRINT R6\n"
        "LOAD_STRING R7 68656c6c6f\n"
        "HOST_CALL socket_send R6 2 R8\n"
        "PRINT R8\n"
        "LOAD_CONST R7 100\n"
        "HOST_CALL socket_recv R6 2 R10\n"
        "PRINT R10\n"
        "HOST_CALL socket_close R6 1 R11\n"
        "RETURN R0\n",
        &chunk) == 0);
    result = run_capture(&chunk, &out);
    CHECK(result == LANA_OK);
    CHECK(out != NULL && strstr(out, "0\n") != NULL);
    CHECK(out != NULL && strstr(out, "5\n") != NULL);
    CHECK(out != NULL && strstr(out, "hello") != NULL);
    free(out);
    return 0;
}

static int test_timeout(void) {
    LanaChunk chunk;
    char *out = NULL;
    LanaError result;
    printf("Testing read timeout against holding server...\n");
    CHECK(assemble(
        ".function main 0 16\n"
        "LOAD_CONST R0 net\n"
        "HOST_CALL shared_information R0 1 R1\n"
        "LOAD_CONST R2 use\n"
        "HOST_CALL grant R1 2 R3\n"
        "LOAD_STRING R4 687474703a2f2f3132372e302e302e313a31383039322f\n"
        "LOAD_CONST R5 k\n"
        "LOAD_CONST R6 0\n"
        "HOST_CALL map_new R5 2 R7\n"
        "MOVE R5 R7\n"
        "LOAD_CONST R6 50\n"
        "HOST_CALL http_get R4 3 R8\n"
        "PRINT R8\n"
        "RETURN R0\n",
        &chunk) == 0);
    result = run_capture(&chunk, &out);
    CHECK(result == LANA_OK);
    CHECK(out != NULL && strstr(out, "\"error\": timeout") != NULL);
    free(out);
    return 0;
}

static int test_capability_denial(void) {
    LanaChunk chunk;
    char *out = NULL;
    LanaError result;
    printf("Testing http_get capability denial...\n");
    CHECK(assemble(
        ".function main 0 16\n"
        "LOAD_STRING R0 687474703a2f2f3132372e302e302e313a31383039302f\n"
        "LOAD_CONST R1 k\n"
        "LOAD_CONST R2 0\n"
        "HOST_CALL map_new R1 2 R3\n"
        "MOVE R1 R3\n"
        "LOAD_CONST R2 5000\n"
        "HOST_CALL http_get R0 3 R5\n"
        "PRINT R5\n"
        "RETURN R0\n",
        &chunk) == 0);
    result = run_capture(&chunk, &out);
    CHECK(result == LANA_ERR_CAPABILITY);
    free(out);
    return 0;
}

int main(void) {
    pthread_t http_thread, echo_thread, hold_thread;
    volatile int http_ready = 0, echo_ready = 0, hold_ready = 0;
    ServerConfig http_cfg = {18090, 0, &http_ready};
    ServerConfig echo_cfg = {18091, 0, &echo_ready};
    ServerConfig hold_cfg = {18092, 1, &hold_ready};
    /* The echo server echoes back whatever it receives. */
    pthread_create(&http_thread, NULL, http_server, &http_cfg);
    pthread_create(&echo_thread, NULL, http_server, &echo_cfg);
    pthread_create(&hold_thread, NULL, http_server, &hold_cfg);
    /* Wait until each server has bound its listening socket. */
    for (int i = 0; i < 100; i++) {
        if (http_ready && echo_ready && hold_ready) break;
        usleep(10000);
    }
    CHECK(http_ready && echo_ready && hold_ready);

    if (test_http_get() != 0) return 1;
    if (test_http_post() != 0) return 1;
    if (test_socket_echo() != 0) return 1;
    if (test_timeout() != 0) return 1;
    if (test_capability_denial() != 0) return 1;
    printf("All networking tests passed.\n");
    return 0;
}
