#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "lana/adapters.h"
#include "lana/data.h"
#include "lana/vm.h"

#ifdef LANA_ADAPTER_HTTP_AVAILABLE

static pid_t start_service(const char *data_dir, int *out_port) {
    int pipe_fds[2];
    pid_t pid;
    char port_text[32];
    size_t length = 0u;
    assert(pipe(pipe_fds) == 0);
    pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        char port_arg[8], data_arg[8];
        dup2(pipe_fds[1], STDOUT_FILENO);
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        (void)snprintf(port_arg, sizeof(port_arg), "--port");
        (void)snprintf(data_arg, sizeof(data_arg), "--data");
        execl(LANA_HTTP_SERVICE, "lana_http_service", port_arg, "0", data_arg, data_dir, (char *)NULL);
        _exit(127);
    }
    close(pipe_fds[1]);
    while (length + 1u < sizeof(port_text)) {
        ssize_t n = read(pipe_fds[0], port_text + length, 1u);
        if (n <= 0) break;
        if (port_text[length] == '\n') break;
        ++length;
    }
    close(pipe_fds[0]);
    port_text[length] = '\0';
    *out_port = atoi(port_text);
    assert(*out_port > 0);
    return pid;
}

static void stop_service(pid_t pid) {
    (void)kill(pid, SIGTERM);
    (void)waitpid(pid, NULL, 0);
}

static void test_http_adapter(void) {
    char path[] = "/tmp/lana-adapter-http-XXXXXX";
    char file[256];
    char port_config[16];
    pid_t pid;
    int port;
    LanaVM vm;
    LanaAdapterOptions options = {sizeof(options), 1u, ADAPTER_HTTP_JSON, NULL};
    void *adapter = NULL;
    Value out;
    assert(mkdtemp(path) != NULL);
    (void)snprintf(file, sizeof(file), "%s/sensor-1.json", path);
    {
        FILE *f = fopen(file, "w");
        assert(f != NULL);
        fprintf(f, "{\"p\":0.9,\"source\":\"sensor-1\"}");
        fclose(f);
    }
    pid = start_service(path, &port);
    lana_vm_init(&vm, NULL);
    (void)snprintf(port_config, sizeof(port_config), "%d", port);
    options.config = port_config;
    assert(lana_adapter_load(&options, &adapter) == LANA_OK);
    assert(lana_adapter_fetch(adapter, &vm, "evidence/sensor-1", &out) == LANA_OK);
    assert(out.type == VAL_MAP);
    {
        Value p;
        assert(lana_map_get(out.as.map, "p", &p) == LANA_OK);
        assert(p.type == VAL_NUMBER && p.as.number == 0.9);
    }
    /* Missing evidence maps to LANA_ERR_NOT_FOUND. */
    assert(lana_adapter_fetch(adapter, &vm, "evidence/missing", &out) == LANA_ERR_NOT_FOUND);
    lana_adapter_close(adapter);
    lana_vm_free(&vm);
    stop_service(pid);
    (void)unlink(file);
    (void)rmdir(path);
    printf("Pass.\n");
}

int main(void) {
    test_http_adapter();
    return 0;
}
#else
int main(void) {
    printf("HTTP adapter not built; skipping.\n");
    return 0;
}
#endif
