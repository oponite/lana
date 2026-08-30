#include "lana/bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    (void)fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #condition); \
    return 1; } } while (0)

int main(int argc, char **argv) {
    LanaBridgeOptions options = {0};
    char *envelope = NULL;
    FILE *request;
    int result;
    CHECK(argc == 4);
    CHECK(strcmp(lana_bridge_version(), "1.1.1") == 0);
    request = fopen(argv[2], "wb");
    CHECK(request != NULL);
    CHECK(fwrite("{\"native\":true}", 1u, 15u, request) == 15u);
    CHECK(fclose(request) == 0);

    options.struct_size = sizeof(options);
    options.abi_version = LANA_BRIDGE_ABI_VERSION;
    result = lana_bridge_run_labc(argv[1], argv[2], argv[3], &options, &envelope);
    CHECK(result == LANA_BRIDGE_OK);
    CHECK(envelope != NULL);
    CHECK(strstr(envelope, "\"ok\":true") != NULL);
    CHECK(strstr(envelope, "\"native\":true") != NULL);
    lana_bridge_free(envelope); envelope = NULL;

    options.struct_size = sizeof(options) - 1u;
    result = lana_bridge_run_labc(argv[1], argv[2], argv[3], &options, &envelope);
    CHECK(result == LANA_BRIDGE_ERR_ABI);
    CHECK(envelope != NULL && strstr(envelope, "LANA_BRIDGE_ABI") != NULL);
    lana_bridge_free(envelope);
    (void)remove(argv[2]); (void)remove(argv[3]);
    (void)printf("native bridge tests passed\n");
    return 0;
}
