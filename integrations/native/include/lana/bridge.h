#ifndef LANA_BRIDGE_H
#define LANA_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#define LANA_BRIDGE_ABI_VERSION 1u

typedef enum {
    LANA_BRIDGE_OK = 0,
    LANA_BRIDGE_ERR_ARGUMENT = -1,
    LANA_BRIDGE_ERR_ABI = -2,
    LANA_BRIDGE_ERR_IO = -3,
    LANA_BRIDGE_ERR_PROTOCOL = -4,
    LANA_BRIDGE_ERR_OOM = -5
} LanaBridgeStatus;

typedef struct {
    size_t struct_size;
    uint32_t abi_version;
    uint64_t seed;
    uint64_t instruction_limit;
    size_t memory_limit_bytes;
    size_t workers;
    size_t max_tasks;
} LanaBridgeOptions;

/*
 * Returns zero on success, a positive LanaError on VM failure, or a negative
 * LanaBridgeStatus on bridge failure. envelope_json is always bridge-owned and
 * must be released with lana_bridge_free when non-NULL.
 */
int lana_bridge_run_labc(const char *labc_path, const char *request_path,
                         const char *response_path,
                         const LanaBridgeOptions *options,
                         char **envelope_json);

void lana_bridge_free(char *value);
const char *lana_bridge_version(void);

#endif
