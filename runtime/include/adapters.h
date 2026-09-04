#ifndef LANA_ADAPTERS_H
#define LANA_ADAPTERS_H

#include <stddef.h>
#include <stdint.h>

#include "error.h"
#include "value.h"
#include "vm.h"

typedef enum {
    ADAPTER_JSON = 0,
    ADAPTER_CSV,
    ADAPTER_SQLITE,
    ADAPTER_HTTP_JSON
} LanaAdapterKind;

typedef struct {
    size_t struct_size;
    uint32_t schema_version;
    LanaAdapterKind kind;
    const char *config;
} LanaAdapterOptions;

/* Load an adapter. JSON and CSV are in-core; SQLite and HTTP_JSON are
 * dlopen plugins. Returns LANA_ERR_IO when a plugin library is missing. */
LanaError lana_adapter_load(const LanaAdapterOptions *options, void **out_adapter);

/* Fetch evidence. Values are built in the caller's VM (GC-owned). */
LanaError lana_adapter_fetch(void *adapter, LanaVM *vm, const char *query,
                             Value *out_value);

/* NULL-safe. */
void lana_adapter_close(void *adapter);

#endif
