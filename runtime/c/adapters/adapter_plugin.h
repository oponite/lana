#ifndef LANA_ADAPTER_PLUGIN_H
#define LANA_ADAPTER_PLUGIN_H

#include "adapters.h"
#include "vm.h"

/* ABI for dlopen adapter plugins. Each plugin exports
 * lana_adapter_plugin_load / lana_adapter_plugin_fetch / lana_adapter_plugin_close.
 * The plugin context is opaque; fetch builds values in the caller's VM. */
typedef struct {
    LanaError (*load)(const LanaAdapterOptions *options, void **out_ctx);
    LanaError (*fetch)(void *ctx, LanaVM *vm, const char *query, Value *out_value);
    void (*close)(void *ctx);
} LanaAdapterPluginVtable;

#endif
