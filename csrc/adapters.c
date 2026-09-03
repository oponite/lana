#include "lana/adapters.h"

#include "lana/data.h"
#include "adapters/adapter_plugin.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    void *handle; /* dlopen handle; NULL for in-core adapters */
    void *ctx;    /* plugin context; NULL for in-core adapters */
    LanaAdapterPluginVtable vtable;
} AdapterInstance;

static LanaError json_fetch(void *ctx, LanaVM *vm, const char *query, Value *out_value) {
    (void)ctx;
    return lana_json_parse(vm, query, out_value);
}

static LanaError csv_fetch(void *ctx, LanaVM *vm, const char *query, Value *out_value) {
    (void)ctx;
    return lana_csv_read(vm, query, out_value);
}

static void noop_close(void *ctx) { (void)ctx; }

static LanaError plugin_load(const char *name, const LanaAdapterOptions *options,
                            AdapterInstance *inst) {
    char path[1024];
    void *handle;
    union { void *p; LanaError (*f)(const LanaAdapterOptions *, void **); } u_load;
    union { void *p; LanaError (*f)(void *, LanaVM *, const char *, Value *); } u_fetch;
    union { void *p; void (*f)(void *); } u_close;
    LanaError error;
    (void)snprintf(path, sizeof(path), "%s/liblana_adapter_%s%s",
                   LANA_ADAPTER_DIR, name, LANA_ADAPTER_SUFFIX);
    handle = dlopen(path, RTLD_NOW);
    if (handle == NULL) return LANA_ERR_IO;
    u_load.p = dlsym(handle, "lana_adapter_plugin_load");
    u_fetch.p = dlsym(handle, "lana_adapter_plugin_fetch");
    u_close.p = dlsym(handle, "lana_adapter_plugin_close");
    if (u_load.f == NULL || u_fetch.f == NULL || u_close.f == NULL) {
        dlclose(handle);
        return LANA_ERR_UNSUPPORTED_OPERATION;
    }
    inst->handle = handle;
    inst->vtable.load = u_load.f;
    inst->vtable.fetch = u_fetch.f;
    inst->vtable.close = u_close.f;
    error = inst->vtable.load(options, &inst->ctx);
    if (error != LANA_OK) {
        dlclose(handle);
        inst->handle = NULL;
    }
    return error;
}

LanaError lana_adapter_load(const LanaAdapterOptions *options, void **out_adapter) {
    AdapterInstance *inst;
    LanaError error;
    if (options == NULL || out_adapter == NULL ||
        options->struct_size < sizeof(*options) || options->schema_version != 1u)
        return LANA_ERR_INVALID_STATE;
    inst = calloc(1u, sizeof(*inst));
    if (inst == NULL) return LANA_ERR_OOM;
    switch (options->kind) {
        case ADAPTER_JSON:
            inst->vtable.fetch = json_fetch;
            inst->vtable.close = noop_close;
            break;
        case ADAPTER_CSV:
            inst->vtable.fetch = csv_fetch;
            inst->vtable.close = noop_close;
            break;
        case ADAPTER_SQLITE:
            error = plugin_load("sqlite", options, inst);
            if (error != LANA_OK) { free(inst); return error; }
            break;
        case ADAPTER_HTTP_JSON:
            error = plugin_load("http_json", options, inst);
            if (error != LANA_OK) { free(inst); return error; }
            break;
        default:
            free(inst);
            return LANA_ERR_UNSUPPORTED_OPERATION;
    }
    *out_adapter = inst;
    return LANA_OK;
}

LanaError lana_adapter_fetch(void *adapter, LanaVM *vm, const char *query,
                             Value *out_value) {
    AdapterInstance *inst;
    if (adapter == NULL || vm == NULL || query == NULL || out_value == NULL)
        return LANA_ERR_INVALID_STATE;
    inst = (AdapterInstance *)adapter;
    return inst->vtable.fetch(inst->ctx, vm, query, out_value);
}

void lana_adapter_close(void *adapter) {
    AdapterInstance *inst;
    if (adapter == NULL) return;
    inst = (AdapterInstance *)adapter;
    inst->vtable.close(inst->ctx);
    if (inst->handle != NULL) dlclose(inst->handle);
    free(inst);
}
