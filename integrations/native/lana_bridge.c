#include "lana/bridge.h"

#include "lana/bytecode.h"
#include "lana/data.h"
#include "lana/error.h"
#include "lana/vm.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef LANA_VERSION
#define LANA_VERSION "unknown"
#endif

#define LANA_BRIDGE_RESPONSE_LIMIT (64u * 1024u * 1024u)

static char *json_escape(const char *text) {
    static const char hex[] = "0123456789abcdef";
    size_t length = 0u, index, output_index = 0u;
    char *output;
    if (text == NULL) text = "";
    for (index = 0u; text[index] != '\0'; ++index) {
        unsigned char byte = (unsigned char)text[index];
        size_t addition = byte == '"' || byte == '\\' ? 2u : byte < 0x20u ? 6u : 1u;
        if (length > SIZE_MAX - addition) return NULL;
        length += addition;
    }
    output = malloc(length + 1u);
    if (output == NULL) return NULL;
    for (index = 0u; text[index] != '\0'; ++index) {
        unsigned char byte = (unsigned char)text[index];
        if (byte == '"' || byte == '\\') {
            output[output_index++] = '\\'; output[output_index++] = (char)byte;
        } else if (byte < 0x20u) {
            output[output_index++] = '\\'; output[output_index++] = 'u';
            output[output_index++] = '0'; output[output_index++] = '0';
            output[output_index++] = hex[byte >> 4u]; output[output_index++] = hex[byte & 15u];
        } else output[output_index++] = (char)byte;
    }
    output[output_index] = '\0'; return output;
}

static char *error_envelope(const char *phase, const char *code,
                            const char *message, int status) {
    char *escaped_phase = json_escape(phase);
    char *escaped_code = json_escape(code);
    char *escaped_message = json_escape(message);
    char *output;
    size_t needed;
    if (escaped_phase == NULL || escaped_code == NULL || escaped_message == NULL) {
        free(escaped_phase); free(escaped_code); free(escaped_message); return NULL;
    }
    needed = strlen(escaped_phase) + strlen(escaped_code) + strlen(escaped_message) + 256u;
    output = malloc(needed);
    if (output != NULL)
        (void)snprintf(output, needed,
            "{\"schema\":1,\"ok\":false,\"phase\":\"%s\","
            "\"error\":{\"code\":\"%s\",\"message\":\"%s\"},"
            "\"exit_code\":%d,\"stdout\":\"\",\"stderr\":\"\","
            "\"execution\":{\"engine\":\"native\",\"lana_version\":\"%s\"}}",
            escaped_phase, escaped_code, escaped_message, status, LANA_VERSION);
    free(escaped_phase); free(escaped_code); free(escaped_message); return output;
}

static char *success_envelope(const char *response) {
    size_t needed = strlen(response) + 256u;
    char *output = malloc(needed);
    if (output != NULL)
        (void)snprintf(output, needed,
            "{\"schema\":1,\"ok\":true,\"result\":%s,"
            "\"stdout\":\"\",\"stderr\":\"\","
            "\"execution\":{\"engine\":\"native\",\"lana_version\":\"%s\"}}",
            response, LANA_VERSION);
    return output;
}

static char *read_response(const char *path, int *status) {
    FILE *file = fopen(path, "rb");
    long length;
    char *text;
    if (file == NULL) { *status = LANA_BRIDGE_ERR_IO; return NULL; }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        (unsigned long)length > LANA_BRIDGE_RESPONSE_LIMIT ||
        fseek(file, 0, SEEK_SET) != 0) {
        (void)fclose(file); *status = LANA_BRIDGE_ERR_PROTOCOL; return NULL;
    }
    text = malloc((size_t)length + 1u);
    if (text == NULL) { (void)fclose(file); *status = LANA_BRIDGE_ERR_OOM; return NULL; }
    if (fread(text, 1u, (size_t)length, file) != (size_t)length || fclose(file) != 0) {
        free(text); *status = LANA_BRIDGE_ERR_IO; return NULL;
    }
    text[length] = '\0'; return text;
}

static int fail(char **envelope_json, int status, const char *phase,
                const char *code, const char *message) {
    *envelope_json = error_envelope(phase, code, message, status);
    if (*envelope_json == NULL) return LANA_BRIDGE_ERR_OOM;
    return status;
}

int lana_bridge_run_labc(const char *labc_path, const char *request_path,
                         const char *response_path,
                         const LanaBridgeOptions *options,
                         char **envelope_json) {
    LanaChunk chunk;
    LanaVM vm;
    LanaErrorInfo error = {0};
    LanaError result;
    const char *arguments[2];
    char *response;
    Value parsed;
    int response_status = LANA_BRIDGE_OK;
    if (envelope_json == NULL) return LANA_BRIDGE_ERR_ARGUMENT;
    *envelope_json = NULL;
    if (labc_path == NULL || request_path == NULL || response_path == NULL)
        return fail(envelope_json, LANA_BRIDGE_ERR_ARGUMENT, "input",
                    "LANA_BRIDGE_ARGUMENT", "paths must not be null");
    if (options != NULL && (options->struct_size < sizeof(*options) ||
                            options->abi_version != LANA_BRIDGE_ABI_VERSION))
        return fail(envelope_json, LANA_BRIDGE_ERR_ABI, "compatibility",
                    "LANA_BRIDGE_ABI", "unsupported bridge options ABI");

    (void)remove(response_path);
    result = lana_chunk_read_file(&chunk, labc_path, &error);
    if (result != LANA_OK)
        return fail(envelope_json, (int)result, "load", lana_error_name(result),
                    error.message[0] == '\0' ? "failed to load LABC" : error.message);

    lana_vm_init(&vm, &chunk);
    if (options != NULL) {
        if (options->seed != 0u) lana_vm_seed(&vm, options->seed);
        if (options->instruction_limit != 0u) vm.instruction_limit = options->instruction_limit;
        if (options->memory_limit_bytes != 0u) vm.memory_limit = options->memory_limit_bytes;
        if ((options->workers != 0u &&
             lana_vm_set_worker_count(&vm, options->workers) != LANA_OK) ||
            (options->max_tasks != 0u &&
             lana_vm_set_task_limit(&vm, options->max_tasks) != LANA_OK)) {
            lana_vm_free(&vm); lana_chunk_free(&chunk);
            return fail(envelope_json, LANA_ERR_TASK, "run", "LANA_ERR_TASK",
                        "invalid scheduler options");
        }
    }
    arguments[0] = request_path; arguments[1] = response_path;
    lana_vm_set_program_args(&vm, 2, arguments);
    result = lana_vm_run(&vm);
    if (result != LANA_OK) {
        char message[LANA_ERROR_MESSAGE_CAPACITY];
        (void)snprintf(message, sizeof(message), "%s",
                       vm.error.message[0] == '\0' ? lana_error_name(result) : vm.error.message);
        lana_vm_free(&vm); lana_chunk_free(&chunk);
        return fail(envelope_json, (int)result, "run", lana_error_name(result), message);
    }
    response = read_response(response_path, &response_status);
    if (response == NULL) {
        lana_vm_free(&vm); lana_chunk_free(&chunk);
        return fail(envelope_json, response_status, "protocol",
                    response_status == LANA_BRIDGE_ERR_OOM ? "LANA_BRIDGE_OOM" :
                    response_status == LANA_BRIDGE_ERR_IO ? "LANA_RESPONSE_MISSING" :
                    "LANA_RESPONSE_LIMIT",
                    response_status == LANA_BRIDGE_ERR_IO ?
                    "program did not write a readable response file" :
                    "response could not be loaded");
    }
    result = lana_json_parse(&vm, response, &parsed);
    if (result != LANA_OK) {
        free(response); lana_vm_free(&vm); lana_chunk_free(&chunk);
        return fail(envelope_json, LANA_BRIDGE_ERR_PROTOCOL, "protocol",
                    "LANA_RESPONSE_INVALID", "program wrote invalid response JSON");
    }
    *envelope_json = success_envelope(response);
    free(response); lana_vm_free(&vm); lana_chunk_free(&chunk);
    if (*envelope_json == NULL) return LANA_BRIDGE_ERR_OOM;
    return LANA_BRIDGE_OK;
}

void lana_bridge_free(char *value) { free(value); }

const char *lana_bridge_version(void) { return LANA_VERSION; }
