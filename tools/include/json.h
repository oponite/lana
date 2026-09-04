#ifndef LANA_JSON_H
#define LANA_JSON_H

#include <stdbool.h>
#include <stddef.h>

/* Minimal JSON DOM for the LSP frontend. Parses a single JSON value into a
 * tree; callers look up object members by key and read string/number leaves. */

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct JsonValue JsonValue;

struct JsonValue {
    JsonType type;
    union {
        bool boolean;
        double number;
        char *string;
        struct {
            JsonValue **items;
            size_t count;
        } array;
        struct {
            char **keys;
            JsonValue **values;
            size_t count;
        } object;
    } as;
};

/* Parse `text` into a JSON value. Returns NULL on malformed input. */
JsonValue *json_parse(const char *text);

/* Free a value returned by json_parse. */
void json_free(JsonValue *value);

/* Return the member `key` of an object, or NULL. */
JsonValue *json_get(const JsonValue *object, const char *key);

/* Return the string value of a JSON_STRING, or NULL. */
const char *json_string(const JsonValue *value);

/* Return the number value of a JSON_NUMBER, or 0.0. */
double json_number(const JsonValue *value);

#endif
