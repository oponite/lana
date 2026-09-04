#include "json.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *text;
    size_t length;
    size_t position;
} Parser;

static void skip_whitespace(Parser *parser) {
    while (parser->position < parser->length) {
        char c = parser->text[parser->position];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++parser->position;
        else break;
    }
}

static JsonValue *parse_value(Parser *parser);

static JsonValue *new_value(JsonType type) {
    JsonValue *value = calloc(1u, sizeof(JsonValue));
    if (value != NULL) value->type = type;
    return value;
}

static void append_utf8(char **out, size_t *length, size_t *capacity, unsigned int codepoint) {
    char bytes[4];
    size_t count;
    if (codepoint < 0x80u) {
        bytes[0] = (char)codepoint; count = 1u;
    } else if (codepoint < 0x800u) {
        bytes[0] = (char)(0xC0u | (codepoint >> 6));
        bytes[1] = (char)(0x80u | (codepoint & 0x3Fu));
        count = 2u;
    } else if (codepoint < 0x10000u) {
        bytes[0] = (char)(0xE0u | (codepoint >> 12));
        bytes[1] = (char)(0x80u | ((codepoint >> 6) & 0x3Fu));
        bytes[2] = (char)(0x80u | (codepoint & 0x3Fu));
        count = 3u;
    } else {
        bytes[0] = (char)(0xF0u | (codepoint >> 18));
        bytes[1] = (char)(0x80u | ((codepoint >> 12) & 0x3Fu));
        bytes[2] = (char)(0x80u | ((codepoint >> 6) & 0x3Fu));
        bytes[3] = (char)(0x80u | (codepoint & 0x3Fu));
        count = 4u;
    }
    while (*length + count + 1u > *capacity) {
        size_t next = *capacity == 0u ? 16u : *capacity * 2u;
        char *grown = realloc(*out, next);
        if (grown == NULL) return;
        *out = grown; *capacity = next;
    }
    memcpy(*out + *length, bytes, count);
    *length += count;
    (*out)[*length] = '\0';
}

static unsigned int hex_value(char c) {
    if (c >= '0' && c <= '9') return (unsigned int)(c - '0');
    if (c >= 'a' && c <= 'f') return (unsigned int)(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return (unsigned int)(c - 'A' + 10);
    return 0u;
}

static char *parse_string(Parser *parser) {
    char *out = NULL;
    size_t length = 0u, capacity = 0u;
    if (parser->position >= parser->length || parser->text[parser->position] != '"') return NULL;
    ++parser->position;
    while (parser->position < parser->length) {
        char c = parser->text[parser->position++];
        if (c == '"') {
            if (out == NULL) { out = malloc(1u); if (out != NULL) out[0] = '\0'; }
            return out;
        }
        if (c == '\\') {
            if (parser->position >= parser->length) break;
            char escape = parser->text[parser->position++];
            char decoded;
            switch (escape) {
                case '"': decoded = '"'; break;
                case '\\': decoded = '\\'; break;
                case '/': decoded = '/'; break;
                case 'b': decoded = '\b'; break;
                case 'f': decoded = '\f'; break;
                case 'n': decoded = '\n'; break;
                case 'r': decoded = '\r'; break;
                case 't': decoded = '\t'; break;
                case 'u': {
                    unsigned int codepoint;
                    if (parser->position + 4u > parser->length) { free(out); return NULL; }
                    codepoint = (hex_value(parser->text[parser->position]) << 12) |
                                (hex_value(parser->text[parser->position + 1u]) << 8) |
                                (hex_value(parser->text[parser->position + 2u]) << 4) |
                                hex_value(parser->text[parser->position + 3u]);
                    parser->position += 4u;
                    append_utf8(&out, &length, &capacity, codepoint);
                    continue;
                }
                default: free(out); return NULL;
            }
            append_utf8(&out, &length, &capacity, (unsigned int)(unsigned char)decoded);
            continue;
        }
        append_utf8(&out, &length, &capacity, (unsigned int)(unsigned char)c);
    }
    free(out);
    return NULL;
}

static JsonValue *parse_number(Parser *parser) {
    char *end;
    double number = strtod(parser->text + parser->position, &end);
    if (end == parser->text + parser->position) return NULL;
    parser->position = (size_t)(end - parser->text);
    JsonValue *value = new_value(JSON_NUMBER);
    if (value != NULL) value->as.number = number;
    return value;
}

static JsonValue *parse_literal(Parser *parser, const char *literal, JsonType type, bool boolean) {
    size_t length = strlen(literal);
    if (parser->position + length > parser->length) return NULL;
    if (strncmp(parser->text + parser->position, literal, length) != 0) return NULL;
    parser->position += length;
    JsonValue *value = new_value(type);
    if (value != NULL && type == JSON_BOOL) value->as.boolean = boolean;
    return value;
}

static JsonValue *parse_array(Parser *parser) {
    JsonValue *value = new_value(JSON_ARRAY);
    if (value == NULL) return NULL;
    ++parser->position; /* consume '[' */
    skip_whitespace(parser);
    if (parser->position < parser->length && parser->text[parser->position] == ']') {
        ++parser->position;
        return value;
    }
    for (;;) {
        JsonValue *item = parse_value(parser);
        if (item == NULL) { json_free(value); return NULL; }
        JsonValue **grown = realloc(value->as.array.items,
                                    (value->as.array.count + 1u) * sizeof(JsonValue *));
        if (grown == NULL) { json_free(item); json_free(value); return NULL; }
        value->as.array.items = grown;
        value->as.array.items[value->as.array.count++] = item;
        skip_whitespace(parser);
        if (parser->position < parser->length && parser->text[parser->position] == ',') {
            ++parser->position; skip_whitespace(parser); continue;
        }
        if (parser->position < parser->length && parser->text[parser->position] == ']') {
            ++parser->position; return value;
        }
        json_free(value); return NULL;
    }
}

static JsonValue *parse_object(Parser *parser) {
    JsonValue *value = new_value(JSON_OBJECT);
    if (value == NULL) return NULL;
    ++parser->position; /* consume '{' */
    skip_whitespace(parser);
    if (parser->position < parser->length && parser->text[parser->position] == '}') {
        ++parser->position;
        return value;
    }
    for (;;) {
        char *key;
        JsonValue *member;
        char **grown_keys;
        JsonValue **grown_values;
        skip_whitespace(parser);
        key = parse_string(parser);
        if (key == NULL) { json_free(value); return NULL; }
        skip_whitespace(parser);
        if (parser->position >= parser->length || parser->text[parser->position] != ':') {
            free(key); json_free(value); return NULL;
        }
        ++parser->position;
        skip_whitespace(parser);
        member = parse_value(parser);
        if (member == NULL) { free(key); json_free(value); return NULL; }
        grown_keys = realloc(value->as.object.keys,
                             (value->as.object.count + 1u) * sizeof(char *));
        grown_values = realloc(value->as.object.values,
                               (value->as.object.count + 1u) * sizeof(JsonValue *));
        if (grown_keys == NULL || grown_values == NULL) {
            free(key); json_free(member); json_free(value); return NULL;
        }
        value->as.object.keys = grown_keys;
        value->as.object.values = grown_values;
        value->as.object.keys[value->as.object.count] = key;
        value->as.object.values[value->as.object.count] = member;
        ++value->as.object.count;
        skip_whitespace(parser);
        if (parser->position < parser->length && parser->text[parser->position] == ',') {
            ++parser->position; continue;
        }
        if (parser->position < parser->length && parser->text[parser->position] == '}') {
            ++parser->position; return value;
        }
        json_free(value); return NULL;
    }
}

static JsonValue *parse_value(Parser *parser) {
    skip_whitespace(parser);
    if (parser->position >= parser->length) return NULL;
    switch (parser->text[parser->position]) {
        case '{': return parse_object(parser);
        case '[': return parse_array(parser);
        case '"': {
            char *string = parse_string(parser);
            JsonValue *value;
            if (string == NULL) return NULL;
            value = new_value(JSON_STRING);
            if (value == NULL) { free(string); return NULL; }
            value->as.string = string;
            return value;
        }
        case 't': return parse_literal(parser, "true", JSON_BOOL, true);
        case 'f': return parse_literal(parser, "false", JSON_BOOL, false);
        case 'n': return parse_literal(parser, "null", JSON_NULL, false);
        default:
            if (parser->text[parser->position] == '-' ||
                (parser->text[parser->position] >= '0' && parser->text[parser->position] <= '9'))
                return parse_number(parser);
            return NULL;
    }
}

JsonValue *json_parse(const char *text) {
    Parser parser;
    JsonValue *value;
    if (text == NULL) return NULL;
    parser.text = text;
    parser.length = strlen(text);
    parser.position = 0u;
    value = parse_value(&parser);
    if (value == NULL) return NULL;
    skip_whitespace(&parser);
    if (parser.position != parser.length) { json_free(value); return NULL; }
    return value;
}

void json_free(JsonValue *value) {
    size_t index;
    if (value == NULL) return;
    switch (value->type) {
        case JSON_STRING: free(value->as.string); break;
        case JSON_ARRAY:
            for (index = 0u; index < value->as.array.count; ++index)
                json_free(value->as.array.items[index]);
            free(value->as.array.items);
            break;
        case JSON_OBJECT:
            for (index = 0u; index < value->as.object.count; ++index) {
                free(value->as.object.keys[index]);
                json_free(value->as.object.values[index]);
            }
            free(value->as.object.keys);
            free(value->as.object.values);
            break;
        default: break;
    }
    free(value);
}

JsonValue *json_get(const JsonValue *object, const char *key) {
    size_t index;
    if (object == NULL || object->type != JSON_OBJECT) return NULL;
    for (index = 0u; index < object->as.object.count; ++index) {
        if (strcmp(object->as.object.keys[index], key) == 0)
            return object->as.object.values[index];
    }
    return NULL;
}

const char *json_string(const JsonValue *value) {
    if (value == NULL || value->type != JSON_STRING) return NULL;
    return value->as.string;
}

double json_number(const JsonValue *value) {
    if (value == NULL || value->type != JSON_NUMBER) return 0.0;
    return value->as.number;
}
