#include "codec.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static LanaError append_bytes(LanaBuffer *buffer, const void *data, size_t length) {
    size_t required, capacity;
    unsigned char *grown;
    if (buffer == NULL || (data == NULL && length != 0u)) return LANA_ERR_INVALID_STATE;
    if (length > SIZE_MAX - buffer->length) return LANA_ERR_LIMIT;
    required = buffer->length + length;
    if (required > buffer->capacity) {
        capacity = buffer->capacity == 0u ? 128u : buffer->capacity;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2u) { capacity = required; break; }
            capacity *= 2u;
        }
        grown = realloc(buffer->data, capacity);
        if (grown == NULL) return LANA_ERR_OOM;
        buffer->data = grown;
        buffer->capacity = capacity;
    }
    if (length != 0u) memcpy(buffer->data + buffer->length, data, length);
    buffer->length = required;
    return LANA_OK;
}

static LanaError append_text(LanaBuffer *buffer, const char *text) {
    return append_bytes(buffer, text, strlen(text));
}

static bool valid_utf8(const unsigned char *text, size_t length) {
    size_t index = 0u;
    while (index < length) {
        unsigned char first = text[index++];
        size_t extra;
        uint32_t codepoint;
        if (first < 0x80u) continue;
        if (first >= 0xc2u && first <= 0xdfu) { extra = 1u; codepoint = first & 0x1fu; }
        else if (first >= 0xe0u && first <= 0xefu) { extra = 2u; codepoint = first & 0x0fu; }
        else if (first >= 0xf0u && first <= 0xf4u) { extra = 3u; codepoint = first & 0x07u; }
        else return false;
        if (extra > length - index) return false;
        while (extra-- != 0u) {
            unsigned char continuation = text[index++];
            if ((continuation & 0xc0u) != 0x80u) return false;
            codepoint = (codepoint << 6u) | (continuation & 0x3fu);
        }
        if (codepoint > 0x10ffffu || (codepoint >= 0xd800u && codepoint <= 0xdfffu) ||
            (codepoint <= 0x7fu) || (codepoint <= 0x7ffu && first >= 0xe0u) ||
            (codepoint <= 0xffffu && first >= 0xf0u)) return false;
    }
    return true;
}

static LanaError append_string(LanaBuffer *buffer, const char *string) {
    const unsigned char *text = (const unsigned char *)(string == NULL ? "" : string);
    size_t length = strlen((const char *)text), index;
    char escape[7];
    if (!valid_utf8(text, length)) return LANA_ERR_SCHEMA;
    if (append_bytes(buffer, "\"", 1u) != LANA_OK) return LANA_ERR_OOM;
    for (index = 0u; index < length; ++index) {
        unsigned char character = text[index];
        const char *replacement = NULL;
        switch (character) {
            case '"': replacement = "\\\""; break;
            case '\\': replacement = "\\\\"; break;
            case '\b': replacement = "\\b"; break;
            case '\f': replacement = "\\f"; break;
            case '\n': replacement = "\\n"; break;
            case '\r': replacement = "\\r"; break;
            case '\t': replacement = "\\t"; break;
            default: break;
        }
        if (character < 0x20u && replacement == NULL) {
            (void)snprintf(escape, sizeof(escape), "\\u%04x", character);
            replacement = escape;
        }
        if (replacement != NULL) {
            if (append_text(buffer, replacement) != LANA_OK) return LANA_ERR_OOM;
        } else if (append_bytes(buffer, &character, 1u) != LANA_OK) {
            return LANA_ERR_OOM;
        }
    }
    return append_bytes(buffer, "\"", 1u);
}

static int map_entry_compare(const void *left, const void *right) {
    const LanaMapEntry *const *a = left;
    const LanaMapEntry *const *b = right;
    return strcmp((*a)->key, (*b)->key);
}

static LanaError encode_value(LanaBuffer *buffer, Value value) {
    size_t index;
    char number[64];
    LanaError error;
    if (buffer == NULL) return LANA_ERR_INVALID_STATE;
    switch (value.type) {
        case VAL_NULL: return append_text(buffer, "null");
        case VAL_BOOL: return append_text(buffer, value.as.boolean ? "true" : "false");
        case VAL_NUMBER:
            if (!isfinite(value.as.number)) return LANA_ERR_UNSUPPORTED_VALUE;
            (void)snprintf(number, sizeof(number), "%.17g", value.as.number);
            return append_text(buffer, number);
        case VAL_STRING: return append_string(buffer, value.as.string);
        case VAL_ARRAY:
            if (value.as.array == NULL) return LANA_ERR_UNSUPPORTED_VALUE;
            if ((error = append_bytes(buffer, "[", 1u)) != LANA_OK) return error;
            for (index = 0u; index < value.as.array->count; ++index) {
                if (index != 0u && (error = append_bytes(buffer, ",", 1u)) != LANA_OK) return error;
                if ((error = encode_value(buffer, value.as.array->items[index])) != LANA_OK) return error;
            }
            return append_bytes(buffer, "]", 1u);
        case VAL_MAP: {
            LanaMapEntry **entries;
            if (value.as.map == NULL) return LANA_ERR_UNSUPPORTED_VALUE;
            entries = malloc(value.as.map->count * sizeof(*entries));
            if (value.as.map->count != 0u && entries == NULL) return LANA_ERR_OOM;
            for (index = 0u; index < value.as.map->count; ++index) {
                if (value.as.map->entries[index].key == NULL || value.as.map->entries[index].value == NULL) {
                    free(entries); return LANA_ERR_SCHEMA;
                }
                entries[index] = &value.as.map->entries[index];
            }
            qsort(entries, value.as.map->count, sizeof(*entries), map_entry_compare);
            error = append_bytes(buffer, "{", 1u);
            for (index = 0u; error == LANA_OK && index < value.as.map->count; ++index) {
                if (index != 0u) error = append_bytes(buffer, ",", 1u);
                if (error == LANA_OK) error = append_string(buffer, entries[index]->key);
                if (error == LANA_OK) error = append_bytes(buffer, ":", 1u);
                if (error == LANA_OK) error = encode_value(buffer, *entries[index]->value);
            }
            if (error == LANA_OK) error = append_bytes(buffer, "}", 1u);
            free(entries);
            return error;
        }
        default: return LANA_ERR_UNSUPPORTED_VALUE;
    }
}

LanaError lana_codec_encode_value(LanaBuffer *buffer, Value value) {
    return encode_value(buffer, value);
}

typedef struct { const unsigned char *data; size_t length; size_t offset; } Parser;

static void skip_space(Parser *parser) {
    while (parser->offset < parser->length && isspace(parser->data[parser->offset])) ++parser->offset;
}

static bool take(Parser *parser, unsigned char expected) {
    if (parser->offset < parser->length && parser->data[parser->offset] == expected) {
        ++parser->offset; return true;
    }
    return false;
}

static LanaError parse_value(Parser *parser, Value *out);

static LanaError parse_string(Parser *parser, char **out) {
    LanaBuffer buffer = {0};
    if (!take(parser, '"')) return LANA_ERR_PARSE;
    while (parser->offset < parser->length) {
        unsigned char character = parser->data[parser->offset++];
        if (character == '"') {
            if (append_bytes(&buffer, "", 1u) != LANA_OK) { free(buffer.data); return LANA_ERR_OOM; }
            *out = (char *)buffer.data;
            if (!valid_utf8(buffer.data, buffer.length - 1u)) { free(buffer.data); return LANA_ERR_SCHEMA; }
            return LANA_OK;
        }
        if (character < 0x20u) { free(buffer.data); return LANA_ERR_PARSE; }
        if (character == '\\') {
            if (parser->offset >= parser->length) { free(buffer.data); return LANA_ERR_PARSE; }
            character = parser->data[parser->offset++];
            switch (character) {
                case '"': case '\\': case '/': break;
                case 'b': character = '\b'; break; case 'f': character = '\f'; break;
                case 'n': character = '\n'; break; case 'r': character = '\r'; break;
                case 't': character = '\t'; break;
                default: free(buffer.data); return LANA_ERR_PARSE;
            }
        }
        if (append_bytes(&buffer, &character, 1u) != LANA_OK) { free(buffer.data); return LANA_ERR_OOM; }
    }
    free(buffer.data);
    return LANA_ERR_PARSE;
}

static LanaError parse_value(Parser *parser, Value *out) {
    char *string, *end;
    double number;
    if (parser == NULL || out == NULL) return LANA_ERR_INVALID_STATE;
    skip_space(parser);
    if (parser->offset >= parser->length) return LANA_ERR_PARSE;
    if (parser->data[parser->offset] == '"') {
        LanaError error = parse_string(parser, &string);
        if (error != LANA_OK) return error;
        *out = lana_value_string(string); return LANA_OK;
    }
    if (parser->data[parser->offset] == 'n' && parser->length - parser->offset >= 4u && memcmp(parser->data + parser->offset, "null", 4u) == 0) { parser->offset += 4u; *out = lana_value_null(); return LANA_OK; }
    if (parser->data[parser->offset] == 't' && parser->length - parser->offset >= 4u && memcmp(parser->data + parser->offset, "true", 4u) == 0) { parser->offset += 4u; *out = lana_value_bool(true); return LANA_OK; }
    if (parser->data[parser->offset] == 'f' && parser->length - parser->offset >= 5u && memcmp(parser->data + parser->offset, "false", 5u) == 0) { parser->offset += 5u; *out = lana_value_bool(false); return LANA_OK; }
    if (parser->data[parser->offset] == '[') {
        LanaArray *array = calloc(1u, sizeof(*array));
        if (array == NULL) return LANA_ERR_OOM;
        ++parser->offset; skip_space(parser);
        if (take(parser, ']')) { *out = lana_value_array(array); return LANA_OK; }
        for (;;) {
            Value item = lana_value_null(); LanaError error = parse_value(parser, &item);
            if (error != LANA_OK) { lana_value_free(lana_value_array(array)); return error; }
            if (array->count == array->capacity) {
                size_t capacity = array->capacity == 0u ? 4u : array->capacity * 2u;
                Value *items = realloc(array->items, capacity * sizeof(*items));
                if (items == NULL) {
                    lana_value_free(item); lana_value_free(lana_value_array(array)); return LANA_ERR_OOM;
                }
                array->items = items; array->capacity = capacity;
            }
            array->items[array->count++] = item; skip_space(parser);
            if (take(parser, ']')) { *out = lana_value_array(array); return LANA_OK; }
            if (!take(parser, ',')) { lana_value_free(lana_value_array(array)); return LANA_ERR_PARSE; }
        }
    }
    if (parser->data[parser->offset] == '{') {
        LanaMap *map = calloc(1u, sizeof(*map));
        if (map == NULL) return LANA_ERR_OOM;
        ++parser->offset; skip_space(parser);
        if (take(parser, '}')) { *out = lana_value_map(map); return LANA_OK; }
        for (;;) {
            Value key = lana_value_null(), value = lana_value_null();
            LanaError error = parse_value(parser, &key);
            if (error != LANA_OK || key.type != VAL_STRING) {
                lana_value_free(key); lana_value_free(lana_value_map(map));
                return error == LANA_OK ? LANA_ERR_SCHEMA : error;
            }
            skip_space(parser);
            if (!take(parser, ':')) {
                lana_value_free(key); lana_value_free(lana_value_map(map)); return LANA_ERR_PARSE;
            }
            error = parse_value(parser, &value);
            if (error != LANA_OK) {
                lana_value_free(key); lana_value_free(lana_value_map(map)); return error;
            }
            if (map->count == map->capacity) {
                size_t capacity = map->capacity == 0u ? 4u : map->capacity * 2u;
                LanaMapEntry *entries = realloc(map->entries, capacity * sizeof(*entries));
                if (entries == NULL) {
                    lana_value_free(key); lana_value_free(value); lana_value_free(lana_value_map(map));
                    return LANA_ERR_OOM;
                }
                map->entries = entries; map->capacity = capacity;
            }
            for (size_t index = 0u; index < map->count; ++index)
                if (strcmp(map->entries[index].key, key.as.string) == 0) {
                    lana_value_free(key); lana_value_free(value); lana_value_free(lana_value_map(map));
                    return LANA_ERR_SCHEMA;
                }
            Value *entry_value = malloc(sizeof(*entry_value));
            if (entry_value == NULL) {
                lana_value_free(key); lana_value_free(value); lana_value_free(lana_value_map(map));
                return LANA_ERR_OOM;
            }
            map->entries[map->count].key = key.as.string;
            map->entries[map->count].value = entry_value;
            *map->entries[map->count++].value = value; skip_space(parser);
            if (take(parser, '}')) { *out = lana_value_map(map); return LANA_OK; }
            if (!take(parser, ',')) { lana_value_free(lana_value_map(map)); return LANA_ERR_PARSE; }
        }
    }
    number = strtod((const char *)parser->data + parser->offset, &end);
    if (end == (const char *)parser->data + parser->offset || !isfinite(number)) return LANA_ERR_PARSE;
    parser->offset = (size_t)(end - (const char *)parser->data);
    *out = lana_value_number(number); return LANA_OK;
}

LanaError lana_codec_decode_value(LanaBuffer *buffer, size_t *offset, Value *out_value) {
    Parser parser;
    LanaError error;
    if (buffer == NULL || offset == NULL || out_value == NULL || *offset > buffer->length) return LANA_ERR_INVALID_STATE;
    parser = (Parser){buffer->data, buffer->length, *offset};
    error = parse_value(&parser, out_value);
    if (error == LANA_OK) *offset = parser.offset;
    return error;
}

LanaError lana_codec_decode_document(LanaBuffer *buffer, Value *out_value) {
    size_t offset = 0u;
    LanaError error = lana_codec_decode_value(buffer, &offset, out_value);
    if (error != LANA_OK) return error;
    while (offset < buffer->length && isspace(buffer->data[offset])) ++offset;
    return offset == buffer->length ? LANA_OK : LANA_ERR_PARSE;
}
