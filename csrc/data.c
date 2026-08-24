#include "lana/data.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LANA_DATA_DEPTH_LIMIT 128u

typedef struct { char *data; size_t length; size_t capacity; } Buffer;

static bool buffer_reserve(Buffer *buffer, size_t extra) {
    size_t needed = buffer->length + extra + 1u;
    char *grown;
    if (needed <= buffer->capacity) return true;
    if (needed > SIZE_MAX / 2u) return false;
    buffer->capacity = buffer->capacity == 0u ? 128u : buffer->capacity;
    while (buffer->capacity < needed) buffer->capacity *= 2u;
    grown = realloc(buffer->data, buffer->capacity);
    if (grown == NULL) return false;
    buffer->data = grown;
    return true;
}

static bool buffer_add(Buffer *buffer, const char *text, size_t length) {
    if (!buffer_reserve(buffer, length)) return false;
    memcpy(buffer->data + buffer->length, text, length);
    buffer->length += length;
    buffer->data[buffer->length] = '\0';
    return true;
}

static char *vm_string(LanaVM *vm, const char *text, size_t length) {
    char *copy = lana_vm_alloc(vm, length + 1u);
    if (copy == NULL) return NULL;
    memcpy(copy, text, length); copy[length] = '\0';
    return copy;
}

LanaError lana_map_new(LanaVM *vm, size_t capacity, LanaMap **out) {
    LanaMap *map = lana_vm_alloc(vm, sizeof(*map));
    if (map == NULL) return LANA_ERR_OOM;
    map->count = 0u; map->capacity = capacity;
    map->entries = capacity == 0u ? NULL : lana_vm_alloc(vm, capacity * sizeof(*map->entries));
    if (capacity > 0u && map->entries == NULL) return LANA_ERR_OOM;
    *out = map; return LANA_OK;
}

static ptrdiff_t map_find(const LanaMap *map, const char *key) {
    size_t index;
    if (map == NULL || key == NULL) return -1;
    for (index = 0; index < map->count; ++index)
        if (strcmp(map->entries[index].key, key) == 0) return (ptrdiff_t)index;
    return -1;
}

LanaError lana_map_get(const LanaMap *map, const char *key, Value *out) {
    ptrdiff_t index = map_find(map, key);
    if (index < 0) return LANA_ERR_KEY;
    *out = *map->entries[(size_t)index].value;
    return LANA_OK;
}

ptrdiff_t lana_map_has(const LanaMap *map, const char *key) {
    return map_find(map, key);
}

LanaError lana_map_set(LanaVM *vm, LanaMap *map, const char *key, const Value *value,
                   bool reject_existing) {
    ptrdiff_t found;
    Value *slot;
    if (map == NULL || key == NULL || value == NULL || strchr(key, '\0') == NULL) return LANA_ERR_TYPE;
    found = map_find(map, key);
    if (found >= 0) {
        if (reject_existing) return LANA_ERR_KEY;
        lana_vm_write_barrier_value(vm, map->entries[(size_t)found].value,
                                    value);
        *map->entries[(size_t)found].value = *value;
        return LANA_OK;
    }
    if (map->count == map->capacity) {
        size_t capacity = map->capacity == 0u ? 4u : map->capacity * 2u;
        LanaMapEntry *entries = lana_vm_alloc(vm, capacity * sizeof(*entries));
        if (entries == NULL) return LANA_ERR_OOM;
        if (map->count > 0u) memcpy(entries, map->entries, map->count * sizeof(*entries));
        map->entries = entries; map->capacity = capacity;
    }
    map->entries[map->count].key = vm_string(vm, key, strlen(key));
    slot = lana_vm_alloc(vm, sizeof(*slot));
    if (map->entries[map->count].key == NULL || slot == NULL) return LANA_ERR_OOM;
    (void)lana_gc_write_barrier(&vm->gc, map, slot);
    lana_vm_write_barrier_value(vm, slot, value);
    *slot = *value; map->entries[map->count].value = slot; ++map->count;
    return LANA_OK;
}

typedef struct { LanaVM *vm; const unsigned char *cursor; const unsigned char *end; } JsonParser;

static bool utf8_valid(const unsigned char *text, size_t length) {
    size_t i = 0u;
    while (i < length) {
        unsigned char c = text[i++]; size_t extra; unsigned code;
        if (c < 0x80u) { if (c == 0u) return false; continue; }
        if (c >= 0xc2u && c <= 0xdfu) { extra = 1u; code = c & 0x1fu; }
        else if (c >= 0xe0u && c <= 0xefu) { extra = 2u; code = c & 0x0fu; }
        else if (c >= 0xf0u && c <= 0xf4u) { extra = 3u; code = c & 0x07u; }
        else return false;
        if (i + extra > length) return false;
        while (extra-- > 0u) { if ((text[i] & 0xc0u) != 0x80u) return false; code = (code << 6u) | (text[i++] & 0x3fu); }
        if (code > 0x10ffffu || (code >= 0xd800u && code <= 0xdfffu) ||
            (code < 0x800u && c >= 0xe0u) || (code < 0x10000u && c >= 0xf0u)) return false;
    }
    return true;
}

static void json_space(JsonParser *parser) {
    while (parser->cursor < parser->end && isspace(*parser->cursor)) ++parser->cursor;
}

static int hex_value(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool add_utf8(Buffer *buffer, unsigned code) {
    char bytes[4]; size_t count;
    if (code == 0u || code > 0x10ffffu || (code >= 0xd800u && code <= 0xdfffu)) return false;
    if (code < 0x80u) { bytes[0] = (char)code; count = 1u; }
    else if (code < 0x800u) { bytes[0] = (char)(0xc0u | (code >> 6u)); bytes[1] = (char)(0x80u | (code & 63u)); count = 2u; }
    else if (code < 0x10000u) { bytes[0] = (char)(0xe0u | (code >> 12u)); bytes[1] = (char)(0x80u | ((code >> 6u) & 63u)); bytes[2] = (char)(0x80u | (code & 63u)); count = 3u; }
    else { bytes[0] = (char)(0xf0u | (code >> 18u)); bytes[1] = (char)(0x80u | ((code >> 12u) & 63u)); bytes[2] = (char)(0x80u | ((code >> 6u) & 63u)); bytes[3] = (char)(0x80u | (code & 63u)); count = 4u; }
    return buffer_add(buffer, bytes, count);
}

static LanaError json_string(JsonParser *parser, char **out) {
    Buffer buffer = {0};
    if (parser->cursor >= parser->end || *parser->cursor++ != '"') return LANA_ERR_PARSE;
    while (parser->cursor < parser->end && *parser->cursor != '"') {
        unsigned char c = *parser->cursor++;
        if (c < 0x20u) { free(buffer.data); return LANA_ERR_PARSE; }
        if (c != '\\') { if (!buffer_add(&buffer, (const char *)&c, 1u)) { free(buffer.data); return LANA_ERR_OOM; } continue; }
        if (parser->cursor >= parser->end) { free(buffer.data); return LANA_ERR_PARSE; }
        c = *parser->cursor++;
        if (strchr("\"\\/", c) != NULL) { if (!buffer_add(&buffer, (const char *)&c, 1u)) { free(buffer.data); return LANA_ERR_OOM; } }
        else if (strchr("bfnrt", c) != NULL) { const char *from = "bfnrt", *to = "\b\f\n\r\t"; char decoded = to[strchr(from, c) - from]; if (!buffer_add(&buffer, &decoded, 1u)) { free(buffer.data); return LANA_ERR_OOM; } }
        else if (c == 'u') {
            unsigned code = 0u; size_t i;
            if ((size_t)(parser->end - parser->cursor) < 4u) { free(buffer.data); return LANA_ERR_PARSE; }
            for (i = 0; i < 4u; ++i) { int h = hex_value(*parser->cursor++); if (h < 0) { free(buffer.data); return LANA_ERR_PARSE; } code = (code << 4u) | (unsigned)h; }
            if (code >= 0xd800u && code <= 0xdbffu) {
                unsigned low = 0u;
                if ((size_t)(parser->end - parser->cursor) < 6u || parser->cursor[0] != '\\' || parser->cursor[1] != 'u') { free(buffer.data); return LANA_ERR_PARSE; }
                parser->cursor += 2;
                for (i = 0; i < 4u; ++i) { int h = hex_value(*parser->cursor++); if (h < 0) { free(buffer.data); return LANA_ERR_PARSE; } low = (low << 4u) | (unsigned)h; }
                if (low < 0xdc00u || low > 0xdfffu) { free(buffer.data); return LANA_ERR_PARSE; }
                code = 0x10000u + ((code - 0xd800u) << 10u) + low - 0xdc00u;
            }
            if (!add_utf8(&buffer, code)) { free(buffer.data); return LANA_ERR_PARSE; }
        } else { free(buffer.data); return LANA_ERR_PARSE; }
    }
    if (parser->cursor >= parser->end) { free(buffer.data); return LANA_ERR_PARSE; }
    ++parser->cursor;
    if (buffer.data == NULL) { buffer.data = malloc(1u); if (buffer.data == NULL) return LANA_ERR_OOM; buffer.data[0] = '\0'; }
    if (!utf8_valid((const unsigned char *)buffer.data, buffer.length)) { free(buffer.data); return LANA_ERR_PARSE; }
    *out = buffer.data; return LANA_OK;
}

static LanaError json_value(JsonParser *parser, unsigned depth, Value *out) {
    LanaError error;
    if (depth > LANA_DATA_DEPTH_LIMIT) return LANA_ERR_LIMIT;
    json_space(parser);
    if (parser->cursor >= parser->end) return LANA_ERR_PARSE;
    if (*parser->cursor == '"') {
        char *temporary; char *stored;
        error = json_string(parser, &temporary); if (error != LANA_OK) return error;
        stored = vm_string(parser->vm, temporary, strlen(temporary)); free(temporary);
        if (stored == NULL) return LANA_ERR_OOM;
        *out = lana_value_string(stored);
        return LANA_OK;
    }
    if (*parser->cursor == '[') {
        LanaArray *array = lana_vm_alloc(parser->vm, sizeof(*array)); Value *temporary = NULL; size_t count = 0u, capacity = 0u;
        if (array == NULL) return LANA_ERR_OOM;
        ++parser->cursor;
        json_space(parser);
        while (parser->cursor < parser->end && *parser->cursor != ']') {
            Value item;
            error = json_value(parser, depth + 1u, &item); if (error != LANA_OK) { free(temporary); return error; }
            if (count == capacity) { size_t grown = capacity == 0u ? 4u : capacity * 2u; Value *items = realloc(temporary, grown * sizeof(*items)); if (items == NULL) { free(temporary); return LANA_ERR_OOM; } temporary = items; capacity = grown; }
            temporary[count++] = item; json_space(parser);
            if (parser->cursor < parser->end && *parser->cursor == ',') { ++parser->cursor; json_space(parser); if (parser->cursor < parser->end && *parser->cursor == ']') { free(temporary); return LANA_ERR_PARSE; } }
            else break;
        }
        if (parser->cursor >= parser->end || *parser->cursor++ != ']') { free(temporary); return LANA_ERR_PARSE; }
        array->count = count; array->capacity = count; array->items = count == 0u ? NULL : lana_vm_alloc(parser->vm, count * sizeof(*array->items));
        if (count > 0u && array->items == NULL) { free(temporary); return LANA_ERR_OOM; }
        if (count > 0u) memcpy(array->items, temporary, count * sizeof(*array->items));
        free(temporary);
        out->type = VAL_ARRAY;
        out->as.array = array;
        return LANA_OK;
    }
    if (*parser->cursor == '{') {
        LanaMap *map; error = lana_map_new(parser->vm, 4u, &map); if (error != LANA_OK) return error;
        ++parser->cursor; json_space(parser);
        while (parser->cursor < parser->end && *parser->cursor != '}') {
            char *key; Value item;
            error = json_string(parser, &key); if (error != LANA_OK) return error; json_space(parser);
            if (parser->cursor >= parser->end || *parser->cursor++ != ':') { free(key); return LANA_ERR_PARSE; }
            error = json_value(parser, depth + 1u, &item);
            if (error == LANA_OK) error = lana_map_set(parser->vm, map, key, &item, true);
            free(key); if (error != LANA_OK) return error == LANA_ERR_KEY ? LANA_ERR_PARSE : error; json_space(parser);
            if (parser->cursor < parser->end && *parser->cursor == ',') { ++parser->cursor; json_space(parser); if (parser->cursor < parser->end && *parser->cursor == '}') return LANA_ERR_PARSE; }
            else break;
        }
        if (parser->cursor >= parser->end || *parser->cursor++ != '}') return LANA_ERR_PARSE;
        *out = lana_value_map(map); return LANA_OK;
    }
    if ((size_t)(parser->end - parser->cursor) >= 4u && memcmp(parser->cursor, "null", 4u) == 0) { parser->cursor += 4; *out = lana_value_null(); return LANA_OK; }
    if ((size_t)(parser->end - parser->cursor) >= 4u && memcmp(parser->cursor, "true", 4u) == 0) { parser->cursor += 4; *out = lana_value_bool(true); return LANA_OK; }
    if ((size_t)(parser->end - parser->cursor) >= 5u && memcmp(parser->cursor, "false", 5u) == 0) { parser->cursor += 5; *out = lana_value_bool(false); return LANA_OK; }
    {
        char *end; double number; errno = 0; number = strtod((const char *)parser->cursor, &end);
        if (end == (char *)parser->cursor || errno != 0 || !isfinite(number)) return LANA_ERR_PARSE;
        if (*parser->cursor == '+' || (*parser->cursor == '0' && end > (char *)parser->cursor + 1 && isdigit(parser->cursor[1])) ||
            (parser->cursor[0] == '-' && parser->cursor + 2 < (const unsigned char *)end && parser->cursor[1] == '0' && isdigit(parser->cursor[2]))) return LANA_ERR_PARSE;
        parser->cursor = (const unsigned char *)end; *out = lana_value_number(number); return LANA_OK;
    }
}

LanaError lana_json_parse(LanaVM *vm, const char *text, Value *out) {
    JsonParser parser; LanaError error; size_t length;
    if (text == NULL) return LANA_ERR_TYPE;
    length = strlen(text);
    if (!utf8_valid((const unsigned char *)text, length)) return LANA_ERR_PARSE;
    parser.vm = vm; parser.cursor = (const unsigned char *)text; parser.end = parser.cursor + length;
    error = json_value(&parser, 0u, out); json_space(&parser);
    return error == LANA_OK && parser.cursor != parser.end ? LANA_ERR_PARSE : error;
}

static bool json_escape(Buffer *buffer, const char *text) {
    static const char hex[] = "0123456789abcdef"; const unsigned char *cursor = (const unsigned char *)text;
    if (!utf8_valid(cursor, strlen(text)) || !buffer_add(buffer, "\"", 1u)) return false;
    while (*cursor != 0u) { unsigned char c = *cursor++;
        if (c == '"' || c == '\\') { char pair[2] = {'\\', (char)c}; if (!buffer_add(buffer, pair, 2u)) return false; }
        else if (c < 0x20u) { char escaped[6] = {'\\','u','0','0',hex[c >> 4u],hex[c & 15u]}; if (!buffer_add(buffer, escaped, 6u)) return false; }
        else if (!buffer_add(buffer, (const char *)&c, 1u)) return false;
    }
    return buffer_add(buffer, "\"", 1u);
}

static LanaError json_emit(const Value *value, Buffer *buffer, const void **stack, unsigned depth) {
    size_t index; char number[32]; const void *identity = NULL;
    if (depth > LANA_DATA_DEPTH_LIMIT) return LANA_ERR_LIMIT;
    if (value->type == VAL_ARRAY) identity = value->as.array;
    else if (value->type == VAL_MAP) identity = value->as.map;
    if (identity != NULL) for (index = 0; index < depth; ++index) if (stack[index] == identity) return LANA_ERR_UNSUPPORTED_OPERATION;
    if (identity != NULL) stack[depth] = identity;
    switch (value->type) {
        case VAL_NULL: return buffer_add(buffer, "null", 4u) ? LANA_OK : LANA_ERR_OOM;
        case VAL_BOOL: return buffer_add(buffer, value->as.boolean ? "true" : "false", value->as.boolean ? 4u : 5u) ? LANA_OK : LANA_ERR_OOM;
        case VAL_NUMBER:
            if (!isfinite(value->as.number)) return LANA_ERR_UNSUPPORTED_OPERATION;
            if (value->as.number == 0.0) return buffer_add(buffer, "0", 1u) ? LANA_OK : LANA_ERR_OOM;
            (void)snprintf(number, sizeof(number), "%.17g", value->as.number);
            return buffer_add(buffer, number, strlen(number)) ? LANA_OK : LANA_ERR_OOM;
        case VAL_STRING: return json_escape(buffer, value->as.string) ? LANA_OK : LANA_ERR_PARSE;
        case VAL_ARRAY:
            if (!buffer_add(buffer, "[", 1u)) return LANA_ERR_OOM;
            for (index = 0; index < value->as.array->count; ++index) { LanaError error; if (index > 0u && !buffer_add(buffer, ",", 1u)) return LANA_ERR_OOM; error = json_emit(&value->as.array->items[index], buffer, stack, depth + 1u); if (error != LANA_OK) return error; }
            return buffer_add(buffer, "]", 1u) ? LANA_OK : LANA_ERR_OOM;
        case VAL_MAP:
            if (!buffer_add(buffer, "{", 1u)) return LANA_ERR_OOM;
            for (index = 0; index < value->as.map->count; ++index) { LanaError error; if (index > 0u && !buffer_add(buffer, ",", 1u)) return LANA_ERR_OOM; if (!json_escape(buffer, value->as.map->entries[index].key) || !buffer_add(buffer, ":", 1u)) return LANA_ERR_OOM; error = json_emit(value->as.map->entries[index].value, buffer, stack, depth + 1u); if (error != LANA_OK) return error; }
            return buffer_add(buffer, "}", 1u) ? LANA_OK : LANA_ERR_OOM;
        default: return LANA_ERR_UNSUPPORTED_OPERATION;
    }
}

LanaError lana_json_stringify(LanaVM *vm, const Value *value, Value *out) {
    Buffer buffer = {0}; const void *stack[LANA_DATA_DEPTH_LIMIT + 1u] = {0}; LanaError error = json_emit(value, &buffer, stack, 0u); char *stored;
    if (error != LANA_OK) { free(buffer.data); return error; }
    stored = vm_string(vm, buffer.data == NULL ? "" : buffer.data, buffer.length); free(buffer.data);
    if (stored == NULL) return LANA_ERR_OOM;
    *out = lana_value_string(stored);
    return LANA_OK;
}

typedef struct { char **items; size_t count; size_t capacity; } Fields;
static void fields_free(Fields *fields) { size_t i; for (i = 0; i < fields->count; ++i) free(fields->items[i]); free(fields->items); memset(fields, 0, sizeof(*fields)); }
static LanaError field_add(Fields *fields, Buffer *field) { char **grown; if (fields->count == fields->capacity) { size_t capacity = fields->capacity == 0u ? 8u : fields->capacity * 2u; grown = realloc(fields->items, capacity * sizeof(*grown)); if (grown == NULL) return LANA_ERR_OOM; fields->items = grown; fields->capacity = capacity; } if (field->data == NULL) { field->data = malloc(1u); if (field->data == NULL) return LANA_ERR_OOM; field->data[0] = '\0'; } fields->items[fields->count++] = field->data; memset(field, 0, sizeof(*field)); return LANA_OK; }

static LanaError csv_records(const char *text, size_t length, Fields **records_out, size_t *count_out) {
    Fields *records = NULL; size_t record_count = 0u, record_capacity = 0u, i = 0u; Fields fields = {0}; Buffer field = {0}; LanaError error = LANA_OK;
    while (i < length) {
        bool quoted = false;
        if (text[i] == '"') { quoted = true; ++i; while (i < length) { if (text[i] == '"') { if (i + 1u < length && text[i + 1u] == '"') { if (!buffer_add(&field, "\"", 1u)) { error = LANA_ERR_OOM; goto fail; } i += 2u; } else { ++i; break; } } else { if (!buffer_add(&field, &text[i++], 1u)) { error = LANA_ERR_OOM; goto fail; } } } if (i > length || (i == length && (length == 0u || text[length - 1u] != '"'))) { error = LANA_ERR_PARSE; goto fail; } }
        else while (i < length && text[i] != ',' && text[i] != '\r' && text[i] != '\n') { if (text[i] == '"' || !buffer_add(&field, &text[i], 1u)) { error = text[i] == '"' ? LANA_ERR_PARSE : LANA_ERR_OOM; goto fail; } ++i; }
        if (quoted && i < length && text[i] != ',' && text[i] != '\r' && text[i] != '\n') { error = LANA_ERR_PARSE; goto fail; }
        error = field_add(&fields, &field); if (error != LANA_OK) goto fail;
        if (i < length && text[i] == ',') { ++i; if (i == length) { error = field_add(&fields, &field); if (error != LANA_OK) goto fail; } continue; }
        if (i < length && text[i] == '\r') { if (i + 1u >= length || text[i + 1u] != '\n') { error = LANA_ERR_PARSE; goto fail; } i += 2u; }
        else if (i < length && text[i] == '\n') ++i;
        if (record_count == record_capacity) { size_t capacity = record_capacity == 0u ? 8u : record_capacity * 2u; Fields *grown = realloc(records, capacity * sizeof(*grown)); if (grown == NULL) { error = LANA_ERR_OOM; goto fail; } records = grown; record_capacity = capacity; }
        records[record_count++] = fields; memset(&fields, 0, sizeof(fields));
    }
    *records_out = records; *count_out = record_count; return LANA_OK;
fail:
    free(field.data); fields_free(&fields); while (record_count > 0u) fields_free(&records[--record_count]); free(records); return error;
}

LanaError lana_csv_read(LanaVM *vm, const char *path, Value *out) {
    FILE *file; long size; char *text; size_t read, record_count = 0u, row, column; Fields *records = NULL; LanaArray *array; LanaError error;
    if (path == NULL) return LANA_ERR_TYPE;
    file = fopen(path, "rb");
    if (file == NULL) return LANA_ERR_IO;
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) { fclose(file); return LANA_ERR_IO; }
    text = malloc((size_t)size + 1u); if (text == NULL) { fclose(file); return LANA_ERR_OOM; }
    read = fread(text, 1u, (size_t)size, file); if (read != (size_t)size || fclose(file) != 0) { free(text); return LANA_ERR_IO; } text[read] = '\0';
    if (read >= 3u && memcmp(text, "\xef\xbb\xbf", 3u) == 0) { memmove(text, text + 3, read - 2u); read -= 3u; }
    if (!utf8_valid((const unsigned char *)text, read)) { free(text); return LANA_ERR_PARSE; }
    error = csv_records(text, read, &records, &record_count); free(text); if (error != LANA_OK) return error;
    if (record_count > 0u) { for (column = 0; column < records[0].count; ++column) { size_t other; if (records[0].items[column][0] == '\0') { error = LANA_ERR_PARSE; goto cleanup; } for (other = 0; other < column; ++other) if (strcmp(records[0].items[column], records[0].items[other]) == 0) { error = LANA_ERR_PARSE; goto cleanup; } } }
    array = lana_vm_alloc(vm, sizeof(*array)); if (array == NULL) { error = LANA_ERR_OOM; goto cleanup; }
    array->count = record_count > 0u ? record_count - 1u : 0u; array->capacity = array->count; array->items = array->count == 0u ? NULL : lana_vm_alloc(vm, array->count * sizeof(*array->items)); if (array->count > 0u && array->items == NULL) { error = LANA_ERR_OOM; goto cleanup; }
    for (row = 1u; row < record_count; ++row) { LanaMap *map; if (records[row].count != records[0].count) { error = LANA_ERR_PARSE; goto cleanup; } error = lana_map_new(vm, records[0].count, &map); if (error != LANA_OK) goto cleanup; for (column = 0; column < records[0].count; ++column) { char *stored = vm_string(vm, records[row].items[column], strlen(records[row].items[column])); Value value; if (stored == NULL) { error = LANA_ERR_OOM; goto cleanup; } value = lana_value_string(stored); error = lana_map_set(vm, map, records[0].items[column], &value, true); if (error != LANA_OK) goto cleanup; } array->items[row - 1u] = lana_value_map(map); }
    out->type = VAL_ARRAY; out->as.array = array; error = LANA_OK;
cleanup:
    while (record_count > 0u) fields_free(&records[--record_count]);
    free(records);
    return error;
}

static bool csv_scalar(const Value *value, Buffer *field) { char number[32]; const char *text; size_t i; if (value->type == VAL_NULL) text = ""; else if (value->type == VAL_STRING) text = value->as.string; else if (value->type == VAL_BOOL) text = value->as.boolean ? "true" : "false"; else if (value->type == VAL_NUMBER && isfinite(value->as.number)) { (void)snprintf(number, sizeof(number), value->as.number == 0.0 ? "0" : "%.17g", value->as.number); text = number; } else return false; if (strpbrk(text, ",\"\r\n") == NULL) return buffer_add(field, text, strlen(text)); if (!buffer_add(field, "\"", 1u)) return false; for (i = 0; text[i] != '\0'; ++i) { if (text[i] == '"' && !buffer_add(field, "\"", 1u)) return false; if (!buffer_add(field, &text[i], 1u)) return false; } return buffer_add(field, "\"", 1u); }
LanaError lana_csv_write(LanaVM *vm, const char *path, const Value *rows, Value *out) {
    FILE *file; Buffer buffer = {0}; size_t row, column; LanaMap *header;
    (void)vm; if (path == NULL || rows == NULL || rows->type != VAL_ARRAY || rows->as.array->count == 0u || rows->as.array->items[0].type != VAL_MAP) return LANA_ERR_TYPE; header = rows->as.array->items[0].as.map;
    for (row = 0u; row <= rows->as.array->count; ++row) { LanaMap *map = row == 0u ? NULL : rows->as.array->items[row - 1u].as.map; if (row > 0u && (rows->as.array->items[row - 1u].type != VAL_MAP || map->count != header->count)) { free(buffer.data); return LANA_ERR_TYPE; } for (column = 0u; column < header->count; ++column) { Buffer field = {0}; const Value *value = row == 0u ? NULL : map->entries[column].value; if (row > 0u && strcmp(map->entries[column].key, header->entries[column].key) != 0) { free(buffer.data); return LANA_ERR_TYPE; } if (column > 0u && !buffer_add(&buffer, ",", 1u)) { free(buffer.data); return LANA_ERR_OOM; } if (row == 0u) { Value key = lana_value_string(header->entries[column].key); if (!csv_scalar(&key, &field)) { free(buffer.data); return LANA_ERR_TYPE; } } else if (!csv_scalar(value, &field)) { free(buffer.data); return LANA_ERR_TYPE; } if (!buffer_add(&buffer, field.data == NULL ? "" : field.data, field.length)) { free(field.data); free(buffer.data); return LANA_ERR_OOM; } free(field.data); } if (!buffer_add(&buffer, "\r\n", 2u)) { free(buffer.data); return LANA_ERR_OOM; } }
    file = fopen(path, "wb"); if (file == NULL) { free(buffer.data); return LANA_ERR_IO; } if (fwrite(buffer.data, 1u, buffer.length, file) != buffer.length || fclose(file) != 0) { free(buffer.data); return LANA_ERR_IO; } free(buffer.data); *out = lana_value_bool(true); return LANA_OK;
}
