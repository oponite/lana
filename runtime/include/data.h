#ifndef LANA_DATA_H
#define LANA_DATA_H

#include "vm.h"

LanaError lana_map_new(LanaVM *vm, size_t capacity, LanaMap **out);
LanaError lana_map_get(const LanaMap *map, const char *key, Value *out);
ptrdiff_t lana_map_has(const LanaMap *map, const char *key);
LanaError lana_map_set(LanaVM *vm, LanaMap *map, const char *key, const Value *value,
                   bool reject_existing);
LanaError lana_json_parse(LanaVM *vm, const char *text, Value *out);
LanaError lana_json_parse_offset(LanaVM *vm, const char *text, Value *out, size_t *error_offset);
/* Decode one JSON string; the caller owns the returned allocation. */
LanaError lana_json_decode_string(const unsigned char *data, size_t length, size_t *offset, char **out);
LanaError lana_json_stringify(LanaVM *vm, const Value *value, Value *out);
LanaError lana_csv_read(LanaVM *vm, const char *path, Value *out);
LanaError lana_csv_write(LanaVM *vm, const char *path, const Value *rows, Value *out);

#endif
