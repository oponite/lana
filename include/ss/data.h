#ifndef SS_DATA_H
#define SS_DATA_H

#include "ss/vm.h"

SSError ss_map_new(VM *vm, size_t capacity, SSMap **out);
SSError ss_map_get(const SSMap *map, const char *key, Value *out);
SSError ss_map_set(VM *vm, SSMap *map, const char *key, const Value *value,
                   bool reject_existing);
SSError ss_json_parse(VM *vm, const char *text, Value *out);
SSError ss_json_stringify(VM *vm, const Value *value, Value *out);
SSError ss_csv_read(VM *vm, const char *path, Value *out);
SSError ss_csv_write(VM *vm, const char *path, const Value *rows, Value *out);

#endif
