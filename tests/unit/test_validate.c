#include "data.h"
#include "state.h"
#include "vm.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { (void)fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } } while (0)

static LanaInstruction instruction(OpCode opcode, uint32_t a, uint32_t b,
                                 uint32_t c, uint32_t imm) {
    LanaInstruction value = {(uint8_t)opcode, a, b, c, imm, 1u};
    return value;
}

static LanaError build_schema(LanaVM *vm, LanaMap **out) {
    LanaMap *schema, *constraints;
    Value type_value, min_value, max_value, constraints_value, exactness_value;
    LanaError error;
    if ((error = lana_map_new(vm, 5u, &schema)) != LANA_OK) return error;
    type_value = lana_value_string("number");
    if ((error = lana_map_set(vm, schema, "type", &type_value, true)) != LANA_OK) return error;
    if ((error = lana_map_new(vm, 2u, &constraints)) != LANA_OK) return error;
    min_value = lana_value_number(0.0);
    if ((error = lana_map_set(vm, constraints, "min", &min_value, true)) != LANA_OK) return error;
    max_value = lana_value_number(10.0);
    if ((error = lana_map_set(vm, constraints, "max", &max_value, true)) != LANA_OK) return error;
    constraints_value = lana_value_map(constraints);
    if ((error = lana_map_set(vm, schema, "constraints", &constraints_value, true)) != LANA_OK) return error;
    exactness_value = lana_value_string("exact");
    if ((error = lana_map_set(vm, schema, "exactness", &exactness_value, true)) != LANA_OK) return error;
    *out = schema;
    return LANA_OK;
}

static int test_vm_validate_opcode(void) {
    LanaChunk chunk;
    LanaVM vm;
    LanaMap *schema;
    Value field;
    LanaErrorInfo error = {0};

    /* Valid value returns status valid. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_VALIDATE, 2, 0, 1, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(build_schema(&vm, &schema) == LANA_OK);
    vm.frames[0].registers[0] = lana_value_number(3.0);
    vm.frames[0].registers[1] = lana_value_map(schema);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(vm.frames[0].registers[2].type == VAL_MAP);
    CHECK(lana_map_get(vm.frames[0].registers[2].as.map, "status", &field) == LANA_OK);
    CHECK(field.type == VAL_STRING && strcmp(field.as.string, "valid") == 0);
    CHECK(lana_map_get(vm.frames[0].registers[2].as.map, "schema_version", &field) == LANA_OK);
    CHECK(field.type == VAL_NUMBER && field.as.number == 1.0);
    CHECK(vm.frames[0].registers[2].derivation != NULL);
    CHECK(strcmp(vm.frames[0].registers[2].derivation->operation, "validate") == 0);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Constraint violation returns invalid. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_VALIDATE, 2, 0, 1, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(build_schema(&vm, &schema) == LANA_OK);
    vm.frames[0].registers[0] = lana_value_number(15.0);
    vm.frames[0].registers[1] = lana_value_map(schema);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(lana_map_get(vm.frames[0].registers[2].as.map, "status", &field) == LANA_OK);
    CHECK(field.type == VAL_STRING && strcmp(field.as.string, "invalid") == 0);
    CHECK(lana_map_get(vm.frames[0].registers[2].as.map, "reason", &field) == LANA_OK);
    CHECK(field.type == VAL_STRING && strcmp(field.as.string, "above_maximum") == 0);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Type mismatch returns invalid. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_VALIDATE, 2, 0, 1, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(build_schema(&vm, &schema) == LANA_OK);
    vm.frames[0].registers[0] = lana_value_string("hello");
    vm.frames[0].registers[1] = lana_value_map(schema);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(lana_map_get(vm.frames[0].registers[2].as.map, "status", &field) == LANA_OK);
    CHECK(field.type == VAL_STRING && strcmp(field.as.string, "invalid") == 0);
    CHECK(lana_map_get(vm.frames[0].registers[2].as.map, "reason", &field) == LANA_OK);
    CHECK(field.type == VAL_STRING && strcmp(field.as.string, "type_mismatch") == 0);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Malformed schema (not a map) returns LANA_ERR_SCHEMA. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_VALIDATE, 2, 0, 1, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    vm.frames[0].registers[0] = lana_value_number(3.0);
    vm.frames[0].registers[1] = lana_value_number(1.0);
    CHECK(lana_vm_run(&vm) == LANA_ERR_SCHEMA);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Unresolved derivation returns insufficient_evidence. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_VALIDATE, 2, 0, 1, 0)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    lana_vm_init(&vm, &chunk);
    CHECK(build_schema(&vm, &schema) == LANA_OK);
    {
        LanaDerivation *derivation = lana_vm_alloc(&vm, sizeof(*derivation));
        Value value;
        CHECK(derivation != NULL);
        memset(derivation, 0, sizeof(*derivation));
        derivation->outcome = LANA_DERIVATION_UNRESOLVED;
        value = lana_value_number(3.0);
        value.derivation = derivation;
        vm.frames[0].registers[0] = value;
    }
    vm.frames[0].registers[1] = lana_value_map(schema);
    CHECK(lana_vm_run(&vm) == LANA_OK);
    CHECK(lana_map_get(vm.frames[0].registers[2].as.map, "status", &field) == LANA_OK);
    CHECK(field.type == VAL_STRING && strcmp(field.as.string, "insufficient_evidence") == 0);
    lana_vm_free(&vm);
    lana_chunk_free(&chunk);

    /* Verifier rejects non-zero imm. */
    lana_chunk_init(&chunk);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_VALIDATE, 2, 0, 1, 1)) == LANA_OK);
    CHECK(lana_chunk_emit(&chunk, instruction(OP_HALT, 0, 0, 0, 0)) == LANA_OK);
    CHECK(lana_chunk_verify(&chunk, &error) == LANA_ERR_FORMAT);
    lana_chunk_free(&chunk);
    return 0;
}

int main(void) {
    CHECK(test_vm_validate_opcode() == 0);
    (void)printf("VALIDATE tests passed\n");
    return 0;
}
