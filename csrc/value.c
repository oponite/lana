#include "ss/value.h"
#include "ss/vm.h"

#include <stdio.h>

Value ss_value_null(void) {
    Value value = {0};
    value.type = VAL_NULL;
    return value;
}

Value ss_value_number(double number) {
    Value value = {0};
    value.type = VAL_NUMBER;
    value.as.number = number;
    return value;
}

Value ss_value_bool(bool boolean) {
    Value value = {0};
    value.type = VAL_BOOL;
    value.as.boolean = boolean;
    return value;
}

Value ss_value_string(const char *string) {
    Value value = {0};
    value.type = VAL_STRING;
    value.as.string = string;
    return value;
}

Value ss_value_state(SSState state) {
    Value value = {0};
    value.type = VAL_STATE;
    value.as.state.state = state;
    return value;
}

Value ss_value_distribution(double p0, double p1) {
    Value value = {0};
    value.type = VAL_DISTRIBUTION;
    value.as.distribution.p0 = p0;
    value.as.distribution.p1 = p1;
    return value;
}

Value ss_value_sample(int sample) {
    Value value = {0};
    value.type = VAL_SAMPLE;
    value.as.sample = sample;
    return value;
}

const char *ss_value_type_name(ValueType type) {
    static const char *names[] = {
        "null", "number", "bool", "string", "state", "distribution",
        "sample", "joint_state", "array", "function", "task"
    };
    if ((size_t)type >= sizeof(names) / sizeof(names[0])) {
        return "unknown";
    }
    return names[type];
}

void ss_value_print(const Value *value) {
    size_t index;
    if (value == NULL) {
        (void)printf("null");
        return;
    }
    switch (value->type) {
        case VAL_NULL: (void)printf("null"); break;
        case VAL_NUMBER: (void)printf("%.12g", value->as.number); break;
        case VAL_BOOL: (void)printf("%s", value->as.boolean ? "true" : "false"); break;
        case VAL_STRING: (void)printf("%s", value->as.string); break;
        case VAL_STATE:
            (void)printf("state(p=%.12g, d=%.12g)", value->as.state.state.p,
                         value->as.state.state.d);
            break;
        case VAL_DISTRIBUTION:
            (void)printf("distribution(p0=%.12g, p1=%.12g)",
                         value->as.distribution.p0, value->as.distribution.p1);
            break;
        case VAL_SAMPLE: (void)printf("%d", value->as.sample); break;
        case VAL_JOINT_STATE: (void)printf("joint_state"); break;
        case VAL_ARRAY:
            (void)printf("[");
            for (index = 0; index < value->as.array->count; ++index) {
                if (index > 0) (void)printf(", ");
                ss_value_print(&value->as.array->items[index]);
            }
            (void)printf("]");
            break;
        case VAL_FUNCTION: (void)printf("function(%u)", value->as.function); break;
        case VAL_TASK:
            (void)printf("task(%llu)",
                         (unsigned long long)(value->as.task == NULL ? 0u : value->as.task->id));
            break;
        default: (void)printf("<invalid>"); break;
    }
}
