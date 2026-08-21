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

Value ss_value_model(SSMLModel *model) {
    Value value = {0};
    value.type = VAL_MODEL;
    value.as.model = model;
    return value;
}

Value ss_value_state_dist(SSStateDist *distribution) {
    Value value = {0};
    value.type = VAL_STATE_DIST;
    value.as.state_dist = distribution;
    return value;
}

Value ss_value_map(SSMap *map) {
    Value value = {0};
    value.type = VAL_MAP;
    value.as.map = map;
    return value;
}

Value ss_value_array(SSArray *array) {
    Value value = {0};
    value.type = VAL_ARRAY;
    value.as.array = array;
    return value;
}

Value ss_value_possibility(SSPossibility *possibility) {
    Value value = {0};
    value.type = VAL_POSSIBILITY;
    value.as.possibility = possibility;
    return value;
}

Value ss_value_paths(SSPathSet *paths) {
    Value value = {0};
    value.type = VAL_PATH_SET;
    value.as.paths = paths;
    return value;
}

const char *ss_value_type_name(ValueType type) {
    static const char *names[] = {
        "null", "number", "bool", "string", "state", "distribution",
        "sample", "joint_state", "array", "function", "task", "model",
        "state_dist", "map", "possibility", "paths"
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
            (void)printf("state(p=%.12g, d_re=%.12g, d_im=%.12g)",
                         value->as.state.state.p, value->as.state.state.d_re,
                         value->as.state.state.d_im);
            break;
        case VAL_DISTRIBUTION:
            (void)printf("distribution(p0=%.12g, p1=%.12g)",
                         value->as.distribution.p0, value->as.distribution.p1);
            break;
        case VAL_SAMPLE: (void)printf("%d", value->as.sample); break;
        case VAL_JOINT_STATE:
            (void)printf("joint_state{");
            if (value->as.joint != NULL) {
                for (index = 0; index < value->as.joint->count; ++index) {
                    if (index > 0) (void)printf(", ");
                    (void)printf("%s: ", value->as.joint->names[index]);
                    if (value->as.joint->values != NULL)
                        ss_value_print(&value->as.joint->values[index]);
                    else
                        (void)printf("<finite-law>");
                }
            }
            (void)printf("}");
            break;
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
        case VAL_MODEL:
            (void)printf("model(%s, features=%zu)",
                         value->as.model != NULL && value->as.model->kind == SS_ML_LOGISTIC
                             ? "logistic" : "ridge",
                         value->as.model == NULL ? 0u : value->as.model->feature_count);
            break;
        case VAL_STATE_DIST: (void)printf("state_dist"); break;
        case VAL_MAP:
            (void)printf("{");
            for (index = 0; index < value->as.map->count; ++index) {
                if (index > 0) (void)printf(", ");
                (void)printf("\"%s\": ", value->as.map->entries[index].key);
                ss_value_print(value->as.map->entries[index].value);
            }
            (void)printf("}");
            break;
        case VAL_POSSIBILITY:
            (void)printf("possibility{");
            for (index = 0; index < value->as.possibility->count; ++index) {
                if (index > 0) (void)printf(", ");
                ss_value_print(&value->as.possibility->values[index]);
            }
            (void)printf("}");
            break;
        case VAL_PATH_SET:
            (void)printf("paths{");
            for (index = 0; index < value->as.paths->count; ++index) {
                if (index > 0) (void)printf(", ");
                (void)printf("%s => ", value->as.paths->alternatives[index].guard ? "true" : "false");
                ss_value_print(value->as.paths->alternatives[index].result);
            }
            (void)printf("}");
            break;
        default: (void)printf("<invalid>"); break;
    }
}
