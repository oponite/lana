#ifndef SS_VALUE_H
#define SS_VALUE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ss/state.h"

typedef enum {
    VAL_NULL = 0,
    VAL_NUMBER,
    VAL_BOOL,
    VAL_STRING,
    VAL_STATE,
    VAL_DISTRIBUTION,
    VAL_SAMPLE,
    VAL_JOINT_STATE,
    VAL_ARRAY,
    VAL_FUNCTION,
    VAL_TASK
} ValueType;

typedef struct {
    double p0;
    double p1;
} SSDistribution;

typedef struct {
    SSStateValue left;
    SSStateValue right;
} SSJointState;

typedef struct Value Value;
typedef struct SSTask SSTask;

typedef struct {
    size_t count;
    Value *items;
} SSArray;

struct Value {
    ValueType type;
    union {
        double number;
        bool boolean;
        const char *string;
        SSStateValue state;
        SSDistribution distribution;
        int sample;
        SSJointState *joint;
        SSArray *array;
        uint32_t function;
        SSTask *task;
    } as;
};

Value ss_value_null(void);
Value ss_value_number(double number);
Value ss_value_bool(bool boolean);
Value ss_value_string(const char *string);
Value ss_value_state(SSState state);
Value ss_value_distribution(double p0, double p1);
Value ss_value_sample(int sample);
const char *ss_value_type_name(ValueType type);
void ss_value_print(const Value *value);

#endif
