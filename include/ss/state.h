#ifndef SS_STATE_H
#define SS_STATE_H

#include <stdbool.h>
#include <stddef.h>

#include "ss/error.h"

#define SS_STATE_EPSILON 1e-12

typedef struct {
    double p;
    double d;
} SSState;

typedef struct {
    bool has_timestamp;
    bool has_source;
    bool has_weight;
    bool has_confidence;
    double timestamp;
    double weight;
    double confidence;
    const char *source;
} SSIndexes;

typedef struct {
    SSState state;
    SSIndexes indexes;
} SSStateValue;

bool ss_state_valid(const SSState *state);
SSError ss_state_make(double p, double d, SSState *out);
SSError ss_apply(const SSState *source, SSState *target);
SSError ss_compose_merge(const SSState *left, const SSState *right, SSState *out);
SSError ss_compose_update(const SSState *left, const SSState *right,
                          SSState *left_out, SSState *right_out);

#endif
