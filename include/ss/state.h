#ifndef SS_STATE_H
#define SS_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ss/error.h"

#define SS_STATE_EPSILON 1e-12

typedef struct {
    double p;
    union {
        double d_re;
        double d; /* Legacy v1/v2 source compatibility: real axis only. */
    };
    double d_im;
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

typedef enum {
    SS_BASIS_COMPUTATIONAL = 0,
    SS_BASIS_X = 1,
    SS_BASIS_Y = 2
} SSBasisId;

bool ss_state_valid(const SSState *state);
bool ss_state_legacy_valid(const SSState *state);
SSError ss_state_make_complex(double p, double d_re, double d_im, SSState *out);
SSError ss_state_make(double p, double d, SSState *out);
double ss_state_disposition_squared(const SSState *state);
void ss_state_reconstruct_c(const SSState *state, double *c_re, double *c_im);
SSError ss_state_basis_probability(uint32_t basis, const SSState *state,
                                   double *out);
SSError ss_state_append_parameters(const SSState *left, const SSState *right,
                                   double *p, double *m_re, double *m_im,
                                   double *sigma);

typedef SSError (*SSConcreteTransformFn)(const SSState *source, SSState *out);
typedef double (*SSExpectedProbabilityFn)(double expected_probability);

typedef struct {
    uint32_t identifier;
    const char *name;
    SSConcreteTransformFn concrete;
    SSExpectedProbabilityFn exact_expected_probability;
    bool distribution_liftable;
} SSTransformSpecV3;

const SSTransformSpecV3 *ss_transform_v3_spec(uint32_t identifier);
SSError ss_transform_v3_apply(uint32_t identifier, const SSState *source, SSState *out);
SSError ss_transform_v3_expected_probability(uint32_t identifier, double input, double *out);

/* The functions below are legacy-only signed-real v1/v2 semantics. */
SSError ss_apply(const SSState *source, SSState *target);
SSError ss_compose_merge(const SSState *left, const SSState *right, SSState *out);
SSError ss_compose_update(const SSState *left, const SSState *right,
                          SSState *left_out, SSState *right_out);

#endif
