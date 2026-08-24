#ifndef LANA_STATE_H
#define LANA_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "lana/error.h"

#define LANA_STATE_EPSILON 1e-12

typedef struct {
    double p;
    union {
        double d_re;
        double d; /* Real-axis shorthand for d_re. */
    };
    double d_im;
} LanaState;

typedef struct {
    bool has_timestamp;
    bool has_source;
    bool has_weight;
    bool has_confidence;
    double timestamp;
    double weight;
    double confidence;
    const char *source;
} LanaIndexes;

typedef struct {
    LanaState state;
    LanaIndexes indexes;
} LanaStateValue;

typedef enum {
    LANA_BASIS_COMPUTATIONAL = 0,
    LANA_BASIS_X = 1,
    LANA_BASIS_Y = 2
} LanaBasisId;

bool lana_state_valid(const LanaState *state);
LanaError lana_state_make_complex(double p, double d_re, double d_im, LanaState *out);
LanaError lana_state_make(double p, double d, LanaState *out);
double lana_state_disposition_squared(const LanaState *state);
void lana_state_reconstruct_c(const LanaState *state, double *c_re, double *c_im);
LanaError lana_state_basis_probability(uint32_t basis, const LanaState *state,
                                   double *out);
LanaError lana_state_append_parameters(const LanaState *left, const LanaState *right,
                                   double *p, double *m_re, double *m_im,
                                   double *sigma);

typedef LanaError (*LanaConcreteTransformFn)(const LanaState *source, LanaState *out);
typedef double (*LanaExpectedProbabilityFn)(double expected_probability);

typedef struct {
    uint32_t identifier;
    const char *name;
    LanaConcreteTransformFn concrete;
    LanaExpectedProbabilityFn exact_expected_probability;
    bool distribution_liftable;
} LanaTransformSpec;

const LanaTransformSpec *lana_transform_spec(uint32_t identifier);
LanaError lana_transform_apply(uint32_t identifier, const LanaState *source, LanaState *out);
LanaError lana_transform_expected_probability(uint32_t identifier, double input, double *out);

#endif
