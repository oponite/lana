#include "state.h"

#include <math.h>

static double positive_zero(double value) { return value == 0.0 ? 0.0 : value; }

double lana_state_disposition_squared(const LanaState *state) {
    if (state == NULL) return NAN;
    return state->d_re * state->d_re + state->d_im * state->d_im;
}

bool lana_state_valid(const LanaState *state) {
    double radius_squared;
    if (state == NULL || !isfinite(state->p) || !isfinite(state->d_re) ||
        !isfinite(state->d_im) || state->p < 0.0 || state->p > 1.0) return false;
    radius_squared = lana_state_disposition_squared(state);
    return isfinite(radius_squared) && radius_squared <= 1.0 &&
           ((state->p != 0.0 && state->p != 1.0) ||
            (state->d_re == 0.0 && state->d_im == 0.0));
}

LanaError lana_state_make_complex(double p, double d_re, double d_im, LanaState *out) {
    double radius_squared, radius;
    if (out == NULL || !isfinite(p) || !isfinite(d_re) || !isfinite(d_im))
        return LANA_ERR_INVALID_STATE;
    if (p < -LANA_STATE_EPSILON || p > 1.0 + LANA_STATE_EPSILON)
        return LANA_ERR_INVALID_STATE;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    radius_squared = d_re * d_re + d_im * d_im;
    radius = sqrt(radius_squared);
    if (!isfinite(radius) || radius > 1.0 + LANA_STATE_EPSILON)
        return LANA_ERR_INVALID_STATE;
    if (radius > 1.0) {
        d_re /= radius;
        d_im /= radius;
    }
    if (p == 0.0 || p == 1.0) d_re = d_im = 0.0;
    out->p = positive_zero(p);
    out->d_re = positive_zero(d_re);
    out->d_im = positive_zero(d_im);
    return lana_state_valid(out) ? LANA_OK : LANA_ERR_INVALID_STATE;
}

void lana_state_reconstruct_c(const LanaState *state, double *c_re, double *c_im) {
    double scale = 0.0;
    if (state != NULL && lana_state_valid(state)) scale = sqrt(state->p * (1.0 - state->p));
    if (c_re != NULL) *c_re = state == NULL ? NAN : positive_zero(state->d_re * scale);
    if (c_im != NULL) *c_im = state == NULL ? NAN : positive_zero(state->d_im * scale);
}

LanaError lana_state_basis_probability(uint32_t basis, const LanaState *state,
                                   double *out) {
    double c_re, c_im, probability;
    if (out == NULL || !lana_state_valid(state)) return LANA_ERR_INVALID_STATE;
    lana_state_reconstruct_c(state, &c_re, &c_im);
    switch ((LanaBasisId)basis) {
        case LANA_BASIS_COMPUTATIONAL:
            probability = state->p;
            break;
        case LANA_BASIS_X:
            probability = 0.5 - c_re;
            break;
        case LANA_BASIS_Y:
            /* |-y> is outcome 1, so P(-y) = 1/2 + Im(c). */
            probability = 0.5 + c_im;
            break;
        default:
            return LANA_ERR_MEASURE;
    }
    if (!isfinite(probability) || probability < -LANA_STATE_EPSILON ||
        probability > 1.0 + LANA_STATE_EPSILON)
        return LANA_ERR_INVALID_STATE;
    if (probability < 0.0) probability = 0.0;
    if (probability > 1.0) probability = 1.0;
    *out = positive_zero(probability);
    return LANA_OK;
}

LanaError lana_state_append_parameters(const LanaState *left, const LanaState *right,
                                   double *p, double *m_re, double *m_im,
                                   double *sigma) {
    double delta_re, delta_im;
    if (!lana_state_valid(left) || !lana_state_valid(right) || p == NULL ||
        m_re == NULL || m_im == NULL || sigma == NULL) return LANA_ERR_INVALID_STATE;
    *p = 1.0 - (1.0 - left->p) * (1.0 - right->p);
    *m_re = (left->d_re + right->d_re) / 2.0;
    *m_im = (left->d_im + right->d_im) / 2.0;
    delta_re = left->d_re - right->d_re;
    delta_im = left->d_im - right->d_im;
    *sigma = hypot(delta_re, delta_im) / 2.0;
    return LANA_OK;
}

static LanaError transform_invert_current(const LanaState *source, LanaState *out) {
    return lana_state_make_complex(1.0 - source->p, source->d_re, -source->d_im, out);
}

static LanaError transform_neutralize_current(const LanaState *source, LanaState *out) {
    return lana_state_make_complex(source->p, 0.0, 0.0, out);
}

static double expectation_invert(double probability) { return 1.0 - probability; }
static double expectation_identity(double probability) { return probability; }

static const LanaTransformSpec transform_specs_current[] = {
    {0u, "invert", transform_invert_current, expectation_invert, true},
    {1u, "neutralize", transform_neutralize_current, expectation_identity, true}
};

const LanaTransformSpec *lana_transform_spec(uint32_t identifier) {
    if (identifier >= sizeof(transform_specs_current) / sizeof(transform_specs_current[0])) return NULL;
    return &transform_specs_current[identifier];
}

LanaError lana_transform_apply(uint32_t identifier, const LanaState *source, LanaState *out) {
    const LanaTransformSpec *specification = lana_transform_spec(identifier);
    LanaError error;
    if (specification == NULL || specification->concrete == NULL)
        return LANA_ERR_UNSUPPORTED_OPERATION;
    if (!lana_state_valid(source)) return LANA_ERR_INVALID_STATE;
    error = specification->concrete(source, out);
    if (error != LANA_OK || !lana_state_valid(out)) return LANA_ERR_INVALID_TRANSFORM_RESULT;
    return LANA_OK;
}

LanaError lana_transform_expected_probability(uint32_t identifier, double input, double *out) {
    const LanaTransformSpec *specification = lana_transform_spec(identifier);
    if (specification == NULL || !specification->distribution_liftable ||
        specification->exact_expected_probability == NULL)
        return LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT;
    if (out == NULL || !isfinite(input) || input < 0.0 || input > 1.0)
        return LANA_ERR_INVALID_DISTRIBUTION;
    *out = specification->exact_expected_probability(input);
    return LANA_OK;
}

LanaError lana_state_make(double p, double d, LanaState *out) {
    return lana_state_make_complex(p, d, 0.0, out);
}

LanaError lana_state_mix(const LanaState *left, const LanaState *right, double w,
                         LanaState *out) {
    double c_left_re, c_left_im, c_right_re, c_right_im;
    double c_re, c_im, p, scale;
    if (!lana_state_valid(left) || !lana_state_valid(right) || out == NULL)
        return LANA_ERR_INVALID_STATE;
    if (!isfinite(w) || w < 0.0 || w > 1.0) return LANA_ERR_INVALID_PARAMETERS;
    p = w * left->p + (1.0 - w) * right->p;
    lana_state_reconstruct_c(left, &c_left_re, &c_left_im);
    lana_state_reconstruct_c(right, &c_right_re, &c_right_im);
    c_re = w * c_left_re + (1.0 - w) * c_right_re;
    c_im = w * c_left_im + (1.0 - w) * c_right_im;
    if (p == 0.0 || p == 1.0) return lana_state_make_complex(p, 0.0, 0.0, out);
    scale = sqrt(p * (1.0 - p));
    return lana_state_make_complex(p, c_re / scale, c_im / scale, out);
}

LanaError lana_state_attenuate(const LanaState *source, double factor, LanaState *out) {
    if (!lana_state_valid(source) || out == NULL) return LANA_ERR_INVALID_STATE;
    if (!isfinite(factor) || factor < 0.0 || factor > 1.0)
        return LANA_ERR_INVALID_PARAMETERS;
    return lana_state_make_complex(source->p, factor * source->d_re,
                                   factor * source->d_im, out);
}

LanaError lana_state_trace_distance(const LanaState *left, const LanaState *right,
                                    double *out) {
    double c_left_re, c_left_im, c_right_re, c_right_im;
    double dp, dc_re, dc_im;
    if (!lana_state_valid(left) || !lana_state_valid(right) || out == NULL)
        return LANA_ERR_INVALID_STATE;
    lana_state_reconstruct_c(left, &c_left_re, &c_left_im);
    lana_state_reconstruct_c(right, &c_right_re, &c_right_im);
    dp = left->p - right->p;
    dc_re = c_left_re - c_right_re;
    dc_im = c_left_im - c_right_im;
    *out = sqrt(dp * dp + dc_re * dc_re + dc_im * dc_im);
    return LANA_OK;
}

static double append_overlap(uint32_t mode, double p_a, double p_b, double strength,
                             LanaError *error) {
    double independent = p_a * p_b;
    double bound;
    switch ((LanaAppendRelationshipId)mode) {
        case LANA_APPEND_INDEPENDENT:
            return independent;
        case LANA_APPEND_REDUNDANT:
            if (!isfinite(strength) || strength < 0.0 || strength >= 1.0) {
                *error = LANA_ERR_INVALID_PARAMETERS;
                return NAN;
            }
            bound = p_a < p_b ? p_a : p_b;
            return (1.0 - strength) * independent + strength * bound;
        case LANA_APPEND_FULL_REDUNDANCY:
            if (p_a != p_b) {
                *error = LANA_ERR_INVALID_PARAMETERS;
                return NAN;
            }
            return p_a;
        case LANA_APPEND_COMPLEMENTARY:
            if (!isfinite(strength) || strength < 0.0 || strength > 1.0) {
                *error = LANA_ERR_INVALID_PARAMETERS;
                return NAN;
            }
            bound = p_a + p_b - 1.0;
            if (bound < 0.0) bound = 0.0;
            return (1.0 - strength) * independent + strength * bound;
        default:
            *error = LANA_ERR_UNSUPPORTED_OPERATION;
            return NAN;
    }
}

LanaError lana_state_append_relationship_parameters(const LanaState *left,
                                    const LanaState *right, uint32_t mode,
                                    double strength, double *p, double *m_re,
                                    double *m_im, double *sigma) {
    double delta_re, delta_im, q;
    LanaError error = LANA_OK;
    if (!lana_state_valid(left) || !lana_state_valid(right) || p == NULL ||
        m_re == NULL || m_im == NULL || sigma == NULL) return LANA_ERR_INVALID_STATE;
    q = append_overlap(mode, left->p, right->p, strength, &error);
    if (error != LANA_OK) return error;
    *p = left->p + right->p - q;
    *m_re = (left->d_re + right->d_re) / 2.0;
    *m_im = (left->d_im + right->d_im) / 2.0;
    delta_re = left->d_re - right->d_re;
    delta_im = left->d_im - right->d_im;
    *sigma = hypot(delta_re, delta_im) / 2.0;
    return LANA_OK;
}
