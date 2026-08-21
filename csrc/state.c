#include "ss/state.h"

#include <math.h>

static double clip_probability(double value) {
    if (value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

static double positive_zero(double value) { return value == 0.0 ? 0.0 : value; }

double ss_state_disposition_squared(const SSState *state) {
    if (state == NULL) return NAN;
    return state->d_re * state->d_re + state->d_im * state->d_im;
}

bool ss_state_valid(const SSState *state) {
    double radius_squared;
    if (state == NULL || !isfinite(state->p) || !isfinite(state->d_re) ||
        !isfinite(state->d_im) || state->p < 0.0 || state->p > 1.0) return false;
    radius_squared = ss_state_disposition_squared(state);
    return isfinite(radius_squared) && radius_squared <= 1.0 &&
           ((state->p != 0.0 && state->p != 1.0) ||
            (state->d_re == 0.0 && state->d_im == 0.0));
}

bool ss_state_legacy_valid(const SSState *state) {
    return state != NULL && isfinite(state->p) && isfinite(state->d_re) &&
           state->p >= 0.0 && state->p <= 1.0 && state->d_re > -1.0 &&
           state->d_re < 1.0 && state->d_im == 0.0;
}

SSError ss_state_make_complex(double p, double d_re, double d_im, SSState *out) {
    double radius_squared, radius;
    if (out == NULL || !isfinite(p) || !isfinite(d_re) || !isfinite(d_im))
        return SS_ERR_INVALID_STATE;
    if (p < -SS_STATE_EPSILON || p > 1.0 + SS_STATE_EPSILON)
        return SS_ERR_INVALID_STATE;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    radius_squared = d_re * d_re + d_im * d_im;
    radius = sqrt(radius_squared);
    if (!isfinite(radius) || radius > 1.0 + SS_STATE_EPSILON)
        return SS_ERR_INVALID_STATE;
    if (radius > 1.0) {
        d_re /= radius;
        d_im /= radius;
    }
    if (p == 0.0 || p == 1.0) d_re = d_im = 0.0;
    out->p = positive_zero(p);
    out->d_re = positive_zero(d_re);
    out->d_im = positive_zero(d_im);
    return ss_state_valid(out) ? SS_OK : SS_ERR_INVALID_STATE;
}

void ss_state_reconstruct_c(const SSState *state, double *c_re, double *c_im) {
    double scale = 0.0;
    if (state != NULL && ss_state_valid(state)) scale = sqrt(state->p * (1.0 - state->p));
    if (c_re != NULL) *c_re = state == NULL ? NAN : positive_zero(state->d_re * scale);
    if (c_im != NULL) *c_im = state == NULL ? NAN : positive_zero(state->d_im * scale);
}

SSError ss_state_basis_probability(uint32_t basis, const SSState *state,
                                   double *out) {
    double c_re, c_im, probability;
    if (out == NULL || !ss_state_valid(state)) return SS_ERR_INVALID_STATE;
    ss_state_reconstruct_c(state, &c_re, &c_im);
    switch ((SSBasisId)basis) {
        case SS_BASIS_COMPUTATIONAL:
            probability = state->p;
            break;
        case SS_BASIS_X:
            probability = 0.5 + c_re;
            break;
        case SS_BASIS_Y:
            /* |+y> = (|0> + i|1>) / sqrt(2), so P(+y) = 1/2 - Im(c). */
            probability = 0.5 - c_im;
            break;
        default:
            return SS_ERR_MEASURE;
    }
    if (!isfinite(probability) || probability < -SS_STATE_EPSILON ||
        probability > 1.0 + SS_STATE_EPSILON)
        return SS_ERR_INVALID_STATE;
    if (probability < 0.0) probability = 0.0;
    if (probability > 1.0) probability = 1.0;
    *out = positive_zero(probability);
    return SS_OK;
}

SSError ss_state_append_parameters(const SSState *left, const SSState *right,
                                   double *p, double *m_re, double *m_im,
                                   double *sigma) {
    double delta_re, delta_im;
    if (!ss_state_valid(left) || !ss_state_valid(right) || p == NULL ||
        m_re == NULL || m_im == NULL || sigma == NULL) return SS_ERR_INVALID_STATE;
    *p = 1.0 - (1.0 - left->p) * (1.0 - right->p);
    *m_re = (left->d_re + right->d_re) / 2.0;
    *m_im = (left->d_im + right->d_im) / 2.0;
    delta_re = left->d_re - right->d_re;
    delta_im = left->d_im - right->d_im;
    *sigma = hypot(delta_re, delta_im) / 2.0;
    return SS_OK;
}

static SSError transform_invert_v3(const SSState *source, SSState *out) {
    return ss_state_make_complex(1.0 - source->p, source->d_re, -source->d_im, out);
}

static SSError transform_neutralize_v3(const SSState *source, SSState *out) {
    return ss_state_make_complex(source->p, 0.0, 0.0, out);
}

static double expectation_invert(double probability) { return 1.0 - probability; }
static double expectation_identity(double probability) { return probability; }

static const SSTransformSpecV3 transform_specs_v3[] = {
    {0u, "invert", transform_invert_v3, expectation_invert, true},
    {1u, "neutralize", transform_neutralize_v3, expectation_identity, true}
};

const SSTransformSpecV3 *ss_transform_v3_spec(uint32_t identifier) {
    if (identifier >= sizeof(transform_specs_v3) / sizeof(transform_specs_v3[0])) return NULL;
    return &transform_specs_v3[identifier];
}

SSError ss_transform_v3_apply(uint32_t identifier, const SSState *source, SSState *out) {
    const SSTransformSpecV3 *specification = ss_transform_v3_spec(identifier);
    SSError error;
    if (specification == NULL || specification->concrete == NULL)
        return SS_ERR_UNSUPPORTED_OPERATION;
    if (!ss_state_valid(source)) return SS_ERR_INVALID_STATE;
    error = specification->concrete(source, out);
    if (error != SS_OK || !ss_state_valid(out)) return SS_ERR_INVALID_TRANSFORM_RESULT;
    return SS_OK;
}

SSError ss_transform_v3_expected_probability(uint32_t identifier, double input, double *out) {
    const SSTransformSpecV3 *specification = ss_transform_v3_spec(identifier);
    if (specification == NULL || !specification->distribution_liftable ||
        specification->exact_expected_probability == NULL)
        return SS_ERR_UNSUPPORTED_EXACT_MEASUREMENT;
    if (out == NULL || !isfinite(input) || input < 0.0 || input > 1.0)
        return SS_ERR_INVALID_DISTRIBUTION;
    *out = specification->exact_expected_probability(input);
    return SS_OK;
}

/* Legacy v1/v2 signed-real implementation. */
SSError ss_state_make(double p, double d, SSState *out) {
    if (out == NULL) return SS_ERR_INVALID_STATE;
    if (!isfinite(p) || p < 0.0 || p > 1.0) return SS_ERR_INVALID_PROBABILITY;
    if (!isfinite(d) || d <= -1.0 || d >= 1.0) return SS_ERR_INVALID_DEPENDENCY;
    out->p = p;
    out->d_re = d;
    out->d_im = 0.0;
    return SS_OK;
}

SSError ss_apply(const SSState *source, SSState *target) {
    double influence_target;
    double strength;
    if (!ss_state_legacy_valid(source) || !ss_state_legacy_valid(target))
        return SS_ERR_INVALID_STATE;
    influence_target = source->d_re < 0.0 ? 1.0 - source->p : source->p;
    strength = fabs(source->d_re);
    target->p = clip_probability(target->p + strength * (influence_target - target->p));
    return ss_state_legacy_valid(target) ? SS_OK : SS_ERR_INVALID_STATE;
}

SSError ss_compose_merge(const SSState *left, const SSState *right, SSState *out) {
    if (!ss_state_legacy_valid(left) || !ss_state_legacy_valid(right) || out == NULL)
        return SS_ERR_INVALID_STATE;
    return ss_state_make((left->p + right->p) / 2.0,
                         (left->d_re + right->d_re) / 2.0, out);
}

SSError ss_compose_update(const SSState *left, const SSState *right,
                          SSState *left_out, SSState *right_out) {
    SSError error;
    if (!ss_state_legacy_valid(left) || !ss_state_legacy_valid(right) ||
        left_out == NULL || right_out == NULL)
        return SS_ERR_INVALID_STATE;
    *left_out = *left;
    *right_out = *right;
    error = ss_apply(right, left_out);
    return error == SS_OK ? ss_apply(left, right_out) : error;
}
