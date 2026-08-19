#include "ss/state.h"

#include <math.h>

static double clip_probability(double value) {
    if (value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

bool ss_state_valid(const SSState *state) {
    return state != NULL && isfinite(state->p) && isfinite(state->d) &&
           state->p >= 0.0 && state->p <= 1.0 &&
           state->d > -1.0 && state->d < 1.0;
}

SSError ss_state_make(double p, double d, SSState *out) {
    if (out == NULL) return SS_ERR_INVALID_STATE;
    if (!isfinite(p) || p < 0.0 || p > 1.0) return SS_ERR_INVALID_PROBABILITY;
    if (!isfinite(d) || d <= -1.0 || d >= 1.0) return SS_ERR_INVALID_DEPENDENCY;
    out->p = p;
    out->d = d;
    return SS_OK;
}

SSError ss_apply(const SSState *source, SSState *target) {
    double influence_target;
    double strength;
    if (!ss_state_valid(source) || !ss_state_valid(target)) return SS_ERR_INVALID_STATE;
    influence_target = source->d < 0.0 ? 1.0 - source->p : source->p;
    strength = fabs(source->d);
    target->p = clip_probability(target->p + strength * (influence_target - target->p));
    return ss_state_valid(target) ? SS_OK : SS_ERR_INVALID_STATE;
}

SSError ss_compose_merge(const SSState *left, const SSState *right, SSState *out) {
    if (!ss_state_valid(left) || !ss_state_valid(right) || out == NULL)
        return SS_ERR_INVALID_STATE;
    return ss_state_make((left->p + right->p) / 2.0, (left->d + right->d) / 2.0, out);
}

SSError ss_compose_update(const SSState *left, const SSState *right,
                          SSState *left_out, SSState *right_out) {
    SSError error;
    if (!ss_state_valid(left) || !ss_state_valid(right) || left_out == NULL || right_out == NULL)
        return SS_ERR_INVALID_STATE;
    *left_out = *left;
    *right_out = *right;
    error = ss_apply(right, left_out);
    return error == SS_OK ? ss_apply(left, right_out) : error;
}
