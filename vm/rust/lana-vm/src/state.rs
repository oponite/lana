//! State math, mirroring `vm/c/state.c` and `vm/include/state.h`.
//!
//! The numeric behavior (epsilon clamping, `positive_zero`, radius
//! normalization) is preserved exactly so the Rust VM produces identical
//! results to the C11 reference.

use lana_bytecode::LanaError;

pub const STATE_EPSILON: f64 = 1e-12;

/// A concrete state: probability `p` and a disposition vector `(d_re, d_im)`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct State {
    pub p: f64,
    pub d_re: f64,
    pub d_im: f64,
}

/// Indexes attached to a state value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Indexes {
    pub has_timestamp: bool,
    pub has_source: bool,
    pub has_weight: bool,
    pub has_confidence: bool,
    pub timestamp: f64,
    pub weight: f64,
    pub confidence: f64,
    pub source: Option<std::sync::Arc<str>>,
}

/// A state value: the state plus its indexes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StateValue {
    pub state: State,
    pub indexes: Indexes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BasisId {
    Computational = 0,
    X = 1,
    Y = 2,
}

fn positive_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

pub fn disposition_squared(state: &State) -> f64 {
    state.d_re * state.d_re + state.d_im * state.d_im
}

pub fn state_valid(state: &State) -> bool {
    if !state.p.is_finite() || !state.d_re.is_finite() || !state.d_im.is_finite()
        || state.p < 0.0 || state.p > 1.0
    {
        return false;
    }
    let radius_squared = disposition_squared(state);
    radius_squared.is_finite()
        && radius_squared <= 1.0
        && ((state.p != 0.0 && state.p != 1.0) || (state.d_re == 0.0 && state.d_im == 0.0))
}

pub fn make_complex(p: f64, d_re: f64, d_im: f64, out: &mut State) -> LanaError {
    if !p.is_finite() || !d_re.is_finite() || !d_im.is_finite() {
        return LanaError::InvalidState;
    }
    if p < -STATE_EPSILON || p > 1.0 + STATE_EPSILON {
        return LanaError::InvalidState;
    }
    let mut p = p;
    let mut d_re = d_re;
    let mut d_im = d_im;
    if p < 0.0 {
        p = 0.0;
    }
    if p > 1.0 {
        p = 1.0;
    }
    let radius_squared = d_re * d_re + d_im * d_im;
    let radius = radius_squared.sqrt();
    if !radius.is_finite() || radius > 1.0 + STATE_EPSILON {
        return LanaError::InvalidState;
    }
    if radius > 1.0 {
        d_re /= radius;
        d_im /= radius;
    }
    if p == 0.0 || p == 1.0 {
        d_re = 0.0;
        d_im = 0.0;
    }
    *out = State {
        p: positive_zero(p),
        d_re: positive_zero(d_re),
        d_im: positive_zero(d_im),
    };
    if state_valid(out) { LanaError::Ok } else { LanaError::InvalidState }
}

pub fn make(p: f64, d: f64, out: &mut State) -> LanaError {
    make_complex(p, d, 0.0, out)
}

pub fn reconstruct_c(state: &State, c_re: &mut f64, c_im: &mut f64) {
    let mut scale = 0.0;
    if state_valid(state) {
        scale = (state.p * (1.0 - state.p)).sqrt();
    }
    *c_re = positive_zero(state.d_re * scale);
    *c_im = positive_zero(state.d_im * scale);
}

pub fn basis_probability(basis: u32, state: &State, out: &mut f64) -> LanaError {
    if !state_valid(state) {
        return LanaError::InvalidState;
    }
    let mut c_re = 0.0;
    let mut c_im = 0.0;
    reconstruct_c(state, &mut c_re, &mut c_im);
    let probability = match basis {
        0 => state.p,
        1 => 0.5 - c_re,
        2 => 0.5 + c_im,
        _ => return LanaError::Measure,
    };
    if !probability.is_finite() || probability < -STATE_EPSILON || probability > 1.0 + STATE_EPSILON {
        return LanaError::InvalidState;
    }
    let mut probability = probability;
    if probability < 0.0 {
        probability = 0.0;
    }
    if probability > 1.0 {
        probability = 1.0;
    }
    *out = positive_zero(probability);
    LanaError::Ok
}

pub fn append_parameters(
    left: &State,
    right: &State,
    p: &mut f64,
    m_re: &mut f64,
    m_im: &mut f64,
    sigma: &mut f64,
) -> LanaError {
    if !state_valid(left) || !state_valid(right) {
        return LanaError::InvalidState;
    }
    *p = 1.0 - (1.0 - left.p) * (1.0 - right.p);
    *m_re = (left.d_re + right.d_re) / 2.0;
    *m_im = (left.d_im + right.d_im) / 2.0;
    let delta_re = left.d_re - right.d_re;
    let delta_im = left.d_im - right.d_im;
    *sigma = delta_re.hypot(delta_im) / 2.0;
    LanaError::Ok
}

fn transform_invert_current(source: &State, out: &mut State) -> LanaError {
    make_complex(1.0 - source.p, source.d_re, -source.d_im, out)
}

fn transform_neutralize_current(source: &State, out: &mut State) -> LanaError {
    make_complex(source.p, 0.0, 0.0, out)
}

fn expectation_invert(probability: f64) -> f64 {
    1.0 - probability
}

fn expectation_identity(probability: f64) -> f64 {
    probability
}

/// A registered concrete transform, mirroring `LanaTransformSpec`.
pub struct TransformSpec {
    pub identifier: u32,
    pub name: &'static str,
    pub concrete: fn(&State, &mut State) -> LanaError,
    pub exact_expected_probability: fn(f64) -> f64,
    pub distribution_liftable: bool,
}

const TRANSFORM_SPECS: [TransformSpec; 2] = [
    TransformSpec {
        identifier: 0,
        name: "invert",
        concrete: transform_invert_current,
        exact_expected_probability: expectation_invert,
        distribution_liftable: true,
    },
    TransformSpec {
        identifier: 1,
        name: "neutralize",
        concrete: transform_neutralize_current,
        exact_expected_probability: expectation_identity,
        distribution_liftable: true,
    },
];

pub fn transform_spec(identifier: u32) -> Option<&'static TransformSpec> {
    TRANSFORM_SPECS.get(identifier as usize)
}

pub fn transform_apply(identifier: u32, source: &State, out: &mut State) -> LanaError {
    let specification = match transform_spec(identifier) {
        Some(spec) => spec,
        None => return LanaError::UnsupportedOperation,
    };
    if !state_valid(source) {
        return LanaError::InvalidState;
    }
    let error = (specification.concrete)(source, out);
    if error != LanaError::Ok || !state_valid(out) {
        return LanaError::InvalidTransformResult;
    }
    LanaError::Ok
}

pub fn transform_expected_probability(identifier: u32, input: f64, out: &mut f64) -> LanaError {
    let specification = match transform_spec(identifier) {
        Some(spec) => spec,
        None => return LanaError::UnsupportedExactMeasurement,
    };
    if !specification.distribution_liftable {
        return LanaError::UnsupportedExactMeasurement;
    }
    if !input.is_finite() || input < 0.0 || input > 1.0 {
        return LanaError::InvalidDistribution;
    }
    *out = (specification.exact_expected_probability)(input);
    LanaError::Ok
}

pub fn mix(left: &State, right: &State, w: f64, out: &mut State) -> LanaError {
    if !state_valid(left) || !state_valid(right) {
        return LanaError::InvalidState;
    }
    if !w.is_finite() || w < 0.0 || w > 1.0 {
        return LanaError::InvalidParameters;
    }
    let p = w * left.p + (1.0 - w) * right.p;
    let mut c_left_re = 0.0;
    let mut c_left_im = 0.0;
    let mut c_right_re = 0.0;
    let mut c_right_im = 0.0;
    reconstruct_c(left, &mut c_left_re, &mut c_left_im);
    reconstruct_c(right, &mut c_right_re, &mut c_right_im);
    let c_re = w * c_left_re + (1.0 - w) * c_right_re;
    let c_im = w * c_left_im + (1.0 - w) * c_right_im;
    if p == 0.0 || p == 1.0 {
        return make_complex(p, 0.0, 0.0, out);
    }
    let scale = (p * (1.0 - p)).sqrt();
    make_complex(p, c_re / scale, c_im / scale, out)
}

/// The relationship modes for relationship-aware append, matching
/// `LanaAppendRelationshipId` in `vm/include/bytecode.h`.
pub const APPEND_INDEPENDENT: u32 = 0;
pub const APPEND_REDUNDANT: u32 = 1;
pub const APPEND_FULL_REDUNDANCY: u32 = 2;
pub const APPEND_COMPLEMENTARY: u32 = 3;

pub fn attenuate(source: &State, factor: f64, out: &mut State) -> LanaError {
    if !state_valid(source) {
        return LanaError::InvalidState;
    }
    if !factor.is_finite() || factor < 0.0 || factor > 1.0 {
        return LanaError::InvalidParameters;
    }
    make_complex(source.p, factor * source.d_re, factor * source.d_im, out)
}

pub fn trace_distance(left: &State, right: &State, out: &mut f64) -> LanaError {
    if !state_valid(left) || !state_valid(right) {
        return LanaError::InvalidState;
    }
    let mut c_left_re = 0.0;
    let mut c_left_im = 0.0;
    let mut c_right_re = 0.0;
    let mut c_right_im = 0.0;
    reconstruct_c(left, &mut c_left_re, &mut c_left_im);
    reconstruct_c(right, &mut c_right_re, &mut c_right_im);
    let dp = left.p - right.p;
    let dc_re = c_left_re - c_right_re;
    let dc_im = c_left_im - c_right_im;
    *out = (dp * dp + dc_re * dc_re + dc_im * dc_im).sqrt();
    LanaError::Ok
}

fn append_overlap(mode: u32, p_a: f64, p_b: f64, strength: f64) -> Result<f64, LanaError> {
    let independent = p_a * p_b;
    match mode {
        APPEND_INDEPENDENT => Ok(independent),
        APPEND_REDUNDANT => {
            if !strength.is_finite() || strength < 0.0 || strength >= 1.0 {
                return Err(LanaError::InvalidParameters);
            }
            let bound = p_a.min(p_b);
            Ok((1.0 - strength) * independent + strength * bound)
        }
        APPEND_FULL_REDUNDANCY => {
            if p_a != p_b {
                return Err(LanaError::InvalidParameters);
            }
            Ok(p_a)
        }
        APPEND_COMPLEMENTARY => {
            if !strength.is_finite() || strength < 0.0 || strength > 1.0 {
                return Err(LanaError::InvalidParameters);
            }
            let bound = (p_a + p_b - 1.0).max(0.0);
            Ok((1.0 - strength) * independent + strength * bound)
        }
        _ => Err(LanaError::UnsupportedOperation),
    }
}

pub fn append_relationship_parameters(
    left: &State,
    right: &State,
    mode: u32,
    strength: f64,
    p: &mut f64,
    m_re: &mut f64,
    m_im: &mut f64,
    sigma: &mut f64,
) -> LanaError {
    if !state_valid(left) || !state_valid(right) {
        return LanaError::InvalidState;
    }
    let q = match append_overlap(mode, left.p, right.p, strength) {
        Ok(q) => q,
        Err(error) => return error,
    };
    *p = left.p + right.p - q;
    *m_re = (left.d_re + right.d_re) / 2.0;
    *m_im = (left.d_im + right.d_im) / 2.0;
    let delta_re = left.d_re - right.d_re;
    let delta_im = left.d_im - right.d_im;
    *sigma = delta_re.hypot(delta_im) / 2.0;
    LanaError::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_complex_clamps_and_normalizes() {
        let mut state = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(make_complex(1.5, 0.0, 0.0, &mut state), LanaError::InvalidState);
        // Radius beyond 1 + epsilon is rejected, matching `lana_state_make_complex`.
        assert_eq!(make_complex(0.5, 2.0, 0.0, &mut state), LanaError::InvalidState);
        // Radius within epsilon of 1 is normalized to the unit circle.
        assert_eq!(make_complex(0.5, 1.0 + 1e-13, 0.0, &mut state), LanaError::Ok);
        assert!((state.d_re * state.d_re + state.d_im * state.d_im - 1.0).abs() < 1e-12);
        assert_eq!(make_complex(0.0, 0.3, 0.4, &mut state), LanaError::Ok);
        assert_eq!(state.d_re, 0.0);
        assert_eq!(state.d_im, 0.0);
    }

    #[test]
    fn basis_probability_matches_c11() {
        let mut state = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(make_complex(0.5, 0.0, 0.0, &mut state), LanaError::Ok);
        let mut out = 0.0;
        assert_eq!(basis_probability(0, &state, &mut out), LanaError::Ok);
        assert_eq!(out, 0.5);
        assert_eq!(basis_probability(1, &state, &mut out), LanaError::Ok);
        assert_eq!(out, 0.5);
        assert_eq!(basis_probability(2, &state, &mut out), LanaError::Ok);
        assert_eq!(out, 0.5);
        assert_eq!(basis_probability(3, &state, &mut out), LanaError::Measure);
    }

    #[test]
    fn mix_is_weight_symmetric() {
        let mut left = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        let mut right = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(make_complex(0.2, 0.0, 0.0, &mut left), LanaError::Ok);
        assert_eq!(make_complex(0.8, 0.0, 0.0, &mut right), LanaError::Ok);
        let mut a = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        let mut b = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(mix(&left, &right, 0.3, &mut a), LanaError::Ok);
        assert_eq!(mix(&right, &left, 0.7, &mut b), LanaError::Ok);
        // The two orderings compute the same products; `1.0 - 0.7` is not exactly
        // `0.3`, so the sums agree only within floating-point tolerance.
        assert!((a.p - b.p).abs() < 1e-12);
        assert!((a.d_re - b.d_re).abs() < 1e-12);
        assert!((a.d_im - b.d_im).abs() < 1e-12);
    }

    #[test]
    fn mix_rejects_bad_weight() {
        let mut left = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        let mut right = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(make_complex(0.5, 0.0, 0.0, &mut left), LanaError::Ok);
        assert_eq!(make_complex(0.5, 0.0, 0.0, &mut right), LanaError::Ok);
        let mut out = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(mix(&left, &right, 1.5, &mut out), LanaError::InvalidParameters);
        assert_eq!(mix(&left, &right, f64::NAN, &mut out), LanaError::InvalidParameters);
    }

    #[test]
    fn transform_apply_matches_c11() {
        let mut source = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(make_complex(0.3, 0.4, 0.0, &mut source), LanaError::Ok);
        let mut out = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(transform_apply(0, &source, &mut out), LanaError::Ok);
        assert_eq!(out.p, 0.7);
        assert_eq!(out.d_im, -0.0);
        assert_eq!(transform_apply(1, &source, &mut out), LanaError::Ok);
        assert_eq!(out.d_re, 0.0);
        assert_eq!(transform_apply(9, &source, &mut out), LanaError::UnsupportedOperation);
    }

    #[test]
    fn attenuate_scales_disposition_only() {
        let mut source = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(make_complex(0.5, 0.6, 0.0, &mut source), LanaError::Ok);
        let mut out = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(attenuate(&source, 0.5, &mut out), LanaError::Ok);
        assert_eq!(out.p, 0.5);
        assert!((out.d_re - 0.3).abs() < 1e-12);
        assert_eq!(out.d_im, 0.0);
        assert_eq!(attenuate(&source, 1.5, &mut out), LanaError::InvalidParameters);
        assert_eq!(attenuate(&source, f64::NAN, &mut out), LanaError::InvalidParameters);
    }

    #[test]
    fn trace_distance_matches_n1_formula() {
        let mut left = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        let mut right = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(make_complex(0.5, 0.6, 0.0, &mut left), LanaError::Ok);
        assert_eq!(make_complex(0.5, 0.4, 0.0, &mut right), LanaError::Ok);
        let mut out = 0.0;
        assert_eq!(trace_distance(&left, &right, &mut out), LanaError::Ok);
        assert!((out - 0.1).abs() < 1e-12);
    }

    #[test]
    fn append_relationship_parameters_match_modes() {
        let mut left = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        let mut right = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(make_complex(0.4, 0.2, 0.0, &mut left), LanaError::Ok);
        assert_eq!(make_complex(0.8, -0.1, 0.0, &mut right), LanaError::Ok);
        let mut p = 0.0;
        let mut m_re = 0.0;
        let mut m_im = 0.0;
        let mut sigma = 0.0;
        // independent: p_C = 0.4 + 0.8 - 0.32 = 0.88.
        assert_eq!(
            append_relationship_parameters(&left, &right, APPEND_INDEPENDENT, 0.0, &mut p, &mut m_re, &mut m_im, &mut sigma),
            LanaError::Ok
        );
        assert!((p - 0.88).abs() < 1e-12);
        // redundant r=0.8: q = 0.2*0.32 + 0.8*0.4 = 0.384, p_C = 0.816.
        assert_eq!(
            append_relationship_parameters(&left, &right, APPEND_REDUNDANT, 0.8, &mut p, &mut m_re, &mut m_im, &mut sigma),
            LanaError::Ok
        );
        assert!((p - 0.816).abs() < 1e-12);
        // complementary k=0.3: q = 0.7*0.32 + 0.3*0.2 = 0.284, p_C = 0.916.
        assert_eq!(
            append_relationship_parameters(&left, &right, APPEND_COMPLEMENTARY, 0.3, &mut p, &mut m_re, &mut m_im, &mut sigma),
            LanaError::Ok
        );
        assert!((p - 0.916).abs() < 1e-12);
        // full_redundancy requires p_a == p_b.
        assert_eq!(
            append_relationship_parameters(&left, &right, APPEND_FULL_REDUNDANCY, 0.0, &mut p, &mut m_re, &mut m_im, &mut sigma),
            LanaError::InvalidParameters
        );
        // redundant strength must be in [0,1).
        assert_eq!(
            append_relationship_parameters(&left, &right, APPEND_REDUNDANT, 1.0, &mut p, &mut m_re, &mut m_im, &mut sigma),
            LanaError::InvalidParameters
        );
    }
}
