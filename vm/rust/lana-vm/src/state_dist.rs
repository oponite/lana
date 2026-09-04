//! State-dist operations, mirroring the state-dist helpers in `vm/c/vm.c`
//! (`lana_vm_state_dist_*`).
//!
//! The pure evaluators (`expected_probability`, `support`) live here; the
//! VM-dependent operations (construction, sampling, estimation) are methods on
//! `Vm` in `vm.rs`.

use std::cmp::Ordering;
use std::sync::Arc;

use lana_bytecode::LanaError;

use crate::state::{self, State, StateValue};
use crate::value::{DistOperand, StateDist, StateDistKind, Value, ValueKind};

/// The maximum state-dist evaluation depth, matching
/// `LANA_STATE_DIST_DEPTH_LIMIT` in `vm/include/value.h`.
pub const LANA_STATE_DIST_DEPTH_LIMIT: usize = 1024;

/// One frame of the iterative state-dist evaluators, mirroring
/// `LanaDistEvalFrame` in `vm/c/vm.c`.
pub(crate) struct DistEvalFrame {
    pub node: Arc<StateDist>,
    pub stage: u32,
    pub left: f64,
    pub right: f64,
    pub left_state: StateValue,
    pub right_state: StateValue,
}

impl DistEvalFrame {
    pub fn new(node: Arc<StateDist>) -> Self {
        Self {
            node,
            stage: 0,
            left: 0.0,
            right: 0.0,
            left_state: StateValue::default(),
            right_state: StateValue::default(),
        }
    }
}

/// The action an evaluator frame takes after inspecting its node, used to
/// avoid holding a borrow of the stack across a push or pop.
pub(crate) enum EvalAction {
    Pop,
    Push(Arc<StateDist>),
    Continue,
}

/// Convert a value into a distribution operand, mirroring
/// `distribution_from_value`.
pub fn distribution_from_value(value: &Value) -> Result<DistOperand, LanaError> {
    match &value.kind {
        ValueKind::State(state) => Ok(DistOperand::Inline(state.clone())),
        ValueKind::StateDist(distribution) => Ok(DistOperand::Node(distribution.clone())),
        _ => Err(LanaError::Type),
    }
}

/// The expected probability of a state distribution, mirroring
/// `lana_vm_state_dist_expected_probability`. Iterative to bound the stack.
pub fn expected_probability(distribution: &Arc<StateDist>) -> Result<f64, LanaError> {
    let mut stack: Vec<DistEvalFrame> = Vec::new();
    let mut result = 0.0;
    stack.push(DistEvalFrame::new(distribution.clone()));
    loop {
        if stack.is_empty() {
            break;
        }
        let top = stack.len() - 1;
        let stage = stack[top].stage;
        if stage == 0 {
            let action = {
                let frame = &mut stack[top];
                match &frame.node.kind {
                    StateDistKind::Dirac(state) => {
                        if !state::state_valid(&state.state) {
                            return Err(LanaError::InvalidDistribution);
                        }
                        result = state.state.p;
                        EvalAction::Pop
                    }
                    StateDistKind::Append { left, has_cached_parameters, p, .. } => {
                        if *has_cached_parameters {
                            result = *p;
                            EvalAction::Pop
                        } else if let DistOperand::Inline(state) = left {
                            if !state::state_valid(&state.state) {
                                return Err(LanaError::InvalidDistribution);
                            }
                            frame.left = state.state.p;
                            frame.stage = 2;
                            EvalAction::Continue
                        } else {
                            let DistOperand::Node(node) = left else {
                                return Err(LanaError::InvalidDistribution);
                            };
                            frame.stage = 1;
                            EvalAction::Push(node.clone())
                        }
                    }
                    StateDistKind::Transform { child, .. } => {
                        frame.stage = 4;
                        EvalAction::Push(child.clone())
                    }
                    StateDistKind::Attenuate { child, .. } => {
                        frame.stage = 5;
                        EvalAction::Push(child.clone())
                    }
                }
            };
            match action {
                EvalAction::Pop => {
                    stack.pop();
                }
                EvalAction::Push(node) => {
                    if stack.len() >= LANA_STATE_DIST_DEPTH_LIMIT + 1 {
                        return Err(LanaError::InvalidDistribution);
                    }
                    stack.push(DistEvalFrame::new(node));
                }
                EvalAction::Continue => {}
            }
        } else if stage == 1 {
            stack[top].left = result;
            stack[top].stage = 2;
        } else if stage == 2 {
            let action = {
                let frame = &mut stack[top];
                let right = match &frame.node.kind {
                    StateDistKind::Append { right, .. } => right,
                    _ => return Err(LanaError::InvalidDistribution),
                };
                if let DistOperand::Inline(state) = right {
                    if !state::state_valid(&state.state) {
                        return Err(LanaError::InvalidDistribution);
                    }
                    frame.right = state.state.p;
                    frame.stage = 3;
                    EvalAction::Continue
                } else {
                    let DistOperand::Node(node) = right else {
                        return Err(LanaError::InvalidDistribution);
                    };
                    frame.stage = 3;
                    EvalAction::Push(node.clone())
                }
            };
            match action {
                EvalAction::Pop => unreachable!("stage 2 never pops"),
                EvalAction::Push(node) => {
                    if stack.len() >= LANA_STATE_DIST_DEPTH_LIMIT + 1 {
                        return Err(LanaError::InvalidDistribution);
                    }
                    stack.push(DistEvalFrame::new(node));
                }
                EvalAction::Continue => {}
            }
        } else if stage == 3 {
            let right_is_node = matches!(
                &stack[top].node.kind,
                StateDistKind::Append { right: DistOperand::Node(_), .. }
            );
            if right_is_node {
                stack[top].right = result;
            }
            result = 1.0 - (1.0 - stack[top].left) * (1.0 - stack[top].right);
            if !result.is_finite() || result < 0.0 || result > 1.0 {
                return Err(LanaError::InvalidDistribution);
            }
            stack.pop();
        } else if stage == 4 {
            let transform_id = match &stack[top].node.kind {
                StateDistKind::Transform { transform_id, .. } => *transform_id,
                _ => return Err(LanaError::InvalidDistribution),
            };
            let mut out = 0.0;
            let error = state::transform_expected_probability(transform_id, result, &mut out);
            if error != LanaError::Ok {
                return Err(error);
            }
            result = out;
            stack.pop();
        } else {
            // ATTENUATE is the identity on probability: the child's result is
            // already the expected probability.
            stack.pop();
        }
    }
    Ok(result)
}

/// The support of a state distribution, mirroring
/// `lana_vm_state_dist_support`. Returns the sorted list of states.
pub fn support(distribution: &Arc<StateDist>, limit: u32) -> Result<Vec<StateValue>, LanaError> {
    if limit == 0 {
        return Err(LanaError::InvalidDistribution);
    }
    let mut items: Vec<StateValue> = Vec::new();
    support_collect(distribution, &mut items, limit)?;
    items.sort_by(support_compare);
    Ok(items)
}

/// Sort by `(p, d_re, d_im)`, matching `support_compare` in `vm/c/vm.c`.
fn support_compare(a: &StateValue, b: &StateValue) -> Ordering {
    a.state
        .p
        .partial_cmp(&b.state.p)
        .unwrap_or(Ordering::Equal)
        .then(a.state.d_re.partial_cmp(&b.state.d_re).unwrap_or(Ordering::Equal))
        .then(a.state.d_im.partial_cmp(&b.state.d_im).unwrap_or(Ordering::Equal))
}

/// Collect the support of a node into `items`, mirroring `support_collect`.
/// The C11 resets the depth on each recursive call, so the depth check is
/// effectively dead; the recursion is bounded by the tree depth.
fn support_collect(node: &Arc<StateDist>, items: &mut Vec<StateValue>, limit: u32) -> Result<(), LanaError> {
    match &node.kind {
        StateDistKind::Dirac(state) => {
            if !state::state_valid(&state.state) {
                return Err(LanaError::InvalidDistribution);
            }
            if items.len() >= limit as usize {
                return Err(LanaError::Limit);
            }
            items.push(state.clone());
            Ok(())
        }
        StateDistKind::Transform { child, transform_id } => {
            let child_support = support(child, limit)?;
            for child_state in child_support {
                if items.len() >= limit as usize {
                    return Err(LanaError::Limit);
                }
                let mut transformed = StateValue {
                    state: child_state.state,
                    indexes: Default::default(),
                };
                let mut state = transformed.state;
                let source = state;
                let error = state::transform_apply(*transform_id, &source, &mut state);
                if error != LanaError::Ok {
                    return Err(error);
                }
                transformed.state = state;
                items.push(transformed);
            }
            Ok(())
        }
        StateDistKind::Attenuate { child, factor } => {
            let child_support = support(child, limit)?;
            for child_state in child_support {
                if items.len() >= limit as usize {
                    return Err(LanaError::Limit);
                }
                let mut attenuated = StateValue {
                    state: child_state.state,
                    indexes: Default::default(),
                };
                let mut state = attenuated.state;
                let source = state;
                let error = state::attenuate(&source, *factor, &mut state);
                if error != LanaError::Ok {
                    return Err(error);
                }
                attenuated.state = state;
                items.push(attenuated);
            }
            Ok(())
        }
        StateDistKind::Append { has_cached_parameters, sigma, p, m_re, m_im, .. } => {
            if *has_cached_parameters && *sigma == 0.0 {
                if items.len() >= limit as usize {
                    return Err(LanaError::Limit);
                }
                let mut result = StateValue {
                    state: State { p: 0.0, d_re: 0.0, d_im: 0.0 },
                    indexes: Default::default(),
                };
                let error = state::make_complex(*p, *m_re, *m_im, &mut result.state);
                if error != LanaError::Ok {
                    return Err(LanaError::InvalidDistribution);
                }
                items.push(result);
                Ok(())
            } else {
                Err(LanaError::UnsupportedOperation)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::StateDist;

    fn dirac(p: f64) -> Arc<StateDist> {
        let mut state = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(state::make_complex(p, 0.0, 0.0, &mut state), LanaError::Ok);
        Arc::new(StateDist {
            kind: StateDistKind::Dirac(StateValue { state, indexes: Default::default() }),
        })
    }

    #[test]
    fn expected_probability_of_dirac() {
        assert_eq!(expected_probability(&dirac(0.4)).unwrap(), 0.4);
    }

    #[test]
    fn expected_probability_of_append() {
        let left = dirac(0.2);
        let right = dirac(0.3);
        let append = Arc::new(StateDist {
            kind: StateDistKind::Append {
                left: DistOperand::Node(left),
                right: DistOperand::Node(right),
                has_cached_parameters: false,
                p: 0.0,
                m_re: 0.0,
                m_im: 0.0,
                sigma: 0.0,
            },
        });
        // 1 - (1 - 0.2) * (1 - 0.3) = 0.44 (within floating-point tolerance)
        let expected = expected_probability(&append).unwrap();
        assert!((expected - 0.44).abs() < 1e-12, "expected ~0.44, got {expected}");
    }

    #[test]
    fn expected_probability_of_transform() {
        let child = dirac(0.3);
        let transform = Arc::new(StateDist {
            kind: StateDistKind::Transform { child, transform_id: 0 },
        });
        // invert: 1 - 0.3 = 0.7
        assert_eq!(expected_probability(&transform).unwrap(), 0.7);
    }

    #[test]
    fn support_of_dirac() {
        let items = support(&dirac(0.4), 10).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].state.p, 0.4);
    }

    #[test]
    fn support_of_append_with_zero_sigma() {
        let left = dirac(0.2);
        let right = dirac(0.3);
        let append = Arc::new(StateDist {
            kind: StateDistKind::Append {
                left: DistOperand::Node(left),
                right: DistOperand::Node(right),
                has_cached_parameters: true,
                p: 0.44,
                m_re: 0.0,
                m_im: 0.0,
                sigma: 0.0,
            },
        });
        let items = support(&append, 10).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].state.p, 0.44);
    }

    #[test]
    fn support_of_append_with_sigma_is_unsupported() {
        let left = dirac(0.2);
        let right = dirac(0.3);
        let append = Arc::new(StateDist {
            kind: StateDistKind::Append {
                left: DistOperand::Node(left),
                right: DistOperand::Node(right),
                has_cached_parameters: true,
                p: 0.44,
                m_re: 0.0,
                m_im: 0.0,
                sigma: 0.5,
            },
        });
        assert_eq!(support(&append, 10), Err(LanaError::UnsupportedOperation));
    }
}
