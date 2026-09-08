//! State-dist operations, mirroring the state-dist helpers in `vm/c/vm.c`
//! (`lana_vm_state_dist_*`).
//!
//! The pure evaluator (`support`) lives here; the VM-dependent operations
//! (construction, sampling, estimation, expected probability) are methods on
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
