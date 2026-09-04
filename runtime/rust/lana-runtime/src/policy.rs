//! Policy evaluation and decision storage, mirroring `runtime/c/policy.c` and
//! `runtime/include/policy.h`.

use std::sync::Arc;

use lana_bytecode::LanaError;
use lana_vm::value::{Value, ValueKind};

use crate::codec;
use crate::sha256;
use crate::store::{Store, store_get, store_put};

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PolicyRuleKind {
    ProbabilityAtLeast = 0,
    Equals,
    OrderLessThan,
    Present,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum PolicyOutcome {
    Authorize = 0,
    Refuse,
    RequestMoreEvidence,
}

pub struct PolicyRule {
    pub schema_version: u32,
    pub kind: PolicyRuleKind,
    pub field: Arc<str>,
    pub threshold: f64,
    pub expected: Option<Arc<str>>,
    pub effect: Arc<str>,
}

pub struct Policy {
    pub schema_version: u32,
    pub policy_id: Arc<str>,
    pub rule: PolicyRule,
}

pub struct PolicyEvaluation {
    pub schema_version: u32,
    pub decision_id: u64,
    pub target: Arc<str>,
    pub scope: Arc<str>,
    pub input_revision: u64,
    pub evidence_ids: Option<Arc<str>>,
    pub derivation_ids: Option<Arc<str>>,
    pub relationship_resolution: Option<Arc<str>>,
    pub evaluation_time: u64,
    pub reason: Arc<str>,
    pub requested_evidence: Option<Arc<str>>,
}

pub struct Decision {
    pub schema_version: u32,
    pub decision_id: u64,
    pub policy_id: Arc<str>,
    pub policy_version: [u8; 32],
    pub target: Arc<str>,
    pub scope: Arc<str>,
    pub input_revision: u64,
    pub evidence_ids: Option<Arc<str>>,
    pub derivation_ids: Option<Arc<str>>,
    pub relationship_resolution: Option<Arc<str>>,
    pub outcome: PolicyOutcome,
    pub effect: Option<Arc<str>>,
    pub evaluation_time: u64,
    pub reason: Arc<str>,
    pub requested_evidence: Option<Arc<str>>,
}

fn text_ok(text: Option<&str>) -> bool {
    text.map_or(false, |t| !t.is_empty())
}

fn json_safe(text: Option<&str>) -> bool {
    match text {
        None => true,
        Some(text) => text.bytes().all(|b| b >= 0x20 && b != b'\\' && b != b'"'),
    }
}

fn hex_digest(digest: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

fn validate_policy(policy: &Policy) -> Result<(), LanaError> {
    if policy.schema_version != 1
        || !text_ok(Some(&policy.policy_id))
        || policy.rule.schema_version != 1
        || !text_ok(Some(&policy.rule.field))
        || !text_ok(Some(&policy.rule.effect))
    {
        return Err(LanaError::Schema);
    }
    if (policy.rule.kind == PolicyRuleKind::ProbabilityAtLeast
        || policy.rule.kind == PolicyRuleKind::OrderLessThan)
        && (policy.rule.threshold < 0.0 || policy.rule.threshold > 1.0)
    {
        return Err(LanaError::InvalidProbability);
    }
    if (policy.rule.kind == PolicyRuleKind::Equals && !text_ok(policy.rule.expected.as_deref()))
        || (policy.rule.kind as u32) > PolicyRuleKind::Present as u32
    {
        return Err(LanaError::Schema);
    }
    Ok(())
}

pub fn policy_version(policy: &Policy) -> Result<[u8; 32], LanaError> {
    validate_policy(policy)?;
    let mut canonical = String::new();
    canonical.push_str(&codec::encode_value(&Value::string(policy.policy_id.clone()))?);
    canonical.push_str(&codec::encode_value(&Value::string(policy.rule.field.clone()))?);
    canonical.push_str(&codec::encode_value(&Value::number(policy.rule.kind as u32 as f64))?);
    canonical.push_str(&codec::encode_value(&Value::number(policy.rule.threshold))?);
    canonical.push_str(&codec::encode_value(&Value::string(
        policy.rule.expected.clone().unwrap_or_else(|| Arc::from("")),
    ))?);
    canonical.push_str(&codec::encode_value(&Value::string(policy.rule.effect.clone()))?);
    Ok(sha256::sha256(canonical.as_bytes()))
}

pub fn policy_store(store: &mut Store, policy: &Policy) -> Result<(), LanaError> {
    validate_policy(policy)?;
    let digest = policy_version(policy)?;
    if !json_safe(Some(&policy.policy_id))
        || !json_safe(Some(&policy.rule.field))
        || !json_safe(policy.rule.expected.as_deref())
        || !json_safe(Some(&policy.rule.effect))
    {
        return Err(LanaError::Schema);
    }
    let version = hex_digest(&digest);
    let key = format!("policy/{}/{}", policy.policy_id, version);
    match store_get(store, &key) {
        Ok(_) => return Err(LanaError::Conflict),
        Err(LanaError::NotFound) => {}
        Err(e) => return Err(e),
    }
    let json = format!(
        "{{\"effect\":\"{}\",\"expected\":\"{}\",\"field\":\"{}\",\"kind\":{},\"policy_id\":\"{}\",\"schema\":1,\"threshold\":{},\"version\":\"{}\"}}",
        policy.rule.effect,
        policy.rule.expected.as_deref().unwrap_or(""),
        policy.rule.field,
        policy.rule.kind as u32,
        policy.policy_id,
        crate::codec::format_17g(policy.rule.threshold),
        version
    );
    store_put(store, &key, &Value::string(Arc::from(json.as_str())))
}

fn condition_matches(rule: &PolicyRule, input: &Value) -> bool {
    let map = match &input.kind {
        ValueKind::Map(map) => map,
        _ => return false,
    };
    let map = map.lock().unwrap();
    let value = match map.get(&rule.field) {
        Some(value) => value,
        None => return false,
    };
    match rule.kind {
        PolicyRuleKind::ProbabilityAtLeast => {
            matches!(value.kind, ValueKind::Number(n) if n >= rule.threshold && n <= 1.0)
        }
        PolicyRuleKind::OrderLessThan => {
            matches!(value.kind, ValueKind::Number(n) if n < rule.threshold && n >= 0.0)
        }
        PolicyRuleKind::Equals => match (&value.kind, &rule.expected) {
            (ValueKind::String(s), Some(expected)) => &**s == &**expected,
            _ => false,
        },
        PolicyRuleKind::Present => true,
    }
}

pub fn policy_evaluate(
    policy: &Policy,
    input: &Value,
    evaluation: &PolicyEvaluation,
) -> Result<Decision, LanaError> {
    if evaluation.schema_version != 1
        || !text_ok(Some(&evaluation.target))
        || !text_ok(Some(&evaluation.scope))
        || !text_ok(Some(&evaluation.reason))
    {
        return Err(LanaError::Schema);
    }
    validate_policy(policy)?;
    let policy_version = policy_version(policy)?;
    let matches = condition_matches(&policy.rule, input);
    Ok(Decision {
        schema_version: 1,
        decision_id: evaluation.decision_id,
        policy_id: policy.policy_id.clone(),
        policy_version,
        target: evaluation.target.clone(),
        scope: evaluation.scope.clone(),
        input_revision: evaluation.input_revision,
        evidence_ids: evaluation.evidence_ids.clone(),
        derivation_ids: evaluation.derivation_ids.clone(),
        relationship_resolution: evaluation.relationship_resolution.clone(),
        outcome: if matches { PolicyOutcome::Authorize } else { PolicyOutcome::RequestMoreEvidence },
        effect: if matches { Some(policy.rule.effect.clone()) } else { None },
        evaluation_time: evaluation.evaluation_time,
        reason: evaluation.reason.clone(),
        requested_evidence: evaluation.requested_evidence.clone(),
    })
}

pub fn policy_store_decision(store: &mut Store, decision: &Decision) -> Result<(), LanaError> {
    if decision.schema_version != 1 {
        return Err(LanaError::Schema);
    }
    let key = format!("decision/{}", decision.decision_id);
    match store_get(store, &key) {
        Ok(_) => return Err(LanaError::Conflict),
        Err(LanaError::NotFound) => {}
        Err(e) => return Err(e),
    }
    if !text_ok(Some(&decision.policy_id))
        || !text_ok(Some(&decision.target))
        || !text_ok(Some(&decision.scope))
        || !text_ok(Some(&decision.reason))
        || !json_safe(Some(&decision.policy_id))
        || !json_safe(Some(&decision.target))
        || !json_safe(Some(&decision.scope))
        || !json_safe(Some(&decision.reason))
        || !json_safe(decision.evidence_ids.as_deref())
        || !json_safe(decision.derivation_ids.as_deref())
        || !json_safe(decision.relationship_resolution.as_deref())
        || !json_safe(decision.effect.as_deref())
        || !json_safe(decision.requested_evidence.as_deref())
    {
        return Err(LanaError::Schema);
    }
    let version = hex_digest(&decision.policy_version);
    let json = format!(
        "{{\"decision_id\":{},\"derivation_ids\":\"{}\",\"effect\":\"{}\",\"evaluation_time\":{},\"evidence_ids\":\"{}\",\"input_revision\":{},\"outcome\":{},\"policy_id\":\"{}\",\"policy_version\":\"{}\",\"relationship_resolution\":\"{}\",\"requested_evidence\":\"{}\",\"reason\":\"{}\",\"scope\":\"{}\",\"target\":\"{}\"}}",
        decision.decision_id,
        decision.derivation_ids.as_deref().unwrap_or(""),
        decision.effect.as_deref().unwrap_or(""),
        decision.evaluation_time,
        decision.evidence_ids.as_deref().unwrap_or(""),
        decision.input_revision,
        decision.outcome as u32,
        decision.policy_id,
        version,
        decision.relationship_resolution.as_deref().unwrap_or(""),
        decision.requested_evidence.as_deref().unwrap_or(""),
        decision.reason,
        decision.scope,
        decision.target
    );
    store_put(store, &key, &Value::string(Arc::from(json.as_str())))
}

pub fn policy_replay(
    policy: &Policy,
    input: &Value,
    evaluation: &PolicyEvaluation,
    expected: &Decision,
) -> Result<Decision, LanaError> {
    if expected.schema_version != 1 {
        return Err(LanaError::Schema);
    }
    let replayed = policy_evaluate(policy, input, evaluation)?;
    if replayed.policy_version != expected.policy_version
        || replayed.outcome != expected.outcome
        || replayed.decision_id != expected.decision_id
        || replayed.input_revision != expected.input_revision
    {
        return Err(LanaError::Integrity);
    }
    Ok(replayed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_policy() -> Policy {
        Policy {
            schema_version: 1,
            policy_id: Arc::from("p1"),
            rule: PolicyRule {
                schema_version: 1,
                kind: PolicyRuleKind::ProbabilityAtLeast,
                field: Arc::from("p"),
                threshold: 0.5,
                expected: None,
                effect: Arc::from("grant"),
            },
        }
    }

    #[test]
    fn version_is_stable() {
        let policy = sample_policy();
        let a = policy_version(&policy).unwrap();
        let b = policy_version(&policy).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn evaluate_authorizes_when_above_threshold() {
        let policy = sample_policy();
        let mut map = lana_vm::value::Map::new(1);
        map.set(Arc::from("p"), Value::number(0.75), false).unwrap();
        let input = Value::map(Arc::new(std::sync::Mutex::new(map)));
        let evaluation = PolicyEvaluation {
            schema_version: 1,
            decision_id: 1,
            target: Arc::from("t"),
            scope: Arc::from("s"),
            input_revision: 0,
            evidence_ids: None,
            derivation_ids: None,
            relationship_resolution: None,
            evaluation_time: 0,
            reason: Arc::from("r"),
            requested_evidence: None,
        };
        let decision = policy_evaluate(&policy, &input, &evaluation).unwrap();
        assert_eq!(decision.outcome, PolicyOutcome::Authorize);
        assert_eq!(decision.effect.as_deref(), Some("grant"));
    }
}
