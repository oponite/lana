//! Effect execution and attempt storage, mirroring `runtime/c/effects.c` and
//! `runtime/include/effects.h`.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use lana_bytecode::LanaError;
use lana_vm::value::Value;

use crate::policy::{Decision, PolicyOutcome};
use crate::store::{Store, store_commit, store_get, store_put};

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum EffectStatus {
    Pending = 0,
    Succeeded,
    Failed,
    Cancelled,
}

pub struct EffectAttempt {
    pub schema_version: u32,
    pub attempt_id: u64,
    pub decision_id: u64,
    pub effect_desc: Arc<str>,
    pub target: Option<Arc<str>>,
    pub status: EffectStatus,
    pub start_time: u64,
    pub end_time: u64,
    pub failure_reason: Arc<str>,
}

/// An effect executor, mirroring `LanaEffectExecutor`.
pub type EffectExecutor = Box<dyn Fn(&Value) -> Result<Value, LanaError>>;

fn json_safe(text: Option<&str>) -> bool {
    match text {
        None => true,
        Some(text) => text.bytes().all(|b| b >= 0x20 && b != b'\\' && b != b'"'),
    }
}

fn now_seconds() -> Result<u64, LanaError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|_| LanaError::Io)
}

fn store_attempt(store: &mut Store, attempt: &EffectAttempt, allow_overwrite: bool) -> Result<(), LanaError> {
    if attempt.schema_version != 1
        || attempt.attempt_id == 0
        || attempt.decision_id == 0
        || attempt.effect_desc.is_empty()
        || (attempt.status as u32) > EffectStatus::Cancelled as u32
    {
        return Err(LanaError::Schema);
    }
    if !json_safe(Some(&attempt.effect_desc))
        || !json_safe(attempt.target.as_deref())
        || !json_safe(Some(&attempt.failure_reason))
    {
        return Err(LanaError::Schema);
    }
    let key = format!("effect-attempt/{}/{}", attempt.decision_id, attempt.attempt_id);
    if !allow_overwrite {
        match store_get(store, &key) {
            Ok(_) => return Err(LanaError::Conflict),
            Err(LanaError::NotFound) => {}
            Err(e) => return Err(e),
        }
    }
    let json = format!(
        "{{\"attempt_id\":{},\"decision_id\":{},\"effect\":\"{}\",\"target\":\"{}\",\"end_time\":{},\"failure_reason\":\"{}\",\"schema\":1,\"start_time\":{},\"status\":{}}}",
        attempt.attempt_id,
        attempt.decision_id,
        attempt.effect_desc,
        attempt.target.as_deref().unwrap_or(""),
        attempt.end_time,
        attempt.failure_reason,
        attempt.start_time,
        attempt.status as u32
    );
    store_put(store, &key, &Value::string(Arc::from(json.as_str())))
}

/// Execute a planned effect through the caller-supplied executor, mirroring
/// `lana_effect_execute_planned`.
pub fn effect_execute_planned(
    executor: &EffectExecutor,
    plan: &Value,
) -> Result<Value, LanaError> {
    executor(plan)
}

/// Store a PENDING attempt, mirroring `lana_effect_store_attempt`.
pub fn effect_store_attempt(store: &mut Store, attempt: &EffectAttempt) -> Result<(), LanaError> {
    store_attempt(store, attempt, false)
}

/// Execute an effect attempt end-to-end, mirroring `lana_effect_execute_attempt`.
pub fn effect_execute_attempt(
    store: &mut Store,
    decision: &Decision,
    attempt_id: u64,
    plan: &Value,
    executor: &EffectExecutor,
) -> Result<(Value, EffectAttempt), LanaError> {
    if decision.outcome != PolicyOutcome::Authorize || decision.effect.is_none() || attempt_id == 0 {
        return Err(LanaError::Capability);
    }
    let started = now_seconds()?;
    let pending = EffectAttempt {
        schema_version: 1,
        attempt_id,
        decision_id: decision.decision_id,
        effect_desc: decision.effect.clone().unwrap(),
        target: Some(decision.target.clone()),
        status: EffectStatus::Pending,
        start_time: started,
        end_time: 0,
        failure_reason: Arc::from(""),
    };
    store_attempt(store, &pending, false)?;
    store_commit(store)?;

    let execution = effect_execute_planned(executor, plan);
    let ended = now_seconds()?;
    let execution_error = execution.as_ref().err().copied();
    let attempt = EffectAttempt {
        schema_version: 1,
        attempt_id,
        decision_id: decision.decision_id,
        effect_desc: decision.effect.clone().unwrap(),
        target: Some(decision.target.clone()),
        status: if execution_error.is_none() { EffectStatus::Succeeded } else { EffectStatus::Failed },
        start_time: started,
        end_time: ended,
        failure_reason: match execution_error {
            Some(e) => Arc::from(e.name()),
            None => Arc::from(""),
        },
    };
    store_attempt(store, &attempt, true)?;
    store_commit(store)?;

    match execution {
        Ok(value) => Ok((value, attempt)),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_attempt_rejects_duplicate() {
        let dir = std::env::temp_dir().join(format!("lana_effects_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.to_str().unwrap().to_string();
        let mut store = crate::store::store_open(&crate::store::StoreOptions {
            schema_version: 1,
            path: path.clone(),
            timeout_ms: 0,
        })
        .unwrap();
        let attempt = EffectAttempt {
            schema_version: 1,
            attempt_id: 1,
            decision_id: 1,
            effect_desc: Arc::from("grant"),
            target: None,
            status: EffectStatus::Pending,
            start_time: 0,
            end_time: 0,
            failure_reason: Arc::from(""),
        };
        effect_store_attempt(&mut store, &attempt).unwrap();
        crate::store::store_commit(&mut store).unwrap();
        assert_eq!(effect_store_attempt(&mut store, &attempt).unwrap_err(), LanaError::Conflict);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
