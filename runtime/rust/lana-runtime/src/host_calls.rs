//! Durable-pipeline host calls, exposing the store, policy engine, and event
//! ledger to Lana bytecode.
//!
//! The VM owns a single store (Option A: one store per VM). The store lives
//! here, in the runtime layer, because `lana-vm` cannot depend on
//! `lana-runtime`; the CLI registers a `StoreHost` as the VM's host-call
//! extension, and the VM delegates unknown host-call IDs to it.
//!
//! Store values cross the boundary by serialization: `store_put` encodes the
//! value to JSON immediately and `store_get` decodes a fresh value, so no
//! aliasing between the VM's value graph and the store is possible.

use std::sync::{Arc, Mutex};

use lana_bytecode::LanaError;
use lana_vm::value::{Array, Map, Value, ValueKind};
use lana_vm::{
    LANA_HOST_ADAPTER_FETCH, LANA_HOST_ADAPTER_LOAD, LANA_HOST_LEDGER_APPEND,
    LANA_HOST_LEDGER_QUERY, LANA_HOST_POLICY_EVALUATE, LANA_HOST_POLICY_STORE_DECISION,
    LANA_HOST_STORE_COMMIT, LANA_HOST_STORE_COMMIT_IF, LANA_HOST_STORE_CURRENT_REVISION,
    LANA_HOST_STORE_DELETE, LANA_HOST_STORE_GET, LANA_HOST_STORE_GET_AT, LANA_HOST_STORE_OPEN,
    LANA_HOST_STORE_PUT, LANA_HOST_STORE_SCAN, LANA_HOST_STORE_SNAPSHOT,
};

use crate::adapters::{self, Adapter, AdapterKind, AdapterOptions};
use crate::ledger::{self, Event, EventInput, LedgerQuery};
use crate::policy::{self, Decision, Policy, PolicyEvaluation, PolicyOutcome, PolicyRule, PolicyRuleKind};
use crate::store::{self, Store, StoreOptions};

/// Owns the single durable store the VM drives, and dispatches the
/// store/policy/ledger host calls against it.
pub struct StoreHost {
    store: Option<Store>,
    adapter: Option<Adapter>,
    heap: lana_vm::heap::Heap,
}

impl StoreHost {
    pub fn new() -> Self {
        Self::with_heap(lana_vm::heap::Heap::new(256 * 1024 * 1024))
    }

    pub fn with_heap(heap: lana_vm::heap::Heap) -> Self {
        Self { store: None, adapter: None, heap }
    }

    /// Dispatch one durable-pipeline host call. Returns `LanaError::Format` for
    /// an ID outside the store/policy/ledger range (the VM's own host calls are
    /// handled before the extension is consulted).
    pub fn dispatch(&mut self, host_id: u32, args: &[Value], out: &mut Value) -> LanaError {
        *out = Value::null();
        match host_id {
            LANA_HOST_STORE_OPEN => self.store_open(args, out),
            LANA_HOST_STORE_PUT => self.store_put(args, out),
            LANA_HOST_STORE_GET => self.store_get(args, out),
            LANA_HOST_STORE_DELETE => self.store_delete(args, out),
            LANA_HOST_STORE_COMMIT => self.store_commit(args, out),
            LANA_HOST_STORE_SCAN => self.store_scan(args, out),
            LANA_HOST_STORE_CURRENT_REVISION => self.store_current_revision(args, out),
            LANA_HOST_STORE_GET_AT => self.store_get_at(args, out),
            LANA_HOST_STORE_SNAPSHOT => self.store_snapshot(args, out),
            LANA_HOST_STORE_COMMIT_IF => self.store_commit_if(args, out),
            LANA_HOST_ADAPTER_LOAD => self.adapter_load(args, out),
            LANA_HOST_ADAPTER_FETCH => self.adapter_fetch(args, out),
            LANA_HOST_POLICY_EVALUATE => self.policy_evaluate(args, out),
            LANA_HOST_POLICY_STORE_DECISION => self.policy_store_decision(args, out),
            LANA_HOST_LEDGER_APPEND => self.ledger_append(args, out),
            LANA_HOST_LEDGER_QUERY => self.ledger_query(args, out),
            _ => LanaError::Format,
        }
    }

    fn store_open(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::String(path) = &args[0].kind else {
            return LanaError::Type;
        };
        if self.store.is_some() {
            return LanaError::Conflict;
        }
        let options = StoreOptions {
            schema_version: 1,
            path: path.to_string(),
            timeout_ms: 0,
        };
        match store::store_open(&options) {
            Ok(store) => {
                self.store = Some(store);
                *out = Value::null();
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn store_put(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 2 {
            return LanaError::Type;
        }
        let ValueKind::String(key) = &args[0].kind else {
            return LanaError::Type;
        };
        let store = match self.store.as_mut() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        match store::store_put(store, key, &args[1]) {
            Ok(()) => {
                *out = Value::null();
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn store_get(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::String(key) = &args[0].kind else {
            return LanaError::Type;
        };
        let store = match self.store.as_ref() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        match store::store_get(store, key) {
            Ok(value) => {
                *out = value;
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn store_delete(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::String(key) = &args[0].kind else {
            return LanaError::Type;
        };
        let store = match self.store.as_mut() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        match store::store_delete(store, key) {
            Ok(()) => {
                *out = Value::null();
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn store_commit(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if !args.is_empty() {
            return LanaError::Type;
        }
        let store = match self.store.as_mut() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        match store::store_commit(store) {
            Ok(info) => {
                *out = Value::number(info.revision_id as f64);
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn store_scan(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::String(prefix) = &args[0].kind else {
            return LanaError::Type;
        };
        let store = match self.store.as_ref() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        match store::store_scan(store, prefix) {
            Ok(records) => {
                let mut items = Vec::with_capacity(records.len());
                for record in records {
                    let mut map = match Map::new(&self.heap, 2) { Ok(map) => map, Err(error) => return error };
                    let key = match self.heap.string(&record.key) { Ok(key) => key, Err(error) => return error };
                    if let Err(error) = map.set(Arc::from("key"), Value::string(key), false) { return error; }
                    if let Err(error) = map.set(Arc::from("value"), record.value, false) { return error; }
                    items.push(Value::map(Arc::new(Mutex::new(map))));
                }
                let array = match Array::from_items(&self.heap, items) {
                    Ok(array) => array, Err(error) => return error,
                };
                *out = Value::array(Arc::new(Mutex::new(array)));
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn store_current_revision(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if !args.is_empty() {
            return LanaError::Type;
        }
        let store = match self.store.as_ref() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        match store::store_current_revision(store) {
            Ok(info) => {
                *out = Value::number(info.revision_id as f64);
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn store_get_at(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 2 {
            return LanaError::Type;
        }
        let ValueKind::Number(revision) = &args[0].kind else {
            return LanaError::Type;
        };
        let ValueKind::String(key) = &args[1].kind else {
            return LanaError::Type;
        };
        let store = match self.store.as_ref() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        let revision = match number_as_u64(*revision) { Ok(value) => value, Err(error) => return error };
        match store::store_get_at(store, revision, key) {
            Ok(value) => {
                *out = value;
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn store_snapshot(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if !args.is_empty() {
            return LanaError::Type;
        }
        let store = match self.store.as_mut() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        match store::store_snapshot(store) {
            Ok((value, _info)) => {
                *out = value;
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn store_commit_if(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::Number(base_rev) = &args[0].kind else {
            return LanaError::Type;
        };
        let store = match self.store.as_mut() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        let current = match store::store_current_revision(store) {
            Ok(info) => info,
            Err(e) => return e,
        };
        let base_rev = match number_as_u64(*base_rev) { Ok(value) => value, Err(error) => return error };
        if current.revision_id != base_rev {
            return LanaError::Conflict;
        }
        match store::store_commit(store) {
            Ok(info) => {
                *out = Value::number(info.revision_id as f64);
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn adapter_load(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 2 {
            return LanaError::Type;
        }
        let ValueKind::Number(kind) = &args[0].kind else {
            return LanaError::Type;
        };
        let ValueKind::String(config) = &args[1].kind else {
            return LanaError::Type;
        };
        let options = AdapterOptions {
            schema_version: 1,
            kind: match *kind as u32 {
                0 => AdapterKind::Json,
                1 => AdapterKind::Csv,
                2 => AdapterKind::Sqlite,
                _ => AdapterKind::HttpJson,
            },
            config: Some(config.clone()),
        };
        match adapters::adapter_load(&options) {
            Ok(adapter) => {
                self.adapter = Some(adapter);
                *out = Value::null();
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn adapter_fetch(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::String(query) = &args[0].kind else {
            return LanaError::Type;
        };
        let adapter = match self.adapter.as_ref() {
            Some(adapter) => adapter,
            None => return LanaError::InvalidState,
        };
        match adapters::adapter_fetch(adapter, query) {
            Ok(value) => {
                *out = value;
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn policy_evaluate(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 3 {
            return LanaError::Type;
        }
        let policy = match policy_from_value(&args[0]) {
            Ok(policy) => policy,
            Err(e) => return e,
        };
        let evaluation = match evaluation_from_value(&args[2]) {
            Ok(evaluation) => evaluation,
            Err(e) => return e,
        };
        match policy::policy_evaluate(&policy, &args[1], &evaluation) {
            Ok(decision) => {
                *out = match decision_to_value(&self.heap, &decision) { Ok(value) => value, Err(error) => return error };
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn policy_store_decision(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 1 {
            return LanaError::Type;
        }
        let decision = match decision_from_value(&args[0]) {
            Ok(decision) => decision,
            Err(e) => return e,
        };
        let store = match self.store.as_mut() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        match policy::policy_store_decision(store, &decision) {
            Ok(()) => {
                *out = Value::null();
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn ledger_append(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 1 {
            return LanaError::Type;
        }
        let input = match event_input_from_value(&args[0]) {
            Ok(input) => input,
            Err(e) => return e,
        };
        let store = match self.store.as_mut() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        let mut ledger = match ledger::ledger_open(store) {
            Ok(ledger) => ledger,
            Err(e) => return e,
        };
        match ledger::ledger_append(&mut ledger, &input) {
            Ok(event) => {
                *out = match event_to_value(&self.heap, &event) { Ok(value) => value, Err(error) => return error };
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn ledger_query(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 1 {
            return LanaError::Type;
        }
        let query = match ledger_query_from_value(&args[0]) {
            Ok(query) => query,
            Err(e) => return e,
        };
        let store = match self.store.as_mut() {
            Some(store) => store,
            None => return LanaError::InvalidState,
        };
        let ledger = match ledger::ledger_open(store) {
            Ok(ledger) => ledger,
            Err(e) => return e,
        };
        match ledger::ledger_query(&ledger, &query) {
            Ok(events) => {
                let mut items = Vec::with_capacity(events.len());
                for event in &events {
                    items.push(match event_to_value(&self.heap, event) { Ok(value) => value, Err(error) => return error });
                }
                let array = match Array::from_items(&self.heap, items) {
                    Ok(array) => array, Err(error) => return error,
                };
                *out = Value::array(Arc::new(Mutex::new(array)));
                LanaError::Ok
            }
            Err(e) => e,
        }
    }
}

// --- Value <-> struct conversion -------------------------------------------

fn map_get_string(map: &Map, key: &str) -> Result<Arc<str>, LanaError> {
    match map.get(key) {
        Some(Value { kind: ValueKind::String(s), .. }) => Ok(s.clone()),
        _ => Err(LanaError::Type),
    }
}

fn map_get_optional_string(map: &Map, key: &str) -> Result<Option<Arc<str>>, LanaError> {
    match map.get(key) {
        None => Ok(None),
        Some(Value { kind: ValueKind::String(s), .. }) => Ok(Some(s.clone())),
        Some(_) => Err(LanaError::Type),
    }
}

fn map_get_number(map: &Map, key: &str) -> Result<f64, LanaError> {
    match map.get(key) {
        Some(Value { kind: ValueKind::Number(n), .. }) => Ok(*n),
        _ => Err(LanaError::Type),
    }
}

fn map_get_u64(map: &Map, key: &str) -> Result<u64, LanaError> {
    number_as_u64(map_get_number(map, key)?)
}

fn number_as_u64(number: f64) -> Result<u64, LanaError> {
    if !number.is_finite() || number < 0.0 || number >= 18446744073709551616.0 || number.fract() != 0.0 {
        return Err(LanaError::InvalidParameters);
    }
    Ok(number as u64)
}

fn map_get_u32(map: &Map, key: &str) -> Result<u32, LanaError> {
    u32::try_from(map_get_u64(map, key)?).map_err(|_| LanaError::Schema)
}

fn policy_from_value(value: &Value) -> Result<Policy, LanaError> {
    let map = match &value.kind {
        ValueKind::Map(map) => map,
        _ => return Err(LanaError::Type),
    };
    let map = map.lock().unwrap();
    let kind = match map_get_u32(&map, "rule_kind")? {
        0 => PolicyRuleKind::ProbabilityAtLeast,
        1 => PolicyRuleKind::Equals,
        2 => PolicyRuleKind::OrderLessThan,
        3 => PolicyRuleKind::Present,
        _ => return Err(LanaError::Schema),
    };
    Ok(Policy {
        schema_version: 1,
        policy_id: map_get_string(&map, "policy_id")?,
        rule: PolicyRule {
            schema_version: 1,
            kind,
            field: map_get_string(&map, "rule_field")?,
            threshold: map_get_number(&map, "rule_threshold")?,
            expected: map_get_optional_string(&map, "rule_expected")?,
            effect: map_get_string(&map, "rule_effect")?,
        },
    })
}

fn evaluation_from_value(value: &Value) -> Result<PolicyEvaluation, LanaError> {
    let map = match &value.kind {
        ValueKind::Map(map) => map,
        _ => return Err(LanaError::Type),
    };
    let map = map.lock().unwrap();
    Ok(PolicyEvaluation {
        schema_version: 1,
        decision_id: map_get_u64(&map, "decision_id")?,
        target: map_get_string(&map, "target")?,
        scope: map_get_string(&map, "scope")?,
        input_revision: map_get_u64(&map, "input_revision")?,
        evidence_ids: map_get_optional_string(&map, "evidence_ids")?,
        derivation_ids: map_get_optional_string(&map, "derivation_ids")?,
        relationship_resolution: map_get_optional_string(&map, "relationship_resolution")?,
        evaluation_time: map_get_u64(&map, "evaluation_time")?,
        reason: map_get_string(&map, "reason")?,
        requested_evidence: map_get_optional_string(&map, "requested_evidence")?,
    })
}

fn decision_from_value(value: &Value) -> Result<Decision, LanaError> {
    let map = match &value.kind {
        ValueKind::Map(map) => map,
        _ => return Err(LanaError::Type),
    };
    let map = map.lock().unwrap();
    let outcome = match map_get_u32(&map, "outcome")? {
        0 => PolicyOutcome::Authorize,
        1 => PolicyOutcome::Refuse,
        2 => PolicyOutcome::RequestMoreEvidence,
        _ => return Err(LanaError::Schema),
    };
    Ok(Decision {
        schema_version: 1,
        decision_id: map_get_u64(&map, "decision_id")?,
        policy_id: map_get_string(&map, "policy_id")?,
        policy_version: hex_decode(&map_get_string(&map, "policy_version")?)?,
        target: map_get_string(&map, "target")?,
        scope: map_get_string(&map, "scope")?,
        input_revision: map_get_u64(&map, "input_revision")?,
        evidence_ids: map_get_optional_string(&map, "evidence_ids")?,
        derivation_ids: map_get_optional_string(&map, "derivation_ids")?,
        relationship_resolution: map_get_optional_string(&map, "relationship_resolution")?,
        outcome,
        effect: map_get_optional_string(&map, "effect")?,
        evaluation_time: map_get_u64(&map, "evaluation_time")?,
        reason: map_get_string(&map, "reason")?,
        requested_evidence: map_get_optional_string(&map, "requested_evidence")?,
    })
}

fn event_input_from_value(value: &Value) -> Result<EventInput, LanaError> {
    let map = match &value.kind {
        ValueKind::Map(map) => map,
        _ => return Err(LanaError::Type),
    };
    let map = map.lock().unwrap();
    Ok(EventInput {
        schema_version: 1,
        entity: map_get_string(&map, "entity")?,
        actor: map_get_string(&map, "actor")?,
        action: map_get_string(&map, "action")?,
        reason: map_get_optional_string(&map, "reason")?,
        timestamp: map_get_u64(&map, "timestamp")?,
        correction_of: map_get_u64(&map, "correction_of")?,
    })
}

fn ledger_query_from_value(value: &Value) -> Result<LedgerQuery, LanaError> {
    let map = match &value.kind {
        ValueKind::Map(map) => map,
        _ => return Err(LanaError::Type),
    };
    let map = map.lock().unwrap();
    Ok(LedgerQuery {
        schema_version: 1,
        entity: map_get_optional_string(&map, "entity")?,
        actor: map_get_optional_string(&map, "actor")?,
        action: map_get_optional_string(&map, "action")?,
        start_timestamp: map_get_u64(&map, "start_timestamp")?,
        end_timestamp: map_get_u64(&map, "end_timestamp")?,
    })
}

fn decision_to_value(heap: &lana_vm::heap::Heap, decision: &Decision) -> Result<Value, LanaError> {
    let mut map = Map::new(heap, 14)?;
    map.set(Arc::from("decision_id"), Value::number(decision.decision_id as f64), false)?;
    map.set(Arc::from("policy_id"), Value::string(decision.policy_id.clone()), false)?;
    map.set(Arc::from("policy_version"), Value::string(Arc::from(hex_encode(&decision.policy_version).as_str())), false)?;
    map.set(Arc::from("target"), Value::string(decision.target.clone()), false)?;
    map.set(Arc::from("scope"), Value::string(decision.scope.clone()), false)?;
    map.set(Arc::from("input_revision"), Value::number(decision.input_revision as f64), false)?;
    if let Some(s) = &decision.evidence_ids {
        map.set(Arc::from("evidence_ids"), Value::string(s.clone()), false)?;
    }
    if let Some(s) = &decision.derivation_ids {
        map.set(Arc::from("derivation_ids"), Value::string(s.clone()), false)?;
    }
    if let Some(s) = &decision.relationship_resolution {
        map.set(Arc::from("relationship_resolution"), Value::string(s.clone()), false)?;
    }
    map.set(Arc::from("outcome"), Value::number(decision.outcome as u32 as f64), false)?;
    if let Some(s) = &decision.effect {
        map.set(Arc::from("effect"), Value::string(s.clone()), false)?;
    }
    map.set(Arc::from("evaluation_time"), Value::number(decision.evaluation_time as f64), false)?;
    map.set(Arc::from("reason"), Value::string(decision.reason.clone()), false)?;
    if let Some(s) = &decision.requested_evidence {
        map.set(Arc::from("requested_evidence"), Value::string(s.clone()), false)?;
    }
    Ok(Value::map(Arc::new(Mutex::new(map))))
}

fn event_to_value(heap: &lana_vm::heap::Heap, event: &Event) -> Result<Value, LanaError> {
    let mut map = Map::new(heap, 8)?;
    map.set(Arc::from("event_id"), Value::number(event.event_id as f64), false)?;
    map.set(Arc::from("entity"), Value::string(event.entity.clone()), false)?;
    map.set(Arc::from("actor"), Value::string(event.actor.clone()), false)?;
    map.set(Arc::from("action"), Value::string(event.action.clone()), false)?;
    map.set(Arc::from("reason"), Value::string(event.reason.clone()), false)?;
    map.set(Arc::from("timestamp"), Value::number(event.timestamp as f64), false)?;
    map.set(Arc::from("revision"), Value::number(event.revision as f64), false)?;
    map.set(Arc::from("correction_of"), Value::number(event.correction_of as f64), false)?;
    Ok(Value::map(Arc::new(Mutex::new(map))))
}

fn hex_encode(digest: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

fn hex_value(character: u8) -> Option<u8> {
    match character {
        b'0'..=b'9' => Some(character - b'0'),
        b'a'..=b'f' => Some(character - b'a' + 10),
        b'A'..=b'F' => Some(character - b'A' + 10),
        _ => None,
    }
}

fn hex_decode(text: &str) -> Result<[u8; 32], LanaError> {
    if text.len() != 64 {
        return Err(LanaError::Corruption);
    }
    let bytes = text.as_bytes();
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        let high = hex_value(bytes[index * 2]).ok_or(LanaError::Corruption)?;
        let low = hex_value(bytes[index * 2 + 1]).ok_or(LanaError::Corruption)?;
        *byte = (high << 4) | low;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn numeric_ids_reject_truncation_saturation_and_nonfinite_values() {
        for number in [-1.0, 0.5, f64::NAN, f64::INFINITY, 18446744073709551616.0] {
            assert_eq!(super::number_as_u64(number), Err(LanaError::InvalidParameters));
        }
        assert_eq!(super::number_as_u64(0.0), Ok(0));
        assert_eq!(super::number_as_u64(42.0), Ok(42));
    }

    use super::*;

    fn temp_store(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("lana_hostcalls_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.to_str().unwrap().to_string()
    }

    fn map(entries: &[(&str, Value)]) -> Value {
        let mut map = Map::new(&lana_vm::heap::Heap::default(), entries.len()).unwrap();
        for (key, value) in entries {
            map.set(Arc::from(*key), value.clone(), false).unwrap();
        }
        Value::map(Arc::new(Mutex::new(map)))
    }

    fn string(s: &str) -> Value {
        Value::string(Arc::from(s))
    }

    fn number(n: f64) -> Value {
        Value::number(n)
    }

    fn dispatch(host: &mut StoreHost, id: u32, args: &[Value]) -> (LanaError, Value) {
        let mut out = Value::null();
        let code = host.dispatch(id, args, &mut out);
        (code, out)
    }

    #[test]
    fn store_roundtrip() {
        let path = temp_store("store");
        let mut host = StoreHost::new();

        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]);
        assert_eq!(code, LanaError::Ok);

        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_PUT, &[string("k1"), number(42.0)]);
        assert_eq!(code, LanaError::Ok);

        // The store is append-only: a put is staged until commit.
        let (code, out) = dispatch(&mut host, LANA_HOST_STORE_COMMIT, &[]);
        assert_eq!(code, LanaError::Ok);
        assert_eq!(out.as_number(), 1.0);

        let (code, out) = dispatch(&mut host, LANA_HOST_STORE_GET, &[string("k1")]);
        assert_eq!(code, LanaError::Ok);
        assert_eq!(out.as_number(), 42.0);

        let (code, out) = dispatch(&mut host, LANA_HOST_STORE_CURRENT_REVISION, &[]);
        assert_eq!(code, LanaError::Ok);
        assert_eq!(out.as_number(), 1.0);

        let (code, out) = dispatch(&mut host, LANA_HOST_STORE_SCAN, &[string("k")]);
        assert_eq!(code, LanaError::Ok);
        let ValueKind::Array(array) = &out.kind else { panic!("expected array") };
        assert_eq!(array.lock().unwrap().items().len(), 1);

        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_DELETE, &[string("k1")]);
        assert_eq!(code, LanaError::Ok);
        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_COMMIT, &[]);
        assert_eq!(code, LanaError::Ok);

        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_GET, &[string("k1")]);
        assert_eq!(code, LanaError::NotFound);

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn store_mvcc_get_at_and_snapshot() {
        let path = temp_store("mvcc");
        let mut host = StoreHost::new();
        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]);
        assert_eq!(code, LanaError::Ok);

        // Rev 1: k = 1.
        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_PUT, &[string("k"), number(1.0)]);
        assert_eq!(code, LanaError::Ok);
        let (code, out) = dispatch(&mut host, LANA_HOST_STORE_COMMIT, &[]);
        assert_eq!(code, LanaError::Ok);
        assert_eq!(out.as_number(), 1.0);

        // Rev 2: k = 2.
        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_PUT, &[string("k"), number(2.0)]);
        assert_eq!(code, LanaError::Ok);
        let (code, out) = dispatch(&mut host, LANA_HOST_STORE_COMMIT, &[]);
        assert_eq!(code, LanaError::Ok);
        assert_eq!(out.as_number(), 2.0);

        // Point-in-time read at rev 1 sees the first value.
        let (code, out) = dispatch(&mut host, LANA_HOST_STORE_GET_AT, &[number(1.0), string("k")]);
        assert_eq!(code, LanaError::Ok);
        assert_eq!(out.as_number(), 1.0);

        // Snapshot reflects the current (rev 2) state.
        let (code, out) = dispatch(&mut host, LANA_HOST_STORE_SNAPSHOT, &[]);
        assert_eq!(code, LanaError::Ok);
        let ValueKind::Map(map) = &out.kind else { panic!("expected map") };
        let map = map.lock().unwrap();
        assert_eq!(map.get("k").unwrap().as_number(), 2.0);

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn store_commit_if_optimistic() {
        let path = temp_store("commit_if");
        let mut host = StoreHost::new();
        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]);
        assert_eq!(code, LanaError::Ok);

        // Commit against the current base (rev 0) succeeds.
        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_PUT, &[string("k"), number(1.0)]);
        assert_eq!(code, LanaError::Ok);
        let (code, out) = dispatch(&mut host, LANA_HOST_STORE_COMMIT_IF, &[number(0.0)]);
        assert_eq!(code, LanaError::Ok);
        assert_eq!(out.as_number(), 1.0);

        // A stale base (rev 0 again) now conflicts.
        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_PUT, &[string("k"), number(2.0)]);
        assert_eq!(code, LanaError::Ok);
        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_COMMIT_IF, &[number(0.0)]);
        assert_eq!(code, LanaError::Conflict);

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn adapter_json_fetch() {
        let mut host = StoreHost::new();
        let (code, _) = dispatch(&mut host, LANA_HOST_ADAPTER_LOAD, &[number(0.0), string("{}")]);
        assert_eq!(code, LanaError::Ok);

        let (code, out) = dispatch(&mut host, LANA_HOST_ADAPTER_FETCH, &[string("[{\"a\":1}]")]);
        assert_eq!(code, LanaError::Ok);
        let ValueKind::Array(array) = &out.kind else { panic!("expected array") };
        assert_eq!(array.lock().unwrap().items().len(), 1);
    }

    #[test]
    fn adapter_fetch_without_load_is_invalid_state() {
        let mut host = StoreHost::new();
        let (code, _) = dispatch(&mut host, LANA_HOST_ADAPTER_FETCH, &[string("[]")]);
        assert_eq!(code, LanaError::InvalidState);
    }

    #[test]
    fn policy_evaluate_authorizes() {
        let mut host = StoreHost::new();
        let policy = map(&[
            ("policy_id", string("p1")),
            ("rule_kind", number(0.0)),
            ("rule_field", string("p")),
            ("rule_threshold", number(0.5)),
            ("rule_effect", string("grant")),
        ]);
        let input = map(&[("p", number(0.75))]);
        let evaluation = map(&[
            ("decision_id", number(1.0)),
            ("target", string("t")),
            ("scope", string("s")),
            ("input_revision", number(0.0)),
            ("evaluation_time", number(0.0)),
            ("reason", string("r")),
        ]);
        let (code, out) = dispatch(&mut host, LANA_HOST_POLICY_EVALUATE, &[policy, input, evaluation]);
        assert_eq!(code, LanaError::Ok);
        let ValueKind::Map(map) = &out.kind else { panic!("expected map") };
        let map = map.lock().unwrap();
        assert_eq!(map.get("outcome").unwrap().as_number(), 0.0); // Authorize
        assert_eq!(map.get("effect").unwrap().as_string(), Arc::from("grant"));
    }

    #[test]
    fn ledger_append_and_query() {
        let path = temp_store("ledger");
        let mut host = StoreHost::new();
        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]);
        assert_eq!(code, LanaError::Ok);

        let event = map(&[
            ("entity", string("e1")),
            ("actor", string("a1")),
            ("action", string("grant")),
            ("reason", string("approved")),
            ("timestamp", number(100.0)),
            ("correction_of", number(0.0)),
        ]);
        let (code, out) = dispatch(&mut host, LANA_HOST_LEDGER_APPEND, &[event]);
        assert_eq!(code, LanaError::Ok);
        let ValueKind::Map(event_map) = &out.kind else { panic!("expected map") };
        assert_eq!(event_map.lock().unwrap().get("event_id").unwrap().as_number(), 1.0);

        let query = map(&[
            ("entity", string("e1")),
            ("start_timestamp", number(0.0)),
            ("end_timestamp", number(0.0)),
        ]);
        let (code, out) = dispatch(&mut host, LANA_HOST_LEDGER_QUERY, &[query]);
        assert_eq!(code, LanaError::Ok);
        let ValueKind::Array(array) = &out.kind else { panic!("expected array") };
        assert_eq!(array.lock().unwrap().items().len(), 1);

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn store_requires_open() {
        let mut host = StoreHost::new();
        let (code, _) = dispatch(&mut host, LANA_HOST_STORE_PUT, &[string("k"), number(1.0)]);
        assert_eq!(code, LanaError::InvalidState);
    }
}
