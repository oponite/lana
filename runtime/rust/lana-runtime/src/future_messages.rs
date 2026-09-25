//! Durable local inbox for messages released by explicit time and event checks.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use lana_bytecode::LanaError;
use lana_vm::heap::Heap;
use lana_vm::value::{Array, Map, Value, ValueKind};

use crate::codec;
use crate::data;
use crate::store::{self, Store};

const ROOT: &str = "__future_messages/v1/";
const STATUSES: [&str; 4] = ["pending", "ready", "acknowledged", "cancelled"];
const MAX_ID: usize = 128;
const MAX_PAYLOAD: usize = 64 * 1024;
const MAX_RECORD: usize = 128 * 1024;
const MAX_PENDING: usize = 1024;
const MAX_COMPARISONS: usize = 16;
const MAX_CONTEXT_FIELDS: usize = 64;
const MAX_PROVENANCE: usize = 16;
const MAX_PAGE: usize = 100;
const MAX_SCALAR_STRING: usize = 4096;

struct Comparison {
    field: String,
    op: String,
    value: Value,
}

struct Condition {
    not_before_utc: Option<f64>,
    event: Option<String>,
    comparisons: Vec<Comparison>,
}

struct Context {
    id: String,
    event: String,
    values: Vec<(String, Value)>,
}

enum Match {
    Ready,
    Pending(String),
}

pub fn utc_now() -> Result<f64, LanaError> {
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|_| LanaError::Io)?;
    Ok(duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) / 1_000_000_000.0)
}

fn object(heap: &Heap, fields: &[(&str, Value)]) -> Result<Value, LanaError> {
    let mut map = Map::new(heap, fields.len())?;
    for (name, value) in fields {
        map.set(Arc::from(*name), value.clone(), false)?;
    }
    Ok(Value::map(Arc::new(Mutex::new(map))))
}

fn array(heap: &Heap, values: Vec<Value>) -> Result<Value, LanaError> {
    Ok(Value::array(Arc::new(Mutex::new(Array::from_items(heap, values)?))))
}

fn field(map: &Map, name: &str) -> Result<Value, LanaError> {
    map.get(name).cloned().ok_or(LanaError::Schema)
}

fn string(value: &Value) -> Result<String, LanaError> {
    match &value.kind {
        ValueKind::String(text) => Ok(text.to_string()),
        _ => Err(LanaError::Schema),
    }
}

fn bounded_name(value: &Value) -> Result<String, LanaError> {
    let name = string(value)?;
    if name.is_empty() || name.len() > MAX_ID { return Err(LanaError::Limit); }
    Ok(name)
}

fn id(value: &Value) -> Result<String, LanaError> {
    let id = bounded_name(value)?;
    if !id.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte)) {
        return Err(LanaError::Key);
    }
    Ok(id)
}

fn timestamp(value: &Value) -> Result<f64, LanaError> {
    match &value.kind {
        ValueKind::Number(number) if number.is_finite() && *number >= 0.0 => Ok(*number),
        _ => Err(LanaError::Schema),
    }
}

fn scalar(value: &Value) -> bool {
    matches!(&value.kind, ValueKind::Null | ValueKind::Bool(_) | ValueKind::Number(_) | ValueKind::String(_))
}

fn plain(value: &Value) -> Result<(), LanaError> {
    if value.derivation.is_some() || value.reactive.is_some() || value.claim.is_some() || value.planned_effect.is_some() {
        return Err(LanaError::UnsupportedValue);
    }
    match &value.kind {
        ValueKind::Array(items) => {
            for item in items.lock().unwrap().items() { plain(item)?; }
        }
        ValueKind::Map(fields) => {
            for entry in fields.lock().unwrap().entries() { plain(&entry.value)?; }
        }
        _ => {}
    }
    Ok(())
}

fn snapshot(value: &Value, max_bytes: usize) -> Result<Value, LanaError> {
    // Encoding first rejects cycles, unsupported handles, and non-finite values.
    let encoded = codec::encode_value(value)?;
    if encoded.len() > max_bytes { return Err(LanaError::Limit); }
    plain(value)?;
    data::json_parse(&encoded)
}

fn condition(value: &Value) -> Result<Condition, LanaError> {
    let ValueKind::Map(map) = &value.kind else { return Err(LanaError::Schema); };
    let map = map.lock().unwrap();
    if map.entries().iter().any(|entry| !matches!(&*entry.key, "not_before_utc" | "event" | "comparisons")) {
        return Err(LanaError::Schema);
    }
    let not_before_utc = map.get("not_before_utc").map(timestamp).transpose()?;
    let event = map.get("event").map(bounded_name).transpose()?;
    let comparisons = match map.get("comparisons") {
        None => Vec::new(),
        Some(Value { kind: ValueKind::Array(items), .. }) => {
            let items = items.lock().unwrap();
            if items.items().len() > MAX_COMPARISONS { return Err(LanaError::Limit); }
            let mut comparisons = Vec::with_capacity(items.items().len());
            for item in items.items() {
                let ValueKind::Map(fields) = &item.kind else { return Err(LanaError::Schema); };
                let fields = fields.lock().unwrap();
                if fields.entries().len() != 3 { return Err(LanaError::Schema); }
                let field_name = bounded_name(&field(&fields, "field")?)?;
                let op = string(&field(&fields, "op")?)?;
                if !matches!(op.as_str(), "eq" | "ne" | "lt" | "le" | "gt" | "ge") { return Err(LanaError::Schema); }
                let expected = field(&fields, "value")?;
                if !scalar(&expected) { return Err(LanaError::Schema); }
                if let ValueKind::Number(n) = &expected.kind { if !n.is_finite() { return Err(LanaError::Schema); } }
                if let ValueKind::String(text) = &expected.kind { if text.len() > MAX_SCALAR_STRING { return Err(LanaError::Limit); } }
                if !matches!(op.as_str(), "eq" | "ne") && !matches!(&expected.kind, ValueKind::Number(_) | ValueKind::String(_)) {
                    return Err(LanaError::Schema);
                }
                comparisons.push(Comparison { field: field_name, op, value: expected });
            }
            comparisons
        }
        Some(_) => return Err(LanaError::Schema),
    };
    if not_before_utc.is_none() && event.is_none() { return Err(LanaError::Schema); }
    if event.is_none() && !comparisons.is_empty() { return Err(LanaError::Schema); }
    Ok(Condition { not_before_utc, event, comparisons })
}

fn context(value: &Value) -> Result<Option<Context>, LanaError> {
    let ValueKind::Map(map) = &value.kind else {
        return if matches!(&value.kind, ValueKind::Null) { Ok(None) } else { Err(LanaError::Schema) };
    };
    let map = map.lock().unwrap();
    if map.entries().len() != 3 { return Err(LanaError::Schema); }
    let id = bounded_name(&field(&map, "id")?)?;
    let event = bounded_name(&field(&map, "event")?)?;
    let values = field(&map, "values")?;
    drop(map);
    let ValueKind::Map(values) = values.kind else { return Err(LanaError::Schema); };
    let values = values.lock().unwrap();
    if values.entries().len() > MAX_CONTEXT_FIELDS { return Err(LanaError::Limit); }
    let mut captured = Vec::with_capacity(values.entries().len());
    for entry in values.entries() {
        if entry.key.is_empty() || entry.key.len() > MAX_ID { return Err(LanaError::Limit); }
        let value = match &entry.value.reactive {
            Some(reactive) => reactive.lock().unwrap().current.clone().unwrap_or_else(|| entry.value.clone()),
            None => entry.value.clone(),
        };
        if let ValueKind::String(text) = &value.kind { if text.len() > MAX_SCALAR_STRING { return Err(LanaError::Limit); } }
        captured.push((entry.key.to_string(), value));
    }
    Ok(Some(Context { id, event, values: captured }))
}

fn compare(actual: &Value, comparison: &Comparison) -> Result<bool, LanaError> {
    let result = match (&actual.kind, &comparison.value.kind) {
        (ValueKind::Null, ValueKind::Null) => 0,
        (ValueKind::Bool(a), ValueKind::Bool(b)) => a.cmp(b) as i32,
        (ValueKind::Number(a), ValueKind::Number(b)) if a.is_finite() => {
            if a < b { -1 } else if a > b { 1 } else { 0 }
        }
        (ValueKind::String(a), ValueKind::String(b)) => a.cmp(b) as i32,
        _ => return Err(LanaError::Type),
    };
    Ok(match comparison.op.as_str() {
        "eq" => result == 0,
        "ne" => result != 0,
        "lt" => result < 0,
        "le" => result <= 0,
        "gt" => result > 0,
        "ge" => result >= 0,
        _ => return Err(LanaError::Schema),
    })
}

fn evaluate(condition: &Condition, context: Option<&Context>, now: f64) -> Result<Match, LanaError> {
    let mut context_result = Match::Ready;
    if let Some(event) = &condition.event {
        context_result = Match::Pending("event".into());
        if let Some(context) = context {
            if &context.event == event {
                context_result = Match::Ready;
                let mut first_pending = None;
                for comparison in &condition.comparisons {
                    let Some((_, actual)) = context.values.iter().find(|(name, _)| name == &comparison.field) else {
                        first_pending.get_or_insert_with(|| format!("missing:{}", comparison.field));
                        continue;
                    };
                    if actual.reactive.is_some() || matches!(&actual.kind, ValueKind::Possibility(_) | ValueKind::PathSet(_) | ValueKind::Distribution { .. } | ValueKind::StateDist(_) | ValueKind::Joint(_)) {
                        first_pending.get_or_insert_with(|| format!("unresolved:{}", comparison.field));
                        continue;
                    }
                    if !compare(actual, comparison)? {
                        first_pending.get_or_insert_with(|| format!("comparison:{}", comparison.field));
                    }
                }
                if let Some(reason) = first_pending { context_result = Match::Pending(reason); }
            }
        }
    }
    if condition.not_before_utc.is_some_and(|threshold| now < threshold) {
        return Ok(Match::Pending("time".into()));
    }
    Ok(context_result)
}

fn key(status: &str, id: &str) -> String { format!("{ROOT}{status}/{id}") }

fn find(store: &Store, id: &str) -> Result<Option<(String, Value)>, LanaError> {
    let mut found = None;
    for status in STATUSES {
        match store::store_get(store, &key(status, id)) {
            Ok(record) if found.is_none() => found = Some((status.into(), record)),
            Ok(_) => return Err(LanaError::Corruption),
            Err(LanaError::NotFound) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(found)
}

fn record_field(record: &Value, name: &str) -> Result<Value, LanaError> {
    let ValueKind::Map(map) = &record.kind else { return Err(LanaError::Corruption); };
    field(&map.lock().unwrap(), name).map_err(|_| LanaError::Corruption)
}

fn validate_record(record: &Value, status: &str, id: &str) -> Result<Condition, LanaError> {
    if string(&record_field(record, "id")?)? != id || string(&record_field(record, "status")?)? != status {
        return Err(LanaError::Corruption);
    }
    if !matches!(record_field(record, "schema_version")?.kind, ValueKind::Number(1.0)) {
        return Err(LanaError::IncompatibleFormat);
    }
    if !matches!(record_field(record, "record_schema")?.kind, ValueKind::Number(1.0)) {
        return Err(LanaError::IncompatibleFormat);
    }
    condition(&record_field(record, "condition")?)
}

fn created_spec(record: &Value, heap: &Heap) -> Result<String, LanaError> {
    let spec = object(heap, &[
        ("id", record_field(record, "id")?),
        ("payload", record_field(record, "payload")?),
        ("condition", record_field(record, "condition")?),
        ("provenance_refs", record_field(record, "provenance_refs")?),
    ])?;
    codec::encode_value(&spec)
}

fn next_revision(store: &Store) -> Result<String, LanaError> {
    Ok(store::store_current_revision(store)?.revision_id.checked_add(1).ok_or(LanaError::Limit)?.to_string())
}

fn commit(store: &mut Store, old_status: Option<&str>, id: &str, new_status: &str, record: &Value) -> Result<(), LanaError> {
    let result = (|| {
        store::store_put(store, &key(new_status, id), record)?;
        if let Some(old_status) = old_status { store::store_delete(store, &key(old_status, id))?; }
        store::store_commit(store)?;
        Ok(())
    })();
    if result.is_err() {
        // Commit may have reached the journal. Reopen and inspect this ID.
        let _ = store::store_close(store);
    }
    result
}

fn changed(record: &Value, status: &str, receipt_name: &str, receipt: Value, heap: &Heap) -> Result<Value, LanaError> {
    let ValueKind::Map(map) = &record.kind else { return Err(LanaError::Corruption); };
    let entries: Vec<(String, Value)> = map.lock().unwrap().entries().iter()
        .map(|entry| (entry.key.to_string(), entry.value.clone())).collect();
    let mut new = Map::new(heap, entries.len() + 1)?;
    for (name, value) in entries { new.set(Arc::from(name), value, false)?; }
    new.set(Arc::from("status"), Value::string(Arc::from(status)), false)?;
    new.set(Arc::from("domain_status"), Value::string(Arc::from(status)), false)?;
    new.set(Arc::from(receipt_name), receipt, false)?;
    Ok(Value::map(Arc::new(Mutex::new(new))))
}

pub fn create(store: &mut Store, input: &Value, now: f64, heap: &Heap) -> Result<Value, LanaError> {
    store.ensure_clean()?;
    let ValueKind::Map(map) = &input.kind else { return Err(LanaError::Schema); };
    let map = map.lock().unwrap();
    if map.entries().len() != 4 { return Err(LanaError::Schema); }
    let id_value = field(&map, "id")?;
    let payload_value = field(&map, "payload")?;
    let condition_input = field(&map, "condition")?;
    let provenance_input = field(&map, "provenance_refs")?;
    drop(map);
    let id = id(&id_value)?;
    let payload = snapshot(&payload_value, MAX_PAYLOAD)?;
    let condition_value = snapshot(&condition_input, MAX_RECORD)?;
    condition(&condition_value)?;
    let provenance = snapshot(&provenance_input, MAX_RECORD)?;
    let ValueKind::Array(refs) = &provenance.kind else { return Err(LanaError::Schema); };
    let refs = refs.lock().unwrap();
    if refs.items().len() > MAX_PROVENANCE { return Err(LanaError::Limit); }
    for reference in refs.items() { bounded_name(reference)?; }
    drop(refs);
    let candidate = object(heap, &[
        ("id", Value::string(Arc::from(id.as_str()))),
        ("schema_version", Value::number(1.0)),
        ("record_schema", Value::number(1.0)),
        ("kind", Value::string(Arc::from("future_message"))),
        ("transport_status", Value::string(Arc::from("stored"))),
        ("domain_status", Value::string(Arc::from("pending"))),
        ("payload", payload),
        ("error", Value::null()),
        ("evidence", provenance.clone()),
        ("assumptions", Value::null()),
        ("exactness", Value::string(Arc::from("exact"))),
        ("metadata", Value::null()),
        ("condition", condition_value),
        ("provenance_refs", provenance),
        ("created_at_utc", Value::number(now)),
        ("status", Value::string(Arc::from("pending"))),
    ])?;
    if codec::encode_value(&candidate)?.len() > MAX_RECORD { return Err(LanaError::Limit); }
    if let Some((status, existing)) = find(store, &id)? {
        validate_record(&existing, &status, &id)?;
        if created_spec(&existing, heap)? == created_spec(&candidate, heap)? { return Ok(existing); }
        return Err(LanaError::Conflict);
    }
    if store::store_scan_page(store, &key("pending", ""), "", MAX_PENDING)?.len() >= MAX_PENDING { return Err(LanaError::Limit); }
    let receipt = object(heap, &[
        ("at_utc", Value::number(now)),
        ("store_revision", Value::string(Arc::from(next_revision(store)?))),
    ])?;
    let candidate = changed(&candidate, "pending", "creation_receipt", receipt, heap)?;
    if codec::encode_value(&candidate)?.len() > MAX_RECORD { return Err(LanaError::Limit); }
    commit(store, None, &id, "pending", &candidate)?;
    Ok(candidate)
}

pub fn inspect(store: &Store, value: &Value) -> Result<Value, LanaError> {
    let id = id(value)?;
    let (status, record) = find(store, &id)?.ok_or(LanaError::NotFound)?;
    validate_record(&record, &status, &id)?;
    Ok(record)
}

pub fn check(store: &mut Store, context_value: &Value, now: f64, heap: &Heap) -> Result<Value, LanaError> {
    store.ensure_clean()?;
    let context = context(context_value)?;
    let pending = store::store_scan_page(store, &key("pending", ""), "", MAX_PENDING + 1)?;
    if pending.len() > MAX_PENDING { return Err(LanaError::Limit); }
    let mut ready = Vec::new();
    let mut waiting = Vec::new();
    // Validate every message before the first transition.
    for item in pending {
        let id = item.key.strip_prefix(&key("pending", "")).ok_or(LanaError::Corruption)?.to_string();
        let condition = validate_record(&item.value, "pending", &id)?;
        match evaluate(&condition, context.as_ref(), now)? {
            Match::Ready => ready.push((id, item.value)),
            Match::Pending(reason) => waiting.push((id, reason)),
        }
    }
    let ready_ids = array(heap, ready.iter().map(|(id, _)| Value::string(Arc::from(id.as_str()))).collect())?;
    let mut pending_out = Vec::with_capacity(waiting.len());
    for (id, reason) in waiting {
        pending_out.push(object(heap, &[("id", Value::string(Arc::from(id))), ("reason", Value::string(Arc::from(reason)))] )?);
    }
    let result = object(heap, &[("released_ids", ready_ids), ("pending", array(heap, pending_out)?)])?;
    let first_revision: u64 = next_revision(store)?.parse().map_err(|_| LanaError::Limit)?;
    let mut prepared = Vec::with_capacity(ready.len());
    for (index, (id, record)) in ready.into_iter().enumerate() {
        let revision = first_revision.checked_add(index as u64).ok_or(LanaError::Limit)?.to_string();
        let receipt = object(heap, &[
            ("matched_at_utc", Value::number(now)),
            ("context_id", context.as_ref().map_or_else(Value::null, |ctx| Value::string(Arc::from(ctx.id.as_str())))),
            ("condition", record_field(&record, "condition")?),
            ("store_revision", Value::string(Arc::from(revision))),
        ])?;
        let updated = changed(&record, "ready", "ready_receipt", receipt, heap)?;
        if codec::encode_value(&updated)?.len() > MAX_RECORD { return Err(LanaError::Limit); }
        prepared.push((id, updated));
    }
    for (id, updated) in prepared {
        commit(store, Some("pending"), &id, "ready", &updated)?;
    }
    Ok(result)
}

pub fn receive(store: &Store, after: &Value, limit: &Value, heap: &Heap) -> Result<Value, LanaError> {
    let after = string(after)?;
    if !after.is_empty() { id(&Value::string(Arc::from(after.as_str())))?; }
    let ValueKind::Number(limit) = &limit.kind else { return Err(LanaError::Schema); };
    if !limit.is_finite() || limit.fract() != 0.0 || !(1.0..=MAX_PAGE as f64).contains(limit) { return Err(LanaError::Limit); }
    let records = store::store_scan_page(store, &key("ready", ""), &if after.is_empty() { String::new() } else { key("ready", &after) }, *limit as usize)?;
    for item in &records {
        let id = item.key.strip_prefix(&key("ready", "")).ok_or(LanaError::Corruption)?;
        validate_record(&item.value, "ready", id)?;
    }
    array(heap, records.into_iter().map(|item| item.value).collect())
}

pub fn acknowledge(store: &mut Store, value: &Value, now: f64, heap: &Heap) -> Result<Value, LanaError> {
    transition(store, value, now, heap, "acknowledged")
}

pub fn cancel(store: &mut Store, value: &Value, now: f64, heap: &Heap) -> Result<Value, LanaError> {
    transition(store, value, now, heap, "cancelled")
}

fn transition(store: &mut Store, value: &Value, now: f64, heap: &Heap, target: &str) -> Result<Value, LanaError> {
    store.ensure_clean()?;
    let id = id(value)?;
    let (status, record) = find(store, &id)?.ok_or(LanaError::NotFound)?;
    validate_record(&record, &status, &id)?;
    if status == target { return Ok(record); }
    if status == "acknowledged" || status == "cancelled" || (target == "acknowledged" && status != "ready") {
        return Err(LanaError::InvalidState);
    }
    let revision = next_revision(store)?;
    let receipt = object(heap, &[
        ("at_utc", Value::number(now)),
        ("store_revision", Value::string(Arc::from(revision))),
    ])?;
    let updated = changed(&record, target, if target == "acknowledged" { "acknowledged_receipt" } else { "cancelled_receipt" }, receipt, heap)?;
    if codec::encode_value(&updated)?.len() > MAX_RECORD { return Err(LanaError::Limit); }
    commit(store, Some(&status), &id, target, &updated)?;
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{store_close, store_commit, store_get, store_open, store_put, StoreOptions};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn text(value: &str) -> Value { Value::string(Arc::from(value)) }

    fn message(heap: &Heap, id: &str, condition: Value, payload: Value) -> Value {
        object(heap, &[
            ("id", text(id)), ("payload", payload), ("condition", condition),
            ("provenance_refs", array(heap, vec![text("source:1")]).unwrap()),
        ]).unwrap()
    }

    fn event_condition(heap: &Heap, event: &str, field: &str, value: Value) -> Value {
        object(heap, &[
            ("event", text(event)),
            ("comparisons", array(heap, vec![object(heap, &[
                ("field", text(field)), ("op", text("eq")), ("value", value),
            ]).unwrap()]).unwrap()),
        ]).unwrap()
    }

    fn event_context(heap: &Heap, event: &str, field: &str, value: Value) -> Value {
        object(heap, &[
            ("id", text("event-1")), ("event", text(event)),
            ("values", object(heap, &[(field, value)]).unwrap()),
        ]).unwrap()
    }

    fn fresh_store() -> (String, Store) {
        let path = std::env::temp_dir().join(format!(
            "lana-future-messages-{}-{}", std::process::id(), NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let path = path.to_string_lossy().into_owned();
        let store = store_open(&StoreOptions { schema_version: 1, path: path.clone(), timeout_ms: 0 }).unwrap();
        (path, store)
    }

    #[test]
    fn local_inbox_survives_reopen_and_never_releases_twice() {
        let heap = Heap::default();
        let (path, mut store) = fresh_store();
        let time = message(&heap, "time", object(&heap, &[("not_before_utc", Value::number(10.0))]).unwrap(), text("later"));
        let project = message(&heap, "project", event_condition(&heap, "project_opened", "project_id", text("p1")), text("remember"));
        let combined = message(&heap, "combined", object(&heap, &[
            ("not_before_utc", Value::number(10.0)),
            ("event", text("assumption_contradicted")),
            ("comparisons", array(&heap, vec![object(&heap, &[
                ("field", text("assumption_id")), ("op", text("eq")), ("value", text("a1")),
            ]).unwrap()]).unwrap()),
        ]).unwrap(), text("reconsider"));
        create(&mut store, &time, 1.0, &heap).unwrap();
        create(&mut store, &project, 1.0, &heap).unwrap();
        create(&mut store, &combined, 1.0, &heap).unwrap();
        assert_eq!(string(&record_field(&create(&mut store, &time, 2.0, &heap).unwrap(), "status").unwrap()).unwrap(), "pending");
        check(&mut store, &Value::null(), 9.0, &heap).unwrap();
        let empty = receive(&store, &text(""), &Value::number(10.0), &heap).unwrap();
        let ValueKind::Array(empty) = empty.kind else { panic!("ready array"); };
        assert!(empty.lock().unwrap().items().is_empty());
        let project_event = event_context(&heap, "project_opened", "project_id", text("p1"));
        check(&mut store, &project_event, 9.0, &heap).unwrap();
        check(&mut store, &Value::null(), 10.0, &heap).unwrap();
        let contradiction = event_context(&heap, "assumption_contradicted", "assumption_id", text("a1"));
        check(&mut store, &contradiction, 10.0, &heap).unwrap();
        check(&mut store, &contradiction, 0.0, &heap).unwrap();
        let ready = receive(&store, &text(""), &Value::number(10.0), &heap).unwrap();
        let ValueKind::Array(ready) = ready.kind else { panic!("ready array"); };
        assert_eq!(ready.lock().unwrap().items().len(), 3);
        let first_page = receive(&store, &text(""), &Value::number(1.0), &heap).unwrap();
        let ValueKind::Array(first_page) = first_page.kind else { panic!("first page"); };
        let first_id = string(&record_field(&first_page.lock().unwrap().items()[0], "id").unwrap()).unwrap();
        assert_eq!(first_id, "combined");
        let second_page = receive(&store, &text(&first_id), &Value::number(1.0), &heap).unwrap();
        let ValueKind::Array(second_page) = second_page.kind else { panic!("second page"); };
        assert_eq!(string(&record_field(&second_page.lock().unwrap().items()[0], "id").unwrap()).unwrap(), "project");
        let first = inspect(&store, &text("combined")).unwrap();
        assert_eq!(string(&record_field(&first, "status").unwrap()).unwrap(), "ready");
        let receipt = record_field(&first, "ready_receipt").unwrap();
        assert_eq!(string(&record_field(&receipt, "context_id").unwrap()).unwrap(), "event-1");
        let revision = string(&record_field(&receipt, "store_revision").unwrap()).unwrap();
        assert!(revision.parse::<u64>().unwrap() > 0);
        acknowledge(&mut store, &text("combined"), 11.0, &heap).unwrap();
        let same = acknowledge(&mut store, &text("combined"), 12.0, &heap).unwrap();
        assert_eq!(string(&record_field(&record_field(&same, "acknowledged_receipt").unwrap(), "store_revision").unwrap()).unwrap(),
                   string(&record_field(&record_field(&inspect(&store, &text("combined")).unwrap(), "acknowledged_receipt").unwrap(), "store_revision").unwrap()).unwrap());
        assert_eq!(cancel(&mut store, &text("combined"), 13.0, &heap).unwrap_err(), LanaError::InvalidState);
        cancel(&mut store, &text("project"), 13.0, &heap).unwrap();
        cancel(&mut store, &text("project"), 14.0, &heap).unwrap();
        store_close(&mut store).unwrap();
        let mut reopened = store_open(&StoreOptions { schema_version: 1, path: path.clone(), timeout_ms: 0 }).unwrap();
        assert_eq!(string(&record_field(&inspect(&reopened, &text("time")).unwrap(), "status").unwrap()).unwrap(), "ready");
        assert_eq!(string(&record_field(&inspect(&reopened, &text("combined")).unwrap(), "status").unwrap()).unwrap(), "acknowledged");
        let ready = receive(&reopened, &text(""), &Value::number(10.0), &heap).unwrap();
        let ValueKind::Array(ready) = ready.kind else { panic!("ready array"); };
        assert_eq!(ready.lock().unwrap().items().len(), 1);
        store_close(&mut reopened).unwrap();
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn invalid_inputs_and_unrelated_staging_change_nothing() {
        let heap = Heap::default();
        let (path, mut store) = fresh_store();
        let due = object(&heap, &[("not_before_utc", Value::number(10.0))]).unwrap();
        let good = message(&heap, "one", due.clone(), text("body"));
        store_put(&mut store, "other", &Value::number(7.0)).unwrap();
        assert_eq!(create(&mut store, &good, 1.0, &heap).unwrap_err(), LanaError::InvalidState);
        store_commit(&mut store).unwrap();
        assert_eq!(store_get(&store, "other").unwrap().print(), "7");
        create(&mut store, &good, 1.0, &heap).unwrap();
        assert_eq!(create(&mut store, &message(&heap, "one", due.clone(), text("different")), 2.0, &heap).unwrap_err(), LanaError::Conflict);
        assert_eq!(create(&mut store, &message(&heap, "bad", due.clone(), Value::number(f64::NAN)), 1.0, &heap).unwrap_err(), LanaError::UnsupportedValue);
        assert_eq!(create(&mut store, &message(&heap, "large", due.clone(), text(&"x".repeat(MAX_PAYLOAD + 1))), 1.0, &heap).unwrap_err(), LanaError::Limit);
        let malformed = object(&heap, &[("event", text("e")), ("comparisons", array(&heap, vec![object(&heap, &[
            ("field", text("x")), ("op", text("bad")), ("value", Value::number(1.0)),
        ]).unwrap()]).unwrap())]).unwrap();
        assert_eq!(create(&mut store, &message(&heap, "malformed", malformed, text("body")), 1.0, &heap).unwrap_err(), LanaError::Schema);
        let both = object(&heap, &[
            ("not_before_utc", Value::number(10.0)), ("event", text("e")),
            ("comparisons", array(&heap, vec![object(&heap, &[
                ("field", text("x")), ("op", text("eq")), ("value", Value::number(1.0)),
            ]).unwrap()]).unwrap()),
        ]).unwrap();
        create(&mut store, &message(&heap, "both", both, text("body")), 1.0, &heap).unwrap();
        let wrong_type = event_context(&heap, "e", "x", text("wrong"));
        assert_eq!(check(&mut store, &wrong_type, 9.0, &heap).unwrap_err(), LanaError::Type);
        assert_eq!(string(&record_field(&inspect(&store, &text("both")).unwrap(), "status").unwrap()).unwrap(), "pending");
        assert_eq!(string(&record_field(&cancel(&mut store, &text("both"), 9.0, &heap).unwrap(), "status").unwrap()).unwrap(), "cancelled");
        store_close(&mut store).unwrap();
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn matcher_requires_definite_compatible_context() {
        let heap = Heap::default();
        let expected = event_condition(&heap, "operation_preparing", "name", text("deploy"));
        let expected = condition(&expected).unwrap();
        let missing = context(&event_context(&heap, "operation_preparing", "other", text("deploy"))).unwrap();
        assert!(matches!(evaluate(&expected, missing.as_ref(), 20.0).unwrap(), Match::Pending(reason) if reason == "missing:name"));
        let unresolved = context(&event_context(&heap, "operation_preparing", "name", Value::distribution(0.5, 0.5))).unwrap();
        assert!(matches!(evaluate(&expected, unresolved.as_ref(), 20.0).unwrap(), Match::Pending(reason) if reason == "unresolved:name"));
        let wrong_type = context(&event_context(&heap, "operation_preparing", "name", Value::number(1.0))).unwrap();
        assert!(matches!(evaluate(&expected, wrong_type.as_ref(), 20.0), Err(LanaError::Type)));
        let matched = context(&event_context(&heap, "operation_preparing", "name", text("deploy"))).unwrap();
        assert!(matches!(evaluate(&expected, matched.as_ref(), 20.0).unwrap(), Match::Ready));
        let before = condition(&object(&heap, &[
            ("not_before_utc", Value::number(21.0)), ("event", text("operation_preparing")),
            ("comparisons", array(&heap, vec![object(&heap, &[
                ("field", text("name")), ("op", text("eq")), ("value", text("deploy")),
            ]).unwrap()]).unwrap()),
        ]).unwrap()).unwrap();
        assert!(matches!(evaluate(&before, matched.as_ref(), 20.0).unwrap(), Match::Pending(reason) if reason == "time"));
        assert!(matches!(evaluate(&before, matched.as_ref(), 21.0).unwrap(), Match::Ready));
    }

    #[test]
    fn limits_are_checked_before_staging() {
        let heap = Heap::default();
        let (path, mut store) = fresh_store();
        for index in 0..MAX_PENDING {
            store_put(&mut store, &key("pending", &format!("id{index}")), &Value::null()).unwrap();
        }
        store_commit(&mut store).unwrap();
        let due = object(&heap, &[("not_before_utc", Value::number(1.0))]).unwrap();
        let input = message(&heap, "overflow", due, text("body"));
        assert_eq!(create(&mut store, &input, 2.0, &heap).unwrap_err(), LanaError::Limit);
        assert_eq!(store_get(&store, &key("pending", "overflow")).unwrap_err(), LanaError::NotFound);
        store_close(&mut store).unwrap();
        std::fs::remove_dir_all(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn uncertain_commit_is_reconciled_after_reopen() {
        use std::os::unix::fs::PermissionsExt;

        let heap = Heap::default();
        let (path, mut store) = fresh_store();
        let original = std::fs::metadata(&path).unwrap().permissions();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o500)).unwrap();
        let input = message(&heap, "uncertain", object(&heap, &[("not_before_utc", Value::number(2.0))]).unwrap(), text("saved"));
        let result = create(&mut store, &input, 1.0, &heap);
        std::fs::set_permissions(&path, original).unwrap();
        assert_eq!(result.unwrap_err(), LanaError::Io);
        assert_eq!(store_get(&store, &key("pending", "uncertain")).unwrap_err(), LanaError::InvalidState);
        let mut reopened = store_open(&StoreOptions { schema_version: 1, path: path.clone(), timeout_ms: 0 }).unwrap();
        let saved = inspect(&reopened, &text("uncertain")).unwrap();
        assert_eq!(string(&record_field(&saved, "status").unwrap()).unwrap(), "pending");
        assert_eq!(string(&record_field(&saved, "payload").unwrap()).unwrap(), "saved");
        store_close(&mut reopened).unwrap();
        std::fs::remove_dir_all(path).unwrap();
    }
}
