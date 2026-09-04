//! Append-only event ledger, mirroring `runtime/c/ledger.c` and
//! `runtime/include/ledger.h`.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Arc, Mutex};

use lana_bytecode::LanaError;
use lana_vm::value::{Map, Value, ValueKind};

use crate::data::json_parse;
use crate::effects::EffectStatus;
use crate::policy::PolicyOutcome;
use crate::store::{ScanRecord, Store, store_commit, store_current_revision, store_get, store_get_path, store_put, store_scan};

/// Input to `ledger_append`, mirroring `LanaEventInput`.
pub struct EventInput {
    pub schema_version: u32,
    pub entity: Arc<str>,
    pub actor: Arc<str>,
    pub action: Arc<str>,
    pub reason: Option<Arc<str>>,
    pub timestamp: u64,
    pub correction_of: u64,
}

/// A stored event, mirroring `LanaEvent`.
pub struct Event {
    pub schema_version: u32,
    pub event_id: u64,
    pub entity: Arc<str>,
    pub actor: Arc<str>,
    pub action: Arc<str>,
    pub reason: Arc<str>,
    pub timestamp: u64,
    pub revision: u64,
    pub correction_of: u64,
}

/// A query filter, mirroring `LanaLedgerQuery`.
pub struct LedgerQuery {
    pub schema_version: u32,
    pub entity: Option<Arc<str>>,
    pub actor: Option<Arc<str>>,
    pub action: Option<Arc<str>>,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
}

/// A coverage query, mirroring `LanaLedgerCoverageQuery`.
pub struct CoverageQuery {
    pub schema_version: u32,
    pub entity: Option<Arc<str>>,
    pub actor: Option<Arc<str>>,
    pub action: Option<Arc<str>>,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub target_entities: Vec<Arc<str>>,
}

/// One coverage result, mirroring `LanaLedgerCoverageEntry`.
pub struct CoverageEntry {
    pub schema_version: u32,
    pub entity: Arc<str>,
    pub matched: bool,
    pub events: Vec<Event>,
}

/// An evidence-to-event traversal query, mirroring `LanaTraversalQuery`.
pub struct TraversalQuery {
    pub schema_version: u32,
    pub evidence_id: Arc<str>,
}

/// One traversal record, mirroring `LanaTraversalRecord`.
pub struct TraversalRecord {
    pub schema_version: u32,
    pub decision_id: u64,
    pub policy_id: Arc<str>,
    pub policy_version: [u8; 32],
    pub outcome: PolicyOutcome,
    pub effect: Arc<str>,
    pub attempt_id: u64,
    pub attempt_status: EffectStatus,
    pub event_id: u64,
}

/// The ledger, mirroring `struct LanaLedger`. Borrows the store it appends to.
pub struct Ledger<'a> {
    store: &'a mut Store,
    journal: Option<File>,
}

fn text_ok(text: &str) -> bool {
    !text.is_empty()
}

fn map_text(map: &Map, key: &str) -> Result<Arc<str>, LanaError> {
    match map.get(key) {
        Some(Value { kind: ValueKind::String(s), .. }) if !s.is_empty() => Ok(s.clone()),
        _ => Err(LanaError::Corruption),
    }
}

fn map_number(map: &Map, key: &str) -> Result<u64, LanaError> {
    match map.get(key) {
        Some(Value { kind: ValueKind::Number(n), .. }) => Ok(*n as u64),
        _ => Err(LanaError::Corruption),
    }
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

fn key_tail_id(key: &str) -> u64 {
    match key.rfind('/') {
        Some(pos) if pos + 1 < key.len() => key[pos + 1..].parse().unwrap_or(0),
        _ => 0,
    }
}

/// Decode a stored record (JSON text or an already-parsed map) to a map.
fn scan_value_map(value: &Value) -> Result<Arc<Mutex<Map>>, LanaError> {
    match &value.kind {
        ValueKind::Map(map) => Ok(map.clone()),
        ValueKind::String(s) => {
            let parsed = json_parse(s)?;
            match parsed.kind {
                ValueKind::Map(map) => Ok(map),
                _ => Err(LanaError::Corruption),
            }
        }
        _ => Err(LanaError::Corruption),
    }
}

pub fn ledger_open(store: &mut Store) -> Result<Ledger<'_>, LanaError> {
    let path = store_get_path(store)?;
    let journal_path = format!("{}/ledger.journal", path);
    let journal = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(&journal_path)
        .map_err(|_| LanaError::Io)?;
    Ok(Ledger { store, journal: Some(journal) })
}

pub fn ledger_close(ledger: &mut Ledger<'_>) -> Result<(), LanaError> {
    ledger.journal = None;
    Ok(())
}

pub fn ledger_append(ledger: &mut Ledger<'_>, input: &EventInput) -> Result<Event, LanaError> {
    if input.schema_version != 1
        || !text_ok(&input.entity)
        || !text_ok(&input.actor)
        || !text_ok(&input.action)
    {
        return Err(LanaError::Schema);
    }
    if input.reason.as_deref().map_or(false, |r| r.contains('"')) {
        return Err(LanaError::Schema);
    }
    let revision = store_current_revision(ledger.store)?;
    if revision.revision_id == u64::MAX {
        return Err(LanaError::Io);
    }
    let key = format!("event/{}", revision.revision_id + 1);
    let json = format!(
        "{{\"action\":\"{}\",\"actor\":\"{}\",\"correction_of\":{},\"entity\":\"{}\",\"reason\":\"{}\",\"timestamp\":{}}}",
        input.action,
        input.actor,
        input.correction_of,
        input.entity,
        input.reason.as_deref().unwrap_or(""),
        input.timestamp
    );
    if let Some(journal) = ledger.journal.as_mut() {
        writeln!(journal, "{}", json).map_err(|_| LanaError::Io)?;
        journal.flush().map_err(|_| LanaError::Io)?;
    }
    store_put(ledger.store, &key, &Value::string(Arc::from(json.as_str())))?;
    let committed = store_commit(ledger.store)?;
    Ok(Event {
        schema_version: 1,
        event_id: committed.revision_id,
        entity: input.entity.clone(),
        actor: input.actor.clone(),
        action: input.action.clone(),
        reason: input.reason.clone().unwrap_or_else(|| Arc::from("")),
        timestamp: input.timestamp,
        revision: committed.revision_id,
        correction_of: input.correction_of,
    })
}

fn scan_events(
    store: &Store,
    entity: Option<&str>,
    actor: Option<&str>,
    action: Option<&str>,
    start_timestamp: u64,
    end_timestamp: u64,
) -> Result<Vec<Event>, LanaError> {
    let revision = store_current_revision(store)?;
    let mut events = Vec::new();
    for index in 1..=revision.revision_id {
        let key = format!("event/{}", index);
        let raw = match store_get(store, &key) {
            Ok(raw) => raw,
            Err(LanaError::NotFound) => continue,
            Err(e) => return Err(e),
        };
        let map = match &raw.kind {
            ValueKind::String(s) => {
                let parsed = json_parse(s)?;
                match parsed.kind {
                    ValueKind::Map(map) => map,
                    _ => return Err(LanaError::Corruption),
                }
            }
            _ => return Err(LanaError::Corruption),
        };
        let map = map.lock().unwrap();
        let event_entity = map_text(&map, "entity")?;
        let event_actor = map_text(&map, "actor")?;
        let event_action = map_text(&map, "action")?;
        let event_reason = map_text(&map, "reason")?;
        let timestamp = map_number(&map, "timestamp")?;
        if entity.map_or(false, |e| e != &*event_entity) {
            continue;
        }
        if actor.map_or(false, |a| a != &*event_actor) {
            continue;
        }
        if action.map_or(false, |a| a != &*event_action) {
            continue;
        }
        if start_timestamp != 0 && timestamp < start_timestamp {
            continue;
        }
        if end_timestamp != 0 && timestamp > end_timestamp {
            continue;
        }
        events.push(Event {
            schema_version: 1,
            event_id: index,
            entity: event_entity,
            actor: event_actor,
            action: event_action,
            reason: event_reason,
            timestamp,
            revision: index,
            correction_of: 0,
        });
    }
    Ok(events)
}

pub fn ledger_query(ledger: &Ledger<'_>, query: &LedgerQuery) -> Result<Vec<Event>, LanaError> {
    if query.schema_version != 1
        || (query.start_timestamp > query.end_timestamp && query.end_timestamp != 0)
    {
        return Err(LanaError::InvalidState);
    }
    let events = scan_events(
        ledger.store,
        query.entity.as_deref(),
        query.actor.as_deref(),
        query.action.as_deref(),
        query.start_timestamp,
        query.end_timestamp,
    )?;
    if events.is_empty() {
        return Err(LanaError::NoMatchingEvent);
    }
    Ok(events)
}

pub fn ledger_query_coverage(
    ledger: &Ledger<'_>,
    query: &CoverageQuery,
) -> Result<Vec<CoverageEntry>, LanaError> {
    if query.schema_version != 1
        || query.target_entities.is_empty()
        || (query.start_timestamp > query.end_timestamp && query.end_timestamp != 0)
    {
        return Err(LanaError::InvalidState);
    }
    let mut entries = Vec::with_capacity(query.target_entities.len());
    for target in &query.target_entities {
        if target.is_empty() {
            return Err(LanaError::Schema);
        }
        let events = scan_events(
            ledger.store,
            Some(target),
            query.actor.as_deref(),
            query.action.as_deref(),
            query.start_timestamp,
            query.end_timestamp,
        )?;
        entries.push(CoverageEntry {
            schema_version: 1,
            entity: target.clone(),
            matched: !events.is_empty(),
            events,
        });
    }
    Ok(entries)
}

pub fn ledger_traverse(
    ledger: &Ledger<'_>,
    query: &TraversalQuery,
) -> Result<Vec<TraversalRecord>, LanaError> {
    if query.schema_version != 1 || query.evidence_id.is_empty() {
        return Err(LanaError::InvalidState);
    }
    let decisions: Vec<ScanRecord> = store_scan(ledger.store, "decision/")?;
    let events: Vec<ScanRecord> = store_scan(ledger.store, "event/")?;
    let mut records: Vec<TraversalRecord> = Vec::new();
    for decision in &decisions {
        let parsed = scan_value_map(&decision.value)?;
        let parsed = parsed.lock().unwrap();
        let evidence_ids = map_text(&parsed, "evidence_ids")?;
        if !evidence_ids.contains(&*query.evidence_id) {
            continue;
        }
        let decision_id = map_number(&parsed, "decision_id")?;
        let policy_id = map_text(&parsed, "policy_id")?;
        let policy_version_text = map_text(&parsed, "policy_version")?;
        let policy_version = hex_decode(&policy_version_text)?;
        let outcome_value = map_number(&parsed, "outcome")?;
        let effect = map_text(&parsed, "effect")?;
        drop(parsed);

        let attempt_prefix = format!("effect-attempt/{}/", decision_id);
        let attempts: Vec<ScanRecord> = store_scan(ledger.store, &attempt_prefix)?;
        let mut attempt_id = 0u64;
        let mut attempt_status = EffectStatus::Pending;
        if let Some(first) = attempts.first() {
            let attempt_parsed = scan_value_map(&first.value)?;
            let attempt_parsed = attempt_parsed.lock().unwrap();
            let status_value = map_number(&attempt_parsed, "status")?;
            attempt_id = key_tail_id(&first.key);
            attempt_status = match status_value {
                0 => EffectStatus::Pending,
                1 => EffectStatus::Succeeded,
                2 => EffectStatus::Failed,
                3 => EffectStatus::Cancelled,
                _ => return Err(LanaError::Corruption),
            };
        }

        let decision_text = decision_id.to_string();
        let mut event_id = 0u64;
        for event in &events {
            let event_parsed = scan_value_map(&event.value)?;
            let event_parsed = event_parsed.lock().unwrap();
            let reason = map_text(&event_parsed, "reason")?;
            if reason.contains(&decision_text) {
                event_id = key_tail_id(&event.key);
                break;
            }
        }

        records.push(TraversalRecord {
            schema_version: 1,
            decision_id,
            policy_id,
            policy_version,
            outcome: match outcome_value {
                0 => PolicyOutcome::Authorize,
                1 => PolicyOutcome::Refuse,
                2 => PolicyOutcome::RequestMoreEvidence,
                _ => return Err(LanaError::Corruption),
            },
            effect,
            attempt_id,
            attempt_status,
            event_id,
        });
    }
    if records.is_empty() {
        return Err(LanaError::NoMatchingEvent);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("lana_ledger_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.to_str().unwrap().to_string()
    }

    #[test]
    fn append_and_query() {
        let path = temp_store("append");
        let mut store = crate::store::store_open(&crate::store::StoreOptions {
            schema_version: 1,
            path: path.clone(),
            timeout_ms: 0,
        })
        .unwrap();
        let mut ledger = ledger_open(&mut store).unwrap();
        let event = ledger_append(
            &mut ledger,
            &EventInput {
                schema_version: 1,
                entity: Arc::from("e1"),
                actor: Arc::from("a1"),
                action: Arc::from("grant"),
                reason: Some(Arc::from("approved")),
                timestamp: 100,
                correction_of: 0,
            },
        )
        .unwrap();
        assert_eq!(event.event_id, 1);
        let events = ledger_query(
            &ledger,
            &LedgerQuery {
                schema_version: 1,
                entity: Some(Arc::from("e1")),
                actor: None,
                action: None,
                start_timestamp: 0,
                end_timestamp: 0,
            },
        )
        .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].action, Arc::from("grant"));
        ledger_close(&mut ledger).unwrap();
        let _ = std::fs::remove_dir_all(&path);
    }
}
