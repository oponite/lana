//! File-backed key/value store, mirroring `runtime/c/store.c` and
//! `runtime/include/store.h`.
//!
//! The on-disk format is byte-identical to the C11 reference: a text
//! `manifest`, an append-only `journal` of `LREV`/`LCMT` records, and
//! `snapshot-<rev>` files with an `LSNP` header. Values are stored as canonical
//! JSON (encoded with `codec::encode_value`, decoded with `data::json_parse`).

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use lana_bytecode::LanaError;
use lana_vm::value::{Map, Value};

use crate::codec;
use crate::data;
use crate::sha256;
use crate::state_codec::{PersistentState, persistent_state_decode, persistent_state_encode};

const STORE_SCHEMA: u32 = 1;
const STORE_LIMIT: usize = 256 * 1024 * 1024;

/// A staged mutation (put or delete), mirroring `StoreMutation`.
#[derive(Clone)]
struct Mutation {
    key: String,
    deleted: bool,
    data: Vec<u8>,
}

/// A committed revision, mirroring `StoreRevision`.
struct Revision {
    id: u64,
    previous: u64,
    mutations: Vec<Mutation>,
    digest: [u8; 32],
}

/// Options for opening a store, mirroring `LanaStoreOptions`.
pub struct StoreOptions {
    pub schema_version: u32,
    pub path: String,
    pub timeout_ms: u32,
}

/// Revision metadata, mirroring `LanaStoreRevisionInfo`.
#[derive(Clone)]
pub struct StoreRevisionInfo {
    pub revision_id: u64,
    pub schema_version: u32,
    pub timestamp: u64,
    pub digest: [u8; 32],
}

/// One history record, mirroring `LanaStoreHistoryRecord`.
pub struct HistoryRecord {
    pub revision: u64,
    pub key: String,
    pub value: Value,
}

/// One scan record, mirroring `LanaStoreScanRecord`.
pub struct ScanRecord {
    pub key: String,
    pub value: Value,
}

/// The store, mirroring `struct LanaStore`.
pub struct Store {
    path: String,
    index: BTreeMap<String, Vec<u8>>,
    staged: Vec<Mutation>,
    revisions: Vec<Revision>,
    current_rev: u64,
    snapshot_rev: u64,
    retention_boundary: u64,
    journal: File,
    journal_offset: u64,
}

fn decode_bytes(data: &[u8]) -> Result<Value, LanaError> {
    let text = std::str::from_utf8(data).map_err(|_| LanaError::Parse)?;
    data::json_parse(text)
}

fn store_path(store_path: &str, name: &str) -> String {
    format!("{}/{}", store_path, name)
}

fn snapshot_path(path: &str, revision: u64, temporary: bool) -> String {
    store_path(path, &format!("snapshot-{}{}", revision, if temporary { ".tmp" } else { "" }))
}

// ---------------------------------------------------------------------------
// Payload / digest helpers
// ---------------------------------------------------------------------------

fn build_payload(mutations: &[Mutation]) -> Result<Vec<u8>, LanaError> {
    let mut payload = Vec::new();
    for mutation in mutations {
        let key = mutation.key.as_bytes();
        if key.len() > u32::MAX as usize {
            return Err(LanaError::Limit);
        }
        payload.push(if mutation.deleted { b'D' } else { b'P' });
        payload.extend_from_slice(&(key.len() as u32).to_be_bytes());
        payload.extend_from_slice(key);
        if !mutation.deleted {
            let value_digest = sha256::sha256(&mutation.data);
            payload.extend_from_slice(&value_digest);
            payload.extend_from_slice(&(mutation.data.len() as u64).to_be_bytes());
            payload.extend_from_slice(&mutation.data);
        }
    }
    Ok(payload)
}

fn revision_digest(revision: u64, previous: u64, count: u32, payload: &[u8]) -> [u8; 32] {
    let mut canonical = Vec::new();
    canonical.extend_from_slice(&revision.to_be_bytes());
    canonical.extend_from_slice(&previous.to_be_bytes());
    canonical.extend_from_slice(&count.to_be_bytes());
    canonical.extend_from_slice(payload);
    sha256::sha256(&canonical)
}

fn parse_payload(data: &[u8], count: u32) -> Result<Vec<Mutation>, LanaError> {
    let mut cursor = 0usize;
    let end = data.len();
    let mut mutations = Vec::with_capacity(count as usize);
    for _ in 0..count {
        if cursor == end || (data[cursor] != b'P' && data[cursor] != b'D') {
            return Err(LanaError::Corruption);
        }
        let deleted = data[cursor] == b'D';
        cursor += 1;
        if end - cursor < 4 {
            return Err(LanaError::Corruption);
        }
        let key_length = u32::from_be_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if key_length == 0 || end - cursor < key_length {
            return Err(LanaError::Corruption);
        }
        let key = String::from_utf8(data[cursor..cursor + key_length].to_vec())
            .map_err(|_| LanaError::Corruption)?;
        cursor += key_length;
        let mut mutation = Mutation { key, deleted, data: Vec::new() };
        if !deleted {
            if end - cursor < 32 {
                return Err(LanaError::Corruption);
            }
            let expected = &data[cursor..cursor + 32];
            cursor += 32;
            if end - cursor < 8 {
                return Err(LanaError::Corruption);
            }
            let value_length = u64::from_be_bytes(data[cursor..cursor + 8].try_into().unwrap()) as usize;
            cursor += 8;
            if value_length > STORE_LIMIT || value_length > end - cursor {
                return Err(LanaError::Corruption);
            }
            let actual = sha256::sha256(&data[cursor..cursor + value_length]);
            if actual != expected {
                return Err(LanaError::Corruption);
            }
            mutation.data = data[cursor..cursor + value_length].to_vec();
            cursor += value_length;
        }
        mutations.push(mutation);
    }
    if cursor != end {
        return Err(LanaError::Corruption);
    }
    Ok(mutations)
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

fn write_manifest(store: &Store) -> Result<(), LanaError> {
    let path = store_path(&store.path, "manifest");
    let text = format!(
        "LANA_STORE {}\ncurrent {}\nsnapshot {}\nretention {}\n",
        STORE_SCHEMA, store.current_rev, store.snapshot_rev, store.retention_boundary
    );
    std::fs::write(&path, text.as_bytes()).map_err(|_| LanaError::Io)
}

fn read_manifest(store: &mut Store) -> Result<(), LanaError> {
    let path = store_path(&store.path, "manifest");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(_) => return Err(LanaError::Io),
    };
    let mut lines = text.lines();
    let header = lines.next().ok_or(LanaError::Corruption)?;
    let mut parts = header.split_whitespace();
    if parts.next() != Some("LANA_STORE") {
        return Err(LanaError::Corruption);
    }
    let schema: u32 = parts.next().and_then(|s| s.parse().ok()).ok_or(LanaError::Corruption)?;
    let current: u64 = lines.next().and_then(|l| l.strip_prefix("current ")).and_then(|s| s.parse().ok()).ok_or(LanaError::Corruption)?;
    let snapshot: u64 = lines.next().and_then(|l| l.strip_prefix("snapshot ")).and_then(|s| s.parse().ok()).ok_or(LanaError::Corruption)?;
    let retention: u64 = lines.next().and_then(|l| l.strip_prefix("retention ")).and_then(|s| s.parse().ok()).ok_or(LanaError::Corruption)?;
    if schema != STORE_SCHEMA || snapshot > current || retention > current {
        return Err(LanaError::Corruption);
    }
    store.snapshot_rev = snapshot;
    store.retention_boundary = retention;
    Ok(())
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

fn load_snapshot(store: &mut Store) -> Result<(), LanaError> {
    let path = snapshot_path(&store.path, store.snapshot_rev, false);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(LanaError::Corruption),
        Err(_) => return Err(LanaError::Io),
    };
    if bytes.len() < 4 || &bytes[..4] != b"LSNP" {
        return Err(LanaError::Corruption);
    }
    if bytes.len() < 4 + 48 {
        return Err(LanaError::Corruption);
    }
    let revision = u64::from_be_bytes(bytes[4..12].try_into().unwrap());
    let length = u64::from_be_bytes(bytes[12..20].try_into().unwrap()) as usize;
    let expected_digest = &bytes[20..52];
    if revision != store.snapshot_rev || length > STORE_LIMIT {
        return Err(LanaError::Corruption);
    }
    if bytes.len() != 4 + 48 + length {
        return Err(LanaError::Corruption);
    }
    let payload = &bytes[52..52 + length];
    let actual = sha256::sha256(payload);
    if actual != expected_digest {
        return Err(LanaError::Corruption);
    }
    let value = codec::decode_document(payload)?;
    let map = match &value.kind {
        lana_vm::value::ValueKind::Map(map) => map,
        _ => return Err(LanaError::Corruption),
    };
    let map = map.lock().unwrap();
    for entry in &map.entries {
        let encoded = codec::encode_value(&entry.value)?;
        store.index.insert(entry.key.to_string(), encoded.into_bytes());
    }
    store.current_rev = store.snapshot_rev;
    Ok(())
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

fn replay(store: &mut Store) -> Result<(), LanaError> {
    let mut journal_revision = 0u64;
    let mut saw_record = false;
    let mut offset = 0u64;
    let mut buf = [0u8; 4];
    loop {
        let start = offset;
        store.journal.seek(SeekFrom::Start(start)).map_err(|_| LanaError::Io)?;
        let read = store.journal.read(&mut buf).map_err(|_| LanaError::Io)?;
        if read == 0 {
            let ok = (!saw_record && store.snapshot_rev != 0) || journal_revision >= store.snapshot_rev;
            return if ok { Ok(()) } else { Err(LanaError::Corruption) };
        }
        if read != 4 {
            store.journal.seek(SeekFrom::Start(start)).map_err(|_| LanaError::Io)?;
            return Ok(());
        }
        if &buf != b"LREV" {
            return Err(LanaError::Corruption);
        }
        offset += 4;
        let mut header = [0u8; 28];
        let mut digest = [0u8; 32];
        store.journal.seek(SeekFrom::Start(offset)).map_err(|_| LanaError::Io)?;
        if store.journal.read(&mut header).map_err(|_| LanaError::Io)? != 28
            || store.journal.read(&mut digest).map_err(|_| LanaError::Io)? != 32
        {
            store.journal.seek(SeekFrom::Start(start)).map_err(|_| LanaError::Io)?;
            return Ok(());
        }
        offset += 28 + 32;
        let revision = u64::from_be_bytes(header[0..8].try_into().unwrap());
        let previous = u64::from_be_bytes(header[8..16].try_into().unwrap());
        let count = u32::from_be_bytes(header[16..20].try_into().unwrap());
        let payload_length = u64::from_be_bytes(header[20..28].try_into().unwrap()) as usize;
        if payload_length > STORE_LIMIT {
            return Err(LanaError::Corruption);
        }
        let mut payload = vec![0u8; payload_length];
        store.journal.seek(SeekFrom::Start(offset)).map_err(|_| LanaError::Io)?;
        if store.journal.read(&mut payload).map_err(|_| LanaError::Io)? != payload_length {
            store.journal.seek(SeekFrom::Start(start)).map_err(|_| LanaError::Io)?;
            return Ok(());
        }
        offset += payload_length as u64;
        let mut magic = [0u8; 4];
        store.journal.seek(SeekFrom::Start(offset)).map_err(|_| LanaError::Io)?;
        if store.journal.read(&mut magic).map_err(|_| LanaError::Io)? != 4 {
            store.journal.seek(SeekFrom::Start(start)).map_err(|_| LanaError::Io)?;
            return Ok(());
        }
        offset += 4;
        let mut trailer = [0u8; 8];
        let mut commit_digest = [0u8; 32];
        store.journal.seek(SeekFrom::Start(offset)).map_err(|_| LanaError::Io)?;
        if store.journal.read(&mut trailer).map_err(|_| LanaError::Io)? != 8
            || store.journal.read(&mut commit_digest).map_err(|_| LanaError::Io)? != 32
        {
            store.journal.seek(SeekFrom::Start(start)).map_err(|_| LanaError::Io)?;
            return Ok(());
        }
        offset += 8 + 32;
        if journal_revision == 0 && store.snapshot_rev != 0
            && revision == store.snapshot_rev + 1 && previous == store.snapshot_rev
        {
            journal_revision = store.snapshot_rev;
        }
        let commit_revision = u64::from_be_bytes(trailer);
        if &magic != b"LCMT"
            || commit_revision != revision
            || previous != journal_revision
            || revision != previous + 1
            || digest != commit_digest
        {
            return Err(LanaError::Corruption);
        }
        let actual = revision_digest(revision, previous, count, &payload);
        if actual != digest {
            return Err(LanaError::Corruption);
        }
        let mutations = parse_payload(&payload, count)?;
        let revision_mutations = mutations.clone();
        store.revisions.push(Revision {
            id: revision,
            previous,
            mutations: revision_mutations.clone(),
            digest,
        });
        if revision > store.snapshot_rev {
            for mutation in &mutations {
                if mutation.deleted {
                    store.index.remove(&mutation.key);
                } else {
                    store.index.insert(mutation.key.clone(), mutation.data.clone());
                }
            }
        }
        journal_revision = revision;
        saw_record = true;
        if revision > store.snapshot_rev {
            store.current_rev = revision;
        }
        store.journal_offset = offset;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn store_open(options: &StoreOptions) -> Result<Store, LanaError> {
    if options.path.is_empty() {
        return Err(LanaError::InvalidState);
    }
    if options.schema_version != STORE_SCHEMA {
        return Err(LanaError::IncompatibleFormat);
    }
    std::fs::create_dir_all(&options.path).map_err(|_| LanaError::Io)?;
    let journal_path = store_path(&options.path, "journal");
    let journal = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&journal_path)
        .map_err(|_| LanaError::Io)?;
    let mut store = Store {
        path: options.path.clone(),
        index: BTreeMap::new(),
        staged: Vec::new(),
        revisions: Vec::new(),
        current_rev: 0,
        snapshot_rev: 0,
        retention_boundary: 0,
        journal,
        journal_offset: 0,
    };
    read_manifest(&mut store)?;
    if store.snapshot_rev != 0 {
        load_snapshot(&mut store)?;
    }
    replay(&mut store)?;
    store.journal_offset = store.journal.seek(SeekFrom::End(0)).map_err(|_| LanaError::Io)?;
    write_manifest(&store)?;
    Ok(store)
}

pub fn store_close(_store: &mut Store) -> Result<(), LanaError> {
    Ok(())
}

pub fn store_get_path(store: &Store) -> Result<String, LanaError> {
    Ok(store.path.clone())
}

fn stage(store: &mut Store, key: &str, value: Option<&Value>) -> Result<(), LanaError> {
    if key.is_empty() {
        return Err(LanaError::Key);
    }
    let deleted = value.is_none();
    let data = match value {
        Some(value) => codec::encode_value(value)?.into_bytes(),
        None => Vec::new(),
    };
    store.staged.push(Mutation { key: key.to_string(), deleted, data });
    Ok(())
}

pub fn store_put(store: &mut Store, key: &str, value: &Value) -> Result<(), LanaError> {
    stage(store, key, Some(value))
}

pub fn store_delete(store: &mut Store, key: &str) -> Result<(), LanaError> {
    stage(store, key, None)
}

pub fn store_commit(store: &mut Store) -> Result<StoreRevisionInfo, LanaError> {
    if store.staged.is_empty() {
        return Err(LanaError::UnsupportedOperation);
    }
    let payload = build_payload(&store.staged)?;
    let revision = store.current_rev + 1;
    let digest = revision_digest(revision, store.current_rev, store.staged.len() as u32, &payload);
    let mut header = Vec::new();
    header.extend_from_slice(b"LREV");
    header.extend_from_slice(&revision.to_be_bytes());
    header.extend_from_slice(&store.current_rev.to_be_bytes());
    header.extend_from_slice(&(store.staged.len() as u32).to_be_bytes());
    header.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    header.extend_from_slice(&digest);
    let mut trailer = Vec::new();
    trailer.extend_from_slice(&revision.to_be_bytes());
    trailer.extend_from_slice(&digest);

    store.journal.set_len(store.journal_offset).map_err(|_| LanaError::Io)?;
    store.journal.seek(SeekFrom::Start(store.journal_offset)).map_err(|_| LanaError::Io)?;
    store.journal.write_all(&header).map_err(|_| LanaError::Io)?;
    store.journal.write_all(&payload).map_err(|_| LanaError::Io)?;
    store.journal.write_all(b"LCMT").map_err(|_| LanaError::Io)?;
    store.journal.write_all(&trailer).map_err(|_| LanaError::Io)?;
    store.journal.flush().map_err(|_| LanaError::Io)?;
    store.journal.sync_all().map_err(|_| LanaError::Io)?;

    let mutations = store.staged.clone();
    store.revisions.push(Revision {
        id: revision,
        previous: store.current_rev,
        mutations: mutations.clone(),
        digest,
    });
    for mutation in &mutations {
        if mutation.deleted {
            store.index.remove(&mutation.key);
        } else {
            store.index.insert(mutation.key.clone(), mutation.data.clone());
        }
    }
    store.current_rev = revision;
    store.journal_offset = store.journal.seek(SeekFrom::End(0)).map_err(|_| LanaError::Io)?;
    store.staged.clear();
    write_manifest(store)?;
    Ok(StoreRevisionInfo {
        revision_id: revision,
        schema_version: STORE_SCHEMA,
        timestamp: 0,
        digest,
    })
}

pub fn store_current_revision(store: &Store) -> Result<StoreRevisionInfo, LanaError> {
    let digest = store.revisions.last().map(|r| r.digest).unwrap_or([0u8; 32]);
    Ok(StoreRevisionInfo {
        revision_id: store.current_rev,
        schema_version: STORE_SCHEMA,
        timestamp: 0,
        digest,
    })
}

pub fn store_get(store: &Store, key: &str) -> Result<Value, LanaError> {
    match store.index.get(key) {
        Some(data) => decode_bytes(data),
        None => Err(LanaError::NotFound),
    }
}

pub fn store_get_at(store: &Store, revision: u64, key: &str) -> Result<Value, LanaError> {
    if revision > store.current_rev {
        return Err(LanaError::NotFound);
    }
    if revision < store.retention_boundary {
        return Err(LanaError::CompactedHistory);
    }
    if revision == store.current_rev {
        return store_get(store, key);
    }
    let mut latest: Option<&Mutation> = None;
    for rev in &store.revisions {
        if rev.id > revision {
            break;
        }
        for mutation in &rev.mutations {
            if mutation.key == key {
                latest = Some(mutation);
            }
        }
    }
    match latest {
        Some(mutation) if !mutation.deleted => decode_bytes(&mutation.data),
        _ => Err(LanaError::NotFound),
    }
}

pub fn store_put_persistent_state(store: &mut Store, key: &str, state: &PersistentState) -> Result<(), LanaError> {
    let encoded = persistent_state_encode(state)?;
    let value = codec::decode_document(&encoded)?;
    store_put(store, key, &value)
}

fn persistent_state_from_value(value: &Value) -> Result<PersistentState, LanaError> {
    let encoded = codec::encode_value(value)?;
    persistent_state_decode(encoded.as_bytes())
}

pub fn store_get_persistent_state(store: &Store, key: &str) -> Result<PersistentState, LanaError> {
    let value = store_get(store, key)?;
    persistent_state_from_value(&value)
}

pub fn store_get_persistent_state_at(store: &Store, revision: u64, key: &str) -> Result<PersistentState, LanaError> {
    let value = store_get_at(store, revision, key)?;
    persistent_state_from_value(&value)
}

pub fn store_history(store: &Store, key: &str) -> Result<Vec<HistoryRecord>, LanaError> {
    let mut records = Vec::new();
    for rev in &store.revisions {
        for mutation in &rev.mutations {
            if mutation.key != key {
                continue;
            }
            let value = if mutation.deleted {
                Value::null()
            } else {
                codec::decode_document(&mutation.data)?
            };
            records.push(HistoryRecord { revision: rev.id, key: key.to_string(), value });
        }
    }
    if records.is_empty() {
        return Err(LanaError::NotFound);
    }
    Ok(records)
}

pub fn store_scan(store: &Store, prefix: &str) -> Result<Vec<ScanRecord>, LanaError> {
    let mut records = Vec::new();
    for (key, data) in store.index.range(prefix.to_string()..) {
        if !key.starts_with(prefix) {
            break;
        }
        records.push(ScanRecord { key: key.clone(), value: decode_bytes(data)? });
    }
    Ok(records)
}

pub fn store_snapshot(store: &mut Store) -> Result<(Value, StoreRevisionInfo), LanaError> {
    let mut map = Map::new(store.index.len());
    for (key, data) in &store.index {
        let value = decode_bytes(data)?;
        map.set(Arc::from(key.as_str()), value, true).map_err(|_| LanaError::Key)?;
    }
    let out_value = Value::map(Arc::new(Mutex::new(map)));
    let encoded = codec::encode_value(&out_value)?;
    let digest = sha256::sha256(encoded.as_bytes());

    let mut header = Vec::new();
    header.extend_from_slice(b"LSNP");
    header.extend_from_slice(&store.current_rev.to_be_bytes());
    header.extend_from_slice(&(encoded.len() as u64).to_be_bytes());
    header.extend_from_slice(&digest);

    let temporary = snapshot_path(&store.path, store.current_rev, true);
    let published = snapshot_path(&store.path, store.current_rev, false);
    let mut file = File::create(&temporary).map_err(|_| LanaError::Io)?;
    file.write_all(&header).map_err(|_| LanaError::Io)?;
    file.write_all(encoded.as_bytes()).map_err(|_| LanaError::Io)?;
    file.flush().map_err(|_| LanaError::Io)?;
    file.sync_all().map_err(|_| LanaError::Io)?;
    drop(file);
    std::fs::rename(&temporary, &published).map_err(|_| LanaError::Io)?;

    store.snapshot_rev = store.current_rev;
    write_manifest(store)?;
    Ok((
        out_value,
        StoreRevisionInfo {
            revision_id: store.current_rev,
            schema_version: STORE_SCHEMA,
            timestamp: 0,
            digest,
        },
    ))
}

pub fn store_compact(store: &mut Store, retention: u64) -> Result<StoreRevisionInfo, LanaError> {
    if retention == 0 {
        store_snapshot(store)?;
    }
    let boundary = if retention >= store.current_rev { 0 } else { store.current_rev - retention };
    store.retention_boundary = boundary;
    if retention == 0 {
        store.journal.set_len(0).map_err(|_| LanaError::Io)?;
        store.journal.seek(SeekFrom::Start(0)).map_err(|_| LanaError::Io)?;
        store.journal.sync_all().map_err(|_| LanaError::Io)?;
        store.journal_offset = 0;
        store.revisions.clear();
    }
    write_manifest(store)?;
    let digest = store.revisions.last().map(|r| r.digest).unwrap_or([0u8; 32]);
    Ok(StoreRevisionInfo {
        revision_id: store.current_rev,
        schema_version: STORE_SCHEMA,
        timestamp: 0,
        digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("lana_store_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.to_str().unwrap().to_string()
    }

    #[test]
    fn put_get_round_trip() {
        let path = temp_store("roundtrip");
        let mut store = store_open(&StoreOptions { schema_version: 1, path: path.clone(), timeout_ms: 0 }).unwrap();
        let value = Value::number(3.14);
        store_put(&mut store, "k", &value).unwrap();
        store_commit(&mut store).unwrap();
        let got = store_get(&store, "k").unwrap();
        assert_eq!(got.as_number(), 3.14);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn reopen_persists() {
        let path = temp_store("reopen");
        {
            let mut store = store_open(&StoreOptions { schema_version: 1, path: path.clone(), timeout_ms: 0 }).unwrap();
            store_put(&mut store, "a", &Value::string(Arc::from("x"))).unwrap();
            store_commit(&mut store).unwrap();
        }
        {
            let store = store_open(&StoreOptions { schema_version: 1, path: path.clone(), timeout_ms: 0 }).unwrap();
            let got = store_get(&store, "a").unwrap();
            assert_eq!(got.as_string(), Arc::from("x"));
        }
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn delete_and_history() {
        let path = temp_store("history");
        let mut store = store_open(&StoreOptions { schema_version: 1, path: path.clone(), timeout_ms: 0 }).unwrap();
        store_put(&mut store, "k", &Value::number(1.0)).unwrap();
        store_commit(&mut store).unwrap();
        store_put(&mut store, "k", &Value::number(2.0)).unwrap();
        store_commit(&mut store).unwrap();
        store_delete(&mut store, "k").unwrap();
        store_commit(&mut store).unwrap();
        assert_eq!(store_get(&store, "k").unwrap_err(), LanaError::NotFound);
        let history = store_history(&store, "k").unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].value.as_number(), 1.0);
        assert_eq!(history[1].value.as_number(), 2.0);
        let _ = std::fs::remove_dir_all(&path);
    }
}
