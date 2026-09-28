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

use std::sync::{Arc, Mutex, Condvar};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use lana_bytecode::LanaError;
use lana_vm::value::{Array, Dataset, DatasetOp, Map, Value, ValueKind};
use lana_vm::Vm;
use lana_vm::{
    LANA_HOST_ADAPTER_FETCH, LANA_HOST_ADAPTER_LOAD, LANA_HOST_LEDGER_APPEND,
    LANA_HOST_LEDGER_QUERY, LANA_HOST_POLICY_EVALUATE, LANA_HOST_POLICY_STORE_DECISION,
    LANA_HOST_STORE_COMMIT, LANA_HOST_STORE_COMMIT_IF, LANA_HOST_STORE_CURRENT_REVISION,
    LANA_HOST_STORE_DELETE, LANA_HOST_STORE_GET, LANA_HOST_STORE_GET_AT, LANA_HOST_STORE_OPEN,
    LANA_HOST_STORE_PUT, LANA_HOST_STORE_SCAN, LANA_HOST_STORE_SNAPSHOT,
    LANA_HOST_EXECUTION_CAPABILITY, LANA_HOST_EXECUTION_AUTHORIZE, LANA_HOST_EXECUTION_EXECUTE,
    LANA_HOST_FUTURE_MESSAGE,
    LANA_HOST_DATASET_SOURCE, LANA_HOST_DATASET_QUERY, LANA_HOST_DATASET_APPLY,
    LANA_HOST_DATASET_SNAPSHOT, LANA_HOST_DATASET_EVIDENCE, LANA_HOST_DATASET_EXCLUSIONS,
    LANA_HOST_DOCUMENT_EXTRACT, LANA_HOST_DATASET_SQLITE, LANA_HOST_RULES_LEARN, LANA_HOST_RULES_PREDICT,
    LANA_HOST_RULES_SAVE, LANA_HOST_RULES_ADD_COUNTEREXAMPLE, LANA_HOST_RULES_INSPECT,
    LANA_HOST_RULES_ROLLBACK,
    LANA_HOST_TREES_FIT, LANA_HOST_TREES_PREDICT, LANA_HOST_TREES_EXPLAIN,
    LANA_HOST_TREES_SAVE, LANA_HOST_TREES_LOAD,
};
use serde::{Deserialize, Serialize};

use crate::adapters::{self, Adapter, AdapterKind, AdapterOptions};
use crate::ledger::{self, Event, EventInput, LedgerQuery};
use crate::policy::{self, Decision, Policy, PolicyEvaluation, PolicyOutcome, PolicyRule, PolicyRuleKind};
use crate::store::{self, Store, StoreOptions};
use crate::execution::{self, Authorization, CurlTransport, ExecutionCapability};
use crate::future_messages;
use crate::dataset_identity::DatasetIdentity;
use crate::dataset_snapshot::DatasetSnapshot;

static NEXT_EXECUTION_TOKEN: AtomicU64 = AtomicU64::new(1);

struct ExecutionAuthorization {
    token: Arc<lana_vm::value::CapabilityToken>,
    authorization: Authorization,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetQueryRecord {
    schema_version: u32,
    format: String,
    query_id: String,
    source_ids: Vec<String>,
    plan_name: String,
    calculation_version: String,
    plan_digest: String,
    source_revision: String,
    snapshot_revision: String,
}

fn load_dataset_query_record(store: &Store, key: &str, value: Value,
    current: u64) -> Result<DatasetQueryRecord, LanaError> {
    let ValueKind::String(saved) = value.kind else { return Err(LanaError::Corruption); };
    let record: DatasetQueryRecord = serde_json::from_str(&saved).map_err(|_| LanaError::Corruption)?;
    let hex = record.query_id.as_bytes().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    let expected_key = format!("dataset/query/{hex}");
    if record.schema_version != 1 || record.format != "dataset_query_v1" || key != expected_key
        || record.query_id.is_empty() || record.query_id.len() > 128
        || record.plan_name.is_empty() || record.plan_name.len() > 128
        || record.calculation_version.is_empty()
        || record.source_ids.iter().any(|id| id.is_empty() || id.len() > 128)
        || crate::information_codec::canonical(&record).ok().as_deref() != Some(saved.as_bytes()) {
        return Err(LanaError::Corruption);
    }
    let source_revision = crate::information_codec::revision(&record.source_revision)
        .map_err(|_| LanaError::Corruption)?;
    let snapshot_revision = crate::information_codec::revision(&record.snapshot_revision)
        .map_err(|_| LanaError::Corruption)?;
    if snapshot_revision > current || snapshot_revision < source_revision { return Err(LanaError::Corruption); }
    let snapshot_key = format!("dataset/snapshot/{hex}/{snapshot_revision}");
    let snapshot = store::store_get_at(store, current, &snapshot_key)
        .map_err(|_| LanaError::Corruption)?;
    let ValueKind::String(snapshot) = snapshot.kind else { return Err(LanaError::Corruption); };
    let snapshot = DatasetSnapshot::load(snapshot.as_bytes(), &record.query_id,
        source_revision, &record.plan_digest).map_err(|error|
            if matches!(error, LanaError::Schema | LanaError::Limit | LanaError::Oom) { error } else { LanaError::Corruption })?;
    if snapshot.calculation_version != record.calculation_version { return Err(LanaError::Corruption); }
    Ok(record)
}

/// Owns the single durable store the VM drives, and dispatches the
/// store/policy/ledger host calls against it.
pub struct StoreHost {
    store: Option<Store>,
    bound_queries: HashMap<String, DatasetQueryRecord>,
    source_relationships: crate::dataset_source_codec::Registry,
    chunk_bytes: Option<Vec<u8>>,
    adapter: Option<Adapter>,
    heap: lana_vm::heap::Heap,
    execution: Option<(Arc<lana_vm::value::CapabilityToken>, ExecutionCapability)>,
    execution_credential: Option<Arc<str>>,
    execution_ca_file: Option<Arc<str>>,
    authorizations: Vec<ExecutionAuthorization>,
    issued_decisions: Vec<(Arc<Mutex<Map>>, [u8; 32])>,
}

impl StoreHost {
    pub fn new() -> Self {
        Self::with_heap(lana_vm::heap::Heap::new(256 * 1024 * 1024))
    }

    pub fn with_heap(heap: lana_vm::heap::Heap) -> Self {
        Self { source_relationships: Default::default(), store: None, bound_queries: HashMap::new(), chunk_bytes: None, adapter: None, heap, execution: None, execution_credential: None, execution_ca_file: None, authorizations: Vec::new(), issued_decisions: Vec::new() }
    }

    /// Keep the exact verified, linked LABC bytes used to run this VM.
    pub fn set_chunk_bytes(&mut self, bytes: Vec<u8>) {
        self.chunk_bytes = Some(bytes);
    }

    pub fn dataset_plan_digest(&self, plan_name: &str) -> Result<String, LanaError> {
        if plan_name.is_empty() || plan_name.len() > 128 { return Err(LanaError::InvalidParameters); }
        let bytes = self.chunk_bytes.as_deref().ok_or(LanaError::InvalidState)?;
        let mut hash = crate::sha256::Sha256::new();
        hash.update(bytes);
        hash.update(&(plan_name.len() as u32).to_le_bytes());
        hash.update(plan_name.as_bytes());
        let mut digest = [0; 32];
        hash.finalize(&mut digest);
        Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    /// Evaluate registered source records at one revision without publishing.
    /// Row IDs stay separate from row payloads for the later evidence pass.
    pub fn evaluate_dataset_plan(&self, vm: &mut Vm, source_ids: &[String], plan_name: &str)
        -> Result<(u64, Vec<Vec<String>>, Value), LanaError> {
        let store = self.store.as_ref().ok_or(LanaError::InvalidState)?;
        store.ensure_clean()?;
        let revision = store::store_current_revision(store)?.revision_id;
        self.evaluate_dataset_plan_at(vm, source_ids, plan_name, revision, None)
    }

    fn evaluate_dataset_plan_at(&self, vm: &mut Vm, source_ids: &[String], plan_name: &str,
        revision: u64, replacement: Option<(&str, &Value)>)
        -> Result<(u64, Vec<Vec<String>>, Value), LanaError> {
        vm.clear_dataset_decisions();
        if plan_name.is_empty() || plan_name.len() > 128
            || source_ids.iter().any(|id| id.is_empty() || id.len() > 128) {
            return Err(LanaError::InvalidParameters);
        }
        let store = self.store.as_ref().ok_or(LanaError::InvalidState)?;
        store.ensure_clean()?;
        let committed = store::store_current_revision(store)?.revision_id;
        if revision != committed && committed.checked_add(1) != Some(revision) { return Err(LanaError::Conflict); }
        let mut decoder = crate::dataset_source_codec::Decoder::default();
        let mut inputs = Vec::with_capacity(source_ids.len());
        let mut source_row_ids = Vec::with_capacity(source_ids.len());
        for source_id in source_ids {
            let hex = source_id.as_bytes().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
            let source = if replacement.as_ref().is_some_and(|(id, _)| *id == source_id) {
                replacement.unwrap().1.clone()
            } else {
                store::store_get_at(store, committed, &format!("dataset/source/{hex}"))?
            };
            let ValueKind::Map(record) = source.kind else { return Err(LanaError::Corruption); };
            let record = record.lock().unwrap();
            if record.entries().len() != 5
                || !matches!(record.get("schema_version"), Some(Value { kind: ValueKind::Number(1.0), .. }))
                || !matches!(record.get("format"), Some(Value { kind: ValueKind::String(value), .. }) if value.as_ref() == "dataset_source_v1")
                || !matches!(record.get("source_id"), Some(Value { kind: ValueKind::String(value), .. }) if value.as_ref() == source_id) {
                return Err(LanaError::Corruption);
            }
            let (Some(Value { kind: ValueKind::Array(ids), .. }), Some(Value { kind: ValueKind::Array(rows), .. })) =
                (record.get("row_ids"), record.get("rows")) else { return Err(LanaError::Corruption); };
            let ids = ids.lock().unwrap().items().to_vec();
            let rows = rows.lock().unwrap().items().to_vec();
            if ids.len() != rows.len() || rows.len() > 10_000 { return Err(LanaError::Corruption); }
            let mut seen = HashSet::new();
            let mut decoded = Vec::with_capacity(rows.len());
            let mut labels = Vec::with_capacity(ids.len());
            for (id, row) in ids.iter().zip(&rows) {
                let (ValueKind::String(id), ValueKind::String(row)) = (&id.kind, &row.kind) else { return Err(LanaError::Corruption); };
                if id.is_empty() || id.len() > 128 || !seen.insert(id.as_ref()) { return Err(LanaError::Corruption); }
                labels.push(id.to_string());
                let value = decoder.row(row.as_bytes(), vm)?;
                decoded.push(vm.dataset_source_row(source_id, id, value)?);
            }
            drop(record);
            let rows = crate::information_codec::array(vm, decoded)?;
            inputs.push(vm.dataset_value(Dataset {
                op: DatasetOp::Source, source: rows, function: 0,
                columns: Value::null(), key: Value::null(), limit: Value::null(),
                other: Value::null(), aggregate: Value::null(),
            })?);
            source_row_ids.push(labels);
        }
        let result = vm.run_pure_dataset_plan(plan_name, &inputs)?;
        Ok((revision, source_row_ids, result))
    }

    /// Evaluate and assign stable row/derivation IDs before any publication.
    pub fn evaluate_dataset_plan_with_identity(&self, vm: &mut Vm, query_id: &str,
        source_ids: &[String], plan_name: &str)
        -> Result<(u64, Vec<Vec<String>>, Value, DatasetIdentity), LanaError> {
        let digest = self.dataset_plan_digest(plan_name)?;
        let (revision, source_row_ids, result) = self.evaluate_dataset_plan(vm, source_ids, plan_name)?;
        let identity = DatasetIdentity::from_run(query_id, &digest, revision, &result, vm.dataset_decisions())?;
        Ok((revision, source_row_ids, result, identity))
    }

    /// Build and validate canonical snapshot bytes without publishing them.
    pub fn evaluate_dataset_plan_snapshot(&self, vm: &mut Vm, query_id: &str,
        source_ids: &[String], plan_name: &str, calculation_version: &str)
        -> Result<(u64, Vec<Vec<String>>, Vec<u8>), LanaError> {
        let store = self.store.as_ref().ok_or(LanaError::InvalidState)?;
        store.ensure_clean()?;
        let revision = store::store_current_revision(store)?.revision_id;
        self.evaluate_dataset_plan_snapshot_at(vm, query_id, source_ids, plan_name,
            calculation_version, revision, None)
    }

    fn evaluate_dataset_plan_snapshot_at(&self, vm: &mut Vm, query_id: &str,
        source_ids: &[String], plan_name: &str, calculation_version: &str, revision: u64,
        replacement: Option<(&str, &Value)>)
        -> Result<(u64, Vec<Vec<String>>, Vec<u8>), LanaError> {
        let digest = self.dataset_plan_digest(plan_name)?;
        let (revision, source_row_ids, result) =
            self.evaluate_dataset_plan_at(vm, source_ids, plan_name, revision, replacement)?;
        let identity = DatasetIdentity::from_run(query_id, &digest, revision, &result, vm.dataset_decisions())?;
        let snapshot = DatasetSnapshot::from_run(query_id, calculation_version, &digest, revision, &result, identity)?;
        let bytes = snapshot.encode()?;
        DatasetSnapshot::load(&bytes, query_id, revision, &digest)?;
        Ok((revision, source_row_ids, bytes))
    }

    /// Dispatch one durable-pipeline host call. Returns `LanaError::Format` for
    /// an ID outside the store/policy/ledger range (the VM's own host calls are
    /// handled before the extension is consulted).
    pub fn dispatch(&mut self, vm: &mut Vm, host_id: u32, args: &[Value], out: &mut Value) -> LanaError {
        *out = Value::null();
        self.heap = vm.heap();
        #[cfg(target_arch = "wasm32")]
        if !matches!(host_id, LANA_HOST_RULES_LEARN | LANA_HOST_RULES_PREDICT
            | LANA_HOST_TREES_FIT | LANA_HOST_TREES_PREDICT | LANA_HOST_TREES_EXPLAIN
            | LANA_HOST_POLICY_EVALUATE) {
            return LanaError::UnsupportedOperation;
        }
        let status = match host_id {
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
            LANA_HOST_EXECUTION_CAPABILITY => self.execution_capability(args, out),
            LANA_HOST_EXECUTION_AUTHORIZE => self.execution_authorize(args, out),
            LANA_HOST_EXECUTION_EXECUTE => self.execution_execute(args, out),
            LANA_HOST_FUTURE_MESSAGE => self.future_message_call(args, out),
            LANA_HOST_DATASET_SOURCE => self.dataset_source(args, out),
            LANA_HOST_DATASET_QUERY => self.dataset_query(vm, args, out),
            LANA_HOST_DATASET_APPLY => self.dataset_apply(vm, args, out),
            LANA_HOST_DATASET_SNAPSHOT => self.dataset_snapshot(args, out),
            LANA_HOST_DATASET_EVIDENCE => self.dataset_evidence(args, out),
            LANA_HOST_DATASET_EXCLUSIONS => self.dataset_exclusions(args, out),
            LANA_HOST_DOCUMENT_EXTRACT => self.document_extract(args, out),
            LANA_HOST_DATASET_SQLITE => self.dataset_sqlite(vm, args, out),
            LANA_HOST_RULES_LEARN => match crate::rules::learn(args, vm) {
                Ok(value) => { *out = value; LanaError::Ok }
                Err(error) => error,
            },
            LANA_HOST_RULES_PREDICT => match crate::rules::predict(args, vm) {
                Ok(value) => { *out = value; LanaError::Ok }
                Err(error) => error,
            },
            LANA_HOST_RULES_SAVE => self.rules_save(vm, args, out),
            LANA_HOST_RULES_ADD_COUNTEREXAMPLE => self.rules_add_counterexample(vm, args, out),
            LANA_HOST_RULES_INSPECT => self.rules_inspect(vm, args, out),
            LANA_HOST_RULES_ROLLBACK => self.rules_rollback(vm, args, out),
            LANA_HOST_TREES_FIT => match crate::trees::fit(args, vm) {
                Ok(value) => { *out = value; LanaError::Ok }
                Err(error) => error,
            },
            LANA_HOST_TREES_PREDICT => match crate::trees::predict(args, vm) {
                Ok(value) => { *out = value; LanaError::Ok }
                Err(error) => error,
            },
            LANA_HOST_TREES_EXPLAIN => match crate::trees::explain(args, vm) {
                Ok(value) => { *out = value; LanaError::Ok }
                Err(error) => error,
            },
            LANA_HOST_TREES_SAVE => self.trees_save(vm, args, out),
            LANA_HOST_TREES_LOAD => self.trees_load(vm, args, out),
            _ => LanaError::Format,
        };
        // Issued decisions are sealed by map identity. They are allocated on
        // this VM's heap above and must keep the exact issued identity.
        if status == LanaError::Ok && host_id != LANA_HOST_POLICY_EVALUATE {
            match vm.import_value(out) {
                Ok(value) => *out = value,
                Err(error) => { *out = Value::null(); return error; }
            }
        }
        status
    }

    fn opaque_token(&self) -> Arc<lana_vm::value::CapabilityToken> {
        let id = NEXT_EXECUTION_TOKEN.fetch_add(1, Ordering::Relaxed);
        let shared = Arc::new(lana_vm::value::SharedInformation {
            identity: id,
            base_snapshot: Value::null(),
            state: Mutex::new(lana_vm::value::SharedState::default()),
            condition: Condvar::new(),
        });
        Arc::new(lana_vm::value::CapabilityToken {
            shared,
            id,
            permissions: lana_vm::value::LANA_CAPABILITY_ADMIN,
            revoked: AtomicBool::new(false),
        })
    }

    fn execution_capability(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if !args.is_empty() { return LanaError::Type; }
        if self.execution.is_none() {
            let metadata = match std::env::var("LANA_EXECUTION_METADATA") { Ok(value) => value, Err(_) => return LanaError::Capability };
            let key = match std::env::var("LANA_EXECUTION_KEY") { Ok(value) => value, Err(_) => return LanaError::Capability };
            let config = match execution::ExecutionConfig::load(std::path::Path::new(&metadata), std::path::Path::new(&key)) { Ok(config) => config, Err(error) => return error };
            let credential_name = format!("LANA_EXECUTION_CREDENTIAL_{}", config.credential_key_id);
            let credential = match std::env::var(&credential_name) { Ok(value) if !value.is_empty() && !value.contains(['\r', '\n']) => Arc::<str>::from(value), _ => return LanaError::Capability };
            let capability = config.capability;
            self.execution_credential = Some(credential);
            self.execution_ca_file = config.ca_file;
            self.execution = Some((self.opaque_token(), capability));
        }
        *out = Value::capability(self.execution.as_ref().unwrap().0.clone());
        LanaError::Ok
    }

    fn execution_authorize(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 3 { return LanaError::Type; }
        let ValueKind::Capability(token) = &args[0].kind else { return LanaError::Type; };
        let Some((expected, capability)) = &self.execution else { return LanaError::Capability; };
        if token.revoked.load(Ordering::Acquire) || !Arc::ptr_eq(token, expected) { return LanaError::Capability; }
        let ValueKind::Map(decision_map) = &args[1].kind else { return LanaError::Capability; };
        let encoded = match crate::codec::encode_value(&args[1]) { Ok(value) => value, Err(error) => return error };
        let digest_of_decision = crate::sha256::sha256(encoded.as_bytes());
        if !self.issued_decisions.iter().any(|(issued, digest)| Arc::ptr_eq(issued, decision_map) && digest == &digest_of_decision) {
            return LanaError::Capability;
        }
        let decision = match decision_from_value(&args[1]) { Ok(decision) => decision, Err(error) => return error };
        if decision.outcome != PolicyOutcome::Authorize { return LanaError::Capability; }
        let digest = match execution::plan_digest(&args[2]) { Ok(digest) => digest, Err(error) => return error };
        let authorization = Authorization { decision_id: decision.decision_id, capability_id: Arc::from(capability.id()), plan_digest: digest, authorized: true };
        let token = self.opaque_token();
        self.authorizations.push(ExecutionAuthorization { token: token.clone(), authorization });
        *out = Value::capability(token);
        LanaError::Ok
    }

    fn execution_execute(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        if args.len() != 3 { return LanaError::Type; }
        let ValueKind::Capability(capability_token) = &args[0].kind else { return LanaError::Type; };
        let ValueKind::Capability(authorization_token) = &args[1].kind else { return LanaError::Type; };
        let Some((expected, capability)) = &self.execution else { return LanaError::Capability; };
        if capability_token.revoked.load(Ordering::Acquire) || authorization_token.revoked.load(Ordering::Acquire) || !Arc::ptr_eq(capability_token, expected) { return LanaError::Capability; }
        let Some(authorization) = self.authorizations.iter().find(|entry| Arc::ptr_eq(authorization_token, &entry.token)).map(|entry| entry.authorization.clone()) else { return LanaError::Capability; };
        let ValueKind::Map(plan) = &args[2].kind else { return LanaError::Type; };
        let path = match plan.lock().unwrap().get("path") { Some(Value { kind: ValueKind::String(path), .. }) => path.clone(), _ => return LanaError::Schema };
        let Some(store) = self.store.as_mut() else { return LanaError::InvalidState; };
        let transport = CurlTransport { credential: self.execution_credential.clone(), ca_file: self.execution_ca_file.clone() };
        match execution::execute(store, capability, &authorization, &args[2], &path, &transport) {
            Ok(_) => match execution::read_receipt(store, capability, &authorization.plan_digest) {
                Ok(receipt) => { *out = receipt; LanaError::Ok }
                Err(error) => error,
            },
            Err(error) => error,
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
                self.source_relationships = Default::default();
                self.store = Some(store);
                *out = Value::null();
                LanaError::Ok
            }
            Err(e) => e,
        }
    }

    fn dataset_source(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        let [Value { kind: ValueKind::String(id), .. }] = args else { return LanaError::Type; };
        if id.is_empty() || id.len() > 128 { return LanaError::InvalidParameters; }
        let Some(store) = self.store.as_mut() else { return LanaError::InvalidState; };
        if let Err(error) = store.ensure_clean() { return error; }
        let key = format!("dataset/source/{}", id.as_bytes().iter().map(|byte| format!("{byte:02x}")).collect::<String>());
        let mut record = match Map::new(&self.heap, 5) { Ok(record) => record, Err(error) => return error };
        let empty_rows = match Array::from_items(&self.heap, Vec::new()) { Ok(rows) => rows, Err(error) => return error };
        let empty_ids = match Array::from_items(&self.heap, Vec::new()) { Ok(ids) => ids, Err(error) => return error };
        for (name, value) in [
            ("schema_version", Value::number(1.0)),
            ("format", Value::string(Arc::from("dataset_source_v1"))),
            ("source_id", Value::string(id.clone())),
            ("rows", Value::array(Arc::new(Mutex::new(empty_rows)))),
            ("row_ids", Value::array(Arc::new(Mutex::new(empty_ids)))),
        ] {
            if let Err(error) = record.set(Arc::from(name), value, true) { return error; }
        }
        let record = Value::map(Arc::new(Mutex::new(record)));
        match store::store_get(store, &key) {
            Ok(existing) => {
                let ValueKind::Map(existing) = existing.kind else { return LanaError::Corruption; };
                let existing = existing.lock().unwrap();
                if !matches!(existing.get("schema_version"), Some(Value { kind: ValueKind::Number(1.0), .. }))
                    || !matches!(existing.get("format"), Some(Value { kind: ValueKind::String(value), .. }) if value.as_ref() == "dataset_source_v1")
                    || !matches!(existing.get("source_id"), Some(Value { kind: ValueKind::String(value), .. }) if value == id)
                    || !matches!(existing.get("rows"), Some(Value { kind: ValueKind::Array(_), .. }))
                    || !matches!(existing.get("row_ids"), Some(Value { kind: ValueKind::Array(_), .. })) {
                    return LanaError::Corruption;
                }
                *out = Value::null();
                return LanaError::Ok;
            }
            Err(LanaError::NotFound) => {},
            Err(_) => return LanaError::Corruption,
        }
        if let Err(error) = store::store_put(store, &key, &record) { return error; }
        match store::store_commit(store) {
            Ok(_) => { *out = Value::null(); LanaError::Ok }
            Err(error) => { self.store = None; error }
        }
    }

    fn dataset_query(&mut self, vm: &mut Vm, args: &[Value], out: &mut Value) -> LanaError {
        let [query, sources, plan, version] = args else { return LanaError::Type; };
        let (ValueKind::String(query_id), ValueKind::Array(sources),
            ValueKind::String(plan_name), ValueKind::String(calculation_version)) =
            (&query.kind, &sources.kind, &plan.kind, &version.kind) else { return LanaError::Type; };
        if query_id.is_empty() || query_id.len() > 128 || plan_name.is_empty() || plan_name.len() > 128
            || calculation_version.is_empty() { return LanaError::InvalidParameters; }
        let sources = sources.lock().unwrap().items().to_vec();
        let mut source_ids = Vec::with_capacity(sources.len());
        for source in &sources {
            let ValueKind::String(id) = &source.kind else { return LanaError::Type; };
            if id.is_empty() || id.len() > 128 { return LanaError::InvalidParameters; }
            source_ids.push(id.to_string());
        }
        let digest = match self.dataset_plan_digest(plan_name) { Ok(digest) => digest, Err(error) => return error };
        let hex = query_id.as_bytes().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        let query_key = format!("dataset/query/{hex}");
        let Some(store) = self.store.as_ref() else { return LanaError::InvalidState; };
        if let Err(error) = store.ensure_clean() { return error; }
        let current = match store::store_current_revision(store) { Ok(info) => info.revision_id, Err(error) => return error };
        match store::store_get(store, &query_key) {
            Ok(saved) => {
                let record = match load_dataset_query_record(store, &query_key, saved, current) {
                    Ok(record) => record, Err(error) => return error,
                };
                if record.calculation_version == calculation_version.as_ref() {
                    if record.source_ids != source_ids || record.plan_name != plan_name.as_ref()
                        || record.plan_digest != digest { return LanaError::Conflict; }
                    self.bound_queries.insert(query_id.to_string(), record);
                    *out = Value::null();
                    return LanaError::Ok;
                }
            }
            Err(LanaError::NotFound) => {},
            Err(error) => return error,
        }
        let Some(next_revision) = current.checked_add(1) else { return LanaError::Limit; };
        let (source_revision, _, bytes) = match self.evaluate_dataset_plan_snapshot(vm, query_id,
            &source_ids, plan_name, calculation_version) { Ok(result) => result, Err(error) => return error };
        if source_revision != current { return LanaError::Conflict; }
        let record = DatasetQueryRecord { schema_version: 1, format: "dataset_query_v1".into(),
            query_id: query_id.to_string(), source_ids, plan_name: plan_name.to_string(),
            calculation_version: calculation_version.to_string(), plan_digest: digest,
            source_revision: source_revision.to_string(), snapshot_revision: next_revision.to_string() };
        let record_bytes = match crate::information_codec::canonical(&record) { Ok(bytes) => bytes, Err(error) => return error };
        let record_value = Value::string(Arc::from(String::from_utf8(record_bytes).expect("canonical JSON is UTF-8")));
        let snapshot = Value::string(Arc::from(String::from_utf8(bytes).expect("canonical JSON is UTF-8")));
        let snapshot_key = format!("dataset/snapshot/{hex}/{next_revision}");
        let Some(store) = self.store.as_mut() else { return LanaError::InvalidState; };
        if let Err(error) = store::store_put(store, &query_key, &record_value) { return error; }
        if let Err(error) = store::store_put(store, &snapshot_key, &snapshot) {
            store.discard_staged();
            return error;
        }
        match store::store_commit(store) {
            Ok(_) => { self.bound_queries.insert(query_id.to_string(), record); *out = Value::null(); LanaError::Ok }
            Err(error) => { self.store = None; error }
        }
    }

    fn dataset_apply(&mut self, vm: &mut Vm, args: &[Value], out: &mut Value) -> LanaError {
        let [source, expected, batch, changes] = args else { return LanaError::Type; };
        let (ValueKind::String(source_id), ValueKind::Number(expected), ValueKind::String(batch_id), ValueKind::Array(changes)) =
            (&source.kind, &expected.kind, &batch.kind, &changes.kind) else { return LanaError::Type; };
        if source_id.is_empty() || source_id.len() > 128 || batch_id.is_empty() || batch_id.len() > 128
            || !expected.is_finite() || *expected < 0.0 || expected.fract() != 0.0 || *expected > 9_007_199_254_740_991.0 {
            return LanaError::InvalidParameters;
        }
        let changes = changes.lock().unwrap().items().to_vec();
        if changes.is_empty() || changes.len() > 10_000 { return LanaError::InvalidParameters; }
        let mut encoder = crate::dataset_source_codec::Encoder::default();
        let mut normalized = Vec::with_capacity(changes.len());
        let mut changed_ids = HashSet::new();
        for change in &changes {
            let ValueKind::Map(map) = &change.kind else { return LanaError::Type; };
            let map = map.lock().unwrap();
            let (Some(Value { kind: ValueKind::String(op), .. }), Some(Value { kind: ValueKind::String(id), .. })) =
                (map.get("op"), map.get("id")) else { return LanaError::Type; };
            if id.is_empty() || id.len() > 128 || !changed_ids.insert(id.to_string()) { return LanaError::InvalidParameters; }
            let row = match op.as_ref() {
                "add" | "correct" if map.entries().len() == 3 => {
                    let Some(row) = map.get("row") else { return LanaError::Type; };
                    let bytes = match encoder.row(row) { Ok(bytes) => bytes, Err(error) => return error };
                    Some(String::from_utf8(bytes).expect("canonical JSON is UTF-8"))
                }
                "delete" if map.entries().len() == 2 => None,
                _ => return LanaError::Type,
            };
            normalized.push((op.to_string(), id.to_string(), row));
        }
        let payload = match crate::information_codec::canonical(&normalized) { Ok(bytes) => bytes, Err(error) => return error };
        if payload.len() > 64 * 1024 * 1024 { return LanaError::Limit; }
        let digest: String = crate::sha256::sha256(&payload).iter().map(|byte| format!("{byte:02x}")).collect();
        let hex = |id: &str| id.as_bytes().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        let source_key = format!("dataset/source/{}", hex(source_id));
        let batch_key = format!("dataset/batch/{}/{}", hex(source_id), hex(batch_id));
        let Some(store) = self.store.as_ref() else { return LanaError::InvalidState; };
        if let Err(error) = store.ensure_clean() { return error; }
        match store::store_get(store, &batch_key) {
            Ok(receipt) => {
                let ValueKind::Map(receipt) = receipt.kind else { return LanaError::Corruption; };
                let receipt = receipt.lock().unwrap();
                if !matches!(receipt.get("format"), Some(Value { kind: ValueKind::String(saved), .. }) if saved.as_ref() == "dataset_batch_v1")
                    || !matches!(receipt.get("source_id"), Some(Value { kind: ValueKind::String(saved), .. }) if saved == source_id) {
                    return LanaError::Corruption;
                }
                if !matches!(receipt.get("payload_digest"), Some(Value { kind: ValueKind::String(saved), .. }) if saved.as_ref() == digest) {
                    return LanaError::Conflict;
                }
                let Some(Value { kind: ValueKind::String(revision), .. }) = receipt.get("revision") else { return LanaError::Corruption; };
                let revision = match crate::information_codec::revision(revision) { Ok(revision) => revision, Err(_) => return LanaError::Corruption };
                if revision > 9_007_199_254_740_991 { return LanaError::Corruption; }
                let bindings = match receipt.get("information_bindings") {
                    Some(Value { kind: ValueKind::String(text), .. }) => text.as_ref(),
                    None => "[{},{}]",
                    _ => return LanaError::Corruption,
                };
                if let Err(error) = encoder.remember(&mut self.source_relationships, bindings) { return error; }
                *out = Value::number(revision as f64);
                return LanaError::Ok;
            }
            Err(LanaError::NotFound) => {},
            Err(error) => return error,
        }
        let bindings = match encoder.bindings(&self.source_relationships, source_id, batch_id) {
            Ok(text) => text, Err(error) => return error,
        };
        for (_, _, row) in &mut normalized {
            if let Some(text) = row {
                *text = match crate::dataset_source_codec::bind(text, &bindings) { Ok(text) => text, Err(error) => return error };
            }
        }
        let source_record = match store::store_get(store, &source_key) { Ok(record) => record, Err(error) => return error };
        let current = match store::store_current_revision(store) { Ok(info) => info.revision_id, Err(error) => return error };
        if current != *expected as u64 { return LanaError::Conflict; }
        if current >= 9_007_199_254_740_991 { return LanaError::Limit; }
        let ValueKind::Map(source_record) = source_record.kind else { return LanaError::Corruption; };
        let source_record = source_record.lock().unwrap();
        if !matches!(source_record.get("schema_version"), Some(Value { kind: ValueKind::Number(1.0), .. }))
            || !matches!(source_record.get("format"), Some(Value { kind: ValueKind::String(value), .. }) if value.as_ref() == "dataset_source_v1")
            || !matches!(source_record.get("source_id"), Some(Value { kind: ValueKind::String(value), .. }) if value == source_id) {
            return LanaError::Corruption;
        }
        let (Some(Value { kind: ValueKind::Array(row_ids), .. }), Some(Value { kind: ValueKind::Array(rows), .. })) =
            (source_record.get("row_ids"), source_record.get("rows")) else { return LanaError::Corruption; };
        let row_ids = row_ids.lock().unwrap().items().to_vec();
        let rows = rows.lock().unwrap().items().to_vec();
        if row_ids.len() != rows.len() || rows.len() > 10_000 { return LanaError::Corruption; }
        let mut ids = Vec::with_capacity(row_ids.len());
        let mut encoded_rows = Vec::with_capacity(rows.len());
        let mut existing_ids = HashSet::new();
        let mut decoder = crate::dataset_source_codec::Decoder::default();
        for (id, row) in row_ids.iter().zip(&rows) {
            let (ValueKind::String(id), ValueKind::String(row)) = (&id.kind, &row.kind) else { return LanaError::Corruption; };
            if id.is_empty() || id.len() > 128 || !existing_ids.insert(id.to_string())
                || decoder.row(row.as_bytes(), vm).is_err() {
                return LanaError::Corruption;
            }
            ids.push(id.to_string());
            encoded_rows.push(row.to_string());
        }
        drop(source_record);
        let mut positions: HashMap<String, usize> = ids.iter().enumerate().map(|(index, id)| (id.clone(), index)).collect();
        let mut retained = vec![true; ids.len()];
        for (op, id, row) in &normalized {
            let position = positions.get(id).copied();
            match (op.as_str(), position) {
                ("add", None) => {
                    positions.insert(id.clone(), ids.len());
                    ids.push(id.clone());
                    encoded_rows.push(row.clone().unwrap());
                    retained.push(true);
                }
                ("correct", Some(index)) => encoded_rows[index] = row.clone().unwrap(),
                ("delete", Some(index)) => { positions.remove(id); retained[index] = false; }
                _ => return LanaError::Conflict,
            }
        }
        if positions.len() > 10_000 { return LanaError::Limit; }
        let (id_values, row_values): (Vec<_>, Vec<_>) = ids.into_iter().zip(encoded_rows).zip(retained)
            .filter_map(|((id, row), keep)| keep.then(|| (Value::string(Arc::from(id)), Value::string(Arc::from(row)))))
            .unzip();
        let id_array = match Array::from_items(&self.heap, id_values) { Ok(array) => array, Err(error) => return error };
        let row_array = match Array::from_items(&self.heap, row_values) { Ok(array) => array, Err(error) => return error };
        let mut next = match Map::new(&self.heap, 5) { Ok(map) => map, Err(error) => return error };
        for (name, value) in [
            ("schema_version", Value::number(1.0)),
            ("format", Value::string(Arc::from("dataset_source_v1"))),
            ("source_id", Value::string(source_id.clone())),
            ("row_ids", Value::array(Arc::new(Mutex::new(id_array)))),
            ("rows", Value::array(Arc::new(Mutex::new(row_array)))),
        ] {
            if let Err(error) = next.set(Arc::from(name), value, false) { return error; }
        }
        let candidate = Value::map(Arc::new(Mutex::new(next)));
        let mut affected = Vec::new();
        let registry = match store::store_scan(self.store.as_ref().unwrap(), "dataset/query/") {
            Ok(records) => records, Err(error) => return error,
        };
        for saved in registry {
            let mut query = match load_dataset_query_record(self.store.as_ref().unwrap(), &saved.key,
                saved.value, current) { Ok(query) => query, Err(error) => return error };
            if !query.source_ids.iter().any(|id| id == source_id.as_ref()) { continue; }
            if self.bound_queries.get(&query.query_id) != Some(&query)
                || self.dataset_plan_digest(&query.plan_name).ok().as_deref() != Some(&query.plan_digest) {
                return LanaError::Conflict;
            }
            let (revision, _, bytes) = match self.evaluate_dataset_plan_snapshot_at(vm,
                &query.query_id, &query.source_ids, &query.plan_name, &query.calculation_version,
                current + 1, Some((source_id, &candidate))) {
                Ok(result) => result, Err(error) => return error,
            };
            if revision != current + 1 { return LanaError::Conflict; }
            query.source_revision = revision.to_string();
            query.snapshot_revision = revision.to_string();
            let record = match crate::information_codec::canonical(&query) {
                Ok(bytes) => bytes, Err(error) => return error,
            };
            let hex = hex(&query.query_id);
            affected.push((format!("dataset/query/{hex}"), query,
                Value::string(Arc::from(String::from_utf8(record).expect("canonical JSON is UTF-8"))),
                format!("dataset/snapshot/{hex}/{revision}"),
                Value::string(Arc::from(String::from_utf8(bytes).expect("canonical JSON is UTF-8")))));
        }
        let mut receipt = match Map::new(&self.heap, 4) { Ok(map) => map, Err(error) => return error };
        for (name, value) in [
            ("format", Value::string(Arc::from("dataset_batch_v1"))),
            ("source_id", Value::string(source_id.clone())),
            ("payload_digest", Value::string(Arc::from(digest))),
            ("revision", Value::string(Arc::from((current + 1).to_string()))),
        ] {
            if let Err(error) = receipt.set(Arc::from(name), value, false) { return error; }
        }
        if bindings != "[{},{}]" {
            if let Err(error) = receipt.set(Arc::from("information_bindings"), Value::string(Arc::from(bindings.as_str())), false) { return error; }
        }
        let Some(store) = self.store.as_mut() else { return LanaError::InvalidState; };
        if let Err(error) = store::store_put(store, &source_key, &candidate) { return error; }
        for (query_key, _, record, snapshot_key, snapshot) in &affected {
            if let Err(error) = store::store_put(store, query_key, record)
                .and_then(|()| store::store_put(store, snapshot_key, snapshot)) {
                store.discard_staged();
                return error;
            }
        }
        if let Err(error) = store::store_put(store, &batch_key, &Value::map(Arc::new(Mutex::new(receipt)))) {
            store.discard_staged();
            return error;
        }
        match store::store_commit(store) {
            Ok(info) => {
                if let Err(error) = encoder.remember(&mut self.source_relationships, &bindings) { return error; }
                for (_, query, _, _, _) in affected { self.bound_queries.insert(query.query_id.clone(), query); }
                *out = Value::number(info.revision_id as f64); LanaError::Ok
            }
            Err(error) => { self.store = None; error }
        }
    }

    fn dataset_snapshot(&self, args: &[Value], out: &mut Value) -> LanaError {
        let [Value { kind: ValueKind::String(query_id), .. }, revision] = args else { return LanaError::Type; };
        if query_id.is_empty() || query_id.len() > 128 { return LanaError::InvalidParameters; }
        let result = (|| -> Result<Value, LanaError> {
            let store = self.store.as_ref().ok_or(LanaError::InvalidState)?;
            let current = store::store_current_revision(store)?.revision_id;
            let at = match revision.kind {
                ValueKind::Null => current,
                ValueKind::Number(value) if value.is_finite() && value >= 0.0
                    && value.fract() == 0.0 && value <= 9_007_199_254_740_991.0 => value as u64,
                ValueKind::Number(_) => return Err(LanaError::InvalidParameters),
                _ => return Err(LanaError::Type),
            };
            let hex = query_id.as_bytes().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
            let query_key = format!("dataset/query/{hex}");
            let saved = store::store_get_at(store, at, &query_key)?;
            let record = load_dataset_query_record(store, &query_key, saved, at)?;
            let snapshot_revision = crate::information_codec::revision(&record.snapshot_revision)
                .map_err(|_| LanaError::Corruption)?;
            let key = format!("dataset/snapshot/{hex}/{snapshot_revision}");
            let saved = store::store_get_at(store, at, &key)?;
            let ValueKind::String(saved) = saved.kind else { return Err(LanaError::Corruption); };
            let source_revision = crate::information_codec::revision(&record.source_revision)
                .map_err(|_| LanaError::Corruption)?;
            DatasetSnapshot::load(saved.as_bytes(), query_id, source_revision, &record.plan_digest)?;
            crate::data::json_parse(&saved).map_err(|error|
                if matches!(error, LanaError::Limit | LanaError::Oom) { error } else { LanaError::Corruption })
        })();
        match result { Ok(value) => { *out = value; LanaError::Ok }, Err(error) => error }
    }

    fn dataset_evidence(&self, args: &[Value], out: &mut Value) -> LanaError {
        let [snapshot, Value { kind: ValueKind::String(output_id), .. }] = args else { return LanaError::Type; };
        let result = (|| -> Result<Value, LanaError> {
            let snapshot = DatasetSnapshot::from_value(snapshot)?;
            let evidence = snapshot.row_evidence(output_id)?;
            let bytes = crate::information_codec::canonical(&evidence)?;
            crate::data::json_parse(std::str::from_utf8(&bytes).map_err(|_| LanaError::Corruption)?)
        })();
        match result { Ok(value) => { *out = value; LanaError::Ok }, Err(error) => error }
    }

    fn dataset_exclusions(&self, args: &[Value], out: &mut Value) -> LanaError {
        let [snapshot] = args else { return LanaError::Type; };
        let result = (|| -> Result<Value, LanaError> {
            let snapshot = DatasetSnapshot::from_value(snapshot)?;
            let bytes = crate::information_codec::canonical(&snapshot.exclusions)?;
            crate::data::json_parse(std::str::from_utf8(&bytes).map_err(|_| LanaError::Corruption)?)
        })();
        match result { Ok(value) => { *out = value; LanaError::Ok }, Err(error) => error }
    }

    fn document_extract(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        let [Value { kind: ValueKind::String(path), .. }, Value { kind: ValueKind::String(format), .. }] = args else { return LanaError::Type; };
        #[cfg(target_arch = "wasm32")]
        { let _ = (path, format, out); return LanaError::UnsupportedOperation; }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let document = match crate::document_extract::extract(std::path::Path::new(path.as_ref()), format) {
                Ok(document) => document, Err(error) => return error,
            };
            let value = (|| -> Result<Value, LanaError> {
                let mut chunks = Vec::with_capacity(document.chunks.len());
                for chunk in document.chunks {
                    let headings = chunk.heading_path.into_iter().map(|text| Value::string(Arc::from(text))).collect();
                    let headings = Value::array(Arc::new(Mutex::new(Array::from_items(&self.heap, headings)?)));
                    let mut record = Map::new(&self.heap, 6)?;
                    for (name, value) in [
                        ("text", Value::string(Arc::from(chunk.text))),
                        ("start_line", Value::number(chunk.start_line as f64)),
                        ("end_line", Value::number(chunk.end_line as f64)),
                        ("start_byte", Value::number(chunk.start_byte as f64)),
                        ("end_byte", Value::number(chunk.end_byte as f64)),
                        ("heading_path", headings),
                    ] { record.set(Arc::from(name), value, false)?; }
                    chunks.push(Value::map(Arc::new(Mutex::new(record))));
                }
                let chunks = Value::array(Arc::new(Mutex::new(Array::from_items(&self.heap, chunks)?)));
                let mut record = Map::new(&self.heap, 4)?;
                for (name, value) in [
                    ("format", Value::string(Arc::from(document.format))),
                    ("sha256", Value::string(Arc::from(document.sha256))),
                    ("chunks", chunks),
                    ("status", Value::string(Arc::from(document.status))),
                ] { record.set(Arc::from(name), value, false)?; }
                Ok(Value::map(Arc::new(Mutex::new(record))))
            })();
            match value {
                Ok(value) => { *out = value; LanaError::Ok }
                Err(error) => error,
            }
        }
    }

    fn dataset_sqlite(&mut self, vm: &mut Vm, args: &[Value], out: &mut Value) -> LanaError {
        let [Value { kind: ValueKind::String(path), .. }, Value { kind: ValueKind::String(sql), .. },
            Value { kind: ValueKind::Array(parameters), .. }, Value { kind: ValueKind::Array(schema), .. }] = args else { return LanaError::Type; };
        #[cfg(target_arch = "wasm32")]
        { let _ = (path, sql, parameters, schema, vm, out); return LanaError::UnsupportedOperation; }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let parameters = parameters.lock().unwrap().items().to_vec();
            let schema = schema.lock().unwrap().items().to_vec();
            let declared = (|| -> Result<Vec<(String, String)>, LanaError> {
                schema.iter().map(|entry| {
                    let ValueKind::Map(entry) = &entry.kind else { return Err(LanaError::Schema); };
                    let entry = entry.lock().unwrap();
                    if entry.entries().len() != 2 { return Err(LanaError::Schema); }
                    let (Some(Value { kind: ValueKind::String(name), .. }), Some(Value { kind: ValueKind::String(kind), .. })) =
                        (entry.get("name"), entry.get("kind")) else { return Err(LanaError::Schema); };
                    Ok((name.to_string(), kind.to_string()))
                }).collect()
            })();
            let declared = match declared { Ok(declared) => declared, Err(error) => return error };
            match crate::dataset_sqlite::execute(path, sql, &parameters, &declared, &self.heap, vm) {
                Ok(value) => { *out = value; LanaError::Ok }
                Err(error) => error,
            }
        }
    }

    fn future_message_call(&mut self, args: &[Value], out: &mut Value) -> LanaError {
        let Some(Value { kind: ValueKind::String(operation), .. }) = args.first() else { return LanaError::Type; };
        let Some(store) = self.store.as_mut() else { return LanaError::InvalidState; };
        let result = match (operation.as_ref(), args.len()) {
            ("create", 2) => future_messages::utc_now().and_then(|now| future_messages::create(store, &args[1], now, &self.heap)),
            ("inspect", 2) => future_messages::inspect(store, &args[1]),
            ("check", 2) => future_messages::utc_now().and_then(|now| future_messages::check(store, &args[1], now, &self.heap)),
            ("receive", 3) => future_messages::receive(store, &args[1], &args[2], &self.heap),
            ("acknowledge", 2) => future_messages::utc_now().and_then(|now| future_messages::acknowledge(store, &args[1], now, &self.heap)),
            ("cancel", 2) => future_messages::utc_now().and_then(|now| future_messages::cancel(store, &args[1], now, &self.heap)),
            _ => Err(LanaError::Type),
        };
        if store::store_current_revision(store).is_err() { self.store = None; }
        match result {
            Ok(value) => { *out = value; LanaError::Ok }
            Err(error) => error,
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

    fn rules_save(&mut self, vm: &mut Vm, args: &[Value], out: &mut Value) -> LanaError {
        let [Value { kind: ValueKind::Null, .. },
            Value { kind: ValueKind::String(task_id), .. },
            Value { kind: ValueKind::Number(expected), .. }, report] = args else { return LanaError::Type; };
        let expected = match number_as_u64(*expected) { Ok(value) => value, Err(error) => return error };
        let report = match crate::rules::validate_report(report, vm) {
            Ok(report) => report, Err(error) => return error,
        };
        let record = match crate::rules::initial_record(task_id, &report, vm) {
            Ok(record) => record, Err(error) => return error,
        };
        let key = format!("learned/task/{}", task_id.as_bytes().iter()
            .map(|byte| format!("{byte:02x}")).collect::<String>());
        let Some(store) = self.store.as_mut() else { return LanaError::InvalidState; };
        if let Err(error) = store.ensure_clean() { return error; }
        let current = match store::store_current_revision(store) { Ok(value) => value.revision_id, Err(error) => return error };
        if current != expected { return LanaError::Conflict; }
        match store::store_get(store, &key) {
            Ok(_) => return LanaError::Conflict,
            Err(LanaError::NotFound) => {},
            Err(error) => return error,
        }
        if let Err(error) = store::store_put(store, &key, &Value::string(Arc::from(record))) {
            store.discard_staged();
            return error;
        }
        match store::store_commit(store) {
            Ok(_) => { *out = Value::string(Arc::from("1")); LanaError::Ok }
            Err(error) => { self.store = None; error }
        }
    }

    fn rules_inspect(&self, vm: &mut Vm, args: &[Value], out: &mut Value) -> LanaError {
        let [Value { kind: ValueKind::Null, .. },
            Value { kind: ValueKind::String(task_id), .. }, revision] = args else { return LanaError::Type; };
        if task_id.is_empty() || task_id.len() > 128 { return LanaError::Schema; }
        let Some(store) = self.store.as_ref() else { return LanaError::InvalidState; };
        if let Err(error) = store.ensure_clean() { return error; }
        let at = match &revision.kind {
            ValueKind::Null => match store::store_current_revision(store) { Ok(info) => info.revision_id, Err(error) => return error },
            ValueKind::Number(number) => match number_as_u64(*number) { Ok(number) => number, Err(error) => return error },
            _ => return LanaError::Type,
        };
        let key = format!("learned/task/{}", task_id.as_bytes().iter()
            .map(|byte| format!("{byte:02x}")).collect::<String>());
        let value = match store::store_get_at(store, at, &key) { Ok(value) => value, Err(error) => return error };
        let ValueKind::String(text) = value.kind else { return LanaError::Corruption; };
        let record = match crate::rules::load_record(&text, vm) { Ok(record) => record, Err(error) => return error };
        match crate::data::json_parse_with_heap(&record.to_string(), &vm.heap()) {
            Ok(value) => { *out = value; LanaError::Ok },
            Err(error) => error,
        }
    }

    fn rules_add_counterexample(&mut self, vm: &mut Vm, args: &[Value], out: &mut Value) -> LanaError {
        let [Value { kind: ValueKind::Null, .. }, Value { kind: ValueKind::String(task_id), .. },
            Value { kind: ValueKind::Number(expected), .. }, example, new_holdout] = args
            else { return LanaError::Type; };
        let expected = match number_as_u64(*expected) { Ok(value) => value, Err(error) => return error };
        if task_id.is_empty() || task_id.len() > 128 { return LanaError::Schema; }
        let parse = |value: &Value| -> Result<serde_json::Value, LanaError> {
            serde_json::from_str(&crate::codec::encode_value(value)?).map_err(|_| LanaError::Schema)
        };
        let (example, new_holdout) = match (parse(example), parse(new_holdout)) {
            (Ok(example), Ok(holdout)) => (example, holdout),
            (Err(error), _) | (_, Err(error)) => return error,
        };
        let Some(example_id) = example["id"].as_str().map(str::to_owned) else { return LanaError::Schema; };
        if example.as_object().is_none_or(|fields| fields.len() != 3)
            || example_id.is_empty() || example_id.len() > 128
            || !new_holdout.is_array() { return LanaError::Schema; }
        let payload_digest = match crate::rules::correction_digest(&example, &new_holdout) {
            Ok(value) => value, Err(error) => return error,
        };
        let key = format!("learned/task/{}", task_id.as_bytes().iter()
            .map(|byte| format!("{byte:02x}")).collect::<String>());
        let Some(store) = self.store.as_mut() else { return LanaError::InvalidState; };
        if let Err(error) = store.ensure_clean() { return error; }
        let saved = match store::store_get(store, &key) { Ok(value) => value, Err(error) => return error };
        let ValueKind::String(saved) = saved.kind else { return LanaError::Corruption; };
        let mut record = match crate::rules::load_record(&saved, vm) { Ok(value) => value, Err(error) => return error };
        if record["task_id"] != task_id.as_ref() { return LanaError::Corruption; }
        let receipts = match record["receipts"].as_array() { Some(value) => value, None => return LanaError::Corruption };
        if let Some(receipt) = receipts.iter().find(|receipt| receipt["counterexample_id"] == example_id) {
            if receipt["payload_digest"] != payload_digest { return LanaError::Conflict; }
            *out = Value::string(Arc::from(receipt["version"].as_str().unwrap_or_default()));
            return LanaError::Ok;
        }
        let current = match store::store_current_revision(store) { Ok(value) => value.revision_id, Err(error) => return error };
        if current != expected { return LanaError::Conflict; }
        let versions = record["versions"].as_array().unwrap();
        let latest = match crate::rules::decoded_version(versions.last().unwrap()) {
            Ok(value) => value, Err(error) => return error,
        };
        let mut train = latest["train"].as_array().unwrap().clone();
        let mut used = HashSet::new();
        for version in versions {
            for row in version["train"].as_array().unwrap().iter().chain(version["holdout"].as_array().unwrap()) {
                used.insert(row["id"].as_str().unwrap().to_owned());
            }
        }
        if used.contains(&example_id) { return LanaError::Conflict; }
        let Some(holdout) = new_holdout.as_array() else { return LanaError::Schema; };
        for row in holdout {
            let Some(id) = row["id"].as_str() else { return LanaError::Schema; };
            if !used.insert(id.to_owned()) { return LanaError::Conflict; }
        }
        train.push(example);
        let inputs = [&latest["task"], &serde_json::Value::Array(train), &new_holdout, &latest["options"]]
            .iter().map(|input| crate::data::json_parse_with_heap(&input.to_string(), &vm.heap()))
            .collect::<Result<Vec<_>, _>>();
        let inputs = match inputs { Ok(value) => value, Err(error) => return error };
        let learned = match crate::rules::learn(&inputs, vm) { Ok(value) => value, Err(error) => return error };
        let report = match crate::rules::validate_report(&learned, vm) { Ok(value) => value, Err(error) => return error };
        let (version_id, text) = match crate::rules::append_version(&mut record, &report, &example_id,
            &payload_digest, vm) { Ok(value) => value, Err(error) => return error };
        if let Err(error) = store::store_put(store, &key, &Value::string(Arc::from(text))) {
            store.discard_staged();
            return error;
        }
        match store::store_commit(store) {
            Ok(_) => { *out = Value::string(Arc::from(version_id)); LanaError::Ok }
            Err(error) => { self.store = None; error }
        }
    }

    fn rules_rollback(&mut self, vm: &mut Vm, args: &[Value], out: &mut Value) -> LanaError {
        let [Value { kind: ValueKind::Null, .. }, Value { kind: ValueKind::String(task_id), .. },
            Value { kind: ValueKind::Number(expected), .. }, Value { kind: ValueKind::String(prior), .. }] = args
            else { return LanaError::Type; };
        let expected = match number_as_u64(*expected) { Ok(value) => value, Err(error) => return error };
        let number = match crate::information_codec::revision(prior) { Ok(value) => value as usize, Err(error) => return error };
        let key = format!("learned/task/{}", task_id.as_bytes().iter()
            .map(|byte| format!("{byte:02x}")).collect::<String>());
        let Some(store) = self.store.as_mut() else { return LanaError::InvalidState; };
        if let Err(error) = store.ensure_clean() { return error; }
        let current = match store::store_current_revision(store) { Ok(value) => value.revision_id, Err(error) => return error };
        if current != expected { return LanaError::Conflict; }
        let saved = match store::store_get(store, &key) { Ok(value) => value, Err(error) => return error };
        let ValueKind::String(saved) = saved.kind else { return LanaError::Corruption; };
        let mut record = match crate::rules::load_record(&saved, vm) { Ok(value) => value, Err(error) => return error };
        if record["task_id"] != task_id.as_ref() { return LanaError::Corruption; }
        let versions = record["versions"].as_array().unwrap();
        if number == 0 || number >= versions.len() || versions[number - 1]["report"]["status"] != "validated" {
            return LanaError::NotFound;
        }
        if record["active_version"] == prior.as_ref() { return LanaError::Conflict; }
        record["active_version"] = serde_json::json!(prior.as_ref());
        let text = match crate::rules::seal_record(&mut record, vm) { Ok(value) => value, Err(error) => return error };
        if let Err(error) = store::store_put(store, &key, &Value::string(Arc::from(text))) {
            store.discard_staged();
            return error;
        }
        match store::store_commit(store) {
            Ok(_) => { *out = Value::string(prior.clone()); LanaError::Ok }
            Err(error) => { self.store = None; error }
        }
    }

    fn trees_save(&mut self, vm: &mut Vm, args: &[Value], out: &mut Value) -> LanaError {
        let [Value { kind: ValueKind::Null, .. }, Value { kind: ValueKind::String(task_id), .. },
            Value { kind: ValueKind::Number(expected), .. }, report] = args else { return LanaError::Type; };
        let expected = match number_as_u64(*expected) { Ok(value) => value, Err(error) => return error };
        let report = match crate::trees::validate_report(report, vm) { Ok(value) => value, Err(error) => return error };
        let key = format!("learned/task/{}", task_id.as_bytes().iter()
            .map(|byte| format!("{byte:02x}")).collect::<String>());
        let Some(store) = self.store.as_mut() else { return LanaError::InvalidState; };
        if let Err(error) = store.ensure_clean() { return error; }
        let current = match store::store_current_revision(store) { Ok(value) => value.revision_id, Err(error) => return error };
        if current != expected { return LanaError::Conflict; }
        let previous = match store::store_get(store, &key) {
            Ok(value) => {
                let ValueKind::String(text) = value.kind else { return LanaError::Corruption; };
                match crate::trees::load_record(&text, vm) { Ok(value) => Some(value), Err(error) => return error }
            }
            Err(LanaError::NotFound) => None,
            Err(error) => return error,
        };
        let (version_id, text) = match crate::trees::append_record(previous, task_id, &report, vm) {
            Ok(value) => value, Err(error) => return error,
        };
        if let Err(error) = store::store_put(store, &key, &Value::string(Arc::from(text))) {
            store.discard_staged();
            return error;
        }
        match store::store_commit(store) {
            Ok(_) => { *out = Value::string(Arc::from(version_id)); LanaError::Ok }
            Err(error) => { self.store = None; error }
        }
    }

    fn trees_load(&self, vm: &mut Vm, args: &[Value], out: &mut Value) -> LanaError {
        let [Value { kind: ValueKind::Null, .. }, Value { kind: ValueKind::String(task_id), .. }, revision] = args
            else { return LanaError::Type; };
        if task_id.is_empty() || task_id.len() > 128 { return LanaError::Schema; }
        let Some(store) = self.store.as_ref() else { return LanaError::InvalidState; };
        if let Err(error) = store.ensure_clean() { return error; }
        let at = match &revision.kind {
            ValueKind::Null => match store::store_current_revision(store) { Ok(info) => info.revision_id, Err(error) => return error },
            ValueKind::Number(number) => match number_as_u64(*number) { Ok(number) => number, Err(error) => return error },
            _ => return LanaError::Type,
        };
        let key = format!("learned/task/{}", task_id.as_bytes().iter()
            .map(|byte| format!("{byte:02x}")).collect::<String>());
        let value = match store::store_get_at(store, at, &key) { Ok(value) => value, Err(error) => return error };
        let ValueKind::String(text) = value.kind else { return LanaError::Corruption; };
        let record = match crate::trees::load_record(&text, vm) { Ok(record) => record, Err(error) => return error };
        if record["task_id"] != task_id.as_ref() { return LanaError::Corruption; }
        match crate::data::json_parse_with_heap(&record.to_string(), &vm.heap()) {
            Ok(value) => { *out = value; LanaError::Ok }, Err(error) => error,
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
        if !kind.is_finite() || *kind < 0.0 || *kind > 3.0 || kind.fract() != 0.0 {
            return LanaError::UnsupportedOperation;
        }
        let options = AdapterOptions {
            schema_version: 1,
            kind: match *kind as u32 {
                0 => AdapterKind::Json,
                1 => AdapterKind::Csv,
                2 => AdapterKind::Sqlite,
                3 => AdapterKind::HttpJson,
                _ => return LanaError::UnsupportedOperation,
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
                let value = match decision_to_value(&self.heap, &decision) { Ok(value) => value, Err(error) => return error };
                let encoded = match crate::codec::encode_value(&value) { Ok(value) => value, Err(error) => return error };
                if let ValueKind::Map(map) = &value.kind {
                    self.issued_decisions.push((map.clone(), crate::sha256::sha256(encoded.as_bytes())));
                }
                *out = value;
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
    execution::validate_record_envelope(&map, "decision")?;
    if !matches!(map.get("transport_status"), Some(Value { kind: ValueKind::String(status), .. }) if status.as_ref() == "ok") {
        return Err(LanaError::Schema);
    }
    let outcome = match map_get_u32(&map, "outcome")? {
        0 => PolicyOutcome::Authorize,
        1 => PolicyOutcome::Refuse,
        2 => PolicyOutcome::RequestMoreEvidence,
        _ => return Err(LanaError::Schema),
    };
    let decision_id = map_get_u64(&map, "decision_id")?;
    let outcome_name = match outcome {
        PolicyOutcome::Authorize => "Authorize",
        PolicyOutcome::Refuse => "Refuse",
        PolicyOutcome::RequestMoreEvidence => "RequestMoreEvidence",
    };
    if !matches!(map.get("id"), Some(Value { kind: ValueKind::String(id), .. }) if id.as_ref() == format!("decision/{decision_id}"))
        || !matches!(map.get("domain_status"), Some(Value { kind: ValueKind::String(status), .. }) if status.as_ref() == outcome_name)
        || !matches!(map.get("payload"), Some(Value { kind: ValueKind::Number(number), .. }) if *number == outcome as u32 as f64)
    { return Err(LanaError::Schema); }
    Ok(Decision {
        schema_version: 1,
        decision_id,
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
    let mut map = Map::new(heap, 26)?;
    let outcome = match decision.outcome {
        PolicyOutcome::Authorize => "Authorize",
        PolicyOutcome::Refuse => "Refuse",
        PolicyOutcome::RequestMoreEvidence => "RequestMoreEvidence",
    };
    map.set(Arc::from("record_schema"), Value::number(1.0), false)?;
    map.set(Arc::from("id"), Value::string(Arc::from(format!("decision/{}", decision.decision_id))), false)?;
    map.set(Arc::from("kind"), Value::string(Arc::from("decision")), false)?;
    map.set(Arc::from("transport_status"), Value::string(Arc::from("ok")), false)?;
    map.set(Arc::from("domain_status"), Value::string(Arc::from(outcome)), false)?;
    map.set(Arc::from("payload"), Value::number(decision.outcome as u32 as f64), false)?;
    map.set(Arc::from("error"), Value::null(), false)?;
    map.set(Arc::from("evidence"), decision.evidence_ids.as_ref().map_or_else(Value::null, |ids| Value::string(ids.clone())), false)?;
    map.set(Arc::from("assumptions"), Value::null(), false)?;
    map.set(Arc::from("exactness"), Value::string(Arc::from("exact")), false)?;
    map.set(Arc::from("metadata"), Value::null(), false)?;
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
    let mut map = Map::new(heap, 19)?;
    map.set(Arc::from("record_schema"), Value::number(1.0), false)?;
    map.set(Arc::from("id"), Value::string(Arc::from(format!("ledger/{}", event.event_id))), false)?;
    map.set(Arc::from("kind"), Value::string(Arc::from("ledger_event")), false)?;
    map.set(Arc::from("transport_status"), Value::string(Arc::from("ok")), false)?;
    map.set(Arc::from("domain_status"), Value::string(Arc::from("recorded")), false)?;
    map.set(Arc::from("payload"), Value::null(), false)?;
    map.set(Arc::from("error"), Value::null(), false)?;
    map.set(Arc::from("evidence"), Value::null(), false)?;
    map.set(Arc::from("assumptions"), Value::null(), false)?;
    map.set(Arc::from("exactness"), Value::string(Arc::from("exact")), false)?;
    map.set(Arc::from("metadata"), Value::null(), false)?;
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
    fn dataset_plan_digest_uses_exact_bytes_and_length_delimited_name() {
        use super::*;
        let mut host = StoreHost::new();
        assert_eq!(host.dataset_plan_digest("plan"), Err(LanaError::InvalidState));
        host.set_chunk_bytes(b"LABC".to_vec());
        assert_eq!(host.dataset_plan_digest("plan").unwrap(),
            "70dfa89b3054841327ddd07a0b88c0af7b53fcd12bdd0a25d2e1a62ff4ad7c8d");
        assert_ne!(host.dataset_plan_digest("plan").unwrap(), host.dataset_plan_digest("plans").unwrap());
        assert_eq!(host.dataset_plan_digest(""), Err(LanaError::InvalidParameters));
    }

    #[test]
    fn execution_rejects_revoked_host_tokens_before_effects() {
        use super::*;
        let mut host = StoreHost::new();
        let capability = host.opaque_token();
        let authorization = host.opaque_token();
        host.execution = Some((capability.clone(), ExecutionCapability::new("test", "https://example.test").unwrap()));
        let mut output = Value::null();
        capability.revoked.store(true, Ordering::Release);
        assert_eq!(host.execution_authorize(&[Value::capability(capability.clone()), Value::null(), Value::null()], &mut output), LanaError::Capability);
        assert_eq!(host.execution_execute(&[Value::capability(capability.clone()), Value::capability(authorization.clone()), Value::null()], &mut output), LanaError::Capability);
        capability.revoked.store(false, Ordering::Release);
        authorization.revoked.store(true, Ordering::Release);
        assert_eq!(host.execution_execute(&[Value::capability(capability), Value::capability(authorization), Value::null()], &mut output), LanaError::Capability);
    }

    #[test]
    fn execution_rejects_forged_decision() {
        use super::*;
        let mut host = StoreHost::new();
        let capability = host.opaque_token();
        host.execution = Some((capability.clone(), ExecutionCapability::new("test", "https://example.test").unwrap()));
        let forged = map(&[
            ("decision_id", number(1.0)), ("policy_id", string("forged")),
            ("policy_version", string("0000000000000000000000000000000000000000000000000000000000000000")),
            ("target", string("webhook")), ("scope", string("test")),
            ("input_revision", number(0.0)), ("outcome", number(0.0)),
            ("evaluation_time", number(0.0)), ("reason", string("forged")),
        ]);
        let plan = map(&[("kind", string("webhook")), ("path", string("/events")), ("payload", Value::null())]);
        let mut output = Value::null();
        assert_eq!(host.execution_authorize(&[Value::capability(capability), forged, plan], &mut output), LanaError::Capability);
    }
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
        let chunk = lana_bytecode::Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        let code = host.dispatch(&mut vm, id, args, &mut out);
        (code, out)
    }

    #[test]
    fn store_callback_imports_decoded_graph_before_vm_publication() {
        let path = temp_store("callback_heap_import");
        let mut host = StoreHost::new();
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        let record = map(&[("value", number(7.0))]);
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_PUT, &[string("record"), record]).0, LanaError::Ok);
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_COMMIT, &[]).0, LanaError::Ok);
        let chunk = lana_bytecode::assembler::assemble(
            "LOAD_STRING R0 7265636f7264\nHOST_CALL store_get R0 1 R1\nRETURN R1\n"
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        vm.set_host_call_extension(Box::new(move |vm, id, args, out| host.dispatch(vm, id, args, out)));
        assert_eq!(vm.run(), LanaError::Ok, "{:?}", vm.error());
        let root = vm.result().unwrap();
        drop(vm);
        assert_eq!(root.print(), "{\"value\": 7}");
        drop(root);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn dataset_source_registers_once_and_reopens_without_partial_rows() {
        let path = temp_store("dataset_source");
        let mut host = StoreHost::new();
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string("s")]).0, LanaError::InvalidState);
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        for id in ["", &"x".repeat(129)] {
            assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string(id)]).0, LanaError::InvalidParameters);
        }
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 0);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string("α/source")]).0, LanaError::Ok);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string("α/source")]).0, LanaError::Ok);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 1);
        drop(host);

        let mut host = StoreHost::new();
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string("α/source")]).0, LanaError::Ok);
        let records = store::store_scan(host.store.as_ref().unwrap(), "dataset/source/").unwrap();
        assert_eq!(records.len(), 1);
        let ValueKind::Map(record) = &records[0].value.kind else { panic!("source record"); };
        let record = record.lock().unwrap();
        assert_eq!(record.get("source_id").unwrap().print(), "α/source");
        assert_eq!(record.get("row_ids").unwrap().print(), "[]");
        assert_eq!(record.get("rows").unwrap().print(), "[]");
        drop(record);
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_PUT, &[string("other"), number(1.0)]).0, LanaError::Ok);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string("new")]).0, LanaError::InvalidState);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 1);
        drop(host);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn dataset_apply_commits_source_only_changes_and_retries_once() {
        fn change(op: &str, id: &str, row: Option<Value>) -> Value {
            let mut fields = vec![("op", string(op)), ("id", string(id))];
            if let Some(row) = row { fields.push(("row", row)); }
            map(&fields)
        }
        fn changes(items: Vec<Value>) -> Value {
            Value::array(Arc::new(Mutex::new(Array::from_items(&lana_vm::heap::Heap::default(), items).unwrap())))
        }
        let path = temp_store("dataset_apply");
        let mut host = StoreHost::new();
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string("s")]).0, LanaError::Ok);
        let add = changes(vec![
            change("add", "a", Some(map(&[("v", number(1.0))]))),
            change("add", "b", Some(map(&[("v", number(2.0))]))),
        ]);
        let args = [string("s"), number(1.0), string("batch-1"), add.clone()];
        let (code, revision) = dispatch(&mut host, LANA_HOST_DATASET_APPLY, &args);
        assert_eq!(code, LanaError::Ok);
        assert_eq!(revision.as_number(), 2.0);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY, &args).1.as_number(), 2.0);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 2);
        drop(host);
        let mut host = StoreHost::new();
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY, &args).1.as_number(), 2.0);
        let changed = [string("s"), number(1.0), string("batch-1"),
            changes(vec![change("add", "a", Some(map(&[("v", number(9.0))])))])];
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY, &changed).0, LanaError::Conflict);
        let update = changes(vec![
            change("correct", "a", Some(map(&[("v", number(3.0))]))),
            change("delete", "b", None),
        ]);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY,
            &[string("s"), number(1.0), string("stale"), update.clone()]).0, LanaError::Conflict);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY,
            &[string("s"), number(2.0), string("batch-2"), update]).1.as_number(), 3.0);
        let source = store::store_get(host.store.as_ref().unwrap(), "dataset/source/73").unwrap();
        let ValueKind::Map(source) = source.kind else { panic!("source record"); };
        let source = source.lock().unwrap();
        let ValueKind::Array(ids) = &source.get("row_ids").unwrap().kind else { panic!("ids"); };
        let ValueKind::Array(rows) = &source.get("rows").unwrap().kind else { panic!("rows"); };
        assert_eq!(ids.lock().unwrap().items().len(), 1);
        assert_eq!(ids.lock().unwrap().items()[0].print(), "a");
        let encoded = rows.lock().unwrap().items()[0].as_string().to_string();
        assert!(encoded.contains("4008000000000000"));
        drop(source);
        let invalid = changes(vec![change("add", "c", Some(map(&[("bad", Value::function(0))])))]);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY,
            &[string("s"), number(3.0), string("bad"), invalid]).0, LanaError::UnsupportedValue);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 3);
        let duplicate = changes(vec![change("delete", "a", None), change("delete", "a", None)]);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY,
            &[string("s"), number(3.0), string("duplicate"), duplicate]).0, LanaError::InvalidParameters);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY,
            &[string("s"), number(3.0), string("existing"),
              changes(vec![change("add", "a", Some(map(&[("v", number(4.0))])))])]).0, LanaError::Conflict);
        store::store_put(host.store.as_mut().unwrap(), "dataset/query/future", &string("registered")).unwrap();
        store::store_commit(host.store.as_mut().unwrap()).unwrap();
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY,
            &[string("s"), number(4.0), string("blocked"),
              changes(vec![change("delete", "a", None)])]).0, LanaError::Corruption);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 4);
        drop(host);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn dataset_plan_reads_ordered_sources_at_one_revision_and_rejects_corrupt_rows() {
        use super::*;
        let chunk = lana_bytecode::assembler::assemble(
            ".function main 0 4\nHALT\n.function plan 2 5\nHOST_CALL dataset_materialize R0 1 R2\nRETURN R2\n.function reject_plan 1 5\nLOAD_FUNCTION R1 reject\nHOST_CALL dataset_filter R0 2 R2\nHOST_CALL dataset_materialize R2 1 R3\nRETURN R3\n.function reject 1 2\nLOAD_CONST R0 false\nRETURN R0\n",
        ).unwrap();
        let chunk_bytes = lana_bytecode::encoder::encode(&chunk);
        let mut vm = Vm::new(&chunk);
        let path = temp_store("dataset_plan_sources");
        let mut host = StoreHost::with_heap(vm.heap());
        host.set_chunk_bytes(chunk_bytes.clone());
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        for source in ["s", "t"] {
            assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string(source)]).0, LanaError::Ok);
        }
        let changes = |id: &str, value: f64| Value::array(Arc::new(Mutex::new(
            Array::from_items(&vm.heap(), vec![map(&[("op", string("add")), ("id", string(id)),
                ("row", map(&[("v", number(value))]))])]).unwrap())));
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY,
            &[string("s"), number(2.0), string("s-1"), changes("a", 1.0)]).0, LanaError::Ok);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY,
            &[string("t"), number(3.0), string("t-1"), changes("b", 7.0)]).0, LanaError::Ok);
        let (revision, ids, result) = host.evaluate_dataset_plan(&mut vm, &["t".into(), "s".into()], "plan").unwrap();
        assert_eq!(revision, 4);
        assert_eq!(ids, vec![vec!["b"], vec!["a"]]);
        assert_eq!(result.print(), "[{\"v\": 7}]");
        let ValueKind::Array(rows) = &result.kind else { panic!("plan rows"); };
        let rows = rows.lock().unwrap();
        assert_eq!(rows.items()[0].derivation.as_ref().unwrap().label.as_ref(), "1:t1:b");
        let ValueKind::Map(first) = &rows.items()[0].kind else { panic!("source map"); };
        assert_eq!(first.lock().unwrap().get("v").unwrap().derivation.as_ref().unwrap().label.as_ref(), "1:t1:b");
        drop(rows);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 4);
        assert_eq!(host.evaluate_dataset_plan(&mut vm, &["s".into()], "reject_plan").unwrap().2.print(), "[]");
        assert_eq!(vm.dataset_decisions().len(), 1);
        let (rejected_revision, _, rejected_bytes) = host.evaluate_dataset_plan_snapshot(&mut vm, "rejected",
            &["s".into()], "reject_plan", "v1").unwrap();
        let rejected = DatasetSnapshot::load(&rejected_bytes, "rejected", rejected_revision,
            &host.dataset_plan_digest("reject_plan").unwrap()).unwrap();
        assert_eq!(rejected.exclusions.len(), 1);
        assert_eq!((rejected.exclusions[0].source_id.as_str(), rejected.exclusions[0].row_id.as_str()), ("s", "a"));
        assert_eq!(rejected.exclusions[0].predicate_value, Some(false));
        let rejected_value = crate::data::json_parse(std::str::from_utf8(&rejected_bytes).unwrap()).unwrap();
        let (code, excluded) = dispatch(&mut host, LANA_HOST_DATASET_EXCLUSIONS, &[rejected_value]);
        assert_eq!(code, LanaError::Ok);
        let ValueKind::Array(excluded) = excluded.kind else { panic!("excluded rows"); };
        assert_eq!(excluded.lock().unwrap().items().len(), 1);
        assert!(matches!(host.evaluate_dataset_plan(&mut vm, &["missing".into()], "plan"), Err(LanaError::NotFound)));
        assert!(vm.dataset_decisions().is_empty());
        assert!(matches!(host.evaluate_dataset_plan(&mut vm, &["".into()], "plan"), Err(LanaError::InvalidParameters)));
        let (_, repeated_ids, result) = host.evaluate_dataset_plan(&mut vm, &["s".into(), "s".into()], "plan").unwrap();
        assert_eq!(repeated_ids, vec![vec!["a"], vec!["a"]]);
        assert_eq!(result.print(), "[{\"v\": 1}]");
        let before = host.evaluate_dataset_plan_with_identity(&mut vm, "q", &["s".into(), "t".into()], "plan").unwrap().3;
        let snapshot_before = host.evaluate_dataset_plan_snapshot(&mut vm, "q", &["s".into(), "t".into()], "plan", "v1").unwrap().2;
        drop(host);

        let mut host = StoreHost::with_heap(vm.heap());
        host.set_chunk_bytes(chunk_bytes);
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        let (revision, ids, result, after) = host.evaluate_dataset_plan_with_identity(&mut vm, "q", &["s".into(), "t".into()], "plan").unwrap();
        assert_eq!(before, after);
        let snapshot_after = host.evaluate_dataset_plan_snapshot(&mut vm, "q", &["s".into(), "t".into()], "plan", "v1").unwrap().2;
        assert_eq!(snapshot_before, snapshot_after);
        assert_eq!(revision, 4);
        assert_eq!(ids, vec![vec!["a"], vec!["b"]]);
        assert_eq!(result.print(), "[{\"v\": 1}]");
        let ValueKind::Array(rows) = &result.kind else { panic!("reopened plan rows"); };
        assert_eq!(rows.lock().unwrap().items()[0].derivation.as_ref().unwrap().label.as_ref(), "1:s1:a");
        let store = host.store.as_mut().unwrap();
        let source = store::store_get(store, "dataset/source/73").unwrap();
        let ValueKind::Map(record) = &source.kind else { panic!("source record"); };
        record.lock().unwrap().set(Arc::from("rows"), Value::array(Arc::new(Mutex::new(
            Array::from_items(&vm.heap(), vec![string("invalid tagged row")]).unwrap()))), false).unwrap();
        store::store_put(store, "dataset/source/73", &source).unwrap();
        store::store_commit(store).unwrap();
        assert!(matches!(host.evaluate_dataset_plan(&mut vm, &["s".into(), "t".into()], "plan"), Err(LanaError::Corruption)));
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 5);
        drop(host);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn dataset_query_commits_initial_snapshot_and_rebinds_after_reopen() {
        let chunk = lana_bytecode::assembler::assemble(
            ".function main 0 4\nHALT\n.function plan 1 4\nHOST_CALL dataset_materialize R0 1 R1\nRETURN R1\n",
        ).unwrap();
        let chunk_bytes = lana_bytecode::encoder::encode(&chunk);
        let mut vm = Vm::new(&chunk);
        let path = temp_store("dataset_query");
        let mut host = StoreHost::with_heap(vm.heap());
        host.set_chunk_bytes(chunk_bytes.clone());
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string("s")]).0, LanaError::Ok);
        let changes = Value::array(Arc::new(Mutex::new(Array::from_items(&vm.heap(), vec![
            map(&[("op", string("add")), ("id", string("a")),
                ("row", map(&[("v", number(7.0))]))]),
        ]).unwrap())));
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_APPLY,
            &[string("s"), number(1.0), string("seed"), changes]).0, LanaError::Ok);
        let source_ids = Value::array(Arc::new(Mutex::new(Array::from_items(&vm.heap(), vec![string("s")]).unwrap())));
        let args = [string("q"), source_ids.clone(), string("plan"), string("v1")];
        let mut out = Value::null();
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_QUERY, &args, &mut out), LanaError::Ok);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 3);
        let snapshot = store::store_get(host.store.as_ref().unwrap(), "dataset/snapshot/71/3").unwrap();
        let ValueKind::String(snapshot) = snapshot.kind else { panic!("snapshot bytes"); };
        let digest = host.dataset_plan_digest("plan").unwrap();
        assert_eq!(DatasetSnapshot::load(snapshot.as_bytes(), "q", 2, &digest).unwrap().rows.len(), 1);
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_QUERY,
            &[string("missing"), source_ids.clone(), string("absent"), string("v1")], &mut out), LanaError::NotFound);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 3);
        host.store.as_ref().unwrap().ensure_clean().unwrap();
        drop(host);

        let mut host = StoreHost::with_heap(vm.heap());
        host.set_chunk_bytes(chunk_bytes);
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_QUERY, &args, &mut out), LanaError::Ok);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 3);
        host.set_chunk_bytes(b"changed program".to_vec());
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_QUERY, &args, &mut out), LanaError::Conflict);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 3);
        host.set_chunk_bytes(lana_bytecode::encoder::encode(&chunk));
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_QUERY,
            &[string("q"), source_ids, string("plan"), string("v2")], &mut out), LanaError::Ok);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 4);
        assert!(store::store_get(host.store.as_ref().unwrap(), "dataset/snapshot/71/3").is_ok());
        assert!(store::store_get(host.store.as_ref().unwrap(), "dataset/snapshot/71/4").is_ok());
        store::store_put(host.store.as_mut().unwrap(), "dataset/snapshot/71/4", &string("broken")).unwrap();
        store::store_commit(host.store.as_mut().unwrap()).unwrap();
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SNAPSHOT,
            &[string("q"), Value::null()]).0, LanaError::Corruption);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SNAPSHOT,
            &[string("q"), number(3.0)]).0, LanaError::Ok);
        drop(host);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn dataset_apply_reruns_all_bound_queries_in_one_revision() {
        let chunk = lana_bytecode::assembler::assemble(
            ".function main 0 4\nHALT\n.function plan 1 4\nHOST_CALL dataset_materialize R0 1 R1\nRETURN R1\n",
        ).unwrap();
        let chunk_bytes = lana_bytecode::encoder::encode(&chunk);
        let mut vm = Vm::new(&chunk);
        let path = temp_store("dataset_dependent_apply");
        let mut host = StoreHost::with_heap(vm.heap());
        host.set_chunk_bytes(chunk_bytes.clone());
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string("s")]).0, LanaError::Ok);
        let sources = Value::array(Arc::new(Mutex::new(Array::from_items(&vm.heap(), vec![string("s")]).unwrap())));
        let mut out = Value::null();
        for id in ["q1", "q2"] {
            assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_QUERY,
                &[string(id), sources.clone(), string("plan"), string("v1")], &mut out), LanaError::Ok);
        }
        let heap = vm.heap();
        let changes = |op: &str, value: f64| Value::array(Arc::new(Mutex::new(Array::from_items(&heap, vec![
            map(&[("op", string(op)), ("id", string("a")), ("row", map(&[("v", number(value))]))]),
        ]).unwrap())));
        let args = [string("s"), number(3.0), string("add-a"), changes("add", 1.0)];
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_APPLY, &args, &mut out), LanaError::Ok);
        assert_eq!(out.as_number(), 4.0);
        let digest = host.dataset_plan_digest("plan").unwrap();
        for (id, hex) in [("q1", "7131"), ("q2", "7132")] {
            let saved = store::store_get(host.store.as_ref().unwrap(), &format!("dataset/snapshot/{hex}/4")).unwrap();
            let ValueKind::String(saved) = saved.kind else { panic!("snapshot bytes"); };
            assert_eq!(DatasetSnapshot::load(saved.as_bytes(), id, 4, &digest).unwrap().rows.len(), 1);
            assert_eq!(saved.as_bytes(), host.evaluate_dataset_plan_snapshot(&mut vm,
                id, &["s".into()], "plan", "v1").unwrap().2);
        }
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_APPLY, &args, &mut out), LanaError::Ok);
        assert_eq!(out.as_number(), 4.0);
        drop(host);

        let mut host = StoreHost::with_heap(vm.heap());
        host.set_chunk_bytes(chunk_bytes);
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        let (code, historical) = dispatch(&mut host, LANA_HOST_DATASET_SNAPSHOT,
            &[string("q1"), number(4.0)]);
        assert_eq!(code, LanaError::Ok);
        let parsed = DatasetSnapshot::from_value(&historical).unwrap();
        assert_eq!(parsed.rows.len(), 1);
        let (code, initial) = dispatch(&mut host, LANA_HOST_DATASET_SNAPSHOT,
            &[string("q1"), number(2.0)]);
        assert_eq!(code, LanaError::Ok);
        assert!(DatasetSnapshot::from_value(&initial).unwrap().rows.is_empty());
        let (code, evidence) = dispatch(&mut host, LANA_HOST_DATASET_EVIDENCE,
            &[historical.clone(), string(&parsed.row_ids[0])]);
        assert_eq!(code, LanaError::Ok);
        let ValueKind::Map(evidence) = evidence.kind else { panic!("evidence map"); };
        let source_rows = evidence.lock().unwrap().get("source_rows").unwrap().clone();
        let ValueKind::Array(source_rows) = source_rows.kind else { panic!("source rows"); };
        assert_eq!(source_rows.lock().unwrap().items().len(), 1);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_EVIDENCE,
            &[historical.clone(), string("missing")]).0, LanaError::NotFound);
        let (code, excluded) = dispatch(&mut host, LANA_HOST_DATASET_EXCLUSIONS, &[historical]);
        assert_eq!(code, LanaError::Ok);
        let ValueKind::Array(excluded) = excluded.kind else { panic!("exclusions"); };
        assert!(excluded.lock().unwrap().items().is_empty());
        let correction = [string("s"), number(4.0), string("correct-a"), changes("correct", 2.0)];
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_APPLY, &correction, &mut out), LanaError::Conflict);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 4);
        for id in ["q1", "q2"] {
            assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_QUERY,
                &[string(id), sources.clone(), string("plan"), string("v1")], &mut out), LanaError::Ok);
        }
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_APPLY, &correction, &mut out), LanaError::Ok);
        assert_eq!(out.as_number(), 5.0);
        for hex in ["7131", "7132"] {
            assert!(store::store_get(host.store.as_ref().unwrap(), &format!("dataset/snapshot/{hex}/5")).is_ok());
        }
        store::store_compact(host.store.as_mut().unwrap(), 0).unwrap();
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SNAPSHOT,
            &[string("q1"), number(4.0)]).0, LanaError::CompactedHistory);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SNAPSHOT,
            &[string("q1"), Value::null()]).0, LanaError::Ok);
        drop(host);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn dataset_captured_updates_match_clean_reruns_after_restart() {
        let chunk = lana_bytecode::assembler::assemble(
            ".function main 0 2\nHALT\n.function plan 1 2\nHOST_CALL dataset_materialize R0 1 R1\nRETURN R1\n",
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        let path = temp_store("dataset_captured_rerun");
        let mut host = StoreHost::with_heap(vm.heap());
        host.set_chunk_bytes(lana_bytecode::encoder::encode(&chunk));
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string("s")]).0, LanaError::Ok);
        let mut out = Value::null();
        let sources = crate::information_codec::array(&vm, vec![string("s")]).unwrap();
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_QUERY,
            &[string("q"), sources.clone(), string("plan"), string("v1")], &mut out), LanaError::Ok);
        let law = Value::possibility(vm.possibility_build(&[number(1.), number(2.)]).unwrap());
        let captured = vm.information_snapshot(&law).unwrap();
        for (index, op) in ["add", "correct", "delete"].iter().enumerate() {
            let change = if *op == "delete" { map(&[("op", string(op)), ("id", string("a"))]) }
                else { map(&[("op", string(op)), ("id", string("a")), ("row", map(&[("v", captured.clone())]))]) };
            let changes = crate::information_codec::array(&vm, vec![change]).unwrap();
            assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_APPLY,
                &[string("s"), number((index + 2) as f64), string(op), changes], &mut out), LanaError::Ok);
            let revision = index + 3;
            let ValueKind::String(saved) = store::store_get(host.store.as_ref().unwrap(), &format!("dataset/snapshot/71/{revision}")).unwrap().kind else { panic!() };
            assert_eq!(saved.as_bytes(), host.evaluate_dataset_plan_snapshot(&mut vm, "q", &["s".into()], "plan", "v1").unwrap().2);
            drop(host);
            vm = Vm::new(&chunk);
            host = StoreHost::with_heap(vm.heap());
            host.set_chunk_bytes(lana_bytecode::encoder::encode(&chunk));
            assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
            assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_QUERY,
                &[string("q"), sources.clone(), string("plan"), string("v1")], &mut out), LanaError::Ok);
            assert_eq!(saved.as_bytes(), host.evaluate_dataset_plan_snapshot(&mut vm, "q", &["s".into()], "plan", "v1").unwrap().2);
        }
        drop(host);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn dataset_apply_failed_rerun_keeps_source_and_query_unchanged() {
        let chunk = lana_bytecode::assembler::assemble(
            ".function main 0 4\nHALT\n.function plan 1 8\nLOAD_STRING R1 76\nARRAY_NEW R2 R1 1\nMOVE R3 R0\nMOVE R4 R2\nHOST_CALL dataset_select R3 2 R5\nHOST_CALL dataset_materialize R5 1 R6\nRETURN R6\n",
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        let path = temp_store("dataset_failed_rerun");
        let mut host = StoreHost::with_heap(vm.heap());
        host.set_chunk_bytes(lana_bytecode::encoder::encode(&chunk));
        assert_eq!(dispatch(&mut host, LANA_HOST_STORE_OPEN, &[string(&path)]).0, LanaError::Ok);
        assert_eq!(dispatch(&mut host, LANA_HOST_DATASET_SOURCE, &[string("s")]).0, LanaError::Ok);
        let sources = Value::array(Arc::new(Mutex::new(Array::from_items(&vm.heap(), vec![string("s")]).unwrap())));
        let mut out = Value::null();
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_QUERY,
            &[string("q"), sources, string("plan"), string("v1")], &mut out), LanaError::Ok);
        let heap = vm.heap();
        let change = |column: &str| Value::array(Arc::new(Mutex::new(Array::from_items(&heap, vec![
            map(&[("op", string("add")), ("id", string("a")), ("row", map(&[(column, number(1.0))]))]),
        ]).unwrap())));
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_APPLY,
            &[string("s"), number(2.0), string("add-a"), change("w")], &mut out), LanaError::Type);
        assert_eq!(store::store_current_revision(host.store.as_ref().unwrap()).unwrap().revision_id, 2);
        host.store.as_ref().unwrap().ensure_clean().unwrap();
        assert_eq!(host.dispatch(&mut vm, LANA_HOST_DATASET_APPLY,
            &[string("s"), number(2.0), string("add-a"), change("v")], &mut out), LanaError::Ok);
        assert_eq!(out.as_number(), 3.0);
        drop(host);
        std::fs::remove_dir_all(path).unwrap();
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
        let (code, _) = dispatch(&mut host, LANA_HOST_ADAPTER_LOAD, &[number(2.5), string("")]);
        assert_eq!(code, LanaError::UnsupportedOperation);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn adapter_sqlite_reaches_host_call() {
        let path = std::env::temp_dir().join(format!("lana-host-sqlite-{}.db", std::process::id()));
        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute_batch("CREATE TABLE evidence(p REAL); INSERT INTO evidence VALUES (0.8);").unwrap();
        drop(db);
        let mut host = StoreHost::new();
        let (code, _) = dispatch(&mut host, LANA_HOST_ADAPTER_LOAD,
            &[number(2.0), string(path.to_str().unwrap())]);
        assert_eq!(code, LanaError::Ok);
        let (code, out) = dispatch(&mut host, LANA_HOST_ADAPTER_FETCH,
            &[string("SELECT p FROM evidence")]);
        assert_eq!(code, LanaError::Ok);
        let ValueKind::Map(map) = out.kind else { panic!("expected map") };
        assert_eq!(map.lock().unwrap().get("p").unwrap().as_number(), 0.8);
        drop(host);
        std::fs::remove_file(path).unwrap();
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
        assert_eq!(map.get("record_schema").unwrap().as_number(), 1.0);
        assert_eq!(map.get("kind").unwrap().as_string(), Arc::from("decision"));
        execution::validate_record_envelope(&map, "decision").unwrap();
        drop(map);
        assert!(decision_from_value(&out).is_ok());
        let ValueKind::Map(map) = &out.kind else { unreachable!() };
        map.lock().unwrap().set(Arc::from("record_schema"), number(2.0), false).unwrap();
        assert!(matches!(decision_from_value(&out), Err(LanaError::Schema)));
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
        assert_eq!(event_map.lock().unwrap().get("record_schema").unwrap().as_number(), 1.0);
        execution::validate_record_envelope(&event_map.lock().unwrap(), "ledger_event").unwrap();

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
