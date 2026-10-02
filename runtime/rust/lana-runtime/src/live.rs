//! Process-local persistent programs over the VM's transactional Information graph.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use lana_bytecode::{Chunk, LanaError};
use lana_vm::heap::Reservation;
use lana_vm::value::ValueKind;
use lana_vm::{Value, Vm};
use serde_json::{json, Value as Json};

use crate::{data, host_calls::StoreHost, information_codec::Tagged};

const MAX_QUEUE_BYTES: usize = 64 * 1024 * 1024;
const EVENT_REPLY_OVERHEAD: usize = 256;
static NEXT_HOST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct LiveFailure {
    pub code: LanaError,
    pub message: String,
}

impl LiveFailure {
    fn new(code: LanaError, message: impl Into<String>) -> Self { Self { code, message: message.into() } }
}

impl From<LanaError> for LiveFailure {
    fn from(code: LanaError) -> Self { Self::new(code, code.name()) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveState { Live, Quiescent, Suspended, Failed, Deleted }

impl LiveState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Live => "LIVE", Self::Quiescent => "QUIESCENT",
            Self::Suspended => "SUSPENDED", Self::Failed => "FAILED", Self::Deleted => "DELETED",
        }
    }
}

struct QueuedObservation { name: String, evidence: String, bytes: usize, _reservation: Reservation }

struct LiveInstance {
    vm: Vm<'static>,
    state: LiveState,
    queue: VecDeque<QueuedObservation>,
    queue_bytes: usize,
    inspections: BTreeMap<String, Json>,
}

impl LiveInstance {
    fn refresh_inspections(&mut self) -> Result<(), LiveFailure> {
        for name in self.vm.live_names() {
            if self.inspections.get(&name).and_then(|value| value.get("revision"))
                .and_then(Json::as_u64) == self.vm.live_revision(&name).ok()
                && self.inspections.get(&name).and_then(|value| value.get("partial")) != Some(&Json::Bool(true)) {
                continue;
            }
            match self.vm.live_inspect(&name).map_err(LiveFailure::from)
                .and_then(|value| encode_value(&value)) {
                Ok(value) => { self.inspections.insert(name, value); }
                Err(error) => {
                    let revision = self.vm.live_revision(&name)?;
                    self.inspections.insert(name, json!({"revision":revision,"reactive":true,"partial":true}));
                    return Err(error);
                }
            }
        }
        Ok(())
    }
}

pub struct LiveHost {
    host_id: String,
    next_handle: u64,
    instances: BTreeMap<String, LiveInstance>,
}

impl Default for LiveHost {
    fn default() -> Self {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|time| time.as_nanos()).unwrap_or(0);
        let sequence = NEXT_HOST_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            host_id: format!("{:x}-{:x}-{:x}", std::process::id(), nanos, sequence),
            next_handle: 0,
            instances: BTreeMap::new(),
        }
    }
}

impl LiveHost {
    pub fn new() -> Self { Self::default() }

    pub fn start_live(&mut self, chunk: Arc<Chunk>) -> Result<String, LiveFailure> {
        let bytes = lana_bytecode::encoder::encode(&chunk);
        self.start_live_with(chunk, bytes, |_| Ok(()))
    }

    pub fn start_live_labc(&mut self, bytes: &[u8]) -> Result<String, LiveFailure> {
        let chunk = lana_bytecode::loader::load(bytes)
            .map_err(|error| LiveFailure::new(error.code, error.message))?;
        self.start_live_with(Arc::new(chunk), bytes.to_vec(), |_| Ok(()))
    }

    pub fn start_live_labc_with<F>(&mut self, bytes: &[u8], configure: F) -> Result<String, LiveFailure>
    where F: FnOnce(&mut Vm<'static>) -> Result<(), LanaError> {
        let chunk = lana_bytecode::loader::load(bytes)
            .map_err(|error| LiveFailure::new(error.code, error.message))?;
        self.start_live_with(Arc::new(chunk), bytes.to_vec(), configure)
    }

    fn start_live_with<F>(&mut self, chunk: Arc<Chunk>, bytes: Vec<u8>, configure: F) -> Result<String, LiveFailure>
    where F: FnOnce(&mut Vm<'static>) -> Result<(), LanaError> {
        let mut vm = Vm::new_owned(chunk);
        configure(&mut vm)?;
        vm.capture_output();
        let mut store = StoreHost::new();
        store.set_chunk_bytes(bytes);
        vm.set_host_call_extension(Box::new(move |vm, id, args, out| store.dispatch(vm, id, args, out)));
        let status = vm.run();
        if status != LanaError::Ok { return Err(LiveFailure::new(status, vm.error().message.clone())); }
        self.next_handle = self.next_handle.checked_add(1).ok_or(LanaError::Limit)?;
        let handle = format!("lanaprog_{}_{}", self.host_id, self.next_handle);
        let mut instance = LiveInstance {
            vm, state: LiveState::Quiescent, queue: VecDeque::new(), queue_bytes: 0,
            inspections: BTreeMap::new(),
        };
        instance.refresh_inspections()?;
        self.instances.insert(handle.clone(), instance);
        Ok(handle)
    }

    pub fn state(&self, handle: &str) -> Result<LiveState, LiveFailure> {
        Ok(self.instance(handle)?.state)
    }

    pub fn names(&self, handle: &str) -> Result<Vec<String>, LiveFailure> {
        Ok(self.instance(handle)?.vm.live_names())
    }

    pub fn output(&self, handle: &str) -> Result<String, LiveFailure> {
        Ok(self.instance(handle)?.vm.output().unwrap_or_default())
    }

    pub fn inspect_live(&mut self, handle: &str, name: &str) -> Result<Json, LiveFailure> {
        let instance = self.instance_mut(handle)?;
        let inspection = if instance.state == LiveState::Failed {
            let revision = instance.vm.live_revision(name)?;
            instance.inspections.get(name)
                .filter(|cached| cached.get("revision").and_then(Json::as_u64) == Some(revision))
                .cloned()
                .unwrap_or_else(|| json!({"revision":revision,"reactive":true,"partial":true}))
        } else {
            let inspection = match instance.vm.live_inspect(name).map_err(LiveFailure::from)
                .and_then(|value| encode_value(&value)) {
                Ok(inspection) => inspection,
                Err(error) => {
                    if terminal(error.code) { instance.state = LiveState::Failed; }
                    return Err(error);
                }
            };
            instance.inspections.insert(name.to_owned(), inspection.clone());
            inspection
        };
        Ok(json!({"state":instance.state.as_str(),"name":name,"inspection":inspection}))
    }

    pub fn observe_live(&mut self, handle: &str, name: &str, evidence: Json) -> Result<Json, LiveFailure> {
        let instance = self.instance_mut(handle)?;
        if instance.state == LiveState::Failed { return Err(LiveFailure::new(LanaError::Task, "live instance failed")); }
        if !instance.vm.live_is_root(name)? { return Err(LanaError::Type.into()); }
        if instance.state == LiveState::Suspended {
            let evidence = serde_json::to_string(&evidence).map_err(|_| LanaError::Schema)?;
            let bytes = evidence.capacity().checked_add(name.len())
                .and_then(|size| size.checked_add(2 * std::mem::size_of::<QueuedObservation>() + EVENT_REPLY_OVERHEAD))
                .ok_or(LanaError::Limit)?;
            let total = instance.queue_bytes.checked_add(bytes).ok_or(LanaError::Limit)?;
            if total > MAX_QUEUE_BYTES {
                return Err(LiveFailure::new(LanaError::Limit, "live observation queue limit exceeded"));
            }
            let reservation = instance.vm.heap().reserve(bytes)
                .map_err(|_| LiveFailure::new(LanaError::Limit, "live observation queue limit exceeded"))?;
            instance.queue.push_back(QueuedObservation { name: name.to_owned(), evidence, bytes, _reservation: reservation });
            instance.queue_bytes = total;
            return Ok(json!({"state":"SUSPENDED","queued":instance.queue.len()}));
        }
        Self::apply(instance, name, &evidence)
    }

    pub fn pause_live(&mut self, handle: &str) -> Result<Json, LiveFailure> {
        let instance = self.instance_mut(handle)?;
        if instance.state == LiveState::Failed { return Err(LiveFailure::new(LanaError::Task, "live instance failed")); }
        instance.state = LiveState::Suspended;
        Ok(json!({"state":"SUSPENDED","queued":instance.queue.len()}))
    }

    pub fn resume_live(&mut self, handle: &str) -> Result<Json, LiveFailure> {
        let instance = self.instance_mut(handle)?;
        if instance.state == LiveState::Failed { return Err(LiveFailure::new(LanaError::Task, "live instance failed")); }
        instance.state = LiveState::Quiescent;
        let mut events = Vec::new();
        while let Some(event) = instance.queue.pop_front() {
            let QueuedObservation { name, evidence, bytes, _reservation } = event;
            instance.queue_bytes -= bytes;
            drop(_reservation);
            let result = serde_json::from_str(&evidence)
                .map_err(|error| LiveFailure::new(LanaError::Schema, error.to_string()))
                .and_then(|value| Self::apply(instance, &name, &value));
            let outcome = match result {
                Ok(value) => json!({"name":name,"ok":true,"result":value}),
                Err(error) => json!({"name":name,"ok":false,"error":{"code":error.code.name(),"message":error.message}}),
            };
            events.push(outcome);
            if instance.state == LiveState::Failed { break; }
        }
        Ok(json!({"state":instance.state.as_str(),"events":events,"queued":instance.queue.len()}))
    }

    pub fn delete_live(&mut self, handle: &str) -> Result<Json, LiveFailure> {
        self.instances.remove(handle).ok_or(LanaError::NotFound)?;
        Ok(json!({"state":"DELETED","handle":handle}))
    }

    fn instance(&self, handle: &str) -> Result<&LiveInstance, LiveFailure> {
        self.instances.get(handle).ok_or_else(|| LanaError::NotFound.into())
    }

    fn instance_mut(&mut self, handle: &str) -> Result<&mut LiveInstance, LiveFailure> {
        self.instances.get_mut(handle).ok_or_else(|| LanaError::NotFound.into())
    }

    fn apply(instance: &mut LiveInstance, name: &str, evidence: &Json) -> Result<Json, LiveFailure> {
        if !instance.vm.live_is_root(name)? { return Err(LanaError::Type.into()); }
        let evidence = match decode_evidence(&mut instance.vm, name, evidence) {
            Ok(value) => value,
            Err(error) => {
                if terminal(error.code) { instance.state = LiveState::Failed; }
                return Err(error);
            }
        };
        instance.state = LiveState::Live;
        let result = instance.vm.live_observe(name, &evidence);
        match result {
            Ok(_) => {
                let revision = instance.vm.live_revision(name)?;
                let inspection_error = instance.refresh_inspections().err();
                instance.state = if inspection_error.as_ref().is_some_and(|error| terminal(error.code)) {
                    LiveState::Failed
                } else { LiveState::Quiescent };
                Ok(json!({"state":instance.state.as_str(),"name":name,"revision":revision,
                    "inspection_error":inspection_error.map(|error| json!({"code":error.code.name(),"message":error.message}))}))
            }
            Err(code) => {
                if terminal(code) {
                    instance.state = LiveState::Failed;
                } else { instance.state = LiveState::Quiescent; }
                Err(code.into())
            }
        }
    }
}

fn terminal(code: LanaError) -> bool {
    matches!(code, LanaError::Oom | LanaError::Limit | LanaError::BudgetExhausted
        | LanaError::Cancelled | LanaError::Task | LanaError::Corruption)
}

fn current_values(vm: &Vm, name: &str) -> Result<Vec<Value>, LiveFailure> {
    let current = vm.live_current(name)?;
    if let ValueKind::Possibility(possibility) = &current.kind {
        Ok(possibility.values.clone())
    } else { Ok(vec![current]) }
}

fn select_existing(values: &[Value], evidence: &Json) -> Result<Option<Value>, LiveFailure> {
    let mut found = None;
    for value in values {
        if encode_value(value).ok().as_ref() == Some(evidence) {
            if found.is_some() {
                return Err(LiveFailure::new(LanaError::InvalidConditioning, "ambiguous JSON evidence"));
            }
            found = Some(value.clone());
        }
    }
    Ok(found)
}

fn decode_evidence(vm: &mut Vm, name: &str, evidence: &Json) -> Result<Value, LiveFailure> {
    let current = current_values(vm, name)?;
    if let Some(value) = select_existing(&current, evidence)? { return Ok(value); }
    if let Some(values) = evidence.as_object().filter(|object| object.len() == 1)
        .and_then(|object| object.get("possibility")) {
        let values = values.as_array().ok_or(LanaError::Schema)?;
        let values = values.iter().map(|value| {
            select_existing(&current, value)?.map(Ok).unwrap_or_else(||
                data::json_parse_with_heap(&value.to_string(), &vm.heap()).map_err(LiveFailure::from))
        })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Value::possibility(vm.possibility_build(&values)?));
    }
    if evidence.get("tag").is_some() {
        if let Ok(tagged) = serde_json::from_value::<Tagged>(evidence.clone()) {
            return Ok(tagged.to_live(vm)?);
        }
    }
    Ok(data::json_parse_with_heap(&evidence.to_string(), &vm.heap())?)
}

fn encode_value(value: &Value) -> Result<Json, LiveFailure> {
    if let Ok(text) = data::json_stringify(value) {
        return serde_json::from_str(&text).map_err(|_| LanaError::Schema.into());
    }
    let tagged = Tagged::plain(value, 0)?;
    serde_json::to_value(tagged).map_err(|_| LanaError::Schema.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROGRAM: &str = r#".version 5
LOAD_CONST R0 1
LOAD_CONST R1 2
ARRAY_NEW R2 R0 2
POSSIBILITY_BUILD R2 R3
HOST_CALL information_new R3 1 R4
LOAD_CONST R5 1
BINARY R4 + R5 R6
LOAD_STRING R7 726f6f74
MOVE R8 R4
HOST_CALL live_register R7 2 R9
LOAD_STRING R10 64657269766564
MOVE R11 R6
HOST_CALL live_register R10 2 R12
HALT
"#;

    fn program() -> Vec<u8> {
        lana_bytecode::encoder::encode(&lana_bytecode::assemble(PROGRAM).unwrap())
    }

    #[test]
    fn retained_graph_observations_and_lifecycle() {
        let mut host = LiveHost::new();
        let bytes = program();
        let first = host.start_live_labc(&bytes).unwrap();
        let second = host.start_live_labc(&bytes).unwrap();
        assert_ne!(first, second);
        assert_eq!(host.inspect_live(&first, "derived").unwrap()["inspection"]["support"][0]["value"], 2);
        assert_eq!(host.inspect_live(&first, "root").unwrap()["inspection"]["revision"], 0);
        host.pause_live(&first).unwrap();
        host.observe_live(&first, "root", json!(7)).unwrap();
        host.observe_live(&first, "root", json!(2)).unwrap();
        let resumed = host.resume_live(&first).unwrap();
        assert_eq!(resumed["events"][0]["error"]["code"], "LANA_ERR_INVALID_CONDITIONING");
        assert_eq!(resumed["events"][1]["ok"], true);
        let derived = host.inspect_live(&first, "derived").unwrap();
        assert_eq!(derived["inspection"]["revision"], 1);
        assert_eq!(derived["inspection"]["support"][0]["value"], 3);
        assert_eq!(host.inspect_live(&second, "root").unwrap()["inspection"]["revision"], 0);
        assert_eq!(host.observe_live(&first, "derived", json!(3)).unwrap_err().code, LanaError::Type);
        host.delete_live(&first).unwrap();
        assert_eq!(host.inspect_live(&first, "root").unwrap_err().code, LanaError::NotFound);
    }

    #[test]
    fn queue_budget_and_terminal_failure() {
        let mut host = LiveHost::new();
        let handle = host.start_live_labc(&program()).unwrap();
        host.instance_mut(&handle).unwrap().vm.set_memory_limit(2 * 1024 * 1024).unwrap();
        host.pause_live(&handle).unwrap();
        let large = Json::String("x".repeat(2 * 1024 * 1024));
        assert_eq!(host.observe_live(&handle, "root", large).unwrap_err().code, LanaError::Limit);
        assert_eq!(host.resume_live(&handle).unwrap()["events"].as_array().unwrap().len(), 0);
        let before_queue = host.instance(&handle).unwrap().vm.allocated_bytes();
        host.pause_live(&handle).unwrap();
        host.observe_live(&handle, "root", json!(7)).unwrap();
        assert!(host.instance(&handle).unwrap().vm.allocated_bytes() > before_queue);
        assert_eq!(host.resume_live(&handle).unwrap()["events"][0]["ok"], false);
        assert_eq!(host.instance(&handle).unwrap().vm.allocated_bytes(), before_queue);
        let vm = &mut host.instance_mut(&handle).unwrap().vm;
        vm.set_memory_limit(vm.allocated_bytes() + 1).unwrap();
        assert_eq!(host.observe_live(&handle, "root", json!(2)).unwrap_err().code, LanaError::Oom);
        assert_eq!(host.state(&handle).unwrap(), LiveState::Failed);
        assert_eq!(host.inspect_live(&handle, "root").unwrap()["inspection"]["revision"], 0);
        host.delete_live(&handle).unwrap();
        assert_eq!(LiveHost::new().state(&handle).unwrap_err().code, LanaError::NotFound);
    }

    #[test]
    fn inspection_failure_replaces_stale_cache() {
        let mut host = LiveHost::new();
        let handle = host.start_live_labc(&program()).unwrap();
        let instance = host.instance_mut(&handle).unwrap();
        instance.inspections.insert("root".into(), json!({"revision":99,"support":[]}));
        let allocated = instance.vm.allocated_bytes();
        instance.vm.set_memory_limit(allocated + 1).unwrap();
        assert_eq!(instance.refresh_inspections().unwrap_err().code, LanaError::Oom);
        assert_eq!(instance.inspections["root"]["revision"], 0);
        assert_eq!(instance.inspections["root"]["partial"], true);
    }
}
