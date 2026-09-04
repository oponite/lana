//! Runtime value model, mirroring `Value` in `vm/include/value.h` and
//! `vm/c/value.c`.
//!
//! The C11 VM uses a tagged union with heap pointers and a mark-sweep GC. The
//! Rust VM uses an owned/shared representation: `Arc` for shared heap objects,
//! `Mutex` where the C code mutates in place (arrays, maps). The value graph
//! is acyclic — derivations are immutable records forming a DAG, state dists
//! form a tree, reactives form a dependency DAG — so `Arc` without cycle
//! collection is sound. `Arc`/`Mutex` (rather than `Rc`/`RefCell`) keep every
//! value `Send`, so a child VM's value graph can cross a task boundary.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use lana_bytecode::{LanaError, ValueType};

use crate::derivation::{Derivation, DerivationExactness};
use crate::state::StateValue;

/// A runtime failure, mirroring the fields `lana_vm_run` records before
/// returning an error code. Defined here (rather than in `vm`) so the shared
/// task state can carry a child VM's error across a task boundary.
#[derive(Debug, Clone)]
pub struct VmError {
    pub code: LanaError,
    pub ip: usize,
    pub opcode: u8,
    pub line: u32,
    pub message: String,
    pub function: String,
    pub operation: String,
    /// The resolution reason, matching `LanaResolutionReason` in
    /// `vm/include/error.h`. `None` when the failure carries no resolution
    /// detail.
    pub resolution_reason: u32,
    /// The number of remaining alternatives for an unresolved-value failure,
    /// matching `LanaErrorInfo.remaining_alternatives`.
    pub remaining_alternatives: usize,
    /// Cancellation detail `(task_lineage, reason)`, matching
    /// `LanaErrorInfo.cancellation`.
    pub cancellation: Option<(u64, String)>,
    /// Resource-limit detail `(resource, limit, observed, unit)`, matching
    /// `LanaErrorInfo.resource_limit`.
    pub resource_limit: Option<(u32, u64, u64, String)>,
    /// Exact-support detail `(support, detail)`, matching
    /// `LanaErrorInfo.exact_support`.
    pub exact_support: Option<(u32, String)>,
}

impl Default for VmError {
    fn default() -> Self {
        Self {
            code: LanaError::Ok,
            ip: 0,
            opcode: 0,
            line: 0,
            message: String::new(),
            function: String::new(),
            operation: String::new(),
            resolution_reason: 0,
            remaining_alternatives: 0,
            cancellation: None,
            resource_limit: None,
            exact_support: None,
        }
    }
}

/// The joint-law kind, matching `LanaJointKind` in `vm/include/value.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum JointKind {
    Independent = 0,
    FiniteLaw,
    Conditional,
    Projected,
}

/// Joint capabilities, matching `LanaJointCapability`.
pub const LANA_JOINT_CAN_PROJECT: u32 = 1 << 0;
pub const LANA_JOINT_CAN_CONDITION: u32 = 1 << 1;
pub const LANA_JOINT_CAN_SAMPLE: u32 = 1 << 2;
pub const LANA_JOINT_CAN_RESOLVE: u32 = 1 << 3;

/// One row of a finite correlated law, matching `LanaJointRow`.
#[derive(Debug, Clone)]
pub struct JointRow {
    pub values: Vec<Value>,
    pub weight: f64,
}

/// A named product-space law or view, mirroring `struct LanaJointState` in
/// `vm/include/value.h`. Independent marginals live in `values`; a finite
/// correlated law lives in `rows` (with `values` empty).
#[derive(Debug, Clone)]
pub struct JointState {
    pub names: Vec<Arc<str>>,
    pub domains: Vec<ValueType>,
    pub values: Vec<Value>,
    pub rows: Vec<JointRow>,
    pub kind: JointKind,
    pub capabilities: u32,
}

/// A lazy state distribution, mirroring `struct LanaStateDist` in
/// `vm/include/value.h`. Dirac/append/transform nodes form a tree, so `Arc`
/// without cycle collection is sound.
#[derive(Debug, Clone)]
pub struct StateDist {
    pub kind: StateDistKind,
}

/// The state-dist node kind, matching `LanaStateDistKind`.
#[derive(Debug, Clone)]
pub enum StateDistKind {
    Dirac(StateValue),
    Append {
        left: DistOperand,
        right: DistOperand,
        has_cached_parameters: bool,
        p: f64,
        m_re: f64,
        m_im: f64,
        sigma: f64,
    },
    Transform {
        child: Arc<StateDist>,
        transform_id: u32,
    },
    Attenuate {
        child: Arc<StateDist>,
        factor: f64,
    },
}

/// One side of an append, matching `LanaDistOperand`: either an inline state
/// or a reference to a child distribution.
#[derive(Debug, Clone)]
pub enum DistOperand {
    Inline(StateValue),
    Node(Arc<StateDist>),
}

/// A mutable array of values.
#[derive(Debug, Clone, Default)]
pub struct Array {
    pub items: Vec<Value>,
}

/// A key/value map (increment 2).
#[derive(Debug, Clone, Default)]
pub struct Map {
    pub entries: Vec<MapEntry>,
}

#[derive(Debug, Clone)]
pub struct MapEntry {
    pub key: Arc<str>,
    pub value: Value,
}

impl Map {
    /// Create an empty map, mirroring `lana_map_new`.
    pub fn new(capacity: usize) -> Self {
        Self { entries: Vec::with_capacity(capacity) }
    }

    /// Look up a key, mirroring `lana_map_get`. Returns `None` when absent.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|entry| &*entry.key == key).map(|entry| &entry.value)
    }

    /// Whether a key is present, mirroring `lana_map_has >= 0`.
    pub fn has(&self, key: &str) -> bool {
        self.entries.iter().any(|entry| &*entry.key == key)
    }

    /// Insert or replace a key, mirroring `lana_map_set`. With
    /// `reject_existing` the insert fails with `Key` when the key is present.
    pub fn set(&mut self, key: Arc<str>, value: Value, reject_existing: bool) -> Result<(), LanaError> {
        if let Some(entry) = self.entries.iter_mut().find(|entry| &*entry.key == &*key) {
            if reject_existing {
                return Err(LanaError::Key);
            }
            entry.value = value;
            return Ok(());
        }
        self.entries.push(MapEntry { key, value });
        Ok(())
    }
}

/// An equipossible support set, mirroring `struct LanaPossibility`. `weights`
/// is `None` for a non-probabilistic, equipossible support.
#[derive(Debug, Clone)]
pub struct Possibility {
    pub values: Vec<Value>,
    pub weights: Option<Vec<f64>>,
    pub dependency_id: u64,
}

/// A set of guarded alternatives, mirroring `struct LanaPathSet`.
#[derive(Debug, Clone)]
pub struct PathSet {
    pub alternatives: Vec<PathAlternative>,
    pub dependency_id: u64,
}

#[derive(Debug, Clone)]
pub struct PathAlternative {
    pub guard: bool,
    pub weight: f64,
    pub result: Value,
}

/// An algebraic data type value, mirroring `struct LanaAdt` in
/// `vm/include/value.h`. The reserved variant `0xFFFFFFFF` is the built-in
/// `unknown` value available to every ADT.
#[derive(Debug, Clone)]
pub struct Adt {
    pub variant: u32,
    pub fields: Vec<Value>,
}

/// The reserved `unknown` variant tag, matching the C11 `0xFFFFFFFF`.
pub const ADT_UNKNOWN_VARIANT: u32 = 0xFFFF_FFFF;

/// A forked task handle, mirroring `struct LanaTask` in `vm/include/vm.h`.
/// The handle is shared between the parent VM and the worker that runs the
/// child VM: the worker writes the completion state, the parent reads it on
/// `JOIN`. The child VM itself lives in the scheduler's queue, not here.
#[derive(Debug, Clone)]
pub struct Task {
    pub id: u64,
    pub group_id: u64,
    /// The completion state, written by the worker and read by the parent.
    pub state: Arc<Mutex<TaskState>>,
    /// Signalled when the worker finishes the child VM.
    pub completed_cond: Arc<Condvar>,
    /// Set by `CANCEL`; the child VM polls it at each instruction.
    pub cancelled: Arc<AtomicBool>,
}

/// The shared completion state of a task, mirroring the `status`/`error`/
/// `result`/`completed`/`joined` fields of `LanaTask`.
#[derive(Debug, Clone)]
pub struct TaskState {
    pub status: LanaError,
    pub error: VmError,
    pub result: Value,
    pub completed: bool,
    pub joined: bool,
}

impl Task {
    /// Create a task handle with a fresh completion state.
    pub fn new(id: u64, group_id: u64) -> Self {
        Self {
            id,
            group_id,
            state: Arc::new(Mutex::new(TaskState {
                status: LanaError::Ok,
                error: VmError::default(),
                result: Value::null(),
                completed: false,
                joined: false,
            })),
            completed_cond: Arc::new(Condvar::new()),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Whether the task has been cancelled, mirroring `cancel_task`.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

/// The reactive-node kind, matching `LanaReactiveKind` in
/// `vm/include/value.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ReactiveKind {
    Root = 0,
    Binary,
    Compare,
    Unary,
}

/// The relationship kind, matching `LanaRelationshipKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RelationshipKind {
    Exact = 0,
    SameDependency,
    ExplicitJoint,
}

/// One historical version of a reactive node, matching `LanaReactiveVersion`.
#[derive(Debug, Clone)]
pub struct ReactiveVersion {
    pub revision: u64,
    pub value: Option<Value>,
}

/// A reactive dependency node, matching `struct LanaReactive` in
/// `vm/include/value.h`. Nodes form a DAG (`inputs` reference pre-existing
/// nodes), so `Arc` without cycle collection is sound. The node is wrapped in a
/// `Mutex` because `reactive_recompute_transaction` mutates `current`,
/// `history`, and `revision` in place, and a value graph carrying a reactive
/// node can cross a task boundary.
#[derive(Debug)]
pub struct Reactive {
    pub id: u64,
    pub dependency_id: u64,
    pub revision: u64,
    pub kind: ReactiveKind,
    pub relationship: RelationshipKind,
    pub exactness: DerivationExactness,
    pub operation: u32,
    pub inputs: [Option<Arc<Mutex<Reactive>>>; 2],
    pub constants: [Option<Value>; 2],
    pub current: Option<Value>,
    pub history: Vec<ReactiveVersion>,
}

/// A claim, matching `struct LanaClaim` in `vm/include/value.h`.
#[derive(Debug, Clone)]
pub struct Claim {
    pub value: Value,
    pub proposition: Arc<str>,
    pub exactness: DerivationExactness,
    pub tolerance: f64,
    pub source_valid: bool,
}

/// An effect receipt, matching `struct LanaEffectReceipt`.
#[derive(Debug, Clone)]
pub struct EffectReceipt {
    pub revision: u64,
    pub result: Value,
}

/// A planned effect, matching `struct LanaPlannedEffect`.
///
/// The C implementation mutates `receipts` and `execution_count` in place when
/// an effect executes; the Rust VM wraps them in a `Mutex` so a plan shared
/// across cloned values (via `Arc`) still observes the mutation.
#[derive(Debug)]
pub struct PlannedEffect {
    pub id: u64,
    pub kind: Arc<str>,
    pub payload: Value,
    pub state: Mutex<PlannedEffectState>,
}

/// The mutable execution state of a planned effect, guarded by the effect's
/// mutex. Mirrors the `receipts`/`execution_count` fields of
/// `LanaPlannedEffect` that `lana_vm_execute_planned_effect` mutates in place.
#[derive(Debug, Default)]
pub struct PlannedEffectState {
    pub receipts: Vec<EffectReceipt>,
    pub execution_count: usize,
}

/// A shared-information capability token, matching `struct LanaCapabilityToken`
/// in `runtime/c/shared.c`.
#[derive(Debug)]
pub struct CapabilityToken {
    pub shared: Arc<SharedInformation>,
    pub id: u64,
    pub permissions: u32,
    pub revoked: bool,
}

/// Capability permission bits, matching `LanaCapability` in
/// `runtime/include/shared.h`.
pub const LANA_CAPABILITY_READ: u32 = 1 << 0;
pub const LANA_CAPABILITY_OBSERVE: u32 = 1 << 1;
pub const LANA_CAPABILITY_ADMIN: u32 = 1 << 2;

/// One observation pending against a shared-information cell, matching
/// `LanaSharedObservation` in `runtime/c/shared.c`.
#[derive(Debug, Clone)]
pub struct SharedObservation {
    pub effective_time: f64,
    pub sequence: u64,
    pub evidence: Value,
}

/// One committed version of a shared-information cell, matching
/// `LanaSharedVersion`.
#[derive(Debug, Clone)]
pub struct SharedVersion {
    pub effective_time: f64,
    pub observation_sequence: u64,
    pub snapshot: Value,
}

/// A committed set of versions, matching `LanaSharedCommit`.
#[derive(Debug, Clone)]
pub struct SharedCommit {
    pub revision: u64,
    pub versions: Vec<SharedVersion>,
}

/// The mutable state of a shared-information cell, guarded by the cell's
/// mutex. Mirrors the fields of `LanaSharedInformation` that are read and
/// written under `shared->mutex` in `runtime/c/shared.c`.
#[derive(Debug, Default)]
pub struct SharedState {
    pub capability_epoch: u64,
    pub next_capability_id: u64,
    pub next_observation_sequence: u64,
    pub capabilities: Vec<Arc<CapabilityToken>>,
    pub observations: Vec<SharedObservation>,
    pub current: Option<SharedCommit>,
}

/// A shared-information cell, matching `struct LanaSharedInformation` in
/// `runtime/c/shared.c`. The C implementation keeps a per-version storage VM so the
/// mark-sweep GC can trace cloned snapshots; the Rust VM's `Arc`-based values
/// are self-contained, so the snapshots are stored directly and the transient
/// clone/recompute happens against the caller's VM.
#[derive(Debug)]
pub struct SharedInformation {
    pub identity: u64,
    pub base_snapshot: Value,
    pub state: Mutex<SharedState>,
    pub condition: Condvar,
}

/// The runtime value: a kind plus provenance metadata, mirroring the C11
/// `Value` struct (type tag + derivation/reactive/claim/planned_effect).
#[derive(Debug, Clone)]
pub struct Value {
    pub kind: ValueKind,
    pub derivation: Option<Arc<Derivation>>,
    pub reactive: Option<Arc<Mutex<Reactive>>>,
    pub claim: Option<Arc<Claim>>,
    pub planned_effect: Option<Arc<PlannedEffect>>,
}

/// The value payload, mirroring the C11 `Value.as` union.
#[derive(Debug, Clone)]
pub enum ValueKind {
    Null,
    Number(f64),
    Bool(bool),
    String(Arc<str>),
    State(StateValue),
    Distribution { p0: f64, p1: f64 },
    Sample(i32),
    Joint(Arc<JointState>),
    Array(Arc<Mutex<Array>>),
    Function(u32),
    Task(Arc<Task>),
    StateDist(Arc<StateDist>),
    Map(Arc<Mutex<Map>>),
    Possibility(Arc<Possibility>),
    PathSet(Arc<PathSet>),
    Capability(Arc<CapabilityToken>),
    Adt(Arc<Adt>),
    Lazy { function: u32, bound: usize },
}

impl Value {
    pub fn null() -> Self {
        Self { kind: ValueKind::Null, derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn number(number: f64) -> Self {
        Self { kind: ValueKind::Number(number), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn boolean(boolean: bool) -> Self {
        Self { kind: ValueKind::Bool(boolean), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn string(string: Arc<str>) -> Self {
        Self { kind: ValueKind::String(string), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn state(state: StateValue) -> Self {
        Self { kind: ValueKind::State(state), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn distribution(p0: f64, p1: f64) -> Self {
        Self { kind: ValueKind::Distribution { p0, p1 }, derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn sample(sample: i32) -> Self {
        Self { kind: ValueKind::Sample(sample), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn array(array: Arc<Mutex<Array>>) -> Self {
        Self { kind: ValueKind::Array(array), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn function(function: u32) -> Self {
        Self { kind: ValueKind::Function(function), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn task(task: Arc<Task>) -> Self {
        Self { kind: ValueKind::Task(task), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn state_dist(distribution: Arc<StateDist>) -> Self {
        Self { kind: ValueKind::StateDist(distribution), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn map(map: Arc<Mutex<Map>>) -> Self {
        Self { kind: ValueKind::Map(map), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn possibility(possibility: Arc<Possibility>) -> Self {
        Self { kind: ValueKind::Possibility(possibility), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn paths(paths: Arc<PathSet>) -> Self {
        Self { kind: ValueKind::PathSet(paths), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn joint(joint: Arc<JointState>) -> Self {
        Self { kind: ValueKind::Joint(joint), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn capability(capability: Arc<CapabilityToken>) -> Self {
        Self { kind: ValueKind::Capability(capability), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn adt(adt: Arc<Adt>) -> Self {
        Self { kind: ValueKind::Adt(adt), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn lazy(function: u32, bound: usize) -> Self {
        Self { kind: ValueKind::Lazy { function, bound }, derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    /// The stable type tag, matching `ValueType` in `vm/include/value.h`.
    pub fn value_type(&self) -> ValueType {
        match self.kind {
            ValueKind::Null => ValueType::Null,
            ValueKind::Number(_) => ValueType::Number,
            ValueKind::Bool(_) => ValueType::Bool,
            ValueKind::String(_) => ValueType::String,
            ValueKind::State(_) => ValueType::State,
            ValueKind::Distribution { .. } => ValueType::Distribution,
            ValueKind::Sample(_) => ValueType::Sample,
            ValueKind::Joint(_) => ValueType::JointState,
            ValueKind::Array(_) => ValueType::Array,
            ValueKind::Function(_) => ValueType::Function,
            ValueKind::Task(_) => ValueType::Task,
            ValueKind::StateDist(_) => ValueType::StateDist,
            ValueKind::Map(_) => ValueType::Map,
            ValueKind::Possibility(_) => ValueType::Possibility,
            ValueKind::PathSet(_) => ValueType::PathSet,
            ValueKind::Capability(_) => ValueType::SharedCapability,
            ValueKind::Adt(_) => ValueType::Adt,
            ValueKind::Lazy { .. } => ValueType::Lazy,
        }
    }

    /// The stable type name, matching `lana_value_type_name` in `vm/c/value.c`.
    pub fn type_name(&self) -> &'static str {
        match self.kind {
            ValueKind::Null => "null",
            ValueKind::Number(_) => "number",
            ValueKind::Bool(_) => "bool",
            ValueKind::String(_) => "string",
            ValueKind::State(_) => "state",
            ValueKind::Distribution { .. } => "distribution",
            ValueKind::Sample(_) => "sample",
            ValueKind::Joint(_) => "joint_state",
            ValueKind::Array(_) => "array",
            ValueKind::Function(_) => "function",
            ValueKind::Task(_) => "task",
            ValueKind::StateDist(_) => "state_dist",
            ValueKind::Map(_) => "map",
            ValueKind::Possibility(_) => "possibility",
            ValueKind::PathSet(_) => "paths",
            ValueKind::Capability(_) => "shared_capability",
            ValueKind::Adt(_) => "adt",
            ValueKind::Lazy { .. } => "lazy",
        }
    }

    /// The numeric payload, mirroring `value->as.number`. Callers check the
    /// type first.
    pub fn as_number(&self) -> f64 {
        match self.kind {
            ValueKind::Number(number) => number,
            _ => 0.0,
        }
    }

    /// The boolean payload, mirroring `value->as.boolean`.
    pub fn as_bool(&self) -> bool {
        match self.kind {
            ValueKind::Bool(boolean) => boolean,
            _ => false,
        }
    }

    /// The string payload, mirroring `value->as.string`.
    pub fn as_string(&self) -> Arc<str> {
        match &self.kind {
            ValueKind::String(string) => string.clone(),
            _ => Arc::from(""),
        }
    }

    /// The state payload, mirroring `value->as.state`. Callers check the type
    /// first.
    pub fn as_state(&self) -> &StateValue {
        match &self.kind {
            ValueKind::State(state) => state,
            _ => unreachable!("as_state on non-state value"),
        }
    }

    /// Whether the value is unresolved, mirroring `value_is_unresolved` in
    /// `vm/c/vm.c`. Possibilities and path sets are always unresolved; arrays
    /// and maps are unresolved if any element is.
    pub fn is_unresolved(&self) -> bool {
        match &self.kind {
            ValueKind::Possibility(_) | ValueKind::PathSet(_) => true,
            ValueKind::Array(array) => array.lock().unwrap().items.iter().any(|item| item.is_unresolved()),
            ValueKind::Map(map) => map.lock().unwrap().entries.iter().any(|entry| entry.value.is_unresolved()),
            _ => false,
        }
    }

    /// Render the value exactly as `lana_value_print` in `vm/c/value.c`.
    pub fn print(&self) -> String {
        let mut out = String::new();
        self.print_into(&mut out);
        out
    }

    fn print_into(&self, out: &mut String) {
        use std::fmt::Write;
        match &self.kind {
            ValueKind::Null => out.push_str("null"),
            ValueKind::Number(number) => out.push_str(&lana_bytecode::format_g(*number)),
            ValueKind::Bool(boolean) => out.push_str(if *boolean { "true" } else { "false" }),
            ValueKind::String(string) => out.push_str(string),
            ValueKind::State(state) => {
                let _ = write!(
                    out,
                    "state(p={}, d_re={}, d_im={})",
                    lana_bytecode::format_g(state.state.p),
                    lana_bytecode::format_g(state.state.d_re),
                    lana_bytecode::format_g(state.state.d_im));
            }
            ValueKind::Distribution { p0, p1 } => {
                let _ = write!(
                    out,
                    "distribution(p0={}, p1={})",
                    lana_bytecode::format_g(*p0),
                    lana_bytecode::format_g(*p1));
            }
            ValueKind::Sample(sample) => {
                let _ = write!(out, "{sample}");
            }
            ValueKind::Joint(joint) => {
                out.push_str("joint_state{");
                for (index, name) in joint.names.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{name}: ");
                    if let Some(value) = joint.values.get(index) {
                        value.print_into(out);
                    } else {
                        out.push_str("<finite-law>");
                    }
                }
                out.push('}');
            }
            ValueKind::Array(array) => {
                out.push('[');
                let array = array.lock().unwrap();
                for (index, item) in array.items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    item.print_into(out);
                }
                out.push(']');
            }
            ValueKind::Function(function) => {
                let _ = write!(out, "function({function})");
            }
            ValueKind::Task(task) => {
                let _ = write!(out, "task({})", task.id);
            }
            ValueKind::StateDist(_) => out.push_str("state_dist"),
            ValueKind::Map(map) => {
                out.push('{');
                let map = map.lock().unwrap();
                for (index, entry) in map.entries.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "\"{}\": ", entry.key);
                    entry.value.print_into(out);
                }
                out.push('}');
            }
            ValueKind::Possibility(possibility) => {
                out.push_str("possibility{");
                for (index, value) in possibility.values.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    value.print_into(out);
                }
                out.push('}');
            }
            ValueKind::PathSet(paths) => {
                out.push_str("paths{");
                for (index, alternative) in paths.alternatives.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{} => ", if alternative.guard { "true" } else { "false" });
                    alternative.result.print_into(out);
                }
                out.push('}');
            }
            ValueKind::Capability(_) => out.push_str("shared_capability"),
            ValueKind::Adt(adt) => {
                let _ = write!(out, "adt(variant={}){{", adt.variant);
                for (index, field) in adt.fields.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    field.print_into(out);
                }
                out.push('}');
            }
            ValueKind::Lazy { function, bound } => {
                let _ = write!(out, "lazy(function={function}, bound={bound})");
            }
        }
    }
}

impl From<&lana_bytecode::Value> for Value {
    /// Convert a constant-pool value into a runtime value.
    fn from(constant: &lana_bytecode::Value) -> Self {
        match constant {
            lana_bytecode::Value::Null => Value::null(),
            lana_bytecode::Value::Number(number) => Value::number(*number),
            lana_bytecode::Value::Bool(boolean) => Value::boolean(*boolean),
            lana_bytecode::Value::String(string) => Value::string(Arc::from(string.as_str())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_names_match_c11() {
        assert_eq!(Value::null().type_name(), "null");
        assert_eq!(Value::number(1.0).type_name(), "number");
        assert_eq!(Value::boolean(true).type_name(), "bool");
        assert_eq!(Value::string(Arc::from("x")).type_name(), "string");
        assert_eq!(Value::state(StateValue::default()).type_name(), "state");
        assert_eq!(Value::distribution(0.0, 1.0).type_name(), "distribution");
        assert_eq!(Value::sample(1).type_name(), "sample");
        assert_eq!(Value::function(0).type_name(), "function");
        assert_eq!(
            Value::state_dist(Arc::new(StateDist { kind: StateDistKind::Dirac(StateValue::default()) }))
                .type_name(),
            "state_dist"
        );
        let shared = Arc::new(SharedInformation {
            identity: 1,
            base_snapshot: Value::null(),
            state: Mutex::new(SharedState::default()),
            condition: Condvar::new(),
        });
        let token = Arc::new(CapabilityToken {
            shared,
            id: 1,
            permissions: LANA_CAPABILITY_ADMIN,
            revoked: false,
        });
        assert_eq!(Value::capability(token).type_name(), "shared_capability");
    }

    #[test]
    fn print_matches_c11_scalars() {
        assert_eq!(Value::null().print(), "null");
        assert_eq!(Value::number(0.4).print(), "0.4");
        assert_eq!(Value::number(1.0 / 3.0).print(), "0.333333333333");
        assert_eq!(Value::boolean(true).print(), "true");
        assert_eq!(Value::boolean(false).print(), "false");
        assert_eq!(Value::string(Arc::from("hello")).print(), "hello");
        assert_eq!(Value::sample(7).print(), "7");
        assert_eq!(Value::function(3).print(), "function(3)");
    }

    #[test]
    fn print_matches_c11_state() {
        let state = crate::state::StateValue {
            state: crate::state::State { p: 0.5, d_re: 0.0, d_im: 0.0 },
            indexes: crate::state::Indexes::default(),
        };
        assert_eq!(Value::state(state).print(), "state(p=0.5, d_re=0, d_im=0)");
    }

    #[test]
    fn print_matches_c11_array() {
        let array = Arc::new(Mutex::new(Array {
            items: vec![Value::number(1.0), Value::boolean(true)],
        }));
        assert_eq!(Value::array(array).print(), "[1, true]");
    }
}
