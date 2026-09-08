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
use crate::tensor::{tensor_get_imag, tensor_get_real};

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

/// A suspended generator frame (LIP-022 §2), mirroring `struct LanaGenerator`
/// in `vm/include/value.h`. `registers` is the snapshot of the generator's
/// frame registers; register 0 is reserved for the generator value itself and
/// is never part of the saved state.
#[derive(Debug, Clone)]
pub struct Generator {
    pub function: u32,
    pub ip: usize,
    pub registers: Vec<Value>,
    pub exhausted: bool,
}

/// A suspended async frame (LIP-024 §5), mirroring `struct LanaFuture` in
/// `vm/include/value.h`. `registers` is the snapshot of the async frame's
/// registers; register 0 is reserved for the future value itself and is never
/// part of the saved state. `ready` is false while the future is suspended on
/// an `OP_AWAIT`; the event loop only schedules futures with `ready == true`.
#[derive(Debug, Clone)]
pub struct Future {
    pub function: u32,
    pub ip: usize,
    pub registers: Vec<Value>,
    pub exhausted: bool,
    pub ready: bool,
    /// Rust-internal event-loop state (not part of the C11 `LanaFuture`
    /// contract): true while the future sits in the ready queue, so a future
    /// is never enqueued twice.
    pub queued: bool,
}

/// An immutable set of ordinary values (LIP-022 §1), mirroring `struct LanaSet`
/// in `vm/include/value.h`. Membership is linear over `items` (no hash
/// function), matching the C11 VM. `STATE`, `STATE_DIST`, and `Information` are
/// not set members.
#[derive(Debug, Clone, Default)]
pub struct Set {
    pub items: Vec<Value>,
}

/// A compiled regular expression (LIP-021 §2), mirroring `struct LanaRegex` in
/// `vm/include/value.h`: a Thompson NFA program plus its character classes.
/// No `Value` references inside, so it is shared immutably through `Arc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegexOp {
    Char,
    Any,
    Class,
    Bol,
    Eol,
    Split,
    Jmp,
    Match,
}

#[derive(Debug, Clone, Copy)]
pub struct RegexInst {
    pub op: RegexOp,
    pub c: u32,
    pub x: u32,
    pub y: u32,
}

#[derive(Debug, Clone)]
pub struct RegexClass {
    pub bitmap: [u32; 8],
    pub negated: bool,
}

#[derive(Debug, Clone)]
pub struct Regex {
    pub insts: Vec<RegexInst>,
    pub classes: Vec<RegexClass>,
}

/// LIP-006 optimizer descriptor, mirroring `struct LanaOptimizer` in
/// `vm/include/value.h`. `name` is "sgd" or "adam". For SGD, `learning_rate`
/// and `momentum` are used and the Adam fields are zero; for Adam,
/// `learning_rate`, `beta1`, `beta2`, and `epsilon` are used and `momentum` is
/// zero.
#[derive(Debug, Clone)]
pub struct Optimizer {
    pub name: Arc<str>,
    pub learning_rate: f64,
    pub momentum: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

/// LIP-006 training result, mirroring `struct LanaTrainingResult` in
/// `vm/include/value.h`: the trained parameters plus the per-step history.
/// LIP-010 adds the model/loss function indices and the optimizer descriptor so
/// an incremental `update` (or a reactive recomputation) can resume the
/// optimizer from the last step map. LIP-014 adds the resolved dataset and the
/// effective batch size so `resume` can continue the run from any step.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub params: Arc<Tensor>,
    pub steps: Arc<Mutex<Array>>,
    pub model_function: u32,
    pub loss_function: u32,
    pub optimizer: Arc<Optimizer>,
    pub data: Value,
    pub batch_size: usize,
}

/// LIP-009 inference algorithm descriptor, mirroring
/// `struct LanaInferenceAlgorithm` in `vm/include/value.h`. `name` is "mcmc",
/// "vi", or "smc". For MCMC, `samples` and `burn_in` are used; for VI, `family`
/// ("gaussian"/"mean_field") and `iterations` are used; for SMC, `samples` is
/// the particle count. Unused fields are zero (or `None` for `family`).
#[derive(Debug, Clone)]
pub struct InferenceAlgorithm {
    pub name: Arc<str>,
    pub family: Option<Arc<str>>,
    pub samples: f64,
    pub burn_in: f64,
    pub iterations: f64,
}

/// LIP-009 posterior, mirroring `struct LanaPosterior` in `vm/include/value.h`:
/// a distribution over parameters produced by `infer`. `mean` is the point
/// estimate, `variance` the per-element uncertainty, `samples` the sample
/// matrix (None for VI), `steps` the per-step provenance, and `seed` the RNG
/// seed the run used.
#[derive(Debug, Clone)]
pub struct Posterior {
    pub mean: Arc<Tensor>,
    pub variance: Arc<Tensor>,
    pub samples: Option<Arc<Tensor>>,
    pub steps: Arc<Mutex<Array>>,
    pub seed: u64,
}

/// LIP-015 lazy relational-algebra plan node, mirroring `struct LanaDataset` in
/// `vm/include/value.h`. `op` selects the operator; `source` is the upstream
/// plan (a `Lazy` value for `Source`, otherwise another `Dataset`). `function`
/// is the predicate/transform function index for `Filter`/`Map`. `columns` is
/// the column-name array for `Select`; `key` is the column name for
/// `Sort`/`GroupBy`/`Join`; `limit` is the row cap for `Limit`; `other` is the
/// right-hand dataset for `Join`; `aggregate` is the aggregate descriptor
/// (e.g. `["sum","v"]` or `["count"]`) for `Aggregate`.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub op: DatasetOp,
    pub source: Value,
    pub function: u32,
    pub columns: Value,
    pub key: Value,
    pub limit: Value,
    pub other: Value,
    pub aggregate: Value,
}

/// The dataset operator, matching `LanaDatasetOp` in `vm/include/value.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetOp {
    Source,
    Filter,
    Map,
    Select,
    Limit,
    Sort,
    GroupBy,
    Aggregate,
    Join,
}

/// A first-class tensor, mirroring `struct LanaTensor` in `vm/include/tensor.h`.
///
/// `data` is a row-major buffer of `f64` values; for a complex tensor the real
/// and imaginary parts are interleaved `[re, im, re, im, …]`, so the buffer
/// length is `prod(shape) * (is_complex ? 2 : 1)`. `offset` is the first
/// element in elements (0 for a base tensor). A slicing view shares the source
/// buffer through `Arc` and carries its own shape, strides, and offset, so
/// strides may be non-contiguous; every element access goes through `offset`
/// and `strides`. Tensors are immutable once constructed, so `Arc` sharing
/// (matching the C11 VM's GC-rooted base chain) is sound.
/// LIP-027: tensor numeric dtype. `is_complex` is derived from this: a
/// `Complex` tensor has `is_complex == true`, every other dtype has it false.
/// The default is `F64`, so existing programs are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    F64,
    F32,
    F16,
    Bf16,
    Complex,
}

impl TensorDtype {
    /// The dtype's string name, matching the C11 VM's `dtype_to_string`.
    pub fn as_str(self) -> &'static str {
        match self {
            TensorDtype::F64 => "f64",
            TensorDtype::F32 => "f32",
            TensorDtype::F16 => "f16",
            TensorDtype::Bf16 => "bf16",
            TensorDtype::Complex => "complex",
        }
    }

    /// Parse a dtype string; `None` for an unknown string (the caller maps
    /// that to `LanaError::InvalidParameters`).
    pub fn from_str(s: &str) -> Option<TensorDtype> {
        match s {
            "f64" => Some(TensorDtype::F64),
            "f32" => Some(TensorDtype::F32),
            "f16" => Some(TensorDtype::F16),
            "bf16" => Some(TensorDtype::Bf16),
            "complex" => Some(TensorDtype::Complex),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDevice {
    Cpu,
    Metal,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub ndim: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub is_complex: bool,
    /// LIP-027: numeric dtype (F64 default).
    pub dtype: TensorDtype,
    pub device: TensorDevice,
    pub metal_buffer: Option<Arc<crate::metal::ResidentBuffer>>,
    /// LIP-027: compact byte buffer, `prod(shape) * element_width` bytes.
    /// Element access goes through the `tensor_get_*`/`tensor_set_*` helpers
    /// in `tensor.rs`, which convert between the storage dtype and f64.
    pub data: Arc<Vec<u8>>,
    pub offset: usize,
    /// LIP-007: a STATE tensor (each element is a density matrix).
    pub is_state: bool,
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
    Train,
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
    /// LIP-010: a training data root accepts a single [x, target] observation
    /// on `observe` (its support is structural, not a membership test).
    pub is_training_data: bool,
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
    pub revoked: AtomicBool,
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
    Tensor(Arc<Tensor>),
    NQubitState(Arc<Tensor>),
    Povm(Arc<Tensor>),
    Channel(Arc<Tensor>),
    Observable(Arc<Tensor>),
    Lazy { function: u32, bound: usize },
    Generator(Arc<Mutex<Generator>>),
    Future(Arc<Mutex<Future>>),
    Set(Arc<Mutex<Set>>),
    Regex(Arc<Regex>),
    Optimizer(Arc<Optimizer>),
    TrainingResult(Arc<TrainingResult>),
    InferenceAlgorithm(Arc<InferenceAlgorithm>),
    Posterior(Arc<Posterior>),
    Dataset(Arc<Dataset>),
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

    pub fn tensor(tensor: Arc<Tensor>) -> Self {
        Self { kind: ValueKind::Tensor(tensor), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn nqubit_state(tensor: Arc<Tensor>) -> Self {
        Self { kind: ValueKind::NQubitState(tensor), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn povm(tensor: Arc<Tensor>) -> Self {
        Self { kind: ValueKind::Povm(tensor), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn channel(tensor: Arc<Tensor>) -> Self {
        Self { kind: ValueKind::Channel(tensor), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn observable(tensor: Arc<Tensor>) -> Self {
        Self { kind: ValueKind::Observable(tensor), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn lazy(function: u32, bound: usize) -> Self {
        Self { kind: ValueKind::Lazy { function, bound }, derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn generator(generator: Arc<Mutex<Generator>>) -> Self {
        Self { kind: ValueKind::Generator(generator), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn future(future: Arc<Mutex<Future>>) -> Self {
        Self { kind: ValueKind::Future(future), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn set(set: Arc<Mutex<Set>>) -> Self {
        Self { kind: ValueKind::Set(set), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn regex(regex: Arc<Regex>) -> Self {
        Self { kind: ValueKind::Regex(regex), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn optimizer(optimizer: Arc<Optimizer>) -> Self {
        Self { kind: ValueKind::Optimizer(optimizer), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn training_result(result: Arc<TrainingResult>) -> Self {
        Self { kind: ValueKind::TrainingResult(result), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn inference_algorithm(algorithm: Arc<InferenceAlgorithm>) -> Self {
        Self { kind: ValueKind::InferenceAlgorithm(algorithm), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn posterior(posterior: Arc<Posterior>) -> Self {
        Self { kind: ValueKind::Posterior(posterior), derivation: None, reactive: None, claim: None, planned_effect: None }
    }

    pub fn dataset(dataset: Arc<Dataset>) -> Self {
        Self { kind: ValueKind::Dataset(dataset), derivation: None, reactive: None, claim: None, planned_effect: None }
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
            ValueKind::Tensor(_) => ValueType::Tensor,
            ValueKind::NQubitState(_) => ValueType::NQubitState,
            ValueKind::Povm(_) => ValueType::Povm,
            ValueKind::Channel(_) => ValueType::Channel,
            ValueKind::Observable(_) => ValueType::Observable,
            ValueKind::Lazy { .. } => ValueType::Lazy,
            ValueKind::Generator(_) => ValueType::Generator,
            ValueKind::Future(_) => ValueType::Future,
            ValueKind::Set(_) => ValueType::Set,
            ValueKind::Regex(_) => ValueType::Regex,
            ValueKind::Optimizer(_) => ValueType::Optimizer,
            ValueKind::TrainingResult(_) => ValueType::TrainingResult,
            ValueKind::InferenceAlgorithm(_) => ValueType::InferenceAlgorithm,
            ValueKind::Posterior(_) => ValueType::Posterior,
            ValueKind::Dataset(_) => ValueType::Dataset,
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
            ValueKind::Tensor(_) => "tensor",
            ValueKind::NQubitState(_) => "nqubit_state",
            ValueKind::Povm(_) => "povm",
            ValueKind::Channel(_) => "channel",
            ValueKind::Observable(_) => "observable",
            ValueKind::Lazy { .. } => "lazy",
            ValueKind::Generator(_) => "generator",
            ValueKind::Future(_) => "future",
            ValueKind::Set(_) => "set",
            ValueKind::Regex(_) => "regex",
            ValueKind::Optimizer(_) => "optimizer",
            ValueKind::TrainingResult(_) => "training_result",
            ValueKind::InferenceAlgorithm(_) => "inference_algorithm",
            ValueKind::Posterior(_) => "posterior",
            ValueKind::Dataset(_) => "dataset",
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
        self.is_unresolved_at(&mut std::collections::HashSet::new())
    }

    fn is_unresolved_at(&self, seen: &mut std::collections::HashSet<usize>) -> bool {
        let current = self.reactive.as_ref().and_then(|r| r.lock().unwrap().current.clone());
        let value = current.as_ref().unwrap_or(self);
        match &value.kind {
            ValueKind::Possibility(_) | ValueKind::PathSet(_) => true,
            ValueKind::Array(array) => {
                seen.insert(Arc::as_ptr(array) as usize) &&
                    array.lock().unwrap().items.iter().any(|item| item.is_unresolved_at(seen))
            }
            ValueKind::Map(map) => {
                seen.insert(Arc::as_ptr(map) as usize) &&
                    map.lock().unwrap().entries.iter().any(|entry| entry.value.is_unresolved_at(seen))
            }
            ValueKind::Set(set) => {
                seen.insert(Arc::as_ptr(set) as usize) &&
                    set.lock().unwrap().items.iter().any(|item| item.is_unresolved_at(seen))
            }
            _ => false,
        }
    }

    /// Whether the value contains a revoked capability token, mirroring
    /// `value_has_revoked_capability` in `vm/c/vm.c`. A capability is revoked
    /// when its `revoked` flag is set; arrays and maps are checked recursively.
    pub fn has_revoked_capability(&self) -> bool {
        self.has_revoked_capability_at(&mut std::collections::HashSet::new())
    }

    fn has_revoked_capability_at(&self, seen: &mut std::collections::HashSet<usize>) -> bool {
        let current = self.reactive.as_ref().and_then(|r| r.lock().unwrap().current.clone());
        let value = current.as_ref().unwrap_or(self);
        match &value.kind {
            ValueKind::Capability(capability) => capability.revoked.load(Ordering::Acquire),
            ValueKind::Array(array) => {
                seen.insert(Arc::as_ptr(array) as usize) &&
                    array.lock().unwrap().items.iter().any(|item| item.has_revoked_capability_at(seen))
            }
            ValueKind::Map(map) => {
                seen.insert(Arc::as_ptr(map) as usize) &&
                    map.lock().unwrap().entries.iter().any(|entry| entry.value.has_revoked_capability_at(seen))
            }
            ValueKind::Set(set) => {
                seen.insert(Arc::as_ptr(set) as usize) &&
                    set.lock().unwrap().items.iter().any(|item| item.has_revoked_capability_at(seen))
            }
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
            ValueKind::Tensor(tensor) => {
                tensor_print_rec(tensor, 0, tensor.offset, out);
            }
            ValueKind::NQubitState(tensor) => {
                tensor_print_rec(tensor, 0, tensor.offset, out);
            }
            ValueKind::Povm(tensor) => {
                tensor_print_rec(tensor, 0, tensor.offset, out);
            }
            ValueKind::Channel(tensor) => {
                tensor_print_rec(tensor, 0, tensor.offset, out);
            }
            ValueKind::Observable(tensor) => {
                tensor_print_rec(tensor, 0, tensor.offset, out);
            }
            ValueKind::Lazy { function, bound } => {
                let _ = write!(out, "lazy(function={function}, bound={bound})");
            }
            ValueKind::Generator(generator) => {
                let generator = generator.lock().unwrap();
                let _ = write!(
                    out,
                    "generator(function={}, exhausted={})",
                    generator.function,
                    generator.exhausted
                );
            }
            ValueKind::Future(future) => {
                let future = future.lock().unwrap();
                let _ = write!(
                    out,
                    "future(function={}, exhausted={}, ready={})",
                    future.function,
                    future.exhausted,
                    future.ready
                );
            }
            ValueKind::Set(set) => {
                out.push_str("set{");
                let set = set.lock().unwrap();
                for (index, item) in set.items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    item.print_into(out);
                }
                out.push('}');
            }
            ValueKind::Regex(regex) => {
                let _ = write!(out, "regex(insts={})", regex.insts.len());
            }
            ValueKind::Optimizer(optimizer) => {
                let _ = write!(
                    out,
                    "optimizer(name={}, learning_rate={}, momentum={}, beta1={}, beta2={}, epsilon={})",
                    optimizer.name,
                    lana_bytecode::format_g(optimizer.learning_rate),
                    lana_bytecode::format_g(optimizer.momentum),
                    lana_bytecode::format_g(optimizer.beta1),
                    lana_bytecode::format_g(optimizer.beta2),
                    lana_bytecode::format_g(optimizer.epsilon)
                );
            }
            ValueKind::TrainingResult(result) => {
                let _ = write!(out, "training_result(steps={})", result.steps.lock().unwrap().items.len());
            }
            ValueKind::InferenceAlgorithm(algorithm) => {
                let _ = write!(
                    out,
                    "inference_algorithm(name={}, family={}, samples={}, burn_in={}, iterations={})",
                    algorithm.name,
                    algorithm.family.as_deref().unwrap_or(""),
                    lana_bytecode::format_g(algorithm.samples),
                    lana_bytecode::format_g(algorithm.burn_in),
                    lana_bytecode::format_g(algorithm.iterations)
                );
            }
            ValueKind::Posterior(posterior) => {
                let _ = write!(out, "posterior(steps={})", posterior.steps.lock().unwrap().items.len());
            }
            ValueKind::Dataset(dataset) => {
                let _ = write!(out, "dataset(op={})", dataset.op as i32);
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

/// Recursively render a tensor, mirroring `tensor_print_rec` in `vm/c/value.c`.
/// Nested `[...]` with `, ` separators; real elements use `%.12g`, complex
/// elements render as `[re, im]` with `%.12g` each. A 0-d tensor prints its
/// single scalar (or `[re, im]` pair) with no brackets.
fn tensor_print_rec(tensor: &Tensor, dim: usize, offset: usize, out: &mut String) {
    use std::fmt::Write;
    if dim == tensor.ndim {
        if tensor.is_complex {
            let _ = write!(
                out,
                "[{}, {}]",
                lana_bytecode::format_g(tensor_get_real(&tensor, offset)),
                lana_bytecode::format_g(tensor_get_imag(&tensor, offset))
            );
        } else {
            out.push_str(&lana_bytecode::format_g(tensor_get_real(&tensor, offset)));
        }
        return;
    }
    out.push('[');
    for i in 0..tensor.shape[dim] {
        if i > 0 {
            out.push_str(", ");
        }
        tensor_print_rec(tensor, dim + 1, offset + i * tensor.strides[dim], out);
    }
    out.push(']');
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
            revoked: AtomicBool::new(false),
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
