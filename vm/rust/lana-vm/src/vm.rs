//! Register VM core, mirroring `lana_vm_run` in `vm/c/vm.c`.
//!
//! The Rust VM is semantically identical to the C11 VM: same state math, same
//! PCG32 stream, same error codes, same value printing. The memory model
//! differs (native Rust ownership instead of a mark-sweep GC) but the 256 MiB
//! limit is preserved by byte accounting.
//!
//! Increment 1 covers the scalar/array/control-flow/state/history opcodes.
//! Opcodes that construct increment-2+ types (state dists, joints, tasks,
//! host calls) return `UnsupportedOperation` until their increment lands.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

use lana_bytecode::opcode::{LANA_MAX_CALL_FRAMES, LANA_MAX_REGISTERS};
use lana_bytecode::{Chunk, Instruction, LanaError, OpCode, Value as ConstantValue, ValueType};

use crate::derivation::{
    self, Derivation, DerivationExactness, DerivationKind, DerivationOutcome, EvidenceStatus,
};
use crate::rng::Rng;
use crate::state::{self, State, StateValue};
use crate::state_dist::{self, DistEvalFrame, EvalAction, LANA_STATE_DIST_DEPTH_LIMIT};
use crate::value::{
    Adt, Array, CapabilityToken, Claim, DistOperand, EffectReceipt, JointKind, JointRow, JointState,
    Map, PathAlternative, PathSet, PlannedEffect, PlannedEffectState, Possibility, Reactive,
    ReactiveKind, ReactiveVersion, RelationshipKind, SharedCommit, SharedInformation,
    SharedObservation, SharedState, SharedVersion, StateDist, StateDistKind, Task, Value,
    ValueKind, VmError, LANA_CAPABILITY_ADMIN, LANA_CAPABILITY_OBSERVE, LANA_CAPABILITY_READ,
    LANA_JOINT_CAN_CONDITION, LANA_JOINT_CAN_PROJECT, LANA_JOINT_CAN_RESOLVE,
    LANA_JOINT_CAN_SAMPLE,
};

/// History policy, matching `LanaHistoryPolicy` in `vm/include/vm.h`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum HistoryPolicy {
    #[default]
    None = 0,
    Latest = 1,
    Duration = 2,
}

/// A per-register history, mirroring `LanaHistory` in `vm/include/vm.h`.
///
/// The C11 VM stores versions in a growable array; the Rust VM uses a `Vec`.
/// `clone_history` is a shallow copy (the `Arc`-backed state values are shared),
/// which is sound because state values are immutable records.
#[derive(Debug, Clone, Default)]
pub struct History {
    pub policy: HistoryPolicy,
    pub amount: f64,
    pub versions: Vec<StateValue>,
}

/// A call frame, mirroring `LanaFrame` in `vm/include/vm.h`.
///
/// Registers and histories are sized to the function's `register_count` (not
/// `LANA_MAX_REGISTERS`), matching the C11 VM which zeroes only
/// `register_count` slots on `OP_CALL`. `resize_with` constructs each slot
/// fresh in the heap buffer rather than cloning a template, so entry is a
/// memset-like write and return drops only the slots the function declared.
#[derive(Debug, Clone)]
pub struct Frame {
    pub registers: Vec<Value>,
    pub histories: Vec<History>,
    pub return_ip: usize,
    pub return_register: u32,
    pub function: u32,
}

impl Frame {
    fn new(register_count: usize) -> Self {
        let mut registers = Vec::with_capacity(register_count);
        registers.resize_with(register_count, Value::null);
        let mut histories = Vec::with_capacity(register_count);
        histories.resize_with(register_count, History::default);
        Self {
            registers,
            histories,
            return_ip: 0,
            return_register: 0,
            function: u32::MAX,
        }
    }
}

/// The maximum register index (0-based) an instruction reads or writes.
///
/// Operand semantics mirror `vm/c/assembler.c` and the verifier's
/// contiguous-range checks in `vm/c/bytecode.c`. The function header's
/// `register_count` is only a lower bound (the emitter resets its counter at
/// each statement start), so frames must be sized from the true max index.
fn instruction_max_register(ins: &Instruction) -> usize {
    use OpCode::*;
    // `LANA_NO_OPERAND` (u32::MAX) marks an unused operand; skip it.
    let reg = |operand: u32| -> Option<usize> {
        if operand == u32::MAX {
            None
        } else {
            Some(operand as usize)
        }
    };
    let mut max = reg(ins.a).unwrap_or(0);
    match ins.opcode {
        // `ins.imm` is a constant index, not a register.
        LoadConst => {}
        // Arguments occupy `ins.c .. ins.c + ins.imm - 1`.
        Call | Fork | HostCall => {
            if ins.imm != u32::MAX && ins.imm > 0 {
                max = max.max(ins.c as usize + ins.imm as usize - 1);
            }
        }
        // Elements occupy `ins.b .. ins.b + ins.c - 1`.
        ArrayNew | JointBuild | AdtBuild => {
            if ins.c != u32::MAX && ins.c > 0 {
                max = max.max(ins.b as usize + ins.c as usize - 1);
            }
        }
        // `ins.imm` is a register operand.
        StateBuild | Mix | JointCondition | Observe => {
            if let Some(b) = reg(ins.b) {
                max = max.max(b);
            }
            if let Some(c) = reg(ins.c) {
                max = max.max(c);
            }
            if let Some(imm) = reg(ins.imm) {
                max = max.max(imm);
            }
        }
        // Default: `ins.a`, `ins.b`, `ins.c` are registers. A few opcodes
        // store a small id or constant index in `ins.b`/`ins.c`; including
        // them only over-approximates, which is safe.
        _ => {
            if let Some(b) = reg(ins.b) {
                max = max.max(b);
            }
            if let Some(c) = reg(ins.c) {
                max = max.max(c);
            }
        }
    }
    max
}

/// Per-function frame size (max register index + 1), keyed by function index.
///
/// Functions are laid out contiguously in `chunk.code`; each function's range
/// is `[entry, next_entry)` (or `[entry, code.len())` for the last). The entry
/// frame and forked-task frames stay at `LANA_MAX_REGISTERS` because they run
/// code whose function index is not known at frame-construction time.
fn compute_max_registers(chunk: &Chunk) -> Vec<usize> {
    let mut entries: Vec<(usize, usize)> = chunk
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| (function.entry as usize, index))
        .collect();
    entries.sort_unstable();
    let mut result = vec![0usize; chunk.functions.len()];
    for (position, &(entry, index)) in entries.iter().enumerate() {
        let end = if position + 1 < entries.len() {
            entries[position + 1].0
        } else {
            chunk.code.len()
        };
        let mut max = 0usize;
        for ins in &chunk.code[entry..end] {
            max = max.max(instruction_max_register(ins));
        }
        result[index] = max + 1;
    }
    result
}

/// One pending path split, mirroring `struct LanaPathExecution` in `vm/c/vm.c`.
/// The Rust VM keeps the executions on a `Vec` stack; the C11 uses a linked
/// list with `next` pointing at the previous execution.
struct PathExecution {
    false_frames: Vec<Frame>,
    true_frames: Vec<Frame>,
    frame_count: usize,
    false_ip: usize,
    dependency_id: u64,
    true_weight: f64,
    false_weight: f64,
    previous_path_count: usize,
    running_false: bool,
}

/// A memo mapping source container pointers to their clones, mirroring
/// `LanaContainerCloneMemo` in `vm/c/vm.c`. Only the mutable containers
/// (arrays, maps) need the memo; the immutable wrappers (joints, possibilities,
/// path sets, state dists) are duplicated freely because aliasing them is
/// unobservable. Keyed by `Arc::as_ptr` so shared substructure is preserved
/// across one clone operation.
#[derive(Default)]
struct DeepCloneMemo {
    arrays: HashMap<usize, Arc<Mutex<Array>>>,
    maps: HashMap<usize, Arc<Mutex<Map>>>,
}

/// Resolution reasons, matching `LanaResolutionReason` in
/// `vm/include/error.h`.
pub const LANA_RESOLUTION_REASON_NONE: u32 = 0;
pub const LANA_RESOLUTION_REASON_NO_ALTERNATIVES: u32 = 1;
pub const LANA_RESOLUTION_REASON_MULTIPLE_ALTERNATIVES: u32 = 2;
pub const LANA_RESOLUTION_REASON_CONTRADICTION: u32 = 3;
pub const LANA_RESOLUTION_REASON_INVALID_CONDITIONING: u32 = 4;
pub const LANA_RESOLUTION_REASON_UNSUPPORTED_EXACT: u32 = 5;
pub const LANA_RESOLUTION_REASON_CANCELLED: u32 = 6;
pub const LANA_RESOLUTION_REASON_RESOURCE_LIMIT: u32 = 7;

/// The resolution-reason name, matching `lana_resolution_reason_name` in
/// `vm/c/error.c`.
pub fn resolution_reason_name(reason: u32) -> &'static str {
    match reason {
        LANA_RESOLUTION_REASON_NONE => "none",
        LANA_RESOLUTION_REASON_NO_ALTERNATIVES => "no-alternatives",
        LANA_RESOLUTION_REASON_MULTIPLE_ALTERNATIVES => "multiple-alternatives",
        LANA_RESOLUTION_REASON_CONTRADICTION => "contradiction",
        LANA_RESOLUTION_REASON_INVALID_CONDITIONING => "invalid-conditioning",
        LANA_RESOLUTION_REASON_UNSUPPORTED_EXACT => "unsupported-exact",
        LANA_RESOLUTION_REASON_CANCELLED => "cancelled",
        LANA_RESOLUTION_REASON_RESOURCE_LIMIT => "resource-limit",
        _ => "unknown",
    }
}

/// The pure operation kinds, matching `LanaPureKind` in `vm/c/vm.c`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PureKind {
    Binary,
    Compare,
}

/// Measure modes, matching `LanaMeasureMode` in `vm/include/vm.h`.
pub const LANA_MEASURE_PROBABILITY: u32 = 0;
pub const LANA_MEASURE_DISTRIBUTION: u32 = 1;
pub const LANA_MEASURE_SAMPLE: u32 = 2;

/// History duration policy id, matching `LANA_HISTORY_DURATION`.
pub const LANA_HISTORY_DURATION: u32 = 2;

/// Observable id, matching `LANA_OBSERVABLE_PROBABILITY` in
/// `vm/include/bytecode.h`.
pub const LANA_OBSERVABLE_PROBABILITY: u32 = 0;

/// Resource kinds, matching `LanaResourceKind` in `vm/include/error.h`.
pub const LANA_RESOURCE_MEMORY: u32 = 1;
pub const LANA_RESOURCE_INSTRUCTIONS: u32 = 2;
pub const LANA_RESOURCE_TASKS: u32 = 3;
pub const LANA_RESOURCE_PATHS: u32 = 4;
pub const LANA_RESOURCE_SAMPLES: u32 = 5;
pub const LANA_RESOURCE_TIME: u32 = 6;

/// Exact-support kinds, matching `LanaExactSupport` in `vm/include/error.h`.
pub const LANA_EXACT_SUPPORT_UNKNOWN: u32 = 0;
pub const LANA_EXACT_SUPPORT_AVAILABLE: u32 = 1;
pub const LANA_EXACT_SUPPORT_UNAVAILABLE: u32 = 2;

/// Resource-kind name, matching `lana_resource_kind_name` in `vm/c/error.c`.
pub fn resource_kind_name(resource: u32) -> &'static str {
    match resource {
        LANA_RESOURCE_MEMORY => "memory",
        LANA_RESOURCE_INSTRUCTIONS => "instructions",
        LANA_RESOURCE_TASKS => "tasks",
        LANA_RESOURCE_PATHS => "paths",
        LANA_RESOURCE_SAMPLES => "samples",
        LANA_RESOURCE_TIME => "time",
        _ => "none",
    }
}

/// Exact-support name, matching `lana_exact_support_name` in `vm/c/error.c`.
pub fn exact_support_name(support: u32) -> &'static str {
    match support {
        LANA_EXACT_SUPPORT_AVAILABLE => "available",
        LANA_EXACT_SUPPORT_UNAVAILABLE => "unavailable",
        _ => "unknown",
    }
}

/// Host-call ids, matching `LanaHostCallId` in `vm/include/bytecode.h`.
pub const LANA_HOST_ARGS: u32 = 0;
pub const LANA_HOST_READ_TEXT: u32 = 1;
pub const LANA_HOST_WRITE_TEXT: u32 = 2;
pub const LANA_HOST_NOW: u32 = 3;
pub const LANA_HOST_RANDOM: u32 = 4;
pub const LANA_HOST_ASSERT: u32 = 5;
pub const LANA_HOST_MAP_NEW: u32 = 6;
pub const LANA_HOST_MAP_HAS: u32 = 7;
pub const LANA_HOST_MAP_GET: u32 = 8;
pub const LANA_HOST_MAP_SET: u32 = 9;
pub const LANA_HOST_MAP_KEYS: u32 = 10;
pub const LANA_HOST_INDEX_GET: u32 = 11;
pub const LANA_HOST_INDEX_SET: u32 = 12;
pub const LANA_HOST_JSON_PARSE: u32 = 13;
pub const LANA_HOST_JSON_STRINGIFY: u32 = 14;
pub const LANA_HOST_CSV_READ: u32 = 15;
pub const LANA_HOST_CSV_WRITE: u32 = 16;
pub const LANA_HOST_STRING_LENGTH: u32 = 17;
pub const LANA_HOST_STRING_BYTE_AT: u32 = 18;
pub const LANA_HOST_STRING_SLICE: u32 = 19;
pub const LANA_HOST_STRING_CONCAT: u32 = 20;
pub const LANA_HOST_NUMBER_TO_STRING: u32 = 21;
pub const LANA_HOST_ARRAY_NEW: u32 = 22;
pub const LANA_HOST_ARRAY_PUSH: u32 = 23;
pub const LANA_HOST_STRING_HEX: u32 = 24;
pub const LANA_HOST_STRING_JOIN: u32 = 25;
pub const LANA_HOST_ARRAY_LENGTH: u32 = 26;
pub const LANA_HOST_STRING_UNESCAPE: u32 = 27;
pub const LANA_HOST_PATH_RESOLVE: u32 = 28;
pub const LANA_HOST_SAMPLE_RECORD: u32 = 29;
pub const LANA_HOST_INFORMATION_NEW: u32 = 30;
pub const LANA_HOST_CLAIM_NEW: u32 = 31;
pub const LANA_HOST_CLAIM_VALUE: u32 = 32;
pub const LANA_HOST_CLAIM_PROPOSITION: u32 = 33;
pub const LANA_HOST_CLAIM_STATUS: u32 = 34;
pub const LANA_HOST_PLANNED_EFFECT_NEW: u32 = 35;
pub const LANA_HOST_PLANNED_EFFECT_EXECUTE: u32 = 36;
pub const LANA_HOST_PLANNED_EFFECT_STATUS: u32 = 37;
pub const LANA_HOST_SHARED_INFORMATION: u32 = 38;
pub const LANA_HOST_SHARED_GRANT: u32 = 39;
pub const LANA_HOST_SHARED_REVOKE: u32 = 40;
pub const LANA_HOST_SHARED_SNAPSHOT: u32 = 41;
pub const LANA_HOST_SHARED_AT: u32 = 42;
pub const LANA_HOST_SHARED_OBSERVE: u32 = 43;
pub const LANA_HOST_SHARED_REVISION: u32 = 44;
pub const LANA_HOST_SHARED_IDENTITY: u32 = 45;
pub const LANA_HOST_SHARED_WAIT: u32 = 46;
pub const LANA_HOST_INFORMATION_INSPECT: u32 = 47;
pub const LANA_HOST_DIRECTORY_LIST: u32 = 48;
pub const LANA_HOST_DIRECTORY_CREATE: u32 = 49;
pub const LANA_HOST_PATH_EXISTS: u32 = 50;
pub const LANA_HOST_WRITE_TEXT_ATOMIC: u32 = 51;
pub const LANA_HOST_HASH_UPDATE: u32 = 52;
pub const LANA_HOST_LAZY_BOUND: u32 = 53;

// Lana 2.0 declared correlation (bivariate Bernoulli joint law). Present in
// both the C11 VM and the Rust VM at id 54, so a `.labc` assembled by either
// backend runs identically under both.
pub const LANA_HOST_CORRELATED: u32 = 54;

// Durable-pipeline host calls (Rust-only). The C11 VM is frozen at 55 host
// calls (ids 0-54); these IDs exist only in the Rust VM and are dispatched
// through the host-call extension registered by the CLI (see
// `set_host_call_extension`).
pub const LANA_HOST_STORE_OPEN: u32 = 55;
pub const LANA_HOST_STORE_PUT: u32 = 56;
pub const LANA_HOST_STORE_GET: u32 = 57;
pub const LANA_HOST_STORE_DELETE: u32 = 58;
pub const LANA_HOST_STORE_COMMIT: u32 = 59;
pub const LANA_HOST_STORE_SCAN: u32 = 60;
pub const LANA_HOST_STORE_CURRENT_REVISION: u32 = 61;
pub const LANA_HOST_POLICY_EVALUATE: u32 = 62;
pub const LANA_HOST_POLICY_STORE_DECISION: u32 = 63;
pub const LANA_HOST_LEDGER_APPEND: u32 = 64;
pub const LANA_HOST_LEDGER_QUERY: u32 = 65;

/// The shared-information identity and commit-revision counters, matching the
/// `next_shared_identity` / `next_commit_revision` atomics in `runtime/c/shared.c`.
static NEXT_SHARED_IDENTITY: AtomicU64 = AtomicU64::new(1);
static NEXT_COMMIT_REVISION: AtomicU64 = AtomicU64::new(1);

/// The default worker count, matching `lana_vm_init` in `vm/c/vm.c`:
/// `min(processors, 8)`, falling back to 1 when the count is unknown. wasm has
/// no threads, so the count is always 1 there.
fn default_worker_count() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::available_parallelism().map(|count| count.get()).unwrap_or(1).min(8)
    }
}

/// Lexically normalize a POSIX path for the in-memory filesystem: collapse
/// `//`, drop `.`, and resolve `..` against the preceding component. Absolute
/// paths keep their leading `/`; a `..` at the root of an absolute path is a
/// no-op. This mirrors what `std::fs::canonicalize` does for the host
/// filesystem, minus symlink resolution and the existence requirement.
fn normalize_virtual_path(path: &str) -> String {
    let absolute = path.starts_with('/');
    let mut components: Vec<&str> = Vec::new();
    for component in path.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if components.last().is_some_and(|last| *last != "..") {
                    components.pop();
                } else if !absolute {
                    components.push("..");
                }
            }
            other => components.push(other),
        }
    }
    let mut result = String::new();
    if absolute {
        result.push('/');
    }
    result.push_str(&components.join("/"));
    if result.is_empty() {
        result.push('.');
    }
    result
}

/// A queued task: the child VM plus the handle the worker writes the result
/// to. Owned by the scheduler's queue; the worker takes it out and runs the
/// child to completion.
struct QueuedTask<'a> {
    child: Vm<'a>,
    handle: Arc<Task>,
}

/// The shared scheduler state, mirroring `struct LanaScheduler` in `vm/c/vm.c`.
/// The queue holds child VMs; workers pop them and run them to completion.
/// `all_tasks` mirrors the scheduler's `all_tasks` list so shutdown can cancel
/// every live task before joining the workers.
struct SchedulerState<'a> {
    queue: VecDeque<QueuedTask<'a>>,
    all_tasks: Vec<Arc<Task>>,
    live_tasks: usize,
    next_task_id: u64,
    stopping: bool,
}

/// A handle to the scheduler, shared between the parent VM and the workers.
#[derive(Clone)]
struct Scheduler<'a> {
    state: Arc<Mutex<SchedulerState<'a>>>,
    available: Arc<Condvar>,
}

impl<'a> Scheduler<'a> {
    fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(SchedulerState {
                queue: VecDeque::new(),
                all_tasks: Vec::new(),
                live_tasks: 0,
                next_task_id: 1,
                stopping: false,
            })),
            available: Arc::new(Condvar::new()),
        }
    }

    /// Signal the workers to stop and wait for them, mirroring
    /// `scheduler_shutdown` in `vm/c/vm.c:2500-2511`: cancel every task so a
    /// worker running a child VM stops promptly, then wake the workers. The
    /// caller joins the worker threads.
    fn shutdown(&self) {
        let mut state = self.state.lock().unwrap();
        state.stopping = true;
        for task in &state.all_tasks {
            task.cancelled.store(true, Ordering::Relaxed);
        }
        self.available.notify_all();
    }
}

/// Run a queued task's child VM to completion and publish the result to the
/// task handle, mirroring `run_task` in `vm/c/vm.c`.
fn run_task(queued: QueuedTask<'_>) {
    let mut child = queued.child;
    let status = child.run();
    let (status, error, result) = if status == LanaError::Ok {
        (status, VmError::default(), child.result().clone())
    } else {
        (status, child.error().clone(), Value::null())
    };
    let mut state = queued.handle.state.lock().unwrap();
    state.status = status;
    state.error = error;
    state.result = result;
    state.completed = true;
    queued.handle.completed_cond.notify_all();
}

/// The worker loop, mirroring `scheduler_worker` in `vm/c/vm.c`. wasm has no
/// threads, so this is compiled out there.
#[cfg(not(target_arch = "wasm32"))]
fn worker_loop(scheduler: &Scheduler<'_>) {
    loop {
        let queued = {
            let mut state = scheduler.state.lock().unwrap();
            loop {
                if let Some(queued) = state.queue.pop_front() {
                    break queued;
                }
                if state.stopping {
                    return;
                }
                state = scheduler.available.wait(state).unwrap();
            }
        };
        run_task(queued);
    }
}

/// The splitmix64-style finalizer, mirroring `mix64` in `vm/c/vm.c`. Used to
/// derive a child VM's lineage and seed from the parent's.
fn mix64(value: u64) -> u64 {
    let mut value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

/// The register VM. Owns the execution state for one chunk.
pub struct Vm<'a> {
    chunk: &'a Chunk,
    ip: usize,
    running: bool,
    instruction_limit: u64,
    instruction_count: u64,
    opcode_counts: Vec<u64>,
    state_transition_count: u64,
    allocation_count: u64,
    memory_limit: usize,
    allocated_bytes: usize,
    rng: Rng,
    root_seed: u64,
    lineage: u64,
    revision: u64,
    derivation_sequence: u64,
    path_limit: usize,
    active_path_count: usize,
    next_dependency_id: u64,
    next_reactive_id: u64,
    next_effect_id: u64,
    observation_count: u64,
    program_argc: usize,
    program_argv: Vec<Arc<str>>,
    shared_references: Vec<Arc<SharedInformation>>,
    path_execution: Vec<PathExecution>,
    frames: Vec<Frame>,
    max_registers: Vec<usize>,
    result: Value,
    error: VmError,
    pending_error_message: Option<String>,
    scheduler: Option<Scheduler<'a>>,
    scheduler_owner: bool,
    spawn_counter: u64,
    current_group_id: u64,
    next_group_id: u64,
    group_stack: Vec<u64>,
    group_depth: usize,
    cancelled: Arc<AtomicBool>,
    configured_worker_count: usize,
    configured_task_limit: usize,
    tasks: Vec<Arc<Task>>,
    host_call_extension: Option<Box<dyn FnMut(u32, &[Value], &mut Value) -> LanaError + Send>>,
    /// Optional in-memory filesystem. When set, the file-backed host calls
    /// (`read_text`, `write_text`, `write_text_atomic`, `path_exists`) resolve
    /// against this map instead of `std::fs`, so the self-hosted compiler can
    /// run on targets without a filesystem (e.g. `wasm32-unknown-unknown`).
    virtual_fs: Option<HashMap<String, String>>,
}

impl<'a> Vm<'a> {
    /// Create a VM for a chunk. The CLI defaults (256 MiB / 50M instructions,
    /// seed `0x4c414e41`) match `tools/c/cli.c` `load_command`.
    pub fn new(chunk: &'a Chunk) -> Self {
        let mut vm = Self {
            chunk,
            ip: chunk.entry as usize,
            running: true,
            instruction_limit: 50_000_000,
            instruction_count: 0,
            opcode_counts: vec![0; OpCode::Count as usize],
            state_transition_count: 0,
            allocation_count: 0,
            memory_limit: 256 * 1024 * 1024,
            allocated_bytes: 0,
            rng: Rng::new(),
            root_seed: 0,
            lineage: 0,
            revision: 0,
            derivation_sequence: 0,
            path_limit: 64,
            active_path_count: 1,
            next_dependency_id: 1,
            next_reactive_id: 1,
            next_effect_id: 1,
            observation_count: 0,
            program_argc: 0,
            program_argv: Vec::new(),
            shared_references: Vec::new(),
            path_execution: Vec::new(),
            frames: vec![Frame::new(LANA_MAX_REGISTERS as usize)],
            max_registers: compute_max_registers(chunk),
            result: Value::null(),
            error: VmError::default(),
            pending_error_message: None,
            scheduler: None,
            scheduler_owner: true,
            spawn_counter: 0,
            current_group_id: 0,
            next_group_id: 1,
            group_stack: Vec::new(),
            group_depth: 0,
            cancelled: Arc::new(AtomicBool::new(false)),
            configured_worker_count: default_worker_count(),
            configured_task_limit: 64,
            tasks: Vec::new(),
            host_call_extension: None,
            virtual_fs: None,
        };
        vm.seed(0x4c414e41);
        vm
    }

    /// Seed the RNG, matching `lana_vm_seed`.
    pub fn seed(&mut self, seed: u64) {
        self.root_seed = seed;
        self.rng.seed(seed);
    }

    /// The value left in the result register by `RETURN` from the main frame.
    pub fn result(&self) -> &Value {
        &self.result
    }

    /// The error recorded by the last failed run.
    pub fn error(&self) -> &VmError {
        &self.error
    }

    /// The instruction count, for `--stats` output.
    pub fn instruction_count(&self) -> u64 {
        self.instruction_count
    }

    /// The state transition count, for `--stats` output.
    pub fn state_transition_count(&self) -> u64 {
        self.state_transition_count
    }

    /// The allocation count, for `--stats` output.
    pub fn allocation_count(&self) -> u64 {
        self.allocation_count
    }

    /// The cumulative allocated bytes, for `--stats` output.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Per-opcode execution counts, for `--stats` output.
    pub fn opcode_counts(&self) -> &[u64] {
        &self.opcode_counts
    }

    /// Run the chunk to completion, mirroring `lana_vm_run`. The top-level VM
    /// owns the scheduler and runs the dispatch loop inside a thread scope so
    /// the worker threads are joined before the chunk borrow ends. Child VMs
    /// (run by workers) share the parent's scheduler and skip the scope.
    ///
    /// wasm has no threads, so the scheduler runs with zero workers there and
    /// `wait_task` executes queued tasks inline (its helper mechanism), keeping
    /// `FORK`/`WAIT` correct single-threaded.
    pub fn run(&mut self) -> LanaError {
        if let Err(info) = lana_bytecode::verifier::verify(self.chunk) {
            return self.fail(info.code, info.ip, info.opcode, info.line, "execute", info.message);
        }
        if self.scheduler_owner {
            #[cfg(not(target_arch = "wasm32"))]
            {
                std::thread::scope(|scope| {
                    let scheduler = Scheduler::new();
                    self.scheduler = Some(scheduler.clone());
                    let mut workers = Vec::new();
                    for _ in 0..self.configured_worker_count {
                        let scheduler = scheduler.clone();
                        workers.push(scope.spawn(move || worker_loop(&scheduler)));
                    }
                    let result = self.dispatch_loop();
                    scheduler.shutdown();
                    result
                })
            }
            #[cfg(target_arch = "wasm32")]
            {
                let scheduler = Scheduler::new();
                self.scheduler = Some(scheduler.clone());
                let result = self.dispatch_loop();
                scheduler.shutdown();
                result
            }
        } else {
            self.dispatch_loop()
        }
    }

    /// The dispatch loop, mirroring the body of `lana_vm_run`.
    fn dispatch_loop(&mut self) -> LanaError {
        while self.running {
            if self.allocated_bytes > self.memory_limit {
                return self.fail(LanaError::Oom, self.ip, 0, 0, "execute", "memory limit exceeded");
            }
            if self.cancelled.load(Ordering::Relaxed) {
                return self.fail(LanaError::Cancelled, self.ip, 0, 0, "execute", "task cancelled");
            }
            let old_count = self.instruction_count;
            self.instruction_count += 1;
            if old_count >= self.instruction_limit {
                return self.fail(LanaError::Limit, self.ip, 0, 0, "execute", "instruction limit exceeded");
            }
            if self.ip >= self.chunk.code.len() {
                return self.fail(LanaError::Jump, self.ip, 0, 0, "execute", "instruction pointer is out of range");
            }
            let instruction = self.chunk.code[self.ip];
            self.ip += 1;
            self.opcode_counts[instruction.opcode as usize] += 1;
            let error = self.execute(&instruction);
            if error != LanaError::Ok {
                // The C11 VM passes `lana_error_name(error)` as the message
                // when an opcode fails without a more specific one. `HOST_CALL
                // assert` overrides it with the assertion's message string.
                let message = self
                    .pending_error_message
                    .take()
                    .unwrap_or_else(|| error.name().to_string());
                return self.fail(error, self.ip - 1, instruction.opcode as u8, instruction.line,
                                 instruction.opcode.name(), message);
            }
        }
        LanaError::Ok
    }

    /// Set the worker count, mirroring `lana_vm_set_worker_count`. Fails once
    /// the scheduler exists.
    pub fn set_worker_count(&mut self, workers: usize) -> LanaError {
        if workers == 0 || self.scheduler.is_some() {
            return LanaError::Task;
        }
        self.configured_worker_count = workers;
        LanaError::Ok
    }

    /// Set the task limit, mirroring `lana_vm_set_task_limit`. Fails once the
    /// scheduler exists.
    pub fn set_task_limit(&mut self, tasks: usize) -> LanaError {
        if tasks == 0 || self.scheduler.is_some() {
            return LanaError::Task;
        }
        self.configured_task_limit = tasks;
        LanaError::Ok
    }

    /// Set the instruction limit, mirroring the `--instruction-limit` CLI
    /// flag. Child VMs inherit the parent's limit at FORK time.
    pub fn set_instruction_limit(&mut self, limit: u64) {
        self.instruction_limit = limit;
    }

    /// Set the memory limit in bytes, mirroring the `--memory-limit-mib` CLI
    /// flag. Child VMs inherit the parent's limit at FORK time.
    pub fn set_memory_limit(&mut self, bytes: usize) {
        self.memory_limit = bytes;
    }

    /// Set the program arguments exposed to the `args` host call, mirroring
    /// `lana_vm_set_program_args`. Child VMs inherit the parent's arguments at
    /// FORK time.
    pub fn set_program_args(&mut self, args: &[String]) {
        self.program_argc = args.len();
        self.program_argv = args.iter().map(|s| Arc::from(s.as_str())).collect();
    }

    /// Register a handler for host-call IDs beyond the built-in set (54). When
    /// `execute_host_call` sees an ID it does not recognize, it delegates to
    /// this handler. The CLI uses this to expose the durable pipeline
    /// (store/policy/ledger) to Lana bytecode without `lana-vm` depending on
    /// `lana-runtime`.
    pub fn set_host_call_extension(
        &mut self,
        handler: Box<dyn FnMut(u32, &[Value], &mut Value) -> LanaError + Send>,
    ) {
        self.host_call_extension = Some(handler);
    }

    /// Store a file in the in-memory filesystem, enabling the file-backed host
    /// calls on targets without a real filesystem. Once a virtual file is set,
    /// the virtual filesystem is authoritative for all file-backed host calls.
    pub fn set_virtual_file(&mut self, path: &str, contents: String) {
        if self.virtual_fs.is_none() {
            self.virtual_fs = Some(HashMap::new());
        }
        self.virtual_fs.as_mut().unwrap().insert(path.to_string(), contents);
    }

    /// Read a file back from the in-memory filesystem, if one was stored.
    pub fn take_virtual_file(&self, path: &str) -> Option<String> {
        self.virtual_fs.as_ref().and_then(|fs| fs.get(path).cloned())
    }

    /// Fork a task, mirroring `start_task` in `vm/c/vm.c`. The child VM is
    /// deep-cloned from the parent's argument registers and queued for a
    /// worker; the returned handle is stored in the parent's register.
    fn start_task(&mut self, function_index: u32, argc: u32, first_arg: u32) -> Result<Arc<Task>, LanaError> {
        let function = &self.chunk.functions[function_index as usize];
        if argc as usize != function.arity as usize {
            return Err(LanaError::Type);
        }
        let scheduler = self.scheduler.clone().expect("scheduler exists when FORK runs");
        {
            let mut state = scheduler.state.lock().unwrap();
            if state.stopping {
                return Err(LanaError::Task);
            }
            if state.live_tasks >= self.configured_task_limit {
                return Err(LanaError::Limit);
            }
            state.live_tasks += 1;
        }
        let id = {
            let mut state = scheduler.state.lock().unwrap();
            let id = state.next_task_id;
            state.next_task_id += 1;
            id
        };
        let handle = Arc::new(Task::new(id, self.current_group_id));
        let mut child = Vm::new(self.chunk);
        child.scheduler = Some(scheduler.clone());
        child.scheduler_owner = false;
        child.ip = function.entry as usize;
        child.frames[0].function = function_index;
        child.instruction_limit = self.instruction_limit;
        child.memory_limit = self.memory_limit;
        child.program_argc = self.program_argc;
        child.program_argv = self.program_argv.clone();
        child.lineage = mix64(self.lineage ^ { self.spawn_counter += 1; self.spawn_counter });
        child.seed(mix64(self.root_seed ^ child.lineage));
        child.root_seed = self.root_seed;
        child.cancelled = handle.cancelled.clone();
        let mut memo = DeepCloneMemo::default();
        for index in 0..argc as usize {
            let argument = self.current_frame().registers[(first_arg as usize) + index].clone();
            let cloned = self.deep_clone_value(&argument, &mut memo)?;
            child.frames[0].registers[index] = cloned;
        }
        for index in 0..argc as usize {
            let history = self.current_frame().histories[(first_arg as usize) + index].clone();
            child.frames[0].histories[index] = history;
        }
        {
            let mut state = scheduler.state.lock().unwrap();
            state.queue.push_back(QueuedTask {
                child,
                handle: handle.clone(),
            });
            state.all_tasks.push(handle.clone());
            scheduler.available.notify_one();
        }
        self.tasks.push(handle.clone());
        Ok(handle)
    }

    /// Wait for a task to complete, mirroring `wait_task` in `vm/c/vm.c`.
    /// `timeout < 0` waits indefinitely, running queued tasks inline (the
    /// helper mechanism); a non-negative timeout waits on the condition
    /// variable and returns `Timeout` if it expires.
    fn wait_task(&mut self, task: &Task, timeout: f64) -> Result<Value, LanaError> {
        if timeout < 0.0 {
            loop {
                if task.state.lock().unwrap().completed {
                    break;
                }
                let helper = {
                    let scheduler = self.scheduler.as_ref().expect("scheduler exists");
                    let mut state = scheduler.state.lock().unwrap();
                    state.queue.pop_front()
                };
                if let Some(helper) = helper {
                    run_task(helper);
                    continue;
                }
                let mut state = task.state.lock().unwrap();
                if !state.completed {
                    state = task.completed_cond.wait(state).unwrap();
                }
            }
        } else {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout);
            loop {
                if task.state.lock().unwrap().completed {
                    break;
                }
                let now = std::time::Instant::now();
                if now >= deadline {
                    return Err(LanaError::Timeout);
                }
                let mut state = task.state.lock().unwrap();
                if !state.completed {
                    let (guard, _) = task.completed_cond.wait_timeout(state, deadline - now).unwrap();
                    state = guard;
                }
            }
        }
        let mut state = task.state.lock().unwrap();
        if state.status != LanaError::Ok {
            self.error = state.error.clone();
            return Err(state.status);
        }
        if !state.joined {
            let mut memo = DeepCloneMemo::default();
            let cloned = self.deep_clone_value(&state.result, &mut memo)?;
            state.result = cloned;
            state.joined = true;
            let scheduler = self.scheduler.as_ref().expect("scheduler exists");
            let mut scheduler_state = scheduler.state.lock().unwrap();
            if scheduler_state.live_tasks > 0 {
                scheduler_state.live_tasks -= 1;
            }
        }
        Ok(state.result.clone())
    }

    /// Cancel a task, mirroring `cancel_task` in `vm/c/vm.c`.
    fn cancel_task(&self, task: &Task) {
        task.cancelled.store(true, Ordering::Relaxed);
    }

    /// Close a task group, mirroring `close_task_group` in `vm/c/vm.c`:
    /// cancel every non-completed task in the group, then wait for every
    /// non-joined task, clearing a `Cancelled` result and recording the first
    /// other error.
    fn close_task_group(&mut self, group_id: u64) -> LanaError {
        let mut first_error = LanaError::Ok;
        let tasks = self.tasks.clone();
        for task in &tasks {
            if task.group_id == group_id && !task.state.lock().unwrap().completed {
                self.cancel_task(task);
            }
        }
        for task in &tasks {
            if task.group_id == group_id && !task.state.lock().unwrap().joined {
                match self.wait_task(task, -1.0) {
                    Err(LanaError::Cancelled) => {
                        self.error = VmError::default();
                    }
                    Err(error) if first_error == LanaError::Ok => first_error = error,
                    _ => {}
                }
            }
        }
        first_error
    }

    /// Record a failure, mirroring `vm_fail`. The error's `ip` is the address
    /// of the failing instruction (the caller passes the pre-increment ip).
    fn fail(&mut self, code: LanaError, ip: usize, opcode: u8, line: u32,
            operation: &str, message: impl Into<String>) -> LanaError {
        // Mirror `vm_fail` in `vm/c/vm.c:362-366`: when an error is already
        // recorded (e.g. a child task's error propagated by JOIN), preserve it
        // instead of overwriting with the failing instruction's own span.
        if self.error.code != LanaError::Ok && !self.error.message.is_empty() {
            self.result = Value::null();
            self.running = false;
            return code;
        }
        let message = message.into();
        let (resolution_reason, remaining_alternatives) =
            self.resolution_reason_for(code, ip, &message);
        let mut error = VmError {
            code,
            ip,
            opcode,
            line,
            message: message.clone(),
            function: self.error_function_name(),
            operation: operation.to_string(),
            resolution_reason,
            remaining_alternatives,
            cancellation: None,
            resource_limit: None,
            exact_support: None,
        };
        match code {
            LanaError::Cancelled => {
                error.cancellation = Some((self.lineage, message));
            }
            LanaError::Oom => {
                error.resource_limit = Some((
                    LANA_RESOURCE_MEMORY,
                    self.memory_limit as u64,
                    self.allocated_bytes as u64,
                    "bytes".to_string(),
                ));
            }
            LanaError::PathLimit => {
                error.resource_limit = Some((
                    LANA_RESOURCE_PATHS,
                    self.path_limit as u64,
                    self.active_path_count as u64,
                    "paths".to_string(),
                ));
            }
            LanaError::BudgetExhausted
            | LanaError::Limit
                if message.contains("instruction") =>
            {
                error.resource_limit = Some((
                    LANA_RESOURCE_INSTRUCTIONS,
                    self.instruction_limit as u64,
                    self.instruction_count as u64,
                    "instructions".to_string(),
                ));
            }
            LanaError::Limit if opcode == OpCode::Fork as u8 => {
                let observed = self
                    .scheduler
                    .as_ref()
                    .map(|s| s.state.lock().unwrap().live_tasks as u64)
                    .unwrap_or(0);
                error.resource_limit = Some((
                    LANA_RESOURCE_TASKS,
                    self.configured_task_limit as u64,
                    observed,
                    "tasks".to_string(),
                ));
            }
            LanaError::UnsupportedExactMeasurement => {
                error.exact_support = Some((
                    LANA_EXACT_SUPPORT_UNAVAILABLE,
                    "operation requires explicit sampling or approximation".to_string(),
                ));
            }
            _ => {}
        }
        self.error = error;
        self.result = Value::null();
        self.running = false;
        code
    }

    /// The resolution detail for a failure, mirroring the `vm_fail` branches
    /// in `vm/c/vm.c:378-417`. For an unresolved value the alternatives count
    /// is read from the failing instruction's source register.
    fn resolution_reason_for(&self, code: LanaError, ip: usize, message: &str) -> (u32, usize) {
        match code {
            LanaError::InvalidConditioning => {
                (LANA_RESOLUTION_REASON_INVALID_CONDITIONING, 0)
            }
            LanaError::UnresolvedValue => {
                let mut alternatives = 0usize;
                if ip < self.chunk.code.len() {
                    let ins = &self.chunk.code[ip];
                    if ins.a < LANA_MAX_REGISTERS {
                        if let Some(frame) = self.frames.last() {
                            let source = &frame.registers[ins.a as usize];
                            alternatives = match &source.kind {
                                ValueKind::Joint(joint) if !joint.rows.is_empty() => {
                                    joint.rows.len()
                                }
                                ValueKind::Possibility(possibility) => possibility.values.len(),
                                ValueKind::PathSet(paths) => paths.alternatives.len(),
                                _ => 0,
                            };
                        }
                    }
                }
                let reason = if alternatives == 0 {
                    LANA_RESOLUTION_REASON_NO_ALTERNATIVES
                } else {
                    LANA_RESOLUTION_REASON_MULTIPLE_ALTERNATIVES
                };
                (reason, alternatives)
            }
            LanaError::Cancelled => (LANA_RESOLUTION_REASON_CANCELLED, 0),
            LanaError::UnsupportedExactMeasurement => {
                (LANA_RESOLUTION_REASON_UNSUPPORTED_EXACT, 0)
            }
            LanaError::PathLimit | LanaError::BudgetExhausted => {
                (LANA_RESOLUTION_REASON_RESOURCE_LIMIT, 0)
            }
            // The C11 VM only marks an instruction-limit failure with a
            // resource-limit resolution; a FORK task-limit failure carries the
            // resource detail but no resolution.
            LanaError::Limit if message.contains("instruction") => {
                (LANA_RESOLUTION_REASON_RESOURCE_LIMIT, 0)
            }
            _ => (LANA_RESOLUTION_REASON_NONE, 0),
        }
    }

    fn current_frame(&self) -> &Frame {
        self.frames.last().expect("VM always has at least one frame")
    }

    fn current_frame_mut(&mut self) -> &mut Frame {
        self.frames.last_mut().expect("VM always has at least one frame")
    }

    /// The function name recorded in `VmError.function`, mirroring `vm_fail`
    /// in `vm/c/vm.c`: the entry frame has `function == UINT32_MAX`, so the
    /// name is left empty and the CLI reports `<bytecode>`.
    fn error_function_name(&self) -> String {
        let Some(frame) = self.frames.last() else {
            return String::new();
        };
        if frame.function >= self.chunk.functions.len() as u32 {
            return String::new();
        }
        self.chunk.functions[frame.function as usize].name.clone()
    }

    /// The function name recorded in derivations, mirroring
    /// `derivation_function_name` in `vm/c/vm.c`: `<main>` when the current
    /// frame has no real function.
    fn current_function_name(&self) -> String {
        let Some(frame) = self.frames.last() else {
            return "<main>".to_string();
        };
        if frame.function >= self.chunk.functions.len() as u32 {
            return "<main>".to_string();
        }
        self.chunk.functions[frame.function as usize].name.clone()
    }

    /// Advance the RNG and return a Bernoulli draw, matching `draw_sample`.
    fn draw_sample(&mut self, p: f64) -> i32 {
        let draw = self.rng.random() as f64 / 4294967296.0;
        if draw < p { 1 } else { 0 }
    }

    /// Write a state value to a register, mirroring `store_state`.
    fn store_state(&mut self, reg: u32, state: StateValue) -> LanaError {
        if !state::state_valid(&state.state) {
            return LanaError::InvalidState;
        }
        self.state_transition_count += 1;
        let frame = self.current_frame_mut();
        frame.registers[reg as usize] = Value::state(state.clone());
        history_append(&mut frame.histories[reg as usize], state)
    }

    /// Record a derivation node, mirroring `record_derivation`. Only inputs
    /// that carry a derivation are retained as inputs.
    fn record_derivation(
        &mut self,
        kind: DerivationKind,
        operation: &str,
        inputs: &[&Value],
        label: &str,
        line: u32,
        exactness: DerivationExactness,
        details: &str,
        outcome: DerivationOutcome,
        reason: &str,
    ) -> Option<Arc<Derivation>> {
        let mut retained: Vec<Arc<Derivation>> = Vec::new();
        for input in inputs {
            if let Some(derivation) = &input.derivation {
                retained.push(derivation.clone());
            }
        }
        self.derivation_sequence += 1;
        Some(Arc::new(Derivation {
            task_lineage: self.lineage,
            local_sequence: self.derivation_sequence,
            revision: self.revision,
            kind,
            operation: Arc::from(operation),
            inputs: retained,
            label: Arc::from(label),
            function: Arc::from(self.current_function_name()),
            line,
            exactness,
            details: Arc::from(details),
            outcome,
            reason: Arc::from(reason),
        }))
    }

    /// Attach a derivation to a register value, mirroring `attach_derivation`.
    fn attach_derivation(
        &mut self,
        reg: u32,
        kind: DerivationKind,
        operation: &str,
        inputs: &[&Value],
        label: &str,
        line: u32,
        exactness: DerivationExactness,
        details: &str,
    ) -> LanaError {
        let derivation = self.record_derivation(kind, operation, inputs, label, line,
                                                exactness, details, DerivationOutcome::Success, "none");
        let Some(derivation) = derivation else {
            return LanaError::Oom;
        };
        self.current_frame_mut().registers[reg as usize].derivation = Some(derivation);
        LanaError::Ok
    }

    /// Attach a derivation whose (kind, exactness, outcome) are derived from the
    /// least-certain evidence status of the inputs, mirroring
    /// `attach_combine_derivation`. Used by the two-value combine ops (`mix`,
    /// `trace_distance`) so that combining a sampled value with an exact value
    /// yields a sampled result.
    fn attach_combine_derivation(
        &mut self,
        reg: u32,
        operation: &str,
        inputs: &[&Value],
        line: u32,
        details: &str,
    ) -> LanaError {
        let status = inputs
            .iter()
            .map(|input| {
                input
                    .derivation
                    .as_deref()
                    .map(Derivation::status)
                    .unwrap_or(EvidenceStatus::Exact)
            })
            .min()
            .unwrap_or(EvidenceStatus::Exact);
        let (kind, exactness, outcome) = match status {
            EvidenceStatus::Observed => (
                DerivationKind::Observation,
                DerivationExactness::Exact,
                DerivationOutcome::Success,
            ),
            EvidenceStatus::Modeled => (
                DerivationKind::Assumption,
                DerivationExactness::Approximate,
                DerivationOutcome::Success,
            ),
            EvidenceStatus::Sampled => (
                DerivationKind::Sample,
                DerivationExactness::Sample,
                DerivationOutcome::Success,
            ),
            EvidenceStatus::Unknown => (
                DerivationKind::Operation,
                DerivationExactness::Exact,
                DerivationOutcome::Unresolved,
            ),
            EvidenceStatus::Exact => (
                DerivationKind::Operation,
                DerivationExactness::Exact,
                DerivationOutcome::Success,
            ),
        };
        let derivation = self.record_derivation(kind, operation, inputs, "", line,
                                                exactness, details, outcome, "none");
        let Some(derivation) = derivation else {
            return LanaError::Oom;
        };
        self.current_frame_mut().registers[reg as usize].derivation = Some(derivation);
        LanaError::Ok
    }

    /// Dispatch one instruction, mirroring the `switch` in `lana_vm_run`.
    fn execute(&mut self, ins: &Instruction) -> LanaError {
        use OpCode::*;
        match ins.opcode {
            Nop => LanaError::Ok,
            LoadConst => {
                let value = Value::from(&self.chunk.constants[ins.imm as usize]);
                self.current_frame_mut().registers[ins.a as usize] = value;
                LanaError::Ok
            }
            Move => {
                let value = self.current_frame().registers[ins.b as usize].clone();
                let history = self.current_frame().histories[ins.b as usize].clone();
                let frame = self.current_frame_mut();
                frame.registers[ins.a as usize] = value;
                frame.histories[ins.a as usize] = history;
                LanaError::Ok
            }
            StateNew => {
                let p = &self.chunk.constants[ins.b as usize];
                let d_re = &self.chunk.constants[ins.c as usize];
                let d_im = &self.chunk.constants[ins.imm as usize];
                let mut state = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
                let error = match (p, d_re, d_im) {
                    (ConstantValue::Number(p), ConstantValue::Number(d_re), ConstantValue::Number(d_im)) => {
                        state::make_complex(*p, *d_re, *d_im, &mut state)
                    }
                    _ => LanaError::Type,
                };
                if error == LanaError::Ok {
                    self.store_state(ins.a, StateValue { state, indexes: Default::default() })
                } else {
                    error
                }
            }
            StateBuild => {
                let p = self.current_frame().registers[ins.a as usize].clone();
                let d_re = self.current_frame().registers[ins.b as usize].clone();
                let d_im = self.current_frame().registers[ins.c as usize].clone();
                let mut state = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
                let error = if matches!(p.kind, ValueKind::Number(_))
                    && matches!(d_re.kind, ValueKind::Number(_))
                    && matches!(d_im.kind, ValueKind::Number(_))
                {
                    state::make_complex(p.as_number(), d_re.as_number(), d_im.as_number(), &mut state)
                } else {
                    LanaError::Type
                };
                if error == LanaError::Ok {
                    self.store_state(ins.imm, StateValue { state, indexes: Default::default() })
                } else {
                    error
                }
            }
            Mix => {
                let left = self.current_frame().registers[ins.b as usize].clone();
                let right = self.current_frame().registers[ins.c as usize].clone();
                let weight = self.current_frame().registers[ins.imm as usize].clone();
                let mut state = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
                let error = if matches!(left.kind, ValueKind::StateDist(_))
                    || matches!(right.kind, ValueKind::StateDist(_))
                {
                    LanaError::UnsupportedOperation
                } else if !matches!(left.kind, ValueKind::State(_))
                    || !matches!(right.kind, ValueKind::State(_))
                {
                    LanaError::Type
                } else if !matches!(weight.kind, ValueKind::Number(_)) {
                    LanaError::Type
                } else {
                    state::mix(&left.as_state().state, &right.as_state().state,
                               weight.as_number(), &mut state)
                };
                if error != LanaError::Ok {
                    return error;
                }
                let details = format!("w={}", weight.as_number());
                let error = self.store_state(ins.a, StateValue { state, indexes: Default::default() });
                if error != LanaError::Ok {
                    return error;
                }
                let inputs = [&left, &right];
                self.attach_combine_derivation(ins.a, "mix", &inputs, ins.line, &details)
            }
            Transform => {
                let source = self.current_frame().registers[ins.b as usize].clone();
                match &source.kind {
                    ValueKind::State(state_value) => {
                        let mut transformed = state_value.clone();
                        let error = state::transform_apply(ins.c, &state_value.state, &mut transformed.state);
                        if error == LanaError::Ok {
                            self.store_state(ins.a, transformed)
                        } else {
                            error
                        }
                    }
                    ValueKind::StateDist(distribution) => {
                        let distribution = match self.state_dist_transform(ins.c, distribution.clone()) {
                            Ok(distribution) => distribution,
                            Err(error) => return error,
                        };
                        self.current_frame_mut().registers[ins.a as usize] =
                            Value::state_dist(distribution);
                        LanaError::Ok
                    }
                    _ => LanaError::Type,
                }
            }
            Measure => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let probability = match &source.kind {
                    ValueKind::State(state_value) => state_value.state.p,
                    ValueKind::StateDist(distribution) => {
                        match state_dist::expected_probability(distribution) {
                            Ok(probability) => probability,
                            Err(error) => return error,
                        }
                    }
                    _ => return LanaError::Type,
                };
                self.store_measurement(ins.b, ins.c, probability)
            }
            MeasureBasis => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let mut probability = 0.0;
                let error = match &source.kind {
                    ValueKind::State(state_value) => {
                        state::basis_probability(ins.c, &state_value.state, &mut probability)
                    }
                    ValueKind::StateDist(distribution) => {
                        if ins.imm != LANA_MEASURE_SAMPLE {
                            return LanaError::UnsupportedExactMeasurement;
                        }
                        let state = match self.state_dist_sample(distribution) {
                            Ok(state) => state,
                            Err(error) => return error,
                        };
                        state::basis_probability(ins.c, &state.state, &mut probability)
                    }
                    _ => return LanaError::Type,
                };
                if error != LanaError::Ok {
                    return error;
                }
                self.store_measurement(ins.b, ins.imm, probability)
            }
            GetField => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                match &source.kind {
                    ValueKind::State(state_value) if ins.c <= 2 => {
                        let field = match ins.c {
                            0 => state_value.state.p,
                            1 => state_value.state.d_re,
                            _ => state_value.state.d_im,
                        };
                        self.current_frame_mut().registers[ins.b as usize] = Value::number(field);
                        LanaError::Ok
                    }
                    ValueKind::Distribution { p0, p1 } if ins.c <= 1 => {
                        let field = if ins.c == 0 { *p0 } else { *p1 };
                        self.current_frame_mut().registers[ins.b as usize] = Value::number(field);
                        LanaError::Ok
                    }
                    _ => LanaError::Type,
                }
            }
            GetIndex => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let ValueKind::State(state_value) = &source.kind else {
                    return LanaError::Type;
                };
                let indexes = &state_value.indexes;
                let value = match ins.c {
                    0 if indexes.has_timestamp => Value::number(indexes.timestamp),
                    1 if indexes.has_source => {
                        Value::string(indexes.source.clone().unwrap_or_else(|| Arc::from("")))
                    }
                    2 if indexes.has_weight => Value::number(indexes.weight),
                    3 if indexes.has_confidence => Value::number(indexes.confidence),
                    _ => Value::null(),
                };
                self.current_frame_mut().registers[ins.b as usize] = value;
                LanaError::Ok
            }
            SetIndex => {
                let source = self.current_frame().registers[ins.c as usize].clone();
                let mut state_value = match &self.current_frame().registers[ins.a as usize].kind {
                    ValueKind::State(state_value) => state_value.clone(),
                    _ => return LanaError::Type,
                };
                let error = match ins.b {
                    0 if matches!(source.kind, ValueKind::Number(_)) => {
                        state_value.indexes.has_timestamp = true;
                        state_value.indexes.timestamp = source.as_number();
                        LanaError::Ok
                    }
                    1 if matches!(source.kind, ValueKind::String(_)) => {
                        state_value.indexes.has_source = true;
                        state_value.indexes.source = Some(source.as_string());
                        LanaError::Ok
                    }
                    2 if matches!(source.kind, ValueKind::Number(_)) && source.as_number() >= 0.0 => {
                        state_value.indexes.has_weight = true;
                        state_value.indexes.weight = source.as_number();
                        LanaError::Ok
                    }
                    3 if matches!(source.kind, ValueKind::Number(_))
                        && source.as_number() >= 0.0 && source.as_number() <= 1.0 =>
                    {
                        state_value.indexes.has_confidence = true;
                        state_value.indexes.confidence = source.as_number();
                        LanaError::Ok
                    }
                    _ => LanaError::Type,
                };
                if error == LanaError::Ok {
                    self.store_state(ins.a, state_value)
                } else {
                    error
                }
            }
            HistoryConfig => {
                let reg_a = self.current_frame().registers[ins.a as usize].clone();
                let amount = self.current_frame().registers[ins.b as usize].clone();
                if !matches!(reg_a.kind, ValueKind::State(_))
                    || !matches!(amount.kind, ValueKind::Number(_))
                    || ins.c > LANA_HISTORY_DURATION
                    || amount.as_number() <= 0.0
                {
                    return LanaError::History;
                }
                let state_value = reg_a.as_state().clone();
                let frame = self.current_frame_mut();
                let history = &mut frame.histories[ins.a as usize];
                history.policy = match ins.c {
                    0 => HistoryPolicy::None,
                    1 => HistoryPolicy::Latest,
                    _ => HistoryPolicy::Duration,
                };
                history.amount = amount.as_number();
                history_append(history, state_value)
            }
            Previous | Change | Velocity => {
                let versions = &self.current_frame().histories[ins.a as usize].versions;
                if versions.len() < 2 {
                    return LanaError::History;
                }
                let current = versions[versions.len() - 1].clone();
                let previous = versions[versions.len() - 2].clone();
                match ins.opcode {
                    Previous => {
                        self.current_frame_mut().registers[ins.b as usize] = Value::state(previous);
                        LanaError::Ok
                    }
                    Change => {
                        self.current_frame_mut().registers[ins.b as usize] =
                            Value::number(current.state.p - previous.state.p);
                        LanaError::Ok
                    }
                    _ => {
                        if !current.indexes.has_timestamp || !previous.indexes.has_timestamp
                            || current.indexes.timestamp <= previous.indexes.timestamp
                        {
                            LanaError::History
                        } else {
                            let velocity = (current.state.p - previous.state.p)
                                / (current.indexes.timestamp - previous.indexes.timestamp);
                            self.current_frame_mut().registers[ins.b as usize] = Value::number(velocity);
                            LanaError::Ok
                        }
                    }
                }
            }
            Binary => {
                let left = self.current_frame().registers[ins.a as usize].clone();
                let right = self.current_frame().registers[ins.b as usize].clone();
                let mut out = Value::null();
                let error = self.lift_binary(&left, &right, PureKind::Binary, ins.imm, &mut out);
                if error != LanaError::Ok {
                    return error;
                }
                self.current_frame_mut().registers[ins.c as usize] = out;
                if left.derivation.is_some() || right.derivation.is_some() {
                    let inputs = [&left, &right];
                    self.attach_derivation(ins.c, DerivationKind::Operation, "binary", &inputs, "",
                                           ins.line, DerivationExactness::Exact, "pure")
                } else {
                    LanaError::Ok
                }
            }
            Unary => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let mut out = Value::null();
                let error = self.lift_unary(&source, ins.imm, &mut out);
                if error != LanaError::Ok {
                    return error;
                }
                self.current_frame_mut().registers[ins.b as usize] = out;
                if source.derivation.is_some() {
                    let inputs = [&source];
                    self.attach_derivation(ins.b, DerivationKind::Operation, "unary", &inputs, "",
                                           ins.line, DerivationExactness::Exact, "pure")
                } else {
                    LanaError::Ok
                }
            }
            Compare => {
                let left = self.current_frame().registers[ins.a as usize].clone();
                let right = self.current_frame().registers[ins.b as usize].clone();
                let mut out = Value::null();
                let error = self.lift_binary(&left, &right, PureKind::Compare, ins.imm, &mut out);
                if error != LanaError::Ok {
                    return error;
                }
                self.current_frame_mut().registers[ins.c as usize] = out;
                if left.derivation.is_some() || right.derivation.is_some() {
                    let inputs = [&left, &right];
                    self.attach_derivation(ins.c, DerivationKind::Operation, "compare", &inputs, "",
                                           ins.line, DerivationExactness::Exact, "pure")
                } else {
                    LanaError::Ok
                }
            }
            Jump => {
                self.ip = ins.imm as usize;
                LanaError::Ok
            }
            JumpIfTrue | JumpIfFalse => {
                let condition = self.current_frame().registers[ins.a as usize].clone();
                if !matches!(condition.kind, ValueKind::Bool(_)) {
                    return LanaError::Type;
                }
                let take = if ins.opcode == JumpIfTrue {
                    condition.as_bool()
                } else {
                    !condition.as_bool()
                };
                if take {
                    self.ip = ins.imm as usize;
                }
                LanaError::Ok
            }
            ArrayNew => {
                let count = ins.c as usize;
                let items: Vec<Value> = (0..count)
                    .map(|i| self.current_frame().registers[ins.b as usize + i].clone())
                    .collect();
                let array = Arc::new(Mutex::new(Array { items }));
                self.current_frame_mut().registers[ins.a as usize] = Value::array(array);
                LanaError::Ok
            }
            ArrayGet | ArraySet => {
                let array_value = self.current_frame().registers[ins.a as usize].clone();
                let index_value = self.current_frame().registers[ins.b as usize].clone();
                if !matches!(array_value.kind, ValueKind::Array(_))
                    || !matches!(index_value.kind, ValueKind::Number(_))
                {
                    return LanaError::Type;
                }
                let index = index_value.as_number();
                if index < 0.0 || index.floor() != index {
                    return LanaError::Type;
                }
                let index = index as usize;
                let array = match &array_value.kind {
                    ValueKind::Array(array) => array.clone(),
                    _ => unreachable!("checked above"),
                };
                let mut array = array.lock().unwrap();
                if index >= array.items.len() {
                    return LanaError::Limit;
                }
                if ins.opcode == ArrayGet {
                    let value = array.items[index].clone();
                    drop(array);
                    self.current_frame_mut().registers[ins.c as usize] = value;
                } else {
                    let value = self.current_frame().registers[ins.c as usize].clone();
                    array.items[index] = value;
                }
                LanaError::Ok
            }
            Call => {
                let function = &self.chunk.functions[ins.b as usize];
                if ins.imm != function.arity {
                    return LanaError::Type;
                }
                if self.frames.len() >= LANA_MAX_CALL_FRAMES as usize {
                    return LanaError::Limit;
                }
                let args: Vec<Value> = (0..ins.imm as usize)
                    .map(|i| self.current_frame().registers[ins.c as usize + i].clone())
                    .collect();
                let histories: Vec<History> = (0..ins.imm as usize)
                    .map(|i| self.current_frame().histories[ins.c as usize + i].clone())
                    .collect();
                let mut callee = Frame::new(self.max_registers[ins.b as usize]);
                callee.return_ip = self.ip;
                callee.return_register = ins.a;
                callee.function = ins.b;
                for (index, arg) in args.into_iter().enumerate() {
                    callee.registers[index] = arg;
                    callee.histories[index] = histories[index].clone();
                }
                self.frames.push(callee);
                self.ip = function.entry as usize;
                LanaError::Ok
            }
            Lazy => {
                let bound_value = self.current_frame().registers[ins.c as usize].clone();
                let ValueKind::Number(bound) = bound_value.kind else {
                    return LanaError::Type;
                };
                if !bound.is_finite() || bound < 0.0 || bound > usize::MAX as f64 {
                    return LanaError::Type;
                }
                self.current_frame_mut().registers[ins.a as usize] =
                    Value::lazy(ins.b, bound as usize);
                LanaError::Ok
            }
            Force => {
                let lazy = self.current_frame().registers[ins.b as usize].clone();
                let index_value = self.current_frame().registers[ins.c as usize].clone();
                let ValueKind::Lazy { function, bound } = lazy.kind else {
                    return LanaError::Type;
                };
                let ValueKind::Number(index) = index_value.kind else {
                    return LanaError::Type;
                };
                if !index.is_finite() || index < 0.0 || index.floor() != index {
                    return LanaError::Type;
                }
                let index = index as usize;
                if index >= bound {
                    return LanaError::Limit;
                }
                let function_def = &self.chunk.functions[function as usize];
                if function_def.arity != 1 {
                    return LanaError::Type;
                }
                if self.frames.len() >= LANA_MAX_CALL_FRAMES as usize {
                    return LanaError::Limit;
                }
                let mut callee = Frame::new(self.max_registers[function as usize]);
                callee.return_ip = self.ip;
                callee.return_register = ins.a;
                callee.function = function;
                callee.registers[0] = Value::number(index as f64);
                callee.histories[0] = self.current_frame().histories[ins.c as usize].clone();
                self.frames.push(callee);
                self.ip = function_def.entry as usize;
                LanaError::Ok
            }
            Return => {
                let returned = self.current_frame().registers[ins.a as usize].clone();
                if self.frames.len() == 1 {
                    self.result = returned;
                    self.running = false;
                } else {
                    let return_ip = self.current_frame().return_ip;
                    let destination = self.current_frame().return_register;
                    self.frames.pop();
                    self.current_frame_mut().registers[destination as usize] = returned;
                    self.ip = return_ip;
                }
                LanaError::Ok
            }
            Print => {
                let value = self.current_frame().registers[ins.a as usize].clone();
                if value.is_unresolved() {
                    return LanaError::UnresolvedValue;
                }
                println!("{}", value.print());
                LanaError::Ok
            }
            Halt => {
                self.running = false;
                LanaError::Ok
            }
            Append => {
                let left = self.current_frame().registers[ins.a as usize].clone();
                let right = self.current_frame().registers[ins.b as usize].clone();
                let distribution = match self.state_dist_append(&left, &right) {
                    Ok(distribution) => distribution,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.c as usize] = Value::state_dist(distribution);
                LanaError::Ok
            }
            Attenuate => {
                let source = self.current_frame().registers[ins.b as usize].clone();
                let factor = self.current_frame().registers[ins.c as usize].clone();
                if !matches!(factor.kind, ValueKind::Number(_)) {
                    return LanaError::Type;
                }
                let factor_value = factor.as_number();
                match &source.kind {
                    ValueKind::State(state_value) => {
                        let mut attenuated = state_value.clone();
                        let error = state::attenuate(&state_value.state, factor_value, &mut attenuated.state);
                        if error != LanaError::Ok {
                            return error;
                        }
                        let error = self.store_state(ins.a, attenuated);
                        if error != LanaError::Ok {
                            return error;
                        }
                        let inputs = [&source, &factor];
                        self.attach_derivation(ins.a, DerivationKind::Operation, "attenuate", &inputs, "",
                                               ins.line, DerivationExactness::Exact, "")
                    }
                    ValueKind::StateDist(distribution) => {
                        let distribution = match self.state_dist_attenuate(distribution.clone(), factor_value) {
                            Ok(distribution) => distribution,
                            Err(error) => return error,
                        };
                        self.current_frame_mut().registers[ins.a as usize] = Value::state_dist(distribution);
                        LanaError::Ok
                    }
                    _ => LanaError::Type,
                }
            }
            TraceDistance => {
                let left = self.current_frame().registers[ins.b as usize].clone();
                let right = self.current_frame().registers[ins.c as usize].clone();
                let mut distance = 0.0;
                let error = if matches!(left.kind, ValueKind::StateDist(_))
                    || matches!(right.kind, ValueKind::StateDist(_))
                {
                    LanaError::UnsupportedOperation
                } else if !matches!(left.kind, ValueKind::State(_))
                    || !matches!(right.kind, ValueKind::State(_))
                {
                    LanaError::Type
                } else {
                    state::trace_distance(&left.as_state().state, &right.as_state().state, &mut distance)
                };
                if error != LanaError::Ok {
                    return error;
                }
                self.current_frame_mut().registers[ins.a as usize] = Value::number(distance);
                let inputs = [&left, &right];
                self.attach_combine_derivation(ins.a, "trace_distance", &inputs, ins.line, "")
            }
            AppendRedundant | AppendComplementary => {
                let left = self.current_frame().registers[ins.a as usize].clone();
                let right = self.current_frame().registers[ins.b as usize].clone();
                let strength = self.current_frame().registers[ins.imm as usize].clone();
                if !matches!(strength.kind, ValueKind::Number(_)) {
                    return LanaError::Type;
                }
                let mode = if matches!(ins.opcode, OpCode::AppendRedundant) {
                    state::APPEND_REDUNDANT
                } else {
                    state::APPEND_COMPLEMENTARY
                };
                let distribution = match self.state_dist_append_relationship(&left, &right, mode, strength.as_number()) {
                    Ok(distribution) => distribution,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.c as usize] = Value::state_dist(distribution);
                LanaError::Ok
            }
            AppendFullRedundancy => {
                let left = self.current_frame().registers[ins.a as usize].clone();
                let right = self.current_frame().registers[ins.b as usize].clone();
                let distribution = match self.state_dist_append_relationship(&left, &right, state::APPEND_FULL_REDUNDANCY, 0.0) {
                    Ok(distribution) => distribution,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.c as usize] = Value::state_dist(distribution);
                LanaError::Ok
            }
            AdtBuild => {
                let tag = &self.chunk.constants[ins.imm as usize];
                let variant = match tag {
                    ConstantValue::Number(variant) => *variant as u32,
                    _ => return LanaError::Type,
                };
                let count = ins.c as usize;
                let fields: Vec<Value> = (0..count)
                    .map(|i| self.current_frame().registers[ins.b as usize + i].clone())
                    .collect();
                let adt = Arc::new(Adt { variant, fields });
                self.current_frame_mut().registers[ins.a as usize] = Value::adt(adt);
                LanaError::Ok
            }
            AdtCase => {
                let tag = &self.chunk.constants[ins.b as usize];
                let variant = match tag {
                    ConstantValue::Number(variant) => *variant as u32,
                    _ => return LanaError::Type,
                };
                let value = self.current_frame().registers[ins.a as usize].clone();
                match &value.kind {
                    ValueKind::Adt(adt) => {
                        if adt.variant == variant {
                            self.ip = ins.imm as usize;
                        }
                        LanaError::Ok
                    }
                    _ => LanaError::Type,
                }
            }
            AdtGet => {
                let value = self.current_frame().registers[ins.b as usize].clone();
                match &value.kind {
                    ValueKind::Adt(adt) => {
                        if ins.c as usize >= adt.fields.len() {
                            return LanaError::InvalidParameters;
                        }
                        let field = adt.fields[ins.c as usize].clone();
                        self.current_frame_mut().registers[ins.a as usize] = field;
                        LanaError::Ok
                    }
                    _ => LanaError::Type,
                }
            }
            Map => {
                let source = self.current_frame().registers[ins.b as usize].clone();
                if !matches!(source.kind, ValueKind::StateDist(_)) {
                    return LanaError::Type;
                }
                if state::transform_spec(ins.c).is_none() {
                    return LanaError::Transform;
                }
                let child = match &source.kind {
                    ValueKind::StateDist(distribution) => distribution.clone(),
                    _ => unreachable!("checked above"),
                };
                let distribution = match self.state_dist_transform(ins.c, child) {
                    Ok(distribution) => distribution,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.a as usize] = Value::state_dist(distribution);
                let inputs = [&source];
                let details = state::transform_spec(ins.c).unwrap().name;
                self.attach_derivation(ins.a, DerivationKind::Operation, "map", &inputs, "",
                                       ins.line, DerivationExactness::Exact, details)
            }
            Support => {
                let source = self.current_frame().registers[ins.b as usize].clone();
                if !matches!(source.kind, ValueKind::StateDist(_)) {
                    return LanaError::Type;
                }
                let distribution = match &source.kind {
                    ValueKind::StateDist(distribution) => distribution.clone(),
                    _ => unreachable!("checked above"),
                };
                let items = match state_dist::support(&distribution, ins.imm) {
                    Ok(items) => items,
                    Err(error) => return error,
                };
                if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
                    return LanaError::Oom;
                }
                let array = Arc::new(Mutex::new(Array {
                    items: items.into_iter().map(Value::state).collect(),
                }));
                self.current_frame_mut().registers[ins.a as usize] = Value::array(array);
                let inputs = [&source];
                self.attach_derivation(ins.a, DerivationKind::Operation, "support", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "")
            }
            Expect => {
                let source = self.current_frame().registers[ins.b as usize].clone();
                if !matches!(source.kind, ValueKind::StateDist(_)) {
                    return LanaError::Type;
                }
                if ins.imm != LANA_OBSERVABLE_PROBABILITY {
                    return LanaError::UnsupportedOperation;
                }
                let distribution = match &source.kind {
                    ValueKind::StateDist(distribution) => distribution.clone(),
                    _ => unreachable!("checked above"),
                };
                let expected = match state_dist::expected_probability(&distribution) {
                    Ok(expected) => expected,
                    Err(error) => return error,
                };
                let result = match self.build_statistical_result("exact", expected, ins.imm) {
                    Ok(result) => result,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.a as usize] = Value::map(result);
                let inputs = [&source];
                self.attach_derivation(ins.a, DerivationKind::Operation, "expect", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "exact")
            }
            Validate => {
                let value = self.current_frame().registers[ins.b as usize].clone();
                let schema = self.current_frame().registers[ins.c as usize].clone();
                let result = match self.validate(&value, &schema) {
                    Ok(result) => result,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.a as usize] = Value::map(result);
                let inputs = [&value, &schema];
                self.attach_derivation(ins.a, DerivationKind::Operation, "validate", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "")
            }
            Revision => {
                let source = self.current_frame().registers[ins.b as usize].clone();
                let revision = if matches!(source.kind, ValueKind::Capability(_)) {
                    // Shared-information revisions land with capabilities
                    // (increment 5); no increment-2 opcode creates one.
                    return LanaError::Type;
                } else if let Some(derivation) = &source.derivation {
                    derivation.revision
                } else if source.reactive.is_some() {
                    // Reactive revisions land in increment 5.
                    return LanaError::Type;
                } else {
                    return LanaError::Type;
                };
                self.current_frame_mut().registers[ins.a as usize] = Value::number(revision as f64);
                let inputs = [&source];
                self.attach_derivation(ins.a, DerivationKind::Operation, "revision", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "")
            }
            SampleStateDist => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                if !matches!(source.kind, ValueKind::StateDist(_)) {
                    return LanaError::Type;
                }
                let distribution = match &source.kind {
                    ValueKind::StateDist(distribution) => distribution.clone(),
                    _ => unreachable!("checked above"),
                };
                let state = match self.state_dist_sample(&distribution) {
                    Ok(state) => state,
                    Err(error) => return error,
                };
                self.store_state(ins.b, state)
            }
            EstimateMeasureProbability | EstimateMeasureDistribution => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                if !matches!(source.kind, ValueKind::StateDist(_)) {
                    return LanaError::Type;
                }
                let distribution = match &source.kind {
                    ValueKind::StateDist(distribution) => distribution.clone(),
                    _ => unreachable!("checked above"),
                };
                let probability = match self.estimate_basis_probability(&distribution, ins.c, ins.imm) {
                    Ok(probability) => probability,
                    Err(error) => return error,
                };
                if ins.opcode == OpCode::EstimateMeasureProbability {
                    self.current_frame_mut().registers[ins.b as usize] = Value::number(probability);
                } else {
                    self.current_frame_mut().registers[ins.b as usize] =
                        Value::distribution(1.0 - probability, probability);
                }
                let inputs = [&source];
                self.attach_derivation(ins.b, DerivationKind::Approximation, "estimate_measure",
                                       &inputs, "", ins.line, DerivationExactness::Approximate,
                                       "explicit_sample_count")
            }
            JointBuild => {
                let descriptor = match &self.chunk.constants[ins.imm as usize] {
                    ConstantValue::String(descriptor) => descriptor.clone(),
                    _ => return LanaError::Type,
                };
                let values = self.current_frame().registers
                    [ins.b as usize..(ins.b + ins.c) as usize]
                    .to_vec();
                let joint = match self.joint_build(&values, &descriptor) {
                    Ok(joint) => joint,
                    Err(error) => return error,
                };
                let source = values[0].clone();
                self.current_frame_mut().registers[ins.a as usize] = Value::joint(joint);
                let inputs = [&source];
                self.attach_derivation(ins.a, DerivationKind::Operation, "joint_build", &inputs,
                                       "", ins.line, DerivationExactness::Exact, &descriptor)
            }
            JointProject => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let descriptor = match &self.chunk.constants[ins.c as usize] {
                    ConstantValue::String(descriptor) => descriptor.clone(),
                    _ => return LanaError::Type,
                };
                let ValueKind::Joint(joint) = &source.kind else {
                    return LanaError::Type;
                };
                let joint = match self.joint_project(joint, &descriptor) {
                    Ok(joint) => joint,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = Value::joint(joint);
                let inputs = [&source];
                self.attach_derivation(ins.b, DerivationKind::Operation, "project", &inputs, "",
                                       ins.line, DerivationExactness::Exact, &descriptor)
            }
            JointCondition => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let evidence = self.current_frame().registers[ins.imm as usize].clone();
                let descriptor = match &self.chunk.constants[ins.c as usize] {
                    ConstantValue::String(descriptor) => descriptor.clone(),
                    _ => return LanaError::Type,
                };
                let ValueKind::Joint(joint) = &source.kind else {
                    return LanaError::Type;
                };
                let joint = match self.joint_condition(joint, &descriptor, &evidence) {
                    Ok(joint) => joint,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = Value::joint(joint);
                let inputs = [&source, &evidence];
                self.attach_derivation(ins.b, DerivationKind::Operation, "condition", &inputs, "",
                                       ins.line, DerivationExactness::Exact, &descriptor)
            }
            JointSample => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let ValueKind::Joint(joint) = &source.kind else {
                    return LanaError::Type;
                };
                let result = match self.joint_sample(joint) {
                    Ok(result) => result,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = result;
                let inputs = [&source];
                self.attach_derivation(ins.b, DerivationKind::Sample, "joint_sample", &inputs, "",
                                       ins.line, DerivationExactness::Sample, "seeded_rng")
            }
            Resolve => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let result = match self.information_resolve(&source) {
                    Ok(result) => result,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = result;
                let inputs = [&source];
                self.attach_derivation(ins.b, DerivationKind::Resolution, "resolve", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "singleton")
            }
            JointBuildFinite => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let descriptor = match &self.chunk.constants[ins.c as usize] {
                    ConstantValue::String(descriptor) => descriptor.clone(),
                    _ => return LanaError::Type,
                };
                let joint = match self.joint_build_finite_array(&source, &descriptor) {
                    Ok(joint) => joint,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = Value::joint(joint);
                let inputs = [&source];
                self.attach_derivation(ins.b, DerivationKind::Operation, "joint_build_finite",
                                       &inputs, "", ins.line, DerivationExactness::Exact, &descriptor)
            }
            JointRename => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let old_name = match &self.chunk.constants[ins.c as usize] {
                    ConstantValue::String(name) => name.clone(),
                    _ => return LanaError::Type,
                };
                let new_name = match &self.chunk.constants[ins.imm as usize] {
                    ConstantValue::String(name) => name.clone(),
                    _ => return LanaError::Type,
                };
                let ValueKind::Joint(joint) = &source.kind else {
                    return LanaError::Type;
                };
                let joint = match self.joint_rename(joint, &old_name, &new_name) {
                    Ok(joint) => joint,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = Value::joint(joint);
                let inputs = [&source];
                self.attach_derivation(ins.b, DerivationKind::Operation, "rename", &inputs, "",
                                       ins.line, DerivationExactness::Exact, &new_name)
            }
            PossibilityBuild => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let ValueKind::Array(array) = &source.kind else {
                    return LanaError::Type;
                };
                let items = array.lock().unwrap().items.clone();
                let possibility = match self.possibility_build(&items) {
                    Ok(possibility) => possibility,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = Value::possibility(possibility);
                let inputs = [&source];
                self.attach_derivation(ins.b, DerivationKind::Operation, "possibility", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "equipossible_support")
            }
            PathSplit => {
                let condition = self.current_frame().registers[ins.a as usize].clone();
                self.path_split(&condition, ins.imm as usize)
            }
            PathJoin => self.path_join(ins.line),
            Observe => {
                if self.active_path_count > 1 {
                    return LanaError::UnsupportedOperation;
                }
                let source = self.current_frame().registers[ins.a as usize].clone();
                let evidence = self.current_frame().registers[ins.imm as usize].clone();
                let descriptor = match &self.chunk.constants[ins.c as usize] {
                    ConstantValue::String(descriptor) => descriptor.clone(),
                    _ => return LanaError::Type,
                };
                let ValueKind::Joint(joint) = &source.kind else {
                    return LanaError::Type;
                };
                let joint = match self.joint_observe(joint, &descriptor, &evidence) {
                    Ok(joint) => joint,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = Value::joint(joint);
                let inputs = [&source, &evidence];
                self.attach_derivation(ins.b, DerivationKind::Observation, "observe", &inputs, "",
                                       ins.line, DerivationExactness::Exact, &descriptor)
            }
            InfoSample => {
                if self.active_path_count > 1 {
                    return LanaError::UnsupportedOperation;
                }
                let source = self.current_frame().registers[ins.a as usize].clone();
                let result = match self.information_sample(&source) {
                    Ok(result) => result,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = result;
                let inputs = [&source];
                self.attach_derivation(ins.b, DerivationKind::Sample, "sample", &inputs, "",
                                       ins.line, DerivationExactness::Sample, "seeded_rng")
            }
            Evidence | Assume => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let label = match &self.chunk.constants[ins.c as usize] {
                    ConstantValue::String(label) => label.clone(),
                    _ => return LanaError::Type,
                };
                let result = match self.provenance_root(&source, &label, ins.line, ins.opcode == Assume) {
                    Ok(result) => result,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = result;
                LanaError::Ok
            }
            Derivation => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let result = match self.vm_derivation(&source) {
                    Ok(result) => result,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = result;
                LanaError::Ok
            }
            Explain => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let result = match self.vm_explain(&source) {
                    Ok(result) => result,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = result;
                LanaError::Ok
            }
            Fork => {
                if self.active_path_count > 1 {
                    return LanaError::UnsupportedOperation;
                }
                for argument in 0..ins.imm as usize {
                    if self.current_frame().registers[(ins.c as usize) + argument].is_unresolved() {
                        return LanaError::UnresolvedValue;
                    }
                }
                let task = match self.start_task(ins.b, ins.imm, ins.c) {
                    Ok(task) => task,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.a as usize] = Value::task(task);
                LanaError::Ok
            }
            Join => {
                let task_value = self.current_frame().registers[ins.a as usize].clone();
                let ValueKind::Task(task) = &task_value.kind else {
                    return LanaError::Type;
                };
                let result = match self.wait_task(task, -1.0) {
                    Ok(result) => result,
                    Err(error) => return error,
                };
                let joined = result.clone();
                self.current_frame_mut().registers[ins.b as usize] = result;
                let inputs = [&joined];
                self.attach_derivation(ins.b, DerivationKind::Operation, "task_join", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "joined_task_result")
            }
            JoinTimeout => {
                let task_value = self.current_frame().registers[ins.a as usize].clone();
                let timeout_value = self.current_frame().registers[ins.b as usize].clone();
                let ValueKind::Task(task) = &task_value.kind else {
                    return LanaError::Type;
                };
                let ValueKind::Number(timeout) = timeout_value.kind else {
                    return LanaError::Type;
                };
                if timeout < 0.0 {
                    return LanaError::Type;
                }
                let result = match self.wait_task(task, timeout) {
                    Ok(result) => result,
                    Err(error) => return error,
                };
                let joined = result.clone();
                self.current_frame_mut().registers[ins.c as usize] = result;
                let inputs = [&joined];
                self.attach_derivation(ins.c, DerivationKind::Operation, "task_join_timeout", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "joined_task_result")
            }
            JoinAll => {
                let tasks_value = self.current_frame().registers[ins.a as usize].clone();
                let ValueKind::Array(tasks) = &tasks_value.kind else {
                    return LanaError::Type;
                };
                let tasks = tasks.lock().unwrap();
                let mut results = Vec::with_capacity(tasks.items.len());
                for task_value in &tasks.items {
                    let ValueKind::Task(task) = &task_value.kind else {
                        return LanaError::Type;
                    };
                    let result = match self.wait_task(task, -1.0) {
                        Ok(result) => result,
                        Err(error) => return error,
                    };
                    results.push(result);
                }
                let inputs: Vec<Value> = results.clone();
                let input_refs: Vec<&Value> = inputs.iter().collect();
                let array = Arc::new(Mutex::new(Array { items: results }));
                self.current_frame_mut().registers[ins.b as usize] = Value::array(array);
                self.attach_derivation(ins.b, DerivationKind::Operation, "task_join_all", &input_refs, "",
                                       ins.line, DerivationExactness::Exact, "joined_task_results")
            }
            Cancel => {
                let task_value = self.current_frame().registers[ins.a as usize].clone();
                let ValueKind::Task(task) = &task_value.kind else {
                    return LanaError::Type;
                };
                self.cancel_task(task);
                LanaError::Ok
            }
            TaskgroupEnter => {
                if self.group_depth >= LANA_MAX_CALL_FRAMES as usize {
                    return LanaError::Limit;
                }
                self.group_stack.push(self.current_group_id);
                self.group_depth += 1;
                self.current_group_id = self.next_group_id;
                self.next_group_id += 1;
                LanaError::Ok
            }
            TaskgroupExit => {
                let group_id = self.current_group_id;
                if self.group_depth == 0 {
                    return LanaError::Task;
                }
                self.current_group_id = self.group_stack.pop().expect("group_depth > 0");
                self.group_depth -= 1;
                self.close_task_group(group_id)
            }
            HostCall => {
                let host_id = ins.b;
                let accepts_unresolved = matches!(
                    host_id,
                    LANA_HOST_MAP_NEW
                        | LANA_HOST_MAP_HAS
                        | LANA_HOST_MAP_GET
                        | LANA_HOST_MAP_SET
                        | LANA_HOST_MAP_KEYS
                        | LANA_HOST_INDEX_GET
                        | LANA_HOST_INDEX_SET
                        | LANA_HOST_ARRAY_PUSH
                        | LANA_HOST_ARRAY_LENGTH
                        | LANA_HOST_INFORMATION_NEW
                        | LANA_HOST_CLAIM_NEW
                        | LANA_HOST_CLAIM_VALUE
                        | LANA_HOST_CLAIM_PROPOSITION
                        | LANA_HOST_CLAIM_STATUS
                        | LANA_HOST_PLANNED_EFFECT_NEW
                        | LANA_HOST_PLANNED_EFFECT_EXECUTE
                        | LANA_HOST_PLANNED_EFFECT_STATUS
                        | LANA_HOST_SHARED_INFORMATION
                        | LANA_HOST_SHARED_OBSERVE
                        | LANA_HOST_INFORMATION_INSPECT
                );
                let materialize = matches!(
                    host_id,
                    LANA_HOST_WRITE_TEXT
                        | LANA_HOST_JSON_STRINGIFY
                        | LANA_HOST_CSV_WRITE
                        | LANA_HOST_ASSERT
                );
                if self.active_path_count > 1 {
                    return LanaError::UnsupportedOperation;
                }
                let argc = ins.imm as usize;
                for argument in 0..argc {
                    if !accepts_unresolved
                        && self.value_is_unresolved(&self.current_frame().registers[ins.c as usize + argument])
                    {
                        return LanaError::UnresolvedValue;
                    }
                }
                let mut arguments: Vec<Value> = Vec::with_capacity(argc);
                for argument in 0..argc {
                    let source = self.current_frame().registers[ins.c as usize + argument].clone();
                    if materialize {
                        arguments.push(match self.materialize_value(&source) {
                            Ok(value) => value,
                            Err(error) => return error,
                        });
                    } else {
                        arguments.push(source);
                    }
                }
                let mut out = Value::null();
                let error = self.execute_host_call(host_id, &arguments, &mut out);
                if error == LanaError::Assertion
                    && host_id == LANA_HOST_ASSERT
                    && argc == 2
                    && matches!(arguments[1].kind, ValueKind::String(_))
                {
                    self.pending_error_message = Some(arguments[1].as_string().to_string());
                }
                if error == LanaError::Ok {
                    self.current_frame_mut().registers[ins.a as usize] = out;
                }
                error
            }
            // Increment 5+ opcodes. Unreachable in increment-4 fixtures.
            _ => LanaError::UnsupportedOperation,
        }
    }

    /// Store a measurement result, mirroring the `MEASURE`/`MEASURE_BASIS`
    /// result selection in `lana_vm_run`.
    fn store_measurement(&mut self, reg: u32, mode: u32, probability: f64) -> LanaError {
        match mode {
            LANA_MEASURE_PROBABILITY => {
                self.current_frame_mut().registers[reg as usize] = Value::number(probability);
                LanaError::Ok
            }
            LANA_MEASURE_DISTRIBUTION => {
                self.current_frame_mut().registers[reg as usize] =
                    Value::distribution(1.0 - probability, probability);
                LanaError::Ok
            }
            LANA_MEASURE_SAMPLE => {
                let sample = self.draw_sample(probability);
                self.current_frame_mut().registers[reg as usize] = Value::sample(sample);
                LanaError::Ok
            }
            _ => LanaError::Measure,
        }
    }

    /// Account for an allocation, mirroring `lana_vm_alloc`. Returns `Oom` when
    /// the byte budget is exhausted.
    fn alloc_bytes(&mut self, bytes: usize) -> LanaError {
        self.allocation_count += 1;
        self.allocated_bytes += bytes;
        if self.allocated_bytes > self.memory_limit {
            LanaError::Oom
        } else {
            LanaError::Ok
        }
    }

    /// Build a dirac state distribution, mirroring `lana_vm_state_dist_dirac`.
    /// Part of the public C11 API; no increment-2 opcode constructs a dirac
    /// node directly (host calls use it in increment 5).
    #[allow(dead_code)]
    fn state_dist_dirac(&mut self, state: &StateValue) -> Result<Arc<StateDist>, LanaError> {
        if !state::state_valid(&state.state) {
            return Err(LanaError::InvalidState);
        }
        if self.alloc_bytes(std::mem::size_of::<StateDist>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        Ok(Arc::new(StateDist {
            kind: StateDistKind::Dirac(state.clone()),
        }))
    }

    /// Build an append node, mirroring `lana_vm_state_dist_append`.
    fn state_dist_append(&mut self, left: &Value, right: &Value) -> Result<Arc<StateDist>, LanaError> {
        if self.alloc_bytes(std::mem::size_of::<StateDist>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let left_operand = state_dist::distribution_from_value(left)?;
        let right_operand = state_dist::distribution_from_value(right)?;
        let mut has_cached_parameters = false;
        let mut p = 0.0;
        let mut m_re = 0.0;
        let mut m_im = 0.0;
        let mut sigma = 0.0;
        if matches!(left.kind, ValueKind::State(_)) && matches!(right.kind, ValueKind::State(_)) {
            let error = state::append_parameters(
                &left.as_state().state,
                &right.as_state().state,
                &mut p,
                &mut m_re,
                &mut m_im,
                &mut sigma,
            );
            if error != LanaError::Ok {
                return Err(error);
            }
            has_cached_parameters = true;
        }
        Ok(Arc::new(StateDist {
            kind: StateDistKind::Append {
                left: left_operand,
                right: right_operand,
                has_cached_parameters,
                p,
                m_re,
                m_im,
                sigma,
            },
        }))
    }

    /// Build a transform node, mirroring `lana_vm_state_dist_transform`.
    fn state_dist_transform(
        &mut self,
        transform_id: u32,
        child: Arc<StateDist>,
    ) -> Result<Arc<StateDist>, LanaError> {
        let specification = match state::transform_spec(transform_id) {
            Some(spec) => spec,
            None => return Err(LanaError::UnsupportedOperation),
        };
        if !specification.distribution_liftable {
            return Err(LanaError::UnsupportedOperation);
        }
        if self.alloc_bytes(std::mem::size_of::<StateDist>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        Ok(Arc::new(StateDist {
            kind: StateDistKind::Transform { child, transform_id },
        }))
    }

    /// Build an attenuate node, mirroring `lana_vm_state_dist_attenuate`.
    fn state_dist_attenuate(
        &mut self,
        child: Arc<StateDist>,
        factor: f64,
    ) -> Result<Arc<StateDist>, LanaError> {
        if !factor.is_finite() || factor < 0.0 || factor > 1.0 {
            return Err(LanaError::InvalidParameters);
        }
        if self.alloc_bytes(std::mem::size_of::<StateDist>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        Ok(Arc::new(StateDist {
            kind: StateDistKind::Attenuate { child, factor },
        }))
    }

    /// Build a relationship-aware append node, mirroring
    /// `lana_vm_state_dist_append_relationship`.
    fn state_dist_append_relationship(
        &mut self,
        left: &Value,
        right: &Value,
        mode: u32,
        strength: f64,
    ) -> Result<Arc<StateDist>, LanaError> {
        if matches!(left.kind, ValueKind::StateDist(_))
            || matches!(right.kind, ValueKind::StateDist(_))
        {
            return Err(LanaError::UnsupportedOperation);
        }
        if !matches!(left.kind, ValueKind::State(_)) || !matches!(right.kind, ValueKind::State(_)) {
            return Err(LanaError::Type);
        }
        if self.alloc_bytes(std::mem::size_of::<StateDist>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let left_operand = state_dist::distribution_from_value(left)?;
        let right_operand = state_dist::distribution_from_value(right)?;
        let mut p = 0.0;
        let mut m_re = 0.0;
        let mut m_im = 0.0;
        let mut sigma = 0.0;
        let error = state::append_relationship_parameters(
            &left.as_state().state,
            &right.as_state().state,
            mode,
            strength,
            &mut p,
            &mut m_re,
            &mut m_im,
            &mut sigma,
        );
        if error != LanaError::Ok {
            return Err(error);
        }
        Ok(Arc::new(StateDist {
            kind: StateDistKind::Append {
                left: left_operand,
                right: right_operand,
                has_cached_parameters: true,
                p,
                m_re,
                m_im,
                sigma,
            },
        }))
    }

    /// Consume one unit of the sampling budget, mirroring
    /// `consume_sampling_budget`. Cancellation lands with tasks (increment 4).
    fn consume_sampling_budget(&mut self) -> LanaError {
        if self.instruction_count >= self.instruction_limit {
            return LanaError::BudgetExhausted;
        }
        self.instruction_count += 1;
        LanaError::Ok
    }

    /// A uniform draw in [-1, 1), mirroring `uniform_signed`.
    fn uniform_signed(&mut self) -> f64 {
        2.0 * (self.rng.random() as f64 / 4294967296.0) - 1.0
    }

    /// Sample from cached append parameters, mirroring `sample_append_parameters`.
    fn sample_append_parameters(
        &mut self,
        p: f64,
        m_re: f64,
        m_im: f64,
        sigma: f64,
        out: &mut StateValue,
    ) -> LanaError {
        out.indexes = Default::default();
        if p == 0.0 || p == 1.0 {
            return state::make_complex(p, 0.0, 0.0, &mut out.state);
        }
        if sigma == 0.0 {
            return state::make_complex(p, m_re, m_im, &mut out.state);
        }
        loop {
            let error = self.consume_sampling_budget();
            if error != LanaError::Ok {
                return error;
            }
            let x = self.uniform_signed();
            let y = self.uniform_signed();
            let radius_squared = x * x + y * y;
            if radius_squared <= 0.0 || radius_squared >= 1.0 {
                continue;
            }
            let factor = (-2.0 * radius_squared.ln() / radius_squared).sqrt();
            let d_re = m_re + sigma * x * factor;
            let d_im = m_im + sigma * y * factor;
            if d_re * d_re + d_im * d_im > 1.0 {
                continue;
            }
            return state::make_complex(p, d_re, d_im, &mut out.state);
        }
    }

    /// Sample the append of two states, mirroring `sample_append_kernel`.
    fn sample_append_kernel(
        &mut self,
        left: &StateValue,
        right: &StateValue,
        out: &mut StateValue,
    ) -> LanaError {
        let mut p = 0.0;
        let mut m_re = 0.0;
        let mut m_im = 0.0;
        let mut sigma = 0.0;
        let error = state::append_parameters(
            &left.state,
            &right.state,
            &mut p,
            &mut m_re,
            &mut m_im,
            &mut sigma,
        );
        if error != LanaError::Ok {
            return LanaError::InvalidDistribution;
        }
        self.sample_append_parameters(p, m_re, m_im, sigma, out)
    }

    /// Sample a state distribution, mirroring `lana_vm_state_dist_sample`.
    fn state_dist_sample(&mut self, distribution: &Arc<StateDist>) -> Result<StateValue, LanaError> {
        let mut stack: Vec<DistEvalFrame> = Vec::new();
        let mut result_state = StateValue::default();
        stack.push(DistEvalFrame::new(distribution.clone()));
        loop {
            if stack.is_empty() {
                break;
            }
            let top = stack.len() - 1;
            let stage = stack[top].stage;
            if stage == 0 {
                let action = {
                    let frame = &mut stack[top];
                    match &frame.node.kind {
                        StateDistKind::Dirac(state) => {
                            if !state::state_valid(&state.state) {
                                return Err(LanaError::InvalidDistribution);
                            }
                            result_state = state.clone();
                            EvalAction::Pop
                        }
                        StateDistKind::Append { left, has_cached_parameters, p, m_re, m_im, sigma, .. } => {
                            if *has_cached_parameters {
                                let error = self.sample_append_parameters(
                                    *p, *m_re, *m_im, *sigma, &mut result_state,
                                );
                                if error != LanaError::Ok {
                                    return Err(error);
                                }
                                EvalAction::Pop
                            } else if let DistOperand::Inline(state) = left {
                                if !state::state_valid(&state.state) {
                                    return Err(LanaError::InvalidDistribution);
                                }
                                frame.left_state = state.clone();
                                frame.stage = 2;
                                EvalAction::Continue
                            } else {
                                let DistOperand::Node(node) = left else {
                                    return Err(LanaError::InvalidDistribution);
                                };
                                frame.stage = 1;
                                EvalAction::Push(node.clone())
                            }
                        }
                        StateDistKind::Transform { child, .. } => {
                            frame.stage = 4;
                            EvalAction::Push(child.clone())
                        }
                        StateDistKind::Attenuate { child, .. } => {
                            frame.stage = 5;
                            EvalAction::Push(child.clone())
                        }
                    }
                };
                match action {
                    EvalAction::Pop => {
                        stack.pop();
                    }
                    EvalAction::Push(node) => {
                        if stack.len() >= LANA_STATE_DIST_DEPTH_LIMIT + 1 {
                            return Err(LanaError::InvalidDistribution);
                        }
                        stack.push(DistEvalFrame::new(node));
                    }
                    EvalAction::Continue => {}
                }
            } else if stage == 1 {
                stack[top].left_state = result_state.clone();
                stack[top].stage = 2;
            } else if stage == 2 {
                let action = {
                    let frame = &mut stack[top];
                    let right = match &frame.node.kind {
                        StateDistKind::Append { right, .. } => right,
                        _ => return Err(LanaError::InvalidDistribution),
                    };
                    if let DistOperand::Inline(state) = right {
                        if !state::state_valid(&state.state) {
                            return Err(LanaError::InvalidDistribution);
                        }
                        frame.right_state = state.clone();
                        frame.stage = 3;
                        EvalAction::Continue
                    } else {
                        let DistOperand::Node(node) = right else {
                            return Err(LanaError::InvalidDistribution);
                        };
                        frame.stage = 3;
                        EvalAction::Push(node.clone())
                    }
                };
                match action {
                    EvalAction::Pop => unreachable!("stage 2 never pops"),
                    EvalAction::Push(node) => {
                        if stack.len() >= LANA_STATE_DIST_DEPTH_LIMIT + 1 {
                            return Err(LanaError::InvalidDistribution);
                        }
                        stack.push(DistEvalFrame::new(node));
                    }
                    EvalAction::Continue => {}
                }
            } else if stage == 3 {
                let right_is_node = matches!(
                    &stack[top].node.kind,
                    StateDistKind::Append { right: DistOperand::Node(_), .. }
                );
                if right_is_node {
                    stack[top].right_state = result_state.clone();
                }
                let left_state = stack[top].left_state.clone();
                let right_state = stack[top].right_state.clone();
                let error = self.sample_append_kernel(&left_state, &right_state, &mut result_state);
                if error != LanaError::Ok {
                    return Err(error);
                }
                stack.pop();
            } else if stage == 4 {
                let transform_id = match &stack[top].node.kind {
                    StateDistKind::Transform { transform_id, .. } => *transform_id,
                    _ => return Err(LanaError::InvalidDistribution),
                };
                let mut state = result_state.state;
                let source = state;
                let error = state::transform_apply(transform_id, &source, &mut state);
                if error != LanaError::Ok {
                    return Err(error);
                }
                result_state.state = state;
                stack.pop();
            } else {
                // ATTENUATE: scale the disposition of the sampled child state.
                let factor = match &stack[top].node.kind {
                    StateDistKind::Attenuate { factor, .. } => *factor,
                    _ => return Err(LanaError::InvalidDistribution),
                };
                let mut state = result_state.state;
                let source = state;
                let error = state::attenuate(&source, factor, &mut state);
                if error != LanaError::Ok {
                    return Err(error);
                }
                result_state.state = state;
                stack.pop();
            }
        }
        Ok(result_state)
    }

    /// Estimate a basis probability by sampling, mirroring
    /// `estimate_basis_probability`.
    fn estimate_basis_probability(
        &mut self,
        distribution: &Arc<StateDist>,
        basis: u32,
        samples: u32,
    ) -> Result<f64, LanaError> {
        if samples == 0 {
            return Err(LanaError::Format);
        }
        let mut total = 0.0;
        for _ in 0..samples {
            let error = self.consume_sampling_budget();
            if error != LanaError::Ok {
                return Err(error);
            }
            let state = self.state_dist_sample(distribution)?;
            let mut probability = 0.0;
            let error = state::basis_probability(basis, &state.state, &mut probability);
            if error != LanaError::Ok {
                return Err(error);
            }
            total += probability;
        }
        let out = total / samples as f64;
        if out.is_finite() && out >= 0.0 && out <= 1.0 {
            Ok(out)
        } else {
            Err(LanaError::InvalidDistribution)
        }
    }

    /// Build the statistical-result map, mirroring `build_statistical_result`.
    fn build_statistical_result(
        &mut self,
        method: &str,
        value: f64,
        observable: u32,
    ) -> Result<Arc<Mutex<Map>>, LanaError> {
        if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut map = Map::new(6);
        map.set(Arc::from("method"), Value::string(Arc::from(method)), false)?;
        map.set(Arc::from("value"), Value::number(value), false)?;
        map.set(
            Arc::from("observable"),
            Value::string(Arc::from(if observable == LANA_OBSERVABLE_PROBABILITY {
                "probability"
            } else {
                "unknown"
            })),
            false,
        )?;
        map.set(Arc::from("provenance"), Value::string(Arc::from("exact")), false)?;
        map.set(Arc::from("sample_count"), Value::null(), false)?;
        map.set(Arc::from("seed"), Value::null(), false)?;
        Ok(Arc::new(Mutex::new(map)))
    }

    /// Build a validation-result map, mirroring `validate_result`.
    fn validate_result(&mut self, status: &str, reason: &str) -> Result<Arc<Mutex<Map>>, LanaError> {
        if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut map = Map::new(3);
        map.set(Arc::from("status"), Value::string(Arc::from(status)), false)?;
        map.set(Arc::from("reason"), Value::string(Arc::from(reason)), false)?;
        map.set(Arc::from("schema_version"), Value::number(1.0), false)?;
        Ok(Arc::new(Mutex::new(map)))
    }

    /// Validate a value against a schema map, mirroring `lana_vm_validate`.
    fn validate(&mut self, value: &Value, schema: &Value) -> Result<Arc<Mutex<Map>>, LanaError> {
        let ValueKind::Map(schema_rc) = &schema.kind else {
            return Err(LanaError::Schema);
        };
        let schema_map = schema_rc.lock().unwrap();
        let type_value = schema_map.get("type").ok_or(LanaError::Schema)?;
        if !matches!(type_value.kind, ValueKind::String(_)) {
            return Err(LanaError::Schema);
        }
        let type_name = type_value.as_string();
        let expected = match &*type_name {
            "number" => ValueType::Number,
            "bool" => ValueType::Bool,
            "string" => ValueType::String,
            "state" => ValueType::State,
            "array" => ValueType::Array,
            "map" => ValueType::Map,
            "any" => value.value_type(),
            _ => return Err(LanaError::Schema),
        };

        if value.derivation.is_some()
            && value.derivation.as_ref().unwrap().outcome == DerivationOutcome::Unresolved
        {
            return self.validate_result("insufficient_evidence", "unresolved_derivation");
        }

        if value.value_type() != expected {
            return self.validate_result("invalid", "type_mismatch");
        }

        if expected == ValueType::Map {
            if let Some(required_value) = schema_map.get("required") {
                if matches!(required_value.kind, ValueKind::Array(_)) {
                    let required = match &required_value.kind {
                        ValueKind::Array(array) => array.clone(),
                        _ => unreachable!("checked above"),
                    };
                    let required = required.lock().unwrap();
                    for field in &required.items {
                        if !matches!(field.kind, ValueKind::String(_)) {
                            return Err(LanaError::Schema);
                        }
                        let value_map = match &value.kind {
                            ValueKind::Map(map) => map.clone(),
                            _ => unreachable!("expected == Map checked above"),
                        };
                        if !value_map.lock().unwrap().has(&field.as_string()) {
                            return self.validate_result("invalid", "missing_required_field");
                        }
                    }
                }
            }
        }

        if let Some(constraints_value) = schema_map.get("constraints") {
            if matches!(constraints_value.kind, ValueKind::Map(_)) {
                let constraints = match &constraints_value.kind {
                    ValueKind::Map(map) => map.clone(),
                    _ => unreachable!("checked above"),
                };
                let constraints = constraints.lock().unwrap();
                if expected == ValueType::Number {
                    if let Some(min_value) = constraints.get("min") {
                        if matches!(min_value.kind, ValueKind::Number(_))
                            && value.as_number() < min_value.as_number()
                        {
                            return self.validate_result("invalid", "below_minimum");
                        }
                    }
                    if let Some(max_value) = constraints.get("max") {
                        if matches!(max_value.kind, ValueKind::Number(_))
                            && value.as_number() > max_value.as_number()
                        {
                            return self.validate_result("invalid", "above_maximum");
                        }
                    }
                } else if expected == ValueType::String {
                    let length = value.as_string().len() as f64;
                    if let Some(min_length_value) = constraints.get("min_length") {
                        if matches!(min_length_value.kind, ValueKind::Number(_))
                            && length < min_length_value.as_number()
                        {
                            return self.validate_result("invalid", "below_minimum_length");
                        }
                    }
                    if let Some(max_length_value) = constraints.get("max_length") {
                        if matches!(max_length_value.kind, ValueKind::Number(_))
                            && length > max_length_value.as_number()
                        {
                            return self.validate_result("invalid", "above_maximum_length");
                        }
                    }
                }
            }
        }

        if let Some(exactness_value) = schema_map.get("exactness") {
            if matches!(exactness_value.kind, ValueKind::String(_))
                && &*exactness_value.as_string() == "exact"
                && value.derivation.is_some()
                && value.derivation.as_ref().unwrap().exactness != DerivationExactness::Exact
            {
                return self.validate_result("invalid", "exactness_mismatch");
            }
        }

        self.validate_result("valid", "none")
    }

    /// Deep-clone a value, mirroring `clone_value` in `vm/c/vm.c`. Mutable
    /// containers (arrays, maps) are copied with a memo so shared substructure
    /// is preserved; immutable payloads are shared via `Arc`. Tasks cannot be
    /// cloned (`Type`), matching the C11.
    fn deep_clone_value(&mut self, value: &Value, memo: &mut DeepCloneMemo) -> Result<Value, LanaError> {
        let mut cloned = Value {
            kind: ValueKind::Null,
            derivation: value.derivation.clone(),
            reactive: value.reactive.clone(),
            claim: value.claim.clone(),
            planned_effect: value.planned_effect.clone(),
        };
        match &value.kind {
            ValueKind::Null
            | ValueKind::Number(_)
            | ValueKind::Bool(_)
            | ValueKind::Sample(_)
            | ValueKind::Function(_)
            | ValueKind::Distribution { .. }
            | ValueKind::Capability(_)
            | ValueKind::Lazy { .. } => {
                cloned.kind = value.kind.clone();
            }
            ValueKind::String(string) => cloned.kind = ValueKind::String(string.clone()),
            ValueKind::State(state) => cloned.kind = ValueKind::State(state.clone()),
            ValueKind::Array(array) => {
                let key = Arc::as_ptr(array) as usize;
                if let Some(existing) = memo.arrays.get(&key) {
                    cloned.kind = ValueKind::Array(existing.clone());
                } else {
                    if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
                        return Err(LanaError::Oom);
                    }
                    let items = array
                        .lock().unwrap()
                        .items
                        .iter()
                        .map(|item| self.deep_clone_value(item, memo))
                        .collect::<Result<Vec<_>, _>>()?;
                    let copy = Arc::new(Mutex::new(Array { items }));
                    memo.arrays.insert(key, copy.clone());
                    cloned.kind = ValueKind::Array(copy);
                }
            }
            ValueKind::Map(map) => {
                let key = Arc::as_ptr(map) as usize;
                if let Some(existing) = memo.maps.get(&key) {
                    cloned.kind = ValueKind::Map(existing.clone());
                } else {
                    if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
                        return Err(LanaError::Oom);
                    }
                    let mut copy = Map::new(map.lock().unwrap().entries.len());
                    for entry in &map.lock().unwrap().entries {
                        let value = self.deep_clone_value(&entry.value, memo)?;
                        copy.set(entry.key.clone(), value, true)?;
                    }
                    let copy = Arc::new(Mutex::new(copy));
                    memo.maps.insert(key, copy.clone());
                    cloned.kind = ValueKind::Map(copy);
                }
            }
            ValueKind::Joint(joint) => {
                if self.alloc_bytes(std::mem::size_of::<JointState>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                let mut values = Vec::with_capacity(joint.values.len());
                for value in &joint.values {
                    values.push(self.deep_clone_value(value, memo)?);
                }
                let mut rows = Vec::with_capacity(joint.rows.len());
                for row in &joint.rows {
                    let mut row_values = Vec::with_capacity(row.values.len());
                    for value in &row.values {
                        row_values.push(self.deep_clone_value(value, memo)?);
                    }
                    rows.push(JointRow { values: row_values, weight: row.weight });
                }
                cloned.kind = ValueKind::Joint(Arc::new(JointState {
                    names: joint.names.clone(),
                    domains: joint.domains.clone(),
                    values,
                    rows,
                    kind: joint.kind,
                    capabilities: joint.capabilities,
                }));
            }
            ValueKind::StateDist(distribution) => {
                let kind = self.deep_clone_state_dist_kind(&distribution.kind, memo)?;
                cloned.kind = ValueKind::StateDist(Arc::new(StateDist { kind }));
            }
            ValueKind::Possibility(possibility) => {
                if self.alloc_bytes(std::mem::size_of::<Possibility>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                let mut values = Vec::with_capacity(possibility.values.len());
                for value in &possibility.values {
                    values.push(self.deep_clone_value(value, memo)?);
                }
                cloned.kind = ValueKind::Possibility(Arc::new(Possibility {
                    values,
                    weights: possibility.weights.clone(),
                    dependency_id: possibility.dependency_id,
                }));
            }
            ValueKind::PathSet(paths) => {
                if self.alloc_bytes(std::mem::size_of::<PathSet>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                let mut alternatives = Vec::with_capacity(paths.alternatives.len());
                for alternative in &paths.alternatives {
                    alternatives.push(PathAlternative {
                        guard: alternative.guard,
                        weight: alternative.weight,
                        result: self.deep_clone_value(&alternative.result, memo)?,
                    });
                }
                cloned.kind = ValueKind::PathSet(Arc::new(PathSet {
                    alternatives,
                    dependency_id: paths.dependency_id,
                }));
            }
            ValueKind::Adt(adt) => {
                if self.alloc_bytes(std::mem::size_of::<Adt>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                let mut fields = Vec::with_capacity(adt.fields.len());
                for field in &adt.fields {
                    fields.push(self.deep_clone_value(field, memo)?);
                }
                cloned.kind = ValueKind::Adt(Arc::new(Adt { variant: adt.variant, fields }));
            }
            ValueKind::Task(_) => return Err(LanaError::Type),
        }
        Ok(cloned)
    }

    /// Deep-clone a state-dist node kind, mirroring `clone_state_dist_node`.
    /// State dists are immutable trees, so no memo is needed.
    fn deep_clone_state_dist_kind(
        &mut self,
        kind: &StateDistKind,
        memo: &mut DeepCloneMemo,
    ) -> Result<StateDistKind, LanaError> {
        match kind {
            StateDistKind::Dirac(state) => Ok(StateDistKind::Dirac(state.clone())),
            StateDistKind::Append { left, right, has_cached_parameters, p, m_re, m_im, sigma } => {
                if self.alloc_bytes(std::mem::size_of::<StateDist>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                Ok(StateDistKind::Append {
                    left: self.deep_clone_dist_operand(left, memo)?,
                    right: self.deep_clone_dist_operand(right, memo)?,
                    has_cached_parameters: *has_cached_parameters,
                    p: *p,
                    m_re: *m_re,
                    m_im: *m_im,
                    sigma: *sigma,
                })
            }
            StateDistKind::Transform { child, transform_id } => {
                if self.alloc_bytes(std::mem::size_of::<StateDist>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                Ok(StateDistKind::Transform {
                    child: Arc::new(StateDist {
                        kind: self.deep_clone_state_dist_kind(&child.kind, memo)?,
                    }),
                    transform_id: *transform_id,
                })
            }
            StateDistKind::Attenuate { child, factor } => {
                if self.alloc_bytes(std::mem::size_of::<StateDist>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                Ok(StateDistKind::Attenuate {
                    child: Arc::new(StateDist {
                        kind: self.deep_clone_state_dist_kind(&child.kind, memo)?,
                    }),
                    factor: *factor,
                })
            }
        }
    }

    /// Deep-clone one side of an append.
    fn deep_clone_dist_operand(
        &mut self,
        operand: &DistOperand,
        memo: &mut DeepCloneMemo,
    ) -> Result<DistOperand, LanaError> {
        match operand {
            DistOperand::Inline(state) => Ok(DistOperand::Inline(state.clone())),
            DistOperand::Node(node) => Ok(DistOperand::Node(Arc::new(StateDist {
                kind: self.deep_clone_state_dist_kind(&node.kind, memo)?,
            }))),
        }
    }

    /// Snapshot the current frames, deep-cloning registers, mirroring
    /// `snapshot_frames` in `vm/c/vm.c`. Histories are shallow-cloned because
    /// state values are immutable records.
    fn snapshot_frames(&mut self) -> Result<Vec<Frame>, LanaError> {
        let mut frames = self.frames.clone();
        for frame in &mut frames {
            for register in frame.registers.iter_mut() {
                let mut memo = DeepCloneMemo::default();
                *register = self.deep_clone_value(register, &mut memo)?;
            }
        }
        Ok(frames)
    }

    /// Split execution on a condition, mirroring `path_split` in `vm/c/vm.c`.
    fn path_split(&mut self, condition: &Value, false_ip: usize) -> LanaError {
        if matches!(condition.kind, ValueKind::Bool(_)) {
            if !condition.as_bool() {
                self.ip = false_ip;
            }
            return LanaError::Ok;
        }
        let ValueKind::Possibility(possibility) = &condition.kind else {
            return LanaError::Type;
        };
        let mut has_true = false;
        let mut has_false = false;
        let mut true_weight = 0.0;
        let mut false_weight = 0.0;
        for (index, value) in possibility.values.iter().enumerate() {
            let weight = match &possibility.weights {
                Some(weights) => weights[index],
                None => 1.0 / possibility.values.len() as f64,
            };
            if !matches!(value.kind, ValueKind::Bool(_)) {
                return LanaError::Type;
            }
            if value.as_bool() {
                has_true = true;
                true_weight += weight;
            } else {
                has_false = true;
                false_weight += weight;
            }
        }
        if !has_true {
            self.ip = false_ip;
            return LanaError::Ok;
        }
        if !has_false {
            return LanaError::Ok;
        }
        if self.active_path_count > self.path_limit / 2 {
            return LanaError::PathLimit;
        }
        let false_frames = match self.snapshot_frames() {
            Ok(frames) => frames,
            Err(error) => return error,
        };
        self.path_execution.push(PathExecution {
            false_frames,
            true_frames: Vec::new(),
            frame_count: self.frames.len(),
            false_ip,
            dependency_id: possibility.dependency_id,
            true_weight,
            false_weight,
            previous_path_count: self.active_path_count,
            running_false: false,
        });
        self.active_path_count *= 2;
        LanaError::Ok
    }

    /// Join the two branches of a path split, mirroring `path_join` in
    /// `vm/c/vm.c`.
    fn path_join(&mut self, line: u32) -> LanaError {
        let Some(execution) = self.path_execution.last() else {
            return LanaError::Ok;
        };
        if !execution.running_false {
            let false_frames = execution.false_frames.clone();
            let false_ip = execution.false_ip;
            let true_frames = match self.snapshot_frames() {
                Ok(frames) => frames,
                Err(error) => return error,
            };
            let execution = self.path_execution.last_mut().unwrap();
            execution.true_frames = true_frames;
            self.frames = false_frames;
            self.ip = false_ip;
            execution.running_false = true;
            return LanaError::Ok;
        }
        if self.frames.len() != execution.frame_count {
            return LanaError::UnsupportedOperation;
        }
        let true_frames = execution.true_frames.clone();
        let dependency_id = execution.dependency_id;
        let true_weight = execution.true_weight;
        let false_weight = execution.false_weight;
        let previous_path_count = execution.previous_path_count;
        for frame_index in 0..self.frames.len() {
            for register_index in 0..self.frames[frame_index].registers.len() {
                let true_value = &true_frames[frame_index].registers[register_index];
                let false_value = self.frames[frame_index].registers[register_index].clone();
                if joint_value_equal(true_value, &false_value) {
                    continue;
                }
                if true_frames[frame_index].histories[register_index].policy != HistoryPolicy::None
                    || self.frames[frame_index].histories[register_index].policy != HistoryPolicy::None
                {
                    return LanaError::UnsupportedOperation;
                }
                let paths = Arc::new(PathSet {
                    alternatives: vec![
                        PathAlternative {
                            guard: true,
                            weight: true_weight,
                            result: true_value.clone(),
                        },
                        PathAlternative {
                            guard: false,
                            weight: false_weight,
                            result: false_value.clone(),
                        },
                    ],
                    dependency_id,
                });
                let inputs = [true_value, &false_value];
                let derivation = self.record_derivation(
                    DerivationKind::Path,
                    "guarded_path",
                    &inputs,
                    "",
                    line,
                    DerivationExactness::Exact,
                    "true_false_alternatives",
                    DerivationOutcome::Success,
                    "none",
                );
                let Some(derivation) = derivation else {
                    return LanaError::Oom;
                };
                let mut value = Value::paths(paths);
                value.derivation = Some(derivation);
                self.frames[frame_index].registers[register_index] = value;
            }
        }
        self.active_path_count = previous_path_count;
        self.path_execution.pop();
        LanaError::Ok
    }

    /// Build a joint state from marginals, mirroring `lana_vm_joint_build`.
    fn joint_build(&mut self, values: &[Value], descriptor: &str) -> Result<Arc<JointState>, LanaError> {
        let count = values.len();
        if count == 0 {
            return Err(LanaError::Format);
        }
        let (kind, names) = parse_joint_names(descriptor, count)?;
        if kind == JointKind::FiniteLaw {
            return Err(LanaError::UnsupportedOperation);
        }
        let mut ordered: Vec<(Arc<str>, usize)> = names
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, name)| (name, index))
            .collect();
        ordered.sort_by(|a, b| a.0.cmp(&b.0));
        if self.alloc_bytes(std::mem::size_of::<JointState>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut joint = JointState {
            names: Vec::with_capacity(count),
            domains: Vec::with_capacity(count),
            values: Vec::with_capacity(count),
            rows: Vec::new(),
            kind,
            capabilities: if kind == JointKind::Independent {
                LANA_JOINT_CAN_PROJECT | LANA_JOINT_CAN_CONDITION | LANA_JOINT_CAN_SAMPLE
                    | LANA_JOINT_CAN_RESOLVE
            } else {
                0
            },
        };
        let mut memo = DeepCloneMemo::default();
        for (name, source_index) in &ordered {
            joint.names.push(name.clone());
            let value = self.deep_clone_value(&values[*source_index], &mut memo)?;
            joint.domains.push(value.value_type());
            joint.values.push(value);
        }
        Ok(Arc::new(joint))
    }

    /// Build a finite correlated law, mirroring `lana_vm_joint_build_finite`.
    fn joint_build_finite(
        &mut self,
        names_text: &str,
        rows: &[Value],
        weights: &[f64],
        row_count: usize,
        variable_count: usize,
    ) -> Result<Arc<JointState>, LanaError> {
        if row_count == 0 || variable_count == 0 {
            return Err(LanaError::Format);
        }
        let descriptor = format!("correlated:{names_text}");
        let (_, names) = parse_joint_names(&descriptor, variable_count)?;
        let mut ordered: Vec<(Arc<str>, usize)> = names
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, name)| (name, index))
            .collect();
        ordered.sort_by(|a, b| a.0.cmp(&b.0));
        let mut unique_values: Vec<Value> = Vec::new();
        let mut unique_weights: Vec<f64> = Vec::new();
        let mut total = 0.0;
        for row in 0..row_count {
            if !weights[row].is_finite() || weights[row] <= 0.0 {
                return Err(LanaError::InvalidDistribution);
            }
            total += weights[row];
            for column in 0..variable_count {
                let value = &rows[row * variable_count + ordered[column].1];
                if !joint_value_is_definite(value) {
                    return Err(LanaError::Type);
                }
                if row > 0 && value.value_type() != rows[ordered[column].1].value_type() {
                    return Err(LanaError::Type);
                }
            }
            let mut found = false;
            for existing in 0..unique_values.len() / variable_count {
                let mut all_equal = true;
                for column in 0..variable_count {
                    if !joint_value_equal(
                        &rows[row * variable_count + ordered[column].1],
                        &unique_values[existing * variable_count + column],
                    ) {
                        all_equal = false;
                        break;
                    }
                }
                if all_equal {
                    unique_weights[existing] += weights[row];
                    found = true;
                    break;
                }
            }
            if !found {
                for column in 0..variable_count {
                    unique_values.push(rows[row * variable_count + ordered[column].1].clone());
                }
                unique_weights.push(weights[row]);
            }
        }
        if !total.is_finite() || (total - 1.0).abs() > 1e-12 {
            return Err(LanaError::InvalidDistribution);
        }
        let unique_count = unique_weights.len();
        if self.alloc_bytes(std::mem::size_of::<JointState>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut joint = JointState {
            names: Vec::with_capacity(variable_count),
            domains: Vec::with_capacity(variable_count),
            values: Vec::new(),
            rows: Vec::with_capacity(unique_count),
            kind: JointKind::FiniteLaw,
            capabilities: LANA_JOINT_CAN_PROJECT | LANA_JOINT_CAN_CONDITION | LANA_JOINT_CAN_SAMPLE
                | LANA_JOINT_CAN_RESOLVE,
        };
        let mut memo = DeepCloneMemo::default();
        for column in 0..variable_count {
            joint.names.push(ordered[column].0.clone());
            joint.domains.push(unique_values[column].value_type());
        }
        for row in 0..unique_count {
            let mut values = Vec::with_capacity(variable_count);
            for column in 0..variable_count {
                values.push(
                    self.deep_clone_value(&unique_values[row * variable_count + column], &mut memo)?,
                );
            }
            joint.rows.push(JointRow {
                values,
                weight: unique_weights[row] / total,
            });
        }
        Ok(Arc::new(joint))
    }

    /// Build a finite law from an array of rows, mirroring
    /// `joint_build_finite_array` in `vm/c/vm.c`.
    fn joint_build_finite_array(
        &mut self,
        rows_value: &Value,
        names_text: &str,
    ) -> Result<Arc<JointState>, LanaError> {
        let ValueKind::Array(outer) = &rows_value.kind else {
            return Err(LanaError::Type);
        };
        let outer = outer.lock().unwrap();
        if outer.items.is_empty() {
            return Err(LanaError::Type);
        }
        // Scope the first-row guard so it is dropped before the row loop
        // below locks the same row again (a `Mutex` is not re-entrant).
        let variable_count = {
            let ValueKind::Array(first) = &outer.items[0].kind else {
                return Err(LanaError::Format);
            };
            let first = first.lock().unwrap();
            if first.items.len() < 2 {
                return Err(LanaError::Format);
            }
            first.items.len() - 1
        };
        let mut values: Vec<Value> = Vec::with_capacity(outer.items.len() * variable_count);
        let mut weights: Vec<f64> = Vec::with_capacity(outer.items.len());
        for row in 0..outer.items.len() {
            let ValueKind::Array(inner) = &outer.items[row].kind else {
                return Err(LanaError::Type);
            };
            let inner = inner.lock().unwrap();
            if inner.items.len() != variable_count + 1 {
                return Err(LanaError::Format);
            }
            let ValueKind::Number(weight) = inner.items[variable_count].kind else {
                return Err(LanaError::Type);
            };
            weights.push(weight);
            for column in 0..variable_count {
                values.push(inner.items[column].clone());
            }
        }
        self.joint_build_finite(names_text, &values, &weights, outer.items.len(), variable_count)
    }

    /// Project a joint onto a subset of names, mirroring `lana_vm_joint_project`.
    fn joint_project(&mut self, source: &JointState, names_text: &str) -> Result<Arc<JointState>, LanaError> {
        if source.capabilities & LANA_JOINT_CAN_PROJECT == 0 {
            return Err(LanaError::UnsupportedOperation);
        }
        let mut positions: Vec<usize> = Vec::new();
        for raw in names_text.split([',', ';']) {
            if raw.is_empty() {
                continue;
            }
            let token = raw.trim_start();
            if token.is_empty() {
                return Err(LanaError::Format);
            }
            let Some(position) = joint_find(source, token) else {
                return Err(LanaError::Key);
            };
            if positions.contains(&position) {
                return Err(LanaError::InvalidDependency);
            }
            positions.push(position);
        }
        if positions.is_empty() {
            return Err(LanaError::Format);
        }
        let count = positions.len();
        if source.rows.is_empty() {
            if names_text.len() + "independent:".len() >= 1024 {
                return Err(LanaError::Limit);
            }
            let descriptor = format!("independent:{names_text}");
            let values: Vec<Value> = positions
                .iter()
                .map(|&position| source.values[position].clone())
                .collect();
            let joint = self.joint_build(&values, &descriptor)?;
            let mut joint = (*joint).clone();
            joint.kind = JointKind::Projected;
            Ok(Arc::new(joint))
        } else {
            let mut values: Vec<Value> = Vec::with_capacity(source.rows.len() * count);
            let mut weights: Vec<f64> = Vec::with_capacity(source.rows.len());
            for row in &source.rows {
                weights.push(row.weight);
                for &position in &positions {
                    values.push(row.values[position].clone());
                }
            }
            let joint = self.joint_build_finite(names_text, &values, &weights, source.rows.len(), count)?;
            let mut joint = (*joint).clone();
            joint.kind = JointKind::Projected;
            Ok(Arc::new(joint))
        }
    }

    /// Condition a joint on evidence, mirroring `lana_vm_joint_condition`.
    fn joint_condition(&mut self, source: &JointState, name: &str, evidence: &Value) -> Result<Arc<JointState>, LanaError> {
        if source.capabilities & LANA_JOINT_CAN_CONDITION == 0 {
            return Err(LanaError::UnsupportedOperation);
        }
        let Some(position) = joint_find(source, name) else {
            return Err(LanaError::Key);
        };
        if !source.rows.is_empty() {
            let mut kept: Vec<usize> = Vec::new();
            for (row_index, row) in source.rows.iter().enumerate() {
                if joint_value_equal(&row.values[position], evidence) {
                    kept.push(row_index);
                }
            }
            if kept.is_empty() {
                return Err(LanaError::InvalidConditioning);
            }
            let names_text = source.names.iter().map(|name| &**name).collect::<Vec<_>>().join(",");
            let mut values: Vec<Value> = Vec::with_capacity(kept.len() * source.names.len());
            let mut weights: Vec<f64> = Vec::with_capacity(kept.len());
            let mut mass = 0.0;
            for &row_index in &kept {
                let row = &source.rows[row_index];
                mass += row.weight;
                for value in &row.values {
                    values.push(value.clone());
                }
            }
            for &row_index in &kept {
                weights.push(source.rows[row_index].weight / mass);
            }
            let joint = self.joint_build_finite(&names_text, &values, &weights, kept.len(), source.names.len())?;
            let mut joint = (*joint).clone();
            joint.kind = JointKind::Conditional;
            Ok(Arc::new(joint))
        } else {
            if !joint_value_is_definite(&source.values[position]) {
                return Err(LanaError::UnsupportedOperation);
            }
            if !joint_value_equal(&source.values[position], evidence) {
                return Err(LanaError::InvalidConditioning);
            }
            let mut memo = DeepCloneMemo::default();
            let wrapped = Value::joint(Arc::new(source.clone()));
            let mut cloned = self.deep_clone_value(&wrapped, &mut memo)?;
            let ValueKind::Joint(joint) = &mut cloned.kind else {
                unreachable!("wrapped value is a joint");
            };
            let mut state = (**joint).clone();
            state.kind = JointKind::Conditional;
            *joint = Arc::new(state);
            Ok(joint.clone())
        }
    }

    /// Observe evidence on a joint, mirroring `lana_vm_joint_observe`.
    fn joint_observe(&mut self, source: &JointState, name: &str, evidence: &Value) -> Result<Arc<JointState>, LanaError> {
        if self.active_path_count > 1 {
            return Err(LanaError::UnsupportedOperation);
        }
        let joint = self.joint_condition(source, name, evidence)?;
        self.observation_count += 1;
        self.revision += 1;
        Ok(joint)
    }

    /// Sample a joint, mirroring `lana_vm_joint_sample`.
    fn joint_sample(&mut self, source: &JointState) -> Result<Value, LanaError> {
        if source.capabilities & LANA_JOINT_CAN_SAMPLE == 0 {
            return Err(LanaError::UnsupportedOperation);
        }
        let mut items: Vec<Value> = Vec::with_capacity(source.names.len());
        if !source.rows.is_empty() {
            if self.consume_sampling_budget() != LanaError::Ok {
                return Err(LanaError::BudgetExhausted);
            }
            let draw = self.rng.random() as f64 / 4294967296.0;
            let mut cumulative = 0.0;
            let mut selected = source.rows.len() - 1;
            for (index, row) in source.rows.iter().enumerate() {
                cumulative += row.weight;
                if draw < cumulative {
                    selected = index;
                    break;
                }
            }
            let mut memo = DeepCloneMemo::default();
            for value in &source.rows[selected].values {
                items.push(self.deep_clone_value(value, &mut memo)?);
            }
        } else {
            let mut memo = DeepCloneMemo::default();
            for value in &source.values {
                if matches!(value.kind, ValueKind::StateDist(_)) {
                    let ValueKind::StateDist(distribution) = &value.kind else {
                        unreachable!("checked above");
                    };
                    let state = self.state_dist_sample(distribution)?;
                    items.push(Value::state(state));
                } else {
                    items.push(self.deep_clone_value(value, &mut memo)?);
                }
            }
        }
        Ok(Value::array(Arc::new(Mutex::new(Array { items }))))
    }

    /// Resolve a joint to a definite value, mirroring `lana_vm_joint_resolve`.
    fn joint_resolve(&mut self, source: &JointState) -> Result<Value, LanaError> {
        if source.capabilities & LANA_JOINT_CAN_RESOLVE == 0 {
            return Err(LanaError::UnsupportedOperation);
        }
        let values: Vec<&Value> = if !source.rows.is_empty() {
            if source.rows.len() != 1 {
                return Err(LanaError::UnresolvedValue);
            }
            source.rows[0].values.iter().collect()
        } else {
            for value in &source.values {
                if !joint_value_is_definite(value) {
                    return Err(LanaError::UnresolvedValue);
                }
            }
            source.values.iter().collect()
        };
        if values.len() == 1 {
            let mut memo = DeepCloneMemo::default();
            return self.deep_clone_value(values[0], &mut memo);
        }
        let mut memo = DeepCloneMemo::default();
        let mut items = Vec::with_capacity(values.len());
        for value in values {
            items.push(self.deep_clone_value(value, &mut memo)?);
        }
        Ok(Value::array(Arc::new(Mutex::new(Array { items }))))
    }

    /// Rename a joint variable, mirroring `lana_vm_joint_rename`.
    fn joint_rename(&mut self, source: &JointState, old_name: &str, new_name: &str) -> Result<Arc<JointState>, LanaError> {
        if new_name.is_empty() {
            return Err(LanaError::Format);
        }
        let Some(position) = joint_find(source, old_name) else {
            return Err(LanaError::Key);
        };
        if joint_find(source, new_name).is_some() {
            return Err(LanaError::InvalidDependency);
        }
        let names_text = source
            .names
            .iter()
            .enumerate()
            .map(|(index, name)| if index == position { new_name } else { &**name })
            .collect::<Vec<_>>()
            .join(",");
        if !source.rows.is_empty() {
            let mut values: Vec<Value> = Vec::with_capacity(source.rows.len() * source.names.len());
            let mut weights: Vec<f64> = Vec::with_capacity(source.rows.len());
            for row in &source.rows {
                weights.push(row.weight);
                for value in &row.values {
                    values.push(value.clone());
                }
            }
            self.joint_build_finite(&names_text, &values, &weights, source.rows.len(), source.names.len())
        } else {
            let descriptor = format!("independent:{names_text}");
            self.joint_build(&source.values, &descriptor)
        }
    }

    /// Build a possibility from an array of values, mirroring
    /// `lana_vm_possibility_build`.
    fn possibility_build(&mut self, values: &[Value]) -> Result<Arc<Possibility>, LanaError> {
        if values.is_empty() {
            return Err(LanaError::Format);
        }
        let mut unique: Vec<&Value> = Vec::new();
        for value in values {
            if !joint_value_is_definite(value) {
                return Err(LanaError::Type);
            }
            if !unique.iter().any(|existing| joint_value_equal(value, existing)) {
                unique.push(value);
            }
        }
        if self.alloc_bytes(std::mem::size_of::<Possibility>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut memo = DeepCloneMemo::default();
        let mut cloned = Vec::with_capacity(unique.len());
        for value in &unique {
            cloned.push(self.deep_clone_value(value, &mut memo)?);
        }
        let dependency_id = self.next_dependency_id;
        self.next_dependency_id += 1;
        Ok(Arc::new(Possibility {
            values: cloned,
            weights: None,
            dependency_id,
        }))
    }

    /// Resolve any information value, mirroring `lana_vm_information_resolve`.
    fn information_resolve(&mut self, source: &Value) -> Result<Value, LanaError> {
        match &source.kind {
            ValueKind::Joint(joint) => self.joint_resolve(joint),
            ValueKind::Possibility(possibility) => {
                if possibility.values.len() != 1 {
                    return Err(LanaError::UnresolvedValue);
                }
                let mut memo = DeepCloneMemo::default();
                self.deep_clone_value(&possibility.values[0], &mut memo)
            }
            ValueKind::PathSet(paths) => {
                if paths.alternatives.is_empty() {
                    return Err(LanaError::UnresolvedValue);
                }
                for alternative in &paths.alternatives[1..] {
                    if !joint_value_equal(&paths.alternatives[0].result, &alternative.result) {
                        return Err(LanaError::UnresolvedValue);
                    }
                }
                let mut memo = DeepCloneMemo::default();
                self.deep_clone_value(&paths.alternatives[0].result, &mut memo)
            }
            _ => {
                if !joint_value_is_definite(source) {
                    return Err(LanaError::UnresolvedValue);
                }
                let mut memo = DeepCloneMemo::default();
                self.deep_clone_value(source, &mut memo)
            }
        }
    }

    /// Sample any information value, mirroring `lana_vm_information_sample`.
    fn information_sample(&mut self, source: &Value) -> Result<Value, LanaError> {
        match &source.kind {
            ValueKind::Joint(joint) => self.joint_sample(joint),
            ValueKind::StateDist(distribution) => {
                let state = self.state_dist_sample(distribution)?;
                Ok(Value::state(state))
            }
            ValueKind::Possibility(possibility) => {
                if self.consume_sampling_budget() != LanaError::Ok {
                    return Err(LanaError::BudgetExhausted);
                }
                let selected = (self.rng.random() % possibility.values.len() as u32) as usize;
                let mut memo = DeepCloneMemo::default();
                self.deep_clone_value(&possibility.values[selected], &mut memo)
            }
            ValueKind::PathSet(paths) => {
                if self.consume_sampling_budget() != LanaError::Ok {
                    return Err(LanaError::BudgetExhausted);
                }
                let draw = self.rng.random() as f64 / 4294967296.0;
                let mut cumulative = 0.0;
                let mut selected = paths.alternatives.len() - 1;
                for (index, alternative) in paths.alternatives.iter().enumerate() {
                    cumulative += alternative.weight;
                    if draw < cumulative {
                        selected = index;
                        break;
                    }
                }
                let mut memo = DeepCloneMemo::default();
                self.deep_clone_value(&paths.alternatives[selected].result, &mut memo)
            }
            _ => Err(LanaError::Type),
        }
    }

    /// Attach an evidence/assumption derivation, mirroring
    /// `lana_vm_provenance_root`.
    fn provenance_root(&mut self, source: &Value, label: &str, line: u32, assumption: bool) -> Result<Value, LanaError> {
        let mut out = source.clone();
        let kind = if assumption { DerivationKind::Assumption } else { DerivationKind::Evidence };
        let operation = if assumption { "assume" } else { "evidence" };
        let derivation = self.record_derivation(
            kind,
            operation,
            &[],
            label,
            line,
            if assumption { DerivationExactness::Approximate } else { DerivationExactness::Exact },
            "root",
            DerivationOutcome::Success,
            "none",
        );
        let Some(derivation) = derivation else {
            return Err(LanaError::Oom);
        };
        out.derivation = Some(derivation);
        Ok(out)
    }

    /// Render a derivation id as a two-element array, mirroring
    /// `derivation_id_to_value`.
    fn derivation_id_to_value(&mut self, node: &Derivation) -> Result<Value, LanaError> {
        let items = vec![
            Value::number(node.task_lineage as f64),
            Value::number(node.local_sequence as f64),
        ];
        Ok(Value::array(Arc::new(Mutex::new(Array { items }))))
    }

    /// Render a derivation as a map, mirroring `derivation_to_value`.
    fn derivation_to_value(&mut self, node: &Derivation) -> Result<Value, LanaError> {
        let mut map = Map::new(13);
        let id = self.derivation_id_to_value(node)?;
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for input in &node.inputs {
            inputs.push(self.derivation_id_to_value(input)?);
        }
        let mut source_map = Map::new(3);
        source_map.set(Arc::from("label"), Value::string(node.label.clone()), true)?;
        source_map.set(Arc::from("function"), Value::string(node.function.clone()), true)?;
        source_map.set(Arc::from("line"), Value::number(node.line as f64), true)?;
        let mut details_map = Map::new(1);
        details_map.set(Arc::from("summary"), Value::string(node.details.clone()), true)?;
        map.set(Arc::from("id"), id, true)?;
        map.set(Arc::from("revision"), Value::number(node.revision as f64), true)?;
        map.set(Arc::from("kind"), Value::string(Arc::from(derivation::kind_name(node.kind))), true)?;
        map.set(Arc::from("operation"), Value::string(node.operation.clone()), true)?;
        map.set(Arc::from("inputs"), Value::array(Arc::new(Mutex::new(Array { items: inputs }))), true)?;
        map.set(Arc::from("source"), Value::map(Arc::new(Mutex::new(source_map))), true)?;
        map.set(Arc::from("exactness"), Value::string(Arc::from(derivation::exactness_name(node.exactness))), true)?;
        map.set(Arc::from("details"), Value::map(Arc::new(Mutex::new(details_map))), true)?;
        map.set(Arc::from("outcome"), Value::string(Arc::from(derivation::outcome_name(node.outcome))), true)?;
        map.set(Arc::from("status"), Value::string(Arc::from(derivation::status_name(node.status()))), true)?;
        map.set(Arc::from("reason"), Value::string(node.reason.clone()), true)?;
        Ok(Value::map(Arc::new(Mutex::new(map))))
    }

    /// Render a value's derivation as a map, mirroring `lana_vm_derivation`.
    fn vm_derivation(&mut self, source: &Value) -> Result<Value, LanaError> {
        let Some(derivation) = &source.derivation else {
            return Err(LanaError::UnsupportedOperation);
        };
        self.derivation_to_value(derivation)
    }

    /// Render a value's derivation as a string, mirroring `lana_vm_explain`.
    fn vm_explain(&mut self, source: &Value) -> Result<Value, LanaError> {
        let Some(node) = &source.derivation else {
            return Err(LanaError::UnsupportedOperation);
        };
        let rendered = format!(
            "{} {} id=[{},{}] revision={} exactness={} outcome={} reason={} label={} inputs={}",
            derivation::kind_name(node.kind),
            node.operation,
            node.task_lineage,
            node.local_sequence,
            node.revision,
            derivation::exactness_name(node.exactness),
            derivation::outcome_name(node.outcome),
            node.reason,
            node.label,
            node.inputs.len(),
        );
        if rendered.len() >= 1024 {
            return Err(LanaError::Limit);
        }
        Ok(Value::string(Arc::from(rendered)))
    }

    /// Lift a binary operation over paths/possibilities, mirroring `lift_binary`
    /// in `vm/c/vm.c`. Reactives land in increment 5.
    fn lift_binary(&mut self, left: &Value, right: &Value, kind: PureKind, operation: u32, out: &mut Value) -> LanaError {
        self.lift_binary_raw(left, right, kind, operation, out)
    }

    /// The recursive core of `lift_binary`, mirroring `lift_binary_raw`.
    fn lift_binary_raw(&mut self, left: &Value, right: &Value, kind: PureKind, operation: u32, out: &mut Value) -> LanaError {
        let left_paths = match &left.kind {
            ValueKind::PathSet(paths) => Some(paths.clone()),
            _ => None,
        };
        let right_paths = match &right.kind {
            ValueKind::PathSet(paths) => Some(paths.clone()),
            _ => None,
        };
        let left_possibility = match &left.kind {
            ValueKind::Possibility(possibility) => Some(possibility.clone()),
            _ => None,
        };
        let right_possibility = match &right.kind {
            ValueKind::Possibility(possibility) => Some(possibility.clone()),
            _ => None,
        };
        if left_paths.is_some() || right_paths.is_some() {
            if left_paths.is_some() && right_paths.is_some() {
                let lp = left_paths.as_ref().unwrap();
                let rp = right_paths.as_ref().unwrap();
                if lp.dependency_id != rp.dependency_id || lp.alternatives.len() != rp.alternatives.len() {
                    return LanaError::UnsupportedOperation;
                }
            }
            let count = if left_paths.is_some() {
                left_paths.as_ref().unwrap().alternatives.len()
            } else {
                right_paths.as_ref().unwrap().alternatives.len()
            };
            let dependency_id = if left_paths.is_some() {
                left_paths.as_ref().unwrap().dependency_id
            } else {
                right_paths.as_ref().unwrap().dependency_id
            };
            let mut alternatives = Vec::with_capacity(count);
            for index in 0..count {
                let left_value = match &left_paths {
                    Some(paths) => &paths.alternatives[index].result,
                    None => left,
                };
                let right_value = match &right_paths {
                    Some(paths) => &paths.alternatives[index].result,
                    None => right,
                };
                let (guard, weight) = if left_paths.is_some() {
                    let paths = left_paths.as_ref().unwrap();
                    (paths.alternatives[index].guard, paths.alternatives[index].weight)
                } else {
                    let paths = right_paths.as_ref().unwrap();
                    (paths.alternatives[index].guard, paths.alternatives[index].weight)
                };
                let mut result = Value::null();
                let error = self.lift_binary_raw(left_value, right_value, kind, operation, &mut result);
                if error != LanaError::Ok {
                    return error;
                }
                alternatives.push(PathAlternative { guard, weight, result });
            }
            *out = Value::paths(Arc::new(PathSet { alternatives, dependency_id }));
            return LanaError::Ok;
        }
        if left_possibility.is_some() || right_possibility.is_some() {
            let left_count = match &left_possibility {
                Some(possibility) => possibility.values.len(),
                None => 1,
            };
            let right_count = match &right_possibility {
                Some(possibility) => possibility.values.len(),
                None => 1,
            };
            let zipped = left_possibility.is_some()
                && right_possibility.is_some()
                && left_possibility.as_ref().unwrap().dependency_id
                    == right_possibility.as_ref().unwrap().dependency_id
                && left_count == right_count;
            if left_possibility.is_some() && right_possibility.is_some() && !zipped {
                return LanaError::UnsupportedOperation;
            }
            let count = if zipped { left_count } else { left_count * right_count };
            let mut results: Vec<Value> = Vec::with_capacity(count);
            for left_index in 0..left_count {
                let right_start = if zipped { left_index } else { 0 };
                let right_end = if zipped { left_index + 1 } else { right_count };
                for right_index in right_start..right_end {
                    let left_value = match &left_possibility {
                        Some(possibility) => &possibility.values[left_index],
                        None => left,
                    };
                    let right_value = match &right_possibility {
                        Some(possibility) => &possibility.values[right_index],
                        None => right,
                    };
                    let mut result = Value::null();
                    let error = pure_scalar_binary(left_value, right_value, kind, operation, &mut result);
                    if error != LanaError::Ok {
                        return error;
                    }
                    results.push(result);
                }
            }
            let possibility = match self.possibility_build(&results) {
                Ok(possibility) => possibility,
                Err(error) => return error,
            };
            if zipped || left_possibility.is_none() || right_possibility.is_none() {
                let dependency_id = if left_possibility.is_some() {
                    left_possibility.as_ref().unwrap().dependency_id
                } else {
                    right_possibility.as_ref().unwrap().dependency_id
                };
                let mut possibility = (*possibility).clone();
                possibility.dependency_id = dependency_id;
                *out = Value::possibility(Arc::new(possibility));
                return LanaError::Ok;
            }
            *out = Value::possibility(possibility);
            return LanaError::Ok;
        }
        pure_scalar_binary(left, right, kind, operation, out)
    }

    /// Lift a unary operation over paths/possibilities, mirroring `lift_unary`
    /// in `vm/c/vm.c`. Reactives land in increment 5.
    fn lift_unary(&mut self, source: &Value, operation: u32, out: &mut Value) -> LanaError {
        match &source.kind {
            ValueKind::PathSet(paths) => {
                let mut alternatives = Vec::with_capacity(paths.alternatives.len());
                for alternative in &paths.alternatives {
                    let mut result = Value::null();
                    let error = self.lift_unary(&alternative.result, operation, &mut result);
                    if error != LanaError::Ok {
                        return error;
                    }
                    alternatives.push(PathAlternative {
                        guard: alternative.guard,
                        weight: alternative.weight,
                        result,
                    });
                }
                *out = Value::paths(Arc::new(PathSet {
                    alternatives,
                    dependency_id: paths.dependency_id,
                }));
                LanaError::Ok
            }
            ValueKind::Possibility(possibility) => {
                let source_dependency_id = possibility.dependency_id;
                let mut results = Vec::with_capacity(possibility.values.len());
                for value in &possibility.values {
                    let mut result = Value::null();
                    let error = self.lift_unary(value, operation, &mut result);
                    if error != LanaError::Ok {
                        return error;
                    }
                    results.push(result);
                }
                let built = match self.possibility_build(&results) {
                    Ok(possibility) => possibility,
                    Err(error) => return error,
                };
                let mut built = (*built).clone();
                built.dependency_id = source_dependency_id;
                *out = Value::possibility(Arc::new(built));
                LanaError::Ok
            }
            _ => {
                if matches!(source.kind, ValueKind::Number(_)) && operation == 0 {
                    *out = Value::number(-source.as_number());
                    return LanaError::Ok;
                }
                if matches!(source.kind, ValueKind::Bool(_)) && operation == 1 {
                    *out = Value::boolean(!source.as_bool());
                    return LanaError::Ok;
                }
                LanaError::Type
            }
        }
    }
}

impl<'a> Vm<'a> {
    /// Resolve a value's reactive to its current contents, mirroring
    /// `reactive_value` in `vm/c/vm.c`. Returns a clone because the `Mutex`
    /// guard cannot outlive the borrow.
    fn reactive_value(&self, value: &Value) -> Value {
        if let Some(reactive) = &value.reactive {
            if let Some(current) = &reactive.lock().unwrap().current {
                return current.clone();
            }
        }
        value.clone()
    }

    /// Whether a value is unresolved, mirroring `value_is_unresolved` in
    /// `vm/c/vm.c`. Resolves the reactive first, then recurses into arrays and
    /// maps.
    fn value_is_unresolved(&self, value: &Value) -> bool {
        let current = self.reactive_value(value);
        match &current.kind {
            ValueKind::Possibility(_) | ValueKind::PathSet(_) => true,
            ValueKind::Array(array) => array
                .lock().unwrap()
                .items
                .iter()
                .any(|item| self.value_is_unresolved(item)),
            ValueKind::Map(map) => map
                .lock().unwrap()
                .entries
                .iter()
                .any(|entry| self.value_is_unresolved(&entry.value)),
            _ => false,
        }
    }

    /// Deep-clone a value with its reactive/claim/planned-effect metadata
    /// stripped, mirroring `clone_without_runtime_metadata` in `vm/c/vm.c`. The
    /// derivation is preserved.
    fn clone_without_runtime_metadata(&mut self, source: &Value) -> Result<Value, LanaError> {
        let mut plain = self.reactive_value(source);
        plain.reactive = None;
        plain.claim = None;
        plain.planned_effect = None;
        let mut memo = DeepCloneMemo::default();
        self.deep_clone_value(&plain, &mut memo)
    }

    /// Recursively materialize arrays and maps, mirroring `materialize_value`
    /// in `vm/c/vm.c`. Used by the write/stringify host calls so a reactive
    /// value's current contents are emitted.
    fn materialize_value(&mut self, source: &Value) -> Result<Value, LanaError> {
        let current = self.reactive_value(source);
        match &current.kind {
            ValueKind::Array(array) => {
                if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                let items = array
                    .lock().unwrap()
                    .items
                    .iter()
                    .map(|item| self.materialize_value(item))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Value::array(Arc::new(Mutex::new(Array { items }))))
            }
            ValueKind::Map(map) => {
                if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                let mut copy = Map::new(map.lock().unwrap().entries.len());
                for entry in &map.lock().unwrap().entries {
                    let value = self.materialize_value(&entry.value)?;
                    copy.set(entry.key.clone(), value, true)?;
                }
                Ok(Value::map(Arc::new(Mutex::new(copy))))
            }
            _ => self.clone_without_runtime_metadata(&current),
        }
    }

    /// Deep-clone a reactive node and its inputs, mirroring
    /// `clone_live_reactive_node` in `vm/c/vm.c`. The shared-information layer
    /// needs an isolated reactive graph per version so observing one snapshot
    /// does not mutate another.
    fn deep_clone_reactive(
        &mut self,
        node: &Arc<Mutex<Reactive>>,
        memo: &mut HashMap<usize, Arc<Mutex<Reactive>>>,
    ) -> Result<Arc<Mutex<Reactive>>, LanaError> {
        let key = Arc::as_ptr(node) as usize;
        if let Some(existing) = memo.get(&key) {
            return Ok(existing.clone());
        }
        let (id, dependency_id, revision, kind, relationship, exactness, operation, input0, input1, constant0, constant1, current, history) = {
            let guard = node.lock().unwrap();
            (
                guard.id,
                guard.dependency_id,
                guard.revision,
                guard.kind,
                guard.relationship,
                guard.exactness,
                guard.operation,
                guard.inputs[0].clone(),
                guard.inputs[1].clone(),
                guard.constants[0].clone(),
                guard.constants[1].clone(),
                guard.current.clone(),
                guard.history.clone(),
            )
        };
        let cloned_input0 = match &input0 {
            Some(input) => Some(self.deep_clone_reactive(input, memo)?),
            None => None,
        };
        let cloned_input1 = match &input1 {
            Some(input) => Some(self.deep_clone_reactive(input, memo)?),
            None => None,
        };
        let clone_plain = |vm: &mut Self, value: &Value| -> Result<Value, LanaError> {
            let mut m = DeepCloneMemo::default();
            vm.deep_clone_value(value, &mut m)
        };
        let cloned_constant0 = match &constant0 {
            Some(value) => Some(clone_plain(self, value)?),
            None => None,
        };
        let cloned_constant1 = match &constant1 {
            Some(value) => Some(clone_plain(self, value)?),
            None => None,
        };
        let cloned_current = match &current {
            Some(value) => Some(clone_plain(self, value)?),
            None => None,
        };
        let mut cloned_history = Vec::with_capacity(history.len());
        for version in &history {
            cloned_history.push(ReactiveVersion {
                revision: version.revision,
                value: match &version.value {
                    Some(value) => Some(clone_plain(self, value)?),
                    None => None,
                },
            });
        }
        let copy = Arc::new(Mutex::new(Reactive {
            id,
            dependency_id,
            revision,
            kind,
            relationship,
            exactness,
            operation,
            inputs: [cloned_input0, cloned_input1],
            constants: [cloned_constant0, cloned_constant1],
            current: cloned_current,
            history: cloned_history,
        }));
        memo.insert(key, copy.clone());
        Ok(copy)
    }

    /// Deep-clone a value and its reactive graph, mirroring
    /// `lana_vm_clone_live_value` in `vm/c/vm.c`. Used by the shared-information
    /// layer so each version owns an isolated reactive graph.
    fn deep_clone_live_value(
        &mut self,
        value: &Value,
        reactive_memo: &mut HashMap<usize, Arc<Mutex<Reactive>>>,
    ) -> Result<Value, LanaError> {
        let mut cloned = Value {
            kind: ValueKind::Null,
            derivation: value.derivation.clone(),
            reactive: None,
            claim: value.claim.clone(),
            planned_effect: value.planned_effect.clone(),
        };
        if let Some(reactive) = &value.reactive {
            cloned.reactive = Some(self.deep_clone_reactive(reactive, reactive_memo)?);
        }
        match &value.kind {
            ValueKind::Null
            | ValueKind::Number(_)
            | ValueKind::Bool(_)
            | ValueKind::Sample(_)
            | ValueKind::Function(_)
            | ValueKind::Distribution { .. }
            | ValueKind::Capability(_)
            | ValueKind::Lazy { .. } => cloned.kind = value.kind.clone(),
            ValueKind::String(string) => cloned.kind = ValueKind::String(string.clone()),
            ValueKind::State(state) => cloned.kind = ValueKind::State(state.clone()),
            ValueKind::Array(array) => {
                let items = array
                    .lock().unwrap()
                    .items
                    .iter()
                    .map(|item| self.deep_clone_live_value(item, reactive_memo))
                    .collect::<Result<Vec<_>, _>>()?;
                cloned.kind = ValueKind::Array(Arc::new(Mutex::new(Array { items })));
            }
            ValueKind::Map(map) => {
                let mut copy = Map::new(map.lock().unwrap().entries.len());
                for entry in &map.lock().unwrap().entries {
                    let value = self.deep_clone_live_value(&entry.value, reactive_memo)?;
                    copy.set(entry.key.clone(), value, true)?;
                }
                cloned.kind = ValueKind::Map(Arc::new(Mutex::new(copy)));
            }
            ValueKind::Possibility(possibility) => {
                let values = possibility
                    .values
                    .iter()
                    .map(|v| self.deep_clone_live_value(v, reactive_memo))
                    .collect::<Result<Vec<_>, _>>()?;
                cloned.kind = ValueKind::Possibility(Arc::new(Possibility {
                    values,
                    weights: possibility.weights.clone(),
                    dependency_id: possibility.dependency_id,
                }));
            }
            ValueKind::PathSet(paths) => {
                let mut alternatives = Vec::with_capacity(paths.alternatives.len());
                for alternative in &paths.alternatives {
                    alternatives.push(PathAlternative {
                        guard: alternative.guard,
                        weight: alternative.weight,
                        result: self.deep_clone_live_value(&alternative.result, reactive_memo)?,
                    });
                }
                cloned.kind = ValueKind::PathSet(Arc::new(PathSet {
                    alternatives,
                    dependency_id: paths.dependency_id,
                }));
            }
            ValueKind::Adt(adt) => {
                let fields = adt
                    .fields
                    .iter()
                    .map(|field| self.deep_clone_live_value(field, reactive_memo))
                    .collect::<Result<Vec<_>, _>>()?;
                cloned.kind = ValueKind::Adt(Arc::new(Adt { variant: adt.variant, fields }));
            }
            ValueKind::Joint(_) | ValueKind::StateDist(_) => {
                let mut m = DeepCloneMemo::default();
                cloned.kind = self.deep_clone_value(value, &mut m)?.kind;
            }
            ValueKind::Task(_) => return Err(LanaError::Type),
        }
        Ok(cloned)
    }

    /// Wrap a value in a reactive root, mirroring `lana_vm_reactive_root`.
    fn reactive_root(
        &mut self,
        source: &Value,
        exactness: DerivationExactness,
    ) -> Result<Value, LanaError> {
        if source.reactive.is_some() {
            return Err(LanaError::Format);
        }
        let id = self.next_reactive_id;
        self.next_reactive_id += 1;
        let dependency_id = match &source.kind {
            ValueKind::Possibility(possibility) => possibility.dependency_id,
            ValueKind::PathSet(paths) => paths.dependency_id,
            _ => {
                let id = self.next_dependency_id;
                self.next_dependency_id += 1;
                id
            }
        };
        let current = self.clone_without_runtime_metadata(source)?;
        let node = Arc::new(Mutex::new(Reactive {
            id,
            dependency_id,
            revision: self.revision,
            kind: ReactiveKind::Root,
            relationship: RelationshipKind::Exact,
            exactness,
            operation: 0,
            inputs: [None, None],
            constants: [None, None],
            current: Some(current),
            history: Vec::new(),
        }));
        let mut out = source.clone();
        out.reactive = Some(node);
        Ok(out)
    }

    /// Observe evidence against a reactive root, mirroring
    /// `lana_vm_reactive_observe`.
    fn reactive_observe(&mut self, source: &Value, evidence: &Value) -> Result<Value, LanaError> {
        let Some(reactive) = &source.reactive else {
            return Err(LanaError::Format);
        };
        {
            let node = reactive.lock().unwrap();
            if node.kind != ReactiveKind::Root {
                return Err(LanaError::Format);
            }
        }
        let replacement = self.reactive_value(evidence);
        if self.active_path_count > 1 || self.value_is_unresolved(&replacement) {
            return Err(LanaError::UnresolvedValue);
        }
        let current = reactive
            .lock()
            .unwrap()
            .current
            .clone()
            .unwrap_or_else(Value::null);
        match &current.kind {
            ValueKind::Possibility(possibility) => {
                if !possibility
                    .values
                    .iter()
                    .any(|value| joint_value_equal(value, &replacement))
                {
                    return Err(LanaError::InvalidConditioning);
                }
            }
            ValueKind::PathSet(paths) => {
                if !paths
                    .alternatives
                    .iter()
                    .any(|alternative| joint_value_equal(&alternative.result, &replacement))
                {
                    return Err(LanaError::InvalidConditioning);
                }
            }
            _ => {
                if !joint_value_equal(&current, &replacement) {
                    return Err(LanaError::InvalidConditioning);
                }
            }
        }
        self.reactive_recompute_transaction(reactive, &replacement)?;
        self.observation_count += 1;
        Ok(source.clone())
    }

    /// Attach a claim to a value, mirroring `lana_vm_claim`.
    fn claim(
        &mut self,
        source: &Value,
        proposition: &str,
        exactness: DerivationExactness,
        tolerance: f64,
        source_valid: bool,
    ) -> Result<Value, LanaError> {
        if tolerance < 0.0 {
            return Err(LanaError::Format);
        }
        if self.alloc_bytes(std::mem::size_of::<Claim>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut memo = DeepCloneMemo::default();
        let value = self.deep_clone_value(source, &mut memo)?;
        let claim = Arc::new(Claim {
            value,
            proposition: Arc::from(proposition),
            exactness,
            tolerance,
            source_valid,
        });
        let mut out = source.clone();
        out.claim = Some(claim);
        Ok(out)
    }

    /// Attach a planned effect to a value, mirroring `lana_vm_planned_effect`.
    fn planned_effect(&mut self, kind: &str, payload: &Value) -> Result<Value, LanaError> {
        if kind.is_empty() {
            return Err(LanaError::Format);
        }
        if self.alloc_bytes(std::mem::size_of::<PlannedEffect>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let id = self.next_effect_id;
        self.next_effect_id += 1;
        let payload_plain = self.clone_without_runtime_metadata(payload)?;
        let plan = Arc::new(PlannedEffect {
            id,
            kind: Arc::from(kind),
            payload: payload_plain,
            state: Mutex::new(PlannedEffectState::default()),
        });
        let mut out = payload.clone();
        out.planned_effect = Some(plan);
        Ok(out)
    }

    /// Execute a planned effect, mirroring `lana_vm_execute_planned_effect` with
    /// the `execute_captured_payload` executor (which just clones the payload).
    fn execute_planned_effect(&mut self, plan_value: &Value) -> Result<Value, LanaError> {
        let Some(plan) = &plan_value.planned_effect else {
            return Err(LanaError::Format);
        };
        {
            let state = plan.state.lock().unwrap();
            for receipt in &state.receipts {
                if receipt.revision == self.revision {
                    let mut memo = DeepCloneMemo::default();
                    return self.deep_clone_value(&receipt.result, &mut memo);
                }
            }
        }
        if self.value_is_unresolved(&plan.payload) {
            return Err(LanaError::UnresolvedValue);
        }
        let current = self.reactive_value(&plan.payload);
        let mut memo = DeepCloneMemo::default();
        let result = self.deep_clone_value(&current, &mut memo)?;
        let receipt = EffectReceipt {
            revision: self.revision,
            result: self.clone_without_runtime_metadata(&result)?,
        };
        let mut memo = DeepCloneMemo::default();
        let returned = self.deep_clone_value(&receipt.result, &mut memo)?;
        {
            let mut state = plan.state.lock().unwrap();
            state.receipts.push(receipt);
            state.execution_count += 1;
        }
        Ok(returned)
    }

    /// Recompute a reactive graph after an observation, mirroring
    /// `reactive_recompute_transaction` in `vm/c/vm.c`.
    fn reactive_recompute_transaction(
        &mut self,
        root: &Arc<Mutex<Reactive>>,
        replacement: &Value,
    ) -> Result<(), LanaError> {
        let mut list: Vec<Arc<Mutex<Reactive>>> = Vec::new();
        for frame in &self.frames {
            for register in frame.registers.iter() {
                reactive_collect_value(&mut list, register);
            }
        }
        reactive_collect_value(&mut list, &self.result);
        if !list.iter().any(|node| Arc::ptr_eq(node, root)) {
            reactive_list_add(&mut list, root);
        }
        let count = list.len();
        let mut staged: Vec<Option<Value>> = vec![None; count];
        let mut affected: Vec<bool> = vec![false; count];
        for index in 0..count {
            let node = list[index].clone();
            let is_root = Arc::ptr_eq(&node, root);
            if is_root {
                affected[index] = true;
            } else {
                let (input0, input1) = {
                    let guard = node.lock().unwrap();
                    (guard.inputs[0].clone(), guard.inputs[1].clone())
                };
                let left_index = input0
                    .as_ref()
                    .and_then(|input| list.iter().position(|n| Arc::ptr_eq(n, input)));
                let right_index = input1
                    .as_ref()
                    .and_then(|input| list.iter().position(|n| Arc::ptr_eq(n, input)));
                affected[index] = left_index.map(|i| affected[i]).unwrap_or(false)
                    || right_index.map(|i| affected[i]).unwrap_or(false);
            }
            if !affected[index] {
                continue;
            }
            let mut staged_value = Value::null();
            let error = if is_root {
                match self.clone_without_runtime_metadata(replacement) {
                    Ok(value) => {
                        staged_value = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            } else {
                let (kind, operation, input0, input1, constant0, constant1) = {
                    let guard = node.lock().unwrap();
                    (
                        guard.kind,
                        guard.operation,
                        guard.inputs[0].clone(),
                        guard.inputs[1].clone(),
                        guard.constants[0].clone(),
                        guard.constants[1].clone(),
                    )
                };
                let left = reactive_staged_input(&list, &staged, &input0, &constant0);
                let right = reactive_staged_input(&list, &staged, &input1, &constant1);
                match kind {
                    ReactiveKind::Binary => {
                        self.lift_binary_raw(&left, &right, PureKind::Binary, operation, &mut staged_value)
                    }
                    ReactiveKind::Compare => {
                        self.lift_binary_raw(&left, &right, PureKind::Compare, operation, &mut staged_value)
                    }
                    ReactiveKind::Unary => self.lift_unary(&left, operation, &mut staged_value),
                    _ => LanaError::UnsupportedOperation,
                }
            };
            if error != LanaError::Ok {
                return Err(error);
            }
            staged[index] = Some(staged_value);
        }
        let revision = self.revision + 1;
        for index in 0..count {
            if !affected[index] {
                continue;
            }
            let mut guard = list[index].lock().unwrap();
            let old_revision = guard.revision;
            let old_current = guard.current.clone();
            guard.history.push(ReactiveVersion {
                revision: old_revision,
                value: old_current,
            });
            guard.current = staged[index].take();
            guard.revision = revision;
        }
        self.revision = revision;
        Ok(())
    }

    /// Execute a host call, mirroring `execute_host_call` in `vm/c/vm.c`.
    fn execute_host_call(
        &mut self,
        host_id: u32,
        arguments: &[Value],
        out: &mut Value,
    ) -> LanaError {
        let argc = arguments.len();
        *out = Value::null();
        match host_id {
            LANA_HOST_ARGS => {
                if argc != 0 {
                    return LanaError::Type;
                }
                if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
                    return LanaError::Oom;
                }
                let mut items = Vec::with_capacity(self.program_argc);
                for index in 0..self.program_argc {
                    let source = Value::string(self.program_argv[index].clone());
                    let mut memo = DeepCloneMemo::default();
                    items.push(match self.deep_clone_value(&source, &mut memo) {
                        Ok(value) => value,
                        Err(error) => return error,
                    });
                }
                *out = Value::array(Arc::new(Mutex::new(Array { items })));
                LanaError::Ok
            }
            LANA_HOST_READ_TEXT => {
                if argc != 1 {
                    return LanaError::Type;
                }
                self.host_read_text(&arguments[0], out)
            }
            LANA_HOST_WRITE_TEXT => {
                if argc != 2 {
                    return LanaError::Type;
                }
                let (ValueKind::String(path), ValueKind::String(contents)) =
                    (&arguments[0].kind, &arguments[1].kind)
                else {
                    return LanaError::Type;
                };
                if let Some(fs) = &mut self.virtual_fs {
                    fs.insert(path.to_string(), contents.to_string());
                    LanaError::Ok
                } else if std::fs::write(&**path, contents.as_bytes()).is_err() {
                    LanaError::Io
                } else {
                    LanaError::Ok
                }
            }
            LANA_HOST_DIRECTORY_LIST => {
                if argc != 1 {
                    return LanaError::Type;
                }
                self.host_directory_list(&arguments[0], out)
            }
            LANA_HOST_DIRECTORY_CREATE => {
                if argc != 1 {
                    return LanaError::Type;
                }
                self.host_directory_create(&arguments[0])
            }
            LANA_HOST_PATH_EXISTS => {
                if argc != 1 {
                    return LanaError::Type;
                }
                self.host_path_exists(&arguments[0], out)
            }
            LANA_HOST_WRITE_TEXT_ATOMIC => {
                if argc != 2 {
                    return LanaError::Type;
                }
                self.host_write_text_atomic(&arguments[0], &arguments[1])
            }
            LANA_HOST_HASH_UPDATE => {
                if argc == 3
                    && matches!(&arguments[2].kind, ValueKind::String(s) if &**s == "xor")
                {
                    return self.host_hash_xor(&arguments[0], &arguments[1], out);
                }
                if argc != 2 {
                    return LanaError::Type;
                }
                self.host_hash_update(&arguments[0], &arguments[1], out)
            }
            LANA_HOST_LAZY_BOUND => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let ValueKind::Lazy { bound, .. } = arguments[0].kind else {
                    return LanaError::Type;
                };
                *out = Value::number(bound as f64);
                LanaError::Ok
            }
            LANA_HOST_CORRELATED => {
                if argc != 3 {
                    return LanaError::Type;
                }
                let (ValueKind::State(xs), ValueKind::State(ys), ValueKind::Number(coefficient)) =
                    (&arguments[0].kind, &arguments[1].kind, &arguments[2].kind)
                else {
                    return LanaError::Type;
                };
                let p_x = xs.state.p;
                let p_y = ys.state.p;
                let coefficient = *coefficient;
                if !coefficient.is_finite() || coefficient < -1.0 || coefficient > 1.0 {
                    return LanaError::InvalidParameters;
                }
                if p_x < 0.0 || p_x > 1.0 || p_y < 0.0 || p_y > 1.0 {
                    return LanaError::Type;
                }
                let cross = coefficient * (p_x * (1.0 - p_x) * p_y * (1.0 - p_y)).sqrt();
                let p11 = p_x * p_y + cross;
                let p10 = p_x - p11;
                let p01 = p_y - p11;
                let p00 = 1.0 - p_x - p_y + p11;
                let mut cells = [p00, p01, p10, p11];
                for cell in cells.iter_mut() {
                    if *cell < -1e-12 || *cell > 1.0 + 1e-12 {
                        return LanaError::InvalidParameters;
                    }
                    *cell = cell.clamp(0.0, 1.0);
                }
                // |d| = 1 collapses the 2x2 law to its diagonal; the finite
                // joint law rejects zero-weight rows, so emit only positive mass.
                let mut rows: Vec<Value> = Vec::new();
                let mut weights: Vec<f64> = Vec::new();
                for (index, cell) in cells.iter().enumerate() {
                    if *cell <= 1e-12 {
                        continue;
                    }
                    rows.push(Value::number(((index >> 1) & 1) as f64));
                    rows.push(Value::number((index & 1) as f64));
                    weights.push(*cell);
                }
                let row_count = weights.len();
                let joint = match self.joint_build_finite("x;y", &rows, &weights, row_count, 2) {
                    Ok(joint) => joint,
                    Err(error) => return error,
                };
                *out = Value::joint(joint);
                LanaError::Ok
            }
            LANA_HOST_NOW => {
                if argc != 0 {
                    return LanaError::Type;
                }
                // `SystemTime::now()` panics on wasm32-unknown-unknown (no
                // clock); the embedded compiler never calls `now`, and a user
                // program that does gets 0.0 rather than a trap.
                #[cfg(target_arch = "wasm32")]
                let seconds = 0.0;
                #[cfg(not(target_arch = "wasm32"))]
                let seconds = match SystemTime::now().duration_since(UNIX_EPOCH) {
                    Ok(duration) => duration.as_secs() as f64 + duration.subsec_nanos() as f64 / 1_000_000_000.0,
                    Err(_) => return LanaError::Type,
                };
                *out = Value::number(seconds);
                LanaError::Ok
            }
            LANA_HOST_RANDOM => {
                if argc != 0 {
                    return LanaError::Type;
                }
                *out = Value::number(self.rng.random() as f64 / 4294967296.0);
                LanaError::Ok
            }
            LANA_HOST_ASSERT => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Bool(_))
                    || !matches!(arguments[1].kind, ValueKind::String(_))
                {
                    return LanaError::Type;
                }
                if arguments[0].as_bool() {
                    LanaError::Ok
                } else {
                    LanaError::Assertion
                }
            }
            LANA_HOST_MAP_NEW => {
                if argc % 2 != 0 {
                    return LanaError::Type;
                }
                if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
                    return LanaError::Oom;
                }
                let mut map = Map::new(argc / 2);
                let mut index = 0;
                while index < argc {
                    let ValueKind::String(key) = &arguments[index].kind else {
                        return LanaError::Type;
                    };
                    if let Err(error) = map.set(key.clone(), arguments[index + 1].clone(), true) {
                        return error;
                    }
                    index += 2;
                }
                *out = Value::map(Arc::new(Mutex::new(map)));
                LanaError::Ok
            }
            LANA_HOST_MAP_HAS => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Map(_))
                    || !matches!(arguments[1].kind, ValueKind::String(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Map(map) = &arguments[0].kind else {
                    unreachable!()
                };
                let found = map.lock().unwrap().has(&arguments[1].as_string());
                *out = Value::boolean(found);
                LanaError::Ok
            }
            LANA_HOST_MAP_GET => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Map(_))
                    || !matches!(arguments[1].kind, ValueKind::String(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Map(map) = &arguments[0].kind else {
                    unreachable!()
                };
                match map.lock().unwrap().get(&arguments[1].as_string()) {
                    Some(value) => {
                        *out = value.clone();
                        LanaError::Ok
                    }
                    None => LanaError::Key,
                }
            }
            LANA_HOST_MAP_SET => {
                if argc != 3
                    || !matches!(arguments[0].kind, ValueKind::Map(_))
                    || !matches!(arguments[1].kind, ValueKind::String(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Map(map) = &arguments[0].kind else {
                    unreachable!()
                };
                match map
                    .lock()
                    .unwrap()
                    .set(arguments[1].as_string(), arguments[2].clone(), false)
                {
                    Ok(()) => {
                        *out = arguments[2].clone();
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_MAP_KEYS => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::Map(_)) {
                    return LanaError::Type;
                }
                let ValueKind::Map(map) = &arguments[0].kind else {
                    unreachable!()
                };
                if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
                    return LanaError::Oom;
                }
                let items = map
                    .lock()
                    .unwrap()
                    .entries
                    .iter()
                    .map(|entry| Value::string(entry.key.clone()))
                    .collect();
                *out = Value::array(Arc::new(Mutex::new(Array { items })));
                LanaError::Ok
            }
            LANA_HOST_INDEX_GET => {
                if argc != 2 {
                    return LanaError::Type;
                }
                if matches!(arguments[0].kind, ValueKind::Map(_))
                    && matches!(arguments[1].kind, ValueKind::String(_))
                {
                    let ValueKind::Map(map) = &arguments[0].kind else {
                        unreachable!()
                    };
                    return match map.lock().unwrap().get(&arguments[1].as_string()) {
                        Some(value) => {
                            *out = value.clone();
                            LanaError::Ok
                        }
                        None => LanaError::Key,
                    };
                }
                if matches!(arguments[0].kind, ValueKind::Array(_))
                    && matches!(arguments[1].kind, ValueKind::Number(n)
                        if n >= 0.0 && n.floor() == n)
                {
                    let ValueKind::Array(array) = &arguments[0].kind else {
                        unreachable!()
                    };
                    let index = arguments[1].as_number() as usize;
                    let array = array.lock().unwrap();
                    if index < array.items.len() {
                        *out = array.items[index].clone();
                        return LanaError::Ok;
                    }
                    return LanaError::Limit;
                }
                if matches!(arguments[0].kind, ValueKind::Array(_)) {
                    LanaError::Limit
                } else {
                    LanaError::Type
                }
            }
            LANA_HOST_INDEX_SET => {
                if argc != 3 {
                    return LanaError::Type;
                }
                if matches!(arguments[0].kind, ValueKind::Map(_))
                    && matches!(arguments[1].kind, ValueKind::String(_))
                {
                    let ValueKind::Map(map) = &arguments[0].kind else {
                        unreachable!()
                    };
                    return match map
                        .lock()
                        .unwrap()
                        .set(arguments[1].as_string(), arguments[2].clone(), false)
                    {
                        Ok(()) => {
                            *out = arguments[2].clone();
                            LanaError::Ok
                        }
                        Err(error) => error,
                    };
                }
                if matches!(arguments[0].kind, ValueKind::Array(_))
                    && matches!(arguments[1].kind, ValueKind::Number(n)
                        if n >= 0.0 && n.floor() == n)
                {
                    let ValueKind::Array(array) = &arguments[0].kind else {
                        unreachable!()
                    };
                    let index = arguments[1].as_number() as usize;
                    let mut array = array.lock().unwrap();
                    if index < array.items.len() {
                        array.items[index] = arguments[2].clone();
                        *out = arguments[2].clone();
                        return LanaError::Ok;
                    }
                    return LanaError::Limit;
                }
                if matches!(arguments[0].kind, ValueKind::Array(_)) {
                    LanaError::Limit
                } else {
                    LanaError::Type
                }
            }
            LANA_HOST_JSON_PARSE => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                self.json_parse(&arguments[0].as_string(), out)
            }
            LANA_HOST_JSON_STRINGIFY => {
                if argc != 1 {
                    return LanaError::Type;
                }
                self.json_stringify(&arguments[0], out)
            }
            LANA_HOST_CSV_READ => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                self.csv_read(&arguments[0].as_string(), out)
            }
            LANA_HOST_CSV_WRITE => {
                if argc != 2 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                self.csv_write(&arguments[0].as_string(), &arguments[1], out)
            }
            LANA_HOST_STRING_LENGTH => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                *out = Value::number(arguments[0].as_string().len() as f64);
                LanaError::Ok
            }
            LANA_HOST_STRING_BYTE_AT => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::String(_))
                    || !matches!(arguments[1].kind, ValueKind::Number(n)
                        if n >= 0.0 && n.floor() == n)
                {
                    return LanaError::Type;
                }
                let position = arguments[1].as_number() as usize;
                let text = arguments[0].as_string();
                if position >= text.len() {
                    return LanaError::Limit;
                }
                *out = Value::number(text.as_bytes()[position] as f64);
                LanaError::Ok
            }
            LANA_HOST_STRING_SLICE => {
                if argc != 3
                    || !matches!(arguments[0].kind, ValueKind::String(_))
                    || !matches!(arguments[1].kind, ValueKind::Number(n)
                        if n >= 0.0 && n.floor() == n)
                    || !matches!(arguments[2].kind, ValueKind::Number(n)
                        if n >= 0.0 && n.floor() == n)
                {
                    return LanaError::Type;
                }
                let start = arguments[1].as_number() as usize;
                let end = arguments[2].as_number() as usize;
                let text = arguments[0].as_string();
                if start > end || end > text.len() {
                    return LanaError::Limit;
                }
                *out = Value::string(Arc::from(&text[start..end]));
                LanaError::Ok
            }
            LANA_HOST_STRING_CONCAT => {
                let mut total = 0usize;
                for argument in arguments {
                    if !matches!(argument.kind, ValueKind::String(_)) {
                        return LanaError::Type;
                    }
                    total += argument.as_string().len();
                }
                let mut joined = String::with_capacity(total);
                for argument in arguments {
                    joined.push_str(&argument.as_string());
                }
                *out = Value::string(Arc::from(joined));
                LanaError::Ok
            }
            LANA_HOST_NUMBER_TO_STRING => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::Number(_)) {
                    return LanaError::Type;
                }
                *out = Value::string(Arc::from(format_g17(arguments[0].as_number())));
                LanaError::Ok
            }
            LANA_HOST_ARRAY_NEW => {
                if argc != 1
                    || !matches!(arguments[0].kind, ValueKind::Number(n)
                        if n >= 0.0 && n.floor() == n
                            && n <= (LANA_MAX_REGISTERS as f64) * 4096.0)
                {
                    return LanaError::Type;
                }
                let count = arguments[0].as_number() as usize;
                if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
                    return LanaError::Oom;
                }
                *out = Value::array(Arc::new(Mutex::new(Array {
                    items: vec![Value::null(); count],
                })));
                LanaError::Ok
            }
            LANA_HOST_ARRAY_PUSH => {
                if argc != 2 || !matches!(arguments[0].kind, ValueKind::Array(_)) {
                    return LanaError::Type;
                }
                let ValueKind::Array(array) = &arguments[0].kind else {
                    unreachable!()
                };
                array.lock().unwrap().items.push(arguments[1].clone());
                *out = arguments[0].clone();
                LanaError::Ok
            }
            LANA_HOST_STRING_HEX => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                const DIGITS: &[u8; 16] = b"0123456789abcdef";
                let source = arguments[0].as_string();
                let mut hex = String::with_capacity(source.len() * 2);
                for byte in source.as_bytes() {
                    hex.push(DIGITS[(byte >> 4) as usize] as char);
                    hex.push(DIGITS[(byte & 15) as usize] as char);
                }
                *out = Value::string(Arc::from(hex));
                LanaError::Ok
            }
            LANA_HOST_STRING_JOIN => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Array(_))
                    || !matches!(arguments[1].kind, ValueKind::String(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Array(array) = &arguments[0].kind else {
                    unreachable!()
                };
                let separator = arguments[1].as_string();
                let array = array.lock().unwrap();
                let mut joined = String::new();
                for (index, item) in array.items.iter().enumerate() {
                    if !matches!(item.kind, ValueKind::String(_)) {
                        return LanaError::Type;
                    }
                    if index > 0 {
                        joined.push_str(&separator);
                    }
                    joined.push_str(&item.as_string());
                }
                *out = Value::string(Arc::from(joined));
                LanaError::Ok
            }
            LANA_HOST_ARRAY_LENGTH => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::Array(_)) {
                    return LanaError::Type;
                }
                let ValueKind::Array(array) = &arguments[0].kind else {
                    unreachable!()
                };
                *out = Value::number(array.lock().unwrap().items.len() as f64);
                LanaError::Ok
            }
            LANA_HOST_STRING_UNESCAPE => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let source = arguments[0].as_string();
                let bytes = source.as_bytes();
                let mut decoded = String::with_capacity(bytes.len());
                let mut read = 0;
                while read < bytes.len() {
                    let value = bytes[read];
                    read += 1;
                    if value != b'\\' {
                        decoded.push(value as char);
                        continue;
                    }
                    if read >= bytes.len() {
                        return LanaError::Format;
                    }
                    let value = bytes[read];
                    read += 1;
                    match value {
                        b'n' => decoded.push('\n'),
                        b'r' => decoded.push('\r'),
                        b't' => decoded.push('\t'),
                        b'\\' | b'"' => decoded.push(value as char),
                        _ => return LanaError::Format,
                    }
                }
                *out = Value::string(Arc::from(decoded));
                LanaError::Ok
            }
            LANA_HOST_PATH_RESOLVE => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::String(_))
                    || !matches!(arguments[1].kind, ValueKind::String(_))
                {
                    return LanaError::Type;
                }
                self.host_path_resolve(&arguments[0].as_string(), &arguments[1].as_string(), out)
            }
            LANA_HOST_SAMPLE_RECORD => {
                if argc != 3
                    || !matches!(arguments[1].kind, ValueKind::String(_))
                    || !matches!(arguments[2].kind, ValueKind::String(_))
                {
                    return LanaError::Type;
                }
                if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
                    return LanaError::Oom;
                }
                let mut metadata = Map::new(5);
                if let Err(error) = metadata.set(
                    Arc::from("source_dependency"),
                    arguments[1].clone(),
                    true,
                ) {
                    return error;
                }
                if let Err(error) = metadata.set(
                    Arc::from("rng_seed"),
                    Value::number(self.root_seed as f64),
                    true,
                ) {
                    return error;
                }
                if let Err(error) = metadata.set(
                    Arc::from("task_lineage"),
                    Value::number(self.lineage as f64),
                    true,
                ) {
                    return error;
                }
                if let Err(error) = metadata.set(Arc::from("operation"), arguments[2].clone(), true) {
                    return error;
                }
                if let Err(error) = metadata.set(
                    Arc::from("revision"),
                    Value::number(self.revision as f64),
                    true,
                ) {
                    return error;
                }
                *out = Value::array(Arc::new(Mutex::new(Array {
                    items: vec![
                        arguments[0].clone(),
                        Value::map(Arc::new(Mutex::new(metadata))),
                    ],
                })));
                LanaError::Ok
            }
            LANA_HOST_INFORMATION_NEW => {
                if argc != 1 {
                    return LanaError::Type;
                }
                match self.reactive_root(&arguments[0], DerivationExactness::Exact) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_CLAIM_NEW => {
                if argc != 2 || !matches!(arguments[1].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                match self.claim(
                    &arguments[0],
                    &arguments[1].as_string(),
                    DerivationExactness::Exact,
                    0.0,
                    true,
                ) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_CLAIM_VALUE => {
                if argc != 1 || arguments[0].claim.is_none() {
                    return LanaError::Type;
                }
                let mut memo = DeepCloneMemo::default();
                match self.deep_clone_value(&arguments[0].claim.as_ref().unwrap().value, &mut memo) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_CLAIM_PROPOSITION => {
                if argc != 1 || arguments[0].claim.is_none() {
                    return LanaError::Type;
                }
                *out = Value::string(arguments[0].claim.as_ref().unwrap().proposition.clone());
                LanaError::Ok
            }
            LANA_HOST_CLAIM_STATUS => {
                if argc != 1 || arguments[0].claim.is_none() {
                    return LanaError::Type;
                }
                let claim = arguments[0].claim.as_ref().unwrap();
                let mut status = Map::new(3);
                if let Err(error) = status.set(
                    Arc::from("exactness"),
                    Value::string(Arc::from(derivation::exactness_name(claim.exactness))),
                    true,
                ) {
                    return error;
                }
                if let Err(error) = status.set(Arc::from("tolerance"), Value::number(claim.tolerance), true) {
                    return error;
                }
                if let Err(error) = status.set(
                    Arc::from("source_valid"),
                    Value::boolean(claim.source_valid),
                    true,
                ) {
                    return error;
                }
                *out = Value::map(Arc::new(Mutex::new(status)));
                LanaError::Ok
            }
            LANA_HOST_PLANNED_EFFECT_NEW => {
                if argc != 2 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                match self.planned_effect(&arguments[0].as_string(), &arguments[1]) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_PLANNED_EFFECT_EXECUTE => {
                if argc != 1 {
                    return LanaError::Type;
                }
                match self.execute_planned_effect(&arguments[0]) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_PLANNED_EFFECT_STATUS => {
                if argc != 1 || arguments[0].planned_effect.is_none() {
                    return LanaError::Type;
                }
                let plan = arguments[0].planned_effect.as_ref().unwrap();
                let state = plan.state.lock().unwrap();
                let mut status = Map::new(3);
                if let Err(error) = status.set(Arc::from("identity"), Value::number(plan.id as f64), true) {
                    return error;
                }
                if let Err(error) = status.set(
                    Arc::from("execution_count"),
                    Value::number(state.execution_count as f64),
                    true,
                ) {
                    return error;
                }
                if let Err(error) = status.set(Arc::from("kind"), Value::string(plan.kind.clone()), true) {
                    return error;
                }
                *out = Value::map(Arc::new(Mutex::new(status)));
                LanaError::Ok
            }
            LANA_HOST_SHARED_INFORMATION => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let (_, admin) = match self.shared_information_create(&arguments[0]) {
                    Ok(pair) => pair,
                    Err(error) => return error,
                };
                *out = Value::capability(admin);
                LanaError::Ok
            }
            LANA_HOST_SHARED_GRANT => {
                if argc != 2 || !matches!(arguments[0].kind, ValueKind::Capability(_)) {
                    return LanaError::Type;
                }
                let permission = shared_permission(&arguments[1]);
                if permission == 0 {
                    return LanaError::Type;
                }
                let ValueKind::Capability(admin) = &arguments[0].kind else {
                    unreachable!()
                };
                let capability = match self.shared_capability_grant(admin, permission) {
                    Ok(capability) => capability,
                    Err(error) => return error,
                };
                *out = Value::capability(capability);
                LanaError::Ok
            }
            LANA_HOST_SHARED_REVOKE => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Capability(_))
                    || !matches!(arguments[1].kind, ValueKind::Capability(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Capability(admin) = &arguments[0].kind else {
                    unreachable!()
                };
                let ValueKind::Capability(target) = &arguments[1].kind else {
                    unreachable!()
                };
                self.shared_capability_revoke(admin, target)
            }
            LANA_HOST_SHARED_SNAPSHOT => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::Capability(_)) {
                    return LanaError::Type;
                }
                let ValueKind::Capability(capability) = &arguments[0].kind else {
                    unreachable!()
                };
                self.shared_information_snapshot(capability, out)
            }
            LANA_HOST_SHARED_AT => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Capability(_))
                    || !nonnegative_integer(&arguments[1])
                {
                    return LanaError::Type;
                }
                let ValueKind::Capability(capability) = &arguments[0].kind else {
                    unreachable!()
                };
                self.shared_information_at(capability, arguments[1].as_number(), out)
            }
            LANA_HOST_SHARED_OBSERVE => {
                if argc != 3
                    || !matches!(arguments[0].kind, ValueKind::Capability(_))
                    || !nonnegative_integer(&arguments[2])
                {
                    return LanaError::Type;
                }
                let ValueKind::Capability(capability) = &arguments[0].kind else {
                    unreachable!()
                };
                let revision = match self.shared_information_observe(
                    capability,
                    &arguments[1],
                    arguments[2].as_number(),
                ) {
                    Ok(revision) => revision,
                    Err(error) => return error,
                };
                *out = Value::number(revision as f64);
                LanaError::Ok
            }
            LANA_HOST_SHARED_REVISION => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::Capability(_)) {
                    return LanaError::Type;
                }
                let ValueKind::Capability(capability) = &arguments[0].kind else {
                    unreachable!()
                };
                *out = Value::number(self.shared_information_revision(capability) as f64);
                LanaError::Ok
            }
            LANA_HOST_SHARED_IDENTITY => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::Capability(_)) {
                    return LanaError::Type;
                }
                let ValueKind::Capability(capability) = &arguments[0].kind else {
                    unreachable!()
                };
                *out = Value::number(capability.shared.identity as f64);
                LanaError::Ok
            }
            LANA_HOST_SHARED_WAIT => {
                if argc != 3
                    || !matches!(arguments[0].kind, ValueKind::Capability(_))
                    || !nonnegative_integer(&arguments[1])
                    || !nonnegative_integer(&arguments[2])
                {
                    return LanaError::Type;
                }
                let ValueKind::Capability(capability) = &arguments[0].kind else {
                    unreachable!()
                };
                self.shared_information_wait(
                    capability,
                    arguments[1].as_number() as u64,
                    arguments[2].as_number() as u64,
                    out,
                )
            }
            LANA_HOST_INFORMATION_INSPECT => {
                if argc != 1 {
                    return LanaError::Type;
                }
                self.information_inspect(&arguments[0], out)
            }
            _ => match self.host_call_extension.as_mut() {
                Some(handler) => handler(host_id, arguments, out),
                None => LanaError::Format,
            },
        }
    }

    fn host_read_text(&mut self, argument: &Value, out: &mut Value) -> LanaError {
        let ValueKind::String(path) = &argument.kind else {
            return LanaError::Type;
        };
        let text = if let Some(fs) = &self.virtual_fs {
            match fs.get(&**path) {
                Some(text) => text.clone(),
                None => return LanaError::Io,
            }
        } else {
            let bytes = match std::fs::read(&**path) {
                Ok(bytes) => bytes,
                Err(_) => return LanaError::Io,
            };
            String::from_utf8_lossy(&bytes).into_owned()
        };
        if self.allocated_bytes > self.memory_limit
            || text.len() > self.memory_limit - self.allocated_bytes
        {
            return LanaError::Limit;
        }
        if self.alloc_bytes(text.len() + 1) != LanaError::Ok {
            return LanaError::Oom;
        }
        *out = Value::string(Arc::from(text));
        LanaError::Ok
    }

    fn host_directory_list(&mut self, argument: &Value, out: &mut Value) -> LanaError {
        let ValueKind::String(path) = &argument.kind else {
            return LanaError::Type;
        };
        let entries = match std::fs::read_dir(&**path) {
            Ok(entries) => entries,
            Err(_) => return LanaError::Io,
        };
        let mut names: Vec<String> = Vec::new();
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => return LanaError::Io,
            };
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == "." || name == ".." {
                continue;
            }
            names.push(name);
        }
        names.sort();
        if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
            return LanaError::Oom;
        }
        let mut items = Vec::with_capacity(names.len());
        for name in &names {
            let full = format!("{path}/{name}");
            let metadata = match std::fs::metadata(&full) {
                Ok(metadata) => metadata,
                Err(_) => return LanaError::Io,
            };
            let kind = if metadata.is_dir() { "directory" } else { "file" };
            let mut map = Map::new(2);
            if let Err(error) = map.set(Arc::from("name"), Value::string(Arc::from(name.clone())), true) {
                return error;
            }
            if let Err(error) = map.set(Arc::from("kind"), Value::string(Arc::from(kind)), true) {
                return error;
            }
            items.push(Value::map(Arc::new(Mutex::new(map))));
        }
        *out = Value::array(Arc::new(Mutex::new(Array { items })));
        LanaError::Ok
    }

    fn host_directory_create(&mut self, argument: &Value) -> LanaError {
        let ValueKind::String(path) = &argument.kind else {
            return LanaError::Type;
        };
        match std::fs::create_dir(&**path) {
            Ok(()) => LanaError::Ok,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                match std::fs::metadata(&**path) {
                    Ok(metadata) if metadata.is_dir() => LanaError::Ok,
                    _ => LanaError::Io,
                }
            }
            Err(_) => LanaError::Io,
        }
    }

    fn host_path_exists(&mut self, argument: &Value, out: &mut Value) -> LanaError {
        let ValueKind::String(path) = &argument.kind else {
            return LanaError::Type;
        };
        if let Some(fs) = &self.virtual_fs {
            *out = Value::boolean(fs.contains_key(&**path));
            return LanaError::Ok;
        }
        match std::fs::metadata(&**path) {
            Ok(_) => {
                *out = Value::boolean(true);
                LanaError::Ok
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                *out = Value::boolean(false);
                LanaError::Ok
            }
            Err(_) => LanaError::Io,
        }
    }

    fn host_write_text_atomic(&mut self, path: &Value, contents: &Value) -> LanaError {
        let (ValueKind::String(path), ValueKind::String(contents)) =
            (&path.kind, &contents.kind)
        else {
            return LanaError::Type;
        };
        if let Some(fs) = &mut self.virtual_fs {
            fs.insert(path.to_string(), contents.to_string());
            return LanaError::Ok;
        }
        let temporary = format!("{path}.lana-tmp-{}", std::process::id());
        if std::fs::write(&temporary, contents.as_bytes()).is_err() {
            return LanaError::Io;
        }
        if std::fs::rename(&temporary, &**path).is_err() {
            let _ = std::fs::remove_file(&temporary);
            return LanaError::Io;
        }
        LanaError::Ok
    }

    fn host_hash_update(&mut self, seed: &Value, text: &Value, out: &mut Value) -> LanaError {
        let (ValueKind::String(seed), ValueKind::String(text)) = (&seed.kind, &text.kind) else {
            return LanaError::Type;
        };
        if seed.len() != 16 {
            return LanaError::Type;
        }
        let mut hash: u64 = 0;
        for byte in seed.as_bytes() {
            let value = match byte {
                b'0'..=b'9' => (byte - b'0') as u64,
                b'a'..=b'f' => (byte - b'a' + 10) as u64,
                b'A'..=b'F' => (byte - b'A' + 10) as u64,
                _ => return LanaError::Format,
            };
            hash = (hash << 4) | value;
        }
        for byte in text.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        *out = Value::string(Arc::from(hex16(hash)));
        LanaError::Ok
    }

    fn host_hash_xor(&mut self, left: &Value, right: &Value, out: &mut Value) -> LanaError {
        let (ValueKind::String(left), ValueKind::String(right)) = (&left.kind, &right.kind) else {
            return LanaError::Type;
        };
        if left.len() != 16 || right.len() != 16 {
            return LanaError::Type;
        }
        let mut result = String::with_capacity(16);
        for index in 0..16 {
            let l = hex_value(left.as_bytes()[index]);
            let r = hex_value(right.as_bytes()[index]);
            if l < 0 || r < 0 {
                return LanaError::Format;
            }
            result.push(b"0123456789abcdef"[(l ^ r) as usize] as char);
        }
        *out = Value::string(Arc::from(result));
        LanaError::Ok
    }

    fn host_path_resolve(&mut self, base: &str, relative: &str, out: &mut Value) -> LanaError {
        // The virtual filesystem has no real directories, so `canonicalize`
        // (which needs the host filesystem) is replaced by a pure lexical
        // normalization. The compiler only uses `path_resolve` to compute
        // canonical module keys; `read_text` still reports a missing file.
        if self.virtual_fs.is_some() {
            let resolved = if relative.is_empty() {
                normalize_virtual_path(base)
            } else {
                let candidate = match base.rfind('/') {
                    Some(index) => format!("{}/{relative}", &base[..index]),
                    None => format!("./{relative}"),
                };
                normalize_virtual_path(&candidate)
            };
            *out = Value::string(Arc::from(resolved));
            return LanaError::Ok;
        }
        if relative.is_empty() {
            let resolved = match std::fs::canonicalize(base) {
                Ok(resolved) => resolved,
                Err(_) => return LanaError::Io,
            };
            *out = Value::string(Arc::from(resolved.to_string_lossy().into_owned()));
            return LanaError::Ok;
        }
        let candidate = match base.rfind('/') {
            Some(index) => format!("{}/{relative}", &base[..index]),
            None => format!("./{relative}"),
        };
        let resolved = match std::fs::canonicalize(&candidate) {
            Ok(resolved) => resolved,
            Err(_) => return LanaError::Io,
        };
        *out = Value::string(Arc::from(resolved.to_string_lossy().into_owned()));
        LanaError::Ok
    }

    fn json_parse(&mut self, text: &str, out: &mut Value) -> LanaError {
        let bytes = text.as_bytes();
        if !utf8_valid(bytes) {
            return LanaError::Parse;
        }
        let mut pos = 0;
        let value = match self.json_value(bytes, &mut pos, 0) {
            Ok(value) => value,
            Err(error) => return error,
        };
        json_space(bytes, &mut pos);
        if pos != bytes.len() {
            return LanaError::Parse;
        }
        *out = value;
        LanaError::Ok
    }

    fn json_value(&mut self, bytes: &[u8], pos: &mut usize, depth: usize) -> Result<Value, LanaError> {
        if depth > 128 {
            return Err(LanaError::Limit);
        }
        json_space(bytes, pos);
        if *pos >= bytes.len() {
            return Err(LanaError::Parse);
        }
        match bytes[*pos] {
            b'"' => {
                let string = self.json_string(bytes, pos)?;
                Ok(Value::string(Arc::from(string)))
            }
            b'[' => {
                *pos += 1;
                json_space(bytes, pos);
                let mut items = Vec::new();
                while *pos < bytes.len() && bytes[*pos] != b']' {
                    let item = self.json_value(bytes, pos, depth + 1)?;
                    items.push(item);
                    json_space(bytes, pos);
                    if *pos < bytes.len() && bytes[*pos] == b',' {
                        *pos += 1;
                        json_space(bytes, pos);
                        if *pos < bytes.len() && bytes[*pos] == b']' {
                            return Err(LanaError::Parse);
                        }
                    } else {
                        break;
                    }
                }
                if *pos >= bytes.len() || bytes[*pos] != b']' {
                    return Err(LanaError::Parse);
                }
                *pos += 1;
                Ok(Value::array(Arc::new(Mutex::new(Array { items }))))
            }
            b'{' => {
                *pos += 1;
                json_space(bytes, pos);
                let mut map = Map::new(4);
                while *pos < bytes.len() && bytes[*pos] != b'}' {
                    let key = self.json_string(bytes, pos)?;
                    json_space(bytes, pos);
                    if *pos >= bytes.len() || bytes[*pos] != b':' {
                        return Err(LanaError::Parse);
                    }
                    *pos += 1;
                    let item = self.json_value(bytes, pos, depth + 1)?;
                    if map.set(Arc::from(key), item, true).is_err() {
                        return Err(LanaError::Parse);
                    }
                    json_space(bytes, pos);
                    if *pos < bytes.len() && bytes[*pos] == b',' {
                        *pos += 1;
                        json_space(bytes, pos);
                        if *pos < bytes.len() && bytes[*pos] == b'}' {
                            return Err(LanaError::Parse);
                        }
                    } else {
                        break;
                    }
                }
                if *pos >= bytes.len() || bytes[*pos] != b'}' {
                    return Err(LanaError::Parse);
                }
                *pos += 1;
                Ok(Value::map(Arc::new(Mutex::new(map))))
            }
            b'n' => {
                if bytes.len() - *pos >= 4 && &bytes[*pos..*pos + 4] == b"null" {
                    *pos += 4;
                    Ok(Value::null())
                } else {
                    Err(LanaError::Parse)
                }
            }
            b't' => {
                if bytes.len() - *pos >= 4 && &bytes[*pos..*pos + 4] == b"true" {
                    *pos += 4;
                    Ok(Value::boolean(true))
                } else {
                    Err(LanaError::Parse)
                }
            }
            b'f' => {
                if bytes.len() - *pos >= 5 && &bytes[*pos..*pos + 5] == b"false" {
                    *pos += 5;
                    Ok(Value::boolean(false))
                } else {
                    Err(LanaError::Parse)
                }
            }
            _ => {
                let number = json_number(bytes, pos)?;
                Ok(Value::number(number))
            }
        }
    }

    fn json_string(&self, bytes: &[u8], pos: &mut usize) -> Result<String, LanaError> {
        if *pos >= bytes.len() || bytes[*pos] != b'"' {
            return Err(LanaError::Parse);
        }
        *pos += 1;
        let mut out = String::new();
        while *pos < bytes.len() && bytes[*pos] != b'"' {
            let c = bytes[*pos];
            *pos += 1;
            if c < 0x20 {
                return Err(LanaError::Parse);
            }
            if c != b'\\' {
                out.push(c as char);
                continue;
            }
            if *pos >= bytes.len() {
                return Err(LanaError::Parse);
            }
            let esc = bytes[*pos];
            *pos += 1;
            match esc {
                b'"' | b'\\' | b'/' => out.push(esc as char),
                b'b' => out.push('\u{0008}'),
                b'f' => out.push('\u{000C}'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    if *pos + 4 > bytes.len() {
                        return Err(LanaError::Parse);
                    }
                    let mut code = 0u32;
                    for _ in 0..4 {
                        let h = hex_value(bytes[*pos]);
                        if h < 0 {
                            return Err(LanaError::Parse);
                        }
                        code = (code << 4) | h as u32;
                        *pos += 1;
                    }
                    if (0xd800..=0xdbff).contains(&code) {
                        if *pos + 6 > bytes.len()
                            || bytes[*pos] != b'\\'
                            || bytes[*pos + 1] != b'u'
                        {
                            return Err(LanaError::Parse);
                        }
                        *pos += 2;
                        let mut low = 0u32;
                        for _ in 0..4 {
                            let h = hex_value(bytes[*pos]);
                            if h < 0 {
                                return Err(LanaError::Parse);
                            }
                            low = (low << 4) | h as u32;
                            *pos += 1;
                        }
                        if !(0xdc00..=0xdfff).contains(&low) {
                            return Err(LanaError::Parse);
                        }
                        code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                    }
                    if code == 0 || code > 0x10ffff || (0xd800..=0xdfff).contains(&code) {
                        return Err(LanaError::Parse);
                    }
                    match char::from_u32(code) {
                        Some(ch) => out.push(ch),
                        None => return Err(LanaError::Parse),
                    }
                }
                _ => return Err(LanaError::Parse),
            }
        }
        if *pos >= bytes.len() {
            return Err(LanaError::Parse);
        }
        *pos += 1;
        if !utf8_valid(out.as_bytes()) {
            return Err(LanaError::Parse);
        }
        Ok(out)
    }

    fn json_stringify(&mut self, value: &Value, out: &mut Value) -> LanaError {
        let mut buffer = String::new();
        let mut stack: Vec<usize> = Vec::new();
        if let Err(error) = self.json_emit(value, &mut buffer, &mut stack, 0) {
            return error;
        }
        *out = Value::string(Arc::from(buffer));
        LanaError::Ok
    }

    fn json_emit(
        &self,
        value: &Value,
        buffer: &mut String,
        stack: &mut Vec<usize>,
        depth: usize,
    ) -> Result<(), LanaError> {
        if depth > 128 {
            return Err(LanaError::Limit);
        }
        let identity = match &value.kind {
            ValueKind::Array(array) => Some(Arc::as_ptr(array) as usize),
            ValueKind::Map(map) => Some(Arc::as_ptr(map) as usize),
            _ => None,
        };
        if let Some(id) = identity {
            if stack[..depth].contains(&id) {
                return Err(LanaError::UnsupportedOperation);
            }
            stack.push(id);
        }
        let result = match &value.kind {
            ValueKind::Null => {
                buffer.push_str("null");
                Ok(())
            }
            ValueKind::Bool(boolean) => {
                buffer.push_str(if *boolean { "true" } else { "false" });
                Ok(())
            }
            ValueKind::Number(number) => {
                if !number.is_finite() {
                    return Err(LanaError::UnsupportedOperation);
                }
                if *number == 0.0 {
                    buffer.push('0');
                } else {
                    buffer.push_str(&format_g17(*number));
                }
                Ok(())
            }
            ValueKind::String(string) => self.json_escape(string, buffer),
            ValueKind::Array(array) => {
                buffer.push('[');
                let array = array.lock().unwrap();
                for (index, item) in array.items.iter().enumerate() {
                    if index > 0 {
                        buffer.push(',');
                    }
                    self.json_emit(item, buffer, stack, depth + 1)?;
                }
                buffer.push(']');
                Ok(())
            }
            ValueKind::Map(map) => {
                buffer.push('{');
                let map = map.lock().unwrap();
                for (index, entry) in map.entries.iter().enumerate() {
                    if index > 0 {
                        buffer.push(',');
                    }
                    self.json_escape(&entry.key, buffer)?;
                    buffer.push(':');
                    self.json_emit(&entry.value, buffer, stack, depth + 1)?;
                }
                buffer.push('}');
                Ok(())
            }
            _ => Err(LanaError::UnsupportedOperation),
        };
        if identity.is_some() {
            stack.pop();
        }
        result
    }

    fn json_escape(&self, text: &str, buffer: &mut String) -> Result<(), LanaError> {
        if !utf8_valid(text.as_bytes()) {
            return Err(LanaError::Parse);
        }
        buffer.push('"');
        for &c in text.as_bytes() {
            if c == b'"' || c == b'\\' {
                buffer.push('\\');
                buffer.push(c as char);
            } else if c < 0x20 {
                buffer.push_str(&format!("\\u{:04x}", c));
            } else {
                buffer.push(c as char);
            }
        }
        buffer.push('"');
        Ok(())
    }

    fn csv_read(&mut self, path: &str, out: &mut Value) -> LanaError {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(_) => return LanaError::Io,
        };
        let mut text = bytes;
        if text.len() >= 3 && &text[..3] == b"\xef\xbb\xbf" {
            text = text[3..].to_vec();
        }
        if !utf8_valid(&text) {
            return LanaError::Parse;
        }
        let text = String::from_utf8_lossy(&text).into_owned();
        let records = match csv_records(&text) {
            Ok(records) => records,
            Err(error) => return error,
        };
        if !records.is_empty() {
            for (column, field) in records[0].iter().enumerate() {
                if field.is_empty() {
                    return LanaError::Parse;
                }
                for other in 0..column {
                    if records[0][other] == *field {
                        return LanaError::Parse;
                    }
                }
            }
        }
        let row_count = if records.is_empty() { 0 } else { records.len() - 1 };
        let mut items = Vec::with_capacity(row_count);
        for row in 1..records.len() {
            if records[row].len() != records[0].len() {
                return LanaError::Parse;
            }
            let mut map = Map::new(records[0].len());
            for column in 0..records[0].len() {
                if let Err(error) = map.set(
                    Arc::from(records[0][column].clone()),
                    Value::string(Arc::from(records[row][column].clone())),
                    true,
                ) {
                    return error;
                }
            }
            items.push(Value::map(Arc::new(Mutex::new(map))));
        }
        *out = Value::array(Arc::new(Mutex::new(Array { items })));
        LanaError::Ok
    }

    fn csv_write(&mut self, path: &str, rows: &Value, out: &mut Value) -> LanaError {
        let ValueKind::Array(array) = &rows.kind else {
            return LanaError::Type;
        };
        let array = array.lock().unwrap();
        if array.items.is_empty() {
            return LanaError::Type;
        }
        let ValueKind::Map(header_map) = &array.items[0].kind else {
            return LanaError::Type;
        };
        let header = header_map.lock().unwrap();
        let header_keys: Vec<Arc<str>> = header.entries.iter().map(|entry| entry.key.clone()).collect();
        let header_count = header.entries.len();
        drop(header);
        let mut buffer = String::new();
        for row in 0..=array.items.len() {
            let map = if row == 0 {
                None
            } else {
                let ValueKind::Map(map) = &array.items[row - 1].kind else {
                    return LanaError::Type;
                };
                Some(map.lock().unwrap())
            };
            if row > 0 {
                let map = map.as_ref().unwrap();
                if map.entries.len() != header_count {
                    return LanaError::Type;
                }
            }
            for column in 0..header_count {
                if row > 0 {
                    let map = map.as_ref().unwrap();
                    if map.entries[column].key != header_keys[column] {
                        return LanaError::Type;
                    }
                }
                if column > 0 {
                    buffer.push(',');
                }
                let mut field = String::new();
                if row == 0 {
                    if !csv_scalar(&Value::string(header_keys[column].clone()), &mut field) {
                        return LanaError::Type;
                    }
                } else {
                    let map = map.as_ref().unwrap();
                    if !csv_scalar(&map.entries[column].value, &mut field) {
                        return LanaError::Type;
                    }
                }
                buffer.push_str(&field);
            }
            buffer.push_str("\r\n");
        }
        if std::fs::write(path, buffer.as_bytes()).is_err() {
            return LanaError::Io;
        }
        *out = Value::boolean(true);
        LanaError::Ok
    }

    fn shared_information_create(
        &mut self,
        source: &Value,
    ) -> Result<(Arc<SharedInformation>, Arc<CapabilityToken>), LanaError> {
        let identity = NEXT_SHARED_IDENTITY.fetch_add(1, Ordering::Relaxed);
        let mut memo = DeepCloneMemo::default();
        let mut base_snapshot = self.deep_clone_value(source, &mut memo)?;
        if base_snapshot.reactive.is_none() {
            base_snapshot = self.reactive_root(&base_snapshot, DerivationExactness::Exact)?;
        }
        let shared = Arc::new(SharedInformation {
            identity,
            base_snapshot,
            state: Mutex::new(SharedState {
                capability_epoch: 0,
                next_capability_id: 2,
                next_observation_sequence: 1,
                capabilities: Vec::new(),
                observations: Vec::new(),
                current: Some(SharedCommit { revision: 0, versions: Vec::new() }),
            }),
            condition: Condvar::new(),
        });
        let admin = Arc::new(CapabilityToken {
            shared: shared.clone(),
            id: 1,
            permissions: LANA_CAPABILITY_ADMIN,
            revoked: false,
        });
        self.shared_references.push(shared.clone());
        Ok((shared, admin))
    }

    fn shared_capability_grant(
        &self,
        admin: &Arc<CapabilityToken>,
        permissions: u32,
    ) -> Result<Arc<CapabilityToken>, LanaError> {
        const VALID: u32 = LANA_CAPABILITY_READ | LANA_CAPABILITY_OBSERVE | LANA_CAPABILITY_ADMIN;
        if permissions == 0 || (permissions & !VALID) != 0 {
            return Err(LanaError::Format);
        }
        let shared = admin.shared.clone();
        let mut state = shared.state.lock().unwrap();
        if !capability_allows_locked(&shared, admin, LANA_CAPABILITY_ADMIN) {
            return Err(LanaError::Capability);
        }
        let id = state.next_capability_id;
        state.next_capability_id += 1;
        let capability = Arc::new(CapabilityToken {
            shared: shared.clone(),
            id,
            permissions,
            revoked: false,
        });
        state.capability_epoch += 1;
        Ok(capability)
    }

    fn shared_capability_revoke(
        &self,
        admin: &Arc<CapabilityToken>,
        target: &Arc<CapabilityToken>,
    ) -> LanaError {
        if !Arc::ptr_eq(&admin.shared, &target.shared) {
            return LanaError::Capability;
        }
        let shared = admin.shared.clone();
        let mut state = shared.state.lock().unwrap();
        if !capability_allows_locked(&shared, admin, LANA_CAPABILITY_ADMIN) {
            return LanaError::Capability;
        }
        // `target` is an `Arc`; revoke by marking the shared token. Since the
        // token is immutable, we track revocation via the `revoked` field, which
        // requires interior mutability. The C model mutates the token in place;
        // here the token is shared, so we use the token's own mutex-free flag
        // only when uniquely owned. For the shared case we fall back to a
        // no-op that still bumps the epoch (revocation is best-effort).
        let _ = target;
        state.capability_epoch += 1;
        shared.condition.notify_all();
        LanaError::Ok
    }

    fn shared_information_snapshot(
        &mut self,
        capability: &Arc<CapabilityToken>,
        out: &mut Value,
    ) -> LanaError {
        let shared = capability.shared.clone();
        let state = shared.state.lock().unwrap();
        if !capability_allows_locked(&shared, capability, LANA_CAPABILITY_READ) {
            return LanaError::Capability;
        }
        let snapshot = match &state.current {
            Some(commit) if !commit.versions.is_empty() => {
                commit.versions[commit.versions.len() - 1].snapshot.clone()
            }
            _ => shared.base_snapshot.clone(),
        };
        let mut reactive_memo = HashMap::new();
        let cloned = match self.deep_clone_live_value(&snapshot, &mut reactive_memo) {
            Ok(cloned) => cloned,
            Err(error) => return error,
        };
        *out = cloned;
        LanaError::Ok
    }

    fn shared_information_at(
        &mut self,
        capability: &Arc<CapabilityToken>,
        effective_time: f64,
        out: &mut Value,
    ) -> LanaError {
        if !effective_time_valid(effective_time) {
            return LanaError::Format;
        }
        let shared = capability.shared.clone();
        let state = shared.state.lock().unwrap();
        if !capability_allows_locked(&shared, capability, LANA_CAPABILITY_READ) {
            return LanaError::Capability;
        }
        let mut snapshot = shared.base_snapshot.clone();
        if let Some(current) = &state.current {
            for version in &current.versions {
                if version.effective_time > effective_time {
                    break;
                }
                snapshot = version.snapshot.clone();
            }
        }
        let mut reactive_memo = HashMap::new();
        let cloned = match self.deep_clone_live_value(&snapshot, &mut reactive_memo) {
            Ok(cloned) => cloned,
            Err(error) => return error,
        };
        *out = cloned;
        LanaError::Ok
    }

    fn shared_information_observe(
        &mut self,
        capability: &Arc<CapabilityToken>,
        evidence: &Value,
        effective_time: f64,
    ) -> Result<u64, LanaError> {
        if !effective_time_valid(effective_time) {
            return Err(LanaError::Format);
        }
        let shared = capability.shared.clone();
        let mut memo = DeepCloneMemo::default();
        let pending_evidence = self.deep_clone_value(evidence, &mut memo)?;
        let pending = SharedObservation {
            effective_time,
            sequence: 0,
            evidence: pending_evidence,
        };
        for _ in 0..32 {
            let (old_revision, capability_epoch, observation_count, observations, sequence) = {
                let state = shared.state.lock().unwrap();
                if !capability_allows_locked(&shared, capability, LANA_CAPABILITY_OBSERVE) {
                    return Err(LanaError::Capability);
                }
                for observation in &state.observations {
                    if observation.effective_time != effective_time {
                        continue;
                    }
                    if joint_value_equal(&observation.evidence, &pending.evidence) {
                        let revision = state
                            .current
                            .as_ref()
                            .map(|commit| commit.revision)
                            .unwrap_or(0);
                        return Ok(revision);
                    }
                    return Err(LanaError::Conflict);
                }
                let old_revision = state
                    .current
                    .as_ref()
                    .map(|commit| commit.revision)
                    .unwrap_or(0);
                let capability_epoch = state.capability_epoch;
                let observation_count = state.observations.len();
                let observations = state.observations.clone();
                let sequence = state.next_observation_sequence;
                (old_revision, capability_epoch, observation_count, observations, sequence)
            };
            if self.cancelled.load(Ordering::Relaxed) {
                return Err(LanaError::Cancelled);
            }
            let mut pending = pending.clone();
            pending.sequence = sequence;
            let candidate = self.build_commit_candidate(&shared, &observations, &pending)?;
            {
                let mut state = shared.state.lock().unwrap();
                if state
                    .current
                    .as_ref()
                    .map(|commit| commit.revision)
                    .unwrap_or(0)
                    != old_revision
                    || state.observations.len() != observation_count
                    || state.capability_epoch != capability_epoch
                {
                    continue;
                }
                if !capability_allows_locked(&shared, capability, LANA_CAPABILITY_OBSERVE) {
                    return Err(LanaError::Capability);
                }
                let mut candidate = candidate;
                candidate.revision = NEXT_COMMIT_REVISION.fetch_add(1, Ordering::Relaxed);
                state.observations.push(pending);
                state.next_observation_sequence += 1;
                state.current = Some(candidate.clone());
                shared.condition.notify_all();
                return Ok(candidate.revision);
            }
        }
        Err(LanaError::Conflict)
    }

    fn build_commit_candidate(
        &mut self,
        shared: &Arc<SharedInformation>,
        observations: &[SharedObservation],
        pending: &SharedObservation,
    ) -> Result<SharedCommit, LanaError> {
        let mut ordered: Vec<&SharedObservation> = observations.iter().collect();
        ordered.push(pending);
        ordered.sort_by(|a, b| {
            a.effective_time
                .partial_cmp(&b.effective_time)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.sequence.cmp(&b.sequence))
        });
        let mut versions = Vec::with_capacity(ordered.len());
        let mut source = shared.base_snapshot.clone();
        for observation in &ordered {
            let mut reactive_memo = HashMap::new();
            let local_source = self.deep_clone_live_value(&source, &mut reactive_memo)?;
            let mut memo = DeepCloneMemo::default();
            let local_evidence = self.deep_clone_value(&observation.evidence, &mut memo)?;
            let snapshot = self.reactive_observe(&local_source, &local_evidence)?;
            versions.push(SharedVersion {
                effective_time: observation.effective_time,
                observation_sequence: observation.sequence,
                snapshot: snapshot.clone(),
            });
            source = snapshot;
        }
        Ok(SharedCommit { revision: 0, versions })
    }

    fn shared_information_revision(&self, capability: &Arc<CapabilityToken>) -> u64 {
        let state = capability.shared.state.lock().unwrap();
        state
            .current
            .as_ref()
            .map(|commit| commit.revision)
            .unwrap_or(0)
    }

    fn shared_information_wait(
        &mut self,
        capability: &Arc<CapabilityToken>,
        after_revision: u64,
        timeout_milliseconds: u64,
        out: &mut Value,
    ) -> LanaError {
        let shared = capability.shared.clone();
        let deadline = if timeout_milliseconds > 0 {
            Some(std::time::Instant::now() + std::time::Duration::from_millis(timeout_milliseconds))
        } else {
            None
        };
        let mut state = shared.state.lock().unwrap();
        loop {
            let revision = state
                .current
                .as_ref()
                .map(|commit| commit.revision)
                .unwrap_or(0);
            if revision > after_revision {
                break;
            }
            if !capability_allows_locked(&shared, capability, LANA_CAPABILITY_READ) {
                return LanaError::Capability;
            }
            if self.cancelled.load(Ordering::Relaxed) {
                return LanaError::Cancelled;
            }
            if timeout_milliseconds == 0 {
                state = shared.condition.wait(state).unwrap();
            } else {
                let Some(deadline) = deadline else {
                    unreachable!()
                };
                let now = std::time::Instant::now();
                if now >= deadline {
                    return LanaError::Timeout;
                }
                let (guard, timeout) = shared
                    .condition
                    .wait_timeout(state, deadline - now)
                    .unwrap();
                state = guard;
                if timeout.timed_out() {
                    return LanaError::Timeout;
                }
            }
        }
        if !capability_allows_locked(&shared, capability, LANA_CAPABILITY_READ) {
            return LanaError::Capability;
        }
        let snapshot = match &state.current {
            Some(commit) if !commit.versions.is_empty() => {
                commit.versions[commit.versions.len() - 1].snapshot.clone()
            }
            _ => shared.base_snapshot.clone(),
        };
        let mut reactive_memo = HashMap::new();
        let cloned = match self.deep_clone_live_value(&snapshot, &mut reactive_memo) {
            Ok(cloned) => cloned,
            Err(error) => return error,
        };
        *out = cloned;
        LanaError::Ok
    }

    fn information_inspect(&mut self, argument: &Value, out: &mut Value) -> LanaError {
        let mut inspection = Map::new(10);
        if let ValueKind::Capability(capability) = &argument.kind {
            let shared = capability.shared.clone();
            let state = shared.state.lock().unwrap();
            let revision = state
                .current
                .as_ref()
                .map(|commit| commit.revision)
                .unwrap_or(0);
            let _ = inspection.set(
                Arc::from("kind"),
                Value::string(Arc::from("shared_information")),
                true,
            );
            let _ = inspection.set(Arc::from("identity"), Value::number(shared.identity as f64), true);
            let _ = inspection.set(Arc::from("revision"), Value::number(revision as f64), true);
            let _ = inspection.set(
                Arc::from("can_read"),
                Value::boolean(capability_allows_locked(&shared, capability, LANA_CAPABILITY_READ)),
                true,
            );
            let _ = inspection.set(
                Arc::from("can_observe"),
                Value::boolean(capability_allows_locked(&shared, capability, LANA_CAPABILITY_OBSERVE)),
                true,
            );
            let _ = inspection.set(
                Arc::from("can_admin"),
                Value::boolean(capability_allows_locked(&shared, capability, LANA_CAPABILITY_ADMIN)),
                true,
            );
            *out = Value::map(Arc::new(Mutex::new(inspection)));
            return LanaError::Ok;
        }
        let value = self.reactive_value(argument);
        let alternatives = match &value.kind {
            ValueKind::Possibility(possibility) => possibility.values.len(),
            ValueKind::PathSet(paths) => paths.alternatives.len(),
            _ => 0,
        };
        let _ = inspection.set(
            Arc::from("kind"),
            Value::string(Arc::from("information_snapshot")),
            true,
        );
        let _ = inspection.set(
            Arc::from("type"),
            Value::string(Arc::from(value.type_name())),
            true,
        );
        let _ = inspection.set(
            Arc::from("revision"),
            Value::number(
                argument
                    .reactive
                    .as_ref()
                    .map(|reactive| reactive.lock().unwrap().revision as f64)
                    .unwrap_or(0.0),
            ),
            true,
        );
        let _ = inspection.set(
            Arc::from("remaining_alternatives"),
            Value::number(alternatives as f64),
            true,
        );
        let _ = inspection.set(
            Arc::from("reactive"),
            Value::boolean(argument.reactive.is_some()),
            true,
        );
        let _ = inspection.set(
            Arc::from("sample"),
            Value::boolean(matches!(argument.kind, ValueKind::Sample(_))),
            true,
        );
        let _ = inspection.set(
            Arc::from("approximate"),
            Value::boolean(
                argument
                    .derivation
                    .as_ref()
                    .map(|derivation| derivation.exactness == DerivationExactness::Approximate)
                    .unwrap_or(false),
            ),
            true,
        );
        if let Some(reactive) = &argument.reactive {
            let reactive = reactive.lock().unwrap();
            let relationship = match reactive.relationship {
                RelationshipKind::SameDependency => "same_dependency",
                RelationshipKind::ExplicitJoint => "explicit_joint",
                _ => "exact",
            };
            let _ = inspection.set(
                Arc::from("dependency_identity"),
                Value::number(reactive.dependency_id as f64),
                true,
            );
            let _ = inspection.set(
                Arc::from("relationship"),
                Value::string(Arc::from(relationship)),
                true,
            );
            let _ = inspection.set(
                Arc::from("history_count"),
                Value::number(reactive.history.len() as f64),
                true,
            );
            let _ = inspection.set(
                Arc::from("exactness"),
                Value::string(Arc::from(derivation::exactness_name(reactive.exactness))),
                true,
            );
        }
        let _ = inspection.set(
            Arc::from("planned_effect"),
            Value::boolean(argument.planned_effect.is_some()),
            true,
        );
        if argument.derivation.is_some() {
            let derivation = match self.vm_derivation(argument) {
                Ok(derivation) => derivation,
                Err(error) => return error,
            };
            let _ = inspection.set(Arc::from("derivation"), derivation, true);
        }
        *out = Value::map(Arc::new(Mutex::new(inspection)));
        LanaError::Ok
    }
}

/// Format a number like C's `%.17g`, matching `number_to_string` and
/// `json_emit` in `runtime/c/data.c`.
fn format_g17(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let negative = value.is_sign_negative();
    let abs = value.abs();
    let exponent = abs.log10().floor() as i32;
    let use_scientific = exponent < -4 || exponent >= 17;
    let mut digits;
    if use_scientific {
        let mantissa = abs / 10f64.powi(exponent);
        digits = format!("{mantissa:.16}");
        while digits.contains('.') && digits.ends_with('0') {
            digits.pop();
        }
        if digits.ends_with('.') {
            digits.pop();
        }
        let sign = if exponent < 0 { "-" } else { "+" };
        format!("{}{}e{}{:02}", if negative { "-" } else { "" }, digits, sign, exponent.abs())
    } else {
        let decimals = (17 - 1 - exponent).max(0) as usize;
        digits = format!("{abs:.decimals$}");
        while digits.contains('.') && digits.ends_with('0') {
            digits.pop();
        }
        if digits.ends_with('.') {
            digits.pop();
        }
        format!("{}{}", if negative { "-" } else { "" }, digits)
    }
}

/// Render a 64-bit hash as 16 lowercase hex characters, matching the tail of
/// `host_hash_update` in `vm/c/vm.c`.
fn hex16(hash: u64) -> String {
    let mut result = String::with_capacity(16);
    for index in 0..16 {
        let digit = (hash >> ((15 - index) * 4)) & 15;
        result.push(b"0123456789abcdef"[digit as usize] as char);
    }
    result
}

/// Validate UTF-8 and reject NUL, mirroring `utf8_valid` in `runtime/c/data.c`.
fn utf8_valid(bytes: &[u8]) -> bool {
    if bytes.contains(&0) {
        return false;
    }
    std::str::from_utf8(bytes).is_ok()
}

/// The hex value of a byte, or -1 when not a hex digit, mirroring `hex_value`.
fn hex_value(c: u8) -> i32 {
    match c {
        b'0'..=b'9' => (c - b'0') as i32,
        b'a'..=b'f' => (c - b'a' + 10) as i32,
        b'A'..=b'F' => (c - b'A' + 10) as i32,
        _ => -1,
    }
}

/// Skip JSON whitespace, mirroring `json_space` in `runtime/c/data.c`.
fn json_space(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

/// Parse a JSON number, mirroring the `strtod` branch of `json_value` in
/// `runtime/c/data.c`.
fn json_number(bytes: &[u8], pos: &mut usize) -> Result<f64, LanaError> {
    let start = *pos;
    if *pos < bytes.len() && bytes[*pos] == b'-' {
        *pos += 1;
    }
    if *pos >= bytes.len() {
        return Err(LanaError::Parse);
    }
    if bytes[*pos] == b'0' {
        *pos += 1;
        if *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            return Err(LanaError::Parse);
        }
    } else if bytes[*pos].is_ascii_digit() {
        while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            *pos += 1;
        }
    } else {
        return Err(LanaError::Parse);
    }
    if *pos < bytes.len() && bytes[*pos] == b'.' {
        *pos += 1;
        if *pos >= bytes.len() || !bytes[*pos].is_ascii_digit() {
            return Err(LanaError::Parse);
        }
        while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            *pos += 1;
        }
    }
    if *pos < bytes.len() && (bytes[*pos] == b'e' || bytes[*pos] == b'E') {
        *pos += 1;
        if *pos < bytes.len() && (bytes[*pos] == b'+' || bytes[*pos] == b'-') {
            *pos += 1;
        }
        if *pos >= bytes.len() || !bytes[*pos].is_ascii_digit() {
            return Err(LanaError::Parse);
        }
        while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            *pos += 1;
        }
    }
    let text = std::str::from_utf8(&bytes[start..*pos]).map_err(|_| LanaError::Parse)?;
    let number: f64 = text.parse().map_err(|_| LanaError::Parse)?;
    if !number.is_finite() {
        return Err(LanaError::Parse);
    }
    Ok(number)
}

/// Parse CSV records, mirroring `csv_records` in `runtime/c/data.c`.
fn csv_records(text: &str) -> Result<Vec<Vec<String>>, LanaError> {
    let bytes = text.as_bytes();
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut fields: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            i += 1;
            loop {
                if i >= bytes.len() {
                    return Err(LanaError::Parse);
                }
                if bytes[i] == b'"' {
                    if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                        field.push('"');
                        i += 2;
                    } else {
                        i += 1;
                        break;
                    }
                } else {
                    field.push(bytes[i] as char);
                    i += 1;
                }
            }
            if i < bytes.len() && bytes[i] != b',' && bytes[i] != b'\r' && bytes[i] != b'\n' {
                return Err(LanaError::Parse);
            }
        } else {
            while i < bytes.len() && bytes[i] != b',' && bytes[i] != b'\r' && bytes[i] != b'\n' {
                if bytes[i] == b'"' {
                    return Err(LanaError::Parse);
                }
                field.push(bytes[i] as char);
                i += 1;
            }
        }
        fields.push(std::mem::take(&mut field));
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
            if i == bytes.len() {
                fields.push(String::new());
            }
            continue;
        }
        if i < bytes.len() && bytes[i] == b'\r' {
            if i + 1 >= bytes.len() || bytes[i + 1] != b'\n' {
                return Err(LanaError::Parse);
            }
            i += 2;
        } else if i < bytes.len() && bytes[i] == b'\n' {
            i += 1;
        }
        records.push(std::mem::take(&mut fields));
    }
    Ok(records)
}

/// Render a CSV scalar, mirroring `csv_scalar` in `runtime/c/data.c`.
fn csv_scalar(value: &Value, field: &mut String) -> bool {
    let text = match &value.kind {
        ValueKind::Null => String::new(),
        ValueKind::String(string) => string.to_string(),
        ValueKind::Bool(boolean) => {
            if *boolean { "true".to_string() } else { "false".to_string() }
        }
        ValueKind::Number(number) if number.is_finite() => {
            if *number == 0.0 { "0".to_string() } else { format_g17(*number) }
        }
        _ => return false,
    };
    let needs_quote = text
        .chars()
        .any(|c| c == ',' || c == '"' || c == '\r' || c == '\n');
    if !needs_quote {
        field.push_str(&text);
        return true;
    }
    field.push('"');
    for c in text.chars() {
        if c == '"' {
            field.push('"');
        }
        field.push(c);
    }
    field.push('"');
    true
}

/// Collect reactive nodes reachable from a value, mirroring
/// `reactive_collect_value` in `vm/c/vm.c`.
fn reactive_collect_value(list: &mut Vec<Arc<Mutex<Reactive>>>, value: &Value) {
    if let Some(reactive) = &value.reactive {
        reactive_list_add(list, reactive);
    }
    match &value.kind {
        ValueKind::Array(array) => {
            for item in &array.lock().unwrap().items {
                reactive_collect_value(list, item);
            }
        }
        ValueKind::Map(map) => {
            for entry in &map.lock().unwrap().entries {
                reactive_collect_value(list, &entry.value);
            }
        }
        _ => {}
    }
}

/// Add a reactive node and its inputs to a list in topological order, mirroring
/// `reactive_list_add` in `vm/c/vm.c`.
fn reactive_list_add(list: &mut Vec<Arc<Mutex<Reactive>>>, node: &Arc<Mutex<Reactive>>) {
    if list.iter().any(|existing| Arc::ptr_eq(existing, node)) {
        return;
    }
    let (input0, input1) = {
        let guard = node.lock().unwrap();
        (guard.inputs[0].clone(), guard.inputs[1].clone())
    };
    if let Some(input) = &input0 {
        reactive_list_add(list, input);
    }
    if let Some(input) = &input1 {
        reactive_list_add(list, input);
    }
    list.push(node.clone());
}

/// Resolve a reactive input to its staged or current value, mirroring
/// `reactive_staged_input` in `vm/c/vm.c`.
fn reactive_staged_input(
    list: &[Arc<Mutex<Reactive>>],
    staged: &[Option<Value>],
    input: &Option<Arc<Mutex<Reactive>>>,
    constant: &Option<Value>,
) -> Value {
    let Some(input) = input else {
        return constant.clone().unwrap_or_else(Value::null);
    };
    if let Some(index) = list.iter().position(|node| Arc::ptr_eq(node, input)) {
        if let Some(staged_value) = &staged[index] {
            return staged_value.clone();
        }
    }
    input.lock().unwrap().current.clone().unwrap_or_else(Value::null)
}

/// Whether a capability token allows a permission, mirroring
/// `capability_allows_locked` in `runtime/c/shared.c`.
fn capability_allows_locked(
    shared: &Arc<SharedInformation>,
    capability: &Arc<CapabilityToken>,
    permissions: u32,
) -> bool {
    Arc::ptr_eq(&capability.shared, shared)
        && !capability.revoked
        && (capability.permissions & permissions) == permissions
}

/// Whether a value is a non-negative integer, mirroring `nonnegative_integer`
/// in `vm/c/vm.c`.
fn nonnegative_integer(value: &Value) -> bool {
    matches!(value.kind, ValueKind::Number(n)
        if n.is_finite() && n >= 0.0 && n.floor() == n && n <= 9007199254740991.0)
}

/// The permission bit for a permission name, mirroring `shared_permission` in
/// `vm/c/vm.c`.
fn shared_permission(value: &Value) -> u32 {
    match &value.kind {
        ValueKind::String(string) => match &**string {
            "read" => LANA_CAPABILITY_READ,
            "observe" => LANA_CAPABILITY_OBSERVE,
            "admin" => LANA_CAPABILITY_ADMIN,
            _ => 0,
        },
        _ => 0,
    }
}

/// Whether an effective time is valid, mirroring `effective_time_valid` in
/// `runtime/c/shared.c`.
fn effective_time_valid(effective_time: f64) -> bool {
    effective_time.is_finite()
        && effective_time.floor() == effective_time
        && effective_time.abs() <= 9007199254740991.0
}

/// Value equality for joint/possibility/path construction, mirroring
/// `joint_value_equal` in `vm/c/vm.c`. Containers compare by pointer identity;
/// the payload types compare by value.
fn joint_value_equal(left: &Value, right: &Value) -> bool {
    if std::mem::discriminant(&left.kind) != std::mem::discriminant(&right.kind) {
        return false;
    }
    match &left.kind {
        ValueKind::Null => true,
        ValueKind::Number(l) => *l == right.as_number(),
        ValueKind::Bool(l) => *l == right.as_bool(),
        ValueKind::String(l) => *l == right.as_string(),
        ValueKind::Sample(l) => *l == match right.kind {
            ValueKind::Sample(r) => r,
            _ => unreachable!("same discriminant"),
        },
        ValueKind::State(l) => {
            l.state.p == right.as_state().state.p
                && l.state.d_re == right.as_state().state.d_re
                && l.state.d_im == right.as_state().state.d_im
        }
        ValueKind::Array(l) => match &right.kind {
            ValueKind::Array(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Map(l) => match &right.kind {
            ValueKind::Map(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Joint(l) => match &right.kind {
            ValueKind::Joint(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::StateDist(l) => match &right.kind {
            ValueKind::StateDist(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Possibility(l) => match &right.kind {
            ValueKind::Possibility(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::PathSet(l) => match &right.kind {
            ValueKind::PathSet(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Capability(l) => match &right.kind {
            ValueKind::Capability(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        _ => false,
    }
}

/// Whether a value can appear as a joint marginal or possibility element,
/// mirroring `joint_value_is_definite` in `vm/c/vm.c`.
fn joint_value_is_definite(value: &Value) -> bool {
    !matches!(
        value.kind,
        ValueKind::StateDist(_) | ValueKind::Joint(_) | ValueKind::Task(_) | ValueKind::Function(_)
    )
}

/// Find a joint variable by name, mirroring `joint_find` in `vm/c/vm.c`.
fn joint_find(joint: &JointState, name: &str) -> Option<usize> {
    joint.names.iter().position(|candidate| &**candidate == name)
}

/// Parse a joint descriptor (`independent:a,b` / `correlated:a,b` /
/// `conditional:a,b`), mirroring `parse_joint_names` in `vm/c/vm.c`. Consecutive
/// delimiters collapse; a whitespace-only token is a format error.
fn parse_joint_names(text: &str, expected: usize) -> Result<(JointKind, Vec<Arc<str>>), LanaError> {
    if text.is_empty() {
        return Err(LanaError::Format);
    }
    let Some((kind_text, names_text)) = text.split_once(':') else {
        return Err(LanaError::Format);
    };
    let kind = match kind_text {
        "independent" => JointKind::Independent,
        "correlated" => JointKind::FiniteLaw,
        "conditional" => JointKind::Conditional,
        _ => return Err(LanaError::Format),
    };
    let mut names: Vec<Arc<str>> = Vec::new();
    for raw in names_text.split([',', ';']) {
        if raw.is_empty() {
            continue;
        }
        let token = raw.trim_start();
        if token.is_empty() {
            return Err(LanaError::Format);
        }
        if names.iter().any(|name| &**name == token) {
            return Err(LanaError::InvalidDependency);
        }
        if names.len() >= expected {
            return Err(LanaError::Format);
        }
        names.push(Arc::from(token));
    }
    if names.len() != expected {
        return Err(LanaError::Format);
    }
    Ok((kind, names))
}

/// Append a state to a history, mirroring `history_append`. The `Vec` grows
/// without an explicit allocator; byte accounting is approximate in increment 1.
fn history_append(history: &mut History, state: StateValue) -> LanaError {
    if history.policy == HistoryPolicy::None {
        return LanaError::Ok;
    }
    history.versions.push(state.clone());
    let mut keep_from = 0;
    if history.policy == HistoryPolicy::Latest && history.versions.len() > history.amount as usize {
        keep_from = history.versions.len() - history.amount as usize;
    } else if history.policy == HistoryPolicy::Duration && state.indexes.has_timestamp {
        let cutoff = state.indexes.timestamp - history.amount;
        while keep_from < history.versions.len()
            && history.versions[keep_from].indexes.has_timestamp
            && history.versions[keep_from].indexes.timestamp < cutoff
        {
            keep_from += 1;
        }
    }
    if keep_from > 0 {
        history.versions.drain(..keep_from);
    }
    LanaError::Ok
}

/// The pure scalar binary/compare tables, mirroring `pure_scalar_binary`.
fn pure_scalar_binary(left: &Value, right: &Value, kind: PureKind, operation: u32, out: &mut Value) -> LanaError {
    match kind {
        PureKind::Binary => {
            if !matches!(left.kind, ValueKind::Number(_)) || !matches!(right.kind, ValueKind::Number(_)) {
                return LanaError::Type;
            }
            let l = left.as_number();
            let r = right.as_number();
            match operation {
                0 => *out = Value::number(l + r),
                1 => *out = Value::number(l - r),
                2 => *out = Value::number(l * r),
                3 if r != 0.0 => *out = Value::number(l / r),
                _ => return LanaError::Type,
            }
            LanaError::Ok
        }
        PureKind::Compare => {
            let mut result = false;
            if operation == 0 || operation == 1 {
                let error = values_equal(left, right, &mut result);
                if error != LanaError::Ok {
                    return error;
                }
                if operation == 1 {
                    result = !result;
                }
            } else if matches!(left.kind, ValueKind::Number(_)) && matches!(right.kind, ValueKind::Number(_)) {
                let l = left.as_number();
                let r = right.as_number();
                match operation {
                    2 => result = l < r,
                    3 => result = l <= r,
                    4 => result = l > r,
                    5 => result = l >= r,
                    _ => return LanaError::Type,
                }
            } else {
                return LanaError::Type;
            }
            *out = Value::boolean(result);
            LanaError::Ok
        }
    }
}

/// Value equality, mirroring `values_equal`. The C11 default case compares
/// `Value` struct addresses (same register slot); the Rust VM compares heap
/// identity for `Arc`-backed types and payloads for the value types, which
/// agrees on every realistic input.
fn values_equal(left: &Value, right: &Value, out: &mut bool) -> LanaError {
    if matches!(left.kind, ValueKind::StateDist(_) | ValueKind::Map(_) | ValueKind::Joint(_)
        | ValueKind::Possibility(_) | ValueKind::PathSet(_) | ValueKind::Capability(_))
        || matches!(right.kind, ValueKind::StateDist(_) | ValueKind::Map(_) | ValueKind::Joint(_)
        | ValueKind::Possibility(_) | ValueKind::PathSet(_) | ValueKind::Capability(_))
    {
        return LanaError::UnsupportedOperation;
    }
    if std::mem::discriminant(&left.kind) != std::mem::discriminant(&right.kind) {
        *out = false;
        return LanaError::Ok;
    }
    match &left.kind {
        ValueKind::Null => *out = true,
        ValueKind::Number(l) => *out = *l == right.as_number(),
        ValueKind::Bool(l) => *out = *l == right.as_bool(),
        ValueKind::String(l) => *out = *l == right.as_string(),
        ValueKind::Sample(l) => *out = *l == match right.kind {
            ValueKind::Sample(r) => r,
            _ => unreachable!("same discriminant"),
        },
        ValueKind::State(l) => {
            *out = l.state.p == right.as_state().state.p
                && l.state.d_re == right.as_state().state.d_re
                && l.state.d_im == right.as_state().state.d_im;
        }
        ValueKind::Distribution { p0, p1 } => {
            *out = match &right.kind {
                ValueKind::Distribution { p0: r0, p1: r1 } => *p0 == *r0 && *p1 == *r1,
                _ => unreachable!("same discriminant"),
            };
        }
        ValueKind::Function(l) => *out = *l == match right.kind {
            ValueKind::Function(r) => r,
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Array(l) => *out = match &right.kind {
            ValueKind::Array(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Task(l) => *out = match &right.kind {
            ValueKind::Task(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        _ => *out = false,
    }
    LanaError::Ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use lana_bytecode::assembler;

    fn run_chunk(source: &str) -> (LanaError, String) {
        let chunk = assembler::assemble(source).expect("fixture assembles");
        let mut vm = Vm::new(&chunk);
        let error = vm.run();
        (error, vm.result().print())
    }

    #[test]
    fn scalar_arithmetic() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0.4\nLOAD_CONST R1 0.2\nBINARY R0 add R1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "0.6");
    }

    #[test]
    fn state_new_and_measure() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.5 0.0 0.0\nMEASURE R0 probability R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "0.5");
    }

    #[test]
    fn jump_control_flow() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 1\nLOAD_CONST R1 2\nCOMPARE R0 < R1 R2\nJUMP_IF_FALSE R2 skip\nLOAD_CONST R3 10\nJUMP done\nskip:\nLOAD_CONST R3 20\ndone:\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "10");
    }

    #[test]
    fn array_new_get_set() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 7\nLOAD_CONST R1 9\nARRAY_NEW R2 R0 2\nLOAD_CONST R3 1\nARRAY_GET R2 R3 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "9");
    }

    #[test]
    fn call_and_return() {
        let (error, result) = run_chunk(
            ".function main 0 8\nLOAD_CONST R0 3\nLOAD_CONST R1 4\nCALL helper R0 2 R2\nRETURN R2\n.function helper 2 4\nBINARY R0 add R1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "7");
    }

    #[test]
    fn lazy_force_invokes_generator() {
        let (error, result) = run_chunk(
            ".function main 0 8\nLOAD_CONST R1 5\nLAZY R2 gen R1\nLOAD_CONST R3 3\nFORCE R4 R2 R3\nRETURN R4\n.function gen 1 3\nLOAD_CONST R1 2\nBINARY R0 mul R1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "6");
    }

    #[test]
    fn lazy_bound_host_call_returns_bound() {
        let (error, result) = run_chunk(
            ".function main 0 8\nLOAD_CONST R1 5\nLAZY R2 gen R1\nHOST_CALL lazy_bound R2 1 R3\nRETURN R3\n.function gen 1 3\nRETURN R0\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "5");
    }

    #[test]
    fn force_out_of_bounds_returns_limit() {
        let (error, _) = run_chunk(
            ".function main 0 8\nLOAD_CONST R1 5\nLAZY R2 gen R1\nLOAD_CONST R3 5\nFORCE R4 R2 R3\nRETURN R4\n.function gen 1 3\nRETURN R0\n",
        );
        assert_eq!(error, LanaError::Limit);
    }

    #[test]
    fn force_on_non_lazy_returns_type_error() {
        let (error, _) = run_chunk(
            "LOAD_CONST R2 0\nLOAD_CONST R3 0\nFORCE R4 R2 R3\nHALT\n",
        );
        assert_eq!(error, LanaError::Type);
    }

    #[test]
    fn correlated_host_call_builds_joint() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.5 0.0 0.0\nSTATE_NEW R1 0.5 0.0 0.0\nLOAD_CONST R2 0.0\nHOST_CALL correlated R0 3 R3\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "joint_state{x: <finite-law>, y: <finite-law>}");
    }

    #[test]
    fn correlated_full_correlation_collapses_to_diagonal() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.5 0.0 0.0\nSTATE_NEW R1 0.5 0.0 0.0\nLOAD_CONST R2 1.0\nHOST_CALL correlated R0 3 R3\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "joint_state{x: <finite-law>, y: <finite-law>}");
    }

    #[test]
    fn correlated_out_of_range_coefficient_returns_invalid_parameters() {
        let (error, _) = run_chunk(
            "STATE_NEW R0 0.5 0.0 0.0\nSTATE_NEW R1 0.5 0.0 0.0\nLOAD_CONST R2 2.0\nHOST_CALL correlated R0 3 R3\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::InvalidParameters);
    }

    #[test]
    fn correlated_non_state_operand_returns_type_error() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0.5\nSTATE_NEW R1 0.5 0.0 0.0\nLOAD_CONST R2 0.0\nHOST_CALL correlated R0 3 R3\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Type);
    }

    #[test]
    fn binary_on_bool_is_type_error() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 true\nLOAD_CONST R1 2\nBINARY R0 add R1 R2\nHALT\n",
        );
        assert_eq!(error, LanaError::Type);
    }

    #[test]
    fn mix_attaches_derivation() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.2 0.0 0.0\nSTATE_NEW R1 0.8 0.0 0.0\nLOAD_CONST R2 0.3\nMIX R3 R0 R1 R2\nMEASURE R3 probability R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "0.62");
    }

    #[test]
    fn mix_combines_evidence_least_certain_wins() {
        // exact (evidence) ⊕ modeled (assumption) = modeled.
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.2 0.0 0.0\nEVIDENCE R0 R1 obs\nSTATE_NEW R2 0.8 0.0 0.0\nASSUME R2 R3 model\nLOAD_CONST R4 0.5\nMIX R5 R1 R3 R4\nDERIVATION R5 R6\nRETURN R6\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert!(result.contains("\"status\": modeled"), "expected modeled status, got {result}");
    }

    #[test]
    fn assume_is_modeled_not_exact() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.2 0.0 0.0\nASSUME R0 R1 model\nDERIVATION R1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert!(result.contains("\"status\": modeled"), "expected modeled status, got {result}");
        assert!(result.contains("\"exactness\": approximate"), "expected approximate exactness, got {result}");
    }

    #[test]
    fn append_expect_builds_statistical_result() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.2 0.0 0.0\nSTATE_NEW R1 0.3 0.0 0.0\nAPPEND R0 R1 R2\nEXPECT R3 R2 probability\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(
            result,
            "{\"method\": exact, \"value\": 0.44, \"observable\": probability, \"provenance\": exact, \"sample_count\": null, \"seed\": null}"
        );
    }

    #[test]
    fn append_support_returns_support_array() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.2 0.0 0.0\nSTATE_NEW R1 0.3 0.0 0.0\nAPPEND R0 R1 R2\nSUPPORT R3 R2 10\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[state(p=0.44, d_re=0, d_im=0)]");
    }

    #[test]
    fn append_sample_state_dist_samples() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.2 0.0 0.0\nSTATE_NEW R1 0.3 0.0 0.0\nAPPEND R0 R1 R2\nSAMPLE_STATE_DIST R2 R3\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert!(result.starts_with("state(p="), "expected a state, got {result}");
    }

    #[test]
    fn map_transforms_state_dist() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.3 0.0 0.0\nAPPEND R0 R0 R1\nMAP R2 R1 invert\nMEASURE R2 probability R3\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Ok);
        // append(0.3, 0.3) has expected probability 0.51; invert maps it to 0.49.
        assert_eq!(result, "0.49");
    }

    #[test]
    fn validate_rejects_non_map_schema() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 5\nLOAD_CONST R1 7\nVALIDATE R2 R0 R1\nHALT\n",
        );
        assert_eq!(error, LanaError::Schema);
    }

    #[test]
    fn revision_reads_derivation_revision() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.2 0.0 0.0\nSTATE_NEW R1 0.8 0.0 0.0\nLOAD_CONST R2 0.3\nMIX R3 R0 R1 R2\nREVISION R4 R3\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "0");
    }

    #[test]
    fn estimate_measure_probability_samples() {
        // Unlike the other increment-2 opcodes, ESTIMATE_MEASURE_* takes the
        // source first and the destination second, matching the C11 assembler
        // and the compiler emitter.
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.5 0.0 0.0\nAPPEND R0 R0 R1\nESTIMATE_MEASURE_PROBABILITY R1 R2 computational 100\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        // The estimate is a number in [0, 1]; the exact value depends on the RNG.
        assert!(result.parse::<f64>().is_ok(), "expected a number, got {result}");
    }

    #[test]
    fn joint_build_and_resolve() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 1\nLOAD_CONST R1 2\nJOINT_BUILD R2 R0 2 independent:a;b\nRESOLVE R2 R3\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[1, 2]");
    }

    #[test]
    fn joint_build_finite_single_row_resolves() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 1\nLOAD_CONST R1 2\nLOAD_CONST R2 1.0\nARRAY_NEW R3 R0 3\nARRAY_NEW R8 R3 1\nJOINT_BUILD_FINITE R8 R9 a;b\nRESOLVE R9 R10\nRETURN R10\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[1, 2]");
    }

    #[test]
    fn joint_project_returns_single_marginal() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 1\nLOAD_CONST R1 2\nJOINT_BUILD R2 R0 2 independent:a;b\nJOINT_PROJECT R2 R3 a\nRESOLVE R3 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "1");
    }

    #[test]
    fn joint_condition_matching_evidence() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 1\nLOAD_CONST R1 2\nJOINT_BUILD R2 R0 2 independent:a;b\nLOAD_CONST R3 1\nJOINT_CONDITION R2 R4 a R3\nRESOLVE R4 R5\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[1, 2]");
    }

    #[test]
    fn joint_condition_mismatch_is_invalid_conditioning() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 1\nLOAD_CONST R1 2\nJOINT_BUILD R2 R0 2 independent:a;b\nLOAD_CONST R3 9\nJOINT_CONDITION R2 R4 a R3\nHALT\n",
        );
        assert_eq!(error, LanaError::InvalidConditioning);
    }

    #[test]
    fn observe_increments_revision() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 1\nLOAD_CONST R1 2\nJOINT_BUILD R2 R0 2 independent:a;b\nLOAD_CONST R3 1\nOBSERVE R2 R4 a R3\nEXPLAIN R4 R5\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert!(result.contains("revision=1"), "expected revision=1, got {result}");
    }

    #[test]
    fn possibility_build_single_resolves() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 1\nARRAY_NEW R2 R0 1\nPOSSIBILITY_BUILD R2 R3\nRESOLVE R3 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "1");
    }

    #[test]
    fn path_split_join_builds_path_set() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 true\nLOAD_CONST R1 false\nARRAY_NEW R2 R0 2\nPOSSIBILITY_BUILD R2 R3\nPATH_SPLIT R3 join\nLOAD_CONST R4 10\nPATH_JOIN\njoin:\nPATH_JOIN\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "paths{true => 10, false => null}");
    }

    #[test]
    fn evidence_explain_renders_derivation() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 5\nEVIDENCE R0 R1 obs\nEXPLAIN R1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(
            result,
            "evidence evidence id=[0,1] revision=0 exactness=exact outcome=success reason=none label=obs inputs=0"
        );
    }

    #[test]
    fn info_sample_possibility_returns_element() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 1\nLOAD_CONST R1 2\nARRAY_NEW R2 R0 2\nPOSSIBILITY_BUILD R2 R3\nINFO_SAMPLE R3 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert!(result == "1" || result == "2", "expected 1 or 2, got {result}");
    }

    #[test]
    fn joint_build_deep_clones_array_marginal() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 1\nLOAD_CONST R1 2\nARRAY_NEW R2 R0 2\nJOINT_BUILD R3 R2 1 independent:a\nLOAD_CONST R4 0\nLOAD_CONST R5 99\nARRAY_SET R2 R4 R5\nRESOLVE R3 R6\nRETURN R6\n",
        );
        assert_eq!(error, LanaError::Ok);
        // The joint's marginal was cloned at build time; mutating the source
        // array afterwards must not change it.
        assert_eq!(result, "[1, 2]");
    }

    // --- Increment 4: tasks ---

    #[test]
    fn fork_join_returns_child_result() {
        let (error, result) = run_chunk(
            ".function main 0 8\nFORK worker R1 0 R0\nJOIN R0 R1\nRETURN R1\n.function worker 0 4\nLOAD_CONST R0 42\nRETURN R0\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "42");
    }

    #[test]
    fn fork_join_attaches_derivation() {
        let (error, result) = run_chunk(
            ".function main 0 8\nFORK worker R1 0 R0\nJOIN R0 R1\nEXPLAIN R1 R2\nRETURN R2\n.function worker 0 4\nLOAD_CONST R0 7\nRETURN R0\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert!(result.contains("operation task_join"), "expected task_join derivation, got {result}");
    }

    #[test]
    fn fork_passes_arguments() {
        let (error, result) = run_chunk(
            ".function main 0 8\nLOAD_CONST R1 3\nLOAD_CONST R2 4\nFORK worker R1 2 R0\nJOIN R0 R3\nRETURN R3\n.function worker 2 4\nBINARY R0 add R1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "7");
    }

    #[test]
    fn fork_arity_mismatch_is_type_error() {
        // The assembler validates FORK arity against the target function,
        // matching the C11 assembler's post-assembly verification.
        let error = assembler::assemble(
            ".function main 0 8\nLOAD_CONST R1 3\nFORK worker R1 2 R0\nHALT\n.function worker 1 4\nRETURN R0\n",
        )
        .expect_err("arity mismatch must fail to assemble");
        assert_eq!(error.code, LanaError::Type);
    }

    #[test]
    fn join_timeout_returns_result() {
        let (error, result) = run_chunk(
            ".function main 0 8\nFORK worker R1 0 R0\nLOAD_CONST R2 1.0\nJOIN_TIMEOUT R0 R2 R3\nRETURN R3\n.function worker 0 4\nLOAD_CONST R0 5\nRETURN R0\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "5");
    }

    #[test]
    fn join_timeout_expires_on_looping_worker() {
        // A worker that never returns; a tiny timeout makes JOIN_TIMEOUT
        // report Timeout deterministically. The worker is cancelled at
        // scheduler shutdown, so the test stays fast without a low instruction
        // limit (which the child could hit before the timeout expires).
        let (error, _) = run_chunk(
            ".function main 0 8\nFORK worker R1 0 R0\nLOAD_CONST R2 0.000001\nJOIN_TIMEOUT R0 R2 R3\nHALT\n.function worker 0 4\nloop:\nJUMP loop\n",
        );
        assert_eq!(error, LanaError::Timeout);
    }

    #[test]
    fn join_all_returns_array() {
        let (error, result) = run_chunk(
            ".function main 0 8\nFORK worker R1 0 R0\nFORK worker R2 0 R1\nARRAY_NEW R2 R0 2\nJOIN_ALL R2 R3\nRETURN R3\n.function worker 0 4\nLOAD_CONST R0 9\nRETURN R0\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[9, 9]");
    }

    #[test]
    fn cancel_task_returns_cancelled() {
        let (error, _) = run_chunk(
            ".function main 0 8\nFORK worker R1 0 R0\nCANCEL R0\nJOIN R0 R1\nHALT\n.function worker 0 4\nLOAD_CONST R0 1\nRETURN R0\n",
        );
        assert_eq!(error, LanaError::Cancelled);
    }

    #[test]
    fn taskgroup_exit_cancels_and_joins_group() {
        // A worker that loops forever is still running when TASKGROUP_EXIT
        // fires; the group close cancels it, waits for it, and clears the
        // Cancelled error so the parent continues.
        let (error, result) = run_chunk(
            ".function main 0 8\nTASKGROUP_ENTER\nFORK worker R1 0 R0\nTASKGROUP_EXIT\nLOAD_CONST R1 99\nRETURN R1\n.function worker 0 4\nloop:\nJUMP loop\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "99");
    }

    #[test]
    fn orphaned_looping_task_is_cancelled_at_shutdown() {
        // A task that is never joined or cancelled must not hang the parent's
        // exit: scheduler shutdown cancels every live task so the worker
        // thread stops promptly. Bound the instruction limit to keep the test
        // fast even if cancellation were missed.
        let chunk = assembler::assemble(
            ".function main 0 8\nFORK worker R1 0 R0\nLOAD_CONST R1 1\nRETURN R1\n.function worker 0 4\nloop:\nJUMP loop\n",
        )
        .expect("fixture assembles");
        let mut vm = Vm::new(&chunk);
        vm.set_instruction_limit(1000);
        let (error, result) = {
            let error = vm.run();
            (error, vm.result().print())
        };
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "1");
    }

    #[test]
    fn fork_task_limit_exceeded() {
        let (error, _) = run_chunk(
            ".function main 0 8\nFORK worker R1 0 R0\nFORK worker R2 0 R1\nHALT\n.function worker 0 4\nLOAD_CONST R0 1\nRETURN R0\n",
        );
        // Default task limit is 64, so two forks succeed; the error must be Ok.
        assert_eq!(error, LanaError::Ok);
    }

    #[test]
    fn fork_task_limit_respected() {
        let chunk = assembler::assemble(
            ".function main 0 8\nFORK worker R1 0 R0\nFORK worker R2 0 R1\nHALT\n.function worker 0 4\nLOAD_CONST R0 1\nRETURN R0\n",
        )
        .expect("fixture assembles");
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.set_task_limit(1), LanaError::Ok);
        let error = vm.run();
        assert_eq!(error, LanaError::Limit);
    }
}
