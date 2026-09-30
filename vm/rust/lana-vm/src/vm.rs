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

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

use lana_bytecode::opcode::{LANA_MAX_CALL_FRAMES, LANA_MAX_REGISTERS};
use lana_bytecode::{Chunk, Instruction, LanaError, OpCode, Value as ConstantValue, ValueType};

use crate::derivation::{
    self, Derivation, DerivationExactness, DerivationKind, DerivationOutcome, EvidenceStatus,
};
use crate::rng::Rng;
use crate::heap::{Buffer, Heap};
use crate::sha256::{hex_digest, sha256};
use crate::state::{self, Indexes, State, StateValue};
use crate::state_dist::{self, DistEvalFrame, EvalAction, LANA_STATE_DIST_DEPTH_LIMIT};
use crate::tensor;
use crate::tensor::{tensor_get_imag, tensor_get_real, tensor_set_imag, tensor_set_real};
use crate::value::{
    Adt, Array, CapabilityToken, Claim, Dataset, DatasetOp, DistOperand, EffectReceipt, FiniteKernel, FiniteNetwork, Generator, NetworkNode, InferenceAlgorithm, JointKind, JointRow, ObjectValue,
    JointState, Map, Optimizer, PathAlternative, PathSet, PlannedEffect, PlannedEffectState, Possibility, Posterior, Reactive,
    ReactiveKind, ReactiveVersion, RelationshipKind, SharedCommit, SharedInformation,
    SharedObservation, SharedState, SharedVersion, Set, StateDist, StateDistKind, Task, Tensor, TensorDtype, TrainingResult, Value,
    ValueKind, VmError, Regex, RegexClass, RegexInst, RegexOp, Future, TensorDevice, LANA_CAPABILITY_ADMIN, LANA_CAPABILITY_OBSERVE, LANA_CAPABILITY_READ,
    LANA_JOINT_CAN_CONDITION, LANA_JOINT_CAN_PROJECT, LANA_JOINT_CAN_RESOLVE,
    LANA_JOINT_CAN_SAMPLE,
};

mod evaluation;
mod http;
mod objects;
mod classes;
pub(crate) mod class_gc;
pub use class_gc::RootedValue;
mod object_effects;

fn shared_derivation_text(text: &'static str) -> Arc<str> {
    static EMPTY: OnceLock<Arc<str>> = OnceLock::new();
    static AUTODIFF: OnceLock<Arc<str>> = OnceLock::new();
    static INPUT: OnceLock<Arc<str>> = OnceLock::new();
    static NONE: OnceLock<Arc<str>> = OnceLock::new();
    match text {
        "" => EMPTY.get_or_init(|| Arc::from("")).clone(),
        "autodiff" => AUTODIFF.get_or_init(|| Arc::from("autodiff")).clone(),
        "input" => INPUT.get_or_init(|| Arc::from("input")).clone(),
        "none" => NONE.get_or_init(|| Arc::from("none")).clone(),
        _ => unreachable!("non-autodiff derivation text"),
    }
}

/// Reinterpret a state tensor's byte buffer as `&[f64]`. State tensors are
/// always complex (16 bytes/element = 2 f64s), so the buffer length is a
/// multiple of 8.
fn state_f64(t: &Tensor) -> &[f64] {
    unsafe { std::slice::from_raw_parts(t.data.as_ptr() as *const f64, t.data.len() / 8) }
}

/// Reinterpret a state tensor's byte buffer as `&mut [f64]`. Panics if the
/// buffer is shared (a view); callers write only to freshly-created tensors.
fn state_f64_mut(t: &mut Tensor) -> &mut [f64] {
    let data = Arc::get_mut(&mut t.data).expect("state_f64_mut on a shared buffer");
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f64, data.len() / 8) }
}

/// LIP-027: parse an optional trailing dtype string argument (the last of
/// `argc` arguments). Returns `None` for an invalid dtype string; callers must
/// already have checked argc is 1 or 2.
fn tensor_optional_dtype(arguments: &[Value], argc: usize) -> Option<TensorDtype> {
    if argc == 1 {
        return Some(TensorDtype::F64);
    }
    let ValueKind::String(s) = &arguments[1].kind else {
        return None;
    };
    TensorDtype::from_str(s)
}

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

/// Ephemeral source-row decision from one named dataset-plan run.
#[derive(Debug, Clone)]
pub struct DatasetDecision {
    pub operation: &'static str,
    pub reason: &'static str,
    pub input: Arc<Derivation>,
    pub predicate: Option<Arc<Derivation>>,
    pub predicate_value: Option<bool>,
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
    pub is_generator: bool,
    pub is_async: bool,
    object_method: Option<(u32, u32)>,
    _allocation: Option<Arc<crate::heap::Reservation>>,
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
            is_generator: false,
            is_async: false,
            object_method: None,
            _allocation: None,
        }
    }

    fn charged(register_count: usize, heap: &Heap) -> Result<Self, LanaError> {
        let bytes = register_count.checked_mul(std::mem::size_of::<Value>() + std::mem::size_of::<History>())
            .ok_or(LanaError::Oom)?;
        let allocation = Arc::new(heap.reserve(bytes)?);
        let mut frame = Self::new(register_count);
        frame._allocation = Some(allocation);
        Ok(frame)
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
        Call | Fork | HostCall | Generator => {
            if ins.imm != u32::MAX && ins.imm > 0 {
                max = max.max(ins.c as usize + ins.imm as usize - 1);
            }
        }
        ValueNew | ObjectNew => {
            if ins.imm > 0 { max = max.max(ins.b as usize + ins.imm as usize - 1); }
        }
        OoGet | OoSet | OoCall | OoStaticCall | OoAsInterface => {
            max = max.max(ins.b as usize);
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
        result[index] = if chunk.version == 6 {
            (max + 1).max(chunk.functions[index].register_count as usize)
        } else { max + 1 };
    }
    result
}

/// One pending path split, mirroring `struct LanaPathExecution` in `vm/c/vm.c`.
/// The Rust VM keeps the executions on a `Vec` stack; the C11 uses a linked
/// list with `next` pointing at the previous execution.
#[derive(Default)]
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
    split: bool,
}

/// A memo mapping source container pointers to their clones, mirroring
/// `LanaContainerCloneMemo` in `vm/c/vm.c`. Mutable containers preserve
/// shared substructure. Joints also preserve aliasing because that identity
/// represents a declared relationship in captured dataset cells.
#[derive(Default)]
struct DeepCloneMemo {
    freeze: bool,
    transfer: bool,
    classes: HashMap<usize, Arc<crate::value::ClassReference>>,
    depth: usize,
    arrays: HashMap<usize, Arc<Mutex<Array>>>,
    maps: HashMap<usize, Arc<Mutex<Map>>>,
    joints: HashMap<usize, Arc<JointState>>,
    distributions: HashMap<usize, Arc<StateDist>>,
    derivations: HashMap<usize, Arc<Derivation>>,
    sets: HashMap<usize, Arc<Mutex<Set>>>,
    generators: HashMap<usize, Arc<Mutex<Generator>>>,
    futures: HashMap<usize, Arc<Mutex<Future>>>,
    claims: HashMap<usize, Arc<Claim>>,
    effects: HashMap<usize, Arc<PlannedEffect>>,
    strings: HashMap<usize, Arc<str>>,
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

// Lana 2.0 surprisal (natural-log information content in nats). Present in
// both the C11 VM and the Rust VM at id 55.
pub const LANA_HOST_SURPRISAL: u32 = 55;

// LIP-004 first-class tensors. Present in both the C11 VM and the Rust VM at
// ids 56-73, so a `.labc` assembled by either backend runs identically under
// both.
pub const LANA_HOST_TENSOR_ALLOC: u32 = 56;
pub const LANA_HOST_TENSOR_ZEROS: u32 = 57;
pub const LANA_HOST_TENSOR_ONES: u32 = 58;
pub const LANA_HOST_TENSOR_EYE: u32 = 59;
pub const LANA_HOST_TENSOR_DTYPE: u32 = 60;
pub const LANA_HOST_TENSOR_SHAPE: u32 = 61;
pub const LANA_HOST_TENSOR_NDIM: u32 = 62;
pub const LANA_HOST_TENSOR_ADD: u32 = 63;
pub const LANA_HOST_TENSOR_SUB: u32 = 64;
pub const LANA_HOST_TENSOR_MUL: u32 = 65;
pub const LANA_HOST_TENSOR_DIV: u32 = 66;
pub const LANA_HOST_TENSOR_MATMUL: u32 = 67;
pub const LANA_HOST_TENSOR_SUM: u32 = 68;
pub const LANA_HOST_TENSOR_MEAN: u32 = 69;
pub const LANA_HOST_TENSOR_MAX: u32 = 70;
pub const LANA_HOST_TENSOR_MIN: u32 = 71;
pub const LANA_HOST_TENSOR: u32 = 72;
pub const LANA_HOST_TENSOR_COMPLEX: u32 = 73;

// LIP-012 capability grant/revoke. Present in both the C11 VM and the Rust VM
// at ids 74-75, so a `.labc` assembled by either backend runs identically under
// both.
pub const LANA_HOST_GRANT: u32 = 74;
pub const LANA_HOST_REVOKE: u32 = 75;

// LIP-022 §1 immutable sets. Present in both the C11 VM and the Rust VM at
// ids 76-81, so a `.labc` assembled by either backend runs identically under
// both.
pub const LANA_HOST_SET_NEW: u32 = 76;
pub const LANA_HOST_SET_ADD: u32 = 77;
pub const LANA_HOST_SET_CONTAINS: u32 = 78;
pub const LANA_HOST_SET_UNION: u32 = 79;
pub const LANA_HOST_SET_INTERSECT: u32 = 80;
pub const LANA_HOST_SET_DIFFERENCE: u32 = 81;

// LIP-016 installed stdlib: read an environment variable. Present in both the
// C11 VM and the Rust VM at id 82.
pub const LANA_HOST_GETENV: u32 = 82;
// LIP-016 stdlib: reseed the RNG and floor a number. Present in both VMs.
pub const LANA_HOST_RANDOM_SEED: u32 = 83;
pub const LANA_HOST_FLOOR: u32 = 84;
// LIP-023 data interchange: parse a number from text. Present in both VMs.
pub const LANA_HOST_STRING_TO_NUMBER: u32 = 85;
// LIP-023 data interchange: runtime type name of a value. Present in both VMs.
pub const LANA_HOST_TYPE_OF: u32 = 86;
// LIP-021 §3 explicit formatting. Present in both VMs.
pub const LANA_HOST_FORMAT: u32 = 87;
pub const LANA_HOST_FORMAT_NUMBER: u32 = 88;
// LIP-021 §1/§4 Unicode code points and simple case mapping. Present in both VMs.
pub const LANA_HOST_CHAR_LENGTH: u32 = 89;
pub const LANA_HOST_STRING_CODEPOINT_SLICE: u32 = 90;
pub const LANA_HOST_TO_UPPER: u32 = 91;
pub const LANA_HOST_TO_LOWER: u32 = 92;
// LIP-021 §2 regular expressions. Present in both VMs.
pub const LANA_HOST_REGEX_COMPILE: u32 = 93;
pub const LANA_HOST_REGEX_MATCH: u32 = 94;
pub const LANA_HOST_REGEX_SEARCH: u32 = 95;
pub const LANA_HOST_REGEX_REPLACE: u32 = 96;
// LIP-004 §5 explicit GPU matmul. Present in both VMs at id 97.
pub const LANA_HOST_GPU_MATMUL: u32 = 97;

// LIP-005 linear algebra on STATEs. Present in both the C11 VM and the Rust VM
// at ids 98-110, so a `.labc` assembled by either backend runs identically
// under both.
pub const LANA_HOST_DENSITY_OPERATOR: u32 = 98;
pub const LANA_HOST_POVM: u32 = 99;
pub const LANA_HOST_CHANNEL: u32 = 100;
pub const LANA_HOST_OBSERVABLE: u32 = 101;
pub const LANA_HOST_TENSOR_PRODUCT: u32 = 102;
pub const LANA_HOST_PARTIAL_TRACE: u32 = 103;
pub const LANA_HOST_MEASURE_WITH: u32 = 104;
pub const LANA_HOST_APPLY_TO: u32 = 105;
pub const LANA_HOST_EXPECT: u32 = 106;
pub const LANA_HOST_MIX: u32 = 107;
pub const LANA_HOST_TRACE_DISTANCE: u32 = 108;
pub const LANA_HOST_IS_SEPARABLE: u32 = 109;
pub const LANA_HOST_TO_STATE: u32 = 110;

// LIP-011 reverse-mode automatic differentiation. Present in both the C11 VM
// and the Rust VM at ids 111-112, so a `.labc` assembled by either backend
// runs identically under both.
pub const LANA_HOST_GRAD: u32 = 111;
pub const LANA_HOST_VJP: u32 = 112;

// LIP-006 auditable, replayable training primitive. Present in both the C11 VM
// and the Rust VM at ids 113-115, so a `.labc` assembled by either backend
// runs identically under both.
pub const LANA_HOST_SGD: u32 = 113;
pub const LANA_HOST_ADAM: u32 = 114;
pub const LANA_HOST_TRAIN: u32 = 115;

// LIP-009 Bayesian inference as a first-class training mode. Present in both
// the C11 VM and the Rust VM at ids 116-119, so a `.labc` assembled by either
// backend runs identically under both.
pub const LANA_HOST_MCMC: u32 = 116;
pub const LANA_HOST_VI: u32 = 117;
pub const LANA_HOST_SMC: u32 = 118;
pub const LANA_HOST_INFER: u32 = 119;

// LIP-010 incremental / online learning. Present in both the C11 VM and the
// Rust VM at id 120, so a `.labc` assembled by either backend runs identically
// under both.
pub const LANA_HOST_UPDATE: u32 = 120;

// LIP-014 whole-run reproducibility and resumability. Present in both the C11
// VM and the Rust VM at id 121, so a `.labc` assembled by either backend runs
// identically under both.
pub const LANA_HOST_RESUME: u32 = 121;

// LIP-007 differentiable STATE tensors. Present in both the C11 VM and the
// Rust VM at ids 122-125, so a `.labc` assembled by either backend runs
// identically under both.
pub const LANA_HOST_STATE_TENSOR: u32 = 122;
pub const LANA_HOST_APPEND: u32 = 123;
pub const LANA_HOST_MEASURE: u32 = 124;
pub const LANA_HOST_TRANSFORM: u32 = 125;

// Durable-pipeline host calls. The store/policy/ledger calls (141-151) are
// dispatched through the host-call extension registered by the CLI (see
// `set_host_call_extension`); the C11 VM dispatches the store calls directly.
// They sit at the END of the id range (after the shared async/dataset calls at
// 126-140) so the shared ids match the C11 VM.
pub const LANA_HOST_STORE_OPEN: u32 = 141;
pub const LANA_HOST_STORE_PUT: u32 = 142;
pub const LANA_HOST_STORE_GET: u32 = 143;
pub const LANA_HOST_STORE_DELETE: u32 = 144;
pub const LANA_HOST_STORE_COMMIT: u32 = 145;
pub const LANA_HOST_STORE_SCAN: u32 = 146;
pub const LANA_HOST_STORE_CURRENT_REVISION: u32 = 147;
pub const LANA_HOST_POLICY_EVALUATE: u32 = 148;
pub const LANA_HOST_POLICY_STORE_DECISION: u32 = 149;
pub const LANA_HOST_LEDGER_APPEND: u32 = 150;
pub const LANA_HOST_LEDGER_QUERY: u32 = 151;
// LIP-015 §3, §5 data layer: MVCC reads, optimistic commit, adapters.
pub const LANA_HOST_STORE_GET_AT: u32 = 152;
pub const LANA_HOST_STORE_SNAPSHOT: u32 = 153;
pub const LANA_HOST_STORE_COMMIT_IF: u32 = 154;
pub const LANA_HOST_ADAPTER_LOAD: u32 = 155;
pub const LANA_HOST_ADAPTER_FETCH: u32 = 156;
// LIP-018 two-way FFI.
pub const LANA_HOST_FFI_DECLARE: u32 = 157;
pub const LANA_HOST_FFI_LOAD: u32 = 158;
pub const LANA_HOST_FFI_CALL: u32 = 159;
// LIP-019 networking and HTTP.
pub const LANA_HOST_HTTP_GET: u32 = 160;
pub const LANA_HOST_HTTP_POST: u32 = 161;
pub const LANA_HOST_SOCKET_CONNECT: u32 = 162;
pub const LANA_HOST_SOCKET_SEND: u32 = 163;
pub const LANA_HOST_SOCKET_RECV: u32 = 164;
pub const LANA_HOST_SOCKET_CLOSE: u32 = 165;
// LIP-027: cast a tensor to another real dtype.
pub const LANA_HOST_TENSOR_CAST: u32 = 166;
pub const LANA_HOST_TENSOR_RESHAPE: u32 = 167;
pub const LANA_HOST_TENSOR_TRANSPOSE: u32 = 168;
pub const LANA_HOST_TENSOR_EXP: u32 = 169;
pub const LANA_HOST_TENSOR_LOG: u32 = 170;
pub const LANA_HOST_TENSOR_SQRT: u32 = 171;
pub const LANA_HOST_TENSOR_RELU: u32 = 172;
pub const LANA_HOST_TENSOR_SOFTMAX: u32 = 173;
pub const LANA_HOST_TENSOR_LOGSUMEXP: u32 = 174;
pub const LANA_HOST_TENSOR_ARGMAX: u32 = 175;
pub const LANA_HOST_TENSOR_COMPARE: u32 = 176;
pub const LANA_HOST_TENSOR_SELECT: u32 = 177;
pub const LANA_HOST_TENSOR_GATHER: u32 = 178;
pub const LANA_HOST_CHOLESKY_SOLVE: u32 = 179;
pub const LANA_HOST_RANDOM_UNIFORM: u32 = 180;
pub const LANA_HOST_RANDOM_NORMAL: u32 = 181;
pub const LANA_HOST_TENSOR_DEVICE: u32 = 182;
pub const LANA_HOST_TENSOR_TO_DEVICE: u32 = 183;
pub const LANA_HOST_TENSOR_TO_CPU: u32 = 184;
// LIP-029 execution boundary. Rust-only: the C11 VM remains frozen at v1-v4.
pub const LANA_HOST_EXECUTION_CAPABILITY: u32 = 185;
pub const LANA_HOST_EXECUTION_AUTHORIZE: u32 = 186;
pub const LANA_HOST_EXECUTION_EXECUTE: u32 = 187;
pub const LANA_HOST_FUTURE_MESSAGE: u32 = 188;
pub const LANA_HOST_CORE_ENTROPY: u32 = 189;
pub const LANA_HOST_CORE_CONDITIONAL_ENTROPY: u32 = 190;
pub const LANA_HOST_CORE_MUTUAL_INFORMATION: u32 = 191;
pub const LANA_HOST_CORE_BROJA: u32 = 192;
pub const LANA_HOST_CORE_KERNEL: u32 = 193;
pub const LANA_HOST_CORE_IDENTITY_KERNEL: u32 = 194;
pub const LANA_HOST_CORE_COMPOSE_KERNELS: u32 = 195;
pub const LANA_HOST_CORE_NETWORK: u32 = 196;
pub const LANA_HOST_CORE_INFER: u32 = 197;
pub const LANA_HOST_CORE_FORGET_WEIGHTS: u32 = 198;
pub const LANA_HOST_CORE_ASSIGN_WEIGHTS: u32 = 199;
pub const LANA_HOST_DATASET_SOURCE: u32 = 200;
pub const LANA_HOST_DATASET_QUERY: u32 = 201;
pub const LANA_HOST_DATASET_APPLY: u32 = 202;
pub const LANA_HOST_DATASET_SNAPSHOT: u32 = 203;
pub const LANA_HOST_DATASET_EVIDENCE: u32 = 204;
pub const LANA_HOST_DATASET_EXCLUSIONS: u32 = 205;
pub const LANA_HOST_RULES_LEARN: u32 = 206;
pub const LANA_HOST_RULES_PREDICT: u32 = 207;
pub const LANA_HOST_RULES_SAVE: u32 = 208;
pub const LANA_HOST_RULES_ADD_COUNTEREXAMPLE: u32 = 209;
pub const LANA_HOST_RULES_INSPECT: u32 = 210;
pub const LANA_HOST_RULES_ROLLBACK: u32 = 211;
pub const LANA_HOST_TREES_FIT: u32 = 212;
pub const LANA_HOST_TREES_PREDICT: u32 = 213;
pub const LANA_HOST_TREES_EXPLAIN: u32 = 214;
pub const LANA_HOST_TREES_SAVE: u32 = 215;
pub const LANA_HOST_TREES_LOAD: u32 = 216;
pub const LANA_HOST_EVALUATION_WALK_FORWARD: u32 = 217;
pub const LANA_HOST_DATASET_SQLITE: u32 = 218;
pub const LANA_HOST_DOCUMENT_EXTRACT: u32 = 219;
pub const LANA_HOST_INFORMATION_SNAPSHOT: u32 = 220;

// LIP-024 async/await. Present in both the C11 VM and the Rust VM at ids
// 126-129 (matching the C11 assembler's host-call table and the
// `LanaHostCallId` enum in `vm/include/bytecode.h`), so a `.labc` assembled
// by either backend runs identically under both.
pub const LANA_HOST_RUN_ASYNC: u32 = 126;
pub const LANA_HOST_FUTURE_ALL: u32 = 127;
pub const LANA_HOST_FUTURE_RACE: u32 = 128;
pub const LANA_HOST_SLEEP: u32 = 129;
pub const LANA_HOST_DATASET: u32 = 130;
pub const LANA_HOST_DATASET_FILTER: u32 = 131;
pub const LANA_HOST_DATASET_MAP: u32 = 132;
pub const LANA_HOST_DATASET_SELECT: u32 = 133;
pub const LANA_HOST_DATASET_LIMIT: u32 = 134;
pub const LANA_HOST_DATASET_SORT: u32 = 135;
pub const LANA_HOST_DATASET_GROUP_BY: u32 = 136;
pub const LANA_HOST_DATASET_AGGREGATE: u32 = 137;
pub const LANA_HOST_DATASET_JOIN: u32 = 138;
pub const LANA_HOST_DATASET_MATERIALIZE: u32 = 139;
pub const LANA_HOST_DATASET_EXPLAIN: u32 = 140;

/// The shared-information identity and commit-revision counters, matching the
/// `next_shared_identity` / `next_commit_revision` atomics in `runtime/c/shared.c`.
static NEXT_SHARED_IDENTITY: AtomicU64 = AtomicU64::new(1);
static NEXT_COMMIT_REVISION: AtomicU64 = AtomicU64::new(1);
static NEXT_ATOMIC_WRITE: AtomicU64 = AtomicU64::new(0);

fn write_text_atomic_with_sync(
    path: &Path,
    contents: &[u8],
    sync_file: impl FnOnce(&std::fs::File) -> std::io::Result<()>,
    sync_parent: impl FnOnce(&std::fs::File) -> std::io::Result<()>,
) -> Result<(), (std::io::Error, bool)> {
    let name = path.file_name().ok_or_else(|| (std::io::Error::other("missing file name"), false))?;
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let directory = std::fs::File::open(parent).map_err(|error| (error, false))?;
    let mut temporary_name = name.to_os_string();
    temporary_name.push(format!(".lana-{}-{}.tmp", std::process::id(), NEXT_ATOMIC_WRITE.fetch_add(1, Ordering::Relaxed)));
    let temporary = parent.join(temporary_name);
    let result = (|| {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)] {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary).map_err(|error| (error, false))?;
        file.write_all(contents).map_err(|error| (error, false))?;
        sync_file(&file).map_err(|error| (error, false))?;
        drop(file);
        std::fs::rename(&temporary, path).map_err(|error| (error, false))?;
        sync_parent(&directory).map_err(|error| (error, true))
    })();
    if result.is_err() { let _ = std::fs::remove_file(&temporary); }
    result
}

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
    all_tasks: Vec<(Arc<Task>, RootedValue)>,
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
        for (task, _) in &state.all_tasks {
            task.cancelled.store(true, Ordering::Relaxed);
        }
        self.available.notify_all();
    }
}

struct SchedulerShutdown<'run, 'chunk>(&'run Scheduler<'chunk>);

impl Drop for SchedulerShutdown<'_, '_> {
    fn drop(&mut self) { self.0.shutdown(); }
}

/// Run a queued task's child VM to completion and publish the result to the
/// task handle, mirroring `run_task` in `vm/c/vm.c`.
fn run_task(queued: QueuedTask<'_>) {
    let mut child = queued.child;
    let mut status = child.run();
    let (mut error, mut result) = if status == LanaError::Ok {
        (VmError::default(), child.result.clone())
    } else {
        (child.error().clone(), Value::null())
    };
    let result_root = if status == LanaError::Ok {
        match child.result() {
            Ok(root) => Some(root),
            Err(root_error) => { status = root_error; error = VmError::default(); result = Value::null(); None }
        }
    } else { None };
    // Publish completion only after the result lease owns the retired child heap.
    drop(child);
    let mut state = queued.handle.state.lock().unwrap();
    state.status = status;
    state.error = error;
    state.result = result;
    // The result lease owns the child's storage after Vm::drop retires it.
    // Moving storage into TaskState would invalidate independent result leases.
    state.result_root = result_root;
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
    dataset_work_remaining: Option<u64>,
    dataset_decisions: Vec<DatasetDecision>,
    opcode_counts: Vec<u64>,
    class_owner: Arc<()>,
    class_objects: Vec<Arc<crate::value::ClassObject>>,
    class_objects_reservation: Arc<Mutex<crate::heap::Reservation>>,
    class_allocations_since_gc: usize,
    owned_cycles: Buffer<Arc<crate::heap::CycleSlot>>,
    host_roots: Arc<class_gc::RootOwner>,
    constructing: Vec<Arc<crate::value::ClassReference>>,
    constructions: Vec<classes::Construction>,
    object_methods: HashMap<u32, (u32, u32)>,
    object_function_entries: Vec<(usize, Option<(u32, u32)>)>,
    object_descriptors: Option<std::collections::BTreeMap<u32, Arc<lana_bytecode::objects::Descriptor>>>,
    state_transition_count: u64,
    allocation_count: u64,
    memory_limit: usize,
    allocated_bytes: usize,
    heap: Heap,
    rng: Rng,
    root_seed: u64,
    lineage: u64,
    revision: u64,
    derivation_sequence: u64,
    ad_recording: bool,
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
    pending_io_outcome: Option<(bool, String)>,
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
    /// The single-threaded event loop's ready queue (LIP-024 §6). Futures
    /// become runnable in FIFO creation order; the loop always picks the oldest
    /// runnable future next, guaranteeing reproducible scheduling.
    ready_futures: VecDeque<Arc<Mutex<Future>>>,
    /// Futures awaiting a given future, keyed by `Arc::as_ptr` of the awaited
    /// future. When a future completes, its awaiters are made ready and
    /// re-queued (LIP-024 §6 "Completion").
    awaiters: HashMap<usize, Vec<Arc<Mutex<Future>>>>,
    /// True while the event loop is driving async frames. The dispatch loop
    /// breaks back to the event loop when the frame stack returns to
    /// `event_loop_base_depth` (an async frame returned or suspended).
    event_loop_active: bool,
    event_loop_base_depth: usize,
    host_call_extension: Option<Box<dyn FnMut(&mut Vm<'a>, u32, &[Value], &mut Value) -> LanaError + Send>>,
    captured_output: Option<Arc<Mutex<String>>>,
    /// Optional in-memory filesystem. When set, the file-backed host calls
    /// (`read_text`, `write_text`, `write_text_atomic`, `path_exists`) resolve
    /// against this map instead of `std::fs`, so the self-hosted compiler can
    /// run on targets without a filesystem (e.g. `wasm32-unknown-unknown`).
    virtual_fs: Option<HashMap<String, String>>,
    package_paths: HashMap<String, String>,
    breakpoint_line: Option<u32>,
    breakpoint_hit: bool,
    /// LIP-018 two-way FFI: declared signatures and the single loaded library.
    /// `libloading` is unavailable on `wasm32`, so the loaded library is
    /// compiled out there and `ffi_load`/`ffi_call` return `UnsupportedOperation`.
    ffi_sigs: Vec<String>,
    #[cfg(not(target_arch = "wasm32"))]
    ffi_lib: Option<libloading::Library>,
    /// LIP-019 networking: open sockets, indexed by handle.
    sockets: Vec<NetSocket>,
    pure_callback_depth: usize,
    evaluation_callback_depth: usize,
}

/// LIP-018 two-way FFI: a parsed C-style signature and the bounded set of
/// marshallable types. `map`/`STATE`/`STATE_DIST`/`Information` are rejected.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FfiType {
    Void,
    Int,
    Double,
    String,
    Array,
}

/// LIP-019 networking: an open socket. Plain TCP, or TLS-wrapped when the
/// `net-tls` feature is enabled (https). The wasm target disables `net-tls`
/// and only ever holds plain sockets.
enum NetSocket {
    Plain(std::net::TcpStream),
    #[cfg(all(feature = "net-tls", not(target_arch = "wasm32")))]
    Tls(Box<rustls::StreamOwned<rustls::ClientConnection, std::net::TcpStream>>),
}

/// LIP-019 networking: a network failure, mapped to a `Result` error reason.
#[derive(Debug, PartialEq)]
enum NetError {
    Timeout,
    Io,
}

/// Parse an HTTP URI without allowing authority or request-line injection.
fn net_parse_url(url: &str) -> Option<(String, String, u16, String)> {
    if url.len() > http::HEADER_LIMIT || !url.is_ascii()
        || url.bytes().any(|byte| byte <= b' ' || byte == 127 || byte == b'\\') { return None; }
    let (scheme, rest) = url.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    if !matches!(scheme.as_str(), "http" | "https") { return None; }
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..end];
    if authority.is_empty() || authority.contains('@') { return None; }
    let default_port = if scheme == "https" { 443 } else { 80 };
    let (host, port) = if let Some(rest) = authority.strip_prefix('[') {
        let (host, suffix) = rest.split_once(']')?;
        host.parse::<std::net::Ipv6Addr>().ok()?;
        let port = if suffix.is_empty() { default_port } else {
            let port = suffix.strip_prefix(':')?;
            if port.is_empty() || !port.bytes().all(|byte| byte.is_ascii_digit()) { return None; }
            port.parse::<u16>().ok()?
        };
        (host, port)
    } else {
        let (host, port) = if let Some((host, port)) = authority.rsplit_once(':') {
            if port.is_empty() || !port.bytes().all(|byte| byte.is_ascii_digit()) { return None; }
            (host, port.parse::<u16>().ok()?)
        } else { (authority, default_port) };
        if host.is_empty() || !host.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"-._".contains(&byte)) { return None; }
        (host, port)
    };
    if port == 0 { return None; }
    let path = rest[end..].split('#').next().unwrap();
    let path = if path.starts_with('/') { path.to_string() } else { format!("/{path}") };
    let bytes = path.as_bytes();
    for (index, &byte) in bytes.iter().enumerate() {
        if byte == b'%' && (index + 2 >= bytes.len() || !bytes[index + 1..index + 3].iter().all(u8::is_ascii_hexdigit)) { return None; }
    }
    Some((scheme, host.to_string(), port, path))
}

/// Connect a TCP socket to `host:port` with a timeout.
fn net_connect(host: &str, port: u16, timeout_ms: u64) -> Result<std::net::TcpStream, NetError> {
    use std::net::TcpStream;
    use std::time::Duration;
    let addrs = std::net::ToSocketAddrs::to_socket_addrs(&(host, port)).map_err(|_| NetError::Io)?;
    let mut last_err = NetError::Io;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, Duration::from_millis(timeout_ms)) {
            Ok(s) => {
                let _ = s.set_read_timeout(Some(Duration::from_millis(timeout_ms)));
                let _ = s.set_write_timeout(Some(Duration::from_millis(timeout_ms)));
                return Ok(s);
            }
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => return Err(NetError::Timeout),
            Err(_) => last_err = NetError::Io,
        }
    }
    Err(last_err)
}

impl From<std::io::Error> for NetError {
    fn from(error: std::io::Error) -> Self {
        match error.kind() {
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => Self::Timeout,
            _ => Self::Io,
        }
    }
}

impl NetSocket {
    fn read(&mut self, buf: &mut [u8], timeout_ms: u64) -> Result<usize, NetError> {
        use std::io::Read;
        let timeout = Some(std::time::Duration::from_millis(timeout_ms));
        match self {
            NetSocket::Plain(socket) => {
                socket.set_read_timeout(timeout).map_err(NetError::from)?;
                socket.read(buf).map_err(NetError::from)
            }
            #[cfg(all(feature = "net-tls", not(target_arch = "wasm32")))]
            NetSocket::Tls(socket) => {
                socket.sock.set_read_timeout(timeout).map_err(NetError::from)?;
                socket.read(buf).map_err(NetError::from)
            }
        }
    }

    fn write_all(&mut self, data: &[u8]) -> Result<(), NetError> {
        use std::io::Write;
        match self {
            NetSocket::Plain(socket) => socket.write_all(data).map_err(NetError::from),
            #[cfg(all(feature = "net-tls", not(target_arch = "wasm32")))]
            NetSocket::Tls(socket) => socket.write_all(data).map_err(NetError::from),
        }
    }
}

/// Build a rustls client config: verify against the Mozilla roots when
/// `verify` is on, or accept any certificate when it is off (`verify:false`).
#[cfg(all(feature = "net-tls", not(target_arch = "wasm32")))]
fn net_tls_config(verify: bool) -> std::result::Result<rustls::ClientConfig, NetError> {
    use rustls::client::danger::{
        HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier,
    };
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::SignatureScheme;
    use std::sync::Arc;

    if verify {
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        return Ok(rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth());
    }

    // LIP-019 `verify:false` explicit opt-out: accept any server certificate.
    #[derive(Debug)]
    struct AcceptAny;
    impl ServerCertVerifier for AcceptAny {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp_response: &[u8],
            _now: UnixTime,
        ) -> std::result::Result<ServerCertVerified, rustls::Error> {
            Ok(ServerCertVerified::assertion())
        }
        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }
        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }
        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            use SignatureScheme::*;
            vec![
                RSA_PKCS1_SHA256,
                RSA_PKCS1_SHA384,
                RSA_PKCS1_SHA512,
                ECDSA_NISTP256_SHA256,
                ECDSA_NISTP384_SHA384,
                ECDSA_NISTP521_SHA512,
                RSA_PSS_SHA256,
                RSA_PSS_SHA384,
                RSA_PSS_SHA512,
                ED25519,
            ]
        }
    }
    Ok(rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAny))
        .with_no_client_auth())
}

/// Wrap a connected TCP stream in TLS for an https host, driving the
/// handshake to completion so certificate verification fails loudly.
#[cfg(all(feature = "net-tls", not(target_arch = "wasm32")))]
fn net_tls_connect(
    stream: std::net::TcpStream,
    host: &str,
    verify: bool,
) -> Result<NetSocket, NetError> {
    use std::sync::Arc;
    let config = Arc::new(net_tls_config(verify)?);
    let server_name = rustls::pki_types::ServerName::try_from(host.to_string())
        .map_err(|_| NetError::Io)?;
    let mut conn =
        rustls::ClientConnection::new(config, server_name).map_err(|_| NetError::Io)?;
    let mut tcp = stream;
    while conn.is_handshaking() {
        conn.complete_io(&mut tcp).map_err(NetError::from)?;
    }
    Ok(NetSocket::Tls(Box::new(rustls::StreamOwned::new(conn, tcp))))
}

struct FfiSignature {
    ret: FfiType,
    name: String,
    args: Vec<FfiType>,
}

fn ffi_type_end(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b')' | b',' | 0)
}

fn ffi_parse_type(bytes: &[u8], pos: &mut usize) -> Option<FfiType> {
    while *pos < bytes.len() && (bytes[*pos] == b' ' || bytes[*pos] == b'\t') {
        *pos += 1;
    }
    let rest = &bytes[*pos..];
    for (tok, ty) in [
        (b"void".as_slice(), FfiType::Void),
        (b"int".as_slice(), FfiType::Int),
        (b"double".as_slice(), FfiType::Double),
        (b"const char *".as_slice(), FfiType::String),
        (b"array".as_slice(), FfiType::Array),
    ] {
        if rest.starts_with(tok) {
            let end = *pos + tok.len();
            if end >= bytes.len() || ffi_type_end(bytes[end]) {
                *pos = end;
                return Some(ty);
            }
        }
    }
    None
}

fn parse_ffi_signature(sig: &str) -> Option<FfiSignature> {
    let bytes = sig.as_bytes();
    let mut pos = 0usize;
    let ret = ffi_parse_type(bytes, &mut pos)?;
    while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\t') {
        pos += 1;
    }
    let name_start = pos;
    while pos < bytes.len() && bytes[pos] != b'(' {
        pos += 1;
    }
    if pos >= bytes.len() {
        return None;
    }
    let name = sig[name_start..pos].trim().to_string();
    if name.is_empty() {
        return None;
    }
    pos += 1; // skip '('
    let mut args = Vec::new();
    loop {
        while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\t') {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b')' {
            return Some(FfiSignature { ret, name, args });
        }
        // C's `(void)` means no arguments.
        if bytes[pos..].starts_with(b"void)") {
            return Some(FfiSignature { ret, name, args });
        }
        let ty = ffi_parse_type(bytes, &mut pos)?;
        args.push(ty);
        while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\t') {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b',' {
            pos += 1;
            continue;
        }
        if pos < bytes.len() && bytes[pos] == b')' {
            return Some(FfiSignature { ret, name, args });
        }
        return None;
    }
}

/// A bounded FFI call failure: a marshalling/type error or an external failure
/// (missing symbol, load failure). A crash in the callee is not caught on the
/// Rust side (unlike the C11 VM's sigsetjmp guard); crash containment is C-only.
enum FfiError {
    Type,
    External,
}

/// Validate that the argument values match the declared signature types.
/// Mirrors `ffi_validate_args` in `vm/c/vm.c`.
fn ffi_validate_args(sig: &FfiSignature, args: &[Value]) -> bool {
    if args.len() != sig.args.len() {
        return false;
    }
    for (i, ty) in sig.args.iter().enumerate() {
        let ok = match ty {
            FfiType::Int | FfiType::Double => matches!(args[i].kind, ValueKind::Number(_)),
            FfiType::String => matches!(args[i].kind, ValueKind::String(_)),
            FfiType::Array => matches!(args[i].kind, ValueKind::Array(_)),
            FfiType::Void => false,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Marshal the arguments and invoke the symbol through libloading. The Rust
/// side supports a bounded set of call shapes: all-`double` or all-`int`
/// scalar arguments (0-4) returning `double`, `int`, or `void`. `string` and
/// `array` arguments are rejected with `FfiError::Type` in this initial
/// release; the C11 VM's libffi path is more general.
#[cfg(not(target_arch = "wasm32"))]
fn ffi_call_impl(
    lib: &libloading::Library,
    sig: &FfiSignature,
    args: &[Value],
) -> Result<Value, FfiError> {
    if args.len() != sig.args.len() {
        return Err(FfiError::Type);
    }
    let all_double = sig.args.iter().all(|t| *t == FfiType::Double);
    let all_int = sig.args.iter().all(|t| *t == FfiType::Int);
    if !all_double && !all_int {
        return Err(FfiError::Type);
    }
    let n = args.len();
    let mut dargs = [0.0f64; 4];
    let mut iargs = [0i32; 4];
    for (i, arg) in args.iter().enumerate() {
        let ValueKind::Number(num) = arg.kind else {
            return Err(FfiError::Type);
        };
        if all_double {
            dargs[i] = num;
        } else {
            iargs[i] = num as i32;
        }
    }
    let name = sig.name.as_bytes();
    let result: f64 = unsafe { if all_double {
        match (sig.ret, n) {
            (FfiType::Double, 0) => lib.get::<unsafe extern "C" fn() -> f64>(name)
                .map(|f| f())
                .map_err(|_| FfiError::External)?,
            (FfiType::Double, 1) => lib.get::<unsafe extern "C" fn(f64) -> f64>(name)
                .map(|f| f(dargs[0]))
                .map_err(|_| FfiError::External)?,
            (FfiType::Double, 2) => lib.get::<unsafe extern "C" fn(f64, f64) -> f64>(name)
                .map(|f| f(dargs[0], dargs[1]))
                .map_err(|_| FfiError::External)?,
            (FfiType::Double, 3) => {
                lib.get::<unsafe extern "C" fn(f64, f64, f64) -> f64>(name)
                    .map(|f| f(dargs[0], dargs[1], dargs[2]))
                    .map_err(|_| FfiError::External)?
            }
            (FfiType::Double, 4) => {
                lib.get::<unsafe extern "C" fn(f64, f64, f64, f64) -> f64>(name)
                    .map(|f| f(dargs[0], dargs[1], dargs[2], dargs[3]))
                    .map_err(|_| FfiError::External)?
            }
            (FfiType::Int, 0) => lib.get::<unsafe extern "C" fn() -> i32>(name)
                .map(|f| f() as f64)
                .map_err(|_| FfiError::External)?,
            (FfiType::Int, 1) => lib.get::<unsafe extern "C" fn(f64) -> i32>(name)
                .map(|f| f(dargs[0]) as f64)
                .map_err(|_| FfiError::External)?,
            (FfiType::Int, 2) => lib.get::<unsafe extern "C" fn(f64, f64) -> i32>(name)
                .map(|f| f(dargs[0], dargs[1]) as f64)
                .map_err(|_| FfiError::External)?,
            (FfiType::Int, 3) => {
                lib.get::<unsafe extern "C" fn(f64, f64, f64) -> i32>(name)
                    .map(|f| f(dargs[0], dargs[1], dargs[2]) as f64)
                    .map_err(|_| FfiError::External)?
            }
            (FfiType::Int, 4) => {
                lib.get::<unsafe extern "C" fn(f64, f64, f64, f64) -> i32>(name)
                    .map(|f| f(dargs[0], dargs[1], dargs[2], dargs[3]) as f64)
                    .map_err(|_| FfiError::External)?
            }
            (FfiType::Void, 0) => lib.get::<unsafe extern "C" fn()>(name)
                .map(|f| {
                    f();
                    0.0
                })
                .map_err(|_| FfiError::External)?,
            (FfiType::Void, 1) => lib.get::<unsafe extern "C" fn(f64)>(name)
                .map(|f| {
                    f(dargs[0]);
                    0.0
                })
                .map_err(|_| FfiError::External)?,
            (FfiType::Void, 2) => lib.get::<unsafe extern "C" fn(f64, f64)>(name)
                .map(|f| {
                    f(dargs[0], dargs[1]);
                    0.0
                })
                .map_err(|_| FfiError::External)?,
            (FfiType::Void, 3) => lib.get::<unsafe extern "C" fn(f64, f64, f64)>(name)
                .map(|f| {
                    f(dargs[0], dargs[1], dargs[2]);
                    0.0
                })
                .map_err(|_| FfiError::External)?,
            (FfiType::Void, 4) => {
                lib.get::<unsafe extern "C" fn(f64, f64, f64, f64)>(name)
                    .map(|f| {
                        f(dargs[0], dargs[1], dargs[2], dargs[3]);
                        0.0
                    })
                    .map_err(|_| FfiError::External)?
            }
            _ => return Err(FfiError::Type),
        }
    } else {
        match (sig.ret, n) {
            (FfiType::Double, 0) => lib.get::<unsafe extern "C" fn() -> f64>(name)
                .map(|f| f())
                .map_err(|_| FfiError::External)?,
            (FfiType::Double, 1) => lib.get::<unsafe extern "C" fn(i32) -> f64>(name)
                .map(|f| f(iargs[0]))
                .map_err(|_| FfiError::External)?,
            (FfiType::Double, 2) => lib.get::<unsafe extern "C" fn(i32, i32) -> f64>(name)
                .map(|f| f(iargs[0], iargs[1]))
                .map_err(|_| FfiError::External)?,
            (FfiType::Double, 3) => {
                lib.get::<unsafe extern "C" fn(i32, i32, i32) -> f64>(name)
                    .map(|f| f(iargs[0], iargs[1], iargs[2]))
                    .map_err(|_| FfiError::External)?
            }
            (FfiType::Double, 4) => {
                lib.get::<unsafe extern "C" fn(i32, i32, i32, i32) -> f64>(name)
                    .map(|f| f(iargs[0], iargs[1], iargs[2], iargs[3]))
                    .map_err(|_| FfiError::External)?
            }
            (FfiType::Int, 0) => lib.get::<unsafe extern "C" fn() -> i32>(name)
                .map(|f| f() as f64)
                .map_err(|_| FfiError::External)?,
            (FfiType::Int, 1) => lib.get::<unsafe extern "C" fn(i32) -> i32>(name)
                .map(|f| f(iargs[0]) as f64)
                .map_err(|_| FfiError::External)?,
            (FfiType::Int, 2) => lib.get::<unsafe extern "C" fn(i32, i32) -> i32>(name)
                .map(|f| f(iargs[0], iargs[1]) as f64)
                .map_err(|_| FfiError::External)?,
            (FfiType::Int, 3) => {
                lib.get::<unsafe extern "C" fn(i32, i32, i32) -> i32>(name)
                    .map(|f| f(iargs[0], iargs[1], iargs[2]) as f64)
                    .map_err(|_| FfiError::External)?
            }
            (FfiType::Int, 4) => {
                lib.get::<unsafe extern "C" fn(i32, i32, i32, i32) -> i32>(name)
                    .map(|f| f(iargs[0], iargs[1], iargs[2], iargs[3]) as f64)
                    .map_err(|_| FfiError::External)?
            }
            (FfiType::Void, 0) => lib.get::<unsafe extern "C" fn()>(name)
                .map(|f| {
                    f();
                    0.0
                })
                .map_err(|_| FfiError::External)?,
            (FfiType::Void, 1) => lib.get::<unsafe extern "C" fn(i32)>(name)
                .map(|f| {
                    f(iargs[0]);
                    0.0
                })
                .map_err(|_| FfiError::External)?,
            (FfiType::Void, 2) => lib.get::<unsafe extern "C" fn(i32, i32)>(name)
                .map(|f| {
                    f(iargs[0], iargs[1]);
                    0.0
                })
                .map_err(|_| FfiError::External)?,
            (FfiType::Void, 3) => lib.get::<unsafe extern "C" fn(i32, i32, i32)>(name)
                .map(|f| {
                    f(iargs[0], iargs[1], iargs[2]);
                    0.0
                })
                .map_err(|_| FfiError::External)?,
            (FfiType::Void, 4) => {
                lib.get::<unsafe extern "C" fn(i32, i32, i32, i32)>(name)
                    .map(|f| {
                        f(iargs[0], iargs[1], iargs[2], iargs[3]);
                        0.0
                    })
                    .map_err(|_| FfiError::External)?
            }
            _ => return Err(FfiError::Type),
        }
    } };
    Ok(Value::number(result))
}

impl<'a> Vm<'a> {
    /// Create a VM for a chunk. The CLI defaults (256 MiB / 50M instructions,
    /// seed `0x4c414e41`) match `tools/c/cli.c` `load_command`.
    pub fn new(chunk: &'a Chunk) -> Self {
        let heap = Heap::new(256 * 1024 * 1024);
        let mut vm = Self {
            chunk,
            ip: chunk.entry as usize,
            running: true,
            instruction_limit: 50_000_000,
            instruction_count: 0,
            dataset_work_remaining: None,
            dataset_decisions: Vec::new(),
            opcode_counts: vec![0; OpCode::Count as usize],
            object_descriptors: None,
            class_owner: Arc::new(()),
            class_objects: Vec::new(),
            class_objects_reservation: Arc::new(Mutex::new(heap.reserve(0).expect("empty class arena reservation"))),
            class_allocations_since_gc: 0,
            owned_cycles: Buffer::new(&heap, 0, 0).expect("empty collector registry"),
            host_roots: class_gc::RootOwner::new(&heap),
            constructing: Vec::new(),
            constructions: Vec::new(),
            object_methods: HashMap::new(),
            object_function_entries: Vec::new(),
            state_transition_count: 0,
            allocation_count: 0,
            memory_limit: 256 * 1024 * 1024,
            allocated_bytes: 0,
            heap,
            rng: Rng::new(),
            root_seed: 0,
            lineage: 0,
            revision: 0,
            derivation_sequence: 0,
            ad_recording: false,
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
            pending_io_outcome: None,
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
            ready_futures: VecDeque::new(),
            awaiters: HashMap::new(),
            event_loop_active: false,
            event_loop_base_depth: 0,
            host_call_extension: None,
            captured_output: None,
            virtual_fs: None,
            package_paths: HashMap::new(),
            breakpoint_line: None,
            breakpoint_hit: false,
            ffi_sigs: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            ffi_lib: None,
            sockets: Vec::new(),
            pure_callback_depth: 0,
            evaluation_callback_depth: 0,
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
    pub fn result(&self) -> Result<RootedValue, LanaError> {
        self.rooted_result()
    }

    pub fn set_breakpoint_line(&mut self, line: u32) { self.breakpoint_line = Some(line); self.breakpoint_hit = false; }
    pub fn debug_location(&self) -> Option<(usize, u32, String, usize)> {
        let instruction = self.chunk.code.get(self.ip)?;
        let function = self.frames.last().map(|frame| frame.function)
            .and_then(|index| self.chunk.functions.get(index as usize))
            .map(|function| function.name.clone()).unwrap_or_else(|| "<entry>".to_string());
        let function = self.frames.last().and_then(|frame| frame.object_method)
            .and_then(|(owner, member)| self.object_descriptors.as_ref()?.get(&owner).map(|descriptor| {
                let name = descriptor.qualified_name.rsplit('/').next().unwrap();
                if let Some(method) = descriptor.methods.get(member as usize) { format!("{name}.{}", method.name) }
                else { format!("{name}.<default:{}>", descriptor.fields[member as usize - descriptor.methods.len()].name) }
            })).unwrap_or(function);
        Some((self.ip, instruction.line, function, self.frames.len()))
    }
    pub fn debug_step(&mut self) -> LanaError { self.breakpoint_line = None; self.running = true; self.dispatch_loop(true) }
    pub fn debug_continue(&mut self) -> LanaError { self.breakpoint_line = None; self.running = true; self.run() }

    /// The error recorded by the last failed run.
    pub fn error(&self) -> &VmError {
        &self.error
    }

    /// The instruction count, for `--stats` output.
    pub fn instruction_count(&self) -> u64 {
        self.instruction_count
    }

    /// Charge bounded native host work to the same limit as bytecode execution.
    pub fn charge_bounded_work(&mut self, steps: u64) -> Result<(), LanaError> {
        if self.cancelled.load(Ordering::Relaxed) { return Err(LanaError::Cancelled); }
        if steps > self.instruction_limit.saturating_sub(self.instruction_count) {
            return Err(LanaError::Limit);
        }
        self.instruction_count += steps;
        Ok(())
    }

    /// The state transition count, for `--stats` output.
    pub fn state_transition_count(&self) -> u64 {
        self.state_transition_count
    }

    /// The allocation count, for `--stats` output.
    pub fn allocation_count(&self) -> u64 {
        self.allocation_count.saturating_add(self.heap.allocations())
    }

    /// The cumulative allocated bytes, for `--stats` output.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes + self.heap.live_bytes()
    }

    pub fn heap(&self) -> Heap { self.heap.clone() }

    /// Construct an isolated replay root using the same boundary as Core source values.
    pub fn information_root(&mut self, source: &Value) -> Result<Value, LanaError> {
        self.reactive_root(source, DerivationExactness::Exact)
    }

    /// Replay explicit evidence through the ordinary transactional Core observation.
    pub fn information_observe(&mut self, root: &Value, evidence: &Value) -> Result<Value, LanaError> {
        if self.chunk.version < lana_bytecode::opcode::LABC_VERSION_5 { return Err(LanaError::UnsupportedOperation); }
        self.reactive_observe(root, evidence, 0)?;
        Ok(self.reactive_value(root))
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
                    let _shutdown = SchedulerShutdown(&scheduler);
                    self.dispatch_loop(false)
                })
            }
            #[cfg(target_arch = "wasm32")]
            {
                let scheduler = Scheduler::new();
                self.scheduler = Some(scheduler.clone());
                let _shutdown = SchedulerShutdown(&scheduler);
                self.dispatch_loop(false)
            }
        } else {
            self.dispatch_loop(false)
        }
    }

    /// The dispatch loop, mirroring the body of `lana_vm_run`.
    fn dispatch_loop(&mut self, single_step: bool) -> LanaError {
        while self.running {
            // The event loop (LIP-024 §6) drives async frames. When an async
            // frame returns or suspends, the frame stack returns to the depth
            // at which the event loop started; break back to the loop so it can
            // schedule the next ready future.
            if self.event_loop_active && self.frames.len() <= self.event_loop_base_depth {
                self.event_loop_active = false;
                break;
            }
            if self.allocated_bytes() > self.memory_limit {
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
            if self.breakpoint_line == Some(instruction.line) {
                self.breakpoint_hit = true;
                self.running = false;
                break;
            }
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
            if single_step { self.running = false; }
        }
        LanaError::Ok
    }

    /// Enqueue a future on the event loop's ready queue (LIP-024 §6). A future
    /// is enqueued at most once: exhausted futures are skipped, and a future
    /// already in the queue is not re-enqueued.
    fn enqueue_future(&mut self, future: Arc<Mutex<Future>>) {
        let should_enqueue = {
            let mut guard = future.lock().unwrap();
            if guard.exhausted || guard.queued {
                false
            } else {
                self.heap.mutate_cycle(Arc::as_ptr(&future) as usize);
                guard.queued = true;
                guard.ready = true;
                true
            }
        };
        if should_enqueue {
            self.ready_futures.push_back(future);
        }
    }

    /// Run the single-threaded event loop to completion (LIP-024 §6). Pops the
    /// oldest ready future, pushes a fresh async frame restoring its registers,
    /// and drives the dispatch loop until the frame returns or suspends. When
    /// the ready queue is empty, the loop is done.
    fn run_event_loop(&mut self) -> LanaError {
        while let Some(future) = self.ready_futures.pop_front() {
            {
                let mut guard = future.lock().unwrap();
                self.heap.mutate_cycle(Arc::as_ptr(&future) as usize);
                guard.queued = false;
                if guard.exhausted {
                    continue;
                }
            }
            // Composite futures (`future_all`, `future_race`, `sleep`) have no
            // async function to run; poll them for completion instead.
            let is_composite = {
                let guard = future.lock().unwrap();
                guard.function == u32::MAX
            };
            if is_composite {
                if let Err(error) = self.poll_composite_future(future) { return error; }
                continue;
            }
            let (function, ip, registers) = {
                let guard = future.lock().unwrap();
                (guard.function, guard.ip, guard.registers.clone())
            };
            if self.frames.len() >= LANA_MAX_CALL_FRAMES as usize {
                return LanaError::Limit;
            }
            let mut callee = Frame::new(registers.len());
            for index in 0..registers.len() {
                callee.registers[index] = registers[index].clone();
            }
            callee.registers[0] = Value::future(future.clone());
            callee.return_ip = 0;
            callee.return_register = 0;
            callee.function = function;
            callee.is_async = true;
            self.frames.push(callee);
            self.ip = ip;
            self.event_loop_active = true;
            let error = self.dispatch_loop(false);
            if error != LanaError::Ok {
                return error;
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
    /// A limit below current live allocations fails without changing the limit.
    pub fn set_memory_limit(&mut self, bytes: usize) -> Result<(), LanaError> {
        let available = bytes.checked_sub(self.allocated_bytes).ok_or(LanaError::Oom)?;
        self.heap.set_limit(available)?;
        self.memory_limit = bytes;
        Ok(())
    }

    /// Set the program arguments exposed to the `args` host call, mirroring
    /// `lana_vm_set_program_args`. Child VMs inherit the parent's arguments at
    /// FORK time.
    pub fn set_program_args(&mut self, args: &[String]) {
        self.program_argc = args.len();
        self.program_argv = args.iter().map(|s| Arc::from(s.as_str())).collect();
    }

    /// Compiler-only roots supplied after the CLI verifies the lock and cache.
    pub fn set_package_paths(&mut self, paths: HashMap<String, String>) {
        self.package_paths = paths;
    }

    /// Register a handler for host-call IDs beyond the built-in set (54). When
    /// `execute_host_call` sees an ID it does not recognize, it delegates to
    /// this handler. The CLI uses this to expose the durable pipeline
    /// (store/policy/ledger) to Lana bytecode without `lana-vm` depending on
    /// `lana-runtime`. The handler receives the same VM so a store call can
    /// evaluate a pure named plan before committing its result.
    pub fn set_host_call_extension(
        &mut self,
        handler: Box<dyn FnMut(&mut Vm<'a>, u32, &[Value], &mut Value) -> LanaError + Send>,
    ) {
        self.host_call_extension = Some(handler);
    }

    /// Capture PRINT output for the JSON worker without mixing it with the protocol.
    pub fn capture_output(&mut self) {
        self.captured_output = Some(Arc::new(Mutex::new(String::new())));
    }

    pub fn output(&self) -> Option<String> {
        self.captured_output.as_ref().map(|output| output.lock().unwrap().clone())
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
        if let Err(error) = child.set_memory_limit(self.memory_limit) {
            scheduler.state.lock().unwrap().live_tasks -= 1;
            return Err(error);
        }
        child.program_argc = self.program_argc;
        child.program_argv = self.program_argv.clone();
        child.package_paths = self.package_paths.clone();
        child.captured_output = self.captured_output.clone();
        child.lineage = mix64(self.lineage ^ { self.spawn_counter += 1; self.spawn_counter });
        child.seed(mix64(self.root_seed ^ child.lineage));
        child.root_seed = self.root_seed;
        child.cancelled = self.cancelled.clone();
        let mut memo = DeepCloneMemo { transfer: true, ..DeepCloneMemo::default() };
        for index in 0..argc as usize {
            let argument = self.current_frame().registers[(first_arg as usize) + index].clone();
            let cloned = match child.deep_clone_value(&argument, &mut memo) {
                Ok(value) => value,
                Err(error) => { scheduler.state.lock().unwrap().live_tasks -= 1; return Err(error); }
            };
            child.frames[0].registers[index] = cloned;
        }
        for index in 0..argc as usize {
            let history = self.current_frame().histories[(first_arg as usize) + index].clone();
            child.frames[0].histories[index] = history;
        }
        if let Err(error) = self.track_cycle(crate::heap::CycleWeak::Task(Arc::downgrade(&handle))) {
            scheduler.state.lock().unwrap().live_tasks -= 1;
            return Err(error);
        }
        let service_root = match self.host_roots.register_root(Value::task(handle.clone()), 0, 0) {
            Ok(root) => root,
            Err(error) => { scheduler.state.lock().unwrap().live_tasks -= 1; return Err(error); }
        };
        {
            let mut state = scheduler.state.lock().unwrap();
            if self.cancelled.load(Ordering::Relaxed) || state.stopping {
                state.live_tasks -= 1;
                return Err(if self.cancelled.load(Ordering::Relaxed) { LanaError::Cancelled } else { LanaError::Task });
            }
            child.cancelled = handle.cancelled.clone();
            state.queue.push_back(QueuedTask {
                child,
                handle: handle.clone(),
            });
            state.all_tasks.push((handle.clone(), service_root));
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
        #[cfg(target_arch = "wasm32")]
        if timeout >= 0.0 { return Err(LanaError::UnsupportedOperation); }
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
        class_gc::RootOwner::drain_retired(&mut |steps| self.charge_bounded_work(steps))?;
        let mut state = task.state.lock().unwrap();
        if state.status != LanaError::Ok {
            self.error = state.error.clone();
            return Err(state.status);
        }
        if !state.joined {
            let mut memo = DeepCloneMemo { transfer: true, ..DeepCloneMemo::default() };
            let checkpoint = self.class_objects.len();
            let cloned = match self.deep_clone_value(&state.result, &mut memo) {
                Ok(value) => value,
                Err(error) => { self.class_objects.truncate(checkpoint); return Err(error); }
            };
            let _defer = class_gc::RootOwner::defer_retired();
            self.heap.mutate_cycle(task as *const Task as usize);
            state.result = cloned;
            state.result_root = None;
            state.joined = true;
            let scheduler = self.scheduler.as_ref().expect("scheduler exists");
            let mut scheduler_state = scheduler.state.lock().unwrap();
            if scheduler_state.live_tasks > 0 {
                scheduler_state.live_tasks -= 1;
            }
            scheduler_state.all_tasks.retain(|(handle, _)| handle.id != task.id);
            self.tasks.retain(|handle| handle.id != task.id);
        }
        let result = state.result.clone();
        drop(state);
        class_gc::RootOwner::drain_retired(&mut |steps| self.charge_bounded_work(steps))?;
        Ok(result)
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
        if let Some(construction) = self.constructions.first() {
            let checkpoint = construction.checkpoint;
            self.class_objects.truncate(checkpoint);
            self.constructing.clear();
            self.constructions.clear();
        }
        // Mirror `vm_fail` in `vm/c/vm.c:362-366`: when an error is already
        // recorded (e.g. a child task's error propagated by JOIN), preserve it
        // instead of overwriting with the failing instruction's own span.
        if self.error.code != LanaError::Ok && !self.error.message.is_empty() {
            self.result = Value::null();
            self.running = false;
            return code;
        }
        let message = message.into();
        let io_outcome = self.pending_io_outcome.take();
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
            durability: io_outcome.as_ref().and_then(|(uncertain, _)| uncertain.then(|| "uncertain".to_string())),
            path: io_outcome.map(|(_, path)| path),
        };
        match code {
            LanaError::Cancelled => {
                error.cancellation = Some((self.lineage, message));
            }
            LanaError::Oom => {
                error.resource_limit = Some((
                    LANA_RESOURCE_MEMORY,
                    self.memory_limit as u64,
                    self.allocated_bytes() as u64,
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
        let payload = self.record_derivation_payload(kind, operation, inputs, label, line,
            exactness, details, outcome, reason)?;
        self.managed_payload(payload).ok()
    }

    fn record_derivation_payload(
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
    ) -> Option<Derivation> {
        let mut retained: Vec<Arc<Derivation>> = Vec::new();
        retained.try_reserve_exact(inputs.len()).ok()?;
        for input in inputs {
            if let Some(derivation) = &input.derivation {
                retained.push(derivation.clone());
            }
        }
        self.derivation_sequence += 1;
        let autodiff = details == "autodiff" && (operation == "autodiff" || operation == "input");
        Some(Derivation {
            task_lineage: self.lineage,
            local_sequence: self.derivation_sequence,
            revision: self.revision,
            kind,
            operation: if autodiff {
                shared_derivation_text(if operation == "input" { "input" } else { "autodiff" })
            } else {
                self.heap.string(operation).ok()?
            },
            inputs: retained,
            label: if autodiff {
                shared_derivation_text("")
            } else {
                self.heap.string(label).ok()?
            },
            function: self.heap.string(&self.current_function_name()).ok()?,
            line,
            exactness,
            details: if autodiff {
                shared_derivation_text("autodiff")
            } else {
                self.heap.string(details).ok()?
            },
            outcome,
            reason: if autodiff {
                shared_derivation_text("none")
            } else {
                self.heap.string(reason).ok()?
            },
            ad_op: -1,
            ad_a: None,
            ad_b: None,
            ad_a_deriv: None,
            ad_b_deriv: None,
            ad_grad: Arc::new(Mutex::new(None)),
            ad_axis: -1,
        })
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

    /// Run a single-argument function to completion, mirroring the nested
    /// execution loop used by `OP_BOOTSTRAP`. The result is written to
    /// `scratch_register` in the caller frame and copied to `result`.
    fn run_function(&mut self, function_index: u32, arg: &Value, scratch_register: u32, result: &mut Value) -> LanaError {
        self.run_function_args(function_index, std::slice::from_ref(arg), scratch_register, result)
    }

    /// Run a two-argument function to completion, mirroring `run_function2` in
    /// `vm/c/vm.c`. The result is written to `scratch_register` in the caller
    /// frame and copied to `result`.
    fn run_function2(
        &mut self,
        function_index: u32,
        arg0: &Value,
        arg1: &Value,
        scratch_register: u32,
        result: &mut Value,
    ) -> LanaError {
        self.run_function_args(function_index, &[arg0.clone(), arg1.clone()], scratch_register, result)
    }

    fn run_function_args(&mut self, function_index: u32, args: &[Value], scratch_register: u32, result: &mut Value) -> LanaError {
        self.run_function_args_owned(function_index, args, scratch_register, result, None)
    }

    fn run_function_args_owned(&mut self, function_index: u32, args: &[Value], scratch_register: u32,
        result: &mut Value, owner: Option<(u32, u32)>) -> LanaError {
        let Some(function) = self.chunk.functions.get(function_index as usize) else { return LanaError::Type; };
        let arity = function.arity;
        let entry = self.chunk.functions[function_index as usize].entry as usize;
        if arity as usize != args.len() {
            return LanaError::Type;
        }
        if args.len() > self.max_registers[function_index as usize] as usize
            || scratch_register as usize >= self.current_frame().registers.len() {
            return LanaError::Format;
        }
        if self.frames.len() >= LANA_MAX_CALL_FRAMES as usize {
            return LanaError::Limit;
        }
        let saved_frame_count = self.frames.len();
        let saved_ip = self.ip;
        let mut callee = if self.chunk.version == 6 {
            match Frame::charged(self.max_registers[function_index as usize], &self.heap) {
                Ok(frame) => frame, Err(error) => return error,
            }
        } else { Frame::new(self.max_registers[function_index as usize]) };
        callee.return_ip = self.ip;
        callee.return_register = scratch_register;
        callee.function = function_index;
        callee.object_method = owner;
        callee.registers[..args.len()].clone_from_slice(args);
        self.frames.push(callee);
        self.ip = entry;
        while self.frames.len() > saved_frame_count && self.running {
            if self.cancelled.load(Ordering::Relaxed) {
                self.frames.truncate(saved_frame_count);
                self.ip = saved_ip;
                return LanaError::Cancelled;
            }
            if self.allocated_bytes() > self.memory_limit {
                self.frames.truncate(saved_frame_count);
                self.ip = saved_ip;
                return LanaError::Oom;
            }
            if self.ip >= self.chunk.code.len() {
                self.frames.truncate(saved_frame_count);
                self.ip = saved_ip;
                return LanaError::Jump;
            }
            let old_count = self.instruction_count;
            self.instruction_count += 1;
            if old_count >= self.instruction_limit {
                self.frames.truncate(saved_frame_count);
                self.ip = saved_ip;
                return LanaError::Limit;
            }
            let instruction = self.chunk.code[self.ip];
            self.ip += 1;
            self.opcode_counts[instruction.opcode as usize] += 1;
            let error = self.execute(&instruction);
            if error != LanaError::Ok {
                self.frames.truncate(saved_frame_count);
                self.ip = saved_ip;
                return error;
            }
        }
        *result = self.frames[saved_frame_count - 1].registers[scratch_register as usize].clone();
        LanaError::Ok
    }

    /// Execute a named dataset plan inside the current VM and its resource limits.
    /// The caller owns publication; a failed plan returns no result.
    pub fn run_pure_dataset_plan(&mut self, name: &str, args: &[Value]) -> Result<Value, LanaError> {
        self.dataset_decisions.clear();
        let index = self.chunk.functions.iter().position(|function| function.name == name)
            .ok_or(LanaError::NotFound)? as u32;
        if self.chunk.functions[index as usize].arity as usize != args.len() { return Err(LanaError::Type); }
        let saved = self.current_frame().registers[0].clone();
        let mut result = Value::null();
        let previous_budget = self.dataset_work_remaining.replace(5_000_000);
        self.pure_callback_depth += 1;
        let error = self.run_function_args(index, args, 0, &mut result);
        self.pure_callback_depth -= 1;
        self.dataset_work_remaining = previous_budget;
        self.current_frame_mut().registers[0] = saved;
        if error != LanaError::Ok { self.dataset_decisions.clear(); return Err(error); }
        let ValueKind::Array(rows) = &result.kind else { self.dataset_decisions.clear(); return Err(LanaError::Type); };
        let rows = rows.lock().unwrap();
        if rows.items().len() > 100_000 { self.dataset_decisions.clear(); return Err(LanaError::Limit); }
        if rows.items().iter().any(|row| !matches!(row.kind, ValueKind::Map(_))) {
            self.dataset_decisions.clear();
            return Err(LanaError::Type);
        }
        drop(rows);
        Ok(result)
    }

    /// Attach a source identity without inserting an `id` column into the row.
    pub fn dataset_source_row(&mut self, source_id: &str, row_id: &str, mut row: Value) -> Result<Value, LanaError> {
        if source_id.is_empty() || source_id.len() > 128 || row_id.is_empty() || row_id.len() > 128 {
            return Err(LanaError::InvalidParameters);
        }
        let ValueKind::Map(source) = &row.kind else { return Err(LanaError::Type); };
        let label = format!("{}:{}{}:{}", source_id.len(), source_id, row_id.len(), row_id);
        row.derivation = self.record_derivation(DerivationKind::Evidence, "dataset_source", &[&row], &label, 0,
            DerivationExactness::Exact, "source_row", DerivationOutcome::Success, "none");
        let mut cells = Map::new(&self.heap, source.lock().unwrap().entries().len())?;
        for entry in source.lock().unwrap().entries() {
            let mut cell = entry.value.clone();
            if cell.derivation.is_none() { cell.derivation = row.derivation.clone(); }
            cells.set(entry.key.clone(), cell, false)?;
        }
        row.kind = ValueKind::Map(Arc::new(Mutex::new(cells)));
        Ok(row)
    }

    pub fn dataset_decisions(&self) -> &[DatasetDecision] { &self.dataset_decisions }

    pub fn clear_dataset_decisions(&mut self) { self.dataset_decisions.clear(); }

    /// Record a differentiable primitive onto `result`'s derivation, mirroring
    /// `ad_record` in `vm/c/vm.c`. `ad_op` is 0=add 1=sub 2=mul 3=div 4=matmul
    /// 5=sum 6=mean 10=reshape 11=transpose 12=exp 13=log 14=sqrt 15=relu
    /// 16=softmax 17=logsumexp 18=gather 19=cholesky_solve.
    fn ad_record(
        &mut self,
        ad_op: i32,
        a: &Value,
        b: Option<&Value>,
        ad_axis: i32,
        result: &mut Value,
    ) -> LanaError {
        let mut retained: Vec<Arc<Derivation>> = Vec::new();
        if let Some(derivation) = &a.derivation {
            retained.push(derivation.clone());
        }
        if let Some(bv) = b {
            if let Some(derivation) = &bv.derivation {
                retained.push(derivation.clone());
            }
        }
        let a_tensor = match value_tensor(a) {
            Some(t) => t.clone(),
            None => return LanaError::Type,
        };
        let b_tensor = match b {
            Some(bv) => match value_tensor(bv) {
                Some(t) => Some(t.clone()),
                None => return LanaError::Type,
            },
            None => None,
        };
        self.derivation_sequence += 1;
        let function_name = self.current_function_name();
        let node = match self.managed_payload(Derivation {
            task_lineage: self.lineage,
            local_sequence: self.derivation_sequence,
            revision: self.revision,
            kind: DerivationKind::Operation,
            operation: shared_derivation_text("autodiff"),
            inputs: retained,
            label: shared_derivation_text(""),
            function: match self.heap.string(&function_name) { Ok(text) => text, Err(error) => return error },
            line: 0,
            exactness: DerivationExactness::Exact,
            details: shared_derivation_text("autodiff"),
            outcome: DerivationOutcome::Success,
            reason: shared_derivation_text("none"),
            ad_op,
            ad_a: Some(a_tensor.clone()),
            ad_b: b_tensor,
            ad_a_deriv: a.derivation.clone(),
            ad_b_deriv: b.and_then(|bv| bv.derivation.clone()),
            ad_grad: Arc::new(Mutex::new(None)),
            ad_axis,
        }) { Ok(node) => node, Err(error) => return error };
        result.derivation = Some(node);
        LanaError::Ok
    }

    /// Reverse-mode backward pass, mirroring `ad_backward` in `vm/c/vm.c`.
    /// `seed` is the cotangent of `node`'s output, a contiguous base tensor.
    /// Accumulates into each node's `ad_grad` and recurses into the input
    /// derivations in left-then-right order.
    fn ad_backward(&mut self, node: &Arc<Derivation>, seed: &Tensor) -> LanaError {
        let count = tensor::tensor_element_count(seed);
        {
            let mut guard = node.ad_grad.lock().unwrap();
            if guard.is_none() {
                let alloc = self.heap.clone();
                let t = match tensor::tensor_new(&alloc, seed.ndim, &seed.shape, seed.is_complex) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *guard = Some(t);
            }
            let grad = guard.as_mut().unwrap();
            for e in 0..count {
                let cur = tensor_get_real(grad, e) + tensor_get_real(seed, seed.offset + e);
                tensor_set_real(grad, e, cur);
                let curi = tensor_get_imag(grad, e) + tensor_get_imag(seed, seed.offset + e);
                tensor_set_imag(grad, e, curi);
            }
        }
        if node.ad_op < 0 {
            return LanaError::Ok;
        }
        match node.ad_op {
            0 | 1 | 2 | 3 => {
                let a = node.ad_a.as_ref().unwrap();
                let b = node.ad_b.as_ref().unwrap();
                let (ga, gb) = {
                    let alloc = self.heap.clone();
                    match node.ad_op {
                        0 => (
                            match seed.try_clone(&alloc) { Ok(t) => t, Err(error) => return error },
                            match seed.try_clone(&alloc) { Ok(t) => t, Err(error) => return error },
                        ),
                        1 => {
                            let gb = match tensor::tensor_negate(&alloc, seed) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            (match seed.try_clone(&alloc) { Ok(t) => t, Err(error) => return error }, gb)
                        }
                        2 => {
                            let ga = match tensor::tensor_elementwise(&alloc, seed, b, 2) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            let gb = match tensor::tensor_elementwise(&alloc, seed, a, 2) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            (ga, gb)
                        }
                        _ => {
                            let ga = match tensor::tensor_elementwise(&alloc, seed, b, 3) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            let t1 = match tensor::tensor_elementwise(&alloc, seed, a, 2) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            let t2 = match tensor::tensor_elementwise(&alloc, b, b, 2) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            let t3 = match tensor::tensor_elementwise(&alloc, &t1, &t2, 3) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            let gb = match tensor::tensor_negate(&alloc, &t3) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            (ga, gb)
                        }
                    }
                };
                let (ga_u, gb_u) = {
                    let alloc = self.heap.clone();
                    let ga_u = match tensor::tensor_unbroadcast(&alloc, &ga, a) {
                        Ok(t) => t,
                        Err(error) => return error,
                    };
                    let gb_u = match tensor::tensor_unbroadcast(&alloc, &gb, b) {
                        Ok(t) => t,
                        Err(error) => return error,
                    };
                    (ga_u, gb_u)
                };
                if let Some(a_deriv) = &node.ad_a_deriv {
                    let error = self.ad_backward(a_deriv, &ga_u);
                    if error != LanaError::Ok {
                        return error;
                    }
                }
                if let Some(b_deriv) = &node.ad_b_deriv {
                    let error = self.ad_backward(b_deriv, &gb_u);
                    if error != LanaError::Ok {
                        return error;
                    }
                }
                LanaError::Ok
            }
            4 => {
                let a = node.ad_a.as_ref().unwrap();
                let b = node.ad_b.as_ref().unwrap();
                let a_ndim = a.ndim;
                let b_ndim = b.ndim;
                let (ga, gb) = {
                    let alloc = self.heap.clone();
                    if a_ndim == 1 && b_ndim == 1 {
                        let ga = match tensor::tensor_elementwise(&alloc, seed, b, 2) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let gb = match tensor::tensor_elementwise(&alloc, seed, a, 2) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        (ga, gb)
                    } else if a_ndim == 1 {
                        let bt = match tensor::tensor_transpose_last_two(&alloc, b) { Ok(t) => t, Err(error) => return error };
                        let ga = match tensor::tensor_matmul(
                            &alloc, seed, &bt, tensor::matmul_default_dtype(seed, &bt),
                        ) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let gb = match tensor::tensor_outer(&alloc, a, seed) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        (ga, gb)
                    } else if b_ndim == 1 {
                        let at = match tensor::tensor_transpose_last_two(&alloc, a) { Ok(t) => t, Err(error) => return error };
                        let ga = match tensor::tensor_outer(&alloc, seed, b) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let gb = match tensor::tensor_matmul(
                            &alloc, &at, seed, tensor::matmul_default_dtype(&at, seed),
                        ) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        (ga, gb)
                    } else {
                        let bt = match tensor::tensor_transpose_last_two(&alloc, b) { Ok(t) => t, Err(error) => return error };
                        let at = match tensor::tensor_transpose_last_two(&alloc, a) { Ok(t) => t, Err(error) => return error };
                        let ga_raw = match tensor::tensor_matmul(
                            &alloc, seed, &bt, tensor::matmul_default_dtype(seed, &bt),
                        ) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let gb_raw = match tensor::tensor_matmul(
                            &alloc, &at, seed, tensor::matmul_default_dtype(&at, seed),
                        ) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let ga = match tensor::tensor_unbroadcast(&alloc, &ga_raw, a) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let gb = match tensor::tensor_unbroadcast(&alloc, &gb_raw, b) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        (ga, gb)
                    }
                };
                if let Some(a_deriv) = &node.ad_a_deriv {
                    let error = self.ad_backward(a_deriv, &ga);
                    if error != LanaError::Ok {
                        return error;
                    }
                }
                if let Some(b_deriv) = &node.ad_b_deriv {
                    let error = self.ad_backward(b_deriv, &gb);
                    if error != LanaError::Ok {
                        return error;
                    }
                }
                LanaError::Ok
            }
            5 | 6 => {
                let a = node.ad_a.as_ref().unwrap();
                let n = if node.ad_axis < 0 {
                    tensor::tensor_element_count(a)
                } else {
                    a.shape[node.ad_axis as usize]
                };
                let scale = if node.ad_op == 6 { 1.0 / n as f64 } else { 1.0 };
                let ga = {
                    let alloc = self.heap.clone();
                    match tensor::tensor_broadcast_reduce(&alloc, seed, a, node.ad_axis, scale) {
                        Ok(t) => t,
                        Err(error) => return error,
                    }
                };
                if let Some(a_deriv) = &node.ad_a_deriv {
                    let error = self.ad_backward(a_deriv, &ga);
                    if error != LanaError::Ok {
                        return error;
                    }
                }
                LanaError::Ok
            }
            10 | 11 => {
                let a = node.ad_a.as_ref().unwrap();
                let mut ga = {
                    let alloc = self.heap.clone();
                    match tensor::tensor_new(&alloc, a.ndim, &a.shape, false) {
                        Ok(t) => t,
                        Err(error) => return error,
                    }
                };
                let total = tensor::tensor_element_count(a);
                for linear in 0..total {
                    let source_index = if node.ad_op == 10 {
                        let mut rem = linear;
                        let mut index = seed.offset;
                        for i in (0..seed.ndim).rev() {
                            index += (if seed.shape[i] == 0 { 0 } else { rem % seed.shape[i] }) * seed.strides[i];
                            rem /= seed.shape[i];
                        }
                        index
                    } else {
                        if seed.ndim < 2 { return LanaError::InvalidParameters; }
                        let mut rem = linear;
                        let mut coordinates = [0; tensor::TENSOR_MAX_RANK];
                        for i in (0..a.ndim).rev() {
                            coordinates[i] = if a.shape[i] == 0 { 0 } else { rem % a.shape[i] };
                            rem /= a.shape[i];
                        }
                        let last = a.ndim - 1;
                        let second = a.ndim - 2;
                        coordinates.swap(last, second);
                        (0..seed.ndim).fold(seed.offset, |index, i| index + coordinates[i] * seed.strides[i])
                    };
                    tensor_set_real(&mut ga, linear, tensor_get_real(seed, source_index));
                }
                if let Some(a_deriv) = &node.ad_a_deriv { return self.ad_backward(a_deriv, &ga); }
                LanaError::Ok
            }
            12 | 13 | 14 | 15 => {
                let a = node.ad_a.as_ref().unwrap();
                let mut ga = {
                    let alloc = self.heap.clone();
                    match tensor::tensor_new(&alloc, a.ndim, &a.shape, false) {
                        Ok(t) => t,
                        Err(error) => return error,
                    }
                };
                for i in 0..tensor::tensor_element_count(a) {
                    let mut rem = i;
                    let mut a_index = a.offset;
                    let mut seed_index = seed.offset;
                    for axis in (0..a.ndim).rev() {
                        let coordinate = if a.shape[axis] == 0 { 0 } else { rem % a.shape[axis] };
                        rem /= a.shape[axis];
                        a_index += coordinate * a.strides[axis];
                        seed_index += coordinate * seed.strides[axis];
                    }
                    let x = tensor_get_real(a, a_index);
                    let derivative = match node.ad_op {
                        12 => x.exp(),
                        13 => 1.0 / x,
                        14 => 0.5 / x.sqrt(),
                        _ => if x > 0.0 { 1.0 } else { 0.0 },
                    };
                    tensor_set_real(&mut ga, i, tensor_get_real(seed, seed_index) * derivative);
                }
                if let Some(a_deriv) = &node.ad_a_deriv { return self.ad_backward(a_deriv, &ga); }
                LanaError::Ok
            }
            16 | 17 => {
                let a = node.ad_a.as_ref().unwrap();
                let axis = node.ad_axis as usize;
                let width = a.shape[axis];
                let inner = a.shape[axis + 1..].iter().product::<usize>();
                let outer = a.shape[..axis].iter().product::<usize>();
                let mut ga = {
                    let alloc = self.heap.clone();
                    match tensor::tensor_new(&alloc, a.ndim, &a.shape, false) { Ok(t) => t, Err(error) => return error }
                };
                for o in 0..outer { for ii in 0..inner {
                    let base = o * width * inner + ii;
                    let mut maximum = f64::NEG_INFINITY;
                    for k in 0..width { maximum = maximum.max(tensor_get_real(a, tensor::logical_index(a, base + k * inner))); }
                    let mut total = 0.0;
                    for k in 0..width { total += (tensor_get_real(a, tensor::logical_index(a, base + k * inner)) - maximum).exp(); }
                    let mut weighted = 0.0;
                    if node.ad_op == 16 { for k in 0..width {
                        let probability = (tensor_get_real(a, tensor::logical_index(a, base + k * inner)) - maximum).exp() / total;
                        weighted += tensor_get_real(seed, tensor::logical_index(seed, base + k * inner)) * probability;
                    } }
                    for k in 0..width {
                        let probability = (tensor_get_real(a, tensor::logical_index(a, base + k * inner)) - maximum).exp() / total;
                        let incoming = if node.ad_op == 16 { tensor_get_real(seed, tensor::logical_index(seed, base + k * inner)) - weighted }
                            else { tensor_get_real(seed, tensor::logical_index(seed, o * inner + ii)) };
                        tensor_set_real(&mut ga, base + k * inner, probability * incoming);
                    }
                } }
                if let Some(a_deriv) = &node.ad_a_deriv { return self.ad_backward(a_deriv, &ga); }
                LanaError::Ok
            }
            18 => {
                let source = node.ad_a.as_ref().unwrap();
                let indices = node.ad_b.as_ref().unwrap();
                let axis = node.ad_axis as usize;
                let mut ga = {
                    let alloc = self.heap.clone();
                    match tensor::tensor_new(&alloc, source.ndim, &source.shape, false) { Ok(t) => t, Err(error) => return error }
                };
                let mut coordinates = [0; tensor::TENSOR_MAX_RANK];
                for linear in 0..tensor::tensor_element_count(seed) {
                    let mut rem = linear;
                    for i in (0..seed.ndim).rev() { coordinates[i] = if seed.shape[i] == 0 { 0 } else { rem % seed.shape[i] }; rem /= seed.shape[i]; }
                    let index_linear = indices.shape.iter().enumerate().fold(0, |n, (i, width)| n * width + coordinates[axis + i]);
                    let mut requested = tensor_get_real(indices, tensor::logical_index(indices, index_linear));
                    if requested < 0.0 { requested += source.shape[axis] as f64; }
                    let mut source_linear = 0;
                    for i in 0..axis { source_linear = source_linear * source.shape[i] + coordinates[i]; }
                    source_linear = source_linear * source.shape[axis] + requested as usize;
                    for i in axis + 1..source.ndim { source_linear = source_linear * source.shape[i] + coordinates[i - 1 + indices.ndim]; }
                    let value = tensor_get_real(&ga, source_linear) + tensor_get_real(seed, tensor::logical_index(seed, linear));
                    tensor_set_real(&mut ga, source_linear, value);
                }
                if let Some(a_deriv) = &node.ad_a_deriv { return self.ad_backward(a_deriv, &ga); }
                LanaError::Ok
            }
            19 => {
                let matrix = node.ad_a.as_ref().unwrap();
                let rhs = node.ad_b.as_ref().unwrap();
                let (db, x) = {
                    let alloc = self.heap.clone();
                    let db = match tensor::tensor_cholesky_solve(&alloc, matrix, seed) { Ok(v) => v, Err(error) => return error };
                    let x = match tensor::tensor_cholesky_solve(&alloc, matrix, rhs) { Ok(v) => v, Err(error) => return error };
                    let (ValueKind::Tensor(db), ValueKind::Tensor(x)) = (db.kind, x.kind) else { unreachable!() };
                    (db, x)
                };
                let n = matrix.shape[0];
                let columns = if rhs.ndim == 1 { 1 } else { rhs.shape[1] };
                let mut ga = {
                    let alloc = self.heap.clone();
                    match tensor::tensor_new(&alloc, 2, &matrix.shape, false) { Ok(t) => t, Err(error) => return error }
                };
                for i in 0..n { for j in 0..n {
                    let mut value = 0.0;
                    for c in 0..columns {
                        let dbi = tensor_get_real(&db, tensor::logical_index(&db, i * columns + c));
                        let dbj = tensor_get_real(&db, tensor::logical_index(&db, j * columns + c));
                        let xi = tensor_get_real(&x, tensor::logical_index(&x, i * columns + c));
                        let xj = tensor_get_real(&x, tensor::logical_index(&x, j * columns + c));
                        value -= 0.5 * (dbi * xj + xi * dbj);
                    }
                    tensor_set_real(&mut ga, i * n + j, value);
                } }
                if let Some(a_deriv) = &node.ad_a_deriv { let error = self.ad_backward(a_deriv, &ga); if error != LanaError::Ok { return error; } }
                if let Some(b_deriv) = &node.ad_b_deriv { return self.ad_backward(b_deriv, &db); }
                LanaError::Ok
            }
            7 => {
                // append (LIP-007): mean-state distribution-valued APPEND.
                let a = node.ad_a.as_ref().unwrap();
                let b = node.ad_b.as_ref().unwrap();
                let d = a.shape[a.ndim - 1];
                let mut batch = 1;
                for i in 0..a.ndim.saturating_sub(2) {
                    batch *= a.shape[i];
                }
                let alloc = self.heap.clone();
                let mut ga = match tensor::tensor_new(&alloc, a.ndim, &a.shape, true) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                let mut gb = match tensor::tensor_new(&alloc, b.ndim, &b.shape, true) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                {
                    let ga_data = state_f64_mut(&mut ga);
                    let gb_data = state_f64_mut(&mut gb);
                    for bi in 0..batch {
                        let da = &state_f64(a)[bi * d * d * 2..];
                        let db = &state_f64(b)[bi * d * d * 2..];
                        let dg = &state_f64(seed)[(seed.offset + bi * d * d) * 2..];
                        let p_a = da[0];
                        let c_a_re = da[2];
                        let c_a_im = da[3];
                        let p_b = db[0];
                        let c_b_re = db[2];
                        let c_b_im = db[3];
                        let s_a = (p_a * (1.0 - p_a)).sqrt();
                        let s_b = (p_b * (1.0 - p_b)).sqrt();
                        let d_a_re = if s_a > 0.0 { c_a_re / s_a } else { 0.0 };
                        let d_a_im = if s_a > 0.0 { c_a_im / s_a } else { 0.0 };
                        let d_b_re = if s_b > 0.0 { c_b_re / s_b } else { 0.0 };
                        let d_b_im = if s_b > 0.0 { c_b_im / s_b } else { 0.0 };
                        let p_c = p_a + p_b - p_a * p_b;
                        let d_c_re = (d_a_re + d_b_re) / 2.0;
                        let d_c_im = (d_a_im + d_b_im) / 2.0;
                        let s_c = (p_c * (1.0 - p_c)).sqrt();
                        // Reduce the seed to (g_p, g_c): cotangents of p_C and c_C.
                        let g_p = dg[0] - dg[6];
                        let g_c_re = dg[2] + dg[4];
                        let g_c_im = dg[3] - dg[5];
                        let g_dc_re = s_c * g_c_re;
                        let g_dc_im = s_c * g_c_im;
                        let g_sc = g_c_re * d_c_re + g_c_im * d_c_im;
                        let mut g_pc = g_p;
                        if s_c > 0.0 {
                            g_pc += g_sc * (1.0 - 2.0 * p_c) / (2.0 * s_c);
                        }
                        let g_da_re = g_dc_re / 2.0;
                        let g_da_im = g_dc_im / 2.0;
                        let g_db_re = g_dc_re / 2.0;
                        let g_db_im = g_dc_im / 2.0;
                        let mut g_pa = g_pc * (1.0 - p_b);
                        let mut g_pb = g_pc * (1.0 - p_a);
                        let mut g_ca_re = 0.0;
                        let mut g_ca_im = 0.0;
                        let mut g_cb_re = 0.0;
                        let mut g_cb_im = 0.0;
                        if s_a > 0.0 {
                            g_ca_re = g_da_re / s_a;
                            g_ca_im = g_da_im / s_a;
                            let g_sa = -(g_da_re * c_a_re + g_da_im * c_a_im) / (s_a * s_a);
                            g_pa += g_sa * (1.0 - 2.0 * p_a) / (2.0 * s_a);
                        }
                        if s_b > 0.0 {
                            g_cb_re = g_db_re / s_b;
                            g_cb_im = g_db_im / s_b;
                            let g_sb = -(g_db_re * c_b_re + g_db_im * c_b_im) / (s_b * s_b);
                            g_pb += g_sb * (1.0 - 2.0 * p_b) / (2.0 * s_b);
                        }
                        let ga_off = bi * d * d * 2;
                        ga_data[ga_off] = g_pa;
                        ga_data[ga_off + 1] = 0.0;
                        ga_data[ga_off + 2] = g_ca_re;
                        ga_data[ga_off + 3] = g_ca_im;
                        ga_data[ga_off + 4] = g_ca_re;
                        ga_data[ga_off + 5] = if g_ca_im == 0.0 { 0.0 } else { -g_ca_im };
                        ga_data[ga_off + 6] = -g_pa;
                        ga_data[ga_off + 7] = 0.0;
                        let gb_off = bi * d * d * 2;
                        gb_data[gb_off] = g_pb;
                        gb_data[gb_off + 1] = 0.0;
                        gb_data[gb_off + 2] = g_cb_re;
                        gb_data[gb_off + 3] = g_cb_im;
                        gb_data[gb_off + 4] = g_cb_re;
                        gb_data[gb_off + 5] = if g_cb_im == 0.0 { 0.0 } else { -g_cb_im };
                        gb_data[gb_off + 6] = -g_pb;
                        gb_data[gb_off + 7] = 0.0;
                    }
                }
                if let Some(a_deriv) = &node.ad_a_deriv {
                    let error = self.ad_backward(a_deriv, &ga);
                    if error != LanaError::Ok {
                        return error;
                    }
                }
                if let Some(b_deriv) = &node.ad_b_deriv {
                    let error = self.ad_backward(b_deriv, &gb);
                    if error != LanaError::Ok {
                        return error;
                    }
                }
                LanaError::Ok
            }
            8 => {
                // measure (LIP-007): q[..., i] = Tr(ρ E_i).
                let s = node.ad_a.as_ref().unwrap();
                let povm = node.ad_b.as_ref().unwrap();
                let d = s.shape[s.ndim - 1];
                let k = povm.shape[0];
                let mut batch = 1;
                for i in 0..s.ndim.saturating_sub(2) {
                    batch *= s.shape[i];
                }
                let alloc = self.heap.clone();
                let mut gs = match tensor::tensor_new(&alloc, s.ndim, &s.shape, true) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                {
                    let gs_data = state_f64_mut(&mut gs);
                    for bi in 0..batch {
                        for r in 0..d {
                            for c in 0..d {
                                let mut re = 0.0;
                                let mut im = 0.0;
                                for m in 0..k {
                                    let seed_val = tensor_get_real(seed, seed.offset + bi * k + m);
                                    let (e_re, e_im) = linalg_get3(povm, m, c, r);
                                    re += seed_val * e_re;
                                    im += seed_val * (-e_im);
                                }
                                gs_data[(bi * d * d + r * d + c) * 2] = re;
                                gs_data[(bi * d * d + r * d + c) * 2 + 1] = im;
                            }
                        }
                    }
                }
                if let Some(a_deriv) = &node.ad_a_deriv {
                    let error = self.ad_backward(a_deriv, &gs);
                    if error != LanaError::Ok {
                        return error;
                    }
                }
                LanaError::Ok
            }
            9 => {
                // transform (LIP-007): Φ(ρ) = Σ_k K_k ρ K_k†.
                let s = node.ad_a.as_ref().unwrap();
                let chan = node.ad_b.as_ref().unwrap();
                let d = s.shape[s.ndim - 1];
                let k = chan.shape[0];
                let mut batch = 1;
                for i in 0..s.ndim.saturating_sub(2) {
                    batch *= s.shape[i];
                }
                let alloc = self.heap.clone();
                let mut gs = match tensor::tensor_new(&alloc, s.ndim, &s.shape, true) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                {
                    let gs_data = state_f64_mut(&mut gs);
                    for bi in 0..batch {
                        let dg = &state_f64(seed)[(seed.offset + bi * d * d) * 2..];
                        for m in 0..d {
                            for n in 0..d {
                                let mut re = 0.0;
                                let mut im = 0.0;
                                for kk in 0..k {
                                    for i in 0..d {
                                        for j in 0..d {
                                            let (ki_re, ki_im) = linalg_get3(chan, kk, i, m);
                                            let g_re = dg[(i * d + j) * 2];
                                            let g_im = dg[(i * d + j) * 2 + 1];
                                            let (kj_re, kj_im) = linalg_get3(chan, kk, j, n);
                                            // conj(K[i][m]) * G[i][j] * K[j][n]
                                            let t_re = ki_re * g_re + ki_im * g_im;
                                            let t_im = ki_re * g_im - ki_im * g_re;
                                            re += t_re * kj_re - t_im * kj_im;
                                            im += t_re * kj_im + t_im * kj_re;
                                        }
                                    }
                                }
                                gs_data[(bi * d * d + m * d + n) * 2] = re;
                                gs_data[(bi * d * d + m * d + n) * 2 + 1] = im;
                            }
                        }
                    }
                }
                if let Some(a_deriv) = &node.ad_a_deriv {
                    let error = self.ad_backward(a_deriv, &gs);
                    if error != LanaError::Ok {
                        return error;
                    }
                }
                LanaError::Ok
            }
            _ => LanaError::Type,
        }
    }

    /// Validate `f`/`x`, create the input leaf, and run `f(x)` with recording
    /// on, mirroring `ad_run` in `vm/c/vm.c`. On success `leaf_out` is the input
    /// leaf and `result_out` is `f`'s output.
    fn ad_run(
        &mut self,
        function_value: &Value,
        x: &Value,
        scratch_register: u32,
        leaf_out: &mut Option<Arc<Derivation>>,
        result_out: &mut Value,
    ) -> LanaError {
        let ValueKind::Function(function_index) = function_value.kind else {
            return LanaError::Type;
        };
        let ValueKind::Tensor(x_tensor) = &x.kind else {
            return LanaError::Type;
        };
        if x_tensor.is_complex && !x_tensor.is_state {
            return LanaError::Type;
        }
        if function_index as usize >= self.chunk.functions.len() {
            return LanaError::Type;
        }
        if self.chunk.functions[function_index as usize].arity != 1 {
            return LanaError::Type;
        }
        self.derivation_sequence += 1;
        let function_name = self.current_function_name();
        let leaf = match self.managed_payload(Derivation {
            task_lineage: self.lineage,
            local_sequence: self.derivation_sequence,
            revision: self.revision,
            kind: DerivationKind::Operation,
            operation: shared_derivation_text("input"),
            inputs: Vec::new(),
            label: shared_derivation_text(""),
            function: match self.heap.string(&function_name) { Ok(text) => text, Err(error) => return error },
            line: 0,
            exactness: DerivationExactness::Exact,
            details: shared_derivation_text("autodiff"),
            outcome: DerivationOutcome::Success,
            reason: shared_derivation_text("none"),
            ad_op: -1,
            ad_a: Some(x_tensor.clone()),
            ad_b: None,
            ad_a_deriv: None,
            ad_b_deriv: None,
            ad_grad: Arc::new(Mutex::new(None)),
            ad_axis: -1,
        }) { Ok(node) => node, Err(error) => return error };
        let mut x_with_deriv = x.clone();
        x_with_deriv.derivation = Some(leaf.clone());
        self.ad_recording = true;
        let error = self.run_function(function_index, &x_with_deriv, scratch_register, result_out);
        self.ad_recording = false;
        if error != LanaError::Ok {
            return error;
        }
        *leaf_out = Some(leaf);
        LanaError::Ok
    }

    /// Read the leaf's accumulated gradient, check finiteness, and return a
    /// fresh tensor carrying a provenance derivation that traces to the input
    /// leaf, mirroring `ad_finish` in `vm/c/vm.c`.
    fn ad_finish(&mut self, leaf: &Arc<Derivation>, operation: &str, out: &mut Value) -> LanaError {
        let grad = {
            let guard = leaf.ad_grad.lock().unwrap();
            if let Some(t) = guard.as_ref() {
                match t.try_clone(&self.heap) { Ok(t) => t, Err(error) => return error }
            } else {
                let a = leaf.ad_a.as_ref().unwrap();
                let alloc = self.heap.clone();
                match tensor::tensor_new(&alloc, a.ndim, &a.shape, a.is_complex) {
                    Ok(t) => t,
                    Err(error) => return error,
                }
            }
        };
        let count = tensor::tensor_element_count(&grad);
        let components = if grad.is_complex { 2 } else { 1 };
        for e in 0..count {
            if !tensor_get_real(&grad, e).is_finite() || !tensor_get_imag(&grad, e).is_finite() {
                return LanaError::InvalidParameters;
            }
        }
        let mut result_tensor = {
            let alloc = self.heap.clone();
            match tensor::tensor_new(&alloc, grad.ndim, &grad.shape, grad.is_complex) {
                Ok(t) => t,
                Err(error) => return error,
            }
        };
        {
            let data = Arc::get_mut(&mut result_tensor.data).unwrap();
            data.copy_from_slice(&grad.data[..count * components * 8]);
        }
        let mut leaf_value = Value::tensor(leaf.ad_a.as_ref().unwrap().clone());
        leaf_value.derivation = Some(leaf.clone());
        self.derivation_sequence += 1;
        let function_name = self.current_function_name();
        let grad_deriv = match self.managed_payload(Derivation {
            task_lineage: self.lineage,
            local_sequence: self.derivation_sequence,
            revision: self.revision,
            kind: DerivationKind::Operation,
            operation: match self.heap.string(operation) { Ok(text) => text, Err(error) => return error },
            inputs: vec![leaf.clone()],
            label: shared_derivation_text(""),
            function: match self.heap.string(&function_name) { Ok(text) => text, Err(error) => return error },
            line: 0,
            exactness: DerivationExactness::Exact,
            details: shared_derivation_text("autodiff"),
            outcome: DerivationOutcome::Success,
            reason: shared_derivation_text("none"),
            ad_op: -1,
            ad_a: None,
            ad_b: None,
            ad_a_deriv: None,
            ad_b_deriv: None,
            ad_grad: Arc::new(Mutex::new(None)),
            ad_axis: -1,
        }) { Ok(node) => node, Err(error) => return error };
        let mut grad_value = Value::tensor(Arc::new(result_tensor));
        grad_value.derivation = Some(grad_deriv);
        *out = grad_value;
        LanaError::Ok
    }

    /// `grad(f, x)`: gradient of a scalar-output pure function at `x`, mirroring
    /// `ad_grad` in `vm/c/vm.c`.
    fn ad_grad(
        &mut self,
        function_value: &Value,
        x: &Value,
        scratch_register: u32,
        out: &mut Value,
    ) -> LanaError {
        let mut leaf: Option<Arc<Derivation>> = None;
        let mut result = Value::null();
        let error = self.ad_run(function_value, x, scratch_register, &mut leaf, &mut result);
        if error != LanaError::Ok {
            return error;
        }
        let leaf = leaf.unwrap();
        if !matches!(result.kind, ValueKind::Number(_)) {
            return LanaError::Type;
        }
        let mut seed = {
            let alloc = self.heap.clone();
            match tensor::tensor_new(&alloc, 0, &[], false) {
                Ok(t) => t,
                Err(error) => return error,
            }
        };
        tensor_set_real(&mut seed, 0, 1.0);
        if let Some(derivation) = &result.derivation {
            let error = self.ad_backward(derivation, &seed);
            if error != LanaError::Ok {
                return error;
            }
        }
        self.ad_finish(&leaf, "grad", out)
    }

    /// `vjp(f, x, v)`: vector-Jacobian product `v^T . J_f(x)`, mirroring
    /// `ad_vjp` in `vm/c/vm.c`.
    fn ad_vjp(
        &mut self,
        function_value: &Value,
        x: &Value,
        v: &Value,
        scratch_register: u32,
        out: &mut Value,
    ) -> LanaError {
        let ValueKind::Tensor(v_tensor) = &v.kind else {
            return LanaError::Type;
        };
        if v_tensor.is_complex {
            return LanaError::Type;
        }
        let mut leaf: Option<Arc<Derivation>> = None;
        let mut result = Value::null();
        let error = self.ad_run(function_value, x, scratch_register, &mut leaf, &mut result);
        if error != LanaError::Ok {
            return error;
        }
        let leaf = leaf.unwrap();
        let ValueKind::Tensor(result_tensor) = &result.kind else {
            return LanaError::Type;
        };
        if !tensor::tensor_shape_equal(v_tensor, result_tensor) {
            return LanaError::Type;
        }
        let mut seed = {
            let alloc = self.heap.clone();
            match tensor::tensor_new(&alloc, v_tensor.ndim, &v_tensor.shape, false) {
                Ok(t) => t,
                Err(error) => return error,
            }
        };
        let count = tensor::tensor_element_count(v_tensor);
        {
            for lin in 0..count {
                let mut rem = lin;
                let mut index = v_tensor.offset;
                for d in (0..v_tensor.ndim).rev() {
                    index += (if v_tensor.shape[d] == 0 { 0 } else { rem % v_tensor.shape[d] })
                        * v_tensor.strides[d];
                    rem /= v_tensor.shape[d];
                }
                tensor_set_real(&mut seed, lin, tensor_get_real(v_tensor, index));
            }
        }
        if let Some(derivation) = &result.derivation {
            let error = self.ad_backward(derivation, &seed);
            if error != LanaError::Ok {
                return error;
            }
        }
        self.ad_finish(&leaf, "vjp", out)
    }

    /// Copy a tensor (possibly a view) into a fresh contiguous base tensor,
    /// mirroring `tensor_copy_contiguous` in `vm/c/vm.c`.
    fn tensor_copy_contiguous(&mut self, t: &Tensor) -> Result<Arc<Tensor>, LanaError> {
        let alloc = self.heap.clone();
        let mut copy = tensor::tensor_new_dtype(&alloc, t.ndim, &t.shape, t.dtype)?;
        copy.is_state = t.is_state;
        copy.device = t.device;
        let count = tensor::tensor_element_count(t);
        for lin in 0..count {
            let mut rem = lin;
            let mut index = t.offset;
            for d in (0..t.ndim).rev() {
                index += (if t.shape[d] == 0 { 0 } else { rem % t.shape[d] }) * t.strides[d];
                rem /= t.shape[d];
            }
            if t.is_complex {
                tensor_set_real(&mut copy, lin, tensor_get_real(t, index));
                tensor_set_imag(&mut copy, lin, tensor_get_imag(t, index));
            } else {
                tensor_set_real(&mut copy, lin, tensor_get_real(t, index));
            }
        }
        Ok(Arc::new(copy))
    }

    /// Whether the VM holds a non-revoked READ capability over a shared
    /// information whose base snapshot is the string `name`, mirroring
    /// `vm_has_named_capability` in `vm/c/vm.c`.
    fn has_named_capability(&self, name: &str) -> bool {
        for shared in &self.shared_references {
            if let ValueKind::String(s) = &shared.base_snapshot.kind {
                if &**s == name {
                    let state = shared.state.lock().unwrap();
                    for capability in &state.capabilities {
                        if capability_allows_locked(shared, capability, LANA_CAPABILITY_READ) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// `sgd(learning_rate, momentum)`, mirroring `host_sgd` in `vm/c/vm.c`.
    fn host_sgd(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        let mut learning_rate = 0.01;
        let mut momentum = 0.9;
        if arguments.len() > 2 {
            return LanaError::Type;
        }
        if arguments.len() >= 1 {
            let ValueKind::Number(n) = arguments[0].kind else {
                return LanaError::Type;
            };
            learning_rate = n;
        }
        if arguments.len() >= 2 {
            let ValueKind::Number(n) = arguments[1].kind else {
                return LanaError::Type;
            };
            momentum = n;
        }
        if !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !momentum.is_finite()
            || momentum < 0.0
            || momentum >= 1.0
        {
            return LanaError::InvalidParameters;
        }
        if self.alloc_bytes(std::mem::size_of::<Optimizer>()) != LanaError::Ok {
            return LanaError::Oom;
        }
        *out = Value::optimizer(Arc::new(Optimizer {
            name: Arc::from("sgd"),
            learning_rate,
            momentum,
            beta1: 0.0,
            beta2: 0.0,
            epsilon: 0.0,
        }));
        LanaError::Ok
    }

    /// `adam(learning_rate, beta1, beta2, epsilon)`, mirroring `host_adam` in
    /// `vm/c/vm.c`.
    fn host_adam(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        let mut learning_rate = 0.001;
        let mut beta1 = 0.9;
        let mut beta2 = 0.999;
        let mut epsilon = 1e-8;
        if arguments.len() > 4 {
            return LanaError::Type;
        }
        if arguments.len() >= 1 {
            let ValueKind::Number(n) = arguments[0].kind else {
                return LanaError::Type;
            };
            learning_rate = n;
        }
        if arguments.len() >= 2 {
            let ValueKind::Number(n) = arguments[1].kind else {
                return LanaError::Type;
            };
            beta1 = n;
        }
        if arguments.len() >= 3 {
            let ValueKind::Number(n) = arguments[2].kind else {
                return LanaError::Type;
            };
            beta2 = n;
        }
        if arguments.len() >= 4 {
            let ValueKind::Number(n) = arguments[3].kind else {
                return LanaError::Type;
            };
            epsilon = n;
        }
        if !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !beta1.is_finite()
            || beta1 < 0.0
            || beta1 >= 1.0
            || !beta2.is_finite()
            || beta2 < 0.0
            || beta2 >= 1.0
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return LanaError::InvalidParameters;
        }
        if self.alloc_bytes(std::mem::size_of::<Optimizer>()) != LanaError::Ok {
            return LanaError::Oom;
        }
        *out = Value::optimizer(Arc::new(Optimizer {
            name: Arc::from("adam"),
            learning_rate,
            momentum: 0.0,
            beta1,
            beta2,
            epsilon,
        }));
        LanaError::Ok
    }

    /// `train(model, data, loss, optimizer, initial_params, [epochs],
    /// [batch_size])`, mirroring `host_train` in `vm/c/vm.c`.
    fn host_train(
        &mut self,
        arguments: &[Value],
        scratch_register: u32,
        out: &mut Value,
    ) -> Result<(), LanaError> {
        if arguments.len() < 5 || arguments.len() > 7 {
            return Err(LanaError::Type);
        }
        let model = &arguments[0];
        let data = &arguments[1];
        let loss = &arguments[2];
        let optimizer_value = &arguments[3];
        let initial_params = &arguments[4];
        // LIP-010: a live data root produces reactive parameters. Resolve the
        // current dataset and remember whether the data was reactive so the
        // result can be wired into the reactive DAG.
        let data_is_reactive = data.reactive.is_some();
        let data_current = self.reactive_value(data);

        let ValueKind::Function(model_fn) = model.kind else {
            return Err(LanaError::Type);
        };
        let ValueKind::Function(loss_fn) = loss.kind else {
            return Err(LanaError::Type);
        };
        if model_fn as usize >= self.chunk.functions.len()
            || loss_fn as usize >= self.chunk.functions.len()
        {
            return Err(LanaError::Type);
        }
        if self.chunk.functions[model_fn as usize].arity != 2
            || self.chunk.functions[loss_fn as usize].arity != 2
        {
            return Err(LanaError::Type);
        }

        let ValueKind::Optimizer(optimizer) = &optimizer_value.kind else {
            return Err(LanaError::Type);
        };

        let ValueKind::Tensor(initial_params_tensor) = &initial_params.kind else {
            return Err(LanaError::Type);
        };
        if initial_params_tensor.is_complex && !initial_params_tensor.is_state {
            return Err(LanaError::Type);
        }

        let mut epochs = 10.0;
        let mut batch_size = 0.0;
        if arguments.len() >= 6 {
            let ValueKind::Number(n) = arguments[5].kind else {
                return Err(LanaError::Type);
            };
            epochs = n;
        }
        if arguments.len() >= 7 {
            let ValueKind::Number(n) = arguments[6].kind else {
                return Err(LanaError::Type);
            };
            batch_size = n;
        }
        if !epochs.is_finite()
            || epochs < 1.0
            || epochs.floor() != epochs
            || epochs > usize::MAX as f64
        {
            return Err(LanaError::InvalidParameters);
        }
        if !batch_size.is_finite()
            || batch_size < 0.0
            || batch_size.floor() != batch_size
            || batch_size > usize::MAX as f64
        {
            return Err(LanaError::InvalidParameters);
        }

        if !self.has_named_capability("train") {
            return Err(LanaError::Capability);
        }

        let (lazy_fn, dataset_size) = match &data_current.kind {
            ValueKind::Array(array) => (None, array.lock().unwrap().items.len()),
            ValueKind::Lazy { function, bound } => {
                if *function as usize >= self.chunk.functions.len()
                    || self.chunk.functions[*function as usize].arity != 1
                {
                    return Err(LanaError::Type);
                }
                (Some(*function), *bound)
            }
            _ => return Err(LanaError::Type),
        };
        if dataset_size == 0 {
            return Err(LanaError::InvalidParameters);
        }

        let mut batch = batch_size as usize;
        if batch == 0 || batch > dataset_size {
            batch = dataset_size;
        }
        let epoch_count = epochs as usize;

        let mut params = self.tensor_copy_contiguous(initial_params_tensor)?;
        let param_count = tensor::tensor_element_count(&params);
        let param_components = if params.is_complex { 2 } else { 1 };

        let is_adam = &*optimizer.name == "adam";
        let mut adam_t = 0usize;
        let mut m: Option<Arc<Tensor>> = None;
        let mut v: Option<Arc<Tensor>> = None;
        let mut velocity: Option<Arc<Tensor>> = None;
        if is_adam {
            let alloc = self.heap.clone();
            let m_t = tensor::tensor_new(&alloc, params.ndim, &params.shape, params.is_complex)?;
            let v_t = tensor::tensor_new(&alloc, params.ndim, &params.shape, params.is_complex)?;
            m = Some(Arc::new(m_t));
            v = Some(Arc::new(v_t));
        } else if optimizer.momentum != 0.0 {
            let alloc = self.heap.clone();
            let vel = tensor::tensor_new(&alloc, params.ndim, &params.shape, params.is_complex)?;
            velocity = Some(Arc::new(vel));
        }

        let mut batch_grad = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, params.ndim, &params.shape, params.is_complex)?)
        };

        let total_steps = epoch_count * ((dataset_size + batch - 1) / batch);
        let mut steps = Array::new(&self.heap, total_steps)?;

        let mut params_deriv: Option<Arc<Derivation>> = None;

        for epoch in 0..epoch_count {
            let mut batch_start = 0usize;
            while batch_start < dataset_size {
                let mut batch_end = batch_start + batch;
                if batch_end > dataset_size {
                    batch_end = dataset_size;
                }
                let batch_actual = batch_end - batch_start;

                {
                    let bg = Arc::get_mut(&mut batch_grad).unwrap();
                    for k in 0..param_count * param_components {
                        tensor_set_real(bg, k, 0.0);
                    }
                }

                let mut i = batch_start;
                while i < batch_end {
                    let pair = if let Some(lazy_fn) = lazy_fn {
                        let index_value = Value::number(i as f64);
                        let mut pair = Value::null();
                        let error = self.run_function(lazy_fn, &index_value, scratch_register, &mut pair);
                        if error != LanaError::Ok {
                            return Err(error);
                        }
                        pair
                    } else {
                        let ValueKind::Array(array) = &data_current.kind else {
                            unreachable!("array dataset");
                        };
                        array.lock().unwrap().items[i].clone()
                    };
                    let ValueKind::Array(pair_array) = &pair.kind else {
                        return Err(LanaError::Type);
                    };
                    let pair_items = pair_array.lock().unwrap();
                    if pair_items.items.len() != 2 {
                        return Err(LanaError::Type);
                    }
                    let x = pair_items.items[0].clone();
                    let target = pair_items.items[1].clone();
                    drop(pair_items);

                    let leaf = self.record_derivation_payload(
                        DerivationKind::Operation,
                        "input",
                        &[],
                        "",
                        0,
                        DerivationExactness::Exact,
                        "autodiff",
                        DerivationOutcome::Success,
                        "none",
                    );
                    let Some(mut leaf) = leaf else {
                        return Err(LanaError::Oom);
                    };
                    leaf.ad_a = Some(params.clone());
                let leaf = self.managed_payload(leaf)?;

                    let mut params_with_deriv = Value::tensor(params.clone());
                    params_with_deriv.derivation = Some(leaf.clone());

                    self.ad_recording = true;
                    let mut y = Value::null();
                    let mut error = self.run_function2(
                        model_fn,
                        &params_with_deriv,
                        &x,
                        scratch_register,
                        &mut y,
                    );
                    if error == LanaError::Ok {
                        let y_arg = y.clone();
                        error = self.run_function2(loss_fn, &y_arg, &target, scratch_register, &mut y);
                    }
                    self.ad_recording = false;
                    if error != LanaError::Ok {
                        return Err(error);
                    }

                    let loss_value = match &y.kind {
                        ValueKind::Number(n) => *n,
                        _ => return Err(LanaError::Type),
                    };
                    if !loss_value.is_finite() {
                        return Err(LanaError::InvalidParameters);
                    }

                    let mut seed = {
                        let alloc = self.heap.clone();
                        tensor::tensor_new(&alloc, 0, &[], false)?
                    };
                    tensor_set_real(&mut seed, 0, 1.0 / batch_actual as f64);
                    if let Some(derivation) = &y.derivation {
                        let error = self.ad_backward(derivation, &seed);
                        if error != LanaError::Ok {
                            return Err(error);
                        }
                    }

                    let grad = {
                        let guard = leaf.ad_grad.lock().unwrap();
                        if let Some(t) = guard.as_ref() {
                            t.try_clone(&self.heap)?
                        } else {
                            let alloc = self.heap.clone();
                            tensor::tensor_new(&alloc, params.ndim, &params.shape, params.is_complex)?
                        }
                    };
                    {
                        let bg = Arc::get_mut(&mut batch_grad).unwrap();
                        for k in 0..param_count * param_components {
                            if !tensor_get_real(&grad, k).is_finite() {
                                return Err(LanaError::InvalidParameters);
                            }
                            let cur = tensor_get_real(bg, k);
                            tensor_set_real(bg, k, cur + tensor_get_real(&grad, k));
                        }
                    }

                    i += 1;
                }

                let mut new_params = {
                    let alloc = self.heap.clone();
                    Arc::new(tensor::tensor_new(&alloc, params.ndim, &params.shape, params.is_complex)?)
                };
                Arc::get_mut(&mut new_params).unwrap().is_state = params.is_state;
                if is_adam {
                    adam_t += 1;
                    let bc1 = 1.0 - optimizer.beta1.powf(adam_t as f64);
                    let bc2 = 1.0 - optimizer.beta2.powf(adam_t as f64);
                    let m_tensor = Arc::get_mut(m.as_mut().unwrap()).unwrap();
                    let v_tensor = Arc::get_mut(v.as_mut().unwrap()).unwrap();
                    let bg_tensor = Arc::get_mut(&mut batch_grad).unwrap();
                    let np_tensor = Arc::get_mut(&mut new_params).unwrap();
                    for k in 0..param_count * param_components {
                        let g = tensor_get_real(bg_tensor, k);
                        let m_cur = tensor_get_real(m_tensor, k);
                        tensor_set_real(m_tensor, k, optimizer.beta1 * m_cur + (1.0 - optimizer.beta1) * g);
                        let v_cur = tensor_get_real(v_tensor, k);
                        tensor_set_real(v_tensor, k, optimizer.beta2 * v_cur + (1.0 - optimizer.beta2) * g * g);
                        let m_hat = tensor_get_real(m_tensor, k) / bc1;
                        let v_hat = tensor_get_real(v_tensor, k) / bc2;
                        tensor_set_real(np_tensor, k, tensor_get_real(&params, k)
                            - optimizer.learning_rate * m_hat / (v_hat.sqrt() + optimizer.epsilon));
                    }
                } else if let Some(vel) = velocity.as_mut() {
                    let vel_tensor = Arc::get_mut(vel).unwrap();
                    let bg_tensor = Arc::get_mut(&mut batch_grad).unwrap();
                    let np_tensor = Arc::get_mut(&mut new_params).unwrap();
                    for k in 0..param_count * param_components {
                        let vel_cur = tensor_get_real(vel_tensor, k);
                        let bg_cur = tensor_get_real(bg_tensor, k);
                        tensor_set_real(vel_tensor, k, optimizer.momentum * vel_cur
                            - optimizer.learning_rate * bg_cur);
                        let vel_new = tensor_get_real(vel_tensor, k);
                        tensor_set_real(np_tensor, k, tensor_get_real(&params, k) + vel_new);
                    }
                } else {
                    let bg_tensor = Arc::get_mut(&mut batch_grad).unwrap();
                    let np_tensor = Arc::get_mut(&mut new_params).unwrap();
                    for k in 0..param_count * param_components {
                        let bg_cur = tensor_get_real(bg_tensor, k);
                        tensor_set_real(np_tensor, k, tensor_get_real(&params, k) - optimizer.learning_rate * bg_cur);
                    }
                }

                let mut pre_value = Value::tensor(params.clone());
                pre_value.derivation = params_deriv.clone();
                let grad_snap = self.tensor_copy_contiguous(&batch_grad)?;
                let grad_value = Value::tensor(grad_snap);
                let inputs = [&pre_value, &grad_value];
                let details = format!("epoch={} batch={}", epoch, batch_start / batch);
                let step_deriv = self.record_derivation(
                    DerivationKind::Operation,
                    "train_step",
                    &inputs,
                    "",
                    0,
                    DerivationExactness::Exact,
                    &details,
                    DerivationOutcome::Success,
                    "none",
                );
                let Some(step_deriv) = step_deriv else {
                    return Err(LanaError::Oom);
                };
                let mut new_params_value = Value::tensor(new_params.clone());
                new_params_value.derivation = Some(step_deriv.clone());

                let mut step_map = Map::new(&self.heap, 5)?;
                step_map.set(Arc::from("epoch"), Value::number(epoch as f64), true)?;
                step_map.set(
                    Arc::from("batch"),
                    Value::number((batch_start / batch) as f64),
                    true,
                )?;
                step_map.set(Arc::from("parameters"), new_params_value, true)?;
                step_map.set(Arc::from("gradient"), grad_value, true)?;

                let mut state_map = Map::new(&self.heap, 3)?;
                if is_adam {
                    let m_snap = self.tensor_copy_contiguous(m.as_ref().unwrap())?;
                    let v_snap = self.tensor_copy_contiguous(v.as_ref().unwrap())?;
                    state_map.set(Arc::from("m"), Value::tensor(m_snap), true)?;
                    state_map.set(Arc::from("v"), Value::tensor(v_snap), true)?;
                    state_map.set(Arc::from("t"), Value::number(adam_t as f64), true)?;
                } else if let Some(vel) = velocity.as_ref() {
                    let vel_snap = self.tensor_copy_contiguous(vel)?;
                    state_map.set(Arc::from("velocity"), Value::tensor(vel_snap), true)?;
                }
                step_map.set(
                    Arc::from("optimizer_state"),
                    Value::map(Arc::new(Mutex::new(state_map))),
                    true,
                )?;

                steps.items.push(Value::map(Arc::new(Mutex::new(step_map))))?;

                params = new_params;
                params_deriv = Some(step_deriv);
                batch_start += batch;
            }
        }

        // LIP-014: keep the resolved dataset and effective batch size so
        // `resume` can continue the run from any step. The dataset is
        // immutable, so a shallow clone (sharing the array/lazy payload) is
        // sufficient; strip the runtime metadata like the C11 VM does.
        let mut data_copy = data_current.clone();
        data_copy.reactive = None;
        data_copy.claim = None;
        data_copy.planned_effect = None;
        let result = self.managed_payload(TrainingResult {
            params,
            steps: Arc::new(Mutex::new(steps)),
            model_function: model_fn,
            loss_function: loss_fn,
            optimizer: optimizer.clone(),
            data: data_copy,
            batch_size: batch,
        })?;
        let mut result_value = Value::training_result(result);
        if data_is_reactive {
            // Wire the training result into the reactive DAG: a TRAIN node whose
            // input is the data root. On `observe` the node is recomputed with
            // one incremental optimizer step over the new observation.
            let data_reactive = data.reactive.clone().unwrap();
            let (dependency_id, exactness) = {
                let mut guard = data_reactive.lock().unwrap();
                guard.is_training_data = true;
                (guard.dependency_id, guard.exactness)
            };
            let id = self.next_reactive_id;
            self.next_reactive_id += 1;
            let current = self.clone_without_runtime_metadata(&result_value)?;
            let node = Arc::new(Mutex::new(Reactive {
                id,
                dependency_id,
                revision: self.revision,
                kind: ReactiveKind::Train,
                relationship: RelationshipKind::Exact,
                exactness,
                operation: 0,
                inputs: [Some(data_reactive), None],
                constants: [None, None],
                current: Some(current),
                history: Vec::new(),
                is_training_data: false,
            }));
            self.track_cycle(crate::heap::CycleWeak::Reactive(Arc::downgrade(&node)))?;
            result_value.reactive = Some(node);
        }
        *out = result_value;
        Ok(())
    }

    /// Run `step_count` incremental optimizer steps over `data_current` (an
    /// array of [x, target] pairs or a lazy dataset), extending the prior
    /// training result's step history. The prior result is unchanged; a new
    /// `VAL_TRAINING_RESULT` is returned. Shared by `update` (explicit) and the
    /// reactive `observe` path. Mirrors `incremental_train` in `vm/c/vm.c`.
    fn incremental_train(
        &mut self,
        prior: &TrainingResult,
        data_current: &Value,
        dataset_size: usize,
        step_count: usize,
        scratch_register: u32,
        out: &mut Value,
    ) -> Result<(), LanaError> {
        if prior.model_function as usize >= self.chunk.functions.len()
            || prior.loss_function as usize >= self.chunk.functions.len()
        {
            return Err(LanaError::Type);
        }
        if self.chunk.functions[prior.model_function as usize].arity != 2
            || self.chunk.functions[prior.loss_function as usize].arity != 2
        {
            return Err(LanaError::Type);
        }
        if dataset_size == 0 {
            return Err(LanaError::InvalidParameters);
        }

        let optimizer = &prior.optimizer;
        let is_adam = &*optimizer.name == "adam";
        let prior_count = prior.steps.lock().unwrap().items.len();

        // Recover the optimizer state and params provenance from the last step
        // map, so the incremental run resumes exactly where batch training left
        // off.
        let mut m: Option<Arc<Tensor>> = None;
        let mut v: Option<Arc<Tensor>> = None;
        let mut velocity: Option<Arc<Tensor>> = None;
        let mut adam_t = 0usize;
        let mut params_deriv: Option<Arc<Derivation>> = None;
        if prior_count > 0 {
            let last_step = prior.steps.lock().unwrap().items[prior_count - 1].clone();
            let ValueKind::Map(last_map) = &last_step.kind else {
                return Err(LanaError::Type);
            };
            let state_value = last_map
                .lock()
                .unwrap()
                .get("optimizer_state")
                .cloned()
                .ok_or(LanaError::Type)?;
            let ValueKind::Map(state_map) = &state_value.kind else {
                return Err(LanaError::Type);
            };
            let (m_value, v_value, t_value, vel_value) = {
                let guard = state_map.lock().unwrap();
                let m_value = if is_adam { guard.get("m").cloned() } else { None };
                let v_value = if is_adam { guard.get("v").cloned() } else { None };
                let t_value = if is_adam { guard.get("t").cloned() } else { None };
                let vel_value = if !is_adam && optimizer.momentum != 0.0 {
                    guard.get("velocity").cloned()
                } else {
                    None
                };
                (m_value, v_value, t_value, vel_value)
            };
            if is_adam {
                let m_value = m_value.ok_or(LanaError::Type)?;
                let v_value = v_value.ok_or(LanaError::Type)?;
                let t_value = t_value.ok_or(LanaError::Type)?;
                let ValueKind::Tensor(m_tensor) = &m_value.kind else {
                    return Err(LanaError::Type);
                };
                let ValueKind::Tensor(v_tensor) = &v_value.kind else {
                    return Err(LanaError::Type);
                };
                let ValueKind::Number(t) = t_value.kind else {
                    return Err(LanaError::Type);
                };
                m = Some(self.tensor_copy_contiguous(m_tensor)?);
                v = Some(self.tensor_copy_contiguous(v_tensor)?);
                adam_t = t as usize;
            } else if optimizer.momentum != 0.0 {
                let vel_value = vel_value.ok_or(LanaError::Type)?;
                let ValueKind::Tensor(vel_tensor) = &vel_value.kind else {
                    return Err(LanaError::Type);
                };
                velocity = Some(self.tensor_copy_contiguous(vel_tensor)?);
            }
            let params_value = last_map
                .lock()
                .unwrap()
                .get("parameters")
                .cloned()
                .ok_or(LanaError::Type)?;
            params_deriv = params_value.derivation.clone();
        } else if is_adam {
            let alloc = self.heap.clone();
            let m_t = tensor::tensor_new(&alloc, prior.params.ndim, &prior.params.shape, false)?;
            let v_t = tensor::tensor_new(&alloc, prior.params.ndim, &prior.params.shape, false)?;
            m = Some(Arc::new(m_t));
            v = Some(Arc::new(v_t));
        } else if optimizer.momentum != 0.0 {
            let alloc = self.heap.clone();
            let vel = tensor::tensor_new(&alloc, prior.params.ndim, &prior.params.shape, false)?;
            velocity = Some(Arc::new(vel));
        }

        let mut params = prior.params.clone();
        let param_count = tensor::tensor_element_count(&params);

        let mut batch_grad = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, params.ndim, &params.shape, false)?)
        };

        let mut steps = Array::new(&self.heap, prior_count + step_count)?;
        {
            let prior_items = prior.steps.lock().unwrap();
            for item in prior_items.items.iter() {
                steps.items.push(item.clone())?;
            }
        }

        for step in 0..step_count {
            {
                let bg = Arc::get_mut(&mut batch_grad).unwrap();
                for k in 0..param_count {
                    tensor_set_real(bg, k, 0.0);
                }
            }

            let mut i = 0usize;
            while i < dataset_size {
                let pair = match &data_current.kind {
                    ValueKind::Array(array) => array.lock().unwrap().items[i].clone(),
                    ValueKind::Lazy { function, .. } => {
                        let index_value = Value::number(i as f64);
                        let mut pair = Value::null();
                        let error = self.run_function(*function, &index_value, scratch_register, &mut pair);
                        if error != LanaError::Ok {
                            return Err(error);
                        }
                        pair
                    }
                    _ => return Err(LanaError::Type),
                };
                let ValueKind::Array(pair_array) = &pair.kind else {
                    return Err(LanaError::Type);
                };
                let pair_items = pair_array.lock().unwrap();
                if pair_items.items.len() != 2 {
                    return Err(LanaError::Type);
                }
                let x = pair_items.items[0].clone();
                let target = pair_items.items[1].clone();
                drop(pair_items);

                let leaf = self.record_derivation_payload(
                    DerivationKind::Operation,
                    "input",
                    &[],
                    "",
                    0,
                    DerivationExactness::Exact,
                    "autodiff",
                    DerivationOutcome::Success,
                    "none",
                );
                let Some(mut leaf) = leaf else {
                    return Err(LanaError::Oom);
                };
                leaf.ad_a = Some(params.clone());
                let leaf = self.managed_payload(leaf)?;

                let mut params_with_deriv = Value::tensor(params.clone());
                params_with_deriv.derivation = Some(leaf.clone());

                self.ad_recording = true;
                let mut y = Value::null();
                let mut error = self.run_function2(
                    prior.model_function,
                    &params_with_deriv,
                    &x,
                    scratch_register,
                    &mut y,
                );
                if error == LanaError::Ok {
                    let y_arg = y.clone();
                    error = self.run_function2(prior.loss_function, &y_arg, &target, scratch_register, &mut y);
                }
                self.ad_recording = false;
                if error != LanaError::Ok {
                    return Err(error);
                }

                let loss_value = match &y.kind {
                    ValueKind::Number(n) => *n,
                    _ => return Err(LanaError::Type),
                };
                if !loss_value.is_finite() {
                    return Err(LanaError::InvalidParameters);
                }

                let mut seed = {
                    let alloc = self.heap.clone();
                    tensor::tensor_new(&alloc, 0, &[], false)?
                };
                tensor_set_real(&mut seed, 0, 1.0 / dataset_size as f64);
                if let Some(derivation) = &y.derivation {
                    let error = self.ad_backward(derivation, &seed);
                    if error != LanaError::Ok {
                        return Err(error);
                    }
                }

                let grad = {
                    let guard = leaf.ad_grad.lock().unwrap();
                    if let Some(t) = guard.as_ref() {
                        t.try_clone(&self.heap)?
                    } else {
                        let alloc = self.heap.clone();
                        tensor::tensor_new(&alloc, params.ndim, &params.shape, false)?
                    }
                };
                {
                    let bg = Arc::get_mut(&mut batch_grad).unwrap();
                    for k in 0..param_count {
                        if !tensor_get_real(&grad, k).is_finite() {
                            return Err(LanaError::InvalidParameters);
                        }
                        let cur = tensor_get_real(bg, k);
                        tensor_set_real(bg, k, cur + tensor_get_real(&grad, k));
                    }
                }

                i += 1;
            }

            let mut new_params = {
                let alloc = self.heap.clone();
                Arc::new(tensor::tensor_new(&alloc, params.ndim, &params.shape, false)?)
            };
            if is_adam {
                adam_t += 1;
                let bc1 = 1.0 - optimizer.beta1.powf(adam_t as f64);
                let bc2 = 1.0 - optimizer.beta2.powf(adam_t as f64);
                let m_tensor = Arc::get_mut(m.as_mut().unwrap()).unwrap();
                let v_tensor = Arc::get_mut(v.as_mut().unwrap()).unwrap();
                let bg_tensor = Arc::get_mut(&mut batch_grad).unwrap();
                let np_tensor = Arc::get_mut(&mut new_params).unwrap();
                for k in 0..param_count {
                    let g = tensor_get_real(bg_tensor, k);
                    let m_cur = tensor_get_real(m_tensor, k);
                    tensor_set_real(m_tensor, k, optimizer.beta1 * m_cur + (1.0 - optimizer.beta1) * g);
                    let v_cur = tensor_get_real(v_tensor, k);
                    tensor_set_real(v_tensor, k, optimizer.beta2 * v_cur + (1.0 - optimizer.beta2) * g * g);
                    let m_hat = tensor_get_real(m_tensor, k) / bc1;
                    let v_hat = tensor_get_real(v_tensor, k) / bc2;
                    tensor_set_real(np_tensor, k, tensor_get_real(&params, k)
                        - optimizer.learning_rate * m_hat / (v_hat.sqrt() + optimizer.epsilon));
                }
            } else if let Some(vel) = velocity.as_mut() {
                let vel_tensor = Arc::get_mut(vel).unwrap();
                let bg_tensor = Arc::get_mut(&mut batch_grad).unwrap();
                let np_tensor = Arc::get_mut(&mut new_params).unwrap();
                for k in 0..param_count {
                    let vel_cur = tensor_get_real(vel_tensor, k);
                    let bg_cur = tensor_get_real(bg_tensor, k);
                    tensor_set_real(vel_tensor, k, optimizer.momentum * vel_cur
                        - optimizer.learning_rate * bg_cur);
                    let vel_new = tensor_get_real(vel_tensor, k);
                    tensor_set_real(np_tensor, k, tensor_get_real(&params, k) + vel_new);
                }
            } else {
                let bg_tensor = Arc::get_mut(&mut batch_grad).unwrap();
                let np_tensor = Arc::get_mut(&mut new_params).unwrap();
                for k in 0..param_count {
                    let bg_cur = tensor_get_real(bg_tensor, k);
                    tensor_set_real(np_tensor, k, tensor_get_real(&params, k) - optimizer.learning_rate * bg_cur);
                }
            }

            let mut pre_value = Value::tensor(params.clone());
            pre_value.derivation = params_deriv.clone();
            let grad_snap = self.tensor_copy_contiguous(&batch_grad)?;
            let grad_value = Value::tensor(grad_snap);
            let inputs = [&pre_value, &grad_value];
            let details = format!("epoch={} batch={}", prior_count + step, 0usize);
            let step_deriv = self.record_derivation(
                DerivationKind::Operation,
                "train_step",
                &inputs,
                "",
                0,
                DerivationExactness::Exact,
                &details,
                DerivationOutcome::Success,
                "none",
            );
            let Some(step_deriv) = step_deriv else {
                return Err(LanaError::Oom);
            };
            let mut new_params_value = Value::tensor(new_params.clone());
            new_params_value.derivation = Some(step_deriv.clone());

            let mut step_map = Map::new(&self.heap, 5)?;
            step_map.set(Arc::from("epoch"), Value::number((prior_count + step) as f64), true)?;
            step_map.set(Arc::from("batch"), Value::number(0.0), true)?;
            step_map.set(Arc::from("parameters"), new_params_value, true)?;
            step_map.set(Arc::from("gradient"), grad_value, true)?;

            let mut state_map = Map::new(&self.heap, 3)?;
            if is_adam {
                let m_snap = self.tensor_copy_contiguous(m.as_ref().unwrap())?;
                let v_snap = self.tensor_copy_contiguous(v.as_ref().unwrap())?;
                state_map.set(Arc::from("m"), Value::tensor(m_snap), true)?;
                state_map.set(Arc::from("v"), Value::tensor(v_snap), true)?;
                state_map.set(Arc::from("t"), Value::number(adam_t as f64), true)?;
            } else if let Some(vel) = velocity.as_ref() {
                let vel_snap = self.tensor_copy_contiguous(vel)?;
                state_map.set(Arc::from("velocity"), Value::tensor(vel_snap), true)?;
            }
            step_map.set(
                Arc::from("optimizer_state"),
                Value::map(Arc::new(Mutex::new(state_map))),
                true,
            )?;

            steps.items.push(Value::map(Arc::new(Mutex::new(step_map))))?;

            params = new_params;
            params_deriv = Some(step_deriv);
        }

        *out = Value::training_result(self.managed_payload(TrainingResult {
            params,
            steps: Arc::new(Mutex::new(steps)),
            model_function: prior.model_function,
            loss_function: prior.loss_function,
            optimizer: prior.optimizer.clone(),
            data: prior.data.clone(),
            batch_size: prior.batch_size,
        })?);
        Ok(())
    }

    /// `update(model, new_data, steps) -> VAL_TRAINING_RESULT`, mirroring
    /// `host_update` in `vm/c/vm.c`. The prior result is unchanged; the returned
    /// result extends its step history with `steps` incremental optimizer steps
    /// over `new_data`.
    fn host_update(
        &mut self,
        arguments: &[Value],
        scratch_register: u32,
        out: &mut Value,
    ) -> Result<(), LanaError> {
        if arguments.len() != 3 {
            return Err(LanaError::Type);
        }
        let model = &arguments[0];
        let new_data = &arguments[1];
        let steps_value = &arguments[2];

        let ValueKind::TrainingResult(prior) = &model.kind else {
            return Err(LanaError::Type);
        };

        let ValueKind::Number(steps) = steps_value.kind else {
            return Err(LanaError::Type);
        };
        if !steps.is_finite() || steps < 1.0 || steps.floor() != steps || steps > usize::MAX as f64 {
            return Err(LanaError::InvalidParameters);
        }
        let step_count = steps as usize;

        if !self.has_named_capability("train") {
            return Err(LanaError::Capability);
        }

        let data_current = self.reactive_value(new_data);
        let dataset_size = match &data_current.kind {
            ValueKind::Array(array) => array.lock().unwrap().items.len(),
            ValueKind::Lazy { function, bound } => {
                if *function as usize >= self.chunk.functions.len()
                    || self.chunk.functions[*function as usize].arity != 1
                {
                    return Err(LanaError::Type);
                }
                *bound
            }
            _ => return Err(LanaError::Type),
        };
        if dataset_size == 0 {
            return Err(LanaError::InvalidParameters);
        }

        self.incremental_train(prior, &data_current, dataset_size, step_count, scratch_register, out)
    }

    /// Recompute a `ReactiveKind::Train` node on `observe`: run one incremental
    /// optimizer step over the new observation (a [x, target] point). Mirrors
    /// `reactive_train_recompute` in `vm/c/vm.c`.
    fn reactive_train_recompute(
        &mut self,
        node: &Arc<Mutex<Reactive>>,
        observation: &Value,
        scratch_register: u32,
        out: &mut Value,
    ) -> LanaError {
        let prior = {
            let guard = node.lock().unwrap();
            let Some(current) = &guard.current else {
                return LanaError::Type;
            };
            let ValueKind::TrainingResult(prior) = &current.kind else {
                return LanaError::Type;
            };
            prior.clone()
        };

        let single = match Array::from_items(&self.heap, vec![observation.clone()]) {
            Ok(array) => Arc::new(Mutex::new(array)),
            Err(error) => return error,
        };
        let single_value = Value::array(single);

        match self.incremental_train(&prior, &single_value, 1, 1, scratch_register, out) {
            Ok(()) => LanaError::Ok,
            Err(error) => error,
        }
    }

    /// `resume(run, i) -> VAL_TRAINING_RESULT`, mirroring `host_resume` in
    /// `vm/c/vm.c`. The original run is unchanged; the returned run continues
    /// from step `i` with the parameters and optimizer state the original run
    /// had at that step, recomputing steps `i+1..` byte-identically to the
    /// original. The training loop is deterministic (no RNG consumption), so the
    /// continuation is byte-identical by construction.
    fn host_resume(
        &mut self,
        arguments: &[Value],
        scratch_register: u32,
        out: &mut Value,
    ) -> Result<(), LanaError> {
        if arguments.len() != 2 {
            return Err(LanaError::Type);
        }
        let run = &arguments[0];
        let index_value = &arguments[1];

        let ValueKind::TrainingResult(prior) = &run.kind else {
            return Err(LanaError::Type);
        };

        // Step index must be a nonnegative integer.
        let ValueKind::Number(index) = index_value.kind else {
            return Err(LanaError::InvalidParameters);
        };
        if !index.is_finite() || index < 0.0 || index.floor() != index || index > usize::MAX as f64 {
            return Err(LanaError::InvalidParameters);
        }
        let step_index = index as usize;

        if prior.model_function as usize >= self.chunk.functions.len()
            || prior.loss_function as usize >= self.chunk.functions.len()
        {
            return Err(LanaError::Type);
        }
        if self.chunk.functions[prior.model_function as usize].arity != 2
            || self.chunk.functions[prior.loss_function as usize].arity != 2
        {
            return Err(LanaError::Type);
        }

        if !self.has_named_capability("train") {
            return Err(LanaError::Capability);
        }

        let total_steps = prior.steps.lock().unwrap().items.len();
        if step_index >= total_steps {
            return Err(LanaError::Key);
        }

        // Resolve the stored dataset and effective batch size.
        let data_current = &prior.data;
        let (lazy_fn, dataset_size) = match &data_current.kind {
            ValueKind::Array(array) => (None, array.lock().unwrap().items.len()),
            ValueKind::Lazy { function, bound } => {
                if *function as usize >= self.chunk.functions.len()
                    || self.chunk.functions[*function as usize].arity != 1
                {
                    return Err(LanaError::Type);
                }
                (Some(*function), *bound)
            }
            _ => return Err(LanaError::Type),
        };
        if dataset_size == 0 {
            return Err(LanaError::InvalidParameters);
        }

        let mut batch = prior.batch_size;
        if batch == 0 || batch > dataset_size {
            batch = dataset_size;
        }
        let batches_per_epoch = (dataset_size + batch - 1) / batch;

        let optimizer = prior.optimizer.clone();
        let is_adam = &*optimizer.name == "adam";

        // Recover the parameters and optimizer state from the step map at
        // `step_index`, so the continuation resumes exactly where the original
        // run was at that step.
        let mut m: Option<Arc<Tensor>> = None;
        let mut v: Option<Arc<Tensor>> = None;
        let mut velocity: Option<Arc<Tensor>> = None;
        let mut adam_t = 0usize;
        let mut params_deriv: Option<Arc<Derivation>>;

        let step_value = prior.steps.lock().unwrap().items[step_index].clone();
        let ValueKind::Map(step_map) = &step_value.kind else {
            return Err(LanaError::Type);
        };
        let state_value = step_map
            .lock()
            .unwrap()
            .get("optimizer_state")
            .cloned()
            .ok_or(LanaError::Type)?;
        let ValueKind::Map(state_map) = &state_value.kind else {
            return Err(LanaError::Type);
        };
        let (m_value, v_value, t_value, vel_value) = {
            let guard = state_map.lock().unwrap();
            let m_value = if is_adam { guard.get("m").cloned() } else { None };
            let v_value = if is_adam { guard.get("v").cloned() } else { None };
            let t_value = if is_adam { guard.get("t").cloned() } else { None };
            let vel_value = if !is_adam && optimizer.momentum != 0.0 {
                guard.get("velocity").cloned()
            } else {
                None
            };
            (m_value, v_value, t_value, vel_value)
        };
        if is_adam {
            let m_value = m_value.ok_or(LanaError::Type)?;
            let v_value = v_value.ok_or(LanaError::Type)?;
            let t_value = t_value.ok_or(LanaError::Type)?;
            let ValueKind::Tensor(m_tensor) = &m_value.kind else {
                return Err(LanaError::Type);
            };
            let ValueKind::Tensor(v_tensor) = &v_value.kind else {
                return Err(LanaError::Type);
            };
            let ValueKind::Number(t) = t_value.kind else {
                return Err(LanaError::Type);
            };
            m = Some(self.tensor_copy_contiguous(m_tensor)?);
            v = Some(self.tensor_copy_contiguous(v_tensor)?);
            adam_t = t as usize;
        } else if optimizer.momentum != 0.0 {
            let vel_value = vel_value.ok_or(LanaError::Type)?;
            let ValueKind::Tensor(vel_tensor) = &vel_value.kind else {
                return Err(LanaError::Type);
            };
            velocity = Some(self.tensor_copy_contiguous(vel_tensor)?);
        }
        let params_value = step_map
            .lock()
            .unwrap()
            .get("parameters")
            .cloned()
            .ok_or(LanaError::Type)?;
        let ValueKind::Tensor(params_tensor) = &params_value.kind else {
            return Err(LanaError::Type);
        };
        let mut params = self.tensor_copy_contiguous(params_tensor)?;
        params_deriv = params_value.derivation.clone();

        let param_count = tensor::tensor_element_count(&params);

        let mut batch_grad = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, params.ndim, &params.shape, false)?)
        };

        // New step history: copy steps 0..=step_index, recompute step_index+1..
        let mut steps = Array::new(&self.heap, total_steps)?;
        {
            let prior_items = prior.steps.lock().unwrap();
            for item in prior_items.items.iter().take(step_index + 1) {
                steps.items.push(item.clone())?;
            }
        }

        for j in (step_index + 1)..total_steps {
            let epoch = j / batches_per_epoch;
            let batch_index = j % batches_per_epoch;
            let mut batch_start = batch_index * batch;
            let mut batch_end = batch_start + batch;
            if batch_end > dataset_size {
                batch_end = dataset_size;
            }
            let batch_actual = batch_end - batch_start;

            {
                let bg = Arc::get_mut(&mut batch_grad).unwrap();
                for k in 0..param_count {
                    tensor_set_real(bg, k, 0.0);
                }
            }

            while batch_start < batch_end {
                let pair = if let Some(lazy_fn) = lazy_fn {
                    let index_value = Value::number(batch_start as f64);
                    let mut pair = Value::null();
                    let error = self.run_function(lazy_fn, &index_value, scratch_register, &mut pair);
                    if error != LanaError::Ok {
                        return Err(error);
                    }
                    pair
                } else {
                    let ValueKind::Array(array) = &data_current.kind else {
                        unreachable!("array dataset");
                    };
                    array.lock().unwrap().items[batch_start].clone()
                };
                let ValueKind::Array(pair_array) = &pair.kind else {
                    return Err(LanaError::Type);
                };
                let pair_items = pair_array.lock().unwrap();
                if pair_items.items.len() != 2 {
                    return Err(LanaError::Type);
                }
                let x = pair_items.items[0].clone();
                let target = pair_items.items[1].clone();
                drop(pair_items);

                let leaf = self.record_derivation_payload(
                    DerivationKind::Operation,
                    "input",
                    &[],
                    "",
                    0,
                    DerivationExactness::Exact,
                    "autodiff",
                    DerivationOutcome::Success,
                    "none",
                );
                let Some(mut leaf) = leaf else {
                    return Err(LanaError::Oom);
                };
                leaf.ad_a = Some(params.clone());
                let leaf = self.managed_payload(leaf)?;

                let mut params_with_deriv = Value::tensor(params.clone());
                params_with_deriv.derivation = Some(leaf.clone());

                self.ad_recording = true;
                let mut y = Value::null();
                let mut error = self.run_function2(
                    prior.model_function,
                    &params_with_deriv,
                    &x,
                    scratch_register,
                    &mut y,
                );
                if error == LanaError::Ok {
                    let y_arg = y.clone();
                    error = self.run_function2(prior.loss_function, &y_arg, &target, scratch_register, &mut y);
                }
                self.ad_recording = false;
                if error != LanaError::Ok {
                    return Err(error);
                }

                let loss_value = match &y.kind {
                    ValueKind::Number(n) => *n,
                    _ => return Err(LanaError::Type),
                };
                if !loss_value.is_finite() {
                    return Err(LanaError::InvalidParameters);
                }

                let mut seed = {
                    let alloc = self.heap.clone();
                    tensor::tensor_new(&alloc, 0, &[], false)?
                };
                tensor_set_real(&mut seed, 0, 1.0 / batch_actual as f64);
                if let Some(derivation) = &y.derivation {
                    let error = self.ad_backward(derivation, &seed);
                    if error != LanaError::Ok {
                        return Err(error);
                    }
                }

                let grad = {
                    let guard = leaf.ad_grad.lock().unwrap();
                    if let Some(t) = guard.as_ref() {
                        t.try_clone(&self.heap)?
                    } else {
                        let alloc = self.heap.clone();
                        tensor::tensor_new(&alloc, params.ndim, &params.shape, false)?
                    }
                };
                {
                    let bg = Arc::get_mut(&mut batch_grad).unwrap();
                    for k in 0..param_count {
                        if !tensor_get_real(&grad, k).is_finite() {
                            return Err(LanaError::InvalidParameters);
                        }
                        let cur = tensor_get_real(bg, k);
                        tensor_set_real(bg, k, cur + tensor_get_real(&grad, k));
                    }
                }

                batch_start += 1;
            }

            let mut new_params = {
                let alloc = self.heap.clone();
                Arc::new(tensor::tensor_new(&alloc, params.ndim, &params.shape, false)?)
            };
            if is_adam {
                adam_t += 1;
                let bc1 = 1.0 - optimizer.beta1.powf(adam_t as f64);
                let bc2 = 1.0 - optimizer.beta2.powf(adam_t as f64);
                let m_tensor = Arc::get_mut(m.as_mut().unwrap()).unwrap();
                let v_tensor = Arc::get_mut(v.as_mut().unwrap()).unwrap();
                let bg_tensor = Arc::get_mut(&mut batch_grad).unwrap();
                let np_tensor = Arc::get_mut(&mut new_params).unwrap();
                for k in 0..param_count {
                    let g = tensor_get_real(bg_tensor, k);
                    let m_cur = tensor_get_real(m_tensor, k);
                    tensor_set_real(m_tensor, k, optimizer.beta1 * m_cur + (1.0 - optimizer.beta1) * g);
                    let v_cur = tensor_get_real(v_tensor, k);
                    tensor_set_real(v_tensor, k, optimizer.beta2 * v_cur + (1.0 - optimizer.beta2) * g * g);
                    let m_hat = tensor_get_real(m_tensor, k) / bc1;
                    let v_hat = tensor_get_real(v_tensor, k) / bc2;
                    tensor_set_real(np_tensor, k, tensor_get_real(&params, k)
                        - optimizer.learning_rate * m_hat / (v_hat.sqrt() + optimizer.epsilon));
                }
            } else if let Some(vel) = velocity.as_mut() {
                let vel_tensor = Arc::get_mut(vel).unwrap();
                let bg_tensor = Arc::get_mut(&mut batch_grad).unwrap();
                let np_tensor = Arc::get_mut(&mut new_params).unwrap();
                for k in 0..param_count {
                    let vel_cur = tensor_get_real(vel_tensor, k);
                    let bg_cur = tensor_get_real(bg_tensor, k);
                    tensor_set_real(vel_tensor, k, optimizer.momentum * vel_cur
                        - optimizer.learning_rate * bg_cur);
                    let vel_new = tensor_get_real(vel_tensor, k);
                    tensor_set_real(np_tensor, k, tensor_get_real(&params, k) + vel_new);
                }
            } else {
                let bg_tensor = Arc::get_mut(&mut batch_grad).unwrap();
                let np_tensor = Arc::get_mut(&mut new_params).unwrap();
                for k in 0..param_count {
                    let bg_cur = tensor_get_real(bg_tensor, k);
                    tensor_set_real(np_tensor, k, tensor_get_real(&params, k) - optimizer.learning_rate * bg_cur);
                }
            }

            // Non-finite optimizer state after a step → InvalidParameters.
            if is_adam {
                let m_tensor = m.as_ref().unwrap();
                let v_tensor = v.as_ref().unwrap();
                for k in 0..param_count {
                    if !tensor_get_real(m_tensor, k).is_finite() || !tensor_get_real(v_tensor, k).is_finite() {
                        return Err(LanaError::InvalidParameters);
                    }
                }
            } else if let Some(vel) = velocity.as_ref() {
                for k in 0..param_count {
                    if !tensor_get_real(vel, k).is_finite() {
                        return Err(LanaError::InvalidParameters);
                    }
                }
            }

            let mut pre_value = Value::tensor(params.clone());
            pre_value.derivation = params_deriv.clone();
            let grad_snap = self.tensor_copy_contiguous(&batch_grad)?;
            let grad_value = Value::tensor(grad_snap);
            let inputs = [&pre_value, &grad_value];
            let details = format!("epoch={} batch={}", epoch, batch_index);
            let step_deriv = self.record_derivation(
                DerivationKind::Operation,
                "train_step",
                &inputs,
                "",
                0,
                DerivationExactness::Exact,
                &details,
                DerivationOutcome::Success,
                "none",
            );
            let Some(step_deriv) = step_deriv else {
                return Err(LanaError::Oom);
            };
            let mut new_params_value = Value::tensor(new_params.clone());
            new_params_value.derivation = Some(step_deriv.clone());

            let mut step_map = Map::new(&self.heap, 5)?;
            step_map.set(Arc::from("epoch"), Value::number(epoch as f64), true)?;
            step_map.set(Arc::from("batch"), Value::number(batch_index as f64), true)?;
            step_map.set(Arc::from("parameters"), new_params_value, true)?;
            step_map.set(Arc::from("gradient"), grad_value, true)?;

            let mut state_map = Map::new(&self.heap, 3)?;
            if is_adam {
                let m_snap = self.tensor_copy_contiguous(m.as_ref().unwrap())?;
                let v_snap = self.tensor_copy_contiguous(v.as_ref().unwrap())?;
                state_map.set(Arc::from("m"), Value::tensor(m_snap), true)?;
                state_map.set(Arc::from("v"), Value::tensor(v_snap), true)?;
                state_map.set(Arc::from("t"), Value::number(adam_t as f64), true)?;
            } else if let Some(vel) = velocity.as_ref() {
                let vel_snap = self.tensor_copy_contiguous(vel)?;
                state_map.set(Arc::from("velocity"), Value::tensor(vel_snap), true)?;
            }
            step_map.set(
                Arc::from("optimizer_state"),
                Value::map(Arc::new(Mutex::new(state_map))),
                true,
            )?;

            steps.items.push(Value::map(Arc::new(Mutex::new(step_map))))?;

            params = new_params;
            params_deriv = Some(step_deriv);
        }

        let mut result_value = Value::training_result(self.managed_payload(TrainingResult {
            params,
            steps: Arc::new(Mutex::new(steps)),
            model_function: prior.model_function,
            loss_function: prior.loss_function,
            optimizer: prior.optimizer.clone(),
            data: prior.data.clone(),
            batch_size: prior.batch_size,
        })?);

        // Record the resumption point as a run-level derivation.
        let resume_inputs = [run];
        let resume_details = format!("step={}", step_index);
        let resume_deriv = self.record_derivation(
            DerivationKind::Operation,
            "resume",
            &resume_inputs,
            "",
            0,
            DerivationExactness::Exact,
            &resume_details,
            DerivationOutcome::Success,
            "none",
        );
        let Some(resume_deriv) = resume_deriv else {
            return Err(LanaError::Oom);
        };
        result_value.derivation = Some(resume_deriv);

        *out = result_value;
        Ok(())
    }

    /// A standard-normal draw via the Box-Muller transform, mirroring
    /// `gaussian_sample` in `vm/c/vm.c`. Deterministic given the VM RNG.
    fn gaussian_sample(&mut self) -> f64 {
        loop {
            let x = self.uniform_signed();
            let y = self.uniform_signed();
            let radius_squared = x * x + y * y;
            if radius_squared <= 0.0 || radius_squared >= 1.0 {
                continue;
            }
            return x * (-2.0 * radius_squared.ln() / radius_squared).sqrt();
        }
    }

    /// A uniform draw in [0, 1), mirroring `uniform01` in `vm/c/vm.c`.
    fn uniform01(&mut self) -> f64 {
        self.rng.random() as f64 / 4294967296.0
    }

    /// The Gaussian log-likelihood of `data` under `model(params)` with unit
    /// variance: -0.5 * sum((model(params) - data)^2). The model is an arity-1
    /// function returning a real tensor of the same shape as `data`.
    fn infer_log_likelihood(
        &mut self,
        model_fn: u32,
        params: &Tensor,
        data: &Tensor,
        scratch: u32,
    ) -> Result<f64, LanaError> {
        let params_value = Value::tensor(Arc::new(params.try_clone(&self.heap)?));
        let mut pred = Value::null();
        let error = self.run_function(model_fn, &params_value, scratch, &mut pred);
        if error != LanaError::Ok {
            return Err(error);
        }
        let ValueKind::Tensor(p) = &pred.kind else {
            return Err(LanaError::Type);
        };
        if p.is_complex {
            return Err(LanaError::Type);
        }
        if !tensor::tensor_shape_equal(p, data) {
            return Err(LanaError::Type);
        }
        let count = tensor::tensor_element_count(data);
        let mut sse = 0.0;
        for lin in 0..count {
            let mut rem = lin;
            let mut ip = p.offset;
            let mut id = data.offset;
            for d in (0..p.ndim).rev() {
                ip += (if p.shape[d] == 0 { 0 } else { rem % p.shape[d] }) * p.strides[d];
                id += (if data.shape[d] == 0 { 0 } else { rem % data.shape[d] }) * data.strides[d];
                rem /= p.shape[d];
            }
            let diff = tensor_get_real(p, ip) - tensor_get_real(data, id);
            sse += diff * diff;
        }
        Ok(-0.5 * sse)
    }

    /// Append a step map `{ step, parameters, log_likelihood }` to `steps`,
    /// mirroring `infer_record_step` in `vm/c/vm.c`. The `parameters` value
    /// carries a derivation node tracing to the prior and the observed data.
    fn infer_record_step(
        &mut self,
        steps: &mut Array,
        step: usize,
        params: &Tensor,
        log_likelihood: f64,
        prior: &Tensor,
        data: &Tensor,
    ) -> Result<(), LanaError> {
        let mut map = Map::new(&self.heap, 3)?;
        let snap = self.tensor_copy_contiguous(params)?;
        let prior_value = Value::tensor(Arc::new(prior.try_clone(&self.heap)?));
        let data_value = Value::tensor(Arc::new(data.try_clone(&self.heap)?));
        let inputs = [&prior_value, &data_value];
        let step_deriv = self.record_derivation(
            DerivationKind::Operation,
            "infer_step",
            &inputs,
            "",
            0,
            DerivationExactness::Sample,
            "infer",
            DerivationOutcome::Success,
            "none",
        );
        let mut params_value = Value::tensor(snap);
        if let Some(step_deriv) = step_deriv {
            params_value.derivation = Some(step_deriv);
        }
        map.set(Arc::from("step"), Value::number(step as f64), true)?;
        map.set(Arc::from("parameters"), params_value, true)?;
        map.set(Arc::from("log_likelihood"), Value::number(log_likelihood), true)?;
        steps.items.push(Value::map(Arc::new(Mutex::new(map))))?;
        Ok(())
    }

    /// `mcmc(samples, burn_in)`, mirroring `host_mcmc` in `vm/c/vm.c`.
    fn host_mcmc(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        let mut samples = 10000.0;
        let mut burn_in = 1000.0;
        if arguments.len() > 2 {
            return LanaError::Type;
        }
        if arguments.len() >= 1 {
            let ValueKind::Number(n) = arguments[0].kind else {
                return LanaError::Type;
            };
            samples = n;
        }
        if arguments.len() >= 2 {
            let ValueKind::Number(n) = arguments[1].kind else {
                return LanaError::Type;
            };
            burn_in = n;
        }
        if !samples.is_finite()
            || samples < 1.0
            || samples.floor() != samples
            || samples > usize::MAX as f64
        {
            return LanaError::InvalidParameters;
        }
        if !burn_in.is_finite()
            || burn_in < 0.0
            || burn_in.floor() != burn_in
            || burn_in > usize::MAX as f64
        {
            return LanaError::InvalidParameters;
        }
        if burn_in >= samples {
            return LanaError::InvalidParameters;
        }
        if self.alloc_bytes(std::mem::size_of::<InferenceAlgorithm>()) != LanaError::Ok {
            return LanaError::Oom;
        }
        *out = Value::inference_algorithm(Arc::new(InferenceAlgorithm {
            name: Arc::from("mcmc"),
            family: None,
            samples,
            burn_in,
            iterations: 0.0,
        }));
        LanaError::Ok
    }

    /// `vi(family, iterations)`, mirroring `host_vi` in `vm/c/vm.c`.
    fn host_vi(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        let mut family: Arc<str> = Arc::from("gaussian");
        let mut iterations = 1000.0;
        if arguments.len() > 2 {
            return LanaError::Type;
        }
        if arguments.len() >= 1 {
            let ValueKind::String(s) = &arguments[0].kind else {
                return LanaError::Type;
            };
            family = s.clone();
        }
        if arguments.len() >= 2 {
            let ValueKind::Number(n) = arguments[1].kind else {
                return LanaError::Type;
            };
            iterations = n;
        }
        if &*family != "gaussian" && &*family != "mean_field" {
            return LanaError::InvalidParameters;
        }
        if !iterations.is_finite()
            || iterations < 1.0
            || iterations.floor() != iterations
            || iterations > usize::MAX as f64
        {
            return LanaError::InvalidParameters;
        }
        if self.alloc_bytes(std::mem::size_of::<InferenceAlgorithm>()) != LanaError::Ok {
            return LanaError::Oom;
        }
        *out = Value::inference_algorithm(Arc::new(InferenceAlgorithm {
            name: Arc::from("vi"),
            family: Some(family),
            samples: 0.0,
            burn_in: 0.0,
            iterations,
        }));
        LanaError::Ok
    }

    /// `smc(particles)`, mirroring `host_smc` in `vm/c/vm.c`.
    fn host_smc(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        let mut particles = 1000.0;
        if arguments.len() > 1 {
            return LanaError::Type;
        }
        if arguments.len() >= 1 {
            let ValueKind::Number(n) = arguments[0].kind else {
                return LanaError::Type;
            };
            particles = n;
        }
        if !particles.is_finite()
            || particles < 1.0
            || particles.floor() != particles
            || particles > usize::MAX as f64
        {
            return LanaError::InvalidParameters;
        }
        if self.alloc_bytes(std::mem::size_of::<InferenceAlgorithm>()) != LanaError::Ok {
            return LanaError::Oom;
        }
        *out = Value::inference_algorithm(Arc::new(InferenceAlgorithm {
            name: Arc::from("smc"),
            family: None,
            samples: particles,
            burn_in: 0.0,
            iterations: 0.0,
        }));
        LanaError::Ok
    }

    /// Metropolis-Hastings random walk, mirroring `infer_mcmc` in `vm/c/vm.c`.
    fn infer_mcmc(
        &mut self,
        model_fn: u32,
        prior: &Tensor,
        data: &Tensor,
        samples: usize,
        burn_in: usize,
        scratch: u32,
    ) -> Result<Posterior, LanaError> {
        let param_count = tensor::tensor_element_count(prior);
        let kept = samples - burn_in;
        let mut params = self.tensor_copy_contiguous(prior)?;
        let sample_shape = [kept, param_count];
        let mut sample_matrix = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, 2, &sample_shape, false)?)
        };
        let mut steps = Array::new(&self.heap, samples)?;

        let mut current_ll = self.infer_log_likelihood(model_fn, &params, data, scratch)?;

        let mut kept_index = 0usize;
        for i in 0..samples {
            let error = self.consume_sampling_budget();
            if error != LanaError::Ok {
                return Err(error);
            }
            let mut proposal = {
                let alloc = self.heap.clone();
                Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
            };
            {
                let proposal_tensor = Arc::get_mut(&mut proposal).unwrap();
                for k in 0..param_count {
                    tensor_set_real(proposal_tensor, k, tensor_get_real(&params, k) + 0.1 * self.gaussian_sample());
                }
            }
            let proposal_ll = self.infer_log_likelihood(model_fn, &proposal, data, scratch)?;
            let log_alpha = proposal_ll - current_ll;
            let accept = log_alpha >= 0.0 || self.uniform01() < log_alpha.exp();
            if accept {
                params = proposal;
                current_ll = proposal_ll;
            }
            if i >= burn_in {
                let sm = Arc::get_mut(&mut sample_matrix).unwrap();
                for k in 0..param_count {
                    tensor_set_real(sm, kept_index * param_count + k, tensor_get_real(&params, k));
                }
                kept_index += 1;
            }
            self.infer_record_step(&mut steps, i, &params, current_ll, prior, data)?;
        }

        let mut mean = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut variance = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
        };
        {
            let mean_tensor = Arc::get_mut(&mut mean).unwrap();
            let variance_tensor = Arc::get_mut(&mut variance).unwrap();
            let sm = Arc::get_mut(&mut sample_matrix).unwrap();
            for k in 0..param_count {
                let mut sum = 0.0;
                for s in 0..kept {
                    sum += tensor_get_real(sm, s * param_count + k);
                }
                let m = sum / kept as f64;
                let mut var = 0.0;
                for s in 0..kept {
                    let d = tensor_get_real(sm, s * param_count + k) - m;
                    var += d * d;
                }
                tensor_set_real(mean_tensor, k, m);
                tensor_set_real(variance_tensor, k, var / kept as f64);
            }
        }

        Ok(Posterior {
            mean,
            variance,
            samples: Some(sample_matrix),
            steps: Arc::new(Mutex::new(steps)),
            seed: self.root_seed,
        })
    }

    /// Mean-field Gaussian variational inference via the reparameterization
    /// trick and LIP-011 autodiff, mirroring `infer_vi` in `vm/c/vm.c`.
    fn infer_vi(
        &mut self,
        model_fn: u32,
        prior: &Tensor,
        data: &Tensor,
        iterations: usize,
        scratch: u32,
    ) -> Result<Posterior, LanaError> {
        let param_count = tensor::tensor_element_count(prior);
        const LR: f64 = 0.01;
        let mut mu = self.tensor_copy_contiguous(prior)?;
        let mut log_sigma = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut eps = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut sigma = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut params = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
        };
        {
            let log_sigma_tensor = Arc::get_mut(&mut log_sigma).unwrap();
            for k in 0..param_count {
                tensor_set_real(log_sigma_tensor, k, 0.1f64.ln());
            }
        }
        let mut steps = Array::new(&self.heap, iterations)?;

        for i in 0..iterations {
            let error = self.consume_sampling_budget();
            if error != LanaError::Ok {
                return Err(error);
            }
            {
                let eps_tensor = Arc::get_mut(&mut eps).unwrap();
                let sigma_tensor = Arc::get_mut(&mut sigma).unwrap();
                let params_tensor = Arc::get_mut(&mut params).unwrap();
                let log_sigma_tensor = Arc::get_mut(&mut log_sigma).unwrap();
                let mu_tensor = Arc::get_mut(&mut mu).unwrap();
                for k in 0..param_count {
                    let e = self.gaussian_sample();
                    tensor_set_real(eps_tensor, k, e);
                    let s = tensor_get_real(log_sigma_tensor, k).exp();
                    tensor_set_real(sigma_tensor, k, s);
                    tensor_set_real(params_tensor, k, tensor_get_real(mu_tensor, k) + s * e);
                }
            }
            let params_value = Value::tensor(params.clone());
            let mut pred = Value::null();
            let error = self.run_function(model_fn, &params_value, scratch, &mut pred);
            if error != LanaError::Ok {
                return Err(error);
            }
            let ValueKind::Tensor(pred_tensor) = &pred.kind else {
                return Err(LanaError::Type);
            };
            if pred_tensor.is_complex {
                return Err(LanaError::Type);
            }
            if !tensor::tensor_shape_equal(pred_tensor, data) {
                return Err(LanaError::Type);
            }
            let diff = {
                let alloc = self.heap.clone();
                Arc::new(tensor::tensor_elementwise(&alloc, pred_tensor, data, 1)?)
            };
            let diff_value = Value::tensor(diff.clone());
            let model_value = Value::function(model_fn);
            let mut grad_value = Value::null();
            let error = self.ad_vjp(&model_value, &params_value, &diff_value, scratch, &mut grad_value);
            if error != LanaError::Ok {
                return Err(error);
            }
            let ValueKind::Tensor(grad) = &grad_value.kind else {
                return Err(LanaError::Type);
            };
            {
                let mu_tensor = Arc::get_mut(&mut mu).unwrap();
                let log_sigma_tensor = Arc::get_mut(&mut log_sigma).unwrap();
                let sigma_tensor = Arc::get_mut(&mut sigma).unwrap();
                let eps_tensor = Arc::get_mut(&mut eps).unwrap();
                for k in 0..param_count {
                    let g = tensor_get_real(grad, grad.offset + k);
                    let d_mu = g + (tensor_get_real(mu_tensor, k) - tensor_get_real(prior, prior.offset + k));
                    let d_log_sigma = g * tensor_get_real(sigma_tensor, k) * tensor_get_real(eps_tensor, k)
                        + (tensor_get_real(sigma_tensor, k) * tensor_get_real(sigma_tensor, k) - 1.0);
                    tensor_set_real(mu_tensor, k, tensor_get_real(mu_tensor, k) - LR * d_mu);
                    tensor_set_real(log_sigma_tensor, k, tensor_get_real(log_sigma_tensor, k) - LR * d_log_sigma);
                }
            }
            let ll = self.infer_log_likelihood(model_fn, &mu, data, scratch)?;
            self.current_frame_mut().registers[scratch as usize] = Value::null();
            self.infer_record_step(&mut steps, i, &mu, ll, prior, data)?;
        }

        let mut mean = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut variance = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
        };
        {
            let mean_tensor = Arc::get_mut(&mut mean).unwrap();
            let variance_tensor = Arc::get_mut(&mut variance).unwrap();
            let mu_tensor = Arc::get_mut(&mut mu).unwrap();
            let log_sigma_tensor = Arc::get_mut(&mut log_sigma).unwrap();
            for k in 0..param_count {
                tensor_set_real(mean_tensor, k, tensor_get_real(mu_tensor, k));
                tensor_set_real(variance_tensor, k, (2.0 * tensor_get_real(log_sigma_tensor, k)).exp());
            }
        }

        Ok(Posterior {
            mean,
            variance,
            samples: None,
            steps: Arc::new(Mutex::new(steps)),
            seed: self.root_seed,
        })
    }

    /// Sequential Monte Carlo (particle filter) over a single observation,
    /// mirroring `infer_smc` in `vm/c/vm.c`.
    fn infer_smc(
        &mut self,
        model_fn: u32,
        prior: &Tensor,
        data: &Tensor,
        particles: usize,
        scratch: u32,
    ) -> Result<Posterior, LanaError> {
        let param_count = tensor::tensor_element_count(prior);
        let particle_shape = [particles, param_count];
        let mut matrix = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, 2, &particle_shape, false)?)
        };
        {
            let matrix_tensor = Arc::get_mut(&mut matrix).unwrap();
            for p in 0..particles {
                for k in 0..param_count {
                    tensor_set_real(matrix_tensor, p * param_count + k, tensor_get_real(&prior, prior.offset + k));
                }
            }
        }
        let mut weights = vec![0.0; particles];
        if self.alloc_bytes(particles * std::mem::size_of::<f64>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut steps = Array::new(&self.heap, 1)?;

        let error = self.consume_sampling_budget();
        if error != LanaError::Ok {
            return Err(error);
        }

        {
            let matrix_tensor = Arc::get_mut(&mut matrix).unwrap();
            for p in 0..particles {
                for k in 0..param_count {
                    let cur = tensor_get_real(matrix_tensor, p * param_count + k);
                    tensor_set_real(matrix_tensor, p * param_count + k, cur + 0.1 * self.gaussian_sample());
                }
            }
        }

        let mut weight_sum = 0.0;
        for p in 0..particles {
            let mut particle = {
                let alloc = self.heap.clone();
                Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
            };
            {
                let particle_tensor = Arc::get_mut(&mut particle).unwrap();
                for k in 0..param_count {
                    tensor_set_real(particle_tensor, k, tensor_get_real(&matrix, p * param_count + k));
                }
            }
            let ll = self.infer_log_likelihood(model_fn, &particle, data, scratch)?;
            weights[p] = ll.exp();
            weight_sum += weights[p];
        }
        if !(weight_sum > 0.0) || !weight_sum.is_finite() {
            return Err(LanaError::InvalidParameters);
        }
        for p in 0..particles {
            weights[p] /= weight_sum;
        }

        let mut resampled = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, 2, &particle_shape, false)?)
        };
        {
            let u0 = self.uniform01() / particles as f64;
            let mut cumulative = 0.0;
            let mut source = 0usize;
            let resampled_tensor = Arc::get_mut(&mut resampled).unwrap();
            for p in 0..particles {
                let threshold = u0 + p as f64 / particles as f64;
                while cumulative < threshold && source < particles {
                    cumulative += weights[source];
                    source += 1;
                }
                let pick = if source == 0 { 0 } else { source - 1 };
                for k in 0..param_count {
                    tensor_set_real(resampled_tensor, p * param_count + k, tensor_get_real(&matrix, pick * param_count + k));
                }
            }
        }

        let mut mean = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut variance = {
            let alloc = self.heap.clone();
            Arc::new(tensor::tensor_new(&alloc, prior.ndim, &prior.shape, false)?)
        };
        {
            let mean_tensor = Arc::get_mut(&mut mean).unwrap();
            let variance_tensor = Arc::get_mut(&mut variance).unwrap();
            for k in 0..param_count {
                let mut sum = 0.0;
                for p in 0..particles {
                    sum += tensor_get_real(&resampled, p * param_count + k);
                }
                let m = sum / particles as f64;
                let mut var = 0.0;
                for p in 0..particles {
                    let d = tensor_get_real(&resampled, p * param_count + k) - m;
                    var += d * d;
                }
                tensor_set_real(mean_tensor, k, m);
                tensor_set_real(variance_tensor, k, var / particles as f64);
            }
        }

        self.infer_record_step(&mut steps, 0, &mean, 0.0, prior, data)?;

        Ok(Posterior {
            mean,
            variance,
            samples: Some(resampled),
            steps: Arc::new(Mutex::new(steps)),
            seed: self.root_seed,
        })
    }

    /// `infer(prior, model, data, algorithm)`, mirroring `host_infer` in
    /// `vm/c/vm.c`.
    fn host_infer(
        &mut self,
        arguments: &[Value],
        scratch_register: u32,
        out: &mut Value,
    ) -> Result<(), LanaError> {
        if arguments.len() != 4 {
            return Err(LanaError::Type);
        }
        let prior = &arguments[0];
        let model = &arguments[1];
        let data = &arguments[2];
        let algorithm_value = &arguments[3];

        let ValueKind::Function(model_fn) = model.kind else {
            return Err(LanaError::Type);
        };
        if model_fn as usize >= self.chunk.functions.len() {
            return Err(LanaError::Type);
        }
        if self.chunk.functions[model_fn as usize].arity != 1 {
            return Err(LanaError::Type);
        }
        let ValueKind::Tensor(prior_tensor) = &prior.kind else {
            return Err(LanaError::Type);
        };
        if prior_tensor.is_complex {
            return Err(LanaError::Type);
        }
        let ValueKind::Tensor(data_tensor) = &data.kind else {
            return Err(LanaError::Type);
        };
        if data_tensor.is_complex {
            return Err(LanaError::Type);
        }
        let ValueKind::InferenceAlgorithm(algorithm) = &algorithm_value.kind else {
            return Err(LanaError::Type);
        };

        if tensor::tensor_element_count(data_tensor) == 0 {
            return Err(LanaError::InvalidParameters);
        }

        if !self.has_named_capability("infer") {
            return Err(LanaError::Capability);
        }


        let posterior = if &*algorithm.name == "mcmc" {
            self.infer_mcmc(
                model_fn,
                prior_tensor,
                data_tensor,
                algorithm.samples as usize,
                algorithm.burn_in as usize,
                scratch_register,
            )?
        } else if &*algorithm.name == "vi" {
            self.infer_vi(
                model_fn,
                prior_tensor,
                data_tensor,
                algorithm.iterations as usize,
                scratch_register,
            )?
        } else if &*algorithm.name == "smc" {
            self.infer_smc(
                model_fn,
                prior_tensor,
                data_tensor,
                algorithm.samples as usize,
                scratch_register,
            )?
        } else {
            return Err(LanaError::InvalidParameters);
        };

        *out = Value::posterior(self.managed_payload(posterior)?);
        Ok(())
    }

    /// Build a `Result` tagged pair `[ok, value]`, mirroring `make_result` in
    /// `vm/c/vm.c`.
    fn make_result(&self, ok: bool, value: Value) -> Result<Value, LanaError> {
        self.array_value(vec![Value::boolean(ok), value])
    }

    fn array_value(&self, items: Vec<Value>) -> Result<Value, LanaError> {
        Ok(Value::array(Arc::new(Mutex::new(Array::from_items(&self.heap, items)?))))
    }

    /// Dispatch one instruction, mirroring the `switch` in `lana_vm_run`.
    fn execute(&mut self, ins: &Instruction) -> LanaError {
        // Scratch capacity is charged to the heap; retain tracing headroom.
        let allocated = self.allocated_bytes();
        let pressure = allocated > self.memory_limit.saturating_sub(self.memory_limit / 4);
        let object_pressure = allocated > self.memory_limit / 2;
        let class_collection_due = !self.class_objects.is_empty()
            && (self.class_allocations_since_gc >= self.class_objects.len().saturating_sub(self.class_allocations_since_gc).max(64)
                || (object_pressure && (self.class_allocations_since_gc > 0 || ins.opcode == OpCode::ObjectNew)));
        let active_collection = self.host_roots.collection_major();
        let containers_due = self.heap.cycles_due(pressure);
        let severe_container_pressure = containers_due && ins.opcode == OpCode::ArrayNew;
        if self.constructions.is_empty()
            && (active_collection.is_some() || containers_due || class_collection_due) {
            let major = active_collection.unwrap_or_else(|| class_collection_due || self.heap.major_collection_due(pressure));
            let severe_object_pressure = object_pressure && ins.opcode == OpCode::ObjectNew;
            let mut collected = if severe_object_pressure || severe_container_pressure { self.collect_classes() }
                else { self.collect_safepoint_slice(major).map(|_| ()) };
            if collected == Err(LanaError::Oom) && !major {
                collected = self.collect_classes();
            }
            match collected {
                // Scratch admission can fail while the next mutator allocation
                // still fits. Leave the graph intact and let that allocation decide.
                // Failed admission did not perform a major collection, so do
                // not advance the major-generation pressure baseline.
                Err(LanaError::Oom) => self.heap.defer_cycles(),
                Err(error) => return error,
                Ok(()) => {},
            }
        }
        // Check the whole v6 feature set before effects, including debugger entry.
        if self.chunk.version == 6 {
            if let Err(error) = self.prepare_object_values() { return error; }
            let entry = self.object_function_entries.partition_point(|(entry, _)| *entry <= self.ip.saturating_sub(1));
            let owner = self.object_function_entries[entry.saturating_sub(1)].1;
            if owner != self.current_frame().object_method
                || self.object_methods.get(&self.current_frame().function).copied() != self.current_frame().object_method {
                return LanaError::UnsupportedOperation;
            }
            // Retain effect promises across indirect callbacks and ordinary helper frames.
            for frame in &self.frames {
                if let Some((owner, member)) = frame.object_method {
                    let descriptor = &self.object_descriptors.as_ref().unwrap()[&owner];
                    let method = descriptor.methods.get(member as usize);
                    let allowed = method.map_or(0, |m| m.effect_mask);
                    let effect = match object_effects::instruction_effect(ins) {
                        Ok(effect) => effect,
                        Err(error) => return error,
                    };
                    if effect & !allowed != 0 || (method.is_none_or(|m| m.is_init)
                        && effect != 0 && !matches!(ins.opcode, OpCode::ObjectNew | OpCode::OoSet)) {
                        return LanaError::UnsupportedOperation;
                    }
                }
            }
        }
        use OpCode::*;
        if self.pure_callback_depth > 0 {
            if self.evaluation_callback_depth > 0 && ins.opcode == HostCall &&
                (ins.b == LANA_HOST_ARRAY_PUSH ||
                    (LANA_HOST_DATASET..=LANA_HOST_DATASET_EXPLAIN).contains(&ins.b)) {
                return LanaError::UnsupportedOperation;
            }
            let allowed = matches!(ins.opcode, Nop | LoadConst | Move | GetField | GetIndex
                | Binary | Unary | Compare | Jump | JumpIfTrue | JumpIfFalse
                | ArrayNew | ArrayGet | Call | Return | LoadFunction | AdtBuild
                | AdtCase | AdtGet | ValueNew | OoGet | OoCall | OoStaticCall)
                || (ins.opcode == HostCall && matches!(ins.b,
                    LANA_HOST_ARRAY_LENGTH | LANA_HOST_INDEX_GET
                    | LANA_HOST_MAP_NEW | LANA_HOST_MAP_GET | LANA_HOST_MAP_HAS
                    | LANA_HOST_MAP_KEYS | LANA_HOST_TYPE_OF | LANA_HOST_FLOOR
                    | LANA_HOST_INFORMATION_SNAPSHOT))
                || (ins.opcode == HostCall && ins.b == LANA_HOST_ARRAY_PUSH && self.dataset_work_remaining.is_none())
                || (ins.opcode == HostCall && matches!(ins.b,
                    LANA_HOST_RULES_LEARN | LANA_HOST_RULES_PREDICT
                    | LANA_HOST_TREES_FIT | LANA_HOST_TREES_PREDICT | LANA_HOST_TREES_EXPLAIN))
                    || (ins.opcode == HostCall && (LANA_HOST_DATASET..=LANA_HOST_DATASET_EXPLAIN).contains(&ins.b));
            if !allowed { return LanaError::UnsupportedOperation; }
        }
        match ins.opcode {
            ValueNew | OoGet => self.execute_object_value(ins),
            OoAsInterface => self.execute_interface_conversion(ins),
            ObjectNew | OoSet => self.execute_class_instruction(ins),
            OoCall | OoStaticCall => self.execute_object_method(ins),
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
                        match self.state_dist_expected_probability(distribution) {
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
                    ValueKind::TrainingResult(_) if ins.c <= 1 => {
                        // LIP-010: reactive parameters resolve to the current
                        // training result, not the frozen one captured at `train`.
                        let effective = self.reactive_value(&source);
                        let ValueKind::TrainingResult(result) = &effective.kind else {
                            return LanaError::Type;
                        };
                        let field = if ins.c == 0 {
                            Value::tensor(result.params.clone())
                        } else {
                            Value::array(result.steps.clone())
                        };
                        self.current_frame_mut().registers[ins.b as usize] = field;
                        LanaError::Ok
                    }
                    ValueKind::Posterior(posterior) if ins.c <= 3 => {
                        let field = match ins.c {
                            0 => Value::tensor(posterior.mean.clone()),
                            1 => Value::tensor(posterior.variance.clone()),
                            2 => match &posterior.samples {
                                Some(samples) => Value::tensor(samples.clone()),
                                None => Value::null(),
                            },
                            _ => Value::array(posterior.steps.clone()),
                        };
                        self.current_frame_mut().registers[ins.b as usize] = field;
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
                if self.ad_recording
                    && matches!(left.kind, ValueKind::Tensor(_))
                    && matches!(right.kind, ValueKind::Tensor(_))
                {
                    let error = self.ad_record(ins.imm as i32, &left, Some(&right), -1, &mut out);
                    if error != LanaError::Ok {
                        return error;
                    }
                    self.current_frame_mut().registers[ins.c as usize] = out;
                    LanaError::Ok
                } else {
                    self.current_frame_mut().registers[ins.c as usize] = out;
                    if left.derivation.is_some() || right.derivation.is_some() {
                        let inputs = [&left, &right];
                        self.attach_derivation(ins.c, DerivationKind::Operation, "binary", &inputs, "",
                                               ins.line, DerivationExactness::Exact, "pure")
                    } else {
                        LanaError::Ok
                    }
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
                let build = |vm: &mut Self| -> Result<Value, LanaError> {
                    let mut items = vm.allocate_array_items(count)?;
                    items.extend((0..count).map(|i| vm.current_frame().registers[ins.b as usize + i].clone()))?;
                    Ok(Value::array(Arc::new(Mutex::new(Array::from_buffer(items)?))))
                };
                let value = match self.allocate_with_collection(build) { Ok(value) => value, Err(error) => return error };
                self.current_frame_mut().registers[ins.a as usize] = value;
                LanaError::Ok
            }
            ArrayGet | ArraySet => {
                if ins.opcode == ArraySet && self.active_path_count > 1 {
                    return LanaError::UnsupportedOperation;
                }
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
                let array = array.lock().unwrap();
                if index >= array.items.len() {
                    return LanaError::Limit;
                }
                if ins.opcode == ArrayGet {
                    let value = array.items[index].clone();
                    drop(array);
                    self.current_frame_mut().registers[ins.c as usize] = value;
                } else {
                    let value = self.current_frame().registers[ins.c as usize].clone();
                    if array.frozen { return LanaError::UnsupportedOperation; }
                    drop(array);
                    if let Err(error) = self.prepare_container_write(&array_value, &value) { return error; }
                    let ValueKind::Array(array) = &array_value.kind else { unreachable!() };
                    let mut array = array.lock().unwrap();
                    if index >= array.items.len() { return LanaError::Limit; }
                    if array.frozen { return LanaError::UnsupportedOperation; }
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
                let entry = function.entry as usize;
                if !self.constructing.is_empty() {
                    for index in 0..ins.imm as usize {
                        let value = self.current_frame().registers[ins.c as usize + index].clone();
                        if let Err(error) = self.check_complete_objects(&value) { return error; }
                    }
                }
                let mut callee = if self.chunk.version == 6 {
                    match Frame::charged(self.max_registers[ins.b as usize], &self.heap) {
                        Ok(frame) => frame, Err(error) => return error,
                    }
                } else { Frame::new(self.max_registers[ins.b as usize]) };
                callee.return_ip = self.ip;
                callee.return_register = ins.a;
                callee.function = ins.b;
                for index in 0..ins.imm as usize {
                    callee.registers[index] = self.current_frame().registers[ins.c as usize + index].clone();
                    callee.histories[index] = self.current_frame().histories[ins.c as usize + index].clone();
                }
                self.frames.push(callee);
                self.ip = entry;
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
            Generator => {
                let function = &self.chunk.functions[ins.b as usize];
                if ins.imm != function.arity {
                    return LanaError::Type;
                }
                let mut registers = Vec::with_capacity(function.register_count as usize);
                registers.resize_with(function.register_count as usize, Value::null);
                for index in 0..ins.imm as usize {
                    registers[1 + index] =
                        self.current_frame().registers[ins.c as usize + index].clone();
                }
                let generator = crate::value::Generator {
                    function: ins.b,
                    ip: function.entry as usize,
                    registers,
                    exhausted: false,
                };
                let generator = Arc::new(Mutex::new(generator));
                if let Err(error) = self.track_cycle(crate::heap::CycleWeak::Generator(Arc::downgrade(&generator))) { return error; }
                self.current_frame_mut().registers[ins.a as usize] = Value::generator(generator);
                LanaError::Ok
            }
            Yield => {
                if self.active_path_count > 1 { return LanaError::UnsupportedOperation; }
                let gen_value = self.current_frame().registers[ins.a as usize].clone();
                let yielded = self.current_frame().registers[ins.b as usize].clone();
                let ValueKind::Generator(generator) = gen_value.kind else {
                    return LanaError::Type;
                };
                {
                    self.heap.mutate_cycle(Arc::as_ptr(&generator) as usize);
                    let mut generator = generator.lock().unwrap();
                    generator.ip = self.ip;
                    for index in 1..generator.registers.len() {
                        generator.registers[index] =
                            self.current_frame().registers[index].clone();
                    }
                    generator.registers[0] = Value::null();
                }
                if self.frames.len() == 1 {
                    return LanaError::Type;
                }
                let return_ip = self.current_frame().return_ip;
                let destination = self.current_frame().return_register;
                self.frames.pop();
                let result = match self.make_result(true, yielded) { Ok(value) => value, Err(error) => return error };
                self.current_frame_mut().registers[destination as usize] = result;
                self.ip = return_ip;
                LanaError::Ok
            }
            Next => {
                if self.active_path_count > 1 { return LanaError::UnsupportedOperation; }
                let gen_value = self.current_frame().registers[ins.a as usize].clone();
                let ValueKind::Generator(generator) = &gen_value.kind else {
                    return LanaError::Type;
                };
                let generator = generator.clone();
                let (exhausted, function, ip, registers) = {
                    let generator = generator.lock().unwrap();
                    (
                        generator.exhausted,
                        generator.function,
                        generator.ip,
                        generator.registers.clone(),
                    )
                };
                if exhausted {
                    let result = match self.make_result(false, Value::string(Arc::from("exhausted"))) { Ok(value) => value, Err(error) => return error };
                    self.current_frame_mut().registers[ins.b as usize] = result;
                    return LanaError::Ok;
                }
                if self.frames.len() >= LANA_MAX_CALL_FRAMES as usize {
                    return LanaError::Limit;
                }
                let mut callee = Frame::new(self.max_registers[function as usize]);
                for index in 0..registers.len() {
                    callee.registers[index] = registers[index].clone();
                }
                callee.registers[0] = gen_value;
                callee.return_ip = self.ip;
                callee.return_register = ins.b;
                callee.function = function;
                callee.is_generator = true;
                self.frames.push(callee);
                self.ip = ip;
                LanaError::Ok
            }
            Async => {
                let function = &self.chunk.functions[ins.b as usize];
                if ins.imm != function.arity {
                    return LanaError::Type;
                }
                // Size the future's registers from the true max register index
                // (not `register_count`, which is only a lower bound) so the
                // async frame created on resume has room for every register the
                // function body may touch. Mirrors how OP_CALL sizes frames.
                let register_count = self.max_registers[ins.b as usize];
                let mut registers = Vec::with_capacity(register_count);
                registers.resize_with(register_count, Value::null);
                for index in 0..ins.imm as usize {
                    registers[1 + index] =
                        self.current_frame().registers[ins.c as usize + index].clone();
                }
                let future = crate::value::Future {
                    function: ins.b,
                    ip: function.entry as usize,
                    registers,
                    exhausted: false,
                    ready: true,
                    queued: false,
                };
                let future = Arc::new(Mutex::new(future));
                if let Err(error) = self.track_cycle(crate::heap::CycleWeak::Future(Arc::downgrade(&future))) { return error; }
                self.current_frame_mut().registers[ins.a as usize] = Value::future(future);
                LanaError::Ok
            }
            Await => {
                if self.active_path_count > 1 { return LanaError::UnsupportedOperation; }
                let awaited_value = self.current_frame().registers[ins.a as usize].clone();
                let ValueKind::Future(awaited) = &awaited_value.kind else {
                    return LanaError::Type;
                };
                let awaited = awaited.clone();
                // If the awaited future is already complete, store its result
                // and continue without suspending.
                if awaited.lock().unwrap().exhausted {
                    let result = awaited.lock().unwrap().registers[0].clone();
                    self.current_frame_mut().registers[ins.b as usize] = result;
                    return LanaError::Ok;
                }
                let current_value = self.current_frame().registers[0].clone();
                let ValueKind::Future(current) = current_value.kind else {
                    return LanaError::Type;
                };
                let current = current.clone();
                {
                    self.heap.mutate_cycle(Arc::as_ptr(&current) as usize);
                    let mut current = current.lock().unwrap();
                    // Re-execute this AWAIT on resume so the result lands in
                    // `dest` once the awaited future completes.
                    current.ip = self.ip - 1;
                    for index in 1..current.registers.len() {
                        current.registers[index] =
                            self.current_frame().registers[index].clone();
                    }
                    current.ready = false;
                }
                self.awaiters
                    .entry(Arc::as_ptr(&awaited) as usize)
                    .or_default()
                    .push(current.clone());
                self.frames.pop();
                self.enqueue_future(awaited);
                LanaError::Ok
            }
            RunAsync => {
                let future_value = self.current_frame().registers[ins.a as usize].clone();
                let ValueKind::Future(future) = &future_value.kind else {
                    return LanaError::Type;
                };
                let future = future.clone();
                self.enqueue_future(future.clone());
                let saved_ip = self.ip;
                let saved_active = self.event_loop_active;
                let saved_base = self.event_loop_base_depth;
                self.event_loop_base_depth = self.frames.len();
                let error = self.run_event_loop();
                self.ip = saved_ip;
                self.event_loop_active = saved_active;
                self.event_loop_base_depth = saved_base;
                if error != LanaError::Ok {
                    return error;
                }
                let result = {
                    let future = future.lock().unwrap();
                    future.registers[0].clone()
                };
                self.current_frame_mut().registers[ins.b as usize] = result;
                LanaError::Ok
            }
            LoadFunction => {
                self.current_frame_mut().registers[ins.a as usize] = Value::function(ins.b);
                LanaError::Ok
            }
            Bootstrap => {
                let data_value = self.current_frame().registers[ins.imm as usize].clone();
                let b_value = self.current_frame().registers[ins.c as usize].clone();
                let ValueKind::Array(ref data_arc) = data_value.kind else {
                    return LanaError::Type;
                };
                let ValueKind::Number(b) = b_value.kind else {
                    return LanaError::Type;
                };
                if !b.is_finite() || b < 1.0 || b.floor() != b || b > usize::MAX as f64 {
                    return LanaError::InvalidParameters;
                }
                let b = b as usize;
                let data = data_arc.lock().unwrap().items.to_vec();
                let n = data.len();
                if n == 0 {
                    return LanaError::InvalidParameters;
                }
                if ins.b as usize >= self.chunk.functions.len() {
                    return LanaError::Opcode;
                }
                if self.chunk.functions[ins.b as usize].arity != 1 {
                    return LanaError::Type;
                }
                let estimate = {
                    let mut result = Value::null();
                    let error = self.run_function(ins.b, &data_value, ins.a, &mut result);
                    if error != LanaError::Ok {
                        return error;
                    }
                    let ValueKind::Number(value) = result.kind else {
                        return LanaError::Type;
                    };
                    value
                };
                let mut resamples = Vec::with_capacity(b);
                for _ in 0..b {
                    let mut items = Vec::with_capacity(n);
                    for _ in 0..n {
                        let draw = (self.rng.random() as usize) % n;
                        items.push(data[draw].clone());
                    }
                    let resampled = match self.array_value(items) { Ok(value) => value, Err(error) => return error };
                    let mut result = Value::null();
                    let error = self.run_function(ins.b, &resampled, ins.a, &mut result);
                    if error != LanaError::Ok {
                        return error;
                    }
                    let ValueKind::Number(value) = result.kind else {
                        return LanaError::Type;
                    };
                    resamples.push(value);
                }
                resamples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let lo = ((0.025 * b as f64) as usize).min(b - 1);
                let hi = ((0.975 * b as f64) as usize).min(b - 1);
                let ci_low = resamples[lo];
                let ci_high = resamples[hi];
                let mut map = match crate::value::Map::new(&self.heap, 7) { Ok(map) => map, Err(error) => return error };
                let method = match self.string_value("sampled") { Ok(value) => value, Err(error) => return error };
                let procedure = match self.string_value("bootstrap") { Ok(value) => value, Err(error) => return error };
                for (key, value) in [
                    ("estimate", Value::number(estimate)), ("ci_low", Value::number(ci_low)),
                    ("ci_high", Value::number(ci_high)), ("method", method), ("procedure", procedure),
                    ("sample_count", Value::number(b as f64)), ("seed", Value::number(self.root_seed as f64)),
                ] {
                    if let Err(error) = map.set(Arc::from(key), value, false) { return error; }
                }
                self.current_frame_mut().registers[ins.a as usize] =
                    Value::map(Arc::new(Mutex::new(map)));
                LanaError::Ok
            }
            Return => {
                let returned = self.current_frame().registers[ins.a as usize].clone();
                if let Err(error) = self.check_object_return(&returned) { return error; }
                if self.constructions.last().is_some_and(|construction| construction.caller_depth + 1 == self.frames.len()) {
                    self.frames.pop();
                    return self.resume_construction(Some(returned)).err().unwrap_or(LanaError::Ok);
                }
                if self.current_frame().is_async {
                    let future_value = self.current_frame().registers[0].clone();
                    let ValueKind::Future(future) = future_value.kind else {
                        return LanaError::Type;
                    };
                    {
                        self.heap.mutate_cycle(Arc::as_ptr(&future) as usize);
                        let mut future = future.lock().unwrap();
                        future.exhausted = true;
                        future.ready = false;
                        future.queued = false;
                        future.registers[0] = returned;
                    }
                    if let Some(awaiters) = self.awaiters.remove(&(Arc::as_ptr(&future) as usize)) {
                        for awaiter in awaiters {
                            self.enqueue_future(awaiter);
                        }
                    }
                    self.frames.pop();
                } else if self.current_frame().is_generator {
                    let gen_value = self.current_frame().registers[0].clone();
                    let ValueKind::Generator(generator) = gen_value.kind else {
                        return LanaError::Type;
                    };
                    self.heap.mutate_cycle(Arc::as_ptr(&generator) as usize);
                    generator.lock().unwrap().exhausted = true;
                    if self.frames.len() == 1 {
                        return LanaError::Type;
                    }
                    let return_ip = self.current_frame().return_ip;
                    let destination = self.current_frame().return_register;
                    self.frames.pop();
                    let result = match self.make_result(false, Value::string(Arc::from("exhausted"))) { Ok(value) => value, Err(error) => return error };
                    self.current_frame_mut().registers[destination as usize] = result;
                    self.ip = return_ip;
                } else if self.frames.len() == 1 {
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
                if self.active_path_count > 1 { return LanaError::UnsupportedOperation; }
                let value = self.current_frame().registers[ins.a as usize].clone();
                if let Err(error) = self.check_resolved(&value) { return error; }
                let rendered = match value.try_print(self.memory_limit.saturating_sub(self.allocated_bytes())) {
                    Ok(text) => text,
                    Err(error) => return error,
                };
                if let Some(output) = &self.captured_output {
                    let mut output = output.lock().unwrap();
                    if output.len().saturating_add(rendered.len()).saturating_add(1) > 64 * 1024 * 1024 {
                        return LanaError::Limit;
                    }
                    output.push_str(&rendered);
                    output.push('\n');
                } else {
                    println!("{}", rendered);
                }
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
                let adt = match self.managed_payload(Adt { variant, fields }) { Ok(payload) => payload, Err(error) => return error };
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
                let array = match self.array_value(items.into_iter().map(Value::state).collect()) {
                    Ok(array) => array,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.a as usize] = array;
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
                let expected = match self.state_dist_expected_probability(&distribution) {
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
            JointConditionMap => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let evidence = self.current_frame().registers[ins.c as usize].clone();
                let result = match self.core_condition(&source, &evidence) {
                    Ok(result) => result,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = result;
                let inputs = [&source, &evidence];
                self.attach_derivation(ins.b, DerivationKind::Operation, "condition", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "evidence_map")
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
                let items = array.lock().unwrap().items.to_vec();
                let possibility = match self.possibility_build(&items) {
                    Ok(possibility) => possibility,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = Value::possibility(possibility);
                let inputs = [&source];
                self.attach_derivation(ins.b, DerivationKind::Operation, "possibility", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "equipossible_support")
            }
            DistributionBuild => {
                let source = self.current_frame().registers[ins.a as usize].clone();
                let ValueKind::Array(array) = &source.kind else {
                    return LanaError::Type;
                };
                let items = array.lock().unwrap().items.to_vec();
                let distribution = match self.distribution_build(&items) {
                    Ok(distribution) => distribution,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = Value::possibility(distribution);
                let inputs = [&source];
                self.attach_derivation(ins.b, DerivationKind::Operation, "distribution", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "finite_weighted_support")
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
                if source.reactive.is_some() {
                    let result = match self.reactive_observe(&source, &evidence, ins.b) {
                        Ok(result) => result,
                        Err(error) => return error,
                    };
                    self.current_frame_mut().registers[ins.b as usize] = result;
                    let inputs = [&source, &evidence];
                    self.attach_derivation(ins.b, DerivationKind::Observation, "observe", &inputs, "",
                                           ins.line, DerivationExactness::Exact, &descriptor)
                } else {
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
            }
            ObserveMap => {
                if self.active_path_count > 1 {
                    return LanaError::UnsupportedOperation;
                }
                let source = self.current_frame().registers[ins.a as usize].clone();
                let evidence = self.current_frame().registers[ins.c as usize].clone();
                let result = if source.reactive.is_some() {
                    match self.reactive_observe(&source, &evidence, ins.b) {
                        Ok(result) => result,
                        Err(error) => return error,
                    }
                } else {
                    let result = match self.core_condition(&source, &evidence) {
                        Ok(result) => result,
                        Err(error) => return error,
                    };
                    self.observation_count += 1;
                    self.revision += 1;
                    result
                };
                self.current_frame_mut().registers[ins.b as usize] = result;
                let inputs = [&source, &evidence];
                self.attach_derivation(ins.b, DerivationKind::Observation, "observe", &inputs, "",
                                       ins.line, DerivationExactness::Exact, "evidence_map")
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
                    if let Err(error) = self.check_resolved(&self.current_frame().registers[(ins.c as usize) + argument]) { return error; }
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
                let array = match self.array_value(results) {
                    Ok(array) => array,
                    Err(error) => return error,
                };
                self.current_frame_mut().registers[ins.b as usize] = array;
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
                let core_measure = matches!(host_id, LANA_HOST_CORE_ENTROPY
                    | LANA_HOST_CORE_CONDITIONAL_ENTROPY | LANA_HOST_CORE_MUTUAL_INFORMATION
                    | LANA_HOST_CORE_BROJA
                    | LANA_HOST_CORE_FORGET_WEIGHTS | LANA_HOST_CORE_ASSIGN_WEIGHTS);
                let accepts_unresolved = (LANA_HOST_DATASET..=LANA_HOST_DATASET_EXPLAIN).contains(&host_id) || matches!(
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
                        | LANA_HOST_INFORMATION_SNAPSHOT
                );
                let materialize = matches!(
                    host_id,
                    LANA_HOST_WRITE_TEXT
                        | LANA_HOST_JSON_STRINGIFY
                        | LANA_HOST_CSV_WRITE
                        | LANA_HOST_ASSERT
                );
                if self.active_path_count > 1 && host_id != LANA_HOST_INFORMATION_SNAPSHOT {
                    return LanaError::UnsupportedOperation;
                }
                let argc = ins.imm as usize;
                for argument in 0..argc {
                    if !accepts_unresolved && !(host_id == LANA_HOST_DATASET_APPLY && argument == 3)
                        && !((core_measure || host_id == LANA_HOST_CORE_NETWORK) && argument == 0) {
                        if let Err(error) = self.check_resolved(&self.current_frame().registers[ins.c as usize + argument]) { return error; }
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
                let error = if host_id == LANA_HOST_GRAD {
                    if argc != 2 {
                        LanaError::Type
                    } else {
                        self.ad_grad(&arguments[0], &arguments[1], ins.a, &mut out)
                    }
                } else if host_id == LANA_HOST_VJP {
                    if argc != 3 {
                        LanaError::Type
                    } else {
                        self.ad_vjp(&arguments[0], &arguments[1], &arguments[2], ins.a, &mut out)
                    }
                } else if host_id == LANA_HOST_TRAIN {
                    match self.host_train(&arguments, ins.a, &mut out) {
                        Ok(()) => LanaError::Ok,
                        Err(error) => error,
                    }
                } else if host_id == LANA_HOST_INFER {
                    match self.host_infer(&arguments, ins.a, &mut out) {
                        Ok(()) => LanaError::Ok,
                        Err(error) => error,
                    }
                } else if host_id == LANA_HOST_UPDATE {
                    match self.host_update(&arguments, ins.a, &mut out) {
                        Ok(()) => LanaError::Ok,
                        Err(error) => error,
                    }
                } else if host_id == LANA_HOST_RESUME {
                    match self.host_resume(&arguments, ins.a, &mut out) {
                        Ok(()) => LanaError::Ok,
                        Err(error) => error,
                    }
                } else if host_id == LANA_HOST_CORE_KERNEL {
                    self.core_kernel(&arguments, ins.a, &mut out)
                } else if host_id == LANA_HOST_EVALUATION_WALK_FORWARD {
                    self.evaluation_walk_forward(&arguments, ins.a, &mut out)
                } else {
                    self.execute_host_call(host_id, &arguments, &mut out)
                };
                if error == LanaError::Assertion
                    && host_id == LANA_HOST_ASSERT
                    && argc == 2
                    && matches!(arguments[1].kind, ValueKind::String(_))
                {
                    self.pending_error_message = Some(arguments[1].as_string().to_string());
                }
                if error == LanaError::Ok {
                    self.current_frame_mut().registers[ins.a as usize] = out;
                    if host_id == LANA_HOST_GPU_MATMUL {
                        let inputs = [&arguments[0], &arguments[1]];
                        let e = self.attach_derivation(
                            ins.a,
                            DerivationKind::Approximation,
                            "gpu_matmul",
                            &inputs,
                            "",
                            ins.line,
                            DerivationExactness::Approximate,
                            "backend=metal precision=float32",
                        );
                        if e != LanaError::Ok {
                            return e;
                        }
                    }
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
        if bytes > self.memory_limit.saturating_sub(self.allocated_bytes()) {
            return LanaError::Oom;
        }
        self.allocation_count = self.allocation_count.saturating_add(1);
        self.allocated_bytes += bytes;
        let _ = self.heap.set_limit(self.memory_limit - self.allocated_bytes);
        LanaError::Ok
    }

    fn allocate_array_items(&mut self, count: usize) -> Result<crate::heap::Buffer<Value>, LanaError> {
        crate::heap::Buffer::new(&self.heap, count, std::mem::size_of::<Array>())
    }

    fn allocate_with_collection<T>(&mut self, mut build: impl FnMut(&mut Self) -> Result<T, LanaError>) -> Result<T, LanaError> {
        match build(self) {
            Err(LanaError::Oom) if self.constructions.is_empty() => {
                self.collect_classes()?;
                build(self)
            }
            result => result,
        }
    }

    fn string_value(&self, text: &str) -> Result<Value, LanaError> {
        Ok(Value::string(self.heap.string(text)?))
    }

    fn clone_string(&self, string: &Arc<str>, memo: &mut DeepCloneMemo) -> Result<Arc<str>, LanaError> {
        let key = Arc::as_ptr(string) as *const () as usize;
        if let Some(copy) = memo.strings.get(&key) { return Ok(copy.clone()); }
        memo.strings.try_reserve(1).map_err(|_| LanaError::Oom)?;
        let copy = self.heap.string(string)?;
        memo.strings.insert(key, copy.clone());
        Ok(copy)
    }

    fn string_output(&self, text: &str, out: &mut Value) -> LanaError {
        match self.string_value(text) {
            Ok(value) => { *out = value; LanaError::Ok }
            Err(error) => { *out = Value::null(); error }
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
        self.managed_payload(StateDist {
            kind: StateDistKind::Dirac(state.clone()),
        })
    }

    /// Build an append node, mirroring `lana_vm_state_dist_append`.
    fn state_dist_append(&mut self, left: &Value, right: &Value) -> Result<Arc<StateDist>, LanaError> {
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
        self.managed_payload(StateDist {
            kind: StateDistKind::Append {
                left: left_operand,
                right: right_operand,
                has_cached_parameters,
                p,
                m_re,
                m_im,
                sigma,
            },
        })
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
        self.managed_payload(StateDist {
            kind: StateDistKind::Transform { child, transform_id },
        })
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
        self.managed_payload(StateDist {
            kind: StateDistKind::Attenuate { child, factor },
        })
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
        self.managed_payload(StateDist {
            kind: StateDistKind::Append {
                left: left_operand,
                right: right_operand,
                has_cached_parameters: true,
                p,
                m_re,
                m_im,
                sigma,
            },
        })
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

    /// The expected probability of a state distribution, mirroring
    /// `lana_vm_state_dist_expected_probability`. Iterative to bound the stack;
    /// each visited node is charged against the sampling budget so a DAG that
    /// shares subtrees exponentially cannot make one evaluation unbounded.
    fn state_dist_expected_probability(
        &mut self,
        distribution: &Arc<StateDist>,
    ) -> Result<f64, LanaError> {
        let mut stack: Vec<DistEvalFrame> = Vec::new();
        let mut result = 0.0;
        stack.push(DistEvalFrame::new(distribution.clone()));
        loop {
            if stack.is_empty() {
                break;
            }
            let error = self.consume_sampling_budget();
            if error != LanaError::Ok {
                return Err(error);
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
                            result = state.state.p;
                            EvalAction::Pop
                        }
                        StateDistKind::Append { left, has_cached_parameters, p, .. } => {
                            if *has_cached_parameters {
                                result = *p;
                                EvalAction::Pop
                            } else if let DistOperand::Inline(state) = left {
                                if !state::state_valid(&state.state) {
                                    return Err(LanaError::InvalidDistribution);
                                }
                                frame.left = state.state.p;
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
                stack[top].left = result;
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
                        frame.right = state.state.p;
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
                    stack[top].right = result;
                }
                result = 1.0 - (1.0 - stack[top].left) * (1.0 - stack[top].right);
                if !result.is_finite() || result < 0.0 || result > 1.0 {
                    return Err(LanaError::InvalidDistribution);
                }
                stack.pop();
            } else if stage == 4 {
                let transform_id = match &stack[top].node.kind {
                    StateDistKind::Transform { transform_id, .. } => *transform_id,
                    _ => return Err(LanaError::InvalidDistribution),
                };
                let mut out = 0.0;
                let error = state::transform_expected_probability(transform_id, result, &mut out);
                if error != LanaError::Ok {
                    return Err(error);
                }
                result = out;
                stack.pop();
            } else {
                // ATTENUATE is the identity on probability: the child's result is
                // already the expected probability.
                stack.pop();
            }
        }
        Ok(result)
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
            let error = self.consume_sampling_budget();
            if error != LanaError::Ok {
                return Err(error);
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
        let mut map = Map::new(&self.heap, 6)?;
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
        let mut map = Map::new(&self.heap, 3)?;
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

    /// Import validated historical evidence with a fresh VM-local identity.
    pub fn import_derivation(&mut self, mut node: Derivation) -> Result<Arc<Derivation>, LanaError> {
        self.charge_bounded_work(1)?;
        let mut memo = DeepCloneMemo::default();
        node.operation = self.clone_string(&node.operation, &mut memo)?;
        node.label = self.clone_string(&node.label, &mut memo)?;
        node.function = self.clone_string(&node.function, &mut memo)?;
        node.details = self.clone_string(&node.details, &mut memo)?;
        node.reason = self.clone_string(&node.reason, &mut memo)?;
        self.derivation_sequence += 1;
        node.task_lineage = self.lineage;
        node.local_sequence = self.derivation_sequence;
        self.managed_payload(node)
    }

    /// Capture current Information without resolution or a live dependency.
    /// Copy and freeze containers; reject effectful or executable payloads.
    pub fn information_snapshot(&mut self, source: &Value) -> Result<Value, LanaError> {
        self.deep_clone_value(source, &mut DeepCloneMemo { freeze: true, ..DeepCloneMemo::default() })
    }

    /// Copy a host-owned snapshot into this heap, preserving aliases and cycles.
    /// Live Information becomes a snapshot at this ownership boundary.
    pub fn import_value(&mut self, source: &Value) -> Result<Value, LanaError> {
        let checkpoint = self.class_objects.len();
        let result = self.deep_clone_value(source, &mut DeepCloneMemo { transfer: true, ..DeepCloneMemo::default() });
        if result.is_err() { self.class_objects.truncate(checkpoint); }
        result
    }

    fn deep_clone_value(&mut self, value: &Value, memo: &mut DeepCloneMemo) -> Result<Value, LanaError> {
        if memo.transfer && value.reactive.is_some() && !memo.freeze {
            let mut capture = DeepCloneMemo { freeze: true, transfer: true, depth: memo.depth,
                classes: std::mem::take(&mut memo.classes),
                derivations: std::mem::take(&mut memo.derivations),
                distributions: std::mem::take(&mut memo.distributions),
                strings: std::mem::take(&mut memo.strings), ..DeepCloneMemo::default() };
            let result = self.deep_clone_value(value, &mut capture);
            memo.classes = capture.classes;
            memo.derivations = capture.derivations;
            memo.distributions = capture.distributions;
            memo.strings = capture.strings;
            return result;
        }
        if memo.transfer {
            if memo.depth >= 64 { return Err(LanaError::Limit); }
            self.charge_bounded_work(1)?;
            memo.depth += 1;
        }
        let result = self.deep_clone_value_checked(value, memo);
        if memo.transfer { memo.depth -= 1; }
        result
    }

    fn deep_clone_value_checked(&mut self, value: &Value, memo: &mut DeepCloneMemo) -> Result<Value, LanaError> {
        if !memo.freeze { return self.deep_clone_value_inner(value, memo); }
        if memo.depth >= 64 { return Err(LanaError::Limit); }
        self.charge_bounded_work(1)?;
        if value.claim.is_some() || value.planned_effect.is_some() { return Err(LanaError::UnsupportedValue); }
        let (mut current, captured_revision) = if let Some(reactive) = &value.reactive {
            let reactive = reactive.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
            (reactive.current.clone().unwrap_or_else(|| value.clone()), Some(reactive.revision))
        } else { (value.clone(), None) };
        if !matches!(current.kind, ValueKind::Null | ValueKind::Number(_) | ValueKind::Bool(_)
            | ValueKind::String(_) | ValueKind::State(_) | ValueKind::StateDist(_) | ValueKind::Tensor(_)
            | ValueKind::Array(_) | ValueKind::Map(_) | ValueKind::Possibility(_)
            | ValueKind::Joint(_) | ValueKind::PathSet(_) | ValueKind::Adt(_) | ValueKind::ObjectValue(_)) && !(memo.transfer && matches!(current.kind, ValueKind::ClassObject(_))) {
            return Err(LanaError::UnsupportedValue);
        }
        current.reactive = None;
        memo.depth += 1;
        let mut result = self.deep_clone_value_inner(&current, memo);
        memo.depth -= 1;
        if memo.depth == 0 || value.reactive.is_some() {
            if let Ok(captured) = &mut result {
                let prior = current.derivation.as_ref().or(value.derivation.as_ref());
                let revision = captured_revision.or_else(|| prior.map(|node| node.revision)).unwrap_or(0);
                let mut original_input = Value::null();
                let mut current_input = Value::null();
                if memo.transfer {
                    original_input.derivation = value.derivation.as_ref()
                        .map(|node| self.deep_clone_derivation(node, memo)).transpose()?;
                    current_input.derivation = captured.derivation.clone();
                }
                let inputs = if memo.transfer { [&original_input, &current_input] } else { [value, &current] };
                let mut derivation = self.record_derivation_payload(DerivationKind::Operation, "snapshot",
                    &inputs, "", 0,
                    prior.map_or(DerivationExactness::Exact, |node| node.exactness), "immutable_capture",
                    prior.map_or(DerivationOutcome::Success, |node| node.outcome), "none").ok_or(LanaError::Oom)?;
                derivation.revision = revision;
                let derivation = self.managed_payload(derivation)?;
                captured.derivation = Some(derivation);
            }
        }
        result
    }

    fn deep_clone_state(&self, state: &StateValue, memo: &mut DeepCloneMemo) -> Result<StateValue, LanaError> {
        let mut copy = state.clone();
        if memo.transfer {
            copy.indexes.source = state.indexes.source.as_ref().map(|source| self.clone_string(source, memo)).transpose()?;
        }
        Ok(copy)
    }

    fn deep_clone_value_inner(&mut self, value: &Value, memo: &mut DeepCloneMemo) -> Result<Value, LanaError> {
        let mut cloned = Value {
            kind: ValueKind::Null,
            derivation: if memo.transfer { value.derivation.as_ref().map(|node| self.deep_clone_derivation(node, memo)).transpose()? } else { value.derivation.clone() },
            reactive: value.reactive.clone(),
            claim: value.claim.clone(),
            planned_effect: value.planned_effect.clone(),
        };
        if memo.transfer {
            if let Some(claim) = &value.claim {
                let key = Arc::as_ptr(claim) as usize;
                cloned.claim = Some(if let Some(copy) = memo.claims.get(&key) { copy.clone() }
                    else {
                        let claimed = self.deep_clone_value(&claim.value, memo)?;
                        let copy = self.managed_payload(Claim {
                            value: claimed, proposition: claim.proposition.clone(), exactness: claim.exactness,
                            tolerance: claim.tolerance, source_valid: claim.source_valid,
                        })?;
                        memo.claims.insert(key, copy.clone());
                        copy
                    });
            }
            if let Some(effect) = &value.planned_effect {
                let key = Arc::as_ptr(effect) as usize;
                cloned.planned_effect = Some(if let Some(copy) = memo.effects.get(&key) { copy.clone() }
                    else {
                        let payload = self.deep_clone_value(&effect.payload, memo)?;
                        let source = effect.state.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
                        let copy = Arc::new(PlannedEffect {
                            id: effect.id, kind: effect.kind.clone(), payload,
                            state: Mutex::new(PlannedEffectState {
                                receipts: vec![EffectReceipt { revision: 0, result: Value::null() }; source.receipts.len()],
                                execution_count: source.execution_count,
                            }),
                        });
                        self.track_cycle(crate::heap::CycleWeak::Effect(Arc::downgrade(&copy)))?;
                        memo.effects.insert(key, copy.clone());
                        for (index, receipt) in source.receipts.iter().enumerate() {
                            let result = self.deep_clone_value(&receipt.result, memo)?;
                            copy.state.lock().unwrap().receipts[index] = EffectReceipt { revision: receipt.revision, result };
                        }
                        copy
                    });
            }
        }
        match &value.kind {
            ValueKind::ClassObject(object) => { cloned.kind = ValueKind::ClassObject(self.clone_class_reference(object, memo)?); }
            ValueKind::Generator(generator) if memo.transfer => {
                let key = Arc::as_ptr(generator) as usize;
                if let Some(copy) = memo.generators.get(&key) {
                    cloned.kind = ValueKind::Generator(copy.clone());
                    return Ok(cloned);
                }
                let source = generator.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
                let copy = Arc::new(Mutex::new(Generator {
                    function: source.function, ip: source.ip,
                    registers: vec![Value::null(); source.registers.len()], exhausted: source.exhausted,
                }));
                self.track_cycle(crate::heap::CycleWeak::Generator(Arc::downgrade(&copy)))?;
                memo.generators.insert(key, copy.clone());
                for (index, register) in source.registers.iter().enumerate() {
                    if index == 0 && !source.exhausted { continue; }
                    let register = self.deep_clone_value(register, memo)?;
                    copy.lock().unwrap().registers[index] = register;
                }
                cloned.kind = ValueKind::Generator(copy);
            }
            ValueKind::Future(future) if memo.transfer => {
                let key = Arc::as_ptr(future) as usize;
                if let Some(copy) = memo.futures.get(&key) {
                    cloned.kind = ValueKind::Future(copy.clone());
                    return Ok(cloned);
                }
                let source = future.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
                let copy = Arc::new(Mutex::new(Future {
                    function: source.function, ip: source.ip,
                    registers: vec![Value::null(); source.registers.len()], exhausted: source.exhausted,
                    ready: source.ready, queued: false,
                }));
                self.track_cycle(crate::heap::CycleWeak::Future(Arc::downgrade(&copy)))?;
                memo.futures.insert(key, copy.clone());
                for (index, register) in source.registers.iter().enumerate() {
                    if index == 0 && !source.exhausted && source.function != u32::MAX { continue; }
                    let register = self.deep_clone_value(register, memo)?;
                    copy.lock().unwrap().registers[index] = register;
                }
                if source.function == u32::MAX && !source.exhausted {
                    let registers = copy.lock().unwrap().registers.clone();
                    for register in registers.iter().skip(1) {
                        if let ValueKind::Future(dependency) = &register.kind {
                            self.awaiters.entry(Arc::as_ptr(dependency) as usize).or_default().push(copy.clone());
                            self.enqueue_future(dependency.clone());
                        }
                    }
                } else if !source.ready && !source.exhausted {
                    let instruction = self.chunk.code.get(source.ip).ok_or(LanaError::Task)?;
                    if instruction.opcode != OpCode::Await { return Err(LanaError::Task); }
                    let dependency = copy.lock().unwrap().registers.get(instruction.a as usize)
                        .cloned().ok_or(LanaError::Task)?;
                    let ValueKind::Future(dependency) = dependency.kind else { return Err(LanaError::Task); };
                    if dependency.lock().unwrap().exhausted { copy.lock().unwrap().ready = true; }
                    else {
                        self.awaiters.entry(Arc::as_ptr(&dependency) as usize).or_default().push(copy.clone());
                        self.enqueue_future(dependency);
                    }
                }
                cloned.kind = ValueKind::Future(copy);
            }
            ValueKind::TrainingResult(training) if memo.transfer => {
                let ValueKind::Array(steps) = self.deep_clone_value(&Value::array(training.steps.clone()), memo)?.kind else { return Err(LanaError::Type); };
                let data = self.deep_clone_value(&training.data, memo)?;
                cloned.kind = ValueKind::TrainingResult(self.managed_payload(TrainingResult {
                    params: training.params.clone(), steps, model_function: training.model_function,
                    loss_function: training.loss_function, optimizer: training.optimizer.clone(),
                    data, batch_size: training.batch_size,
                })?);
            }
            ValueKind::Posterior(posterior) if memo.transfer => {
                let ValueKind::Array(steps) = self.deep_clone_value(&Value::array(posterior.steps.clone()), memo)?.kind else { return Err(LanaError::Type); };
                cloned.kind = ValueKind::Posterior(self.managed_payload(Posterior {
                    mean: posterior.mean.clone(), variance: posterior.variance.clone(), samples: posterior.samples.clone(),
                    steps, seed: posterior.seed,
                })?);
            }
            ValueKind::Dataset(dataset) if memo.transfer => {
                let mut copy = (**dataset).clone();
                copy.source = self.deep_clone_value(&dataset.source, memo)?;
                copy.columns = self.deep_clone_value(&dataset.columns, memo)?;
                copy.key = self.deep_clone_value(&dataset.key, memo)?;
                copy.limit = self.deep_clone_value(&dataset.limit, memo)?;
                copy.other = self.deep_clone_value(&dataset.other, memo)?;
                copy.aggregate = self.deep_clone_value(&dataset.aggregate, memo)?;
                cloned.kind = ValueKind::Dataset(self.managed_payload(copy)?);
            }
            ValueKind::ObjectValue(object) if memo.transfer || memo.freeze => {
                let mut fields = Vec::with_capacity(object.fields.len());
                for field in &object.fields { fields.push(self.deep_clone_value(field, memo)?); }
                cloned.kind = ValueKind::ObjectValue(self.managed_payload(ObjectValue { descriptor: object.descriptor.clone(), fields })?);
            }
            ValueKind::Null
            | ValueKind::Number(_)
            | ValueKind::Bool(_)
            | ValueKind::Sample(_)
            | ValueKind::Function(_)
            | ValueKind::Distribution { .. }
            | ValueKind::Capability(_)
            | ValueKind::Tensor(_)
            | ValueKind::NQubitState(_)
            | ValueKind::Povm(_)
            | ValueKind::Channel(_)
            | ValueKind::Observable(_)
            | ValueKind::Lazy { .. }
            | ValueKind::Generator(_)
            | ValueKind::Future(_)
            | ValueKind::Regex(_)
            | ValueKind::Optimizer(_)
            | ValueKind::TrainingResult(_)
            | ValueKind::InferenceAlgorithm(_)
            | ValueKind::Posterior(_)
            | ValueKind::Dataset(_)
            | ValueKind::ObjectValue(_) => {
                cloned.kind = value.kind.clone();
            }
            ValueKind::String(string) => cloned.kind = ValueKind::String(self.clone_string(string, memo)?),
            ValueKind::State(state) => cloned.kind = ValueKind::State(self.deep_clone_state(state, memo)?),
            ValueKind::Array(array) => {
                let key = Arc::as_ptr(array) as usize;
                if let Some(existing) = memo.arrays.get(&key) {
                    cloned.kind = ValueKind::Array(existing.clone());
                } else {
                    let array = array.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
                    let count = array.items.len();
                    let items = self.allocate_array_items(count)?;
                    let copy = Arc::new(Mutex::new(Array::from_buffer(items)?));
                    let _ = Value::array(copy.clone());
                    memo.arrays.insert(key, copy.clone());
                    for index in 0..count {
                        let item = array.items[index].clone();
                        let item = self.deep_clone_value(&item, memo)?;
                        copy.lock().unwrap().items.push(item)?;
                    }
                    copy.lock().unwrap().frozen = memo.freeze || array.frozen;
                    cloned.kind = ValueKind::Array(copy);
                }
            }
            ValueKind::Map(map) => {
                let key = Arc::as_ptr(map) as usize;
                if let Some(existing) = memo.maps.get(&key) {
                    cloned.kind = ValueKind::Map(existing.clone());
                } else {
                    let map = map.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
                    let copy = Arc::new(Mutex::new(Map::new(&self.heap, map.entries.len())?));
                    let _ = Value::map(copy.clone());
                    memo.maps.insert(key, copy.clone());
                    let count = map.entries.len();
                    for index in 0..count {
                        let entry = map.entries[index].clone();
                        let value = self.deep_clone_value(&entry.value, memo)?;
                        let key = if memo.transfer { self.clone_string(&entry.key, memo)? } else { entry.key.clone() };
                        copy.lock().unwrap().set(key, value, true)?;
                    }
                    copy.lock().unwrap().frozen = memo.freeze || map.frozen;
                    cloned.kind = ValueKind::Map(copy);
                }
            }
            ValueKind::Joint(joint) => {
                let key = Arc::as_ptr(joint) as usize;
                if let Some(existing) = memo.joints.get(&key) {
                    cloned.kind = ValueKind::Joint(existing.clone());
                    return Ok(cloned);
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
                let copy = self.managed_payload(JointState {
                    names: joint.names.clone(),
                    domains: joint.domains.clone(),
                    values,
                    rows,
                    kind: joint.kind,
                    capabilities: joint.capabilities,
                })?;
                memo.joints.insert(key, copy.clone());
                cloned.kind = ValueKind::Joint(copy);
            }
            ValueKind::Kernel(kernel) => {
                let cells = kernel.rows.iter().try_fold(0usize, |total, row|
                    total.checked_add(row.len())).ok_or(LanaError::Oom)?;
                let _bytes = cells.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|bytes| bytes.checked_add(std::mem::size_of::<FiniteKernel>()))
                    .ok_or(LanaError::Oom)?;
                let mut input_domains = Vec::with_capacity(kernel.input_domains.len());
                for domain in &kernel.input_domains {
                    let mut copied = Vec::with_capacity(domain.len());
                    for value in domain { copied.push(self.deep_clone_value(value, memo)?); }
                    input_domains.push(copied);
                }
                let mut output_domain = Vec::with_capacity(kernel.output_domain.len());
                for value in &kernel.output_domain { output_domain.push(self.deep_clone_value(value, memo)?); }
                cloned.kind = ValueKind::Kernel(self.managed_payload(FiniteKernel {
                    input_domains, output_domain, rows: kernel.rows.clone(),
                })?);
            }
            ValueKind::Network(network) => {
                let ValueKind::Joint(root) = self.deep_clone_value(&Value::joint(network.root.clone()), memo)?.kind
                    else { unreachable!() };
                let mut nodes = Vec::with_capacity(network.nodes.len());
                for node in &network.nodes {
                    let ValueKind::Kernel(kernel) = self.deep_clone_value(&Value::kernel(node.kernel.clone()), memo)?.kind
                        else { unreachable!() };
                    nodes.push(NetworkNode { name: node.name.clone(), parents: node.parents.clone(), kernel });
                }
                cloned.kind = ValueKind::Network(self.managed_payload(FiniteNetwork {
                    root, nodes, order: network.order.clone(),
                })?);
            }
            ValueKind::StateDist(distribution) if memo.freeze && !memo.transfer => {
                cloned.kind = ValueKind::StateDist(distribution.clone());
            }
            ValueKind::StateDist(distribution) => {
                cloned.kind = ValueKind::StateDist(self.deep_clone_state_dist(distribution, memo)?);
            }
            ValueKind::Possibility(possibility) => {
                let mut values = Vec::with_capacity(possibility.values.len());
                for value in &possibility.values {
                    values.push(self.deep_clone_value(value, memo)?);
                }
                cloned.kind = ValueKind::Possibility(self.managed_payload(Possibility {
                    values,
                    weights: possibility.weights.clone(),
                    dependency_id: possibility.dependency_id,
                })?);
            }
            ValueKind::PathSet(paths) => {
                let mut alternatives = Vec::with_capacity(paths.alternatives.len());
                for alternative in &paths.alternatives {
                    alternatives.push(PathAlternative {
                        guard: alternative.guard,
                        weight: alternative.weight,
                        result: self.deep_clone_value(&alternative.result, memo)?,
                    });
                }
                cloned.kind = ValueKind::PathSet(self.managed_payload(PathSet {
                    alternatives,
                    dependency_id: paths.dependency_id,
                })?);
            }
            ValueKind::Adt(adt) => {
                let mut fields = Vec::with_capacity(adt.fields.len());
                for field in &adt.fields {
                    fields.push(self.deep_clone_value(field, memo)?);
                }
                cloned.kind = ValueKind::Adt(self.managed_payload(Adt { variant: adt.variant, fields })?);
            }
            ValueKind::Set(set) => {
                let key = Arc::as_ptr(set) as usize;
                if let Some(existing) = memo.sets.get(&key) {
                    cloned.kind = ValueKind::Set(existing.clone());
                } else {
                    let set = set.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
                    let count = set.items.len();
                    let copy = Arc::new(Mutex::new(Set::new(&self.heap, count)?));
                    let _ = Value::set(copy.clone());
                    memo.sets.insert(key, copy.clone());
                    for index in 0..count {
                        let item = set.items[index].clone();
                        let item = self.deep_clone_value(&item, memo)?;
                        copy.lock().unwrap().items.push(item)?;
                    }
                    cloned.kind = ValueKind::Set(copy);
                }
            }
            ValueKind::Task(_) => return Err(LanaError::Type),
        }
        Ok(cloned)
    }

    fn deep_clone_derivation(
        &mut self,
        source: &Arc<Derivation>,
        memo: &mut DeepCloneMemo,
    ) -> Result<Arc<Derivation>, LanaError> {
        let root = Arc::as_ptr(source) as usize;
        if let Some(copy) = memo.derivations.get(&root) { return Ok(copy.clone()); }
        let mut pending = crate::heap::Buffer::new(&self.heap, 0, 0)?;
        pending.push((source.clone(), false))?;
        while let Some((node, expanded)) = pending.pop() {
            self.charge_bounded_work(1)?;
            let key = Arc::as_ptr(&node) as usize;
            if memo.derivations.contains_key(&key) { continue; }
            if !expanded {
                pending.push((node.clone(), true))?;
                for child in node.inputs.iter().chain(node.ad_a_deriv.iter()).chain(node.ad_b_deriv.iter()) {
                    pending.push((child.clone(), false))?;
                }
            } else {
                self.deep_clone_derivation_node(&node, memo)?;
            }
        }
        Ok(memo.derivations.get(&root).unwrap().clone())
    }

    /// Preserve provenance identity while isolating task-local graph and gradients.
    fn deep_clone_derivation_node(
        &mut self,
        source: &Arc<Derivation>,
        memo: &mut DeepCloneMemo,
    ) -> Result<Arc<Derivation>, LanaError> {
        let key = Arc::as_ptr(source) as usize;
        if let Some(copy) = memo.derivations.get(&key) { return Ok(copy.clone()); }
        if memo.depth >= 64 { return Err(LanaError::Limit); }
        self.charge_bounded_work(1)?;
        memo.depth += 1;
        let result = (|| {
            let mut copy = (**source).clone();
            copy.inputs = Vec::new();
            copy.inputs.try_reserve_exact(source.inputs.len()).map_err(|_| LanaError::Oom)?;
            for input in &source.inputs { copy.inputs.push(self.deep_clone_derivation(input, memo)?); }
            copy.ad_a_deriv = source.ad_a_deriv.as_ref().map(|node| self.deep_clone_derivation(node, memo)).transpose()?;
            copy.ad_b_deriv = source.ad_b_deriv.as_ref().map(|node| self.deep_clone_derivation(node, memo)).transpose()?;
            copy.operation = self.clone_string(&source.operation, memo)?;
            copy.label = self.clone_string(&source.label, memo)?;
            copy.function = self.clone_string(&source.function, memo)?;
            copy.details = self.clone_string(&source.details, memo)?;
            copy.reason = self.clone_string(&source.reason, memo)?;
            let gradient = source.ad_grad.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
            let gradient = gradient.as_ref().map(|tensor| {
                self.charge_bounded_work(tensor.data.len() as u64)?;
                let mut copy = tensor.try_clone(&self.heap)?;
                copy.data = Arc::new(crate::heap::Buffer::from_slice(&self.heap, &tensor.data)?);
                copy.metal_buffer = None;
                copy.metal_charge = None;
                Ok::<_, LanaError>(copy)
            }).transpose()?;
            copy.ad_grad = Arc::new(Mutex::new(gradient));
            self.managed_payload(copy)
        })();
        memo.depth -= 1;
        let copy = result?;
        memo.derivations.try_reserve(1).map_err(|_| LanaError::Oom)?;
        memo.derivations.insert(key, copy.clone());
        Ok(copy)
    }

    /// Copy a distribution DAG with charged work and a heap-owned traversal stack.
    fn deep_clone_state_dist(
        &mut self,
        distribution: &Arc<StateDist>,
        memo: &mut DeepCloneMemo,
    ) -> Result<Arc<StateDist>, LanaError> {
        let root = Arc::as_ptr(distribution) as usize;
        if let Some(copy) = memo.distributions.get(&root) { return Ok(copy.clone()); }
        let mut pending = crate::heap::Buffer::new(&self.heap, 0, 0)?;
        pending.push((distribution.clone(), false))?;
        while let Some((node, expanded)) = pending.pop() {
            self.charge_bounded_work(1)?;
            let key = Arc::as_ptr(&node) as usize;
            if memo.distributions.contains_key(&key) { continue; }
            if !expanded {
                pending.push((node.clone(), true))?;
                match &node.kind {
                    StateDistKind::Append { left, right, .. } => {
                        for operand in [right, left] {
                            if let DistOperand::Node(child) = operand { pending.push((child.clone(), false))?; }
                        }
                    }
                    StateDistKind::Transform { child, .. } | StateDistKind::Attenuate { child, .. } => pending.push((child.clone(), false))?,
                    StateDistKind::Dirac(_) => {},
                }
                continue;
            }
            let kind = self.deep_clone_state_dist_kind(&node.kind, memo)?;
            let copy = self.managed_payload(StateDist { kind })?;
            memo.distributions.try_reserve(1).map_err(|_| LanaError::Oom)?;
            memo.distributions.insert(key, copy);
        }
        Ok(memo.distributions.get(&root).unwrap().clone())
    }

    /// Clone edges through the shared distribution memo.
    fn deep_clone_state_dist_kind(
        &mut self,
        kind: &StateDistKind,
        memo: &mut DeepCloneMemo,
    ) -> Result<StateDistKind, LanaError> {
        match kind {
            StateDistKind::Dirac(state) => Ok(StateDistKind::Dirac(self.deep_clone_state(state, memo)?)),
            StateDistKind::Append { left, right, has_cached_parameters, p, m_re, m_im, sigma } => {
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
                Ok(StateDistKind::Transform {
                    child: self.deep_clone_state_dist(child, memo)?,
                    transform_id: *transform_id,
                })
            }
            StateDistKind::Attenuate { child, factor } => {
                Ok(StateDistKind::Attenuate {
                    child: self.deep_clone_state_dist(child, memo)?,
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
            DistOperand::Inline(state) => Ok(DistOperand::Inline(self.deep_clone_state(state, memo)?)),
            DistOperand::Node(node) => Ok(DistOperand::Node(self.deep_clone_state_dist(node, memo)?)),
        }
    }

    /// Snapshot register bindings without changing shared heap identity.
    /// Guarded execution rejects heap mutation and suspension.
    fn snapshot_frames(&mut self) -> Result<Vec<Frame>, LanaError> {
        Ok(self.frames.clone())
    }

    /// Split execution on a condition, mirroring `path_split` in `vm/c/vm.c`.
    fn path_split(&mut self, condition: &Value, false_ip: usize) -> LanaError {
        if matches!(condition.kind, ValueKind::Bool(_)) {
            if !condition.as_bool() {
                self.ip = false_ip;
            }
            self.path_execution.push(PathExecution::default());
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
            self.path_execution.push(PathExecution::default());
            return LanaError::Ok;
        }
        if !has_false {
            self.path_execution.push(PathExecution::default());
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
            split: true,
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
        if !execution.split {
            self.path_execution.pop();
            return LanaError::Ok;
        }
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
                let paths = match self.managed_payload(PathSet {
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
                }) { Ok(payload) => payload, Err(error) => return error };
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

    fn core_measure_names(joint: &JointState, value: &Value) -> Result<Vec<usize>, LanaError> {
        let ValueKind::Array(array) = &value.kind else { return Err(LanaError::Type); };
        let array = array.lock().unwrap();
        if array.items.is_empty() || array.items.len() > joint.names.len() { return Err(LanaError::InvalidParameters); }
        let mut positions = Vec::with_capacity(array.items.len());
        for item in array.items.iter() {
            let ValueKind::String(name) = &item.kind else { return Err(LanaError::Type); };
            let position = joint_find(joint, name).ok_or(LanaError::InvalidParameters)?;
            if positions.contains(&position) { return Err(LanaError::InvalidParameters); }
            positions.push(position);
        }
        positions.sort_unstable();
        Ok(positions)
    }

    fn core_entropy(&mut self, joint: &JointState, positions: &[usize]) -> Result<f64, LanaError> {
        let mut entropy = 0.0;
        if joint.rows.is_empty() {
            if joint.values.len() != joint.names.len() { return Err(LanaError::UnsupportedOperation); }
            for &position in positions {
                let marginal = &joint.values[position];
                match &marginal.kind {
                    ValueKind::Possibility(possibility) => {
                        let weights = possibility.weights.as_ref().ok_or(LanaError::UnsupportedOperation)?;
                        let count = weights.len();
                        let work = count.checked_mul(count).ok_or(LanaError::Limit)? as u64;
                        self.charge_core_work(work)?;
                        let bytes = count.checked_mul(std::mem::size_of::<(usize, f64)>()).ok_or(LanaError::Oom)?;
                        let _reservation = self.heap.reserve(bytes)?;
                        let mut groups: Vec<(usize, f64)> = Vec::new();
                        groups.try_reserve(count).map_err(|_| LanaError::Oom)?;
                        for (index, &weight) in weights.iter().enumerate() {
                            if !weight.is_finite() || weight < 0.0 { return Err(LanaError::InvalidDistribution); }
                            if let Some(group) = groups.iter_mut().find(|(prior, _)|
                                joint_value_equal(&possibility.values[*prior], &possibility.values[index])) {
                                group.1 += weight;
                            } else { groups.push((index, weight)); }
                        }
                        if (groups.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() > 1e-12 {
                            return Err(LanaError::InvalidDistribution);
                        }
                        entropy += groups.iter().filter(|(_, weight)| *weight > 0.0)
                            .map(|(_, weight)| -weight * weight.log2()).sum::<f64>();
                    }
                    _ if joint_value_is_definite(marginal) => {}
                    _ => return Err(LanaError::UnsupportedOperation),
                }
            }
        } else {
            let count = joint.rows.len();
            let work = count.checked_mul(count).and_then(|n| n.checked_mul(positions.len()))
                .ok_or(LanaError::Limit)? as u64;
            self.charge_core_work(work)?;
            let bytes = count.checked_mul(std::mem::size_of::<(usize, f64)>()).ok_or(LanaError::Oom)?;
            let _reservation = self.heap.reserve(bytes)?;
            let mut groups: Vec<(usize, f64)> = Vec::new();
            groups.try_reserve(count).map_err(|_| LanaError::Oom)?;
            for (index, row) in joint.rows.iter().enumerate() {
                if !row.weight.is_finite() || row.weight < 0.0 { return Err(LanaError::InvalidDistribution); }
                if let Some(group) = groups.iter_mut().find(|(prior, _)| positions.iter().all(|&position|
                    joint_value_equal(&joint.rows[*prior].values[position], &row.values[position]))) {
                    group.1 += row.weight;
                } else { groups.push((index, row.weight)); }
            }
            if (groups.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() > 1e-12 {
                return Err(LanaError::InvalidDistribution);
            }
            entropy = groups.iter().filter(|(_, weight)| *weight > 0.0)
                .map(|(_, weight)| -weight * weight.log2()).sum();
        }
        if !entropy.is_finite() || entropy < -1e-10 { return Err(LanaError::InvalidDistribution); }
        Ok(entropy.max(0.0))
    }

    fn charge_core_work(&mut self, work: u64) -> Result<(), LanaError> {
        if work > self.instruction_limit.saturating_sub(self.instruction_count) { return Err(LanaError::Limit); }
        self.instruction_count += work;
        Ok(())
    }

    fn kernel_domain(&mut self, source: &Value) -> Result<Vec<Value>, LanaError> {
        let ValueKind::Array(array) = &source.kind else { return Err(LanaError::Type); };
        let items = array.lock().unwrap();
        let count = items.items.len();
        if count == 0 { return Err(LanaError::InvalidParameters); }
        self.charge_core_work(count.checked_mul(count).ok_or(LanaError::Limit)? as u64)?;
        let mut domain = Vec::new();
        domain.try_reserve(count).map_err(|_| LanaError::Oom)?;
        let mut memo = DeepCloneMemo::default();
        for (index, candidate) in items.items.iter().enumerate() {
            if !matches!(candidate.kind, ValueKind::Null | ValueKind::Number(_) | ValueKind::Bool(_)
                | ValueKind::String(_) | ValueKind::Sample(_) | ValueKind::State(_)
                | ValueKind::Array(_) | ValueKind::Map(_)) { return Err(LanaError::Type); }
            self.check_resolved(candidate)?;
            if items.items[..index].iter().any(|prior| joint_value_equal(prior, candidate)) {
                return Err(LanaError::InvalidParameters);
            }
            domain.push(self.deep_clone_value(candidate, &mut memo)?);
        }
        Ok(domain)
    }

    fn core_kernel(&mut self, arguments: &[Value], scratch: u32, out: &mut Value) -> LanaError {
        if arguments.len() != 3 { return LanaError::Type; }
        let ValueKind::Function(function) = arguments[2].kind else { return LanaError::Type; };
        let saved = self.current_frame().registers[scratch as usize].clone();
        let result = (|| -> Result<Value, LanaError> {
            let ValueKind::Array(inputs) = &arguments[0].kind else { return Err(LanaError::Type); };
            let sources: Vec<Value> = inputs.lock().unwrap().items.iter().cloned().collect();
            if sources.is_empty() { return Err(LanaError::InvalidParameters); }
            let mut input_domains = Vec::new();
            input_domains.try_reserve(sources.len()).map_err(|_| LanaError::Oom)?;
            let mut row_count = 1usize;
            for source in &sources {
                let domain = self.kernel_domain(source)?;
                row_count = row_count.checked_mul(domain.len()).ok_or(LanaError::Limit)?;
                input_domains.push(domain);
            }
            let output_domain = self.kernel_domain(&arguments[1])?;
            let width = output_domain.len();
            let cells = row_count.checked_mul(width).ok_or(LanaError::Limit)?;
            self.charge_core_work(u64::try_from(cells).map_err(|_| LanaError::Limit)?)?;
            let _bytes = cells.checked_mul(std::mem::size_of::<f64>())
                .and_then(|n| n.checked_add(row_count.checked_mul(std::mem::size_of::<Vec<f64>>())?))
                .and_then(|n| n.checked_add(std::mem::size_of::<FiniteKernel>()))
                .ok_or(LanaError::Oom)?;
            let mut rows = Vec::new();
            rows.try_reserve(row_count).map_err(|_| LanaError::Oom)?;
            for index in 0..row_count {
                if self.cancelled.load(Ordering::Relaxed) { return Err(LanaError::Cancelled); }
                let mut offset = index;
                let mut tuple = Vec::new();
                tuple.try_reserve(input_domains.len()).map_err(|_| LanaError::Oom)?;
                for domain in input_domains.iter().rev() {
                    let mut memo = DeepCloneMemo::default();
                    tuple.push(self.deep_clone_value(&domain[offset % domain.len()], &mut memo)?);
                    offset /= domain.len();
                }
                tuple.reverse();
                let tuple = self.array_value(tuple)?;
                let mut returned = Value::null();
                self.pure_callback_depth += 1;
                let error = self.run_function(function, &tuple, scratch, &mut returned);
                self.pure_callback_depth -= 1;
                if error != LanaError::Ok { return Err(error); }
                let ValueKind::Array(pairs) = &returned.kind else { return Err(LanaError::Type); };
                let pairs: Vec<Value> = pairs.lock().unwrap().items.iter().cloned().collect();
                if pairs.len() != width { return Err(LanaError::InvalidDistribution); }
                let mut row = Vec::new();
                row.try_reserve(width).map_err(|_| LanaError::Oom)?;
                let mut total = 0.0;
                for (pair, output) in pairs.iter().zip(&output_domain) {
                    let ValueKind::Array(pair) = &pair.kind else { return Err(LanaError::Type); };
                    let pair = pair.lock().unwrap();
                    if pair.items.len() != 2 { return Err(LanaError::Type); }
                    if !joint_value_equal(&pair.items[0], output) { return Err(LanaError::InvalidDistribution); }
                    let ValueKind::Number(weight) = pair.items[1].kind else { return Err(LanaError::Type); };
                    if !weight.is_finite() || weight < 0.0 { return Err(LanaError::InvalidDistribution); }
                    total += weight;
                    row.push(weight);
                }
                if !total.is_finite() || (total - 1.0).abs() > 1e-12 { return Err(LanaError::InvalidDistribution); }
                for weight in &mut row { *weight /= total; }
                rows.push(row);
            }
            Ok(Value::kernel(self.managed_payload(FiniteKernel { input_domains, output_domain, rows })?))
        })();
        self.current_frame_mut().registers[scratch as usize] = saved;
        match result { Ok(value) => { *out = value; LanaError::Ok }, Err(error) => error }
    }

    fn core_kernel_identity(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 1 { return LanaError::Type; }
        let domain = match self.kernel_domain(&arguments[0]) { Ok(domain) => domain, Err(error) => return error };
        let count = domain.len();
        let cells = match count.checked_mul(count) { Some(cells) => cells, None => return LanaError::Limit };
        if let Err(error) = self.charge_core_work(cells as u64) { return error; }
        let _bytes = match cells.checked_mul(std::mem::size_of::<f64>()) { Some(bytes) => bytes, None => return LanaError::Oom };
        let mut rows = Vec::new();
        if rows.try_reserve(count).is_err() { return LanaError::Oom; }
        for index in 0..count {
            let mut row = Vec::new();
            if row.try_reserve(count).is_err() { return LanaError::Oom; }
            row.resize(count, 0.0);
            row[index] = 1.0;
            rows.push(row);
        }
        let output_domain = domain.clone();
        *out = Value::kernel(match self.managed_payload(FiniteKernel { input_domains: vec![domain], output_domain, rows }) { Ok(payload) => payload, Err(error) => return error });
        LanaError::Ok
    }

    fn core_kernel_compose(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 2 { return LanaError::Type; }
        let ValueKind::Kernel(later) = &arguments[0].kind else { return LanaError::Type; };
        let ValueKind::Kernel(earlier) = &arguments[1].kind else { return LanaError::Type; };
        if later.input_domains.len() != 1 { return LanaError::InvalidParameters; }
        let middle = earlier.output_domain.len();
        if middle == 0 || middle != later.input_domains[0].len() ||
            !earlier.output_domain.iter().zip(&later.input_domains[0]).all(|(a, b)| joint_value_equal(a, b)) {
            return LanaError::InvalidParameters;
        }
        let left = earlier.rows.len();
        let right = later.output_domain.len();
        if right == 0 || later.rows.len() != middle ||
            earlier.rows.iter().any(|row| row.len() != middle) ||
            later.rows.iter().any(|row| row.len() != right) {
            return LanaError::InvalidParameters;
        }
        let Some(cells) = left.checked_mul(right) else { return LanaError::Limit; };
        let Some(work) = cells.checked_mul(middle) else { return LanaError::Limit; };
        if let Err(error) = self.charge_core_work(work as u64) { return error; }
        let Some(_bytes) = cells.checked_mul(std::mem::size_of::<f64>()) else { return LanaError::Oom; };
        let mut rows = Vec::new();
        if rows.try_reserve(left).is_err() { return LanaError::Oom; }
        for input in 0..left {
            let mut row = Vec::new();
            if row.try_reserve(right).is_err() { return LanaError::Oom; }
            for output in 0..right {
                let mut weight = 0.0;
                for intermediate in 0..middle {
                    weight += earlier.rows[input][intermediate] * later.rows[intermediate][output];
                }
                if !weight.is_finite() || weight < 0.0 { return LanaError::InvalidDistribution; }
                row.push(weight);
            }
            let total = row.iter().sum::<f64>();
            if !total.is_finite() || (total - 1.0).abs() > 1e-12 { return LanaError::InvalidDistribution; }
            for weight in &mut row { *weight /= total; }
            rows.push(row);
        }
        let mut memo = DeepCloneMemo::default();
        let mut input_domains = Vec::new();
        for domain in &earlier.input_domains {
            let mut copied = Vec::new();
            for candidate in domain {
                match self.deep_clone_value(candidate, &mut memo) {
                    Ok(candidate) => copied.push(candidate), Err(error) => return error,
                }
            }
            input_domains.push(copied);
        }
        let mut output_domain = Vec::new();
        for candidate in &later.output_domain {
            match self.deep_clone_value(candidate, &mut memo) {
                Ok(candidate) => output_domain.push(candidate), Err(error) => return error,
            }
        }
        *out = Value::kernel(match self.managed_payload(FiniteKernel { input_domains, output_domain, rows }) { Ok(payload) => payload, Err(error) => return error });
        LanaError::Ok
    }

    fn network_root_domains(&mut self, root: &JointState) -> Result<Vec<Vec<Value>>, LanaError> {
        if root.names.is_empty() { return Err(LanaError::InvalidParameters); }
        let mut domains = vec![Vec::new(); root.names.len()];
        match root.kind {
            JointKind::FiniteLaw if !root.rows.is_empty() => {
                let work = root.rows.len().checked_mul(root.rows.len())
                    .and_then(|count| count.checked_mul(root.names.len())).ok_or(LanaError::Limit)?;
                self.charge_core_work(u64::try_from(work).map_err(|_| LanaError::Limit)?)?;
                let mut total = 0.0;
                for row in &root.rows {
                    self.charge_core_work(root.names.len() as u64)?;
                    if row.values.len() != root.names.len() || !row.weight.is_finite() || row.weight < 0.0 {
                        return Err(LanaError::InvalidDistribution);
                    }
                    total += row.weight;
                    if row.weight == 0.0 { continue; }
                    for (domain, value) in domains.iter_mut().zip(&row.values) {
                        if !joint_value_is_definite(value) { return Err(LanaError::Type); }
                        if !domain.iter().any(|prior| joint_value_equal(prior, value)) { domain.push(value.clone()); }
                    }
                }
                if !total.is_finite() || (total - 1.0).abs() > 1e-12 { return Err(LanaError::InvalidDistribution); }
            }
            JointKind::Independent if root.values.len() == root.names.len() => {
                for (domain, marginal) in domains.iter_mut().zip(&root.values) {
                    if let ValueKind::Possibility(possibility) = &marginal.kind {
                        let weights = possibility.weights.as_ref().ok_or(LanaError::UnsupportedOperation)?;
                        if weights.len() != possibility.values.len() { return Err(LanaError::InvalidDistribution); }
                        let work = weights.len().checked_mul(weights.len()).ok_or(LanaError::Limit)?;
                        self.charge_core_work(u64::try_from(work).map_err(|_| LanaError::Limit)?)?;
                        let mut total = 0.0;
                        for (value, weight) in possibility.values.iter().zip(weights) {
                            self.charge_core_work(1)?;
                            if !weight.is_finite() || *weight < 0.0 { return Err(LanaError::InvalidDistribution); }
                            total += weight;
                            if *weight > 0.0 && !domain.iter().any(|prior| joint_value_equal(prior, value)) {
                                domain.push(value.clone());
                            }
                        }
                        if !total.is_finite() || (total - 1.0).abs() > 1e-12 { return Err(LanaError::InvalidDistribution); }
                    } else if joint_value_is_definite(marginal) { domain.push(marginal.clone()); }
                    else { return Err(LanaError::UnsupportedOperation); }
                }
            }
            _ => return Err(LanaError::UnsupportedOperation),
        }
        if domains.iter().any(Vec::is_empty) { return Err(LanaError::InvalidDistribution); }
        Ok(domains)
    }

    fn core_network(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        let result = (|| -> Result<Value, LanaError> {
            if arguments.len() != 2 { return Err(LanaError::Type); }
            let ValueKind::Joint(root) = &arguments[0].kind else { return Err(LanaError::Type); };
            let ValueKind::Array(source_nodes) = &arguments[1].kind else { return Err(LanaError::Type); };
            let node_values: Vec<Value> = source_nodes.lock().unwrap().items.iter().cloned().collect();
            let mut domains = self.network_root_domains(root)?;
            let root_count = root.names.len();
            let mut names = root.names.clone();
            let mut declarations = Vec::new();
            declarations.try_reserve(node_values.len()).map_err(|_| LanaError::Oom)?;
            for record in &node_values {
                self.charge_core_work(names.len() as u64)?;
                let ValueKind::Map(record) = &record.kind else { return Err(LanaError::Type); };
                let record = record.lock().unwrap();
                let Some(name) = record.get("name") else { return Err(LanaError::InvalidParameters); };
                let ValueKind::String(name) = &name.kind else { return Err(LanaError::Type); };
                if name.is_empty() || names.iter().any(|prior| prior == name) { return Err(LanaError::InvalidParameters); }
                let Some(parents) = record.get("parents") else { return Err(LanaError::InvalidParameters); };
                let ValueKind::Array(parents) = &parents.kind else { return Err(LanaError::Type); };
                let parents: Vec<Value> = parents.lock().unwrap().items.iter().cloned().collect();
                let Some(kernel) = record.get("kernel") else { return Err(LanaError::InvalidParameters); };
                let ValueKind::Kernel(kernel) = &kernel.kind else { return Err(LanaError::Type); };
                declarations.push((name.clone(), parents, kernel.clone()));
                names.push(name.clone());
                domains.push(kernel.output_domain.clone());
            }
            let mut nodes = Vec::new();
            nodes.try_reserve(declarations.len()).map_err(|_| LanaError::Oom)?;
            for (name, parent_names, kernel) in declarations {
                if parent_names.len() != kernel.input_domains.len() { return Err(LanaError::InvalidParameters); }
                let mut parents = Vec::new();
                for (index, parent) in parent_names.iter().enumerate() {
                    self.charge_core_work(names.len() as u64)?;
                    let ValueKind::String(parent) = &parent.kind else { return Err(LanaError::Type); };
                    let Some(position) = names.iter().position(|candidate| candidate == parent) else {
                        return Err(LanaError::InvalidParameters);
                    };
                    self.charge_core_work(u64::try_from(domains[position].len()).map_err(|_| LanaError::Limit)?)?;
                    if parents.contains(&position) || domains[position].len() != kernel.input_domains[index].len()
                        || !domains[position].iter().zip(&kernel.input_domains[index])
                            .all(|(left, right)| joint_value_equal(left, right)) {
                        return Err(LanaError::InvalidParameters);
                    }
                    parents.push(position);
                }
                nodes.push(NetworkNode { name, parents, kernel });
            }
            let mut order = Vec::new();
            let mut done = vec![false; nodes.len()];
            while order.len() < nodes.len() {
                self.charge_core_work(nodes.len() as u64)?;
                let Some(index) = nodes.iter().enumerate().position(|(index, node)|
                    !done[index] && node.parents.iter().all(|parent|
                        *parent < root_count || done[*parent - root_count])) else {
                    return Err(LanaError::InvalidParameters);
                };
                done[index] = true;
                order.push(index);
            }
            let mut memo = DeepCloneMemo::default();
            let ValueKind::Joint(root) = self.deep_clone_value(&Value::joint(root.clone()), &mut memo)?.kind
                else { unreachable!() };
            for node in &mut nodes {
                let ValueKind::Kernel(kernel) = self.deep_clone_value(&Value::kernel(node.kernel.clone()), &mut memo)?.kind
                    else { unreachable!() };
                node.kernel = kernel;
            }
            Ok(Value::network(self.managed_payload(FiniteNetwork { root, nodes, order })?))
        })();
        match result { Ok(value) => { *out = value; LanaError::Ok }, Err(error) => error }
    }

    fn core_infer(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        let result = (|| -> Result<Value, LanaError> {
            if arguments.len() != 3 { return Err(LanaError::Type); }
            let ValueKind::Network(network) = &arguments[0].kind else { return Err(LanaError::Type); };
            let ValueKind::Array(query) = &arguments[1].kind else { return Err(LanaError::Type); };
            let ValueKind::Map(evidence) = &arguments[2].kind else { return Err(LanaError::Type); };
            let root_count = network.root.names.len();
            let mut names = network.root.names.clone();
            names.extend(network.nodes.iter().map(|node| node.name.clone()));
            let mut positions = Vec::new();
            for item in query.lock().unwrap().items.iter() {
                let ValueKind::String(name) = &item.kind else { return Err(LanaError::Type); };
                let Some(position) = names.iter().position(|candidate| candidate == name) else {
                    return Err(LanaError::InvalidParameters);
                };
                if positions.contains(&position) { return Err(LanaError::InvalidParameters); }
                positions.push(position);
            }
            if positions.is_empty() { return Err(LanaError::InvalidParameters); }
            let mut domains = self.network_root_domains(&network.root)?;
            domains.extend(network.nodes.iter().map(|node| node.kernel.output_domain.clone()));
            let mut conditions = Vec::new();
            for entry in evidence.lock().unwrap().entries() {
                let Some(position) = names.iter().position(|candidate| candidate == &entry.key) else {
                    return Err(LanaError::InvalidParameters);
                };
                if !joint_value_is_definite(&entry.value)
                    || !domains[position].iter().any(|candidate| joint_value_equal(candidate, &entry.value)) {
                    return Err(LanaError::InvalidParameters);
                }
                conditions.push((position, entry.value.clone()));
            }
            let mut max_rows = if network.root.kind == JointKind::FiniteLaw {
                network.root.rows.len()
            } else { 1 };
            if network.root.kind == JointKind::Independent {
                for domain in &domains[..root_count] {
                    max_rows = max_rows.checked_mul(domain.len()).ok_or(LanaError::Limit)?;
                }
            }
            for node in &network.nodes {
                max_rows = max_rows.checked_mul(node.kernel.output_domain.len()).ok_or(LanaError::Limit)?;
            }
            let bytes = max_rows.checked_mul(names.len().checked_mul(std::mem::size_of::<Value>())
                .and_then(|n| n.checked_add(std::mem::size_of::<JointRow>())).ok_or(LanaError::Oom)?)
                .and_then(|n| n.checked_mul(2)).ok_or(LanaError::Oom)?;
            let reservation = self.heap.reserve(bytes)?;
            let mut rows = Vec::new();
            if network.root.kind == JointKind::FiniteLaw {
                rows.try_reserve(network.root.rows.len()).map_err(|_| LanaError::Oom)?;
                for root_row in &network.root.rows {
                    self.charge_core_work(root_count as u64)?;
                    if root_row.weight <= 0.0 { continue; }
                    let mut values = vec![Value::null(); names.len()];
                    values[..root_count].clone_from_slice(&root_row.values);
                    rows.push(JointRow { values, weight: root_row.weight });
                }
            } else {
                rows.push(JointRow { values: vec![Value::null(); names.len()], weight: 1.0 });
                for (position, marginal) in network.root.values.iter().enumerate() {
                    let choices: Vec<(Value, f64)> = if let ValueKind::Possibility(possibility) = &marginal.kind {
                        possibility.values.iter().cloned().zip(possibility.weights.as_ref().unwrap().iter().copied()).collect()
                    } else { vec![(marginal.clone(), 1.0)] };
                    let mut expanded = Vec::new();
                    for row in &rows {
                        for (value, weight) in &choices {
                            self.charge_core_work(1)?;
                            if *weight == 0.0 { continue; }
                            let mut next = row.clone();
                            next.values[position] = value.clone();
                            next.weight *= weight;
                            if row.weight > 0.0 && *weight > 0.0 && next.weight == 0.0 {
                                return Err(LanaError::InvalidDistribution);
                            }
                            expanded.push(next);
                        }
                    }
                    rows = expanded;
                }
            }
            for &node_index in &network.order {
                let node = &network.nodes[node_index];
                let mut expanded = Vec::new();
                for row in &rows {
                    let mut kernel_row = 0usize;
                    for (parent, domain) in node.parents.iter().zip(&node.kernel.input_domains) {
                        self.charge_core_work(domain.len() as u64)?;
                        let Some(index) = domain.iter().position(|candidate|
                            joint_value_equal(candidate, &row.values[*parent])) else {
                            return Err(LanaError::InvalidParameters);
                        };
                        kernel_row = kernel_row.checked_mul(domain.len()).and_then(|n| n.checked_add(index))
                            .ok_or(LanaError::Limit)?;
                    }
                    let Some(chances) = node.kernel.rows.get(kernel_row) else { return Err(LanaError::InvalidDistribution); };
                    if chances.len() != node.kernel.output_domain.len() { return Err(LanaError::InvalidDistribution); }
                    for (value, weight) in node.kernel.output_domain.iter().zip(chances) {
                        self.charge_core_work(1)?;
                        if *weight == 0.0 { continue; }
                        let mut next = row.clone();
                        next.values[root_count + node_index] = value.clone();
                        next.weight *= weight;
                        if !next.weight.is_finite() || next.weight == 0.0 { return Err(LanaError::InvalidDistribution); }
                        expanded.push(next);
                    }
                }
                rows = expanded;
            }
            let mut groups: Vec<JointRow> = Vec::new();
            let mut mass = 0.0;
            for row in &rows {
                self.charge_core_work(conditions.len() as u64)?;
                if !conditions.iter().all(|(position, value)| joint_value_equal(&row.values[*position], value)) {
                    continue;
                }
                mass += row.weight;
                let selected: Vec<Value> = positions.iter().map(|position| row.values[*position].clone()).collect();
                let mut match_index = None;
                for (index, group) in groups.iter().enumerate() {
                    self.charge_core_work(positions.len() as u64)?;
                    if group.values.iter().zip(&selected).all(|(left, right)| joint_value_equal(left, right)) {
                        match_index = Some(index);
                        break;
                    }
                }
                if let Some(index) = match_index { groups[index].weight += row.weight; }
                else { groups.push(JointRow { values: selected, weight: row.weight }); }
            }
            if !mass.is_finite() || mass <= 0.0 { return Err(LanaError::InvalidConditioning); }
            drop(reservation);
            let output_bytes = groups.len().checked_mul(positions.len().checked_mul(std::mem::size_of::<Value>())
                .and_then(|n| n.checked_add(std::mem::size_of::<JointRow>())).ok_or(LanaError::Oom)?)
                .and_then(|n| n.checked_add(std::mem::size_of::<JointState>())).ok_or(LanaError::Oom)?;
            if self.alloc_bytes(output_bytes) != LanaError::Ok { return Err(LanaError::Oom); }
            let mut memo = DeepCloneMemo::default();
            for group in &mut groups {
                group.weight /= mass;
                for value in &mut group.values { *value = self.deep_clone_value(value, &mut memo)?; }
            }
            let query_names = positions.iter().map(|position| names[*position].clone()).collect();
            let types = positions.iter().map(|position| domains[*position][0].value_type()).collect();
            Ok(Value::joint(self.managed_payload(JointState {
                names: query_names, domains: types, values: Vec::new(), rows: groups,
                kind: JointKind::FiniteLaw,
                capabilities: LANA_JOINT_CAN_PROJECT | LANA_JOINT_CAN_CONDITION
                    | LANA_JOINT_CAN_SAMPLE | LANA_JOINT_CAN_RESOLVE,
            })?))
        })();
        match result { Ok(value) => { *out = value; LanaError::Ok }, Err(error) => error }
    }

    fn core_information_measure(&mut self, host_id: u32, arguments: &[Value], out: &mut Value) -> LanaError {
        let expected = if host_id == LANA_HOST_CORE_ENTROPY { 2 } else { 3 };
        if arguments.len() != expected { return LanaError::Type; }
        let current = self.reactive_value(&arguments[0]);
        let ValueKind::Joint(joint) = &current.kind else { return LanaError::Type; };
        let mut selected = Vec::new();
        let result = (|| {
            let left = Self::core_measure_names(joint, &arguments[1])?;
            selected.push(left.clone());
            if expected == 2 { return self.core_entropy(joint, &left); }
            let right = Self::core_measure_names(joint, &arguments[2])?;
            selected.push(right.clone());
            let mut union = left.clone();
            for &position in &right { if !union.contains(&position) { union.push(position); } }
            union.sort_unstable();
            let union_entropy = self.core_entropy(joint, &union)?;
            let value = if host_id == LANA_HOST_CORE_CONDITIONAL_ENTROPY {
                union_entropy - self.core_entropy(joint, &right)?
            } else {
                self.core_entropy(joint, &left)? + self.core_entropy(joint, &right)? - union_entropy
            };
            if !value.is_finite() || value < -1e-10 { return Err(LanaError::InvalidDistribution); }
            Ok(value.max(0.0))
        })();
        match result {
            Ok(value) => {
                let operation = match host_id {
                    LANA_HOST_CORE_ENTROPY => "entropy",
                    LANA_HOST_CORE_CONDITIONAL_ENTROPY => "conditional_entropy",
                    _ => "mutual_information",
                };
                let names = selected.iter().map(|group| group.iter().map(|&position|
                    joint.names[position].as_ref()).collect::<Vec<_>>().join(","))
                    .collect::<Vec<_>>().join(";");
                let revision = arguments[0].reactive.as_ref().map(|reactive| reactive.lock().unwrap().revision)
                    .or_else(|| arguments[0].derivation.as_ref().map(|node| node.revision))
                    .unwrap_or(self.revision);
                let details = format!("names={names}; input_revision={revision}");
                let Some(mut derivation) = self.record_derivation_payload(
                    DerivationKind::Operation, operation, &[&arguments[0]], "", 0,
                    DerivationExactness::Exact, &details, DerivationOutcome::Success, "none",
                ) else { return LanaError::Oom; };
                derivation.revision = revision;
            let derivation = match self.managed_payload(derivation) { Ok(node) => node, Err(error) => return error };
                *out = Value::number(value);
                out.derivation = Some(derivation);
                LanaError::Ok
            }
            Err(error) => error,
        }
    }

    fn core_broja(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        let result = (|| -> Result<Value, LanaError> {
            if arguments.len() != 4 { return Err(LanaError::Type); }
            let current = self.reactive_value(&arguments[0]);
            let ValueKind::Joint(joint) = &current.kind else { return Err(LanaError::Type); };
            if joint.kind != JointKind::FiniteLaw || joint.rows.is_empty() {
                return Err(LanaError::UnsupportedOperation);
            }
            let mut positions = Vec::new();
            for argument in &arguments[1..] {
                let ValueKind::String(name) = &argument.kind else { return Err(LanaError::Type); };
                let position = joint_find(joint, name).ok_or(LanaError::InvalidParameters)?;
                if positions.contains(&position) { return Err(LanaError::InvalidParameters); }
                positions.push(position);
            }
            let [target, x, y] = [positions[0], positions[1], positions[2]];
            let ht = self.core_entropy(joint, &[target])?;
            let hx = self.core_entropy(joint, &[x])?;
            let hy = self.core_entropy(joint, &[y])?;
            let htx = self.core_entropy(joint, &[target, x])?;
            let hty = self.core_entropy(joint, &[target, y])?;
            let hxy = self.core_entropy(joint, &[x, y])?;
            let htxy = self.core_entropy(joint, &[target, x, y])?;
            let a = (ht + hx - htx).max(0.0);
            let b = (ht + hy - hty).max(0.0);
            let c = (ht + hxy - htxy).max(0.0);
            if !a.is_finite() || !b.is_finite() || !c.is_finite() { return Err(LanaError::InvalidParameters); }
            let lower = a.max(b);
            if c < lower - 1e-10 { return Err(LanaError::InvalidParameters); }
            let mut upper = c;
            let mut marginal_residual = 0.0_f64;
            let mut mass = 0.0_f64;
            let count = joint.rows.len();
            let work = count.checked_mul(count).and_then(|n| n.checked_mul(3)).ok_or(LanaError::Limit)?;
            self.charge_core_work(u64::try_from(work).map_err(|_| LanaError::Limit)?)?;
            let domain_bytes = count.checked_mul(3).and_then(|n| n.checked_mul(std::mem::size_of::<Value>()))
                .ok_or(LanaError::Oom)?;
            let _domain_reservation = self.heap.reserve(domain_bytes)?;
            let mut domains = [Vec::<Value>::new(), Vec::new(), Vec::new()];
            for domain in &mut domains { domain.try_reserve(count).map_err(|_| LanaError::Oom)?; }
            for row in &joint.rows {
                if row.values.len() != joint.names.len() || !row.weight.is_finite() || row.weight < 0.0 {
                    return Err(LanaError::InvalidDistribution);
                }
                mass += row.weight;
                if row.weight == 0.0 { continue; }
                for (domain, position) in domains.iter_mut().zip(&positions) {
                    let value = &row.values[*position];
                    if !domain.iter().any(|prior| joint_value_equal(prior, value)) { domain.push(value.clone()); }
                }
            }
            let mut mass_residual = (mass - 1.0).abs();
            if !mass.is_finite() || mass_residual > 1e-12 { return Err(LanaError::InvalidDistribution); }
            let [nt, nx, ny] = [domains[0].len(), domains[1].len(), domains[2].len()];
            let tx_cells = nt.checked_mul(nx).ok_or(LanaError::Limit)?;
            let ty_cells = nt.checked_mul(ny).ok_or(LanaError::Limit)?;
            let cells = nt.checked_add(nx).and_then(|n| n.checked_add(ny))
                .and_then(|n| n.checked_add(tx_cells)).and_then(|n| n.checked_add(ty_cells))
                .ok_or(LanaError::Limit)?;
            let bytes = cells.checked_mul(std::mem::size_of::<f64>()).ok_or(LanaError::Oom)?;
            let _reservation = self.heap.reserve(bytes)?;
            let mut pt = vec![0.0; nt];
            let mut px = vec![0.0; nx];
            let mut py = vec![0.0; ny];
            let mut ptx = vec![0.0; tx_cells];
            let mut pty = vec![0.0; ty_cells];
            for row in &joint.rows {
                self.charge_core_work((nt + nx + ny) as u64)?;
                if row.weight == 0.0 { continue; }
                let t = domains[0].iter().position(|value| joint_value_equal(value, &row.values[target])).unwrap();
                let xv = domains[1].iter().position(|value| joint_value_equal(value, &row.values[x])).unwrap();
                let yv = domains[2].iter().position(|value| joint_value_equal(value, &row.values[y])).unwrap();
                pt[t] += row.weight;
                px[xv] += row.weight;
                py[yv] += row.weight;
                ptx[t * nx + xv] += row.weight;
                pty[t * ny + yv] += row.weight;
            }
            // When both source marginals factor exactly, T-independent Q is
            // feasible and has objective zero; nonnegativity certifies optimality.
            let mut factorized = mass_residual == 0.0;
            for t in 0..nt {
                for xv in 0..nx {
                    self.charge_core_work(1)?;
                    let residual = (ptx[t * nx + xv] - pt[t] * px[xv]).abs();
                    marginal_residual = marginal_residual.max(residual);
                    if residual != 0.0 { factorized = false; }
                }
                for yv in 0..ny {
                    self.charge_core_work(1)?;
                    let residual = (pty[t * ny + yv] - pt[t] * py[yv]).abs();
                    marginal_residual = marginal_residual.max(residual);
                    if residual != 0.0 { factorized = false; }
                }
            }
            let mut bound = (upper - lower).max(0.0) + 1e-10;
            if factorized { upper = 0.0; bound = 1e-10; }
            else {
                marginal_residual = 0.0; // Original P is the feasible upper candidate.
                if nx >= 2 && ny >= 2 {
                    let optimized = if nx == 2 && ny == 2 {
                        self.broja_binary(&pt, &ptx, &pty)?
                    } else if ny == 2 {
                        self.broja_one_binary(&pt, &ptx, nx, &pty)?
                    } else if nx == 2 {
                        self.broja_one_binary(&pt, &pty, ny, &ptx)?
                    } else {
                        self.broja_general(&pt, &ptx, nx, &pty, ny)?
                    };
                    if let Some((candidate, gap, residual, candidate_mass)) = optimized {
                        if candidate < upper && residual <= 1e-9 && candidate_mass <= 1e-9 {
                            if candidate < lower - 1e-10 { return Err(LanaError::InvalidParameters); }
                            upper = candidate;
                            marginal_residual = residual;
                            mass_residual = candidate_mass;
                            bound = gap.min((upper - lower).max(0.0)) + 1e-10;
                        }
                    }
                }
            }
            drop((domains, pt, px, py, ptx, pty));
            drop(_reservation);
            drop(_domain_reservation);
            let converged = bound <= 1e-6 && marginal_residual <= 1e-9 && mass_residual <= 1e-9;
            let revision = arguments[0].reactive.as_ref().map(|reactive| reactive.lock().unwrap().revision)
                .or_else(|| arguments[0].derivation.as_ref().map(|node| node.revision))
                .unwrap_or(self.revision);
            let mut map = Map::new(&self.heap, if converged { 11 } else { 9 })?;
            map.set(Arc::from("status"), Value::string(Arc::from(if converged { "converged" } else { "unconverged" })), true)?;
            map.set(Arc::from("error_bound_bits"), Value::number(bound), true)?;
            map.set(Arc::from("input_revision"), Value::number(revision as f64), true)?;
            map.set(Arc::from("target"), Value::string(joint.names[target].clone()), true)?;
            map.set(Arc::from("x"), Value::string(joint.names[x].clone()), true)?;
            map.set(Arc::from("y"), Value::string(joint.names[y].clone()), true)?;
            if converged {
                map.set(Arc::from("shared"), Value::number((a + b - upper).max(0.0)), true)?;
                map.set(Arc::from("unique_x"), Value::number((upper - b).max(0.0)), true)?;
                map.set(Arc::from("unique_y"), Value::number((upper - a).max(0.0)), true)?;
                map.set(Arc::from("synergy"), Value::number((c - upper).max(0.0)), true)?;
                map.set(Arc::from("total"), Value::number(c), true)?;
            } else {
                map.set(Arc::from("reason"), Value::string(Arc::from("objective_gap")), true)?;
                map.set(Arc::from("marginal_residual"), Value::number(0.0), true)?;
                map.set(Arc::from("mass_residual"), Value::number(mass_residual), true)?;
            }
            let details = format!("target={}; x={}; y={}; input_revision={revision}",
                joint.names[target], joint.names[x], joint.names[y]);
            let Some(mut derivation) = self.record_derivation_payload(
                DerivationKind::Operation, "broja", &[&arguments[0]], "", 0,
                DerivationExactness::Exact, &details, DerivationOutcome::Success, "none",
            ) else { return Err(LanaError::Oom); };
            derivation.revision = revision;
            let derivation = self.managed_payload(derivation)?;
            let mut output = Value::map(Arc::new(Mutex::new(map)));
            output.derivation = Some(derivation);
            Ok(output)
        })();
        match result { Ok(value) => { *out = value; LanaError::Ok }, Err(error) => error }
    }

    fn broja_binary(
        &mut self, pt: &[f64], ptx: &[f64], pty: &[f64],
    ) -> Result<Option<(f64, f64, f64, f64)>, LanaError> {
        let count = pt.len();
        let scratch_bytes = count.checked_mul(7).and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .ok_or(LanaError::Oom)?;
        let _scratch = self.heap.reserve(scratch_bytes)?;
        let row = |t: usize, z: f64| {
            let a = ptx[2 * t];
            let b = pty[2 * t];
            [z, a - z, b - z, pt[t] - a - b + z]
        };
        let mut lower = Vec::with_capacity(count);
        let mut upper = Vec::with_capacity(count);
        let mut coupling = Vec::with_capacity(count);
        for t in 0..count {
            let a = ptx[2 * t];
            let b = pty[2 * t];
            let lo = (a + b - pt[t]).max(0.0);
            let hi = a.min(b);
            if lo > hi + 1e-12 { return Err(LanaError::InvalidDistribution); }
            lower.push(lo);
            upper.push(hi.max(lo));
            coupling.push((lo + hi.max(lo)) / 2.0);
        }
        let gradient = |t: usize, candidate: f64, current: &[f64]| -> Option<f64> {
            let own = row(t, candidate);
            let mut xy = own;
            for other in 0..count {
                if other == t { continue; }
                let contribution = row(other, current[other]);
                for cell in 0..4 { xy[cell] += contribution[cell]; }
            }
            if own.iter().any(|value| *value <= 0.0) || xy.iter().any(|value| *value <= 0.0) {
                return None;
            }
            let logs = [0, 1, 2, 3].map(|cell| (own[cell] / xy[cell]).log2());
            Some(logs[0] - logs[1] - logs[2] + logs[3])
        };
        let mut gap = f64::INFINITY;
        for _ in 0..128 {
            if self.cancelled.load(Ordering::Relaxed) { return Err(LanaError::Cancelled); }
            for t in 0..count {
                let width = upper[t] - lower[t];
                if width <= 0.0 { continue; }
                let mut lo = lower[t] + width * 1e-14;
                let mut hi = upper[t] - width * 1e-14;
                for _ in 0..48 {
                    self.charge_core_work(u64::try_from(count.checked_mul(4).ok_or(LanaError::Limit)?)
                        .map_err(|_| LanaError::Limit)?)?;
                    let mid = (lo + hi) / 2.0;
                    let Some(slope) = gradient(t, mid, &coupling) else { return Ok(None); };
                    if slope > 0.0 { hi = mid; } else { lo = mid; }
                }
                coupling[t] = (lo + hi) / 2.0;
            }
            gap = 0.0;
            // Convexity of I(T;XY) at fixed P(T) makes this box-linearization
            // gap an upper bound on the remaining objective error.
            for t in 0..count {
                if upper[t] == lower[t] { continue; }
                self.charge_core_work(u64::try_from(count.checked_mul(4).ok_or(LanaError::Limit)?)
                    .map_err(|_| LanaError::Limit)?)?;
                let Some(slope) = gradient(t, coupling[t], &coupling) else { return Ok(None); };
                gap += if slope >= 0.0 {
                    (coupling[t] - lower[t]) * slope
                } else {
                    (upper[t] - coupling[t]) * -slope
                };
            }
            if !gap.is_finite() { return Ok(None); }
            if gap <= 1e-6 { break; }
        }
        let mut xy = [0.0; 4];
        let mut rows = Vec::with_capacity(count);
        for t in 0..count {
            self.charge_core_work(4)?;
            let values = row(t, coupling[t]);
            for cell in 0..4 { xy[cell] += values[cell]; }
            rows.push(values);
        }
        let mut objective = 0.0;
        let mut residual = 0.0_f64;
        let mut mass = 0.0;
        for t in 0..count {
            let q = rows[t];
            self.charge_core_work(4)?;
            residual = residual.max((q[0] + q[1] - ptx[2 * t]).abs());
            residual = residual.max((q[2] + q[3] - ptx[2 * t + 1]).abs());
            residual = residual.max((q[0] + q[2] - pty[2 * t]).abs());
            residual = residual.max((q[1] + q[3] - pty[2 * t + 1]).abs());
            for cell in 0..4 {
                if q[cell] < 0.0 { return Ok(None); }
                let value = q[cell];
                mass += value;
                if value > 0.0 {
                    objective += value * (value / (pt[t] * xy[cell])).log2();
                }
            }
        }
        if !objective.is_finite() || !residual.is_finite() || !mass.is_finite() {
            return Err(LanaError::InvalidParameters);
        }
        if objective < -1e-10 { return Err(LanaError::InvalidParameters); }
        Ok(Some((objective.max(0.0), gap.max(0.0), residual, (mass - 1.0).abs())))
    }

    fn broja_one_binary(
        &mut self, pt: &[f64], wide: &[f64], width: usize, binary: &[f64],
    ) -> Result<Option<(f64, f64, f64, f64)>, LanaError> {
        let count = pt.len();
        let cells = count.checked_mul(width).ok_or(LanaError::Limit)?;
        let scratch_bytes = cells.checked_add(width.checked_mul(4).ok_or(LanaError::Oom)?)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>())).ok_or(LanaError::Oom)?;
        let _scratch = self.heap.reserve(scratch_bytes)?;
        let mut coupling = Vec::new();
        coupling.try_reserve(cells).map_err(|_| LanaError::Oom)?;
        let mut wide_total = vec![0.0; width];
        for t in 0..count {
            for i in 0..width {
                self.charge_core_work(1)?;
                let supply = wide[t * width + i];
                if supply < 0.0 || !supply.is_finite() { return Err(LanaError::InvalidDistribution); }
                wide_total[i] += supply;
                coupling.push(if pt[t] > 0.0 { supply * binary[2 * t] / pt[t] } else { 0.0 });
            }
        }
        let gradient = |t: usize, i: usize, candidate: f64, values: &[f64]| -> Option<f64> {
            let supply = wide[t * width + i];
            if supply == 0.0 { return Some(0.0); }
            let mut xy0 = candidate;
            for other in 0..count {
                if other != t { xy0 += values[other * width + i]; }
            }
            let xy1 = wide_total[i] - xy0;
            if candidate <= 0.0 || candidate >= supply || xy0 <= 0.0 || xy1 <= 0.0 {
                return None;
            }
            Some((candidate / xy0).log2() - ((supply - candidate) / xy1).log2())
        };
        let mut gap = f64::INFINITY;
        // ponytail: bounded pair sweeps; a full transport oracle is needed when both source domains exceed two.
        for _ in 0..512 {
            if self.cancelled.load(Ordering::Relaxed) { return Err(LanaError::Cancelled); }
            for t in 0..count {
                if binary[2 * t] <= 0.0 || binary[2 * t] >= pt[t] { continue; }
                for i in 0..width {
                    if wide[t * width + i] == 0.0 { continue; }
                    for j in i + 1..width {
                        if wide[t * width + j] == 0.0 { continue; }
                        let left = t * width + i;
                        let right = t * width + j;
                        let lo = (-coupling[left]).max(coupling[right] - wide[right]);
                        let hi = (wide[left] - coupling[left]).min(coupling[right]);
                        if hi <= lo { continue; }
                        let margin = (hi - lo) * 1e-14;
                        let mut a = lo + margin;
                        let mut b = hi - margin;
                        for _ in 0..48 {
                            self.charge_core_work(u64::try_from(count.checked_mul(2).ok_or(LanaError::Limit)?)
                                .map_err(|_| LanaError::Limit)?)?;
                            let delta = (a + b) / 2.0;
                            let Some(gi) = gradient(t, i, coupling[left] + delta, &coupling) else { return Ok(None); };
                            let Some(gj) = gradient(t, j, coupling[right] - delta, &coupling) else { return Ok(None); };
                            if gi > gj { b = delta; } else { a = delta; }
                        }
                        let delta = (a + b) / 2.0;
                        coupling[left] += delta;
                        coupling[right] -= delta;
                    }
                }
            }
            gap = 0.0;
            for t in 0..count {
                if binary[2 * t] <= 0.0 || binary[2 * t] >= pt[t] { continue; }
                let mut slopes = Vec::new();
                slopes.try_reserve(width).map_err(|_| LanaError::Oom)?;
                for i in 0..width {
                    self.charge_core_work(count as u64)?;
                    let Some(slope) = gradient(t, i, coupling[t * width + i], &coupling) else { return Ok(None); };
                    if !slope.is_finite() { return Ok(None); }
                    slopes.push((i, slope));
                }
                self.charge_core_work(u64::try_from(width.checked_mul(width).ok_or(LanaError::Limit)?)
                    .map_err(|_| LanaError::Limit)?)?;
                slopes.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                let mut remaining = binary[2 * t];
                for (i, slope) in slopes {
                    let assigned = remaining.min(wide[t * width + i]).max(0.0);
                    remaining -= assigned;
                    gap += slope * (coupling[t * width + i] - assigned);
                }
                if remaining.abs() > 1e-9 { return Ok(None); }
            }
            if !gap.is_finite() || gap < -1e-8 { return Ok(None); }
            if gap <= 1e-6 { break; }
        }
        let mut xy0 = vec![0.0; width];
        for t in 0..count {
            for i in 0..width { xy0[i] += coupling[t * width + i]; }
        }
        let mut objective = 0.0;
        let mut residual = 0.0_f64;
        let mut mass = 0.0;
        for t in 0..count {
            let mut binary_zero = 0.0;
            let mut binary_one = 0.0;
            for i in 0..width {
                self.charge_core_work(2)?;
                let q0 = coupling[t * width + i];
                let q1 = wide[t * width + i] - q0;
                if q0 < 0.0 || q1 < 0.0 { return Ok(None); }
                binary_zero += q0;
                binary_one += q1;
                residual = residual.max((q0 + q1 - wide[t * width + i]).abs());
                mass += q0 + q1;
                if q0 > 0.0 { objective += q0 * (q0 / (pt[t] * xy0[i])).log2(); }
                if q1 > 0.0 { objective += q1 * (q1 / (pt[t] * (wide_total[i] - xy0[i]))).log2(); }
            }
            residual = residual.max((binary_zero - binary[2 * t]).abs());
            residual = residual.max((binary_one - binary[2 * t + 1]).abs());
        }
        if !objective.is_finite() || !residual.is_finite() || !mass.is_finite() || objective < -1e-10 {
            return Err(LanaError::InvalidParameters);
        }
        Ok(Some((objective.max(0.0), gap.max(0.0), residual, (mass - 1.0).abs())))
    }

    fn broja_transport_dual(
        &mut self, rows: &[f64], cols: &[f64], costs: &[f64],
    ) -> Result<Option<f64>, LanaError> {
        let (nx, ny) = (rows.len(), cols.len());
        let nodes = nx.checked_add(ny).ok_or(LanaError::Limit)?;
        let cells = nx.checked_mul(ny).ok_or(LanaError::Limit)?;
        let bytes = cells.checked_mul(std::mem::size_of::<f64>())
            .and_then(|n| nodes.checked_mul(3 * std::mem::size_of::<f64>()
                + std::mem::size_of::<(usize, usize, bool)>()
                + std::mem::size_of::<(usize, bool)>()).and_then(|extra| n.checked_add(extra)))
            .ok_or(LanaError::Oom)?;
        let _scratch = self.heap.reserve(bytes)?;
        let mut flow = vec![0.0; cells];
        let mut supply = rows.to_vec();
        let mut demand = cols.to_vec();
        // Northwest corner gives a feasible starting transport, including zero rows.
        for i in 0..nx {
            for j in 0..ny {
                self.charge_core_work(1)?;
                let amount = supply[i].min(demand[j]).max(0.0);
                flow[i * ny + j] = amount;
                supply[i] -= amount;
                demand[j] -= amount;
            }
        }
        if supply.iter().chain(&demand).any(|value| value.abs() > 1e-10) { return Ok(None); }
        let mut dist = vec![0.0; nodes];
        let mut previous = vec![(0usize, 0usize, false); nodes];
        for _ in 0..10_000 {
            if self.cancelled.load(Ordering::Relaxed) { return Err(LanaError::Cancelled); }
            dist.fill(0.0);
            let mut changed = None;
            for _ in 0..nodes {
                changed = None;
                for i in 0..nx {
                    for j in 0..ny {
                        self.charge_core_work(2)?;
                        let cell = i * ny + j;
                        if dist[nx + j] > dist[i] + costs[cell] + 1e-12 {
                            dist[nx + j] = dist[i] + costs[cell];
                            previous[nx + j] = (i, cell, false);
                            changed = Some(nx + j);
                        }
                        if flow[cell] > 1e-15 && dist[i] > dist[nx + j] - costs[cell] + 1e-12 {
                            dist[i] = dist[nx + j] - costs[cell];
                            previous[i] = (nx + j, cell, true);
                            changed = Some(i);
                        }
                    }
                }
                if changed.is_none() { break; }
            }
            let Some(mut node) = changed else {
                let mut dual = 0.0;
                let mut violation = 0.0_f64;
                for i in 0..nx { dual -= rows[i] * dist[i]; }
                for j in 0..ny { dual += cols[j] * dist[nx + j]; }
                // A feasible transport dual is a lower bound on every linearized coupling.
                for i in 0..nx {
                    for j in 0..ny {
                        self.charge_core_work(1)?;
                        violation = violation.max(dist[nx + j] - dist[i] - costs[i * ny + j]);
                    }
                }
                if violation > 1e-9 { return Ok(None); }
                return Ok(Some(dual - violation * rows.iter().sum::<f64>() - 1e-10));
            };
            for _ in 0..nodes { self.charge_core_work(1)?; node = previous[node].0; }
            let start = node;
            let mut cycle = Vec::with_capacity(nodes);
            let mut amount = f64::INFINITY;
            loop {
                self.charge_core_work(1)?;
                let (parent, cell, reverse) = previous[node];
                if reverse { amount = amount.min(flow[cell]); }
                cycle.push((cell, reverse));
                node = parent;
                if node == start { break; }
                if cycle.len() > nodes { return Ok(None); }
            }
            if !amount.is_finite() || amount <= 1e-15 { return Ok(None); }
            for (cell, reverse) in cycle {
                self.charge_core_work(1)?;
                if reverse { flow[cell] -= amount; } else { flow[cell] += amount; }
            }
        }
        Ok(None)
    }

    fn broja_general(
        &mut self, pt: &[f64], ptx: &[f64], nx: usize, pty: &[f64], ny: usize,
    ) -> Result<Option<(f64, f64, f64, f64)>, LanaError> {
        let plane = nx.checked_mul(ny).ok_or(LanaError::Limit)?;
        let cells = pt.len().checked_mul(plane).ok_or(LanaError::Limit)?;
        let bytes = cells.checked_add(plane.checked_mul(2).ok_or(LanaError::Oom)?)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>())).ok_or(LanaError::Oom)?;
        let _scratch = self.heap.reserve(bytes)?;
        let mut q = vec![0.0; cells];
        let mut xy = vec![0.0; plane];
        for t in 0..pt.len() {
            for i in 0..nx {
                for j in 0..ny {
                    self.charge_core_work(1)?;
                    let cell = i * ny + j;
                    let value = if pt[t] > 0.0 { ptx[t * nx + i] * pty[t * ny + j] / pt[t] } else { 0.0 };
                    q[t * plane + cell] = value;
                    xy[cell] += value;
                }
            }
        }
        let mut gap = f64::INFINITY;
        for _ in 0..512 {
            if self.cancelled.load(Ordering::Relaxed) { return Err(LanaError::Cancelled); }
            for t in 0..pt.len() {
                for i in 0..nx {
                    if ptx[t * nx + i] == 0.0 { continue; }
                    for k in i + 1..nx {
                        if ptx[t * nx + k] == 0.0 { continue; }
                        for j in 0..ny {
                            if pty[t * ny + j] == 0.0 { continue; }
                            for l in j + 1..ny {
                                if pty[t * ny + l] == 0.0 { continue; }
                                let cells = [i * ny + j, i * ny + l, k * ny + j, k * ny + l];
                                let signs = [1.0, -1.0, -1.0, 1.0];
                                let lo = (-q[t * plane + cells[0]]).max(-q[t * plane + cells[3]]);
                                let hi = q[t * plane + cells[1]].min(q[t * plane + cells[2]]);
                                if hi <= lo { continue; }
                                let margin = (hi - lo) * 1e-9;
                                let (mut left, mut right) = (lo + margin, hi - margin);
                                for _ in 0..48 {
                                    self.charge_core_work(4)?;
                                    let delta = (left + right) / 2.0;
                                    let mut slope = 0.0;
                                    for n in 0..4 {
                                        let own = q[t * plane + cells[n]] + signs[n] * delta;
                                        let total = xy[cells[n]] + signs[n] * delta;
                                        if own <= 0.0 || total <= 0.0 { return Ok(None); }
                                        slope += signs[n] * (own / total).log2();
                                    }
                                    if slope > 0.0 { right = delta; } else { left = delta; }
                                }
                                let delta = (left + right) / 2.0;
                                for n in 0..4 {
                                    q[t * plane + cells[n]] += signs[n] * delta;
                                    xy[cells[n]] += signs[n] * delta;
                                }
                            }
                        }
                    }
                }
            }
            xy.fill(0.0);
            for t in 0..pt.len() {
                for cell in 0..plane {
                    self.charge_core_work(1)?;
                    xy[cell] += q[t * plane + cell];
                }
            }
            gap = 0.0;
            for t in 0..pt.len() {
                let mut costs = vec![0.0; plane];
                let mut current = 0.0;
                for cell in 0..plane {
                    self.charge_core_work(1)?;
                    let own = q[t * plane + cell];
                    let i = cell / ny;
                    let j = cell % ny;
                    if ptx[t * nx + i] > 0.0 && pty[t * ny + j] > 0.0 && own <= 0.0 {
                        return Ok(None);
                    }
                    if own > 0.0 {
                        costs[cell] = (own / xy[cell]).log2();
                        current += own * costs[cell];
                    }
                    if !costs[cell].is_finite() { return Ok(None); }
                }
                let Some(dual) = self.broja_transport_dual(
                    &ptx[t * nx..(t + 1) * nx], &pty[t * ny..(t + 1) * ny], &costs,
                )? else { return Ok(None); };
                gap += current - dual;
            }
            if !gap.is_finite() || gap < -1e-8 { return Ok(None); }
            if gap <= 1e-6 { break; }
        }
        let mut objective = 0.0;
        let mut residual = 0.0_f64;
        let mut mass = 0.0;
        for t in 0..pt.len() {
            for i in 0..nx {
                let mut row = 0.0;
                for j in 0..ny {
                    self.charge_core_work(1)?;
                    let value = q[t * plane + i * ny + j];
                    if value < -1e-14 { return Ok(None); }
                    row += value;
                    mass += value;
                    if value > 0.0 { objective += value * (value / (pt[t] * xy[i * ny + j])).log2(); }
                }
                residual = residual.max((row - ptx[t * nx + i]).abs());
            }
            for j in 0..ny {
                let mut column = 0.0;
                for i in 0..nx { column += q[t * plane + i * ny + j]; }
                residual = residual.max((column - pty[t * ny + j]).abs());
            }
        }
        if !objective.is_finite() || !residual.is_finite() || !mass.is_finite() || objective < -1e-10 {
            return Ok(None);
        }
        Ok(Some((objective.max(0.0), gap.max(0.0), residual, (mass - 1.0).abs())))
    }

    fn core_convert_weights(&mut self, host_id: u32, arguments: &[Value], out: &mut Value) -> LanaError {
        let expected = if host_id == LANA_HOST_CORE_FORGET_WEIGHTS { 1 } else { 2 };
        if arguments.len() != expected { return LanaError::Type; }
        let current = self.reactive_value(&arguments[0]);
        let ValueKind::Possibility(source) = &current.kind else { return LanaError::Type; };
        let built = if host_id == LANA_HOST_CORE_FORGET_WEIGHTS {
            let Some(source_weights) = &source.weights else { return LanaError::Type; };
            if source_weights.len() != source.values.len() { return LanaError::InvalidDistribution; }
            let mut total = 0.0;
            let mut positive = Vec::new();
            if positive.try_reserve(source.values.len()).is_err() { return LanaError::Oom; }
            for (index, &weight) in source_weights.iter().enumerate() {
                if !weight.is_finite() || weight < 0.0 { return LanaError::InvalidDistribution; }
                total += weight;
                if weight > 0.0 { positive.push(source.values[index].clone()); }
            }
            if !total.is_finite() || (total - 1.0).abs() > 1e-12 { return LanaError::InvalidDistribution; }
            if positive.is_empty() { return LanaError::InvalidDistribution; }
            let built = match self.possibility_build(&positive) { Ok(value) => value, Err(error) => return error };
            let mut built = (*built).clone();
            built.dependency_id = source.dependency_id;
            built
        } else {
            if source.weights.is_some() { return LanaError::Type; }
            let ValueKind::Array(rows) = &arguments[1].kind else { return LanaError::Type; };
            let rows = rows.lock().unwrap();
            let count = source.values.len();
            if rows.items.len() != count { return LanaError::InvalidDistribution; }
            // ponytail: Linear candidate matching is bounded by the VM work limit; index only if large finite supports become common.
            let work = match count.checked_mul(count) { Some(work) => work as u64, None => return LanaError::Limit };
            if let Err(error) = self.charge_core_work(work) { return error; }
            let mut weights = Vec::new();
            if weights.try_reserve(count).is_err() { return LanaError::Oom; }
            weights.resize(count, None);
            let mut total = 0.0;
            for row in rows.items.iter() {
                let ValueKind::Array(pair) = &row.kind else { return LanaError::Type; };
                let pair = pair.lock().unwrap();
                if pair.items.len() != 2 { return LanaError::Type; }
                let ValueKind::Number(weight) = pair.items[1].kind else { return LanaError::Type; };
                if !weight.is_finite() || weight <= 0.0 { return LanaError::InvalidDistribution; }
                let Some(index) = source.values.iter().position(|candidate|
                    joint_value_equal(candidate, &pair.items[0])) else { return LanaError::InvalidDistribution; };
                if weights[index].replace(weight).is_some() { return LanaError::InvalidDistribution; }
                total += weight;
            }
            if !total.is_finite() || (total - 1.0).abs() > 1e-12 { return LanaError::InvalidDistribution; }
            let Some(weights) = weights.into_iter().collect::<Option<Vec<_>>>() else {
                return LanaError::InvalidDistribution;
            };
            drop(rows);
            let built = match self.possibility_build(&source.values) { Ok(value) => value, Err(error) => return error };
            let mut built = (*built).clone();
            built.weights = Some(weights);
            built.dependency_id = source.dependency_id;
            built
        };
        let operation = if host_id == LANA_HOST_CORE_FORGET_WEIGHTS { "forget_weights" } else { "assign_weights" };
        let details = if host_id == LANA_HOST_CORE_FORGET_WEIGHTS { "probabilities_discarded" } else { "explicit_weights" };
        let Some(mut derivation) = self.record_derivation_payload(
            DerivationKind::Operation, operation, &[&arguments[0]], "", 0,
            DerivationExactness::Exact, details, DerivationOutcome::Success, "none",
        ) else { return LanaError::Oom; };
        let revision = arguments[0].reactive.as_ref().map(|reactive| reactive.lock().unwrap().revision)
            .or_else(|| arguments[0].derivation.as_ref().map(|node| node.revision))
            .unwrap_or(self.revision);
        derivation.revision = revision;
            let derivation = match self.managed_payload(derivation) { Ok(node) => node, Err(error) => return error };
        *out = Value::possibility(match self.managed_payload(built) { Ok(payload) => payload, Err(error) => return error });
        out.derivation = Some(derivation);
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
        self.managed_payload(joint)
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
        self.managed_payload(joint)
    }

    /// Build a finite law from an array of rows, mirroring
    /// `joint_build_finite_array` in `vm/c/vm.c`.
    pub fn joint_build_finite_array(
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
            self.managed_payload(joint)
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
            self.managed_payload(joint)
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
            self.managed_payload(joint)
        } else {
            if !joint_value_is_definite(&source.values[position]) {
                return Err(LanaError::UnsupportedOperation);
            }
            if !joint_value_equal(&source.values[position], evidence) {
                return Err(LanaError::InvalidConditioning);
            }
            let mut memo = DeepCloneMemo::default();
            let wrapped = Value::joint(self.managed_payload(source.clone())?);
            let mut cloned = self.deep_clone_value(&wrapped, &mut memo)?;
            let ValueKind::Joint(joint) = &mut cloned.kind else {
                unreachable!("wrapped value is a joint");
            };
            let mut state = (**joint).clone();
            state.kind = JointKind::Conditional;
            *joint = self.managed_payload(state)?;
            Ok(joint.clone())
        }
    }

    fn joint_condition_map(&mut self, source: &JointState, evidence: &Value) -> Result<Arc<JointState>, LanaError> {
        let ValueKind::Map(map) = &evidence.kind else {
            return Err(LanaError::Type);
        };
        let entries = map.lock().unwrap().entries().iter()
            .map(|entry| (entry.key.to_string(), entry.value.clone()))
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return Err(LanaError::InvalidConditioning);
        }
        let mut conditioned = None;
        for (name, value) in entries {
            let input = conditioned.as_deref().unwrap_or(source);
            conditioned = Some(self.joint_condition(input, &name, &value)?);
        }
        Ok(conditioned.unwrap())
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

    fn core_condition(&mut self, source: &Value, evidence: &Value) -> Result<Value, LanaError> {
        let current = self.reactive_value(source);
        if let ValueKind::Joint(joint) = &current.kind {
            return self.joint_condition_map(joint, evidence).map(Value::joint);
        }
        if matches!(current.kind, ValueKind::PathSet(_) | ValueKind::StateDist(_)) {
            return Err(LanaError::UnsupportedOperation);
        }
        let candidates: Vec<&Value> = match &evidence.kind {
            ValueKind::Possibility(subset) if subset.weights.is_none() => subset.values.iter().collect(),
            ValueKind::Possibility(_) => return Err(LanaError::Type),
            _ if joint_value_is_definite(evidence) => vec![evidence],
            _ => return Err(LanaError::Type),
        };
        if candidates.is_empty() || candidates.iter().any(|value| {
            matches!(value.kind, ValueKind::Possibility(_) | ValueKind::PathSet(_))
                || !joint_value_is_definite(value)
        }) {
            return Err(LanaError::Type);
        }
        for candidate in &candidates {
            self.check_resolved(candidate)?;
        }
        let matches = |value: &Value| candidates.iter().any(|candidate| joint_value_equal(value, candidate));
        if let ValueKind::Possibility(possibility) = &current.kind {
            let indices: Vec<usize> = possibility.values.iter().enumerate()
                .filter_map(|(index, value)| matches(value).then_some(index)).collect();
            if indices.is_empty() {
                return Err(LanaError::InvalidConditioning);
            }
            let mut memo = DeepCloneMemo::default();
            let mut values = Vec::with_capacity(indices.len());
            for &index in &indices {
                values.push(self.deep_clone_value(&possibility.values[index], &mut memo)?);
            }
            let weights = possibility.weights.as_ref().map(|weights| {
                let total: f64 = indices.iter().map(|&index| weights[index]).sum();
                indices.iter().map(|&index| weights[index] / total).collect()
            });
            return Ok(Value::possibility(self.managed_payload(Possibility {
                values, weights, dependency_id: possibility.dependency_id,
            })?));
        }
        if !joint_value_is_definite(&current) {
            return Err(LanaError::Type);
        }
        self.check_resolved(&current)?;
        if !matches(&current) {
            return Err(LanaError::InvalidConditioning);
        }
        let mut memo = DeepCloneMemo::default();
        self.deep_clone_value(&current, &mut memo)
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
        self.array_value(items)
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
        self.array_value(items)
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
    pub fn possibility_build(&mut self, values: &[Value]) -> Result<Arc<Possibility>, LanaError> {
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
        let mut memo = DeepCloneMemo::default();
        let mut cloned = Vec::with_capacity(unique.len());
        for value in &unique {
            cloned.push(self.deep_clone_value(value, &mut memo)?);
        }
        let dependency_id = self.next_dependency_id;
        self.next_dependency_id += 1;
        Ok(self.managed_payload(Possibility {
            values: cloned,
            weights: None,
            dependency_id,
        })?)
    }

    /// Build a finite weighted Core distribution from `[[value, weight], ...]`.
    pub fn distribution_build(&mut self, rows: &[Value]) -> Result<Arc<Possibility>, LanaError> {
        if rows.is_empty() {
            return Err(LanaError::InvalidDistribution);
        }
        let mut values = Vec::with_capacity(rows.len());
        let mut weights = Vec::with_capacity(rows.len());
        let mut total = 0.0;
        for row in rows {
            let ValueKind::Array(pair) = &row.kind else {
                return Err(LanaError::Type);
            };
            let pair = pair.lock().unwrap();
            if pair.items.len() != 2 || !joint_value_is_definite(&pair.items[0]) {
                return Err(LanaError::Type);
            }
            let ValueKind::Number(weight) = pair.items[1].kind else {
                return Err(LanaError::Type);
            };
            if !weight.is_finite() || weight <= 0.0 {
                return Err(LanaError::InvalidDistribution);
            }
            if values.iter().any(|value| joint_value_equal(value, &pair.items[0])) {
                return Err(LanaError::InvalidDistribution);
            }
            total += weight;
            values.push(pair.items[0].clone());
            weights.push(weight);
        }
        if !total.is_finite() || (total - 1.0).abs() > 1e-12 {
            return Err(LanaError::InvalidDistribution);
        }
        let mut memo = DeepCloneMemo::default();
        let mut cloned = Vec::with_capacity(values.len());
        for value in &values {
            cloned.push(self.deep_clone_value(value, &mut memo)?);
        }
        let dependency_id = self.next_dependency_id;
        self.next_dependency_id += 1;
        Ok(self.managed_payload(Possibility { values: cloned, weights: Some(weights), dependency_id })?)
    }

    /// Resolve any information value, mirroring `lana_vm_information_resolve`.
    pub fn information_resolve(&mut self, source: &Value) -> Result<Value, LanaError> {
        if source.reactive.is_some() {
            let current = self.reactive_value(source);
            return self.information_resolve(&current);
        }
        match &source.kind {
            ValueKind::Joint(joint) => self.joint_resolve(joint),
            ValueKind::Possibility(possibility) => {
                if possibility.values.is_empty() || possibility.values[1..].iter()
                    .any(|value| !joint_value_equal(&possibility.values[0], value)) {
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
                if let Some(weights) = &possibility.weights {
                    if self.consume_sampling_budget() != LanaError::Ok {
                        return Err(LanaError::BudgetExhausted);
                    }
                    let draw = self.rng.random() as f64 / 4294967296.0;
                    let mut cumulative = 0.0;
                    let mut selected = possibility.values.len() - 1;
                    for (index, weight) in weights.iter().enumerate() {
                        cumulative += weight;
                        if draw < cumulative {
                            selected = index;
                            break;
                        }
                    }
                    let mut memo = DeepCloneMemo::default();
                    return self.deep_clone_value(&possibility.values[selected], &mut memo);
                }
                if self.chunk.version >= lana_bytecode::opcode::LABC_VERSION_5 {
                    return Err(LanaError::UnsupportedOperation);
                }
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
            &[source],
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
        self.array_value(items)
    }

    /// Render a derivation as a map, mirroring `derivation_to_value`.
    fn derivation_to_value(&mut self, node: &Derivation) -> Result<Value, LanaError> {
        let mut map = Map::new(&self.heap, 13)?;
        let id = self.derivation_id_to_value(node)?;
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for input in &node.inputs {
            inputs.push(self.derivation_id_to_value(input)?);
        }
        let mut source_map = Map::new(&self.heap, 3)?;
        source_map.set(Arc::from("label"), Value::string(node.label.clone()), true)?;
        source_map.set(Arc::from("function"), Value::string(node.function.clone()), true)?;
        source_map.set(Arc::from("line"), Value::number(node.line as f64), true)?;
        let mut details_map = Map::new(&self.heap, 1)?;
        details_map.set(Arc::from("summary"), Value::string(node.details.clone()), true)?;
        map.set(Arc::from("id"), id, true)?;
        map.set(Arc::from("revision"), Value::number(node.revision as f64), true)?;
        map.set(Arc::from("kind"), Value::string(Arc::from(derivation::kind_name(node.kind))), true)?;
        map.set(Arc::from("operation"), Value::string(node.operation.clone()), true)?;
        map.set(Arc::from("inputs"), self.array_value(inputs)?, true)?;
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
        self.string_value(&rendered)
    }

    fn reactive_derived_value(
        &mut self,
        left: &Value,
        right: Option<&Value>,
        kind: ReactiveKind,
        operation: u32,
        out: &mut Value,
    ) -> LanaError {
        let left_reactive = left.reactive.clone();
        let right_reactive = right.and_then(|value| value.reactive.clone());
        if let (Some(left), Some(right)) = (&left_reactive, &right_reactive) {
            let left_dependency_id = left.lock().unwrap().dependency_id;
            let right_dependency_id = right.lock().unwrap().dependency_id;
            if left_dependency_id != right_dependency_id {
                return LanaError::UnsupportedOperation;
            }
        }
        let (dependency_id, exactness) = left_reactive
            .as_ref()
            .or(right_reactive.as_ref())
            .map(|node| {
                let node = node.lock().unwrap();
                (node.dependency_id, node.exactness)
            })
            .unwrap();
        let exactness = right_reactive.as_ref().map_or(exactness, |node| {
            exactness.max(node.lock().unwrap().exactness)
        });
        let constant0 = if left_reactive.is_none() {
            match self.clone_without_runtime_metadata(left) {
                Ok(value) => Some(value),
                Err(error) => return error,
            }
        } else {
            None
        };
        let constant1 = match (right, &right_reactive) {
            (Some(value), None) => match self.clone_without_runtime_metadata(value) {
                Ok(value) => Some(value),
                Err(error) => return error,
            },
            _ => None,
        };
        let current = match self.clone_without_runtime_metadata(out) {
            Ok(value) => value,
            Err(error) => return error,
        };
        let node = Arc::new(Mutex::new(Reactive {
            id: self.next_reactive_id,
            dependency_id,
            revision: self.revision,
            kind,
            relationship: if left_reactive.is_some() && right_reactive.is_some() {
                RelationshipKind::SameDependency
            } else {
                RelationshipKind::Exact
            },
            exactness,
            operation,
            inputs: [left_reactive, right_reactive],
            constants: [constant0, constant1],
            current: Some(current),
            history: Vec::new(),
            is_training_data: false,
        }));
        if let Err(error) = self.track_cycle(crate::heap::CycleWeak::Reactive(Arc::downgrade(&node))) { return error; }
        self.next_reactive_id += 1;
        out.reactive = Some(node);
        LanaError::Ok
    }

    /// Lift a binary operation over paths/possibilities, mirroring `lift_binary`
    /// in `vm/c/vm.c`.
    fn lift_binary(&mut self, left: &Value, right: &Value, kind: PureKind, operation: u32, out: &mut Value) -> LanaError {
        let error = self.lift_binary_raw(
            &self.reactive_value(left),
            &self.reactive_value(right),
            kind,
            operation,
            out,
        );
        if error != LanaError::Ok || (left.reactive.is_none() && right.reactive.is_none()) {
            return error;
        }
        self.reactive_derived_value(
            left,
            Some(right),
            if kind == PureKind::Compare { ReactiveKind::Compare } else { ReactiveKind::Binary },
            operation,
            out,
        )
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
        if matches!(left.kind, ValueKind::Tensor(_)) || matches!(right.kind, ValueKind::Tensor(_))
            || matches!(left.kind, ValueKind::Map(_)) || matches!(right.kind, ValueKind::Map(_))
        {
            if kind != PureKind::Binary {
                return LanaError::Type;
            }
            // LIP-027: a scalar operand adopts the tensor's dtype.
            if matches!(left.kind, ValueKind::Tensor(_)) && matches!(right.kind, ValueKind::Number(_)) {
                let ValueKind::Tensor(t) = &left.kind else { unreachable!() };
                let ValueKind::Number(s) = &right.kind else { unreachable!() };
                let alloc = self.heap.clone();
                let t = match tensor::tensor_elementwise_scalar(&alloc, t, *s, operation) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *out = Value::tensor(Arc::new(t));
                return LanaError::Ok;
            }
            if matches!(left.kind, ValueKind::Number(_)) && matches!(right.kind, ValueKind::Tensor(_)) {
                let ValueKind::Number(s) = &left.kind else { unreachable!() };
                let ValueKind::Tensor(t) = &right.kind else { unreachable!() };
                let alloc = self.heap.clone();
                let t = match tensor::tensor_elementwise_scalar(&alloc, t, *s, operation) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *out = Value::tensor(Arc::new(t));
                return LanaError::Ok;
            }
            let (a_pred, a_var, a_unc) = match tensor::tensor_uncertainty_unpack(left) {
                Ok(v) => v,
                Err(error) => return error,
            };
            let (b_pred, b_var, b_unc) = match tensor::tensor_uncertainty_unpack(right) {
                Ok(v) => v,
                Err(error) => return error,
            };
            let alloc = self.heap.clone();
            if a_unc || b_unc {
                let a_var = match a_var {
                    Some(v) => v,
                    None => match tensor::tensor_zeros_like(&alloc, &a_pred) {
                        Ok(t) => Arc::new(t),
                        Err(error) => return error,
                    },
                };
                let b_var = match b_var {
                    Some(v) => v,
                    None => match tensor::tensor_zeros_like(&alloc, &b_pred) {
                        Ok(t) => Arc::new(t),
                        Err(error) => return error,
                    },
                };
                *out = match tensor::tensor_elementwise_uncertain(
                    &alloc, &a_pred, &a_var, &b_pred, &b_var, operation,
                ) {
                    Ok(v) => v,
                    Err(error) => return error,
                };
                return LanaError::Ok;
            }
            let t = match tensor::tensor_elementwise(&alloc, &a_pred, &b_pred, operation) {
                Ok(t) => t,
                Err(error) => return error,
            };
            *out = Value::tensor(Arc::new(t));
            return LanaError::Ok;
        }
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
            *out = Value::paths(match self.managed_payload(PathSet { alternatives, dependency_id }) { Ok(payload) => payload, Err(error) => return error });
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
            let source = left_possibility.as_ref().or(right_possibility.as_ref()).unwrap();
            let weights = left_possibility.as_ref().and_then(|value| value.weights.as_ref())
                .or_else(|| right_possibility.as_ref().and_then(|value| value.weights.as_ref()))
                .cloned();
            if let (Some(left), Some(right)) = (&left_possibility, &right_possibility) {
                if let (Some(left_weights), Some(right_weights)) = (&left.weights, &right.weights) {
                    if left_weights != right_weights {
                        return LanaError::UnsupportedOperation;
                    }
                }
            }
            *out = Value::possibility(match self.managed_payload(Possibility {
                values: results,
                weights,
                dependency_id: source.dependency_id,
            }) { Ok(payload) => payload, Err(error) => return error });
            return LanaError::Ok;
        }
        pure_scalar_binary(left, right, kind, operation, out)
    }

    /// Lift a unary operation over paths/possibilities, mirroring `lift_unary`
    /// in `vm/c/vm.c`.
    fn lift_unary(&mut self, source: &Value, operation: u32, out: &mut Value) -> LanaError {
        let current = self.reactive_value(source);
        let error = self.lift_unary_raw(&current, operation, out);
        if error != LanaError::Ok || source.reactive.is_none() {
            return error;
        }
        self.reactive_derived_value(source, None, ReactiveKind::Unary, operation, out)
    }

    fn lift_unary_raw(&mut self, source: &Value, operation: u32, out: &mut Value) -> LanaError {
        self.lift_pointwise(source, out, 0, &mut |_, source, out| {
            if matches!(source.kind, ValueKind::Number(_)) && operation == 0 {
                *out = Value::number(-source.as_number());
                LanaError::Ok
            } else if matches!(source.kind, ValueKind::Bool(_)) && operation == 1 {
                *out = Value::boolean(!source.as_bool());
                LanaError::Ok
            } else { LanaError::Type }
        })
    }

    fn lift_pointwise<F>(&mut self, source: &Value, out: &mut Value, depth: usize, map: &mut F) -> LanaError
    where F: FnMut(&mut Self, &Value, &mut Value) -> LanaError {
        if depth >= 64 { return LanaError::Limit; }
        if let Err(error) = self.charge_bounded_work(1) { return error; }
        match &source.kind {
            ValueKind::PathSet(paths) => {
                let mut alternatives = Vec::with_capacity(paths.alternatives.len());
                for alternative in &paths.alternatives {
                    let mut result = Value::null();
                    let error = self.lift_pointwise(&alternative.result, &mut result, depth + 1, map);
                    if error != LanaError::Ok {
                        return error;
                    }
                    alternatives.push(PathAlternative {
                        guard: alternative.guard,
                        weight: alternative.weight,
                        result,
                    });
                }
                *out = Value::paths(match self.managed_payload(PathSet {
                    alternatives,
                    dependency_id: paths.dependency_id,
                }) { Ok(payload) => payload, Err(error) => return error });
                LanaError::Ok
            }
            ValueKind::Possibility(possibility) => {
                let source_dependency_id = possibility.dependency_id;
                let mut results = Vec::with_capacity(possibility.values.len());
                for value in &possibility.values {
                    let mut result = Value::null();
                    let error = self.lift_pointwise(value, &mut result, depth + 1, map);
                    if error != LanaError::Ok {
                        return error;
                    }
                    results.push(result);
                }
                *out = Value::possibility(match self.managed_payload(Possibility {
                    values: results,
                    weights: possibility.weights.clone(),
                    dependency_id: source_dependency_id,
                }) { Ok(payload) => payload, Err(error) => return error });
                LanaError::Ok
            }
            _ => map(self, source, out),
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
    fn check_resolved(&self, value: &Value) -> Result<(), LanaError> {
        value.check_resolved(self.memory_limit.saturating_sub(self.allocated_bytes()))
    }

    /// Deep-clone a value with its reactive/claim/planned-effect metadata
    /// stripped, mirroring `clone_without_runtime_metadata` in `vm/c/vm.c`. The
    /// derivation is preserved.
    fn clone_without_runtime_metadata(&mut self, source: &Value) -> Result<Value, LanaError> {
        self.clone_without_runtime_metadata_memo(source, &mut DeepCloneMemo::default())
    }

    fn clone_without_runtime_metadata_memo(&mut self, source: &Value, memo: &mut DeepCloneMemo) -> Result<Value, LanaError> {
        let mut plain = self.reactive_value(source);
        plain.reactive = None;
        plain.claim = None;
        plain.planned_effect = None;
        self.deep_clone_value(&plain, memo)
    }

    /// Recursively materialize arrays and maps, mirroring `materialize_value`
    /// in `vm/c/vm.c`. Used by the write/stringify host calls so a reactive
    /// value's current contents are emitted.
    fn materialize_value(&mut self, source: &Value) -> Result<Value, LanaError> {
        self.materialize_value_memo(source, &mut DeepCloneMemo::default())
    }

    fn materialize_value_memo(&mut self, source: &Value, memo: &mut DeepCloneMemo) -> Result<Value, LanaError> {
        let current = self.reactive_value(source);
        match &current.kind {
            ValueKind::Array(array) => {
                let key = Arc::as_ptr(array) as usize;
                if let Some(copy) = memo.arrays.get(&key) {
                    return Ok(Value::array(copy.clone()));
                }
                let count = array.lock().unwrap().items.len();
                let items = self.allocate_array_items(count)?;
                let copy = Arc::new(Mutex::new(Array::from_buffer(items)?));
                let _ = Value::array(copy.clone());
                memo.arrays.insert(key, copy.clone());
                for index in 0..count {
                    let item = array.lock().unwrap().items[index].clone();
                    let item = self.materialize_value_memo(&item, memo)?;
                    copy.lock().unwrap().items.push(item)?;
                }
                Ok(Value::array(copy))
            }
            ValueKind::Map(map) => {
                let key = Arc::as_ptr(map) as usize;
                if let Some(copy) = memo.maps.get(&key) {
                    return Ok(Value::map(copy.clone()));
                }
                let copy = Arc::new(Mutex::new(Map::new(&self.heap, map.lock().unwrap().entries.len())?));
                let _ = Value::map(copy.clone());
                memo.maps.insert(key, copy.clone());
                let count = map.lock().unwrap().entries.len();
                for index in 0..count {
                    let entry = map.lock().unwrap().entries[index].clone();
                    let value = self.materialize_value_memo(&entry.value, memo)?;
                    copy.lock().unwrap().set(entry.key.clone(), value, true)?;
                }
                Ok(Value::map(copy))
            }
            ValueKind::Set(set) => {
                let key = Arc::as_ptr(set) as usize;
                if let Some(copy) = memo.sets.get(&key) {
                    return Ok(Value::set(copy.clone()));
                }
                let count = set.lock().unwrap().items.len();
                let copy = Arc::new(Mutex::new(Set::new(&self.heap, count)?));
                memo.sets.insert(key, copy.clone());
                for index in 0..count {
                    let item = set.lock().unwrap().items[index].clone();
                    let item = self.materialize_value_memo(&item, memo)?;
                    copy.lock().unwrap().items.push(item)?;
                }
                Ok(Value::set(copy))
            }
            _ => self.clone_without_runtime_metadata_memo(&current, memo),
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
        containers: &mut DeepCloneMemo,
    ) -> Result<Arc<Mutex<Reactive>>, LanaError> {
        let key = Arc::as_ptr(node) as usize;
        if let Some(existing) = memo.get(&key) {
            return Ok(existing.clone());
        }
        let (id, dependency_id, revision, kind, relationship, exactness, operation, input0, input1, constant0, constant1, current, history, is_training_data) = {
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
                guard.is_training_data,
            )
        };
        let copy = Arc::new(Mutex::new(Reactive {
            id, dependency_id, revision, kind, relationship, exactness, operation,
            inputs: [None, None], constants: [None, None], current: None,
            history: Vec::new(), is_training_data,
        }));
        self.track_cycle(crate::heap::CycleWeak::Reactive(Arc::downgrade(&copy)))?;
        memo.insert(key, copy.clone());
        let cloned_input0 = match &input0 {
            Some(input) => Some(self.deep_clone_reactive(input, memo, containers)?),
            None => None,
        };
        let cloned_input1 = match &input1 {
            Some(input) => Some(self.deep_clone_reactive(input, memo, containers)?),
            None => None,
        };
        let mut clone_plain = |vm: &mut Self, value: &Value| -> Result<Value, LanaError> {
            vm.clone_without_runtime_metadata_memo(value, containers)
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
        *copy.lock().unwrap() = Reactive {
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
            is_training_data,
        };
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
        self.deep_clone_live_value_memo(value, reactive_memo, &mut DeepCloneMemo::default())
    }

    fn deep_clone_live_value_memo(
        &mut self,
        value: &Value,
        reactive_memo: &mut HashMap<usize, Arc<Mutex<Reactive>>>,
        containers: &mut DeepCloneMemo,
    ) -> Result<Value, LanaError> {
        let mut cloned = Value {
            kind: ValueKind::Null,
            derivation: value.derivation.clone(),
            reactive: None,
            claim: value.claim.clone(),
            planned_effect: value.planned_effect.clone(),
        };
        if let Some(reactive) = &value.reactive {
            cloned.reactive = Some(self.deep_clone_reactive(reactive, reactive_memo, containers)?);
        }
        match &value.kind {
            ValueKind::Null
            | ValueKind::Number(_)
            | ValueKind::Bool(_)
            | ValueKind::Sample(_)
            | ValueKind::Function(_)
            | ValueKind::Distribution { .. }
            | ValueKind::Capability(_)
            | ValueKind::Tensor(_)
            | ValueKind::NQubitState(_)
            | ValueKind::Povm(_)
            | ValueKind::Channel(_)
            | ValueKind::Observable(_)
            | ValueKind::Lazy { .. }
            | ValueKind::Generator(_)
            | ValueKind::Future(_)
            | ValueKind::Regex(_)
            | ValueKind::Optimizer(_)
            | ValueKind::TrainingResult(_)
            | ValueKind::InferenceAlgorithm(_)
            | ValueKind::Posterior(_)
            | ValueKind::Dataset(_)
            | ValueKind::ObjectValue(_)
            | ValueKind::ClassObject(_) => cloned.kind = value.kind.clone(),
            ValueKind::String(string) => cloned.kind = ValueKind::String(self.clone_string(string, containers)?),
            ValueKind::State(state) => cloned.kind = ValueKind::State(state.clone()),
            ValueKind::Array(array) => {
                let key = Arc::as_ptr(array) as usize;
                if let Some(copy) = containers.arrays.get(&key) {
                    cloned.kind = ValueKind::Array(copy.clone());
                    return Ok(cloned);
                }
                let count = array.lock().unwrap().items.len();
                let items = self.allocate_array_items(count)?;
                let copy = Arc::new(Mutex::new(Array::from_buffer(items)?));
                let _ = Value::array(copy.clone());
                containers.arrays.insert(key, copy.clone());
                for index in 0..count {
                    let item = array.lock().unwrap().items[index].clone();
                    let item = self.deep_clone_live_value_memo(&item, reactive_memo, containers)?;
                    copy.lock().unwrap().items.push(item)?;
                }
                cloned.kind = ValueKind::Array(copy);
            }
            ValueKind::Map(map) => {
                let key = Arc::as_ptr(map) as usize;
                if let Some(copy) = containers.maps.get(&key) {
                    cloned.kind = ValueKind::Map(copy.clone());
                    return Ok(cloned);
                }
                let copy = Arc::new(Mutex::new(Map::new(&self.heap, 0)?));
                let _ = Value::map(copy.clone());
                containers.maps.insert(key, copy.clone());
                let count = map.lock().unwrap().entries.len();
                for index in 0..count {
                    let entry = map.lock().unwrap().entries[index].clone();
                    let value = self.deep_clone_live_value_memo(&entry.value, reactive_memo, containers)?;
                    copy.lock().unwrap().set(entry.key.clone(), value, true)?;
                }
                cloned.kind = ValueKind::Map(copy);
            }
            ValueKind::Possibility(possibility) => {
                let values = possibility
                    .values
                    .iter()
                    .map(|v| self.deep_clone_live_value_memo(v, reactive_memo, containers))
                    .collect::<Result<Vec<_>, _>>()?;
                cloned.kind = ValueKind::Possibility(self.managed_payload(Possibility {
                    values,
                    weights: possibility.weights.clone(),
                    dependency_id: possibility.dependency_id,
                })?);
            }
            ValueKind::PathSet(paths) => {
                let mut alternatives = Vec::with_capacity(paths.alternatives.len());
                for alternative in &paths.alternatives {
                    alternatives.push(PathAlternative {
                        guard: alternative.guard,
                        weight: alternative.weight,
                        result: self.deep_clone_live_value_memo(&alternative.result, reactive_memo, containers)?,
                    });
                }
                cloned.kind = ValueKind::PathSet(self.managed_payload(PathSet {
                    alternatives,
                    dependency_id: paths.dependency_id,
                })?);
            }
            ValueKind::Adt(adt) => {
                let fields = adt
                    .fields
                    .iter()
                    .map(|field| self.deep_clone_live_value_memo(field, reactive_memo, containers))
                    .collect::<Result<Vec<_>, _>>()?;
                cloned.kind = ValueKind::Adt(self.managed_payload(Adt { variant: adt.variant, fields })?);
            }
            ValueKind::Set(set) => {
                let key = Arc::as_ptr(set) as usize;
                if let Some(copy) = containers.sets.get(&key) {
                    cloned.kind = ValueKind::Set(copy.clone());
                    return Ok(cloned);
                }
                let count = set.lock().unwrap().items.len();
                let copy = Arc::new(Mutex::new(Set::new(&self.heap, count)?));
                let _ = Value::set(copy.clone());
                containers.sets.insert(key, copy.clone());
                for index in 0..count {
                    let item = set.lock().unwrap().items[index].clone();
                    let item = self.deep_clone_live_value_memo(&item, reactive_memo, containers)?;
                    copy.lock().unwrap().items.push(item)?;
                }
                cloned.kind = ValueKind::Set(copy);
            }
            ValueKind::Joint(_) | ValueKind::StateDist(_) | ValueKind::Kernel(_) | ValueKind::Network(_) => {
                cloned.kind = self.deep_clone_value(value, containers)?.kind;
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
            is_training_data: false,
        }));
        self.track_cycle(crate::heap::CycleWeak::Reactive(Arc::downgrade(&node)))?;
        let mut out = source.clone();
        out.reactive = Some(node);
        Ok(out)
    }

    /// Observe evidence against a reactive root, mirroring
    /// `lana_vm_reactive_observe`.
    fn reactive_observe(
        &mut self,
        source: &Value,
        evidence: &Value,
        scratch_register: u32,
    ) -> Result<Value, LanaError> {
        let Some(reactive) = &source.reactive else {
            return Err(LanaError::Format);
        };
        let is_training_data = {
            let node = reactive.lock().unwrap();
            if node.kind != ReactiveKind::Root {
                return Err(LanaError::Format);
            }
            node.is_training_data
        };
        if self.active_path_count > 1 {
            return Err(LanaError::UnresolvedValue);
        }
        let current = reactive
            .lock()
            .unwrap()
            .current
            .clone()
            .unwrap_or_else(Value::null);
        if self.chunk.version >= lana_bytecode::opcode::LABC_VERSION_5 && !is_training_data {
            let refined = self.core_condition(&current, evidence)?;
            self.reactive_recompute_transaction(reactive, &refined, scratch_register)?;
            self.observation_count += 1;
            return Ok(source.clone());
        }
        let replacement = self.reactive_value(evidence);
        self.check_resolved(&replacement)?;
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
            _ if is_training_data => {
                // LIP-010: a training data root's support is structural — a
                // single [x, target] observation.
                let ValueKind::Array(array) = &replacement.kind else {
                    return Err(LanaError::InvalidParameters);
                };
                if array.lock().unwrap().items.len() != 2 {
                    return Err(LanaError::InvalidParameters);
                }
            }
            _ => {
                if !joint_value_equal(&current, &replacement) {
                    return Err(LanaError::InvalidConditioning);
                }
            }
        }
        self.reactive_recompute_transaction(reactive, &replacement, scratch_register)?;
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
        let mut memo = DeepCloneMemo::default();
        let value = self.deep_clone_value(source, &mut memo)?;
        let claim = self.managed_payload(Claim {
            value,
            proposition: Arc::from(proposition),
            exactness,
            tolerance,
            source_valid,
        })?;
        let mut out = source.clone();
        out.claim = Some(claim);
        Ok(out)
    }

    /// Attach a planned effect to a value, mirroring `lana_vm_planned_effect`.
    fn planned_effect(&mut self, kind: &str, payload: &Value) -> Result<Value, LanaError> {
        if kind.is_empty() {
            return Err(LanaError::Format);
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
        self.track_cycle(crate::heap::CycleWeak::Effect(Arc::downgrade(&plan)))?;
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
        self.check_resolved(&plan.payload)?;
        plan.payload.check_capabilities(self.memory_limit.saturating_sub(self.allocated_bytes()))?;
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
            let capacity = state.receipts.len().checked_add(1).ok_or(LanaError::Oom)?;
            let growth = capacity.saturating_sub(state.receipts.capacity());
            self.reserve_cycle_edges(Arc::as_ptr(plan) as usize, capacity.checked_add(1).ok_or(LanaError::Oom)?,
                growth.checked_mul(std::mem::size_of::<EffectReceipt>()).ok_or(LanaError::Oom)?)?;
            state.receipts.try_reserve_exact(growth).map_err(|_| LanaError::Oom)?;
            self.heap.mutate_cycle(Arc::as_ptr(plan) as usize);
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
        scratch_register: u32,
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
                    ReactiveKind::Unary => self.lift_unary_raw(&left, operation, &mut staged_value),
                    ReactiveKind::ObjectMethod => self.lift_object_method_raw(&left, &right, operation, scratch_register, &mut staged_value),
                    ReactiveKind::Train => {
                        self.reactive_train_recompute(&node, &left, scratch_register, &mut staged_value)
                    }
                    _ => LanaError::UnsupportedOperation,
                }
            };
            if error != LanaError::Ok {
                return Err(error);
            }
            staged[index] = Some(staged_value);
        }
        // Admit every history extension before publishing any revision.
        for index in 0..count {
            if !affected[index] { continue; }
            let mut guard = list[index].lock().unwrap();
            let capacity = guard.history.len().checked_add(1).ok_or(LanaError::Oom)?;
            let growth = capacity.saturating_sub(guard.history.capacity());
            self.reserve_cycle_edges(Arc::as_ptr(&list[index]) as usize, capacity.checked_add(5).ok_or(LanaError::Oom)?,
                growth.checked_mul(std::mem::size_of::<ReactiveVersion>()).ok_or(LanaError::Oom)?)?;
            guard.history.try_reserve_exact(growth).map_err(|_| LanaError::Oom)?;
        }
        let revision = self.revision + 1;
        for index in 0..count {
            if !affected[index] {
                continue;
            }
            let mut guard = list[index].lock().unwrap();
            self.heap.mutate_cycle(Arc::as_ptr(&list[index]) as usize);
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

    fn set_host_call(&self, host: u32, arguments: &[Value]) -> Result<Value, LanaError> {
        if host == LANA_HOST_SET_NEW {
            if !arguments.is_empty() { return Err(LanaError::Type); }
            return Ok(Value::set(Arc::new(Mutex::new(Set::new(&self.heap, 0)?))));
        }
        if arguments.len() != 2 { return Err(LanaError::Type); }
        let ValueKind::Set(left) = &arguments[0].kind else { return Err(LanaError::Type); };
        if host == LANA_HOST_SET_ADD || host == LANA_HOST_SET_CONTAINS {
            if !value_is_set_member(&arguments[1]) { return Err(LanaError::Type); }
            let left = left.lock().unwrap();
            let found = left.items.iter().any(|item| set_value_equal(item, &arguments[1]));
            if host == LANA_HOST_SET_CONTAINS { return Ok(Value::boolean(found)); }
            let capacity = left.items.len().checked_add(usize::from(!found)).ok_or(LanaError::Oom)?;
            let mut result = Set::new(&self.heap, capacity)?;
            result.items.extend(left.items.iter().cloned())?;
            if !found { result.items.push(arguments[1].clone())?; }
            return Ok(Value::set(Arc::new(Mutex::new(result))));
        }
        let ValueKind::Set(right) = &arguments[1].kind else { return Err(LanaError::Type); };
        // Release the left lock before taking the right lock: operands can alias.
        let left = crate::heap::Buffer::from_slice(&self.heap, &left.lock().unwrap().items)?;
        let right = right.lock().unwrap();
        let mut result = Set::new(&self.heap, left.len())?;
        for item in &left {
            let present = right.items.iter().any(|other| set_value_equal(item, other));
            if host == LANA_HOST_SET_UNION || (host == LANA_HOST_SET_INTERSECT) == present {
                result.items.push(item.clone())?;
            }
        }
        if host == LANA_HOST_SET_UNION {
            for item in &right.items {
                if !result.items.iter().any(|other| set_value_equal(item, other)) {
                    result.items.push(item.clone())?;
                }
            }
        }
        Ok(Value::set(Arc::new(Mutex::new(result))))
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
        #[cfg(target_arch = "wasm32")]
        if matches!(host_id, LANA_HOST_HTTP_GET | LANA_HOST_HTTP_POST | LANA_HOST_NOW
            | LANA_HOST_SOCKET_CONNECT | LANA_HOST_SOCKET_SEND | LANA_HOST_SOCKET_RECV | LANA_HOST_SOCKET_CLOSE
            | LANA_HOST_FFI_LOAD | LANA_HOST_FFI_CALL | LANA_HOST_SLEEP
            | LANA_HOST_SHARED_WAIT | LANA_HOST_DIRECTORY_CREATE | LANA_HOST_DIRECTORY_LIST
            | LANA_HOST_CSV_READ | LANA_HOST_CSV_WRITE)
            || (self.virtual_fs.is_none() && matches!(host_id, LANA_HOST_READ_TEXT
                | LANA_HOST_WRITE_TEXT | LANA_HOST_WRITE_TEXT_ATOMIC
                | LANA_HOST_PATH_RESOLVE | LANA_HOST_PATH_EXISTS)) {
            return LanaError::UnsupportedOperation;
        }
        match host_id {
            LANA_HOST_INFORMATION_SNAPSHOT => {
                if argc != 1 { return LanaError::Type; }
                match self.information_snapshot(&arguments[0]) {
                    Ok(value) => { *out = value; LanaError::Ok },
                    Err(error) => error,
                }
            }
            LANA_HOST_ARGS => {
                if argc != 0 {
                    return LanaError::Type;
                }
                let mut items = match self.allocate_array_items(self.program_argc) {
                    Ok(items) => items, Err(error) => return error,
                };
                for index in 0..self.program_argc {
                    let source = Value::string(self.program_argv[index].clone());
                    let mut memo = DeepCloneMemo::default();
                    let item = match self.deep_clone_value(&source, &mut memo) {
                        Ok(value) => value,
                        Err(error) => return error,
                    };
                    if let Err(error) = items.push(item) { return error; }
                }
                *out = Value::array(Arc::new(Mutex::new(match Array::from_buffer(items) { Ok(array) => array, Err(error) => return error })));
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
            LANA_HOST_SURPRISAL => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let ValueKind::Number(probability) = arguments[0].kind else {
                    return LanaError::Type;
                };
                if probability < 0.0 {
                    return LanaError::InvalidParameters;
                }
                let result = -probability.ln();
                // Normalize -0.0 to 0.0 so surprisal(1.0) prints as "0" in both
                // VMs (C11's %.17g renders -0.0 as "-0").
                let result = if result == 0.0 { 0.0 } else { result };
                *out = Value::number(result);
                LanaError::Ok
            }
            LANA_HOST_CORE_ENTROPY | LANA_HOST_CORE_CONDITIONAL_ENTROPY | LANA_HOST_CORE_MUTUAL_INFORMATION => {
                self.core_information_measure(host_id, arguments, out)
            }
            LANA_HOST_CORE_BROJA => self.core_broja(arguments, out),
            LANA_HOST_CORE_FORGET_WEIGHTS | LANA_HOST_CORE_ASSIGN_WEIGHTS => {
                self.core_convert_weights(host_id, arguments, out)
            }
            LANA_HOST_CORE_IDENTITY_KERNEL => self.core_kernel_identity(arguments, out),
            LANA_HOST_CORE_COMPOSE_KERNELS => self.core_kernel_compose(arguments, out),
            LANA_HOST_CORE_NETWORK => self.core_network(arguments, out),
            LANA_HOST_CORE_INFER => self.core_infer(arguments, out),
            LANA_HOST_NOW => {
                if argc != 0 {
                    return LanaError::Type;
                }
                // The WASM dispatch boundary rejects this host call before any
                // clock access. Only native execution reaches this branch.
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
            LANA_HOST_TENSOR_ALLOC => {
                // Arguments: shape array (VAL_ARRAY), optional is_complex (VAL_BOOL).
                if argc < 1 || argc > 2 {
                    return LanaError::Type;
                }
                let mut is_complex = false;
                if argc == 2 {
                    let ValueKind::Bool(b) = arguments[1].kind else {
                        return LanaError::Type;
                    };
                    is_complex = b;
                }
                let alloc = self.heap.clone();
                let shape = match tensor::tensor_shape_from_array(&alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let t = match tensor::tensor_new(&alloc, shape.len(), &shape, is_complex) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_ZEROS => {
                if argc != 1 && argc != 2 {
                    return LanaError::Type;
                }
                let dtype = match tensor_optional_dtype(&arguments, argc) {
                    Some(d) => d,
                    None => return LanaError::InvalidParameters,
                };
                let alloc = self.heap.clone();
                let shape = match tensor::tensor_shape_from_array(&alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&alloc, shape.len(), &shape, false) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                t.dtype = dtype;
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_ONES => {
                if argc != 1 && argc != 2 {
                    return LanaError::Type;
                }
                let dtype = match tensor_optional_dtype(&arguments, argc) {
                    Some(d) => d,
                    None => return LanaError::InvalidParameters,
                };
                let alloc = self.heap.clone();
                let shape = match tensor::tensor_shape_from_array(&alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&alloc, shape.len(), &shape, false) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                t.dtype = dtype;
                let mut total: usize = 1;
                for i in 0..t.ndim {
                    total *= t.shape[i];
                }
                for i in 0..total {
                    tensor_set_real(&mut t, i, 1.0);
                }
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_EYE => {
                if argc != 1 && argc != 2 {
                    return LanaError::Type;
                }
                let ValueKind::Number(d) = arguments[0].kind else {
                    return LanaError::Type;
                };
                let dtype = match tensor_optional_dtype(&arguments, argc) {
                    Some(d) => d,
                    None => return LanaError::InvalidParameters,
                };
                let n = match tensor::tensor_dimension(d) {
                    Ok(n) => n,
                    Err(error) => return error,
                };
                let shape = [n, n];
                let alloc = self.heap.clone();
                let mut t = match tensor::tensor_new(&alloc, 2, &shape, false) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                t.dtype = dtype;
                for i in 0..n {
                    tensor_set_real(&mut t, i * n + i, 1.0);
                }
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_DTYPE => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(t) = &arguments[0].kind else {
                    return LanaError::Type;
                };
                *out = Value::string(Arc::from(t.dtype.as_str()));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_CAST => {
                if argc != 2 {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(t) = &arguments[0].kind else {
                    return LanaError::Type;
                };
                let ValueKind::String(s) = &arguments[1].kind else {
                    return LanaError::Type;
                };
                let Some(dtype) = TensorDtype::from_str(s) else {
                    return LanaError::InvalidParameters;
                };
                let alloc = self.heap.clone();
                match tensor::tensor_cast(&alloc, t, dtype) {
                    Ok(t) => {
                        *out = Value::tensor(Arc::new(t));
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_RESHAPE => {
                if argc != 2 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                let result = { let alloc = self.heap.clone(); tensor::tensor_reshape(&alloc, source, &arguments[1]) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_some() { self.ad_record(10, &arguments[0], None, -1, out) } else { LanaError::Ok }
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_TRANSPOSE => {
                if argc != 1 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                let result = { let alloc = self.heap.clone(); tensor::tensor_transpose(&alloc, source) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_some() { self.ad_record(11, &arguments[0], None, -1, out) } else { LanaError::Ok }
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_EXP | LANA_HOST_TENSOR_LOG | LANA_HOST_TENSOR_SQRT | LANA_HOST_TENSOR_RELU => {
                if argc != 1 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                let op = host_id - LANA_HOST_TENSOR_EXP;
                let result = { let alloc = self.heap.clone(); tensor::tensor_unary_math(&alloc, source, op) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_some() { self.ad_record(12 + op as i32, &arguments[0], None, -1, out) } else { LanaError::Ok }
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_SOFTMAX | LANA_HOST_TENSOR_LOGSUMEXP => {
                if argc != 1 && argc != 2 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                let axis_value = if argc == 2 { Some(&arguments[1]) } else { None };
                let result = { let alloc = self.heap.clone(); tensor::tensor_softmax_like(&alloc, source, axis_value, host_id == LANA_HOST_TENSOR_LOGSUMEXP) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_none() { return LanaError::Ok; }
                        let mut axis_number = match axis_value { None => -1.0, Some(v) => v.as_number() };
                        if axis_number < 0.0 { axis_number += source.ndim as f64; }
                        self.ad_record(if host_id == LANA_HOST_TENSOR_LOGSUMEXP { 17 } else { 16 }, &arguments[0], None, axis_number as i32, out)
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_ARGMAX => {
                if argc != 1 && argc != 2 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                let alloc = self.heap.clone();
                match tensor::tensor_argmax(&alloc, source, if argc == 2 { Some(&arguments[1]) } else { None }) {
                    Ok(value) => { *out = value; LanaError::Ok }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_COMPARE => {
                if argc != 3 { return LanaError::Type; }
                let (ValueKind::Tensor(a), ValueKind::Tensor(b), ValueKind::String(operation)) =
                    (&arguments[0].kind, &arguments[1].kind, &arguments[2].kind) else { return LanaError::Type; };
                let alloc = self.heap.clone();
                match tensor::tensor_compare_values(&alloc, a, b, operation) {
                    Ok(value) => { *out = value; LanaError::Ok }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_SELECT => {
                if argc != 3 { return LanaError::Type; }
                let (ValueKind::Tensor(mask), ValueKind::Tensor(yes), ValueKind::Tensor(no)) =
                    (&arguments[0].kind, &arguments[1].kind, &arguments[2].kind) else { return LanaError::Type; };
                let result = { let alloc = self.heap.clone(); tensor::tensor_select_values(&alloc, mask, yes, no) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[1].derivation.is_none() && arguments[2].derivation.is_none() { return LanaError::Ok; }
                        let shape = match &out.kind { ValueKind::Tensor(t) => t.shape.clone(), _ => unreachable!() };
                        let alloc = self.heap.clone();
                        let mut yes_mask = match tensor::tensor_new_dtype(&alloc, shape.len(), &shape, yes.dtype) { Ok(t) => t, Err(error) => return error };
                        let mut no_mask = match tensor::tensor_new_dtype(&alloc, shape.len(), &shape, no.dtype) { Ok(t) => t, Err(error) => return error };
                        for i in 0..shape.iter().product() {
                            let selected = if tensor_get_real(mask, tensor::broadcast_index(mask, &shape, i)) != 0.0 { 1.0 } else { 0.0 };
                            tensor_set_real(&mut yes_mask, i, selected); tensor_set_real(&mut no_mask, i, 1.0 - selected);
                        }
                        drop(alloc);
                        let yes_mask_value = Value::tensor(Arc::new(match yes_mask.try_clone(&self.heap) { Ok(t) => t, Err(error) => return error }));
                        let no_mask_value = Value::tensor(Arc::new(match no_mask.try_clone(&self.heap) { Ok(t) => t, Err(error) => return error }));
                        let mut yes_product = { let alloc = self.heap.clone(); match tensor::tensor_elementwise(&alloc, yes, &yes_mask, 2) { Ok(t) => Value::tensor(Arc::new(t)), Err(error) => return error } };
                        if arguments[1].derivation.is_some() { let error = self.ad_record(2, &arguments[1], Some(&yes_mask_value), -1, &mut yes_product); if error != LanaError::Ok { return error; } }
                        let mut no_product = { let alloc = self.heap.clone(); match tensor::tensor_elementwise(&alloc, no, &no_mask, 2) { Ok(t) => Value::tensor(Arc::new(t)), Err(error) => return error } };
                        if arguments[2].derivation.is_some() { let error = self.ad_record(2, &arguments[2], Some(&no_mask_value), -1, &mut no_product); if error != LanaError::Ok { return error; } }
                        self.ad_record(0, &yes_product, Some(&no_product), -1, out)
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_GATHER => {
                if argc != 3 { return LanaError::Type; }
                let (ValueKind::Tensor(source), ValueKind::Tensor(indices)) =
                    (&arguments[0].kind, &arguments[1].kind) else { return LanaError::Type; };
                let result = { let alloc = self.heap.clone(); tensor::tensor_gather_values(&alloc, source, indices, &arguments[2]) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_none() { return LanaError::Ok; }
                        let mut axis = arguments[2].as_number(); if axis < 0.0 { axis += source.ndim as f64; }
                        self.ad_record(18, &arguments[0], Some(&arguments[1]), axis as i32, out)
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_CHOLESKY_SOLVE => {
                if argc != 2 { return LanaError::Type; }
                let (ValueKind::Tensor(matrix), ValueKind::Tensor(rhs)) =
                    (&arguments[0].kind, &arguments[1].kind) else { return LanaError::Type; };
                let result = { let alloc = self.heap.clone(); tensor::tensor_cholesky_solve(&alloc, matrix, rhs) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_some() || arguments[1].derivation.is_some() { self.ad_record(19, &arguments[0], Some(&arguments[1]), -1, out) } else { LanaError::Ok }
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_RANDOM_UNIFORM | LANA_HOST_RANDOM_NORMAL => {
                if argc != 2 { return LanaError::Type; }
                let alloc = self.heap.clone();
                match tensor::tensor_random(&alloc, &arguments[0], &arguments[1], host_id == LANA_HOST_RANDOM_NORMAL) {
                    Ok(value) => { *out = value; LanaError::Ok }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_DEVICE => {
                if argc != 1 { return LanaError::Type; }
                let ValueKind::Tensor(tensor) = &arguments[0].kind else { return LanaError::Type; };
                *out = Value::string(Arc::from(if tensor.device == TensorDevice::Metal { "metal" } else { "cpu" }));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_TO_DEVICE => {
                if argc != 3 { return LanaError::Type; }
                let (ValueKind::Tensor(source), ValueKind::String(device), ValueKind::Capability(token)) =
                    (&arguments[0].kind, &arguments[1].kind, &arguments[2].kind) else { return LanaError::Type; };
                let target = match &**device { "cpu" => TensorDevice::Cpu, "metal" => TensorDevice::Metal, _ => return LanaError::InvalidParameters };
                if target == TensorDevice::Metal {
                    let shared = &token.shared;
                    let named_gpu = matches!(&shared.base_snapshot.kind, ValueKind::String(name) if &**name == "gpu");
                    if !named_gpu || !capability_allows_locked(shared, token, LANA_CAPABILITY_READ) { return LanaError::Capability; }
                    if source.dtype == TensorDtype::F64 || source.is_complex || !crate::metal::metal_available() { return LanaError::UnsupportedOperation; }
                }
                let result = self.tensor_copy_contiguous(source);
                match result { Ok(mut tensor) => {
                    if target == TensorDevice::Metal {
                        let charge = match self.heap.reserve(source.data.len()) { Ok(charge) => charge, Err(error) => return error };
                        let Some(buffer) = crate::metal::ResidentBuffer::new(&tensor.data) else { return LanaError::UnsupportedOperation; };
                        let tensor_mut = Arc::get_mut(&mut tensor).expect("new tensor is unique");
                        tensor_mut.metal_buffer = Some(Arc::new(buffer));
                        tensor_mut.metal_charge = Some(Arc::new(charge));
                        tensor_mut.device = target;
                    } else { Arc::get_mut(&mut tensor).expect("new tensor is unique").device = target; }
                    *out = Value::tensor(tensor); LanaError::Ok
                }, Err(error) => error }
            }
            LANA_HOST_TENSOR_TO_CPU => {
                if argc != 1 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                match self.tensor_copy_contiguous(source) { Ok(mut tensor) => {
                    let tensor_mut = Arc::get_mut(&mut tensor).expect("new tensor is unique");
                    if let Some(buffer) = &source.metal_buffer {
                        buffer.copy_into(Arc::get_mut(&mut tensor_mut.data).expect("new tensor buffer is unique"));
                    }
                    tensor_mut.metal_buffer = None;
                    tensor_mut.metal_charge = None;
                    tensor_mut.device = TensorDevice::Cpu;
                    *out = Value::tensor(tensor); LanaError::Ok
                }, Err(error) => error }
            }
            LANA_HOST_TENSOR_SHAPE => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(t) = &arguments[0].kind else {
                    return LanaError::Type;
                };
                let mut items = match self.allocate_array_items(t.shape.len()) {
                    Ok(items) => items, Err(error) => return error,
                };
                if let Err(error) = items.extend(t.shape.iter().map(|&dim| Value::number(dim as f64))) { return error; }
                *out = Value::array(Arc::new(Mutex::new(match Array::from_buffer(items) { Ok(array) => array, Err(error) => return error })));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_NDIM => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(t) = &arguments[0].kind else {
                    return LanaError::Type;
                };
                *out = Value::number(t.ndim as f64);
                LanaError::Ok
            }
            LANA_HOST_TENSOR_ADD | LANA_HOST_TENSOR_SUB | LANA_HOST_TENSOR_MUL | LANA_HOST_TENSOR_DIV => {
                if argc != 2 {
                    return LanaError::Type;
                }
                let op = host_id - LANA_HOST_TENSOR_ADD;
                let (a_pred, a_var, a_unc) = match tensor::tensor_uncertainty_unpack(&arguments[0]) {
                    Ok(x) => x,
                    Err(error) => return error,
                };
                let (b_pred, b_var, b_unc) = match tensor::tensor_uncertainty_unpack(&arguments[1]) {
                    Ok(x) => x,
                    Err(error) => return error,
                };
                let alloc = self.heap.clone();
                if a_unc || b_unc {
                    let a_var = match a_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&alloc, &a_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    let b_var = match b_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&alloc, &b_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    return match tensor::tensor_elementwise_uncertain(
                        &alloc, &a_pred, &a_var, &b_pred, &b_var, op,
                    ) {
                        Ok(value) => {
                            *out = value;
                            LanaError::Ok
                        }
                        Err(error) => error,
                    };
                }
                let t = match tensor::tensor_elementwise(&alloc, &a_pred, &b_pred, op) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *out = Value::tensor(Arc::new(t));
                if self.ad_recording {
                    return self.ad_record(op as i32, &arguments[0], Some(&arguments[1]), -1, out);
                }
                LanaError::Ok
            }
            LANA_HOST_TENSOR_MATMUL => {
                if argc != 2 && argc != 3 {
                    return LanaError::Type;
                }
                if argc == 3 && !matches!(arguments[2].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let (a_pred, a_var, a_unc) = match tensor::tensor_uncertainty_unpack(&arguments[0]) {
                    Ok(x) => x,
                    Err(error) => return error,
                };
                let (b_pred, b_var, b_unc) = match tensor::tensor_uncertainty_unpack(&arguments[1]) {
                    Ok(x) => x,
                    Err(error) => return error,
                };
                let alloc = self.heap.clone();
                if a_unc || b_unc {
                    let a_var = match a_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&alloc, &a_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    let b_var = match b_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&alloc, &b_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    return match tensor::tensor_matmul_uncertain(
                        &alloc, &a_pred, &a_var, &b_pred, &b_var,
                    ) {
                        Ok(value) => {
                            *out = value;
                            LanaError::Ok
                        }
                        Err(error) => error,
                    };
                }
                // LIP-027: optional out_dtype: named parameter. Defaults to the
                // input dtype when both match, else the higher-precision operand.
                let out_dtype = if argc == 3 {
                    let ValueKind::String(s) = &arguments[2].kind else {
                        return LanaError::Type;
                    };
                    match TensorDtype::from_str(s) {
                        Some(d) => d,
                        None => return LanaError::InvalidParameters,
                    }
                } else {
                    tensor::matmul_default_dtype(&a_pred, &b_pred)
                };
                let t = match tensor::tensor_matmul(&alloc, &a_pred, &b_pred, out_dtype) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *out = Value::tensor(Arc::new(t));
                if self.ad_recording {
                    return self.ad_record(4, &arguments[0], Some(&arguments[1]), -1, out);
                }
                LanaError::Ok
            }
            LANA_HOST_GPU_MATMUL => {
                if argc != 3 {
                    return LanaError::Type;
                }
                let (ValueKind::Tensor(a), ValueKind::Tensor(b), ValueKind::String(precision)) =
                    (&arguments[0].kind, &arguments[1].kind, &arguments[2].kind)
                else {
                    return LanaError::Type;
                };
                if &**precision != "float32" {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                let t = match tensor::tensor_gpu_matmul(&alloc, a, b) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_SUM | LANA_HOST_TENSOR_MEAN | LANA_HOST_TENSOR_MAX | LANA_HOST_TENSOR_MIN => {
                if argc != 1 && argc != 2 {
                    return LanaError::Type;
                }
                let op = host_id - LANA_HOST_TENSOR_SUM;
                let (pred, var, unc) = match tensor::tensor_uncertainty_unpack(&arguments[0]) {
                    Ok(x) => x,
                    Err(error) => return error,
                };
                let alloc = self.heap.clone();
                if unc {
                    if op >= 2 {
                        return LanaError::Type;
                    }
                    let var = var.expect("uncertain tensor has a variance");
                    let axis = if argc == 2 { Some(&arguments[1]) } else { None };
                    return match tensor::tensor_reduce_uncertain(&alloc, &pred, &var, op, axis) {
                        Ok(value) => {
                            *out = value;
                            LanaError::Ok
                        }
                        Err(error) => error,
                    };
                }
                let result = if argc == 2 {
                    tensor::tensor_reduce_axis(&alloc, &pred, op, &arguments[1])
                } else {
                    tensor::tensor_reduce(&alloc, &pred, op)
                };
                match result {
                    Ok(value) => {
                        *out = value;
                        if self.ad_recording && op < 2 {
                            let mut ad_axis = -1;
                            if argc == 2 {
                                let axis_number = arguments[1].as_number();
                                let ndim = pred.ndim;
                                ad_axis = if axis_number < 0.0 {
                                    (axis_number + ndim as f64) as i32
                                } else {
                                    axis_number as i32
                                };
                            }
                            return self.ad_record(5 + op as i32, &arguments[0], None, ad_axis, out);
                        }
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR => {
                if argc != 1 && argc != 2 {
                    return LanaError::Type;
                }
                let dtype = match tensor_optional_dtype(&arguments, argc) {
                    Some(d) => d,
                    None => return LanaError::InvalidParameters,
                };
                let alloc = self.heap.clone();
                let shape = match tensor::tensor_infer_shape(&alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&alloc, shape.len(), &shape, false) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                t.dtype = dtype;
                let mut offset = 0usize;
                if let Err(error) = tensor::tensor_fill_data(&arguments[0], &mut t, &mut offset) {
                    return error;
                }
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_COMPLEX => {
                if argc != 2 {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                let shape = match tensor::tensor_infer_shape(&alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&alloc, shape.len(), &shape, true) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                let mut offset = 0usize;
                if let Err(error) =
                    tensor::tensor_fill_complex(&arguments[0], &arguments[1], &mut t, &mut offset)
                {
                    return error;
                }
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_DENSITY_OPERATOR => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                match &arguments[0].kind {
                    ValueKind::State(state) => {
                        linalg_density_from_state(&alloc, &state.state)
                    }
                    ValueKind::Tensor(t) => linalg_density_from_tensor(&alloc, t),
                    _ => Err(LanaError::Type),
                }
                .map(|value| {
                    *out = value;
                    LanaError::Ok
                })
                .unwrap_or_else(|error| error)
            }
            LANA_HOST_POVM => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                match linalg_povm(&alloc, &arguments[0]) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_CHANNEL => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                match linalg_channel(&alloc, &arguments[0]) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_OBSERVABLE => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                match linalg_observable(&alloc, &arguments[0]) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_PRODUCT => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::NQubitState(_))
                    || !matches!(arguments[1].kind, ValueKind::NQubitState(_))
                {
                    return LanaError::Type;
                }
                let (ValueKind::NQubitState(a), ValueKind::NQubitState(b)) =
                    (&arguments[0].kind, &arguments[1].kind)
                else {
                    unreachable!()
                };
                let alloc = self.heap.clone();
                match linalg_tensor_product(&alloc, a, b) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_PARTIAL_TRACE => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::NQubitState(_))
                    || !matches!(arguments[1].kind, ValueKind::Number(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::NQubitState(ab) = &arguments[0].kind else {
                    unreachable!()
                };
                let ValueKind::Number(subsystem) = arguments[1].kind else {
                    unreachable!()
                };
                let alloc = self.heap.clone();
                match linalg_partial_trace(&alloc, ab, subsystem) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_MEASURE_WITH => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::NQubitState(_))
                    || !matches!(arguments[1].kind, ValueKind::Povm(_))
                {
                    return LanaError::Type;
                }
                let (ValueKind::NQubitState(rho), ValueKind::Povm(povm)) =
                    (&arguments[0].kind, &arguments[1].kind)
                else {
                    unreachable!()
                };
                let heap = self.heap.clone();
                match linalg_measure_with(&heap, rho, povm) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_APPLY_TO => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Channel(_))
                    || !matches!(arguments[1].kind, ValueKind::NQubitState(_))
                {
                    return LanaError::Type;
                }
                let (ValueKind::Channel(chan), ValueKind::NQubitState(rho)) =
                    (&arguments[0].kind, &arguments[1].kind)
                else {
                    unreachable!()
                };
                let alloc = self.heap.clone();
                match linalg_apply_to(&alloc, chan, rho) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_EXPECT => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::NQubitState(_))
                    || !matches!(arguments[1].kind, ValueKind::Observable(_))
                {
                    return LanaError::Type;
                }
                let (ValueKind::NQubitState(rho), ValueKind::Observable(obs)) =
                    (&arguments[0].kind, &arguments[1].kind)
                else {
                    unreachable!()
                };
                *out = linalg_expect(rho, obs);
                LanaError::Ok
            }
            LANA_HOST_MIX => {
                if argc != 3
                    || !matches!(arguments[0].kind, ValueKind::NQubitState(_))
                    || !matches!(arguments[1].kind, ValueKind::NQubitState(_))
                    || !matches!(arguments[2].kind, ValueKind::Number(_))
                {
                    return LanaError::Type;
                }
                let (ValueKind::NQubitState(a), ValueKind::NQubitState(b)) =
                    (&arguments[0].kind, &arguments[1].kind)
                else {
                    unreachable!()
                };
                let ValueKind::Number(w) = arguments[2].kind else {
                    unreachable!()
                };
                if !w.is_finite() || w < 0.0 || w > 1.0 {
                    return LanaError::InvalidParameters;
                }
                let alloc = self.heap.clone();
                match linalg_mix(&alloc, a, b, w) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TRACE_DISTANCE => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::NQubitState(_))
                    || !matches!(arguments[1].kind, ValueKind::NQubitState(_))
                {
                    return LanaError::Type;
                }
                let (ValueKind::NQubitState(a), ValueKind::NQubitState(b)) =
                    (&arguments[0].kind, &arguments[1].kind)
                else {
                    unreachable!()
                };
                let alloc = self.heap.clone();
                match linalg_trace_distance(&alloc, a, b) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_IS_SEPARABLE => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::NQubitState(_))
                    || !matches!(arguments[1].kind, ValueKind::Number(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::NQubitState(ab) = &arguments[0].kind else {
                    unreachable!()
                };
                let ValueKind::Number(bipartition) = arguments[1].kind else {
                    unreachable!()
                };
                let alloc = self.heap.clone();
                match linalg_is_separable(&alloc, ab, bipartition) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TO_STATE => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::NQubitState(_)) {
                    return LanaError::Type;
                }
                let ValueKind::NQubitState(rho) = &arguments[0].kind else {
                    unreachable!()
                };
                match linalg_to_state(rho) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_STATE_TENSOR => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                match linalg_state_tensor(&alloc, &arguments[0]) {
                    Ok(value) => {
                        *out = value;
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_APPEND => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Tensor(_))
                    || !matches!(arguments[1].kind, ValueKind::Tensor(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(a) = &arguments[0].kind else {
                    unreachable!()
                };
                let ValueKind::Tensor(b) = &arguments[1].kind else {
                    unreachable!()
                };
                if !a.is_state || !b.is_state {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                match linalg_state_append(&alloc, a, b) {
                    Ok(value) => {
                        *out = value;
                        if self.ad_recording {
                            return self.ad_record(7, &arguments[0], Some(&arguments[1]), -1, out);
                        }
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_MEASURE => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Tensor(_))
                    || !matches!(arguments[1].kind, ValueKind::Povm(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(s) = &arguments[0].kind else {
                    unreachable!()
                };
                let ValueKind::Povm(povm) = &arguments[1].kind else {
                    unreachable!()
                };
                if !s.is_state {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                match linalg_state_measure(&alloc, s, povm) {
                    Ok(value) => {
                        *out = value;
                        if self.ad_recording {
                            return self.ad_record(8, &arguments[0], Some(&arguments[1]), -1, out);
                        }
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TRANSFORM => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Tensor(_))
                    || !matches!(arguments[1].kind, ValueKind::Channel(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(s) = &arguments[0].kind else {
                    unreachable!()
                };
                let ValueKind::Channel(chan) = &arguments[1].kind else {
                    unreachable!()
                };
                if !s.is_state {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                match linalg_state_transform(&alloc, s, chan) {
                    Ok(value) => {
                        *out = value;
                        if self.ad_recording {
                            return self.ad_record(9, &arguments[0], Some(&arguments[1]), -1, out);
                        }
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
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
                let mut map = match self.allocate_with_collection(|vm| Map::new(&vm.heap, argc / 2)) {
                    Ok(map) => map, Err(error) => return error,
                };
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
                if let Err(error) = self.prepare_container_write(&arguments[0], &arguments[2]) { return error; }
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
                let map = map.lock().unwrap();
                let mut items = match self.allocate_array_items(map.entries.len()) {
                    Ok(items) => items, Err(error) => return error,
                };
                if let Err(error) = items.extend(map.entries.iter().map(|entry| Value::string(entry.key.clone()))) { return error; }
                *out = Value::array(Arc::new(Mutex::new(match Array::from_buffer(items) { Ok(array) => array, Err(error) => return error })));
                LanaError::Ok
            }
            LANA_HOST_INDEX_GET => {
                if argc != 2 {
                    return LanaError::Type;
                }
                if matches!(arguments[0].kind, ValueKind::Tensor(_)) {
                    let ValueKind::Tensor(tensor) = &arguments[0].kind else {
                        unreachable!()
                    };
                    let alloc = self.heap.clone();
                    return match tensor::tensor_index(&alloc, tensor, &arguments[1]) {
                        Ok(value) => {
                            *out = value;
                            LanaError::Ok
                        }
                        Err(error) => error,
                    };
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
                    if let Err(error) = self.prepare_container_write(&arguments[0], &arguments[2]) { return error; }
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
                    if index >= array.lock().unwrap().items.len() { return LanaError::Limit; }
                    if let Err(error) = self.prepare_container_write(&arguments[0], &arguments[2]) { return error; }
                    let mut array = array.lock().unwrap();
                    if index < array.items.len() {
                        if array.frozen { return LanaError::UnsupportedOperation; }
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
                self.string_output(&text[start..end], out)
            }
            LANA_HOST_STRING_CONCAT => {
                let mut total = 0usize;
                for argument in arguments {
                    if !matches!(argument.kind, ValueKind::String(_)) {
                        return LanaError::Type;
                    }
                    total = match total.checked_add(argument.as_string().len()) {
                        Some(total) => total, None => return LanaError::Oom,
                    };
                }
                let mut joined = match Buffer::new(&self.heap, total, 0) {
                    Ok(buffer) => buffer, Err(error) => return error,
                };
                for argument in arguments {
                    if let Err(error) = joined.extend_from_slice(argument.as_string().as_bytes()) { return error; }
                }
                self.string_output(std::str::from_utf8(&joined).unwrap(), out)
            }
            LANA_HOST_NUMBER_TO_STRING => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::Number(_)) {
                    return LanaError::Type;
                }
                self.string_output(&format_g17(arguments[0].as_number()), out)
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
                let value = self.allocate_with_collection(|vm| {
                    let mut items = vm.allocate_array_items(count)?;
                    items.resize(count, Value::null())?;
                    Ok(Value::array(Arc::new(Mutex::new(Array::from_buffer(items)?))))
                });
                *out = match value { Ok(value) => value, Err(error) => return error };
                LanaError::Ok
            }
            LANA_HOST_ARRAY_PUSH => {
                if argc != 2 || !matches!(arguments[0].kind, ValueKind::Array(_)) {
                    return LanaError::Type;
                }
                let ValueKind::Array(array) = &arguments[0].kind else {
                    unreachable!()
                };
                if let Err(error) = self.prepare_container_write(&arguments[0], &arguments[1]) { return error; }
                if let Err(error) = array.lock().unwrap().push(arguments[1].clone()) { return error; }
                *out = arguments[0].clone();
                LanaError::Ok
            }
            LANA_HOST_STRING_HEX => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                const DIGITS: &[u8; 16] = b"0123456789abcdef";
                let source = arguments[0].as_string();
                let Some(capacity) = source.len().checked_mul(2) else { return LanaError::Oom; };
                let mut hex = match Buffer::new(&self.heap, capacity, 0) {
                    Ok(buffer) => buffer, Err(error) => return error,
                };
                for byte in source.as_bytes() {
                    if let Err(error) = hex.extend_from_slice(&[DIGITS[(byte >> 4) as usize], DIGITS[(byte & 15) as usize]]) { return error; }
                }
                self.string_output(std::str::from_utf8(&hex).unwrap(), out)
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
                let mut total = match separator.len().checked_mul(array.items.len().saturating_sub(1)) {
                    Some(total) => total, None => return LanaError::Oom,
                };
                for item in &array.items {
                    if !matches!(item.kind, ValueKind::String(_)) {
                        return LanaError::Type;
                    }
                    total = match total.checked_add(item.as_string().len()) {
                        Some(total) => total, None => return LanaError::Oom,
                    };
                }
                let mut joined = match Buffer::new(&self.heap, total, 0) {
                    Ok(buffer) => buffer, Err(error) => return error,
                };
                for (index, item) in array.items.iter().enumerate() {
                    if index > 0 {
                        if let Err(error) = joined.extend_from_slice(separator.as_bytes()) { return error; }
                    }
                    if let Err(error) = joined.extend_from_slice(item.as_string().as_bytes()) { return error; }
                }
                self.string_output(std::str::from_utf8(&joined).unwrap(), out)
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
                let mut decoded = match Buffer::new(&self.heap, bytes.len(), 0) {
                    Ok(buffer) => buffer, Err(error) => return error,
                };
                let mut read = 0;
                while read < bytes.len() {
                    let value = bytes[read];
                    read += 1;
                    if value != b'\\' {
                        if let Err(error) = decoded.push(value) { return error; }
                        continue;
                    }
                    if read >= bytes.len() {
                        return LanaError::Format;
                    }
                    let value = bytes[read];
                    read += 1;
                    let byte = match value {
                        b'n' => b'\n',
                        b'r' => b'\r',
                        b't' => b'\t',
                        b'\\' | b'"' => value,
                        _ => return LanaError::Format,
                    };
                    if let Err(error) = decoded.push(byte) { return error; }
                }
                self.string_output(std::str::from_utf8(&decoded).unwrap(), out)
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
                let mut metadata = match Map::new(&self.heap, 5) { Ok(map) => map, Err(error) => return error };
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
                *out = match self.array_value(vec![
                        arguments[0].clone(),
                        Value::map(Arc::new(Mutex::new(metadata))),
                    ]) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
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
                let mut status = match Map::new(&self.heap, 3) { Ok(map) => map, Err(error) => return error };
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
                let mut status = match Map::new(&self.heap, 3) { Ok(map) => map, Err(error) => return error };
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
            LANA_HOST_GRANT => {
                if argc != 2 || !matches!(arguments[0].kind, ValueKind::Capability(_)) {
                    return LanaError::Type;
                }
                let permission = grant_permission(&arguments[1]);
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
            LANA_HOST_REVOKE => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::Capability(_)) {
                    return LanaError::Type;
                }
                let ValueKind::Capability(target) = &arguments[0].kind else {
                    unreachable!()
                };
                self.shared_capability_invalidate(target)
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
            LANA_HOST_SET_NEW | LANA_HOST_SET_ADD | LANA_HOST_SET_CONTAINS
            | LANA_HOST_SET_UNION | LANA_HOST_SET_INTERSECT | LANA_HOST_SET_DIFFERENCE => {
                match self.set_host_call(host_id, arguments) {
                    Ok(value) => { *out = value; LanaError::Ok }
                    Err(error) => error,
                }
            }
            LANA_HOST_GETENV => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let ValueKind::String(name) = &arguments[0].kind else {
                    unreachable!()
                };
                let value = std::env::var(&**name).unwrap_or_default();
                self.string_output(&value, out)
            }
            LANA_HOST_RANDOM_SEED => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::Number(_)) {
                    return LanaError::Type;
                }
                let ValueKind::Number(seed) = arguments[0].kind else {
                    unreachable!()
                };
                self.seed(seed as u64);
                *out = Value::null();
                LanaError::Ok
            }
            LANA_HOST_FLOOR => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::Number(_)) {
                    return LanaError::Type;
                }
                let ValueKind::Number(x) = arguments[0].kind else {
                    unreachable!()
                };
                *out = Value::number(x.floor());
                LanaError::Ok
            }
            LANA_HOST_STRING_TO_NUMBER => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let ValueKind::String(text) = &arguments[0].kind else {
                    unreachable!()
                };
                match text.parse::<f64>() {
                    Ok(number) => {
                        *out = match self.make_result(true, Value::number(number)) { Ok(value) => value, Err(error) => return error };
                        LanaError::Ok
                    }
                    Err(_) => {
                        *out = match self.make_result(false, Value::string(Arc::from("invalid number"))) { Ok(value) => value, Err(error) => return error };
                        LanaError::Ok
                    }
                }
            }
            LANA_HOST_TYPE_OF => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let name: &'static str = match arguments[0].kind {
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
                    ValueKind::PathSet(_) => "path_set",
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
                    ValueKind::Kernel(_) => "kernel",
                    ValueKind::Network(_) => "network",
                    ValueKind::ObjectValue(_) => "value",
                    ValueKind::ClassObject(_) => "class",
                };
                self.string_output(name, out)
            }
            LANA_HOST_FORMAT => {
                if argc < 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let format = arguments[0].as_string();
                let bytes = format.as_bytes();
                let mut result = match Buffer::new(&self.heap, 0, 0) {
                    Ok(buffer) => buffer, Err(error) => return error,
                };
                let mut arg_index = 1usize;
                let mut i = 0usize;
                let mut last = 0usize;
                while i < bytes.len() {
                    if bytes[i] == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'}' {
                        if let Err(error) = result.extend_from_slice(&bytes[last..i]) { return error; }
                        if arg_index >= argc {
                            return LanaError::Format;
                        }
                        let append = match &arguments[arg_index].kind {
                            ValueKind::Null => result.extend_from_slice(b"null"),
                            ValueKind::Bool(value) => result.extend_from_slice(if *value { b"true" } else { b"false" }),
                            ValueKind::Number(value) => result.extend_from_slice(format_g17(*value).as_bytes()),
                            ValueKind::String(value) => result.extend_from_slice(value.as_bytes()),
                            ValueKind::Array(_) | ValueKind::Map(_) => {
                                let mut stringified = Value::null();
                                let error =
                                    self.json_stringify(&arguments[arg_index], &mut stringified);
                                if error != LanaError::Ok {
                                    return error;
                                }
                                result.extend_from_slice(stringified.as_string().as_bytes())
                            }
                            _ => return LanaError::Type,
                        };
                        if let Err(error) = append { return error; }
                        arg_index += 1;
                        i += 2;
                        last = i;
                    } else {
                        i += 1;
                    }
                }
                if let Err(error) = result.extend_from_slice(&bytes[last..]) { return error; }
                if arg_index != argc {
                    return LanaError::Format;
                }
                self.string_output(std::str::from_utf8(&result).unwrap(), out)
            }
            LANA_HOST_FORMAT_NUMBER => {
                let text = match argc {
                    1 => {
                        if !matches!(arguments[0].kind, ValueKind::Number(_)) {
                            return LanaError::Type;
                        }
                        format_g17(arguments[0].as_number())
                    }
                    2 => {
                        let ValueKind::Number(precision) = arguments[1].kind else {
                            return LanaError::Type;
                        };
                        if !matches!(arguments[0].kind, ValueKind::Number(_))
                            || precision < 0.0
                            || precision.floor() != precision
                            || precision > 1000.0
                        {
                            return LanaError::Type;
                        }
                        format_fixed(arguments[0].as_number(), precision as usize)
                    }
                    _ => return LanaError::Type,
                };
                self.string_output(&text, out)
            }
            LANA_HOST_CHAR_LENGTH => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let text = arguments[0].as_string();
                let bytes = text.as_bytes();
                let mut i = 0usize;
                let mut count = 0usize;
                while i < bytes.len() {
                    let Some((_, consumed)) = utf8_decode(&bytes[i..]) else {
                        return LanaError::Schema;
                    };
                    i += consumed;
                    count += 1;
                }
                *out = Value::number(count as f64);
                LanaError::Ok
            }
            LANA_HOST_STRING_CODEPOINT_SLICE => {
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
                if start > end {
                    return LanaError::Limit;
                }
                let text = arguments[0].as_string();
                let bytes = text.as_bytes();
                let mut i = 0usize;
                let mut cp_count = 0usize;
                while i < bytes.len() {
                    let Some((_, consumed)) = utf8_decode(&bytes[i..]) else {
                        return LanaError::Schema;
                    };
                    i += consumed;
                    cp_count += 1;
                }
                if end > cp_count {
                    return LanaError::Limit;
                }
                let mut start_byte = bytes.len();
                let mut end_byte = bytes.len();
                let mut cp_index = 0usize;
                i = 0usize;
                while i < bytes.len() {
                    let (_, consumed) = utf8_decode(&bytes[i..]).unwrap();
                    if cp_index == start {
                        start_byte = i;
                    }
                    if cp_index == end {
                        end_byte = i;
                    }
                    i += consumed;
                    cp_index += 1;
                }
                self.string_output(&text[start_byte..end_byte], out)
            }
            LANA_HOST_TO_UPPER | LANA_HOST_TO_LOWER => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let upper = host_id == LANA_HOST_TO_UPPER;
                let text = arguments[0].as_string();
                let bytes = text.as_bytes();
                let mut result = match Buffer::new(&self.heap, bytes.len(), 0) {
                    Ok(buffer) => buffer, Err(error) => return error,
                };
                let mut i = 0usize;
                while i < bytes.len() {
                    let Some((cp, consumed)) = utf8_decode(&bytes[i..]) else {
                        return LanaError::Schema;
                    };
                    let mapped = if upper {
                        crate::unicode_case::unicode_upper(cp)
                    } else {
                        crate::unicode_case::unicode_lower(cp)
                    };
                    if let Err(error) = result.extend_from_slice(&utf8_encode(mapped)) { return error; }
                    i += consumed;
                }
                self.string_output(std::str::from_utf8(&result).unwrap(), out)
            }
            LANA_HOST_REGEX_COMPILE => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let pattern = arguments[0].as_string();
                match regex_compile(pattern.as_bytes()) {
                    Ok(re) => *out = match self.make_result(true, Value::regex(Arc::new(re))) { Ok(value) => value, Err(error) => return error },
                    Err(msg) => *out = match self.make_result(false, Value::string(Arc::from(msg))) { Ok(value) => value, Err(error) => return error },
                }
                LanaError::Ok
            }
            LANA_HOST_REGEX_MATCH | LANA_HOST_REGEX_SEARCH => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Regex(_))
                    || !matches!(arguments[1].kind, ValueKind::String(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Regex(re) = &arguments[0].kind else {
                    return LanaError::Type;
                };
                let text = arguments[1].as_string();
                let bytes = text.as_bytes();
                let matched = if host_id == LANA_HOST_REGEX_MATCH {
                    match regex_match_from(re, bytes, 0) {
                        Some(end) if end == bytes.len() => Some((0usize, bytes.len())),
                        _ => None,
                    }
                } else {
                    regex_search(re, bytes, 0)
                };
                match matched {
                    Some((start, end)) => {
                        let mut map = match Map::new(&self.heap, 3) { Ok(map) => map, Err(error) => return error };
                        let matched_text = match self.heap.lossy_string(&bytes[start..end]) {
                            Ok(text) => Value::string(text), Err(error) => return error,
                        };
                        for (key, value) in [("start", Value::number(start as f64)),
                            ("end", Value::number(end as f64)), ("text", matched_text)] {
                            if let Err(error) = map.set(Arc::from(key), value, false) { return error; }
                        }
                        *out = match self.make_result(true, Value::map(Arc::new(Mutex::new(map)))) { Ok(value) => value, Err(error) => return error };
                    }
                    None => {
                        *out = match self.make_result(false, Value::string(Arc::from("no match"))) { Ok(value) => value, Err(error) => return error }
                    }
                }
                LanaError::Ok
            }
            LANA_HOST_REGEX_REPLACE => {
                if argc != 3
                    || !matches!(arguments[0].kind, ValueKind::Regex(_))
                    || !matches!(arguments[1].kind, ValueKind::String(_))
                    || !matches!(arguments[2].kind, ValueKind::String(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Regex(re) = &arguments[0].kind else {
                    return LanaError::Type;
                };
                let text = arguments[1].as_string();
                let replacement = arguments[2].as_string();
                let bytes = text.as_bytes();
                let mut result = match Buffer::new(&self.heap, 0, 0) {
                    Ok(buffer) => buffer, Err(error) => return error,
                };
                let mut pos = 0usize;
                while pos <= bytes.len() {
                    match regex_search(re, bytes, pos) {
                        Some((start, end)) => {
                            if let Err(error) = result.extend_from_slice(&bytes[pos..start]) { return error; }
                            if let Err(error) = result.extend_from_slice(replacement.as_bytes()) { return error; }
                            pos = end;
                            if start == end {
                                if pos < bytes.len() {
                                    if let Err(error) = result.push(bytes[pos]) { return error; }
                                    pos += 1;
                                } else {
                                    break;
                                }
                            }
                        }
                        None => {
                            if let Err(error) = result.extend_from_slice(&bytes[pos..]) { return error; }
                            break;
                        }
                    }
                }
                *out = match self.heap.lossy_string(&result) {
                    Ok(text) => Value::string(text), Err(error) => return error,
                };
                LanaError::Ok
            }
            LANA_HOST_SGD => self.host_sgd(arguments, out),
            LANA_HOST_ADAM => self.host_adam(arguments, out),
            LANA_HOST_MCMC => self.host_mcmc(arguments, out),
            LANA_HOST_VI => self.host_vi(arguments, out),
            LANA_HOST_SMC => self.host_smc(arguments, out),
            LANA_HOST_RUN_ASYNC => self.host_run_async(arguments, out),
            LANA_HOST_FUTURE_ALL => self.host_future_all(arguments, out),
            LANA_HOST_FUTURE_RACE => self.host_future_race(arguments, out),
            LANA_HOST_SLEEP => self.host_sleep(arguments, out),
            LANA_HOST_DATASET => self.host_dataset(arguments, out),
            LANA_HOST_DATASET_FILTER => self.host_dataset_filter(arguments, out),
            LANA_HOST_DATASET_MAP => self.host_dataset_map(arguments, out),
            LANA_HOST_DATASET_SELECT => self.host_dataset_select(arguments, out),
            LANA_HOST_DATASET_LIMIT => self.host_dataset_limit(arguments, out),
            LANA_HOST_DATASET_SORT => self.host_dataset_sort(arguments, out),
            LANA_HOST_DATASET_GROUP_BY => self.host_dataset_group_by(arguments, out),
            LANA_HOST_DATASET_AGGREGATE => self.host_dataset_aggregate(arguments, out),
            LANA_HOST_DATASET_JOIN => self.host_dataset_join(arguments, out),
            LANA_HOST_DATASET_MATERIALIZE => self.host_dataset_materialize(arguments, out),
            LANA_HOST_DATASET_EXPLAIN => self.host_dataset_explain(arguments, out),
            LANA_HOST_FFI_DECLARE => self.host_ffi_declare(arguments, out),
            LANA_HOST_FFI_LOAD => self.host_ffi_load(arguments, out),
            LANA_HOST_FFI_CALL => self.host_ffi_call(arguments, out),
            LANA_HOST_HTTP_GET => self.host_http_get(arguments, out),
            LANA_HOST_HTTP_POST => self.host_http_post(arguments, out),
            LANA_HOST_SOCKET_CONNECT => self.host_socket_connect(arguments, out),
            LANA_HOST_SOCKET_SEND => self.host_socket_send(arguments, out),
            LANA_HOST_SOCKET_RECV => self.host_socket_recv(arguments, out),
            LANA_HOST_SOCKET_CLOSE => self.host_socket_close(arguments, out),
            _ => {
                let Some(mut handler) = self.host_call_extension.take() else { return LanaError::Format; };
                let error = handler(self, host_id, arguments, out);
                self.host_call_extension = Some(handler);
                if error == LanaError::Ok && class_gc::has_gc_graph(out) {
                    if let Err(error) = self.retain_value_impl(out, true) { return error; }
                }
                error
            },
        }
    }

    /// `ffi_declare(signature) -> index` (LIP-018): parse a C-style signature
    /// and store it in the per-VM table, returning its index. Pure: no
    /// capability is required to declare a signature.
    fn host_ffi_declare(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::String(sig) = &arguments[0].kind else {
            return LanaError::Type;
        };
        if parse_ffi_signature(sig).is_none() {
            return LanaError::External;
        }
        let index = self.ffi_sigs.len();
        self.ffi_sigs.push(sig.to_string());
        *out = Value::number(index as f64);
        LanaError::Ok
    }

    /// `ffi_load(path)` (LIP-018): dlopen a shared library. Requires the `ffi`
    /// capability; denial is `LANA_ERR_EXTERNAL`.
    fn host_ffi_load(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::String(path) = &arguments[0].kind else {
            return LanaError::Type;
        };
        if !self.has_named_capability("ffi") {
            return LanaError::External;
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (path, out);
            return LanaError::UnsupportedOperation;
        }
        #[cfg(not(target_arch = "wasm32"))]
        match unsafe { libloading::Library::new(path.as_ref()) } {
            Ok(lib) => {
                self.ffi_lib = Some(lib);
                *out = Value::null();
                LanaError::Ok
            }
            Err(_) => LanaError::External,
        }
    }

    /// `ffi_call(lib, fn, args) -> Result<T, E>` (LIP-018): invoke a declared
    /// symbol. Requires the `ffi` capability. Returns a `{"ok": ...}` /
    /// `{"error": ...}` map; capability denial is `LANA_ERR_EXTERNAL`.
    #[cfg(not(target_arch = "wasm32"))]
    fn host_ffi_call(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 3 {
            return LanaError::Type;
        }
        let ValueKind::Number(_lib) = &arguments[0].kind else {
            return LanaError::Type;
        };
        let ValueKind::Number(fn_index) = &arguments[1].kind else {
            return LanaError::Type;
        };
        let ValueKind::Array(args) = &arguments[2].kind else {
            return LanaError::Type;
        };
        if !self.has_named_capability("ffi") {
            return LanaError::External;
        }
        let index = *fn_index as usize;
        if index >= self.ffi_sigs.len() {
            return LanaError::External;
        }
        let sig = match parse_ffi_signature(&self.ffi_sigs[index]) {
            Some(sig) => sig,
            None => return LanaError::External,
        };
        let arg_values: Vec<Value> = args.lock().unwrap().items.to_vec();
        /* Validate the argument types before the library check so a bad
         * argument is reported as a Result error even with no library loaded
         * (deterministic across both VMs). */
        if !ffi_validate_args(&sig, &arg_values) {
            let e = Value::string(Arc::from("type"));
            return self.ffi_result_map("error", &e, out);
        }
        let lib = match self.ffi_lib.as_ref() {
            Some(lib) => lib,
            None => return LanaError::InvalidState,
        };
        match ffi_call_impl(lib, &sig, &arg_values) {
            Ok(value) => self.ffi_result_map("ok", &value, out),
            Err(FfiError::Type) => {
                let e = Value::string(Arc::from("type"));
                self.ffi_result_map("error", &e, out)
            }
            Err(FfiError::External) => {
                let e = Value::string(Arc::from("external"));
                self.ffi_result_map("error", &e, out)
            }
        }
    }

    /// `wasm32` has no dynamic library loading, so `ffi_call` is unsupported.
    #[cfg(target_arch = "wasm32")]
    fn host_ffi_call(&mut self, _arguments: &[Value], _out: &mut Value) -> LanaError {
        LanaError::UnsupportedOperation
    }

    fn ffi_result_map(&self, key: &str, value: &Value, out: &mut Value) -> LanaError {
        let mut map = match Map::new(&self.heap, 1) { Ok(map) => map, Err(error) => return error };
        if map.set(Arc::from(key), value.clone(), false).is_err() {
            return LanaError::Oom;
        }
        *out = Value::map(Arc::new(Mutex::new(map)));
        LanaError::Ok
    }

    /// LIP-019 networking: error result `[false, reason]`, mirroring the
    /// language `Result` tagged-pair that `result_error` and json_parse emit.
    fn net_error_result(&self, reason: &str, out: &mut Value) -> LanaError {
        let reason = match self.string_value(reason) { Ok(value) => value, Err(error) => return error };
        *out = match self.make_result(false, reason) { Ok(value) => value, Err(error) => return error };
        LanaError::Ok
    }

    fn net_receive_response(&mut self, socket: &mut NetSocket, timeout: u64) -> Result<http::Response, http::Error> {
        let deadline = std::time::Instant::now().checked_add(std::time::Duration::from_millis(timeout))
            .ok_or(http::Error::Resource(LanaError::InvalidParameters))?;
        let mut response = http::Response::new(&self.heap)?;
        let mut buffer = [0u8; 4096];
        while !response.complete() {
            let remaining = deadline.checked_duration_since(std::time::Instant::now()).ok_or(http::Error::Timeout)?;
            match socket.read(&mut buffer, (remaining.as_millis() as u64).max(1)) {
                Ok(0) => response.eof()?,
                Ok(count) => {
                    self.charge_bounded_work(count as u64)?;
                    response.feed(&buffer[..count])?;
                }
                Err(NetError::Timeout) => return Err(http::Error::Timeout),
                Err(NetError::Io) => return Err(http::Error::Io),
            }
        }
        Ok(response)
    }

    fn net_header_values(&self, headers: &http::Headers) -> Result<Value, LanaError> {
        let mut map = Map::new(&self.heap, headers.len())?;
        for (name, value) in headers {
            if let Some(Value { kind: ValueKind::Array(values), .. }) = map.get(name) {
                values.lock().unwrap().push(Value::string(value.clone()))?;
            } else {
                map.set(name.clone(), self.array_value(vec![Value::string(value.clone())])?, true)?;
            }
        }
        Ok(Value::map(Arc::new(Mutex::new(map))))
    }

    /// Perform an HTTP request and build the `Result<HttpResponse, E>` tagged
    /// pair. On success the response map is rooted as Information with a
    /// derivation recording the operation and URL (LIP-019 §4).
    fn net_http_request(
        &mut self,
        method: &str,
        url: &str,
        body: Option<&str>,
        timeout_ms: f64,
        verify: bool,
        headers: &[u8],
        out: &mut Value,
    ) -> LanaError {
        if !timeout_ms.is_finite() || timeout_ms > u64::MAX as f64 { return LanaError::InvalidParameters; }
        let timeout = if timeout_ms > 0.0 { (timeout_ms as u64).max(1) } else { 5000 };
        let (scheme, host, port, path) = match net_parse_url(url) {
            Some(x) => x,
            None => return self.net_error_result("url", out),
        };
        let authority = if host.contains(':') { format!("[{host}]:{port}") } else { format!("{host}:{port}") };
        let prefix = format!("{method} {path} HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\nContent-Length: {}\r\n", body.map_or(0, str::len));
        if prefix.len().saturating_add(headers.len()).saturating_add(2) > http::HEADER_LIMIT {
            return self.net_error_result("headers", out);
        }
        let mut request = match Buffer::new(&self.heap, prefix.len() + headers.len() + 2, 0) {
            Ok(request) => request, Err(error) => return error,
        };
        if request.extend_from_slice(prefix.as_bytes()).is_err()
            || request.extend_from_slice(headers).is_err() || request.extend_from_slice(b"\r\n").is_err() { return LanaError::Oom; }
        #[cfg(any(not(feature = "net-tls"), target_arch = "wasm32"))]
        {
            let _ = verify;
            if scheme == "https" { return self.net_error_result("tls", out); }
        }
        let stream = match net_connect(&host, port, timeout) {
            Ok(s) => s,
            Err(NetError::Timeout) => return self.net_error_result("timeout", out),
            Err(_) => return self.net_error_result("connect", out),
        };
        let mut sock = if scheme == "https" {
            #[cfg(all(feature = "net-tls", not(target_arch = "wasm32")))]
            {
                match net_tls_connect(stream, &host, verify) {
                    Ok(s) => s,
                    Err(NetError::Timeout) => return self.net_error_result("timeout", out),
                    Err(_) => return self.net_error_result("tls", out),
                }
            }
            #[cfg(any(not(feature = "net-tls"), target_arch = "wasm32"))]
            {
                unreachable!("HTTPS requires TLS support")
            }
        } else {
            NetSocket::Plain(stream)
        };
        if let Err(error) = sock.write_all(&request).and_then(|_| sock.write_all(body.unwrap_or("").as_bytes())) {
            return self.net_error_result(if matches!(error, NetError::Timeout) { "timeout" } else { "send" }, out);
        }
        drop(request);
        let response = match self.net_receive_response(&mut sock, timeout) {
            Ok(response) => response,
            Err(http::Error::Resource(error)) => return error,
            Err(http::Error::Protocol) => return self.net_error_result("response", out),
            Err(http::Error::Timeout) => return self.net_error_result("timeout", out),
            Err(http::Error::Io) => return self.net_error_result("recv", out),
        };
        let mut resp = match Map::new(&self.heap, 4) { Ok(map) => map, Err(error) => return error };
        let headers = match self.net_header_values(&response.headers) { Ok(value) => value, Err(error) => return error };
        let trailers = match self.net_header_values(&response.trailers) { Ok(value) => value, Err(error) => return error };
        let body = match self.heap.lossy_string(&response.body) { Ok(value) => Value::string(value), Err(error) => return error };
        for (name, value) in [("status", Value::number(response.status as f64)), ("headers", headers), ("trailers", trailers), ("body", body)] {
            if let Err(error) = resp.set(Arc::from(name), value, false) { return error; }
        }
        // LIP-019 §4: root the response as Information with an evidence
        // derivation recording the operation and URL, then wrap `[true, val]`.
        let resp_value = Value::map(Arc::new(Mutex::new(resp)));
        let mut rooted = match self.reactive_root(&resp_value, DerivationExactness::Exact) {
            Ok(rooted) => rooted,
            Err(error) => return error,
        };
        let op = if method == "POST" { "http_post" } else { "http_get" };
        rooted.derivation = self.record_derivation(
            DerivationKind::Evidence,
            op,
            &[],
            url,
            0,
            DerivationExactness::Exact,
            "root",
            DerivationOutcome::Success,
            "none",
        );
        *out = match self.make_result(true, rooted) { Ok(value) => value, Err(error) => return error };
        LanaError::Ok
    }

    /// `http_get(url, headers, timeout_ms) -> Result<HttpResponse, E>` (LIP-019).
    fn host_http_get(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 3 {
            return LanaError::Type;
        }
        let ValueKind::String(url) = &arguments[0].kind else {
            return LanaError::Type;
        };
        let ValueKind::Map(headers) = &arguments[1].kind else {
            return LanaError::Type;
        };
        let ValueKind::Number(timeout_ms) = &arguments[2].kind else {
            return LanaError::Type;
        };
        if !self.has_named_capability("net") {
            return LanaError::Capability;
        }
        let (verify, bytes) = match http::request_headers(&self.heap, &headers.lock().unwrap()) {
            Ok(headers) => headers,
            Err(http::Error::Resource(error)) => return error,
            Err(_) => return self.net_error_result("headers", out),
        };
        self.net_http_request("GET", url, None, *timeout_ms, verify, &bytes, out)
    }

    /// `http_post(url, body, headers, timeout_ms) -> Result<HttpResponse, E>`.
    fn host_http_post(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 4 {
            return LanaError::Type;
        }
        let ValueKind::String(url) = &arguments[0].kind else {
            return LanaError::Type;
        };
        let ValueKind::String(body) = &arguments[1].kind else {
            return LanaError::Type;
        };
        let ValueKind::Map(headers) = &arguments[2].kind else {
            return LanaError::Type;
        };
        let ValueKind::Number(timeout_ms) = &arguments[3].kind else {
            return LanaError::Type;
        };
        if !self.has_named_capability("net") {
            return LanaError::Capability;
        }
        let (verify, bytes) = match http::request_headers(&self.heap, &headers.lock().unwrap()) {
            Ok(headers) => headers,
            Err(http::Error::Resource(error)) => return error,
            Err(_) => return self.net_error_result("headers", out),
        };
        self.net_http_request("POST", url, Some(body), *timeout_ms, verify, &bytes, out)
    }

    /// `socket_connect(host, port) -> Result<Socket, E>` (LIP-019).
    fn host_socket_connect(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 2 {
            return LanaError::Type;
        }
        let ValueKind::String(host) = &arguments[0].kind else {
            return LanaError::Type;
        };
        let ValueKind::Number(port) = &arguments[1].kind else {
            return LanaError::Type;
        };
        if !self.has_named_capability("net") {
            return LanaError::Capability;
        }
        let stream = match net_connect(host, *port as u16, 5000) {
            Ok(s) => s,
            Err(NetError::Timeout) => return self.net_error_result("timeout", out),
            Err(_) => return self.net_error_result("connect", out),
        };
        let handle = self.sockets.len();
        self.sockets.push(NetSocket::Plain(stream));
        *out = match self.make_result(true, Value::number(handle as f64)) { Ok(value) => value, Err(error) => return error };
        LanaError::Ok
    }

    /// `socket_send(sock, bytes) -> Result<number, E>` (LIP-019).
    fn host_socket_send(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 2 {
            return LanaError::Type;
        }
        let ValueKind::Number(handle) = &arguments[0].kind else {
            return LanaError::Type;
        };
        let ValueKind::String(bytes) = &arguments[1].kind else {
            return LanaError::Type;
        };
        if !self.has_named_capability("net") {
            return LanaError::Capability;
        }
        let index = *handle as usize;
        let Some(sock) = self.sockets.get_mut(index) else {
            return LanaError::InvalidState;
        };
        match sock.write_all(bytes.as_bytes()) {
            Ok(()) => {
                *out = match self.make_result(true, Value::number(bytes.len() as f64)) { Ok(value) => value, Err(error) => return error };
                LanaError::Ok
            }
            Err(_) => self.net_error_result("send", out),
        }
    }

    /// `socket_recv(sock, max_bytes) -> Result<string, E>` (LIP-019).
    fn host_socket_recv(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 2 {
            return LanaError::Type;
        }
        let ValueKind::Number(handle) = &arguments[0].kind else {
            return LanaError::Type;
        };
        let ValueKind::Number(max_bytes) = &arguments[1].kind else {
            return LanaError::Type;
        };
        if !self.has_named_capability("net") {
            return LanaError::Capability;
        }
        let index = *handle as usize;
        let Some(sock) = self.sockets.get_mut(index) else {
            return LanaError::InvalidState;
        };
        let mut buf = match Buffer::filled(&self.heap, (*max_bytes as usize).min(65535), 0u8) {
            Ok(buffer) => buffer, Err(error) => return error,
        };
        match sock.read(&mut buf, 5000) {
            Ok(n) => {
                let text = match self.heap.lossy_string(&buf[..n]) {
                    Ok(text) => text, Err(error) => return error,
                };
                *out = match self.make_result(true, Value::string(text)) { Ok(value) => value, Err(error) => return error };
                LanaError::Ok
            }
            Err(NetError::Timeout) => self.net_error_result("timeout", out),
            Err(_) => self.net_error_result("recv", out),
        }
    }

    /// `socket_close(sock)` (LIP-019).
    fn host_socket_close(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::Number(handle) = &arguments[0].kind else {
            return LanaError::Type;
        };
        if !self.has_named_capability("net") {
            return LanaError::Capability;
        }
        let index = *handle as usize;
        if index >= self.sockets.len() {
            return LanaError::InvalidState;
        }
        self.sockets.remove(index);
        *out = Value::null();
        LanaError::Ok
    }

    /// `run_async(future) -> value` (LIP-024 §4): run the event loop to
    /// completion on the given future, then return its result.
    fn host_run_async(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::Future(future) = &arguments[0].kind else {
            return LanaError::Type;
        };
        let future = future.clone();
        self.enqueue_future(future.clone());
        let saved_ip = self.ip;
        let saved_active = self.event_loop_active;
        let saved_base = self.event_loop_base_depth;
        self.event_loop_base_depth = self.frames.len();
        let error = self.run_event_loop();
        self.ip = saved_ip;
        self.event_loop_active = saved_active;
        self.event_loop_base_depth = saved_base;
        if error != LanaError::Ok {
            return error;
        }
        let result = {
            let future = future.lock().unwrap();
            future.registers[0].clone()
        };
        *out = result;
        LanaError::Ok
    }

    /// `future_all(futures) -> future` (LIP-024 §4): a composite future that
    /// completes when all input futures complete, yielding an array of their
    /// results in input order.
    fn host_future_all(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::Array(array) = &arguments[0].kind else {
            return LanaError::Type;
        };
        let items = array.lock().unwrap().items.to_vec();
        for item in &items {
            if !matches!(item.kind, ValueKind::Future(_)) {
                return LanaError::Type;
            }
        }
        let mut registers = Vec::with_capacity(items.len() + 1);
        registers.push(Value::string(Arc::from("all")));
        registers.extend(items.clone());
        let composite = Arc::new(Mutex::new(crate::value::Future {
            function: u32::MAX,
            ip: 0,
            registers,
            exhausted: false,
            ready: true,
            queued: false,
        }));
        if let Err(error) = self.track_cycle(crate::heap::CycleWeak::Future(Arc::downgrade(&composite))) { return error; }
        for item in &items {
            let ValueKind::Future(f) = &item.kind else {
                continue;
            };
            self.awaiters
                .entry(Arc::as_ptr(f) as usize)
                .or_default()
                .push(composite.clone());
            self.enqueue_future(f.clone());
        }
        self.enqueue_future(composite.clone());
        *out = Value::future(composite);
        LanaError::Ok
    }

    /// `future_race(futures) -> future` (LIP-024 §4): a composite future that
    /// completes with the first input future to complete, yielding that
    /// future's result.
    fn host_future_race(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::Array(array) = &arguments[0].kind else {
            return LanaError::Type;
        };
        let items = array.lock().unwrap().items.to_vec();
        for item in &items {
            if !matches!(item.kind, ValueKind::Future(_)) {
                return LanaError::Type;
            }
        }
        let mut registers = Vec::with_capacity(items.len() + 1);
        registers.push(Value::string(Arc::from("race")));
        registers.extend(items.clone());
        let composite = Arc::new(Mutex::new(crate::value::Future {
            function: u32::MAX,
            ip: 0,
            registers,
            exhausted: false,
            ready: true,
            queued: false,
        }));
        if let Err(error) = self.track_cycle(crate::heap::CycleWeak::Future(Arc::downgrade(&composite))) { return error; }
        for item in &items {
            let ValueKind::Future(f) = &item.kind else {
                continue;
            };
            self.awaiters
                .entry(Arc::as_ptr(f) as usize)
                .or_default()
                .push(composite.clone());
            self.enqueue_future(f.clone());
        }
        self.enqueue_future(composite.clone());
        *out = Value::future(composite);
        LanaError::Ok
    }

    /// `sleep(ms) -> future` (LIP-024 §4): a composite future that completes
    /// after `ms` milliseconds, yielding `null`. The single-threaded VM blocks
    /// for the duration; WASM rejects this host call at dispatch (no clock).
    fn host_sleep(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::Number(ms) = arguments[0].kind else {
            return LanaError::Type;
        };
        if !ms.is_finite() || ms < 0.0 {
            return LanaError::InvalidParameters;
        }
        let composite = Arc::new(Mutex::new(crate::value::Future {
            function: u32::MAX,
            ip: 0,
            registers: vec![Value::string(Arc::from("sleep")), Value::number(ms)],
            exhausted: false,
            ready: true,
            queued: false,
        }));
        if let Err(error) = self.track_cycle(crate::heap::CycleWeak::Future(Arc::downgrade(&composite))) { return error; }
        self.enqueue_future(composite.clone());
        *out = Value::future(composite);
        LanaError::Ok
    }

    // ===== LIP-015: lazy relational-algebra dataset engine =====

    /// `dataset(source) -> dataset` (LIP-015): wrap a lazy source into a
    /// dataset plan. `source` must be a `Lazy` value.
    fn host_dataset(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 1 {
            return LanaError::Type;
        }
        /* LIP-015 §3: accept either a lazy generator (the lazy path) or an
         * in-memory array of rows (the persistence / adapter load path). */
        if !matches!(arguments[0].kind, ValueKind::Lazy { .. } | ValueKind::Array(_)) {
            return LanaError::Type;
        }
        *out = Value::dataset(match self.managed_payload(Dataset {
            op: DatasetOp::Source,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: Value::null(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }) { Ok(payload) => payload, Err(error) => return error });
        LanaError::Ok
    }

    /// `dataset_filter(ds, fn) -> dataset`: keep rows for which `fn(row)` is true.
    fn host_dataset_filter(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 2 {
            return LanaError::Type;
        }
        if !matches!(arguments[0].kind, ValueKind::Dataset(_)) {
            return LanaError::Type;
        }
        let ValueKind::Function(function) = arguments[1].kind else {
            return LanaError::Type;
        };
        *out = Value::dataset(match self.managed_payload(Dataset {
            op: DatasetOp::Filter,
            source: arguments[0].clone(),
            function,
            columns: Value::null(),
            key: Value::null(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }) { Ok(payload) => payload, Err(error) => return error });
        LanaError::Ok
    }

    /// `dataset_map(ds, fn) -> dataset`: transform each row with `fn(row)`.
    fn host_dataset_map(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 2 {
            return LanaError::Type;
        }
        if !matches!(arguments[0].kind, ValueKind::Dataset(_)) {
            return LanaError::Type;
        }
        let ValueKind::Function(function) = arguments[1].kind else {
            return LanaError::Type;
        };
        *out = Value::dataset(match self.managed_payload(Dataset {
            op: DatasetOp::Map,
            source: arguments[0].clone(),
            function,
            columns: Value::null(),
            key: Value::null(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }) { Ok(payload) => payload, Err(error) => return error });
        LanaError::Ok
    }

    /// `dataset_select(ds, columns) -> dataset`: project rows onto `columns`.
    fn host_dataset_select(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 2 {
            return LanaError::Type;
        }
        if !matches!(arguments[0].kind, ValueKind::Dataset(_)) {
            return LanaError::Type;
        }
        if !matches!(arguments[1].kind, ValueKind::Array(_)) {
            return LanaError::Type;
        }
        *out = Value::dataset(match self.managed_payload(Dataset {
            op: DatasetOp::Select,
            source: arguments[0].clone(),
            function: 0,
            columns: arguments[1].clone(),
            key: Value::null(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }) { Ok(payload) => payload, Err(error) => return error });
        LanaError::Ok
    }

    /// `dataset_limit(ds, n) -> dataset`: cap the row count at `n`.
    fn host_dataset_limit(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 2 {
            return LanaError::Type;
        }
        if !matches!(arguments[0].kind, ValueKind::Dataset(_)) {
            return LanaError::Type;
        }
        let ValueKind::Number(n) = arguments[1].kind else {
            return LanaError::Type;
        };
        if !n.is_finite() || n < 0.0 {
            return LanaError::Type;
        }
        *out = Value::dataset(match self.managed_payload(Dataset {
            op: DatasetOp::Limit,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: Value::null(),
            limit: arguments[1].clone(),
            other: Value::null(),
            aggregate: Value::null(),
        }) { Ok(payload) => payload, Err(error) => return error });
        LanaError::Ok
    }

    /// `dataset_sort(ds, key) -> dataset`: sort rows by the `key` column.
    fn host_dataset_sort(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 2 {
            return LanaError::Type;
        }
        if !matches!(arguments[0].kind, ValueKind::Dataset(_)) {
            return LanaError::Type;
        }
        if !matches!(arguments[1].kind, ValueKind::String(_)) {
            return LanaError::Type;
        }
        *out = Value::dataset(match self.managed_payload(Dataset {
            op: DatasetOp::Sort,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: arguments[1].clone(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }) { Ok(payload) => payload, Err(error) => return error });
        LanaError::Ok
    }

    /// `dataset_group_by(ds, key) -> dataset`: group rows by the `key` column.
    fn host_dataset_group_by(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 2 {
            return LanaError::Type;
        }
        if !matches!(arguments[0].kind, ValueKind::Dataset(_)) {
            return LanaError::Type;
        }
        if !matches!(arguments[1].kind, ValueKind::String(_)) {
            return LanaError::Type;
        }
        *out = Value::dataset(match self.managed_payload(Dataset {
            op: DatasetOp::GroupBy,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: arguments[1].clone(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }) { Ok(payload) => payload, Err(error) => return error });
        LanaError::Ok
    }

    /// `dataset_aggregate(ds, spec) -> dataset`: aggregate grouped rows. `spec`
    /// is `["count"]` or `["sum"|"mean"|"max"|"min", column]`.
    fn host_dataset_aggregate(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 2 {
            return LanaError::Type;
        }
        if !matches!(arguments[0].kind, ValueKind::Dataset(_)) {
            return LanaError::Type;
        }
        if !matches!(arguments[1].kind, ValueKind::Array(_)) {
            return LanaError::Type;
        }
        *out = Value::dataset(match self.managed_payload(Dataset {
            op: DatasetOp::Aggregate,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: Value::null(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: arguments[1].clone(),
        }) { Ok(payload) => payload, Err(error) => return error });
        LanaError::Ok
    }

    /// `dataset_join(left, right, key) -> dataset`: inner-join two datasets on
    /// the `key` column, merging matching rows.
    fn host_dataset_join(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 3 {
            return LanaError::Type;
        }
        if !matches!(arguments[0].kind, ValueKind::Dataset(_)) {
            return LanaError::Type;
        }
        if !matches!(arguments[1].kind, ValueKind::Dataset(_)) {
            return LanaError::Type;
        }
        if !matches!(arguments[2].kind, ValueKind::String(_)) {
            return LanaError::Type;
        }
        *out = Value::dataset(match self.managed_payload(Dataset {
            op: DatasetOp::Join,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: arguments[2].clone(),
            limit: Value::null(),
            other: arguments[1].clone(),
            aggregate: Value::null(),
        }) { Ok(payload) => payload, Err(error) => return error });
        LanaError::Ok
    }

    /// `dataset_materialize(ds) -> array`: eagerly evaluate the plan to rows.
    fn host_dataset_materialize(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::Dataset(dataset) = &arguments[0].kind else {
            return LanaError::Type;
        };
        let rows = match self.dataset_materialize(dataset, 0) {
            Ok(rows) => rows,
            Err(error) => return error,
        };
        if self.dataset_work_remaining.is_some() && rows.len() > 100_000 { return LanaError::Limit; }
        *out = match self.array_value(rows) { Ok(value) => value, Err(error) => return error };
        LanaError::Ok
    }

    /// `dataset_explain(ds) -> map`: build an inspectable plan value.
    fn host_dataset_explain(&mut self, arguments: &[Value], out: &mut Value) -> LanaError {
        if arguments.len() != 1 {
            return LanaError::Type;
        }
        let ValueKind::Dataset(dataset) = &arguments[0].kind else {
            return LanaError::Type;
        };
        *out = match self.dataset_explain(dataset) {
            Ok(value) => value,
            Err(error) => return error,
        };
        LanaError::Ok
    }

    /// Materialize a lazy source into a `Vec<Value>` of rows by invoking the
    /// generator function for each index in `[0, bound)`, mirroring
    /// `dataset_materialize_source` in `vm/c/vm.c`.
    fn dataset_materialize_source(&mut self, lazy: &Value, scratch: u32) -> Result<Vec<Value>, LanaError> {
        /* LIP-015 §3: an in-memory array source is returned as-is. */
        if let ValueKind::Array(array) = &lazy.kind {
            let items = array.lock().unwrap();
            if self.dataset_work_remaining.is_some() && items.items.len() > 100_000 { return Err(LanaError::Limit); }
            self.charge_dataset_work(items.items.len() as u64)?;
            return Ok(items.items.to_vec());
        }
        let ValueKind::Lazy { function, bound } = lazy.kind else {
            return Err(LanaError::Type);
        };
        if self.dataset_work_remaining.is_some() && bound > 100_000 { return Err(LanaError::Limit); }
        if self.dataset_work_remaining.is_some_and(|remaining| bound as u64 > remaining) {
            return Err(LanaError::Limit);
        }
        let mut rows = Vec::with_capacity(bound);
        for i in 0..bound {
            self.charge_dataset_work(1)?;
            let index_value = Value::number(i as f64);
            let mut row = Value::null();
            let error = self.run_function(function, &index_value, scratch, &mut row);
            if error != LanaError::Ok {
                return Err(error);
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// Read the value of `key` from a row (a map), mirroring `dataset_row_key`.
    fn dataset_row_key(row: &Value, key: &str) -> Result<Value, LanaError> {
        let ValueKind::Map(map) = &row.kind else {
            return Err(LanaError::Type);
        };
        map.lock().unwrap().get(key).cloned().ok_or(LanaError::Type)
    }

    fn dataset_checked_key(&self, row: &Value, key: &str) -> Result<Value, LanaError> {
        let value = Self::dataset_row_key(row, key)?;
        value.check_dataset_key(self.memory_limit.saturating_sub(self.allocated_bytes()))?;
        Ok(self.reactive_value(&value))
    }

    fn dataset_check_number(&self, value: &Value) -> Result<(), LanaError> {
        match &self.reactive_value(value).kind {
            ValueKind::Number(number) if number.is_finite() => Ok(()),
            ValueKind::Possibility(possibility) => {
                for candidate in &possibility.values { self.dataset_check_number(candidate)?; }
                Ok(())
            }
            ValueKind::PathSet(paths) => {
                for alternative in &paths.alternatives { self.dataset_check_number(&alternative.result)?; }
                Ok(())
            }
            ValueKind::Joint(_) | ValueKind::StateDist(_) | ValueKind::Distribution { .. } => Err(LanaError::UnsupportedOperation),
            _ => Err(LanaError::Type),
        }
    }

    /// Compare two values for sort ordering, mirroring `dataset_compare_values`.
    fn dataset_compare_values(a: &Value, b: &Value) -> i32 {
        match (&a.kind, &b.kind) {
            (ValueKind::Number(x), ValueKind::Number(y)) => {
                if x < y { -1 } else if x > y { 1 } else { 0 }
            }
            (ValueKind::String(x), ValueKind::String(y)) => {
                let ord = x.cmp(y);
                if ord == std::cmp::Ordering::Less { -1 } else if ord == std::cmp::Ordering::Greater { 1 } else { 0 }
            }
            (ValueKind::Bool(x), ValueKind::Bool(y)) => {
                (*x as i32) - (*y as i32)
            }
            _ => {
                let dx = a.value_type() as u8 as i32;
                let dy = b.value_type() as u8 as i32;
                if dx != dy { dx - dy } else { 0 }
            }
        }
    }

    fn charge_dataset_work(&mut self, steps: u64) -> Result<(), LanaError> {
        if let Some(remaining) = &mut self.dataset_work_remaining {
            *remaining = remaining.checked_sub(steps).ok_or(LanaError::Limit)?;
        }
        Ok(())
    }

    fn dataset_key_identity(value: &Value, bytes: &mut Vec<u8>, depth: usize) -> Result<(), LanaError> {
        if depth > 64 { return Err(LanaError::Limit); }
        let append = |bytes: &mut Vec<u8>, part: &[u8]| -> Result<(), LanaError> {
            if bytes.len().checked_add(part.len()).is_none_or(|size| size > 64 * 1024 * 1024) {
                return Err(LanaError::Limit);
            }
            bytes.try_reserve(part.len()).map_err(|_| LanaError::Oom)?;
            bytes.extend_from_slice(part);
            Ok(())
        };
        match &value.kind {
            ValueKind::Null => append(bytes, b"n"),
            ValueKind::Bool(flag) => append(bytes, if *flag { b"t" } else { b"f" }),
            ValueKind::String(text) => {
                append(bytes, b"s")?;
                append(bytes, &(text.len() as u64).to_le_bytes())?;
                append(bytes, text.as_bytes())
            }
            ValueKind::Number(number) if number.is_finite() => {
                append(bytes, b"d")?;
                append(bytes, &(if *number == 0.0 { 0.0 } else { *number }).to_bits().to_le_bytes())
            }
            ValueKind::State(state) if state::state_valid(&state.state) => {
                append(bytes, b"q")?;
                for number in [state.state.p, state.state.d_re, state.state.d_im] {
                    append(bytes, &(if number == 0.0 { 0.0 } else { number }).to_bits().to_le_bytes())?;
                }
                Ok(())
            }
            ValueKind::Array(items) => {
                let items = items.lock().unwrap().items().to_vec();
                append(bytes, b"a")?;
                append(bytes, &(items.len() as u64).to_le_bytes())?;
                for item in &items { Self::dataset_key_identity(item, bytes, depth + 1)?; }
                Ok(())
            }
            ValueKind::Map(fields) => {
                let mut fields = fields.lock().unwrap().entries().to_vec();
                fields.sort_by(|a, b| a.key.cmp(&b.key));
                append(bytes, b"m")?;
                append(bytes, &(fields.len() as u64).to_le_bytes())?;
                for entry in &fields {
                    append(bytes, &(entry.key.len() as u64).to_le_bytes())?;
                    append(bytes, entry.key.as_bytes())?;
                    Self::dataset_key_identity(&entry.value, bytes, depth + 1)?;
                }
                Ok(())
            }
            _ => Err(LanaError::UnsupportedValue),
        }
    }

    fn trace_dataset_row(&mut self, output: &mut Value, operation: &str, inputs: &[&Value]) -> Result<(), LanaError> {
        if self.dataset_work_remaining.is_some() && inputs.iter().any(|input| input.derivation.is_some()) {
            let label = if operation == "group_by" {
                let key = Self::dataset_row_key(output, "key")?;
                let mut bytes = Vec::new();
                Self::dataset_key_identity(&key, &mut bytes, 0)?;
                if bytes.len() > (64 * 1024 * 1024 - 32) / 2 { return Err(LanaError::Limit); }
                format!("g{}:{}", bytes.len(), bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>())
            } else if operation == "join" {
                let left = inputs[0].derivation.as_ref().ok_or(LanaError::UnsupportedValue)?.label.as_ref();
                let right = inputs[1].derivation.as_ref().ok_or(LanaError::UnsupportedValue)?.label.as_ref();
                if left.len().checked_add(right.len()).is_none_or(|size| size > 64 * 1024 * 1024 - 64) {
                    return Err(LanaError::Limit);
                }
                format!("j{}:{}{}:{}", left.len(), left, right.len(), right)
            } else {
                inputs[0].derivation.as_ref().ok_or(LanaError::UnsupportedValue)?.label.to_string()
            };
            if label.len() > 64 * 1024 * 1024 { return Err(LanaError::Limit); }
            output.derivation = self.record_derivation(DerivationKind::Operation, operation, inputs, &label, 0,
                DerivationExactness::Exact, "dataset_row", DerivationOutcome::Success,
                if operation == "filter" { "true" } else { "none" });
            if let ValueKind::Map(fields) = &output.kind {
                let guard = fields.lock().unwrap();
                if guard.entries().iter().any(|entry| entry.value.derivation.is_none()) {
                    let entries = guard.entries().to_vec();
                    drop(guard);
                    let mut traced = Map::new(&self.heap, entries.len())?;
                    for entry in entries {
                        let mut cell = entry.value;
                        if cell.derivation.is_none() { cell.derivation = output.derivation.clone(); }
                        traced.set(entry.key, cell, false)?;
                    }
                    output.kind = ValueKind::Map(Arc::new(Mutex::new(traced)));
                }
            }
        }
        Ok(())
    }

    fn exclude_dataset_row(&mut self, row: &Value, operation: &'static str, reason: &'static str, predicate: Option<&Value>) {
        if self.dataset_work_remaining.is_some() {
            if let Some(input) = &row.derivation {
                self.dataset_decisions.push(DatasetDecision {
                    operation, reason, input: input.clone(),
                    predicate: predicate.and_then(|value| value.derivation.clone()),
                    predicate_value: predicate.and_then(|value| match self.reactive_value(value).kind { ValueKind::Bool(result) => Some(result), _ => None }),
                });
            }
        }
    }

    /// Materialize a dataset plan to a `Vec<Value>` of rows, mirroring
    /// `dataset_materialize` in `vm/c/vm.c`.
    fn dataset_materialize(&mut self, dataset: &Dataset, scratch: u32) -> Result<Vec<Value>, LanaError> {
        match dataset.op {
            DatasetOp::Source => {
                self.dataset_materialize_source(&dataset.source, scratch)
            }
            DatasetOp::Filter => {
                let source_rows = self.dataset_materialize(&self.dataset_as(dataset)?, scratch)?;
                let mut result = Vec::new();
                for row in source_rows {
                    self.charge_dataset_work(1)?;
                    let mut pred_result = Value::null();
                    let error = self.run_function(dataset.function, &row, scratch, &mut pred_result);
                    if error != LanaError::Ok {
                        return Err(error);
                    }
                    pred_result.check_dataset_key(self.memory_limit.saturating_sub(self.allocated_bytes()))?;
                    match self.reactive_value(&pred_result).kind {
                        ValueKind::Bool(true) => {
                            let mut retained = row.clone();
                            self.trace_dataset_row(&mut retained, "filter", &[&row, &pred_result])?;
                            result.push(retained);
                        },
                        ValueKind::Bool(false) => self.exclude_dataset_row(&row, "filter", "filter_false", Some(&pred_result)),
                        _ => return Err(LanaError::Type),
                    }
                }
                Ok(result)
            }
            DatasetOp::Map => {
                let source_rows = self.dataset_materialize(&self.dataset_as(dataset)?, scratch)?;
                let mut result = Vec::with_capacity(source_rows.len());
                for row in source_rows {
                    self.charge_dataset_work(1)?;
                    let mut mapped = Value::null();
                    let error = self.run_function(dataset.function, &row, scratch, &mut mapped);
                    if error != LanaError::Ok {
                        return Err(error);
                    }
                    let mut mapped = mapped;
                    let prior = mapped.clone();
                    self.trace_dataset_row(&mut mapped, "map", &[&row, &prior])?;
                    result.push(mapped);
                }
                Ok(result)
            }
            DatasetOp::Select => {
                let ValueKind::Array(columns) = &dataset.columns.kind else {
                    return Err(LanaError::Type);
                };
                let columns = columns.lock().unwrap();
                let source_rows = self.dataset_materialize(&self.dataset_as(dataset)?, scratch)?;
                let mut result = Vec::with_capacity(source_rows.len());
                for row in source_rows {
                    let ValueKind::Map(row_map) = &row.kind else {
                        return Err(LanaError::Type);
                    };
                    let row_map = row_map.lock().unwrap();
                    let mut projected = Map::new(&self.heap, columns.items.len())?;
                    for col in &columns.items {
                        let ValueKind::String(col_name) = &col.kind else {
                            return Err(LanaError::Type);
                        };
                        let col_value = row_map.get(&**col_name).cloned().ok_or(LanaError::Type)?;
                        projected.set(col_name.clone(), col_value, false)?;
                    }
                    let mut selected = Value::map(Arc::new(Mutex::new(projected)));
                    self.trace_dataset_row(&mut selected, "select", &[&row])?;
                    result.push(selected);
                }
                Ok(result)
            }
            DatasetOp::Limit => {
                let ValueKind::Number(n) = dataset.limit.kind else {
                    return Err(LanaError::Type);
                };
                if !n.is_finite() || n < 0.0 {
                    return Err(LanaError::Type);
                }
                let n = n as usize;
                let source_rows = self.dataset_materialize(&self.dataset_as(dataset)?, scratch)?;
                let take = n.min(source_rows.len());
                let mut result = Vec::with_capacity(take);
                for (index, row) in source_rows.into_iter().enumerate() {
                    if index < take {
                        let mut retained = row.clone();
                        self.trace_dataset_row(&mut retained, "limit", &[&row])?;
                        result.push(retained);
                    } else {
                        self.exclude_dataset_row(&row, "limit", "limit_excluded", None);
                    }
                }
                Ok(result)
            }
            DatasetOp::Sort => {
                let ValueKind::String(key) = &dataset.key.kind else {
                    return Err(LanaError::Type);
                };
                let mut result = self.dataset_materialize(&self.dataset_as(dataset)?, scratch)?;
                for row in &result { self.dataset_checked_key(row, key)?; }
                // Insertion sort by key value (rows are maps).
                for i in 1..result.len() {
                    let key_i = self.dataset_checked_key(&result[i], key)?;
                    let pivot = result[i].clone();
                    let mut j = i;
                    while j > 0 {
                        self.charge_dataset_work(1)?;
                        let key_j = self.dataset_checked_key(&result[j - 1], key)?;
                        if Self::dataset_compare_values(&key_j, &key_i) <= 0 {
                            break;
                        }
                        result[j] = result[j - 1].clone();
                        j -= 1;
                    }
                    result[j] = pivot;
                }
                for row in &mut result {
                    let input = row.clone();
                    self.trace_dataset_row(row, "sort", &[&input])?;
                }
                Ok(result)
            }
            DatasetOp::GroupBy => {
                let ValueKind::String(key) = &dataset.key.kind else {
                    return Err(LanaError::Type);
                };
                let source_rows = self.dataset_materialize(&self.dataset_as(dataset)?, scratch)?;
                let mut result: Vec<Value> = Vec::new();
                for row in source_rows {
                    let key_value = self.dataset_checked_key(&row, key)?;
                    // Find an existing group record with this key.
                    let mut group_index: Option<usize> = None;
                    for g in 0..result.len() {
                        self.charge_dataset_work(1)?;
                        let ValueKind::Map(existing) = &result[g].kind else {
                            return Err(LanaError::Type);
                        };
                        let existing_guard = existing.lock().unwrap();
                        let existing_key = existing_guard.get("key").cloned().ok_or(LanaError::Type)?;
                        if set_value_equal(&existing_key, &key_value) {
                            group_index = Some(g);
                            break;
                        }
                    }
                    let group_rows: Arc<Mutex<Array>>;
                    if let Some(index) = group_index {
                        let ValueKind::Map(gm) = &result[index].kind else { return Err(LanaError::Type); };
                        let guard = gm.lock().unwrap();
                        let rows_value = guard.get("rows").cloned().ok_or(LanaError::Type)?;
                        let ValueKind::Array(rows) = rows_value.kind else {
                            return Err(LanaError::Type);
                        };
                        group_rows = rows;
                        drop(guard);
                        group_rows.lock().unwrap().items.push(row.clone())?;
                        let prior = result[index].clone();
                        self.trace_dataset_row(&mut result[index], "group_by", &[&prior, &row])?;
                    } else {
                        let mut new_map = Map::new(&self.heap, 2)?;
                        let new_rows = Arc::new(Mutex::new(Array::from_items(&self.heap, vec![row.clone()])?));
                        new_map.set(Arc::from("key"), key_value, false)?;
                        new_map.set(Arc::from("rows"), Value::array(new_rows.clone()), false)?;
                        let mut group = Value::map(Arc::new(Mutex::new(new_map)));
                        self.trace_dataset_row(&mut group, "group_by", &[&row])?;
                        result.push(group);
                    }
                }
                Ok(result)
            }
            DatasetOp::Aggregate => {
                let ValueKind::Array(agg) = &dataset.aggregate.kind else {
                    return Err(LanaError::Type);
                };
                let agg = agg.lock().unwrap();
                if agg.items.is_empty() {
                    return Err(LanaError::Type);
                }
                let ValueKind::String(agg_op) = &agg.items[0].kind else {
                    return Err(LanaError::Type);
                };
                let agg_col: Option<Arc<str>> = match (&**agg_op, &agg.items[..]) {
                    ("count", [_]) => None,
                    ("sum" | "mean" | "min" | "max", [_, Value { kind: ValueKind::String(col), .. }]) => Some(col.clone()),
                    _ => return Err(LanaError::Type),
                };
                let group_records = self.dataset_materialize(&self.dataset_as(dataset)?, scratch)?;
                let mut result = Vec::with_capacity(group_records.len());
                for record in &group_records {
                    let ValueKind::Map(record_map) = &record.kind else {
                        return Err(LanaError::Type);
                    };
                    let record_guard = record_map.lock().unwrap();
                    let key_value = record_guard.get("key").cloned().ok_or(LanaError::Type)?;
                    let rows_value = record_guard.get("rows").cloned().ok_or(LanaError::Type)?;
                    let ValueKind::Array(rows) = rows_value.kind else {
                        return Err(LanaError::Type);
                    };
                    let rows = rows.lock().unwrap();
                    let mut agg_value = if &**agg_op == "count" {
                        self.charge_dataset_work(rows.items.len() as u64)?;
                        Value::number(rows.items.len() as f64)
                    } else {
                        let col = agg_col.as_ref().ok_or(LanaError::Type)?;
                        let mut acc: Option<Value> = None;
                        for r in &rows.items {
                            self.charge_dataset_work(1)?;
                            let cell = Self::dataset_row_key(r, &**col)?;
                            self.dataset_check_number(&cell)?;
                            acc = Some(if let Some(previous) = acc {
                                let operation = match &**agg_op {
                                    "sum" | "mean" => 0,
                                    "min" => 4,
                                    "max" => 5,
                                    _ => unreachable!(),
                                };
                                let mut next = Value::null();
                                let error = self.lift_binary(&previous, &cell, PureKind::Binary, operation, &mut next);
                                if error != LanaError::Ok { return Err(error); }
                                self.dataset_check_number(&next)?;
                                if previous.derivation.is_some() || cell.derivation.is_some() {
                                    next.derivation = self.record_derivation(DerivationKind::Operation,
                                        agg_op, &[&previous, &cell], "", 0,
                                        DerivationExactness::Exact, "pure", DerivationOutcome::Success, "none");
                                }
                                next
                            } else { cell });
                        }
                        let mut acc = acc.unwrap_or_else(|| Value::number(0.0));
                        if rows.items.is_empty() && &**agg_op != "sum" { return Err(LanaError::Type); }
                        if &**agg_op == "mean" {
                            let count = Value::number(rows.items.len() as f64);
                            let mut mean = Value::null();
                            let error = self.lift_binary(&acc, &count, PureKind::Binary, 3, &mut mean);
                            if error != LanaError::Ok { return Err(error); }
                            self.dataset_check_number(&mean)?;
                            if acc.derivation.is_some() {
                                mean.derivation = self.record_derivation(DerivationKind::Operation,
                                    "mean", &[&acc, &count], "", 0,
                                    DerivationExactness::Exact, "pure", DerivationOutcome::Success, "none");
                            }
                            acc = mean;
                        }
                        acc
                    };
                    if self.dataset_work_remaining.is_some() {
                        agg_value.derivation = self.record_derivation(DerivationKind::Operation,
                            agg_op, &[record, &agg_value], "", 0,
                            DerivationExactness::Exact, "dataset_cell", DerivationOutcome::Success, "none");
                    }
                    // Output row: {<group key name>: key_value, <op>: agg_value}.
                    let mut key_name = Arc::from("key");
                    if let ValueKind::Dataset(source) = &dataset.source.kind {
                        if source.op == DatasetOp::GroupBy {
                            if let ValueKind::String(sk) = &source.key.kind {
                                key_name = sk.clone();
                            }
                        }
                    }
                    let mut out_row = Map::new(&self.heap, 2)?;
                    out_row.set(key_name, key_value, false)?;
                    out_row.set(agg_op.clone(), agg_value.clone(), false)?;
                    let mut aggregated = Value::map(Arc::new(Mutex::new(out_row)));
                    self.trace_dataset_row(&mut aggregated, "aggregate", &[record, &agg_value])?;
                    result.push(aggregated);
                }
                Ok(result)
            }
            DatasetOp::Join => {
                let ValueKind::String(key) = &dataset.key.kind else {
                    return Err(LanaError::Type);
                };
                let ValueKind::Dataset(other) = &dataset.other.kind else {
                    return Err(LanaError::Type);
                };
                let left_rows = self.dataset_materialize(&self.dataset_as(dataset)?, scratch)?;
                let right_rows = self.dataset_materialize(other, scratch)?;
                for row in left_rows.iter().chain(&right_rows) { self.dataset_checked_key(row, key)?; }
                let mut result = Vec::new();
                let mut right_matches = self.dataset_work_remaining.map(|_| vec![false; right_rows.len()]);
                for left in &left_rows {
                    let left_key = self.dataset_checked_key(left, key)?;
                    let mut matched_left = false;
                    for (right_index, right) in right_rows.iter().enumerate() {
                        self.charge_dataset_work(1)?;
                        let right_key = self.dataset_checked_key(right, key)?;
                        if !set_value_equal(&left_key, &right_key) {
                            continue;
                        }
                        matched_left = true;
                        if let Some(matches) = &mut right_matches { matches[right_index] = true; }
                        if self.dataset_work_remaining.is_some() && result.len() == 100_000 { return Err(LanaError::Limit); }
                        // Merge left and right rows into one map.
                        let ValueKind::Map(left_map) = &left.kind else {
                            return Err(LanaError::Type);
                        };
                        let ValueKind::Map(right_map) = &right.kind else {
                            return Err(LanaError::Type);
                        };
                        // Snapshot separately: a self-join may reference the same mutex.
                        let left_entries = left_map.lock().unwrap().entries.to_vec();
                        let right_entries = right_map.lock().unwrap().entries.to_vec();
                        let mut merged = Map::new(&self.heap, left_entries.len() + right_entries.len())?;
                        for entry in &left_entries {
                            merged.set(entry.key.clone(), entry.value.clone(), false)?;
                        }
                        for entry in &right_entries {
                            merged.set(entry.key.clone(), entry.value.clone(), false)?;
                        }
                        let mut joined = Value::map(Arc::new(Mutex::new(merged)));
                        self.trace_dataset_row(&mut joined, "join", &[left, right])?;
                        result.push(joined);
                    }
                    if !matched_left { self.exclude_dataset_row(left, "join", "no_matching_key", None); }
                }
                if let Some(matches) = right_matches {
                    for (right, matched) in right_rows.iter().zip(matches) {
                        if !matched { self.exclude_dataset_row(right, "join", "no_matching_key", None); }
                    }
                }
                Ok(result)
            }
        }
    }

    /// Extract the upstream dataset from a plan node's `source` field.
    fn dataset_as(&self, dataset: &Dataset) -> Result<Dataset, LanaError> {
        let ValueKind::Dataset(source) = &dataset.source.kind else {
            return Err(LanaError::Type);
        };
        Ok((**source).clone())
    }

    /// Build an inspectable plan value for `explain(ds)`, mirroring
    /// `dataset_explain` in `vm/c/vm.c`.
    fn dataset_explain(&self, dataset: &Dataset) -> Result<Value, LanaError> {
        let op_names = ["source", "filter", "map", "select", "limit", "sort",
                        "group_by", "aggregate", "join"];
        let mut map = Map::new(&self.heap, 4)?;
        let op_name = op_names[dataset.op as usize];
        map.set(Arc::from("op"), Value::string(Arc::from(op_name)), false)?;
        if dataset.op == DatasetOp::Source {
            /* LIP-015 §3: a source is either a lazy generator (report its bound)
             * or an in-memory array of rows (report the row count). */
            match &dataset.source.kind {
                ValueKind::Lazy { bound, .. } => {
                    map.set(Arc::from("bound"), Value::number(*bound as f64), false)?;
                }
                ValueKind::Array(array) => {
                    let count = array.lock().unwrap().items.len();
                    map.set(Arc::from("rows"), Value::number(count as f64), false)?;
                }
                _ => return Err(LanaError::Type),
            }
        } else {
            let ValueKind::Dataset(source) = &dataset.source.kind else {
                return Err(LanaError::Type);
            };
            let source_value = self.dataset_explain(source)?;
            map.set(Arc::from("source"), source_value, false)?;
        }
        if dataset.op == DatasetOp::Filter || dataset.op == DatasetOp::Map {
            map.set(Arc::from("function"), Value::function(dataset.function), false)?;
        }
        if dataset.op == DatasetOp::Select {
            map.set(Arc::from("columns"), dataset.columns.clone(), false)?;
        }
        if dataset.op == DatasetOp::Limit {
            map.set(Arc::from("limit"), dataset.limit.clone(), false)?;
        }
        if dataset.op == DatasetOp::Sort || dataset.op == DatasetOp::GroupBy
            || dataset.op == DatasetOp::Join {
            map.set(Arc::from("key"), dataset.key.clone(), false)?;
        }
        if dataset.op == DatasetOp::Aggregate {
            map.set(Arc::from("aggregate"), dataset.aggregate.clone(), false)?;
        }
        if dataset.op == DatasetOp::Join {
            let ValueKind::Dataset(other) = &dataset.other.kind else {
                return Err(LanaError::Type);
            };
            let other_value = self.dataset_explain(other)?;
            map.set(Arc::from("other"), other_value, false)?;
        }
        Ok(Value::map(Arc::new(Mutex::new(map))))
    }

    /// Poll a composite future (`future_all` / `future_race` / `sleep`) for
    /// completion. When done, marks it complete, stores its result, and
    /// re-queues any futures awaiting it. When not done, leaves it suspended;
    /// it is re-queued when an input future completes.
    fn poll_composite_future(&mut self, future: Arc<Mutex<Future>>) -> Result<(), LanaError> {
        let (kind, inputs) = {
            let guard = future.lock().unwrap();
            let kind = guard.registers[0].clone();
            let inputs = guard.registers[1..].to_vec();
            (kind, inputs)
        };
        let ValueKind::String(kind) = kind.kind else {
            return Ok(());
        };
        let mut done = false;
        let mut result = Value::null();
        if &*kind == "all" {
            let all_done = inputs.iter().all(|item| {
                matches!(&item.kind, ValueKind::Future(f) if f.lock().unwrap().exhausted)
            });
            if all_done {
                done = true;
                let mut items = Vec::with_capacity(inputs.len());
                for item in &inputs {
                    let ValueKind::Future(f) = &item.kind else {
                        return Ok(());
                    };
                    items.push(f.lock().unwrap().registers[0].clone());
                }
                result = self.array_value(items)?;
            }
        } else if &*kind == "race" {
            for item in &inputs {
                let ValueKind::Future(f) = &item.kind else {
                    return Ok(());
                };
                let f = f.lock().unwrap();
                if f.exhausted {
                    done = true;
                    result = f.registers[0].clone();
                    break;
                }
            }
        } else if &*kind == "sleep" {
            #[cfg(not(target_arch = "wasm32"))]
            let ms = match &inputs[0].kind {
                ValueKind::Number(ms) => *ms,
                _ => 0.0,
            };
            #[cfg(not(target_arch = "wasm32"))]
            if ms > 0.0 {
                std::thread::sleep(std::time::Duration::from_millis(ms as u64));
            }
            done = true;
            result = Value::null();
        }
        if done {
            let mut guard = future.lock().unwrap();
            self.heap.mutate_cycle(Arc::as_ptr(&future) as usize);
            guard.exhausted = true;
            guard.ready = false;
            guard.queued = false;
            guard.registers[0] = result;
            if let Some(awaiters) = self.awaiters.remove(&(Arc::as_ptr(&future) as usize)) {
                for awaiter in awaiters {
                    self.enqueue_future(awaiter);
                }
            }
        }
        Ok(())
    }

    fn host_read_text(&mut self, argument: &Value, out: &mut Value) -> LanaError {
        let ValueKind::String(path) = &argument.kind else {
            return LanaError::Type;
        };
        if let Some(fs) = &self.virtual_fs {
            return match fs.get(&**path) {
                Some(text) => self.string_output(text, out), None => LanaError::Io,
            };
        }
        use std::io::Read;
        let mut file = match std::fs::File::open(&**path) {
            Ok(file) => file, Err(_) => return LanaError::Io,
        };
        let mut bytes = match Buffer::new(&self.heap, 0, 0) {
            Ok(buffer) => buffer, Err(error) => return error,
        };
        let mut block = [0u8; 8192];
        loop {
            match file.read(&mut block) {
                Ok(0) => break,
                Ok(count) => if let Err(error) = bytes.extend_from_slice(&block[..count]) { return error; },
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => return LanaError::Io,
            }
        }
        *out = match self.heap.lossy_string(&bytes) {
            Ok(text) => Value::string(text), Err(error) => return error,
        };
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
        let mut items = Vec::with_capacity(names.len());
        for name in &names {
            let full = format!("{path}/{name}");
            let metadata = match std::fs::metadata(&full) {
                Ok(metadata) => metadata,
                Err(_) => return LanaError::Io,
            };
            let kind = if metadata.is_dir() { "directory" } else { "file" };
            let mut map = match Map::new(&self.heap, 2) { Ok(map) => map, Err(error) => return error };
            if let Err(error) = map.set(Arc::from("name"), Value::string(Arc::from(name.clone())), true) {
                return error;
            }
            if let Err(error) = map.set(Arc::from("kind"), Value::string(Arc::from(kind)), true) {
                return error;
            }
            items.push(Value::map(Arc::new(Mutex::new(map))));
        }
        *out = match self.array_value(items) { Ok(value) => value, Err(error) => return error };
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
        match write_text_atomic_with_sync(
            Path::new(&**path),
            contents.as_bytes(),
            |file| file.sync_all(),
            |directory| directory.sync_all(),
        ) {
            Ok(()) => LanaError::Ok,
            Err((error, uncertain)) => {
                self.pending_io_outcome = Some((uncertain, path.to_string()));
                self.pending_error_message = Some(if uncertain {
                    format!("file replaced; durability: uncertain; path: {path}; {error}")
                } else {
                    format!("atomic write failed before replacement; path: {path}; {error}")
                });
                LanaError::Io
            }
        }
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
        self.string_output(&hex16(hash), out)
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
        self.string_output(&result, out)
    }

    fn package_path_allowed(&self, path: &std::path::Path) -> bool {
        let mut in_lana = false;
        for component in path.components() {
            if in_lana && component.as_os_str() == "packages" {
                return self.package_paths.values().any(|root| path.starts_with(root));
            }
            in_lana = component.as_os_str() == ".lana";
        }
        true
    }

    fn host_path_resolve(&mut self, base: &str, relative: &str, out: &mut Value) -> LanaError {
        if let Some(import) = relative.strip_prefix("pkg/") {
            let parts = import.split('/').collect::<Vec<_>>();
            if parts.len() < 4 || parts[2] != "src" ||
                parts.iter().any(|part| part.is_empty() || *part == "." || *part == ".." ||
                    part.contains(['\\', ':']) || part.chars().any(char::is_control)) {
                return LanaError::Schema;
            }
            let identity = format!("{}/{}", parts[0], parts[1]);
            let Some(root) = self.package_paths.get(&identity) else { return LanaError::UnsupportedValue; };
            let root = std::path::Path::new(root);
            let path = match std::fs::canonicalize(root.join(parts[2..].join("/"))) {
                Ok(path) if path.starts_with(root.join("src")) && path.is_file() => path,
                _ => return LanaError::Schema,
            };
            return self.string_output(&path.to_string_lossy(), out);
        }
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
            return self.string_output(&resolved, out);
        }
        if relative.is_empty() {
            let resolved = match std::fs::canonicalize(base) {
                Ok(resolved) => resolved,
                Err(_) => return LanaError::Io,
            };
            if !self.package_path_allowed(&resolved) { return LanaError::UnsupportedValue; }
            return self.string_output(&resolved.to_string_lossy(), out);
        }
        let candidate = match base.rfind('/') {
            Some(index) => format!("{}/{relative}", &base[..index]),
            None => format!("./{relative}"),
        };
        let resolved = match std::fs::canonicalize(&candidate) {
            Ok(resolved) => resolved,
            Err(_) => return LanaError::Io,
        };
        if !self.package_path_allowed(&resolved) { return LanaError::UnsupportedValue; }
        for root in self.package_paths.values() {
            let root = std::path::Path::new(root);
            if std::path::Path::new(base).starts_with(root) && !resolved.starts_with(root) {
                return LanaError::Schema;
            }
        }
        self.string_output(&resolved.to_string_lossy(), out)
    }

    fn json_parse(&mut self, text: &str, out: &mut Value) -> LanaError {
        let bytes = text.as_bytes();
        if !utf8_valid(bytes) {
            *out = match self.make_result(false, Value::string(Arc::from("invalid JSON at byte 0"))) { Ok(value) => value, Err(error) => return error };
            return LanaError::Ok;
        }
        let mut pos = 0;
        let result = self.json_value(bytes, &mut pos, 0);
        json_space(bytes, &mut pos);
        let value = match result {
            Ok(value) if pos == bytes.len() => value,
            Err(error @ (LanaError::Oom | LanaError::Limit)) => return error,
            _ => {
                *out = match self.make_result(
                    false,
                    Value::string(Arc::from(format!("invalid JSON at byte {pos}"))),
                ) { Ok(value) => value, Err(error) => return error };
                return LanaError::Ok;
            }
        };
        // LIP-023 §4: root the parsed value as Information and record the
        // source-text identity (SHA-256) so a decision that consumed the record
        // can be audited and replayed against the exact input.
        let mut rooted = match self.reactive_root(&value, DerivationExactness::Exact) {
            Ok(rooted) => rooted,
            Err(error) => return error,
        };
        let label = hex_digest(&sha256(bytes));
        rooted.derivation = self.record_derivation(
            DerivationKind::Evidence,
            "json_parse",
            &[],
            &label,
            0,
            DerivationExactness::Exact,
            "root",
            DerivationOutcome::Success,
            "none",
        );
        *out = match self.make_result(true, rooted) { Ok(value) => value, Err(error) => return error };
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
                Ok(Value::string(string))
            }
            b'[' => {
                *pos += 1;
                json_space(bytes, pos);
                let mut array = Array::new(&self.heap, 0)?;
                while *pos < bytes.len() && bytes[*pos] != b']' {
                    let item = self.json_value(bytes, pos, depth + 1)?;
                    array.push(item)?;
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
                Ok(Value::array(Arc::new(Mutex::new(array))))
            }
            b'{' => {
                *pos += 1;
                json_space(bytes, pos);
                let mut map = Map::new(&self.heap, 4)?;
                while *pos < bytes.len() && bytes[*pos] != b'}' {
                    let key = self.json_string(bytes, pos)?;
                    json_space(bytes, pos);
                    if *pos >= bytes.len() || bytes[*pos] != b':' {
                        return Err(LanaError::Parse);
                    }
                    *pos += 1;
                    let item = self.json_value(bytes, pos, depth + 1)?;
                    map.set(key, item, false)?;
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
                let start = *pos;
                let number = json_number(bytes, pos)?;
                if json_large_integer(&bytes[start..*pos]) {
                    let text = std::str::from_utf8(&bytes[start..*pos]).map_err(|_| LanaError::Parse)?;
                    self.string_value(text)
                } else {
                    Ok(Value::number(number))
                }
            }
        }
    }

    fn json_string(&self, bytes: &[u8], pos: &mut usize) -> Result<Arc<str>, LanaError> {
        if *pos >= bytes.len() || bytes[*pos] != b'"' {
            return Err(LanaError::Parse);
        }
        *pos += 1;
        let mut out = Buffer::new(&self.heap, 0, 0)?;
        while *pos < bytes.len() && bytes[*pos] != b'"' {
            let c = bytes[*pos];
            *pos += 1;
            if c < 0x20 {
                return Err(LanaError::Parse);
            }
            if c != b'\\' {
                out.push(c)?;
                continue;
            }
            if *pos >= bytes.len() {
                return Err(LanaError::Parse);
            }
            let esc = bytes[*pos];
            *pos += 1;
            match esc {
                b'"' | b'\\' | b'/' => out.push(esc)?,
                b'b' => out.push(8)?,
                b'f' => out.push(12)?,
                b'n' => out.push(b'\n')?,
                b'r' => out.push(b'\r')?,
                b't' => out.push(b'\t')?,
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
                        Some(ch) => out.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes())?,
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
        let text = std::str::from_utf8(&out).map_err(|_| LanaError::Parse)?;
        self.heap.string(text)
    }

    fn json_stringify(&mut self, value: &Value, out: &mut Value) -> LanaError {
        let mut buffer = match Buffer::new(&self.heap, 0, 0) {
            Ok(buffer) => buffer, Err(error) => return error,
        };
        let mut stack = match Buffer::new(&self.heap, 0, 0) {
            Ok(buffer) => buffer, Err(error) => return error,
        };
        if let Err(error) = self.json_emit(value, &mut buffer, &mut stack, 0) {
            return error;
        }
        self.string_output(std::str::from_utf8(&buffer).unwrap(), out)
    }

    fn json_emit(
        &self,
        value: &Value,
        buffer: &mut Buffer<u8>,
        stack: &mut Buffer<usize>,
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
            if stack.contains(&id) {
                return Err(LanaError::UnsupportedOperation);
            }
            stack.push(id)?;
        }
        let result = match &value.kind {
            ValueKind::Null => {
                buffer.extend_from_slice(b"null")?;
                Ok(())
            }
            ValueKind::Bool(boolean) => {
                buffer.extend_from_slice(if *boolean { b"true" } else { b"false" })?;
                Ok(())
            }
            ValueKind::Number(number) => {
                if !number.is_finite() {
                    return Err(LanaError::UnsupportedOperation);
                }
                if *number == 0.0 {
                    buffer.push(b'0')?;
                } else {
                    buffer.extend_from_slice(format_g17(*number).as_bytes())?;
                }
                Ok(())
            }
            ValueKind::String(string) => self.json_escape(string, buffer),
            ValueKind::ObjectValue(object) => {
                if object.descriptor.fields.iter().any(|field| field.visibility != "public") {
                    return Err(LanaError::UnsupportedOperation);
                }
                buffer.extend_from_slice(b"{\"$lana_value\":")?;
                self.json_escape(&object.descriptor.qualified_name, buffer)?;
                buffer.extend_from_slice(b",\"fields\":{")?;
                let mut fields = Buffer::new(&self.heap, object.fields.len(), 0)?;
                fields.extend(object.descriptor.fields.iter().zip(&object.fields))?;
                fields.sort_unstable_by(|a, b| a.0.name.cmp(&b.0.name));
                for (index, (field, value)) in fields.iter().enumerate() {
                    if index != 0 { buffer.push(b',')?; }
                    self.json_escape(&field.name, buffer)?;
                    buffer.push(b':')?;
                    self.json_emit(value, buffer, stack, depth + 1)?;
                }
                buffer.extend_from_slice(b"}}")?;
                Ok(())
            }
            ValueKind::Array(array) => {
                buffer.push(b'[')?;
                let array = array.lock().unwrap();
                for (index, item) in array.items.iter().enumerate() {
                    if index > 0 {
                        buffer.push(b',')?;
                    }
                    self.json_emit(item, buffer, stack, depth + 1)?;
                }
                buffer.push(b']')?;
                Ok(())
            }
            ValueKind::Map(map) => {
                buffer.push(b'{')?;
                let map = map.lock().unwrap();
                let mut entries = Buffer::new(&self.heap, map.entries.len(), 0)?;
                entries.extend(map.entries.iter())?;
                entries.sort_unstable_by(|a, b| a.key.cmp(&b.key));
                for (index, entry) in entries.iter().enumerate() {
                    if index > 0 {
                        buffer.push(b',')?;
                    }
                    self.json_escape(&entry.key, buffer)?;
                    buffer.push(b':')?;
                    self.json_emit(&entry.value, buffer, stack, depth + 1)?;
                }
                buffer.push(b'}')?;
                Ok(())
            }
            _ => Err(LanaError::UnsupportedOperation),
        };
        if identity.is_some() {
            stack.pop();
        }
        result
    }

    fn json_escape(&self, text: &str, buffer: &mut Buffer<u8>) -> Result<(), LanaError> {
        if !utf8_valid(text.as_bytes()) {
            return Err(LanaError::Parse);
        }
        buffer.push(b'"')?;
        for c in text.chars() {
            if c == '"' || c == '\\' {
                buffer.extend_from_slice(&[b'\\', c as u8])?;
            } else if c < '\u{20}' {
                buffer.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes())?;
            } else {
                buffer.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes())?;
            }
        }
        buffer.push(b'"')?;
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
            let mut map = match Map::new(&self.heap, records[0].len()) { Ok(map) => map, Err(error) => return error };
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
        *out = match self.array_value(items) { Ok(value) => value, Err(error) => return error };
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
        self.promote_shared_graph(&base_snapshot)?;
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
            revoked: AtomicBool::new(false),
        });
        shared.state.lock().unwrap().capabilities.push(admin.clone());
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
            revoked: AtomicBool::new(false),
        });
        state.capabilities.push(capability.clone());
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
        target.revoked.store(true, Ordering::Release);
        state.capability_epoch += 1;
        shared.condition.notify_all();
        LanaError::Ok
    }

    fn shared_capability_invalidate(&self, target: &Arc<CapabilityToken>) -> LanaError {
        let shared = target.shared.clone();
        let mut state = shared.state.lock().unwrap();
        target.revoked.store(true, Ordering::Release);
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
        self.promote_shared_graph(&pending_evidence)?;
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
                self.heap.bump_mutation_epoch();
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
            let snapshot = self.reactive_observe(&local_source, &local_evidence, 0u32)?;
            versions.push(SharedVersion {
                effective_time: observation.effective_time,
                observation_sequence: observation.sequence,
                snapshot: snapshot.clone(),
            });
            source = snapshot;
        }
        for version in &versions {
            self.promote_shared_graph(&version.snapshot)?;
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
        macro_rules! checked_set {
            ($map:expr, $key:expr, $value:expr, $reject:expr $(,)?) => {
                if let Err(error) = $map.set($key, $value, $reject) { return error; }
            };
        }
        let mut inspection = match Map::new(&self.heap, 28) { Ok(map) => map, Err(error) => return error };
        if let ValueKind::Capability(capability) = &argument.kind {
            let shared = capability.shared.clone();
            let state = shared.state.lock().unwrap();
            let revision = state
                .current
                .as_ref()
                .map(|commit| commit.revision)
                .unwrap_or(0);
            checked_set!(inspection,
                Arc::from("kind"),
                Value::string(Arc::from("shared_information")),
                true,
            );
            checked_set!(inspection, Arc::from("record_schema"), Value::number(1.0), true);
            checked_set!(inspection, Arc::from("id"), Value::string(Arc::from(format!("shared/{}", shared.identity))), true);
            checked_set!(inspection, Arc::from("transport_status"), Value::string(Arc::from("ok")), true);
            checked_set!(inspection, Arc::from("domain_status"), Value::string(Arc::from("available")), true);
            checked_set!(inspection, Arc::from("payload"), Value::null(), true);
            checked_set!(inspection, Arc::from("error"), Value::null(), true);
            let empty = match self.array_value(Vec::new()) { Ok(value) => value, Err(error) => return error };
            checked_set!(inspection, Arc::from("evidence"), empty.clone(), true);
            checked_set!(inspection, Arc::from("assumptions"), empty, true);
            checked_set!(inspection, Arc::from("exactness"), Value::string(Arc::from("exact")), true);
            let mut metadata = match Map::new(&self.heap, 2) { Ok(map) => map, Err(error) => return error };
            checked_set!(metadata, Arc::from("identity"), Value::number(shared.identity as f64), true);
            checked_set!(metadata, Arc::from("revision"), Value::number(revision as f64), true);
            checked_set!(inspection, Arc::from("metadata"), Value::map(Arc::new(Mutex::new(metadata))), true);
            checked_set!(inspection, Arc::from("identity"), Value::number(shared.identity as f64), true);
            checked_set!(inspection, Arc::from("revision"), Value::number(revision as f64), true);
            checked_set!(inspection,
                Arc::from("can_read"),
                Value::boolean(capability_allows_locked(&shared, capability, LANA_CAPABILITY_READ)),
                true,
            );
            checked_set!(inspection,
                Arc::from("can_observe"),
                Value::boolean(capability_allows_locked(&shared, capability, LANA_CAPABILITY_OBSERVE)),
                true,
            );
            checked_set!(inspection,
                Arc::from("can_admin"),
                Value::boolean(capability_allows_locked(&shared, capability, LANA_CAPABILITY_ADMIN)),
                true,
            );
            *out = Value::map(Arc::new(Mutex::new(inspection)));
            return LanaError::Ok;
        }
        let value = self.reactive_value(argument);
        let mut seen = HashSet::new();
        let mut stack = argument.derivation.iter().cloned().collect::<Vec<_>>();
        let mut evidence_nodes = Vec::new();
        let mut assumption_nodes = Vec::new();
        while let Some(node) = stack.pop() {
            if !seen.insert((node.task_lineage, node.local_sequence)) { continue; }
            match node.kind {
                DerivationKind::Evidence => evidence_nodes.push(node.clone()),
                DerivationKind::Assumption => assumption_nodes.push(node.clone()),
                _ => {}
            }
            stack.extend(node.inputs.iter().cloned());
        }
        evidence_nodes.sort_by_key(|node| (node.task_lineage, node.local_sequence));
        assumption_nodes.sort_by_key(|node| (node.task_lineage, node.local_sequence));
        let mut evidence_ids = Vec::with_capacity(evidence_nodes.len());
        let mut assumption_ids = Vec::with_capacity(assumption_nodes.len());
        for node in evidence_nodes {
            evidence_ids.push(match self.derivation_id_to_value(&node) { Ok(id) => id, Err(error) => return error });
        }
        for node in assumption_nodes {
            assumption_ids.push(match self.derivation_id_to_value(&node) { Ok(id) => id, Err(error) => return error });
        }
        let evidence_ids = match self.array_value(evidence_ids) { Ok(value) => value, Err(error) => return error };
        let assumption_ids = match self.array_value(assumption_ids) { Ok(value) => value, Err(error) => return error };
        let record_id = if let Some(reactive) = &argument.reactive {
            Value::string(Arc::from(format!("information/{}", reactive.lock().unwrap().id)))
        } else if let Some(derivation) = &argument.derivation {
            Value::string(Arc::from(format!("derivation/{}/{}", derivation.task_lineage, derivation.local_sequence)))
        } else {
            Value::null()
        };
        checked_set!(inspection, Arc::from("record_schema"), Value::number(1.0), true);
        checked_set!(inspection, Arc::from("id"), record_id, true);
        checked_set!(inspection, Arc::from("transport_status"), Value::string(Arc::from("ok")), true);
        checked_set!(inspection, Arc::from("domain_status"), Value::string(Arc::from("available")), true);
        checked_set!(inspection, Arc::from("payload"), Value::null(), true);
        checked_set!(inspection, Arc::from("error"), Value::null(), true);
        checked_set!(inspection, Arc::from("evidence"), evidence_ids, true);
        checked_set!(inspection, Arc::from("assumptions"), assumption_ids, true);
        let mut metadata = match Map::new(&self.heap, 5) { Ok(map) => map, Err(error) => return error };
        if let Some(node) = &argument.derivation {
            let id = match self.derivation_id_to_value(node) { Ok(id) => id, Err(error) => return error };
            checked_set!(metadata, Arc::from("derivation_id"), id, true);
        }
        if let Some(reactive) = &argument.reactive {
            let reactive = reactive.lock().unwrap();
            let relationship = match reactive.relationship {
                RelationshipKind::SameDependency => "same_dependency",
                RelationshipKind::ExplicitJoint => "explicit_joint",
                _ => "exact",
            };
            checked_set!(metadata, Arc::from("dependency_identity"), Value::number(reactive.dependency_id as f64), true);
            checked_set!(metadata, Arc::from("relationship"), Value::string(Arc::from(relationship)), true);
            checked_set!(metadata, Arc::from("revision"), Value::number(reactive.revision as f64), true);
            checked_set!(metadata, Arc::from("history_count"), Value::number(reactive.history.len() as f64), true);
        }
        checked_set!(inspection, Arc::from("metadata"), Value::map(Arc::new(Mutex::new(metadata))), true);
        let alternatives = match &value.kind {
            ValueKind::Possibility(possibility) => possibility.values.len(),
            ValueKind::PathSet(paths) => paths.alternatives.len(),
            ValueKind::Joint(joint) => joint.rows.len(),
            ValueKind::StateDist(_) => 0,
            _ => 1,
        };
        let form = match &value.kind {
            ValueKind::Possibility(possibility) if possibility.weights.is_some() => "distribution",
            ValueKind::Possibility(_) => "possibility",
            ValueKind::Joint(_) => "joint",
            ValueKind::PathSet(_) => "paths",
            ValueKind::StateDist(_) => "state_dist",
            _ => "definite",
        };
        checked_set!(inspection, Arc::from("form"), Value::string(Arc::from(form)), true);
        checked_set!(inspection,
            Arc::from("finite_support"),
            Value::boolean(match &value.kind {
                ValueKind::Joint(joint) => !joint.rows.is_empty(),
                ValueKind::StateDist(_) => false,
                _ => true,
            }),
            true,
        );
        checked_set!(inspection,
            Arc::from("kind"),
            Value::string(Arc::from("information_snapshot")),
            true,
        );
        checked_set!(inspection,
            Arc::from("type"),
            Value::string(Arc::from(value.type_name())),
            true,
        );
        checked_set!(inspection,
            Arc::from("revision"),
            Value::number(
                argument
                    .reactive
                    .as_ref()
                    .map(|reactive| reactive.lock().unwrap().revision as f64)
                    .unwrap_or_else(|| argument.derivation.as_ref()
                        .filter(|node| node.operation.as_ref() == "snapshot")
                        .map_or(0.0, |node| node.revision as f64)),
            ),
            true,
        );
        checked_set!(inspection,
            Arc::from("remaining_alternatives"),
            Value::number(alternatives as f64),
            true,
        );
        checked_set!(inspection,
            Arc::from("reactive"),
            Value::boolean(argument.reactive.is_some()),
            true,
        );
        checked_set!(inspection,
            Arc::from("sample"),
            Value::boolean(matches!(argument.kind, ValueKind::Sample(_))),
            true,
        );
        checked_set!(inspection,
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
        checked_set!(inspection, Arc::from("exactness"),
            Value::string(Arc::from(if argument.derivation.as_ref().is_some_and(|d| d.exactness == DerivationExactness::Approximate) { "approximate" } else { "exact" })),
            true);
        if let ValueKind::Possibility(possibility) = &value.kind {
            let mut unique: Vec<(&Value, f64)> = Vec::new();
            for (index, candidate) in possibility.values.iter().enumerate() {
                let weight = possibility.weights.as_ref().map_or(0.0, |weights| weights[index]);
                if let Some((_, total)) = unique.iter_mut().find(|(value, _)| joint_value_equal(value, candidate)) {
                    *total += weight;
                } else {
                    unique.push((candidate, weight));
                }
            }
            let mut support = Vec::with_capacity(unique.len());
            for (candidate, weight) in unique {
                let mut row = match Map::new(&self.heap, 2) { Ok(map) => map, Err(error) => return error };
                let mut memo = DeepCloneMemo::default();
                let candidate = match self.deep_clone_value(candidate, &mut memo) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                checked_set!(row, Arc::from("value"), candidate, true);
                if possibility.weights.is_some() {
                    checked_set!(row, Arc::from("weight"), Value::number(weight), true);
                }
                support.push(Value::map(Arc::new(Mutex::new(row))));
            }
            let support = match self.array_value(support) { Ok(value) => value, Err(error) => return error };
            checked_set!(inspection, Arc::from("support"), support, true);
        }
        if let ValueKind::Joint(joint) = &value.kind {
            if !joint.rows.is_empty() {
                let mut support = Vec::with_capacity(joint.rows.len());
                for joint_row in &joint.rows {
                    let mut assignment = match Map::new(&self.heap, joint.names.len()) { Ok(map) => map, Err(error) => return error };
                    let mut memo = DeepCloneMemo::default();
                    for (name, candidate) in joint.names.iter().zip(&joint_row.values) {
                        let candidate = match self.deep_clone_value(candidate, &mut memo) { Ok(value) => value, Err(error) => return error };
                        if let Err(error) = assignment.set(name.clone(), candidate, true) { return error; }
                    }
                    let mut row = match Map::new(&self.heap, 2) { Ok(map) => map, Err(error) => return error };
                    checked_set!(row, Arc::from("assignment"), Value::map(Arc::new(Mutex::new(assignment))), true);
                    checked_set!(row, Arc::from("weight"), Value::number(joint_row.weight), true);
                    support.push(Value::map(Arc::new(Mutex::new(row))));
                }
                let support = match self.array_value(support) { Ok(value) => value, Err(error) => return error };
                checked_set!(inspection, Arc::from("support"), support, true);
            }
        }
        if let Some(reactive) = &argument.reactive {
            let reactive = reactive.lock().unwrap();
            let relationship = match reactive.relationship {
                RelationshipKind::SameDependency => "same_dependency",
                RelationshipKind::ExplicitJoint => "explicit_joint",
                _ => "exact",
            };
            checked_set!(inspection,
                Arc::from("dependency_identity"),
                Value::number(reactive.dependency_id as f64),
                true,
            );
            checked_set!(inspection,
                Arc::from("relationship"),
                Value::string(Arc::from(relationship)),
                true,
            );
            checked_set!(inspection,
                Arc::from("history_count"),
                Value::number(reactive.history.len() as f64),
                true,
            );
            checked_set!(inspection,
                Arc::from("exactness"),
                Value::string(Arc::from(derivation::exactness_name(reactive.exactness))),
                false,
            );
        }
        checked_set!(inspection,
            Arc::from("planned_effect"),
            Value::boolean(argument.planned_effect.is_some()),
            true,
        );
        if let Some(claim) = &argument.claim {
            let mut summary = match Map::new(&self.heap, 4) { Ok(map) => map, Err(error) => return error };
            checked_set!(summary, Arc::from("proposition"), Value::string(claim.proposition.clone()), true);
            checked_set!(summary, Arc::from("exactness"), Value::string(Arc::from(derivation::exactness_name(claim.exactness))), true);
            checked_set!(summary, Arc::from("tolerance"), Value::number(claim.tolerance), true);
            checked_set!(summary, Arc::from("source_valid"), Value::boolean(claim.source_valid), true);
            checked_set!(inspection, Arc::from("claim"), Value::map(Arc::new(Mutex::new(summary))), true);
            checked_set!(inspection, Arc::from("exactness"), Value::string(Arc::from(derivation::exactness_name(claim.exactness))), false);
        }
        if argument.derivation.is_some() {
            let derivation = match self.vm_derivation(argument) {
                Ok(derivation) => derivation,
                Err(error) => return error,
            };
            checked_set!(inspection, Arc::from("derivation"), derivation, true);
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

/// Render a number with a fixed number of decimal places, matching C's `%.*f`
/// (including the lowercase `nan`/`inf` spellings).
fn format_fixed(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() { "inf" } else { "-inf" }.to_string();
    }
    format!("{value:.precision$}")
}

/// Decode one UTF-8 code point from `s`. On success returns `(cp, consumed)`;
/// on invalid UTF-8 (overlong, surrogate, out-of-range, truncated) returns
/// `None`. Mirrors the C `utf8_decode`.
fn utf8_decode(s: &[u8]) -> Option<(u32, usize)> {
    let b0 = *s.first()?;
    if b0 < 0x80 {
        return Some((b0 as u32, 1));
    }
    if b0 < 0xC2 {
        return None;
    }
    if b0 < 0xE0 {
        let b1 = *s.get(1)?;
        if b1 & 0xC0 != 0x80 {
            return None;
        }
        return Some((((b0 as u32 & 0x1F) << 6) | (b1 as u32 & 0x3F), 2));
    }
    if b0 < 0xF0 {
        let b1 = *s.get(1)?;
        let b2 = *s.get(2)?;
        if b1 & 0xC0 != 0x80 || b2 & 0xC0 != 0x80 {
            return None;
        }
        if b0 == 0xE0 && b1 < 0xA0 {
            return None;
        }
        if b0 == 0xED && b1 >= 0xA0 {
            return None;
        }
        let cp = ((b0 as u32 & 0x0F) << 12) | ((b1 as u32 & 0x3F) << 6) | (b2 as u32 & 0x3F);
        return Some((cp, 3));
    }
    if b0 < 0xF5 {
        let b1 = *s.get(1)?;
        let b2 = *s.get(2)?;
        let b3 = *s.get(3)?;
        if b1 & 0xC0 != 0x80 || b2 & 0xC0 != 0x80 || b3 & 0xC0 != 0x80 {
            return None;
        }
        if b0 == 0xF0 && b1 < 0x90 {
            return None;
        }
        if b0 == 0xF4 && b1 >= 0x90 {
            return None;
        }
        let cp = ((b0 as u32 & 0x07) << 18)
            | ((b1 as u32 & 0x3F) << 12)
            | ((b2 as u32 & 0x3F) << 6)
            | (b3 as u32 & 0x3F);
        return Some((cp, 4));
    }
    None
}

/// Encode `cp` as UTF-8, returning the bytes. Mirrors the C `utf8_encode`.
fn utf8_encode(cp: u32) -> Vec<u8> {
    if cp < 0x80 {
        return vec![cp as u8];
    }
    if cp < 0x800 {
        return vec![0xC0 | (cp >> 6) as u8, 0x80 | (cp & 0x3F) as u8];
    }
    if cp < 0x10000 {
        return vec![
            0xE0 | (cp >> 12) as u8,
            0x80 | ((cp >> 6) & 0x3F) as u8,
            0x80 | (cp & 0x3F) as u8,
        ];
    }
    vec![
        0xF0 | (cp >> 18) as u8,
        0x80 | ((cp >> 12) & 0x3F) as u8,
        0x80 | ((cp >> 6) & 0x3F) as u8,
        0x80 | (cp & 0x3F) as u8,
    ]
}

/// A Thompson NFA regular-expression engine (LIP-021 §2), mirroring the C11
/// engine in `vm/c/vm.c`. Byte-oriented and linear-time (no backtracking).
struct RegexCompiler<'a> {
    pattern: &'a [u8],
    pos: usize,
    insts: Vec<RegexInst>,
    classes: Vec<RegexClass>,
    error: Option<&'static str>,
}

impl<'a> RegexCompiler<'a> {
    fn fail(&mut self, message: &'static str) {
        if self.error.is_none() {
            self.error = Some(message);
        }
    }

    fn emit(&mut self, op: RegexOp, c: u32, x: u32, y: u32) -> u32 {
        let index = self.insts.len() as u32;
        self.insts.push(RegexInst { op, c, x, y });
        index
    }

    fn add_class(&mut self, bitmap: [u32; 8], negated: bool) -> u32 {
        let index = self.classes.len() as u32;
        self.classes.push(RegexClass { bitmap, negated });
        index
    }

    fn compile_class(&mut self) -> u32 {
        let mut bitmap = [0u32; 8];
        let mut negated = false;
        let mut first = true;
        self.pos += 1; // skip '['
        if self.pos < self.pattern.len() && self.pattern[self.pos] == b'^' {
            negated = true;
            self.pos += 1;
        }
        while self.pos < self.pattern.len() {
            let ch = self.pattern[self.pos];
            if ch == b']' && !first {
                self.pos += 1;
                break;
            }
            first = false;
            let lo = ch as u32;
            self.pos += 1;
            if self.pos + 1 < self.pattern.len()
                && self.pattern[self.pos] == b'-'
                && self.pattern[self.pos + 1] != b']'
            {
                self.pos += 1; // skip '-'
                let hi = self.pattern[self.pos] as u32;
                self.pos += 1;
                if hi < lo {
                    self.fail("invalid range in character class");
                    return 0;
                }
                for b in lo..=hi {
                    bitmap[(b >> 5) as usize] |= 1 << (b & 31);
                }
            } else {
                bitmap[(lo >> 5) as usize] |= 1 << (lo & 31);
            }
        }
        if self.pos >= self.pattern.len() {
            self.fail("unterminated character class");
            return 0;
        }
        let class_index = self.add_class(bitmap, negated);
        self.emit(RegexOp::Class, class_index, 0, 0)
    }

    fn compile_atom(&mut self) -> u32 {
        if self.pos >= self.pattern.len() {
            self.fail("unexpected end of pattern");
            return 0;
        }
        let ch = self.pattern[self.pos];
        match ch {
            b'.' => {
                self.pos += 1;
                self.emit(RegexOp::Any, 0, 0, 0)
            }
            b'^' => {
                self.pos += 1;
                self.emit(RegexOp::Bol, 0, 0, 0)
            }
            b'$' => {
                self.pos += 1;
                self.emit(RegexOp::Eol, 0, 0, 0)
            }
            b'[' => self.compile_class(),
            b'(' => {
                self.pos += 1;
                let start = self.compile_alternation();
                if self.error.is_some() {
                    return start;
                }
                if self.pos >= self.pattern.len() || self.pattern[self.pos] != b')' {
                    self.fail("unterminated group");
                    return start;
                }
                self.pos += 1; // skip ')'
                start
            }
            b')' => {
                self.fail("unmatched ')'");
                0
            }
            b'*' | b'+' | b'?' | b'|' => {
                self.fail("dangling metacharacter");
                0
            }
            b'\\' => {
                self.pos += 1;
                if self.pos >= self.pattern.len() {
                    self.fail("trailing backslash");
                    return 0;
                }
                let esc = self.pattern[self.pos];
                self.pos += 1;
                self.emit(RegexOp::Char, esc as u32, 0, 0)
            }
            _ => {
                self.pos += 1;
                self.emit(RegexOp::Char, ch as u32, 0, 0)
            }
        }
    }

    fn compile_repeat(&mut self) -> u32 {
        let atom_start = self.compile_atom();
        if self.error.is_some() || self.pos >= self.pattern.len() {
            return atom_start;
        }
        match self.pattern[self.pos] {
            b'*' => {
                self.pos += 1;
                let split = self.emit(RegexOp::Split, 0, 0, 0);
                self.emit(RegexOp::Jmp, 0, 0, 0);
                let split_inst = self.insts.remove(split as usize);
                self.insts.insert(atom_start as usize, split_inst);
                self.insts[atom_start as usize].x = atom_start + 1;
                self.insts[atom_start as usize].y = split + 2;
                self.insts[(split + 1) as usize].x = atom_start;
                atom_start
            }
            b'+' => {
                self.pos += 1;
                let split = self.emit(RegexOp::Split, 0, 0, 0);
                self.insts[split as usize].x = atom_start;
                self.insts[split as usize].y = split + 1;
                atom_start
            }
            b'?' => {
                self.pos += 1;
                let split = self.emit(RegexOp::Split, 0, 0, 0);
                let split_inst = self.insts.remove(split as usize);
                self.insts.insert(atom_start as usize, split_inst);
                self.insts[atom_start as usize].x = atom_start + 1;
                self.insts[atom_start as usize].y = split + 1;
                atom_start
            }
            _ => atom_start,
        }
    }

    fn compile_concat(&mut self) -> u32 {
        let start = self.insts.len() as u32;
        while self.error.is_none() && self.pos < self.pattern.len() {
            let ch = self.pattern[self.pos];
            if ch == b'|' || ch == b')' {
                break;
            }
            self.compile_repeat();
        }
        start
    }

    fn compile_alternation(&mut self) -> u32 {
        let start = self.compile_concat();
        while self.error.is_none()
            && self.pos < self.pattern.len()
            && self.pattern[self.pos] == b'|'
        {
            self.pos += 1; // skip '|'
            let split = self.emit(RegexOp::Split, 0, 0, 0);
            self.emit(RegexOp::Jmp, 0, 0, 0);
            let split_inst = self.insts.remove(split as usize);
            self.insts.insert(start as usize, split_inst);
            let right = self.compile_concat();
            if self.error.is_some() {
                return start;
            }
            self.insts[start as usize].x = start + 1;
            self.insts[start as usize].y = right;
            self.insts[(split + 1) as usize].x = self.insts.len() as u32;
        }
        start
    }
}

fn regex_compile(pattern: &[u8]) -> Result<Regex, &'static str> {
    let mut c = RegexCompiler {
        pattern,
        pos: 0,
        insts: Vec::new(),
        classes: Vec::new(),
        error: None,
    };
    c.compile_alternation();
    if let Some(err) = c.error {
        return Err(err);
    }
    if c.pos < c.pattern.len() {
        return Err("unmatched ')'");
    }
    c.emit(RegexOp::Match, 0, 0, 0);
    Ok(Regex { insts: c.insts, classes: c.classes })
}

fn regex_addstate(re: &Regex, list: &mut Vec<u32>, seen: &mut [bool], pc: u32, pos: usize, len: usize) {
    if seen[pc as usize] {
        return;
    }
    seen[pc as usize] = true;
    let inst = &re.insts[pc as usize];
    match inst.op {
        RegexOp::Split => {
            regex_addstate(re, list, seen, inst.x, pos, len);
            regex_addstate(re, list, seen, inst.y, pos, len);
        }
        RegexOp::Jmp => regex_addstate(re, list, seen, inst.x, pos, len),
        RegexOp::Bol => {
            if pos == 0 {
                regex_addstate(re, list, seen, pc + 1, pos, len);
            }
        }
        RegexOp::Eol => {
            if pos == len {
                regex_addstate(re, list, seen, pc + 1, pos, len);
            }
        }
        _ => list.push(pc),
    }
}

fn regex_step(
    re: &Regex,
    clist: &[u32],
    nlist: &mut Vec<u32>,
    seen: &mut [bool],
    c: u8,
    pos: usize,
    len: usize,
) {
    for &pc in clist {
        let inst = &re.insts[pc as usize];
        match inst.op {
            RegexOp::Char => {
                if inst.c == c as u32 {
                    regex_addstate(re, nlist, seen, pc + 1, pos + 1, len);
                }
            }
            RegexOp::Any => {
                if c != b'\n' {
                    regex_addstate(re, nlist, seen, pc + 1, pos + 1, len);
                }
            }
            RegexOp::Class => {
                let cls = &re.classes[inst.c as usize];
                let mut in_class = (cls.bitmap[(c as usize) >> 5] & (1 << ((c as usize) & 31))) != 0;
                if cls.negated {
                    in_class = !in_class;
                }
                if in_class {
                    regex_addstate(re, nlist, seen, pc + 1, pos + 1, len);
                }
            }
            _ => {}
        }
    }
}

fn regex_has_match(re: &Regex, list: &[u32]) -> bool {
    list.iter().any(|&pc| re.insts[pc as usize].op == RegexOp::Match)
}

fn regex_match_from(re: &Regex, text: &[u8], start: usize) -> Option<usize> {
    let n = re.insts.len();
    let mut cur: Vec<u32> = Vec::with_capacity(n);
    let mut next: Vec<u32> = Vec::with_capacity(n);
    let mut seen: Vec<bool> = vec![false; n];
    let len = text.len();
    let mut matched = false;
    let mut end = start;
    regex_addstate(re, &mut cur, &mut seen, 0, start, len);
    if regex_has_match(re, &cur) {
        matched = true;
    }
    for pos in start..len {
        seen.fill(false);
        next.clear();
        regex_step(re, &cur, &mut next, &mut seen, text[pos], pos, len);
        std::mem::swap(&mut cur, &mut next);
        if regex_has_match(re, &cur) {
            matched = true;
            end = pos + 1;
        }
    }
    if matched {
        Some(end)
    } else {
        None
    }
}

fn regex_search(re: &Regex, text: &[u8], from: usize) -> Option<(usize, usize)> {
    for s in from..=text.len() {
        if let Some(end) = regex_match_from(re, text, s) {
            return Some((s, end));
        }
    }
    None
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

/// An integer literal whose magnitude exceeds 2^53 is not exactly representable
/// in binary64; LIP-023 §3 preserves it as a string. Returns false for any
/// token with a fraction or exponent.
fn json_large_integer(token: &[u8]) -> bool {
    if token.iter().any(|&b| b == b'.' || b == b'e' || b == b'E') {
        return false;
    }
    let mut p = 0;
    if p < token.len() && token[p] == b'-' {
        p += 1;
    }
    while p < token.len() && token[p] == b'0' {
        p += 1;
    }
    let first = p;
    let digits = token.len() - p;
    if digits < 16 {
        return false;
    }
    if digits > 16 {
        return true;
    }
    const LIMIT: &[u8] = b"9007199254740992";
    for i in 0..16 {
        if token[first + i] > LIMIT[i] {
            return true;
        }
        if token[first + i] < LIMIT[i] {
            return false;
        }
    }
    false
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
    let mut pending = vec![value.clone()];
    let mut seen = std::collections::HashSet::new();
    while let Some(value) = pending.pop() {
        if let Some(reactive) = &value.reactive { reactive_list_add(list, reactive); }
        let Some(identity) = value.container_identity() else { continue; };
        if !seen.insert(identity) { continue; }
        let mut index = 0;
        while let Some(child) = value.inspection_child(index) { pending.push(child); index += 1; }
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
        && !capability.revoked.load(Ordering::Acquire)
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

/// The permission bit for a `grant` permission name, mirroring
/// `grant_permission` in `vm/c/vm.c`.
fn grant_permission(value: &Value) -> u32 {
    match &value.kind {
        ValueKind::String(string) => match &**string {
            "use" => LANA_CAPABILITY_READ,
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
        ValueKind::ObjectValue(_) | ValueKind::ClassObject(_) => {
            let mut equal = false;
            values_equal(left, right, &mut equal) == LanaError::Ok && equal
        }
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

/// Set membership equality, mirroring `set_value_equal` in `vm/c/vm.c`: scalars
/// compare by value, containers by pointer identity. Byte-identical with the
/// C11 VM.
fn set_value_equal(left: &Value, right: &Value) -> bool {
    if std::mem::discriminant(&left.kind) != std::mem::discriminant(&right.kind) {
        return false;
    }
    match &left.kind {
        ValueKind::Null => true,
        ValueKind::Number(l) => *l == right.as_number(),
        ValueKind::Bool(l) => *l == right.as_bool(),
        ValueKind::String(l) => *l == right.as_string(),
        ValueKind::Sample(l) => *l == match &right.kind {
            ValueKind::Sample(r) => *r,
            _ => unreachable!("same discriminant"),
        },
        ValueKind::State(l) => {
            l.state.p == right.as_state().state.p
                && l.state.d_re == right.as_state().state.d_re
                && l.state.d_im == right.as_state().state.d_im
        }
        ValueKind::Distribution { p0, p1 } => match &right.kind {
            ValueKind::Distribution { p0: r0, p1: r1 } => *p0 == *r0 && *p1 == *r1,
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Function(l) => *l == match &right.kind {
            ValueKind::Function(r) => *r,
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Lazy { function, bound } => match &right.kind {
            ValueKind::Lazy { function: rf, bound: rb } => *function == *rf && *bound == *rb,
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Array(l) => match &right.kind {
            ValueKind::Array(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Map(l) => match &right.kind {
            ValueKind::Map(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Task(l) => match &right.kind {
            ValueKind::Task(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Capability(l) => match &right.kind {
            ValueKind::Capability(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Adt(l) => match &right.kind {
            ValueKind::Adt(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Tensor(l) => match &right.kind {
            ValueKind::Tensor(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::NQubitState(l) => match &right.kind {
            ValueKind::NQubitState(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Povm(l) => match &right.kind {
            ValueKind::Povm(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Channel(l) => match &right.kind {
            ValueKind::Channel(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Observable(l) => match &right.kind {
            ValueKind::Observable(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Generator(l) => match &right.kind {
            ValueKind::Generator(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Future(l) => match &right.kind {
            ValueKind::Future(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Set(l) => match &right.kind {
            ValueKind::Set(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        ValueKind::Regex(l) => match &right.kind {
            ValueKind::Regex(r) => Arc::ptr_eq(l, r),
            _ => unreachable!("same discriminant"),
        },
        _ => false,
    }
}

/// Whether a value can appear as a set member, mirroring `value_is_set_member`
/// in `vm/c/vm.c`: STATE, STATE_DIST, and Information are rejected.
fn value_is_set_member(value: &Value) -> bool {
    if value.reactive.is_some() {
        return false;
    }
    !matches!(
        value.kind,
        ValueKind::State(_)
            | ValueKind::StateDist(_)
            | ValueKind::Joint(_)
            | ValueKind::Possibility(_)
            | ValueKind::PathSet(_)
    )
}

/// Whether a value can appear as a joint marginal or possibility element,
/// mirroring `joint_value_is_definite` in `vm/c/vm.c`.
fn joint_value_is_definite(value: &Value) -> bool {
    !matches!(
        value.kind,
        ValueKind::StateDist(_) | ValueKind::Joint(_) | ValueKind::Kernel(_) | ValueKind::Network(_)
            | ValueKind::Task(_) | ValueKind::Function(_)
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
                4 => *out = Value::number(l.min(r)),
                5 => *out = Value::number(l.max(r)),
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

// ---------------------------------------------------------------------------
// LIP-005 linear algebra on STATEs. The four first-class value types
// (NQubitState, Povm, Channel, Observable) are thin wrappers over a complex
// tensor: a density operator is a d×d matrix, a POVM or channel is a
// [k, d, d] stack of operators, and an observable is a d×d Hermitian matrix.
// The arithmetic below is scalar double loops, mirroring `vm/c/vm.c`.
// ---------------------------------------------------------------------------

/// Read the (i, j) entry of a 2-D tensor as a complex number. A real tensor
/// has zero imaginary part.
fn linalg_get2(t: &Tensor, i: usize, j: usize) -> (f64, f64) {
    let lin = t.offset + i * t.strides[0] + j * t.strides[1];
    if t.is_complex {
        (tensor_get_real(t, lin), tensor_get_imag(t, lin))
    } else {
        (tensor_get_real(t, lin), 0.0)
    }
}

/// Read the (k, i, j) entry of a 3-D tensor as a complex number.
fn linalg_get3(t: &Tensor, k: usize, i: usize, j: usize) -> (f64, f64) {
    let lin = t.offset + k * t.strides[0] + i * t.strides[1] + j * t.strides[2];
    if t.is_complex {
        (tensor_get_real(t, lin), tensor_get_imag(t, lin))
    } else {
        (tensor_get_real(t, lin), 0.0)
    }
}

/// Copy a 2-D tensor (real or complex) into a fresh complex tensor.
fn linalg_copy_complex2(
    alloc: &Heap,
    t: &Tensor,
) -> Result<Tensor, LanaError> {
    let shape = [t.shape[0], t.shape[1]];
    let mut r = tensor::tensor_new(alloc, 2, &shape, true)?;
    for i in 0..t.shape[0] {
        for j in 0..t.shape[1] {
            let (re, im) = linalg_get2(t, i, j);
            let lin = i * t.shape[1] + j;
            tensor_set_real(&mut r, lin, re);
            tensor_set_imag(&mut r, lin, im);
        }
    }
    Ok(r)
}

/// The base-2 logarithm of a power-of-two dimension (the qubit count).
fn linalg_qubits(d: usize) -> usize {
    let mut n = 0;
    let mut d = d;
    while d > 1 {
        d >>= 1;
        n += 1;
    }
    n
}

/// Jacobi eigenvalue algorithm for a Hermitian d×d matrix stored as an
/// interleaved [re, im] row-major array. On return the diagonal holds the real
/// eigenvalues (the off-diagonal has been driven to ~0).
fn linalg_jacobi(a: &mut [f64], d: usize, eigenvalues: &mut [f64]) {
    const TOL: f64 = 1e-12;
    const MAX_SWEEPS: usize = 50;
    for _sweep in 0..MAX_SWEEPS {
        let mut off = 0.0;
        for p in 0..d {
            for q in p + 1..d {
                let re = a[(p * d + q) * 2];
                let im = a[(p * d + q) * 2 + 1];
                off += re * re + im * im;
            }
        }
        if off <= TOL * TOL {
            break;
        }
        for p in 0..d {
            for q in p + 1..d {
                let apq_re = a[(p * d + q) * 2];
                let apq_im = a[(p * d + q) * 2 + 1];
                let m = apq_re.hypot(apq_im);
                if m <= TOL {
                    continue;
                }
                let app = a[(p * d + p) * 2];
                let aqq = a[(q * d + q) * 2];
                let theta = 0.5 * (2.0 * m).atan2(app - aqq);
                let c = theta.cos();
                let s = theta.sin();
                let cos_phi = apq_re / m;
                let sin_phi = apq_im / m;
                a[(p * d + p) * 2] = c * c * app + 2.0 * c * s * m + s * s * aqq;
                a[(p * d + p) * 2 + 1] = 0.0;
                a[(q * d + q) * 2] = s * s * app - 2.0 * c * s * m + c * c * aqq;
                a[(q * d + q) * 2 + 1] = 0.0;
                a[(p * d + q) * 2] = 0.0;
                a[(p * d + q) * 2 + 1] = 0.0;
                a[(q * d + p) * 2] = 0.0;
                a[(q * d + p) * 2 + 1] = 0.0;
                for k in 0..d {
                    if k == p || k == q {
                        continue;
                    }
                    let akp_re = a[(k * d + p) * 2];
                    let akp_im = a[(k * d + p) * 2 + 1];
                    let akq_re = a[(k * d + q) * 2];
                    let akq_im = a[(k * d + q) * 2 + 1];
                    // new_akp = c*akp + s*e^{-iφ}*akq
                    let t_re = cos_phi * akq_re + sin_phi * akq_im;
                    let t_im = cos_phi * akq_im - sin_phi * akq_re;
                    let new_akp_re = c * akp_re + s * t_re;
                    let new_akp_im = c * akp_im + s * t_im;
                    // new_akq = -s*e^{iφ}*akp + c*akq
                    let u_re = cos_phi * akp_re - sin_phi * akp_im;
                    let u_im = cos_phi * akp_im + sin_phi * akp_re;
                    let new_akq_re = -s * u_re + c * akq_re;
                    let new_akq_im = -s * u_im + c * akq_im;
                    a[(k * d + p) * 2] = new_akp_re;
                    a[(k * d + p) * 2 + 1] = new_akp_im;
                    a[(p * d + k) * 2] = new_akp_re;
                    a[(p * d + k) * 2 + 1] = -new_akp_im;
                    a[(k * d + q) * 2] = new_akq_re;
                    a[(k * d + q) * 2 + 1] = new_akq_im;
                    a[(q * d + k) * 2] = new_akq_re;
                    a[(q * d + k) * 2 + 1] = -new_akq_im;
                }
            }
        }
    }
    for p in 0..d {
        eigenvalues[p] = a[(p * d + p) * 2];
    }
}

/// Whether a 2-D tensor is Hermitian (A[i][j] == conj(A[j][i])) within tol.
fn linalg_is_hermitian(t: &Tensor, tol: f64) -> bool {
    let d = t.shape[0];
    for i in 0..d {
        for j in i..d {
            let (a_re, a_im) = linalg_get2(t, i, j);
            let (b_re, b_im) = linalg_get2(t, j, i);
            if (a_re - b_re).abs() > tol || (a_im + b_im).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Trace of a 2-D tensor (the real part; the imaginary part is ~0 for a
/// Hermitian matrix).
fn linalg_trace(t: &Tensor) -> f64 {
    let mut sum = 0.0;
    for i in 0..t.shape[0] {
        let (re, _im) = linalg_get2(t, i, i);
        sum += re;
    }
    sum
}

/// Eigenvalues of a 2-D tensor (assumed Hermitian), in accounted scratch.
fn linalg_eigenvalues(
    alloc: &Heap,
    t: &Tensor,
) -> Result<crate::heap::Buffer<f64>, LanaError> {
    let d = t.shape[0];
    let mut a = crate::heap::Buffer::filled(alloc, d * d * 2, 0.0)?;
    for i in 0..d {
        for j in 0..d {
            let (re, im) = linalg_get2(t, i, j);
            a[(i * d + j) * 2] = re;
            a[(i * d + j) * 2 + 1] = im;
        }
    }
    let mut eig = crate::heap::Buffer::filled(alloc, d, 0.0)?;
    linalg_jacobi(&mut a, d, &mut eig);
    Ok(eig)
}

/// Whether a 2-D tensor is positive semidefinite (all eigenvalues >= -1e-9).
fn linalg_is_psd(
    alloc: &Heap,
    t: &Tensor,
) -> Result<bool, LanaError> {
    let eig = linalg_eigenvalues(alloc, t)?;
    Ok(eig.iter().all(|&e| e >= -1e-9))
}

/// density_operator from an N=1 STATE: the 2×2 matrix [[p, c], [c*, 1-p]].
fn linalg_density_from_state(
    alloc: &Heap,
    state: &State,
) -> Result<Value, LanaError> {
    let shape = [2usize, 2usize];
    let mut r = tensor::tensor_new(alloc, 2, &shape, true)?;
    let mut c_re = 0.0;
    let mut c_im = 0.0;
    state::reconstruct_c(state, &mut c_re, &mut c_im);
    tensor_set_real(&mut r, 0, state.p);
    tensor_set_imag(&mut r, 0, 0.0);
    tensor_set_real(&mut r, 1, c_re);
    tensor_set_imag(&mut r, 1, c_im);
    tensor_set_real(&mut r, 2, c_re);
    tensor_set_imag(&mut r, 2, if c_im == 0.0 { 0.0 } else { -c_im }); // normalize -0.0
    tensor_set_real(&mut r, 3, 1.0 - state.p);
    tensor_set_imag(&mut r, 3, 0.0);
    Ok(Value::nqubit_state(Arc::new(r)))
}

/// density_operator from a tensor: validate Hermitian, PSD, unit trace, and
/// that d = 2^N with N <= 10.
fn linalg_density_from_tensor(
    alloc: &Heap,
    t: &Tensor,
) -> Result<Value, LanaError> {
    if t.ndim != 2 || t.shape[0] != t.shape[1] {
        return Err(LanaError::InvalidState);
    }
    let d = t.shape[0];
    if d == 0 || (d & (d - 1)) != 0 {
        return Err(LanaError::InvalidState);
    }
    let n = linalg_qubits(d);
    if n > 10 {
        return Err(LanaError::InvalidParameters);
    }
    if !linalg_is_hermitian(t, 1e-9) {
        return Err(LanaError::InvalidState);
    }
    if !linalg_is_psd(alloc, t)? {
        return Err(LanaError::InvalidState);
    }
    if (linalg_trace(t) - 1.0).abs() > 1e-9 {
        return Err(LanaError::InvalidState);
    }
    let r = linalg_copy_complex2(alloc, t)?;
    Ok(Value::nqubit_state(Arc::new(r)))
}

/// povm([E...]): stack the operators and validate each PSD and Σ E_i = I.
fn linalg_povm(
    alloc: &Heap,
    arg: &Value,
) -> Result<Value, LanaError> {
    let ValueKind::Array(arr) = &arg.kind else {
        return Err(LanaError::Type);
    };
    let arr = arr.lock().unwrap();
    let k = arr.items.len();
    if k == 0 {
        return Err(LanaError::InvalidParameters);
    }
    let ValueKind::Tensor(first) = &arr.items[0].kind else {
        return Err(LanaError::Type);
    };
    if first.ndim != 2 || first.shape[0] != first.shape[1] {
        return Err(LanaError::InvalidParameters);
    }
    let d = first.shape[0];
    let shape = [k, d, d];
    let mut stack = tensor::tensor_new(alloc, 3, &shape, true)?;
    for (i, item) in arr.items.iter().enumerate() {
        let ValueKind::Tensor(e) = &item.kind else {
            return Err(LanaError::Type);
        };
        if e.ndim != 2 || e.shape[0] != d || e.shape[1] != d {
            return Err(LanaError::InvalidParameters);
        }
        if !linalg_is_psd(alloc, e)? {
            return Err(LanaError::InvalidParameters);
        }
        for r in 0..d {
            for c in 0..d {
                let (re, im) = linalg_get2(e, r, c);
                let lin = i * d * d + r * d + c;
                tensor_set_real(&mut stack, lin, re);
                tensor_set_imag(&mut stack, lin, im);
            }
        }
    }
    for r in 0..d {
        for c in 0..d {
            let mut re = 0.0;
            let mut im = 0.0;
            for i in 0..k {
                let lin = i * d * d + r * d + c;
                re += tensor_get_real(&stack, lin);
                im += tensor_get_imag(&stack, lin);
            }
            let want_re = if r == c { 1.0 } else { 0.0 };
            if (re - want_re).abs() > 1e-9 || im.abs() > 1e-9 {
                return Err(LanaError::InvalidParameters);
            }
        }
    }
    Ok(Value::povm(Arc::new(stack)))
}

/// channel([K...]): stack the Kraus operators and validate Σ K_k† K_k = I.
fn linalg_channel(
    alloc: &Heap,
    arg: &Value,
) -> Result<Value, LanaError> {
    let ValueKind::Array(arr) = &arg.kind else {
        return Err(LanaError::Type);
    };
    let arr = arr.lock().unwrap();
    let k = arr.items.len();
    if k == 0 {
        return Err(LanaError::InvalidParameters);
    }
    let ValueKind::Tensor(first) = &arr.items[0].kind else {
        return Err(LanaError::Type);
    };
    if first.ndim != 2 || first.shape[0] != first.shape[1] {
        return Err(LanaError::InvalidParameters);
    }
    let d = first.shape[0];
    let shape = [k, d, d];
    let mut stack = tensor::tensor_new(alloc, 3, &shape, true)?;
    for (i, item) in arr.items.iter().enumerate() {
        let ValueKind::Tensor(e) = &item.kind else {
            return Err(LanaError::Type);
        };
        if e.ndim != 2 || e.shape[0] != d || e.shape[1] != d {
            return Err(LanaError::InvalidParameters);
        }
        for r in 0..d {
            for c in 0..d {
                let (re, im) = linalg_get2(e, r, c);
                let lin = i * d * d + r * d + c;
                tensor_set_real(&mut stack, lin, re);
                tensor_set_imag(&mut stack, lin, im);
            }
        }
    }
    for r in 0..d {
        for c in 0..d {
            let mut re = 0.0;
            let mut im = 0.0;
            for i in 0..k {
                for m in 0..d {
                    let (k_mr_re, k_mr_im) = linalg_get3(&stack, i, m, r);
                    let (k_mc_re, k_mc_im) = linalg_get3(&stack, i, m, c);
                    // conj(K[m][r]) * K[m][c]
                    re += k_mr_re * k_mc_re + k_mr_im * k_mc_im;
                    im += k_mr_re * k_mc_im - k_mr_im * k_mc_re;
                }
            }
            let want_re = if r == c { 1.0 } else { 0.0 };
            if (re - want_re).abs() > 1e-9 || im.abs() > 1e-9 {
                return Err(LanaError::InvalidParameters);
            }
        }
    }
    Ok(Value::channel(Arc::new(stack)))
}

/// observable(A): validate Hermitian.
fn linalg_observable(
    alloc: &Heap,
    arg: &Value,
) -> Result<Value, LanaError> {
    let ValueKind::Tensor(t) = &arg.kind else {
        return Err(LanaError::Type);
    };
    if t.ndim != 2 || t.shape[0] != t.shape[1] {
        return Err(LanaError::InvalidParameters);
    }
    if !linalg_is_hermitian(t, 1e-9) {
        return Err(LanaError::InvalidParameters);
    }
    let r = linalg_copy_complex2(alloc, t)?;
    Ok(Value::observable(Arc::new(r)))
}

/// tensor_product(a, b): the Kronecker product ρ_A ⊗ ρ_B.
fn linalg_tensor_product(
    alloc: &Heap,
    a: &Tensor,
    b: &Tensor,
) -> Result<Value, LanaError> {
    let da = a.shape[0];
    let db = b.shape[0];
    let shape = [da * db, da * db];
    let mut r = tensor::tensor_new(alloc, 2, &shape, true)?;
    for i1 in 0..da {
        for i2 in 0..da {
            for j1 in 0..db {
                for j2 in 0..db {
                    let (a_re, a_im) = linalg_get2(a, i1, i2);
                    let (b_re, b_im) = linalg_get2(b, j1, j2);
                    let lin = (i1 * db + j1) * (da * db) + (i2 * db + j2);
                    tensor_set_real(&mut r, lin, a_re * b_re - a_im * b_im);
                    tensor_set_imag(&mut r, lin, a_re * b_im + a_im * b_re);
                }
            }
        }
    }
    Ok(Value::nqubit_state(Arc::new(r)))
}

/// partial_trace(ab, subsystem): trace out the second subsystem, keeping the
/// first `subsystem` qubits. `subsystem` is the qubit count of the kept
/// subsystem A (1 <= subsystem < N).
fn linalg_partial_trace(
    alloc: &Heap,
    ab: &Tensor,
    subsystem: f64,
) -> Result<Value, LanaError> {
    let d = ab.shape[0];
    let n = linalg_qubits(d);
    if !subsystem.is_finite() || subsystem < 1.0 || subsystem.floor() != subsystem {
        return Err(LanaError::InvalidParameters);
    }
    let k = subsystem as usize;
    if k >= n {
        return Err(LanaError::InvalidParameters);
    }
    let da = 1usize << k;
    let db = d / da;
    let shape = [da, da];
    let mut r = tensor::tensor_new(alloc, 2, &shape, true)?;
    for i in 0..da {
        for j in 0..da {
            let mut re = 0.0;
            let mut im = 0.0;
            for m in 0..db {
                let (e_re, e_im) = linalg_get2(ab, i * db + m, j * db + m);
                re += e_re;
                im += e_im;
            }
            let lin = i * da + j;
            tensor_set_real(&mut r, lin, re);
            tensor_set_imag(&mut r, lin, im);
        }
    }
    Ok(Value::nqubit_state(Arc::new(r)))
}

/// measure_with(rho, povm): the outcome distribution p(i) = Tr(ρ E_i).
fn linalg_measure_with(
    heap: &Heap,
    rho: &Tensor,
    povm: &Tensor,
) -> Result<Value, LanaError> {
    let k = povm.shape[0];
    let d = povm.shape[1];
    let mut array = Array::new(heap, k)?;
    for i in 0..k {
        let mut re = 0.0;
        for r in 0..d {
            for c in 0..d {
                let (rho_re, rho_im) = linalg_get2(rho, r, c);
                let (e_re, e_im) = linalg_get3(povm, i, c, r);
                re += rho_re * e_re - rho_im * e_im;
            }
        }
        array.push(Value::number(re))?;
    }
    Ok(Value::array(Arc::new(Mutex::new(array))))
}

/// apply_to(chan, rho): Φ(ρ) = Σ_k K_k ρ K_k†.
fn linalg_apply_to(
    alloc: &Heap,
    chan: &Tensor,
    rho: &Tensor,
) -> Result<Value, LanaError> {
    let k = chan.shape[0];
    let d = chan.shape[1];
    let shape = [d, d];
    let mut r = tensor::tensor_new(alloc, 2, &shape, true)?;
    for kk in 0..k {
        for i in 0..d {
            for j in 0..d {
                let mut re = 0.0;
                let mut im = 0.0;
                for m in 0..d {
                    for n in 0..d {
                        let (k_re, k_im) = linalg_get3(chan, kk, i, m);
                        let (rho_re, rho_im) = linalg_get2(rho, m, n);
                        let (kd_re, kd_im) = linalg_get3(chan, kk, j, n);
                        // K[i][m] * ρ[m][n] * conj(K[j][n])
                        let t_re = k_re * rho_re - k_im * rho_im;
                        let t_im = k_re * rho_im + k_im * rho_re;
                        re += t_re * kd_re + t_im * kd_im;
                        im += t_im * kd_re - t_re * kd_im;
                    }
                }
                let lin = i * d + j;
                let cur_re = tensor_get_real(&r, lin);
                let cur_im = tensor_get_imag(&r, lin);
                tensor_set_real(&mut r, lin, cur_re + re);
                tensor_set_imag(&mut r, lin, cur_im + im);
            }
        }
    }
    Ok(Value::nqubit_state(Arc::new(r)))
}

/// expect(rho, obs): ⟨A⟩ = Tr(ρ A).
fn linalg_expect(rho: &Tensor, obs: &Tensor) -> Value {
    let d = rho.shape[0];
    let mut re = 0.0;
    for r in 0..d {
        for c in 0..d {
            let (rho_re, rho_im) = linalg_get2(rho, r, c);
            let (a_re, a_im) = linalg_get2(obs, c, r);
            re += rho_re * a_re - rho_im * a_im;
        }
    }
    Value::number(re)
}

/// mix(a, b, w): the convex mixture w·a + (1-w)·b.
fn linalg_mix(
    alloc: &Heap,
    a: &Tensor,
    b: &Tensor,
    w: f64,
) -> Result<Value, LanaError> {
    let d = a.shape[0];
    let shape = [d, d];
    let mut r = tensor::tensor_new(alloc, 2, &shape, true)?;
    for i in 0..d {
        for j in 0..d {
            let (a_re, a_im) = linalg_get2(a, i, j);
            let (b_re, b_im) = linalg_get2(b, i, j);
            let lin = i * d + j;
            tensor_set_real(&mut r, lin, w * a_re + (1.0 - w) * b_re);
            tensor_set_imag(&mut r, lin, w * a_im + (1.0 - w) * b_im);
        }
    }
    Ok(Value::nqubit_state(Arc::new(r)))
}

/// trace_distance(a, b): ½‖ρ − σ‖₁ = ½ Σ |λ_i(ρ − σ)|.
fn linalg_trace_distance(
    alloc: &Heap,
    a: &Tensor,
    b: &Tensor,
) -> Result<Value, LanaError> {
    let d = a.shape[0];
    let mut diff = crate::heap::Buffer::filled(alloc, d * d * 2, 0.0)?;
    for i in 0..d {
        for j in 0..d {
            let (a_re, a_im) = linalg_get2(a, i, j);
            let (b_re, b_im) = linalg_get2(b, i, j);
            diff[(i * d + j) * 2] = a_re - b_re;
            diff[(i * d + j) * 2 + 1] = a_im - b_im;
        }
    }
    let mut eig = crate::heap::Buffer::filled(alloc, d, 0.0)?;
    linalg_jacobi(&mut diff, d, &mut eig);
    let sum: f64 = eig.iter().map(|&e| e.abs()).sum();
    Ok(Value::number(0.5 * sum))
}

/// is_separable(ab, bipartition): the PPT (Peres-Horodecki) criterion. A
/// negative partial-transpose eigenvalue proves entanglement; a PSD partial
/// transpose proves separability for 2×2 and 2×3 systems and is otherwise
/// inconclusive. `bipartition` is the qubit count of the first subsystem.
fn linalg_is_separable(
    alloc: &Heap,
    ab: &Tensor,
    bipartition: f64,
) -> Result<Value, LanaError> {
    let d = ab.shape[0];
    let n = linalg_qubits(d);
    if !bipartition.is_finite() || bipartition < 1.0 || bipartition.floor() != bipartition {
        return Err(LanaError::InvalidParameters);
    }
    let k = bipartition as usize;
    if k >= n {
        return Err(LanaError::InvalidParameters);
    }
    let da = 1usize << k;
    let db = d / da;
    let mut pt = crate::heap::Buffer::filled(alloc, d * d * 2, 0.0)?;
    for i1 in 0..da {
        for i2 in 0..da {
            for j1 in 0..db {
                for j2 in 0..db {
                    let (re, im) = linalg_get2(ab, i1 * db + j2, i2 * db + j1);
                    let lin = (i1 * db + j1) * d + (i2 * db + j2);
                    pt[lin * 2] = re;
                    pt[lin * 2 + 1] = im;
                }
            }
        }
    }
    let mut eig = crate::heap::Buffer::filled(alloc, d, 0.0)?;
    linalg_jacobi(&mut pt, d, &mut eig);
    let negative = eig.iter().any(|&e| e < -1e-9);
    let provably_separable = da == 1
        || db == 1
        || (da == 2 && db == 2)
        || (da == 2 && db == 3)
        || (da == 3 && db == 2);
    if negative {
        Ok(Value::string(Arc::from("entangled")))
    } else if provably_separable {
        Ok(Value::string(Arc::from("separable")))
    } else {
        Ok(Value::string(Arc::from("inconclusive")))
    }
}

/// to_state(rho): recover the N=1 (p, d_re, d_im) form from a 1-qubit density
/// operator.
fn linalg_to_state(rho: &Tensor) -> Result<Value, LanaError> {
    if rho.shape[0] != 2 {
        return Err(LanaError::InvalidParameters);
    }
    let (p, _dummy) = linalg_get2(rho, 0, 0);
    let (c_re, c_im) = linalg_get2(rho, 0, 1);
    let scale = (p * (1.0 - p)).sqrt();
    let mut state = State::default();
    let error = if scale > 0.0 {
        state::make_complex(p, c_re / scale, c_im / scale, &mut state)
    } else {
        state::make_complex(p, 0.0, 0.0, &mut state)
    };
    if error != LanaError::Ok {
        return Err(error);
    }
    Ok(Value::state(StateValue {
        state,
        indexes: Indexes::default(),
    }))
}

/// The underlying tensor of a tensor-like value (tensor, POVM, channel,
/// N-qubit state, or observable), mirroring the C11 `Value.as.tensor` union
/// field shared by those value types.
fn value_tensor(value: &Value) -> Option<&Arc<Tensor>> {
    match &value.kind {
        ValueKind::Tensor(t) => Some(t),
        ValueKind::Povm(t) => Some(t),
        ValueKind::Channel(t) => Some(t),
        ValueKind::NQubitState(t) => Some(t),
        ValueKind::Observable(t) => Some(t),
        _ => None,
    }
}

// ===== LIP-007 differentiable STATE tensors =====

/// Validate a d×d density matrix stored as interleaved `[re, im]` row-major
/// data. Checks Hermitian, PSD, and unit trace (LIP-005 §1.2).
fn linalg_validate_density_data(
    alloc: &Heap,
    data: &[f64],
    d: usize,
) -> Result<(), LanaError> {
    for i in 0..d {
        for j in i..d {
            let a_re = data[(i * d + j) * 2];
            let a_im = data[(i * d + j) * 2 + 1];
            let b_re = data[(j * d + i) * 2];
            let b_im = data[(j * d + i) * 2 + 1];
            if (a_re - b_re).abs() > 1e-9 || (a_im + b_im).abs() > 1e-9 {
                return Err(LanaError::InvalidState);
            }
        }
    }
    let mut a = crate::heap::Buffer::from_slice(alloc, &data[..d * d * 2])?;
    let mut eig = crate::heap::Buffer::filled(alloc, d, 0.0)?;
    linalg_jacobi(&mut a, d, &mut eig);
    for i in 0..d {
        if eig[i] < -1e-9 {
            return Err(LanaError::InvalidState);
        }
    }
    let mut trace = 0.0;
    for i in 0..d {
        trace += data[(i * d + i) * 2];
    }
    if (trace - 1.0).abs() > 1e-9 {
        return Err(LanaError::InvalidState);
    }
    Ok(())
}

/// Recursively fill a complex tensor's interleaved `[re, im]` buffer from a
/// nested array of real numbers (imaginary parts are zero).
fn tensor_fill_state(v: &Value, data: &mut [f64], offset: &mut usize) -> Result<(), LanaError> {
    match &v.kind {
        ValueKind::Number(n) => {
            data[2 * *offset] = *n;
            data[2 * *offset + 1] = 0.0;
            *offset += 1;
            Ok(())
        }
        ValueKind::Array(array) => {
            let array = array.lock().unwrap();
            for item in &array.items {
                tensor_fill_state(item, data, offset)?;
            }
            Ok(())
        }
        _ => Err(LanaError::Type),
    }
}

/// state_tensor(literal): construct a STATE tensor from a nested literal of
/// density matrices (real entries; imaginary parts are zero). The literal shape
/// is `[s_1, ..., s_k, d, d]`; each d×d element is validated as a density
/// operator.
fn linalg_state_tensor(
    alloc: &Heap,
    arg: &Value,
) -> Result<Value, LanaError> {
    if !matches!(arg.kind, ValueKind::Array(_)) {
        return Err(LanaError::Type);
    }
    let shape = tensor::tensor_infer_shape(alloc, arg)?;
    let ndim = shape.len();
    if ndim < 2 {
        return Err(LanaError::InvalidParameters);
    }
    if shape[ndim - 1] != shape[ndim - 2] {
        return Err(LanaError::InvalidParameters);
    }
    let d = shape[ndim - 1];
    if d < 2 || (d & (d - 1)) != 0 {
        return Err(LanaError::InvalidParameters);
    }
    let n = linalg_qubits(d);
    if n > 10 {
        return Err(LanaError::InvalidParameters);
    }
    let mut t = tensor::tensor_new(alloc, ndim, &shape, true)?;
    t.is_state = true;
    {
        let data = state_f64_mut(&mut t);
        let mut offset = 0usize;
        tensor_fill_state(arg, data, &mut offset)?;
    }
    let mut batch = 1;
    for i in 0..ndim.saturating_sub(2) {
        batch *= shape[i];
    }
    for i in 0..batch {
        linalg_validate_density_data(alloc, &state_f64(&t)[i * d * d * 2..], d)?;
    }
    Ok(Value::tensor(Arc::new(t)))
}

/// append(a, b): element-wise distribution-valued APPEND (mean state). N=1
/// (single-qubit) only.
fn linalg_state_append(
    alloc: &Heap,
    a: &Tensor,
    b: &Tensor,
) -> Result<Value, LanaError> {
    if a.ndim != b.ndim {
        return Err(LanaError::InvalidParameters);
    }
    for i in 0..a.ndim {
        if a.shape[i] != b.shape[i] {
            return Err(LanaError::InvalidParameters);
        }
    }
    if a.ndim < 2 {
        return Err(LanaError::InvalidParameters);
    }
    let d = a.shape[a.ndim - 1];
    if d != 2 {
        return Err(LanaError::InvalidParameters);
    }
    let mut res = tensor::tensor_new(alloc, a.ndim, &a.shape, true)?;
    res.is_state = true;
    let mut batch = 1;
    for i in 0..a.ndim.saturating_sub(2) {
        batch *= a.shape[i];
    }
    {
        let dr = state_f64_mut(&mut res);
        for i in 0..batch {
            let da = &state_f64(&a)[i * d * d * 2..];
            let db = &state_f64(&b)[i * d * d * 2..];
            let p_a = da[0];
            let c_a_re = da[2];
            let c_a_im = da[3];
            let p_b = db[0];
            let c_b_re = db[2];
            let c_b_im = db[3];
            let s_a = (p_a * (1.0 - p_a)).sqrt();
            let s_b = (p_b * (1.0 - p_b)).sqrt();
            let d_a_re = if s_a > 0.0 { c_a_re / s_a } else { 0.0 };
            let d_a_im = if s_a > 0.0 { c_a_im / s_a } else { 0.0 };
            let d_b_re = if s_b > 0.0 { c_b_re / s_b } else { 0.0 };
            let d_b_im = if s_b > 0.0 { c_b_im / s_b } else { 0.0 };
            let p_c = p_a + p_b - p_a * p_b;
            let d_c_re = (d_a_re + d_b_re) / 2.0;
            let d_c_im = (d_a_im + d_b_im) / 2.0;
            let s_c = (p_c * (1.0 - p_c)).sqrt();
            let c_c_re = d_c_re * s_c;
            let c_c_im = d_c_im * s_c;
            // ρ_C = [[p_C, c_C], [c_C*, 1-p_C]].
            let off = i * d * d * 2;
            dr[off] = p_c;
            dr[off + 1] = 0.0;
            dr[off + 2] = c_c_re;
            dr[off + 3] = c_c_im;
            dr[off + 4] = c_c_re;
            dr[off + 5] = if c_c_im == 0.0 { 0.0 } else { -c_c_im };
            dr[off + 6] = 1.0 - p_c;
            dr[off + 7] = 0.0;
        }
    }
    Ok(Value::tensor(Arc::new(res)))
}

/// measure(s, povm): element-wise outcome probability q[..., i] = Tr(ρ E_i).
fn linalg_state_measure(
    alloc: &Heap,
    s: &Tensor,
    povm: &Tensor,
) -> Result<Value, LanaError> {
    let d = s.shape[s.ndim - 1];
    let k = povm.shape[0];
    if povm.shape[1] != d || povm.shape[2] != d {
        return Err(LanaError::InvalidParameters);
    }
    let mut batch = 1;
    for i in 0..s.ndim.saturating_sub(2) {
        batch *= s.shape[i];
    }
    let out_ndim = s.ndim - 1;
    let mut out_shape = Vec::with_capacity(out_ndim);
    for i in 0..s.ndim.saturating_sub(2) {
        out_shape.push(s.shape[i]);
    }
    out_shape.push(k);
    let mut res = tensor::tensor_new(alloc, out_ndim, &out_shape, false)?;
    {
        let dr = state_f64_mut(&mut res);
        for i in 0..batch {
            let ds = &state_f64(&s)[i * d * d * 2..];
            for m in 0..k {
                let mut re = 0.0;
                for r in 0..d {
                    for c in 0..d {
                        let rho_re = ds[(r * d + c) * 2];
                        let rho_im = ds[(r * d + c) * 2 + 1];
                        let (e_re, e_im) = linalg_get3(povm, m, c, r);
                        re += rho_re * e_re - rho_im * e_im;
                    }
                }
                dr[i * k + m] = re;
            }
        }
    }
    Ok(Value::tensor(Arc::new(res)))
}

/// transform(s, chan): element-wise channel application Φ(ρ) = Σ_k K_k ρ K_k†.
fn linalg_state_transform(
    alloc: &Heap,
    s: &Tensor,
    chan: &Tensor,
) -> Result<Value, LanaError> {
    let d = s.shape[s.ndim - 1];
    let k = chan.shape[0];
    if chan.shape[1] != d || chan.shape[2] != d {
        return Err(LanaError::InvalidParameters);
    }
    let mut batch = 1;
    for i in 0..s.ndim.saturating_sub(2) {
        batch *= s.shape[i];
    }
    let mut res = tensor::tensor_new(alloc, s.ndim, &s.shape, true)?;
    res.is_state = true;
    {
        let dr = state_f64_mut(&mut res);
        for i in 0..batch {
            let ds = &state_f64(&s)[i * d * d * 2..];
            for r in 0..d {
                for c in 0..d {
                    let mut re = 0.0;
                    let mut im = 0.0;
                    for kk in 0..k {
                        for m in 0..d {
                            for n in 0..d {
                                let (k_re, k_im) = linalg_get3(chan, kk, r, m);
                                let rho_re = ds[(m * d + n) * 2];
                                let rho_im = ds[(m * d + n) * 2 + 1];
                                let (kd_re, kd_im) = linalg_get3(chan, kk, c, n);
                                // K[r][m] * ρ[m][n] * conj(K[c][n])
                                let t_re = k_re * rho_re - k_im * rho_im;
                                let t_im = k_re * rho_im + k_im * rho_re;
                                re += t_re * kd_re + t_im * kd_im;
                                im += t_im * kd_re - t_re * kd_im;
                            }
                        }
                    }
                    dr[(r * d + c) * 2] = re;
                    dr[(r * d + c) * 2 + 1] = im;
                }
            }
        }
    }
    Ok(Value::tensor(Arc::new(res)))
}

/// Value equality, mirroring `values_equal`. The C11 default case compares
/// `Value` struct addresses (same register slot); the Rust VM compares heap
/// identity for `Arc`-backed types and payloads for the value types, which
/// agrees on every realistic input.
fn values_equal(left: &Value, right: &Value, out: &mut bool) -> LanaError {
    if matches!(left.kind, ValueKind::ObjectValue(_)) || matches!(right.kind, ValueKind::ObjectValue(_)) {
        return objects::values_equal(left, right, out);
    }
    if matches!(left.kind, ValueKind::StateDist(_) | ValueKind::Map(_) | ValueKind::Joint(_)
        | ValueKind::Kernel(_) | ValueKind::Network(_)
        | ValueKind::Possibility(_) | ValueKind::PathSet(_) | ValueKind::Capability(_))
        || matches!(right.kind, ValueKind::StateDist(_) | ValueKind::Map(_) | ValueKind::Joint(_)
        | ValueKind::Kernel(_) | ValueKind::Network(_)
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
        ValueKind::ClassObject(left) => {
            let ValueKind::ClassObject(right) = &right.kind else { unreachable!() };
            if left.object.upgrade().is_none_or(|o| !o.initialized.load(Ordering::Acquire))
                || right.object.upgrade().is_none_or(|o| !o.initialized.load(Ordering::Acquire)) {
                return LanaError::UnsupportedOperation;
            }
            *out = std::sync::Weak::ptr_eq(&left.object, &right.object);
        }
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

    #[test]
    fn locked_graph_imports_return_errors_and_preserve_source_values() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let array = vm.array_value(vec![Value::number(7.0)]).unwrap();
        let ValueKind::Array(storage) = &array.kind else { unreachable!() };
        let guard = storage.lock().unwrap();
        assert!(matches!(vm.import_value(&array), Err(LanaError::UnsupportedOperation)));
        assert_eq!(guard.items[0].as_number(), 7.0);
        drop(guard);
        let mut storage = Map::new(&vm.heap, 1).unwrap();
        storage.set(Arc::from("x"), Value::number(7.0), true).unwrap();
        let map = Value::map(Arc::new(Mutex::new(storage)));
        let ValueKind::Map(storage) = &map.kind else { unreachable!() };
        let guard = storage.lock().unwrap();
        assert!(matches!(vm.import_value(&map), Err(LanaError::UnsupportedOperation)));
        assert_eq!(guard.get("x").unwrap().as_number(), 7.0);
        drop(guard);
        let set = Value::set(Arc::new(Mutex::new(Set::new(&vm.heap, 1).unwrap())));
        let ValueKind::Set(storage) = &set.kind else { unreachable!() };
        let guard = storage.lock().unwrap();
        assert!(matches!(vm.import_value(&set), Err(LanaError::UnsupportedOperation)));
        drop(guard);
        let live = vm.reactive_root(&Value::number(7.0), DerivationExactness::Exact).unwrap();
        let node = live.reactive.as_ref().unwrap();
        let guard = node.lock().unwrap();
        assert!(matches!(vm.import_value(&live), Err(LanaError::UnsupportedOperation)));
        assert_eq!(guard.current.as_ref().unwrap().as_number(), 7.0);
        drop(guard);
        assert_eq!(vm.import_value(&live).unwrap().as_number(), 7.0);
    }

    #[test]
    fn derivation_text_admission_obeys_heap_limit_before_publication() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let baseline = vm.allocated_bytes();
        vm.set_memory_limit(baseline + 512).unwrap();
        let label = "x".repeat(8192);
        assert!(vm.record_derivation(DerivationKind::Evidence, "input", &[], &label, 0,
            DerivationExactness::Exact, "", DerivationOutcome::Success, "none").is_none());
        assert!(vm.allocated_bytes() <= vm.memory_limit);
        vm.heap.collect_strings();
        assert_eq!(vm.allocated_bytes(), baseline);
        assert!(matches!(vm.current_frame().registers[0].kind, ValueKind::Null));
    }

    #[test]
    fn join_cleanup_obeys_budget_and_retries_without_partial_result() {
        let chunk = assembler::assemble(".function main 0 4\nHALT\n.function worker 0 4\nLOAD_CONST R0 1000\nHOST_CALL array_new R0 1 R1\nRETURN R1\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let scheduler = Scheduler::new();
        vm.scheduler = Some(scheduler.clone());
        let task = vm.start_task(1, 0, 0).unwrap();
        let queued = scheduler.state.lock().unwrap().queue.pop_front().unwrap();
        run_task(queued);
        let source = {
            let state = task.state.lock().unwrap();
            assert!(state.completed);
            let ValueKind::Array(array) = &state.result.kind else { unreachable!() };
            Arc::downgrade(array)
        };
        vm.set_instruction_limit(vm.instruction_count + 1100);
        assert!(matches!(vm.wait_task(&task, -1.0), Err(LanaError::Limit)));
        assert!(vm.instruction_count <= vm.instruction_limit);
        assert!(source.upgrade().is_some(), "failed cleanup keeps the retired heap owned");
        {
            let state = task.state.lock().unwrap();
            assert!(state.joined);
            let ValueKind::Array(array) = &state.result.kind else { unreachable!() };
            assert_eq!(array.lock().unwrap().items.len(), 1000);
        }
        vm.set_instruction_limit(1_000_000);
        let result = vm.wait_task(&task, -1.0).unwrap();
        assert!(source.upgrade().is_none());
        let ValueKind::Array(array) = &result.kind else { unreachable!() };
        assert_eq!(array.lock().unwrap().items.len(), 1000);
        let root = vm.retain_value(&result).unwrap();
        drop(result);
        drop(task);
        let heap = vm.heap();
        drop(scheduler);
        drop(vm);
        assert_eq!(root.child(999).unwrap().unwrap().as_number(), 0.0);
        drop(root);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn live_distribution_transfer_copies_into_receiving_heap() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut source = Vm::new(&chunk);
        let mut receiver = Vm::new(&chunk);
        let source_heap = source.heap();
        let receiver_heap = receiver.heap();
        let distribution = source.state_dist_dirac(&StateValue::default()).unwrap();
        let live = source.reactive_root(&Value::state_dist(distribution.clone()), DerivationExactness::Exact).unwrap();
        let copied = receiver.import_value(&live).unwrap();
        assert!(copied.reactive.is_none());
        let ValueKind::StateDist(copied_distribution) = &copied.kind else { unreachable!() };
        assert!(!Arc::ptr_eq(&distribution, copied_distribution));
        let root = receiver.retain_value(&copied).unwrap();
        drop(copied);
        drop(live);
        drop(distribution);
        drop(source);
        assert_eq!(source_heap.live_bytes(), 0);
        drop(receiver);
        assert!(root.inspect_state_dist(crate::InspectFormat::Json).is_ok());
        drop(root);
        assert_eq!(receiver_heap.live_bytes(), 0);
    }

    #[test]
    fn foreign_immutable_retention_requires_import_and_keeps_heaps_independent() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut source = Vm::new(&chunk);
        let mut receiver = Vm::new(&chunk);
        let source_heap = source.heap();
        let receiver_heap = receiver.heap();
        let value = Value::adt(source.managed_payload(Adt { variant: 0, fields: vec![Value::number(9.0)] }).unwrap());
        let root = source.retain_value(&value).unwrap();
        assert!(matches!(receiver.retain_value(&value), Err(LanaError::Task)));
        let copy = receiver.import_value(&value).unwrap();
        let copied_root = receiver.retain_value(&copy).unwrap();
        drop(copy);
        drop(value);
        drop(source);
        assert_eq!(root.child(0).unwrap().unwrap().as_number(), 9.0);
        drop(root);
        assert_eq!(source_heap.live_bytes(), 0);
        drop(receiver);
        assert_eq!(copied_root.child(0).unwrap().unwrap().as_number(), 9.0);
        drop(copied_root);
        assert_eq!(receiver_heap.live_bytes(), 0);
    }

    #[test]
    fn retained_deep_derivation_copies_and_releases_after_vm_teardown() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let heap = vm.heap();
        let mut value = Value::number(1.0);
        value.derivation = vm.record_derivation(DerivationKind::Evidence, "input", &[], "", 0,
            DerivationExactness::Exact, "", DerivationOutcome::Success, "none");
        let leaf = Arc::downgrade(value.derivation.as_ref().unwrap());
        for _ in 0..20_000 {
            value.derivation = vm.record_derivation(DerivationKind::Operation, "next", &[&value], "", 0,
                DerivationExactness::Exact, "", DerivationOutcome::Success, "none");
            assert!(value.derivation.is_some());
        }
        let top = Arc::downgrade(value.derivation.as_ref().unwrap());
        let copied = vm.import_value(&value).unwrap();
        assert!(!Arc::ptr_eq(value.derivation.as_ref().unwrap(), copied.derivation.as_ref().unwrap()));
        drop(copied);
        let root = vm.retain_value(&value).unwrap();
        drop(value);
        drop(vm);
        assert!(top.upgrade().is_some());
        assert!(leaf.upgrade().is_some());
        drop(root);
        assert!(top.upgrade().is_none());
        assert!(leaf.upgrade().is_none());
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn retained_deep_distribution_releases_headers_after_vm_teardown() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let heap = vm.heap();
        let mut node = vm.state_dist_dirac(&StateValue::default()).unwrap();
        let leaf = Arc::downgrade(&node);
        for _ in 0..20_000 { node = vm.state_dist_attenuate(node, 0.5).unwrap(); }
        let top = Arc::downgrade(&node);
        let value = Value::state_dist(node);
        let copied = vm.import_value(&value).unwrap();
        let ValueKind::StateDist(copied_node) = &copied.kind else { unreachable!() };
        assert!(!Arc::ptr_eq(copied_node, &top.upgrade().unwrap()));
        drop(copied);
        let root = vm.retain_value(&value).unwrap();
        drop(value);
        drop(vm);
        assert!(top.upgrade().is_some());
        assert!(leaf.upgrade().is_some());
        drop(root);
        assert!(top.upgrade().is_none());
        assert!(leaf.upgrade().is_none());
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn derivation_transfer_copies_shared_dag_and_gradient_storage() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let leaf = vm.record_derivation(DerivationKind::Evidence, "input", &[], "", 0,
            DerivationExactness::Exact, "", DerivationOutcome::Success, "none").unwrap();
        let mut gradient = tensor::tensor_new(&vm.heap, 1, &[1], false).unwrap();
        tensor::tensor_set_real(&mut gradient, 0, 7.0);
        *leaf.ad_grad.lock().unwrap() = Some(gradient);
        let mut value = Value::number(1.0);
        value.derivation = Some(leaf.clone());
        let root = vm.record_derivation(DerivationKind::Operation, "add", &[&value, &value], "", 0,
            DerivationExactness::Exact, "", DerivationOutcome::Success, "none").unwrap();
        value.derivation = Some(root.clone());
        let copied = vm.import_value(&value).unwrap().derivation.unwrap();
        assert!(!Arc::ptr_eq(&root, &copied));
        assert_eq!((root.task_lineage, root.local_sequence), (copied.task_lineage, copied.local_sequence));
        assert!(Arc::ptr_eq(&copied.inputs[0], &copied.inputs[1]));
        assert!(!Arc::ptr_eq(&leaf, &copied.inputs[0]));
        let mut gradient = copied.inputs[0].ad_grad.lock().unwrap();
        tensor::tensor_set_real(gradient.as_mut().unwrap(), 0, 9.0);
        assert_eq!(tensor::tensor_get_real(leaf.ad_grad.lock().unwrap().as_ref().unwrap(), 0), 7.0);
        drop(gradient);
        let captured = vm.deep_clone_value_checked(&value, &mut DeepCloneMemo {
            transfer: true, freeze: true, ..DeepCloneMemo::default()
        }).unwrap().derivation.unwrap();
        assert_eq!(&*captured.operation, "snapshot");
        assert_eq!(captured.inputs.len(), 2);
        assert!(Arc::ptr_eq(&captured.inputs[0], &captured.inputs[1]));
        assert!(!Arc::ptr_eq(&root, &captured.inputs[0]));
        let lock = leaf.ad_grad.lock().unwrap();
        assert_eq!(vm.import_value(&value).unwrap_err(), LanaError::UnsupportedOperation);
        drop(lock);
    }

    #[test]
    fn distribution_transfer_preserves_dag_aliases_and_handles_depth() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let source_heap = crate::heap::Heap::new(1024);
        let source_name = source_heap.string("distribution-source").unwrap();
        let mut state = StateValue::default();
        state.indexes.source = Some(source_name.clone());
        let leaf = Arc::new(StateDist { kind: StateDistKind::Dirac(state) });
        let dag = Arc::new(StateDist { kind: StateDistKind::Append {
            left: DistOperand::Node(leaf.clone()), right: DistOperand::Node(leaf.clone()),
            has_cached_parameters: false, p: 0.0, m_re: 0.0, m_im: 0.0, sigma: 0.0,
        }});
        let ValueKind::StateDist(copy) = vm.import_value(&Value::state_dist(dag)).unwrap().kind else { unreachable!() };
        let StateDistKind::Append { left: DistOperand::Node(a), right: DistOperand::Node(b), .. } = &copy.kind else { unreachable!() };
        assert!(Arc::ptr_eq(a, b));
        assert!(!Arc::ptr_eq(a, &leaf));
        let StateDistKind::Dirac(state) = &a.kind else { unreachable!() };
        assert_eq!(state.indexes.source.as_deref(), Some("distribution-source"));
        assert!(!Arc::ptr_eq(state.indexes.source.as_ref().unwrap(), &source_name));
        let mut deep = leaf;
        for _ in 0..100 {
            deep = Arc::new(StateDist { kind: StateDistKind::Attenuate { child: deep, factor: 0.5 } });
        }
        let copied = vm.import_value(&Value::state_dist(deep)).unwrap();
        assert!(matches!(copied.kind, ValueKind::StateDist(_)));
    }

    #[test]
    fn v6_empty_interface_declarations_execute_and_debug() {
        let descriptor = r#"{"fields":[],"implements":[],"kind":"interface","methods":[],"qualified_name":"file/main.lana/Sensor","schema_version":1}"#;
        let hex: String = descriptor.bytes().map(|b| format!("{b:02x}")).collect();
        let chunk = assembler::assemble(&format!(".version 6\n.function main 0 1\nLOAD_CONST R0 42\nPRINT R0\nLOAD_STRING R0 {hex}\nHALT\n")).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run(), LanaError::Ok);
        assert!(matches!(vm.current_frame().registers[0].kind, ValueKind::String(_)));
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.debug_step(), LanaError::Ok);
        assert_eq!(vm.current_frame().registers[0].as_number(), 42.0);
    }

    #[test]
    fn immutable_snapshot_preserves_joint_paths_and_rejects_nested_mutation() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let joint = Value::joint(vm.joint_build_finite("x,y", &[
            Value::boolean(false), Value::boolean(false), Value::boolean(true), Value::boolean(true)
        ], &[0.25, 0.75], 2, 2).unwrap());
        let copied = vm.information_snapshot(&joint).unwrap();
        let (ValueKind::Joint(original), ValueKind::Joint(captured)) = (&joint.kind, &copied.kind) else { panic!("joint"); };
        assert_eq!(original.names, captured.names);
        assert_eq!(captured.rows[1].weight, 0.75);
        assert!(!Arc::ptr_eq(original, captured));
        let aliases = vm.array_value(vec![joint.clone(), joint.clone()]).unwrap();
        let aliases = vm.information_snapshot(&aliases).unwrap();
        let ValueKind::Array(aliases) = aliases.kind else { panic!("array"); };
        let aliases = aliases.lock().unwrap();
        let (ValueKind::Joint(a), ValueKind::Joint(b)) = (&aliases.items()[0].kind, &aliases.items()[1].kind) else { panic!("joint"); };
        assert!(Arc::ptr_eq(a, b));
        let mut fields = Map::new(&vm.heap, 1).unwrap();
        fields.set(Arc::from("x"), vm.array_value(vec![Value::number(7.0)]).unwrap(), false).unwrap();
        let fields = Value::map(Arc::new(Mutex::new(fields)));
        let paths = Value::paths(Arc::new(PathSet { dependency_id: 17, alternatives: vec![
            PathAlternative { guard:true, weight:0.75, result:fields.clone() },
            PathAlternative { guard:false, weight:0.25, result:Value::number(2.0) },
        ] }));
        let snapshot = vm.information_snapshot(&paths).unwrap();
        let ValueKind::PathSet(captured) = snapshot.kind else { panic!("paths"); };
        assert_eq!(captured.dependency_id, 17);
        assert!(captured.alternatives[0].guard);
        assert_eq!(captured.alternatives[1].weight, 0.25);
        let ValueKind::Map(map) = &captured.alternatives[0].result.kind else { panic!("map"); };
        let array = map.lock().unwrap().get("x").unwrap().clone();
        assert_eq!(map.lock().unwrap().set(Arc::from("x"), Value::null(), false), Err(LanaError::UnsupportedOperation));
        let mut out = Value::null();
        assert_eq!(vm.execute_host_call(LANA_HOST_ARRAY_PUSH, &[array.clone(), Value::null()], &mut out), LanaError::UnsupportedOperation);
        assert_eq!(vm.execute_host_call(LANA_HOST_INDEX_SET, &[array, Value::number(0.0), Value::null()], &mut out), LanaError::UnsupportedOperation);
        assert!(matches!(vm.information_snapshot(&Value::function(0)), Err(LanaError::UnsupportedValue)));
        let mut limited = Vm::new(&chunk);
        limited.set_instruction_limit(0);
        assert!(matches!(limited.information_snapshot(&fields), Err(LanaError::Limit)));
    }

    #[test]
    fn failed_host_call_does_not_publish_partial_value_or_revision() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        vm.current_frame_mut().registers[5] = Value::number(77.0);
        vm.revision = 4;
        vm.set_host_call_extension(Box::new(|_, _, _, out| {
            *out = Value::number(99.0);
            LanaError::Oom
        }));
        assert_eq!(vm.execute(&Instruction::new(OpCode::HostCall, 5, 9999, 0, 0, 1)), LanaError::Oom);
        assert_eq!(vm.current_frame().registers[5].print(), "77");
        assert_eq!(vm.revision, 4);
        assert!(matches!(vm.result().unwrap().value.kind, ValueKind::Null));
    }

    #[test]
    fn panicking_host_callback_stops_scheduler_workers() {
        let chunk = assembler::assemble("HOST_CALL store_open R0 0 R1\nHALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        vm.set_worker_count(2);
        vm.set_host_call_extension(Box::new(|_, _, _, _| panic!("host callback")));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| vm.run()));
        assert!(result.is_err());
        assert!(vm.scheduler.as_ref().unwrap().state.lock().unwrap().stopping);
    }

    #[test]
    fn host_graph_output_is_admitted_before_register_publication() {
        let chunk = assembler::assemble("HOST_CALL store_open R0 0 R1\nRETURN R1\n").unwrap();
        for exhausted in [false, true] {
            let mut vm = Vm::new(&chunk);
            vm.current_frame_mut().registers[1] = Value::number(77.0);
            vm.set_host_call_extension(Box::new(|vm, _, _, out| {
                let source = Value::adt(Arc::new(Adt { variant: 0, fields: vec![Value::number(9.0)] }));
                match vm.import_value(&source) {
                    Ok(value) => { *out = value; LanaError::Ok },
                    Err(error) => error,
                }
            }));
            if exhausted { vm.set_memory_limit(vm.allocated_bytes()).unwrap(); }
            let error = vm.run();
            if exhausted {
                assert_eq!(error, LanaError::Oom);
                assert_eq!(vm.current_frame().registers[1].as_number(), 77.0);
            } else {
                assert_eq!(error, LanaError::Ok);
                let root = vm.result().unwrap();
                drop(vm);
                assert_eq!(root.child(0).unwrap().unwrap().as_number(), 9.0);
            }
        }
    }

    #[test]
    fn vm_failure_matrix_keeps_public_state_unchanged() {
        for (failure, expected) in [
            ("cancelled", LanaError::Cancelled),
            ("instruction_limit", LanaError::Limit),
            ("memory_limit", LanaError::Oom),
            ("malformed_bytecode", LanaError::Format),
        ] {
            let mut chunk = assembler::assemble(
                "LOAD_CONST R0 7\nARRAY_NEW R1 R0 1\nRETURN R1\n",
            ).unwrap();
            if failure == "malformed_bytecode" { chunk.entry = 999; }
            let mut vm = Vm::new(&chunk);
            vm.current_frame_mut().registers[1] = Value::number(77.0);
            vm.revision = 4;
            match failure {
                "cancelled" => vm.cancelled.store(true, Ordering::Relaxed),
                "instruction_limit" => vm.set_instruction_limit(0),
                "memory_limit" => vm.set_memory_limit(vm.allocated_bytes()).unwrap(),
                _ => {}
            }
            assert_eq!(vm.run(), expected, "{failure}");
            assert_eq!(vm.current_frame().registers[1].print(), "77", "{failure}");
            assert_eq!(vm.revision, 4, "{failure}");
            assert_eq!(vm.observation_count, 0, "{failure}");
            assert!(matches!(vm.result().unwrap().value.kind, ValueKind::Null), "{failure}");
        }
    }

    #[test]
    fn failed_observation_preserves_register_and_revision() {
        let chunk = assembler::assemble(
            ".version 5\nLOAD_CONST R0 1\nLOAD_CONST R1 2\nARRAY_NEW R2 R0 2\nPOSSIBILITY_BUILD R2 R3\nLOAD_CONST R4 3\nLOAD_CONST R5 77\nOBSERVE_MAP R3 R5 R4\nRETURN R5\n",
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run(), LanaError::InvalidConditioning);
        assert_eq!(vm.current_frame().registers[5].print(), "77");
        assert_eq!(vm.revision, 0);
        assert_eq!(vm.observation_count, 0);
        assert!(matches!(vm.result().unwrap().value.kind, ValueKind::Null));
    }

    #[test]
    fn failed_planned_effect_does_not_record_receipt() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let payload = vm.array_value(vec![Value::number(7.0)]).unwrap();
        let plan = vm.planned_effect("copy", &payload).unwrap();
        let state = plan.planned_effect.as_ref().unwrap().state.lock().unwrap();
        assert!(state.receipts.is_empty());
        drop(state);
        assert_eq!(vm.set_memory_limit(0), Err(LanaError::Oom));
        vm.set_memory_limit(vm.allocated_bytes()).unwrap();
        let failed = vm.execute_planned_effect(&plan);
        assert!(matches!(failed, Err(LanaError::Oom)), "{failed:?}");
        let state = plan.planned_effect.as_ref().unwrap().state.lock().unwrap();
        assert!(state.receipts.is_empty());
        assert_eq!(state.execution_count, 0);
        assert_eq!(vm.revision, 0);
    }

    #[test]
    fn pure_kernel_callback_rejects_effect_opcodes() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        vm.pure_callback_depth = 1;
        assert_eq!(vm.execute(&Instruction::new(OpCode::Print, 0, 0, 0, 0, 1)), LanaError::UnsupportedOperation);
        assert_eq!(vm.execute(&Instruction::new(OpCode::HostCall, 0, LANA_HOST_WRITE_TEXT, 0, 0, 1)), LanaError::UnsupportedOperation);
    }

    #[test]
    fn named_pure_dataset_plan_runs_with_ordered_sources_and_rejects_effects() {
        let chunk = assembler::assemble(
            ".function main 0 4\nHALT\n.function plan 2 5\nHOST_CALL dataset R0 1 R2\nHOST_CALL dataset_materialize R2 1 R3\nRETURN R3\n.function effect 0 2\nLOAD_CONST R0 1\nPRINT R0\nRETURN R0\n.function scalar 0 2\nLOAD_CONST R0 1\nRETURN R0\n.function mutate 1 4\nLOAD_CONST R1 1\nHOST_CALL array_push R0 2 R2\nRETURN R0\n",
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        let mut fields = Map::new(&vm.heap, 1).unwrap();
        fields.set(Arc::from("v"), Value::number(7.0), false).unwrap();
        let rows = vm.array_value(vec![Value::map(Arc::new(Mutex::new(fields)))]).unwrap();
        vm.current_frame_mut().registers[0] = Value::number(99.0);
        let result = vm.run_pure_dataset_plan("plan", &[rows, Value::null()]).unwrap();
        assert_eq!(result.print(), "[{\"v\": 7}]");
        let invalid_rows = vm.array_value(vec![Value::number(7.0)]).unwrap();
        assert_eq!(vm.run_pure_dataset_plan("plan", &[invalid_rows, Value::null()]).unwrap_err(), LanaError::Type);
        assert_eq!(vm.current_frame().registers[0].print(), "99");
        assert!(matches!(vm.run_pure_dataset_plan("effect", &[]), Err(LanaError::UnsupportedOperation)));
        assert!(matches!(vm.run_pure_dataset_plan("missing", &[]), Err(LanaError::NotFound)));
        assert!(matches!(vm.run_pure_dataset_plan("plan", &[]), Err(LanaError::Type)));
        assert!(matches!(vm.run_pure_dataset_plan("scalar", &[]), Err(LanaError::Type)));
        let input = vm.array_value(vec![Value::number(3.0)]).unwrap();
        assert!(matches!(vm.run_pure_dataset_plan("mutate", &[input.clone()]), Err(LanaError::UnsupportedOperation)));
        assert_eq!(input.print(), "[3]");
    }

    #[test]
    fn dataset_plan_budget_counts_source_and_filter_work_without_partial_result() {
        let chunk = assembler::assemble(
            ".function main 0 4\nHALT\n.function predicate 1 2\nLOAD_CONST R0 true\nRETURN R0\n.function plan 1 2\nRETURN R0\n",
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        let rows = vm.array_value(vec![Value::number(1.0); 3]).unwrap();
        let mut source = Value::null();
        assert_eq!(vm.host_dataset(&[rows], &mut source), LanaError::Ok);
        let mut filtered = Value::null();
        assert_eq!(vm.host_dataset_filter(&[source, Value::function(1)], &mut filtered), LanaError::Ok);
        vm.dataset_work_remaining = Some(5);
        let mut output = Value::number(99.0);
        assert_eq!(vm.host_dataset_materialize(&[filtered.clone()], &mut output), LanaError::Limit);
        assert_eq!(output.as_number(), 99.0);
        vm.dataset_work_remaining = Some(6);
        assert_eq!(vm.host_dataset_materialize(&[filtered], &mut output), LanaError::Ok);
        assert_eq!(vm.dataset_work_remaining, Some(0));
        vm.dataset_work_remaining = None;
        let oversized = vm.array_value(vec![Value::null(); 100_001]).unwrap();
        assert_eq!(vm.run_pure_dataset_plan("plan", &[oversized]).unwrap_err(), LanaError::Limit);
        assert_eq!(vm.dataset_work_remaining, None);
    }

    #[test]
    fn dataset_plan_trace_keeps_source_ids_and_excluded_decisions() {
        let chunk = assembler::assemble(
            ".function main 0 4\nHALT\n.function plan 1 4\nHOST_CALL dataset_materialize R0 1 R1\nRETURN R1\n.function keep 1 2\nLOAD_CONST R0 true\nRETURN R0\n.function reject 1 2\nLOAD_CONST R0 false\nRETURN R0\n",
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        let row = |vm: &mut Vm, source: &str, id: &str, number: f64| {
            let mut fields = Map::new(&vm.heap, 1).unwrap();
            fields.set(Arc::from("v"), Value::number(number), false).unwrap();
            vm.dataset_source_row(source, id, Value::map(Arc::new(Mutex::new(fields)))).unwrap()
        };
        let left_items = vec![row(&mut vm, "left", "a", 1.0), row(&mut vm, "left", "b", 2.0)];
        let ValueKind::Map(first) = &left_items[0].kind else { panic!("source row"); };
        assert!(Arc::ptr_eq(first.lock().unwrap().get("v").unwrap().derivation.as_ref().unwrap(),
            left_items[0].derivation.as_ref().unwrap()));
        let left_rows = vm.array_value(left_items).unwrap();
        let mut left = Value::null();
        assert_eq!(vm.host_dataset(&[left_rows], &mut left), LanaError::Ok);
        let mut filtered = Value::null();
        assert_eq!(vm.host_dataset_filter(&[left.clone(), Value::function(2)], &mut filtered), LanaError::Ok);
        let mut limited = Value::null();
        assert_eq!(vm.host_dataset_limit(&[filtered, Value::number(1.0)], &mut limited), LanaError::Ok);
        let output = vm.run_pure_dataset_plan("plan", &[limited]).unwrap();
        let ValueKind::Array(rows) = output.kind else { panic!("rows"); };
        let rows = rows.lock().unwrap();
        let derivation = rows.items()[0].derivation.as_ref().unwrap();
        assert_eq!(derivation.operation.as_ref(), "limit");
        assert_eq!(derivation.inputs[0].operation.as_ref(), "filter");
        assert_eq!(derivation.inputs[0].inputs[0].label.as_ref(), "4:left1:a");
        let ValueKind::Map(retained) = &rows.items()[0].kind else { panic!("retained row"); };
        assert_eq!(retained.lock().unwrap().get("v").unwrap().derivation.as_ref().unwrap().label.as_ref(), "4:left1:a");
        assert_eq!(vm.dataset_decisions().len(), 1);
        assert_eq!(vm.dataset_decisions()[0].reason, "limit_excluded");
        assert_eq!(vm.dataset_decisions()[0].predicate_value, None);
        assert_eq!(vm.dataset_decisions()[0].input.inputs[0].label.as_ref(), "4:left1:b");
        drop(rows);

        let mut rejected = Value::null();
        assert_eq!(vm.host_dataset_filter(&[left.clone(), Value::function(3)], &mut rejected), LanaError::Ok);
        assert_eq!(vm.run_pure_dataset_plan("plan", &[rejected]).unwrap().print(), "[]");
        assert_eq!(vm.dataset_decisions().len(), 2);
        assert!(vm.dataset_decisions().iter().all(|decision| decision.reason == "filter_false"));
        assert!(vm.dataset_decisions().iter().all(|decision| decision.predicate_value == Some(false)));

        let right_items = vec![row(&mut vm, "right", "c", 2.0), row(&mut vm, "right", "d", 3.0)];
        let right_rows = vm.array_value(right_items).unwrap();
        let mut right = Value::null();
        assert_eq!(vm.host_dataset(&[right_rows], &mut right), LanaError::Ok);
        let mut joined = Value::null();
        assert_eq!(vm.host_dataset_join(&[left.clone(), right, Value::string(Arc::from("v"))], &mut joined), LanaError::Ok);
        let output = vm.run_pure_dataset_plan("plan", &[joined]).unwrap();
        let ValueKind::Array(rows) = output.kind else { panic!("joined rows"); };
        let rows = rows.lock().unwrap();
        let derivation = rows.items()[0].derivation.as_ref().unwrap();
        assert_eq!(derivation.operation.as_ref(), "join");
        assert_eq!(derivation.label.as_ref(), "j9:4:left1:b10:5:right1:c");
        assert_eq!(derivation.inputs[0].label.as_ref(), "4:left1:b");
        assert_eq!(derivation.inputs[1].label.as_ref(), "5:right1:c");
        assert_eq!(vm.dataset_decisions().len(), 2);
        assert_eq!(vm.dataset_decisions()[0].reason, "no_matching_key");
        assert_eq!(vm.dataset_decisions()[0].input.label.as_ref(), "4:left1:a");
        assert_eq!(vm.dataset_decisions()[1].input.label.as_ref(), "5:right1:d");
        assert!(matches!(vm.run_pure_dataset_plan("missing", &[]), Err(LanaError::NotFound)));
        assert!(vm.dataset_decisions().is_empty());
    }

    #[test]
    fn dataset_source_cell_trace_does_not_mutate_a_shared_input_map() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let mut fields = Map::new(&vm.heap, 1).unwrap();
        fields.set(Arc::from("v"), Value::number(1.0), false).unwrap();
        let original = Value::map(Arc::new(Mutex::new(fields)));
        let traced = vm.dataset_source_row("s", "r", original.clone()).unwrap();
        let ValueKind::Map(original_map) = &original.kind else { panic!("original map"); };
        assert!(original_map.lock().unwrap().get("v").unwrap().derivation.is_none());
        let ValueKind::Map(traced_map) = &traced.kind else { panic!("traced map"); };
        assert!(Arc::ptr_eq(traced_map.lock().unwrap().get("v").unwrap().derivation.as_ref().unwrap(),
            traced.derivation.as_ref().unwrap()));
    }

    #[test]
    fn dataset_plan_trace_survives_map_projection_sort_and_group_aggregate() {
        let chunk = assembler::assemble(
            ".function main 0 4\nHALT\n.function plan 1 4\nHOST_CALL dataset_materialize R0 1 R1\nRETURN R1\n.function identity 1 2\nRETURN R0\n",
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        let mut items = Vec::new();
        for (id, number) in [("a", 2.0), ("b", 1.0), ("c", 1.0)] {
            let mut fields = Map::new(&vm.heap, 1).unwrap();
            fields.set(Arc::from("v"), Value::number(number), false).unwrap();
            items.push(vm.dataset_source_row("s", id, Value::map(Arc::new(Mutex::new(fields)))).unwrap());
        }
        let source_rows = vm.array_value(items).unwrap();
        let mut source = Value::null();
        assert_eq!(vm.host_dataset(&[source_rows], &mut source), LanaError::Ok);
        let mut mapped = Value::null();
        assert_eq!(vm.host_dataset_map(&[source.clone(), Value::function(2)], &mut mapped), LanaError::Ok);
        let columns = vm.array_value(vec![Value::string(Arc::from("v"))]).unwrap();
        let mut selected = Value::null();
        assert_eq!(vm.host_dataset_select(&[mapped, columns], &mut selected), LanaError::Ok);
        let mut sorted = Value::null();
        assert_eq!(vm.host_dataset_sort(&[selected, Value::string(Arc::from("v"))], &mut sorted), LanaError::Ok);
        let output = vm.run_pure_dataset_plan("plan", &[sorted]).unwrap();
        assert_eq!(output.print(), "[{\"v\": 1}, {\"v\": 1}, {\"v\": 2}]");
        let ValueKind::Array(rows) = output.kind else { panic!("sorted rows"); };
        let rows = rows.lock().unwrap();
        let first = rows.items()[0].derivation.as_ref().unwrap();
        assert_eq!(first.operation.as_ref(), "sort");
        assert_eq!(first.inputs[0].operation.as_ref(), "select");
        assert_eq!(first.inputs[0].inputs[0].operation.as_ref(), "map");
        assert_eq!(first.inputs[0].inputs[0].inputs[0].label.as_ref(), "1:s1:b");
        let ValueKind::Map(first_row) = &rows.items()[0].kind else { panic!("sorted map"); };
        assert_eq!(first_row.lock().unwrap().get("v").unwrap().derivation.as_ref().unwrap().label.as_ref(), "1:s1:b");
        drop(rows);

        let mut grouped = Value::null();
        assert_eq!(vm.host_dataset_group_by(&[source, Value::string(Arc::from("v"))], &mut grouped), LanaError::Ok);
        let descriptor = vm.array_value(vec![Value::string(Arc::from("count"))]).unwrap();
        let mut aggregate = Value::null();
        assert_eq!(vm.host_dataset_aggregate(&[grouped, descriptor], &mut aggregate), LanaError::Ok);
        let output = vm.run_pure_dataset_plan("plan", &[aggregate]).unwrap();
        assert_eq!(output.print(), "[{\"v\": 2, \"count\": 1}, {\"v\": 1, \"count\": 2}]");
        let ValueKind::Array(rows) = output.kind else { panic!("aggregate rows"); };
        let rows = rows.lock().unwrap();
        let second = rows.items()[1].derivation.as_ref().unwrap();
        assert_eq!(second.operation.as_ref(), "aggregate");
        let group = &second.inputs[0];
        assert_eq!(group.operation.as_ref(), "group_by");
        assert_eq!(second.label, group.label);
        assert!(second.label.starts_with("g9:"));
        assert_eq!(group.inputs[0].inputs[0].label.as_ref(), "1:s1:b");
        assert_eq!(group.inputs[1].label.as_ref(), "1:s1:c");
        let ValueKind::Map(aggregate_row) = &rows.items()[1].kind else { panic!("aggregate map"); };
        let aggregate_row = aggregate_row.lock().unwrap();
        let count = aggregate_row.get("count").unwrap().derivation.as_ref().unwrap();
        assert_eq!(count.operation.as_ref(), "count");
        assert_eq!(count.inputs[0].operation.as_ref(), "group_by");
        assert!(vm.dataset_decisions().is_empty());
    }

    #[test]
    fn dataset_group_key_identity_is_typed_and_order_independent() {
        let mut number = Vec::new();
        Vm::dataset_key_identity(&Value::number(1.0), &mut number, 0).unwrap();
        let mut text = Vec::new();
        Vm::dataset_key_identity(&Value::string(Arc::from("1")), &mut text, 0).unwrap();
        assert_ne!(number, text);
        let mut negative_zero = Vec::new();
        Vm::dataset_key_identity(&Value::number(-0.0), &mut negative_zero, 0).unwrap();
        let mut positive_zero = Vec::new();
        Vm::dataset_key_identity(&Value::number(0.0), &mut positive_zero, 0).unwrap();
        assert_eq!(negative_zero, positive_zero);

        let chunk = assembler::assemble("HALT\n").unwrap();
        let vm = Vm::new(&chunk);
        let mut left = Map::new(&vm.heap, 2).unwrap();
        left.set(Arc::from("z"), Value::number(2.0), false).unwrap();
        left.set(Arc::from("a"), Value::boolean(true), false).unwrap();
        let mut right = Map::new(&vm.heap, 2).unwrap();
        right.set(Arc::from("a"), Value::boolean(true), false).unwrap();
        right.set(Arc::from("z"), Value::number(2.0), false).unwrap();
        let mut first = Vec::new();
        Vm::dataset_key_identity(&Value::map(Arc::new(Mutex::new(left))), &mut first, 0).unwrap();
        let mut second = Vec::new();
        Vm::dataset_key_identity(&Value::map(Arc::new(Mutex::new(right))), &mut second, 0).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn host_extension_can_run_named_plan_in_same_vm() {
        let chunk = assembler::assemble(
            ".function main 0 4\nHOST_CALL store_open R0 0 R1\nRETURN R1\n.function plan 0 3\nARRAY_NEW R1 R0 0\nRETURN R1\n",
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        vm.set_host_call_extension(Box::new(|vm, _, _, out| match vm.run_pure_dataset_plan("plan", &[]) {
            Ok(value) => { *out = value; LanaError::Ok }
            Err(error) => error,
        }));
        assert_eq!(vm.run(), LanaError::Ok);
        assert_eq!(vm.result().unwrap().print(), "[]");
    }

    #[test]
    fn finite_identity_kernel_composes_without_changing_rows() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let domain = vm.array_value(vec![Value::number(0.0), Value::number(1.0)]).unwrap();
        let mut identity = Value::null();
        assert_eq!(vm.core_kernel_identity(&[domain], &mut identity), LanaError::Ok);
        let mut composed = Value::null();
        assert_eq!(vm.core_kernel_compose(&[identity.clone(), identity], &mut composed), LanaError::Ok);
        let ValueKind::Kernel(kernel) = composed.kind else { panic!("expected kernel"); };
        assert_eq!(kernel.rows, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(kernel.input_domains.len(), 1);
        let member = vm.array_value(vec![Value::number(7.0)]).unwrap();
        let nested_domain = vm.array_value(vec![member.clone()]).unwrap();
        let mut nested = Value::null();
        assert_eq!(vm.core_kernel_identity(&[nested_domain], &mut nested), LanaError::Ok);
        let mut nested_composed = Value::null();
        assert_eq!(vm.core_kernel_compose(&[nested.clone(), nested.clone()], &mut nested_composed), LanaError::Ok);
        let copied = vm.deep_clone_value(&nested_composed, &mut DeepCloneMemo::default()).unwrap();
        assert!(matches!(copied.kind, ValueKind::Kernel(_)));
        let duplicate_domain = vm.array_value(vec![member.clone(), member]).unwrap();
        assert_eq!(vm.core_kernel_identity(&[duplicate_domain], &mut Value::null()), LanaError::InvalidParameters);
    }

    #[test]
    fn atomic_text_write_preserves_old_or_absent_and_reports_uncertain_rename() {
        let root = std::env::temp_dir().join(format!(
            "lana-vm-atomic-{}-{}",
            std::process::id(),
            NEXT_ATOMIC_WRITE.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir(&root).unwrap();
        let destination = root.join("output");
        for old in [Some(b"old".as_slice()), None] {
            if let Some(old) = old { std::fs::write(&destination, old).unwrap(); }
            else { let _ = std::fs::remove_file(&destination); }
            let failure = write_text_atomic_with_sync(
                &destination,
                b"new",
                |_| Err(std::io::Error::other("before rename")),
                |_| Ok(()),
            ).unwrap_err();
            assert!(!failure.1);
            assert_eq!(std::fs::read(&destination).ok().as_deref(), old);
            assert_eq!(std::fs::read_dir(&root).unwrap().count(), usize::from(old.is_some()));
        }
        let failure = write_text_atomic_with_sync(
            &destination,
            b"complete",
            |file| file.sync_all(),
            |_| Err(std::io::Error::other("after rename")),
        ).unwrap_err();
        assert!(failure.1);
        assert_eq!(std::fs::read(&destination).unwrap(), b"complete");
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_text_host_call_publishes_and_reports_failed_destination() {
        let root = std::env::temp_dir().join(format!(
            "lana-vm-host-atomic-{}-{}",
            std::process::id(),
            NEXT_ATOMIC_WRITE.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir(&root).unwrap();
        for (destination, expected) in [(root.join("output"), LanaError::Ok), (root.join("directory"), LanaError::Io)] {
            if expected == LanaError::Io { std::fs::create_dir(&destination).unwrap(); }
            let path_hex: String = destination.to_string_lossy().as_bytes().iter().map(|byte| format!("{byte:02x}")).collect();
            let source = format!("LOAD_STRING R0 {path_hex}\nLOAD_STRING R1 6e6577\nHOST_CALL write_text_atomic R0 2 R2\nRETURN R2\n");
            let chunk = assembler::assemble(&source).unwrap();
            let mut vm = Vm::new(&chunk);
            assert_eq!(vm.run(), expected);
            if expected == LanaError::Ok {
                assert_eq!(std::fs::read(&destination).unwrap(), b"new");
            } else {
                assert!(destination.is_dir());
                assert_eq!(vm.error().path.as_deref(), Some(destination.to_str().unwrap()));
                assert_eq!(vm.error().durability, None);
            }
        }
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    fn run_chunk(source: &str) -> (LanaError, String) {
        let chunk = assembler::assemble(source).expect("fixture assembles");
        let mut vm = Vm::new(&chunk);
        let error = vm.run();
        (error, vm.result().unwrap().print())
    }

    #[test]
    fn scheduler_service_root_keeps_a_retired_task_heap_until_release() {
        let chunk = assembler::assemble(
            ".function main 0 4\nHALT\n.function worker 1 4\nRETURN R0\n"
        ).unwrap();
        let scheduler = Scheduler::new();
        let mut source = Vm::new(&chunk);
        source.scheduler = Some(scheduler.clone());
        source.scheduler_owner = false;
        source.frames[0].registers[0] = source.array_value(vec![Value::number(7.0)]).unwrap();
        let owner = Arc::downgrade(&source.host_roots);
        let task = source.start_task(1, 1, 0).unwrap();
        let task_weak = Arc::downgrade(&task);
        drop(task);
        drop(source);
        assert!(owner.upgrade().is_some());
        let queued = scheduler.state.lock().unwrap().queue.pop_front().unwrap();
        run_task(queued);
        let result_weak = {
            let task = task_weak.upgrade().unwrap();
            let state = task.state.lock().unwrap();
            let ValueKind::Array(array) = &state.result.kind else { unreachable!() };
            Arc::downgrade(array)
        };
        assert!(result_weak.upgrade().is_some());
        scheduler.state.lock().unwrap().all_tasks.clear();
        assert!(owner.upgrade().is_none());
        assert!(task_weak.upgrade().is_none());
        assert!(result_weak.upgrade().is_none());
    }

    #[test]
    fn retained_joined_task_cycle_releases_after_last_root() {
        let chunk = assembler::assemble(
            ".function main 0 4\nFORK worker R1 0 R0\nJOIN R0 R1\nRETURN R0\n.function worker 0 2\nARRAY_NEW R0 R1 0\nRETURN R0\n"
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        vm.configured_worker_count = 0;
        assert_eq!(vm.run(), LanaError::Ok);
        let root = vm.result().unwrap();
        let ValueKind::Task(task) = &root.value.kind else { unreachable!() };
        let task_weak = Arc::downgrade(task);
        let state = task.state.lock().unwrap();
        let ValueKind::Array(array) = &state.result.kind else { unreachable!() };
        let array_weak = Arc::downgrade(array);
        array.lock().unwrap().items.push(Value::task(task.clone())).unwrap();
        drop(state);
        vm.heap.mutate_cycle(array_weak.as_ptr() as usize);
        let heap = vm.heap();
        drop(vm);
        assert!(task_weak.upgrade().is_some());
        assert!(array_weak.upgrade().is_some());
        drop(root);
        assert!(task_weak.upgrade().is_none());
        assert!(array_weak.upgrade().is_none());
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn fork_transfers_composite_future_dependencies() {
        for (composite, expected) in [("future_all", "[10, 20]"), ("future_race", "10")] {
            let source = format!(
                ".version 4\n.function main 0 10\nASYNC foo R0 0 R1\nASYNC bar R0 0 R2\nARRAY_NEW R3 R1 2\nHOST_CALL {composite} R3 1 R4\nFORK worker R4 1 R5\nJOIN R5 R6\nRETURN R6\n.function foo 0 3\nLOAD_CONST R1 10\nRETURN R1\n.function bar 0 3\nLOAD_CONST R1 20\nRETURN R1\n.function worker 1 3\nRUN_ASYNC R0 R1\nRETURN R1\n"
            );
            let (status, result) = run_chunk(&source);
            assert_eq!(status, LanaError::Ok);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn task_transfer_copies_dataset_and_claim_graphs_with_one_memo() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut source_vm = Vm::new(&chunk);
        let mut receiver = Vm::new(&chunk);
        let captured = source_vm.array_value(vec![Value::number(7.0)]).unwrap();
        let dataset = Value::dataset(source_vm.managed_payload(Dataset {
            op: DatasetOp::Source, source: captured.clone(), function: 0,
            columns: captured.clone(), key: Value::null(), limit: Value::null(),
            other: Value::null(), aggregate: Value::null(),
        }).unwrap());
        let mut claimed = captured.clone();
        claimed.claim = Some(source_vm.managed_payload(Claim {
            value: captured.clone(), proposition: Arc::from("captured"),
            exactness: DerivationExactness::Exact, tolerance: 0.0, source_valid: true,
        }).unwrap());
        let graph = source_vm.array_value(vec![dataset, claimed]).unwrap();
        let copy = receiver.deep_clone_value(&graph, &mut DeepCloneMemo {
            transfer: true, ..DeepCloneMemo::default()
        }).unwrap();
        let ValueKind::Array(graph) = &copy.kind else { unreachable!() };
        let graph = graph.lock().unwrap();
        let ValueKind::Dataset(dataset) = &graph.items[0].kind else { unreachable!() };
        let ValueKind::Array(source) = &dataset.source.kind else { unreachable!() };
        let ValueKind::Array(columns) = &dataset.columns.kind else { unreachable!() };
        let ValueKind::Array(claimed) = &graph.items[1].kind else { unreachable!() };
        let claim = graph.items[1].claim.as_ref().unwrap();
        let ValueKind::Array(claim_value) = &claim.value.kind else { unreachable!() };
        assert!(Arc::ptr_eq(source, columns));
        assert!(Arc::ptr_eq(source, claimed));
        assert!(Arc::ptr_eq(source, claim_value));
        source.lock().unwrap().items[0] = Value::number(9.0);
        assert_eq!(captured.print(), "[7]");
        assert_eq!(claim.proposition.as_ref(), "captured");
        drop(graph);
        let root = receiver.retain_value(&copy).unwrap();
        drop(copy);
        drop(receiver);
        assert!(root.print().contains('9'));
        drop(root);
    }

    #[test]
    fn cancelled_task_transfer_publishes_no_task() {
        let chunk = assembler::assemble(
            ".function main 0 4\nHALT\n.function worker 1 4\nRETURN R0\n"
        ).unwrap();
        for arguments in [0, 1] {
            let mut vm = Vm::new(&chunk);
            let scheduler = Scheduler::new();
            vm.scheduler = Some(scheduler.clone());
            vm.frames[0].registers[0] = vm.array_value(vec![Value::number(7.0)]).unwrap();
            vm.cancelled.store(true, Ordering::Relaxed);
            let function = if arguments == 0 { 0 } else { 1 };
            assert_eq!(vm.start_task(function, arguments, 0).unwrap_err(), LanaError::Cancelled);
            let state = scheduler.state.lock().unwrap();
            assert_eq!(state.live_tasks, 0);
            assert!(state.queue.is_empty());
            assert!(state.all_tasks.is_empty());
            assert!(vm.tasks.is_empty());
        }
    }

    #[test]
    fn task_transfer_rebuilds_waiting_future_dependencies() {
        let chunk = assembler::assemble(
            ".version 4\n.function main 0 4\nHALT\n.function waiting 0 4\nAWAIT R1 R2\nRETURN R2\n.function dependency 0 3\nLOAD_CONST R1 42\nRETURN R1\n"
        ).unwrap();
        let dependency = Arc::new(Mutex::new(Future {
            function: 2, ip: chunk.functions[2].entry as usize,
            registers: vec![Value::null(); 3], exhausted: false, ready: true, queued: true,
        }));
        let waiting = Arc::new(Mutex::new(Future {
            function: 1, ip: chunk.functions[1].entry as usize,
            registers: vec![Value::null(), Value::future(dependency.clone()), Value::null(), Value::null()],
            exhausted: false, ready: false, queued: false,
        }));
        let source = Value::future(waiting.clone());
        let mut receiver = Vm::new(&chunk);
        let copy = receiver.deep_clone_value(&source, &mut DeepCloneMemo {
            transfer: true, ..DeepCloneMemo::default()
        }).unwrap();
        let mut result = Value::null();
        assert_eq!(receiver.host_run_async(&[copy], &mut result), LanaError::Ok);
        assert_eq!(result.as_number(), 42.0);
        assert!(!waiting.lock().unwrap().exhausted);
        assert!(!dependency.lock().unwrap().exhausted);
        let guard = waiting.lock().unwrap();
        assert_eq!(receiver.deep_clone_value(&source, &mut DeepCloneMemo {
            transfer: true, ..DeepCloneMemo::default()
        }).unwrap_err(), LanaError::UnsupportedOperation);
        drop(guard);
    }

    #[test]
    fn task_transfer_copies_suspended_captures_and_preserves_aliases() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        for future in [false, true] {
            let source_vm = Vm::new(&chunk);
            let mut receiver = Vm::new(&chunk);
            let array = Arc::new(Mutex::new(Array::new(&source_vm.heap, 1).unwrap()));
            let captured = Value::array(array.clone());
            array.lock().unwrap().items.push(Value::number(7.0)).unwrap();
            let registers = vec![Value::null(), captured.clone(), captured];
            let source = if future {
                Value { kind: ValueKind::Future(Arc::new(Mutex::new(Future {
                    function: 0, ip: 0, registers, exhausted: false, ready: true, queued: true,
                }))), ..Value::null() }
            } else {
                Value { kind: ValueKind::Generator(Arc::new(Mutex::new(Generator {
                    function: 0, ip: 0, registers, exhausted: false,
                }))), ..Value::null() }
            };
            let copy = receiver.deep_clone_value(&source, &mut DeepCloneMemo {
                transfer: true, ..DeepCloneMemo::default()
            }).unwrap();
            let registers = match &copy.kind {
                ValueKind::Future(frame) => {
                    let frame = frame.lock().unwrap();
                    assert!(!frame.queued);
                    frame.registers.clone()
                }
                ValueKind::Generator(frame) => frame.lock().unwrap().registers.clone(),
                _ => unreachable!(),
            };
            let ValueKind::Array(first) = &registers[1].kind else { unreachable!() };
            let ValueKind::Array(second) = &registers[2].kind else { unreachable!() };
            assert!(Arc::ptr_eq(first, second));
            assert!(!Arc::ptr_eq(first, &array));
            first.lock().unwrap().items[0] = Value::number(9.0);
            assert_eq!(array.lock().unwrap().items[0].as_number(), 7.0);
            let root = receiver.retain_value(&copy).unwrap();
            drop(registers);
            drop(copy);
            drop(receiver);
            assert_eq!(root.type_name(), if future { "future" } else { "generator" });
            drop(root);
        }
    }

    #[test]
    fn cloned_set_cycles_keep_managed_ownership_through_teardown() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        for mode in 0..3 {
            let mut vm = Vm::new(&chunk);
            let set = Arc::new(Mutex::new(Set::new(&vm.heap, 1).unwrap()));
            let source = Value::set(set.clone());
            set.lock().unwrap().items.push(source.clone()).unwrap();
            let copy = match mode {
                0 => vm.deep_clone_value(&source, &mut DeepCloneMemo { transfer: true, ..DeepCloneMemo::default() }).unwrap(),
                1 => vm.materialize_value(&source).unwrap(),
                _ => vm.deep_clone_live_value(&source, &mut HashMap::new()).unwrap(),
            };
            let ValueKind::Set(copied) = &copy.kind else { unreachable!() };
            let weak = Arc::downgrade(copied);
            let back_edge = copied.lock().unwrap().items[0].clone();
            let ValueKind::Set(back_edge) = back_edge.kind else { unreachable!() };
            assert!(Arc::ptr_eq(copied, &back_edge));
            drop(back_edge);
            let root = vm.retain_value(&copy).unwrap();
            drop(copy);
            set.lock().unwrap().items.clear();
            drop(source);
            drop(set);
            let heap = vm.heap();
            drop(vm);
            assert!(weak.upgrade().is_some());
            drop(root);
            assert!(weak.upgrade().is_none());
            assert_eq!(heap.live_bytes(), 0);
        }
    }

    #[test]
    fn cyclic_container_clone_and_materialization_preserve_identity() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let array = Arc::new(Mutex::new(Array::new(&vm.heap, 0).unwrap()));
        let source = Value::array(array.clone());
        array.lock().unwrap().items.push(source.clone()).unwrap();
        for copy in [
            vm.deep_clone_value(&source, &mut DeepCloneMemo::default()).unwrap(),
            vm.materialize_value(&source).unwrap(),
            vm.deep_clone_live_value(&source, &mut HashMap::new()).unwrap(),
        ] {
            let ValueKind::Array(cloned) = &copy.kind else { panic!("array expected") };
            assert!(!Arc::ptr_eq(&array, cloned));
            let child = cloned.lock().unwrap().items[0].clone();
            let ValueKind::Array(back_edge) = child.kind else { panic!("array expected") };
            assert!(Arc::ptr_eq(cloned, &back_edge));
            assert_eq!(copy.print(), "[<cycle>]");
            cloned.lock().unwrap().items.clear();
        }
        array.lock().unwrap().items.clear();
    }

    #[test]
    fn clone_memos_survive_immutable_wrappers_and_reactive_history() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let array = Arc::new(Mutex::new(Array::new(&vm.heap, 1).unwrap()));
        let source = Value::array(array.clone());
        array.lock().unwrap().push(Value::adt(Arc::new(Adt { variant: 1, fields: vec![source.clone()] }))).unwrap();
        for copy in [vm.deep_clone_value(&source, &mut DeepCloneMemo::default()).unwrap(),
            vm.materialize_value(&source).unwrap(), vm.deep_clone_live_value(&source, &mut HashMap::new()).unwrap()] {
            let ValueKind::Array(cloned) = copy.kind else { panic!("array expected") };
            let wrapper = cloned.lock().unwrap().items[0].clone();
            let ValueKind::Adt(wrapper) = wrapper.kind else { panic!("ADT expected") };
            let ValueKind::Array(back_edge) = &wrapper.fields[0].kind else { panic!("array expected") };
            assert!(Arc::ptr_eq(&cloned, back_edge));
            cloned.lock().unwrap().items.clear();
        }
        array.lock().unwrap().items.clear();
        let root = vm.reactive_root(&source, DerivationExactness::Exact).unwrap();
        let reactive = root.reactive.as_ref().unwrap();
        let current = reactive.lock().unwrap().current.clone();
        reactive.lock().unwrap().history.push(ReactiveVersion { revision: 0, value: current });
        let copy = vm.deep_clone_live_value(&root, &mut HashMap::new()).unwrap();
        let cloned_reactive = copy.reactive.unwrap();
        assert!(!Arc::ptr_eq(reactive, &cloned_reactive));
        let cloned_reactive = cloned_reactive.lock().unwrap();
        let ValueKind::Array(current) = &cloned_reactive.current.as_ref().unwrap().kind else { panic!("array expected") };
        let ValueKind::Array(history) = &cloned_reactive.history[0].value.as_ref().unwrap().kind else { panic!("array expected") };
        assert!(Arc::ptr_eq(current, history));
    }

    #[test]
    fn lowering_memory_limit_is_transactional() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let value = vm.array_value(vec![Value::number(7.0)]).unwrap();
        let bytes = vm.allocated_bytes();
        let original = vm.memory_limit;
        assert_eq!(vm.set_memory_limit(bytes - 1), Err(LanaError::Oom));
        assert_eq!(vm.memory_limit, original);
        assert_eq!(value.print(), "[7]");
        assert!(vm.array_value(vec![Value::null()]).is_ok());
        vm.set_memory_limit(vm.allocated_bytes()).unwrap();
        assert!(matches!(vm.array_value(vec![Value::null()]), Err(LanaError::Oom)));
        assert_eq!(value.print(), "[7]");
        vm.set_memory_limit(original).unwrap();
        assert!(vm.array_value(vec![Value::null()]).is_ok());
    }

    #[test]
    fn cross_heap_string_clones_are_charged_once_per_source() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let source = Value::string(Arc::from("abc"));
        let baseline = vm.allocated_bytes();
        vm.set_memory_limit(vm.allocated_bytes() + 3).unwrap();
        assert!(matches!(vm.deep_clone_value(&source, &mut DeepCloneMemo::default()), Err(LanaError::Oom)));
        vm.set_memory_limit(vm.allocated_bytes() + 4).unwrap();
        let mut memo = DeepCloneMemo::default();
        let first = vm.deep_clone_value(&source, &mut memo).unwrap().as_string();
        let second = vm.deep_clone_value(&source, &mut memo).unwrap().as_string();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(!Arc::ptr_eq(&first, &source.as_string()));
        assert_eq!(vm.heap.live_bytes(), baseline + 4);
    }

    #[test]
    fn json_rejects_cyclic_materialized_argument() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0\nARRAY_NEW R1 R0 1\nARRAY_SET R1 R0 R1\nHOST_CALL json_stringify R1 1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::UnsupportedOperation);
    }

    #[test]
    fn array_budget_rejects_payload_before_allocation() {
        let chunk = assembler::assemble(
            "LOAD_CONST R0 1000000\nHOST_CALL array_new R0 1 R1\nRETURN R1\n",
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        vm.set_memory_limit(1024 * 1024).unwrap();
        let baseline = vm.allocated_bytes();
        assert_eq!(vm.run(), LanaError::Oom);
        assert!(matches!(vm.result().unwrap().value.kind, ValueKind::Null));
        assert_eq!(vm.allocated_bytes(), baseline);
        assert_eq!(vm.alloc_bytes(usize::MAX), LanaError::Oom);
        assert_eq!(vm.allocated_bytes(), baseline);
    }

    #[test]
    fn dynamic_strings_obey_the_heap_budget() {
        let source = "LOAD_CONST R0 \"abcdefgh\"\nMOVE R1 R0\nHOST_CALL string_concat R0 2 R2\nRETURN R2\n";
        let chunk = assembler::assemble(source).unwrap();
        let mut vm = Vm::new(&chunk);
        vm.set_memory_limit(vm.allocated_bytes() + 8).unwrap();
        assert_eq!(vm.run(), LanaError::Oom);
        assert!(matches!(vm.result().unwrap().value.kind, ValueKind::Null));
        for (host, arguments) in [
            (LANA_HOST_STRING_HEX, vec![Value::string(Arc::from("abc"))]),
            (LANA_HOST_STRING_UNESCAPE, vec![Value::string(Arc::from("abc"))]),
            (LANA_HOST_TO_UPPER, vec![Value::string(Arc::from("abc"))]),
            (LANA_HOST_TO_LOWER, vec![Value::string(Arc::from("abc"))]),
            (LANA_HOST_STRING_SLICE, vec![Value::string(Arc::from("abc")), Value::number(0.), Value::number(3.)]),
            (LANA_HOST_FORMAT, vec![Value::string(Arc::from("{}")), Value::string(Arc::from("abc"))]),
            (LANA_HOST_JSON_PARSE, vec![Value::string(Arc::from("\"abc\""))]),
            (LANA_HOST_JSON_STRINGIFY, vec![Value::string(Arc::from("abc"))]),
        ] {
            let mut vm = Vm::new(&chunk);
            vm.set_memory_limit(vm.allocated_bytes() + 2).unwrap();
            let mut out = Value::number(99.);
            assert_eq!(vm.execute_host_call(host, &arguments, &mut out), LanaError::Oom, "host {host}");
            assert!(matches!(out.kind, ValueKind::Null));
            vm.heap.collect_strings();
            let heap = vm.heap();
            drop(vm);
            assert_eq!(heap.live_bytes(), 0);
        }
    }

    #[test]
    fn repeated_string_allocation_keeps_live_aliases_without_cumulative_exhaustion() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        vm.set_memory_limit(vm.allocated_bytes() + 16).unwrap();
        let arguments = [Value::string(Arc::from("abc")), Value::number(0.), Value::number(3.)];
        let mut first = Value::null();
        assert_eq!(vm.execute_host_call(LANA_HOST_STRING_SLICE, &arguments, &mut first), LanaError::Ok);
        let alias = first.as_string();
        drop(first);
        for _ in 0..1000 {
            let mut next = Value::null();
            assert_eq!(vm.execute_host_call(LANA_HOST_STRING_SLICE, &arguments, &mut next), LanaError::Ok);
        }
        let heap = vm.heap();
        drop(vm);
        heap.collect_strings();
        assert_eq!(heap.live_bytes(), 4);
        assert_eq!(&*alias, "abc");
        drop(alias);
        heap.collect_strings();
        assert_eq!(heap.live_bytes(), 0);
    }

    #[cfg(any(not(feature = "net-tls"), target_arch = "wasm32"))]
    #[test]
    fn https_without_tls_never_falls_back_to_plaintext() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let mut out = Value::null();
        assert_eq!(vm.net_http_request("GET", "https://127.0.0.1:9/", None, 1., true, &[], &mut out), LanaError::Ok);
        assert_eq!(out.print(), "[false, tls]");
    }

    #[test]
    fn json_and_unescape_preserve_literal_unicode() {
        let (error, result) = run_chunk(
            "LOAD_STRING R0 636166c3a9f09f8c8a5c6e\nHOST_CALL string_unescape R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "café🌊\n");
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let mut pos = 0;
        assert_eq!(vm.json_value("\"café🌊\"".as_bytes(), &mut pos, 0).unwrap().as_string().as_ref(), "café🌊");
    }

    #[test]
    fn array_growth_failure_preserves_existing_items() {
        let chunk = assembler::assemble(
            ".function main 0 4\nLOAD_CONST R0 1\nARRAY_NEW R1 R0 1\nMOVE R2 R0\nHOST_CALL array_push R1 2 R3\nRETURN R3\n",
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        for instruction in &chunk.code[..3] {
            assert_eq!(vm.execute(instruction), LanaError::Ok);
        }
        vm.set_memory_limit(vm.allocated_bytes()).unwrap();
        assert_eq!(vm.execute(&chunk.code[3]), LanaError::Oom);
        let ValueKind::Array(array) = &vm.frames[0].registers[1].kind else { panic!("array expected") };
        assert_eq!(array.lock().unwrap().items.len(), 1);
        assert!(matches!(vm.frames[0].registers[3].kind, ValueKind::Null));
    }

    #[test]
    fn fork_argument_payload_is_charged_to_child() {
        for (body, expected) in [
            ("LOAD_CONST R1 7\nRETURN R1\n", LanaError::Ok),
            ("LOAD_CONST R1 6000\nHOST_CALL array_new R1 1 R2\nRETURN R2\n", LanaError::Oom),
        ] {
            let chunk = assembler::assemble(&format!(
                ".function main 0 8\nLOAD_CONST R0 6000\nHOST_CALL array_new R0 1 R1\nFORK worker R1 1 R2\nJOIN R2 R3\nRETURN R3\n.function worker 1 4\n{body}"
            )).unwrap();
            let mut vm = Vm::new(&chunk);
            vm.set_memory_limit(vm.allocated_bytes() + 6000 * (std::mem::size_of::<Value>() + 10 * std::mem::size_of::<usize>()) + 32768).unwrap();
            assert_eq!(vm.run(), expected, "{:?}; live={} limit={}", vm.error(), vm.allocated_bytes(), vm.memory_limit);
        }
    }

    #[test]
    fn repeated_array_allocation_tracks_live_owners_not_total_work() {
        let chunk = assembler::assemble(
            "LOAD_CONST R0 1000\nLOAD_CONST R1 16\nLOAD_CONST R2 1\nloop:\nHOST_CALL array_new R1 1 R3\nBINARY R0 sub R2 R0\nLOAD_CONST R4 0\nCOMPARE R0 > R4 R5\nJUMP_IF_TRUE R5 loop\nRETURN R3\n",
        ).unwrap();
        let probe_heap = Heap::new(4096);
        let probe = Array::new(&probe_heap, 16).unwrap();
        let payload_bytes = probe_heap.live_bytes();
        drop(probe);
        let mut probe_vm = Vm::new(&chunk);
        let first = probe_vm.array_value(vec![Value::null(); 16]).unwrap();
        probe_vm.result = probe_vm.array_value(vec![Value::null(); 16]).unwrap();
        let probe_root = probe_vm.result().unwrap();
        let budget = probe_vm.heap.peak_bytes() + payload_bytes;
        drop((first, probe_root, probe_vm));
        let mut vm = Vm::new(&chunk);
        vm.set_memory_limit(budget).unwrap();
        assert_eq!(vm.run(), LanaError::Ok);
        let heap = vm.heap();
        let external_root = vm.result().unwrap();
        drop(vm);
        assert!(heap.live_bytes() <= budget);
        drop(external_root);
        assert_eq!(heap.live_bytes(), 0);
        assert!(heap.peak_bytes() <= budget);
    }

    #[test]
    fn set_allocation_is_fallible_and_releases_its_payload() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let baseline = vm.allocated_bytes();
        assert_eq!(vm.set_memory_limit(1), Err(LanaError::Oom));
        vm.set_memory_limit(baseline).unwrap();
        assert!(matches!(vm.set_host_call(LANA_HOST_SET_NEW, &[]), Err(LanaError::Oom)));
        assert_eq!(vm.allocated_bytes(), baseline);
        vm.set_memory_limit(256 * 1024).unwrap();
        for _ in 0..1000 {
            let empty = vm.set_host_call(LANA_HOST_SET_NEW, &[]).unwrap();
            let single = vm.set_host_call(LANA_HOST_SET_ADD, &[empty, Value::number(7.)]).unwrap();
            let same = vm.set_host_call(LANA_HOST_SET_UNION, &[single.clone(), single.clone()]).unwrap();
            assert_eq!(same.print(), "set{7}");
            drop((single, same));
            vm.collect_classes().unwrap();
            assert_eq!(vm.allocated_bytes(), baseline);
        }
        assert!(vm.heap.peak_bytes() <= 256 * 1024);
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
    fn dataset_filter_materialize() {
        // Source rows are indices 0..5; keep those > 1 -> [2, 3, 4].
        let (error, result) = run_chunk(
            ".function main 0 8\nLOAD_CONST R0 5\nLAZY R1 gen R0\nHOST_CALL dataset R1 1 R2\nLOAD_FUNCTION R3 is_gt1\nHOST_CALL dataset_filter R2 2 R4\nHOST_CALL dataset_materialize R4 1 R5\nRETURN R5\n.function gen 1 3\nRETURN R0\n.function is_gt1 1 3\nLOAD_CONST R1 1\nCOMPARE R0 > R1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[2, 3, 4]");
    }

    #[test]
    fn dataset_map_materialize() {
        // Source rows are indices 0..4; map each to index*10 -> [0, 10, 20, 30].
        let (error, result) = run_chunk(
            ".function main 0 8\nLOAD_CONST R0 4\nLAZY R1 gen R0\nHOST_CALL dataset R1 1 R2\nLOAD_FUNCTION R3 times10\nHOST_CALL dataset_map R2 2 R4\nHOST_CALL dataset_materialize R4 1 R5\nRETURN R5\n.function gen 1 3\nRETURN R0\n.function times10 1 3\nLOAD_CONST R1 10\nBINARY R0 mul R1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[0, 10, 20, 30]");
    }

    #[test]
    fn dataset_filter_rejects_uncertain_and_non_boolean_results() {
        for (predicate, expected) in [
            ("LOAD_CONST R1 1\nRETURN R1\n", LanaError::Type),
            ("LOAD_CONST R1 true\nLOAD_CONST R2 false\nARRAY_NEW R3 R1 2\nPOSSIBILITY_BUILD R3 R4\nRETURN R4\n", LanaError::UnresolvedValue),
        ] {
            let program = format!(".function main 0 8\nLOAD_CONST R0 2\nLAZY R1 gen R0\nHOST_CALL dataset R1 1 R2\nLOAD_FUNCTION R3 predicate\nHOST_CALL dataset_filter R2 2 R4\nHOST_CALL dataset_materialize R4 1 R5\nRETURN R5\n.function gen 1 3\nRETURN R0\n.function predicate 1 5\n{predicate}");
            assert_eq!(run_chunk(&program).0, expected);
        }
    }

    #[test]
    fn dataset_keys_are_checked_before_comparison_and_self_join_works() {
        let chunk = Chunk::new(lana_bytecode::opcode::LABC_VERSION, 0);
        let mut vm = Vm::new(&chunk);
        let uncertain = Value::possibility(vm.possibility_build(&[Value::number(1.0), Value::number(2.0)]).unwrap());
        let nested = vm.array_value(vec![uncertain.clone()]).unwrap();
        let make_source = |vm: &mut Vm, key: Option<Value>| {
            let mut rows = Vec::new();
            if let Some(key) = key {
                let mut row = Map::new(&vm.heap, 1).unwrap();
                row.set(Arc::from("key"), key, false).unwrap();
                rows.push(Value::map(Arc::new(Mutex::new(row))));
            }
            let rows = vm.array_value(rows).unwrap();
            let mut source = Value::null();
            assert_eq!(vm.host_dataset(&[rows], &mut source), LanaError::Ok);
            source
        };
        let empty = make_source(&mut vm, None);
        for key in [uncertain, nested, Value::distribution(0.5, 0.5)] {
            let source = make_source(&mut vm, Some(key));
            for operation in [DatasetOp::Sort, DatasetOp::GroupBy, DatasetOp::Join] {
                for reverse in [false, true] {
                    let mut plan = Value::null();
                    let key = Value::string(Arc::from("key"));
                    let error = match operation {
                        DatasetOp::Sort => vm.host_dataset_sort(&[source.clone(), key], &mut plan),
                        DatasetOp::GroupBy => vm.host_dataset_group_by(&[source.clone(), key], &mut plan),
                        _ if reverse => vm.host_dataset_join(&[empty.clone(), source.clone(), key], &mut plan),
                        _ => vm.host_dataset_join(&[source.clone(), empty.clone(), key], &mut plan),
                    };
                    assert_eq!(error, LanaError::Ok);
                    let mut output = Value::number(99.0);
                    assert_eq!(vm.host_dataset_materialize(&[plan], &mut output), LanaError::UnresolvedValue);
                    assert_eq!(output.as_number(), 99.0);
                }
            }
        }
        let source = make_source(&mut vm, Some(Value::number(1.0)));
        let mut joined = Value::null();
        assert_eq!(vm.host_dataset_join(&[source.clone(), source, Value::string(Arc::from("key"))], &mut joined), LanaError::Ok);
        let mut output = Value::null();
        assert_eq!(vm.host_dataset_materialize(&[joined], &mut output), LanaError::Ok);
        assert_eq!(output.print(), "[{\"key\": 1}]");
    }

    #[test]
    fn dataset_aggregate_rejects_bad_descriptors_and_nonfinite_results() {
        let chunk = Chunk::new(lana_bytecode::opcode::LABC_VERSION, 0);
        let mut vm = Vm::new(&chunk);
        let mut rows = Vec::new();
        for number in [f64::MAX, f64::MAX] {
            let mut row = Map::new(&vm.heap, 2).unwrap();
            row.set(Arc::from("key"), Value::number(1.0), false).unwrap();
            row.set(Arc::from("value"), Value::number(number), false).unwrap();
            rows.push(Value::map(Arc::new(Mutex::new(row))));
        }
        let source_rows = vm.array_value(rows).unwrap();
        let mut source = Value::null();
        assert_eq!(vm.host_dataset(&[source_rows], &mut source), LanaError::Ok);
        let mut grouped = Value::null();
        assert_eq!(vm.host_dataset_group_by(&[source, Value::string(Arc::from("key"))], &mut grouped), LanaError::Ok);
        for (descriptor, expected) in [
            (vec![Value::string(Arc::from("sum"))], LanaError::Type),
            (vec![Value::string(Arc::from("count")), Value::string(Arc::from("value"))], LanaError::Type),
            (vec![Value::string(Arc::from("sum")), Value::string(Arc::from("value"))], LanaError::Type),
        ] {
            let descriptor = vm.array_value(descriptor).unwrap();
            let mut plan = Value::null();
            assert_eq!(vm.host_dataset_aggregate(&[grouped.clone(), descriptor], &mut plan), LanaError::Ok);
            let mut output = Value::number(99.0);
            assert_eq!(vm.host_dataset_materialize(&[plan], &mut output), expected);
            assert_eq!(output.as_number(), 99.0);
        }
    }

    #[test]
    fn dataset_explain_reports_plan() {
        // explain(filter(source)) -> {op: "filter", source: {op: "source", bound: 5}}.
        let (error, result) = run_chunk(
            ".function main 0 8\nLOAD_CONST R0 5\nLAZY R1 gen R0\nHOST_CALL dataset R1 1 R2\nLOAD_FUNCTION R3 is_gt1\nHOST_CALL dataset_filter R2 2 R4\nHOST_CALL dataset_explain R4 1 R5\nRETURN R5\n.function gen 1 3\nRETURN R0\n.function is_gt1 1 3\nLOAD_CONST R1 1\nCOMPARE R0 > R1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "{\"op\": filter, \"source\": {\"op\": source, \"bound\": 5}, \"function\": function(2)}");
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
    fn generator_yield_returns_result_ok() {
        let (error, result) = run_chunk(
            ".version 3\n.function main 0 8\nGENERATOR gen R0 0 R1\nNEXT R1 R2\nRETURN R2\n.function gen 0 2\nLOAD_CONST R1 10\nYIELD R0 R1\nLOAD_CONST R1 20\nYIELD R0 R1\nRETURN R0\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[true, 10]");
    }

    #[test]
    fn generator_exhaustion_returns_result_error() {
        let (error, result) = run_chunk(
            ".version 3\n.function main 0 8\nGENERATOR gen R0 0 R1\nNEXT R1 R2\nNEXT R1 R3\nNEXT R1 R4\nRETURN R4\n.function gen 0 2\nLOAD_CONST R1 10\nYIELD R0 R1\nLOAD_CONST R1 20\nYIELD R0 R1\nRETURN R0\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[false, exhausted]");
    }

    #[test]
    fn next_on_non_generator_returns_type_error() {
        let (error, _) = run_chunk(
            ".version 3\nLOAD_CONST R1 0\nNEXT R1 R2\nHALT\n",
        );
        assert_eq!(error, LanaError::Type);
    }

    #[test]
    fn run_async_runs_cold_future_to_completion() {
        let (error, result) = run_chunk(
            ".version 4\n.function main 0 8\nASYNC foo R0 0 R1\nRUN_ASYNC R1 R2\nRETURN R2\n.function foo 0 2\nLOAD_CONST R1 42\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "42");
    }

    #[test]
    fn await_suspends_and_resumes_async_frame() {
        let (error, result) = run_chunk(
            ".version 4\n.function main 0 8\nASYNC foo R0 0 R1\nRUN_ASYNC R1 R2\nRETURN R2\n.function foo 0 4\nASYNC bar R0 0 R1\nAWAIT R1 R2\nRETURN R2\n.function bar 0 2\nLOAD_CONST R1 7\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "7");
    }

    #[test]
    fn run_async_on_non_future_returns_type_error() {
        let (error, _) = run_chunk(
            ".version 4\nLOAD_CONST R1 0\nRUN_ASYNC R1 R2\nHALT\n",
        );
        assert_eq!(error, LanaError::Type);
    }

    #[test]
    fn future_all_waits_for_all_inputs_in_order() {
        let (error, result) = run_chunk(
            ".version 4\n.function main 0 8\nASYNC foo R0 0 R1\nASYNC bar R0 0 R2\nARRAY_NEW R3 R1 2\nHOST_CALL future_all R3 1 R4\nRUN_ASYNC R4 R5\nRETURN R5\n.function foo 0 2\nLOAD_CONST R1 10\nRETURN R1\n.function bar 0 2\nLOAD_CONST R1 20\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[10, 20]");
    }

    #[test]
    fn future_race_returns_first_completed_input() {
        let (error, result) = run_chunk(
            ".version 4\n.function main 0 8\nASYNC foo R0 0 R1\nASYNC bar R0 0 R2\nARRAY_NEW R3 R1 2\nHOST_CALL future_race R3 1 R4\nRUN_ASYNC R4 R5\nRETURN R5\n.function foo 0 2\nLOAD_CONST R1 10\nRETURN R1\n.function bar 0 2\nLOAD_CONST R1 20\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "10");
    }

    #[test]
    fn sleep_zero_completes_immediately_with_null() {
        let (error, result) = run_chunk(
            ".version 4\n.function main 0 8\nLOAD_CONST R0 0\nHOST_CALL sleep R0 1 R1\nRUN_ASYNC R1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "null");
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
    fn set_new_is_empty() {
        let (error, result) = run_chunk("HOST_CALL set_new R0 0 R1\nRETURN R1\n");
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "set{}");
    }

    #[test]
    fn set_add_and_contains() {
        let (error, result) = run_chunk(
            "HOST_CALL set_new R0 0 R1\nLOAD_CONST R2 a\nHOST_CALL set_add R1 2 R3\nLOAD_CONST R4 b\nHOST_CALL set_add R3 2 R5\nLOAD_CONST R6 a\nHOST_CALL set_add R5 2 R7\nRETURN R7\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "set{a, b}");
    }

    #[test]
    fn set_contains_true_and_false() {
        let (error, result) = run_chunk(
            "HOST_CALL set_new R0 0 R1\nLOAD_CONST R2 a\nHOST_CALL set_add R1 2 R3\nLOAD_CONST R4 a\nHOST_CALL set_contains R3 2 R5\nLOAD_CONST R4 z\nHOST_CALL set_contains R3 2 R6\nRETURN R6\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "false");
    }

    #[test]
    fn set_union_intersect_difference() {
        let (error, result) = run_chunk(
            "HOST_CALL set_new R0 0 R1\nLOAD_CONST R2 a\nHOST_CALL set_add R1 2 R3\nLOAD_CONST R4 b\nHOST_CALL set_add R3 2 R5\nHOST_CALL set_new R0 0 R6\nLOAD_CONST R7 b\nHOST_CALL set_add R6 2 R8\nLOAD_CONST R9 c\nHOST_CALL set_add R8 2 R10\nMOVE R6 R10\nHOST_CALL set_union R5 2 R11\nHOST_CALL set_intersect R5 2 R12\nHOST_CALL set_difference R5 2 R13\nRETURN R13\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "set{a}");
    }

    #[test]
    fn set_add_state_is_type_error() {
        let (error, _) = run_chunk(
            "HOST_CALL set_new R0 0 R1\nSTATE_NEW R2 0.5 0.0 0.0\nHOST_CALL set_add R1 2 R3\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Type);
    }

    #[test]
    fn getenv_returns_value_and_empty_for_unset() {
        std::env::set_var("LANA_TEST_GETENV", "hello");
        std::env::remove_var("LANA_TEST_GETENV_UNSET");
        let (error, result) = run_chunk(
            "LOAD_CONST R0 LANA_TEST_GETENV\nHOST_CALL getenv R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "hello");
        let (error, result) = run_chunk(
            "LOAD_CONST R0 LANA_TEST_GETENV_UNSET\nHOST_CALL getenv R0 1 R1\nRETURN R1\n",
        );
        std::env::remove_var("LANA_TEST_GETENV");
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "");
    }

    #[test]
    fn floor_rounds_toward_negative_infinity() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 3.7\nHOST_CALL floor R0 1 R1\nLOAD_CONST R2 -1.2\nHOST_CALL floor R2 1 R3\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "-2");
    }

    #[test]
    fn random_seed_returns_null_and_random_is_in_range() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 42\nHOST_CALL random_seed R0 1 R1\nHOST_CALL random R0 0 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        let value: f64 = result.parse().unwrap();
        assert!((0.0..1.0).contains(&value));
    }

    #[test]
    fn string_to_number_returns_result_and_type_of_names() {
        let (error, result) = run_chunk(
            "LOAD_STRING R0 3432\nHOST_CALL string_to_number R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[true, 42]");
        let (error, result) = run_chunk(
            "LOAD_CONST R0 abc\nHOST_CALL string_to_number R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[false, invalid number]");
        let (error, result) = run_chunk(
            "LOAD_CONST R0 42\nHOST_CALL type_of R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "number");
        let (error, result) = run_chunk(
            "LOAD_CONST R0 x\nHOST_CALL type_of R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "string");
    }

    #[test]
    fn format_and_format_number() {
        // format("value: {} ({})", 3.14, "ok") -> "value: 3.1400000000000001 (ok)".
        let (error, result) = run_chunk(
            "LOAD_STRING R0 76616c75653a207b7d20287b7d29\nLOAD_CONST R1 3.14\nLOAD_STRING R2 6f6b\nHOST_CALL format R0 3 R3\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "value: 3.1400000000000001 (ok)");

        // format_number(3.14159) -> "3.1415899999999999" (round-trip).
        let (error, result) = run_chunk(
            "LOAD_CONST R0 3.14159\nHOST_CALL format_number R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "3.1415899999999999");

        // format_number(3.14159, 2) -> "3.14".
        let (error, result) = run_chunk(
            "LOAD_CONST R0 3.14159\nLOAD_CONST R1 2\nHOST_CALL format_number R0 2 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "3.14");

        // format_number(3.14159, 0) -> "3".
        let (error, result) = run_chunk(
            "LOAD_CONST R0 3.14159\nLOAD_CONST R1 0\nHOST_CALL format_number R0 2 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "3");

        // format("{} {} {}", true, null, 5) -> "true null 5".
        let (error, result) = run_chunk(
            "LOAD_STRING R0 7b7d207b7d207b7d\nLOAD_CONST R1 true\nLOAD_CONST R2 null\nLOAD_CONST R3 5\nHOST_CALL format R0 4 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "true null 5");
    }

    #[test]
    fn unicode_code_points_and_case_mapping() {
        // char_length("hello") -> 5.
        let (error, result) = run_chunk(
            "LOAD_STRING R0 68656c6c6f\nHOST_CALL char_length R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "5");

        // char_length("héllo") -> 5 (é is one code point).
        let (error, result) = run_chunk(
            "LOAD_STRING R0 68c3a96c6c6f\nHOST_CALL char_length R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "5");

        // char_length("😀") -> 1.
        let (error, result) = run_chunk(
            "LOAD_STRING R0 f09f9880\nHOST_CALL char_length R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "1");

        // string_codepoint_slice("héllo", 0, 2) -> "hé".
        let (error, result) = run_chunk(
            "LOAD_STRING R0 68c3a96c6c6f\nLOAD_CONST R1 0\nLOAD_CONST R2 2\nHOST_CALL string_codepoint_slice R0 3 R3\nRETURN R3\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "hé");

        // to_upper("hello") -> "HELLO".
        let (error, result) = run_chunk(
            "LOAD_STRING R0 68656c6c6f\nHOST_CALL to_upper R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "HELLO");

        // to_lower("HELLO") -> "hello".
        let (error, result) = run_chunk(
            "LOAD_STRING R0 48454c4c4f\nHOST_CALL to_lower R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "hello");

        // to_upper("Straße") -> "STRAßE" (ß has no simple uppercase mapping).
        let (error, result) = run_chunk(
            "LOAD_STRING R0 53747261c39f65\nHOST_CALL to_upper R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "STRAßE");

        // to_lower("İ") -> "i".
        let (error, result) = run_chunk(
            "LOAD_STRING R0 c4b0\nHOST_CALL to_lower R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "i");

        // to_upper("ı") -> "I".
        let (error, result) = run_chunk(
            "LOAD_STRING R0 c4b1\nHOST_CALL to_upper R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "I");

        // to_upper("ς") -> "Σ".
        let (error, result) = run_chunk(
            "LOAD_STRING R0 cf82\nHOST_CALL to_upper R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "Σ");

        // to_lower("ẞ") -> "ß".
        let (error, result) = run_chunk(
            "LOAD_STRING R0 e1ba9e\nHOST_CALL to_lower R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "ß");
    }

    #[test]
    fn regex_compile_match_search_replace() {
        // regex_compile("a+") -> [true, regex(insts=3)].
        let (error, result) = run_chunk(
            "LOAD_STRING R0 612b\nHOST_CALL regex_compile R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[true, regex(insts=3)]");

        // regex_match(re, "aaa") -> [true, {"start": 0, "end": 3, "text": "aaa"}].
        let (error, result) = run_chunk(
            "LOAD_STRING R0 612b\nHOST_CALL regex_compile R0 1 R1\nLOAD_CONST R2 1\nARRAY_GET R1 R2 R3\nLOAD_STRING R4 616161\nHOST_CALL regex_match R3 2 R5\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[true, {\"start\": 0, \"end\": 3, \"text\": aaa}]");

        // regex_search(re, "xxaaayy") -> [true, {"start": 2, "end": 5, "text": "aaa"}].
        let (error, result) = run_chunk(
            "LOAD_STRING R0 612b\nHOST_CALL regex_compile R0 1 R1\nLOAD_CONST R2 1\nARRAY_GET R1 R2 R3\nLOAD_STRING R4 78786161617979\nHOST_CALL regex_search R3 2 R5\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[true, {\"start\": 2, \"end\": 5, \"text\": aaa}]");

        // regex_replace(re, "xxaaayy", "-") -> "xx-yy".
        let (error, result) = run_chunk(
            "LOAD_STRING R0 612b\nHOST_CALL regex_compile R0 1 R1\nLOAD_CONST R2 1\nARRAY_GET R1 R2 R3\nLOAD_STRING R4 78786161617979\nLOAD_STRING R5 2d\nHOST_CALL regex_replace R3 3 R6\nRETURN R6\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "xx-yy");

        // regex_compile("(") -> [false, "unterminated group"].
        let (error, result) = run_chunk(
            "LOAD_STRING R0 28\nHOST_CALL regex_compile R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[false, unterminated group]");
    }

    #[test]
    fn json_parse_returns_result_and_preserves_large_integers() {
        // Valid JSON -> [true, value].
        let (error, result) = run_chunk(
            "LOAD_STRING R0 7b2261223a317d\nHOST_CALL json_parse R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[true, {\"a\": 1}]");
        // Invalid JSON -> [false, "invalid JSON at byte N"].
        let (error, result) = run_chunk(
            "LOAD_STRING R0 5b315d2078\nHOST_CALL json_parse R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[false, invalid JSON at byte 4]");
        // Large integer beyond 2^53 is preserved as a string.
        let (error, result) = run_chunk(
            "LOAD_STRING R0 39303037313939323534373430393933\nHOST_CALL json_parse R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[true, 9007199254740993]");
    }

    #[test]
    fn json_stringify_sorts_keys_and_duplicate_keys_last_wins() {
        // Duplicate keys: last occurrence wins.
        let (error, result) = run_chunk(
            "LOAD_STRING R0 7b2261223a312c2261223a327d\nHOST_CALL json_parse R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[true, {\"a\": 2}]");
        // Stringify sorts object keys.
        let (error, result) = run_chunk(
            "LOAD_STRING R0 7b2262223a322c2261223a317d\nHOST_CALL json_parse R0 1 R1\nLOAD_CONST R2 1\nARRAY_GET R1 R2 R3\nHOST_CALL json_stringify R3 1 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "{\"a\":1,\"b\":2}");
    }

    #[test]
    fn json_parse_roots_information_with_text_identity() {
        // LIP-023 §4: the parsed value is an Information value whose derivation
        // records the source-text identity (SHA-256).
        let (error, result) = run_chunk(
            "LOAD_STRING R0 7b2261223a317d\nHOST_CALL json_parse R0 1 R1\nLOAD_CONST R2 1\nARRAY_GET R1 R2 R3\nDERIVATION R3 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert!(
            result.contains("\"operation\": json_parse"),
            "expected json_parse operation, got {result}"
        );
        assert!(
            result.contains("\"label\": 015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"),
            "expected source-text identity, got {result}"
        );
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
    fn state_dist_walks_are_budget_bounded() {
        // append(shared, shared) references the same subtree twice, so a naive
        // walk visits exponentially many nodes. Both the sample and expected-
        // probability walks must charge each visited node against the sampling
        // budget and return BudgetExhausted rather than running unbounded.
        let chunk = assembler::assemble("HALT\n").unwrap();
        let dirac = |p: f64| {
            Arc::new(StateDist {
                kind: StateDistKind::Dirac(StateValue {
                    state: State { p, d_re: 0.0, d_im: 0.0 },
                    indexes: Default::default(),
                }),
            })
        };
        let shared = Arc::new(StateDist {
            kind: StateDistKind::Append {
                left: DistOperand::Node(dirac(0.2)),
                right: DistOperand::Node(dirac(0.3)),
                has_cached_parameters: false,
                p: 0.0,
                m_re: 0.0,
                m_im: 0.0,
                sigma: 0.0,
            },
        });
        let outer = Arc::new(StateDist {
            kind: StateDistKind::Append {
                left: DistOperand::Node(shared.clone()),
                right: DistOperand::Node(shared),
                has_cached_parameters: false,
                p: 0.0,
                m_re: 0.0,
                m_im: 0.0,
                sigma: 0.0,
            },
        });

        let mut vm = Vm::new(&chunk);
        vm.set_instruction_limit(10);
        assert_eq!(vm.state_dist_sample(&outer), Err(LanaError::BudgetExhausted));

        let mut vm = Vm::new(&chunk);
        vm.set_instruction_limit(10);
        assert_eq!(
            vm.state_dist_expected_probability(&outer),
            Err(LanaError::BudgetExhausted)
        );
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
    fn joint_condition_map_refines_named_evidence() {
        let (error, result) = run_chunk(
            ".version 5\nLOAD_CONST R0 1\nLOAD_CONST R1 2\nJOINT_BUILD R2 R0 2 independent:a;b\nLOAD_STRING R3 61\nLOAD_CONST R4 1\nHOST_CALL map_new R3 2 R5\nJOINT_CONDITION_MAP R2 R6 R5\nRESOLVE R6 R7\nRETURN R7\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[1, 2]");
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
    fn nested_definite_path_join_does_not_consume_outer_split() {
        for (condition, expected) in [
            ("LOAD_CONST R5 true\n", 10),
            ("LOAD_CONST R5 false\n", 30),
            ("LOAD_CONST R6 true\nARRAY_NEW R7 R6 1\nPOSSIBILITY_BUILD R7 R5\n", 10),
            ("LOAD_CONST R6 false\nARRAY_NEW R7 R6 1\nPOSSIBILITY_BUILD R7 R5\n", 30),
        ] {
            let source = format!("LOAD_CONST R0 true\nLOAD_CONST R1 false\nARRAY_NEW R2 R0 2\nPOSSIBILITY_BUILD R2 R3\nPATH_SPLIT R3 outer_else\n{condition}PATH_SPLIT R5 inner_else\nLOAD_CONST R4 10\nJUMP inner_join\ninner_else:\nLOAD_CONST R4 30\ninner_join:\nPATH_JOIN\nJUMP outer_join\nouter_else:\nLOAD_CONST R4 20\nouter_join:\nPATH_JOIN\nRETURN R4\n");
            let (error, result) = run_chunk(&source);
            assert_eq!(error, LanaError::Ok);
            assert_eq!(result, format!("paths{{true => {expected}, false => 20}}"));
        }
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
    fn info_sample_possibility_is_legacy_only() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 1\nLOAD_CONST R1 2\nARRAY_NEW R2 R0 2\nPOSSIBILITY_BUILD R2 R3\nINFO_SAMPLE R3 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert!(result == "1" || result == "2", "expected 1 or 2, got {result}");

        let (error, _) = run_chunk(
            ".version 5\nLOAD_CONST R0 1\nLOAD_CONST R1 2\nARRAY_NEW R2 R0 2\nPOSSIBILITY_BUILD R2 R3\nINFO_SAMPLE R3 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::UnsupportedOperation);
    }

    #[test]
    fn distribution_build_samples_weighted_support() {
        let (error, result) = run_chunk(
            ".version 5\nLOAD_CONST R0 1\nLOAD_CONST R1 0.25\nARRAY_NEW R2 R0 2\nLOAD_CONST R3 2\nLOAD_CONST R4 0.75\nARRAY_NEW R5 R3 2\nMOVE R6 R2\nMOVE R7 R5\nARRAY_NEW R8 R6 2\nDISTRIBUTION_BUILD R8 R9\nINFO_SAMPLE R9 R10\nRETURN R10\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert!(result == "1" || result == "2", "expected supported value, got {result}");

        let (error, _) = run_chunk(
            ".version 5\nLOAD_CONST R0 1\nLOAD_CONST R1 0.2\nARRAY_NEW R2 R0 2\nARRAY_NEW R3 R2 1\nDISTRIBUTION_BUILD R3 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::InvalidDistribution);
    }

    #[test]
    fn v5_core_operation_matrix() {
        // Exercise actual bytecode dispatch, including rejected cells.
        let forms = [
            ("definite", "LOAD_CONST R10 1\n", false, false, true),
            ("possibility", "LOAD_CONST R0 1\nLOAD_CONST R1 2\nARRAY_NEW R2 R0 2\nPOSSIBILITY_BUILD R2 R10\n", false, false, false),
            ("distribution", "LOAD_CONST R0 1\nLOAD_CONST R1 0.5\nARRAY_NEW R2 R0 2\nLOAD_CONST R3 2\nLOAD_CONST R4 0.5\nARRAY_NEW R5 R3 2\nMOVE R6 R2\nMOVE R7 R5\nARRAY_NEW R8 R6 2\nDISTRIBUTION_BUILD R8 R10\n", false, true, false),
            ("joint", "LOAD_CONST R0 1\nJOINT_BUILD R10 R0 1 independent:x\n", true, true, true),
            ("paths", "LOAD_CONST R0 true\nLOAD_CONST R1 false\nARRAY_NEW R2 R0 2\nPOSSIBILITY_BUILD R2 R3\nPATH_SPLIT R3 other\nLOAD_CONST R4 1\nJUMP joined\nother:\nLOAD_CONST R4 2\njoined:\nPATH_JOIN\nMOVE R10 R4\n", false, true, false),
        ];
        for (name, prefix, joint, sample, resolve) in forms {
            for (operation, instruction, expected) in [
                ("project", "JOINT_PROJECT R10 R11 x\n", if joint { LanaError::Ok } else { LanaError::Type }),
                ("sample", "INFO_SAMPLE R10 R11\n", if sample { LanaError::Ok } else if name == "possibility" { LanaError::UnsupportedOperation } else { LanaError::Type }),
                ("resolve", "RESOLVE R10 R11\n", if resolve { LanaError::Ok } else { LanaError::UnresolvedValue }),
                ("inspect", "HOST_CALL information_inspect R10 1 R11\n", LanaError::Ok),
            ] {
                let (actual, _) = run_chunk(&format!(".version 5\n{prefix}{instruction}RETURN R11\n"));
                assert_eq!(actual, expected, "{name}/{operation}");
            }
            let evidence = if joint {
                "LOAD_STRING R3 78\nLOAD_CONST R4 1\nHOST_CALL map_new R3 2 R12\n"
            } else {
                "LOAD_CONST R12 1\n"
            };
            for (operation, instruction) in [
                ("condition", "JOINT_CONDITION_MAP R10 R11 R12\n"),
                ("observe", "OBSERVE_MAP R10 R11 R12\n"),
            ] {
                let (actual, _) = run_chunk(&format!(".version 5\n{prefix}{evidence}{instruction}RETURN R11\n"));
                assert_eq!(actual, if name == "paths" { LanaError::UnsupportedOperation } else { LanaError::Ok }, "{name}/{operation}");
            }
        }
    }

    #[test]
    fn finite_refinement_rejects_impossible_evidence() {
        let (error, _) = run_chunk(
            ".version 5\nLOAD_CONST R0 1\nLOAD_CONST R1 2\nARRAY_NEW R2 R0 2\nPOSSIBILITY_BUILD R2 R3\nLOAD_CONST R4 3\nJOINT_CONDITION_MAP R3 R5 R4\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::InvalidConditioning);
        let (error, _) = run_chunk(
            ".version 5\nLOAD_CONST R0 1\nLOAD_CONST R1 2\nARRAY_NEW R2 R0 2\nPOSSIBILITY_BUILD R2 R3\nLOAD_CONST R4 3\nOBSERVE_MAP R3 R5 R4\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::InvalidConditioning);
    }

    #[test]
    fn possibility_build_pair_rows_remains_unweighted() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 1\nLOAD_CONST R1 0.25\nARRAY_NEW R2 R0 2\nLOAD_CONST R3 2\nLOAD_CONST R4 0.75\nARRAY_NEW R5 R3 2\nMOVE R6 R2\nMOVE R7 R5\nARRAY_NEW R8 R6 2\nPOSSIBILITY_BUILD R8 R9\nRETURN R9\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert!(result.starts_with("possibility{"), "expected possibility, got {result}");
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
        // The child cannot finish normally before CANCEL wins the race.
        let (error, _) = run_chunk(
            ".function main 0 8\nFORK worker R1 0 R0\nCANCEL R0\nJOIN R0 R1\nHALT\n.function worker 0 4\nloop:\nJUMP loop\n",
        );
        assert_eq!(error, LanaError::Cancelled);
    }

    #[test]
    fn cancellation_after_execution_has_started() {
        use std::time::Duration;
        let chunk = assembler::assemble(
            "LOAD_STRING R0 78\nHOST_CALL store_open R0 1 R1\nloop:\nJUMP loop\n",
        ).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let flag = cancelled.clone();
            let worker = scope.spawn(move || {
                let mut vm = Vm::new(&chunk);
                vm.cancelled = flag;
                vm.set_host_call_extension(Box::new(move |_, _, _, _| {
                    ready_tx.send(()).unwrap();
                    resume_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    LanaError::Ok
                }));
                vm.run()
            });
            ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            cancelled.store(true, Ordering::Release);
            resume_tx.send(()).unwrap();
            assert_eq!(worker.join().unwrap(), LanaError::Cancelled);
        });
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
            (error, vm.result().unwrap().print())
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

    #[test]
    fn capability_grant_and_revoke() {
        // grant "use" and "admin" from an admin token, then revoke both.
        let (error, result) = run_chunk(
            ".function main 0 16\n\
             LOAD_CONST R0 42\n\
             HOST_CALL shared_information R0 1 R1\n\
             LOAD_CONST R2 use\n\
             HOST_CALL grant R1 2 R3\n\
             LOAD_CONST R2 admin\n\
             HOST_CALL grant R1 2 R4\n\
             HOST_CALL shared_identity R3 1 R5\n\
             HOST_CALL revoke R3 1 R6\n\
             HOST_CALL revoke R4 1 R7\n\
             RETURN R5\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert!(result.parse::<f64>().is_ok(), "expected identity, got {result}");
    }

    #[test]
    fn revoked_capability_denies_snapshot() {
        // After revoke, a read capability no longer authorizes a snapshot.
        let (error, _) = run_chunk(
            ".function main 0 16\n\
             LOAD_CONST R0 42\n\
             HOST_CALL shared_information R0 1 R1\n\
             LOAD_CONST R2 use\n\
             HOST_CALL grant R1 2 R3\n\
             HOST_CALL revoke R3 1 R4\n\
             HOST_CALL shared_snapshot R3 1 R5\n\
             RETURN R5\n",
        );
        assert_eq!(error, LanaError::Capability);
    }

    // LIP-005: linear algebra on STATEs. Each test mirrors a differential
    // fixture in tests/conformance/differential/hostcalls/lip005_*.lasm.

    #[test]
    fn lip005_density_operator_from_state() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.4 0.2 0.0\nHOST_CALL density_operator R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(
            result,
            "[[[0.4, 0], [0.0979795897113, 0]], [[0.0979795897113, 0], [0.6, 0]]]"
        );
    }

    #[test]
    fn lip005_to_state_round_trips() {
        let (error, result) = run_chunk(
            "STATE_NEW R0 0.4 0.2 0.0\nHOST_CALL density_operator R0 1 R1\nHOST_CALL to_state R1 1 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "state(p=0.4, d_re=0.2, d_im=0)");
    }

    #[test]
    fn lip005_density_operator_rejects_non_hermitian() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 1\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R2 1\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R4 1\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R5\nMOVE R6 R1\nHOST_CALL array_push R5 2 R5\nMOVE R6 R3\nHOST_CALL array_push R5 2 R5\nHOST_CALL tensor R5 1 R7\nHOST_CALL density_operator R7 1 R8\nRETURN R8\n",
        );
        assert_eq!(error, LanaError::InvalidState);
    }

    #[test]
    fn lip005_density_operator_rejects_non_unit_trace() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 1\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R2 0\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R4 1\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R5\nMOVE R6 R1\nHOST_CALL array_push R5 2 R5\nMOVE R6 R3\nHOST_CALL array_push R5 2 R5\nHOST_CALL tensor R5 1 R7\nHOST_CALL density_operator R7 1 R8\nRETURN R8\n",
        );
        assert_eq!(error, LanaError::InvalidState);
    }

    #[test]
    fn lip005_observable_rejects_non_hermitian() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 1\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R2 1\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R4 1\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R5\nMOVE R6 R1\nHOST_CALL array_push R5 2 R5\nMOVE R6 R3\nHOST_CALL array_push R5 2 R5\nHOST_CALL tensor R5 1 R7\nHOST_CALL observable R7 1 R8\nRETURN R8\n",
        );
        assert_eq!(error, LanaError::InvalidParameters);
    }

    #[test]
    fn lip005_expect_of_pauli_z_on_mixed_state_is_zero() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 0.5\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R2 0\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R4 0.5\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R5\nMOVE R6 R1\nHOST_CALL array_push R5 2 R5\nMOVE R6 R3\nHOST_CALL array_push R5 2 R5\nHOST_CALL tensor R5 1 R7\nHOST_CALL density_operator R7 1 R7\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 1\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R2 0\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R4 -1\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R5\nMOVE R6 R1\nHOST_CALL array_push R5 2 R5\nMOVE R6 R3\nHOST_CALL array_push R5 2 R5\nHOST_CALL tensor R5 1 R8\nHOST_CALL observable R8 1 R8\nHOST_CALL expect R7 2 R9\nRETURN R9\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "0");
    }

    #[test]
    fn lip005_trace_distance_of_mixed_and_pure_is_half() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 0.5\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R2 0\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R4 0.5\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R5\nMOVE R6 R1\nHOST_CALL array_push R5 2 R5\nMOVE R6 R3\nHOST_CALL array_push R5 2 R5\nHOST_CALL tensor R5 1 R7\nHOST_CALL density_operator R7 1 R7\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 1\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R2 0\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R5\nMOVE R6 R1\nHOST_CALL array_push R5 2 R5\nMOVE R6 R3\nHOST_CALL array_push R5 2 R5\nHOST_CALL tensor R5 1 R8\nHOST_CALL density_operator R8 1 R8\nHOST_CALL trace_distance R7 2 R9\nRETURN R9\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "0.5");
    }

    // LIP-006: auditable, replayable training primitive. These mirror the C
    // unit test tests/unit/test_training.c and the differential fixture
    // tests/conformance/differential/hostcalls/lip006_train.lasm.

    #[test]
    fn lip006_sgd_defaults() {
        let (error, result) = run_chunk("HOST_CALL sgd R0 0 R1\nRETURN R1\n");
        assert_eq!(error, LanaError::Ok);
        assert_eq!(
            result,
            "optimizer(name=sgd, learning_rate=0.01, momentum=0.9, beta1=0, beta2=0, epsilon=0)"
        );
    }

    #[test]
    fn lip006_sgd_explicit() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0.1\nLOAD_CONST R1 0\nHOST_CALL sgd R0 2 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(
            result,
            "optimizer(name=sgd, learning_rate=0.1, momentum=0, beta1=0, beta2=0, epsilon=0)"
        );
    }

    #[test]
    fn lip006_adam_defaults() {
        let (error, result) = run_chunk("HOST_CALL adam R0 0 R1\nRETURN R1\n");
        assert_eq!(error, LanaError::Ok);
        assert_eq!(
            result,
            "optimizer(name=adam, learning_rate=0.001, momentum=0, beta1=0.9, beta2=0.999, epsilon=1e-08)"
        );
    }

    #[test]
    fn lip006_sgd_rejects_non_positive_learning_rate() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0\nLOAD_CONST R1 0\nHOST_CALL sgd R0 2 R2\nRETURN R2\n",
        );
        assert_eq!(error, LanaError::InvalidParameters);
    }

    #[test]
    fn lip006_adam_rejects_beta1_at_one() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0.001\nLOAD_CONST R1 1\nLOAD_CONST R2 0.999\nLOAD_CONST R3 0.00000001\nHOST_CALL adam R0 4 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::InvalidParameters);
    }

    #[test]
    fn lip010_update_rejects_non_training_result() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0\nLOAD_CONST R1 0\nLOAD_CONST R2 1\nHOST_CALL update R0 3 R3\nHALT\n",
        );
        assert_eq!(error, LanaError::Type);
    }

    #[test]
    fn lip010_update_rejects_non_positive_steps() {
        let (error, _) = run_chunk(
            ".version 3\n.function main 0 64\n\
             LOAD_CONST R0 train\nHOST_CALL shared_information R0 1 R1\n\
             LOAD_CONST R2 use\nHOST_CALL grant R1 2 R3\n\
             LOAD_CONST R4 0.1\nLOAD_CONST R5 0.0\nHOST_CALL sgd R4 2 R6\n\
             LOAD_CONST R7 0\nHOST_CALL array_new R7 1 R8\nLOAD_CONST R9 1.0\nHOST_CALL array_push R8 2 R8\nHOST_CALL tensor R8 1 R10\n\
             LOAD_CONST R11 0\nHOST_CALL array_new R11 1 R12\nLOAD_CONST R13 2.0\nHOST_CALL array_push R12 2 R12\nHOST_CALL tensor R12 1 R14\n\
             LOAD_CONST R15 0\nHOST_CALL array_new R15 1 R16\nMOVE R17 R10\nHOST_CALL array_push R16 2 R16\nMOVE R17 R14\nHOST_CALL array_push R16 2 R16\n\
             LOAD_CONST R18 0\nHOST_CALL array_new R18 1 R19\nMOVE R20 R16\nHOST_CALL array_push R19 2 R19\n\
             LOAD_CONST R21 0\nHOST_CALL array_new R21 1 R22\nLOAD_CONST R23 0.0\nHOST_CALL array_push R22 2 R22\nHOST_CALL tensor R22 1 R24\n\
             LOAD_FUNCTION R25 model\nLOAD_FUNCTION R26 loss\nLOAD_CONST R27 1\nLOAD_CONST R28 1\n\
             MOVE R29 R25\nMOVE R30 R19\nMOVE R31 R26\nMOVE R32 R6\nMOVE R33 R24\nMOVE R34 R27\nMOVE R35 R28\n\
             HOST_CALL train R29 7 R36\n\
             MOVE R40 R36\nMOVE R41 R19\nLOAD_CONST R42 0\nHOST_CALL update R40 3 R43\nHALT\n\
             .function model 2 4\nBINARY R0 mul R1 R2\nRETURN R2\n\
             .function loss 2 8\nBINARY R0 sub R1 R2\nBINARY R0 sub R1 R3\nBINARY R2 mul R3 R4\nMOVE R5 R4\nHOST_CALL tensor_sum R5 1 R6\nRETURN R6\n",
        );
        assert_eq!(error, LanaError::InvalidParameters);
    }

    #[test]
    fn lip010_observe_rejects_non_reactive_value() {
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0\nLOAD_CONST R1 0\nOBSERVE R0 R2 x R1\nHALT\n",
        );
        assert_eq!(error, LanaError::Type);
    }

    #[test]
    fn lip005_is_separable_bell_state_is_entangled() {
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 0.5\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R2 0\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R2 0\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R2 0.5\nHOST_CALL array_push R1 2 R1\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R4 0\nHOST_CALL array_push R3 2 R3\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R5\nLOAD_CONST R6 0\nHOST_CALL array_push R5 2 R5\nLOAD_CONST R6 0\nHOST_CALL array_push R5 2 R5\nLOAD_CONST R6 0\nHOST_CALL array_push R5 2 R5\nLOAD_CONST R6 0\nHOST_CALL array_push R5 2 R5\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R7\nLOAD_CONST R8 0.5\nHOST_CALL array_push R7 2 R7\nLOAD_CONST R8 0\nHOST_CALL array_push R7 2 R7\nLOAD_CONST R8 0\nHOST_CALL array_push R7 2 R7\nLOAD_CONST R8 0.5\nHOST_CALL array_push R7 2 R7\nLOAD_CONST R0 0\nHOST_CALL array_new R0 1 R9\nMOVE R10 R1\nHOST_CALL array_push R9 2 R9\nMOVE R10 R3\nHOST_CALL array_push R9 2 R9\nMOVE R10 R5\nHOST_CALL array_push R9 2 R9\nMOVE R10 R7\nHOST_CALL array_push R9 2 R9\nHOST_CALL tensor R9 1 R11\nHOST_CALL density_operator R11 1 R11\nLOAD_CONST R12 1\nHOST_CALL is_separable R11 2 R13\nRETURN R13\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "entangled");
    }

    // --- LIP-007: differentiable STATE tensors ---

    #[test]
    fn lip007_state_tensor_measure() {
        // s0 = I/2; measure in the computational basis gives [0.5, 0.5].
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0.5\nLOAD_CONST R1 0.0\nMOVE R2 R0\nMOVE R3 R1\nARRAY_NEW R4 R2 2\n\
             LOAD_CONST R5 0.0\nLOAD_CONST R6 0.5\nMOVE R7 R5\nMOVE R8 R6\nARRAY_NEW R9 R7 2\n\
             MOVE R10 R4\nMOVE R11 R9\nARRAY_NEW R12 R10 2\nMOVE R13 R12\nARRAY_NEW R14 R13 1\nMOVE R15 R14\n\
             HOST_CALL state_tensor R15 1 R16\n\
             LOAD_CONST R17 1.0\nLOAD_CONST R18 0.0\nMOVE R19 R17\nMOVE R20 R18\nARRAY_NEW R21 R19 2\n\
             LOAD_CONST R22 0.0\nLOAD_CONST R23 0.0\nMOVE R24 R22\nMOVE R25 R23\nARRAY_NEW R26 R24 2\n\
             MOVE R27 R21\nMOVE R28 R26\nARRAY_NEW R29 R27 2\nMOVE R30 R29\nHOST_CALL tensor R30 1 R31\n\
             LOAD_CONST R32 0.0\nLOAD_CONST R33 0.0\nMOVE R34 R32\nMOVE R35 R33\nARRAY_NEW R36 R34 2\n\
             LOAD_CONST R37 0.0\nLOAD_CONST R38 1.0\nMOVE R39 R37\nMOVE R40 R38\nARRAY_NEW R41 R39 2\n\
             MOVE R42 R36\nMOVE R43 R41\nARRAY_NEW R44 R42 2\nMOVE R45 R44\nHOST_CALL tensor R45 1 R46\n\
             MOVE R47 R31\nMOVE R48 R46\nARRAY_NEW R49 R47 2\nMOVE R50 R49\nHOST_CALL povm R50 1 R51\n\
             MOVE R52 R16\nMOVE R53 R51\nHOST_CALL measure R52 2 R54\nRETURN R54\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[[0.5, 0.5]]");
    }

    #[test]
    fn lip007_append_combines_states() {
        // append(I/2, |0><0|): p_C = 0.5 + 1 - 0.5 = 1.0 -> |0><0|.
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0.5\nLOAD_CONST R1 0.0\nMOVE R2 R0\nMOVE R3 R1\nARRAY_NEW R4 R2 2\n\
             LOAD_CONST R5 0.0\nLOAD_CONST R6 0.5\nMOVE R7 R5\nMOVE R8 R6\nARRAY_NEW R9 R7 2\n\
             MOVE R10 R4\nMOVE R11 R9\nARRAY_NEW R12 R10 2\nMOVE R13 R12\nARRAY_NEW R14 R13 1\nMOVE R15 R14\n\
             HOST_CALL state_tensor R15 1 R16\n\
             LOAD_CONST R17 1.0\nLOAD_CONST R18 0.0\nMOVE R19 R17\nMOVE R20 R18\nARRAY_NEW R21 R19 2\n\
             LOAD_CONST R22 0.0\nLOAD_CONST R23 0.0\nMOVE R24 R22\nMOVE R25 R23\nARRAY_NEW R26 R24 2\n\
             MOVE R27 R21\nMOVE R28 R26\nARRAY_NEW R29 R27 2\nMOVE R30 R29\nARRAY_NEW R31 R30 1\nMOVE R32 R31\n\
             HOST_CALL state_tensor R32 1 R33\n\
             MOVE R34 R16\nMOVE R35 R33\nHOST_CALL append R34 2 R36\nRETURN R36\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[[[[1, 0], [0, 0]], [[0, 0], [0, 0]]]]");
    }

    #[test]
    fn lip007_grad_measure_is_identity() {
        // grad of sum(measure(s, pov)) w.r.t. s = identity (Tr(ρ) = 1).
        let (error, result) = run_chunk(
            ".function main 0 16\n\
             LOAD_CONST R0 0.5\nLOAD_CONST R1 0.0\nMOVE R2 R0\nMOVE R3 R1\nARRAY_NEW R4 R2 2\n\
             LOAD_CONST R5 0.0\nLOAD_CONST R6 0.5\nMOVE R7 R5\nMOVE R8 R6\nARRAY_NEW R9 R7 2\n\
             MOVE R10 R4\nMOVE R11 R9\nARRAY_NEW R12 R10 2\nMOVE R13 R12\nARRAY_NEW R14 R13 1\nMOVE R15 R14\n\
             HOST_CALL state_tensor R15 1 R16\n\
             LOAD_FUNCTION R17 loss\nMOVE R18 R17\nMOVE R19 R16\nHOST_CALL grad R18 2 R20\nRETURN R20\n\
             .function loss 1 8\n\
             LOAD_CONST R1 1.0\nLOAD_CONST R2 0.0\nMOVE R3 R1\nMOVE R4 R2\nARRAY_NEW R5 R3 2\n\
             LOAD_CONST R6 0.0\nLOAD_CONST R7 0.0\nMOVE R8 R6\nMOVE R9 R7\nARRAY_NEW R10 R8 2\n\
             MOVE R11 R5\nMOVE R12 R10\nARRAY_NEW R13 R11 2\nMOVE R14 R13\nHOST_CALL tensor R14 1 R15\n\
             LOAD_CONST R16 0.0\nLOAD_CONST R17 0.0\nMOVE R18 R16\nMOVE R19 R17\nARRAY_NEW R20 R18 2\n\
             LOAD_CONST R21 0.0\nLOAD_CONST R22 1.0\nMOVE R23 R21\nMOVE R24 R22\nARRAY_NEW R25 R23 2\n\
             MOVE R26 R20\nMOVE R27 R25\nARRAY_NEW R28 R26 2\nMOVE R29 R28\nHOST_CALL tensor R29 1 R30\n\
             MOVE R31 R15\nMOVE R32 R30\nARRAY_NEW R33 R31 2\nMOVE R34 R33\nHOST_CALL povm R34 1 R35\n\
             MOVE R36 R0\nMOVE R37 R35\nHOST_CALL measure R36 2 R38\nHOST_CALL tensor_sum R38 1 R39\nRETURN R39\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[[[[1, 0], [0, 0]], [[0, 0], [1, 0]]]]");
    }

    #[test]
    fn lip007_grad_append_is_diagonal() {
        // grad of sum(measure(append(s,s), pov) * [1,0]) w.r.t. s = [[1,0],[0,-1]].
        let (error, result) = run_chunk(
            ".function main 0 16\n\
             LOAD_CONST R0 0.5\nLOAD_CONST R1 0.0\nMOVE R2 R0\nMOVE R3 R1\nARRAY_NEW R4 R2 2\n\
             LOAD_CONST R5 0.0\nLOAD_CONST R6 0.5\nMOVE R7 R5\nMOVE R8 R6\nARRAY_NEW R9 R7 2\n\
             MOVE R10 R4\nMOVE R11 R9\nARRAY_NEW R12 R10 2\nMOVE R13 R12\nARRAY_NEW R14 R13 1\nMOVE R15 R14\n\
             HOST_CALL state_tensor R15 1 R16\n\
             LOAD_FUNCTION R17 loss\nMOVE R18 R17\nMOVE R19 R16\nHOST_CALL grad R18 2 R20\nRETURN R20\n\
             .function loss 1 8\n\
             MOVE R1 R0\nMOVE R2 R0\nHOST_CALL append R1 2 R3\n\
             LOAD_CONST R4 1.0\nLOAD_CONST R5 0.0\nMOVE R6 R4\nMOVE R7 R5\nARRAY_NEW R8 R6 2\n\
             LOAD_CONST R9 0.0\nLOAD_CONST R10 0.0\nMOVE R11 R9\nMOVE R12 R10\nARRAY_NEW R13 R11 2\n\
             MOVE R14 R8\nMOVE R15 R13\nARRAY_NEW R16 R14 2\nMOVE R17 R16\nHOST_CALL tensor R17 1 R18\n\
             LOAD_CONST R19 0.0\nLOAD_CONST R20 0.0\nMOVE R21 R19\nMOVE R22 R20\nARRAY_NEW R23 R21 2\n\
             LOAD_CONST R24 0.0\nLOAD_CONST R25 1.0\nMOVE R26 R24\nMOVE R27 R25\nARRAY_NEW R28 R26 2\n\
             MOVE R29 R23\nMOVE R30 R28\nARRAY_NEW R31 R29 2\nMOVE R32 R31\nHOST_CALL tensor R32 1 R33\n\
             MOVE R34 R18\nMOVE R35 R33\nARRAY_NEW R36 R34 2\nMOVE R37 R36\nHOST_CALL povm R37 1 R38\n\
             MOVE R39 R3\nMOVE R40 R38\nHOST_CALL measure R39 2 R41\n\
             LOAD_CONST R42 1.0\nLOAD_CONST R43 0.0\nMOVE R44 R42\nMOVE R45 R43\nARRAY_NEW R46 R44 2\nMOVE R47 R46\nHOST_CALL tensor R47 1 R48\n\
             BINARY R41 * R48 R49\nHOST_CALL tensor_sum R49 1 R50\nRETURN R50\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "[[[[1, 0], [0, 0]], [[0, 0], [-1, 0]]]]");
    }

    #[test]
    fn tensor_dtype_construction_and_reporting() {
        // Default dtype is f64.
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 0.1\nHOST_CALL array_push R1 2 R1\n\
             HOST_CALL tensor R1 1 R4\nHOST_CALL tensor_dtype R4 1 R5\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "f64");

        // Explicit dtype: on tensor for each dtype.
        for (dtype, hex) in [("f32", "663332"), ("f16", "663136"), ("bf16", "62663136")] {
            let (error, result) = run_chunk(&format!(
                "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 0.1\nHOST_CALL array_push R1 2 R1\n\
                 LOAD_STRING R2 {hex}\nHOST_CALL tensor R1 2 R4\nHOST_CALL tensor_dtype R4 1 R5\nRETURN R5\n"
            ));
            assert_eq!(error, LanaError::Ok);
            assert_eq!(result, dtype);
        }

        // dtype: on zeros/ones/eye.
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 2\nHOST_CALL array_push R1 2 R1\n\
             LOAD_STRING R2 663332\nHOST_CALL tensor_zeros R1 2 R4\nHOST_CALL tensor_dtype R4 1 R5\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "f32");
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 2\nHOST_CALL array_push R1 2 R1\n\
             LOAD_STRING R2 663332\nHOST_CALL tensor_ones R1 2 R4\nHOST_CALL tensor_dtype R4 1 R5\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "f32");
        let (error, result) = run_chunk(
            "LOAD_CONST R0 2\nLOAD_STRING R1 663332\nHOST_CALL tensor_eye R0 2 R4\nHOST_CALL tensor_dtype R4 1 R5\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "f32");
    }

    #[test]
    fn tensor_dtype_rounds_f16_and_bf16_literals() {
        // f16 round of 0.1 -> 0.0999755859375 (round-to-nearest-even).
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 0.1\nHOST_CALL array_push R1 2 R1\n\
             LOAD_STRING R2 663136\nHOST_CALL tensor R1 2 R4\nHOST_CALL tensor_sum R4 1 R5\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "0.0999755859375");

        // bf16 round of 3.14159 -> 3.140625 (mantissa rounded to 7 bits).
        let (error, result) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 3.14159\nHOST_CALL array_push R1 2 R1\n\
             LOAD_STRING R2 62663136\nHOST_CALL tensor R1 2 R4\nHOST_CALL tensor_sum R4 1 R5\nRETURN R5\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "3.140625");
    }

    #[test]
    fn tensor_dtype_error_cases() {
        // Unknown dtype -> LANA_ERR_INVALID_PARAMETERS.
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 0.1\nHOST_CALL array_push R1 2 R1\n\
             LOAD_STRING R2 696e7438\nHOST_CALL tensor R1 2 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::InvalidParameters);

        // dtype: on a non-tensor constructor (tensor_complex takes no dtype) ->
        // LANA_ERR_TYPE.
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0\nHOST_CALL array_new R0 1 R1\nLOAD_CONST R2 0.1\nHOST_CALL array_push R1 2 R1\n\
             LOAD_STRING R3 663332\nHOST_CALL tensor_complex R1 3 R4\nRETURN R4\n",
        );
        assert_eq!(error, LanaError::Type);

        // dtype(t) on a non-tensor -> LANA_ERR_TYPE.
        let (error, _) = run_chunk(
            "LOAD_CONST R0 42\nHOST_CALL tensor_dtype R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Type);
    }

    #[test]
    fn ffi_declare_parses_signature() {
        // "double add(double, double)" -> handle 0.
        let (error, result) = run_chunk(
            "LOAD_STRING R0 646f75626c652061646428646f75626c652c20646f75626c6529\n\
             HOST_CALL ffi_declare R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "0");

        // C's `(void)` means no arguments, not one void argument.
        let (error, result) = run_chunk(
            "LOAD_STRING R0 766f6964206e6f6f7028766f696429\n\
             HOST_CALL ffi_declare R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "0");
    }

    #[test]
    fn ffi_declare_rejects_bad_signature() {
        // Malformed signature -> LANA_ERR_EXTERNAL.
        let (error, _) = run_chunk(
            "LOAD_STRING R0 6e6f742d612d7369676e6174757265\n\
             HOST_CALL ffi_declare R0 1 R1\nRETURN R1\n",
        );
        assert_eq!(error, LanaError::External);
    }

    #[test]
    fn ffi_call_capability_denial() {
        // No `ffi` capability -> LANA_ERR_EXTERNAL.
        let (error, _) = run_chunk(
            "LOAD_CONST R0 0\nLOAD_CONST R1 0\nLOAD_CONST R2 2\nLOAD_CONST R3 3\n\
             ARRAY_NEW R4 R2 2\nLOAD_CONST R5 1\nARRAY_SET R4 R5 R3\nMOVE R2 R4\n\
             HOST_CALL ffi_call R0 3 R6\nRETURN R6\n",
        );
        assert_eq!(error, LanaError::External);
    }

    #[test]
    fn ffi_call_type_rejection() {
        // With the `ffi` capability, a map argument is rejected as a Result
        // error before any library is loaded.
        let (error, result) = run_chunk(
            "LOAD_CONST R0 ffi\nHOST_CALL shared_information R0 1 R1\n\
             LOAD_CONST R2 use\nHOST_CALL grant R1 2 R3\n\
             LOAD_STRING R4 646f75626c652061646428646f75626c652c20646f75626c6529\n\
             HOST_CALL ffi_declare R4 1 R5\n\
             LOAD_CONST R6 k\nLOAD_CONST R7 1\nHOST_CALL map_new R6 2 R8\n\
             ARRAY_NEW R9 R8 2\nLOAD_CONST R10 1\nLOAD_CONST R11 2\nARRAY_SET R9 R10 R11\n\
             LOAD_CONST R0 0\nLOAD_CONST R1 0\nMOVE R2 R9\n\
             HOST_CALL ffi_call R0 3 R12\nRETURN R12\n",
        );
        assert_eq!(error, LanaError::Ok);
        assert_eq!(result, "{\"error\": type}");
    }
}
