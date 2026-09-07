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
use crate::sha256::{hex_digest, sha256};
use crate::state::{self, Indexes, State, StateValue};
use crate::state_dist::{self, DistEvalFrame, EvalAction, LANA_STATE_DIST_DEPTH_LIMIT};
use crate::tensor;
use crate::tensor::{tensor_get_imag, tensor_get_real, tensor_set_imag, tensor_set_real};
use crate::value::{
    Adt, Array, CapabilityToken, Claim, Dataset, DatasetOp, DistOperand, EffectReceipt, InferenceAlgorithm, JointKind, JointRow,
    JointState, Map, MapEntry, Optimizer, PathAlternative, PathSet, PlannedEffect, PlannedEffectState, Possibility, Posterior, Reactive,
    ReactiveKind, ReactiveVersion, RelationshipKind, SharedCommit, SharedInformation,
    SharedObservation, SharedState, SharedVersion, Set, StateDist, StateDistKind, Task, Tensor, TensorDtype, TrainingResult, Value,
    ValueKind, VmError, Regex, RegexClass, RegexInst, RegexOp, Future, LANA_CAPABILITY_ADMIN, LANA_CAPABILITY_OBSERVE, LANA_CAPABILITY_READ,
    LANA_JOINT_CAN_CONDITION, LANA_JOINT_CAN_PROJECT, LANA_JOINT_CAN_RESOLVE,
    LANA_JOINT_CAN_SAMPLE,
};

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
        Call | Fork | HostCall | Generator => {
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
    sets: HashMap<usize, Arc<Mutex<Set>>>,
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
    host_call_extension: Option<Box<dyn FnMut(u32, &[Value], &mut Value) -> LanaError + Send>>,
    /// Optional in-memory filesystem. When set, the file-backed host calls
    /// (`read_text`, `write_text`, `write_text_atomic`, `path_exists`) resolve
    /// against this map instead of `std::fs`, so the self-hosted compiler can
    /// run on targets without a filesystem (e.g. `wasm32-unknown-unknown`).
    virtual_fs: Option<HashMap<String, String>>,
    /// LIP-018 two-way FFI: declared signatures and the single loaded library.
    /// `libloading` is unavailable on `wasm32`, so the loaded library is
    /// compiled out there and `ffi_load`/`ffi_call` return `UnsupportedOperation`.
    ffi_sigs: Vec<String>,
    #[cfg(not(target_arch = "wasm32"))]
    ffi_lib: Option<libloading::Library>,
    /// LIP-019 networking: open sockets, indexed by handle.
    sockets: Vec<NetSocket>,
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
    #[cfg(feature = "net-tls")]
    Tls(Box<rustls::StreamOwned<rustls::ClientConnection, std::net::TcpStream>>),
}

/// LIP-019 networking: a network failure, mapped to a `Result` error reason.
enum NetError {
    Timeout,
    Io,
}

/// Parse a URL into `(scheme, host, port, path)`. Returns `None` on malformed.
fn net_parse_url(url: &str) -> Option<(String, String, u16, String)> {
    let (scheme, rest) = url.split_once("://")?;
    if scheme.is_empty() {
        return None;
    }
    let (host_port, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match host_port.rsplit_once(':') {
        Some((h, p)) if !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()) => {
            (h.to_string(), p.parse::<u16>().ok()?)
        }
        _ => (host_port.to_string(), if scheme == "https" { 443 } else { 80 }),
    };
    if host.is_empty() {
        return None;
    }
    Some((scheme.to_string(), host, port, path.to_string()))
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

impl NetSocket {
    fn read(&mut self, buf: &mut [u8], timeout_ms: u64) -> Result<usize, NetError> {
        use std::io::Read;
        match self {
            NetSocket::Plain(s) => {
                let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(timeout_ms)));
                match s.read(buf) {
                    Ok(n) => Ok(n),
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            || e.kind() == std::io::ErrorKind::TimedOut =>
                    {
                        Err(NetError::Timeout)
                    }
                    Err(_) => Err(NetError::Io),
                }
            }
            #[cfg(feature = "net-tls")]
            NetSocket::Tls(s) => match s.read(buf) {
                Ok(n) => Ok(n),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Err(NetError::Timeout),
                Err(_) => Err(NetError::Io),
            },
        }
    }

    fn write_all(&mut self, data: &[u8]) -> Result<(), NetError> {
        use std::io::Write;
        match self {
            NetSocket::Plain(s) => s.write_all(data).map_err(|_| NetError::Io),
            #[cfg(feature = "net-tls")]
            NetSocket::Tls(s) => s.write_all(data).map_err(|_| NetError::Io),
        }
    }

}

/// Build a rustls client config: verify against the Mozilla roots when
/// `verify` is on, or accept any certificate when it is off (`verify:false`).
#[cfg(feature = "net-tls")]
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
#[cfg(feature = "net-tls")]
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
        conn.complete_io(&mut tcp).map_err(|_| NetError::Io)?;
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
            virtual_fs: None,
            ffi_sigs: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            ffi_lib: None,
            sockets: Vec::new(),
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
            // The event loop (LIP-024 §6) drives async frames. When an async
            // frame returns or suspends, the frame stack returns to the depth
            // at which the event loop started; break back to the loop so it can
            // schedule the next ready future.
            if self.event_loop_active && self.frames.len() <= self.event_loop_base_depth {
                self.event_loop_active = false;
                break;
            }
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

    /// Enqueue a future on the event loop's ready queue (LIP-024 §6). A future
    /// is enqueued at most once: exhausted futures are skipped, and a future
    /// already in the queue is not re-enqueued.
    fn enqueue_future(&mut self, future: Arc<Mutex<Future>>) {
        let should_enqueue = {
            let mut guard = future.lock().unwrap();
            if guard.exhausted || guard.queued {
                false
            } else {
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
                self.poll_composite_future(future);
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
            let error = self.dispatch_loop();
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
            ad_op: -1,
            ad_a: None,
            ad_b: None,
            ad_a_deriv: None,
            ad_b_deriv: None,
            ad_grad: Arc::new(Mutex::new(None)),
            ad_axis: -1,
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

    /// Run a single-argument function to completion, mirroring the nested
    /// execution loop used by `OP_BOOTSTRAP`. The result is written to
    /// `scratch_register` in the caller frame and copied to `result`.
    fn run_function(&mut self, function_index: u32, arg: &Value, scratch_register: u32, result: &mut Value) -> LanaError {
        let arity = self.chunk.functions[function_index as usize].arity;
        let entry = self.chunk.functions[function_index as usize].entry as usize;
        if arity != 1 {
            return LanaError::Type;
        }
        if self.frames.len() >= LANA_MAX_CALL_FRAMES as usize {
            return LanaError::Limit;
        }
        let saved_frame_count = self.frames.len();
        let mut callee = Frame::new(self.max_registers[function_index as usize]);
        callee.return_ip = self.ip;
        callee.return_register = scratch_register;
        callee.function = function_index;
        callee.registers[0] = arg.clone();
        self.frames.push(callee);
        self.ip = entry;
        while self.frames.len() > saved_frame_count && self.running {
            if self.ip >= self.chunk.code.len() {
                self.frames.truncate(saved_frame_count);
                return LanaError::Jump;
            }
            let old_count = self.instruction_count;
            self.instruction_count += 1;
            if old_count >= self.instruction_limit {
                self.frames.truncate(saved_frame_count);
                return LanaError::Limit;
            }
            let instruction = self.chunk.code[self.ip];
            self.ip += 1;
            self.opcode_counts[instruction.opcode as usize] += 1;
            let error = self.execute(&instruction);
            if error != LanaError::Ok {
                self.frames.truncate(saved_frame_count);
                return error;
            }
        }
        *result = self.frames[saved_frame_count - 1].registers[scratch_register as usize].clone();
        LanaError::Ok
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
        let arity = self.chunk.functions[function_index as usize].arity;
        let entry = self.chunk.functions[function_index as usize].entry as usize;
        if arity != 2 {
            return LanaError::Type;
        }
        if self.frames.len() >= LANA_MAX_CALL_FRAMES as usize {
            return LanaError::Limit;
        }
        let saved_frame_count = self.frames.len();
        let mut callee = Frame::new(self.max_registers[function_index as usize]);
        callee.return_ip = self.ip;
        callee.return_register = scratch_register;
        callee.function = function_index;
        callee.registers[0] = arg0.clone();
        callee.registers[1] = arg1.clone();
        self.frames.push(callee);
        self.ip = entry;
        while self.frames.len() > saved_frame_count && self.running {
            if self.ip >= self.chunk.code.len() {
                self.frames.truncate(saved_frame_count);
                return LanaError::Jump;
            }
            let old_count = self.instruction_count;
            self.instruction_count += 1;
            if old_count >= self.instruction_limit {
                self.frames.truncate(saved_frame_count);
                return LanaError::Limit;
            }
            let instruction = self.chunk.code[self.ip];
            self.ip += 1;
            self.opcode_counts[instruction.opcode as usize] += 1;
            let error = self.execute(&instruction);
            if error != LanaError::Ok {
                self.frames.truncate(saved_frame_count);
                return error;
            }
        }
        *result = self.frames[saved_frame_count - 1].registers[scratch_register as usize].clone();
        LanaError::Ok
    }

    /// Record a differentiable primitive onto `result`'s derivation, mirroring
    /// `ad_record` in `vm/c/vm.c`. `ad_op` is 0=add 1=sub 2=mul 3=div 4=matmul
    /// 5=sum 6=mean. `b` is `None` for reductions.
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
        let node = Arc::new(Derivation {
            task_lineage: self.lineage,
            local_sequence: self.derivation_sequence,
            revision: self.revision,
            kind: DerivationKind::Operation,
            operation: Arc::from("autodiff"),
            inputs: retained,
            label: Arc::from(""),
            function: Arc::from(function_name),
            line: 0,
            exactness: DerivationExactness::Exact,
            details: Arc::from("autodiff"),
            outcome: DerivationOutcome::Success,
            reason: Arc::from("none"),
            ad_op,
            ad_a: Some(a_tensor.clone()),
            ad_b: b_tensor,
            ad_a_deriv: a.derivation.clone(),
            ad_b_deriv: b.and_then(|bv| bv.derivation.clone()),
            ad_grad: Arc::new(Mutex::new(None)),
            ad_axis,
        });
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let t = match tensor::tensor_new(&mut alloc, seed.ndim, &seed.shape, seed.is_complex) {
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
                    let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                    match node.ad_op {
                        0 => (seed.clone(), seed.clone()),
                        1 => {
                            let gb = match tensor::tensor_negate(&mut alloc, seed) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            (seed.clone(), gb)
                        }
                        2 => {
                            let ga = match tensor::tensor_elementwise(&mut alloc, seed, b, 2) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            let gb = match tensor::tensor_elementwise(&mut alloc, seed, a, 2) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            (ga, gb)
                        }
                        _ => {
                            let ga = match tensor::tensor_elementwise(&mut alloc, seed, b, 3) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            let t1 = match tensor::tensor_elementwise(&mut alloc, seed, a, 2) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            let t2 = match tensor::tensor_elementwise(&mut alloc, b, b, 2) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            let t3 = match tensor::tensor_elementwise(&mut alloc, &t1, &t2, 3) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            let gb = match tensor::tensor_negate(&mut alloc, &t3) {
                                Ok(t) => t,
                                Err(error) => return error,
                            };
                            (ga, gb)
                        }
                    }
                };
                let (ga_u, gb_u) = {
                    let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                    let ga_u = match tensor::tensor_unbroadcast(&mut alloc, &ga, a) {
                        Ok(t) => t,
                        Err(error) => return error,
                    };
                    let gb_u = match tensor::tensor_unbroadcast(&mut alloc, &gb, b) {
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
                    let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                    if a_ndim == 1 && b_ndim == 1 {
                        let ga = match tensor::tensor_elementwise(&mut alloc, seed, b, 2) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let gb = match tensor::tensor_elementwise(&mut alloc, seed, a, 2) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        (ga, gb)
                    } else if a_ndim == 1 {
                        let bt = tensor::tensor_transpose_last_two(b);
                        let ga = match tensor::tensor_matmul(
                            &mut alloc, seed, &bt, tensor::matmul_default_dtype(seed, &bt),
                        ) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let gb = match tensor::tensor_outer(&mut alloc, a, seed) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        (ga, gb)
                    } else if b_ndim == 1 {
                        let at = tensor::tensor_transpose_last_two(a);
                        let ga = match tensor::tensor_outer(&mut alloc, seed, b) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let gb = match tensor::tensor_matmul(
                            &mut alloc, &at, seed, tensor::matmul_default_dtype(&at, seed),
                        ) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        (ga, gb)
                    } else {
                        let bt = tensor::tensor_transpose_last_two(b);
                        let at = tensor::tensor_transpose_last_two(a);
                        let ga_raw = match tensor::tensor_matmul(
                            &mut alloc, seed, &bt, tensor::matmul_default_dtype(seed, &bt),
                        ) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let gb_raw = match tensor::tensor_matmul(
                            &mut alloc, &at, seed, tensor::matmul_default_dtype(&at, seed),
                        ) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let ga = match tensor::tensor_unbroadcast(&mut alloc, &ga_raw, a) {
                            Ok(t) => t,
                            Err(error) => return error,
                        };
                        let gb = match tensor::tensor_unbroadcast(&mut alloc, &gb_raw, b) {
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
                    let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                    match tensor::tensor_broadcast_reduce(&mut alloc, seed, a, node.ad_axis, scale) {
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
            7 => {
                // append (LIP-007): mean-state distribution-valued APPEND.
                let a = node.ad_a.as_ref().unwrap();
                let b = node.ad_b.as_ref().unwrap();
                let d = a.shape[a.ndim - 1];
                let mut batch = 1;
                for i in 0..a.ndim.saturating_sub(2) {
                    batch *= a.shape[i];
                }
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let mut ga = match tensor::tensor_new(&mut alloc, a.ndim, &a.shape, true) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                let mut gb = match tensor::tensor_new(&mut alloc, b.ndim, &b.shape, true) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let mut gs = match tensor::tensor_new(&mut alloc, s.ndim, &s.shape, true) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let mut gs = match tensor::tensor_new(&mut alloc, s.ndim, &s.shape, true) {
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
        let leaf = Arc::new(Derivation {
            task_lineage: self.lineage,
            local_sequence: self.derivation_sequence,
            revision: self.revision,
            kind: DerivationKind::Operation,
            operation: Arc::from("input"),
            inputs: Vec::new(),
            label: Arc::from(""),
            function: Arc::from(function_name),
            line: 0,
            exactness: DerivationExactness::Exact,
            details: Arc::from("autodiff"),
            outcome: DerivationOutcome::Success,
            reason: Arc::from("none"),
            ad_op: -1,
            ad_a: Some(x_tensor.clone()),
            ad_b: None,
            ad_a_deriv: None,
            ad_b_deriv: None,
            ad_grad: Arc::new(Mutex::new(None)),
            ad_axis: -1,
        });
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
                t.clone()
            } else {
                let a = leaf.ad_a.as_ref().unwrap();
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match tensor::tensor_new(&mut alloc, a.ndim, &a.shape, a.is_complex) {
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            match tensor::tensor_new(&mut alloc, grad.ndim, &grad.shape, grad.is_complex) {
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
        let grad_deriv = Arc::new(Derivation {
            task_lineage: self.lineage,
            local_sequence: self.derivation_sequence,
            revision: self.revision,
            kind: DerivationKind::Operation,
            operation: Arc::from(operation),
            inputs: vec![leaf.clone()],
            label: Arc::from(""),
            function: Arc::from(function_name),
            line: 0,
            exactness: DerivationExactness::Exact,
            details: Arc::from("autodiff"),
            outcome: DerivationOutcome::Success,
            reason: Arc::from("none"),
            ad_op: -1,
            ad_a: None,
            ad_b: None,
            ad_a_deriv: None,
            ad_b_deriv: None,
            ad_grad: Arc::new(Mutex::new(None)),
            ad_axis: -1,
        });
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            match tensor::tensor_new(&mut alloc, 0, &[], false) {
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            match tensor::tensor_new(&mut alloc, v_tensor.ndim, &v_tensor.shape, false) {
                Ok(t) => t,
                Err(error) => return error,
            }
        };
        let count = tensor::tensor_element_count(v_tensor);
        if v_tensor.ndim > 0 {
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            if alloc(v_tensor.ndim * std::mem::size_of::<usize>()) != LanaError::Ok {
                return LanaError::Oom;
            }
        }
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
        let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
        let mut copy = tensor::tensor_new(&mut alloc, t.ndim, &t.shape, t.is_complex)?;
        copy.is_state = t.is_state;
        let count = tensor::tensor_element_count(t);
        if t.ndim > 0 {
            if self.alloc_bytes(t.ndim * std::mem::size_of::<usize>()) != LanaError::Ok {
                return Err(LanaError::Oom);
            }
        }
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            let m_t = tensor::tensor_new(&mut alloc, params.ndim, &params.shape, params.is_complex)?;
            let v_t = tensor::tensor_new(&mut alloc, params.ndim, &params.shape, params.is_complex)?;
            m = Some(Arc::new(m_t));
            v = Some(Arc::new(v_t));
        } else if optimizer.momentum != 0.0 {
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            let vel = tensor::tensor_new(&mut alloc, params.ndim, &params.shape, params.is_complex)?;
            velocity = Some(Arc::new(vel));
        }

        let mut batch_grad = {
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, params.ndim, &params.shape, params.is_complex)?)
        };

        let total_steps = epoch_count * ((dataset_size + batch - 1) / batch);
        if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut steps = Array { items: Vec::with_capacity(total_steps) };

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

                    let leaf = self.record_derivation(
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
                    Arc::get_mut(&mut leaf).unwrap().ad_a = Some(params.clone());

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
                        let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                        tensor::tensor_new(&mut alloc, 0, &[], false)?
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
                            t.clone()
                        } else {
                            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                            tensor::tensor_new(&mut alloc, params.ndim, &params.shape, params.is_complex)?
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
                    let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                    Arc::new(tensor::tensor_new(&mut alloc, params.ndim, &params.shape, params.is_complex)?)
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

                if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                let mut step_map = Map::new(5);
                step_map.set(Arc::from("epoch"), Value::number(epoch as f64), true)?;
                step_map.set(
                    Arc::from("batch"),
                    Value::number((batch_start / batch) as f64),
                    true,
                )?;
                step_map.set(Arc::from("parameters"), new_params_value, true)?;
                step_map.set(Arc::from("gradient"), grad_value, true)?;

                if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                let mut state_map = Map::new(3);
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

                steps.items.push(Value::map(Arc::new(Mutex::new(step_map))));

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
        let result = Arc::new(TrainingResult {
            params,
            steps: Arc::new(Mutex::new(steps)),
            model_function: model_fn,
            loss_function: loss_fn,
            optimizer: optimizer.clone(),
            data: data_copy,
            batch_size: batch,
        });
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            let m_t = tensor::tensor_new(&mut alloc, prior.params.ndim, &prior.params.shape, false)?;
            let v_t = tensor::tensor_new(&mut alloc, prior.params.ndim, &prior.params.shape, false)?;
            m = Some(Arc::new(m_t));
            v = Some(Arc::new(v_t));
        } else if optimizer.momentum != 0.0 {
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            let vel = tensor::tensor_new(&mut alloc, prior.params.ndim, &prior.params.shape, false)?;
            velocity = Some(Arc::new(vel));
        }

        let mut params = prior.params.clone();
        let param_count = tensor::tensor_element_count(&params);

        let mut batch_grad = {
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, params.ndim, &params.shape, false)?)
        };

        if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut steps = Array { items: Vec::with_capacity(prior_count + step_count) };
        {
            let prior_items = prior.steps.lock().unwrap();
            for item in prior_items.items.iter() {
                steps.items.push(item.clone());
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

                let leaf = self.record_derivation(
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
                Arc::get_mut(&mut leaf).unwrap().ad_a = Some(params.clone());

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
                    let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                    tensor::tensor_new(&mut alloc, 0, &[], false)?
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
                        t.clone()
                    } else {
                        let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                        tensor::tensor_new(&mut alloc, params.ndim, &params.shape, false)?
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                Arc::new(tensor::tensor_new(&mut alloc, params.ndim, &params.shape, false)?)
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

            if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
                return Err(LanaError::Oom);
            }
            let mut step_map = Map::new(5);
            step_map.set(Arc::from("epoch"), Value::number((prior_count + step) as f64), true)?;
            step_map.set(Arc::from("batch"), Value::number(0.0), true)?;
            step_map.set(Arc::from("parameters"), new_params_value, true)?;
            step_map.set(Arc::from("gradient"), grad_value, true)?;

            if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
                return Err(LanaError::Oom);
            }
            let mut state_map = Map::new(3);
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

            steps.items.push(Value::map(Arc::new(Mutex::new(step_map))));

            params = new_params;
            params_deriv = Some(step_deriv);
        }

        *out = Value::training_result(Arc::new(TrainingResult {
            params,
            steps: Arc::new(Mutex::new(steps)),
            model_function: prior.model_function,
            loss_function: prior.loss_function,
            optimizer: prior.optimizer.clone(),
            data: prior.data.clone(),
            batch_size: prior.batch_size,
        }));
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

        let single = Arc::new(Mutex::new(Array { items: vec![observation.clone()] }));
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, params.ndim, &params.shape, false)?)
        };

        // New step history: copy steps 0..=step_index, recompute step_index+1..
        if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut steps = Array { items: Vec::with_capacity(total_steps) };
        {
            let prior_items = prior.steps.lock().unwrap();
            for item in prior_items.items.iter().take(step_index + 1) {
                steps.items.push(item.clone());
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

                let leaf = self.record_derivation(
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
                Arc::get_mut(&mut leaf).unwrap().ad_a = Some(params.clone());

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
                    let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                    tensor::tensor_new(&mut alloc, 0, &[], false)?
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
                        t.clone()
                    } else {
                        let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                        tensor::tensor_new(&mut alloc, params.ndim, &params.shape, false)?
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                Arc::new(tensor::tensor_new(&mut alloc, params.ndim, &params.shape, false)?)
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

            if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
                return Err(LanaError::Oom);
            }
            let mut step_map = Map::new(5);
            step_map.set(Arc::from("epoch"), Value::number(epoch as f64), true)?;
            step_map.set(Arc::from("batch"), Value::number(batch_index as f64), true)?;
            step_map.set(Arc::from("parameters"), new_params_value, true)?;
            step_map.set(Arc::from("gradient"), grad_value, true)?;

            if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
                return Err(LanaError::Oom);
            }
            let mut state_map = Map::new(3);
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

            steps.items.push(Value::map(Arc::new(Mutex::new(step_map))));

            params = new_params;
            params_deriv = Some(step_deriv);
        }

        let mut result_value = Value::training_result(Arc::new(TrainingResult {
            params,
            steps: Arc::new(Mutex::new(steps)),
            model_function: prior.model_function,
            loss_function: prior.loss_function,
            optimizer: prior.optimizer.clone(),
            data: prior.data.clone(),
            batch_size: prior.batch_size,
        }));

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
        let params_value = Value::tensor(Arc::new(params.clone()));
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
        if self.alloc_bytes(std::mem::size_of::<Map>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut map = Map::new(3);
        let snap = self.tensor_copy_contiguous(params)?;
        let prior_value = Value::tensor(Arc::new(prior.clone()));
        let data_value = Value::tensor(Arc::new(data.clone()));
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
        steps.items.push(Value::map(Arc::new(Mutex::new(map))));
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, 2, &sample_shape, false)?)
        };
        if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut steps = Array { items: Vec::with_capacity(samples) };

        let mut current_ll = self.infer_log_likelihood(model_fn, &params, data, scratch)?;

        let mut kept_index = 0usize;
        for i in 0..samples {
            let error = self.consume_sampling_budget();
            if error != LanaError::Ok {
                return Err(error);
            }
            let mut proposal = {
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut variance = {
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut eps = {
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut sigma = {
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut params = {
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
        };
        {
            let log_sigma_tensor = Arc::get_mut(&mut log_sigma).unwrap();
            for k in 0..param_count {
                tensor_set_real(log_sigma_tensor, k, 0.1f64.ln());
            }
        }
        if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut steps = Array { items: Vec::with_capacity(iterations) };

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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                Arc::new(tensor::tensor_elementwise(&mut alloc, pred_tensor, data, 1)?)
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut variance = {
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, 2, &particle_shape, false)?)
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
        if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut steps = Array { items: Vec::with_capacity(1) };

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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, 2, &particle_shape, false)?)
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
        };
        let mut variance = {
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            Arc::new(tensor::tensor_new(&mut alloc, prior.ndim, &prior.shape, false)?)
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

        if self.alloc_bytes(std::mem::size_of::<Posterior>()) != LanaError::Ok {
            return Err(LanaError::Oom);
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

        *out = Value::posterior(Arc::new(posterior));
        Ok(())
    }

    /// Build a `Result` tagged pair `[ok, value]`, mirroring `make_result` in
    /// `vm/c/vm.c`.
    fn make_result(&self, ok: bool, value: Value) -> Value {
        let array = Array { items: vec![Value::boolean(ok), value] };
        Value::array(Arc::new(Mutex::new(array)))
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
                self.current_frame_mut().registers[ins.a as usize] =
                    Value::generator(Arc::new(Mutex::new(generator)));
                LanaError::Ok
            }
            Yield => {
                let gen_value = self.current_frame().registers[ins.a as usize].clone();
                let yielded = self.current_frame().registers[ins.b as usize].clone();
                let ValueKind::Generator(generator) = gen_value.kind else {
                    return LanaError::Type;
                };
                {
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
                let result = self.make_result(true, yielded);
                self.current_frame_mut().registers[destination as usize] = result;
                self.ip = return_ip;
                LanaError::Ok
            }
            Next => {
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
                    let result = self.make_result(false, Value::string(Arc::from("exhausted")));
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
                self.current_frame_mut().registers[ins.a as usize] =
                    Value::future(Arc::new(Mutex::new(future)));
                LanaError::Ok
            }
            Await => {
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
                let data = data_arc.lock().unwrap().items.clone();
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
                    let resampled = Value::array(Arc::new(Mutex::new(Array { items })));
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
                if self.alloc_bytes(std::mem::size_of::<crate::value::Map>()) != LanaError::Ok {
                    return LanaError::Oom;
                }
                let mut map = crate::value::Map::new(7);
                map.set(Arc::from("estimate"), Value::number(estimate), false).ok();
                map.set(Arc::from("ci_low"), Value::number(ci_low), false).ok();
                map.set(Arc::from("ci_high"), Value::number(ci_high), false).ok();
                map.set(Arc::from("method"), Value::string(Arc::from("sampled")), false).ok();
                map.set(Arc::from("procedure"), Value::string(Arc::from("bootstrap")), false).ok();
                map.set(Arc::from("sample_count"), Value::number(b as f64), false).ok();
                map.set(Arc::from("seed"), Value::number(self.root_seed as f64), false).ok();
                self.current_frame_mut().registers[ins.a as usize] =
                    Value::map(Arc::new(Mutex::new(map)));
                LanaError::Ok
            }
            Return => {
                let returned = self.current_frame().registers[ins.a as usize].clone();
                if self.current_frame().is_async {
                    let future_value = self.current_frame().registers[0].clone();
                    let ValueKind::Future(future) = future_value.kind else {
                        return LanaError::Type;
                    };
                    {
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
                    generator.lock().unwrap().exhausted = true;
                    if self.frames.len() == 1 {
                        return LanaError::Type;
                    }
                    let return_ip = self.current_frame().return_ip;
                    let destination = self.current_frame().return_register;
                    self.frames.pop();
                    let result = self.make_result(false, Value::string(Arc::from("exhausted")));
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
            | ValueKind::Dataset(_) => {
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
            ValueKind::Set(set) => {
                let key = Arc::as_ptr(set) as usize;
                if let Some(existing) = memo.sets.get(&key) {
                    cloned.kind = ValueKind::Set(existing.clone());
                } else {
                    if self.alloc_bytes(std::mem::size_of::<Set>()) != LanaError::Ok {
                        return Err(LanaError::Oom);
                    }
                    let items = set
                        .lock().unwrap()
                        .items
                        .iter()
                        .map(|item| self.deep_clone_value(item, memo))
                        .collect::<Result<Vec<_>, _>>()?;
                    let copy = Arc::new(Mutex::new(Set { items }));
                    memo.sets.insert(key, copy.clone());
                    cloned.kind = ValueKind::Set(copy);
                }
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let t = match tensor::tensor_elementwise_scalar(&mut alloc, t, *s, operation) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *out = Value::tensor(Arc::new(t));
                return LanaError::Ok;
            }
            if matches!(left.kind, ValueKind::Number(_)) && matches!(right.kind, ValueKind::Tensor(_)) {
                let ValueKind::Number(s) = &left.kind else { unreachable!() };
                let ValueKind::Tensor(t) = &right.kind else { unreachable!() };
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let t = match tensor::tensor_elementwise_scalar(&mut alloc, t, *s, operation) {
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
            let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
            if a_unc || b_unc {
                let a_var = match a_var {
                    Some(v) => v,
                    None => match tensor::tensor_zeros_like(&mut alloc, &a_pred) {
                        Ok(t) => Arc::new(t),
                        Err(error) => return error,
                    },
                };
                let b_var = match b_var {
                    Some(v) => v,
                    None => match tensor::tensor_zeros_like(&mut alloc, &b_pred) {
                        Ok(t) => Arc::new(t),
                        Err(error) => return error,
                    },
                };
                *out = match tensor::tensor_elementwise_uncertain(
                    &mut alloc, &a_pred, &a_var, &b_pred, &b_var, operation,
                ) {
                    Ok(v) => v,
                    Err(error) => return error,
                };
                return LanaError::Ok;
            }
            let t = match tensor::tensor_elementwise(&mut alloc, &a_pred, &b_pred, operation) {
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
        value.is_unresolved()
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
            ValueKind::Set(set) => {
                if self.alloc_bytes(std::mem::size_of::<Set>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                let items = set
                    .lock().unwrap()
                    .items
                    .iter()
                    .map(|item| self.materialize_value(item))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Value::set(Arc::new(Mutex::new(Set { items }))))
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
            is_training_data,
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
            | ValueKind::Dataset(_) => cloned.kind = value.kind.clone(),
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
            ValueKind::Set(set) => {
                let items = set
                    .lock().unwrap()
                    .items
                    .iter()
                    .map(|item| self.deep_clone_live_value(item, reactive_memo))
                    .collect::<Result<Vec<_>, _>>()?;
                cloned.kind = ValueKind::Set(Arc::new(Mutex::new(Set { items })));
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
            is_training_data: false,
        }));
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
        if plan.payload.has_revoked_capability() {
            return Err(LanaError::ClaimRevoked);
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
                    ReactiveKind::Unary => self.lift_unary(&left, operation, &mut staged_value),
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let shape = match tensor::tensor_shape_from_array(&mut alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let t = match tensor::tensor_new(&mut alloc, shape.len(), &shape, is_complex) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let shape = match tensor::tensor_shape_from_array(&mut alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&mut alloc, shape.len(), &shape, false) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let shape = match tensor::tensor_shape_from_array(&mut alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&mut alloc, shape.len(), &shape, false) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let mut t = match tensor::tensor_new(&mut alloc, 2, &shape, false) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match tensor::tensor_cast(&mut alloc, t, dtype) {
                    Ok(t) => {
                        *out = Value::tensor(Arc::new(t));
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_SHAPE => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(t) = &arguments[0].kind else {
                    return LanaError::Type;
                };
                if self.alloc_bytes(std::mem::size_of::<Array>()) != LanaError::Ok {
                    return LanaError::Oom;
                }
                let items = t.shape.iter().map(|&dim| Value::number(dim as f64)).collect();
                *out = Value::array(Arc::new(Mutex::new(Array { items })));
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                if a_unc || b_unc {
                    let a_var = match a_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&mut alloc, &a_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    let b_var = match b_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&mut alloc, &b_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    return match tensor::tensor_elementwise_uncertain(
                        &mut alloc, &a_pred, &a_var, &b_pred, &b_var, op,
                    ) {
                        Ok(value) => {
                            *out = value;
                            LanaError::Ok
                        }
                        Err(error) => error,
                    };
                }
                let t = match tensor::tensor_elementwise(&mut alloc, &a_pred, &b_pred, op) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                if a_unc || b_unc {
                    let a_var = match a_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&mut alloc, &a_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    let b_var = match b_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&mut alloc, &b_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    return match tensor::tensor_matmul_uncertain(
                        &mut alloc, &a_pred, &a_var, &b_pred, &b_var,
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
                let t = match tensor::tensor_matmul(&mut alloc, &a_pred, &b_pred, out_dtype) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let t = match tensor::tensor_gpu_matmul(&mut alloc, a, b) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                if unc {
                    if op >= 2 {
                        return LanaError::Type;
                    }
                    let var = var.expect("uncertain tensor has a variance");
                    let axis = if argc == 2 { Some(&arguments[1]) } else { None };
                    return match tensor::tensor_reduce_uncertain(&mut alloc, &pred, &var, op, axis) {
                        Ok(value) => {
                            *out = value;
                            LanaError::Ok
                        }
                        Err(error) => error,
                    };
                }
                let result = if argc == 2 {
                    tensor::tensor_reduce_axis(&mut alloc, &pred, op, &arguments[1])
                } else {
                    tensor::tensor_reduce(&mut alloc, &pred, op)
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let shape = match tensor::tensor_infer_shape(&mut alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&mut alloc, shape.len(), &shape, false) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                let shape = match tensor::tensor_infer_shape(&mut alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&mut alloc, shape.len(), &shape, true) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match &arguments[0].kind {
                    ValueKind::State(state) => {
                        linalg_density_from_state(&mut alloc, &state.state)
                    }
                    ValueKind::Tensor(t) => linalg_density_from_tensor(&mut alloc, t),
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_povm(&mut alloc, &arguments[0]) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_channel(&mut alloc, &arguments[0]) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_observable(&mut alloc, &arguments[0]) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_tensor_product(&mut alloc, a, b) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_partial_trace(&mut alloc, ab, subsystem) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_measure_with(&mut alloc, rho, povm) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_apply_to(&mut alloc, chan, rho) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_mix(&mut alloc, a, b, w) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_trace_distance(&mut alloc, a, b) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_is_separable(&mut alloc, ab, bipartition) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_state_tensor(&mut alloc, &arguments[0]) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_state_append(&mut alloc, a, b) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_state_measure(&mut alloc, s, povm) {
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
                let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                match linalg_state_transform(&mut alloc, s, chan) {
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
                if matches!(arguments[0].kind, ValueKind::Tensor(_)) {
                    let ValueKind::Tensor(tensor) = &arguments[0].kind else {
                        unreachable!()
                    };
                    let mut alloc = |bytes: usize| self.alloc_bytes(bytes);
                    return match tensor::tensor_index(&mut alloc, tensor, &arguments[1]) {
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
            LANA_HOST_SET_NEW => {
                if argc != 0 {
                    return LanaError::Type;
                }
                *out = Value::set(Arc::new(Mutex::new(Set { items: Vec::new() })));
                LanaError::Ok
            }
            LANA_HOST_SET_ADD => {
                if argc != 2 || !matches!(arguments[0].kind, ValueKind::Set(_)) {
                    return LanaError::Type;
                }
                if !value_is_set_member(&arguments[1]) {
                    return LanaError::Type;
                }
                let ValueKind::Set(source) = &arguments[0].kind else {
                    unreachable!()
                };
                let source = source.lock().unwrap();
                let mut items = source.items.clone();
                if !items.iter().any(|item| set_value_equal(item, &arguments[1])) {
                    items.push(arguments[1].clone());
                }
                *out = Value::set(Arc::new(Mutex::new(Set { items })));
                LanaError::Ok
            }
            LANA_HOST_SET_CONTAINS => {
                if argc != 2 || !matches!(arguments[0].kind, ValueKind::Set(_)) {
                    return LanaError::Type;
                }
                if !value_is_set_member(&arguments[1]) {
                    return LanaError::Type;
                }
                let ValueKind::Set(source) = &arguments[0].kind else {
                    unreachable!()
                };
                let source = source.lock().unwrap();
                let found = source.items.iter().any(|item| set_value_equal(item, &arguments[1]));
                *out = Value::boolean(found);
                LanaError::Ok
            }
            LANA_HOST_SET_UNION => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Set(_))
                    || !matches!(arguments[1].kind, ValueKind::Set(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Set(left) = &arguments[0].kind else {
                    unreachable!()
                };
                let ValueKind::Set(right) = &arguments[1].kind else {
                    unreachable!()
                };
                let left = left.lock().unwrap();
                let right = right.lock().unwrap();
                let mut items = left.items.clone();
                for item in right.items.iter() {
                    if !items.iter().any(|existing| set_value_equal(existing, item)) {
                        items.push(item.clone());
                    }
                }
                *out = Value::set(Arc::new(Mutex::new(Set { items })));
                LanaError::Ok
            }
            LANA_HOST_SET_INTERSECT => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Set(_))
                    || !matches!(arguments[1].kind, ValueKind::Set(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Set(left) = &arguments[0].kind else {
                    unreachable!()
                };
                let ValueKind::Set(right) = &arguments[1].kind else {
                    unreachable!()
                };
                let left = left.lock().unwrap();
                let right = right.lock().unwrap();
                let items = left
                    .items
                    .iter()
                    .filter(|item| right.items.iter().any(|r| set_value_equal(item, r)))
                    .cloned()
                    .collect();
                *out = Value::set(Arc::new(Mutex::new(Set { items })));
                LanaError::Ok
            }
            LANA_HOST_SET_DIFFERENCE => {
                if argc != 2
                    || !matches!(arguments[0].kind, ValueKind::Set(_))
                    || !matches!(arguments[1].kind, ValueKind::Set(_))
                {
                    return LanaError::Type;
                }
                let ValueKind::Set(left) = &arguments[0].kind else {
                    unreachable!()
                };
                let ValueKind::Set(right) = &arguments[1].kind else {
                    unreachable!()
                };
                let left = left.lock().unwrap();
                let right = right.lock().unwrap();
                let items = left
                    .items
                    .iter()
                    .filter(|item| !right.items.iter().any(|r| set_value_equal(item, r)))
                    .cloned()
                    .collect();
                *out = Value::set(Arc::new(Mutex::new(Set { items })));
                LanaError::Ok
            }
            LANA_HOST_GETENV => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let ValueKind::String(name) = &arguments[0].kind else {
                    unreachable!()
                };
                let value = std::env::var(&**name).unwrap_or_default();
                *out = Value::string(value.into());
                LanaError::Ok
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
                        *out = self.make_result(true, Value::number(number));
                        LanaError::Ok
                    }
                    Err(_) => {
                        *out = self.make_result(false, Value::string(Arc::from("invalid number")));
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
                };
                *out = Value::string(Arc::from(name));
                LanaError::Ok
            }
            LANA_HOST_FORMAT => {
                if argc < 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let format = arguments[0].as_string();
                let bytes = format.as_bytes();
                let mut result = String::new();
                let mut arg_index = 1usize;
                let mut i = 0usize;
                let mut last = 0usize;
                while i < bytes.len() {
                    if bytes[i] == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'}' {
                        result.push_str(&format[last..i]);
                        if arg_index >= argc {
                            return LanaError::Format;
                        }
                        match &arguments[arg_index].kind {
                            ValueKind::Null => result.push_str("null"),
                            ValueKind::Bool(value) => {
                                result.push_str(if *value { "true" } else { "false" })
                            }
                            ValueKind::Number(value) => result.push_str(&format_g17(*value)),
                            ValueKind::String(value) => result.push_str(value),
                            ValueKind::Array(_) | ValueKind::Map(_) => {
                                let mut stringified = Value::null();
                                let error =
                                    self.json_stringify(&arguments[arg_index], &mut stringified);
                                if error != LanaError::Ok {
                                    return error;
                                }
                                result.push_str(&stringified.as_string());
                            }
                            _ => return LanaError::Type,
                        }
                        arg_index += 1;
                        i += 2;
                        last = i;
                    } else {
                        i += 1;
                    }
                }
                result.push_str(&format[last..]);
                if arg_index != argc {
                    return LanaError::Format;
                }
                *out = Value::string(Arc::from(result));
                LanaError::Ok
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
                *out = Value::string(Arc::from(text));
                LanaError::Ok
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
                *out = Value::string(Arc::from(&text[start_byte..end_byte]));
                LanaError::Ok
            }
            LANA_HOST_TO_UPPER | LANA_HOST_TO_LOWER => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let upper = host_id == LANA_HOST_TO_UPPER;
                let text = arguments[0].as_string();
                let bytes = text.as_bytes();
                let mut result = Vec::with_capacity(bytes.len());
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
                    result.extend_from_slice(&utf8_encode(mapped));
                    i += consumed;
                }
                *out = Value::string(Arc::from(String::from_utf8(result).unwrap()));
                LanaError::Ok
            }
            LANA_HOST_REGEX_COMPILE => {
                if argc != 1 || !matches!(arguments[0].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let pattern = arguments[0].as_string();
                match regex_compile(pattern.as_bytes()) {
                    Ok(re) => *out = self.make_result(true, Value::regex(Arc::new(re))),
                    Err(msg) => *out = self.make_result(false, Value::string(Arc::from(msg))),
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
                        let mut map = Map::new(3);
                        map.set(Arc::from("start"), Value::number(start as f64), false).ok();
                        map.set(Arc::from("end"), Value::number(end as f64), false).ok();
                        let matched_text =
                            String::from_utf8_lossy(&bytes[start..end]).into_owned();
                        map.set(Arc::from("text"), Value::string(Arc::from(matched_text)), false)
                            .ok();
                        *out = self.make_result(true, Value::map(Arc::new(Mutex::new(map))));
                    }
                    None => {
                        *out = self.make_result(false, Value::string(Arc::from("no match")))
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
                let mut result: Vec<u8> = Vec::new();
                let mut pos = 0usize;
                while pos <= bytes.len() {
                    match regex_search(re, bytes, pos) {
                        Some((start, end)) => {
                            result.extend_from_slice(&bytes[pos..start]);
                            result.extend_from_slice(replacement.as_bytes());
                            pos = end;
                            if start == end {
                                if pos < bytes.len() {
                                    result.extend_from_slice(&bytes[pos..pos + 1]);
                                    pos += 1;
                                } else {
                                    break;
                                }
                            }
                        }
                        None => {
                            result.extend_from_slice(&bytes[pos..]);
                            break;
                        }
                    }
                }
                *out = Value::string(Arc::from(String::from_utf8_lossy(&result).into_owned()));
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
            _ => match self.host_call_extension.as_mut() {
                Some(handler) => handler(host_id, arguments, out),
                None => LanaError::Format,
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
        let arg_values: Vec<Value> = args.lock().unwrap().items.clone();
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
        let mut map = Map::new(1);
        if map.set(Arc::from(key), value.clone(), false).is_err() {
            return LanaError::Oom;
        }
        *out = Value::map(Arc::new(Mutex::new(map)));
        LanaError::Ok
    }

    /// LIP-019 networking: error result `[false, reason]`, mirroring the
    /// language `Result` tagged-pair that `result_error` and json_parse emit.
    fn net_error_result(&self, reason: &str, out: &mut Value) -> LanaError {
        *out = self.make_result(false, Value::string(Arc::from(reason)));
        LanaError::Ok
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
        out: &mut Value,
    ) -> LanaError {
        let timeout = if timeout_ms > 0.0 { timeout_ms as u64 } else { 5000 };
        let (scheme, host, port, path) = match net_parse_url(url) {
            Some(x) => x,
            None => return self.net_error_result("url", out),
        };
        let stream = match net_connect(&host, port, timeout) {
            Ok(s) => s,
            Err(NetError::Timeout) => return self.net_error_result("timeout", out),
            Err(_) => return self.net_error_result("connect", out),
        };
        let mut sock = if scheme == "https" {
            #[cfg(feature = "net-tls")]
            {
                match net_tls_connect(stream, &host, verify) {
                    Ok(s) => s,
                    Err(NetError::Timeout) => return self.net_error_result("timeout", out),
                    Err(_) => return self.net_error_result("tls", out),
                }
            }
            #[cfg(not(feature = "net-tls"))]
            {
                NetSocket::Plain(stream)
            }
        } else {
            NetSocket::Plain(stream)
        };
        let mut request = format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n",
            method, path, host
        );
        match body {
            Some(b) => request.push_str(&format!("Content-Length: {}\r\n\r\n{}", b.len(), b)),
            None => request.push_str("Content-Length: 0\r\n\r\n"),
        }
        if sock.write_all(request.as_bytes()).is_err() {
            return self.net_error_result("send", out);
        }
        let mut response = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            match sock.read(&mut buf, timeout) {
                Ok(0) => break,
                Ok(n) => response.extend_from_slice(&buf[..n]),
                Err(NetError::Timeout) => return self.net_error_result("timeout", out),
                Err(_) => break,
            }
        }
        let text = String::from_utf8_lossy(&response).to_string();
        let header_end = match text.find("\r\n\r\n") {
            Some(i) => i,
            None => return self.net_error_result("response", out),
        };
        let status_line = &text[..text.find("\r\n").unwrap_or(0)];
        let status = if let Some(sp) = status_line.find(' ') {
            status_line[sp + 1..].split(' ').next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0)
        } else {
            0.0
        };
        let body_text = &text[header_end + 4..];
        let mut resp = Map::new(3);
        let status_value = Value::number(status);
        let body_value = Value::string(Arc::from(body_text));
        if resp.set(Arc::from("status"), status_value, false).is_err()
            || resp.set(Arc::from("headers"), Value::map(Arc::new(Mutex::new(Map::new(0)))), false).is_err()
            || resp.set(Arc::from("body"), body_value, false).is_err()
        {
            return LanaError::Oom;
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
        *out = self.make_result(true, rooted);
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
        // LIP-019 `verify:false` is an explicit per-call opt-out read from the
        // headers map; TLS validation is otherwise ON by default.
        let mut verify = true;
        if let Some(v) = headers.lock().unwrap().get("verify") {
            if let ValueKind::Bool(b) = &v.kind {
                verify = *b;
            }
        }
        self.net_http_request("GET", url, None, *timeout_ms, verify, out)
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
        let mut verify = true;
        if let Some(v) = headers.lock().unwrap().get("verify") {
            if let ValueKind::Bool(b) = &v.kind {
                verify = *b;
            }
        }
        self.net_http_request("POST", url, Some(body), *timeout_ms, verify, out)
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
        *out = self.make_result(true, Value::number(handle as f64));
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
                *out = self.make_result(true, Value::number(bytes.len() as f64));
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
        let mut buf = vec![0u8; (*max_bytes as usize).min(65535)];
        match sock.read(&mut buf, 5000) {
            Ok(0) => {
                *out = self.make_result(true, Value::string(Arc::from("")));
                LanaError::Ok
            }
            Ok(n) => {
                *out = self.make_result(
                    true,
                    Value::string(Arc::from(String::from_utf8_lossy(&buf[..n]).to_string())),
                );
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
        let items = array.lock().unwrap().items.clone();
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
        let items = array.lock().unwrap().items.clone();
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
    /// for the duration; on wasm (no clock) it completes immediately.
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
        *out = Value::dataset(Arc::new(Dataset {
            op: DatasetOp::Source,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: Value::null(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }));
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
        *out = Value::dataset(Arc::new(Dataset {
            op: DatasetOp::Filter,
            source: arguments[0].clone(),
            function,
            columns: Value::null(),
            key: Value::null(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }));
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
        *out = Value::dataset(Arc::new(Dataset {
            op: DatasetOp::Map,
            source: arguments[0].clone(),
            function,
            columns: Value::null(),
            key: Value::null(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }));
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
        *out = Value::dataset(Arc::new(Dataset {
            op: DatasetOp::Select,
            source: arguments[0].clone(),
            function: 0,
            columns: arguments[1].clone(),
            key: Value::null(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }));
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
        *out = Value::dataset(Arc::new(Dataset {
            op: DatasetOp::Limit,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: Value::null(),
            limit: arguments[1].clone(),
            other: Value::null(),
            aggregate: Value::null(),
        }));
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
        *out = Value::dataset(Arc::new(Dataset {
            op: DatasetOp::Sort,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: arguments[1].clone(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }));
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
        *out = Value::dataset(Arc::new(Dataset {
            op: DatasetOp::GroupBy,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: arguments[1].clone(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: Value::null(),
        }));
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
        *out = Value::dataset(Arc::new(Dataset {
            op: DatasetOp::Aggregate,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: Value::null(),
            limit: Value::null(),
            other: Value::null(),
            aggregate: arguments[1].clone(),
        }));
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
        *out = Value::dataset(Arc::new(Dataset {
            op: DatasetOp::Join,
            source: arguments[0].clone(),
            function: 0,
            columns: Value::null(),
            key: arguments[2].clone(),
            limit: Value::null(),
            other: arguments[1].clone(),
            aggregate: Value::null(),
        }));
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
        *out = Value::array(Arc::new(Mutex::new(Array { items: rows })));
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
            return Ok(array.lock().unwrap().items.clone());
        }
        let ValueKind::Lazy { function, bound } = lazy.kind else {
            return Err(LanaError::Type);
        };
        let mut rows = Vec::with_capacity(bound);
        for i in 0..bound {
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
                    let mut pred_result = Value::null();
                    let error = self.run_function(dataset.function, &row, scratch, &mut pred_result);
                    if error != LanaError::Ok {
                        return Err(error);
                    }
                    if matches!(pred_result.kind, ValueKind::Bool(true)) {
                        result.push(row);
                    }
                }
                Ok(result)
            }
            DatasetOp::Map => {
                let source_rows = self.dataset_materialize(&self.dataset_as(dataset)?, scratch)?;
                let mut result = Vec::with_capacity(source_rows.len());
                for row in source_rows {
                    let mut mapped = Value::null();
                    let error = self.run_function(dataset.function, &row, scratch, &mut mapped);
                    if error != LanaError::Ok {
                        return Err(error);
                    }
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
                    let mut projected = Map::new(columns.items.len());
                    for col in &columns.items {
                        let ValueKind::String(col_name) = &col.kind else {
                            return Err(LanaError::Type);
                        };
                        let col_value = row_map.get(&**col_name).cloned().ok_or(LanaError::Type)?;
                        projected.set(col_name.clone(), col_value, false)?;
                    }
                    result.push(Value::map(Arc::new(Mutex::new(projected))));
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
                Ok(source_rows.into_iter().take(take).collect())
            }
            DatasetOp::Sort => {
                let ValueKind::String(key) = &dataset.key.kind else {
                    return Err(LanaError::Type);
                };
                let mut result = self.dataset_materialize(&self.dataset_as(dataset)?, scratch)?;
                // Insertion sort by key value (rows are maps).
                for i in 1..result.len() {
                    let key_i = Self::dataset_row_key(&result[i], &**key)?;
                    let pivot = result[i].clone();
                    let mut j = i;
                    while j > 0 {
                        let key_j = Self::dataset_row_key(&result[j - 1], &**key)?;
                        if Self::dataset_compare_values(&key_j, &key_i) <= 0 {
                            break;
                        }
                        result[j] = result[j - 1].clone();
                        j -= 1;
                    }
                    result[j] = pivot;
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
                    let key_value = Self::dataset_row_key(&row, &**key)?;
                    // Find an existing group record with this key.
                    let mut group_map: Option<Arc<Mutex<Map>>> = None;
                    for g in 0..result.len() {
                        let ValueKind::Map(existing) = &result[g].kind else {
                            return Err(LanaError::Type);
                        };
                        let existing_guard = existing.lock().unwrap();
                        let existing_key = existing_guard.get("key").cloned().ok_or(LanaError::Type)?;
                        if set_value_equal(&existing_key, &key_value) {
                            group_map = Some(existing.clone());
                            break;
                        }
                    }
                    let group_rows: Arc<Mutex<Array>>;
                    if let Some(gm) = group_map {
                        let guard = gm.lock().unwrap();
                        let rows_value = guard.get("rows").cloned().ok_or(LanaError::Type)?;
                        let ValueKind::Array(rows) = rows_value.kind else {
                            return Err(LanaError::Type);
                        };
                        group_rows = rows;
                        drop(guard);
                        group_rows.lock().unwrap().items.push(row);
                    } else {
                        let mut new_map = Map::new(2);
                        let new_rows = Arc::new(Mutex::new(Array { items: vec![row] }));
                        new_map.set(Arc::from("key"), key_value, false)?;
                        new_map.set(Arc::from("rows"), Value::array(new_rows.clone()), false)?;
                        result.push(Value::map(Arc::new(Mutex::new(new_map))));
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
                let agg_col: Option<Arc<str>> = if agg.items.len() >= 2 {
                    match &agg.items[1].kind {
                        ValueKind::String(s) => Some(s.clone()),
                        _ => return Err(LanaError::Type),
                    }
                } else {
                    None
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
                    let agg_value = if &**agg_op == "count" {
                        Value::number(rows.items.len() as f64)
                    } else {
                        let col = agg_col.as_ref().ok_or(LanaError::Type)?;
                        let mut acc = 0.0;
                        let mut have = false;
                        for r in &rows.items {
                            let cell = Self::dataset_row_key(r, &**col)?;
                            let ValueKind::Number(cell_n) = cell.kind else {
                                return Err(LanaError::Type);
                            };
                            if !have {
                                acc = cell_n;
                                have = true;
                            } else if &**agg_op == "sum" || &**agg_op == "mean" {
                                acc += cell_n;
                            } else if &**agg_op == "max" {
                                if cell_n > acc { acc = cell_n; }
                            } else if &**agg_op == "min" {
                                if cell_n < acc { acc = cell_n; }
                            } else {
                                return Err(LanaError::Type);
                            }
                        }
                        if &**agg_op == "mean" {
                            if rows.items.is_empty() {
                                return Err(LanaError::Type);
                            }
                            acc /= rows.items.len() as f64;
                        }
                        Value::number(acc)
                    };
                    // Output row: {<group key name>: key_value, <op>: agg_value}.
                    let mut key_name = Arc::from("key");
                    if let ValueKind::Dataset(source) = &dataset.source.kind {
                        if source.op == DatasetOp::GroupBy {
                            if let ValueKind::String(sk) = &source.key.kind {
                                key_name = sk.clone();
                            }
                        }
                    }
                    let mut out_row = Map::new(2);
                    out_row.set(key_name, key_value, false)?;
                    out_row.set(agg_op.clone(), agg_value, false)?;
                    result.push(Value::map(Arc::new(Mutex::new(out_row))));
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
                let mut result = Vec::new();
                for left in &left_rows {
                    let left_key = Self::dataset_row_key(left, &**key)?;
                    for right in &right_rows {
                        let right_key = Self::dataset_row_key(right, &**key)?;
                        if !set_value_equal(&left_key, &right_key) {
                            continue;
                        }
                        // Merge left and right rows into one map.
                        let ValueKind::Map(left_map) = &left.kind else {
                            return Err(LanaError::Type);
                        };
                        let ValueKind::Map(right_map) = &right.kind else {
                            return Err(LanaError::Type);
                        };
                        let left_guard = left_map.lock().unwrap();
                        let right_guard = right_map.lock().unwrap();
                        let mut merged = Map::new(left_guard.entries.len() + right_guard.entries.len());
                        for entry in &left_guard.entries {
                            merged.set(entry.key.clone(), entry.value.clone(), false)?;
                        }
                        for entry in &right_guard.entries {
                            merged.set(entry.key.clone(), entry.value.clone(), false)?;
                        }
                        result.push(Value::map(Arc::new(Mutex::new(merged))));
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
        let mut map = Map::new(4);
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
    fn poll_composite_future(&mut self, future: Arc<Mutex<Future>>) {
        let (kind, inputs) = {
            let guard = future.lock().unwrap();
            let kind = guard.registers[0].clone();
            let inputs = guard.registers[1..].to_vec();
            (kind, inputs)
        };
        let ValueKind::String(kind) = kind.kind else {
            return;
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
                        return;
                    };
                    items.push(f.lock().unwrap().registers[0].clone());
                }
                result = Value::array(Arc::new(Mutex::new(Array { items })));
            }
        } else if &*kind == "race" {
            for item in &inputs {
                let ValueKind::Future(f) = &item.kind else {
                    return;
                };
                let f = f.lock().unwrap();
                if f.exhausted {
                    done = true;
                    result = f.registers[0].clone();
                    break;
                }
            }
        } else if &*kind == "sleep" {
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
            *out = self.make_result(false, Value::string(Arc::from("invalid JSON at byte 0")));
            return LanaError::Ok;
        }
        let mut pos = 0;
        let result = self.json_value(bytes, &mut pos, 0);
        json_space(bytes, &mut pos);
        let value = match result {
            Ok(value) if pos == bytes.len() => value,
            _ => {
                *out = self.make_result(
                    false,
                    Value::string(Arc::from(format!("invalid JSON at byte {pos}"))),
                );
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
        *out = self.make_result(true, rooted);
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
                    if map.set(Arc::from(key), item, false).is_err() {
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
                let start = *pos;
                let number = json_number(bytes, pos)?;
                if json_large_integer(&bytes[start..*pos]) {
                    let text = std::str::from_utf8(&bytes[start..*pos]).map_err(|_| LanaError::Parse)?;
                    Ok(Value::string(Arc::from(text)))
                } else {
                    Ok(Value::number(number))
                }
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
                let mut entries: Vec<&MapEntry> = map.entries.iter().collect();
                entries.sort_by(|a, b| a.key.cmp(&b.key));
                for (index, entry) in entries.iter().enumerate() {
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
            let snapshot = self.reactive_observe(&local_source, &local_evidence, 0u32)?;
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
    t: &Tensor,
) -> Result<Vec<f64>, LanaError> {
    let d = t.shape[0];
    if alloc(d * d * 2 * std::mem::size_of::<f64>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    if alloc(d * std::mem::size_of::<f64>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    let mut a = vec![0.0; d * d * 2];
    for i in 0..d {
        for j in 0..d {
            let (re, im) = linalg_get2(t, i, j);
            a[(i * d + j) * 2] = re;
            a[(i * d + j) * 2 + 1] = im;
        }
    }
    let mut eig = vec![0.0; d];
    linalg_jacobi(&mut a, d, &mut eig);
    Ok(eig)
}

/// Whether a 2-D tensor is positive semidefinite (all eigenvalues >= -1e-9).
fn linalg_is_psd(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    t: &Tensor,
) -> Result<bool, LanaError> {
    let eig = linalg_eigenvalues(alloc, t)?;
    Ok(eig.iter().all(|&e| e >= -1e-9))
}

/// density_operator from an N=1 STATE: the 2×2 matrix [[p, c], [c*, 1-p]].
fn linalg_density_from_state(
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
    rho: &Tensor,
    povm: &Tensor,
) -> Result<Value, LanaError> {
    let k = povm.shape[0];
    let d = povm.shape[1];
    if alloc(std::mem::size_of::<Array>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    let mut items = Vec::with_capacity(k);
    for i in 0..k {
        let mut re = 0.0;
        for r in 0..d {
            for c in 0..d {
                let (rho_re, rho_im) = linalg_get2(rho, r, c);
                let (e_re, e_im) = linalg_get3(povm, i, c, r);
                re += rho_re * e_re - rho_im * e_im;
            }
        }
        items.push(Value::number(re));
    }
    Ok(Value::array(Arc::new(Mutex::new(Array { items }))))
}

/// apply_to(chan, rho): Φ(ρ) = Σ_k K_k ρ K_k†.
fn linalg_apply_to(
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
    a: &Tensor,
    b: &Tensor,
) -> Result<Value, LanaError> {
    let d = a.shape[0];
    if alloc(d * d * 2 * std::mem::size_of::<f64>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    if alloc(d * std::mem::size_of::<f64>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    let mut diff = vec![0.0; d * d * 2];
    for i in 0..d {
        for j in 0..d {
            let (a_re, a_im) = linalg_get2(a, i, j);
            let (b_re, b_im) = linalg_get2(b, i, j);
            diff[(i * d + j) * 2] = a_re - b_re;
            diff[(i * d + j) * 2 + 1] = a_im - b_im;
        }
    }
    let mut eig = vec![0.0; d];
    linalg_jacobi(&mut diff, d, &mut eig);
    let sum: f64 = eig.iter().map(|&e| e.abs()).sum();
    Ok(Value::number(0.5 * sum))
}

/// is_separable(ab, bipartition): the PPT (Peres-Horodecki) criterion. A
/// negative partial-transpose eigenvalue proves entanglement; a PSD partial
/// transpose proves separability for 2×2 and 2×3 systems and is otherwise
/// inconclusive. `bipartition` is the qubit count of the first subsystem.
fn linalg_is_separable(
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    if alloc(d * d * 2 * std::mem::size_of::<f64>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    if alloc(d * std::mem::size_of::<f64>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    let mut pt = vec![0.0; d * d * 2];
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
    let mut eig = vec![0.0; d];
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    if alloc(d * d * 2 * std::mem::size_of::<f64>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    if alloc(d * std::mem::size_of::<f64>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    let mut a = data[..d * d * 2].to_vec();
    let mut eig = vec![0.0; d];
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
    alloc: &mut dyn FnMut(usize) -> LanaError,
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
