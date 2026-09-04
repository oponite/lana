//! C ABI for the Rust Lana runtime, mirroring the C11 public API headers
//! (`vm/include/`, `runtime/include/`, `tools/include/`) so existing C hosts can
//! link against the Rust runtime without source changes.
//!
//! # Representation decisions
//!
//! * **`Value` is an opaque handle** (`*mut lana_vm::value::Value`), not the
//!   C by-value tagged union. The Rust `Value` is an owned, `Arc`/`Mutex`-based
//!   graph (derivations, reactive state, claims, planned effects, maps, arrays,
//!   tasks, ...) that cannot be passed by value across the FFI boundary. A
//!   `#[repr(C)]` mirror would mean re-implementing the entire C value model as
//!   parallel structs. So every C function that takes or returns `Value` by
//!   value instead takes or returns `*mut Value` here, and `lana_value_free`
//!   releases a handle. This is the one systematic ABI deviation; all function
//!   *names* are preserved.
//! * **Stateful objects are opaque handles** (`Box<T>` -> `*mut T`) with
//!   `*_free`/`*_close`/`*_destroy` functions: `Vm`, `Store`, `Chunk`,
//!   `Sha256`, `Buffer`, `Adapter`, `Map`.
//! * **Error codes are `i32`** and match the C `LanaError` enum exactly
//!   (`lana_bytecode::LanaError` is `#[repr(i32)]`, discriminants 0..=44).
//!   `lana_error_name`/`lana_error_kind_name` return the same strings as
//!   `vm/c/error.c`.
//! * **`LanaState` is `#[repr(C)]`** (three `f64`: `p`, `d_re`, `d_im`) and is
//!   passed by value, matching `vm/include/state.h`.
//!
//! # Known gaps (not ported)
//!
//! * `lana_error_set*` (varargs / fixed-size `LanaErrorInfo` struct mutation).
//! * Heap-typed value constructors (`lana_value_state_dist`, `_map`, `_array`,
//!   `_possibility`, `_paths`, `_shared_capability`) — need the C heap structs.
//! * `lana_transform_spec` (returns a function-pointer struct).
//! * `lana_chunk_*` init/emit/write/read and `lana_disassemble*` (C `LanaChunk`
//!   raw-pointer layout and `FILE*` output; the Rust `Chunk` is a different
//!   model). `lana_opcode_name` is provided.
//! * `lana_vm_init` and the internal VM ops (`alloc`, `clone`, `collect`,
//!   `root_*`, `write_barrier`, `random`, `state_dist_*`, `joint_*`,
//!   `possibility_build`, `information_*`, `reactive_*`, `claim`,
//!   `planned_effect`, `provenance`, `derivation`, `explain`) — the Rust `Vm`
//!   borrows its `Chunk` immutably for its lifetime and does not expose these
//!   as public methods.
//! * `lana_store_scan`/`_history`/`_snapshot`/`_put_persistent_state`/
//!   `_get_persistent_state(_at)` (records embed by-value `Value`; persistent
//!   state struct is nested).
//! * `lana_policy_store`/`_evaluate`/`_store_decision`/`_replay` (struct-heavy
//!   `LanaPolicyEvaluation`/`LanaDecision`). `lana_policy_version` is provided.
//! * `lana_ledger_*`, `lana_claim_*`/`lana_relationship_*`, `lana_effect_*`
//!   (struct-heavy, and `LanaEffectExecutor` is a C function pointer vs the
//!   Rust `Box<dyn Fn>` closure type).
//! * `lana_persistent_state_encode`/`_decode`/`_free` (nested struct).
//!
//! # Memory notes
//!
//! * `lana_vm_create` leaks the (empty) `Chunk` it borrows; `lana_vm_free`
//!   cannot reclaim it because the `Vm` holds a `&'static Chunk`. One small
//!   chunk is leaked per VM, which is bounded and acceptable at an FFI edge.
//! * Strings returned by `lana_store_get_path` must be released with
//!   `lana_string_free` (not C `free`). Byte buffers returned by
//!   `lana_state_encode` must be released with `lana_bytes_free`.

use std::ffi::{c_char, c_void, CStr, CString};
use std::os::raw::c_int;
use std::ptr;
use std::sync::Arc;
use std::sync::LazyLock;

use lana_bytecode::opcode::LABC_VERSION;
use lana_bytecode::{Chunk, LanaError, OpCode};
use lana_vm::state::{self, Indexes, State, StateValue};
use lana_vm::value::{Map, Value, VmError};
use lana_vm::vm::Vm;

use lana_runtime::adapters::{adapter_fetch, adapter_load, Adapter, AdapterKind, AdapterOptions};
use lana_runtime::codec::{decode_document, decode_value, encode_value};
use lana_runtime::data::{csv_read, csv_write, json_parse, json_stringify};
use lana_runtime::policy::{policy_version, Policy, PolicyRule, PolicyRuleKind};
use lana_runtime::sha256::{sha256, Sha256};
use lana_runtime::state_codec::{state_decode, state_encode};
use lana_runtime::store::{
    store_close, store_commit, store_compact, store_current_revision, store_delete, store_get,
    store_get_at, store_get_path, store_open, store_put, Store, StoreOptions,
};

// ---------------------------------------------------------------------------
// C struct mirrors
// ---------------------------------------------------------------------------

/// Mirrors `LanaState` in `vm/include/state.h` (three `f64`; the C union of
/// `d_re`/`d` is a single field here).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LanaState {
    pub p: f64,
    pub d_re: f64,
    pub d_im: f64,
}

/// Mirrors `LanaStoreOptions` in `runtime/include/store.h`.
#[repr(C)]
pub struct LanaStoreOptions {
    pub struct_size: usize,
    pub schema_version: u32,
    pub path: *const c_char,
    pub timeout_ms: u32,
}

/// Mirrors `LanaStoreRevisionInfo` in `runtime/include/store.h`.
#[repr(C)]
pub struct LanaStoreRevisionInfo {
    pub revision_id: u64,
    pub schema_version: u32,
    pub timestamp: u64,
    pub digest: [u8; 32],
}

/// Mirrors `LanaAdapterOptions` in `runtime/include/adapters.h`.
#[repr(C)]
pub struct LanaAdapterOptions {
    pub struct_size: usize,
    pub schema_version: u32,
    pub kind: i32,
    pub config: *const c_char,
}

/// Mirrors `LanaPolicyRule` in `runtime/include/policy.h`.
#[repr(C)]
pub struct LanaPolicyRule {
    pub struct_size: usize,
    pub schema_version: u32,
    pub kind: i32,
    pub field: *const c_char,
    pub threshold: f64,
    pub expected: *const c_char,
    pub effect: *const c_char,
}

/// Mirrors `LanaPolicy` in `runtime/include/policy.h`.
#[repr(C)]
pub struct LanaPolicy {
    pub struct_size: usize,
    pub schema_version: u32,
    pub policy_id: *const c_char,
    pub rule: LanaPolicyRule,
}

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Owns a `Vm` that borrows a leaked `Chunk`.
pub struct VmHandle {
    vm: Vm<'static>,
}

/// Growable byte buffer backing the codec functions (opaque `LanaBuffer`).
pub struct Buffer {
    data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

unsafe fn cstr<'a>(ptr: *const c_char) -> &'a str {
    if ptr.is_null() {
        return "";
    }
    CStr::from_ptr(ptr).to_str().unwrap_or("")
}

unsafe fn state_from_c(s: *const LanaState) -> State {
    State {
        p: (*s).p,
        d_re: (*s).d_re,
        d_im: (*s).d_im,
    }
}

unsafe fn state_to_c(s: &State, out: *mut LanaState) {
    (*out).p = s.p;
    (*out).d_re = s.d_re;
    (*out).d_im = s.d_im;
}

fn leak_bytes(data: &[u8]) -> *mut c_void {
    let boxed: Box<[u8]> = data.to_vec().into_boxed_slice();
    Box::into_raw(boxed) as *mut u8 as *mut c_void
}

// ---------------------------------------------------------------------------
// error.h
// ---------------------------------------------------------------------------

const ERROR_KIND_NAMES: [&CStr; 13] = [
    c"none",
    c"validation",
    c"type",
    c"parse",
    c"bytecode",
    c"io",
    c"resource-limit",
    c"task",
    c"cancellation",
    c"timeout",
    c"unsupported",
    c"resolution",
    c"assertion",
];

const RESOLUTION_REASON_NAMES: [&CStr; 8] = [
    c"none",
    c"no-alternatives",
    c"multiple-alternatives",
    c"contradiction",
    c"invalid-conditioning",
    c"unsupported-exact",
    c"cancelled",
    c"resource-limit",
];

const EXACT_SUPPORT_NAMES: [&CStr; 3] = [c"unknown", c"available", c"unavailable"];

const RESOURCE_KIND_NAMES: [&CStr; 7] = [
    c"none",
    c"memory",
    c"instructions",
    c"tasks",
    c"paths",
    c"samples",
    c"time",
];

const VALUE_TYPE_NAMES: [&CStr; 16] = [
    c"null",
    c"number",
    c"bool",
    c"string",
    c"state",
    c"distribution",
    c"sample",
    c"joint_state",
    c"array",
    c"function",
    c"task",
    c"state_dist",
    c"map",
    c"possibility",
    c"paths",
    c"shared_capability",
];

/// NUL-terminated `LanaError` names, built once from `LanaError::name`.
static ERROR_NAMES: LazyLock<[CString; 46]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let e: LanaError = unsafe { std::mem::transmute::<i32, LanaError>(i as i32) };
        CString::new(e.name()).expect("error name contains no NUL")
    })
});

/// NUL-terminated opcode mnemonics, built once from `OpCode::name`.
static OPCODE_NAMES: LazyLock<[CString; OpCode::Count as usize]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let op = OpCode::try_from(i as u8).expect("valid opcode");
        CString::new(op.name()).expect("opcode name contains no NUL")
    })
});

fn error_from_i32(code: i32) -> Option<LanaError> {
    if (0..=45).contains(&code) {
        Some(unsafe { std::mem::transmute::<i32, LanaError>(code) })
    } else {
        None
    }
}

/// Mirrors `lana_error_name`.
#[no_mangle]
pub extern "C" fn lana_error_name(error: i32) -> *const c_char {
    match error_from_i32(error) {
        Some(e) => ERROR_NAMES[e as usize].as_ptr(),
        None => c"LANA_ERR_UNKNOWN".as_ptr(),
    }
}

/// Mirrors `lana_error_kind_from_code`. Delegates to `LanaError::kind_name`,
/// which mirrors the C switch in `vm/c/error.c`, then maps the name to its
/// `LanaErrorKind` index.
#[no_mangle]
pub extern "C" fn lana_error_kind_from_code(error: i32) -> i32 {
    let Some(e) = error_from_i32(error) else {
        return 0; // LANA_ERROR_KIND_NONE
    };
    let name = e.kind_name();
    ERROR_KIND_NAMES
        .iter()
        .position(|&n| n.to_bytes() == name.as_bytes())
        .unwrap_or(0) as i32
}

/// Mirrors `lana_error_kind_name`.
#[no_mangle]
pub extern "C" fn lana_error_kind_name(kind: i32) -> *const c_char {
    ERROR_KIND_NAMES
        .get(kind as usize)
        .copied()
        .unwrap_or(c"none")
        .as_ptr()
}

/// Mirrors `lana_resolution_reason_name`.
#[no_mangle]
pub extern "C" fn lana_resolution_reason_name(reason: i32) -> *const c_char {
    RESOLUTION_REASON_NAMES
        .get(reason as usize)
        .copied()
        .unwrap_or(c"none")
        .as_ptr()
}

/// Mirrors `lana_exact_support_name`.
#[no_mangle]
pub extern "C" fn lana_exact_support_name(support: i32) -> *const c_char {
    EXACT_SUPPORT_NAMES
        .get(support as usize)
        .copied()
        .unwrap_or(c"unknown")
        .as_ptr()
}

/// Mirrors `lana_resource_kind_name`.
#[no_mangle]
pub extern "C" fn lana_resource_kind_name(resource: i32) -> *const c_char {
    RESOURCE_KIND_NAMES
        .get(resource as usize)
        .copied()
        .unwrap_or(c"none")
        .as_ptr()
}

// ---------------------------------------------------------------------------
// value.h
// ---------------------------------------------------------------------------

/// Mirrors `lana_value_null` (returns an opaque `*mut Value`).
#[no_mangle]
pub extern "C" fn lana_value_null() -> *mut Value {
    Box::into_raw(Box::new(Value::null()))
}

/// Mirrors `lana_value_number` (returns an opaque `*mut Value`).
#[no_mangle]
pub extern "C" fn lana_value_number(number: f64) -> *mut Value {
    Box::into_raw(Box::new(Value::number(number)))
}

/// Mirrors `lana_value_bool` (returns an opaque `*mut Value`).
#[no_mangle]
pub extern "C" fn lana_value_bool(boolean: bool) -> *mut Value {
    Box::into_raw(Box::new(Value::boolean(boolean)))
}

/// Mirrors `lana_value_string` (returns an opaque `*mut Value`).
#[no_mangle]
pub unsafe extern "C" fn lana_value_string(string: *const c_char) -> *mut Value {
    Box::into_raw(Box::new(Value::string(Arc::from(cstr(string)))))
}

/// Mirrors `lana_value_state` (returns an opaque `*mut Value`).
#[no_mangle]
pub extern "C" fn lana_value_state(state: LanaState) -> *mut Value {
    let sv = StateValue {
        state: State {
            p: state.p,
            d_re: state.d_re,
            d_im: state.d_im,
        },
        indexes: Indexes::default(),
    };
    Box::into_raw(Box::new(Value::state(sv)))
}

/// Mirrors `lana_value_distribution` (returns an opaque `*mut Value`).
#[no_mangle]
pub extern "C" fn lana_value_distribution(p0: f64, p1: f64) -> *mut Value {
    Box::into_raw(Box::new(Value::distribution(p0, p1)))
}

/// Mirrors `lana_value_sample` (returns an opaque `*mut Value`).
#[no_mangle]
pub extern "C" fn lana_value_sample(sample: i32) -> *mut Value {
    Box::into_raw(Box::new(Value::sample(sample)))
}

/// Mirrors `lana_value_type_name`.
#[no_mangle]
pub extern "C" fn lana_value_type_name(type_: i32) -> *const c_char {
    VALUE_TYPE_NAMES
        .get(type_ as usize)
        .copied()
        .unwrap_or(c"unknown")
        .as_ptr()
}

/// Mirrors `lana_value_print` (writes to stdout without a trailing newline).
#[no_mangle]
pub unsafe extern "C" fn lana_value_print(value: *const Value) {
    if value.is_null() {
        return;
    }
    let v = &*value;
    print!("{}", v.print());
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

/// Releases an opaque `*mut Value` handle (extension; C passes `Value` by value).
#[no_mangle]
pub unsafe extern "C" fn lana_value_free(value: *mut Value) {
    if !value.is_null() {
        drop(Box::from_raw(value));
    }
}

// ---------------------------------------------------------------------------
// state.h
// ---------------------------------------------------------------------------

/// Mirrors `lana_state_valid`.
#[no_mangle]
pub unsafe extern "C" fn lana_state_valid(state: *const LanaState) -> bool {
    if state.is_null() {
        return false;
    }
    state::state_valid(&state_from_c(state))
}

/// Mirrors `lana_state_make_complex`.
#[no_mangle]
pub unsafe extern "C" fn lana_state_make_complex(
    p: f64,
    d_re: f64,
    d_im: f64,
    out: *mut LanaState,
) -> i32 {
    if out.is_null() {
        return LanaError::InvalidState as i32;
    }
    let mut s = State::default();
    let code = state::make_complex(p, d_re, d_im, &mut s);
    state_to_c(&s, out);
    code as i32
}

/// Mirrors `lana_state_make`.
#[no_mangle]
pub unsafe extern "C" fn lana_state_make(p: f64, d: f64, out: *mut LanaState) -> i32 {
    if out.is_null() {
        return LanaError::InvalidState as i32;
    }
    let mut s = State::default();
    let code = state::make(p, d, &mut s);
    state_to_c(&s, out);
    code as i32
}

/// Mirrors `lana_state_disposition_squared`.
#[no_mangle]
pub unsafe extern "C" fn lana_state_disposition_squared(state: *const LanaState) -> f64 {
    if state.is_null() {
        return 0.0;
    }
    state::disposition_squared(&state_from_c(state))
}

/// Mirrors `lana_state_reconstruct_c`.
#[no_mangle]
pub unsafe extern "C" fn lana_state_reconstruct_c(
    state: *const LanaState,
    c_re: *mut f64,
    c_im: *mut f64,
) {
    if state.is_null() || c_re.is_null() || c_im.is_null() {
        return;
    }
    let mut a = 0.0;
    let mut b = 0.0;
    state::reconstruct_c(&state_from_c(state), &mut a, &mut b);
    *c_re = a;
    *c_im = b;
}

/// Mirrors `lana_state_basis_probability`.
#[no_mangle]
pub unsafe extern "C" fn lana_state_basis_probability(
    basis: u32,
    state: *const LanaState,
    out: *mut f64,
) -> i32 {
    if state.is_null() || out.is_null() {
        return LanaError::InvalidState as i32;
    }
    let mut o = 0.0;
    let code = state::basis_probability(basis, &state_from_c(state), &mut o);
    *out = o;
    code as i32
}

/// Mirrors `lana_state_append_parameters`.
#[no_mangle]
pub unsafe extern "C" fn lana_state_append_parameters(
    left: *const LanaState,
    right: *const LanaState,
    p: *mut f64,
    m_re: *mut f64,
    m_im: *mut f64,
    sigma: *mut f64,
) -> i32 {
    if left.is_null() || right.is_null() || p.is_null() || m_re.is_null() || m_im.is_null() || sigma.is_null() {
        return LanaError::InvalidState as i32;
    }
    let mut p_ = 0.0;
    let mut m_re_ = 0.0;
    let mut m_im_ = 0.0;
    let mut sigma_ = 0.0;
    let code = state::append_parameters(
        &state_from_c(left),
        &state_from_c(right),
        &mut p_,
        &mut m_re_,
        &mut m_im_,
        &mut sigma_,
    );
    *p = p_;
    *m_re = m_re_;
    *m_im = m_im_;
    *sigma = sigma_;
    code as i32
}

/// Mirrors `lana_state_mix`.
#[no_mangle]
pub unsafe extern "C" fn lana_state_mix(
    left: *const LanaState,
    right: *const LanaState,
    w: f64,
    out: *mut LanaState,
) -> i32 {
    if left.is_null() || right.is_null() || out.is_null() {
        return LanaError::InvalidState as i32;
    }
    let mut o = State::default();
    let code = state::mix(&state_from_c(left), &state_from_c(right), w, &mut o);
    state_to_c(&o, out);
    code as i32
}

/// Mirrors `lana_transform_apply`.
#[no_mangle]
pub unsafe extern "C" fn lana_transform_apply(
    identifier: u32,
    source: *const LanaState,
    out: *mut LanaState,
) -> i32 {
    if source.is_null() || out.is_null() {
        return LanaError::InvalidState as i32;
    }
    let mut o = State::default();
    let code = state::transform_apply(identifier, &state_from_c(source), &mut o);
    state_to_c(&o, out);
    code as i32
}

/// Mirrors `lana_transform_expected_probability`.
#[no_mangle]
pub unsafe extern "C" fn lana_transform_expected_probability(
    identifier: u32,
    input: f64,
    out: *mut f64,
) -> i32 {
    if out.is_null() {
        return LanaError::InvalidState as i32;
    }
    let mut o = 0.0;
    let code = state::transform_expected_probability(identifier, input, &mut o);
    *out = o;
    code as i32
}

// ---------------------------------------------------------------------------
// sha256.h
// ---------------------------------------------------------------------------

/// Allocates an opaque `LanaSha256` context (extension; C stack-allocates the
/// struct, which is not possible with an opaque handle).
#[no_mangle]
pub extern "C" fn lana_sha256_new() -> *mut Sha256 {
    Box::into_raw(Box::new(Sha256::new()))
}

/// Mirrors `lana_sha256_init` (resets an opaque context).
#[no_mangle]
pub unsafe extern "C" fn lana_sha256_init(context: *mut Sha256) {
    if context.is_null() {
        return;
    }
    *context = Sha256::new();
}

/// Mirrors `lana_sha256_update`.
#[no_mangle]
pub unsafe extern "C" fn lana_sha256_update(context: *mut Sha256, data: *const c_void, length: usize) {
    if context.is_null() || (data.is_null() && length != 0) {
        return;
    }
    let slice = std::slice::from_raw_parts(data as *const u8, length);
    (*context).update(slice);
}

/// Mirrors `lana_sha256_final`.
#[no_mangle]
pub unsafe extern "C" fn lana_sha256_final(context: *mut Sha256, out: *mut u8) {
    if context.is_null() || out.is_null() {
        return;
    }
    let mut digest = [0u8; 32];
    (*context).finalize(&mut digest);
    ptr::copy_nonoverlapping(digest.as_ptr(), out, 32);
}

/// Mirrors `lana_sha256` (one-shot).
#[no_mangle]
pub unsafe extern "C" fn lana_sha256(data: *const c_void, length: usize, out: *mut u8) {
    if (data.is_null() && length != 0) || out.is_null() {
        return;
    }
    let slice = std::slice::from_raw_parts(data as *const u8, length);
    let digest = sha256(slice);
    ptr::copy_nonoverlapping(digest.as_ptr(), out, 32);
}

/// Releases an opaque `LanaSha256` context (extension).
#[no_mangle]
pub unsafe extern "C" fn lana_sha256_free(context: *mut Sha256) {
    if !context.is_null() {
        drop(Box::from_raw(context));
    }
}

// ---------------------------------------------------------------------------
// codec.h
// ---------------------------------------------------------------------------

/// Allocates an opaque `LanaBuffer` (extension; C stack-allocates the struct).
#[no_mangle]
pub extern "C" fn lana_buffer_new() -> *mut Buffer {
    Box::into_raw(Box::new(Buffer { data: Vec::new() }))
}

/// Releases an opaque `LanaBuffer` (extension).
#[no_mangle]
pub unsafe extern "C" fn lana_buffer_free(buffer: *mut Buffer) {
    if !buffer.is_null() {
        drop(Box::from_raw(buffer));
    }
}

/// Returns the buffer's data pointer (extension).
#[no_mangle]
pub unsafe extern "C" fn lana_buffer_data(buffer: *const Buffer) -> *const u8 {
    if buffer.is_null() {
        return ptr::null();
    }
    (*buffer).data.as_ptr()
}

/// Returns the buffer's length (extension).
#[no_mangle]
pub unsafe extern "C" fn lana_buffer_length(buffer: *const Buffer) -> usize {
    if buffer.is_null() {
        return 0;
    }
    (*buffer).data.len()
}

/// Mirrors `lana_codec_encode_value` (appends JSON to the buffer).
#[no_mangle]
pub unsafe extern "C" fn lana_codec_encode_value(buffer: *mut Buffer, value: *const Value) -> i32 {
    if buffer.is_null() || value.is_null() {
        return LanaError::InvalidState as i32;
    }
    let buf = &mut *buffer;
    let v = &*value;
    match encode_value(v) {
        Ok(s) => {
            buf.data.extend_from_slice(s.as_bytes());
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_codec_decode_value` (advances `offset`, writes an opaque value).
#[no_mangle]
pub unsafe extern "C" fn lana_codec_decode_value(
    buffer: *const Buffer,
    offset: *mut usize,
    out_value: *mut *mut Value,
) -> i32 {
    if buffer.is_null() || offset.is_null() || out_value.is_null() {
        return LanaError::InvalidState as i32;
    }
    let buf = &*buffer;
    let off = &mut *offset;
    match decode_value(&buf.data, off) {
        Ok(v) => {
            *out_value = Box::into_raw(Box::new(v));
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_codec_decode_document` (writes an opaque value).
#[no_mangle]
pub unsafe extern "C" fn lana_codec_decode_document(
    buffer: *const Buffer,
    out_value: *mut *mut Value,
) -> i32 {
    if buffer.is_null() || out_value.is_null() {
        return LanaError::InvalidState as i32;
    }
    let buf = &*buffer;
    match decode_document(&buf.data) {
        Ok(v) => {
            *out_value = Box::into_raw(Box::new(v));
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

// ---------------------------------------------------------------------------
// bytecode.h
// ---------------------------------------------------------------------------

/// Mirrors `lana_opcode_name`.
#[no_mangle]
pub extern "C" fn lana_opcode_name(opcode: u8) -> *const c_char {
    match OpCode::try_from(opcode) {
        Ok(op) => OPCODE_NAMES[op as usize].as_ptr(),
        Err(()) => c"unknown".as_ptr(),
    }
}

// ---------------------------------------------------------------------------
// vm.h
// ---------------------------------------------------------------------------

/// Mirrors `lana_vm_create` (creates a VM over an empty chunk).
#[no_mangle]
pub extern "C" fn lana_vm_create() -> *mut VmHandle {
    let chunk: &'static Chunk = Box::leak(Box::new(Chunk::new(LABC_VERSION, 0)));
    let vm = Vm::new(chunk);
    Box::into_raw(Box::new(VmHandle { vm }))
}

/// Creates a VM over a caller-provided chunk (extension; the chunk handle's
/// ownership transfers to the VM and is leaked on free).
#[no_mangle]
pub unsafe extern "C" fn lana_vm_create_with_chunk(chunk: *mut Chunk) -> *mut VmHandle {
    if chunk.is_null() {
        return ptr::null_mut();
    }
    let chunk_box = Box::from_raw(chunk);
    let chunk_ref: &'static Chunk = Box::leak(chunk_box);
    let vm = Vm::new(chunk_ref);
    Box::into_raw(Box::new(VmHandle { vm }))
}

/// Mirrors `lana_vm_seed`.
#[no_mangle]
pub unsafe extern "C" fn lana_vm_seed(vm: *mut VmHandle, seed: u64) {
    if vm.is_null() {
        return;
    }
    (*vm).vm.seed(seed);
}

/// Mirrors `lana_vm_set_program_args`.
#[no_mangle]
pub unsafe extern "C" fn lana_vm_set_program_args(
    vm: *mut VmHandle,
    argc: c_int,
    argv: *const *const c_char,
) {
    if vm.is_null() {
        return;
    }
    let mut args = Vec::with_capacity(argc.max(0) as usize);
    for i in 0..argc.max(0) as usize {
        args.push(cstr(*argv.add(i)).to_string());
    }
    (*vm).vm.set_program_args(&args);
}

/// Mirrors `lana_vm_set_worker_count`.
#[no_mangle]
pub unsafe extern "C" fn lana_vm_set_worker_count(vm: *mut VmHandle, workers: usize) -> i32 {
    if vm.is_null() {
        return LanaError::InvalidState as i32;
    }
    (*vm).vm.set_worker_count(workers) as i32
}

/// Mirrors `lana_vm_set_task_limit`.
#[no_mangle]
pub unsafe extern "C" fn lana_vm_set_task_limit(vm: *mut VmHandle, tasks: usize) -> i32 {
    if vm.is_null() {
        return LanaError::InvalidState as i32;
    }
    (*vm).vm.set_task_limit(tasks) as i32
}

/// Sets the instruction limit (extension; not in the C header).
#[no_mangle]
pub unsafe extern "C" fn lana_vm_set_instruction_limit(vm: *mut VmHandle, limit: u64) {
    if vm.is_null() {
        return;
    }
    (*vm).vm.set_instruction_limit(limit);
}

/// Sets the memory limit (extension; not in the C header).
#[no_mangle]
pub unsafe extern "C" fn lana_vm_set_memory_limit(vm: *mut VmHandle, limit: usize) {
    if vm.is_null() {
        return;
    }
    (*vm).vm.set_memory_limit(limit);
}

/// Mirrors `lana_vm_run`.
#[no_mangle]
pub unsafe extern "C" fn lana_vm_run(vm: *mut VmHandle) -> i32 {
    if vm.is_null() {
        return LanaError::InvalidState as i32;
    }
    (*vm).vm.run() as i32
}

/// Returns the VM's result value (extension; the C struct exposes `result`).
#[no_mangle]
pub unsafe extern "C" fn lana_vm_result(vm: *const VmHandle) -> *const Value {
    if vm.is_null() {
        return ptr::null();
    }
    (*vm).vm.result() as *const Value
}

/// Returns the VM's error info (extension; the C struct exposes `error`).
#[no_mangle]
pub unsafe extern "C" fn lana_vm_error(vm: *const VmHandle) -> *const VmError {
    if vm.is_null() {
        return ptr::null();
    }
    (*vm).vm.error() as *const VmError
}

/// Mirrors `lana_vm_free`.
#[no_mangle]
pub unsafe extern "C" fn lana_vm_free(vm: *mut VmHandle) {
    if vm.is_null() {
        return;
    }
    drop(Box::from_raw(vm));
}

/// Mirrors `lana_vm_destroy` (alias of `lana_vm_free`).
#[no_mangle]
pub unsafe extern "C" fn lana_vm_destroy(vm: *mut VmHandle) {
    lana_vm_free(vm);
}

// ---------------------------------------------------------------------------
// data.h
// ---------------------------------------------------------------------------

/// Mirrors `lana_map_new` (the VM argument is unused; Rust maps are GC-free).
#[no_mangle]
pub unsafe extern "C" fn lana_map_new(
    _vm: *mut VmHandle,
    capacity: usize,
    out: *mut *mut Map,
) -> i32 {
    if out.is_null() {
        return LanaError::InvalidState as i32;
    }
    *out = Box::into_raw(Box::new(Map::new(capacity)));
    LanaError::Ok as i32
}

/// Mirrors `lana_map_get` (writes an opaque value).
#[no_mangle]
pub unsafe extern "C" fn lana_map_get(
    map: *const Map,
    key: *const c_char,
    out: *mut *mut Value,
) -> i32 {
    if map.is_null() || out.is_null() {
        return LanaError::InvalidState as i32;
    }
    let m = &*map;
    let key = cstr(key);
    match m.get(key) {
        Some(v) => {
            *out = Box::into_raw(Box::new(v.clone()));
            LanaError::Ok as i32
        }
        None => LanaError::NotFound as i32,
    }
}

/// Mirrors `lana_map_has` (returns the entry index or -1).
#[no_mangle]
pub unsafe extern "C" fn lana_map_has(map: *const Map, key: *const c_char) -> isize {
    if map.is_null() {
        return -1;
    }
    let m = &*map;
    let key = cstr(key);
    match m.entries.iter().position(|e| &*e.key == key) {
        Some(i) => i as isize,
        None => -1,
    }
}

/// Mirrors `lana_map_set` (the VM argument is unused).
#[no_mangle]
pub unsafe extern "C" fn lana_map_set(
    _vm: *mut VmHandle,
    map: *mut Map,
    key: *const c_char,
    value: *const Value,
    reject_existing: bool,
) -> i32 {
    if map.is_null() || value.is_null() {
        return LanaError::InvalidState as i32;
    }
    let m = &mut *map;
    let key = cstr(key);
    let v = &*value;
    match m.set(Arc::from(key), v.clone(), reject_existing) {
        Ok(()) => LanaError::Ok as i32,
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_json_parse` (the VM argument is unused).
#[no_mangle]
pub unsafe extern "C" fn lana_json_parse(
    _vm: *mut VmHandle,
    text: *const c_char,
    out: *mut *mut Value,
) -> i32 {
    if out.is_null() {
        return LanaError::InvalidState as i32;
    }
    match json_parse(cstr(text)) {
        Ok(v) => {
            *out = Box::into_raw(Box::new(v));
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_json_stringify` (writes a string value).
#[no_mangle]
pub unsafe extern "C" fn lana_json_stringify(
    _vm: *mut VmHandle,
    value: *const Value,
    out: *mut *mut Value,
) -> i32 {
    if value.is_null() || out.is_null() {
        return LanaError::InvalidState as i32;
    }
    match json_stringify(&*value) {
        Ok(s) => {
            *out = Box::into_raw(Box::new(Value::string(Arc::from(s.as_str()))));
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_csv_read` (the VM argument is unused).
#[no_mangle]
pub unsafe extern "C" fn lana_csv_read(
    _vm: *mut VmHandle,
    path: *const c_char,
    out: *mut *mut Value,
) -> i32 {
    if out.is_null() {
        return LanaError::InvalidState as i32;
    }
    match csv_read(cstr(path)) {
        Ok(v) => {
            *out = Box::into_raw(Box::new(v));
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_csv_write` (the VM argument is unused).
#[no_mangle]
pub unsafe extern "C" fn lana_csv_write(
    _vm: *mut VmHandle,
    path: *const c_char,
    rows: *const Value,
    out: *mut *mut Value,
) -> i32 {
    if rows.is_null() || out.is_null() {
        return LanaError::InvalidState as i32;
    }
    match csv_write(cstr(path), &*rows) {
        Ok(v) => {
            *out = Box::into_raw(Box::new(v));
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

// ---------------------------------------------------------------------------
// adapters.h
// ---------------------------------------------------------------------------

/// Mirrors `lana_adapter_load`.
#[no_mangle]
pub unsafe extern "C" fn lana_adapter_load(
    options: *const LanaAdapterOptions,
    out_adapter: *mut *mut Adapter,
) -> i32 {
    if options.is_null() || out_adapter.is_null() {
        return LanaError::InvalidState as i32;
    }
    let opts = &*options;
    let kind = match opts.kind {
        0 => AdapterKind::Json,
        1 => AdapterKind::Csv,
        2 => AdapterKind::Sqlite,
        3 => AdapterKind::HttpJson,
        _ => return LanaError::Schema as i32,
    };
    let rust_opts = AdapterOptions {
        schema_version: opts.schema_version,
        kind,
        config: if opts.config.is_null() {
            None
        } else {
            Some(Arc::from(cstr(opts.config)))
        },
    };
    match adapter_load(&rust_opts) {
        Ok(a) => {
            *out_adapter = Box::into_raw(Box::new(a));
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_adapter_fetch` (the VM argument is unused).
#[no_mangle]
pub unsafe extern "C" fn lana_adapter_fetch(
    adapter: *const Adapter,
    _vm: *mut VmHandle,
    query: *const c_char,
    out: *mut *mut Value,
) -> i32 {
    if adapter.is_null() || out.is_null() {
        return LanaError::InvalidState as i32;
    }
    match adapter_fetch(&*adapter, cstr(query)) {
        Ok(v) => {
            *out = Box::into_raw(Box::new(v));
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_adapter_close`.
#[no_mangle]
pub unsafe extern "C" fn lana_adapter_close(adapter: *mut Adapter) {
    if !adapter.is_null() {
        drop(Box::from_raw(adapter));
    }
}

// ---------------------------------------------------------------------------
// state_codec.h
// ---------------------------------------------------------------------------

/// Mirrors `lana_state_encode` (writes a byte buffer; free with `lana_bytes_free`).
#[no_mangle]
pub unsafe extern "C" fn lana_state_encode(
    state: LanaState,
    out_buf: *mut *mut c_void,
    out_len: *mut usize,
) -> i32 {
    if out_buf.is_null() || out_len.is_null() {
        return LanaError::InvalidState as i32;
    }
    let s = State {
        p: state.p,
        d_re: state.d_re,
        d_im: state.d_im,
    };
    match state_encode(&s) {
        Ok(bytes) => {
            *out_len = bytes.len();
            *out_buf = leak_bytes(&bytes);
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_state_decode`.
#[no_mangle]
pub unsafe extern "C" fn lana_state_decode(
    buf: *const c_void,
    len: usize,
    out_state: *mut LanaState,
) -> i32 {
    if (buf.is_null() && len != 0) || out_state.is_null() {
        return LanaError::InvalidState as i32;
    }
    let slice = std::slice::from_raw_parts(buf as *const u8, len);
    match state_decode(slice) {
        Ok(s) => {
            state_to_c(&s, out_state);
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

// ---------------------------------------------------------------------------
// store.h
// ---------------------------------------------------------------------------

/// Mirrors `lana_store_open`.
#[no_mangle]
pub unsafe extern "C" fn lana_store_open(
    options: *const LanaStoreOptions,
    out_store: *mut *mut Store,
) -> i32 {
    if options.is_null() || out_store.is_null() {
        return LanaError::InvalidState as i32;
    }
    let opts = &*options;
    let rust_opts = StoreOptions {
        schema_version: opts.schema_version,
        path: cstr(opts.path).to_string(),
        timeout_ms: opts.timeout_ms,
    };
    match store_open(&rust_opts) {
        Ok(store) => {
            *out_store = Box::into_raw(Box::new(store));
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_store_close` (also frees the handle).
#[no_mangle]
pub unsafe extern "C" fn lana_store_close(store: *mut Store) -> i32 {
    if store.is_null() {
        return LanaError::InvalidState as i32;
    }
    let mut s = Box::from_raw(store);
    match store_close(&mut s) {
        Ok(()) => LanaError::Ok as i32,
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_store_get_path` (free the result with `lana_string_free`).
#[no_mangle]
pub unsafe extern "C" fn lana_store_get_path(store: *const Store, out_path: *mut *mut c_char) -> i32 {
    if store.is_null() || out_path.is_null() {
        return LanaError::InvalidState as i32;
    }
    match store_get_path(&*store) {
        Ok(path) => {
            *out_path = CString::new(path).map(|c| c.into_raw()).unwrap_or(ptr::null_mut());
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_store_get` (the VM argument is unused; writes an opaque value).
#[no_mangle]
pub unsafe extern "C" fn lana_store_get(
    store: *const Store,
    _vm: *mut VmHandle,
    key: *const c_char,
    out_value: *mut *mut Value,
) -> i32 {
    if store.is_null() || out_value.is_null() {
        return LanaError::InvalidState as i32;
    }
    match store_get(&*store, cstr(key)) {
        Ok(v) => {
            *out_value = Box::into_raw(Box::new(v));
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_store_put` (takes an opaque value).
#[no_mangle]
pub unsafe extern "C" fn lana_store_put(
    store: *mut Store,
    key: *const c_char,
    value: *const Value,
) -> i32 {
    if store.is_null() || value.is_null() {
        return LanaError::InvalidState as i32;
    }
    match store_put(&mut *store, cstr(key), &*value) {
        Ok(()) => LanaError::Ok as i32,
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_store_delete`.
#[no_mangle]
pub unsafe extern "C" fn lana_store_delete(store: *mut Store, key: *const c_char) -> i32 {
    if store.is_null() {
        return LanaError::InvalidState as i32;
    }
    match store_delete(&mut *store, cstr(key)) {
        Ok(()) => LanaError::Ok as i32,
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_store_commit`.
#[no_mangle]
pub unsafe extern "C" fn lana_store_commit(
    store: *mut Store,
    out_revision: *mut LanaStoreRevisionInfo,
) -> i32 {
    if store.is_null() || out_revision.is_null() {
        return LanaError::InvalidState as i32;
    }
    match store_commit(&mut *store) {
        Ok(info) => {
            (*out_revision).revision_id = info.revision_id;
            (*out_revision).schema_version = info.schema_version;
            (*out_revision).timestamp = info.timestamp;
            (*out_revision).digest = info.digest;
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_store_current_revision`.
#[no_mangle]
pub unsafe extern "C" fn lana_store_current_revision(
    store: *const Store,
    out_revision: *mut LanaStoreRevisionInfo,
) -> i32 {
    if store.is_null() || out_revision.is_null() {
        return LanaError::InvalidState as i32;
    }
    match store_current_revision(&*store) {
        Ok(info) => {
            (*out_revision).revision_id = info.revision_id;
            (*out_revision).schema_version = info.schema_version;
            (*out_revision).timestamp = info.timestamp;
            (*out_revision).digest = info.digest;
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_store_get_at` (the VM argument is unused; writes an opaque value).
#[no_mangle]
pub unsafe extern "C" fn lana_store_get_at(
    store: *const Store,
    _vm: *mut VmHandle,
    revision: u64,
    key: *const c_char,
    out_value: *mut *mut Value,
) -> i32 {
    if store.is_null() || out_value.is_null() {
        return LanaError::InvalidState as i32;
    }
    match store_get_at(&*store, revision, cstr(key)) {
        Ok(v) => {
            *out_value = Box::into_raw(Box::new(v));
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

/// Mirrors `lana_store_compact`.
#[no_mangle]
pub unsafe extern "C" fn lana_store_compact(
    store: *mut Store,
    retention: u64,
    out_revision: *mut LanaStoreRevisionInfo,
) -> i32 {
    if store.is_null() || out_revision.is_null() {
        return LanaError::InvalidState as i32;
    }
    match store_compact(&mut *store, retention) {
        Ok(info) => {
            (*out_revision).revision_id = info.revision_id;
            (*out_revision).schema_version = info.schema_version;
            (*out_revision).timestamp = info.timestamp;
            (*out_revision).digest = info.digest;
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

// ---------------------------------------------------------------------------
// policy.h
// ---------------------------------------------------------------------------

unsafe fn policy_from_c(p: *const LanaPolicy) -> Option<Policy> {
    if p.is_null() {
        return None;
    }
    let p = &*p;
    let rule = &p.rule;
    let kind = match rule.kind {
        0 => PolicyRuleKind::ProbabilityAtLeast,
        1 => PolicyRuleKind::Equals,
        2 => PolicyRuleKind::OrderLessThan,
        3 => PolicyRuleKind::Present,
        _ => return None,
    };
    Some(Policy {
        schema_version: p.schema_version,
        policy_id: Arc::from(cstr(p.policy_id)),
        rule: PolicyRule {
            schema_version: rule.schema_version,
            kind,
            field: Arc::from(cstr(rule.field)),
            threshold: rule.threshold,
            expected: if rule.expected.is_null() {
                None
            } else {
                Some(Arc::from(cstr(rule.expected)))
            },
            effect: Arc::from(cstr(rule.effect)),
        },
    })
}

/// Mirrors `lana_policy_version`.
#[no_mangle]
pub unsafe extern "C" fn lana_policy_version(
    policy: *const LanaPolicy,
    out_version: *mut u8,
) -> i32 {
    if out_version.is_null() {
        return LanaError::InvalidState as i32;
    }
    let Some(policy) = policy_from_c(policy) else {
        return LanaError::InvalidState as i32;
    };
    match policy_version(&policy) {
        Ok(digest) => {
            ptr::copy_nonoverlapping(digest.as_ptr(), out_version, 32);
            LanaError::Ok as i32
        }
        Err(e) => e as i32,
    }
}

// ---------------------------------------------------------------------------
// Memory helpers (extensions)
// ---------------------------------------------------------------------------

/// Releases a string returned by `lana_store_get_path` (extension).
#[no_mangle]
pub unsafe extern "C" fn lana_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Releases a byte buffer returned by `lana_state_encode` (extension).
#[no_mangle]
pub unsafe extern "C" fn lana_bytes_free(ptr: *mut c_void, len: usize) {
    if !ptr.is_null() {
        let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, len);
        drop(Box::from_raw(slice as *mut [u8]));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn cstr_to_string(ptr: *const c_char) -> String {
        if ptr.is_null() {
            return String::new();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }

    #[test]
    fn value_encode_round_trip() {
        unsafe {
            let value = lana_value_number(0.5);
            assert!(!value.is_null());

            let buffer = lana_buffer_new();
            let code = lana_codec_encode_value(buffer, value);
            assert_eq!(code, LanaError::Ok as i32);

            let len = lana_buffer_length(buffer);
            let data = lana_buffer_data(buffer);
            let slice = std::slice::from_raw_parts(data, len);
            assert_eq!(slice, b"0.5");

            lana_value_free(value);
            lana_buffer_free(buffer);
        }
    }

    #[test]
    fn error_names_match_c() {
        unsafe {
            assert_eq!(cstr_to_string(lana_error_name(LanaError::Ok as i32)), "LANA_OK");
            assert_eq!(
                cstr_to_string(lana_error_name(LanaError::Type as i32)),
                "LANA_ERR_TYPE"
            );
            assert_eq!(
                cstr_to_string(lana_error_name(9999)),
                "LANA_ERR_UNKNOWN"
            );
            assert_eq!(lana_error_kind_from_code(LanaError::Type as i32), 2);
            assert_eq!(cstr_to_string(lana_error_kind_name(2)), "type");
            assert_eq!(cstr_to_string(lana_error_kind_name(6)), "resource-limit");
            assert_eq!(cstr_to_string(lana_value_type_name(1)), "number");
            assert_eq!(cstr_to_string(lana_opcode_name(0)), "NOP");
        }
    }

    #[test]
    fn state_round_trip() {
        unsafe {
            let mut encoded: *mut c_void = ptr::null_mut();
            let mut len = 0usize;
            let state = LanaState {
                p: 0.5,
                d_re: 0.25,
                d_im: 0.0,
            };
            let code = lana_state_encode(state, &mut encoded, &mut len);
            assert_eq!(code, LanaError::Ok as i32);
            assert!(len > 0);

            let mut decoded = LanaState {
                p: 0.0,
                d_re: 0.0,
                d_im: 0.0,
            };
            let code = lana_state_decode(encoded, len, &mut decoded);
            assert_eq!(code, LanaError::Ok as i32);
            assert!((decoded.p - 0.5).abs() < 1e-9);
            assert!((decoded.d_re - 0.25).abs() < 1e-9);

            lana_bytes_free(encoded, len);
        }
    }
}
