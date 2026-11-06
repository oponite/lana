#![no_main]

use lana_bytecode::{loader, verifier, OpCode};
use lana_vm::vm::*;
use libfuzzer_sys::fuzz_target;

/// Host calls that only compute on VM values: no files, store, network, FFI,
/// environment, clock, sleep, threads, tasks, datasets, or GPU.
const PURE_HOST_CALLS: &[u32] = &[
    LANA_HOST_ASSERT,
    LANA_HOST_RANDOM,
    LANA_HOST_RANDOM_SEED,
    LANA_HOST_MAP_NEW,
    LANA_HOST_MAP_HAS,
    LANA_HOST_MAP_GET,
    LANA_HOST_MAP_SET,
    LANA_HOST_MAP_KEYS,
    LANA_HOST_INDEX_GET,
    LANA_HOST_INDEX_SET,
    LANA_HOST_JSON_PARSE,
    LANA_HOST_JSON_STRINGIFY,
    LANA_HOST_ARRAY_NEW,
    LANA_HOST_ARRAY_PUSH,
    LANA_HOST_ARRAY_LENGTH,
    LANA_HOST_STRING_LENGTH,
    LANA_HOST_STRING_BYTE_AT,
    LANA_HOST_STRING_SLICE,
    LANA_HOST_STRING_CONCAT,
    LANA_HOST_STRING_HEX,
    LANA_HOST_STRING_JOIN,
    LANA_HOST_STRING_UNESCAPE,
    LANA_HOST_NUMBER_TO_STRING,
    LANA_HOST_STRING_TO_NUMBER,
    LANA_HOST_FLOOR,
    LANA_HOST_TYPE_OF,
    LANA_HOST_FORMAT,
    LANA_HOST_FORMAT_NUMBER,
    LANA_HOST_CHAR_LENGTH,
    LANA_HOST_STRING_CODEPOINT_SLICE,
    LANA_HOST_TO_UPPER,
    LANA_HOST_TO_LOWER,
    LANA_HOST_REGEX_COMPILE,
    LANA_HOST_REGEX_MATCH,
    LANA_HOST_REGEX_SEARCH,
    LANA_HOST_REGEX_REPLACE,
    LANA_HOST_SET_NEW,
    LANA_HOST_SET_ADD,
    LANA_HOST_SET_CONTAINS,
    LANA_HOST_SET_UNION,
    LANA_HOST_SET_INTERSECT,
    LANA_HOST_SET_DIFFERENCE,
    LANA_HOST_HASH_UPDATE,
    LANA_HOST_SAMPLE_RECORD,
    LANA_HOST_INFORMATION_NEW,
    LANA_HOST_INFORMATION_INSPECT,
    LANA_HOST_CLAIM_NEW,
    LANA_HOST_CLAIM_VALUE,
    LANA_HOST_CLAIM_PROPOSITION,
    LANA_HOST_CLAIM_STATUS,
    LANA_HOST_LAZY_BOUND,
    LANA_HOST_CORRELATED,
    LANA_HOST_SURPRISAL,
    LANA_HOST_TENSOR,
    LANA_HOST_TENSOR_COMPLEX,
    LANA_HOST_TENSOR_ALLOC,
    LANA_HOST_TENSOR_ZEROS,
    LANA_HOST_TENSOR_ONES,
    LANA_HOST_TENSOR_EYE,
    LANA_HOST_TENSOR_DTYPE,
    LANA_HOST_TENSOR_CAST,
    LANA_HOST_TENSOR_SHAPE,
    LANA_HOST_TENSOR_NDIM,
    LANA_HOST_TENSOR_RESHAPE,
    LANA_HOST_TENSOR_TRANSPOSE,
    LANA_HOST_TENSOR_ADD,
    LANA_HOST_TENSOR_SUB,
    LANA_HOST_TENSOR_MUL,
    LANA_HOST_TENSOR_DIV,
    LANA_HOST_TENSOR_MATMUL,
    LANA_HOST_TENSOR_SUM,
    LANA_HOST_TENSOR_MEAN,
    LANA_HOST_TENSOR_MAX,
    LANA_HOST_TENSOR_MIN,
    LANA_HOST_TENSOR_EXP,
    LANA_HOST_TENSOR_LOG,
    LANA_HOST_TENSOR_SQRT,
    LANA_HOST_TENSOR_RELU,
    LANA_HOST_TENSOR_SOFTMAX,
    LANA_HOST_TENSOR_LOGSUMEXP,
    LANA_HOST_TENSOR_ARGMAX,
    LANA_HOST_TENSOR_COMPARE,
    LANA_HOST_TENSOR_SELECT,
    LANA_HOST_TENSOR_GATHER,
    LANA_HOST_CHOLESKY_SOLVE,
    LANA_HOST_RANDOM_UNIFORM,
    LANA_HOST_RANDOM_NORMAL,
    LANA_HOST_DENSITY_OPERATOR,
    LANA_HOST_POVM,
    LANA_HOST_CHANNEL,
    LANA_HOST_OBSERVABLE,
    LANA_HOST_TENSOR_PRODUCT,
    LANA_HOST_PARTIAL_TRACE,
    LANA_HOST_MEASURE_WITH,
    LANA_HOST_APPLY_TO,
    LANA_HOST_EXPECT,
    LANA_HOST_MIX,
    LANA_HOST_TRACE_DISTANCE,
    LANA_HOST_IS_SEPARABLE,
    LANA_HOST_TO_STATE,
    LANA_HOST_STATE_TENSOR,
    LANA_HOST_APPEND,
    LANA_HOST_MEASURE,
    LANA_HOST_TRANSFORM,
    LANA_HOST_GRAD,
    LANA_HOST_VJP,
    LANA_HOST_SGD,
    LANA_HOST_ADAM,
    LANA_HOST_TRAIN,
    LANA_HOST_MCMC,
    LANA_HOST_VI,
    LANA_HOST_SMC,
    LANA_HOST_INFER,
    LANA_HOST_UPDATE,
    LANA_HOST_RESUME,
    LANA_HOST_CORE_ENTROPY,
    LANA_HOST_CORE_CONDITIONAL_ENTROPY,
    LANA_HOST_CORE_MUTUAL_INFORMATION,
    LANA_HOST_CORE_BROJA,
    LANA_HOST_CORE_KERNEL,
    LANA_HOST_CORE_IDENTITY_KERNEL,
    LANA_HOST_CORE_COMPOSE_KERNELS,
    LANA_HOST_CORE_NETWORK,
    LANA_HOST_CORE_INFER,
    LANA_HOST_CORE_FORGET_WEIGHTS,
    LANA_HOST_CORE_ASSIGN_WEIGHTS,
];

fuzz_target!(|data: &[u8]| {
    let Ok(chunk) = loader::load(data) else {
        return;
    };
    if verifier::verify(&chunk).is_err() {
        return;
    }
    // Skip output, task and async opcodes, and host calls with effects.
    if chunk
        .code
        .iter()
        .any(|instruction| match instruction.opcode {
            OpCode::Print
            | OpCode::Fork
            | OpCode::Join
            | OpCode::JoinTimeout
            | OpCode::JoinAll
            | OpCode::Cancel
            | OpCode::TaskgroupEnter
            | OpCode::TaskgroupExit
            | OpCode::Async
            | OpCode::Await
            | OpCode::RunAsync => true,
            OpCode::HostCall => !PURE_HOST_CALLS.contains(&instruction.b),
            _ => false,
        })
    {
        return;
    }
    let mut vm = Vm::new(&chunk);
    vm.set_instruction_limit(10_000);
    // shortcut: 64 KiB of headroom, because Buffer::reserve grows one element per push
    // once doubling would cross the limit, which is quadratic under ASan's copying
    // realloc and times out string-growth loops; restore 64 MiB once that growth is amortized.
    let limit = vm.allocated_bytes() + 64 * 1024;
    if vm.set_memory_limit(limit).is_ok() {
        let _ = vm.run();
    }
});
