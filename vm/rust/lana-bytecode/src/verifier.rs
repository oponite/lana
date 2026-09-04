//! LABC verifier, mirroring `lana_chunk_verify` in `vm/c/bytecode.c`.
//!
//! Every rule — register bounds, constant indices, per-opcode operand shapes,
//! jump targets, function metadata, and the OP_STATE_NEW state validation —
//! matches the C11 verifier so that a chunk verifies identically under both
//! implementations.

use crate::chunk::{Chunk, Instruction};
use crate::error::{LanaError, LanaErrorInfo};
use crate::opcode::{OpCode, LANA_MAX_REGISTERS};

/// Maximum host-call id. The C11 reference caps this at 55 (54 built-in host
/// calls, ids 0-54); the Rust VM adds 11 durable-pipeline host calls
/// (store/policy/ledger, ids 55-65) behind the host-call extension, so the
/// Rust verifier accepts the wider range. This is a deliberate, documented
/// divergence from the frozen C11 verifier.
pub const LANA_HOST_COUNT: u32 = 66;

const LANA_TRANSFORM_NEUTRALIZE: u32 = 1;
const LANA_MEASURE_SAMPLE: u32 = 2;
const LANA_MEASURE_BASIS_Y: u32 = 2;
const LANA_STATE_EPSILON: f64 = 1e-12;

fn register_valid(value: u32) -> bool {
    value < LANA_MAX_REGISTERS
}

fn constant_valid(chunk: &Chunk, value: u32) -> bool {
    value < chunk.constants.len() as u32
}

fn verify_register(error: &mut LanaErrorInfo, ip: usize, instruction: &Instruction, reg: u32) -> Result<(), LanaError> {
    if !register_valid(reg) {
        *error = LanaErrorInfo::new(
            LanaError::Register, ip, instruction.opcode as u8, instruction.line,
            format!("register R{reg} is out of range"));
        return Err(LanaError::Register);
    }
    Ok(())
}

/// Mirror of `lana_state_make_complex` in `vm/c/state.c`, used to validate the
/// three constant operands of OP_STATE_NEW.
fn state_make_complex(p: f64, d_re: f64, d_im: f64) -> Result<(), LanaError> {
    if !p.is_finite() || !d_re.is_finite() || !d_im.is_finite() {
        return Err(LanaError::InvalidState);
    }
    if p < -LANA_STATE_EPSILON || p > 1.0 + LANA_STATE_EPSILON {
        return Err(LanaError::InvalidState);
    }
    let p = p.clamp(0.0, 1.0);
    let radius_squared = d_re * d_re + d_im * d_im;
    let radius = radius_squared.sqrt();
    if !radius.is_finite() || radius > 1.0 + LANA_STATE_EPSILON {
        return Err(LanaError::InvalidState);
    }
    let (d_re, d_im) = if radius > 1.0 {
        (d_re / radius, d_im / radius)
    } else {
        (d_re, d_im)
    };
    let (d_re, d_im) = if p == 0.0 || p == 1.0 { (0.0, 0.0) } else { (d_re, d_im) };
    let p = if p == 0.0 { 0.0 } else { p };
    let d_re = if d_re == 0.0 { 0.0 } else { d_re };
    let d_im = if d_im == 0.0 { 0.0 } else { d_im };
    let radius_squared = d_re * d_re + d_im * d_im;
    if !radius_squared.is_finite() || radius_squared > 1.0 {
        return Err(LanaError::InvalidState);
    }
    if (p != 0.0 && p != 1.0) || (d_re == 0.0 && d_im == 0.0) {
        Ok(())
    } else {
        Err(LanaError::InvalidState)
    }
}

/// Verify a chunk, mirroring `lana_chunk_verify`.
///
/// Returns the first `LanaErrorInfo` the C11 verifier would produce for the
/// same chunk, or `Ok(())` when the chunk is well-formed.
pub fn verify(chunk: &Chunk) -> Result<(), LanaErrorInfo> {
    if chunk.code.is_empty() || chunk.entry >= chunk.code.len() as u32 {
        return Err(LanaErrorInfo::new(
            LanaError::Format, 0, OpCode::Nop as u8, 0, "chunk has no valid entry point"));
    }
    if chunk.version != crate::opcode::LABC_VERSION && chunk.version != crate::opcode::LABC_VERSION_1 {
        return Err(LanaErrorInfo::new(
            LanaError::Format, 0, OpCode::Nop as u8, 0,
            format!("unsupported LABC version {}", chunk.version)));
    }
    for (function_index, function) in chunk.functions.iter().enumerate() {
        if function.entry >= chunk.code.len() as u32
            || function.register_count > LANA_MAX_REGISTERS
            || function.arity > function.register_count
        {
            return Err(LanaErrorInfo::new(
                LanaError::Format, function.entry as usize, OpCode::Call as u8, 0,
                format!("function {function_index} has invalid metadata")));
        }
    }
    for (ip, ins) in chunk.code.iter().enumerate() {
        let result = verify_instruction(chunk, ip, ins);
        if let Err(code) = result {
            let message = format!("invalid operands for {}", ins.opcode.name());
            return Err(LanaErrorInfo::new(code, ip, ins.opcode as u8, ins.line, message));
        }
    }
    Ok(())
}

fn verify_instruction(chunk: &Chunk, ip: usize, ins: &Instruction) -> Result<(), LanaError> {
    use OpCode::*;
    let mut error = LanaErrorInfo::new(LanaError::Ok, 0, 0, 0, "");
    let mut result = Ok(());
    let check = |result: &mut Result<(), LanaError>, error: &mut LanaErrorInfo, ip: usize, ins: &Instruction, reg: u32| {
        if result.is_ok() {
            if let Err(code) = verify_register(error, ip, ins, reg) {
                *result = Err(code);
            }
        }
    };
    match ins.opcode {
        Nop | Halt | TaskgroupEnter | TaskgroupExit => {}
        LoadConst => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && !constant_valid(chunk, ins.imm) {
                result = Err(LanaError::Constant);
            }
        }
        StateNew => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok()
                && (!constant_valid(chunk, ins.b)
                    || !constant_valid(chunk, ins.c)
                    || !constant_valid(chunk, ins.imm))
            {
                result = Err(LanaError::Constant);
            }
            if result.is_ok() {
                let b = &chunk.constants[ins.b as usize];
                let c = &chunk.constants[ins.c as usize];
                let imm = &chunk.constants[ins.imm as usize];
                if b.value_type() != crate::value::ValueType::Number
                    || c.value_type() != crate::value::ValueType::Number
                    || imm.value_type() != crate::value::ValueType::Number
                {
                    result = Err(LanaError::Type);
                } else if let (crate::value::Value::Number(p), crate::value::Value::Number(d_re), crate::value::Value::Number(d_im)) =
                    (b, c, imm)
                {
                    result = state_make_complex(*p, *d_re, *d_im);
                }
            }
        }
        StateBuild | Mix => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.c);
            check(&mut result, &mut error, ip, ins, ins.imm);
        }
        Map => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && ins.c > LANA_TRANSFORM_NEUTRALIZE {
                result = Err(LanaError::Transform);
            }
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        Support => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && ins.imm == 0 {
                result = Err(LanaError::Format);
            }
            if result.is_ok() && ins.c != 0 {
                result = Err(LanaError::Format);
            }
        }
        Expect => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && ins.c != 0 {
                result = Err(LanaError::Format);
            }
        }
        Validate => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.c);
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        Revision => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && (ins.c != 0 || ins.imm != 0) {
                result = Err(LanaError::Format);
            }
        }
        Transform => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && ins.c > LANA_TRANSFORM_NEUTRALIZE {
                result = Err(LanaError::Transform);
            }
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        Measure => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && ins.c > LANA_MEASURE_SAMPLE {
                result = Err(LanaError::Measure);
            }
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        Append => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.c);
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        Attenuate => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.c);
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        TraceDistance => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.c);
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        AppendRedundant | AppendComplementary => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.c);
            check(&mut result, &mut error, ip, ins, ins.imm);
        }
        AppendFullRedundancy => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.c);
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        AdtBuild => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && (ins.b >= LANA_MAX_REGISTERS || ins.b + ins.c > LANA_MAX_REGISTERS) {
                result = Err(LanaError::Register);
            }
            if result.is_ok() && !constant_valid(chunk, ins.imm) {
                result = Err(LanaError::Constant);
            }
            if result.is_ok()
                && chunk.constants[ins.imm as usize].value_type() != crate::value::ValueType::Number
            {
                result = Err(LanaError::Type);
            }
        }
        AdtCase => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && !constant_valid(chunk, ins.b) {
                result = Err(LanaError::Constant);
            }
            if result.is_ok()
                && chunk.constants[ins.b as usize].value_type() != crate::value::ValueType::Number
            {
                result = Err(LanaError::Type);
            }
            if result.is_ok() && ins.imm >= chunk.code.len() as u32 {
                result = Err(LanaError::Jump);
            }
        }
        AdtGet => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
        }
        Lazy => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && ins.b >= chunk.functions.len() as u32 {
                result = Err(LanaError::Format);
            }
            check(&mut result, &mut error, ip, ins, ins.c);
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        Force => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.c);
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        SampleStateDist => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && (ins.c != 0 || ins.imm != 0) {
                result = Err(LanaError::Format);
            }
        }
        MeasureBasis => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && ins.c > LANA_MEASURE_BASIS_Y {
                result = Err(LanaError::Measure);
            }
            if result.is_ok() && ins.imm > LANA_MEASURE_SAMPLE {
                result = Err(LanaError::Measure);
            }
        }
        EstimateMeasureProbability | EstimateMeasureDistribution => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && ins.c > LANA_MEASURE_BASIS_Y {
                result = Err(LanaError::Measure);
            }
            if result.is_ok() && ins.imm == 0 {
                result = Err(LanaError::Format);
            }
        }
        JointBuild => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && (ins.c == 0 || ins.b + ins.c > LANA_MAX_REGISTERS) {
                result = Err(LanaError::Register);
            }
            if result.is_ok() && !constant_valid(chunk, ins.imm) {
                result = Err(LanaError::Constant);
            }
            if result.is_ok() && chunk.constants[ins.imm as usize].value_type() != crate::value::ValueType::String {
                result = Err(LanaError::Type);
            }
        }
        JointProject => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && !constant_valid(chunk, ins.c) {
                result = Err(LanaError::Constant);
            }
            if result.is_ok() && chunk.constants[ins.c as usize].value_type() != crate::value::ValueType::String {
                result = Err(LanaError::Type);
            }
        }
        JointCondition => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.imm);
            if result.is_ok() && !constant_valid(chunk, ins.c) {
                result = Err(LanaError::Constant);
            }
            if result.is_ok() && chunk.constants[ins.c as usize].value_type() != crate::value::ValueType::String {
                result = Err(LanaError::Type);
            }
        }
        JointSample | Resolve => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && (ins.c != 0 || ins.imm != 0) {
                result = Err(LanaError::Format);
            }
        }
        JointBuildFinite => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && !constant_valid(chunk, ins.c) {
                result = Err(LanaError::Constant);
            }
            if result.is_ok() && chunk.constants[ins.c as usize].value_type() != crate::value::ValueType::String {
                result = Err(LanaError::Type);
            }
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        JointRename => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && (!constant_valid(chunk, ins.c) || !constant_valid(chunk, ins.imm)) {
                result = Err(LanaError::Constant);
            }
            if result.is_ok()
                && (chunk.constants[ins.c as usize].value_type() != crate::value::ValueType::String
                    || chunk.constants[ins.imm as usize].value_type() != crate::value::ValueType::String)
            {
                result = Err(LanaError::Type);
            }
        }
        PossibilityBuild | InfoSample => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && (ins.c != 0 || ins.imm != 0) {
                result = Err(LanaError::Format);
            }
        }
        PathSplit => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && ins.imm >= chunk.code.len() as u32 {
                result = Err(LanaError::Jump);
            }
            if result.is_ok() && (ins.b != 0 || ins.c != 0) {
                result = Err(LanaError::Format);
            }
        }
        PathJoin => {
            if ins.a != 0 || ins.b != 0 || ins.c != 0 || ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        Observe => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.imm);
            if result.is_ok() && !constant_valid(chunk, ins.c) {
                result = Err(LanaError::Constant);
            }
            if result.is_ok() && chunk.constants[ins.c as usize].value_type() != crate::value::ValueType::String {
                result = Err(LanaError::Type);
            }
        }
        Evidence | Assume => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && !constant_valid(chunk, ins.c) {
                result = Err(LanaError::Constant);
            }
            if result.is_ok() && chunk.constants[ins.c as usize].value_type() != crate::value::ValueType::String {
                result = Err(LanaError::Type);
            }
            if result.is_ok() && ins.imm != 0 {
                result = Err(LanaError::Format);
            }
        }
        Derivation | Explain => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok() && (ins.c != 0 || ins.imm != 0) {
                result = Err(LanaError::Format);
            }
        }
        Jump => {
            if ins.imm >= chunk.code.len() as u32 {
                result = Err(LanaError::Jump);
            }
        }
        JumpIfTrue | JumpIfFalse => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && ins.imm >= chunk.code.len() as u32 {
                result = Err(LanaError::Jump);
            }
        }
        Move | Previous | Change | Velocity | Return => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && ins.opcode != Return {
                check(&mut result, &mut error, ip, ins, ins.b);
            }
        }
        ArrayGet | ArraySet => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.c);
        }
        GetField | GetIndex | SetIndex | HistoryConfig | Binary | Compare => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            if result.is_ok()
                && (ins.opcode == Binary || ins.opcode == Compare || ins.opcode == SetIndex)
            {
                check(&mut result, &mut error, ip, ins, ins.c);
            }
        }
        Unary | Print => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && ins.opcode == Unary {
                check(&mut result, &mut error, ip, ins, ins.b);
            }
        }
        ArrayNew => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && (ins.b >= LANA_MAX_REGISTERS || ins.b + ins.c > LANA_MAX_REGISTERS) {
                result = Err(LanaError::Register);
            }
        }
        Call => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && ins.b >= chunk.functions.len() as u32 {
                result = Err(LanaError::Format);
            }
            if result.is_ok() && (ins.c >= LANA_MAX_REGISTERS || ins.c + ins.imm > LANA_MAX_REGISTERS) {
                result = Err(LanaError::Register);
            }
        }
        Fork => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && ins.b >= chunk.functions.len() as u32 {
                result = Err(LanaError::Format);
            }
            if result.is_ok() && (ins.c >= LANA_MAX_REGISTERS || ins.c + ins.imm > LANA_MAX_REGISTERS) {
                result = Err(LanaError::Register);
            }
            if result.is_ok() && ins.imm != chunk.functions[ins.b as usize].arity {
                result = Err(LanaError::Type);
            }
        }
        Join | JoinAll => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
        }
        JoinTimeout => {
            check(&mut result, &mut error, ip, ins, ins.a);
            check(&mut result, &mut error, ip, ins, ins.b);
            check(&mut result, &mut error, ip, ins, ins.c);
        }
        Cancel => {
            check(&mut result, &mut error, ip, ins, ins.a);
        }
        HostCall => {
            check(&mut result, &mut error, ip, ins, ins.a);
            if result.is_ok() && ins.b >= LANA_HOST_COUNT {
                result = Err(LanaError::Format);
            }
            if result.is_ok() && (ins.c >= LANA_MAX_REGISTERS || ins.c + ins.imm > LANA_MAX_REGISTERS) {
                result = Err(LanaError::Register);
            }
        }
        Count => {
            result = Err(LanaError::Opcode);
        }
    }
    result
}
