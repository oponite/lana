//! LABC assembler, mirroring `lana_assemble_file` in `vm/c/assembler.c`.
//!
//! Accepts the same textual assembly format as the C11 assembler and produces
//! the same chunk, so that `.lasm` sources assemble identically under both
//! implementations.

use crate::chunk::{Chunk, Function, Instruction};
use crate::error::{LanaError, LanaErrorInfo};
use crate::opcode::{OpCode, LABC_VERSION, LABC_VERSION_1, LANA_MAX_REGISTERS};
use crate::value::{Value, ValueType};

const LANA_NO_OPERAND: u32 = u32::MAX;
const LANA_ASSEMBLER_MAX_FIXUPS: usize = 4096;

const LANA_TRANSFORM_INVERT: u32 = 0;
const LANA_TRANSFORM_NEUTRALIZE: u32 = 1;
const LANA_OBSERVABLE_PROBABILITY: u32 = 0;
const LANA_MEASURE_PROBABILITY: u32 = 0;
const LANA_MEASURE_DISTRIBUTION: u32 = 1;
const LANA_MEASURE_SAMPLE: u32 = 2;
const LANA_MEASURE_BASIS_COMPUTATIONAL: u32 = 0;
const LANA_MEASURE_BASIS_X: u32 = 1;
const LANA_MEASURE_BASIS_Y: u32 = 2;
const LANA_BINARY_ADD: u32 = 0;
const LANA_BINARY_SUBTRACT: u32 = 1;
const LANA_BINARY_MULTIPLY: u32 = 2;
const LANA_BINARY_DIVIDE: u32 = 3;
const LANA_COMPARE_EQUAL: u32 = 0;
const LANA_COMPARE_NOT_EQUAL: u32 = 1;
const LANA_COMPARE_LESS: u32 = 2;
const LANA_COMPARE_LESS_EQUAL: u32 = 3;
const LANA_COMPARE_GREATER: u32 = 4;
const LANA_COMPARE_GREATER_EQUAL: u32 = 5;
const LANA_HISTORY_LATEST: u32 = 0;
const LANA_HISTORY_DURATION: u32 = 1;

/// Host-call names in id order, mirroring the table in `vm/c/assembler.c`.
const HOST_CALL_NAMES: &[&str] = &[
    "args", "read_text", "write_text", "now", "random", "assert",
    "map_new", "map_has", "map_get", "map_set", "map_keys", "index_get", "index_set", "json_parse", "json_stringify",
    "csv_read", "csv_write", "string_length", "string_byte_at",
    "string_slice", "string_concat", "number_to_string", "array_new",
    "array_push", "string_hex", "string_join", "array_length",
    "string_unescape", "path_resolve", "sample_record",
    "information_new", "claim_new", "claim_value", "claim_proposition",
    "claim_status", "planned_effect_new", "planned_effect_execute",
    "planned_effect_status", "shared_information", "shared_grant",
    "shared_revoke", "shared_snapshot", "shared_at", "shared_observe",
    "shared_revision", "shared_identity", "shared_wait",
    "information_inspect", "directory_list", "directory_create",
    "path_exists", "write_text_atomic", "hash_update", "lazy_bound",
    "correlated", "surprisal",
    "store_open", "store_put", "store_get", "store_delete", "store_commit",
    "store_scan", "store_current_revision", "policy_evaluate",
    "policy_store_decision", "ledger_append", "ledger_query",
];

struct Label {
    name: String,
    offset: u32,
}

struct Fixup {
    name: String,
    instruction: u32,
}

struct FunctionFixup {
    name: String,
    instruction: u32,
}

fn parse_register(text: &str) -> Option<u32> {
    let rest = text.strip_prefix('R').or_else(|| text.strip_prefix('r'))?;
    if rest.is_empty() {
        return None;
    }
    let value: u32 = rest.parse().ok()?;
    if value >= LANA_MAX_REGISTERS {
        return None;
    }
    Some(value)
}

fn parse_number(text: &str) -> Option<f64> {
    text.parse::<f64>().ok()
}

fn sample_count(text: &str) -> Option<u32> {
    if text.is_empty() || text.starts_with('-') {
        return None;
    }
    let value: u64 = text.parse().ok()?;
    if value == 0 || value > u32::MAX as u64 {
        return None;
    }
    Some(value as u32)
}

fn transform_id(name: &str) -> Option<u32> {
    match name {
        "invert" => Some(LANA_TRANSFORM_INVERT),
        "neutralize" => Some(LANA_TRANSFORM_NEUTRALIZE),
        _ => None,
    }
}

fn observable_id(name: &str) -> Option<u32> {
    match name {
        "probability" => Some(LANA_OBSERVABLE_PROBABILITY),
        _ => None,
    }
}

fn measure_id_current(name: &str) -> Option<u32> {
    match name {
        "probability" => Some(LANA_MEASURE_PROBABILITY),
        "distribution" => Some(LANA_MEASURE_DISTRIBUTION),
        "sample" => Some(LANA_MEASURE_SAMPLE),
        _ => None,
    }
}

fn basis_id(name: &str) -> Option<u32> {
    match name {
        "computational" => Some(LANA_MEASURE_BASIS_COMPUTATIONAL),
        "x" => Some(LANA_MEASURE_BASIS_X),
        "y" => Some(LANA_MEASURE_BASIS_Y),
        _ => None,
    }
}

fn field_id(name: &str) -> Option<u32> {
    match name {
        "p" => Some(0),
        "d" | "d_re" => Some(1),
        "d_im" => Some(2),
        _ => None,
    }
}

fn index_id(name: &str) -> Option<u32> {
    match name {
        "timestamp" => Some(0),
        "source" => Some(1),
        "weight" => Some(2),
        "confidence" => Some(3),
        _ => None,
    }
}

fn binary_id(name: &str) -> Option<u32> {
    match name {
        "+" | "add" => Some(LANA_BINARY_ADD),
        "-" | "sub" => Some(LANA_BINARY_SUBTRACT),
        "*" | "mul" => Some(LANA_BINARY_MULTIPLY),
        "/" | "div" => Some(LANA_BINARY_DIVIDE),
        _ => None,
    }
}

fn compare_id(name: &str) -> Option<u32> {
    match name {
        "==" => Some(LANA_COMPARE_EQUAL),
        "!=" => Some(LANA_COMPARE_NOT_EQUAL),
        "<" => Some(LANA_COMPARE_LESS),
        "<=" => Some(LANA_COMPARE_LESS_EQUAL),
        ">" => Some(LANA_COMPARE_GREATER),
        ">=" => Some(LANA_COMPARE_GREATER_EQUAL),
        _ => None,
    }
}

fn host_call_id(name: &str) -> Option<u32> {
    HOST_CALL_NAMES
        .iter()
        .position(|candidate| *candidate == name)
        .map(|index| index as u32)
}

fn hex_digit(character: u8) -> Option<u32> {
    match character {
        b'0'..=b'9' => Some((character - b'0') as u32),
        b'a'..=b'f' => Some((character - b'a' + 10) as u32),
        b'A'..=b'F' => Some((character - b'A' + 10) as u32),
        _ => None,
    }
}

fn add_number(chunk: &mut Chunk, number: f64) -> u32 {
    let index = chunk.constants.len() as u32;
    chunk.constants.push(Value::Number(number));
    index
}

fn number_constant(chunk: &mut Chunk, text: &str) -> Result<u32, LanaError> {
    if let Some(rest) = text.strip_prefix('K').or_else(|| text.strip_prefix('k')) {
        if !rest.is_empty() {
            let index: u32 = rest.parse().map_err(|_| LanaError::Constant)?;
            if (index as usize) < chunk.constants.len()
                && chunk.constants[index as usize].value_type() == ValueType::Number
            {
                return Ok(index);
            }
            return Err(LanaError::Constant);
        }
    }
    let number = parse_number(text).ok_or(LanaError::Format)?;
    Ok(add_number(chunk, number))
}

fn string_constant(chunk: &mut Chunk, text: &str) -> Result<u32, LanaError> {
    if text.is_empty() {
        return Err(LanaError::Format);
    }
    if let Some(rest) = text.strip_prefix('S').or_else(|| text.strip_prefix('s')) {
        if let Some(first) = rest.chars().next() {
            if first.is_ascii_digit() {
                let index: u32 = rest.parse().map_err(|_| LanaError::Constant)?;
                if (index as usize) < chunk.constants.len()
                    && chunk.constants[index as usize].value_type() == ValueType::String
                {
                    return Ok(index);
                }
                return Err(LanaError::Constant);
            }
        }
    }
    let index = chunk.constants.len() as u32;
    chunk.constants.push(Value::String(text.to_string()));
    Ok(index)
}

fn add_hex_string(chunk: &mut Chunk, encoded: &str) -> Result<u32, LanaError> {
    let encoded_length = if encoded == "-" { 0 } else { encoded.len() };
    if encoded_length % 2 != 0 {
        return Err(LanaError::Format);
    }
    let mut decoded = Vec::with_capacity(encoded_length / 2);
    for index in 0..encoded_length / 2 {
        let high = hex_digit(encoded.as_bytes()[index * 2]).ok_or(LanaError::Format)?;
        let low = hex_digit(encoded.as_bytes()[index * 2 + 1]).ok_or(LanaError::Format)?;
        let byte = ((high << 4) | low) as u8;
        if byte == 0 {
            return Err(LanaError::Format);
        }
        decoded.push(byte);
    }
    let index = chunk.constants.len() as u32;
    chunk
        .constants
        .push(Value::String(String::from_utf8_lossy(&decoded).into_owned()));
    Ok(index)
}

/// Emit one instruction line, mirroring `emit_line` in `vm/c/assembler.c`.
fn emit_line(
    chunk: &mut Chunk,
    tokens: &[&str],
    line: u32,
    fixups: &mut Vec<Fixup>,
    function_fixups: &mut Vec<FunctionFixup>,
) -> Result<(), LanaErrorInfo> {
    let mut ins = Instruction::new(
        OpCode::Nop,
        LANA_NO_OPERAND,
        LANA_NO_OPERAND,
        LANA_NO_OPERAND,
        LANA_NO_OPERAND,
        line,
    );
    let ip = chunk.code.len();
    let expect = |count: usize| -> Result<(), LanaErrorInfo> {
        if tokens.len() != count {
            return Err(LanaErrorInfo::new(
                LanaError::Format,
                ip,
                OpCode::Nop as u8,
                line,
                format!("{} expects {} operands", tokens[0], count - 1),
            ));
        }
        Ok(())
    };
    let reg = |token: &str| -> Result<u32, LanaErrorInfo> {
        parse_register(token).ok_or_else(|| {
            LanaErrorInfo::new(
                LanaError::Register,
                ip,
                OpCode::Nop as u8,
                line,
                format!("invalid register {token}"),
            )
        })
    };
    match tokens[0] {
        "NOP" => {
            expect(1)?;
            ins.opcode = OpCode::Nop;
        }
        "HALT" => {
            expect(1)?;
            ins.opcode = OpCode::Halt;
        }
        "LOAD_CONST" => {
            expect(3)?;
            let a = reg(tokens[1])?;
            ins.opcode = OpCode::LoadConst;
            ins.a = a;
            let imm = if tokens[2] == "true" {
                let index = chunk.constants.len() as u32;
                chunk.constants.push(Value::Bool(true));
                index
            } else if tokens[2] == "false" {
                let index = chunk.constants.len() as u32;
                chunk.constants.push(Value::Bool(false));
                index
            } else if tokens[2] == "null" {
                let index = chunk.constants.len() as u32;
                chunk.constants.push(Value::Null);
                index
            } else if let Some(number) = parse_number(tokens[2]) {
                add_number(chunk, number)
            } else {
                let index = chunk.constants.len() as u32;
                chunk.constants.push(Value::String(tokens[2].to_string()));
                index
            };
            ins.imm = imm;
        }
        "LOAD_STRING" => {
            expect(3)?;
            let a = reg(tokens[1])?;
            ins.opcode = OpCode::LoadConst;
            ins.a = a;
            ins.imm = add_hex_string(chunk, tokens[2])?;
        }
        "STATE_NEW" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = number_constant(chunk, tokens[2])?;
            let c = number_constant(chunk, tokens[3])?;
            let imm = number_constant(chunk, tokens[4])?;
            ins.opcode = OpCode::StateNew;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = imm;
        }
        "STATE_BUILD" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            let imm = reg(tokens[4])?;
            ins.opcode = OpCode::StateBuild;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = imm;
        }
        "MIX" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            let imm = reg(tokens[4])?;
            ins.opcode = OpCode::Mix;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = imm;
        }
        "MAP" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let id = transform_id(tokens[3]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Transform,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "unknown transform",
                )
            })?;
            ins.opcode = OpCode::Map;
            ins.a = a;
            ins.b = b;
            ins.c = id;
            ins.imm = 0;
        }
        "SUPPORT" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let imm = sample_count(tokens[3]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Format,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "invalid sample count",
                )
            })?;
            ins.opcode = OpCode::Support;
            ins.a = a;
            ins.b = b;
            ins.c = 0;
            ins.imm = imm;
        }
        "EXPECT" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let id = observable_id(tokens[3]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::UnsupportedOperation,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "unknown observable",
                )
            })?;
            ins.opcode = OpCode::Expect;
            ins.a = a;
            ins.b = b;
            ins.c = 0;
            ins.imm = id;
        }
        "VALIDATE" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            ins.opcode = OpCode::Validate;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = 0;
        }
        "REVISION" => {
            expect(3)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            ins.opcode = OpCode::Revision;
            ins.a = a;
            ins.b = b;
            ins.c = 0;
            ins.imm = 0;
        }
        "MOVE" => {
            expect(3)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            ins.opcode = OpCode::Move;
            ins.a = a;
            ins.b = b;
        }
        "TRANSFORM" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let id = transform_id(tokens[3]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Transform,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "unknown transform",
                )
            })?;
            ins.opcode = OpCode::Transform;
            ins.a = a;
            ins.b = b;
            ins.c = id;
            ins.imm = 0;
        }
        "MEASURE" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let id = measure_id_current(tokens[2]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Measure,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "unknown measure",
                )
            })?;
            let b = reg(tokens[3])?;
            ins.opcode = OpCode::Measure;
            ins.a = a;
            ins.b = b;
            ins.c = id;
            ins.imm = 0;
        }
        "MEASURE_BASIS" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let id = basis_id(tokens[3]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Measure,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "unknown basis",
                )
            })?;
            let imm = measure_id_current(tokens[4]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Measure,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "unknown measure",
                )
            })?;
            ins.opcode = OpCode::MeasureBasis;
            ins.a = a;
            ins.b = b;
            ins.c = id;
            ins.imm = imm;
        }
        "ESTIMATE_MEASURE_PROBABILITY" | "ESTIMATE_MEASURE_DISTRIBUTION" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let id = basis_id(tokens[3]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Measure,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "unknown basis",
                )
            })?;
            let imm = sample_count(tokens[4]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Format,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "invalid sample count",
                )
            })?;
            ins.opcode = if tokens[0] == "ESTIMATE_MEASURE_PROBABILITY" {
                OpCode::EstimateMeasureProbability
            } else {
                OpCode::EstimateMeasureDistribution
            };
            ins.a = a;
            ins.b = b;
            ins.c = id;
            ins.imm = imm;
        }
        "APPEND" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            ins.opcode = OpCode::Append;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = 0;
        }
        "ATTENUATE" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            ins.opcode = OpCode::Attenuate;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = 0;
        }
        "TRACE_DISTANCE" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            ins.opcode = OpCode::TraceDistance;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = 0;
        }
        "APPEND_REDUNDANT" | "APPEND_COMPLEMENTARY" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            let imm = reg(tokens[4])?;
            ins.opcode = if tokens[0] == "APPEND_REDUNDANT" {
                OpCode::AppendRedundant
            } else {
                OpCode::AppendComplementary
            };
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = imm;
        }
        "APPEND_FULL_REDUNDANCY" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            ins.opcode = OpCode::AppendFullRedundancy;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = 0;
        }
        "ADT_BUILD" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let count = parse_number(tokens[3]).ok_or_else(|| {
                LanaErrorInfo::new(LanaError::Format, ip, OpCode::Nop as u8, line, "invalid field count")
            })?;
            if count < 0.0 || count > LANA_MAX_REGISTERS as f64 {
                return Err(LanaErrorInfo::new(
                    LanaError::Limit,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "field count out of range",
                ));
            }
            let imm = number_constant(chunk, tokens[4])?;
            ins.opcode = OpCode::AdtBuild;
            ins.a = a;
            ins.b = b;
            ins.c = count as u32;
            ins.imm = imm;
        }
        "ADT_CASE" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = number_constant(chunk, tokens[2])?;
            if fixups.len() >= LANA_ASSEMBLER_MAX_FIXUPS
                || tokens[3].is_empty()
                || tokens[3].len() >= 64
            {
                return Err(LanaErrorInfo::new(
                    LanaError::Limit,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "too many fixups or invalid label",
                ));
            }
            fixups.push(Fixup {
                name: tokens[3].to_string(),
                instruction: chunk.code.len() as u32,
            });
            ins.opcode = OpCode::AdtCase;
            ins.a = a;
            ins.b = b;
            ins.c = 0;
            ins.imm = 0;
        }
        "ADT_GET" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let index = parse_number(tokens[3]).ok_or_else(|| {
                LanaErrorInfo::new(LanaError::Format, ip, OpCode::Nop as u8, line, "invalid field index")
            })?;
            if index < 0.0 || index > LANA_MAX_REGISTERS as f64 {
                return Err(LanaErrorInfo::new(
                    LanaError::Limit,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "field index out of range",
                ));
            }
            ins.opcode = OpCode::AdtGet;
            ins.a = a;
            ins.b = b;
            ins.c = index as u32;
            ins.imm = 0;
        }
        "LAZY" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let c = reg(tokens[3])?;
            let function_index = match chunk.functions.iter().position(|f| f.name == tokens[2]) {
                Some(index) => index as u32,
                None => {
                    if function_fixups.len() >= LANA_ASSEMBLER_MAX_FIXUPS || tokens[2].len() >= 64 {
                        return Err(LanaErrorInfo::new(
                            LanaError::Limit,
                            ip,
                            OpCode::Nop as u8,
                            line,
                            "too many function fixups",
                        ));
                    }
                    function_fixups.push(FunctionFixup {
                        name: tokens[2].to_string(),
                        instruction: chunk.code.len() as u32,
                    });
                    0
                }
            };
            ins.opcode = OpCode::Lazy;
            ins.a = a;
            ins.b = function_index;
            ins.c = c;
            ins.imm = 0;
        }
        "FORCE" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            ins.opcode = OpCode::Force;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = 0;
        }
        "BOOTSTRAP" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            let function_index = match chunk.functions.iter().position(|f| f.name == tokens[4]) {
                Some(index) => index as u32,
                None => {
                    if function_fixups.len() >= LANA_ASSEMBLER_MAX_FIXUPS || tokens[4].len() >= 64 {
                        return Err(LanaErrorInfo::new(
                            LanaError::Limit,
                            ip,
                            OpCode::Nop as u8,
                            line,
                            "too many function fixups",
                        ));
                    }
                    function_fixups.push(FunctionFixup {
                        name: tokens[4].to_string(),
                        instruction: chunk.code.len() as u32,
                    });
                    0
                }
            };
            ins.opcode = OpCode::Bootstrap;
            ins.a = a;
            ins.b = function_index;
            ins.c = c;
            ins.imm = b;
        }
        "JOINT_BUILD" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let joint_count = sample_count(tokens[3]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Register,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "invalid joint count",
                )
            })?;
            if b + joint_count > LANA_MAX_REGISTERS {
                return Err(LanaErrorInfo::new(
                    LanaError::Register,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "joint count out of range",
                ));
            }
            let imm = string_constant(chunk, tokens[4])?;
            ins.opcode = OpCode::JointBuild;
            ins.a = a;
            ins.b = b;
            ins.c = joint_count;
            ins.imm = imm;
        }
        "JOINT_PROJECT" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = string_constant(chunk, tokens[3])?;
            ins.opcode = OpCode::JointProject;
            ins.a = a;
            ins.b = b;
            ins.c = c;
        }
        "JOINT_CONDITION" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = string_constant(chunk, tokens[3])?;
            let imm = reg(tokens[4])?;
            ins.opcode = OpCode::JointCondition;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = imm;
        }
        "JOINT_BUILD_FINITE" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = string_constant(chunk, tokens[3])?;
            ins.opcode = OpCode::JointBuildFinite;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = 0;
        }
        "JOINT_RENAME" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = string_constant(chunk, tokens[3])?;
            let imm = string_constant(chunk, tokens[4])?;
            ins.opcode = OpCode::JointRename;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = imm;
        }
        "POSSIBILITY_BUILD" | "INFO_SAMPLE" => {
            expect(3)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            ins.opcode = if tokens[0] == "POSSIBILITY_BUILD" {
                OpCode::PossibilityBuild
            } else {
                OpCode::InfoSample
            };
            ins.a = a;
            ins.b = b;
            ins.c = 0;
            ins.imm = 0;
        }
        "PATH_SPLIT" => {
            expect(3)?;
            let a = reg(tokens[1])?;
            ins.opcode = OpCode::PathSplit;
            ins.a = a;
            ins.b = 0;
            ins.c = 0;
            if fixups.len() >= LANA_ASSEMBLER_MAX_FIXUPS
                || tokens[2].is_empty()
                || tokens[2].len() >= 64
            {
                return Err(LanaErrorInfo::new(
                    LanaError::Limit,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "too many fixups or invalid label",
                ));
            }
            fixups.push(Fixup {
                name: tokens[2].to_string(),
                instruction: chunk.code.len() as u32,
            });
        }
        "PATH_JOIN" => {
            expect(1)?;
            ins.opcode = OpCode::PathJoin;
            ins.a = 0;
            ins.b = 0;
            ins.c = 0;
            ins.imm = 0;
        }
        "OBSERVE" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = string_constant(chunk, tokens[3])?;
            let imm = reg(tokens[4])?;
            ins.opcode = OpCode::Observe;
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = imm;
        }
        "EVIDENCE" | "ASSUME" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = string_constant(chunk, tokens[3])?;
            ins.opcode = if tokens[0] == "EVIDENCE" {
                OpCode::Evidence
            } else {
                OpCode::Assume
            };
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = 0;
        }
        "DERIVATION" | "EXPLAIN" => {
            expect(3)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            ins.opcode = if tokens[0] == "DERIVATION" {
                OpCode::Derivation
            } else {
                OpCode::Explain
            };
            ins.a = a;
            ins.b = b;
            ins.c = 0;
            ins.imm = 0;
        }
        "JOINT_SAMPLE" | "RESOLVE" => {
            expect(3)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            ins.opcode = if tokens[0] == "JOINT_SAMPLE" {
                OpCode::JointSample
            } else {
                OpCode::Resolve
            };
            ins.a = a;
            ins.b = b;
            ins.c = 0;
            ins.imm = 0;
        }
        "SAMPLE_STATE_DIST" => {
            expect(3)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            ins.opcode = OpCode::SampleStateDist;
            ins.a = a;
            ins.b = b;
            ins.c = 0;
            ins.imm = 0;
        }
        "GET_FIELD" | "GET_INDEX" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let id = if tokens[0] == "GET_FIELD" {
                field_id(tokens[2])
            } else {
                index_id(tokens[2])
            }
            .ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Format,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "unknown field",
                )
            })?;
            let b = reg(tokens[3])?;
            ins.opcode = if tokens[0] == "GET_FIELD" {
                OpCode::GetField
            } else {
                OpCode::GetIndex
            };
            ins.a = a;
            ins.b = b;
            ins.c = id;
        }
        "SET_INDEX" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let id = index_id(tokens[2]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Format,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "unknown index",
                )
            })?;
            let c = reg(tokens[3])?;
            ins.opcode = OpCode::SetIndex;
            ins.a = a;
            ins.b = id;
            ins.c = c;
        }
        "HISTORY" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[3])?;
            let id = match tokens[2] {
                "latest" => LANA_HISTORY_LATEST,
                "duration" => LANA_HISTORY_DURATION,
                _ => {
                    return Err(LanaErrorInfo::new(
                        LanaError::History,
                        ip,
                        OpCode::Nop as u8,
                        line,
                        "unknown history mode",
                    ))
                }
            };
            ins.opcode = OpCode::HistoryConfig;
            ins.a = a;
            ins.b = b;
            ins.c = id;
        }
        "PREVIOUS" | "CHANGE" | "VELOCITY" => {
            expect(3)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            ins.opcode = match tokens[0] {
                "PREVIOUS" => OpCode::Previous,
                "CHANGE" => OpCode::Change,
                _ => OpCode::Velocity,
            };
            ins.a = a;
            ins.b = b;
        }
        "BINARY" | "COMPARE" => {
            expect(5)?;
            let a = reg(tokens[1])?;
            let id = if tokens[0] == "BINARY" {
                binary_id(tokens[2])
            } else {
                compare_id(tokens[2])
            }
            .ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Format,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "unknown operator",
                )
            })?;
            let b = reg(tokens[3])?;
            let c = reg(tokens[4])?;
            ins.opcode = if tokens[0] == "BINARY" {
                OpCode::Binary
            } else {
                OpCode::Compare
            };
            ins.a = a;
            ins.b = b;
            ins.c = c;
            ins.imm = id;
        }
        "UNARY" => {
            expect(4)?;
            let a = reg(tokens[2])?;
            let b = reg(tokens[3])?;
            let id = match tokens[1] {
                "-" => 0,
                "!" => 1,
                _ => {
                    return Err(LanaErrorInfo::new(
                        LanaError::Format,
                        ip,
                        OpCode::Nop as u8,
                        line,
                        "unknown unary operator",
                    ))
                }
            };
            ins.opcode = OpCode::Unary;
            ins.a = a;
            ins.b = b;
            ins.imm = id;
        }
        "JUMP" | "JUMP_IF_TRUE" | "JUMP_IF_FALSE" => {
            let label_token;
            if tokens[0] == "JUMP" {
                expect(2)?;
                ins.opcode = OpCode::Jump;
                label_token = 1;
            } else {
                expect(3)?;
                let a = reg(tokens[1])?;
                ins.a = a;
                label_token = 2;
                ins.opcode = if tokens[0] == "JUMP_IF_TRUE" {
                    OpCode::JumpIfTrue
                } else {
                    OpCode::JumpIfFalse
                };
            }
            if fixups.len() >= LANA_ASSEMBLER_MAX_FIXUPS
                || tokens[label_token].is_empty()
                || tokens[label_token].len() >= 64
            {
                return Err(LanaErrorInfo::new(
                    LanaError::Limit,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "too many fixups or invalid label",
                ));
            }
            fixups.push(Fixup {
                name: tokens[label_token].to_string(),
                instruction: chunk.code.len() as u32,
            });
        }
        "ARRAY_NEW" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let number = tokens[3].parse::<f64>().unwrap_or(0.0);
            if number < 0.0 || number > LANA_MAX_REGISTERS as f64 {
                return Err(LanaErrorInfo::new(
                    LanaError::Limit,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "array count out of range",
                ));
            }
            ins.opcode = OpCode::ArrayNew;
            ins.a = a;
            ins.b = b;
            ins.c = number as u32;
        }
        "ARRAY_GET" | "ARRAY_SET" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            ins.opcode = if tokens[0] == "ARRAY_GET" {
                OpCode::ArrayGet
            } else {
                OpCode::ArraySet
            };
            ins.a = a;
            ins.b = b;
            ins.c = c;
        }
        "CALL" | "FORK" => {
            expect(5)?;
            let a = reg(tokens[2])?;
            let number = tokens[3].parse::<f64>().unwrap_or(0.0);
            let b = reg(tokens[4])?;
            let function_index = match chunk.functions.iter().position(|f| f.name == tokens[1]) {
                Some(index) => index as u32,
                None => {
                    if function_fixups.len() >= LANA_ASSEMBLER_MAX_FIXUPS || tokens[1].len() >= 64 {
                        return Err(LanaErrorInfo::new(
                            LanaError::Limit,
                            ip,
                            OpCode::Nop as u8,
                            line,
                            "too many function fixups",
                        ));
                    }
                    function_fixups.push(FunctionFixup {
                        name: tokens[1].to_string(),
                        instruction: chunk.code.len() as u32,
                    });
                    0
                }
            };
            if tokens[0] == "CALL" {
                if number < 0.0 || number > LANA_MAX_REGISTERS as f64 {
                    return Err(LanaErrorInfo::new(
                        LanaError::Format,
                        ip,
                        OpCode::Nop as u8,
                        line,
                        "invalid argument count",
                    ));
                }
                ins.opcode = OpCode::Call;
                ins.a = b;
                ins.b = function_index;
                ins.c = a;
                ins.imm = number as u32;
            } else {
                if number < 0.0 || number > LANA_MAX_REGISTERS as f64 || number.fract() != 0.0 {
                    return Err(LanaErrorInfo::new(
                        LanaError::Format,
                        ip,
                        OpCode::Nop as u8,
                        line,
                        "invalid argument count",
                    ));
                }
                ins.opcode = OpCode::Fork;
                ins.a = b;
                ins.b = function_index;
                ins.c = a;
                ins.imm = number as u32;
            }
        }
        "JOIN" | "JOIN_ALL" => {
            expect(3)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            ins.opcode = if tokens[0] == "JOIN" {
                OpCode::Join
            } else {
                OpCode::JoinAll
            };
            ins.a = a;
            ins.b = b;
        }
        "JOIN_TIMEOUT" => {
            expect(4)?;
            let a = reg(tokens[1])?;
            let b = reg(tokens[2])?;
            let c = reg(tokens[3])?;
            ins.opcode = OpCode::JoinTimeout;
            ins.a = a;
            ins.b = b;
            ins.c = c;
        }
        "CANCEL" => {
            expect(2)?;
            let a = reg(tokens[1])?;
            ins.opcode = OpCode::Cancel;
            ins.a = a;
        }
        "TASKGROUP_ENTER" | "TASKGROUP_EXIT" => {
            expect(1)?;
            ins.opcode = if tokens[0] == "TASKGROUP_ENTER" {
                OpCode::TaskgroupEnter
            } else {
                OpCode::TaskgroupExit
            };
        }
        "HOST_CALL" => {
            expect(5)?;
            let id = host_call_id(tokens[1]).ok_or_else(|| {
                LanaErrorInfo::new(
                    LanaError::Format,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "unknown host call",
                )
            })?;
            let a = reg(tokens[2])?;
            let number = tokens[3].parse::<f64>().unwrap_or(0.0);
            let b = reg(tokens[4])?;
            if number < 0.0 || number > LANA_MAX_REGISTERS as f64 || number.fract() != 0.0 {
                return Err(LanaErrorInfo::new(
                    LanaError::Format,
                    ip,
                    OpCode::Nop as u8,
                    line,
                    "invalid argument count",
                ));
            }
            ins.opcode = OpCode::HostCall;
            ins.a = b;
            ins.b = id;
            ins.c = a;
            ins.imm = number as u32;
        }
        "RETURN" | "PRINT" => {
            expect(2)?;
            let a = reg(tokens[1])?;
            ins.opcode = if tokens[0] == "RETURN" {
                OpCode::Return
            } else {
                OpCode::Print
            };
            ins.a = a;
        }
        _ => {
            return Err(LanaErrorInfo::new(
                LanaError::Opcode,
                ip,
                OpCode::Nop as u8,
                line,
                format!("unknown instruction {}", tokens[0]),
            ));
        }
    }
    chunk.code.push(ins);
    Ok(())
}

/// Assemble LABC assembly text into a verified chunk, mirroring
/// `lana_assemble_file` in `vm/c/assembler.c`.
pub fn assemble(text: &str) -> Result<Chunk, LanaErrorInfo> {
    let mut chunk = Chunk::new(LABC_VERSION, 0);
    let mut labels: Vec<Label> = Vec::new();
    let mut fixups: Vec<Fixup> = Vec::new();
    let mut function_fixups: Vec<FunctionFixup> = Vec::new();
    let mut line = 0u32;
    let mut source_line = 0u32;

    for raw_line in text.lines() {
        line += 1;
        let mut cursor = raw_line;
        if let Some(comment) = cursor.find('#') {
            cursor = &cursor[..comment];
        }
        let cursor = cursor.trim();
        if cursor.is_empty() {
            continue;
        }
        if cursor.ends_with(':') {
            let name = cursor[..cursor.len() - 1].trim();
            if labels.len() >= LANA_ASSEMBLER_MAX_FIXUPS || name.is_empty() || name.len() >= 64 {
                return Err(LanaErrorInfo::new(
                    LanaError::Limit,
                    line as usize,
                    OpCode::Nop as u8,
                    line,
                    "too many labels or invalid label",
                ));
            }
            labels.push(Label {
                name: name.to_string(),
                offset: chunk.code.len() as u32,
            });
            continue;
        }
        let tokens: Vec<&str> = cursor
            .split(|c: char| c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ',')
            .filter(|token| !token.is_empty())
            .collect();
        if tokens.is_empty() {
            continue;
        }
        if tokens.len() > 16 {
            return Err(LanaErrorInfo::new(
                LanaError::Format,
                line as usize,
                OpCode::Nop as u8,
                line,
                "too many tokens",
            ));
        }
        match tokens[0] {
            ".function" => {
                if tokens.len() != 4 {
                    return Err(LanaErrorInfo::new(
                        LanaError::Format,
                        line as usize,
                        OpCode::Nop as u8,
                        line,
                        ".function expects 3 operands",
                    ));
                }
                let arity: u32 = tokens[2].parse().map_err(|_| {
                    LanaErrorInfo::new(
                        LanaError::Format,
                        line as usize,
                        OpCode::Nop as u8,
                        line,
                        "invalid arity",
                    )
                })?;
                let registers: u32 = tokens[3].parse().map_err(|_| {
                    LanaErrorInfo::new(
                        LanaError::Format,
                        line as usize,
                        OpCode::Nop as u8,
                        line,
                        "invalid register count",
                    )
                })?;
                if registers > LANA_MAX_REGISTERS || arity > registers {
                    return Err(LanaErrorInfo::new(
                        LanaError::Format,
                        line as usize,
                        OpCode::Nop as u8,
                        line,
                        "invalid function metadata",
                    ));
                }
                chunk.functions.push(Function {
                    name: tokens[1].to_string(),
                    entry: chunk.code.len() as u32,
                    register_count: registers,
                    arity,
                });
                continue;
            }
            ".version" => {
                if tokens.len() != 2 {
                    return Err(LanaErrorInfo::new(
                        LanaError::Format,
                        line as usize,
                        OpCode::Nop as u8,
                        line,
                        ".version expects 1 operand",
                    ));
                }
                let version: u32 = tokens[1].parse().map_err(|_| {
                    LanaErrorInfo::new(
                        LanaError::Format,
                        line as usize,
                        OpCode::Nop as u8,
                        line,
                        "invalid version",
                    )
                })?;
                if version != LABC_VERSION && version != LABC_VERSION_1 {
                    return Err(LanaErrorInfo::new(
                        LanaError::IncompatibleFormat,
                        line as usize,
                        OpCode::Nop as u8,
                        line,
                        "unsupported version",
                    ));
                }
                chunk.version = version;
                continue;
            }
            ".line" => {
                if tokens.len() != 2 {
                    return Err(LanaErrorInfo::new(
                        LanaError::Format,
                        line as usize,
                        OpCode::Nop as u8,
                        line,
                        ".line expects 1 operand",
                    ));
                }
                source_line = tokens[1].parse().map_err(|_| {
                    LanaErrorInfo::new(
                        LanaError::Format,
                        line as usize,
                        OpCode::Nop as u8,
                        line,
                        "invalid line number",
                    )
                })?;
                continue;
            }
            _ => {}
        }
        let instruction_line = if source_line == 0 { line } else { source_line };
        emit_line(&mut chunk, &tokens, instruction_line, &mut fixups, &mut function_fixups)?;
    }

    for fixup in &fixups {
        match labels.iter().find(|label| label.name == fixup.name) {
            Some(label) => {
                chunk.code[fixup.instruction as usize].imm = label.offset;
            }
            None => {
                return Err(LanaErrorInfo::new(
                    LanaError::Jump,
                    fixup.instruction as usize,
                    OpCode::Jump as u8,
                    0,
                    format!("unknown label {}", fixup.name),
                ));
            }
        }
    }

    for fixup in &function_fixups {
        match chunk.functions.iter().position(|f| f.name == fixup.name) {
            Some(index) => {
                chunk.code[fixup.instruction as usize].b = index as u32;
            }
            None => {
                return Err(LanaErrorInfo::new(
                    LanaError::Format,
                    fixup.instruction as usize,
                    OpCode::Call as u8,
                    0,
                    format!("unknown function {}", fixup.name),
                ));
            }
        }
    }

    crate::verifier::verify(&chunk)?;
    Ok(chunk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_isa_ops() {
        let text = "\
.function main 0 8
LOAD_CONST R0 0.4
LOAD_CONST R1 0.2
LOAD_CONST R2 0.8
LOAD_CONST R3 -0.1
STATE_NEW R4 K0 K1 K1
STATE_NEW R5 K2 K3 K3
APPEND R4 R5 R6
MAP R7 R6 neutralize
SUPPORT R8 R6 4
EXPECT R9 R6 probability
VALIDATE R10 R9 R11
REVISION R12 R7
HALT
";
        let chunk = assemble(text).expect("assembly should succeed");
        assert_eq!(chunk.version, LABC_VERSION);
        assert_eq!(chunk.code.len(), 13);
        assert_eq!(chunk.code[6].opcode, OpCode::Append);
        assert_eq!(chunk.code[7].opcode, OpCode::Map);
        assert_eq!(chunk.code[7].c, LANA_TRANSFORM_NEUTRALIZE);
        assert_eq!(chunk.code[8].opcode, OpCode::Support);
        assert_eq!(chunk.code[8].imm, 4);
        assert_eq!(chunk.code[9].opcode, OpCode::Expect);
        assert_eq!(chunk.code[9].imm, LANA_OBSERVABLE_PROBABILITY);
        assert_eq!(chunk.code[10].opcode, OpCode::Validate);
        assert_eq!(chunk.code[11].opcode, OpCode::Revision);
        assert_eq!(chunk.code[12].opcode, OpCode::Halt);
    }

    #[test]
    fn assembles_labels_and_functions() {
        let text = "\
.function main 0 4
.function helper 1 2
LOAD_CONST R0 1
JUMP done
LOAD_CONST R1 2
done:
CALL helper R2 1 R0
HALT
";
        let chunk = assemble(text).expect("assembly should succeed");
        assert_eq!(chunk.functions.len(), 2);
        assert_eq!(chunk.code[1].opcode, OpCode::Jump);
        assert_eq!(chunk.code[1].imm, 3);
        assert_eq!(chunk.code[3].opcode, OpCode::Call);
        assert_eq!(chunk.code[3].b, 1);
    }

    #[test]
    fn rejects_unknown_transform() {
        let text = "\
.function main 0 4
LOAD_CONST R0 0.5
LOAD_CONST R1 0.0
STATE_NEW R2 K0 K1 K1
MAP R3 R2 bogus
HALT
";
        let error = assemble(text).unwrap_err();
        assert_eq!(error.code, LanaError::Transform);
    }

    #[test]
    fn rejects_unknown_observable() {
        let text = "\
.function main 0 4
LOAD_CONST R0 0.5
LOAD_CONST R1 0.0
STATE_NEW R2 K0 K1 K1
EXPECT R3 R2 bogus
HALT
";
        let error = assemble(text).unwrap_err();
        assert_eq!(error.code, LanaError::UnsupportedOperation);
    }

    #[test]
    fn rejects_zero_support_limit() {
        let text = "\
.function main 0 4
LOAD_CONST R0 0.5
LOAD_CONST R1 0.0
STATE_NEW R2 K0 K1 K1
SUPPORT R3 R2 0
HALT
";
        let error = assemble(text).unwrap_err();
        assert_eq!(error.code, LanaError::Format);
    }

    #[test]
    fn rejects_unknown_label() {
        let text = "\
.function main 0 2
JUMP nowhere
HALT
";
        let error = assemble(text).unwrap_err();
        assert_eq!(error.code, LanaError::Jump);
    }

    #[test]
    fn rejects_unknown_instruction() {
        let text = "\
.function main 0 1
BOGUS R0
HALT
";
        let error = assemble(text).unwrap_err();
        assert_eq!(error.code, LanaError::Opcode);
    }

    #[test]
    fn disassembly_matches_c11_for_isa_ops() {
        // Expected output captured from `lanavm dis` on the C11-assembled
        // chunk; the Rust assembler must produce the identical chunk.
        let text = "\
.function main 0 8
LOAD_CONST R0 0.4
LOAD_CONST R1 0.2
LOAD_CONST R2 0.8
LOAD_CONST R3 -0.1
STATE_NEW R4 K0 K1 K1
STATE_NEW R5 K2 K3 K3
APPEND R4 R5 R6
MAP R7 R6 neutralize
SUPPORT R8 R6 4
EXPECT R9 R6 probability
VALIDATE R10 R9 R11
REVISION R12 R7
HALT
";
        let chunk = assemble(text).expect("assembly should succeed");
        let expected = "\
LABC v2 entry=0 constants=4 functions=1 instructions=13
0000 LOAD_CONST           R0 constant[0]  ; line 2
0001 LOAD_CONST           R1 constant[1]  ; line 3
0002 LOAD_CONST           R2 constant[2]  ; line 4
0003 LOAD_CONST           R3 constant[3]  ; line 5
0004 STATE_NEW            R4 p=K0(0.4) d_re=K1(0.2) d_im=K1(0.2)  ; line 6
0005 STATE_NEW            R5 p=K2(0.8) d_re=K3(-0.1) d_im=K3(-0.1)  ; line 7
0006 APPEND               R4 R5 -> R6  ; line 8
0007 MAP                  R7 <- map(R6, neutralize)  ; line 9
0008 SUPPORT              R8 <- support(R6, limit=4)  ; line 10
0009 EXPECT               R9 <- expect(R6, probability)  ; line 11
0010 VALIDATE             R10 <- validate(R9, R11)  ; line 12
0011 REVISION             R12 <- revision(R7)  ; line 13
0012 HALT                   ; line 14
";
        assert_eq!(crate::disassembler::disassemble(&chunk), expected);
    }
}
