//! LABC disassembler, mirroring `lana_disassemble_instruction` and
//! `lana_disassemble` in `vm/c/bytecode.c`.
//!
//! Output is byte-identical to the C11 disassembler for the same chunk.

use crate::chunk::Chunk;
use crate::opcode::OpCode;
use crate::value::Value;

const LANA_TRANSFORM_INVERT: u32 = 0;
const LANA_OBSERVABLE_PROBABILITY: u32 = 0;
const LANA_MEASURE_PROBABILITY: u32 = 0;
const LANA_MEASURE_DISTRIBUTION: u32 = 1;
const LANA_MEASURE_SAMPLE: u32 = 2;
const LANA_MEASURE_BASIS_COMPUTATIONAL: u32 = 0;
const LANA_MEASURE_BASIS_X: u32 = 1;
const LANA_MEASURE_BASIS_Y: u32 = 2;

fn measure_name(id: u32) -> &'static str {
    match id {
        LANA_MEASURE_PROBABILITY => "probability",
        LANA_MEASURE_DISTRIBUTION => "distribution",
        LANA_MEASURE_SAMPLE => "sample",
        _ => "unknown",
    }
}

fn basis_name(id: u32) -> &'static str {
    match id {
        LANA_MEASURE_BASIS_COMPUTATIONAL => "computational",
        LANA_MEASURE_BASIS_X => "x",
        LANA_MEASURE_BASIS_Y => "y",
        _ => "unknown",
    }
}

fn number(constants: &[Value], index: u32) -> f64 {
    match constants.get(index as usize) {
        Some(Value::Number(value)) => *value,
        _ => 0.0,
    }
}

/// Format a number like C's `%.12g`, matching the STATE_NEW disassembly in
/// `vm/c/bytecode.c` and `lana_value_print` in `vm/c/value.c`.
pub fn format_g(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() { "inf".to_string() } else { "-inf".to_string() };
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let negative = value.is_sign_negative();
    let abs = value.abs();
    let exponent = abs.log10().floor() as i32;
    let use_scientific = exponent < -4 || exponent >= 12;
    let mut digits;
    if use_scientific {
        let mantissa = abs / 10f64.powi(exponent);
        digits = format!("{:.11}", mantissa);
        while digits.contains('.') && digits.ends_with('0') {
            digits.pop();
        }
        if digits.ends_with('.') {
            digits.pop();
        }
        let sign = if exponent < 0 { "-" } else { "+" };
        let exp_str = format!("{}{:02}", sign, exponent.abs());
        format!("{}{}e{}", if negative { "-" } else { "" }, digits, exp_str)
    } else {
        let decimals = (12 - 1 - exponent).max(0) as usize;
        digits = format!("{:.*}", decimals, abs);
        while digits.contains('.') && digits.ends_with('0') {
            digits.pop();
        }
        if digits.ends_with('.') {
            digits.pop();
        }
        format!("{}{}", if negative { "-" } else { "" }, digits)
    }
}

/// Disassemble a single instruction, mirroring `lana_disassemble_instruction`.
pub fn disassemble_instruction(chunk: &Chunk, offset: usize) -> String {
    let ins = &chunk.code[offset];
    let mut out = String::new();
    out.push_str(&format!("{offset:04} {:<20} ", ins.opcode.name()));
    match ins.opcode {
        OpCode::StateNew => {
            out.push_str(&format!(
                "R{} p=K{}({}) d_re=K{}({}) d_im=K{}({})",
                ins.a, ins.b, format_g(number(&chunk.constants, ins.b)),
                ins.c, format_g(number(&chunk.constants, ins.c)),
                ins.imm, format_g(number(&chunk.constants, ins.imm))));
        }
        OpCode::StateBuild => {
            out.push_str(&format!("R{} p=R{} d_re=R{} d_im=R{}", ins.imm, ins.a, ins.b, ins.c));
        }
        OpCode::Mix => {
            out.push_str(&format!("R{} <- mix(R{}, R{}, w=R{})", ins.a, ins.b, ins.c, ins.imm));
        }
        OpCode::Map => {
            let name = if ins.c == LANA_TRANSFORM_INVERT { "invert" } else { "neutralize" };
            out.push_str(&format!("R{} <- map(R{}, {})", ins.a, ins.b, name));
        }
        OpCode::Support => {
            out.push_str(&format!("R{} <- support(R{}, limit={})", ins.a, ins.b, ins.imm));
        }
        OpCode::Expect => {
            let name = if ins.imm == LANA_OBSERVABLE_PROBABILITY { "probability" } else { "unknown" };
            out.push_str(&format!("R{} <- expect(R{}, {})", ins.a, ins.b, name));
        }
        OpCode::Validate => {
            out.push_str(&format!("R{} <- validate(R{}, R{})", ins.a, ins.b, ins.c));
        }
        OpCode::Revision => {
            out.push_str(&format!("R{} <- revision(R{})", ins.a, ins.b));
        }
        OpCode::Transform => {
            let name = if ins.c == LANA_TRANSFORM_INVERT { "invert" } else { "neutralize" };
            out.push_str(&format!("R{} <- {}(R{})", ins.a, name, ins.b));
        }
        OpCode::Measure => {
            out.push_str(&format!("R{} {} -> R{}", ins.a, measure_name(ins.c), ins.b));
        }
        OpCode::Append => {
            out.push_str(&format!("R{} R{} -> R{}", ins.a, ins.b, ins.c));
        }
        OpCode::Attenuate => {
            out.push_str(&format!("R{} <- attenuate(R{}, factor=R{})", ins.a, ins.b, ins.c));
        }
        OpCode::TraceDistance => {
            out.push_str(&format!("R{} <- trace_distance(R{}, R{})", ins.a, ins.b, ins.c));
        }
        OpCode::AppendRedundant => {
            out.push_str(&format!("R{} R{} redundant(strength=R{}) -> R{}", ins.a, ins.b, ins.imm, ins.c));
        }
        OpCode::AppendFullRedundancy => {
            out.push_str(&format!("R{} R{} full_redundancy -> R{}", ins.a, ins.b, ins.c));
        }
        OpCode::AppendComplementary => {
            out.push_str(&format!("R{} R{} complementary(strength=R{}) -> R{}", ins.a, ins.b, ins.imm, ins.c));
        }
        OpCode::AdtBuild => {
            out.push_str(&format!("R{}..R{} variant=K{} -> R{}", ins.b, ins.b + ins.c - 1, ins.imm, ins.a));
        }
        OpCode::AdtCase => {
            out.push_str(&format!("R{} variant=K{} -> {}", ins.a, ins.b, ins.imm));
        }
        OpCode::AdtGet => {
            out.push_str(&format!("R{} <- R{}[{}]", ins.a, ins.b, ins.c));
        }
        OpCode::Lazy => {
            out.push_str(&format!("R{} <- lazy(function[{}], bound=R{})", ins.a, ins.b, ins.c));
        }
        OpCode::Force => {
            out.push_str(&format!("R{} <- force(R{}, index=R{})", ins.a, ins.b, ins.c));
        }
        OpCode::SampleStateDist => {
            out.push_str(&format!("R{} -> R{}", ins.a, ins.b));
        }
        OpCode::MeasureBasis => {
            out.push_str(&format!(
                "R{} basis={} mode={} -> R{}",
                ins.a, basis_name(ins.c), measure_name(ins.imm), ins.b));
        }
        OpCode::EstimateMeasureProbability | OpCode::EstimateMeasureDistribution => {
            out.push_str(&format!(
                "R{} basis={} samples={} -> R{}",
                ins.a, basis_name(ins.c), ins.imm, ins.b));
        }
        OpCode::JointBuild => {
            out.push_str(&format!(
                "R{}..R{} descriptor[{}] -> R{}",
                ins.b, ins.b + ins.c - 1, ins.imm, ins.a));
        }
        OpCode::JointProject => {
            out.push_str(&format!("R{} names[{}] -> R{}", ins.a, ins.c, ins.b));
        }
        OpCode::JointCondition => {
            out.push_str(&format!("R{} condition[{}]=R{} -> R{}", ins.a, ins.c, ins.imm, ins.b));
        }
        OpCode::JointSample | OpCode::Resolve | OpCode::PossibilityBuild
        | OpCode::InfoSample | OpCode::Derivation | OpCode::Explain
        | OpCode::Join | OpCode::JoinAll => {
            out.push_str(&format!("R{} -> R{}", ins.a, ins.b));
        }
        OpCode::JointBuildFinite => {
            out.push_str(&format!("rows=R{} names[{}] -> R{}", ins.a, ins.c, ins.b));
        }
        OpCode::JointRename => {
            out.push_str(&format!("R{} name[{}]->name[{}] -> R{}", ins.a, ins.c, ins.imm, ins.b));
        }
        OpCode::PathSplit => {
            out.push_str(&format!("R{} false->{}", ins.a, ins.imm));
        }
        OpCode::PathJoin => {
            out.push_str("join");
        }
        OpCode::Observe => {
            out.push_str(&format!("R{} name[{}]=R{} -> R{}", ins.a, ins.c, ins.imm, ins.b));
        }
        OpCode::Evidence | OpCode::Assume => {
            out.push_str(&format!("R{} label[{}] -> R{}", ins.a, ins.c, ins.b));
        }
        OpCode::Fork => {
            out.push_str(&format!("function[{}] R{} argc={} -> R{}", ins.b, ins.c, ins.imm, ins.a));
        }
        OpCode::JoinTimeout => {
            out.push_str(&format!("R{} timeout=R{} -> R{}", ins.a, ins.b, ins.c));
        }
        OpCode::Cancel => {
            out.push_str(&format!("R{}", ins.a));
        }
        OpCode::LoadConst => {
            out.push_str(&format!("R{} constant[{}]", ins.a, ins.imm));
        }
        OpCode::Move => {
            out.push_str(&format!("R{} <- R{}", ins.a, ins.b));
        }
        OpCode::Jump => {
            out.push_str(&format!("-> {}", ins.imm));
        }
        OpCode::JumpIfTrue | OpCode::JumpIfFalse => {
            out.push_str(&format!("R{} -> {}", ins.a, ins.imm));
        }
        OpCode::Print | OpCode::Return => {
            out.push_str(&format!("R{}", ins.a));
        }
        OpCode::Halt | OpCode::Nop => {}
        _ => {
            out.push_str(&format!("a={} b={} c={} imm={}", ins.a, ins.b, ins.c, ins.imm));
        }
    }
    out.push_str(&format!("  ; line {}\n", ins.line));
    out
}

/// Disassemble a whole chunk, mirroring `lana_disassemble`.
pub fn disassemble(chunk: &Chunk) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "LABC v{} entry={} constants={} functions={} instructions={}\n",
        chunk.version, chunk.entry, chunk.constants.len(), chunk.functions.len(), chunk.code.len()));
    for offset in 0..chunk.code.len() {
        out.push_str(&disassemble_instruction(chunk, offset));
    }
    out
}
