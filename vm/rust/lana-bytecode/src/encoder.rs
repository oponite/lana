//! LABC binary encoding, inverse of the loader for verified chunks.

use crate::chunk::Chunk;
use crate::value::Value;

pub fn encode(chunk: &Chunk) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"LABC");
    out.extend_from_slice(&chunk.version.to_le_bytes());
    out.extend_from_slice(&(chunk.constants.len() as u32).to_le_bytes());
    out.extend_from_slice(&(chunk.functions.len() as u32).to_le_bytes());
    out.extend_from_slice(&(chunk.code.len() as u32).to_le_bytes());
    out.extend_from_slice(&chunk.entry.to_le_bytes());
    for constant in &chunk.constants {
        out.push(constant.value_type() as u8);
        match constant {
            Value::Null => {}
            Value::Number(number) => out.extend_from_slice(&number.to_bits().to_le_bytes()),
            Value::Bool(boolean) => out.push(u8::from(*boolean)),
            Value::String(string) => {
                out.extend_from_slice(&(string.len() as u32).to_le_bytes());
                out.extend_from_slice(string.as_bytes());
            }
        }
    }
    for function in &chunk.functions {
        out.extend_from_slice(&(function.name.len() as u32).to_le_bytes());
        out.extend_from_slice(function.name.as_bytes());
        out.extend_from_slice(&function.entry.to_le_bytes());
        out.extend_from_slice(&function.register_count.to_le_bytes());
        out.extend_from_slice(&function.arity.to_le_bytes());
    }
    for instruction in &chunk.code {
        out.push(instruction.opcode as u8);
        out.extend_from_slice(&instruction.a.to_le_bytes());
        out.extend_from_slice(&instruction.b.to_le_bytes());
        out.extend_from_slice(&instruction.c.to_le_bytes());
        out.extend_from_slice(&instruction.imm.to_le_bytes());
        out.extend_from_slice(&instruction.line.to_le_bytes());
    }
    out
}
