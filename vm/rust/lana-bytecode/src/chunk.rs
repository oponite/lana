//! Chunk model, mirroring `LanaChunk` in `vm/include/bytecode.h`.

use crate::opcode::OpCode;
use crate::value::Value;

/// A single instruction: opcode byte plus four 32-bit operands and a source line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction {
    pub opcode: OpCode,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub imm: u32,
    pub line: u32,
}

impl Instruction {
    pub fn new(opcode: OpCode, a: u32, b: u32, c: u32, imm: u32, line: u32) -> Self {
        Self { opcode, a, b, c, imm, line }
    }
}

/// A named function entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub entry: u32,
    pub register_count: u32,
    pub arity: u32,
}

/// A verified LABC chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub version: u32,
    pub entry: u32,
    pub code: Vec<Instruction>,
    pub constants: Vec<Value>,
    pub functions: Vec<Function>,
}

impl Chunk {
    pub fn new(version: u32, entry: u32) -> Self {
        Self { version, entry, code: Vec::new(), constants: Vec::new(), functions: Vec::new() }
    }
}
