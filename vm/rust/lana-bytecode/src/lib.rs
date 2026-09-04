//! Lana bytecode: the LABC v2 binary format, loader, verifier, and
//! disassembler, mirroring the C11 implementation in `vm/c/bytecode.c`.
//!
//! This crate is the first stage of the Rust runtime boundary: it must accept
//! exactly the chunks the C11 loader accepts and reject exactly the chunks it
//! rejects, so that differential conformance can compare the two byte-for-byte.

pub mod assembler;
pub mod chunk;
pub mod disassembler;
pub mod error;
pub mod loader;
pub mod opcode;
pub mod value;
pub mod verifier;

pub use assembler::assemble;
pub use chunk::{Chunk, Function, Instruction};
pub use disassembler::format_g;
pub use error::{LanaError, LanaErrorInfo};
pub use opcode::OpCode;
pub use value::{Value, ValueType};

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid chunk: one constant, one function, and a
    /// LOAD_CONST + HALT program.
    fn sample_chunk() -> Chunk {
        let mut chunk = Chunk::new(opcode::LABC_VERSION, 0);
        chunk.constants.push(Value::Number(42.0));
        chunk.functions.push(Function {
            name: "main".to_string(),
            entry: 0,
            register_count: 1,
            arity: 0,
        });
        chunk.code.push(Instruction::new(OpCode::LoadConst, 0, 0, 0, 0, 1));
        chunk.code.push(Instruction::new(OpCode::Halt, 0, 0, 0, 0, 2));
        chunk
    }

    #[test]
    fn verify_accepts_valid_chunk() {
        let chunk = sample_chunk();
        assert!(verifier::verify(&chunk).is_ok());
    }

    #[test]
    fn verify_rejects_bad_entry() {
        let mut chunk = sample_chunk();
        chunk.entry = 99;
        let error = verifier::verify(&chunk).unwrap_err();
        assert_eq!(error.code, LanaError::Format);
    }

    #[test]
    fn verify_rejects_register_out_of_range() {
        let mut chunk = sample_chunk();
        chunk.code[0] = Instruction::new(OpCode::LoadConst, 300, 0, 0, 0, 1);
        let error = verifier::verify(&chunk).unwrap_err();
        assert_eq!(error.code, LanaError::Register);
    }

    #[test]
    fn verify_rejects_constant_out_of_range() {
        let mut chunk = sample_chunk();
        chunk.code[0] = Instruction::new(OpCode::LoadConst, 0, 0, 0, 7, 1);
        let error = verifier::verify(&chunk).unwrap_err();
        assert_eq!(error.code, LanaError::Constant);
    }

    #[test]
    fn verify_rejects_unknown_opcode() {
        let mut chunk = sample_chunk();
        chunk.code[0] = Instruction::new(OpCode::Count, 0, 0, 0, 0, 1);
        let error = verifier::verify(&chunk).unwrap_err();
        assert_eq!(error.code, LanaError::Opcode);
    }

    #[test]
    fn verify_rejects_bad_jump_target() {
        let mut chunk = sample_chunk();
        chunk.code[0] = Instruction::new(OpCode::Jump, 0, 0, 0, 99, 1);
        let error = verifier::verify(&chunk).unwrap_err();
        assert_eq!(error.code, LanaError::Jump);
    }

    #[test]
    fn verify_rejects_invalid_state_new() {
        let mut chunk = sample_chunk();
        chunk.constants.push(Value::Number(1.5)); // p > 1
        chunk.code[0] = Instruction::new(OpCode::StateNew, 0, 1, 0, 0, 1);
        let error = verifier::verify(&chunk).unwrap_err();
        assert_eq!(error.code, LanaError::InvalidState);
    }

    #[test]
    fn disassemble_matches_c11_format() {
        let chunk = sample_chunk();
        let text = disassembler::disassemble(&chunk);
        assert!(text.starts_with("LABC v2 entry=0 constants=1 functions=1 instructions=2\n"));
        assert!(text.contains("LOAD_CONST"));
        assert!(text.contains("; line 1"));
    }

    /// Serialize a LABC header (magic + version + counts + entry) to bytes.
    fn header_bytes(constants: u32, functions: u32, instructions: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"LABC");
        bytes.extend_from_slice(&opcode::LABC_VERSION.to_le_bytes());
        bytes.extend_from_slice(&constants.to_le_bytes());
        bytes.extend_from_slice(&functions.to_le_bytes());
        bytes.extend_from_slice(&instructions.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // entry
        bytes
    }

    #[test]
    fn loader_rejects_trailing_bytes() {
        // A minimal valid chunk: no constants, functions, or instructions.
        let valid = header_bytes(0, 0, 0);
        assert!(loader::load(&valid).is_ok());

        // The C11 loader rejects any byte after the last instruction.
        let mut trailing = valid.clone();
        trailing.push(0x00);
        let error = loader::load(&trailing).unwrap_err();
        assert_eq!(error.code, LanaError::Format);
    }

    #[test]
    fn loader_rejects_long_function_name() {
        // C11 caps function names at 1,000,000 bytes (string constants are
        // capped separately at 10,000,000). The length field alone triggers the
        // rejection before any name bytes are read.
        let mut bytes = header_bytes(0, 1, 0);
        bytes.extend_from_slice(&1_000_001u32.to_le_bytes());
        let error = loader::load(&bytes).unwrap_err();
        assert_eq!(error.code, LanaError::Format);
    }
}
