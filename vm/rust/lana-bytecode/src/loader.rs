//! LABC binary loader, mirroring `lana_chunk_read_file` in `vm/c/bytecode.c`.
//!
//! The on-disk layout is little-endian:
//!
//! ```text
//! "LABC" | version u32 | constant_count u32 | function_count u32
//!       | instruction_count u32 | entry u32
//! constants: type u8, then per type:
//!   Number: u64 (IEEE-754 bits)
//!   Bool:   u8 (0 or 1)
//!   String: u32 length + bytes
//!   Null:   (nothing)
//! functions: u32 name length + bytes | entry u32 | register_count u32 | arity u32
//! instructions: opcode u8 | a u32 | b u32 | c u32 | imm u32 | line u32
//! ```

use crate::chunk::{Chunk, Function, Instruction};
use crate::error::{LanaError, LanaErrorInfo};
use crate::opcode::{OpCode, LABC_VERSION, LABC_VERSION_1};
use crate::value::{Value, ValueType};

/// Maximum serialized chunk size, matching the C11 loader's 64 MiB cap.
pub const MAX_CHUNK_SIZE: usize = 64 * 1024 * 1024;
const MAX_CONSTANTS: u32 = 100_000;
const MAX_FUNCTIONS: u32 = 10_000;
const MAX_INSTRUCTIONS: u32 = 1_000_000;
const MAX_STRING_LENGTH: u32 = 10_000_000;
const MAX_FUNCTION_NAME_LENGTH: u32 = 1_000_000;

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_u8(&mut self) -> Option<u8> {
        let value = *self.bytes.get(self.offset)?;
        self.offset += 1;
        Some(value)
    }

    fn read_u32(&mut self) -> Option<u32> {
        let slice = self.bytes.get(self.offset..self.offset + 4)?;
        self.offset += 4;
        Some(u32::from_le_bytes(slice.try_into().ok()?))
    }

    fn read_u64(&mut self) -> Option<u64> {
        let slice = self.bytes.get(self.offset..self.offset + 8)?;
        self.offset += 8;
        Some(u64::from_le_bytes(slice.try_into().ok()?))
    }

    fn read_bytes(&mut self, length: usize) -> Option<&'a [u8]> {
        let slice = self.bytes.get(self.offset..self.offset + length)?;
        self.offset += length;
        Some(slice)
    }
}

/// Load a LABC chunk from a byte slice.
///
/// Returns `LanaErrorInfo` on failure with the same error code the C11 loader
/// would produce for the same malformed input.
pub fn load(bytes: &[u8]) -> Result<Chunk, LanaErrorInfo> {
    let mut reader = Reader::new(bytes);
    if bytes.len() > MAX_CHUNK_SIZE {
        return Err(LanaErrorInfo::new(LanaError::Limit, 0, 0, 0, "chunk exceeds size limit"));
    }
    let magic = reader.read_bytes(4).ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
    if magic != b"LABC" {
        return Err(LanaErrorInfo::new(
            LanaError::IncompatibleFormat, 0, 0, 0, "invalid LABC magic"));
    }
    let version = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
    let constants = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
    let functions = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
    let instructions = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
    let entry = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;

    if version != LABC_VERSION && version != LABC_VERSION_1 {
        return Err(LanaErrorInfo::new(
            LanaError::IncompatibleFormat, 0, 0, 0,
            format!("unsupported LABC version {version}")));
    }
    if constants > MAX_CONSTANTS || functions > MAX_FUNCTIONS || instructions > MAX_INSTRUCTIONS {
        return Err(LanaErrorInfo::new(LanaError::Limit, 0, 0, 0, "chunk exceeds count limits"));
    }
    if constants as usize > bytes.len()
        || functions as usize > bytes.len() / 16
        || instructions as usize > bytes.len() / 21
    {
        return Err(LanaErrorInfo::new(LanaError::Format, 0, 0, 0, "chunk counts exceed file size"));
    }

    let mut chunk = Chunk::new(version, entry);
    for _ in 0..constants {
        let type_byte = reader.read_u8().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        let value_type = ValueType::try_from(type_byte)
            .map_err(|_| format_error(LanaError::Format, 0, 0, 0))?;
        let value = match value_type {
            ValueType::Number => {
                let bits = reader.read_u64().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
                Value::Number(f64::from_bits(bits))
            }
            ValueType::Bool => {
                let boolean = reader.read_u8().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
                if boolean > 1 {
                    return Err(LanaErrorInfo::new(
                        LanaError::Format, 0, 0, 0, "invalid boolean constant"));
                }
                Value::Bool(boolean != 0)
            }
            ValueType::String => {
                let length = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
                if length > MAX_STRING_LENGTH {
                    return Err(LanaErrorInfo::new(
                        LanaError::Format, 0, 0, 0, "string constant exceeds length limit"));
                }
                let bytes = reader
                    .read_bytes(length as usize)
                    .ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
                Value::String(String::from_utf8_lossy(bytes).into_owned())
            }
            ValueType::Null => Value::Null,
            _ => {
                return Err(LanaErrorInfo::new(
                    LanaError::Format, 0, 0, 0, "constant type is not serializable"));
            }
        };
        chunk.constants.push(value);
    }

    for _ in 0..functions {
        let name_length = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        if name_length > MAX_FUNCTION_NAME_LENGTH {
            return Err(LanaErrorInfo::new(
                LanaError::Format, 0, 0, 0, "function name exceeds length limit"));
        }
        let name_bytes = reader
            .read_bytes(name_length as usize)
            .ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        let function_entry = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        let register_count = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        let arity = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        chunk.functions.push(Function { name, entry: function_entry, register_count, arity });
    }

    for _ in 0..instructions {
        let opcode_byte = reader.read_u8().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        let opcode = OpCode::try_from(opcode_byte)
            .map_err(|_| format_error(LanaError::Format, 0, 0, 0))?;
        let a = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        let b = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        let c = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        let imm = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        let line = reader.read_u32().ok_or_else(|| format_error(LanaError::Format, 0, 0, 0))?;
        chunk.code.push(Instruction::new(opcode, a, b, c, imm, line));
    }

    // The C11 loader rejects any trailing bytes after the last instruction
    // (`fgetc(file) != EOF`), so a well-formed LABC file must be fully consumed.
    if reader.offset != bytes.len() {
        return Err(LanaErrorInfo::new(
            LanaError::Format, 0, 0, 0, "trailing bytes after LABC chunk"));
    }

    Ok(chunk)
}

fn format_error(code: LanaError, ip: usize, opcode: u8, line: u32) -> LanaErrorInfo {
    LanaErrorInfo::new(code, ip, opcode, line, "truncated LABC file")
}
