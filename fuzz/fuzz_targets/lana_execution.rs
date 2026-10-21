#![no_main]

use lana_bytecode::{loader, verifier, OpCode};
use lana_vm::vm::Vm;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(chunk) = loader::load(data) else {
        return;
    };
    if verifier::verify(&chunk).is_err() {
        return;
    }
    // Only execute opcodes without host effects, output, or task creation.
    if chunk.code.iter().any(|instruction| {
        !matches!(
            instruction.opcode,
            OpCode::Nop
                | OpCode::LoadConst
                | OpCode::Move
                | OpCode::Binary
                | OpCode::Unary
                | OpCode::Compare
                | OpCode::Jump
                | OpCode::JumpIfTrue
                | OpCode::JumpIfFalse
                | OpCode::ArrayNew
                | OpCode::ArrayGet
                | OpCode::ArraySet
                | OpCode::Call
                | OpCode::Return
                | OpCode::Halt
        )
    }) {
        return;
    }
    let mut vm = Vm::new(&chunk);
    vm.set_instruction_limit(10_000);
    if vm.set_memory_limit(64 * 1024 * 1024).is_ok() {
        let _ = vm.run();
    }
});
