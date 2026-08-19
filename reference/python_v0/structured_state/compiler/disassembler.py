from structured_state.vm.bytecode import Program


def disassemble(program: Program) -> str:
    lines = [f"Lana bytecode v{program.version}"]
    for offset, instruction in enumerate(program.instructions):
        args = ", ".join(repr(arg) for arg in instruction.args)
        lines.append(f"{offset:04d} {instruction.op.value:<16} {args}")
    return "\n".join(lines)
