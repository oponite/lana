"""Auxiliary Python code generation for Lana's current file-I/O-free frontend."""

from __future__ import annotations


def number(value: float) -> str:
    return format(float(value), ".17g")


def dependency(event: dict, mode: str) -> float:
    return event["d"] if mode == "dynamic" else 0.28 if mode == "fixed" else 0.0


def set_index(lines: list[str], state: int, name: str, value: float | str, temporary: int) -> None:
    lines.append(f"LOAD_CONST R{temporary} {value}")
    lines.append(f"SET_INDEX R{state} {name} R{temporary}")


def new_evidence(lines: list[str], register: int, event: dict, mode: str, temporary: int) -> None:
    lines.append(f"STATE_NEW R{register} {number(event['signal_p'])} {number(dependency(event, mode))}")
    set_index(lines, register, "timestamp", number(event["t"]), temporary)
    set_index(lines, register, "source", event["source"], temporary)
    set_index(lines, register, "weight", number(event["weight"]), temporary)
    set_index(lines, register, "confidence", number(event["confidence"]), temporary)
    decay = min(event["age"] / 120.0, 0.8)
    if decay > 0.0:
        lines.append(f"LOAD_CONST R{temporary} {number(decay)}")
        lines.append(f"TRANSFORM R{register} decay R{temporary}")


def generate(
    steps: list[list[dict]], layer: str, mode: str = "dynamic", emit_predictions: bool = True
) -> str:
    lines: list[str] = []
    if layer == "A":
        lines.append("STATE_NEW R0 0.5 0.2")
        for events in steps:
            new_evidence(lines, 1, events[0], mode, 30)
            lines.append("APPLY R1 R0")
            if emit_predictions:
                lines.extend(("MEASURE R0 probability R20", "PRINT R20"))
        if not emit_predictions:
            lines.extend(("MEASURE R0 probability R20", "PRINT R20"))
        lines.append("HALT")
        return "\n".join(lines) + "\n"
    if layer == "B":
        lines.extend(("STATE_NEW R0 0.5 0.2", "LOAD_CONST R10 64", "HISTORY R0 latest R10"))
        for events in steps:
            set_index(lines, 0, "timestamp", number(events[0]["t"]), 30)
            lines.extend(("LOAD_CONST R31 0.015", "TRANSFORM R0 decay R31"))
            for register, event in enumerate(events, 1):
                new_evidence(lines, register, event, mode, 30)
            lines.extend(("APPLY_MANY R1 3 R0 weighted", "CHANGE R0 R22"))
            if emit_predictions:
                lines.extend(("MEASURE R0 probability R20", "PRINT R20"))
        if not emit_predictions:
            lines.extend(("MEASURE R0 probability R20", "PRINT R20"))
        lines.append("HALT")
        return "\n".join(lines) + "\n"
    node_d = (0.35, 0.25, 0.30, 0.15) if mode == "dynamic" else ((0.28,) * 4 if mode == "fixed" else (0.0,) * 4)
    for register, value in enumerate(node_d):
        lines.append(f"STATE_NEW R{register} 0.5 {number(value)}")
    lines.extend(("LOAD_CONST R24 64", "HISTORY R3 latest R24"))
    for index, events in enumerate(steps):
        set_index(lines, 3, "timestamp", number(events[0]["t"]), 30)
        lines.append("LOAD_CONST R20 0.01")
        for register in range(4):
            lines.append(f"TRANSFORM R{register} decay R20")
        new_evidence(lines, 5, events[0], mode, 30)
        new_evidence(lines, 6, events[1], mode, 30)
        new_evidence(lines, 4, events[2], mode, 30)
        lines.extend(("APPLY R5 R0", "APPLY R6 R1"))
        set_index(lines, 0, "weight", number(events[0]["weight"]), 30)
        set_index(lines, 0, "confidence", number(events[0]["confidence"]), 30)
        set_index(lines, 1, "weight", number(events[1]["weight"]), 30)
        set_index(lines, 1, "confidence", number(events[1]["confidence"]), 30)
        lines.extend(("APPLY_MANY R0 2 R2 weighted", "APPLY R2 R3", "GET_FIELD R2 p R21", "LOAD_CONST R22 0.55", "COMPARE R21 > R22 R23"))
        skip = f"__skip_conditional_{index}"
        lines.extend((f"JUMP_IF_FALSE R23 {skip}", "APPLY R4 R3", f"{skip}:", "CHANGE R3 R26"))
        if emit_predictions:
            lines.extend(("MEASURE R3 probability R25", "PRINT R25"))
    if not emit_predictions:
        lines.extend(("MEASURE R3 probability R25", "PRINT R25"))
    lines.append("HALT")
    return "\n".join(lines) + "\n"


def stress_program(transitions: int, source_p: float, source_d: float, target_p: float) -> str:
    return f"""STATE_NEW R0 {number(target_p)} 0.2
STATE_NEW R1 {number(source_p)} {number(source_d)}
LOAD_CONST R2 0
LOAD_CONST R3 {transitions}
LOAD_CONST R4 1
__loop:
COMPARE R2 < R3 R5
JUMP_IF_FALSE R5 __done
APPLY R1 R0
BINARY R2 + R4 R2
JUMP __loop
__done:
MEASURE R0 probability R6
PRINT R6
HALT
"""
