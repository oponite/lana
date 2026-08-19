from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Op(str, Enum):
    LOAD_CONST = "LOAD_CONST"
    MOVE = "MOVE"
    STATE_NEW = "STATE_NEW"
    STATE_SET = "STATE_SET"
    APPLY = "APPLY"
    APPLY_IF = "APPLY_IF"
    APPLY_MANY = "APPLY_MANY"
    TRANSFORM = "TRANSFORM"
    COMPOSE_MERGE = "COMPOSE_MERGE"
    COMPOSE_UPDATE = "COMPOSE_UPDATE"
    COMPOSE_JOINT = "COMPOSE_JOINT"
    MEASURE = "MEASURE"
    GET_FIELD = "GET_FIELD"
    GET_INDEX = "GET_INDEX"
    SET_INDEX = "SET_INDEX"
    PREVIOUS = "PREVIOUS"
    CHANGE = "CHANGE"
    VELOCITY = "VELOCITY"
    BINARY = "BINARY"
    UNARY = "UNARY"
    JUMP = "JUMP"
    JUMP_IF_FALSE = "JUMP_IF_FALSE"
    CALL = "CALL"
    RETURN = "RETURN"
    PRINT = "PRINT"
    HALT = "HALT"


@dataclass(frozen=True)
class Instruction:
    op: Op
    args: tuple[Any, ...] = ()
    line: int = 0


@dataclass
class Program:
    instructions: list[Instruction] = field(default_factory=list)
    constants: list[Any] = field(default_factory=list)
    functions: dict[str, int] = field(default_factory=dict)
    version: int = 1

    def add_constant(self, value: Any) -> int:
        self.constants.append(value)
        return len(self.constants) - 1

    def emit(self, op: Op, *args: Any, line: int = 0) -> int:
        self.instructions.append(Instruction(op, args, line))
        return len(self.instructions) - 1

    def patch(self, offset: int, *args: Any) -> None:
        old = self.instructions[offset]
        self.instructions[offset] = Instruction(old.op, args, old.line)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "constants": self.constants, "functions": self.functions,
                "instructions": [{"op": i.op.value, "args": list(i.args), "line": i.line} for i in self.instructions]}
