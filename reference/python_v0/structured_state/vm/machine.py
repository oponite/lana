from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from structured_state.errors import LanaNameError, LanaTypeError, MeasurementError, VMError
from .bytecode import Op, Program
from .values import Distribution, Indexes, JointState, State, StateDelta


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class History:
    policy: tuple[str, float | int] | None = None
    versions: list[State] = field(default_factory=list)

    def append(self, state: State) -> None:
        if self.policy is None:
            return
        self.versions.append(state)
        kind, amount = self.policy
        if kind == "latest":
            self.versions[:] = self.versions[-int(amount):]
        elif kind == "duration" and state.indexes.timestamp is not None:
            cutoff = state.indexes.timestamp - float(amount)
            self.versions[:] = [item for item in self.versions if item.indexes.timestamp is None or item.indexes.timestamp >= cutoff]


class VM:
    """Fetch/decode/execute machine whose state operations are native VM semantics."""

    def __init__(self, program: Program, *, seed: int | None = None, trace: bool = False, output: Callable[[str], None] | None = None):
        self.program = program
        self.registers: dict[int, Any] = {}
        self.histories: dict[int, History] = {}
        self.ip = 0
        self.running = True
        self.random = random.Random(seed)
        self.trace = trace
        self.output = output or print
        self.call_stack: list[tuple[int, int]] = []

    def run(self) -> Any:
        while self.running and self.ip < len(self.program.instructions):
            instruction = self.program.instructions[self.ip]
            self.ip += 1
            self._execute(instruction.op, instruction.args)
        return self.registers.get(-1)

    def _trace(self, op: Op, args: tuple[Any, ...]) -> None:
        if self.trace:
            self.output(f"{self.ip - 1:04d} {op.value} " + ", ".join(map(str, args)))

    def _require_state(self, register: int) -> State:
        value = self.registers.get(register)
        if not isinstance(value, State):
            raise LanaTypeError(f"R{register} must contain State, got {type(value).__name__}")
        return value

    def _record(self, register: int, state: State) -> None:
        history = self.histories.get(register)
        if history:
            history.append(state)

    def _store_state(self, register: int, state: State) -> None:
        self.registers[register] = state
        self._record(register, state)

    def _apply(self, source: State, target: State) -> State:
        next_p1 = clip(target.p1 + source.d * (source.p1 - target.p1))
        return target.with_values(p1=next_p1)

    def _measure(self, source: State, mode: str) -> Any:
        if mode == "distribution":
            return Distribution(source.p0, source.p1)
        if mode in {"sample", "collapse"}:
            return 1 if self.random.random() < source.p1 else 0
        if mode == "expectation":
            return source.p1
        raise MeasurementError(f"unknown measurement mode '{mode}'")

    def _execute(self, op: Op, args: tuple[Any, ...]) -> None:
        self._trace(op, args)
        if op is Op.LOAD_CONST:
            dest, constant = args
            self.registers[dest] = self.program.constants[constant]
        elif op is Op.MOVE:
            dest, source = args
            self.registers[dest] = self.registers[source]
        elif op is Op.STATE_NEW:
            dest, p0_reg, p1_reg, d_reg, index_regs, history = args
            p0 = None if p0_reg is None else self.registers[p0_reg]
            p1 = None if p1_reg is None else self.registers[p1_reg]
            indexes = Indexes(**{name: self.registers[reg] for name, reg in index_regs.items()})
            self.histories[dest] = History(history)
            self._store_state(dest, State(p0=p0, p1=p1, d=self.registers[d_reg], indexes=indexes))
        elif op is Op.STATE_SET:
            dest, source = args
            self._store_state(dest, self._require_state(source))
        elif op is Op.APPLY:
            source, target = args
            self._store_state(target, self._apply(self._require_state(source), self._require_state(target)))
        elif op is Op.APPLY_IF:
            source, target, condition = args
            if self.registers[condition] is True:
                self._store_state(target, self._apply(self._require_state(source), self._require_state(target)))
        elif op is Op.APPLY_MANY:
            sources, target, strategy, tolerance = args
            values = [self._require_state(register) for register in sources]
            if not values:
                return
            chosen: list[State]
            if strategy == "sequential":
                result = self._require_state(target)
                for source in values:
                    result = self._apply(source, result)
                self._store_state(target, result)
                return
            if strategy == "strongest":
                chosen = [max(values, key=lambda item: abs(item.d) * (item.indexes.weight or 1) * (item.indexes.confidence if item.indexes.confidence is not None else 1))]
            else:
                if strategy == "consensus":
                    if tolerance is None:
                        raise VMError("consensus requires tolerance")
                    if max(item.p1 for item in values) - min(item.p1 for item in values) > tolerance:
                        return
                chosen = values
            weights = [1.0] * len(chosen) if strategy in {"mean", "consensus"} else [(item.indexes.weight or 1) * (item.indexes.confidence if item.indexes.confidence is not None else 1) for item in chosen]
            total = sum(weights)
            if total == 0:
                return
            aggregate = State(p1=sum(item.p1 * weight for item, weight in zip(chosen, weights)) / total,
                              d=sum(item.d * weight for item, weight in zip(chosen, weights)) / total)
            self._store_state(target, self._apply(aggregate, self._require_state(target)))
        elif op is Op.TRANSFORM:
            target, name, arg_registers = args
            from structured_state.stdlib import TRANSFORMS
            try:
                transform = TRANSFORMS[name]
            except KeyError as exc:
                raise VMError(f"unknown transform '{name}'") from exc
            state = self._require_state(target)
            result = transform(state, *[self.registers[reg] for reg in arg_registers])
            if not isinstance(result, State):
                raise LanaTypeError(f"transform '{name}' must return State")
            self._store_state(target, result)
        elif op is Op.COMPOSE_MERGE:
            left, right, dest = args
            a, b = self._require_state(left), self._require_state(right)
            self._store_state(dest, State(p1=(a.p1 + b.p1) / 2, d=(a.d + b.d) / 2))
        elif op is Op.COMPOSE_UPDATE:
            left, right = args
            a, b = self._require_state(left), self._require_state(right)
            self._store_state(left, self._apply(b, a))
            self._store_state(right, self._apply(a, b))
        elif op is Op.COMPOSE_JOINT:
            left, right, dest = args
            self.registers[dest] = JointState(self._require_state(left), self._require_state(right))
        elif op is Op.MEASURE:
            source, mode, dest = args
            state = self._require_state(source)
            result = self._measure(state, mode)
            self.registers[dest] = result
            if mode == "collapse":
                self._store_state(source, State(p1=float(result), d=0, indexes=state.indexes))
        elif op is Op.GET_FIELD:
            source, name, dest = args
            value = self.registers[source]
            if not hasattr(value, name):
                raise LanaTypeError(f"{type(value).__name__} has no field '{name}'")
            self.registers[dest] = getattr(value, name)
        elif op is Op.GET_INDEX:
            source, name, dest = args
            self.registers[dest] = self._require_state(source).indexes.get(name)
        elif op is Op.SET_INDEX:
            target, name, value = args
            state = self._require_state(target)
            self._store_state(target, state.with_values(indexes=state.indexes.set(name, self.registers[value])))
        elif op in {Op.PREVIOUS, Op.CHANGE, Op.VELOCITY}:
            source, dest = args
            versions = self.histories.get(source, History()).versions
            if len(versions) < 2:
                raise VMError("state has no previous retained version")
            current, previous = versions[-1], versions[-2]
            if op is Op.PREVIOUS:
                self.registers[dest] = previous
            elif op is Op.CHANGE:
                self.registers[dest] = current.p1 - previous.p1
            else:
                if current.indexes.timestamp is None or previous.indexes.timestamp is None:
                    raise VMError("velocity requires timestamps")
                elapsed = current.indexes.timestamp - previous.indexes.timestamp
                if elapsed <= 0:
                    raise VMError("velocity requires increasing timestamps")
                self.registers[dest] = (current.p1 - previous.p1) / elapsed
        elif op is Op.BINARY:
            operator, left, right, dest = args
            a, b = self.registers[left], self.registers[right]
            operations = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b,
                          "==": lambda: a == b, "!=": lambda: a != b, ">": lambda: a > b, ">=": lambda: a >= b,
                          "<": lambda: a < b, "<=": lambda: a <= b}
            self.registers[dest] = operations[operator]()
        elif op is Op.UNARY:
            operator, source, dest = args
            value = self.registers[source]
            self.registers[dest] = -value if operator == "-" else not value
        elif op is Op.JUMP:
            (self.ip,) = args
        elif op is Op.JUMP_IF_FALSE:
            condition, target = args
            if self.registers[condition] is not True:
                self.ip = target
        elif op is Op.PRINT:
            (source,) = args
            self.output(str(self.registers[source]))
        elif op is Op.CALL:
            name, argument_registers, destination = args
            try:
                function = self.program.functions[name]
            except KeyError as exc:
                raise LanaNameError(f"unknown function '{name}'") from exc
            parameters = function["parameters"]
            if len(parameters) != len(argument_registers):
                raise VMError(f"function '{name}' expects {len(parameters)} arguments")
            self.call_stack.append((self.ip, destination))
            for parameter, argument in zip(parameters, argument_registers): self.registers[parameter] = self.registers[argument]
            self.ip = function["address"]
        elif op is Op.RETURN:
            (source,) = args
            if self.call_stack:
                self.ip, destination = self.call_stack.pop(); self.registers[destination] = self.registers[source]
            else:
                self.registers[-1] = self.registers[source]; self.running = False
        elif op is Op.HALT:
            self.running = False
        else:
            raise VMError(f"unsupported opcode {op.value}")
