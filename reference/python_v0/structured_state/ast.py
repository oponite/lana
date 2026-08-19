from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node: line: int = 0
@dataclass
class Program(Node): statements: list[Node] = field(default_factory=list)
@dataclass
class Number(Node): value: float = 0
@dataclass
class String(Node): value: str = ""
@dataclass
class Boolean(Node): value: bool = False
@dataclass
class Name(Node): value: str = ""
@dataclass
class StateLiteral(Node): p0: Node | None = None; p1: Node | None = None; d: Node | None = None; indexes: dict[str, Node] = field(default_factory=dict)
@dataclass
class Measure(Node): source: Node | None = None; mode: str = "distribution"
@dataclass
class Call(Node): callee: str = ""; args: list[Node] = field(default_factory=list)
@dataclass
class Field(Node): object: Node | None = None; name: str = ""
@dataclass
class Binary(Node): left: Node | None = None; operator: str = ""; right: Node | None = None
@dataclass
class Unary(Node): operator: str = ""; operand: Node | None = None
@dataclass
class StateDeclaration(Node): name: str = ""; value: StateLiteral | None = None; history: tuple[str, Node | None] | None = None
@dataclass
class Let(Node): name: str = ""; value: Node | None = None
@dataclass
class SetStatement(Node): name: str = ""; value: Node | None = None
@dataclass
class Apply(Node): sources: list[Node] = field(default_factory=list); target: Node | None = None; condition: Node | None = None; strategy: str | None = None; tolerance: Node | None = None
@dataclass
class Transform(Node): targets: list[Node] = field(default_factory=list); name: str = ""; args: list[Node] = field(default_factory=list)
@dataclass
class TransformDefinition(Node): name: str = ""; parameters: list[str] = field(default_factory=list); body: list[Node] = field(default_factory=list)
@dataclass
class Compose(Node): left: Node | None = None; right: Node | None = None; mode: str = ""; outputs: list[str] = field(default_factory=list)
@dataclass
class If(Node): condition: Node | None = None; then_body: list[Node] = field(default_factory=list); else_body: list[Node] = field(default_factory=list)
@dataclass
class While(Node): condition: Node | None = None; body: list[Node] = field(default_factory=list)
@dataclass
class Function(Node): name: str = ""; parameters: list[str] = field(default_factory=list); body: list[Node] = field(default_factory=list)
@dataclass
class Return(Node): value: Node | None = None
@dataclass
class ExpressionStatement(Node): value: Node | None = None
