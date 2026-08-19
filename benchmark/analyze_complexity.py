"""Transparent static metrics for the hand-authored benchmark implementations."""

from __future__ import annotations

import ast
import io
from pathlib import Path
import re
import tokenize


STATE_NAMES = {"p", "d", "influence", "timestamp", "source", "weight", "confidence", "history"}


class PythonVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.branches = 0
        self.loops = 0
        self.cyclomatic = 1
        self.depth = 0
        self.max_depth = 0
        self.state_operations = 0
        self.state_variables: set[str] = set()
        self.metadata_variables: set[str] = set()
        self.history_containers = 0
        self.update_functions = 0

    def nested(self, nodes: list[ast.stmt]) -> None:
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        for node in nodes:
            self.visit(node)
        self.depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self.branches += 1
        self.cyclomatic += 1
        self.visit(node.test)
        self.nested(node.body)
        self.nested(node.orelse)

    def visit_For(self, node: ast.For) -> None:
        self.loops += 1
        self.cyclomatic += 1
        self.visit(node.iter)
        self.nested(node.body)
        self.nested(node.orelse)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.loops += 1
        self.cyclomatic += 1
        self.visit(node.test)
        self.nested(node.body)
        self.nested(node.orelse)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.cyclomatic += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if any(word in node.name for word in ("apply", "decay", "aggregate", "evidence", "predict", "update")):
            self.update_functions += 1
        self.nested(node.body)

    def visit_Call(self, node: ast.Call) -> None:
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
        if name in {"State", "_apply", "_decay", "_aggregate", "apply_to", "decay", "aggregate", "evidence", "append"}:
            self.state_operations += 1
        if name == "deque":
            self.history_containers += 1
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Constant) and node.slice.value in STATE_NAMES:
            self.state_operations += 1
            if node.slice.value in {"p", "d", "influence"}:
                self.state_variables.add(str(node.slice.value))
            else:
                self.metadata_variables.add(str(node.slice.value))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in STATE_NAMES:
            self.state_operations += 1
            if node.attr in {"p", "d", "influence"}:
                self.state_variables.add(node.attr)
            else:
                self.metadata_variables.add(node.attr)
        self.generic_visit(node)


def source_lines(path: Path) -> list[str]:
    lines = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "//")):
            lines.append(line)
    return lines


def python_tokens(text: str) -> int:
    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT,
               tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT}
    return sum(1 for token in tokenize.tokenize(io.BytesIO(text.encode()).readline) if token.type not in ignored)


def python_metrics(path: Path) -> dict[str, int]:
    text = path.read_text()
    visitor = PythonVisitor()
    visitor.visit(ast.parse(text))
    return {
        "loc": len(source_lines(path)), "tokens": python_tokens(text), "files": 1,
        "branches": visitor.branches, "cyclomatic": visitor.cyclomatic,
        "max_nesting": visitor.max_depth, "loops": visitor.loops,
        "state_management_operations": visitor.state_operations,
        "explicit_state_variables": len(visitor.state_variables),
        "explicit_metadata_variables": len(visitor.metadata_variables),
        "history_containers": visitor.history_containers,
        "update_functions": visitor.update_functions,
    }


def lana_metrics(path: Path) -> dict[str, int]:
    lines = source_lines(path)
    text = "\n".join(re.sub(r"//.*", "", line) for line in lines)
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|->|==|!=|<=|>=|[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]', text)
    branches = len(re.findall(r"\bif\s*\(", text))
    loops = len(re.findall(r"\bwhile\s*\(", text))
    state_ops = len(re.findall(r"\b(?:state|apply|transform|compose|history|measure|previous|change|velocity)\b", text))
    max_depth = 0
    depth = 0
    for line in lines:
        if re.match(r"\s*(?:if|while|fn)\b", line):
            depth += 1
            max_depth = max(max_depth, depth)
        depth = max(0, depth - line.count("}"))
    return {
        "loc": len(lines), "tokens": len(tokens), "files": 1,
        "branches": branches, "cyclomatic": 1 + branches + loops,
        "max_nesting": max_depth, "loops": loops,
        "state_management_operations": state_ops,
        "explicit_state_variables": 2,
        "explicit_metadata_variables": 0,
        "history_containers": len(re.findall(r"\bhistory\b", text)),
        "update_functions": 0,
    }


def collect(root: Path) -> dict[str, dict[str, int]]:
    return {
        "Python Plain": python_metrics(root / "python" / "plain.py"),
        "Python State Class": python_metrics(root / "python" / "state_class.py"),
        "Lana": lana_metrics(root / "lana" / "policies.lana"),
    }
