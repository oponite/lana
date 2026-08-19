from __future__ import annotations

from structured_state import ast
from structured_state.errors import CompileError
from structured_state.parser import parse
from structured_state.vm.bytecode import Op, Program


class Compiler:
    def __init__(self) -> None:
        self.program = Program(); self.names: dict[str, int] = {}; self.next_register = 0
    def register(self, name: str | None = None) -> int:
        value = self.next_register; self.next_register += 1
        if name is not None: self.names[name] = value
        return value
    def compile(self, tree: ast.Program) -> Program:
        functions = [statement for statement in tree.statements if isinstance(statement, (ast.Function, ast.TransformDefinition))]
        self.user_transforms = {item.name for item in functions if isinstance(item, ast.TransformDefinition)}
        for statement in tree.statements:
            if not isinstance(statement, (ast.Function, ast.TransformDefinition)): self.statement(statement)
        self.program.emit(Op.HALT)
        for function in functions:
            if isinstance(function, ast.TransformDefinition):
                self.function(ast.Function(name=function.name, parameters=["input", *function.parameters], body=function.body))
            else: self.function(function)
        return self.program
    def function(self, node: ast.Function) -> None:
        if node.name in self.program.functions: raise CompileError(f"duplicate function '{node.name}'")
        saved_names = self.names.copy(); params = [self.register() for _ in node.parameters]
        self.names = dict(zip(node.parameters, params))
        self.program.functions[node.name] = {"address": len(self.program.instructions), "parameters": params}
        for statement in node.body: self.statement(statement)
        if not node.body or not isinstance(node.body[-1], ast.Return):
            null = self.register(); self.program.emit(Op.LOAD_CONST, null, self.program.add_constant(None)); self.program.emit(Op.RETURN, null)
        self.names = saved_names
    def named(self, node: ast.Node) -> int:
        if not isinstance(node, ast.Name): raise CompileError("operation target must be a named variable")
        try: return self.names[node.value]
        except KeyError as exc: raise CompileError(f"unknown name '{node.value}'") from exc
    def statement(self, node: ast.Node) -> None:
        if isinstance(node, ast.StateDeclaration):
            dest = self.register(node.name); literal = node.value; assert literal
            p0 = None if literal.p0 is None else self.expression(literal.p0); p1 = None if literal.p1 is None else self.expression(literal.p1); d = self.expression(literal.d)
            indexes = {name: self.expression(value) for name, value in literal.indexes.items()}
            history = None
            if node.history and node.history[0] != "none": history = (node.history[0], self.constant_value(node.history[1]))
            self.program.emit(Op.STATE_NEW, dest, p0, p1, d, indexes, history, line=node.line)
        elif isinstance(node, ast.Let):
            self.names[node.name] = self.expression(node.value)
        elif isinstance(node, ast.SetStatement):
            self.program.emit(Op.STATE_SET, self.named(ast.Name(value=node.name)), self.expression(node.value))
        elif isinstance(node, ast.Apply):
            sources = [self.named(item) for item in node.sources]; target = self.named(node.target)
            if len(sources) == 1:
                if node.condition is None: self.program.emit(Op.APPLY, sources[0], target)
                else: self.program.emit(Op.APPLY_IF, sources[0], target, self.expression(node.condition))
            else:
                if node.condition is not None: raise CompileError("conditional multi-state apply is not yet supported")
                tolerance = None if node.tolerance is None else self.constant_value(node.tolerance)
                self.program.emit(Op.APPLY_MANY, tuple(sources), target, node.strategy, tolerance)
        elif isinstance(node, ast.Transform):
            if len(node.targets) != 1: raise CompileError("multi-state user transforms are not yet supported")
            target = self.named(node.targets[0])
            arguments = tuple(self.expression(arg) for arg in node.args)
            if node.name in getattr(self, "user_transforms", set()):
                result = self.register(); self.program.emit(Op.CALL, node.name, (target, *arguments), result); self.program.emit(Op.STATE_SET, target, result)
            else: self.program.emit(Op.TRANSFORM, target, node.name, arguments)
        elif isinstance(node, ast.Compose):
            left, right = self.named(node.left), self.named(node.right)
            if node.mode == "merge":
                if len(node.outputs) != 1: raise CompileError("merge needs one output")
                self.program.emit(Op.COMPOSE_MERGE, left, right, self.register(node.outputs[0]))
            elif node.mode == "joint":
                if len(node.outputs) != 1: raise CompileError("joint needs one output")
                self.program.emit(Op.COMPOSE_JOINT, left, right, self.register(node.outputs[0]))
            elif node.mode == "update":
                if len(node.outputs) != 2: raise CompileError("update needs two outputs")
                self.program.emit(Op.COMPOSE_UPDATE, left, right)
                self.names[node.outputs[0]] = left; self.names[node.outputs[1]] = right
            else: raise CompileError(f"unknown composition mode '{node.mode}'")
        elif isinstance(node, ast.If):
            condition = self.expression(node.condition); jump = self.program.emit(Op.JUMP_IF_FALSE, condition, None)
            for statement in node.then_body: self.statement(statement)
            if node.else_body:
                end = self.program.emit(Op.JUMP, None); self.program.patch(jump, condition, len(self.program.instructions))
                for statement in node.else_body: self.statement(statement)
                self.program.patch(end, len(self.program.instructions))
            else: self.program.patch(jump, condition, len(self.program.instructions))
        elif isinstance(node, ast.While):
            start = len(self.program.instructions); condition = self.expression(node.condition); jump = self.program.emit(Op.JUMP_IF_FALSE, condition, None)
            for statement in node.body: self.statement(statement)
            self.program.emit(Op.JUMP, start); self.program.patch(jump, condition, len(self.program.instructions))
        elif isinstance(node, ast.Return): self.program.emit(Op.RETURN, self.expression(node.value))
        elif isinstance(node, ast.ExpressionStatement): self.program.emit(Op.PRINT, self.expression(node.value))
        else: raise CompileError(f"cannot compile {type(node).__name__}")
    def expression(self, node: ast.Node | None) -> int:
        if node is None: raise CompileError("missing expression")
        if isinstance(node, (ast.Number, ast.String, ast.Boolean)):
            register = self.register(); self.program.emit(Op.LOAD_CONST, register, self.program.add_constant(node.value)); return register
        if isinstance(node, ast.Name): return self.named(node)
        if isinstance(node, ast.StateLiteral):
            register = self.register(); p0 = None if node.p0 is None else self.expression(node.p0); p1 = None if node.p1 is None else self.expression(node.p1); d = self.expression(node.d)
            self.program.emit(Op.STATE_NEW, register, p0, p1, d, {key: self.expression(value) for key, value in node.indexes.items()}, None); return register
        if isinstance(node, ast.Measure):
            register = self.register(); self.program.emit(Op.MEASURE, self.named(node.source), node.mode, register); return register
        if isinstance(node, ast.Field):
            register = self.register(); self.program.emit(Op.GET_FIELD, self.expression(node.object), node.name, register); return register
        if isinstance(node, ast.Binary):
            register = self.register(); self.program.emit(Op.BINARY, node.operator, self.expression(node.left), self.expression(node.right), register); return register
        if isinstance(node, ast.Unary):
            register = self.register(); self.program.emit(Op.UNARY, node.operator, self.expression(node.operand), register); return register
        if isinstance(node, ast.Call):
            if node.callee in {"previous", "change", "velocity"}:
                if len(node.args) != 1: raise CompileError(f"{node.callee} needs one state")
                register = self.register(); self.program.emit({"previous": Op.PREVIOUS, "change": Op.CHANGE, "velocity": Op.VELOCITY}[node.callee], self.named(node.args[0]), register); return register
            if node.callee not in self.program.functions:
                # Functions are registered after main compilation; allow a forward
                # reference and let the VM resolve it from the final function table.
                pass
            register = self.register(); self.program.emit(Op.CALL, node.callee, tuple(self.expression(arg) for arg in node.args), register); return register
        raise CompileError(f"cannot compile expression {type(node).__name__}")
    def constant_value(self, node: ast.Node | None):
        if isinstance(node, ast.Number): return node.value
        raise CompileError("history and consensus parameters must be numeric literals")


def compile_source(source: str) -> Program:
    return Compiler().compile(parse(source))
