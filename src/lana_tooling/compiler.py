"""Small C-like Lana frontend that emits inspectable SSBC assembly."""

from __future__ import annotations

from dataclasses import dataclass
import re


class CompileError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    text: str
    line: int


TOKEN = re.compile(
    r"(?P<space>\s+)|(?P<comment>//[^\n]*|\#[^\n]*)|"
    r"(?P<string>\"(?:\\.|[^\"\\])*\")|(?P<number>\d+(?:\.\d+)?)|"
    r"(?P<identifier>[A-Za-z_][A-Za-z0-9_]*)|(?P<operator>->|==|!=|<=|>=|[{}()\[\];,:.=+\-*/<>!])"
)


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    offset = 0
    line = 1
    while offset < len(source):
        match = TOKEN.match(source, offset)
        if match is None:
            raise CompileError(f"line {line}: unexpected character {source[offset]!r}")
        text = match.group(0)
        if match.lastgroup not in {"space", "comment"}:
            tokens.append(Token(text, line))
        line += text.count("\n")
        offset = match.end()
    tokens.append(Token("<eof>", line))
    return tokens


class Parser:
    def __init__(self, source: str):
        self.tokens = tokenize(source)
        self.current = 0

    def peek(self, text: str | None = None) -> Token | bool:
        token = self.tokens[self.current]
        return token if text is None else token.text == text

    def take(self, text: str | None = None) -> Token:
        token = self.tokens[self.current]
        if text is not None and token.text != text:
            raise CompileError(f"line {token.line}: expected {text!r}, got {token.text!r}")
        self.current += 1
        return token

    def identifier(self) -> Token:
        token = self.peek()
        assert isinstance(token, Token)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token.text) or token.text in {
            "state", "let", "apply", "transform", "with", "compose", "using",
            "measure", "as", "print", "if", "else", "while", "fn", "return",
            "history", "latest", "duration", "previous", "change", "velocity",
            "true", "false", "null", "fork", "join", "join_all", "join_timeout",
            "cancel", "taskgroup",
        }:
            raise CompileError(f"line {token.line}: expected identifier, got {token.text!r}")
        return self.take()

    def program(self) -> tuple[list[dict], list[dict]]:
        functions: list[dict] = []
        statements: list[dict] = []
        while not self.peek("<eof>"):
            if self.peek("fn"):
                functions.append(self.function())
            else:
                statements.append(self.statement())
        return functions, statements

    def function(self) -> dict:
        line = self.take("fn").line
        name = self.identifier().text
        self.take("(")
        params: list[str] = []
        if not self.peek(")"):
            params.append(self.identifier().text)
            while self.peek(","):
                self.take(",")
                params.append(self.identifier().text)
        self.take(")")
        return {"kind": "function", "name": name, "params": params, "body": self.block(), "line": line}

    def block(self) -> list[dict]:
        self.take("{")
        body: list[dict] = []
        while not self.peek("}"):
            if self.peek("<eof>"):
                raise CompileError("unterminated block")
            body.append(self.statement())
        self.take("}")
        return body

    def statement(self) -> dict:
        token = self.peek()
        assert isinstance(token, Token)
        if self.peek("state"):
            self.take("state")
            name = self.identifier().text
            self.take("=")
            value = self.state_literal()
            self.take(";")
            return {"kind": "state", "name": name, "value": value, "line": token.line}
        if self.peek("history"):
            self.take("history")
            target = self.identifier().text
            policy = self.take().text
            if policy not in {"latest", "duration"}:
                raise CompileError(f"line {token.line}: history policy must be latest or duration")
            amount = self.expression()
            self.take(";")
            return {"kind": "history", "target": target, "policy": policy, "amount": amount, "line": token.line}
        if self.peek("let"):
            self.take("let")
            name = self.identifier().text
            self.take("=")
            value = self.expression()
            self.take(";")
            return {"kind": "let", "name": name, "value": value, "line": token.line}
        if self.peek("apply"):
            self.take("apply")
            sources: list[str] = []
            if self.peek("["):
                self.take("[")
                sources.append(self.identifier().text)
                while self.peek(","):
                    self.take(",")
                    sources.append(self.identifier().text)
                self.take("]")
            else:
                sources.append(self.identifier().text)
            self.take("->")
            target = self.identifier().text
            strategy = None
            if self.peek("using"):
                self.take("using")
                strategy = self.identifier().text
            self.take(";")
            if len(sources) > 1 and strategy is None:
                raise CompileError(f"line {token.line}: multi-state apply requires an aggregation strategy")
            return {"kind": "apply", "sources": sources, "target": target, "strategy": strategy, "line": token.line}
        if self.peek("transform"):
            self.take("transform")
            target = self.identifier().text
            self.take("with")
            name = self.identifier().text
            args = self.arguments()
            self.take(";")
            return {"kind": "transform", "target": target, "name": name, "args": args, "line": token.line}
        if self.peek("compose"):
            self.take("compose")
            left = self.identifier().text
            self.take(",")
            right = self.identifier().text
            self.take("using")
            mode = self.identifier().text
            self.take("->")
            output = self.identifier().text
            self.take(";")
            return {"kind": "compose", "left": left, "right": right, "mode": mode, "output": output, "line": token.line}
        if self.peek("print"):
            self.take("print")
            self.take("(")
            value = self.expression()
            self.take(")")
            self.take(";")
            return {"kind": "print", "value": value, "line": token.line}
        if self.peek("return"):
            self.take("return")
            value = self.expression()
            self.take(";")
            return {"kind": "return", "value": value, "line": token.line}
        if self.peek("if"):
            self.take("if")
            self.take("(")
            condition = self.expression()
            self.take(")")
            then = self.block()
            otherwise = []
            if self.peek("else"):
                self.take("else")
                otherwise = self.block()
            return {"kind": "if", "condition": condition, "then": then, "else": otherwise, "line": token.line}
        if self.peek("while"):
            self.take("while")
            self.take("(")
            condition = self.expression()
            self.take(")")
            return {"kind": "while", "condition": condition, "body": self.block(), "line": token.line}
        if self.peek("taskgroup"):
            self.take("taskgroup")
            return {"kind": "taskgroup", "body": self.block(), "line": token.line}
        if self.peek("cancel"):
            self.take("cancel")
            self.take("(")
            task = self.expression()
            self.take(")")
            self.take(";")
            return {"kind": "cancel", "task": task, "line": token.line}
        name = self.identifier().text
        self.take("=")
        value = self.expression()
        self.take(";")
        return {"kind": "assign", "name": name, "value": value, "line": token.line}

    def state_literal(self) -> dict:
        self.take("state")
        closing = ")" if self.peek("(") else "}"
        self.take("(" if closing == ")" else "{")
        fields: dict[str, dict] = {}
        while not self.peek(closing):
            name = self.identifier().text
            self.take(":")
            fields[name] = self.expression()
            if not self.peek(","):
                break
            self.take(",")
        self.take(closing)
        if "p" not in fields or "d" not in fields:
            raise CompileError("state requires p and d")
        if set(fields) - {"p", "d", "timestamp", "source", "weight", "confidence"}:
            raise CompileError("unknown state field")
        for field in ("p", "d", "timestamp", "weight", "confidence"):
            value = fields.get(field)
            if value is not None and value["kind"] == "literal" and value.get("literal_type") == "number":
                number = float(value["value"])
                if field == "p" and not 0.0 <= number <= 1.0:
                    raise CompileError("p must be between zero and one")
                if field == "d" and not -1.0 < number < 1.0:
                    raise CompileError("d must be strictly between negative one and one")
                if field == "weight" and number < 0.0:
                    raise CompileError("weight must be non-negative")
                if field == "confidence" and not 0.0 <= number <= 1.0:
                    raise CompileError("confidence must be between zero and one")
        return fields

    def arguments(self) -> list[dict]:
        self.take("(")
        args: list[dict] = []
        if not self.peek(")"):
            args.append(self.expression())
            while self.peek(","):
                self.take(",")
                args.append(self.expression())
        self.take(")")
        return args

    def expression(self) -> dict:
        return self.comparison()

    def comparison(self) -> dict:
        value = self.term()
        while any(self.peek(operator) for operator in ("==", "!=", "<", "<=", ">", ">=")):
            operator = self.take().text
            value = {"kind": "binary", "operator": operator, "left": value, "right": self.term()}
        return value

    def term(self) -> dict:
        value = self.factor()
        while self.peek("+") or self.peek("-"):
            operator = self.take().text
            value = {"kind": "binary", "operator": operator, "left": value, "right": self.factor()}
        return value

    def factor(self) -> dict:
        value = self.unary()
        while self.peek("*") or self.peek("/"):
            operator = self.take().text
            value = {"kind": "binary", "operator": operator, "left": value, "right": self.unary()}
        return value

    def unary(self) -> dict:
        if self.peek("-") or self.peek("!"):
            operator = self.take().text
            return {"kind": "unary", "operator": operator, "value": self.unary()}
        return self.primary()

    def primary(self) -> dict:
        token = self.take()
        if re.fullmatch(r"\d+(?:\.\d+)?", token.text):
            value: dict = {"kind": "literal", "literal_type": "number", "value": token.text}
        elif token.text in {"true", "false", "null"}:
            value = {"kind": "literal", "literal_type": token.text, "value": token.text}
        elif token.text.startswith('"'):
            decoded = bytes(token.text[1:-1], "utf-8").decode("unicode_escape")
            value = {"kind": "literal", "literal_type": "string", "value": decoded}
        elif token.text == "fork":
            function = self.identifier().text
            value = {"kind": "fork", "name": function, "args": self.arguments()}
        elif token.text in {"join", "join_all"}:
            self.take("(")
            task = self.expression()
            self.take(")")
            value = {"kind": token.text, "task": task}
        elif token.text == "join_timeout":
            self.take("(")
            task = self.expression()
            self.take(",")
            timeout = self.expression()
            self.take(")")
            value = {"kind": "join_timeout", "task": task, "timeout": timeout}
        elif token.text == "measure":
            source = self.identifier().text
            mode = "probability"
            if self.peek("as"):
                self.take("as")
                mode = self.identifier().text
            value = {"kind": "measure", "source": source, "mode": mode}
        elif token.text in {"previous", "change", "velocity"}:
            self.take("(")
            source = self.identifier().text
            self.take(")")
            value = {"kind": "history_value", "operation": token.text, "source": source}
        elif token.text == "[":
            items: list[dict] = []
            if not self.peek("]"):
                items.append(self.expression())
                while self.peek(","):
                    self.take(",")
                    items.append(self.expression())
            self.take("]")
            value = {"kind": "array", "items": items}
        elif token.text == "(":
            value = self.expression()
            self.take(")")
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token.text):
            value = {"kind": "call", "name": token.text, "args": self.arguments()} if self.peek("(") else {"kind": "name", "name": token.text}
        else:
            raise CompileError(f"line {token.line}: expected expression")
        while self.peek(".") or self.peek("["):
            if self.peek("."):
                self.take(".")
                value = {"kind": "field", "value": value, "field": self.identifier().text}
            else:
                self.take("[")
                index = self.expression()
                self.take("]")
                value = {"kind": "index", "value": value, "index": index}
        return value


class Emitter:
    def __init__(self, symbols: dict[str, int] | None = None, next_register: int = 0):
        self.symbols = dict(symbols or {})
        self.next_register = next_register
        self.lines: list[str] = []
        self.label_counter = 0

    def register(self) -> int:
        if self.next_register >= 256:
            raise CompileError("function requires more than 256 registers")
        result = self.next_register
        self.next_register += 1
        return result

    def label(self, prefix: str) -> str:
        self.label_counter += 1
        return f"__{prefix}_{self.label_counter}"

    def named(self, name: str) -> int:
        try:
            return self.symbols[name]
        except KeyError as exc:
            raise CompileError(f"unknown name {name!r}") from exc

    def contiguous(self, expressions: list[dict]) -> tuple[int, int]:
        if not expressions:
            return 0, 0
        values = [self.expression(expression) for expression in expressions]
        base = self.next_register
        for value in values:
            destination = self.register()
            self.lines.append(f"MOVE R{destination} R{value}")
        return base, len(values)

    def expression(self, node: dict) -> int:
        kind = node["kind"]
        if kind == "literal":
            destination = self.register()
            if node.get("literal_type") == "string":
                encoded = node["value"].encode("utf-8").hex()
                self.lines.append(f"LOAD_STRING R{destination} {encoded or '-'}")
            else:
                self.lines.append(f"LOAD_CONST R{destination} {node['value']}")
            return destination
        if kind == "name":
            return self.named(node["name"])
        if kind == "binary":
            left, right, destination = self.expression(node["left"]), self.expression(node["right"]), self.register()
            operation = "COMPARE" if node["operator"] in {"==", "!=", "<", "<=", ">", ">="} else "BINARY"
            self.lines.append(f"{operation} R{left} {node['operator']} R{right} R{destination}")
            return destination
        if kind == "unary":
            source, destination = self.expression(node["value"]), self.register()
            self.lines.append(f"UNARY {node['operator']} R{source} R{destination}")
            return destination
        if kind == "measure":
            destination = self.register()
            self.lines.append(f"MEASURE R{self.named(node['source'])} {node['mode']} R{destination}")
            return destination
        if kind == "field":
            source, destination = self.expression(node["value"]), self.register()
            operation = "GET_INDEX" if node["field"] in {"timestamp", "source", "weight", "confidence"} else "GET_FIELD"
            self.lines.append(f"{operation} R{source} {node['field']} R{destination}")
            return destination
        if kind == "history_value":
            destination = self.register()
            self.lines.append(f"{node['operation'].upper()} R{self.named(node['source'])} R{destination}")
            return destination
        if kind == "array":
            base, count = self.contiguous(node["items"])
            destination = self.register()
            self.lines.append(f"ARRAY_NEW R{destination} R{base} {count}")
            return destination
        if kind == "index":
            source, index, destination = self.expression(node["value"]), self.expression(node["index"]), self.register()
            self.lines.append(f"ARRAY_GET R{source} R{index} R{destination}")
            return destination
        if kind == "call":
            base, count = self.contiguous(node["args"])
            destination = self.register()
            host_calls = {"args", "read_text", "write_text", "now", "random", "assert"}
            operation = "HOST_CALL" if node["name"] in host_calls else "CALL"
            self.lines.append(f"{operation} {node['name']} R{base} {count} R{destination}")
            return destination
        if kind == "fork":
            base, count = self.contiguous(node["args"])
            destination = self.register()
            self.lines.append(f"FORK {node['name']} R{base} {count} R{destination}")
            return destination
        if kind in {"join", "join_all"}:
            task, destination = self.expression(node["task"]), self.register()
            self.lines.append(f"{kind.upper()} R{task} R{destination}")
            return destination
        if kind == "join_timeout":
            task = self.expression(node["task"])
            timeout = self.expression(node["timeout"])
            destination = self.register()
            self.lines.append(f"JOIN_TIMEOUT R{task} R{timeout} R{destination}")
            return destination
        raise CompileError(f"cannot compile expression {kind}")

    def statement(self, node: dict) -> None:
        kind = node["kind"]
        if kind == "state":
            destination = self.register()
            self.symbols[node["name"]] = destination
            p = self.expression(node["value"]["p"])
            d = self.expression(node["value"]["d"])
            self.lines.append(f"STATE_BUILD R{p} R{d} R{destination}")
            for index_name in ("timestamp", "source", "weight", "confidence"):
                if index_name in node["value"]:
                    value_register = self.expression(node["value"][index_name])
                    self.lines.append(f"SET_INDEX R{destination} {index_name} R{value_register}")
        elif kind == "history":
            amount = self.expression(node["amount"])
            self.lines.append(f"HISTORY R{self.named(node['target'])} {node['policy']} R{amount}")
        elif kind in {"let", "assign"}:
            source = self.expression(node["value"])
            if kind == "let":
                destination = self.register()
                self.symbols[node["name"]] = destination
            else:
                destination = self.named(node["name"])
            self.lines.append(f"MOVE R{destination} R{source}")
        elif kind == "apply":
            sources = [self.named(name) for name in node["sources"]]
            if len(sources) == 1:
                self.lines.append(f"APPLY R{sources[0]} R{self.named(node['target'])}")
            else:
                base = self.next_register
                for source in sources:
                    destination = self.register()
                    self.lines.append(f"MOVE R{destination} R{source}")
                self.lines.append(
                    f"APPLY_MANY R{base} {len(sources)} R{self.named(node['target'])} {node['strategy']}"
                )
        elif kind == "transform":
            base, count = self.contiguous(node["args"])
            registers = " ".join(f"R{base + index}" for index in range(count))
            self.lines.append(f"TRANSFORM R{self.named(node['target'])} {node['name']}" + (f" {registers}" if registers else ""))
        elif kind == "compose":
            if node["mode"] == "update":
                self.lines.append(f"COMPOSE_UPDATE R{self.named(node['left'])} R{self.named(node['right'])}")
                self.symbols[node["output"]] = self.named(node["left"])
            else:
                destination = self.register()
                self.symbols[node["output"]] = destination
                self.lines.append(f"COMPOSE_{node['mode'].upper()} R{self.named(node['left'])} R{self.named(node['right'])} R{destination}")
        elif kind == "print":
            self.lines.append(f"PRINT R{self.expression(node['value'])}")
        elif kind == "return":
            self.lines.append(f"RETURN R{self.expression(node['value'])}")
        elif kind == "if":
            otherwise, end = self.label("else"), self.label("endif")
            self.lines.append(f"JUMP_IF_FALSE R{self.expression(node['condition'])} {otherwise}")
            for statement in node["then"]:
                self.statement(statement)
            self.lines.extend([f"JUMP {end}", f"{otherwise}:"])
            for statement in node["else"]:
                self.statement(statement)
            self.lines.append(f"{end}:")
        elif kind == "while":
            start, end = self.label("while"), self.label("endwhile")
            self.lines.append(f"{start}:")
            self.lines.append(f"JUMP_IF_FALSE R{self.expression(node['condition'])} {end}")
            for statement in node["body"]:
                self.statement(statement)
            self.lines.extend([f"JUMP {start}", f"{end}:"])
        elif kind == "taskgroup":
            self.lines.append("TASKGROUP_ENTER")
            for statement in node["body"]:
                self.statement(statement)
            self.lines.append("TASKGROUP_EXIT")
        elif kind == "cancel":
            self.lines.append(f"CANCEL R{self.expression(node['task'])}")
        else:
            raise CompileError(f"cannot compile statement {kind}")


def compile_source(source: str) -> str:
    functions, statements = Parser(source).program()
    output = ["JUMP __main"] if functions else []
    for function in functions:
        parameters = {name: index for index, name in enumerate(function["params"])}
        emitter = Emitter(parameters, len(parameters))
        for statement in function["body"]:
            emitter.statement(statement)
        if not emitter.lines or not emitter.lines[-1].startswith("RETURN"):
            null = emitter.register()
            emitter.lines.extend([f"LOAD_CONST R{null} null", f"RETURN R{null}"])
        output.append(f".function {function['name']} {len(parameters)} {emitter.next_register}")
        output.extend(emitter.lines)
    if functions:
        output.append("__main:")
    main = Emitter()
    for statement in statements:
        main.statement(statement)
    output.extend(main.lines)
    output.append("HALT")
    return "\n".join(output) + "\n"
