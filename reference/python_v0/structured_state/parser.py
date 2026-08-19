from __future__ import annotations

from structured_state import ast
from structured_state.errors import ParseError
from structured_state.lexer import Token, lex


class Parser:
    def __init__(self, source: str): self.tokens = lex(source); self.current = 0
    def parse(self) -> ast.Program:
        statements = []
        while not self._check("EOF"): statements.append(self._statement())
        return ast.Program(statements=statements)
    def _statement(self) -> ast.Node:
        if self._match("STATE"): return self._state()
        if self._match("LET"): return self._let()
        if self._match("SET"): return self._set()
        if self._match("APPLY"): return self._apply()
        if self._match("TRANSFORM"): return self._transform_or_definition()
        if self._match("COMPOSE"): return self._compose()
        if self._match("IF"): return self._if()
        if self._match("WHILE"): return self._while()
        if self._match("FUNCTION"): return self._function()
        if self._match("RETURN"): return ast.Return(value=self._expression())
        return ast.ExpressionStatement(value=self._expression())
    def _state(self) -> ast.StateDeclaration:
        name = self._consume("IDENTIFIER", "expected state name"); self._consume("ASSIGN", "expected '='")
        literal = self._state_literal(); history = None
        if self._match("HISTORY"):
            self._consume("COLON", "expected ':' after history")
            if self._match("IDENTIFIER") and self._previous().value == "none": history = ("none", None)
            else:
                if self._match("LATEST", "DURATION"):
                    kind = self._previous().value
                else:
                    raise self._error(self._peek(), "expected latest, duration, or none")
                self._consume("LPAREN", "expected '('"); amount = self._expression(); self._consume("RPAREN", "expected ')'"); history = (str(kind), amount)
        return ast.StateDeclaration(name=str(name.value), value=literal, history=history)
    def _state_literal(self) -> ast.StateLiteral:
        self._consume("STATE", "expected state(...)"); self._consume("LPAREN", "expected '('")
        fields = {}
        while not self._check("RPAREN"):
            key = str(self._consume("IDENTIFIER", "expected state field").value); self._consume("COLON", "expected ':'"); fields[key] = self._expression()
            if not self._match("COMMA"): break
        self._consume("RPAREN", "expected ')'")
        unknown = set(fields) - {"p0", "p1", "d", "timestamp", "source", "weight", "confidence"}
        if unknown: raise self._error(self._previous(), f"unknown state field {unknown.pop()}")
        if "d" not in fields: raise self._error(self._previous(), "state requires d")
        return ast.StateLiteral(p0=fields.get("p0"), p1=fields.get("p1"), d=fields["d"], indexes={key: fields[key] for key in fields if key in {"timestamp", "source", "weight", "confidence"}})
    def _let(self) -> ast.Let:
        name = self._consume("IDENTIFIER", "expected variable name"); self._consume("ASSIGN", "expected '='"); return ast.Let(name=str(name.value), value=self._expression())
    def _set(self) -> ast.SetStatement:
        name = self._consume("IDENTIFIER", "expected state name"); self._consume("ASSIGN", "expected '='"); return ast.SetStatement(name=str(name.value), value=self._expression())
    def _apply(self) -> ast.Apply:
        sources = []
        if self._match("LBRACKET"):
            sources.append(self._expression())
            while self._match("COMMA"): sources.append(self._expression())
            self._consume("RBRACKET", "expected ']'")
        else: sources = [self._expression()]
        self._consume("ARROW", "expected '->'"); target = self._expression(); condition = strategy = tolerance = None
        if self._match("WHEN"): condition = self._expression()
        if self._match("USING"):
            strategy = str(self._consume("IDENTIFIER", "expected aggregation strategy").value)
            if self._match("LPAREN"):
                tolerance = self._expression(); self._consume("RPAREN", "expected ')'")
        if len(sources) > 1 and strategy is None: raise self._error(self._previous(), "multi-state apply requires 'using strategy'")
        return ast.Apply(sources=sources, target=target, condition=condition, strategy=strategy, tolerance=tolerance)
    def _transform_or_definition(self) -> ast.Transform | ast.TransformDefinition:
        if self._check("IDENTIFIER") and self.tokens[self.current + 1].kind == "LPAREN":
            name = str(self._consume("IDENTIFIER", "expected transform name").value)
            self._consume("LPAREN", "expected '('"); parameters = []
            if not self._check("RPAREN"):
                parameters.append(str(self._consume("IDENTIFIER", "expected parameter").value))
                while self._match("COMMA"): parameters.append(str(self._consume("IDENTIFIER", "expected parameter").value))
            self._consume("RPAREN", "expected ')'")
            return ast.TransformDefinition(name=name, parameters=parameters, body=self._block())
        targets = [self._expression()]
        while self._match("COMMA"): targets.append(self._expression())
        self._consume("WITH", "expected with"); call = self._call_name(); return ast.Transform(targets=targets, name=call.callee, args=call.args)
    def _compose(self) -> ast.Compose:
        left = self._expression(); self._consume("COMMA", "expected ','"); right = self._expression(); self._consume("USING", "expected using")
        mode = str(self._consume("IDENTIFIER", "expected composition mode").value); self._consume("ARROW", "expected '->'")
        outputs = []
        if self._match("LPAREN"):
            outputs.append(str(self._consume("IDENTIFIER", "expected output name").value)); self._consume("COMMA", "expected ','"); outputs.append(str(self._consume("IDENTIFIER", "expected output name").value)); self._consume("RPAREN", "expected ')'")
        else: outputs.append(str(self._consume("IDENTIFIER", "expected output name").value))
        return ast.Compose(left=left, right=right, mode=mode, outputs=outputs)
    def _if(self) -> ast.If:
        condition = self._expression(); then_body = self._block(); else_body = self._block() if self._match("ELSE") else []; return ast.If(condition=condition, then_body=then_body, else_body=else_body)
    def _while(self) -> ast.While:
        condition = self._expression(); return ast.While(condition=condition, body=self._block())
    def _function(self) -> ast.Function:
        name = str(self._consume("IDENTIFIER", "expected function name").value)
        self._consume("LPAREN", "expected '('"); parameters = []
        if not self._check("RPAREN"):
            parameters.append(str(self._consume("IDENTIFIER", "expected parameter name").value))
            while self._match("COMMA"): parameters.append(str(self._consume("IDENTIFIER", "expected parameter name").value))
        self._consume("RPAREN", "expected ')'")
        return ast.Function(name=name, parameters=parameters, body=self._block())
    def _block(self) -> list[ast.Node]:
        self._consume("LBRACE", "expected '{'"); body=[]
        while not self._check("RBRACE") and not self._check("EOF"): body.append(self._statement())
        self._consume("RBRACE", "expected '}'"); return body
    def _expression(self) -> ast.Node: return self._comparison()
    def _comparison(self) -> ast.Node:
        value = self._term()
        while self._match("EQ", "NE", "GT", "GE", "LT", "LE"):
            value = ast.Binary(left=value, operator=str(self._previous().value), right=self._term())
        return value
    def _term(self) -> ast.Node:
        value = self._factor()
        while self._match("PLUS", "MINUS"): value = ast.Binary(left=value, operator=str(self._previous().value), right=self._factor())
        return value
    def _factor(self) -> ast.Node:
        value = self._unary()
        while self._match("STAR", "SLASH"): value = ast.Binary(left=value, operator=str(self._previous().value), right=self._unary())
        return value
    def _unary(self) -> ast.Node:
        if self._match("MINUS"): return ast.Unary(operator="-", operand=self._unary())
        return self._primary()
    def _primary(self) -> ast.Node:
        if self._match("NUMBER"): return ast.Number(value=float(self._previous().value))
        if self._match("STRING"): return ast.String(value=str(self._previous().value))
        if self._match("TRUE"): return ast.Boolean(value=True)
        if self._match("FALSE"): return ast.Boolean(value=False)
        if self._match("MEASURE"):
            # Measurement consumes one state expression; the surrounding comparison
            # (for example measure(a).p1 > 0.8) belongs to the caller.
            if self._match("LPAREN"):
                source = self._expression(); self._consume("RPAREN", "expected ')' after measured state")
            else:
                source = ast.Name(value=str(self._consume("IDENTIFIER", "expected measured state name").value))
            mode="distribution"
            if self._match("USING"): mode=str(self._consume("IDENTIFIER", "expected measurement mode").value)
            value: ast.Node = ast.Measure(source=source, mode=mode)
        elif self._match("STATE"):
            self.current -= 1; value = self._state_literal()
        elif self._match("IDENTIFIER"):
            name = str(self._previous().value); value = self._call_name(name) if self._check("LPAREN") else ast.Name(value=name)
        elif self._match("LPAREN"):
            value=self._expression(); self._consume("RPAREN", "expected ')'")
        else: raise self._error(self._peek(), "expected expression")
        while self._match("DOT"): value=ast.Field(object=value, name=str(self._consume("IDENTIFIER", "expected property").value))
        return value
    def _call_name(self, name: str | None = None) -> ast.Call:
        if name is None: name=str(self._consume("IDENTIFIER", "expected function name").value)
        self._consume("LPAREN", "expected '('"); args=[]
        if not self._check("RPAREN"):
            args.append(self._expression())
            while self._match("COMMA"): args.append(self._expression())
        self._consume("RPAREN", "expected ')'"); return ast.Call(callee=name, args=args)
    def _match(self, *kinds: str) -> bool:
        if self._check(*kinds): self.current += 1; return True
        return False
    def _check(self, *kinds: str) -> bool: return self._peek().kind in kinds
    def _consume(self, kind: str, message: str) -> Token:
        if self._check(kind): self.current += 1; return self._previous()
        raise self._error(self._peek(), message)
    def _peek(self) -> Token: return self.tokens[self.current]
    def _previous(self) -> Token: return self.tokens[self.current - 1]
    def _error(self, token: Token, message: str) -> ParseError: return ParseError(f"line {token.line}, column {token.column}: {message}")


def parse(source: str) -> ast.Program: return Parser(source).parse()
