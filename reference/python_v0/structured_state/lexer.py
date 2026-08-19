from __future__ import annotations

from dataclasses import dataclass

from structured_state.errors import ParseError


KEYWORDS = {word: word.upper() for word in "state set apply when transform with compose using measure let if else while function return history latest duration true false".split()}
MULTI = {"->": "ARROW", "==": "EQ", "!=": "NE", ">=": "GE", "<=": "LE"}
SINGLE = {"(": "LPAREN", ")": "RPAREN", "{": "LBRACE", "}": "RBRACE", "[": "LBRACKET", "]": "RBRACKET", ",": "COMMA", ":": "COLON", ".": "DOT", "=": "ASSIGN", "+": "PLUS", "-": "MINUS", "*": "STAR", "/": "SLASH", ">": "GT", "<": "LT", ";": "SEMICOLON"}


@dataclass(frozen=True)
class Token:
    kind: str
    value: object
    line: int
    column: int


def lex(source: str) -> list[Token]:
    tokens: list[Token] = []
    index = line = 0
    column = 1
    while index < len(source):
        char = source[index]
        if char in " \t\r":
            index += 1; column += 1; continue
        if char == "\n":
            index += 1; line += 1; column = 1; continue
        if char == "#":
            while index < len(source) and source[index] != "\n": index += 1
            continue
        pair = source[index:index + 2]
        if pair in MULTI:
            tokens.append(Token(MULTI[pair], pair, line + 1, column)); index += 2; column += 2; continue
        if char in SINGLE:
            tokens.append(Token(SINGLE[char], char, line + 1, column)); index += 1; column += 1; continue
        if char == '"':
            start = column; index += 1; column += 1; value = ""
            while index < len(source) and source[index] != '"':
                if source[index] == "\n": raise ParseError(f"line {line + 1}: unterminated string")
                value += source[index]; index += 1; column += 1
            if index == len(source): raise ParseError(f"line {line + 1}: unterminated string")
            index += 1; column += 1; tokens.append(Token("STRING", value, line + 1, start)); continue
        if char.isdigit():
            start = index; start_col = column
            while index < len(source) and (source[index].isdigit() or source[index] == "."): index += 1; column += 1
            try: value = float(source[start:index])
            except ValueError as exc: raise ParseError(f"line {line + 1}: invalid number") from exc
            tokens.append(Token("NUMBER", value, line + 1, start_col)); continue
        if char.isalpha() or char == "_":
            start = index; start_col = column
            while index < len(source) and (source[index].isalnum() or source[index] == "_"): index += 1; column += 1
            value = source[start:index]; tokens.append(Token(KEYWORDS.get(value, "IDENTIFIER"), value, line + 1, start_col)); continue
        raise ParseError(f"line {line + 1}, column {column}: unexpected '{char}'")
    tokens.append(Token("EOF", None, line + 1, column))
    return tokens
