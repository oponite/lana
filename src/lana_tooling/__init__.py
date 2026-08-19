"""Python frontend tooling for the native Lana VM."""

from .compiler import CompileError, compile_source

__all__ = ["CompileError", "compile_source"]
