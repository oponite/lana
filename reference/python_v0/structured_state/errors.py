class LanaError(Exception):
    """Base class for Lana errors."""


class InvalidStateError(LanaError):
    pass


class InvalidProbabilityError(InvalidStateError):
    pass


class InvalidDependencyError(InvalidStateError):
    pass


class ParseError(LanaError):
    pass


class CompileError(LanaError):
    pass


class VMError(LanaError):
    pass


class LanaTypeError(VMError):
    pass


class LanaNameError(VMError):
    pass


class MeasurementError(VMError):
    pass
