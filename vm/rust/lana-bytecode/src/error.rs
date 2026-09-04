//! Lana error codes, mirroring `vm/include/error.h`.
//!
//! The numeric values are stable and MUST match the C11 `LanaError` enum so
//! that differential conformance can compare error codes byte-for-byte.

/// Lana error codes. The discriminants match `LanaError` in `vm/include/error.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum LanaError {
    Ok = 0,
    InvalidState,
    InvalidProbability,
    InvalidDependency,
    Type,
    Register,
    Opcode,
    Constant,
    Jump,
    Transform,
    Compose,
    Measure,
    History,
    Format,
    IncompatibleFormat,
    Io,
    Oom,
    Limit,
    Task,
    Cancelled,
    Timeout,
    InvalidTransformResult,
    UnsupportedOperation,
    UnsupportedExactMeasurement,
    InvalidDistribution,
    BudgetExhausted,
    Key,
    Parse,
    Assertion,
    InvalidConditioning,
    UnresolvedValue,
    PathLimit,
    Capability,
    Conflict,
    NotFound,
    CompactedHistory,
    Schema,
    UnsupportedValue,
    Corruption,
    InvalidParameters,
    ClaimMismatch,
    ClaimRevoked,
    ClaimExpired,
    UnauthorizedIssuer,
    Integrity,
    NoMatchingEvent,
}

impl LanaError {
    /// The error kind name, mirroring `lana_error_kind_from_code` +
    /// `lana_error_kind_name` in `vm/c/error.c` (e.g. `validation` for
    /// `LANA_ERR_HISTORY`). Used by the CLI to format `error[kind/code]`.
    pub fn kind_name(self) -> &'static str {
        use LanaError::*;
        match self {
            Ok => "none",
            Type => "type",
            Parse => "parse",
            Register | Opcode | Constant | Jump | Format | IncompatibleFormat => "bytecode",
            Io => "io",
            Oom | Limit | BudgetExhausted | PathLimit => "resource-limit",
            Task => "task",
            Cancelled => "cancellation",
            Timeout => "timeout",
            UnsupportedOperation | UnsupportedExactMeasurement => "unsupported",
            InvalidConditioning | UnresolvedValue => "resolution",
            Assertion => "assertion",
            InvalidState | InvalidProbability | InvalidDependency | Transform | Compose
            | Measure | History | InvalidTransformResult | InvalidDistribution | Key
            | Capability | Conflict | NotFound | CompactedHistory | Schema
            | UnsupportedValue | Corruption | InvalidParameters | ClaimMismatch
            | ClaimRevoked | ClaimExpired | UnauthorizedIssuer | Integrity
            | NoMatchingEvent => "validation",
        }
    }

    /// The stable C11 name for this error code, e.g. `LANA_ERR_TYPE`.
    pub fn name(self) -> &'static str {
        use LanaError::*;
        match self {
            Ok => "LANA_OK",
            InvalidState => "LANA_ERR_INVALID_STATE",
            InvalidProbability => "LANA_ERR_INVALID_PROBABILITY",
            InvalidDependency => "LANA_ERR_INVALID_DEPENDENCY",
            Type => "LANA_ERR_TYPE",
            Register => "LANA_ERR_REGISTER",
            Opcode => "LANA_ERR_OPCODE",
            Constant => "LANA_ERR_CONSTANT",
            Jump => "LANA_ERR_JUMP",
            Transform => "LANA_ERR_TRANSFORM",
            Compose => "LANA_ERR_COMPOSE",
            Measure => "LANA_ERR_MEASURE",
            History => "LANA_ERR_HISTORY",
            Format => "LANA_ERR_FORMAT",
            IncompatibleFormat => "LANA_ERR_INCOMPATIBLE_FORMAT",
            Io => "LANA_ERR_IO",
            Oom => "LANA_ERR_OOM",
            Limit => "LANA_ERR_LIMIT",
            Task => "LANA_ERR_TASK",
            Cancelled => "LANA_ERR_CANCELLED",
            Timeout => "LANA_ERR_TIMEOUT",
            InvalidTransformResult => "LANA_ERR_INVALID_TRANSFORM_RESULT",
            UnsupportedOperation => "LANA_ERR_UNSUPPORTED_OPERATION",
            UnsupportedExactMeasurement => "LANA_ERR_UNSUPPORTED_EXACT_MEASUREMENT",
            InvalidDistribution => "LANA_ERR_INVALID_DISTRIBUTION",
            BudgetExhausted => "LANA_ERR_BUDGET_EXHAUSTED",
            Key => "LANA_ERR_KEY",
            Parse => "LANA_ERR_PARSE",
            Assertion => "LANA_ERR_ASSERTION",
            InvalidConditioning => "LANA_ERR_INVALID_CONDITIONING",
            UnresolvedValue => "LANA_ERR_UNRESOLVED_VALUE",
            PathLimit => "LANA_ERR_PATH_LIMIT",
            Capability => "LANA_ERR_CAPABILITY",
            Conflict => "LANA_ERR_CONFLICT",
            NotFound => "LANA_ERR_NOT_FOUND",
            CompactedHistory => "LANA_ERR_COMPACTED_HISTORY",
            Schema => "LANA_ERR_SCHEMA",
            UnsupportedValue => "LANA_ERR_UNSUPPORTED_VALUE",
            Corruption => "LANA_ERR_CORRUPTION",
            InvalidParameters => "LANA_ERR_INVALID_PARAMETERS",
            ClaimMismatch => "LANA_ERR_CLAIM_MISMATCH",
            ClaimRevoked => "LANA_ERR_CLAIM_REVOKED",
            ClaimExpired => "LANA_ERR_CLAIM_EXPIRED",
            UnauthorizedIssuer => "LANA_ERR_UNAUTHORIZED_ISSUER",
            Integrity => "LANA_ERR_INTEGRITY",
            NoMatchingEvent => "LANA_ERR_NO_MATCHING_EVENT",
        }
    }
}

impl core::fmt::Display for LanaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl core::error::Error for LanaError {}

/// A structured verification failure, mirroring the fields the C11 verifier
/// records in `LanaErrorInfo` that are relevant to bytecode verification.
#[derive(Debug, Clone)]
pub struct LanaErrorInfo {
    pub code: LanaError,
    pub ip: usize,
    pub opcode: u8,
    pub line: u32,
    pub message: String,
}

impl LanaErrorInfo {
    pub fn new(code: LanaError, ip: usize, opcode: u8, line: u32, message: impl Into<String>) -> Self {
        Self { code, ip, opcode, line, message: message.into() }
    }
}

impl From<LanaError> for LanaErrorInfo {
    fn from(code: LanaError) -> Self {
        Self::new(code, 0, 0, 0, "")
    }
}
