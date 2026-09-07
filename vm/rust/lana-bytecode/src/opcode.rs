//! LABC opcodes, mirroring `OpCode` in `vm/include/bytecode.h`.
//!
//! The discriminants are stable and MUST match the C11 enum so that chunks
//! verify identically under both implementations.

/// LABC opcodes. The discriminants match `OpCode` in `vm/include/bytecode.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OpCode {
    Nop = 0,
    LoadConst,
    Move,
    StateNew,
    StateBuild,
    Transform,
    Measure,
    Append,
    SampleStateDist,
    MeasureBasis,
    EstimateMeasureProbability,
    EstimateMeasureDistribution,
    GetField,
    GetIndex,
    SetIndex,
    HistoryConfig,
    Previous,
    Change,
    Velocity,
    Binary,
    Unary,
    Compare,
    Jump,
    JumpIfTrue,
    JumpIfFalse,
    ArrayNew,
    ArrayGet,
    ArraySet,
    Call,
    Return,
    Print,
    Halt,

    Fork,
    Join,
    JoinTimeout,
    JoinAll,
    Cancel,
    TaskgroupEnter,
    TaskgroupExit,
    HostCall,

    JointBuild,
    JointProject,
    JointCondition,
    JointSample,
    Resolve,
    JointBuildFinite,
    JointRename,
    PossibilityBuild,
    PathSplit,
    PathJoin,
    Observe,
    InfoSample,
    Evidence,
    Assume,
    Derivation,
    Explain,
    /* Lana 2.0 ISA operations. */
    Mix,
    Map,
    Support,
    Expect,
    Validate,
    Revision,
    Attenuate,
    TraceDistance,
    AppendRedundant,
    AppendFullRedundancy,
    AppendComplementary,
    /* Lana 2.0 ADTs and pattern matching. */
    AdtBuild,
    AdtCase,
    AdtGet,
    /* Lana 2.0 lazy bounded datasets. */
    Lazy,
    Force,
    /* Lana 2.0 deterministic resampling. */
    Bootstrap,
    /* Lana 2.1 reverse-mode autodiff (LIP-011): load a function value so it
     * can be passed to the `grad`/`vjp` host calls. */
    LoadFunction,
    /* Lana 2.1 generator suspension (LIP-022 §2). */
    Generator,
    Yield,
    Next,
    /* Lana 2.2 async/await (LIP-024). Operand semantics:
     *   OP_ASYNC <target> <arg0> <argcount> <dest> — create a cold future by
     *     calling async function <target> with <argcount> arguments starting at
     *     register <arg0>, storing the future in <dest>. The body is NOT
     *     executed (mirrors OP_GENERATOR).
     *   OP_AWAIT <future> <dest> — suspend the current async frame until the
     *     future in <future> completes, then store its result in <dest> and
     *     yield control to the event loop.
     *   OP_RUN_ASYNC <future> <dest> — run the event loop to completion on the
     *     future in <future>, then store its result in <dest>. */
    Async,
    Await,
    RunAsync,
    Count,
}

impl OpCode {
    /// The stable C11 mnemonic for this opcode, matching `lana_opcode_name`
    /// in `vm/c/bytecode.c`, e.g. `MIX`.
    pub fn name(self) -> &'static str {
        use OpCode::*;
        match self {
            Nop => "NOP",
            LoadConst => "LOAD_CONST",
            Move => "MOVE",
            StateNew => "STATE_NEW",
            StateBuild => "STATE_BUILD",
            Transform => "TRANSFORM",
            Measure => "MEASURE",
            Append => "APPEND",
            SampleStateDist => "SAMPLE_STATE_DIST",
            MeasureBasis => "MEASURE_BASIS",
            EstimateMeasureProbability => "ESTIMATE_MEASURE_PROBABILITY",
            EstimateMeasureDistribution => "ESTIMATE_MEASURE_DISTRIBUTION",
            GetField => "GET_FIELD",
            GetIndex => "GET_INDEX",
            SetIndex => "SET_INDEX",
            HistoryConfig => "HISTORY_CONFIG",
            Previous => "PREVIOUS",
            Change => "CHANGE",
            Velocity => "VELOCITY",
            Binary => "BINARY",
            Unary => "UNARY",
            Compare => "COMPARE",
            Jump => "JUMP",
            JumpIfTrue => "JUMP_IF_TRUE",
            JumpIfFalse => "JUMP_IF_FALSE",
            ArrayNew => "ARRAY_NEW",
            ArrayGet => "ARRAY_GET",
            ArraySet => "ARRAY_SET",
            Call => "CALL",
            Return => "RETURN",
            Print => "PRINT",
            Halt => "HALT",
            Fork => "FORK",
            Join => "JOIN",
            JoinTimeout => "JOIN_TIMEOUT",
            JoinAll => "JOIN_ALL",
            Cancel => "CANCEL",
            TaskgroupEnter => "TASKGROUP_ENTER",
            TaskgroupExit => "TASKGROUP_EXIT",
            HostCall => "HOST_CALL",
            JointBuild => "JOINT_BUILD",
            JointProject => "JOINT_PROJECT",
            JointCondition => "JOINT_CONDITION",
            JointSample => "JOINT_SAMPLE",
            Resolve => "RESOLVE",
            JointBuildFinite => "JOINT_BUILD_FINITE",
            JointRename => "JOINT_RENAME",
            PossibilityBuild => "POSSIBILITY_BUILD",
            PathSplit => "PATH_SPLIT",
            PathJoin => "PATH_JOIN",
            Observe => "OBSERVE",
            InfoSample => "INFO_SAMPLE",
            Evidence => "EVIDENCE",
            Assume => "ASSUME",
            Derivation => "DERIVATION",
            Explain => "EXPLAIN",
            Mix => "MIX",
            Map => "MAP",
            Support => "SUPPORT",
            Expect => "EXPECT",
            Validate => "VALIDATE",
            Revision => "REVISION",
            Attenuate => "ATTENUATE",
            TraceDistance => "TRACE_DISTANCE",
            AppendRedundant => "APPEND_REDUNDANT",
            AppendFullRedundancy => "APPEND_FULL_REDUNDANCY",
            AppendComplementary => "APPEND_COMPLEMENTARY",
            AdtBuild => "ADT_BUILD",
            AdtCase => "ADT_CASE",
            AdtGet => "ADT_GET",
            Lazy => "LAZY",
            Force => "FORCE",
            Bootstrap => "BOOTSTRAP",
            Generator => "GENERATOR",
            Yield => "YIELD",
            Next => "NEXT",
            Async => "ASYNC",
            Await => "AWAIT",
            RunAsync => "RUN_ASYNC",
            LoadFunction => "LOAD_FUNCTION",
            Count => "COUNT",
        }
    }
}

impl TryFrom<u8> for OpCode {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if value < Self::Count as u8 {
            // SAFETY: every discriminant below Count is a valid OpCode.
            Ok(unsafe { core::mem::transmute::<u8, OpCode>(value) })
        } else {
            Err(())
        }
    }
}

/// LABC version constants, mirroring `vm/include/bytecode.h`.
pub const LABC_VERSION: u32 = 2;
pub const LABC_VERSION_1: u32 = 1;
pub const LABC_VERSION_3: u32 = 3;
pub const LABC_VERSION_4: u32 = 4;
pub const LANA_MAX_REGISTERS: u32 = 256;
pub const LANA_MAX_CALL_FRAMES: u32 = 64;
