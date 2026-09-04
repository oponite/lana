//! Lana VM core (phase 2 of the Rust runtime boundary).
//!
//! A register VM semantically identical to the C11 reference (`vm/c/vm.c`):
//! same state math, same PCG32 stream, same error codes, same value printing.
//! The memory model differs — native Rust ownership instead of a mark-sweep GC
//! — but the 256 MiB limit is preserved by byte accounting.
//!
//! Increment 1 covers the scalar/array/control-flow/state/history opcodes.
//! Increments 2-5 add state dists and the 2.0 ISA ops, information types,
//! tasks, and host calls; increment 6 is the differential conformance harness.

pub mod derivation;
pub mod inspect;
pub mod rng;
pub mod state;
pub mod state_dist;
pub mod value;
pub mod vm;

pub use derivation::{Derivation, DerivationExactness, DerivationKind, DerivationOutcome};
pub use inspect::{inspect, InspectFormat};
pub use rng::Rng;
pub use state::{State, StateValue};
pub use value::{
    Adt, CapabilityToken, Claim, EffectReceipt, JointKind, JointRow, JointState, PathAlternative,
    PathSet, PlannedEffect, Possibility, Reactive, ReactiveKind, ReactiveVersion,
    RelationshipKind, SharedCommit, SharedInformation, SharedObservation, SharedState,
    SharedVersion, Task, Value, ValueKind, VmError, ADT_UNKNOWN_VARIANT, LANA_CAPABILITY_ADMIN,
    LANA_CAPABILITY_OBSERVE, LANA_CAPABILITY_READ,
};
pub use vm::{
    exact_support_name, resolution_reason_name, resource_kind_name, Frame, History, HistoryPolicy,
    Vm, LANA_EXACT_SUPPORT_AVAILABLE, LANA_EXACT_SUPPORT_UNAVAILABLE, LANA_EXACT_SUPPORT_UNKNOWN,
    LANA_RESOLUTION_REASON_CANCELLED, LANA_RESOLUTION_REASON_CONTRADICTION,
    LANA_RESOLUTION_REASON_INVALID_CONDITIONING, LANA_RESOLUTION_REASON_MULTIPLE_ALTERNATIVES,
    LANA_RESOLUTION_REASON_NONE, LANA_RESOLUTION_REASON_NO_ALTERNATIVES,
    LANA_RESOLUTION_REASON_RESOURCE_LIMIT, LANA_RESOLUTION_REASON_UNSUPPORTED_EXACT,
    LANA_RESOURCE_INSTRUCTIONS, LANA_RESOURCE_MEMORY, LANA_RESOURCE_PATHS, LANA_RESOURCE_SAMPLES,
    LANA_RESOURCE_TASKS, LANA_RESOURCE_TIME, LANA_HOST_STORE_OPEN, LANA_HOST_STORE_PUT,
    LANA_HOST_STORE_GET, LANA_HOST_STORE_DELETE, LANA_HOST_STORE_COMMIT, LANA_HOST_STORE_SCAN,
    LANA_HOST_STORE_CURRENT_REVISION, LANA_HOST_POLICY_EVALUATE, LANA_HOST_POLICY_STORE_DECISION,
    LANA_HOST_LEDGER_APPEND, LANA_HOST_LEDGER_QUERY,
};
