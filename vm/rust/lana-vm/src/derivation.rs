//! Derivation model, mirroring `LanaDerivation` in `vm/include/value.h` and
//! the derivation helpers in `vm/c/vm.c`.
//!
//! Derivations are immutable records: each operation that produces a value
//! attaches a derivation describing how the value was obtained. The derivation
//! graph is a DAG (inputs reference pre-existing derivations), so `Arc`
//! without cycle collection is sound.

use std::sync::Arc;

/// How a value was derived, matching `LanaDerivationKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DerivationKind {
    Evidence = 0,
    Assumption,
    Operation,
    Observation,
    Path,
    Sample,
    Approximation,
    Resolution,
}

/// The exactness of a derivation, matching `LanaDerivationExactness`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum DerivationExactness {
    Exact = 0,
    Sample,
    Approximate,
}

/// The outcome of a derivation, matching `LanaDerivationOutcome`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DerivationOutcome {
    Success = 0,
    Unresolved,
    Unsupported,
    Error,
}

/// The evidence status of a derivation, ordered least to most certain. It is a
/// derived label over (kind, exactness, outcome), not a stored field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum EvidenceStatus {
    Unknown = 0,
    Sampled,
    Modeled,
    Exact,
    Observed,
}

/// A derivation record, matching `struct LanaDerivation`.
#[derive(Debug, Clone)]
pub struct Derivation {
    pub task_lineage: u64,
    pub local_sequence: u64,
    pub revision: u64,
    pub kind: DerivationKind,
    pub operation: Arc<str>,
    pub inputs: Vec<Arc<Derivation>>,
    pub label: Arc<str>,
    pub function: Arc<str>,
    pub line: u32,
    pub exactness: DerivationExactness,
    pub details: Arc<str>,
    pub outcome: DerivationOutcome,
    pub reason: Arc<str>,
}

/// The stable kind name, matching `derivation_kind_name` in `vm/c/vm.c`.
pub fn kind_name(kind: DerivationKind) -> &'static str {
    match kind {
        DerivationKind::Evidence => "evidence",
        DerivationKind::Assumption => "assumption",
        DerivationKind::Operation => "operation",
        DerivationKind::Observation => "observation",
        DerivationKind::Path => "path",
        DerivationKind::Sample => "sample",
        DerivationKind::Approximation => "approximation",
        DerivationKind::Resolution => "resolution",
    }
}

/// The stable exactness name, matching `derivation_exactness_name`.
pub fn exactness_name(exactness: DerivationExactness) -> &'static str {
    match exactness {
        DerivationExactness::Exact => "exact",
        DerivationExactness::Sample => "sample",
        DerivationExactness::Approximate => "approximate",
    }
}

/// The stable outcome name, matching `derivation_outcome_name`.
pub fn outcome_name(outcome: DerivationOutcome) -> &'static str {
    match outcome {
        DerivationOutcome::Success => "success",
        DerivationOutcome::Unresolved => "unresolved",
        DerivationOutcome::Unsupported => "unsupported",
        DerivationOutcome::Error => "error",
    }
}

/// The stable evidence status name, matching `lana_evidence_status_name`.
pub fn status_name(status: EvidenceStatus) -> &'static str {
    match status {
        EvidenceStatus::Unknown => "unknown",
        EvidenceStatus::Sampled => "sampled",
        EvidenceStatus::Modeled => "modeled",
        EvidenceStatus::Exact => "exact",
        EvidenceStatus::Observed => "observed",
    }
}

impl Derivation {
    /// The evidence status of this derivation, matching `lana_derivation_status`.
    /// A `None` derivation (a bare literal) is treated as exact.
    pub fn status(&self) -> EvidenceStatus {
        if self.outcome == DerivationOutcome::Unresolved {
            return EvidenceStatus::Unknown;
        }
        if self.kind == DerivationKind::Observation && self.exactness == DerivationExactness::Exact {
            return EvidenceStatus::Observed;
        }
        if self.kind == DerivationKind::Assumption
            && self.exactness == DerivationExactness::Approximate
        {
            return EvidenceStatus::Modeled;
        }
        if self.kind == DerivationKind::Sample && self.exactness == DerivationExactness::Sample {
            return EvidenceStatus::Sampled;
        }
        match self.exactness {
            DerivationExactness::Exact => EvidenceStatus::Exact,
            DerivationExactness::Approximate => EvidenceStatus::Modeled,
            DerivationExactness::Sample => EvidenceStatus::Sampled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(kind: DerivationKind, exactness: DerivationExactness, outcome: DerivationOutcome) -> Derivation {
        Derivation {
            task_lineage: 0,
            local_sequence: 0,
            revision: 0,
            kind,
            operation: Arc::from("op"),
            inputs: Vec::new(),
            label: Arc::from(""),
            function: Arc::from(""),
            line: 0,
            exactness,
            details: Arc::from(""),
            outcome,
            reason: Arc::from("none"),
        }
    }

    #[test]
    fn status_classifies_each_row() {
        assert_eq!(
            d(DerivationKind::Observation, DerivationExactness::Exact, DerivationOutcome::Success).status(),
            EvidenceStatus::Observed
        );
        assert_eq!(
            d(DerivationKind::Evidence, DerivationExactness::Exact, DerivationOutcome::Success).status(),
            EvidenceStatus::Exact
        );
        assert_eq!(
            d(DerivationKind::Operation, DerivationExactness::Exact, DerivationOutcome::Success).status(),
            EvidenceStatus::Exact
        );
        assert_eq!(
            d(DerivationKind::Assumption, DerivationExactness::Approximate, DerivationOutcome::Success).status(),
            EvidenceStatus::Modeled
        );
        assert_eq!(
            d(DerivationKind::Sample, DerivationExactness::Sample, DerivationOutcome::Success).status(),
            EvidenceStatus::Sampled
        );
        assert_eq!(
            d(DerivationKind::Operation, DerivationExactness::Exact, DerivationOutcome::Unresolved).status(),
            EvidenceStatus::Unknown
        );
    }

    #[test]
    fn status_orders_least_to_most_certain() {
        assert!(EvidenceStatus::Unknown < EvidenceStatus::Sampled);
        assert!(EvidenceStatus::Sampled < EvidenceStatus::Modeled);
        assert!(EvidenceStatus::Modeled < EvidenceStatus::Exact);
        assert!(EvidenceStatus::Exact < EvidenceStatus::Observed);
    }

    #[test]
    fn status_name_is_stable() {
        assert_eq!(status_name(EvidenceStatus::Unknown), "unknown");
        assert_eq!(status_name(EvidenceStatus::Sampled), "sampled");
        assert_eq!(status_name(EvidenceStatus::Modeled), "modeled");
        assert_eq!(status_name(EvidenceStatus::Exact), "exact");
        assert_eq!(status_name(EvidenceStatus::Observed), "observed");
    }
}
