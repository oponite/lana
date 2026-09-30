//! Lana runtime (phase 2 of the Rust runtime boundary).
//!
//! Host calls, hardware adapters, and the durable pipeline (store, policy,
//! ledger, claims, effects) live here. Hardware adapters are isolated behind
//! the explicit C adapter ABI and are never an implicit dependency of the
//! portable core.

pub mod adapters;
#[doc(hidden)]
pub mod atomic_file;
pub mod information_codec;
pub mod rules;
pub mod trees;
pub mod dataset_identity;
pub mod dataset_snapshot;
mod dataset_source_codec;
pub mod document_extract;
#[cfg(not(target_arch = "wasm32"))]
pub mod dataset_sqlite;
pub mod claims;
pub mod codec;
pub mod data;
pub mod effects;
pub mod execution;
pub mod future_messages;
pub mod host_calls;
pub mod ledger;
pub mod policy;
pub mod sha256;
pub mod state_codec;
pub mod store;

pub use codec::{decode_document, decode_value, encode_value, format_17g};
pub use data::{csv_read, csv_write, json_parse, json_stringify};
pub use sha256::{sha256, Sha256, SHA256_DIGEST_SIZE};
