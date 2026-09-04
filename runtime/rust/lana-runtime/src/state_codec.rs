//! State codec, mirroring `vm/c/state.codec.c` and
//! `runtime/include/state_codec.h`.
//!
//! Two encodings live here:
//!
//! * `state_encode`/`state_decode` — a fixed 32-byte binary record
//!   (`"LST1"` + big-endian schema + three big-endian IEEE-754 doubles).
//! * `persistent_state_encode`/`persistent_state_decode` — a canonical JSON
//!   document (via `crate::codec`) carrying the state, its metadata indexes,
//!   and optional provenance.
//!
//! Both are byte-identical to the C11 reference.

use std::sync::{Arc, Mutex};

use lana_bytecode::LanaError;
use lana_vm::state::{state_valid, State};
use lana_vm::value::{Map, Value, ValueKind};

use crate::codec::{decode_document, encode_value};

pub const STATE_CODEC_SCHEMA: u32 = 1;
pub const STATE_CODEC_LENGTH: usize = 32;

/// Encode a state as the fixed 32-byte binary record.
pub fn state_encode(state: &State) -> Result<Vec<u8>, LanaError> {
    if !state_valid(state) {
        return Err(LanaError::InvalidState);
    }
    let mut out = Vec::with_capacity(STATE_CODEC_LENGTH);
    out.extend_from_slice(b"LST1");
    out.extend_from_slice(&STATE_CODEC_SCHEMA.to_be_bytes());
    out.extend_from_slice(&state.p.to_be_bytes());
    out.extend_from_slice(&state.d_re.to_be_bytes());
    out.extend_from_slice(&state.d_im.to_be_bytes());
    Ok(out)
}

/// Decode the fixed 32-byte binary record back into a state.
pub fn state_decode(buf: &[u8]) -> Result<State, LanaError> {
    if buf.len() != STATE_CODEC_LENGTH || buf[0..4] != *b"LST1" {
        return Err(LanaError::Schema);
    }
    let schema = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    if schema != STATE_CODEC_SCHEMA {
        return Err(LanaError::Schema);
    }
    let p = f64::from_be_bytes(buf[8..16].try_into().unwrap());
    let d_re = f64::from_be_bytes(buf[16..24].try_into().unwrap());
    let d_im = f64::from_be_bytes(buf[24..32].try_into().unwrap());
    let state = State { p, d_re, d_im };
    if !state_valid(&state) {
        return Err(LanaError::Schema);
    }
    Ok(state)
}

/// Metadata indexes attached to a persistent state, mirroring
/// `LanaPersistentStateMetadata`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PersistentStateMetadata {
    pub has_timestamp: bool,
    pub has_source: bool,
    pub has_weight: bool,
    pub has_confidence: bool,
    pub timestamp: f64,
    pub weight: f64,
    pub confidence: f64,
    pub source: Option<Arc<str>>,
}

/// Provenance attached to a persistent state, mirroring
/// `LanaPersistentStateProvenance`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PersistentStateProvenance {
    pub derivation_id: u64,
    pub input_revision: u64,
    pub kind: Option<Arc<str>>,
    pub operation: Option<Arc<str>>,
    pub captured_inputs: Option<Arc<str>>,
}

/// A persistent state: the state plus metadata and optional provenance,
/// mirroring `LanaPersistentState`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PersistentState {
    pub schema_version: u32,
    pub state: State,
    pub metadata: PersistentStateMetadata,
    pub has_provenance: bool,
    pub provenance: PersistentStateProvenance,
}

const TOP_KEYS: [&str; 6] = ["schema", "p", "d_re", "d_im", "metadata", "provenance"];
const METADATA_KEYS: [&str; 4] = ["timestamp", "source", "weight", "confidence"];
const PROVENANCE_KEYS: [&str; 5] =
    ["derivation_id", "input_revision", "kind", "operation", "captured_inputs"];

/// Whether `map` has exactly the keys in `keys` (and no others), mirroring
/// `map_has_exact_keys`.
fn map_has_exact_keys(map: &Map, keys: &[&str]) -> bool {
    if map.entries.len() != keys.len() {
        return false;
    }
    keys.iter().all(|key| map.has(key))
}

/// Look up a key, mirroring `map_find_value`. Callers guarantee the key is
/// present (via `map_has_exact_keys`), so a missing key is a bug.
fn map_find_value<'a>(map: &'a Map, key: &str) -> &'a Value {
    map.get(key).expect("key guaranteed present by map_has_exact_keys")
}

/// Parse a decimal `u64` from a string value, mirroring `parse_u64` (which
/// uses `strtoull` and rejects empty, partial, and overflowing input).
fn parse_u64(value: &Value) -> Result<u64, LanaError> {
    match &value.kind {
        ValueKind::String(string) => {
            if string.is_empty() {
                return Err(LanaError::Schema);
            }
            string.parse::<u64>().map_err(|_| LanaError::Schema)
        }
        _ => Err(LanaError::Schema),
    }
}

/// Require a number value, mirroring the `value.type != VAL_NUMBER` checks.
fn expect_number(value: &Value) -> Result<f64, LanaError> {
    match &value.kind {
        ValueKind::Number(number) => Ok(*number),
        _ => Err(LanaError::Schema),
    }
}

/// Require a string value, mirroring the `value.type != VAL_STRING` checks.
fn expect_string(value: &Value) -> Result<Arc<str>, LanaError> {
    match &value.kind {
        ValueKind::String(string) => Ok(string.clone()),
        _ => Err(LanaError::Schema),
    }
}

/// A number-or-null value, returning `(has_value, value)`.
fn number_or_null(value: &Value) -> Result<(bool, f64), LanaError> {
    match &value.kind {
        ValueKind::Null => Ok((false, 0.0)),
        ValueKind::Number(number) => Ok((true, *number)),
        _ => Err(LanaError::Schema),
    }
}

/// A string-or-null value, returning `(has_value, value)`.
fn string_or_null(value: &Value) -> Result<(bool, Option<Arc<str>>), LanaError> {
    match &value.kind {
        ValueKind::Null => Ok((false, None)),
        ValueKind::String(string) => Ok((true, Some(string.clone()))),
        _ => Err(LanaError::Schema),
    }
}

/// Encode a persistent state as a canonical JSON document.
pub fn persistent_state_encode(state: &PersistentState) -> Result<Vec<u8>, LanaError> {
    if !state_valid(&state.state) || state.schema_version != STATE_CODEC_SCHEMA {
        return Err(LanaError::InvalidState);
    }
    if (state.metadata.has_source && state.metadata.source.is_none())
        || (state.has_provenance
            && (state.provenance.kind.is_none()
                || state.provenance.operation.is_none()
                || state.provenance.captured_inputs.is_none()))
    {
        return Err(LanaError::Schema);
    }

    let mut metadata = Map::new(4);
    metadata
        .set(
            Arc::from("confidence"),
            if state.metadata.has_confidence {
                Value::number(state.metadata.confidence)
            } else {
                Value::null()
            },
            false,
        )
        .unwrap();
    metadata
        .set(
            Arc::from("source"),
            if state.metadata.has_source {
                Value::string(state.metadata.source.clone().unwrap())
            } else {
                Value::null()
            },
            false,
        )
        .unwrap();
    metadata
        .set(
            Arc::from("timestamp"),
            if state.metadata.has_timestamp {
                Value::number(state.metadata.timestamp)
            } else {
                Value::null()
            },
            false,
        )
        .unwrap();
    metadata
        .set(
            Arc::from("weight"),
            if state.metadata.has_weight {
                Value::number(state.metadata.weight)
            } else {
                Value::null()
            },
            false,
        )
        .unwrap();

    let mut top = Map::new(6);
    top.set(Arc::from("d_im"), Value::number(state.state.d_im), false).unwrap();
    top.set(Arc::from("d_re"), Value::number(state.state.d_re), false).unwrap();
    top.set(Arc::from("metadata"), Value::map(Arc::new(Mutex::new(metadata))), false).unwrap();
    top.set(Arc::from("p"), Value::number(state.state.p), false).unwrap();
    if state.has_provenance {
        let mut provenance = Map::new(5);
        provenance
            .set(
                Arc::from("captured_inputs"),
                Value::string(state.provenance.captured_inputs.clone().unwrap()),
                false,
            )
            .unwrap();
        provenance
            .set(
                Arc::from("derivation_id"),
                Value::string(Arc::from(state.provenance.derivation_id.to_string())),
                false,
            )
            .unwrap();
        provenance
            .set(
                Arc::from("input_revision"),
                Value::string(Arc::from(state.provenance.input_revision.to_string())),
                false,
            )
            .unwrap();
        provenance
            .set(
                Arc::from("kind"),
                Value::string(state.provenance.kind.clone().unwrap()),
                false,
            )
            .unwrap();
        provenance
            .set(
                Arc::from("operation"),
                Value::string(state.provenance.operation.clone().unwrap()),
                false,
            )
            .unwrap();
        top.set(Arc::from("provenance"), Value::map(Arc::new(Mutex::new(provenance))), false).unwrap();
    } else {
        top.set(Arc::from("provenance"), Value::null(), false).unwrap();
    }
    top.set(Arc::from("schema"), Value::number(STATE_CODEC_SCHEMA as f64), false).unwrap();

    let encoded = encode_value(&Value::map(Arc::new(Mutex::new(top))))?;
    Ok(encoded.into_bytes())
}

/// Decode a persistent state from a canonical JSON document.
pub fn persistent_state_decode(buf: &[u8]) -> Result<PersistentState, LanaError> {
    let root = decode_document(buf)?;
    let root_arc = match &root.kind {
        ValueKind::Map(map) => map,
        _ => return Err(LanaError::Schema),
    };
    let root_map = root_arc.lock().unwrap();
    if !map_has_exact_keys(&root_map, &TOP_KEYS) {
        return Err(LanaError::Schema);
    }

    let schema = expect_number(map_find_value(&root_map, "schema"))?;
    if schema != STATE_CODEC_SCHEMA as f64 {
        return Err(LanaError::Schema);
    }
    let p = expect_number(map_find_value(&root_map, "p"))?;
    let d_re = expect_number(map_find_value(&root_map, "d_re"))?;
    let d_im = expect_number(map_find_value(&root_map, "d_im"))?;
    let state = State { p, d_re, d_im };
    if !state_valid(&state) {
        return Err(LanaError::Schema);
    }

    let metadata_value = map_find_value(&root_map, "metadata");
    let metadata_arc = match &metadata_value.kind {
        ValueKind::Map(map) => map,
        _ => return Err(LanaError::Schema),
    };
    let metadata_map = metadata_arc.lock().unwrap();
    if !map_has_exact_keys(&metadata_map, &METADATA_KEYS) {
        return Err(LanaError::Schema);
    }
    let (has_timestamp, timestamp) = number_or_null(map_find_value(&metadata_map, "timestamp"))?;
    let (has_weight, weight) = number_or_null(map_find_value(&metadata_map, "weight"))?;
    let (has_confidence, confidence) = number_or_null(map_find_value(&metadata_map, "confidence"))?;
    let (has_source, source) = string_or_null(map_find_value(&metadata_map, "source"))?;
    let metadata = PersistentStateMetadata {
        has_timestamp,
        has_source,
        has_weight,
        has_confidence,
        timestamp,
        weight,
        confidence,
        source,
    };

    let provenance_value = map_find_value(&root_map, "provenance");
    let mut has_provenance = false;
    let mut provenance = PersistentStateProvenance::default();
    match &provenance_value.kind {
        ValueKind::Null => {}
        ValueKind::Map(map) => {
            let provenance_map = map.lock().unwrap();
            if !map_has_exact_keys(&provenance_map, &PROVENANCE_KEYS) {
                return Err(LanaError::Schema);
            }
            has_provenance = true;
            provenance.derivation_id = parse_u64(map_find_value(&provenance_map, "derivation_id"))?;
            provenance.input_revision = parse_u64(map_find_value(&provenance_map, "input_revision"))?;
            provenance.kind = Some(expect_string(map_find_value(&provenance_map, "kind"))?);
            provenance.operation = Some(expect_string(map_find_value(&provenance_map, "operation"))?);
            provenance.captured_inputs =
                Some(expect_string(map_find_value(&provenance_map, "captured_inputs"))?);
        }
        _ => return Err(LanaError::Schema),
    }

    Ok(PersistentState {
        schema_version: STATE_CODEC_SCHEMA,
        state,
        metadata,
        has_provenance,
        provenance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_round_trip() {
        let state = State { p: 0.5, d_re: 0.3, d_im: 0.1 };
        let encoded = state_encode(&state).unwrap();
        assert_eq!(encoded.len(), STATE_CODEC_LENGTH);
        assert_eq!(&encoded[0..4], b"LST1");
        assert_eq!(state_decode(&encoded).unwrap(), state);
    }

    #[test]
    fn state_decode_rejects_wrong_length() {
        let mut buf = vec![0u8; STATE_CODEC_LENGTH - 1];
        buf[0..4].copy_from_slice(b"LST1");
        assert_eq!(state_decode(&buf), Err(LanaError::Schema));
    }

    #[test]
    fn state_decode_rejects_wrong_magic() {
        let mut buf = vec![0u8; STATE_CODEC_LENGTH];
        buf[0..4].copy_from_slice(b"XXXX");
        assert_eq!(state_decode(&buf), Err(LanaError::Schema));
    }

    #[test]
    fn persistent_state_round_trip() {
        let state = PersistentState {
            schema_version: STATE_CODEC_SCHEMA,
            state: State { p: 0.5, d_re: 0.3, d_im: 0.1 },
            metadata: PersistentStateMetadata {
                has_timestamp: true,
                has_source: true,
                has_weight: false,
                has_confidence: false,
                timestamp: 1234.5,
                weight: 0.0,
                confidence: 0.0,
                source: Some(Arc::from("test-source")),
            },
            has_provenance: true,
            provenance: PersistentStateProvenance {
                derivation_id: 42,
                input_revision: 7,
                kind: Some(Arc::from("derive")),
                operation: Some(Arc::from("op")),
                captured_inputs: Some(Arc::from("inputs")),
            },
        };
        let encoded = persistent_state_encode(&state).unwrap();
        let decoded = persistent_state_decode(&encoded).unwrap();
        assert_eq!(decoded, state);
    }

    #[test]
    fn persistent_state_decode_rejects_missing_key() {
        let text = br#"{"schema":1,"d_re":0.3,"d_im":0.1,"metadata":{"confidence":null,"source":null,"timestamp":null,"weight":null},"provenance":null}"#;
        assert_eq!(persistent_state_decode(text), Err(LanaError::Schema));
    }
}
