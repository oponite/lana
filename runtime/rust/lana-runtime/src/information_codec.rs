//! Exact finite Information snapshots for dataset evidence.
use std::sync::{Arc, Mutex};
use lana_bytecode::LanaError;
use lana_vm::{Vm, Value, ValueKind, State, StateValue};
use lana_vm::value::{Array, Map};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Real { Bits(String), Number(f64) }

impl Real {
    pub fn value(&self) -> Result<f64, LanaError> {
        let value = match self {
            Self::Number(value) => *value,
            Self::Bits(bits) => {
                if bits.len() != 16 || !bits.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                    return Err(LanaError::Schema);
                }
                f64::from_bits(u64::from_str_radix(bits, 16).map_err(|_| LanaError::Schema)?)
            }
        };
        if !value.is_finite() { return Err(LanaError::Schema); }
        Ok(if value == 0.0 { 0.0 } else { value })
    }

    pub(crate) fn bits(value: f64) -> Self {
        Self::Bits(format!("{:016x}", (if value == 0.0 { 0.0 } else { value }).to_bits()))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", rename_all = "snake_case", deny_unknown_fields)]
pub enum Tagged {
    Null,
    Bool { value: bool },
    String { value: String },
    Number { bits: Real },
    Array { items: Vec<Tagged> },
    Map { entries: Vec<(String, Tagged)> },
    State { p: Real, d_re: Real, d_im: Real },
    Definite { value: Box<Tagged> },
    Possibility { dependency_id: String, support: Vec<Tagged> },
    Distribution { dependency_id: String, rows: Vec<(Tagged, Real)> },
    FiniteJoint { relationship_id: String, names: Vec<String>, domains: Vec<String>, rows: Vec<JointRow> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointRow { pub values: Vec<Tagged>, pub weight: Real }

pub fn revision(value: &str) -> Result<u64, LanaError> {
    let number: u64 = value.parse().map_err(|_| LanaError::Schema)?;
    if number.to_string() != value { return Err(LanaError::Schema); }
    Ok(number)
}

pub fn array(vm: &Vm, items: Vec<Value>) -> Result<Value, LanaError> {
    let mut result = Array::new(&vm.heap(), items.len())?;
    for item in items { result.push(item)?; }
    Ok(Value::array(Arc::new(Mutex::new(result))))
}

impl Tagged {
    pub(crate) fn scalar(&self) -> bool {
        matches!(self, Self::Null | Self::Bool { .. } | Self::String { .. } | Self::Number { .. } | Self::State { .. })
    }

    pub(crate) fn domain(&self) -> &'static str {
        match self {
            Self::Null => "null", Self::Bool { .. } => "bool", Self::String { .. } => "string",
            Self::Number { .. } => "number", Self::State { .. } => "state", _ => "unsupported",
        }
    }

    pub fn to_live(&self, vm: &mut Vm) -> Result<Value, LanaError> { self.decode(vm, true, 0) }

    /// Dataset alternatives are aligned worlds: equal outcomes must not merge.
    pub(crate) fn to_dataset_live(&self, vm: &mut Vm) -> Result<Value, LanaError> {
        let (label, values, weights) = match self {
            Self::Possibility { dependency_id, support } => (dependency_id, support.clone(), None),
            Self::Distribution { dependency_id, rows } => (dependency_id,
                rows.iter().map(|(value, _)| value.clone()).collect(),
                Some(rows.iter().map(|(_, weight)| weight.value()).collect::<Result<Vec<_>, _>>()?)),
            _ => return self.to_live(vm),
        };
        revision(label)?;
        if values.is_empty() || values.len() > 1_024 { return Err(LanaError::Limit); }
        if values.iter().any(|value| !value.scalar()) { return Err(LanaError::UnsupportedValue); }
        let values = values.iter().map(|value| value.to_live(vm)).collect::<Result<Vec<_>, _>>()?;
        let worlds = (0..values.len()).map(|index| Value::number(index as f64)).collect::<Vec<_>>();
        let law = if let Some(weights) = weights {
            let pairs = worlds.into_iter().zip(weights).map(|(world, weight)| array(vm, vec![world, Value::number(weight)]))
                .collect::<Result<Vec<_>, _>>()?;
            vm.distribution_build(&pairs)?
        } else { vm.possibility_build(&worlds)? };
        let mut law = vm.into_possibility_payload(law)?;
        law.values = values;
        vm.possibility_value(law)
    }

    fn decode(&self, vm: &mut Vm, information: bool, depth: usize) -> Result<Value, LanaError> {
        if depth > 64 { return Err(LanaError::Limit); }
        match self {
            Self::Null => Ok(Value::null()),
            Self::Bool { value } => Ok(Value::boolean(*value)),
            Self::String { value } => Ok(Value::string(Arc::from(value.as_str()))),
            Self::Number { bits } => Ok(Value::number(bits.value()?)),
            Self::State { p, d_re, d_im } => {
                let state = State { p: p.value()?, d_re: d_re.value()?, d_im: d_im.value()? };
                if !lana_vm::state::state_valid(&state) { return Err(LanaError::Schema); }
                Ok(Value::state(StateValue { state, ..StateValue::default() }))
            }
            Self::Array { items } => {
                let values = items.iter().map(|item| item.decode(vm, false, depth + 1)).collect::<Result<_, _>>()?;
                array(vm, values)
            }
            Self::Map { entries } => {
                if entries.windows(2).any(|pair| pair[0].0 >= pair[1].0) { return Err(LanaError::Schema); }
                let mut result = Map::new(&vm.heap(), entries.len())?;
                for (key, item) in entries {
                    result.set(Arc::from(key.as_str()), item.decode(vm, false, depth + 1)?, false)?;
                }
                Ok(Value::map(Arc::new(Mutex::new(result))))
            }
            Self::Definite { value } if information => value.decode(vm, false, depth + 1),
            Self::Possibility { dependency_id, support } if information => {
                revision(dependency_id)?;
                if support.iter().any(|item| !item.scalar()) { return Err(LanaError::UnsupportedValue); }
                let values = support.iter().map(|item| item.decode(vm, false, depth + 1)).collect::<Result<Vec<_>, _>>()?;
                let live = vm.possibility_build(&values)?;
                if live.values.len() != support.len() { return Err(LanaError::Schema); }
                // Persisted IDs are evidence labels; the VM assigns fresh live identities.
                Ok(Value::possibility(live))
            }
            Self::Distribution { dependency_id, rows } if information => {
                revision(dependency_id)?;
                let mut pairs = Vec::with_capacity(rows.len());
                for (item, weight) in rows {
                    if !item.scalar() { return Err(LanaError::UnsupportedValue); }
                    let value = item.decode(vm, false, depth + 1)?;
                    pairs.push(array(vm, vec![value, Value::number(weight.value()?)])?);
                }
                Ok(Value::possibility(vm.distribution_build(&pairs)?))
            }
            Self::FiniteJoint { relationship_id, names, domains, rows } if information => {
                revision(relationship_id)?;
                if names.is_empty() || names.len() != domains.len() ||
                    names.iter().any(|name| name.is_empty() || name.contains([',', ':', '\0'])) ||
                    names.windows(2).any(|pair| pair[0] >= pair[1]) { return Err(LanaError::Schema); }
                let mut values = Vec::with_capacity(rows.len());
                for row in rows {
                    if row.values.len() != names.len() { return Err(LanaError::Schema); }
                    let mut items = Vec::with_capacity(names.len() + 1);
                    for (item, domain) in row.values.iter().zip(domains) {
                        if !item.scalar() || item.domain() != domain { return Err(LanaError::Schema); }
                        items.push(item.decode(vm, false, depth + 1)?);
                    }
                    items.push(Value::number(row.weight.value()?));
                    values.push(array(vm, items)?);
                }
                let values = array(vm, values)?;
                let live = vm.joint_build_finite_array(&values, &names.join(","))?;
                if live.rows.len() != rows.len() || live.names.iter().map(ToString::to_string).collect::<Vec<_>>() != *names {
                    return Err(LanaError::Schema);
                }
                // Validation must not repeatedly renormalize stored binary64 bits on reload.
                let mut live = vm.into_joint_payload(live)?;
                for (saved, row) in rows.iter().zip(&mut live.rows) { row.weight = saved.weight.value()?; }
                vm.joint_value(live)
            }
            _ => Err(LanaError::UnsupportedValue),
        }
    }

    /// Serialize a materialized value while retaining the snapshot's historical identity.
    pub fn snapshot(&self, live: &Value) -> Result<Self, LanaError> {
        match (self, &live.kind) {
            (Self::Definite { .. }, _) => Ok(Self::Definite { value: Box::new(Self::plain(live, 0)?) }),
            (Self::Possibility { dependency_id, .. }, ValueKind::Possibility(law)) if law.weights.is_none() => {
                Ok(Self::Possibility { dependency_id: dependency_id.clone(),
                    support: law.values.iter().map(|value| Self::plain(value, 0)).collect::<Result<_, _>>()? })
            }
            (Self::Distribution { dependency_id, .. }, ValueKind::Possibility(law)) => {
                let weights = law.weights.as_ref().ok_or(LanaError::Schema)?;
                Ok(Self::Distribution { dependency_id: dependency_id.clone(), rows: law.values.iter().zip(weights)
                    .map(|(value, weight)| Ok((Self::plain(value, 0)?, Real::bits(*weight)))).collect::<Result<_, LanaError>>()? })
            }
            (Self::FiniteJoint { relationship_id, .. }, ValueKind::Joint(law)) => {
                let rows = law.rows.iter().map(|row| Ok(JointRow {
                    values: row.values.iter().map(|value| Self::plain(value, 0)).collect::<Result<_, _>>()?,
                    weight: Real::bits(row.weight),
                })).collect::<Result<Vec<_>, LanaError>>()?;
                let domains = rows.first().ok_or(LanaError::Schema)?.values.iter().map(|item| item.domain().to_string()).collect();
                Ok(Self::FiniteJoint { relationship_id: relationship_id.clone(),
                    names: law.names.iter().map(ToString::to_string).collect(), domains, rows })
            }
            _ => Self::plain(live, 0),
        }
    }

    pub fn plain(value: &Value, depth: usize) -> Result<Self, LanaError> {
        if depth > 64 { return Err(LanaError::Limit); }
        if value.reactive.is_some() || value.claim.is_some() || value.planned_effect.is_some() { return Err(LanaError::UnsupportedValue); }
        match &value.kind {
            ValueKind::Null => Ok(Self::Null),
            ValueKind::Bool(value) => Ok(Self::Bool { value: *value }),
            ValueKind::String(value) => Ok(Self::String { value: value.to_string() }),
            ValueKind::Number(value) if value.is_finite() => Ok(Self::Number { bits: Real::bits(*value) }),
            ValueKind::State(value) if lana_vm::state::state_valid(&value.state) => Ok(Self::State {
                p: Real::bits(value.state.p), d_re: Real::bits(value.state.d_re), d_im: Real::bits(value.state.d_im) }),
            ValueKind::Array(value) => {
                let items = value.lock().unwrap().items().to_vec();
                Ok(Self::Array { items: items.iter().map(|item| Self::plain(item, depth + 1)).collect::<Result<_, _>>()? })
            }
            ValueKind::Map(value) => {
                let entries = value.lock().unwrap().entries().to_vec();
                let mut entries = entries.iter().map(|entry| Ok((entry.key.to_string(), Self::plain(&entry.value, depth + 1)?)))
                    .collect::<Result<Vec<_>, LanaError>>()?;
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                Ok(Self::Map { entries })
            }
            _ => Err(LanaError::UnsupportedValue),
        }
    }
}

/// Canonical JSON differs from serde's default only in control-character escapes.
pub fn canonical(value: &impl Serialize) -> Result<Vec<u8>, LanaError> {
    use serde_json::ser::{CharEscape, Formatter};
    struct Format;
    impl Formatter for Format {
        fn write_char_escape<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, escape: CharEscape) -> std::io::Result<()> {
            let byte = match escape {
                CharEscape::Backspace => 8, CharEscape::FormFeed => 12, CharEscape::LineFeed => 10,
                CharEscape::CarriageReturn => 13, CharEscape::Tab => 9, CharEscape::AsciiControl(byte) => byte,
                _ => return serde_json::ser::CompactFormatter.write_char_escape(writer, escape),
            };
            write!(writer, "\\u{byte:04x}")
        }
    }
    let sorted = serde_json::to_value(value).map_err(|_| LanaError::Schema)?;
    let mut bytes = Vec::new();
    sorted.serialize(&mut serde_json::Serializer::with_formatter(&mut bytes, Format)).map_err(|_| LanaError::Schema)?;
    Ok(bytes)
}

/// Stable typed bytes for a definite dataset key, including nested keys.
pub fn dataset_key(value: &Value) -> Result<Vec<u8>, LanaError> {
    canonical(&Tagged::plain(value, 0).map_err(|_| LanaError::UnsupportedValue)?)
}

/// Encode one finite, definite source-row map without losing number types.
pub fn dataset_source_row(row: &Value) -> Result<Vec<u8>, LanaError> {
    if !matches!(row.kind, ValueKind::Map(_)) { return Err(LanaError::Type); }
    dataset_key(row)
}

/// Reject noncanonical or corrupt stored rows before exposing them to a VM.
pub fn load_dataset_source_row(bytes: &[u8], vm: &mut Vm) -> Result<Value, LanaError> {
    let tagged: Tagged = serde_json::from_slice(bytes).map_err(|_| LanaError::Corruption)?;
    if !matches!(tagged, Tagged::Map { .. }) { return Err(LanaError::Corruption); }
    let row = tagged.to_live(vm).map_err(|error| match error {
        LanaError::Limit | LanaError::Oom => error,
        _ => LanaError::Corruption,
    })?;
    if dataset_source_row(&row).map_err(|_| LanaError::Corruption)? != bytes {
        return Err(LanaError::Corruption);
    }
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_rows_and_keys_keep_typed_canonical_identity() {
        let chunk = lana_bytecode::Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        let mut row = Map::new(&vm.heap(), 2).unwrap();
        row.set(Arc::from("z"), Value::number(-0.0), false).unwrap();
        row.set(Arc::from("a"), Value::string(Arc::from("1")), false).unwrap();
        let row = Value::map(Arc::new(Mutex::new(row)));
        let bytes = dataset_source_row(&row).unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("0000000000000000"));
        assert_eq!(dataset_source_row(&load_dataset_source_row(&bytes, &mut vm).unwrap()).unwrap(), bytes);
        assert_ne!(dataset_key(&Value::number(1.0)).unwrap(), dataset_key(&Value::string(Arc::from("1"))).unwrap());
        assert_eq!(load_dataset_source_row(b"{\"tag\":\"map\",\"entries\":[[\"z\",{\"tag\":\"number\",\"bits\":-0.0}]]}", &mut vm)
            .err(), Some(LanaError::Corruption));
        assert_eq!(dataset_source_row(&Value::number(1.0)).err(), Some(LanaError::Type));
        let cyclic = Value::map(Arc::new(Mutex::new(Map::new(&vm.heap(), 1).unwrap())));
        if let ValueKind::Map(map) = &cyclic.kind { map.lock().unwrap().set(Arc::from("self"), cyclic.clone(), false).unwrap(); }
        assert_eq!(dataset_source_row(&cyclic).err(), Some(LanaError::UnsupportedValue));
    }

    #[test]
    fn finite_snapshots_use_core_validation_and_exact_canonical_bytes() {
        let chunk = lana_bytecode::Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        for input in [
            r#"{"tag":"definite","value":{"tag":"string","value":"猫\n"}}"#,
            r#"{"tag":"possibility","dependency_id":"7","support":[{"tag":"number","bits":-0.0},{"tag":"number","bits":2}]}"#,
            r#"{"tag":"distribution","dependency_id":"8","rows":[[{"tag":"bool","value":false},0.3],[{"tag":"bool","value":true},0.7]]}"#,
            r#"{"tag":"finite_joint","relationship_id":"9","names":["x","y"],"domains":["bool","bool"],"rows":[{"values":[{"tag":"bool","value":false},{"tag":"bool","value":true}],"weight":1}]}"#,
        ] {
            let tagged: Tagged = serde_json::from_str(input).unwrap();
            let live = tagged.to_live(&mut vm).unwrap();
            let snapshot = tagged.snapshot(&live).unwrap();
            let bytes = canonical(&snapshot).unwrap();
            let decoded: Tagged = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(canonical(&decoded.snapshot(&decoded.to_live(&mut vm).unwrap()).unwrap()).unwrap(), bytes);
            assert!(!String::from_utf8(bytes).unwrap().contains("\\n"));
        }
        for input in [
            r#"{"tag":"bool","value":true,"value":false}"#,
            r#"{"tag":"bool","value":true,"extra":0}"#,
        ] { assert!(serde_json::from_str::<Tagged>(input).is_err()); }
        for input in [
            r#"{"tag":"possibility","dependency_id":"7","support":[{"tag":"null"},{"tag":"null"}]}"#,
            r#"{"tag":"distribution","dependency_id":"8","rows":[[{"tag":"null"},0.7]]}"#,
            r#"{"tag":"definite","value":{"tag":"possibility","dependency_id":"1","support":[{"tag":"null"}]}}"#,
            r#"{"tag":"number","bits":"7ff0000000000000"}"#,
            r#"{"tag":"map","entries":[["a",{"tag":"null"}],["a",{"tag":"null"}]]}"#,
        ] { assert!(serde_json::from_str::<Tagged>(input).unwrap().to_live(&mut vm).is_err()); }
    }
}
