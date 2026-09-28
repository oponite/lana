//! Canonical, self-contained historical dataset evidence records.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lana_bytecode::{Chunk, LanaError};
use lana_vm::{Vm, Value, ValueKind};
use lana_vm::value::JointKind;
use serde::{Deserialize, Serialize};

use crate::dataset_identity::{DatasetIdentity, ExclusionIdentity, NodeIdentity, parse_hex32, source_row, stable_node_id};
use crate::information_codec::{self, JointRow, Real, Tagged};

const MAX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetSnapshot {
    pub schema_version: u32,
    pub format: String,
    pub query_id: String,
    pub source_revision: String,
    pub calculation_version: String,
    pub plan_digest: String,
    pub rows: Vec<Tagged>,
    pub row_ids: Vec<String>,
    pub evidence: SnapshotEvidence,
    pub exclusions: Vec<ExclusionIdentity>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotEvidence {
    pub nodes: Vec<NodeIdentity>,
    pub rows: Vec<RowEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowEvidence {
    pub output_id: String,
    pub derivation_id: String,
    pub operation: String,
    pub input_ids: Vec<String>,
    pub cell_derivations: Vec<(String, String)>,
    pub decisions: Vec<RowDecision>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowDecision {
    pub operation: String,
    pub predicate_value: bool,
    pub predicate_derivation: Option<String>,
}

fn valid_id(id: &str) -> bool { !id.is_empty() && id.len() <= 128 }

fn corruption(error: LanaError) -> LanaError {
    match error { LanaError::Limit | LanaError::Oom => error, _ => LanaError::Corruption }
}

fn validate_tagged(value: &Tagged, depth: usize) -> Result<(), LanaError> {
    if depth > 64 { return Err(LanaError::Limit); }
    match value {
        Tagged::Array { items } => {
            for item in items { validate_tagged(item, depth + 1)?; }
        }
        Tagged::Map { entries } => {
            if entries.windows(2).any(|pair| pair[0].0 >= pair[1].0) { return Err(LanaError::Corruption); }
            for (_, item) in entries { validate_tagged(item, depth + 1)?; }
        }
        Tagged::Definite { .. } => return Err(LanaError::Corruption),
        Tagged::Possibility { support, .. } if support.len() > 1_024 => return Err(LanaError::Limit),
        Tagged::Distribution { rows, .. } if rows.len() > 1_024 => return Err(LanaError::Limit),
        Tagged::FiniteJoint { rows, .. } if rows.len() > 1_024 => return Err(LanaError::Limit),
        _ => {
            let chunk = Chunk::new(5, 0);
            let mut vm = Vm::new(&chunk);
            let live = value.to_dataset_live(&mut vm).map_err(corruption)?;
            if value.snapshot(&live).map_err(corruption)? != *value { return Err(LanaError::Corruption); }
        }
    }
    Ok(())
}

#[derive(Default)]
pub(crate) struct CellEncoder {
    pub(crate) dependencies: HashMap<u64, String>,
    pub(crate) relationships: HashMap<usize, String>,
    active: HashSet<usize>,
}

impl CellEncoder {
    fn dependency(&mut self, live: u64) -> String {
        if let Some(id) = self.dependencies.get(&live) { return id.clone(); }
        let id = (self.dependencies.len() + 1).to_string();
        self.dependencies.insert(live, id.clone());
        id
    }

    fn relationship(&mut self, live: usize) -> String {
        if let Some(id) = self.relationships.get(&live) { return id.clone(); }
        let id = (self.relationships.len() + 1).to_string();
        self.relationships.insert(live, id.clone());
        id
    }

    pub(crate) fn encode(&mut self, value: &Value, depth: usize) -> Result<Tagged, LanaError> {
        if depth > 64 { return Err(LanaError::Limit); }
        if value.reactive.is_some() || value.claim.is_some() || value.planned_effect.is_some() {
            return Err(LanaError::UnsupportedValue);
        }
        let tagged = match &value.kind {
            ValueKind::Array(items) => {
                let identity = Arc::as_ptr(items) as usize;
                if !self.active.insert(identity) { return Err(LanaError::UnsupportedValue); }
                let items = items.lock().unwrap().items().to_vec();
                let result = items.iter().map(|item| self.encode(item, depth + 1)).collect::<Result<Vec<_>, _>>();
                self.active.remove(&identity);
                Tagged::Array { items: result? }
            }
            ValueKind::Map(fields) => {
                let identity = Arc::as_ptr(fields) as usize;
                if !self.active.insert(identity) { return Err(LanaError::UnsupportedValue); }
                let mut fields = fields.lock().unwrap().entries().to_vec();
                fields.sort_by(|a, b| a.key.cmp(&b.key));
                let result = fields.iter().map(|entry| Ok((entry.key.to_string(), self.encode(&entry.value, depth + 1)?)))
                    .collect::<Result<Vec<_>, LanaError>>();
                self.active.remove(&identity);
                Tagged::Map { entries: result? }
            }
            ValueKind::Possibility(law) => {
                if law.values.is_empty() || law.values.len() > 1_024 { return Err(LanaError::Limit); }
                let dependency_id = self.dependency(law.dependency_id);
                let values = law.values.iter().map(|item| Tagged::plain(item, 0)).collect::<Result<Vec<_>, _>>()?;
                if values.iter().any(|item| !item.scalar()) { return Err(LanaError::UnsupportedValue); }
                if let Some(weights) = &law.weights {
                    if weights.len() != values.len() { return Err(LanaError::Corruption); }
                    Tagged::Distribution { dependency_id, rows: values.into_iter().zip(weights)
                        .map(|(item, weight)| (item, Real::bits(*weight))).collect() }
                } else {
                    Tagged::Possibility { dependency_id, support: values }
                }
            }
            ValueKind::Joint(law) if law.kind == JointKind::FiniteLaw => {
                if law.rows.is_empty() || law.rows.len() > 1_024 { return Err(LanaError::Limit); }
                let relationship_id = self.relationship(Arc::as_ptr(law) as usize);
                let rows = law.rows.iter().map(|row| Ok(JointRow {
                    values: row.values.iter().map(|item| Tagged::plain(item, 0)).collect::<Result<_, _>>()?,
                    weight: Real::bits(row.weight),
                })).collect::<Result<Vec<_>, LanaError>>()?;
                let domains = rows[0].values.iter().map(|value| value.domain().to_string()).collect();
                Tagged::FiniteJoint { relationship_id, names: law.names.iter().map(ToString::to_string).collect(),
                    domains, rows }
            }
            _ => Tagged::plain(value, depth)?,
        };
        validate_tagged(&tagged, depth)?;
        Ok(tagged)
    }
}

fn row_decisions(root: &str, nodes: &[NodeIdentity], by_id: &HashMap<&str, usize>,
    remaining: &mut usize) -> Result<Vec<RowDecision>, LanaError> {
    let mut reachable = HashSet::new();
    let mut indices = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        *remaining = remaining.checked_sub(1).ok_or(LanaError::Limit)?;
        if !reachable.insert(id) { continue; }
        let index = *by_id.get(id).ok_or(LanaError::Corruption)?;
        indices.push(index);
        let node = &nodes[index];
        stack.extend(node.input_ids.iter().map(String::as_str));
    }
    indices.sort_unstable();
    let mut decisions = Vec::new();
    for index in indices {
        let node = &nodes[index];
        if node.operation == "filter" && node.details == "dataset_row" {
            if node.reason != "true" { return Err(LanaError::Corruption); }
            decisions.push(RowDecision { operation: "filter".into(), predicate_value: true,
                predicate_derivation: node.input_ids.get(1).cloned() });
        }
    }
    Ok(decisions)
}

fn source_nodes(root: &str, nodes: &[NodeIdentity], by_id: &HashMap<&str, usize>,
    remaining: &mut usize) -> Result<Vec<(String, String)>, LanaError> {
    let mut stack = vec![root];
    let mut sources = Vec::new();
    while let Some(id) = stack.pop() {
        *remaining = remaining.checked_sub(1).ok_or(LanaError::Limit)?;
        let node = &nodes[*by_id.get(id).ok_or(LanaError::Corruption)?];
        if node.details == "source_row" && node.operation == "dataset_source" {
            sources.push(source_row(&node.label)?);
        } else {
            stack.extend(node.input_ids.iter().rev().map(String::as_str));
        }
    }
    if sources.is_empty() { return Err(LanaError::Corruption); }
    Ok(sources)
}

impl DatasetSnapshot {
    pub fn from_value(value: &Value) -> Result<Self, LanaError> {
        if !matches!(value.kind, ValueKind::Map(_)) { return Err(LanaError::Type); }
        let encoded = crate::codec::encode_value(value)?;
        let snapshot: Self = serde_json::from_str(&encoded).map_err(|_| LanaError::Corruption)?;
        snapshot.encode()?;
        Ok(snapshot)
    }

    pub fn row_evidence(&self, output_id: &str) -> Result<serde_json::Value, LanaError> {
        let row = self.evidence.rows.iter().find(|row| row.output_id == output_id)
            .ok_or(LanaError::NotFound)?;
        let by_id = self.evidence.nodes.iter().enumerate()
            .map(|(index, node)| (node.id.as_str(), index)).collect::<HashMap<_, _>>();
        let mut remaining = 5_000_000;
        let sources = source_nodes(&row.derivation_id, &self.evidence.nodes, &by_id, &mut remaining)?;
        let mut selected = HashSet::new();
        let mut stack = vec![row.derivation_id.as_str()];
        stack.extend(row.cell_derivations.iter().map(|(_, id)| id.as_str()));
        while let Some(id) = stack.pop() {
            remaining = remaining.checked_sub(1).ok_or(LanaError::Limit)?;
            if !selected.insert(id) { continue; }
            let node = &self.evidence.nodes[*by_id.get(id).ok_or(LanaError::Corruption)?];
            stack.extend(node.input_ids.iter().map(String::as_str));
        }
        let nodes = self.evidence.nodes.iter().filter(|node| selected.contains(node.id.as_str()))
            .collect::<Vec<_>>();
        let source_rows = sources.into_iter().map(|(source_id, row_id)|
            serde_json::json!({"source_id": source_id, "row_id": row_id})).collect::<Vec<_>>();
        Ok(serde_json::json!({
            "output_id": row.output_id, "derivation_id": row.derivation_id,
            "operation": row.operation, "input_ids": row.input_ids,
            "cell_derivations": row.cell_derivations, "decisions": row.decisions,
            "nodes": nodes, "source_rows": source_rows,
        }))
    }

    pub fn from_run(query_id: &str, calculation_version: &str, plan_digest: &str,
        source_revision: u64, result: &Value, identity: DatasetIdentity) -> Result<Self, LanaError> {
        if !valid_id(query_id) || calculation_version.is_empty() || parse_hex32(plan_digest).is_none() {
            return Err(LanaError::InvalidParameters);
        }
        let ValueKind::Array(rows) = &result.kind else { return Err(LanaError::Type); };
        let rows = rows.lock().unwrap().items().to_vec();
        if rows.len() != identity.row_ids.len() || rows.len() != identity.row_derivations.len()
            || rows.len() != identity.cell_derivations.len() { return Err(LanaError::Schema); }
        let mut encoder = CellEncoder::default();
        let rows = rows.iter().map(|row| encoder.encode(row, 0)).collect::<Result<Vec<_>, _>>()?;
        let by_id = identity.nodes.iter().enumerate().map(|(index, node)| (node.id.as_str(), index)).collect::<HashMap<_, _>>();
        let mut work_remaining = 5_000_000;
        let evidence_rows = identity.row_ids.iter().enumerate().map(|(index, row_id)| {
            let derivation_id = &identity.row_derivations[index];
            let node = &identity.nodes[*by_id.get(derivation_id.as_str()).ok_or(LanaError::Corruption)?];
            Ok(RowEvidence { output_id: row_id.clone(), derivation_id: derivation_id.clone(),
                operation: node.operation.clone(), input_ids: node.input_row_ids.clone(),
                cell_derivations: identity.cell_derivations[index].clone(),
                decisions: row_decisions(derivation_id, &identity.nodes, &by_id, &mut work_remaining)? })
        }).collect::<Result<Vec<_>, LanaError>>()?;
        let snapshot = Self { schema_version: 1, format: "dataset_snapshot_v1".into(),
            query_id: query_id.into(), source_revision: source_revision.to_string(),
            calculation_version: calculation_version.into(), plan_digest: plan_digest.into(),
            rows, row_ids: identity.row_ids, evidence: SnapshotEvidence {
                nodes: identity.nodes, rows: evidence_rows }, exclusions: identity.exclusions };
        snapshot.validate(query_id, source_revision, plan_digest)?;
        Ok(snapshot)
    }

    pub fn encode(&self) -> Result<Vec<u8>, LanaError> {
        let revision = information_codec::revision(&self.source_revision).map_err(corruption)?;
        self.validate(&self.query_id, revision, &self.plan_digest)?;
        self.canonical_bytes()
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, LanaError> {
        let bytes = information_codec::canonical(self)?;
        if bytes.len() > MAX_BYTES { return Err(LanaError::Limit); }
        Ok(bytes)
    }

    pub fn load(bytes: &[u8], query_id: &str, source_revision: u64, plan_digest: &str) -> Result<Self, LanaError> {
        if bytes.len() > MAX_BYTES { return Err(LanaError::Limit); }
        let parsed: serde_json::Value = serde_json::from_slice(bytes).map_err(|_| LanaError::Corruption)?;
        if parsed.get("schema_version").and_then(serde_json::Value::as_u64) != Some(1)
            || parsed.get("format").and_then(serde_json::Value::as_str) != Some("dataset_snapshot_v1") {
            return Err(LanaError::Schema);
        }
        let snapshot: Self = serde_json::from_value(parsed).map_err(|_| LanaError::Corruption)?;
        snapshot.validate(query_id, source_revision, plan_digest)?;
        if snapshot.canonical_bytes()? != bytes { return Err(LanaError::Corruption); }
        Ok(snapshot)
    }

    fn validate(&self, query_id: &str, source_revision: u64, plan_digest: &str) -> Result<(), LanaError> {
        if self.schema_version != 1 || self.format != "dataset_snapshot_v1" { return Err(LanaError::Schema); }
        if self.query_id != query_id || self.source_revision != source_revision.to_string()
            || self.plan_digest != plan_digest { return Err(LanaError::Corruption); }
        if !valid_id(&self.query_id) || self.calculation_version.is_empty()
            || parse_hex32(&self.plan_digest).is_none() { return Err(LanaError::Schema); }
        if self.rows.len() > 100_000 || self.evidence.nodes.len() > 100_000 { return Err(LanaError::Limit); }
        if self.rows.len() != self.row_ids.len() || self.rows.len() != self.evidence.rows.len() {
            return Err(LanaError::Corruption);
        }
        let mut node_positions = HashMap::new();
        for (index, node) in self.evidence.nodes.iter().enumerate() {
            let path = parse_hex32(&node.operator_path).ok_or(LanaError::Corruption)?;
            if node.id != stable_node_id(query_id, plan_digest, &self.source_revision, &path,
                &node.operation, &node.output_row_id, &node.input_row_ids)
                || node.input_ids.len() != node.input_row_ids.len()
                || node.output_row_id.is_empty() || node.operation.is_empty()
                || !matches!(node.kind.as_str(), "evidence" | "assumption" | "operation" | "observation" | "path" | "sample" | "approximation" | "resolution")
                || !matches!(node.exactness.as_str(), "exact" | "sample" | "approximate")
                || !matches!(node.outcome.as_str(), "success" | "unresolved" | "unsupported" | "error") {
                return Err(LanaError::Corruption);
            }
            if node.details == "dataset_row" && (node.kind != "operation" || node.exactness != "exact"
                || node.outcome != "success" || node.label != node.output_row_id
                || (node.operation == "filter" && node.reason != "true")) {
                return Err(LanaError::Corruption);
            }
            if node.details == "source_row" && (node.operation != "dataset_source" || node.kind != "evidence"
                || node.label != node.output_row_id || source_row(&node.label).is_err()) {
                return Err(LanaError::Corruption);
            }
            if node.input_ids.iter().any(|id| !node_positions.contains_key(id.as_str()))
                || node_positions.insert(node.id.as_str(), index).is_some() { return Err(LanaError::Corruption); }
            for (input_id, input_row_id) in node.input_ids.iter().zip(&node.input_row_ids) {
                let input = &self.evidence.nodes[*node_positions.get(input_id.as_str()).ok_or(LanaError::Corruption)?];
                let expected = if matches!(input.details.as_str(), "dataset_row" | "source_row") {
                    input.label.as_str()
                } else {
                    node.output_row_id.as_str()
                };
                if input_row_id != expected { return Err(LanaError::Corruption); }
            }
        }
        let mut row_ids = HashSet::new();
        let mut work_remaining = 5_000_000;
        for (index, (row, evidence)) in self.rows.iter().zip(&self.evidence.rows).enumerate() {
            let row_id = &self.row_ids[index];
            if row_id.is_empty() || !row_ids.insert(row_id.as_str()) || &evidence.output_id != row_id {
                return Err(LanaError::Corruption);
            }
            let root = self.evidence.nodes.get(*node_positions.get(evidence.derivation_id.as_str()).ok_or(LanaError::Corruption)?)
                .ok_or(LanaError::Corruption)?;
            if root.output_row_id != *row_id || root.operation != evidence.operation
                || root.input_row_ids != evidence.input_ids
                || !matches!(root.details.as_str(), "dataset_row" | "source_row") {
                return Err(LanaError::Corruption);
            }
            let Tagged::Map { entries } = row else { return Err(LanaError::Corruption); };
            validate_tagged(row, 0)?;
            if entries.len() != evidence.cell_derivations.len() { return Err(LanaError::Corruption); }
            for ((name, _), (cell_name, id)) in entries.iter().zip(&evidence.cell_derivations) {
                if name != cell_name || !node_positions.contains_key(id.as_str()) { return Err(LanaError::Corruption); }
            }
            if evidence.decisions != row_decisions(&evidence.derivation_id, &self.evidence.nodes, &node_positions, &mut work_remaining)? {
                return Err(LanaError::Corruption);
            }
        }
        let mut exclusion_index = 0;
        while exclusion_index < self.exclusions.len() {
            let exclusion = &self.exclusions[exclusion_index];
            if !node_positions.contains_key(exclusion.input_derivation.as_str())
                || exclusion.predicate_derivation.as_ref().is_some_and(|id| !node_positions.contains_key(id.as_str())) {
                return Err(LanaError::Corruption);
            }
            match exclusion.reason.as_str() {
                "filter_false" if exclusion.operation == "filter" && exclusion.predicate_value == Some(false) => {},
                "limit_excluded" if exclusion.operation == "limit" && exclusion.predicate_value.is_none()
                    && exclusion.predicate_derivation.is_none() => {},
                "no_matching_key" if exclusion.operation == "join" && exclusion.predicate_value.is_none()
                    && exclusion.predicate_derivation.is_none() => {},
                _ => return Err(LanaError::Corruption),
            }
            let sources = source_nodes(&exclusion.input_derivation, &self.evidence.nodes, &node_positions,
                &mut work_remaining)?;
            for (offset, (source_id, row_id)) in sources.iter().enumerate() {
                let saved = self.exclusions.get(exclusion_index + offset).ok_or(LanaError::Corruption)?;
                if &saved.source_id != source_id || &saved.row_id != row_id
                    || saved.operation != exclusion.operation || saved.reason != exclusion.reason
                    || saved.input_derivation != exclusion.input_derivation
                    || saved.predicate_derivation != exclusion.predicate_derivation
                    || saved.predicate_value != exclusion.predicate_value {
                    return Err(LanaError::Corruption);
                }
            }
            exclusion_index += sources.len();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use lana_vm::value::{Array, Map};

    fn finite_snapshot() -> (DatasetSnapshot, Vec<u8>) {
        let chunk = Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        let possibility: Tagged = serde_json::from_str(
            r#"{"tag":"possibility","dependency_id":"9","support":[{"tag":"number","bits":"3ff0000000000000"},{"tag":"number","bits":"4000000000000000"}]}"#,
        ).unwrap();
        let distribution: Tagged = serde_json::from_str(
            r#"{"tag":"distribution","dependency_id":"10","rows":[[{"tag":"bool","value":false},"3fd3333333333333"],[{"tag":"bool","value":true},"3fe6666666666666"]]}"#,
        ).unwrap();
        let joint: Tagged = serde_json::from_str(
            r#"{"tag":"finite_joint","relationship_id":"11","names":["x","y"],"domains":["bool","bool"],"rows":[{"values":[{"tag":"bool","value":false},{"tag":"bool","value":true}],"weight":"3ff0000000000000"}]}"#,
        ).unwrap();
        let mut row = Map::new(&vm.heap(), 4).unwrap();
        row.set(Arc::from("p"), possibility.to_live(&mut vm).unwrap(), false).unwrap();
        row.set(Arc::from("d"), distribution.to_live(&mut vm).unwrap(), false).unwrap();
        row.set(Arc::from("j"), joint.to_live(&mut vm).unwrap(), false).unwrap();
        let nested = Value::array(Arc::new(Mutex::new(Array::from_items(&vm.heap(),
            vec![possibility.to_live(&mut vm).unwrap()]).unwrap())));
        row.set(Arc::from("nested"), nested, false).unwrap();
        let row = vm.dataset_source_row("s", "a", Value::map(Arc::new(Mutex::new(row)))).unwrap();
        let result = Value::array(Arc::new(Mutex::new(Array::from_items(&vm.heap(), vec![row]).unwrap())));
        let digest = "0".repeat(64);
        let identity = DatasetIdentity::from_run("q", &digest, 7, &result, &[]).unwrap();
        let snapshot = DatasetSnapshot::from_run("q", "v1", &digest, 7, &result, identity).unwrap();
        let bytes = snapshot.encode().unwrap();
        (snapshot, bytes)
    }

    #[test]
    fn finite_cells_and_nested_information_reload_exactly() {
        let (snapshot, bytes) = finite_snapshot();
        let loaded = DatasetSnapshot::load(&bytes, "q", 7, &"0".repeat(64)).unwrap();
        assert_eq!(loaded, snapshot);
        assert_eq!(loaded.encode().unwrap(), bytes);
        let Tagged::Map { entries } = &loaded.rows[0] else { panic!("typed row"); };
        assert!(entries.iter().any(|(name, value)| name == "nested"
            && matches!(value, Tagged::Array { items } if matches!(&items[0], Tagged::Possibility { .. }))));
    }

    #[test]
    fn snapshot_loader_rejects_corrupt_schema_values_and_references() {
        let (_, bytes) = finite_snapshot();
        let digest = "0".repeat(64);
        assert_eq!(DatasetSnapshot::load(&bytes, "q", 8, &digest).unwrap_err(), LanaError::Corruption);
        assert_eq!(DatasetSnapshot::load(&bytes, "q", 7, &"1".repeat(64)).unwrap_err(), LanaError::Corruption);
        let mut padded = bytes.clone();
        padded.push(b' ');
        assert_eq!(DatasetSnapshot::load(&padded, "q", 7, &digest).unwrap_err(), LanaError::Corruption);
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        value["schema_version"] = serde_json::json!(2);
        let changed = information_codec::canonical(&value).unwrap();
        assert_eq!(DatasetSnapshot::load(&changed, "q", 7, &digest).unwrap_err(), LanaError::Schema);
        value["schema_version"] = serde_json::json!(1);
        value["evidence"]["rows"][0]["derivation_id"] = serde_json::json!("missing");
        let changed = information_codec::canonical(&value).unwrap();
        assert_eq!(DatasetSnapshot::load(&changed, "q", 7, &digest).unwrap_err(), LanaError::Corruption);
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        value["rows"][0]["entries"][0][1]["rows"][0][1] = serde_json::json!("7ff0000000000000");
        let changed = information_codec::canonical(&value).unwrap();
        assert_eq!(DatasetSnapshot::load(&changed, "q", 7, &digest).unwrap_err(), LanaError::Corruption);
    }
}
