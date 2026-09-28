//! Immutable source cells. Batch-local labels make retries deterministic; bindings
//! give those labels durable identities without serializing process-local IDs.
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use lana_bytecode::LanaError;
use lana_vm::{Value, ValueKind, Vm};
use lana_vm::value::{JointState, Map};
use lana_vm::derivation::{Derivation, DerivationKind, DerivationExactness, DerivationOutcome};
use serde::{Deserialize, Serialize};
use crate::information_codec::{self, Tagged};
use crate::dataset_snapshot::CellEncoder;

type Bindings = (BTreeMap<String, String>, BTreeMap<String, String>);

#[derive(Default)]
pub(crate) struct Registry {
    dependencies: HashMap<u64, String>,
    // Retain the allocation while its pointer is an in-memory lookup key.
    relationships: HashMap<usize, (Arc<JointState>, String)>,
}

#[derive(Default)]
pub(crate) struct Encoder {
    cells: CellEncoder,
    joints: HashMap<usize, Arc<JointState>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    format: String,
    value: Tagged,
    dependencies: BTreeMap<String, String>,
    relationships: BTreeMap<String, String>,
    evidence: Evidence,
    nodes: Vec<Node>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Evidence { node: Option<usize>, children: Vec<Evidence> }

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Node {
    revision: String, kind: u32, operation: String, inputs: Vec<usize>,
    label: String, function: String, line: u32, exactness: u32,
    details: String, outcome: u32, reason: String,
}

fn save_node(root: &Arc<Derivation>, nodes: &mut Vec<Node>, known: &mut HashMap<usize, usize>) -> Result<usize, LanaError> {
    let mut stack = vec![(root.clone(), false)];
    let mut active = std::collections::HashSet::new();
    while let Some((node, visited)) = stack.pop() {
        let key = Arc::as_ptr(&node) as usize;
        if known.contains_key(&key) { continue; }
        if !visited {
            if !active.insert(key) { return Err(LanaError::UnsupportedValue); }
            if nodes.len() + active.len() > 100_000 { return Err(LanaError::Limit); }
            stack.push((node.clone(), true));
            for input in node.inputs.iter().rev() { stack.push((input.clone(), false)); }
            continue;
        }
        if node.ad_op != -1 || node.ad_a.is_some() || node.ad_b.is_some() { return Err(LanaError::UnsupportedValue); }
        active.remove(&key);
        known.insert(key, nodes.len());
        nodes.push(Node { revision: node.revision.to_string(), kind: node.kind as u32,
            operation: node.operation.to_string(), inputs: node.inputs.iter().map(|input| known[&(Arc::as_ptr(input) as usize)]).collect(),
            label: node.label.to_string(), function: node.function.to_string(), line: node.line,
            exactness: node.exactness as u32, details: node.details.to_string(), outcome: node.outcome as u32, reason: node.reason.to_string() });
    }
    Ok(known[&(Arc::as_ptr(root) as usize)])
}

impl Encoder {
    pub fn row(&mut self, row: &Value) -> Result<Vec<u8>, LanaError> {
        if !matches!(row.kind, ValueKind::Map(_)) { return Err(LanaError::Type); }
        // Preserve the published definite-row encoding and receipt digests.
        if let Ok(bytes) = information_codec::dataset_source_row(row) { return Ok(bytes); }
        let value = self.cells.encode(row, 0)?;
        let mut nodes = Vec::new();
        let evidence = self.evidence(row, false, &mut nodes, &mut HashMap::new(), 0)?;
        information_codec::canonical(&Row { format: "dataset_information_row_v1".into(), value,
            dependencies: BTreeMap::new(), relationships: BTreeMap::new(), evidence, nodes })
    }

    fn evidence(&mut self, value: &Value, captured: bool, nodes: &mut Vec<Node>, known: &mut HashMap<usize, usize>, depth: usize) -> Result<Evidence, LanaError> {
        if depth > 64 { return Err(LanaError::Limit); }
        if value.reactive.is_some() || value.claim.is_some() || value.planned_effect.is_some() { return Err(LanaError::UnsupportedValue); }
        let captured = captured || value.derivation.as_ref().is_some_and(|d| d.operation.as_ref() == "snapshot");
        let children = match &value.kind {
            ValueKind::Array(items) => items.lock().unwrap().items().to_vec(),
            ValueKind::Map(fields) => {
                let mut fields = fields.lock().unwrap().entries().to_vec();
                fields.sort_by(|a,b| a.key.cmp(&b.key));
                fields.into_iter().map(|field| field.value).collect()
            }
            ValueKind::Possibility(_) if !captured => return Err(LanaError::UnsupportedValue),
            ValueKind::Joint(law) => {
                if !captured { return Err(LanaError::UnsupportedValue); }
                self.joints.insert(Arc::as_ptr(law) as usize, law.clone());
                Vec::new()
            }
            _ => Vec::new(),
        };
        Ok(Evidence { node: value.derivation.as_ref().map(|node| save_node(node, nodes, known)).transpose()?,
            children: children.iter().map(|child| self.evidence(child, captured, nodes, known, depth + 1)).collect::<Result<_,_>>()? })
    }

    pub fn bindings(&self, registry: &Registry, source: &str, batch: &str) -> Result<String, LanaError> {
        let id = |kind: &str, label: &str| -> Result<String, LanaError> {
            Ok(String::from_utf8(information_codec::canonical(&(source, batch, kind, label))?).unwrap())
        };
        let dependencies: BTreeMap<String, String> = self.cells.dependencies.iter().map(|(live, label)| Ok((label.clone(),
            registry.dependencies.get(live).cloned().map(Ok).unwrap_or_else(|| id("dependency", label))?))).collect::<Result<_, LanaError>>()?;
        let relationships: BTreeMap<String, String> = self.cells.relationships.iter().map(|(live, label)| Ok((label.clone(),
            registry.relationships.get(live).map(|(_, id)| id.clone()).map(Ok).unwrap_or_else(|| id("joint", label))?))).collect::<Result<_, LanaError>>()?;
        Ok(String::from_utf8(information_codec::canonical(&(dependencies, relationships))?).unwrap())
    }

    pub fn remember(&self, registry: &mut Registry, text: &str) -> Result<(), LanaError> {
        let (dependencies, relationships) = parse_bindings(text)?;
        if dependencies.len() != self.cells.dependencies.len() || relationships.len() != self.cells.relationships.len() { return Err(LanaError::Corruption); }
        // Validate before changing the session's bindings, including retries.
        for (live, label) in &self.cells.dependencies {
            let id = dependencies.get(label).ok_or(LanaError::Corruption)?;
            if registry.dependencies.get(live).is_some_and(|prior| prior != id) { return Err(LanaError::Conflict); }
        }
        for (live, label) in &self.cells.relationships {
            let id = relationships.get(label).ok_or(LanaError::Corruption)?;
            if registry.relationships.get(live).is_some_and(|(_, prior)| prior != id) { return Err(LanaError::Conflict); }
        }
        for (live, label) in &self.cells.dependencies { registry.dependencies.insert(*live, dependencies[label].clone()); }
        for (live, label) in &self.cells.relationships { registry.relationships.insert(*live, (self.joints[live].clone(), relationships[label].clone())); }
        Ok(())
    }
}

fn parse_bindings(text: &str) -> Result<Bindings, LanaError> {
    let bindings: Bindings = serde_json::from_str(text).map_err(|_| LanaError::Corruption)?;
    if information_codec::canonical(&bindings)? != text.as_bytes() { return Err(LanaError::Corruption); }
    for (kind, ids) in [("dependency", &bindings.0), ("joint", &bindings.1)] {
        for (label, id) in ids {
            if information_codec::revision(label)? == 0 { return Err(LanaError::Corruption); }
            let (source, batch, saved_kind, ordinal): (String, String, String, String) = serde_json::from_str(id).map_err(|_| LanaError::Corruption)?;
            if source.is_empty() || source.len() > 128 || batch.is_empty() || batch.len() > 128 || saved_kind != kind
                || information_codec::revision(&ordinal)? == 0
                || information_codec::canonical(&(source, batch, saved_kind, ordinal))? != id.as_bytes() { return Err(LanaError::Corruption); }
        }
    }
    Ok(bindings)
}

pub(crate) fn bind(bytes: &str, text: &str) -> Result<String, LanaError> {
    if serde_json::from_str::<Tagged>(bytes).is_ok() { return Ok(bytes.to_string()); }
    let mut row: Row = serde_json::from_str(bytes).map_err(|_| LanaError::Corruption)?;
    let (dependencies, relationships) = parse_bindings(text)?;
    fn visit(value: &Tagged, dependencies: &BTreeMap<String,String>, relationships: &BTreeMap<String,String>, row_deps: &mut BTreeMap<String,String>, row_joints: &mut BTreeMap<String,String>) -> Result<(), LanaError> {
        match value {
            Tagged::Array { items } => for item in items { visit(item, dependencies, relationships, row_deps, row_joints)?; },
            Tagged::Map { entries } => for (_, item) in entries { visit(item, dependencies, relationships, row_deps, row_joints)?; },
            Tagged::Possibility { dependency_id, .. } | Tagged::Distribution { dependency_id, .. } => { row_deps.insert(dependency_id.clone(), dependencies.get(dependency_id).ok_or(LanaError::Corruption)?.clone()); },
            Tagged::FiniteJoint { relationship_id, .. } => { row_joints.insert(relationship_id.clone(), relationships.get(relationship_id).ok_or(LanaError::Corruption)?.clone()); },
            _ => {},
        }
        Ok(())
    }
    visit(&row.value, &dependencies, &relationships, &mut row.dependencies, &mut row.relationships)?;
    Ok(String::from_utf8(information_codec::canonical(&row)?).unwrap())
}

#[derive(Default)]
pub(crate) struct Decoder {
    dependencies: HashMap<String, (u64, usize, Option<Vec<f64>>)>,
    relationships: HashMap<String, (Tagged, Value)>,
}

impl Decoder {
    pub fn row(&mut self, bytes: &[u8], vm: &mut Vm) -> Result<Value, LanaError> {
        if bytes.len() > 64 * 1024 * 1024 { return Err(LanaError::Limit); }
        if serde_json::from_slice::<Tagged>(bytes).is_ok() { return information_codec::load_dataset_source_row(bytes, vm); }
        let row: Row = serde_json::from_slice(bytes).map_err(|_| LanaError::Corruption)?;
        if row.format != "dataset_information_row_v1" { return Err(LanaError::Schema); }
        if !matches!(row.value, Tagged::Map { .. }) || row.nodes.len() > 100_000 || information_codec::canonical(&row)? != bytes { return Err(LanaError::Corruption); }
        let bindings = String::from_utf8(information_codec::canonical(&(&row.dependencies, &row.relationships))?).unwrap();
        parse_bindings(&bindings)?;
        // Rebinding also detects missing and extraneous relationship labels.
        let mut normalized: Row = serde_json::from_slice(bytes).unwrap();
        normalized.dependencies.clear(); normalized.relationships.clear();
        if bind(&String::from_utf8(information_codec::canonical(&normalized)?).unwrap(), &bindings)?.as_bytes() != bytes { return Err(LanaError::Corruption); }
        fn validate_evidence(value: &Tagged, evidence: &Evidence, row: &Row, captured: bool, used: &mut std::collections::HashSet<usize>) -> Result<(), LanaError> {
            let captured = captured || evidence.node.is_some_and(|index| row.nodes.get(index).is_some_and(|node| node.operation == "snapshot"));
            let mut stack: Vec<usize> = evidence.node.into_iter().collect();
            while let Some(index) = stack.pop() {
                let node = row.nodes.get(index).ok_or(LanaError::Corruption)?;
                if used.insert(index) { stack.extend(&node.inputs); }
            }
            let children: Vec<&Tagged> = match value {
                Tagged::Array { items } => items.iter().collect(),
                Tagged::Map { entries } => entries.iter().map(|(_, item)| item).collect(),
                Tagged::Possibility { .. } | Tagged::Distribution { .. } | Tagged::FiniteJoint { .. } if !captured => return Err(LanaError::Corruption),
                _ => Vec::new(),
            };
            if children.len() != evidence.children.len() { return Err(LanaError::Corruption); }
            for (child, evidence) in children.iter().zip(&evidence.children) { validate_evidence(child, evidence, row, captured, used)?; }
            Ok(())
        }
        let mut used = std::collections::HashSet::new();
        validate_evidence(&row.value, &row.evidence, &row, false, &mut used)?;
        if used.len() != row.nodes.len() { return Err(LanaError::Corruption); }
        let mut nodes = Vec::with_capacity(row.nodes.len());
        for node in &row.nodes {
            let kind = match node.kind { 0 => DerivationKind::Evidence, 1 => DerivationKind::Assumption, 2 => DerivationKind::Operation, 3 => DerivationKind::Observation, 4 => DerivationKind::Path, 5 => DerivationKind::Sample, 6 => DerivationKind::Approximation, 7 => DerivationKind::Resolution, _ => return Err(LanaError::Corruption) };
            let exactness = match node.exactness { 0 => DerivationExactness::Exact, 1 => DerivationExactness::Sample, 2 => DerivationExactness::Approximate, _ => return Err(LanaError::Corruption) };
            let outcome = match node.outcome { 0 => DerivationOutcome::Success, 1 => DerivationOutcome::Unresolved, 2 => DerivationOutcome::Unsupported, 3 => DerivationOutcome::Error, _ => return Err(LanaError::Corruption) };
            let inputs = node.inputs.iter().map(|index| nodes.get(*index).cloned().ok_or(LanaError::Corruption)).collect::<Result<_,_>>()?;
            nodes.push(vm.import_derivation(Derivation { task_lineage: 0, local_sequence: 0, revision: information_codec::revision(&node.revision)?, kind,
                operation: Arc::from(node.operation.as_str()), inputs, label: Arc::from(node.label.as_str()), function: Arc::from(node.function.as_str()), line: node.line,
                exactness, details: Arc::from(node.details.as_str()), outcome, reason: Arc::from(node.reason.as_str()),
                ad_op: -1, ad_a: None, ad_b: None, ad_a_deriv: None, ad_b_deriv: None, ad_grad: Arc::new(Mutex::new(None)), ad_axis: 0 })?);
        }
        self.value(&row.value, &row.evidence, &row, &nodes, vm, 0)
    }

    fn value(&mut self, tagged: &Tagged, evidence: &Evidence, row: &Row, nodes: &[Arc<Derivation>], vm: &mut Vm, depth: usize) -> Result<Value, LanaError> {
        if depth > 64 { return Err(LanaError::Limit); }
        let mut value = match tagged {
            Tagged::Array { items } => {
                if items.len() != evidence.children.len() { return Err(LanaError::Corruption); }
                let items = items.iter().zip(&evidence.children).map(|(item, evidence)| self.value(item, evidence, row, nodes, vm, depth+1)).collect::<Result<_,_>>()?;
                information_codec::array(vm, items)?
            }
            Tagged::Map { entries } => {
                if entries.len() != evidence.children.len() || entries.windows(2).any(|p| p[0].0 >= p[1].0) { return Err(LanaError::Corruption); }
                let mut map = Map::new(&vm.heap(), entries.len())?;
                for ((key, item), evidence) in entries.iter().zip(&evidence.children) { map.set(Arc::from(key.as_str()), self.value(item, evidence, row, nodes, vm, depth+1)?, false)?; }
                Value::map(Arc::new(Mutex::new(map)))
            }
            _ => {
                if !evidence.children.is_empty() { return Err(LanaError::Corruption); }
                let mut value = tagged.to_dataset_live(vm).map_err(|error| match error {
                    LanaError::Limit | LanaError::Oom => error, _ => LanaError::Corruption,
                })?;
                if tagged.snapshot(&value)? != *tagged { return Err(LanaError::Corruption); }
                match (tagged, &mut value.kind) {
                    (Tagged::Possibility { dependency_id, .. } | Tagged::Distribution { dependency_id, .. }, ValueKind::Possibility(law)) => {
                        if law.values.len() > 1_024 { return Err(LanaError::Limit); }
                        let id = row.dependencies.get(dependency_id).ok_or(LanaError::Corruption)?;
                        if let Some((live, count, weights)) = self.dependencies.get(id) {
                            if *count != law.values.len() || *weights != law.weights { return Err(LanaError::UnsupportedOperation); }
                            Arc::make_mut(law).dependency_id = *live;
                        } else { self.dependencies.insert(id.clone(), (law.dependency_id, law.values.len(), law.weights.clone())); }
                    }
                    (Tagged::FiniteJoint { relationship_id, rows, .. }, _) => {
                        if rows.len() > 1_024 { return Err(LanaError::Limit); }
                        let id = row.relationships.get(relationship_id).ok_or(LanaError::Corruption)?;
                        let mut canonical = tagged.clone();
                        if let Tagged::FiniteJoint { relationship_id, .. } = &mut canonical { *relationship_id = "1".into(); }
                        if let Some((prior, live)) = self.relationships.get(id) {
                            if *prior != canonical { return Err(LanaError::Corruption); }
                            value = live.clone();
                        } else { self.relationships.insert(id.clone(), (canonical, value.clone())); }
                    }
                    _ => {},
                }
                value
            }
        };
        value.derivation = evidence.node.map(|index| nodes.get(index).cloned().ok_or(LanaError::Corruption)).transpose()?;
        match &value.kind {
            ValueKind::Array(items) => items.lock().unwrap().freeze(),
            ValueKind::Map(fields) => fields.lock().unwrap().freeze(),
            _ => {},
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(vm: &Vm, value: Value) -> Value {
        let mut row = Map::new(&vm.heap(), 1).unwrap();
        row.set(Arc::from("value"), value, false).unwrap();
        Value::map(Arc::new(Mutex::new(row)))
    }
    fn cell(row: &Value) -> Value {
        let ValueKind::Map(row) = &row.kind else { panic!() };
        row.lock().unwrap().get("value").unwrap().clone()
    }
    fn dependency(row: &Value) -> u64 {
        let ValueKind::Possibility(law) = cell(row).kind else { panic!() };
        law.dependency_id
    }

    #[test]
    fn dataset_captures_reopen_with_shared_identity_provenance_and_strict_validation() {
        let chunk = lana_bytecode::Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        let law = Value::possibility(vm.possibility_build(&[Value::number(1.), Value::number(2.)]).unwrap());
        assert_eq!(Encoder::default().row(&row(&vm, law.clone())).unwrap_err(), LanaError::UnsupportedValue);
        let captured = vm.information_snapshot(&law).unwrap();
        let mut encoder = Encoder::default();
        let first = String::from_utf8(encoder.row(&row(&vm, captured.clone())).unwrap()).unwrap();
        let mut registry = Registry::default();
        let bindings = encoder.bindings(&registry, "source", "batch1").unwrap();
        let first = bind(&first, &bindings).unwrap();
        encoder.remember(&mut registry, &bindings).unwrap();
        let mut second_encoder = Encoder::default();
        let second = String::from_utf8(second_encoder.row(&row(&vm, captured)).unwrap()).unwrap();
        let second = bind(&second, &second_encoder.bindings(&registry, "other_source", "batch2").unwrap()).unwrap();
        let independent = Value::possibility(vm.possibility_build(&[Value::number(1.), Value::number(2.)]).unwrap());
        let independent = vm.information_snapshot(&independent).unwrap();
        let mut third_encoder = Encoder::default();
        let third = String::from_utf8(third_encoder.row(&row(&vm, independent)).unwrap()).unwrap();
        let third = bind(&third, &third_encoder.bindings(&registry, "source", "batch3").unwrap()).unwrap();
        let mut reopened = Vm::new(&chunk);
        let mut decoder = Decoder::default();
        let a = decoder.row(first.as_bytes(), &mut reopened).unwrap();
        let b = decoder.row(second.as_bytes(), &mut reopened).unwrap();
        let c = decoder.row(third.as_bytes(), &mut reopened).unwrap();
        assert_eq!(dependency(&a), dependency(&b));
        assert_ne!(dependency(&a), dependency(&c));
        assert_eq!(cell(&a).derivation.unwrap().operation.as_ref(), "snapshot");
        if let ValueKind::Map(map) = a.kind { assert_eq!(map.lock().unwrap().set(Arc::from("new"), Value::null(), false), Err(LanaError::UnsupportedOperation)); }
        let mut repeated = vm.information_snapshot(&law).unwrap();
        if let ValueKind::Possibility(law) = &mut repeated.kind { Arc::make_mut(law).values[1] = Value::number(1.); }
        let mut repeated_encoder = Encoder::default();
        let repeated = String::from_utf8(repeated_encoder.row(&row(&vm, repeated)).unwrap()).unwrap();
        let repeated = bind(&repeated, &repeated_encoder.bindings(&registry, "source", "repeated").unwrap()).unwrap();
        let repeated = decoder.row(repeated.as_bytes(), &mut reopened).unwrap();
        assert_eq!(dependency(&repeated), dependency(&b));
        if let ValueKind::Possibility(law) = cell(&repeated).kind { assert_eq!(law.values.len(), 2); }
        let pair = information_codec::array(&vm, vec![Value::number(1.), Value::number(2.), Value::number(1.)]).unwrap();
        let pairs = information_codec::array(&vm, vec![pair]).unwrap();
        let joint = Value::joint(vm.joint_build_finite_array(&pairs, "x,y").unwrap());
        let joint = vm.information_snapshot(&joint).unwrap();
        let nested = information_codec::array(&vm, vec![joint.clone(), joint]).unwrap();
        let mut joint_encoder = Encoder::default();
        let joint = String::from_utf8(joint_encoder.row(&row(&vm, nested)).unwrap()).unwrap();
        let joint = bind(&joint, &joint_encoder.bindings(&registry, "source", "joint").unwrap()).unwrap();
        let joint = decoder.row(joint.as_bytes(), &mut reopened).unwrap();
        let ValueKind::Array(nested) = cell(&joint).kind else { panic!() };
        let nested = nested.lock().unwrap();
        let (ValueKind::Joint(a), ValueKind::Joint(b)) = (&nested.items()[0].kind, &nested.items()[1].kind) else { panic!() };
        assert!(Arc::ptr_eq(a, b));
        let mut corrupt: serde_json::Value = serde_json::from_str(&first).unwrap();
        corrupt["nodes"][0]["inputs"] = serde_json::json!([0]);
        assert!(Decoder::default().row(&information_codec::canonical(&corrupt).unwrap(), &mut reopened).is_err());
        let mut corrupt: serde_json::Value = serde_json::from_str(&first).unwrap();
        corrupt["dependencies"] = serde_json::json!({});
        assert!(Decoder::default().row(&information_codec::canonical(&corrupt).unwrap(), &mut reopened).is_err());
        let mut corrupt: serde_json::Value = serde_json::from_str(&first).unwrap();
        corrupt["value"]["entries"][0][1]["support"][0]["bits"] = serde_json::json!("7ff0000000000000");
        assert!(Decoder::default().row(&information_codec::canonical(&corrupt).unwrap(), &mut reopened).is_err());
    }
}
