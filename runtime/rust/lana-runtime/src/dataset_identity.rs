//! Stable identities for one evaluated named dataset plan.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lana_bytecode::LanaError;
use lana_vm::derivation::{Derivation, kind_name, exactness_name, outcome_name};
use lana_vm::value::{Value, ValueKind};
use lana_vm::vm::DatasetDecision;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeIdentity {
    pub id: String,
    pub operation: String,
    pub input_ids: Vec<String>,
    pub input_row_ids: Vec<String>,
    pub output_row_id: String,
    pub operator_path: String,
    pub kind: String,
    pub exactness: String,
    pub outcome: String,
    pub reason: String,
    pub label: String,
    pub details: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExclusionIdentity {
    pub source_id: String,
    pub row_id: String,
    pub operation: String,
    pub reason: String,
    pub input_derivation: String,
    pub predicate_derivation: Option<String>,
    pub predicate_value: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetIdentity {
    pub row_ids: Vec<String>,
    pub row_derivations: Vec<String>,
    pub cell_derivations: Vec<Vec<(String, String)>>,
    pub exclusions: Vec<ExclusionIdentity>,
    pub nodes: Vec<NodeIdentity>,
}

struct Frame {
    derivation: Arc<Derivation>,
    output_row_id: String,
    path: [u8; 32],
    next: usize,
}

struct Builder<'a> {
    query_id: &'a str,
    plan_digest: &'a str,
    source_revision: String,
    known: HashMap<usize, String>,
    nodes: Vec<NodeIdentity>,
}

fn hash_parts(parts: &[&[u8]]) -> [u8; 32] {
    let mut hash = crate::sha256::Sha256::new();
    for part in parts {
        hash.update(&(part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    let mut digest = [0; 32];
    hash.finalize(&mut digest);
    digest
}

fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn parse_hex32(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 || !text.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        return None;
    }
    let mut result = [0; 32];
    for (index, slot) in result.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(result)
}

pub(crate) fn stable_node_id(query_id: &str, plan_digest: &str, source_revision: &str,
    path: &[u8; 32], operation: &str, output_row_id: &str, input_row_ids: &[String]) -> String {
    let mut parts: Vec<&[u8]> = vec![
        b"lana.dataset.derivation.v1", query_id.as_bytes(), plan_digest.as_bytes(),
        source_revision.as_bytes(), path, operation.as_bytes(), output_row_id.as_bytes(),
    ];
    parts.extend(input_row_ids.iter().map(|id| id.as_bytes()));
    hex(&hash_parts(&parts))
}

fn child_path(parent: &[u8; 32], index: usize) -> [u8; 32] {
    hash_parts(&[parent, &(index as u64).to_le_bytes()])
}

fn row_path<'a>(derivation: &'a Derivation, default: &'a str) -> &'a str {
    if matches!(derivation.details.as_ref(), "dataset_row" | "source_row") {
        derivation.label.as_ref()
    } else {
        default
    }
}

pub(crate) fn source_row(label: &str) -> Result<(String, String), LanaError> {
    fn field<'a>(bytes: &'a [u8], offset: &mut usize) -> Result<&'a str, LanaError> {
        let start = *offset;
        let colon = bytes[start..].iter().position(|byte| *byte == b':').ok_or(LanaError::Corruption)? + start;
        let length = std::str::from_utf8(&bytes[start..colon]).map_err(|_| LanaError::Corruption)?;
        let length: usize = length.parse().map_err(|_| LanaError::Corruption)?;
        if length.to_string() != std::str::from_utf8(&bytes[start..colon]).map_err(|_| LanaError::Corruption)? {
            return Err(LanaError::Corruption);
        }
        if length == 0 || length > 128 { return Err(LanaError::Corruption); }
        *offset = colon.checked_add(1).and_then(|start| start.checked_add(length)).ok_or(LanaError::Corruption)?;
        let text = bytes.get(colon + 1..*offset).ok_or(LanaError::Corruption)?;
        std::str::from_utf8(text).map_err(|_| LanaError::Corruption)
    }
    let bytes = label.as_bytes();
    let mut offset = 0;
    let source = field(bytes, &mut offset)?.to_string();
    let row = field(bytes, &mut offset)?.to_string();
    if offset != bytes.len() { return Err(LanaError::Corruption); }
    Ok((source, row))
}

fn contributing_sources(root: &Arc<Derivation>) -> Result<Vec<(String, String)>, LanaError> {
    let mut stack = vec![root.clone()];
    let mut sources = Vec::new();
    while let Some(node) = stack.pop() {
        if sources.len().checked_add(stack.len()).is_none_or(|count| count > 100_000) {
            return Err(LanaError::Limit);
        }
        if node.operation.as_ref() == "dataset_source" && node.details.as_ref() == "source_row" {
            sources.push(source_row(&node.label)?);
        } else {
            stack.extend(node.inputs.iter().rev().cloned());
        }
    }
    if sources.is_empty() { return Err(LanaError::UnsupportedValue); }
    Ok(sources)
}

impl Builder<'_> {
    fn visit(&mut self, root: Arc<Derivation>, output_row_id: &str, path: [u8; 32]) -> Result<String, LanaError> {
        let mut stack = vec![Frame { derivation: root.clone(), output_row_id: output_row_id.to_string(), path, next: 0 }];
        while !stack.is_empty() {
            if self.known.len().checked_add(stack.len()).is_none_or(|count| count > 100_000) {
                return Err(LanaError::Limit);
            }
            let top = stack.len() - 1;
            let ptr = Arc::as_ptr(&stack[top].derivation) as usize;
            if self.known.contains_key(&ptr) { stack.pop(); continue; }
            if stack[top].next < stack[top].derivation.inputs.len() {
                let index = stack[top].next;
                stack[top].next += 1;
                let input = stack[top].derivation.inputs[index].clone();
                if !self.known.contains_key(&(Arc::as_ptr(&input) as usize)) {
                    let row_id = row_path(&input, &stack[top].output_row_id).to_string();
                    stack.push(Frame { derivation: input, output_row_id: row_id,
                        path: child_path(&stack[top].path, index), next: 0 });
                }
                continue;
            }
            let frame = stack.pop().unwrap();
            let input_ids = frame.derivation.inputs.iter().map(|input| {
                self.known.get(&(Arc::as_ptr(input) as usize)).cloned().ok_or(LanaError::Corruption)
            }).collect::<Result<Vec<_>, _>>()?;
            let input_row_ids = frame.derivation.inputs.iter()
                .map(|input| row_path(input, &frame.output_row_id)).collect::<Vec<_>>();
            let input_row_ids = input_row_ids.iter().map(ToString::to_string).collect::<Vec<_>>();
            let id = stable_node_id(self.query_id, self.plan_digest, &self.source_revision,
                &frame.path, &frame.derivation.operation, &frame.output_row_id, &input_row_ids);
            self.known.insert(Arc::as_ptr(&frame.derivation) as usize, id.clone());
            self.nodes.push(NodeIdentity { id, operation: frame.derivation.operation.to_string(),
                input_ids, input_row_ids,
                output_row_id: frame.output_row_id, operator_path: hex(&frame.path),
                kind: kind_name(frame.derivation.kind).to_string(),
                exactness: exactness_name(frame.derivation.exactness).to_string(),
                outcome: outcome_name(frame.derivation.outcome).to_string(),
                reason: frame.derivation.reason.to_string(), label: frame.derivation.label.to_string(),
                details: frame.derivation.details.to_string() });
        }
        self.known.get(&(Arc::as_ptr(&root) as usize)).cloned().ok_or(LanaError::Corruption)
    }
}

impl DatasetIdentity {
    pub fn from_run(query_id: &str, plan_digest: &str, source_revision: u64,
        result: &Value, decisions: &[DatasetDecision]) -> Result<Self, LanaError> {
        if query_id.is_empty() || query_id.len() > 128 || plan_digest.len() != 64
            || !plan_digest.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
            return Err(LanaError::InvalidParameters);
        }
        let ValueKind::Array(rows) = &result.kind else { return Err(LanaError::Type); };
        let rows = rows.lock().unwrap().items().to_vec();
        if rows.len() > 100_000 { return Err(LanaError::Limit); }
        let mut builder = Builder { query_id, plan_digest, source_revision: source_revision.to_string(),
            known: HashMap::new(), nodes: Vec::new() };
        let mut seen_rows = HashSet::new();
        let mut row_ids = Vec::with_capacity(rows.len());
        let mut row_derivations = Vec::with_capacity(rows.len());
        let mut cell_derivations = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let ValueKind::Map(map) = &row.kind else { return Err(LanaError::Type); };
            let root = row.derivation.as_ref().ok_or(LanaError::UnsupportedValue)?;
            if !matches!(root.details.as_ref(), "dataset_row" | "source_row") { return Err(LanaError::UnsupportedValue); }
            let row_id = root.label.as_ref();
            if row_id.is_empty() || !seen_rows.insert(row_id.to_string()) { return Err(LanaError::Schema); }
            row_ids.push(row_id.to_string());
            row_derivations.push(builder.visit(root.clone(), row_id,
                hash_parts(&[b"row", &(index as u64).to_le_bytes()]))?);
            let mut fields = map.lock().unwrap().entries().to_vec();
            fields.sort_by(|a, b| a.key.cmp(&b.key));
            let mut cells = Vec::with_capacity(fields.len());
            for entry in fields {
                let cell = entry.value.derivation.as_ref().ok_or(LanaError::UnsupportedValue)?;
                let id = builder.visit(cell.clone(), row_id,
                    hash_parts(&[b"cell", &(index as u64).to_le_bytes(), entry.key.as_bytes()]))?;
                cells.push((entry.key.to_string(), id));
            }
            cell_derivations.push(cells);
        }
        let mut exclusions = Vec::with_capacity(decisions.len());
        for (index, decision) in decisions.iter().enumerate() {
            if !matches!(decision.input.details.as_ref(), "dataset_row" | "source_row") {
                return Err(LanaError::UnsupportedValue);
            }
            let row_id = decision.input.label.as_ref();
            if row_id.is_empty() { return Err(LanaError::Schema); }
            let input_derivation = builder.visit(decision.input.clone(), row_id,
                hash_parts(&[b"exclusion", &(index as u64).to_le_bytes()]))?;
            let predicate_derivation = decision.predicate.as_ref().map(|predicate| builder.visit(predicate.clone(), row_id,
                hash_parts(&[b"predicate", &(index as u64).to_le_bytes()]))).transpose()?;
            for (source_id, source_row_id) in contributing_sources(&decision.input)? {
                exclusions.push(ExclusionIdentity { source_id, row_id: source_row_id,
                    operation: decision.operation.to_string(), reason: decision.reason.to_string(),
                    input_derivation: input_derivation.clone(),
                    predicate_derivation: predicate_derivation.clone(), predicate_value: decision.predicate_value });
            }
        }
        Ok(Self { row_ids, row_derivations, cell_derivations, exclusions, nodes: builder.nodes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use lana_vm::{Vm, value::{Array, Dataset, DatasetOp, Map}};

    fn evaluated(op: DatasetOp, query_id: &str, digest: &str, revision: u64) -> DatasetIdentity {
        let chunk = lana_bytecode::assembler::assemble(
            ".function main 0 4\nHALT\n.function plan 1 4\nHOST_CALL dataset_materialize R0 1 R1\nRETURN R1\n",
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        let mut rows = Vec::new();
        for (id, number) in [("a", 1.0), ("b", 1.0), ("c", 2.0)] {
            let mut map = Map::new(&vm.heap(), 1).unwrap();
            map.set(Arc::from("v"), Value::number(number), false).unwrap();
            rows.push(vm.dataset_source_row("s", id, Value::map(Arc::new(Mutex::new(map)))).unwrap());
        }
        let source = Value::dataset(Arc::new(Dataset {
            op: DatasetOp::Source,
            source: Value::array(Arc::new(Mutex::new(Array::from_items(&vm.heap(), rows).unwrap()))),
            function: 0, columns: Value::null(), key: Value::null(), limit: Value::null(),
            other: Value::null(), aggregate: Value::null(),
        }));
        let plan = match op {
            DatasetOp::Join => Value::dataset(Arc::new(Dataset {
                op, source: source.clone(), other: source, key: Value::string(Arc::from("v")),
                function: 0, columns: Value::null(), limit: Value::null(), aggregate: Value::null(),
            })),
            DatasetOp::Aggregate => {
                let grouped = Value::dataset(Arc::new(Dataset {
                    op: DatasetOp::GroupBy, source, other: Value::null(), key: Value::string(Arc::from("v")),
                    function: 0, columns: Value::null(), limit: Value::null(), aggregate: Value::null(),
                }));
                Value::dataset(Arc::new(Dataset {
                    op, source: grouped, other: Value::null(), key: Value::null(), function: 0,
                    columns: Value::null(), limit: Value::null(),
                    aggregate: Value::array(Arc::new(Mutex::new(Array::from_items(&vm.heap(),
                        vec![Value::string(Arc::from("count"))]).unwrap()))),
                }))
            }
            _ => panic!("unsupported fixture"),
        };
        let result = vm.run_pure_dataset_plan("plan", &[plan]).unwrap();
        let identity = DatasetIdentity::from_run(query_id, digest, revision, &result, vm.dataset_decisions()).unwrap();
        let snapshot = crate::dataset_snapshot::DatasetSnapshot::from_run(
            query_id, "v1", digest, revision, &result, identity.clone()).unwrap();
        let bytes = snapshot.encode().unwrap();
        assert_eq!(crate::dataset_snapshot::DatasetSnapshot::load(&bytes, query_id, revision, digest).unwrap(), snapshot);
        identity
    }

    #[test]
    fn row_paths_and_derivation_ids_match_fresh_reruns() {
        for op in [DatasetOp::Join, DatasetOp::Aggregate] {
            let first = evaluated(op, "q", &"0".repeat(64), 7);
            let second = evaluated(op, "q", &"0".repeat(64), 7);
            assert_eq!(first, second);
            let another_query = evaluated(op, "other", &"0".repeat(64), 7);
            assert_eq!(first.row_ids, another_query.row_ids);
            assert_ne!(first.row_derivations, another_query.row_derivations);
            let another_revision = evaluated(op, "q", &"0".repeat(64), 8);
            assert_eq!(first.row_ids, another_revision.row_ids);
            assert_ne!(first.row_derivations, another_revision.row_derivations);
            let another_digest = evaluated(op, "q", &"1".repeat(64), 7);
            assert_eq!(first.row_ids, another_digest.row_ids);
            assert_ne!(first.row_derivations, another_digest.row_derivations);
            assert_eq!(first.row_ids.len(), if op == DatasetOp::Join { 5 } else { 2 });
            assert!(first.nodes.iter().all(|node| node.input_ids.iter().all(|input|
                first.nodes.iter().position(|candidate| candidate.id == *input).unwrap()
                    < first.nodes.iter().position(|candidate| candidate.id == node.id).unwrap())));
        }
    }

    #[test]
    fn identity_rejects_unattributed_or_duplicate_rows() {
        let chunk = lana_bytecode::assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let mut map = Map::new(&vm.heap(), 1).unwrap();
        map.set(Arc::from("v"), Value::number(1.0), false).unwrap();
        let row = Value::map(Arc::new(Mutex::new(map)));
        let rows = Value::array(Arc::new(Mutex::new(Array::from_items(&vm.heap(), vec![row.clone()]).unwrap())));
        assert_eq!(DatasetIdentity::from_run("q", &"0".repeat(64), 1, &rows, &[]).unwrap_err(), LanaError::UnsupportedValue);
        let traced = vm.dataset_source_row("s", "a", row).unwrap();
        let repeated = Value::array(Arc::new(Mutex::new(Array::from_items(&vm.heap(), vec![traced.clone(), traced]).unwrap())));
        assert_eq!(DatasetIdentity::from_run("q", &"0".repeat(64), 1, &repeated, &[]).unwrap_err(), LanaError::Schema);
    }
}
