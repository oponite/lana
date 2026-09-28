//! Bounded, deterministic search for version-one Boolean rules.

use std::collections::{BTreeMap, HashMap, HashSet};

use lana_bytecode::LanaError;
use lana_vm::{Value, Vm};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

use crate::{codec, data};

const MAX_ROWS: usize = 10_000;
const MAX_FEATURES: usize = 64;
const MAX_CANDIDATES: u64 = 100_000;
const MAX_VISITS: u64 = 5_000_000;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Feature {
    name: String,
    kind: String,
    nullable: bool,
    categories: Vec<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Task {
    id: String,
    feature_schema: Vec<Feature>,
    target_labels: Vec<bool>,
    allowed: Vec<String>,
    #[serde(default)]
    known_facts: BTreeMap<String, Json>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Example {
    id: String,
    features: BTreeMap<String, Json>,
    target: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    max_candidates: Option<u64>,
    max_predicate_visits: Option<u64>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Atom {
    feature: String,
    operator: String,
    constant: Json,
    negated: bool,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Rule {
    clauses: Vec<Vec<Atom>>,
}

fn valid_id(id: &str) -> bool { !id.is_empty() && id.len() <= 128 && !id.contains('\0') }

fn valid_value(value: &Json, feature: &Feature) -> bool {
    if value.is_null() { return feature.nullable; }
    match feature.kind.as_str() {
        "number" => value.as_f64().is_some_and(f64::is_finite),
        "boolean" => value.is_boolean(),
        "category" => value.as_str().is_some_and(|text| feature.categories.iter().any(|item| item == text)),
        _ => false,
    }
}

fn validate(task: &Task, train: &mut [Example], holdout: &mut [Example]) -> Result<(), LanaError> {
    if !valid_id(&task.id) || task.feature_schema.is_empty() || task.feature_schema.len() > MAX_FEATURES
        || task.target_labels != [false, true] || task.allowed.is_empty()
        || train.is_empty() || train.len() > MAX_ROWS || holdout.len() > MAX_ROWS {
        return Err(LanaError::Schema);
    }
    let mut feature_names = HashSet::new();
    for feature in &task.feature_schema {
        if !valid_id(&feature.name) || !feature_names.insert(feature.name.as_str()) { return Err(LanaError::Schema); }
        match feature.kind.as_str() {
            "number" | "boolean" if feature.categories.is_empty() => {},
            "category" if !feature.categories.is_empty() && feature.categories.len() <= 64
                && feature.categories.iter().all(|item| valid_id(item))
                && feature.categories.iter().collect::<HashSet<_>>().len() == feature.categories.len() => {},
            _ => return Err(LanaError::Schema),
        }
    }
    let mut operators = HashSet::new();
    for operator in &task.allowed {
        if !matches!(operator.as_str(), "eq" | "ne" | "lt" | "le" | "gt" | "ge" | "and" | "or" | "not")
            || !operators.insert(operator.as_str()) { return Err(LanaError::Schema); }
    }
    if !task.allowed.iter().any(|op| matches!(op.as_str(), "eq" | "ne" | "lt" | "le" | "gt" | "ge")) {
        return Err(LanaError::Schema);
    }
    for (name, value) in &task.known_facts {
        let feature = task.feature_schema.iter().find(|feature| &feature.name == name).ok_or(LanaError::Schema)?;
        if value.is_null() || !valid_value(value, feature) { return Err(LanaError::Schema); }
    }
    let mut ids = HashSet::new();
    for example in train.iter_mut().chain(holdout.iter_mut()) {
        if !valid_id(&example.id) || !ids.insert(example.id.clone()) { return Err(LanaError::Schema); }
        if example.features.keys().any(|name| !feature_names.contains(name.as_str())) { return Err(LanaError::Schema); }
        for (name, value) in &task.known_facts {
            if example.features.get(name).is_some_and(|existing| existing != value) { return Err(LanaError::Schema); }
            example.features.insert(name.clone(), value.clone());
        }
        for feature in &task.feature_schema {
            let value = example.features.get(&feature.name).ok_or(LanaError::Schema)?;
            if !valid_value(value, feature) { return Err(LanaError::Schema); }
        }
    }
    Ok(())
}

fn constants(feature: &Feature, train: &[Example], legacy: bool) -> Vec<Json> {
    match feature.kind.as_str() {
        "boolean" => vec![json!(false), json!(true)],
        "category" => feature.categories.iter().map(|item| json!(item)).collect(),
        _ => {
            let mut numbers: Vec<f64> = train.iter().filter_map(|row| row.features[&feature.name].as_f64()).collect();
            numbers.sort_by(f64::total_cmp);
            numbers.dedup();
            let mut values = Vec::with_capacity(numbers.len().saturating_mul(2));
            for (index, number) in numbers.iter().enumerate() {
                values.push(json!(number));
                if let Some(next) = numbers.get(index + 1) {
                    let difference = *next - *number;
                    let midpoint = if legacy || difference.is_finite() { *number + difference / 2.0 }
                        else { *number / 2.0 + *next / 2.0 };
                    if midpoint.is_finite() && midpoint > *number && midpoint < *next {
                        values.push(json!(midpoint));
                    }
                }
            }
            values
        }
    }
}

fn atoms(task: &Task, train: &[Example], legacy: bool) -> Result<Vec<Atom>, LanaError> {
    let mut atoms = Vec::new();
    let negation = task.allowed.iter().any(|op| op == "not");
    for feature in &task.feature_schema {
        let values = constants(feature, train, legacy);
        for operator in &task.allowed {
            if !matches!(operator.as_str(), "eq" | "ne" | "lt" | "le" | "gt" | "ge")
                || (feature.kind != "number" && !matches!(operator.as_str(), "eq" | "ne")) { continue; }
            for value in &values {
                if atoms.len() >= MAX_CANDIDATES as usize { return Err(LanaError::Limit); }
                atoms.push(Atom { feature: feature.name.clone(), operator: operator.clone(), constant: value.clone(), negated: false });
                if negation {
                    if atoms.len() >= MAX_CANDIDATES as usize { return Err(LanaError::Limit); }
                    atoms.push(Atom { feature: feature.name.clone(), operator: operator.clone(), constant: value.clone(), negated: true });
                }
            }
        }
    }
    Ok(atoms)
}

fn atom_result(atom: &Atom, features: &BTreeMap<String, Json>) -> Option<bool> {
    let value = features.get(&atom.feature)?;
    if value.is_null() { return None; }
    let result = match (value.as_f64(), atom.constant.as_f64()) {
        (Some(left), Some(right)) => match atom.operator.as_str() {
            "eq" => left == right, "ne" => left != right, "lt" => left < right,
            "le" => left <= right, "gt" => left > right, "ge" => left >= right, _ => return None,
        },
        _ => match atom.operator.as_str() {
            "eq" => value == &atom.constant,
            "ne" => value != &atom.constant,
            _ => return None,
        },
    };
    Some(result ^ atom.negated)
}

fn rule_result(rule: &Rule, features: &BTreeMap<String, Json>, mut visit: impl FnMut() -> Result<(), LanaError>)
    -> Result<Option<bool>, LanaError> {
    let mut unresolved_clause = false;
    for clause in &rule.clauses {
        let mut all = true;
        let mut missing = false;
        for atom in clause {
            visit()?;
            match atom_result(atom, features) {
                Some(result) => all &= result,
                None => missing = true,
            }
        }
        if all && !missing { return Ok(Some(true)); }
        if all && missing { unresolved_clause = true; }
    }
    Ok(if unresolved_clause { None } else { Some(false) })
}

// Enumerate atom multisets in declared atom order; each atom can occur once
// per clause. The callback resolves clause partitions by canonical syntax.
fn combinations(n: usize, count: usize, first: usize, picked: &mut Vec<usize>,
    callback: &mut impl FnMut(&[usize]) -> Result<bool, LanaError>) -> Result<bool, LanaError> {
    if picked.len() == count { return callback(picked); }
    for index in first..n {
        if picked.iter().rev().take_while(|&&prior| prior == index).count() == 2 { continue; }
        picked.push(index);
        if !combinations(n, count, index, picked, callback)? { return Ok(false); }
        picked.pop();
    }
    Ok(true)
}

// Replay only: rules-v1 records keep the original enumeration, including its
// incomplete unequal-clause ordering. New learning always uses rules-v2.
fn legacy_combinations(n: usize, count: usize, first: usize, picked: &mut Vec<usize>,
    callback: &mut impl FnMut(&[usize]) -> Result<bool, LanaError>) -> Result<bool, LanaError> {
    if picked.len() == count { return callback(picked); }
    let remaining = count - picked.len();
    if remaining > n.saturating_sub(first) { return Ok(true); }
    for index in first..=n - remaining {
        picked.push(index);
        if !legacy_combinations(n, count, index + 1, picked, callback)? { return Ok(false); }
        picked.pop();
    }
    Ok(true)
}

struct Search<'a> {
    legacy: bool,
    train: &'a [Example],
    features: &'a [Feature],
    atoms: Vec<Atom>,
    max_candidates: u64,
    max_visits: u64,
    candidate_count: u64,
    visits: u64,
    seen: HashMap<Vec<u8>, u64>,
    rejections: Vec<Json>,
    rejection_bytes: usize,
    best: Option<(Rule, Vec<String>)>,
    exact: bool,
    exhausted_limit: bool,
}

impl Search<'_> {
    fn reject(&mut self, entry: Json) -> Result<(), LanaError> {
        let bytes = serde_json::to_vec(&entry).map_err(|_| LanaError::Schema)?.len();
        self.rejection_bytes = self.rejection_bytes.checked_add(bytes).ok_or(LanaError::Limit)?;
        if self.rejection_bytes > 64 * 1024 * 1024 { return Err(LanaError::Limit); }
        self.rejections.push(entry);
        Ok(())
    }

    fn candidate(&mut self, indices: &[Vec<usize>], vm: &mut Vm) -> Result<bool, LanaError> {
        if self.candidate_count >= self.max_candidates { self.exhausted_limit = true; return Ok(false); }
        self.candidate_count += 1;
        let rule = Rule { clauses: indices.iter().map(|clause| clause.iter().map(|&index| self.atoms[index].clone()).collect()).collect() };
        for clause in rule.clauses.iter().filter(|_| !self.legacy) {
            for feature in self.features {
                let constraints = clause.iter().filter(|atom| atom.feature == feature.name).collect::<Vec<_>>();
                if constraints.len() < 2 { continue; }
                let values = match feature.kind.as_str() {
                    "boolean" => vec![json!(false), json!(true)],
                    "category" => feature.categories.iter().map(|value| json!(value)).collect(),
                    _ => constraints.iter().flat_map(|atom| {
                        let value = atom.constant.as_f64().unwrap();
                        [value.next_down(), value, value.next_up()].into_iter()
                            .filter(|value| value.is_finite()).map(|value| json!(value))
                    }).collect(),
                };
                let mut possible = false;
                for value in values {
                    let input = BTreeMap::from([(feature.name.clone(), value)]);
                    vm.charge_bounded_work(constraints.len() as u64)?;
                    if constraints.iter().all(|atom| atom_result(atom, &input) == Some(true)) {
                        possible = true;
                        break;
                    }
                }
                if !possible {
                    self.reject(json!({"candidate":rule,"reason":"contradictory_clause",
                        "representative_id":format!("candidate-{}", self.candidate_count)}))?;
                    return Ok(true);
                }
            }
        }
        let mut truth = Vec::with_capacity(self.train.len());
        let mut mistakes = Vec::new();
        let mut missing = Vec::new();
        for row in self.train {
            let has_missing = rule.clauses.iter().flatten().any(|atom| atom_result(atom, &row.features).is_none());
            let result = rule_result(&rule, &row.features, || {
                if self.visits >= self.max_visits { return Err(LanaError::Limit); }
                vm.charge_bounded_work(1)?;
                self.visits += 1;
                Ok(())
            });
            let result = match result {
                Err(LanaError::Limit) if self.visits >= self.max_visits => {
                    self.exhausted_limit = true;
                    return Ok(false);
                }
                other => other?,
            };
            match if has_missing { None } else { result } {
                Some(value) => { truth.push(u8::from(value)); if value != row.target { mistakes.push(row.id.clone()); } },
                None => { truth.push(2); missing.push(row.id.clone()); },
            }
        }
        if !missing.is_empty() {
            for id in missing.into_iter().take(if self.legacy { 1 } else { usize::MAX }) {
                self.reject(json!({"candidate":rule,"reason":"missing_feature","representative_id":id}))?;
            }
            return Ok(true);
        }
        if let Some(prior) = self.seen.get(&truth) {
            let prior = *prior;
            self.reject(json!({"candidate":rule,"reason":"equivalent_truth_vector","representative_id":format!("candidate-{prior}")}))?;
            return Ok(true);
        }
        self.seen.insert(truth, self.candidate_count);
        if self.best.as_ref().is_none_or(|(_, prior)| mistakes.len() < prior.len()) {
            self.exact = mistakes.is_empty();
            self.best = Some((rule, mistakes));
        }
        Ok(!self.exact)
    }
}

fn search<'a>(task: &'a Task, train: &'a [Example], max_candidates: u64, max_visits: u64, vm: &mut Vm, legacy: bool)
    -> Result<Search<'a>, LanaError> {
    let atoms = atoms(task, train, legacy)?;
    if atoms.is_empty() { return Err(LanaError::UnsupportedOperation); }
    let atoms_len = atoms.len();
    let mut search = Search { legacy, train, features: &task.feature_schema, atoms, max_candidates, max_visits,
        candidate_count: 0, visits: 0, seen: HashMap::new(), rejections: Vec::new(), rejection_bytes: 0,
        best: None, exact: false, exhausted_limit: false };
    let allow_and = task.allowed.iter().any(|op| op == "and");
    let allow_or = task.allowed.iter().any(|op| op == "or");
    if legacy {
        for total in 1..=if allow_or { 6 } else if allow_and { 3 } else { 1 } {
            if total <= if allow_and { 3 } else { 1 } {
                let mut callback = |one: &[usize]| search.candidate(&[one.to_vec()], vm);
                if !legacy_combinations(atoms_len, total, 0, &mut Vec::new(), &mut callback)? { break; }
            }
            if allow_or {
                let mut stopped = false;
                for first_size in 1..=3.min(total / 2) {
                    let second_size = total - first_size;
                    if second_size > 3 || (!allow_and && (first_size > 1 || second_size > 1)) { continue; }
                    let mut first_callback = |first: &[usize]| {
                        let mut second_callback = |second: &[usize]| {
                            if first >= second { return Ok(true); }
                            search.candidate(&[first.to_vec(), second.to_vec()], vm)
                        };
                        legacy_combinations(atoms_len, second_size, 0, &mut Vec::new(), &mut second_callback)
                    };
                    if !legacy_combinations(atoms_len, first_size, 0, &mut Vec::new(), &mut first_callback)? {
                        stopped = true; break;
                    }
                }
                if stopped { break; }
            }
        }
        return Ok(search);
    }
    let max_clause = if allow_and { 3 } else { 1 };
    for total in 1..=max_clause * if allow_or { 2 } else { 1 } {
        let mut callback = |indices: &[usize]| {
            let mut partitions = std::collections::BTreeSet::new();
            for mask in 0..(1usize << indices.len()) {
                vm.charge_bounded_work(1)?;
                let mut left = Vec::new();
                let mut right = Vec::new();
                for (position, &index) in indices.iter().enumerate() {
                    if mask & (1 << position) == 0 { left.push(index); } else { right.push(index); }
                }
                if left.is_empty() || left.len() > max_clause || right.len() > max_clause
                    || (!allow_or && !right.is_empty())
                    || left.windows(2).any(|pair| pair[0] == pair[1])
                    || right.windows(2).any(|pair| pair[0] == pair[1]) { continue; }
                let mut clauses = vec![left];
                if !right.is_empty() { clauses.push(right); }
                clauses.sort();
                if clauses.len() == 2 && clauses[0] == clauses[1] { continue; }
                partitions.insert(clauses);
            }
            let mut ordered = partitions.into_iter().map(|clauses| {
                let rule = Rule { clauses: clauses.iter().map(|clause|
                    clause.iter().map(|&index| search.atoms[index].clone()).collect()).collect() };
                Ok((crate::information_codec::canonical(&rule)?, clauses))
            }).collect::<Result<Vec<_>, LanaError>>()?;
            ordered.sort_by(|left, right| left.0.cmp(&right.0));
            for (_, clauses) in ordered {
                if !search.candidate(&clauses, vm)? { return Ok(false); }
            }
            Ok(true)
        };
        if !combinations(atoms_len, total, 0, &mut Vec::new(), &mut callback)? { break; }
    }
    Ok(search)
}

pub fn learn(args: &[Value], vm: &mut Vm) -> Result<Value, LanaError> {
    learn_version(args, vm, false)
}

fn learn_version(args: &[Value], vm: &mut Vm, legacy: bool) -> Result<Value, LanaError> {
    if args.len() != 4 { return Err(LanaError::InvalidParameters); }
    let inputs = args.iter().map(|value| {
        let encoded = codec::encode_value(value)?;
        serde_json::from_str::<Json>(&encoded).map_err(|_| LanaError::Schema)
    }).collect::<Result<Vec<_>, _>>()?;
    let task: Task = serde_json::from_value(inputs[0].clone()).map_err(|_| LanaError::Schema)?;
    let mut train: Vec<Example> = serde_json::from_value(inputs[1].clone()).map_err(|_| LanaError::Schema)?;
    let mut holdout: Vec<Example> = serde_json::from_value(inputs[2].clone()).map_err(|_| LanaError::Schema)?;
    let options: Options = serde_json::from_value(inputs[3].clone()).map_err(|_| LanaError::Schema)?;
    let max_candidates = options.max_candidates.unwrap_or(MAX_CANDIDATES);
    let max_visits = options.max_predicate_visits.unwrap_or(MAX_VISITS);
    if max_candidates == 0 || max_candidates > MAX_CANDIDATES || max_visits == 0 || max_visits > MAX_VISITS {
        return Err(LanaError::Schema);
    }
    validate(&task, &mut train, &mut holdout)?;
    let search = search(&task, &train, max_candidates, max_visits, vm, legacy)?;
    let search_status = if search.exhausted_limit { "limit_exhausted" }
        else if search.exact { "exact_found" } else { "exhausted" };
    let (rule, mistakes) = if search.exhausted_limit { (None, Vec::new()) }
        else { search.best.clone().map_or((None, Vec::new()), |(rule, mistakes)| (Some(rule), mistakes)) };
    let mut trace = Vec::with_capacity(holdout.len());
    let mut correct = 0usize;
    let mut per_label_count = [0usize; 2];
    let mut per_label_correct = [0usize; 2];
    if let Some(rule) = &rule {
        for row in &holdout {
            let prediction = rule_result(rule, &row.features, || vm.charge_bounded_work(1))?;
            per_label_count[usize::from(row.target)] += 1;
            if prediction == Some(row.target) {
                correct += 1;
                per_label_correct[usize::from(row.target)] += 1;
            }
            trace.push(json!({"id":row.id,"target":row.target,"prediction":prediction}));
        }
    }
    let enough = holdout.len() >= 20 && per_label_count.iter().all(|count| *count > 0);
    let accuracy = if holdout.is_empty() || rule.is_none() { Json::Null }
        else { json!(correct as f64 / holdout.len() as f64) };
    let status = if search.exhausted_limit { "limit_exhausted" }
        else if rule.is_none() { "candidate" }
        else if !enough { "insufficient_evidence" }
        else if mistakes.is_empty() && accuracy.as_f64().is_some_and(|value| value >= 0.90) { "validated" }
        else { "candidate" };
    let limits = json!({"candidate_evaluations":search.candidate_count,
        "predicate_visits":search.visits,"split_evaluations":0});
    let training_ids: Vec<_> = train.iter().map(|row| row.id.clone()).collect();
    let source_ids: Vec<_> = train.iter().chain(holdout.iter()).map(|row| row.id.clone()).collect();
    let metrics = json!({"count":trace.len(),"correct":correct,"accuracy":accuracy,
        "per_label":[{"label":false,"count":per_label_count[0],"correct":per_label_correct[0]},
            {"label":true,"count":per_label_count[1],"correct":per_label_correct[1]}]});
    let rejections = search.rejections;
    let training_report = json!({"training_ids":training_ids,"mistakes":mistakes,
        "limits_used":limits,"search_status":search_status,"rejections":rejections});
    let result = json!({"schema_version":1,"status":status,"task":task,
        "options":{"max_candidates":max_candidates,"max_predicate_visits":max_visits},
        "train":train,"holdout":holdout,"model_or_rule":rule,
        "training_report":training_report,
        "validation_report":{"holdout_trace":trace,"metrics":metrics},
        "mistakes":mistakes,"source_ids":source_ids,
        "calculation_version":if legacy { "rules-v1" } else { "rules-v2" },"limits_used":limits});
    let encoded = serde_json::to_string(&result).map_err(|_| LanaError::Schema)?;
    if encoded.len() > 64 * 1024 * 1024 { return Err(LanaError::Limit); }
    data::json_parse_with_heap(&encoded, &vm.heap())
}

pub fn predict(args: &[Value], vm: &mut Vm) -> Result<Value, LanaError> {
    if args.len() != 2 { return Err(LanaError::InvalidParameters); }
    let input: Json = serde_json::from_str(&codec::encode_value(&args[0])?).map_err(|_| LanaError::Schema)?;
    let (rule, active_version, task) = if input["format"] == "learned_task_v1" {
        let mut bytes = crate::information_codec::canonical(&input)?;
        bytes.push(b'\n');
        let saved = String::from_utf8(bytes).map_err(|_| LanaError::Schema)?;
        let record = load_record(&saved, vm)?;
        let active = record["active_version"].as_str().ok_or(LanaError::UnsupportedOperation)?;
        let version = record["versions"].as_array().unwrap().iter()
            .find(|version| version["version"] == active).ok_or(LanaError::Corruption)?;
        let decoded = decoded_version(version)?;
        let task: Task = serde_json::from_value(decoded["task"].clone()).map_err(|_| LanaError::Corruption)?;
        let rule: Rule = serde_json::from_value(decoded["model"].clone()).map_err(|_| LanaError::Corruption)?;
        (rule, json!(active), Some(task))
    } else {
        (serde_json::from_value(input).map_err(|_| LanaError::Schema)?, Json::Null, None)
    };
    let mut features: BTreeMap<String, Json> = serde_json::from_str(&codec::encode_value(&args[1])?).map_err(|_| LanaError::Schema)?;
    if let Some(task) = &task {
        for (name, value) in &features {
            let feature = task.feature_schema.iter().find(|feature| feature.name == *name).ok_or(LanaError::Schema)?;
            if !valid_value(value, feature) { return Err(LanaError::Type); }
        }
        for (name, value) in &task.known_facts {
            if features.get(name).is_some_and(|existing| existing != value) { return Err(LanaError::Schema); }
            features.insert(name.clone(), value.clone());
        }
    }
    if rule.clauses.is_empty() || rule.clauses.len() > 2 || features.len() > MAX_FEATURES
        || features.iter().any(|(name, value)| !valid_id(name) ||
            !(value.is_null() || value.is_boolean() || value.is_string() || value.as_f64().is_some_and(f64::is_finite))) {
        return Err(LanaError::Schema);
    }
    let mut matched = Vec::new();
    let mut unresolved = false;
    for (index, clause) in rule.clauses.iter().enumerate() {
        if clause.is_empty() || clause.len() > 3 { return Err(LanaError::Schema); }
        let mut seen = HashSet::new();
        let mut all = true;
        let mut missing = false;
        for atom in clause {
            if !valid_id(&atom.feature)
                || !matches!(atom.operator.as_str(), "eq" | "ne" | "lt" | "le" | "gt" | "ge")
                || !(atom.constant.is_boolean() || atom.constant.is_string()
                    || atom.constant.as_f64().is_some_and(f64::is_finite))
                || (matches!(atom.operator.as_str(), "lt" | "le" | "gt" | "ge") && !atom.constant.is_number())
                || !seen.insert(serde_json::to_string(atom).map_err(|_| LanaError::Schema)?) {
                return Err(LanaError::Schema);
            }
            if let Some(value) = features.get(&atom.feature).filter(|value| !value.is_null()) {
                if value.is_number() != atom.constant.is_number()
                    || value.is_boolean() != atom.constant.is_boolean()
                    || value.is_string() != atom.constant.is_string() {
                    return Err(LanaError::Type);
                }
            }
            vm.charge_bounded_work(1)?;
            match atom_result(atom, &features) {
                Some(value) => all &= value,
                None => missing = true,
            }
        }
        if all && !missing { matched.push(index); }
        if all && missing { unresolved = true; }
    }
    let value = if !matched.is_empty() { Some(true) }
        else if unresolved { None } else { Some(false) };
    let result = json!({"value":value,"matched_clauses":matched,"features":features,
        "active_version":active_version,"unsupported":value.is_none()});
    let encoded = serde_json::to_string(&result).map_err(|_| LanaError::Schema)?;
    data::json_parse_with_heap(&encoded, &vm.heap())
}

/// Validate a complete learning result before it can enter durable state.
pub fn validate_report(value: &Value, vm: &mut Vm) -> Result<Json, LanaError> {
    let encoded = codec::encode_value(value)?;
    if encoded.len() > 64 * 1024 * 1024 { return Err(LanaError::Limit); }
    let report: Json = serde_json::from_str(&encoded).map_err(|_| LanaError::Schema)?;
    let Some(fields) = report.as_object() else { return Err(LanaError::Schema); };
    let keys = ["schema_version", "status", "task", "options", "train", "holdout",
        "model_or_rule", "training_report", "validation_report", "mistakes", "source_ids",
        "calculation_version", "limits_used"];
    if fields.len() != keys.len() || keys.iter().any(|key| !fields.contains_key(*key))
        || report["schema_version"] != 1 || !matches!(report["calculation_version"].as_str(), Some("rules-v1" | "rules-v2")) {
        return Err(LanaError::Schema);
    }
    // Replay the bounded learner: a digest or plausible metrics cannot prove
    // candidate order, rejection reasons, search completion, or the chosen rule.
    let inputs = ["task", "train", "holdout", "options"].iter().map(|key|
        data::json_parse_with_heap(&report[*key].to_string(), &vm.heap()))
        .collect::<Result<Vec<_>, _>>()?;
    let replay = learn_version(&inputs, vm, report["calculation_version"] == "rules-v1")?;
    let expected: Json = serde_json::from_str(&codec::encode_value(&replay)?).map_err(|_| LanaError::Schema)?;
    if report != expected { return Err(LanaError::Schema); }
    Ok(report)
}

pub(crate) fn transform_scalar(value: &mut Json, encode: bool) -> Result<(), LanaError> {
    if encode {
        *value = match value {
            Json::Null => json!({"tag":"null"}),
            Json::Bool(inner) => json!({"tag":"bool","value":inner}),
            Json::String(inner) => json!({"tag":"string","value":inner}),
            Json::Number(inner) => {
                let number = inner.as_f64().filter(|number| number.is_finite()).ok_or(LanaError::Schema)?;
                json!({"tag":"number","bits":format!("{:016x}", (if number == 0.0 { 0.0 } else { number }).to_bits())})
            }
            _ => return Err(LanaError::Schema),
        };
    } else {
        let fields = value.as_object().ok_or(LanaError::Corruption)?;
        *value = match fields.get("tag").and_then(Json::as_str) {
            Some("null") if fields.len() == 1 => Json::Null,
            Some("bool") if fields.len() == 2 => json!(fields.get("value").and_then(Json::as_bool).ok_or(LanaError::Corruption)?),
            Some("string") if fields.len() == 2 => json!(fields.get("value").and_then(Json::as_str).ok_or(LanaError::Corruption)?),
            Some("number") if fields.len() == 2 => {
                let bits = fields.get("bits").and_then(Json::as_str).ok_or(LanaError::Corruption)?;
                if bits.len() != 16 || !bits.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
                    return Err(LanaError::Corruption);
                }
                let number = f64::from_bits(u64::from_str_radix(bits, 16).map_err(|_| LanaError::Corruption)?);
                if !number.is_finite() || number.to_bits() == (-0.0f64).to_bits() { return Err(LanaError::Corruption); }
                json!(number)
            }
            _ => return Err(LanaError::Corruption),
        };
    }
    Ok(())
}

pub(crate) fn transform_examples(rows: &mut Json, encode: bool) -> Result<(), LanaError> {
    for row in rows.as_array_mut().ok_or(LanaError::Schema)? {
        let features = row["features"].as_object_mut().ok_or(LanaError::Schema)?;
        for value in features.values_mut() { transform_scalar(value, encode)?; }
        transform_scalar(&mut row["target"], encode)?;
    }
    Ok(())
}

fn transform_version(version: &mut Json, encode: bool) -> Result<(), LanaError> {
    let facts = version["task"]["known_facts"].as_object_mut().ok_or(LanaError::Schema)?;
    for value in facts.values_mut() { transform_scalar(value, encode)?; }
    for group in ["train", "holdout"] { transform_examples(&mut version[group], encode)?; }
    fn transform_rule(rule: &mut Json, encode: bool) -> Result<(), LanaError> {
        if rule.is_null() { return Ok(()); }
        let clauses = rule["clauses"].as_array_mut().ok_or(LanaError::Schema)?;
        for clause in clauses {
            for atom in clause.as_array_mut().ok_or(LanaError::Schema)? {
                transform_scalar(&mut atom["constant"], encode)?;
            }
        }
        Ok(())
    }
    transform_rule(&mut version["model"], encode)?;
    for rejection in version["report"]["rejections"].as_array_mut().ok_or(LanaError::Schema)? {
        transform_rule(&mut rejection["candidate"], encode)?;
    }
    let accuracy = &mut version["report"]["metrics"]["accuracy"];
    if !accuracy.is_null() {
        if encode {
            let number = accuracy.as_f64().filter(|number| number.is_finite()).ok_or(LanaError::Schema)?;
            *accuracy = json!(format!("{:016x}", number.to_bits()));
        } else {
            let bits = accuracy.as_str().ok_or(LanaError::Corruption)?;
            if bits.len() != 16 || !bits.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
                return Err(LanaError::Corruption);
            }
            let number = f64::from_bits(u64::from_str_radix(bits, 16).map_err(|_| LanaError::Corruption)?);
            if !number.is_finite() { return Err(LanaError::Corruption); }
            *accuracy = json!(number);
        }
    }
    Ok(())
}

pub fn decoded_version(version: &Json) -> Result<Json, LanaError> {
    let mut decoded = version.clone();
    transform_version(&mut decoded, false)?;
    Ok(decoded)
}

fn version_from_report(version_id: &str, parent: Json, report: &Json) -> Result<Json, LanaError> {
    let training = &report["training_report"];
    let validation = &report["validation_report"];
    let version_report = json!({"status":report["status"],
        "training_ids":training["training_ids"],"holdout_trace":validation["holdout_trace"],
        "metrics":validation["metrics"],"mistakes":report["mistakes"],
        "calculation_version":report["calculation_version"],"limits_used":report["limits_used"],
        "search_status":training["search_status"],"rejections":training["rejections"]});
    let mut version = json!({"version":version_id,"parent_version":parent,"task":report["task"],
        "options":report["options"],"train":report["train"],"holdout":report["holdout"],
        "model":report["model_or_rule"],"report":version_report});
    transform_version(&mut version, true)?;
    Ok(version)
}

pub fn seal_record(record: &mut Json, vm: &mut Vm) -> Result<String, LanaError> {
    record.as_object_mut().ok_or(LanaError::Schema)?.remove("digest");
    let body = crate::information_codec::canonical(record)?;
    let digest = crate::sha256::sha256(&body).iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    record["digest"] = json!(digest);
    let mut encoded = crate::information_codec::canonical(record)?;
    encoded.push(b'\n');
    if encoded.len() > 64 * 1024 * 1024 { return Err(LanaError::Limit); }
    let text = String::from_utf8(encoded).map_err(|_| LanaError::Schema)?;
    let loaded = load_record(&text, vm)?;
    if loaded != *record { return Err(LanaError::Corruption); }
    Ok(text)
}

pub fn initial_record(task_id: &str, report: &Json, vm: &mut Vm) -> Result<String, LanaError> {
    if !valid_id(task_id) || report["task"]["id"] != task_id { return Err(LanaError::Schema); }
    let version = version_from_report("1", Json::Null, report)?;
    let mut record = json!({"schema_version":1,"format":"learned_task_v1","kind":"rule",
        "task_id":task_id,"versions":[version],"active_version":
            if report["status"] == "validated" { Json::String("1".into()) } else { Json::Null },
        "receipts":[]});
    seal_record(&mut record, vm)
}

pub fn append_version(record: &mut Json, report: &Json, example_id: &str, payload_digest: &str,
    vm: &mut Vm) -> Result<(String, String), LanaError> {
    let versions = record["versions"].as_array().ok_or(LanaError::Corruption)?;
    let version_id = (versions.len() as u64).checked_add(1).ok_or(LanaError::Limit)?.to_string();
    let parent = versions.last().ok_or(LanaError::Corruption)?["version"].clone();
    record["versions"].as_array_mut().ok_or(LanaError::Corruption)?
        .push(version_from_report(&version_id, parent, report)?);
    if report["status"] == "validated" { record["active_version"] = json!(version_id); }
    let receipts = record["receipts"].as_array_mut().ok_or(LanaError::Corruption)?;
    receipts.push(json!({"counterexample_id":example_id,"payload_digest":payload_digest,"version":version_id}));
    receipts.sort_by(|a, b| a["counterexample_id"].as_str().cmp(&b["counterexample_id"].as_str()));
    let text = seal_record(record, vm)?;
    Ok((version_id, text))
}

pub fn correction_digest(example: &Json, holdout: &Json) -> Result<String, LanaError> {
    let mut tagged_example = json!([example]);
    let mut tagged_holdout = holdout.clone();
    transform_examples(&mut tagged_example, true)?;
    transform_examples(&mut tagged_holdout, true)?;
    let bytes = crate::information_codec::canonical(&json!({"example":tagged_example[0],"new_holdout":tagged_holdout}))?;
    Ok(crate::sha256::sha256(&bytes).iter().map(|byte| format!("{byte:02x}")).collect())
}

pub fn load_record(text: &str, vm: &mut Vm) -> Result<Json, LanaError> {
    if text.len() > 64 * 1024 * 1024 { return Err(LanaError::Limit); }
    let record: Json = serde_json::from_str(text).map_err(|_| LanaError::Corruption)?;
    let Some(fields) = record.as_object() else { return Err(LanaError::Corruption); };
    let expected = ["schema_version", "format", "kind", "task_id", "versions", "active_version", "receipts", "digest"];
    if fields.len() != expected.len() || expected.iter().any(|key| !fields.contains_key(*key))
        || record["schema_version"] != 1 || record["format"] != "learned_task_v1"
        || record["kind"] != "rule" || record["task_id"].as_str().is_none_or(|id| !valid_id(id)) {
        return Err(LanaError::Corruption);
    }
    let canonical = crate::information_codec::canonical(&record).map_err(|_| LanaError::Corruption)?;
    if text.as_bytes() != [canonical.as_slice(), b"\n"].concat() { return Err(LanaError::Corruption); }
    let mut preimage = record.clone();
    preimage.as_object_mut().ok_or(LanaError::Corruption)?.remove("digest");
    let digest = crate::sha256::sha256(&crate::information_codec::canonical(&preimage).map_err(|_| LanaError::Corruption)?)
        .iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    if record["digest"] != digest { return Err(LanaError::Corruption); }
    let versions = record["versions"].as_array().ok_or(LanaError::Corruption)?;
    if versions.is_empty() || versions.len() > 10_000 { return Err(LanaError::Corruption); }
    let mut used_ids = HashSet::new();
    for (index, version) in versions.iter().enumerate() {
        let version_id = (index + 1).to_string();
        let parent = if index == 0 { Json::Null } else { json!(index.to_string()) };
        if version.as_object().is_none_or(|fields| fields.len() != 8)
            || version["version"] != version_id || version["parent_version"] != parent
            || version["task"] != versions[0]["task"] || version["task"]["id"] != record["task_id"]
            || version["options"] != versions[0]["options"] { return Err(LanaError::Corruption); }
        let mut decoded = version.clone();
        transform_version(&mut decoded, false).map_err(|_| LanaError::Corruption)?;
        let version_report = &decoded["report"];
        if version_report.as_object().is_none_or(|fields| fields.len() != 9) { return Err(LanaError::Corruption); }
        let train = version["train"].as_array().ok_or(LanaError::Corruption)?;
        let holdout = version["holdout"].as_array().ok_or(LanaError::Corruption)?;
        if index > 0 {
            let previous = versions[index - 1]["train"].as_array().ok_or(LanaError::Corruption)?;
            if train.len() != previous.len() + 1 || train[..previous.len()] != previous[..] {
                return Err(LanaError::Corruption);
            }
        }
        for row in if index == 0 { &train[..] } else { &train[train.len() - 1..] } {
            if !used_ids.insert(row["id"].as_str().ok_or(LanaError::Corruption)?.to_owned()) {
                return Err(LanaError::Corruption);
            }
        }
        for row in holdout {
            if !used_ids.insert(row["id"].as_str().ok_or(LanaError::Corruption)?.to_owned()) {
                return Err(LanaError::Corruption);
            }
        }
        let source_ids = train.iter().chain(holdout).map(|row| row["id"].clone()).collect::<Vec<_>>();
        let report = json!({"schema_version":1,"status":version_report["status"],
            "task":decoded["task"],"options":decoded["options"],"train":decoded["train"],
            "holdout":decoded["holdout"],"model_or_rule":decoded["model"],
            "training_report":{"training_ids":version_report["training_ids"],
                "mistakes":version_report["mistakes"],"limits_used":version_report["limits_used"],
                "search_status":version_report["search_status"],"rejections":version_report["rejections"]},
            "validation_report":{"holdout_trace":version_report["holdout_trace"],
                "metrics":version_report["metrics"]},"mistakes":version_report["mistakes"],
            "source_ids":source_ids,"calculation_version":version_report["calculation_version"],
            "limits_used":version_report["limits_used"]});
        let saved = data::json_parse_with_heap(&serde_json::to_string(&report).map_err(|_| LanaError::Corruption)?, &vm.heap())?;
        validate_report(&saved, vm).map_err(|_| LanaError::Corruption)?;
    }
    let receipts = record["receipts"].as_array().ok_or(LanaError::Corruption)?;
    if receipts.len() != versions.len() - 1 { return Err(LanaError::Corruption); }
    let mut last_id = "";
    let mut receipt_versions = HashSet::new();
    for receipt in receipts {
        let id = receipt["counterexample_id"].as_str().ok_or(LanaError::Corruption)?;
        let payload = receipt["payload_digest"].as_str().ok_or(LanaError::Corruption)?;
        let version_id = receipt["version"].as_str().ok_or(LanaError::Corruption)?;
        let number = crate::information_codec::revision(version_id).map_err(|_| LanaError::Corruption)? as usize;
        if receipt.as_object().is_none_or(|fields| fields.len() != 3)
            || !valid_id(id) || id <= last_id || payload.len() != 64
            || !payload.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            || number < 2 || number > versions.len() || !receipt_versions.insert(number)
            || versions[number - 1]["train"].as_array().and_then(|train| train.last())
                .is_none_or(|example| example["id"] != id) { return Err(LanaError::Corruption); }
        let decoded = decoded_version(&versions[number - 1]).map_err(|_| LanaError::Corruption)?;
        let example = decoded["train"].as_array().and_then(|train| train.last()).ok_or(LanaError::Corruption)?;
        if correction_digest(example, &decoded["holdout"]).map_err(|_| LanaError::Corruption)? != payload {
            return Err(LanaError::Corruption);
        }
        last_id = id;
    }
    let active = &record["active_version"];
    if !active.is_null() {
        let id = active.as_str().ok_or(LanaError::Corruption)?;
        let number = crate::information_codec::revision(id).map_err(|_| LanaError::Corruption)? as usize;
        if number == 0 || number > versions.len() || versions[number - 1]["report"]["status"] != "validated" {
            return Err(LanaError::Corruption);
        }
    }
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lana_bytecode::Chunk;

    fn report(vm: &mut Vm, missing: bool) -> Value {
        let task = json!({"id":"mixed", "feature_schema": (["a", "b", "c"].map(|name|
            json!({"name":name,"kind":"boolean","nullable":missing,"categories":[]}))),
            "target_labels":[false,true],"allowed":["eq","and","or"]});
        let rows = |prefix: &str, count: usize| (0..count).map(|i| {
            let (a,b,c) = (i & 1 != 0, i & 2 != 0, i & 4 != 0);
            json!({"id":format!("{prefix}{i}"),"features":{
                "a":if missing { Json::Null } else { json!(a) },
                "b":if missing { Json::Null } else { json!(b) },
                "c":if missing { Json::Null } else { json!(c) }},"target":(a && b) || c})
        }).collect::<Vec<_>>();
        let inputs = [task,json!(rows("t",8)),json!(rows("h",24)),json!({})].map(|input|
            data::json_parse_with_heap(&input.to_string(), &vm.heap()).unwrap());
        learn(&inputs, vm).unwrap()
    }

    #[test]
    fn mixed_clause_sizes_replay_and_forged_search_reports_are_rejected() {
        let chunk = Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        let value = report(&mut vm, false);
        let valid = validate_report(&value, &mut vm).unwrap();
        assert_eq!(valid["status"], "validated");
        let clauses = valid["model_or_rule"]["clauses"].as_array().unwrap();
        assert_eq!(clauses.iter().map(|c| c.as_array().unwrap().len()).sum::<usize>(), 3);
        for path in ["counts", "rejections", "rule"] {
            let mut forged = valid.clone();
            match path {
                "counts" => {
                    forged["limits_used"]["candidate_evaluations"] = json!(1);
                    forged["training_report"]["limits_used"] = forged["limits_used"].clone();
                }
                "rejections" => { forged["training_report"]["rejections"] = json!([]); }
                _ => { forged["model_or_rule"]["clauses"].as_array_mut().unwrap().reverse(); }
            }
            let value = data::json_parse_with_heap(&forged.to_string(), &vm.heap()).unwrap();
            assert!(matches!(validate_report(&value, &mut vm), Err(LanaError::Schema)));
        }
        let inputs = ["task", "train", "holdout", "options"].map(|key|
            data::json_parse_with_heap(&valid[key].to_string(), &vm.heap()).unwrap());
        let legacy = learn_version(&inputs, &mut vm, true).unwrap();
        let legacy_report = validate_report(&legacy, &mut vm).unwrap();
        assert_eq!(legacy_report["calculation_version"], "rules-v1");
        let saved = initial_record("mixed", &legacy_report, &mut vm).unwrap();
        assert!(load_record(&saved, &mut vm).is_ok());
        let value = report(&mut vm, true);
        let missing = validate_report(&value, &mut vm).unwrap();
        assert_eq!(missing["status"], "candidate");
        assert_eq!(missing["training_report"]["search_status"], "exhausted");
        assert!(missing["model_or_rule"].is_null());
    }

    #[test]
    fn opposite_extreme_thresholds_retain_the_finite_midpoint() {
        let feature = Feature { name:"x".into(),kind:"number".into(),nullable:false,categories:vec![] };
        let rows = [-f64::MAX, f64::MAX].into_iter().enumerate().map(|(i,x)| Example {
            id:i.to_string(),features:BTreeMap::from([("x".into(),json!(x))]),target:i == 1
        }).collect::<Vec<_>>();
        assert_eq!(constants(&feature, &rows, false), vec![json!(-f64::MAX),json!(0.0),json!(f64::MAX)]);
    }
}
