//! Bounded CART models over definite, declared features.

use std::collections::{BTreeMap, HashSet};

use lana_bytecode::LanaError;
use lana_vm::{Rng, Value, Vm};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

use crate::{codec, data, rules};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Feature {
    name: String,
    kind: String,
    nullable: bool,
    categories: Vec<String>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Task {
    id: String,
    feature_schema: Vec<Feature>,
    problem: String,
    labels: Vec<Json>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Example {
    id: String,
    features: BTreeMap<String, Json>,
    target: Json,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    family: Option<String>,
    seed: Option<u64>,
    max_depth: Option<usize>,
    min_leaf: Option<usize>,
    ensemble_size: Option<usize>,
}

#[derive(Clone)]
struct Settings {
    family: String,
    seed: u64,
    max_depth: usize,
    min_leaf: usize,
    ensemble_size: usize,
}

#[derive(Clone)]
enum Objective {
    Classification { labels: Vec<Json> },
    Regression { values: Vec<f64> },
    Boosted { gradients: Vec<f64>, hessians: Vec<f64> },
}

fn valid_id(value: &str) -> bool { !value.is_empty() && value.len() <= 128 && !value.contains('\0') }

fn real(value: &Json) -> Result<f64, LanaError> {
    value.as_f64().filter(|number| number.is_finite()).ok_or(LanaError::Schema)
}

fn bits(value: f64) -> Result<String, LanaError> {
    if !value.is_finite() { return Err(LanaError::Schema); }
    Ok(format!("{:016x}", (if value == 0.0 { 0.0 } else { value }).to_bits()))
}

fn from_bits(value: &Json) -> Result<f64, LanaError> {
    let text = value.as_str().ok_or(LanaError::Schema)?;
    if text.len() != 16 || !text.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        return Err(LanaError::Schema);
    }
    let result = f64::from_bits(u64::from_str_radix(text, 16).map_err(|_| LanaError::Schema)?);
    if !result.is_finite() || result.to_bits() == (-0.0f64).to_bits() { return Err(LanaError::Schema); }
    Ok(result)
}

fn input(value: &Value) -> Result<Json, LanaError> {
    serde_json::from_str(&codec::encode_value(value)?).map_err(|_| LanaError::Schema)
}

fn validate_schema(schema: &[Feature]) -> Result<HashSet<&str>, LanaError> {
    if schema.is_empty() || schema.len() > 64 { return Err(LanaError::Schema); }
    let mut names = HashSet::new();
    for feature in schema {
        if !valid_id(&feature.name) || !names.insert(feature.name.as_str()) { return Err(LanaError::Schema); }
        match feature.kind.as_str() {
            "number" | "boolean" if feature.categories.is_empty() => {},
            "category" if !feature.categories.is_empty() && feature.categories.len() <= 64
                && feature.categories.iter().all(|value| valid_id(value))
                && feature.categories.iter().collect::<HashSet<_>>().len() == feature.categories.len() => {},
            _ => return Err(LanaError::Schema),
        }
    }
    Ok(names)
}

fn valid_feature(value: &Json, feature: &Feature) -> bool {
    if value.is_null() { return feature.nullable; }
    match feature.kind.as_str() {
        "number" => value.as_f64().is_some_and(f64::is_finite),
        "boolean" => value.is_boolean(),
        "category" => value.as_str().is_some_and(|text| feature.categories.iter().any(|item| item == text)),
        _ => false,
    }
}

fn validate(task: &Task, train: &[Example], holdout: &[Example], options: &Options) -> Result<Settings, LanaError> {
    if !valid_id(&task.id) || task.feature_schema.is_empty() || task.feature_schema.len() > 64
        || train.is_empty() || train.len() > 10_000 || holdout.len() > 10_000 { return Err(LanaError::Schema); }
    let names = validate_schema(&task.feature_schema)?;
    match task.problem.as_str() {
        "classification" if (2..=128).contains(&task.labels.len()) => {
            for (index, label) in task.labels.iter().enumerate() {
                if !(label.is_boolean() || label.as_str().is_some_and(valid_id))
                    || task.labels[..index].contains(label) { return Err(LanaError::Schema); }
            }
        }
        "regression" if task.labels.is_empty() => {},
        _ => return Err(LanaError::Schema),
    }
    let mut ids = HashSet::new();
    for row in train.iter().chain(holdout) {
        if !valid_id(&row.id) || !ids.insert(row.id.as_str())
            || row.features.keys().any(|name| !names.contains(name.as_str())) { return Err(LanaError::Schema); }
        for feature in &task.feature_schema {
            let value = row.features.get(&feature.name).ok_or(LanaError::Schema)?;
            let valid = valid_feature(value, feature);
            if !valid { return Err(LanaError::Schema); }
        }
        if task.problem == "classification" {
            if !task.labels.contains(&row.target) { return Err(LanaError::Schema); }
        } else { real(&row.target)?; }
    }
    let family = options.family.as_deref().unwrap_or("tree");
    if !matches!(family, "tree" | "forest" | "boosted") { return Err(LanaError::Schema); }
    let ensemble_size = options.ensemble_size.unwrap_or(if family == "tree" { 1 } else { 25 });
    let max_depth = options.max_depth.unwrap_or(6);
    let min_leaf = options.min_leaf.unwrap_or(2);
    if !(1..=16).contains(&max_depth) || !(1..=100).contains(&min_leaf)
        || !(1..=100).contains(&ensemble_size) || (family == "tree" && ensemble_size != 1) { return Err(LanaError::Schema); }
    if family == "boosted" && task.problem == "classification" && task.labels.len() != 2 {
        return Err(LanaError::UnsupportedOperation);
    }
    Ok(Settings { family: family.into(), seed: options.seed.unwrap_or(0), max_depth, min_leaf, ensemble_size })
}

fn label_index(labels: &[Json], value: &Json) -> usize {
    labels.iter().position(|label| label == value).unwrap()
}

fn objective_value(objective: &Objective, rows: &[usize], train: &[Example]) -> Result<f64, LanaError> {
    if rows.is_empty() { return Err(LanaError::Schema); }
    let count = rows.len() as f64;
    let value = match objective {
        Objective::Classification { labels } => {
            let mut counts = vec![0usize; labels.len()];
            for &index in rows { counts[label_index(labels, &train[index].target)] += 1; }
            1.0 - counts.iter().map(|value| (*value as f64 / count).powi(2)).sum::<f64>()
        }
        Objective::Regression { values } => {
            let mean = rows.iter().map(|&index| values[index]).sum::<f64>() / count;
            rows.iter().map(|&index| (values[index] - mean).powi(2)).sum::<f64>() / count
        }
        Objective::Boosted { gradients, hessians } => {
            let gradient = rows.iter().map(|&index| gradients[index]).sum::<f64>();
            let hessian = rows.iter().map(|&index| hessians[index]).sum::<f64>();
            gradient * gradient / (hessian + 1.0)
        }
    };
    if !value.is_finite() { return Err(LanaError::Schema); }
    Ok(value)
}

fn leaf(objective: &Objective, rows: &[usize], train: &[Example]) -> Result<Json, LanaError> {
    let count = rows.len();
    match objective {
        Objective::Classification { labels } => {
            let mut counts = vec![0usize; labels.len()];
            for &index in rows { counts[label_index(labels, &train[index].target)] += 1; }
            Ok(json!({"kind":"leaf","count":count,"label_counts":counts,"value":null}))
        }
        Objective::Regression { values } => {
            let mean = rows.iter().map(|&index| values[index]).sum::<f64>() / count as f64;
            Ok(json!({"kind":"leaf","count":count,"label_counts":[],"value":bits(mean)?}))
        }
        Objective::Boosted { gradients, hessians } => {
            let gradient = rows.iter().map(|&index| gradients[index]).sum::<f64>();
            let hessian = rows.iter().map(|&index| hessians[index]).sum::<f64>();
            Ok(json!({"kind":"leaf","count":count,"label_counts":[],"value":bits(-gradient / (hessian + 1.0))?}))
        }
    }
}

fn candidates(feature: &Feature, rows: &[usize], train: &[Example]) -> Result<Vec<Json>, LanaError> {
    match feature.kind.as_str() {
        "number" => {
            let mut values = rows.iter().filter_map(|&index| train[index].features[&feature.name].as_f64()).collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            values.dedup();
            let mut result = Vec::new();
            for pair in values.windows(2) {
                let difference = pair[1] - pair[0];
                let midpoint = if difference.is_finite() { pair[0] + difference / 2.0 }
                    else { pair[0] / 2.0 + pair[1] / 2.0 };
                if midpoint.is_finite() && midpoint > pair[0] && midpoint < pair[1] { result.push(json!(midpoint)); }
            }
            Ok(result)
        }
        "boolean" => Ok(vec![json!(false), json!(true)]),
        "category" => Ok(feature.categories.iter().map(|value| json!(value)).collect()),
        _ => Err(LanaError::Schema),
    }
}

fn goes_left(value: &Json, feature: &Feature, constant: &Json, missing_left: bool) -> bool {
    if value.is_null() { return missing_left; }
    if feature.kind == "number" { value.as_f64().unwrap() <= constant.as_f64().unwrap() }
    else { value == constant }
}

struct Builder<'a, 'vm> {
    task: &'a Task,
    train: &'a [Example],
    objective: &'a Objective,
    settings: &'a Settings,
    rng: &'a mut Rng,
    vm: &'a mut Vm<'vm>,
    nodes: Vec<Json>,
    split_evaluations: u64,
    forest: bool,
}

impl Builder<'_, '_> {
    fn build(&mut self, rows: &[usize], depth: usize) -> Result<usize, LanaError> {
        self.vm.charge_bounded_work(rows.len() as u64)?;
        let here = self.nodes.len();
        // ponytail: bound native node allocation; use heap-tracked nodes if larger models are needed.
        if here >= 100_000 { return Err(LanaError::Limit); }
        self.nodes.push(Json::Null);
        if depth >= self.settings.max_depth || rows.len() < self.settings.min_leaf * 2 {
            self.nodes[here] = leaf(self.objective, rows, self.train)?;
            return Ok(here);
        }
        let parent = objective_value(self.objective, rows, self.train)?;
        let mut feature_indices = (0..self.task.feature_schema.len()).collect::<Vec<_>>();
        if self.forest {
            let count = ((feature_indices.len() as f64).sqrt().floor() as usize).max(1);
            for index in 0..count {
                let chosen = index + self.rng.random() as usize % (feature_indices.len() - index);
                feature_indices.swap(index, chosen);
            }
            feature_indices.truncate(count);
            feature_indices.sort_unstable();
        }
        let mut best: Option<(f64, usize, Json, bool, Vec<usize>, Vec<usize>)> = None;
        for feature_index in feature_indices {
            let feature = &self.task.feature_schema[feature_index];
            for constant in candidates(feature, rows, self.train)? {
                for missing_left in [true, false] {
                    self.vm.charge_bounded_work(rows.len() as u64)?;
                    self.split_evaluations = self.split_evaluations.checked_add(1).ok_or(LanaError::Limit)?;
                    let mut left = Vec::new();
                    let mut right = Vec::new();
                    for &index in rows {
                        if goes_left(&self.train[index].features[&feature.name], feature, &constant, missing_left) {
                            left.push(index);
                        } else { right.push(index); }
                    }
                    if left.len() < self.settings.min_leaf || right.len() < self.settings.min_leaf { continue; }
                    let left_score = objective_value(self.objective, &left, self.train)?;
                    let right_score = objective_value(self.objective, &right, self.train)?;
                    let gain = if matches!(self.objective, Objective::Boosted { .. }) {
                        left_score + right_score - parent
                    } else {
                        parent - (left_score * left.len() as f64 + right_score * right.len() as f64) / rows.len() as f64
                    };
                    if gain.is_finite() && gain > 0.0 && best.as_ref().is_none_or(|prior| gain > prior.0) {
                        best = Some((gain, feature_index, constant.clone(), missing_left, left, right));
                    }
                }
            }
        }
        if let Some((gain, feature_index, constant, missing_left, left_rows, right_rows)) = best {
            let left = self.build(&left_rows, depth + 1)?;
            let right = self.build(&right_rows, depth + 1)?;
            let feature = &self.task.feature_schema[feature_index];
            self.nodes[here] = json!({"kind":"split","feature":feature.name,
                "operator":if feature.kind == "number" { "le" } else { "eq" },
                "constant":constant,"missing_left":missing_left,"left":left,"right":right,"gain":bits(gain)?});
        } else { self.nodes[here] = leaf(self.objective, rows, self.train)?; }
        Ok(here)
    }
}

fn fit_tree(task: &Task, train: &[Example], objective: &Objective, settings: &Settings,
    rows: &[usize], rng: &mut Rng, vm: &mut Vm, forest: bool) -> Result<(Json, u64), LanaError> {
    let mut builder = Builder { task, train, objective, settings, rng, vm, nodes: Vec::new(), split_evaluations: 0, forest };
    builder.build(rows, 0)?;
    Ok((json!({"nodes":builder.nodes}), builder.split_evaluations))
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 { 1.0 / (1.0 + (-value).exp()) }
    else { let exponential = value.exp(); exponential / (1.0 + exponential) }
}

fn validate_model_shape(model: &Json) -> Result<Vec<Feature>, LanaError> {
    let fields = model.as_object().ok_or(LanaError::Schema)?;
    if fields.len() != 8 || !["family", "problem", "labels", "feature_schema", "seed", "base_score", "learning_rate", "trees"]
        .iter().all(|key| fields.contains_key(*key)) { return Err(LanaError::Schema); }
    let family = model["family"].as_str().ok_or(LanaError::Schema)?;
    let problem = model["problem"].as_str().ok_or(LanaError::Schema)?;
    if !matches!(family, "tree" | "forest" | "boosted") || !matches!(problem, "classification" | "regression") {
        return Err(LanaError::Schema);
    }
    let seed = model["seed"].as_str().ok_or(LanaError::Schema)?;
    if seed.parse::<u64>().ok().is_none_or(|value| value.to_string() != seed) { return Err(LanaError::Schema); }
    let labels = model["labels"].as_array().ok_or(LanaError::Schema)?;
    if (problem == "regression" && !labels.is_empty())
        || (problem == "classification" && !(2..=128).contains(&labels.len())) {
        return Err(LanaError::Schema);
    }
    for (index, label) in labels.iter().enumerate() {
        if !(label.is_boolean() || label.as_str().is_some_and(valid_id)) || labels[..index].contains(label) {
            return Err(LanaError::Schema);
        }
    }
    if family == "boosted" {
        from_bits(&model["base_score"])?;
        if from_bits(&model["learning_rate"])? != 0.1 { return Err(LanaError::Schema); }
        if problem == "classification" && labels.len() != 2 { return Err(LanaError::UnsupportedOperation); }
    } else if !model["base_score"].is_null() || !model["learning_rate"].is_null() {
        return Err(LanaError::Schema);
    }
    let trees = model["trees"].as_array().ok_or(LanaError::Schema)?;
    if trees.is_empty() || trees.len() > 100 || (family == "tree" && trees.len() != 1) { return Err(LanaError::Schema); }
    let features: Vec<Feature> = serde_json::from_value(model["feature_schema"].clone()).map_err(|_| LanaError::Schema)?;
    validate_schema(&features)?;
    let mut total_nodes = 0usize;
    for tree in trees {
        if tree.as_object().is_none_or(|fields| fields.len() != 1) { return Err(LanaError::Schema); }
        let nodes = tree["nodes"].as_array().ok_or(LanaError::Schema)?;
        if nodes.is_empty() || nodes.len() > 131_071 { return Err(LanaError::Schema); }
        total_nodes = total_nodes.checked_add(nodes.len()).ok_or(LanaError::Limit)?;
        if total_nodes > 100_000 { return Err(LanaError::Limit); }
        let mut seen = vec![false; nodes.len()];
        let mut stack = vec![0usize];
        while let Some(index) = stack.pop() {
            if seen[index] { return Err(LanaError::Schema); }
            seen[index] = true;
            let node = &nodes[index];
            let fields = node.as_object().ok_or(LanaError::Schema)?;
            match node["kind"].as_str() {
                Some("leaf") => {
                    if fields.len() != 4 || node["count"].as_u64().is_none_or(|count| count == 0) {
                        return Err(LanaError::Schema);
                    }
                    let counts = node["label_counts"].as_array().ok_or(LanaError::Schema)?;
                    if problem == "classification" && family != "boosted" {
                        if counts.len() != labels.len() || counts.iter().any(|count| count.as_u64().is_none())
                            || !node["value"].is_null() { return Err(LanaError::Schema); }
                        let total = counts.iter().try_fold(0u64, |sum, count|
                            sum.checked_add(count.as_u64().unwrap()).ok_or(LanaError::Schema))?;
                        if total != node["count"].as_u64().unwrap() { return Err(LanaError::Schema); }
                    } else if !counts.is_empty() { return Err(LanaError::Schema); }
                    if problem == "regression" || family == "boosted" { from_bits(&node["value"])?; }
                }
                Some("split") => {
                    if fields.len() != 8 || node["missing_left"].as_bool().is_none() { return Err(LanaError::Schema); }
                    let name = node["feature"].as_str().ok_or(LanaError::Schema)?;
                    if !valid_id(name) { return Err(LanaError::Schema); }
                    let feature = features.iter().find(|feature| feature.name == name).ok_or(LanaError::Schema)?;
                    if !valid_feature(&node["constant"], feature)
                        || (feature.kind == "number") != (node["operator"] == "le") { return Err(LanaError::Schema); }
                    match node["operator"].as_str() {
                        Some("le") if node["constant"].is_number() => { real(&node["constant"])?; },
                        Some("eq") if node["constant"].is_boolean() || node["constant"].is_string() => {},
                        _ => return Err(LanaError::Schema),
                    }
                    let gain = from_bits(&node["gain"])?;
                    if gain <= 0.0 { return Err(LanaError::Schema); }
                    let left = usize::try_from(node["left"].as_u64().ok_or(LanaError::Schema)?)
                        .map_err(|_| LanaError::Schema)?;
                    let right = usize::try_from(node["right"].as_u64().ok_or(LanaError::Schema)?)
                        .map_err(|_| LanaError::Schema)?;
                    if left <= index || right <= index || left >= nodes.len() || right >= nodes.len() || left == right {
                        return Err(LanaError::Schema);
                    }
                    stack.push(right);
                    stack.push(left);
                }
                _ => return Err(LanaError::Schema),
            }
        }
        if seen.iter().any(|seen| !seen) { return Err(LanaError::Schema); }
    }
    Ok(features)
}

fn traverse<'a>(tree: &'a Json, features: &BTreeMap<String, Json>, vm: &mut Vm)
    -> Result<(&'a Json, Vec<Json>), LanaError> {
    let nodes = tree["nodes"].as_array().ok_or(LanaError::Schema)?;
    if nodes.is_empty() || nodes.len() > 131_071 { return Err(LanaError::Schema); }
    let mut index = 0usize;
    let mut path = Vec::new();
    for _ in 0..=nodes.len() {
        vm.charge_bounded_work(1)?;
        let node = nodes.get(index).ok_or(LanaError::Schema)?;
        match node["kind"].as_str() {
            Some("leaf") => return Ok((node, path)),
            Some("split") => {
                let name = node["feature"].as_str().ok_or(LanaError::Schema)?;
                let operator = node["operator"].as_str().ok_or(LanaError::Schema)?;
                let constant = &node["constant"];
                let missing_left = node["missing_left"].as_bool().ok_or(LanaError::Schema)?;
                let value = features.get(name).unwrap_or(&Json::Null);
                let left = if value.is_null() { missing_left } else { match operator {
                    "le" => {
                        let number = real(value).map_err(|_| LanaError::Type)?;
                        number <= real(constant)?
                    }
                    "eq" => {
                        if value.is_number() != constant.is_number()
                            || value.is_boolean() != constant.is_boolean()
                            || value.is_string() != constant.is_string() { return Err(LanaError::Type); }
                        value == constant
                    }
                    _ => return Err(LanaError::Schema),
                }};
                path.push(json!({"node":index,"feature":name,"operator":operator,
                    "constant":constant,"value":value,"went_left":left,"gain":node["gain"]}));
                let next = node[if left { "left" } else { "right" }].as_u64().ok_or(LanaError::Schema)? as usize;
                if next <= index || next >= nodes.len() { return Err(LanaError::Schema); }
                index = next;
            }
            _ => return Err(LanaError::Schema),
        }
    }
    Err(LanaError::Schema)
}

fn predicted(model: &Json, features: &BTreeMap<String, Json>, vm: &mut Vm)
    -> Result<(Json, Vec<Json>), LanaError> {
    validate_model_shape(model)?;
    let nodes = model["trees"].as_array().unwrap().iter()
        .map(|tree| tree["nodes"].as_array().unwrap().len() as u64).sum();
    vm.charge_bounded_work(nodes)?;
    let family = model["family"].as_str().ok_or(LanaError::Schema)?;
    let problem = model["problem"].as_str().ok_or(LanaError::Schema)?;
    let trees = model["trees"].as_array().ok_or(LanaError::Schema)?;
    if trees.is_empty() || trees.len() > 100 { return Err(LanaError::Schema); }
    let labels = model["labels"].as_array().ok_or(LanaError::Schema)?;
    let mut paths = Vec::new();
    let mut leaves = Vec::new();
    for tree in trees {
        let (leaf, path) = traverse(tree, features, vm)?;
        leaves.push(leaf);
        paths.push(json!({"path":path,"leaf":leaf}));
    }
    let result = match (family, problem) {
        ("tree", "classification") => {
            let counts = leaves[0]["label_counts"].as_array().ok_or(LanaError::Schema)?;
            if counts.len() != labels.len() { return Err(LanaError::Schema); }
            let total = counts.iter().try_fold(0u64, |sum, value|
                sum.checked_add(value.as_u64().ok_or(LanaError::Schema)?).ok_or(LanaError::Schema))? as f64;
            if total <= 0.0 { return Err(LanaError::Schema); }
            let fractions = counts.iter().map(|value| value.as_u64().unwrap() as f64 / total).collect::<Vec<_>>();
            let selected = fractions.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .ok_or(LanaError::Schema)?.0;
            json!({"label":labels[selected],"probabilities":labels.iter().zip(fractions)
                .map(|(label, probability)| json!({"label":label,"probability":probability})).collect::<Vec<_>>(),
                "calibration":"uncalibrated"})
        }
        ("forest", "classification") => {
            let mut votes = vec![0usize; labels.len()];
            for leaf in &leaves {
                let counts = leaf["label_counts"].as_array().ok_or(LanaError::Schema)?;
                if counts.len() != labels.len() { return Err(LanaError::Schema); }
                let selected = counts.iter().enumerate().max_by(|a, b|
                    a.1.as_u64().cmp(&b.1.as_u64()).then_with(|| b.0.cmp(&a.0))).ok_or(LanaError::Schema)?.0;
                votes[selected] += 1;
            }
            let selected = votes.iter().enumerate().max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .ok_or(LanaError::Schema)?.0;
            json!({"label":labels[selected],"probabilities":labels.iter().zip(votes)
                .map(|(label, count)| json!({"label":label,"probability":count as f64 / trees.len() as f64})).collect::<Vec<_>>(),
                "calibration":"uncalibrated"})
        }
        ("boosted", "classification") if labels.len() == 2 => {
            let mut score = from_bits(&model["base_score"])?;
            let learning_rate = from_bits(&model["learning_rate"])?;
            for leaf in leaves { score += learning_rate * from_bits(&leaf["value"])?; }
            if !score.is_finite() { return Err(LanaError::Schema); }
            let positive = sigmoid(score);
            let selected = if positive > 0.5 { 1 } else { 0 };
            json!({"label":labels[selected],"probabilities":[
                {"label":labels[0],"probability":1.0-positive},
                {"label":labels[1],"probability":positive}],"calibration":"uncalibrated"})
        }
        ("tree" | "forest" | "boosted", "regression") => {
            let values = leaves.iter().map(|leaf| from_bits(&leaf["value"]))
                .collect::<Result<Vec<_>, _>>()?;
            let value = if family == "boosted" {
                from_bits(&model["base_score"])? + from_bits(&model["learning_rate"])? * values.iter().sum::<f64>()
            } else { values.iter().sum::<f64>() / values.len() as f64 };
            if !value.is_finite() { return Err(LanaError::Schema); }
            json!({"value":value,"calibration":"uncalibrated"})
        }
        _ => return Err(LanaError::Schema),
    };
    Ok((result, paths))
}

fn fit_json(task: &Task, train: &[Example], holdout: &[Example], settings: &Settings, vm: &mut Vm)
    -> Result<Json, LanaError> {
    let mut rng = Rng::new();
    rng.seed(settings.seed);
    let rows = (0..train.len()).collect::<Vec<_>>();
    let mut trees = Vec::new();
    let mut total_nodes = 0usize;
    let mut split_evaluations = 0u64;
    let base_score = if settings.family == "boosted" {
        if task.problem == "regression" {
            Some(train.iter().map(|row| real(&row.target)).collect::<Result<Vec<_>, _>>()?
                .iter().sum::<f64>() / train.len() as f64)
        } else {
            let positives = train.iter().filter(|row| row.target == task.labels[1]).count() as f64;
            let fraction = (positives / train.len() as f64).clamp(1e-6, 1.0 - 1e-6);
            Some((fraction / (1.0 - fraction)).ln())
        }
    } else { None };
    let mut scores = vec![base_score.unwrap_or(0.0); train.len()];
    for _ in 0..settings.ensemble_size {
        let (sample, objective, forest) = match (settings.family.as_str(), task.problem.as_str()) {
            ("tree", "classification") | ("forest", "classification") => {
                let sample = if settings.family == "forest" {
                    (0..train.len()).map(|_| rng.random() as usize % train.len()).collect()
                } else { rows.clone() };
                (sample, Objective::Classification { labels: task.labels.clone() }, settings.family == "forest")
            }
            ("tree", "regression") | ("forest", "regression") => {
                let sample = if settings.family == "forest" {
                    (0..train.len()).map(|_| rng.random() as usize % train.len()).collect()
                } else { rows.clone() };
                (sample, Objective::Regression { values: train.iter().map(|row| real(&row.target)).collect::<Result<_, _>>()? },
                    settings.family == "forest")
            }
            ("boosted", "regression") => {
                let values = train.iter().enumerate().map(|(index, row)| Ok(real(&row.target)? - scores[index]))
                    .collect::<Result<Vec<_>, LanaError>>()?;
                (rows.clone(), Objective::Regression { values }, false)
            }
            ("boosted", "classification") => {
                let mut gradients = Vec::with_capacity(train.len());
                let mut hessians = Vec::with_capacity(train.len());
                for (index, row) in train.iter().enumerate() {
                    let probability = sigmoid(scores[index]);
                    gradients.push(probability - f64::from(row.target == task.labels[1]));
                    hessians.push((probability * (1.0 - probability)).max(1e-6));
                }
                (rows.clone(), Objective::Boosted { gradients, hessians }, false)
            }
            _ => return Err(LanaError::Schema),
        };
        let (tree, count) = fit_tree(task, train, &objective, settings, &sample, &mut rng, vm, forest)?;
        total_nodes = total_nodes.checked_add(tree["nodes"].as_array().unwrap().len()).ok_or(LanaError::Limit)?;
        if total_nodes > 100_000 { return Err(LanaError::Limit); }
        split_evaluations = split_evaluations.checked_add(count).ok_or(LanaError::Limit)?;
        if settings.family == "boosted" {
            for (index, row) in train.iter().enumerate() {
                let (leaf, _) = traverse(&tree, &row.features, vm)?;
                scores[index] += 0.1 * from_bits(&leaf["value"])?;
                if !scores[index].is_finite() { return Err(LanaError::Schema); }
            }
        }
        trees.push(tree);
    }
    let model = json!({"family":settings.family,"problem":task.problem,"labels":task.labels,"feature_schema":task.feature_schema,
        "seed":settings.seed.to_string(),"base_score":base_score.map(bits).transpose()?,
        "learning_rate":if settings.family == "boosted" { Some(bits(0.1)?) } else { None },"trees":trees});
    let mut trace = Vec::with_capacity(holdout.len());
    let mut correct = 0usize;
    let mut per_count = vec![0usize; task.labels.len()];
    let mut per_correct = vec![0usize; task.labels.len()];
    let mut absolute_error = 0.0;
    let mut squared_error = 0.0;
    for row in holdout {
        let (prediction, _) = predicted(&model, &row.features, vm)?;
        let value = if task.problem == "classification" { prediction["label"].clone() } else { prediction["value"].clone() };
        if task.problem == "classification" {
            let label = label_index(&task.labels, &row.target);
            per_count[label] += 1;
            if value == row.target { correct += 1; per_correct[label] += 1; }
        } else {
            let error = real(&row.target)? - real(&value)?;
            absolute_error += error.abs();
            squared_error += error * error;
            if !absolute_error.is_finite() || !squared_error.is_finite() { return Err(LanaError::Schema); }
        }
        trace.push(json!({"id":row.id,"target":row.target,"prediction":value}));
    }
    let count = holdout.len();
    let metrics = if task.problem == "classification" {
        json!({"count":count,"correct":correct,"accuracy":if count == 0 { Json::Null } else { json!(correct as f64 / count as f64) },
            "per_label":task.labels.iter().enumerate().map(|(index, label)|
                json!({"label":label,"count":per_count[index],"correct":per_correct[index]})).collect::<Vec<_>>()})
    } else {
        json!({"count":count,"mae":if count == 0 { Json::Null } else { json!(absolute_error / count as f64) },
            "rmse":if count == 0 { Json::Null } else { json!((squared_error / count as f64).sqrt()) }})
    };
    let enough = count >= 20 && (task.problem == "regression" || per_count.iter().all(|value| *value > 0));
    let limits = json!({"candidate_evaluations":0,"predicate_visits":0,"split_evaluations":split_evaluations});
    let training_ids = train.iter().map(|row| row.id.clone()).collect::<Vec<_>>();
    let source_ids = train.iter().chain(holdout).map(|row| row.id.clone()).collect::<Vec<_>>();
    Ok(json!({"schema_version":1,"status":if enough { "validated" } else { "insufficient_evidence" },
        "task":task,"options":{"family":settings.family,"seed":settings.seed.to_string(),
            "max_depth":settings.max_depth,"min_leaf":settings.min_leaf,"ensemble_size":settings.ensemble_size},
        "train":train,"holdout":holdout,"model_or_rule":model,
        "training_report":{"training_ids":training_ids,"mistakes":[],"limits_used":limits,
            "search_status":null,"rejections":[]},
        "validation_report":{"holdout_trace":trace,"metrics":metrics},"mistakes":[],"source_ids":source_ids,
        "calculation_version":"trees-v1","limits_used":limits}))
}

pub fn fit(args: &[Value], vm: &mut Vm) -> Result<Value, LanaError> {
    if args.len() != 4 { return Err(LanaError::InvalidParameters); }
    let task: Task = serde_json::from_value(input(&args[0])?).map_err(|_| LanaError::Schema)?;
    let train: Vec<Example> = serde_json::from_value(input(&args[1])?).map_err(|_| LanaError::Schema)?;
    let holdout: Vec<Example> = serde_json::from_value(input(&args[2])?).map_err(|_| LanaError::Schema)?;
    let options: Options = serde_json::from_value(input(&args[3])?).map_err(|_| LanaError::Schema)?;
    let settings = validate(&task, &train, &holdout, &options)?;
    let result = fit_json(&task, &train, &holdout, &settings, vm)?;
    let encoded = serde_json::to_string(&result).map_err(|_| LanaError::Schema)?;
    if encoded.len() > 64 * 1024 * 1024 { return Err(LanaError::Limit); }
    data::json_parse_with_heap(&encoded, &vm.heap())
}

pub fn predict(args: &[Value], vm: &mut Vm) -> Result<Value, LanaError> {
    if args.len() != 2 { return Err(LanaError::InvalidParameters); }
    let (model, task) = selected_model(&args[0], vm)?;
    let features: BTreeMap<String, Json> = serde_json::from_value(input(&args[1])?).map_err(|_| LanaError::Schema)?;
    if features.len() > 64 { return Err(LanaError::Schema); }
    validate_features(&model, task.as_ref(), &features)?;
    let (result, _) = predicted(&model, &features, vm)?;
    data::json_parse_with_heap(&result.to_string(), &vm.heap())
}

pub fn explain(args: &[Value], vm: &mut Vm) -> Result<Value, LanaError> {
    if args.len() != 2 { return Err(LanaError::InvalidParameters); }
    let (model, task) = selected_model(&args[0], vm)?;
    let features: BTreeMap<String, Json> = serde_json::from_value(input(&args[1])?).map_err(|_| LanaError::Schema)?;
    if features.len() > 64 { return Err(LanaError::Schema); }
    validate_features(&model, task.as_ref(), &features)?;
    let (prediction, contributions) = predicted(&model, &features, vm)?;
    let result = json!({"prediction":prediction,"contributions":contributions,
        "aggregation":model["family"]});
    data::json_parse_with_heap(&result.to_string(), &vm.heap())
}

fn selected_model(value: &Value, vm: &mut Vm) -> Result<(Json, Option<Task>), LanaError> {
    let input = input(value)?;
    if input["format"] != "learned_task_v1" { return Ok((input, None)); }
    let mut bytes = crate::information_codec::canonical(&input)?;
    bytes.push(b'\n');
    let saved = String::from_utf8(bytes).map_err(|_| LanaError::Schema)?;
    let record = load_record(&saved, vm)?;
    let active = record["active_version"].as_str().ok_or(LanaError::UnsupportedOperation)?;
    let version = record["versions"].as_array().unwrap().iter().find(|version| version["version"] == active)
        .ok_or(LanaError::Corruption)?;
    let mut decoded = version.clone();
    transform_version(&mut decoded, false)?;
    let task: Task = serde_json::from_value(decoded["task"].clone()).map_err(|_| LanaError::Corruption)?;
    if decoded["model"].get("feature_schema").is_none() {
        decoded["model"]["feature_schema"] = json!(task.feature_schema);
    }
    Ok((decoded["model"].clone(), Some(task)))
}

fn validate_features(model: &Json, task: Option<&Task>, features: &BTreeMap<String, Json>)
    -> Result<(), LanaError> {
    let schema = validate_model_shape(model)?;
    if let Some(task) = task {
        if json!(task.feature_schema) != json!(schema) { return Err(LanaError::Schema); }
    }
    for (name, value) in features {
        let feature = schema.iter().find(|feature| feature.name == *name).ok_or(LanaError::Schema)?;
        if !valid_feature(value, feature) { return Err(LanaError::Type); }
    }
    Ok(())
}

fn equal_numbers(left: &Json, right: &Json) -> bool {
    match (left, right) {
        (Json::Number(a), Json::Number(b)) => a.as_f64() == b.as_f64(),
        (Json::Array(a), Json::Array(b)) => a.len() == b.len()
            && a.iter().zip(b).all(|(a, b)| equal_numbers(a, b)),
        (Json::Object(a), Json::Object(b)) => a.len() == b.len()
            && a.iter().all(|(key, value)| b.get(key).is_some_and(|other| equal_numbers(value, other))),
        _ => left == right,
    }
}

/// Refit from complete ordered inputs to reject forged metrics or model nodes.
pub fn validate_report(value: &Value, vm: &mut Vm) -> Result<Json, LanaError> {
    let report = input(value)?;
    let task: Task = serde_json::from_value(report["task"].clone()).map_err(|_| LanaError::Schema)?;
    let train: Vec<Example> = serde_json::from_value(report["train"].clone()).map_err(|_| LanaError::Schema)?;
    let holdout: Vec<Example> = serde_json::from_value(report["holdout"].clone()).map_err(|_| LanaError::Schema)?;
    let mut options = report["options"].clone();
    let seed = options["seed"].as_str().ok_or(LanaError::Schema)?;
    let seed = seed.parse::<u64>().map_err(|_| LanaError::Schema)?;
    if seed.to_string() != options["seed"] { return Err(LanaError::Schema); }
    options["seed"] = json!(seed);
    let options: Options = serde_json::from_value(options).map_err(|_| LanaError::Schema)?;
    let settings = validate(&task, &train, &holdout, &options)?;
    let mut expected = fit_json(&task, &train, &holdout, &settings, vm)?;
    // Older complete reports retain the schema in their task. Refit before
    // accepting that representation; standalone models cannot infer a schema.
    if report["model_or_rule"].get("feature_schema").is_none() {
        expected["model_or_rule"].as_object_mut().unwrap().remove("feature_schema");
    }
    if !equal_numbers(&expected, &report) { return Err(LanaError::Schema); }
    Ok(report)
}

fn transform_version(version: &mut Json, encode: bool) -> Result<(), LanaError> {
    for group in ["train", "holdout"] { rules::transform_examples(&mut version[group], encode)?; }
    for tree in version["model"]["trees"].as_array_mut().ok_or(LanaError::Schema)? {
        for node in tree["nodes"].as_array_mut().ok_or(LanaError::Schema)? {
            if node["kind"] == "split" { rules::transform_scalar(&mut node["constant"], encode)?; }
        }
    }
    let metrics = &mut version["report"]["metrics"];
    for name in ["accuracy", "mae", "rmse"] {
        if metrics.get(name).is_none() || metrics[name].is_null() { continue; }
        if encode {
            metrics[name] = json!(bits(real(&metrics[name])?)?);
        } else { metrics[name] = json!(from_bits(&metrics[name])?); }
    }
    Ok(())
}

fn version_from_report(version_id: &str, parent: Json, report: &Json) -> Result<Json, LanaError> {
    let training = &report["training_report"];
    let validation = &report["validation_report"];
    let version_report = json!({"status":report["status"],
        "training_ids":training["training_ids"],"holdout_trace":validation["holdout_trace"],
        "metrics":validation["metrics"],"mistakes":report["mistakes"],
        "calculation_version":report["calculation_version"],"limits_used":report["limits_used"],
        "search_status":null,"rejections":[]});
    let mut version = json!({"version":version_id,"parent_version":parent,"task":report["task"],
        "options":report["options"],"train":report["train"],"holdout":report["holdout"],
        "model":report["model_or_rule"],"report":version_report});
    transform_version(&mut version, true)?;
    Ok(version)
}

fn report_from_version(version: &Json) -> Result<Json, LanaError> {
    let mut decoded = version.clone();
    transform_version(&mut decoded, false)?;
    let r = &decoded["report"];
    let train = decoded["train"].as_array().ok_or(LanaError::Corruption)?;
    let holdout = decoded["holdout"].as_array().ok_or(LanaError::Corruption)?;
    let source_ids = train.iter().chain(holdout).map(|row| row["id"].clone()).collect::<Vec<_>>();
    Ok(json!({"schema_version":1,"status":r["status"],"task":decoded["task"],
        "options":decoded["options"],"train":decoded["train"],"holdout":decoded["holdout"],
        "model_or_rule":decoded["model"],"training_report":{"training_ids":r["training_ids"],
            "mistakes":r["mistakes"],"limits_used":r["limits_used"],
            "search_status":r["search_status"],"rejections":r["rejections"]},
        "validation_report":{"holdout_trace":r["holdout_trace"],"metrics":r["metrics"]},
        "mistakes":r["mistakes"],"source_ids":source_ids,
        "calculation_version":r["calculation_version"],"limits_used":r["limits_used"]}))
}

fn seal_record(record: &mut Json, vm: &mut Vm) -> Result<String, LanaError> {
    record.as_object_mut().ok_or(LanaError::Schema)?.remove("digest");
    let body = crate::information_codec::canonical(record)?;
    let digest = crate::sha256::sha256(&body).iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    record["digest"] = json!(digest);
    let mut encoded = crate::information_codec::canonical(record)?;
    encoded.push(b'\n');
    if encoded.len() > 64 * 1024 * 1024 { return Err(LanaError::Limit); }
    let text = String::from_utf8(encoded).map_err(|_| LanaError::Schema)?;
    load_record(&text, vm)?;
    Ok(text)
}

pub fn append_record(previous: Option<Json>, task_id: &str, report: &Json, vm: &mut Vm)
    -> Result<(String, String), LanaError> {
    if !valid_id(task_id) || report["task"]["id"] != task_id { return Err(LanaError::Schema); }
    let mut record = if let Some(previous) = previous { previous } else {
        json!({"schema_version":1,"format":"learned_task_v1","kind":"tree",
            "task_id":task_id,"versions":[],"active_version":null,"receipts":[]})
    };
    if record["kind"] != "tree" || record["task_id"] != task_id { return Err(LanaError::Conflict); }
    let versions = record["versions"].as_array().ok_or(LanaError::Corruption)?;
    if versions.len() >= 10_000 { return Err(LanaError::Limit); }
    if let Some(first) = versions.first() {
        if first["task"] != report["task"] { return Err(LanaError::Conflict); }
    }
    let id = (versions.len() + 1).to_string();
    let parent = versions.last().map_or(Json::Null, |version| version["version"].clone());
    record["versions"].as_array_mut().ok_or(LanaError::Corruption)?
        .push(version_from_report(&id, parent, report)?);
    if report["status"] == "validated" { record["active_version"] = json!(id); }
    let text = seal_record(&mut record, vm)?;
    Ok((id, text))
}

pub fn load_record(text: &str, vm: &mut Vm) -> Result<Json, LanaError> {
    if text.len() > 64 * 1024 * 1024 { return Err(LanaError::Limit); }
    let record: Json = serde_json::from_str(text).map_err(|_| LanaError::Corruption)?;
    let fields = record.as_object().ok_or(LanaError::Corruption)?;
    if fields.len() != 8 || record["schema_version"] != 1 || record["format"] != "learned_task_v1"
        || record["kind"] != "tree" || record["task_id"].as_str().is_none_or(|id| !valid_id(id))
        || record["receipts"] != json!([]) { return Err(LanaError::Corruption); }
    let canonical = crate::information_codec::canonical(&record).map_err(|_| LanaError::Corruption)?;
    if text.as_bytes() != [canonical.as_slice(), b"\n"].concat() { return Err(LanaError::Corruption); }
    let mut body = record.clone();
    body.as_object_mut().unwrap().remove("digest");
    let digest = crate::sha256::sha256(&crate::information_codec::canonical(&body).map_err(|_| LanaError::Corruption)?)
        .iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    if record["digest"] != digest { return Err(LanaError::Corruption); }
    let versions = record["versions"].as_array().ok_or(LanaError::Corruption)?;
    if versions.is_empty() || versions.len() > 10_000 { return Err(LanaError::Corruption); }
    for (index, version) in versions.iter().enumerate() {
        let id = (index + 1).to_string();
        let parent = if index == 0 { Json::Null } else { json!(index.to_string()) };
        if version.as_object().is_none_or(|fields| fields.len() != 8)
            || version["version"] != id || version["parent_version"] != parent
            || version["task"]["id"] != record["task_id"]
            || version["task"] != versions[0]["task"]
            || version["report"].as_object().is_none_or(|fields| fields.len() != 9) {
            return Err(LanaError::Corruption);
        }
        let report = report_from_version(version).map_err(|_| LanaError::Corruption)?;
        let saved = data::json_parse_with_heap(&report.to_string(), &vm.heap())?;
        validate_report(&saved, vm).map_err(|_| LanaError::Corruption)?;
    }
    if let Some(active) = record["active_version"].as_str() {
        let number = crate::information_codec::revision(active).map_err(|_| LanaError::Corruption)? as usize;
        if number == 0 || number > versions.len() || versions[number - 1]["report"]["status"] != "validated" {
            return Err(LanaError::Corruption);
        }
    } else if !record["active_version"].is_null() { return Err(LanaError::Corruption); }
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lana_bytecode::Chunk;

    fn inputs(vm: &Vm, family: &str, regression: bool, flipped: bool) -> [Value; 4] {
        let target = |yes: bool| if regression { json!(if yes { 6.0 } else { 2.0 }) } else { json!(yes) };
        let rows = |prefix: &str, count: usize, flip: bool| (0..count).map(|i| json!({
            "id":format!("{prefix}{i}"), "features":{"x":if i % 2 == 0 { 1 } else { 3 }},
            "target":target((i % 2 == 1) ^ flip)})).collect::<Vec<_>>();
        [json!({"id":"test","feature_schema":[{"name":"x","kind":"number","nullable":false,"categories":[]}],
            "problem":if regression { "regression" } else { "classification" },
            "labels":if regression { json!([]) } else { json!([false,true]) }}),
            json!(rows("t",2,false)),json!(rows("h",24,flipped)),
            json!({"family":family,"min_leaf":1,"max_depth":1,"ensemble_size":if family == "forest" {3} else {1}})]
            .map(|value| data::json_parse_with_heap(&value.to_string(), &vm.heap()).unwrap())
    }

    #[test]
    fn all_families_keep_holdout_out_of_training_and_validate_saved_evidence() {
        let chunk = Chunk::new(5, 0);
        for regression in [false, true] {
            for family in ["tree", "forest", "boosted"] {
                let mut vm = Vm::new(&chunk);
                let args = inputs(&vm, family, regression, false);
                let value = fit(&args, &mut vm).unwrap();
                let report = validate_report(&value, &mut vm).unwrap_or_else(|error| panic!("{family}/{regression}: {error:?}"));
                let changed = fit(&inputs(&vm, family, regression, true), &mut vm).unwrap();
                assert_eq!(input(&changed).unwrap()["model_or_rule"], report["model_or_rule"]);
                assert_eq!(report["status"], "validated");
                let (_, saved) = append_record(None, "test", &report, &mut vm).unwrap();
                let original = load_record(&saved, &mut vm).unwrap();
                let mut legacy = original.clone();
                legacy["versions"][0]["model"].as_object_mut().unwrap().remove("feature_schema");
                let legacy_bytes = seal_record(&mut legacy, &mut vm).unwrap();
                let legacy = load_record(&legacy_bytes, &mut vm).unwrap();
                let legacy_value = data::json_parse_with_heap(&legacy.to_string(), &vm.heap()).unwrap();
                let (restored, _) = selected_model(&legacy_value, &mut vm).unwrap();
                assert_eq!(restored["feature_schema"], report["task"]["feature_schema"]);
                for corruption in ["cycle", "metric", "schema", "count"] {
                    let mut damaged = original.clone();
                    match corruption {
                        "cycle" => damaged["versions"][0]["model"]["trees"][0]["nodes"][0]["left"] = json!(0),
                        "metric" => damaged["versions"][0]["report"]["metrics"]["count"] = json!(25),
                        "schema" => damaged["versions"][0]["model"]["feature_schema"][0]["kind"] = json!("boolean"),
                        _ => damaged["versions"][0]["model"]["trees"][0]["nodes"][0]["count"] = json!(999),
                    }
                    // Recompute the digest so the semantic validation must reject it.
                    assert!(seal_record(&mut damaged, &mut vm).is_err(), "{family}/{corruption}");
                }
                let mut short = args.clone();
                short[2] = data::json_parse_with_heap("[]", &vm.heap()).unwrap();
                assert_eq!(input(&fit(&short, &mut vm).unwrap()).unwrap()["status"], "insufficient_evidence");
                let mut limited = Vm::new(&chunk);
                limited.set_instruction_limit(1);
                assert!(matches!(fit(&args, &mut limited), Err(LanaError::Limit)));
            }
        }
    }

    #[test]
    fn numeric_predictions_match_one_round_and_leaf_inputs_keep_the_schema() {
        let chunk = Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        for regression in [false, true] {
            let args = inputs(&vm, "boosted", regression, false);
            let report = input(&fit(&args, &mut vm).unwrap()).unwrap();
            let (prediction, _) = predicted(&report["model_or_rule"], &BTreeMap::from([("x".into(),json!(1))]), &mut vm).unwrap();
            if regression { assert!((prediction["value"].as_f64().unwrap() - 3.8).abs() < 1e-12); }
            else { assert!((prediction["probabilities"][1]["probability"].as_f64().unwrap() - sigmoid(-0.04)).abs() < 1e-12); }
        }
        let mut args = inputs(&vm, "tree", false, false);
        args[3] = data::json_parse_with_heap("{\"min_leaf\":2}", &vm.heap()).unwrap();
        let report = input(&fit(&args, &mut vm).unwrap()).unwrap();
        let model = &report["model_or_rule"];
        assert_eq!(model["trees"][0]["nodes"].as_array().unwrap().len(), 1);
        assert!(validate_features(model, None, &BTreeMap::from([("unknown".into(),json!(1))])).is_err());
        assert!(validate_features(model, None, &BTreeMap::from([("x".into(),json!("wrong"))])).is_err());
        assert!(validate_features(model, None, &BTreeMap::from([("x".into(),json!(1))])).is_ok());
        let mut overflow = inputs(&vm, "tree", true, false);
        let mut holdout = input(&overflow[2]).unwrap();
        holdout[0]["target"] = json!(f64::MAX);
        overflow[2] = data::json_parse_with_heap(&holdout.to_string(), &vm.heap()).unwrap();
        assert!(matches!(fit(&overflow, &mut vm), Err(LanaError::Schema)));
    }
}
