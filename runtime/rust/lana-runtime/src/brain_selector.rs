//! Frozen-embedding logistic selection; durable evidence is never rewritten.
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Path;
use std::time::Instant;
use lana_bytecode::{Chunk, LanaError};
use lana_vm::Vm;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use crate::brain::Brain;
use crate::brain_index::{f32_bits, from_bits, vector, Index};
use crate::brain_memory::Memory;
use crate::information_codec::canonical;

const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_PAIRS: usize = 100_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pair { question_id: String, question: String, record_id: String, relevant: bool }

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Validation {
    question_id: String, record_id: String, relevant: bool, selected: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selector {
    schema_version: u32,
    parameter_sha256: String,
    tokenizer_sha256: String,
    fact_revision: String,
    memory_revision: String,
    corpus_sha256: String,
    feature_width: String,
    weight_bits: Vec<String>,
    bias_bits: String,
    train_pair_ids: Vec<[String; 2]>,
    validation_results: Vec<Validation>,
    calculation_version: u32,
    pub active: bool,
}

fn pairs(bytes: &[u8], index: &Index) -> Result<Vec<Pair>, LanaError> {
    if bytes.len() > MAX_BYTES { return Err(LanaError::Limit); }
    let known = index.records.iter().map(|record| record.id.as_str()).collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    let mut questions = BTreeMap::new();
    let mut result = Vec::new();
    for line in bytes.split(|byte| *byte == b'\n').filter(|line| !line.is_empty()) {
        if result.len() == MAX_PAIRS { return Err(LanaError::Limit); }
        let pair: Pair = serde_json::from_slice(line).map_err(|_| LanaError::Schema)?;
        if pair.question_id.trim().is_empty() || pair.question.trim().is_empty() ||
            !known.contains(pair.record_id.as_str()) ||
            !seen.insert((pair.question_id.clone(), pair.record_id.clone())) ||
            questions.insert(pair.question_id.clone(), pair.question.clone())
                .is_some_and(|old| old != pair.question) { return Err(LanaError::Schema); }
        result.push(pair);
    }
    if result.is_empty() { return Err(LanaError::Schema); }
    Ok(result)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 { 1.0 / (1.0 + (-value).exp()) }
    else { let exp = value.exp(); exp / (1.0 + exp) }
}

fn probability(weights: &[f32], bias: f32, query: &[f32], record: &[f32]) -> Result<f32, LanaError> {
    let mut value = bias;
    for ((weight, q), r) in weights.iter().zip(query).zip(record) { value += weight * (q * r); }
    if !value.is_finite() { return Err(LanaError::Schema); }
    Ok(sigmoid(value))
}

fn query_vector(brain: &Brain, question: &str, unknown: usize, vm: &mut Vm,
    tokenize: &mut impl FnMut(&str) -> Result<Vec<usize>, LanaError>) -> Result<Vec<f32>, LanaError> {
    let ids = tokenize(question)?;
    vm.charge_bounded_work((ids.len() as u64).saturating_mul(brain.embedding_width as u64))?;
    vector(brain, &ids, unknown)
}

fn record_vectors(index: &Index) -> Result<Vec<Vec<f32>>, LanaError> {
    index.records.iter().map(|record| record.vector_bits.iter().map(|bits| from_bits(bits)).collect()).collect()
}

fn context_tokens(question: &str, context: &[Json], vm: &mut Vm,
    tokenize: &mut impl FnMut(&str) -> Result<Vec<usize>, LanaError>) -> Result<usize, LanaError> {
    if context.len() > 64 { return Err(LanaError::Limit); }
    let mut count = tokenize(question)?.len();
    for entry in context {
        let bytes = canonical(entry)?;
        if bytes.len() > MAX_BYTES { return Err(LanaError::Limit); }
        count = count.checked_add(tokenize(std::str::from_utf8(&bytes).map_err(|_| LanaError::Schema)?)?.len())
            .ok_or(LanaError::Limit)?;
        if count > 4096 { return Err(LanaError::Limit); }
    }
    vm.charge_bounded_work(count as u64)?;
    if count > 4096 { return Err(LanaError::Limit); }
    Ok(count)
}

struct Selection { ids: Vec<String>, context: Vec<Json>, tokens: usize }

fn select(index: &Index, weights: &[f32], bias: f32, query: &[f32], records: &[Vec<f32>],
    required: &[Json], vm: &mut Vm) -> Result<Selection, LanaError> {
    vm.charge_bounded_work((records.len() as u64).saturating_mul(weights.len() as u64 + 1))?;
    let mut ids = Vec::new();
    // Keep original evidence objects, including observations and forecast references
    // that have no text embedding in the semantic corpus.
    for entry in required {
        if let Some(id) = entry["record_id"].as_str() {
            if !ids.iter().any(|selected| selected == id) && index.records.iter().any(|record| record.id == id) {
                ids.push(id.to_owned());
            }
        }
    }
    let mut ranked = index.records.iter().zip(records).map(|(record, vector)|
        Ok((record, probability(weights, bias, query, vector)?))).collect::<Result<Vec<_>, LanaError>>()?;
    ranked.sort_by(|(a, sa), (b, sb)| sb.total_cmp(sa).then_with(|| a.id.cmp(&b.id)));
    let mut context = required.to_vec();
    for (record, score) in ranked {
        if score < 0.5 || ids.contains(&record.id) { continue; }
        ids.push(record.id.clone());
        context.push(json!({"record_id":record.id,"source":record.source,"typed_status":record.typed_status,
            "derivation_ref":record.derivation_ref,"text":record.text,"revision":record.revision,"score":score}));
        if context.len() > 64 { return Err(LanaError::Limit); }
    }
    Ok(Selection { ids, context, tokens: 0 })
}

impl Selector {
    pub fn bytes(&self) -> Result<Vec<u8>, LanaError> {
        let mut bytes = canonical(self)?;
        bytes.push(b'\n');
        if bytes.len() > MAX_BYTES { return Err(LanaError::Limit); }
        Ok(bytes)
    }

    pub fn validate(&self, index: &Index) -> Result<(), LanaError> {
        if self.schema_version != 1 || self.calculation_version != 1 { return Err(LanaError::Schema); }
        if self.parameter_sha256 != index.parameter_sha256 || self.tokenizer_sha256 != index.tokenizer_sha256 ||
            self.fact_revision != index.fact_revision || self.memory_revision != index.memory_revision ||
            self.corpus_sha256 != index.corpus_sha256 { return Err(LanaError::InvalidState); }
        let width = usize::try_from(crate::information_codec::revision(&self.feature_width)?)
            .map_err(|_| LanaError::Schema)?;
        if width == 0 || self.weight_bits.len() != width ||
            index.records.iter().any(|record| record.vector_bits.len() != width) { return Err(LanaError::Schema); }
        for bits in &self.weight_bits { from_bits(bits)?; }
        from_bits(&self.bias_bits)?;
        if self.train_pair_ids.len() > MAX_PAIRS || self.validation_results.len() > MAX_PAIRS { return Err(LanaError::Limit); }
        if self.train_pair_ids.is_empty() || self.validation_results.is_empty() { return Err(LanaError::Schema); }
        let known = index.records.iter().map(|record| record.id.as_str()).collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        let mut train_questions = BTreeSet::new();
        let mut train_records = BTreeSet::new();
        for [question, record] in &self.train_pair_ids {
            if question.trim().is_empty() || !known.contains(record.as_str()) || !seen.insert((question, record)) {
                return Err(LanaError::Schema);
            }
            train_questions.insert(question);
            train_records.insert(record);
        }
        seen.clear();
        for item in &self.validation_results {
            if item.question_id.trim().is_empty() || !known.contains(item.record_id.as_str()) ||
                train_questions.contains(&item.question_id) || train_records.contains(&item.record_id) ||
                !seen.insert((&item.question_id, &item.record_id)) { return Err(LanaError::Schema); }
        }
        if self.active && (!self.validation_results.iter().any(|item| item.relevant) ||
            self.validation_results.iter().any(|item| item.relevant && !item.selected)) { return Err(LanaError::Schema); }
        self.bytes()?;
        Ok(())
    }

    pub fn load(path: &Path, index: &Index) -> Result<Self, LanaError> {
        let mut bytes = Vec::new();
        std::fs::File::open(path).map_err(|_| LanaError::Io)?.take((MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes).map_err(|_| LanaError::Io)?;
        if bytes.len() > MAX_BYTES { return Err(LanaError::Limit); }
        let result: Self = serde_json::from_slice(&bytes).map_err(|_| LanaError::Schema)?;
        if result.bytes()? != bytes { return Err(LanaError::Schema); }
        result.validate(index)?;
        if !result.active { return Err(LanaError::InvalidState); }
        Ok(result)
    }

    pub fn fit(brain: &Brain, memory: &Memory, index: &Index, tokenizer: &[u8], train: &[u8], valid: &[u8],
        unknown: usize, mut tokenize: impl FnMut(&str) -> Result<Vec<usize>, LanaError>) -> Result<(Self, Json), LanaError> {
        let started = Instant::now();
        index.validate(brain, memory, tokenizer)?;
        let train = pairs(train, index)?;
        let valid = pairs(valid, index)?;
        let train_questions = train.iter().map(|item| &item.question_id).collect::<BTreeSet<_>>();
        let train_records = train.iter().map(|item| &item.record_id).collect::<BTreeSet<_>>();
        if valid.iter().any(|item| train_questions.contains(&item.question_id) || train_records.contains(&item.record_id)) {
            return Err(LanaError::Schema);
        }
        let width = brain.embedding_width;
        if (train_questions.len() + index.records.len()).checked_mul(width).and_then(|n| n.checked_mul(4))
            .is_none_or(|bytes| bytes > 256 * 1024 * 1024) { return Err(LanaError::Limit); }
        let chunk = Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        vm.charge_bounded_work((train.len() as u64).saturating_mul(width as u64).saturating_mul(10))?;
        let records = record_vectors(index)?;
        let record_indices = index.records.iter().enumerate().map(|(i, record)| (record.id.as_str(), i))
            .collect::<BTreeMap<_, _>>();
        let mut queries = BTreeMap::new();
        for item in &train {
            if !queries.contains_key(&item.question_id) {
                queries.insert(item.question_id.clone(), query_vector(brain, &item.question, unknown, &mut vm, &mut tokenize)?);
            }
        }
        let mut weights = vec![0.0f32; width];
        let mut bias = 0.0f32;
        for _ in 0..5 {
            for item in &train {
                let query = &queries[&item.question_id];
                let record = &records[record_indices[item.record_id.as_str()]];
                let error = probability(&weights, bias, query, record)? - if item.relevant { 1.0 } else { 0.0 };
                for ((weight, q), r) in weights.iter_mut().zip(query).zip(record) {
                    *weight -= 0.01 * (error * (q * r) + 0.001 * *weight);
                    if !weight.is_finite() { return Err(LanaError::Schema); }
                }
                bias -= 0.01 * error;
                if !bias.is_finite() { return Err(LanaError::Schema); }
            }
        }
        drop(queries);
        let questions = valid.iter().map(|item| (item.question_id.as_str(), item.question.as_str())).collect::<BTreeMap<_, _>>();
        let mut selected = BTreeMap::new();
        let mut paired = Vec::new();
        let mut baseline_count = 0usize;
        let mut selected_count = 0usize;
        let mut outcome_regressions = 0usize;
        for (id, question) in questions {
            let required = memory.required_context(brain, question)?;
            let baseline_tokens = context_tokens(question, &required, &mut vm, &mut tokenize)?;
            let baseline = memory.grounded(brain, question)?;
            let query = query_vector(brain, question, unknown, &mut vm, &mut tokenize)?;
            let mut choice = select(index, &weights, bias, &query, &records, &required, &mut vm)?;
            choice.tokens = context_tokens(question, &choice.context, &mut vm, &mut tokenize)?;
            // Answer derivation remains deterministic over the retained original evidence.
            let retained = choice.context.starts_with(&required);
            if !retained { outcome_regressions += 1; }
            paired.push(json!({"question_id":id,"baseline_count":required.len(),"selected_count":choice.context.len(),
                "baseline_tokens":baseline_tokens,"selected_tokens":choice.tokens,
                "baseline_outcome":{"resolution":baseline["resolution"],"answer":baseline["answer"]},
                "selected_outcome":{"resolution":if retained { baseline["resolution"].clone() } else { json!("unsupported") },
                    "answer":if retained { baseline["answer"].clone() } else { Json::Null }},
                "required_evidence_retained":retained}));
            baseline_count += required.len();
            selected_count += choice.context.len();
            selected.insert(id.to_string(), choice.ids);
        }
        let validation_results = valid.iter().map(|item| Validation {
            question_id:item.question_id.clone(), record_id:item.record_id.clone(), relevant:item.relevant,
            selected:selected[&item.question_id].contains(&item.record_id),
        }).collect::<Vec<_>>();
        let relevant = validation_results.iter().filter(|item| item.relevant).count();
        let retained = validation_results.iter().filter(|item| item.relevant && item.selected).count();
        let active = relevant > 0 && relevant == retained && outcome_regressions == 0 && selected_count <= baseline_count;
        let result = Self { schema_version:1, parameter_sha256:index.parameter_sha256.clone(),
            tokenizer_sha256:index.tokenizer_sha256.clone(), fact_revision:index.fact_revision.clone(),
            memory_revision:index.memory_revision.clone(), corpus_sha256:index.corpus_sha256.clone(),
            feature_width:width.to_string(), weight_bits:weights.into_iter().map(f32_bits).collect::<Result<_, _>>()?,
            bias_bits:f32_bits(bias)?, train_pair_ids:train.iter().map(|item| [item.question_id.clone(),item.record_id.clone()]).collect(),
            validation_results, calculation_version:1, active };
        result.validate(index)?;
        let count = paired.len() as f64;
        let report = json!({"status":if active { "active" } else { "inactive" },"relevant":relevant,"retained":retained,
            "validation_pairs":valid.len(),"validation_questions":paired.len(),"baseline_count":baseline_count,
            "selected_count":selected_count,"baseline_average":baseline_count as f64 / count,
            "selected_average":selected_count as f64 / count,"answer_regressions":outcome_regressions,"paired":paired,
            "elapsed_ns":started.elapsed().as_nanos().to_string(),"work_units":vm.instruction_count(),
            "parameter_sha256_before":index.parameter_sha256,"parameter_sha256_after":brain.parameter_sha256()});
        Ok((result, report))
    }

    pub fn query(&self, brain: &Brain, memory: &Memory, index: &Index, tokenizer: &[u8], question: &str,
        unknown: usize, mut tokenize: impl FnMut(&str) -> Result<Vec<usize>, LanaError>) -> Result<Json, LanaError> {
        index.validate(brain, memory, tokenizer)?;
        self.validate(index)?;
        if !self.active { return Err(LanaError::InvalidState); }
        let chunk = Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        let ids = tokenize(question)?;
        vm.charge_bounded_work((ids.len() as u64).saturating_mul(brain.embedding_width as u64))?;
        let query = vector(brain, &ids, unknown)?;
        let required = memory.required_context(brain, question)?;
        context_tokens(question, &required, &mut vm, &mut tokenize)?;
        let weights = self.weight_bits.iter().map(|bits| from_bits(bits)).collect::<Result<Vec<_>, _>>()?;
        let choice = select(index, &weights, from_bits(&self.bias_bits)?, &query, &record_vectors(index)?, &required, &mut vm)?;
        let tokens = context_tokens(question, &choice.context, &mut vm, &mut tokenize)?;
        vm.charge_bounded_work(((ids.len() + index.records.len()) as u64)
            .saturating_mul(brain.embedding_width as u64))?;
        let mut answer = index.query(brain, memory, tokenizer, &ids, unknown)?;
        let top = answer["candidates"].as_array().and_then(|items| items.first()).and_then(|item| item["id"].as_str());
        if answer["resolution"] == "exact" && !top.is_some_and(|id| choice.ids.iter().any(|selected| selected == id)) {
            answer["resolution"] = json!("unsupported");
            answer["answer"] = Json::Null;
            answer["source_refs"] = json!([]);
        }
        if let Some(candidates) = answer["candidates"].as_array_mut() {
            candidates.retain(|item| item["id"].as_str().is_some_and(|id| choice.ids.iter().any(|selected| selected == id)));
        }
        if answer["resolution"] != "exact" {
            answer["source_refs"] = answer["candidates"].as_array().unwrap().iter()
                .map(|item| item["derivation_ref"].clone()).collect::<Vec<_>>().into();
        }
        answer["record_ids"] = answer["candidates"].as_array().unwrap().iter().map(|item| item["id"].clone()).collect::<Vec<_>>().into();
        answer["selected_record_ids"] = json!(choice.ids);
        answer["selected_context"] = json!(choice.context);
        answer["context_tokens"] = json!(tokens);
        answer["selector_calculation_version"] = json!(self.calculation_version);
        Ok(answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tokens(text: &str) -> Result<Vec<usize>, LanaError> { Ok(text.split_whitespace().map(|_| 1).collect()) }
    fn setup() -> (Brain, Memory, Index) {
        let mut brain = Brain::new(2, 1, 2, 7).unwrap();
        brain.embedding.fill(1.0);
        brain.remember_fact("train", "value").unwrap();
        brain.remember_fact("answer", "value").unwrap();
        let mut memory = Memory::empty();
        memory.alias("fact", "answer", "question").unwrap();
        let index = Index::build(&brain, &memory, b"tokenizer", 0, tokens).unwrap();
        (brain, memory, index)
    }
    const TRAIN: &[u8] = b"{\"question_id\":\"train\",\"question\":\"training\",\"record_id\":\"fact:train\",\"relevant\":false}\n";
    const VALID: &[u8] = b"{\"question_id\":\"valid\",\"question\":\"question\",\"record_id\":\"fact:answer\",\"relevant\":true}\n";
    #[test]
    fn frozen_fit_gate_replay_and_negative_holdout() {
        let (brain, memory, index) = setup();
        let before = brain.clone();
        let (model, report) = Selector::fit(&brain, &memory, &index, b"tokenizer", TRAIN, VALID, 0, tokens).unwrap();
        assert!(model.active);
        assert_eq!(report["retained"], 1);
        assert_eq!(report["selected_count"], report["baseline_count"]);
        assert_eq!(report["paired"][0]["selected_outcome"], report["paired"][0]["baseline_outcome"]);
        assert_eq!(brain, before);
        let replay = Selector::fit(&brain, &memory, &index, b"tokenizer", TRAIN, VALID, 0, tokens).unwrap().0;
        assert_eq!(model.bytes().unwrap(), replay.bytes().unwrap());
        // Five scalar F32 SGD updates, independently calculated for q*r = 1.
        let mut weight = 0.0f32;
        let mut bias = 0.0f32;
        for _ in 0..5 {
            let exp = (bias + weight).exp();
            let gradient = exp / (1.0 + exp);
            weight -= 0.01 * (gradient + 0.001 * weight);
            bias -= 0.01 * gradient;
        }
        assert!((from_bits(&model.weight_bits[0]).unwrap() - weight).abs() < 1e-7);
        assert!((from_bits(&model.bias_bits).unwrap() - bias).abs() < 1e-7);
        let answer = model.query(&brain, &memory, &index, b"tokenizer", "question", 0, tokens).unwrap();
        assert_eq!(answer["selected_record_ids"], json!(["fact:answer"]));
        let absent = String::from_utf8(VALID.to_vec()).unwrap().replace("question\":\"question", "question\":\"absent");
        let (inactive, report) = Selector::fit(&brain, &memory, &index, b"tokenizer", TRAIN, absent.as_bytes(), 0, tokens).unwrap();
        assert!(!inactive.active);
        assert_eq!(report["retained"], 0);
        assert_eq!(inactive.query(&brain, &memory, &index, b"tokenizer", "question", 0, tokens), Err(LanaError::InvalidState));
        let positive = String::from_utf8(TRAIN.to_vec()).unwrap().replace("false", "true");
        let (inactive, report) = Selector::fit(&brain, &memory, &index, b"tokenizer", positive.as_bytes(), VALID, 0, tokens).unwrap();
        assert!(!inactive.active);
        assert!(report["selected_count"].as_u64().unwrap() > report["baseline_count"].as_u64().unwrap());
        let mut stale = index.clone();
        stale.memory_revision = "999".into();
        assert_eq!(model.validate(&stale), Err(LanaError::InvalidState));
        let mut corrupt = model.clone();
        corrupt.weight_bits[0] = "7f800000".into();
        assert_eq!(corrupt.validate(&index), Err(LanaError::Schema));
        assert!(Selector::fit(&brain, &memory, &index, b"tokenizer", TRAIN, TRAIN, 0, tokens).is_err());
        let contradictory = [TRAIN, positive.as_bytes()].concat();
        assert!(Selector::fit(&brain, &memory, &index, b"tokenizer", &contradictory, VALID, 0, tokens).is_err());
    }

    #[test]
    fn selection_keeps_late_action_evidence_and_conflicts_and_enforces_caps() {
        use crate::brain_memory::ForecastInput;
        use crate::information_codec::Tagged;
        let (mut brain, mut memory, _) = setup();
        for i in 0..80 { brain.remember_fact(&format!("noise-{i}"), "value").unwrap(); }
        memory.add("late", "fact:late", Tagged::Definite { value:Box::new(Tagged::String { value:"value".into() }) }).unwrap();
        memory.forecast_add("late-forecast", "answer", 20, ForecastInput { created_at:10,
            labels:vec!["yes".into(),"no".into()], probabilities:vec![0.5,0.5], evidence_refs:vec!["late".into()] }).unwrap();
        let index = Index::build(&brain, &memory, b"tokenizer", 0, tokens).unwrap();
        let (model, _) = Selector::fit(&brain, &memory, &index, b"tokenizer", TRAIN, VALID, 0, tokens).unwrap();
        assert!(model.active);
        let answer = model.query(&brain, &memory, &index, b"tokenizer", "question", 0, tokens).unwrap();
        assert_eq!(answer["selected_context"].as_array().unwrap().len(), 3);
        assert_eq!(answer["selected_record_ids"], json!(["fact:answer","root:late"]));
        assert_eq!(answer["selected_context"][2]["selected_by_forecast"], "late-forecast");
        let positive = String::from_utf8(TRAIN.to_vec()).unwrap().replace("false", "true");
        assert!(matches!(Selector::fit(&brain, &memory, &index, b"tokenizer", positive.as_bytes(), VALID, 0, tokens), Err(LanaError::Limit)));
        assert_eq!(model.query(&brain, &memory, &index, b"tokenizer", &"word ".repeat(4097), 0, tokens), Err(LanaError::Limit));
        memory.alias("fact", "train", "question").unwrap();
        let index = Index::build(&brain, &memory, b"tokenizer", 0, tokens).unwrap();
        let (model, report) = Selector::fit(&brain, &memory, &index, b"tokenizer", TRAIN, VALID, 0, tokens).unwrap();
        assert!(model.active);
        assert_eq!(report["paired"][0]["selected_outcome"]["resolution"], "ambiguous");
        assert_eq!(report["paired"][0]["baseline_outcome"], report["paired"][0]["selected_outcome"]);
        let chunk = Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        vm.set_instruction_limit(0);
        assert_eq!(query_vector(&brain, "question", 0, &mut vm, &mut tokens), Err(LanaError::Limit));
    }

    #[test]
    fn semantic_exactness_cannot_be_created_by_selection() {
        use crate::brain_index::Calibration;
        let (mut brain, _, _) = setup();
        let memory = Memory::empty();
        let valid = String::from_utf8(VALID.to_vec()).unwrap().replace("question\":\"question", "question\":\"answer");
        let mut index = Index::build(&brain, &memory, b"tokenizer", 0, tokens).unwrap();
        index.calibration = Some(Calibration { threshold_bits:f32_bits(-1.0).unwrap(),
            margin_bits:f32_bits(0.0).unwrap(), development_sha256:"0".repeat(64),
            heldout_sha256:"1".repeat(64), gate_passed:true });
        let (model, _) = Selector::fit(&brain, &memory, &index, b"tokenizer", TRAIN, valid.as_bytes(), 0, tokens).unwrap();
        let kept = model.query(&brain, &memory, &index, b"tokenizer", "answer", 0, tokens).unwrap();
        assert_eq!(kept["resolution"], "exact");
        let omitted = model.query(&brain, &memory, &index, b"tokenizer", "value", 0, tokens).unwrap();
        assert_eq!(omitted["resolution"], "unsupported");
        assert_eq!(omitted["answer"], Json::Null);
        brain.remember_fact("conflict", "different").unwrap();
        let mut conflicting = Index::build(&brain, &memory, b"tokenizer", 0, tokens).unwrap();
        conflicting.calibration = index.calibration;
        let (model, _) = Selector::fit(&brain, &memory, &conflicting, b"tokenizer", TRAIN, valid.as_bytes(), 0, tokens).unwrap();
        let answer = model.query(&brain, &memory, &conflicting, b"tokenizer", "answer", 0, tokens).unwrap();
        assert_eq!(answer["resolution"], "ambiguous");
        assert_eq!(answer["answer"], Json::Null);
        let huge_context = |text: &str| if text.starts_with('{') { Ok(vec![1;4097]) } else { tokens(text) };
        assert_eq!(model.query(&brain, &memory, &conflicting, b"tokenizer", "answer", 0, huge_context), Err(LanaError::Limit));
    }
}
