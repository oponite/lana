//! Optional, read-only semantic lookup over a frozen Brain corpus.

use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::path::Path;

use lana_bytecode::LanaError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

use crate::brain::Brain;
use crate::brain_memory::Memory;
use crate::information_codec::canonical;

const MAX_RECORDS: usize = 10_000;
const MAX_TOKENS: usize = 4_096;
const MAX_INDEX_BYTES: usize = 64 * 1024 * 1024;

fn digest(bytes: &[u8]) -> String {
    crate::sha256::sha256(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn f32_bits(value: f32) -> Result<String, LanaError> {
    if !value.is_finite() { return Err(LanaError::Schema); }
    Ok(format!("{:08x}", (if value == 0.0 { 0.0 } else { value }).to_bits()))
}

pub(crate) fn from_bits(value: &str) -> Result<f32, LanaError> {
    if value.len() != 8 || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        return Err(LanaError::Schema);
    }
    let number = f32::from_bits(u32::from_str_radix(value, 16).map_err(|_| LanaError::Schema)?);
    if !number.is_finite() || (number == 0.0 && number.to_bits() != 0) { return Err(LanaError::Schema); }
    Ok(number)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub id: String,
    pub source: String,
    pub typed_status: String,
    pub derivation_ref: String,
    pub text: String,
    pub revision: String,
    pub vector_bits: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    pub threshold_bits: String,
    pub margin_bits: String,
    pub development_sha256: String,
    pub heldout_sha256: String,
    pub gate_passed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Index {
    pub schema_version: u32,
    pub parameter_sha256: String,
    pub tokenizer_sha256: String,
    pub fact_revision: String,
    pub memory_revision: String,
    pub corpus_sha256: String,
    pub records: Vec<Record>,
    pub calibration: Option<Calibration>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabeledQuestion {
    pub id: String,
    pub question: String,
    pub relevant_ids: Vec<String>,
    pub forbidden_ids: Vec<String>,
}

fn record_without_vector(record: &Record) -> Json {
    json!({"id":record.id,"source":record.source,"typed_status":record.typed_status,
        "derivation_ref":record.derivation_ref,"text":record.text,"revision":record.revision})
}

fn corpus_digest(records: &[Record]) -> Result<String, LanaError> {
    Ok(digest(&canonical(&records.iter().map(record_without_vector).collect::<Vec<_>>())?))
}

fn corpus(brain: &Brain, memory: &Memory) -> Result<(Vec<Record>, BTreeMap<String, Json>), LanaError> {
    let mut records = Vec::new();
    let mut values = BTreeMap::new();
    let mut fact_revision = 0usize;
    for entry in &brain.memory {
        let Some(fact) = entry.strip_prefix("\0fact\0") else { continue; };
        let Some((key, value)) = fact.split_once('\0') else { return Err(LanaError::Schema); };
        fact_revision += 1;
        let id = format!("fact:{key}");
        records.push(Record { id: id.clone(), source: id.clone(), typed_status: "definite".into(),
            derivation_ref: id.clone(), text: format!("{key} {value}"), revision: fact_revision.to_string(),
            vector_bits: Vec::new() });
        values.insert(id, json!(value));
    }
    let current = memory.current_values()?;
    for (root, current) in memory.roots.iter().zip(&current) {
        let text = match current {
            crate::information_codec::Tagged::String { value } => value,
            crate::information_codec::Tagged::Definite { value } => match value.as_ref() {
                crate::information_codec::Tagged::String { value } => value,
                _ => continue,
            },
            _ => continue,
        };
        let id = format!("root:{}", root.id);
        let reference = memory.observations.iter().rev().find(|item| item.root_id == root.id)
            .map(|item| item.derivation_ref.clone()).unwrap_or_else(|| format!("brain/memory/{}", root.creation_revision));
        records.push(Record { id: id.clone(), source: root.source.clone(), typed_status: "definite".into(),
            derivation_ref: reference, text: format!("{} {text}", root.id),
            revision: memory.memory_revision.clone(), vector_bits: Vec::new() });
        values.insert(id, json!(text));
    }
    for alias in &memory.aliases {
        let id = format!("alias:{}:{}:{}", alias.creation_revision, alias.target_kind, alias.target_id);
        let target = format!("{}:{}", alias.target_kind, alias.target_id);
        let value = values.get(&target).cloned();
        let source = if alias.target_kind == "fact" { target.clone() } else {
            memory.roots.iter().find(|root| root.id == alias.target_id)
                .map(|root| root.source.clone()).unwrap_or(target.clone())
        };
        records.push(Record { id: id.clone(), source,
            typed_status: if value.is_some() { "definite" } else { "unresolved" }.into(),
            derivation_ref: target, text: alias.original_question.clone(),
            revision: alias.creation_revision.clone(), vector_bits: Vec::new() });
        if let Some(value) = value { values.insert(id, value); }
    }
    if records.len() > MAX_RECORDS { return Err(LanaError::Limit); }
    records.sort_by(|a, b| a.id.cmp(&b.id));
    if records.windows(2).any(|pair| pair[0].id == pair[1].id) { return Err(LanaError::Schema); }
    Ok((records, values))
}

pub(crate) fn vector(brain: &Brain, ids: &[usize], unknown_id: usize) -> Result<Vec<f32>, LanaError> {
    if ids.is_empty() || ids.iter().all(|id| *id == unknown_id) { return Err(LanaError::UnsupportedValue); }
    if ids.len() > MAX_TOKENS { return Err(LanaError::Limit); }
    let mut pooled = vec![0.0f64; brain.embedding_width];
    for id in ids {
        if *id >= brain.vocabulary { return Err(LanaError::Schema); }
        for (column, total) in pooled.iter_mut().enumerate() {
            *total += f64::from(brain.embedding[*id * brain.embedding_width + column]);
        }
    }
    let divisor = ids.len() as f64;
    let norm = pooled.iter().map(|value| (value / divisor).powi(2)).sum::<f64>().sqrt();
    if !norm.is_finite() || norm == 0.0 { return Err(LanaError::UnsupportedValue); }
    pooled.into_iter().map(|value| {
        let value = (value / divisor / norm) as f32;
        if value.is_finite() { Ok(value) } else { Err(LanaError::UnsupportedValue) }
    }).collect()
}

#[derive(Clone)]
struct Match<'a> { record: &'a Record, score: f32 }

impl Index {
    pub fn build(brain: &Brain, memory: &Memory, tokenizer_bytes: &[u8], unknown_id: usize,
        mut tokenize: impl FnMut(&str) -> Result<Vec<usize>, LanaError>) -> Result<Self, LanaError> {
        let (mut records, _) = corpus(brain, memory)?;
        if records.len().checked_mul(brain.embedding_width).and_then(|count| count.checked_mul(11))
            .is_none_or(|bytes| bytes > MAX_INDEX_BYTES) { return Err(LanaError::Limit); }
        let corpus_sha256 = corpus_digest(&records)?;
        for record in &mut records {
            let embedding = vector(brain, &tokenize(&record.text)?, unknown_id)?;
            record.vector_bits = embedding.into_iter().map(f32_bits).collect::<Result<_, _>>()?;
        }
        let result = Self { schema_version: 1, parameter_sha256: brain.parameter_sha256(),
            tokenizer_sha256: digest(tokenizer_bytes), fact_revision: brain.fact_revision().to_string(),
            memory_revision: memory.memory_revision.clone(), corpus_sha256, records, calibration: None };
        if result.bytes()?.len() > MAX_INDEX_BYTES { return Err(LanaError::Limit); }
        Ok(result)
    }

    pub fn bytes(&self) -> Result<Vec<u8>, LanaError> {
        let mut bytes = canonical(self)?;
        bytes.push(b'\n');
        if bytes.len() > MAX_INDEX_BYTES { return Err(LanaError::Limit); }
        Ok(bytes)
    }

    pub fn load(path: &Path, brain: &Brain, memory: &Memory, tokenizer_bytes: &[u8]) -> Result<Self, LanaError> {
        let mut bytes = Vec::new();
        std::fs::File::open(path).map_err(|_| LanaError::Io)?.take((MAX_INDEX_BYTES + 1) as u64)
            .read_to_end(&mut bytes).map_err(|_| LanaError::Io)?;
        if bytes.len() > MAX_INDEX_BYTES { return Err(LanaError::Limit); }
        let result: Self = serde_json::from_slice(&bytes).map_err(|_| LanaError::Schema)?;
        if result.bytes()? != bytes { return Err(LanaError::Schema); }
        result.validate(brain, memory, tokenizer_bytes)?;
        Ok(result)
    }

    pub fn validate(&self, brain: &Brain, memory: &Memory, tokenizer_bytes: &[u8]) -> Result<(), LanaError> {
        if self.schema_version != 1 || self.parameter_sha256 != brain.parameter_sha256() ||
            self.tokenizer_sha256 != digest(tokenizer_bytes) ||
            self.fact_revision != brain.fact_revision().to_string() ||
            self.memory_revision != memory.memory_revision { return Err(LanaError::InvalidState); }
        let (current, _) = corpus(brain, memory)?;
        if self.records.len() != current.len() || self.corpus_sha256 != corpus_digest(&current)? ||
            self.corpus_sha256 != corpus_digest(&self.records)? { return Err(LanaError::InvalidState); }
        for (saved, current) in self.records.iter().zip(&current) {
            if record_without_vector(saved) != record_without_vector(current) ||
                saved.vector_bits.len() != brain.embedding_width { return Err(LanaError::Schema); }
            let values = saved.vector_bits.iter().map(|bits| from_bits(bits)).collect::<Result<Vec<_>, _>>()?;
            let norm = values.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>();
            if !norm.is_finite() || (norm - 1.0).abs() > 1e-5 { return Err(LanaError::Schema); }
        }
        if let Some(calibration) = &self.calibration {
            from_bits(&calibration.threshold_bits)?;
            from_bits(&calibration.margin_bits)?;
            if [&calibration.development_sha256, &calibration.heldout_sha256].iter().any(|digest|
                digest.len() != 64 || !digest.bytes().all(|byte|
                    byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))) {
                return Err(LanaError::Schema);
            }
        }
        Ok(())
    }

    fn matches(&self, brain: &Brain, ids: &[usize], unknown_id: usize) -> Result<Vec<Match<'_>>, LanaError> {
        let query = vector(brain, ids, unknown_id)?;
        let mut matches = self.records.iter().map(|record| {
            let score = query.iter().zip(&record.vector_bits).map(|(left, right)|
                Ok(f64::from(*left) * f64::from(from_bits(right)?))).sum::<Result<f64, LanaError>>()? as f32;
            if !score.is_finite() { return Err(LanaError::Schema); }
            Ok(Match { record, score })
        }).collect::<Result<Vec<_>, LanaError>>()?;
        matches.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.record.id.cmp(&b.record.id)));
        Ok(matches)
    }

    pub fn query(&self, brain: &Brain, memory: &Memory, tokenizer_bytes: &[u8], ids: &[usize],
        unknown_id: usize) -> Result<Json, LanaError> {
        self.validate(brain, memory, tokenizer_bytes)?;
        let matches = match self.matches(brain, ids, unknown_id) {
            Ok(matches) => matches,
            Err(LanaError::UnsupportedValue) => Vec::new(),
            Err(error) => return Err(error),
        };
        let (_, values) = corpus(brain, memory)?;
        let top = matches.first();
        let margin = if matches.len() > 1 { matches[0].score - matches[1].score } else { f32::INFINITY };
        let thresholds = self.calibration.as_ref();
        let eligible = top.is_some_and(|top| thresholds.filter(|item| item.gate_passed).is_some_and(|calibration|
            top.score >= from_bits(&calibration.threshold_bits).unwrap_or(f32::INFINITY) &&
            margin >= from_bits(&calibration.margin_bits).unwrap_or(f32::INFINITY)));
        let tied = top.is_some_and(|top| matches.iter().take_while(|item| item.score == top.score)
            .all(|item| values.get(&item.record.id) == values.get(&top.record.id)));
        let exact = eligible && tied && top.is_some_and(|item| item.record.typed_status == "definite") &&
            top.and_then(|item| values.get(&item.record.id)).is_some();
        let candidates = matches.iter().take(5).map(|item| json!({"id":item.record.id,"score":item.score,
            "source":item.record.source,"derivation_ref":item.record.derivation_ref,
            "typed_status":item.record.typed_status})).collect::<Vec<_>>();
        Ok(json!({"schema_version":1,"resolution":if exact { "exact" } else if top.is_some() && !tied { "ambiguous" } else { "unsupported" },
            "answer":if exact { top.and_then(|item| values.get(&item.record.id)).cloned() } else { None },
            "candidates":candidates,"score":top.map(|item| item.score),
            "threshold":thresholds.and_then(|item| from_bits(&item.threshold_bits).ok()),
            "margin":if margin.is_finite() { Some(margin) } else { None },
            "margin_threshold":thresholds.and_then(|item| from_bits(&item.margin_bits).ok()),
            "corpus_revision":self.memory_revision,"brain_digest":self.parameter_sha256,
            "record_ids":matches.iter().take(5).map(|item| &item.record.id).collect::<Vec<_>>(),
            "source_refs":top.map(|item| vec![item.record.derivation_ref.clone()]).unwrap_or_default()}))
    }

    pub fn calibrate(&mut self, brain: &Brain, memory: &Memory,
        development_bytes: &[u8], heldout_bytes: &[u8], unknown_id: usize,
        mut tokenize: impl FnMut(&str) -> Result<Vec<usize>, LanaError>) -> Result<Json, LanaError> {
        if development_bytes.len() > 16 * 1024 * 1024 || heldout_bytes.len() > 16 * 1024 * 1024 {
            return Err(LanaError::Limit);
        }
        let development = labeled(development_bytes, &self.records)?;
        let heldout = labeled(heldout_bytes, &self.records)?;
        let development_ids = development.iter().map(|item| item.id.as_str()).collect::<HashSet<_>>();
        if heldout.iter().any(|item| development_ids.contains(item.id.as_str())) { return Err(LanaError::Schema); }
        let (_, values) = corpus(brain, memory)?;
        let mut scored_development = Vec::new();
        for item in &development {
            let ids = tokenize(&item.question)?;
            let matches = match self.matches(brain, &ids, unknown_id) {
                Ok(matches) => matches,
                Err(LanaError::UnsupportedValue) => Vec::new(),
                Err(error) => return Err(error),
            };
            scored_development.push(scored(matches, item, &values));
        }
        let mut thresholds = scored_development.iter().filter_map(|item| item.score).collect::<Vec<_>>();
        thresholds.sort_by(f32::total_cmp);
        thresholds.dedup();
        let mut margins = vec![0.0f32];
        margins.extend(scored_development.iter().filter_map(|item| item.margin.filter(|value| value.is_finite())));
        margins.sort_by(f32::total_cmp);
        margins.dedup();
        let mut choice: Option<(usize, f32, f32)> = None;
        for threshold in thresholds {
            for &margin in &margins {
                let accepted = scored_development.iter().filter(|item| item.accepts(threshold, margin)).collect::<Vec<_>>();
                if accepted.iter().any(|item| !item.correct) { continue; }
                let recall = accepted.len();
                if recall == 0 { continue; }
                if choice.is_none_or(|(best, best_threshold, best_margin)|
                    (recall, threshold, margin) > (best, best_threshold, best_margin)) {
                    choice = Some((recall, threshold, margin));
                }
            }
        }
        let mut scored_heldout = Vec::new();
        for item in &heldout {
            let ids = tokenize(&item.question)?;
            let matches = match self.matches(brain, &ids, unknown_id) {
                Ok(matches) => matches,
                Err(LanaError::UnsupportedValue) => Vec::new(),
                Err(error) => return Err(error),
            };
            scored_heldout.push(scored(matches, item, &values));
        }
        let mut baseline_recall = 0usize;
        let mut baseline_ambiguous = HashSet::new();
        let mut baseline_correct = Vec::new();
        for item in &heldout {
            let baseline = memory.grounded(brain, &item.question)?;
            if baseline["resolution"] == "ambiguous" { baseline_ambiguous.insert(item.id.as_str()); }
            let correct = baseline["resolution"] == "exact" && item.relevant_ids.iter()
                .any(|id| values.get(id) == baseline.get("answer"));
            if correct { baseline_recall += 1; }
            baseline_correct.push(correct);
        }
        let (development_recall, threshold, margin) = choice.unwrap_or((0, 1.0, 1.0));
        let accepted_heldout = scored_heldout.iter().filter(|item| item.accepts(threshold, margin)).collect::<Vec<_>>();
        let heldout_recall = accepted_heldout.iter().filter(|item| item.correct).count();
        let false_exact = accepted_heldout.len() - heldout_recall;
        let ambiguity_regressions = scored_heldout.iter().filter(|item|
            baseline_ambiguous.contains(item.id) && item.accepts(threshold, margin)).count();
        let gate_passed = choice.is_some() && false_exact == 0 && ambiguity_regressions == 0 &&
            heldout_recall > baseline_recall;
        let paired = heldout.iter().zip(&scored_heldout).zip(&baseline_correct).map(|((item, semantic), baseline)|
            json!({"id":item.id,"baseline_correct":baseline,
                "semantic_exact":semantic.accepts(threshold, margin),
                "semantic_correct":semantic.accepts(threshold, margin) && semantic.correct}))
            .collect::<Vec<_>>();
        self.calibration = Some(Calibration { threshold_bits: f32_bits(threshold)?, margin_bits: f32_bits(margin)?,
            development_sha256: digest(development_bytes), heldout_sha256: digest(heldout_bytes), gate_passed });
        self.bytes()?;
        Ok(json!({"status":if gate_passed { "active" } else { "inactive" },
            "development_recall":development_recall,"development_total":development.len(),
            "heldout_recall":heldout_recall,"heldout_total":heldout.len(),
            "heldout_false_exact":false_exact,"baseline_recall":baseline_recall,
            "ambiguity_regressions":ambiguity_regressions,
            "paired":paired,
            "parameter_sha256":self.parameter_sha256}))
    }
}

struct ScoredQuestion<'a> {
    id: &'a str,
    score: Option<f32>,
    margin: Option<f32>,
    correct: bool,
}

impl ScoredQuestion<'_> {
    fn accepts(&self, threshold: f32, margin: f32) -> bool {
        self.score.is_some_and(|score| score >= threshold) &&
            self.margin.is_some_and(|actual| actual >= margin)
    }
}

fn scored<'a>(matches: Vec<Match<'_>>, item: &'a LabeledQuestion,
    values: &BTreeMap<String, Json>) -> ScoredQuestion<'a> {
    let top = matches.first();
    let margin = if matches.len() > 1 { Some(matches[0].score - matches[1].score) }
        else if top.is_some() { Some(f32::INFINITY) } else { None };
    let tied = top.is_some_and(|top| matches.iter().take_while(|candidate| candidate.score == top.score)
        .all(|candidate| values.get(&candidate.record.id) == values.get(&top.record.id)));
    let correct = tied && top.is_some_and(|top| top.record.typed_status == "definite" &&
        item.relevant_ids.contains(&top.record.id) && !item.forbidden_ids.contains(&top.record.id) &&
        values.contains_key(&top.record.id));
    ScoredQuestion { id: &item.id, score: top.map(|top| top.score), margin, correct }
}

fn labeled(bytes: &[u8], records: &[Record]) -> Result<Vec<LabeledQuestion>, LanaError> {
    let known = records.iter().map(|record| record.id.as_str()).collect::<HashSet<_>>();
    let mut ids = HashSet::new();
    let mut result = Vec::new();
    for line in bytes.split(|byte| *byte == b'\n') {
        if line.is_empty() { continue; }
        let item: LabeledQuestion = serde_json::from_slice(line).map_err(|_| LanaError::Schema)?;
        if item.id.is_empty() || item.question.is_empty() || !ids.insert(item.id.clone()) ||
            item.relevant_ids.iter().chain(&item.forbidden_ids).any(|id| !known.contains(id.as_str())) ||
            item.relevant_ids.iter().collect::<HashSet<_>>().len() != item.relevant_ids.len() ||
            item.forbidden_ids.iter().collect::<HashSet<_>>().len() != item.forbidden_ids.len() ||
            item.relevant_ids.iter().any(|id| item.forbidden_ids.contains(id)) { return Err(LanaError::Schema); }
        result.push(item);
    }
    if result.is_empty() { return Err(LanaError::Schema); }
    Ok(result)
}
