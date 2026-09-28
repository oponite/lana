//! Durable finite roots and observations, validated by complete isolated Core replay.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use lana_bytecode::{Chunk, LanaError};
use lana_vm::{Vm, Value};
use lana_vm::value::Map;
use serde::{Deserialize, Serialize};
use crate::information_codec::{canonical, revision, Tagged};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Evidence { Value(Tagged), Joint(#[serde(deserialize_with = "unique_evidence")] BTreeMap<String, Tagged>) }

fn unique_evidence<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<BTreeMap<String, Tagged>, D::Error> {
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = BTreeMap<String, Tagged>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result { formatter.write_str("unique joint evidence names") }
        fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut result = BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, Tagged>()? {
                if result.insert(key, value).is_some() { return Err(serde::de::Error::custom("duplicate evidence name")); }
            }
            Ok(result)
        }
    }
    deserializer.deserialize_map(Visitor)
}

impl Evidence {
    fn live(&self, vm: &mut Vm) -> Result<Value, LanaError> {
        match self {
            Self::Value(value) => value.to_live(vm),
            Self::Joint(entries) => {
                let mut map = Map::new(&vm.heap(), entries.len())?;
                for (name, value) in entries { map.set(Arc::from(name.as_str()), value.to_live(vm)?, false)?; }
                Ok(Value::map(Arc::new(Mutex::new(map))))
            }
        }
    }

    pub fn normalized(&self) -> Result<Self, LanaError> {
        let chunk = Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        match self {
            Self::Value(value) => Ok(Self::Value(value.snapshot(&value.to_live(&mut vm)?)?)),
            Self::Joint(entries) => Ok(Self::Joint(entries.iter().map(|(name, value)|
                Ok((name.clone(), value.snapshot(&value.to_live(&mut vm)?)?))).collect::<Result<_, LanaError>>()?)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Root {
    pub id: String,
    pub source: String,
    pub initial_information: Tagged,
    pub creation_revision: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub id: String,
    pub root_id: String,
    pub evidence: Evidence,
    pub memory_revision: String,
    pub derivation_ref: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Derivation {
    pub id: String,
    pub operation: String,
    pub revision: String,
    pub input_ids: Vec<String>,
    pub source: String,
    pub exactness: String,
    pub outcome: Tagged,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Alias {
    pub normalized_question: String,
    pub original_question: String,
    pub target_kind: String,
    pub target_id: String,
    pub creation_revision: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastInput {
    pub created_at: u64,
    pub labels: Vec<String>,
    pub probabilities: Vec<f64>,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastScore {
    pub observed_label: String,
    pub observed_at: u64,
    pub score: f64,
    pub score_revision: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Forecast {
    pub id: String,
    pub target: String,
    pub horizon: u64,
    #[serde(flatten)]
    pub input: ForecastInput,
    pub creation_revision: String,
    pub score: Option<ForecastScore>,
}

fn forecast_error(input: &ForecastInput, label: &str) -> Result<f64, LanaError> {
    let index = input.labels.iter().position(|item| item == label).ok_or(LanaError::InvalidParameters)?;
    Ok(input.probabilities.iter().enumerate().map(|(i, probability)|
        (probability - f64::from(i == index)).powi(2)).sum())
}

fn validate_forecast(forecast: &Forecast) -> Result<(), LanaError> {
    id(&forecast.id)?;
    id(&forecast.target)?;
    if forecast.horizon <= forecast.input.created_at ||
        !(2..=128).contains(&forecast.input.labels.len()) ||
        forecast.input.labels.len() != forecast.input.probabilities.len() ||
        forecast.input.evidence_refs.len() > 64 { return Err(LanaError::Schema); }
    let mut labels = BTreeSet::new();
    for label in &forecast.input.labels {
        if label.is_empty() || !labels.insert(label) { return Err(LanaError::Schema); }
    }
    if forecast.input.probabilities.iter().any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability)) ||
        (forecast.input.probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-12 { return Err(LanaError::Schema); }
    if let Some(score) = &forecast.score {
        if score.observed_at < forecast.horizon ||
            score.score != forecast_error(&forecast.input, &score.observed_label).map_err(|_| LanaError::Schema)? {
            return Err(LanaError::Schema);
        }
    }
    Ok(())
}

pub fn normalize(question: &str) -> Result<String, LanaError> {
    let normalized = question.split(|c: char| c == ' ' || ('\t'..='\r').contains(&c))
        .filter(|part| !part.is_empty()).collect::<Vec<_>>().join(" ").to_ascii_lowercase();
    if normalized.is_empty() || normalized.len() > 4096 { return Err(LanaError::InvalidParameters); }
    Ok(normalized)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    pub schema_version: u32,
    pub memory_revision: String,
    pub roots: Vec<Root>,
    pub observations: Vec<Observation>,
    pub derivations: Vec<Derivation>,
    pub aliases: Vec<Alias>,
    pub forecasts: Vec<Forecast>,
    pub final_digest: String,
}

struct Replay { values: Vec<Tagged>, derivations: Vec<Derivation>, digest: String }

fn id(value: &str) -> Result<(), LanaError> {
    if value.is_empty() || value.len() > 128 { return Err(LanaError::Schema); }
    Ok(())
}

fn digest(value: &impl Serialize) -> Result<String, LanaError> {
    Ok(crate::sha256::sha256(&canonical(value)?).iter().map(|byte| format!("{byte:02x}")).collect())
}

impl Memory {
    pub fn empty() -> Self {
        Self { schema_version: 1, memory_revision: "0".into(), roots: Vec::new(), observations: Vec::new(),
            derivations: Vec::new(), aliases: Vec::new(), forecasts: Vec::new(),
            final_digest: digest(&serde_json::json!({"roots":[]})).unwrap() }
    }

    pub fn load(bytes: &[u8]) -> Result<Self, LanaError> {
        if bytes.is_empty() { return Ok(Self::empty()); }
        if bytes.len() > 256 * 1024 * 1024 { return Err(LanaError::Limit); }
        let record: Self = serde_json::from_slice(bytes).map_err(|_| LanaError::Schema)?;
        if canonical(&record)? != bytes { return Err(LanaError::Schema); }
        let replay = record.replay()?;
        if replay.derivations != record.derivations { return Err(LanaError::Schema); }
        if replay.digest != record.final_digest { return Err(LanaError::Integrity); }
        Ok(record)
    }

    fn replay(&self) -> Result<Replay, LanaError> {
        if self.schema_version != 1 { return Err(LanaError::Schema); }
        if self.roots.len() > 10_000 || self.observations.len() > 100_000 || self.derivations.len() > 100_000 ||
            self.roots.len() + self.observations.len() > 100_000 { return Err(LanaError::Limit); }
        let events = self.roots.len() + self.observations.len() + self.aliases.len() +
            self.forecasts.len() + self.forecasts.iter().filter(|item| item.score.is_some()).count();
        if events > 100_000 { return Err(LanaError::Limit); }
        if revision(&self.memory_revision)? != events as u64 { return Err(LanaError::Schema); }
        let mut forecast_events = BTreeMap::new();
        let mut forecast_ids = BTreeSet::new();
        for (index, forecast) in self.forecasts.iter().enumerate() {
            validate_forecast(forecast)?;
            if !forecast_ids.insert(&forecast.id) { return Err(LanaError::Schema); }
            let created = revision(&forecast.creation_revision)?;
            if created == 0 || created > events as u64 || forecast_events.insert(created, (index, false)).is_some() {
                return Err(LanaError::Schema);
            }
            if let Some(score) = &forecast.score {
                let scored = revision(&score.score_revision)?;
                if scored <= created || scored > events as u64 || forecast_events.insert(scored, (index, true)).is_some() {
                    return Err(LanaError::Schema);
                }
            }
        }
        let chunk = Chunk::new(5, 0);
        let mut live = Vec::<(Vm, Value)>::new();
        let mut values = Vec::new();
        let mut references = Vec::new();
        let mut indices = BTreeMap::new();
        let mut ids = BTreeSet::new();
        let mut evidence_ids = BTreeSet::new();
        let mut derivations = Vec::new();
        let (mut root_index, mut observation_index, mut alias_index) = (0, 0, 0);
        let mut alias_targets = BTreeSet::new();
        for next in 1..=events {
            let derivation_id = format!("brain/memory/{next}");
            if self.roots.get(root_index).is_some_and(|root| root.creation_revision == next.to_string()) {
                let root = &self.roots[root_index];
                id(&root.id)?;
                if !ids.insert(root.id.clone()) || root.source.len() > 4096 { return Err(LanaError::Schema); }
                if !matches!(root.initial_information, Tagged::Definite { .. } | Tagged::Possibility { .. } |
                    Tagged::Distribution { .. } | Tagged::FiniteJoint { .. }) { return Err(LanaError::UnsupportedValue); }
                let mut vm = Vm::new(&chunk);
                let value = root.initial_information.to_live(&mut vm)?;
                let snapshot = root.initial_information.snapshot(&value)?;
                if snapshot != root.initial_information { return Err(LanaError::Schema); }
                let value = vm.information_root(&value)?;
                live.push((vm, value));
                values.push(snapshot.clone());
                references.push(derivation_id.clone());
                indices.insert(root.id.clone(), root_index);
                evidence_ids.insert(root.id.clone());
                evidence_ids.insert(derivation_id.clone());
                derivations.push(Derivation { id: derivation_id, operation: "root".into(), revision: "0".into(),
                    input_ids: Vec::new(), source: root.source.clone(), exactness: "exact_law".into(), outcome: snapshot, reason: None });
                root_index += 1;
            } else if self.observations.get(observation_index).is_some_and(|item| item.memory_revision == next.to_string()) {
                let observation = self.observations.get(observation_index).ok_or(LanaError::Schema)?;
                if observation.memory_revision != next.to_string() || observation.derivation_ref != derivation_id { return Err(LanaError::Schema); }
                id(&observation.id)?;
                if !ids.insert(observation.id.clone()) { return Err(LanaError::Schema); }
                let &index = indices.get(&observation.root_id).ok_or(LanaError::Schema)?;
                let (vm, value) = &mut live[index];
                if observation.evidence.normalized()? != observation.evidence { return Err(LanaError::Schema); }
                let evidence = observation.evidence.live(vm)?;
                let observed = vm.information_observe(value, &evidence)?;
                let snapshot = self.roots[index].initial_information.snapshot(&observed)?;
                let core_revision = value.reactive.as_ref().ok_or(LanaError::Schema)?.lock().unwrap().revision;
                derivations.push(Derivation { id: derivation_id.clone(), operation: "observe".into(), revision: core_revision.to_string(),
                    input_ids: vec![references[index].clone()], source: self.roots[index].source.clone(), exactness: "exact_law".into(),
                    outcome: snapshot.clone(), reason: None });
                references[index] = derivation_id;
                evidence_ids.insert(observation.id.clone());
                evidence_ids.insert(observation.derivation_ref.clone());
                values[index] = snapshot;
                observation_index += 1;
            } else if self.aliases.get(alias_index).is_some_and(|alias| alias.creation_revision == next.to_string()) {
                let alias = &self.aliases[alias_index];
                if normalize(&alias.original_question)? != alias.normalized_question ||
                    !matches!(alias.target_kind.as_str(), "fact" | "root") || alias.target_id.is_empty() ||
                    (alias.target_kind == "root" && !indices.contains_key(&alias.target_id)) ||
                    !alias_targets.insert((&alias.normalized_question, &alias.target_kind, &alias.target_id)) {
                    return Err(LanaError::Schema);
                }
                alias_index += 1;
            } else if let Some(&(index, scoring)) = forecast_events.get(&(next as u64)) {
                let forecast = &self.forecasts[index];
                if scoring {
                    if forecast.score.is_none() { return Err(LanaError::Schema); }
                } else {
                    if ids.contains(&forecast.id) || forecast.input.evidence_refs.iter().any(|reference| !evidence_ids.contains(reference)) {
                        return Err(LanaError::Schema);
                    }
                    ids.insert(forecast.id.clone());
                }
            } else {
                return Err(LanaError::Schema);
            }
        }
        if root_index != self.roots.len() || observation_index != self.observations.len() || alias_index != self.aliases.len() {
            return Err(LanaError::Schema);
        }
        let current: Vec<_> = self.roots.iter().zip(&values).map(|(root, value)| serde_json::json!({"id":root.id,"value":value})).collect();
        Ok(Replay { values, derivations, digest: digest(&serde_json::json!({"roots":current}))? })
    }

    fn commit(&mut self, mut next: Self) -> Result<(), LanaError> {
        let replay = next.replay()?;
        next.derivations = replay.derivations;
        next.final_digest = replay.digest;
        let bytes = canonical(&next)?;
        if bytes.len() > 256 * 1024 * 1024 { return Err(LanaError::Limit); }
        *self = next;
        Ok(())
    }

    pub fn add(&mut self, root_id: &str, source: &str, information: Tagged) -> Result<bool, LanaError> {
        let chunk = Chunk::new(5, 0);
        let mut vm = Vm::new(&chunk);
        let information = information.snapshot(&information.to_live(&mut vm)?)?;
        if let Some(root) = self.roots.iter().find(|root| root.id == root_id) {
            return if root.source == source && root.initial_information == information { Ok(false) } else { Err(LanaError::Conflict) };
        }
        if self.observations.iter().any(|item| item.id == root_id) { return Err(LanaError::Conflict); }
        let mut next = self.clone();
        next.memory_revision = revision(&self.memory_revision)?.checked_add(1).ok_or(LanaError::Limit)?.to_string();
        next.roots.push(Root { id: root_id.into(), source: source.into(), initial_information: information, creation_revision: next.memory_revision.clone() });
        self.commit(next)?;
        Ok(true)
    }

    pub fn observe(&mut self, root_id: &str, observation_id: &str, evidence: Evidence) -> Result<bool, LanaError> {
        let evidence = evidence.normalized()?;
        if let Some(observation) = self.observations.iter().find(|item| item.id == observation_id) {
            return if observation.root_id == root_id && observation.evidence == evidence { Ok(false) } else { Err(LanaError::Conflict) };
        }
        if self.roots.iter().any(|root| root.id == observation_id) { return Err(LanaError::Conflict); }
        let mut next = self.clone();
        next.memory_revision = revision(&self.memory_revision)?.checked_add(1).ok_or(LanaError::Limit)?.to_string();
        next.observations.push(Observation { id: observation_id.into(), root_id: root_id.into(), evidence,
            memory_revision: next.memory_revision.clone(), derivation_ref: format!("brain/memory/{}", next.memory_revision) });
        self.commit(next)?;
        Ok(true)
    }

    pub fn current_values(&self) -> Result<Vec<Tagged>, LanaError> {
        Ok(self.replay()?.values)
    }

    pub fn inspect(&self, root_id: &str) -> Result<serde_json::Value, LanaError> {
        let index = self.roots.iter().position(|root| root.id == root_id).ok_or(LanaError::UnsupportedValue)?;
        let replay = self.replay()?;
        let root = &self.roots[index];
        let observations: Vec<_> = self.observations.iter().filter(|item| item.root_id == root_id).collect();
        let mut references = vec![format!("brain/memory/{}", root.creation_revision)];
        references.extend(observations.iter().map(|item| item.derivation_ref.clone()));
        Ok(serde_json::json!({"id":root_id,"source":root.source,"initial_information":root.initial_information,
            "current_information":replay.values[index],"observations":observations,"derivation_refs":references,
            "memory_revision":self.memory_revision}))
    }

    pub fn alias(&mut self, kind: &str, target: &str, question: &str) -> Result<bool, LanaError> {
        let normalized = normalize(question)?;
        if self.aliases.iter().any(|alias| alias.normalized_question == normalized && alias.target_kind == kind && alias.target_id == target) {
            return Ok(false);
        }
        let mut next = self.clone();
        next.memory_revision = revision(&self.memory_revision)?.checked_add(1).ok_or(LanaError::Limit)?.to_string();
        next.aliases.push(Alias { normalized_question: normalized, original_question: question.into(), target_kind: kind.into(),
            target_id: target.into(), creation_revision: next.memory_revision.clone() });
        self.commit(next)?;
        Ok(true)
    }

    pub fn forecast_add(&mut self, forecast_id: &str, target: &str, horizon: u64, input: ForecastInput) -> Result<bool, LanaError> {
        let candidate = Forecast { id: forecast_id.into(), target: target.into(), horizon, input,
            creation_revision: String::new(), score: None };
        validate_forecast(&candidate)?;
        if let Some(existing) = self.forecasts.iter().find(|item| item.id == forecast_id) {
            return if existing.target == candidate.target && existing.horizon == candidate.horizon && existing.input == candidate.input {
                Ok(false)
            } else { Err(LanaError::Conflict) };
        }
        let mut next = self.clone();
        next.memory_revision = revision(&self.memory_revision)?.checked_add(1).ok_or(LanaError::Limit)?.to_string();
        let mut candidate = candidate;
        candidate.creation_revision = next.memory_revision.clone();
        next.forecasts.push(candidate);
        self.commit(next)?;
        Ok(true)
    }

    pub fn forecast_score(&mut self, forecast_id: &str, observed_label: &str, observed_at: u64) -> Result<(bool, f64), LanaError> {
        let index = self.forecasts.iter().position(|item| item.id == forecast_id).ok_or(LanaError::UnsupportedValue)?;
        let forecast = &self.forecasts[index];
        if observed_at < forecast.horizon { return Err(LanaError::InvalidParameters); }
        let score = forecast_error(&forecast.input, observed_label)?;
        if let Some(existing) = &forecast.score {
            return if existing.observed_label == observed_label && existing.observed_at == observed_at {
                Ok((false, existing.score))
            } else { Err(LanaError::Conflict) };
        }
        let mut next = self.clone();
        next.memory_revision = revision(&self.memory_revision)?.checked_add(1).ok_or(LanaError::Limit)?.to_string();
        next.forecasts[index].score = Some(ForecastScore { observed_label: observed_label.into(), observed_at,
            score, score_revision: next.memory_revision.clone() });
        self.commit(next)?;
        Ok((true, score))
    }

    fn selected_root_context(&self, index: usize) -> Result<Vec<serde_json::Value>, LanaError> {
        let root = &self.roots[index];
        let mut selected = vec![serde_json::json!({"target_id":root.id,"value":root.initial_information,
            "sources":[root.source],"observation_ids":[],
            "derivation_refs":[format!("brain/memory/{}", root.creation_revision)]})];
        let mut seen = BTreeMap::new();
        seen.insert(canonical(&root.initial_information)?, 0usize);
        for observation in self.observations.iter().filter(|item| item.root_id == root.id) {
            let key = canonical(&observation.evidence)?;
            if let Some(&group) = seen.get(&key) {
                selected[group]["observation_ids"].as_array_mut().ok_or(LanaError::Schema)?
                    .push(observation.id.clone().into());
                selected[group]["derivation_refs"].as_array_mut().ok_or(LanaError::Schema)?
                    .push(observation.derivation_ref.clone().into());
            } else {
                if selected.len() == 64 { return Err(LanaError::Limit); }
                seen.insert(key, selected.len());
                selected.push(serde_json::json!({"target_id":root.id,"value":observation.evidence,
                    "sources":[root.source],"observation_ids":[observation.id],
                    "derivation_refs":[observation.derivation_ref]}));
            }
        }
        Ok(selected)
    }

    fn append_target_forecasts(&self, target_id: &str, selected: &mut Vec<serde_json::Value>) -> Result<(), LanaError> {
        for forecast in self.forecasts.iter().filter(|item| item.target == target_id) {
            if selected.len() == 64 { return Err(LanaError::Limit); }
            selected.push(serde_json::json!({"target_id":target_id,"forecast_id":forecast.id,
                "value":{"labels":forecast.input.labels,"probabilities":forecast.input.probabilities,
                         "horizon":forecast.horizon,"score":forecast.score},
                "sources":[format!("forecast:{}", forecast.id)],
                "evidence_refs":forecast.input.evidence_refs,
                "creation_revision":forecast.creation_revision}));
            for reference in &forecast.input.evidence_refs {
                if selected.iter().any(|entry| entry["target_id"] == *reference ||
                    entry["observation_ids"].as_array().is_some_and(|ids| ids.iter().any(|id| id == reference)) ||
                    entry["derivation_refs"].as_array().is_some_and(|ids| ids.iter().any(|id| id == reference))) {
                    continue;
                }
                if selected.len() == 64 { return Err(LanaError::Limit); }
                if let Some(root) = self.roots.iter().find(|root| root.id == *reference ||
                    format!("brain/memory/{}", root.creation_revision) == *reference) {
                    selected.push(serde_json::json!({"target_id":root.id,"value":root.initial_information,
                        "sources":[root.source],"observation_ids":[],
                        "derivation_refs":[format!("brain/memory/{}", root.creation_revision)],
                        "selected_by_forecast":forecast.id}));
                } else if let Some(observation) = self.observations.iter().find(|item| item.id == *reference || item.derivation_ref == *reference) {
                    let source = self.roots.iter().find(|root| root.id == observation.root_id).ok_or(LanaError::Schema)?;
                    selected.push(serde_json::json!({"target_id":observation.root_id,"value":observation.evidence,
                        "sources":[source.source],"observation_ids":[observation.id],
                        "derivation_refs":[observation.derivation_ref],
                        "selected_by_forecast":forecast.id}));
                } else {
                    return Err(LanaError::Schema);
                }
            }
        }
        Ok(())
    }

    /// Evidence selection must retain unresolved and conflicting exact targets too.
    pub(crate) fn required_context(&self, brain: &crate::brain::Brain, question: &str) -> Result<Vec<serde_json::Value>, LanaError> {
        let normalized = normalize(question)?;
        let mut targets = self.aliases.iter().filter(|alias| alias.normalized_question == normalized)
            .map(|alias| (alias.target_kind.as_str(), alias.target_id.as_str())).collect::<BTreeSet<_>>();
        if brain.recall_fact(question).is_some() { targets.insert(("fact", question)); }
        let mut result = Vec::new();
        let mut seen = BTreeSet::new();
        for (kind, target) in targets {
            let mut context = if kind == "fact" {
                let value = brain.recall_fact(target).ok_or(LanaError::UnsupportedValue)?;
                let reference = format!("fact:{target}");
                vec![serde_json::json!({"target_id":target,"value":value,"sources":[reference],
                    "observation_ids":[],"derivation_refs":[reference]})]
            } else {
                let index = self.roots.iter().position(|root| root.id == target).ok_or(LanaError::Schema)?;
                self.selected_root_context(index)?
            };
            self.append_target_forecasts(target, &mut context)?;
            for (position, mut entry) in context.into_iter().enumerate() {
                if entry.get("forecast_id").is_none() {
                    let kind = if kind == "fact" && position == 0 { "fact" } else { "root" };
                    let id = entry["target_id"].as_str().ok_or(LanaError::Schema)?;
                    entry["record_id"] = serde_json::json!(format!("{kind}:{id}"));
                }
                if seen.insert(canonical(&entry)?) { result.push(entry); }
                if result.len() > 64 { return Err(LanaError::Limit); }
            }
        }
        Ok(result)
    }

    pub fn grounded(&self, brain: &crate::brain::Brain, question: &str) -> Result<serde_json::Value, LanaError> {
        let question = normalize(question)?;
        let matches: Vec<_> = self.aliases.iter().filter(|alias| alias.normalized_question == question).collect();
        let mut result = serde_json::json!({"status":"ok","resolution":"unsupported","answer":null,
            "target_kind":null,"target_id":null,"evidence_refs":[],"memory_revision":self.memory_revision,
            "assumptions":[],"unsupported":true,"reason":"no_alias"});
        if matches.len() > 1 {
            result["resolution"] = "ambiguous".into();
            result["reason"] = "multiple_targets".into();
            result["evidence_refs"] = matches.iter().map(|alias| format!("{}:{}", alias.target_kind, alias.target_id))
                .collect::<Vec<_>>().into();
            return Ok(result);
        }
        let Some(alias) = matches.first() else { return Ok(result); };
        let (answer, reference, selected_context) = if alias.target_kind == "fact" {
            let Some(value) = brain.recall_fact(&alias.target_id) else { result["reason"] = "missing_target".into(); return Ok(result); };
            let reference = format!("fact:{}", alias.target_id);
            let mut context = vec![serde_json::json!({"target_id":alias.target_id,"value":value,
                "sources":[reference],"observation_ids":[],"derivation_refs":[reference]})];
            self.append_target_forecasts(&alias.target_id, &mut context)?;
            (serde_json::Value::String(value.into()), reference, context)
        } else {
            let Some(index) = self.roots.iter().position(|root| root.id == alias.target_id) else {
                result["reason"] = "missing_target".into(); return Ok(result);
            };
            let replay = self.replay()?;
            let mut context = self.selected_root_context(index)?;
            self.append_target_forecasts(&alias.target_id, &mut context)?;
            let reference = self.observations.iter().rev().find(|item| item.root_id == alias.target_id)
                .map(|item| item.derivation_ref.clone()).unwrap_or_else(|| format!("brain/memory/{}", self.roots[index].creation_revision));
            result["selected_context"] = serde_json::json!(context);
            result["evidence_refs"] = serde_json::json!([reference]);
            let chunk = Chunk::new(5, 0);
            let mut vm = Vm::new(&chunk);
            let value = replay.values[index].to_live(&mut vm)?;
            let resolved = match vm.information_resolve(&value) {
                Ok(value) => value,
                Err(LanaError::UnresolvedValue) => { result["reason"] = "unresolved_value".into(); return Ok(result); }
                Err(LanaError::UnsupportedOperation | LanaError::UnsupportedValue) => {
                    result["reason"] = "unsupported_inference".into(); return Ok(result);
                }
                Err(error) => return Err(error),
            };
            (serde_json::to_value(Tagged::plain(&resolved, 0)?).map_err(|_| LanaError::Schema)?, reference, context)
        };
        result["resolution"] = "exact".into();
        result["answer"] = answer;
        result["target_kind"] = alias.target_kind.clone().into();
        result["target_id"] = alias.target_id.clone().into();
        result["evidence_refs"] = vec![reference].into();
        result["selected_context"] = selected_context.into();
        result["unsupported"] = false.into();
        result["reason"] = serde_json::Value::Null;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_is_transactional_and_rejects_corrupt_history() {
        let mut memory = Memory::empty();
        let law: Tagged = serde_json::from_str(r#"{"tag":"distribution","dependency_id":"1","rows":[[{"tag":"bool","value":false},0.25],[{"tag":"bool","value":true},0.75]]}"#).unwrap();
        assert!(memory.add("weather", "sensor", law.clone()).unwrap());
        assert!(!memory.add("weather", "sensor", law).unwrap());
        let evidence = Evidence::Value(Tagged::Bool { value: true });
        assert!(memory.observe("weather", "sunny", evidence.clone()).unwrap());
        assert!(!memory.observe("weather", "sunny", evidence).unwrap());
        let before = canonical(&memory).unwrap();
        assert_eq!(memory.observe("weather", "sunny", Evidence::Value(Tagged::Null)), Err(LanaError::Conflict));
        assert_eq!(memory.observe("weather", "impossible", Evidence::Value(Tagged::Bool { value: false })), Err(LanaError::InvalidConditioning));
        assert_eq!(canonical(&memory).unwrap(), before);
        assert_eq!(Memory::load(&before).unwrap(), memory);
        let mut corrupt = memory.clone();
        corrupt.derivations[1].input_ids.clear();
        assert_eq!(Memory::load(&canonical(&corrupt).unwrap()), Err(LanaError::Schema));
        corrupt = memory.clone();
        corrupt.final_digest = "0".repeat(64);
        assert_eq!(Memory::load(&canonical(&corrupt).unwrap()), Err(LanaError::Integrity));
        assert_eq!(memory.inspect("weather").unwrap()["memory_revision"], "2");
    }

    #[test]
    fn forecast_is_declared_scored_and_replayed_without_changing_evidence() {
        let mut memory = Memory::empty();
        memory.add("weather", "sensor", Tagged::Definite { value: Box::new(Tagged::Bool { value: true }) }).unwrap();
        let input = ForecastInput { created_at: 10, labels: vec!["yes".into(), "no".into()],
            probabilities: vec![0.75, 0.25], evidence_refs: vec!["weather".into()] };
        assert!(memory.forecast_add("forecast-1", "rain", 20, input.clone()).unwrap());
        assert!(!memory.forecast_add("forecast-1", "rain", 20, input).unwrap());
        let before = canonical(&memory).unwrap();
        assert_eq!(memory.forecast_score("forecast-1", "yes", 19), Err(LanaError::InvalidParameters));
        assert_eq!(canonical(&memory).unwrap(), before);
        assert_eq!(memory.forecast_score("forecast-1", "yes", 21).unwrap(), (true, 0.125));
        assert_eq!(memory.forecast_score("forecast-1", "yes", 21).unwrap(), (false, 0.125));
        assert_eq!(memory.forecast_score("forecast-1", "no", 21), Err(LanaError::Conflict));
        let bytes = canonical(&memory).unwrap();
        assert_eq!(Memory::load(&bytes).unwrap(), memory);
        let mut corrupt = memory.clone();
        corrupt.forecasts[0].score.as_mut().unwrap().score = 0.0;
        assert_eq!(Memory::load(&canonical(&corrupt).unwrap()), Err(LanaError::Schema));
        assert_eq!(memory.inspect("weather").unwrap()["current_information"]["tag"], "definite");
    }

    #[test]
    fn grounded_context_collapses_same_target_evidence_and_omits_unrelated_roots() {
        let mut memory = Memory::empty();
        let law: Tagged = serde_json::from_str(r#"{"tag":"distribution","dependency_id":"1","rows":[[{"tag":"bool","value":false},0.25],[{"tag":"bool","value":true},0.75]]}"#).unwrap();
        memory.add("weather", "sensor", law).unwrap();
        memory.add("unrelated", "other", Tagged::Definite { value: Box::new(Tagged::Bool { value: false }) }).unwrap();
        memory.add("noise", "other", Tagged::Definite { value: Box::new(Tagged::Bool { value: false }) }).unwrap();
        memory.observe("weather", "seen-1", Evidence::Value(Tagged::Bool { value: true })).unwrap();
        memory.observe("weather", "seen-2", Evidence::Value(Tagged::Bool { value: true })).unwrap();
        memory.alias("root", "weather", "Was it sunny?").unwrap();
        let input = ForecastInput { created_at: 10, labels: vec!["yes".into(), "no".into()],
            probabilities: vec![0.75, 0.25], evidence_refs: vec!["unrelated".into()] };
        memory.forecast_add("relevant", "weather", 20, input.clone()).unwrap();
        memory.forecast_add("irrelevant", "unrelated", 20, input).unwrap();
        let brain = crate::brain::Brain::new(3, 2, 2, 7).unwrap();
        let answer = memory.grounded(&brain, "Was it sunny?").unwrap();
        assert_eq!(answer["resolution"], "exact");
        let context = answer["selected_context"].as_array().unwrap();
        assert_eq!(context.len(), 4);
        assert_eq!(context[1]["observation_ids"], serde_json::json!(["seen-1", "seen-2"]));
        assert_eq!(context[2]["forecast_id"], "relevant");
        assert_eq!(context[3]["target_id"], "unrelated");
        assert!(context.iter().all(|item| item["target_id"] != "noise"));
        assert_eq!(memory.observations.len(), 2);
    }

    #[test]
    fn grounded_selection_limit_leaves_memory_unchanged() {
        let mut memory = Memory::empty();
        memory.add("target", "source", Tagged::Definite { value: Box::new(Tagged::Bool { value: true }) }).unwrap();
        memory.alias("root", "target", "Question?").unwrap();
        let input = ForecastInput { created_at: 10, labels: vec!["yes".into(), "no".into()],
            probabilities: vec![0.5, 0.5], evidence_refs: vec!["target".into()] };
        for number in 0..64 {
            memory.forecast_add(&format!("forecast-{number}"), "target", 20, input.clone()).unwrap();
        }
        let before = canonical(&memory).unwrap();
        let brain = crate::brain::Brain::new(3, 2, 2, 7).unwrap();
        assert_eq!(memory.grounded(&brain, "Question?"), Err(LanaError::Limit));
        assert_eq!(canonical(&memory).unwrap(), before);
    }

    #[test]
    fn grounded_selection_keeps_late_relevant_evidence_after_irrelevant_records() {
        let mut memory = Memory::empty();
        memory.add("target", "source", Tagged::Definite { value: Box::new(Tagged::Bool { value: true }) }).unwrap();
        for number in 0..80 {
            memory.add(&format!("noise-{number}"), "other", Tagged::Definite {
                value: Box::new(Tagged::Bool { value: false }) }).unwrap();
        }
        memory.add("late", "decision evidence", Tagged::Definite {
            value: Box::new(Tagged::Bool { value: true }) }).unwrap();
        memory.alias("root", "target", "Question?").unwrap();
        memory.forecast_add("late-forecast", "target", 20, ForecastInput {
            created_at: 10, labels: vec!["yes".into(), "no".into()],
            probabilities: vec![0.75, 0.25], evidence_refs: vec!["late".into()],
        }).unwrap();
        let brain = crate::brain::Brain::new(3, 2, 2, 7).unwrap();
        let answer = memory.grounded(&brain, "Question?").unwrap();
        let context = answer["selected_context"].as_array().unwrap();
        assert_eq!(context.len(), 3);
        assert_eq!(context[1]["forecast_id"], "late-forecast");
        assert_eq!(context[2]["target_id"], "late");
        assert!(context.iter().all(|entry| !entry["target_id"].as_str().unwrap().starts_with("noise-")));
    }
}
