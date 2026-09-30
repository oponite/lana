//! Read-only walk-forward evaluation with pure, fresh callbacks per fold.

use super::*;
use serde_json::{json, Value as Json};

#[derive(Clone)]
struct EvaluationRow {
    id: String,
    observed_at: i64,
    target_at: i64,
    features: Json,
    target: Json,
    training: Json,
}

struct EvaluationOptions {
    initial_train: usize,
    test_size: usize,
    step_size: usize,
    gap: usize,
    seed: u64,
}

fn positive_integer(value: Option<&Json>, default: usize) -> Result<usize, LanaError> {
    let Some(value) = value else { return Ok(default); };
    let number = value.as_f64().ok_or(LanaError::Schema)?;
    if !number.is_finite() || number.fract() != 0.0 || number <= 0.0 ||
        number > 9_007_199_254_740_991.0 || number >= usize::MAX as f64 {
        return Err(LanaError::Schema);
    }
    Ok(number as usize)
}

fn nonnegative_integer(value: Option<&Json>, default: usize) -> Result<usize, LanaError> {
    let Some(value) = value else { return Ok(default); };
    let number = value.as_f64().ok_or(LanaError::Schema)?;
    if !number.is_finite() || number.fract() != 0.0 || number < 0.0 ||
        number > 9_007_199_254_740_991.0 || number >= usize::MAX as f64 {
        return Err(LanaError::Schema);
    }
    Ok(number as usize)
}

fn nonnegative_seed(value: Option<&Json>) -> Result<u64, LanaError> {
    let Some(value) = value else { return Ok(0); };
    if let Some(value) = value.as_str() {
        let parsed = value.parse::<u64>().map_err(|_| LanaError::Schema)?;
        return if parsed.to_string() == value { Ok(parsed) } else { Err(LanaError::Schema) };
    }
    let number = value.as_f64().ok_or(LanaError::Schema)?;
    if !number.is_finite() || number.fract() != 0.0 || number < 0.0 ||
        number > 9_007_199_254_740_991.0 { return Err(LanaError::Schema); }
    Ok(number as u64)
}

fn utc_time(value: &Json) -> Result<i64, LanaError> {
    let number = value.as_f64().ok_or(LanaError::Schema)?;
    if !number.is_finite() || number.fract() != 0.0 || number.abs() > 9_007_199_254_740_991.0 {
        return Err(LanaError::Schema);
    }
    Ok(number as i64)
}

fn canonical(value: &Json) -> Result<Vec<u8>, LanaError> {
    use serde_json::ser::{CharEscape, Formatter};
    struct Format;
    impl Formatter for Format {
        fn write_char_escape<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, escape: CharEscape)
            -> std::io::Result<()> {
            let byte = match escape {
                CharEscape::Backspace => 8, CharEscape::FormFeed => 12, CharEscape::LineFeed => 10,
                CharEscape::CarriageReturn => 13, CharEscape::Tab => 9,
                CharEscape::AsciiControl(byte) => byte,
                _ => return serde_json::ser::CompactFormatter.write_char_escape(writer, escape),
            };
            write!(writer, "\\u{byte:04x}")
        }
    }
    let mut bytes = Vec::new();
    serde::Serialize::serialize(value, &mut serde_json::Serializer::with_formatter(&mut bytes, Format))
        .map_err(|_| LanaError::Schema)?;
    Ok(bytes)
}

fn bits(value: f64) -> Result<String, LanaError> {
    if !value.is_finite() { return Err(LanaError::Schema); }
    Ok(format!("{:016x}", (if value == 0.0 { 0.0 } else { value }).to_bits()))
}

fn tagged(value: &Value, depth: usize) -> Result<Json, LanaError> {
    if depth > 64 { return Err(LanaError::Limit); }
    if value.reactive.is_some() || value.claim.is_some() || value.planned_effect.is_some() {
        return Err(LanaError::UnsupportedValue);
    }
    match &value.kind {
        ValueKind::Null => Ok(json!({"tag":"null"})),
        ValueKind::Bool(boolean) => Ok(json!({"tag":"bool","value":boolean})),
        ValueKind::String(string) => Ok(json!({"tag":"string","value":string.as_ref()})),
        ValueKind::Number(number) => Ok(json!({"tag":"number","bits":bits(*number)?})),
        ValueKind::State(state) if state.indexes == Default::default() =>
            Ok(json!({"tag":"state",
                "p":bits(state.state.p)?,
                "d_re":bits(state.state.d_re)?,
                "d_im":bits(state.state.d_im)?})),
        ValueKind::Array(items) => {
            let items = items.lock().unwrap().items().to_vec();
            Ok(json!({"tag":"array","items":items.iter()
                .map(|item| tagged(item, depth + 1)).collect::<Result<Vec<_>, _>>()?}))
        }
        ValueKind::Map(fields) => {
            let fields = fields.lock().unwrap().entries().to_vec();
            let mut entries = fields.iter().map(|entry|
                Ok((entry.key.to_string(), tagged(&entry.value, depth + 1)?)))
                .collect::<Result<Vec<_>, LanaError>>()?;
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(json!({"tag":"map","entries":entries}))
        }
        _ => Err(LanaError::UnsupportedValue),
    }
}

fn label_key(value: &Json) -> Result<String, LanaError> {
    match value {
        Json::Bool(value) => Ok(value.to_string()),
        Json::String(value) if !value.is_empty() => Ok(value.clone()),
        _ => Err(LanaError::Schema),
    }
}

impl Vm<'_> {
    fn evaluation_plain(&self, value: &Value, depth: usize) -> Result<Json, LanaError> {
        if depth > 64 { return Err(LanaError::Limit); }
        if value.reactive.is_some() || value.claim.is_some() || value.planned_effect.is_some() {
            return Err(LanaError::UnsupportedValue);
        }
        match &value.kind {
            ValueKind::Null => Ok(Json::Null),
            ValueKind::Bool(value) => Ok(json!(value)),
            ValueKind::Number(value) if value.is_finite() => Ok(json!(value)),
            ValueKind::String(value) => Ok(json!(value.as_ref())),
            ValueKind::Array(value) => {
                let items = value.lock().unwrap().items().to_vec();
                items.iter().map(|item| self.evaluation_plain(item, depth + 1))
                    .collect::<Result<Vec<_>, _>>().map(Json::Array)
            }
            ValueKind::Map(value) => {
                let mut fields = serde_json::Map::new();
                let entries = value.lock().unwrap().entries().to_vec();
                for entry in entries {
                    fields.insert(entry.key.to_string(), self.evaluation_plain(&entry.value, depth + 1)?);
                }
                Ok(Json::Object(fields))
            }
            _ => Err(LanaError::UnsupportedValue),
        }
    }

    fn evaluation_value(&self, value: &Json, depth: usize) -> Result<Value, LanaError> {
        if depth > 64 { return Err(LanaError::Limit); }
        match value {
            Json::Null => Ok(Value::null()),
            Json::Bool(value) => Ok(Value::boolean(*value)),
            Json::Number(value) => Ok(Value::number(value.as_f64().filter(|number| number.is_finite()).ok_or(LanaError::Schema)?)),
            Json::String(value) => Ok(Value::string(self.heap.string(value)?)),
            Json::Array(items) => self.array_value(items.iter().map(|item| self.evaluation_value(item, depth + 1))
                .collect::<Result<Vec<_>, _>>()?),
            Json::Object(fields) => {
                let mut result = Map::new(&self.heap, fields.len())?;
                for (key, value) in fields {
                    result.set(self.heap.string(key)?, self.evaluation_value(value, depth + 1)?, true)?;
                }
                Ok(Value::map(Arc::new(Mutex::new(result))))
            }
        }
    }

    fn evaluation_call(&mut self, function: u32, args: &[Value], scratch: u32) -> Result<Value, LanaError> {
        let saved = self.current_frame().registers[scratch as usize].clone();
        let mut result = Value::null();
        self.pure_callback_depth += 1;
        self.evaluation_callback_depth += 1;
        let error = self.run_function_args(function, args, scratch, &mut result);
        self.evaluation_callback_depth -= 1;
        self.pure_callback_depth -= 1;
        self.current_frame_mut().registers[scratch as usize] = saved;
        if error != LanaError::Ok { return Err(error); }
        Ok(result)
    }

    pub(super) fn evaluation_walk_forward(&mut self, arguments: &[Value], scratch: u32, out: &mut Value) -> LanaError {
        match self.evaluation_walk_forward_result(arguments, scratch) {
            Ok(result) => { *out = result; LanaError::Ok },
            Err(error) => error,
        }
    }

    fn evaluation_walk_forward_result(&mut self, arguments: &[Value], scratch: u32) -> Result<Value, LanaError> {
        if arguments.len() != 3 { return Err(LanaError::Type); }
        let ValueKind::Array(examples) = &arguments[0].kind else { return Err(LanaError::Type); };
        let live_rows = examples.lock().unwrap().items().to_vec();
        if live_rows.len() > 10_000 { return Err(LanaError::Limit); }
        let ValueKind::Map(trainer) = &arguments[1].kind else { return Err(LanaError::Type); };
        let trainer = trainer.lock().unwrap();
        if trainer.entries().len() != 5 { return Err(LanaError::Schema); }
        let ValueKind::Function(fit_fn) = trainer.get("fit").ok_or(LanaError::Schema)?.kind else { return Err(LanaError::Type); };
        let ValueKind::Function(predict_fn) = trainer.get("predict").ok_or(LanaError::Schema)?.kind else { return Err(LanaError::Type); };
        let kind = trainer.get("kind").ok_or(LanaError::Schema)?.clone();
        let labels = trainer.get("labels").ok_or(LanaError::Schema)?.clone();
        let calculation_version = trainer.get("calculation_version").ok_or(LanaError::Schema)?.clone();
        drop(trainer);
        let kind = self.evaluation_plain(&kind, 0)?;
        let kind = kind.as_str().ok_or(LanaError::Schema)?.to_owned();
        if !matches!(kind.as_str(), "classification" | "regression") { return Err(LanaError::Schema); }
        let labels = self.evaluation_plain(&labels, 0)?;
        let labels = labels.as_array().ok_or(LanaError::Schema)?.clone();
        let calculation_version = self.evaluation_plain(&calculation_version, 0)?;
        let calculation_version = calculation_version.as_str().filter(|value| !value.is_empty())
            .ok_or(LanaError::Schema)?.to_owned();
        if kind == "classification" {
            if !(2..=128).contains(&labels.len()) { return Err(LanaError::Schema); }
            let mut keys = HashSet::new();
            for label in &labels { if !keys.insert(label_key(label)?) { return Err(LanaError::Schema); } }
        } else if !labels.is_empty() { return Err(LanaError::Schema); }
        let options = self.evaluation_plain(&arguments[2], 0)?;
        let fields = options.as_object().ok_or(LanaError::Schema)?;
        if fields.len() > 5 || fields.keys().any(|key|
            !matches!(key.as_str(), "initial_train" | "test_size" | "step_size" | "gap" | "seed")) {
            return Err(LanaError::Schema);
        }
        let options = EvaluationOptions {
            initial_train: positive_integer(fields.get("initial_train"), 100)?,
            test_size: positive_integer(fields.get("test_size"), 20)?,
            step_size: positive_integer(fields.get("step_size"), 20)?,
            gap: nonnegative_integer(fields.get("gap"), 0)?,
            seed: nonnegative_seed(fields.get("seed"))?,
        };
        let mut rows = Vec::with_capacity(live_rows.len());
        let mut ids = HashSet::new();
        for live in live_rows {
            self.charge_bounded_work(1)?;
            let row = self.evaluation_plain(&live, 0)?;
            let fields = row.as_object().ok_or(LanaError::Schema)?;
            if fields.len() != 5 || !["id", "observed_at", "target_at", "features", "target"]
                .iter().all(|key| fields.contains_key(*key)) { return Err(LanaError::Schema); }
            let id = row["id"].as_str().filter(|value| !value.is_empty() && value.len() <= 128)
                .ok_or(LanaError::Schema)?.to_owned();
            if !ids.insert(id.clone()) { return Err(LanaError::Schema); }
            let observed_at = utc_time(&row["observed_at"])?;
            let target_at = utc_time(&row["target_at"])?;
            if observed_at > target_at { return Err(LanaError::Schema); }
            let features = row["features"].as_object().ok_or(LanaError::Schema)?;
            if features.len() > 64 { return Err(LanaError::Limit); }
            let mut plain_features = serde_json::Map::new();
            for (name, feature) in features {
                if name.is_empty() || name.len() > 128 { return Err(LanaError::Schema); }
                let fields = feature.as_object().ok_or(LanaError::Schema)?;
                if fields.len() != 2 || !fields.contains_key("value") || !fields.contains_key("available_at")
                    || utc_time(&feature["available_at"])? > observed_at { return Err(LanaError::Schema); }
                plain_features.insert(name.clone(), feature["value"].clone());
            }
            let target = row["target"].clone();
            if kind == "classification" {
                if !labels.contains(&target) { return Err(LanaError::Schema); }
            } else if target.as_f64().is_none_or(|value| !value.is_finite()) { return Err(LanaError::Schema); }
            if rows.last().is_some_and(|previous: &EvaluationRow|
                previous.observed_at > observed_at ||
                    (previous.observed_at == observed_at && previous.id >= id)) { return Err(LanaError::Schema); }
            let features = Json::Object(plain_features);
            let training = json!({"id":id,"observed_at":observed_at,"target_at":target_at,
                "features":features,"target":target});
            rows.push(EvaluationRow { id, observed_at, target_at, features, target, training });
        }
        self.evaluation_folds(&rows, fit_fn, predict_fn, &kind, &labels, &calculation_version,
            &options, scratch)
    }

    fn evaluation_folds(&mut self, rows: &[EvaluationRow], fit_fn: u32, predict_fn: u32,
        kind: &str, labels: &[Json], calculation_version: &str, options: &EvaluationOptions,
        scratch: u32) -> Result<Value, LanaError> {
        let start = options.initial_train.checked_add(options.gap).ok_or(LanaError::Limit)?;
        let mut folds = Vec::new();
        let mut all_predictions = Vec::new();
        let mut repeated_test_ids = Vec::new();
        let mut seen_test_ids = HashSet::new();
        let mut tail_start = start.min(rows.len());
        for fold_index in 0..100usize {
            let offset = fold_index.checked_mul(options.step_size).ok_or(LanaError::Limit)?;
            let test_start = start.checked_add(offset).ok_or(LanaError::Limit)?;
            let Some(test_end) = test_start.checked_add(options.test_size) else { return Err(LanaError::Limit); };
            if test_end > rows.len() { tail_start = test_start.min(rows.len()); break; }
            tail_start = test_end;
            let cutoff = test_start.checked_sub(options.gap).ok_or(LanaError::Limit)?;
            let first_time = rows[test_start].observed_at;
            let eligible = rows[..cutoff].iter().filter(|row| row.target_at < first_time).collect::<Vec<_>>();
            let validation_count = 20usize.max(eligible.len().div_ceil(5));
            if eligible.len() <= validation_count {
                return self.evaluation_value(&json!({"schema_version":1,"status":"insufficient_evidence",
                    "folds":[],"aggregate":null,"repeated_test_ids":[],
                    "incomplete_tail_count":rows.len().saturating_sub(tail_start),
                    "calculation_version":calculation_version}), 0);
            }
            let seed = options.seed.checked_add(fold_index as u64).ok_or(LanaError::Limit)?;
            let training = self.array_value(eligible[..eligible.len() - validation_count].iter()
                .map(|row| self.evaluation_value(&row.training, 0)).collect::<Result<Vec<_>, _>>()?)?;
            let validation = self.array_value(eligible[eligible.len() - validation_count..].iter()
                .map(|row| self.evaluation_value(&row.training, 0)).collect::<Result<Vec<_>, _>>()?)?;
            let model = self.evaluation_call(fit_fn, &[training, validation,
                Value::string(self.heap.string(&seed.to_string())?)], scratch)?;
            let ValueKind::Map(model_fields) = &model.kind else { return Err(LanaError::Schema); };
            let model_fields = model_fields.lock().unwrap();
            if model_fields.entries().len() != 2 || model_fields.get("payload").is_none() {
                return Err(LanaError::Schema);
            }
            let version = model_fields.get("calculation_version").ok_or(LanaError::Schema)?.clone();
            drop(model_fields);
            if self.evaluation_plain(&version, 0)? != calculation_version { return Err(LanaError::Schema); }
            let digest = hex_digest(&sha256(&canonical(&tagged(&model, 0)?)?));
            let mut predictions = Vec::new();
            for row in &rows[test_start..test_end] {
                self.charge_bounded_work(1)?;
                let features = self.evaluation_value(&row.features, 0)?;
                let returned = self.evaluation_call(predict_fn, &[model.clone(), features], scratch)?;
                let prediction = self.evaluation_plain(&returned, 0)?;
                validate_prediction(&prediction, &row.target, kind, labels)?;
                if !seen_test_ids.insert(row.id.clone()) && !repeated_test_ids.contains(&row.id) {
                    repeated_test_ids.push(row.id.clone());
                }
                predictions.push(json!({"id":row.id,"prediction":prediction,"target":row.target}));
                all_predictions.push((prediction, row.target.clone()));
            }
            let paired = predictions.iter().map(|item| (item["prediction"].clone(), item["target"].clone()))
                .collect::<Vec<_>>();
            let metrics = metrics(&paired, kind, labels)?;
            folds.push(json!({"train_ids":eligible[..eligible.len() - validation_count].iter()
                    .map(|row| row.id.clone()).collect::<Vec<_>>(),
                "validation_ids":eligible[eligible.len() - validation_count..].iter()
                    .map(|row| row.id.clone()).collect::<Vec<_>>(),
                "test_ids":rows[test_start..test_end].iter().map(|row| row.id.clone()).collect::<Vec<_>>(),
                "test_start":rows[test_start].observed_at,"test_end":rows[test_end - 1].observed_at,
                "seed":seed.to_string(),"model_digest":digest,"predictions":predictions,"metrics":metrics}));
        }
        if folds.len() == 100 && start.checked_add(100usize.checked_mul(options.step_size)
            .ok_or(LanaError::Limit)?).and_then(|next| next.checked_add(options.test_size))
            .is_some_and(|next_end| next_end <= rows.len()) { return Err(LanaError::Limit); }
        let status = if folds.is_empty() { "insufficient_evidence" } else { "complete" };
        let aggregate = if folds.is_empty() { Json::Null } else { metrics(&all_predictions, kind, labels)? };
        self.evaluation_value(&json!({"schema_version":1,"status":status,"folds":folds,
            "aggregate":aggregate,"repeated_test_ids":repeated_test_ids,
            "incomplete_tail_count":rows.len().saturating_sub(tail_start),
            "calculation_version":calculation_version}), 0)
    }
}

fn validate_prediction(prediction: &Json, target: &Json, kind: &str, labels: &[Json]) -> Result<(), LanaError> {
    let fields = prediction.as_object().ok_or(LanaError::Schema)?;
    if kind == "regression" {
        if fields.len() != 1 || prediction["value"].as_f64().is_none_or(|value| !value.is_finite()) {
            return Err(LanaError::Schema);
        }
        return Ok(());
    }
    if fields.len() != 2 || !fields.contains_key("label") || !fields.contains_key("probabilities")
        || !labels.contains(&prediction["label"]) { return Err(LanaError::Schema); }
    if prediction["probabilities"].is_null() { return Ok(()); }
    let probabilities = prediction["probabilities"].as_object().ok_or(LanaError::Schema)?;
    if probabilities.len() != labels.len() { return Err(LanaError::Schema); }
    let mut sum = 0.0;
    let mut best = 0usize;
    let mut best_probability = -1.0f64;
    for (index, label) in labels.iter().enumerate() {
        let probability = probabilities.get(&label_key(label)?).and_then(Json::as_f64).ok_or(LanaError::Schema)?;
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) { return Err(LanaError::Schema); }
        sum += probability;
        if probability > best_probability { best_probability = probability; best = index; }
    }
    if (sum - 1.0).abs() > 1e-12 || prediction["label"] != labels[best]
        || !labels.contains(target) { return Err(LanaError::Schema); }
    Ok(())
}

fn metrics(pairs: &[(Json, Json)], kind: &str, labels: &[Json]) -> Result<Json, LanaError> {
    let count = pairs.len();
    if count == 0 { return Err(LanaError::Schema); }
    if kind == "regression" {
        let mut absolute = 0.0;
        let mut squared = 0.0;
        for (prediction, target) in pairs {
            let error = prediction["value"].as_f64().ok_or(LanaError::Schema)?
                - target.as_f64().ok_or(LanaError::Schema)?;
            absolute += error.abs(); squared += error * error;
        }
        if !absolute.is_finite() || !squared.is_finite() { return Err(LanaError::Schema); }
        return Ok(json!({"count":count,"mae":absolute / count as f64,"rmse":(squared / count as f64).sqrt()}));
    }
    let mut correct = 0usize;
    let mut log_loss = 0.0;
    let mut unavailable = false;
    let mut infinite = false;
    for (prediction, target) in pairs {
        if prediction["label"] == *target { correct += 1; }
        if let Some(probabilities) = prediction["probabilities"].as_object() {
            let probability = probabilities[&label_key(target)?].as_f64().ok_or(LanaError::Schema)?;
            if probability == 0.0 { infinite = true; }
            else { log_loss -= probability.ln(); }
        } else { unavailable = true; }
    }
    let status = if unavailable { "unavailable" } else if infinite { "infinite" } else { "finite" };
    let log_loss = if status == "finite" { json!(log_loss / count as f64) } else { Json::Null };
    let _ = labels;
    Ok(json!({"count":count,"correct":correct,"accuracy":correct as f64 / count as f64,
        "log_loss":log_loss,"log_loss_status":status}))
}
