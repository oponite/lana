//! Small deterministic CPU brain primitives. This is intentionally not a
//! general tensor framework: it owns only the three parameter groups required
//! by the first Brain workflow.

use std::io::{Read, Write};
use std::path::Path;

use lana_bytecode::LanaError;

const MAX_BRAIN_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct Brain {
    pub vocabulary: usize,
    pub embedding_width: usize,
    pub hidden_width: usize,
    pub version: u64,
    pub seed: u64,
    pub embedding: Vec<f32>,
    pub hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub output: Vec<f32>,
    pub output_bias: Vec<f32>,
    pub memory: Vec<String>,
    pub training_history: Vec<f32>,
    pub replay_steps: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingResult {
    pub brain: Brain,
    pub loss: f32,
    pub changed_groups: Vec<&'static str>,
}

impl Brain {
    pub fn relu(values: &[f32]) -> Result<Vec<f32>, LanaError> {
        if values.iter().any(|value| !value.is_finite()) { return Err(LanaError::InvalidParameters); }
        Ok(values.iter().map(|value| value.max(0.0)).collect())
    }

    pub fn gelu(values: &[f32]) -> Result<Vec<f32>, LanaError> {
        if values.iter().any(|value| !value.is_finite()) { return Err(LanaError::InvalidParameters); }
        Ok(values.iter().map(|value| 0.5 * value * (1.0 + (0.79788456 * (value + 0.044715 * value.powi(3))).tanh())).collect())
    }

    pub fn dense(input: &[f32], weights: &[f32], bias: &[f32]) -> Result<Vec<f32>, LanaError> {
        if input.is_empty() || bias.is_empty() || weights.len() != input.len() * bias.len() || input.iter().chain(weights).chain(bias).any(|value| !value.is_finite()) { return Err(LanaError::InvalidParameters); }
        Ok((0..bias.len()).map(|row| bias[row] + input.iter().enumerate().map(|(column, value)| value * weights[row * input.len() + column]).sum::<f32>()).collect())
    }

    pub fn layer_norm(values: &[f32]) -> Result<Vec<f32>, LanaError> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) { return Err(LanaError::InvalidParameters); }
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / values.len() as f32;
        Ok(values.iter().map(|value| (value - mean) / (variance + 1e-5).sqrt()).collect())
    }

    pub fn causal_mask(length: usize) -> Result<Vec<Vec<bool>>, LanaError> {
        if length == 0 || length > 50_000_000 { return Err(LanaError::Limit); }
        Ok((0..length).map(|row| (0..length).map(|column| column <= row).collect()).collect())
    }

    pub fn mean_pool(rows: &[Vec<f32>]) -> Result<Vec<f32>, LanaError> {
        let Some(width) = rows.first().map(Vec::len) else { return Err(LanaError::InvalidParameters); };
        if width == 0 || rows.iter().any(|row| row.len() != width || row.iter().any(|value| !value.is_finite())) { return Err(LanaError::InvalidParameters); }
        let mut pooled = vec![0.0; width];
        for row in rows { for (out, value) in pooled.iter_mut().zip(row) { *out += value; } }
        for value in &mut pooled { *value /= rows.len() as f32; }
        Ok(pooled)
    }

    pub fn softmax(logits: &[f32]) -> Result<Vec<f32>, LanaError> {
        if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) { return Err(LanaError::InvalidParameters); }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut values: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
        let total: f32 = values.iter().sum();
        for value in &mut values { *value /= total; }
        Ok(values)
    }

    pub fn scheduled_learning_rate(initial: f32, step: u64, warmup_steps: u64, total_steps: u64) -> Result<f32, LanaError> {
        if !initial.is_finite() || initial <= 0.0 || total_steps == 0 || step >= total_steps { return Err(LanaError::InvalidParameters); }
        if warmup_steps > total_steps { return Err(LanaError::InvalidParameters); }
        if warmup_steps > 0 && step < warmup_steps {
            return Ok(initial * (step + 1) as f32 / warmup_steps as f32);
        }
        let remaining = total_steps - warmup_steps;
        Ok(if remaining == 0 { initial } else { initial * (total_steps - step) as f32 / remaining as f32 })
    }

    pub fn trained_next_token(&self, tokens: &[usize], target: usize, learning_rate: f32) -> Result<TrainingResult, LanaError> {
        let mut brain = self.clone();
        let loss = brain.train_next_token(tokens, target, learning_rate)?;
        let mut changed_groups = Vec::new();
        if brain.embedding != self.embedding { changed_groups.push("embedding"); }
        if brain.hidden != self.hidden || brain.hidden_bias != self.hidden_bias { changed_groups.push("hidden"); }
        if brain.output != self.output || brain.output_bias != self.output_bias { changed_groups.push("output"); }
        Ok(TrainingResult { brain, loss, changed_groups })
    }

    pub fn binary_cross_entropy(probability: f32, target: bool) -> Result<f32, LanaError> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) { return Err(LanaError::InvalidParameters); }
        let value = probability.clamp(1e-7, 1.0 - 1e-7);
        Ok(if target { -value.ln() } else { -(1.0 - value).ln() })
    }

    pub fn masked_cross_entropy(logits: &[Vec<f32>], targets: &[usize], mask: &[bool]) -> Result<f32, LanaError> {
        if logits.len() != targets.len() || logits.len() != mask.len() { return Err(LanaError::InvalidParameters); }
        let mut total = 0.0;
        let mut count = 0usize;
        for ((row, &target), &enabled) in logits.iter().zip(targets).zip(mask) {
            if !enabled { continue; }
            if row.is_empty() || target >= row.len() || row.iter().any(|value| !value.is_finite()) { return Err(LanaError::InvalidParameters); }
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            total -= row[target] - max - row.iter().map(|value| (value - max).exp()).sum::<f32>().ln();
            count += 1;
        }
        if count == 0 { return Err(LanaError::InvalidParameters); }
        Ok(total / count as f32)
    }

    pub fn new(vocabulary: usize, embedding_width: usize, hidden_width: usize, seed: u64) -> Result<Self, LanaError> {
        if vocabulary == 0 || embedding_width == 0 || hidden_width == 0 {
            return Err(LanaError::InvalidParameters);
        }
        let count = vocabulary.checked_mul(embedding_width)
            .and_then(|n| n.checked_add(hidden_width.checked_mul(embedding_width)?))
            .and_then(|n| n.checked_add(hidden_width))
            .and_then(|n| n.checked_add(vocabulary.checked_mul(hidden_width)?))
            .and_then(|n| n.checked_add(vocabulary))
            .ok_or(LanaError::Limit)?;
        if count.checked_mul(std::mem::size_of::<f32>()).ok_or(LanaError::Limit)? > MAX_BRAIN_BYTES {
            return Err(LanaError::Limit);
        }
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f32 / u64::MAX as f32) - 0.5) * 0.02
        };
        Ok(Self {
            vocabulary, embedding_width, hidden_width, version: 1, seed,
            embedding: (0..vocabulary * embedding_width).map(|_| next()).collect(),
            hidden: (0..hidden_width * embedding_width).map(|_| next()).collect(),
            hidden_bias: vec![0.0; hidden_width],
            output: (0..vocabulary * hidden_width).map(|_| next()).collect(),
            output_bias: vec![0.0; vocabulary],
            memory: Vec::new(),
            training_history: Vec::new(), replay_steps: 0,
        })
    }

    pub fn logits(&self, tokens: &[usize]) -> Result<Vec<f32>, LanaError> {
        if tokens.is_empty() || tokens.iter().any(|&token| token >= self.vocabulary) {
            return Err(LanaError::InvalidParameters);
        }
        let pooled = self.pool(tokens);
        let hidden = self.hidden_values(&pooled);
        Ok((0..self.vocabulary).map(|row| {
            self.output_bias[row] + (0..self.hidden_width)
                .map(|col| self.output[row * self.hidden_width + col] * hidden[col]).sum::<f32>()
        }).collect())
    }

    pub fn next_token_loss(&self, tokens: &[usize], target: usize) -> Result<f32, LanaError> {
        if target >= self.vocabulary { return Err(LanaError::InvalidParameters); }
        let logits = self.logits(tokens)?;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let normalizer: f32 = logits.iter().map(|value| (value - max).exp()).sum();
        Ok(-(logits[target] - max - normalizer.ln()))
    }

    pub fn train_next_token(&mut self, tokens: &[usize], target: usize, learning_rate: f32) -> Result<f32, LanaError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 || target >= self.vocabulary {
            return Err(LanaError::InvalidParameters);
        }
        let pooled = self.pool_checked(tokens)?;
        let pre_hidden: Vec<f32> = (0..self.hidden_width).map(|row| self.hidden_bias[row] +
            (0..self.embedding_width).map(|col| self.hidden[row * self.embedding_width + col] * pooled[col]).sum::<f32>()).collect();
        let hidden: Vec<f32> = pre_hidden.iter().map(|value| value.max(0.0)).collect();
        let logits: Vec<f32> = (0..self.vocabulary).map(|row| self.output_bias[row] +
            (0..self.hidden_width).map(|col| self.output[row * self.hidden_width + col] * hidden[col]).sum::<f32>()).collect();
        let mut probability = Self::softmax(&logits)?;
        let loss = -probability[target].ln();
        probability[target] -= 1.0;

        let mut hidden_gradient = vec![0.0; self.hidden_width];
        for row in 0..self.vocabulary {
            for col in 0..self.hidden_width {
                hidden_gradient[col] += probability[row] * self.output[row * self.hidden_width + col];
            }
        }
        for col in 0..self.hidden_width {
            if pre_hidden[col] <= 0.0 { hidden_gradient[col] = 0.0; }
        }
        let mut pooled_gradient = vec![0.0; self.embedding_width];
        for row in 0..self.hidden_width {
            for col in 0..self.embedding_width {
                pooled_gradient[col] += hidden_gradient[row] * self.hidden[row * self.embedding_width + col];
            }
        }
        for row in 0..self.vocabulary {
            for col in 0..self.hidden_width {
                self.output[row * self.hidden_width + col] -= learning_rate * probability[row] * hidden[col];
            }
            self.output_bias[row] -= learning_rate * probability[row];
        }
        for row in 0..self.hidden_width {
            for col in 0..self.embedding_width {
                self.hidden[row * self.embedding_width + col] -= learning_rate * hidden_gradient[row] * pooled[col];
            }
            self.hidden_bias[row] -= learning_rate * hidden_gradient[row];
        }
        let scale = learning_rate / tokens.len() as f32;
        for &token in tokens {
            for col in 0..self.embedding_width {
                self.embedding[token * self.embedding_width + col] -= scale * pooled_gradient[col];
            }
        }
        self.version += 1;
        self.training_history.push(loss);
        self.replay_steps += 1;
        Ok(loss)
    }

    pub fn train_next_token_with_weight_decay(&mut self, tokens: &[usize], target: usize, learning_rate: f32, weight_decay: f32) -> Result<f32, LanaError> {
        if !weight_decay.is_finite() || weight_decay < 0.0 { return Err(LanaError::InvalidParameters); }
        let mut next = self.clone();
        let loss = next.train_next_token(tokens, target, learning_rate)?;
        let factor = 1.0 - learning_rate * weight_decay;
        if !factor.is_finite() || factor < 0.0 { return Err(LanaError::InvalidParameters); }
        for group in [&mut next.embedding, &mut next.hidden, &mut next.hidden_bias, &mut next.output, &mut next.output_bias] {
            for value in group.iter_mut() { *value *= factor; }
        }
        *self = next;
        Ok(loss)
    }

    pub fn train_next_token_clipped(&mut self, tokens: &[usize], target: usize, learning_rate: f32, max_update_norm: f32) -> Result<f32, LanaError> {
        if !max_update_norm.is_finite() || max_update_norm <= 0.0 { return Err(LanaError::InvalidParameters); }
        let mut next = self.clone();
        let loss = next.train_next_token(tokens, target, learning_rate)?;
        let norm_squared: f32 = self.embedding.iter().zip(&next.embedding).chain(self.hidden.iter().zip(&next.hidden)).chain(self.hidden_bias.iter().zip(&next.hidden_bias)).chain(self.output.iter().zip(&next.output)).chain(self.output_bias.iter().zip(&next.output_bias)).map(|(before, after)| (after - before).powi(2)).sum();
        let norm = norm_squared.sqrt();
        if norm > max_update_norm {
            let scale = max_update_norm / norm;
            for (before, after) in self.embedding.iter().zip(&mut next.embedding).chain(self.hidden.iter().zip(&mut next.hidden)).chain(self.hidden_bias.iter().zip(&mut next.hidden_bias)).chain(self.output.iter().zip(&mut next.output)).chain(self.output_bias.iter().zip(&mut next.output_bias)) { *after = *before + (*after - *before) * scale; }
        }
        *self = next;
        Ok(loss)
    }

    pub fn save(&self, path: &Path) -> Result<(), LanaError> {
        let mut data = b"LBRN1".to_vec();
        for value in [self.vocabulary as u64, self.embedding_width as u64, self.hidden_width as u64, self.version, self.seed] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        for values in [&self.embedding, &self.hidden, &self.hidden_bias, &self.output, &self.output_bias] {
            data.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values { data.extend_from_slice(&value.to_le_bytes()); }
        }
        data.extend_from_slice(&(self.memory.len() as u64).to_le_bytes());
        for item in &self.memory {
            data.extend_from_slice(&(item.len() as u64).to_le_bytes());
            data.extend_from_slice(item.as_bytes());
        }
        data.extend_from_slice(&(self.training_history.len() as u64).to_le_bytes());
        for loss in &self.training_history { data.extend_from_slice(&loss.to_le_bytes()); }
        data.extend_from_slice(&self.replay_steps.to_le_bytes());
        let staging = path.with_extension(format!("tmp-{}", std::process::id()));
        let result = (|| {
            let mut file = std::fs::File::create(&staging).map_err(|_| LanaError::Io)?;
            file.write_all(&data).map_err(|_| LanaError::Io)?;
            file.sync_all().map_err(|_| LanaError::Io)?;
            std::fs::rename(&staging, path).map_err(|_| LanaError::Io)?;
            if let Some(parent) = path.parent() { std::fs::File::open(parent).map_err(|_| LanaError::Io)?.sync_all().map_err(|_| LanaError::Io)?; }
            Ok(())
        })();
        if result.is_err() { let _ = std::fs::remove_file(staging); }
        result
    }

    pub fn load(path: &Path) -> Result<Self, LanaError> {
        let mut data = Vec::new();
        std::fs::File::open(path).map_err(|_| LanaError::Io)?.read_to_end(&mut data).map_err(|_| LanaError::Io)?;
        if data.get(..5) != Some(b"LBRN1") { return Err(LanaError::Format); }
        let mut offset = 5;
        let read_u64 = |data: &[u8], offset: &mut usize| -> Result<u64, LanaError> {
            let bytes = data.get(*offset..*offset + 8).ok_or(LanaError::Format)?;
            *offset += 8;
            Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
        };
        let vocabulary = usize::try_from(read_u64(&data, &mut offset)?).map_err(|_| LanaError::Limit)?;
        let embedding_width = usize::try_from(read_u64(&data, &mut offset)?).map_err(|_| LanaError::Limit)?;
        let hidden_width = usize::try_from(read_u64(&data, &mut offset)?).map_err(|_| LanaError::Limit)?;
        let version = read_u64(&data, &mut offset)?;
        let seed = read_u64(&data, &mut offset)?;
        let template = Self::new(vocabulary, embedding_width, hidden_width, seed)?;
        let mut groups = Vec::new();
        for expected in [template.embedding.len(), template.hidden.len(), template.hidden_bias.len(), template.output.len(), template.output_bias.len()] {
            if usize::try_from(read_u64(&data, &mut offset)?).map_err(|_| LanaError::Limit)? != expected { return Err(LanaError::Schema); }
            let mut values = Vec::with_capacity(expected);
            for _ in 0..expected {
                let bytes = data.get(offset..offset + 4).ok_or(LanaError::Format)?;
                offset += 4;
                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                if !value.is_finite() { return Err(LanaError::Schema); }
                values.push(value);
            }
            groups.push(values);
        }
        let mut memory = Vec::new();
        if offset != data.len() {
            let count = usize::try_from(read_u64(&data, &mut offset)?).map_err(|_| LanaError::Limit)?;
            for _ in 0..count {
                let length = usize::try_from(read_u64(&data, &mut offset)?).map_err(|_| LanaError::Limit)?;
                let bytes = data.get(offset..offset + length).ok_or(LanaError::Format)?;
                offset += length;
                memory.push(std::str::from_utf8(bytes).map_err(|_| LanaError::Schema)?.to_owned());
            }
        }
        let mut training_history = Vec::new();
        let mut replay_steps = 0;
        if offset != data.len() {
            let count = usize::try_from(read_u64(&data, &mut offset)?).map_err(|_| LanaError::Limit)?;
            for _ in 0..count {
                let bytes = data.get(offset..offset + 4).ok_or(LanaError::Format)?;
                offset += 4;
                let loss = f32::from_le_bytes(bytes.try_into().unwrap());
                if !loss.is_finite() { return Err(LanaError::Schema); }
                training_history.push(loss);
            }
            replay_steps = read_u64(&data, &mut offset)?;
        }
        if offset != data.len() { return Err(LanaError::Format); }
        Ok(Self { vocabulary, embedding_width, hidden_width, version, seed,
            embedding: groups.remove(0), hidden: groups.remove(0), hidden_bias: groups.remove(0),
            output: groups.remove(0), output_bias: groups.remove(0), memory, training_history, replay_steps })
    }

    fn pool_checked(&self, tokens: &[usize]) -> Result<Vec<f32>, LanaError> {
        if tokens.is_empty() || tokens.iter().any(|&token| token >= self.vocabulary) { return Err(LanaError::InvalidParameters); }
        Ok(self.pool(tokens))
    }

    fn pool(&self, tokens: &[usize]) -> Vec<f32> {
        let mut pooled = vec![0.0; self.embedding_width];
        for &token in tokens { for col in 0..self.embedding_width { pooled[col] += self.embedding[token * self.embedding_width + col]; } }
        for value in &mut pooled { *value /= tokens.len() as f32; }
        pooled
    }

    fn hidden_values(&self, pooled: &[f32]) -> Vec<f32> {
        (0..self.hidden_width).map(|row| (self.hidden_bias[row] +
            (0..self.embedding_width).map(|col| self.hidden[row * self.embedding_width + col] * pooled[col]).sum::<f32>()).max(0.0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_changes_every_named_group() {
        let mut brain = Brain::new(3, 2, 2, 7).unwrap();
        let before = brain.clone();
        let loss = brain.train_next_token(&[0, 1], 2, 0.1).unwrap();
        assert!(loss.is_finite());
        assert_ne!(brain.embedding, before.embedding);
        assert_ne!(brain.hidden, before.hidden);
        assert_ne!(brain.output, before.output);
        assert_eq!(brain.version, 2);
    }

    #[test]
    fn save_load_preserves_all_parameter_groups() {
        let path = std::env::temp_dir().join(format!("lana-brain-{}", std::process::id()));
        let mut brain = Brain::new(3, 2, 2, 7).unwrap();
        brain.train_next_token(&[0, 1], 2, 0.1).unwrap();
        brain.memory.push("remembered fact".to_owned());
        brain.save(&path).unwrap();
        assert_eq!(Brain::load(&path).unwrap(), brain);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn gradients_match_finite_differences() {
        const EPSILON: f32 = 1e-3;
        const RELATIVE_TOLERANCE: f32 = 1e-3;
        const ABSOLUTE_TOLERANCE: f32 = 1e-4;
        let mut brain = Brain::new(3, 2, 2, 7).unwrap();
        brain.embedding.fill(0.2);
        brain.hidden.fill(0.3);
        brain.hidden_bias.fill(0.1);
        brain.output.fill(0.4);
        brain.output_bias.fill(0.05);
        let tokens = [0, 1];
        let target = 2;
        let check = |mut plus: Brain, mut minus: Brain, parameter: fn(&mut Brain) -> &mut f32| {
            *parameter(&mut plus) += EPSILON;
            *parameter(&mut minus) -= EPSILON;
            let numerical = (plus.next_token_loss(&tokens, target).unwrap() - minus.next_token_loss(&tokens, target).unwrap()) / (2.0 * EPSILON);
            let mut trained = brain.clone();
            let before = *parameter(&mut trained);
            trained.train_next_token(&tokens, target, EPSILON).unwrap();
            let analytical = (before - *parameter(&mut trained)) / EPSILON;
            let difference = (analytical - numerical).abs();
            assert!(difference <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * numerical.abs(), "analytical={analytical}, numerical={numerical}");
        };
        check(brain.clone(), brain.clone(), |brain| &mut brain.embedding[0]);
        check(brain.clone(), brain.clone(), |brain| &mut brain.hidden[0]);
        check(brain.clone(), brain.clone(), |brain| &mut brain.output[0]);
    }

    #[test]
    fn losses_reject_invalid_inputs_and_honor_masks() {
        assert!(Brain::binary_cross_entropy(f32::NAN, true).is_err());
        assert!(Brain::binary_cross_entropy(0.5, true).unwrap().is_finite());
        assert_eq!(Brain::masked_cross_entropy(&[vec![0.0, 1.0]], &[1], &[false]), Err(LanaError::InvalidParameters));
        assert!(Brain::masked_cross_entropy(&[vec![0.0, 1.0]], &[1], &[true]).unwrap().is_finite());
    }

    #[test]
    fn weight_decay_is_transactional() {
        let mut brain = Brain::new(3, 2, 2, 7).unwrap();
        let before = brain.clone();
        assert_eq!(brain.train_next_token_with_weight_decay(&[0, 1], 2, 0.1, -1.0), Err(LanaError::InvalidParameters));
        assert_eq!(brain, before);
        brain.train_next_token_with_weight_decay(&[0, 1], 2, 0.1, 0.01).unwrap();
        assert_ne!(brain, before);
    }

    #[test]
    fn clipping_bounds_update_norm() {
        let mut brain = Brain::new(3, 2, 2, 7).unwrap();
        let before = brain.clone();
        brain.train_next_token_clipped(&[0, 1], 2, 10.0, 0.01).unwrap();
        let norm: f32 = before.embedding.iter().zip(&brain.embedding).chain(before.hidden.iter().zip(&brain.hidden)).chain(before.hidden_bias.iter().zip(&brain.hidden_bias)).chain(before.output.iter().zip(&brain.output)).chain(before.output_bias.iter().zip(&brain.output_bias)).map(|(a, b)| (b - a).powi(2)).sum::<f32>().sqrt();
        assert!(norm <= 0.010001);
    }

    #[test]
    fn immutable_training_reports_changed_groups() {
        let brain = Brain::new(3, 2, 2, 7).unwrap();
        let result = brain.trained_next_token(&[0, 1], 2, 0.1).unwrap();
        assert_eq!(result.changed_groups, ["embedding", "hidden", "output"]);
        assert_eq!(brain.version, 1);
        assert_eq!(result.brain.version, 2);
        assert!(brain.trained_next_token(&[], 2, 0.1).is_err());
        assert_eq!(brain.version, 1);
    }

    #[test]
    fn learning_rate_schedule_is_bounded() {
        assert_eq!(Brain::scheduled_learning_rate(1.0, 0, 2, 4).unwrap(), 0.5);
        assert_eq!(Brain::scheduled_learning_rate(1.0, 2, 2, 4).unwrap(), 1.0);
        assert!(Brain::scheduled_learning_rate(1.0, 4, 0, 4).is_err());
    }

    #[test]
    fn numerical_layers_validate_shapes() {
        assert_eq!(Brain::mean_pool(&[vec![1.0, 3.0], vec![3.0, 5.0]]).unwrap(), vec![2.0, 4.0]);
        assert!(Brain::mean_pool(&[vec![1.0], vec![1.0, 2.0]]).is_err());
        let values = Brain::softmax(&[1000.0, 1001.0]).unwrap();
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cpu_layers_validate_and_compute() {
        assert_eq!(Brain::relu(&[-1.0, 2.0]).unwrap(), vec![0.0, 2.0]);
        assert!(Brain::gelu(&[f32::NAN]).is_err());
        assert_eq!(Brain::dense(&[2.0], &[3.0], &[1.0]).unwrap(), vec![7.0]);
        assert!(Brain::dense(&[1.0], &[], &[0.0]).is_err());
        assert!(Brain::layer_norm(&[1.0, 3.0]).unwrap().iter().sum::<f32>().abs() < 1e-5);
        assert_eq!(Brain::causal_mask(2).unwrap(), vec![vec![true, false], vec![true, true]]);
    }
}
