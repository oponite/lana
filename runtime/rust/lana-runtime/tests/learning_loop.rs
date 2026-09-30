//! One bounded learning loop: fit, correct, persist, and test unseen cases.

use std::process::Command;

use lana_bytecode::LanaError;
use lana_runtime::store::{
    store_commit, store_current_revision, store_get, store_open, store_put, Store, StoreOptions,
};
use lana_vm::Value;

type Observation = (f64, bool);

// The caller supplies the rule form. The examples determine its threshold.
fn learn_threshold(examples: &[Observation]) -> Option<f64> {
    if examples.len() < 2 || examples.len() > 32 {
        return None;
    }
    let mut highest_safe = f64::NEG_INFINITY;
    let mut lowest_failure = f64::INFINITY;
    for &(temperature, failure) in examples {
        if !temperature.is_finite() {
            return None;
        }
        if failure {
            lowest_failure = lowest_failure.min(temperature);
        } else {
            highest_safe = highest_safe.max(temperature);
        }
    }
    if !highest_safe.is_finite() || !lowest_failure.is_finite() || highest_safe >= lowest_failure {
        return None;
    }
    Some(highest_safe + (lowest_failure - highest_safe) / 2.0)
}

fn predict(threshold: f64, temperature: f64) -> bool {
    temperature > threshold
}

fn open(path: &str) -> Store {
    store_open(&StoreOptions { schema_version: 1, path: path.into(), timeout_ms: 0 }).unwrap()
}

fn put_observation(store: &mut Store, index: usize, (temperature, failure): Observation) {
    store_put(store, &format!("observation/{index}/temperature"), &Value::number(temperature)).unwrap();
    store_put(store, &format!("observation/{index}/failure"), &Value::boolean(failure)).unwrap();
}

fn observations(store: &Store) -> Vec<Observation> {
    let count = store_get(store, "observation/count").unwrap().as_number() as usize;
    (0..count)
        .map(|index| {
            (
                store_get(store, &format!("observation/{index}/temperature")).unwrap().as_number(),
                store_get(store, &format!("observation/{index}/failure")).unwrap().as_bool(),
            )
        })
        .collect()
}

fn initial(path: &str) {
    let examples = [(60.0, false), (70.0, false), (90.0, true), (100.0, true)];
    let threshold = learn_threshold(&examples).unwrap();
    assert_eq!(threshold, 80.0);
    let mut store = open(path);
    for (index, example) in examples.into_iter().enumerate() {
        put_observation(&mut store, index, example);
    }
    store_put(&mut store, "observation/count", &Value::number(4.0)).unwrap();
    store_put(&mut store, "rule/1", &Value::number(threshold)).unwrap();
    store_put(&mut store, "rule/latest", &Value::number(1.0)).unwrap();
    assert_eq!(store_commit(&mut store).unwrap().revision_id, 1);
}

fn correct(path: &str) {
    let mut store = open(path);
    let mut examples = observations(&store);
    assert_eq!(examples.len(), 4);
    assert!(!predict(store_get(&store, "rule/1").unwrap().as_number(), 75.0));
    let correction = (75.0, true);
    examples.push(correction);
    let threshold = learn_threshold(&examples).unwrap();
    assert_eq!(threshold, 72.5);
    put_observation(&mut store, 4, correction);
    store_put(&mut store, "observation/count", &Value::number(5.0)).unwrap();
    store_put(&mut store, "rule/2", &Value::number(threshold)).unwrap();
    store_put(&mut store, "rule/latest", &Value::number(2.0)).unwrap();
    assert_eq!(store_commit(&mut store).unwrap().revision_id, 2);
}

fn reload(path: &str) {
    let store = open(path);
    let first = store_get(&store, "rule/1").unwrap().as_number();
    let revised = store_get(&store, "rule/2").unwrap().as_number();
    assert_eq!((first, revised), (80.0, 72.5));
    assert_eq!(store_get(&store, "rule/latest").unwrap().as_number(), 2.0);
    assert_eq!(observations(&store), vec![(60.0, false), (70.0, false), (90.0, true), (100.0, true), (75.0, true)]);

    let held_out = [(65.0, false), (71.0, false), (73.0, true), (74.0, true), (85.0, true), (95.0, true)];
    let score = |threshold| held_out.iter().filter(|&&(temperature, outcome)| predict(threshold, temperature) == outcome).count();
    assert_eq!(score(first), 4);
    assert_eq!(score(revised), 6);
    assert!(!predict(first, 73.0));
    assert!(predict(revised, 73.0));
    println!("learned {first}, corrected {revised}, held-out 4/6 -> 6/6, recovered after restart");
}

fn reject(path: &str) {
    let mut store = open(path);
    let before = store_current_revision(&store).unwrap().revision_id;
    let mut examples = observations(&store);
    examples.push((90.0, false));
    assert_eq!(learn_threshold(&examples), None);
    assert!(matches!(store_commit(&mut store), Err(LanaError::UnsupportedOperation)));
    assert_eq!(store_current_revision(&store).unwrap().revision_id, before);
    assert_eq!(store_get(&store, "rule/latest").unwrap().as_number(), 2.0);
    assert_eq!(observations(&store).len(), 5);
}

#[test]
fn phase_helper() {
    let Ok(phase) = std::env::var("LANA_LEARNING_PHASE") else { return };
    let path = std::env::var("LANA_LEARNING_STORE").unwrap();
    match phase.as_str() {
        "initial" => initial(&path),
        "correct" => correct(&path),
        "reload" => reload(&path),
        "reject" => reject(&path),
        _ => panic!("unknown phase: {phase}"),
    }
}

#[test]
fn learning_loop() {
    let path = std::env::temp_dir().join(format!(
        "lana-learning-loop-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let executable = std::env::current_exe().unwrap();
    for phase in ["initial", "correct", "reload", "reject", "reload"] {
        let output = Command::new(&executable)
            .args(["--exact", "phase_helper", "--nocapture"])
            .env("LANA_LEARNING_PHASE", phase)
            .env("LANA_LEARNING_STORE", &path)
            .output()
            .unwrap();
        assert!(output.status.success(), "{phase}: {}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        if phase == "reload" {
            print!("{}", String::from_utf8_lossy(&output.stdout));
        }
    }
    std::fs::remove_dir_all(path).unwrap();
}
