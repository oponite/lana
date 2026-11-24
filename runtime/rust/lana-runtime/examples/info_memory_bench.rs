//! Benchmark for the information-aware memory runtime proof of concept.
//!
//! Compares three stores on three reproducible workloads:
//!
//! * `conventional` — each object kept as an uncompressed `Box<[State]>`.
//! * `deflate` — each object compressed independently with DEFLATE (flate2,
//!   default level) after allocation, inflated on recall.
//! * `lana` — `lana_runtime::info_memory::InfoMemory`.
//!
//! Memory is measured, not estimated: a counting global allocator records
//! live and peak heap bytes (payload plus every table, header and index the
//! store allocates), and on Linux the process `VmRSS`/`VmHWM` are read too.
//! Each (workload, store, repeat) runs in a fresh child process.
//!
//! ```text
//! cargo run --release --locked -p lana-runtime --example info_memory_bench -- \
//!     [--objects N] [--len L] [--seed S] [--repeats R] [--out results.json]
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Read;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression;
use lana_bytecode::LanaError;
use lana_runtime::info_memory::InfoMemory;
use lana_vm::state::{make_complex, mix, State};
use lana_vm::Rng;
use serde_json::{json, Value};

// ---------------------------------------------------------------- allocator

struct Counting;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = System.realloc(ptr, layout, new_size);
        if !out.is_null() {
            CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
            grow(new_size);
        }
        out
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn heap_now() -> usize {
    CURRENT.load(Ordering::Relaxed)
}

fn reset_peak() {
    PEAK.store(heap_now(), Ordering::Relaxed);
}

fn heap_peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// `(VmRSS, VmHWM)` in bytes, Linux only.
fn rss() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let field = |name: &str| -> Option<u64> {
        let line = status.lines().find(|line| line.starts_with(name))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    };
    Some((field("VmRSS:")?, field("VmHWM:")?))
}

// ---------------------------------------------------------------- workloads

const STATE_BYTES: usize = std::mem::size_of::<State>();

fn rng_for(seed: u64, stream: u64) -> Rng {
    let mut rng = Rng::new();
    rng.seed(seed ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    rng
}

fn uniform(rng: &mut Rng) -> f64 {
    let high = (rng.random() as u64) << 21;
    let low = (rng.random() as u64) >> 11;
    (high | low) as f64 / (1u64 << 53) as f64
}

/// A uniformly random valid state built with Lana's `make_complex`.
fn random_state(rng: &mut Rng) -> State {
    let p = 1e-6 + (1.0 - 2e-6) * uniform(rng);
    let radius = uniform(rng).sqrt();
    let angle = std::f64::consts::TAU * uniform(rng);
    let mut out = State::default();
    let status = make_complex(p, radius * angle.cos(), radius * angle.sin(), &mut out);
    assert_eq!(status, LanaError::Ok);
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Random,
    Redundant,
    Structured,
}

const KINDS: [Kind; 3] = [Kind::Random, Kind::Redundant, Kind::Structured];

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Random => "random",
            Kind::Redundant => "redundant",
            Kind::Structured => "structured",
        }
    }
    fn parse(name: &str) -> Kind {
        *KINDS.iter().find(|k| k.name() == name).expect("workload")
    }
}

/// Redundant: 8 templates, each made of runs of 256 copies drawn from a
/// 16-state palette; object `i` is template `i % 8`.
const TEMPLATES: usize = 8;
const RUN: usize = 256;
const PALETTE: usize = 16;
/// Structured: 4 high-entropy priors; object `i` is prior `i % 4` with 1% of
/// positions (`--update-per-mille`, default 10) updated by Lana's `mix`
/// against random evidence.
const PRIORS: usize = 4;

struct Workload {
    kind: Kind,
    len: usize,
    seed: u64,
    update_per_mille: usize,
    shared: Vec<Vec<State>>,
}

impl Workload {
    fn new(kind: Kind, len: usize, seed: u64, update_per_mille: usize) -> Self {
        let shared = match kind {
            Kind::Random => Vec::new(),
            Kind::Redundant => (0..TEMPLATES)
                .map(|t| {
                    let mut rng = rng_for(seed, 1_000_000 + t as u64);
                    let palette: Vec<State> =
                        (0..PALETTE).map(|_| random_state(&mut rng)).collect();
                    (0..len)
                        .map(|i| palette[((i / RUN) * 7 + t) % PALETTE])
                        .collect()
                })
                .collect(),
            Kind::Structured => (0..PRIORS)
                .map(|t| {
                    let mut rng = rng_for(seed, 2_000_000 + t as u64);
                    (0..len).map(|_| random_state(&mut rng)).collect()
                })
                .collect(),
        };
        Self {
            kind,
            len,
            seed,
            update_per_mille,
            shared,
        }
    }

    /// Object `i`, regenerated identically on every call.
    fn object(&self, i: usize) -> Vec<State> {
        let mut rng = rng_for(self.seed, i as u64);
        match self.kind {
            Kind::Random => (0..self.len).map(|_| random_state(&mut rng)).collect(),
            Kind::Redundant => self.shared[i % TEMPLATES].clone(),
            Kind::Structured => {
                let mut out = self.shared[i % PRIORS].clone();
                let updates = (self.len * self.update_per_mille).div_ceil(1000);
                for _ in 0..updates {
                    let at = rng.random() as usize % self.len;
                    let evidence = random_state(&mut rng);
                    let weight = 0.05 + 0.9 * uniform(&mut rng);
                    let prior = out[at];
                    let status = mix(&prior, &evidence, weight, &mut out[at]);
                    assert_eq!(status, LanaError::Ok);
                }
                out
            }
        }
    }
}

fn exact(a: &[State], b: &[State]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.p.to_bits() == y.p.to_bits()
                && x.d_re.to_bits() == y.d_re.to_bits()
                && x.d_im.to_bits() == y.d_im.to_bits()
        })
}

// ---------------------------------------------------------------- stores

trait Store {
    fn allocate(&mut self, states: Vec<State>) -> u32;
    fn compact(&mut self, handle: u32);
    fn recall(&mut self, handle: u32) -> Vec<State>;
    fn release(&mut self, handle: u32);
    fn shrink(&mut self);
    fn profile(&self) -> Value {
        Value::Null
    }
}

#[derive(Default)]
struct Conventional {
    slots: Vec<Option<Box<[State]>>>,
}

impl Store for Conventional {
    fn allocate(&mut self, states: Vec<State>) -> u32 {
        self.slots.push(Some(states.into_boxed_slice()));
        (self.slots.len() - 1) as u32
    }
    fn compact(&mut self, _: u32) {}
    fn recall(&mut self, handle: u32) -> Vec<State> {
        self.slots[handle as usize].as_ref().unwrap().to_vec()
    }
    fn release(&mut self, handle: u32) {
        self.slots[handle as usize] = None;
    }
    fn shrink(&mut self) {
        self.slots.shrink_to_fit();
    }
}

enum Packed {
    Raw(Box<[State]>),
    Deflated { len: usize, bytes: Box<[u8]> },
}

#[derive(Default)]
struct Deflate {
    slots: Vec<Option<Packed>>,
}

impl Store for Deflate {
    fn allocate(&mut self, states: Vec<State>) -> u32 {
        self.slots
            .push(Some(Packed::Raw(states.into_boxed_slice())));
        (self.slots.len() - 1) as u32
    }
    fn compact(&mut self, handle: u32) {
        let slot = &mut self.slots[handle as usize];
        let Some(Packed::Raw(states)) = slot else {
            return;
        };
        let mut bytes = Vec::with_capacity(states.len() * STATE_BYTES);
        for s in states.iter() {
            bytes.extend_from_slice(&s.p.to_le_bytes());
            bytes.extend_from_slice(&s.d_re.to_le_bytes());
            bytes.extend_from_slice(&s.d_im.to_le_bytes());
        }
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, &bytes).unwrap();
        let packed = encoder.finish().unwrap().into_boxed_slice();
        *slot = Some(Packed::Deflated {
            len: states.len(),
            bytes: packed,
        });
    }
    fn recall(&mut self, handle: u32) -> Vec<State> {
        match self.slots[handle as usize].as_ref().unwrap() {
            Packed::Raw(states) => states.to_vec(),
            Packed::Deflated { len, bytes } => {
                let mut raw = Vec::with_capacity(len * STATE_BYTES);
                DeflateDecoder::new(&bytes[..])
                    .read_to_end(&mut raw)
                    .unwrap();
                let f = |c: &[u8]| f64::from_le_bytes(c.try_into().unwrap());
                raw.chunks_exact(STATE_BYTES)
                    .map(|c| State {
                        p: f(&c[0..8]),
                        d_re: f(&c[8..16]),
                        d_im: f(&c[16..24]),
                    })
                    .collect()
            }
        }
    }
    fn release(&mut self, handle: u32) {
        self.slots[handle as usize] = None;
    }
    fn shrink(&mut self) {
        self.slots.shrink_to_fit();
    }
}

struct Lana {
    memory: InfoMemory,
    kinds: [usize; 4],
}

impl Store for Lana {
    fn allocate(&mut self, states: Vec<State>) -> u32 {
        self.memory.allocate(states).unwrap()
    }
    fn compact(&mut self, handle: u32) {
        let kind = self.memory.compact(handle).unwrap();
        self.kinds[kind as usize] += 1;
    }
    fn recall(&mut self, handle: u32) -> Vec<State> {
        self.memory.recall(handle).unwrap()
    }
    fn release(&mut self, handle: u32) {
        self.memory.release(handle).unwrap();
    }
    fn shrink(&mut self) {
        self.memory.shrink();
    }
    fn profile(&self) -> Value {
        let p = self.memory.profile();
        json!({
            "compaction": {"raw": self.kinds[0], "runs": self.kinds[1],
                           "delta": self.kinds[2], "shared": self.kinds[3]},
            "live_blocks": p.live_blocks,
            "logical_bytes": p.logical_bytes,
            "self_reported_payload_bytes": p.payload_bytes,
            "self_reported_metadata_bytes": p.metadata_bytes,
            "self_reported_total_bytes": p.payload_bytes + p.metadata_bytes,
        })
    }
}

// ---------------------------------------------------------------- one run

fn median(samples: &mut [u64]) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn p99(samples: &mut [u64]) -> u64 {
    samples.sort_unstable();
    samples[(samples.len() * 99 / 100).min(samples.len() - 1)]
}

fn run_one(
    kind: Kind,
    store_name: &str,
    objects: usize,
    len: usize,
    seed: u64,
    update: usize,
) -> Value {
    let workload = Workload::new(kind, len, seed, update);
    // Instrumentation buffers are allocated before the baseline is taken.
    let mut handles = Vec::with_capacity(objects);
    let mut t_alloc = Vec::with_capacity(objects);
    let mut t_compact = Vec::with_capacity(objects);
    let mut t_recall = Vec::with_capacity(objects);
    let mut t_release = Vec::with_capacity(objects);
    let mut store: Box<dyn Store> = match store_name {
        "conventional" => Box::new(Conventional::default()),
        "deflate" => Box::new(Deflate::default()),
        "lana" => Box::new(Lana {
            memory: InfoMemory::new(),
            kinds: [0; 4],
        }),
        other => panic!("unknown store {other}"),
    };
    let rss_start = rss();
    let base = heap_now();
    reset_peak();

    // ALLOCATE + COMPACT (online: each object is compacted right after it
    // is allocated, as a memory manager would).
    let mut generated = 0usize;
    for i in 0..objects {
        let object = workload.object(i);
        generated = generated.max(object.capacity() * STATE_BYTES);
        let start = Instant::now();
        let handle = store.allocate(object);
        t_alloc.push(start.elapsed().as_nanos() as u64);
        let start = Instant::now();
        store.compact(handle);
        t_compact.push(start.elapsed().as_nanos() as u64);
        handles.push(handle);
    }
    store.shrink();
    let steady = heap_now() - base;
    let peak_fill = heap_peak() - base;
    let rss_steady = rss();
    // The profile report is benchmark output; exclude its own bytes from the
    // after-release/after-drop residue.
    let before_profile = heap_now();
    let profile = store.profile();
    let profile_bytes = heap_now() - before_profile;

    // RECALL + verify against an independently regenerated original.
    let mut exact_objects = 0usize;
    for (i, &handle) in handles.iter().enumerate() {
        let expected = workload.object(i);
        let start = Instant::now();
        let got = store.recall(handle);
        t_recall.push(start.elapsed().as_nanos() as u64);
        exact_objects += exact(&got, &expected) as usize;
    }
    let peak_total = heap_peak() - base;

    // RELEASE.
    for &handle in &handles {
        let start = Instant::now();
        store.release(handle);
        t_release.push(start.elapsed().as_nanos() as u64);
    }
    let after_release = heap_now().saturating_sub(base + profile_bytes);
    drop(store);
    let after_drop = heap_now().saturating_sub(base + profile_bytes);

    let logical = objects * len * STATE_BYTES;
    let sum = |v: &[u64]| v.iter().sum::<u64>();
    json!({
        "workload": kind.name(),
        "store": store_name,
        "objects": objects,
        "len": len,
        "seed": seed,
        "update_per_mille": update,
        "logical_bytes": logical,
        "steady_heap_bytes": steady,
        // The fill-phase peak includes one generated input object (the same
        // `generator_transient_bytes` for every store).
        "peak_heap_bytes_fill": peak_fill,
        "peak_heap_bytes_total": peak_total,
        "generator_transient_bytes": generated,
        "heap_after_release_bytes": after_release,
        "heap_after_drop_bytes": after_drop,
        "effective_ratio": logical as f64 / steady.max(1) as f64,
        "exact_objects": exact_objects,
        "exact": exact_objects == objects,
        "rss_start_bytes": rss_start.map(|r| r.0),
        "rss_steady_bytes": rss_steady.map(|r| r.0),
        "rss_hwm_bytes": rss_steady.map(|r| r.1),
        "allocate_ns": {"median": median(&mut t_alloc), "p99": p99(&mut t_alloc), "total": sum(&t_alloc)},
        "compact_ns": {"median": median(&mut t_compact), "p99": p99(&mut t_compact), "total": sum(&t_compact)},
        "recall_ns": {"median": median(&mut t_recall), "p99": p99(&mut t_recall), "total": sum(&t_recall)},
        "release_ns": {"median": median(&mut t_release), "p99": p99(&mut t_release), "total": sum(&t_release)},
        "profile": profile,
    })
}

/// Reference only: every object in allocation order through ONE DEFLATE
/// stream. No random access (recall would inflate everything before it), so
/// it is not a store; it bounds what generic DEFLATE finds without knowing
/// object boundaries or structure.
fn stream_reference(kind: Kind, objects: usize, len: usize, seed: u64, update: usize) -> Value {
    let workload = Workload::new(kind, len, seed, update);
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    let mut nanos = 0u128;
    let mut bytes = Vec::with_capacity(len * STATE_BYTES);
    for i in 0..objects {
        bytes.clear();
        for s in workload.object(i) {
            bytes.extend_from_slice(&s.p.to_le_bytes());
            bytes.extend_from_slice(&s.d_re.to_le_bytes());
            bytes.extend_from_slice(&s.d_im.to_le_bytes());
        }
        let start = Instant::now();
        std::io::Write::write_all(&mut encoder, &bytes).unwrap();
        nanos += start.elapsed().as_nanos();
    }
    let start = Instant::now();
    let compressed = encoder.finish().unwrap().len();
    nanos += start.elapsed().as_nanos();
    let logical = objects * len * STATE_BYTES;
    json!({"workload": kind.name(), "compressed_bytes": compressed,
           "ratio": logical as f64 / compressed as f64, "total_ns": nanos as u64})
}

// ---------------------------------------------------------------- driver

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse().ok().expect("argument value"))
        .unwrap_or(default)
}

const STORES: [&str; 3] = ["conventional", "deflate", "lana"];

fn mib(bytes: f64) -> String {
    format!("{:.2}", bytes / (1024.0 * 1024.0))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let objects: usize = arg(&args, "--objects", 1024);
    let len: usize = arg(&args, "--len", 4096);
    let seed: u64 = arg(&args, "--seed", 20261124);
    let update: usize = arg(&args, "--update-per-mille", 10);
    if let Some(i) = args.iter().position(|a| a == "--child") {
        let kind = Kind::parse(&args[i + 1]);
        let result = run_one(kind, &args[i + 2], objects, len, seed, update);
        println!("{result}");
        return;
    }
    let repeats: usize = arg(&args, "--repeats", 3);
    let out: String = arg(&args, "--out", String::new());
    let exe = std::env::current_exe().expect("current executable");

    let mut runs = Vec::new();
    for repeat in 0..repeats {
        for kind in KINDS {
            for store in STORES {
                let output = Command::new(&exe)
                    .args(["--child", kind.name(), store])
                    .args(["--objects", &objects.to_string()])
                    .args(["--len", &len.to_string()])
                    .args(["--seed", &seed.to_string()])
                    .args(["--update-per-mille", &update.to_string()])
                    .output()
                    .expect("child process");
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let mut value: Value = serde_json::from_slice(&output.stdout).expect("child json");
                value["repeat"] = json!(repeat);
                runs.push(value);
            }
        }
    }

    // Summary: memory is deterministic per configuration (checked); timings
    // are the median across repeats of each run's per-object median.
    println!(
        "objects={objects} len={len} ({} KiB/object) seed={seed} update_per_mille={update} repeats={repeats}\n",
        len * STATE_BYTES / 1024
    );
    println!("| workload | store | steady heap MiB | peak heap MiB (fill) | ratio | exact | VmHWM MiB | alloc µs | compact µs | recall µs | compact total ms |");
    println!("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|");
    let mut summary = Vec::new();
    for kind in KINDS {
        for store in STORES {
            let rows: Vec<&Value> = runs
                .iter()
                .filter(|r| r["workload"] == kind.name() && r["store"] == store)
                .collect();
            let steady: Vec<u64> = rows
                .iter()
                .map(|r| r["steady_heap_bytes"].as_u64().unwrap())
                .collect();
            let deterministic = steady.iter().all(|&s| s == steady[0]);
            let all_exact = rows.iter().all(|r| r["exact"] == true);
            let med = |path: &[&str]| -> f64 {
                let mut values: Vec<u64> = rows
                    .iter()
                    .map(|r| {
                        let mut v = *r;
                        for p in path {
                            v = &v[*p];
                        }
                        v.as_u64().unwrap_or(0)
                    })
                    .collect();
                median(&mut values) as f64
            };
            let row = &rows[0];
            let line = json!({
                "workload": kind.name(), "store": store,
                "steady_heap_bytes": steady[0], "memory_deterministic": deterministic,
                "peak_heap_bytes_fill": row["peak_heap_bytes_fill"],
                "effective_ratio": row["effective_ratio"], "exact": all_exact,
                "rss_hwm_bytes_median": med(&["rss_hwm_bytes"]),
                "allocate_median_ns": med(&["allocate_ns", "median"]),
                "compact_median_ns": med(&["compact_ns", "median"]),
                "recall_median_ns": med(&["recall_ns", "median"]),
                "recall_p99_ns": med(&["recall_ns", "p99"]),
                "compact_total_ns": med(&["compact_ns", "total"]),
                "allocate_total_ns": med(&["allocate_ns", "total"]),
                "recall_total_ns": med(&["recall_ns", "total"]),
                "heap_after_release_bytes": row["heap_after_release_bytes"],
                "profile": row["profile"],
            });
            println!(
                "| {} | {} | {} | {} | {:.2} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1} |",
                kind.name(),
                store,
                mib(steady[0] as f64),
                mib(row["peak_heap_bytes_fill"].as_f64().unwrap()),
                row["effective_ratio"].as_f64().unwrap(),
                if all_exact { "yes" } else { "NO" },
                mib(med(&["rss_hwm_bytes"])),
                med(&["allocate_ns", "median"]) / 1e3,
                med(&["compact_ns", "median"]) / 1e3,
                med(&["recall_ns", "median"]) / 1e3,
                med(&["compact_ns", "total"]) / 1e6,
            );
            assert!(
                deterministic,
                "memory varied across repeats for {} {store}",
                kind.name()
            );
            summary.push(line);
        }
    }
    println!("\nReference (not a store): whole-store single DEFLATE stream, no random access\n");
    println!("| workload | compressed MiB | ratio | total ms |");
    println!("|---|---:|---:|---:|");
    let mut references = Vec::new();
    for kind in KINDS {
        let r = stream_reference(kind, objects, len, seed, update);
        println!(
            "| {} | {} | {:.2} | {:.1} |",
            kind.name(),
            mib(r["compressed_bytes"].as_f64().unwrap()),
            r["ratio"].as_f64().unwrap(),
            r["total_ns"].as_f64().unwrap() / 1e6
        );
        references.push(r);
    }
    if !out.is_empty() {
        let document = json!({
            "stream_reference": references,
            "config": {"objects": objects, "len": len, "seed": seed, "repeats": repeats,
                       "update_per_mille": update,
                       "os": std::env::consts::OS, "arch": std::env::consts::ARCH},
            "summary": summary,
            "runs": runs,
        });
        std::fs::write(
            &out,
            serde_json::to_string_pretty(&document).unwrap() + "\n",
        )
        .expect("write results");
        eprintln!("wrote {out}");
    }
}
