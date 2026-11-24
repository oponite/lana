# Information-aware memory runtime — proof of concept

Status: experimental, not part of the language, spec, bytecode, or VM.
Nothing here changes `SPEC.md`, `BYTECODE.md`, or `VM.md`.

## Hypothesis

Lana can increase effective memory capacity through information-aware
compaction and on-demand reconstruction, outperforming conventional memory
management on suitable workloads.

## What was built

| File | Purpose |
|---|---|
| `runtime/rust/lana-runtime/src/info_memory.rs` | The runtime (`InfoMemory`) and its unit tests |
| `runtime/rust/lana-runtime/examples/info_memory_bench.rs` | Workloads, baselines, measurement harness |
| `docs/dev/info-memory-poc/run.sh` | Regenerates every result file in this directory |
| `docs/dev/info-memory-poc/*.json`, `*.md` | Raw per-run results and summary tables |

`InfoMemory` is an application-level library store for *information
objects*. Each object is an ordered field of `lana_vm::State` values, Lana's
core information representation `(p, d_re, d_im)`. The store supports five
operations:

| Operation | Behavior |
|---|---|
| `allocate(Vec<State>)` | Validates every state with Lana's `state_valid`, then stores the object uncompacted and returns a handle. |
| `compact(handle)` | Chooses the smallest exact representation it can find. **Shared** points at an existing object with identical content: it looks the object up by Lana's SHA-256, then confirms with a bitwise comparison. **Runs** stores runs of identical states. **Delta** stores a sparse patch over a resident raw object of the same length. **Raw** keeps the object as it is. |
| `recall(handle)` | Rebuilds the object bit-for-bit. Equality is on the `f64` bit patterns, so `-0.0 ≠ 0.0`. |
| `release(handle)` | Reference-counted. A delta base stays resident until every object patched against it is released. |
| `profile()` | Reports the store's own byte accounting, the representation counts, and the count and wall time of each operation. |

Lana primitives reused:

- `State`, `state_valid`, `make_complex`, `mix`
- `sha256`
- the deterministic PCG32 `Rng`
- the field encoding of the `LST1` state codec, which is used as the digest input

No language syntax, opcode, host call, or bytecode change was introduced.
`flate2` was already a workspace dependency (through `lana-cli`). It is
added only as a dev-dependency of `lana-runtime`, for the baseline.

## Method

**Workloads.** All are reproducible from a seed. Objects are regenerated on
demand, so the harness never holds a second copy while it measures. The
main configuration is 1024 objects × 4096 states, which is 96 KiB per object
and 96 MiB logical per workload.

| Workload | Construction |
|---|---|
| `random` | Every state is drawn uniformly over valid states, using `make_complex`. This is high entropy. |
| `redundant` | 8 templates, each built from runs of 256 identical states drawn from a 16-state palette. Object `i` is template `i mod 8`. |
| `structured` | 4 high-entropy prior fields. Object `i` is prior `i mod 4` with 1% of positions (sampled with replacement) updated by Lana's `mix` against random evidence with a random weight. The objects are related to each other but no two are identical. |

**Stores compared.** All three use the same harness. Each object is compacted
right after it is allocated (online), and recall returns an owned copy.

| Store | Behavior |
|---|---|
| `conventional` | The object is kept as `Box<[State]>`. |
| `deflate` | Each object is compressed on its own with DEFLATE (flate2 default level 6, 24 bytes per state little-endian). Recall inflates it. |
| `lana` | `InfoMemory`. |
| *reference* | One DEFLATE stream over the *whole store* in allocation order. It has no random access, so it is not a usable store; it shows what generic DEFLATE finds across objects when it sees everything. |

**Measurement.** Memory figures are measured, not modeled:

- **Heap.** A counting `#[global_allocator]` records live and peak heap
  bytes from a baseline taken just before the store is created. The count
  covers payload, handle and slot tables, hash maps, and allocator-requested
  sizes. It does not include malloc's internal per-chunk headers.
- **Process memory.** On Linux the harness also records `VmRSS` and `VmHWM`
  (process-wide, so they include a baseline of about 3 MiB).
- **Isolation.** Each (workload, store, repeat) runs in a fresh child process.
- **Repeats.** There are 3 repeats. The harness asserts that the memory
  figures are byte-identical across repeats. Timings are the median of the
  per-run medians.
- **Accuracy.** Each recalled object is compared bitwise against an
  independently regenerated original.

The fill-phase peak includes one generated input object, which is the same
for every store.

**Machine.** Linux x86_64, 4 vCPU Intel Xeon @ 2.30 GHz, Rust 1.97, release
profile (`lto = "fat"`). Timings are machine-local evidence. The memory
figures are deterministic for a given seed and configuration.

Reproduce:

```bash
docs/dev/info-memory-poc/run.sh                 # about 1 minute on the machine above
cargo test --locked -p lana-runtime --lib info_memory
```

## Results

### Main configuration: 1024 × 96 KiB, 96 MiB logical (`main.md`)

| workload | store | steady heap MiB | peak heap MiB | ratio vs logical | exact | compact µs/obj | recall µs/obj |
|---|---|---:|---:|---:|---|---:|---:|
| random | conventional | 96.02 | 96.02 | 1.00 | yes | 0 | 24.7 |
| random | deflate | **91.90** | 92.45 | 1.04 | yes | 2228 | 307 |
| random | lana | 96.17 | 96.17 | 1.00 | yes | 343 | 26.9 |
| redundant | conventional | 96.02 | 96.02 | 1.00 | yes | 0 | 24.5 |
| redundant | deflate | 0.74 | 1.26 | 129 | yes | 144 | 32.6 |
| redundant | lana | **0.013** | 0.11 | 7638 | yes | 336 | 14.4 |
| structured | conventional | 96.02 | 96.02 | 1.00 | yes | 0 | 26.1 |
| structured | deflate | 91.91 | 92.47 | 1.04 | yes | 2324 | 316 |
| structured | lana | **2.75** | 2.85 | 34.9 | yes | 351 | 16.4 |

The whole-store DEFLATE reference reaches 91.84 MiB on `structured`, a ratio
of 1.05. Even with the whole store in one stream, DEFLATE's 32 KiB window
cannot reach a related object 4 × 96 KiB back.

On `structured`, Lana uses **33.4× less memory than per-object DEFLATE** and
**34.9× less than conventional storage**. Its recall is faster than
conventional (16 vs 26 µs per object) and about 19× faster than inflating
(16 vs 316 µs).

Lana's `allocate` costs about 11 µs per object, against about 0 for the
others. That is the cost of validating every state, which the baselines do
not do. Lana's compaction costs about 340–350 µs per object, roughly
270 MiB/s, while DEFLATE runs at about 41 MiB/s on high-entropy data.

### Sensitivity: how related must objects be? (256 × 96 KiB, 24 MiB logical)

This table shows `structured` steady heap in MiB as the update rate rises.

| positions updated per object | conventional | deflate | lana |
|---|---:|---:|---:|
| 1% (`seed-7.md`, seed 7) | 24.00 | 22.97 | **0.96** |
| 10% (`update-100.md`) | 24.00 | 22.98 | **5.42** |
| 50% (`update-500.md`) | 24.00 | 22.98 | **17.91** |
| 90% (`update-900.md`) | 24.00 | **22.98** | 23.60 |

At a 90% update rate, Lana loses to DEFLATE but still beats conventional
storage.

### Sensitivity: small objects (16384 × 1.5 KiB, 24 MiB logical, `small-objects.md`)

| workload | conventional | deflate | lana | whole-store DEFLATE reference |
|---|---:|---:|---:|---:|
| random | **24.25** | **24.19** | 26.66 (+9.9%) | 22.96 |
| redundant | 24.25 | 1.40 | **0.13** | 0.08 |
| structured | 24.25 | 24.23 | **3.53** | 0.58 |

### Runtime self-report vs measured

`profile()` agrees with the counting allocator to within 0.33% on 96 KiB
objects (2,875,327 vs 2,884,872 bytes on `structured`). On 1.5 KiB objects
it undercounts by 4.1%, because its hash-table overhead estimate is too low.
Use the measured figures.

## Success criteria

| Criterion | Result |
|---|---|
| Exact reconstruction | **Met.** Every object in every run of every configuration was bit-identical (6 configurations × 3 workloads × 3 stores × 3 repeats). Unit tests cover each representation, release ordering, and `-0.0`. |
| Lower total memory than conventional | **Met on redundant and structured data. Not met on random data:** +0.16% at 96 KiB per object and +9.9% at 1.5 KiB per object, all of it metadata. |
| Measurable advantage over standard compression on a structured workload | **Met.** 2.75 MiB against 91.91 MiB at 1% updates; the advantage holds up to at least a 50% update rate. |
| Reproducible, including negative findings | **Met.** `run.sh` regenerates every number. Memory figures are byte-identical across repeats and reruns. The negative findings are listed below. |

## Negative findings and threats to validity

1. **No gain on high-entropy data.** Lana is 0.16% larger than conventional
   storage on random data and 4.6% larger than DEFLATE, which still gets
   about 4% out of these floats. With small objects the metadata overhead is
   9.9%: about 170 bytes per object in measured heap, against 16 bytes for
   conventional storage (block record, digest-map entry, handle and
   base-pool slots).
2. **Low relatedness defeats delta encoding.** At a 90% update rate DEFLATE
   wins: 22.98 vs 23.60 MiB.
3. **Generic compression does better when its window covers related
   objects.** On small objects, one DEFLATE stream reaches 0.58 MiB on
   `structured`, against Lana's 3.53 MiB, and 0.08 vs 0.13 MiB on
   `redundant`. That stream gives up random access and independent release,
   but the result shows Lana's advantage on large objects comes partly from
   DEFLATE's 32 KiB window. Long-window or dictionary compressors (for
   example `zstd --long`, or zstd with trained dictionaries) were not tested
   and would probably narrow the gap.
4. **Compaction is not free.** Lana spends about 350 ms of CPU per 96 MiB
   compacted, against about 0 for conventional storage. Per-object DEFLATE
   is faster on run-heavy data (144 vs 336 µs per object). Validation adds
   about 11 µs per 96 KiB allocation.
5. **RELEASE does not shrink the tables.** After every object is released,
   178 KiB (main) or 2.8 MiB (small objects) of slot-table capacity remains
   until the store is dropped. After the drop, residual heap is 0 bytes for
   every store.
6. **The workloads are synthetic and favor alignment.** Delta encoding only
   matches objects of equal length at the same positions. Insertions or
   shifts would defeat it. No real Lana program's working set was measured.
7. **A heuristic was tuned after the first sweep.** The first version
   skipped a delta base when more than 50% of 32 sampled positions differed.
   That threshold was inconsistent with the patch cost model, under which a
   patch pays off up to about 86% differing. At a 50% update rate the first
   version stored 231 of 256 objects raw and lost to DEFLATE (23.42 vs
   22.98 MiB). The pre-check now uses the same cost model. Every published
   number comes from the fixed code. Only the 50% and 90% results changed;
   the others are byte-identical.
8. **Nothing here depends on Lana's probabilistic semantics.** The gains
   come from a store-wide view (dedup and delta across objects) and from
   knowing the element width (24-byte `State` records). The same techniques
   would work for any fixed-width record store. This supports the hypothesis
   narrowly. It does not show that Lana's representations are inherently
   more memory-efficient.

## Conclusion

**The architecture merits further development as a targeted store, not as a
general memory manager.**

On workloads made of many related, aligned information fields, the measured
results are large: 35× less RAM than conventional storage and 33× less than
per-object DEFLATE, with exact reconstruction and recall faster than a plain
copy. Examples of such workloads are ensembles of belief states, versioned
snapshots of a field, and repeated priors with local evidence updates.

On high-entropy data, Lana gains nothing and adds metadata. With small
objects, a generic compressor with enough window beats it, at the cost of
random access.

Recommended next steps, in order:

1. **Measure real workloads.** Before more engineering, capture working sets
   from real Lana programs or datasets. The hypothesis stands or falls on
   whether real working sets have this aligned cross-object structure.
2. **Compare against a long-window compressor.** Add zstd with a long window
   or a trained dictionary as a baseline.
3. **Cut per-object metadata.** Slim the `Block` record, and compact the
   handle and slot tables on release.
4. **Make compaction cheaper and more tolerant.** Replace the full SHA-256 on
   every compaction with a cheap hash plus verification, and add
   shift-tolerant matching.
