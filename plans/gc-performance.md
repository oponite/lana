# Collector pause target and evidence

## Target

- A task safepoint performs one young-generation collection or one bounded
  incremental mark slice.
- The default incremental slice traces at most 128 objects.
- The acceptance target is a 10 ms p99 safepoint on the supported release
  build for the 20,000-node collector stress graph.
- Full stop-the-world collection is reserved for explicit collection, severe
  memory pressure, invariant fallback, or shutdown and is not subject to the
  routine-safepoint target.

## Current evidence

- `lana_gc_tests` covers a 20,000-node deep graph, cycles, young reclamation,
  survivor promotion, old-to-young barriers, shared promotion, and one-object
  incremental slices.
- The bounded 128-object incremental slice measured a maximum of 0.181 ms over
  the 20,000-node Debug stress graph on the validation machine, below the 10 ms
  target. Debug CTest runtime for the complete collector test is approximately
  0.12 s; ThreadSanitizer runtime is approximately 0.39 s.
