# Collector pause target and evidence

## Target

- A task safepoint performs one young-generation collection or one bounded
  incremental mark slice.
- The default incremental slice performs at most 128 collector work units,
  including node and edge visits.
- The acceptance target is a 10 ms p99 safepoint on the supported release
  build for the 20,000-node collector stress graph.
- Full stop-the-world collection is reserved for explicit collection, severe
  memory pressure, invariant fallback, or shutdown and is not subject to the
  routine-safepoint target.

## Current Rust qualification

The final Rust candidate passes Release deep and wide 20,000-node checks with
at most 128 work units per routine slice and a 10 ms p99 assertion. Hash probes,
metadata edges, workspace release and wide immutable retirement have one-unit
coverage. Mutation/root changes, interrupted retirement, task transfer and
explicit foreign import pass focused and workspace checks. The fixed-budget
native compiler bootstrap also passes twice with byte-identical output.

Current evidence: plans/requirements-qualification.md, checkpoint 82 in
plans/requirements-implementation.md, and
/tmp/lana-final-small-call-release-slices.log. These are local qualification
results for the content manifest, not the historical HEAD or a publication.

Collector scratch is charged to the checked heap. Ordinary container pressure
starts above three quarters of the memory limit; class construction retains
its half-limit pressure check. Compiler limits remain 256 MiB and 50,000,000
instructions. The fixed-budget bootstrap regression is the acceptance check
for this scheduling policy.

## Historical C collector evidence

The measurements below describe the removed C collector and do not qualify the
current Rust collector.


- `lana_gc_tests` covers a 20,000-node deep graph, cycles, young reclamation,
  survivor promotion, old-to-young barriers, shared promotion, and one-object
  incremental slices.
- The bounded 128-object incremental slice measured a maximum of 0.181 ms over
  the 20,000-node Debug stress graph on the validation machine, below the 10 ms
  target. Debug CTest runtime for the complete collector test is approximately
  0.12 s; ThreadSanitizer runtime is approximately 0.39 s.
