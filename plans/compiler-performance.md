# Compiler Bootstrap Performance

## Result

The compiler-performance work is complete. The self-hosted compiler remains
byte-stable and the measured Release bootstrap executes in a
median 0.543 seconds on the 2026-08-23 development Mac.

## Reproducible profile

Run the same compiler bytecode and bundled compiler source under both VMs:

```bash
build/lanavm run build/lana-compiler.labc --stats \
  --instruction-limit 50000000 --memory-limit-mib 256 -- \
  build/compiler-bootstrap.lana /tmp/compiler-debug.lasm

build/lanavm_release run build/lana-compiler.labc --stats \
  --instruction-limit 50000000 --memory-limit-mib 256 -- \
  build/compiler-bootstrap.lana /tmp/compiler-release.lasm
```

Five alternating Debug/Release runs were captured after one warm build.

| Metric | Recorded baseline | Completed implementation |
| --- | ---: | ---: |
| Instructions | 22,657,819 | 19,549,384 |
| Allocations | 602,446 | 619,386 |
| Allocated bytes | 93,511,662 | 93,151,428 |
| Debug median | 1.049 s | 0.963 s |
| Release median | 0.519 s | 0.543 s |

The map-backed emitter removes 13.7% of executed instructions and reduces the
Debug median by 8.2%. The Release median remains within 5% of the recorded
single-run baseline. Generated output from Debug and Release has the same
SHA-256 as `compiler/bootstrap/compiler.lasm`.

## Implemented changes

- Module loading caches canonical paths for the duration of one compiler
  invocation, so repeated imports reuse one parsed and resolved module.
- Resolver environments and module/import mappings use Lana maps instead of
  repeated linear scans.
- Emitter symbol and Information-category tables use Lana maps. Emitted text is
  assembled with buffered arrays and a final `string_join`.
- The C assembler indexes labels and function names with deterministic
  open-addressed tables before applying fixups. A stress regression covers
  1,024 labels, 256 functions, and unknown label/function diagnostics.
- Bootstrap verification runs the optimized `lanavm_release` while normal
  development and semantic tests retain the Debug VM.

The within-invocation module cache deliberately uses canonical path identity,
not a persistent content cache. A source file is read once per invocation, so a
content hash would add work without changing invalidation behavior. Persistent
project caching belongs to Milestone 9, where the complete source and lockfile
graph forms the cache key.

## Native/AOT evaluation

Production AOT work is rejected for this milestone:

- The Release bootstrap is already about 55 times faster than the 30-second
  evaluation threshold.
- A C emitter, WASM runtime, or compiler-specific native fast path would add a
  second execution architecture or external dependency.
- None offers a credible material user benefit at a 0.543-second bootstrap
  while preserving the C11-only runtime and byte-stable self-hosting gate.

Re-evaluate AOT only if a future representative project exceeds the existing
compiler limit or median clean compilation rises above five seconds.

## Acceptance gate

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
ctest --test-dir build-asan --output-on-failure -E native_compiler_bootstrap
ctest --test-dir build-tsan --output-on-failure -E native_compiler_bootstrap
git diff HEAD --check
```

`native_compiler_bootstrap` is the byte-stability authority. Performance output
is measurement evidence, never a substitute for semantic, sanitizer, installed
CLI, or bootstrap validation.
