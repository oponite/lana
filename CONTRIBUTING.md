# Contributing

Contributors wanted - I'm especially interested in developers who want to have a say in how Lana is actually built.

My strength is in the language's semantics, behavior, and overall design. A fair amount of the lower-level implementation architecture has been developed with heavy AI assistance. I'm comfortable with that where the implementation is correct, maintainable, and meets the performance targets documented in `PERFORMANCE.md`.

However, I don't consider the current implementation sacred. If you're the kind of engineer who has opinions about compiler architecture, VM design, runtime internals, data structures, performance, or systems-level tradeoffs, I'm inviting you to consider getting involved. Semantics define what Lana must do; there's still plenty of room to influence how Lana does it.

## Where to contribute

Implementation work is especially welcome in:

- compiler architecture
- VM design and instruction execution
- runtime internals
- memory management and data representation
- concurrency and task execution
- host-call boundaries
- data structures and algorithms
- performance and profiling
- C/Rust implementation parity
- developer tooling and diagnostics

You do not need to preserve an implementation simply because it already exists.

If you can make something simpler, faster, safer, easier to reason about, or easier to maintain, propose it.

## What is fixed

The language semantics are the contract.

An implementation may change substantially as long as externally observable Lana behavior remains conformant with the language specification and accepted LIPs (Lana Implementation Protocol).

Changes to any of the following require an accepted LIP:
- language semantics
- source-language behavior
- specified bytecode behavior
- specified VM behavior

Implementation-only changes generally do not require a LIP.

Bug fixes, refactors, performance improvements, documentation, tests, and tooling that do not change specified behavior can be submitted through a regular pull request.

## Working on the implementation

Lana currently has both C11 and Rust implementations.

### C11

Relevant code lives under:

```
vm/c/
runtime/c/
tools/c/
```

Build with:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

### Rust

The Rust implementation is a Cargo workspace at the repository root, primarily under:
```
vm/rust/
runtime/rust/
tools/rust/
```

Build the CLI with:
`cargo build -p lana-cli`

### Lana

The self-hosted compiler source lives under:
`compiler/`

## Validation

You’re free to change the machinery. The observable behavior still has to hold.

Run the standard test suite with:
```ctest --test-dir build --output-on-failure```

The differential conformance suite compares the C11 and Rust implementations and requires byte-identical stdout, stderr, and exit codes:
```
tests/conformance/differential/run_core.sh
tests/conformance/differential/run_hostcalls.sh
tests/conformance/differential/run_tasks.sh
tests/conformance/differential/run_fuzz_diff.sh
```

Any source, compiler, bytecode, runtime, or VM change should include focused regression coverage for the behavior it touches.

Performance-sensitive changes should also be evaluated against the expectations and measurements documented in PERFORMANCE.md.

## Implementation principles

A few guardrails matter more than preserving any particular architecture:
- preserve language semantics and observable behavior
- prefer simple designs that are easy to reason about
- measure performance-sensitive changes rather than assuming they are faster
- keep C11 and Rust behavior conformant where both implementations cover the same functionality
- add focused tests for changed behavior and fixed bugs
- prefer existing dependencies and mechanisms when they already solve the problem well
- introduce new dependencies when they provide a clear implementation or maintenance benefit

Architectural disagreement is welcome when it comes with reasoning, evidence, or a working alternative.

## Proposing larger implementation changes

For substantial compiler, VM, runtime, or architectural changes, open an issue or pull request describing:
- what you want to change
- what problem the current implementation creates
- the proposed design
- compatibility implications
- expected performance or maintainability impact
- how the change will be tested

A full LIP is only necessary when the proposal changes Lana's language, bytecode, or VM behavior.

## References

- [Governance](GOVERNANCE.md)
- [Versioning](VERSIONING.md)
- [LIP process](lip/README.md)
