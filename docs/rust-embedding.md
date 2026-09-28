# Retaining Rust VM results

`Vm::result()` now returns `RootedValue`. Keep that handle, or clone it, when a
result must outlive the VM. The handle is opaque: use `print()`, `type_name()`,
`value_type()`, scalar accessors, or `child(index)?` for inspection.
Use `inspect_state_dist(format)?` to render a distribution. This replaces
`as_state_dist()` so callers cannot clone an untracked distribution graph. A child is
a new tracked handle and survives release of its parent. Raw graph-bearing
`Value` and `Arc` clones are not independent embedding roots.

```rust
use lana_vm::{RootedValue, Vm};
use lana_bytecode::{Chunk, LanaError};

fn evaluate(chunk: &Chunk) -> Result<RootedValue, LanaError> {
    let mut vm = Vm::new(chunk);
    let status = vm.run();
    if status != LanaError::Ok { return Err(status); }
    vm.result()
}
```

The call can return `LanaError::Oom` if the VM cannot account for the root
registry entry. The handle keeps class storage alive after VM teardown.
Cloning the handle preserves object identity and shares its tracked root entry.
`Value` remains the representation of internal graph edges and synchronous
host-call inputs. Retain callback inputs with `Vm::retain_value()` before
keeping them after the callback. The REPL and retained task results use this
ownership boundary. The removed `Deref<Target = Value>` must not be replaced
with an untracked clone when migrating a caller.

`Vm::set_memory_limit(bytes)` returns `Result<(), LanaError>`. Handle the
result; a limit below live usage returns `Oom` and keeps the previous limit.

Host-call implementations can use `vm.retain_value(&value)?` to create an
embedding root for another value from that VM. This check uses the task's
instruction and memory budgets. Foreign graph payloads return `LanaError::Task`;
incomplete construction and locked graphs return `UnsupportedOperation`.
Retention failure does not alter the value or its graph.

Use `vm.import_value(&value)?` to copy a snapshot from a host-owned heap before
returning it from a callback. This preserves aliases and cycles, copies mutable
captures, and snapshots live Information. Copying uses the receiving VM's
memory and instruction limits. The durable runtime imports decoded records at
this boundary; foreign callback graph results still fail retention.
This applies to immutable DAGs too: retention does not adopt another heap's
payload. Use `import_value` for foreign DTOs and then retain the local copy.

On VM teardown, managed graph nodes are traced and pinned before execution roots
are released. Class storage
and weak registrations pass to the retained result owner. Dropping a retained
result runs the same synchronous tracer against the remaining roots. Live
aliases are preserved; unreachable class and container cycles can be reclaimed.
Collector scratch allocations still obey the original heap limit.

Managed construction and buffer growth admit collector scratch capacity before
publication. Host roots also reserve their registry and traversal capacity.
Teardown reuses this admitted storage and obeys the original heap limit;
failed admission leaves the candidate unpublished.

Managed immutable payloads have weak collector registrations and reservations
for their headers and vector capacities. Full collection releases parent pins
before child pins; routine collection drains wide immutable payloads in steps.
Foreign graph nodes require explicit import into the receiving heap. Collection
uses try_lock and defers retirement while a graph is locked. Keep a retained
handle while borrowing a VM value; queued retirement retries after locks are
released. Shutdown uses full tracing and has no routine-pause guarantee.
Root release does not wait for the root registry lock. Dead leases remain
detectable through weak root records. Retired child heaps collect iteratively,
so a nested task result does not recursively invoke one collector per child.

Task join charges source-heap retirement to the calling VM instruction budget.
If cleanup fails, the complete copied result stays cached and the source heap
stays queued. A retry completes cleanup before returning that result. Imports
of locked arrays, maps, sets or live snapshots return `UnsupportedOperation`.
