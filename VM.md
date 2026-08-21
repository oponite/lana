# C Virtual Machine

## Information runtime boundary

The VM owns all v5 joint names, domain descriptors, independent marginals,
finite weighted rows, projections, and sampled arrays.
Joint values are immutable after construction; projection and conditioning
allocate new VM-owned views and never mutate the source. Task cloning deep
copies the named joint graph, so no joint pointer crosses VM heaps. Sampling
uses the child VM's deterministic RNG stream. A finite correlated sample draws
one row and never samples columns independently. Resolution is exact and fails
with `SS_ERR_UNRESOLVED_VALUE` unless the joint support is a singleton. Duplicate
names, unknown projection names, impossible evidence, invalid descriptors, and
unsupported exact operations fail before a partial result is exposed. Clone
memoization preserves shared joint graphs while preventing cross-VM pointers.

The v5 VM represents finite possibilities with dependency identifiers. Pure
arithmetic, comparisons, and function bodies lift over those alternatives.
`PATH_SPLIT_V5` snapshots the active frames for an unresolved Boolean and
executes both branches; `PATH_JOIN_V5` restores and merges them into guarded
path values. Nested splits use a bounded execution stack. History-bearing
register merges, incompatible dependency joins, and unresolved loops are
explicitly unsupported. Printing, host calls, task creation, sampling, and
observation are rejected while a split branch is active, preventing partial or
duplicate effects. Successful observation increments the VM observation log
counter only after refinement succeeds.

The self-hosted compiler runs as ordinary verified bytecode with explicit
256 MiB memory and 50,000,000-instruction limits. `path_resolve` is the narrow C
host boundary for canonical module paths; lexing, parsing, resolution, semantic
IR lowering, SSBC emission, import-cycle checks, and call remapping execute in
Lana. Clean builds assemble the checked textual bootstrap artifact and require
no Python runtime.

The canonical runtime is a C11 register VM. `STATE` values store canonical
binary64 `p`, `d_re`, and `d_im` inline with metadata. `STATE_DIST` is a pointer
to an immutable VM-owned node:

- `DIRAC` captures a full state value, including metadata.
- `APPEND` retains left and right nodes and may cache direct-state parameters.
- `TRANSFORM` retains a child and a registered v3 transform identifier.

Moves share nodes inside one VM. Task transfer deep-copies the DAG with a memo
table, copies captured metadata strings, preserves shared subgraphs, and never
shares nodes across VMs. APPEND-generated concrete states have empty metadata;
transforms preserve their sampled input metadata.

Unqualified computational-basis expected probability is evaluated exactly and
recursively: Dirac returns `p`, APPEND returns
`1-(1-E[left])*(1-E[right])`, and TRANSFORM invokes its registered exact
expectation rule. Every distribution-liftable transform must register a concrete
state function, validity guarantee, and exact expectation function.

Basis-aware concrete-state measurement reconstructs
`c = sqrt(p*(1-p)) * (d_re + i*d_im)`. With ordered bases
`computational=(|0>,|1>)`, `x=(|+>,|->)`, and
`y=(|+y>,|-y>)` where `|+y>=(|0>+i|1>)/sqrt(2)`, the basis-0 probabilities are
`p`, `1/2 + Re(c)`, and `1/2 - Im(c)`, respectively. It is read-only.

Qualified basis-aware measurement of `STATE_DIST` supports exact sampling only:
the VM samples one concrete state, computes its exact basis probability, and
uses the existing PCG32 binary draw. Qualified probability/distribution modes
return `SS_ERR_UNSUPPORTED_EXACT_MEASUREMENT`.

`estimate_measure` is the explicit approximate path. For each of `N` trials the
VM consumes one instruction-budget unit, samples the existing distribution, and
computes the exact basis probability of that sampled state. It returns the
arithmetic mean `q_hat`, or `distribution(1-q_hat, q_hat)`, only after all trials
complete. Cancellation, memory errors, invalid distribution nodes, and budget
exhaustion return an error and never expose a partial estimate. The VM's existing
seeded RNG is used, so the same chunk, inputs, sample count, and seed reproduce
the same estimate. This is regular Monte Carlo and is not presented as exact
mathematical evaluation; no confidence interval is returned.

Sampling preserves the binary tree. An APPEND node samples each child, computes
the conditional APPEND parameters, then uses PCG32 and a fixed Marsaglia-polar
normal-pair proposal with no cached spare. Unit-disk rejection implements the
truncated circular complex normal. Before every proposal the VM checks
cancellation and consumes one instruction-budget unit. Cancellation and budget
exhaustion return immediately with no fallback sample.

VM-lifetime allocations are released by `ss_vm_free`. Allocation failure returns
`SS_ERR_OOM`. Malformed lazy nodes return `SS_ERR_INVALID_DISTRIBUTION`; missing
exact transform support returns `SS_ERR_UNSUPPORTED_EXACT_MEASUREMENT` during
expectation evaluation and prevents distribution lifting with
`SS_ERR_UNSUPPORTED_OPERATION`.

Forked functions have independent registers, heap, instruction and memory
budgets, RNG stream, and error state. Bytecode/constants remain immutable and may
be shared. Task groups, cooperative cancellation, joins, tracing, and host calls
retain their existing architecture.
