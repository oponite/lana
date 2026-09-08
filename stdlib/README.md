# Standard library

The Lana standard library — a library written in Lana, layered over the VM's
built-in host calls. Modules are imported through the reserved `std/` prefix:

```lana
import "std/string" as string;
import "std/collections" as collections;
```

The `std/` prefix resolves to the installed standard library (via
`LANA_STDLIB_DIR`, or a cwd-relative `stdlib/` fallback), not to a relative
path.

## Modules

| Module | Contents |
|---|---|
| `std/string` | `split`, `join`, `trim`, `starts_with`, `ends_with`, `contains`, `replace`, `substring` |
| `std/math` | `abs`, `min`, `max`, `clamp` |
| `std/collections` | `set_new`, `set_add`, `set_contains`, `set_union`, `set_intersect`, `set_difference`, `iter`, `enumerate`, `zip` |
| `std/iterators` | `iter`, `enumerate`, `zip` (re-exported from `std/collections`) |
| `std/random` | `random_seed`, `random_float`, `random_int`, `random_choice` |
| `std/datetime` | `now`, `add_duration` |
| `std/testing` | `assert_eq`, `assert_true`, `assert_false` |
| `std/decision` | `value_of_information` (pure finite-table advisory ranking) |
| `std/ml` | uncertainty-aware linear, logistic, neural, Kalman/HMM, boosted-tree, and jump models |
| `std/json` | `parse`, `stringify` |
| `std/http` | `get`, `post` (io effect; return `Result<Information<HttpResponse>, E>`) |
| `std/csv` | `read`, `write` |
| `std/unicode` | `substring` (code-point-indexed); `char_length`, `to_upper`, `to_lower` are host calls |
| `std/regex` | `compile`, `matches`, `search`, `replace` (Thompson NFA, linear-time) |

`map`, `filter`, and `reduce` are compiler builtins (Lana has no first-class
function values), so they are available without an import.

## Conformance

Every stdlib function is `pure` unless it declares an effect (`io`,
`stochastic`, `external_call`). A function that reads the clock (`now`) or
draws a random value (`random_*`) declares its effect explicitly.

Nothing in this directory is part of the language contract until it is
specified and accepted through the LIP process (`../lip/`).

## Value of information

`std/decision.value_of_information(current_information,
candidate_observations, actions, utility, costs)` ranks finite candidate
observations by expected utility improvement minus cost. Priors, observation
joint laws, utilities, and costs are ordinary arrays of maps; the returned plan
contains the baseline action, outcome-specific policy, gross expected value,
cost, net value, exactness, assumptions, and recommendations. Unknown or empty
joint laws remain visible but unranked. The result carries normal Lana
provenance and is advisory: it does not acquire information or execute an
action.

Opaque joint values and function-valued utilities are not accepted because the
current stdlib cannot enumerate joints or receive general function values.

## Machine learning

`std/ml.fit(kind, {x, y}, options)` and `fit_dataset` return a schema-1 Result
containing the fitted model, prediction, uncertainty, diagnostics, method, and
assumptions. `predict` returns the corresponding rich record;
`predict_tensors` returns `[prediction, uncertainty]`. An explicit unavailable
Metal request is an error and never silently falls back or downcasts f64.

The current reference implementation is CPU-only. Its neural model has one
hidden ReLU layer, and boosted trees use configurable-depth histogram CART.
Categorical and diagonal-Gaussian HMMs use Baum-Welch fitting and Viterbi
decoding. Kalman filtering and RTS smoothing support both the original scalar
options and explicit multivariate `f`, `h`, `q`, `r`, `x0`, and `p0` tensors.
These are the behavioral oracle for the resident-Metal LIP-028 implementation,
not evidence that the Metal acceptance gates have passed.
