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
