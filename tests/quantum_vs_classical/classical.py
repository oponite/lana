"""Classical probability model: a single scalar `p`, no disposition.

This is the baseline Lana's density-operator substrate is compared against.
It runs the same five experiments but can only see `p`, so it cannot
distinguish states that differ only in phase (disposition `d`).

The output schema mirrors `quantum.lana` but omits the `quantum` sub-object:
classical probability has no field to hold the phase answer.
"""

import json


def append(pa, pb):
    return 1.0 - (1.0 - pa) * (1.0 - pb)


def invert(p):
    return 1.0 - p


def distance(pa, pb):
    return abs(pa - pb)


out = {
    "distinguish": {"classical": {"plus": 0.5, "minus": 0.5}},
    "invert": {"classical": {"before": 0.3, "after": invert(0.3)}},
    "distance": {"classical": distance(0.5, 0.5)},
    "neutralize": {"classical": {"before": 0.5, "after": 0.5}},
    "agreement": {
        "classical": {"agree": append(0.7, 0.7), "oppose": append(0.7, 0.7)}
    },
}

print(json.dumps(out))
