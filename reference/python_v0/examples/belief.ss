# A complete Lana program: evidence changes a belief without observing it.
state belief = state(p1: 0.50, d: 0.20, timestamp: 0) history: latest(4)
state evidence = state(p1: 0.90, d: 0.65, source: "sensor", confidence: 0.90)

apply evidence -> belief
transform belief with decay(0.05)

let result = measure belief using distribution
result.p1
