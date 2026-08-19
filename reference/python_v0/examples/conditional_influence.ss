# A state changes another state only after an explicit observation passes a threshold.
state belief = state(p1: 0.42, d: 0.20)
state evidence = state(p1: 0.91, d: 0.60, confidence: 0.90)

apply evidence -> belief when measure(evidence).p1 > 0.80
let result = measure belief using distribution
result.p1
