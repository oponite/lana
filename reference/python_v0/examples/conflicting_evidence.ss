# Weighted, conflicting sources influence one target in one explicit operation.
state belief = state(p1: 0.50, d: 0.15)
state sensor = state(p1: 0.88, d: 0.70, source: "sensor", weight: 1.0, confidence: 0.95)
state report = state(p1: 0.22, d: 0.45, source: "report", weight: 0.6, confidence: 0.70)
state prior = state(p1: 0.55, d: 0.10, source: "prior", weight: 0.8, confidence: 0.90)

apply [sensor, report, prior] -> belief using weighted
let result = measure belief using distribution
result.p1
