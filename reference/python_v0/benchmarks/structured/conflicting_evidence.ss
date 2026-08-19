# Matched by benchmarks/conventional/conflicting_evidence.py.
state target = state(p1: 0.50, d: 0.15)
state sensor = state(p1: 0.88, d: 0.70, weight: 1.0, confidence: 0.95)
state report = state(p1: 0.22, d: 0.45, weight: 0.6, confidence: 0.70)
state prior = state(p1: 0.55, d: 0.10, weight: 0.8, confidence: 0.90)

apply [sensor, report, prior] -> target using weighted
return measure target using expectation
