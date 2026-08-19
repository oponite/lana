# Matched by benchmarks/conventional/evolving_belief.py.
state belief = state(p1: 0.50, d: 0.20)
state evidence = state(p1: 0.90, d: 0.65)

apply evidence -> belief
transform belief with decay(0.05)
return measure belief using expectation
