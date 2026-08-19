# Matched by benchmarks/conventional/information_network.py.
state a = state(p1: 0.15, d: 0.50)
state b = state(p1: 0.50, d: 0.30)
state c = state(p1: 0.85, d: -0.20)

apply a -> b
apply b -> c
apply c -> a
return measure b using expectation
