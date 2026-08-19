# Several persistent states influence one another without becoming booleans.
state a = state(p1: 0.15, d: 0.50)
state b = state(p1: 0.50, d: 0.30)
state c = state(p1: 0.85, d: -0.20)

apply a -> b
apply b -> c
apply c -> a

let result = measure b using distribution
result.p1
