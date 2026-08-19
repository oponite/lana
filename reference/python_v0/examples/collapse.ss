# Sampling is reproducible with: ss run examples/collapse.ss --seed 7
state signal = state(p1: 0.73, d: 0.50)

let observed = measure signal using collapse
observed
let after = measure signal using distribution
after.p1
