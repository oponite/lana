belief = State(0.50, 0.20)
evidence = [
    State(0.80, 0.40, weight=1.0, confidence=0.80),
    State(0.65, 0.30, weight=0.7, confidence=0.70),
    State(0.30, 0.35, weight=0.9, confidence=0.75),
]
State.aggregate(evidence).apply_to(belief)
print(belief.p)
