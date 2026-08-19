belief = State(0.50, 0.20)
evidence = [State(0.80, 0.40, weight=1.0, confidence=0.80)]
State.aggregate(evidence).apply_to(belief)
print(belief.p)
