belief = State(0.50, 0.20)
evidence = State(0.80, 0.40, timestamp=0, confidence=0.80)
evidence.apply_to(belief)
print(belief.p)
