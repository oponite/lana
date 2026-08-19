belief = State(0.50, 0.20)
evidence = State(0.80, 0.40, timestamp=0, confidence=0.80)
now = 31
if now - evidence.timestamp > 30 and evidence.confidence <= 0.90:
    evidence.decay(0.50)
evidence.apply_to(belief)
print(belief.p)
