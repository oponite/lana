belief = {"p": 0.50, "d": 0.20}
evidence = {"p": 0.80, "d": 0.40, "timestamp": 0, "confidence": 0.80}
now = 31
if now - evidence["timestamp"] > 30 and evidence["confidence"] <= 0.90:
    evidence["p"] += 0.50 * (0.50 - evidence["p"])
    evidence["d"] *= 0.50
influence_target = 1 - evidence["p"] if evidence["d"] < 0 else evidence["p"]
belief["p"] += abs(evidence["d"]) * (influence_target - belief["p"])
print(belief["p"])
