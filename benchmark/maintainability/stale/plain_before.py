belief = {"p": 0.50, "d": 0.20}
evidence = {"p": 0.80, "d": 0.40, "timestamp": 0, "confidence": 0.80}
influence_target = 1 - evidence["p"] if evidence["d"] < 0 else evidence["p"]
belief["p"] += abs(evidence["d"]) * (influence_target - belief["p"])
print(belief["p"])
