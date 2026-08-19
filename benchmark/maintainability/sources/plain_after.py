belief = {"p": 0.50, "d": 0.20}
evidence = [
    {"p": 0.80, "d": 0.40, "weight": 1.0, "confidence": 0.80},
    {"p": 0.65, "d": 0.30, "weight": 0.7, "confidence": 0.70},
    {"p": 0.30, "d": 0.35, "weight": 0.9, "confidence": 0.75},
]
weights = [item["weight"] * item["confidence"] for item in evidence]
total = sum(weights)
aggregate_p = sum(item["p"] * weight for item, weight in zip(evidence, weights)) / total
aggregate_d = sum(item["d"] * weight for item, weight in zip(evidence, weights)) / total
influence_target = 1 - aggregate_p if aggregate_d < 0 else aggregate_p
belief["p"] += abs(aggregate_d) * (influence_target - belief["p"])
print(belief["p"])
