"""Ordinary Python baseline for the corresponding Lana benchmark."""


def clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def run() -> float:
    target_probability = 0.50
    target_dependency = 0.15
    sources = [
        (0.88, 0.70, 1.0, 0.95),
        (0.22, 0.45, 0.6, 0.70),
        (0.55, 0.10, 0.8, 0.90),
    ]

    effective_weights = [weight * confidence for _, _, weight, confidence in sources]
    total_weight = sum(effective_weights)
    aggregate_probability = sum(item[0] * weight for item, weight in zip(sources, effective_weights)) / total_weight
    aggregate_dependency = sum(item[1] * weight for item, weight in zip(sources, effective_weights)) / total_weight
    target_probability = clip(target_probability + aggregate_dependency * (aggregate_probability - target_probability))
    return target_probability
