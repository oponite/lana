"""Ordinary Python baseline for the corresponding Lana benchmark."""


def clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def run() -> float:
    belief_probability = 0.50
    belief_dependency = 0.20
    evidence_probability = 0.90
    evidence_dependency = 0.65

    belief_probability = clip(
        belief_probability + evidence_dependency * (evidence_probability - belief_probability)
    )
    belief_probability = 0.5 + (belief_probability - 0.5) * (1.0 - 0.05)
    return belief_probability
