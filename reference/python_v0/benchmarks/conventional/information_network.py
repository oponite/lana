"""Ordinary Python baseline for the corresponding Lana benchmark."""


def clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def apply(source_probability: float, source_dependency: float, target_probability: float) -> float:
    return clip(target_probability + source_dependency * (source_probability - target_probability))


def run() -> float:
    a_probability, a_dependency = 0.15, 0.50
    b_probability, b_dependency = 0.50, 0.30
    c_probability, c_dependency = 0.85, -0.20

    b_probability = apply(a_probability, a_dependency, b_probability)
    c_probability = apply(b_probability, b_dependency, c_probability)
    a_probability = apply(c_probability, c_dependency, a_probability)
    return b_probability
