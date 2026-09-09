from __future__ import annotations

"""Small dependency-free statistical helpers shared by training evaluations."""

import math
from collections.abc import Sequence


def mean_ci95(values: Sequence[float]) -> tuple[float, tuple[float, float], float]:
    """Return mean, normal 95% interval and half width for bounded match scores."""

    if not values:
        return math.nan, (math.nan, math.nan), math.inf
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, (-math.inf, math.inf), math.inf
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = 1.96 * math.sqrt(variance / len(values))
    return mean, (mean - half_width, mean + half_width), half_width


def adaptive_sample_complete(
    values: Sequence[float],
    *,
    minimum: int,
    maximum: int,
    target_half_width: float,
) -> bool:
    """Stop only after enough samples and either adequate precision or the cap."""

    if len(values) >= maximum:
        return True
    if len(values) < minimum:
        return False
    _, _, half_width = mean_ci95(values)
    return half_width <= target_half_width
