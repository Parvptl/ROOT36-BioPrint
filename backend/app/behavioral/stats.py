"""Robust statistics helpers.

Everything here is median/MAD based rather than mean/standard-deviation based.
That is a deliberate choice for this problem: a single yawn, a phone buzzing,
or one hunt-for-the-right-key pause produces an outlier two orders of magnitude
larger than a normal inter-key interval. A mean and a standard deviation both
chase that outlier; a median and a MAD ignore it.
"""

from __future__ import annotations

import numpy as np

# Scales MAD to be a consistent estimator of the standard deviation under a
# normal distribution, so MAD-derived scales are comparable to sigmas.
MAD_TO_SIGMA = 1.4826


def median(values: np.ndarray) -> float:
    return float(np.median(values))


def mad(values: np.ndarray) -> float:
    """Median absolute deviation, unscaled."""
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def robust_scale(values: np.ndarray) -> float:
    """MAD scaled to sigma units."""
    return mad(values) * MAD_TO_SIGMA


def iqr(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def robust_cv(values: np.ndarray) -> float:
    """Coefficient of variation, robust form: MAD-sigma over median.

    Used as a dimensionless measure of how irregular a timing series is. Human
    typing sits in a fairly wide band; machine-generated input collapses toward
    zero, which is what the automation detector keys on.
    """
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    if abs(med) < 1e-9:
        return 0.0
    return robust_scale(values) / abs(med)


def safe_log(values: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Natural log with a floor.

    Inter-key intervals are close to log-normal, so timing features are far
    more stable in log space. The floor keeps a zero-millisecond interval
    (which a script will happily produce) from becoming negative infinity and
    poisoning every downstream statistic.
    """
    return np.log(np.maximum(values, floor))


def fraction(mask: np.ndarray) -> float:
    """Fraction of True values, 0.0 for an empty array."""
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask) / mask.size)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
