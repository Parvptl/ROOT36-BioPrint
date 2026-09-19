"""The population prior.

Answers one question per feature: how much do *people in general* differ on
this? That is what makes a deviation meaningful. Typing 15 ms slower than usual
is unremarkable if everyone's dwell times span 60 ms; it is significant if they
span 8 ms.

Availability is the practical problem. At a hackathon you get whatever
volunteers you can actually recruit, and the pipeline must not stall waiting
for a dataset that never arrives. So the prior degrades smoothly rather than
switching between a working and a broken mode:

    n = 0   pure relative prior, weights fall back to uniform
    n small empirical estimate shrunk heavily toward the relative prior
    n large essentially the empirical estimate

No population data is ever fabricated. With none collected, the system says so
through `sample_count` and `is_informative`, and the profile records it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from app.behavioral import stats
from app.behavioral.features.registry import enabled_names

# Below this the empirical spread is too noisy to drive feature weighting, so
# weights go uniform and we say so rather than implying a confidence we lack.
MIN_INFORMATIVE_SAMPLES = 5

# Shrinkage strength. The empirical estimate carries weight n/(n+k); at n=k it
# is believed half as much as the relative prior.
SHRINKAGE_K = 5.0

# With no population data, assume people differ by roughly this fraction of a
# feature's own magnitude. A deliberately wide, uninformative guess: it exists
# to stop a scale collapsing to zero, not to encode real knowledge.
RELATIVE_SPREAD = 0.35

# Absolute floor so a feature whose median is near zero still has a usable
# scale. Log-space and ratio features legitimately sit near zero.
ABSOLUTE_FLOOR = 1e-3


@dataclass(frozen=True)
class PopulationPrior:
    """Per-feature spread across people, with the evidence behind it."""

    scales: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0

    @property
    def is_informative(self) -> bool:
        return self.sample_count >= MIN_INFORMATIVE_SAMPLES

    def scale_for(self, name: str, own_median: float) -> float:
        """Population spread for one feature, shrunk toward the relative prior.

        `own_median` is the enrolling user's own centre for this feature, used
        to build the relative fallback. Passing it keeps the fallback on the
        same units as the feature instead of assuming a magnitude.
        """
        relative = max(RELATIVE_SPREAD * abs(own_median), ABSOLUTE_FLOOR)
        empirical = self.scales.get(name)
        if empirical is None or empirical <= 0.0:
            return relative

        n = float(self.sample_count)
        trust = n / (n + SHRINKAGE_K)
        return max(trust * empirical + (1.0 - trust) * relative, ABSOLUTE_FLOOR)

    def describe(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "informative": self.is_informative,
            "features_covered": len(self.scales),
        }


def fit_population_prior(samples: list[dict[str, float]]) -> PopulationPrior:
    """Estimate per-feature spread from consented volunteer samples.

    Uses MAD rather than standard deviation for the same reason everything else
    here does: one unusual volunteer should not define what 'typical spread'
    means.
    """
    scales: dict[str, float] = {}

    for name in enabled_names():
        values = np.array(
            [s[name] for s in samples if name in s and np.isfinite(s[name])],
            dtype=float,
        )
        # Two points give a MAD of zero by construction, which would be a lie.
        if values.size < 3:
            continue
        scale = stats.robust_scale(values)
        if scale > 0.0:
            scales[name] = scale

    return PopulationPrior(scales=scales, sample_count=len(samples))


# ------------------------------------------------------------------ storage


def record_population_sample(
    conn: sqlite3.Connection, features: dict[str, float], source: str
) -> None:
    """Store one consented sample.

    Derived features only, and no link to a user id. The table is a statistical
    reference, not a second copy of anyone's behavioural profile.
    """
    import time

    conn.execute(
        "INSERT INTO population_samples (source, features_json, created_at) VALUES (?, ?, ?)",
        (source, json.dumps(features, separators=(",", ":")), time.time()),
    )


def load_population_prior(
    conn: sqlite3.Connection, exclude_user_features: list[dict[str, float]] | None = None
) -> PopulationPrior:
    """Build the prior from stored samples.

    `exclude_user_features` drops the enrolling user's own contributions. A
    user must not help define the population they are being compared against:
    that would shrink their apparent distance from 'typical' and weaken exactly
    the discrimination the prior is there to provide.
    """
    rows = conn.execute("SELECT features_json FROM population_samples").fetchall()
    samples = [json.loads(row["features_json"]) for row in rows]

    if exclude_user_features:
        own = {_fingerprint_key(f) for f in exclude_user_features}
        samples = [s for s in samples if _fingerprint_key(s) not in own]

    return fit_population_prior(samples)


def _fingerprint_key(features: dict[str, float]) -> str:
    """Cheap identity for a feature dict, for exclusion by value."""
    return json.dumps(
        {k: round(v, 6) for k, v in sorted(features.items())}, separators=(",", ":")
    )
