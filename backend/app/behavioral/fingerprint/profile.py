"""Building a user's behavioural profile from enrollment sessions.

The model is a shrinkage-regularised robust deviation score. For each feature
it stores a centre, a scale, and a weight; a login is scored by how many
scale-units it sits from the centre, weighted and saturated.

Why not One-Class SVM, Isolation Forest, or an autoencoder
----------------------------------------------------------
Enrollment yields about five samples in roughly thirty dimensions. None of
those models is estimable there; they would fit the enrollment noise and
produce confident nonsense. The evaluation harness fits them anyway and reports
the comparison, so this is a measured choice rather than an assumed one.

The three pieces that make the simple model work are below: a shrinkage floor
on the scale, a discriminability weight, and a saturating contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.behavioral import stats
from app.behavioral.features.registry import SPECS, is_enabled
from app.behavioral.fingerprint.population import ABSOLUTE_FLOOR, PopulationPrior
from enum import StrEnum

class Maturity(StrEnum):
    COLD_START = "COLD_START"
    WARMING = "WARMING"
    ESTABLISHED = "ESTABLISHED"
    MATURE = "MATURE"

# A feature must appear in at least this fraction of enrollment sessions, and
# in at least this many, before it earns a place in the profile. Characterising
# a feature seen once out of five is not possible.
MIN_FEATURE_COVERAGE = 0.6
MIN_FEATURE_SESSIONS = 3

# How much of the population spread is allowed as a floor on a user's own
# scale. Without this a user who happened to be very consistent on one feature
# across five sessions gets a near-zero scale, every genuine login then scores
# an enormous deviation on that axis, and the account becomes unusable. This is
# the single most important constant in the model.
POPULATION_FLOOR_ALPHA = 0.3

# A second floor, relative to the feature's own magnitude, covering the case
# where the population prior is itself thin.
RELATIVE_FLOOR_BETA = 0.06

# Finite-sample widening of MAD. With five samples the median absolute
# deviation systematically understates spread. The exact form is a judgement
# call; it matters less than it looks because the decision threshold is
# calibrated using these same scales, so a consistent bias is absorbed.
def small_sample_inflation(n: int) -> float:
    if n <= 2:
        return 2.0
    return math.sqrt(n / (n - 1.5))


@dataclass(frozen=True)
class FeatureStat:
    name: str
    modality: str
    # The centre scoring uses. When adaptation is running this is the blend of
    # the two tracks below; at enrollment all three are the same value.
    median: float
    mad: float
    scale: float
    weight: float
    coverage: float
    # Slow track: stable identity. Fast track: natural drift. Defaulted to 0.0
    # so a profile written before adaptation existed still loads, and the
    # adapter falls back to `median` when it sees a zero.
    median_long: float = 0.0
    median_recent: float = 0.0
    # The value fitted at enrollment. Never changes. Adaptation is clamped to a
    # bounded neighbourhood of this, so no number of accepted sessions can walk
    # the profile arbitrarily far from the person who actually enrolled.
    median_enrolled: float = 0.0


@dataclass(frozen=True)
class BehaviorProfile:
    """An enrolled user's baseline."""

    features: dict[str, FeatureStat] = field(default_factory=dict)
    session_count: int = 0
    population_size: int = 0
    population_informative: bool = False
    threshold: float = 0.0
    threshold_source: str = "uncalibrated"
    calibration: dict[str, object] = field(default_factory=dict)
    maturity: Maturity = Maturity.MATURE
    # Bumped on every adaptive update; 1 means "as enrolled, never adapted".
    version: int = 1
    update_count: int = 0

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.features)

    def modality_of(self, name: str) -> str:
        return self.features[name].modality


def fit_profile(
    sessions: list[dict[str, float]],
    prior: PopulationPrior,
) -> dict[str, FeatureStat]:
    """Compute per-feature statistics across enrollment sessions.

    `sessions` is one feature dict per enrollment round. Features missing from
    a round are simply not counted for that round, which is what keeps a
    mouse-free session from poisoning the pointer statistics.
    """
    if not sessions:
        return {}

    total = len(sessions)
    stats_out: dict[str, FeatureStat] = {}

    observed: dict[str, list[float]] = {}
    for session in sessions:
        for name, value in session.items():
            if is_enabled(name) and math.isfinite(value):
                observed.setdefault(name, []).append(value)

    for name, raw in observed.items():
        coverage = len(raw) / total
        if coverage < MIN_FEATURE_COVERAGE or len(raw) < MIN_FEATURE_SESSIONS:
            # Seen too rarely to characterise. Excluded from the profile, which
            # means a login is never judged against a statistic we could not
            # actually estimate.
            continue

        values = np.asarray(raw, dtype=float)
        centre = stats.median(values)
        raw_mad = stats.mad(values)

        own_scale = raw_mad * stats.MAD_TO_SIGMA * small_sample_inflation(len(raw))
        population_scale = prior.scale_for(name, centre)

        scale = max(
            own_scale,
            POPULATION_FLOOR_ALPHA * population_scale,
            RELATIVE_FLOOR_BETA * abs(centre),
            # The measurement floor: no scale may be finer than the resolution
            # at which one session can estimate this feature. Without it, a
            # quantised feature such as backspace rate collapses its scale to
            # the gap between two adjacent counts, and a genuine user typing
            # one extra backspace scores nine sigma out. The other floors are
            # all relative to the centre, so for a feature whose centre is
            # near zero they shrink with it and provide no protection.
            SPECS[name].noise_floor,
            ABSOLUTE_FLOOR,
        )

        stats_out[name] = FeatureStat(
            name=name,
            modality=SPECS[name].modality.value,
            median=centre,
            # Both adaptation tracks start at the enrolled centre.
            median_long=centre,
            median_recent=centre,
            median_enrolled=centre,
            mad=raw_mad,
            scale=scale,
            weight=_discriminability_weight(scale, population_scale, prior),
            coverage=coverage,
        )

    return stats_out


def _discriminability_weight(
    scale: float, population_scale: float, prior: PopulationPrior
) -> float:
    """How much this feature should count, in [0, 1].

    A Fisher-ratio in spirit: a feature earns influence when the user is tight
    on it relative to how far people spread. Typing speed where this user
    varies by 5% but the population varies by 40% is worth listening to;
    a feature where the user is as variable as everyone else tells us nothing
    and is damped accordingly.

    With no usable population data the honest answer is that we do not know
    which features discriminate, so all of them count equally. Pretending
    otherwise would be inventing a ranking from nothing.
    """
    if not prior.is_informative:
        return 1.0

    variance = population_scale ** 2
    return float(variance / (variance + scale ** 2)) if variance > 0 else 1.0

def fit_early_profile(
    sessions: list[dict[str, float]],
    prior: PopulationPrior,
) -> dict[str, FeatureStat]:
    """Fit from two observations, where a dispersion cannot yet be estimated.

    `fit_profile` needs MIN_FEATURE_SESSIONS (3) before it will characterise a
    feature at all, and below that it returns an empty profile that blocks
    every attempt as INSUFFICIENT_SIGNAL. So two captures need their own path.

    Two observations give a usable *centre* — the midpoint of two draws is a
    considerably better estimate of a person than a single draw — but they give
    almost nothing about spread. A MAD computed from two points is half their
    absolute difference, which for two nearby draws collapses toward zero and
    would make ordinary variation score many sigma out. That is the failure
    mode the noise floors were added to prevent, and it is worse here because
    there is no third point to contradict it.

    So the centre is personal and the scale stays population-derived, exactly
    as in the single-sample path. The observed spread is recorded in `mad` for
    inspection, and deliberately does NOT tighten the scale. Adaptation widens
    personalisation from here as genuine logins arrive.
    """
    if not sessions:
        return {}
    if len(sessions) == 1:
        return fit_cold_start_profile(sessions[0], prior)

    observed: dict[str, list[float]] = {}
    for session in sessions:
        for name, value in session.items():
            if is_enabled(name) and math.isfinite(value):
                observed.setdefault(name, []).append(value)

    total = len(sessions)
    stats_out: dict[str, FeatureStat] = {}

    for name, raw in observed.items():
        # A feature seen in only one of the two captures is kept: with two
        # observations, requiring both would discard pointer features from any
        # user who moved the mouse once. Coverage records how thin it is, and
        # the scoring path already weights by coverage.
        values = np.asarray(raw, dtype=float)
        centre = stats.median(values)
        raw_mad = stats.mad(values)

        population_scale = prior.scale_for(name, centre)
        scale = max(
            population_scale,
            RELATIVE_FLOOR_BETA * abs(centre),
            SPECS[name].noise_floor,
            ABSOLUTE_FLOOR,
        )

        stats_out[name] = FeatureStat(
            name=name,
            modality=SPECS[name].modality.value,
            median=centre,
            median_long=centre,
            median_recent=centre,
            median_enrolled=centre,
            # Recorded, not used as the scale. See the docstring.
            mad=raw_mad,
            scale=scale,
            # Uniform, as in the cold-start path: with two observations we
            # cannot tell which features discriminate for this person, and
            # inventing a ranking from two points would be worse than none.
            weight=1.0,
            coverage=len(raw) / total,
        )

    return stats_out


def fit_cold_start_profile(
    session: dict[str, float],
    prior: PopulationPrior,
) -> dict[str, FeatureStat]:
    """Compute a conservative single-sample profile.
    
    With only one session, variance cannot be estimated empirically. The scale
    is entirely derived from the population prior.
    """
    stats_out: dict[str, FeatureStat] = {}
    
    for name, value in session.items():
        if is_enabled(name) and math.isfinite(value):
            # Only generate features if the prior can provide a scale for them.
            # E.g. pointer features might not have Aalto prior data, they fall back 
            # to RELATIVE_SPREAD.
            centre = value
            
            # The scale comes purely from the population prior.
            population_scale = prior.scale_for(name, centre)
            scale = max(population_scale, ABSOLUTE_FLOOR)
            
            stats_out[name] = FeatureStat(
                name=name,
                modality=SPECS[name].modality.value,
                median=centre,
                median_long=centre,
                median_recent=centre,
                median_enrolled=centre,
                mad=0.0, # Cannot compute MAD from 1 sample
                scale=scale,
                weight=1.0, # Uniform weight since we don't know discriminability
                coverage=1.0,
            )
            
    return stats_out
