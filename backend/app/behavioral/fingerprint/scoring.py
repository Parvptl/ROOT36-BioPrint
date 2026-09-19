"""Scoring a login attempt against an enrolled profile.

Produces a deviation in [0, 1]: 0 means indistinguishable from the baseline,
1 means as different as this model can express.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.behavioral.fingerprint.profile import BehaviorProfile

# Saturation point, in scale units. Beyond three scale-units a feature is
# already as wrong as it can usefully be; letting it keep growing would mean a
# single odd axis could outvote every other feature combined. A genuine user
# with a sore wrist, a different chair, or a borrowed mouse produces exactly
# that kind of isolated outlier, and saturating is what stops it becoming a
# lockout.
SATURATION_Z = 3.0
_SATURATION_RHO = SATURATION_Z ** 2


@dataclass(frozen=True)
class FeatureContribution:
    name: str
    modality: str
    z: float
    share: float  # fraction of the total deviation this feature accounts for


@dataclass(frozen=True)
class IdentityResult:
    score: float
    modality_scores: dict[str, float] = field(default_factory=dict)
    coverage: float = 0.0
    compared: int = 0
    contributions: list[FeatureContribution] = field(default_factory=list)

    @property
    def is_comparable(self) -> bool:
        return self.compared > 0


def _rho(z: float) -> float:
    """Saturating contribution. Quadratic near zero, flat past the cut."""
    return min(z * z, _SATURATION_RHO)


def score_identity(
    profile: BehaviorProfile, features: dict[str, float]
) -> IdentityResult:
    """Weighted, saturated deviation of an attempt from a profile.

    Only features present in both the profile and this attempt are compared.
    A feature the attempt could not produce is skipped and reduces coverage
    rather than counting as a deviation: not observing the mouse is not
    evidence that the mouse moved wrongly.
    """
    if not profile.features:
        return IdentityResult(score=0.0)

    total_weight = sum(stat.weight for stat in profile.features.values())
    if total_weight <= 0:
        return IdentityResult(score=0.0)

    weighted_sum = 0.0
    compared_weight = 0.0
    per_modality: dict[str, list[tuple[float, float]]] = {}
    raw_contributions: list[tuple[str, str, float, float]] = []

    for name, stat in profile.features.items():
        value = features.get(name)
        if value is None:
            continue

        z = (value - stat.median) / stat.scale
        contribution = stat.weight * _rho(z)

        weighted_sum += contribution
        compared_weight += stat.weight
        per_modality.setdefault(stat.modality, []).append((stat.weight, _rho(z)))
        raw_contributions.append((name, stat.modality, z, contribution))

    if compared_weight <= 0:
        return IdentityResult(score=0.0, coverage=0.0, compared=0)

    deviation = weighted_sum / compared_weight
    score = min(deviation / _SATURATION_RHO, 1.0)

    modality_scores = {
        modality: min(
            sum(w * r for w, r in pairs) / sum(w for w, _ in pairs) / _SATURATION_RHO,
            1.0,
        )
        for modality, pairs in per_modality.items()
        if sum(w for w, _ in pairs) > 0
    }

    return IdentityResult(
        score=score,
        modality_scores=modality_scores,
        coverage=compared_weight / total_weight,
        compared=len(raw_contributions),
        contributions=_rank_contributions(raw_contributions, weighted_sum),
    )


def _rank_contributions(
    raw: list[tuple[str, str, float, float]], total: float
) -> list[FeatureContribution]:
    """Order features by how much of the deviation they account for.

    Used for explanations. Shares are relative to the deviation actually
    observed, so they answer "what drove this decision" rather than "which
    feature is largest in absolute terms".
    """
    if total <= 0:
        return []

    ranked = sorted(raw, key=lambda item: item[3], reverse=True)
    return [
        FeatureContribution(name=name, modality=modality, z=z, share=contribution / total)
        for name, modality, z, contribution in ranked
    ]
