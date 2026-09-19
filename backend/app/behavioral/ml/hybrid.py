"""Combining the statistical and ML identity evidence.

Both layers answer the same question — does this behaviour resemble the
enrolled user — by different means, and both report on the same scale where 0
is indistinguishable from the baseline and 1 is as different as the layer can
express. That shared convention is what makes a weighted blend meaningful
rather than an arbitrary mixing of units.

They are kept as separate reported values as well as a blend, because they
fail differently. The statistical layer compares against per-feature medians
and is strong when a single trait is clearly off. The forest looks at the
joint shape and can catch a vector whose individual features are each
unremarkable but whose combination never occurs for this user.

Neither is merged into automation or integrity. "Not you", "not a person" and
"cannot trust this evidence" stay three separate findings throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.behavioral import stats
from app.behavioral.ml.anomaly_model import AnomalyScore
from app.config import settings

# Below this share of the model's feature schema, the vector is mostly imputed
# and the forest is scoring our imputation rather than the user. The ML term is
# dropped for that attempt and the statistical score stands alone.
MIN_ML_COVERAGE = 0.60


@dataclass(frozen=True)
class HybridIdentity:
    """The identity verdict, with its components kept visible."""

    score: float                      # what the risk engine acts on
    statistical_score: float
    ml_score: float | None            # None when the ML layer did not apply
    ml_applied: bool
    ml_windows: int = 0
    statistical_weight: float = 1.0
    ml_weight: float = 0.0
    reason: str = ""                  # why ML was or was not used

    def as_dict(self) -> dict[str, object]:
        return {
            "identity_score": round(self.score, 4),
            "statistical_identity_score": round(self.statistical_score, 4),
            "ml_anomaly_score": None if self.ml_score is None else round(self.ml_score, 4),
            "ml_applied": self.ml_applied,
            "ml_windows": self.ml_windows,
            "weights": {
                "statistical": self.statistical_weight,
                "ml": self.ml_weight,
            },
        }


def combine(
    statistical_score: float,
    anomaly: AnomalyScore | None,
    statistical_weight: float | None = None,
    ml_weight: float | None = None,
) -> HybridIdentity:
    """Blend the two identity signals, or fall back to statistics alone.

    Weights come from configuration so alternatives can be benchmarked without
    editing code. They are renormalised rather than assumed to sum to one, so a
    misconfigured pair cannot silently scale the whole score up or down and
    move the effective threshold with it.
    """
    w_stat = settings.statistical_weight if statistical_weight is None else statistical_weight
    w_ml = settings.ml_weight if ml_weight is None else ml_weight

    statistical_score = stats.clamp(statistical_score)

    if anomaly is None or not anomaly.is_usable:
        return HybridIdentity(
            score=statistical_score,
            statistical_score=statistical_score,
            ml_score=None,
            ml_applied=False,
            statistical_weight=1.0,
            ml_weight=0.0,
            reason="no model available for this account",
        )

    if anomaly.mean_coverage < MIN_ML_COVERAGE:
        return HybridIdentity(
            score=statistical_score,
            statistical_score=statistical_score,
            ml_score=round(anomaly.score, 4),
            ml_applied=False,
            ml_windows=anomaly.windows_scored,
            statistical_weight=1.0,
            ml_weight=0.0,
            reason=(
                f"capture supplied {anomaly.mean_coverage:.0%} of the model's "
                f"features, below the {MIN_ML_COVERAGE:.0%} needed to trust it"
            ),
        )

    total = w_stat + w_ml
    if total <= 0:
        return HybridIdentity(
            score=statistical_score,
            statistical_score=statistical_score,
            ml_score=round(anomaly.score, 4),
            ml_applied=False,
            ml_windows=anomaly.windows_scored,
            statistical_weight=1.0,
            ml_weight=0.0,
            reason="weights are not configured; using statistics alone",
        )

    w_stat, w_ml = w_stat / total, w_ml / total
    blended = w_stat * statistical_score + w_ml * anomaly.score

    return HybridIdentity(
        score=stats.clamp(blended),
        statistical_score=statistical_score,
        ml_score=round(anomaly.score, 4),
        ml_applied=True,
        ml_windows=anomaly.windows_scored,
        statistical_weight=round(w_stat, 4),
        ml_weight=round(w_ml, 4),
        reason="statistical and ML evidence combined",
    )
