"""Training the per-user model at enrollment, and recalibrating for the blend.

Two things happen here, and the second matters as much as the first.

Training is the obvious part: fit an Isolation Forest on the windows collected
across enrollment rounds and store it.

Recalibration is the part that is easy to skip and wrong to skip. The profile
carries one threshold, and it was derived from statistical scores. Blending ML
evidence into the identity score changes the distribution that threshold is
applied to, so leaving it alone would silently move the operating point — the
system would get stricter or looser than measured, for no stated reason. So the
threshold is re-derived from leave-one-session-out scores of the *blended*
score, using the same derivation the statistical path uses.

The leave-one-out here holds out a whole enrollment round, never individual
windows. Windows from the same round overlap by design, so splitting between
them would leak the held-out data straight back into training and report a
threshold far tighter than reality.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from app.behavioral.fingerprint.calibration import (
    CalibrationResult,
    derive_threshold,
    leave_one_session_out_scores,
)
from app.behavioral.fingerprint.population import PopulationPrior
from app.behavioral.ml.anomaly_model import (
    MIN_TRAINING_WINDOWS,
    BehavioralAnomalyModel,
    train_model,
)
from app.behavioral.ml.hybrid import combine
from app.behavioral.ml.store import delete_model, save_model
from app.config import settings
from app.db import repository

log = logging.getLogger("bioprint.ml")


@dataclass(frozen=True)
class MLTrainingOutcome:
    """What happened when we tried to build a model for this user."""

    trained: bool
    reason: str
    n_windows: int = 0
    n_features: int = 0
    calibration: CalibrationResult | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "ml_model_trained": self.trained,
            "ml_reason": self.reason,
            "ml_training_windows": self.n_windows,
            "ml_feature_count": self.n_features,
        }


def train_and_store_model(
    conn: sqlite3.Connection,
    user_id: int,
    sessions: list[dict[str, float]],
    prior: PopulationPrior,
) -> MLTrainingOutcome:
    """Fit, persist and recalibrate. Never raises into the enrollment path.

    Every failure degrades to the statistical layer, which is a complete
    working system on its own. An enrollment must not fail because an optional
    model could not be fitted.
    """
    if not settings.ml_enabled:
        return MLTrainingOutcome(False, "ML layer disabled by configuration")

    stored = repository.load_enrollment_windows(conn, user_id)
    if len(stored) < MIN_TRAINING_WINDOWS:
        return MLTrainingOutcome(
            False,
            f"only {len(stored)} behavioural windows captured, "
            f"{MIN_TRAINING_WINDOWS} needed to fit a model",
            n_windows=len(stored),
        )

    try:
        model = train_model([features for _, features in stored])
    except Exception:  # noqa: BLE001 - enrollment must survive a model failure
        log.warning("model training failed for user %s", user_id, exc_info=True)
        return MLTrainingOutcome(False, "model training failed", n_windows=len(stored))

    if model is None:
        return MLTrainingOutcome(
            False,
            "captured windows did not support a usable model",
            n_windows=len(stored),
        )

    try:
        save_model(user_id, model)
    except Exception:  # noqa: BLE001
        log.warning("could not persist model for user %s", user_id, exc_info=True)
        delete_model(user_id)
        return MLTrainingOutcome(False, "model could not be stored", n_windows=len(stored))

    calibration = _recalibrate_for_blend(stored, sessions, prior)

    log.info(
        "ml model trained user=%s windows=%d features=%d threshold=%.4f source=%s",
        user_id,
        model.n_training_windows,
        model.processor.n_features,
        calibration.threshold if calibration else -1.0,
        calibration.source if calibration else "none",
    )

    return MLTrainingOutcome(
        trained=True,
        reason="model trained and stored",
        n_windows=model.n_training_windows,
        n_features=model.processor.n_features,
        calibration=calibration,
    )


def _recalibrate_for_blend(
    stored_windows: list[tuple[int, dict[str, float]]],
    sessions: list[dict[str, float]],
    prior: PopulationPrior,
) -> CalibrationResult | None:
    """Re-derive the threshold from blended leave-one-session-out scores.

    For each enrollment round: fit a model on the windows of every *other*
    round, score the held-out round's windows with it, blend that with the
    held-out round's statistical LOO score, and collect the result.

    Returns None when there are too few rounds for this to mean anything, in
    which case the caller keeps the statistical threshold.
    """
    # With no ML weight the blend is the statistical score exactly, so the
    # existing threshold already describes it. Recalibrating would cost eight
    # model fits to arrive at the same number and would label the threshold
    # "+ml" when the ML term contributes nothing.
    if settings.ml_weight <= 0.0:
        return None

    statistical = leave_one_session_out_scores(sessions, prior)
    if len(statistical) < 3:
        return None

    by_session: dict[int, list[dict[str, float]]] = {}
    for index, features in stored_windows:
        by_session.setdefault(index, []).append(features)

    indices = sorted(by_session)
    if len(indices) < 3:
        return None

    blended: list[float] = []
    for position, index in enumerate(indices):
        if position >= len(statistical):
            break

        training = [
            features
            for other in indices
            if other != index
            for features in by_session[other]
        ]
        if len(training) < MIN_TRAINING_WINDOWS:
            continue

        fold_model = train_model(training)
        if fold_model is None:
            continue

        anomaly = fold_model.anomaly_for(by_session[index])
        blended.append(combine(statistical[position], anomaly).score)

    if len(blended) < 3:
        return None

    result = derive_threshold(blended, [])
    # Make it explicit in the stored profile that this threshold describes the
    # blended score, not the statistical one, so nobody later compares it
    # against a statistical-only number and concludes the system drifted.
    return CalibrationResult(
        threshold=result.threshold,
        source=f"{result.source}+ml",
        genuine_scores=result.genuine_scores,
        impostor_scores=result.impostor_scores,
        note=(
            f"{result.note} Scores are the hybrid statistical+ML identity score, "
            f"blended at weights {settings.statistical_weight}/{settings.ml_weight}, "
            f"with each enrollment round held out of both the profile and the model."
        ),
        metrics={**result.metrics, "blended": True},
    )


def load_model_for(user_id: int) -> BehavioralAnomalyModel | None:
    """Fetch a user's model if the ML layer is on and one exists."""
    if not settings.ml_enabled:
        return None
    from app.behavioral.ml.store import load_model

    return load_model(user_id)
