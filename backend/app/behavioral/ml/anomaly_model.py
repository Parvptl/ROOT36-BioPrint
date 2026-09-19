"""Per-user anomaly detection with Isolation Forest.

Why an anomaly detector rather than a classifier: enrollment observes one
class. We see how the account owner behaves and nothing else. There is no
labelled set of impostors for this user and there never will be at enrollment
time, so the only question the data can answer is "how unusual is this relative
to what we have seen from this person".

What Isolation Forest does NOT do, and the report says so explicitly: it does
not identify an attacker. It scores how easily a point is isolated from the
training distribution. Unusual is not the same as malicious — a genuine user
on a new keyboard is also unusual — which is exactly why this score is one
input to a risk engine rather than a verdict on its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest

from app.behavioral import stats
from app.behavioral.ml.preprocessing import BehavioralFeatureProcessor

MODEL_TYPE = "IsolationForest"
MODEL_VERSION = "1.0"

# Fixed so a rebuild from identical enrollment data gives an identical model.
# Authentication decisions that move between runs for no reason are not
# debuggable and not defensible.
RANDOM_STATE = 20260919

# Below this there is not enough data to fit anything meaningful, and the
# caller falls back to the statistical layer rather than pretending.
MIN_TRAINING_WINDOWS = 12

# Trees. More is steadier but slower; at a few dozen samples the curve is flat
# well before this, and inference stays around a millisecond.
N_ESTIMATORS = 150

# Single-threaded on purpose. Thread pool setup costs more than the work for a
# matrix this small, and it measurably hurt inference latency.
N_JOBS = 1

# How far below the training centre counts as fully anomalous, in robust
# scale units of the training score distribution. Three matches the saturation
# point of the statistical layer, so the two scores speak the same dialect.
ANOMALY_SATURATION_SIGMAS = 3.0


@dataclass
class AnomalyScore:
    """One login's ML verdict."""

    score: float = 0.0            # 0 normal, 1 highly anomalous
    windows_scored: int = 0
    mean_coverage: float = 0.0
    available: bool = False       # False means fall back to statistics alone

    @property
    def is_usable(self) -> bool:
        return self.available and self.windows_scored > 0


@dataclass
class BehavioralAnomalyModel:
    """A fitted per-user model plus everything needed to reproduce its scoring."""

    processor: BehavioralFeatureProcessor
    forest: IsolationForest
    train_centre: float = 0.0     # median raw score over training windows
    train_scale: float = 1.0      # robust spread of those scores
    n_training_windows: int = 0
    trained_at: float = field(default_factory=time.time)
    model_type: str = MODEL_TYPE
    model_version: str = MODEL_VERSION

    # ------------------------------------------------------------- scoring

    def raw_scores(self, windows: list[dict[str, float]]) -> np.ndarray:
        """sklearn's score_samples: higher means more normal."""
        matrix = self.processor.transform(windows)
        if matrix.size == 0:
            return np.empty(0, dtype=float)
        return self.forest.score_samples(matrix)

    def anomaly_for(self, windows: list[dict[str, float]]) -> AnomalyScore:
        """Normalised anomaly in [0, 1] for one capture's windows.

        sklearn's convention is inverted relative to what we want, and the raw
        values sit in a narrow band around -0.45 that means nothing on its own.
        Both problems are handled the same way: express the login's score as a
        distance *below* the training centre, in robust scale units of the
        training distribution, then saturate.

            anomaly = clamp((train_centre - score) / (3 * train_scale), 0, 1)

        So a login scoring at the training median is 0, and one sitting three
        robust sigmas below the typical training window is 1. Scoring *above*
        the centre, meaning even more typical than training, clamps to 0 rather
        than going negative.
        """
        if not windows:
            return AnomalyScore(available=True, windows_scored=0)

        raw = self.raw_scores(windows)
        if raw.size == 0:
            return AnomalyScore(available=True, windows_scored=0)

        denominator = max(ANOMALY_SATURATION_SIGMAS * self.train_scale, 1e-9)
        per_window = np.clip((self.train_centre - raw) / denominator, 0.0, 1.0)

        # Median across windows, not mean: one window covering a pause or a
        # correction should not drag the whole login.
        aggregate = float(np.median(per_window))
        coverage = float(
            np.mean([self.processor.coverage_of(w) for w in windows])
        )

        return AnomalyScore(
            score=stats.clamp(aggregate),
            windows_scored=int(raw.size),
            mean_coverage=coverage,
            available=True,
        )

    # ------------------------------------------------------------- metadata

    def describe(self) -> dict[str, object]:
        """Model card. Deliberately carries nothing user-identifying."""
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "preprocessing_version": self.processor.version,
            "feature_schema_version": self.processor.version,
            "feature_names": list(self.processor.feature_names),
            "feature_count": self.processor.n_features,
            "n_training_windows": self.n_training_windows,
            "trained_at": self.trained_at,
            "configuration": {
                "n_estimators": N_ESTIMATORS,
                "contamination": "auto",
                "random_state": RANDOM_STATE,
                "n_jobs": N_JOBS,
                "anomaly_saturation_sigmas": ANOMALY_SATURATION_SIGMAS,
            },
            "train_centre": self.train_centre,
            "train_scale": self.train_scale,
        }


def train_model(windows: list[dict[str, float]]) -> BehavioralAnomalyModel | None:
    """Fit a per-user model, or return None when the data cannot support one.

    Returning None is a normal outcome, not an error. The caller falls back to
    the statistical layer and records that no model was trained. Fabricating a
    model from six windows would produce confident scores backed by nothing.
    """
    if len(windows) < MIN_TRAINING_WINDOWS:
        return None

    processor = BehavioralFeatureProcessor()
    try:
        processor.fit(windows)
    except ValueError:
        return None

    matrix = processor.transform(windows)
    if matrix.shape[0] < MIN_TRAINING_WINDOWS or matrix.shape[1] < 2:
        return None

    forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        # Every tree sees every sample. With a few dozen points, subsampling
        # throws away data the model cannot spare.
        max_samples=matrix.shape[0],
    )
    forest.fit(matrix)

    # Calibrate the score mapping against the training distribution itself, so
    # the anomaly scale is expressed in units of this user's own variability.
    training_scores = forest.score_samples(matrix)
    centre = stats.median(training_scores)
    scale = max(stats.robust_scale(training_scores), 1e-4)

    return BehavioralAnomalyModel(
        processor=processor,
        forest=forest,
        train_centre=float(centre),
        train_scale=float(scale),
        n_training_windows=matrix.shape[0],
    )
