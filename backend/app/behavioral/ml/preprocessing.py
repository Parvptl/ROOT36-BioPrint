"""Turning window feature dicts into a fixed, versioned matrix.

The single most dangerous failure mode for a deployed model is a silent
train/inference mismatch: the model was fitted on columns in one order with one
set of imputation values, and is later asked to score a vector assembled
differently. It does not error, it just returns nonsense.

This module exists so that cannot happen. The schema, the column order and the
imputation values are decided once at training, stored with the model, and
replayed exactly at inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.behavioral import stats

# Bumped whenever the meaning of a stored schema changes. A model whose
# metadata carries a different version is refused rather than reinterpreted.
PREPROCESSING_VERSION = "1.0"

# A feature must appear in at least this share of training windows to enter the
# schema. Below it there is not enough signal to impute the rest sensibly, and
# a column that is mostly imputed is a column of constants.
MIN_FEATURE_PRESENCE = 0.70

# Guard against a zero scale on a feature that happened to be constant across
# training windows. Without it the standardised value is a division by zero.
MIN_SCALE = 1e-6


class SchemaMismatch(ValueError):
    """Raised when a stored schema cannot be applied to current features."""


@dataclass
class BehavioralFeatureProcessor:
    """Fixed-schema, robustly standardised feature matrices.

    Standardisation is median and MAD rather than mean and standard deviation,
    matching the statistical layer. Isolation Forest splits on raw values, so
    features spanning wildly different magnitudes (milliseconds against ratios
    in [0,1]) would otherwise be split on very unevenly.
    """

    feature_names: list[str] = field(default_factory=list)
    centres: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    version: str = PREPROCESSING_VERSION

    @property
    def is_fitted(self) -> bool:
        return bool(self.feature_names)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    # ---------------------------------------------------------------- fitting

    def fit(self, windows: list[dict[str, float]]) -> "BehavioralFeatureProcessor":
        if not windows:
            raise ValueError("cannot fit a processor on zero windows")

        presence: dict[str, int] = {}
        for window in windows:
            for name, value in window.items():
                if np.isfinite(value):
                    presence[name] = presence.get(name, 0) + 1

        threshold = MIN_FEATURE_PRESENCE * len(windows)
        names = sorted(name for name, count in presence.items() if count >= threshold)
        if not names:
            raise ValueError("no feature was present in enough windows to model")

        centres: list[float] = []
        scales: list[float] = []
        for name in names:
            observed = np.array(
                [w[name] for w in windows if name in w and np.isfinite(w[name])],
                dtype=float,
            )
            centre = stats.median(observed)
            scale = max(stats.robust_scale(observed), abs(centre) * 0.05, MIN_SCALE)
            centres.append(float(centre))
            scales.append(float(scale))

        self.feature_names = names
        self.centres = centres
        self.scales = scales
        self.version = PREPROCESSING_VERSION
        return self

    # -------------------------------------------------------------- transform

    def transform_one(self, features: dict[str, float]) -> np.ndarray:
        """One feature dict to one standardised row, in schema order.

        A feature the caller did not supply is imputed with the training
        centre, which places it at the user's own normal rather than at zero.
        Zero would be an extreme value for most of these features and would
        manufacture an anomaly out of a missing measurement.
        """
        if not self.is_fitted:
            raise SchemaMismatch("processor has not been fitted")

        row = np.empty(len(self.feature_names), dtype=float)
        for i, name in enumerate(self.feature_names):
            value = features.get(name)
            if value is None or not np.isfinite(value):
                value = self.centres[i]
            row[i] = (value - self.centres[i]) / self.scales[i]
        return row

    def transform(self, windows: list[dict[str, float]]) -> np.ndarray:
        if not windows:
            return np.empty((0, self.n_features), dtype=float)
        return np.vstack([self.transform_one(w) for w in windows])

    def coverage_of(self, features: dict[str, float]) -> float:
        """Share of the schema this vector actually supplies.

        Callers use it to decide whether a score is worth trusting: a vector
        that is three-quarters imputed is not really being scored.
        """
        if not self.is_fitted:
            return 0.0
        present = sum(
            1
            for name in self.feature_names
            if name in features and np.isfinite(features[name])
        )
        return present / len(self.feature_names)

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "centres": list(self.centres),
            "scales": list(self.scales),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BehavioralFeatureProcessor":
        version = str(payload.get("version", ""))
        if version != PREPROCESSING_VERSION:
            raise SchemaMismatch(
                f"stored preprocessing version {version!r} does not match "
                f"{PREPROCESSING_VERSION!r}"
            )

        names = list(payload.get("feature_names") or [])
        centres = list(payload.get("centres") or [])
        scales = list(payload.get("scales") or [])
        if not (len(names) == len(centres) == len(scales)) or not names:
            raise SchemaMismatch("stored schema is incomplete or inconsistent")

        return cls(
            feature_names=[str(n) for n in names],
            centres=[float(c) for c in centres],
            scales=[float(s) for s in scales],
            version=version,
        )
