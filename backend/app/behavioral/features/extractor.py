"""Feature extraction entry point.

Runs the three modality extractors and applies the registry's gate: a feature
is kept only if it is enabled and backed by at least its declared minimum
number of observations. Everything else is reported as missing.

The distinction between "missing" and "zero" is load-bearing. Downstream,
missing features are skipped in the distance computation and lower the coverage
score; zeros would be treated as real measurements and would reject genuine
users for not having used the mouse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.behavioral.events import SessionView, build_session_view
from app.behavioral.features import interaction, keyboard, pointer
from app.behavioral.features.registry import (
    Modality,
    enabled_names,
    is_enabled,
    specs_for,
    SPECS,
)
from app.models.events import BehaviorSessionIn


@dataclass(frozen=True)
class ExtractedFeatures:
    """Features that survived the coverage gate, plus why the rest did not."""

    values: dict[str, float] = field(default_factory=dict)
    observations: dict[str, int] = field(default_factory=dict)
    # Computed, but with too few observations to trust: name -> count seen.
    underpowered: dict[str, int] = field(default_factory=dict)
    # Every enabled feature that is not in `values`, for whatever reason.
    # Superset of `underpowered`: also covers features that could not be
    # computed at all, such as pointer features when no pointer was used.
    # Kept distinct because "we measured it and it was thin" and "there was
    # nothing to measure" call for different responses.
    missing: tuple[str, ...] = ()
    coverage_by_modality: dict[str, float] = field(default_factory=dict)
    coverage: float = 0.0

    def present(self, name: str) -> bool:
        return name in self.values

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)


def _coverage(modality: Modality, values: dict[str, float]) -> float:
    specs = specs_for(modality)
    if not specs:
        return 0.0
    got = sum(1 for spec in specs if spec.name in values)
    return got / len(specs)


def extract_features(view: SessionView) -> ExtractedFeatures:
    """Extract every enabled feature from a prepared session view."""
    raw: dict[str, tuple[float, int]] = {}
    raw.update(keyboard.extract(view))
    raw.update(pointer.extract(view))
    raw.update(interaction.extract(view))

    values: dict[str, float] = {}
    observations: dict[str, int] = {}
    underpowered: dict[str, int] = {}

    for name, (value, count) in raw.items():
        if not is_enabled(name):
            continue
        # A NaN or infinity here means a degenerate input reached a division.
        # Dropping it is right: propagating it would silently corrupt every
        # aggregate that touches this feature.
        if not math.isfinite(value):
            underpowered[name] = count
            continue
        if count < SPECS[name].min_observations:
            underpowered[name] = count
            continue
        values[name] = float(value)
        observations[name] = count

    all_enabled = enabled_names()
    total_enabled = len(all_enabled)
    return ExtractedFeatures(
        values=values,
        observations=observations,
        underpowered=underpowered,
        missing=tuple(name for name in all_enabled if name not in values),
        coverage_by_modality={
            Modality.KEYBOARD.value: _coverage(Modality.KEYBOARD, values),
            Modality.POINTER.value: _coverage(Modality.POINTER, values),
            Modality.INTERACTION.value: _coverage(Modality.INTERACTION, values),
        },
        coverage=len(values) / total_enabled if total_enabled else 0.0,
    )


def extract_from_session(session: BehaviorSessionIn) -> tuple[SessionView, ExtractedFeatures]:
    """Convenience wrapper: reshape then extract.

    Returns the view as well, because the automation detector works on the same
    reshaped events and rebuilding it would double the work on the latency path.
    """
    view = build_session_view(session)
    return view, extract_features(view)
