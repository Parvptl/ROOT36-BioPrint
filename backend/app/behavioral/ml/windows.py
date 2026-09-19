"""Slicing one capture into several behavioural windows.

A single login produces one feature vector. That is enough for the statistical
layer, which compares it against per-feature medians, but it is not a training
set: fitting an anomaly detector on eight points in twenty-seven dimensions
fits the noise, not the person.

Windowing turns each enrollment round into several observations of the same
behaviour. Eight rounds at roughly four windows each gives about thirty-two
training vectors, which is a different regime entirely.

Design rule: this module does NOT extract features. It slices the event stream
and hands each slice to the existing extractor, so there is exactly one feature
implementation in the project and no chance of enrollment and login computing
different things.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.behavioral.events import build_session_view
from app.behavioral.features.extractor import ExtractedFeatures, extract_features
from app.models.events import BehaviorSessionIn

# Windows are cut over keystrokes rather than wall-clock time. The features are
# overwhelmingly keystroke-driven, and a fixed time window would contain fifty
# presses during fast typing and four during a pause, so the feature vectors
# would not be comparable to each other.
WINDOW_PRESSES = 20
WINDOW_STRIDE = 10

# Below this a window cannot support the feature minimums in the registry and
# would contribute mostly absent values.
MIN_WINDOW_PRESSES = 14

# A window must yield at least this many features to be worth keeping.
MIN_WINDOW_FEATURES = 6

# Features whose meaning is session-level and becomes misleading once windowed.
#
# int_time_to_first_interaction measures the gap before the first event. Inside
# a window that is the offset of the window itself, so it encodes *where in the
# session the window sits* rather than anything about the person. Feeding it to
# the model would teach it the shape of our windowing, not the user.
#
# int_presubmit_hesitation only exists in the window containing the submit, so
# it is present in one window per session and absent in the rest.
WINDOW_EXCLUDED_FEATURES = frozenset(
    {
        "int_time_to_first_interaction",
        "int_presubmit_hesitation",
    }
)


@dataclass(frozen=True)
class BehaviorWindow:
    """One slice of a capture, already reduced to features."""

    index: int
    start_ms: float
    end_ms: float
    press_count: int
    features: dict[str, float]


def _phrase_presses(session: BehaviorSessionIn) -> list:
    """Keydown events in the challenge-phrase field, in order.

    Phrase context only, for the same reason the identity features use it: the
    phrase is regenerated every attempt, so windows cut over it describe how
    the person types rather than what they typed. Cutting over the password
    field would tie the model to a fixed string.
    """
    presses = [
        event
        for event in session.events
        if event.type == "keydown"
        and getattr(event, "ctx", None) == "phrase"
        and not getattr(event, "repeat", False)
    ]
    presses.sort(key=lambda e: e.t)
    return presses


def _slice_events(session: BehaviorSessionIn, start_ms: float, end_ms: float) -> list:
    """Every event inside a time span, whatever its kind.

    Deliberately not limited to keystrokes: pointer movement and focus changes
    that happened during this stretch of typing belong to the same window and
    carry their own features.
    """
    return [event for event in session.events if start_ms <= event.t <= end_ms]


def extract_windows(session: BehaviorSessionIn) -> list[BehaviorWindow]:
    """Cut a capture into overlapping windows and extract features from each.

    Returns an empty list when the capture is too short to window, which is a
    normal outcome the callers handle by falling back to the statistical layer.
    """
    presses = _phrase_presses(session)
    if len(presses) < MIN_WINDOW_PRESSES:
        return []

    windows: list[BehaviorWindow] = []
    index = 0
    start = 0

    while start < len(presses):
        chunk = presses[start : start + WINDOW_PRESSES]
        if len(chunk) < MIN_WINDOW_PRESSES:
            break

        start_ms = chunk[0].t
        # Extend to the end of the last press so its key release, and therefore
        # its dwell time, falls inside the window rather than being truncated.
        end_ms = chunk[-1].t + 1000.0

        sliced = _slice_events(session, start_ms, end_ms)
        if sliced:
            features = _window_features(session, sliced)
            if len(features) >= MIN_WINDOW_FEATURES:
                windows.append(
                    BehaviorWindow(
                        index=index,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        press_count=len(chunk),
                        features=features,
                    )
                )
                index += 1

        if start + WINDOW_PRESSES >= len(presses):
            break
        start += WINDOW_STRIDE

    return windows


def _window_features(session: BehaviorSessionIn, events: list) -> dict[str, float]:
    """Run the existing extractor over a slice.

    The slice is revalidated into a session so `build_session_view` and
    `extract_features` behave exactly as they do on a full capture. No feature
    logic is duplicated here.
    """
    windowed = session.model_copy(update={"events": events})
    view = build_session_view(windowed)
    extracted: ExtractedFeatures = extract_features(view)

    return {
        name: value
        for name, value in extracted.values.items()
        if name not in WINDOW_EXCLUDED_FEATURES
    }


def window_feature_dicts(session: BehaviorSessionIn) -> list[dict[str, float]]:
    """Convenience wrapper: just the feature vectors."""
    return [window.features for window in extract_windows(session)]
