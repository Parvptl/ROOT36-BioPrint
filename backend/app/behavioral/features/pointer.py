"""Pointer dynamics.

Movement is analysed per segment — a continuous stretch of travel between
pauses — rather than over the whole capture. Averaging across a pause would mix
a fast flick to the submit button with thirty seconds of a resting hand and
produce a number that describes neither.

These features are the most modality-dependent in the system. A trackpad and a
mouse produce genuinely different values for the same person, which is why
coverage and the discriminability weighting matter here more than anywhere
else: if a user enrolls on a trackpad and logs in with a mouse, the pointer
features should lose influence, not veto the login.
"""

from __future__ import annotations

import numpy as np

from app.behavioral import stats
from app.behavioral.events import PointerSegment, SessionView

Observation = tuple[float, int]

# Samples closer together than this give a division that amplifies rounding
# noise into an implausible velocity.
MIN_SAMPLE_DT_MS = 1.0
# Movement direction is only meaningful once the pointer has actually moved.
MIN_REVERSAL_STEP_PX = 1.5


def extract(view: SessionView) -> dict[str, Observation]:
    out: dict[str, Observation] = {}
    _motion_features(view.segments, out)
    _click_features(view, out)
    return out


def _motion_features(
    segments: list[PointerSegment], out: dict[str, Observation]
) -> None:
    if not segments:
        return

    speeds: list[np.ndarray] = []
    accels: list[np.ndarray] = []
    tremors: list[np.ndarray] = []
    straightness: list[float] = []
    reversals: list[float] = []

    for segment in segments:
        dt = np.diff(segment.t)
        dx = np.diff(segment.x)
        dy = np.diff(segment.y)
        usable = dt >= MIN_SAMPLE_DT_MS
        if np.count_nonzero(usable) < 2:
            continue

        step = np.hypot(dx[usable], dy[usable])
        speed = step / dt[usable]
        speeds.append(speed)

        if speed.size >= 2:
            accels.append(np.abs(np.diff(speed)) / dt[usable][1:])

        # Second difference of position: what is left after the intended
        # trajectory is removed, i.e. fine motor wobble. A synthesised path
        # interpolated between waypoints has almost none.
        if segment.x.size >= 3:
            tremors.append(np.hypot(np.diff(segment.x, 2), np.diff(segment.y, 2)))

        displacement = segment.displacement
        if displacement > 1.0:
            # 1.0 is a perfectly straight line. Human movement lands above it.
            straightness.append(segment.path_length / displacement)

        reversals.append(_direction_changes(dx, dy))

    if speeds:
        all_speeds = np.concatenate(speeds)
        if all_speeds.size:
            out["ptr_velocity_median"] = (stats.median(all_speeds), all_speeds.size)
            out["ptr_velocity_p90"] = (stats.percentile(all_speeds, 90), all_speeds.size)

    if accels:
        all_accels = np.concatenate(accels)
        if all_accels.size:
            out["ptr_accel_peak"] = (stats.percentile(all_accels, 90), all_accels.size)

    if tremors:
        all_tremors = np.concatenate(tremors)
        if all_tremors.size:
            out["ptr_tremor"] = (stats.median(all_tremors), all_tremors.size)

    if straightness:
        values = np.asarray(straightness, dtype=float)
        out["ptr_straightness"] = (stats.median(values), values.size)

    if reversals:
        values = np.asarray(reversals, dtype=float)
        out["ptr_reversal_rate"] = (stats.median(values), values.size)


def _direction_changes(dx: np.ndarray, dy: np.ndarray) -> float:
    """Count sign flips in each axis, normalised by the number of steps.

    Overshooting a target and correcting back is a motor-control habit. Its
    absence is one of the cheaper ways a straight-line synthetic path gives
    itself away.
    """
    moving = np.hypot(dx, dy) >= MIN_REVERSAL_STEP_PX
    if np.count_nonzero(moving) < 3:
        return 0.0

    changes = 0
    for axis in (dx[moving], dy[moving]):
        signs = np.sign(axis)
        signs = signs[signs != 0]
        if signs.size >= 2:
            changes += int(np.count_nonzero(np.diff(signs) != 0))

    return changes / float(np.count_nonzero(moving))


def _click_features(view: SessionView, out: dict[str, Observation]) -> None:
    """Button hold time, matched down-to-up per button."""
    pending: dict[int, float] = {}
    dwells: list[float] = []

    for event in view.click_events:
        if event.type == "pointerdown":
            pending[event.button] = event.t
        elif event.type == "pointerup":
            down_t = pending.pop(event.button, None)
            if down_t is not None:
                held = event.t - down_t
                # A "click" held for over a second is a drag or a stuck button.
                if 0 <= held <= 1000:
                    dwells.append(held)

    values = np.asarray(dwells, dtype=float)
    if values.size:
        out["ptr_click_dwell_median"] = (stats.median(values), values.size)
