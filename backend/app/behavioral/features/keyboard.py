"""Keystroke dynamics.

Scope: the challenge-phrase context only.

That restriction is the heart of the password-independence requirement. The
phrase is different on every attempt, so these features describe *how* someone
types rather than *what* they typed, and the profile does not break when the
password changes. The password field is still captured, but it feeds automation
detection alone and never contributes to identity.
"""

from __future__ import annotations

import numpy as np

from app.behavioral import stats
from app.behavioral.events import KeyPress, SessionView

# Consecutive keys landing closer than this are a burst: a familiar letter run
# fired as one motor programme rather than two decisions.
BURST_MS = 80.0
# A gap this long is a pause for thought, not typing tempo.
PAUSE_MS = 500.0
# Intervals beyond this are the user being interrupted. Clipped so an
# interruption cannot masquerade as a person who types very slowly.
MAX_TEMPO_IKI_MS = 5000.0
# Keys per window for the tempo-stability measure.
SPEED_WINDOW = 5

LEFT_HAND_CODES = frozenset(
    {
        "Backquote", "Digit1", "Digit2", "Digit3", "Digit4", "Digit5",
        "KeyQ", "KeyW", "KeyE", "KeyR", "KeyT",
        "KeyA", "KeyS", "KeyD", "KeyF", "KeyG",
        "KeyZ", "KeyX", "KeyC", "KeyV", "KeyB",
        "Tab", "CapsLock", "ShiftLeft", "ControlLeft", "AltLeft",
    }
)

# value, number of observations backing it
Observation = tuple[float, int]


def _character_presses(presses: list[KeyPress]) -> list[KeyPress]:
    """Deliberate character presses only.

    Auto-repeat is excluded: a held key emits presses microseconds apart, which
    would read as impossibly fast typing and drag the tempo features toward a
    machine-like profile.
    """
    return [p for p in presses if p.key_class == "char" and not p.repeat]


def extract(view: SessionView) -> dict[str, Observation]:
    presses = view.presses_in("phrase")
    chars = _character_presses(presses)
    out: dict[str, Observation] = {}

    _dwell_features(chars, out)
    _transition_features(chars, out)
    _rhythm_features(chars, out)
    _correction_features(presses, out)
    _shift_features(presses, out)

    return out


def _dwell_features(chars: list[KeyPress], out: dict[str, Observation]) -> None:
    dwells = np.array([d for p in chars if (d := p.dwell) is not None], dtype=float)
    n = dwells.size
    if n:
        out["kbd_dwell_median"] = (stats.median(dwells), n)
        out["kbd_dwell_mad"] = (stats.mad(dwells), n)

    left = np.array(
        [d for p in chars if p.code in LEFT_HAND_CODES and (d := p.dwell) is not None],
        dtype=float,
    )
    right = np.array(
        [
            d
            for p in chars
            if p.code is not None and p.code not in LEFT_HAND_CODES and (d := p.dwell) is not None
        ],
        dtype=float,
    )
    if left.size:
        out["kbd_dwell_left_median"] = (stats.median(left), left.size)
    if right.size:
        out["kbd_dwell_right_median"] = (stats.median(right), right.size)


def _transition_features(chars: list[KeyPress], out: dict[str, Observation]) -> None:
    """Flight time: release of one key to press of the next.

    Negative values are not errors. They mean the next key went down before the
    previous came up, which is exactly what fluent touch typing looks like, and
    the proportion of overlapping transitions is one of the strongest
    person-level signals available here.
    """
    flights: list[float] = []
    for previous, current in zip(chars[:-1], chars[1:]):
        if previous.up_t is None:
            continue
        flights.append(current.down_t - previous.up_t)

    values = np.asarray(flights, dtype=float)
    if values.size == 0:
        return

    out["kbd_flight_median"] = (stats.median(values), values.size)
    out["kbd_flight_iqr"] = (stats.iqr(values), values.size)
    out["kbd_flight_negative_frac"] = (stats.fraction(values < 0), values.size)


def _rhythm_features(chars: list[KeyPress], out: dict[str, Observation]) -> None:
    if len(chars) < 2:
        return

    down_times = np.array([p.down_t for p in chars], dtype=float)
    ikis = np.diff(down_times)
    n = ikis.size
    if n == 0:
        return

    tempo = np.clip(ikis, 1.0, MAX_TEMPO_IKI_MS)
    log_tempo = stats.safe_log(tempo)

    out["kbd_logiki_median"] = (stats.median(log_tempo), n)
    out["kbd_logiki_mad"] = (stats.mad(log_tempo), n)
    out["kbd_burst_fraction"] = (stats.fraction(ikis < BURST_MS), n)
    # Expressed per 100 keys so the value does not depend on phrase length.
    out["kbd_pause_rate"] = (float(np.count_nonzero(ikis > PAUSE_MS)) / n * 100.0, n)

    elapsed = float(down_times[-1] - down_times[0])
    if elapsed > 0:
        out["kbd_speed_kps"] = (len(chars) / (elapsed / 1000.0), len(chars))

    if len(chars) > SPEED_WINDOW:
        # Rolling rate over fixed-size key windows. A person who sprints through
        # familiar words and stalls on unfamiliar ones scores high here even if
        # their average speed matches someone who types metronomically.
        spans = down_times[SPEED_WINDOW:] - down_times[:-SPEED_WINDOW]
        spans = spans[spans > 0]
        if spans.size:
            rates = SPEED_WINDOW / (spans / 1000.0)
            out["kbd_speed_cv"] = (stats.robust_cv(rates), rates.size)


def _correction_features(presses: list[KeyPress], out: dict[str, Observation]) -> None:
    typed = [p for p in presses if not p.repeat and p.key_class in ("char", "backspace")]
    if not typed:
        return

    corrections = [p for p in typed if p.key_class == "backspace"]
    out["kbd_backspace_rate"] = (len(corrections) / len(typed), len(typed))

    # How long after the offending keystroke the correction arrives: a measure
    # of how quickly someone notices their own mistakes.
    latencies: list[float] = []
    for previous, current in zip(typed[:-1], typed[1:]):
        if current.key_class == "backspace" and previous.key_class == "char":
            latencies.append(current.down_t - previous.down_t)

    values = np.asarray(latencies, dtype=float)
    if values.size:
        out["kbd_backspace_latency_median"] = (stats.median(values), values.size)


def _shift_features(presses: list[KeyPress], out: dict[str, Observation]) -> None:
    """Which Shift key the person reaches for.

    Only observable because the challenge phrase capitalises two random words.
    Sourcing it there rather than from the password means the trait costs no
    information about the secret.
    """
    left = sum(1 for p in presses if p.key_class == "shift_left" and not p.repeat)
    right = sum(1 for p in presses if p.key_class == "shift_right" and not p.repeat)
    total = left + right
    if total:
        out["kbd_shift_right_ratio"] = (right / total, total)
