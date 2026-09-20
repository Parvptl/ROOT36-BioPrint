"""Canonical Aalto → BioPrint adapter.

Converts rows of the Aalto 136M Keystrokes dataset into the *same* `KeyPress`
and `SessionView` structures the browser pipeline builds, then hands them to
the *same* `keyboard.extract`. No keyboard feature mathematics is reimplemented
here, and none should ever be: if this file ever computes a dwell or a flight
time itself, the resulting prior stops describing the shipped extractor.

Two entry points, deliberately different:

``parse_aalto_file`` / ``extract_features``
    Ungated. Every value `keyboard.extract` produced, regardless of how few
    observations backed it, plus an explicit NaN for the unobservable Shift
    feature. Used by the ML dataset tooling, which does its own filtering.

``iter_sessions`` / ``extract_session_features``
    Gated exactly as production gates. Streaming, standard library only, no
    pandas — this is the path that walks 168,595 files.

The gated path applies the same three filters as
``app.behavioral.features.extractor.extract_features``: enabled, finite, and at
least ``min_observations`` backing observations. A prior built without those
gates would be describing feature values the running system would have
discarded.

The Shift limitation
--------------------
Aalto records keycode 16 for Shift and does not say which one. So keycode 16
maps to ``key_class="shift"``, which matches neither ``shift_left`` nor
``shift_right`` in ``keyboard._shift_features``. The count of shift presses is
therefore zero, and ``kbd_shift_right_ratio`` is simply **never emitted** —
not zero, not imputed, not guessed.

That is the safe encoding, and it is safe by construction rather than by a
downstream exclusion list. Mapping keycode 16 to ``ShiftLeft`` instead would
make the extractor emit a ratio of exactly 0.0, a fabricated value that would
then have to be remembered and removed somewhere else.
"""

from __future__ import annotations

import csv
import os
import sys
from collections.abc import Iterator
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.events import KeyPress, SessionView  # noqa: E402
from app.behavioral.features.keyboard import extract  # noqa: E402
from app.behavioral.features.registry import SPECS, Modality  # noqa: E402

# Features the Aalto dataset structurally cannot produce. Single source of
# truth: the precompute script imports this rather than keeping its own copy.
UNOBSERVED_FEATURES = frozenset({"kbd_shift_right_ratio"})

UNOBSERVED_REASON = (
    "Aalto records keycode 16 for Shift without distinguishing ShiftLeft from "
    "ShiftRight, so the left/right preference this feature measures is not "
    "present in the data."
)

# JS keyCode → DOM `event.code`, which is what KeyPress.code carries and what
# keyboard.LEFT_HAND_CODES is expressed in.
KEYCODE_TO_CODE: dict[int, str] = {
    8: "Backspace",
    9: "Tab",
    13: "Enter",
    # Deliberately NOT "ShiftLeft" — see the module docstring.
    16: "Shift",
    17: "Control",
    18: "Alt",
    20: "CapsLock",
    27: "Escape",
    32: "Space",
    # Digits
    48: "Digit0", 49: "Digit1", 50: "Digit2", 51: "Digit3", 52: "Digit4",
    53: "Digit5", 54: "Digit6", 55: "Digit7", 56: "Digit8", 57: "Digit9",
    # Letters
    65: "KeyA", 66: "KeyB", 67: "KeyC", 68: "KeyD", 69: "KeyE", 70: "KeyF",
    71: "KeyG", 72: "KeyH", 73: "KeyI", 74: "KeyJ", 75: "KeyK", 76: "KeyL",
    77: "KeyM", 78: "KeyN", 79: "KeyO", 80: "KeyP", 81: "KeyQ", 82: "KeyR",
    83: "KeyS", 84: "KeyT", 85: "KeyU", 86: "KeyV", 87: "KeyW", 88: "KeyX",
    89: "KeyY", 90: "KeyZ",
    # Punctuation (US layout)
    186: "Semicolon", 187: "Equal", 188: "Comma", 189: "Minus",
    190: "Period", 191: "Slash", 192: "Backquote",
    219: "BracketLeft", 220: "Backslash", 221: "BracketRight", 222: "Quote",
}

# Keycodes that produce a typed character. Space, digits, letters and the US
# punctuation block. Anything else is a modifier, a navigation key, or unknown.
_CHAR_KEYCODES = (
    frozenset({32})
    | frozenset(range(48, 58))
    | frozenset(range(65, 91))
    | frozenset({186, 187, 188, 189, 190, 191, 192, 219, 220, 221, 222})
)

_KEYBOARD_FEATURES = frozenset(
    name for name, spec in SPECS.items() if spec.modality is Modality.KEYBOARD
)


def _get_key_class(keycode: int) -> str:
    if keycode == 8:
        return "backspace"
    if keycode == 16:
        # Not shift_left, not shift_right. See the module docstring.
        return "shift"
    if keycode in _CHAR_KEYCODES:
        return "char"
    return "other"


def _to_keypress(keycode: int, press_time: float, release_time: float) -> KeyPress:
    return KeyPress(
        ctx="phrase",  # Aalto transcription is the analogue of our challenge phrase
        code=KEYCODE_TO_CODE.get(keycode, f"Unknown{keycode}"),
        key_class=_get_key_class(keycode),
        down_t=press_time,
        up_t=release_time,
        # Aalto does not flag auto-repeat, so we cannot claim a press was one.
        repeat=False,
        trusted=True,
    )


# --------------------------------------------------------------- streaming


class SessionRejection(Exception):
    """A session could not be turned into features. Carries the reason code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def iter_sessions(path: str | Path) -> Iterator[tuple[str, SessionView, int]]:
    """Yield (section_id, view, raw_row_count) for each sentence in one file.

    Streams: one participant file is held at a time, and within it one section
    at a time, so walking the whole dataset does not grow with its size. Rows
    arrive grouped by section in the published files; this does not rely on
    that, but it does rely on a section's rows being contiguous, which they
    are. Sections are emitted when the id changes.
    """
    path = Path(path)
    try:
        handle = open(path, "r", encoding="utf-8", errors="replace", newline="")
    except OSError:
        return

    with handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        try:
            header = next(reader)
        except StopIteration:
            return

        try:
            i_section = header.index("TEST_SECTION_ID")
            i_press = header.index("PRESS_TIME")
            i_release = header.index("RELEASE_TIME")
            i_keycode = header.index("KEYCODE")
        except ValueError:
            return  # not an Aalto keystroke file

        current_id: str | None = None
        presses: list[KeyPress] = []
        raw_rows = 0

        for row in reader:
            if len(row) <= i_keycode:
                continue
            section_id = row[i_section]
            if section_id != current_id:
                if current_id is not None and presses:
                    yield current_id, SessionView(presses=presses), raw_rows
                current_id, presses, raw_rows = section_id, [], 0

            raw_rows += 1
            try:
                keycode = int(row[i_keycode])
                press_time = float(row[i_press])
                release_time = float(row[i_release])
            except (TypeError, ValueError):
                continue
            presses.append(_to_keypress(keycode, press_time, release_time))

        if current_id is not None and presses:
            # keyboard.extract pairs consecutive presses, so order matters.
            yield current_id, SessionView(presses=presses), raw_rows


def extract_session_features(view: SessionView) -> dict[str, float]:
    """Keyboard features for one session, gated exactly as production gates.

    Raises SessionRejection with a reason code when nothing usable survives,
    so a caller can report *why* sessions were dropped rather than only how
    many.
    """
    import math

    presses = view.presses_in("phrase")
    if len(presses) < 10:
        raise SessionRejection("too_few_presses")

    raw = extract(view)
    if not raw:
        raise SessionRejection("no_features_produced")

    features: dict[str, float] = {}
    for name, (value, count) in raw.items():
        if name not in _KEYBOARD_FEATURES:
            continue
        if name in UNOBSERVED_FEATURES:
            # Should be unreachable given the keycode 16 mapping above. Kept as
            # a second line of defence: if the mapping is ever changed, a
            # fabricated Shift ratio still cannot reach the prior.
            continue
        if not math.isfinite(value):
            continue
        if count < SPECS[name].min_observations:
            continue
        features[name] = float(value)

    if not features:
        raise SessionRejection("all_features_below_min_observations")
    return features


# ------------------------------------------------------- ungated (ML tooling)


def parse_aalto_file(filepath: str) -> Iterator[SessionView]:
    """Yield one SessionView per sentence. Ungated; kept for the ML tooling."""
    if not os.path.exists(filepath):
        return
    for _section_id, view, _rows in iter_sessions(filepath):
        yield view


def extract_features(view: SessionView) -> dict:
    """Every value keyboard.extract produced, plus explicit NaN for Shift.

    Ungated on purpose: the ML dataset generator applies its own filtering.
    Do not use this to build the population prior — it would admit values the
    running system discards.
    """
    obs = extract(view)
    features = {k: v[0] for k, v in obs.items()}
    for name in UNOBSERVED_FEATURES:
        features[name] = float("nan")
    return features
