"""Form-level interaction patterns.

These describe how a person works through a form rather than how they type or
move: where they hesitate, whether they Tab or click between fields, how long
they sit on a filled form before committing.

They are cheap to compute, orthogonal to the keystroke and pointer features,
and they are the features an impostor is least likely to think about imitating.
Someone deliberately mimicking a typing rhythm will still navigate the form the
way they always navigate forms.
"""

from __future__ import annotations

import numpy as np

from app.behavioral import stats
from app.behavioral.events import SessionView

Observation = tuple[float, int]

# Focus arrivals we attribute to the user rather than to script. Programmatic
# focus (including our own autofocus) says nothing about navigation habit.
_USER_DRIVEN_FOCUS = frozenset({"tab", "pointer"})


def extract(view: SessionView) -> dict[str, Observation]:
    out: dict[str, Observation] = {}
    _first_interaction(view, out)
    _navigation_habit(view, out)
    _focus_to_key_latency(view, out)
    _presubmit_hesitation(view, out)
    return out


def _first_interaction(view: SessionView, out: dict[str, Observation]) -> None:
    """How long the form sits untouched before the first real action.

    Logged because the range is enormous — a quarter second to half a minute —
    and a raw millisecond value would let this one feature dominate every
    distance computation.
    """
    candidates: list[float] = [p.down_t for p in view.presses]
    candidates.extend(e.t for e in view.click_events if e.type == "pointerdown")
    if view.pointer_t.size:
        candidates.append(float(view.pointer_t[0]))

    if not candidates:
        return

    first_t = min(candidates)
    out["int_time_to_first_interaction"] = (float(np.log(max(first_t, 1.0))), 1)


def _navigation_habit(view: SessionView, out: dict[str, Observation]) -> None:
    focus_arrivals = [e for e in view.focus_events if e.type == "focus"]
    user_driven = [e for e in focus_arrivals if e.via in _USER_DRIVEN_FOCUS]
    if not user_driven:
        # Everything was programmatic focus, so there is no habit to observe.
        # Absent, not zero: zero would claim "never uses Tab", which is a
        # different statement from "we could not tell".
        return

    tabbed = sum(1 for e in user_driven if e.via == "tab")
    out["int_tab_transition_ratio"] = (tabbed / len(user_driven), len(user_driven))


def _focus_to_key_latency(view: SessionView, out: dict[str, Observation]) -> None:
    """Gap between landing in a field and the first keystroke there."""
    presses_by_ctx: dict[str, list[float]] = {}
    for press in view.presses:
        presses_by_ctx.setdefault(press.ctx, []).append(press.down_t)
    for times in presses_by_ctx.values():
        times.sort()

    latencies: list[float] = []
    for event in view.focus_events:
        if event.type != "focus":
            continue
        times = presses_by_ctx.get(event.ctx)
        if not times:
            continue
        following = [t for t in times if t >= event.t]
        if not following:
            continue
        gap = following[0] - event.t
        # Beyond a few seconds the user went and did something else; that is
        # not a measurement of reaction time.
        if 0 <= gap <= 5000:
            latencies.append(gap)

    values = np.asarray(latencies, dtype=float)
    if values.size:
        out["int_focus_to_first_key_median"] = (stats.median(values), values.size)


def _presubmit_hesitation(view: SessionView, out: dict[str, Observation]) -> None:
    """Pause between the last keystroke and committing the form.

    Logged for the same reason as time-to-first-interaction: the raw range
    spans three orders of magnitude.
    """
    if view.submit_t is None or not view.presses:
        return

    last_key_t = max(p.down_t for p in view.presses)
    gap = view.submit_t - last_key_t
    if gap < 0:
        return
    out["int_presubmit_hesitation"] = (float(np.log(max(gap, 1.0))), 1)
