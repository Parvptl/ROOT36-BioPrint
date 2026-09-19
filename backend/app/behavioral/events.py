"""Turns a raw event list into the structures the feature modules need.

Built once per attempt and passed to every extractor. Re-deriving key pairings
or pointer segments inside each feature would multiply the work on the latency
critical path for no benefit.

This module only reshapes. It computes no features and makes no judgements.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from app.models.events import BehaviorSessionIn

# A key held longer than this is being held, not typed. Auto-repeat and
# alt-tabbing away mid-press both produce these; including them would make a
# distracted user look like a different person.
MAX_PLAUSIBLE_DWELL_MS = 1000.0

# Pointer motion separated by a gap this long is a new movement, not a
# continuation. Roughly the boundary between a pause and a new intent.
POINTER_SEGMENT_GAP_MS = 200.0

# Movements shorter than this are jitter around a resting hand, not travel.
MIN_SEGMENT_POINTS = 4
MIN_SEGMENT_DISTANCE_PX = 12.0


@dataclass(frozen=True)
class KeyPress:
    """A keydown matched with its keyup."""

    ctx: str
    code: str | None
    key_class: str
    down_t: float
    up_t: float | None
    repeat: bool
    trusted: bool

    @property
    def dwell(self) -> float | None:
        """How long the key was held, or None if unusable."""
        if self.up_t is None:
            return None
        value = self.up_t - self.down_t
        if value < 0 or value > MAX_PLAUSIBLE_DWELL_MS:
            return None
        return value


@dataclass
class PointerSegment:
    """One continuous stretch of pointer travel."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @property
    def path_length(self) -> float:
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        return float(np.sum(np.hypot(dx, dy)))

    @property
    def displacement(self) -> float:
        return float(np.hypot(self.x[-1] - self.x[0], self.y[-1] - self.y[0]))

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])


@dataclass
class SessionView:
    """Normalised view of one capture."""

    presses: list[KeyPress] = field(default_factory=list)
    segments: list[PointerSegment] = field(default_factory=list)
    pointer_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    pointer_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    pointer_y: np.ndarray = field(default_factory=lambda: np.empty(0))
    focus_events: list = field(default_factory=list)
    input_events: list = field(default_factory=list)
    click_events: list = field(default_factory=list)
    submit_t: float | None = None
    first_event_t: float | None = None
    duration_ms: float = 0.0
    untrusted_count: int = 0
    total_events: int = 0
    paste_count: int = 0
    unmatched_keyups: int = 0
    paste_contexts: set[str] = field(default_factory=set)

    def presses_in(self, *contexts: str) -> list[KeyPress]:
        wanted = set(contexts)
        return [p for p in self.presses if p.ctx in wanted]


def _pair_presses(events) -> tuple[list[KeyPress], int]:
    """Match each keydown with the keyup that released it.

    Keyed by physical code where available. In the password context the code is
    deliberately absent, so pairing falls back to the coarse class and matches
    FIFO. With key rollover that is an approximation — two characters can be
    down at once — but it is a consistent approximation, and the password
    context feeds automation detection only, never identity.
    """
    pending: dict[tuple[str, str], deque] = defaultdict(deque)
    presses: list[KeyPress] = []
    unmatched_keyups = 0

    for event in events:
        if event.type not in ("keydown", "keyup"):
            continue
        key = (event.ctx, event.code or f"class:{event.key_class}")

        if event.type == "keydown":
            press = KeyPress(
                ctx=event.ctx,
                code=event.code,
                key_class=event.key_class,
                down_t=event.t,
                up_t=None,
                repeat=event.repeat,
                trusted=event.trusted,
            )
            presses.append(press)
            pending[key].append(len(presses) - 1)
        else:
            queue = pending[key]
            if queue:
                index = queue.popleft()
                held = presses[index]
                presses[index] = KeyPress(
                    ctx=held.ctx,
                    code=held.code,
                    key_class=held.key_class,
                    down_t=held.down_t,
                    up_t=event.t,
                    repeat=held.repeat,
                    trusted=held.trusted and event.trusted,
                )
            else:
                # A release with nothing to release. Counted as a structural
                # fact about the stream; the automation detector decides what
                # it means.
                unmatched_keyups += 1

    presses.sort(key=lambda p: p.down_t)
    return presses, unmatched_keyups


def _segment_pointer(
    t: np.ndarray, x: np.ndarray, y: np.ndarray
) -> list[PointerSegment]:
    if t.size < MIN_SEGMENT_POINTS:
        return []

    gaps = np.diff(t)
    break_indices = np.flatnonzero(gaps > POINTER_SEGMENT_GAP_MS) + 1
    bounds = [0, *break_indices.tolist(), t.size]

    segments: list[PointerSegment] = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end - start < MIN_SEGMENT_POINTS:
            continue
        segment = PointerSegment(t=t[start:end], x=x[start:end], y=y[start:end])
        if segment.path_length < MIN_SEGMENT_DISTANCE_PX or segment.duration <= 0:
            continue
        segments.append(segment)
    return segments


def build_session_view(session: BehaviorSessionIn) -> SessionView:
    """Reshape a validated event stream. Pure function, no I/O."""
    view = SessionView()
    view.total_events = len(session.events)

    move_t: list[float] = []
    move_x: list[float] = []
    move_y: list[float] = []

    for event in session.events:
        if not event.trusted:
            view.untrusted_count += 1
        if view.first_event_t is None:
            view.first_event_t = event.t
        view.duration_ms = max(view.duration_ms, event.t)

        match event.type:
            case "pointermove":
                move_t.append(event.t)
                move_x.append(event.x)
                move_y.append(event.y)
            case "pointerdown" | "pointerup":
                view.click_events.append(event)
            case "focus" | "blur":
                view.focus_events.append(event)
            case "input":
                view.input_events.append(event)
            case "submit":
                if view.submit_t is None:
                    view.submit_t = event.t
            case "paste":
                view.paste_count += 1
                view.paste_contexts.add(event.ctx)

    view.presses, view.unmatched_keyups = _pair_presses(session.events)
    view.pointer_t = np.asarray(move_t, dtype=float)
    view.pointer_x = np.asarray(move_x, dtype=float)
    view.pointer_y = np.asarray(move_y, dtype=float)
    view.segments = _segment_pointer(view.pointer_t, view.pointer_x, view.pointer_y)

    return view
