"""Synthetic behavioural event streams for testing.

IMPORTANT, and repeated in the README and report: these generators exist to
test that extraction, scoring and detection logic behave correctly on
known-shaped input. They are NOT a source of accuracy numbers. A model
evaluated against the same assumptions used to fabricate its test data
measures nothing.

Every reported true/false acceptance rate must come from real captures made
through the app by real people. These fixtures only answer "does the code do
what it says".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from app.models.events import BehaviorSessionIn

# Rough QWERTY mapping, enough for the lowercase words and spaces the
# challenge phrases are built from.
_CODE_FOR_CHAR = {c: f"Key{c.upper()}" for c in "abcdefghijklmnopqrstuvwxyz"}
_CODE_FOR_CHAR[" "] = "Space"


@dataclass
class TypingStyle:
    """Parameters describing one synthetic typist.

    Two styles with different parameters stand in for two different people;
    the same style with a different seed stands in for the same person on a
    different day.
    """

    iki_mean_ms: float = 150.0
    iki_jitter: float = 0.35          # lognormal sigma on inter-key intervals
    dwell_mean_ms: float = 85.0
    dwell_jitter: float = 0.25
    overlap_prob: float = 0.30        # chance the next key goes down early
    pause_prob: float = 0.05          # chance of a thinking pause
    pause_ms: float = 700.0
    backspace_prob: float = 0.03
    right_shift_prob: float = 0.8     # which Shift key this typist reaches for
    tab_between_fields: bool = True
    focus_delay_ms: float = 320.0
    presubmit_ms: float = 620.0
    pointer_speed: float = 1.4        # px per ms
    pointer_wobble: float = 2.2


@dataclass
class _Builder:
    rng: random.Random
    # Pointer motion draws from its own stream.
    #
    # With a single shared stream, typing a longer or shorter phrase consumes a
    # different number of draws, so the mouse path changes purely because the
    # text changed. Real pointer behaviour has no such dependency, and the
    # coupling measurably inflated apparent genuine variance: pointer feature
    # spread roughly doubled between same-phrase and different-phrase captures,
    # which is impossible outside the simulator. Separating the streams removes
    # an artefact that was making the evaluation pessimistic for the wrong
    # reason.
    prng: random.Random = field(default_factory=lambda: random.Random(0))
    events: list[dict[str, Any]] = field(default_factory=list)
    t: float = 0.0

    def advance(self, ms: float) -> None:
        self.t += max(0.0, ms)

    def add(self, **event: Any) -> None:
        self.events.append({"t": round(self.t, 3), "trusted": True, **event})

    def sorted_events(self) -> list[dict[str, Any]]:
        # Stable sort keeps keydown before keyup when they share a timestamp.
        return sorted(self.events, key=lambda e: e["t"])


def _lognormal(rng: random.Random, mean: float, sigma: float) -> float:
    """Draw from a lognormal with the given median. Timing is not symmetric:
    you can be much slower than usual, never much faster than zero."""
    return mean * rng.lognormvariate(0.0, sigma)


def _type_text(
    builder: _Builder,
    text: str,
    ctx: str,
    style: TypingStyle,
    redact: bool,
) -> None:
    """Emit a plausible keydown/keyup stream for `text`.

    When `redact` is set the stream mimics what the real collector produces in
    the password context: no key codes, and Shift collapsed to `other`.
    """
    rng = builder.rng
    length = 0
    previous_up_t: float | None = None

    for char in text:
        if rng.random() < style.pause_prob:
            builder.advance(style.pause_ms * rng.uniform(0.5, 1.6))

        if char.isupper():
            shift_right = rng.random() < style.right_shift_prob
            shift_class = "shift_right" if shift_right else "shift_left"
            shift_code = "ShiftRight" if shift_right else "ShiftLeft"
            if redact:
                shift_class, shift_code = "other", None
            builder.add(
                type="keydown", ctx=ctx, key_class=shift_class,
                code=shift_code, repeat=False,
            )
            shift_down_t = builder.t
            builder.advance(_lognormal(rng, 40.0, 0.2))

        code = None if redact else _CODE_FOR_CHAR.get(char.lower(), "KeyX")
        builder.add(type="keydown", ctx=ctx, key_class="char", code=code, repeat=False)
        down_t = builder.t

        dwell = _lognormal(rng, style.dwell_mean_ms, style.dwell_jitter)
        gap = _lognormal(rng, style.iki_mean_ms, style.iki_jitter)

        if rng.random() < style.overlap_prob and gap > 12:
            # Next key goes down before this one comes up: fluent typing.
            builder.advance(gap)
            release_at = down_t + dwell
            builder.events.append(
                {"t": round(release_at, 3), "trusted": True, "type": "keyup",
                 "ctx": ctx, "key_class": "char", "code": code, "repeat": False}
            )
        else:
            builder.advance(dwell)
            builder.add(type="keyup", ctx=ctx, key_class="char", code=code, repeat=False)
            builder.advance(max(0.0, gap - dwell))

        previous_up_t = down_t + dwell
        length += 1
        builder.events.append(
            {"t": round(builder.t, 3), "trusted": True, "type": "input",
             "ctx": ctx, "length": length}
        )

        if char.isupper():
            builder.events.append(
                {"t": round(shift_down_t + 90.0, 3), "trusted": True, "type": "keyup",
                 "ctx": ctx, "key_class": shift_class, "code": shift_code, "repeat": False}
            )

        if rng.random() < style.backspace_prob:
            builder.advance(_lognormal(rng, 260.0, 0.3))
            bs_code = None if redact else "Backspace"
            builder.add(type="keydown", ctx=ctx, key_class="backspace", code=bs_code, repeat=False)
            builder.advance(_lognormal(rng, 80.0, 0.2))
            builder.add(type="keyup", ctx=ctx, key_class="backspace", code=bs_code, repeat=False)
            length = max(0, length - 1)
            builder.events.append(
                {"t": round(builder.t, 3), "trusted": True, "type": "input",
                 "ctx": ctx, "length": length}
            )
            builder.advance(_lognormal(rng, style.iki_mean_ms, style.iki_jitter))

    _ = previous_up_t


def _move_pointer(
    builder: _Builder, start: tuple[float, float], end: tuple[float, float], style: TypingStyle
) -> None:
    """A curved, jittery path with a deceleration toward the target."""
    rng = builder.prng
    x0, y0 = start
    x1, y1 = end
    distance = max(1.0, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    steps = max(6, int(distance / 14))

    # Bow the path sideways so it is not a straight line, then add per-sample
    # wobble so the second difference is non-trivial.
    bow = rng.uniform(-0.22, 0.22) * distance

    for i in range(steps + 1):
        p = i / steps
        ease = p * p * (3 - 2 * p)  # smoothstep: accelerate then decelerate
        perp = 4 * ease * (1 - ease) * bow
        nx = x0 + (x1 - x0) * ease - (y1 - y0) / distance * perp
        ny = y0 + (y1 - y0) * ease + (x1 - x0) / distance * perp
        nx += rng.gauss(0, style.pointer_wobble)
        ny += rng.gauss(0, style.pointer_wobble)
        builder.add(type="pointermove", x=round(nx, 1), y=round(ny, 1))
        builder.advance(max(4.0, distance / steps / style.pointer_speed * rng.uniform(0.8, 1.3)))


def human_session(
    phrase: str,
    nonce: str = "test-nonce-0000000000000000",
    style: TypingStyle | None = None,
    seed: int = 0,
    username: str = "testuser",
    password: str = "hunter2hunter2",
    use_pointer: bool = True,
) -> BehaviorSessionIn:
    """A full form fill: username, password, phrase, submit."""
    style = style or TypingStyle()
    builder = _Builder(
        rng=random.Random(seed),
        # Offset so the two streams never coincide, still fully determined by
        # `seed` so captures stay reproducible.
        prng=random.Random(seed + 0x5EED0),
    )

    builder.advance(_lognormal(builder.rng, 900.0, 0.4))  # time to first interaction

    cursor = (240.0, 180.0)
    if use_pointer:
        _move_pointer(builder, (60.0, 500.0), (240.0, 200.0), style)
        builder.add(type="pointerdown", ctx="username", x=240, y=200, button=0)
        builder.advance(_lognormal(builder.rng, 70.0, 0.25))
        builder.add(type="pointerup", ctx="username", x=240, y=200, button=0)
        cursor = (240.0, 200.0)
        builder.add(type="focus", ctx="username", via="pointer")
    else:
        builder.add(type="focus", ctx="username", via="programmatic")

    builder.advance(style.focus_delay_ms * builder.rng.uniform(0.7, 1.4))
    _type_text(builder, username, "username", style, redact=False)

    _enter_field(builder, style, "username", "password", cursor, (240.0, 280.0), use_pointer)
    if style.tab_between_fields:
        cursor = cursor
    else:
        cursor = (240.0, 280.0)
    builder.advance(style.focus_delay_ms * builder.rng.uniform(0.7, 1.4))
    _type_text(builder, password, "password", style, redact=True)

    _enter_field(builder, style, "password", "phrase", cursor, (240.0, 400.0), use_pointer)
    if not style.tab_between_fields:
        cursor = (240.0, 400.0)
    builder.advance(style.focus_delay_ms * builder.rng.uniform(0.8, 1.6))
    _type_text(builder, phrase, "phrase", style, redact=False)

    builder.add(type="blur", ctx="phrase", via="unknown")
    builder.advance(style.presubmit_ms * builder.rng.uniform(0.6, 1.5))

    if use_pointer:
        _move_pointer(builder, cursor, (240.0, 470.0), style)
        builder.add(type="pointerdown", ctx="page", x=240, y=470, button=0)
        builder.advance(_lognormal(builder.rng, 72.0, 0.25))
        builder.add(type="pointerup", ctx="page", x=240, y=470, button=0)

    builder.add(type="submit", ctx="page")

    return BehaviorSessionIn.model_validate(
        {
            "nonce": nonce,
            "phrase_typed": phrase,
            "started_at_ms": 1_700_000_000_000.0,
            "events": builder.sorted_events(),
            "meta": {
                "webdriver": False,
                "pointer_type": "mouse" if use_pointer else "none",
                "screen_w": 1920,
                "screen_h": 1080,
                "hardware_concurrency": 8,
            },
        }
    )


def _enter_field(
    builder: _Builder,
    style: TypingStyle,
    from_ctx: str,
    to_ctx: str,
    cursor: tuple[float, float],
    target: tuple[float, float],
    use_pointer: bool,
) -> None:
    if style.tab_between_fields:
        # The real collector redacts the key code for everything typed in the
        # password field, Tab included. Mirrored here so fixtures exercise the
        # same shape the server actually receives.
        tab_code = None if from_ctx == "password" else "Tab"
        builder.add(type="keydown", ctx=from_ctx, key_class="nav", code=tab_code, repeat=False)
        builder.advance(_lognormal(builder.rng, 75.0, 0.2))
        builder.add(type="keyup", ctx=from_ctx, key_class="nav", code=tab_code, repeat=False)
        builder.add(type="blur", ctx=from_ctx, via="unknown")
        builder.add(type="focus", ctx=to_ctx, via="tab")
        return

    builder.add(type="blur", ctx=from_ctx, via="unknown")
    if use_pointer:
        _move_pointer(builder, cursor, target, style)
        builder.add(type="pointerdown", ctx=to_ctx, x=int(target[0]), y=int(target[1]), button=0)
        builder.advance(_lognormal(builder.rng, 70.0, 0.25))
        builder.add(type="pointerup", ctx=to_ctx, x=int(target[0]), y=int(target[1]), button=0)
        builder.add(type="focus", ctx=to_ctx, via="pointer")
    else:
        builder.add(type="focus", ctx=to_ctx, via="programmatic")


def events_as_dicts(session: BehaviorSessionIn) -> list[dict[str, Any]]:
    """Plain dicts for a session's events, for tests that mutate the stream."""
    return [event.model_dump() for event in session.events]


def rebuild_session(
    session: BehaviorSessionIn,
    events: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> BehaviorSessionIn:
    """Revalidate a mutated event list back into a session.

    Going through full validation rather than model_copy keeps mutated fixtures
    honest: a test cannot accidentally construct a stream the real endpoint
    would have rejected.
    """
    return BehaviorSessionIn.model_validate(
        {
            "nonce": session.nonce,
            "phrase_typed": session.phrase_typed,
            "started_at_ms": session.started_at_ms,
            "events": sorted(events, key=lambda e: e["t"]),
            "meta": meta if meta is not None else session.meta.model_dump(),
        }
    )


# ----------------------------------------------------------------- attackers


def scripted_session(
    phrase: str,
    nonce: str = "test-nonce-0000000000000000",
    interval_ms: float = 45.0,
    dwell_ms: float = 20.0,
    username: str = "testuser",
    password: str = "hunter2hunter2",
) -> BehaviorSessionIn:
    """Keystrokes dispatched on a fixed timer.

    What a naive automation script produces: constant interval, constant dwell,
    no pointer activity, no hesitation anywhere.
    """
    events: list[dict[str, Any]] = []
    t = 5.0

    for ctx, text, redact in (
        ("username", username, False),
        ("password", password, True),
        ("phrase", phrase, False),
    ):
        events.append({"t": t, "trusted": False, "type": "focus", "ctx": ctx, "via": "programmatic"})
        for i, char in enumerate(text, start=1):
            code = None if redact else _CODE_FOR_CHAR.get(char.lower(), "KeyX")
            events.append({"t": t, "trusted": False, "type": "keydown", "ctx": ctx,
                           "key_class": "char", "code": code, "repeat": False})
            t += dwell_ms
            events.append({"t": t, "trusted": False, "type": "keyup", "ctx": ctx,
                           "key_class": "char", "code": code, "repeat": False})
            events.append({"t": t, "trusted": False, "type": "input", "ctx": ctx, "length": i})
            t += interval_ms - dwell_ms
        events.append({"t": t, "trusted": False, "type": "blur", "ctx": ctx, "via": "unknown"})

    events.append({"t": t, "trusted": False, "type": "submit", "ctx": "page"})

    return BehaviorSessionIn.model_validate(
        {
            "nonce": nonce,
            "phrase_typed": phrase,
            "started_at_ms": 1_700_000_000_000.0,
            "events": sorted(events, key=lambda e: e["t"]),
            "meta": {"webdriver": True, "pointer_type": "none",
                     "screen_w": 1280, "screen_h": 720, "hardware_concurrency": 2},
        }
    )


def value_injection_session(
    phrase: str,
    nonce: str = "test-nonce-0000000000000000",
    username: str = "testuser",
    password: str = "hunter2hunter2",
) -> BehaviorSessionIn:
    """Fields filled by assigning to element.value.

    The signature is an `input` event with no keydown before it. Cheapest and
    most common scripted fill there is.
    """
    events: list[dict[str, Any]] = []
    t = 2.0
    for ctx, text in (("username", username), ("password", password), ("phrase", phrase)):
        events.append({"t": t, "trusted": False, "type": "focus", "ctx": ctx, "via": "programmatic"})
        t += 1.0
        events.append({"t": t, "trusted": False, "type": "input", "ctx": ctx, "length": len(text)})
        t += 1.0

    events.append({"t": t, "trusted": False, "type": "submit", "ctx": "page"})

    return BehaviorSessionIn.model_validate(
        {
            "nonce": nonce,
            "phrase_typed": phrase,
            "started_at_ms": 1_700_000_000_000.0,
            "events": events,
            "meta": {"webdriver": True, "pointer_type": "none",
                     "screen_w": 1280, "screen_h": 720, "hardware_concurrency": 2},
        }
    )


def linear_pointer_session(
    phrase: str,
    nonce: str = "test-nonce-0000000000000000",
    seed: int = 1,
) -> BehaviorSessionIn:
    """Human-plausible typing with perfectly straight, evenly spaced pointer moves.

    Isolates the pointer-path signal: everything else looks human, so a
    detection here can only have come from the movement geometry.
    """
    session = human_session(phrase, nonce=nonce, seed=seed, use_pointer=False)
    events = events_as_dicts(session)

    # Two separate straight runs, so the path features have more than one
    # segment to average over, as a real login would.
    t = 30.0
    for origin_x, origin_y in ((100.0, 100.0), (700.0, 420.0)):
        for i in range(40):
            events.append({"t": t, "trusted": True, "type": "pointermove",
                           "x": origin_x + i * 8.0, "y": origin_y + i * 5.0})
            t += 10.0  # exactly 10 ms apart, exactly 8 px across: no human does this
        t += 500.0  # pause long enough to end the segment

    return rebuild_session(session, events)
