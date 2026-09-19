"""Automation detection.

Half of these tests are false-positive checks. A bot detector that fires on a
fast typist, a keyboard-only user, or someone pasting from a password manager
is worse than no bot detector, because it blocks genuine people for a reason
they cannot act on.
"""

from __future__ import annotations

from app.behavioral.bot_detection.detector import detect_automation
from app.behavioral.events import build_session_view
from app.behavioral.scoring.reasons import ReasonCode
from app.auth.challenge import generate_phrase
from tests.factories import (
    TypingStyle,
    events_as_dicts,
    human_session,
    rebuild_session,
    linear_pointer_session,
    scripted_session,
    value_injection_session,
)

PHRASE = "amber Willow granite pattern Copper thunder meadow"


def run(session):
    return detect_automation(build_session_view(session), session)


def codes(result) -> set[ReasonCode]:
    return {signal.code for signal in result.signals}


# ------------------------------------------------------------- true positives


def test_fixed_timer_script_is_detected():
    result = run(scripted_session(PHRASE))

    assert result.score > 0.9
    assert ReasonCode.LOW_TIMING_VARIABILITY in codes(result)
    assert ReasonCode.DWELL_DEGENERACY in codes(result)


def test_value_injection_is_detected_by_event_order():
    """Assigning element.value leaves input events with no key press behind them."""
    result = run(value_injection_session(PHRASE))

    assert result.score > 0.8
    assert ReasonCode.SYNTHETIC_EVENT_ORDER in codes(result)


def test_linear_pointer_path_is_detected():
    """Typing looks human; only the pointer geometry is synthetic."""
    result = run(linear_pointer_session(PHRASE, seed=21))

    assert ReasonCode.POINTER_ANOMALY in codes(result)


def test_faster_than_humanly_possible_typing_is_detected():
    result = run(scripted_session(PHRASE, interval_ms=8.0, dwell_ms=3.0))

    assert result.score > 0.9
    assert ReasonCode.IMPOSSIBLE_INTERACTION_SPEED in codes(result)


def test_quantised_timing_is_detected_even_with_varied_intervals():
    """The case that survives the variability test.

    A script that randomises its delay but rounds to a 20 ms grid has plenty of
    variation; the intervals just all sit on lattice points.
    """
    import random

    session = scripted_session(PHRASE)
    events = events_as_dicts(session)

    rng = random.Random(4)
    t = 0.0
    for event in sorted(events, key=lambda e: e["t"]):
        if event["type"] == "keydown":
            t += 20.0 * rng.randint(3, 9)  # varied, but always a multiple of 20
            event["t"] = t
        elif event["type"] in ("keyup", "input"):
            event["t"] = t + 20.0

    result = run(rebuild_session(session, events))
    assert ReasonCode.AUTOMATION_TIMING in codes(result)


def test_top_code_identifies_the_strongest_signal():
    result = run(value_injection_session(PHRASE))
    assert result.top_code is ReasonCode.SYNTHETIC_EVENT_ORDER


# ------------------------------------------------------------ false positives


def test_ordinary_human_session_is_not_flagged():
    for seed in range(8):
        result = run(human_session(generate_phrase(), seed=seed))
        assert result.score < 0.3, f"seed {seed}: {[s.detail for s in result.signals]}"


def test_a_fast_consistent_typist_is_not_flagged():
    """A touch typist is quick and fairly even. That is not automation."""
    quick = TypingStyle(
        iki_mean_ms=95.0, iki_jitter=0.20, dwell_mean_ms=58.0,
        dwell_jitter=0.18, overlap_prob=0.6, pause_prob=0.01,
    )
    for seed in range(5):
        result = run(human_session(generate_phrase(), style=quick, seed=seed))
        assert result.score < 0.45, f"seed {seed}: {[s.detail for s in result.signals]}"


def test_keyboard_only_user_is_not_flagged():
    """No pointer activity is not evidence of automation.

    Plenty of people fill a login form entirely from the keyboard. Treating
    absence of mouse movement as suspicious would reject them for being
    efficient.
    """
    for seed in range(5):
        result = run(human_session(generate_phrase(), seed=seed, use_pointer=False))
        assert ReasonCode.POINTER_ANOMALY not in codes(result)
        assert result.score < 0.3


def test_pasting_into_a_field_is_not_treated_as_injection():
    """A password manager fills fields without key presses. That is normal."""
    session = human_session(PHRASE, seed=31)
    events = events_as_dicts(session)
    events.append({"t": 1.0, "trusted": True, "type": "paste", "ctx": "password"})
    events.append({"t": 2.0, "trusted": True, "type": "input", "ctx": "password", "length": 14})

    result = run(rebuild_session(session, events))
    assert ReasonCode.SYNTHETIC_EVENT_ORDER not in codes(result)


def test_a_slow_hesitant_typist_is_not_flagged():
    slow = TypingStyle(
        iki_mean_ms=420.0, iki_jitter=0.5, dwell_mean_ms=130.0,
        overlap_prob=0.0, pause_prob=0.25, backspace_prob=0.15,
    )
    for seed in range(5):
        result = run(human_session(generate_phrase(), style=slow, seed=seed))
        assert result.score < 0.35, f"seed {seed}: {[s.detail for s in result.signals]}"


# ------------------------------------------------------------ weighting rules


def test_webdriver_flag_alone_cannot_block():
    """navigator.webdriver is one line to remove, so it must never decide.

    A session that looks entirely human but declares an automation driver
    should stay well under any blocking threshold on that basis alone.
    """
    session = human_session(PHRASE, seed=41)
    declared = rebuild_session(
        session,
        events_as_dicts(session),
        meta={**session.meta.model_dump(), "webdriver": True},
    )

    result = run(declared)
    assert result.score < 0.35
    assert result.top_code is ReasonCode.UNTRUSTED_EVENTS


def test_untrusted_events_alone_cannot_block():
    session = human_session(PHRASE, seed=42)
    events = [{**event, "trusted": False} for event in events_as_dicts(session)]

    result = run(rebuild_session(session, events))
    # Weight 0.35 caps what this signal can contribute on its own, by design.
    assert result.score < 0.4


def test_signals_below_the_reporting_floor_are_not_surfaced():
    result = run(human_session(generate_phrase(), seed=50))
    for signal in result.signals:
        assert signal.strength >= 0.15


def test_score_is_always_bounded():
    for session in (
        human_session(PHRASE, seed=60),
        scripted_session(PHRASE),
        value_injection_session(PHRASE),
        linear_pointer_session(PHRASE, seed=61),
    ):
        result = run(session)
        assert 0.0 <= result.score <= 1.0


def test_detection_survives_a_nearly_empty_stream():
    session = human_session(PHRASE, seed=70)
    trimmed = session.model_copy(update={"events": session.events[:3]})

    result = detect_automation(build_session_view(trimmed), trimmed)
    # Too little evidence to fire anything. Deciding "bot" from three events
    # would be guessing; that case is handled as insufficient signal instead.
    assert result.score < 0.3
