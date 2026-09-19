"""The detector against a real browser-automation capture.

Unlike the synthetic fixtures, this stream was produced by an actual
CDP-driven browser filling the live enrollment form. It is the only test here
that is not checking the code against our own assumptions about what
automation looks like.

Two properties of the capture make it worth keeping permanently:

  * every event has trusted=true, because CDP input is real browser input
  * navigator.webdriver is false

A detector built on either flag would have let this through.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.behavioral.bot_detection.detector import detect_automation
from app.behavioral.events import build_session_view
from app.behavioral.scoring.reasons import ReasonCode
from app.behavioral.scoring.risk_engine import AUTOMATION_BLOCK
from app.models.events import BehaviorSessionIn

CAPTURE = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "captures"
    / "cdp_automation_2026-09-19.json"
)


def load_capture() -> BehaviorSessionIn:
    payload = json.loads(CAPTURE.read_text(encoding="utf-8"))
    return BehaviorSessionIn.model_validate(payload["session"])


def test_capture_really_does_look_legitimate_by_the_naive_checks():
    """Establishes why this capture matters before testing detection on it."""
    session = load_capture()

    assert session.meta.webdriver is False
    assert all(event.trusted for event in session.events)


def test_real_cdp_automation_is_detected():
    session = load_capture()
    result = detect_automation(build_session_view(session), session)

    assert result.score >= AUTOMATION_BLOCK, [s.detail for s in result.signals]


def test_detection_comes_from_event_ordering_not_from_a_self_declared_flag():
    session = load_capture()
    result = detect_automation(build_session_view(session), session)
    codes = {signal.code for signal in result.signals}

    assert ReasonCode.SYNTHETIC_EVENT_ORDER in codes
    # The low-weight flag signal must not be what carried this.
    assert ReasonCode.UNTRUSTED_EVENTS not in codes


def test_no_keystrokes_were_produced_at_all():
    """The tool inserts text in bulk rather than dispatching key events.

    That is what the orphaned input events are: three field updates across a
    70-character form fill, with not one keydown behind them.
    """
    session = load_capture()
    view = build_session_view(session)

    assert view.presses == []
    assert len([e for e in session.events if e.type == "input"]) == 3
