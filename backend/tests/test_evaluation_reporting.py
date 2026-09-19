"""The evaluation harness must not mix credential failures into FAR and FRR.

These cover evaluation/attempts.py rather than the product. They are here
because the number a judge reads comes out of that module, and a reporting bug
there is indistinguishable from an accuracy claim — the first real-human run
reported 85.7% genuine acceptance when the behavioural figure was 92.3%, purely
because three mistyped passwords were counted as behavioural rejections.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVALUATION = Path(__file__).resolve().parents[2] / "evaluation"
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

from attempts import (  # noqa: E402
    ALLOWED,
    BLOCKED,
    CLEAN_PROTOCOL_MIN,
    MIN_SAMPLES_FOR_A_RATE,
    PASSWORD_FAILURE,
    PhaseSummary,
    classify,
)


def row(
    attempt_id: int,
    decision: str = ALLOWED,
    reason: str = "BEHAVIOR_MATCH",
    identity: float | None = 0.1,
    automation: float | None = 0.0,
    threshold: float | None = 0.33,
) -> dict:
    return {
        "id": attempt_id,
        "decision": decision,
        "reason": reason,
        "identity_score": identity,
        "automation_score": automation,
        "threshold": threshold,
        "coverage": 1.0,
        "latency_ms": 120.0,
        "created_at": 1_700_000_000.0 + attempt_id,
        "reasons_json": '["KEYSTROKE_MISMATCH"]',
    }


def password_failure(attempt_id: int) -> dict:
    """What the server actually writes for a wrong password: no scores at all."""
    return row(
        attempt_id,
        decision=BLOCKED,
        reason="INVALID_CREDENTIALS",
        identity=None,
        automation=None,
        threshold=None,
    )


def test_a_wrong_password_is_not_a_behavioural_decision():
    record = classify(password_failure(1), "GENUINE")

    assert record.password_valid is False
    assert record.behavioral_decision is None
    assert record.behavioral_score is None
    assert record.outcome == PASSWORD_FAILURE


def test_a_correct_password_that_was_blocked_behaviourally_is_one():
    record = classify(
        row(1, decision=BLOCKED, reason="KEYSTROKE_MISMATCH", identity=0.51),
        "IMPOSTOR",
    )

    assert record.password_valid is True
    assert record.behavioral_decision == BLOCKED
    assert record.rejection_reason == "KEYSTROKE_MISMATCH"


def test_frr_is_computed_without_the_password_failures():
    # 12 accepted, 1 behavioural rejection, 1 mistyped password: the shape of
    # the first real-human genuine phase.
    records = [classify(row(i), "GENUINE") for i in range(12)]
    records.append(
        classify(row(12, decision=BLOCKED, reason="AUTOMATION_DETECTED"), "GENUINE")
    )
    records.append(classify(password_failure(13), "GENUINE"))
    phase = PhaseSummary("GENUINE", records)

    assert phase.total == 14
    assert len(phase.behavioral) == 13
    assert len(phase.password_failures) == 1

    frr = phase.rate(lambda r: r.behavioral_decision == BLOCKED)
    assert frr == pytest.approx(1 / 13)
    # and emphatically not 1/14, which is what counting the typo would give
    assert frr != pytest.approx(1 / 14)


def test_password_failures_are_reported_rather_than_dropped():
    phase = PhaseSummary(
        "IMPOSTOR",
        [classify(row(i, decision=BLOCKED, reason="KEYSTROKE_MISMATCH"), "IMPOSTOR")
         for i in range(8)]
        + [classify(password_failure(8), "IMPOSTOR"),
           classify(password_failure(9), "IMPOSTOR")],
    )

    summary = phase.as_dict()
    assert summary["attempts_total"] == 10
    assert summary["behavioral_n"] == 8
    assert summary["password_failures"] == 2
    assert summary["password_failure_ids"] == [8, 9]


def test_a_phase_with_any_password_failure_does_not_meet_the_clean_protocol():
    clean = PhaseSummary(
        "GENUINE", [classify(row(i), "GENUINE") for i in range(CLEAN_PROTOCOL_MIN)]
    )
    assert clean.as_dict()["meets_clean_protocol"] is True

    dirty = PhaseSummary(
        "GENUINE",
        [classify(row(i), "GENUINE") for i in range(CLEAN_PROTOCOL_MIN)]
        + [classify(password_failure(99), "GENUINE")],
    )
    assert dirty.as_dict()["meets_clean_protocol"] is False


def test_too_few_behavioural_attempts_yields_no_rate_at_all():
    """Four attempts do not make a percentage, whatever they did."""
    phase = PhaseSummary(
        "GENUINE",
        [classify(row(i), "GENUINE") for i in range(MIN_SAMPLES_FOR_A_RATE - 1)],
    )

    assert phase.rate(lambda r: r.behavioral_decision == ALLOWED) is None


def test_password_failures_do_not_pad_a_phase_to_the_minimum():
    """Five attempts of which four are typos is still one observation."""
    phase = PhaseSummary(
        "GENUINE",
        [classify(row(0), "GENUINE")]
        + [classify(password_failure(i), "GENUINE") for i in range(1, 5)],
    )

    assert phase.total == MIN_SAMPLES_FOR_A_RATE
    assert phase.rate(lambda r: r.behavioral_decision == ALLOWED) is None


def test_a_recorded_threshold_is_used_and_a_missing_one_is_derived():
    recorded = classify(row(1, automation=0.4, threshold=0.2), "GENUINE")
    assert recorded.threshold == pytest.approx(0.2)
    assert recorded.threshold_source == "recorded"

    # Rows written before auth_attempts carried the column reconstruct it from
    # the profile threshold and the automation tightening, and say so.
    legacy = row(2, automation=0.4, threshold=None)
    del legacy["threshold"]
    derived = classify(legacy, "GENUINE", profile_threshold=0.3280)
    assert derived.threshold == pytest.approx(0.3280 * (1 - 0.5 * 0.4))
    assert derived.threshold_source == "derived"


def test_an_unenrolled_account_is_not_counted_as_a_behavioural_rejection():
    """The credential was fine but scoring never ran; it judges nothing."""
    record = classify(
        row(1, decision=BLOCKED, reason="ACCOUNT_NOT_ENROLLED", identity=None),
        "GENUINE",
    )

    assert record.behavioral_decision is None
    assert record.outcome == PASSWORD_FAILURE
