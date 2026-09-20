"""End-to-end: register, enroll, then the four demo cases.

The four cases the problem statement requires a demo to show:
  A. genuine user enrolls and logs in                      -> ALLOW
  B. correct password, different human behaviour           -> BLOCK
  C. scripted / automated attempt                          -> BLOCK
  D. wrong password                                        -> BLOCK (credential)
"""

from __future__ import annotations

import pytest

from app.api.ratelimit import limiter
from app.api.routes_auth import ENROLLMENT_ROUNDS
from app.behavioral.fingerprint.profile import MIN_FEATURE_SESSIONS
from app.auth.challenge import normalise_phrase
from tests.factories import (
    TypingStyle,
    human_session,
    rebuild_session,
    events_as_dicts,
    scripted_session,
    value_injection_session,
)

USERNAME = "alice"
PASSWORD = "correct-horse-battery"

ALICE = TypingStyle(
    iki_mean_ms=140.0, dwell_mean_ms=82.0, overlap_prob=0.45,
    right_shift_prob=0.95, tab_between_fields=True, pointer_speed=1.6,
)
MALLORY = TypingStyle(
    iki_mean_ms=320.0, dwell_mean_ms=140.0, overlap_prob=0.01,
    right_shift_prob=0.02, tab_between_fields=False, pointer_speed=0.6,
    backspace_prob=0.14, pause_prob=0.20,
)


@pytest.fixture(autouse=True)
def _clear_rate_limits():
    """Rate limiting is process-global; tests must not inherit each other's counts."""
    limiter.reset()
    yield
    limiter.reset()


# ------------------------------------------------------------------ helpers


def register(client, username=USERNAME, password=PASSWORD):
    response = client.post(
        "/auth/register",
        json={"username": username, "password": password, "consent": True},
    )
    assert response.status_code == 201, response.text
    return response.json()


def simulate_elapsed_time(age_challenge, nonce, session):
    """Age the challenge by however long this capture claims to have taken.

    Stands in for the wall-clock time a real user spends filling the form.
    """
    duration_seconds = max(e.t for e in session.events) / 1000.0
    age_challenge(nonce, duration_seconds + 1.0)


def enroll(client, age_challenge, style=ALICE, username=USERNAME, password=PASSWORD):
    """Run every enrollment round, returning the final progress payload."""
    last = None
    for index in range(20):  # generous bound; the server decides when it is done
        start = client.post(
            "/auth/enrollment/start", json={"username": username, "password": password}
        )
        if start.status_code == 409:
            break
        assert start.status_code == 200, start.text
        challenge = start.json()

        session = human_session(
            challenge["phrase"],
            nonce=challenge["nonce"],
            style=style,
            seed=index,
            username=username,
            password=password,
        )
        simulate_elapsed_time(age_challenge, challenge["nonce"], session)

        submitted = client.post(
            "/auth/enrollment/submit",
            json={"username": username, "session": session.model_dump()},
        )
        assert submitted.status_code == 200, submitted.text
        last = submitted.json()
        if last["profile_built"]:
            break

    assert last is not None and last["profile_built"], last
    return last


def attempt_login(
    client, age_challenge, session_factory, username=USERNAME, password=PASSWORD
):
    challenge = client.post("/auth/login/challenge", json={"username": username})
    assert challenge.status_code == 200, challenge.text
    issued = challenge.json()

    session = session_factory(issued["phrase"], issued["nonce"])
    simulate_elapsed_time(age_challenge, issued["nonce"], session)

    response = client.post(
        "/auth/login/behavior",
        json={
            "username": username,
            "password": password,
            "session": session.model_dump(),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def genuine(seed: int):
    def factory(phrase: str, nonce: str):
        return human_session(
            phrase, nonce=nonce, style=ALICE, seed=seed,
            username=USERNAME, password=PASSWORD,
        )
    return factory


def impostor(seed: int):
    def factory(phrase: str, nonce: str):
        return human_session(
            phrase, nonce=nonce, style=MALLORY, seed=seed,
            username=USERNAME, password=PASSWORD,
        )
    return factory


# ------------------------------------------------------------- registration


def test_register_then_status_reports_not_yet_enrolled(client):
    register(client)
    status = client.get(f"/auth/profile/status?username={USERNAME}").json()

    assert status["enrolled"] is False
    assert status["sessions_captured"] == 0


def test_registration_requires_consent(client):
    response = client.post(
        "/auth/register",
        json={"username": "bob", "password": "a-long-enough-password", "consent": False},
    )
    assert response.status_code == 400


def test_duplicate_username_is_rejected(client):
    register(client)
    response = client.post(
        "/auth/register",
        json={"username": USERNAME, "password": PASSWORD, "consent": True},
    )
    assert response.status_code == 409


def test_status_of_an_unknown_user_looks_like_an_unenrolled_one(client):
    """No enumeration: both answer the same way."""
    register(client)
    known = client.get(f"/auth/profile/status?username={USERNAME}").json()
    unknown = client.get("/auth/profile/status?username=nobodyhere").json()

    assert known["enrolled"] == unknown["enrolled"] is False
    assert known["sessions_captured"] == unknown["sessions_captured"]


# --------------------------------------------------------------- enrollment


def test_enrollment_builds_a_calibrated_profile(client, age_challenge):
    register(client)
    result = enroll(client, age_challenge)

    assert result["profile_built"] is True
    assert result["quality"]["features_modelled"] > 15

    status = client.get(f"/auth/profile/status?username={USERNAME}").json()
    assert status["enrolled"] is True
    # The exact threshold must NOT be here. This endpoint is unauthenticated,
    # and DecisionOut withholds the number precisely so an attacker cannot read
    # the bar they need to clear; serving it from an open GET would have given
    # it back. It previously did.
    assert "threshold" not in status
    assert status["maturity"] in {"COLD_START", "WARMING", "ESTABLISHED", "MATURE"}
    # The "+ml" suffix marks a threshold re-derived from the blended
    # statistical+ML score rather than the statistical score alone. Blending
    # without recalibrating would silently move the operating point.
    #
    # Which sources are legitimate depends on the enrollment size, and saying
    # so is the point of this assertion. The product enrolls in two captures,
    # which give a personal centre but no dispersion to calibrate against, so
    # the threshold is the static cold-start cut. Only the research baseline
    # has enough rounds for leave-one-out calibration.
    expected = (
        {"calibrated", "genuine_only", "fallback_prior"}
        if ENROLLMENT_ROUNDS >= MIN_FEATURE_SESSIONS
        else {"cold_start_prior"}
    )
    assert status["threshold_source"].removesuffix("+ml") in expected
    assert status["calibration_note"]


def test_enrollment_requires_the_correct_password(client):
    register(client)
    response = client.post(
        "/auth/enrollment/start",
        json={"username": USERNAME, "password": "not-the-password"},
    )
    # Otherwise anyone knowing a username could append their own behaviour to
    # that user's baseline until it accepted them.
    assert response.status_code == 401


def test_enrollment_rejects_an_automated_round(client):
    register(client)
    start = client.post(
        "/auth/enrollment/start", json={"username": USERNAME, "password": PASSWORD}
    ).json()

    session = scripted_session(
        start["phrase"], nonce=start["nonce"], username=USERNAME, password=PASSWORD
    )
    result = client.post(
        "/auth/enrollment/submit",
        json={"username": USERNAME, "session": session.model_dump()},
    ).json()

    assert result["accepted"] is False
    assert result["sessions_captured"] == 0
    assert "automated" in result["message"].lower()


def test_enrollment_rejects_a_round_that_ignores_the_phrase(client):
    register(client)
    start = client.post(
        "/auth/enrollment/start", json={"username": USERNAME, "password": PASSWORD}
    ).json()

    session = human_session(
        "completely different words that were never asked for here",
        nonce=start["nonce"], username=USERNAME, password=PASSWORD,
    )
    result = client.post(
        "/auth/enrollment/submit",
        json={"username": USERNAME, "session": session.model_dump()},
    ).json()

    assert result["accepted"] is False


def test_enrollment_cannot_be_extended_after_completion(client, age_challenge):
    register(client)
    enroll(client, age_challenge)

    response = client.post(
        "/auth/enrollment/start", json={"username": USERNAME, "password": PASSWORD}
    )
    assert response.status_code == 409


# ------------------------------------------------------- the four demo cases


def test_case_a_genuine_user_is_allowed(client, age_challenge):
    register(client)
    enroll(client, age_challenge)

    verdict = attempt_login(client, age_challenge, genuine(600))

    assert verdict["decision"] == "ALLOW", verdict
    assert verdict["reason"] == "BEHAVIOR_MATCH"
    assert verdict["session_token"]
    assert verdict["latency"]["total_ms"] > 0


def test_case_b_correct_password_wrong_behaviour_is_blocked(client, age_challenge, audit):
    """The central claim: the password is necessary but not sufficient."""
    register(client)
    enroll(client, age_challenge)

    verdict = attempt_login(client, age_challenge, impostor(700))

    assert verdict["decision"] == "BLOCK", verdict
    assert verdict["reason"] in {
        "BEHAVIORAL_MISMATCH", "KEYSTROKE_MISMATCH",
        "POINTER_MISMATCH", "INTERACTION_MISMATCH",
    }
    assert verdict["session_token"] is None
    assert verdict["signals"]
    # The response withholds the exact score on purpose; the audit trail keeps it.
    assert audit.last()["identity_score"] > 0.0


def test_case_c_scripted_attempt_is_blocked_as_automation(client, age_challenge, audit):
    """Reported as automation, not as a behavioural mismatch.

    They are different findings and the system distinguishes them.
    """
    register(client)
    enroll(client, age_challenge)

    verdict = attempt_login(
        client,
        age_challenge,
        lambda phrase, nonce: scripted_session(
            phrase, nonce=nonce, username=USERNAME, password=PASSWORD
        ),
    )

    assert verdict["decision"] == "BLOCK", verdict
    assert verdict["reason"] == "AUTOMATION_DETECTED"
    assert verdict["session_token"] is None
    assert audit.last()["automation_score"] > 0.5


def test_case_d_wrong_password_is_an_ordinary_credential_rejection(client, age_challenge, audit):
    register(client)
    enroll(client, age_challenge)

    verdict = attempt_login(client, age_challenge, genuine(601), password="wrong-password")

    assert verdict["decision"] == "BLOCK"
    assert verdict["reason"] == "INVALID_CREDENTIALS"
    assert verdict["session_token"] is None
    # Behavioural analysis never ran, so nothing was scored.
    recorded = audit.last()
    assert recorded["identity_score"] is None
    assert recorded["automation_score"] is None


def test_value_injection_attempt_is_blocked(client, age_challenge):
    register(client)
    enroll(client, age_challenge)

    verdict = attempt_login(
        client,
        age_challenge,
        lambda phrase, nonce: value_injection_session(
            phrase, nonce=nonce, username=USERNAME, password=PASSWORD
        ),
    )

    assert verdict["decision"] == "BLOCK"
    assert verdict["reason"] in {"AUTOMATION_DETECTED", "INSUFFICIENT_SIGNAL"}


def test_genuine_user_is_allowed_repeatedly(client, age_challenge):
    """Reliability across repeated attempts, asserted as a rate not a perfect run.

    This used to require all eight attempts to succeed, which is an assertion
    that the false rejection rate is exactly zero. It is not, and the system
    does not claim it is: a behavioural comparison is a probabilistic signal
    with overlapping genuine and impostor distributions, and each attempt draws
    a fresh random challenge phrase.

    Measured false rejection on synthetic typists is roughly 7 percent
    (evaluation/reliability_sweep.py, 8 enrollment rounds). The bound below is
    set well above that so ordinary variation never fails the suite, and well
    below a broken system so a real regression still does. It is a bound on the
    property actually claimed, not a weakened version of the old assertion.
    """
    register(client)
    enroll(client, age_challenge)

    attempts = 20
    verdicts = []
    for seed in range(610, 610 + attempts):
        # Measuring a rate needs more attempts than the rate limiter allows in
        # one window. Resetting here keeps this test about the behavioural
        # decision; the limiter has its own tests.
        limiter.reset()
        verdicts.append(attempt_login(client, age_challenge, genuine(seed)))
    allowed = [v for v in verdicts if v["decision"] == "ALLOW"]
    rejected = [v for v in verdicts if v["decision"] != "ALLOW"]

    # At a true 7 percent rate, seeing more than 25 percent of 20 attempts
    # rejected has probability well under one percent.
    assert len(allowed) >= int(attempts * 0.75), (
        f"false rejection rate {len(rejected) / attempts:.0%} exceeds the 25% bound; "
        f"rejections: {[v['reason'] for v in rejected]}"
    )
    # Every rejection must still be a behavioural decision, never a crash, an
    # integrity failure, or an automation misfire on a genuine human.
    for verdict in rejected:
        assert verdict["reason"] in {
            "BEHAVIORAL_MISMATCH", "KEYSTROKE_MISMATCH",
            "POINTER_MISMATCH", "INTERACTION_MISMATCH",
        }, verdict


def test_the_allow_path_is_reachable(client, age_challenge):
    """Wiring check: a genuine user can get in, and gets a session when they do.

    Takes a few attempts rather than one. An earlier version asserted that a
    single genuine login must succeed and called itself deterministic, which
    was wrong: the challenge phrase is drawn from the CSPRNG, so every genuine
    attempt carries the system's real false rejection rate of roughly 7
    percent. That test failed about one run in twenty, and the failure was
    correct behaviour being asserted away.

    What this test is actually for is catching the ALLOW path being broken
    outright. Three attempts make that near-certain to detect while leaving
    the rate itself to the test above, which measures it properly.
    """
    register(client)
    enroll(client, age_challenge)

    verdicts = []
    for seed in (900, 901, 902):
        limiter.reset()
        verdicts.append(attempt_login(client, age_challenge, genuine(seed)))

    allowed = [v for v in verdicts if v["decision"] == "ALLOW"]
    assert allowed, (
        "no genuine attempt was allowed in three tries; at the measured rate "
        f"that is a one-in-3000 coincidence: {[(v['reason']) for v in verdicts]}"
    )
    assert allowed[0]["session_token"], "an allowed login must issue a session"

    # A session token must never accompany a block.
    for verdict in verdicts:
        if verdict["decision"] == "BLOCK":
            assert verdict["session_token"] is None


def test_impostor_is_blocked_repeatedly(client, age_challenge):
    register(client)
    enroll(client, age_challenge)

    verdicts = [attempt_login(client, age_challenge, impostor(seed)) for seed in range(710, 718)]
    blocked = [v for v in verdicts if v["decision"] == "BLOCK"]

    assert len(blocked) == len(verdicts), [
        (v["decision"], v["reason"]) for v in verdicts
    ]


# ------------------------------------------------------- replay and integrity


def test_a_nonce_cannot_be_used_twice(client, age_challenge):
    register(client)
    enroll(client, age_challenge)

    issued = client.post("/auth/login/challenge", json={"username": USERNAME}).json()
    session = genuine(620)(issued["phrase"], issued["nonce"])
    simulate_elapsed_time(age_challenge, issued["nonce"], session)
    body = {
        "username": USERNAME,
        "password": PASSWORD,
        "session": session.model_dump(),
    }

    first = client.post("/auth/login/behavior", json=body).json()
    second = client.post("/auth/login/behavior", json=body).json()

    assert first["decision"] == "ALLOW"
    assert second["decision"] == "BLOCK"
    assert second["reason"] == "CHALLENGE_REUSED"


def test_replaying_a_capture_against_a_fresh_challenge_fails(client, age_challenge):
    """The case the randomised phrase exists to defeat.

    An attacker records a complete genuine login and replays the event stream
    against a newly issued nonce. The recording answers the wrong prompt.
    """
    register(client)
    enroll(client, age_challenge)

    old = client.post("/auth/login/challenge", json={"username": USERNAME}).json()
    recorded = genuine(630)(old["phrase"], old["nonce"])

    fresh = client.post("/auth/login/challenge", json={"username": USERNAME}).json()
    assert normalise_phrase(fresh["phrase"]) != normalise_phrase(old["phrase"])

    replayed = rebuild_session(recorded, events_as_dicts(recorded))
    body = {
        "username": USERNAME,
        "password": PASSWORD,
        "session": {**replayed.model_dump(), "nonce": fresh["nonce"]},
    }

    verdict = client.post("/auth/login/behavior", json=body).json()
    assert verdict["decision"] == "BLOCK"
    assert verdict["reason"] == "PHRASE_MISMATCH"


def test_an_unknown_nonce_is_rejected(client, age_challenge):
    register(client)
    enroll(client, age_challenge)

    session = genuine(640)("amber willow granite", "never-issued-nonce-000000000")
    verdict = client.post(
        "/auth/login/behavior",
        json={
            "username": USERNAME,
            "password": PASSWORD,
            "session": session.model_dump(),
        },
    ).json()

    assert verdict["decision"] == "BLOCK"
    assert verdict["reason"] == "CHALLENGE_UNKNOWN"


def test_a_challenge_issued_for_another_user_is_rejected(client, age_challenge):
    register(client)
    enroll(client, age_challenge)
    register(client, username="mallory", password="another-long-password")

    issued = client.post("/auth/login/challenge", json={"username": "mallory"}).json()
    session = genuine(650)(issued["phrase"], issued["nonce"])

    verdict = client.post(
        "/auth/login/behavior",
        json={
            "username": USERNAME,
            "password": PASSWORD,
            "session": session.model_dump(),
        },
    ).json()

    assert verdict["decision"] == "BLOCK"
    assert verdict["reason"] == "CHALLENGE_WRONG_USER"


def test_login_to_an_unenrolled_account_is_refused(client, age_challenge):
    register(client)
    verdict = attempt_login(client, age_challenge, genuine(660))

    assert verdict["decision"] == "BLOCK"
    assert verdict["reason"] == "ACCOUNT_NOT_ENROLLED"


def test_a_wrong_password_attempt_still_burns_the_nonce(client, age_challenge):
    """Otherwise a wrong-password probe becomes a free source of live challenges."""
    register(client)
    enroll(client, age_challenge)

    issued = client.post("/auth/login/challenge", json={"username": USERNAME}).json()
    session = genuine(670)(issued["phrase"], issued["nonce"])

    client.post(
        "/auth/login/behavior",
        json={"username": USERNAME, "password": "wrong", "session": session.model_dump()},
    )
    retry = client.post(
        "/auth/login/behavior",
        json={
            "username": USERNAME,
            "password": PASSWORD,
            "session": session.model_dump(),
        },
    ).json()

    assert retry["reason"] == "CHALLENGE_REUSED"


# ------------------------------------------------------------- what is stored


def test_no_raw_events_are_persisted(client, age_challenge, fresh_db_path):
    """The database must never accumulate keystroke streams."""
    register(client)
    enroll(client, age_challenge)
    attempt_login(client, age_challenge, genuine(680))

    import sqlite3

    conn = sqlite3.connect(fresh_db_path)
    try:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        for table in tables:
            columns = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
            assert "events" not in columns
            assert "raw_events" not in columns
    finally:
        conn.close()


def test_no_plaintext_password_reaches_the_database(client, age_challenge, fresh_db_path):
    register(client)
    enroll(client, age_challenge)
    attempt_login(client, age_challenge, genuine(690))

    blob = fresh_db_path.read_bytes()
    assert PASSWORD.encode() not in blob


def test_decisions_are_recorded_for_audit(client, age_challenge, fresh_db_path):
    register(client)
    enroll(client, age_challenge)
    # Several genuine attempts, because any single one can legitimately be
    # rejected. The audit trail only needs to show that both outcomes are
    # recorded, not that a particular attempt went a particular way.
    for seed in range(691, 695):
        attempt_login(client, age_challenge, genuine(seed))
    attempt_login(client, age_challenge, impostor(700))

    import sqlite3

    conn = sqlite3.connect(fresh_db_path)
    try:
        rows = conn.execute(
            "SELECT decision, reason FROM auth_attempts ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    decisions = [r[0] for r in rows]
    assert "ALLOW" in decisions
    assert "BLOCK" in decisions
