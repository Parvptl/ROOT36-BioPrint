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
    assert status["threshold"] is not None
    assert status["threshold_source"] in {"calibrated", "genuine_only", "fallback_prior"}
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


def test_case_b_correct_password_wrong_behaviour_is_blocked(client, age_challenge):
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
    assert verdict["identity_score"] > verdict["threshold"]
    assert verdict["signals"]


def test_case_c_scripted_attempt_is_blocked_as_automation(client, age_challenge):
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
    assert verdict["automation_score"] > 0.5
    assert verdict["session_token"] is None


def test_case_d_wrong_password_is_an_ordinary_credential_rejection(client, age_challenge):
    register(client)
    enroll(client, age_challenge)

    verdict = attempt_login(client, age_challenge, genuine(601), password="wrong-password")

    assert verdict["decision"] == "BLOCK"
    assert verdict["reason"] == "INVALID_CREDENTIALS"
    assert verdict["session_token"] is None
    # Behavioural analysis never ran, so there is nothing to report about it.
    assert verdict["identity_score"] is None


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
    """Reliability, not just a single lucky attempt."""
    register(client)
    enroll(client, age_challenge)

    verdicts = [attempt_login(client, age_challenge, genuine(seed)) for seed in range(610, 618)]
    allowed = [v for v in verdicts if v["decision"] == "ALLOW"]

    assert len(allowed) == len(verdicts), [
        (v["decision"], v["reason"], v["identity_score"]) for v in verdicts
    ]


def test_impostor_is_blocked_repeatedly(client, age_challenge):
    register(client)
    enroll(client, age_challenge)

    verdicts = [attempt_login(client, age_challenge, impostor(seed)) for seed in range(710, 718)]
    blocked = [v for v in verdicts if v["decision"] == "BLOCK"]

    assert len(blocked) == len(verdicts), [
        (v["decision"], v["reason"], v["identity_score"]) for v in verdicts
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
    attempt_login(client, age_challenge, genuine(691))
    attempt_login(client, age_challenge, impostor(692))

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
