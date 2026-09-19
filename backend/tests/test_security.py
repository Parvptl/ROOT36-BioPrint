"""Security test suite.

Verifies the properties an attacker would target:
  - replay resistance (reused nonce, expired challenge)
  - client cannot submit scores, features, or decisions
  - malformed and impossible event streams are rejected
  - rate limiting prevents brute force
  - enrollment abuse is prevented
  - no password material in behavioural data
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.api.ratelimit import limiter
from app.auth.challenge import normalise_phrase
from tests.factories import (
    TypingStyle,
    human_session,
    scripted_session,
    value_injection_session,
)

USERNAME = "sectest"
PASSWORD = "secure-enough-password"

ALICE = TypingStyle(
    iki_mean_ms=140.0, dwell_mean_ms=82.0, overlap_prob=0.45,
    right_shift_prob=0.95, tab_between_fields=True, pointer_speed=1.6,
)
# A clearly different typist, for tests that need a reliable BLOCK rather than
# a realistic impostor. Reliability rates are measured in the evaluation sweep,
# not here.
MALLORY = TypingStyle(
    iki_mean_ms=320.0, dwell_mean_ms=140.0, overlap_prob=0.01,
    right_shift_prob=0.02, tab_between_fields=False, pointer_speed=0.6,
    backspace_prob=0.14, pause_prob=0.20,
)


@pytest.fixture(autouse=True)
def _clear_rate_limits():
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
    duration_seconds = max(e.t for e in session.events) / 1000.0
    age_challenge(nonce, duration_seconds + 1.0)


def enroll(client, age_challenge, style=ALICE, username=USERNAME, password=PASSWORD):
    last = None
    for index in range(20):
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


def get_challenge(client, username=USERNAME):
    response = client.post("/auth/login/challenge", json={"username": username})
    assert response.status_code == 200
    return response.json()


def attempt_login(client, age_challenge, session_factory, username=USERNAME, password=PASSWORD):
    issued = get_challenge(client, username)
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
    return response


# ===================================================================
# REPLAY RESISTANCE
# ===================================================================


def test_reused_nonce_is_rejected(client, age_challenge):
    """A nonce cannot be submitted twice — replay attack defeated."""
    register(client)
    enroll(client, age_challenge)

    issued = get_challenge(client)
    session = human_session(
        issued["phrase"], nonce=issued["nonce"], style=ALICE, seed=800,
        username=USERNAME, password=PASSWORD,
    )
    simulate_elapsed_time(age_challenge, issued["nonce"], session)

    payload = {
        "username": USERNAME,
        "password": PASSWORD,
        "session": session.model_dump(),
    }

    # First submission
    r1 = client.post("/auth/login/behavior", json=payload)
    assert r1.status_code == 200

    # Replay: same nonce, same payload
    r2 = client.post("/auth/login/behavior", json=payload)
    assert r2.status_code == 200
    body = r2.json()
    assert body["decision"] == "BLOCK"
    assert "reused" in body["reason"].lower() or "challenge" in body["reason"].lower()


def test_expired_challenge_is_rejected(client, age_challenge, fresh_db_path):
    """An expired challenge cannot be used to authenticate."""
    register(client)
    enroll(client, age_challenge)

    issued = get_challenge(client)
    session = human_session(
        issued["phrase"], nonce=issued["nonce"], style=ALICE, seed=801,
        username=USERNAME, password=PASSWORD,
    )

    # Expire the challenge by advancing its expiry far into the past.
    import sqlite3
    conn = sqlite3.connect(fresh_db_path)
    try:
        conn.execute(
            "UPDATE auth_challenges SET expires_at = issued_at - 10 WHERE nonce = ?",
            (issued["nonce"],),
        )
        conn.commit()
    finally:
        conn.close()

    r = client.post(
        "/auth/login/behavior",
        json={
            "username": USERNAME,
            "password": PASSWORD,
            "session": session.model_dump(),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "BLOCK"
    assert "expired" in body["reason"].lower() or "challenge" in body["reason"].lower()


def test_wrong_user_challenge_is_rejected(client, age_challenge):
    """A challenge issued for one user cannot be used by another."""
    register(client, "user_a")
    register(client, "user_b")
    enroll(client, age_challenge, username="user_a", password=PASSWORD)
    enroll(client, age_challenge, username="user_b", password=PASSWORD)

    # Get a challenge for user_a
    issued = client.post("/auth/login/challenge", json={"username": "user_a"}).json()
    session = human_session(
        issued["phrase"], nonce=issued["nonce"], style=ALICE, seed=802,
        username="user_b", password=PASSWORD,
    )
    simulate_elapsed_time(age_challenge, issued["nonce"], session)

    # Submit against user_b
    r = client.post(
        "/auth/login/behavior",
        json={
            "username": "user_b",
            "password": PASSWORD,
            "session": session.model_dump(),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "BLOCK"


# ===================================================================
# CLIENT CANNOT SUBMIT SCORES OR DECISIONS
# ===================================================================


def test_extra_fields_in_login_rejected(client, age_challenge):
    """The client cannot inject a score or decision via extra fields."""
    register(client)
    enroll(client, age_challenge)

    issued = get_challenge(client)
    session = human_session(
        issued["phrase"], nonce=issued["nonce"], style=ALICE, seed=810,
        username=USERNAME, password=PASSWORD,
    )
    simulate_elapsed_time(age_challenge, issued["nonce"], session)

    payload = {
        "username": USERNAME,
        "password": PASSWORD,
        "session": session.model_dump(),
        "decision": "ALLOW",  # attacker tries to inject
        "identity_score": 0.0,
    }

    r = client.post("/auth/login/behavior", json=payload)
    # extra="forbid" should reject this with 422
    assert r.status_code == 422


def test_extra_fields_in_session_rejected(client, age_challenge):
    """Extra fields inside the behavioral session are rejected."""
    register(client)
    enroll(client, age_challenge)

    issued = get_challenge(client)
    session = human_session(
        issued["phrase"], nonce=issued["nonce"], style=ALICE, seed=811,
        username=USERNAME, password=PASSWORD,
    )
    session_dict = session.model_dump()
    session_dict["score"] = 0.0  # attacker tries to inject a pre-computed score
    session_dict["decision"] = "ALLOW"

    r = client.post(
        "/auth/login/behavior",
        json={
            "username": USERNAME,
            "password": PASSWORD,
            "session": session_dict,
        },
    )
    assert r.status_code == 422


def test_extra_fields_in_register_rejected(client):
    """Extra fields in registration are rejected."""
    r = client.post(
        "/auth/register",
        json={
            "username": USERNAME,
            "password": PASSWORD,
            "consent": True,
            "role": "admin",
        },
    )
    assert r.status_code == 422


# ===================================================================
# MALFORMED EVENT STREAMS
# ===================================================================


def test_empty_events_rejected(client, age_challenge):
    """An event stream with zero events is rejected at the schema level."""
    register(client)
    enroll(client, age_challenge)

    issued = get_challenge(client)

    r = client.post(
        "/auth/login/behavior",
        json={
            "username": USERNAME,
            "password": PASSWORD,
            "session": {
                "nonce": issued["nonce"],
                "phrase_typed": issued["phrase"],
                "started_at_ms": 0.0,
                "events": [],
            },
        },
    )
    assert r.status_code == 422


def test_impossible_timestamps_rejected(client, age_challenge):
    """Events with decreasing timestamps are rejected."""
    register(client)
    enroll(client, age_challenge)

    issued = get_challenge(client)

    r = client.post(
        "/auth/login/behavior",
        json={
            "username": USERNAME,
            "password": PASSWORD,
            "session": {
                "nonce": issued["nonce"],
                "phrase_typed": issued["phrase"],
                "started_at_ms": 0.0,
                "events": [
                    {"type": "keydown", "t": 100.0, "trusted": True, "ctx": "username", "key_class": "char"},
                    {"type": "keydown", "t": 50.0, "trusted": True, "ctx": "username", "key_class": "char"},
                ],
            },
        },
    )
    assert r.status_code == 422


def test_password_code_in_events_rejected(client, age_challenge):
    """Password-context key events must not carry the key code."""
    register(client)
    enroll(client, age_challenge)

    issued = get_challenge(client)

    r = client.post(
        "/auth/login/behavior",
        json={
            "username": USERNAME,
            "password": PASSWORD,
            "session": {
                "nonce": issued["nonce"],
                "phrase_typed": issued["phrase"],
                "started_at_ms": 0.0,
                "events": [
                    {
                        "type": "keydown", "t": 100.0, "trusted": True,
                        "ctx": "password", "key_class": "char",
                        "code": "KeyA",  # leaking password character
                    },
                ],
            },
        },
    )
    assert r.status_code == 422


def test_unknown_event_type_rejected(client, age_challenge):
    """An unknown event type should be rejected."""
    register(client)
    enroll(client, age_challenge)

    issued = get_challenge(client)

    r = client.post(
        "/auth/login/behavior",
        json={
            "username": USERNAME,
            "password": PASSWORD,
            "session": {
                "nonce": issued["nonce"],
                "phrase_typed": issued["phrase"],
                "started_at_ms": 0.0,
                "events": [
                    {"type": "custom_evil", "t": 100.0, "trusted": True},
                ],
            },
        },
    )
    assert r.status_code == 422


def test_too_many_events_rejected(client, age_challenge):
    """An event stream exceeding MAX_EVENTS must be rejected."""
    register(client)
    enroll(client, age_challenge)

    issued = get_challenge(client)

    # Create 20001 minimal events
    events = [
        {"type": "keydown", "t": float(i), "trusted": True, "ctx": "username", "key_class": "char"}
        for i in range(20_001)
    ]

    r = client.post(
        "/auth/login/behavior",
        json={
            "username": USERNAME,
            "password": PASSWORD,
            "session": {
                "nonce": issued["nonce"],
                "phrase_typed": issued["phrase"],
                "started_at_ms": 0.0,
                "events": events,
            },
        },
    )
    assert r.status_code == 422


# ===================================================================
# CREDENTIAL SECURITY
# ===================================================================


def test_wrong_password_blocks_before_behavioral_analysis(client, age_challenge):
    """Wrong password returns BLOCK with credential reason, not behavioral."""
    register(client)
    enroll(client, age_challenge)

    r = attempt_login(
        client, age_challenge,
        lambda phrase, nonce: human_session(
            phrase, nonce=nonce, style=ALICE, seed=820,
            username=USERNAME, password=PASSWORD,
        ),
        password="wrong-password-here",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "BLOCK"
    assert "credential" in body["reason"].lower()
    # No identity or automation score should be computed for wrong password
    # Scores are no longer returned to the client at all; see DecisionOut.
    assert "identity_score" not in body
    assert "automation_score" not in body


def test_nonexistent_user_gets_block(client, age_challenge):
    """Login for a user that doesn't exist returns BLOCK without leaking info."""
    issued = client.post("/auth/login/challenge", json={"username": "noone"}).json()
    session = human_session(
        issued["phrase"], nonce=issued["nonce"], style=ALICE, seed=821,
        username="noone", password="any-password-here",
    )
    simulate_elapsed_time(age_challenge, issued["nonce"], session)

    r = client.post(
        "/auth/login/behavior",
        json={
            "username": "noone",
            "password": "any-password-here",
            "session": session.model_dump(),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "BLOCK"


# ===================================================================
# RATE LIMITING
# ===================================================================


def test_login_rate_limit_triggers(client, age_challenge):
    """Repeated challenge requests are rate-limited."""
    register(client)

    # The challenge limit is 30 per 60 seconds. Hit it.
    for i in range(30):
        r = client.post("/auth/login/challenge", json={"username": USERNAME})
        assert r.status_code == 200, f"Request {i+1} failed unexpectedly"

    # The 31st should be rate-limited
    r = client.post("/auth/login/challenge", json={"username": USERNAME})
    assert r.status_code == 429


def test_registration_rate_limit_triggers(client):
    """Repeated registrations are rate-limited."""
    for i in range(5):
        r = client.post(
            "/auth/register",
            json={"username": f"ratelimit{i}", "password": PASSWORD, "consent": True},
        )
        assert r.status_code in (201, 409), f"Request {i+1}: {r.status_code}"

    r = client.post(
        "/auth/register",
        json={"username": "ratelimit99", "password": PASSWORD, "consent": True},
    )
    assert r.status_code == 429


# ===================================================================
# ENROLLMENT SECURITY
# ===================================================================


def test_enrollment_requires_correct_password(client):
    """Cannot start enrollment with wrong password."""
    register(client)
    r = client.post(
        "/auth/enrollment/start",
        json={"username": USERNAME, "password": "wrong-password"},
    )
    assert r.status_code == 401


def test_cannot_re_enroll_after_profile_built(client, age_challenge):
    """Once a profile exists, enrollment start returns 409."""
    register(client)
    enroll(client, age_challenge)

    r = client.post(
        "/auth/enrollment/start",
        json={"username": USERNAME, "password": PASSWORD},
    )
    assert r.status_code == 409


def test_enrollment_rejects_automated_round(client, age_challenge):
    """Scripted input during enrollment is detected and rejected."""
    register(client)

    start = client.post(
        "/auth/enrollment/start",
        json={"username": USERNAME, "password": PASSWORD},
    ).json()

    session = scripted_session(
        start["phrase"], nonce=start["nonce"],
        username=USERNAME, password=PASSWORD,
    )

    r = client.post(
        "/auth/enrollment/submit",
        json={"username": USERNAME, "session": session.model_dump()},
    )
    body = r.json()
    assert body["accepted"] is False
    assert "automated" in body["message"].lower()


# ===================================================================
# DATABASE: NO PASSWORD MATERIAL
# ===================================================================


def test_no_password_in_database(client, age_challenge, fresh_db_path):
    """The raw password never appears in the database file."""
    register(client)
    enroll(client, age_challenge)

    def genuine_factory(phrase, nonce):
        return human_session(
            phrase, nonce=nonce, style=ALICE, seed=850,
            username=USERNAME, password=PASSWORD,
        )

    attempt_login(client, age_challenge, genuine_factory)

    blob = fresh_db_path.read_bytes()
    assert PASSWORD.encode() not in blob


# ===================================================================
# AUTOMATION: KNOWN ATTACKS
# ===================================================================


def test_scripted_input_blocked_at_login(client, age_challenge):
    """A timer-based scripted login is caught by automation detection."""
    register(client)
    enroll(client, age_challenge)

    r = attempt_login(
        client, age_challenge,
        lambda phrase, nonce: scripted_session(
            phrase, nonce=nonce, username=USERNAME, password=PASSWORD,
        ),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "BLOCK"
    # The client is told the category, not the number.
    assert body["headline"] == "AUTOMATION DETECTED"


def test_value_injection_blocked_at_login(client, age_challenge):
    """Direct value assignment (element.value = ...) is caught."""
    register(client)
    enroll(client, age_challenge)

    r = attempt_login(
        client, age_challenge,
        lambda phrase, nonce: value_injection_session(
            phrase, nonce=nonce, username=USERNAME, password=PASSWORD,
        ),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "BLOCK"


# --------------------------------------------------- decision is not an oracle


# Fields that would let an attacker holding a correct password hill-climb
# toward acceptance by reading their exact distance from the bar.
ORACLE_FIELDS = ("identity_score", "automation_score", "threshold", "modality_scores")


def test_login_response_leaks_no_score_or_threshold(client, age_challenge):
    """The subject of a decision must not receive the gradient.

    An earlier version returned the exact identity score, automation score and
    threshold on every attempt. That turns the endpoint into a tuning oracle:
    an attacker who already has the password can vary their behaviour and read
    off whether they are getting warmer.
    """
    register(client)
    enroll(client, age_challenge)

    for factory in (
        lambda p, n: human_session(p, nonce=n, style=ALICE, seed=880,
                                   username=USERNAME, password=PASSWORD),
        lambda p, n: human_session(p, nonce=n, style=MALLORY, seed=881,
                                   username=USERNAME, password=PASSWORD),
        lambda p, n: scripted_session(p, nonce=n, username=USERNAME, password=PASSWORD),
    ):
        body = attempt_login(client, age_challenge, factory).json()
        for field in ORACLE_FIELDS:
            assert field not in body, f"{field} leaked in {body['reason']} response"


def test_blocked_response_still_explains_which_category_disagreed(client, age_challenge):
    """Withholding the number must not mean withholding the reason.

    Explainability is a product requirement; the gradient is not.
    """
    register(client)
    enroll(client, age_challenge)

    body = attempt_login(
        client, age_challenge,
        lambda p, n: human_session(p, nonce=n, style=MALLORY, seed=882,
                                   username=USERNAME, password=PASSWORD),
    ).json()

    assert body["decision"] == "BLOCK"
    assert body["headline"]
    assert body["message"]
    assert body["signals"], "a blocked user must still be told which category disagreed"
    for signal in body["signals"]:
        assert signal["band"] in {"LOW", "MEDIUM", "HIGH"}


def test_exact_scores_are_still_recorded_for_the_operator(client, age_challenge, audit):
    """Withheld from the client, not discarded."""
    register(client)
    enroll(client, age_challenge)
    attempt_login(
        client, age_challenge,
        lambda p, n: human_session(p, nonce=n, style=MALLORY, seed=883,
                                   username=USERNAME, password=PASSWORD),
    )

    recorded = audit.last()
    assert recorded["identity_score"] is not None
    assert recorded["automation_score"] is not None


# ------------------------------------------------------- operator dashboard


def test_dashboard_is_disabled_when_no_operator_key_is_configured(client):
    """Closed by default. It serves exactly what the login response withholds."""
    assert client.get("/security/dashboard").status_code == 404


def test_dashboard_rejects_a_missing_or_wrong_operator_key(client, with_operator_key):
    assert client.get("/security/dashboard").status_code == 401
    assert client.get(
        "/security/dashboard", headers={"X-Operator-Key": "wrong"}
    ).status_code == 401


def test_dashboard_serves_exact_scores_with_the_operator_key(
    client, age_challenge, with_operator_key
):
    register(client)
    enroll(client, age_challenge)
    attempt_login(
        client, age_challenge,
        lambda p, n: human_session(p, nonce=n, style=MALLORY, seed=884,
                                   username=USERNAME, password=PASSWORD),
    )

    response = client.get(
        "/security/dashboard", headers={"X-Operator-Key": with_operator_key}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["latest"]["identity_score"] is not None
    assert body["attempts"]


# ------------------------------------------------- bundled frontend serving


def test_static_fallback_cannot_escape_the_bundle(client, tmp_path, monkeypatch):
    """Path traversal through the SPA fallback must not serve backend files.

    The fallback resolves an arbitrary user-supplied path against the bundle
    directory. Without resolving and checking containment, a request for
    ../../backend/.env would hand over the signing secret.
    """
    import app.main as main

    bundle = tmp_path / "dist"
    (bundle / "assets").mkdir(parents=True)
    (bundle / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    secret = tmp_path / "secret.env"
    secret.write_text("BIOPRINT_SECRET_KEY=leaked", encoding="utf-8")

    monkeypatch.setattr(main, "FRONTEND_DIST", bundle)

    for attempt in ("../secret.env", "../../secret.env", "./../secret.env"):
        response = main.serve_frontend(attempt)
        served = Path(response.path).read_text(encoding="utf-8")
        assert "leaked" not in served, f"traversal succeeded via {attempt}"
        assert served == "<html>spa</html>"


def test_static_fallback_serves_index_for_client_routes(client, tmp_path, monkeypatch):
    import app.main as main

    bundle = tmp_path / "dist"
    bundle.mkdir()
    (bundle / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    monkeypatch.setattr(main, "FRONTEND_DIST", bundle)

    for route in ("login", "enroll", "security", ""):
        response = main.serve_frontend(route)
        assert Path(response.path).name == "index.html"


def test_api_routes_are_not_shadowed_by_the_static_catch_all():
    """The catch-all must be registered last or it swallows the whole API."""
    from app.main import app

    paths = [getattr(route, "path", "") for route in app.routes]
    catch_all = [i for i, p in enumerate(paths) if "{requested_path" in p]

    if catch_all:  # only mounted when a build exists
        assert catch_all[0] == len(paths) - 1, "catch-all is not the last route"
