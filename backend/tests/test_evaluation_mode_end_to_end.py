"""The capture seam, exercised through the real HTTP API.

Unit tests can show the capture function behaves. Only driving the actual
enrollment and login routes can show that it is wired in at the right place,
that it does not change any response, and that with the flag off the product
writes nothing extra at all.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app import evaluation_capture
from app.api.ratelimit import limiter
from app.api.routes_auth import ENROLLMENT_ROUNDS
from app.db.database import init_db, set_db_path_override
from app.evaluation_capture import LABEL_GENUINE, LABEL_IMPOSTOR, pseudonym
from tests.factories import TypingStyle, human_session

OWNER, OWNER_PW = "ownerperson", "owner-password-12345"
OTHER, OTHER_PW = "otherperson", "other-password-12345"

OWNER_STYLE = TypingStyle(iki_mean_ms=135.0, dwell_mean_ms=80.0, overlap_prob=0.45)
OTHER_STYLE = TypingStyle(iki_mean_ms=280.0, dwell_mean_ms=130.0, overlap_prob=0.02)


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A throwaway server with its own database."""
    db = tmp_path / "prod.db"
    set_db_path_override(db)
    init_db(db)
    # The limiter is process-global. Without this reset an earlier test's
    # traffic exhausts the enrollment quota and this one fails on a challenge
    # that was never issued — which looks nothing like a rate limit.
    limiter.reset()

    from app.main import app

    client = TestClient(app)
    yield client, db
    limiter.reset()
    set_db_path_override(None)


def enable_capture(tmp_path, monkeypatch):
    tuned = replace(
        evaluation_capture.settings,
        evaluation_mode=True,
        eval_db_path=tmp_path / "eval" / "eval.db",
        secret_is_ephemeral=False,
    )
    monkeypatch.setattr(evaluation_capture, "settings", tuned)
    return tuned


def _age(db: Path, nonce: str, seconds: float) -> None:
    """Synthetic captures are built instantly; the challenge must look older."""
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE auth_challenges SET issued_at = issued_at - ?, "
            "expires_at = expires_at - ? WHERE nonce = ?",
            (seconds, seconds, nonce),
        )
        conn.commit()
    finally:
        conn.close()


def register(client, username, password):
    return client.post("/auth/register",
                       json={"username": username, "password": password, "consent": True})


def enroll(client, db, username, password, style, rounds=None, seed=0):
    # Follow the product's enrollment size rather than pinning a number:
    # a test that hardcodes 8 silently breaks when the product changes,
    # and the breakage looks like a missing challenge rather than a
    # stale expectation.
    rounds = ENROLLMENT_ROUNDS if rounds is None else rounds
    for i in range(rounds):
        started = client.post("/auth/enrollment/start",
                              json={"username": username, "password": password}).json()
        session = human_session(started["phrase"], nonce=started["nonce"],
                                username=username, password=password,
                                style=style, seed=seed + i)
        _age(db, started["nonce"], max(e.t for e in session.events) / 1000.0 + 1.0)
        client.post("/auth/enrollment/submit",
                    json={"username": username, "session": session.model_dump()})


def login(client, db, claimed, password, style, seed):
    issued = client.post("/auth/login/challenge", json={"username": claimed}).json()
    session = human_session(issued["phrase"], nonce=issued["nonce"],
                            username=claimed, password=password,
                            style=style, seed=seed)
    _age(db, issued["nonce"], max(e.t for e in session.events) / 1000.0 + 1.0)
    return client.post("/auth/login/behavior",
                       json={"username": claimed, "password": password,
                             "session": session.model_dump()})


def eval_rows(db: Path) -> list[dict]:
    if not db.exists():
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM evaluation_sessions ORDER BY id")]
    finally:
        conn.close()


# -------------------------------------------------- production is unchanged


def test_with_evaluation_mode_off_nothing_extra_is_written(api, tmp_path, monkeypatch):
    client, db = api
    eval_db = tmp_path / "eval" / "eval.db"
    monkeypatch.setattr(
        evaluation_capture, "settings",
        replace(evaluation_capture.settings, evaluation_mode=False, eval_db_path=eval_db),
    )

    register(client, OWNER, OWNER_PW)
    enroll(client, db, OWNER, OWNER_PW, OWNER_STYLE)
    response = login(client, db, OWNER, OWNER_PW, OWNER_STYLE, seed=500)

    assert response.status_code == 200
    assert response.json()["decision"] in ("ALLOW", "BLOCK")
    assert not eval_db.exists(), "evaluation data was written with the flag off"


def test_the_login_response_is_identical_with_capture_on(api, tmp_path, monkeypatch):
    """Capture must be observationally invisible to a caller."""
    client, db = api
    register(client, OWNER, OWNER_PW)
    enroll(client, db, OWNER, OWNER_PW, OWNER_STYLE)

    monkeypatch.setattr(
        evaluation_capture, "settings",
        replace(evaluation_capture.settings, evaluation_mode=False),
    )
    off = login(client, db, OWNER, OWNER_PW, OWNER_STYLE, seed=900).json()

    enable_capture(tmp_path, monkeypatch)
    evaluation_capture.set_actor(OWNER)
    on = login(client, db, OWNER, OWNER_PW, OWNER_STYLE, seed=900).json()

    # The two logins answer DIFFERENT randomly generated challenge phrases, so
    # the per-feature explanation shares legitimately differ by a percent or
    # two. What must not differ is the decision contract.
    contract = ("decision", "reason", "headline", "integrity_status", "coverage_band")
    assert {k: off[k] for k in contract} == {k: on[k] for k in contract}
    assert off.keys() == on.keys()

    # The signal ORDER is by descending modality score and legitimately moves
    # between two different phrases, so it is not asserted. The property that
    # matters is structural and stronger: capture runs after `decide()` has
    # already produced the verdict, and it must not touch the vector it is
    # handed.
    vector = {"kbd_dwell_median": 104.0, "kbd_speed_kps": 4.1}
    before = dict(vector)
    evaluation_capture.capture(role="login", claimed_username=OWNER,
                               features=vector, coverage=0.9)
    assert vector == before


# ------------------------------------------------------- capture works


def test_enrollment_and_logins_are_captured_with_correct_labels(api, tmp_path, monkeypatch):
    client, db = api
    tuned = enable_capture(tmp_path, monkeypatch)

    register(client, OWNER, OWNER_PW)
    register(client, OTHER, OTHER_PW)

    evaluation_capture.set_actor(OWNER)
    enroll(client, db, OWNER, OWNER_PW, OWNER_STYLE)
    login(client, db, OWNER, OWNER_PW, OWNER_STYLE, seed=700)

    # The other person now types at the owner's account with the owner's
    # credential: a human impostor attempt.
    evaluation_capture.set_actor(OTHER)
    login(client, db, OWNER, OWNER_PW, OTHER_STYLE, seed=800)

    rows = eval_rows(tuned.eval_db_path)
    enrollment = [r for r in rows if r["role"] == "enrollment"]
    logins = [r for r in rows if r["role"] == "login"]

    assert len(enrollment) == ENROLLMENT_ROUNDS
    assert [r["enrollment_index"] for r in enrollment] == list(range(ENROLLMENT_ROUNDS))
    assert all(r["label"] == LABEL_GENUINE for r in enrollment)
    assert all(r["participant"] == pseudonym(OWNER) for r in enrollment)

    assert len(logins) == 2
    assert logins[0]["label"] == LABEL_GENUINE
    assert logins[1]["label"] == LABEL_IMPOSTOR
    assert logins[1]["participant"] == pseudonym(OTHER)
    assert logins[1]["target"] == pseudonym(OWNER)


def test_captured_vectors_carry_only_registry_features(api, tmp_path, monkeypatch):
    from app.behavioral.features.registry import enabled_names

    client, db = api
    tuned = enable_capture(tmp_path, monkeypatch)
    register(client, OWNER, OWNER_PW)
    evaluation_capture.set_actor(OWNER)
    enroll(client, db, OWNER, OWNER_PW, OWNER_STYLE, rounds=1)

    rows = eval_rows(tuned.eval_db_path)
    assert rows

    allowed = set(enabled_names())
    for row in rows:
        features = json.loads(row["features_json"])
        assert set(features) <= allowed
        assert len(features) <= 27
        assert all(isinstance(v, (int, float)) for v in features.values())


def test_the_evaluation_database_is_separate_from_the_product_database(api, tmp_path, monkeypatch):
    """So privacy verification is a check on one file, not an audit of two."""
    client, db = api
    tuned = enable_capture(tmp_path, monkeypatch)
    register(client, OWNER, OWNER_PW)
    evaluation_capture.set_actor(OWNER)
    enroll(client, db, OWNER, OWNER_PW, OWNER_STYLE, rounds=1)

    assert tuned.eval_db_path != db
    assert tuned.eval_db_path.exists()

    conn = sqlite3.connect(db)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()

    assert "evaluation_sessions" not in tables
    assert "evaluation_rejects" not in tables
