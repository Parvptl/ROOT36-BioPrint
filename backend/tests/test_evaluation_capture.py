"""Phase E capture seam: off by default, derived features only.

The seam exists so one real human session can be scored under more than one
population prior. It writes behavioural data, so the properties that matter are
privacy properties, and they are enforced here rather than left as intentions:

  * disabled by default, and a no-op when disabled
  * no password, no raw event, no username, no challenge text is ever written
  * participant ids are pseudonymous and not reversible without the secret
  * the genuine/impostor label comes from a declaration, never from the verdict
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app import evaluation_capture
from app.config import Settings, load_settings
from app.evaluation_capture import (
    FORBIDDEN_KEYS,
    LABEL_GENUINE,
    LABEL_IMPOSTOR,
    ROLE_ENROLLMENT,
    ROLE_LOGIN,
    EphemeralSecretError,
    assert_stable_identifiers,
    capture,
    is_enabled,
    pseudonym,
    validate_vector,
)

VECTOR = {
    "kbd_dwell_median": 104.0,
    "kbd_flight_median": 42.0,
    "kbd_speed_kps": 4.1,
    "ptr_velocity_median": 1.2,
}


@pytest.fixture
def eval_mode(tmp_path, monkeypatch):
    """Turn capture on, pointed at a throwaway database."""
    tuned = replace(
        evaluation_capture.settings,
        evaluation_mode=True,
        eval_db_path=tmp_path / "eval.db",
        secret_is_ephemeral=False,
        secret_key="a-fixed-test-secret",
    )
    monkeypatch.setattr(evaluation_capture, "settings", tuned)
    return tuned


def rows(db: Path) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM evaluation_sessions")]
    finally:
        conn.close()


# ------------------------------------------------------------ off by default


def test_evaluation_mode_is_off_in_the_shipped_configuration(monkeypatch):
    """The default must be off, and turning it on must be deliberate.

    Asserted against the dataclass default and an unset environment rather
    than against whatever the current process inherited — otherwise running
    the suite with BIOPRINT_EVALUATION_MODE=1 (which an operator legitimately
    does during collection) would fail this test for the wrong reason.
    """
    assert Settings.__dataclass_fields__["evaluation_mode"].default is False

    monkeypatch.delenv("BIOPRINT_EVALUATION_MODE", raising=False)
    assert load_settings().evaluation_mode is False

    monkeypatch.setenv("BIOPRINT_EVALUATION_MODE", "1")
    assert load_settings().evaluation_mode is True


def test_capture_is_a_no_op_when_disabled(tmp_path, monkeypatch):
    db = tmp_path / "eval.db"
    monkeypatch.setattr(
        evaluation_capture, "settings",
        replace(evaluation_capture.settings, evaluation_mode=False, eval_db_path=db),
    )

    assert is_enabled() is False
    result = capture(role=ROLE_LOGIN, claimed_username="alice",
                     features=VECTOR, coverage=1.0)

    assert result is None
    # Not one byte: the database is never even created.
    assert not db.exists()


# ------------------------------------------------------------------- privacy


def test_only_derived_features_are_written(eval_mode):
    evaluation_capture.set_actor("alice")
    capture(role=ROLE_LOGIN, claimed_username="alice", features=VECTOR, coverage=0.9)

    row = rows(eval_mode.eval_db_path)[0]

    assert set(row) == {
        "id", "session_id", "participant", "target", "role", "label",
        "enrollment_index", "features_json", "coverage", "created_at",
    }
    assert json.loads(row["features_json"]) == VECTOR


def test_no_password_username_or_raw_event_reaches_the_dataset(eval_mode):
    evaluation_capture.set_actor("alice")
    capture(role=ROLE_LOGIN, claimed_username="alice", features=VECTOR, coverage=0.9)

    blob = eval_mode.eval_db_path.read_bytes().lower()

    for secret in (b"alice", b"password", b"hunter2", b"keydown", b"pointermove"):
        assert secret not in blob, f"{secret!r} reached the evaluation dataset"


def test_a_vector_carrying_a_forbidden_key_is_refused(eval_mode):
    evaluation_capture.set_actor("alice")

    for key in ("password", "username", "phrase", "raw_events"):
        assert key in FORBIDDEN_KEYS
        result = capture(role=ROLE_LOGIN, claimed_username="alice",
                         features={**VECTOR, key: 1.0}, coverage=0.9)
        assert result is None

    assert rows(eval_mode.eval_db_path) == []


def test_participant_ids_are_pseudonymous_and_stable(eval_mode):
    first, again = pseudonym("alice"), pseudonym("alice")

    assert first == again
    assert first != pseudonym("bob")
    assert "alice" not in first
    assert first.startswith("E")


def test_collection_refuses_an_ephemeral_secret(eval_mode, monkeypatch):
    """Otherwise a restart silently splits one person into two identities."""
    monkeypatch.setattr(
        evaluation_capture, "settings",
        replace(eval_mode, secret_is_ephemeral=True),
    )

    with pytest.raises(EphemeralSecretError):
        assert_stable_identifiers()

    evaluation_capture.set_actor("alice")
    assert capture(role=ROLE_LOGIN, claimed_username="alice",
                   features=VECTOR, coverage=0.9) is None


# -------------------------------------------------------------- labelling


def test_the_label_comes_from_the_declared_actor_not_the_verdict(eval_mode):
    """Labelling by outcome would define an impostor as 'someone we rejected'."""
    evaluation_capture.set_actor("alice")
    capture(role=ROLE_LOGIN, claimed_username="alice", features=VECTOR, coverage=0.9)

    evaluation_capture.set_actor("bob")
    capture(role=ROLE_LOGIN, claimed_username="alice", features=VECTOR, coverage=0.9)

    captured = rows(eval_mode.eval_db_path)
    assert [r["label"] for r in captured] == [LABEL_GENUINE, LABEL_IMPOSTOR]
    # The impostor row is filed against the account that was attacked.
    assert captured[1]["participant"] == pseudonym("bob")
    assert captured[1]["target"] == pseudonym("alice")


def test_nothing_is_captured_without_a_declared_actor(eval_mode):
    evaluation_capture.set_actor(None)

    assert capture(role=ROLE_LOGIN, claimed_username="alice",
                   features=VECTOR, coverage=0.9) is None
    assert rows(eval_mode.eval_db_path) == []


def test_enrollment_by_a_non_owner_is_refused(eval_mode):
    """A third party's rounds must never become someone's baseline data."""
    evaluation_capture.set_actor("bob")

    assert capture(role=ROLE_ENROLLMENT, claimed_username="alice",
                   features=VECTOR, coverage=0.9, enrollment_index=0) is None
    assert rows(eval_mode.eval_db_path) == []


# ---------------------------------------------------------- data quality


def test_invalid_vectors_are_rejected_with_a_reason():
    assert validate_vector({}) == "empty_feature_vector"
    assert validate_vector({"a": float("nan")}).startswith("nan:")
    assert validate_vector({"a": float("inf")}).startswith("infinite:")
    assert validate_vector({"a": "x"}).startswith("non_numeric:")
    assert validate_vector({"password": 1.0}).startswith("forbidden_key:")
    assert validate_vector(VECTOR) is None


def test_a_rejected_capture_is_recorded_rather_than_dropped(eval_mode):
    evaluation_capture.set_actor("alice")
    capture(role=ROLE_LOGIN, claimed_username="alice",
            features={"kbd_dwell_median": float("nan")}, coverage=0.9)

    conn = sqlite3.connect(eval_mode.eval_db_path)
    try:
        rejects = conn.execute(
            "SELECT reason FROM evaluation_rejects"
        ).fetchall()
    finally:
        conn.close()

    assert [r[0] for r in rejects] == ["nan:kbd_dwell_median"]


def test_capture_never_raises_even_on_an_unwritable_database(eval_mode, monkeypatch, tmp_path):
    """An evaluation harness must not be able to break a login.

    The unwritable target is a path whose parent is an existing *file*, so
    mkdir cannot succeed. An earlier version of this test used an absent
    absolute path, which on Windows resolves onto the current drive and got
    cheerfully created — the test passed nothing and littered the filesystem.
    """
    # Declare the actor while the paths are still writable, then break only the
    # database target, so the failure under test is the write and nothing else.
    evaluation_capture.set_actor("alice")

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        evaluation_capture, "settings",
        replace(eval_mode, eval_db_path=blocker / "eval.db"),
    )

    assert capture(role=ROLE_LOGIN, claimed_username="alice",
                   features=VECTOR, coverage=0.9) is None
    assert blocker.is_file(), "the blocker must not have been replaced"
