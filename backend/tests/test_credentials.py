"""Credential layer: hashing, session issuance, session lifetime."""

from __future__ import annotations

import time

from app.auth.passwords import hash_password, needs_rehash, verify_password
from app.auth.sessions import issue_session, resolve_session, revoke_session


def test_hash_is_not_plaintext_and_is_salted():
    password = "correct horse battery staple"
    a = hash_password(password)
    b = hash_password(password)

    assert password not in a
    assert a.startswith("$argon2id$")
    # Distinct salts, so identical passwords do not produce identical hashes
    # and a leaked table cannot be attacked with a single precomputed guess.
    assert a != b


def test_verify_accepts_correct_and_rejects_wrong():
    encoded = hash_password("s3cret-passphrase")
    assert verify_password("s3cret-passphrase", encoded) is True
    assert verify_password("s3cret-passphras", encoded) is False
    assert verify_password("", encoded) is False


def test_verify_returns_false_on_corrupt_hash_instead_of_raising():
    # A damaged column must not turn into a 500 that reveals internals.
    assert verify_password("anything", "not-a-real-argon2-hash") is False
    assert needs_rehash("not-a-real-argon2-hash") is True


def test_current_parameters_do_not_request_rehash():
    assert needs_rehash(hash_password("abcdefgh")) is False


def _make_user(db) -> int:
    cur = db.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        ("tester", hash_password("abcdefgh"), time.time()),
    )
    return int(cur.lastrowid)


def test_session_round_trip(db):
    user_id = _make_user(db)
    token, expires_at = issue_session(db, user_id)

    assert expires_at > time.time()
    assert resolve_session(db, token) == user_id


def test_raw_token_is_never_stored(db):
    user_id = _make_user(db)
    token, _ = issue_session(db, user_id)

    stored = db.execute("SELECT token_hash FROM sessions").fetchone()["token_hash"]
    assert stored != token
    assert len(stored) == 64  # sha256 hex


def test_unknown_and_revoked_tokens_resolve_to_none(db):
    user_id = _make_user(db)
    token, _ = issue_session(db, user_id)

    assert resolve_session(db, "made-up-token") is None

    revoke_session(db, token)
    assert resolve_session(db, token) is None


def test_expired_session_is_rejected_and_cleaned_up(db):
    user_id = _make_user(db)
    token, _ = issue_session(db, user_id)

    db.execute("UPDATE sessions SET expires_at = ?", (time.time() - 1,))

    assert resolve_session(db, token) is None
    assert db.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"] == 0
