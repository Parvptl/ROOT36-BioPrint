"""Challenge/nonce system: single use, expiry, ownership, unpredictability."""

from __future__ import annotations

import time

from app.auth.challenge import (
    PHRASE_WORD_COUNT,
    consume_challenge,
    create_challenge,
    generate_phrase,
    normalise_phrase,
    purge_expired_challenges,
)
from app.behavioral.scoring.reasons import ReasonCode


def test_phrases_are_fresh_each_time():
    phrases = {generate_phrase() for _ in range(200)}
    # A duplicate in 200 draws would mean the generator is not doing its job;
    # a repeated phrase is exactly what makes a recorded session replayable.
    assert len(phrases) > 195
    assert all(len(p.split()) == PHRASE_WORD_COUNT for p in phrases)


def test_phrase_is_plain_lowercase_words():
    for _ in range(50):
        assert all(w.isalpha() and w.islower() for w in generate_phrase().split())


def test_normalise_phrase_tolerates_typing_artefacts():
    assert normalise_phrase("  Amber   Willow ") == "amber willow"
    assert normalise_phrase("AMBER WILLOW") == "amber willow"


def test_valid_challenge_is_consumed_once(db):
    ch = create_challenge(db, "alice", "login")

    got, err = consume_challenge(db, ch.nonce, "alice", "login")
    assert err is None
    assert got is not None and got.phrase == ch.phrase


def test_replaying_a_nonce_is_rejected(db):
    ch = create_challenge(db, "alice", "login")

    _, first = consume_challenge(db, ch.nonce, "alice", "login")
    _, second = consume_challenge(db, ch.nonce, "alice", "login")

    assert first is None
    assert second is ReasonCode.CHALLENGE_REUSED


def test_unknown_nonce_is_rejected(db):
    _, err = consume_challenge(db, "never-issued-this-one", "alice", "login")
    assert err is ReasonCode.CHALLENGE_UNKNOWN


def test_expired_challenge_is_rejected(db):
    ch = create_challenge(db, "alice", "login")
    db.execute(
        "UPDATE auth_challenges SET expires_at = ? WHERE nonce = ?",
        (time.time() - 1, ch.nonce),
    )

    _, err = consume_challenge(db, ch.nonce, "alice", "login")
    assert err is ReasonCode.CHALLENGE_EXPIRED


def test_challenge_issued_for_another_user_is_rejected(db):
    ch = create_challenge(db, "alice", "login")

    _, err = consume_challenge(db, ch.nonce, "mallory", "login")
    assert err is ReasonCode.CHALLENGE_WRONG_USER


def test_enrollment_challenge_cannot_be_spent_on_login(db):
    ch = create_challenge(db, "alice", "enrollment")

    _, err = consume_challenge(db, ch.nonce, "alice", "login")
    assert err is ReasonCode.CHALLENGE_WRONG_USER


def test_nonce_is_burned_even_when_validation_fails(db):
    """A failed attempt must still cost the attacker the nonce.

    Otherwise the same challenge could be resubmitted with tweaked event
    streams, turning the decision endpoint into a tuning oracle.
    """
    ch = create_challenge(db, "alice", "login")

    _, first = consume_challenge(db, ch.nonce, "mallory", "login")
    assert first is ReasonCode.CHALLENGE_WRONG_USER

    _, second = consume_challenge(db, ch.nonce, "alice", "login")
    assert second is ReasonCode.CHALLENGE_REUSED


def test_username_is_matched_case_insensitively(db):
    ch = create_challenge(db, "Alice", "login")
    _, err = consume_challenge(db, ch.nonce, "ALICE", "login")
    assert err is None


def test_purge_removes_only_long_dead_challenges(db):
    fresh = create_challenge(db, "alice", "login")
    stale = create_challenge(db, "bob", "login")
    db.execute(
        "UPDATE auth_challenges SET expires_at = ? WHERE nonce = ?",
        (time.time() - 10_000, stale.nonce),
    )

    removed = purge_expired_challenges(db)

    assert removed == 1
    remaining = {r["nonce"] for r in db.execute("SELECT nonce FROM auth_challenges")}
    assert remaining == {fresh.nonce}
