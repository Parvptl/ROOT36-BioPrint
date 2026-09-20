"""Evaluation-only capture of derived behavioural feature vectors.

Phase D could not answer whether the real Aalto prior helps real people,
because BioPrint keeps nothing that could answer it: raw events are discarded
at extraction and enrollment vectors are deleted once a profile is fitted. That
is the correct product behaviour and it is not changed here.

This module adds a seam that is **off by default**. With
``BIOPRINT_EVALUATION_MODE=1`` the 27 derived numbers that already reach the
profile are additionally copied to a SEPARATE database, so one real capture can
later be scored under more than one population prior. Without the flag, not one
extra byte is written and the product behaves exactly as before.

What is written
---------------
    participant        keyed pseudonym (E-prefixed), never a username
    target             keyed pseudonym of the account being attempted
    session_id         random uuid4
    role               'enrollment' | 'login'
    label              'genuine' | 'human_impostor'
    enrollment_index   ordinal within that participant's enrollment
    features_json      the derived 27-dimensional vector
    coverage           fraction of the registry observed
    created_at         unix time

What is NOT written, ever
-------------------------
    passwords, password hashes, any password-derived value
    raw keystroke or pointer events
    the challenge phrase, or any typed text
    usernames, emails, names, IP addresses, user agents

The pseudonym is an HMAC of the username under the server secret, truncated.
It is stable, so a participant's eight enrollment rounds group together, and it
is not reversible without the secret. No table maps it back to a person: the
operator keeps that mapping outside the dataset, which is what
``evaluation/eval_collect.py pseudonym`` is for.

The label is not guessed. The operator declares who is actually at the keyboard
between phases, and the collector compares that to the account being claimed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import sqlite3
import time
import uuid
from pathlib import Path

from app.config import settings

log = logging.getLogger("bioprint.evaluation_capture")

ROLE_ENROLLMENT = "enrollment"
ROLE_LOGIN = "login"

LABEL_GENUINE = "genuine"
LABEL_IMPOSTOR = "human_impostor"

# Fields that must never appear in a captured row. Asserted by the tests rather
# than merely intended.
FORBIDDEN_KEYS = frozenset(
    {"password", "password_hash", "username", "email", "name", "phrase",
     "events", "raw_events", "keystrokes", "pointer", "nonce"}
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT    NOT NULL UNIQUE,
    participant      TEXT    NOT NULL,   -- who was actually typing (pseudonym)
    target           TEXT    NOT NULL,   -- whose account was claimed (pseudonym)
    role             TEXT    NOT NULL,   -- enrollment | login
    label            TEXT    NOT NULL,   -- genuine | human_impostor
    enrollment_index INTEGER,            -- ordinal within enrollment, else NULL
    features_json    TEXT    NOT NULL,   -- derived vector only
    coverage         REAL    NOT NULL,
    created_at       REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_participant
    ON evaluation_sessions(participant, role);

-- Rows the collector refused, with the reason. Recorded rather than dropped:
-- "we captured 96 of 120 rounds" is only meaningful alongside why 24 are
-- missing.
CREATE TABLE IF NOT EXISTS evaluation_rejects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    participant TEXT,
    role        TEXT,
    reason      TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


def is_enabled() -> bool:
    """Evaluation capture is opt-in and off by default."""
    return bool(settings.evaluation_mode)


class EphemeralSecretError(RuntimeError):
    """The pseudonym key would not survive a restart."""


def assert_stable_identifiers() -> None:
    """Refuse to collect when participant ids cannot survive a restart.

    Pseudonyms are keyed on the server secret. With no BIOPRINT_SECRET_KEY
    configured the secret is regenerated per process, so restarting the backend
    halfway through a collection would give every participant a brand new id
    and silently split one person's eight enrollment rounds across two
    identities that can never be rejoined.

    Failing loudly at the start of collection is the only recoverable moment:
    afterwards the data is simply wrong and there is no way to tell.
    """
    if settings.secret_is_ephemeral:
        raise EphemeralSecretError(
            "BIOPRINT_SECRET_KEY is unset, so participant pseudonyms would "
            "change on every restart and a participant's sessions could not be "
            "grouped. Set it in backend/.env before collecting."
        )


def pseudonym(username: str) -> str:
    """Stable, non-reversible participant id derived from the account name.

    HMAC under the server secret rather than a bare hash: a bare SHA-256 of a
    short username is trivially reversible by guessing, which would put the
    account name back into the dataset in all but name.
    """
    digest = hmac.new(
        settings.secret_key.encode("utf-8"),
        f"evaluation-participant:{username.lower()}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"E{digest[:10]}"


# ----------------------------------------------------------------- actor file


def _actor_path() -> Path:
    return settings.eval_db_path.parent / "current_actor.txt"


def current_actor() -> str | None:
    """Which participant the operator says is presently at the keyboard.

    A file rather than an environment variable, so the operator can switch
    between the genuine and impostor phases without restarting the server, and
    rather than an HTTP endpoint, so no new authenticated surface is added to
    the product for the sake of an experiment.
    """
    path = _actor_path()
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def set_actor(username: str | None) -> Path:
    """Record who is typing. Writes the PSEUDONYM, not the name."""
    path = _actor_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("" if username is None else pseudonym(username), encoding="utf-8")
    return path


# -------------------------------------------------------------------- storage


def _connect() -> sqlite3.Connection:
    settings.eval_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.eval_db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _record_reject(participant: str | None, role: str, reason: str) -> None:
    try:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO evaluation_rejects (participant, role, reason, created_at) "
                "VALUES (?, ?, ?, ?)",
                (participant, role, reason, time.time()),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:  # pragma: no cover - diagnostics only
        log.warning("evaluation reject not recorded: %s", exc)


def validate_vector(features: dict[str, float]) -> str | None:
    """Reasons a vector must not enter the dataset. None means it is usable."""
    if not features:
        return "empty_feature_vector"
    for name, value in features.items():
        if name in FORBIDDEN_KEYS:
            return f"forbidden_key:{name}"
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"non_numeric:{name}"
        if math.isnan(value):
            return f"nan:{name}"
        if math.isinf(value):
            return f"infinite:{name}"
    return None


def capture(
    *,
    role: str,
    claimed_username: str,
    features: dict[str, float],
    coverage: float,
    enrollment_index: int | None = None,
) -> str | None:
    """Copy one derived vector into the evaluation dataset.

    Returns the session id, or None when nothing was written. Never raises:
    an evaluation harness must not be able to break a login.
    """
    if not is_enabled():
        return None

    try:
        assert_stable_identifiers()
    except EphemeralSecretError as exc:
        _record_reject(None, role, "ephemeral_secret")
        log.error("evaluation capture disabled: %s", exc)
        return None

    target = pseudonym(claimed_username)
    actor = current_actor()

    try:
        if actor is None:
            _record_reject(None, role, "no_actor_declared")
            return None

        problem = validate_vector(features)
        if problem is not None:
            _record_reject(actor, role, problem)
            return None

        # The label is derived from a declaration, not inferred from the
        # decision. Labelling by outcome would define an impostor as "an
        # attempt the system rejected", which is the thing under test.
        label = LABEL_GENUINE if actor == target else LABEL_IMPOSTOR
        if role == ROLE_ENROLLMENT and label != LABEL_GENUINE:
            _record_reject(actor, role, "enrollment_by_non_owner")
            return None

        session_id = str(uuid.uuid4())
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO evaluation_sessions "
                "(session_id, participant, target, role, label, enrollment_index, "
                " features_json, coverage, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, actor, target, role, label, enrollment_index,
                    json.dumps(features, separators=(",", ":"), sort_keys=True),
                    float(coverage), time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return session_id

    except Exception as exc:  # noqa: BLE001 - never break authentication
        log.warning("evaluation capture failed (ignored): %s", exc)
        return None
