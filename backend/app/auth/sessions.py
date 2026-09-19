"""Server-side session issuance.

A session token is only minted after the risk engine returns ALLOW. A correct
password alone never produces one, which is the property that makes the
behavioural layer load-bearing rather than decorative.

Tokens are random 256-bit values. Only their SHA-256 hash is stored, so a
database leak does not hand an attacker usable sessions. Lookup is by hash,
and the comparison is done by the database on an indexed primary key.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time

from app.config import settings


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_session(conn: sqlite3.Connection, user_id: int) -> tuple[str, float]:
    """Create a session for `user_id`. Returns (token, expires_at)."""
    token = secrets.token_urlsafe(32)
    now = time.time()
    expires_at = now + settings.session_ttl_seconds
    conn.execute(
        "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) "
        "VALUES (?, ?, ?, ?)",
        (_hash_token(token), user_id, now, expires_at),
    )
    return token, expires_at


def resolve_session(conn: sqlite3.Connection, token: str) -> int | None:
    """Return the user_id for a live session token, or None."""
    row = conn.execute(
        "SELECT user_id, expires_at FROM sessions WHERE token_hash = ?",
        (_hash_token(token),),
    ).fetchone()
    if row is None:
        return None
    if row["expires_at"] <= time.time():
        # Expired tokens are removed on sight rather than left to accumulate.
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))
        return None
    return int(row["user_id"])


def revoke_session(conn: sqlite3.Connection, token: str) -> None:
    conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))


def purge_expired_sessions(conn: sqlite3.Connection) -> int:
    cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (time.time(),))
    return cur.rowcount
