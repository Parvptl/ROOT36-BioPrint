"""Shared test fixtures.

The environment is pointed at a throwaway database *before* `app.config` is
imported, so tests can never touch a real development database.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP_DIR = Path(tempfile.mkdtemp(prefix="bioprint-test-"))
os.environ["BIOPRINT_DB_PATH"] = str(_TMP_DIR / "test.db")
os.environ["BIOPRINT_SECRET_KEY"] = "test-only-secret-value-not-used-anywhere-else"
os.environ["BIOPRINT_CHALLENGE_TTL_SECONDS"] = "120"
# Pin every optional key OFF, so the suite does not inherit whatever the
# developer happens to have in backend/.env. Setting BIOPRINT_OPERATOR_KEY
# locally silently broke the test asserting the dashboard is closed by
# default: the test was right, the environment was leaking.
os.environ["BIOPRINT_OPERATOR_KEY"] = ""
os.environ["BIOPRINT_DEMO_RESET_KEY"] = ""
os.environ["BIOPRINT_DISABLED_FEATURES"] = ""

import itertools  # noqa: E402

import pytest  # noqa: E402

from app.db.database import get_connection, init_db, set_db_path_override  # noqa: E402

_counter = itertools.count()


@pytest.fixture
def fresh_db_path() -> Path:
    """A unique database file per test.

    Using a new filename rather than deleting a shared one avoids fighting
    Windows file locks left by SQLite's WAL sidecar files.
    """
    path = _TMP_DIR / f"test-{next(_counter)}.db"
    set_db_path_override(path)
    init_db()
    return path


@pytest.fixture
def db(fresh_db_path):
    """A connection to a freshly created, isolated database."""
    with get_connection() as conn:
        yield conn


@pytest.fixture
def client(fresh_db_path):
    """FastAPI test client against a fresh database."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def audit(fresh_db_path):
    """Read exact scores from the decision audit trail.

    The login response deliberately withholds identity and automation scores
    so it cannot be used as a tuning oracle. Tests that need the real numbers
    read them from where they are actually kept.
    """
    import sqlite3

    def _rows() -> list[dict]:
        conn = sqlite3.connect(fresh_db_path)
        conn.row_factory = sqlite3.Row
        try:
            return [
                dict(r)
                for r in conn.execute("SELECT * FROM auth_attempts ORDER BY id")
            ]
        finally:
            conn.close()

    def _last() -> dict:
        rows = _rows()
        assert rows, "no authentication attempt was recorded"
        return rows[-1]

    _rows.last = _last  # type: ignore[attr-defined]
    return _rows


@pytest.fixture
def age_challenge(fresh_db_path):
    """Back-date a challenge to simulate time the user spent filling the form.

    The integrity check rejects a capture describing more interaction time than
    the challenge has existed for, which is what stops a long pre-recorded
    stream being spliced onto a freshly fetched nonce. Synthetic fixtures
    generate their whole timeline instantly, so tests have to move the clock
    rather than sleep for twenty seconds a piece.
    """
    import sqlite3

    def _age(nonce: str, seconds: float) -> None:
        conn = sqlite3.connect(fresh_db_path)
        try:
            conn.execute(
                "UPDATE auth_challenges SET issued_at = issued_at - ?, "
                "expires_at = expires_at - ? WHERE nonce = ?",
                (seconds, seconds, nonce),
            )
            conn.commit()
        finally:
            conn.close()

    return _age


OPERATOR_KEY = "test-operator-key"


@pytest.fixture
def with_operator_key(monkeypatch):
    """Enable the operator dashboard for one test.

    Settings is a frozen dataclass, so the module global is rebound with a
    modified copy rather than mutated. routes_ops holds `settings` as a module
    attribute, which is what the endpoint reads.
    """
    from dataclasses import replace

    import app.api.routes_ops as routes_ops
    from app.config import settings

    monkeypatch.setattr(
        routes_ops, "settings", replace(settings, operator_key=OPERATOR_KEY)
    )
    return OPERATOR_KEY
