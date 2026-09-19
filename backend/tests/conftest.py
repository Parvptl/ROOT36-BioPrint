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
