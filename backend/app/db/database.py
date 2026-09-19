"""SQLite access layer.

Deliberately thin: sqlite3 from the standard library, one connection per
request, parameterised queries everywhere. An ORM would add a dependency and a
migration story we do not need for a single-file local database.

Every query in this codebase uses bound parameters. There is no string
interpolation of user input into SQL anywhere, which is what keeps the
username field from being an injection surface.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import settings

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create the database file and apply the schema. Idempotent."""
    target = db_path or settings.db_path
    target.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with _connect(target) as conn:
        conn.executescript(schema)


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a connection, committing on success and rolling back on error."""
    target = db_path or settings.db_path
    conn = _connect(target)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def db_dependency() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency wrapper around get_connection."""
    with get_connection() as conn:
        yield conn
