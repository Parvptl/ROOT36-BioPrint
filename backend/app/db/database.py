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

# Lets tests and the offline evaluation harness point at their own database
# without mutating the frozen Settings object.
_db_path_override: Path | None = None


def set_db_path_override(path: Path | None) -> None:
    global _db_path_override
    _db_path_override = path


def resolve_db_path(db_path: Path | None = None) -> Path:
    return db_path or _db_path_override or settings.db_path


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create the database file and apply the schema. Idempotent.

    Note: `with sqlite3.connect(...)` commits but does NOT close the handle.
    On Windows a leaked handle keeps a lock on the file, so the close is
    explicit here.
    """
    target = resolve_db_path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    conn = _connect(target)
    try:
        conn.executescript(schema)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


# Columns added to tables that already exist in databases out in the wild.
# CREATE TABLE IF NOT EXISTS silently does nothing for an existing table, so a
# new column has to be added explicitly or an older database keeps working
# while quietly missing it.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("auth_attempts", "threshold", "REAL"),
    ("behavior_profiles", "maturity", "TEXT NOT NULL DEFAULT 'MATURE'"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, decl in _ADDED_COLUMNS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue  # table not present in this database at all
        if column not in existing:
            # Table and column names here are literals from the tuple above;
            # no caller supplies them. SQLite cannot bind identifiers.
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a connection, committing on success and rolling back on error."""
    target = resolve_db_path(db_path)
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
