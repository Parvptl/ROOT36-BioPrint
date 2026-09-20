"""Operational endpoints used by the live demo: dashboard and safe reset.

The reset path never forces ALLOW, never writes scores, and never plants
behavioural samples. It only deletes operational rows so enrollment can be
re-run honestly.
"""

from __future__ import annotations

import hmac
import logging
import sqlite3

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.behavioral.ml.store import delete_all_models
from app.config import settings
from app.db import repository
from app.db.database import db_dependency
from app.models.schemas import AttemptLogOut, DashboardOut, DemoResetOut

log = logging.getLogger("bioprint.ops")
router = APIRouter(tags=["ops"])


def _row_to_attempt(row: sqlite3.Row) -> AttemptLogOut:
    return AttemptLogOut(
        attempt_id=int(row["id"]),
        username=row["username_attempt"],
        decision=row["decision"],
        reason=row["reason"],
        identity_score=row["identity_score"],
        statistical_identity_score=row["statistical_identity_score"],
        automation_score=row["automation_score"],
        integrity_score=row["integrity_score"],
        coverage=row["coverage"],
        latency_ms=row["latency_ms"],
        created_at=float(row["created_at"]),
    )


def _require_operator_key(provided: str | None) -> None:
    """The dashboard is operator-facing and must be authenticated.

    It carries what the login response deliberately withholds: exact identity
    and automation scores per attempt. Left open, it would restore the tuning
    oracle that was just removed from the login endpoint, and it would leak
    which usernames exist.

    Disabled entirely when no key is configured, rather than defaulting to
    open. The local demo sets one in backend/.env.
    """
    expected = settings.operator_key
    if not expected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    if not hmac.compare_digest(provided or "", expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Operator key is missing or incorrect.",
        )


@router.get("/security/dashboard", response_model=DashboardOut)
def security_dashboard(
    x_operator_key: str | None = Header(default=None),
    conn: sqlite3.Connection = Depends(db_dependency),
) -> DashboardOut:
    _require_operator_key(x_operator_key)
    rows = repository.recent_attempts(conn, limit=20)
    attempts = [_row_to_attempt(row) for row in rows]
    return DashboardOut(
        latest=attempts[0] if attempts else None,
        attempts=attempts,
        demo_reset_enabled=bool(settings.demo_reset_key),
    )


@router.post("/demo/reset", response_model=DemoResetOut)
def demo_reset(
    x_demo_reset_key: str | None = Header(default=None),
    conn: sqlite3.Connection = Depends(db_dependency),
) -> DemoResetOut:
    expected = settings.demo_reset_key
    if not expected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    provided = x_demo_reset_key or ""
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Demo reset key is incorrect.",
        )

    deleted = repository.reset_operational_state(conn)
    delete_all_models()
    log.info("demo reset deleted=%s", deleted)
    return DemoResetOut(reset=True, rows_deleted=deleted)
