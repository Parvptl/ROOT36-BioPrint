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
        automation_score=row["automation_score"],
        integrity_score=row["integrity_score"],
        coverage=row["coverage"],
        latency_ms=row["latency_ms"],
        created_at=float(row["created_at"]),
    )


@router.get("/security/dashboard", response_model=DashboardOut)
def security_dashboard(
    conn: sqlite3.Connection = Depends(db_dependency),
) -> DashboardOut:
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
    log.info("demo reset deleted=%s", deleted)
    return DemoResetOut(reset=True, rows_deleted=deleted)
