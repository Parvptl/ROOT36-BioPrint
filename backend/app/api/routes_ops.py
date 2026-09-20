"""Operational endpoints used by the live demo for safe reset.

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
from app.models.schemas import DemoResetOut

log = logging.getLogger("bioprint.ops")
router = APIRouter(tags=["ops"])





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
