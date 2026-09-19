"""BioPrint API entry point.

Run locally with:
    uvicorn app.main:app --reload --port 8000
from the backend/ directory with the virtualenv active.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("bioprint")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    log.info("database ready at %s", settings.db_path)
    if settings.secret_is_ephemeral:
        log.warning(
            "BIOPRINT_SECRET_KEY is unset; generated an ephemeral key. "
            "Sessions will not survive a restart. Set it in backend/.env."
        )
    if settings.retain_raw_events:
        log.warning(
            "BIOPRINT_RETAIN_RAW_EVENTS is enabled. Raw behavioural event "
            "streams will be written to disk. Debug use only."
        )
    yield


app = FastAPI(
    title="BioPrint",
    description="Behavioural-biometric login security.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe. Reports config flags but never the secret itself."""
    return {
        "status": "ok",
        "service": "bioprint",
        "version": app.version,
        "raw_event_retention": settings.retain_raw_events,
    }
