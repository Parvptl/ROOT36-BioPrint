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

from app.api import routes_auth, routes_enrollment, routes_login, routes_ops
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
    # X-Operator-Key and X-Demo-Reset-Key must be listed explicitly: a browser
    # will not even send the preflight-approved request without them here, so
    # omitting one makes the endpoint look unreachable rather than forbidden.
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Operator-Key",
        "X-Demo-Reset-Key",
    ],
)


app.include_router(routes_auth.router)
app.include_router(routes_enrollment.router)
app.include_router(routes_login.router)
app.include_router(routes_ops.router)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe. Reports config flags but never the secret itself."""
    return {
        "status": "ok",
        "service": "bioprint",
        "version": app.version,
        # Stated as a constant, not read from config: there is no switch that
        # turns raw-event retention on. See Settings for why.
        "raw_event_retention": False,
        "demo_reset_enabled": bool(settings.demo_reset_key),
    }
