"""BioPrint API entry point.

Run locally with:
    uvicorn app.main:app --reload --port 8000
from the backend/ directory with the virtualenv active.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_auth, routes_enrollment, routes_login, routes_ops
from app.config import settings
from app.db.database import init_db

# The production frontend build, if it has been made. Serving it from here
# means the whole product runs on one port from one command, with the browser
# and the API sharing an origin so no CORS is involved at all. A judge who
# serves dist/ separately on some other port otherwise gets a login page that
# silently cannot reach the backend.
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

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
        "frontend_bundled": FRONTEND_DIST.is_dir(),
    }


# --- static frontend -------------------------------------------------------
#
# Registered last, deliberately. FastAPI matches routes in registration order,
# so every API route above wins over the catch-all below.

if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/{requested_path:path}", include_in_schema=False)
    def serve_frontend(requested_path: str) -> FileResponse:
        """Serve the built app, falling back to index.html for client routes.

        /login and /enroll are React Router paths with no file behind them, so
        anything that is not a real file has to return index.html and let the
        router sort it out.
        """
        index = FRONTEND_DIST / "index.html"

        if requested_path:
            # Resolve before comparing. Without this, a request for
            # ../../backend/.env would escape the bundle and serve the secret
            # key. `resolve()` collapses the traversal so the containment
            # check below actually means something.
            candidate = (FRONTEND_DIST / requested_path).resolve()
            try:
                inside_bundle = candidate.is_relative_to(FRONTEND_DIST.resolve())
            except ValueError:  # pragma: no cover - differing drives on Windows
                inside_bundle = False
            if inside_bundle and candidate.is_file():
                return FileResponse(candidate)

        if not index.is_file():
            raise HTTPException(status_code=404, detail="Not found.")
        return FileResponse(index)
