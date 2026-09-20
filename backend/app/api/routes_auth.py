"""Account creation and profile status."""

from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.api.ratelimit import REGISTER_LIMIT, limiter
from app.auth.passwords import hash_password
from app.db import repository
from app.db.database import db_dependency
from app.models.schemas import (
    MIN_ENROLLMENT_SESSIONS,
    ProfileStatusOut,
    RegisterIn,
    RegisterOut,
    USERNAME_RE,
)

log = logging.getLogger("bioprint.auth")
router = APIRouter(prefix="/auth", tags=["auth"])

# Eight, chosen by measurement rather than by feel.
#
# A sweep over 30 independent enrollments per setting (evaluation/reliability_
# sweep.py) gave, against moderately-different impostors:
#     5 rounds  -> false rejection 16.2%, equal-error about 10.2%
#     8 rounds  ->                  7.1%,                   7.7%
#    12 rounds  ->                  8.3%,                   7.3%
#
# Five rounds leaves the per-feature scale estimates too noisy: the median
# absolute deviation of five samples is a poor estimate of spread, so genuine
# logins land outside a threshold fitted to it. Eight roughly halves that.
# Twelve buys almost nothing for another ninety seconds of the user's time.
#
# Those figures are synthetic mechanism validation, not real accuracy.
#
# Kept as the RESEARCH baseline and still reachable, because it remains the
# strongest profile the system can build and every comparison is anchored to it.
RESEARCH_ENROLLMENT_ROUNDS = 8

# What the product actually asks a new user for.
#
# Eight dedicated rounds is about two minutes of typing before anyone has
# logged in once, and that is real abandonment. The question is what the
# shortest defensible enrollment is, and it was measured rather than guessed
# (evaluation/enrollment_size_ablation.py, 120 generated typists, 1200 genuine
# and 1200 impostor attempts per condition, the real Aalto prior held fixed,
# thresholds calibrated on 60 users and reported on 60 disjoint held-out ones):
#
#     at a matched 10% false-acceptance budget
#     1 capture   false rejection  9.7%   false acceptance  9.8%
#     2 captures                   6.5%                     8.8%
#     8 captures                   0.0%                    10.0%
#
# Two captures beat one on BOTH axes, at every budget swept (2, 5, 10, 15%).
# That is dominance rather than a trade-off, and it is why the product asks for
# two. Threshold-free: ROC-AUC 0.9758 -> 0.9845, equal-error 8.50% -> 6.17%.
#
# Eight remains far stronger than either, which is the honest shape of this:
# a two-capture profile is a usable starting point, not a mature one. It is
# why the cold-start threshold is tighter than the mature one and why
# adaptation graduates the profile as genuine logins arrive.
#
# Generated typists. Not a claim about real-human accuracy.
ENROLLMENT_ROUNDS = 2


def client_key(request: Request, suffix: str = "") -> str:
    host = request.client.host if request.client else "unknown"
    return f"{host}:{suffix}"


def enforce(request: Request, suffix: str, limit) -> None:
    allowed, retry_after = limiter.check(client_key(request, suffix), limit)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts. Please wait before trying again.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


@router.post("/register", response_model=RegisterOut, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterIn,
    request: Request,
    conn: sqlite3.Connection = Depends(db_dependency),
) -> RegisterOut:
    enforce(request, "register", REGISTER_LIMIT)

    if not payload.consent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Behavioural enrollment requires consent to data collection.",
        )

    if repository.get_user(conn, payload.username) is not None:
        # Registration inherently reveals whether a username is taken; there is
        # no way to offer account creation without that. The login and
        # challenge endpoints do not leak it.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="That username is already taken.",
        )

    user_id = repository.create_user(
        conn, payload.username, hash_password(payload.password)
    )
    # Username only. The password never appears in a log line, at any level.
    log.info("registered user id=%s username=%s", user_id, payload.username)

    return RegisterOut(
        user_id=user_id,
        username=payload.username,
        enrolled=False,
        sessions_required=ENROLLMENT_ROUNDS,
    )


@router.get("/profile/status", response_model=ProfileStatusOut)
def profile_status(
    username: str = Query(min_length=3, max_length=32),
    conn: sqlite3.Connection = Depends(db_dependency),
) -> ProfileStatusOut:
    if not USERNAME_RE.match(username):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid username format.",
        )

    user = repository.get_user(conn, username)
    if user is None:
        # Answers identically for an unknown account and an unenrolled one, so
        # this endpoint cannot be used to enumerate registrations.
        return ProfileStatusOut(
            username=username.lower(),
            enrolled=False,
            sessions_captured=0,
            sessions_required=ENROLLMENT_ROUNDS,
        )

    captured = repository.count_enrollment_sessions(conn, user["id"])
    profile = repository.load_profile(conn, user["id"])

    if profile is None:
        return ProfileStatusOut(
            username=user["username"],
            enrolled=False,
            sessions_captured=captured,
            sessions_required=ENROLLMENT_ROUNDS,
        )

    return ProfileStatusOut(
        username=user["username"],
        enrolled=True,
        sessions_captured=profile.session_count,
        sessions_required=MIN_ENROLLMENT_SESSIONS,
        feature_count=len(profile.features),
        population_size=profile.population_size,
        # No exact threshold here. See ProfileStatusOut.
        threshold_source=profile.threshold_source,
        calibration_note=str(profile.calibration.get("note", "")),
        maturity=profile.maturity.value,
        profile_version=profile.version,
    )
