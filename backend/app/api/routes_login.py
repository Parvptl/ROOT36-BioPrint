"""Login: issue a challenge, then decide on the captured behaviour."""

from __future__ import annotations

import logging
import sqlite3
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.pipeline import run_pipeline
from app.api.ratelimit import CHALLENGE_LIMIT, LOGIN_LIMIT, limiter
from app.api.routes_auth import enforce
from app.auth.challenge import Challenge, consume_challenge, create_challenge
from app.auth.passwords import verify_password, waste_time_like_a_real_verify
from app.auth.sessions import issue_session
from app.behavioral.scoring.reasons import ReasonCode, explain, headline, integrity_status
from app.behavioral.scoring.risk_engine import RiskDecision, Signal, decide
from app.db import repository
from app.db.database import db_dependency
from app.models.schemas import (
    ChallengeOut,
    DecisionOut,
    LatencyBreakdown,
    LoginBehaviorIn,
    LoginChallengeIn,
    SignalDetail,
)

log = logging.getLogger("bioprint.login")
router = APIRouter(prefix="/auth/login", tags=["login"])


@router.post("/challenge", response_model=ChallengeOut)
def login_challenge(
    payload: LoginChallengeIn,
    request: Request,
    conn: sqlite3.Connection = Depends(db_dependency),
) -> ChallengeOut:
    """Hand out a fresh phrase and nonce.

    Takes no password: behaviour has to be captured across the whole form fill,
    not only after the credential is known. It also means the response is
    identical for a registered and an unregistered username, so this cannot be
    used to enumerate accounts.
    """
    enforce(request, "challenge", CHALLENGE_LIMIT)

    challenge = create_challenge(conn, payload.username, "login")
    return ChallengeOut(
        nonce=challenge.nonce,
        phrase=challenge.phrase,
        expires_in=challenge.expires_in,
    )


@router.post("/behavior", response_model=DecisionOut)
def login_behavior(
    payload: LoginBehaviorIn,
    request: Request,
    conn: sqlite3.Connection = Depends(db_dependency),
) -> DecisionOut:
    """Evaluate one login attempt and return ALLOW or BLOCK.

    Order of operations matters. The credential is checked first, because a
    wrong password is an ordinary credential rejection and should not consume
    behavioural analysis. Everything after that assumes the attacker may
    already hold a correct password, which is the threat this system exists
    for.

    There is no OTP, no email and no SMS fallback anywhere below. A
    behavioural mismatch results in BLOCK.
    """
    enforce(request, "login", LOGIN_LIMIT)
    started = time.perf_counter()

    username = payload.username.lower()
    user = repository.get_user(conn, username)

    # --- credential gate ---------------------------------------------------
    if user is None:
        waste_time_like_a_real_verify()
        credential_ms = (time.perf_counter() - started) * 1000.0
        consume_challenge(conn, payload.session.nonce, username, "login")
        return _reject(
            conn,
            user_id=None,
            username=username,
            reason=ReasonCode.INVALID_CREDENTIALS,
            started=started,
            http_status=None,
            credential_ms=credential_ms,
        )

    password_ok = verify_password(payload.password, user["password_hash"])
    credential_ms = (time.perf_counter() - started) * 1000.0
    if not password_ok:
        # The nonce is burned even here, so a wrong-password attempt cannot be
        # used to farm live challenges.
        consume_challenge(conn, payload.session.nonce, username, "login")
        return _reject(
            conn,
            user_id=user["id"],
            username=username,
            reason=ReasonCode.INVALID_CREDENTIALS,
            started=started,
            http_status=None,
            credential_ms=credential_ms,
        )

    profile = repository.load_profile(conn, user["id"])
    if profile is None:
        consume_challenge(conn, payload.session.nonce, username, "login")
        return _reject(
            conn,
            user_id=user["id"],
            username=username,
            reason=ReasonCode.ACCOUNT_NOT_ENROLLED,
            started=started,
            http_status=None,
            credential_ms=credential_ms,
        )

    # --- challenge gate ----------------------------------------------------
    challenge, error = consume_challenge(conn, payload.session.nonce, username, "login")
    if error is not None:
        return _reject(
            conn,
            user_id=user["id"],
            username=username,
            reason=error,
            started=started,
            http_status=None,
            credential_ms=credential_ms,
        )
    assert challenge is not None

    # --- behavioural analysis ----------------------------------------------
    result = run_pipeline(payload.session, challenge, profile)
    verdict = decide(
        profile=profile,
        identity=result.identity,
        automation=result.automation,
        integrity=result.integrity,
        features=result.features,
    )

    persist_started = time.perf_counter()
    attempt_id = repository.record_attempt(
        conn,
        user_id=user["id"],
        username_attempt=username,
        decision=verdict.decision,
        reason=verdict.reason.value,
        reasons=[s.code for s in verdict.signals],
        identity_score=verdict.identity_score,
        automation_score=verdict.automation_score,
        integrity_score=verdict.integrity_score,
        coverage=verdict.coverage,
        latency_ms=0.0,
    )
    persist_ms = (time.perf_counter() - persist_started) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0
    conn.execute(
        "UPDATE auth_attempts SET latency_ms = ? WHERE id = ?",
        (total_ms, attempt_id),
    )
    latency = LatencyBreakdown(
        total_ms=total_ms,
        validation_ms=result.latency.validation_ms,
        extraction_ms=result.latency.extraction_ms,
        identity_ms=result.latency.identity_ms,
        automation_ms=result.latency.automation_ms,
        credential_ms=credential_ms,
        persistence_ms=persist_ms,
    )

    log.info(
        "login user=%s decision=%s reason=%s identity=%s automation=%.3f latency=%.1fms",
        username,
        verdict.decision,
        verdict.reason.value,
        f"{verdict.identity_score:.4f}" if verdict.identity_score is not None else "n/a",
        verdict.automation_score or 0.0,
        total_ms,
    )

    token = None
    if verdict.allowed:
        # A session is only ever minted here, after the behavioural check.
        # A correct password on its own never produces one.
        token, _ = issue_session(conn, user["id"])

    return _to_response(verdict, latency, token, attempt_id)


def _to_response(
    verdict: RiskDecision,
    latency: LatencyBreakdown,
    token: str | None,
    attempt_id: int | None,
) -> DecisionOut:
    return DecisionOut(
        decision=verdict.decision,
        reason=verdict.reason.value,
        message=verdict.message,
        headline=headline(verdict.reason),
        integrity_status=integrity_status(verdict.reason, verdict.integrity_score),
        identity_score=_round(verdict.identity_score),
        automation_score=_round(verdict.automation_score),
        integrity_score=round(verdict.integrity_score, 4),
        coverage=_round(verdict.coverage),
        threshold=_round(verdict.threshold),
        modality_scores={k: round(v, 4) for k, v in verdict.modality_scores.items()} or None,
        signals=[_signal(s) for s in verdict.signals],
        latency=latency,
        session_token=token,
        attempt_id=attempt_id,
    )


def _signal(signal: Signal) -> SignalDetail:
    return SignalDetail(
        code=signal.code, label=signal.label, band=signal.band, detail=signal.detail
    )


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _reject(
    conn: sqlite3.Connection,
    user_id: int | None,
    username: str,
    reason: ReasonCode,
    started: float,
    http_status: int | None,
    credential_ms: float = 0.0,
) -> DecisionOut:
    """Block before behavioural analysis ran.

    Returned as a normal 200 BLOCK rather than a 4xx so that credential
    failures, challenge failures and behavioural failures are indistinguishable
    in shape to a caller timing responses.
    """
    total_ms = (time.perf_counter() - started) * 1000.0
    attempt_id = repository.record_attempt(
        conn,
        user_id=user_id,
        username_attempt=username,
        decision="BLOCK",
        reason=reason.value,
        reasons=[reason.value],
        identity_score=None,
        automation_score=None,
        integrity_score=1.0 if reason is not ReasonCode.INVALID_CREDENTIALS else 0.0,
        coverage=None,
        latency_ms=total_ms,
    )

    if http_status is not None:
        raise HTTPException(status_code=http_status, detail=explain(reason))

    return DecisionOut(
        decision="BLOCK",
        reason=reason.value,
        message=explain(reason),
        headline=headline(reason),
        integrity_status=integrity_status(reason, 1.0 if reason is not ReasonCode.INVALID_CREDENTIALS else 0.0),
        integrity_score=1.0 if reason is not ReasonCode.INVALID_CREDENTIALS else 0.0,
        signals=[],
        latency=LatencyBreakdown(
            total_ms=total_ms,
            validation_ms=total_ms,
            extraction_ms=0.0,
            identity_ms=0.0,
            automation_ms=0.0,
            credential_ms=credential_ms,
            persistence_ms=0.0,
        ),
        session_token=None,
        attempt_id=attempt_id,
    )
