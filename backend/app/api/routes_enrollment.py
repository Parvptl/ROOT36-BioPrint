"""Enrollment: capture rounds, then fit and calibrate a profile."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import replace

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.ratelimit import ENROLLMENT_LIMIT, limiter
from app.api.pipeline import run_pipeline
from app.api.routes_auth import (
    ENROLLMENT_ROUNDS,
    RESEARCH_ENROLLMENT_ROUNDS,
    enforce,
)
from app.auth.challenge import consume_challenge, create_challenge
from app.auth.passwords import verify_password, waste_time_like_a_real_verify
from app.behavioral.bot_detection.detector import AutomationResult
from app.behavioral.fingerprint.calibration import build_calibrated_profile
from app.behavioral.fingerprint.population import (
    contributor_tag,
    load_population_prior,
    load_population_samples,
    record_population_sample,
)
from app.behavioral.scoring.reasons import ReasonCode, explain
from app.behavioral.scoring.risk_engine import AUTOMATION_BLOCK
from app import evaluation_capture
from app.db import repository
from app.db.database import db_dependency
from app.models.schemas import (
    ChallengeOut,
    EnrollmentProgressOut,
    EnrollmentStartIn,
    EnrollmentSubmitIn,
)
from app.behavioral.fingerprint.profile import (
    MIN_FEATURE_SESSIONS as MIN_SESSIONS_FOR_OWN_SCALE,
    BehaviorProfile,
    Maturity,
    fit_cold_start_profile,
    fit_early_profile,
)
from app.behavioral.fingerprint.calibration import cold_start_threshold

log = logging.getLogger("bioprint.enrollment")
router = APIRouter(prefix="/auth/enrollment", tags=["enrollment"])

# A round must produce at least this much of the feature set to be worth
# keeping. A baseline built from thin captures would be a bad baseline, and
# every login afterwards would pay for it.
MIN_ROUND_COVERAGE = 0.45

# Enrollment sizes the API will honour. Anything else is coerced to the product
# default rather than rejected, because an arbitrary size is not a user error —
# but only these three have a fitting path behind them:
#
#   1  fit_cold_start_profile   centre from one observation, scale from prior
#   2  fit_early_profile        centre from two, scale still from prior
#   8  fit_profile + LOO        centre and scale both personal (research)
#
# 3 to 7 are deliberately absent. fit_profile needs MIN_FEATURE_SESSIONS (3)
# before it characterises a feature, and between 3 and 7 the per-feature MAD is
# too noisy to calibrate a threshold against — the sweep that chose 8 measured
# 5 rounds at 16.2% false rejection. Allowing them would ship a profile whose
# threshold cannot be trusted.
ALLOWED_ROUNDS = (1, ENROLLMENT_ROUNDS, RESEARCH_ENROLLMENT_ROUNDS)


def _authenticate(conn: sqlite3.Connection, username: str, password: str):
    """Verify credentials, in constant-ish time for unknown usernames."""
    user = repository.get_user(conn, username)
    if user is None:
        waste_time_like_a_real_verify()
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


@router.post("/start", response_model=ChallengeOut)
def start_round(
    payload: EnrollmentStartIn,
    request: Request,
    conn: sqlite3.Connection = Depends(db_dependency),
) -> ChallengeOut:
    """Issue a challenge for the next enrollment round.

    Credentials are re-verified before every round. Without that, anyone who
    knew a username could append their own behaviour to someone else's
    baseline and poison the profile until it accepted them.
    """
    enforce(request, "enroll", ENROLLMENT_LIMIT)

    user = _authenticate(conn, payload.username, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username or password is incorrect.",
        )

    # Guard on the profile, not on the captured-round count. The rounds are
    # deleted once a profile is fitted from them, so counting them would read
    # zero for an enrolled user and let the baseline be silently rebuilt.
    if repository.load_profile(conn, user["id"]) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Enrollment is already complete. Reset the profile to enroll again.",
        )

    rounds = getattr(payload, 'rounds', ENROLLMENT_ROUNDS)
    
    # Check if a requested custom rounds is valid
    if rounds not in ALLOWED_ROUNDS:
        rounds = ENROLLMENT_ROUNDS

    captured = repository.count_enrollment_sessions(conn, user["id"])
    if captured >= rounds:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Enrollment is already complete for this account.",
        )

    challenge = create_challenge(conn, payload.username, "enrollment")
    return ChallengeOut(
        nonce=challenge.nonce,
        phrase=challenge.phrase,
        expires_in=challenge.expires_in,
        round_index=captured + 1,
        rounds_total=rounds,
    )


@router.post("/submit", response_model=EnrollmentProgressOut)
def submit_round(
    payload: EnrollmentSubmitIn,
    request: Request,
    conn: sqlite3.Connection = Depends(db_dependency),
) -> EnrollmentProgressOut:
    """Accept one captured round; fit the profile once enough have arrived."""
    enforce(request, "enroll", ENROLLMENT_LIMIT)

    user = repository.get_user(conn, payload.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username or password is incorrect.",
        )

    # Same guard on the submit path: an enrollment challenge must not be
    # spendable against an account that already has a baseline.
    if repository.load_profile(conn, user["id"]) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Enrollment is already complete. Reset the profile to enroll again.",
        )

    challenge, error = consume_challenge(
        conn, payload.session.nonce, payload.username, "enrollment"
    )
    if error is not None or challenge is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This enrollment round is no longer valid. Please start a new one.",
        )

    result = run_pipeline(payload.session, challenge, profile=None)
    captured = repository.count_enrollment_sessions(conn, user["id"])

    rejection = _reject_round(result.integrity, result.automation, result.features.coverage)
    if rejection is not None:
        return EnrollmentProgressOut(
            accepted=False,
            sessions_captured=captured,
            sessions_required=ENROLLMENT_ROUNDS,
            profile_built=False,
            message=rejection,
        )

    repository.add_enrollment_session(
        conn, user["id"], result.features.as_dict(), result.features.coverage
    )
    # Evaluation-only, no-op unless BIOPRINT_EVALUATION_MODE is set. Copies the
    # derived vector (not the events it came from) to a separate database so
    # one real capture can be scored under more than one population prior.
    evaluation_capture.capture(
        role=evaluation_capture.ROLE_ENROLLMENT,
        claimed_username=user["username"],
        features=result.features.as_dict(),
        coverage=result.features.coverage,
        enrollment_index=captured,
    )
    # The windowed training set that used to be produced here fed the retired
    # Isolation Forest and nothing else, so it is no longer collected. One less
    # derived artifact retained per enrollment round.
    captured += 1

    rounds = getattr(payload, 'rounds', ENROLLMENT_ROUNDS)
    if rounds not in ALLOWED_ROUNDS:
        rounds = ENROLLMENT_ROUNDS

    if captured < rounds:
        return EnrollmentProgressOut(
            accepted=True,
            sessions_captured=captured,
            sessions_required=rounds,
            profile_built=False,
            message=f"Round {captured} recorded.",
        )

    profile = _build_profile(conn, user["id"], rounds)
    log.info(
        "profile built user=%s features=%d threshold=%.4f source=%s population=%d",
        user["username"],
        len(profile.features),
        profile.threshold,
        profile.threshold_source,
        profile.population_size,
    )

    return EnrollmentProgressOut(
        accepted=True,
        sessions_captured=captured,
        sessions_required=rounds,
        profile_built=True,
        message="Behavioural profile created.",
        quality={
            "features_modelled": len(profile.features),
            "enrollment_rounds": profile.session_count,
            "population_samples": profile.population_size,
            "threshold": round(profile.threshold, 4),
            "threshold_source": profile.threshold_source,
        },
    )


def _reject_round(
    integrity, automation: AutomationResult, coverage: float
) -> str | None:
    """Reasons a round must not enter the baseline.

    Enrollment is the one place where accepting bad data is unrecoverable: a
    poisoned or automated round becomes part of what the user is measured
    against forever after.
    """
    if not integrity.ok:
        # Report the actual finding. Telling a user their phrase was wrong when
        # the real problem was timing sends them to fix something that is not
        # broken.
        reason = integrity.reason or ReasonCode.MALFORMED_EVENT_STREAM
        return f"{explain(reason)} A new phrase has been issued."

    if automation.score >= AUTOMATION_BLOCK:
        return (
            "That round looked automated rather than typed. Enrollment needs genuine "
            "interaction, so it was not recorded."
        )

    if coverage < MIN_ROUND_COVERAGE:
        return (
            "Not enough interaction was captured in that round. Please fill the whole "
            "form before submitting."
        )

    return None


def _build_profile(conn: sqlite3.Connection, user_id: int, rounds: int):
    """Fit, calibrate, train the anomaly model, store, and seed the prior."""
    sessions = repository.load_enrollment_sessions(conn, user_id)
    tag = contributor_tag(user_id)

    # This user's own samples are excluded from the population they will be
    # scored against.
    prior = load_population_prior(conn, exclude_contributor=tag)
    population_samples = load_population_samples(conn, exclude_contributor=tag)

    if rounds < MIN_SESSIONS_FOR_OWN_SCALE:
        # One or two captures. The centre is personal; the scale still comes
        # from the population prior, because a dispersion estimated from one or
        # two points is not a dispersion. See fit_early_profile.
        features = (
            fit_cold_start_profile(sessions[0], prior)
            if rounds == 1
            else fit_early_profile(sessions[:rounds], prior)
        )
        threshold_res = cold_start_threshold()
        profile = BehaviorProfile(
            features=features,
            session_count=rounds,
            population_size=prior.sample_count,
            population_informative=prior.is_informative,
            threshold=threshold_res.threshold,
            threshold_source=threshold_res.source,
            calibration=threshold_res.as_json(),
            maturity=Maturity.COLD_START if rounds == 1 else Maturity.WARMING,
        )
    else:
        # An Isolation Forest was trained here and could replace the threshold
        # with one recalibrated for the blended score. Retired: the blend
        # weight was 0.0, so that recalibration returned None and the
        # statistical threshold was always what shipped. See
        # app/behavioral/ml/__init__.py.
        profile = build_calibrated_profile(sessions, prior, population_samples)

    repository.save_profile(conn, user_id, profile)
    repository.mark_enrolled(conn, user_id)

    # Contribute to the prior so it improves as more people enroll. Covered by
    # the consent given at registration, stored as derived features under a
    # keyed tag rather than a user id.
    for features in sessions:
        record_population_sample(conn, features, source="enrollment", contributor=tag)

    if rounds != 1:
        repository.clear_enrollment_sessions(conn, user_id)
        repository.clear_enrollment_windows(conn, user_id)
    return profile

