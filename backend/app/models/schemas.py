"""API request and response schemas."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.events import BehaviorSessionIn

USERNAME_RE = re.compile(r"^[a-zA-Z0-9._-]{3,32}$")

# Minimum enrollment rounds needed before a profile can be built. Below this
# the leave-one-out calibration has too few folds to say anything.
MIN_ENROLLMENT_SESSIONS = 4


class RegisterIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=256)
    consent: bool = Field(description="User agreed to behavioural data collection")

    @field_validator("username")
    @classmethod
    def _username_charset(cls, v: str) -> str:
        if not USERNAME_RE.match(v):
            raise ValueError("username may contain only letters, digits, dot, underscore, hyphen")
        return v.lower()


class RegisterOut(BaseModel):
    user_id: int
    username: str
    enrolled: bool
    sessions_required: int


class ChallengeOut(BaseModel):
    """A single-use behavioural challenge.

    The phrase is freshly generated per attempt. That is what makes a recorded
    interaction stream useless on replay, and it is also why the resulting
    fingerprint is content-independent rather than tied to one fixed string.
    """

    nonce: str
    phrase: str
    expires_in: int
    round_index: int | None = None
    rounds_total: int | None = None


class EnrollmentStartIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=256)


class EnrollmentSubmitIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=32)
    session: BehaviorSessionIn


class EnrollmentProgressOut(BaseModel):
    accepted: bool
    sessions_captured: int
    sessions_required: int
    profile_built: bool
    message: str
    quality: dict[str, object] | None = None


class LoginChallengeIn(BaseModel):
    """Requested on page load, before any password is typed.

    Deliberately takes no password: behaviour must be captured across the whole
    form fill, not just after the credential is known. It also means this
    endpoint always answers identically for real and unknown usernames.
    """

    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=32)


class LoginBehaviorIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=256)
    session: BehaviorSessionIn


class SignalDetail(BaseModel):
    """One contributing signal, shaped for the dashboard."""

    code: str
    label: str
    band: Literal["LOW", "MEDIUM", "HIGH"]
    detail: str


class LatencyBreakdown(BaseModel):
    """Measured, not estimated. Every field is timed with perf_counter."""

    total_ms: float
    validation_ms: float
    extraction_ms: float
    identity_ms: float
    automation_ms: float


class DecisionOut(BaseModel):
    """The authentication verdict.

    Every numeric field here is computed server-side from the raw event stream.
    None of them is accepted from the client under any circumstance.
    """

    decision: Literal["ALLOW", "BLOCK"]
    reason: str
    message: str
    identity_score: float | None = None
    automation_score: float | None = None
    integrity_score: float
    coverage: float | None = None
    threshold: float | None = None
    modality_scores: dict[str, float] | None = None
    signals: list[SignalDetail] = Field(default_factory=list)
    latency: LatencyBreakdown
    session_token: str | None = None
    attempt_id: int | None = None


class ProfileStatusOut(BaseModel):
    username: str
    enrolled: bool
    sessions_captured: int
    sessions_required: int
    feature_count: int | None = None
    population_size: int | None = None
    threshold: float | None = None
    threshold_source: str | None = None
    calibration_note: str | None = None
