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
    rounds: int | None = Field(
        default=None,
        description="Enrollment captures to collect. None uses the product default; "
                    "8 selects the research baseline.",
    )


class EnrollmentSubmitIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=32)
    session: BehaviorSessionIn
    rounds: int | None = Field(
        default=None,
        description="Enrollment captures to collect. None uses the product default; "
                    "8 selects the research baseline.",
    )


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
    """One contributing signal."""

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
    credential_ms: float = 0.0
    persistence_ms: float = 0.0


class DecisionOut(BaseModel):
    """What the person attempting to log in is told.

    Deliberately carries no exact identity score, no automation score and no
    threshold. An earlier version returned all three, which made this endpoint
    a tuning oracle: an attacker holding a correct password could read their
    precise deviation and the exact bar to clear, then hill-climb toward it.
    Returning coarse categories instead costs them roughly a bit and a half per
    attempt rather than a full gradient, and every attempt already costs a
    single-use challenge and is rate limited.

    The exact numbers still exist. They are written to the audit trail. The
    subject of a decision does not get the gradient; the operator does.
    Everything here is computed server-side from the raw event stream. None of
    it is accepted from the client under any circumstance.
    """

    decision: Literal["ALLOW", "BLOCK"]
    reason: str
    message: str
    headline: str | None = None
    # PASS or FAIL. Integrity is genuinely binary, so a category loses nothing.
    integrity_status: str | None = None
    # Which categories of signal disagreed, each banded LOW/MEDIUM/HIGH. This
    # is the explainability the user is owed without handing over a gradient.
    signals: list[SignalDetail] = Field(default_factory=list)
    # How much of the enrolled profile could be compared, banded. A user who is
    # told "not enough was captured" needs to know that much to retry usefully.
    coverage_band: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    latency: LatencyBreakdown
    session_token: str | None = None
    attempt_id: int | None = None


class DemoResetOut(BaseModel):
    reset: bool
    rows_deleted: dict[str, int]


class ProfileStatusOut(BaseModel):
    """Enrollment progress, for the account's own UI.

    This endpoint is UNAUTHENTICATED — it answers for any username, and answers
    identically for unknown and unenrolled accounts so it cannot be used to
    enumerate registrations.

    It therefore must not carry the exact decision threshold. DecisionOut
    withholds that number on purpose, because an attacker holding a correct
    password who can read their precise deviation and the exact bar to clear
    can hill-climb toward it. Serving the same number from an unauthenticated
    GET would have handed back exactly what the login response refuses to give.
    The exact value still goes to the audit trail and the key-gated operator
    dashboard.

    `threshold_source` stays: it names the method, not the operating point, and
    a user is entitled to know whether their profile was calibrated or is still
    on a cold-start default.
    """

    username: str
    enrolled: bool
    sessions_captured: int
    sessions_required: int
    feature_count: int | None = None
    population_size: int | None = None
    threshold_source: str | None = None
    calibration_note: str | None = None
    # Coarse lifecycle state, for the UI. COLD_START | WARMING | ESTABLISHED |
    # MATURE. Reveals no gradient: it says how personalised the profile is, not
    # where the bar sits.
    maturity: str | None = None
    profile_version: int | None = None
