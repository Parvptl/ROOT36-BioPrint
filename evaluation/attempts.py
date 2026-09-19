"""Turning audit-trail rows into labelled attempt records, and rates from those.

Shared by run_evaluation.py and baseline_rebuild.py so a preserved run and a
fresh one are summarised by exactly the same code.

The distinction this module exists to enforce
---------------------------------------------

A wrong password and a behavioural rejection are not the same event, and
averaging them together produces a number that describes neither.

BioPrint checks the credential first and returns INVALID_CREDENTIALS without
running any behavioural analysis at all. Such an attempt has no identity score,
no threshold comparison and no bot score — there is nothing behavioural in it
to be right or wrong about. Counting it as a false rejection inflates FRR with
typing accuracy; counting a wrong-password impostor as a correct rejection
inflates the impostor rejection rate with a credential check the behavioural
engine never contributed to.

So attempts are classified into three outcomes, and FAR/FRR are computed over
the behavioural ones only. PASSWORD_FAILURE attempts are reported with their
count, never silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

# The server returns this for a wrong password AND for an unknown username,
# deliberately: the two are indistinguishable to a caller, which is what stops
# the endpoint being used to enumerate accounts. Both mean the same thing here.
PASSWORD_FAILURE_REASONS = {"INVALID_CREDENTIALS"}

# Password was fine, but the attempt still never reached behavioural scoring.
NO_BEHAVIORAL_DECISION_REASONS = {"ACCOUNT_NOT_ENROLLED"}

ALLOWED = "ALLOW"
BLOCKED = "BLOCK"

PASSWORD_FAILURE = "PASSWORD_FAILURE"

# Below this a rate is not computed at all; three attempts do not make a
# percentage.
MIN_SAMPLES_FOR_A_RATE = 5

# The clean-protocol target for each human phase. Falling short does not
# suppress the rate, it marks the run as not meeting protocol.
CLEAN_PROTOCOL_MIN = 10

AUTOMATION_REASONS = {"AUTOMATION_DETECTED"}
INTEGRITY_REASONS = {
    "CHALLENGE_REUSED", "CHALLENGE_EXPIRED", "CHALLENGE_UNKNOWN",
    "CHALLENGE_WRONG_USER", "PHRASE_MISMATCH", "TIMESTAMP_INCONSISTENT",
    "MALFORMED_EVENT_STREAM",
}

# Kept in step with risk_engine.AUTOMATION_TIGHTENING. Only used to reconstruct
# the effective threshold for rows recorded before auth_attempts carried a
# threshold column; live rows report the stored value.
_AUTOMATION_TIGHTENING = 0.5


@dataclass(frozen=True)
class AttemptRecord:
    """One labelled attempt, as reported."""

    attempt_id: int
    phase: str
    at: float

    password_valid: bool
    behavioral_decision: str | None   # None when analysis never ran
    behavioral_score: float | None
    threshold: float | None
    rejection_reason: str | None
    bot_score: float | None
    latency_ms: float | None

    outcome: str                      # ALLOW | BLOCK | PASSWORD_FAILURE
    coverage: float | None = None
    signals: tuple[str, ...] = ()
    threshold_source: str = "recorded"  # or 'derived' for pre-migration rows

    def as_dict(self) -> dict:
        return asdict(self)


def classify(row: dict, phase: str, profile_threshold: float | None = None) -> AttemptRecord:
    """Label one auth_attempts row.

    `profile_threshold` is only consulted for rows written before the threshold
    was persisted; when it is used the record says so.
    """
    reason = row["reason"]
    password_valid = reason not in PASSWORD_FAILURE_REASONS

    if not password_valid:
        behavioral_decision = None
        outcome = PASSWORD_FAILURE
    elif reason in NO_BEHAVIORAL_DECISION_REASONS:
        behavioral_decision = None
        outcome = PASSWORD_FAILURE  # not a behavioural verdict either
    else:
        behavioral_decision = row["decision"]
        outcome = row["decision"]

    threshold = row.get("threshold")
    source = "recorded"
    if threshold is None and behavioral_decision is not None and profile_threshold:
        automation = row.get("automation_score")
        if automation is not None:
            threshold = profile_threshold * (
                1.0 - _AUTOMATION_TIGHTENING * min(1.0, max(0.0, automation))
            )
            source = "derived"

    signals: tuple[str, ...] = ()
    raw = row.get("reasons_json")
    if raw:
        signals = tuple(json.loads(raw))

    return AttemptRecord(
        attempt_id=row["id"],
        phase=phase,
        at=row["created_at"],
        password_valid=password_valid,
        behavioral_decision=behavioral_decision,
        behavioral_score=row.get("identity_score"),
        threshold=threshold,
        rejection_reason=(None if row["decision"] == ALLOWED else reason),
        bot_score=row.get("automation_score"),
        latency_ms=row.get("latency_ms"),
        outcome=outcome,
        coverage=row.get("coverage"),
        signals=signals,
        threshold_source=source,
    )


@dataclass
class PhaseSummary:
    label: str
    records: list[AttemptRecord]

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def behavioral(self) -> list[AttemptRecord]:
        """Attempts the behavioural engine actually judged."""
        return [r for r in self.records if r.behavioral_decision is not None]

    @property
    def password_failures(self) -> list[AttemptRecord]:
        return [r for r in self.records if r.outcome == PASSWORD_FAILURE]

    def rate(self, predicate) -> float | None:
        judged = self.behavioral
        if len(judged) < MIN_SAMPLES_FOR_A_RATE:
            return None
        return sum(1 for r in judged if predicate(r)) / len(judged)

    def as_dict(self) -> dict:
        return {
            "attempts_total": self.total,
            "behavioral_n": len(self.behavioral),
            "password_failures": len(self.password_failures),
            "password_failure_ids": [r.attempt_id for r in self.password_failures],
            "meets_clean_protocol": len(self.behavioral) >= CLEAN_PROTOCOL_MIN
            and not self.password_failures,
        }


def fmt_rate(rate: float | None, n: int) -> str:
    if rate is None:
        return f"insufficient data (n={n}, need {MIN_SAMPLES_FOR_A_RATE})"
    return f"{rate:6.1%}  (n={n})"
