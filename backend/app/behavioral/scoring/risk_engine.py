"""The decision engine.

Combines identity deviation, automation likelihood and integrity into one
ALLOW or BLOCK, with reason codes explaining which signal drove it.

Structure: hard gates first, then a fused identity decision.

The hard gates exist because some findings are categorical rather than
matters of degree. A reused nonce is not 40% of a replay. Automation past a
certain point is not a behavioural mismatch to be weighed against typing
rhythm; it is a different kind of finding and deserves its own verdict.

Below those gates, automation and identity genuinely fuse: an attempt that is
somewhat machine-like gets held to a tighter identity threshold than a clearly
human one. That is the multi-signal part, and it means there is no single
magic number deciding logins.

There is no OTP, no email, no SMS and no second channel anywhere in this file
or anywhere it calls. A mismatch results in BLOCK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.behavioral import stats
from app.behavioral.bot_detection.detector import AutomationResult
from app.behavioral.features.extractor import ExtractedFeatures
from app.behavioral.fingerprint.profile import BehaviorProfile
from app.behavioral.fingerprint.scoring import IdentityResult
from app.behavioral.scoring.integrity import IntegrityResult
from app.behavioral.scoring.reasons import ReasonCode, band, explain

# Past this, the attempt is reported as automation rather than as a
# behavioural mismatch. Set above the observed range for synthetic humans and
# below the observed range for every scripted fixture, with the gap in between
# left as margin rather than tuned to the fixtures.
AUTOMATION_BLOCK = 0.50

# The attempt must be comparable on at least this share of the profile's
# weighted features. Below it we decline to guess. This is not a second
# authentication factor: the user is asked for more of the same behaviour, not
# for a code from another channel.
MIN_IDENTITY_COVERAGE = 0.45

# How much a sub-blocking automation score tightens the identity threshold.
# At the gate value of 0.50 the threshold shrinks by a quarter.
AUTOMATION_TIGHTENING = 0.5

# A feature must account for at least this share of the total deviation before
# it is worth naming in an explanation.
EXPLANATION_SHARE_FLOOR = 0.08
MAX_EXPLAINED_SIGNALS = 4

_MODALITY_REASONS = {
    "keyboard": ReasonCode.KEYSTROKE_MISMATCH,
    "pointer": ReasonCode.POINTER_MISMATCH,
    "interaction": ReasonCode.INTERACTION_MISMATCH,
}

_MODALITY_LABELS = {
    "keyboard": "Typing rhythm",
    "pointer": "Pointer movement",
    "interaction": "Form interaction pattern",
}


@dataclass(frozen=True)
class Signal:
    code: str
    label: str
    band: Literal["LOW", "MEDIUM", "HIGH"]
    detail: str


@dataclass(frozen=True)
class RiskDecision:
    decision: Literal["ALLOW", "BLOCK"]
    reason: ReasonCode
    message: str
    identity_score: float | None = None
    # Components kept visible alongside the blend, because they fail
    # differently and an operator needs to see which one drove a decision.
    statistical_identity_score: float | None = None
    automation_score: float | None = None
    integrity_score: float = 0.0
    coverage: float | None = None
    threshold: float | None = None
    modality_scores: dict[str, float] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.decision == "ALLOW"


def decide(
    profile: BehaviorProfile,
    identity: IdentityResult,
    automation: AutomationResult,
    integrity: IntegrityResult,
    features: ExtractedFeatures,
) -> RiskDecision:
    """Produce the verdict for one authentication attempt.

    The identity score is the statistical fingerprint's, full stop. A blended
    statistical+ML score used to be accepted here; the ML term is retired (see
    app/behavioral/ml/__init__.py) and the behaviour is identical to what
    shipped, because its weight was 0.0.
    """

    # --- gate 1: integrity -------------------------------------------------
    # Ordered first because if the capture does not answer this challenge,
    # nothing computed from it means anything.
    if not integrity.ok:
        reason = integrity.reason or ReasonCode.MALFORMED_EVENT_STREAM
        return RiskDecision(
            decision="BLOCK",
            reason=reason,
            message=explain(reason),
            integrity_score=integrity.score,
            automation_score=automation.score,
            signals=[
                Signal(
                    code=reason.value,
                    label="Challenge integrity",
                    band=band(integrity.score),
                    detail=integrity.detail,
                )
            ],
        )

    automation_signals = [
        Signal(
            code=signal.code.value,
            label=_automation_label(signal.code),
            band=band(signal.strength),
            detail=signal.detail,
        )
        for signal in automation.signals
    ]

    # --- gate 2: automation ------------------------------------------------
    # Reported as its own verdict rather than folded into identity, because
    # "this is not a person" is a different finding from "this is not you".
    if automation.score >= AUTOMATION_BLOCK:
        return RiskDecision(
            decision="BLOCK",
            reason=ReasonCode.AUTOMATION_DETECTED,
            message=explain(ReasonCode.AUTOMATION_DETECTED),
            automation_score=automation.score,
            integrity_score=integrity.score,
            coverage=features.coverage,
            signals=automation_signals,
        )

    # --- gate 3: coverage --------------------------------------------------
    # Declining to guess is a real answer. Guessing from a fifth of the
    # profile would produce a decision nobody could defend.
    if not identity.is_comparable or identity.coverage < MIN_IDENTITY_COVERAGE:
        return RiskDecision(
            decision="BLOCK",
            reason=ReasonCode.INSUFFICIENT_SIGNAL,
            message=explain(ReasonCode.INSUFFICIENT_SIGNAL),
            automation_score=automation.score,
            integrity_score=integrity.score,
            coverage=identity.coverage,
            signals=automation_signals
            + [
                Signal(
                    code=ReasonCode.INSUFFICIENT_SIGNAL.value,
                    label="Captured behaviour",
                    band=band(1.0 - identity.coverage),
                    detail=(
                        f"only {identity.coverage:.0%} of the enrolled profile could be "
                        f"compared; {MIN_IDENTITY_COVERAGE:.0%} is required"
                    ),
                )
            ],
        )

    # --- fused identity decision -------------------------------------------
    effective_threshold = profile.threshold * (
        1.0 - AUTOMATION_TIGHTENING * stats.clamp(automation.score)
    )

    identity_signals = _explain_identity(identity, profile)
    all_signals = identity_signals + automation_signals

    decision_score = identity.score
    statistical_score = identity.score

    if decision_score > effective_threshold:
        return RiskDecision(
            decision="BLOCK",
            reason=_dominant_modality_reason(identity),
            message=explain(ReasonCode.BEHAVIORAL_MISMATCH),
            identity_score=decision_score,
            statistical_identity_score=statistical_score,
            automation_score=automation.score,
            integrity_score=integrity.score,
            coverage=identity.coverage,
            threshold=effective_threshold,
            modality_scores=identity.modality_scores,
            signals=all_signals,
        )

    return RiskDecision(
        decision="ALLOW",
        reason=ReasonCode.BEHAVIOR_MATCH,
        message=explain(ReasonCode.BEHAVIOR_MATCH),
        identity_score=decision_score,
        statistical_identity_score=statistical_score,
        automation_score=automation.score,
        integrity_score=integrity.score,
        coverage=identity.coverage,
        threshold=effective_threshold,
        modality_scores=identity.modality_scores,
        signals=all_signals,
    )


def _dominant_modality_reason(identity: IdentityResult) -> ReasonCode:
    """Name the modality that drove the block, when one clearly did."""
    if not identity.modality_scores:
        return ReasonCode.BEHAVIORAL_MISMATCH

    modality, score = max(identity.modality_scores.items(), key=lambda kv: kv[1])
    others = [v for k, v in identity.modality_scores.items() if k != modality]
    # Only single one out when it genuinely stands apart; otherwise the
    # mismatch is general and saying "typing rhythm" would be misleading.
    if others and score < max(others) * 1.4:
        return ReasonCode.BEHAVIORAL_MISMATCH
    return _MODALITY_REASONS.get(modality, ReasonCode.BEHAVIORAL_MISMATCH)


def _explain_identity(
    identity: IdentityResult, profile: BehaviorProfile
) -> list[Signal]:
    """Summarise the identity comparison by modality.

    Reported per modality rather than per feature on purpose. "Typing rhythm
    deviated" is something a user or a judge can understand; naming
    kbd_flight_negative_frac and its z-score would be both meaningless to them
    and a gift to an attacker tuning their mimicry.
    """
    signals = [
        Signal(
            code=_MODALITY_REASONS[modality].value,
            label=_MODALITY_LABELS[modality],
            band=band(score),
            detail=_modality_detail(modality, identity),
        )
        for modality, score in sorted(
            identity.modality_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        if modality in _MODALITY_REASONS
    ]
    return signals[:MAX_EXPLAINED_SIGNALS]


def _modality_detail(modality: str, identity: IdentityResult) -> str:
    share = sum(
        c.share for c in identity.contributions
        if c.modality == modality and c.share >= EXPLANATION_SHARE_FLOOR
    )
    return f"accounted for {share:.0%} of the measured difference"


def _automation_label(code: ReasonCode) -> str:
    return {
        ReasonCode.SYNTHETIC_EVENT_ORDER: "Event ordering",
        ReasonCode.DWELL_DEGENERACY: "Key hold times",
        ReasonCode.LOW_TIMING_VARIABILITY: "Timing variability",
        ReasonCode.AUTOMATION_TIMING: "Timing regularity",
        ReasonCode.POINTER_ANOMALY: "Pointer movement",
        ReasonCode.IMPOSSIBLE_INTERACTION_SPEED: "Interaction speed",
        ReasonCode.UNTRUSTED_EVENTS: "Automation markers",
    }.get(code, "Automation signal")
