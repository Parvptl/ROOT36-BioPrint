"""Confidence-gated adaptive profile updates.

Behaviour drifts. A user changes keyboard, recovers from an injury, gets faster
with a phrase style, learns the form. A profile fitted once at enrollment slowly
stops describing them, and false rejections climb.

Adaptation fixes that, and introduces the worst failure mode in the system if
done carelessly:

    attacker is accepted -> system learns attacker -> attacker becomes baseline

Everything in this module exists to make that path unavailable. Four independent
protections, because any single one can be wrong:

1. **Confidence gate.** Only a HIGH-confidence accept updates anything. Not a
   plain ALLOW — an ALLOW with margin to spare on every signal.
2. **Drift clamp.** One session can move a feature by at most a fraction of its
   own scale, so even a wrongly-admitted session cannot yank the profile.
3. **Two timescales.** The long-term track moves ten times slower than the
   recent one, so a burst of bad sessions cannot erase stable identity.
4. **Audited refusals.** Declined updates are recorded too, so "this profile
   stopped moving" and "this profile moved suspiciously often" are both
   visible after the fact.

The scoring path does not know adaptation exists. It reads `median`; this
module maintains the two tracks behind it and recomputes that blend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.behavioral.fingerprint.profile import BehaviorProfile, FeatureStat, Maturity

# --- confidence tiers ------------------------------------------------------


class Confidence(StrEnum):
    HIGH = "HIGH"              # accepted with margin on every signal -> may update
    MEDIUM = "MEDIUM"          # accepted, but not comfortably       -> no update
    SUSPICIOUS = "SUSPICIOUS"  # blocked on identity or coverage     -> no update
    BLOCKED = "BLOCKED"        # blocked on integrity/credential     -> no update
    BOT = "BOT"                # automation detected                 -> no update


# A HIGH-confidence accept must clear the threshold by a real margin, not
# scrape under it. At 0.6 the attempt scored at most 60% of the bar.
HIGH_CONFIDENCE_IDENTITY_RATIO = 0.6

# ...and must look clearly human. Well below the 0.50 automation block gate.
HIGH_CONFIDENCE_MAX_AUTOMATION = 0.20

# ...and must have compared most of the profile. Adapting from a thin capture
# would move features that were barely measured.
HIGH_CONFIDENCE_MIN_COVERAGE = 0.70


# --- update rates ----------------------------------------------------------
#
# Both calibrated in evaluation/adaptation_sweep.py against simulated drift
# rather than chosen by feel. See that script for the measurement.

# Recent track: follows drift.
ALPHA_RECENT = 0.25

# Long track: slow reference, an order of magnitude below the recent one.
ALPHA_LONG = 0.015

# Effective centre = LAMBDA_LONG * long + (1 - LAMBDA_LONG) * recent.
LAMBDA_LONG = 0.4

# On why the recent track is weighted higher than the long one, which looks
# backwards for a "long-term holds identity" design:
#
# The sweep measured poisoning resistance as essentially flat across every
# rate combination tried (genuine score 0.118 to 0.138 after fifty forced
# impostor updates, against a threshold near 0.21). The cumulative anchor —
# not the rate split — is what bounds a poisoning campaign. That leaves the
# rates free to be chosen on drift performance, where they matter a great deal:
#
#     no adaptation          35.0% of drifting genuine sessions rejected
#     0.15 / 0.6             20.6%
#     0.15 / 0.4             17.8%
#     0.25 / 0.4             14.4%   <- shipped
#
# So the division of labour is: the anchor holds identity, and the two tracks
# shape how tracking moves inside the neighbourhood it permits.
#
# Caveat: these runs draw random phrases, and repeat runs of the same setting
# varied by several points (0.15/0.6 measured 33.9% and 20.6% on two runs).
# 0.25/0.4 was best on both, which is why it was chosen, but the margin between
# adjacent settings is inside the noise.

# Hard cap on how far ONE session may move a feature, in units of that
# feature's own scale. Even if the gate is wrong, a single accepted session
# cannot relocate the profile.
MAX_DRIFT_PER_UPDATE = 0.35

# Hard cap on how far the profile may EVER travel from the enrolled value.
#
# Added because the per-update clamp alone was measured to be insufficient. Ten
# consecutively-accepted impostor sessions each moved a feature by a legal
# 0.35 scale units, and the accumulated drift pushed the genuine user's own
# score from 0.064 to 0.275 against a threshold of 0.273 — the real user was
# locked out of their own account by a poisoning campaign that never once broke
# the per-step rule.
#
# Bounding the per-step size does not bound the walk. This bounds the walk.
# The cost is that genuine drift beyond this radius needs re-enrollment, which
# is the correct trade for an authentication control.
MAX_TOTAL_DRIFT = 0.75


@dataclass(frozen=True)
class AdaptationOutcome:
    """What happened, and the evidence for it. Recorded whether or not applied."""

    applied: bool
    confidence: Confidence
    reason: str
    features_updated: int = 0
    from_version: int = 1
    to_version: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_updated": self.applied,
            "confidence": self.confidence.value,
            "reason": self.reason,
            "features_updated": self.features_updated,
            "profile_version": self.to_version,
        }


def classify_confidence(
    decision: str,
    reason: str,
    identity_score: float | None,
    automation_score: float | None,
    coverage: float | None,
    threshold: float | None,
) -> Confidence:
    """Grade an attempt. Only HIGH is ever allowed to teach the profile.

    Ordered so the most dangerous classifications win: an automated attempt is
    BOT even if it somehow scored well on identity.
    """
    automation = automation_score or 0.0

    if reason == "AUTOMATION_DETECTED" or automation >= HIGH_CONFIDENCE_MAX_AUTOMATION * 2.5:
        return Confidence.BOT

    if decision != "ALLOW":
        # Distinguish "we think this is the wrong person" from "we could not
        # trust the evidence at all". Neither adapts, but they are different
        # findings and the audit trail should say which.
        if reason in {"BEHAVIORAL_MISMATCH", "KEYSTROKE_MISMATCH",
                      "POINTER_MISMATCH", "INTERACTION_MISMATCH",
                      "INSUFFICIENT_SIGNAL"}:
            return Confidence.SUSPICIOUS
        return Confidence.BLOCKED

    if identity_score is None or threshold is None or threshold <= 0:
        return Confidence.MEDIUM

    comfortable = identity_score <= threshold * HIGH_CONFIDENCE_IDENTITY_RATIO
    human = automation <= HIGH_CONFIDENCE_MAX_AUTOMATION
    well_covered = (coverage or 0.0) >= HIGH_CONFIDENCE_MIN_COVERAGE

    if comfortable and human and well_covered:
        return Confidence.HIGH
    return Confidence.MEDIUM


def adapt_profile(
    profile: BehaviorProfile,
    session_features: dict[str, float],
    confidence: Confidence,
) -> tuple[BehaviorProfile, AdaptationOutcome]:
    """Return a possibly-updated profile plus an account of what was done.

    Pure: takes a profile, returns a new one. Persistence is the caller's job,
    which keeps the policy testable without a database.
    """
    if confidence is not Confidence.HIGH:
        return profile, AdaptationOutcome(
            applied=False,
            confidence=confidence,
            reason=f"{confidence.value} confidence does not qualify for adaptation",
            from_version=profile.version,
            to_version=profile.version,
        )

    updated: dict[str, FeatureStat] = {}
    changed = 0

    for name, stat in profile.features.items():
        observed = session_features.get(name)
        if observed is None:
            # Feature absent from this capture. Leaving it untouched is the
            # point: an unobserved feature must not decay toward anything.
            updated[name] = stat
            continue

        long_track = stat.median_long if stat.median_long is not None else stat.median
        recent_track = stat.median_recent if stat.median_recent is not None else stat.median

        new_long = (1 - ALPHA_LONG) * long_track + ALPHA_LONG * observed
        new_recent = (1 - ALPHA_RECENT) * recent_track + ALPHA_RECENT * observed

        blended = LAMBDA_LONG * new_long + (1 - LAMBDA_LONG) * new_recent

        # Tighter clamp during COLD_START and WARMING
        effective_max_drift = MAX_DRIFT_PER_UPDATE
        effective_total_drift = MAX_TOTAL_DRIFT
        if profile.maturity in (Maturity.COLD_START, Maturity.WARMING):
            effective_max_drift *= 0.5
            effective_total_drift *= 0.5

        # Two clamps, both in units of the feature's own scale.
        # Per-step: no single session lurches the profile.
        step_limit = effective_max_drift * stat.scale
        delta = max(-step_limit, min(step_limit, blended - stat.median))
        new_median = stat.median + delta

        # Cumulative: no sequence of sessions walks it away from enrollment.
        anchor = stat.median_enrolled if stat.median_enrolled is not None else stat.median
        total_limit = effective_total_drift * stat.scale
        new_median = max(
            anchor - total_limit, min(anchor + total_limit, new_median)
        )

        if new_median != stat.median:
            changed += 1

        updated[name] = FeatureStat(
            name=stat.name,
            modality=stat.modality,
            median=new_median,
            median_long=new_long,
            median_recent=new_recent,
            median_enrolled=anchor,
            # Scale, MAD, weight and coverage are NOT adapted. Letting the
            # scale drift would let a run of consistent sessions tighten the
            # profile until ordinary variation starts failing, which is a
            # self-inflicted lockout rather than a security gain.
            mad=stat.mad,
            scale=stat.scale,
            weight=stat.weight,
            coverage=stat.coverage,
        )

    # Maturity state progression
    total_sessions = profile.update_count + 1 + profile.session_count
    next_maturity = profile.maturity
    
    # We do NOT progress to MATURE here. Graduation to MATURE requires a full 
    # recalibration event which is triggered separately once 8 sessions are collected.
    if profile.maturity != Maturity.MATURE:
        if total_sessions >= 5:
            next_maturity = Maturity.ESTABLISHED
        elif total_sessions >= 2:
            next_maturity = Maturity.WARMING
        
    next_version = profile.version + 1
    adapted = BehaviorProfile(
        features=updated,
        session_count=profile.session_count,
        population_size=profile.population_size,
        population_informative=profile.population_informative,
        threshold=profile.threshold,
        threshold_source=profile.threshold_source,
        calibration=profile.calibration,
        maturity=next_maturity,
        version=next_version,
        update_count=profile.update_count + 1,
    )

    return adapted, AdaptationOutcome(
        applied=True,
        confidence=confidence,
        reason="high-confidence genuine session",
        features_updated=changed,
        from_version=profile.version,
        to_version=next_version,
    )
