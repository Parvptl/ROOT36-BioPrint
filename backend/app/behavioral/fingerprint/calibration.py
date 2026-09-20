"""Choosing the decision threshold.

The threshold is derived from data, not picked. Two distributions feed it:

* Genuine scores, from leave-one-session-out over the enrollment rounds. Each
  round is scored against a profile fitted from the *other* rounds, so no
  session is ever scored against a baseline that contains it.

* Impostor scores, from the population samples scored against this profile.
  These are other people, so they stand in for the attacker who has the
  password but not the behaviour.

Both are small. The honest consequence is that the threshold is a reasonable
operating point, not a statistically tight one, and every result carries the
sample sizes that produced it. Where there is not enough data to calibrate, the
profile records that it fell back rather than implying a calibration happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.behavioral import stats
from app.behavioral.fingerprint.population import PopulationPrior
from app.behavioral.fingerprint.profile import BehaviorProfile, fit_profile, Maturity
from app.behavioral.fingerprint.scoring import score_identity

# Enough impostor scores to attempt a real sweep. Below this, a threshold
# "optimised" against the sample would just be fitting three numbers.
MIN_IMPOSTOR_SCORES = 5
MIN_GENUINE_SCORES = 3

# Multiplier on the genuine spread when no impostor data exists. Three robust
# scale units above the genuine centre: wide enough to absorb an off day,
# narrow enough to still mean something.
GENUINE_ONLY_K = 3.0

# Extra margin above the worst leave-one-out fold, in units of the genuine
# spread. Each fold is fitted on one fewer round than the real profile, so its
# scores are systematically a little pessimistic; this covers that gap and
# nothing more. Expressed relative to the user's own spread rather than as a
# flat fraction, so a consistent user gets a tight threshold and a variable one
# gets a looser threshold that they have earned.
LOO_PESSIMISM_MARGIN = 1.5

# Hard bounds. Below the lower bound almost every genuine login fails; above
# the upper bound almost nothing is ever rejected. Either would be a broken
# product regardless of what the arithmetic suggested.
MIN_THRESHOLD = 0.08
MAX_THRESHOLD = 0.80

# Used only when there is no usable data at all. Deliberately permissive: a
# freshly enrolled user being unable to log in is a worse first impression than
# a slightly loose first session, and the profile is flagged as uncalibrated.
FALLBACK_THRESHOLD = 0.45


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    source: str  # calibrated | genuine_only | fallback_prior
    genuine_scores: list[float] = field(default_factory=list)
    impostor_scores: list[float] = field(default_factory=list)
    note: str = ""
    metrics: dict[str, object] = field(default_factory=dict)

    def as_json(self) -> dict[str, object]:
        return {
            "source": self.source,
            "threshold": self.threshold,
            "genuine_scores": [round(s, 5) for s in self.genuine_scores],
            "impostor_scores": [round(s, 5) for s in self.impostor_scores],
            "note": self.note,
            "metrics": self.metrics,
        }


def leave_one_session_out_scores(
    sessions: list[dict[str, float]], prior: PopulationPrior
) -> list[float]:
    """Score each enrollment round against a profile built without it.

    This is the only way to get an unbiased genuine-score estimate from
    enrollment data. Scoring a session against a profile that includes it would
    report a deviation near zero and a uselessly tight threshold.
    """
    if len(sessions) < 2:
        return []

    scores: list[float] = []
    for index in range(len(sessions)):
        held_out = sessions[index]
        remaining = sessions[:index] + sessions[index + 1 :]
        features = fit_profile(remaining, prior)
        if not features:
            continue
        fold = BehaviorProfile(features=features, session_count=len(remaining))
        result = score_identity(fold, held_out)
        if result.is_comparable:
            scores.append(result.score)

    return scores


def _impostor_scores(
    profile: BehaviorProfile, population_samples: list[dict[str, float]]
) -> list[float]:
    scores: list[float] = []
    for sample in population_samples:
        result = score_identity(profile, sample)
        if result.is_comparable:
            scores.append(result.score)
    return scores


def _sweep(genuine: list[float], impostor: list[float]) -> tuple[float, dict[str, object]]:
    """Pick the cut point minimising false rejections plus false acceptances.

    Ties break toward the lower threshold, which prefers rejecting a genuine
    user over admitting an impostor. That is the right bias for an
    authentication control, and the retry cost to a real user is one more
    attempt.
    """
    candidates = sorted({*genuine, *impostor})
    if not candidates:
        return FALLBACK_THRESHOLD, {}

    # Midpoints between observed scores, plus a little headroom either side.
    cuts = [max(MIN_THRESHOLD, candidates[0] * 0.5)]
    cuts.extend((a + b) / 2 for a, b in zip(candidates[:-1], candidates[1:]))
    cuts.append(min(MAX_THRESHOLD, candidates[-1] * 1.2 + 0.02))

    genuine_arr = np.asarray(genuine, dtype=float)
    impostor_arr = np.asarray(impostor, dtype=float)

    best_cut = FALLBACK_THRESHOLD
    best_cost = float("inf")
    best_rates = (1.0, 1.0)

    for cut in cuts:
        cut = float(np.clip(cut, MIN_THRESHOLD, MAX_THRESHOLD))
        frr = float(np.mean(genuine_arr > cut)) if genuine_arr.size else 0.0
        far = float(np.mean(impostor_arr <= cut)) if impostor_arr.size else 0.0
        cost = frr + far
        if cost < best_cost - 1e-12:
            best_cost, best_cut, best_rates = cost, cut, (frr, far)

    metrics = {
        "loo_false_rejection_rate": round(best_rates[0], 4),
        "population_false_acceptance_rate": round(best_rates[1], 4),
        "genuine_n": len(genuine),
        "impostor_n": len(impostor),
        "separation": round(
            float(np.median(impostor_arr) - np.median(genuine_arr)), 4
        )
        if impostor_arr.size and genuine_arr.size
        else None,
    }
    return best_cut, metrics


def calibrate(
    profile_features: dict,
    sessions: list[dict[str, float]],
    prior: PopulationPrior,
    population_samples: list[dict[str, float]],
) -> CalibrationResult:
    """Derive a threshold for a freshly fitted profile."""
    genuine = leave_one_session_out_scores(sessions, prior)

    provisional = BehaviorProfile(features=profile_features, session_count=len(sessions))
    impostor = _impostor_scores(provisional, population_samples)

    return derive_threshold(genuine, impostor)


def derive_threshold(
    genuine: list[float], impostor: list[float]
) -> CalibrationResult:
    """Pick an operating point from genuine and impostor score distributions.

    Split out of calibrate() so the hybrid path can reuse it verbatim. When ML
    evidence is blended into the identity score, the threshold has to be
    derived from the *blended* scores; reusing this function is what keeps the
    two paths from drifting into different threshold policies.
    """
    if len(genuine) >= MIN_GENUINE_SCORES and len(impostor) >= MIN_IMPOSTOR_SCORES:
        threshold, metrics = _sweep(genuine, impostor)
        note = (
            f"Threshold swept against {len(genuine)} leave-one-out genuine scores and "
            f"{len(impostor)} population impostor scores. Both samples are small, so "
            f"the reported rates are indicative and carry wide uncertainty."
        )
        source = "calibrated"

    elif len(genuine) >= MIN_GENUINE_SCORES:
        genuine_arr = np.asarray(genuine, dtype=float)
        centre = stats.median(genuine_arr)
        spread = stats.robust_scale(genuine_arr)
        observed_max = float(np.max(genuine_arr))
        # Sit above the worst enrollment fold, but only by a margin proportional
        # to the spread the user actually showed.
        #
        # An earlier version added a flat 40 percent of the observed maximum
        # here, to stop genuine logins against fresh phrases being rejected.
        # That was the wrong fix and it was measured doing real harm: it pushed
        # the threshold to 0.46, where a moderately different impostor scoring
        # 0.24 was accepted every time. The genuine-login problem was never the
        # threshold, it was features whose scale collapsed below their own
        # measurement resolution, which the registry noise floors now prevent.
        #
        # What remains is honest headroom for leave-one-out pessimism: each
        # fold fits on four rounds rather than five, so its scores run slightly
        # high relative to a real login against the full profile.
        threshold = float(
            np.clip(
                max(
                    centre + GENUINE_ONLY_K * spread,
                    observed_max + LOO_PESSIMISM_MARGIN * max(spread, 0.02),
                ),
                MIN_THRESHOLD,
                MAX_THRESHOLD,
            )
        )
        metrics = {
            "loo_false_rejection_rate": round(
                float(np.mean(np.asarray(genuine) > threshold)), 4
            ),
            "population_false_acceptance_rate": None,
            "genuine_n": len(genuine),
            "impostor_n": len(impostor),
            "separation": None,
        }
        note = (
            f"No population data was available ({len(impostor)} impostor scores, "
            f"{MIN_IMPOSTOR_SCORES} needed). The threshold is set from the spread of "
            f"{len(genuine)} genuine scores alone, so the false-acceptance rate for this "
            f"profile is unmeasured."
        )
        source = "genuine_only"

    else:
        threshold = FALLBACK_THRESHOLD
        metrics = {
            "loo_false_rejection_rate": None,
            "population_false_acceptance_rate": None,
            "genuine_n": len(genuine),
            "impostor_n": len(impostor),
            "separation": None,
        }
        note = (
            f"Too few usable enrollment sessions ({len(genuine)}) to calibrate. Using a "
            f"conservative default threshold. This profile's accuracy is unmeasured; "
            f"re-enroll with more rounds for a calibrated threshold."
        )
        source = "fallback_prior"

    return CalibrationResult(
        threshold=threshold,
        source=source,
        genuine_scores=genuine,
        impostor_scores=impostor,
        note=note,
        metrics=metrics,
    )


def build_calibrated_profile(
    sessions: list[dict[str, float]],
    prior: PopulationPrior,
    population_samples: list[dict[str, float]],
) -> BehaviorProfile:
    """Fit a profile and attach its calibrated threshold."""
    features = fit_profile(sessions, prior)
    result = calibrate(features, sessions, prior, population_samples)

    return BehaviorProfile(
        features=features,
        session_count=len(sessions),
        population_size=prior.sample_count,
        population_informative=prior.is_informative,
        threshold=result.threshold,
        threshold_source=result.source,
        calibration=result.as_json(),
        maturity=Maturity.MATURE,
    )

import os

# Cold-start operating point, for one- and two-capture profiles.
#
# These profiles have no leave-one-out folds to calibrate against, so the
# threshold is static. It was previously 0.20, chosen for the single-capture
# distribution, and measured at roughly 1% false rejection and 28% false
# acceptance. Twenty-eight percent is not an operating point; it means a
# freshly enrolled account admits more than a quarter of strangers who hold the
# password.
#
# Recalibrated in evaluation/enrollment_size_ablation.py against a 10% false-
# acceptance budget, on 60 users, reported on 60 disjoint held-out users, three
# seeds. The derived cut was 0.129 / 0.143 / 0.140; 0.14 sits inside that range.
#
# Held-out effect at two captures, mean of three seeds:
#     threshold 0.20  ->  false rejection  0.2%   false acceptance 34.8%
#     threshold 0.14  ->                   4.6%                    11.9%
#
# This TIGHTENS the gate. It costs genuine users roughly one retry in twenty
# during a transitional state that ends as soon as the profile matures, and
# there is no account lockout, so a retry is cheap. Cutting cold-start false
# acceptance threefold is worth that.
#
# Generated typists, so the number is a calibrated starting point rather than a
# measured real-world operating point. Override with
# BIOPRINT_COLD_START_THRESHOLD.
COLD_START_THRESHOLD = 0.14


def cold_start_threshold(threshold: float = None) -> CalibrationResult:
    """The static threshold for a one- or two-capture profile.

    With one or two observations there is no empirical dispersion to calibrate
    against, so leave-one-out is unavailable and the cut is fixed. See
    COLD_START_THRESHOLD for how the value was derived.
    """
    if threshold is None:
        threshold = float(
            os.environ.get("BIOPRINT_COLD_START_THRESHOLD", str(COLD_START_THRESHOLD))
        )

    return CalibrationResult(
        threshold=threshold,
        source="cold_start_prior",
        genuine_scores=[],
        impostor_scores=[],
        note=(
            f"Static cold-start threshold {threshold}. One or two captures give a "
            f"personal centre but no dispersion estimate, so the scale comes from "
            f"the population prior and the cut is fixed rather than calibrated "
            f"per user. Adaptation personalises this as genuine logins arrive."
        ),
        metrics={"loo_false_rejection_rate": None, "population_false_acceptance_rate": None}
    )
