#!/usr/bin/env python3
"""How many genuine logins does a two-capture profile need to mature?

The cold-start threshold and the adaptation gate are coupled: a session earns
HIGH confidence only when it scores at or below
HIGH_CONFIDENCE_IDENTITY_RATIO (0.6) of the profile's threshold, and only a
HIGH session teaches the profile. So tightening the cold-start threshold for
security also raises the bar for maturation, and a threshold chosen purely on
false acceptance could leave profiles stuck in a weak state for longer.

That interaction is measured here rather than assumed. Generated typists.

    cd backend
    ./.venv/Scripts/python.exe evaluation/maturation_speed.py
"""

from __future__ import annotations

import argparse
import logging
import statistics as st
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.fingerprint.adaptation import (  # noqa: E402
    Confidence,
    adapt_profile,
    classify_confidence,
)
from app.behavioral.fingerprint.aalto_prior import aalto_population_prior  # noqa: E402
from app.behavioral.fingerprint.calibration import cold_start_threshold  # noqa: E402
from app.behavioral.fingerprint.profile import (  # noqa: E402
    BehaviorProfile,
    Maturity,
    fit_early_profile,
)
from app.behavioral.fingerprint.scoring import score_identity  # noqa: E402
from evaluation.prior_ablation import REAL_PRIOR, build_population  # noqa: E402

GRADUATION_SESSIONS = 8  # routes_login rebuilds a mature profile at this many


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--logins", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--thresholds", type=float, nargs="*", default=None,
                        help="cold-start thresholds to compare (default: shipped)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    prior = aalto_population_prior(REAL_PRIOR)
    people = build_population(args.users, args.logins, args.seed)

    thresholds = args.thresholds or [0.20, cold_start_threshold().threshold]

    print("MATURATION SPEED — GENERATED TYPISTS")
    print(f"  users={args.users}  logins simulated={args.logins}")
    print(f"  a profile graduates once it has accumulated {GRADUATION_SESSIONS} "
          f"high-confidence sessions")
    print()
    header = (f"{'cold thr':>9s} {'HIGH rate':>10s} {'median logins':>14s} "
              f"{'p90 logins':>11s} {'never matured':>14s}")
    print(header)
    print("-" * len(header))

    for threshold in thresholds:
        logins_needed: list[int] = []
        never = 0
        high_total = attempts_total = 0

        for person in people:
            profile = BehaviorProfile(
                features=fit_early_profile(person.enrollment[:2], prior),
                session_count=2,
                population_size=prior.sample_count,
                population_informative=prior.is_informative,
                threshold=threshold,
                threshold_source="cold_start_prior",
                maturity=Maturity.WARMING,
            )
            # Enrollment already contributed 2 sessions toward graduation.
            accumulated = 2
            matured_at = None

            for n, features in enumerate(person.genuine, start=1):
                result = score_identity(profile, features)
                attempts_total += 1
                confidence = classify_confidence(
                    decision="ALLOW" if result.score <= profile.threshold else "BLOCK",
                    reason="BEHAVIOR_MATCH" if result.score <= profile.threshold
                    else "BEHAVIORAL_MISMATCH",
                    identity_score=result.score,
                    automation_score=0.0,
                    coverage=result.coverage,
                    threshold=profile.threshold,
                )
                if confidence is Confidence.HIGH:
                    high_total += 1
                    accumulated += 1
                    profile, _outcome = adapt_profile(
                        profile, features, confidence
                    )
                    if accumulated >= GRADUATION_SESSIONS:
                        matured_at = n
                        break

            if matured_at is None:
                never += 1
            else:
                logins_needed.append(matured_at)

        rate = high_total / attempts_total if attempts_total else 0.0
        if logins_needed:
            ordered = sorted(logins_needed)
            median = st.median(ordered)
            p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
        else:
            median = p90 = float("nan")

        print(f"{threshold:9.3f} {rate:10.1%} {median:14.1f} {p90:11.0f} "
              f"{never:>8d}/{len(people)}")

    print()
    print("A lower cold-start threshold buys false-acceptance resistance and")
    print("costs maturation speed, because only a HIGH-confidence session")
    print("teaches the profile and HIGH is defined relative to the threshold.")


if __name__ == "__main__":
    main()
