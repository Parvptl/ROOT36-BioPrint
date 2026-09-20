#!/usr/bin/env python3
"""Which features change under the real prior, and does any one dominate?

The real Aalto prior differs substantially from the synthetic bootstrap on the
dwell family, burst fraction and negative-flight fraction. A population scale
feeds two places in the model — the scale floor and the discriminability
weight — so a large change there can quietly hand one feature most of the
decision. This checks that it has not.

`kbd_shift_right_ratio` is genuinely unobserved in Aalto. It must fall back to
the relative prior, never to a fabricated population variability.

Generated typists. Read-only; changes nothing.

    cd backend
    ./.venv/Scripts/python.exe evaluation/prior_feature_effect.py
"""

from __future__ import annotations

import argparse
import logging
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.fingerprint.aalto_prior import aalto_population_prior  # noqa: E402
from app.behavioral.fingerprint.profile import BehaviorProfile, fit_early_profile  # noqa: E402
from app.behavioral.fingerprint.scoring import score_identity  # noqa: E402
from evaluation.prior_ablation import (  # noqa: E402
    REAL_PRIOR,
    SYNTHETIC_PRIOR,
    build_population,
)

WATCH = (
    "kbd_dwell_median", "kbd_dwell_right_median", "kbd_dwell_left_median",
    "kbd_dwell_mad", "kbd_speed_kps", "kbd_burst_fraction",
    "kbd_flight_negative_frac", "kbd_shift_right_ratio",
)
SHIFT = "kbd_shift_right_ratio"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    people = build_population(args.users, 6, args.seed)

    priors = {
        "SYNTHETIC": aalto_population_prior(SYNTHETIC_PRIOR),
        "REAL": aalto_population_prior(REAL_PRIOR),
    }

    print("FEATURE EFFECT OF THE POPULATION PRIOR (generated typists)")
    print(f"  {args.users} users, 2-capture profiles, identical enrollment sessions")
    print()

    # --- scale and weight, averaged over users ---------------------------
    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    shares: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for label, prior in priors.items():
        for person in people:
            profile = BehaviorProfile(
                features=fit_early_profile(person.enrollment[:2], prior),
                session_count=2, threshold=0.14,
            )
            for name, stat in profile.features.items():
                stats[name][f"{label}_scale"].append(stat.scale)
                stats[name][f"{label}_weight"].append(stat.weight)

            # How much of the actual deviation each feature accounts for, on a
            # genuine attempt. This is the number that matters: a big scale
            # change is harmless if the feature still contributes little.
            for features in person.genuine[:3]:
                result = score_identity(profile, features)
                for c in result.contributions:
                    shares[c.name][label].append(c.share)

    header = (f"{'feature':<28s} {'syn scale':>10s} {'real scale':>10s} {'ratio':>7s} "
              f"{'syn w':>7s} {'real w':>7s} {'syn share':>10s} {'real share':>11s}")
    print(header)
    print("-" * len(header))

    def row(name: str) -> None:
        d = stats[name]
        if not d:
            print(f"{name:<28s} {'not modelled':>60s}")
            return
        ss, rs = st.mean(d["SYNTHETIC_scale"]), st.mean(d["REAL_scale"])
        sw, rw = st.mean(d["SYNTHETIC_weight"]), st.mean(d["REAL_weight"])
        s_sh = st.mean(shares[name]["SYNTHETIC"]) if shares[name]["SYNTHETIC"] else 0.0
        r_sh = st.mean(shares[name]["REAL"]) if shares[name]["REAL"] else 0.0
        print(f"{name:<28s} {ss:10.4f} {rs:10.4f} {rs/ss if ss else float('inf'):7.3f} "
              f"{sw:7.4f} {rw:7.4f} {s_sh:9.1%} {r_sh:10.1%}")

    print("  -- features the real prior changes most --")
    for name in WATCH:
        row(name)

    print()
    print("  -- every other modelled feature --")
    for name in sorted(stats):
        if name not in WATCH:
            row(name)

    # --- concentration check ---------------------------------------------
    print()
    print("=" * 78)
    print("  CONCENTRATION: is any single feature dominating the decision?")
    print("=" * 78)
    for label in ("SYNTHETIC", "REAL"):
        means = {n: st.mean(v[label]) for n, v in shares.items() if v[label]}
        if not means:
            continue
        top = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"  {label:<10s} top contributors: " +
              ", ".join(f"{n.replace('kbd_','').replace('ptr_','')} {s:.0%}"
                        for n, s in top))
        print(f"  {'':10s} largest single share {top[0][1]:.1%}  "
              f"(saturation caps any one axis at rho=9, so no feature can "
              f"outvote the rest)")

    # --- the unobserved feature -------------------------------------------
    print()
    print("=" * 78)
    print("  UNOBSERVED FEATURE")
    print("=" * 78)
    for label, prior in priors.items():
        in_scales = SHIFT in prior.scales
        fallback = prior.scale_for(SHIFT, 0.4)
        print(f"  {label:<10s} {SHIFT} in prior scales: {in_scales}  "
              f"scale_for(own_median=0.4) = {fallback:.4f}")
    print(f"  {'':10s} relative-prior fallback would be 0.35 x 0.4 = 0.1400")
    print()
    d = stats[SHIFT]
    if d:
        print(f"  Still modelled from the user's own captures under both priors "
              f"(mean scale synthetic {st.mean(d['SYNTHETIC_scale']):.4f}, "
              f"real {st.mean(d['REAL_scale']):.4f}).")
        print("  Aalto not measuring it does not stop the browser measuring it.")


if __name__ == "__main__":
    main()
