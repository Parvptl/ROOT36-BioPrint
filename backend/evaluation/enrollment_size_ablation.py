#!/usr/bin/env python3
"""How many enrollment captures does an initial profile actually need?

The product question: 8 dedicated enrollment rounds is real friction. Can one
or two natural interactions carry the cold start instead?

    A   1 capture   (production COLD_START path, fit_cold_start_profile)
    B   2 captures  (proposed, fit_early_profile)
    C   8 captures  (research baseline, fit_profile + LOO calibration)

Same methodology as the Phase D prior ablation so the two are comparable: the
same generated-typist population, the same production scoring, the same
coverage gate, the same user-level bootstrap. The independent variable here is
the number of enrollment captures, and the prior is held fixed.

These are GENERATED TYPISTS. The result describes how the model responds to
enrollment size under controlled conditions, not accuracy on real people.

    cd backend
    ./.venv/Scripts/python.exe evaluation/enrollment_size_ablation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.fingerprint.aalto_prior import aalto_population_prior  # noqa: E402
from app.behavioral.fingerprint.calibration import (  # noqa: E402
    build_calibrated_profile,
    cold_start_threshold,
)
from app.behavioral.fingerprint.profile import (  # noqa: E402
    BehaviorProfile,
    Maturity,
    fit_cold_start_profile,
    fit_early_profile,
)
from app.behavioral.fingerprint.scoring import score_identity  # noqa: E402
from app.behavioral.scoring.risk_engine import MIN_IDENTITY_COVERAGE  # noqa: E402
from evaluation.prior_ablation import (  # noqa: E402
    REAL_PRIOR,
    SYNTHETIC_PRIOR,
    bootstrap_ci,
    build_population,
    describe,
    eer,
    roc_auc,
)

CONDITIONS = (
    (1, "1 capture", Maturity.COLD_START),
    (2, "2 captures", Maturity.WARMING),
    (8, "8 captures", Maturity.MATURE),
)


def build(enrollment: list[dict[str, float]], prior, population_samples):
    """Build a profile the way production would for this enrollment size."""
    n = len(enrollment)
    if n == 1:
        cal = cold_start_threshold()
        return BehaviorProfile(
            features=fit_cold_start_profile(enrollment[0], prior),
            session_count=1,
            population_size=prior.sample_count,
            population_informative=prior.is_informative,
            threshold=cal.threshold,
            threshold_source=cal.source,
            calibration=cal.as_json(),
            maturity=Maturity.COLD_START,
        )
    if n < 3:
        cal = cold_start_threshold()
        return BehaviorProfile(
            features=fit_early_profile(enrollment, prior),
            session_count=n,
            population_size=prior.sample_count,
            population_informative=prior.is_informative,
            threshold=cal.threshold,
            threshold_source=cal.source,
            calibration=cal.as_json(),
            maturity=Maturity.WARMING,
        )
    return build_calibrated_profile(enrollment, prior, population_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=120)
    parser.add_argument("--genuine-per-user", type=int, default=10)
    parser.add_argument("--impostors-per-user", type=int, default=10)
    parser.add_argument("--population-donors", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--far-budget", type=float, default=0.05,
                        help="cold-start false-acceptance budget used for "
                             "threshold calibration")
    parser.add_argument("--prior", type=Path, default=REAL_PRIOR,
                        help="held fixed across conditions")
    parser.add_argument("--out", type=Path,
                        default=_BACKEND.parent / "evaluation" / "out"
                        / "phase_f_enrollment_size.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)

    prior = aalto_population_prior(args.prior)
    if prior is None:
        raise SystemExit(f"could not load prior: {args.prior}")

    print("ENROLLMENT SIZE ABLATION — GENERATED TYPISTS, NOT REAL HUMANS")
    print(f"  prior held fixed: {args.prior.name} ({prior.sample_count} participants)")
    print(f"  seed={args.seed} users={args.users} "
          f"genuine/user={args.genuine_per_user} impostors/user={args.impostors_per_user}")
    print()

    people = build_population(args.users, args.genuine_per_user, args.seed)
    donors = build_population(args.population_donors, 1, args.seed + 500_000)
    population_samples = [s for d in donors for s in d.enrollment]

    results: dict[str, object] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": "SYNTHETIC generated typists",
        "prior": str(args.prior),
        "seed": args.seed,
        "users": len(people),
        "conditions": {},
    }

    header = (f"{'condition':<12s} {'maturity':<12s} {'thresh':>8s} {'gen acc':>8s} "
              f"{'FRR':>8s} {'FAR':>8s} {'ROC-AUC':>9s} {'EER':>8s} "
              f"{'feats':>6s} {'gen n':>6s} {'imp n':>6s}")
    print(header)
    print("-" * len(header))

    for n_enroll, label, maturity in CONDITIONS:
        gen_scores: list[float] = []
        imp_scores: list[float] = []
        gen_per_user: list[tuple[int, int]] = []
        imp_per_user: list[tuple[int, int]] = []
        thresholds: list[float] = []
        feature_counts: list[int] = []
        gen_ok = gen_n = imp_bad = imp_n = 0

        for person in people:
            profile = build(person.enrollment[:n_enroll], prior, population_samples)
            thresholds.append(profile.threshold)
            feature_counts.append(len(profile.features))

            u_frr = [0, 0]
            for features in person.genuine:
                r = score_identity(profile, features)
                gated = not r.is_comparable or r.coverage < MIN_IDENTITY_COVERAGE
                accepted = (not gated) and r.score <= profile.threshold
                gen_n += 1
                gen_ok += int(accepted)
                u_frr[0] += int(not accepted)
                u_frr[1] += 1
                if not gated:
                    gen_scores.append(r.score)
            gen_per_user.append(tuple(u_frr))

            others = [p for p in people if p.uid != person.uid]
            step = max(1, len(others) // max(args.impostors_per_user, 1))
            u_far = [0, 0]
            for other in others[::step][: args.impostors_per_user]:
                features = other.genuine[person.uid % len(other.genuine)]
                r = score_identity(profile, features)
                gated = not r.is_comparable or r.coverage < MIN_IDENTITY_COVERAGE
                accepted = (not gated) and r.score <= profile.threshold
                imp_n += 1
                imp_bad += int(accepted)
                u_far[0] += int(accepted)
                u_far[1] += 1
                if not gated:
                    imp_scores.append(r.score)
            imp_per_user.append(tuple(u_far))

        med_thr = float(np.median(thresholds))
        frr = 1 - gen_ok / gen_n
        far = imp_bad / imp_n
        auc = roc_auc(gen_scores, imp_scores)
        e = eer(gen_scores, imp_scores)

        print(f"{label:<12s} {maturity.value:<12s} {med_thr:8.4f} "
              f"{gen_ok / gen_n:8.4f} {frr:8.4f} {far:8.4f} "
              f"{(auc if auc is not None else float('nan')):9.5f} "
              f"{(e[0] if e else float('nan')):8.5f} "
              f"{float(np.median(feature_counts)):6.0f} {gen_n:6d} {imp_n:6d}")

        results["conditions"][str(n_enroll)] = {  # type: ignore[index]
            "label": label,
            "maturity": maturity.value,
            "median_threshold": round(med_thr, 5),
            "median_features_modelled": float(np.median(feature_counts)),
            "genuine_n": gen_n, "impostor_n": imp_n,
            "genuine_accept": round(gen_ok / gen_n, 5),
            "frr": round(frr, 5), "far": round(far, 5),
            "roc_auc": round(auc, 5) if auc is not None else None,
            "eer": round(e[0], 5) if e else None,
            "genuine_scores": describe(gen_scores),
            "impostor_scores": describe(imp_scores),
            "frr_ci95": bootstrap_ci(gen_per_user, args.bootstrap, args.seed),
            "far_ci95": bootstrap_ci(imp_per_user, args.bootstrap, args.seed),
        }

    print()
    print("=" * 84)
    print("  SCORE SEPARATION AND UNCERTAINTY")
    print("=" * 84)
    print(f"{'condition':<12s} {'gen p50':>9s} {'imp p50':>9s} {'gap':>9s} "
          f"{'gen p95':>9s} {'FRR CI95':>20s} {'FAR CI95':>20s}")
    for n_enroll, label, _m in CONDITIONS:
        c = results["conditions"][str(n_enroll)]  # type: ignore[index]
        g, i = c["genuine_scores"], c["impostor_scores"]
        gap = (i["p50"] or 0) - (g["p50"] or 0)
        print(f"{label:<12s} {g['p50']:9.4f} {i['p50']:9.4f} {gap:9.4f} "
              f"{g['p95']:9.4f} {str(c['frr_ci95']):>20s} {str(c['far_ci95']):>20s}")

    # --- cold-start threshold, calibrated on DISJOINT users ----------------
    #
    # The fixed comparison above uses one static cold-start threshold (0.20)
    # for every condition. That number was chosen empirically for the
    # single-capture distribution. A two-capture profile has a tighter genuine
    # distribution AND pulls impostors lower, so the same cut admits more of
    # them — which is a mismatched operating point, not a worse representation.
    #
    # So: split users in half, derive each condition's cut on one half at a
    # fixed false-acceptance budget, and report it on the other half. Tuning on
    # the evaluation split is how a threshold comes to describe the sample
    # rather than the system.
    print()
    print("=" * 96)
    print("  COLD-START THRESHOLD CALIBRATED ON DISJOINT USERS, REPORTED ON HELD-OUT USERS")
    print(f"  Budget: FAR <= {args.far_budget:.0%} on the calibration half.")
    print("=" * 96)

    half = len(people) // 2
    cal_people, test_people = people[:half], people[half:]
    print(f"  calibration users {len(cal_people)}  held-out users {len(test_people)}")
    print()
    head = (f"{'condition':<12s} {'static thr':>10s} {'calib thr':>10s} "
            f"{'test FRR':>9s} {'test FAR':>9s} {'FRR CI95':>20s} {'FAR CI95':>20s}")
    print(head)
    print("-" * len(head))

    def score_split(group, n_enroll):
        gen: list[float] = []
        imp: list[float] = []
        per_user = []
        for person in group:
            profile = build(person.enrollment[:n_enroll], prior, population_samples)
            g = []
            for features in person.genuine:
                r = score_identity(profile, features)
                if r.is_comparable and r.coverage >= MIN_IDENTITY_COVERAGE:
                    gen.append(r.score)
                    g.append(r.score)
            others = [p for p in group if p.uid != person.uid]
            step = max(1, len(others) // max(args.impostors_per_user, 1))
            i_scores = []
            for other in others[::step][: args.impostors_per_user]:
                features = other.genuine[person.uid % len(other.genuine)]
                r = score_identity(profile, features)
                if r.is_comparable and r.coverage >= MIN_IDENTITY_COVERAGE:
                    imp.append(r.score)
                    i_scores.append(r.score)
            per_user.append((g, i_scores))
        return gen, imp, per_user

    results["calibrated"] = {}  # type: ignore[index]
    for n_enroll, label, _m in CONDITIONS:
        _cg, cal_imp, _pu = score_split(cal_people, n_enroll)
        if not cal_imp:
            continue
        allowed = int(len(cal_imp) * args.far_budget)
        ordered = sorted(cal_imp)
        cut = float(ordered[allowed - 1]) if allowed >= 1 else float(
            np.nextafter(ordered[0], -np.inf)
        )

        _tg, _ti, per_user = score_split(test_people, n_enroll)
        frr_pu = [(sum(1 for s in g if s > cut), len(g)) for g, _i in per_user if g]
        far_pu = [(sum(1 for s in i if s <= cut), len(i)) for _g, i in per_user if i]
        frr = sum(a for a, _b in frr_pu) / max(sum(b for _a, b in frr_pu), 1)
        far = sum(a for a, _b in far_pu) / max(sum(b for _a, b in far_pu), 1)

        static = results["conditions"][str(n_enroll)]["median_threshold"]  # type: ignore[index]
        print(f"{label:<12s} {static:10.4f} {cut:10.4f} {frr:9.4f} {far:9.4f} "
              f"{str(bootstrap_ci(frr_pu, args.bootstrap, args.seed)):>20s} "
              f"{str(bootstrap_ci(far_pu, args.bootstrap, args.seed)):>20s}")

        results["calibrated"][str(n_enroll)] = {  # type: ignore[index]
            "label": label,
            "static_threshold": static,
            "calibrated_threshold": round(cut, 5),
            "far_budget": args.far_budget,
            "calibration_impostor_n": len(cal_imp),
            "test_frr": round(frr, 5), "test_far": round(far, 5),
            "test_frr_ci95": bootstrap_ci(frr_pu, args.bootstrap, args.seed),
            "test_far_ci95": bootstrap_ci(far_pu, args.bootstrap, args.seed),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
