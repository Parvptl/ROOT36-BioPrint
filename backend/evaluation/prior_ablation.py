#!/usr/bin/env python3
"""Phase D: does swapping the population prior change authentication accuracy?

ONE independent variable: which Aalto population prior is loaded.

    Condition A   data/aalto_prior_synthetic_bootstrap.json   (currently production)
    Condition B   data/aalto_prior_real.json                  (Phase C, 168,593 participants)

Everything else is held identical by construction, not by convention: both
conditions consume the *same* pre-generated session objects, built once before
either condition runs, and each condition fits a fresh profile from them. A
profile is never carried between conditions, so Condition A's adaptation
history cannot reach Condition B.

Why the sessions are synthetic
------------------------------
They have to be. BioPrint discards raw events at extraction and deletes
enrollment feature vectors once a profile is fitted, so the feature vectors
behind the real human login attempts do not exist anywhere and cannot be
re-scored under a second prior. That is the privacy design working as intended,
and it makes a real-data A/B structurally impossible without collecting new
data. See --real-supplement for the small real-data check that IS possible.

These are generated typists. The result describes how the scoring model
responds to a prior change under controlled conditions. It is not a claim about
accuracy on real people.

Scoring direction
-----------------
Verified against the production decision code, not assumed from the name:
``scoring.score_identity`` returns a DEVIATION in [0, 1] where 0 means
indistinguishable from the baseline, and ``risk_engine.decide`` blocks when
``decision_score > effective_threshold``. So LOWER is more genuine, and every
ROC/EER computation below uses ``-score`` as the discriminant.

    cd backend
    ./.venv/Scripts/python.exe evaluation/prior_ablation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics as st
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.features.extractor import extract_from_session  # noqa: E402
from app.behavioral.fingerprint.aalto_prior import aalto_population_prior  # noqa: E402
from app.behavioral.fingerprint.calibration import (  # noqa: E402
    build_calibrated_profile,
    cold_start_threshold,
)
from app.behavioral.fingerprint.population import PopulationPrior  # noqa: E402
from app.behavioral.fingerprint.profile import (  # noqa: E402
    BehaviorProfile,
    Maturity,
    fit_cold_start_profile,
)
from app.behavioral.fingerprint.scoring import score_identity  # noqa: E402
from app.behavioral.scoring.risk_engine import MIN_IDENTITY_COVERAGE  # noqa: E402
from tests.factories import TypingStyle, human_session  # noqa: E402

log = logging.getLogger("bioprint.prior_ablation")

DATA = _BACKEND / "data"
SYNTHETIC_PRIOR = DATA / "aalto_prior_synthetic_bootstrap.json"
REAL_PRIOR = DATA / "aalto_prior_real.json"

# Enrollment sizes, mapped to the maturity bands the product uses.
MATURITIES: tuple[tuple[int, Maturity], ...] = (
    (1, Maturity.COLD_START),
    (2, Maturity.WARMING),
    (4, Maturity.WARMING),
    (8, Maturity.MATURE),
)

SEED = 20260920


# ────────────────────────────────────────────────────────────────── population


@dataclass
class Person:
    """One generated typist: enrollment rounds plus held-out genuine attempts."""

    uid: int
    style: TypingStyle
    enrollment: list[dict[str, float]]
    genuine: list[dict[str, float]]


def make_style(rng: np.random.Generator) -> TypingStyle:
    """A typist drawn from a deliberately wide population.

    Ranges are wide enough that the population contains both fluent overlapping
    typists and slow hunt-and-peck ones, because a prior ablation that only
    sees one kind of typist would not exercise the features whose scales the
    two priors most disagree about (the dwell family, burst fraction and
    negative-flight fraction).
    """
    return TypingStyle(
        iki_mean_ms=float(rng.uniform(95.0, 330.0)),
        iki_jitter=float(rng.uniform(0.18, 0.55)),
        dwell_mean_ms=float(rng.uniform(55.0, 150.0)),
        dwell_jitter=float(rng.uniform(0.15, 0.40)),
        overlap_prob=float(rng.uniform(0.0, 0.62)),
        pause_prob=float(rng.uniform(0.01, 0.18)),
        backspace_prob=float(rng.uniform(0.0, 0.12)),
        right_shift_prob=float(rng.uniform(0.02, 0.98)),
        tab_between_fields=bool(rng.random() < 0.6),
        focus_delay_ms=float(rng.uniform(180.0, 700.0)),
        presubmit_ms=float(rng.uniform(300.0, 1200.0)),
        pointer_speed=float(rng.uniform(0.6, 2.4)),
    )


def seeded_phrase(seed: int) -> str:
    """A challenge phrase drawn reproducibly for the experiment.

    Production uses `challenge.generate_phrase`, which draws from `secrets`
    precisely so an observer cannot predict the next phrase. That is correct
    there and useless here: an unseeded phrase makes every run of this
    experiment draw different text, and the run-to-run movement was large
    enough to be mistaken for an effect of the prior.

    So the experiment draws from the SAME word list with the SAME shape
    (word count, two capitalised positions), through a seeded generator. The
    production CSPRNG path is untouched.
    """
    from app.auth.challenge import PHRASE_WORD_COUNT, _WORDS

    rng = random.Random(seed)
    words = [rng.choice(_WORDS) for _ in range(PHRASE_WORD_COUNT)]
    for position in rng.sample(range(PHRASE_WORD_COUNT), 2):
        words[position] = words[position].capitalize()
    return " ".join(words)


def session_features(style: TypingStyle, seed: int) -> dict[str, float]:
    """One capture, through the production extractor."""
    _view, extracted = extract_from_session(
        human_session(seeded_phrase(seed), style=style, seed=seed)
    )
    return extracted.values


def build_population(n_users: int, genuine_per_user: int, seed: int) -> list[Person]:
    """Generate every session ONCE, before any condition runs.

    This is what makes the two conditions comparable: they are handed the same
    dictionaries, so no difference in the results can come from a difference in
    the inputs.
    """
    rng = np.random.default_rng(seed)
    people: list[Person] = []
    for uid in range(n_users):
        style = make_style(rng)
        base = seed + uid * 1000
        people.append(
            Person(
                uid=uid,
                style=style,
                enrollment=[session_features(style, base + i) for i in range(8)],
                genuine=[
                    session_features(style, base + 100 + i)
                    for i in range(genuine_per_user)
                ],
            )
        )
    return people


# ─────────────────────────────────────────────────────────────────── metrics


def roc_auc(genuine: list[float], impostor: list[float]) -> float | None:
    """AUC on the DEVIATION scale, where lower means more genuine.

    Computed as the Mann-Whitney statistic: the probability that a randomly
    chosen genuine attempt scores lower (more genuine) than a randomly chosen
    impostor attempt, with ties counted as half.
    """
    if not genuine or not impostor:
        return None
    wins = 0.0
    impostor_sorted = sorted(impostor)
    for g in genuine:
        lo = _bisect_left(impostor_sorted, g)
        hi = _bisect_right(impostor_sorted, g)
        wins += (len(impostor_sorted) - hi) + 0.5 * (hi - lo)
    return wins / (len(genuine) * len(impostor))


def _bisect_left(values: list[float], x: float) -> int:
    import bisect

    return bisect.bisect_left(values, x)


def _bisect_right(values: list[float], x: float) -> int:
    import bisect

    return bisect.bisect_right(values, x)


def eer(genuine: list[float], impostor: list[float]) -> tuple[float, float] | None:
    """Equal error rate and the threshold achieving it.

    Sweeps every candidate cut on the deviation scale. Accept when
    score <= threshold, matching the production comparison.
    """
    if not genuine or not impostor:
        return None
    candidates = sorted(set(genuine) | set(impostor))
    best = None
    for cut in candidates:
        frr = sum(1 for g in genuine if g > cut) / len(genuine)
        far = sum(1 for i in impostor if i <= cut) / len(impostor)
        gap = abs(frr - far)
        if best is None or gap < best[0]:
            best = (gap, (frr + far) / 2.0, cut)
    return (best[1], best[2]) if best else None


def threshold_at_far(impostor: list[float], target_far: float) -> float | None:
    """Largest threshold whose impostor acceptance stays at or below target.

    Returns None when the target is unreachable — with 100 impostor scores a
    FAR of 0.1% cannot be demonstrated, and inventing a threshold for it would
    be false precision.
    """
    if not impostor:
        return None
    if target_far < 1.0 / len(impostor):
        return None
    ordered = sorted(impostor)
    allowed = int(math.floor(target_far * len(ordered)))
    if allowed == 0:
        cut = ordered[0]
        return float(np.nextafter(cut, -np.inf))
    return float(ordered[allowed - 1])


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in
                ("n", "mean", "std", "min", "p5", "p25", "p50", "p75", "p95", "max")}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 5),
        "std": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 5),
        "min": round(float(arr.min()), 5),
        "p5": round(float(np.percentile(arr, 5)), 5),
        "p25": round(float(np.percentile(arr, 25)), 5),
        "p50": round(float(np.percentile(arr, 50)), 5),
        "p75": round(float(np.percentile(arr, 75)), 5),
        "p95": round(float(np.percentile(arr, 95)), 5),
        "max": round(float(arr.max()), 5),
    }


def bootstrap_ci(
    per_user: list[tuple[int, int]], reps: int, seed: int
) -> tuple[float, float] | None:
    """Percentile CI for a rate, resampling USERS rather than attempts.

    Attempts from one person are not independent observations: they share a
    profile, a threshold and a typing style. Resampling attempts would treat
    100 attempts from 10 people as 100 independent facts and report an interval
    far tighter than the evidence supports.

    `per_user` is [(events, trials), ...], one entry per user.
    """
    users = [u for u in per_user if u[1] > 0]
    if len(users) < 2:
        return None
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(reps):
        picks = rng.integers(0, len(users), size=len(users))
        events = sum(users[i][0] for i in picks)
        trials = sum(users[i][1] for i in picks)
        if trials:
            rates.append(events / trials)
    if not rates:
        return None
    return (
        round(float(np.percentile(rates, 2.5)), 4),
        round(float(np.percentile(rates, 97.5)), 4),
    )


# ──────────────────────────────────────────────────────────── one condition


@dataclass
class Attempt:
    uid: int
    label: str          # "genuine" | "impostor"
    score: float
    threshold: float    # the profile's own calibrated threshold
    latency_ms: float
    # The risk engine blocks on coverage BEFORE it ever compares a score to a
    # threshold. Carrying that here is not optional: a profile fitted from too
    # few sessions has no features at all, scores every attempt 0.0, and would
    # look like a perfect acceptor if judged on the score alone. Production
    # returns INSUFFICIENT_SIGNAL for exactly that case.
    comparable: bool = True
    coverage: float = 1.0

    @property
    def gated_out(self) -> bool:
        return not self.comparable or self.coverage < MIN_IDENTITY_COVERAGE


@dataclass
class ConditionRun:
    name: str
    prior_path: Path
    prior: PopulationPrior
    # maturity_key -> attempts
    attempts: dict[str, list[Attempt]] = field(default_factory=dict)
    thresholds: dict[str, list[float]] = field(default_factory=dict)
    profiles: dict[str, dict[int, BehaviorProfile]] = field(default_factory=dict)


def build_profile(
    enrollment: list[dict[str, float]],
    prior: PopulationPrior,
    population_samples: list[dict[str, float]],
) -> BehaviorProfile:
    """Mirror the production enrollment path, minus the ML layer.

    The ML anomaly layer ships at weight 0.0 (shadow mode, see
    app/config.py::ml_weight) and is byte-identical in both conditions, so
    leaving it out cannot favour either prior. It is excluded because training
    it writes model artifacts to disk, which a read-only experiment should not
    do.
    """
    if len(enrollment) == 1:
        features = fit_cold_start_profile(enrollment[0], prior)
        cal = cold_start_threshold()
        return BehaviorProfile(
            features=features,
            session_count=1,
            population_size=prior.sample_count,
            population_informative=prior.is_informative,
            threshold=cal.threshold,
            threshold_source=cal.source,
            calibration=cal.as_json(),
            maturity=Maturity.COLD_START,
        )
    return build_calibrated_profile(enrollment, prior, population_samples)


def run_condition(
    name: str,
    prior_path: Path,
    people: list[Person],
    population_samples: list[dict[str, float]],
    impostors_per_user: int,
    rng_seed: int,
) -> ConditionRun:
    prior = aalto_population_prior(prior_path)
    if prior is None:
        raise SystemExit(f"could not load prior from {prior_path}")

    run = ConditionRun(name=name, prior_path=prior_path, prior=prior)

    for n_enroll, maturity in MATURITIES:
        key = f"{n_enroll}"
        run.attempts[key] = []
        run.thresholds[key] = []
        run.profiles[key] = {}

        for person in people:
            # A fresh profile, every time, from the same slice of the same
            # pre-generated enrollment sessions.
            profile = build_profile(
                person.enrollment[:n_enroll], prior, population_samples
            )
            run.profiles[key][person.uid] = profile
            run.thresholds[key].append(profile.threshold)

            for features in person.genuine:
                started = time.perf_counter()
                result = score_identity(profile, features)
                elapsed = (time.perf_counter() - started) * 1000.0
                run.attempts[key].append(
                    Attempt(person.uid, "genuine", result.score, profile.threshold,
                            elapsed, result.is_comparable, result.coverage)
                )

            # Impostors: other generated people's genuine sessions, submitted
            # against this profile. Deterministic choice of which others, so
            # both conditions face the same impostor set.
            others = [p for p in people if p.uid != person.uid]
            step = max(1, len(others) // max(impostors_per_user, 1))
            chosen = others[:: step][:impostors_per_user]
            for other in chosen:
                features = other.genuine[person.uid % len(other.genuine)]
                started = time.perf_counter()
                result = score_identity(profile, features)
                elapsed = (time.perf_counter() - started) * 1000.0
                run.attempts[key].append(
                    Attempt(person.uid, "impostor", result.score, profile.threshold,
                            elapsed, result.is_comparable, result.coverage)
                )

    return run


# ───────────────────────────────────────────────────────────────── reporting


def split_scores(attempts: list[Attempt]) -> tuple[list[float], list[float]]:
    """Scores of attempts the engine actually compared to a threshold.

    Gated-out attempts are excluded: they carry a placeholder 0.0 that is not a
    measurement of anything, and feeding it to a ROC curve would manufacture
    perfect separation out of missing data.
    """
    genuine = [a.score for a in attempts if a.label == "genuine" and not a.gated_out]
    impostor = [a.score for a in attempts if a.label == "impostor" and not a.gated_out]
    return genuine, impostor


def rates_at(attempts: list[Attempt], threshold: float | None) -> dict[str, object]:
    """Accept when score <= threshold. None means each profile's own threshold."""
    gen_ok = gen_n = imp_bad = imp_n = 0
    per_user_frr: dict[int, list[int]] = {}
    per_user_far: dict[int, list[int]] = {}

    for a in attempts:
        cut = a.threshold if threshold is None else threshold
        # Gate first, exactly as risk_engine.decide does: coverage is checked
        # before the score is compared to anything, and a failure there is a
        # BLOCK regardless of the score.
        accepted = (not a.gated_out) and a.score <= cut
        if a.label == "genuine":
            gen_n += 1
            gen_ok += int(accepted)
            slot = per_user_frr.setdefault(a.uid, [0, 0])
            slot[0] += int(not accepted)
            slot[1] += 1
        else:
            imp_n += 1
            imp_bad += int(accepted)
            slot = per_user_far.setdefault(a.uid, [0, 0])
            slot[0] += int(accepted)
            slot[1] += 1

    return {
        "genuine_n": gen_n,
        "impostor_n": imp_n,
        "gated_out": sum(1 for a in attempts if a.gated_out),
        "genuine_accept": round(gen_ok / gen_n, 5) if gen_n else None,
        "frr": round(1 - gen_ok / gen_n, 5) if gen_n else None,
        "far": round(imp_bad / imp_n, 5) if imp_n else None,
        "impostor_reject": round(1 - imp_bad / imp_n, 5) if imp_n else None,
        "_frr_per_user": [tuple(v) for v in per_user_frr.values()],
        "_far_per_user": [tuple(v) for v in per_user_far.values()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=120,
                        help="generated typists (split into calibration and test)")
    parser.add_argument("--genuine-per-user", type=int, default=10)
    parser.add_argument("--impostors-per-user", type=int, default=10)
    parser.add_argument("--population-donors", type=int, default=12,
                        help="separate typists supplying population_samples for "
                             "the system's own impostor calibration")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path,
                        default=_BACKEND.parent / "evaluation" / "out" / "phase_d_prior_ablation.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)

    for path in (SYNTHETIC_PRIOR, REAL_PRIOR):
        if not path.exists():
            raise SystemExit(f"missing prior: {path}")

    print("Phase D: population prior ablation")
    print(f"  A  synthetic  {SYNTHETIC_PRIOR.name}")
    print(f"  B  real       {REAL_PRIOR.name}")
    print(f"  seed={args.seed}  users={args.users}  "
          f"genuine/user={args.genuine_per_user}  impostors/user={args.impostors_per_user}")
    print()

    # --- one generation of sessions, shared by both conditions --------------
    print("generating sessions...")
    people = build_population(args.users, args.genuine_per_user, args.seed)

    donors = build_population(args.population_donors, 1, args.seed + 500_000)
    population_samples = [s for d in donors for s in d.enrollment]

    # Disjoint USERS, not merely disjoint attempts. A threshold chosen on the
    # same people it is later tested on would be reporting how well it fits
    # them, not how well it generalises.
    half = len(people) // 2
    calibration_people, test_people = people[:half], people[half:]
    print(f"  calibration users {len(calibration_people)}  test users {len(test_people)}")
    print(f"  population donors {len(donors)} -> {len(population_samples)} samples")
    print()

    results: dict[str, object] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "conditions": {
            "A_synthetic": str(SYNTHETIC_PRIOR),
            "B_real": str(REAL_PRIOR),
        },
        "data": {
            "kind": "SYNTHETIC generated typists",
            "users_total": len(people),
            "calibration_users": len(calibration_people),
            "test_users": len(test_people),
            "genuine_per_user": args.genuine_per_user,
            "impostors_per_user": args.impostors_per_user,
            "population_donor_users": len(donors),
            "population_samples": len(population_samples),
        },
        "maturities": {},
        "calibration": {},
        "held_out": {},
    }

    runs: dict[str, dict[str, ConditionRun]] = {}
    for split_name, group in (("calibration", calibration_people), ("test", test_people)):
        runs[split_name] = {}
        for cond, path in (("A_synthetic", SYNTHETIC_PRIOR), ("B_real", REAL_PRIOR)):
            print(f"running {split_name}/{cond} ...")
            runs[split_name][cond] = run_condition(
                cond, path, group, population_samples,
                args.impostors_per_user, args.seed,
            )

    # --- 1. fixed-threshold and own-threshold, by maturity ------------------
    print()
    print("=" * 108)
    print("  FIXED-THRESHOLD COMPARISON (test users)")
    print("  'own'   = each profile's own calibrated threshold, as production derives it")
    print("  'fixed' = Condition A's median threshold applied to BOTH conditions")
    print("=" * 108)
    header = (f"{'enroll':>7s} {'maturity':<12s} {'prior':<12s} {'mode':<6s} "
              f"{'thresh':>8s} {'gen acc':>8s} {'FRR':>8s} {'FAR':>8s} "
              f"{'gen n':>7s} {'imp n':>7s}")
    print(header)
    print("-" * len(header))

    for n_enroll, maturity in MATURITIES:
        key = str(n_enroll)
        fixed_cut = float(np.median(runs["test"]["A_synthetic"].thresholds[key]))
        entry: dict[str, object] = {"maturity": maturity.value,
                                    "fixed_threshold_from_A": round(fixed_cut, 5)}

        for cond in ("A_synthetic", "B_real"):
            attempts = runs["test"][cond].attempts[key]
            own = rates_at(attempts, None)
            fixed = rates_at(attempts, fixed_cut)
            median_thr = float(np.median(runs["test"][cond].thresholds[key]))

            for mode, r, thr in (("own", own, median_thr), ("fixed", fixed, fixed_cut)):
                print(f"{n_enroll:>7d} {maturity.value:<12s} {cond:<12s} {mode:<6s} "
                      f"{thr:8.4f} {r['genuine_accept']:8.4f} {r['frr']:8.4f} "
                      f"{r['far']:8.4f} {r['genuine_n']:7d} {r['impostor_n']:7d}")

            genuine, impostor = split_scores(attempts)
            auc = roc_auc(genuine, impostor)
            e = eer(genuine, impostor)
            entry[cond] = {
                "median_threshold": round(median_thr, 5),
                "own_threshold": {k: v for k, v in own.items() if not k.startswith("_")},
                "fixed_threshold": {k: v for k, v in fixed.items() if not k.startswith("_")},
                "roc_auc": None if auc is None else round(auc, 5),
                "eer": None if e is None else round(e[0], 5),
                "eer_threshold": None if e is None else round(e[1], 5),
                "genuine_scores": describe(genuine),
                "impostor_scores": describe(impostor),
                "frr_ci95_own": bootstrap_ci(own["_frr_per_user"], args.bootstrap, args.seed),
                "far_ci95_own": bootstrap_ci(own["_far_per_user"], args.bootstrap, args.seed),
                "frr_ci95_fixed": bootstrap_ci(fixed["_frr_per_user"], args.bootstrap, args.seed),
                "far_ci95_fixed": bootstrap_ci(fixed["_far_per_user"], args.bootstrap, args.seed),
                "latency_ms": describe([a.latency_ms for a in attempts]),
            }
        results["maturities"][key] = entry  # type: ignore[index]
        print("-" * len(header))

    # --- 2. threshold-free separation --------------------------------------
    print()
    print("=" * 76)
    print("  THRESHOLD-FREE SEPARATION (test users) — independent of any cut")
    print("=" * 76)
    print(f"{'enroll':>7s} {'prior':<12s} {'ROC-AUC':>9s} {'EER':>9s} "
          f"{'gen p50':>9s} {'imp p50':>9s} {'gap':>9s}")
    def _n(value, spec):
        """Format a metric that may not exist. None means 'not computable',
        which is a different statement from a bad score and must not be
        printed as one."""
        return f"{value:{spec}}" if value is not None else f"{'n/a':>{spec.split('.')[0]}s}"

    for n_enroll, _m in MATURITIES:
        key = str(n_enroll)
        for cond in ("A_synthetic", "B_real"):
            e = results["maturities"][key][cond]  # type: ignore[index]
            gp = e["genuine_scores"]["p50"]
            ip = e["impostor_scores"]["p50"]
            gap = (ip - gp) if (gp is not None and ip is not None) else None
            print(f"{n_enroll:>7d} {cond:<12s} {_n(e['roc_auc'], '9.5f')} "
                  f"{_n(e['eer'], '9.5f')} {_n(gp, '9.4f')} {_n(ip, '9.4f')} "
                  f"{_n(gap, '9.4f')}")
            if e["genuine_scores"]["n"] is None:
                print(f"{'':>7s} {'':<12s} all attempts gated out before scoring "
                      f"(profile had too few features to compare)")

    # --- 3. calibration on calibration users, evaluated on test users -------
    print()
    print("=" * 96)
    print("  CALIBRATED THRESHOLD, chosen on CALIBRATION users, applied to TEST users")
    print("  Target: FAR = 1% on the calibration split. Chosen because it is the")
    print("  tightest operating point the calibration sample can actually demonstrate.")
    print("=" * 96)
    header2 = (f"{'enroll':>7s} {'prior':<12s} {'cal thr':>9s} {'prod thr':>9s} "
               f"{'test FRR':>9s} {'test FAR':>9s} {'gen n':>7s} {'imp n':>7s}")
    print(header2)
    print("-" * len(header2))

    for n_enroll, _m in MATURITIES:
        key = str(n_enroll)
        cal_entry: dict[str, object] = {}
        for cond in ("A_synthetic", "B_real"):
            cal_attempts = runs["calibration"][cond].attempts[key]
            _g, cal_impostor = split_scores(cal_attempts)
            cut = threshold_at_far(cal_impostor, 0.01)

            test_attempts = runs["test"][cond].attempts[key]
            prod_thr = float(np.median(runs["test"][cond].thresholds[key]))

            if cut is None:
                print(f"{n_enroll:>7d} {cond:<12s} {'n/a':>9s} {prod_thr:9.4f} "
                      f"{'—':>9s} {'—':>9s} — FAR=1% not demonstrable "
                      f"with {len(cal_impostor)} calibration impostor scores")
                cal_entry[cond] = {"calibrated_threshold": None,
                                   "reason": "target FAR below calibration resolution"}
                continue

            r = rates_at(test_attempts, cut)
            print(f"{n_enroll:>7d} {cond:<12s} {cut:9.4f} {prod_thr:9.4f} "
                  f"{r['frr']:9.4f} {r['far']:9.4f} {r['genuine_n']:7d} {r['impostor_n']:7d}")
            cal_entry[cond] = {
                "calibrated_threshold": round(cut, 5),
                "production_median_threshold": round(prod_thr, 5),
                "calibration_impostor_n": len(cal_impostor),
                "test": {k: v for k, v in r.items() if not k.startswith("_")},
                "test_frr_ci95": bootstrap_ci(r["_frr_per_user"], args.bootstrap, args.seed),
                "test_far_ci95": bootstrap_ci(r["_far_per_user"], args.bootstrap, args.seed),
            }
        results["calibration"][key] = cal_entry  # type: ignore[index]

    # --- 4. feature-level effect on one representative profile --------------
    print()
    print("=" * 96)
    print("  FEATURE SCALES AND WEIGHTS — same user, same 8 enrollment sessions")
    print("=" * 96)
    probe = test_people[0]
    feature_rows = []
    profiles = {c: runs["test"][c].profiles["8"][probe.uid] for c in ("A_synthetic", "B_real")}
    names = sorted(set(profiles["A_synthetic"].features) | set(profiles["B_real"].features))
    print(f"{'feature':<32s} {'A scale':>10s} {'B scale':>10s} {'ratio':>7s} "
          f"{'A weight':>9s} {'B weight':>9s} {'w ratio':>8s}")
    print("-" * 92)
    for name in names:
        a = profiles["A_synthetic"].features.get(name)
        b = profiles["B_real"].features.get(name)
        if a is None or b is None:
            continue
        sr = b.scale / a.scale if a.scale else float("inf")
        wr = b.weight / a.weight if a.weight else float("inf")
        if abs(sr - 1.0) > 1e-9 or abs(wr - 1.0) > 1e-9:
            print(f"{name:<32s} {a.scale:10.4f} {b.scale:10.4f} {sr:7.3f} "
                  f"{a.weight:9.4f} {b.weight:9.4f} {wr:8.3f}")
        feature_rows.append({
            "feature": name, "a_scale": round(a.scale, 6), "b_scale": round(b.scale, 6),
            "scale_ratio": round(sr, 5), "a_weight": round(a.weight, 6),
            "b_weight": round(b.weight, 6), "weight_ratio": round(wr, 5),
            "a_median": round(a.median, 6),
        })
    results["feature_effect"] = {
        "probe_user": probe.uid, "enrollment_sessions": 8, "features": feature_rows,
    }
    unchanged = [r["feature"] for r in feature_rows
                 if abs(r["scale_ratio"] - 1.0) < 1e-9 and abs(r["weight_ratio"] - 1.0) < 1e-9]
    print(f"\nunchanged ({len(unchanged)}): {', '.join(unchanged) or '-'}")

    # --- 5. the Shift feature ----------------------------------------------
    shift = "kbd_shift_right_ratio"
    a_prior = aalto_population_prior(SYNTHETIC_PRIOR)
    b_prior = aalto_population_prior(REAL_PRIOR)
    shift_report = {
        "in_synthetic_prior_scales": shift in a_prior.scales,
        "in_real_prior_scales": shift in b_prior.scales,
        "real_scale_for_own_median_0.4": round(b_prior.scale_for(shift, 0.4), 6),
        "relative_fallback_0.4": round(0.35 * 0.4, 6),
        "present_in_probe_profile": shift in profiles["B_real"].features,
        "probe_profile_scale": (
            round(profiles["B_real"].features[shift].scale, 6)
            if shift in profiles["B_real"].features else None
        ),
    }
    results["shift_feature"] = shift_report
    print()
    print("=" * 76)
    print("  UNOBSERVED SHIFT FEATURE")
    print("=" * 76)
    for k, v in shift_report.items():
        print(f"  {k:<36s} {v}")

    # --- write --------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
