"""Does adding the ML layer actually help?

Compares, on identical data and identical attempts:

    BASELINE   statistical fingerprint alone (the system before this layer)
    ML-ONLY    Isolation Forest anomaly score alone
    HYBRID     blended, across a sweep of weights

Reported as equal-error rate, which is threshold-free, so the comparison
measures discriminative power rather than whose threshold happened to be
better placed. The calibrated operating point is reported separately.

An earlier benchmark in this directory (baseline_comparison.py) found
Isolation Forest losing badly to the statistical model. That fitted it on
eight session-level vectors. This fits it on windowed vectors, roughly
thirty-two per user, which is a different regime — so it is re-measured here
rather than assumed to carry over either way.

SYNTHETIC MECHANISM VALIDATION. Generated typists, not people. This compares
algorithms under identical conditions; it is not authentication accuracy, and
real-human accuracy remains unmeasured.

    cd backend
    ./.venv/Scripts/python.exe ../evaluation/hybrid_comparison.py
"""

from __future__ import annotations

import statistics as st
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

from app.api.routes_auth import ENROLLMENT_ROUNDS  # noqa: E402
from app.auth.challenge import generate_phrase  # noqa: E402
from app.behavioral.features.extractor import extract_from_session  # noqa: E402
from app.behavioral.fingerprint.calibration import build_calibrated_profile  # noqa: E402
from app.behavioral.fingerprint.population import (  # noqa: E402
    PopulationPrior,
    fit_population_prior,
)
from app.behavioral.fingerprint.scoring import score_identity  # noqa: E402
from app.behavioral.ml.anomaly_model import train_model  # noqa: E402
from app.behavioral.ml.windows import window_feature_dicts  # noqa: E402
from tests.factories import TypingStyle, human_session  # noqa: E402

TRIALS = 25
GENUINE_PER_TRIAL = 8
IMPOSTOR_PER_TRIAL = 8

# Ordinary people who differ from one another, not caricatures. Using an
# extreme impostor would make every configuration look excellent.
POPULATION = {
    "alice":  TypingStyle(iki_mean_ms=140, dwell_mean_ms=82,  overlap_prob=0.45, right_shift_prob=0.95, tab_between_fields=True,  pointer_speed=1.6),
    "bob":    TypingStyle(iki_mean_ms=200, dwell_mean_ms=105, overlap_prob=0.20, right_shift_prob=0.30, tab_between_fields=True,  pointer_speed=1.1),
    "carol":  TypingStyle(iki_mean_ms=260, dwell_mean_ms=120, overlap_prob=0.05, right_shift_prob=0.10, tab_between_fields=False, pointer_speed=0.8),
    "dan":    TypingStyle(iki_mean_ms=165, dwell_mean_ms=95,  overlap_prob=0.30, right_shift_prob=0.60, tab_between_fields=True,  pointer_speed=1.3),
    "erin":   TypingStyle(iki_mean_ms=120, dwell_mean_ms=70,  overlap_prob=0.60, right_shift_prob=0.85, tab_between_fields=False, pointer_speed=1.9),
    "frank":  TypingStyle(iki_mean_ms=230, dwell_mean_ms=110, overlap_prob=0.12, right_shift_prob=0.20, tab_between_fields=True,  pointer_speed=0.95),
}
NAMES = list(POPULATION)

WEIGHT_SWEEP = [
    ("statistical only  (1.0 / 0.0)", 1.0, 0.0),
    ("hybrid            (0.8 / 0.2)", 0.8, 0.2),
    ("hybrid            (0.7 / 0.3)", 0.7, 0.3),
    ("hybrid  DEFAULT   (0.6 / 0.4)", 0.6, 0.4),
    ("hybrid            (0.5 / 0.5)", 0.5, 0.5),
    ("hybrid            (0.4 / 0.6)", 0.4, 0.6),
    ("ML only           (0.0 / 1.0)", 0.0, 1.0),
]


def session_for(who: str, seed: int):
    return human_session(generate_phrase(), style=POPULATION[who], seed=seed)


def features_of(session) -> dict[str, float]:
    _, extracted = extract_from_session(session)
    return extracted.as_dict()


def equal_error(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Sweep every cut; return (EER, threshold achieving it)."""
    if genuine.size == 0 or impostor.size == 0:
        return 0.5, 0.0
    cuts = np.linspace(0.0, 1.0, 1001)
    best = (1.0, 0.5, 0.0)
    for cut in cuts:
        frr = float(np.mean(genuine > cut))
        far = float(np.mean(impostor <= cut))
        gap = abs(frr - far)
        if gap < best[0]:
            best = (gap, (frr + far) / 2.0, float(cut))
    return best[1], best[2]


def main() -> None:
    print(__doc__.split("\n")[0])
    print("\nSYNTHETIC MECHANISM VALIDATION - not real authentication accuracy")
    print(f"{TRIALS} independent enrollments of {ENROLLMENT_ROUNDS} rounds each.\n")

    population_features = [
        features_of(session_for(NAMES[i % len(NAMES)], 960_000 + i)) for i in range(36)
    ]
    prior = fit_population_prior(population_features)

    stat_g: list[float] = []
    stat_i: list[float] = []
    ml_g: list[float] = []
    ml_i: list[float] = []
    train_windows: list[int] = []
    infer_ms: list[float] = []
    models_trained = 0

    for trial in range(TRIALS):
        subject = NAMES[trial % len(NAMES)]
        others = [n for n in NAMES if n != subject]
        base = 700_000 + trial * 1_000

        enrollment_sessions = [
            session_for(subject, base + r) for r in range(ENROLLMENT_ROUNDS)
        ]
        enrollment_features = [features_of(s) for s in enrollment_sessions]
        profile = build_calibrated_profile(
            enrollment_features, prior, population_features
        )

        windows: list[dict[str, float]] = []
        for session in enrollment_sessions:
            windows.extend(window_feature_dicts(session))
        model = train_model(windows)
        if model is None:
            continue
        models_trained += 1
        train_windows.append(model.n_training_windows)

        for i in range(GENUINE_PER_TRIAL):
            session = session_for(subject, base + 100 + i)
            stat_g.append(score_identity(profile, features_of(session)).score)
            started = time.perf_counter()
            anomaly = model.anomaly_for(window_feature_dicts(session))
            infer_ms.append((time.perf_counter() - started) * 1000.0)
            ml_g.append(anomaly.score)

        for i in range(IMPOSTOR_PER_TRIAL):
            session = session_for(others[i % len(others)], base + 200 + i)
            stat_i.append(score_identity(profile, features_of(session)).score)
            ml_i.append(model.anomaly_for(window_feature_dicts(session)).score)

    sg, si = np.asarray(stat_g), np.asarray(stat_i)
    mg, mi = np.asarray(ml_g), np.asarray(ml_i)

    print(f"models trained        : {models_trained}/{TRIALS}")
    print(f"training windows/user : median {int(st.median(train_windows))}")
    print(f"attempts scored       : {sg.size} genuine, {si.size} impostor\n")

    header = f"{'configuration':32s} {'EER':>8s} {'FRR@EER':>9s} {'FAR@EER':>9s} {'cut':>7s}"
    print(header)
    print("-" * len(header))

    results = []
    for label, w_stat, w_ml in WEIGHT_SWEEP:
        total = w_stat + w_ml
        a, b = w_stat / total, w_ml / total
        genuine = a * sg + b * mg
        impostor = a * si + b * mi
        eer, cut = equal_error(genuine, impostor)
        frr = float(np.mean(genuine > cut))
        far = float(np.mean(impostor <= cut))
        results.append((eer, label))
        print(f"{label:32s} {eer:8.1%} {frr:9.1%} {far:9.1%} {cut:7.3f}")

    baseline = next(e for e, l in results if l.startswith("statistical only"))
    best_eer, best_label = min(results)
    default = next(e for e, l in results if "DEFAULT" in l)

    print()
    print(f"baseline (statistical only) : {baseline:.1%} EER")
    print(f"default hybrid (0.6/0.4)    : {default:.1%} EER")
    print(f"best configuration          : {best_label.strip()} at {best_eer:.1%}")
    print()
    if default < baseline - 0.005:
        print(f"The ML layer helps: {baseline:.1%} -> {default:.1%} at the default weights.")
    elif default > baseline + 0.005:
        print(
            f"The ML layer HURTS at the default weights ({baseline:.1%} -> {default:.1%}).\n"
            f"That is a real result. Either reweight toward the statistical layer or\n"
            f"leave the ML layer off; do not ship it because it was asked for."
        )
    else:
        print(
            f"The ML layer is roughly neutral at the default weights "
            f"({baseline:.1%} vs {default:.1%}).\n"
            f"It adds an independent signal without measurably improving separation\n"
            f"on this data."
        )

    if infer_ms:
        ordered = sorted(infer_ms)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        print(
            f"\nML inference (windowing + forest): p50 {st.median(infer_ms):.2f} ms, "
            f"p95 {p95:.2f} ms, n={len(infer_ms)}"
        )


if __name__ == "__main__":
    main()
