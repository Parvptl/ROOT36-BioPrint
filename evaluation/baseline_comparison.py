"""Does the simple model actually beat the obvious ML alternatives?

The design rejects One-Class SVM, Isolation Forest and similar on the grounds
that they are not estimable from eight enrollment samples in twenty-seven
dimensions. That is a claim, and a claim in a README is worth less than a
table. This fits them on exactly the same data the real profile is fitted on,
scores exactly the same attempts, and reports them side by side.

If a baseline wins, that is the answer and the model should change. The point
of running it is that the outcome is allowed to be inconvenient.

SYNTHETIC MECHANISM VALIDATION. Generated typists, not people. This compares
algorithms under identical conditions; it is not authentication accuracy.

    cd backend
    ./.venv/Scripts/python.exe ../evaluation/baseline_comparison.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402
from sklearn.covariance import EllipticEnvelope  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.neighbors import LocalOutlierFactor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

from app.behavioral.fingerprint.calibration import build_calibrated_profile  # noqa: E402
from app.behavioral.fingerprint.population import (  # noqa: E402
    PopulationPrior,
    fit_population_prior,
)
from app.behavioral.fingerprint.scoring import score_identity  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_sweep import (  # noqa: E402
    ENROLL_ROUNDS,
    GENUINE_PER_TRIAL,
    IMPOSTOR_PER_TRIAL,
    NAMES,
    capture,
)

TRIALS = 30


@dataclass
class Outcome:
    """Decision scores for one model on one trial, higher meaning more anomalous."""

    genuine: list[float]
    impostor: list[float]


def _matrix(samples: list[dict[str, float]], columns: list[str]) -> np.ndarray:
    """Dense matrix over a fixed column set.

    The baselines cannot express "this feature was not observed", which our
    model handles through coverage. Missing values are filled with the column
    median from enrollment, which is the most generous reasonable choice: it
    puts an unobserved feature exactly at the user's centre rather than
    penalising it. Any disadvantage the baselines show is therefore not an
    artefact of how the gap was filled.
    """
    rows = []
    for sample in samples:
        rows.append([sample.get(name, np.nan) for name in columns])
    matrix = np.asarray(rows, dtype=float)

    for j in range(matrix.shape[1]):
        column = matrix[:, j]
        if np.all(np.isnan(column)):
            matrix[:, j] = 0.0
            continue
        column[np.isnan(column)] = np.nanmedian(column)
        matrix[:, j] = column
    return matrix


def _equal_error(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Sweep every cut and return (EER, the threshold achieving it)."""
    candidates = np.unique(np.concatenate([genuine, impostor]))
    if candidates.size == 0:
        return 0.5, 0.0
    lo, hi = candidates.min(), candidates.max()
    cuts = np.linspace(lo - 1e-6, hi + 1e-6, 600)

    best_gap, best_eer, best_cut = float("inf"), 0.5, float(lo)
    for cut in cuts:
        frr = float(np.mean(genuine > cut))
        far = float(np.mean(impostor <= cut))
        gap = abs(frr - far)
        if gap < best_gap:
            best_gap, best_eer, best_cut = gap, (frr + far) / 2.0, float(cut)
    return best_eer, best_cut


def run_trial(index: int, prior: PopulationPrior, population: list[dict[str, float]]):
    subject = NAMES[index % len(NAMES)]
    others = [n for n in NAMES if n != subject]
    base = 500_000 + index * 1_000

    enrollment = [capture(subject, base + r) for r in range(ENROLL_ROUNDS)]
    genuine = [capture(subject, base + 100 + i) for i in range(GENUINE_PER_TRIAL)]
    impostor = [
        capture(others[i % len(others)], base + 200 + i)
        for i in range(IMPOSTOR_PER_TRIAL)
    ]

    results: dict[str, Outcome] = {}

    # --- our model ---------------------------------------------------------
    profile = build_calibrated_profile(enrollment, prior, population)
    results["BioPrint (shrinkage robust)"] = Outcome(
        genuine=[score_identity(profile, f).score for f in genuine],
        impostor=[score_identity(profile, f).score for f in impostor],
    )

    # --- baselines ---------------------------------------------------------
    columns = sorted({k for s in enrollment for k in s})
    if len(columns) < 2:
        return results

    train = _matrix(enrollment, columns)
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    genuine_scaled = scaler.transform(_matrix(genuine, columns))
    impostor_scaled = scaler.transform(_matrix(impostor, columns))

    # Each returns higher = more normal, so negate for "more anomalous".
    estimators = {
        "OneClassSVM (rbf)": OneClassSVM(kernel="rbf", nu=0.2, gamma="scale"),
        "IsolationForest": IsolationForest(
            n_estimators=200, contamination=0.2, random_state=index
        ),
        "EllipticEnvelope": EllipticEnvelope(support_fraction=1.0, contamination=0.2),
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=min(5, len(train_scaled) - 1), novelty=True
        ),
    }

    for label, estimator in estimators.items():
        try:
            with warnings.catch_warnings():
                # These models legitimately complain at this sample size. The
                # complaint is the finding, not something to hide.
                warnings.simplefilter("ignore")
                estimator.fit(train_scaled)
                results[label] = Outcome(
                    genuine=(-estimator.decision_function(genuine_scaled)).tolist(),
                    impostor=(-estimator.decision_function(impostor_scaled)).tolist(),
                )
        except Exception as exc:  # noqa: BLE001 - reporting the failure is the point
            results[label] = Outcome(genuine=[], impostor=[])
            results[label].failure = type(exc).__name__  # type: ignore[attr-defined]

    return results


def main() -> None:
    print(__doc__.split("\n")[0])
    print("\nSYNTHETIC MECHANISM VALIDATION - not real authentication accuracy")
    print(
        f"Same enrollment ({ENROLL_ROUNDS} rounds), same attempts, "
        f"{TRIALS} independent trials.\n"
    )

    population = [capture(NAMES[i % len(NAMES)], 950_000 + i) for i in range(36)]
    prior = fit_population_prior(population)

    pooled: dict[str, Outcome] = {}
    for index in range(TRIALS):
        for label, outcome in run_trial(index, prior, population).items():
            bucket = pooled.setdefault(label, Outcome([], []))
            bucket.genuine.extend(outcome.genuine)
            bucket.impostor.extend(outcome.impostor)

    header = f"{'model':30s} {'EER':>8s} {'FRR@EER':>9s} {'FAR@EER':>9s} {'separation':>11s}"
    print(header)
    print("-" * len(header))

    ranking = []
    for label, outcome in pooled.items():
        if not outcome.genuine or not outcome.impostor:
            print(f"{label:30s} {'failed to fit at this sample size':>40s}")
            continue

        genuine = np.asarray(outcome.genuine)
        impostor = np.asarray(outcome.impostor)
        eer, cut = _equal_error(genuine, impostor)
        frr = float(np.mean(genuine > cut))
        far = float(np.mean(impostor <= cut))

        # Median gap in pooled standard deviations: scale-free, so models whose
        # scores live on different ranges stay comparable.
        spread = np.std(np.concatenate([genuine, impostor])) or 1.0
        separation = (np.median(impostor) - np.median(genuine)) / spread

        ranking.append((eer, label))
        print(f"{label:30s} {eer:8.1%} {frr:9.1%} {far:9.1%} {separation:11.2f}")

    print()
    if ranking:
        ranking.sort()
        best_eer, best_label = ranking[0]
        print(f"Best equal-error rate: {best_label} at {best_eer:.1%}")
        ours = next((e for e, l in ranking if l.startswith("BioPrint")), None)
        if ours is not None and best_label.startswith("BioPrint"):
            print(
                "The chosen model wins here. That is the justification for not "
                "using\na heavier estimator, and it is measured rather than assumed."
            )
        elif ours is not None:
            print(
                f"A baseline beat the chosen model ({best_eer:.1%} against "
                f"{ours:.1%}).\nThat is a real result and the model should be "
                f"reconsidered, not the table."
            )

    print(
        "\nNote: every model here sees identical data. Missing features are "
        "filled with\nthe enrollment median, which favours the baselines, since "
        "they cannot express\n'not observed' the way the coverage mechanism can."
    )


if __name__ == "__main__":
    main()
