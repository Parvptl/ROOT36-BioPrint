"""Reliability sweep over many independent enrollments.

A single enrolled profile tells you almost nothing: one unluckily consistent
set of five rounds produces a tight threshold, and one unluckily variable set
produces a loose one. What matters is the distribution across enrollments, so
this repeats the whole enroll-then-authenticate cycle many times.

SYNTHETIC MECHANISM VALIDATION. Every typist here is generated. These numbers
describe how the algorithm behaves on controlled input whose properties we
chose; they are NOT authentication accuracy on real people, and must never be
reported as such. Real rates require real captures, which is what
evaluation/run_evaluation.py consumes.

Run:
    cd backend
    ./.venv/Scripts/python.exe ../evaluation/reliability_sweep.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

from app.auth.challenge import generate_phrase  # noqa: E402
from app.behavioral.features.extractor import extract_from_session  # noqa: E402
from app.behavioral.fingerprint.calibration import build_calibrated_profile  # noqa: E402
from app.behavioral.fingerprint.population import PopulationPrior, fit_population_prior  # noqa: E402
from app.behavioral.fingerprint.scoring import score_identity  # noqa: E402
from tests.factories import TypingStyle, human_session  # noqa: E402

TRIALS = 40
ENROLL_ROUNDS = 5
GENUINE_PER_TRIAL = 8
IMPOSTOR_PER_TRIAL = 8

# A population of distinct typists. The "impostor" for each trial is drawn from
# the others, so impostors are ordinary different people rather than
# caricatures chosen to be easy to reject.
POPULATION = {
    "alice":  TypingStyle(iki_mean_ms=140, dwell_mean_ms=82,  overlap_prob=0.45, right_shift_prob=0.95, tab_between_fields=True,  pointer_speed=1.6),
    "bob":    TypingStyle(iki_mean_ms=200, dwell_mean_ms=105, overlap_prob=0.20, right_shift_prob=0.30, tab_between_fields=True,  pointer_speed=1.1),
    "carol":  TypingStyle(iki_mean_ms=260, dwell_mean_ms=120, overlap_prob=0.05, right_shift_prob=0.10, tab_between_fields=False, pointer_speed=0.8),
    "dan":    TypingStyle(iki_mean_ms=165, dwell_mean_ms=95,  overlap_prob=0.30, right_shift_prob=0.60, tab_between_fields=True,  pointer_speed=1.3),
    "erin":   TypingStyle(iki_mean_ms=120, dwell_mean_ms=70,  overlap_prob=0.60, right_shift_prob=0.85, tab_between_fields=False, pointer_speed=1.9),
    "frank":  TypingStyle(iki_mean_ms=230, dwell_mean_ms=110, overlap_prob=0.12, right_shift_prob=0.20, tab_between_fields=True,  pointer_speed=0.95),
}
NAMES = list(POPULATION)


def capture(who: str, seed: int) -> dict[str, float]:
    _, features = extract_from_session(
        human_session(generate_phrase(), style=POPULATION[who], seed=seed)
    )
    return features.as_dict()


@dataclass
class Trial:
    threshold: float
    source: str
    genuine: list[float]
    impostor: list[float]

    @property
    def false_rejections(self) -> int:
        return sum(1 for s in self.genuine if s > self.threshold)

    @property
    def false_acceptances(self) -> int:
        return sum(1 for s in self.impostor if s <= self.threshold)


def run_trial(index: int, prior: PopulationPrior, samples: list[dict[str, float]]) -> Trial:
    subject = NAMES[index % len(NAMES)]
    others = [n for n in NAMES if n != subject]

    base = 100_000 + index * 1_000
    enrollment = [capture(subject, base + r) for r in range(ENROLL_ROUNDS)]
    profile = build_calibrated_profile(enrollment, prior, samples)

    genuine = [
        score_identity(profile, capture(subject, base + 100 + i)).score
        for i in range(GENUINE_PER_TRIAL)
    ]
    impostor = [
        score_identity(
            profile, capture(others[i % len(others)], base + 200 + i)
        ).score
        for i in range(IMPOSTOR_PER_TRIAL)
    ]
    return Trial(profile.threshold, profile.threshold_source, genuine, impostor)


def summarise(label: str, trials: list[Trial]) -> None:
    genuine = np.array([s for t in trials for s in t.genuine])
    impostor = np.array([s for t in trials for s in t.impostor])
    thresholds = np.array([t.threshold for t in trials])

    fr = sum(t.false_rejections for t in trials)
    fa = sum(t.false_acceptances for t in trials)

    print(f"\n--- {label} ---")
    print(f"  trials {len(trials)}  genuine attempts {genuine.size}  impostor attempts {impostor.size}")
    print(f"  threshold      median {np.median(thresholds):.4f}   range {thresholds.min():.4f} to {thresholds.max():.4f}")
    print(f"  genuine score  median {np.median(genuine):.4f}   p95 {np.percentile(genuine, 95):.4f}   max {genuine.max():.4f}")
    print(f"  impostor score median {np.median(impostor):.4f}   p5  {np.percentile(impostor, 5):.4f}   min {impostor.min():.4f}")
    print(f"  false rejection rate {fr / genuine.size:.1%}  ({fr}/{genuine.size})")
    print(f"  false acceptance rate {fa / impostor.size:.1%}  ({fa}/{impostor.size})")

    # How separable the two populations are regardless of where the cut sits.
    # A single global threshold that would be optimal across every trial.
    cuts = np.linspace(0.02, 0.8, 400)
    best = min(
        cuts,
        key=lambda c: np.mean(genuine > c) + np.mean(impostor <= c),
    )
    print(
        f"  best single global cut {best:.4f} -> FRR {np.mean(genuine > best):.1%}, "
        f"FAR {np.mean(impostor <= best):.1%}"
    )
    overlap = (genuine.max() > impostor.min())
    print(f"  distributions overlap: {'YES' if overlap else 'no'}")


def main() -> None:
    print("SYNTHETIC MECHANISM VALIDATION - not real authentication accuracy\n")

    print("Pass 1: no population prior (a first user on a fresh deployment)")
    cold = [run_trial(i, PopulationPrior(), []) for i in range(TRIALS)]
    summarise("cold start, threshold_source=genuine_only", cold)

    # Pass 2: a prior built from people other than the subjects, as would
    # accumulate once several users have enrolled.
    print("\nBuilding a population prior from consented samples...")
    samples = [capture(NAMES[i % len(NAMES)], 900_000 + i) for i in range(36)]
    prior = fit_population_prior(samples)
    print(f"  prior from {prior.sample_count} samples, informative={prior.is_informative}")

    warm = [run_trial(i, prior, samples) for i in range(TRIALS)]
    summarise("with population prior", warm)

    sources = {}
    for t in warm:
        sources[t.source] = sources.get(t.source, 0) + 1
    print(f"\n  threshold sources used: {sources}")


if __name__ == "__main__":
    main()
