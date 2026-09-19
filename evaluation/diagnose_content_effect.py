"""Does the challenge phrase move the fingerprint more than the person does?

The challenge phrase is randomised for replay resistance. That is only sound if
changing the text does not, by itself, make a genuine user look like someone
else. This script measures whether that holds, per feature.

Three variance sources are separated:

  WITHIN   same typist, same phrase, different runs
           -> irreducible noise; the floor any feature has

  CONTENT  same typist, different phrases
           -> WITHIN plus whatever the text itself contributes

  BETWEEN  different typists
           -> the signal the whole system depends on

A feature is content-sensitive when CONTENT sits well above WITHIN relative to
BETWEEN: the phrase is moving it nearly as much as a different person would,
so the feature is partly measuring the text rather than the typist.

Run:
    cd backend
    ./.venv/Scripts/python.exe ../evaluation/diagnose_content_effect.py

Note on data: this uses synthetic typists. It is a diagnostic of the feature
design under controlled conditions, not a measurement of real human behaviour.
It can prove a feature reacts to content; it cannot tell you how much real
people vary.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

from app.auth.challenge import generate_phrase  # noqa: E402
from app.behavioral import stats  # noqa: E402
from app.behavioral.features.extractor import extract_from_session  # noqa: E402
from app.behavioral.features.registry import SPECS, enabled_names  # noqa: E402
from tests.factories import TypingStyle, human_session  # noqa: E402

SAMPLES = 40

ALICE = TypingStyle(
    iki_mean_ms=140.0, dwell_mean_ms=82.0, overlap_prob=0.45,
    right_shift_prob=0.95, tab_between_fields=True, pointer_speed=1.6,
)
BOB = TypingStyle(
    iki_mean_ms=200.0, dwell_mean_ms=105.0, overlap_prob=0.20,
    right_shift_prob=0.30, tab_between_fields=True, pointer_speed=1.1,
)
CAROL = TypingStyle(
    iki_mean_ms=260.0, dwell_mean_ms=120.0, overlap_prob=0.05,
    right_shift_prob=0.10, tab_between_fields=False, pointer_speed=0.8,
)

FIXED_PHRASE = "amber Willow granite pattern Copper thunder meadow"


def capture(style: TypingStyle, phrase: str, seed: int) -> dict[str, float]:
    _, features = extract_from_session(human_session(phrase, style=style, seed=seed))
    return features.as_dict()


def spread(samples: list[dict[str, float]], name: str) -> float | None:
    """Robust scale of one feature across samples, or None if too few."""
    values = np.array(
        [s[name] for s in samples if name in s and np.isfinite(s[name])], dtype=float
    )
    if values.size < 5:
        return None
    return stats.robust_scale(values)


def presence(samples: list[dict[str, float]], name: str) -> float:
    return sum(1 for s in samples if name in s) / len(samples)


def main() -> None:
    print("Collecting synthetic captures...\n")

    within = [capture(ALICE, FIXED_PHRASE, seed) for seed in range(SAMPLES)]
    content = [capture(ALICE, generate_phrase(), 1000 + seed) for seed in range(SAMPLES)]
    between = (
        [capture(ALICE, generate_phrase(), 2000 + s) for s in range(SAMPLES // 3)]
        + [capture(BOB, generate_phrase(), 3000 + s) for s in range(SAMPLES // 3)]
        + [capture(CAROL, generate_phrase(), 4000 + s) for s in range(SAMPLES // 3)]
    )

    rows = []
    for name in enabled_names():
        s_within = spread(within, name)
        s_content = spread(content, name)
        s_between = spread(between, name)
        if s_within is None or s_content is None or s_between is None:
            rows.append((name, None, None, None, None, presence(content, name)))
            continue

        # Variance the text adds on top of run-to-run noise, as a share of the
        # between-person variance the feature is supposed to supply.
        extra = max(0.0, s_content**2 - s_within**2)
        if s_between <= 1e-12:
            # No between-person spread in this fixture set, so the ratio is 0/0
            # and means nothing. Reporting it as "content-sensitive" would be a
            # division artefact, not a finding. Flagged separately: it says the
            # synthetic typists fail to differ on this axis, which is a gap in
            # the fixtures rather than a defect in the feature.
            ratio = None
        else:
            ratio = extra / (s_between**2)
        rows.append((name, s_within, s_content, s_between, ratio, presence(content, name)))

    rows.sort(key=lambda r: (r[4] is None, -(r[4] or 0)))

    header = f"{'feature':34s} {'within':>9s} {'content':>9s} {'between':>9s} {'C/B':>7s} {'seen':>6s}"
    print(header)
    print("-" * len(header))
    no_signal = []
    for name, w, c, b, ratio, seen in rows:
        if w is None:
            print(f"{name:34s} {'--':>9s} {'--':>9s} {'--':>9s} {'--':>7s} {seen:5.0%}")
            continue
        if ratio is None:
            no_signal.append(name)
            print(f"{name:34s} {w:9.4f} {c:9.4f} {b:9.4f} {'n/a':>7s} {seen:5.0%}")
            continue
        flag = "  <== content-sensitive" if ratio > 0.30 else ""
        print(f"{name:34s} {w:9.4f} {c:9.4f} {b:9.4f} {ratio:7.2f} {seen:5.0%}{flag}")

    print()
    print("C/B is the share of between-person variance that the phrase alone")
    print("reproduces. Above ~0.30 the feature is substantially measuring the")
    print("text rather than the typist.")
    print()
    print("Low 'seen' matters too: a feature present in only some captures gives")
    print("the profile little to fit and appears or vanishes between attempts.")

    flagged = [r[0] for r in rows if r[4] is not None and r[4] > 0.30]
    thin = [r[0] for r in rows if r[5] < 0.9]
    print()
    print(f"content-sensitive : {flagged or 'none'}")
    print(f"intermittent      : {thin or 'none'}")
    print(f"no between-signal : {no_signal or 'none'}  (fixture gap, not a feature defect)")
    for name in sorted(set(flagged) | set(thin)):
        print(f"    {name}: min_observations={SPECS[name].min_observations}")

    _score_decomposition()


def _score_decomposition() -> None:
    """Where does a genuine user's deviation actually come from?

    Builds a profile the way enrollment does, scores fresh genuine logins
    against it, and reports which features carry the deviation. If the top
    contributors are the same features the variance table flags, the two
    measurements agree on the cause.
    """
    from app.behavioral.fingerprint.calibration import build_calibrated_profile
    from app.behavioral.fingerprint.population import PopulationPrior
    from app.behavioral.fingerprint.scoring import score_identity

    print("\n" + "=" * 79)
    print("IDENTITY SCORE DECOMPOSITION (genuine user, fresh random phrases)")
    print("=" * 79 + "\n")

    enrollment = [capture(ALICE, generate_phrase(), 5000 + i) for i in range(5)]
    profile = build_calibrated_profile(enrollment, PopulationPrior(), [])

    genuine = [capture(ALICE, generate_phrase(), 6000 + i) for i in range(30)]
    impostor = [capture(BOB, generate_phrase(), 7000 + i) for i in range(30)]

    g_scores = [score_identity(profile, f).score for f in genuine]
    i_scores = [score_identity(profile, f).score for f in impostor]

    print(f"threshold {profile.threshold:.4f}  ({profile.threshold_source})")
    print(
        f"genuine   n={len(g_scores)} med={np.median(g_scores):.4f} "
        f"max={max(g_scores):.4f}  rejected={sum(s > profile.threshold for s in g_scores)}"
    )
    print(
        f"impostor  n={len(i_scores)} med={np.median(i_scores):.4f} "
        f"min={min(i_scores):.4f}  accepted={sum(s <= profile.threshold for s in i_scores)}"
    )

    # Average share of a genuine attempt's deviation carried by each feature.
    totals: dict[str, float] = {}
    for features in genuine:
        for contribution in score_identity(profile, features).contributions:
            totals[contribution.name] = totals.get(contribution.name, 0.0) + contribution.share

    print("\ntop contributors to GENUINE deviation (mean share of the total):")
    for name, total in sorted(totals.items(), key=lambda kv: -kv[1])[:10]:
        spec = SPECS[name]
        print(
            f"  {name:34s} {total / len(genuine):6.1%}   "
            f"min_obs={spec.min_observations:<3d} {spec.modality.value}"
        )
    print("\nA genuine user's deviation should be spread thinly across many")
    print("features. Any feature carrying a large share is driving false")
    print("rejections on its own.")


if __name__ == "__main__":
    main()
