#!/usr/bin/env python3
"""Generate a bootstrap population prior from synthetic typists.

This script generates an initial population prior using BioPrint's own
synthetic data generation (factories.py) rather than real Aalto data.

WHY THIS EXISTS:

The Aalto 136M Keystrokes dataset is 1.4 GB compressed / 16 GB uncompressed.
Downloading and processing it during the hackathon is impractical.  This script
generates a defensible bootstrap prior by:

1. Creating 500 synthetic typists with diverse, independently-varied parameters
2. Generating multiple sentences per typist using the SAME feature extraction
   pipeline the production system uses
3. Computing per-feature population statistics across all typists

The bootstrap prior is CLEARLY LABELED as synthetic, not Aalto-derived.
When real Aalto data is processed via precompute_aalto_prior.py, the bootstrap
is replaced.

WHAT MAKES THIS DEFENSIBLE:

- The features are computed by the SAME extractor (keyboard.py)
- The typing parameters span empirically justified ranges
- Per-feature scales are independent (not a single RELATIVE_SPREAD constant)
- The statistics are reproducible (fixed seeds)
- The source is documented in the artifact

WHAT THIS IS NOT:

- This is NOT "Aalto-derived".  The artifact says "bootstrap_synthetic".
- This does NOT claim to represent real human variability.
- This IS a structured per-feature prior, which is strictly better than the
  current single-constant RELATIVE_SPREAD=0.35 for all features.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral import stats as bpstats
from app.behavioral.events import build_session_view
from app.behavioral.features import keyboard
from app.behavioral.features.registry import Modality, SPECS

log = logging.getLogger("bioprint.bootstrap_prior")


# Challenge phrases: diverse enough to exercise different letter distributions.
_PHRASES = [
    "the quick brown fox jumps over the lazy dog near Sunny fields",
    "seven Wild elephants Danced across the bright meadow at noon",
    "clever foxes Navigate through Hidden valleys and steep Mountains",
    "ancient rivers Flow beneath the Towering marble bridges Quietly",
    "purple Clouds drifted above the Sleeping coastal village today",
    "bright Yellow butterflies Explored the garden throughout warm spring",
    "small Children played Beyond the wooden fence until Evening came",
    "distant Thunder echoed Through the misty mountain Valley below",
    "golden Autumn leaves Scattered across the empty cobblestone road",
    "Fresh morning Dew covered Every blade of grass in Silent meadows",
    "curious Rabbits hopped Beside the crystal clear Stream at dawn",
    "Tall pine Trees swayed Gently in the cool evening breeze tonight",
]


def _make_typing_style(rng: np.random.RandomState) -> dict:
    """Generate diverse typing parameters spanning real human ranges.

    Ranges are based on published keystroke dynamics research:
    - Dhakal et al. (2018): median WPM ~52, range ~20-120
    - Killourhy & Maxion (2009): hold times 92-127ms across subjects
    - General literature: IKI range 80-400ms for free text
    """
    # Inter-key interval: 80ms (fast touch typist) to 350ms (slow hunt-and-peck)
    iki_mean = float(rng.uniform(80, 350))
    # Dwell: 50ms (light tapper) to 170ms (heavy presser)
    dwell_mean = float(rng.uniform(50, 170))
    # Overlap probability: 0.0 (hunt-and-peck) to 0.5 (very fluent)
    overlap = float(rng.uniform(0.0, 0.50))
    # Jitter: 0.15 (consistent) to 0.55 (erratic)
    iki_jitter = float(rng.uniform(0.15, 0.55))
    dwell_jitter = float(rng.uniform(0.12, 0.45))
    # Pause probability: 0.0 (no pauses) to 0.12 (frequent pauses)
    pause_prob = float(rng.uniform(0.0, 0.12))
    # Backspace rate: 0.0 (no errors) to 0.10 (frequent corrections)
    backspace_prob = float(rng.uniform(0.0, 0.10))
    # Right shift preference: 0.0 (always left) to 1.0 (always right)
    right_shift = float(rng.choice([0.0, 0.1, 0.2, 0.8, 0.9, 1.0]))

    return {
        "iki_mean_ms": iki_mean,
        "iki_jitter": iki_jitter,
        "dwell_mean_ms": dwell_mean,
        "dwell_jitter": dwell_jitter,
        "overlap_prob": overlap,
        "pause_prob": pause_prob,
        "pause_ms": float(rng.uniform(400, 1200)),
        "backspace_prob": backspace_prob,
        "right_shift_prob": right_shift,
        "tab_between_fields": bool(rng.random() > 0.3),
        "focus_delay_ms": float(rng.uniform(150, 600)),
        "presubmit_ms": float(rng.uniform(300, 1200)),
        "pointer_speed": float(rng.uniform(0.5, 3.0)),
        "pointer_wobble": float(rng.uniform(0.8, 4.0)),
    }


def _generate_session(phrase: str, style_params: dict, seed: int):
    """Generate a synthetic session and extract keyboard features."""
    import random as stdlib_random
    from tests.factories import TypingStyle, human_session

    style = TypingStyle(**style_params)
    session = human_session(
        phrase=phrase,
        nonce=f"bootstrap-{seed:08d}",
        style=style,
        seed=seed,
        use_pointer=False,  # Keyboard only — matching Aalto's data
    )

    view = build_session_view(session)
    raw = keyboard.extract(view)

    features: dict[str, float] = {}
    for name, (value, n_obs) in raw.items():
        spec = SPECS.get(name)
        if spec is None or spec.modality != Modality.KEYBOARD:
            continue
        if n_obs < spec.min_observations:
            continue
        if not np.isfinite(value):
            continue
        features[name] = value

    return features if features else None


def generate_bootstrap_prior(
    n_users: int = 500,
    sentences_per_user: int = 5,
    master_seed: int = 20260920,
    output_path: Path | None = None,
) -> dict:
    """Generate a bootstrap population prior from synthetic typists.

    Returns the population statistics dict.
    """
    rng = np.random.RandomState(master_seed)

    log.info("Generating bootstrap prior: %d users × %d sentences", n_users, sentences_per_user)

    user_features: list[dict[str, float]] = []

    for user_idx in range(n_users):
        style_params = _make_typing_style(rng)
        sentence_feats: list[dict[str, float]] = []

        for sent_idx in range(sentences_per_user):
            phrase = _PHRASES[(user_idx + sent_idx) % len(_PHRASES)]
            seed = master_seed + user_idx * 1000 + sent_idx
            feats = _generate_session(phrase, style_params, seed)
            if feats is not None:
                sentence_feats.append(feats)

        if len(sentence_feats) >= 2:
            # Aggregate: median across sentences
            all_names = set()
            for sf in sentence_feats:
                all_names.update(sf.keys())

            user_agg: dict[str, float] = {}
            for name in all_names:
                values = [sf[name] for sf in sentence_feats if name in sf]
                if len(values) >= 2:
                    user_agg[name] = float(np.median(values))

            if user_agg:
                user_features.append(user_agg)

        if (user_idx + 1) % 100 == 0:
            log.info("  ... generated %d/%d users", user_idx + 1, n_users)

    log.info("Generated %d usable user profiles from %d attempts", len(user_features), n_users)

    # Compute population statistics
    from app.behavioral.fingerprint.precompute_aalto_prior import (
        compute_population_stats,
        validate_population_stats,
        compare_with_relative_spread,
        save_prior,
    )

    stats_dict = compute_population_stats(user_features)

    # Validate
    warnings = validate_population_stats(stats_dict)
    if warnings:
        log.warning("Bootstrap prior validation warnings:")
        for w in warnings:
            log.warning("  - %s", w)

    # Comparison
    comparison = compare_with_relative_spread(stats_dict)
    log.info("Comparison with current RELATIVE_SPREAD=0.35:")
    for name, comp in sorted(comparison.items()):
        log.info(
            "  %s: bootstrap=%.4f  current=%.4f  ratio=%.2fx",
            name, comp["empirical_scale"], comp["current_relative_scale"],
            comp["ratio_empirical_over_current"],
        )

    # Save
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "data" / "aalto_prior.json"

    save_prior(
        stats_dict,
        output_path,
        n_users=len(user_features),
        source="bootstrap_synthetic",
        note=(
            f"Bootstrap prior from {len(user_features)} synthetic typists with diverse "
            f"parameters. NOT derived from real Aalto data. Provides per-feature scales "
            f"which are strictly better than the single RELATIVE_SPREAD=0.35 constant. "
            f"Replace by running precompute_aalto_prior.py with real Aalto data."
        ),
    )

    return stats_dict


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    output = Path(__file__).resolve().parent / "data" / "aalto_prior.json"
    generate_bootstrap_prior(output_path=output)
