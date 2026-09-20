#!/usr/bin/env python3
"""Precompute the Aalto population prior from raw Aalto 136M Keystrokes data.

Offline script. Run once; the output ships with the repo. It starts no server
and opens no database.

    python -m app.behavioral.fingerprint.precompute_aalto_prior \
        --data-dir evaluation/ml/data/aalto/extracted/Keystrokes/files \
        --output   data/aalto_prior_real.json \
        --max-users 0                       # 0 = all

What it does:

  1. Streams Aalto per-participant TSV files (one file per participant).
  2. Groups keystrokes by TEST_SECTION_ID (each = one transcribed sentence).
  3. Converts raw rows to KeyPress via the CANONICAL adapter in
     evaluation/ml/aalto_adapter.py — this script contains no keycode map and
     no feature mathematics of its own.
  4. Extracts keyboard features with the same `keyboard.extract` the browser
     pipeline uses, gated the same way production gates them.
  5. Aggregates per participant (median across that participant's sentences →
     exactly one observation per participant).
  6. Computes robust population statistics across participants.
  7. Saves a JSON artifact.

Why the per-participant median matters
--------------------------------------
Aalto participants contributed differing numbers of sentences. Pooling
sentences would let a participant with 30 sections count thirty times as much
as one with 5 toward "how much do humans vary". Collapsing to one vector per
participant first makes the population statistics a statement about *people*
rather than about *sentences*, which is what a population prior is for.

Dataset:
    Dhakal, Feit, Kristensson, Oulasvirta (2018).
    "Observations on Typing from 136 Million Keystrokes." CHI 2018.
    https://userinterfaces.aalto.fi/136Mkeystrokes/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral import stats as bpstats  # noqa: E402
from app.behavioral.features.registry import SPECS, Modality  # noqa: E402
from evaluation.ml.aalto_adapter import (  # noqa: E402
    UNOBSERVED_FEATURES,
    UNOBSERVED_REASON,
    SessionRejection,
    extract_session_features,
    iter_sessions,
)

log = logging.getLogger("bioprint.precompute_aalto")

ADAPTER_VERSION = "aalto_adapter/2026-09-20"

# A feature needs this many participants before a population statistic is
# worth stating at all.
MIN_PARTICIPANTS_PER_FEATURE = 10

# A participant needs this many usable sentences before their median is a
# median rather than a coin flip.
MIN_SESSIONS_PER_PARTICIPANT = 2

_AALTO_SHIFT_LIMITATION = (
    "Aalto keycode 16 does not distinguish ShiftLeft from ShiftRight. "
    "kbd_shift_right_ratio cannot be computed and is recorded as unobserved."
)


# ────────────────────────────────────────────────────────────────────────────
# Per-participant aggregation
# ────────────────────────────────────────────────────────────────────────────


def aggregate_user_features(
    sentence_features: list[dict[str, float]],
) -> dict[str, float] | None:
    """Collapse one participant's sentences into a single feature vector.

    Median across that participant's sentences, per feature. A feature seen in
    fewer than MIN_SESSIONS_PER_PARTICIPANT of their sentences is dropped
    rather than represented by a single observation.
    """
    if not sentence_features:
        return None

    all_names: set[str] = set()
    for sf in sentence_features:
        all_names.update(sf.keys())

    aggregated: dict[str, float] = {}
    for name in sorted(all_names):
        values = [sf[name] for sf in sentence_features if name in sf]
        if len(values) < MIN_SESSIONS_PER_PARTICIPANT:
            continue
        aggregated[name] = float(np.median(np.asarray(values, dtype=float)))

    return aggregated if aggregated else None


# ────────────────────────────────────────────────────────────────────────────
# Population statistics
# ────────────────────────────────────────────────────────────────────────────


def compute_population_stats(
    user_features: list[dict[str, float]],
) -> dict[str, dict[str, object]]:
    """Per-feature population statistics across participants.

    One participant contributes at most one value to each feature, so `n_users`
    is a participant count and not a sentence count.

    p10/p25/p75/p90 are kept because the shipped loader reads them. p5/p50/p95,
    min, max and the coverage figures are added alongside; the loader ignores
    keys it does not know, so this stays backward compatible in both directions.
    """
    total_participants = len(user_features)

    all_names: set[str] = set()
    for uf in user_features:
        all_names.update(uf.keys())

    population: dict[str, dict[str, object]] = {}
    for name in sorted(all_names):
        values = np.array(
            [uf[name] for uf in user_features if name in uf and np.isfinite(uf[name])],
            dtype=float,
        )
        if values.size < MIN_PARTICIPANTS_PER_FEATURE:
            log.info("Feature %s has only %d participants — skipping", name, values.size)
            continue

        missing = total_participants - values.size
        population[name] = {
            "population_median": round(float(np.median(values)), 6),
            "population_mad": round(float(bpstats.mad(values)), 6),
            "population_scale": round(float(bpstats.robust_scale(values)), 6),
            "p5": round(float(np.percentile(values, 5)), 6),
            "p10": round(float(np.percentile(values, 10)), 6),
            "p25": round(float(np.percentile(values, 25)), 6),
            "p50": round(float(np.percentile(values, 50)), 6),
            "p75": round(float(np.percentile(values, 75)), 6),
            "p90": round(float(np.percentile(values, 90)), 6),
            "p95": round(float(np.percentile(values, 95)), 6),
            "min": round(float(np.min(values)), 6),
            "max": round(float(np.max(values)), 6),
            "n_users": int(values.size),
            "missing_users": int(missing),
            "missing_pct": round(
                100.0 * missing / total_participants if total_participants else 0.0, 4
            ),
            "participant_coverage_pct": round(
                100.0 * values.size / total_participants if total_participants else 0.0, 4
            ),
            "status": "observed",
        }

    return population


def unobserved_entries() -> dict[str, dict[str, object]]:
    """Explicit records for features Aalto structurally cannot produce.

    Written into the artifact so the absence is a stated fact rather than a
    gap the reader has to notice. Carries no numeric fields at all: there is
    no median to round and no scale to shrink toward, and inventing a 0.0 for
    either would be indistinguishable from a measurement.
    """
    return {
        name: {
            "status": "unobserved",
            "reason": UNOBSERVED_REASON,
            "n_users": 0,
        }
        for name in sorted(UNOBSERVED_FEATURES)
    }


def validate_population_stats(stats_dict: dict[str, dict[str, object]]) -> list[str]:
    """Check computed statistics for problems. Empty list means all checks passed.

    Entries marked unobserved are skipped: they have no numbers to check, and
    checking them would be the same mistake as computing them.
    """
    warnings: list[str] = []

    for name, data in stats_dict.items():
        if data.get("status") == "unobserved":
            continue

        scale = float(data.get("population_scale", 0.0))
        median = float(data.get("population_median", 0.0))
        n = int(data.get("n_users", 0))

        if not np.isfinite(scale):
            warnings.append(f"{name}: population_scale is not finite")
        elif scale <= 0:
            warnings.append(f"{name}: population_scale is {scale} (non-positive)")
        if not np.isfinite(median):
            warnings.append(f"{name}: population_median is not finite")
        if n < 50:
            warnings.append(f"{name}: only {n} participants (want >=50 for stability)")

        # Percentiles must be monotone. Anything else means the array they were
        # computed from was not what we think it was.
        ordered = ["p5", "p10", "p25", "p50", "p75", "p90", "p95"]
        present = [(k, float(data[k])) for k in ordered if k in data]
        for (lo_name, lo), (hi_name, hi) in zip(present, present[1:]):
            if hi < lo:
                warnings.append(
                    f"{name}: {hi_name} ({hi}) < {lo_name} ({lo}) — percentiles not monotone"
                )

        if "p50" in data and "population_median" in data:
            if abs(float(data["p50"]) - median) > 1e-6:
                warnings.append(
                    f"{name}: p50 ({data['p50']}) disagrees with median ({median})"
                )

        lo, hi = data.get("min"), data.get("max")
        if lo is not None and hi is not None and float(hi) < float(lo):
            warnings.append(f"{name}: max < min")

    observed = [n for n, d in stats_dict.items() if d.get("status") != "unobserved"]
    keyboard_specs = [s for s in SPECS.values() if s.modality == Modality.KEYBOARD]
    expected = len(keyboard_specs) - len(UNOBSERVED_FEATURES)
    if len(observed) < expected:
        warnings.append(
            f"Only {len(observed)} of {expected} computable keyboard features found"
        )

    return warnings


def check_subset_stability(
    user_features: list[dict[str, float]],
    sizes: tuple[int, ...] = (500, 1000, 2000, 5000),
    seed: int = 42,
) -> dict[str, dict[int, float]]:
    """Robust scale at increasing subset sizes, to see whether it converges."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(user_features))

    stability: dict[str, dict[int, float]] = {}
    for size in sizes:
        if size > len(user_features):
            continue
        subset = [user_features[i] for i in indices[:size]]
        for name, data in compute_population_stats(subset).items():
            stability.setdefault(name, {})[size] = float(data["population_scale"])

    return stability


def compare_with_relative_spread(
    stats_dict: dict[str, dict[str, object]],
    relative_spread: float = 0.35,
) -> dict[str, dict[str, float]]:
    """Compare empirical scales against the RELATIVE_SPREAD fallback constant."""
    comparison: dict[str, dict[str, float]] = {}
    for name, data in stats_dict.items():
        if data.get("status") == "unobserved":
            continue
        median = float(data.get("population_median", 0.0))
        empirical_scale = float(data.get("population_scale", 0.0))
        current_scale = max(relative_spread * abs(median), 1e-3)

        ratio = empirical_scale / current_scale if current_scale > 0 else float("inf")
        comparison[name] = {
            "empirical_scale": round(empirical_scale, 6),
            "current_relative_scale": round(current_scale, 6),
            "ratio_empirical_over_current": round(ratio, 3),
        }
    return comparison


# ────────────────────────────────────────────────────────────────────────────
# Processing the corpus
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ProcessingReport:
    """Everything that happened during a pass over the corpus."""

    files_discovered: int = 0
    files_attempted: int = 0
    participants_processed: int = 0
    participants_failed: int = 0
    participants_contributing: int = 0
    raw_events: int = 0
    sessions_discovered: int = 0
    sessions_extracted: int = 0
    sessions_rejected: int = 0
    rejection_reasons: Counter = field(default_factory=Counter)
    feature_value_count: Counter = field(default_factory=Counter)
    elapsed_seconds: float = 0.0
    subset: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "files_discovered": self.files_discovered,
            "files_attempted": self.files_attempted,
            "participants_processed": self.participants_processed,
            "participants_failed": self.participants_failed,
            "participants_contributing": self.participants_contributing,
            "raw_events": self.raw_events,
            "sessions_discovered": self.sessions_discovered,
            "sessions_extracted": self.sessions_extracted,
            "sessions_rejected": self.sessions_rejected,
            "rejection_reasons": dict(self.rejection_reasons),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "subset": self.subset,
        }


def process_aalto_data(
    data_dir: Path,
    max_users: int = 0,
    progress_every: int = 2000,
) -> tuple[list[dict[str, float]], ProcessingReport]:
    """Stream Aalto files into one feature vector per participant.

    Memory is bounded by the number of participants, not the size of the
    corpus: each file is parsed one section at a time, each section collapses
    to a small dict of floats, and each participant collapses to one dict
    before the next file is opened.
    """
    files = sorted(data_dir.glob("*_keystrokes.txt"))
    if not files:
        files = sorted(data_dir.glob("*.txt"))

    report = ProcessingReport(files_discovered=len(files))
    if not files:
        log.error("No Aalto data files found in %s", data_dir)
        return [], report

    if max_users > 0:
        files = files[:max_users]
        report.subset = True
    report.files_attempted = len(files)

    log.info("Processing %d Aalto participant files from %s", len(files), data_dir)
    started = time.perf_counter()

    user_features: list[dict[str, float]] = []

    for i, filepath in enumerate(files, start=1):
        if progress_every and i % progress_every == 0:
            rate = i / max(time.perf_counter() - started, 1e-9)
            remaining = (len(files) - i) / rate if rate > 0 else 0
            log.info(
                "  %d/%d files (%d contributing) %.0f files/s, ~%.0f min left",
                i, len(files), len(user_features), rate, remaining / 60.0,
            )

        sentence_feats: list[dict[str, float]] = []
        had_any_section = False

        try:
            for _section_id, view, raw_rows in iter_sessions(filepath):
                had_any_section = True
                report.sessions_discovered += 1
                report.raw_events += raw_rows
                try:
                    feats = extract_session_features(view)
                except SessionRejection as exc:
                    report.sessions_rejected += 1
                    report.rejection_reasons[exc.reason] += 1
                    continue
                report.sessions_extracted += 1
                for name in feats:
                    report.feature_value_count[name] += 1
                sentence_feats.append(feats)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            report.participants_failed += 1
            report.rejection_reasons[f"file_error:{type(exc).__name__}"] += 1
            log.warning("Failed on %s: %s", filepath.name, exc)
            continue

        if not had_any_section:
            report.participants_failed += 1
            report.rejection_reasons["file_unreadable_or_empty"] += 1
            continue

        report.participants_processed += 1

        user_agg = aggregate_user_features(sentence_feats)
        if user_agg is not None:
            user_features.append(user_agg)

    report.participants_contributing = len(user_features)
    report.elapsed_seconds = time.perf_counter() - started

    log.info(
        "Done: %d files, %d participants processed, %d contributing, "
        "%d sessions extracted, %d rejected, %d raw events, %.1f min",
        report.files_attempted, report.participants_processed,
        report.participants_contributing, report.sessions_extracted,
        report.sessions_rejected, report.raw_events,
        report.elapsed_seconds / 60.0,
    )
    return user_features, report


# ────────────────────────────────────────────────────────────────────────────
# Artifact
# ────────────────────────────────────────────────────────────────────────────


def save_prior(
    stats_dict: dict[str, dict[str, object]],
    output_path: Path,
    n_users: int,
    source: str,
    note: str = "",
    dataset_type: str = "unknown",
    report: ProcessingReport | None = None,
    manifest: dict[str, object] | None = None,
) -> None:
    """Write the population prior to a JSON artifact.

    Unobserved features are written into the same `features` map with
    ``"status": "unobserved"`` and no numeric fields, so a reader sees the
    absence stated rather than having to infer it from a missing key.
    """
    observed = {n: d for n, d in stats_dict.items() if d.get("status") != "unobserved"}
    unobserved = {n: d for n, d in stats_dict.items() if d.get("status") == "unobserved"}

    payload: dict[str, object] = {
        "meta": {
            "source": source,
            "dataset_type": dataset_type,
            "n_users": n_users,
            "n_features": len(observed),
            "n_unobserved_features": len(unobserved),
            "observed_features": sorted(observed),
            "unobserved_features": sorted(unobserved),
            "adapter_version": ADAPTER_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "statistics_method": {
                "per_participant": "median across that participant's sentences",
                "min_sessions_per_participant": MIN_SESSIONS_PER_PARTICIPANT,
                "population_centre": "median across participants",
                "population_scale": "MAD * 1.4826 across participants "
                                    "(app.behavioral.stats.robust_scale)",
                "min_participants_per_feature": MIN_PARTICIPANTS_PER_FEATURE,
                "session_gating": "app.behavioral.features.extractor gates: enabled, "
                                  "finite, count >= SPECS[name].min_observations",
                "feature_extractor": "app.behavioral.features.keyboard.extract "
                                     "(the same function the browser pipeline calls)",
            },
            "citation": (
                "Dhakal, Feit, Kristensson, Oulasvirta. "
                "Observations on Typing from 136 Million Keystrokes. CHI 2018."
            ),
            "dataset_url": "https://userinterfaces.aalto.fi/136Mkeystrokes/",
            "note": note,
            "limitations": [
                _AALTO_SHIFT_LIMITATION,
                "Pointer features (7) not available — Aalto has no mouse data.",
                "Interaction features (4) not available — Aalto has no form events.",
                "Aalto captured desktop transcription typing, not browser form-filling.",
            ],
        },
        "features": {**observed, **unobserved},
    }

    if report is not None:
        payload["processing"] = report.as_dict()
    if manifest is not None:
        payload["manifest"] = manifest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info(
        "Saved prior: %d observed + %d unobserved features, %d participants → %s",
        len(observed), len(unobserved), n_users, output_path,
    )


def build_manifest(data_dir: Path, files_attempted: int) -> dict[str, object]:
    """A deterministic fingerprint of which files a run consumed.

    Hashing 168k file *contents* would take longer than the run itself. Hashing
    the sorted list of names plus their sizes is enough to establish later that
    the same corpus was processed, and costs a directory walk.
    """
    names: list[str] = []
    total_bytes = 0
    for path in sorted(data_dir.glob("*_keystrokes.txt"))[: files_attempted or None]:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        names.append(f"{path.name}:{size}")
        total_bytes += size

    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return {
        "input_dir": str(data_dir),
        "file_count": len(names),
        "total_bytes": total_bytes,
        "manifest_sha256": digest,
        "manifest_method": "sha256 over sorted 'filename:bytes' lines",
    }


# ────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute the Aalto population prior for BioPrint."
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing Aalto *_keystrokes.txt files")
    parser.add_argument("--output", type=Path,
                        default=_BACKEND / "data" / "aalto_prior_real.json",
                        help="Output JSON path (default: data/aalto_prior_real.json)")
    parser.add_argument("--max-users", type=int, default=0,
                        help="Maximum participant files to process (0 = all)")
    parser.add_argument("--check-stability", action="store_true",
                        help="Run subset stability analysis before the full prior")
    parser.add_argument("--no-manifest", action="store_true",
                        help="Skip the input manifest (saves a directory walk)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.output.exists():
        log.error(
            "Refusing to overwrite an existing artifact at %s. Move it aside or "
            "choose another --output.", args.output,
        )
        sys.exit(2)

    user_features, report = process_aalto_data(args.data_dir, args.max_users)
    if not user_features:
        log.error("No usable data. Check that --data-dir contains Aalto TSV files.")
        sys.exit(1)

    if args.check_stability and len(user_features) >= 1000:
        log.info("Subset stability check:")
        for name, sizes in sorted(check_subset_stability(user_features).items()):
            trail = " → ".join(f"{sz}: {scale:.4f}" for sz, scale in sorted(sizes.items()))
            log.info("  %s: %s", name, trail)

    stats_dict = compute_population_stats(user_features)
    stats_dict.update(unobserved_entries())

    warnings = validate_population_stats(stats_dict)
    if warnings:
        log.warning("Validation warnings:")
        for w in warnings:
            log.warning("  - %s", w)
    else:
        log.info("Validation: all checks passed.")

    log.info("Empirical scale vs RELATIVE_SPREAD=0.35 fallback:")
    for name, comp in sorted(compare_with_relative_spread(stats_dict).items()):
        log.info(
            "  %-32s empirical=%10.4f  fallback=%10.4f  ratio=%.2fx",
            name, comp["empirical_scale"], comp["current_relative_scale"],
            comp["ratio_empirical_over_current"],
        )

    n_users = report.participants_contributing
    if report.subset:
        source = f"aalto_136m_subset_{n_users}"
        note = (
            f"SUBSET: {report.files_attempted} of {report.files_discovered} "
            f"participant files processed; {n_users} contributed."
        )
    else:
        source = "aalto_136m"
        note = (
            f"Full local corpus: {report.files_attempted} participant files "
            f"processed; {n_users} contributed usable feature vectors."
        )

    manifest = None if args.no_manifest else build_manifest(args.data_dir, args.max_users)

    save_prior(
        stats_dict, args.output, n_users, source, note,
        dataset_type="real", report=report, manifest=manifest,
    )
    log.info("Done. Prior saved to %s", args.output)


if __name__ == "__main__":
    main()
