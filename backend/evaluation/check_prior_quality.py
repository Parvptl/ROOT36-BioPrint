#!/usr/bin/env python3
"""Data-quality checks over a generated population prior.

Reports anomalies; does not fix them. Silently "cleaning" an unusual value in a
population prior would hide the thing the prior exists to tell us.

Some values that look wrong are not:

  * **negative flight times are legitimate.** Flight is release-of-one-key to
    press-of-the-next, so a negative value means the next key went down before
    the previous came up. That is fluent touch typing, and the proportion of
    overlapping transitions is one of the strongest person-level signals we
    have. A prior with `kbd_flight_negative_frac` pinned at zero is the
    suspicious one.
  * **negative dwell is not.** `KeyPress.dwell` returns None for a negative or
    implausibly long hold, so such rows never reach a feature. If a dwell
    statistic came out negative, the structure was bypassed.

    cd backend
    ./.venv/Scripts/python.exe evaluation/check_prior_quality.py data/aalto_prior_real.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.features.registry import SPECS, Modality  # noqa: E402

# Features whose values cannot be negative without something being wrong.
# Everything derived from a duration-difference that may legitimately invert
# (flight) is deliberately absent from this set.
NON_NEGATIVE = {
    "kbd_dwell_median", "kbd_dwell_mad",
    "kbd_dwell_left_median", "kbd_dwell_right_median",
    "kbd_flight_iqr", "kbd_flight_negative_frac",
    "kbd_logiki_mad", "kbd_burst_fraction", "kbd_pause_rate",
    "kbd_backspace_rate", "kbd_backspace_latency_median",
    "kbd_speed_kps", "kbd_speed_cv", "kbd_shift_right_ratio",
}

# Features bounded to [0, 1] by their own definition.
FRACTIONS = {
    "kbd_flight_negative_frac", "kbd_burst_fraction",
    "kbd_backspace_rate", "kbd_shift_right_ratio",
}

MIN_REASONABLE_COVERAGE_PCT = 50.0
PERCENTILES = ("p5", "p10", "p25", "p50", "p75", "p90", "p95")


def check(path: Path) -> tuple[list[str], list[str], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload["features"]

    problems: list[str] = []
    notes: list[str] = []
    counts = {"observed": 0, "unobserved": 0, "checked": 0}

    for name in sorted(features):
        data = features[name]

        if str(data.get("status", "observed")) == "unobserved":
            counts["unobserved"] += 1
            for numeric in ("population_median", "population_mad", "population_scale"):
                if numeric in data:
                    problems.append(
                        f"{name}: marked unobserved but carries {numeric} — an "
                        f"unobserved feature must have no statistics at all"
                    )
            if not data.get("reason"):
                problems.append(f"{name}: unobserved without a stated reason")
            continue

        counts["observed"] += 1
        spec = SPECS.get(name)
        if spec is None:
            problems.append(f"{name}: not in the feature registry")
            continue
        if spec.modality is not Modality.KEYBOARD:
            problems.append(
                f"{name}: {spec.modality.value} feature in an Aalto prior — "
                f"Aalto has no {spec.modality.value} data"
            )
            continue

        counts["checked"] += 1
        median = float(data["population_median"])
        mad = float(data["population_mad"])
        scale = float(data["population_scale"])

        # --- finiteness ----------------------------------------------------
        for key in ("population_median", "population_mad", "population_scale",
                    "min", "max", *PERCENTILES):
            if key in data:
                value = float(data[key])
                if math.isnan(value):
                    problems.append(f"{name}: {key} is NaN")
                elif math.isinf(value):
                    problems.append(f"{name}: {key} is infinite")

        # --- scale ---------------------------------------------------------
        if scale < 0:
            problems.append(f"{name}: population_scale is negative ({scale})")
        elif scale == 0.0:
            problems.append(
                f"{name}: population_scale is exactly 0 — zero variance across "
                f"{data.get('n_users')} participants. This reads as 'nobody "
                f"differs', which is almost never true of a timing feature and "
                f"usually means the value was constant by construction."
            )
        if mad < 0:
            problems.append(f"{name}: population_mad is negative ({mad})")

        # --- sign and bounds ------------------------------------------------
        if name in NON_NEGATIVE:
            for key in ("population_median", "min", *PERCENTILES):
                if key in data and float(data[key]) < 0:
                    problems.append(
                        f"{name}: {key} is negative ({data[key]}) but this "
                        f"feature cannot be"
                    )
        if name in FRACTIONS:
            hi = float(data.get("max", median))
            if hi > 1.0 + 1e-9:
                problems.append(f"{name}: max {hi} exceeds 1.0 for a fraction")

        # --- percentile ordering ---------------------------------------------
        present = [(k, float(data[k])) for k in PERCENTILES if k in data]
        for (lo_name, lo), (hi_name, hi) in zip(present, present[1:]):
            if hi < lo:
                problems.append(
                    f"{name}: {hi_name} ({hi}) < {lo_name} ({lo}) — percentiles "
                    f"not monotone"
                )
        if "p25" in data and "p75" in data:
            if not (float(data["p25"]) <= median <= float(data["p75"])):
                problems.append(
                    f"{name}: median {median} outside [p25, p75] "
                    f"[{data['p25']}, {data['p75']}]"
                )
        if "min" in data and "max" in data:
            lo, hi = float(data["min"]), float(data["max"])
            if hi < lo:
                problems.append(f"{name}: max ({hi}) < min ({lo})")
            if not (lo <= median <= hi):
                problems.append(f"{name}: median outside [min, max]")

        # --- coverage --------------------------------------------------------
        coverage = data.get("participant_coverage_pct")
        if coverage is not None and float(coverage) < MIN_REASONABLE_COVERAGE_PCT:
            notes.append(
                f"{name}: only {coverage}% of participants supplied this "
                f"feature (n={data.get('n_users')})"
            )

        # --- distribution shape ------------------------------------------------
        if scale > 0 and abs(median) > 0 and scale / abs(median) > 1.0:
            notes.append(
                f"{name}: spread exceeds the centre (scale {scale:.4f} vs "
                f"median {median:.4f}) — a very wide population"
            )

    return problems, notes, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path, nargs="?",
                        default=_BACKEND / "data" / "aalto_prior_real.json")
    args = parser.parse_args()

    if not args.prior.exists():
        sys.exit(f"prior not found: {args.prior}")

    problems, notes, counts = check(args.prior)

    print(f"Quality check: {args.prior}")
    print(f"  {counts['observed']} observed, {counts['unobserved']} unobserved, "
          f"{counts['checked']} fully checked")
    print()

    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("PROBLEMS: none.")

    print()
    if notes:
        print(f"NOTES ({len(notes)}) — unusual but not necessarily wrong:")
        for n in notes:
            print(f"  - {n}")
    else:
        print("NOTES: none.")

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
