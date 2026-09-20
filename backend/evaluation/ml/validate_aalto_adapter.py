"""Validate the Aalto adapter on a sample of participants.

Prints per-feature distributions so the adapter's output can be eyeballed
against what a keyboard feature ought to look like, before anything is spent
walking the full corpus.

Reports the gated and ungated paths side by side, because the difference
between them is exactly the set of feature values the running system would
have thrown away:

    ungated  every value keyboard.extract produced
    gated    only values with enough observations behind them, as production
             requires (app/behavioral/features/extractor.py)

    cd backend
    ./.venv/Scripts/python.exe evaluation/ml/validate_aalto_adapter.py --files 10
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from evaluation.ml.aalto_adapter import (  # noqa: E402
    UNOBSERVED_FEATURES,
    SessionRejection,
    extract_features,
    extract_session_features,
    iter_sessions,
)

DEFAULT_DATA_DIR = (
    _BACKEND / "evaluation" / "ml" / "data" / "aalto" / "extracted" / "Keystrokes" / "files"
)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--files", type=int, default=10)
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*_keystrokes.txt"))[: args.files]
    if not files:
        sys.exit(f"no Aalto files under {args.data_dir}")

    raw_events = 0
    sessions = 0
    gated_ok = 0
    rejections: dict[str, int] = {}
    gated: dict[str, list[float]] = {}
    ungated: dict[str, list[float]] = {}
    participants = set()

    for path in files:
        participants.add(path.name.split("_")[0])
        for _section_id, view, rows in iter_sessions(path):
            sessions += 1
            raw_events += rows

            for name, value in extract_features(view).items():
                if isinstance(value, float) and math.isfinite(value):
                    ungated.setdefault(name, []).append(value)

            try:
                features = extract_session_features(view)
            except SessionRejection as exc:
                rejections[exc.reason] = rejections.get(exc.reason, 0) + 1
                continue
            gated_ok += 1
            for name, value in features.items():
                gated.setdefault(name, []).append(value)

    print("=== Validation Summary ===")
    print(f"Participant files : {len(files)}")
    print(f"Participants      : {len(participants)}")
    print(f"Raw events parsed : {raw_events}")
    print(f"Sessions found    : {sessions}")
    print(f"Sessions extracted: {gated_ok} (gated as production gates)")
    print(f"Sessions rejected : {sessions - gated_ok}")
    for reason, count in sorted(rejections.items()):
        print(f"    {reason}: {count}")

    print("\n=== Feature distributions (gated) ===")
    header = (
        f"{'feature':<32s} {'n':>6s} {'drop':>5s} {'min':>10s} {'p5':>10s} "
        f"{'median':>10s} {'p95':>10s} {'max':>10s} {'MAD':>9s}"
    )
    print(header)
    print("-" * len(header))

    for name in sorted(set(gated) | set(ungated) | set(UNOBSERVED_FEATURES)):
        if name in UNOBSERVED_FEATURES:
            print(f"{name:<32s} {'UNOBSERVED — Aalto cannot measure this':>62s}")
            continue

        values = sorted(gated.get(name, []))
        dropped = len(ungated.get(name, [])) - len(values)
        if not values:
            print(f"{name:<32s} {0:6d} {dropped:5d}   no value survived the observation gate")
            continue

        median = st.median(values)
        mad = st.median([abs(v - median) for v in values])
        print(
            f"{name:<32s} {len(values):6d} {dropped:5d} {values[0]:10.4f} "
            f"{percentile(values, 0.05):10.4f} {median:10.4f} "
            f"{percentile(values, 0.95):10.4f} {values[-1]:10.4f} {mad:9.4f}"
        )

    print("\n'drop' is how many session values the production observation gate")
    print("discarded. A large number there means short sessions, not bad data.")


if __name__ == "__main__":
    main()
