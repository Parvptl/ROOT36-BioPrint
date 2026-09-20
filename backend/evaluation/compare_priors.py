#!/usr/bin/env python3
"""Compare two population priors feature by feature.

Built for one question: how realistic was the synthetic bootstrap prior?

Not "which is better". A synthetic prior that differs from the real one is not
thereby wrong, and one that matches is not thereby validated — the synthetic
generator was written by someone who had read about typing, so agreement could
mean the assumptions were good or merely that both encode the same folklore.
What the comparison can establish is *where* the bootstrap was standing in
roughly the right place and where it was not, which is what decides how much a
result measured against the bootstrap should be trusted.

    cd backend
    ./.venv/Scripts/python.exe evaluation/compare_priors.py \
        --real data/aalto_prior_real.json \
        --synthetic data/aalto_prior_synthetic_bootstrap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# A scale ratio outside this band is called out as a substantive difference
# rather than noise. Chosen so a factor of two in either direction is flagged.
SIMILAR_SCALE_RATIO = (0.5, 2.0)

# And a median that moved by more than this share of the synthetic scale is a
# shift worth naming: it means the bootstrap centred the population somewhere
# the real data does not.
SIMILAR_MEDIAN_SHIFT_IN_SCALES = 1.0


def load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=_BACKEND / "data" / "aalto_prior_real.json")
    parser.add_argument(
        "--synthetic", type=Path,
        default=_BACKEND / "data" / "aalto_prior_synthetic_bootstrap.json",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    for path in (args.real, args.synthetic):
        if not path.exists():
            sys.exit(f"missing prior: {path}")

    real, synthetic = load(args.real), load(args.synthetic)
    real_f, syn_f = real["features"], synthetic["features"]

    print(f"REAL      {args.real.name}: {real['meta'].get('n_users')} participants, "
          f"source={real['meta'].get('source')}")
    print(f"SYNTHETIC {args.synthetic.name}: {synthetic['meta'].get('n_users')} users, "
          f"source={synthetic['meta'].get('source')}")
    print()

    header = (
        f"{'feature':<32s} {'syn median':>11s} {'real median':>11s} {'abs diff':>10s} "
        f"{'rel %':>8s} {'syn scale':>10s} {'real scale':>10s} {'ratio':>7s} {'verdict':>10s}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict[str, object]] = []
    similar, different, unobserved = [], [], []

    for name in sorted(set(real_f) | set(syn_f)):
        r, s = real_f.get(name), syn_f.get(name)

        if r is not None and str(r.get("status", "observed")) == "unobserved":
            # No numeric comparison is possible or honest here.
            unobserved.append(name)
            syn_note = "has a number" if s is not None else "absent"
            print(f"{name:<32s} {'REAL = UNOBSERVED':>60s}   (synthetic: {syn_note})")
            rows.append({
                "feature": name, "status": "real_unobserved",
                "synthetic_present": s is not None,
                "synthetic_median": None if s is None else s.get("population_median"),
                "note": "Aalto cannot measure this; no comparison attempted.",
            })
            continue

        if r is None or s is None:
            side = "real only" if s is None else "synthetic only"
            print(f"{name:<32s} {side:>60s}")
            rows.append({"feature": name, "status": f"present_in_{side.split()[0]}_only"})
            continue

        sm, rm = float(s["population_median"]), float(r["population_median"])
        ss, rs = float(s["population_scale"]), float(r["population_scale"])

        abs_diff = rm - sm
        rel = (abs_diff / abs(sm) * 100.0) if sm != 0 else float("nan")
        ratio = (rs / ss) if ss > 0 else float("inf")
        shift_in_scales = abs(abs_diff) / ss if ss > 0 else float("inf")

        is_similar = (
            SIMILAR_SCALE_RATIO[0] <= ratio <= SIMILAR_SCALE_RATIO[1]
            and shift_in_scales <= SIMILAR_MEDIAN_SHIFT_IN_SCALES
        )
        (similar if is_similar else different).append(name)

        print(
            f"{name:<32s} {sm:11.4f} {rm:11.4f} {abs_diff:10.4f} "
            f"{rel:7.1f}% {ss:10.4f} {rs:10.4f} {ratio:7.2f} "
            f"{'similar' if is_similar else 'DIFFERENT':>10s}"
        )

        rows.append({
            "feature": name,
            "status": "compared",
            "synthetic_median": sm, "real_median": rm,
            "absolute_difference": round(abs_diff, 6),
            "relative_difference_pct": None if sm == 0 else round(rel, 3),
            "synthetic_scale": ss, "real_scale": rs,
            "scale_ratio_real_over_synthetic": round(ratio, 4),
            "median_shift_in_synthetic_scales": round(shift_in_scales, 4),
            "synthetic_p5": s.get("p5"), "real_p5": r.get("p5"),
            "synthetic_p95": s.get("p95"), "real_p95": r.get("p95"),
            "synthetic_p10": s.get("p10"), "real_p10": r.get("p10"),
            "synthetic_p90": s.get("p90"), "real_p90": r.get("p90"),
            "verdict": "similar" if is_similar else "different",
        })

    print()
    print(f"similar   ({len(similar)}): {', '.join(similar) or '-'}")
    print(f"different ({len(different)}): {', '.join(different) or '-'}")
    print(f"unobserved in real ({len(unobserved)}): {', '.join(unobserved) or '-'}")
    print()
    print("A difference is not a defect in either prior. It says the bootstrap's")
    print("assumption about that feature did not match what people actually do,")
    print("which is exactly what generating a real prior was for.")

    missing_p5 = [r["feature"] for r in rows
                  if r.get("status") == "compared" and r.get("synthetic_p5") is None]
    if missing_p5:
        print()
        print(f"NOTE: the synthetic prior predates p5/p95, so those columns are "
              f"absent for {len(missing_p5)} features. p10/p90 are compared instead.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({
                "real": str(args.real), "synthetic": str(args.synthetic),
                "real_participants": real["meta"].get("n_users"),
                "synthetic_users": synthetic["meta"].get("n_users"),
                "similar": similar, "different": different,
                "real_unobserved": unobserved,
                "features": rows,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"\nwritten to {args.json_out}")


if __name__ == "__main__":
    main()
