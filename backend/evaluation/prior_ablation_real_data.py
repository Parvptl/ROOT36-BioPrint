#!/usr/bin/env python3
"""Phase D supplement: the same A/B on the only real human feature data we have.

What exists, and why there is so little of it
---------------------------------------------
BioPrint discards raw events at extraction and deletes enrollment feature
vectors once a profile is fitted. `auth_attempts` stores scores and reason
codes, never a feature vector. So the ~14 genuine and ~10 impostor real login
attempts from the earlier human evaluation **cannot be re-scored under a second
prior** — the inputs no longer exist. That is the privacy design working
exactly as documented, and it makes a real-data A/B on login attempts
structurally impossible without collecting new sessions.

What does survive is `population_samples`: consented enrollment feature vectors
retained to serve as the population prior. That is the entire real-human
feature corpus in the system.

This script runs the identical two-condition comparison over those vectors,
using leave-one-session-out for genuine scores and cross-user submission for
impostor scores. The sample is far too small for a rate to mean much; it is
reported with its n and never merged with the synthetic experiment.

    cd backend
    ./.venv/Scripts/python.exe evaluation/prior_ablation_real_data.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.fingerprint.aalto_prior import aalto_population_prior  # noqa: E402
from app.behavioral.fingerprint.profile import fit_profile  # noqa: E402
from app.behavioral.fingerprint.scoring import score_identity  # noqa: E402
from app.behavioral.fingerprint.profile import BehaviorProfile  # noqa: E402
from evaluation.prior_ablation import (  # noqa: E402
    REAL_PRIOR,
    SYNTHETIC_PRIOR,
    describe,
    eer,
    roc_auc,
)


def load_real_vectors(db: Path) -> dict[str, list[dict[str, float]]]:
    """Consented enrollment vectors, grouped by their keyed contributor tag.

    The tag is a keyed hash of the user id, not the id itself — it cannot be
    linked back to a person without the server secret. It is used here only to
    keep one contributor's sessions together.
    """
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT contributor, features_json FROM population_samples "
            "WHERE contributor IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["contributor"]].append(json.loads(row["features_json"]))
    return dict(grouped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=_BACKEND / "data" / "bioprint.db")
    parser.add_argument("--out", type=Path,
                        default=_BACKEND.parent / "evaluation" / "out"
                        / "phase_d_real_data_supplement.json")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"database not found: {args.db}")

    grouped = load_real_vectors(args.db)
    print("REAL HUMAN FEATURE DATA AVAILABLE")
    print(f"  contributors: {len(grouped)}")
    for tag, vectors in grouped.items():
        print(f"    {tag[:8]}…  {len(vectors)} enrollment vectors, "
              f"{len(vectors[0])} features each")
    total = sum(len(v) for v in grouped.values())
    print(f"  total vectors: {total}")
    print()
    print("  NOTE: login-attempt feature vectors do not exist — raw events are")
    print("  discarded at extraction and auth_attempts stores scores only. These")
    print("  enrollment vectors are the entire real-human feature corpus.")
    print()

    if len(grouped) < 2:
        raise SystemExit(
            "need at least 2 contributors to produce any impostor score; "
            f"found {len(grouped)}. No real-data comparison is possible."
        )

    results: dict[str, object] = {
        "kind": "REAL human enrollment vectors",
        "contributors": len(grouped),
        "vectors_total": total,
        "limitation": (
            "Login-attempt feature vectors are not retained by design, so this "
            "uses enrollment vectors via leave-one-session-out. Sample is far "
            "too small for a meaningful rate."
        ),
        "conditions": {},
    }

    tags = sorted(grouped)
    header = (f"{'prior':<12s} {'gen n':>6s} {'imp n':>6s} {'gen p50':>9s} "
              f"{'imp p50':>9s} {'gap':>9s} {'ROC-AUC':>9s} {'EER':>8s}")
    print(header)
    print("-" * len(header))

    for label, path in (("A_synthetic", SYNTHETIC_PRIOR), ("B_real", REAL_PRIOR)):
        prior = aalto_population_prior(path)
        if prior is None:
            raise SystemExit(f"could not load {path}")

        genuine: list[float] = []
        impostor: list[float] = []

        for tag in tags:
            sessions = grouped[tag]
            # Leave-one-session-out genuine scores.
            for i in range(len(sessions)):
                held_out = sessions[i]
                remaining = sessions[:i] + sessions[i + 1:]
                features = fit_profile(remaining, prior)
                if not features:
                    continue
                fold = BehaviorProfile(features=features, session_count=len(remaining))
                result = score_identity(fold, held_out)
                if result.is_comparable:
                    genuine.append(result.score)

            # Impostor: every other contributor's sessions against a profile
            # fitted from all of this contributor's sessions.
            full = fit_profile(sessions, prior)
            if not full:
                continue
            profile = BehaviorProfile(features=full, session_count=len(sessions))
            for other in tags:
                if other == tag:
                    continue
                for vector in grouped[other]:
                    result = score_identity(profile, vector)
                    if result.is_comparable:
                        impostor.append(result.score)

        auc = roc_auc(genuine, impostor)
        e = eer(genuine, impostor)
        gen_d, imp_d = describe(genuine), describe(impostor)
        gap = (imp_d["p50"] or 0) - (gen_d["p50"] or 0)

        print(f"{label:<12s} {len(genuine):6d} {len(impostor):6d} "
              f"{gen_d['p50']:9.4f} {imp_d['p50']:9.4f} {gap:9.4f} "
              f"{(auc if auc is not None else float('nan')):9.5f} "
              f"{(e[0] if e else float('nan')):8.5f}")

        results["conditions"][label] = {  # type: ignore[index]
            "prior": str(path),
            "genuine_n": len(genuine),
            "impostor_n": len(impostor),
            "genuine_scores": gen_d,
            "impostor_scores": imp_d,
            "median_gap": round(gap, 5),
            "roc_auc": None if auc is None else round(auc, 5),
            "eer": None if e is None else round(e[0], 5),
            "eer_threshold": None if e is None else round(e[1], 5),
        }

    print()
    print(f"With {results['contributors']} contributors this cannot support a rate.")
    print("Reported for direction only, never merged with the synthetic experiment.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
