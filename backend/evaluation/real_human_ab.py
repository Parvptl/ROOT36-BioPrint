#!/usr/bin/env python3
"""Phase E: the synthetic vs real Aalto prior A/B on REAL human sessions.

Every real capture is collected ONCE and replayed through both scoring
conditions. The two conditions never see different sessions, and neither can
mutate the other's profile.

              one real human session
                        |
                 feature vector
                        |
              +---------+---------+
              |                   |
        synthetic prior     real Aalto prior
              |                   |
           score A             score B

Refuses to compute a rate it cannot support. With fewer than the configured
minimum of participants it reports the shortfall and stops rather than
producing a percentage from four people.

    cd backend
    ./.venv/Scripts/python.exe evaluation/real_human_ab.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.fingerprint.aalto_prior import aalto_population_prior  # noqa: E402
from app.behavioral.fingerprint.profile import BehaviorProfile  # noqa: E402
from app.behavioral.fingerprint.scoring import score_identity  # noqa: E402
from app.behavioral.scoring.risk_engine import MIN_IDENTITY_COVERAGE  # noqa: E402
from app.config import settings  # noqa: E402
from app.evaluation_capture import (  # noqa: E402
    LABEL_GENUINE,
    LABEL_IMPOSTOR,
    ROLE_ENROLLMENT,
    ROLE_LOGIN,
)
from evaluation.prior_ablation import (  # noqa: E402
    REAL_PRIOR,
    SYNTHETIC_PRIOR,
    bootstrap_ci,
    build_profile,
    describe,
    eer,
    roc_auc,
)

# Production reaches exactly two enrollment sizes: 1 (COLD_START) and 8
# (MATURE). A 2- or 4-session profile is not a production state and is not
# evaluated here — Phase D showed a sub-3-session profile has no features at
# all and blocks everything as INSUFFICIENT_SIGNAL.
MATURITIES = (1, 8)

# Below this many participants a rate is not reported. Chosen to match the
# protocol's stated minimum, not tuned after seeing the data.
MIN_PARTICIPANTS = 10
MIN_GENUINE_PER_PARTICIPANT = 5


def load_dataset(db: Path) -> dict[str, dict]:
    """Group captured sessions by participant."""
    if not db.exists():
        return {}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM evaluation_sessions ORDER BY enrollment_index, id"
        )]
    finally:
        conn.close()

    people: dict[str, dict] = defaultdict(
        lambda: {"enrollment": [], "genuine": [], "impostor_against_me": []}
    )
    for row in rows:
        features = json.loads(row["features_json"])
        if row["role"] == ROLE_ENROLLMENT:
            people[row["participant"]]["enrollment"].append(features)
        elif row["label"] == LABEL_GENUINE:
            people[row["participant"]]["genuine"].append(features)
        elif row["label"] == LABEL_IMPOSTOR:
            # Filed against the account that was ATTACKED, which is whose
            # profile it must be scored against.
            people[row["target"]]["impostor_against_me"].append(features)
    return dict(people)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=settings.eval_db_path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out", type=Path,
                        default=_BACKEND.parent / "evaluation" / "out" / "phase_e_real_human_ab.json")
    parser.add_argument("--allow-underpowered", action="store_true",
                        help="report rates even below the participant minimum; "
                             "they will be labelled as such")
    args = parser.parse_args()

    people = load_dataset(args.db)

    print("PHASE E — REAL-HUMAN PRIOR A/B")
    print(f"  dataset: {args.db}")
    print(f"  participants with any data: {len(people)}")
    print()

    usable = {
        pid: data for pid, data in people.items()
        if len(data["enrollment"]) >= 8
        and len(data["genuine"]) >= MIN_GENUINE_PER_PARTICIPANT
    }

    header = (f"{'participant':<14s} {'enroll':>7s} {'genuine':>8s} "
              f"{'impostor':>9s} {'usable':>7s}")
    print(header)
    print("-" * len(header))
    for pid in sorted(people):
        d = people[pid]
        print(f"{pid:<14s} {len(d['enrollment']):>7d} {len(d['genuine']):>8d} "
              f"{len(d['impostor_against_me']):>9d} "
              f"{'yes' if pid in usable else 'no':>7s}")
    print()

    total_genuine = sum(len(d["genuine"]) for d in usable.values())
    total_impostor = sum(len(d["impostor_against_me"]) for d in usable.values())

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(args.db),
        "participants_any_data": len(people),
        "participants_usable": len(usable),
        "genuine_attempts": total_genuine,
        "impostor_attempts": total_impostor,
        "minimum_participants": MIN_PARTICIPANTS,
        "status": None,
    }

    if len(usable) < MIN_PARTICIPANTS and not args.allow_underpowered:
        summary["status"] = "INSUFFICIENT_DATA"
        summary["shortfall"] = MIN_PARTICIPANTS - len(usable)
        print(f"INSUFFICIENT DATA: {len(usable)} usable participants, "
              f"{MIN_PARTICIPANTS} required.")
        print()
        print("  Needed per participant: 8 enrollment rounds and at least")
        print(f"  {MIN_GENUINE_PER_PARTICIPANT} genuine logins, plus impostor attempts "
              "from other participants.")
        print()
        print("  No rate is reported. A false-rejection rate computed from a")
        print("  handful of people would carry an interval so wide it could not")
        print("  inform a production decision, and reporting it anyway is how a")
        print("  prior gets activated on evidence that does not exist.")
        print()
        print("  Real-human collection is pending manual participant sessions.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  written to {args.out}")
        return

    summary["status"] = (
        "UNDERPOWERED" if len(usable) < MIN_PARTICIPANTS else "OK"
    )
    summary["maturities"] = {}

    # ---- the A/B ----------------------------------------------------------
    print("=" * 96)
    print("  FIXED-THRESHOLD COMPARISON — production thresholds, no tuning")
    print("=" * 96)
    head = (f"{'enroll':>7s} {'prior':<12s} {'thresh':>8s} {'gen acc':>8s} "
            f"{'FRR':>8s} {'imp acc':>8s} {'FAR':>8s} {'gen n':>6s} {'imp n':>6s}")
    print(head)
    print("-" * len(head))

    for n_enroll in MATURITIES:
        entry: dict[str, object] = {}
        for label, prior_path in (("A_synthetic", SYNTHETIC_PRIOR),
                                  ("B_real", REAL_PRIOR)):
            prior = aalto_population_prior(prior_path)
            gen_scores: list[float] = []
            imp_scores: list[float] = []
            gen_per_user: list[tuple[int, int]] = []
            imp_per_user: list[tuple[int, int]] = []
            thresholds: list[float] = []
            gen_ok = gen_n = imp_bad = imp_n = 0

            for pid, data in sorted(usable.items()):
                # Fresh profile per condition, from the SAME enrollment slice.
                profile = build_profile(data["enrollment"][:n_enroll], prior, [])
                thresholds.append(profile.threshold)

                u_frr = [0, 0]
                for features in data["genuine"]:
                    r = score_identity(profile, features)
                    gated = not r.is_comparable or r.coverage < MIN_IDENTITY_COVERAGE
                    accepted = (not gated) and r.score <= profile.threshold
                    gen_n += 1
                    gen_ok += int(accepted)
                    u_frr[0] += int(not accepted)
                    u_frr[1] += 1
                    if not gated:
                        gen_scores.append(r.score)
                gen_per_user.append(tuple(u_frr))

                u_far = [0, 0]
                for features in data["impostor_against_me"]:
                    r = score_identity(profile, features)
                    gated = not r.is_comparable or r.coverage < MIN_IDENTITY_COVERAGE
                    accepted = (not gated) and r.score <= profile.threshold
                    imp_n += 1
                    imp_bad += int(accepted)
                    u_far[0] += int(accepted)
                    u_far[1] += 1
                    if not gated:
                        imp_scores.append(r.score)
                if u_far[1]:
                    imp_per_user.append(tuple(u_far))

            med_thr = float(np.median(thresholds)) if thresholds else float("nan")
            frr = 1 - gen_ok / gen_n if gen_n else None
            far = imp_bad / imp_n if imp_n else None
            auc = roc_auc(gen_scores, imp_scores)
            e = eer(gen_scores, imp_scores)

            print(f"{n_enroll:>7d} {label:<12s} {med_thr:8.4f} "
                  f"{(gen_ok / gen_n if gen_n else float('nan')):8.4f} "
                  f"{(frr if frr is not None else float('nan')):8.4f} "
                  f"{(imp_bad / imp_n if imp_n else float('nan')):8.4f} "
                  f"{(far if far is not None else float('nan')):8.4f} "
                  f"{gen_n:6d} {imp_n:6d}")

            entry[label] = {
                "median_threshold": round(med_thr, 5),
                "genuine_n": gen_n, "impostor_n": imp_n,
                "genuine_accept": round(gen_ok / gen_n, 5) if gen_n else None,
                "frr": round(frr, 5) if frr is not None else None,
                "far": round(far, 5) if far is not None else None,
                "roc_auc": round(auc, 5) if auc is not None else None,
                "eer": round(e[0], 5) if e else None,
                "genuine_scores": describe(gen_scores),
                "impostor_scores": describe(imp_scores),
                "frr_ci95": bootstrap_ci(gen_per_user, args.bootstrap, args.seed),
                "far_ci95": bootstrap_ci(imp_per_user, args.bootstrap, args.seed),
            }
        summary["maturities"][str(n_enroll)] = entry  # type: ignore[index]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
