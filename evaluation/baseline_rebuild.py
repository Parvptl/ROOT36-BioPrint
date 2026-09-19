"""Re-express the first real-human run in the password-separated format.

Run 1 (2026-09-20, ~00:57 to 01:09) was summarised only as aggregate rates, and
it mixed wrong-password attempts into FRR and FAR. This script reads the
preserved snapshot of that run's database and re-reports it through the same
code a new run uses, so the two are comparable.

The phase boundaries are reconstructed, not recorded — run 1 predates the
harness writing per-attempt labels. They are reconstructed from the attempt
timestamps and cross-checked against the aggregate counts the run itself
produced (genuine 14, impostor 10, bot 12, replay 7), all four of which match
exactly. Where something could not be established it is left unexplained rather
than guessed.

    cd backend
    ./.venv/Scripts/python.exe ../evaluation/baseline_rebuild.py
"""

from __future__ import annotations

import json
import sqlite3
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / "backend"
for p in (str(BACKEND), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from attempts import (  # noqa: E402
    ALLOWED,
    AUTOMATION_REASONS,
    BLOCKED,
    INTEGRITY_REASONS,
    PhaseSummary,
    classify,
)

SNAPSHOT = HERE / "out" / "baseline" / "run1_2026-09-20_bioprint.db"
OUT = HERE / "out" / "baseline" / "run1_2026-09-20_per_attempt.json"

# Reconstructed phase windows. Each was confirmed against run 1's own aggregate
# counts before being written down here.
#
#   GENUINE   ids 8-21   14 attempts, 12 ALLOW -> 85.7%, matches run 1
#   IMPOSTOR  ids 22-31  10 attempts, 0 ALLOW  -> 100%,  matches run 1
#   BOT       ids 32-43  12 attempts,          -> 100%,  matches run 1
#   REPLAY    7 explicit ids                   -> 100%,  matches run 1
#
# The replay phase is listed by id rather than as a range because the harness
# deliberately does not count the priming login each round has to submit before
# it has anything to replay. Ids 44, 48, 51, 54 and 56 are those priming
# submissions and id 47 is a password failure inside the same window whose
# cause was not established; none of the six were counted by run 1 and none are
# counted here.
GENUINE = range(8, 22)
IMPOSTOR = range(22, 32)
BOT = range(32, 44)
REPLAY = (45, 46, 49, 50, 52, 53, 55)
UNATTRIBUTED = (44, 47, 48, 51, 54, 56)


def main() -> None:
    if not SNAPSHOT.exists():
        sys.exit(f"snapshot not found at {SNAPSHOT}")

    conn = sqlite3.connect(f"file:{SNAPSHOT}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM auth_attempts")}
    threshold = conn.execute(
        "SELECT threshold FROM behavior_profiles WHERE user_id = 1"
    ).fetchone()[0]
    conn.close()

    phases = {
        "GENUINE": GENUINE,
        "IMPOSTOR": IMPOSTOR,
        "BOT": BOT,
        "REPLAY": REPLAY,
    }

    summaries = {
        label: PhaseSummary(
            label=label,
            records=[classify(rows[i], label, threshold) for i in ids if i in rows],
        )
        for label, ids in phases.items()
    }

    out: dict[str, object] = {
        "run": "run1",
        "collected": "2026-09-20 00:57 to 01:09",
        "account": "parvptl",
        "profile_threshold": threshold,
        "phase_labels": "RECONSTRUCTED from timestamps; cross-checked against "
                        "run 1's own aggregate counts, all four of which match",
        "build_warning":
            "collected against a uvicorn process started 2026-09-19 20:30, which "
            "predates both the Isolation Forest layer (21:45) and the adaptive "
            "profile layer (00:40). Confirmed by the database: zero rows in "
            "profile_updates, no model files on disk, and ml_anomaly_score NULL "
            "on every attempt. These numbers describe the statistical layer "
            "alone.",
        "protocol": {"password_failures_excluded_from_far_frr": True},
    }

    print(f"{'phase':<10s} {'total':>6s} {'judged':>7s} {'pw_fail':>8s} "
          f"{'rate':>8s}  note")
    print("-" * 72)

    for label, s in summaries.items():
        if label == "GENUINE":
            rate = s.rate(lambda r: r.behavioral_decision == ALLOWED)
            name, frr = "acceptance", s.rate(lambda r: r.behavioral_decision == BLOCKED)
            out["genuine"] = {**s.as_dict(), "acceptance_rate": rate,
                              "false_rejection_rate": frr}
        elif label == "IMPOSTOR":
            rate = s.rate(lambda r: r.behavioral_decision == BLOCKED)
            name = "rejection"
            out["impostor"] = {
                **s.as_dict(), "rejection_rate": rate,
                "false_acceptance_rate": s.rate(lambda r: r.behavioral_decision == ALLOWED),
            }
        elif label == "BOT":
            rate = s.rate(lambda r: r.behavioral_decision == BLOCKED)
            name = "block"
            out["bot"] = {
                **s.as_dict(), "block_rate": rate,
                "automation_reason_rate": s.rate(
                    lambda r: r.rejection_reason in AUTOMATION_REASONS),
            }
        else:
            rate = s.rate(lambda r: r.behavioral_decision == BLOCKED)
            name = "rejection"
            out["replay"] = {
                **s.as_dict(), "rejection_rate": rate,
                "integrity_reason_rate": s.rate(
                    lambda r: r.rejection_reason in INTEGRITY_REASONS),
            }

        shown = "n/a" if rate is None else f"{rate:.1%}"
        print(f"{label:<10s} {s.total:6d} {len(s.behavioral):7d} "
              f"{len(s.password_failures):8d} {shown:>8s}  {name}")

    failures = [r for s in summaries.values() for r in s.password_failures]
    out["password_failures"] = {
        "n": len(failures),
        "attempts": [r.as_dict() for r in failures],
    }

    latencies = [r.latency_ms for s in summaries.values() for r in s.records if r.latency_ms]
    ordered = sorted(latencies)
    out["latency"] = {
        "n": len(ordered),
        "p50_ms": st.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }

    out["attempts"] = [r.as_dict() for s in summaries.values() for r in s.records]
    out["unattributed_attempt_ids"] = list(UNATTRIBUTED)

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\npassword failures excluded from FAR/FRR: {len(failures)}")
    for r in failures:
        print(f"  id {r.attempt_id}  phase {r.phase}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
