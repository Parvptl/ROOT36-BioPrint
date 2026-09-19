"""Per-attempt forensic dump of the audit trail.

Read-only. Prints every recorded attempt with its decision, scores, threshold,
contributing reason codes, adaptation outcome and latency.

Reports only what is stored. A value the system did not record prints as '-'
and is never inferred. In particular:

  * per-modality scores are NOT stored — only the reason code that names the
    modality which drove a block, and the full list of contributing signals;
  * `confidence` is the tier recorded at decision time. Where no row exists in
    profile_updates, `would_be` recomputes the tier from the stored
    identity/automation/coverage/threshold, which is exactly the input
    classify_confidence takes. That is a derivation from recorded values, not
    a reconstruction of anything missing.

    cd backend
    ./.venv/Scripts/python.exe ../evaluation/forensics.py
    ./.venv/Scripts/python.exe ../evaluation/forensics.py --range 8:21 --detail
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.behavioral.fingerprint.adaptation import classify_confidence  # noqa: E402
from app.behavioral.scoring.risk_engine import AUTOMATION_TIGHTENING  # noqa: E402


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def fmt(value, spec: str) -> str:
    width = spec.split(".")[0]
    if value is None:
        return f"{'-':>{width}s}"
    return f"{value:{spec}}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/bioprint.db")
    parser.add_argument("--range", help="attempt id range, e.g. 8:21")
    parser.add_argument("--detail", action="store_true",
                        help="print the full contributing-signal list per attempt")
    args = parser.parse_args()

    conn = open_db(Path(args.db).resolve())

    thresholds: dict[int, float] = {}
    print("PROFILES")
    for row in conn.execute("SELECT * FROM behavior_profiles"):
        thresholds[row["user_id"]] = row["threshold"]
        cal = json.loads(row["calibration_json"])
        print(f"  user_id={row['user_id']}  threshold={row['threshold']:.4f} "
              f"({row['threshold_source']})  profile v{row['version']} after "
              f"{row['update_count']} adaptive updates  "
              f"fitted from {row['session_count']} enrollment sessions")
        print(f"    calibration: {cal.get('metrics', {})}")
        loo = cal.get("genuine_scores") or cal.get("loo_scores")
        if loo:
            print(f"    leave-one-out genuine scores: {[round(s, 4) for s in loo]}")

    updates = {
        r["attempt_id"]: r
        for r in conn.execute("SELECT * FROM profile_updates")
        if r["attempt_id"] is not None
    }
    total_updates = conn.execute("SELECT COUNT(*) FROM profile_updates").fetchone()[0]
    print(f"\n  profile_updates rows recorded: {total_updates}")

    where, params = "", []
    if args.range:
        lo, hi = (int(x) for x in args.range.split(":"))
        where, params = "WHERE id BETWEEN ? AND ?", [lo, hi]

    header = (
        f"\n{'id':>4s} {'time':>8s} {'usr':>3s} {'decision':>8s} {'reason':<22s} "
        f"{'ident':>7s} {'thresh':>7s} {'margin':>7s} {'autom':>7s} {'integ':>6s} "
        f"{'cover':>6s} {'confidence':>10s} {'upd':>4s} {'ms':>6s}"
    )
    print(header)
    print("-" * (len(header) - 1))

    for r in conn.execute(
        f"SELECT * FROM auth_attempts {where} ORDER BY id", params
    ):
        u = updates.get(r["id"])
        base = thresholds.get(r["user_id"])
        ident = r["identity_score"]
        # The bar an attempt was actually judged against is the profile
        # threshold tightened by that attempt's own automation score
        # (risk_engine.AUTOMATION_TIGHTENING). Comparing an identity score to
        # the untightened profile threshold would overstate the margin.
        thr = base
        if base is not None and r["automation_score"] is not None:
            thr = base * (1.0 - AUTOMATION_TIGHTENING * min(1.0, max(0.0, r["automation_score"])))
        margin = (thr - ident) if (thr is not None and ident is not None) else None

        if u is not None:
            conf, upd = u["confidence"], ("yes" if u["applied"] else "no")
        else:
            # No adaptation row. Derive the tier the shipped policy would
            # assign from the values that ARE recorded, and mark it as derived.
            conf = classify_confidence(
                decision=r["decision"], reason=r["reason"],
                identity_score=ident, automation_score=r["automation_score"],
                coverage=r["coverage"], threshold=thr,
            ).value + "*"
            upd = "none"

        print(
            f"{r['id']:4d} {time.strftime('%H:%M:%S', time.localtime(r['created_at'])):>8s} "
            f"{str(r['user_id']):>3s} {r['decision']:>8s} {r['reason']:<22s} "
            f"{fmt(ident, '7.4f')} {fmt(thr, '7.4f')} {fmt(margin, '7.4f')} "
            f"{fmt(r['automation_score'], '7.4f')} {fmt(r['integrity_score'], '6.3f')} "
            f"{fmt(r['coverage'], '6.3f')} {conf:>10s} {upd:>4s} "
            f"{fmt(r['latency_ms'], '6.1f')}"
        )
        if args.detail:
            signals = json.loads(r["reasons_json"])
            print(f"       signals: {', '.join(signals) if signals else '(none)'}")
            print(f"       statistical={fmt(r['statistical_identity_score'], '.4f').strip()} "
                  f"ml_anomaly={fmt(r['ml_anomaly_score'], '.4f').strip()}")

    print("\n  * derived from recorded scores; no profile_updates row existed.")
    print("\nREASON CODE TOTALS")
    for r in conn.execute(
        f"SELECT reason, decision, COUNT(*) AS n FROM auth_attempts {where} "
        "GROUP BY reason, decision ORDER BY n DESC", params
    ):
        print(f"  {r['reason']:<24s} {r['decision']:>6s}  n={r['n']}")

    conn.close()


if __name__ == "__main__":
    main()
