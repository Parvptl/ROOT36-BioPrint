"""Create a synthetic enrolled account directly in the database.

EVALUATION TOOLING ONLY. This exists so the scripted phases of
run_evaluation.py can be exercised without a human sitting through eight
enrollment rounds first.

Two things it is deliberately not:

  * It is NOT reachable over HTTP. There is no endpoint that does this. It
    requires filesystem access to the database, which is already game over.
  * It does NOT bypass scoring. The profile is fitted by the same
    build_calibrated_profile the real enrollment path uses, from synthetic
    captures produced by the same generator the tests use. A login against
    this profile is scored normally and can fail normally.

Any accuracy number produced against a seeded account describes synthetic
typists, not people, and must be labelled that way.

    cd backend
    ./.venv/Scripts/python.exe ../evaluation/seed_synthetic_account.py \
        --username bench --password bench-password-1234
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.auth.challenge import generate_phrase  # noqa: E402
from app.auth.passwords import hash_password  # noqa: E402
from app.behavioral.features.extractor import extract_from_session  # noqa: E402
from app.behavioral.fingerprint.calibration import build_calibrated_profile  # noqa: E402
from app.behavioral.fingerprint.population import PopulationPrior  # noqa: E402
from app.db import repository  # noqa: E402
from app.db.database import get_connection, init_db, set_db_path_override  # noqa: E402
from tests.factories import TypingStyle, human_session  # noqa: E402

SUBJECT = TypingStyle(
    iki_mean_ms=140.0, dwell_mean_ms=82.0, overlap_prob=0.45,
    right_shift_prob=0.95, tab_between_fields=True, pointer_speed=1.6,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--db", default="data/bioprint.db")
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()

    db = Path(args.db).resolve()
    set_db_path_override(db)
    init_db()

    captures = []
    for i in range(args.rounds):
        _, features = extract_from_session(
            human_session(generate_phrase(), style=SUBJECT, seed=70_000 + i)
        )
        captures.append(features.as_dict())

    profile = build_calibrated_profile(captures, PopulationPrior(), [])

    with get_connection() as conn:
        existing = repository.get_user(conn, args.username)
        if existing is not None:
            repository.delete_profile(conn, existing["id"])
            conn.execute("DELETE FROM users WHERE id = ?", (existing["id"],))
        user_id = repository.create_user(
            conn, args.username, hash_password(args.password)
        )
        repository.save_profile(conn, user_id, profile)
        repository.mark_enrolled(conn, user_id)

    print(f"seeded synthetic account '{args.username}' in {db}")
    print(f"  features modelled : {len(profile.features)}")
    print(f"  threshold         : {profile.threshold:.4f} ({profile.threshold_source})")
    print(f"  enrollment rounds : {profile.session_count}")
    print("\n  SYNTHETIC. Not a real person. Do not report rates from this as accuracy.")


if __name__ == "__main__":
    main()
