#!/usr/bin/env python3
"""Phase E real-human collection server.

The collector needs one thing the server cannot know: who is actually at the
keyboard. The operator declares it between phases with `actor`, and the
capture seam compares that declaration to the account being claimed to derive
the genuine / human_impostor label. Labelling by outcome instead would define
an impostor as "an attempt the system rejected", which is the thing under test.

    cd backend

    # who is this account, in the dataset?
    ./.venv/Scripts/python.exe evaluation/eval_collect.py pseudonym alice

    # genuine phase: alice is typing, at her own account
    ./.venv/Scripts/python.exe evaluation/eval_collect.py actor alice

    # impostor phase: bob is typing, at whichever account he claims
    ./.venv/Scripts/python.exe evaluation/eval_collect.py actor bob

    # between participants, clear it so stray traffic is not mislabelled
    ./.venv/Scripts/python.exe evaluation/eval_collect.py actor --clear

    ./.venv/Scripts/python.exe evaluation/eval_collect.py status
    ./.venv/Scripts/python.exe evaluation/eval_collect.py validate
    ./.venv/Scripts/python.exe evaluation/eval_collect.py export
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.config import settings  # noqa: E402
from app.evaluation_capture import (  # noqa: E402
    LABEL_GENUINE,
    LABEL_IMPOSTOR,
    ROLE_ENROLLMENT,
    ROLE_LOGIN,
    current_actor,
    pseudonym,
    set_actor,
    validate_vector,
)

EXPECTED_DIMENSION = 27


def _open(path: Path) -> sqlite3.Connection:
    if not path.exists():
        sys.exit(
            f"no evaluation database at {path}\n"
            "Nothing has been collected yet. Start the backend with "
            "BIOPRINT_EVALUATION_MODE=1 and declare an actor first."
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def cmd_pseudonym(args) -> None:
    print(f"{args.username}  ->  {pseudonym(args.username)}")
    print()
    print("Keep the name-to-pseudonym mapping OUTSIDE the dataset. The dataset")
    print("stores only the right-hand side, and it cannot be reversed without")
    print("the server secret.")


def cmd_actor(args) -> None:
    if args.clear:
        path = set_actor(None)
        print(f"actor cleared ({path})")
        print("Captures will now be refused with reason 'no_actor_declared'.")
        return
    if not args.username:
        actor = current_actor()
        print(f"current actor: {actor or '(none declared)'}")
        return
    path = set_actor(args.username)
    print(f"actor set to {pseudonym(args.username)}  ({path})")
    print(f"Captures claiming this account will be labelled '{LABEL_GENUINE}';")
    print(f"captures claiming any other account will be labelled '{LABEL_IMPOSTOR}'.")


def cmd_status(args) -> None:
    print(f"evaluation_mode : {settings.evaluation_mode}")
    print(f"eval database   : {settings.eval_db_path}")
    print(f"current actor   : {current_actor() or '(none declared)'}")
    print()

    if not settings.eval_db_path.exists():
        print("no captures yet")
        return

    conn = _open(settings.eval_db_path)
    rows = [dict(r) for r in conn.execute("SELECT * FROM evaluation_sessions")]
    rejects = [dict(r) for r in conn.execute("SELECT * FROM evaluation_rejects")]
    conn.close()

    by_participant: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        key = f"{row['role']}/{row['label']}"
        by_participant[row["participant"]][key] += 1

    print(f"participants seen : {len(by_participant)}")
    print(f"sessions captured : {len(rows)}")
    print(f"rejected captures : {len(rejects)}")
    print()

    header = (f"{'participant':<14s} {'enroll':>7s} {'genuine':>8s} "
              f"{'impostor':>9s} {'ready':>7s}")
    print(header)
    print("-" * len(header))
    for participant in sorted(by_participant):
        counts = by_participant[participant]
        enroll = counts[f"{ROLE_ENROLLMENT}/{LABEL_GENUINE}"]
        genuine = counts[f"{ROLE_LOGIN}/{LABEL_GENUINE}"]
        impostor = counts[f"{ROLE_LOGIN}/{LABEL_IMPOSTOR}"]
        ready = "yes" if enroll >= 8 and genuine >= 5 else "no"
        print(f"{participant:<14s} {enroll:>7d} {genuine:>8d} {impostor:>9d} {ready:>7s}")

    if rejects:
        print()
        print("rejections by reason:")
        for reason, count in Counter(r["reason"] for r in rejects).most_common():
            print(f"  {reason:<32s} {count}")


def cmd_validate(args) -> None:
    """Every check Task 17 asks for, with a recorded reason per rejection."""
    conn = _open(settings.eval_db_path)
    rows = [dict(r) for r in conn.execute("SELECT * FROM evaluation_sessions ORDER BY id")]
    conn.close()

    problems: list[str] = []
    seen_sessions: set[str] = set()
    dimensions: Counter = Counter()
    enrollment_order: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        sid = row["session_id"]
        if sid in seen_sessions:
            problems.append(f"{sid}: duplicate session id")
        seen_sessions.add(sid)

        features = json.loads(row["features_json"])
        dimensions[len(features)] += 1

        problem = validate_vector(features)
        if problem is not None:
            problems.append(f"{sid}: {problem}")

        if row["label"] not in (LABEL_GENUINE, LABEL_IMPOSTOR):
            problems.append(f"{sid}: unknown label {row['label']!r}")
        if row["role"] == ROLE_ENROLLMENT:
            if row["label"] != LABEL_GENUINE:
                problems.append(f"{sid}: enrollment labelled {row['label']!r}")
            if row["enrollment_index"] is None:
                problems.append(f"{sid}: enrollment without an ordinal")
            else:
                enrollment_order[row["participant"]].append(row["enrollment_index"])
        if row["label"] == LABEL_IMPOSTOR and row["participant"] == row["target"]:
            problems.append(f"{sid}: impostor label but actor equals target")
        if row["label"] == LABEL_GENUINE and row["participant"] != row["target"]:
            problems.append(f"{sid}: genuine label but actor differs from target")

    for participant, order in enrollment_order.items():
        if sorted(order) != list(range(len(order))):
            problems.append(
                f"{participant}: enrollment ordinals {sorted(order)} are not contiguous"
            )

    print(f"rows checked       : {len(rows)}")
    print(f"unique session ids : {len(seen_sessions)}")
    print(f"vector dimensions  : {dict(dimensions)} (expected {EXPECTED_DIMENSION})")
    off_dimension = sum(n for d, n in dimensions.items() if d != EXPECTED_DIMENSION)
    if off_dimension:
        print(f"  NOTE: {off_dimension} vectors are not {EXPECTED_DIMENSION}-dimensional. "
              "A feature absent from a capture is legitimately omitted rather "
              "than imputed, so this is expected for pointer-free sessions.")
    print()
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("PROBLEMS: none.")


def cmd_export(args) -> None:
    conn = _open(settings.eval_db_path)
    rows = [dict(r) for r in conn.execute("SELECT * FROM evaluation_sessions ORDER BY id")]
    conn.close()

    payload = {
        "dataset": "bioprint_phase_e_real_human",
        "contains": [
            "pseudonymous participant id", "pseudonymous target id", "session id",
            "role", "label", "enrollment ordinal", "derived feature vector",
            "coverage", "timestamp",
        ],
        "does_not_contain": [
            "passwords", "password hashes", "raw keystroke events",
            "raw pointer events", "challenge text", "usernames", "emails",
            "names", "IP addresses",
        ],
        "sessions": [
            {**{k: v for k, v in row.items() if k != "features_json"},
             "features": json.loads(row["features_json"])}
            for row in rows
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"exported {len(rows)} sessions to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pseudonym", help="show an account's dataset id")
    p.add_argument("username")
    p.set_defaults(func=cmd_pseudonym)

    p = sub.add_parser("actor", help="declare who is at the keyboard")
    p.add_argument("username", nargs="?")
    p.add_argument("--clear", action="store_true")
    p.set_defaults(func=cmd_actor)

    p = sub.add_parser("status", help="collection progress")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("validate", help="data-quality checks")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("export", help="write the dataset as JSON")
    p.add_argument("--out", type=Path,
                   default=_BACKEND.parent / "evaluation" / "out" / "phase_e_dataset.json")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
