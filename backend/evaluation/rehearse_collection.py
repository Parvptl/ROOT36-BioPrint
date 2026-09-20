#!/usr/bin/env python3
"""Rehearse the Phase E operator workflow end to end, in-process.

This proves the collection pipeline works before any human is asked to sit
down and type: evaluation mode captures, the actor declaration labels
correctly, validation passes, and the A/B harness consumes the result.

The typists here are GENERATED. This produces no real-human evidence and its
output must never be reported as such — the dataset it writes is deleted at the
end for exactly that reason. It is a test of the plumbing, not of the system.

    cd backend
    ./.venv/Scripts/python.exe evaluation/rehearse_collection.py
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import evaluation_capture  # noqa: E402
from app.api.ratelimit import limiter  # noqa: E402
from app.db.database import init_db, set_db_path_override  # noqa: E402
from tests.factories import TypingStyle, human_session  # noqa: E402

PARTICIPANTS = 4
ENROLL_ROUNDS = 8
GENUINE_LOGINS = 5
IMPOSTOR_LOGINS = 3


def age(db: Path, nonce: str, seconds: float) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE auth_challenges SET issued_at = issued_at - ?, "
            "expires_at = expires_at - ? WHERE nonce = ?",
            (seconds, seconds, nonce),
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="bioprint_rehearsal_"))
    prod_db = workdir / "prod.db"
    eval_db = workdir / "eval" / "eval.db"

    set_db_path_override(prod_db)
    init_db(prod_db)
    limiter.reset()

    evaluation_capture.settings = replace(
        evaluation_capture.settings,
        evaluation_mode=True,
        eval_db_path=eval_db,
        secret_is_ephemeral=False,
    )

    from app.main import app

    client = TestClient(app)
    rng = np.random.default_rng(31337)

    print("REHEARSAL — GENERATED TYPISTS, NOT REAL HUMANS")
    print(f"  workdir: {workdir}")
    print()

    accounts = []
    for i in range(PARTICIPANTS):
        username, password = f"rehearse{i}", f"rehearse-password-{i:04d}"
        style = TypingStyle(
            iki_mean_ms=float(rng.uniform(110, 300)),
            dwell_mean_ms=float(rng.uniform(60, 140)),
            overlap_prob=float(rng.uniform(0.0, 0.6)),
            right_shift_prob=float(rng.uniform(0.05, 0.95)),
        )
        client.post("/auth/register",
                    json={"username": username, "password": password, "consent": True})
        accounts.append((username, password, style))

    # --- enrollment + genuine logins -----------------------------------
    for idx, (username, password, style) in enumerate(accounts):
        evaluation_capture.set_actor(username)          # operator declaration
        for r in range(ENROLL_ROUNDS):
            started = client.post("/auth/enrollment/start",
                                  json={"username": username, "password": password}).json()
            if "phrase" not in started:
                print(f"  enrollment refused for {username}: {started}")
                break
            session = human_session(started["phrase"], nonce=started["nonce"],
                                    username=username, password=password,
                                    style=style, seed=idx * 100 + r)
            age(prod_db, started["nonce"], max(e.t for e in session.events) / 1000.0 + 1.0)
            client.post("/auth/enrollment/submit",
                        json={"username": username, "session": session.model_dump()})
            limiter.reset()

        for g in range(GENUINE_LOGINS):
            issued = client.post("/auth/login/challenge",
                                 json={"username": username}).json()
            session = human_session(issued["phrase"], nonce=issued["nonce"],
                                    username=username, password=password,
                                    style=style, seed=idx * 100 + 50 + g)
            age(prod_db, issued["nonce"], max(e.t for e in session.events) / 1000.0 + 1.0)
            client.post("/auth/login/behavior",
                        json={"username": username, "password": password,
                              "session": session.model_dump()})
            limiter.reset()
        print(f"  {username}: enrolled + {GENUINE_LOGINS} genuine logins")

    # --- impostor logins ------------------------------------------------
    for idx, (target_user, target_pw, _s) in enumerate(accounts):
        attacker_idx = (idx + 1) % len(accounts)
        attacker_user, _pw, attacker_style = accounts[attacker_idx]
        evaluation_capture.set_actor(attacker_user)     # operator declaration
        for a in range(IMPOSTOR_LOGINS):
            issued = client.post("/auth/login/challenge",
                                 json={"username": target_user}).json()
            session = human_session(issued["phrase"], nonce=issued["nonce"],
                                    username=target_user, password=target_pw,
                                    style=attacker_style, seed=idx * 100 + 80 + a)
            age(prod_db, issued["nonce"], max(e.t for e in session.events) / 1000.0 + 1.0)
            client.post("/auth/login/behavior",
                        json={"username": target_user, "password": target_pw,
                              "session": session.model_dump()})
            limiter.reset()
        print(f"  {attacker_user} -> {target_user}: {IMPOSTOR_LOGINS} impostor attempts")

    evaluation_capture.set_actor(None)

    # --- what landed in the dataset -------------------------------------
    conn = sqlite3.connect(eval_db)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM evaluation_sessions")]
    rejects = [dict(r) for r in conn.execute("SELECT * FROM evaluation_rejects")]
    conn.close()

    print()
    print(f"captured sessions : {len(rows)}")
    print(f"  enrollment      : {sum(1 for r in rows if r['role'] == 'enrollment')}")
    print(f"  genuine logins  : {sum(1 for r in rows if r['role'] == 'login' and r['label'] == 'genuine')}")
    print(f"  impostor logins : {sum(1 for r in rows if r['label'] == 'human_impostor')}")
    print(f"  participants    : {len({r['participant'] for r in rows})}")
    print(f"rejected captures : {len(rejects)}")

    blob = eval_db.read_bytes().lower()
    leaked = [t for t in (b"rehearse0", b"password", b"keydown", b"pointermove")
              if t in blob]
    print()
    print(f"privacy scan of the dataset file: "
          f"{'LEAKED ' + str(leaked) if leaked else 'no username, password or raw-event text found'}")

    print()
    print(f"To inspect before cleanup:  BIOPRINT_EVAL_DB_PATH={eval_db} \\")
    print("    ./.venv/Scripts/python.exe evaluation/eval_collect.py validate")
    print()
    shutil.rmtree(workdir, ignore_errors=True)
    print(f"rehearsal dataset deleted ({workdir}) — generated typists must not")
    print("be mistaken for real-human evidence.")
    set_db_path_override(None)


if __name__ == "__main__":
    main()
