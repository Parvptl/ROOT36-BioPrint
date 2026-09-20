#!/usr/bin/env python3
"""Run every demo scenario end to end and report what actually happened.

Drives the real HTTP API against a THROWAWAY database, so the live demo
database and the consented population samples in it are never touched.

    A  enrollment (2 captures)     -> profile built
    B  genuine login               -> ALLOW
    C  human impostor              -> BLOCK, behavioural mismatch
    D  automated script            -> BLOCK, automation detected
    E  replay of a capture         -> BLOCK, integrity
    F  adaptation over logins      -> profile version climbs, maturity advances

Exit code is non-zero if any scenario does not produce its expected outcome,
so this can gate a demo rehearsal rather than merely describing one.

    cd backend
    ./.venv/Scripts/python.exe evaluation/demo_scenarios.py
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app.api.ratelimit import limiter  # noqa: E402
from app.api.routes_auth import ENROLLMENT_ROUNDS  # noqa: E402
from app.db.database import init_db, set_db_path_override  # noqa: E402
from tests.factories import (  # noqa: E402
    TypingStyle,
    human_session,
    scripted_session,
)

OWNER, OWNER_PW = "demouser", "demo-password-98765"
OWNER_STYLE = TypingStyle(iki_mean_ms=138.0, dwell_mean_ms=80.0, overlap_prob=0.46,
                          right_shift_prob=0.93, tab_between_fields=True)
IMPOSTOR_STYLE = TypingStyle(iki_mean_ms=330.0, dwell_mean_ms=145.0, overlap_prob=0.01,
                             right_shift_prob=0.03, tab_between_fields=False,
                             backspace_prob=0.14, pause_prob=0.22)

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


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
    workdir = Path(tempfile.mkdtemp(prefix="bioprint_demo_"))
    db = workdir / "demo.db"
    set_db_path_override(db)
    init_db(db)
    limiter.reset()

    from app.main import app

    client = TestClient(app)

    print("DEMO SCENARIOS (throwaway database — the live demo data is untouched)")
    print(f"  enrollment captures required: {ENROLLMENT_ROUNDS}")
    print()

    # --- A. enrollment --------------------------------------------------
    client.post("/auth/register",
                json={"username": OWNER, "password": OWNER_PW, "consent": True})
    last = None
    for i in range(ENROLLMENT_ROUNDS):
        limiter.reset()
        started = client.post("/auth/enrollment/start",
                              json={"username": OWNER, "password": OWNER_PW}).json()
        session = human_session(started["phrase"], nonce=started["nonce"],
                                username=OWNER, password=OWNER_PW,
                                style=OWNER_STYLE, seed=i)
        age(db, started["nonce"], max(e.t for e in session.events) / 1000.0 + 1.0)
        last = client.post("/auth/enrollment/submit",
                           json={"username": OWNER, "session": session.model_dump()}).json()
    record("A enrollment", bool(last and last.get("profile_built")),
           f"{ENROLLMENT_ROUNDS} captures -> profile_built={last and last.get('profile_built')}, "
           f"features={last and last.get('quality', {}).get('features_modelled')}")

    status = client.get(f"/auth/profile/status?username={OWNER}").json()
    record("A profile state", status.get("maturity") in {"WARMING", "COLD_START"},
           f"maturity={status.get('maturity')} source={status.get('threshold_source')} "
           f"(exact threshold withheld: {'threshold' not in status})")

    def attempt(style, seed, factory=human_session, password=OWNER_PW):
        limiter.reset()
        issued = client.post("/auth/login/challenge", json={"username": OWNER}).json()
        kwargs = dict(nonce=issued["nonce"], username=OWNER, password=password)
        if factory is human_session:
            kwargs["style"] = style
            kwargs["seed"] = seed
        else:
            # scripted_session is deterministic by construction; vary its
            # machine-perfect cadence instead of seeding a random draw.
            kwargs["interval_ms"] = 40.0 + (seed % 5) * 4.0
        session = factory(issued["phrase"], **kwargs)
        age(db, issued["nonce"], max(e.t for e in session.events) / 1000.0 + 1.0)
        return client.post("/auth/login/behavior",
                           json={"username": OWNER, "password": password,
                                 "session": session.model_dump()}).json(), session

    # --- B. genuine ------------------------------------------------------
    allows = 0
    for s in range(5):
        verdict, _ = attempt(OWNER_STYLE, 200 + s)
        allows += int(verdict["decision"] == "ALLOW")
    record("B genuine login", allows >= 4, f"{allows}/5 accepted")

    # --- C. human impostor ------------------------------------------------
    blocks = 0
    reasons = []
    for s in range(5):
        verdict, _ = attempt(IMPOSTOR_STYLE, 300 + s)
        blocks += int(verdict["decision"] == "BLOCK")
        reasons.append(verdict["reason"])
    record("C human impostor", blocks == 5,
           f"{blocks}/5 blocked, reasons={sorted(set(reasons))}")

    # --- D. automated script ----------------------------------------------
    bot_blocks = 0
    bot_reasons = []
    for s in range(4):
        verdict, _ = attempt(None, 400 + s, factory=scripted_session)
        bot_blocks += int(verdict["decision"] == "BLOCK")
        bot_reasons.append(verdict["reason"])
    record("D automated script", bot_blocks == 4,
           f"{bot_blocks}/4 blocked, reasons={sorted(set(bot_reasons))}")

    # --- E. replay ---------------------------------------------------------
    limiter.reset()
    issued = client.post("/auth/login/challenge", json={"username": OWNER}).json()
    session = human_session(issued["phrase"], nonce=issued["nonce"], username=OWNER,
                            password=OWNER_PW, style=OWNER_STYLE, seed=500)
    age(db, issued["nonce"], max(e.t for e in session.events) / 1000.0 + 1.0)
    body = session.model_dump()
    limiter.reset()
    client.post("/auth/login/behavior",
                json={"username": OWNER, "password": OWNER_PW, "session": body})
    limiter.reset()
    reused = client.post("/auth/login/behavior",
                         json={"username": OWNER, "password": OWNER_PW,
                               "session": body}).json()

    limiter.reset()
    fresh = client.post("/auth/login/challenge", json={"username": OWNER}).json()
    stale = {**body, "nonce": fresh["nonce"]}
    age(db, fresh["nonce"], max(e.t for e in session.events) / 1000.0 + 1.0)
    limiter.reset()
    replayed = client.post("/auth/login/behavior",
                           json={"username": OWNER, "password": OWNER_PW,
                                 "session": stale}).json()
    record("E replay (nonce reuse)", reused["decision"] == "BLOCK",
           f"{reused['decision']} / {reused['reason']}")
    record("E replay (stale capture, new nonce)", replayed["decision"] == "BLOCK",
           f"{replayed['decision']} / {replayed['reason']}")

    # --- F. adaptation ------------------------------------------------------
    before = client.get(f"/auth/profile/status?username={OWNER}").json()
    for s in range(10):
        attempt(OWNER_STYLE, 600 + s)
    after = client.get(f"/auth/profile/status?username={OWNER}").json()
    record("F adaptation", (after.get("profile_version") or 0) > (before.get("profile_version") or 0),
           f"version {before.get('profile_version')} -> {after.get('profile_version')}, "
           f"maturity {before.get('maturity')} -> {after.get('maturity')}")

    # --- D2. wrong password --------------------------------------------------
    limiter.reset()
    bad, _ = attempt(OWNER_STYLE, 700, password="wrong-password-000")
    record("G wrong password", bad["reason"] == "INVALID_CREDENTIALS",
           f"{bad['decision']} / {bad['reason']}")

    shutil.rmtree(workdir, ignore_errors=True)
    set_db_path_override(None)

    failed = [name for name, ok, _d in results if not ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} scenarios produced the expected outcome")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
