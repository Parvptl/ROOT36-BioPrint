#!/usr/bin/env python3
"""Where does a slow login actually spend its time?

The Phase F benchmark measured ~227 ms end to end, but the live UI showed
2394 ms and the server's own log showed 935 ms and 1357 ms for real browser
logins. The server-reported number is `LatencyBreakdown.total_ms`, measured
inside the request handler, so this is not a frontend or network artifact —
something in the decision path is genuinely slower for real traffic than for
the benchmark's traffic.

This walks the two variables that differ between them:

  1. EVENT VOLUME. The browser collector emits up to 2500 pointermove events
     and 8000 events total. The benchmark's synthetic sessions are far
     smaller, so it never measured the extraction cost of a real capture.

  2. PROFILE MATURITY. The benchmark enrolled 8 rounds, which produces a
     MATURE profile. The product now enrolls 2 captures, which produces a
     WARMING one — and routes_login does extra work for a non-MATURE profile
     on every accepted login.

Read-only with respect to security: nothing here changes a parameter, it only
measures. Runs against a throwaway database.

    cd backend
    ./.venv/Scripts/python.exe evaluation/latency_trace.py
"""

from __future__ import annotations

import shutil
import sqlite3
import statistics as st
import sys
import tempfile
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app.api.ratelimit import limiter  # noqa: E402
from app.db.database import init_db, set_db_path_override  # noqa: E402
from app.models.events import PointerMoveIn  # noqa: E402
from tests.factories import TypingStyle, human_session  # noqa: E402

STYLE = TypingStyle(iki_mean_ms=140.0, dwell_mean_ms=82.0, overlap_prob=0.45,
                    right_shift_prob=0.95, tab_between_fields=True)

USER, PW = "latencyprobe", "latency-probe-password-1"

STAGES = ("credential_ms", "validation_ms", "extraction_ms", "identity_ms",
          "automation_ms", "ml_inference_ms", "persistence_ms")


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


def inflate_pointer(session, target_moves: int):
    """Pad a synthetic capture with pointer movement, as a real browser does.

    The collector samples pointermove continuously while the form is open, so
    a capture that took twenty seconds carries far more motion than a
    synthetic one built from a handful of waypoints. The padding follows a
    smooth path so the pointer features stay plausible rather than degenerate.
    """
    import math

    events = list(session.events)
    existing = [e for e in events if getattr(e, "type", None) == "pointermove"]
    if not existing:
        return session

    last_t = max(e.t for e in events)
    start_t = min(e.t for e in existing)
    span = max(last_t - start_t, 1.0)
    need = max(0, target_moves - len(existing))

    padded = []
    for i in range(need):
        frac = i / max(need, 1)
        t = start_t + frac * span
        # A gentle arc with jitter, so straightness/tremor stay human-shaped.
        x = 300.0 + 220.0 * math.sin(frac * 3.1) + 4.0 * math.sin(i * 0.7)
        y = 260.0 + 140.0 * math.cos(frac * 2.3) + 4.0 * math.cos(i * 0.9)
        padded.append(PointerMoveIn(type="pointermove", t=t, x=x, y=y,
                                    trusted=True))

    merged = sorted(events + padded, key=lambda e: e.t)
    return session.model_copy(update={"events": merged})


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="bioprint_latency_"))
    db = workdir / "probe.db"
    set_db_path_override(db)
    init_db(db)
    limiter.reset()

    from app.api.routes_auth import RESEARCH_ENROLLMENT_ROUNDS
    from app.main import app

    client = TestClient(app)

    def enroll(username, password, rounds):
        client.post("/auth/register",
                    json={"username": username, "password": password, "consent": True})
        for i in range(rounds):
            limiter.reset()
            started = client.post("/auth/enrollment/start",
                                  json={"username": username, "password": password,
                                        "rounds": rounds}).json()
            s = human_session(started["phrase"], nonce=started["nonce"],
                              username=username, password=password,
                              style=STYLE, seed=i)
            age(db, started["nonce"], max(e.t for e in s.events) / 1000.0 + 1.0)
            client.post("/auth/enrollment/submit",
                        json={"username": username, "session": s.model_dump(),
                              "rounds": rounds})

    def login(username, password, seed, pointer_moves=0):
        limiter.reset()
        issued = client.post("/auth/login/challenge",
                             json={"username": username}).json()
        s = human_session(issued["phrase"], nonce=issued["nonce"],
                          username=username, password=password,
                          style=STYLE, seed=seed)
        if pointer_moves:
            s = inflate_pointer(s, pointer_moves)
        age(db, issued["nonce"], max(e.t for e in s.events) / 1000.0 + 1.0)
        body = s.model_dump()
        r = client.post("/auth/login/behavior",
                        json={"username": username, "password": password,
                              "session": body}).json()
        return r, len(body["events"])

    print("LATENCY TRACE — throwaway database, no security parameter touched")
    print()

    # --- 1. does event volume explain it? ---------------------------------
    print("=" * 96)
    print("  1. EVENT VOLUME  (2-capture WARMING profile, as the product now enrols)")
    print("=" * 96)
    enroll(USER, PW, 2)
    # warm up: first call loads the prior, imports sklearn, opens the db
    login(USER, PW, 900)

    head = (f"{'events':>8s} {'total':>9s} " +
            " ".join(f"{s.replace('_ms',''):>11s}" for s in STAGES))
    print(head)
    print("-" * len(head))

    volume_rows = []
    for moves in (0, 250, 750, 1500, 2500):
        samples = []
        n_events = 0
        for k in range(5):
            r, n_events = login(USER, PW, 1000 + moves + k, pointer_moves=moves)
            samples.append(r["latency"])
        med = {key: st.median([s[key] for s in samples]) for key in
               ("total_ms", *STAGES)}
        volume_rows.append((n_events, med))
        print(f"{n_events:8d} {med['total_ms']:9.1f} " +
              " ".join(f"{med[s]:11.2f}" for s in STAGES))

    # --- 2. does profile maturity explain it? ------------------------------
    print()
    print("=" * 96)
    print("  2. PROFILE MATURITY  (same event volume, 1500 pointer moves)")
    print("=" * 96)
    enroll("matureprobe", PW, RESEARCH_ENROLLMENT_ROUNDS)
    login("matureprobe", PW, 950)

    print(f"{'profile':<24s} {'total':>9s} " +
          " ".join(f"{s.replace('_ms',''):>11s}" for s in STAGES))
    print("-" * 96)
    for label, username in (("WARMING (2 captures)", USER),
                            ("MATURE (8 captures)", "matureprobe")):
        samples = []
        for k in range(5):
            r, _ = login(username, PW, 2000 + k, pointer_moves=1500)
            samples.append(r["latency"])
        med = {key: st.median([s[key] for s in samples]) for key in
               ("total_ms", *STAGES)}
        print(f"{label:<24s} {med['total_ms']:9.1f} " +
              " ".join(f"{med[s]:11.2f}" for s in STAGES))

    # --- 3. accounted-for vs total ----------------------------------------
    print()
    print("=" * 96)
    print("  3. IS THE TOTAL ACCOUNTED FOR?  (total minus the sum of the stages)")
    print("=" * 96)
    for n_events, med in volume_rows:
        accounted = sum(med[s] for s in STAGES)
        print(f"  {n_events:6d} events: total {med['total_ms']:8.1f} ms, "
              f"stages sum {accounted:8.1f} ms, unexplained "
              f"{med['total_ms'] - accounted:7.1f} ms")

    shutil.rmtree(workdir, ignore_errors=True)
    set_db_path_override(None)


if __name__ == "__main__":
    main()
