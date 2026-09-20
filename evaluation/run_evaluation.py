"""Labelled evaluation against a running BioPrint server.

This is the only script that can produce real authentication accuracy numbers.
Everything else in this directory is synthetic mechanism validation.

Human phases are driven by a person at the keyboard and labelled by bracketing
the audit trail: the highest attempt id is recorded before the phase and every
attempt appearing after it belongs to that phase.

Bot and replay phases are driven by this script through the same HTTP API and
label themselves by attempt id, because bracketing over-collects there. The
replay phase must submit one legitimate capture before it can resubmit it, and
that priming login is not a replay attempt.

Nothing is fabricated. If a category has too few observations the rate is
reported as insufficient rather than computed from three attempts.

Requires the backend running, and the database path so the audit trail can be
read directly. Run from the backend directory:

    ./.venv/Scripts/python.exe ../evaluation/run_evaluation.py --username alice
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics as st
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import httpx  # noqa: E402

from tests.factories import (  # noqa: E402
    human_session,
    scripted_session,
    value_injection_session,
)

BASE_URL = "http://127.0.0.1:8000"
MIN_SAMPLES_FOR_A_RATE = 5

ALLOWED = "ALLOW"
BLOCKED = "BLOCK"

AUTOMATION_REASONS = {"AUTOMATION_DETECTED"}
INTEGRITY_REASONS = {
    "CHALLENGE_REUSED", "CHALLENGE_EXPIRED", "CHALLENGE_UNKNOWN",
    "CHALLENGE_WRONG_USER", "PHRASE_MISMATCH", "TIMESTAMP_INCONSISTENT",
    "MALFORMED_EVENT_STREAM",
}


@dataclass
class Phase:
    label: str
    description: str
    attempts: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.attempts)

    def rate(self, predicate) -> float | None:
        if self.n < MIN_SAMPLES_FOR_A_RATE:
            return None
        return sum(1 for a in self.attempts if predicate(a)) / self.n


def read_attempts(db: Path, after_id: int) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM auth_attempts WHERE id > ? ORDER BY id", (after_id,)
            )
        ]
    finally:
        conn.close()


def read_attempts_by_id(db: Path, ids: list[int]) -> list[dict]:
    """Fetch exactly the attempts a scripted phase created.

    Bracketing a phase by id range over-collects: the replay phase has to
    submit one legitimate capture before it can resubmit it, and that priming
    login is not a replay attempt. Counting it made the replay rejection rate
    read 66.7 percent when every actual replay was in fact rejected. The
    scripted phases know which attempt ids they produced, so they say so.
    """
    if not ids:
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in ids)
    try:
        return [
            dict(r)
            for r in conn.execute(
                f"SELECT * FROM auth_attempts WHERE id IN ({placeholders}) ORDER BY id",
                ids,
            )
        ]
    finally:
        conn.close()


def max_attempt_id(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM auth_attempts").fetchone()
        return int(row[0])
    finally:
        conn.close()


def age_challenge(db: Path, nonce: str, seconds: float) -> None:
    """Only used by the scripted phases, where a whole capture is synthesised
    instantly and would otherwise fail the capture-duration integrity check."""
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


# ------------------------------------------------------------- human phases


def human_phase(db: Path, label: str, instruction: str) -> Phase:
    print(f"\n{'=' * 72}\n  PHASE: {label}\n{'=' * 72}")
    print(instruction)
    start = max_attempt_id(db)
    input("\n  Press Enter when this phase is complete... ")
    attempts = read_attempts(db, start)
    print(f"  recorded {len(attempts)} attempts")
    return Phase(label=label, description=instruction, attempts=attempts)


# ---------------------------------------------------------- scripted phases


def _challenge(client: httpx.Client, username: str) -> dict:
    response = client.post(f"{BASE_URL}/auth/login/challenge", json={"username": username})
    response.raise_for_status()
    return response.json()


class Pacer:
    """Keeps submissions under the server's own login rate limit.

    Worth noting rather than hiding: an attack harness driven from one host
    cannot run flat out against this system, because the rate limiter throttles
    it. That is the limiter working. The pace below is set from LOGIN_LIMIT so
    the evaluation collects a full sample instead of a truncated one.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.seconds:
            time.sleep(self.seconds - elapsed)
        self._last = time.monotonic()


def _submit(
    client: httpx.Client, username: str, password: str, session, pacer: Pacer | None = None
) -> dict:
    if pacer is not None:
        pacer.wait()
    payload = session if isinstance(session, dict) else session.model_dump()
    response = client.post(
        f"{BASE_URL}/auth/login/behavior",
        json={"username": username, "password": password, "session": payload},
    )
    response.raise_for_status()
    return response.json()


def bot_phase(db: Path, username: str, password: str, rounds: int, pacer: Pacer) -> Phase:
    """Automated attempts with a correct password."""
    print(f"\n{'=' * 72}\n  PHASE: BOT (automated by this script)\n{'=' * 72}")
    ids: list[int] = []

    with httpx.Client(timeout=30.0) as client:
        for i in range(rounds):
            for builder in (scripted_session, value_injection_session):
                try:
                    issued = _challenge(client, username)
                except httpx.HTTPStatusError as exc:
                    print(f"  challenge refused ({exc.response.status_code}); pausing")
                    time.sleep(5)
                    continue
                session = builder(
                    issued["phrase"], nonce=issued["nonce"],
                    username=username, password=password,
                )
                duration = max(e.t for e in session.events) / 1000.0
                age_challenge(db, issued["nonce"], duration + 1.0)
                try:
                    verdict = _submit(client, username, password, session, pacer)
                    ids.append(verdict["attempt_id"])
                    print(f"  attempt {i+1} {builder.__name__:26s} -> {verdict['reason']}")
                except httpx.HTTPStatusError as exc:
                    print(f"  rate limited ({exc.response.status_code}); pausing")
                    time.sleep(6)

    attempts = read_attempts_by_id(db, ids)
    print(f"  recorded {len(attempts)} attempts")
    return Phase("BOT", "scripted and value-injection attempts", attempts)


def replay_phase(db: Path, username: str, password: str, rounds: int, pacer: Pacer) -> Phase:
    """Recorded-and-resubmitted attempts, two ways."""
    print(f"\n{'=' * 72}\n  PHASE: REPLAY (automated by this script)\n{'=' * 72}")
    ids: list[int] = []

    with httpx.Client(timeout=30.0) as client:
        for i in range(rounds):
            try:
                issued = _challenge(client, username)
            except httpx.HTTPStatusError:
                time.sleep(6)
                continue

            session = human_session(
                issued["phrase"], nonce=issued["nonce"],
                username=username, password=password, seed=4000 + i,
            )
            duration = max(e.t for e in session.events) / 1000.0
            age_challenge(db, issued["nonce"], duration + 1.0)
            body = session.model_dump()

            try:
                # 1. submit once, then resubmit the identical capture
                # Priming submit. A legitimate login, deliberately NOT counted
                # as a replay attempt.
                _submit(client, username, password, body, pacer)

                verdict = _submit(client, username, password, body, pacer)
                ids.append(verdict["attempt_id"])
                print(f"  attempt {i+1} nonce reuse            -> {verdict['reason']}")

                # 2. old capture against a freshly issued nonce
                fresh = _challenge(client, username)
                stale = {**body, "nonce": fresh["nonce"]}
                age_challenge(db, fresh["nonce"], duration + 1.0)
                verdict = _submit(client, username, password, stale, pacer)
                ids.append(verdict["attempt_id"])
                print(f"  attempt {i+1} old capture, new nonce -> {verdict['reason']}")
            except httpx.HTTPStatusError:
                print("  rate limited; pausing")
                time.sleep(6)

    attempts = read_attempts_by_id(db, ids)
    print(f"  recorded {len(attempts)} attempts")
    return Phase("REPLAY", "nonce reuse and stale-capture resubmission", attempts)


# ------------------------------------------------------------------ report


def _fmt(rate: float | None, n: int) -> str:
    if rate is None:
        return f"insufficient data (n={n}, need {MIN_SAMPLES_FOR_A_RATE})"
    return f"{rate:6.1%}  (n={n})"


def report(phases: dict[str, Phase]) -> dict:
    print(f"\n\n{'=' * 72}\n  RESULTS\n{'=' * 72}\n")

    out: dict[str, object] = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    genuine = phases.get("GENUINE")
    if genuine:
        gar = genuine.rate(lambda a: a["decision"] == ALLOWED)
        frr = genuine.rate(lambda a: a["decision"] == BLOCKED)
        print(f"  Genuine acceptance rate      {_fmt(gar, genuine.n)}")
        print(f"  False rejection rate         {_fmt(frr, genuine.n)}")
        out["genuine"] = {"n": genuine.n, "acceptance_rate": gar, "false_rejection_rate": frr}

    impostor = phases.get("IMPOSTOR")
    if impostor:
        irr = impostor.rate(lambda a: a["decision"] == BLOCKED)
        far = impostor.rate(lambda a: a["decision"] == ALLOWED)
        print(f"  Impostor rejection rate      {_fmt(irr, impostor.n)}")
        print(f"  False acceptance rate        {_fmt(far, impostor.n)}")
        out["impostor"] = {"n": impostor.n, "rejection_rate": irr, "false_acceptance_rate": far}

    bot = phases.get("BOT")
    if bot:
        detected = bot.rate(lambda a: a["decision"] == BLOCKED)
        as_automation = bot.rate(lambda a: a["reason"] in AUTOMATION_REASONS)
        print(f"  Bot block rate               {_fmt(detected, bot.n)}")
        print(f"  ...of which flagged as automation rather than mismatch: "
              f"{_fmt(as_automation, bot.n)}")
        out["bot"] = {"n": bot.n, "block_rate": detected, "automation_reason_rate": as_automation}

    replay = phases.get("REPLAY")
    if replay:
        rejected = replay.rate(lambda a: a["decision"] == BLOCKED)
        by_integrity = replay.rate(lambda a: a["reason"] in INTEGRITY_REASONS)
        print(f"  Replay rejection rate        {_fmt(rejected, replay.n)}")
        print(f"  ...of which by an integrity check: {_fmt(by_integrity, replay.n)}")
        out["replay"] = {"n": replay.n, "rejection_rate": rejected,
                         "integrity_reason_rate": by_integrity}

    latencies = [
        a["latency_ms"] for p in phases.values() for a in p.attempts if a["latency_ms"]
    ]
    if latencies:
        ordered = sorted(latencies)
        p50 = st.median(ordered)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        print(f"\n  Decision latency             p50 {p50:.1f} ms   p95 {p95:.1f} ms   "
              f"(n={len(latencies)}, end to end incl. Argon2id)")
        out["latency"] = {"n": len(latencies), "p50_ms": p50, "p95_ms": p95}

    total = sum(p.n for p in phases.values())
    print(f"\n  Total labelled attempts: {total}")
    if total < 40:
        print("\n  NOTE: this is a small sample. Every rate above carries wide")
        print("  uncertainty and should be reported with its n, not on its own.")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="the enrolled account under test")
    parser.add_argument("--password", required=True, help="that account's password")
    parser.add_argument("--db", default="data/bioprint.db", help="path to the SQLite database")
    parser.add_argument("--bot-rounds", type=int, default=6)
    parser.add_argument("--replay-rounds", type=int, default=5)
    parser.add_argument("--out", default="../evaluation/out/results.json")
    parser.add_argument(
        "--pace",
        type=float,
        default=5.2,
        help="seconds between login submissions, to stay under the server's own "
             "rate limit of 12 per 60 seconds",
    )
    parser.add_argument("--skip-human", action="store_true",
                        help="run only the scripted phases")
    args = parser.parse_args()

    db = Path(args.db).resolve()
    if not db.exists():
        sys.exit(f"database not found at {db}")

    try:
        httpx.get(f"{BASE_URL}/health", timeout=5.0).raise_for_status()
    except Exception as exc:
        sys.exit(f"backend not reachable at {BASE_URL}: {exc}")

    phases: dict[str, Phase] = {}

    if not args.skip_human:
        phases["GENUINE"] = human_phase(
            db, "GENUINE",
            f"  The enrolled user ({args.username}) should now log in repeatedly\n"
            f"  at http://localhost:5173/login, behaving normally.\n"
            f"  Aim for at least 10 attempts. Both successes and failures count.",
        )
        phases["IMPOSTOR"] = human_phase(
            db, "IMPOSTOR",
            f"  A DIFFERENT person now logs in as {args.username}, using the\n"
            f"  correct password. They should type naturally, as themselves.\n"
            f"  Aim for at least 10 attempts.",
        )

    pacer = Pacer(args.pace)
    phases["BOT"] = bot_phase(db, args.username, args.password, args.bot_rounds, pacer)
    phases["REPLAY"] = replay_phase(db, args.username, args.password, args.replay_rounds, pacer)

    results = report(phases)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  written to {out_path}")


if __name__ == "__main__":
    main()
