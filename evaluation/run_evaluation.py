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

**Password failures are separated from behavioural decisions.** A wrong
password is rejected by the credential gate before any behavioural analysis
runs, so it has no identity score and no threshold comparison. Those attempts
are reported as PASSWORD_FAILURE with their count and their attempt ids, and
excluded from FAR and FRR. See evaluation/attempts.py for the reasoning.

Every attempt is also written out individually, so a later forensic pass never
has to reconstruct which database rows belonged to which phase.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attempts import (  # noqa: E402
    ALLOWED,
    AUTOMATION_REASONS,
    BLOCKED,
    CLEAN_PROTOCOL_MIN,
    INTEGRITY_REASONS,
    MIN_SAMPLES_FOR_A_RATE,
    PhaseSummary,
    classify,
    fmt_rate,
)

BASE_URL = "http://127.0.0.1:8000"


@dataclass
class Phase:
    label: str
    description: str
    attempts: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.attempts)

    def summarise(self, profile_threshold: float | None) -> PhaseSummary:
        return PhaseSummary(
            label=self.label,
            records=[classify(a, self.label, profile_threshold) for a in self.attempts],
        )


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


def load_profile_threshold(db: Path, username: str) -> float | None:
    """The account's calibrated threshold.

    Only used to reconstruct the effective threshold for attempts recorded
    before auth_attempts carried a threshold column. Live rows report their
    own; anything reconstructed is labelled as derived in the output.
    """
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT p.threshold FROM behavior_profiles p "
            "JOIN users u ON u.id = p.user_id WHERE u.username = ?",
            (username.lower(),),
        ).fetchone()
        return float(row[0]) if row else None
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


def _protocol_note(summary: PhaseSummary) -> str:
    """One line on whether this phase met the clean protocol."""
    judged, failed = len(summary.behavioral), len(summary.password_failures)
    parts = []
    if failed:
        ids = [r.attempt_id for r in summary.password_failures]
        parts.append(f"{failed} wrong-password attempt(s) excluded (ids {ids})")
    if judged < CLEAN_PROTOCOL_MIN:
        parts.append(f"only {judged} behavioural attempts, protocol asks for "
                     f"{CLEAN_PROTOCOL_MIN}")
    return "; ".join(parts)


def report(phases: dict[str, Phase], profile_threshold: float | None) -> dict:
    print(f"\n\n{'=' * 72}\n  RESULTS\n{'=' * 72}\n")

    summaries = {k: p.summarise(profile_threshold) for k, p in phases.items()}

    out: dict[str, object] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "password_failures_excluded_from_far_frr": True,
            "clean_protocol_min_per_human_phase": CLEAN_PROTOCOL_MIN,
            "min_samples_for_a_rate": MIN_SAMPLES_FOR_A_RATE,
        },
    }

    # A run with no human phases has not met the clean protocol, it has simply
    # not attempted it. Saying "yes" there would be the harness marking its own
    # homework.
    clean = "GENUINE" in summaries and "IMPOSTOR" in summaries

    genuine = summaries.get("GENUINE")
    if genuine:
        gar = genuine.rate(lambda r: r.behavioral_decision == ALLOWED)
        frr = genuine.rate(lambda r: r.behavioral_decision == BLOCKED)
        n = len(genuine.behavioral)
        print(f"  Genuine acceptance rate      {fmt_rate(gar, n)}")
        print(f"  False rejection rate         {fmt_rate(frr, n)}")
        out["genuine"] = {
            **genuine.as_dict(),
            "acceptance_rate": gar,
            "false_rejection_rate": frr,
        }
        note = _protocol_note(genuine)
        if note:
            clean = False
            print(f"       {note}")

    impostor = summaries.get("IMPOSTOR")
    if impostor:
        irr = impostor.rate(lambda r: r.behavioral_decision == BLOCKED)
        far = impostor.rate(lambda r: r.behavioral_decision == ALLOWED)
        n = len(impostor.behavioral)
        print(f"  Impostor rejection rate      {fmt_rate(irr, n)}")
        print(f"  False acceptance rate        {fmt_rate(far, n)}")
        out["impostor"] = {
            **impostor.as_dict(),
            "rejection_rate": irr,
            "false_acceptance_rate": far,
        }
        note = _protocol_note(impostor)
        if note:
            clean = False
            print(f"       {note}")

    bot = summaries.get("BOT")
    if bot:
        detected = bot.rate(lambda r: r.behavioral_decision == BLOCKED)
        as_automation = bot.rate(lambda r: r.rejection_reason in AUTOMATION_REASONS)
        n = len(bot.behavioral)
        print(f"  Bot block rate               {fmt_rate(detected, n)}")
        print(f"  ...of which flagged as automation rather than mismatch: "
              f"{fmt_rate(as_automation, n)}")
        out["bot"] = {**bot.as_dict(), "block_rate": detected,
                      "automation_reason_rate": as_automation}

    replay = summaries.get("REPLAY")
    if replay:
        rejected = replay.rate(lambda r: r.behavioral_decision == BLOCKED)
        by_integrity = replay.rate(lambda r: r.rejection_reason in INTEGRITY_REASONS)
        n = len(replay.behavioral)
        print(f"  Replay rejection rate        {fmt_rate(rejected, n)}")
        print(f"  ...of which by an integrity check: {fmt_rate(by_integrity, n)}")
        out["replay"] = {**replay.as_dict(), "rejection_rate": rejected,
                         "integrity_reason_rate": by_integrity}

    # --- password failures, reported rather than discarded -------------------
    failures = [r for s in summaries.values() for r in s.password_failures]
    print(f"\n  PASSWORD_FAILURE             {len(failures)} attempt(s), excluded from "
          f"FAR and FRR")
    for r in failures:
        print(f"       id {r.attempt_id}  phase {r.phase}  reason {r.rejection_reason}")
    out["password_failures"] = {
        "n": len(failures),
        "attempts": [r.as_dict() for r in failures],
        "note": "credential gate rejected these before any behavioural analysis; "
                "they carry no identity score and no threshold comparison",
    }

    latencies = [
        r.latency_ms for s in summaries.values() for r in s.records if r.latency_ms
    ]
    if latencies:
        ordered = sorted(latencies)
        p50 = st.median(ordered)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        print(f"\n  Decision latency             p50 {p50:.1f} ms   p95 {p95:.1f} ms   "
              f"(n={len(latencies)}, end to end incl. Argon2id)")
        out["latency"] = {"n": len(latencies), "p50_ms": p50, "p95_ms": p95}

    out["attempts"] = [r.as_dict() for s in summaries.values() for r in s.records]

    total = sum(s.total for s in summaries.values())
    out["protocol"]["clean"] = clean  # type: ignore[index]
    print(f"\n  Total labelled attempts: {total}")
    if "GENUINE" not in summaries or "IMPOSTOR" not in summaries:
        print("  Clean protocol met: NO — human phases were not collected")
    else:
        print(f"  Clean protocol met: {'yes' if clean else 'NO — see notes above'}")
    if total < 40:
        print("\n  NOTE: this is a small sample. Every rate above carries wide")
        print("  uncertainty and should be reported with its n, not on its own.")

    return out


# ---------------------------------------------------------------- preflight


def preflight(db: Path) -> None:
    """Refuse to measure a server that is older than the code on disk.

    An earlier run was collected entirely against a uvicorn process started
    before the ML and adaptive-profile layers existed. Nothing in the results
    revealed it: the database schema was current because the file had been
    recreated by a separate process, while the server kept serving the code it
    had loaded hours earlier. The whole run described a build that is not the
    product.
    """
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        health.raise_for_status()
    except Exception as exc:
        sys.exit(f"backend not reachable at {BASE_URL}: {exc}")

    info = health.json()
    started = info.get("started_at")
    if started is None:
        sys.exit(
            "the running server does not report started_at, which means it "
            "predates this check and is definitely stale. Restart it before "
            "collecting an evaluation."
        )

    app_root = Path(info.get("app_root", BACKEND / "app"))
    newest, newest_file = 0.0, None
    for path in app_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        mtime = path.stat().st_mtime
        if mtime > newest:
            newest, newest_file = mtime, path

    if newest > started:
        sys.exit(
            "STALE SERVER. The running process started at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started))} but "
            f"{newest_file} was modified at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest))}.\n"
            "It is serving code older than the source tree. Restart the backend "
            "and start the evaluation again — results from this process would "
            "describe a build that no longer exists."
        )

    if not db.exists():
        sys.exit(f"database not found at {db}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="the enrolled account under test")
    parser.add_argument("--password", required=True, help="that account's password")
    parser.add_argument("--db", default="data/bioprint.db", help="path to the SQLite database")
    parser.add_argument("--bot-rounds", type=int, default=6)
    parser.add_argument("--replay-rounds", type=int, default=5)
    parser.add_argument(
        "--out",
        default=None,
        help="output file; defaults to a timestamped run file in evaluation/out/. "
             "An existing file is never overwritten.",
    )
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
    preflight(db)
    profile_threshold = load_profile_threshold(db, args.username)

    phases: dict[str, Phase] = {}

    if not args.skip_human:
        phases["GENUINE"] = human_phase(
            db, "GENUINE",
            f"  The enrolled user ({args.username}) should now log in repeatedly\n"
            f"  at {BASE_URL}/login, behaving normally.\n"
            f"  Aim for at least {CLEAN_PROTOCOL_MIN} attempts WITH THE CORRECT\n"
            f"  PASSWORD. A mistyped password is rejected by the credential gate\n"
            f"  before any behavioural analysis runs, so it measures nothing —\n"
            f"  if it happens, just do one more attempt. Behavioural successes\n"
            f"  and behavioural failures both count.",
        )
        phases["IMPOSTOR"] = human_phase(
            db, "IMPOSTOR",
            f"  A DIFFERENT person now logs in as {args.username}, using the\n"
            f"  correct password. They should type naturally, as themselves.\n"
            f"  Aim for at least {CLEAN_PROTOCOL_MIN} attempts. The password must\n"
            f"  be entered correctly — the point of this phase is that the\n"
            f"  behavioural layer stops someone who already has the password.\n"
            f"  Repeat any attempt where it was mistyped.",
        )

    pacer = Pacer(args.pace)
    phases["BOT"] = bot_phase(db, args.username, args.password, args.bot_rounds, pacer)
    phases["REPLAY"] = replay_phase(db, args.username, args.password, args.replay_rounds, pacer)

    results = report(phases, profile_threshold)
    results["account"] = args.username
    results["profile_threshold"] = profile_threshold

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else out_dir / f"run_{stamp}.json"

    # Never overwrite. An earlier run is evidence, and a rerun that silently
    # replaced it would destroy the baseline it should be compared against.
    if out_path.exists():
        out_path = out_path.with_name(f"{out_path.stem}_{stamp}{out_path.suffix}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  written to {out_path}")
    print(f"  also copied to {out_dir / 'latest.json'}")
    print(f"  previous runs preserved in {out_dir} and {out_dir / 'baseline'}")


if __name__ == "__main__":
    main()
