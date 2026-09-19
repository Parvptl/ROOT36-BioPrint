"""Latency benchmark for the authentication decision path.

Measures the real endpoint through the ASGI stack, not the scoring functions
in isolation, and breaks the total into its named stages so the dominant term
is visible rather than implied.

The two numbers that matter are reported separately:

  behavioural analysis  what BioPrint adds on top of an ordinary login
  end to end            what the user waits for, Argon2 included

Argon2id is deliberately expensive. Reporting only the total would make the
behavioural system look slow for a cost that is the password hash doing its
job, and reporting only the behavioural path would hide what the user
actually experiences. Both are given.

Run:
    cd backend
    ./.venv/Scripts/python.exe ../evaluation/benchmark_latency.py
"""

from __future__ import annotations

import logging
import sqlite3
import statistics as st
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

logging.disable(logging.CRITICAL)

from app.db.database import init_db, set_db_path_override  # noqa: E402

DB = Path(tempfile.mkdtemp(prefix="bioprint-bench-")) / "bench.db"
set_db_path_override(DB)
init_db()

from fastapi.testclient import TestClient  # noqa: E402

from app.api.ratelimit import limiter  # noqa: E402
from app.main import app  # noqa: E402
from tests.test_api_end_to_end import (  # noqa: E402
    attempt_login,
    enroll,
    genuine,
    impostor,
    register,
)

ATTEMPTS = 60
WARMUP = 10


def age_challenge(nonce: str, seconds: float) -> None:
    conn = sqlite3.connect(DB)
    try:
        conn.execute(
            "UPDATE auth_challenges SET issued_at = issued_at - ?, "
            "expires_at = expires_at - ? WHERE nonce = ?",
            (seconds, seconds, nonce),
        )
        conn.commit()
    finally:
        conn.close()


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def report(label: str, values: list[float]) -> None:
    print(
        f"  {label:26s} p50 {st.median(values):7.2f}   p95 {percentile(values, 0.95):7.2f}   "
        f"max {max(values):7.2f}   mean {st.mean(values):7.2f}"
    )


def main() -> None:
    print("Measured on this machine, through the real endpoint. Milliseconds.\n")

    with TestClient(app) as client:
        register(client)
        enroll(client, age_challenge)

        rows = []
        for i in range(ATTEMPTS + WARMUP):
            limiter.reset()
            verdict = attempt_login(
                client, age_challenge, genuine(9000 + i) if i % 2 else impostor(9500 + i)
            )
            if i >= WARMUP:  # discard warm-up: first calls pay import and JIT costs
                rows.append(verdict["latency"])

    total = [r["total_ms"] for r in rows]
    credential = [r["credential_ms"] for r in rows]
    validation = [r["validation_ms"] for r in rows]
    extraction = [r["extraction_ms"] for r in rows]
    identity = [r["identity_ms"] for r in rows]
    automation = [r["automation_ms"] for r in rows]
    persistence = [r["persistence_ms"] for r in rows]
    behavioural = [
        r["validation_ms"] + r["extraction_ms"] + r["identity_ms"] + r["automation_ms"]
        for r in rows
    ]

    print(f"n = {len(rows)} decisions ({WARMUP} warm-up runs discarded)\n")
    print("STAGES")
    report("credential (Argon2id)", credential)
    report("validate + integrity", validation)
    report("feature extraction", extraction)
    report("identity scoring", identity)
    report("automation detection", automation)
    report("persistence (SQLite)", persistence)

    print("\nAGGREGATES")
    report("behavioural analysis", behavioural)
    report("end to end", total)

    share = st.median(credential) / st.median(total)
    print(
        f"\n  Argon2id is {share:.0%} of the median end-to-end time. It is a "
        f"deliberate cost:\n  a memory-hard password hash is what makes a leaked "
        f"user table expensive to crack."
    )
    print(
        "\n  Network time is not included: these calls go through the ASGI app "
        "in-process.\n  On the demo machine the browser talks to localhost, so "
        "the real added\n  round trip is sub-millisecond, but it is not measured "
        "here and is not claimed."
    )


if __name__ == "__main__":
    main()
