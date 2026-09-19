"""Replay attack, for the live demo.

Answers the question a judge will ask: "what if I record a genuine login and
play it back?"

Two variants, both with the correct password:

  reuse  resubmit the exact capture that was just accepted
  stale  take that capture and submit it against a freshly issued challenge

The second is the interesting one. It is what an attacker would actually try
after capturing a victim's session, and it is what the per-attempt randomised
phrase exists to defeat.

    cd backend
    ./.venv/Scripts/python.exe ../demo/attack_replay.py --username alice --password <pw>
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import httpx  # noqa: E402

from tests.factories import human_session  # noqa: E402

BASE_URL = "http://127.0.0.1:8000"


def age_challenge(db: Path, nonce: str, seconds: float) -> None:
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


def show(step: str, verdict: dict) -> None:
    print()
    print("-" * 62)
    print(f"  {step}")
    print(f"  -> {verdict.get('headline') or verdict['decision']}   [{verdict['reason']}]")
    print(f"     {verdict['message']}")
    print(f"     integrity: {verdict.get('integrity_status')}   "
          f"latency: {verdict['latency']['total_ms']:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True, help="the CORRECT password")
    parser.add_argument("--db", default="data/bioprint.db")
    args = parser.parse_args()

    db = Path(args.db).resolve()

    print()
    print("=" * 62)
    print("  REPLAY ATTACK")
    print("=" * 62)
    print(f"  account  : {args.username}")
    print("  password : CORRECT")
    print("  premise  : the attacker has captured a complete genuine session")

    with httpx.Client(timeout=30.0) as client:
        # Capture a session that the system accepts, so the replay starts from
        # something genuinely valid rather than from something already broken.
        first = client.post(
            f"{BASE_URL}/auth/login/challenge", json={"username": args.username}
        )
        first.raise_for_status()
        challenge = first.json()
        print(f"\n  captured against phrase: {challenge['phrase']}")

        session = human_session(
            challenge["phrase"],
            nonce=challenge["nonce"],
            username=args.username,
            password=args.password,
            seed=20_260_920,
        )
        duration = max(e.t for e in session.events) / 1000.0
        age_challenge(db, challenge["nonce"], duration + 1.0)
        body = session.model_dump()

        def submit(payload: dict) -> dict:
            response = client.post(
                f"{BASE_URL}/auth/login/behavior",
                json={
                    "username": args.username,
                    "password": args.password,
                    "session": payload,
                },
            )
            response.raise_for_status()
            return response.json()

        show("1. the original capture, submitted once", submit(body))
        show("2. REPLAY: the identical capture, submitted again", submit(body))

        fresh = client.post(
            f"{BASE_URL}/auth/login/challenge", json={"username": args.username}
        )
        fresh.raise_for_status()
        new_challenge = fresh.json()
        age_challenge(db, new_challenge["nonce"], duration + 1.0)

        print(f"\n  new phrase issued: {new_challenge['phrase']}")
        show(
            "3. REPLAY: the captured behaviour against the new challenge",
            submit({**body, "nonce": new_challenge["nonce"]}),
        )

    print()
    print("=" * 62)
    print("  The nonce is single-use, so step 2 fails on reuse. Step 3 fails")
    print("  because the recording answers the previous phrase, and the phrase")
    print("  is regenerated every attempt.")
    print()
    print("  Stated limitation: this defeats replaying a RECORDING. It does not")
    print("  defeat an attacker who re-synthesises captured timing onto the new")
    print("  text in real time. That case falls to the automation detector,")
    print("  which is the weaker of the two defences.")
    print()


if __name__ == "__main__":
    main()
