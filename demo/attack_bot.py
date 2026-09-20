"""Single scripted login attempt, for the live demo.

Runs one automated attack against a running BioPrint server and prints the
verdict in a form that reads from the back of a room.

This is the attacker's side of demo step 5. It uses the same HTTP API a real
attacker would, with a correct password.

    cd backend
    ./.venv/Scripts/python.exe ../demo/attack_bot.py --username alice --password <pw>
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

from tests.factories import scripted_session, value_injection_session  # noqa: E402

BASE_URL = "http://127.0.0.1:8000"

STYLES = {
    "timer": (
        scripted_session,
        "keystrokes dispatched on a fixed timer, the way a naive script does it",
    ),
    "inject": (
        value_injection_session,
        "fields filled by assigning to element.value, with no keystrokes at all",
    ),
}


def age_challenge(db: Path, nonce: str, seconds: float) -> None:
    """Match the challenge age to the capture length.

    A synthesised capture is produced instantly, so without this it fails the
    capture-duration integrity check before automation detection is even
    reached. Ageing it lets the demo show the automation verdict rather than an
    integrity one, which is the point of this step.
    """
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True, help="the CORRECT password")
    parser.add_argument("--style", choices=sorted(STYLES), default="timer")
    parser.add_argument("--db", default="data/bioprint.db")
    args = parser.parse_args()

    builder, description = STYLES[args.style]
    db = Path(args.db).resolve()

    print()
    print("=" * 62)
    print("  AUTOMATED LOGIN ATTEMPT")
    print("=" * 62)
    print(f"  account   : {args.username}")
    print("  password  : CORRECT")
    print(f"  method    : {description}")
    print()

    with httpx.Client(timeout=30.0) as client:
        issued = client.post(
            f"{BASE_URL}/auth/login/challenge", json={"username": args.username}
        )
        issued.raise_for_status()
        challenge = issued.json()
        print(f"  phrase    : {challenge['phrase']}")

        session = builder(
            challenge["phrase"],
            nonce=challenge["nonce"],
            username=args.username,
            password=args.password,
        )
        age_challenge(db, challenge["nonce"], max(e.t for e in session.events) / 1000.0 + 1.0)

        response = client.post(
            f"{BASE_URL}/auth/login/behavior",
            json={
                "username": args.username,
                "password": args.password,
                "session": session.model_dump(),
            },
        )
        response.raise_for_status()
        verdict = response.json()

    print()
    print("=" * 62)
    print(f"  {verdict.get('headline') or verdict['decision']}")
    print("=" * 62)
    print(f"  {verdict['message']}")
    print(f"  reason code : {verdict['reason']}")
    print(f"  latency     : {verdict['latency']['total_ms']:.0f} ms")
    if verdict["signals"]:
        print()
        print("  contributing signals:")
        for signal in verdict["signals"]:
            print(f"    [{signal['band']:6s}] {signal['label']}")
            print(f"             {signal['detail']}")
    print()
    print("  Note: the response carries no score and no threshold. Exact values")
    print("  are on the audit trail; the attacker is not given a gradient.")
    print()

    sys.exit(0 if verdict["decision"] == "BLOCK" else 1)


if __name__ == "__main__":
    main()
