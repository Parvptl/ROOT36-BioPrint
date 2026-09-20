#!/usr/bin/env python3
"""Prove the Isolation Forest cannot change an authentication verdict.

Before removing a component from an authentication pipeline, the removal has
to be shown to be a no-op — not argued to be one. `ml_weight = 0.0` says the
blend ignores the anomaly score, but "the config says so" is not evidence: a
weight of zero would still change the outcome if the ML path mutated state,
raised on some input, or fed anything other than the blend.

So this runs the SAME captures through the real HTTP API twice — once with the
model enabled, once with it bypassed — and compares every verdict field that
the product exposes or stores.

Exit code is non-zero if a single verdict differs.

    cd backend
    ./.venv/Scripts/python.exe evaluation/ml_independence.py
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
from app.db.database import init_db, set_db_path_override  # noqa: E402
from tests.factories import (  # noqa: E402
    TypingStyle,
    human_session,
    scripted_session,
    value_injection_session,
)

OWNER, OWNER_PW = "mlprobe", "ml-probe-password-4242"
OWNER_STYLE = TypingStyle(iki_mean_ms=138.0, dwell_mean_ms=80.0, overlap_prob=0.46,
                          right_shift_prob=0.93, tab_between_fields=True)
IMPOSTOR_STYLE = TypingStyle(iki_mean_ms=330.0, dwell_mean_ms=145.0, overlap_prob=0.01,
                             right_shift_prob=0.03, tab_between_fields=False,
                             backspace_prob=0.14, pause_prob=0.22)

# Fields that constitute the authentication outcome. `latency` is excluded —
# it is wall-clock and is EXPECTED to differ, that being the point of removal.
VERDICT_FIELDS = ("decision", "reason", "message", "headline",
                  "integrity_status", "coverage_band")


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


def _install_seeded_phrases(seed: int = 20260920) -> None:
    """Make both runs answer the IDENTICAL sequence of challenge phrases.

    Production draws phrases from `secrets` so an observer cannot predict the
    next challenge. That is correct there and fatal here: with unseeded
    phrases the two runs type different text, so captures differ, so signal
    ORDER (sorted by modality score) and per-round feature coverage differ —
    and those differences look exactly like the ML layer influencing the
    outcome when it is doing nothing of the kind.

    The first version of this script reported four "differences" that were
    entirely this artifact.
    """
    import random

    import app.auth.challenge as challenge

    rng = random.Random(seed)

    def seeded(word_count: int = challenge.PHRASE_WORD_COUNT) -> str:
        words = [rng.choice(challenge._WORDS) for _ in range(word_count)]
        for position in rng.sample(range(word_count), 2):
            words[position] = words[position].capitalize()
        return " ".join(words)

    challenge.generate_phrase = seeded


def run_all(ml_enabled: bool, rounds: int) -> list[dict]:
    """Every scenario, against a fresh database, with ML on or bypassed."""
    import app.behavioral.ml.training as training
    import app.behavioral.ml.hybrid as hybrid
    from dataclasses import replace as dc_replace

    _install_seeded_phrases()
    workdir = Path(tempfile.mkdtemp(prefix=f"bioprint_ml_{int(ml_enabled)}_"))
    db = workdir / "probe.db"
    set_db_path_override(db)
    init_db(db)
    limiter.reset()

    # Flip the master switch in both modules that read it.
    for mod in (training, hybrid):
        mod.settings = dc_replace(mod.settings, ml_enabled=ml_enabled)

    from app.main import app

    client = TestClient(app)
    out: list[dict] = []

    client.post("/auth/register",
                json={"username": OWNER, "password": OWNER_PW, "consent": True})
    for i in range(rounds):
        limiter.reset()
        started = client.post("/auth/enrollment/start",
                              json={"username": OWNER, "password": OWNER_PW,
                                    "rounds": rounds}).json()
        s = human_session(started["phrase"], nonce=started["nonce"],
                          username=OWNER, password=OWNER_PW,
                          style=OWNER_STYLE, seed=i)
        age(db, started["nonce"], max(e.t for e in s.events) / 1000.0 + 1.0)
        r = client.post("/auth/enrollment/submit",
                        json={"username": OWNER, "session": s.model_dump(),
                              "rounds": rounds}).json()
        out.append({"scenario": f"enroll[{i}]", "accepted": r.get("accepted"),
                    "profile_built": r.get("profile_built"),
                    "features": (r.get("quality") or {}).get("features_modelled")})

    def attempt(tag, factory, style, seed, password=OWNER_PW, reuse=None):
        limiter.reset()
        issued = client.post("/auth/login/challenge", json={"username": OWNER}).json()
        if reuse is not None:
            body = reuse
        else:
            kwargs = dict(nonce=issued["nonce"], username=OWNER, password=password)
            if factory is human_session:
                kwargs.update(style=style, seed=seed)
            elif factory is scripted_session:
                # Deterministic by construction; vary its machine cadence.
                kwargs["interval_ms"] = 40.0 + (seed % 5) * 4.0
            s = factory(issued["phrase"], **kwargs)
            age(db, issued["nonce"], max(e.t for e in s.events) / 1000.0 + 1.0)
            body = s.model_dump()
        v = client.post("/auth/login/behavior",
                        json={"username": OWNER, "password": password,
                              "session": body}).json()
        out.append({"scenario": tag,
                    **{k: v.get(k) for k in VERDICT_FIELDS},
                    "signals": [s_["code"] for s_ in v.get("signals", [])]})
        return v, body

    for i in range(6):
        attempt(f"genuine[{i}]", human_session, OWNER_STYLE, 200 + i)
    for i in range(6):
        attempt(f"impostor[{i}]", human_session, IMPOSTOR_STYLE, 300 + i)
    for i in range(3):
        attempt(f"bot_scripted[{i}]", scripted_session, None, 400 + i)
    for i in range(3):
        attempt(f"bot_injection[{i}]", value_injection_session, None, 500 + i)
    for i in range(2):
        attempt(f"wrongpw[{i}]", human_session, OWNER_STYLE, 600 + i,
                password="not-the-password")

    # replay: prime, then resubmit the identical capture, then a stale one
    _v, body = attempt("replay_prime", human_session, OWNER_STYLE, 700)
    attempt("replay_reuse", human_session, OWNER_STYLE, 700, reuse=body)
    limiter.reset()
    fresh = client.post("/auth/login/challenge", json={"username": OWNER}).json()
    stale = {**body, "nonce": fresh["nonce"]}
    age(db, fresh["nonce"], 30.0)
    attempt("replay_stale", human_session, OWNER_STYLE, 700, reuse=stale)

    # post-state: adaptation and maturity must land identically
    status = client.get(f"/auth/profile/status?username={OWNER}").json()
    out.append({"scenario": "final_profile",
                "maturity": status.get("maturity"),
                "profile_version": status.get("profile_version"),
                "threshold_source": status.get("threshold_source"),
                "feature_count": status.get("feature_count")})

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        updates = [dict(r) for r in conn.execute(
            "SELECT confidence, decision, applied, features_updated "
            "FROM profile_updates ORDER BY id")]
    finally:
        conn.close()
    out.append({"scenario": "adaptation_trail", "updates": updates})

    shutil.rmtree(workdir, ignore_errors=True)
    set_db_path_override(None)
    return out


def main() -> None:
    from app.api.routes_auth import ENROLLMENT_ROUNDS, RESEARCH_ENROLLMENT_ROUNDS

    print("ISOLATION FOREST INDEPENDENCE PROOF")
    print("  Same captures, same seeds, ML enabled vs bypassed.")
    print("  Compares verdicts, signals, adaptation trail and final profile state.")
    print()

    failures = 0
    for label, rounds in (("2-capture (production)", ENROLLMENT_ROUNDS),
                          ("8-capture (research, trains a model)",
                           RESEARCH_ENROLLMENT_ROUNDS)):
        print(f"--- {label} ---")
        with_ml = run_all(True, rounds)
        without_ml = run_all(False, rounds)

        if len(with_ml) != len(without_ml):
            print(f"  MISMATCH: {len(with_ml)} vs {len(without_ml)} records")
            failures += 1
            continue

        diffs = [(a["scenario"], a, b) for a, b in zip(with_ml, without_ml) if a != b]
        print(f"  {len(with_ml)} records compared, {len(diffs)} differ")
        for scenario, a, b in diffs:
            print(f"    {scenario}:")
            for key in set(a) | set(b):
                if a.get(key) != b.get(key):
                    print(f"      {key}: ml_on={a.get(key)!r}  ml_off={b.get(key)!r}")
        failures += len(diffs)
        print()

    print("=" * 74)
    if failures:
        print(f"  NOT INDEPENDENT — {failures} difference(s). Do not remove it.")
        sys.exit(1)
    print("  IDENTICAL. Every verdict, signal, adaptation decision and final")
    print("  profile state matches with the Isolation Forest enabled and bypassed.")
    print("  It cannot influence an authentication outcome.")


if __name__ == "__main__":
    main()
