"""Persistence for users, enrollment rounds, profiles and the decision log.

All SQL lives here or in the modules that own their own tables (challenges,
sessions, population). Routes never write SQL, so there is one place to audit
for injection and one place to check what is stored.

Nothing in this module ever writes a raw event stream or a password.
"""

from __future__ import annotations

import json
import sqlite3
import time

from app.behavioral.fingerprint.profile import BehaviorProfile, FeatureStat

CONSENT_VERSION = "2026-09-19.v1"


# ----------------------------------------------------------------- users


def create_user(conn: sqlite3.Connection, username: str, password_hash: str) -> int:
    cursor = conn.execute(
        "INSERT INTO users (username, password_hash, created_at, consent_version) "
        "VALUES (?, ?, ?, ?)",
        (username.lower(), password_hash, time.time(), CONSENT_VERSION),
    )
    return int(cursor.lastrowid)


def get_user(conn: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM users WHERE username = ?", (username.lower(),)
    ).fetchone()


def mark_enrolled(conn: sqlite3.Connection, user_id: int) -> None:
    conn.execute("UPDATE users SET enrolled_at = ? WHERE id = ?", (time.time(), user_id))


# ------------------------------------------------------- enrollment rounds


def add_enrollment_session(
    conn: sqlite3.Connection, user_id: int, features: dict[str, float], coverage: float
) -> None:
    conn.execute(
        "INSERT INTO enrollment_sessions (user_id, features_json, coverage, created_at) "
        "VALUES (?, ?, ?, ?)",
        (
            user_id,
            json.dumps(features, separators=(",", ":")),
            coverage,
            time.time(),
        ),
    )


def load_enrollment_sessions(
    conn: sqlite3.Connection, user_id: int
) -> list[dict[str, float]]:
    rows = conn.execute(
        "SELECT features_json FROM enrollment_sessions WHERE user_id = ? ORDER BY id",
        (user_id,),
    ).fetchall()
    return [json.loads(row["features_json"]) for row in rows]


def count_enrollment_sessions(conn: sqlite3.Connection, user_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM enrollment_sessions WHERE user_id = ?", (user_id,)
    ).fetchone()
    return int(row["n"])


def add_enrollment_windows(
    conn: sqlite3.Connection,
    user_id: int,
    session_index: int,
    windows: list[dict[str, float]],
) -> None:
    """Store the windowed feature vectors for one enrollment round.

    Derived features only, exactly like the session-level row. The raw events
    they came from are discarded in the same request that writes these.
    """
    if not windows:
        return
    now = time.time()
    conn.executemany(
        "INSERT INTO enrollment_windows "
        "(user_id, session_index, features_json, created_at) VALUES (?, ?, ?, ?)",
        [
            (user_id, session_index, json.dumps(w, separators=(",", ":")), now)
            for w in windows
        ],
    )


def load_enrollment_windows(
    conn: sqlite3.Connection, user_id: int
) -> list[tuple[int, dict[str, float]]]:
    """Every stored window as (session_index, features).

    The session index travels with each window so leave-one-session-out
    calibration can hold out a whole round rather than individual windows.
    Windows from the same round overlap, so splitting between them would leak
    the held-out data straight back into training.
    """
    rows = conn.execute(
        "SELECT session_index, features_json FROM enrollment_windows "
        "WHERE user_id = ? ORDER BY id",
        (user_id,),
    ).fetchall()
    return [(int(r["session_index"]), json.loads(r["features_json"])) for r in rows]


def clear_enrollment_windows(conn: sqlite3.Connection, user_id: int) -> None:
    conn.execute("DELETE FROM enrollment_windows WHERE user_id = ?", (user_id,))


def clear_enrollment_sessions(conn: sqlite3.Connection, user_id: int) -> None:
    """Drop captured rounds once a profile has been fitted from them.

    The fitted statistics are what the system needs from here on; keeping the
    per-round feature vectors would retain more behavioural detail than the
    product requires.
    """
    conn.execute("DELETE FROM enrollment_sessions WHERE user_id = ?", (user_id,))


# --------------------------------------------------------------- profiles


def save_profile(conn: sqlite3.Connection, user_id: int, profile: BehaviorProfile) -> int:
    now = time.time()
    conn.execute("DELETE FROM behavior_profiles WHERE user_id = ?", (user_id,))

    cursor = conn.execute(
        "INSERT INTO behavior_profiles "
        "(user_id, session_count, population_size, threshold, threshold_source, "
        " calibration_json, version, update_count, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            profile.session_count,
            profile.population_size,
            profile.threshold,
            profile.threshold_source,
            json.dumps(profile.calibration, separators=(",", ":")),
            profile.version,
            profile.update_count,
            now,
            now,
        ),
    )
    profile_id = int(cursor.lastrowid)

    conn.executemany(
        "INSERT INTO behavior_profile_features "
        "(profile_id, feature, modality, median, median_long, median_recent, "
        " median_enrolled, mad, scale, weight, coverage) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                profile_id,
                stat.name,
                stat.modality,
                stat.median,
                stat.median_long or stat.median,
                stat.median_recent or stat.median,
                stat.median_enrolled or stat.median,
                stat.mad,
                stat.scale,
                stat.weight,
                stat.coverage,
            )
            for stat in profile.features.values()
        ],
    )
    return profile_id


def load_profile(conn: sqlite3.Connection, user_id: int) -> BehaviorProfile | None:
    row = conn.execute(
        "SELECT * FROM behavior_profiles WHERE user_id = ?", (user_id,)
    ).fetchone()
    if row is None:
        return None

    feature_rows = conn.execute(
        "SELECT * FROM behavior_profile_features WHERE profile_id = ?", (row["id"],)
    ).fetchall()

    features = {
        f["feature"]: FeatureStat(
            name=f["feature"],
            modality=f["modality"],
            median=f["median"],
            median_long=f["median_long"],
            median_recent=f["median_recent"],
            median_enrolled=f["median_enrolled"],
            mad=f["mad"],
            scale=f["scale"],
            weight=f["weight"],
            coverage=f["coverage"],
        )
        for f in feature_rows
    }

    calibration = json.loads(row["calibration_json"])
    return BehaviorProfile(
        features=features,
        session_count=row["session_count"],
        population_size=row["population_size"],
        population_informative=bool(calibration.get("metrics", {}).get("impostor_n", 0)),
        threshold=row["threshold"],
        threshold_source=row["threshold_source"],
        calibration=calibration,
        version=row["version"],
        update_count=row["update_count"],
    )


def record_profile_update(
    conn: sqlite3.Connection,
    user_id: int,
    attempt_id: int | None,
    outcome,
    decision: str,
    identity_score: float | None,
    automation_score: float | None,
    coverage: float | None,
) -> None:
    """Audit an adaptation decision, including refusals.

    Declined updates are recorded deliberately. "This profile has not moved in
    forty logins because every one was medium confidence" is exactly the fact
    you need when investigating whether a profile drifted or was poisoned.
    """
    conn.execute(
        "INSERT INTO profile_updates "
        "(user_id, attempt_id, from_version, to_version, confidence, decision, "
        " identity_score, automation_score, coverage, features_updated, reason, "
        " applied, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            attempt_id,
            outcome.from_version,
            outcome.to_version,
            outcome.confidence.value,
            decision,
            identity_score,
            automation_score,
            coverage,
            outcome.features_updated,
            outcome.reason,
            1 if outcome.applied else 0,
            time.time(),
        ),
    )


def profile_update_history(
    conn: sqlite3.Connection, user_id: int, limit: int = 30
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM profile_updates WHERE user_id = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()


def delete_profile(conn: sqlite3.Connection, user_id: int) -> None:
    conn.execute("DELETE FROM behavior_profiles WHERE user_id = ?", (user_id,))
    conn.execute("UPDATE users SET enrolled_at = NULL WHERE id = ?", (user_id,))
    # The anomaly model is part of the profile even though it lives on disk.
    # Leaving it behind would let a stale model score a freshly re-enrolled user.
    from app.behavioral.ml.store import delete_model

    delete_model(user_id)


def reset_operational_state(conn: sqlite3.Connection) -> dict[str, int]:
    """Wipe users, profiles, challenges and audit rows for a live demo.

    Does not insert scores, force ALLOW, or plant behavioural samples.
    """
    tables = (
        "sessions",
        "auth_attempts",
        "auth_challenges",
        "enrollment_sessions",
        "enrollment_windows",
        "behavior_profile_features",
        "behavior_profiles",
        "population_samples",
        "users",
    )
    # The two f-strings below are the only interpolated SQL in this codebase.
    # `tables` is the literal tuple declared directly above; no caller supplies
    # it and no user input reaches it. Table names cannot be bound as
    # parameters in SQLite, so interpolation is the only option here. Every
    # other query in the project uses bound parameters.
    counts: dict[str, int] = {}
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        counts[table] = int(row["n"])
        conn.execute(f"DELETE FROM {table}")
    return counts


# --------------------------------------------------------------- decisions


def record_attempt(
    conn: sqlite3.Connection,
    user_id: int | None,
    username_attempt: str,
    decision: str,
    reason: str,
    reasons: list[str],
    identity_score: float | None,
    automation_score: float | None,
    integrity_score: float,
    coverage: float | None,
    latency_ms: float,
    statistical_identity_score: float | None = None,
    ml_anomaly_score: float | None = None,
) -> int:
    """Append to the decision audit trail.

    Scores and reason codes only. No raw events, no password material, no
    typed text.
    """
    cursor = conn.execute(
        "INSERT INTO auth_attempts "
        "(user_id, username_attempt, decision, reason, reasons_json, identity_score, "
        " statistical_identity_score, ml_anomaly_score, "
        " automation_score, integrity_score, coverage, latency_ms, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            username_attempt,
            decision,
            reason,
            json.dumps(reasons, separators=(",", ":")),
            identity_score,
            statistical_identity_score,
            ml_anomaly_score,
            automation_score,
            integrity_score,
            coverage,
            latency_ms,
            time.time(),
        ),
    )
    return int(cursor.lastrowid)


def recent_attempts(
    conn: sqlite3.Connection, limit: int = 20, user_id: int | None = None
) -> list[sqlite3.Row]:
    if user_id is None:
        return conn.execute(
            "SELECT * FROM auth_attempts ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return conn.execute(
        "SELECT * FROM auth_attempts WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()


def count_recent_failures(
    conn: sqlite3.Connection, username: str, window_seconds: float
) -> int:
    """Failed attempts for a username inside a time window, for rate limiting."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM auth_attempts "
        "WHERE username_attempt = ? AND decision = 'BLOCK' AND created_at > ?",
        (username.lower(), time.time() - window_seconds),
    ).fetchone()
    return int(row["n"])
