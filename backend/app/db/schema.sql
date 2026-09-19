-- BioPrint storage schema.
--
-- Design rule: raw behavioural event streams are NEVER a column here. Events
-- are extracted into features in-process and dropped. What persists is derived
-- statistics plus the decision audit trail.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    username          TEXT    NOT NULL UNIQUE,
    password_hash     TEXT    NOT NULL,          -- Argon2id encoded hash, never plaintext
    created_at        REAL    NOT NULL,
    enrolled_at       REAL,                      -- NULL until a profile is built
    consent_version   TEXT                       -- which consent text the user agreed to
);

-- One row per enrolled user. Holds profile-level scalars; per-feature stats
-- live in behavior_profile_features.
CREATE TABLE IF NOT EXISTS behavior_profiles (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id            INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    session_count      INTEGER NOT NULL,         -- enrollment sessions used to build it
    population_size    INTEGER NOT NULL,         -- population samples backing the prior
    threshold          REAL    NOT NULL,         -- calibrated identity-score cut point
    threshold_source   TEXT    NOT NULL,         -- 'calibrated' | 'fallback_prior'
    calibration_json   TEXT    NOT NULL,         -- LOO genuine scores + method metadata
    created_at         REAL    NOT NULL,
    updated_at         REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS behavior_profile_features (
    profile_id    INTEGER NOT NULL REFERENCES behavior_profiles(id) ON DELETE CASCADE,
    feature       TEXT    NOT NULL,
    modality      TEXT    NOT NULL,
    median        REAL    NOT NULL,              -- robust centre across enrollment sessions
    mad           REAL    NOT NULL,              -- raw median absolute deviation
    scale         REAL    NOT NULL,              -- shrinkage-floored scale actually used
    weight        REAL    NOT NULL,              -- discriminability weight
    coverage      REAL    NOT NULL,              -- fraction of sessions where observable
    PRIMARY KEY (profile_id, feature)
);

-- Population prior: one row per consented sample contributed by a volunteer.
-- Stores extracted features only, and is never linked back to a user identity.
CREATE TABLE IF NOT EXISTS population_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT NOT NULL,                 -- 'enrollment' | 'volunteer'
    features_json TEXT NOT NULL,
    created_at    REAL NOT NULL
);

-- Single-use, expiring behavioural challenges. `consumed_at` is what makes a
-- replayed nonce fail on the second use.
CREATE TABLE IF NOT EXISTS auth_challenges (
    nonce        TEXT PRIMARY KEY,
    user_id      INTEGER REFERENCES users(id) ON DELETE CASCADE,
    purpose      TEXT    NOT NULL,               -- 'login' | 'enrollment'
    phrase       TEXT    NOT NULL,               -- randomised prompt the client must type
    issued_at    REAL    NOT NULL,
    expires_at   REAL    NOT NULL,
    consumed_at  REAL                            -- NULL until first use
);

CREATE INDEX IF NOT EXISTS idx_challenges_expiry ON auth_challenges(expires_at);

-- Audit trail of every decision. Scores and reason codes only; no raw events,
-- no password material.
CREATE TABLE IF NOT EXISTS auth_attempts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER REFERENCES users(id) ON DELETE SET NULL,
    username_attempt  TEXT    NOT NULL,
    decision          TEXT    NOT NULL,          -- ALLOW | BLOCK
    reason            TEXT    NOT NULL,          -- primary reason code
    reasons_json      TEXT    NOT NULL,          -- all contributing reason codes
    identity_score    REAL,
    automation_score  REAL,
    integrity_score   REAL,
    coverage          REAL,
    latency_ms        REAL,
    created_at        REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attempts_user_time ON auth_attempts(user_id, created_at DESC);

-- Server-side sessions issued on ALLOW. Storing a hash means a leaked DB does
-- not hand an attacker usable session tokens.
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);
