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
    -- Incremented on every adaptive update. Recorded on each profile_updates
    -- row so a profile's history is reconstructable.
    version            INTEGER NOT NULL DEFAULT 1,
    update_count       INTEGER NOT NULL DEFAULT 0,
    created_at         REAL    NOT NULL,
    updated_at         REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS behavior_profile_features (
    profile_id    INTEGER NOT NULL REFERENCES behavior_profiles(id) ON DELETE CASCADE,
    feature       TEXT    NOT NULL,
    modality      TEXT    NOT NULL,
    median        REAL    NOT NULL,              -- EFFECTIVE centre used for scoring
    -- Two timescales behind that effective value. The long track holds stable
    -- identity; the recent track follows natural drift. `median` is their
    -- blend, so the scoring path is unchanged by adaptation existing at all.
    median_long   REAL    NOT NULL DEFAULT 0.0,
    median_recent REAL    NOT NULL DEFAULT 0.0,
    median_enrolled REAL  NOT NULL DEFAULT 0.0,
    mad           REAL    NOT NULL,              -- raw median absolute deviation
    scale         REAL    NOT NULL,              -- shrinkage-floored scale actually used
    weight        REAL    NOT NULL,              -- discriminability weight
    coverage      REAL    NOT NULL,              -- fraction of sessions where observable
    PRIMARY KEY (profile_id, feature)
);

-- Captured enrollment rounds, held until there are enough to fit a profile.
-- Features only; the raw events that produced them were discarded at the
-- moment of extraction.
CREATE TABLE IF NOT EXISTS enrollment_sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    features_json TEXT    NOT NULL,
    coverage      REAL    NOT NULL,
    created_at    REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_enrollment_user ON enrollment_sessions(user_id);

-- Windowed feature vectors from enrollment, the training set for the per-user
-- anomaly model.
--
-- These are computed at submission time, while the raw events are still in
-- memory, precisely so the raw events can go on being discarded. What is kept
-- is the same kind of derived statistic already stored per session, just at a
-- finer granularity. Deleted alongside enrollment_sessions once the model is
-- fitted.
CREATE TABLE IF NOT EXISTS enrollment_windows (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_index INTEGER NOT NULL,
    features_json TEXT    NOT NULL,
    created_at    REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_enrollment_windows_user ON enrollment_windows(user_id);

-- Population prior: one row per consented sample.
--
-- `contributor` is a keyed hash of the user id, not the id itself. It exists
-- only so a user's own samples can be excluded from the population they are
-- compared against; without the server secret it cannot be linked to anyone.
CREATE TABLE IF NOT EXISTS population_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT NOT NULL,                 -- 'enrollment' | 'volunteer'
    contributor   TEXT,
    features_json TEXT NOT NULL,
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_population_contributor ON population_samples(contributor);

-- Single-use, expiring behavioural challenges. `consumed_at` is what makes a
-- replayed nonce fail on the second use.
-- `username_claim` rather than a user_id foreign key: a challenge is handed out
-- for whatever username was typed, existing or not, so this endpoint cannot be
-- used to enumerate which accounts are registered.
CREATE TABLE IF NOT EXISTS auth_challenges (
    nonce           TEXT PRIMARY KEY,
    username_claim  TEXT    NOT NULL,
    purpose         TEXT    NOT NULL,            -- 'login' | 'enrollment'
    phrase          TEXT    NOT NULL,            -- randomised prompt the client must type
    issued_at       REAL    NOT NULL,
    expires_at      REAL    NOT NULL,
    consumed_at     REAL                         -- NULL until first use
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
    -- The blend's components, kept beside it so an operator can see which
    -- layer drove a decision rather than only the fused number.
    statistical_identity_score REAL,
    ml_anomaly_score  REAL,
    automation_score  REAL,
    integrity_score   REAL,
    coverage          REAL,
    -- The bar this attempt was actually judged against: the profile threshold
    -- after automation tightening. Stored rather than recomputed at read time,
    -- because a score without the threshold it was compared to cannot be
    -- interpreted, and the tightening factor is not visible in this table.
    threshold         REAL,
    latency_ms        REAL,
    created_at        REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attempts_user_time ON auth_attempts(user_id, created_at DESC);

-- Audit trail for adaptive profile updates, including the ones that were
-- DECLINED.
--
-- Recording refusals matters as much as recording updates: "this profile has
-- not moved in forty logins because every one was medium confidence" is the
-- kind of fact you need when investigating whether a profile was poisoned.
CREATE TABLE IF NOT EXISTS profile_updates (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    attempt_id       INTEGER,
    from_version     INTEGER NOT NULL,
    to_version       INTEGER NOT NULL,
    confidence       TEXT    NOT NULL,           -- HIGH|MEDIUM|SUSPICIOUS|BLOCKED|BOT
    decision         TEXT    NOT NULL,
    identity_score   REAL,
    automation_score REAL,
    coverage         REAL,
    features_updated INTEGER NOT NULL,
    reason           TEXT    NOT NULL,
    applied          INTEGER NOT NULL,           -- 1 updated, 0 declined
    created_at       REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_profile_updates_user
    ON profile_updates(user_id, created_at DESC);

-- Server-side sessions issued on ALLOW. Storing a hash means a leaked DB does
-- not hand an attacker usable session tokens.
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);
