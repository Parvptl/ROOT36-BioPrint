"""Runtime configuration, loaded once at import from the environment.

Everything that could differ between a dev machine, the demo laptop, and a
hosted deployment lives here. Nothing in this module has a hardcoded secret.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(BACKEND_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_db_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    return path


@dataclass(frozen=True)
class Settings:
    secret_key: str
    db_path: Path
    challenge_ttl_seconds: int
    session_ttl_seconds: int
    cors_origins: list[str] = field(default_factory=list)
    retain_raw_events: bool = False
    # True when no secret was configured and we generated a throwaway one. An
    # ephemeral secret means every restart invalidates all sessions: fine for a
    # dev run, wrong for anything else, so startup logs a warning.
    secret_is_ephemeral: bool = False
    # Empty disables the HTTP demo-reset endpoint. The CLI reset never needs it.
    demo_reset_key: str = ""
    # Empty disables the operator dashboard. It serves exact per-attempt
    # identity and automation scores, which the login response withholds on
    # purpose, so it must never default to open.
    operator_key: str = ""


def load_settings() -> Settings:
    configured_secret = os.getenv("BIOPRINT_SECRET_KEY", "").strip()
    ephemeral = False
    placeholder = "change-me-generate-a-real-secret"
    if not configured_secret or configured_secret == placeholder:
        # Never ship a default secret in source. Generating one per process is
        # the safe failure mode: sessions break on restart instead of every
        # deployment sharing a key an attacker can read off GitHub.
        configured_secret = secrets.token_urlsafe(48)
        ephemeral = True

    origins_raw = os.getenv(
        "BIOPRINT_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    origins = [o.strip() for o in origins_raw.split(",") if o.strip()]

    return Settings(
        secret_key=configured_secret,
        db_path=_resolve_db_path(os.getenv("BIOPRINT_DB_PATH", "data/bioprint.db")),
        challenge_ttl_seconds=_env_int("BIOPRINT_CHALLENGE_TTL_SECONDS", 120),
        session_ttl_seconds=_env_int("BIOPRINT_SESSION_TTL_SECONDS", 1800),
        cors_origins=origins,
        retain_raw_events=_env_bool("BIOPRINT_RETAIN_RAW_EVENTS", False),
        secret_is_ephemeral=ephemeral,
        demo_reset_key=os.getenv("BIOPRINT_DEMO_RESET_KEY", "").strip(),
        operator_key=os.getenv("BIOPRINT_OPERATOR_KEY", "").strip(),
    )


settings = load_settings()
