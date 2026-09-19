"""Single-use, expiring behavioural challenges.

Every enrollment round and every login attempt is bound to a fresh nonce and a
freshly generated phrase. Two jobs, one mechanism:

1. Replay resistance. An attacker who records a genuine user's complete event
   stream cannot resubmit it: the nonce is already burned, and the phrase they
   recorded is not the phrase being asked for now.

2. Content independence. Because the prompt text differs every time, the
   fingerprint cannot be a template of one fixed string. Features are computed
   over whatever text was asked for, which is what keeps the profile
   independent of the password.

Residual limitation, stated plainly: this stops *static* replay. An adversary
who records timing distributions and then re-synthesises them onto new prompt
text is not defeated by the nonce alone — that case is the automation
detector's job, and it remains the weaker of the two defences.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
import time
from dataclasses import dataclass

from app.behavioral.scoring.reasons import ReasonCode
from app.config import settings

# Common lowercase English words, 3-7 letters. Chosen for digraph coverage
# ("th", "he", "in", "er", "an", "re", "on", "at", "en", "nd") so that a short
# phrase still yields enough repeated transitions to estimate timing on.
# All alphabetic and unambiguous, so a typo is the user's, not the word list's.
_WORDS: tuple[str, ...] = (
    "anchor", "amber", "andes", "answer", "artist", "autumn", "banner", "barrel",
    "beacon", "bridge", "bronze", "candle", "canyon", "carbon", "castle", "cedar",
    "center", "cinder", "circle", "cobalt", "copper", "coral", "cotton", "crater",
    "crystal", "denim", "desert", "dinner", "dragon", "ember", "engine", "enter",
    "falcon", "father", "feather", "forest", "garden", "gather", "ginger", "granite",
    "harbor", "hunter", "indigo", "inland", "iron", "island", "jungle", "kernel",
    "lantern", "leather", "linen", "lunar", "marble", "meadow", "mentor", "meteor",
    "mineral", "monsoon", "mother", "nectar", "nimbus", "northern", "number", "ocean",
    "onward", "orbit", "orchid", "otter", "painter", "pattern", "pebble", "pigeon",
    "pioneer", "planet", "pollen", "quartz", "random", "rattle", "render", "ribbon",
    "river", "rooster", "sandal", "scatter", "shadow", "shelter", "silver", "simple",
    "sonnet", "spider", "spiral", "stellar", "stone", "summer", "sunset", "tandem",
    "tender", "thunder", "timber", "tinder", "together", "torrent", "tower", "tundra",
    "under", "velvet", "vendor", "verdant", "wander", "wanted", "warden", "weather",
    "willow", "winter", "wonder", "wooden", "yonder", "zenith",
)

PHRASE_WORD_COUNT = 7  # ~45 characters: enough keystrokes for stable timing
                       # statistics, short enough to stay unobtrusive.

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Challenge:
    nonce: str
    phrase: str
    username_claim: str
    purpose: str
    expires_at: float

    @property
    def expires_in(self) -> int:
        return max(0, int(round(self.expires_at - time.time())))


CAPITALISED_WORDS = 2  # see below


def generate_phrase(word_count: int = PHRASE_WORD_COUNT) -> str:
    """Build a random phrase using the CSPRNG, not the `random` module.

    random.choice is a Mersenne Twister: an observer who collects a handful of
    phrases could predict subsequent ones and pre-record matching behaviour.
    secrets.choice removes that path.

    A couple of words are capitalised at random positions. That is not
    decoration: it is the only place the user presses Shift in a field whose
    contents are public. Shift-hand preference (left vs right) is a strong,
    stable trait, and sourcing it here means we never have to infer it from
    password keystrokes, where knowing which positions were shifted would leak
    the password's capitalisation pattern.
    """
    words = [secrets.choice(_WORDS) for _ in range(word_count)]
    # Capitalise distinct positions, never the first word only, so the shift
    # presses land mid-phrase where they are a deliberate act rather than habit.
    positions = set()
    while len(positions) < min(CAPITALISED_WORDS, word_count):
        positions.add(secrets.randbelow(word_count))
    for i in positions:
        words[i] = words[i].capitalize()
    return " ".join(words)


def normalise_phrase(text: str) -> str:
    """Canonical form for comparison: lowercase, single-spaced, trimmed.

    Trailing whitespace and casing are typing artefacts, not evidence of an
    attack, so they must not cause a legitimate user to be blocked.
    """
    return _WS_RE.sub(" ", text.strip().lower())


def create_challenge(
    conn: sqlite3.Connection,
    username_claim: str,
    purpose: str,
    ttl_seconds: int | None = None,
) -> Challenge:
    """Issue a fresh challenge. 256 bits of entropy in the nonce."""
    nonce = secrets.token_urlsafe(32)
    phrase = generate_phrase()
    now = time.time()
    ttl = ttl_seconds if ttl_seconds is not None else settings.challenge_ttl_seconds
    expires_at = now + ttl

    conn.execute(
        "INSERT INTO auth_challenges "
        "(nonce, username_claim, purpose, phrase, issued_at, expires_at, consumed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, NULL)",
        (nonce, username_claim.lower(), purpose, phrase, now, expires_at),
    )
    return Challenge(
        nonce=nonce,
        phrase=phrase,
        username_claim=username_claim.lower(),
        purpose=purpose,
        expires_at=expires_at,
    )


def consume_challenge(
    conn: sqlite3.Connection,
    nonce: str,
    username_claim: str,
    purpose: str,
) -> tuple[Challenge | None, ReasonCode | None]:
    """Atomically burn a nonce and validate it.

    The burn happens *first*, before expiry or ownership is checked. That
    ordering is deliberate: if validation ran first and only burned the nonce
    on success, an attacker could resubmit the same nonce with tweaked event
    streams and use the decision as an oracle to tune their mimicry. Burning on
    first presentation means every attempt costs a fresh challenge.

    Returns (challenge, None) when valid, or (challenge_or_None, ReasonCode).
    """
    now = time.time()

    # Compare-and-swap: rowcount tells us whether we were the ones who burned it.
    burned = conn.execute(
        "UPDATE auth_challenges SET consumed_at = ? WHERE nonce = ? AND consumed_at IS NULL",
        (now, nonce),
    ).rowcount

    row = conn.execute(
        "SELECT nonce, username_claim, purpose, phrase, expires_at "
        "FROM auth_challenges WHERE nonce = ?",
        (nonce,),
    ).fetchone()

    if row is None:
        return None, ReasonCode.CHALLENGE_UNKNOWN

    challenge = Challenge(
        nonce=row["nonce"],
        phrase=row["phrase"],
        username_claim=row["username_claim"],
        purpose=row["purpose"],
        expires_at=row["expires_at"],
    )

    if burned == 0:
        return challenge, ReasonCode.CHALLENGE_REUSED
    if challenge.purpose != purpose or challenge.username_claim != username_claim.lower():
        return challenge, ReasonCode.CHALLENGE_WRONG_USER
    if challenge.expires_at <= now:
        return challenge, ReasonCode.CHALLENGE_EXPIRED

    return challenge, None


def purge_expired_challenges(conn: sqlite3.Connection, grace_seconds: int = 3600) -> int:
    """Drop long-dead challenges. Kept briefly past expiry so that a late
    submission still gets CHALLENGE_EXPIRED rather than a confusing UNKNOWN."""
    cutoff = time.time() - grace_seconds
    return conn.execute(
        "DELETE FROM auth_challenges WHERE expires_at <= ?", (cutoff,)
    ).rowcount
