"""Integrity checks on a submitted capture.

Answers a question the identity and automation models cannot: was this event
stream actually produced in response to *this* challenge, just now?

Three independent checks, each defeating a different replay strategy:

* Nonce validity, handled by the challenge module. Defeats resubmitting the
  same capture twice.

* Phrase match. The prompt is regenerated per attempt, so a capture recorded
  against an earlier prompt contains the wrong text. Defeats replaying an old
  recording against a fresh nonce.

* Capture duration against challenge age. A stream describing ninety seconds
  of typing cannot have been produced against a challenge issued four seconds
  ago. Defeats splicing a long recorded capture onto a freshly fetched nonce.

None of these defeats an attacker who records a user's *timing distribution*
and re-synthesises it onto new prompt text in real time. That case belongs to
the automation detector, and it is the weaker of the two defences.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.auth.challenge import Challenge, normalise_phrase
from app.behavioral.events import SessionView
from app.behavioral.scoring.reasons import ReasonCode

# How closely the typed phrase must match the issued one.
#
# Not an exact match: a genuine user makes typos, and blocking someone for a
# transposed letter would be an awful experience for no security gain. A replay
# of a different seven-word phrase scores far below this, because the words
# themselves differ rather than a character or two within them.
MIN_PHRASE_SIMILARITY = 0.80

# Allowance for network transit, clock skew between the two timestamps, and
# the gap between the challenge being issued and the collector starting.
CAPTURE_DURATION_SLACK_SECONDS = 5.0


@dataclass(frozen=True)
class IntegrityResult:
    ok: bool
    score: float  # 0.0 clean, 1.0 definitively violated
    reason: ReasonCode | None = None
    detail: str = ""


def check_integrity(
    challenge: Challenge,
    view: SessionView,
    phrase_typed: str,
    now: float | None = None,
) -> IntegrityResult:
    """Validate that this capture answers this challenge."""
    now = now if now is not None else time.time()

    similarity = phrase_similarity(challenge.phrase, phrase_typed)
    if similarity < MIN_PHRASE_SIMILARITY:
        return IntegrityResult(
            ok=False,
            # Graded rather than binary: a near-miss reads as a near-miss, and
            # a completely different phrase reads as a replay.
            score=min(1.0, 1.0 - similarity),
            reason=ReasonCode.PHRASE_MISMATCH,
            detail=f"typed phrase matched the issued phrase {similarity:.0%}",
        )

    challenge_age = challenge.age_seconds(now)
    capture_seconds = view.duration_ms / 1000.0
    if capture_seconds > challenge_age + CAPTURE_DURATION_SLACK_SECONDS:
        return IntegrityResult(
            ok=False,
            score=1.0,
            reason=ReasonCode.TIMESTAMP_INCONSISTENT,
            detail=(
                f"capture describes {capture_seconds:.1f}s of interaction but the "
                f"challenge is only {challenge_age:.1f}s old"
            ),
        )

    if view.total_events < 1:
        return IntegrityResult(
            ok=False,
            score=1.0,
            reason=ReasonCode.MALFORMED_EVENT_STREAM,
            detail="no interaction events were captured",
        )

    return IntegrityResult(ok=True, score=0.0)


def phrase_similarity(expected: str, actual: str) -> float:
    """Similarity in [0, 1] between the issued phrase and what was typed.

    Case and whitespace are normalised away first: capitalisation is there to
    make the user press Shift, not to be graded.
    """
    left = normalise_phrase(expected)
    right = normalise_phrase(actual)
    if not left:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()
