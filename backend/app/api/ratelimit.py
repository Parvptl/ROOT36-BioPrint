"""In-process rate limiting.

A fixed-window counter per key, held in memory. Adequate for a single-process
local deployment, which is what this system is; a multi-worker or multi-host
deployment would need shared state, and that is recorded as a limitation
rather than pretended away.

What this mitigates: an attacker who has a password and wants to submit
hundreds of behavioural variations to find one the model accepts. Each attempt
already costs a fresh challenge, and this bounds the rate on top.

What it does not mitigate: a distributed attacker rotating source addresses, or
one patient enough to stay under the limit.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimit:
    max_events: int
    window_seconds: float


# Enough for a user retrying a few times after a genuine rejection, far short
# of what tuning an attack against the scorer would need.
LOGIN_LIMIT = RateLimit(max_events=12, window_seconds=60.0)
CHALLENGE_LIMIT = RateLimit(max_events=30, window_seconds=60.0)
REGISTER_LIMIT = RateLimit(max_events=5, window_seconds=300.0)
ENROLLMENT_LIMIT = RateLimit(max_events=40, window_seconds=300.0)


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._events: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, key: str, limit: RateLimit) -> tuple[bool, float]:
        """Record an attempt. Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        cutoff = now - limit.window_seconds

        with self._lock:
            events = [t for t in self._events[key] if t > cutoff]

            if len(events) >= limit.max_events:
                self._events[key] = events
                retry_after = max(0.0, events[0] + limit.window_seconds - now)
                return False, retry_after

            events.append(now)
            self._events[key] = events
            return True, 0.0

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._events.clear()
            else:
                self._events.pop(key, None)


limiter = SlidingWindowLimiter()
