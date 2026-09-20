#!/usr/bin/env python3
"""Does authentication latency track CPU contention?

Argon2id is deliberately memory-hard: 64 MiB, t=2, p=2. That cost is the
point — it is what makes a leaked user table expensive to crack. But a
memory-hard function is also unusually sensitive to competition for memory
bandwidth and cores, so the same verify can take 90 ms on a quiet machine and
half a second on a busy one.

This measures that directly, because the alternative explanations (event
volume, profile maturity) were tested first and ruled out.

Nothing here changes a security parameter. It adds load and measures.

    cd backend
    ./.venv/Scripts/python.exe evaluation/latency_under_load.py
"""

from __future__ import annotations

import multiprocessing as mp
import statistics as st
import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def burn(stop):
    """Occupy a core with memory traffic, not just arithmetic.

    A pure integer spin would compete for ALU time; Argon2 competes for memory
    bandwidth, so the load has to touch memory to be representative.
    """
    buf = bytearray(8 * 1024 * 1024)
    i = 0
    while not stop.is_set():
        i = (i + 4096) % (len(buf) - 8)
        buf[i] = (buf[i] + 1) & 0xFF
        _ = sum(buf[i:i + 512])


def measure(reps: int = 20) -> list[float]:
    from app.auth.passwords import hash_password, verify_password

    h = hash_password("a-test-password-123")
    for _ in range(3):
        verify_password("a-test-password-123", h)

    times = []
    for _ in range(reps):
        t = time.perf_counter()
        verify_password("a-test-password-123", h)
        times.append((time.perf_counter() - t) * 1000.0)
    return times


def summarise(label: str, times: list[float]) -> dict:
    ordered = sorted(times)
    row = {
        "label": label,
        "min": ordered[0],
        "p50": st.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }
    print(f"  {label:<26s} min {row['min']:7.1f}  p50 {row['p50']:7.1f}  "
          f"p95 {row['p95']:7.1f}  max {row['max']:7.1f}")
    return row


def main() -> None:
    print("ARGON2ID VERIFY LATENCY vs CPU CONTENTION")
    print("  Argon2id m=64MiB t=2 p=2 — parameters UNCHANGED, only load varies")
    print()
    print("  " + "-" * 74)

    baseline = summarise("as-is (current load)", measure())

    rows = [baseline]
    for workers in (4, 8):
        stop = mp.Event()
        procs = [mp.Process(target=burn, args=(stop,), daemon=True)
                 for _ in range(workers)]
        for p in procs:
            p.start()
        time.sleep(1.5)  # let the load settle
        try:
            rows.append(summarise(f"+{workers} busy cores", measure()))
        finally:
            stop.set()
            for p in procs:
                p.join(timeout=3)
                if p.is_alive():
                    p.terminate()
        time.sleep(1.0)

    print("  " + "-" * 74)
    print()
    worst = max(r["p95"] for r in rows)
    best = min(r["min"] for r in rows)
    print(f"  Same work, same parameters: {best:.0f} ms to {worst:.0f} ms "
          f"({worst / best:.1f}x) purely from contention.")
    print()
    print("  Argon2id is ~80% of the login's measured time, so the end-to-end")
    print("  number inherits this spread directly. A latency figure quoted")
    print("  without the machine state it was measured on is not a figure.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
