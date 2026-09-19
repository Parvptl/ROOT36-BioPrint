"""Automation detection.

A separate component from identity, producing its own score and its own reason
codes. The distinction matters for the product: "this is not you" and "this is
not a person" are different findings, and a user deserves to be told which one
happened.

Deliberately not built on navigator.webdriver. That flag is one line to remove,
and input driven through the DevTools protocol arrives with isTrusted true, so
a detector that leans on either is trivially defeated. Both are included as
low-weight corroboration and nothing more.

Everything here keys off properties of the interaction that an attacker has to
actually reproduce: that key presses have varied hold times, that typing rhythm
is irregular, that a pointer path wanders and trembles, that events arrive in a
physically possible order.

Limitation, stated plainly and repeated in the report: this is browser-side
detection against an adversary who controls the browser. It raises the cost of
scripted login attempts from trivial to substantial. It does not defeat a
determined attacker who drives a real browser and samples timings from a
distribution of real human behaviour. The honest claim is cost, not
impossibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.behavioral import stats
from app.behavioral.events import SessionView
from app.behavioral.scoring.reasons import ReasonCode
from app.models.events import BehaviorSessionIn

# --- thresholds, all in the units of the thing they bound -------------------

# Below this robust coefficient of variation, typing rhythm is more regular
# than a human hand produces. Humans typically sit well above it.
HUMAN_IKI_CV_FLOOR = 0.18

# Hold times vary between presses for a human. A script usually emits one
# constant, or a small set of them.
HUMAN_DWELL_CV_FLOOR = 0.10

# Sustained typing above this is beyond human motor limits; competitive typists
# peak around 20 characters per second in short bursts.
MAX_HUMAN_CHARS_PER_SECOND = 22.0

# Distinct characters closer together than this are not two separate presses.
IMPLAUSIBLE_IKI_MS = 12.0

# A path this straight was interpolated, not moved by an arm.
SYNTHETIC_STRAIGHTNESS = 1.004

# Two clicks further apart than this with no pointer movement between them
# means the pointer teleported.
TELEPORT_DISTANCE_PX = 150.0

# Minimum evidence before a signal is allowed to fire. Every detector here is
# capable of false positives on three data points.
MIN_TIMING_SAMPLES = 12
MIN_POINTER_SAMPLES = 25

# Signals weaker than this are not shown to the user; they are noise.
REPORTING_FLOOR = 0.15


@dataclass(frozen=True)
class AutomationSignal:
    code: ReasonCode
    strength: float  # 0..1, how strongly this signal fired
    weight: float    # 0..1, how much this signal is trusted
    detail: str


@dataclass(frozen=True)
class AutomationResult:
    score: float = 0.0
    signals: list[AutomationSignal] = field(default_factory=list)
    evaluated: bool = True

    @property
    def top_code(self) -> ReasonCode | None:
        if not self.signals:
            return None
        return max(self.signals, key=lambda s: s.strength * s.weight).code


def _ramp(value: float, start: float, full: float) -> float:
    """Linear 0..1 ramp. Returns 0 at `start` and 1 at `full`, either direction."""
    if start == full:
        return 0.0
    return stats.clamp((value - start) / (full - start))


def detect_automation(
    view: SessionView, session: BehaviorSessionIn
) -> AutomationResult:
    """Score how machine-like this interaction looks."""
    signals: list[AutomationSignal] = []

    _event_order(view, signals)
    _dwell_degeneracy(view, signals)
    _timing_variability(view, signals)
    _timing_quantisation(view, signals)
    _pointer_geometry(view, signals)
    _interaction_speed(view, signals)
    _trust_flags(view, session, signals)

    # Noisy-OR. Independent tells compound rather than averaging out: one
    # near-conclusive signal is enough, and several weak ones corroborate.
    # Averaging would let an attacker dilute a fatal tell by doing everything
    # else convincingly.
    survival = 1.0
    for signal in signals:
        survival *= 1.0 - stats.clamp(signal.strength * signal.weight)

    return AutomationResult(
        score=stats.clamp(1.0 - survival),
        signals=[s for s in signals if s.strength >= REPORTING_FLOOR],
    )


def _event_order(view: SessionView, out: list[AutomationSignal]) -> None:
    """Physically impossible or script-shaped event sequences.

    The strongest single tell available here is an `input` event with no
    keydown before it in the same field: that is what assigning to
    element.value produces, and it is the cheapest scripted fill there is.
    """
    findings: list[str] = []
    strength = 0.0

    # Earliest key press per field, so "before" is measured per field rather
    # than globally: typing in the username box says nothing about how the
    # password box got filled.
    first_keydown_t: dict[str, float] = {}
    for press in view.presses:
        prior = first_keydown_t.get(press.ctx)
        if prior is None or press.down_t < prior:
            first_keydown_t[press.ctx] = press.down_t

    # A paste legitimately changes a field with no key press behind it, so
    # pasted fields are excluded rather than flagged. Pasting a password from a
    # manager is normal behaviour, not an attack.
    orphan_inputs = sum(
        1
        for event in view.input_events
        if event.ctx not in view.paste_contexts
        and (
            event.ctx not in first_keydown_t
            or event.t < first_keydown_t[event.ctx]
        )
    )

    if view.input_events and orphan_inputs:
        share = orphan_inputs / len(view.input_events)
        strength = max(strength, share)
        findings.append(
            f"{orphan_inputs} of {len(view.input_events)} field updates arrived "
            f"with no key press before them"
        )

    matched_keyups = sum(1 for p in view.presses if p.up_t is not None)
    total_keyups = matched_keyups + view.unmatched_keyups
    if view.unmatched_keyups and total_keyups:
        share = view.unmatched_keyups / total_keyups
        if share > 0.1:
            strength = max(strength, share)
            findings.append(
                f"{view.unmatched_keyups} key releases had no matching press"
            )

    # Pointer teleportation between clicks.
    teleports, click_pairs = _teleport_count(view)
    if click_pairs and teleports:
        share = teleports / click_pairs
        strength = max(strength, share * 0.8)
        findings.append(f"pointer jumped between {teleports} click locations without moving")

    if strength > 0:
        out.append(
            AutomationSignal(
                code=ReasonCode.SYNTHETIC_EVENT_ORDER,
                strength=stats.clamp(strength),
                weight=0.95,
                detail="; ".join(findings),
            )
        )


def _teleport_count(view: SessionView) -> tuple[int, int]:
    downs = [e for e in view.click_events if e.type == "pointerdown"]
    if len(downs) < 2:
        return 0, 0

    teleports = 0
    for previous, current in zip(downs[:-1], downs[1:]):
        distance = float(np.hypot(current.x - previous.x, current.y - previous.y))
        if distance < TELEPORT_DISTANCE_PX:
            continue
        moved_between = np.any(
            (view.pointer_t > previous.t) & (view.pointer_t < current.t)
        )
        if not moved_between:
            teleports += 1

    return teleports, len(downs) - 1


def _dwell_values(view: SessionView) -> np.ndarray:
    return np.array(
        [d for p in view.presses if not p.repeat and (d := p.dwell) is not None],
        dtype=float,
    )


def _dwell_degeneracy(view: SessionView, out: list[AutomationSignal]) -> None:
    """Hold times that repeat exactly, or barely vary.

    Browser timestamps carry sub-millisecond resolution, so two human presses
    essentially never produce the same dwell. A script that holds every key for
    a fixed number of milliseconds produces almost nothing else.
    """
    dwells = _dwell_values(view)
    if dwells.size < MIN_TIMING_SAMPLES:
        return

    unique_share = len(np.unique(np.round(dwells, 3))) / dwells.size
    repetition = _ramp(unique_share, 0.85, 0.25)

    cv = stats.robust_cv(dwells)
    flatness = _ramp(cv, HUMAN_DWELL_CV_FLOOR, 0.0)

    strength = max(repetition, flatness)
    if strength <= 0:
        return

    out.append(
        AutomationSignal(
            code=ReasonCode.DWELL_DEGENERACY,
            strength=stats.clamp(strength),
            weight=0.85,
            detail=(
                f"{unique_share:.0%} of key hold times were distinct "
                f"(variation {cv:.3f})"
            ),
        )
    )


def _inter_key_intervals(view: SessionView) -> np.ndarray:
    chars = sorted(
        (p for p in view.presses if p.key_class == "char" and not p.repeat),
        key=lambda p: p.down_t,
    )
    if len(chars) < 2:
        return np.empty(0)
    return np.diff(np.array([p.down_t for p in chars], dtype=float))


def _timing_variability(view: SessionView, out: list[AutomationSignal]) -> None:
    """Rhythm too regular to be a hand."""
    ikis = _inter_key_intervals(view)
    if ikis.size < MIN_TIMING_SAMPLES:
        return

    cv = stats.robust_cv(ikis)
    strength = _ramp(cv, HUMAN_IKI_CV_FLOOR, 0.0)
    if strength <= 0:
        return

    out.append(
        AutomationSignal(
            code=ReasonCode.LOW_TIMING_VARIABILITY,
            strength=stats.clamp(strength),
            weight=0.85,
            detail=f"typing rhythm varied by {cv:.3f}, below the human range",
        )
    )


def _timing_quantisation(view: SessionView, out: list[AutomationSignal]) -> None:
    """Intervals sitting on a fixed lattice.

    Catches the case that survives the variability test: a script that
    randomises its delays but rounds them, or schedules on a timer tick. For
    each candidate quantum we measure how far the intervals sit from the
    nearest multiple; a lattice gives near-zero residual, arbitrary human
    timing gives roughly a quarter of the quantum.

    Guarded hard on sample count. With few intervals some quantum always fits,
    which is exactly how a detector like this produces false positives.
    """
    ikis = _inter_key_intervals(view)
    if ikis.size < MIN_TIMING_SAMPLES * 2:
        return

    usable = ikis[(ikis > 1.0) & (ikis < 2000.0)]
    if usable.size < MIN_TIMING_SAMPLES * 2:
        return

    best_residual = 1.0
    best_quantum = 0.0
    for quantum in np.arange(4.0, 60.0, 0.5):
        # Distance to the nearest lattice point, normalised so that uniformly
        # distributed phase scores 0.5.
        phase = np.abs(usable / quantum - np.round(usable / quantum))
        residual = float(np.mean(phase)) / 0.5
        if residual < best_residual:
            best_residual, best_quantum = residual, quantum

    strength = _ramp(best_residual, 0.35, 0.05)
    if strength <= 0:
        return

    out.append(
        AutomationSignal(
            code=ReasonCode.AUTOMATION_TIMING,
            strength=stats.clamp(strength),
            weight=0.8,
            detail=f"key intervals aligned to a {best_quantum:.1f} ms grid",
        )
    )


def _pointer_geometry(view: SessionView, out: list[AutomationSignal]) -> None:
    """Movement that no arm produced.

    No pointer activity at all is NOT a signal here. Plenty of people fill a
    login form entirely from the keyboard, and penalising them would reject
    genuine users for being efficient.
    """
    if not view.segments or view.pointer_t.size < MIN_POINTER_SAMPLES:
        return

    straightness: list[float] = []
    tremors: list[float] = []
    for segment in view.segments:
        if segment.displacement > 1.0:
            straightness.append(segment.path_length / segment.displacement)
        if segment.x.size >= 3:
            tremors.append(
                float(np.median(np.hypot(np.diff(segment.x, 2), np.diff(segment.y, 2))))
            )

    findings: list[str] = []
    strength = 0.0

    if straightness:
        value = float(np.median(straightness))
        ruler = _ramp(value, SYNTHETIC_STRAIGHTNESS, 1.0)
        if ruler > 0:
            findings.append(f"pointer path was {value:.4f} times the direct distance")
            strength = max(strength, ruler)

    if tremors:
        value = float(np.median(tremors))
        still = _ramp(value, 0.6, 0.0)
        if still > 0:
            findings.append(f"pointer showed almost no fine movement ({value:.3f} px)")
            strength = max(strength, still)

    # Perfectly even sampling: a real device reports at jittery intervals.
    dts = np.diff(view.pointer_t)
    dts = dts[dts > 0]
    if dts.size >= MIN_POINTER_SAMPLES:
        cv = stats.robust_cv(dts)
        metronome = _ramp(cv, 0.06, 0.0)
        if metronome > 0:
            findings.append("pointer samples arrived at perfectly even intervals")
            strength = max(strength, metronome)

    if strength <= 0:
        return

    out.append(
        AutomationSignal(
            code=ReasonCode.POINTER_ANOMALY,
            strength=stats.clamp(strength),
            weight=0.6,
            detail="; ".join(findings),
        )
    )


def _interaction_speed(view: SessionView, out: list[AutomationSignal]) -> None:
    """Faster than a person can physically be."""
    findings: list[str] = []
    strength = 0.0

    chars = [p for p in view.presses if p.key_class == "char" and not p.repeat]
    if len(chars) >= MIN_TIMING_SAMPLES:
        span = max(p.down_t for p in chars) - min(p.down_t for p in chars)
        if span > 0:
            rate = len(chars) / (span / 1000.0)
            fast = _ramp(rate, MAX_HUMAN_CHARS_PER_SECOND, MAX_HUMAN_CHARS_PER_SECOND * 2.5)
            if fast > 0:
                findings.append(f"typed at {rate:.1f} characters per second")
                strength = max(strength, fast)

    ikis = _inter_key_intervals(view)
    if ikis.size >= MIN_TIMING_SAMPLES:
        share = stats.fraction(ikis < IMPLAUSIBLE_IKI_MS)
        impossible = _ramp(share, 0.1, 0.5)
        if impossible > 0:
            findings.append(f"{share:.0%} of key presses were under {IMPLAUSIBLE_IKI_MS:.0f} ms apart")
            strength = max(strength, impossible)

    if strength <= 0:
        return

    out.append(
        AutomationSignal(
            code=ReasonCode.IMPOSSIBLE_INTERACTION_SPEED,
            strength=stats.clamp(strength),
            weight=0.9,
            detail="; ".join(findings),
        )
    )


def _trust_flags(
    view: SessionView, session: BehaviorSessionIn, out: list[AutomationSignal]
) -> None:
    """Self-declared automation markers.

    Kept at low weight on purpose. navigator.webdriver is one line to delete,
    and input driven through the DevTools protocol arrives with isTrusted set,
    so neither can carry a decision. They corroborate; they never decide.
    """
    findings: list[str] = []
    strength = 0.0

    if view.total_events:
        untrusted_share = view.untrusted_count / view.total_events
        if untrusted_share > 0.05:
            findings.append(f"{untrusted_share:.0%} of events were not browser-generated")
            strength = max(strength, untrusted_share)

    if session.meta.webdriver:
        findings.append("the browser reported an automation driver")
        strength = max(strength, 0.5)

    if strength <= 0:
        return

    out.append(
        AutomationSignal(
            code=ReasonCode.UNTRUSTED_EVENTS,
            strength=stats.clamp(strength),
            weight=0.35,
            detail="; ".join(findings),
        )
    )
