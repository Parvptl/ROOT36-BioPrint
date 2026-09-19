"""The feature registry.

Every behavioural feature the system uses is declared here exactly once, with
what it measures, how it is computed, how much evidence it needs before it may
be trusted, and whether it is switched on.

Two rules this file exists to enforce:

1. A feature with too few observations is ABSENT, never zero. Substituting
   zero for "not observed" is the single most common way these systems break:
   a user who happened not to use the mouse gets a pointer-velocity of 0, which
   is a wild deviation from their profile, and a genuine login is rejected for
   a reason nobody can explain. Coverage is tracked instead.

2. Feature selection is a deliberate, reviewable list rather than whatever the
   extractor happened to emit. Adding a feature because it sounds impressive
   costs reliability, because every extra dimension is another chance for a
   genuine user to look unusual on some axis.

3. Every feature declares a noise floor: the smallest change that means
   different behaviour rather than the sampling resolution of a single
   session. This exists because of a measured failure. `kbd_backspace_rate`
   over a fifty-key phrase moves in steps of 0.02, one backspace. Five
   enrollment rounds produced 0.021, 0.040, 0.040, 0.021, 0.039, whose median
   absolute deviation is 0.0008, which floored the scale at 0.004. A genuine
   login with one more backspace then scored nine standard deviations out and
   saturated, purely from counting noise. Without a floor, any feature whose
   per-session value is quantised coarsely will do the same.

   The floors below are derived from how many observations one session
   actually supplies, not tuned to make tests pass.

To disable a feature, set enabled=False here, or list it in the
BIOPRINT_DISABLED_FEATURES environment variable for a temporary experiment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum


class Modality(StrEnum):
    KEYBOARD = "keyboard"
    POINTER = "pointer"
    INTERACTION = "interaction"


class Usefulness(StrEnum):
    """Prior expectation, recorded so it can be checked against measurement.

    These are hypotheses at the time of writing, not results. The evaluation
    harness reports the discriminability weights actually learned, which is
    what should be believed.
    """

    HIGH = "high"
    MEDIUM = "medium"
    EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    modality: Modality
    description: str
    calculation: str
    min_observations: int
    usefulness: Usefulness
    # The smallest change in this feature that represents different behaviour
    # rather than the sampling resolution of one session. See noise_floor_rationale.
    noise_floor: float = 0.0
    noise_floor_rationale: str = ""
    enabled: bool = True


# Identity keystroke features are computed over the challenge-phrase context
# only. The phrase is different every attempt, so these describe how the person
# types rather than what they typed, and they survive a password change.
_KEYBOARD: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="kbd_dwell_median",
        modality=Modality.KEYBOARD,
        description="Typical time a key is held down.",
        calculation="median(keyup_t - keydown_t) over paired phrase-context presses",
        min_observations=12,
        usefulness=Usefulness.HIGH,
        noise_floor=3.0,
        noise_floor_rationale="ms; below this is finger-to-finger variation within one hand, not a different hand",
    ),
    FeatureSpec(
        name="kbd_dwell_mad",
        modality=Modality.KEYBOARD,
        description="How consistent the hold time is. Consistency is itself personal.",
        calculation="median absolute deviation of the same dwell series",
        min_observations=12,
        usefulness=Usefulness.HIGH,
        noise_floor=3.0,
        noise_floor_rationale="ms; same resolution argument as the dwell median it is derived from",
    ),
    FeatureSpec(
        name="kbd_dwell_left_median",
        modality=Modality.KEYBOARD,
        description="Hold time on left-hand keys.",
        calculation="median dwell restricted to left-hand QWERTY codes",
        min_observations=6,
        usefulness=Usefulness.MEDIUM,
        noise_floor=4.5,
        noise_floor_rationale="ms; drawn from roughly half the presses, so noisier than the overall median",
    ),
    FeatureSpec(
        name="kbd_dwell_right_median",
        modality=Modality.KEYBOARD,
        description="Hold time on right-hand keys. The left/right asymmetry differs per person.",
        calculation="median dwell restricted to right-hand QWERTY codes",
        min_observations=6,
        usefulness=Usefulness.MEDIUM,
        noise_floor=4.5,
        noise_floor_rationale="ms; as above",
    ),
    FeatureSpec(
        name="kbd_flight_median",
        modality=Modality.KEYBOARD,
        description="Typical gap between releasing one key and pressing the next.",
        calculation="median(keydown_t[i+1] - keyup_t[i]) over consecutive character presses",
        min_observations=10,
        usefulness=Usefulness.HIGH,
        noise_floor=5.0,
        noise_floor_rationale="ms; transition timing jitters by a few ms between identical transitions",
    ),
    FeatureSpec(
        name="kbd_flight_iqr",
        modality=Modality.KEYBOARD,
        description="Spread of those gaps.",
        calculation="interquartile range of the flight series",
        min_observations=10,
        usefulness=Usefulness.MEDIUM,
        noise_floor=9.0,
        noise_floor_rationale="ms; a spread statistic over ~45 samples carries more sampling error than its centre",
    ),
    FeatureSpec(
        name="kbd_flight_negative_frac",
        modality=Modality.KEYBOARD,
        description=(
            "Proportion of transitions where the next key goes down before the previous "
            "comes up. Touch typists overlap constantly; hunt-and-peck typists never do."
        ),
        calculation="fraction of flight values below zero",
        min_observations=10,
        usefulness=Usefulness.HIGH,
        noise_floor=0.05,
        noise_floor_rationale="one or two transitions out of roughly forty-five",
    ),
    FeatureSpec(
        name="kbd_logiki_median",
        modality=Modality.KEYBOARD,
        description="Typing tempo, on a log scale because intervals are near log-normal.",
        calculation="median(log(keydown_t[i+1] - keydown_t[i]))",
        min_observations=10,
        usefulness=Usefulness.HIGH,
        noise_floor=0.05,
        noise_floor_rationale="log ms; about a five percent change in tempo",
    ),
    FeatureSpec(
        name="kbd_logiki_mad",
        modality=Modality.KEYBOARD,
        description="Rhythm irregularity. Also the main human-versus-machine separator.",
        calculation="median absolute deviation of the log inter-key series",
        min_observations=10,
        usefulness=Usefulness.HIGH,
        noise_floor=0.05,
        noise_floor_rationale="log ms; as above",
    ),
    FeatureSpec(
        name="kbd_burst_fraction",
        modality=Modality.KEYBOARD,
        description="Share of keys typed in rapid bursts, as in a familiar letter run.",
        calculation="fraction of inter-key intervals under 80 ms",
        min_observations=15,
        usefulness=Usefulness.MEDIUM,
        noise_floor=0.05,
        noise_floor_rationale="two bursts out of roughly forty-five intervals",
    ),
    FeatureSpec(
        name="kbd_pause_rate",
        modality=Modality.KEYBOARD,
        description="How often typing stalls, per 100 keys.",
        calculation="count of inter-key intervals over 500 ms, scaled per 100 presses",
        min_observations=15,
        usefulness=Usefulness.MEDIUM,
        noise_floor=3.0,
        noise_floor_rationale="per 100 keys; about 1.5 pauses in a fifty-key phrase, the counting step",
    ),
    FeatureSpec(
        name="kbd_backspace_rate",
        modality=Modality.KEYBOARD,
        description="Correction frequency.",
        calculation="backspace/delete presses divided by total presses",
        min_observations=15,
        usefulness=Usefulness.MEDIUM,
        noise_floor=0.025,
        noise_floor_rationale="one backspace in a fifty-key phrase IS 0.02, so anything finer is counting noise",
    ),
    FeatureSpec(
        name="kbd_backspace_latency_median",
        modality=Modality.KEYBOARD,
        description=(
            "How quickly a mistake is noticed. Independent of what was typed, and "
            "surprisingly stable within a person."
        ),
        calculation="median gap from the preceding keydown to each backspace keydown",
        min_observations=2,
        usefulness=Usefulness.MEDIUM,
        noise_floor=45.0,
        noise_floor_rationale="ms; a median over one to three samples is barely a median",
    ),
    FeatureSpec(
        name="kbd_speed_kps",
        modality=Modality.KEYBOARD,
        description="Characters per second across the phrase.",
        calculation="character press count divided by elapsed phrase typing time",
        min_observations=15,
        usefulness=Usefulness.HIGH,
        noise_floor=0.3,
        noise_floor_rationale="characters per second; within-session tempo drift on an unfamiliar word",
    ),
    FeatureSpec(
        name="kbd_speed_cv",
        modality=Modality.KEYBOARD,
        description="Whether tempo holds steady or surges. Consistency is as personal as speed.",
        calculation="robust coefficient of variation of per-5-key windowed rates",
        min_observations=20,
        usefulness=Usefulness.MEDIUM,
        noise_floor=0.08,
        noise_floor_rationale="dimensionless; a ratio of two noisy spread estimates",
    ),
    FeatureSpec(
        name="kbd_shift_right_ratio",
        modality=Modality.KEYBOARD,
        description=(
            "Which Shift key the person reaches for. Near-binary, extremely stable, "
            "and measured only on the public challenge phrase."
        ),
        calculation="right-shift presses divided by all shift presses in phrase context",
        min_observations=2,
        usefulness=Usefulness.HIGH,
        noise_floor=0.2,
        noise_floor_rationale="one shift press out of the four the phrase forces",
    ),
)

_POINTER: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="ptr_velocity_median",
        modality=Modality.POINTER,
        description="Typical pointer speed while travelling.",
        calculation="median of per-sample speed across movement segments, px/ms",
        min_observations=20,
        usefulness=Usefulness.HIGH,
        noise_floor=0.05,
        noise_floor_rationale="px/ms; the same hand varies this much between two reaches",
    ),
    FeatureSpec(
        name="ptr_velocity_p90",
        modality=Modality.POINTER,
        description="Peak cruising speed, which separates flickers from gliders.",
        calculation="90th percentile of the same speed series",
        min_observations=20,
        usefulness=Usefulness.MEDIUM,
        noise_floor=0.12,
        noise_floor_rationale="px/ms; a tail statistic, so noisier than the median",
    ),
    FeatureSpec(
        name="ptr_accel_peak",
        modality=Modality.POINTER,
        description="How hard the movement is launched.",
        calculation="90th percentile of absolute speed change between samples",
        min_observations=20,
        usefulness=Usefulness.MEDIUM,
        noise_floor=0.01,
        noise_floor_rationale="px/ms^2; a 90th percentile over a few hundred samples",
    ),
    FeatureSpec(
        name="ptr_straightness",
        modality=Modality.POINTER,
        description=(
            "Path length over straight-line distance. Exactly 1.0 means a perfectly "
            "straight line, which human arms do not produce."
        ),
        calculation="mean of path_length/displacement across segments",
        min_observations=2,
        usefulness=Usefulness.HIGH,
        noise_floor=0.02,
        noise_floor_rationale="ratio; path curvature differs this much between two reaches to the same target",
    ),
    FeatureSpec(
        name="ptr_reversal_rate",
        modality=Modality.POINTER,
        description="Direction changes per movement: overshoot and correction habits.",
        calculation="sign changes in the movement direction, per segment",
        min_observations=2,
        usefulness=Usefulness.MEDIUM,
        noise_floor=0.06,
        noise_floor_rationale="one reversal in a movement, the counting step for this feature",
    ),
    FeatureSpec(
        name="ptr_tremor",
        modality=Modality.POINTER,
        description="High-frequency wobble from fine motor control.",
        calculation="median magnitude of the second difference of position",
        min_observations=20,
        usefulness=Usefulness.MEDIUM,
        noise_floor=0.45,
        noise_floor_rationale="px; second-difference magnitude is dominated by pointer quantisation below this",
    ),
    FeatureSpec(
        name="ptr_click_dwell_median",
        modality=Modality.POINTER,
        description="How long the button is held on a click.",
        calculation="median(pointerup_t - pointerdown_t) over matched clicks",
        min_observations=2,
        usefulness=Usefulness.MEDIUM,
        noise_floor=12.0,
        noise_floor_rationale="ms; a median over two or three clicks",
    ),
)

_INTERACTION: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="int_time_to_first_interaction",
        modality=Modality.INTERACTION,
        description="Pause before touching the form at all, on a log scale.",
        calculation="log(first input-bearing event time), milliseconds",
        min_observations=1,
        usefulness=Usefulness.MEDIUM,
        noise_floor=0.4,
        noise_floor_rationale="log ms; a SINGLE observation with no averaging, so the floor must be wide",
    ),
    FeatureSpec(
        name="int_tab_transition_ratio",
        modality=Modality.INTERACTION,
        description=(
            "Keyboard versus pointer navigation between fields. A settled habit that "
            "varies little within a person and a lot between people."
        ),
        calculation="tab-attributed focus events divided by all user-driven focus events",
        min_observations=2,
        usefulness=Usefulness.HIGH,
        noise_floor=0.2,
        noise_floor_rationale="one transition out of the four the form provides",
    ),
    FeatureSpec(
        name="int_focus_to_first_key_median",
        modality=Modality.INTERACTION,
        description="Delay between landing in a field and starting to type.",
        calculation="median gap from each focus event to the next keydown in that field",
        min_observations=2,
        usefulness=Usefulness.MEDIUM,
        noise_floor=65.0,
        noise_floor_rationale="ms; a median over about three reaction times",
    ),
    FeatureSpec(
        name="int_presubmit_hesitation",
        modality=Modality.INTERACTION,
        description="Pause between finishing typing and committing, on a log scale.",
        calculation="log(submit time - last keydown time), milliseconds",
        min_observations=1,
        usefulness=Usefulness.MEDIUM,
        noise_floor=0.35,
        noise_floor_rationale="log ms; a SINGLE observation, and hesitation before committing is highly variable",
    ),
)

REGISTRY: tuple[FeatureSpec, ...] = _KEYBOARD + _POINTER + _INTERACTION

SPECS: dict[str, FeatureSpec] = {spec.name: spec for spec in REGISTRY}


def _env_disabled() -> frozenset[str]:
    raw = os.getenv("BIOPRINT_DISABLED_FEATURES", "")
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


_DISABLED = _env_disabled()


def is_enabled(name: str) -> bool:
    spec = SPECS.get(name)
    return bool(spec and spec.enabled and name not in _DISABLED)


def enabled_specs() -> tuple[FeatureSpec, ...]:
    return tuple(spec for spec in REGISTRY if is_enabled(spec.name))


def enabled_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in enabled_specs())


def specs_for(modality: Modality) -> tuple[FeatureSpec, ...]:
    return tuple(spec for spec in enabled_specs() if spec.modality == modality)
