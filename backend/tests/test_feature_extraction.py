"""Feature extraction correctness.

These tests check that each feature measures what its registry entry claims,
using synthetic streams with known properties. They say nothing about how well
the features separate real people; that is what the evaluation harness on real
captures is for.
"""

from __future__ import annotations

import pytest

from app.behavioral.events import build_session_view
from app.behavioral.features.extractor import extract_features, extract_from_session
from app.behavioral.features.registry import REGISTRY, SPECS, Modality, enabled_names
from tests.factories import (
    TypingStyle,
    human_session,
    linear_pointer_session,
    scripted_session,
    value_injection_session,
)

PHRASE = "amber Willow granite pattern Copper thunder meadow"


# ------------------------------------------------------------------ registry


def test_registry_entries_are_well_formed():
    for spec in REGISTRY:
        assert spec.name == spec.name.lower()
        assert spec.description.strip()
        assert spec.calculation.strip()
        assert spec.min_observations >= 1


def test_registry_names_are_unique():
    names = [spec.name for spec in REGISTRY]
    assert len(names) == len(set(names))


def test_feature_names_are_prefixed_by_modality():
    prefixes = {
        Modality.KEYBOARD: "kbd_",
        Modality.POINTER: "ptr_",
        Modality.INTERACTION: "int_",
    }
    for spec in REGISTRY:
        assert spec.name.startswith(prefixes[spec.modality])


# ------------------------------------------------------------- happy path


def test_human_session_yields_broad_coverage():
    _, features = extract_from_session(human_session(PHRASE, seed=7))

    assert features.coverage > 0.85, f"missing: {sorted(features.underpowered)}"
    assert features.coverage_by_modality["keyboard"] > 0.85
    assert features.coverage_by_modality["pointer"] > 0.5
    assert features.coverage_by_modality["interaction"] > 0.5


def test_every_extracted_feature_is_declared_in_the_registry():
    _, features = extract_from_session(human_session(PHRASE, seed=3))
    assert set(features.values) <= set(enabled_names())


def test_all_values_are_finite():
    _, features = extract_from_session(human_session(PHRASE, seed=11))
    for name, value in features.values.items():
        assert value == value, name          # not NaN
        assert abs(value) < float("inf"), name


# ---------------------------------------------------------- what they measure


def test_dwell_median_tracks_configured_hold_time():
    slow = TypingStyle(dwell_mean_ms=140.0, dwell_jitter=0.05, overlap_prob=0.0)
    quick = TypingStyle(dwell_mean_ms=55.0, dwell_jitter=0.05, overlap_prob=0.0)

    _, a = extract_from_session(human_session(PHRASE, style=slow, seed=1))
    _, b = extract_from_session(human_session(PHRASE, style=quick, seed=1))

    assert a.values["kbd_dwell_median"] > b.values["kbd_dwell_median"]
    assert a.values["kbd_dwell_median"] == pytest.approx(140.0, rel=0.25)


def test_typing_speed_tracks_configured_interval():
    fast = TypingStyle(iki_mean_ms=90.0, pause_prob=0.0, backspace_prob=0.0)
    slow = TypingStyle(iki_mean_ms=320.0, pause_prob=0.0, backspace_prob=0.0)

    _, a = extract_from_session(human_session(PHRASE, style=fast, seed=2))
    _, b = extract_from_session(human_session(PHRASE, style=slow, seed=2))

    assert a.values["kbd_speed_kps"] > b.values["kbd_speed_kps"]
    assert a.values["kbd_logiki_median"] < b.values["kbd_logiki_median"]


def test_overlap_probability_drives_negative_flight_fraction():
    fluent = TypingStyle(overlap_prob=0.85, dwell_mean_ms=110.0, iki_mean_ms=120.0)
    deliberate = TypingStyle(overlap_prob=0.0, dwell_mean_ms=70.0, iki_mean_ms=260.0)

    _, a = extract_from_session(human_session(PHRASE, style=fluent, seed=4))
    _, b = extract_from_session(human_session(PHRASE, style=deliberate, seed=4))

    assert a.values["kbd_flight_negative_frac"] > b.values["kbd_flight_negative_frac"]
    assert b.values["kbd_flight_negative_frac"] == pytest.approx(0.0, abs=0.02)


def test_shift_hand_preference_is_recovered():
    righty = TypingStyle(right_shift_prob=1.0)
    lefty = TypingStyle(right_shift_prob=0.0)

    _, a = extract_from_session(human_session(PHRASE, style=righty, seed=5))
    _, b = extract_from_session(human_session(PHRASE, style=lefty, seed=5))

    assert a.values["kbd_shift_right_ratio"] == pytest.approx(1.0)
    assert b.values["kbd_shift_right_ratio"] == pytest.approx(0.0)


def test_tab_versus_click_navigation_is_recovered():
    tabber = TypingStyle(tab_between_fields=True)
    clicker = TypingStyle(tab_between_fields=False)

    _, a = extract_from_session(human_session(PHRASE, style=tabber, seed=6))
    _, b = extract_from_session(human_session(PHRASE, style=clicker, seed=6))

    assert a.values["int_tab_transition_ratio"] > b.values["int_tab_transition_ratio"]
    assert b.values["int_tab_transition_ratio"] == pytest.approx(0.0)


def test_backspace_rate_tracks_correction_frequency():
    sloppy = TypingStyle(backspace_prob=0.25)
    clean = TypingStyle(backspace_prob=0.0)

    _, a = extract_from_session(human_session(PHRASE, style=sloppy, seed=8))
    _, b = extract_from_session(human_session(PHRASE, style=clean, seed=8))

    assert a.values["kbd_backspace_rate"] > b.values["kbd_backspace_rate"]
    assert b.values["kbd_backspace_rate"] == pytest.approx(0.0)


def test_pause_probability_drives_pause_rate():
    thoughtful = TypingStyle(pause_prob=0.30, pause_ms=900.0)
    steady = TypingStyle(pause_prob=0.0)

    _, a = extract_from_session(human_session(PHRASE, style=thoughtful, seed=9))
    _, b = extract_from_session(human_session(PHRASE, style=steady, seed=9))

    assert a.values["kbd_pause_rate"] > b.values["kbd_pause_rate"]


def test_human_pointer_path_is_not_perfectly_straight():
    _, features = extract_from_session(human_session(PHRASE, seed=12))
    # 1.0 would be a ruler-straight line; a real arm always overshoots the ideal.
    assert features.values["ptr_straightness"] > 1.001


def test_synthetic_linear_path_is_straighter_than_a_human_one():
    _, human = extract_from_session(human_session(PHRASE, seed=13))
    _, linear = extract_from_session(linear_pointer_session(PHRASE, seed=13))

    assert linear.values["ptr_straightness"] < human.values["ptr_straightness"]
    assert linear.values["ptr_tremor"] < human.values["ptr_tremor"]


# ------------------------------------------------------- coverage, not zeros


def test_missing_modality_is_absent_rather_than_zero():
    """The single most important behaviour in this module.

    A user who never touched the mouse must produce *no* pointer features. If
    they produced zeros, the profile comparison would read that as an extreme
    deviation and reject a genuine login for a reason nobody could explain.
    """
    _, features = extract_from_session(human_session(PHRASE, seed=14, use_pointer=False))

    pointer_features = [n for n in features.values if n.startswith("ptr_")]
    assert pointer_features == []
    assert features.coverage_by_modality["pointer"] == 0.0
    assert features.coverage_by_modality["keyboard"] > 0.85
    # Coverage drops but keyboard evidence is untouched.
    assert 0.3 < features.coverage < 0.85


def test_missing_features_are_reported_not_silently_dropped():
    """Absence must be visible to callers, whatever its cause."""
    _, features = extract_from_session(human_session(PHRASE, seed=15, use_pointer=False))

    # Nothing pointer-shaped could be computed at all, so these land in
    # `missing` rather than `underpowered`.
    assert any(name.startswith("ptr_") for name in features.missing)
    for name in features.missing:
        assert name not in features.values
    assert set(features.underpowered) <= set(features.missing)


def test_underpowered_is_reserved_for_thin_evidence():
    # Three characters produce dwell samples, just nowhere near the twelve the
    # registry demands: computed, then rejected.
    _, features = extract_from_session(human_session("cat", seed=15))

    assert "kbd_dwell_median" in features.underpowered
    assert features.underpowered["kbd_dwell_median"] < 12


def test_features_below_min_observations_are_withheld():
    # Three characters cannot support a dwell median that requires twelve.
    _, features = extract_from_session(human_session("cat", seed=16))
    assert "kbd_dwell_median" not in features.values
    assert SPECS["kbd_dwell_median"].min_observations == 12


def test_value_injection_produces_almost_no_identity_features():
    """Assigning element.value emits input events and nothing else."""
    _, features = extract_from_session(value_injection_session(PHRASE))

    assert not [n for n in features.values if n.startswith("kbd_")]
    assert features.coverage < 0.2


def test_scripted_session_still_extracts_but_looks_degenerate():
    _, features = extract_from_session(scripted_session(PHRASE))

    # Timing features exist; they are simply unnaturally regular. Identifying
    # that is the automation detector's job, not the extractor's.
    assert "kbd_logiki_mad" in features.values
    assert features.values["kbd_logiki_mad"] == pytest.approx(0.0, abs=1e-6)
    assert features.values["kbd_flight_negative_frac"] == pytest.approx(0.0)


# -------------------------------------------------------------- robustness


def test_extraction_survives_a_single_event():
    session = human_session(PHRASE, seed=17)
    trimmed = session.model_copy(update={"events": session.events[:1]})

    features = extract_features(build_session_view(trimmed))

    # One event is enough for time-to-first-interaction and nothing else.
    # The point is that extraction returns near-empty rather than raising.
    assert set(features.values) <= {"int_time_to_first_interaction"}
    assert features.coverage < 0.05


def test_unpaired_keyup_does_not_crash_extraction():
    session = human_session(PHRASE, seed=18)
    without_keydowns = [e for e in session.events if e.type != "keydown"]
    trimmed = session.model_copy(update={"events": without_keydowns})

    features = extract_features(build_session_view(trimmed))
    assert "kbd_dwell_median" not in features.values
