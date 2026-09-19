"""Fingerprint model: profile fitting, scoring, and threshold calibration.

These tests verify the mechanics behave as designed on synthetic typists whose
parameters we control. They are not evidence about real accuracy.
"""

from __future__ import annotations

import pytest

from app.behavioral.features.extractor import extract_from_session
from app.behavioral.fingerprint.calibration import (
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    build_calibrated_profile,
    leave_one_session_out_scores,
)
from app.behavioral.fingerprint.population import (
    PopulationPrior,
    fit_population_prior,
)
from app.behavioral.fingerprint.profile import (
    MIN_FEATURE_SESSIONS,
    BehaviorProfile,
    fit_profile,
)
from app.behavioral.fingerprint.scoring import SATURATION_Z, score_identity
from app.auth.challenge import generate_phrase
from tests.factories import TypingStyle, human_session

EMPTY_PRIOR = PopulationPrior()


def capture(style: TypingStyle, seed: int, use_pointer: bool = True) -> dict[str, float]:
    """One enrollment-shaped capture, with a fresh phrase as the real flow uses."""
    session = human_session(
        generate_phrase(), style=style, seed=seed, use_pointer=use_pointer
    )
    _, features = extract_from_session(session)
    return features.as_dict()


def captures(style: TypingStyle, seeds: range, use_pointer: bool = True):
    return [capture(style, seed, use_pointer) for seed in seeds]


# The same person on different days versus a genuinely different typist.
ALICE = TypingStyle(
    iki_mean_ms=140.0, dwell_mean_ms=82.0, overlap_prob=0.45,
    right_shift_prob=0.95, tab_between_fields=True, pointer_speed=1.6,
)
MALLORY = TypingStyle(
    iki_mean_ms=310.0, dwell_mean_ms=135.0, overlap_prob=0.02,
    right_shift_prob=0.05, tab_between_fields=False, pointer_speed=0.7,
    backspace_prob=0.12, pause_prob=0.18,
)


# ------------------------------------------------------------ profile fitting


def test_profile_covers_the_features_seen_across_sessions():
    profile_features = fit_profile(captures(ALICE, range(5)), EMPTY_PRIOR)

    assert len(profile_features) > 15
    assert any(n.startswith("kbd_") for n in profile_features)
    assert any(n.startswith("ptr_") for n in profile_features)
    assert any(n.startswith("int_") for n in profile_features)


def test_rarely_seen_features_are_excluded_from_the_profile():
    """A feature present in one round of five cannot be characterised."""
    sessions = captures(ALICE, range(4), use_pointer=False)
    sessions.append(capture(ALICE, 99, use_pointer=True))

    profile_features = fit_profile(sessions, EMPTY_PRIOR)

    pointer = [n for n in profile_features if n.startswith("ptr_")]
    assert pointer == [], f"pointer features seen once should be excluded: {pointer}"


def test_scale_never_collapses_to_zero():
    """The failure mode the shrinkage floor exists to prevent.

    Without a floor, a user who is consistent on a feature gets scale ~0, every
    later login scores an enormous z on that axis, and the account is bricked.
    """
    steady = TypingStyle(iki_jitter=0.001, dwell_jitter=0.001, overlap_prob=0.0,
                         pause_prob=0.0, backspace_prob=0.0)
    profile_features = fit_profile(captures(steady, range(5)), EMPTY_PRIOR)

    assert profile_features
    for stat in profile_features.values():
        assert stat.scale > 0.0, stat.name


def test_weights_are_uniform_without_a_population_prior():
    """With no population data we do not know which features discriminate.

    Claiming a ranking from nothing would be inventing information.
    """
    profile_features = fit_profile(captures(ALICE, range(5)), EMPTY_PRIOR)
    assert {stat.weight for stat in profile_features.values()} == {1.0}


def test_population_prior_makes_weights_vary():
    population = captures(MALLORY, range(50, 62)) + captures(ALICE, range(70, 78))
    prior = fit_population_prior(population)
    assert prior.is_informative

    profile_features = fit_profile(captures(ALICE, range(5)), prior)
    weights = {round(stat.weight, 4) for stat in profile_features.values()}

    assert len(weights) > 1
    assert all(0.0 <= w <= 1.0 for w in weights)


# ------------------------------------------------------------------- scoring


def build(style: TypingStyle, seeds: range, prior: PopulationPrior = EMPTY_PRIOR):
    sessions = captures(style, seeds)
    return BehaviorProfile(features=fit_profile(sessions, prior), session_count=len(sessions))


def test_genuine_attempt_scores_lower_than_an_impostor():
    """The central claim of the system, on controlled input."""
    profile = build(ALICE, range(6))

    genuine = score_identity(profile, capture(ALICE, 500))
    impostor = score_identity(profile, capture(MALLORY, 500))

    assert genuine.is_comparable and impostor.is_comparable
    assert impostor.score > genuine.score
    assert impostor.score > 2 * genuine.score


def test_genuine_attempts_are_consistently_below_impostor_attempts():
    profile = build(ALICE, range(6))

    genuine = [score_identity(profile, capture(ALICE, s)).score for s in range(200, 212)]
    impostor = [score_identity(profile, capture(MALLORY, s)).score for s in range(300, 312)]

    assert max(genuine) < min(impostor), (
        f"overlap: genuine max {max(genuine):.3f}, impostor min {min(impostor):.3f}"
    )


def test_score_is_bounded():
    profile = build(ALICE, range(5))
    for seed in range(400, 406):
        for style in (ALICE, MALLORY):
            result = score_identity(profile, capture(style, seed))
            assert 0.0 <= result.score <= 1.0


def test_one_wild_feature_cannot_dominate_the_decision():
    """Saturation check.

    A genuine user with one anomalous axis (a borrowed mouse, a sore wrist)
    must not be rejected on that axis alone.
    """
    profile = build(ALICE, range(6))
    genuine = capture(ALICE, 600)

    sabotaged = dict(genuine)
    sabotaged["kbd_dwell_median"] = genuine["kbd_dwell_median"] * 50

    clean_score = score_identity(profile, genuine).score
    damaged_score = score_identity(profile, sabotaged).score

    assert damaged_score > clean_score
    # One feature out of ~20 can move the aggregate, but nowhere near to the top.
    assert damaged_score < 0.35


def test_saturation_caps_a_single_feature_contribution():
    profile = build(ALICE, range(5))
    features = capture(ALICE, 601)

    mild = dict(features)
    extreme = dict(features)
    stat = profile.features["kbd_speed_kps"]
    mild["kbd_speed_kps"] = stat.median + SATURATION_Z * stat.scale
    extreme["kbd_speed_kps"] = stat.median + 1000 * SATURATION_Z * stat.scale

    assert score_identity(profile, mild).score == pytest.approx(
        score_identity(profile, extreme).score, abs=1e-9
    )


def test_missing_features_lower_coverage_not_score():
    """Not observing the mouse is not evidence the mouse moved wrongly."""
    profile = build(ALICE, range(6))
    full = capture(ALICE, 700)
    without_pointer = {k: v for k, v in full.items() if not k.startswith("ptr_")}

    complete = score_identity(profile, full)
    partial = score_identity(profile, without_pointer)

    assert partial.coverage < complete.coverage
    assert partial.score == pytest.approx(complete.score, abs=0.12)
    assert "pointer" not in partial.modality_scores


def test_modality_scores_localise_the_difference():
    profile = build(ALICE, range(6))
    result = score_identity(profile, capture(MALLORY, 800))

    assert set(result.modality_scores) <= {"keyboard", "pointer", "interaction"}
    assert result.modality_scores["keyboard"] > 0.2


def test_contributions_are_ranked_and_sum_to_one():
    profile = build(ALICE, range(6))
    result = score_identity(profile, capture(MALLORY, 801))

    shares = [c.share for c in result.contributions]
    assert shares == sorted(shares, reverse=True)
    assert sum(shares) == pytest.approx(1.0, abs=1e-9)


def test_empty_profile_scores_nothing_rather_than_crashing():
    result = score_identity(BehaviorProfile(), capture(ALICE, 900))
    assert result.score == 0.0
    assert not result.is_comparable


# --------------------------------------------------------------- calibration


def test_leave_one_out_produces_one_score_per_session():
    sessions = captures(ALICE, range(5))
    scores = leave_one_session_out_scores(sessions, EMPTY_PRIOR)

    assert len(scores) == len(sessions)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_leave_one_out_does_not_score_a_session_against_itself():
    """A session scored against a profile containing it reads near zero.

    If that leaked, the threshold would be set impossibly tight and every real
    login would fail.
    """
    sessions = captures(ALICE, range(5))
    loo = leave_one_session_out_scores(sessions, EMPTY_PRIOR)

    full = BehaviorProfile(features=fit_profile(sessions, EMPTY_PRIOR))
    self_scored = [score_identity(full, s).score for s in sessions]

    assert sum(loo) / len(loo) > sum(self_scored) / len(self_scored)


def test_calibrated_threshold_separates_genuine_from_impostor():
    population = captures(MALLORY, range(50, 62))
    prior = fit_population_prior(population)

    profile = build_calibrated_profile(captures(ALICE, range(6)), prior, population)

    assert profile.threshold_source == "calibrated"
    assert MIN_THRESHOLD <= profile.threshold <= MAX_THRESHOLD

    genuine = [score_identity(profile, capture(ALICE, s)).score for s in range(210, 220)]
    impostor = [score_identity(profile, capture(MALLORY, s)).score for s in range(310, 320)]

    assert all(s <= profile.threshold for s in genuine), genuine
    assert all(s > profile.threshold for s in impostor), impostor


def test_calibration_falls_back_honestly_without_population_data():
    profile = build_calibrated_profile(captures(ALICE, range(5)), EMPTY_PRIOR, [])

    assert profile.threshold_source == "genuine_only"
    assert MIN_THRESHOLD <= profile.threshold <= MAX_THRESHOLD
    # The note must state that false acceptance was not measured, rather than
    # letting a reader assume it was.
    assert "unmeasured" in profile.calibration["note"]
    assert profile.calibration["metrics"]["population_false_acceptance_rate"] is None


def test_calibration_reports_fallback_when_there_are_too_few_sessions():
    profile = build_calibrated_profile(captures(ALICE, range(2)), EMPTY_PRIOR, [])

    assert profile.threshold_source == "fallback_prior"
    assert "unmeasured" in profile.calibration["note"]


def test_genuine_only_threshold_still_admits_the_genuine_user():
    profile = build_calibrated_profile(captures(ALICE, range(6)), EMPTY_PRIOR, [])

    scores = [score_identity(profile, capture(ALICE, s)).score for s in range(220, 232)]
    accepted = sum(1 for s in scores if s <= profile.threshold)

    # Without impostor data the threshold is set from genuine spread alone; it
    # must at least not lock out the person it was built from.
    assert accepted >= len(scores) - 1, scores


def test_profile_records_the_evidence_behind_its_threshold():
    population = captures(MALLORY, range(50, 62))
    prior = fit_population_prior(population)
    profile = build_calibrated_profile(captures(ALICE, range(6)), prior, population)

    calibration = profile.calibration
    assert calibration["metrics"]["genuine_n"] == 6
    assert calibration["metrics"]["impostor_n"] == len(population)
    assert profile.population_size == len(population)
    assert profile.population_informative is True


def test_min_feature_sessions_constant_is_respected():
    sessions = captures(ALICE, range(MIN_FEATURE_SESSIONS - 1))
    assert fit_profile(sessions, EMPTY_PRIOR) == {}
