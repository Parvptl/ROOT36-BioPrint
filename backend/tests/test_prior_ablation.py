"""Phase D experiment integrity.

These do not test authentication. They test that the *experiment* is a valid
experiment: that the only thing differing between the two conditions is the
population prior, that neither condition can contaminate the other, and that
the metrics are computed on the scale the production decision code actually
uses.

An A/B that silently varies a second input produces a number that looks like
evidence and is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.fingerprint.aalto_prior import aalto_population_prior, load_aalto_prior
from app.behavioral.fingerprint.scoring import score_identity
from app.behavioral.scoring.risk_engine import MIN_IDENTITY_COVERAGE
from evaluation.prior_ablation import (
    REAL_PRIOR,
    SYNTHETIC_PRIOR,
    Attempt,
    build_population,
    build_profile,
    eer,
    rates_at,
    roc_auc,
    run_condition,
    seeded_phrase,
    split_scores,
    threshold_at_far,
)

SEED = 4242


@pytest.fixture(scope="module")
def people():
    return build_population(n_users=4, genuine_per_user=3, seed=SEED)


@pytest.fixture(scope="module")
def donors():
    return build_population(n_users=2, genuine_per_user=1, seed=SEED + 9999)


# ------------------------------------------------------------ both conditions


def test_both_conditions_load_their_prior():
    a = aalto_population_prior(SYNTHETIC_PRIOR)
    b = aalto_population_prior(REAL_PRIOR)

    assert a is not None and b is not None
    assert a.scales and b.scales
    # They must actually differ, or the experiment has no independent variable.
    assert a.scales != b.scales


def test_the_real_condition_is_the_phase_c_artifact():
    prior = load_aalto_prior(REAL_PRIOR)
    assert prior.dataset_type == "real"
    assert prior.total_users == 168593
    assert prior.keyboard_feature_count == 15


# --------------------------------------------------- identical inputs


def test_the_phrase_generator_is_reproducible():
    assert seeded_phrase(11) == seeded_phrase(11)
    assert seeded_phrase(11) != seeded_phrase(12)


def test_the_population_is_reproducible_for_a_given_seed():
    first = build_population(2, 2, seed=SEED)
    second = build_population(2, 2, seed=SEED)

    for a, b in zip(first, second):
        assert a.enrollment == b.enrollment
        assert a.genuine == b.genuine


def test_both_conditions_receive_the_identical_sessions(people, donors):
    """The load-bearing property: the inputs are generated once, up front.

    If each condition regenerated its own sessions, any difference in the
    results could be a difference in the data rather than in the prior.
    """
    samples = [s for d in donors for s in d.enrollment]

    before_enroll = [list(p.enrollment) for p in people]
    before_genuine = [list(p.genuine) for p in people]

    run_condition("A", SYNTHETIC_PRIOR, people, samples, 2, SEED)
    run_condition("B", REAL_PRIOR, people, samples, 2, SEED)

    # Scoring must not mutate the sessions either condition was handed.
    assert [list(p.enrollment) for p in people] == before_enroll
    assert [list(p.genuine) for p in people] == before_genuine


def test_the_two_conditions_produce_independent_profiles(people, donors):
    """No object is shared, so no adaptation in A can reach B."""
    samples = [s for d in donors for s in d.enrollment]

    a = run_condition("A", SYNTHETIC_PRIOR, people, samples, 2, SEED)
    b = run_condition("B", REAL_PRIOR, people, samples, 2, SEED)

    for uid in a.profiles["8"]:
        pa, pb = a.profiles["8"][uid], b.profiles["8"][uid]
        assert pa is not pb
        for name in set(pa.features) & set(pb.features):
            assert pa.features[name] is not pb.features[name]
        # Both were fitted at version 1 from enrollment, never adapted.
        assert pa.update_count == 0 and pb.update_count == 0


def test_a_fresh_profile_is_built_for_every_maturity(people, donors):
    samples = [s for d in donors for s in d.enrollment]
    run = run_condition("A", SYNTHETIC_PRIOR, people, samples, 2, SEED)

    ids = [id(p) for level in run.profiles.values() for p in level.values()]
    assert len(ids) == len(set(ids)), "a profile object was reused across maturities"


def test_the_prior_changes_the_fitted_scales(people, donors):
    """Sanity: the independent variable actually reaches the profile."""
    samples = [s for d in donors for s in d.enrollment]
    enrollment = people[0].enrollment

    pa = build_profile(enrollment, aalto_population_prior(SYNTHETIC_PRIOR), samples)
    pb = build_profile(enrollment, aalto_population_prior(REAL_PRIOR), samples)

    shared = set(pa.features) & set(pb.features)
    assert any(
        abs(pa.features[n].scale - pb.features[n].scale) > 1e-9 for n in shared
    ), "the two priors produced identical profiles — the variable is not connected"


# ----------------------------------------------------- metric correctness


def test_roc_auc_uses_the_deviation_direction():
    """Lower score = more genuine. Verified against risk_engine, not assumed.

    A genuine set that scores uniformly lower than the impostor set is perfect
    separation, so AUC must be 1.0, not 0.0.
    """
    genuine = [0.05, 0.08, 0.10]
    impostor = [0.40, 0.55, 0.70]

    assert roc_auc(genuine, impostor) == pytest.approx(1.0)
    assert roc_auc(impostor, genuine) == pytest.approx(0.0)
    assert roc_auc([0.2, 0.2], [0.2, 0.2]) == pytest.approx(0.5)


def test_eer_is_zero_for_perfectly_separated_scores():
    result = eer([0.05, 0.10], [0.40, 0.55])
    assert result is not None
    assert result[0] == pytest.approx(0.0)


def test_a_far_target_below_the_sample_resolution_is_refused():
    """20 impostor scores cannot demonstrate a 0.1% false-acceptance rate."""
    impostor = [i / 100 for i in range(20)]

    assert threshold_at_far(impostor, 0.001) is None
    assert threshold_at_far(impostor, 0.05) is not None


# ------------------------------------------------ the coverage gate


def _attempt(label: str, score: float, coverage: float = 1.0, comparable: bool = True):
    return Attempt(uid=0, label=label, score=score, threshold=0.2,
                   latency_ms=0.1, comparable=comparable, coverage=coverage)


def test_an_uncomparable_attempt_is_blocked_regardless_of_its_score():
    """A featureless profile scores 0.0, which is not a measurement.

    risk_engine.decide checks coverage BEFORE comparing to a threshold. An
    experiment that skipped that gate would report a profile with no features
    as a flawless acceptor.
    """
    attempts = [_attempt("genuine", 0.0, comparable=False),
                _attempt("impostor", 0.0, comparable=False)]

    result = rates_at(attempts, threshold=0.2)

    assert result["genuine_accept"] == 0.0
    assert result["far"] == 0.0
    assert result["gated_out"] == 2


def test_low_coverage_is_blocked_at_the_production_boundary():
    below = _attempt("genuine", 0.01, coverage=MIN_IDENTITY_COVERAGE - 0.01)
    above = _attempt("genuine", 0.01, coverage=MIN_IDENTITY_COVERAGE)

    assert rates_at([below], 0.2)["genuine_accept"] == 0.0
    assert rates_at([above], 0.2)["genuine_accept"] == 1.0


def test_gated_attempts_are_excluded_from_score_distributions():
    """Their 0.0 is a placeholder; a ROC built on it would be fabricated."""
    attempts = [_attempt("genuine", 0.0, comparable=False),
                _attempt("genuine", 0.1),
                _attempt("impostor", 0.0, comparable=False),
                _attempt("impostor", 0.5)]

    genuine, impostor = split_scores(attempts)
    assert genuine == [0.1]
    assert impostor == [0.5]


# ------------------------------------------- fixed-threshold discipline


def test_the_fixed_threshold_mode_ignores_each_profiles_own_threshold():
    """Task 5 forbids tuning; the fixed mode must apply one cut to both."""
    attempts = [
        Attempt(0, "genuine", 0.25, threshold=0.30, latency_ms=0.1),
        Attempt(0, "genuine", 0.25, threshold=0.10, latency_ms=0.1),
    ]

    own = rates_at(attempts, None)
    fixed = rates_at(attempts, 0.20)

    assert own["genuine_accept"] == 0.5      # one profile's own cut admits it
    assert fixed["genuine_accept"] == 0.0    # the shared cut admits neither


# ----------------------------------------------------- unobserved Shift


def test_shift_stays_unobserved_and_never_becomes_zero_variability():
    a = aalto_population_prior(SYNTHETIC_PRIOR)
    b = aalto_population_prior(REAL_PRIOR)
    name = "kbd_shift_right_ratio"

    # The real prior contributes no Aalto-derived scale for it...
    assert name not in b.scales
    # ...and falls back to the relative prior rather than to zero.
    assert b.scale_for(name, 0.4) == pytest.approx(0.35 * 0.4)
    assert b.scale_for(name, 0.4) > 0.0
    # The synthetic prior does carry a (fabricated) number for it.
    assert name in a.scales


def test_shift_can_still_be_profiled_when_a_browser_session_observes_it(people, donors):
    """Aalto not measuring it must not stop the product measuring it."""
    samples = [s for d in donors for s in d.enrollment]
    profile = build_profile(people[0].enrollment, aalto_population_prior(REAL_PRIOR), samples)

    name = "kbd_shift_right_ratio"
    assert name in profile.features, "the browser feature must survive the real prior"
    assert profile.features[name].scale > 0.0

    # And an attempt carrying it still scores without error.
    result = score_identity(profile, people[0].genuine[0])
    assert result.is_comparable
