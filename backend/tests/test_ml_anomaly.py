"""The ML anomaly layer: windowing, training, persistence, scoring, fallback.

Synthetic typists throughout. These verify the mechanism behaves as designed on
known-shaped input; they are not accuracy measurements. Real separation is
measured in evaluation/, and real-human accuracy remains unmeasured.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.auth.challenge import generate_phrase
from app.behavioral.ml import store
from app.behavioral.ml.anomaly_model import (
    MIN_TRAINING_WINDOWS,
    train_model,
)
from app.behavioral.ml.hybrid import MIN_ML_COVERAGE, combine
from app.behavioral.ml.preprocessing import (
    PREPROCESSING_VERSION,
    BehavioralFeatureProcessor,
    SchemaMismatch,
)
from app.behavioral.ml.windows import (
    MIN_WINDOW_PRESSES,
    WINDOW_EXCLUDED_FEATURES,
    extract_windows,
    window_feature_dicts,
)
from tests.factories import TypingStyle, human_session, scripted_session

ALICE = TypingStyle(
    iki_mean_ms=140.0, dwell_mean_ms=82.0, overlap_prob=0.45,
    right_shift_prob=0.95, tab_between_fields=True, pointer_speed=1.6,
)
MALLORY = TypingStyle(
    iki_mean_ms=320.0, dwell_mean_ms=140.0, overlap_prob=0.01,
    right_shift_prob=0.02, tab_between_fields=False, pointer_speed=0.6,
    backspace_prob=0.14, pause_prob=0.20,
)


def capture(style: TypingStyle, seed: int):
    return human_session(generate_phrase(), style=style, seed=seed)


def training_windows(style: TypingStyle, rounds: int = 8) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    for seed in range(rounds):
        windows.extend(window_feature_dicts(capture(style, seed)))
    return windows


@pytest.fixture
def model_dir(tmp_path) -> Path:
    store.clear_model_cache()
    yield tmp_path / "models"
    store.clear_model_cache()


# ------------------------------------------------------------------ windowing


def test_one_capture_yields_several_windows():
    """The whole point: one session must become more than one training vector."""
    windows = extract_windows(capture(ALICE, 1))

    assert len(windows) >= 3
    assert all(w.press_count >= MIN_WINDOW_PRESSES for w in windows)


def test_eight_rounds_give_a_usable_training_set():
    windows = training_windows(ALICE)
    assert len(windows) >= MIN_TRAINING_WINDOWS
    assert all(len(w) >= 6 for w in windows)


def test_windows_overlap_so_stride_is_smaller_than_width():
    windows = extract_windows(capture(ALICE, 2))
    if len(windows) >= 2:
        assert windows[1].start_ms < windows[0].end_ms


def test_session_level_features_are_excluded_from_windows():
    """Windowed, these encode position in the session rather than behaviour."""
    for window in extract_windows(capture(ALICE, 3)):
        assert not (set(window.features) & WINDOW_EXCLUDED_FEATURES)


def test_a_capture_too_short_to_window_yields_nothing():
    short = human_session("cat", style=ALICE, seed=4)
    assert extract_windows(short) == []


def test_windows_carry_no_password_material():
    """Identity windows are cut over the phrase field only."""
    windows = extract_windows(capture(ALICE, 5))
    for window in windows:
        for name in window.features:
            assert "password" not in name
            assert "username" not in name


# ------------------------------------------------------------- preprocessing


def test_processor_fixes_a_schema_and_orders_it():
    processor = BehavioralFeatureProcessor().fit(training_windows(ALICE))

    assert processor.is_fitted
    assert processor.feature_names == sorted(processor.feature_names)
    assert processor.n_features >= 6


def test_missing_features_are_imputed_to_the_training_centre():
    """Not to zero. Zero is an extreme value for most of these features and
    would manufacture an anomaly out of a missing measurement."""
    processor = BehavioralFeatureProcessor().fit(training_windows(ALICE))

    row = processor.transform_one({})
    assert row == pytest.approx([0.0] * processor.n_features, abs=1e-9)


def test_transform_is_order_stable():
    processor = BehavioralFeatureProcessor().fit(training_windows(ALICE))
    sample = window_feature_dicts(capture(ALICE, 20))[0]

    first = processor.transform_one(sample)
    shuffled = dict(reversed(list(sample.items())))
    assert processor.transform_one(shuffled) == pytest.approx(first)


def test_processor_round_trips_through_a_dict():
    processor = BehavioralFeatureProcessor().fit(training_windows(ALICE))
    restored = BehavioralFeatureProcessor.from_dict(processor.to_dict())

    assert restored.feature_names == processor.feature_names
    sample = window_feature_dicts(capture(ALICE, 21))[0]
    assert restored.transform_one(sample) == pytest.approx(processor.transform_one(sample))


def test_a_foreign_schema_version_is_refused():
    processor = BehavioralFeatureProcessor().fit(training_windows(ALICE))
    payload = processor.to_dict()
    payload["version"] = "0.0-from-the-future"

    with pytest.raises(SchemaMismatch):
        BehavioralFeatureProcessor.from_dict(payload)


def test_an_inconsistent_schema_is_refused():
    with pytest.raises(SchemaMismatch):
        BehavioralFeatureProcessor.from_dict(
            {
                "version": PREPROCESSING_VERSION,
                "feature_names": ["a", "b"],
                "centres": [1.0],
                "scales": [1.0],
            }
        )


# ------------------------------------------------------------------ training


def test_a_valid_enrollment_produces_a_model():
    model = train_model(training_windows(ALICE))

    assert model is not None
    assert model.n_training_windows >= MIN_TRAINING_WINDOWS
    assert model.processor.n_features >= 6


def test_insufficient_data_returns_no_model_rather_than_a_bad_one():
    """Graceful fallback, not a fabricated model."""
    assert train_model([]) is None
    assert train_model(training_windows(ALICE, rounds=1)[:4]) is None


def test_training_is_deterministic():
    windows = training_windows(ALICE)
    probe = window_feature_dicts(capture(ALICE, 30))

    first = train_model(windows).anomaly_for(probe)
    second = train_model(windows).anomaly_for(probe)

    assert first.score == pytest.approx(second.score)


# ------------------------------------------------------------------- scoring


def test_genuine_behaviour_scores_lower_than_an_impostor():
    model = train_model(training_windows(ALICE))

    genuine = [
        model.anomaly_for(window_feature_dicts(capture(ALICE, 40 + i))).score
        for i in range(6)
    ]
    impostor = [
        model.anomaly_for(window_feature_dicts(capture(MALLORY, 50 + i))).score
        for i in range(6)
    ]

    assert max(genuine) < min(impostor), f"genuine {genuine} impostor {impostor}"


def test_the_score_is_bounded_and_oriented():
    """0 is normal, 1 is highly anomalous. Never outside that."""
    model = train_model(training_windows(ALICE))

    for style, seed in ((ALICE, 60), (MALLORY, 61)):
        result = model.anomaly_for(window_feature_dicts(capture(style, seed)))
        assert 0.0 <= result.score <= 1.0


def test_scoring_the_same_input_twice_gives_the_same_answer():
    model = train_model(training_windows(ALICE))
    probe = window_feature_dicts(capture(ALICE, 62))

    assert model.anomaly_for(probe).score == pytest.approx(model.anomaly_for(probe).score)


def test_no_windows_means_no_usable_score():
    model = train_model(training_windows(ALICE))
    result = model.anomaly_for([])

    assert not result.is_usable
    assert result.windows_scored == 0


# --------------------------------------------------------------- persistence


def test_a_saved_model_reloads_and_scores_identically(model_dir):
    model = train_model(training_windows(ALICE))
    probe = window_feature_dicts(capture(ALICE, 70))
    before = model.anomaly_for(probe).score

    store.save_model(7, model, model_dir=model_dir)
    store.clear_model_cache()
    reloaded = store.load_model(7, model_dir=model_dir)

    assert reloaded is not None
    assert reloaded.anomaly_for(probe).score == pytest.approx(before)


def test_metadata_is_readable_and_carries_no_secrets(model_dir):
    model = train_model(training_windows(ALICE))
    directory = store.save_model(8, model, model_dir=model_dir)

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["model_type"] == "IsolationForest"
    assert metadata["feature_names"]
    assert metadata["n_training_windows"] >= MIN_TRAINING_WINDOWS
    blob = json.dumps(metadata).lower()
    for forbidden in ("password", "secret", "token", "hash", "username"):
        assert forbidden not in blob


def test_a_missing_model_loads_as_none(model_dir):
    assert store.load_model(999, model_dir=model_dir) is None


def test_a_corrupt_artefact_degrades_instead_of_raising(model_dir):
    model = train_model(training_windows(ALICE))
    directory = store.save_model(9, model, model_dir=model_dir)
    (directory / "behavioral_model.joblib").write_bytes(b"not a model")
    store.clear_model_cache()

    assert store.load_model(9, model_dir=model_dir) is None


def test_a_version_mismatch_is_refused(model_dir):
    model = train_model(training_windows(ALICE))
    directory = store.save_model(10, model, model_dir=model_dir)
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["model_version"] = "99.0"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    store.clear_model_cache()

    assert store.load_model(10, model_dir=model_dir) is None


def test_a_schema_mismatch_between_artefact_and_metadata_is_refused(model_dir):
    """The silent-nonsense failure mode: right model, wrong columns."""
    model = train_model(training_windows(ALICE))
    directory = store.save_model(11, model, model_dir=model_dir)
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["feature_names"] = metadata["feature_names"][:-1]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    store.clear_model_cache()

    assert store.load_model(11, model_dir=model_dir) is None


def test_deleting_a_model_evicts_it_from_the_cache(model_dir):
    model = train_model(training_windows(ALICE))
    store.save_model(12, model, model_dir=model_dir)
    assert store.load_model(12, model_dir=model_dir) is not None

    store.delete_model(12, model_dir=model_dir)
    assert store.load_model(12, model_dir=model_dir) is None


def test_retraining_invalidates_the_cached_model(model_dir):
    """A cache keyed on mtime must not serve a stale forest after a retrain."""
    full = training_windows(ALICE)
    store.save_model(13, train_model(full), model_dir=model_dir)
    assert store.load_model(13, model_dir=model_dir).n_training_windows == len(full)

    smaller = full[: MIN_TRAINING_WINDOWS + 2]
    store.save_model(13, train_model(smaller), model_dir=model_dir)
    assert store.load_model(13, model_dir=model_dir).n_training_windows == len(smaller)


# -------------------------------------------------------------- hybrid blend


def test_without_a_model_the_statistical_score_passes_through_untouched():
    """Backward compatibility: no model means the original behaviour exactly."""
    hybrid = combine(0.42, None)

    assert hybrid.score == pytest.approx(0.42)
    assert hybrid.ml_applied is False
    assert hybrid.ml_weight == 0.0


def test_the_blend_sits_between_its_two_components():
    model = train_model(training_windows(ALICE))
    anomaly = model.anomaly_for(window_feature_dicts(capture(MALLORY, 80)))

    hybrid = combine(0.10, anomaly)

    assert hybrid.ml_applied
    low, high = sorted([0.10, anomaly.score])
    assert low <= hybrid.score <= high


def test_weights_are_renormalised_so_they_cannot_rescale_the_score():
    """Misconfigured weights must not silently move the operating point."""
    model = train_model(training_windows(ALICE))
    anomaly = model.anomaly_for(window_feature_dicts(capture(ALICE, 81)))

    balanced = combine(0.2, anomaly, statistical_weight=0.6, ml_weight=0.4)
    doubled = combine(0.2, anomaly, statistical_weight=6.0, ml_weight=4.0)

    assert balanced.score == pytest.approx(doubled.score)


def test_weight_extremes_select_a_single_component():
    model = train_model(training_windows(ALICE))
    anomaly = model.anomaly_for(window_feature_dicts(capture(MALLORY, 82)))

    only_stat = combine(0.15, anomaly, statistical_weight=1.0, ml_weight=0.0)
    only_ml = combine(0.15, anomaly, statistical_weight=0.0, ml_weight=1.0)

    assert only_stat.score == pytest.approx(0.15)
    assert only_ml.score == pytest.approx(anomaly.score)


def test_a_thinly_covered_capture_drops_the_ml_term():
    """Scoring a mostly-imputed vector would score our imputation, not the user."""
    model = train_model(training_windows(ALICE))
    sparse = [{next(iter(model.processor.feature_names)): 1.0}]
    anomaly = model.anomaly_for(sparse)

    assert anomaly.mean_coverage < MIN_ML_COVERAGE
    hybrid = combine(0.2, anomaly)
    assert hybrid.ml_applied is False
    assert hybrid.score == pytest.approx(0.2)


# ------------------------------------- automation stays a separate concern


def test_a_scripted_capture_is_not_laundered_through_the_ml_score():
    """Automation is detected by the automation detector, not by this layer.

    The point of the assertion is that the ML score is not being asked to do
    the automation detector's job, and that a bot is not quietly reclassified
    as an identity mismatch.
    """
    from app.behavioral.bot_detection.detector import detect_automation
    from app.behavioral.events import build_session_view

    bot = scripted_session(generate_phrase())
    automation = detect_automation(build_session_view(bot), bot)

    assert automation.score > 0.5
