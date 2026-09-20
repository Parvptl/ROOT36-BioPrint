"""The Aalto population prior: aggregation, unobserved features, loading.

The property these tests exist to protect is one semantic distinction:

    unobserved      we do not know how much people vary on this feature
    scale == 0.0    people do not vary on this feature at all

Collapsing the first into the second would make every deviation on that
feature infinitely significant. Aalto cannot measure `kbd_shift_right_ratio`
at all — keycode 16 does not say which Shift — so this is not hypothetical.

Fixtures are tiny and deterministic. Nothing here touches the 168k-file corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.fingerprint.aalto_prior import (
    AaltoPrior,
    UnobservedFeature,
    load_aalto_prior,
)
from app.behavioral.fingerprint.population import ABSOLUTE_FLOOR, RELATIVE_SPREAD
from app.behavioral.fingerprint.precompute_aalto_prior import (
    aggregate_user_features,
    build_manifest,
    compute_population_stats,
    process_aalto_data,
    save_prior,
    unobserved_entries,
    validate_population_stats,
)
from evaluation.ml.aalto_adapter import (
    UNOBSERVED_FEATURES,
    extract_session_features,
    iter_sessions,
)

DATA = _BACKEND / "data"
SYNTHETIC = DATA / "aalto_prior_synthetic_bootstrap.json"
PRODUCTION = DATA / "aalto_prior.json"
REAL = DATA / "aalto_prior_real.json"

HEADER = (
    "PARTICIPANT_ID\tTEST_SECTION_ID\tSENTENCE\tUSER_INPUT\tKEYSTROKE_ID\t"
    "PRESS_TIME\tRELEASE_TIME\tLETTER\tKEYCODE"
)


def _row(participant: str, section: str, i: int, press: int, release: int, keycode: int) -> str:
    return (
        f"{participant}\t{section}\tthe quick brown fox\tthe quick brown fox\t{i}\t"
        f"{press}\t{release}\tx\t{keycode}"
    )


def write_participant(
    path: Path,
    participant: str,
    sections: int,
    presses_per_section: int = 20,
    iki: int = 150,
    dwell: int = 95,
    shift_every: int = 0,
) -> Path:
    """A synthetic Aalto file: deterministic timings, real column layout."""
    lines = [HEADER]
    keystroke_id = 0
    for s in range(sections):
        t = 1_500_000_000_000 + s * 100_000
        for k in range(presses_per_section):
            keystroke_id += 1
            keycode = 65 + (k % 26)
            if shift_every and k % shift_every == 0:
                keycode = 16
            lines.append(
                _row(participant, f"{participant}-{s}", keystroke_id, t, t + dwell, keycode)
            )
            t += iki
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------ the adapter


def test_shift_presses_never_produce_a_ratio(tmp_path):
    """Keycode 16 must not become a measurement of which Shift was pressed."""
    f = write_participant(tmp_path / "1_keystrokes.txt", "1", sections=1, shift_every=4)

    (_section, view, _rows), = list(iter_sessions(f))
    shifts = [p for p in view.presses if p.key_class == "shift"]
    assert shifts, "fixture should contain shift presses"

    features = extract_session_features(view)

    # Not zero, not imputed: absent.
    assert "kbd_shift_right_ratio" not in features
    # And the class must match neither side, or keyboard.extract would emit
    # a ratio of exactly 0.0 or 1.0 from data that says nothing about hands.
    assert all(p.key_class not in ("shift_left", "shift_right") for p in shifts)


def test_sessions_are_gated_the_way_production_gates_them(tmp_path):
    """A session too short to back a feature must not contribute one."""
    short = write_participant(
        tmp_path / "2_keystrokes.txt", "2", sections=1, presses_per_section=11
    )
    (_s, view, _r), = list(iter_sessions(short))
    features = extract_session_features(view)

    # kbd_speed_cv needs 20 observations; 11 presses cannot supply them.
    assert "kbd_speed_cv" not in features
    # kbd_dwell_median needs 12; also not satisfied.
    assert "kbd_dwell_median" not in features


def test_iter_sessions_splits_on_section_id(tmp_path):
    f = write_participant(tmp_path / "3_keystrokes.txt", "3", sections=4)
    sessions = list(iter_sessions(f))
    assert len(sessions) == 4
    assert len({s[0] for s in sessions}) == 4


# ------------------------------------------------- participant aggregation


def test_aggregation_takes_the_median_across_a_participants_sessions():
    sessions = [
        {"kbd_dwell_median": 100.0},
        {"kbd_dwell_median": 110.0},
        {"kbd_dwell_median": 300.0},  # one odd session must not drag the centre
    ]
    assert aggregate_user_features(sessions) == {"kbd_dwell_median": 110.0}


def test_a_feature_seen_in_only_one_session_is_not_aggregated():
    sessions = [{"kbd_dwell_median": 100.0, "kbd_pause_rate": 4.0}, {"kbd_dwell_median": 120.0}]
    out = aggregate_user_features(sessions)
    assert "kbd_dwell_median" in out
    assert "kbd_pause_rate" not in out


def test_participants_are_balanced_not_pooled(tmp_path):
    """A participant with many sessions must not outvote one with few.

    The whole point of a population prior is 'how much do people differ'. If
    sessions were pooled, one prolific participant would answer that question
    on everyone's behalf.
    """
    # Participant A: 20 sessions, fast. Participant B: 3 sessions, slow.
    write_participant(tmp_path / "10_keystrokes.txt", "10", sections=20, iki=120, dwell=80)
    write_participant(tmp_path / "11_keystrokes.txt", "11", sections=3, iki=260, dwell=160)

    user_features, report = process_aalto_data(tmp_path, progress_every=0)

    assert report.participants_processed == 2
    assert report.sessions_discovered == 23
    # Two participants in, two observations out — regardless of session counts.
    assert len(user_features) == 2


# -------------------------------------------------- population statistics


def _population(n: int = 40) -> list[dict[str, float]]:
    return [{"kbd_dwell_median": 80.0 + i, "kbd_pause_rate": float(i % 7)} for i in range(n)]


def test_population_statistics_are_ordered_and_finite():
    stats = compute_population_stats(_population())
    entry = stats["kbd_dwell_median"]

    assert entry["status"] == "observed"
    assert entry["population_scale"] > 0.0
    assert (
        entry["p5"] <= entry["p25"] <= entry["p50"] <= entry["p75"] <= entry["p95"]
    )
    assert entry["min"] <= entry["p5"]
    assert entry["p95"] <= entry["max"]
    assert entry["p50"] == pytest.approx(entry["population_median"])

    # The fixture is deliberately small, so "want >=50 participants" and the
    # feature-count note are expected. Nothing else should fire.
    warnings = [
        w for w in validate_population_stats(stats)
        if "want >=50" not in w and "computable keyboard features" not in w
    ]
    assert warnings == []


def test_missing_and_coverage_are_reported_per_feature():
    population = _population(30)
    for sample in population[:10]:
        del sample["kbd_pause_rate"]

    stats = compute_population_stats(population)
    entry = stats["kbd_pause_rate"]

    assert entry["n_users"] == 20
    assert entry["missing_users"] == 10
    # Stored rounded to 4 decimals, so compare at that resolution.
    assert entry["missing_pct"] == pytest.approx(100 * 10 / 30, abs=1e-4)
    assert entry["participant_coverage_pct"] == pytest.approx(100 * 20 / 30, abs=1e-4)


def test_a_feature_with_too_few_participants_is_omitted_not_estimated():
    population = [{"kbd_dwell_median": 90.0} for _ in range(5)]
    assert compute_population_stats(population) == {}


def test_validation_flags_non_monotone_percentiles():
    broken = {
        "kbd_dwell_median": {
            "status": "observed", "population_scale": 1.0, "population_median": 5.0,
            "p5": 9.0, "p25": 2.0, "p50": 5.0, "p75": 7.0, "p95": 8.0, "n_users": 100,
        }
    }
    warnings = validate_population_stats(broken)
    assert any("not monotone" in w for w in warnings)


# ------------------------------------------------------ unobserved feature


def test_unobserved_entries_carry_no_numbers():
    entries = unobserved_entries()
    assert set(entries) == set(UNOBSERVED_FEATURES)

    entry = entries["kbd_shift_right_ratio"]
    assert entry["status"] == "unobserved"
    assert entry["reason"]
    # Nothing a reader could mistake for a measurement.
    for forbidden in ("population_median", "population_mad", "population_scale", "p50"):
        assert forbidden not in entry


def test_validation_does_not_try_to_check_an_unobserved_feature():
    """No warning may mention the unobserved feature: it has nothing to check."""
    stats = compute_population_stats(_population(60))
    stats.update(unobserved_entries())

    warnings = validate_population_stats(stats)
    assert not any("kbd_shift_right_ratio" in w for w in warnings)


# ------------------------------------------- serialization round trip


def test_prior_round_trips_through_json_with_the_unobserved_marker(tmp_path):
    stats = compute_population_stats(_population(60))
    stats.update(unobserved_entries())
    out = tmp_path / "prior.json"

    save_prior(stats, out, n_users=60, source="test", dataset_type="real")

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["meta"]["dataset_type"] == "real"
    assert payload["meta"]["unobserved_features"] == ["kbd_shift_right_ratio"]
    assert payload["meta"]["n_features"] == 2  # observed only
    assert payload["features"]["kbd_shift_right_ratio"]["status"] == "unobserved"

    prior = load_aalto_prior(out)
    assert prior.keyboard_feature_count == 2
    assert prior.is_unobserved("kbd_shift_right_ratio")
    assert "kbd_shift_right_ratio" not in prior.features


def test_an_unobserved_feature_never_becomes_zero_variability(tmp_path):
    """The load-bearing test of this module."""
    out = tmp_path / "prior.json"
    save_prior(
        {**compute_population_stats(_population(60)), **unobserved_entries()},
        out, n_users=60, source="test", dataset_type="real",
    )

    population_prior = load_aalto_prior(out).to_population_prior()

    assert "kbd_shift_right_ratio" not in population_prior.scales

    own_median = 0.4
    scale = population_prior.scale_for("kbd_shift_right_ratio", own_median)

    # Falls back to the relative prior, exactly as a pointer feature would.
    assert scale == pytest.approx(max(RELATIVE_SPREAD * own_median, ABSOLUTE_FLOOR))
    # Emphatically not zero: a zero scale makes any deviation infinite.
    assert scale > 0.0


def test_a_raw_unobserved_payload_is_not_parsed_as_a_statistic():
    """Defence against the representation being written by hand elsewhere."""
    prior = AaltoPrior()
    payload = {
        "meta": {"n_users": 100, "source": "test"},
        "features": {"kbd_shift_right_ratio": {"status": "unobserved"}},
    }
    from app.behavioral.fingerprint.aalto_prior import _parse_prior

    prior = _parse_prior(payload)
    assert prior.features == {}
    assert prior.unobserved == {
        "kbd_shift_right_ratio": UnobservedFeature(name="kbd_shift_right_ratio", reason="")
    }
    assert prior.to_population_prior().scales == {}


# ------------------------------------------------------ shipped artifacts


def test_the_synthetic_baseline_is_present_and_loads():
    assert SYNTHETIC.exists(), "the preserved synthetic bootstrap must not be deleted"
    prior = load_aalto_prior(SYNTHETIC)
    assert prior.is_loaded
    assert prior.total_users > 0


def test_the_production_prior_still_loads():
    assert PRODUCTION.exists()
    assert load_aalto_prior(PRODUCTION).is_loaded


@pytest.mark.skipif(not REAL.exists(), reason="real prior not generated in this checkout")
def test_the_real_prior_loads_with_shift_unobserved():
    prior = load_aalto_prior(REAL)

    assert prior.is_loaded
    assert prior.dataset_type == "real"
    assert prior.is_unobserved("kbd_shift_right_ratio")
    assert "kbd_shift_right_ratio" not in prior.features
    assert prior.keyboard_feature_count == 15

    population_prior = prior.to_population_prior()
    assert len(population_prior.scales) == 15
    assert all(scale > 0.0 for scale in population_prior.scales.values())


# ------------------------------------------------------------ manifest


def test_the_manifest_is_deterministic_for_the_same_input(tmp_path):
    write_participant(tmp_path / "20_keystrokes.txt", "20", sections=2)
    write_participant(tmp_path / "21_keystrokes.txt", "21", sections=2)

    first = build_manifest(tmp_path, 0)
    second = build_manifest(tmp_path, 0)

    assert first == second
    assert first["file_count"] == 2
    assert len(first["manifest_sha256"]) == 64
