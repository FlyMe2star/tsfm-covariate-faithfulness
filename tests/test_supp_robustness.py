import numpy as np
import pytest

from covfaith_supp.robustness import (
    aggregation_contraction_gap,
    evaluate_metric_fixture_comparison,
    fixed_fine_distance_curve,
    fixed_fine_normalized_distance,
    verify_p2_freeze,
)


def _oracle() -> np.ndarray:
    values = np.zeros(24)
    values[1:4] = [0.5, 1.0, 0.5]
    values[9:12] = [-0.3, -0.6, -0.3]
    return values


def test_fixed_fine_distance_is_gain_invariant_and_monotone() -> None:
    oracle = _oracle()
    assert fixed_fine_normalized_distance(0.7 * oracle, oracle, 1) == pytest.approx(0.0)
    curve = fixed_fine_distance_curve(np.roll(oracle, 1), oracle)
    assert list(curve) == [1, 2, 4, 8]
    assert np.all(np.diff(list(curve.values())) <= 1e-12)
    assert aggregation_contraction_gap(np.roll(oracle, 1), oracle) >= 0.0


def test_fixed_fine_random_curves_obey_l1_contraction() -> None:
    rng = np.random.default_rng(220926)
    for _ in range(100):
        oracle = rng.normal(size=24)
        predicted = rng.normal(size=24)
        curve = fixed_fine_distance_curve(predicted, oracle)
        assert np.all(np.diff(list(curve.values())) <= 1e-12)


def test_zero_prediction_is_maximal_at_all_widths() -> None:
    oracle = _oracle()
    curve = fixed_fine_distance_curve(np.zeros_like(oracle), oracle)
    assert all(value == pytest.approx(1.0) for value in curve.values())


def test_fixture_comparison_is_complete_and_exposes_metric_tradeoffs() -> None:
    rows = evaluate_metric_fixture_comparison()
    assert len(rows) == 8
    by_name = {row["fixture"]: row for row in rows}
    assert by_name["positive_gain_0p70"]["registered_d1"] == pytest.approx(0.0)
    assert by_name["positive_gain_0p70"]["normalized_rmse"] > 0.0
    assert by_name["sign_flip"]["absolute_mass_transport"] == pytest.approx(0.0)
    assert by_name["sign_flip"]["registered_d1"] == pytest.approx(1.0)
    assert by_name["within_width8_shift_1"]["monotone_aggregation_gap"] > 0.0


def test_owner_approved_p2_lock_is_analysis_only() -> None:
    hashes = verify_p2_freeze(".")
    assert hashes["config_hash"] == (
        "7f7c27bc37ffc57ce3b2455877dab2a72d7d0d3255484042377cdaf90c27a284"
    )
    assert hashes["scientific_code_sha256"] == (
        "7178cf9bfa72d63a261765fdad333a30abd34ee0014c8bb082d160e6e7eebbe4"
    )
