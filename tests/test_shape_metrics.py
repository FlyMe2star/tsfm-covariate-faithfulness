import numpy as np
import pytest

from covfaith.shape_metrics import (
    block_signed_response,
    interaction_response,
    shape_distance_curve,
    shape_metric_suite,
    signed_shape_distance,
    temporal_mass_distance,
)


def _oracle() -> np.ndarray:
    values = np.zeros(24)
    values[1:4] = [0.5, 1.0, 0.5]
    values[9:12] = [-0.3, -0.6, -0.3]
    return values


def test_block_operator_requires_exact_divisibility() -> None:
    np.testing.assert_allclose(block_signed_response(np.arange(8), 4), [6, 22])
    with pytest.raises(ValueError):
        block_signed_response(np.arange(7), 4)


def test_oracle_and_positive_scaling_have_zero_shape_distance() -> None:
    oracle = _oracle()
    for width in (1, 2, 4, 8):
        assert signed_shape_distance(oracle, oracle, width) == pytest.approx(0.0)
        assert signed_shape_distance(0.2 * oracle, oracle, width) == pytest.approx(0.0)


def test_within_bin_shift_is_hidden_by_width_eight() -> None:
    oracle = _oracle()
    shifted = np.roll(oracle, 1)
    curve = shape_distance_curve(shifted, oracle)
    assert curve[1] > 0
    assert curve[8] == pytest.approx(0.0)
    assert shape_metric_suite(shifted, oracle).hidden_distortion_gap > 0


def test_sign_flip_is_maximal_signed_shape_error_but_no_mass_displacement() -> None:
    oracle = _oracle()
    assert signed_shape_distance(-oracle, oracle, 1) == pytest.approx(1.0)
    assert temporal_mass_distance(-oracle, oracle) == pytest.approx(0.0)


def test_zero_prediction_has_maximal_shape_and_mass_distance() -> None:
    oracle = _oracle()
    assert signed_shape_distance(np.zeros_like(oracle), oracle) == 1.0
    assert temporal_mass_distance(np.zeros_like(oracle), oracle) == 1.0


def test_interaction_response_recovers_joint_minus_marginals() -> None:
    oracle = _oracle()
    interaction = 0.4 * np.roll(oracle, 2)
    joint = 0.6 * oracle - 0.25 * oracle + interaction
    np.testing.assert_allclose(
        interaction_response(joint, 0.6 * oracle, -0.25 * oracle), interaction
    )
