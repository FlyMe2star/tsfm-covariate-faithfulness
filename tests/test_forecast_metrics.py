import numpy as np
import pytest

from covfaith.forecast_metrics import (
    median_mae,
    pinball_loss,
    scaled_quantile_loss,
    weighted_quantile_loss,
)


def test_perfect_quantiles_have_zero_loss() -> None:
    target = np.array([1.0, 2.0, 3.0])
    levels = np.array([0.1, 0.5, 0.9])
    quantiles = np.repeat(target[:, None], levels.size, axis=1)
    np.testing.assert_array_equal(pinball_loss(target, quantiles, levels), 0.0)
    assert scaled_quantile_loss(target, quantiles, levels, np.array([0.0, 1.0, 2.0])) == 0.0
    assert weighted_quantile_loss(target, quantiles, levels) == 0.0
    assert median_mae(target, target) == 0.0


def test_symmetric_median_pinball_loss() -> None:
    target = np.array([0.0])
    levels = np.array([0.5])
    assert pinball_loss(target, np.array([[2.0]]), levels)[0, 0] == pytest.approx(1.0)
    assert pinball_loss(target, np.array([[-2.0]]), levels)[0, 0] == pytest.approx(1.0)


def test_scaled_loss_is_invariant_to_positive_target_scale() -> None:
    context = np.array([0.0, 1.0, 0.0, 2.0])
    target = np.array([1.0, 2.0])
    levels = np.array([0.1, 0.5, 0.9])
    quantiles = np.array([[0.5, 1.2, 1.6], [1.4, 2.3, 2.8]])
    original = scaled_quantile_loss(target, quantiles, levels, context)
    scaled = scaled_quantile_loss(7.0 * target, 7.0 * quantiles, levels, 7.0 * context)
    assert scaled == pytest.approx(original, abs=1e-7)
