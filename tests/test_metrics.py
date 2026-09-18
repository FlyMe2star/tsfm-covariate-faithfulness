import numpy as np
import pytest

from covfaith.metrics import (
    directional_sign_agreement,
    extract_forecast_response,
    normalized_response_error,
    placebo_response_ratio,
    response_gain_ratio,
    response_metric_suite,
    temporal_localization_score,
)

ORACLE = np.array([0.0, 0.2, 0.8, 1.0, 0.5, 0.1, 0.0], dtype=np.float64)


def test_oracle_receives_ideal_active_scores() -> None:
    result = response_metric_suite(ORACLE, ORACLE)
    assert result.dsa == 1.0
    assert result.rgr == pytest.approx(1.0, abs=1e-7)
    assert result.nre == 0.0
    assert result.tls == pytest.approx(1.0)


def test_sign_flip_isolated_from_gain_and_localization() -> None:
    predicted = -ORACLE
    assert directional_sign_agreement(predicted, ORACLE) == 0.0
    assert response_gain_ratio(predicted, ORACLE) == pytest.approx(1.0, abs=1e-7)
    assert normalized_response_error(predicted, ORACLE) == pytest.approx(2.0, abs=1e-7)
    assert temporal_localization_score(predicted, ORACLE) == pytest.approx(1.0)


def test_gain_distortion_has_expected_scores() -> None:
    predicted = 0.25 * ORACLE
    assert directional_sign_agreement(predicted, ORACLE) == 1.0
    assert response_gain_ratio(predicted, ORACLE) == pytest.approx(0.25, abs=1e-7)
    assert normalized_response_error(predicted, ORACLE) == pytest.approx(0.75, abs=1e-7)
    assert temporal_localization_score(predicted, ORACLE) == pytest.approx(1.0)


def test_zero_response_fails_active_metrics_cleanly() -> None:
    predicted = np.zeros_like(ORACLE)
    assert directional_sign_agreement(predicted, ORACLE) == 0.0
    assert response_gain_ratio(predicted, ORACLE) == 0.0
    assert normalized_response_error(predicted, ORACLE) == pytest.approx(1.0, abs=1e-7)
    assert temporal_localization_score(predicted, ORACLE) == 0.0


def test_lag_shift_reduces_temporal_localization() -> None:
    shifted = np.roll(ORACLE, 2)
    assert temporal_localization_score(shifted, ORACLE) < 0.75


def test_positive_scale_invariance() -> None:
    predicted = 0.7 * ORACLE
    original = response_metric_suite(predicted, ORACLE)
    scaled = response_metric_suite(9.0 * predicted, 9.0 * ORACLE)
    assert scaled.dsa == original.dsa
    assert scaled.rgr == pytest.approx(original.rgr, abs=1e-7)
    assert scaled.nre == pytest.approx(original.nre, abs=1e-7)
    assert scaled.tls == pytest.approx(original.tls)


def test_forecast_offset_cancels_in_paired_response() -> None:
    factual = np.linspace(2.0, 3.0, ORACLE.size)
    intervened = factual + ORACLE
    response = extract_forecast_response(factual + 100.0, intervened + 100.0)
    np.testing.assert_allclose(response, ORACLE)


def test_placebo_ratio_uses_matched_active_reference() -> None:
    reference = np.ones(8)
    assert placebo_response_ratio(np.zeros(8), reference) == 0.0
    assert placebo_response_ratio(0.25 * np.ones(8), reference) == pytest.approx(0.25, abs=1e-7)


@pytest.mark.parametrize(
    "metric",
    [
        directional_sign_agreement,
        response_gain_ratio,
        normalized_response_error,
        temporal_localization_score,
    ],
)
def test_active_metrics_reject_zero_oracle(metric) -> None:
    with pytest.raises(ValueError):
        metric(np.zeros(4), np.zeros(4))
