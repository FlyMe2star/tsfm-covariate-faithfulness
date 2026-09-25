from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from covfaith_p3.constructs import FAMILIES, generate_scenario, validate_scenario
from covfaith_p3.data import SourceSeries, SourceWindow, eligible_origins, select_windows


def _selection_config() -> tuple[dict, dict]:
    data = {
        "context_length": 192,
        "horizon": 24,
        "origin_region_start_fraction": 0.60,
        "origin_stride": 100,
        "context_standard_deviation_min": 1e-6,
        "require_nominal_cadence": True,
        "source_ids_per_dataset": 1,
        "selection_salt": "test-salt",
    }
    source = {
        "id": "traffic_daily",
        "frequency_seconds": 86400,
        "context_positive_fraction_min": 0.95,
    }
    return data, source


def _series() -> SourceSeries:
    index = np.arange(240)
    target = 100 + 5 * np.sin(index / 8)
    timestamps = np.datetime64("2020-01-01") + index.astype("timedelta64[D]")
    return SourceSeries("test-road", target.astype(np.float64), timestamps.astype("datetime64[ms]"))


def _window() -> SourceWindow:
    item = _series()
    return SourceWindow(
        dataset="traffic_daily",
        source_id=item.source_id,
        source_rank=0,
        origin=192,
        background=item.target[:216].copy(),
        timestamps=item.timestamps[:216].copy(),
    )


def _mechanism() -> dict:
    return {
        "families": list(FAMILIES),
        "covariate_ar_range": [0.25, 0.72],
        "response_ar_range": [0.20, 0.58],
        "coefficient_magnitude_range": [0.35, 0.85],
        "burn_in": 96,
        "intervention_scale_in_active_context_sd": 0.65,
        "pulse_lengths": [3, 5, 7],
        "start_indices": [1, 7, 13],
        "clip_covariate_to_context_quantiles": [0.005, 0.995],
        "kernel_parameter_ranges": {
            "biphasic_rebound": {
                "positive_center": [0.5, 1.5],
                "negative_center": [5.5, 8.0],
                "rebound_ratio": [0.45, 0.80],
                "positive_lobe_sd": 1.15,
                "negative_lobe_sd": 1.45,
                "kernel_length": 11,
            },
            "dispersed_delayed_pulse": {
                "delay_onset_inclusive": [1, 3],
                "delay_scale": [0.8, 1.8],
                "kernel_length": 12,
            },
        },
    }


def test_origin_selection_does_not_use_future_target_values() -> None:
    data, source = _selection_config()
    original = _series()
    altered = replace(original, target=original.target.copy())
    altered.target[192:216] = np.nan
    assert eligible_origins(original, data, source) == [192]
    assert eligible_origins(altered, data, source) == [192]
    first, count = select_windows([original], data, source)
    second, altered_count = select_windows([altered], data, source)
    assert count == altered_count == 1
    assert first[0].origin == second[0].origin == 192
    np.testing.assert_array_equal(first[0].background[:192], second[0].background[:192])


def test_future_timestamps_must_still_have_nominal_cadence() -> None:
    data, source = _selection_config()
    original = _series()
    altered = replace(original, timestamps=original.timestamps.copy())
    altered.timestamps[200] += np.timedelta64(1, "h")
    assert eligible_origins(altered, data, source) == []


@pytest.mark.parametrize("family", FAMILIES)
def test_construct_is_paired_deterministic_and_placebo_is_zero(family: str) -> None:
    data, _ = _selection_config()
    window = _window()
    first = generate_scenario(window, family, 401, 0, data, _mechanism())
    second = generate_scenario(window, family, 401, 0, data, _mechanism())
    np.testing.assert_array_equal(first.target_context_factual, first.target_context_intervened)
    np.testing.assert_array_equal(first.oracle_response, second.oracle_response)
    np.testing.assert_array_equal(first.placebo_oracle_response, 0)
    np.testing.assert_array_equal(
        first.oracle_response, first.target_future_intervened - first.target_future_factual
    )
    assert first.target_future_factual.shape == (24,)
    assert first.covariate_context.shape == (2, 192)
    assert validate_scenario(first, 0.0)["oracle_ratio"] > 0
    np.testing.assert_array_equal(window.background, _window().background)


def test_construct_rejects_nonfinite_future_after_selection() -> None:
    data, _ = _selection_config()
    window = _window()
    bad = replace(window, background=window.background.copy())
    bad.background[-1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        generate_scenario(bad, FAMILIES[0], 401, 0, data, _mechanism())


def test_lower_link_strength_does_not_reselect_or_change_covariates() -> None:
    data, _ = _selection_config()
    window = _window()
    main = generate_scenario(window, FAMILIES[0], 401, 0, data, _mechanism())
    lower = generate_scenario(
        window, FAMILIES[0], 401, 0, data, _mechanism(), multiplier_cap=1.2
    )
    np.testing.assert_array_equal(main.covariate_context, lower.covariate_context)
    np.testing.assert_array_equal(
        main.covariate_future_intervened, lower.covariate_future_intervened
    )
    assert not np.array_equal(main.target_future_factual, lower.target_future_factual)
