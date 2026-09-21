from __future__ import annotations

import numpy as np
import pytest

from covfaith_refs.p1_ref import _write_csv_atomic, verify_reference_freeze
from covfaith_refs.transparent import (
    MODEL_IDS,
    build_training_matrix,
    fit_reference,
    forecast_reference,
    residual_index_paths,
)


def _fixture_context() -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(192, dtype=np.float64)
    x1 = np.sin(time / 7.0) + 0.15 * np.cos(time / 3.0)
    x2 = np.cos(time / 11.0) - 0.1 * np.sin(time / 5.0)
    target = np.zeros_like(time)
    for index in range(1, time.size):
        target[index] = (
            0.55 * target[index - 1]
            + 0.35 * x1[index]
            - 0.20 * x2[index]
            + 0.08 * np.sin(index)
        )
    return target, np.stack([x1, x2])


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_training_matrix_is_context_only_and_finite(model_id: str) -> None:
    target, covariates = _fixture_context()
    matrix, response, names, scales = build_training_matrix(target, covariates, model_id)
    assert matrix.shape[0] == target.size - 24
    assert matrix.shape[1] == len(names)
    assert response.shape == (target.size - 24,)
    assert scales.shape == (2,)
    assert np.all(np.isfinite(matrix))


def test_registered_feature_dimensions_are_stable() -> None:
    target, covariates = _fixture_context()
    ridge = build_training_matrix(target, covariates, "ridge_arx")[0]
    nonlinear = build_training_matrix(target, covariates, "nonlinear_dynamic_regression")[0]
    assert ridge.shape[1] == 32
    assert nonlinear.shape[1] == 106


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_fit_and_paired_residual_replay_are_finite(model_id: str) -> None:
    target, covariates = _fixture_context()
    fitted = fit_reference(target, covariates, model_id)
    assert fitted.residuals.shape == (168,)
    assert np.isfinite(fitted.condition_number)
    indices = residual_index_paths(fitted.residuals.size, 32, 24, 200926)
    future_time = np.arange(192, 216, dtype=np.float64)
    factual = np.stack([np.sin(future_time / 7.0), np.cos(future_time / 11.0)])
    intervened = factual.copy()
    intervened[0, 6:12] += 0.5
    factual_forecast = forecast_reference(
        fitted, target, covariates, factual, residual_indices=indices
    )
    repeated = forecast_reference(fitted, target, covariates, factual, residual_indices=indices)
    intervention_forecast = forecast_reference(
        fitted, target, covariates, intervened, residual_indices=indices
    )
    np.testing.assert_array_equal(factual_forecast.paths, repeated.paths)
    assert factual_forecast.quantiles.shape == (24, 9)
    assert np.any(np.abs(intervention_forecast.point - factual_forecast.point) > 1e-10)


def test_invalid_model_is_rejected() -> None:
    target, covariates = _fixture_context()
    with pytest.raises(ValueError, match="unsupported reference model"):
        build_training_matrix(target, covariates, "not_registered")


def test_reference_freeze_preserves_primary_hash() -> None:
    hashes = verify_reference_freeze(".")
    assert hashes["config_hash"] == (
        "330e2317ffb7dcead45e4ef83bf830388ed835b1eb887ad59ae54eb7e5a22bb5"
    )
    assert hashes["scientific_code_sha256"] == (
        "a921a7858c663bb17e93c06398393ab4a5c3193b71c76dbaccd4f75d0bcf0ed5"
    )
    assert hashes["primary_scientific_code_sha256"] == (
        "fc1cf6d73a59a4200deaa964f341a17b72d8fae75fa40b1086ed1c8c714f5925"
    )


def test_csv_writer_unions_optional_summary_fields(tmp_path) -> None:
    path = tmp_path / "matrix.csv"
    _write_csv_atomic(path, [{"model": "a", "core": 1}, {"model": "b", "extra": 2}])
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "model,core,extra"
    assert "b,,2" in text
