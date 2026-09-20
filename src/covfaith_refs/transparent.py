"""Transparent fitted references for the post-primary P1-SHAPE supplement.

This module deliberately lives outside :mod:`covfaith`.  The archived P1 scientific
hash covers every Python file in that package, so keeping supplementary code here
preserves the primary evidence chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

MODEL_IDS = ("ridge_arx", "nonlinear_dynamic_regression")
TARGET_LAGS = (1, 2, 3, 6, 12, 24)
COVARIATE_LAGS = tuple(range(13))
INTERACTION_LAGS = (0, 1, 2, 4, 8, 12)
SIGNED_DIFFERENCE_LAGS = (0, 1, 2, 3)
DEFAULT_QUANTILE_LEVELS = tuple(float(value) for value in np.arange(0.1, 1.0, 0.1))


@dataclass(frozen=True)
class FittedReference:
    """A context-only ridge fit and the state required for recursive forecasts."""

    model_id: str
    alpha: float
    feature_names: tuple[str, ...]
    feature_mean: FloatArray
    feature_scale: FloatArray
    target_mean: float
    target_scale: float
    coefficients: FloatArray
    residuals: FloatArray
    covariate_scale: FloatArray
    condition_number: float


@dataclass(frozen=True)
class ReferenceForecast:
    """Point, sampled, and quantile forecasts for one future-covariate world."""

    point: FloatArray
    paths: FloatArray
    quantiles: FloatArray
    quantile_levels: FloatArray


def _as_context(target: ArrayLike, covariates: ArrayLike) -> tuple[FloatArray, FloatArray]:
    y = np.asarray(target, dtype=np.float64)
    x = np.asarray(covariates, dtype=np.float64)
    if y.ndim != 1 or y.size <= max(TARGET_LAGS):
        raise ValueError("target context must be one-dimensional and longer than 24")
    if x.ndim != 2 or x.shape != (2, y.size):
        raise ValueError("covariate context must have shape [2, context_length]")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        raise ValueError("contexts must be finite")
    return y, x


def _feature_names(model_id: str) -> tuple[str, ...]:
    if model_id not in MODEL_IDS:
        raise ValueError(f"unsupported reference model {model_id!r}")
    names = [*(f"y_lag_{lag}" for lag in TARGET_LAGS)]
    names.extend(
        f"x{channel + 1}_lag_{lag}" for channel in range(2) for lag in COVARIATE_LAGS
    )
    if model_id == "nonlinear_dynamic_regression":
        names.extend(
            f"x{channel + 1}_tanh_lag_{lag}"
            for channel in range(2)
            for lag in COVARIATE_LAGS
        )
        names.extend(
            f"x{channel + 1}_square_lag_{lag}"
            for channel in range(2)
            for lag in COVARIATE_LAGS
        )
        names.extend(f"x1_x2_lag_{lag}" for lag in INTERACTION_LAGS)
        names.extend(
            f"x{channel + 1}_{part}_diff_lag_{lag}"
            for channel in range(2)
            for lag in SIGNED_DIFFERENCE_LAGS
            for part in ("positive", "negative")
        )
    return tuple(names)


def _feature_row(
    target_history: FloatArray,
    covariate_history: FloatArray,
    index: int,
    model_id: str,
    covariate_scale: FloatArray,
) -> FloatArray:
    if index < max(TARGET_LAGS):
        raise ValueError("feature index must provide all frozen lags")
    values = [float(target_history[index - lag]) for lag in TARGET_LAGS]
    for channel in range(2):
        values.extend(float(covariate_history[channel, index - lag]) for lag in COVARIATE_LAGS)
    if model_id == "nonlinear_dynamic_regression":
        scaled = covariate_history / covariate_scale[:, None]
        for channel in range(2):
            values.extend(float(np.tanh(scaled[channel, index - lag])) for lag in COVARIATE_LAGS)
        for channel in range(2):
            values.extend(float(scaled[channel, index - lag] ** 2) for lag in COVARIATE_LAGS)
        values.extend(
            float(scaled[0, index - lag] * scaled[1, index - lag])
            for lag in INTERACTION_LAGS
        )
        for channel in range(2):
            for lag in SIGNED_DIFFERENCE_LAGS:
                difference = float(
                    scaled[channel, index - lag] - scaled[channel, index - lag - 1]
                )
                values.extend((max(difference, 0.0), max(-difference, 0.0)))
    row = np.asarray(values, dtype=np.float64)
    expected = len(_feature_names(model_id))
    if row.shape != (expected,) or not np.all(np.isfinite(row)):
        raise RuntimeError("reference feature construction failed")
    return row


def build_training_matrix(
    target_context: ArrayLike,
    covariate_context: ArrayLike,
    model_id: str,
) -> tuple[FloatArray, FloatArray, tuple[str, ...], FloatArray]:
    """Build context-only one-step rows for a frozen reference specification."""

    target, covariates = _as_context(target_context, covariate_context)
    names = _feature_names(model_id)
    covariate_scale = np.std(covariates, axis=1, ddof=1)
    covariate_scale = np.where(covariate_scale > 1e-12, covariate_scale, 1.0)
    indices = range(max(TARGET_LAGS), target.size)
    matrix = np.stack(
        [
            _feature_row(target, covariates, index, model_id, covariate_scale)
            for index in indices
        ]
    )
    response = target[max(TARGET_LAGS) :].copy()
    return matrix, response, names, covariate_scale.astype(np.float64, copy=False)


def fit_reference(
    target_context: ArrayLike,
    covariate_context: ArrayLike,
    model_id: str,
    *,
    alpha: float = 1.0,
) -> FittedReference:
    """Fit one fixed-basis ridge model without validation or future-target access."""

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    matrix, response, names, covariate_scale = build_training_matrix(
        target_context, covariate_context, model_id
    )
    feature_mean = np.mean(matrix, axis=0)
    feature_scale = np.std(matrix, axis=0, ddof=1)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    target_mean = float(np.mean(response))
    target_scale = float(np.std(response, ddof=1))
    if target_scale <= 1e-12:
        target_scale = 1.0
    standardized = (matrix - feature_mean) / feature_scale
    standardized_target = (response - target_mean) / target_scale
    design = np.column_stack([np.ones(standardized.shape[0]), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    coefficients = np.linalg.solve(gram, design.T @ standardized_target)
    fitted = target_mean + target_scale * (design @ coefficients)
    residuals = response - fitted
    residuals = residuals - float(np.mean(residuals))
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(residuals)):
        raise RuntimeError("reference fit produced non-finite values")
    return FittedReference(
        model_id=model_id,
        alpha=float(alpha),
        feature_names=names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        coefficients=coefficients,
        residuals=residuals,
        covariate_scale=covariate_scale,
        condition_number=float(np.linalg.cond(gram)),
    )


def residual_index_paths(
    residual_count: int,
    path_count: int,
    horizon: int,
    seed: int,
) -> IntArray:
    """Create reusable residual indices for exact paired-world replay."""

    if residual_count <= 0 or path_count <= 0 or horizon <= 0:
        raise ValueError("residual, path, and horizon counts must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, residual_count, size=(path_count, horizon), dtype=np.int64)


def _predict_one(
    fitted: FittedReference,
    target_history: FloatArray,
    covariate_history: FloatArray,
    index: int,
) -> float:
    row = _feature_row(
        target_history,
        covariate_history,
        index,
        fitted.model_id,
        fitted.covariate_scale,
    )
    standardized = (row - fitted.feature_mean) / fitted.feature_scale
    design = np.concatenate([[1.0], standardized])
    return float(fitted.target_mean + fitted.target_scale * (design @ fitted.coefficients))


def forecast_reference(
    fitted: FittedReference,
    target_context: ArrayLike,
    covariate_context: ArrayLike,
    covariate_future: ArrayLike,
    *,
    residual_indices: ArrayLike,
    quantile_levels: ArrayLike = DEFAULT_QUANTILE_LEVELS,
) -> ReferenceForecast:
    """Recursively forecast one known-future-covariate world."""

    target, past_covariates = _as_context(target_context, covariate_context)
    future_covariates = np.asarray(covariate_future, dtype=np.float64)
    if future_covariates.ndim != 2 or future_covariates.shape[0] != 2:
        raise ValueError("future covariates must have shape [2, horizon]")
    if not np.all(np.isfinite(future_covariates)):
        raise ValueError("future covariates must be finite")
    indices = np.asarray(residual_indices, dtype=np.int64)
    horizon = future_covariates.shape[1]
    if indices.ndim != 2 or indices.shape[1] != horizon:
        raise ValueError("residual indices must have shape [paths, horizon]")
    if np.any(indices < 0) or np.any(indices >= fitted.residuals.size):
        raise ValueError("residual index out of range")
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if levels.ndim != 1 or np.any(levels <= 0) or np.any(levels >= 1):
        raise ValueError("quantile levels must be inside (0, 1)")
    if np.any(np.diff(levels) <= 0):
        raise ValueError("quantile levels must be strictly increasing")

    all_covariates = np.concatenate([past_covariates, future_covariates], axis=1)
    point_history = np.empty(target.size + horizon, dtype=np.float64)
    point_history[: target.size] = target
    for step in range(horizon):
        index = target.size + step
        point_history[index] = _predict_one(fitted, point_history, all_covariates, index)
    point = point_history[target.size :].copy()

    paths = np.empty((indices.shape[0], horizon), dtype=np.float64)
    for path_index in range(indices.shape[0]):
        history = np.empty(target.size + horizon, dtype=np.float64)
        history[: target.size] = target
        for step in range(horizon):
            index = target.size + step
            conditional_mean = _predict_one(fitted, history, all_covariates, index)
            history[index] = conditional_mean + fitted.residuals[indices[path_index, step]]
        paths[path_index] = history[target.size :]
    quantiles = np.quantile(paths, levels, axis=0).T
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(quantiles)):
        raise RuntimeError("reference forecast produced non-finite values")
    return ReferenceForecast(
        point=point,
        paths=paths,
        quantiles=quantiles,
        quantile_levels=levels,
    )
