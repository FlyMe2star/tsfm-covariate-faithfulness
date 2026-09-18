"""Forecast-skill metrics used as the complement to response fidelity."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _forecast_arrays(
    target: ArrayLike,
    quantiles: ArrayLike,
    quantile_levels: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    truth = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(quantiles, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if truth.ndim != 1 or predicted.ndim != 2 or levels.ndim != 1:
        raise ValueError("expected target [H], quantiles [H,Q], and levels [Q]")
    if predicted.shape != (truth.size, levels.size) or truth.size == 0:
        raise ValueError("forecast arrays have incompatible shapes")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(predicted)):
        raise ValueError("target and forecasts must be finite")
    if not np.all((levels > 0.0) & (levels < 1.0)) or np.any(np.diff(levels) <= 0):
        raise ValueError("quantile levels must be strictly increasing inside (0,1)")
    return truth, predicted, levels


def pinball_loss(
    target: ArrayLike,
    quantiles: ArrayLike,
    quantile_levels: ArrayLike,
) -> FloatArray:
    """Return horizon-by-quantile pinball losses."""

    truth, predicted, levels = _forecast_arrays(target, quantiles, quantile_levels)
    error = truth[:, None] - predicted
    return np.maximum(levels[None, :] * error, (levels[None, :] - 1.0) * error)


def scaled_quantile_loss(
    target: ArrayLike,
    quantiles: ArrayLike,
    quantile_levels: ArrayLike,
    target_context: ArrayLike,
) -> float:
    """Twice mean pinball loss scaled by context first-difference MAE."""

    context = np.asarray(target_context, dtype=np.float64)
    if context.ndim != 1 or context.size < 2 or not np.all(np.isfinite(context)):
        raise ValueError("target context must be a finite vector of length at least two")
    scale = float(np.mean(np.abs(np.diff(context))))
    epsilon = 1e-8 * max(1.0, float(np.mean(np.abs(context))))
    mean_loss = float(np.mean(pinball_loss(target, quantiles, quantile_levels)))
    return 2.0 * mean_loss / (scale + epsilon)


def weighted_quantile_loss(
    target: ArrayLike,
    quantiles: ArrayLike,
    quantile_levels: ArrayLike,
) -> float:
    """Twice summed pinball loss normalized by target absolute mass and Q."""

    truth = np.asarray(target, dtype=np.float64)
    losses = pinball_loss(truth, quantiles, quantile_levels)
    denominator = losses.shape[1] * float(np.sum(np.abs(truth)))
    epsilon = 1e-8 * max(1.0, denominator)
    return float(2.0 * np.sum(losses) / (denominator + epsilon))


def median_mae(target: ArrayLike, median_forecast: ArrayLike) -> float:
    truth = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(median_forecast, dtype=np.float64)
    if truth.ndim != 1 or truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("target and median forecast must be aligned vectors")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(predicted)):
        raise ValueError("target and median forecast must be finite")
    return float(np.mean(np.abs(truth - predicted)))
