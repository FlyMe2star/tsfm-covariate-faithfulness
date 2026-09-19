"""Multi-resolution response-shape metrics for the frozen P1-SHAPE audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ShapeMetricResult:
    widths: tuple[int, ...]
    signed_shape_distances: tuple[float, ...]
    hidden_distortion_gap: float
    multi_resolution_auc: float
    temporal_mass_distance: float
    onset_error: float
    peak_time_error: float
    positive_mass_relative_error: float
    negative_mass_relative_error: float


def _paired_1d(predicted: ArrayLike, oracle: ArrayLike) -> tuple[FloatArray, FloatArray]:
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(oracle, dtype=np.float64)
    if pred.ndim != 1 or truth.ndim != 1:
        raise ValueError("predicted and oracle responses must be one-dimensional")
    if pred.shape != truth.shape or pred.size < 2:
        raise ValueError("responses must have the same shape and at least two horizons")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ValueError("responses must be finite")
    if float(np.sum(np.abs(truth))) <= _epsilon(truth):
        raise ValueError("shape metrics are undefined for a zero oracle response")
    return pred, truth


def _epsilon(reference: FloatArray) -> float:
    return 1e-8 * max(1.0, float(np.sum(np.abs(reference))))


def block_signed_response(response: ArrayLike, width: int) -> FloatArray:
    """Sum signed response inside equal contiguous blocks."""

    values = np.asarray(response, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("response must be a finite nonempty vector")
    if width <= 0 or values.size % width:
        raise ValueError("block width must be positive and divide the response length")
    return values.reshape(values.size // width, width).sum(axis=1)


def signed_shape_distance(predicted: ArrayLike, oracle: ArrayLike, width: int = 1) -> float:
    """Total-variation distance between signed L1-normalized block responses."""

    pred, truth = _paired_1d(predicted, oracle)
    pred_blocked = block_signed_response(pred, width)
    truth_blocked = block_signed_response(truth, width)
    truth_mass = float(np.sum(np.abs(truth_blocked)))
    if truth_mass <= _epsilon(truth):
        raise ValueError("oracle response cancels to zero at the requested block width")
    pred_mass = float(np.sum(np.abs(pred_blocked)))
    if pred_mass <= _epsilon(truth):
        return 1.0
    pred_profile = pred_blocked / pred_mass
    truth_profile = truth_blocked / truth_mass
    return float(np.clip(0.5 * np.sum(np.abs(pred_profile - truth_profile)), 0.0, 1.0))


def shape_distance_curve(
    predicted: ArrayLike,
    oracle: ArrayLike,
    widths: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[int, float]:
    if len(widths) < 2 or len(set(widths)) != len(widths):
        raise ValueError("at least two unique resolution widths are required")
    if tuple(sorted(widths)) != widths:
        raise ValueError("resolution widths must be strictly increasing")
    return {width: signed_shape_distance(predicted, oracle, width) for width in widths}


def multi_resolution_auc(curve: dict[int, float]) -> float:
    """Area under distance versus log2 block width, normalized to [0, 1]."""

    widths = np.asarray(sorted(curve), dtype=np.float64)
    distances = np.asarray([curve[int(width)] for width in widths], dtype=np.float64)
    if widths.size < 2 or np.any(widths <= 0):
        raise ValueError("curve requires at least two positive widths")
    if not np.all(np.isfinite(distances)) or np.any((distances < 0) | (distances > 1)):
        raise ValueError("curve distances must be finite values in [0, 1]")
    coordinates = np.log2(widths)
    span = float(coordinates[-1] - coordinates[0])
    if span <= 0:
        raise ValueError("curve widths must span more than one resolution")
    return float(np.trapezoid(distances, coordinates) / span)


def temporal_mass_distance(predicted: ArrayLike, oracle: ArrayLike) -> float:
    """Normalized 1-Wasserstein distance between absolute response-mass profiles."""

    pred, truth = _paired_1d(predicted, oracle)
    pred_mass = float(np.sum(np.abs(pred)))
    if pred_mass <= _epsilon(truth):
        return 1.0
    pred_cdf = np.cumsum(np.abs(pred) / pred_mass)
    truth_cdf = np.cumsum(np.abs(truth) / np.sum(np.abs(truth)))
    return float(np.clip(np.sum(np.abs(pred_cdf[:-1] - truth_cdf[:-1])) / (pred.size - 1), 0, 1))


def _active_onset(response: FloatArray, reference: FloatArray, fraction: float) -> int | None:
    threshold = fraction * float(np.max(np.abs(reference)))
    indices = np.flatnonzero(np.abs(response) >= threshold)
    return int(indices[0]) if indices.size else None


def onset_error(
    predicted: ArrayLike,
    oracle: ArrayLike,
    fraction_of_oracle_peak: float = 0.10,
) -> float:
    pred, truth = _paired_1d(predicted, oracle)
    if not 0 < fraction_of_oracle_peak <= 1:
        raise ValueError("fraction_of_oracle_peak must lie in (0, 1]")
    truth_onset = _active_onset(truth, truth, fraction_of_oracle_peak)
    pred_onset = _active_onset(pred, truth, fraction_of_oracle_peak)
    assert truth_onset is not None
    if pred_onset is None:
        return 1.0
    return abs(pred_onset - truth_onset) / (truth.size - 1)


def peak_time_error(predicted: ArrayLike, oracle: ArrayLike) -> float:
    pred, truth = _paired_1d(predicted, oracle)
    if float(np.sum(np.abs(pred))) <= _epsilon(truth):
        return 1.0
    pred_peak = int(np.argmax(np.abs(pred)))
    truth_peak = int(np.argmax(np.abs(truth)))
    return abs(pred_peak - truth_peak) / (truth.size - 1)


def signed_mass_relative_errors(predicted: ArrayLike, oracle: ArrayLike) -> tuple[float, float]:
    pred, truth = _paired_1d(predicted, oracle)
    errors: list[float] = []
    for positive in (True, False):
        truth_mass = (
            float(np.sum(truth[truth > 0]))
            if positive
            else float(-np.sum(truth[truth < 0]))
        )
        pred_mass = (
            float(np.sum(pred[pred > 0]))
            if positive
            else float(-np.sum(pred[pred < 0]))
        )
        if truth_mass <= _epsilon(truth):
            errors.append(float("nan"))
        else:
            errors.append(abs(pred_mass - truth_mass) / (truth_mass + _epsilon(truth)))
    return errors[0], errors[1]


def interaction_response(
    joint: ArrayLike,
    marginal_a: ArrayLike,
    marginal_b: ArrayLike,
) -> FloatArray:
    joint_array = np.asarray(joint, dtype=np.float64)
    a_array = np.asarray(marginal_a, dtype=np.float64)
    b_array = np.asarray(marginal_b, dtype=np.float64)
    if (
        joint_array.ndim != 1
        or joint_array.shape != a_array.shape
        or joint_array.shape != b_array.shape
    ):
        raise ValueError("joint and marginal responses must be aligned vectors")
    if not all(np.all(np.isfinite(value)) for value in (joint_array, a_array, b_array)):
        raise ValueError("interaction inputs must be finite")
    return joint_array - a_array - b_array


def shape_metric_suite(
    predicted: ArrayLike,
    oracle: ArrayLike,
    widths: tuple[int, ...] = (1, 2, 4, 8),
) -> ShapeMetricResult:
    curve = shape_distance_curve(predicted, oracle, widths)
    positive_error, negative_error = signed_mass_relative_errors(predicted, oracle)
    return ShapeMetricResult(
        widths=widths,
        signed_shape_distances=tuple(curve[width] for width in widths),
        hidden_distortion_gap=curve[widths[0]] - curve[widths[-1]],
        multi_resolution_auc=multi_resolution_auc(curve),
        temporal_mass_distance=temporal_mass_distance(predicted, oracle),
        onset_error=onset_error(predicted, oracle),
        peak_time_error=peak_time_error(predicted, oracle),
        positive_mass_relative_error=positive_error,
        negative_mass_relative_error=negative_error,
    )
