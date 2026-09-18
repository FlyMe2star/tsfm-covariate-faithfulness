"""Response-faithfulness metrics with explicit zero-effect handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ResponseMetricResult:
    dsa: float
    rgr: float
    nre: float
    tls: float
    active_count: int


def _paired_1d(predicted: ArrayLike, oracle: ArrayLike) -> tuple[FloatArray, FloatArray]:
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(oracle, dtype=np.float64)
    if pred.ndim != 1 or truth.ndim != 1:
        raise ValueError("predicted and oracle responses must be one-dimensional")
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("predicted and oracle responses must have the same nonempty shape")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ValueError("responses must be finite")
    return pred, truth


def _epsilon(reference: FloatArray, multiplier: float = 1e-8) -> float:
    return multiplier * max(1.0, float(np.linalg.norm(reference, ord=1)))


def active_support(oracle: ArrayLike, fraction_of_peak: float = 0.10) -> BoolArray:
    """Return the predeclared support mask relative to the oracle peak magnitude."""

    truth = np.asarray(oracle, dtype=np.float64)
    if truth.ndim != 1 or truth.size == 0 or not np.all(np.isfinite(truth)):
        raise ValueError("oracle response must be a finite nonempty vector")
    if not 0.0 < fraction_of_peak <= 1.0:
        raise ValueError("fraction_of_peak must lie in (0, 1]")
    peak = float(np.max(np.abs(truth)))
    if peak <= _epsilon(truth):
        raise ValueError("active-support metrics are undefined for a zero oracle response")
    return np.abs(truth) >= fraction_of_peak * peak


def directional_sign_agreement(
    predicted: ArrayLike,
    oracle: ArrayLike,
    fraction_of_peak: float = 0.10,
) -> float:
    pred, truth = _paired_1d(predicted, oracle)
    support = active_support(truth, fraction_of_peak)
    tolerance = 1e-8 * max(1.0, float(np.max(np.abs(truth))))
    pred_sign = np.sign(np.where(np.abs(pred) <= tolerance, 0.0, pred))
    truth_sign = np.sign(truth)
    return float(np.mean(pred_sign[support] == truth_sign[support]))


def response_gain_ratio(
    predicted: ArrayLike,
    oracle: ArrayLike,
    fraction_of_peak: float = 0.10,
) -> float:
    pred, truth = _paired_1d(predicted, oracle)
    support = active_support(truth, fraction_of_peak)
    denominator = float(np.sum(np.abs(truth[support]))) + _epsilon(truth[support])
    return float(np.sum(np.abs(pred[support])) / denominator)


def normalized_response_error(
    predicted: ArrayLike,
    oracle: ArrayLike,
    fraction_of_peak: float = 0.10,
) -> float:
    pred, truth = _paired_1d(predicted, oracle)
    support = active_support(truth, fraction_of_peak)
    denominator = float(np.sum(np.abs(truth[support]))) + _epsilon(truth[support])
    return float(np.sum(np.abs(pred[support] - truth[support])) / denominator)


def temporal_localization_score(predicted: ArrayLike, oracle: ArrayLike) -> float:
    pred, truth = _paired_1d(predicted, oracle)
    truth_mass = float(np.sum(np.abs(truth)))
    pred_mass = float(np.sum(np.abs(pred)))
    tolerance = _epsilon(truth)
    if truth_mass <= tolerance:
        raise ValueError("temporal localization is undefined for a zero oracle response")
    if pred_mass <= tolerance:
        return 0.0
    truth_profile = np.abs(truth) / truth_mass
    pred_profile = np.abs(pred) / pred_mass
    score = 1.0 - 0.5 * float(np.sum(np.abs(truth_profile - pred_profile)))
    return float(np.clip(score, 0.0, 1.0))


def placebo_response_ratio(placebo_predicted: ArrayLike, matched_active_oracle: ArrayLike) -> float:
    placebo = np.asarray(placebo_predicted, dtype=np.float64)
    reference = np.asarray(matched_active_oracle, dtype=np.float64)
    if placebo.ndim != 1 or reference.ndim != 1 or placebo.shape != reference.shape:
        raise ValueError("placebo and matched active responses must be aligned vectors")
    if placebo.size == 0 or not np.all(np.isfinite(placebo)) or not np.all(np.isfinite(reference)):
        raise ValueError("placebo and matched active responses must be finite and nonempty")
    reference_mass = float(np.sum(np.abs(reference)))
    if reference_mass <= _epsilon(reference):
        raise ValueError("matched active oracle response must be nonzero")
    return float(np.sum(np.abs(placebo)) / (reference_mass + _epsilon(reference)))


def extract_forecast_response(
    factual_forecast: ArrayLike,
    intervened_forecast: ArrayLike,
) -> FloatArray:
    factual = np.asarray(factual_forecast, dtype=np.float64)
    intervened = np.asarray(intervened_forecast, dtype=np.float64)
    if factual.shape != intervened.shape or factual.ndim != 1 or factual.size == 0:
        raise ValueError("factual and intervened forecasts must be aligned nonempty vectors")
    if not np.all(np.isfinite(factual)) or not np.all(np.isfinite(intervened)):
        raise ValueError("forecasts must be finite")
    return intervened - factual


def response_metric_suite(
    predicted: ArrayLike,
    oracle: ArrayLike,
    fraction_of_peak: float = 0.10,
) -> ResponseMetricResult:
    support = active_support(oracle, fraction_of_peak)
    return ResponseMetricResult(
        dsa=directional_sign_agreement(predicted, oracle, fraction_of_peak),
        rgr=response_gain_ratio(predicted, oracle, fraction_of_peak),
        nre=normalized_response_error(predicted, oracle, fraction_of_peak),
        tls=temporal_localization_score(predicted, oracle),
        active_count=int(np.sum(support)),
    )
