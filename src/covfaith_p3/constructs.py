"""Paired semi-synthetic target worlds on a fixed real background window."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import SourceWindow

FloatArray = NDArray[np.float64]
FAMILIES = ("biphasic_rebound", "dispersed_delayed_pulse")


@dataclass(frozen=True)
class P3Scenario:
    dataset: str
    source_id: str
    source_rank: int
    origin: int
    family: str
    seed: int
    target_context_factual: FloatArray
    target_context_intervened: FloatArray
    target_future_factual: FloatArray
    target_future_intervened: FloatArray
    oracle_response: FloatArray
    covariate_context: FloatArray
    covariate_future_factual: FloatArray
    covariate_future_intervened: FloatArray
    placebo_future_intervened: FloatArray
    placebo_oracle_response: FloatArray
    multiplier_cap: float
    parameters: dict[str, float]


def _rng(dataset: str, source_id: str, family: str, seed: int) -> np.random.Generator:
    identity = hashlib.sha256(f"{dataset}|{source_id}|{family}".encode()).digest()
    words = [int.from_bytes(identity[i : i + 4], "big") for i in range(0, 16, 4)]
    return np.random.default_rng(np.random.SeedSequence([seed, *words, 0x5033]))


def _ar1(rng: np.random.Generator, phi: float, length: int) -> FloatArray:
    noise = rng.normal(0, np.sqrt(max(1 - phi**2, 1e-8)), size=length)
    values = np.empty(length, dtype=np.float64)
    values[0] = noise[0]
    for step in range(1, length):
        values[step] = phi * values[step - 1] + noise[step]
    return values


def _kernel(
    family: str, rng: np.random.Generator, config: dict[str, Any]
) -> tuple[FloatArray, dict[str, float]]:
    ranges = config["kernel_parameter_ranges"][family]
    if family == "biphasic_rebound":
        positive_center = float(rng.uniform(*ranges["positive_center"]))
        negative_center = float(rng.uniform(*ranges["negative_center"]))
        rebound_ratio = float(rng.uniform(*ranges["rebound_ratio"]))
        lag = np.arange(int(ranges["kernel_length"]), dtype=np.float64)
        positive = np.exp(-0.5 * ((lag - positive_center) / ranges["positive_lobe_sd"]) ** 2)
        negative = np.exp(-0.5 * ((lag - negative_center) / ranges["negative_lobe_sd"]) ** 2)
        kernel = positive / positive.sum() - rebound_ratio * negative / negative.sum()
        return kernel, {
            "positive_center": positive_center,
            "negative_center": negative_center,
            "rebound_ratio": rebound_ratio,
        }
    if family == "dispersed_delayed_pulse":
        low, high = ranges["delay_onset_inclusive"]
        onset = int(rng.integers(low, high + 1))
        delay_scale = float(rng.uniform(*ranges["delay_scale"]))
        lag = np.arange(int(ranges["kernel_length"]), dtype=np.float64)
        shifted = np.maximum(lag - onset, 0.0)
        kernel = shifted**2 * np.exp(-shifted / delay_scale)
        kernel[: onset + 1] = 0.0
        kernel /= kernel.sum()
        return kernel, {"delay_onset": float(onset), "delay_scale": delay_scale}
    raise ValueError(f"unsupported P3 family: {family}")


def _response(active: FloatArray, kernel: FloatArray, phi_y: float, beta: float) -> FloatArray:
    residual = np.empty_like(active)
    for step in range(active.size):
        width = min(step + 1, kernel.size)
        effect = float(np.dot(kernel[:width], active[step - np.arange(width)]))
        residual[step] = (phi_y * residual[step - 1] if step else 0.0) + beta * effect
    return residual


def _pulse(horizon: int, start: int, length: int) -> FloatArray:
    if start < 0 or length <= 0 or start + length > horizon:
        raise ValueError("pulse must fit the P3 forecast horizon")
    pulse = np.zeros(horizon, dtype=np.float64)
    pulse[start : start + length] = np.sin(np.pi * (np.arange(length) + 1) / (length + 1)) ** 2
    return pulse


def _intervene(
    factual: FloatArray,
    context: FloatArray,
    pulse: FloatArray,
    direction: float,
    config: dict[str, Any],
) -> FloatArray:
    proposed = factual + direction * float(config["intervention_scale_in_active_context_sd"]) * (
        max(float(np.std(context, ddof=1)), 1e-6)
    ) * pulse
    lower, upper = np.quantile(context, config["clip_covariate_to_context_quantiles"])
    result = factual.copy()
    active = pulse > 0
    result[active] = np.clip(proposed[active], lower, upper)
    return result


def _link(background: FloatArray, residual: FloatArray, scale: float, cap: float) -> FloatArray:
    return background * np.exp(np.log(cap) * np.tanh(residual / scale))


def generate_scenario(
    window: SourceWindow,
    family: str,
    seed: int,
    seed_rank: int,
    data: dict[str, Any],
    mechanism: dict[str, Any],
    *,
    multiplier_cap: float = 1.4,
) -> P3Scenario:
    """Generate paired worlds without checkpoint access or target-based reselection."""
    if family not in FAMILIES:
        raise ValueError(f"unsupported P3 family: {family}")
    if multiplier_cap <= 1:
        raise ValueError("multiplier cap must exceed one")
    context_length, horizon = int(data["context_length"]), int(data["horizon"])
    if window.background.shape != (context_length + horizon,):
        raise ValueError("background window length does not match design")
    if not np.all(np.isfinite(window.background)) or np.any(window.background < 0):
        raise ValueError("selected background has nonfinite or negative target values")

    rng = _rng(window.dataset, window.source_id, family, seed)
    burn_in = int(mechanism["burn_in"])
    total = burn_in + context_length + horizon
    phi_x1 = float(rng.uniform(*mechanism["covariate_ar_range"]))
    phi_x2 = float(rng.uniform(*mechanism["covariate_ar_range"]))
    phi_y = float(rng.uniform(*mechanism["response_ar_range"]))
    beta_magnitude = float(rng.uniform(*mechanism["coefficient_magnitude_range"]))
    beta = beta_magnitude if (window.source_rank // 2) % 2 == 0 else -beta_magnitude
    direction = 1.0 if window.source_rank % 2 == 0 else -1.0
    active = _ar1(rng, phi_x1, total)
    placebo = _ar1(rng, phi_x2, total)
    kernel, kernel_parameters = _kernel(family, rng, mechanism)
    split = burn_in + context_length
    future_slice = slice(split, total)
    context_slice = slice(burn_in, split)
    pulse_index = (window.source_rank + seed_rank) % len(mechanism["pulse_lengths"])
    pulse_length = int(mechanism["pulse_lengths"][pulse_index])
    start_index = (window.source_rank + seed_rank) % len(mechanism["start_indices"])
    pulse_start = int(mechanism["start_indices"][start_index])
    pulse = _pulse(horizon, pulse_start, pulse_length)
    active_intervened = _intervene(
        active[future_slice], active[context_slice], pulse, direction, mechanism
    )
    placebo_intervened = _intervene(
        placebo[future_slice], placebo[context_slice], pulse, direction, mechanism
    )

    factual_residual = _response(active, kernel, phi_y, beta)
    intervened_active = active.copy()
    intervened_active[future_slice] = active_intervened
    intervened_residual = _response(intervened_active, kernel, phi_y, beta)
    placebo_residual = _response(active.copy(), kernel, phi_y, beta)
    scale = max(float(np.quantile(np.abs(factual_residual[context_slice]), 0.75)), 1e-6)
    factual_target = _link(window.background, factual_residual[burn_in:], scale, multiplier_cap)
    intervened_target = _link(
        window.background, intervened_residual[burn_in:], scale, multiplier_cap
    )
    placebo_target = _link(window.background, placebo_residual[burn_in:], scale, multiplier_cap)
    factual_future = factual_target[context_length:]
    intervened_future = intervened_target[context_length:]
    covariate_context = np.stack([active[context_slice], placebo[context_slice]])
    covariate_future_factual = np.stack([active[future_slice], placebo[future_slice]])
    covariate_future_intervened = np.stack([active_intervened, placebo[future_slice]])
    placebo_future_intervened = np.stack([active[future_slice], placebo_intervened])
    return P3Scenario(
        dataset=window.dataset,
        source_id=window.source_id,
        source_rank=window.source_rank,
        origin=window.origin,
        family=family,
        seed=seed,
        target_context_factual=factual_target[:context_length].copy(),
        target_context_intervened=intervened_target[:context_length].copy(),
        target_future_factual=factual_future.copy(),
        target_future_intervened=intervened_future.copy(),
        oracle_response=(intervened_future - factual_future).copy(),
        covariate_context=covariate_context,
        covariate_future_factual=covariate_future_factual,
        covariate_future_intervened=covariate_future_intervened,
        placebo_future_intervened=placebo_future_intervened,
        placebo_oracle_response=(placebo_target[context_length:] - factual_future).copy(),
        multiplier_cap=multiplier_cap,
        parameters={
            "phi_x1": phi_x1,
            "phi_x2": phi_x2,
            "phi_y": phi_y,
            "beta": beta,
            "direction": direction,
            "pulse_start": float(pulse_start),
            "pulse_length": float(pulse_length),
            "link_scale": scale,
            **kernel_parameters,
        },
    )


def validate_scenario(scenario: P3Scenario, minimum_ratio: float) -> dict[str, float]:
    """Assert construct validity; return only non-model diagnostic magnitudes."""
    if not np.array_equal(scenario.target_context_factual, scenario.target_context_intervened):
        raise ValueError("paired target contexts differ")
    if not np.array_equal(
        scenario.oracle_response,
        scenario.target_future_intervened - scenario.target_future_factual,
    ):
        raise ValueError("oracle is not the exact paired difference")
    if not np.array_equal(
        scenario.placebo_oracle_response, np.zeros_like(scenario.oracle_response)
    ):
        raise ValueError("placebo must have zero structural response")
    arrays = (
        scenario.target_context_factual,
        scenario.target_future_factual,
        scenario.target_future_intervened,
        scenario.covariate_context,
        scenario.covariate_future_factual,
        scenario.covariate_future_intervened,
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("nonfinite paired scenario")
    if any(np.any(array < 0) for array in arrays[:3]):
        raise ValueError("negative target under multiplicative link")
    if not np.array_equal(
        scenario.covariate_future_factual[1], scenario.covariate_future_intervened[1]
    ):
        raise ValueError("placebo covariate changed in active intervention")
    if not np.array_equal(
        scenario.covariate_future_factual[0], scenario.placebo_future_intervened[0]
    ):
        raise ValueError("active covariate changed in placebo intervention")
    if np.array_equal(
        scenario.covariate_future_factual[1], scenario.placebo_future_intervened[1]
    ):
        raise ValueError("placebo perturbation is numerically zero")
    oracle_l1 = float(np.sum(np.abs(scenario.oracle_response)))
    factual_l1 = float(np.sum(np.abs(scenario.target_future_factual)))
    ratio = oracle_l1 / max(factual_l1, 1e-12)
    if oracle_l1 <= 0 or ratio < minimum_ratio:
        raise ValueError(f"insufficient oracle mass: ratio={ratio:.8g}")
    return {"oracle_l1": oracle_l1, "factual_l1": factual_l1, "oracle_ratio": ratio}
