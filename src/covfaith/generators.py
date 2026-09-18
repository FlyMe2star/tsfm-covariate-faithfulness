"""Deterministic paired structural generators for CovIntervene."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

MECHANISMS = (
    "contemporaneous_linear",
    "delayed_distributed_lag",
    "threshold_saturation",
    "irrelevant_placebo",
)


@dataclass(frozen=True)
class GeneratorSpec:
    context_length: int = 192
    horizon: int = 24
    target_ar_range: tuple[float, float] = (0.25, 0.65)
    covariate_ar_range: tuple[float, float] = (0.35, 0.80)
    coefficient_magnitude_range: tuple[float, float] = (0.40, 1.00)
    innovation_sd_range: tuple[float, float] = (0.15, 0.35)
    intervention_scale: float = 0.75
    block_length: int = 6
    start_indices: tuple[int, ...] = (0, 6, 12)
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    first_lag: int = 2
    lag_kernel: tuple[float, ...] = (0.50, 0.30, 0.20)
    transition_width: float = 0.50
    burn_in: int = 64


@dataclass(frozen=True)
class PairedScenario:
    """One paired factual/intervened structural evaluation unit."""

    mechanism: str
    generator_seed: int
    series_index: int
    covariate_names: tuple[str, ...]
    intervention_covariate: str
    target_context: FloatArray
    target_future_factual: FloatArray
    target_future_intervened: FloatArray
    covariate_context: FloatArray
    covariate_future_factual: FloatArray
    covariate_future_intervened: FloatArray
    oracle_response: FloatArray
    matched_active_oracle_response: FloatArray
    intervention_mask: BoolArray
    parameters: Mapping[str, float]

    @property
    def series_id(self) -> str:
        return f"seed{self.generator_seed:05d}-series{self.series_index:04d}"


def _series_rng(generator_seed: int, series_index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(generator_seed), int(series_index), 0xC0FA17])
    return np.random.default_rng(sequence)


def _ar1(rng: np.random.Generator, phi: float, length: int) -> FloatArray:
    innovations = rng.normal(0.0, np.sqrt(max(1.0 - phi**2, 1e-8)), size=length)
    values = np.empty(length, dtype=np.float64)
    values[0] = innovations[0]
    for t in range(1, length):
        values[t] = phi * values[t - 1] + innovations[t]
    return values


def _smooth_block(horizon: int, start: int, length: int) -> tuple[FloatArray, BoolArray]:
    if start < 0 or start >= horizon:
        raise ValueError("intervention start must fall inside the forecast horizon")
    stop = min(start + length, horizon)
    width = stop - start
    window = np.zeros(horizon, dtype=np.float64)
    window[start:stop] = np.sin(np.pi * (np.arange(width) + 1) / (width + 1))
    return window, window > 0


def _target_path(
    mechanism: str,
    active: FloatArray,
    noise: FloatArray,
    phi_y: float,
    beta: float,
    *,
    first_lag: int,
    lag_kernel: FloatArray,
    threshold_center: float,
    threshold_width: float,
) -> FloatArray:
    target = np.zeros_like(active, dtype=np.float64)
    for t in range(active.size):
        previous = target[t - 1] if t else 0.0
        if mechanism == "delayed_distributed_lag":
            effect = 0.0
            for offset, weight in enumerate(lag_kernel):
                source = t - first_lag - offset
                if source >= 0:
                    effect += float(weight) * active[source]
        elif mechanism == "threshold_saturation":
            effect = np.tanh((active[t] - threshold_center) / threshold_width)
        else:
            effect = active[t]
        target[t] = phi_y * previous + beta * effect + noise[t]
    return target


def _intervene(
    factual: FloatArray,
    context: FloatArray,
    window: FloatArray,
    direction: float,
    scale: float,
    clip_quantiles: tuple[float, float],
) -> FloatArray:
    context_sd = max(float(np.std(context, ddof=1)), 1e-6)
    proposed = factual + direction * scale * context_sd * window
    lower, upper = np.quantile(context, clip_quantiles)
    intervened = factual.copy()
    changed = window > 0
    intervened[changed] = np.clip(proposed[changed], lower, upper)
    return intervened.astype(np.float64, copy=False)


def generate_scenario(
    mechanism: str,
    generator_seed: int,
    series_index: int,
    spec: GeneratorSpec | None = None,
) -> PairedScenario:
    """Generate one deterministic paired scenario with an analytic oracle response."""

    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}; expected one of {MECHANISMS}")
    spec = spec or GeneratorSpec()
    if spec.context_length <= 0 or spec.horizon <= 0:
        raise ValueError("context_length and horizon must be positive")

    rng = _series_rng(generator_seed, series_index)
    total = spec.burn_in + spec.context_length + spec.horizon
    phi_x = float(rng.uniform(*spec.covariate_ar_range))
    phi_placebo = float(rng.uniform(*spec.covariate_ar_range))
    phi_y = float(rng.uniform(*spec.target_ar_range))
    beta_magnitude = float(rng.uniform(*spec.coefficient_magnitude_range))
    beta = beta_magnitude if (series_index // 2) % 2 == 0 else -beta_magnitude
    noise_sd = float(rng.uniform(*spec.innovation_sd_range))
    direction = 1.0 if series_index % 2 == 0 else -1.0

    active_full = _ar1(rng, phi_x, total)
    placebo_full = _ar1(rng, phi_placebo, total)
    noise_full = rng.normal(0.0, noise_sd, size=total).astype(np.float64)

    keep_start = spec.burn_in
    split = keep_start + spec.context_length
    active_context = active_full[keep_start:split]
    placebo_context = placebo_full[keep_start:split]
    active_future = active_full[split:].copy()
    placebo_future = placebo_full[split:].copy()

    start = spec.start_indices[series_index % len(spec.start_indices)]
    window, intervention_mask = _smooth_block(spec.horizon, start, spec.block_length)
    active_intervened = _intervene(
        active_future,
        active_context,
        window,
        direction,
        spec.intervention_scale,
        spec.clip_quantiles,
    )
    placebo_intervened = _intervene(
        placebo_future,
        placebo_context,
        window,
        direction,
        spec.intervention_scale,
        spec.clip_quantiles,
    )

    threshold_center = float(np.median(active_context))
    threshold_width = max(spec.transition_width * float(np.std(active_context, ddof=1)), 1e-6)
    lag_kernel = np.asarray(spec.lag_kernel, dtype=np.float64)
    lag_kernel = lag_kernel / lag_kernel.sum()

    active_intervened_full = active_full.copy()
    active_intervened_full[split:] = active_intervened
    target_mechanism = "contemporaneous_linear" if mechanism == "irrelevant_placebo" else mechanism

    target_factual_full = _target_path(
        target_mechanism,
        active_full,
        noise_full,
        phi_y,
        beta,
        first_lag=spec.first_lag,
        lag_kernel=lag_kernel,
        threshold_center=threshold_center,
        threshold_width=threshold_width,
    )
    target_active_intervened_full = _target_path(
        target_mechanism,
        active_intervened_full,
        noise_full,
        phi_y,
        beta,
        first_lag=spec.first_lag,
        lag_kernel=lag_kernel,
        threshold_center=threshold_center,
        threshold_width=threshold_width,
    )

    factual_future_target = target_factual_full[split:]
    active_future_target = target_active_intervened_full[split:]
    matched_active_response = active_future_target - factual_future_target

    if mechanism == "irrelevant_placebo":
        intervened_future_target = factual_future_target.copy()
        intervention_covariate = "placebo"
        active_for_model = active_future
        placebo_for_model = placebo_intervened
    else:
        intervened_future_target = active_future_target
        intervention_covariate = "active"
        active_for_model = active_intervened
        placebo_for_model = placebo_future

    oracle_response = intervened_future_target - factual_future_target
    if mechanism != "irrelevant_placebo" and np.linalg.norm(oracle_response, ord=1) <= 1e-8:
        raise RuntimeError("active intervention produced a numerically zero oracle response")
    if np.linalg.norm(matched_active_response, ord=1) <= 1e-8:
        raise RuntimeError("matched active intervention produced a numerically zero response")

    covariate_context = np.stack([active_context, placebo_context])
    covariate_future_factual = np.stack([active_future, placebo_future])
    covariate_future_intervened = np.stack([active_for_model, placebo_for_model])

    parameters = {
        "phi_x": phi_x,
        "phi_placebo": phi_placebo,
        "phi_y": phi_y,
        "beta": beta,
        "noise_sd": noise_sd,
        "intervention_direction": direction,
        "intervention_start": float(start),
        "threshold_center": threshold_center,
        "threshold_width": threshold_width,
    }
    return PairedScenario(
        mechanism=mechanism,
        generator_seed=generator_seed,
        series_index=series_index,
        covariate_names=("active", "placebo"),
        intervention_covariate=intervention_covariate,
        target_context=target_factual_full[keep_start:split].copy(),
        target_future_factual=factual_future_target.copy(),
        target_future_intervened=intervened_future_target.copy(),
        covariate_context=covariate_context,
        covariate_future_factual=covariate_future_factual,
        covariate_future_intervened=covariate_future_intervened,
        oracle_response=oracle_response.copy(),
        matched_active_oracle_response=matched_active_response.copy(),
        intervention_mask=intervention_mask,
        parameters=parameters,
    )


def generate_mechanism_batch(
    mechanism: str,
    generator_seed: int,
    count: int,
    spec: GeneratorSpec | None = None,
) -> list[PairedScenario]:
    """Generate a deterministic list of independently seeded series."""

    if count <= 0:
        raise ValueError("count must be positive")
    return [generate_scenario(mechanism, generator_seed, index, spec) for index in range(count)]
