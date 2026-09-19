"""Untouched paired structural generators for the frozen P1-SHAPE audit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SHAPE_MECHANISMS = (
    "biphasic_rebound",
    "dispersed_delayed_pulse",
    "rate_asymmetric_hysteresis",
    "two_covariate_synergy",
)


@dataclass(frozen=True)
class ShapeGeneratorSpec:
    context_length: int = 192
    horizon: int = 24
    target_ar_range: tuple[float, float] = (0.20, 0.58)
    covariate_ar_range: tuple[float, float] = (0.25, 0.72)
    coefficient_magnitude_range: tuple[float, float] = (0.35, 0.85)
    innovation_sd_range: tuple[float, float] = (0.12, 0.30)
    intervention_scale: float = 0.65
    pulse_lengths: tuple[int, ...] = (3, 5, 7)
    start_indices: tuple[int, ...] = (1, 7, 13)
    clip_quantiles: tuple[float, float] = (0.005, 0.995)
    burn_in: int = 96


@dataclass(frozen=True)
class ShapeScenario:
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
    covariate_future_worlds: Mapping[str, FloatArray]
    target_future_worlds: Mapping[str, FloatArray]
    oracle_response_worlds: Mapping[str, FloatArray]

    @property
    def series_id(self) -> str:
        return f"p1-seed{self.generator_seed:05d}-series{self.series_index:04d}"


def _rng(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, index, 0x51A9E]))


def _ar1(rng: np.random.Generator, phi: float, length: int) -> FloatArray:
    noise = rng.normal(0, np.sqrt(max(1 - phi**2, 1e-8)), size=length)
    values = np.empty(length, dtype=np.float64)
    values[0] = noise[0]
    for step in range(1, length):
        values[step] = phi * values[step - 1] + noise[step]
    return values


def _pulse(horizon: int, start: int, length: int) -> tuple[FloatArray, BoolArray]:
    stop = min(start + length, horizon)
    width = stop - start
    if start < 0 or width <= 0:
        raise ValueError("pulse must overlap the forecast horizon")
    pulse = np.zeros(horizon, dtype=np.float64)
    pulse[start:stop] = np.sin(np.pi * (np.arange(width) + 1) / (width + 1)) ** 2
    return pulse, pulse > 0


def _intervene(
    factual: FloatArray,
    context: FloatArray,
    pulse: FloatArray,
    direction: float,
    scale: float,
    clip_quantiles: tuple[float, float],
) -> FloatArray:
    context_sd = max(float(np.std(context, ddof=1)), 1e-6)
    proposed = factual + direction * scale * context_sd * pulse
    lower, upper = np.quantile(context, clip_quantiles)
    result = factual.copy()
    active = pulse > 0
    result[active] = np.clip(proposed[active], lower, upper)
    return result


def _convolve(values: FloatArray, kernel: FloatArray, step: int) -> float:
    stop = min(step + 1, kernel.size)
    return float(np.dot(kernel[:stop], values[step - np.arange(stop)]))


def _simulate(
    mechanism: str,
    x1: FloatArray,
    x2: FloatArray,
    innovations: FloatArray,
    *,
    phi_y: float,
    beta: float,
    parameters: Mapping[str, float],
) -> FloatArray:
    target = np.zeros_like(x1)
    state = 0.0
    if mechanism == "biphasic_rebound":
        lag = np.arange(11, dtype=np.float64)
        positive = np.exp(-0.5 * ((lag - parameters["positive_center"]) / 1.15) ** 2)
        negative = np.exp(-0.5 * ((lag - parameters["negative_center"]) / 1.45) ** 2)
        kernel = positive / positive.sum() - parameters["rebound_ratio"] * negative / negative.sum()
    elif mechanism == "dispersed_delayed_pulse":
        lag = np.arange(12, dtype=np.float64)
        onset = int(parameters["delay_onset"])
        shifted = np.maximum(lag - onset, 0.0)
        kernel = shifted ** 2 * np.exp(-shifted / parameters["delay_scale"])
        kernel[: onset + 1] = 0.0
        kernel /= kernel.sum()
    else:
        kernel = np.ones(1)

    previous_x1 = x1[0]
    for step in range(x1.size):
        previous_y = target[step - 1] if step else 0.0
        if mechanism in {"biphasic_rebound", "dispersed_delayed_pulse"}:
            effect = _convolve(x1, kernel, step)
        elif mechanism == "rate_asymmetric_hysteresis":
            change = x1[step] - previous_x1
            gain = parameters["rise_gain"] if change >= 0 else parameters["fall_gain"]
            state = parameters["state_decay"] * state + gain * change
            effect = np.tanh(state / parameters["state_scale"])
            previous_x1 = x1[step]
        elif mechanism == "two_covariate_synergy":
            scale = parameters["interaction_scale"]
            effect = (
                parameters["main_a"] * x1[step]
                + parameters["main_b"] * x2[step]
                + parameters["synergy"] * np.tanh(x1[step] / scale) * np.tanh(x2[step] / scale)
            )
        else:
            raise ValueError(f"unsupported mechanism {mechanism!r}")
        target[step] = phi_y * previous_y + beta * effect + innovations[step]
    return target


def generate_shape_scenario(
    mechanism: str,
    generator_seed: int,
    series_index: int,
    spec: ShapeGeneratorSpec | None = None,
) -> ShapeScenario:
    if mechanism not in SHAPE_MECHANISMS:
        raise ValueError(f"unknown P1-SHAPE mechanism {mechanism!r}")
    spec = spec or ShapeGeneratorSpec()
    if spec.horizon != 24:
        raise ValueError("the frozen P1-SHAPE resolution widths require horizon 24")
    rng = _rng(generator_seed, series_index)
    total = spec.burn_in + spec.context_length + spec.horizon
    phi_x1 = float(rng.uniform(*spec.covariate_ar_range))
    phi_x2 = float(rng.uniform(*spec.covariate_ar_range))
    phi_y = float(rng.uniform(*spec.target_ar_range))
    beta_magnitude = float(rng.uniform(*spec.coefficient_magnitude_range))
    beta = beta_magnitude if (series_index // 2) % 2 == 0 else -beta_magnitude
    innovation_sd = float(rng.uniform(*spec.innovation_sd_range))
    direction_a = 1.0 if series_index % 2 == 0 else -1.0
    direction_b = -direction_a if (series_index // 4) % 2 else direction_a

    x1 = _ar1(rng, phi_x1, total)
    x2 = _ar1(rng, phi_x2, total)
    innovations = rng.normal(0, innovation_sd, size=total).astype(np.float64)
    split = spec.burn_in + spec.context_length
    context_slice = slice(spec.burn_in, split)
    future_slice = slice(split, total)
    x1_context = x1[context_slice]
    x2_context = x2[context_slice]
    x1_future = x1[future_slice].copy()
    x2_future = x2[future_slice].copy()
    pulse_length = spec.pulse_lengths[series_index % len(spec.pulse_lengths)]
    start = spec.start_indices[(series_index // len(spec.pulse_lengths)) % len(spec.start_indices)]
    pulse, mask = _pulse(spec.horizon, start, pulse_length)
    x1_intervened = _intervene(
        x1_future, x1_context, pulse, direction_a, spec.intervention_scale, spec.clip_quantiles
    )
    x2_intervened = _intervene(
        x2_future, x2_context, pulse, direction_b, spec.intervention_scale, spec.clip_quantiles
    )

    parameters: dict[str, float] = {
        "phi_x1": phi_x1,
        "phi_x2": phi_x2,
        "phi_y": phi_y,
        "beta": beta,
        "innovation_sd": innovation_sd,
        "intervention_start": float(start),
        "pulse_length": float(pulse_length),
        "positive_center": float(rng.uniform(0.5, 1.5)),
        "negative_center": float(rng.uniform(5.5, 8.0)),
        "rebound_ratio": float(rng.uniform(0.45, 0.80)),
        "delay_onset": float(rng.integers(1, 4)),
        "delay_scale": float(rng.uniform(0.8, 1.8)),
        "rise_gain": float(rng.uniform(0.65, 1.05)),
        "fall_gain": float(rng.uniform(0.25, 0.55)),
        "state_decay": float(rng.uniform(0.55, 0.82)),
        "state_scale": float(rng.uniform(0.65, 1.20)),
        "main_a": float(rng.uniform(0.30, 0.60)),
        "main_b": float(rng.uniform(0.25, 0.55)),
        "synergy": float(rng.uniform(0.55, 0.95)),
        "interaction_scale": float(rng.uniform(0.75, 1.25)),
    }

    factual_covariates = np.stack([x1_future, x2_future])
    worlds: dict[str, FloatArray] = {"factual": factual_covariates}
    if mechanism == "two_covariate_synergy":
        worlds.update(
            {
                "a_only": np.stack([x1_intervened, x2_future]),
                "b_only": np.stack([x1_future, x2_intervened]),
                "joint": np.stack([x1_intervened, x2_intervened]),
            }
        )
        primary_world = "joint"
        intervention_covariate = "covariate_a+covariate_b"
    else:
        worlds["intervened"] = np.stack([x1_intervened, x2_future])
        primary_world = "intervened"
        intervention_covariate = "covariate_a"

    target_worlds: dict[str, FloatArray] = {}
    for name, future_covariates in worlds.items():
        full_x1 = x1.copy()
        full_x2 = x2.copy()
        full_x1[future_slice] = future_covariates[0]
        full_x2[future_slice] = future_covariates[1]
        target_worlds[name] = _simulate(
            mechanism,
            full_x1,
            full_x2,
            innovations,
            phi_y=phi_y,
            beta=beta,
            parameters=parameters,
        )[future_slice]

    factual_target = target_worlds["factual"]
    oracle_worlds = {
        name: values - factual_target for name, values in target_worlds.items() if name != "factual"
    }
    primary_oracle = oracle_worlds[primary_world]
    if float(np.sum(np.abs(primary_oracle))) <= 1e-8:
        raise RuntimeError("P1 intervention produced a numerically zero oracle response")
    if mechanism == "two_covariate_synergy":
        interaction = oracle_worlds["joint"] - oracle_worlds["a_only"] - oracle_worlds["b_only"]
        if float(np.sum(np.abs(interaction))) <= 1e-4:
            raise RuntimeError("synergy mechanism produced a negligible interaction response")

    return ShapeScenario(
        mechanism=mechanism,
        generator_seed=generator_seed,
        series_index=series_index,
        covariate_names=("covariate_a", "covariate_b"),
        intervention_covariate=intervention_covariate,
        target_context=_simulate(
            mechanism,
            x1,
            x2,
            innovations,
            phi_y=phi_y,
            beta=beta,
            parameters=parameters,
        )[context_slice],
        target_future_factual=factual_target.copy(),
        target_future_intervened=target_worlds[primary_world].copy(),
        covariate_context=np.stack([x1_context, x2_context]),
        covariate_future_factual=factual_covariates.copy(),
        covariate_future_intervened=worlds[primary_world].copy(),
        oracle_response=primary_oracle.copy(),
        matched_active_oracle_response=primary_oracle.copy(),
        intervention_mask=mask,
        parameters=parameters,
        covariate_future_worlds={key: value.copy() for key, value in worlds.items()},
        target_future_worlds={key: value.copy() for key, value in target_worlds.items()},
        oracle_response_worlds={key: value.copy() for key, value in oracle_worlds.items()},
    )


def generate_shape_batch(
    mechanism: str,
    generator_seed: int,
    count: int,
    spec: ShapeGeneratorSpec | None = None,
) -> list[ShapeScenario]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [
        generate_shape_scenario(mechanism, generator_seed, index, spec)
        for index in range(count)
    ]
