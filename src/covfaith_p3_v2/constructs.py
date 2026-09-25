"""Single registered P3-v2 intervention on construct-held-out backgrounds."""

from __future__ import annotations

from typing import Any

import numpy as np

from covfaith_p3.constructs import (
    FAMILIES,
    P3Scenario,
    _ar1,
    _intervene,
    _kernel,
    _link,
    _pulse,
    _response,
    _rng,
)
from covfaith_p3.data import SourceWindow


def generate_v2_scenario(
    window: SourceWindow,
    family: str,
    seed: int,
    seed_rank: int,
    data: dict[str, Any],
    mechanism: dict[str, Any],
    *,
    multiplier_cap: float = 1.4,
) -> P3Scenario:
    """Use the v1 structural model with only the registered v2 pulse changes."""
    if family not in FAMILIES:
        raise ValueError(f"unsupported P3-v2 family: {family}")
    if multiplier_cap <= 1:
        raise ValueError("multiplier cap must exceed one")
    context_length, horizon = int(data["context_length"]), int(data["horizon"])
    if window.background.shape != (context_length + horizon,):
        raise ValueError("background window length does not match v2 design")
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
    starts = mechanism["pulse_start_indices_by_family"][family]
    start_index = (window.source_rank + seed_rank) % len(starts)
    pulse_start = int(starts[start_index])
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
        covariate_context=np.stack([active[context_slice], placebo[context_slice]]),
        covariate_future_factual=np.stack([active[future_slice], placebo[future_slice]]),
        covariate_future_intervened=np.stack([active_intervened, placebo[future_slice]]),
        placebo_future_intervened=np.stack([active[future_slice], placebo_intervened]),
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
