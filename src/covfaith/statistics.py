"""Series-level uncertainty and the frozen P0 continuation decision."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    replicates: int
    seed: int


@dataclass(frozen=True)
class CellSummary:
    backbone: str
    mechanism: str
    relative_sql_difference: float
    dsa: BootstrapInterval | None = None
    median_rgr: BootstrapInterval | None = None
    prr: BootstrapInterval | None = None


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    violating_cells: tuple[str, ...]
    violations_by_cell: Mapping[str, tuple[str, ...]]
    minimum_cell_count_passed: bool
    diversity_passed: bool
    complementary_accuracy_passed: bool


def stratified_series_bootstrap(
    values: ArrayLike,
    strata: Sequence[int | str],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    replicates: int = 5000,
    seed: int = 9102,
) -> BootstrapInterval:
    """Bootstrap complete series within strata and weight strata equally."""

    data = np.asarray(values, dtype=np.float64)
    labels = np.asarray(strata)
    if data.ndim != 1 or labels.ndim != 1 or data.shape != labels.shape or data.size == 0:
        raise ValueError("values and strata must be aligned nonempty vectors")
    if not np.all(np.isfinite(data)):
        raise ValueError("bootstrap values must be finite")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    unique = np.unique(labels)
    groups = [data[labels == label] for label in unique]
    if any(group.size == 0 for group in groups):
        raise ValueError("every stratum must contain at least one series")

    def aggregate(samples: Iterable[np.ndarray]) -> float:
        estimates = [float(statistic(sample)) for sample in samples]
        if not np.all(np.isfinite(estimates)):
            raise ValueError("statistic returned a nonfinite value")
        return float(np.mean(estimates))

    estimate = aggregate(groups)
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        resampled = [group[rng.integers(0, group.size, size=group.size)] for group in groups]
        boot[index] = aggregate(resampled)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return BootstrapInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        replicates=replicates,
        seed=seed,
    )


def cell_violations(
    cell: CellSummary,
    *,
    dsa_minimum: float = 0.80,
    rgr_interval: tuple[float, float] = (0.50, 1.50),
    prr_maximum: float = 0.20,
) -> tuple[str, ...]:
    """Return violations whose complete 95% interval lies beyond a frozen bound."""

    violations: list[str] = []
    if cell.dsa is not None and cell.dsa.upper < dsa_minimum:
        violations.append("directional_sign_agreement_below_minimum")
    if cell.median_rgr is not None:
        lower_bound, upper_bound = rgr_interval
        if cell.median_rgr.upper < lower_bound:
            violations.append("response_gain_ratio_below_minimum")
        elif cell.median_rgr.lower > upper_bound:
            violations.append("response_gain_ratio_above_maximum")
    if cell.prr is not None and cell.prr.lower > prr_maximum:
        violations.append("placebo_response_ratio_above_maximum")
    return tuple(violations)


def evaluate_continuation_gate(
    cells: Sequence[CellSummary],
    *,
    minimum_violating_cells: int = 2,
    complementary_relative_sql_maximum: float = 0.02,
) -> GateDecision:
    """Apply the frozen cell-count, diversity, and forecast-skill conditions."""

    if not cells:
        raise ValueError("at least one cell summary is required")
    keys = [f"{cell.backbone}/{cell.mechanism}" for cell in cells]
    if len(set(keys)) != len(keys):
        raise ValueError("cell summaries must have unique backbone/mechanism keys")
    if any(not np.isfinite(cell.relative_sql_difference) for cell in cells):
        raise ValueError("relative SQL differences must be finite")

    violations = {key: cell_violations(cell) for key, cell in zip(keys, cells, strict=True)}
    violating_cells = tuple(key for key in keys if violations[key])
    selected = [cell for key, cell in zip(keys, cells, strict=True) if violations[key]]
    minimum_passed = len(selected) >= minimum_violating_cells
    mechanisms = {cell.mechanism for cell in selected}
    backbones = {cell.backbone for cell in selected}
    diversity_passed = len(mechanisms) >= 2 or len(backbones) >= 2
    complementary_passed = any(
        cell.relative_sql_difference <= complementary_relative_sql_maximum for cell in selected
    )
    return GateDecision(
        passed=minimum_passed and diversity_passed and complementary_passed,
        violating_cells=violating_cells,
        violations_by_cell={key: value for key, value in violations.items() if value},
        minimum_cell_count_passed=minimum_passed,
        diversity_passed=diversity_passed,
        complementary_accuracy_passed=complementary_passed,
    )
