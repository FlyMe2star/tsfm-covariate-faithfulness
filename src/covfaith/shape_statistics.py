"""Frozen P1-SHAPE cell rule and complete-matrix decision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from covfaith.statistics import BootstrapInterval


@dataclass(frozen=True)
class ShapeCellSummary:
    backbone: str
    mechanism: str
    relative_sql_difference: float
    dsa: BootstrapInterval
    median_rgr: BootstrapInterval
    median_shape_distance_width_1: BootstrapInterval
    median_hidden_distortion_gap: BootstrapInterval


@dataclass(frozen=True)
class ShapeGateDecision:
    passed: bool
    passing_cells: tuple[str, ...]
    checks_by_cell: Mapping[str, Mapping[str, bool]]
    minimum_cell_count_passed: bool
    diversity_passed: bool
    complementary_accuracy_passed: bool


def shape_cell_checks(
    cell: ShapeCellSummary,
    *,
    dsa_lower_minimum: float = 0.85,
    rgr_interval: tuple[float, float] = (0.65, 1.35),
    shape_distance_lower_minimum: float = 0.08,
    hidden_gap_lower_minimum: float = 0.03,
) -> dict[str, bool]:
    if not np.isfinite(cell.relative_sql_difference):
        raise ValueError("relative SQL difference must be finite")
    return {
        "dsa_coarse_eligible": cell.dsa.lower >= dsa_lower_minimum,
        "rgr_coarse_eligible": (
            cell.median_rgr.lower >= rgr_interval[0]
            and cell.median_rgr.upper <= rgr_interval[1]
        ),
        "fine_shape_distorted": (
            cell.median_shape_distance_width_1.lower >= shape_distance_lower_minimum
        ),
        "distortion_hidden_by_aggregation": (
            cell.median_hidden_distortion_gap.lower >= hidden_gap_lower_minimum
        ),
    }


def evaluate_shape_gate(
    cells: Sequence[ShapeCellSummary],
    *,
    dsa_lower_minimum: float = 0.85,
    rgr_interval: tuple[float, float] = (0.65, 1.35),
    shape_distance_lower_minimum: float = 0.08,
    hidden_gap_lower_minimum: float = 0.03,
    minimum_passing_cells: int = 2,
    complementary_relative_sql_maximum: float = 0.02,
) -> ShapeGateDecision:
    if not cells:
        raise ValueError("at least one P1-SHAPE cell is required")
    keys = [f"{cell.backbone}/{cell.mechanism}" for cell in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("P1-SHAPE cell keys must be unique")
    checks = {
        key: shape_cell_checks(
            cell,
            dsa_lower_minimum=dsa_lower_minimum,
            rgr_interval=rgr_interval,
            shape_distance_lower_minimum=shape_distance_lower_minimum,
            hidden_gap_lower_minimum=hidden_gap_lower_minimum,
        )
        for key, cell in zip(keys, cells, strict=True)
    }
    selected = [
        cell
        for key, cell in zip(keys, cells, strict=True)
        if all(checks[key].values())
    ]
    passing = tuple(f"{cell.backbone}/{cell.mechanism}" for cell in selected)
    minimum_passed = len(selected) >= minimum_passing_cells
    diversity_passed = (
        len({cell.mechanism for cell in selected}) >= 2
        or len({cell.backbone for cell in selected}) >= 2
    )
    accuracy_passed = any(
        cell.relative_sql_difference <= complementary_relative_sql_maximum
        for cell in selected
    )
    return ShapeGateDecision(
        passed=minimum_passed and diversity_passed and accuracy_passed,
        passing_cells=passing,
        checks_by_cell=checks,
        minimum_cell_count_passed=minimum_passed,
        diversity_passed=diversity_passed,
        complementary_accuracy_passed=accuracy_passed,
    )
