import numpy as np

from covfaith.statistics import (
    BootstrapInterval,
    CellSummary,
    evaluate_continuation_gate,
    stratified_series_bootstrap,
)


def ci(estimate: float, lower: float, upper: float) -> BootstrapInterval:
    return BootstrapInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        replicates=5000,
        seed=9102,
    )


def test_stratified_bootstrap_is_deterministic_and_equal_weights_strata() -> None:
    values = np.array([0.0, 2.0, 10.0, 10.0, 10.0, 10.0])
    strata = np.array([11, 11, 29, 29, 29, 29])
    first = stratified_series_bootstrap(values, strata, replicates=200, seed=7)
    second = stratified_series_bootstrap(values, strata, replicates=200, seed=7)
    assert first == second
    assert first.estimate == 5.5


def test_gate_passes_two_mechanisms_with_accuracy_complement() -> None:
    cells = [
        CellSummary("chronos_2", "linear", 0.01, dsa=ci(0.60, 0.55, 0.70)),
        CellSummary("chronos_2", "lagged", 0.04, median_rgr=ci(0.30, 0.20, 0.40)),
    ]
    decision = evaluate_continuation_gate(cells)
    assert decision.passed
    assert decision.minimum_cell_count_passed
    assert decision.diversity_passed
    assert decision.complementary_accuracy_passed


def test_gate_passes_same_mechanism_across_both_backbones() -> None:
    cells = [
        CellSummary("chronos_2", "linear", 0.00, dsa=ci(0.60, 0.50, 0.70)),
        CellSummary("timesfm_3", "linear", 0.00, dsa=ci(0.65, 0.60, 0.75)),
    ]
    assert evaluate_continuation_gate(cells).passed


def test_gate_requires_interval_wholly_beyond_threshold() -> None:
    cells = [
        CellSummary("chronos_2", "linear", 0.00, dsa=ci(0.70, 0.65, 0.81)),
        CellSummary("timesfm_3", "lagged", 0.00, prr=ci(0.25, 0.19, 0.31)),
    ]
    decision = evaluate_continuation_gate(cells)
    assert not decision.passed
    assert decision.violating_cells == ()


def test_gate_requires_complementary_accuracy_condition() -> None:
    cells = [
        CellSummary("chronos_2", "linear", 0.03, dsa=ci(0.60, 0.55, 0.70)),
        CellSummary("timesfm_3", "linear", 0.04, dsa=ci(0.65, 0.60, 0.75)),
    ]
    decision = evaluate_continuation_gate(cells)
    assert not decision.passed
    assert not decision.complementary_accuracy_passed
