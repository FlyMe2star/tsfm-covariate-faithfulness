from covfaith.shape_statistics import ShapeCellSummary, evaluate_shape_gate
from covfaith.statistics import BootstrapInterval


def _interval(estimate: float, lower: float, upper: float) -> BootstrapInterval:
    return BootstrapInterval(estimate, lower, upper, 5000, 190926)


def _cell(backbone: str, mechanism: str, *, passing: bool = True) -> ShapeCellSummary:
    return ShapeCellSummary(
        backbone=backbone,
        mechanism=mechanism,
        relative_sql_difference=-0.05,
        dsa=_interval(0.92, 0.88 if passing else 0.80, 0.96),
        median_rgr=_interval(1.0, 0.80, 1.20),
        median_shape_distance_width_1=_interval(0.14, 0.10, 0.18),
        median_hidden_distortion_gap=_interval(0.07, 0.04, 0.10),
    )


def test_shape_gate_passes_two_mechanisms() -> None:
    decision = evaluate_shape_gate(
        [_cell("chronos_2", "biphasic_rebound"), _cell("chronos_2", "hysteresis")]
    )
    assert decision.passed
    assert decision.minimum_cell_count_passed
    assert decision.diversity_passed


def test_shape_gate_requires_complete_cell_rule() -> None:
    decision = evaluate_shape_gate(
        [_cell("chronos_2", "biphasic_rebound", passing=False), _cell("timesfm_3", "x")]
    )
    assert not decision.passed
    assert decision.passing_cells == ("timesfm_3/x",)


def test_shape_gate_requires_accuracy_complement() -> None:
    cells = [_cell("chronos_2", "a"), _cell("timesfm_3", "b")]
    cells = [
        ShapeCellSummary(
            cell.backbone,
            cell.mechanism,
            0.03,
            cell.dsa,
            cell.median_rgr,
            cell.median_shape_distance_width_1,
            cell.median_hidden_distortion_gap,
        )
        for cell in cells
    ]
    assert not evaluate_shape_gate(cells).passed
