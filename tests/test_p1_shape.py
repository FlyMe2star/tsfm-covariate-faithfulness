from __future__ import annotations

from pathlib import Path

import numpy as np

from covfaith.adapters import ForecastBundle
from covfaith.config import verify_config_lock
from covfaith.p1_shape import run_shape_backbone_units


class FakeShapeAdapter:
    backbone_id = "chronos_2"

    def forecast(self, scenarios, variant):
        medians = []
        for scenario in scenarios:
            if variant == "target_only":
                median = np.repeat(scenario.target_context[-1], 24)
            elif variant == "factual":
                median = 0.05 * np.sum(scenario.covariate_future_factual, axis=0)
            else:
                median = 0.05 * np.sum(scenario.covariate_future_intervened, axis=0)
            medians.append(median)
        stacked = np.asarray(medians, dtype=np.float32)
        quantiles = np.stack([stacked - 0.1, stacked, stacked + 0.1], axis=-1)
        return ForecastBundle(stacked, quantiles, (0.1, 0.5, 0.9))


def test_frozen_shape_config_lock_matches() -> None:
    repo_root = Path(__file__).parents[1]
    digest = verify_config_lock(
        repo_root / "configs/p1_shape/covintervene_shape_p1.yaml",
        repo_root / "configs/p1_shape/covintervene_shape_p1.lock.json",
    )
    assert digest == "27826ad34bfe0acd8ee90ff4d0fbafb011a6b00aeea276dacaa9f9fe2a261918"


def test_shape_smoke_runner_is_complete_but_cannot_decide(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    report = run_shape_backbone_units(root, FakeShapeAdapter(), tmp_path, smoke_count=1)
    assert report["result_status"] == "smoke_only_not_scientific_evidence"
    assert report["completed_unit_count"] == 12
    assert report["scientific_gate_computed"] is False
