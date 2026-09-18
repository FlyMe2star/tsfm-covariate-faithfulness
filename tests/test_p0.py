from pathlib import Path

import numpy as np
import pytest

from covfaith.adapters import ForecastBundle
from covfaith.p0 import analyze_complete_p0, run_backbone_units

ROOT = Path(__file__).resolve().parents[1]


class OracleFakeAdapter:
    backbone_id = "chronos_2"
    levels = (0.1, 0.5, 0.9)

    def forecast(self, scenarios, variant):
        medians = []
        for scenario in scenarios:
            if variant == "intervened":
                medians.append(scenario.target_future_intervened)
            else:
                medians.append(scenario.target_future_factual)
        median = np.asarray(medians, dtype=np.float32)
        quantiles = np.repeat(median[:, :, None], len(self.levels), axis=2)
        return ForecastBundle(median, quantiles, self.levels)


def test_smoke_runner_is_resumable_and_never_computes_gate(tmp_path: Path) -> None:
    adapter = OracleFakeAdapter()
    first = run_backbone_units(ROOT, adapter, tmp_path, smoke_count=1)
    assert first["completed_unit_count"] == 12
    assert first["scientific_gate_computed"] is False
    unit = tmp_path / "units" / "chronos_2" / "contemporaneous_linear" / "seed_00011.npz"
    with np.load(unit, allow_pickle=False) as loaded:
        assert loaded["dsa"][0] == 1.0
        assert loaded["nre"][0] == pytest.approx(0.0, abs=1e-6)
    modified = unit.stat().st_mtime_ns
    second = run_backbone_units(ROOT, adapter, tmp_path, smoke_count=1)
    assert second["completed_unit_count"] == 12
    assert unit.stat().st_mtime_ns == modified


def test_analysis_rejects_smoke_inventory(tmp_path: Path) -> None:
    run_backbone_units(ROOT, OracleFakeAdapter(), tmp_path, smoke_count=1)
    with pytest.raises(RuntimeError, match="incomplete or mismatched P0 unit"):
        analyze_complete_p0(ROOT, tmp_path)
