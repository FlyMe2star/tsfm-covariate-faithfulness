from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from covfaith.adapters import ForecastBundle
from covfaith_p3.data import SourceWindow
from covfaith_p3_v2.constructs import generate_v2_scenario
from covfaith_p3_v2_model.runner import (
    _matches_frozen_manifest,
    infer_scenarios,
    load_frozen_contract,
    run_backbone_units,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeAdapter:
    backbone_id = "chronos_2"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def forecast(self, scenarios: list, variant: str) -> ForecastBundle:
        self.calls.append((variant, len(scenarios)))
        medians = []
        for scenario in scenarios:
            base = float(scenario.target_context[-1])
            if variant == "target_only":
                effect = np.zeros(24)
            elif variant == "factual":
                effect = scenario.covariate_future_factual[0] + 0.1 * (
                    scenario.covariate_future_factual[1]
                )
            else:
                effect = scenario.covariate_future_intervened[0] + 0.1 * (
                    scenario.covariate_future_intervened[1]
                )
            medians.append((base + effect).astype(np.float32))
        median = np.stack(medians)
        quantiles = median[:, :, None] + np.asarray([-1, 0, 1], dtype=np.float32)
        return ForecastBundle(median, quantiles, (0.1, 0.5, 0.9))


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/p3_semisynthetic/p3_v2_design_candidate.yaml").read_text()
    )


def _window() -> SourceWindow:
    index = np.arange(216)
    target = (100 + 5 * np.sin(index / 8)).astype(np.float64)
    times = (np.datetime64("2020-01-01") + index.astype("timedelta64[D]")).astype(
        "datetime64[ms]"
    )
    return SourceWindow("traffic_daily", "heldout-test", 0, 192, target, times)


def test_model_freeze_authorization_is_separate_from_construct_design() -> None:
    config, report = load_frozen_contract(ROOT)
    approval = json.loads(
        (ROOT / "configs/p3_semisynthetic/p3_v2_model_freeze_approval.json").read_text()
    )
    assert approval["approval_phrase"] == "APPROVE P3-V2 MODEL FREEZE"
    assert approval["sealed_model_inference_authorized"] is True
    assert config["authorization"]["model_inference_allowed"] is False
    assert report["decision"]["construct_gate_passed"] is True


def test_manifest_replay_accepts_only_exact_frozen_content_under_lf_or_crlf() -> None:
    payload = {"config_hash": "fixed", "sources": {"traffic_daily": [{"source_id": "id-1"}]}}
    lf = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    crlf = lf.replace(b"\n", b"\r\n")
    assert _matches_frozen_manifest(payload, hashlib.sha256(lf).hexdigest())
    assert _matches_frozen_manifest(payload, hashlib.sha256(crlf).hexdigest())
    assert not _matches_frozen_manifest(
        {"config_hash": "changed", "sources": payload["sources"]},
        hashlib.sha256(crlf).hexdigest(),
    )


def test_fake_adapter_exercises_factual_active_sham_and_lower_link() -> None:
    config = _config()
    window = _window()
    scenario = generate_v2_scenario(
        window, "biphasic_rebound", 1801, 0, config["data"], config["mechanism"]
    )
    adapter = FakeAdapter()
    arrays = infer_scenarios(adapter, [scenario], {window.source_id: window}, config)
    assert adapter.calls == [
        ("target_only", 1),
        ("factual", 1),
        ("intervened", 1),
        ("intervened", 1),
        ("factual", 1),
        ("intervened", 1),
    ]
    assert arrays["median_active"].shape == (1, 24)
    assert arrays["quantiles_factual"].shape == (1, 24, 3)
    assert arrays["median_sham"].shape == (1, 24)
    assert arrays["lower_oracle_response"].shape == (1, 24)
    np.testing.assert_array_equal(arrays["oracle_response"][0], scenario.oracle_response)
    assert not np.array_equal(arrays["median_active"], arrays["median_factual"])
    assert not np.array_equal(arrays["median_sham"], arrays["median_factual"])


def test_formal_runner_refuses_fake_adapter(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="pinned checkpoint adapter"):
        run_backbone_units(
            ROOT, FakeAdapter(), tmp_path, tmp_path / "private", mode="checkpoint_smoke"
        )
