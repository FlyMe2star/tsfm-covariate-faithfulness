from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from covfaith.config import canonical_config_hash
from covfaith_p3.constructs import validate_scenario
from covfaith_p3.data import SourceSeries, SourceWindow
from covfaith_p3_v2.constructs import generate_v2_scenario
from covfaith_p3_v2.preflight import _code_hash, _select_heldout_windows

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "p3_semisynthetic" / "p3_v2_design_candidate.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _window(rank: int = 0) -> SourceWindow:
    index = np.arange(216)
    target = (100 + 5 * np.sin(index / 8)).astype(np.float64)
    times = (np.datetime64("2020-01-01") + index.astype("timedelta64[D]")).astype(
        "datetime64[ms]"
    )
    return SourceWindow("traffic_daily", "new-source", rank, 192, target, times)


def test_owner_approval_pins_candidate_without_authorizing_models() -> None:
    config = _config()
    approval = json.loads(
        (ROOT / "configs" / "p3_semisynthetic" / "p3_v2_design_approval.json").read_text()
    )
    assert approval["approved_candidate_config_canonical_sha256"] == canonical_config_hash(
        CONFIG_PATH
    )
    assert approval["cpu_construct_preflight_authorized"] is True
    assert approval["model_inference_authorized"] is False
    assert config["authorization"]["model_inference_allowed"] is False


def test_model_freeze_candidate_matches_construct_evidence_but_is_not_authorized() -> None:
    freeze = json.loads(
        (ROOT / "configs" / "p3_semisynthetic" / "p3_v2_model_freeze_candidate.json")
        .read_text(encoding="utf-8")
    )
    report_path = (
        ROOT / "evidence" / "p3_semisynthetic" / "v2_construct" / "p3_v2_construct_preflight.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    locked = freeze["locked_evidence"]
    assert freeze["sealed_model_inference_authorized"] is False
    assert report["decision"]["construct_gate_passed"] is True
    assert locked["v2_config_canonical_sha256"] == canonical_config_hash(CONFIG_PATH)
    assert locked["v2_construct_report_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert locked["v2_construct_code_sha256"] == _code_hash(ROOT)
    assert locked["v2_selection_manifest_sha256"] == report["v2_selection_manifest_sha256"]


@pytest.mark.parametrize("family", ["biphasic_rebound", "dispersed_delayed_pulse"])
def test_v2_worlds_are_exact_paired_deterministic_and_placebo_zero(family: str) -> None:
    config = _config()
    window = _window()
    first = generate_v2_scenario(window, family, 1801, 0, config["data"], config["mechanism"])
    second = generate_v2_scenario(window, family, 1801, 0, config["data"], config["mechanism"])
    np.testing.assert_array_equal(first.oracle_response, second.oracle_response)
    np.testing.assert_array_equal(first.target_context_factual, first.target_context_intervened)
    np.testing.assert_array_equal(first.placebo_oracle_response, 0)
    assert validate_scenario(first, 0.0)["oracle_ratio"] > 0
    assert first.parameters["pulse_start"] in config["mechanism"][
        "pulse_start_indices_by_family"
    ][family]


def test_latest_delayed_pulse_has_complete_finite_horizon_support() -> None:
    config = _config()
    scenario = generate_v2_scenario(
        _window(rank=2),
        "dispersed_delayed_pulse",
        1801,
        0,
        config["data"],
        config["mechanism"],
    )
    start = int(scenario.parameters["pulse_start"])
    length = int(scenario.parameters["pulse_length"])
    kernel_lags = config["mechanism"]["kernel_parameter_ranges"][
        "dispersed_delayed_pulse"
    ]["kernel_length"]
    assert (start, length) == (5, 7)
    assert start + length - 1 + kernel_lags - 1 < config["data"]["horizon"]


def test_v2_source_selector_never_reuses_v1_ids() -> None:
    index = np.arange(240)
    times = (np.datetime64("2020-01-01") + index.astype("timedelta64[D]")).astype(
        "datetime64[ms]"
    )
    series = [
        SourceSeries(f"road-{i}", (100 + 5 * np.sin(index / 8)).astype(np.float64), times)
        for i in range(4)
    ]
    common = {
        "context_length": 192,
        "horizon": 24,
        "origin_region_start_fraction": 0.60,
        "origin_stride": 100,
        "context_standard_deviation_min": 1e-6,
        "require_nominal_cadence": True,
        "source_ids_per_dataset": 2,
    }
    source = {
        "id": "traffic_daily",
        "frequency_seconds": 86400,
        "context_positive_fraction_min": 0.95,
    }
    v1, v2, v1_eligible, v2_eligible = _select_heldout_windows(
        series,
        {**common, "selection_salt": "v1-test"},
        {**common, "selection_salt": "v2-test"},
        source,
    )
    assert (v1_eligible, v2_eligible) == (4, 2)
    assert {window.source_id for window in v1}.isdisjoint(
        {window.source_id for window in v2}
    )
