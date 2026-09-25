from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import covfaith_p3_v2_analysis.analysis as analysis_module
from covfaith.adapters import ForecastBundle
from covfaith_p3.data import SourceWindow, sha256_file
from covfaith_p3_v2.constructs import generate_v2_scenario
from covfaith_p3_v2_analysis.analysis import (
    _check_unit_arrays,
    cluster_bootstrap,
    summarize_cell,
)
from covfaith_p3_v2_model.runner import infer_scenarios

ROOT = Path(__file__).resolve().parents[1]


class FakeAdapter:
    backbone_id = "chronos_2"

    def forecast(self, scenarios: list, variant: str) -> ForecastBundle:
        medians = []
        for scenario in scenarios:
            base = float(scenario.target_context[-1])
            if variant == "target_only":
                effect = np.zeros(24)
            elif variant == "factual":
                effect = scenario.covariate_future_factual[0]
            else:
                effect = scenario.covariate_future_intervened[0]
            medians.append((base + effect).astype(np.float32))
        median = np.stack(medians)
        quantiles = median[:, :, None] + np.asarray([-1, 0, 1], dtype=np.float32)
        return ForecastBundle(median, quantiles, (0.1, 0.5, 0.9))


def _fixture() -> tuple[dict, dict, SourceWindow, object, dict]:
    config = yaml.safe_load(
        (ROOT / "configs/p3_semisynthetic/p3_v2_design_candidate.yaml").read_text()
    )
    p1 = yaml.safe_load((ROOT / "configs/p1_shape/covintervene_shape_p1.yaml").read_text())
    config["response_metrics"]["uncertainty"]["bootstrap_replicates"] = 20
    index = np.arange(216)
    target = (100 + 5 * np.sin(index / 8)).astype(np.float64)
    times = (np.datetime64("2020-01-01") + index.astype("timedelta64[D]")).astype("datetime64[ms]")
    window = SourceWindow("traffic_daily", "fixture-1", 0, 192, target, times)
    scenario = generate_v2_scenario(
        window, "biphasic_rebound", 1801, 0, config["data"], config["mechanism"]
    )
    arrays = infer_scenarios(FakeAdapter(), [scenario], {window.source_id: window}, config)
    return config, p1, window, scenario, arrays


def test_original_id_bootstrap_is_deterministic_and_keeps_seed_cluster() -> None:
    labels = ["a", "a", "b"]
    values = np.asarray([0.0, 2.0, 10.0])
    first = cluster_bootstrap(labels, values, np.mean, replicates=200, seed=9)
    second = cluster_bootstrap(labels, values, np.mean, replicates=200, seed=9)
    assert first == second
    assert first.estimate == 4.0
    assert first.lower <= first.estimate <= first.upper


def test_archived_worlds_and_control_subset_match_construct_replay() -> None:
    config, _, window, scenario, arrays = _fixture()
    _check_unit_arrays(
        arrays,
        [scenario],
        {window.source_id: window},
        config,
        horizon=24,
        context_length=192,
        seed_rank=0,
    )
    tampered = {**arrays, "oracle_response": arrays["oracle_response"] + 1}
    with pytest.raises(RuntimeError, match="oracle_response"):
        _check_unit_arrays(
            tampered,
            [scenario],
            {window.source_id: window},
            config,
            horizon=24,
            context_length=192,
            seed_rank=0,
        )


def test_construct_replay_admits_float64_roundoff_but_not_data_changes() -> None:
    config, _, window, scenario, arrays = _fixture()
    rounded = {**arrays, "target_context": arrays["target_context"].copy()}
    rounded["target_context"][0, 0] = np.nextafter(
        rounded["target_context"][0, 0], np.inf
    )
    _check_unit_arrays(
        rounded,
        [scenario],
        {window.source_id: window},
        config,
        horizon=24,
        context_length=192,
        seed_rank=0,
    )
    changed = {**rounded, "target_context": rounded["target_context"].copy()}
    changed["target_context"][0, 0] += 1e-5
    with pytest.raises(RuntimeError, match="target_context: model archive differs"):
        _check_unit_arrays(
            changed,
            [scenario],
            {window.source_id: window},
            config,
            horizon=24,
            context_length=192,
            seed_rank=0,
        )


def test_construct_replay_does_not_accept_nonfinite_or_wrong_dtype() -> None:
    config, _, window, scenario, arrays = _fixture()
    for replacement in (
        np.full_like(arrays["target_context"], np.nan),
        arrays["target_context"].astype(np.float32),
    ):
        changed = {**arrays, "target_context": replacement}
        with pytest.raises(RuntimeError, match="target_context"):
            _check_unit_arrays(
                changed,
                [scenario],
                {window.source_id: window},
                config,
                horizon=24,
                context_length=192,
                seed_rank=0,
            )


def test_descriptive_summary_keeps_missing_lower_sql_explicit() -> None:
    config, p1, _, scenario, arrays = _fixture()
    cell = summarize_cell(
        "chronos_2",
        "traffic_daily",
        "biphasic_rebound",
        [{"source_id": scenario.source_id, "scenario": scenario, "arrays": arrays, "index": 0}],
        config,
        p1,
    )
    assert cell["valid_scenario_count"] == 1
    assert cell["relative_sql"] is not None
    assert cell["lower_link_relative_sql"] is None
    assert cell["lower_link_sql_status"] == "unavailable_lower_quantiles_not_archived_no_imputation"


def test_archive_reader_checks_both_backbone_manifests_and_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, window, scenario, arrays = _fixture()
    construct = {
        "config_hash": "fixture-config",
        "v2_selection_manifest_sha256": "fixture-selection",
    }
    selected = {"traffic_daily": [window]}
    prepared = {("traffic_daily", "biphasic_rebound", 1801): [scenario]}
    monkeypatch.setattr(analysis_module, "load_frozen_contract", lambda _: (config, construct))
    monkeypatch.setattr(analysis_module, "load_frozen_windows", lambda *_: selected)
    monkeypatch.setattr(analysis_module, "prepare_frozen_units", lambda *_: prepared)
    monkeypatch.setattr(analysis_module, "model_code_hash", lambda _: "fixture-code")
    report_sha = sha256_file(
        ROOT / "evidence/p3_semisynthetic/v2_construct/p3_v2_construct_preflight.json"
    )
    label = "traffic_daily/biphasic_rebound/seed_01801"
    for model in config["models"]:
        backbone = model["id"]
        unit_dir = tmp_path / "full_units" / backbone / "traffic_daily/biphasic_rebound"
        unit_dir.mkdir(parents=True)
        array_path = unit_dir / "seed_01801.npz"
        np.savez_compressed(array_path, **arrays)
        manifest = {
            "completed": True,
            "result_status": "sealed_unit_only",
            "config_hash": "fixture-config",
            "model_runner_code_sha256": "fixture-code",
            "v2_selection_manifest_sha256": "fixture-selection",
            "backbone": backbone,
            "checkpoint": model["checkpoint"],
            "checkpoint_revision": model["revision"],
            "dataset": "traffic_daily",
            "family": "biphasic_rebound",
            "seed": 1801,
            "valid_scenario_count": 1,
            "excluded_scenario_count": 31,
            "array_sha256": sha256_file(array_path),
            "sham_scenario_count": 1,
            "lower_link_scenario_count": 1,
        }
        (unit_dir / "seed_01801.json").write_text(json.dumps(manifest), encoding="utf-8")
        completion = {
            "result_status": "sealed_units_complete_not_analyzed",
            "mode": "full_units",
            "config_hash": "fixture-config",
            "model_runner_code_sha256": "fixture-code",
            "construct_report_sha256": report_sha,
            "v2_selection_manifest_sha256": "fixture-selection",
            "backbone": backbone,
            "checkpoint": model["checkpoint"],
            "checkpoint_revision": model["revision"],
            "completed_unit_count": 1,
            "completed_units": [label],
            "valid_scenario_count": 1,
            "scientific_gate_computed": False,
        }
        (tmp_path / "full_units" / backbone / "completion.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
    _, _, records, inventory = analysis_module._complete_records(ROOT, tmp_path, tmp_path)
    assert len(records) == 2
    assert all(len(rows) == 1 for rows in records.values())
    assert len(inventory) == 6
