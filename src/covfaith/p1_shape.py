"""Resumable frozen execution and sealed analysis for P1-SHAPE."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from covfaith.adapters import ForecastBundle, ScenarioAdapter
from covfaith.config import load_yaml, verify_config_lock
from covfaith.forecast_metrics import scaled_quantile_loss, weighted_quantile_loss
from covfaith.metrics import directional_sign_agreement, response_gain_ratio
from covfaith.shape_generators import (
    SHAPE_MECHANISMS,
    ShapeGeneratorSpec,
    ShapeScenario,
    generate_shape_batch,
)
from covfaith.shape_metrics import interaction_response, shape_metric_suite, signed_shape_distance
from covfaith.shape_statistics import ShapeCellSummary, evaluate_shape_gate
from covfaith.statistics import BootstrapInterval, stratified_series_bootstrap


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def scientific_code_hash(root: str | Path) -> str:
    root_path = Path(root).resolve()
    paths = sorted((root_path / "src" / "covfaith").glob("*.py"))
    paths.extend(
        [
            root_path / "configs" / "p1_shape" / "covintervene_shape_p1.yaml",
            root_path / "configs" / "p1_shape" / "covintervene_shape_p1.lock.json",
        ]
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root_path).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _spec(config: dict[str, Any]) -> ShapeGeneratorSpec:
    data = config["data"]
    intervention = data["intervention"]
    return ShapeGeneratorSpec(
        context_length=int(data["context_length"]),
        horizon=int(data["horizon"]),
        target_ar_range=tuple(float(x) for x in data["target_ar_range"]),
        covariate_ar_range=tuple(float(x) for x in data["covariate_ar_range"]),
        coefficient_magnitude_range=tuple(float(x) for x in data["coefficient_magnitude_range"]),
        innovation_sd_range=tuple(float(x) for x in data["innovation_sd_range"]),
        intervention_scale=float(intervention["scale_in_covariate_sd"]),
        pulse_lengths=tuple(int(x) for x in intervention["pulse_lengths"]),
        start_indices=tuple(int(x) for x in intervention["start_indices"]),
        clip_quantiles=tuple(float(x) for x in intervention["clip_to_context_quantiles"]),
    )


def _validate_bundle(bundle: ForecastBundle, count: int, horizon: int) -> None:
    if bundle.median.shape != (count, horizon):
        raise RuntimeError(f"unexpected median shape {bundle.median.shape}")
    if bundle.quantiles is None:
        raise RuntimeError("P1-SHAPE requires probabilistic forecasts")
    if bundle.quantiles.shape != (count, horizon, len(bundle.quantile_levels)):
        raise RuntimeError(f"unexpected quantile shape {bundle.quantiles.shape}")
    if not np.all(np.isfinite(bundle.median)) or not np.all(np.isfinite(bundle.quantiles)):
        raise RuntimeError("model produced nonfinite P1-SHAPE forecasts")


def _unit_paths(output_root: Path, backbone: str, mechanism: str, seed: int) -> tuple[Path, Path]:
    directory = output_root / "units" / backbone / mechanism
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"seed_{seed:05d}"
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def _completed_unit(
    array_path: Path,
    manifest_path: Path,
    *,
    config_hash: str,
    code_hash: str,
    count: int,
) -> bool:
    if not array_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        manifest.get("config_hash") == config_hash
        and manifest.get("scientific_code_sha256") == code_hash
        and manifest.get("series_count") == count
        and manifest.get("completed") is True
        and manifest.get("array_sha256")
        == hashlib.sha256(array_path.read_bytes()).hexdigest()
    )


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _world_view(scenario: ShapeScenario, world: str) -> ShapeScenario:
    return replace(
        scenario,
        target_future_intervened=scenario.target_future_worlds[world],
        covariate_future_intervened=scenario.covariate_future_worlds[world],
        oracle_response=scenario.oracle_response_worlds[world],
    )


def run_shape_backbone_units(
    repo_root: str | Path,
    adapter: ScenarioAdapter,
    output_root: str | Path,
    *,
    smoke_count: int | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    config_path = root / "configs" / "p1_shape" / "covintervene_shape_p1.yaml"
    lock_path = root / "configs" / "p1_shape" / "covintervene_shape_p1.lock.json"
    config_hash = verify_config_lock(config_path, lock_path)
    config = load_yaml(config_path)
    if config["status"] != "frozen_owner_approved_pre_inference":
        raise RuntimeError("P1-SHAPE config is not frozen and owner-approved")
    if adapter.backbone_id not in {model["id"] for model in config["models"]}:
        raise ValueError("adapter backbone is not frozen in P1-SHAPE")
    full_count = int(config["data"]["series_per_mechanism_per_seed"])
    count = full_count if smoke_count is None else int(smoke_count)
    if not 0 < count <= full_count:
        raise ValueError(f"smoke_count must lie in [1, {full_count}]")
    mode = "full_screening" if count == full_count else "smoke_only"
    code_hash = scientific_code_hash(root)
    spec = _spec(config)
    widths = tuple(int(x) for x in config["response_metrics"]["resolution_widths"])
    completed: list[str] = []

    for mechanism in config["data"]["mechanisms"]:
        if mechanism not in SHAPE_MECHANISMS:
            raise RuntimeError(f"unsupported frozen P1-SHAPE mechanism {mechanism}")
        for seed_value in config["data"]["generator_seeds"]:
            seed = int(seed_value)
            array_path, manifest_path = _unit_paths(
                destination, adapter.backbone_id, mechanism, seed
            )
            if _completed_unit(
                array_path,
                manifest_path,
                config_hash=config_hash,
                code_hash=code_hash,
                count=count,
            ):
                completed.append(f"{mechanism}/seed_{seed:05d}")
                continue
            scenarios = generate_shape_batch(mechanism, seed, count, spec)
            target_only = adapter.forecast(scenarios, "target_only")
            factual = adapter.forecast(scenarios, "factual")
            intervened = adapter.forecast(scenarios, "intervened")
            for bundle in (target_only, factual, intervened):
                _validate_bundle(bundle, count, spec.horizon)
            if not (
                target_only.quantile_levels == factual.quantile_levels == intervened.quantile_levels
            ):
                raise RuntimeError("P1-SHAPE forecast variants returned different quantiles")

            predicted = intervened.median.astype(np.float64) - factual.median.astype(np.float64)
            oracle = np.stack([scenario.oracle_response for scenario in scenarios])
            dsa = np.empty(count)
            rgr = np.empty(count)
            distances = np.empty((count, len(widths)))
            hidden_gap = np.empty(count)
            distortion_auc = np.empty(count)
            mass_distance = np.empty(count)
            onset = np.empty(count)
            peak = np.empty(count)
            positive_mass_error = np.empty(count)
            negative_mass_error = np.empty(count)
            interaction_distance = np.full(count, np.nan)
            sql_target = np.empty(count)
            sql_factual = np.empty(count)
            wql_target = np.empty(count)
            wql_factual = np.empty(count)

            if mechanism == "two_covariate_synergy":
                a_forecast = adapter.forecast(
                    [_world_view(scenario, "a_only") for scenario in scenarios], "intervened"
                )
                b_forecast = adapter.forecast(
                    [_world_view(scenario, "b_only") for scenario in scenarios], "intervened"
                )
                for bundle in (a_forecast, b_forecast):
                    _validate_bundle(bundle, count, spec.horizon)
                predicted_a = a_forecast.median.astype(np.float64) - factual.median
                predicted_b = b_forecast.median.astype(np.float64) - factual.median

            assert target_only.quantiles is not None
            assert factual.quantiles is not None
            for index, scenario in enumerate(scenarios):
                dsa[index] = directional_sign_agreement(predicted[index], oracle[index])
                rgr[index] = response_gain_ratio(predicted[index], oracle[index])
                suite = shape_metric_suite(predicted[index], oracle[index], widths)
                distances[index] = suite.signed_shape_distances
                hidden_gap[index] = suite.hidden_distortion_gap
                distortion_auc[index] = suite.multi_resolution_auc
                mass_distance[index] = suite.temporal_mass_distance
                onset[index] = suite.onset_error
                peak[index] = suite.peak_time_error
                positive_mass_error[index] = suite.positive_mass_relative_error
                negative_mass_error[index] = suite.negative_mass_relative_error
                if mechanism == "two_covariate_synergy":
                    predicted_interaction = interaction_response(
                        predicted[index], predicted_a[index], predicted_b[index]
                    )
                    oracle_interaction = interaction_response(
                        scenario.oracle_response_worlds["joint"],
                        scenario.oracle_response_worlds["a_only"],
                        scenario.oracle_response_worlds["b_only"],
                    )
                    interaction_distance[index] = signed_shape_distance(
                        predicted_interaction, oracle_interaction
                    )
                levels = factual.quantile_levels
                sql_target[index] = scaled_quantile_loss(
                    scenario.target_future_factual,
                    target_only.quantiles[index],
                    levels,
                    scenario.target_context,
                )
                sql_factual[index] = scaled_quantile_loss(
                    scenario.target_future_factual,
                    factual.quantiles[index],
                    levels,
                    scenario.target_context,
                )
                wql_target[index] = weighted_quantile_loss(
                    scenario.target_future_factual, target_only.quantiles[index], levels
                )
                wql_factual[index] = weighted_quantile_loss(
                    scenario.target_future_factual, factual.quantiles[index], levels
                )

            arrays = {
                "series_ids": np.asarray([scenario.series_id for scenario in scenarios]),
                "oracle_response": oracle,
                "predicted_response": predicted,
                "dsa": dsa,
                "rgr": rgr,
                "shape_distance_by_width": distances,
                "hidden_distortion_gap": hidden_gap,
                "multi_resolution_auc": distortion_auc,
                "temporal_mass_distance": mass_distance,
                "onset_error": onset,
                "peak_time_error": peak,
                "positive_mass_relative_error": positive_mass_error,
                "negative_mass_relative_error": negative_mass_error,
                "interaction_shape_distance": interaction_distance,
                "sql_target": sql_target,
                "sql_factual": sql_factual,
                "wql_target": wql_target,
                "wql_factual": wql_factual,
                "resolution_widths": np.asarray(widths),
            }
            _atomic_npz(array_path, arrays)
            _atomic_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "result_status": "smoke_only_not_scientific_evidence"
                    if mode == "smoke_only"
                    else "screening_only_not_aggregate_result",
                    "completed": True,
                    "mode": mode,
                    "config_hash": config_hash,
                    "scientific_code_sha256": code_hash,
                    "git_commit": _git_commit(root),
                    "backbone": adapter.backbone_id,
                    "mechanism": mechanism,
                    "generator_seed": seed,
                    "series_count": count,
                    "array_file": array_path.name,
                    "array_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
                    "created_at_utc": datetime.now(UTC).isoformat(),
                },
            )
            completed.append(f"{mechanism}/seed_{seed:05d}")

    report = {
        "schema_version": 1,
        "result_status": "smoke_only_not_scientific_evidence"
        if mode == "smoke_only"
        else "screening_units_complete_not_analyzed",
        "mode": mode,
        "config_hash": config_hash,
        "scientific_code_sha256": code_hash,
        "git_commit": _git_commit(root),
        "backbone": adapter.backbone_id,
        "series_per_mechanism_per_seed": count,
        "completed_unit_count": len(completed),
        "completed_units": completed,
        "scientific_gate_computed": False,
        "runtime": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    _atomic_json(destination / f"{adapter.backbone_id}_{mode}_completion.json", report)
    return report


def _interval_payload(interval: BootstrapInterval) -> dict[str, Any]:
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "replicates": interval.replicates,
        "seed": interval.seed,
    }


def analyze_complete_shape(repo_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    config_path = root / "configs" / "p1_shape" / "covintervene_shape_p1.yaml"
    config_hash = verify_config_lock(
        config_path,
        root / "configs" / "p1_shape" / "covintervene_shape_p1.lock.json",
    )
    config = load_yaml(config_path)
    code_hash = scientific_code_hash(root)
    count = int(config["data"]["series_per_mechanism_per_seed"])
    replicates = int(config["uncertainty"]["bootstrap_replicates"])
    bootstrap_seed = int(config["uncertainty"]["bootstrap_seed"])
    cells: list[ShapeCellSummary] = []
    details: dict[str, Any] = {}
    for model in config["models"]:
        backbone = model["id"]
        for mechanism in config["data"]["mechanisms"]:
            loaded_units: list[dict[str, np.ndarray]] = []
            labels: list[int] = []
            for seed_value in config["data"]["generator_seeds"]:
                seed = int(seed_value)
                array_path, manifest_path = _unit_paths(
                    destination, backbone, mechanism, seed
                )
                if not _completed_unit(
                    array_path,
                    manifest_path,
                    config_hash=config_hash,
                    code_hash=code_hash,
                    count=count,
                ):
                    raise RuntimeError(f"incomplete P1-SHAPE unit: {backbone}/{mechanism}/{seed}")
                with np.load(array_path, allow_pickle=False) as loaded:
                    loaded_units.append({key: loaded[key] for key in loaded.files})
                labels.extend([seed] * count)
            strata = np.asarray(labels)
            intervals: dict[str, BootstrapInterval] = {}
            metric_specs = (
                ("dsa", "dsa", np.mean),
                ("rgr", "rgr", np.median),
                ("shape_d1", "shape_distance_by_width", np.median),
                ("hidden_gap", "hidden_distortion_gap", np.median),
            )
            for label, array_name, statistic in metric_specs:
                if array_name == "shape_distance_by_width":
                    values = np.concatenate([unit[array_name][:, 0] for unit in loaded_units])
                else:
                    values = np.concatenate([unit[array_name] for unit in loaded_units])
                intervals[label] = stratified_series_bootstrap(
                    values,
                    strata,
                    statistic=statistic,
                    replicates=replicates,
                    seed=bootstrap_seed,
                )
            sql_target = np.concatenate([unit["sql_target"] for unit in loaded_units])
            sql_factual = np.concatenate([unit["sql_factual"] for unit in loaded_units])
            relative_sql = float((np.mean(sql_factual) - np.mean(sql_target)) / np.mean(sql_target))
            cell = ShapeCellSummary(
                backbone=backbone,
                mechanism=mechanism,
                relative_sql_difference=relative_sql,
                dsa=intervals["dsa"],
                median_rgr=intervals["rgr"],
                median_shape_distance_width_1=intervals["shape_d1"],
                median_hidden_distortion_gap=intervals["hidden_gap"],
            )
            cells.append(cell)
            details[f"{backbone}/{mechanism}"] = {
                "series_count": int(strata.size),
                "relative_sql_difference": relative_sql,
                **{name: _interval_payload(value) for name, value in intervals.items()},
            }
    gate = config["continuation_gate"]
    decision = evaluate_shape_gate(
        cells,
        dsa_lower_minimum=float(gate["coarse_eligibility"]["dsa_lower_bound_minimum"]),
        rgr_interval=tuple(float(x) for x in gate["coarse_eligibility"]["rgr_complete_interval"]),
        shape_distance_lower_minimum=float(
            gate["hidden_shape_violation"]["signed_shape_distance_width_1_lower_bound_minimum"]
        ),
        hidden_gap_lower_minimum=float(
            gate["hidden_shape_violation"]["hidden_distortion_gap_lower_bound_minimum"]
        ),
        minimum_passing_cells=int(gate["minimum_passing_cells"]),
        complementary_relative_sql_maximum=float(
            gate["complementary_relative_sql_maximum"]
        ),
    )
    report = {
        "schema_version": 1,
        "result_status": "verified_p1_shape_decision",
        "paper_eligibility": "eligible_for_bounded_paper_claim"
        if decision.passed
        else "blocked_by_frozen_p1_shape_gate",
        "config_hash": config_hash,
        "scientific_code_sha256": code_hash,
        "git_commit": _git_commit(root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cells": details,
        "decision": {
            "passed": decision.passed,
            "passing_cells": list(decision.passing_cells),
            "checks_by_cell": decision.checks_by_cell,
            "minimum_cell_count_passed": decision.minimum_cell_count_passed,
            "diversity_passed": decision.diversity_passed,
            "complementary_accuracy_passed": decision.complementary_accuracy_passed,
            "next_action": "draft_bounded_paper_claim"
            if decision.passed
            else gate["failure_action"],
        },
    }
    decision_path = destination / "p1_shape_decision.yaml"
    temporary = decision_path.with_suffix(".tmp.yaml")
    temporary.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    os.replace(temporary, decision_path)
    return report
