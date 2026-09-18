"""Resumable P0 execution and sealed analysis."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from covfaith.adapters import ForecastBundle, ScenarioAdapter
from covfaith.config import load_yaml, verify_config_lock
from covfaith.forecast_metrics import median_mae, scaled_quantile_loss, weighted_quantile_loss
from covfaith.generators import MECHANISMS, GeneratorSpec, generate_mechanism_batch
from covfaith.metrics import placebo_response_ratio, response_metric_suite
from covfaith.statistics import CellSummary, evaluate_continuation_gate, stratified_series_bootstrap


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def scientific_code_hash(root: Path) -> str:
    """Hash the ordered source and frozen-config bytes used by P0."""

    paths = sorted((root / "src" / "covfaith").glob("*.py"))
    paths.extend(
        [
            root / "configs" / "p0" / "covintervene_p0.yaml",
            root / "configs" / "p0" / "covintervene_p0.lock.json",
        ]
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _generator_spec(config: dict[str, Any]) -> GeneratorSpec:
    data = config["data"]
    intervention = data["intervention"]
    delayed = data["delayed_distributed_lag"]
    saturation = data["threshold_saturation"]
    return GeneratorSpec(
        context_length=int(data["context_length"]),
        horizon=int(data["horizon"]),
        target_ar_range=tuple(float(x) for x in data["target_ar_range"]),
        covariate_ar_range=tuple(float(x) for x in data["covariate_ar_range"]),
        coefficient_magnitude_range=tuple(float(x) for x in data["coefficient_magnitude_range"]),
        innovation_sd_range=tuple(float(x) for x in data["innovation_sd_range"]),
        intervention_scale=float(intervention["primary_scale_in_covariate_sd"]),
        block_length=int(intervention["block_length"]),
        start_indices=tuple(int(x) for x in intervention["start_indices"]),
        clip_quantiles=tuple(float(x) for x in intervention["clip_to_context_quantiles"]),
        first_lag=int(delayed["first_lag"]),
        lag_kernel=tuple(float(x) for x in delayed["kernel"]),
        transition_width=float(saturation["transition_width_in_covariate_sd"]),
    )


def _validate_bundle(bundle: ForecastBundle, count: int, horizon: int) -> None:
    if bundle.median.shape != (count, horizon):
        raise RuntimeError(f"unexpected median shape {bundle.median.shape}")
    if bundle.quantiles is None:
        raise RuntimeError("P0 requires probabilistic quantile forecasts")
    if bundle.quantiles.shape != (count, horizon, len(bundle.quantile_levels)):
        raise RuntimeError(f"unexpected quantile shape {bundle.quantiles.shape}")
    if not np.all(np.isfinite(bundle.median)) or not np.all(np.isfinite(bundle.quantiles)):
        raise RuntimeError("model produced nonfinite forecasts")


def _unit_paths(output_root: Path, backbone: str, mechanism: str, seed: int) -> tuple[Path, Path]:
    unit_dir = output_root / "units" / backbone / mechanism
    unit_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seed_{seed:05d}"
    return unit_dir / f"{stem}.npz", unit_dir / f"{stem}.json"


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
    )


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def run_backbone_units(
    repo_root: str | Path,
    adapter: ScenarioAdapter,
    output_root: str | Path,
    *,
    smoke_count: int | None = None,
) -> dict[str, Any]:
    """Run or resume one backbone; smoke mode never computes the P0 gate."""

    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    config_path = root / "configs" / "p0" / "covintervene_p0.yaml"
    lock_path = root / "configs" / "p0" / "covintervene_p0.lock.json"
    config_hash = verify_config_lock(config_path, lock_path)
    config = load_yaml(config_path)
    if config["status"] != "frozen_owner_approved_pre_inference":
        raise RuntimeError("P0 config is not in the owner-approved frozen state")
    model_ids = {model["id"] for model in config["models"]}
    if adapter.backbone_id not in model_ids:
        raise ValueError(f"adapter backbone {adapter.backbone_id!r} is not frozen in P0")

    full_count = int(config["data"]["series_per_mechanism_per_seed"])
    count = full_count if smoke_count is None else int(smoke_count)
    if not 0 < count <= full_count:
        raise ValueError(f"smoke_count must lie in [1, {full_count}]")
    mode = "full_screening" if count == full_count else "smoke_only"
    code_hash = scientific_code_hash(root)
    spec = _generator_spec(config)
    completed: list[str] = []

    for mechanism in config["data"]["mechanisms"]:
        if mechanism not in MECHANISMS:
            raise RuntimeError(f"unsupported frozen mechanism {mechanism}")
        for seed in config["data"]["generator_seeds"]:
            array_path, manifest_path = _unit_paths(
                destination,
                adapter.backbone_id,
                mechanism,
                int(seed),
            )
            if _completed_unit(
                array_path,
                manifest_path,
                config_hash=config_hash,
                code_hash=code_hash,
                count=count,
            ):
                completed.append(f"{mechanism}/seed_{int(seed):05d}")
                continue

            scenarios = generate_mechanism_batch(mechanism, int(seed), count, spec)
            target_only = adapter.forecast(scenarios, "target_only")
            factual = adapter.forecast(scenarios, "factual")
            intervened = adapter.forecast(scenarios, "intervened")
            for bundle in (target_only, factual, intervened):
                _validate_bundle(bundle, count, spec.horizon)
            if not (
                target_only.quantile_levels == factual.quantile_levels == intervened.quantile_levels
            ):
                raise RuntimeError("forecast variants returned different quantile levels")

            predicted_response = intervened.median.astype(np.float64) - factual.median.astype(
                np.float64
            )
            oracle = np.stack([scenario.oracle_response for scenario in scenarios])
            matched_active = np.stack(
                [scenario.matched_active_oracle_response for scenario in scenarios]
            )
            dsa = np.full(count, np.nan, dtype=np.float64)
            rgr = np.full(count, np.nan, dtype=np.float64)
            nre = np.full(count, np.nan, dtype=np.float64)
            tls = np.full(count, np.nan, dtype=np.float64)
            prr = np.full(count, np.nan, dtype=np.float64)
            sql_target = np.empty(count, dtype=np.float64)
            sql_factual = np.empty(count, dtype=np.float64)
            wql_target = np.empty(count, dtype=np.float64)
            wql_factual = np.empty(count, dtype=np.float64)
            mae_target = np.empty(count, dtype=np.float64)
            mae_factual = np.empty(count, dtype=np.float64)

            assert target_only.quantiles is not None
            assert factual.quantiles is not None
            for index, scenario in enumerate(scenarios):
                if mechanism == "irrelevant_placebo":
                    prr[index] = placebo_response_ratio(
                        predicted_response[index],
                        matched_active[index],
                    )
                else:
                    metrics = response_metric_suite(predicted_response[index], oracle[index])
                    dsa[index] = metrics.dsa
                    rgr[index] = metrics.rgr
                    nre[index] = metrics.nre
                    tls[index] = metrics.tls
                levels = target_only.quantile_levels
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
                    scenario.target_future_factual,
                    target_only.quantiles[index],
                    levels,
                )
                wql_factual[index] = weighted_quantile_loss(
                    scenario.target_future_factual,
                    factual.quantiles[index],
                    levels,
                )
                mae_target[index] = median_mae(
                    scenario.target_future_factual,
                    target_only.median[index],
                )
                mae_factual[index] = median_mae(
                    scenario.target_future_factual,
                    factual.median[index],
                )

            arrays = {
                "series_ids": np.asarray([scenario.series_id for scenario in scenarios]),
                "target_future_factual": np.stack(
                    [scenario.target_future_factual for scenario in scenarios]
                ),
                "oracle_response": oracle,
                "matched_active_oracle_response": matched_active,
                "target_only_median": target_only.median,
                "factual_median": factual.median,
                "intervened_median": intervened.median,
                "predicted_response": predicted_response,
                "dsa": dsa,
                "rgr": rgr,
                "nre": nre,
                "tls": tls,
                "prr": prr,
                "sql_target": sql_target,
                "sql_factual": sql_factual,
                "wql_target": wql_target,
                "wql_factual": wql_factual,
                "mae_target": mae_target,
                "mae_factual": mae_factual,
                "quantile_levels": np.asarray(target_only.quantile_levels),
            }
            _atomic_npz(array_path, arrays)
            manifest = {
                "schema_version": 1,
                "result_status": "smoke_only_not_scientific_evidence"
                if mode == "smoke_only"
                else "screening_only_not_final_paper_evidence",
                "completed": True,
                "mode": mode,
                "config_hash": config_hash,
                "scientific_code_sha256": code_hash,
                "git_commit": _git_commit(root),
                "backbone": adapter.backbone_id,
                "mechanism": mechanism,
                "generator_seed": int(seed),
                "series_count": count,
                "array_file": array_path.name,
                "array_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
                "created_at_utc": datetime.now(UTC).isoformat(),
            }
            _atomic_json(manifest_path, manifest)
            completed.append(f"{mechanism}/seed_{int(seed):05d}")

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
    report_path = destination / f"{adapter.backbone_id}_{mode}_completion.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, report)
    return report


def _interval_dict(interval: Any) -> dict[str, Any]:
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "replicates": interval.replicates,
        "seed": interval.seed,
    }


def analyze_complete_p0(repo_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Analyze only a complete full P0 inventory and write the one frozen decision."""

    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    config_path = root / "configs" / "p0" / "covintervene_p0.yaml"
    config_hash = verify_config_lock(
        config_path,
        root / "configs" / "p0" / "covintervene_p0.lock.json",
    )
    config = load_yaml(config_path)
    code_hash = scientific_code_hash(root)
    count = int(config["data"]["series_per_mechanism_per_seed"])
    bootstrap = config["uncertainty"]
    cells: list[CellSummary] = []
    details: dict[str, Any] = {}

    for model in config["models"]:
        backbone = model["id"]
        for mechanism in config["data"]["mechanisms"]:
            arrays_by_seed: list[dict[str, np.ndarray]] = []
            labels: list[int] = []
            for seed in config["data"]["generator_seeds"]:
                array_path, manifest_path = _unit_paths(
                    destination,
                    backbone,
                    mechanism,
                    int(seed),
                )
                if not _completed_unit(
                    array_path,
                    manifest_path,
                    config_hash=config_hash,
                    code_hash=code_hash,
                    count=count,
                ):
                    unit = f"{backbone}/{mechanism}/{seed}"
                    raise RuntimeError(f"incomplete or mismatched P0 unit: {unit}")
                with np.load(array_path, allow_pickle=False) as loaded:
                    arrays_by_seed.append({key: loaded[key] for key in loaded.files})
                labels.extend([int(seed)] * count)

            strata = np.asarray(labels)
            sql_target = np.concatenate([arrays["sql_target"] for arrays in arrays_by_seed])
            sql_factual = np.concatenate([arrays["sql_factual"] for arrays in arrays_by_seed])
            relative_sql = float((np.mean(sql_factual) - np.mean(sql_target)) / np.mean(sql_target))
            key = f"{backbone}/{mechanism}"
            cell_detail: dict[str, Any] = {
                "relative_sql_difference": relative_sql,
                "series_count": int(strata.size),
            }
            if mechanism == "irrelevant_placebo":
                prr_values = np.concatenate([arrays["prr"] for arrays in arrays_by_seed])
                prr = stratified_series_bootstrap(
                    prr_values,
                    strata,
                    statistic=np.median,
                    replicates=int(bootstrap["bootstrap_replicates"]),
                    seed=int(bootstrap["bootstrap_seed"]),
                )
                cell = CellSummary(backbone, mechanism, relative_sql, prr=prr)
                cell_detail["PRR"] = _interval_dict(prr)
            else:
                intervals = {}
                for metric, statistic in (
                    ("dsa", np.mean),
                    ("rgr", np.median),
                    ("nre", np.median),
                    ("tls", np.median),
                ):
                    values = np.concatenate([arrays[metric] for arrays in arrays_by_seed])
                    intervals[metric] = stratified_series_bootstrap(
                        values,
                        strata,
                        statistic=statistic,
                        replicates=int(bootstrap["bootstrap_replicates"]),
                        seed=int(bootstrap["bootstrap_seed"]),
                    )
                    cell_detail[metric.upper()] = _interval_dict(intervals[metric])
                cell = CellSummary(
                    backbone,
                    mechanism,
                    relative_sql,
                    dsa=intervals["dsa"],
                    median_rgr=intervals["rgr"],
                )
            cells.append(cell)
            details[key] = cell_detail

    gate_config = config["continuation_gate"]
    decision = evaluate_continuation_gate(
        cells,
        minimum_violating_cells=int(gate_config["minimum_violating_cells"]),
        complementary_relative_sql_maximum=float(
            gate_config["complementary_relative_sql_maximum"]
        ),
    )
    report = {
        "schema_version": 1,
        "result_status": "verified_screening_decision",
        "paper_eligibility": "eligible_for_p1_confirmation"
        if decision.passed
        else "blocked_by_frozen_p0_gate",
        "config_hash": config_hash,
        "scientific_code_sha256": code_hash,
        "git_commit": _git_commit(root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cells": details,
        "decision": {
            "passed": decision.passed,
            "violating_cells": list(decision.violating_cells),
            "violations_by_cell": {
                key: list(value) for key, value in decision.violations_by_cell.items()
            },
            "minimum_cell_count_passed": decision.minimum_cell_count_passed,
            "diversity_passed": decision.diversity_passed,
            "complementary_accuracy_passed": decision.complementary_accuracy_passed,
            "next_action": "freeze_p1_confirmation_contract"
            if decision.passed
            else gate_config["failure_action"],
        },
    }
    decision_path = destination / "p0_phenomenon_decision.yaml"
    temporary = decision_path.with_suffix(".tmp.yaml")
    temporary.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    os.replace(temporary, decision_path)
    return report
