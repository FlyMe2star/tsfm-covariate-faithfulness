"""Audit complete sealed P3-v2 arrays, then apply the unchanged P1 cell rule."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from covfaith.config import canonical_config_hash
from covfaith.forecast_metrics import scaled_quantile_loss, weighted_quantile_loss
from covfaith.metrics import directional_sign_agreement, placebo_response_ratio, response_gain_ratio
from covfaith.shape_metrics import signed_shape_distance
from covfaith.shape_statistics import ShapeCellSummary, shape_cell_checks
from covfaith.statistics import BootstrapInterval
from covfaith_p3.data import sha256_file
from covfaith_p3_v2.constructs import generate_v2_scenario
from covfaith_p3_v2_model.runner import (
    load_frozen_contract,
    load_frozen_windows,
    prepare_frozen_units,
)
from covfaith_p3_v2_model.runner import (
    scientific_code_hash as model_code_hash,
)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def scientific_code_hash(repo: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((repo / "src/covfaith_p3_v2_analysis").glob("*.py")):
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def cluster_bootstrap(
    source_ids: list[str],
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    replicates: int,
    seed: int,
) -> BootstrapInterval:
    """Resample original IDs; each sampled ID carries all its valid scenario seeds."""
    numeric = np.asarray(values, dtype=np.float64)
    if not source_ids or numeric.shape != (len(source_ids),) or not np.all(np.isfinite(numeric)):
        raise ValueError("cluster bootstrap requires aligned finite scenario values")
    distinct = sorted(set(source_ids))
    groups = [numeric[np.asarray(source_ids) == key] for key in distinct]
    if any(group.size == 0 for group in groups):
        raise ValueError("empty original-source-ID cluster")
    estimate = float(statistic(numeric))
    rng = np.random.default_rng(seed)
    simulated = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        chosen = rng.integers(0, len(groups), len(groups))
        simulated[index] = float(statistic(np.concatenate([groups[item] for item in chosen])))
    if not np.isfinite(estimate) or not np.all(np.isfinite(simulated)):
        raise ValueError("bootstrap statistic must be finite")
    lower, upper = np.quantile(simulated, [0.025, 0.975])
    return BootstrapInterval(estimate, float(lower), float(upper), replicates, seed)


def _complete_records(
    repo: Path,
    data_root: Path,
    archive_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    config, construct = load_frozen_contract(repo)
    selected = load_frozen_windows(repo, data_root, config, construct)
    expected_units = prepare_frozen_units(selected, config, construct)
    runner_hash = model_code_hash(repo)
    entries: list[dict[str, str]] = []
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_keys = sorted(expected_units)
    horizon = int(config["data"]["horizon"])
    context_length = int(config["data"]["context_length"])
    source_count = int(config["data"]["source_ids_per_dataset"])
    seed_values = [int(value) for value in config["data"]["scenario_seeds"]]
    for model in config["models"]:
        backbone = model["id"]
        completion_path = archive_root / "full_units" / backbone / "completion.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected_labels = [
            f"{dataset}/{family}/seed_{seed:05d}" for dataset, family, seed in expected_keys
        ]
        if (
            completion.get("result_status") != "sealed_units_complete_not_analyzed"
            or completion.get("mode") != "full_units"
            or completion.get("config_hash") != construct["config_hash"]
            or completion.get("model_runner_code_sha256") != runner_hash
            or completion.get("construct_report_sha256")
            != sha256_file(
                repo / "evidence/p3_semisynthetic/v2_construct/p3_v2_construct_preflight.json"
            )
            or completion.get("v2_selection_manifest_sha256")
            != construct["v2_selection_manifest_sha256"]
            or completion.get("backbone") != backbone
            or completion.get("checkpoint") != model["checkpoint"]
            or completion.get("checkpoint_revision") != model["revision"]
            or completion.get("completed_unit_count") != len(expected_keys)
            or completion.get("completed_units") != expected_labels
            or completion.get("valid_scenario_count") != sum(map(len, expected_units.values()))
            or completion.get("scientific_gate_computed") is not False
        ):
            raise RuntimeError(f"{backbone}: completion receipt differs from frozen matrix")
        entries.append(
            {
                "path": f"full_units/{backbone}/completion.json",
                "sha256": sha256_file(completion_path),
            }
        )
        windows_by_dataset = {
            dataset: {window.source_id: window for window in windows}
            for dataset, windows in selected.items()
        }
        for dataset, family, seed in expected_keys:
            scenarios = expected_units[(dataset, family, seed)]
            stem = archive_root / "full_units" / backbone / dataset / family / f"seed_{seed:05d}"
            array_path = stem.with_suffix(".npz")
            manifest_path = stem.with_suffix(".json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("completed") is not True
                or manifest.get("result_status") != "sealed_unit_only"
                or manifest.get("config_hash") != construct["config_hash"]
                or manifest.get("model_runner_code_sha256") != runner_hash
                or manifest.get("v2_selection_manifest_sha256")
                != construct["v2_selection_manifest_sha256"]
                or manifest.get("backbone") != backbone
                or manifest.get("checkpoint") != model["checkpoint"]
                or manifest.get("checkpoint_revision") != model["revision"]
                or (manifest.get("dataset"), manifest.get("family"), manifest.get("seed"))
                != (dataset, family, seed)
                or manifest.get("valid_scenario_count") != len(scenarios)
                or manifest.get("excluded_scenario_count") != source_count - len(scenarios)
                or manifest.get("array_sha256") != sha256_file(array_path)
            ):
                raise RuntimeError(f"{backbone}/{dataset}/{family}/{seed}: manifest mismatch")
            for path in (manifest_path, array_path):
                entries.append(
                    {
                        "path": path.relative_to(archive_root).as_posix(),
                        "sha256": sha256_file(path),
                    }
                )
            with np.load(array_path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            count = len(scenarios)
            try:
                _check_unit_arrays(
                    arrays,
                    scenarios,
                    windows_by_dataset[dataset],
                    config,
                    horizon=horizon,
                    context_length=context_length,
                    seed_rank=seed_values.index(seed),
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"{backbone}/{dataset}/{family}/seed_{seed:05d}: {error}"
                ) from error
            if manifest.get("sham_scenario_count") != len(arrays["sham_indices"]) or manifest.get(
                "lower_link_scenario_count"
            ) != len(arrays["lower_indices"]):
                raise RuntimeError("control subset count differs from unit manifest")
            for index in range(count):
                records[f"{backbone}/{dataset}/{family}"].append(
                    {
                        "source_id": scenarios[index].source_id,
                        "scenario": scenarios[index],
                        "arrays": arrays,
                        "index": index,
                    }
                )
    return config, construct, dict(records), sorted(entries, key=lambda item: item["path"])


def _check_unit_arrays(
    arrays: dict[str, np.ndarray],
    scenarios: list[Any],
    windows_by_id: dict[str, Any],
    config: dict[str, Any],
    *,
    horizon: int,
    context_length: int,
    seed_rank: int,
) -> None:
    count = len(scenarios)
    expected_shapes = {
        "source_id": (count,),
        "source_rank": (count,),
        "origin": (count,),
        "target_context": (count, context_length),
        "target_future_factual": (count, horizon),
        "oracle_response": (count, horizon),
        "median_target_only": (count, horizon),
        "median_factual": (count, horizon),
        "median_active": (count, horizon),
    }
    for name, shape in expected_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise RuntimeError(f"P3-v2 private array {name} has wrong shape")
    if not np.array_equal(arrays["source_id"], [item.source_id for item in scenarios]):
        raise RuntimeError("model archive has changed source IDs or scenario order")
    if not np.array_equal(arrays["source_rank"], [item.source_rank for item in scenarios]):
        raise RuntimeError("model archive has changed source ranks")
    if not np.array_equal(arrays["origin"], [item.origin for item in scenarios]):
        raise RuntimeError("model archive has changed selected origins")
    for name, attribute in (
        ("target_context", "target_context_factual"),
        ("target_future_factual", "target_future_factual"),
        ("oracle_response", "oracle_response"),
    ):
        expected = np.stack([getattr(item, attribute) for item in scenarios])
        actual = arrays[name]
        if not np.array_equal(actual, expected):
            absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            scale = np.maximum(1.0, np.abs(expected.astype(np.float64)))
            raise RuntimeError(
                f"{name}: model archive differs from frozen construct replay "
                f"(mismatched_elements={int(np.count_nonzero(actual != expected))}/{actual.size}, "
                f"max_abs={float(np.max(absolute)):.6g}, "
                f"max_scaled={float(np.max(absolute / scale)):.6g}, "
                f"stored_dtype={actual.dtype}, replay_dtype={expected.dtype})"
            )
    levels = arrays.get("quantile_levels")
    if levels is None or levels.ndim != 1 or not np.all((levels > 0) & (levels < 1)):
        raise RuntimeError("missing or invalid model quantile levels")
    if np.any(np.diff(levels) <= 0):
        raise RuntimeError("model quantile levels are not strictly increasing")
    for name in ("quantiles_target_only", "quantiles_factual"):
        if arrays.get(name) is None or arrays[name].shape != (count, horizon, len(levels)):
            raise RuntimeError(f"{name}: missing or misaligned quantile forecasts")
    subset = np.asarray(
        [i for i, item in enumerate(scenarios) if item.source_rank < 12], dtype=np.int64
    )
    if not np.array_equal(arrays.get("sham_indices"), subset) or not np.array_equal(
        arrays.get("lower_indices"), subset
    ):
        raise RuntimeError("sham/lower subset differs from frozen first-12-ID rule")
    for name in (
        "median_sham",
        "median_lower_factual",
        "median_lower_active",
        "lower_oracle_response",
    ):
        if arrays.get(name) is None or arrays[name].shape != (len(subset), horizon):
            raise RuntimeError(f"{name}: control array has wrong shape")
    for name, values in arrays.items():
        if name != "source_id" and values.dtype.kind in "fi" and not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name}: nonfinite value in sealed unit")
    for position, index in enumerate(subset):
        scenario = scenarios[index]
        lower = generate_v2_scenario(
            windows_by_id[scenario.source_id],
            scenario.family,
            scenario.seed,
            seed_rank,
            config["data"],
            config["mechanism"],
            multiplier_cap=float(config["effect_strength_sensitivity"]["lower_multiplier_cap"]),
        )
        if not np.array_equal(arrays["lower_oracle_response"][position], lower.oracle_response):
            raise RuntimeError("lower-link oracle differs from frozen construct")


def _metric_values(record: dict[str, Any]) -> dict[str, float | None]:
    arrays, index = record["arrays"], record["index"]
    oracle = arrays["oracle_response"][index].astype(np.float64)
    predicted = arrays["median_active"][index].astype(np.float64) - arrays["median_factual"][
        index
    ].astype(np.float64)
    outcome: dict[str, float | None] = {
        "dsa": directional_sign_agreement(predicted, oracle),
        "rgr": response_gain_ratio(predicted, oracle),
    }
    for width in (1, 2, 4, 8):
        try:
            outcome[f"d{width}"] = signed_shape_distance(predicted, oracle, width)
        except ValueError:
            outcome[f"d{width}"] = None
    outcome["g"] = (
        outcome["d1"] - outcome["d8"]
        if outcome["d1"] is not None and outcome["d8"] is not None
        else None
    )
    truth = arrays["target_future_factual"][index]
    context = arrays["target_context"][index]
    levels = arrays["quantile_levels"]
    outcome["sql_target"] = scaled_quantile_loss(
        truth, arrays["quantiles_target_only"][index], levels, context
    )
    outcome["sql_factual"] = scaled_quantile_loss(
        truth, arrays["quantiles_factual"][index], levels, context
    )
    outcome["wql_target"] = weighted_quantile_loss(
        truth, arrays["quantiles_target_only"][index], levels
    )
    outcome["wql_factual"] = weighted_quantile_loss(
        truth, arrays["quantiles_factual"][index], levels
    )
    indices = np.flatnonzero(arrays["sham_indices"] == index)
    if indices.size:
        position = int(indices[0])
        sham_response = arrays["median_sham"][position].astype(np.float64) - arrays[
            "median_factual"
        ][index].astype(np.float64)
        outcome["sham_ratio"] = placebo_response_ratio(sham_response, oracle)
        lower_predicted = arrays["median_lower_active"][position].astype(np.float64) - arrays[
            "median_lower_factual"
        ][position].astype(np.float64)
        lower_oracle = arrays["lower_oracle_response"][position].astype(np.float64)
        outcome["lower_dsa"] = directional_sign_agreement(lower_predicted, lower_oracle)
        outcome["lower_rgr"] = response_gain_ratio(lower_predicted, lower_oracle)
        try:
            lower_d1 = signed_shape_distance(lower_predicted, lower_oracle, 1)
            lower_d8 = signed_shape_distance(lower_predicted, lower_oracle, 8)
            outcome["lower_d1"] = lower_d1
            outcome["lower_g"] = lower_d1 - lower_d8
        except ValueError:
            outcome["lower_d1"] = None
            outcome["lower_g"] = None
    return outcome


def _interval(
    rows: list[dict[str, Any]],
    metric: str,
    statistic: Callable[[np.ndarray], float],
    config: dict[str, Any],
) -> dict[str, Any]:
    selected = [row for row in rows if metric in row["metrics"]]
    missing = sum(row["metrics"][metric] is None for row in selected)
    if not selected or missing:
        return {
            "estimate": None,
            "lower": None,
            "upper": None,
            "n": len(selected),
            "undefined": missing,
        }
    uncertainty = config["response_metrics"]["uncertainty"]
    interval = cluster_bootstrap(
        [row["source_id"] for row in selected],
        np.asarray([row["metrics"][metric] for row in selected], dtype=np.float64),
        statistic,
        replicates=int(uncertainty["bootstrap_replicates"]),
        seed=int(uncertainty["bootstrap_seed"]),
    )
    return {
        **asdict(interval),
        "n": len(selected),
        "cluster_count": len({row["source_id"] for row in selected}),
        "undefined": 0,
    }


def _relative_loss(rows: list[dict[str, Any]], base: str, candidate: str) -> float | None:
    baseline = float(np.mean([row["metrics"][base] for row in rows]))
    chosen = float(np.mean([row["metrics"][candidate] for row in rows]))
    return (chosen - baseline) / baseline if np.isfinite(baseline) and baseline > 0 else None


def summarize_cell(
    backbone: str,
    dataset: str,
    family: str,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    p1: dict[str, Any],
) -> dict[str, Any]:
    rows = [{"source_id": item["source_id"], "metrics": _metric_values(item)} for item in records]
    intervals = {
        "dsa": _interval(rows, "dsa", np.mean, config),
        "rgr": _interval(rows, "rgr", np.median, config),
        "d1": _interval(rows, "d1", np.median, config),
        "g": _interval(rows, "g", np.median, config),
    }
    diagnostic = {
        key: _interval(rows, key, np.median, config)
        for key in ("d2", "d4", "d8", "sham_ratio", "lower_dsa", "lower_rgr", "lower_d1", "lower_g")
    }
    relative_sql = _relative_loss(rows, "sql_target", "sql_factual")
    relative_wql = _relative_loss(rows, "wql_target", "wql_factual")
    criteria = p1["continuation_gate"]
    assessable = relative_sql is not None and all(
        intervals[key]["estimate"] is not None for key in intervals
    )
    if assessable:
        shape_cell = ShapeCellSummary(
            backbone,
            family,
            relative_sql,
            *(
                BootstrapInterval(
                    **{
                        key: intervals[name][key]
                        for key in ("estimate", "lower", "upper", "replicates", "seed")
                    }
                )
                for name in ("dsa", "rgr", "d1", "g")
            ),
        )
        checks = shape_cell_checks(
            shape_cell,
            dsa_lower_minimum=float(criteria["coarse_eligibility"]["dsa_lower_bound_minimum"]),
            rgr_interval=tuple(criteria["coarse_eligibility"]["rgr_complete_interval"]),
            shape_distance_lower_minimum=float(
                criteria["hidden_shape_violation"][
                    "signed_shape_distance_width_1_lower_bound_minimum"
                ]
            ),
            hidden_gap_lower_minimum=float(
                criteria["hidden_shape_violation"]["hidden_distortion_gap_lower_bound_minimum"]
            ),
        )
        checks["sql_complement"] = relative_sql <= float(
            criteria["complementary_relative_sql_maximum"]
        )
    else:
        checks = {
            name: False
            for name in (
                "dsa_coarse_eligible",
                "rgr_coarse_eligible",
                "fine_shape_distorted",
                "distortion_hidden_by_aggregation",
                "sql_complement",
            )
        }
    return {
        "backbone": backbone,
        "dataset": dataset,
        "family": family,
        "valid_scenario_count": len(rows),
        "source_id_cluster_count": len({row["source_id"] for row in rows}),
        "primary": intervals,
        "diagnostic_and_controls": diagnostic,
        "relative_sql": relative_sql,
        "relative_wql": relative_wql,
        "lower_link_relative_sql": None,
        "lower_link_sql_status": "unavailable_lower_quantiles_not_archived_no_imputation",
        "unchanged_p1_descriptive_checks": checks,
        "complete_cell": bool(assessable and all(checks.values())),
    }


def analyze_full_archives(
    repo: str | Path, data_root: str | Path, archive_root: str | Path
) -> dict[str, Any]:
    """Read existing forecasts only; never load a checkpoint or modify source units."""
    repo_path = Path(repo).resolve()
    data_path = Path(data_root).resolve()
    private = Path(archive_root).resolve()
    if private == repo_path or repo_path in private.parents:
        raise ValueError("private P3-v2 analysis outputs must remain outside public Git")
    config, construct, records, entries = _complete_records(repo_path, data_path, private)
    inventory_sha = hashlib.sha256(_json_bytes(entries)).hexdigest()
    analysis_hash = scientific_code_hash(repo_path)
    result_path = private / "analysis_v1" / "p3_v2_analysis_report.json"
    if result_path.exists():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        matrix = result_path.parent / "complete_twelve_cell_matrix.csv"
        if (
            old.get("source_archive_inventory_sha256") != inventory_sha
            or old.get("analysis_code_sha256") != analysis_hash
            or not matrix.exists()
            or old.get("complete_matrix_sha256") != sha256_file(matrix)
        ):
            raise RuntimeError("existing P3-v2 analysis cannot be overwritten")
        return old
    p1_path = repo_path / "configs/p1_shape/covintervene_shape_p1.yaml"
    p1 = yaml.safe_load(p1_path.read_text(encoding="utf-8"))
    if canonical_config_hash(p1) != config["parent_evidence"]["p1_config_sha256"]:
        raise RuntimeError("frozen P1 metric thresholds changed")
    if tuple(int(x) for x in p1["response_metrics"]["resolution_widths"]) != (1, 2, 4, 8):
        raise RuntimeError("frozen P1 response-resolution widths changed")
    cells = [
        summarize_cell(
            backbone, dataset, family, records[f"{backbone}/{dataset}/{family}"], config, p1
        )
        for backbone in [item["id"] for item in config["models"]]
        for dataset in [item["id"] for item in config["data"]["sources"]]
        for family in config["mechanism"]["families"]
    ]
    passing_sources = sorted({cell["dataset"] for cell in cells if cell["complete_cell"]})
    supported = len(passing_sources) >= 2
    report = {
        "schema_version": 1,
        "experiment": "covintervene-shape-p3-v2-post-primary-analysis",
        "result_status": "verified_post_primary_descriptive_analysis",
        "paper_eligibility": "bounded_secondary_evidence_only",
        "config_hash": construct["config_hash"],
        "model_runner_code_sha256": model_code_hash(repo_path),
        "analysis_code_sha256": analysis_hash,
        "source_archive_inventory_sha256": inventory_sha,
        "source_archive_file_count": len(entries),
        "source_completion_sha256": {
            item["id"]: next(
                entry["sha256"]
                for entry in entries
                if entry["path"] == f"full_units/{item['id']}/completion.json"
            )
            for item in config["models"]
        },
        "counts": {
            "complete_cell_count": len(cells),
            "valid_scenarios_per_backbone": sum(cell["valid_scenario_count"] for cell in cells)
            // 2,
            "p3_v2_construct_exclusions": construct["counts"]["excluded_scenarios"],
        },
        "cells": cells,
        "decision": {
            "transfer_support_rule": config["descriptive_transfer_rule"]["support"],
            "passing_complete_cells": [
                f"{cell['backbone']}/{cell['dataset']}/{cell['family']}"
                for cell in cells
                if cell["complete_cell"]
            ],
            "passing_sources": passing_sources,
            "bounded_transfer_supported": supported,
            "p1_primary_decision_recomputed_or_modified": False,
            "v1_failed_pilot_disclosure_required": True,
            "lower_link_sql_missing": True,
            "next_action": (
                "write_bounded_post_primary_result_with_missing_lower_sql_disclosed"
                if supported
                else "archive_bounded_nonreplication_with_all_twelve_cells"
            ),
        },
        "scope": {
            "model_inference_performed": False,
            "source_prediction_archives_modified": False,
            "p1_p2_or_p3_v1_modified": False,
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    matrix_path = result_path.parent / "complete_twelve_cell_matrix.csv"
    rows = [
        {
            "backbone": cell["backbone"],
            "dataset": cell["dataset"],
            "family": cell["family"],
            "n": cell["valid_scenario_count"],
            "source_clusters": cell["source_id_cluster_count"],
            "dsa": cell["primary"]["dsa"]["estimate"],
            "dsa_lower": cell["primary"]["dsa"]["lower"],
            "rgr": cell["primary"]["rgr"]["estimate"],
            "rgr_lower": cell["primary"]["rgr"]["lower"],
            "rgr_upper": cell["primary"]["rgr"]["upper"],
            "d1": cell["primary"]["d1"]["estimate"],
            "d1_lower": cell["primary"]["d1"]["lower"],
            "d2": cell["diagnostic_and_controls"]["d2"]["estimate"],
            "d4": cell["diagnostic_and_controls"]["d4"]["estimate"],
            "d8": cell["diagnostic_and_controls"]["d8"]["estimate"],
            "g": cell["primary"]["g"]["estimate"],
            "g_lower": cell["primary"]["g"]["lower"],
            "relative_sql": cell["relative_sql"],
            "relative_wql": cell["relative_wql"],
            "sham_ratio": cell["diagnostic_and_controls"]["sham_ratio"]["estimate"],
            "lower_dsa": cell["diagnostic_and_controls"]["lower_dsa"]["estimate"],
            "lower_rgr": cell["diagnostic_and_controls"]["lower_rgr"]["estimate"],
            "lower_d1": cell["diagnostic_and_controls"]["lower_d1"]["estimate"],
            "lower_g": cell["diagnostic_and_controls"]["lower_g"]["estimate"],
            "lower_relative_sql": cell["lower_link_relative_sql"],
            "complete_cell": cell["complete_cell"],
        }
        for cell in cells
    ]
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    matrix_bytes = stream.getvalue().encode("utf-8")
    report["complete_matrix_sha256"] = hashlib.sha256(matrix_bytes).hexdigest()
    _atomic_write(matrix_path, matrix_bytes)
    _atomic_write(result_path, _json_bytes(report))
    return report
