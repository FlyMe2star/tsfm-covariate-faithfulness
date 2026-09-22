"""Analysis-only robustness supplement for the frozen P1-SHAPE artifacts.

Nothing in this module performs model inference or recomputes the registered P1 gate.
The package lives outside ``covfaith`` so adding it does not change the frozen primary
scientific-code hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covfaith.config import canonical_config_hash, load_yaml
from covfaith.metrics import response_gain_ratio
from covfaith.shape_metrics import (
    block_signed_response,
    onset_error,
    peak_time_error,
    shape_metric_suite,
    temporal_mass_distance,
)
from covfaith.statistics import stratified_series_bootstrap

FloatArray = NDArray[np.float64]


def _paired_1d(predicted: ArrayLike, oracle: ArrayLike) -> tuple[FloatArray, FloatArray]:
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(oracle, dtype=np.float64)
    if pred.ndim != 1 or truth.ndim != 1 or pred.shape != truth.shape or pred.size < 2:
        raise ValueError("responses must be aligned one-dimensional vectors")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ValueError("responses must be finite")
    if float(np.sum(np.abs(truth))) <= 1e-12:
        raise ValueError("oracle response must be nonzero")
    return pred, truth


def fixed_fine_normalized_distance(
    predicted: ArrayLike,
    oracle: ArrayLike,
    width: int = 1,
) -> float:
    """Signed distance after one fine-resolution normalization.

    For nonzero responses, both trajectories are normalized once at width one before
    the signed difference is block-summed. Consequently, nested block summation is an
    L1 contraction and the distance cannot increase as the block width coarsens.
    A numerically zero prediction is assigned distance one at every width.
    """

    pred, truth = _paired_1d(predicted, oracle)
    truth_mass = float(np.sum(np.abs(truth)))
    tolerance = 1e-8 * max(1.0, truth_mass)
    pred_mass = float(np.sum(np.abs(pred)))
    if pred_mass <= tolerance:
        return 1.0
    fine_difference = pred / pred_mass - truth / truth_mass
    blocked_difference = block_signed_response(fine_difference, width)
    return float(np.clip(0.5 * np.sum(np.abs(blocked_difference)), 0.0, 1.0))


def fixed_fine_distance_curve(
    predicted: ArrayLike,
    oracle: ArrayLike,
    widths: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[int, float]:
    if len(widths) < 2 or tuple(sorted(set(widths))) != widths:
        raise ValueError("widths must be unique and strictly increasing")
    curve = {
        width: fixed_fine_normalized_distance(predicted, oracle, width) for width in widths
    }
    values = np.asarray([curve[width] for width in widths])
    if np.any(np.diff(values) > 1e-12):
        raise AssertionError("fixed-fine distance violated the coarsening contraction")
    return curve


def aggregation_contraction_gap(
    predicted: ArrayLike,
    oracle: ArrayLike,
    coarse_width: int = 8,
) -> float:
    curve = fixed_fine_distance_curve(predicted, oracle, (1, coarse_width))
    return float(curve[1] - curve[coarse_width])


def _normalized_rmse(predicted: FloatArray, oracle: FloatArray) -> float:
    denominator = float(np.sqrt(np.mean(oracle**2)))
    return float(np.sqrt(np.mean((predicted - oracle) ** 2)) / max(denominator, 1e-12))


def _cosine_distance(predicted: FloatArray, oracle: FloatArray) -> float:
    denominator = float(np.linalg.norm(predicted) * np.linalg.norm(oracle))
    if denominator <= 1e-12:
        return 1.0
    cosine = float(np.dot(predicted, oracle) / denominator)
    return float(np.clip(0.5 * (1.0 - cosine), 0.0, 1.0))


def _correlation_distance(predicted: FloatArray, oracle: FloatArray) -> float:
    if float(np.std(predicted)) <= 1e-12 or float(np.std(oracle)) <= 1e-12:
        return 1.0
    correlation = float(np.corrcoef(predicted, oracle)[0, 1])
    return float(np.clip(0.5 * (1.0 - correlation), 0.0, 1.0))


def _normalized_dtw(predicted: FloatArray, oracle: FloatArray) -> float:
    rows, columns = predicted.size, oracle.size
    costs = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            local = abs(predicted[row - 1] - oracle[column - 1])
            costs[row, column] = local + min(
                costs[row - 1, column],
                costs[row, column - 1],
                costs[row - 1, column - 1],
            )
    return float(costs[rows, columns] / max(float(np.sum(np.abs(oracle))), 1e-12))


def _fixture_response() -> FloatArray:
    response = np.zeros(24, dtype=np.float64)
    response[1:4] = [0.5, 1.0, 0.5]
    response[9:12] = [-0.3, -0.6, -0.3]
    return response


def evaluate_metric_fixture_comparison() -> list[dict[str, Any]]:
    """Compare common trajectory metrics on the frozen analytic fixture family."""

    oracle = _fixture_response()
    rebound_deleted = oracle.copy()
    rebound_deleted[rebound_deleted < 0] = 0.0
    predictions = {
        "exact_oracle": oracle,
        "positive_gain_0p70": 0.70 * oracle,
        "within_width8_shift_1": np.roll(oracle, 1),
        "cross_width8_shift_7": np.roll(oracle, 7),
        "temporal_smoothing": np.convolve(
            oracle, np.array([0.25, 0.50, 0.25]), mode="same"
        ),
        "rebound_deleted": rebound_deleted,
        "sign_flip": -oracle,
        "zero_response": np.zeros_like(oracle),
    }
    rows: list[dict[str, Any]] = []
    for fixture, prediction in predictions.items():
        registered = shape_metric_suite(prediction, oracle)
        monotone = fixed_fine_distance_curve(prediction, oracle)
        pred, truth = _paired_1d(prediction, oracle)
        rows.append(
            {
                "fixture": fixture,
                "rgr": response_gain_ratio(pred, truth),
                "registered_d1": registered.signed_shape_distances[0],
                "registered_resolution_contrast": registered.hidden_distortion_gap,
                "monotone_c1": monotone[1],
                "monotone_aggregation_gap": monotone[1] - monotone[8],
                "normalized_rmse": _normalized_rmse(pred, truth),
                "cosine_distance": _cosine_distance(pred, truth),
                "correlation_distance": _correlation_distance(pred, truth),
                "normalized_dtw": _normalized_dtw(pred, truth),
                "absolute_mass_transport": temporal_mass_distance(pred, truth),
                "onset_error": onset_error(pred, truth),
                "peak_time_error": peak_time_error(pred, truth),
            }
        )
    return rows


def supplement_scientific_code_hash(repo_root: str | Path) -> str:
    root = Path(repo_root).resolve()
    paths = sorted((root / "src" / "covfaith_supp").glob("*.py"))
    paths.append(root / "configs" / "p2_robustness" / "p2_robustness_v1.yaml")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_p2_freeze(repo_root: str | Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    config_path = root / "configs" / "p2_robustness" / "p2_robustness_v1.yaml"
    config = load_yaml(config_path)
    if config["status"] != "frozen_owner_approved_post_primary":
        raise RuntimeError("P2 robustness config is not owner-approved and frozen")
    lock_path = root / "configs" / "p2_robustness" / "p2_robustness_v1.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    config_hash = canonical_config_hash(config_path)
    if lock.get("canonical_sha256") != config_hash:
        raise RuntimeError("P2 robustness config hash does not match its lock")
    if lock.get("analysis_authorized") is not True:
        raise RuntimeError("P2 robustness lock does not authorize analysis")
    if lock.get("model_inference_authorized") is not False:
        raise RuntimeError("P2 robustness lock must prohibit model inference")
    if lock.get("primary_decision_mutation_allowed") is not False:
        raise RuntimeError("P2 robustness lock must prohibit primary-decision mutation")
    return {
        "config_hash": config_hash,
        "scientific_code_sha256": supplement_scientific_code_hash(root),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _unit_paths(root: Path, backbone: str, mechanism: str, seed: int) -> tuple[Path, Path]:
    directory = root / "units" / backbone / mechanism
    stem = f"seed_{seed:05d}"
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def _load_verified_unit(
    input_root: Path,
    backbone: str,
    mechanism: str,
    seed: int,
    *,
    expected_config_hash: str,
    expected_code_hash: str,
    expected_count: int,
) -> dict[str, np.ndarray]:
    array_path, manifest_path = _unit_paths(input_root, backbone, mechanism, seed)
    if not array_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"missing primary unit: {backbone}/{mechanism}/{seed}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "config_hash": manifest.get("config_hash") == expected_config_hash,
        "scientific_code": manifest.get("scientific_code_sha256") == expected_code_hash,
        "series_count": manifest.get("series_count") == expected_count,
        "completed": manifest.get("completed") is True,
        "array_hash": manifest.get("array_sha256")
        == hashlib.sha256(array_path.read_bytes()).hexdigest(),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            f"primary unit verification failed for {backbone}/{mechanism}/{seed}: {failed}"
        )
    with np.load(array_path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _interval(
    values: np.ndarray,
    strata: np.ndarray,
    statistic: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    result = stratified_series_bootstrap(
        values,
        strata,
        statistic=statistic,
        replicates=int(config["uncertainty"]["bootstrap_replicates"]),
        seed=int(config["uncertainty"]["bootstrap_seed"]),
    )
    return {
        "estimate": result.estimate,
        "lower": result.lower,
        "upper": result.upper,
        "replicates": result.replicates,
        "seed": result.seed,
    }


def _threshold_sensitivity(
    decision: dict[str, Any],
    d1_thresholds: list[float],
    g_thresholds: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d1_threshold in d1_thresholds:
        for g_threshold in g_thresholds:
            passing: list[str] = []
            for cell_name, cell in decision["cells"].items():
                if (
                    cell["dsa"]["lower"] >= 0.85
                    and cell["rgr"]["lower"] >= 0.65
                    and cell["rgr"]["upper"] <= 1.35
                    and cell["shape_d1"]["lower"] >= d1_threshold
                    and cell["hidden_gap"]["lower"] >= g_threshold
                ):
                    passing.append(cell_name)
            backbones = {name.split("/", 1)[0] for name in passing}
            mechanisms = {name.split("/", 1)[1] for name in passing}
            diversity = len(mechanisms) >= 2 or len(backbones) >= 2
            accuracy = any(
                decision["cells"][name]["relative_sql_difference"] <= 0.02
                for name in passing
            )
            rows.append(
                {
                    "d1_lower_threshold": d1_threshold,
                    "registered_g_lower_threshold": g_threshold,
                    "passing_cell_count": len(passing),
                    "diversity_passed": diversity,
                    "accuracy_complement_passed": accuracy,
                    "paper_gate_passed": len(passing) >= 2 and diversity and accuracy,
                    "passing_cells": ";".join(passing),
                }
            )
    return rows


def _select_representative(
    arrays: dict[str, np.ndarray],
    cell_name: str,
) -> dict[str, Any]:
    d1 = np.asarray(arrays["shape_distance_by_width"][:, 0], dtype=np.float64)
    gap = np.asarray(arrays["hidden_distortion_gap"], dtype=np.float64)
    d1_center, gap_center = float(np.median(d1)), float(np.median(gap))
    d1_scale = max(float(np.median(np.abs(d1 - d1_center))), 1e-12)
    gap_scale = max(float(np.median(np.abs(gap - gap_center))), 1e-12)
    scores = np.abs(d1 - d1_center) / d1_scale + np.abs(gap - gap_center) / gap_scale
    identifiers = arrays["series_ids"].astype(str)
    ordering = np.lexsort((identifiers, scores))
    index = int(ordering[0])
    oracle = np.asarray(arrays["oracle_response"][index], dtype=np.float64)
    predicted = np.asarray(arrays["predicted_response"][index], dtype=np.float64)
    return {
        "cell": cell_name,
        "selection_rule": (
            "minimum MAD-standardized distance to cell medians of D1 and "
            "registered G; series_id tie break"
        ),
        "series_id": str(identifiers[index]),
        "d1": float(d1[index]),
        "registered_resolution_contrast": float(gap[index]),
        "dsa": float(arrays["dsa"][index]),
        "rgr": float(arrays["rgr"][index]),
        "oracle_response": oracle.tolist(),
        "predicted_response": predicted.tolist(),
        "oracle_width8": block_signed_response(oracle, 8).tolist(),
        "predicted_width8": block_signed_response(predicted, 8).tolist(),
    }


def analyze_p2_robustness(
    repo_root: str | Path,
    primary_input_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Read verified P1 arrays and compute post-primary descriptive robustness outputs."""

    root = Path(repo_root).resolve()
    input_root = Path(primary_input_root).resolve()
    destination = Path(output_root).resolve()
    hashes = verify_p2_freeze(root)
    config = load_yaml(root / "configs" / "p2_robustness" / "p2_robustness_v1.yaml")
    primary = config["source_primary"]
    count = int(primary["series_per_mechanism_per_seed"])
    seeds = tuple(int(value) for value in primary["generator_seeds"])
    widths = tuple(int(value) for value in config["analysis"]["resolution_widths"])
    decision = json.loads(
        (root / primary["decision_artifact"]).read_text(encoding="utf-8")
    )

    cell_rows: list[dict[str, Any]] = []
    wql_rows: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    passing_cells = set(decision["decision"]["passing_cells"])
    source_hashes: dict[str, str] = {}

    for backbone in primary["backbones"]:
        for mechanism in primary["mechanisms"]:
            units: list[dict[str, np.ndarray]] = []
            strata_parts: list[np.ndarray] = []
            for seed in seeds:
                unit = _load_verified_unit(
                    input_root,
                    backbone,
                    mechanism,
                    seed,
                    expected_config_hash=primary["config_hash"],
                    expected_code_hash=primary["scientific_code_sha256"],
                    expected_count=count,
                )
                units.append(unit)
                strata_parts.append(np.full(count, seed, dtype=np.int64))
                array_path, _ = _unit_paths(input_root, backbone, mechanism, seed)
                source_hashes[
                    f"{backbone}/{mechanism}/seed_{seed:05d}"
                ] = hashlib.sha256(array_path.read_bytes()).hexdigest()

            strata = np.concatenate(strata_parts)
            oracle = np.concatenate([unit["oracle_response"] for unit in units])
            predicted = np.concatenate([unit["predicted_response"] for unit in units])
            curves = [
                fixed_fine_distance_curve(pred, truth, widths)
                for pred, truth in zip(predicted, oracle, strict=True)
            ]
            curve_values = np.asarray(
                [[curve[width] for width in widths] for curve in curves], dtype=np.float64
            )
            monotone_gap = curve_values[:, 0] - curve_values[:, -1]
            curve_intervals = {
                width: _interval(curve_values[:, index], strata, np.median, config)
                for index, width in enumerate(widths)
            }
            gap_interval = _interval(monotone_gap, strata, np.median, config)
            cell_name = f"{backbone}/{mechanism}"
            cell_row: dict[str, Any] = {
                "backbone": backbone,
                "mechanism": mechanism,
                "series_count": int(strata.size),
            }
            for width in widths:
                interval = curve_intervals[width]
                cell_row[f"monotone_c{width}_estimate"] = interval["estimate"]
                cell_row[f"monotone_c{width}_lower"] = interval["lower"]
                cell_row[f"monotone_c{width}_upper"] = interval["upper"]
            cell_row.update(
                {
                    "monotone_gap_estimate": gap_interval["estimate"],
                    "monotone_gap_lower": gap_interval["lower"],
                    "monotone_gap_upper": gap_interval["upper"],
                    "all_series_curves_nonincreasing": bool(
                        np.all(np.diff(curve_values, axis=1) <= 1e-12)
                    ),
                }
            )
            cell_rows.append(cell_row)

            wql_target = np.concatenate([unit["wql_target"] for unit in units])
            wql_factual = np.concatenate([unit["wql_factual"] for unit in units])
            wql_rows.append(
                {
                    "backbone": backbone,
                    "mechanism": mechanism,
                    "series_count": int(strata.size),
                    "mean_wql_target_only": float(np.mean(wql_target)),
                    "mean_wql_covariate_aware": float(np.mean(wql_factual)),
                    "relative_wql_difference": float(
                        (np.mean(wql_factual) - np.mean(wql_target)) / np.mean(wql_target)
                    ),
                }
            )

            if cell_name in passing_cells:
                combined = {
                    key: np.concatenate([unit[key] for unit in units]) for key in units[0]
                }
                representatives.append(_select_representative(combined, cell_name))

    fixture_rows = evaluate_metric_fixture_comparison()
    thresholds = config["analysis"]["threshold_sensitivity"]
    sensitivity_rows = _threshold_sensitivity(
        decision,
        [float(value) for value in thresholds["d1_lower_thresholds"]],
        [float(value) for value in thresholds["registered_g_lower_thresholds"]],
    )

    outputs = {
        "monotone_cell_summary.csv": cell_rows,
        "wql_cell_summary.csv": wql_rows,
        "metric_fixture_comparison.csv": fixture_rows,
        "threshold_sensitivity.csv": sensitivity_rows,
    }
    for name, rows in outputs.items():
        _write_csv(destination / name, rows)
    _write_json(destination / "representative_responses.json", representatives)

    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "result_status": "verified_post_primary_analysis_only",
        "config_hash": hashes["config_hash"],
        "scientific_code_sha256": hashes["scientific_code_sha256"],
        "scope": {
            "model_inference_performed": False,
            "primary_arrays_modified": False,
            "registered_metrics_or_thresholds_modified": False,
            "primary_decision_recomputed": False,
        },
        "counts": {
            "source_unit_count": len(source_hashes),
            "cell_count": len(cell_rows),
            "representative_response_count": len(representatives),
            "fixture_count": len(fixture_rows),
            "threshold_grid_count": len(sensitivity_rows),
        },
        "source_primary": {
            "config_hash": primary["config_hash"],
            "scientific_code_sha256": primary["scientific_code_sha256"],
            "decision_sha256": hashlib.sha256(
                (root / primary["decision_artifact"]).read_bytes()
            ).hexdigest(),
            "unit_sha256": source_hashes,
        },
        "artifacts": {},
    }
    for path in sorted(destination.glob("*.csv")):
        report["artifacts"][path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": sum(1 for _ in path.open(encoding="utf-8")) - 1,
        }
    response_path = destination / "representative_responses.json"
    report["artifacts"][response_path.name] = {
        "sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
        "item_count": len(representatives),
    }
    _write_json(destination / "p2_robustness_report.json", report)
    return report
