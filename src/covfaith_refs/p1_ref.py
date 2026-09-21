"""Frozen execution and descriptive analysis for the P1-REF supplement."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from covfaith.config import canonical_config_hash, load_yaml
from covfaith.forecast_metrics import scaled_quantile_loss, weighted_quantile_loss
from covfaith.metrics import directional_sign_agreement, response_gain_ratio
from covfaith.p1_shape import scientific_code_hash as primary_scientific_code_hash
from covfaith.shape_generators import (
    ShapeGeneratorSpec,
    generate_shape_batch,
)
from covfaith.shape_metrics import interaction_response, shape_metric_suite, signed_shape_distance
from covfaith.statistics import BootstrapInterval, stratified_series_bootstrap
from covfaith_refs.transparent import (
    MODEL_IDS,
    fit_reference,
    forecast_reference,
    residual_index_paths,
)

FloatArray = NDArray[np.float64]

CONFIG_RELATIVE = Path("configs/p1_shape_references/reference_baselines_v1.yaml")
LOCK_RELATIVE = Path("configs/p1_shape_references/reference_baselines_v1.lock.json")
PRIMARY_CONFIG_RELATIVE = Path("configs/p1_shape/covintervene_shape_p1.yaml")
WORLD_SLOTS = ("factual", "primary", "a_only", "b_only")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for field in row:
            if field not in known:
                fieldnames.append(field)
                known.add(field)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def reference_scientific_code_hash(root: str | Path) -> str:
    """Hash the isolated supplement package together with its frozen config."""

    root_path = Path(root).resolve()
    paths = sorted((root_path / "src" / "covfaith_refs").glob("*.py"))
    paths.append(root_path / CONFIG_RELATIVE)
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.relative_to(root_path).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_reference_freeze(root: str | Path) -> dict[str, str]:
    """Verify authorization and the isolation boundary before any fitted outcome."""

    root_path = Path(root).resolve()
    config_path = root_path / CONFIG_RELATIVE
    lock_path = root_path / LOCK_RELATIVE
    config = load_yaml(config_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    config_hash = canonical_config_hash(config)
    if config_hash != lock.get("canonical_sha256"):
        raise RuntimeError("P1-REF config hash does not match its external lock")
    if config.get("status") != "frozen_owner_approved_post_primary":
        raise RuntimeError("P1-REF config is not frozen")
    authorization = config.get("authorization", {})
    if authorization.get("outcomes_authorized") is not True:
        raise RuntimeError("P1-REF outcome computation is not authorized")
    if authorization.get("approval_phrase") != lock.get("approval_phrase"):
        raise RuntimeError("P1-REF approval phrase mismatch")
    if lock.get("outcomes_authorized") is not True:
        raise RuntimeError("P1-REF lock does not authorize outcomes")
    if lock.get("primary_decision_mutation_allowed") is not False:
        raise RuntimeError("P1-REF lock must prohibit primary-decision mutation")
    supplement_hash = reference_scientific_code_hash(root_path)
    if supplement_hash != lock.get("scientific_code_sha256"):
        raise RuntimeError("P1-REF scientific code hash does not match its external lock")
    primary_hash = primary_scientific_code_hash(root_path)
    expected_primary = config["isolation"]["primary_scientific_code_sha256"]
    if primary_hash != expected_primary:
        raise RuntimeError(
            f"frozen P1 scientific hash changed: expected {expected_primary}, got {primary_hash}"
        )
    primary_config_hash = canonical_config_hash(root_path / PRIMARY_CONFIG_RELATIVE)
    expected_primary_config = config["isolation"]["primary_config_sha256"]
    if primary_config_hash != expected_primary_config:
        raise RuntimeError("frozen P1 config hash changed")
    return {
        "config_hash": config_hash,
        "scientific_code_sha256": supplement_hash,
        "primary_scientific_code_sha256": primary_hash,
        "primary_config_sha256": primary_config_hash,
    }


def _generator_spec(root: Path) -> ShapeGeneratorSpec:
    data = load_yaml(root / PRIMARY_CONFIG_RELATIVE)["data"]
    intervention = data["intervention"]
    return ShapeGeneratorSpec(
        context_length=int(data["context_length"]),
        horizon=int(data["horizon"]),
        target_ar_range=tuple(float(value) for value in data["target_ar_range"]),
        covariate_ar_range=tuple(float(value) for value in data["covariate_ar_range"]),
        coefficient_magnitude_range=tuple(
            float(value) for value in data["coefficient_magnitude_range"]
        ),
        innovation_sd_range=tuple(float(value) for value in data["innovation_sd_range"]),
        intervention_scale=float(intervention["scale_in_covariate_sd"]),
        pulse_lengths=tuple(int(value) for value in intervention["pulse_lengths"]),
        start_indices=tuple(int(value) for value in intervention["start_indices"]),
        clip_quantiles=tuple(float(value) for value in intervention["clip_to_context_quantiles"]),
    )


def _stable_seed(base: int, *parts: str | int) -> int:
    material = "|".join([str(base), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32)


def _unit_paths(output_root: Path, mechanism: str, seed: int) -> tuple[Path, Path]:
    directory = output_root / "units" / mechanism
    stem = f"seed_{seed:05d}"
    return directory / f"{stem}.npz", directory / f"{stem}.manifest.json"


def _resume_unit(
    array_path: Path,
    manifest_path: Path,
    *,
    mechanism: str,
    seed: int,
    hashes: dict[str, str],
) -> dict[str, Any] | None:
    if not array_path.exists() and not manifest_path.exists():
        return None
    if not array_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"incomplete existing P1-REF unit for {mechanism}/{seed}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "mechanism": mechanism,
        "generator_seed": seed,
        **hashes,
        "array_sha256": _sha256(array_path),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"existing P1-REF unit hash mismatch for {mechanism}/{seed}")
    return manifest


def _world_covariates(scenario: Any) -> dict[str, FloatArray]:
    worlds: dict[str, FloatArray] = {
        "factual": scenario.covariate_future_factual,
        "primary": scenario.covariate_future_intervened,
    }
    if scenario.mechanism == "two_covariate_synergy":
        worlds["a_only"] = scenario.covariate_future_worlds["a_only"]
        worlds["b_only"] = scenario.covariate_future_worlds["b_only"]
    return worlds


def run_reference_unit(
    root: str | Path,
    output_root: str | Path,
    mechanism: str,
    seed: int,
) -> dict[str, Any]:
    """Fit both frozen references for one mechanism/seed unit and save forecasts."""

    root_path = Path(root).resolve()
    output_path = Path(output_root).resolve()
    config = load_yaml(root_path / CONFIG_RELATIVE)
    hashes = verify_reference_freeze(root_path)
    data = config["data"]
    if mechanism not in data["mechanisms"] or int(seed) not in data["generator_seeds"]:
        raise ValueError("unit is outside the frozen P1-REF matrix")
    array_path, manifest_path = _unit_paths(output_path, mechanism, int(seed))
    resumed = _resume_unit(
        array_path,
        manifest_path,
        mechanism=mechanism,
        seed=int(seed),
        hashes=hashes,
    )
    if resumed is not None:
        return {**resumed, "execution_status": "resumed"}

    scenarios = generate_shape_batch(
        mechanism,
        int(seed),
        int(data["series_per_mechanism_per_seed"]),
        _generator_spec(root_path),
    )
    model_count = len(MODEL_IDS)
    series_count = len(scenarios)
    horizon = int(data["horizon"])
    levels = np.asarray(config["forecast"]["quantile_levels"], dtype=np.float64)
    points = np.full(
        (model_count, series_count, len(WORLD_SLOTS), horizon),
        np.nan,
        dtype=np.float64,
    )
    quantiles = np.full(
        (model_count, series_count, len(WORLD_SLOTS), horizon, levels.size),
        np.nan,
        dtype=np.float64,
    )
    fit_condition = np.empty((model_count, series_count), dtype=np.float64)
    residual_scale = np.empty((model_count, series_count), dtype=np.float64)
    fit_seconds = np.empty((model_count, series_count), dtype=np.float64)
    target_context = np.stack([scenario.target_context for scenario in scenarios])
    target_future = np.stack([scenario.target_future_factual for scenario in scenarios])
    oracle_response = np.stack([scenario.oracle_response for scenario in scenarios])
    oracle_interaction = np.full((series_count, horizon), np.nan, dtype=np.float64)
    if mechanism == "two_covariate_synergy":
        oracle_interaction = np.stack(
            [
                interaction_response(
                    scenario.oracle_response_worlds["joint"],
                    scenario.oracle_response_worlds["a_only"],
                    scenario.oracle_response_worlds["b_only"],
                )
                for scenario in scenarios
            ]
        )

    started = time.perf_counter()
    for series_index, scenario in enumerate(scenarios):
        worlds = _world_covariates(scenario)
        for model_index, model_id in enumerate(MODEL_IDS):
            fit_started = time.perf_counter()
            alpha = float(config["models"][model_id]["alpha"])
            fitted = fit_reference(
                scenario.target_context,
                scenario.covariate_context,
                model_id,
                alpha=alpha,
            )
            fit_seconds[model_index, series_index] = time.perf_counter() - fit_started
            fit_condition[model_index, series_index] = fitted.condition_number
            residual_scale[model_index, series_index] = float(
                np.std(fitted.residuals, ddof=1)
            )
            bootstrap_seed = _stable_seed(
                int(config["forecast"]["bootstrap_seed"]),
                mechanism,
                seed,
                series_index,
                model_id,
            )
            residual_indices = residual_index_paths(
                fitted.residuals.size,
                int(config["forecast"]["bootstrap_paths"]),
                horizon,
                bootstrap_seed,
            )
            for world_index, world in enumerate(WORLD_SLOTS):
                if world not in worlds:
                    continue
                forecast = forecast_reference(
                    fitted,
                    scenario.target_context,
                    scenario.covariate_context,
                    worlds[world],
                    residual_indices=residual_indices,
                    quantile_levels=levels,
                )
                points[model_index, series_index, world_index] = forecast.point
                quantiles[model_index, series_index, world_index] = forecast.quantiles
    elapsed = time.perf_counter() - started

    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_suffix(".tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            model_ids=np.asarray(MODEL_IDS),
            world_slots=np.asarray(WORLD_SLOTS),
            quantile_levels=levels,
            series_ids=np.asarray([scenario.series_id for scenario in scenarios]),
            target_context=target_context,
            target_future_factual=target_future,
            oracle_response=oracle_response,
            oracle_interaction=oracle_interaction,
            points=points,
            quantiles=quantiles,
            fit_condition_number=fit_condition,
            residual_scale=residual_scale,
            fit_seconds=fit_seconds,
        )
    temporary.replace(array_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "result_status": "post_primary_supplementary_unit_complete_not_analyzed",
        "evidence_role": config["evidence_role"],
        **hashes,
        "git_commit": _git_commit(root_path),
        "created_at_utc": _utc_now(),
        "mechanism": mechanism,
        "generator_seed": int(seed),
        "series_count": series_count,
        "model_ids": list(MODEL_IDS),
        "world_slots": list(WORLD_SLOTS),
        "bootstrap_paths": int(config["forecast"]["bootstrap_paths"]),
        "array_relative_path": array_path.relative_to(output_path).as_posix(),
        "array_sha256": _sha256(array_path),
        "wall_time_seconds": elapsed,
        "scope": {
            "tsfm_inference_performed": False,
            "primary_decision_recomputed": False,
            "future_targets_used_for_fitting": False,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return {**manifest, "execution_status": "completed"}


def run_all_reference_units(
    root: str | Path,
    output_root: str | Path,
) -> list[dict[str, Any]]:
    """Run or resume the complete frozen 4-by-3 reference matrix."""

    root_path = Path(root).resolve()
    config = load_yaml(root_path / CONFIG_RELATIVE)
    reports: list[dict[str, Any]] = []
    for mechanism in config["data"]["mechanisms"]:
        for seed in config["data"]["generator_seeds"]:
            reports.append(run_reference_unit(root_path, output_root, mechanism, int(seed)))
    return reports


def _interval_dict(interval: BootstrapInterval) -> dict[str, float | int]:
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "replicates": interval.replicates,
        "seed": interval.seed,
    }


def _optional(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _series_records(
    mechanism: str,
    seed: int,
    arrays: Any,
) -> list[dict[str, Any]]:
    levels = arrays["quantile_levels"].astype(np.float64)
    median_index = int(np.flatnonzero(np.isclose(levels, 0.5))[0])
    records: list[dict[str, Any]] = []
    for model_index, model_id in enumerate(arrays["model_ids"].tolist()):
        for series_index, series_id in enumerate(arrays["series_ids"].tolist()):
            factual = arrays["quantiles"][model_index, series_index, 0]
            primary = arrays["quantiles"][model_index, series_index, 1]
            predicted_response = primary[:, median_index] - factual[:, median_index]
            oracle = arrays["oracle_response"][series_index]
            shape = shape_metric_suite(predicted_response, oracle)
            truth = arrays["target_future_factual"][series_index]
            context = arrays["target_context"][series_index]
            median = factual[:, median_index]
            interaction_distance: float | None = None
            if mechanism == "two_covariate_synergy":
                marginal_a = arrays["quantiles"][model_index, series_index, 2]
                marginal_b = arrays["quantiles"][model_index, series_index, 3]
                predicted_interaction = interaction_response(
                    predicted_response,
                    marginal_a[:, median_index] - factual[:, median_index],
                    marginal_b[:, median_index] - factual[:, median_index],
                )
                interaction_distance = signed_shape_distance(
                    predicted_interaction,
                    arrays["oracle_interaction"][series_index],
                )
            denominator = float(np.sum(np.abs(truth)))
            wape = float(np.sum(np.abs(truth - median)) / max(denominator, 1e-8))
            records.append(
                {
                    "model_id": model_id,
                    "mechanism": mechanism,
                    "generator_seed": int(seed),
                    "series_id": series_id,
                    "dsa": directional_sign_agreement(predicted_response, oracle),
                    "rgr": response_gain_ratio(predicted_response, oracle),
                    "shape_distance_width_1": shape.signed_shape_distances[0],
                    "shape_distance_width_2": shape.signed_shape_distances[1],
                    "shape_distance_width_4": shape.signed_shape_distances[2],
                    "shape_distance_width_8": shape.signed_shape_distances[3],
                    "hidden_distortion_gap": shape.hidden_distortion_gap,
                    "multi_resolution_auc": shape.multi_resolution_auc,
                    "temporal_mass_distance": shape.temporal_mass_distance,
                    "onset_error": shape.onset_error,
                    "peak_time_error": shape.peak_time_error,
                    "positive_mass_relative_error": _optional(
                        shape.positive_mass_relative_error
                    ),
                    "negative_mass_relative_error": _optional(
                        shape.negative_mass_relative_error
                    ),
                    "interaction_shape_distance": interaction_distance,
                    "sql": scaled_quantile_loss(truth, factual, levels, context),
                    "wql": weighted_quantile_loss(truth, factual, levels),
                    "mae": float(np.mean(np.abs(truth - median))),
                    "wape": wape,
                    "fit_condition_number": float(
                        arrays["fit_condition_number"][model_index, series_index]
                    ),
                    "residual_scale": float(
                        arrays["residual_scale"][model_index, series_index]
                    ),
                    "fit_seconds": float(arrays["fit_seconds"][model_index, series_index]),
                }
            )
    return records


def _cell_summary(
    records: list[dict[str, Any]],
    model_id: str,
    mechanism: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [
        record
        for record in records
        if record["model_id"] == model_id and record["mechanism"] == mechanism
    ]
    strata = [int(record["generator_seed"]) for record in selected]
    aggregation: dict[str, Any] = {
        "dsa": np.mean,
        "rgr": np.median,
        "shape_distance_width_1": np.median,
        "shape_distance_width_2": np.median,
        "shape_distance_width_4": np.median,
        "shape_distance_width_8": np.median,
        "hidden_distortion_gap": np.median,
        "multi_resolution_auc": np.median,
        "temporal_mass_distance": np.median,
        "onset_error": np.median,
        "peak_time_error": np.median,
        "sql": np.mean,
        "wql": np.mean,
        "mae": np.mean,
        "wape": np.mean,
    }
    for optional_metric in (
        "positive_mass_relative_error",
        "negative_mass_relative_error",
        "interaction_shape_distance",
    ):
        if all(record[optional_metric] is not None for record in selected):
            aggregation[optional_metric] = np.median
    intervals: dict[str, dict[str, float | int]] = {}
    matrix: dict[str, Any] = {
        "model_id": model_id,
        "mechanism": mechanism,
        "series_count": len(selected),
    }
    replicates = int(config["uncertainty"]["bootstrap_replicates"])
    base_seed = int(config["uncertainty"]["bootstrap_seed"])
    for metric, statistic in aggregation.items():
        values = [float(record[metric]) for record in selected]
        interval = stratified_series_bootstrap(
            values,
            strata,
            statistic=statistic,
            replicates=replicates,
            seed=_stable_seed(base_seed, model_id, mechanism, metric),
        )
        intervals[metric] = _interval_dict(interval)
        matrix[f"{metric}_estimate"] = interval.estimate
        matrix[f"{metric}_lower"] = interval.lower
        matrix[f"{metric}_upper"] = interval.upper
    return {
        "model_id": model_id,
        "mechanism": mechanism,
        "series_count": len(selected),
        "intervals": intervals,
    }, matrix


def analyze_reference_units(
    root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Verify all frozen units and create descriptive, non-gating summaries once."""

    root_path = Path(root).resolve()
    output_path = Path(output_root).resolve()
    config = load_yaml(root_path / CONFIG_RELATIVE)
    hashes = verify_reference_freeze(root_path)
    report_path = output_path / config["artifacts"]["analysis_report"]
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in hashes.items()):
            raise RuntimeError("existing P1-REF analysis was produced by different frozen code")
        for artifact in existing.get("artifacts", {}).values():
            artifact_path = output_path / artifact["relative_path"]
            if not artifact_path.is_file() or _sha256(artifact_path) != artifact["sha256"]:
                raise RuntimeError("existing P1-REF analysis artifact failed verification")
        return existing

    records: list[dict[str, Any]] = []
    unit_inventory: list[dict[str, Any]] = []
    for mechanism in config["data"]["mechanisms"]:
        for seed in config["data"]["generator_seeds"]:
            array_path, manifest_path = _unit_paths(output_path, mechanism, int(seed))
            manifest = _resume_unit(
                array_path,
                manifest_path,
                mechanism=mechanism,
                seed=int(seed),
                hashes=hashes,
            )
            if manifest is None:
                raise RuntimeError(f"missing P1-REF unit for {mechanism}/{seed}")
            with np.load(array_path, allow_pickle=False) as arrays:
                records.extend(_series_records(mechanism, int(seed), arrays))
            unit_inventory.append(
                {
                    "mechanism": mechanism,
                    "generator_seed": int(seed),
                    "array_sha256": manifest["array_sha256"],
                    "manifest_sha256": _sha256(manifest_path),
                }
            )

    summaries: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for mechanism in config["data"]["mechanisms"]:
            summary, matrix = _cell_summary(records, model_id, mechanism, config)
            summaries.append(summary)
            matrix_rows.append(matrix)

    series_path = output_path / config["artifacts"]["series_metrics"]
    matrix_path = output_path / config["artifacts"]["complete_matrix"]
    _write_csv_atomic(series_path, records)
    _write_csv_atomic(matrix_path, matrix_rows)
    report: dict[str, Any] = {
        "schema_version": 1,
        "result_status": "verified_post_primary_supplementary_descriptive",
        "evidence_role": config["evidence_role"],
        **hashes,
        "git_commit": _git_commit(root_path),
        "created_at_utc": _utc_now(),
        "counts": {
            "unit_count": len(unit_inventory),
            "series_model_record_count": len(records),
            "summary_cell_count": len(summaries),
        },
        "summaries": summaries,
        "artifacts": {
            "series_metrics": {
                "relative_path": series_path.relative_to(output_path).as_posix(),
                "sha256": _sha256(series_path),
            },
            "complete_matrix": {
                "relative_path": matrix_path.relative_to(output_path).as_posix(),
                "sha256": _sha256(matrix_path),
            },
        },
        "unit_inventory": unit_inventory,
        "scope": {
            "continuation_gate_computed": False,
            "primary_decision_recomputed": False,
            "primary_thresholds_revised": False,
            "tsfm_inference_performed": False,
            "interpretation": "descriptive_context_only_not_preregistered_primary_evidence",
        },
    }
    _write_json_atomic(report_path, report)
    return report
