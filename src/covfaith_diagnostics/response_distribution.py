"""Describe complete P1 response distributions without changing the frozen decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from covfaith.config import load_yaml, verify_config_lock
from covfaith.metrics import response_gain_ratio
from covfaith.p1_shape import scientific_code_hash
from covfaith.shape_metrics import onset_error, peak_time_error, temporal_mass_distance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _distribution(values: NDArray[np.float64]) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("descriptive distribution requires a finite nonempty vector")
    quantiles = np.quantile(values, [0.25, 0.5, 0.75, 0.9])
    return dict(zip(("q25", "median", "q75", "q90"), map(float, quantiles), strict=True))


def _verified_unit(
    source_root: Path,
    key: str,
    *,
    config_hash: str,
    code_hash: str,
    expected_hash: str,
    count: int,
    horizon: int,
    support_fraction: float,
) -> dict[str, np.ndarray]:
    backbone, mechanism, seed_name = key.split("/")
    directory = source_root / "units" / backbone / mechanism
    array_path = directory / f"{seed_name}.npz"
    manifest_path = directory / f"{seed_name}.json"
    if not array_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"missing private full-screening unit: {key}")
    manifest = _json(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"invalid unit manifest: {key}")
    checks = {
        "completed": manifest.get("completed") is True,
        "mode": manifest.get("mode") == "full_screening",
        "identity": (manifest.get("backbone"), manifest.get("mechanism")) == (backbone, mechanism),
        "seed": manifest.get("generator_seed") == int(seed_name.removeprefix("seed_")),
        "count": manifest.get("series_count") == count,
        "config": manifest.get("config_hash") == config_hash,
        "code": manifest.get("scientific_code_sha256") == code_hash,
        "sha256": manifest.get("array_sha256") == expected_hash == _sha256(array_path),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"unit verification failed for {key}: {[k for k, v in checks.items() if not v]}"
        )
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required_shapes = {
        "series_ids": (count,),
        "oracle_response": (count, horizon),
        "predicted_response": (count, horizon),
        "dsa": (count,),
        "rgr": (count,),
        "hidden_distortion_gap": (count,),
        "temporal_mass_distance": (count,),
        "onset_error": (count,),
        "peak_time_error": (count,),
    }
    for name, shape in required_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise RuntimeError(f"{key}: {name} has the wrong shape or is absent")
    if (
        "shape_distance_by_width" not in arrays
        or arrays["shape_distance_by_width"].ndim != 2
        or arrays["shape_distance_by_width"].shape[0] != count
        or arrays["shape_distance_by_width"].shape[1] < 1
    ):
        raise RuntimeError(f"{key}: shape_distance_by_width is missing or incomplete")
    for name in required_shapes.keys() - {"series_ids"}:
        if not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"{key}: nonfinite {name}")
    if not np.all(np.isfinite(arrays["shape_distance_by_width"])):
        raise RuntimeError(f"{key}: nonfinite shape_distance_by_width")
    identifiers = arrays["series_ids"].astype(str)
    if np.unique(identifiers).size != count:
        raise RuntimeError(f"{key}: duplicate series IDs")

    oracle = np.asarray(arrays["oracle_response"], dtype=np.float64)
    predicted = np.asarray(arrays["predicted_response"], dtype=np.float64)
    checks_from_paths = {
        "rgr": [
            response_gain_ratio(p, o, support_fraction)
            for p, o in zip(predicted, oracle, strict=True)
        ],
        "onset_error": [
            onset_error(p, o, support_fraction) for p, o in zip(predicted, oracle, strict=True)
        ],
        "peak_time_error": [peak_time_error(p, o) for p, o in zip(predicted, oracle, strict=True)],
        "temporal_mass_distance": [
            temporal_mass_distance(p, o) for p, o in zip(predicted, oracle, strict=True)
        ],
    }
    for name, recomputed in checks_from_paths.items():
        if not np.allclose(arrays[name], recomputed, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"{key}: archived {name} differs from raw-response replay")
    return arrays


def _cell_summary(
    arrays: dict[str, np.ndarray], horizon: int, support_fraction: float
) -> dict[str, Any]:
    oracle = np.asarray(arrays["oracle_response"], dtype=np.float64)
    predicted = np.asarray(arrays["predicted_response"], dtype=np.float64)
    denominator = np.sum(np.abs(oracle), axis=1)
    if np.any(denominator <= 0):
        raise RuntimeError("zero-mass oracle response")
    full_l1_ratio = np.sum(np.abs(predicted), axis=1) / denominator
    onset_steps = np.asarray(arrays["onset_error"], dtype=np.float64) * (horizon - 1)
    peak_steps = np.asarray(arrays["peak_time_error"], dtype=np.float64) * (horizon - 1)
    for name, values in (("onset", onset_steps), ("peak", peak_steps)):
        if np.any(np.abs(values - np.rint(values)) > 1e-8):
            raise RuntimeError(f"{name} errors are not integral forecast steps")
    threshold = support_fraction * np.max(np.abs(oracle), axis=1)
    no_predicted_onset = ~np.any(np.abs(predicted) >= threshold[:, None], axis=1)
    no_predicted_peak = np.sum(np.abs(predicted), axis=1) <= 1e-8 * np.maximum(
        1.0, np.sum(np.abs(oracle), axis=1)
    )
    values = {
        "rgr": np.asarray(arrays["rgr"], dtype=np.float64),
        "full_l1_ratio": full_l1_ratio,
        "onset_error_steps": onset_steps,
        "peak_time_error_steps": peak_steps,
        "temporal_mass_distance": np.asarray(arrays["temporal_mass_distance"], dtype=np.float64),
    }
    summary: dict[str, Any] = {
        "n": int(oracle.shape[0]),
        "no_predicted_onset_count": int(np.count_nonzero(no_predicted_onset)),
        "no_predicted_peak_count": int(np.count_nonzero(no_predicted_peak)),
        "onset_within_one_step_count": int(np.count_nonzero(onset_steps <= 1 + 1e-8)),
        "peak_within_one_step_count": int(np.count_nonzero(peak_steps <= 1 + 1e-8)),
    }
    summary.update({name: _distribution(value) for name, value in values.items()})
    return summary | {"_series_values": values}


def _reconcile_frozen_cell(
    cell: str, arrays: dict[str, np.ndarray], decision: dict[str, Any]
) -> None:
    expected = decision["cells"][cell]
    checks = {
        "dsa": (np.mean(arrays["dsa"]), expected["dsa"]["estimate"]),
        "rgr": (np.median(arrays["rgr"]), expected["rgr"]["estimate"]),
        "shape_d1": (
            np.median(arrays["shape_distance_by_width"][:, 0]),
            expected["shape_d1"]["estimate"],
        ),
        "hidden_gap": (
            np.median(arrays["hidden_distortion_gap"]),
            expected["hidden_gap"]["estimate"],
        ),
    }
    if arrays["series_ids"].size != expected["series_count"]:
        raise RuntimeError(f"{cell}: frozen series count mismatch")
    for name, (actual, frozen) in checks.items():
        if not np.isclose(actual, frozen, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"{cell}: {name} does not reconcile with the frozen decision")


def _flatten_cell(cell: str, summary: dict[str, Any]) -> dict[str, Any]:
    backbone, mechanism = cell.split("/")
    row: dict[str, Any] = {"backbone": backbone, "mechanism": mechanism, "n": summary["n"]}
    for name in (
        "rgr",
        "full_l1_ratio",
        "onset_error_steps",
        "peak_time_error_steps",
        "temporal_mass_distance",
    ):
        row.update({f"{name}_{q}": value for q, value in summary[name].items()})
    for name in (
        "no_predicted_onset_count",
        "no_predicted_peak_count",
        "onset_within_one_step_count",
        "peak_within_one_step_count",
    ):
        row[name] = summary[name]
    return row


def analyze_response_distribution(
    repo_root: str | Path, source_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Read all frozen P1 units and write only source-ID-free descriptive outputs."""

    repo = Path(repo_root).resolve()
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("diagnostic output must be separate from the private P1 source archive")
    config_path = repo / "configs/p1_shape/covintervene_shape_p1.yaml"
    config_hash = verify_config_lock(
        config_path, repo / "configs/p1_shape/covintervene_shape_p1.lock.json"
    )
    config = load_yaml(config_path)
    code_hash = scientific_code_hash(repo)
    decision_path = repo / "evidence/p1_shape/results/p1_shape_decision.json"
    decision = _json(decision_path)
    if (
        not isinstance(decision, dict)
        or decision.get("result_status") != "verified_p1_shape_decision"
    ):
        raise RuntimeError("frozen P1 decision is missing or unverified")
    if (decision.get("config_hash"), decision.get("scientific_code_sha256")) != (
        config_hash,
        code_hash,
    ):
        raise RuntimeError("current P1 configuration/code differs from the frozen decision")

    p2_path = repo / "evidence/p2_robustness/results/p2_robustness_report.json"
    p2 = _json(p2_path)
    representatives_path = p2_path.parent / "representative_responses.json"
    representatives = _json(representatives_path)
    if not isinstance(p2, dict) or not isinstance(representatives, list):
        raise RuntimeError("invalid P2 representative provenance")
    source_primary = p2["source_primary"]
    if (
        source_primary["decision_sha256"] != _sha256(decision_path)
        or source_primary["config_hash"] != config_hash
        or source_primary["scientific_code_sha256"] != code_hash
        or p2["artifacts"]["representative_responses.json"]["sha256"]
        != _sha256(representatives_path)
    ):
        raise RuntimeError("P2 representative provenance differs from the frozen P1 record")

    data = config["data"]
    count, horizon = int(data["series_per_mechanism_per_seed"]), int(data["horizon"])
    fraction = float(config["response_metrics"]["active_support_fraction_of_peak"])
    source_hashes: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}
    all_series: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for model in config["models"]:
        backbone = model["id"]
        for mechanism in data["mechanisms"]:
            cell = f"{backbone}/{mechanism}"
            pieces: list[dict[str, np.ndarray]] = []
            for seed in data["generator_seeds"]:
                key = f"{cell}/seed_{int(seed):05d}"
                if key not in source_primary["unit_sha256"]:
                    raise RuntimeError(f"P2 source inventory lacks {key}")
                expected_hash = source_primary["unit_sha256"][key]
                pieces.append(
                    _verified_unit(
                        source,
                        key,
                        config_hash=config_hash,
                        code_hash=code_hash,
                        expected_hash=expected_hash,
                        count=count,
                        horizon=horizon,
                        support_fraction=fraction,
                    )
                )
                source_hashes[key] = expected_hash
            names = (
                "series_ids",
                "oracle_response",
                "predicted_response",
                "dsa",
                "rgr",
                "shape_distance_by_width",
                "hidden_distortion_gap",
                "temporal_mass_distance",
                "onset_error",
                "peak_time_error",
            )
            combined = {name: np.concatenate([unit[name] for unit in pieces]) for name in names}
            if np.unique(combined["series_ids"].astype(str)).size != count * len(pieces):
                raise RuntimeError(f"{cell}: duplicate series IDs across generator seeds")
            _reconcile_frozen_cell(cell, combined, decision)
            summary = _cell_summary(combined, horizon, fraction)
            all_series[cell] = combined | {
                "_full_l1_ratio": summary["_series_values"]["full_l1_ratio"]
            }
            summaries[cell] = {
                key: value for key, value in summary.items() if key != "_series_values"
            }
            rows.append(_flatten_cell(cell, summaries[cell]))
    if len(source_hashes) != 24 or len(rows) != 8:
        raise RuntimeError("incomplete eight-cell, 24-unit P1 diagnostic")
    if set(source_hashes) != set(source_primary["unit_sha256"]):
        raise RuntimeError("P2 source inventory differs from the complete P1 unit set")
    if set(summaries) != set(decision["cells"]):
        raise RuntimeError("descriptive cells differ from the frozen eight-cell decision")

    representative_checks: list[dict[str, Any]] = []
    passing_cells = set(decision["decision"]["passing_cells"])
    if (
        len(representatives) != len(passing_cells)
        or {item["cell"] for item in representatives} != passing_cells
    ):
        raise RuntimeError("representatives do not match all frozen passing cells")
    for item in representatives:
        cell = item["cell"]
        arrays = all_series[cell]
        indices = np.flatnonzero(arrays["series_ids"].astype(str) == str(item["series_id"]))
        if indices.size != 1:
            raise RuntimeError(f"{cell}: representative series is absent or duplicated")
        index = int(indices[0])
        for name, archive_name in (
            ("oracle_response", "oracle_response"),
            ("predicted_response", "predicted_response"),
        ):
            if not np.allclose(item[name], arrays[archive_name][index], rtol=0, atol=1e-12):
                raise RuntimeError(f"{cell}: representative {name} differs from P1 archive")
        values = {
            "rgr": np.asarray(arrays["rgr"], dtype=np.float64),
            "full_l1_ratio": np.asarray(arrays["_full_l1_ratio"], dtype=np.float64),
            "onset_error_steps": np.asarray(arrays["onset_error"], dtype=np.float64)
            * (horizon - 1),
            "peak_time_error_steps": np.asarray(arrays["peak_time_error"], dtype=np.float64)
            * (horizon - 1),
        }
        representative_checks.append(
            {
                "cell": cell,
                "n": int(values["rgr"].size),
                "inclusive_empirical_percentile": {
                    name: float(np.mean(vector <= vector[index]) * 100)
                    for name, vector in values.items()
                },
            }
        )

    inventory_hash = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    code_file = Path(__file__).resolve()
    report_path = output / "response_distribution_report.json"
    csv_path = output / "complete_eight_cell_response_distribution.csv"
    if report_path.exists():
        previous = _json(report_path)
        if (
            isinstance(previous, dict)
            and previous.get("source_inventory_sha256") == inventory_hash
            and previous.get("analysis_code_sha256") == _sha256(code_file)
            and previous.get("decision_sha256") == _sha256(decision_path)
            and previous.get("p2_report_sha256") == _sha256(p2_path)
            and previous.get("representatives_sha256") == _sha256(representatives_path)
            and csv_path.is_file()
            and previous.get("summary_csv_sha256") == _sha256(csv_path)
        ):
            return previous
        raise RuntimeError("existing diagnostic report differs; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    temporary_csv = csv_path.with_suffix(".tmp.csv")
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)
    report = {
        "schema_version": 1,
        "result_status": "verified_post_primary_descriptive_response_distribution",
        "scope": {
            "full_cell_count": 8,
            "full_unit_count": 24,
            "series_count": sum(row["n"] for row in rows),
            "model_inference_performed": False,
            "primary_decision_recomputed": False,
            "new_gate_or_interval_computed": False,
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_hash": config_hash,
        "scientific_code_sha256": code_hash,
        "analysis_code_sha256": _sha256(code_file),
        "decision_sha256": _sha256(decision_path),
        "p2_report_sha256": _sha256(p2_path),
        "representatives_sha256": _sha256(representatives_path),
        "source_inventory_sha256": inventory_hash,
        "source_unit_sha256": source_hashes,
        "summary_csv_sha256": _sha256(csv_path),
        "definitions": {
            "quantiles": "empirical linear-interpolation q25, median, q75, q90",
            "rgr": "registered ratio of absolute response on oracle-active support",
            "full_l1_ratio": (
                "sum(abs(predicted_response)) / sum(abs(oracle_response)) over all 24 steps"
            ),
            "onset": "10% of oracle absolute peak applied to both responses",
            "onset_no_detection": "registered normalized error 1.0, displayed as 23-step sentinel",
            "peak": "earliest maximum-absolute-response step",
            "peak_no_detection": (
                "near-zero predicted mass receives the registered 23-step sentinel"
            ),
            "temporal_mass_distance": "registered normalized absolute-mass transport distance",
            "within_one_step": "descriptive count, not an eligibility threshold",
        },
        "cells": summaries,
        "representative_checks": representative_checks,
    }
    temporary_report = report_path.with_suffix(".tmp.json")
    temporary_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_response_distribution(args.repo, args.source, args.output)
    print(
        json.dumps(
            {
                "result_status": report["result_status"],
                "scope": report["scope"],
                "source_inventory_sha256": report["source_inventory_sha256"],
                "summary_csv_sha256": report["summary_csv_sha256"],
                "output": str(args.output / "complete_eight_cell_response_distribution.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
