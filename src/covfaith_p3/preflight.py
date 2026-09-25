"""Approved-design P3 CPU construct preflight; never loads a forecasting model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from covfaith.config import canonical_config_hash
from covfaith.p1_shape import scientific_code_hash as p1_scientific_code_hash

from .constructs import generate_scenario, validate_scenario
from .data import read_source_parquet, select_windows, sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _code_hash(repo: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((repo / "src" / "covfaith_p3").glob("*.py"))
    for path in paths:
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _verified_design(repo: Path) -> tuple[dict[str, Any], str]:
    config_path = repo / "configs" / "p3_semisynthetic" / "p3_design_candidate.yaml"
    approval_path = repo / "configs" / "p3_semisynthetic" / "p3_design_approval.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    config_hash = canonical_config_hash(config)
    if approval["approved_config_canonical_sha256"] != config_hash:
        raise RuntimeError("P3 design approval does not match candidate config")
    if approval["model_inference_authorized"] or config["authorization"]["model_inference_allowed"]:
        raise RuntimeError("P3 CPU preflight must not authorize model inference")
    if approval["approval_phrase"] != "APPROVE P3 DESIGN":
        raise RuntimeError("P3 design lacks owner approval")
    parent = config["parent_evidence"]
    if p1_scientific_code_hash(repo) != parent["p1_scientific_code_sha256"]:
        raise RuntimeError("frozen P1 scientific code hash changed")
    p1_decision = repo / "evidence" / "p1_shape" / "results" / "p1_shape_decision.json"
    p2_report = repo / "evidence" / "p2_robustness" / "results" / "p2_robustness_report.json"
    if sha256_file(p1_decision) != parent["p1_decision_sha256"]:
        raise RuntimeError("frozen P1 decision hash changed")
    if sha256_file(p2_report) != parent["p2_report_sha256"]:
        raise RuntimeError("verified P2 report hash changed")
    return config, config_hash


def run_cpu_preflight(repo: Path, data_root: Path, private_root: Path) -> dict[str, Any]:
    """Check source/construct invariants, recording private selections before scenarios."""
    repo = repo.resolve()
    data_root = data_root.resolve()
    private_root = private_root.resolve()
    config, config_hash = _verified_design(repo)
    if private_root == repo or private_root == data_root:
        raise ValueError("private output must be a dedicated directory")
    private_root.mkdir(parents=True, exist_ok=True)
    data = config["data"]
    mechanism = config["mechanism"]
    seeds = [int(value) for value in data["scenario_seeds"]]
    families = list(mechanism["families"])
    selected_by_source: dict[str, list[Any]] = {}
    source_reports: dict[str, dict[str, Any]] = {}
    private_selection: dict[str, Any] = {"config_hash": config_hash, "sources": {}}

    for source in data["sources"]:
        local_path = data_root / f"{source['config']}.parquet"
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        actual_hash = sha256_file(local_path)
        if actual_hash != source["parquet_sha256"]:
            raise RuntimeError(f"{source['id']}: pinned Parquet SHA-256 mismatch")
        series = read_source_parquet(local_path)
        windows, eligible_count = select_windows(series, data, source)
        selected_by_source[source["id"]] = windows
        source_reports[source["id"]] = {
            "parquet_sha256": actual_hash,
            "original_series_count": len(series),
            "eligible_source_id_count": eligible_count,
            "selected_source_id_count": len(windows),
        }
        private_selection["sources"][source["id"]] = [
            {
                "source_id": window.source_id,
                "source_rank": window.source_rank,
                "origin": window.origin,
                "origin_timestamp": str(window.timestamps[int(data["context_length"])]),
            }
            for window in windows
        ]

    selection_path = private_root / "selection_manifest.json"
    _write_json(selection_path, private_selection)
    selection_hash = sha256_file(selection_path)
    print(f"SELECTION_FROZEN sha256={selection_hash}", flush=True)

    units: list[dict[str, Any]] = []
    cell_attempts: Counter[str] = Counter()
    cell_exclusions: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    oracle_ratios: dict[str, list[float]] = defaultdict(list)
    sham_construct_checks = 0
    lower_cap_construct_checks = 0
    threshold = float(mechanism["scenario_preflight"]["minimum_oracle_l1_over_factual_l1"])
    for dataset, windows in selected_by_source.items():
        for window in windows:
            original_background = window.background.copy()
            for family in families:
                cell = f"{dataset}/{family}"
                for seed_rank, seed in enumerate(seeds):
                    cell_attempts[cell] += 1
                    unit: dict[str, Any] = {
                        "dataset": dataset,
                        "source_id": window.source_id,
                        "source_rank": window.source_rank,
                        "origin": window.origin,
                        "family": family,
                        "seed": seed,
                    }
                    try:
                        scenario = generate_scenario(
                            window, family, seed, seed_rank, data, mechanism
                        )
                        diagnostics = validate_scenario(scenario, threshold)
                        replay = generate_scenario(
                            window, family, seed, seed_rank, data, mechanism
                        )
                        if not np.array_equal(scenario.oracle_response, replay.oracle_response):
                            raise ValueError("scenario replay is not deterministic")
                        if not np.array_equal(window.background, original_background):
                            raise ValueError("background was modified by construct")
                        if window.source_rank < int(
                            config["sham_control"]["source_ids_per_dataset"]
                        ):
                            if np.any(scenario.placebo_oracle_response != 0):
                                raise ValueError("placebo world has a nonzero oracle")
                            sham_construct_checks += 1
                        if window.source_rank < int(
                            config["effect_strength_sensitivity"]["source_ids_per_dataset"]
                        ):
                            lower = generate_scenario(
                                window,
                                family,
                                seed,
                                seed_rank,
                                data,
                                mechanism,
                                multiplier_cap=float(
                                    config["effect_strength_sensitivity"]["lower_multiplier_cap"]
                                ),
                            )
                            validate_scenario(lower, 0.0)
                            lower_cap_construct_checks += 1
                        unit.update({"result_status": "verified", **diagnostics})
                        oracle_ratios[cell].append(diagnostics["oracle_ratio"])
                    except ValueError as error:
                        reason = str(error).split(":", 1)[0]
                        cell_exclusions[cell] += 1
                        reason_counts[reason] += 1
                        unit.update({"result_status": "excluded", "reason": str(error)})
                    units.append(unit)

    unit_path = private_root / "construct_units.jsonl"
    unit_path.write_text(
        "".join(json.dumps(unit, sort_keys=True) + "\n" for unit in units), encoding="utf-8"
    )
    cell_reports = {
        cell: {
            "attempted": count,
            "verified": count - cell_exclusions[cell],
            "excluded": cell_exclusions[cell],
            "excluded_fraction": cell_exclusions[cell] / count,
            "oracle_ratio_min": min(oracle_ratios[cell]) if oracle_ratios[cell] else None,
            "oracle_ratio_max": max(oracle_ratios[cell]) if oracle_ratios[cell] else None,
        }
        for cell, count in sorted(cell_attempts.items())
    }
    max_excluded = float(mechanism["scenario_preflight"]["maximum_excluded_fraction_per_cell"])
    passed = all(cell["excluded_fraction"] <= max_excluded for cell in cell_reports.values())
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "result_status": "verified_cpu_construct_preflight" if passed else "revision_required",
        "paper_eligibility": "not_model_outcome_evidence",
        "config_hash": config_hash,
        "p3_scientific_code_sha256": _code_hash(repo),
        "p1_scientific_code_sha256": config["parent_evidence"]["p1_scientific_code_sha256"],
        "selection_manifest_sha256": selection_hash,
        "construct_units_sha256": sha256_file(unit_path),
        "scope": {
            "model_inference_performed": False,
            "forecast_accuracy_or_selection_metric_computed": False,
            "future_target_values_used_for_selection": False,
            "primary_p1_or_p2_modified": False,
        },
        "sources": source_reports,
        "counts": {
            "selected_windows": sum(len(windows) for windows in selected_by_source.values()),
            "attempted_scenarios": len(units),
            "verified_scenarios": len(units) - sum(cell_exclusions.values()),
            "excluded_scenarios": sum(cell_exclusions.values()),
            "sham_construct_checks": sham_construct_checks,
            "lower_cap_construct_checks": lower_cap_construct_checks,
        },
        "cells": cell_reports,
        "exclusion_reasons": dict(sorted(reason_counts.items())),
        "decision": {
            "maximum_excluded_fraction_per_cell": max_excluded,
            "construct_gate_passed": passed,
            "model_inference_authorized": False,
            "next_action": (
                "owner_review_then_separate_p3_freeze"
                if passed
                else "stop_and_submit_revised_candidate_design_for_owner_review"
            ),
        },
        "runtime": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    _write_json(private_root / "construct_preflight_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_cpu_preflight(args.repo, args.data_root, args.private_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["decision"]["construct_gate_passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
