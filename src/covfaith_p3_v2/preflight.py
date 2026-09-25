"""Single authorized P3-v2 CPU construct preflight; no forecasting model access."""

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
from covfaith_p3.constructs import validate_scenario
from covfaith_p3.data import read_source_parquet, select_windows, sha256_file
from covfaith_p3.preflight import _code_hash as v1_scientific_code_hash

from .constructs import generate_v2_scenario


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _code_hash(repo: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((repo / "src" / "covfaith_p3_v2").glob("*.py")):
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _select_heldout_windows(
    series: list[Any], v1_data: dict[str, Any], v2_data: dict[str, Any], source: dict[str, Any]
) -> tuple[list[Any], list[Any], int, int]:
    v1_windows, v1_eligible = select_windows(series, v1_data, source)
    v1_ids = {window.source_id for window in v1_windows}
    unused_series = [item for item in series if item.source_id not in v1_ids]
    v2_windows, v2_eligible = select_windows(unused_series, v2_data, source)
    if v1_ids & {window.source_id for window in v2_windows}:
        raise RuntimeError(f"{source['id']}: v1/v2 source ID overlap")
    return v1_windows, v2_windows, v1_eligible, v2_eligible


def _load_approved_design(repo: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    config_dir = repo / "configs" / "p3_semisynthetic"
    v2_path = config_dir / "p3_v2_design_candidate.yaml"
    v1_path = config_dir / "p3_design_candidate.yaml"
    approval_path = config_dir / "p3_v2_design_approval.json"
    v2 = yaml.safe_load(v2_path.read_text(encoding="utf-8"))
    v1 = yaml.safe_load(v1_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    config_hash = canonical_config_hash(v2)
    if approval["approved_candidate_config_canonical_sha256"] != config_hash:
        raise RuntimeError("P3-v2 approval does not match candidate config")
    if approval["approval_phrase"] != "APPROVE P3-V2 DESIGN":
        raise RuntimeError("P3-v2 lacks owner design approval")
    if not approval["cpu_construct_preflight_authorized"]:
        raise RuntimeError("P3-v2 CPU construct preflight is not authorized")
    if approval["model_inference_authorized"] or v2["authorization"]["model_inference_allowed"]:
        raise RuntimeError("P3-v2 model inference must remain unauthorized")
    parent = v2["parent_evidence"]
    if canonical_config_hash(v1) != parent["p3_v1_config_sha256"]:
        raise RuntimeError("P3-v1 config hash changed")
    if p1_scientific_code_hash(repo) != parent["p1_scientific_code_sha256"]:
        raise RuntimeError("frozen P1 scientific code changed")
    if v1_scientific_code_hash(repo) != json.loads(
        (repo / "evidence" / "p3_semisynthetic" / "construct" / "p3_v1_construct_preflight.json")
        .read_text(encoding="utf-8")
    )["p3_scientific_code_sha256"]:
        raise RuntimeError("frozen P3-v1 scientific code changed")
    immutable_files = (
        (
            repo / "evidence" / "p1_shape" / "results" / "p1_shape_decision.json",
            "p1_decision_sha256",
        ),
        (
            repo / "evidence" / "p2_robustness" / "results" / "p2_robustness_report.json",
            "p2_report_sha256",
        ),
        (
            repo / "evidence" / "p3_semisynthetic" / "construct" / "p3_v1_construct_preflight.json",
            "p3_v1_construct_report_sha256",
        ),
        (
            repo / "notes" / "design" / "p3_semisynthetic_baselines.csv",
            "inherited_baselines_sha256",
        ),
    )
    for path, key in immutable_files:
        if sha256_file(path) != parent[key]:
            raise RuntimeError(f"immutable parent artifact changed: {key}")
    if v1["data"]["sources"] != v2["data"]["sources"]:
        raise RuntimeError("P3-v2 source definitions differ from the approved v1 sources")
    if v1["models"] != v2["models"]:
        raise RuntimeError("P3-v2 model pins differ from v1")
    for key in ("minimum_oracle_l1_over_factual_l1", "maximum_excluded_fraction_per_cell"):
        if v1["mechanism"]["scenario_preflight"][key] != v2["mechanism"]["scenario_preflight"][key]:
            raise RuntimeError(f"P3-v2 changed the approved construct gate: {key}")
    return v2, v1, config_hash


def run_cpu_preflight(repo: Path, data_root: Path, private_root: Path) -> dict[str, Any]:
    repo, data_root, private_root = repo.resolve(), data_root.resolve(), private_root.resolve()
    v2, v1, config_hash = _load_approved_design(repo)
    if private_root in {repo, data_root}:
        raise ValueError("private output must be a dedicated directory")
    private_root.mkdir(parents=True, exist_ok=True)
    v1_manifest: dict[str, Any] = {
        "config_hash": v2["parent_evidence"]["p3_v1_config_sha256"],
        "sources": {},
    }
    v2_manifest: dict[str, Any] = {
        "config_hash": config_hash,
        "parent_v1_selection_manifest_sha256": v2["parent_evidence"][
            "p3_v1_selection_manifest_sha256"
        ],
        "sources": {},
    }
    selected_by_source: dict[str, list[Any]] = {}
    source_reports: dict[str, dict[str, Any]] = {}

    for source in v2["data"]["sources"]:
        dataset = str(source["id"])
        local_path = data_root / f"{source['config']}.parquet"
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        actual_hash = sha256_file(local_path)
        if actual_hash != source["parquet_sha256"]:
            raise RuntimeError(f"{dataset}: pinned Parquet SHA-256 mismatch")
        series = read_source_parquet(local_path)
        v1_windows, v2_windows, v1_eligible, v2_eligible = _select_heldout_windows(
            series, v1["data"], v2["data"], source
        )
        v1_ids = {window.source_id for window in v1_windows}
        v1_manifest["sources"][dataset] = [
            {
                "source_id": window.source_id,
                "source_rank": window.source_rank,
                "origin": window.origin,
                "origin_timestamp": str(window.timestamps[int(v1["data"]["context_length"])]),
            }
            for window in v1_windows
        ]
        if v1_ids & {window.source_id for window in v2_windows}:
            raise RuntimeError(f"{dataset}: v1/v2 source ID overlap")
        selected_by_source[dataset] = v2_windows
        v2_manifest["sources"][dataset] = [
            {
                "source_id": window.source_id,
                "source_rank": window.source_rank,
                "origin": window.origin,
                "origin_timestamp": str(window.timestamps[int(v2["data"]["context_length"])]),
            }
            for window in v2_windows
        ]
        source_reports[dataset] = {
            "parquet_sha256": actual_hash,
            "original_series_count": len(series),
            "v1_eligible_source_id_count": v1_eligible,
            "v1_selected_source_id_count": len(v1_windows),
            "v2_eligible_unused_source_id_count": v2_eligible,
            "v2_selected_source_id_count": len(v2_windows),
            "v1_v2_source_id_overlap_count": 0,
        }

    v1_reconstruction_path = private_root / "v1_selection_reconstructed.json"
    _write_json(v1_reconstruction_path, v1_manifest)
    v1_selection_hash = sha256_file(v1_reconstruction_path)
    if v1_selection_hash != v2["parent_evidence"]["p3_v1_selection_manifest_sha256"]:
        raise RuntimeError("reconstructed P3-v1 selection manifest hash mismatch")
    v2_selection_path = private_root / "v2_selection_manifest.json"
    _write_json(v2_selection_path, v2_manifest)
    v2_selection_hash = sha256_file(v2_selection_path)
    print(f"V2_SELECTION_FROZEN sha256={v2_selection_hash}", flush=True)

    data, mechanism = v2["data"], v2["mechanism"]
    seeds = [int(value) for value in data["scenario_seeds"]]
    threshold = float(mechanism["scenario_preflight"]["minimum_oracle_l1_over_factual_l1"])
    attempts: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    oracle_ratios: dict[str, list[float]] = defaultdict(list)
    units: list[dict[str, Any]] = []
    sham_checks = 0
    lower_link_checks = 0
    for dataset, windows in selected_by_source.items():
        for window in windows:
            original_background = window.background.copy()
            for family in mechanism["families"]:
                cell = f"{dataset}/{family}"
                for seed_rank, seed in enumerate(seeds):
                    attempts[cell] += 1
                    unit: dict[str, Any] = {
                        "dataset": dataset,
                        "source_id": window.source_id,
                        "source_rank": window.source_rank,
                        "origin": window.origin,
                        "family": family,
                        "seed": seed,
                    }
                    try:
                        scenario = generate_v2_scenario(
                            window, family, seed, seed_rank, data, mechanism
                        )
                        diagnostics = validate_scenario(scenario, threshold)
                        replay = generate_v2_scenario(
                            window, family, seed, seed_rank, data, mechanism
                        )
                        if not np.array_equal(scenario.oracle_response, replay.oracle_response):
                            raise ValueError("v2 scenario replay is not deterministic")
                        if not np.array_equal(window.background, original_background):
                            raise ValueError("background was modified by v2 construct")
                        if window.source_rank < int(v2["sham_control"]["source_ids_per_dataset"]):
                            if np.any(scenario.placebo_oracle_response != 0):
                                raise ValueError("placebo world has nonzero structural response")
                            sham_checks += 1
                        if window.source_rank < int(
                            v2["effect_strength_sensitivity"]["source_ids_per_dataset"]
                        ):
                            lower = generate_v2_scenario(
                                window,
                                family,
                                seed,
                                seed_rank,
                                data,
                                mechanism,
                                multiplier_cap=float(
                                    v2["effect_strength_sensitivity"]["lower_multiplier_cap"]
                                ),
                            )
                            validate_scenario(lower, 0.0)
                            lower_link_checks += 1
                        unit.update({"result_status": "verified", **diagnostics})
                        oracle_ratios[cell].append(diagnostics["oracle_ratio"])
                    except ValueError as error:
                        reason = str(error).split(":", 1)[0]
                        exclusions[cell] += 1
                        reasons[reason] += 1
                        unit.update({"result_status": "excluded", "reason": str(error)})
                    units.append(unit)

    units_path = private_root / "v2_construct_units.jsonl"
    units_path.write_text(
        "".join(json.dumps(unit, sort_keys=True) + "\n" for unit in units), encoding="utf-8"
    )
    cells = {
        cell: {
            "attempted": count,
            "verified": count - exclusions[cell],
            "excluded": exclusions[cell],
            "excluded_fraction": exclusions[cell] / count,
            "oracle_ratio_min": min(oracle_ratios[cell]) if oracle_ratios[cell] else None,
            "oracle_ratio_max": max(oracle_ratios[cell]) if oracle_ratios[cell] else None,
        }
        for cell, count in sorted(attempts.items())
    }
    maximum = float(mechanism["scenario_preflight"]["maximum_excluded_fraction_per_cell"])
    passed = len(cells) == 6 and all(
        cell["excluded_fraction"] <= maximum for cell in cells.values()
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": v2["experiment"],
        "result_status": (
            "verified_cpu_construct_preflight" if passed else "construct_gate_failed_stop_p3"
        ),
        "paper_eligibility": "not_model_outcome_evidence",
        "config_hash": config_hash,
        "v2_scientific_code_sha256": _code_hash(repo),
        "v1_scientific_code_sha256": v1_scientific_code_hash(repo),
        "p1_scientific_code_sha256": p1_scientific_code_hash(repo),
        "v1_selection_manifest_sha256": v1_selection_hash,
        "v2_selection_manifest_sha256": v2_selection_hash,
        "v2_construct_units_sha256": sha256_file(units_path),
        "scope": {
            "model_inference_performed": False,
            "forecast_accuracy_or_selection_metric_computed": False,
            "future_target_values_used_for_selection": False,
            "v1_source_ids_reused": False,
            "primary_p1_p2_or_v1_modified": False,
        },
        "sources": source_reports,
        "counts": {
            "selected_windows": sum(len(windows) for windows in selected_by_source.values()),
            "attempted_scenarios": len(units),
            "verified_scenarios": len(units) - sum(exclusions.values()),
            "excluded_scenarios": sum(exclusions.values()),
            "sham_construct_checks": sham_checks,
            "lower_link_construct_checks": lower_link_checks,
        },
        "cells": cells,
        "exclusion_reasons": dict(sorted(reasons.items())),
        "decision": {
            "minimum_oracle_l1_over_factual_l1": threshold,
            "maximum_excluded_fraction_per_cell": maximum,
            "construct_gate_passed": passed,
            "model_inference_authorized": False,
            "next_action": (
                "owner_review_then_separate_p3_v2_model_freeze"
                if passed
                else "stop_p3_and_complete_paper_with_p1_p2"
            ),
        },
        "runtime": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    _write_json(private_root / "v2_construct_preflight_report.json", report)
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
