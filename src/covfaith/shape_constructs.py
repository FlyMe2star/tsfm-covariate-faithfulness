"""CPU-only construct validation for P1-SHAPE metrics and generators."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from covfaith.metrics import response_gain_ratio
from covfaith.shape_generators import (
    SHAPE_MECHANISMS,
    ShapeGeneratorSpec,
    generate_shape_batch,
    generate_shape_scenario,
)
from covfaith.shape_metrics import (
    interaction_response,
    shape_metric_suite,
    signed_shape_distance,
)


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture_response() -> np.ndarray:
    response = np.zeros(24, dtype=np.float64)
    response[1:4] = [0.5, 1.0, 0.5]
    response[9:12] = [-0.3, -0.6, -0.3]
    return response


def evaluate_construct_fixtures() -> dict[str, Any]:
    oracle = _fixture_response()
    gain_scaled = 0.70 * oracle
    within_bin_shift = np.roll(oracle, 1)
    cross_bin_shift = np.roll(oracle, 7)
    rebound_deleted = oracle.copy()
    rebound_deleted[rebound_deleted < 0] = 0
    smoothed = np.convolve(oracle, np.array([0.25, 0.50, 0.25]), mode="same")
    zero = np.zeros_like(oracle)

    fixture_predictions = {
        "oracle": oracle,
        "gain_scaled_0p70": gain_scaled,
        "within_width8_bin_shift_1": within_bin_shift,
        "cross_width8_bin_shift_7": cross_bin_shift,
        "temporal_smoothing": smoothed,
        "rebound_deleted": rebound_deleted,
        "sign_flip": -oracle,
        "zero_response": zero,
    }
    metrics: dict[str, Any] = {}
    for name, prediction in fixture_predictions.items():
        suite = shape_metric_suite(prediction, oracle)
        metrics[name] = {
            "rgr": response_gain_ratio(prediction, oracle),
            "signed_shape_distance_by_width": {
                str(width): value
                for width, value in zip(suite.widths, suite.signed_shape_distances, strict=True)
            },
            "hidden_distortion_gap": suite.hidden_distortion_gap,
            "multi_resolution_auc": suite.multi_resolution_auc,
            "temporal_mass_distance": suite.temporal_mass_distance,
            "onset_error": suite.onset_error,
            "peak_time_error": suite.peak_time_error,
            "positive_mass_relative_error": suite.positive_mass_relative_error,
            "negative_mass_relative_error": suite.negative_mass_relative_error,
        }

    shift_ladder = []
    for amount in (0, 1, 2, 3):
        shifted = np.roll(oracle, amount)
        shift_ladder.append(signed_shape_distance(shifted, oracle, 1))

    marginal_a = 0.6 * oracle
    marginal_b = -0.25 * oracle
    true_interaction = 0.4 * np.roll(oracle, 2)
    joint = marginal_a + marginal_b + true_interaction
    recovered = interaction_response(joint, marginal_a, marginal_b)
    omitted = interaction_response(marginal_a + marginal_b, marginal_a, marginal_b)
    interaction_metrics = {
        "exact_recovery_max_abs_error": float(np.max(np.abs(recovered - true_interaction))),
        "omission_shape_distance": signed_shape_distance(omitted, true_interaction, 1),
    }

    checks = {
        "oracle_distance_zero": all(
            abs(value) <= 1e-12
            for value in metrics["oracle"]["signed_shape_distance_by_width"].values()
        ),
        "gain_scaling_changes_rgr_only": (
            abs(metrics["gain_scaled_0p70"]["rgr"] - 0.70) <= 1e-6
            and all(
                abs(value) <= 1e-12
                for value in metrics["gain_scaled_0p70"]["signed_shape_distance_by_width"].values()
            )
        ),
        "within_bin_shift_hidden_at_width8": (
            metrics["within_width8_bin_shift_1"]["signed_shape_distance_by_width"]["1"] > 0
            and abs(
                metrics["within_width8_bin_shift_1"]["signed_shape_distance_by_width"]["8"]
            )
            <= 1e-12
        ),
        "cross_bin_shift_visible_at_width8": (
            metrics["cross_width8_bin_shift_7"]["signed_shape_distance_by_width"]["8"]
            > metrics["within_width8_bin_shift_1"]["signed_shape_distance_by_width"]["8"]
        ),
        "rebound_deletion_detected": (
            metrics["rebound_deleted"]["signed_shape_distance_by_width"]["1"] > 0
            and metrics["rebound_deleted"]["negative_mass_relative_error"] > 0.99
        ),
        "interaction_exact_and_omission_detected": (
            interaction_metrics["exact_recovery_max_abs_error"] <= 1e-12
            and interaction_metrics["omission_shape_distance"] == 1.0
        ),
        "shift_severity_monotone": bool(np.all(np.diff(shift_ladder) >= -1e-12)),
    }
    return {
        "fixtures": metrics,
        "shift_severity_ladder": shift_ladder,
        "interaction_fixture": interaction_metrics,
        "checks": checks,
    }


def evaluate_generator_preflight(
    *,
    seeds: tuple[int, ...] = (101, 307, 911),
    series_per_mechanism_per_seed: int = 64,
) -> dict[str, Any]:
    spec = ShapeGeneratorSpec()
    summaries: dict[str, Any] = {}
    for mechanism in SHAPE_MECHANISMS:
        l1_masses: list[float] = []
        interaction_masses: list[float] = []
        deterministic = True
        paired_histories = True
        for seed in seeds:
            scenarios = generate_shape_batch(mechanism, seed, series_per_mechanism_per_seed, spec)
            for scenario in scenarios:
                duplicate = generate_shape_scenario(
                    mechanism, seed, scenario.series_index, spec
                )
                deterministic = deterministic and bool(
                    np.array_equal(scenario.oracle_response, duplicate.oracle_response)
                )
                paired_histories = paired_histories and bool(
                    scenario.target_context.shape == (spec.context_length,)
                    and scenario.covariate_context.shape == (2, spec.context_length)
                    and np.array_equal(
                        scenario.oracle_response,
                        scenario.target_future_intervened - scenario.target_future_factual,
                    )
                )
                l1_masses.append(float(np.sum(np.abs(scenario.oracle_response))))
                if mechanism == "two_covariate_synergy":
                    interaction = interaction_response(
                        scenario.oracle_response_worlds["joint"],
                        scenario.oracle_response_worlds["a_only"],
                        scenario.oracle_response_worlds["b_only"],
                    )
                    interaction_masses.append(float(np.sum(np.abs(interaction))))
        summaries[mechanism] = {
            "scenario_count": len(l1_masses),
            "deterministic": deterministic,
            "paired_replay_valid": paired_histories,
            "oracle_l1_minimum": float(np.min(l1_masses)),
            "oracle_l1_median": float(np.median(l1_masses)),
            "oracle_l1_maximum": float(np.max(l1_masses)),
            "interaction_l1_minimum": float(np.min(interaction_masses))
            if interaction_masses
            else None,
        }
    return summaries


def run_construct_validation(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    fixtures = evaluate_construct_fixtures()
    generators = evaluate_generator_preflight()
    checks = dict(fixtures["checks"])
    checks.update(
        {
            f"{mechanism}_generator_valid": (
                summary["deterministic"]
                and summary["paired_replay_valid"]
                and summary["oracle_l1_minimum"] > 1e-4
                and (summary["interaction_l1_minimum"] or 1.0) > 1e-4
            )
            for mechanism, summary in generators.items()
        }
    )
    report = {
        "schema_version": 1,
        "experiment": "covintervene-shape-construct-validation-v1",
        "result_status": "verified" if all(checks.values()) else "failed",
        "model_inference_performed": False,
        "p0_outputs_accessed": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "fixtures": fixtures,
        "generator_preflight": generators,
        "all_checks": checks,
        "frozen_bounds": {
            "dsa_lower_bound_minimum": 0.85,
            "rgr_interval": [0.65, 1.35],
            "signed_shape_distance_lower_bound_minimum": 0.08,
            "hidden_distortion_gap_lower_bound_minimum": 0.03,
            "minimum_passing_cells": 2,
            "complementary_relative_sql_maximum": 0.02,
        },
        "threshold_rationale": (
            "One-time pre-inference relaxation requested by the owner. Bounds remain materially "
            "separated from the exact oracle and pure-gain fixtures; model outputs were "
            "unavailable."
        ),
    }
    evidence_path = (
        root / "evidence" / "p1_shape" / "construct" / "metric_construct_validation.yaml"
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(report, sort_keys=False, allow_unicode=True)
    evidence_path.write_text(payload, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    receipt = {
        "relative_path": evidence_path.relative_to(root).as_posix(),
        "sha256": digest,
        "all_checks_passed": all(checks.values()),
    }
    (evidence_path.parent / "metric_construct_validation.receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8", newline="\n"
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"P1-SHAPE construct validation failed: {failed}")
    return report
