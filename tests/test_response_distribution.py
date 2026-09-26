"""Tests for the post-primary, read-only P1 response diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from covfaith.metrics import response_gain_ratio
from covfaith.shape_metrics import onset_error, peak_time_error, temporal_mass_distance
from covfaith_diagnostics import response_distribution as diagnostic
from covfaith_diagnostics.response_distribution import (
    _cell_summary,
    _reconcile_frozen_cell,
    _verified_unit,
    analyze_response_distribution,
)


def _fixture_arrays() -> dict[str, np.ndarray]:
    oracle = np.zeros((2, 24), dtype=np.float64)
    oracle[:, 5] = 2.0
    predicted = np.zeros_like(oracle)
    predicted[0, 5] = 1.0
    predicted[1, 6] = 2.0
    return {
        "series_ids": np.asarray(["sample-a", "sample-b"]),
        "oracle_response": oracle,
        "predicted_response": predicted,
        "dsa": np.asarray([1.0, 0.0]),
        "rgr": np.asarray(
            [response_gain_ratio(p, o) for p, o in zip(predicted, oracle, strict=True)]
        ),
        "shape_distance_by_width": np.asarray([[0.1, 0.1], [0.4, 0.2]]),
        "hidden_distortion_gap": np.asarray([0.0, 0.2]),
        "onset_error": np.asarray(
            [onset_error(p, o) for p, o in zip(predicted, oracle, strict=True)]
        ),
        "peak_time_error": np.asarray(
            [peak_time_error(p, o) for p, o in zip(predicted, oracle, strict=True)]
        ),
        "temporal_mass_distance": np.asarray(
            [temporal_mass_distance(p, o) for p, o in zip(predicted, oracle, strict=True)]
        ),
    }


def _write_unit(root: Path, arrays: dict[str, np.ndarray]) -> str:
    directory = root / "units/chronos_2/biphasic_rebound"
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / "seed_00101.npz"
    np.savez_compressed(array_path, **arrays)
    sha256 = hashlib.sha256(array_path.read_bytes()).hexdigest()
    manifest = {
        "completed": True,
        "mode": "full_screening",
        "backbone": "chronos_2",
        "mechanism": "biphasic_rebound",
        "generator_seed": 101,
        "series_count": 2,
        "config_hash": "frozen-config",
        "scientific_code_sha256": "frozen-code",
        "array_sha256": sha256,
    }
    (directory / "seed_00101.json").write_text(json.dumps(manifest), encoding="utf-8")
    return sha256


def _load_unit(root: Path, sha256: str) -> dict[str, np.ndarray]:
    return _verified_unit(
        root,
        "chronos_2/biphasic_rebound/seed_00101",
        config_hash="frozen-config",
        code_hash="frozen-code",
        expected_hash=sha256,
        count=2,
        horizon=24,
        support_fraction=0.1,
    )


def test_full_cell_raw_gain_and_timing_are_descriptive() -> None:
    arrays = _fixture_arrays()
    summary = _cell_summary(arrays, horizon=24, support_fraction=0.1)
    assert summary["n"] == 2
    assert summary["full_l1_ratio"]["median"] == pytest.approx(0.75)
    assert summary["onset_error_steps"]["median"] == pytest.approx(0.5)
    assert summary["peak_time_error_steps"]["q75"] == pytest.approx(0.75)
    assert summary["onset_within_one_step_count"] == 2
    assert summary["peak_within_one_step_count"] == 2
    assert summary["no_predicted_onset_count"] == 0
    assert summary["no_predicted_peak_count"] == 0


def test_verified_unit_replays_registered_metrics_and_detects_tampering(tmp_path: Path) -> None:
    arrays = _fixture_arrays()
    sha256 = _write_unit(tmp_path, arrays)
    assert _load_unit(tmp_path, sha256)["oracle_response"].shape == (2, 24)
    with pytest.raises(RuntimeError, match="sha256"):
        _load_unit(tmp_path, "0" * 64)

    arrays["rgr"][0] += 0.1
    modified_sha256 = _write_unit(tmp_path, arrays)
    with pytest.raises(RuntimeError, match="raw-response replay"):
        _load_unit(tmp_path, modified_sha256)


def test_frozen_cell_reconciliation_rejects_changed_primary_median() -> None:
    arrays = _fixture_arrays()
    decision = {
        "cells": {
            "chronos_2/biphasic_rebound": {
                "series_count": 2,
                "dsa": {"estimate": float(np.mean(arrays["dsa"]))},
                "rgr": {"estimate": float(np.median(arrays["rgr"]))},
                "shape_d1": {"estimate": 0.25},
                "hidden_gap": {"estimate": 0.1},
            }
        }
    }
    _reconcile_frozen_cell("chronos_2/biphasic_rebound", arrays, decision)
    decision["cells"]["chronos_2/biphasic_rebound"]["rgr"]["estimate"] += 0.1
    with pytest.raises(RuntimeError, match="frozen decision"):
        _reconcile_frozen_cell("chronos_2/biphasic_rebound", arrays, decision)


def test_output_cannot_be_inside_source_archive(tmp_path: Path) -> None:
    source = tmp_path / "private-p1"
    with pytest.raises(ValueError, match="separate"):
        analyze_response_distribution(tmp_path, source, source / "diagnostic")


def test_complete_diagnostic_checks_all_units_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source, output = tmp_path / "repo", tmp_path / "private-p1", tmp_path / "summary"
    decision_dir = repo / "evidence/p1_shape/results"
    p2_dir = repo / "evidence/p2_robustness/results"
    decision_dir.mkdir(parents=True)
    p2_dir.mkdir(parents=True)
    backbones = ("chronos_2", "timesfm_3")
    mechanisms = ("biphasic_rebound", "dispersed_delayed_pulse", "hysteresis", "synergy")
    seeds = (101, 307, 911)
    config = {
        "models": [{"id": name} for name in backbones],
        "data": {
            "mechanisms": list(mechanisms),
            "generator_seeds": list(seeds),
            "series_per_mechanism_per_seed": 2,
            "horizon": 24,
        },
        "response_metrics": {"active_support_fraction_of_peak": 0.1},
    }
    monkeypatch.setattr(diagnostic, "verify_config_lock", lambda *_: "frozen-config")
    monkeypatch.setattr(diagnostic, "load_yaml", lambda *_: config)
    monkeypatch.setattr(diagnostic, "scientific_code_hash", lambda *_: "frozen-code")
    source_hashes: dict[str, str] = {}
    decision_cells: dict[str, dict] = {}
    selected: dict[str, dict] = {}
    for backbone in backbones:
        for mechanism in mechanisms:
            cell = f"{backbone}/{mechanism}"
            arrays = _fixture_arrays()
            decision_cells[cell] = {
                "series_count": 6,
                "dsa": {"estimate": float(np.mean(arrays["dsa"]))},
                "rgr": {"estimate": float(np.median(arrays["rgr"]))},
                "shape_d1": {"estimate": 0.25},
                "hidden_gap": {"estimate": 0.1},
            }
            for seed in seeds:
                unit = {name: value.copy() for name, value in arrays.items()}
                unit["series_ids"] = np.asarray([f"{cell}/s{seed}/a", f"{cell}/s{seed}/b"])
                directory = source / "units" / backbone / mechanism
                directory.mkdir(parents=True, exist_ok=True)
                stem = f"seed_{seed:05d}"
                array_path = directory / f"{stem}.npz"
                np.savez_compressed(array_path, **unit)
                digest = hashlib.sha256(array_path.read_bytes()).hexdigest()
                key = f"{cell}/{stem}"
                source_hashes[key] = digest
                manifest = {
                    "completed": True,
                    "mode": "full_screening",
                    "backbone": backbone,
                    "mechanism": mechanism,
                    "generator_seed": seed,
                    "series_count": 2,
                    "config_hash": "frozen-config",
                    "scientific_code_sha256": "frozen-code",
                    "array_sha256": digest,
                }
                (directory / f"{stem}.json").write_text(json.dumps(manifest), encoding="utf-8")
                if seed == 101:
                    selected[cell] = {
                        "cell": cell,
                        "series_id": str(unit["series_ids"][0]),
                        "oracle_response": unit["oracle_response"][0].tolist(),
                        "predicted_response": unit["predicted_response"][0].tolist(),
                    }
    passing = (
        "chronos_2/biphasic_rebound",
        "chronos_2/dispersed_delayed_pulse",
        "timesfm_3/biphasic_rebound",
    )
    decision = {
        "result_status": "verified_p1_shape_decision",
        "config_hash": "frozen-config",
        "scientific_code_sha256": "frozen-code",
        "cells": decision_cells,
        "decision": {"passing_cells": list(passing)},
    }
    decision_path = decision_dir / "p1_shape_decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    representatives_path = p2_dir / "representative_responses.json"
    representatives_path.write_text(
        json.dumps([selected[cell] for cell in passing]), encoding="utf-8"
    )
    p2 = {
        "source_primary": {
            "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
            "config_hash": "frozen-config",
            "scientific_code_sha256": "frozen-code",
            "unit_sha256": source_hashes,
        },
        "artifacts": {
            "representative_responses.json": {
                "sha256": hashlib.sha256(representatives_path.read_bytes()).hexdigest()
            }
        },
    }
    (p2_dir / "p2_robustness_report.json").write_text(json.dumps(p2), encoding="utf-8")

    report = analyze_response_distribution(repo, source, output)
    assert report["scope"]["full_unit_count"] == 24
    assert report["scope"]["series_count"] == 48
    assert len(report["cells"]) == 8
    assert len(report["representative_checks"]) == 3
    assert report["cells"][passing[0]]["full_l1_ratio"]["median"] == pytest.approx(0.75)
    assert analyze_response_distribution(repo, source, output) == report
    (output / "complete_eight_cell_response_distribution.csv").write_text("tampered")
    with pytest.raises(RuntimeError, match="existing diagnostic report differs"):
        analyze_response_distribution(repo, source, output)
