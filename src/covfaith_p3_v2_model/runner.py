"""Resumable P3-v2 inference; this module never computes a paper decision."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import yaml

from covfaith.adapters import ForecastBundle
from covfaith.config import canonical_config_hash
from covfaith.p1_shape import scientific_code_hash as p1_code_hash
from covfaith_p3.constructs import P3Scenario, validate_scenario
from covfaith_p3.data import SourceWindow, read_source_parquet, sha256_file
from covfaith_p3.preflight import _code_hash as v1_code_hash
from covfaith_p3_v2.constructs import generate_v2_scenario
from covfaith_p3_v2.preflight import _code_hash as construct_code_hash
from covfaith_p3_v2.preflight import _select_heldout_windows

Mode = Literal["checkpoint_smoke", "full_units"]


@dataclass(frozen=True)
class ModelView:
    """Minimal duck-typed scenario accepted by the frozen P1 model adapters."""

    target_context: np.ndarray
    target_future_factual: np.ndarray
    covariate_context: np.ndarray
    covariate_future_factual: np.ndarray
    covariate_future_intervened: np.ndarray
    covariate_names: tuple[str, str] = ("active_synthetic", "placebo_synthetic")


class Adapter(Protocol):
    backbone_id: str

    def forecast(
        self, scenarios: list[ModelView], variant: Literal["target_only", "factual", "intervened"]
    ) -> ForecastBundle: ...


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def scientific_code_hash(repo: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((repo / "src" / "covfaith_p3_v2_model").glob("*.py")):
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _matches_frozen_manifest(payload: dict[str, Any], expected_sha256: str) -> bool:
    """Match the original Windows CRLF file or its byte-identical LF serialization."""
    canonical = _json_bytes(payload)
    return expected_sha256 in {
        _sha_bytes(canonical),
        _sha_bytes(canonical.replace(b"\n", b"\r\n")),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_bytes(_json_bytes(payload))
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_frozen_contract(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refuse inference unless every owner-frozen upstream artifact still matches."""
    directory = repo / "configs" / "p3_semisynthetic"
    candidate_path = directory / "p3_v2_model_freeze_candidate.json"
    approval_path = directory / "p3_v2_model_freeze_approval.json"
    config_path = directory / "p3_v2_design_candidate.yaml"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    locked = candidate["locked_evidence"]
    if approval.get("approval_phrase") != "APPROVE P3-V2 MODEL FREEZE":
        raise RuntimeError("P3-v2 model freeze lacks owner approval")
    if not approval.get("sealed_model_inference_authorized"):
        raise RuntimeError("P3-v2 model inference is not authorized")
    if approval.get("p1_p2_or_p3_v1_revision_authorized"):
        raise RuntimeError("model approval may not modify primary or adverse evidence")
    if sha256_file(candidate_path) != approval["approved_candidate_sha256"]:
        raise RuntimeError("P3-v2 model-freeze candidate changed after approval")
    if canonical_config_hash(config) != locked["v2_config_canonical_sha256"]:
        raise RuntimeError("P3-v2 design changed after construct validation")
    report_path = (
        repo / "evidence/p3_semisynthetic/v2_construct/p3_v2_construct_preflight.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if sha256_file(report_path) != locked["v2_construct_report_sha256"]:
        raise RuntimeError("P3-v2 public construct report changed")
    if report["decision"]["construct_gate_passed"] is not True:
        raise RuntimeError("P3-v2 construct gate did not pass")
    if report["config_hash"] != locked["v2_config_canonical_sha256"]:
        raise RuntimeError("P3-v2 construct report is for another design")
    if construct_code_hash(repo) != locked["v2_construct_code_sha256"]:
        raise RuntimeError("P3-v2 construct code changed after CPU preflight")
    if p1_code_hash(repo) != report["p1_scientific_code_sha256"]:
        raise RuntimeError("frozen P1 code changed")
    if v1_code_hash(repo) != report["v1_scientific_code_sha256"]:
        raise RuntimeError("failed P3-v1 code changed")
    v1_report_path = repo / "evidence/p3_semisynthetic/construct/p3_v1_construct_preflight.json"
    if sha256_file(v1_report_path) != locked["v1_failed_construct_report_sha256"]:
        raise RuntimeError("frozen P3-v1 failure report changed")
    if report["v1_selection_manifest_sha256"] != locked["v1_selection_manifest_sha256"]:
        raise RuntimeError("P3-v1 selection hash mismatch")
    if report["v2_selection_manifest_sha256"] != locked["v2_selection_manifest_sha256"]:
        raise RuntimeError("P3-v2 selection hash mismatch")
    if report["v2_construct_units_sha256"] != locked["v2_construct_units_sha256"]:
        raise RuntimeError("P3-v2 construct unit hash mismatch")
    scope = candidate["locked_scientific_scope"]
    if [model["id"] for model in config["models"]] != scope["backbones"]:
        raise RuntimeError("model roster changed after freeze")
    return config, report


def load_frozen_windows(
    repo: Path, data_root: Path, config: dict[str, Any], report: dict[str, Any]
) -> dict[str, list[SourceWindow]]:
    """Reconstruct exact source-ID/origin selection from the pinned public data."""
    v1 = yaml.safe_load(
        (repo / "configs/p3_semisynthetic/p3_design_candidate.yaml").read_text(encoding="utf-8")
    )
    v1_manifest: dict[str, Any] = {
        "config_hash": config["parent_evidence"]["p3_v1_config_sha256"],
        "sources": {},
    }
    v2_manifest: dict[str, Any] = {
        "config_hash": report["config_hash"],
        "parent_v1_selection_manifest_sha256": report["v1_selection_manifest_sha256"],
        "sources": {},
    }
    selected: dict[str, list[SourceWindow]] = {}
    for source in config["data"]["sources"]:
        dataset = source["id"]
        parquet = data_root / f"{source['config']}.parquet"
        if sha256_file(parquet) != source["parquet_sha256"]:
            raise RuntimeError(f"{dataset}: pinned source Parquet hash mismatch")
        v1_windows, v2_windows, _, _ = _select_heldout_windows(
            read_source_parquet(parquet), v1["data"], config["data"], source
        )
        for manifest, windows, context in (
            (v1_manifest, v1_windows, int(v1["data"]["context_length"])),
            (v2_manifest, v2_windows, int(config["data"]["context_length"])),
        ):
            manifest["sources"][dataset] = [
                {
                    "source_id": window.source_id,
                    "source_rank": window.source_rank,
                    "origin": window.origin,
                    "origin_timestamp": str(window.timestamps[context]),
                }
                for window in windows
            ]
        selected[dataset] = v2_windows
    if not _matches_frozen_manifest(v1_manifest, report["v1_selection_manifest_sha256"]):
        raise RuntimeError("reconstructed P3-v1 source selection differs from frozen receipt")
    if not _matches_frozen_manifest(v2_manifest, report["v2_selection_manifest_sha256"]):
        raise RuntimeError("reconstructed P3-v2 source selection differs from frozen receipt")
    return selected


def prepare_frozen_units(
    selected: dict[str, list[SourceWindow]], config: dict[str, Any], report: dict[str, Any]
) -> dict[tuple[str, str, int], list[P3Scenario]]:
    """Replay all 576 constructs and require identical six-cell exclusion counts."""
    data, mechanism = config["data"], config["mechanism"]
    threshold = float(mechanism["scenario_preflight"]["minimum_oracle_l1_over_factual_l1"])
    units: dict[tuple[str, str, int], list[P3Scenario]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"attempted": 0, "excluded": 0})
    for dataset, windows in selected.items():
        for window in windows:
            for family in mechanism["families"]:
                cell = f"{dataset}/{family}"
                for seed_rank, seed_value in enumerate(data["scenario_seeds"]):
                    seed = int(seed_value)
                    counts[cell]["attempted"] += 1
                    scenario = generate_v2_scenario(
                        window, family, seed, seed_rank, data, mechanism
                    )
                    try:
                        validate_scenario(scenario, threshold)
                    except ValueError as error:
                        if not str(error).startswith("insufficient oracle mass:"):
                            raise RuntimeError(f"unexpected construct failure in {cell}") from error
                        counts[cell]["excluded"] += 1
                        continue
                    units[(dataset, family, seed)].append(scenario)
    if set(counts) != set(report["cells"]):
        raise RuntimeError("frozen construct cell roster differs from receipt")
    for cell, count in counts.items():
        expected = report["cells"][cell]
        if count != {key: expected[key] for key in ("attempted", "excluded")}:
            raise RuntimeError(f"{cell}: construct exclusions differ from frozen receipt")
    return dict(units)


def _view(scenario: P3Scenario, *, placebo: bool = False) -> ModelView:
    return ModelView(
        target_context=scenario.target_context_factual,
        target_future_factual=scenario.target_future_factual,
        covariate_context=scenario.covariate_context,
        covariate_future_factual=scenario.covariate_future_factual,
        covariate_future_intervened=(
            scenario.placebo_future_intervened
            if placebo
            else scenario.covariate_future_intervened
        ),
    )


def _bundle(bundle: ForecastBundle, count: int, horizon: int) -> None:
    if bundle.median.shape != (count, horizon):
        raise RuntimeError(f"unexpected P3-v2 median shape {bundle.median.shape}")
    if bundle.quantiles is None or bundle.quantiles.shape != (
        count, horizon, len(bundle.quantile_levels)
    ):
        raise RuntimeError("P3-v2 requires aligned probabilistic forecasts")
    if not np.all(np.isfinite(bundle.median)) or not np.all(np.isfinite(bundle.quantiles)):
        raise RuntimeError("P3-v2 model returned nonfinite forecasts")
    levels = np.asarray(bundle.quantile_levels, dtype=np.float64)
    if not np.all((levels > 0) & (levels < 1)) or np.any(np.diff(levels) <= 0):
        raise RuntimeError("P3-v2 model returned invalid quantile levels")


def infer_scenarios(
    adapter: Adapter,
    scenarios: list[P3Scenario],
    windows_by_id: dict[str, SourceWindow],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Archive predictions and oracle arrays only; compute no aggregate metric."""
    if not scenarios:
        raise ValueError("no verified P3-v2 scenarios in unit")
    horizon = int(config["data"]["horizon"])
    views = [_view(scenario) for scenario in scenarios]
    target = adapter.forecast(views, "target_only")
    factual = adapter.forecast(views, "factual")
    active = adapter.forecast(views, "intervened")
    for bundle in (target, factual, active):
        _bundle(bundle, len(scenarios), horizon)
    if not (target.quantile_levels == factual.quantile_levels == active.quantile_levels):
        raise RuntimeError("P3-v2 forecast variants returned different quantile levels")

    sham_indices = np.asarray(
        [index for index, item in enumerate(scenarios) if item.source_rank < 12], dtype=np.int64
    )
    lower_indices = sham_indices.copy()
    sham_median = np.empty((0, horizon), dtype=np.float32)
    lower_factual_median = np.empty((0, horizon), dtype=np.float32)
    lower_active_median = np.empty((0, horizon), dtype=np.float32)
    lower_oracle = np.empty((0, horizon), dtype=np.float64)
    if len(sham_indices):
        sham_views = [_view(scenarios[index], placebo=True) for index in sham_indices]
        sham = adapter.forecast(sham_views, "intervened")
        _bundle(sham, len(sham_indices), horizon)
        if sham.quantile_levels != factual.quantile_levels:
            raise RuntimeError("sham forecast quantile levels differ")
        sham_median = sham.median.copy()

        data, mechanism = config["data"], config["mechanism"]
        seeds = [int(value) for value in data["scenario_seeds"]]
        lower_scenarios: list[P3Scenario] = []
        for index in lower_indices:
            original = scenarios[index]
            window = windows_by_id[original.source_id]
            lower_scenario = generate_v2_scenario(
                window,
                original.family,
                original.seed,
                seeds.index(original.seed),
                data,
                mechanism,
                multiplier_cap=float(config["effect_strength_sensitivity"]["lower_multiplier_cap"]),
            )
            validate_scenario(lower_scenario, 0.0)
            lower_scenarios.append(lower_scenario)
        lower_views = [_view(scenario) for scenario in lower_scenarios]
        lower_factual = adapter.forecast(lower_views, "factual")
        lower_active = adapter.forecast(lower_views, "intervened")
        for bundle in (lower_factual, lower_active):
            _bundle(bundle, len(lower_scenarios), horizon)
            if bundle.quantile_levels != factual.quantile_levels:
                raise RuntimeError("lower-link forecast quantile levels differ")
        lower_factual_median = lower_factual.median.copy()
        lower_active_median = lower_active.median.copy()
        lower_oracle = np.stack([item.oracle_response for item in lower_scenarios])

    assert target.quantiles is not None and factual.quantiles is not None
    return {
        "source_id": np.asarray([item.source_id for item in scenarios], dtype="U"),
        "source_rank": np.asarray([item.source_rank for item in scenarios], dtype=np.int32),
        "origin": np.asarray([item.origin for item in scenarios], dtype=np.int64),
        "target_context": np.stack([item.target_context_factual for item in scenarios]),
        "target_future_factual": np.stack([item.target_future_factual for item in scenarios]),
        "oracle_response": np.stack([item.oracle_response for item in scenarios]),
        "median_target_only": target.median,
        "median_factual": factual.median,
        "median_active": active.median,
        "quantiles_target_only": target.quantiles,
        "quantiles_factual": factual.quantiles,
        "quantile_levels": np.asarray(factual.quantile_levels, dtype=np.float64),
        "sham_indices": sham_indices,
        "median_sham": sham_median,
        "lower_indices": lower_indices,
        "median_lower_factual": lower_factual_median,
        "median_lower_active": lower_active_median,
        "lower_oracle_response": lower_oracle,
    }


def _unit_paths(
    root: Path, mode: Mode, backbone: str, key: tuple[str, str, int]
) -> tuple[Path, Path]:
    dataset, family, seed = key
    directory = root / mode / backbone / dataset / family
    stem = f"seed_{seed:05d}"
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def _finished_unit(
    array_path: Path,
    manifest_path: Path,
    *,
    config_hash: str,
    code_hash: str,
    selection_hash: str,
    valid_count: int,
    backbone: str,
    checkpoint: str,
    revision: str,
    mode: Mode,
) -> bool:
    if not array_path.exists() and not manifest_path.exists():
        return False
    if not array_path.exists() or not manifest_path.exists():
        return False  # interrupted atomic write; safely recreate this one unit
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "config_hash": config_hash,
        "model_runner_code_sha256": code_hash,
        "v2_selection_manifest_sha256": selection_hash,
        "valid_scenario_count": valid_count,
        "completed": True,
        "array_sha256": sha256_file(array_path),
        "backbone": backbone,
        "checkpoint": checkpoint,
        "checkpoint_revision": revision,
        "result_status": (
            "checkpoint_smoke_only" if mode == "checkpoint_smoke" else "sealed_unit_only"
        ),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"completed P3-v2 unit is hash-mismatched: {manifest_path}")
    return True


def _runtime() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for package in ("chronos-forecasting", "timesfm", "huggingface-hub", "pyarrow"):
        try:
            payload[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            payload[package] = None
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except ImportError:
        payload.update({"torch": None, "cuda": None, "gpu": None})
    return payload


def run_backbone_units(
    repo: str | Path,
    adapter: Adapter,
    data_root: str | Path,
    output_root: str | Path,
    *,
    mode: Mode,
    model_load_seconds: float | None = None,
) -> dict[str, Any]:
    """Run one pinned backbone, with separate smoke/full archives and strict resume."""
    from covfaith.adapters import Chronos2Adapter, TimesFM3Adapter

    if mode not in ("checkpoint_smoke", "full_units"):
        raise ValueError("mode must be checkpoint_smoke or full_units")
    if not isinstance(adapter, (Chronos2Adapter, TimesFM3Adapter)):
        raise TypeError("formal P3-v2 inference requires a pinned checkpoint adapter")
    root = Path(repo).resolve()
    data_path = Path(data_root).resolve()
    destination = Path(output_root).resolve()
    if destination == root or root in destination.parents:
        raise ValueError("private P3-v2 model outputs must stay outside public Git")
    config, construct_report = load_frozen_contract(root)
    model = next((item for item in config["models"] if item["id"] == adapter.backbone_id), None)
    if model is None:
        raise RuntimeError("adapter backbone is not in the owner-frozen P3-v2 roster")
    selected = load_frozen_windows(root, data_path, config, construct_report)
    prepared = prepare_frozen_units(selected, config, construct_report)
    expected_keys = {
        (source["id"], family, int(seed))
        for source in config["data"]["sources"]
        for family in config["mechanism"]["families"]
        for seed in config["data"]["scenario_seeds"]
    }
    if set(prepared) != expected_keys:
        raise RuntimeError("P3-v2 scenario unit roster is incomplete")
    code_hash = scientific_code_hash(root)
    config_hash = construct_report["config_hash"]
    selection_hash = construct_report["v2_selection_manifest_sha256"]
    if mode == "full_units":
        for frozen_model in config["models"]:
            smoke = destination / "checkpoint_smoke" / frozen_model["id"] / "completion.json"
            if not smoke.is_file():
                raise RuntimeError(f"run a checkpoint smoke for {frozen_model['id']} first")
            receipt = json.loads(smoke.read_text(encoding="utf-8"))
            if (
                receipt.get("result_status") != "checkpoint_smoke_only"
                or receipt.get("model_runner_code_sha256") != code_hash
                or receipt.get("config_hash") != config_hash
                or receipt.get("v2_selection_manifest_sha256") != selection_hash
                or receipt.get("checkpoint") != frozen_model["checkpoint"]
                or receipt.get("checkpoint_revision") != frozen_model["revision"]
                or receipt.get("completed_unit_count") != 1
                or receipt.get("valid_scenario_count") != 1
            ):
                raise RuntimeError("P3-v2 checkpoint-smoke receipt differs from frozen run")

    keys = sorted(prepared)
    if mode == "checkpoint_smoke":
        keys = keys[:1]
    completed: list[str] = []
    scenario_count = 0
    for key in keys:
        dataset, family, seed = key
        scenarios = prepared[key]
        if mode == "checkpoint_smoke":
            scenarios = scenarios[:1]
        windows_by_id = {window.source_id: window for window in selected[dataset]}
        array_path, manifest_path = _unit_paths(destination, mode, adapter.backbone_id, key)
        label = f"{dataset}/{family}/seed_{seed:05d}"
        if _finished_unit(
            array_path,
            manifest_path,
            config_hash=config_hash,
            code_hash=code_hash,
            selection_hash=selection_hash,
            valid_count=len(scenarios),
            backbone=adapter.backbone_id,
            checkpoint=model["checkpoint"],
            revision=model["revision"],
            mode=mode,
        ):
            print(f"RESUME {adapter.backbone_id}/{label}", flush=True)
        else:
            start = time.perf_counter()
            arrays = infer_scenarios(adapter, scenarios, windows_by_id, config)
            _atomic_npz(array_path, arrays)
            _atomic_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "result_status": (
                        "checkpoint_smoke_only"
                        if mode == "checkpoint_smoke"
                        else "sealed_unit_only"
                    ),
                    "completed": True,
                    "config_hash": config_hash,
                    "model_runner_code_sha256": code_hash,
                    "v2_selection_manifest_sha256": selection_hash,
                    "backbone": adapter.backbone_id,
                    "checkpoint": model["checkpoint"],
                    "checkpoint_revision": model["revision"],
                    "dataset": dataset,
                    "family": family,
                    "seed": seed,
                    "valid_scenario_count": len(scenarios),
                    "excluded_scenario_count": int(config["data"]["source_ids_per_dataset"])
                    - len(prepared[key]),
                    "sham_scenario_count": int(len(arrays["sham_indices"])),
                    "lower_link_scenario_count": int(len(arrays["lower_indices"])),
                    "array_sha256": sha256_file(array_path),
                    "inference_wall_time_seconds": time.perf_counter() - start,
                    "git_commit": _git_commit(root),
                    "runtime": _runtime(),
                },
            )
            print(f"SAVED {adapter.backbone_id}/{label}: {len(scenarios)} scenarios", flush=True)
        completed.append(label)
        scenario_count += len(scenarios)
    report = {
        "schema_version": 1,
        "result_status": (
            "checkpoint_smoke_only"
            if mode == "checkpoint_smoke"
            else "sealed_units_complete_not_analyzed"
        ),
        "mode": mode,
        "config_hash": config_hash,
        "model_runner_code_sha256": code_hash,
        "construct_report_sha256": sha256_file(
            root / "evidence/p3_semisynthetic/v2_construct/p3_v2_construct_preflight.json"
        ),
        "v2_selection_manifest_sha256": selection_hash,
        "backbone": adapter.backbone_id,
        "checkpoint": model["checkpoint"],
        "checkpoint_revision": model["revision"],
        "model_load_seconds": model_load_seconds,
        "completed_unit_count": len(completed),
        "completed_units": completed,
        "valid_scenario_count": scenario_count,
        "all_construct_cell_counts_verified": True,
        "scientific_gate_computed": False,
        "git_commit": _git_commit(root),
        "runtime": _runtime(),
    }
    completion_path = destination / mode / adapter.backbone_id / "completion.json"
    _atomic_json(completion_path, report)
    return report
