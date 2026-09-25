"""Data-only, past-target-only selection of P3 background windows."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SourceSeries:
    source_id: str
    target: NDArray[np.float64]
    timestamps: NDArray[np.datetime64]


@dataclass(frozen=True)
class SourceWindow:
    dataset: str
    source_id: str
    source_rank: int
    origin: int
    background: NDArray[np.float64]
    timestamps: NDArray[np.datetime64]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source_parquet(path: Path) -> list[SourceSeries]:
    """Read only source IDs, timestamps, and targets; pyarrow is an optional P3 dependency."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("P3 CPU preflight requires pyarrow") from error

    rows = pq.read_table(path, columns=["id", "timestamp", "target"]).to_pylist()
    series: list[SourceSeries] = []
    seen: set[str] = set()
    for row in rows:
        source_id = str(row["id"])
        if source_id in seen:
            raise ValueError(f"duplicate original source ID in {path.name}: {source_id!r}")
        seen.add(source_id)
        target = np.asarray(row["target"], dtype=np.float64)
        timestamps = np.asarray(row["timestamp"], dtype="datetime64[ms]")
        if target.ndim != 1 or timestamps.ndim != 1 or target.size != timestamps.size:
            raise ValueError(f"malformed target/timestamp sequence for {source_id!r}")
        series.append(SourceSeries(source_id, target, timestamps))
    return series


def _hour(timestamp: np.datetime64) -> int:
    return int(timestamp.astype("datetime64[h]").astype(np.int64) % 24)


def eligible_origins(
    series: SourceSeries, data: dict[str, Any], source: dict[str, Any]
) -> list[int]:
    """Do not read target[origin:] when deciding eligibility.

    The complete timestamp window is allowed because calendar values are known
    at forecast time. Future target values are inspected only after selection.
    """
    context_length = int(data["context_length"])
    horizon = int(data["horizon"])
    stride = int(data["origin_stride"])
    if context_length <= 1 or horizon <= 0 or stride <= 0:
        raise ValueError("context, horizon, and stride must be positive")
    n = series.target.size
    first = max(context_length, math.floor(float(data["origin_region_start_fraction"]) * n))
    expected_gap_ms = int(source["frequency_seconds"]) * 1000
    origins: list[int] = []
    for origin in range(first, n - horizon + 1, stride):
        context = series.target[origin - context_length : origin]
        if not np.all(np.isfinite(context)):
            continue
        if np.std(context, ddof=1) <= float(data["context_standard_deviation_min"]):
            continue
        if np.mean(context > 0) < float(source["context_positive_fraction_min"]):
            continue
        if source.get("context_last_24_all_positive") and not np.all(context[-24:] > 0):
            continue
        times = series.timestamps[origin - context_length : origin + horizon]
        if np.any(np.isnat(times)):
            continue
        if bool(data["require_nominal_cadence"]):
            gaps_ms = np.diff(times).astype("timedelta64[ms]").astype(np.int64)
            if not np.all(gaps_ms == expected_gap_ms):
                continue
        allowed_hours = source.get("origin_clock_hours_inclusive")
        if allowed_hours is not None and _hour(series.timestamps[origin]) not in allowed_hours:
            continue
        origins.append(origin)
    return origins


def _rank(salt: str, dataset: str, source_id: str, origin: int | None = None) -> str:
    parts = [salt, dataset, source_id]
    if origin is not None:
        parts.append(str(origin))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def select_windows(
    series: list[SourceSeries], data: dict[str, Any], source: dict[str, Any]
) -> tuple[list[SourceWindow], int]:
    """Select fixed original IDs and one origin per ID without future target inspection."""
    dataset = str(source["id"])
    salt = str(data["selection_salt"])
    eligible = [(item, eligible_origins(item, data, source)) for item in series]
    eligible = [(item, origins) for item, origins in eligible if origins]
    eligible.sort(key=lambda pair: (_rank(salt, dataset, pair[0].source_id), pair[0].source_id))
    requested = int(data["source_ids_per_dataset"])
    if len(eligible) < requested:
        raise RuntimeError(f"{dataset}: {len(eligible)} eligible IDs, need {requested}")
    context_length = int(data["context_length"])
    horizon = int(data["horizon"])
    windows: list[SourceWindow] = []
    for rank, (item, origins) in enumerate(eligible[:requested]):
        origin = min(origins, key=lambda idx: (_rank(salt, dataset, item.source_id, idx), idx))
        span = slice(origin - context_length, origin + horizon)
        windows.append(
            SourceWindow(
                dataset=dataset,
                source_id=item.source_id,
                source_rank=rank,
                origin=origin,
                background=item.target[span].copy(),
                timestamps=item.timestamps[span].copy(),
            )
        )
    return windows, len(eligible)
