"""Configuration loading and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject non-mapping roots."""

    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a YAML mapping at {path}")
    return parsed


def canonical_config_bytes(config: dict[str, Any]) -> bytes:
    """Return stable UTF-8 JSON bytes for a parsed configuration mapping."""

    return json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_config_hash(config_or_path: dict[str, Any] | str | Path) -> str:
    """SHA-256 of the canonical parsed configuration."""

    config = (
        load_yaml(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else config_or_path
    )
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def verify_config_lock(config_path: str | Path, lock_path: str | Path) -> str:
    """Verify a config against its external lock and return the canonical hash."""

    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    expected = lock.get("canonical_sha256")
    actual = canonical_config_hash(config_path)
    if actual != expected:
        raise RuntimeError(f"config hash mismatch: expected {expected}, got {actual}")
    if lock.get("sealed_model_inference_authorized") is not True:
        raise RuntimeError("config lock does not authorize sealed model inference")
    return actual
