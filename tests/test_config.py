import json
from pathlib import Path

from covfaith.config import canonical_config_hash, verify_config_lock

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p0" / "covintervene_p0.yaml"
LOCK = ROOT / "configs" / "p0" / "covintervene_p0.lock.json"


def test_frozen_config_matches_external_lock() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert canonical_config_hash(CONFIG) == lock["canonical_sha256"]
    assert verify_config_lock(CONFIG, LOCK) == lock["canonical_sha256"]
