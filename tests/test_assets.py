from pathlib import Path

from pmtt_physics.config import load_config
from pmtt_physics.validation import validate_assets


ROOT = Path(__file__).resolve().parents[1]


def test_paper_config() -> None:
    config = load_config(ROOT / "config" / "paper.toml")
    assert config.seed == 20261103
    assert config.holdout_seed == 87261103
    assert config.output_size == 224
    assert config.future_offsets == (-6, -3, 3, 6)
    assert config.ranks == (1, 2, 4, 8)
    assert config.bootstrap_draws == 2000


def test_released_asset_identities() -> None:
    counts = validate_assets(ROOT)
    assert counts == {
        "synthetic_conditions": 448,
        "synthetic_identities": 32,
        "tapvid_anchors": 23686,
        "tapvid_interfaces": 24565,
        "tapvid_transitions": 44386,
        "reference_tables": 12,
    }
